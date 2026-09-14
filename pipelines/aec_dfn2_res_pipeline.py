"""
AEC + RES + DeepFilterNet2 pipeline (mono Python reference)

Pipeline: AEC(linear) -> P = E * G_res (+ comfort noise when enabled)   (the
          conventional post-RES spectrum with the denoiser off) -> DFN2
          estimated on E, applied on P -> WOLA

The host runs on the product's own grid (16 kHz by default).  At 48 kHz the
stage takes E and P directly; at 16 kHz the rate bridge
(:class:`pipelines.dfn2_rate_adapter.Dfn2RateBridge`, the twin of
``pipelines/dfn_rate_bridge.c``) runs the same stage on its 48 kHz grid: only
DFN2 leaves the product rate, the AEC, RES and comfort noise never do.

This is the Python twin of the ``mono_aec_dfn_res`` C pipeline.  The
conventional twin (``aec_nr_pipeline.py``) fuses ``min(G_nr, G_res)`` and
applies it once to E; here the conventional post spectrum is formed with the
denoiser off (RES gain, then the comfort-noise fill), and the DeepFilterNet2
stage (``dfn2_stage.py``) runs as a true cascade on that post-RES spectrum
while its features are estimated from the pre-RES one.  The comfort noise
therefore travels through the network like every other component of P.

Usage:
    cd Audio_ALG
    python -m pipelines.aec_dfn2_res_pipeline --mic mic.wav --ref ref.wav --output out.wav \
        --checkpoint AINR/DeepFilterNet2/output/dfn2_best.pth

Experiment switches (Python only; the C pipeline ships pre_res / full):
    --dfn-estimate-source {pre_res,post_res,mic}
    --dfn-compose {full,mask_only}
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_AEC_PY = os.path.join(_ROOT, 'lib', 'aec', 'python')
if _AEC_PY not in sys.path:
    sys.path.insert(0, _AEC_PY)

from lib.aec.python.aec import AecConfig, AecMode, AecPreset, AecResContext  # noqa: E402
from pipelines.aec_nr_pipeline import _project_grid, run_aec_linear, synth_window  # noqa: E402
from pipelines.cng_aec3 import Aec3ComfortNoise  # noqa: E402
from pipelines.dfn2_rate_adapter import Dfn2RateBridge  # noqa: E402
from pipelines.dfn2_stage import (  # noqa: E402
    DF_BINS, DF_ORDER, HOP, MODEL_LOOKAHEAD, N_BINS, N_ERB, Dfn2Assets,
    Dfn2Heads, Dfn2Stage, IdentityHeads, load_dfn2,
)

PSD_SCALE = 32768.0 ** 2
ESTIMATE_SOURCES = ('pre_res', 'post_res', 'mic')
COMPOSE_MODES = ('full', 'mask_only')


class MaskOnlyHeads:
    """Variant (c'): the network's ERB mask with the deep filter disabled."""

    def __init__(self, inner):
        self.inner = inner

    def reset(self):
        self.inner.reset()

    def __call__(self, erb_window, spec_window):
        mask, _coefs, _alpha = self.inner(erb_window, spec_window)
        return mask, np.zeros((DF_BINS, DF_ORDER, 2), np.float32), 0.0


def make_stage(assets: Optional[Dfn2Assets], compose: str = 'full',
               atten_lim_db: float = 0.0, identity: bool = False) -> Dfn2Stage:
    if assets is None:
        assets = load_dfn2(seed=0)
    if identity:
        heads = IdentityHeads()
    else:
        heads = Dfn2Heads(assets.model)
        if compose == 'mask_only':
            heads = MaskOnlyHeads(heads)
        elif compose != 'full':
            raise ValueError(f'unknown compose mode {compose!r}')
    return Dfn2Stage(assets, heads=heads, atten_lim_db=atten_lim_db)


def _bridged_stream(bridge, native_hops, n_freqs, hop, output_length, realtime_timing):
    """The bridge emits one native hop per hop, `added_delay` samples late.
    Offline, feed silence until that tail is out and align with the
    conventional twin; in real time keep the C timeline."""
    if not realtime_timing:
        silence = np.zeros(n_freqs, np.complex64)
        for _ in range((bridge.added_delay + hop - 1) // hop + 1):
            native_hops.append(bridge.push(silence, silence))
    stream = np.concatenate(native_hops) if native_hops else np.zeros(0, np.float32)
    if realtime_timing:
        return stream[:output_length]
    aligned = stream[bridge.added_delay:bridge.added_delay + output_length]
    if len(aligned) < output_length:
        aligned = np.pad(aligned, (0, output_length - len(aligned)))
    return aligned


def run_dfn2_res(aec_contexts: Sequence[AecResContext], config: AecConfig,
                 stage: Dfn2Stage, *,
                 estimate_source: str = 'pre_res',
                 estimate_spectra: Optional[Sequence[np.ndarray]] = None,
                 use_res: bool = True,
                 enable_cng: Optional[bool] = None,
                 realtime_timing: bool = False,
                 output_length: Optional[int] = None) -> np.ndarray:
    """Post-RES spectrum, DFN2 cascade and WOLA, frame by frame.

    Per frame the conventional post spectrum with the denoiser off is formed
    first: ``P = E * G_res`` plus the comfort-noise fill (``cng_aec3``, the
    bit-exact twin of the C recipe with the denoiser factor at unity).  The
    DFN2 heads are then applied to P.

    ``estimate_source`` picks the spectrum the DFN2 features are computed
    from: ``pre_res`` = E (the shipped design), ``post_res`` = P (the naive
    cascade, A == B), ``mic`` = the windowed capture spectrum
    ``near_spec = error_spec + echo_spec``.

    ``estimate_spectra`` overrides that selection with one spectrum per AEC
    context.  The 4-channel evaluator uses this for a raw capture lane or a
    pre-beam linear lane while retaining the fused post-beam context as the
    application path.  It never changes S1.

    ``realtime_timing=False`` (offline reference) writes source frame f at
    sample f*hop, compensating the two-frame model lookahead so the output
    lines up with the conventional pipeline's.  ``True`` reproduces the C
    real-time timing: frame f lands at (f + 2)*hop and the first two hops are
    zero.
    """
    if estimate_source not in ESTIMATE_SOURCES:
        raise ValueError(f'estimate_source must be one of {ESTIMATE_SOURCES}')
    if estimate_spectra is not None and len(estimate_spectra) != len(aec_contexts):
        raise ValueError('estimate_spectra must contain one spectrum per AEC context')
    bs = int(config.frame_size)
    hop = int(config.hop_size)
    fft = int(config.fft_size)
    n_freqs = fft // 2 + 1
    native = (bs, hop, n_freqs) != (2 * HOP, HOP, N_BINS)
    if native and int(config.sample_rate) not in (8000, 16000):
        raise ValueError('the DFN2 stage runs at 48 kHz directly or through the '
                         f'rate bridge from 8/16 kHz; got {config.sample_rate} Hz '
                         f'frame={bs} hop={hop}')
    if enable_cng is None:
        enable_cng = bool(config.enable_cng)

    n_frames = len(aec_contexts)
    if output_length is None:
        output_length = n_frames * hop
    delay_frames = MODEL_LOOKAHEAD if realtime_timing else 0
    total_frames = n_frames + delay_frames
    output = np.zeros(total_frames * hop, dtype=np.float32)
    synth_win = synth_window(bs)
    ola = np.zeros(bs, dtype=np.float32)
    cng = Aec3ComfortNoise()
    stage.reset()
    bridge = Dfn2RateBridge(stage, int(config.sample_rate), fft) if native else None
    native_hops: List[np.ndarray] = []

    emitted = 0

    def emit(out_spec: np.ndarray, frame_index: int):
        nonlocal ola, emitted
        spec = np.asarray(out_spec, dtype=np.complex64)
        e_full = np.fft.irfft(spec, n=fft).astype(np.float32)
        ola += e_full[:bs] * synth_win
        slot = emitted if realtime_timing else frame_index
        start = slot * hop
        output[start:start + hop] = ola[:hop]
        ola[:-hop] = ola[hop:]
        ola[-hop:] = 0.0
        emitted += 1

    if realtime_timing and not native:
        # The two warm-up hops synthesise an empty spectrum, exactly as the C
        # pipeline does, so the delay is a pure delay.
        for _ in range(MODEL_LOOKAHEAD):
            emit(np.zeros(n_freqs, np.complex64), 0)

    unity = np.ones(n_freqs, np.float32)
    for context_index, ctx in enumerate(aec_contexts):
        if ctx.error_spec is None or ctx.res_gain is None:
            raise ValueError('AEC built without the RES seam (return_res_context)')
        e = np.asarray(ctx.error_spec, dtype=np.complex64)
        g_res = np.asarray(ctx.res_gain, dtype=np.float32) if use_res else unity
        # The conventional post spectrum with the denoiser off: RES gain,
        # then the comfort-noise fill (denoiser factor max(1, floor) = 1),
        # drawn at this frame exactly as the C host draws it.
        post = (e * g_res).astype(np.complex64)
        if enable_cng and use_res and ctx.comfort_noise is not None:
            comfort = np.asarray(ctx.comfort_noise, dtype=np.float32)
            n_amp = np.sqrt(np.maximum(comfort / PSD_SCALE, 0.0)).astype(np.float32)
            noise_gain = np.sqrt(np.maximum(1.0 - g_res * g_res, 0.0)).astype(np.float32)
            cng.add(post, (noise_gain * n_amp).astype(np.float32))
        if estimate_spectra is not None:
            a = np.asarray(estimate_spectra[context_index], dtype=np.complex64)
            if a.shape != e.shape:
                raise ValueError('external estimate spectrum shape does not match E')
        elif estimate_source == 'pre_res':
            a = e
        elif estimate_source == 'post_res':
            a = post
        else:
            if ctx.near_spec is None:
                raise ValueError('mic estimation needs ctx.near_spec')
            a = np.asarray(ctx.near_spec, dtype=np.complex64)   # E + echo estimate
        if bridge is not None:
            native_hops.append(bridge.push(a, post))
            continue
        result = stage.push(a, post)
        if result is not None:
            emit(result.spectrum, result.frame_index)

    if bridge is not None:
        return _bridged_stream(bridge, native_hops, n_freqs, hop, output_length,
                               realtime_timing)

    for result in stage.flush():
        emit(result.spectrum, result.frame_index)

    if realtime_timing:
        return output[:output_length + MODEL_LOOKAHEAD * hop]
    return output[:output_length]


def main():
    parser = argparse.ArgumentParser(
        description='AEC + RES + DeepFilterNet2 pipeline (Python reference)')
    parser.add_argument('--mic', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default=os.path.join(
        _ROOT, 'AINR', 'DeepFilterNet2', 'output', 'dfn2_best.pth'))
    parser.add_argument('--aec-preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive'])
    parser.add_argument('--dfn-estimate-source', default='pre_res', choices=ESTIMATE_SOURCES)
    parser.add_argument('--dfn-compose', default='full', choices=COMPOSE_MODES)
    parser.add_argument('--atten-lim', type=float, default=0.0,
                        help='DFN2 attenuation limit in dB (0 = unlimited)')
    parser.add_argument('--cng', action='store_true',
                        help='enable the comfort-noise fill inside P (off by default, '
                             'as in the C pipelines)')
    parser.add_argument('--no-res', action='store_true',
                        help='experiment only: apply DFN2 to E instead of the post-RES spectrum')
    parser.add_argument('--identity-model', action='store_true',
                        help='skip the network: exact delayed identity of the post-RES spectrum')
    parser.add_argument('--realtime-timing', action='store_true',
                        help='keep the model delay in the output (C timing: two hops '
                             'at 48 kHz, the rate bridge delay otherwise)')
    args = parser.parse_args()

    mic_signal, sr_mic = sf.read(args.mic, dtype='float32')
    ref_signal, sr_ref = sf.read(args.ref, dtype='float32')
    if mic_signal.ndim > 1:
        mic_signal = mic_signal[:, 0]
    if ref_signal.ndim > 1:
        ref_signal = ref_signal[:, 0]
    if sr_mic != sr_ref:
        parser.error(f'sample rate mismatch ({sr_mic} vs {sr_ref})')
    native_rate = int(sr_mic)
    native_length = min(len(mic_signal), len(ref_signal))
    mic_signal = np.asarray(mic_signal[:native_length], np.float32)
    ref_signal = np.asarray(ref_signal[:native_length], np.float32)
    if native_rate not in (16000, 48000):
        parser.error('the Python host runs at 16 or 48 kHz (the C mono pipeline '
                     f'also serves 8 kHz); got {native_rate}')
    frame_size, hop_size, fft_size = _project_grid(native_rate)
    preset = {'mild': AecPreset.MILD, 'balanced': AecPreset.BALANCED,
              'aggressive': AecPreset.AGGRESSIVE}[args.aec_preset]
    aec_config = AecConfig.from_preset(
        preset, sample_rate=native_rate, frame_size=frame_size, hop_size=hop_size,
        mode=AecMode.PBFDKF, mu=0.3, enable_res=True,
        enable_cng=args.cng)

    print('AEC + RES + DFN2 Pipeline')
    print(f'Input:  {args.mic} ({native_length} samples @ {native_rate} Hz)')
    print(f'Grid:   frame={frame_size}, hop={hop_size}, fft={fft_size} (host at {native_rate} Hz)')
    if native_rate != 48000:
        print('Rate:   DFN2 at 48 kHz through the rate bridge; AEC/RES stay at '
              f'{native_rate} Hz')
    print(f'DFN2:   estimate={args.dfn_estimate_source} compose={args.dfn_compose} '
          f'atten_lim={args.atten_lim} dB')

    print('Stage 1: AEC (linear, no RES)...')
    aec_output, contexts = run_aec_linear(mic_signal, ref_signal, aec_config)
    print('Stage 2: post-RES spectrum -> DFN2 cascade...')
    assets = None if args.identity_model else load_dfn2(checkpoint=args.checkpoint)
    stage = make_stage(assets, compose=args.dfn_compose,
                       atten_lim_db=args.atten_lim, identity=args.identity_model)
    final = run_dfn2_res(
        contexts, aec_config, stage,
        estimate_source=args.dfn_estimate_source,
        use_res=not args.no_res,
        realtime_timing=args.realtime_timing,
        output_length=len(mic_signal))
    sf.write(args.output, final, native_rate)
    print(f'Done: {args.output} ({len(final)} samples, '
          f'{stage.heads_calls} head evaluations, {stage.skips} skips)')


if __name__ == '__main__':
    main()
