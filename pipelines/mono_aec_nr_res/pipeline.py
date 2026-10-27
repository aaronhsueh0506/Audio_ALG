"""
AEC + NR + RES Processing Pipeline
聲學回聲消除 + 降噪 + 殘留回聲抑制串接管線

Pipeline: AEC(linear) → echo-aware NR(E) → RES  (frequency domain).
The NR folds the AEC residual-echo PSD R²(f) into its noise floor (ξ=S²/(N²+R²))
and g_total = min(G_nr, G_res) is applied as computed; pass --legacy-amin for
the noise-only NR. AEC and NR are each selected by their own preset; the
CLI exposes only presets + switches.

Usage:
    cd Audio_ALG
    python -m pipelines.mono_aec_nr_res.pipeline --mic mic.wav --ref ref.wav --output out.wav
    python -m pipelines.mono_aec_nr_res.pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-preset balanced --nr-preset balanced
    python -m pipelines.mono_aec_nr_res.pipeline --mic mic.wav --ref ref.wav --output out.wav --nr-preset aggressive
    python -m pipelines.mono_aec_nr_res.pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-only
"""

import sys
import os
import argparse
from typing import List, Tuple

import numpy as np
import soundfile as sf

# Add parent directory to path for imports
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)                                       # lib.* + pipelines.*
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'aec', 'python')) # aec + modules
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'nr', 'python'))  # NR modules

from lib.aec.python.aec import (
    AEC, AecConfig, AecMode, AecPreset, AecResContext,
)
from lib.nr.python.denoisers import MmseLsaDenoiser
from lib.nr.python.core.nr_strength import apply_strength
from lib.nr.python.process_audio import build_v3_2_base_params, load_config

# The post-NR RES is a pure frequency multiply on the AEC3 seam
# (AecConfig.return_res_context exposes the linear error spectrum and the
# per-frame AEC3 SuppressionGain); see run_res() and
# docs/freq_domain_pipeline_design.md.


def parse_aec_mode(mode_str: str) -> AecMode:
    modes = {
        'fdaf': AecMode.FDAF, 'pbfdaf': AecMode.PBFDAF,
        'pbfdkf': AecMode.PBFDKF,
        'subband': AecMode.SUBBAND,  # alias of pbfdkf
    }
    return modes.get(mode_str, AecMode.PBFDKF)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_aec_classic(mic_signal: np.ndarray, ref_signal: np.ndarray,
                    config: AecConfig) -> np.ndarray:
    """Run AEC with internal RES (classic mode)."""
    aec = AEC(config)
    hop = aec.hop_size
    min_len = min(len(mic_signal), len(ref_signal))
    output = np.zeros(min_len, dtype=np.float32)

    for i in range(0, min_len - hop + 1, hop):
        output[i:i + hop] = aec.process(
            mic_signal[i:i + hop], ref_signal[i:i + hop])

    print(f"  AEC ERLE: {aec.get_erle():.1f} dB")
    return output


def run_aec_linear(mic_signal: np.ndarray, ref_signal: np.ndarray,
                   config: AecConfig, *,
                   standalone_time_output: bool = False,
                   ) -> Tuple[np.ndarray, List[AecResContext]]:
    """Run AEC without RES, returning per-frame context for external RES.

    By default the time output is the reconstructing WOLA hop underlying
    ``ctx.error_spec``. ``standalone_time_output=True`` instead returns the
    AEC library's direct time output, matching the C ``aec_only`` path; the
    frequency-domain context is unchanged.
    """
    config.enable_res = False
    config.return_res_context = True
    aec = AEC(config)
    hop = aec.hop_size
    min_len = min(len(mic_signal), len(ref_signal))
    output = np.zeros(min_len, dtype=np.float32)
    contexts: List[AecResContext] = []

    for i in range(0, min_len - hop + 1, hop):
        result = aec.process(mic_signal[i:i + hop], ref_signal[i:i + hop])
        _out, ctx = result
        # The external WOLA seam may select/crossfade the shadow output.  The
        # full pipeline needs the exact formed hop underlying ctx.error_spec;
        # AEC-only mirrors C and returns the library's direct output.
        output[i:i + hop] = _out if standalone_time_output else ctx.formed_output
        contexts.append(ctx)

    print(f"  AEC ERLE: {aec.get_erle():.1f} dB  ({len(contexts)} frames)")
    return output, contexts


# NR parameters come from NR's single source of truth
# (lib/nr/python/config/v3_2_config.yaml with core/nr_strength.py's
# apply_strength() overlaid); this module keeps no preset table of its own.
_NR_YAML_CONFIG = os.path.join(_ROOT, 'lib', 'nr', 'python', 'config', 'v3_2_config.yaml')

# Production recipe: the AEC residual-echo PSD R²(f) is folded into the OM-LSA
# noise floor (ξ = S²/(N²+R²)) so the NR also suppresses residual echo, while
# g_res stays in the min (--legacy-amin selects the noise-only NR). There is no
# near-end floor: g_total is applied as computed.
PROD_INJECT_ECHO_PSD = True
# Comfort noise is scaled by the denoiser gain of its bin, never below this
# amplitude (-10 dB); the same value as CNG_NR_GAIN_FLOOR in mono_aec_nr_res/main.c and
# MCAB_SAMPLE_CNG_NR_GAIN_FLOOR in 4ch_aec_bf_nr_res/mcab_sample_post.h.
CNG_NR_GAIN_FLOOR = 0.31622777


def _project_grid(sample_rate: int, fft_size: int = None) -> Tuple[int, int, int]:
    """Resolve an integrated-pipeline no-padding ``(frame, hop, fft)`` grid."""
    allowed = {16000: (256, 512), 48000: (1024,)}
    # 16 kHz default is 256/128 (8 ms hop), as AEC, NR and the 4-channel
    # pipeline; 512 is an explicit alternate.
    defaults = {16000: 256, 48000: 1024}
    if sample_rate not in allowed:
        raise ValueError(f"supported sample rates are 16000 and 48000, got {sample_rate}")
    chosen = defaults[sample_rate] if fft_size is None else int(fft_size)
    if chosen not in allowed[sample_rate]:
        choices = "/".join(str(v) for v in allowed[sample_rate])
        raise ValueError(
            f"unsupported fft_size={chosen} at {sample_rate} Hz; expected {choices}"
        )
    return chosen, chosen // 2, chosen


def _build_denoiser(sample_rate: int,
                    nr_preset: str = 'balanced',
                    frame_size: int = None,
                    frame_shift: int = None,
                    fft_size: int = None) -> MmseLsaDenoiser:
    """MMSE-LSA denoiser on a power-of-two, no-padding 50%-overlap grid.

    Parameters come from NR's single source of truth: the base params of
    lib/nr/python/config/v3_2_config.yaml (``build_v3_2_base_params``) with
    ``apply_strength(nr_preset)`` overlaid, as ``create_denoiser_from_config()``
    does (no content-mode overlay), with no pipeline-specific overlay.
    ``broadband_threshold`` stays at the canonical 1.0 (scene-reset gate off).
    """
    # An explicit fft_size selects the matching frame/hop defaults.
    default_frame, default_hop, default_fft = _project_grid(sample_rate, fft_size)
    frame_size = default_frame if frame_size is None else int(frame_size)
    frame_shift = default_hop if frame_shift is None else int(frame_shift)
    fft_size = default_fft if fft_size is None else int(fft_size)
    if frame_size != fft_size:
        raise ValueError(
            f"no-padding invariant violated: frame_size={frame_size}, fft_size={fft_size}"
        )
    if frame_size != 2 * frame_shift:
        raise ValueError(
            f"50% overlap invariant violated: frame_size={frame_size}, "
            f"frame_shift={frame_shift}"
        )
    if fft_size <= 0 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a positive power of two")
    # Enforce the project whitelist as well as the generic DSP invariants.
    _project_grid(sample_rate, fft_size)

    config = load_config(_NR_YAML_CONFIG)
    if not config:
        raise RuntimeError(
            f"failed to load NR config at {_NR_YAML_CONFIG!r} (missing file, or "
            "PyYAML not installed) -- refusing to silently fall back to "
            "build_v3_2_base_params()'s own built-in defaults, which do NOT "
            "match this project's tuned v3_2_config.yaml (e.g. g_min_db "
            "-40.0 vs the real -30.0) and would reintroduce exactly the kind "
            "of silent divergent-duplicate bug this function was fixed to avoid"
        )
    params = build_v3_2_base_params(config, sample_rate, frame_size, frame_shift, fft_size)
    params = apply_strength(params, nr_preset)
    params['strength'] = nr_preset  # recorded on the denoiser
    params['mode'] = 'full'         # no content-mode selection in this pipeline (empty overlay)
    return MmseLsaDenoiser(**params)


def run_nr(signal: np.ndarray, sample_rate: int,
           return_gain: bool = False, nr_preset: str = 'balanced',
           ) -> np.ndarray:
    """Run NR on the sample-rate-specific no-padding project grid."""
    denoiser = _build_denoiser(sample_rate, nr_preset)
    result = denoiser.denoise(signal, return_gain=return_gain)
    if return_gain:
        enhanced, gains = result
        print(f"  NR: {gains.shape[0]} frames, gain shape {gains.shape}")
        return enhanced, gains
    return result


def run_nr_spectrum(aec_contexts: List[AecResContext], sample_rate: int,
                    nr_preset: str = 'balanced',
                    inject_echo_psd: bool = False) -> np.ndarray:
    """NR directly on the AEC linear error spectra E(f), with no re-FFT.

    Consumes ``ctx.error_spec`` per hop and returns the per-frame gain G_nr(f)
    (``n_frames, n_freqs``) for the frequency-domain RES.  ``inject_echo_psd``
    (default False) folds the residual-echo PSD R²(f) into the OM-LSA noise
    floor (ξ = S²/(N²+R²)) so the gain also suppresses residual echo (combine
    with ``min(g_nr, g_res)``); it is gated per hop on ``ctx.usable_linear``,
    and a hop with it False or None (e.g. a fused multi-lane context) adds no
    extra noise PSD, as a missing ``r2`` does.
    """
    if not aec_contexts:
        raise ValueError("aec_contexts must not be empty")
    spectra = np.stack([np.asarray(c.error_spec, dtype=np.complex64)
                        for c in aec_contexts])          # (n_frames, n_freqs)
    magnitude = np.abs(spectra).astype(np.float64)
    phase = np.angle(spectra).astype(np.float64)
    extra = None
    if inject_echo_psd:
        psd_scale = 32768.0 ** 2   # ctx.r2 is int16²-scaled like comfort_noise
        n_freqs = magnitude.shape[1]
        extra = np.stack([
            (np.asarray(c.r2, dtype=np.float64) / psd_scale)
            if (c.r2 is not None and c.usable_linear) else np.zeros(n_freqs)
            for c in aec_contexts])
    fft_size = 2 * (spectra.shape[1] - 1)
    denoiser = _build_denoiser(
        sample_rate, nr_preset,
        frame_size=fft_size, frame_shift=fft_size // 2, fft_size=fft_size,
    )
    _, _, gains = denoiser.denoise_spectrum(
        magnitude, phase, return_gain=True, extra_noise_psd=extra)
    print(f"  NR(freq): {gains.shape[0]} frames on E(f), gain shape {gains.shape}"
          f"{' [echo-aware ξ=S²/(N²+R²)]' if inject_echo_psd else ''}")
    return gains.astype(np.float32)



def synth_window(frame_size):
    """The periodic sqrt-Hann synthesis window (float32), the AEC's own."""
    idx = np.arange(frame_size, dtype=np.float64)
    return np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * idx / float(frame_size)))
                   ).astype(np.float32)

def total_gain(g_nr, g_res, use_nr: bool, use_res: bool, combine: str,
               n_freqs: int) -> np.ndarray:
    """Per-bin gain applied to E(f): the Python twin of ``sk_min_f32``.

    ``combine='min'`` with both gains on: near-end bins (g_res~1) keep g_nr,
    echo bins (g_res<g_nr) cut the leak, without the double-cut of the product
    on double-talk near-end.  A disabled source reads as the unity gain.
    """
    if use_nr and use_res and combine == 'min':
        return np.minimum(g_nr, g_res).astype(np.float32)
    return (g_nr * g_res).astype(np.float32) if (use_nr or use_res) \
        else np.ones(n_freqs, dtype=np.float32)


def apply_gain(error_spec: np.ndarray, g_total: np.ndarray) -> np.ndarray:
    """S(f) = E(f) . g_total: the Python twin of ``sk_capply_gain_f32``."""
    return error_spec.astype(np.complex64) * g_total


def comfort_noise_fill(spec: np.ndarray, comfort_noise: np.ndarray, g_aec, cn_scale,
                       rng: np.random.RandomState) -> np.ndarray:
    """Comfort noise on the cut bins: the Python twin of ``aec_cng_fill``.

    Port of AEC3 RES CNG; the AEC3 sin-LUT/LCG source is approximated with
    deterministic Gaussian noise at the same per-bin level -- this is a
    separated-pipeline baseline, not bit-identical to the internal RES.  The
    level is sqrt(N^2/PSD_SCALE) * sqrt(1 - G_res^2) * cn_scale; which bins it
    fills is decided by AEC suppression alone, never by g_total.
    """
    n_freqs = spec.shape[0]
    psd_scale = 32768.0 ** 2         # int16² scale of ctx.comfort_noise (AEC3)
    n_amp = np.sqrt(np.maximum(comfort_noise / psd_scale, 0.0)).astype(np.float32)
    noise_gain = np.sqrt(np.maximum(1.0 - g_aec ** 2, 0.0)).astype(np.float32)
    cn = np.zeros(n_freqs, dtype=np.complex64)
    cn[1:-1] = (n_amp[1:-1]
                * (rng.randn(n_freqs - 2) + 1j * rng.randn(n_freqs - 2)))
    return spec + noise_gain * cn_scale * cn


def synthesize_hop(spec: np.ndarray, ola: np.ndarray, synth_win: np.ndarray,
                   fft: int, bs: int, hop: int) -> np.ndarray:
    """Inverse transform, windowed overlap-add and the hop emit/shift: the
    Python twin of ``ac_istft_push``.  Updates ``ola`` in place and returns the
    emitted hop."""
    e_full = np.fft.irfft(spec, n=fft).astype(np.float32)
    ola += e_full[:bs] * synth_win
    emitted = ola[:hop].copy()
    ola[:-hop] = ola[hop:]
    ola[-hop:] = 0.0
    return emitted


def run_res(nr_output: np.ndarray, nr_gains: np.ndarray,
            aec_contexts: List[AecResContext],
            config: AecConfig,
            use_nr: bool = True, use_res: bool = True,
            combine: str = 'product') -> np.ndarray:
    """Residual-echo suppression AFTER NR, in the frequency domain.

    Per hop, in the order of the C applications, ``total_gain`` ->
    ``apply_gain`` -> ``comfort_noise_fill`` -> ``synthesize_hop`` (sqrt-Hann
    OLA, as the AEC's ``_aec3_synth_window``): S(f) = E(f) · G_nr(f) · G_res(f)
    plus comfort noise on the cut bins, from ``ctx.res_gain`` and
    ``ctx.error_spec`` (``AecConfig.return_res_context``).  ``nr_gains`` must
    come from the same spectra (``run_nr_spectrum``); gains from a time-domain
    NR run are one hop ahead and need a one-frame shift.
    """
    bs = int(config.frame_size)
    hop = int(config.hop_size)
    fft = int(config.fft_size)
    n_freqs = fft // 2 + 1
    if bs != fft:
        raise ValueError(
            f"no-padding invariant violated: frame_size={bs}, fft_size={fft}"
        )
    if bs != 2 * hop:
        raise ValueError(
            f"50% overlap invariant violated: frame_size={bs}, hop_size={hop}"
        )

    synth_win = synth_window(bs)
    ola = np.zeros(bs, dtype=np.float32)
    rng = np.random.RandomState(0)   # deterministic comfort noise

    n_frames = min(len(aec_contexts), nr_gains.shape[0])
    output = np.zeros(len(nr_output), dtype=np.float32)

    for i in range(n_frames):
        ctx = aec_contexts[i]
        if ctx.error_spec is None or ctx.res_gain is None:
            n_frames = i        # AEC built without the freq seam → stop here
            break
        start = i * hop
        end = start + hop
        if end > len(output):
            n_frames = i
            break

        g_nr = nr_gains[i].astype(np.float32) if use_nr else 1.0
        g_res = ctx.res_gain.astype(np.float32) if use_res else 1.0
        # Which bins the comfort noise fills is decided by AEC suppression alone
        # (sqrt(1 - G_res^2)), never by g_total; its level is scaled by G_nr
        # (floored at CNG_NR_GAIN_FLOOR) because the denoiser never sees it.
        g_aec = g_res if isinstance(g_res, np.ndarray) else np.full(n_freqs, g_res, dtype=np.float32)
        cn_scale = np.maximum(g_nr, CNG_NR_GAIN_FLOOR) if isinstance(g_nr, np.ndarray) else 1.0
        g_total = total_gain(g_nr, g_res, use_nr, use_res, combine, n_freqs)
        spec = apply_gain(ctx.error_spec, g_total)
        if config.enable_cng and ctx.comfort_noise is not None:
            spec = comfort_noise_fill(spec, ctx.comfort_noise, g_aec, cn_scale, rng)
        output[start:end] = synthesize_hop(spec, ola, synth_win, fft, bs, hop)

    # Copy tail (frames beyond AEC/NR context)
    tail_start = n_frames * hop
    if tail_start < len(nr_output):
        output[tail_start:] = nr_output[tail_start:]

    print(f"  RES: {n_frames} frames (freq-domain E·G_nr·G_res)")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AEC + NR + RES Pipeline (freq A_min_pl)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""AEC(linear) -> noise-only NR(E) -> RES   (g_total = min(G_nr, G_res)).
AEC and NR are each chosen by their own preset; the CLI exposes only
presets + switches.

Presets:
  --aec-preset  mild | balanced | aggressive   (echo-vs-near strength)
  --nr-preset   mild   | balanced | aggressive   (noise-suppression strength)
Switches:
  --aec-only    run AEC only, skip NR/RES
""")
    parser.add_argument('--mic', required=True, help='Microphone input WAV')
    parser.add_argument('--ref', required=True, help='Reference/loudspeaker WAV')
    parser.add_argument('--output', required=True, help='Output WAV')
    parser.add_argument(
        '--fft-size', type=int, choices=[256, 512, 1024], default=None,
        help=('No-padding FFT/frame size. Defaults: 256 at 16 kHz and 1024 at '
              '48 kHz; 16 kHz also supports 512. Hop is always FFT/2.'))
    parser.add_argument('--aec-preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive'],
                        help='AEC preset (default: balanced)')
    parser.add_argument('--nr-preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive'],
                        help='NR preset (default: balanced)')
    parser.add_argument('--aec-only', action='store_true',
                        help='Run AEC only, skip NR/RES')
    parser.add_argument('--legacy-amin', action='store_true',
                        help='Restore the prior min-only A_min_pl (no R² injection)')
    args = parser.parse_args()

    # Load audio
    mic_signal, sr_mic = sf.read(args.mic, dtype='float32')
    ref_signal, sr_ref = sf.read(args.ref, dtype='float32')

    if sr_mic != sr_ref:
        print(f"Error: sample rate mismatch ({sr_mic} vs {sr_ref})", file=sys.stderr)
        sys.exit(1)

    sample_rate = sr_mic
    try:
        frame_size, hop_size, fft_size = _project_grid(sample_rate, args.fft_size)
    except ValueError as exc:
        parser.error(str(exc))
    duration = len(mic_signal) / sample_rate

    preset_map = {
        'mild': AecPreset.MILD, 'balanced': AecPreset.BALANCED,
        'aggressive': AecPreset.AGGRESSIVE,
    }
    preset = preset_map[args.aec_preset]

    print("AEC + NR + RES Pipeline")
    print("=======================")
    print(f"Input:    {args.mic} ({len(mic_signal)} samples, {duration:.2f}s)")
    print(f"Ref:      {args.ref}")
    print(f"Output:   {args.output}")
    print(f"Rate:     {sample_rate} Hz")
    print(f"Grid:     frame={frame_size}, hop={hop_size}, fft={fft_size} (no padding)")
    print(f"AEC:      preset={args.aec_preset}")
    if not args.aec_only:
        print(f"NR:       preset={args.nr_preset}")
    else:
        print("NR/RES:   disabled (--aec-only)")
    print()

    # Production filter (PBFDKF); the preset sets the echo-vs-near min-gain floor.
    aec_config = AecConfig.from_preset(
        preset,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        mode=AecMode.PBFDKF,
        mu=0.3,
        enable_res=True,
    )

    # AEC(linear) -> NR(E) -> g_total = min(G_nr, G_res): NR handles noise, the
    # AEC's near-end-aware echo gain G_res handles echo, and the per-bin min
    # avoids the double-cut of the product on double-talk.
    print("Stage 1: AEC (linear, no RES)...")
    aec_output, contexts = run_aec_linear(
        mic_signal, ref_signal, aec_config,
        standalone_time_output=args.aec_only,
    )

    if args.aec_only:
        final_output = aec_output
    else:
        _legacy = getattr(args, 'legacy_amin', False)
        print("Stage 2: NR on E(f)"
              + ("" if _legacy else " [echo-aware ξ=S²/(N²+R²)]") + "...")
        nr_gains = run_nr_spectrum(contexts, sample_rate, nr_preset=args.nr_preset,
                                   inject_echo_psd=(not _legacy) and PROD_INJECT_ECHO_PSD)
        print("Stage 3: g_total=min(G_nr,G_res)...")
        # Pass aec_output (zero-padded to mic length) as the tail source so
        # samples beyond the last full AEC frame are not silenced (mic>ref case).
        _tail_src = np.zeros(len(mic_signal), dtype=np.float32)
        _tail_src[:len(aec_output)] = aec_output
        final_output = run_res(
            _tail_src, nr_gains, contexts, aec_config,
            use_res=True, combine='min')

    # Save
    sf.write(args.output, final_output, sample_rate)
    print(f"\nDone! Output: {args.output} ({len(final_output)} samples)")


if __name__ == '__main__':
    main()
