"""
AEC + NR + RES Processing Pipeline
聲學回聲消除 + 降噪 + 殘留回聲抑制串接管線

Pipeline: AEC(linear) → noise-only NR(E) → RES  (freq A_min_pl, production).
AEC and NR are each selected by their own preset; the CLI exposes only
presets + switches.

Usage:
    cd Audio_ALG
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-preset balanced --nr-preset balanced
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --nr-preset aggressive
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-only
"""

import sys
import os
import argparse
from typing import List, Tuple

import numpy as np
import soundfile as sf

# Add parent directory to path for imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                                       # lib.* + pipelines.*
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'aec', 'python')) # aec + modules

from lib.aec.python.aec import (
    AEC, AecConfig, AecMode, AecPreset, AecResContext,
)
from lib.nr.denoisers import MmseLsaDenoiser

# The standalone ResFilter was retired in AEC v3.21; the separated
# `--pipeline-mode linear` (AEC-linear -> NR -> RES) is now rebuilt on the
# AEC3 freq-domain seam (AecConfig.return_res_context exposes the linear
# error spectrum + the per-frame AEC3 SuppressionGain), so the post-NR RES is
# a pure freq multiply — see run_res() + docs/freq_domain_pipeline_design.md.


def parse_aec_mode(mode_str: str) -> AecMode:
    modes = {
        'fdaf': AecMode.FDAF, 'pbfdaf': AecMode.PBFDAF,
        'pbfdkf': AecMode.PBFDKF,
        'subband': AecMode.SUBBAND,  # backward compat alias (= PBFDKF)
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
                   config: AecConfig) -> Tuple[np.ndarray, List[AecResContext]]:
    """Run AEC without RES, returning per-frame context for external RES."""
    config.enable_res = False
    config.return_res_context = True
    aec = AEC(config)
    hop = aec.hop_size
    min_len = min(len(mic_signal), len(ref_signal))
    output = np.zeros(min_len, dtype=np.float32)
    contexts: List[AecResContext] = []

    for i in range(0, min_len - hop + 1, hop):
        result = aec.process(mic_signal[i:i + hop], ref_signal[i:i + hop])
        out, ctx = result
        output[i:i + hop] = out
        contexts.append(ctx)

    print(f"  AEC ERLE: {aec.get_erle():.1f} dB  ({len(contexts)} frames)")
    return output, contexts


# NR strength presets — mirror the C config_for_mode strength quartet
# (lib/nr/c_impl/include/mmse_lsa_types.h). MmseLsaDenoiser takes the
# {g_min, q, xi_min, alpha_g} strength set; the pipeline's structural tuning
# (L=150, alpha_s, frame/hop, MCRA — see _build_denoiser) is preserved either
# way. The C-only alpha_d / alpha_attack / alpha_decay are not part of Python's
# NR param structure. BALANCED == the legacy fixed values, so a balanced
# pipeline run is byte-equal to the pre-preset code.
NR_PRESETS = {
    'mild':       dict(g_min_db=-10.0, q=0.60, xi_min_db=-15.0, alpha_g=0.92),
    'balanced':   dict(g_min_db=-15.0, q=0.50, xi_min_db=-20.0, alpha_g=0.88),
    'aggressive': dict(g_min_db=-20.0, q=0.35, xi_min_db=-25.0, alpha_g=0.75),
}


def _build_denoiser(sample_rate: int,
                    nr_preset: str = 'balanced') -> MmseLsaDenoiser:
    """MMSE-LSA denoiser on the shared 10ms-hop grid (frame=320, shift=160).

    ``nr_preset`` (mild/balanced/aggressive) selects the strength quartet
    {g_min, q, xi_min, alpha_g} from NR_PRESETS. The pipeline structural tuning
    below is preset-independent.

    INTENTIONAL divergence from SE/NR/config/v3_2_config.yaml (tuned for
    frame=512/hop=256 at 16ms/frame):
      - alpha_d not passed → falls back to alpha_noise=0.95 (yaml=0.7).
        Slower noise-floor tracking is appropriate at 10ms hop (shorter
        frames = noisier per-frame estimates).
      - broadband_threshold not passed → default 0.8 (yaml=1.0, disabled).
        The broadband scene-reset path is active here; on AEC residual
        signals this provides faster adaptation after echo bursts.
      - L=150 (1.5s window) vs yaml L=32 (320ms). Longer window improves
        stationarity estimation against the AEC-residual echo pedestal.
    The A_min_pl pipeline was benchmarked and shipped with these values.
    DO NOT silently sync to the NR yaml without re-running 800-case bench.
    """
    p = NR_PRESETS[nr_preset]
    return MmseLsaDenoiser(
        sample_rate=sample_rate,
        frame_size=320,          # 20ms — matches AEC frame_size
        frame_shift=160,         # 10ms — matches AEC hop_size
        fft_size=512,
        noise_method='mcra',
        g_min_db=p['g_min_db'],
        alpha_g=p['alpha_g'],
        alpha_xi=0.88,
        q=p['q'],
        xi_min_db=p['xi_min_db'],
        alpha_s=0.95,
        L=150,                   # 150 × 10ms = 1.5s minima window
        delta_db=10.0,
        num_init_frames=20,
        scene_change_threshold_db=10.0,
        scene_change_min_frames=5,
        scene_change_blend=0.5,
    )


def run_nr(signal: np.ndarray, sample_rate: int,
           return_gain: bool = False, nr_preset: str = 'balanced',
           ) -> np.ndarray:
    """Run NR (MMSE-LSA) on a time signal with 10ms hop (frame=320, shift=160)."""
    denoiser = _build_denoiser(sample_rate, nr_preset)
    result = denoiser.denoise(signal, return_gain=return_gain)
    if return_gain:
        enhanced, gains = result
        print(f"  NR: {gains.shape[0]} frames, gain shape {gains.shape}")
        return enhanced, gains
    return result


def run_nr_spectrum(aec_contexts: List[AecResContext], sample_rate: int,
                    nr_preset: str = 'balanced') -> np.ndarray:
    """NR directly on the AEC linear error spectra E(f) — no re-FFT.

    Consumes ``ctx.error_spec`` (the AEC's sqrt-Hann-windowed linear error
    spectrum) for every hop, runs MMSE-LSA in the frequency domain via
    ``denoise_spectrum``, and returns the per-frame gain G_nr(f) (n_frames, 257)
    for the freq-domain RES. This is the FFT-deduplicated path: the spectra the
    time-domain run_nr would re-derive are already in the context.
    """
    spectra = np.stack([np.asarray(c.error_spec, dtype=np.complex64)
                        for c in aec_contexts])          # (n_frames, 257)
    magnitude = np.abs(spectra).astype(np.float64)
    phase = np.angle(spectra).astype(np.float64)
    denoiser = _build_denoiser(sample_rate, nr_preset)
    _, _, gains = denoiser.denoise_spectrum(
        magnitude, phase, return_gain=True)
    print(f"  NR(freq): {gains.shape[0]} frames on E(f), gain shape {gains.shape}")
    return gains.astype(np.float32)


def run_res(nr_output: np.ndarray, nr_gains: np.ndarray,
            aec_contexts: List[AecResContext],
            config: AecConfig,
            use_nr: bool = True, use_res: bool = True,
            dt_relax: float = 0.0, ne_floor: float = 0.0,
            ne_gate: str = 'r2', combine: str = 'product') -> np.ndarray:
    """Residual-echo suppression AFTER NR, in the frequency domain.

    Reuses the linear AEC's own per-frame AEC3 SuppressionGain (``ctx.res_gain``)
    and its windowed error spectrum (``ctx.error_spec``), exposed via
    ``AecConfig.return_res_context``. The post-NR residual is a single freq
    multiply per hop::

        S(f) = E(f) · G_nr(f) · G_res(f)   (+ comfort noise on the cut bins)

    then one sqrt-Hann OLA — no standalone ResFilter, no extra FFT round-trip.
    The sqrt-Hann synthesis window matches the AEC's ``_aec3_synth_window`` exactly
    (periodic Hann, denom = block_size) for perfect reconstruction.

    Frame alignment: ``ctx[i].error_spec`` spans [(i-1)*hop, (i+1)*hop) (AEC
    analysis window ends at the current hop). In ``--pipeline-mode freq`` NR
    is also computed from ``ctx.error_spec`` so frame i = frame i. In
    ``--pipeline-mode linear`` NR is computed from the AEC time-domain output
    whose frame i covers [i*hop, (i+2)*hop), so ``nr_gains[i]`` is one hop
    AHEAD of ``ctx[i].error_spec`` — callers should pass ``nr_gains`` with a
    one-frame shift (``np.concatenate([[nr_gains[0]], nr_gains[:-1]])``) or
    use ``--pipeline-mode freq`` (the default).
    """
    bs = int(config.frame_size)      # 320 — analysis/synthesis block
    hop = int(config.hop_size)       # 160
    fft = 512
    n_freqs = fft // 2 + 1           # 257
    psd_scale = 32768.0 ** 2         # int16² scale of ctx.comfort_noise (AEC3)

    idx = np.arange(bs, dtype=np.float64)
    synth_win = np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * idx / float(bs)))).astype(np.float32)
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
        # Save AEC-only gain before combining with G_nr, for CNG noise level.
        # CNG must reflect AEC suppression only — using g_total would re-inject
        # noise into NR-suppressed bins (BAK ceiling).
        g_aec = g_res if isinstance(g_res, np.ndarray) else np.full(n_freqs, g_res, dtype=np.float32)
        if use_nr and use_res and combine == 'min':
            # A_min_pl: per-bin min recovers the AEC3 echo gain that v0 discarded
            # — near-end bins (g_res≈1) keep g_nr; echo bins (g_res<g_nr) cut the
            # leak — WITHOUT the product's double-cut on double-talk near-end.
            g_total = np.minimum(g_nr, g_res).astype(np.float32)
        else:
            g_total = (g_nr * g_res).astype(np.float32) if (use_nr or use_res) \
                else np.ones(n_freqs, dtype=np.float32)
        # Near-end-aware relax: in double-talk (ctx.dt_indicator high), the
        # G_nr·G_res product over-cuts near speech. Blend the total gain toward
        # 1.0 by dt_relax·dt_indicator so DT near-end is preserved (FS, where
        # dt_indicator≈0, is untouched). dt_relax=0 → off (plain product).
        if dt_relax > 0.0:
            w = float(dt_relax) * float(ctx.dt_indicator)   # 0..(dt_relax·0.8)
            if w > 0.0:
                g_total = (1.0 - w) * g_total + w * 1.0
        # Lightweight near-end preservation floor (per-bin, echo-aware): lift the
        # gain toward 1.0 ONLY where there is little residual echo. echo_frac =
        # R²/|E|² ≈ 0 in clean near-end (→ full lift, preserve speech) and ≈ 1 in
        # echo-dominated bins (→ no lift, stay suppressed). Unlike dt_relax this
        # keys off per-bin echo, not a frame scalar, so it never lifts in far-end.
        if ne_floor > 0.0 and ctx.r2 is not None:
            r2_nr = np.asarray(ctx.r2, dtype=np.float32) / psd_scale
            e2 = np.abs(ctx.error_spec).astype(np.float32) ** 2
            echo_frac = np.clip(r2_nr / (e2 + 1e-12), 0.0, 1.0)
            no_echo_r2 = 1.0 - echo_frac
            g_res = ctx.res_gain.astype(np.float32)  # AEC3 gain ≈1 no-echo, low=echo
            if ne_gate == 'r2':
                no_echo = no_echo_r2
            elif ne_gate == 'resgain':
                no_echo = g_res
            elif ne_gate == 'both_sharp':  # res_gain² → lift drops faster in echo bins
                no_echo = (g_res ** 2) * no_echo_r2
            else:  # 'both' — product gate: lift only where BOTH say no echo
                no_echo = g_res * no_echo_r2
            lift = float(ne_floor) * no_echo
            g_total = (1.0 - lift) * g_total + lift * 1.0
        spec = ctx.error_spec.astype(np.complex64) * g_total

        # Comfort noise fills the suppressed bins (port of AEC3 RES CNG; the
        # AEC3 sin-LUT/LCG source is approximated with deterministic Gaussian
        # noise at the same per-bin level — this is a separated-pipeline
        # baseline, not bit-identical to the internal RES).
        if config.enable_cng and ctx.comfort_noise is not None:
            n_amp = np.sqrt(np.maximum(ctx.comfort_noise / psd_scale, 0.0)).astype(np.float32)
            noise_gain = np.sqrt(np.maximum(1.0 - g_aec ** 2, 0.0)).astype(np.float32)
            cn = np.zeros(n_freqs, dtype=np.complex64)
            cn[1:-1] = (n_amp[1:-1]
                        * (rng.randn(n_freqs - 2) + 1j * rng.randn(n_freqs - 2)))
            spec = spec + noise_gain * cn

        e_full = np.fft.irfft(spec, n=fft).astype(np.float32)
        ola += e_full[:bs] * synth_win
        output[start:end] = ola[:hop]
        ola[:-hop] = ola[hop:]
        ola[-hop:] = 0.0

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
  --aec-preset  gentle | balanced | aggressive   (echo-vs-near strength)
  --nr-preset   mild   | balanced | aggressive   (noise-suppression strength)
Switches:
  --aec-only    run AEC only, skip NR/RES
""")
    parser.add_argument('--mic', required=True, help='Microphone input WAV')
    parser.add_argument('--ref', required=True, help='Reference/loudspeaker WAV')
    parser.add_argument('--output', required=True, help='Output WAV')
    parser.add_argument('--aec-preset', default='balanced',
                        choices=['gentle', 'balanced', 'aggressive'],
                        help='AEC preset (default: balanced)')
    parser.add_argument('--nr-preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive'],
                        help='NR preset (default: balanced)')
    parser.add_argument('--aec-only', action='store_true',
                        help='Run AEC only, skip NR/RES')
    args = parser.parse_args()

    # Load audio
    mic_signal, sr_mic = sf.read(args.mic, dtype='float32')
    ref_signal, sr_ref = sf.read(args.ref, dtype='float32')

    if sr_mic != sr_ref:
        print(f"Error: sample rate mismatch ({sr_mic} vs {sr_ref})", file=sys.stderr)
        sys.exit(1)

    sample_rate = sr_mic
    duration = len(mic_signal) / sample_rate

    preset_map = {
        'gentle': AecPreset.GENTLE, 'balanced': AecPreset.BALANCED,
        'aggressive': AecPreset.AGGRESSIVE,
    }
    preset = preset_map[args.aec_preset]

    print("AEC + NR + RES Pipeline")
    print("=======================")
    print(f"Input:    {args.mic} ({len(mic_signal)} samples, {duration:.2f}s)")
    print(f"Ref:      {args.ref}")
    print(f"Output:   {args.output}")
    print(f"Rate:     {sample_rate} Hz")
    print(f"AEC:      preset={args.aec_preset}")
    if not args.aec_only:
        print(f"NR:       preset={args.nr_preset}")
    else:
        print("NR/RES:   disabled (--aec-only)")
    print()

    # AEC config: production filter (PBFDKF); the preset sets the echo-vs-near
    # min-gain floor (the only inter-preset difference).
    aec_config = AecConfig.from_preset(
        preset,
        sample_rate=sample_rate,
        mode=AecMode.PBFDKF,
        mu=0.3,
        enable_res=True,
    )

    # freq A_min_pl pipeline (production): AEC(linear) -> noise-only NR(E) ->
    # g_total = min(G_nr, G_res) + per-bin echo-gated near-end floor. NR handles
    # noise; the AEC's own near-end-aware AEC3 echo gain G_res handles echo; the
    # per-bin min keeps both without the product's double-talk double-cut.
    print("Stage 1: AEC (linear, no RES)...")
    aec_output, contexts = run_aec_linear(mic_signal, ref_signal, aec_config)

    if args.aec_only:
        final_output = aec_output
    else:
        print("Stage 2: noise-only NR on E(f)...")
        nr_gains = run_nr_spectrum(contexts, sample_rate, nr_preset=args.nr_preset)
        print("Stage 3: g_total=min(G_nr,G_res) + per-bin near-end floor...")
        # Pass aec_output (zero-padded to mic length) as the tail source so
        # samples beyond the last full AEC frame are not silenced (mic>ref case).
        _tail_src = np.zeros(len(mic_signal), dtype=np.float32)
        _tail_src[:len(aec_output)] = aec_output
        final_output = run_res(
            _tail_src, nr_gains, contexts, aec_config,
            use_res=True, combine='min', ne_floor=0.4, ne_gate='both')

    # Save
    sf.write(args.output, final_output, sample_rate)
    print(f"\nDone! Output: {args.output} ({len(final_output)} samples)")


if __name__ == '__main__':
    main()
