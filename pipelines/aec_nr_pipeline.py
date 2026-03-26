"""
AEC + NR + RES Processing Pipeline
聲學回聲消除 + 降噪 + 殘留回聲抑制串接管線

Pipeline modes:
  classic:  AEC(+RES) → NR         (legacy, RES inside AEC)
  linear:   AEC(linear) → NR → RES (recommended, separated RES)

Usage:
    cd Audio_ALG
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --pipeline-mode linear
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-only
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --preset balanced
"""

import sys
import os
import argparse
from typing import List, Tuple

import numpy as np
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.aec.python.aec import (
    AEC, AecConfig, AecMode, AecPreset, AecResContext, ResFilter,
)
from lib.nr.denoisers import MmseLsaDenoiser


def parse_aec_mode(mode_str: str) -> AecMode:
    modes = {
        'lms': AecMode.LMS, 'nlms': AecMode.NLMS,
        'fdaf': AecMode.FDAF, 'pbfdaf': AecMode.PBFDAF,
        'pbfdkf': AecMode.PBFDKF,
        'subband': AecMode.SUBBAND,  # backward compat alias
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


def run_nr(signal: np.ndarray, sample_rate: int,
           g_min_db: float = -15.0, return_gain: bool = False,
           ) -> np.ndarray:
    """Run NR (MMSE-LSA) with 10ms hop (frame=320, shift=160)."""
    denoiser = MmseLsaDenoiser(
        sample_rate=sample_rate,
        frame_size=320,          # 20ms — matches AEC frame_size
        frame_shift=160,         # 10ms — matches AEC hop_size
        fft_size=512,
        noise_method='mcra',
        g_min_db=g_min_db,
        alpha_g=0.88,
        alpha_xi=0.88,
        q=0.5,
        xi_min_db=-20.0,
        alpha_s=0.95,
        L=150,                   # 150 × 10ms = 1.5s minima window
        delta_db=10.0,
        num_init_frames=20,
        scene_change_threshold_db=10.0,
        scene_change_min_frames=5,
        scene_change_blend=0.5,
    )
    result = denoiser.denoise(signal, return_gain=return_gain)
    if return_gain:
        enhanced, gains = result
        print(f"  NR: {gains.shape[0]} frames, gain shape {gains.shape}")
        return enhanced, gains
    return result


def run_res(nr_output: np.ndarray, nr_gains: np.ndarray,
            aec_contexts: List[AecResContext],
            config: AecConfig) -> np.ndarray:
    """Apply RES post-filter with NR-gain-corrected echo PSD.

    All modules use hop=160 (10ms), so frame i of AEC = frame i of NR.
    """
    res = ResFilter(
        block_size=512,
        n_freqs=257,
        g_min_db=config.res_g_min_db,
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        echo_method=config.res_echo_method,
        gain_type=config.res_gain_type,
        enable_reverb=config.res_enable_reverb,
        reverb_decay=config.res_reverb_decay,
        reverb_gain=config.res_reverb_gain,
        alpha_echo_psd=config.res_alpha_echo_psd,
        alpha_error_psd=config.res_alpha_error_psd,
        enr_scale=config.res_enr_scale,
        enable_spectral_floor=config.res_spectral_floor,
        spectral_floor_db=config.res_spectral_floor_db,
        ne_protect_db=config.res_ne_protect_db,
    )
    hop = config.hop_size  # 160
    n_frames = min(len(aec_contexts), nr_gains.shape[0])
    output = np.zeros(len(nr_output), dtype=np.float32)

    for i in range(n_frames):
        ctx = aec_contexts[i]
        nr_gain = nr_gains[i]  # (257,) — 1:1 aligned with AEC frame

        # Correct echo spectrum: NR already attenuated these frequencies
        corrected_echo = ctx.echo_spec * nr_gain

        # Set dynamic over_sub from AEC context
        res.over_sub = ctx.over_sub

        start = i * hop
        end = start + hop
        if end > len(nr_output):
            break

        out = res.process(
            error_hop=nr_output[start:end],
            echo_spec=corrected_echo,
            far_power=ctx.far_power,
            far_spec=ctx.far_spec,
            filter_converged=ctx.filter_converged,
            erle_factor=ctx.erle_factor,
            dt_indicator=ctx.dt_indicator,
            near_spec=ctx.near_spec,
            divergence=ctx.divergence,
        )
        output[start:end] = out

    # Copy tail (frames beyond AEC/NR context)
    tail_start = n_frames * hop
    if tail_start < len(nr_output):
        output[tail_start:] = nr_output[tail_start:]

    print(f"  RES: {n_frames} frames processed")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AEC + NR + RES Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Pipeline modes:
  classic   AEC(+RES) -> NR         (RES inside AEC, legacy)
  linear    AEC(linear) -> NR -> RES (separated, recommended)
""")
    parser.add_argument('--mic', required=True, help='Microphone input WAV')
    parser.add_argument('--ref', required=True, help='Reference/loudspeaker WAV')
    parser.add_argument('--output', required=True, help='Output WAV')
    parser.add_argument('--pipeline-mode', default='linear',
                        choices=['classic', 'linear'],
                        help='Pipeline mode (default: linear)')
    parser.add_argument('--aec-mode', default='pbfdkf',
                        choices=['lms', 'nlms', 'fdaf', 'pbfdaf', 'pbfdkf', 'subband'],
                        help='AEC filter mode (default: pbfdkf)')
    parser.add_argument('--preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive', 'maximum'],
                        help='AEC preset (default: balanced)')
    parser.add_argument('--aec-mu', type=float, default=0.3, help='AEC step size')
    parser.add_argument('--nr-gain', type=float, default=-15.0, help='NR min gain (dB)')
    parser.add_argument('--aec-only', action='store_true', help='Run AEC only, skip NR/RES')
    args = parser.parse_args()

    # Load audio
    mic_signal, sr_mic = sf.read(args.mic, dtype='float32')
    ref_signal, sr_ref = sf.read(args.ref, dtype='float32')

    if sr_mic != sr_ref:
        print(f"Error: sample rate mismatch ({sr_mic} vs {sr_ref})", file=sys.stderr)
        sys.exit(1)

    sample_rate = sr_mic
    duration = len(mic_signal) / sample_rate

    # Parse preset
    preset_map = {
        'mild': AecPreset.MILD, 'balanced': AecPreset.BALANCED,
        'aggressive': AecPreset.AGGRESSIVE, 'maximum': AecPreset.MAXIMUM,
    }
    preset = preset_map[args.preset]

    print("AEC + NR + RES Pipeline")
    print("=======================")
    print(f"Input:    {args.mic} ({len(mic_signal)} samples, {duration:.2f}s)")
    print(f"Ref:      {args.ref}")
    print(f"Output:   {args.output}")
    print(f"Rate:     {sample_rate} Hz")
    print(f"Pipeline: {args.pipeline_mode}")
    print(f"AEC:      mode={args.aec_mode}, preset={args.preset}, mu={args.aec_mu}")
    if not args.aec_only:
        print(f"NR:       g_min={args.nr_gain} dB")
    else:
        print("NR/RES:   disabled (--aec-only)")
    print()

    # Build AEC config from preset
    aec_config = AecConfig.from_preset(
        preset,
        sample_rate=sample_rate,
        mode=parse_aec_mode(args.aec_mode),
        mu=args.aec_mu,
        enable_dtd=True,
        enable_res=True,  # will be overridden for linear mode
    )

    if args.pipeline_mode == 'linear':
        # ---- Linear pipeline: AEC(linear) → NR → RES ----
        print("Stage 1: AEC (linear, no RES)...")
        aec_output, contexts = run_aec_linear(mic_signal, ref_signal, aec_config)

        if args.aec_only:
            final_output = aec_output
        else:
            print("Stage 2: NR (MMSE-LSA, 10ms hop)...")
            nr_output, nr_gains = run_nr(aec_output, sample_rate,
                                         g_min_db=args.nr_gain,
                                         return_gain=True)

            print("Stage 3: RES (NR-gain corrected)...")
            final_output = run_res(nr_output, nr_gains, contexts, aec_config)

    else:
        # ---- Classic pipeline: AEC(+RES) → NR ----
        print("Stage 1: AEC (+RES)...")
        aec_output = run_aec_classic(mic_signal, ref_signal, aec_config)

        if args.aec_only:
            final_output = aec_output
        else:
            print("Stage 2: NR (MMSE-LSA, 10ms hop)...")
            final_output = run_nr(aec_output, sample_rate,
                                  g_min_db=args.nr_gain)

    # Save
    sf.write(args.output, final_output, sample_rate)
    print(f"\nDone! Output: {args.output} ({len(final_output)} samples)")


if __name__ == '__main__':
    main()
