"""
AEC -> NR Processing Pipeline
聲學回聲消除後接降噪處理的串接管線

Usage:
    cd Audio_ALG
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output output.wav
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output output.wav --aec-only
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output output.wav --aec-mode subband
"""

import sys
import os
import argparse
import numpy as np
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.aec.python.aec import AEC, AecConfig, AecMode
from lib.nr.denoisers import MmseLsaDenoiser


def parse_aec_mode(mode_str: str) -> AecMode:
    modes = {'lms': AecMode.LMS, 'nlms': AecMode.NLMS,
             'freq': AecMode.FREQ, 'subband': AecMode.SUBBAND}
    return modes.get(mode_str, AecMode.SUBBAND)


def run_aec(mic_signal: np.ndarray, ref_signal: np.ndarray,
            config: AecConfig) -> np.ndarray:
    """Run AEC on the full signal, block by block."""
    aec = AEC(config)
    hop = aec.hop_size

    # Ensure same length
    min_len = min(len(mic_signal), len(ref_signal))
    mic_signal = mic_signal[:min_len]
    ref_signal = ref_signal[:min_len]

    output = np.zeros(min_len, dtype=np.float32)

    for i in range(0, min_len - hop + 1, hop):
        mic_block = mic_signal[i:i + hop]
        ref_block = ref_signal[i:i + hop]
        output[i:i + hop] = aec.process(mic_block, ref_block)

    print(f"  AEC ERLE: {aec.get_erle():.1f} dB")
    return output


def run_nr(signal: np.ndarray, sample_rate: int, g_min_db: float = -15.0,
           alpha_g: float = 0.88) -> np.ndarray:
    """Run NR (MMSE-LSA) on the full signal."""
    denoiser = MmseLsaDenoiser(
        sample_rate=sample_rate,
        noise_method='mcra',
        g_min_db=g_min_db,
        alpha_g=alpha_g,
        alpha_xi=0.88,
        q=0.5,
        xi_min_db=-20.0,
        alpha_s=0.95,
        L=32,
        delta_db=10.0,
        num_init_frames=20,
        scene_change_threshold_db=10.0,
        scene_change_min_frames=5,
        scene_change_blend=0.5,
    )
    return denoiser.denoise(signal)


def main():
    parser = argparse.ArgumentParser(description='AEC -> NR Pipeline')
    parser.add_argument('--mic', required=True, help='Microphone input WAV')
    parser.add_argument('--ref', required=True, help='Reference/loudspeaker WAV')
    parser.add_argument('--output', required=True, help='Output WAV')
    parser.add_argument('--aec-mode', default='subband',
                        choices=['lms', 'nlms', 'freq', 'subband'],
                        help='AEC filter mode (default: subband)')
    parser.add_argument('--aec-mu', type=float, default=0.3, help='AEC step size')
    parser.add_argument('--nr-gain', type=float, default=-15.0, help='NR min gain (dB)')
    parser.add_argument('--aec-only', action='store_true', help='Run AEC only, skip NR')
    args = parser.parse_args()

    # Load audio
    mic_signal, sr_mic = sf.read(args.mic, dtype='float32')
    ref_signal, sr_ref = sf.read(args.ref, dtype='float32')

    if sr_mic != sr_ref:
        print(f"Error: sample rate mismatch ({sr_mic} vs {sr_ref})", file=sys.stderr)
        sys.exit(1)

    sample_rate = sr_mic
    duration = len(mic_signal) / sample_rate

    print("AEC + NR Pipeline")
    print("=================")
    print(f"Input:  {args.mic} ({len(mic_signal)} samples, {duration:.2f}s)")
    print(f"Ref:    {args.ref}")
    print(f"Output: {args.output}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"\nAEC:  mode={args.aec_mode}, mu={args.aec_mu}")
    if not args.aec_only:
        print(f"NR:   g_min={args.nr_gain} dB")
    else:
        print("NR:   disabled (--aec-only)")
    print()

    # Stage 1: AEC
    print("Stage 1: AEC...")
    aec_config = AecConfig(
        sample_rate=sample_rate,
        mode=parse_aec_mode(args.aec_mode),
        mu=args.aec_mu,
        enable_dtd=True,
    )
    aec_output = run_aec(mic_signal, ref_signal, aec_config)

    # Stage 2: NR
    if not args.aec_only:
        print("Stage 2: NR (MMSE-LSA)...")
        final_output = run_nr(aec_output, sample_rate, g_min_db=args.nr_gain)
    else:
        final_output = aec_output

    # Save
    sf.write(args.output, final_output, sample_rate)
    print(f"\nDone! Output: {args.output} ({len(final_output)} samples)")


if __name__ == '__main__':
    main()
