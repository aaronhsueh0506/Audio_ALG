#!/usr/bin/env python3
"""Compare AEC+RES vs AEC+NR+RES on ONE multi-channel test file.

A standalone test utility (not part of the production bench). It runs the shared
AEC-linear stage ONCE and then two suppression paths that differ ONLY in the NR
stage, so the A/B is apples-to-apples:

    AEC+RES     : S(f) = E(f) · G_res             (AEC's own AEC3 residual-echo gain; no NR)
    AEC+NR+RES  : S(f) = E(f) · min(G_nr, G_res)  (production A_min_pl freq pipeline)

both with the same ne_floor / ne_gate, so adding NR is the only change.

Input WAV channels (2- or 3-channel, single file):
    ch0 = mic    — near-end mic (echo + near speech + noise)
    ch1 = echo   — far-end reference / loudspeaker / loopback (lpb)
    ch2 = near   — OPTIONAL near-end CLEAN speech; passed through as a reference
                   output for your own scoring (PESQ/SDR/listening). Not used by
                   the AEC.

Outputs (next to the input, or under --out-prefix):
    <prefix>_aec.wav         — AEC linear only (no RES, no NR)  [reference]
    <prefix>_aec_res.wav     — AEC + RES
    <prefix>_aec_nr_res.wav  — AEC + NR + RES
    <prefix>_near_clean.wav  — ch2 passthrough (only if 3-channel input)

Usage:
    cd Audio_ALG
    python3 pipelines/compare_res_vs_nr.py input_3ch.wav
    python3 pipelines/compare_res_vs_nr.py input.wav --out-prefix /tmp/cmp --preset balanced
    python3 pipelines/compare_res_vs_nr.py input.wav --ne-floor 0.4 --ne-gate both --nr-preset balanced

Note: uses whatever AEC is checked out in lib/aec. To test the v3.24.0 round-robin
AEC, make sure lib/aec is at the static-memory tip (495566d) / your merged branch.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

# Match aec_nr_pipeline's import wiring (lib.* + pipelines.* + the AEC package).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'aec', 'python'))

from pipelines.aec_nr_pipeline import (                       # noqa: E402
    run_aec_linear, run_nr_spectrum, run_res,
)
from lib.aec.python.aec import AecConfig, AecMode, AecPreset  # noqa: E402

_N_FREQS = 257   # fft=512 → 512//2 + 1 (the AEC/NR/RES shared grid)


def _build_cfg(sample_rate, preset, enable_res, enable_cng):
    """Production-shaped AEC config (PBFDKF + shadow, online delay-est, no pre-align).

    filter_length is left to auto (52 ms → 832 @ 16 kHz, the validated default).
    enable_res only flags the AEC's own RES path; run_aec_linear forces it off and
    turns on return_res_context regardless — we pass a True cfg to the AEC builder
    and a False cfg to run_res, mirroring rebench_joint.
    """
    return AecConfig.from_preset(
        preset, sample_rate=sample_rate, mode=AecMode.PBFDKF,
        enable_shadow=True, enable_res=enable_res, enable_cng=enable_cng,
        enable_delay_est=True,   # online matched-filter align (production no-PA)
    )


def main():
    ap = argparse.ArgumentParser(
        description='Compare AEC+RES vs AEC+NR+RES on one 2/3-ch test file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='channels: ch0=mic, ch1=echo(ref), ch2=near-clean (optional)')
    ap.add_argument('input', help='multi-channel WAV: ch0=mic, ch1=echo, [ch2=near-clean]')
    ap.add_argument('--out-prefix', default=None,
                    help='output path prefix (default: input path without extension)')
    ap.add_argument('--preset', default='balanced',
                    choices=['gentle', 'balanced', 'aggressive'],
                    help='AEC residual-echo preset (gentle/balanced/aggressive)')
    ap.add_argument('--nr-preset', default='balanced',
                    choices=['mild', 'balanced', 'aggressive'],
                    help='NR strength preset (mild/balanced/aggressive; '
                         'g_min = -10/-15/-20 dB). Only affects the +NR path.')
    ap.add_argument('--ne-floor', type=float, default=0.4,
                    help='near-end preservation floor strength, applied to BOTH paths (0=off)')
    ap.add_argument('--ne-gate', default='both',
                    choices=['r2', 'resgain', 'both', 'both_sharp'])
    ap.add_argument('--combine', default='min', choices=['min', 'product'],
                    help='G_nr/G_res combine for the +NR path (default: min = A_min_pl)')
    ap.add_argument('--no-cng', action='store_true', help='disable comfort noise')
    args = ap.parse_args()

    data, sr = sf.read(args.input, dtype='float32')
    if data.ndim != 2 or data.shape[1] < 2:
        sys.exit(f"ERROR: input must be 2- or 3-channel "
                 f"(ch0=mic, ch1=echo[, ch2=near]); got shape {data.shape}")
    n_ch = data.shape[1]
    if n_ch not in (2, 3):
        sys.exit(f"ERROR: expected 2 or 3 channels, got {n_ch}")

    mic = np.ascontiguousarray(data[:, 0])
    ref = np.ascontiguousarray(data[:, 1])
    near_clean = np.ascontiguousarray(data[:, 2]) if n_ch == 3 else None

    prefix = args.out_prefix or os.path.splitext(args.input)[0]
    enable_cng = not args.no_cng

    print("AEC+RES vs AEC+NR+RES comparison")
    print("================================")
    print(f"Input:   {args.input}  ({len(mic)} samples, {len(mic)/sr:.2f}s, {sr} Hz, {n_ch}ch)")
    print(f"Preset:  aec={args.preset} nr={args.nr_preset}   "
          f"ne_floor={args.ne_floor} gate={args.ne_gate} "
          f"combine={args.combine} cng={enable_cng}")
    print(f"near-clean ref: {'present (ch2)' if near_clean is not None else 'absent'}")
    print()

    # ---- Shared AEC linear stage (run once) ----
    print("Stage 1: AEC (linear, no RES)  — shared by both paths...")
    np.random.seed(0)
    aec_linear, contexts = run_aec_linear(mic, ref, _build_cfg(sr, args.preset, True, enable_cng))
    cfg_res = _build_cfg(sr, args.preset, False, enable_cng)

    # ---- Path A: AEC + RES (no NR) ----
    print("Stage 2a: AEC+RES  (g_total = G_res, use_nr=False)...")
    dummy_gains = np.ones((len(contexts), _N_FREQS), dtype=np.float32)
    out_aec_res = run_res(
        np.zeros(len(mic), dtype=np.float32), dummy_gains, contexts, cfg_res,
        use_nr=False, use_res=True,
        ne_floor=args.ne_floor, ne_gate=args.ne_gate)

    # ---- Path B: AEC + NR + RES (production) ----
    print("Stage 2b: AEC+NR+RES  (g_total = min(G_nr, G_res))...")
    nr_gains = run_nr_spectrum(contexts, sr, nr_preset=args.nr_preset)
    out_aec_nr_res = run_res(
        np.zeros(len(mic), dtype=np.float32), nr_gains, contexts, cfg_res,
        use_nr=True, use_res=True, combine=args.combine,
        ne_floor=args.ne_floor, ne_gate=args.ne_gate)

    # ---- Write outputs ----
    def _rms(x):
        return float(np.sqrt(np.mean(np.square(x))) + 1e-20)

    outs = [
        ('_aec.wav', aec_linear, 'AEC linear'),
        ('_aec_res.wav', out_aec_res, 'AEC+RES'),
        ('_aec_nr_res.wav', out_aec_nr_res, 'AEC+NR+RES'),
    ]
    if near_clean is not None:
        outs.append(('_near_clean.wav', near_clean[:len(mic)], 'near-clean (ref passthrough)'))

    print("\nResults (RMS levels for a quick sanity read; do AECMOS/PESQ for quality):")
    print(f"  {'mic (input)':28s} rms={_rms(mic):.5f}")
    for suffix, sig, label in outs:
        path = prefix + suffix
        sf.write(path, sig[:len(mic)].astype(np.float32), sr, subtype='FLOAT')
        print(f"  {label:28s} rms={_rms(sig):.5f}   -> {path}")

    print("\nDone. Compare _aec_res.wav vs _aec_nr_res.wav (only the NR stage differs).")


if __name__ == '__main__':
    main()
