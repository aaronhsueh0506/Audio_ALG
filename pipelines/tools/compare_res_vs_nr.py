#!/usr/bin/env python3
"""Compare AEC+RES vs AEC+NR+RES on ONE multi-channel test file.

A standalone test utility (not part of the production bench). It runs the shared
AEC-linear stage ONCE and then two suppression paths that differ ONLY in the NR
stage, so the A/B is apples-to-apples:

    AEC+RES     : S(f) = E(f) · G_res             (AEC's own AEC3 residual-echo gain; no NR)
    AEC+NR+RES  : S(f) = E(f) · min(G_nr, G_res)  (production far02_near; G_nr is the
                  echo-aware unified gain ξ=S²/(N²+R²) by default — see --no-inject-echo-psd)

both with the same ne_floor / ne_gate, so adding NR is the only change.

Input WAV channels (2- or 3-channel, single file):
    ch0 = mic    — near-end mic (echo + near speech + noise)
    ch1 = echo   — far-end reference / loudspeaker / loopback (lpb)
    ch2 = near   — OPTIONAL near-end CLEAN speech; passed through as a reference
                   output for your own scoring (PESQ/SDR/listening). Not used by
                   the AEC.

Outputs (next to the input, or under --out-prefix):
    <prefix>_aec.wav                  — AEC linear only (no RES, no NR)  [reference]
    <prefix>_aec_res.wav              — AEC + RES
    <prefix>_aec_nr_res.wav           — AEC + NR + RES  (production, ne_floor on)
    <prefix>_aec_nr_res_unmasked.wav  — AEC + NR + RES with ne_floor=0 (NR at FULL
                                        strength; skip with --no-unmasked)
    <prefix>_aec_nr_res_nocng.wav     — AEC + NR + RES with comfort noise OFF
                                        (only with --cng-ab; A/B the CNG fill that
                                        refills the echo-cancelled bins)
    <prefix>_near_clean.wav           — ch2 passthrough (only if 3-channel input)

It also prints an NR-contribution diagnostic (how often / how hard G_nr cuts past
G_res, and how many dB the ne_floor lift claws NR back). With --dnsmos it scores the
outputs with DNSMOS (BAK = the NR noise-cleanup score AECMOS is blind to).

Why NR can look weak: the production gain is min(G_nr, G_res) and a near-end floor
(ne_floor=0.4) lifts the gain back toward 1.0 in low-echo bins — exactly where NR
works — so NR's -15 dB floor becomes ~-6 dB. The _unmasked output (ne_floor=0) shows
the unclawed NR. Both limits are the shipped A_min_pl operating point, not a bug:
min hands echo bins to RES; ne_floor protects near-end speech (and, unavoidably,
near-end noise too). Compare _aec_res vs _aec_nr_res vs _aec_nr_res_unmasked.

Usage:
    cd Audio_ALG
    python3 pipelines/tools/compare_res_vs_nr.py input_3ch.wav --dnsmos
    python3 pipelines/tools/compare_res_vs_nr.py input.wav --out-prefix /tmp/cmp --preset balanced
    python3 pipelines/tools/compare_res_vs_nr.py input.wav --ne-floor 0.4 --ne-gate both --nr-preset balanced
    python3 pipelines/tools/compare_res_vs_nr.py input.wav --cng-ab --dnsmos   # A/B comfort noise on vs off

Note: uses whatever AEC is checked out in lib/aec. To test the v3.24.0 round-robin
AEC, make sure lib/aec is at the static-memory tip (495566d) / your merged branch.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

# Match aec_nr_pipeline's import wiring (lib.* + pipelines.* + the AEC package).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'aec', 'python'))

from pipelines.aec_nr_pipeline import (                       # noqa: E402
    run_aec_linear, run_nr_spectrum, run_res,
    PROD_INJECT_ECHO_PSD, PROD_NE_FLOOR_FAR_ACTIVE,
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


def _nr_contribution(contexts, nr_gains):
    """Gain-domain NR-vs-RES stats: where and how hard G_nr cuts past G_res.

    Returns None if the AEC was built without the freq seam (no res_gain). The
    final gain is min(G_nr, G_res), so NR only 'bites' where G_nr < G_res. We
    also scope NR's territory to low-echo bins (echo_frac<0.1) — mirroring
    run_res's echo_frac = (r2/psd) / |E|^2 — since that is where NR does its
    stationary-noise reduction (echo bins are owned by G_res).
    """
    psd_scale = 32768.0 ** 2
    res_list, e2_list, r2_list = [], [], []
    for c in contexts:
        if c.res_gain is None or c.error_spec is None:
            break
        res_list.append(np.asarray(c.res_gain, dtype=np.float32))
        e2_list.append(np.abs(np.asarray(c.error_spec)).astype(np.float32) ** 2)
        r2_list.append(np.asarray(c.r2, dtype=np.float32) if c.r2 is not None
                       else np.zeros(_N_FREQS, dtype=np.float32))
    n = len(res_list)
    if n == 0:
        return None
    res_g = np.stack(res_list)
    g_nr = np.asarray(nr_gains[:n], dtype=np.float32)
    e2 = np.stack(e2_list)
    r2 = np.stack(r2_list)
    echo_frac = np.clip((r2 / psd_scale) / (e2 + 1e-12), 0.0, 1.0)

    bite = g_nr < res_g
    extra_db = 20.0 * np.log10((g_nr + 1e-12) / (res_g + 1e-12))   # <0 where bites
    low = echo_frac < 0.1
    bite_low = bite & low

    def _mean(mask, vals):
        return float(vals[mask].mean()) if mask.any() else 0.0

    return dict(
        n_frames=n,
        frac_bite=float(bite.mean()),
        extra_db=_mean(bite, extra_db),
        frac_low=float(low.mean()),
        frac_bite_low=(float(bite_low.sum()) / float(low.sum())) if low.any() else 0.0,
        extra_db_low=_mean(bite_low, extra_db),
    )


def _run_dnsmos(named_signals, sr):
    """Score each (label, signal) with DNSMOS, reusing bench_dnsmos's scorer.

    DNSMOS is NOT level-invariant, so every signal is normalized to -26 dBFS RMS
    first (same as bench_dnsmos). BAK is the background-noise score AECMOS can't
    see — the NR's real metric. Degrades gracefully if speechmos is unavailable.
    """
    try:
        import speechmos.dnsmos as dnsmos   # the library API; branch-independent
    except Exception as e:  # noqa: BLE001
        print(f"\n[DNSMOS skipped] {type(e).__name__}: {e}")
        print("  (pip install speechmos / onnxruntime, or drop --dnsmos)")
        return
    print("\nDNSMOS (level-normalized to -26 dBFS; BAK = NR's AECMOS-blind score):")
    print(f"  {'output':24s} {'SIG':>6} {'BAK':>6} {'OVRL':>6}")
    scores = {}
    for label, sig in named_signals:
        a = np.clip(np.asarray(sig, dtype=np.float32), -1.0, 1.0)
        rms = float(np.sqrt(np.mean(a ** 2)))
        if rms > 1e-6:
            a = np.clip(a * (10.0 ** (-26.0 / 20.0) / rms), -1.0, 1.0)
        try:
            r = dnsmos.run(a, sr, return_df=False)
            scores[label] = r
            print(f"  {label:24s} {r['sig_mos']:6.3f} {r['bak_mos']:6.3f} {r['ovrl_mos']:6.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:24s} FAIL: {e}")
    if 'AEC+RES' in scores and 'AEC+NR+RES' in scores:
        dbak = scores['AEC+NR+RES']['bak_mos'] - scores['AEC+RES']['bak_mos']
        print(f"  ΔBAK (AEC+NR+RES − AEC+RES) = {dbak:+.3f}   (NR's noise-cleanup value)")


def main():
    ap = argparse.ArgumentParser(
        description='Compare AEC+RES vs AEC+NR+RES on one 2/3-ch test file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='channels: ch0=mic, ch1=echo(ref), ch2=near-clean (optional)')
    ap.add_argument('input', help='multi-channel WAV: ch0=mic, ch1=echo, [ch2=near-clean]')
    ap.add_argument('--out-prefix', default=None,
                    help='output path prefix (default: input path without extension)')
    ap.add_argument('--preset', default='balanced',
                    choices=['mild', 'balanced', 'aggressive'],
                    help='AEC residual-echo preset (mild/balanced/aggressive)')
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
    ap.add_argument('--ne-floor-far-active', type=float, default=PROD_NE_FLOOR_FAR_ACTIVE,
                    help='near-end floor when the far end is ACTIVE (FS/DT); '
                         'production far02_near = 0.2 (applied to the +NR path)')
    inj = ap.add_mutually_exclusive_group()
    inj.add_argument('--inject-echo-psd', dest='inject_echo_psd', action='store_true',
                     default=PROD_INJECT_ECHO_PSD,
                     help='echo-aware NR: fold residual-echo R² into the NR floor '
                          '(ξ=S²/(N²+R²), production far02_near; default ON)')
    inj.add_argument('--no-inject-echo-psd', dest='inject_echo_psd', action='store_false',
                     help='plain noise-only NR (old A_min_pl) for A/B vs the unified gain')
    ap.add_argument('--no-cng', action='store_true',
                    help='disable comfort noise on ALL outputs (global)')
    ap.add_argument('--cng-ab', action='store_true',
                    help='also write _aec_nr_res_nocng.wav = the production output with '
                         'comfort noise OFF, so you can A/B the CNG fill on the '
                         'echo-cancelled bins (run WITHOUT --no-cng so the main output '
                         'keeps CNG to compare against)')
    ap.add_argument('--no-unmasked', action='store_true',
                    help='skip the ne_floor=0 NR-unmasked output')
    ap.add_argument('--dnsmos', action='store_true',
                    help='score outputs with DNSMOS (BAK = NR noise-cleanup score '
                         'AECMOS is blind to; needs speechmos+onnxruntime)')
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
    # The CNG A/B needs comfort_noise present in the contexts even when the main
    # output is CNG-off, so force the AEC stage to estimate it whenever --cng-ab is
    # set (harmless: enable_res=False, so this only exposes ctx.comfort_noise and
    # does not touch the linear output or any other context field).
    aec_enable_cng = enable_cng or args.cng_ab

    print("AEC+RES vs AEC+NR+RES comparison")
    print("================================")
    print(f"Input:   {args.input}  ({len(mic)} samples, {len(mic)/sr:.2f}s, {sr} Hz, {n_ch}ch)")
    print(f"Preset:  aec={args.preset} nr={args.nr_preset}   "
          f"ne_floor={args.ne_floor}/{args.ne_floor_far_active}(far-active) gate={args.ne_gate} "
          f"combine={args.combine} inject_echo_psd={args.inject_echo_psd} cng={enable_cng}"
          f"{' (+CNG A/B)' if args.cng_ab else ''}")
    print(f"near-clean ref: {'present (ch2)' if near_clean is not None else 'absent'}")
    print()

    # ---- Shared AEC linear stage (run once) ----
    print("Stage 1: AEC (linear, no RES)  — shared by both paths...")
    np.random.seed(0)
    aec_linear, contexts = run_aec_linear(mic, ref, _build_cfg(sr, args.preset, True, aec_enable_cng))
    cfg_res = _build_cfg(sr, args.preset, False, enable_cng)

    # ---- Path A: AEC + RES (no NR) ----
    print("Stage 2a: AEC+RES  (g_total = G_res, use_nr=False)...")
    dummy_gains = np.ones((len(contexts), _N_FREQS), dtype=np.float32)
    out_aec_res = run_res(
        np.zeros(len(mic), dtype=np.float32), dummy_gains, contexts, cfg_res,
        use_nr=False, use_res=True,
        ne_floor=args.ne_floor, ne_gate=args.ne_gate)

    # ---- Path B: AEC + NR + RES (production) ----
    tag = ' [echo-aware ξ=S²/(N²+R²)]' if args.inject_echo_psd else ' [plain noise-only NR]'
    print(f"Stage 2b: AEC+NR+RES  (g_total = min(G_nr, G_res)){tag}...")
    nr_gains = run_nr_spectrum(contexts, sr, nr_preset=args.nr_preset,
                               inject_echo_psd=args.inject_echo_psd)
    out_aec_nr_res = run_res(
        np.zeros(len(mic), dtype=np.float32), nr_gains, contexts, cfg_res,
        use_nr=True, use_res=True, combine=args.combine,
        ne_floor=args.ne_floor, ne_gate=args.ne_gate,
        ne_floor_far_active=args.ne_floor_far_active)

    # ---- Path B' : same, but ne_floor=0 → NR at full strength (unmasked) ----
    out_unmasked = None
    if not args.no_unmasked:
        print("Stage 2c: AEC+NR+RES UNMASKED  (ne_floor=0 → NR at full strength)...")
        out_unmasked = run_res(
            np.zeros(len(mic), dtype=np.float32), nr_gains, contexts, cfg_res,
            use_nr=True, use_res=True, combine=args.combine,
            ne_floor=0.0, ne_gate=args.ne_gate, ne_floor_far_active=None)

    # ---- Path B'' : production B but comfort noise OFF (CNG A/B) ----
    # Same gains/ne_floor as the main output — the ONLY difference is enable_cng,
    # so _aec_nr_res.wav (CNG on) vs _aec_nr_res_nocng.wav isolates the comfort
    # noise that refills the echo-cancelled bins (noise_gain = sqrt(1 - g_res²)).
    out_nocng = None
    if args.cng_ab:
        print("Stage 2d: AEC+NR+RES, comfort noise OFF  (CNG A/B vs the main output)...")
        cfg_res_nocng = _build_cfg(sr, args.preset, False, False)
        out_nocng = run_res(
            np.zeros(len(mic), dtype=np.float32), nr_gains, contexts, cfg_res_nocng,
            use_nr=True, use_res=True, combine=args.combine,
            ne_floor=args.ne_floor, ne_gate=args.ne_gate,
            ne_floor_far_active=args.ne_floor_far_active)

    # ---- Write outputs ----
    def _rms(x):
        return float(np.sqrt(np.mean(np.square(x))) + 1e-20)

    outs = [
        ('_aec.wav', aec_linear, 'AEC linear'),
        ('_aec_res.wav', out_aec_res, 'AEC+RES'),
        ('_aec_nr_res.wav', out_aec_nr_res, 'AEC+NR+RES'),
    ]
    if out_unmasked is not None:
        outs.append(('_aec_nr_res_unmasked.wav', out_unmasked, 'AEC+NR+RES (ne_floor=0)'))
    if out_nocng is not None:
        outs.append(('_aec_nr_res_nocng.wav', out_nocng, 'AEC+NR+RES (CNG off)'))
    if near_clean is not None:
        outs.append(('_near_clean.wav', near_clean[:len(mic)], 'near-clean (ref passthrough)'))

    print("\nResults (RMS levels for a quick sanity read; do AECMOS/PESQ for quality):")
    print(f"  {'mic (input)':30s} rms={_rms(mic):.5f}")
    for suffix, sig, label in outs:
        path = prefix + suffix
        sf.write(path, sig[:len(mic)].astype(np.float32), sr, subtype='FLOAT')
        print(f"  {label:30s} rms={_rms(sig):.5f}   -> {path}")

    # ---- NR-contribution diagnostic (why NR may look weak) ----
    diag = _nr_contribution(contexts, nr_gains)
    print("\nNR contribution (why NR may look weak):")
    if diag is None:
        print("  (AEC built without the freq seam — no res_gain; cannot compute gain stats)")
    else:
        print(f"  G_nr bites (G_nr<G_res):      {diag['frac_bite']*100:5.1f}% of all (frame,bin), "
              f"avg {diag['extra_db']:+.1f} dB extra cut where it bites")
        print(f"  in NR's territory (low-echo): {diag['frac_bite_low']*100:5.1f}% of low-echo bins bitten, "
              f"avg {diag['extra_db_low']:+.1f} dB  (low-echo = {diag['frac_low']*100:.0f}% of all bins)")
    net_db = 20.0 * np.log10(_rms(out_aec_nr_res) / _rms(out_aec_res))
    print(f"  net NR effect (full-signal):  {net_db:+.2f} dB   (AEC+NR+RES vs AEC+RES)")
    if out_unmasked is not None:
        claw_db = 20.0 * np.log10(_rms(out_aec_nr_res) / _rms(out_unmasked))
        print(f"  ne_floor={args.ne_floor:g} claw-back:       {claw_db:+.2f} dB  "
              f"(AEC+NR+RES louder than ne_floor=0 → NR suppression the floor lifted back)")
    if out_nocng is not None:
        if enable_cng:
            n = min(len(out_aec_nr_res), len(out_nocng))
            diff = out_aec_nr_res[:n] - out_nocng[:n]      # = the injected comfort noise
            cng_rms = _rms(diff)
            sig_rms = _rms(out_nocng[:n])
            # CNG energy RELATIVE to the suppressed signal (the audible floor level).
            rel_db = 20.0 * np.log10(cng_rms / sig_rms)
            print(f"  CNG fill (comfort-noise energy): RMS={cng_rms:.5f} ({rel_db:+.1f} dB "
                  f"vs signal), peak |diff|={float(np.max(np.abs(diff))):.5f}")
            print(f"     → noise_gain=sqrt(1-g_res²) fills the echo-cancelled bins at the "
                  f"AEC background level; near-end / NR-only bins (g_res≈1) get ~nothing")
        else:
            print("  CNG fill: main output has --no-cng (both files CNG-off) → "
                  "drop --no-cng so _aec_nr_res.wav keeps CNG to compare against")

    # ---- Optional DNSMOS BAK A/B ----
    if args.dnsmos:
        named = [('AEC linear', aec_linear), ('AEC+RES', out_aec_res),
                 ('AEC+NR+RES', out_aec_nr_res)]
        if out_unmasked is not None:
            named.append(('AEC+NR+RES unmasked', out_unmasked))
        if out_nocng is not None:
            named.append(('AEC+NR+RES CNG off', out_nocng))
        _run_dnsmos(named, sr)

    print("\nDone. Compare _aec_res.wav vs _aec_nr_res.wav (ne_floor) vs "
          "_aec_nr_res_unmasked.wav (ne_floor=0).")


if __name__ == '__main__':
    main()
