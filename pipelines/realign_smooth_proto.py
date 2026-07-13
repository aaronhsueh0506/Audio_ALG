#!/usr/bin/env python3
"""Prototype: smooth the AEC delay-relock transient at the NN seam.

WHY (root cause, see 2uGeP_…_farend_singletalk):
  When the far end is silent at stream start, the online matched-filter delay
  estimator cannot lock the echo delay until far-end energy appears. On the
  first far-end onset it adapts at the WRONG alignment (delay=0) for ~140 ms,
  then LOCKS the true delay mid-stream (e.g. 0→448 samples) and re-aligns the
  ring buffer + RESETS the adaptive filter (coarse_learning→idle→startup). The
  echo-subtraction path steps discontinuously → a broadband vertical line in
  the linear residual E(f) (HF spike ~+8 dB above the mic at the reset frame).

WHY NOT just copy AEC3's fix:
  AEC3 routes the RAW MIC `Y` through its suppressor during re-convergence
  (echo_remover.cc:475  `Y_fft = UseLinearFilterOutput() ? E : Y`) because its
  post-filter is an ECHO suppressor that can re-remove echo from Y. Our planned
  back-end is an NR-style NN (RNNoise-ERB / DFN / GTCRN) that does NOT cancel
  echo — feeding it raw mic mid-stream would pass the echo through, and the
  E→Y input-distribution jump would poison the NN's recurrent state. So we keep
  the seam outputting E and remove the transient IN-PLACE, plus expose the AEC
  state as a side-channel the NN can consume.

WHAT this prototype does (default-OFF fix; measure, don't trust scores):
  1. Run the linear AEC, capture the seam spectra E(f) + per-frame side-channel
     features (delay_samples, delay_changed, post_reset_age, usable_linear,
     far_power) — the signals an NN would also receive.
  2. `--smooth`: apply a per-frame raised-cosine gain dip on E across each
     delay-relock window (kills the broadband step; hands the NN a continuous
     signal). Per-frame scalar → identical whether the NN consumes spectra or
     the OLA waveform.
  3. Run a MOCK NR (MMSE-LSA, the stand-in for a stateful/recurrent post-filter)
     on raw-E vs smoothed-E and measure the TAIL: a stateful filter spreads the
     one-frame transient over many frames. The fix should shrink that tail.

Outputs (under --out-prefix, default next to input):
  <p>_E_raw.wav / <p>_E_smooth.wav         — seam waveform before/after
  <p>_NRmock_raw.wav / <p>_NRmock_smooth.wav — after the mock NR
  <p>_seam_fix.png                          — before/after spectrogram
  <p>_sidechannel.csv  (with --csv)         — the per-frame NN feature stream

Usage:
  cd Audio_ALG
  python3 pipelines/realign_smooth_proto.py mic_lpb_2ch.wav --smooth --csv
  python3 pipelines/realign_smooth_proto.py mic.wav lpb.wav --smooth   # 2 separate files
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'aec', 'python'))

from pipelines.aec_nr_pipeline import _build_denoiser            # noqa: E402
from lib.aec.python.aec import AEC, AecConfig, AecMode           # noqa: E402

_N_FREQS = 257
_BS = 320
_HOP = 160
_FFT = 512


# --------------------------------------------------------------------------
# Seam extraction: linear AEC → E(f) spectra + per-frame side-channel features
# --------------------------------------------------------------------------
def extract_seam(mic, lpb, sr, preset='balanced'):
    cfg = AecConfig.from_preset(preset, sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_shadow=True, enable_res=False,
                                enable_cng=False, enable_delay_est=True)
    np.random.seed(0)
    aec = AEC(cfg)
    aec.config.enable_res = False
    aec.config.return_res_context = True
    hop = aec.hop_size
    E, delay, ulin, farp = [], [], [], []
    for i in range(0, len(mic) - hop + 1, hop):
        _, ctx = aec.process(mic[i:i + hop], lpb[i:i + hop])
        E.append(np.asarray(ctx.error_spec, dtype=np.complex64))
        try:
            d = aec.get_stats().delay_samples
        except Exception:                       # delay est not yet producing
            d = delay[-1] if delay else 0
        delay.append(int(d))
        ulin.append(int(aec._diag.get('usable_linear', 0)))
        farp.append(float(ctx.far_power) if ctx.far_power is not None else 0.0)
    E = np.stack(E)
    delay = np.asarray(delay)
    # Side-channel features the NN can consume.
    delay_changed = np.zeros(len(delay), dtype=int)
    delay_changed[1:] = (delay[1:] != delay[:-1]).astype(int)
    post_reset_age = np.full(len(delay), 9999, dtype=int)
    age = 9999
    for i in range(len(delay)):
        age = 0 if delay_changed[i] else (age + 1 if age < 9999 else 9999)
        post_reset_age[i] = age
    return E, dict(delay=delay, delay_changed=delay_changed,
                   post_reset_age=post_reset_age,
                   usable_linear=np.asarray(ulin),
                   far_power=np.asarray(farp))


# --------------------------------------------------------------------------
# The fix: per-frame raised-cosine gain dip across each delay-relock window
# --------------------------------------------------------------------------
def realign_gain(post_reset_age, pre=1, post=8, floor=0.12):
    """Gain = floor at the reset frame, raised-cosine recovery to 1.0 over `post`
    frames (and a 1-frame taper before, offline-only). `pre`=0 → causal."""
    n = len(post_reset_age)
    g = np.ones(n, dtype=np.float32)
    resets = np.where(post_reset_age == 0)[0]
    for r in resets:
        for j in range(r - pre, r + post + 1):
            if 0 <= j < n:
                w = (r - j) / max(pre, 1) if j < r else (j - r) / max(post, 1)
                w = min(max(w, 0.0), 1.0)
                gg = floor + (1.0 - floor) * 0.5 * (1.0 - np.cos(np.pi * w))
                g[j] = min(g[j], gg)
    return g


def ola(spec, n_out):
    idx = np.arange(_BS, dtype=np.float64)
    sw = np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * idx / _BS))).astype(np.float32)
    out = np.zeros(len(spec) * _HOP + _BS, dtype=np.float32)
    acc = np.zeros(_BS, dtype=np.float32)
    for i in range(len(spec)):
        e = np.fft.irfft(spec[i], n=_FFT).astype(np.float32)
        acc += e[:_BS] * sw
        out[i * _HOP:i * _HOP + _HOP] = acc[:_HOP]
        acc[:-_HOP] = acc[_HOP:]
        acc[-_HOP:] = 0.0
    return out[:n_out]


# --------------------------------------------------------------------------
# Mock NR: MMSE-LSA gain on the seam spectra (stand-in for a stateful NN).
# --------------------------------------------------------------------------
def mock_nr_gain(Espec, sr):
    den = _build_denoiser(sr, 'balanced')
    mag = np.abs(Espec).astype(np.float64)
    ph = np.angle(Espec).astype(np.float64)
    _, _, gains = den.denoise_spectrum(mag, ph, return_gain=True)
    return np.asarray(gains, dtype=np.float32)


def hf_prominence(x, t, sr, fr=160, nfft=256, lo=4000, hi=7900):
    k = int(t * sr / fr)
    win = np.hanning(nfft)
    k0, k1 = int(lo / sr * nfft), int(hi / sr * nfft)

    def hf(kk):
        s = x[kk * fr:kk * fr + nfft]
        s = np.pad(s, (0, max(0, nfft - len(s))))[:nfft]
        return 10 * np.log10((np.abs(np.fft.rfft(s * win)) ** 2)[k0:k1].mean() + 1e-12)
    return hf(k) - np.median([hf(kk) for kk in range(k - 6, k + 7)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='2-ch wav (ch0=mic, ch1=lpb) OR mic.wav (then give lpb)')
    ap.add_argument('lpb', nargs='?', default=None, help='lpb.wav if input is mono mic')
    ap.add_argument('--out-prefix', default=None)
    ap.add_argument('--preset', default='balanced',
                    choices=['gentle', 'balanced', 'aggressive'])
    ap.add_argument('--smooth', action='store_true',
                    help='apply the raised-cosine gain dip across delay-relock windows (default OFF)')
    ap.add_argument('--dip-pre', type=int, default=1)
    ap.add_argument('--dip-post', type=int, default=8)
    ap.add_argument('--dip-floor', type=float, default=0.12)
    ap.add_argument('--csv', action='store_true', help='dump the per-frame side-channel feature stream')
    args = ap.parse_args()

    # ---- load mic / lpb ----
    data, sr = sf.read(args.input, dtype='float32')
    if args.lpb is not None:
        mic = data if data.ndim == 1 else data[:, 0]
        lpb, sr2 = sf.read(args.lpb, dtype='float32')
        lpb = lpb if lpb.ndim == 1 else lpb[:, 0]
        assert sr == sr2, 'mic/lpb sample-rate mismatch'
    else:
        if data.ndim != 2 or data.shape[1] < 2:
            sys.exit('ERROR: single-file input must be 2-ch (ch0=mic, ch1=lpb)')
        mic, lpb = np.ascontiguousarray(data[:, 0]), np.ascontiguousarray(data[:, 1])
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]
    prefix = args.out_prefix or os.path.splitext(args.input)[0]

    print(f"Realign-smooth seam prototype  ({n} samp, {n/sr:.2f}s, {sr} Hz, preset={args.preset})")
    print("=" * 70)

    # ---- 1. seam extraction + side-channel ----
    E, sc = extract_seam(mic, lpb, sr, args.preset)
    resets = np.where(sc['delay_changed'] == 1)[0]
    print(f"linear AEC: {len(E)} frames")
    print(f"delay re-lock events (frame, t, delay_samp): "
          f"{[(int(r), round(r*_HOP/sr, 3), int(sc['delay'][r])) for r in resets]}")

    # ---- 2. fix ----
    g = realign_gain(sc['post_reset_age'], args.dip_pre, args.dip_post, args.dip_floor) \
        if args.smooth else np.ones(len(E), dtype=np.float32)
    E_fix = (E * g[:, None]).astype(np.complex64)

    E_raw_t = ola(E, n)
    E_fix_t = ola(E_fix, n)
    sf.write(prefix + '_E_raw.wav', E_raw_t, sr, subtype='FLOAT')
    sf.write(prefix + '_E_smooth.wav', E_fix_t, sr, subtype='FLOAT')

    # ---- 3. mock NR (stateful post-filter stand-in) on raw vs smoothed ----
    g_raw = mock_nr_gain(E, sr)
    g_smo = mock_nr_gain(E_fix, sr)
    NR_raw = ola((E * g_raw).astype(np.complex64), n)
    NR_smo = ola((E_fix * g_smo).astype(np.complex64), n)
    sf.write(prefix + '_NRmock_raw.wav', NR_raw, sr, subtype='FLOAT')
    sf.write(prefix + '_NRmock_smooth.wav', NR_smo, sr, subtype='FLOAT')

    # ---- metrics around each reset ----
    # HF prominence@reset = how far the broadband line stands above its neighbours.
    # Measured at the SEAM (what the NN ingests) AND after the mock NR (what a
    # stateful post-filter passes through). NOTE on this corpus the re-lock
    # coincides with the real far-end onset, so a gain-tail metric is dominated
    # by legitimate signal change — prominence isolates the transient cleanly.
    print("\nMETRICS (HF prominence@reset; lower = line gone):")
    for r in resets:
        t = r * _HOP / sr
        e_raw, e_fix = hf_prominence(E_raw_t, t, sr), hf_prominence(E_fix_t, t, sr)
        nr_raw, nr_fix = hf_prominence(NR_raw, t, sr), hf_prominence(NR_smo, t, sr)
        print(f"  reset @ frame {r} (t={t:.3f}s):")
        print(f"    seam E (NN input)  : raw {e_raw:+5.1f} dB  ->  smoothed {e_fix:+5.1f} dB"
              f"   (Δ {e_fix-e_raw:+.1f} dB)")
        print(f"    after mock NR      : raw {nr_raw:+5.1f} dB  ->  smoothed {nr_fix:+5.1f} dB"
              f"   (Δ {nr_fix-nr_raw:+.1f} dB)")

    # ---- side-channel CSV (the NN feature stream) ----
    if args.csv:
        import csv
        path = prefix + '_sidechannel.csv'
        with open(path, 'w', newline='') as fp:
            w = csv.writer(fp)
            w.writerow(['frame', 'time_s', 'delay_samp', 'delay_changed',
                        'post_reset_age', 'usable_linear', 'far_power', 'realign_gain'])
            for i in range(len(E)):
                w.writerow([i, round(i * _HOP / sr, 4), int(sc['delay'][i]),
                            int(sc['delay_changed'][i]), int(sc['post_reset_age'][i]),
                            int(sc['usable_linear'][i]), f"{sc['far_power'][i]:.6e}",
                            f"{g[i]:.4f}"])
        print(f"\nwrote side-channel feature stream -> {path}")

    # ---- before/after spectrogram ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        t0 = (resets[0] * _HOP / sr) if len(resets) else 1.0
        for ax, x, ti in [(axs[0], E_raw_t, 'seam E RAW (NN input) — delay-relock line'),
                          (axs[1], E_fix_t, 'seam E SMOOTHED (raised-cosine dip)')]:
            ax.specgram(x, NFFT=256, Fs=sr, noverlap=224, cmap='magma', vmin=-115, vmax=-35)
            ax.set_xlim(max(0, t0 - 0.25), t0 + 0.55)
            ax.set_ylabel('Hz')
            ax.set_title(ti, fontsize=9)
            for r in resets:
                ax.axvline(r * _HOP / sr, color='lime', lw=0.7, ls='--')
        axs[1].set_xlabel('time (s)')
        plt.tight_layout()
        plt.savefig(prefix + '_seam_fix.png', dpi=120)
        print(f"saved spectrogram -> {prefix}_seam_fix.png")
    except Exception as e:  # noqa: BLE001
        print(f"[spectrogram skipped] {e}")

    print("\nDone. Compare _E_raw.wav vs _E_smooth.wav (and _NRmock_*) ; "
          "judge by waveform/spectrogram, NOT AECMOS (this is a perceptual transient).")


if __name__ == '__main__':
    main()
