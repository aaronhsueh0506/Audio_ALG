"""Measure the steady-state feature distribution and emit calibrated init values.

WHY THIS EXISTS
---------------
The four normaliser init constants in ``config.ini`` --
``erb_norm_init_lo_db`` / ``erb_norm_init_hi_db`` / ``spec_norm_init_lo`` /
``spec_norm_init_hi`` -- were imported verbatim from DeepFilterNet's libDF.
They are calibrated for libDF's analysis chain, which aggregates each ERB band
as an energy MEAN (``k = 1/band_size``, rectangular and non-overlapping).  This
port instead uses a triangular, overlapping, partition-of-unity filterbank whose
band value is a weighted SUM, so its band energy grows with band width -- by
0 dB at the narrowest bands and over 13 dB at the widest.  The imported
constants therefore start the EMA in the wrong place, by a band-dependent
amount.

Normally that would wash out: ``band_mean_norm_erb`` keeps a per-band running
mean, so a constant offset is absorbed once the EMA converges.  It does not wash
out here.  With ``tau = 1 s`` against 3-second training segments, 3*tau equals
the whole segment: the normaliser is still converging when the example ends, so
every frame the model ever sees is in the init-dominated transient.

Since the filterbank is fixed (it is correct as designed) and the segment length
is fixed (the corpus is already generated), calibrating the init values is the
only remaining lever -- and if ``init`` equals the steady state, there is no
transient at all regardless of tau.

METHOD
------
``init_lo``/``init_hi`` are the two ends of a ``linspace`` across FREQUENCY, so
they are fitted along that axis: take the per-band (per-bin) MEAN -- the
quantity the EMA actually converges to -- then least-squares a line through it
against the band index and read off both ends.  The RMS residual is printed so
you can see how much of the profile a 2-parameter ramp simply cannot express.

Measured on the training split only; calibrating on held-out clips would bake
the validation set into a constant the model starts from.

⚠ These values are specific to THIS project's grid and filterbank.  The
DeepFilterNet2 directory runs a different sample rate, FFT size, band count and
ERB matrix, so it needs its own calibration -- do not copy numbers between them.

USAGE
-----
    python calibrate_norm_init.py --config config.ini --packed-dir /path/to/packed.pt

Run it on the machine that holds the packed corpus, against the SAME corpus the
models will train on.  It prints the measured distribution and the config lines
to paste back.  It does not modify anything.
"""

import argparse
import configparser
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (  # noqa: E402
    compute_erb_matrix,
    erb_bandborder,
    read_feature_config,
    stft,
)
from dataset_gen import (  # noqa: E402
    load_packed_dataset,
    locality_preserving_random_split,
    split_sizes,
)


# torch.quantile rejects inputs larger than ~16.7M elements, and the default
# --clips 2000 produces roughly 46M, so the flat quantiles have to be taken by
# sorting instead.  (The per-band table below stays on torch.quantile: its
# per-column inputs are 22x smaller and comfortably under the limit.)
def _quantile(x, q):
    flat = x.reshape(-1)
    if flat.numel() <= 16_000_000:
        return torch.quantile(flat, q).item()
    return flat.sort().values[min(int(q * flat.numel()), flat.numel() - 1)].item()


def _fit_ramp(x, axis_name, unit):
    """Endpoints of the least-squares line through the per-column MEAN.

    ``init_lo``/``init_hi`` are the two ends of a ``torch.linspace`` ramp ACROSS
    FREQUENCY (train.py's normalize_log_erb / normalize_complex_spectrum), so
    they have to be fitted along that axis.  This script used to pool every
    band and every frame into one distribution and report its median and 5th
    percentile -- which measures how levels vary over TIME and LEVEL, not over
    frequency, and produces two numbers that describe no ramp at all.

    The statistic is the mean because that is what the EMA converges to:
    ``mean = alpha*mean + (1-alpha)*x`` has expectation E[x], not its median.

    Returns (value_at_first_column, value_at_last_column, rms_residual).  The
    residual is the part of the frequency profile a 2-parameter ramp cannot
    represent; report it so the reader can judge whether the ramp is enough.
    """
    per_col = x.mean(dim=0).double()                  # (n_cols,)
    n = per_col.numel()
    i = torch.arange(n, dtype=torch.float64)
    # least squares fit  per_col ~ a + b*i
    i_mean, y_mean = i.mean(), per_col.mean()
    b = ((i - i_mean) * (per_col - y_mean)).sum() / ((i - i_mean).pow(2).sum())
    a = y_mean - b * i_mean
    fitted = a + b * i
    residual = (per_col - fitted).pow(2).mean().sqrt().item()
    print(f"\n--- {axis_name} ramp fit ---")
    print(f"  measured mean spans {per_col[0].item():.4g} .. {per_col[-1].item():.4g}{unit}"
          f"  (min {per_col.min().item():.4g}, max {per_col.max().item():.4g})")
    print(f"  fitted ramp          {fitted[0].item():.4g} .. {fitted[-1].item():.4g}{unit}")
    print(f"  RMS residual vs the fitted line: {residual:.4g}{unit}"
          f"   <- what the 2-parameter ramp cannot express")
    return fitted[0].item(), fitted[-1].item(), residual


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config.ini')
    ap.add_argument('--packed-dir', required=True)
    ap.add_argument('--clips', type=int, default=2000,
                    help='how many clips to measure (default 2000)')
    ap.add_argument('--skip-frames', type=int, default=8,
                    help='leading frames to discard before measuring (default 8). '
                         'This is NOT an EMA warm-up skip: the init value should '
                         'match the typical band level of the signal itself, '
                         'which is what the EMA converges toward, so the raw '
                         'distribution is the right target.  A few frames are '
                         'dropped only because clip onsets are atypical.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(args.config)

    SR = cfg.getint('signal', 'sr')
    N_FFT = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)
    N_BANDS = cfg.getint('signal', 'n_bands')
    MIN_BINS = cfg.getint('signal', 'min_bins_per_band', fallback=2)
    feat = read_feature_config(cfg, SR, HOP_LEN, N_FFT, WIN_LEN)

    borders = erb_bandborder(N_BANDS, SR, N_FFT, MIN_BINS)
    erb_fwd = torch.from_numpy(compute_erb_matrix(borders, N_FFT, mode=0)).float()
    window = torch.sqrt(torch.hann_window(WIN_LEN, periodic=True))

    # Memory of the EMA actually in use.  alpha may be pinned directly, in which
    # case tau is a dead fallback -- reporting 3*tau then understates the warm-up
    # by the ratio between the two (at alpha=0.99 vs tau=1s/hop 256: 300 vs 187
    # frames).  Derive it from alpha, which is what read_feature_config resolved.
    mem_frames = 1.0 / (1.0 - feat['erb_alpha'])
    skip = args.skip_frames

    # Training split ONLY -- calibrating on held-out data leaks the validation
    # set into a constant the model is initialised with.  Same split function,
    # same seed as train.py, so the two agree by construction.
    full = load_packed_dataset(args.packed_dir, expected_sr=SR)
    n_train, n_val = split_sizes(full)
    ds, _ = locality_preserving_random_split(full, n_train, n_val, args.seed)
    print(f"using the training split only: {len(ds)} of {len(full)} clips "
          f"(seed {args.seed})")
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:min(args.clips, len(ds))].tolist()

    seg_frames = None
    erb_db, spec_mag = [], []
    with torch.no_grad():
        for i in idx:
            noisy, _ = ds[i]
            wav = noisy.float().unsqueeze(0)
            spec = stft(wav, N_FFT, HOP_LEN, WIN_LEN, window)      # (1, n_bins, T)
            if seg_frames is None:
                seg_frames = spec.shape[-1]
            power = spec.real.pow(2) + spec.imag.pow(2)            # (1, n_bins, T)
            band = power.transpose(1, 2).matmul(erb_fwd)           # (1, T, n_bands)
            erb_db.append(10.0 * torch.log10(band.clamp_min(1e-16))[0, skip:])
            n_spec = feat['spec_bins']
            spec_mag.append(spec[0, :n_spec, skip:].abs().transpose(0, 1))

    if not erb_db or erb_db[0].numel() == 0:
        raise SystemExit(
            f"no frames left after skipping {skip} of {seg_frames} frames; "
            f"lower --skip-frames"
        )

    erb_db = torch.cat(erb_db, 0)      # (frames, n_bands)
    spec_mag = torch.cat(spec_mag, 0)  # (frames, spec_bins)

    print(f"\nmeasured {len(idx)} clips, {erb_db.shape[0]} frames "
          f"(skipped {skip} warm-up frames of {seg_frames})")
    print(f"grid: sr={SR} n_fft={N_FFT} win={WIN_LEN} hop={HOP_LEN} "
          f"bands={N_BANDS} erb_alpha={feat['erb_alpha']} "
          f"(memory {mem_frames:.0f} frames, 3x = {3 * mem_frames:.0f})")
    if 3 * mem_frames >= seg_frames:
        print(f"  ⚠ 3x memory >= segment length: the EMA never converges inside a "
              f"training example, so these init values are load-bearing for "
              f"EVERY frame, not just the first few.")

    print("\n--- per-band ERB level (dB) ---")
    print(f"{'band':>5}{'mean':>9}{'p05':>9}{'median':>9}{'p95':>9}")
    q = torch.quantile(erb_db, torch.tensor([0.05, 0.5, 0.95]), dim=0)
    means = erb_db.mean(dim=0)
    for b in range(erb_db.shape[1]):
        print(f"{b:>5}{means[b]:>9.1f}{q[0, b]:>9.1f}{q[1, b]:>9.1f}{q[2, b]:>9.1f}")

    erb_lo, erb_hi, erb_res = _fit_ramp(erb_db, 'ERB band', 'dB')
    sp_lo, sp_hi, sp_res = _fit_ramp(spec_mag, 'complex bin', '')

    print("\n--- current vs measured ---")
    print(f"{'key':<26}{'current':>12}{'measured':>12}{'delta':>12}")
    rows = [
        ('erb_norm_init_lo_db', feat['erb_norm_init_lo_db'], erb_lo, 'dB'),
        ('erb_norm_init_hi_db', feat['erb_norm_init_hi_db'], erb_hi, 'dB'),
        ('spec_norm_init_lo', feat['spec_norm_init_lo'], sp_lo, ''),
        ('spec_norm_init_hi', feat['spec_norm_init_hi'], sp_hi, ''),
    ]
    for k, cur, new, unit in rows:
        d = f"{new - cur:+.4g}{unit}" if unit else f"{new / max(cur, 1e-30):.2f}x"
        print(f"{k:<26}{cur:>12.4g}{new:>12.4g}{d:>12}")

    print("\n--- paste into config.ini [feature] ---")
    print(f"erb_norm_init_lo_db = {erb_lo:.1f}")
    print(f"erb_norm_init_hi_db = {erb_hi:.1f}")
    print(f"spec_norm_init_lo = {sp_lo:.6g}")
    print(f"spec_norm_init_hi = {sp_hi:.6g}")
    print("\n⚠ Changing these changes the feature contract: bump FEATURE_VERSION "
          "in config.ini + train.py + process.h together, and mirror the four "
          "constants into process.h, or tests/test_feature_contract.py will fail.")
    print("⚠ The measured ERB levels also answer whether erb_norm_scale_db (=40) "
          "is the right dynamic range here: compare the p05..p95 spread above "
          "against 40 dB.")


if __name__ == '__main__':
    main()
