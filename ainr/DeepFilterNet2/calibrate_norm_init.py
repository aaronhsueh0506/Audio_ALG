"""Measure this model's steady-state feature distribution and emit init values.

WHY THIS EXISTS
---------------
``erb_norm_init_lo_db`` / ``erb_norm_init_hi_db`` / ``spec_norm_init_lo`` /
``spec_norm_init_hi`` in ``config.ini`` were imported verbatim from
DeepFilterNet's libDF.  libDF aggregates each ERB band as an energy MEAN over a
rectangular, non-overlapping filterbank (``k = 1/band_size``).  This port uses
the triangular, overlapping, partition-of-unity bank built by ``_build_erb_fb``
-- 482 of its 513 bins are shared between bands, and the per-band weights sum to
2..72 rather than to 1 -- so band energy grows with band width and the imported
constants start the EMA between +3.0 dB and +18.6 dB away from where this
model's features actually sit.

⚠ These numbers are NOT transferable from RNNoise-ERB.  That project runs
16 kHz with a 512-point FFT and 22 bands; this one runs
48 kHz, 1024-point FFT, 32 bands.  Different band widths give a different
offset per band, and the corpora differ too.  Fitting them separately is the
whole point -- run this script here, and run RNNoise-ERB's in that directory.

Why it matters at all: a constant offset would normally wash out, because
``band_mean_norm`` keeps a per-band running mean that converges away from its
initial value.  It does not wash out here.  At tau = 1 s, hop 512, 48 kHz,
3*tau is 281 frames and a 3-second training segment is 281 frames, so the
normaliser is still converging when the example ends and every frame the model
sees lies in the init-dominated transient.

USAGE
-----
    python calibrate_norm_init.py --config config.ini --packed-dir /path/to/48k

Run it on the machine holding the packed corpus, against the SAME corpus the
model will train on.  It prints the measured distribution and the config lines
to paste back.  It modifies nothing.
"""

import argparse
import configparser
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DeepFilterNet2  # noqa: E402
from train import read_feature_config, read_model_config  # noqa: E402
from dataset_gen import (  # noqa: E402
    fit_ramp,
    load_packed_dataset,
    locality_preserving_random_split,
    split_sizes,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config.ini')
    ap.add_argument('--packed-dir', required=True)
    ap.add_argument('--clips', type=int, default=2000,
                    help='how many training clips to measure (default 2000)')
    ap.add_argument('--skip-frames', type=int, default=8,
                    help='leading frames to discard (default 8).  NOT an EMA '
                         'warm-up skip: the init value should match the typical '
                         'band level of the signal itself, which is what the EMA '
                         'converges toward.  A few frames are dropped only '
                         'because clip onsets are atypical.')
    ap.add_argument('--seed', type=int, default=42,
                    help='must match the training seed so the same split is used')
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(args.config)

    SR = cfg.getint('signal', 'sr')
    N_FFT = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)
    feat = read_feature_config(cfg, SR, HOP_LEN)
    model_cfg = read_model_config(cfg)
    df_bins = model_cfg['df_bins']

    # The filterbank is a registered buffer, so take it from the model rather
    # than rebuilding it -- that guarantees the calibration matches the matrix
    # the checkpoint will actually carry.
    model = DeepFilterNet2(**model_cfg).eval()
    erb_fb = model.erb_fb                                   # (n_erb, n_bins)
    n_erb = erb_fb.shape[0]

    # Same analysis the trainer uses: sqrt-Hann, normalized, center=True.
    window = torch.hann_window(WIN_LEN).pow(0.5)

    # Training split only; calibrating on held-out clips would bake the
    # validation set into a constant the model is initialised from.
    full = load_packed_dataset(args.packed_dir, expected_sr=SR)
    n_train, n_val = split_sizes(full)
    ds, _ = locality_preserving_random_split(full, n_train, n_val, args.seed)
    print(f"using the training split only: {len(ds)} of {len(full)} clips "
          f"(seed {args.seed})")

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:min(args.clips, len(ds))].tolist()

    mem_frames = 1.0 / (1.0 - feat['erb_alpha'])
    seg_frames = None
    erb_db, spec_mag = [], []
    skip = args.skip_frames
    with torch.no_grad():
        for i in idx:
            noisy, _ = ds[i]
            spec = torch.stft(noisy.float().unsqueeze(0), N_FFT, HOP_LEN, WIN_LEN,
                              window=window, return_complex=True, normalized=True)
            if seg_frames is None:
                seg_frames = spec.shape[-1]
            power = spec.real.pow(2) + spec.imag.pow(2)          # (1, n_bins, T)
            band = power.transpose(1, 2).matmul(erb_fb.T)        # (1, T, n_erb)
            erb_db.append(10.0 * torch.log10(band.clamp_min(1e-16))[0, skip:])
            spec_mag.append(spec[0, :df_bins, skip:].abs().transpose(0, 1))

    if not erb_db or erb_db[0].numel() == 0:
        raise SystemExit(f"no frames left after skipping {skip} of {seg_frames}; "
                         f"lower --skip-frames")

    erb_db = torch.cat(erb_db, 0)         # (frames, n_erb)
    spec_mag = torch.cat(spec_mag, 0)     # (frames, df_bins)

    print(f"\nmeasured {len(idx)} clips, {erb_db.shape[0]} frames "
          f"(skipped {skip} of {seg_frames} per clip)")
    print(f"grid: sr={SR} n_fft={N_FFT} win={WIN_LEN} hop={HOP_LEN} "
          f"n_erb={n_erb} df_bins={df_bins} erb_alpha={feat['erb_alpha']} "
          f"(memory {mem_frames:.0f} frames, 3x = {3 * mem_frames:.0f})")
    if 3 * mem_frames >= seg_frames:
        print("  ⚠ 3x memory >= segment length: the EMA never converges inside a "
              "training example, so these init values are load-bearing for "
              "EVERY frame, not just the first few.")

    print("\n--- per-band ERB level (dB) ---")
    print(f"{'band':>5}{'mean':>9}{'p05':>9}{'median':>9}{'p95':>9}")
    q = torch.quantile(erb_db, torch.tensor([0.05, 0.5, 0.95]), dim=0)
    means = erb_db.mean(dim=0)
    for b in range(n_erb):
        print(f"{b:>5}{means[b]:>9.1f}{q[0, b]:>9.1f}{q[1, b]:>9.1f}{q[2, b]:>9.1f}")

    erb_lo, erb_hi, erb_res = fit_ramp(erb_db, 'ERB band', 'dB')
    sp_lo, sp_hi, sp_res = fit_ramp(spec_mag, 'complex bin', '')

    print("\n--- current vs measured ---")
    print(f"{'key':<26}{'current':>12}{'measured':>12}{'delta':>12}")
    rows = [
        # config.ini spells these with "norm_"; read_feature_config does not.
        ('erb_norm_init_lo_db', feat['erb_init_lo_db'], erb_lo, 'dB'),
        ('erb_norm_init_hi_db', feat['erb_init_hi_db'], erb_hi, 'dB'),
        ('spec_norm_init_lo', feat['spec_init_lo'], sp_lo, ''),
        ('spec_norm_init_hi', feat['spec_init_hi'], sp_hi, ''),
    ]
    for key, cur, new, unit in rows:
        delta = f"{new - cur:+.4g}{unit}" if unit else f"{new / max(cur, 1e-30):.2f}x"
        print(f"{key:<26}{cur:>12.4g}{new:>12.4g}{delta:>12}")

    print("\n--- paste into config.ini [feature] ---")
    print(f"erb_norm_init_lo_db = {erb_lo:.1f}")
    print(f"erb_norm_init_hi_db = {erb_hi:.1f}")
    print(f"spec_norm_init_lo = {sp_lo:.6g}")
    print(f"spec_norm_init_hi = {sp_hi:.6g}")
    print(f"\n⚠ Changing these changes the feature contract: bump FEATURE_VERSION "
          f"in config.ini and train.py together, or the checkpoint gate will "
          f"pair old weights with new constants.")
    print(f"⚠ Ramp residuals were {erb_res:.2f} dB (ERB) and {sp_res:.4g} "
          f"(complex).  A large residual means the frequency profile is not "
          f"well described by two numbers, and the ramp is the wrong shape "
          f"rather than merely mis-placed.")
    print("⚠ Do NOT copy these into RNNoise-ERB: different sr, FFT size, band "
          "count and corpus.")


if __name__ == '__main__':
    main()
