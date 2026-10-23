"""Measure this model's steady-state feature distribution and emit init values.

WHY THIS EXISTS
---------------
``erb_norm_init_lo_db`` / ``erb_norm_init_hi_db`` / ``spec_norm_init_lo`` /
``spec_norm_init_hi`` in ``config.ini`` are libDF's, calibrated for an energy-MEAN
over a rectangular, non-overlapping filterbank (``k = 1/band_size``). This port
uses the triangular, overlapping, partition-of-unity bank built by
``_build_erb_fb`` (482 of its 513 bins are shared between bands, per-band weights
sum to 2..72 rather than 1), so band energy grows with band width and the imported
constants start the EMA between +3.0 dB and +18.6 dB away from where this model's
features sit.

⚠ These numbers are NOT transferable from RNNoise-ERB (16 kHz, 512-point FFT,
22 bands vs 48 kHz, 1024-point FFT, 32 bands); run this script here and
RNNoise-ERB's in that directory.

A constant offset would normally wash out as ``band_mean_norm``'s running mean
converges, but at tau = 1 s, hop 512, 48 kHz, 3*tau is 281 frames, the length
of a 3-second training segment, so every frame lies in the init-dominated
transient. (Against the realised alpha of 0.989, 3*tau is 273 frames: 3-decimal
rounding makes the effective tau 0.9644 s, see config.ini [feature]; the
conclusion holds either way.)

⚠ EXPECT A LARGE DELTA AGAINST config.ini, AND DO NOT "CORRECT" IT. config.ini
ships libDF's literal ``-60/-90`` and ``0.001/0.0001``; this script will
recommend something ~39 dB away, because those literals are calibrated for
libDF's rectangular/energy-mean bank and 1/N analysis scale, not for this
port's triangular/energy-sum bank and ``normalized=True``. The corpus-fitted
pair (-17.35/-57.81, rounded to -15/-60) is not used: training diverged under
it and converges under the literals, on an identical corpus. So this script's
output is a measurement, not a recommendation to paste; read
UPSTREAM_ALIGNMENT.md do-not-touch item 4 before acting on it.

USAGE
-----
    python calibrate_norm_init.py --config config.ini --packed-dir /path/to/48k

Run it on the machine holding the packed corpus, against the SAME corpus the
model will train on. It prints the measured distribution and the config lines
to paste back, and modifies nothing.
"""

import argparse
import configparser
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .model import DeepFilterNet3
    from .train import erb_band_db, read_feature_config, read_model_config
except ImportError:  # direct script execution
    from model import DeepFilterNet3
    from train import erb_band_db, read_feature_config, read_model_config
from dataset_gen import (  # noqa: E402
    describe_bands,
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
    ap.add_argument('--output-json', default=None,
                    help='optional machine-readable calibration report')
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
    model = DeepFilterNet3(**model_cfg).eval()
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
            # The trainer's own band-dB step, called rather than transcribed --
            # the floor, the power form and the wnorm cannot drift from it.
            erb_db.append(
                erb_band_db(spec.transpose(1, 2), erb_fb,
                            feat['analysis_scale'])[0, skip:]
            )
            spec_mag.append(
                (spec[0, :df_bins, skip:] * feat['analysis_scale'])
                .abs().transpose(0, 1)
            )

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
    describe_bands(erb_db, erb_fb.sum(dim=1), feat['erb_scale_db'])

    erb_lo, erb_hi, erb_res = fit_ramp(erb_db, 'ERB band', 'dB')
    sp_lo, sp_hi, sp_res = fit_ramp(spec_mag, 'complex bin', '', require_positive=True)

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

    if args.output_json:
        report = {
            'schema': 'dfn3-normalization-calibration-v1',
            'sample_rate': SR,
            'n_fft': N_FFT,
            'win_len': WIN_LEN,
            'hop_len': HOP_LEN,
            'clips': len(idx),
            'frames': int(erb_db.shape[0]),
            'erb_norm_init_lo_db': float(erb_lo),
            'erb_norm_init_hi_db': float(erb_hi),
            'spec_norm_init_lo': float(sp_lo),
            'spec_norm_init_hi': float(sp_hi),
            'erb_ramp_residual_db': float(erb_res),
            'spec_ramp_residual': float(sp_res),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as fp:
            json.dump(report, fp, indent=2, sort_keys=True)
            fp.write('\n')
        print(f"calibration report: {args.output_json}")


if __name__ == '__main__':
    main()
