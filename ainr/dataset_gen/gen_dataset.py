# -*- coding: utf-8 -*-
"""
Offline pre-generation of training data (WAV pair mode).
Saves augmented (noisy, clean) WAV pairs for flexible downstream use.

This is the standalone, model-independent dataset generator (AINR).
Choose one generation sample rate per run through config.ini's `[signal] sr`
or `--sample-rate`. Use 16 kHz for RNNoise-ERB/GTCRN and 48 kHz for
DeepFilterNet2. See README.md.

Usage:
    python3 gen_dataset.py --config config.ini --output data/ --hours 25
    python3 gen_dataset.py --config config.ini --output data/ --hours 50 --workers 4
    python3 gen_dataset.py --config config.ini --output data_16k/ --hours 25 \
        --sample-rate 16000
    python3 gen_dataset.py --config config.ini --output data_48k/ --hours 25 \
        --sample-rate 48000

Training:
    Training scripts (e.g. RNNoise-ERB's train.py) live in their model
    directories and consume packed copies of the WAV pairs produced here.
    They are NOT part of this package.
"""

import argparse
import configparser
from decimal import Decimal, ROUND_CEILING
import glob
import json
import math
import os
import random
import secrets
import time

import numpy as np
import torch
import torch.utils.data as data
import torchaudio
import tqdm

try:
    from .dataset import DNS4Dataset
except ImportError:
    from dataset import DNS4Dataset


def hours_to_sample_count(hours: float, segment_sec: float) -> int:
    """Return the minimum number of whole segments covering ``hours``."""
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError(f"--hours must be a positive finite number, got {hours}")
    if not math.isfinite(segment_sec) or segment_sec <= 0:
        raise ValueError(
            f"[audio] segment_sec must be a positive finite number, got {segment_sec}"
        )
    # Convert through the decimal spelling supplied by argparse/configparser.
    # Binary float would make an exact case such as 8.3 h / 3 s evaluate to
    # 9960.000000000002 and spuriously add one more segment.
    segment_count = (
        Decimal(str(hours)) * Decimal(3600) / Decimal(str(segment_sec))
    )
    return int(segment_count.to_integral_value(rounding=ROUND_CEILING))


def seed_worker(_worker_id):
    """Give each DataLoader worker independent Python/NumPy random streams."""
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2 ** 32))


def gen_dataset(args):
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    # One generation rate per run. CLI overrides config; omitting the CLI flag
    # genuinely leaves `[signal] sr` in control.
    cfg_sample_rate = cfg.getint('signal', 'sr', fallback=48000)
    cli_sample_rate = getattr(args, 'sample_rate', None)
    generation_sr = (
        cli_sample_rate if cli_sample_rate is not None else cfg_sample_rate
    )
    if generation_sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {generation_sr}")
    if not cfg.has_section('signal'):
        cfg.add_section('signal')
    cfg.set('signal', 'sr', str(generation_sr))
    if cli_sample_rate is not None and generation_sr != cfg_sample_rate:
        print(f"Sample rate: overriding config.ini [signal] sr="
              f"{cfg_sample_rate} → {generation_sr} (--sample-rate)")
    else:
        source = "config.ini [signal] sr" if cli_sample_rate is None else "CLI"
        print(f"Sample rate: {generation_sr} Hz ({source})")

    segment_sec = cfg.getfloat('audio', 'segment_sec', fallback=3.0)
    n_total = hours_to_sample_count(args.hours, segment_sec)

    # Seed: default/config -1 obtains a fresh OS-random seed every run. An
    # explicit non-negative seed gives reproducible generation. start_idx is
    # mixed in so an extension does not replay the first batch's random stream.
    cfg_start_idx = cfg.getint('gen', 'start_idx', fallback=0)
    base_idx = args.start_idx if args.start_idx is not None else cfg_start_idx
    if base_idx < 0:
        raise ValueError(f"start_idx must be non-negative, got {base_idx}")

    cfg_seed = cfg.getint('gen', 'seed', fallback=-1)
    cli_seed = getattr(args, 'seed', None)
    requested_seed = cli_seed if cli_seed is not None else cfg_seed
    if requested_seed < 0:
        run_seed = secrets.randbits(63)
        seed_source = "OS-generated"
    else:
        run_seed = requested_seed
        seed_source = "explicit"
    effective_seed = (run_seed + base_idx) % (2 ** 63)
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2 ** 32))
    torch.manual_seed(effective_seed)
    if base_idx > 0:
        print(f"Random seed ({seed_source}): {run_seed} + start_idx "
              f"{base_idx} = {effective_seed}")
    else:
        print(f"Random seed ({seed_source}): {effective_seed}")

    dataset = DNS4Dataset(cfg, return_raw=True)
    SR = dataset.sr
    epoch_size = len(dataset)
    if epoch_size <= 0:
        raise ValueError("DNS4Dataset contains no samples")

    pairs_dir = os.path.join(args.output, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)

    n_rounds = (n_total + epoch_size - 1) // epoch_size
    actual_hours = n_total * segment_sec / 3600

    # Profile 1 sample
    print("Profiling 1 sample...")
    t0 = time.time()
    _s, _t = dataset[0]
    t_sample = time.time() - t0
    # 2 channels × 2 bytes/sample (16-bit WAV).
    disk_bytes = dataset.segment_samples * 2 * 2 * n_total
    n_workers = args.workers
    speedup = max(1, n_workers) if n_workers > 0 else 1
    est_hours = t_sample * n_total / 3600 / speedup

    disk_str = (f"{disk_bytes / 1024**3:.1f} GB" if disk_bytes >= 1024**3
                else f"{disk_bytes / 1024**2:.0f} MB")
    partial = n_total % epoch_size
    pass_note = (
        f"{n_rounds} dataset pass(es), final pass {partial}/{epoch_size}"
        if partial
        else f"{n_rounds} complete dataset pass(es)"
    )
    print(f"\nRequested {args.hours:g} hours → {n_total} samples "
          f"({actual_hours:.3f} hours; {pass_note})")
    print(f"  Workers          : {n_workers}")
    print(f"  Estimated gen time : {est_hours:.1f} hours ({t_sample:.3f}s/sample)")
    print(f"  Estimated disk     : {disk_str}  (16-bit WAV)")
    print(f"  Sample rate        : {SR} Hz")
    print(f"  Output: {args.output}/")
    print()

    # 起始檔名編號 base_idx 是本批的「地板」:
    #   --resume: 掃 pairs/ 取「最大編號 +1」，但不低於 base_idx。
    #   無 --resume: 直接從 base_idx 起 (擴增, 不掃碟; start_idx 已保證不洗舊檔)。
    sample_count = base_idx
    start_round = 0
    epoch_start = 0
    meta_path = os.path.join(args.output, 'meta.json')

    if args.resume:
        existing = glob.glob(os.path.join(pairs_dir, '*.wav'))
        max_idx = -1
        for path in existing:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.isdigit():
                max_idx = max(max_idx, int(stem))
        sample_count = max(max_idx + 1, base_idx)
        done = sample_count - base_idx
        if done > 0:
            start_round = done // epoch_size
            epoch_start = done % epoch_size
            print(f"Resuming: 本批已完成 {done} 筆 "
                  f"(max={max_idx:06d}.wav), "
                  f"從 {sample_count:06d}.wav 接續 "
                  f"(round {start_round + 1}, epoch idx {epoch_start})...")
        else:
            print(f"Resume: pairs/ 無 >= {base_idx:06d} 的檔 "
                  f"→ 從 {base_idx:06d}.wav 開始")
    elif base_idx > 0:
        print(f"Extending dataset: 從 {base_idx:06d}.wav 開始 (舊檔不會被覆蓋)")

    def _save_meta(rounds_done):
        batch_samples = sample_count - base_idx
        meta = {
            'n_samples': sample_count,
            'sr': SR,
            'segment_sec': segment_sec,
            'segment_samples': dataset.segment_samples,
            'hours': sample_count * segment_sec / 3600,
            'batch_start_idx': base_idx,
            'batch_n_samples': batch_samples,
            'batch_hours': batch_samples * segment_sec / 3600,
            'requested_hours': args.hours,
            'target_n_samples': n_total,
            'n_rounds_done': rounds_done,
            'seed': run_seed,
            'effective_seed': effective_seed,
            'seed_source': seed_source,
            'config': args.config,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _save_pair(noisy, clean, index):
        pair = torch.stack([noisy, clean], dim=0)
        fname = f"{index:06d}.wav"
        torchaudio.save(
            os.path.join(pairs_dir, fname),
            pair,
            SR,
            bits_per_sample=16,
        )

    gen_start = time.time()

    for r in range(n_rounds):
        if r < start_round:
            continue
        if sample_count - base_idx >= n_total:
            break

        if n_rounds > 1:
            dataset._shuffle_indices()
            print(f"\n--- Round {r + 1}/{n_rounds} ---")

        idx_start = epoch_start if r == start_round else 0
        remaining = n_total - (sample_count - base_idx)
        idx_stop = min(epoch_size, idx_start + remaining)

        if n_workers > 0:
            indices = list(range(idx_start, idx_stop))
            subset = data.Subset(dataset, indices)
            loader = data.DataLoader(
                subset, batch_size=1, shuffle=False,
                num_workers=n_workers, prefetch_factor=2,
                worker_init_fn=seed_worker,
                persistent_workers=False,
            )
            for noisy, clean in tqdm.tqdm(loader, desc=f"Round {r+1}/{n_rounds}",
                                          total=len(indices)):
                noisy = noisy.squeeze(0)   # (T,)
                clean = clean.squeeze(0)
                _save_pair(noisy, clean, sample_count)
                sample_count += 1
        else:
            for i in tqdm.tqdm(range(idx_start, idx_stop),
                               desc=f"Round {r+1}/{n_rounds}"):
                noisy, clean = dataset[i]
                _save_pair(noisy, clean, sample_count)
                sample_count += 1

        _save_meta(r + 1)

    gen_elapsed = time.time() - gen_start
    batch_samples = sample_count - base_idx
    if batch_samples >= n_total:
        completed_rounds = (batch_samples + epoch_size - 1) // epoch_size
        _save_meta(min(n_rounds, completed_rounds))
    print(f"\nDone. Batch has {batch_samples}/{n_total} pairs "
          f"({batch_samples * segment_sec / 3600:.3f} audio hours); "
          f"next file index is {sample_count:06d}. "
          f"Generation took {gen_elapsed / 3600:.2f} hours → {args.output}/")
    print(f"  {SR} Hz: {pairs_dir} "
          f"(2-channel WAV: ch0=noisy, ch1=clean)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Offline pre-generation of model-independent (noisy, clean) '
                    'WAV training pairs at one selected sample rate')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--hours', type=float, default=8.3,
                        help='Target audio hours (rounded up by at most one segment)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers (default: 4, 0=single process)')
    parser.add_argument('--resume', action='store_true',
                        help='接續同一批: 從 pairs/ 最大編號後續寫 '
                             '(不低於 start_idx)')
    parser.add_argument('--start-idx', type=int, default=None,
                        help='起始檔名編號, 覆寫 config [gen] start_idx (擴增用)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Reproducible random seed. Omit to use [gen] seed; '
                             'negative means a fresh OS-random seed each run.')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Generation sample rate in Hz. Overrides config.ini '
                             '[signal] sr (16000 for RNNoise-ERB/GTCRN, '
                             '48000 for DeepFilterNet2).')
    args = parser.parse_args()
    gen_dataset(args)
