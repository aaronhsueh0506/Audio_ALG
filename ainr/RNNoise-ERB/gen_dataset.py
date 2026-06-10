# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Offline pre-generation of training data (WAV pair mode).
Saves augmented (noisy, clean) WAV pairs for flexible downstream use.

Usage:
    python3 gen_dataset.py --config config.ini --output data/ --hours 25
    python3 gen_dataset.py --config config.ini --output data/ --hours 50 --workers 4

Training:
    python3 train.py --config config.ini --wav-data data/
"""

import argparse
import configparser
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.utils.data as data
import torchaudio
import tqdm

from dataset import DNS4Dataset


def gen_dataset(args):
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    # Seed: 將 start_idx 摻進 seed, 避免擴增時產生與既有檔案完全相同的資料
    cfg_start_idx = cfg.getint('gen', 'start_idx', fallback=0)
    if args.seed is not None:
        effective_seed = args.seed + cfg_start_idx
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if cfg_start_idx > 0:
            print(f"Random seed: {args.seed} + start_idx {cfg_start_idx} = {effective_seed}")
        else:
            print(f"Random seed: {args.seed}")

    dataset = DNS4Dataset(cfg, return_raw=True)
    SR = dataset.sr
    segment_sec = cfg.getfloat('audio', 'segment_sec', fallback=3.0)
    epoch_size = len(dataset)
    epoch_hours = epoch_size * segment_sec / 3600

    n_rounds = max(1, round(args.hours / epoch_hours))
    n_total = epoch_size * n_rounds
    actual_hours = n_total * segment_sec / 3600

    pairs_dir = os.path.join(args.output, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)

    # Profile 1 sample
    print("Profiling 1 sample...")
    t0 = time.time()
    _s, _t = dataset[0]
    t_sample = time.time() - t0
    # rough disk estimate: 2 WAVs × 2 bytes/sample × segment_samples
    sample_bytes = int(_s.numel() * 2 * 2)
    disk_bytes = int(sample_bytes * n_total)
    n_workers = args.workers
    speedup = max(1, n_workers) if n_workers > 0 else 1
    est_hours = t_sample * n_total / 3600 / speedup

    disk_str = (f"{disk_bytes / 1024**3:.1f} GB" if disk_bytes >= 1024**3
                else f"{disk_bytes / 1024**2:.0f} MB")
    print(f"\nRequested {args.hours:.1f} hours → {n_rounds}x epoch "
          f"({actual_hours:.1f} hours, {n_total} samples)")
    print(f"  Workers          : {n_workers}")
    print(f"  Estimated gen time : {est_hours:.1f} hours ({t_sample:.3f}s/sample)")
    print(f"  Estimated disk     : {disk_str}  (16-bit WAV)")
    print(f"  Output: {args.output}/\n")

    # 初始 sample_count:
    #   --resume 優先 (掃磁碟取 max+1, 跳過缺號)
    #   否則用 config [gen] start_idx (擴增 dataset 不洗舊檔, seed 已被偏移)
    sample_count = cfg_start_idx
    start_round = 0
    start_idx = 0
    meta_path = os.path.join(args.output, 'meta.json')

    if args.resume:
        existing = sorted(glob.glob(os.path.join(pairs_dir, '*.wav')))
        sample_count = len(existing)
        if sample_count > 0:
            start_round = sample_count // epoch_size
            start_idx = sample_count % epoch_size
            print(f"Resuming: {sample_count} samples done. "
                  f"Starting from round {start_round + 1}, idx {start_idx}...")
    elif cfg_start_idx > 0:
        print(f"Extending dataset: start from index {cfg_start_idx:06d}.wav "
              f"(舊檔不會被覆蓋)")

    def _save_meta():
        meta = {
            'n_samples': sample_count,
            'sr': SR,
            'segment_sec': segment_sec,
            'segment_samples': dataset.segment_samples,
            'hours': sample_count * segment_sec / 3600,
            'n_rounds_done': r + 1,
            'config': args.config,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    gen_start = time.time()

    for r in range(n_rounds):
        if r < start_round:
            continue

        if n_rounds > 1:
            dataset._shuffle_indices()
            print(f"\n--- Round {r + 1}/{n_rounds} ---")

        idx_start = start_idx if r == start_round else 0

        if n_workers > 0:
            indices = list(range(idx_start, epoch_size))
            subset = data.Subset(dataset, indices)
            loader = data.DataLoader(
                subset, batch_size=1, shuffle=False,
                num_workers=n_workers, prefetch_factor=2,
                persistent_workers=False,
            )
            for noisy, clean in tqdm.tqdm(loader, desc=f"Round {r+1}/{n_rounds}",
                                          total=len(indices)):
                noisy = noisy.squeeze(0)   # (T,)
                clean = clean.squeeze(0)
                # 2-channel: ch0=noisy, ch1=clean
                pair = torch.stack([noisy, clean], dim=0)   # (2, T)
                fname = f"{sample_count:06d}.wav"
                torchaudio.save(os.path.join(pairs_dir, fname),
                                pair, SR, bits_per_sample=16)
                sample_count += 1
        else:
            for i in tqdm.tqdm(range(idx_start, epoch_size),
                               desc=f"Round {r+1}/{n_rounds}"):
                noisy, clean = dataset[i]
                pair = torch.stack([noisy, clean], dim=0)   # (2, T)
                fname = f"{sample_count:06d}.wav"
                torchaudio.save(os.path.join(pairs_dir, fname),
                                pair, SR, bits_per_sample=16)
                sample_count += 1

        _save_meta()

    gen_elapsed = time.time() - gen_start
    print(f"\nDone. {sample_count} pairs in {gen_elapsed / 3600:.2f} hours → {args.output}/")
    print(f"  pairs/ : {pairs_dir}  (2-channel WAV: ch0=noisy, ch1=clean)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Offline pre-generation of RNNoise-ERB WAV training pairs')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--hours', type=float, default=8.3,
                        help='Target audio hours (auto-rounds to nearest epoch)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers (default: 4, 0=single process)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing WAVs (count noisy/*.wav)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42, -1 to disable)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    gen_dataset(args)
