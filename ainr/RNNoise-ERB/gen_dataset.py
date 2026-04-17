# -*- coding: utf-8 -*-
"""
Offline pre-generation of training data -- run augmentation pipeline once,
save as .pt shard files. Training then loads directly without real-time I/O + DSP.

Usage:
    python3 gen_dataset.py --config config.ini --output data/ --hours 25
    python3 gen_dataset.py --config config.ini --output data/ --hours 50 --workers 8

Training:
    python3 train.py --config config.ini --precomputed data/
"""

import argparse
import configparser
import os
import random
import time

import numpy as np
import torch
import torch.utils.data as data
import tqdm

from dataset import DNS4Dataset


def gen_dataset(args):
    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    dataset = DNS4Dataset(cfg)
    epoch_size = len(dataset)
    segment_sec = cfg.getfloat('audio', 'segment_sec', fallback=3.0)
    epoch_hours = epoch_size * segment_sec / 3600

    n_rounds = max(1, round(args.hours / epoch_hours))
    n_total = epoch_size * n_rounds
    actual_hours = n_total * segment_sec / 3600
    n_shards = args.n_shards

    os.makedirs(args.output, exist_ok=True)

    shard_size = (n_total + n_shards - 1) // n_shards

    # Profile 1 sample for time/disk estimation
    print("Profiling 1 sample...")
    t0 = time.time()
    sample_feat, sample_tgt = dataset[0]
    t_sample = time.time() - t0
    sample_bytes = (sample_feat.numel() + sample_tgt.numel()) * 4
    disk_bytes = int(sample_bytes * n_total * 1.3)

    # Adjust estimate for workers (rough speedup factor)
    n_workers = args.workers
    speedup = max(1, n_workers) if n_workers > 0 else 1
    est_hours = t_sample * n_total / 3600 / speedup

    if disk_bytes >= 1024 ** 3:
        disk_str = f"{disk_bytes / 1024**3:.1f} GB"
    else:
        disk_str = f"{disk_bytes / 1024**2:.0f} MB"

    print(f"\nRequested {args.hours:.1f} hours -> {n_rounds}x epoch "
          f"({actual_hours:.1f} hours, {n_total} samples)")
    print(f"  {n_shards} shards (~{shard_size} per shard)")
    print(f"  Workers          : {n_workers}")
    print(f"  Estimated gen time : {est_hours:.1f} hours ({t_sample:.3f}s/sample)")
    print(f"  Estimated disk     : {disk_str}")
    print(f"  Output: {args.output}/\n")

    # Resume: 偵測已存的 shard，計算從哪個 round / index 繼續
    shard_features = []
    shard_targets = []
    shard_id = 0
    sample_count = 0
    first_feat = None
    start_round = 0
    start_idx = 0  # 第一個 resume round 要從哪個 index 開始

    meta_path = os.path.join(args.output, 'meta.pt')
    if args.resume and os.path.isfile(meta_path):
        prev_meta = torch.load(meta_path, weights_only=False)
        shard_id = prev_meta['n_shards']
        sample_count = prev_meta['n_total']
        first_feat = torch.zeros(prev_meta['seq_len'], prev_meta['n_bands'])
        start_round = sample_count // epoch_size
        start_idx = sample_count % epoch_size
        print(f"Resuming: {shard_id} shards, {sample_count} samples "
              f"({prev_meta.get('hours', 0):.1f} hrs). "
              f"Continuing from round {start_round + 1}, sample {start_idx}...")

    # Generate samples and save shards incrementally (省記憶體)
    gen_start = time.time()

    def _save_shard():
        nonlocal shard_id, shard_features, shard_targets, sample_count
        if not shard_features:
            return
        shard_data = {
            'features': torch.stack(shard_features),
            'targets': torch.stack(shard_targets),
        }
        shard_path = os.path.join(args.output, f'shard_{shard_id:04d}.pt')
        torch.save(shard_data, shard_path)
        print(f"  {shard_path}: {len(shard_features)} samples, "
              f"features={shard_data['features'].shape}")
        shard_id += 1
        shard_features.clear()
        shard_targets.clear()

        # 每存完一個 shard 就更新 meta.pt，中斷後已存的 shard 仍可用
        meta = {
            'n_shards': shard_id,
            'n_total': sample_count,
            'shard_size': shard_size,
            'n_rounds': n_rounds,
            'hours': sample_count * segment_sec / 3600,
            'seq_len': first_feat.shape[0],
            'n_bands': first_feat.shape[1],
            'config': args.config,
        }
        torch.save(meta, os.path.join(args.output, 'meta.pt'))

    for r in range(n_rounds):
        if r < start_round:
            # 完整跳過已完成的 round，不跑 augmentation
            continue

        if n_rounds > 1:
            dataset._shuffle_indices()
            print(f"\n--- Round {r + 1}/{n_rounds} ---")

        # resume 時第一個 round 從 start_idx 開始，之後正常從 0 開始
        idx_start = start_idx if r == start_round else 0

        if n_workers > 0:
            # 用 Subset 讓 DataLoader 只跑剩餘的 index
            indices = list(range(idx_start, epoch_size))
            subset = data.Subset(dataset, indices)
            loader = data.DataLoader(
                subset, batch_size=1, shuffle=False,
                num_workers=n_workers, prefetch_factor=2,
                persistent_workers=False,
            )
            for feat, tgt in tqdm.tqdm(loader, desc=f"Round {r+1}/{n_rounds}",
                                       total=len(indices)):
                feat, tgt = feat.squeeze(0), tgt.squeeze(0)
                shard_features.append(feat)
                shard_targets.append(tgt)
                if first_feat is None:
                    first_feat = feat
                sample_count += 1
                if len(shard_features) >= shard_size:
                    _save_shard()
        else:
            for i in tqdm.tqdm(range(idx_start, epoch_size),
                                desc=f"Round {r+1}/{n_rounds}"):
                feat, tgt = dataset[i]
                shard_features.append(feat)
                shard_targets.append(tgt)
                if first_feat is None:
                    first_feat = feat
                sample_count += 1
                if len(shard_features) >= shard_size:
                    _save_shard()

    # 存最後不滿一個 shard 的剩餘資料
    _save_shard()

    gen_elapsed = time.time() - gen_start
    print(f"\nDone. {sample_count} samples, {shard_id} shards "
          f"in {gen_elapsed / 3600:.2f} hours → {args.output}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Offline pre-generation of RNNoise-ERB training data')
    parser.add_argument('--config', default='config.ini',
                        help='Config file path')
    parser.add_argument('--output', default='data',
                        help='Output directory')
    parser.add_argument('--hours', type=float, default=8.3,
                        help='Target audio hours (auto-rounds to nearest epoch, default: 8.3)')
    parser.add_argument('--n-shards', type=int, default=10,
                        help='Number of shard files (default: 10)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of DataLoader workers (default: 4, 0=single process)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing shards (read meta.pt to skip)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42, -1 to disable)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    gen_dataset(args)
