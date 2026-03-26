"""
離線預生成訓練資料 — 跑一次 augmentation pipeline，存成 .pt shard 檔
後續訓練直接讀取，不需要即時做 I/O + DSP

用法:
    python gen_dataset.py --config config.ini --output data/ --n-shards 10
    python gen_dataset.py --config config.ini --output data/ --n-shards 10 --seed 42

訓練時:
    python train.py --config config.ini --precomputed data/
"""

import argparse
import configparser
import os
import random
import time

import numpy as np
import torch
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
    n_rounds = args.multiply
    n_total = epoch_size * n_rounds
    n_shards = args.n_shards

    os.makedirs(args.output, exist_ok=True)

    # 計算每個 shard 的大小
    shard_size = (n_total + n_shards - 1) // n_shards

    # 先跑一個 sample 來估算時間和空間
    print("Profiling 1 sample...")
    t0 = time.time()
    sample_feat, sample_tgt = dataset[0]
    t_sample = time.time() - t0
    # 每個 sample 的 bytes: features + targets, float32
    sample_bytes = (sample_feat.numel() + sample_tgt.numel()) * 4
    # torch.save overhead ~1.3x (pickle + header)
    disk_bytes = int(sample_bytes * n_total * 1.3)

    est_hours = t_sample * n_total / 3600
    if disk_bytes >= 1024 ** 3:
        disk_str = f"{disk_bytes / 1024**3:.1f} GB"
    else:
        disk_str = f"{disk_bytes / 1024**2:.0f} MB"

    print(f"\nGenerating {n_total} samples ({epoch_size} x {n_rounds} rounds) "
          f"into {n_shards} shards (~{shard_size} per shard)")
    print(f"  Estimated time : {est_hours:.1f} hours ({t_sample:.3f}s/sample)")
    print(f"  Estimated disk : {disk_str}")
    print(f"  Output: {args.output}/\n")

    # 收集所有 samples（每 round 重新 shuffle → 不同 augmentation）
    all_features = []
    all_targets = []
    gen_start = time.time()
    for r in range(n_rounds):
        if n_rounds > 1:
            dataset._shuffle_indices()
            print(f"\n--- Round {r + 1}/{n_rounds} ---")
        for i in tqdm.tqdm(range(epoch_size), desc=f"Round {r + 1}/{n_rounds}"):
            feat, tgt = dataset[i]
            all_features.append(feat)
            all_targets.append(tgt)

    gen_elapsed = time.time() - gen_start
    print(f"\nGeneration done in {gen_elapsed / 3600:.2f} hours")

    # 分 shard 存檔
    for shard_id in range(n_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n_total)

        shard_data = {
            'features': torch.stack(all_features[start:end]),
            'targets': torch.stack(all_targets[start:end]),
        }

        shard_path = os.path.join(args.output, f'shard_{shard_id:04d}.pt')
        torch.save(shard_data, shard_path)
        print(f"  {shard_path}: {end - start} samples, "
              f"features={shard_data['features'].shape}")

    # 存 metadata
    meta = {
        'n_shards': n_shards,
        'n_total': n_total,
        'shard_size': shard_size,
        'multiply': n_rounds,
        'seq_len': all_features[0].shape[0],
        'n_bands': all_features[0].shape[1],
        'config': args.config,
    }
    torch.save(meta, os.path.join(args.output, 'meta.pt'))
    print(f"\nDone. {n_total} samples saved to {args.output}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='離線預生成 RNNoise-ERB 訓練資料')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--output', default='data', help='輸出目錄')
    parser.add_argument('--multiply', type=int, default=1,
                        help='生成幾倍資料 (預設: 1, 每倍不同 augmentation)')
    parser.add_argument('--n-shards', type=int, default=10,
                        help='分成幾個 shard 檔 (預設: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42, 設 -1 關閉)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    gen_dataset(args)
