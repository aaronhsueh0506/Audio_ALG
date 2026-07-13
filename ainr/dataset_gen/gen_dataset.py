# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Offline pre-generation of training data (WAV pair mode).
Saves augmented (noisy, clean) WAV pairs for flexible downstream use.

This is the standalone, model-independent dataset generator (AINR).
Design: generate ONCE at a canonical high sample rate (48 kHz master; see
--sample-rate below), then let each consuming model resample to its own
target rate at pack-time via resample_dataset.py. See README.md.

Usage:
    python3 gen_dataset.py --config config.ini --output data/ --hours 25
    python3 gen_dataset.py --config config.ini --output data/ --hours 50 --workers 4
    python3 gen_dataset.py --config config.ini --output data/ --hours 25 --sample-rate 48000

Training:
    Training scripts (e.g. RNNoise-ERB's train.py) live in their own model repo
    and consume the WAV pairs produced here (via --wav-data data/ or after
    pack_dataset.py / resample_dataset.py). They are NOT part of this package.
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

    # Canonical generation sample rate.
    # Historically the working rate came ONLY from config.ini's [signal] sr
    # (DNS4Dataset reads cfg.getint('signal', 'sr') and everything — segment
    # length, resample targets for speech/noise/RIR loading, STFT — derives
    # from it). --sample-rate makes that contract explicit at the CLI and
    # overrides the config value, defaulting to the 48 kHz canonical master
    # rate so dataset generation no longer implicitly bakes in one model's
    # target rate (e.g. RNNoise-ERB's 16 kHz). Model-side resampling then
    # happens once at pack-time via resample_dataset.py.
    if args.sample_rate is not None:
        if not cfg.has_section('signal'):
            cfg.add_section('signal')
        cfg_sr_before = cfg.get('signal', 'sr', fallback=None)
        cfg.set('signal', 'sr', str(args.sample_rate))
        if cfg_sr_before is not None and int(cfg_sr_before) != args.sample_rate:
            print(f"Sample rate: overriding config.ini [signal] sr="
                  f"{cfg_sr_before} → {args.sample_rate} (--sample-rate)")
        else:
            print(f"Sample rate: {args.sample_rate} Hz (canonical master)")

    # Seed: 將 start_idx 摻進 seed, 避免擴增時產生與既有檔案完全相同的資料
    cfg_start_idx = cfg.getint('gen', 'start_idx', fallback=0)
    # 起始檔名編號 base_idx: CLI --start-idx 優先, 否則 config [gen] start_idx
    base_idx = args.start_idx if args.start_idx is not None else cfg_start_idx
    if args.seed is not None:
        effective_seed = args.seed + base_idx
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if base_idx > 0:
            print(f"Random seed: {args.seed} + start_idx {base_idx} = {effective_seed}")
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

    # 起始檔名編號 base_idx 是本批的「地板」:
    #   --resume: 掃 pairs/ 取「最大編號 +1」(對齊缺號), 但不低於 base_idx;
    #             空目錄 / 無 >= base_idx 的檔 → 從 base_idx 起。
    #   無 --resume: 直接從 base_idx 起 (擴增, 不掃碟; start_idx 已保證不洗舊檔)。
    sample_count = base_idx
    start_round = 0
    epoch_start = 0
    meta_path = os.path.join(args.output, 'meta.json')

    if args.resume:
        existing = glob.glob(os.path.join(pairs_dir, '*.wav'))
        max_idx = -1
        for p in existing:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.isdigit():
                max_idx = max(max_idx, int(stem))
        sample_count = max(max_idx + 1, base_idx)   # 接最大編號之後, 不低於 base_idx
        done = sample_count - base_idx               # 本批已生成數
        if done > 0:
            start_round = done // epoch_size
            epoch_start = done % epoch_size
            print(f"Resuming: 本批已 {done} 筆 (磁碟最大編號 {max_idx:06d}.wav), "
                  f"從 {sample_count:06d}.wav 接續 "
                  f"(round {start_round + 1}, epoch idx {epoch_start})...")
        else:
            print(f"Resume: pairs/ 無 >= {base_idx:06d} 的檔 → 從 {base_idx:06d}.wav 開始")
    elif base_idx > 0:
        print(f"Extending dataset: 從 {base_idx:06d}.wav 開始 (舊檔不會被覆蓋)")

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

        idx_start = epoch_start if r == start_round else 0

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
        description='Offline pre-generation of model-independent (noisy, clean) '
                    'WAV training pairs, generated once at a canonical sample rate')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--hours', type=float, default=8.3,
                        help='Target audio hours (auto-rounds to nearest epoch)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers (default: 4, 0=single process)')
    parser.add_argument('--resume', action='store_true',
                        help='接續同一批: 從 pairs/ 最大編號+1 續寫 (不低於 start_idx)')
    parser.add_argument('--start-idx', type=int, default=None,
                        help='起始檔名編號, 覆寫 config [gen] start_idx (擴增用)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42, -1 to disable)')
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Canonical generation sample rate in Hz (default: 48000 — '
                            'the "48k master"). Overrides config.ini [signal] sr. '
                            'Model-specific rates should be produced downstream via '
                            'resample_dataset.py, not by lowering this at gen time.')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    gen_dataset(args)
