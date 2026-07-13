# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
把 WAV pair 目錄打包成單一 .pt 檔，消除訓練時的逐檔 I/O 開銷。

用法:
    python pack_dataset.py --input data/pairs/ --output data/packed.pt
    python pack_dataset.py --input data/pairs/ --output data/packed.pt --dtype float16

訓練:
    python train.py --config config.ini --packed-data data/packed.pt

注意:
    float16 檔案大小減半，精度對音訊影響可忽略 (WAV 本身就是 16-bit)。
    大資料集建議先試 float16，省記憶體也省磁碟。
"""

import argparse
import glob
import os

import torch
import torchaudio
import tqdm


def pack(args):
    files = sorted(glob.glob(os.path.join(args.input, '**', '*.wav'), recursive=True))
    if not files:
        raise FileNotFoundError(f"在 {args.input} 找不到任何 .wav 檔案")

    N = len(files)
    print(f"找到 {N} 個檔案 → {args.output}")

    dtype = torch.float16 if args.dtype == 'float16' else torch.float32

    # 從第一個檔案取得 T, sr
    sample, sr = torchaudio.load(files[0])  # (2, T)
    T = sample.size(1)

    bytes_per = 2 if dtype == torch.float16 else 4
    total_bytes = N * 2 * T * bytes_per
    size_str = (f"{total_bytes / 1024**3:.1f} GB" if total_bytes >= 1024**3
                else f"{total_bytes / 1024**2:.0f} MB")
    print(f"  SR={sr}, 每段 {T} samples, dtype={args.dtype}, 預估大小: {size_str}")

    data = torch.empty(N, 2, T, dtype=dtype)

    failed = []
    for i, path in enumerate(tqdm.tqdm(files, desc="Packing")):
        try:
            audio, _ = torchaudio.load(path)  # (2, T)
            if audio.shape[0] < 2:
                raise ValueError(f"不是 2-channel WAV: {path}")
            if audio.shape[1] != T:
                raise ValueError(f"長度不符 (expected {T}, got {audio.shape[1]}): {path}")
            data[i] = audio.to(dtype)
        except Exception as e:
            failed.append((path, str(e)))
            data[i] = 0  # 填 0，之後過濾

    if failed:
        print(f"\n警告: {len(failed)} 個檔案讀取失敗，已從 dataset 移除:")
        for path, err in failed:
            print(f"  {path}: {err}")
        good_mask = torch.ones(N, dtype=torch.bool)
        fail_indices = {files.index(p) for p, _ in failed}
        for idx in fail_indices:
            good_mask[idx] = False
        data = data[good_mask]
        N = data.size(0)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    print(f"儲存中 ({size_str})...")
    torch.save({
        'data': data,          # (N, 2, T): ch0=noisy, ch1=clean
        'sr': sr,
        'n_samples': N,
        'segment_samples': T,
        'dtype': args.dtype,
        'source': args.input,
    }, args.output)
    print(f"完成: {args.output}  ({N} pairs, {size_str})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='將 WAV pair 目錄打包成單一 .pt 檔')
    parser.add_argument('--input', required=True,
                        help='WAV pair 目錄 (gen_dataset.py 的輸出, 遞迴掃描)')
    parser.add_argument('--output', required=True,
                        help='輸出 .pt 檔案路徑')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16'],
                        help='儲存精度 (float16 減半大小, 預設: float32)')
    args = parser.parse_args()
    pack(args)
