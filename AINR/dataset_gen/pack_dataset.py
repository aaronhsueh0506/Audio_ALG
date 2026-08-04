# -*- coding: utf-8 -*-
"""
把 WAV pair 目錄打包成單一 .pt 檔，消除訓練時的逐檔 I/O 開銷。
可選擇在打包當下順便 resample，這樣 48 kHz 母帶可以直接產生 16 kHz 的
packed 檔，不需要先用 resample_dataset.py 生一份中繼 WAV 目錄。

用法:
    python pack_dataset.py --input data_48k/pairs --output data_48k/packed.pt
    python pack_dataset.py --input data_48k/pairs --output data_48k/packed.pt --dtype float16

    # 打包時直接 resample 成另一個 rate (跳過中繼 WAV, 少一次 int16 write/read)
    python pack_dataset.py --input data_48k/pairs --output data_16k/packed.pt \
        --target-sr 16000 --dtype float16

訓練:
    python train.py --config config.ini --packed-data data/packed.pt

注意:
    float16 檔案大小減半，精度對音訊影響可忽略 (WAV 本身就是 16-bit)。
    大資料集建議先試 float16，省記憶體也省磁碟。

    --target-sr 用的 anti-aliasing 參數跟 resample_dataset.py 是同一份常數
    (見 resample_dataset.py 的 KAISER_BEST/KAISER_FAST/CLIP_GUARD)，兩條路徑
    數值上等價；resample_dataset.py 仍保留，需要一份實際 WAV 做聽感檢查/外部
    工具比對時用那個。
"""

import argparse
import glob
import os

import torch
import torchaudio
import tqdm

from resample_dataset import KAISER_BEST, KAISER_FAST, CLIP_GUARD


def _load_maybe_resampled(path, target_sr, resample_kwargs):
    """Load a WAV pair, optionally resampling + clip-guarding it in-memory.

    Mirrors resample_dataset.py's _ResampleJob.__getitem__ exactly (same
    anti-aliasing kwargs, same clip-guard rescale), minus the WAV write.
    Returns (audio, source_sr, was_clip_guarded).
    """
    audio, sr = torchaudio.load(path)  # (2, T)
    resampled = target_sr is not None and sr != target_sr
    if resampled:
        audio = torchaudio.functional.resample(audio, sr, target_sr, **resample_kwargs)
    # Only resampling risks the Kaiser-sinc overshoot the guard exists for;
    # leave already-packed-as-is (no --target-sr) behavior untouched.
    clipped = False
    if resampled:
        peak = audio.abs().max().item()
        clipped = peak > CLIP_GUARD
        if clipped:
            audio = audio * (CLIP_GUARD / peak)
    return audio, sr, clipped


def pack(args):
    files = sorted(glob.glob(os.path.join(args.input, '**', '*.wav'), recursive=True))
    if not files:
        raise FileNotFoundError(f"在 {args.input} 找不到任何 .wav 檔案")

    N = len(files)
    print(f"找到 {N} 個檔案 → {args.output}")

    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    resample_kwargs = KAISER_FAST if args.quality == 'fast' else KAISER_BEST

    # 從第一個檔案取得 T (resample 後的長度) 與 out_sr
    sample, source_sr, _ = _load_maybe_resampled(files[0], args.target_sr, resample_kwargs)
    T = sample.size(1)
    out_sr = args.target_sr if args.target_sr is not None else source_sr

    if args.target_sr is not None and args.target_sr != source_sr:
        print(f"  Resampling {source_sr} Hz → {args.target_sr} Hz "
              f"(quality={args.quality}) while packing")

    bytes_per = 2 if dtype == torch.float16 else 4
    total_bytes = N * 2 * T * bytes_per
    size_str = (f"{total_bytes / 1024**3:.1f} GB" if total_bytes >= 1024**3
                else f"{total_bytes / 1024**2:.0f} MB")
    print(f"  SR={out_sr}, 每段 {T} samples, dtype={args.dtype}, 預估大小: {size_str}")

    data = torch.empty(N, 2, T, dtype=dtype)

    failed = []
    n_clip_guarded = 0
    for i, path in enumerate(tqdm.tqdm(files, desc="Packing")):
        try:
            audio, _, clipped = _load_maybe_resampled(path, args.target_sr, resample_kwargs)
            if audio.shape[0] < 2:
                raise ValueError(f"不是 2-channel WAV: {path}")
            if audio.shape[1] != T:
                raise ValueError(f"長度不符 (expected {T}, got {audio.shape[1]}): {path}")
            if clipped:
                n_clip_guarded += 1
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

    if n_clip_guarded:
        print(f"\n{n_clip_guarded} 個檔案在 resample 後超過 {CLIP_GUARD} peak，"
              f"已整組 (noisy, clean) 等比例縮小過（保留相對電平/SNR）。")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    print(f"儲存中 ({size_str})...")
    payload = {
        'data': data,          # (N, 2, T): ch0=noisy, ch1=clean
        'sr': out_sr,
        'n_samples': N,
        'segment_samples': T,
        'dtype': args.dtype,
        'source': args.input,
    }
    if args.target_sr is not None and args.target_sr != source_sr:
        payload['source_sr'] = source_sr
        payload['resample_quality'] = args.quality
    torch.save(payload, args.output)
    print(f"完成: {args.output}  ({N} pairs, {size_str})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='將 WAV pair 目錄打包成單一 .pt 檔，可選擇打包時順便 resample')
    parser.add_argument('--input', required=True,
                        help='WAV pair 目錄 (gen_dataset.py 的輸出, 遞迴掃描)')
    parser.add_argument('--output', required=True,
                        help='輸出 .pt 檔案路徑')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16'],
                        help='儲存精度 (float16 減半大小, 預設: float32)')
    parser.add_argument('--target-sr', type=int, default=None,
                        help='打包時順便 resample 到這個 rate (Hz)，例如 48k 母帶'
                            '→ 16000 給 RNNoise-ERB/GTCRN。省略則照來源 WAV 的 '
                            'rate 打包 (原本的行為)。')
    parser.add_argument('--quality', choices=['best', 'fast'], default='best',
                        help='--target-sr 使用的 Kaiser-window resample 品質 '
                            '(預設: best，跟 resample_dataset.py 同一組常數)。')
    args = parser.parse_args()
    pack(args)
