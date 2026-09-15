# -*- coding: utf-8 -*-
"""
把 WAV pair 目錄打包成單一 .pt 檔或固定筆數的 .pt shards，消除訓練時的
逐檔 I/O 開銷。
可選擇在打包當下順便 resample，這樣 48 kHz 母帶可以直接產生 16 kHz 的
packed 檔，不需要先用 resample_dataset.py 生一份中繼 WAV 目錄。

用法:
    python pack_dataset.py --input data_48k/pairs --output data_48k/packed.pt
    python pack_dataset.py --input data_48k/pairs --output data_48k/packed.pt --dtype float16

    # 大資料集直接切 shard；--output 在這個模式是目錄
    python pack_dataset.py --input data_48k/pairs --output data_48k/packed \
        --shard-clips 512 --dtype float16

    # 打包時直接 resample 成另一個 rate (跳過中繼 WAV, 少一次 int16 write/read)
    python pack_dataset.py --input data_48k/pairs --output data_16k/packed.pt \
        --target-sr 16000 --dtype float16

訓練:
    python train.py --config config.ini --packed-data data/packed.pt
    python train.py --config config.ini --packed-dir data/packed --mmap

輸入:
    --input 是一個裝著 2-channel (ch0=noisy, ch1=clean) WAV 的目錄，就這樣。
    沒有 meta.json、沒有 NNNNNN.json sidecar、沒有 contract version —— 目錄
    裡有什麼 wav 就打包什麼，所以複製過、resample 過、手動刪過幾筆的目錄一樣
    可以直接打包。

    檔名規則只有兩條:
      * `NNNNNN.wav` (純數字檔名) 才算樣本，依編號排序打包。
      * `tmp.*.wav` (中斷的寫入) 一律略過，不算樣本也不是錯誤。

    每個檔案仍然逐一驗證: 必須是 2 channel、長度一致、無 NaN/Inf、
    (沒給 --target-sr 時) 原生取樣率一致。任何一個不符就整批拒絕打包，
    不會靜默跳過 —— 打包階段做這件事不需要任何 JSON。

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

try:
    from .dataset import rms_dbfs
    from .gen_dataset import (
        DatasetContractError,
        _list_sample_indices,
        _sample_wav_path,
    )
    from .resample_dataset import KAISER_BEST, KAISER_FAST, CLIP_GUARD
except ImportError:
    from dataset import rms_dbfs
    from gen_dataset import (
        DatasetContractError,
        _list_sample_indices,
        _sample_wav_path,
    )
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


def _stage_payload(clips, levels, sample_indices, temporary, header):
    """Serialize one complete-but-unpublished payload to ``temporary``."""
    data = torch.stack(clips)
    effective_rms_dbfs = torch.stack(levels)
    torch.save({
        'data': data,          # (N, 2, T): ch0=noisy, ch1=clean
        'effective_rms_dbfs': effective_rms_dbfs,
        'sample_indices': list(sample_indices),
        'n_samples': len(sample_indices),
        **header,
    }, temporary)


def _shard_outputs(output_dir, overwrite):
    """Prepare an output directory without mixing old and new shard sets."""
    os.makedirs(output_dir, exist_ok=True)
    all_pt = sorted(glob.glob(os.path.join(output_dir, '*.pt')))
    existing = sorted(glob.glob(os.path.join(output_dir, 'shard_*.pt')))
    unrelated = sorted(set(all_pt) - set(existing))
    if unrelated:
        raise FileExistsError(
            f"{output_dir} contains non-shard .pt files, e.g. "
            f"{os.path.basename(unrelated[0])}. A trainer loads every .pt in "
            "the directory, so use an empty shard directory."
        )
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} shard(s), e.g. "
            f"{os.path.basename(existing[0])}. Pass --overwrite to replace "
            "them only after the new pack succeeds."
        )
    stale = sorted(glob.glob(os.path.join(output_dir, 'shard_*.pt.tmp')))
    for path in stale:
        os.remove(path)
    if stale:
        print(f"  removed {len(stale)} stale temporary shard(s)")
    return existing


def pack(args):
    # `NNNNNN.wav` only -- a stray `tmp.NNNNNN.wav` from a killed generation
    # run is invisible here, which is exactly the point of the generator
    # writing through a temp name: a visible file is a finished file.
    indices = _list_sample_indices(args.input)
    if not indices:
        raise FileNotFoundError(
            f"在 {args.input} 找不到任何 NNNNNN.wav (檔名必須是純數字)")

    if indices != list(range(indices[0], indices[-1] + 1)):
        missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))
        if not args.allow_index_gaps:
            raise DatasetContractError(
                f"pack refused: {args.input} 的樣本編號有缺口 -- 缺 "
                f"{missing[:20]}{'...' if len(missing) > 20 else ''} "
                f"(範圍 {indices[0]}..{indices[-1]}, 實際 {len(indices)} 筆)。"
                "缺口通常代表有樣本被刪掉但沒重生。確定是刻意篩選過的子集合"
                "就加 --allow-index-gaps。"
            )
        print(f"WARNING: {len(missing)} index gap(s) in {args.input} "
              "(--allow-index-gaps) -- packing the present samples only.")

    files = [_sample_wav_path(args.input, idx) for idx in indices]
    N = len(files)
    shard_clips = getattr(args, 'shard_clips', None)
    if shard_clips is not None and shard_clips <= 0:
        raise ValueError(f"--shard-clips must be positive, got {shard_clips}")
    sharded = shard_clips is not None
    print(f"找到 {N} 個 WAV → {args.output}")

    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    resample_kwargs = KAISER_FAST if args.quality == 'fast' else KAISER_BEST

    # 從第一個檔案取得 T (resample 後的長度)、out_sr 與 source_sr。每個檔案都
    # 必須跟這個第一個檔案一致 -- 見下方 per-file 檢查。
    sample, source_sr, _ = _load_maybe_resampled(files[0], args.target_sr, resample_kwargs)
    T = sample.size(1)
    out_sr = args.target_sr if args.target_sr is not None else source_sr

    bytes_per = 2 if dtype == torch.float16 else 4
    total_bytes = N * 2 * T * bytes_per
    size_str = (f"{total_bytes / 1024**3:.1f} GB" if total_bytes >= 1024**3
                else f"{total_bytes / 1024**2:.0f} MB")
    print(f"  SR={out_sr}, 每段 {T} samples, dtype={args.dtype}, 預估大小: {size_str}")

    # Collect native rates up front so every shard carries the same corpus-wide
    # source-rate declaration. Reading headers does not load waveform payloads
    # and keeps peak RAM bounded by --shard-clips.
    source_srs = set()
    for path in files:
        try:
            source_srs.add(int(torchaudio.info(path).sample_rate))
        except Exception as e:
            raise DatasetContractError(
                f"pack refused: cannot read WAV header {path}: {e}"
            ) from e

    # Measured AFTER any --target-sr resample + clip-guard AND after the
    # cast to the packed dtype -- not on the pre-cast float32
    # tensor: a downsample does not exactly preserve the requested level/SNR
    # (narrowband content is the worst case -- e.g. a 1 kHz-speech/12 kHz-
    # noise pair generated at ~0 dB SNR measures ~48 dB after a 48k->16k
    # downsample, since the noise energy above the new Nyquist is simply
    # gone). This is the packed payload's own record of what a consumer
    # ACTUALLY trains on, byte-for-byte, including any float16 rounding --
    # see README.md's "48 kHz source, 16 kHz pack" section for the full
    # caveat and when to prefer native generation instead
    # (gen_dataset.py --sample-rate 16000) for exact fidelity.
    n_clip_guarded = 0
    header = {
        'sr': out_sr,
        'segment_samples': T,
        'dtype': args.dtype,
        'source': args.input,
    }
    if args.target_sr is not None and source_srs != {out_sr}:
        if len(source_srs) == 1:
            header['source_sr'] = next(iter(source_srs))
        else:
            header['source_srs'] = sorted(source_srs)
        header['resample_quality'] = args.quality

    existing = []
    if sharded:
        existing = _shard_outputs(
            args.output, getattr(args, 'overwrite', False)
        )
    else:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)

    clips = []
    levels = []
    packed_indices = []
    staged = []
    shard_index = 0
    limit = shard_clips if sharded else N

    def flush():
        nonlocal clips, levels, packed_indices, shard_index
        if not clips:
            return
        final = (os.path.join(args.output, f'shard_{shard_index:05d}.pt')
                 if sharded else args.output)
        temporary = final + '.tmp'
        # Register the path before torch.save(): a disk-full/interrupted save
        # may leave a partial file even though the call raises.
        staged.append((temporary, final))
        _stage_payload(
            clips, levels, packed_indices, temporary, header
        )
        shard_index += 1
        clips, levels, packed_indices = [], [], []

    try:
        for index, path in zip(indices, tqdm.tqdm(files, desc="Packing")):
            try:
                audio, sr, clipped = _load_maybe_resampled(
                    path, args.target_sr, resample_kwargs
                )
                if audio.shape[0] != 2:
                    raise ValueError(
                        f"不是 2-channel WAV (got {audio.shape[0]} channel(s))"
                    )
                if audio.shape[1] != T:
                    raise ValueError(
                        f"長度不符 (expected {T}, got {audio.shape[1]})"
                    )
                if not torch.isfinite(audio).all():
                    raise ValueError("含有非 finite 值 (NaN/Inf)")
                if args.target_sr is None and sr != source_sr:
                    raise ValueError(
                        f"sample rate {sr} Hz != 第一個檔案的 {source_sr} Hz "
                        "(沒給 --target-sr 時所有檔案的原生取樣率必須一致)"
                    )
            except Exception as e:
                # Never silently exclude: a bad file means the directory is
                # not what it looks like, and dropping it would also open an
                # index gap the check above just closed.
                raise DatasetContractError(
                    f"pack refused: {path}: {e}. 打包會整批停在這裡而不是靜默跳過"
                    "這一筆 —— 請修掉或刪掉這個檔案 (刪掉的話用 "
                    "--allow-index-gaps 打包剩下的)。"
                ) from e
            if clipped:
                n_clip_guarded += 1
            packed = audio.to(dtype)
            clips.append(packed)
            levels.append(torch.tensor([
                rms_dbfs(packed[0].float()),
                rms_dbfs(packed[1].float()),
            ], dtype=torch.float32))
            packed_indices.append(index)
            if len(clips) == limit:
                flush()
        flush()
    except BaseException:
        for temporary, _final in staged:
            if os.path.exists(temporary):
                os.remove(temporary)
        raise

    if n_clip_guarded:
        print(f"\n{n_clip_guarded} 個檔案在 resample 後超過 {CLIP_GUARD} peak，"
              f"已整組 (noisy, clean) 等比例縮小過（保留相對電平/SNR）。")
    resampled_source_srs = sorted(sr for sr in source_srs if sr != out_sr)
    if resampled_source_srs:
        print(f"  Resampled native rate(s) {sorted(source_srs)} → {out_sr} Hz "
              f"(quality={args.quality}) while packing")

    # Publish only after every source WAV validates and every shard serializes.
    # With --overwrite, the old usable set remains in place until this point.
    for path in existing:
        os.remove(path)
    for temporary, final in staged:
        os.replace(temporary, final)
    if sharded:
        print(f"完成: {len(staged)} shards / {N} pairs → {args.output} "
              f"({size_str} total)")
    else:
        print(f"完成: {args.output}  ({N} pairs, {size_str})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='將 WAV pair 打包成單一 .pt 或固定筆數 shards，可順便 resample')
    parser.add_argument('--input', required=True,
                        help='裝著 2-channel (noisy, clean) WAV 的目錄。只認 '
                             'NNNNNN.wav，tmp.*.wav 自動略過')
    parser.add_argument('--output', required=True,
                        help='輸出 .pt 路徑；使用 --shard-clips 時改為輸出目錄')
    parser.add_argument('--shard-clips', type=int, default=None,
                        help='每個 shard 的固定 clip 上限；設定後輸出為 '
                             'shard_00000.pt...，省略則維持單一 .pt')
    parser.add_argument('--overwrite', action='store_true',
                        help='shard 模式下，在新 pack 全部驗證並寫完後取代輸出'
                             '目錄內既有的 shard_*.pt')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16'],
                        help='儲存精度 (float16 減半大小, 預設: float32)')
    parser.add_argument('--target-sr', type=int, default=None,
                        help='打包時順便 resample 到這個 rate (Hz)，例如 48k 母帶'
                            '→ 16000 給 RNNoise-ERB/GTCRN。省略則照來源 WAV 的 '
                            'rate 打包 (要求所有來源檔案原生 rate 一致)。')
    parser.add_argument('--quality', choices=['best', 'fast'], default='best',
                        help='--target-sr 使用的 Kaiser-window resample 品質 '
                            '(預設: best，跟 resample_dataset.py 同一組常數)。')
    parser.add_argument('--allow-index-gaps', action='store_true',
                        help='樣本編號有缺口時，預設拒絕打包 (通常代表有樣本'
                             '被刪掉沒重生)；此旗標明確允許 (例如刻意篩選過的'
                             '子集合)。')
    args = parser.parse_args()
    pack(args)
