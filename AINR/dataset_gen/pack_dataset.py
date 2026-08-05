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

    --input 必須是 gen_dataset.py 產生的 pairs/ 目錄 (上一層要有 meta.json)。
    只接受 NNNNNN.wav+NNNNNN.json 完整配對。temp 檔 (tmp.NNNNNN.wav，crash
    留下的) 一律靜默忽略、不計入打包 (不是錯誤，繼續正常打包其餘樣本)。
    孤兒檔 (只有 .wav 或只有 .json)、sidecar JSON 損毀或 index 欄位對不上
    自己的檔名、找不到/版本不符的 meta.json，一律拒絕打包 (各有對應的
    --allow-* 旗標明確放行，見 README.md "Pack the generated dataset" 一節)。
    當 meta.json 的 contract_version 與本檔相符時 (versioned 模式)，還會
    fail-closed 核對: 實際完整編號集合必須精確等於 meta.json 宣告的
    batch_start_idx..batch_start_idx+batch_n_samples-1 (缺口/多出來的都拒絕，
    --allow-index-gaps 在此模式下不生效)，且每個 WAV 的原生 sample rate、
    通道數、長度、有限值 (無 NaN/Inf) 都要符合 meta.json 的宣告，任何一個
    樣本壞掉就整批拒絕打包，不會靜默排除。只有 --allow-unversioned-input
    (meta.json 缺失或版本不符) 才會退回舊的、遇到壞檔靜默排除+警告的行為，
    此時 --allow-index-gaps 才有意義。
"""

import argparse
import json
import os

import torch
import torchaudio
import tqdm

try:
    from .dataset import rms_dbfs
    from .gen_dataset import (
        AINR_DATASET_CONTRACT_VERSION,
        DatasetContractError,
        _list_complete_sample_indices,
        _sample_paths,
    )
    from .resample_dataset import KAISER_BEST, KAISER_FAST, CLIP_GUARD
except ImportError:
    from dataset import rms_dbfs
    from gen_dataset import (
        AINR_DATASET_CONTRACT_VERSION,
        DatasetContractError,
        _list_complete_sample_indices,
        _sample_paths,
    )
    from resample_dataset import KAISER_BEST, KAISER_FAST, CLIP_GUARD


def _resolve_meta_path(input_dir):
    """gen_dataset.py's --output layout is `<batch_dir>/pairs/NNNNNN.wav` +
    `<batch_dir>/meta.json` -- `--input` conventionally points at the
    `pairs` subdirectory (see this file's own usage examples), so the
    batch's meta.json is one level up."""
    return os.path.join(os.path.dirname(os.path.normpath(input_dir)), 'meta.json')


def _is_sha256_hex(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _load_batch_contract(input_dir, allow_unversioned):
    """Load and validate the parent batch's meta.json before packing
    anything. Returns (contract_version, config_hash, meta, versioned):

      - contract_version, config_hash: as recorded in meta.json (may be an
        OLD batch's values when contract_version mismatches under
        --allow-unversioned-input -- only missing meta.json entirely
        leaves both None, since then there is nothing to record).
      - meta: the full parsed meta.json dict, or None if it could not be
        found.
      - versioned: True only when meta.json exists AND its
        contract_version exactly matches AINR_DATASET_CONTRACT_VERSION.
        This gates the fail-closed inventory/sample-rate cross-checks in
        pack() -- an old or foreign contract_version gives no guarantee
        that meta's batch_start_idx/batch_n_samples/sr fields mean what
        the CURRENT contract says they mean, so those checks fall back to
        the weaker legacy behavior instead (see pack()).
    """
    meta_path = _resolve_meta_path(input_dir)
    if not os.path.isfile(meta_path):
        if allow_unversioned:
            print(f"WARNING: no meta.json found at {meta_path} -- packing "
                  "anyway (--allow-unversioned-input). contract_version/"
                  "config_hash will NOT be recorded in the packed payload, "
                  "and the fail-closed inventory/sample-rate checks are "
                  "skipped (no authoritative meta.json to check against).")
            return None, None, None, False
        raise DatasetContractError(
            f"pack refused: no meta.json found at {meta_path} (expected "
            f"one level above --input={input_dir}, gen_dataset.py's own "
            "--output/pairs layout). Packing input with no recorded "
            "contract/config means the packed payload can never be traced "
            "back to what produced it. Pass --allow-unversioned-input to "
            "pack anyway (e.g. for data that predates contract "
            "versioning)."
        )
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise DatasetContractError(
            f"pack refused: {meta_path} is not valid JSON ({e}). This "
            "batch's contract record is corrupt -- repair or regenerate it "
            "at the source. --allow-unversioned-input cannot help here: it "
            "waives a version/hash MISMATCH, but there is nothing to read "
            "at all from a file that doesn't parse."
        ) from e
    contract_version = meta.get('contract_version')
    config_hash = meta.get('config_hash')
    if contract_version != AINR_DATASET_CONTRACT_VERSION:
        if not allow_unversioned:
            raise DatasetContractError(
                f"pack refused: {meta_path} has contract_version="
                f"{contract_version!r}, this script expects "
                f"{AINR_DATASET_CONTRACT_VERSION!r}. Pass "
                "--allow-unversioned-input to pack anyway (e.g. for data "
                "that predates the current contract version)."
            )
        print(f"WARNING: {meta_path} has contract_version="
              f"{contract_version!r} (expected "
              f"{AINR_DATASET_CONTRACT_VERSION!r}) -- packing anyway "
              "(--allow-unversioned-input). Fail-closed inventory/sample-"
              "rate checks are skipped; falling back to legacy "
              "gap-tolerant packing.")
        return contract_version, config_hash, meta, False
    if not _is_sha256_hex(config_hash):
        raise DatasetContractError(
            f"pack refused: {meta_path}'s config_hash is not a well-formed "
            f"64-character sha256 hex digest ({config_hash!r}). This "
            "meta.json is corrupt or was hand-edited and its declared "
            "inventory cannot be trusted."
        )
    return contract_version, config_hash, meta, True


def _validate_sidecars(input_dir, indices):
    """Parse every complete index's NNNNNN.json sidecar and confirm its own
    'index' field matches its filename. Always enforced, in every mode
    (versioned or --allow-unversioned-input): unlike a contract/config
    mismatch, a sidecar that fails to parse or disagrees with its own
    filename is plain file-integrity corruption, never a legitimate
    legacy-data scenario -- unlike an orphan (see the caller), it also
    cannot be silently skipped, since a mismatched 'index' field means we
    no longer know which sample this file actually describes."""
    for idx in indices:
        _, json_path = _sample_paths(input_dir, idx)
        try:
            with open(json_path) as f:
                sidecar = json.load(f)
        except (OSError, ValueError) as e:
            raise DatasetContractError(
                f"pack refused: {json_path} is not valid JSON ({e}). This "
                "metadata sidecar is corrupt -- repair or regenerate it at "
                "the source."
            ) from e
        if sidecar.get('index') != idx:
            raise DatasetContractError(
                f"pack refused: {json_path}'s own 'index' field is "
                f"{sidecar.get('index')!r}, expected {idx} (from its own "
                "filename). This sidecar does not describe the sample "
                "it's named after."
            )


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
    contract_version, config_hash, meta, versioned = _load_batch_contract(
        args.input, args.allow_unversioned_input)

    # Only complete (NNNNNN.wav + NNNNNN.json) pairs, never a stray
    # tmp.NNNNNN.wav or an orphan half-pair -- a naive recursive
    # glob('*.wav') would also match an in-progress/crashed write, which is
    # exactly the release-blocking bug this replaces (a temp file from a
    # generation-time crash silently becoming a "real" training sample at
    # pack time). Orphans are always a hard error (never salvageable --
    # regenerate/repair at the source with gen_dataset.py --repair-resume).
    complete_indices, orphan_wavs, orphan_jsons = _list_complete_sample_indices(args.input)
    if not complete_indices:
        raise FileNotFoundError(f"在 {args.input} 找不到任何完整的 NNNNNN.wav+NNNNNN.json pair")
    if orphan_wavs or orphan_jsons:
        raise DatasetContractError(
            f"pack refused: {args.input} has incomplete sample(s) -- "
            f"{len(orphan_wavs)} WAV(s) with no metadata sidecar "
            f"{orphan_wavs}, {len(orphan_jsons)} JSON sidecar(s) with no "
            f"audio {orphan_jsons}. These cannot be packed (no salvageable "
            "content) and must not be silently skipped without telling "
            "you -- run gen_dataset.py --resume --repair-resume on the "
            "source directory to remove them, or delete them manually."
        )
    # Cheap fail-fast: catch corrupt/mismatched sidecar metadata before
    # spending any time loading WAV audio.
    _validate_sidecars(args.input, complete_indices)

    expected_sr = None
    if versioned:
        # Fail-closed: the actual on-disk complete-index set must equal
        # EXACTLY what meta.json itself declares this batch to contain --
        # not merely internally contiguous. A batch still mid-generation
        # (interrupted before its last round's meta.json save) or one
        # tampered with after the fact both fail this, by design: either
        # way the batch is not in the state meta.json claims it is.
        expected_start = meta.get('batch_start_idx')
        expected_n = meta.get('batch_n_samples')
        expected_sr = meta.get('sr')
        if expected_start is None or expected_n is None or expected_sr is None:
            raise DatasetContractError(
                f"pack refused: {_resolve_meta_path(args.input)} is missing "
                "batch_start_idx/batch_n_samples/sr -- cannot validate this "
                f"contract_version={AINR_DATASET_CONTRACT_VERSION!r} batch's "
                "inventory."
            )
        # Type-check before doing arithmetic on them -- a hand-edited or
        # corrupt-but-still-parseable meta.json (e.g. batch_start_idx as a
        # string) must fail closed with a clean DatasetContractError, not
        # crash with a raw TypeError from `expected_start + expected_n` or
        # range() below.
        if (not isinstance(expected_start, int) or isinstance(expected_start, bool)
                or not isinstance(expected_n, int) or isinstance(expected_n, bool)
                or not isinstance(expected_sr, int) or isinstance(expected_sr, bool)):
            raise DatasetContractError(
                f"pack refused: {_resolve_meta_path(args.input)}'s "
                "batch_start_idx/batch_n_samples/sr must all be integers -- "
                f"got {expected_start!r}, {expected_n!r}, {expected_sr!r}. "
                "This meta.json is corrupt or was hand-edited."
            )
        expected_indices = list(range(expected_start, expected_start + expected_n))
        if complete_indices != expected_indices:
            missing = sorted(set(expected_indices) - set(complete_indices))
            extra = sorted(set(complete_indices) - set(expected_indices))
            raise DatasetContractError(
                f"pack refused: {args.input}'s on-disk complete sample "
                f"indices don't match "
                f"{_resolve_meta_path(args.input)}'s declared inventory -- "
                f"meta.json declares {expected_n} sample(s) "
                f"({expected_start}..{expected_start + expected_n - 1}), "
                f"disk actually has {len(complete_indices)} "
                f"({complete_indices[0]}..{complete_indices[-1]}). "
                f"missing={missing[:20]}{'...' if len(missing) > 20 else ''}, "
                f"extra={extra[:20]}{'...' if len(extra) > 20 else ''}. "
                "This usually means generation is still in progress or was "
                "interrupted mid-round -- run gen_dataset.py --resume on "
                "the source directory to finish and re-sync meta.json "
                "first -- or the directory was manually altered after the "
                "fact. There is no partial-pack option for a versioned "
                "batch that fails this check; --allow-unversioned-input "
                "packs from disk without this guarantee if you accept that "
                "trade-off."
            )
    else:
        # Unversioned/foreign-contract input: no authoritative inventory to
        # check against -- fall back to the original internal-contiguity
        # check (still refuses by default; --allow-index-gaps to override).
        if complete_indices != list(range(complete_indices[0], complete_indices[-1] + 1)):
            missing = sorted(
                set(range(complete_indices[0], complete_indices[-1] + 1))
                - set(complete_indices)
            )
            if not args.allow_index_gaps:
                raise DatasetContractError(
                    f"pack refused: {args.input}'s complete sample indices "
                    f"have gap(s) -- missing {missing[:20]}"
                    f"{'...' if len(missing) > 20 else ''} "
                    f"(range {complete_indices[0]}..{complete_indices[-1]}, "
                    f"{len(complete_indices)} present). A gap usually means "
                    "samples were removed (e.g. --repair-resume) without "
                    "ever being regenerated. Pass --allow-index-gaps if "
                    "this is a deliberately curated subset."
                )
            print(f"WARNING: {len(missing)} index gap(s) in {args.input} "
                  "(--allow-index-gaps) -- packing the present samples only.")

    files = [_sample_paths(args.input, idx)[0] for idx in complete_indices]
    N = len(files)
    print(f"找到 {N} 個完整 pair → {args.output}")

    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    resample_kwargs = KAISER_FAST if args.quality == 'fast' else KAISER_BEST

    # 從第一個檔案取得 T (resample 後的長度)、out_sr 與 source_sr (所有檔案在
    # 沒有 --target-sr 時都必須跟這個一致 -- 見下方 per-file 檢查)
    sample, source_sr, _ = _load_maybe_resampled(files[0], args.target_sr, resample_kwargs)
    T = sample.size(1)
    out_sr = args.target_sr if args.target_sr is not None else source_sr

    if versioned and expected_sr is not None and source_sr != expected_sr:
        raise DatasetContractError(
            f"pack refused: {files[0]}'s native sample rate ({source_sr} "
            f"Hz) doesn't match {_resolve_meta_path(args.input)}'s "
            f"declared sr ({expected_sr} Hz)."
        )

    if args.target_sr is not None and args.target_sr != source_sr:
        print(f"  Resampling {source_sr} Hz → {args.target_sr} Hz "
              f"(quality={args.quality}) while packing")

    bytes_per = 2 if dtype == torch.float16 else 4
    total_bytes = N * 2 * T * bytes_per
    size_str = (f"{total_bytes / 1024**3:.1f} GB" if total_bytes >= 1024**3
                else f"{total_bytes / 1024**2:.0f} MB")
    print(f"  SR={out_sr}, 每段 {T} samples, dtype={args.dtype}, 預估大小: {size_str}")

    data = torch.empty(N, 2, T, dtype=dtype)
    # Measured AFTER any --target-sr resample + clip-guard AND after the
    # cast to the packed dtype (`data[i]`) -- not on the pre-cast float32
    # tensor, and not carried forward from generation-time metadata: a
    # downsample does not exactly preserve the requested level/SNR
    # (narrowband content is the worst case -- e.g. a 1 kHz-speech/12 kHz-
    # noise pair generated at ~0 dB SNR measures ~48 dB after a 48k->16k
    # downsample, since the noise energy above the new Nyquist is simply
    # gone). This is the packed payload's own record of what a consumer
    # ACTUALLY trains on (byte-for-byte, including any float16 rounding),
    # independent of whatever requested_level_dbfs/snr_db a NNNNNN.json
    # sidecar claims at the SOURCE rate -- see README.md's "48 kHz source,
    # 16 kHz pack" section for the full caveat and when to prefer native
    # generation instead (gen_dataset.py --sample-rate 16000) for exact
    # fidelity.
    effective_rms_dbfs = torch.empty(N, 2, dtype=torch.float32)

    # check_sr: what each file's NATIVE rate must agree with. In versioned
    # mode this is meta.json's own declared `sr` (already proven == file
    # 0's rate above) -- checked unconditionally, EVEN when --target-sr is
    # given, since a versioned batch is by construction single-native-rate
    # (gen_dataset.py only ever generates one rate per invocation);
    # --target-sr only reconciles the OUTPUT rate, it must not become a
    # blanket excuse to stop verifying the INPUT actually belongs to this
    # batch (a file from a different run/rate mixed in here is exactly the
    # kind of tampering this whole fail-closed rewrite targets). In legacy/
    # unversioned mode there is no meta.json to trust, so it falls back to
    # file 0's rate and stays gated on `--target-sr is None`, exactly as
    # before this fail-closed rewrite (a legacy/curated batch may
    # legitimately mix native rates when --target-sr reconciles them).
    check_sr = expected_sr if versioned else source_sr
    check_native_rate_always = versioned

    failed_positions = set()
    n_clip_guarded = 0
    for i, path in enumerate(tqdm.tqdm(files, desc="Packing")):
        try:
            audio, sr, clipped = _load_maybe_resampled(path, args.target_sr, resample_kwargs)
            if audio.shape[0] < 2:
                raise ValueError(f"不是 2-channel WAV (got {audio.shape[0]} channel(s))")
            if audio.shape[1] != T:
                raise ValueError(f"長度不符 (expected {T}, got {audio.shape[1]})")
            if not torch.isfinite(audio).all():
                raise ValueError("含有非 finite 值 (NaN/Inf)")
            if (check_native_rate_always or args.target_sr is None) and sr != check_sr:
                # Without a forced resample, every file's NATIVE rate must
                # agree with the expected one, or `out_sr` (taken from file
                # 0 only) silently mislabels a subset of `data`'s actual
                # rate. In versioned mode this is checked even WITH
                # --target-sr (see check_native_rate_always above) -- the
                # resample reconciles the OUTPUT, it doesn't excuse a file
                # whose INPUT rate disagrees with what this batch is
                # supposed to contain.
                reason = ("no --target-sr given, so every file must already "
                          "share one native rate" if args.target_sr is None
                          else "a contract-versioned batch must be single-"
                          "native-rate regardless of --target-sr")
                raise ValueError(
                    f"sample rate {sr} Hz != "
                    f"{'meta.json 宣告的' if versioned else '第一個檔案的'} "
                    f"{check_sr} Hz ({reason})"
                )
        except Exception as e:
            if versioned:
                raise DatasetContractError(
                    "pack refused: malformed sample found in a "
                    f"contract-versioned batch -- {path}: {e}. A validated "
                    "batch should never contain a malformed sample, so "
                    "this stops packing immediately rather than silently "
                    "excluding it (which would also reopen an index gap "
                    "the earlier inventory check just closed). Re-run with "
                    "--allow-unversioned-input only if you accept the "
                    "weaker, exclude-and-warn legacy behavior instead."
                ) from e
            failed_positions.add(i)
            print(f"\n警告 (unversioned/legacy 模式): 讀取失敗，已從 dataset "
                  f"移除: {path}: {e}")
            data[i] = 0  # 填 0，之後過濾
            effective_rms_dbfs[i] = 0.0
            continue
        if clipped:
            n_clip_guarded += 1
        data[i] = audio.to(dtype)
        effective_rms_dbfs[i, 0] = rms_dbfs(data[i, 0].float())   # noisy
        effective_rms_dbfs[i, 1] = rms_dbfs(data[i, 1].float())   # clean

    if failed_positions:
        print(f"\n{len(failed_positions)} 個檔案讀取失敗，已從 dataset 移除。")
        good_mask = torch.ones(N, dtype=torch.bool)
        for idx in failed_positions:
            good_mask[idx] = False
        data = data[good_mask]
        effective_rms_dbfs = effective_rms_dbfs[good_mask]
        complete_indices = [
            idx for pos, idx in enumerate(complete_indices) if pos not in failed_positions
        ]
        N = data.size(0)

    if n_clip_guarded:
        print(f"\n{n_clip_guarded} 個檔案在 resample 後超過 {CLIP_GUARD} peak，"
              f"已整組 (noisy, clean) 等比例縮小過（保留相對電平/SNR）。")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    print(f"儲存中 ({size_str})...")
    payload = {
        'data': data,          # (N, 2, T): ch0=noisy, ch1=clean
        'effective_rms_dbfs': effective_rms_dbfs,  # (N, 2): ch0=noisy, ch1=clean;
                                                    # measured post-resample+post-cast --
                                                    # see the comment where this is computed
        'sample_indices': complete_indices,  # original NNNNNN for each row of `data`,
                                              # same order -- traces a packed row back to
                                              # its source pairs/NNNNNN.wav
        'contract_version': contract_version,  # from the source batch's meta.json;
                                                # None only if meta.json was entirely
                                                # missing (--allow-unversioned-input);
                                                # a mismatched-but-present version's
                                                # OLD value is still carried through
        'config_hash': config_hash,            # ditto
        'sr': out_sr,
        'n_samples': N,
        'segment_samples': T,
        'dtype': args.dtype,
        'source': args.input,
    }
    if args.target_sr is not None and args.target_sr != source_sr:
        payload['source_sr'] = source_sr
        payload['resample_quality'] = args.quality
    # Atomic (temp + os.replace): a payload can be many GB, and a crash/kill
    # partway through torch.save() must never leave a truncated file that
    # LOOKS like a complete packed dataset at the final --output path.
    tmp_output = args.output + '.tmp'
    torch.save(payload, tmp_output)
    os.replace(tmp_output, args.output)
    print(f"完成: {args.output}  ({N} pairs, {size_str})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='將 WAV pair 目錄打包成單一 .pt 檔，可選擇打包時順便 resample')
    parser.add_argument('--input', required=True,
                        help='gen_dataset.py 的 pairs/ 目錄 (只接受 NNNNNN.wav+'
                             'NNNNNN.json 完整配對；temp 檔靜默忽略，orphan/'
                             '損毀 sidecar 一律拒絕，見本檔開頭 docstring)')
    parser.add_argument('--output', required=True,
                        help='輸出 .pt 檔案路徑')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16'],
                        help='儲存精度 (float16 減半大小, 預設: float32)')
    parser.add_argument('--target-sr', type=int, default=None,
                        help='打包時順便 resample 到這個 rate (Hz)，例如 48k 母帶'
                            '→ 16000 給 RNNoise-ERB/GTCRN。省略則照來源 WAV 的 '
                            'rate 打包 (要求所有來源檔案原生 rate 一致)。')
    parser.add_argument('--quality', choices=['best', 'fast'], default='best',
                        help='--target-sr 使用的 Kaiser-window resample 品質 '
                            '(預設: best，跟 resample_dataset.py 同一組常數)。')
    parser.add_argument('--allow-unversioned-input', action='store_true',
                        help='--input 上一層找不到 meta.json，或 '
                             'contract_version 不符時，預設拒絕打包；此旗標'
                             '明確允許 (例如舊版無 sidecar 的資料)。同時退回'
                             '舊的、遇壞檔靜默排除+警告的行為 (跳過 fail-'
                             'closed 的 inventory/sample-rate/finite-value '
                             '核對，因為版本不符的 meta.json 無法信任)。')
    parser.add_argument('--allow-index-gaps', action='store_true',
                        help='完整 pair 的編號序列有缺口時，預設拒絕打包；'
                             '此旗標明確允許 (例如刻意篩選過的子集合)。'
                             '只在 --allow-unversioned-input 模式下有意義'
                             '——meta.json 版本相符時一律要求精確匹配其宣告'
                             '的 batch_start_idx/batch_n_samples，此旗標'
                             '不生效。')
    args = parser.parse_args()
    pack(args)
