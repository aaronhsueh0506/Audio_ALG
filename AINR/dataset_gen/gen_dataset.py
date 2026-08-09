# -*- coding: utf-8 -*-
"""
Offline pre-generation of training data (WAV pair mode).
Saves augmented (noisy, clean) WAV pairs for flexible downstream use.

This is the standalone, model-independent dataset generator (AINR).
Choose one generation sample rate per run through config.ini's `[signal] sr`
or `--sample-rate`. Use 16 kHz for RNNoise-ERB/GTCRN and 48 kHz for
DeepFilterNet2. See README.md.

Output is WAV and nothing else:

    <output>/pairs/000000.wav   2-channel, ch0 = noisy, ch1 = clean
    <output>/pairs/000001.wav
    ...

No meta.json, no per-sample JSON sidecar. pack_dataset.py packs whatever
`NNNNNN.wav` files it finds, so a directory stays packable after being copied,
resampled or trimmed by hand. See the note above DatasetContractError for what
that costs and what it does not.

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


# The batch is exactly what is in pairs/: `NNNNNN.wav`, one 2-channel file per
# sample, and nothing else. There is no meta.json, no per-sample JSON sidecar
# and no contract version -- a directory of WAVs is the whole contract, so a
# batch stays packable after being copied, resampled, filtered by hand, or
# produced by something other than this script.
#
# What that costs, stated plainly, because these WERE real checks:
#   * A config.ini/--sample-rate edit between two --resume runs is no longer
#     detected -- nothing records which config produced which sample, so two
#     distributions can be mixed into one directory with no error. Resume into
#     a directory only with the config that started it.
#   * The generation seed is printed, not persisted, so a batch is no longer
#     self-describing enough to reproduce.
# What survives without any JSON, because it never depended on one:
#   * A visible `NNNNNN.wav` is always complete -- every write goes to
#     `tmp.NNNNNN.wav` first and is renamed into place (see _save_pair_atomic),
#     and a `tmp.` file is invisible to every scan here. The sidecar's real
#     structural job was to mark a sample finished; the atomic rename already
#     does that.
#   * pack_dataset.py still re-checks channel count, length, sample rate and
#     finiteness on every file it packs.


class DatasetContractError(RuntimeError):
    """A refusal from --resume's scan of pairs/ (a gap in the numbering, or a
    non-empty directory without --resume)."""


def _sample_wav_path(pairs_dir, index):
    return os.path.join(pairs_dir, f"{index:06d}.wav")


def _tmp_path(final_path):
    """Same-directory temp path for `final_path`, named `tmp.<basename>`
    (prefix, not suffix). Two things need this exact shape:
      1. torchaudio's soundfile backend infers the save format from the
         LAST '.'-separated segment of a string path and ignores its own
         `format=` kwarg in that case (a soundfile-backend quirk) -- so the
         temp path must still literally END in '.wav' for _save_pair_atomic
         to work at all; a '.wav.tmp' suffix breaks that.
      2. _list_sample_indices()'s glob-based scan must never mistake a
         temp/in-progress file for a real sample: `tmp.NNNNNN.wav`'s
         os.path.splitext() stem is `tmp.NNNNNN` (not all-digit), so it is
         correctly excluded, same as a '*.wav.tmp' suffix would have been.
    """
    d = os.path.dirname(final_path)
    b = os.path.basename(final_path)
    return os.path.join(d, f"tmp.{b}")


def _save_pair_atomic(pairs_dir, index, noisy, clean, sr):
    """Write pairs/NNNNNN.wav via temp-file + os.replace: a crash mid-write
    can never leave a partially-written file visible at the final path
    (os.replace is an atomic rename on POSIX; a tmp file orphaned by a
    crash before the replace sits under a name no `*.wav` glob ever
    matches, so it can never be mistaken for a real sample). With the JSON
    sidecar gone this rename is the ONLY completion marker a sample has,
    which is why nothing here may ever write directly to the final path."""
    wav_path = _sample_wav_path(pairs_dir, index)
    tmp_path = _tmp_path(wav_path)
    pair = torch.stack([noisy, clean], dim=0)
    torchaudio.save(tmp_path, pair, sr, bits_per_sample=16)
    os.replace(tmp_path, wav_path)


def _list_sample_indices(pairs_dir):
    """Sorted indices of the `NNNNNN.wav` files in `pairs_dir`.

    A digit-only stem is the whole membership test, so `tmp.NNNNNN.wav`
    (an in-progress or crashed write -- see _tmp_path) and any hand-added
    file with a name of its own are both ignored rather than packed.
    Shared by gen_dataset.py's --resume bookkeeping and pack_dataset.py.
    """
    indices = set()
    for path in glob.glob(os.path.join(pairs_dir, '*.wav')):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.isdigit():
            indices.add(int(stem))
    return sorted(indices)


def _list_stale_temp_files(pairs_dir):
    """Leftover `tmp.*.wav` from a killed run. Harmless (no scan sees them),
    but they occupy disk, so --repair-resume offers to delete them."""
    return sorted(glob.glob(os.path.join(pairs_dir, 'tmp.*.wav')))


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

    pairs_dir = os.path.join(args.output, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)

    # An existing batch's own LOWEST index is its start, and it wins over
    # config/CLI: a shard created with `--start-idx N` stays anchored at N on
    # every later resume without the caller having to repeat the flag. An
    # explicitly supplied conflicting value is almost certainly the wrong
    # output directory, so fail closed rather than renumber silently. (This
    # used to be read back out of meta.json's batch_start_idx; the directory
    # itself carries the same fact.)
    existing_indices = _list_sample_indices(pairs_dir) if args.resume else []
    if existing_indices:
        recorded_base = existing_indices[0]
        if args.start_idx is not None and base_idx != recorded_base:
            raise DatasetContractError(
                f"--resume refused: explicit --start-idx={base_idx} "
                f"conflicts with the lowest sample already in {pairs_dir} "
                f"({recorded_base:06d}.wav). Omit --start-idx when resuming, "
                "or pass exactly that value."
            )
        base_idx = recorded_base

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

    # ---- The pairs/ scan runs BEFORE constructing DNS4Dataset (RIR glob/
    # cache I/O) and the one-sample profiling pass below, so a refusal exits
    # immediately instead of after paying for both. ----
    # 起始檔名編號 base_idx 是本批的「地板」:
    #   --resume: 掃 pairs/ 取最大編號 +1，但不低於 base_idx。
    #   無 --resume: 直接從 base_idx 起 (擴增, 不掃碟; start_idx 已保證不洗舊檔)。
    sample_count = base_idx
    done = 0
    max_idx = -1
    if args.resume:
        stale = _list_stale_temp_files(pairs_dir)
        if stale and args.repair_resume:
            print(f"Repairing: removing {len(stale)} leftover temp file(s) "
                  f"from an interrupted run")
            for path in stale:
                os.remove(path)
        elif stale:
            print(f"Note: {len(stale)} leftover tmp.*.wav file(s) in "
                  f"{pairs_dir} (an interrupted write; ignored by every scan, "
                  f"pass --repair-resume to delete them)")
        # Gap check over the whole index list, not just its max: resuming
        # from max+1 only ever walks FORWARD, so a hole below the highest
        # index can never be backfilled once generation passes it.
        indices = _list_sample_indices(pairs_dir)
        if indices:
            expected = list(range(base_idx, indices[-1] + 1))
            if indices != expected:
                missing = sorted(set(expected) - set(indices))
                unexpected = sorted(set(indices) - set(expected))
                raise DatasetContractError(
                    f"--resume refused: {pairs_dir} has a gap in its sample "
                    f"numbering relative to start index {base_idx} -- "
                    f"missing {missing}, unexpected {unexpected} (disk range "
                    f"{indices[0]}..{indices[-1]}, {len(indices)} present). "
                    "Resuming would silently skip past this gap forever (the "
                    "next sample written continues from the highest index, "
                    "never backfills a hole before it). Inspect "
                    f"{pairs_dir} manually before proceeding."
                )
        max_idx = indices[-1] if indices else -1
        sample_count = max(max_idx + 1, base_idx)
        done = sample_count - base_idx
    else:
        # A non-empty pairs_dir without --resume must never be silently
        # extended or partially overwritten: forgetting --resume and
        # generating a smaller batch with a DIFFERENT config into an
        # existing directory would overwrite only the low indices and leave
        # a stale tail from the old config behind, and nothing downstream
        # can detect that. --start-idx remains available, but only for a
        # genuinely fresh/empty --output directory (e.g. numbering a new
        # shard so it cannot collide with a DIFFERENT, unrelated batch
        # elsewhere) -- extending THIS directory's own batch must go
        # through --resume --hours <new TOTAL, not an increment>.
        if _list_sample_indices(pairs_dir):
            raise DatasetContractError(
                f"refused: {pairs_dir} already contains sample file(s) but "
                "--resume was not passed. Re-run with --resume (and --hours "
                "set to the batch's new TOTAL target, not an increment) to "
                "safely extend it, or pick an empty --output directory for "
                "an unrelated batch."
            )
        if base_idx > 0:
            print(f"New shard starting at {base_idx:06d}.wav (--start-idx, "
                  f"fresh output directory)")

    dataset = DNS4Dataset(cfg, return_raw=True)
    SR = dataset.sr
    pass_size = len(dataset)
    if pass_size <= 0:
        raise ValueError("DNS4Dataset contains no samples")

    n_rounds = (n_total + pass_size - 1) // pass_size
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
    partial = n_total % pass_size
    pass_note = (
        f"{n_rounds} dataset pass(es), final pass {partial}/{pass_size}"
        if partial
        else f"{n_rounds} complete dataset pass(es)"
    )
    print(f"\nRequested {args.hours:g} hours → {n_total} samples "
          f"({actual_hours:.3f} hours; {pass_note})")
    print(f"  Workers          : {n_workers}")
    print(f"  Estimated gen time : {est_hours:.1f} hours ({t_sample:.3f}s/sample)")
    print(f"  Estimated disk     : {disk_str}  (16-bit WAV)")
    print(f"  Sample rate        : {SR} Hz")
    if dataset.p_resample > 0.0:
        source_srs = ", ".join(str(value) for value in dataset.source_sr_values)
        print(f"  Upsampled sources  : {dataset.p_resample:.0%} "
              f"from [{source_srs}] Hz")
    print(f"  Output: {args.output}/")
    print()

    start_round = done // pass_size
    pass_start = done % pass_size
    if args.resume:
        if done > 0:
            print(f"Resuming: 本批已完成 {done} 筆 "
                  f"(max={max_idx:06d}.wav), "
                  f"從 {sample_count:06d}.wav 接續 "
                  f"(round {start_round + 1}, pass idx {pass_start})...")
        else:
            print(f"Resume: pairs/ 無 >= {base_idx:06d} 的樣本 "
                  f"→ 從 {base_idx:06d}.wav 開始")

    gen_start = time.time()

    for r in range(n_rounds):
        if r < start_round:
            continue
        if sample_count - base_idx >= n_total:
            break

        if n_rounds > 1:
            dataset._shuffle_indices()
            print(f"\n--- Round {r + 1}/{n_rounds} ---")

        idx_start = pass_start if r == start_round else 0
        remaining = n_total - (sample_count - base_idx)
        idx_stop = min(pass_size, idx_start + remaining)

        if n_workers > 0:
            indices = list(range(idx_start, idx_stop))
            subset = data.Subset(dataset, indices)
            loader = data.DataLoader(
                subset, batch_size=1, shuffle=False,
                num_workers=n_workers, prefetch_factor=2,
                worker_init_fn=seed_worker,
                persistent_workers=False,
            )
            for noisy, clean in tqdm.tqdm(
                    loader, desc=f"Round {r+1}/{n_rounds}", total=len(indices)):
                noisy = noisy.squeeze(0)   # (T,)
                clean = clean.squeeze(0)
                _save_pair_atomic(pairs_dir, sample_count, noisy, clean, SR)
                sample_count += 1
        else:
            for i in tqdm.tqdm(range(idx_start, idx_stop),
                               desc=f"Round {r+1}/{n_rounds}"):
                noisy, clean = dataset[i]
                _save_pair_atomic(pairs_dir, sample_count, noisy, clean, SR)
                sample_count += 1

    gen_elapsed = time.time() - gen_start
    batch_samples = sample_count - base_idx
    print(f"\nDone. Batch has {batch_samples}/{n_total} pairs "
          f"({batch_samples * segment_sec / 3600:.3f} audio hours); "
          f"next file index is {sample_count:06d}. "
          f"Generation took {gen_elapsed / 3600:.2f} hours → {args.output}/")
    print(f"  {SR} Hz: {pairs_dir} "
          f"(2-channel WAV ch0=noisy/ch1=clean, one atomically-written file "
          f"per sample, no sidecars)")
    print(f"  Next: python3 pack_dataset.py --input {pairs_dir} "
          f"--output {os.path.join(args.output, 'packed.pt')}")


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
                        help='接續同一批: 從 pairs/ 最大編號後續寫 (不低於 '
                             'start_idx)。⚠ 沒有任何 config/取樣率的紀錄可核'
                             '對 (已不寫 meta.json)，接續時請自行確保用的是'
                             '當初那份 config。')
    parser.add_argument('--repair-resume', action='store_true',
                        help='與 --resume 併用: 順手刪掉中斷留下的 tmp.*.wav '
                             '暫存檔 (不刪也不影響，掃描本來就看不到它們)。')
    parser.add_argument('--start-idx', type=int, default=None,
                        help='起始檔名編號, 覆寫 config [gen] start_idx。僅適用於 '
                             '全新/空的 --output 目錄 (例如替一個新 shard 編號)；'
                             '對已有樣本的目錄一律拒絕 (即使搭配 --start-idx)，'
                             '擴增既有批次請改用 --resume --hours <新總時數>。')
    parser.add_argument('--seed', type=int, default=None,
                        help='Reproducible random seed. Omit to use [gen] seed; '
                             'negative means a fresh OS-random seed each run.')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Generation sample rate in Hz. Overrides config.ini '
                             '[signal] sr (16000 for RNNoise-ERB/GTCRN, '
                             '48000 for DeepFilterNet2).')
    args = parser.parse_args()
    gen_dataset(args)
