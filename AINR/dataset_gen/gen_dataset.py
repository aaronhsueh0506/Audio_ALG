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
import hashlib
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


# Bump whenever the on-disk batch format or --resume semantics change in a
# way that would make an old batch directory unsafe to blindly resume into
# (e.g. this version's introduction: metadata.jsonl -> per-sample atomic
# JSON sidecars). Written into meta.json and checked against the CURRENT
# run's value before any --resume proceeds -- see _validate_resume_contract.
AINR_DATASET_CONTRACT_VERSION = 2


class DatasetContractError(RuntimeError):
    """--resume's contract/config validation (_validate_resume_contract) or
    orphan-sample detection (_scan_existing_samples) refused to proceed."""


def _canonical_config_dict(cfg: configparser.ConfigParser) -> dict:
    """Every section/key/value in `cfg` as a plain dict of dicts -- the
    FULL resolved configuration (any CLI override, e.g. --sample-rate, is
    already folded into `cfg` by the time gen_dataset() calls this), not a
    hand-picked subset of fields. Hashing this rather than individually
    tracked dataset.* attributes (snr_values, p_rir, level_mode,
    p_noise_clipping, ...) means a newly added config knob is automatically
    covered without anyone having to remember to extend a hash function.
    """
    return {section: dict(cfg.items(section)) for section in cfg.sections()}


def _canonical_config_hash(cfg: configparser.ConfigParser) -> str:
    """Stable (sorted-keys JSON) sha256 of the whole resolved config, for
    detecting a distribution-changing config edit across a --resume run.
    First 16 hex chars only -- a human-readable fingerprint, not a security
    hash, so truncation collision risk is not a concern here."""
    payload = json.dumps(_canonical_config_dict(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def _validate_resume_contract(meta_path, contract_version, config_hash, sr,
                               force=False):
    """Check an existing meta.json (if any) against the CURRENT run's
    contract_version/config_hash/sr before a --resume run proceeds. Returns
    the loaded meta dict, or None if no meta.json exists yet (a genuinely
    fresh --resume into an empty/new output directory -- not an error).
    Raises DatasetContractError on any mismatch unless force=True: without
    this check, editing config.ini (SNR list, RIR probability, level
    normalization, clipping knobs, ...) or switching --sample-rate and then
    resuming into the same --output directory would silently mix two
    different distributions (or two different sample rates) into one
    dataset with no record anything changed.
    """
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        existing = json.load(f)
    problems = []
    existing_cv = existing.get('contract_version')
    if existing_cv != contract_version:
        problems.append(
            f"contract_version mismatch: existing batch={existing_cv!r}, "
            f"this script={contract_version!r}"
        )
    existing_hash = existing.get('config_hash')
    if existing_hash != config_hash:
        problems.append(
            f"config_hash mismatch: existing batch={existing_hash!r}, "
            f"current config={config_hash!r} -- config.ini (or a CLI "
            "override) changed since this batch was started"
        )
    existing_sr = existing.get('sr')
    if existing_sr != sr:
        problems.append(
            f"sample rate mismatch: existing batch={existing_sr!r} Hz, "
            f"current run={sr!r} Hz"
        )
    if problems and not force:
        raise DatasetContractError(
            "--resume refused: this output directory's meta.json does not "
            "match the current run (would silently mix two distributions "
            "into one dataset):\n  " + "\n  ".join(problems) +
            "\nFix config.ini/--sample-rate to match the original batch, "
            "pick a new --output directory, or pass "
            "--resume-force-contract-mismatch if this is a deliberate "
            "migration."
        )
    return existing


def _sample_paths(pairs_dir, index):
    stem = f"{index:06d}"
    return (
        os.path.join(pairs_dir, f"{stem}.wav"),
        os.path.join(pairs_dir, f"{stem}.json"),
    )


def _tmp_path(final_path):
    """Same-directory temp path for `final_path`, named `tmp.<basename>`
    (prefix, not suffix). Two things need this exact shape:
      1. torchaudio's soundfile backend infers the save format from the
         LAST '.'-separated segment of a string path and ignores its own
         `format=` kwarg in that case (a soundfile-backend quirk) -- so the
         temp path must still literally END in '.wav' for _save_pair_atomic
         to work at all; a '.wav.tmp' suffix breaks that.
      2. _scan_existing_samples()'s glob-based scan must never mistake a
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
    matches, so it can never be mistaken for a real sample)."""
    wav_path, _ = _sample_paths(pairs_dir, index)
    tmp_path = _tmp_path(wav_path)
    pair = torch.stack([noisy, clean], dim=0)
    torchaudio.save(tmp_path, pair, sr, bits_per_sample=16)
    os.replace(tmp_path, wav_path)


def _save_metadata_sidecar_atomic(pairs_dir, index, metadata):
    """Write pairs/NNNNNN.json -- this sample's own metadata, one file per
    sample (see AINR_DATASET_CONTRACT_VERSION's doc: this replaced a single
    shared metadata.jsonl that had no atomicity or reconciliation story) --
    via the same temp-file + os.replace atomicity as _save_pair_atomic.
    Callers write this AFTER the WAV (see the generation loop below): if a
    crash lands between the two renames, the WAV exists without its
    sidecar -- an "orphan" _scan_existing_samples() detects below, that
    --repair-resume can clean up -- rather than a sidecar existing for
    audio that was never actually written.
    """
    _, json_path = _sample_paths(pairs_dir, index)
    tmp_path = _tmp_path(json_path)
    with open(tmp_path, 'w') as f:
        json.dump({'index': index, **metadata}, f)
    os.replace(tmp_path, json_path)


def _scan_existing_samples(pairs_dir):
    """Return (max_complete_index, orphan_wav_indices, orphan_json_indices).

    A "complete" sample has BOTH NNNNNN.wav and NNNNNN.json; only complete
    samples count toward max_complete_index -- an orphan WAV's metadata is
    missing, so it must never be treated as "already generated" (that would
    permanently lose that sample's metadata on --resume). Orphan lists are
    sorted indices for a caller to report and, with --repair-resume,
    delete. `tmp.NNNNNN.wav`/`tmp.NNNNNN.json` (an in-progress or crashed
    write -- see _tmp_path) never match the `*.wav`/`*.json` globs' digit-
    only-stem filter below, so they are correctly ignored here, not
    misread as orphans.
    """
    wav_indices = set()
    json_indices = set()
    for path in glob.glob(os.path.join(pairs_dir, '*.wav')):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.isdigit():
            wav_indices.add(int(stem))
    for path in glob.glob(os.path.join(pairs_dir, '*.json')):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.isdigit():
            json_indices.add(int(stem))
    complete = wav_indices & json_indices
    orphan_wavs = sorted(wav_indices - json_indices)
    orphan_jsons = sorted(json_indices - wav_indices)
    max_complete = max(complete) if complete else -1
    return max_complete, orphan_wavs, orphan_jsons


def _repair_orphans(pairs_dir, orphan_wavs, orphan_jsons):
    """Delete every orphan WAV/JSON --repair-resume identified. An orphan
    WAV's sample was never actually completed (its metadata is missing);
    an orphan JSON's sample's audio is missing. Neither is salvageable --
    the only correct repair is to remove it so that index regenerates."""
    for index in orphan_wavs:
        wav_path, _ = _sample_paths(pairs_dir, index)
        os.remove(wav_path)
    for index in orphan_jsons:
        _, json_path = _sample_paths(pairs_dir, index)
        os.remove(json_path)


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


def _collate_pair_with_metadata(batch):
    """collate_fn for DNS4Dataset(return_raw=True, return_metadata=True).

    batch_size=1 always in this script's DataLoader use -- keep metadata as
    a plain per-sample dict list instead of running it through
    default_collate, which raises on the None-valued fields (rir_file,
    snr_db, ... are None whenever that pipeline step didn't fire for a
    given sample). Must be a module-level function, not a closure: spawn-
    based multiprocessing (the default on macOS) pickles collate_fn to
    hand to worker processes, and a local closure isn't picklable.
    """
    noisy = data.default_collate([item[0] for item in batch])
    clean = data.default_collate([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return noisy, clean, metadata


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

    # Computed AFTER the --sample-rate override is folded into `cfg` above,
    # so a --sample-rate-only change (no config.ini edit) is still caught by
    # --resume's contract check below (see _canonical_config_hash's doc).
    config_hash = _canonical_config_hash(cfg)

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

    dataset = DNS4Dataset(cfg, return_raw=True, return_metadata=True)
    SR = dataset.sr
    pass_size = len(dataset)
    if pass_size <= 0:
        raise ValueError("DNS4Dataset contains no samples")

    pairs_dir = os.path.join(args.output, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)

    n_rounds = (n_total + pass_size - 1) // pass_size
    actual_hours = n_total * segment_sec / 3600

    # Profile 1 sample
    print("Profiling 1 sample...")
    t0 = time.time()
    _s, _t, _m = dataset[0]
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

    # 起始檔名編號 base_idx 是本批的「地板」:
    #   --resume: 掃 pairs/ 取「完整 pair (wav+json) 最大編號 +1」，但不低於 base_idx。
    #   無 --resume: 直接從 base_idx 起 (擴增, 不掃碟; start_idx 已保證不洗舊檔)。
    sample_count = base_idx
    start_round = 0
    pass_start = 0
    meta_path = os.path.join(args.output, 'meta.json')

    if args.resume:
        _validate_resume_contract(
            meta_path, AINR_DATASET_CONTRACT_VERSION, config_hash, SR,
            force=args.resume_force_contract_mismatch,
        )
        max_idx, orphan_wavs, orphan_jsons = _scan_existing_samples(pairs_dir)
        if orphan_wavs or orphan_jsons:
            if args.repair_resume:
                print(f"Repairing: removing {len(orphan_wavs)} orphan WAV(s) "
                      f"(missing sidecar) and {len(orphan_jsons)} orphan "
                      f"JSON(s) (missing audio): "
                      f"{orphan_wavs + orphan_jsons}")
                _repair_orphans(pairs_dir, orphan_wavs, orphan_jsons)
                max_idx, orphan_wavs, orphan_jsons = _scan_existing_samples(pairs_dir)
            else:
                raise DatasetContractError(
                    "--resume refused: found incomplete sample(s) in "
                    f"{pairs_dir} -- {len(orphan_wavs)} WAV(s) with no "
                    f"metadata sidecar {orphan_wavs}, "
                    f"{len(orphan_jsons)} JSON sidecar(s) with no audio "
                    f"{orphan_jsons}. Each was interrupted mid-write (or is "
                    "left over from a pre-contract-version batch) and can "
                    "never be treated as \"already generated\" -- re-run "
                    "with --repair-resume to delete them (that index will "
                    "regenerate), or fix manually."
                )
        sample_count = max(max_idx + 1, base_idx)
        done = sample_count - base_idx
        if done > 0:
            start_round = done // pass_size
            pass_start = done % pass_size
            print(f"Resuming: 本批已完成 {done} 筆 "
                  f"(max={max_idx:06d}.wav), "
                  f"從 {sample_count:06d}.wav 接續 "
                  f"(round {start_round + 1}, pass idx {pass_start})...")
        else:
            print(f"Resume: pairs/ 無 >= {base_idx:06d} 的完整 pair "
                  f"→ 從 {base_idx:06d}.wav 開始")
    elif base_idx > 0:
        collision_wav, collision_json = _sample_paths(pairs_dir, base_idx)
        if os.path.exists(collision_wav) or os.path.exists(collision_json):
            print(f"WARNING: {base_idx:06d}.wav/.json already exist in "
                  f"{pairs_dir} and will be OVERWRITTEN (not --resume -- "
                  "if this is unintentional, pass --resume instead or "
                  "choose a --start-idx past the existing batch)")
        print(f"Extending dataset: 從 {base_idx:06d}.wav 開始")

    def _save_meta(rounds_done):
        batch_samples = sample_count - base_idx
        meta = {
            'contract_version': AINR_DATASET_CONTRACT_VERSION,
            'config_hash': config_hash,
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
            'snr_values_db': dataset.snr_values,
            'noise_only_p': dataset.noise_only_p,
            'speech_only_p': dataset.speech_only_p,
            'p_resample': dataset.p_resample,
            'source_sr_values': dataset.source_sr_values,
            'level_mode': dataset.level_mode,
            'target_level_min_db': dataset.target_level_min_db,
            'target_level_max_db': dataset.target_level_max_db,
            'p_noise_clipping': dataset.p_noise_clipping,
            'p_mixture_clipping': dataset.p_mixture_clipping,
            'generation_split': dataset.generation_split,
            'config': args.config,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

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
                collate_fn=_collate_pair_with_metadata,
            )
            for noisy, clean, metadata in tqdm.tqdm(
                    loader, desc=f"Round {r+1}/{n_rounds}", total=len(indices)):
                noisy = noisy.squeeze(0)   # (T,)
                clean = clean.squeeze(0)
                _save_pair_atomic(pairs_dir, sample_count, noisy, clean, SR)
                _save_metadata_sidecar_atomic(pairs_dir, sample_count, metadata[0])
                sample_count += 1
        else:
            for i in tqdm.tqdm(range(idx_start, idx_stop),
                               desc=f"Round {r+1}/{n_rounds}"):
                noisy, clean, metadata = dataset[i]
                _save_pair_atomic(pairs_dir, sample_count, noisy, clean, SR)
                _save_metadata_sidecar_atomic(pairs_dir, sample_count, metadata)
                sample_count += 1

        _save_meta(r + 1)

    gen_elapsed = time.time() - gen_start
    batch_samples = sample_count - base_idx
    if batch_samples >= n_total:
        completed_rounds = (batch_samples + pass_size - 1) // pass_size
        _save_meta(min(n_rounds, completed_rounds))
    print(f"\nDone. Batch has {batch_samples}/{n_total} pairs "
          f"({batch_samples * segment_sec / 3600:.3f} audio hours); "
          f"next file index is {sample_count:06d}. "
          f"Generation took {gen_elapsed / 3600:.2f} hours → {args.output}/")
    print(f"  {SR} Hz: {pairs_dir} "
          f"(2-channel WAV ch0=noisy/ch1=clean + NNNNNN.json metadata sidecar, "
          f"both written atomically per sample)")


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
                        help='接續同一批: 從 pairs/ 最大「完整 pair」編號後續寫 '
                             '(不低於 start_idx)。核對既有 meta.json 的 '
                             'contract_version/config_hash/sr，不符則拒絕。')
    parser.add_argument('--repair-resume', action='store_true',
                        help='與 --resume 併用: 自動刪除孤兒檔案 (只有 .wav 缺 '
                             '.json，或只有 .json 缺 .wav) 而非直接拒絕接續')
    parser.add_argument('--resume-force-contract-mismatch', action='store_true',
                        help='與 --resume 併用: 略過 contract_version/'
                             'config_hash/sr 核對 (危險 -- 僅供刻意遷移舊批次使用)')
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
