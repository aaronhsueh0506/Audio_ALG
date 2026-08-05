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
# way that would make an old batch directory unsafe to blindly resume into.
# Written into meta.json and checked against the CURRENT run's value before
# any --resume proceeds -- see _validate_resume_contract. There is
# deliberately NO bypass for a contract_version mismatch (an earlier
# --resume-force-contract-mismatch flag was removed for release -- it let a
# caller mix two distributions into one dataset and then silently overwrote
# the only record that had happened, since meta.json's config_hash/
# contract_version get replaced by the new run's values on the very next
# save with no history retained; a real migration path needs to record old/
# new contract identity and the switchover index, which nothing here does
# yet -- pick a new --output directory instead).
#   1: initial contract_version/config_hash tracking (metadata.jsonl ->
#      per-sample atomic JSON sidecars).
#   2: (superseded before shipping -- see 3)
#   3: _canonical_config_hash() now stores the FULL sha256 digest instead of
#      a 16-hex-char truncation (a v1/v2 batch's stored hash can never equal
#      a v3-computed hash for byte-identical config, so this alone forces a
#      version bump); meta.json is now written atomically (temp+rename) and
#      BEFORE the first sample of a run (previously only after each full
#      generation round -- a crash inside a large first round could leave
#      thousands of complete WAV+JSON pairs with no meta.json at all,
#      exactly what the old _validate_resume_contract() misread as "no
#      batch here yet, must be fresh"); --resume now refuses by default
#      when meta.json is missing but pairs/ already has files, instead of
#      treating that as fresh; contract validation now runs BEFORE
#      DNS4Dataset construction (RIR glob/cache) and the one-sample
#      profiling pass, so a mismatch exits immediately instead of after
#      paying for both; meta.json gains generation_history (see gen_dataset()
#      below) so a --resume run's own seed/effective_seed no longer
#      silently overwrites and loses the record of every earlier run's.
AINR_DATASET_CONTRACT_VERSION = 3


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
    Full 64-hex-char digest -- NOT truncated (a prior version truncated to
    16 chars; that was a real loss of collision resistance for no benefit,
    since this value is never manually read/typed by a human, only compared
    programmatically -- see AINR_DATASET_CONTRACT_VERSION's changelog)."""
    payload = json.dumps(_canonical_config_dict(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _validate_resume_contract(meta_path, pairs_dir, contract_version,
                               config_hash, sr):
    """Check an existing meta.json (if any) against the CURRENT run's
    contract_version/config_hash/sr before a --resume run proceeds. Returns
    the loaded meta dict, or None if no meta.json exists yet AND `pairs_dir`
    has no sample files either (a genuinely fresh --resume into an empty/new
    output directory -- not an error).

    Raises DatasetContractError on ANY mismatch, with no bypass: editing
    config.ini (SNR list, RIR probability, level normalization, clipping
    knobs, ...) or switching --sample-rate and then resuming into the same
    --output directory would otherwise silently mix two different
    distributions (or two different sample rates) into one dataset with no
    record anything changed. (An earlier --resume-force-contract-mismatch
    escape hatch was removed before release -- see
    AINR_DATASET_CONTRACT_VERSION's doc for why.)

    Also raises if meta.json is MISSING but `pairs_dir` already contains
    sample files: since meta.json is now written before the first sample of
    every run (see gen_dataset()), this can only happen for a batch that
    predates contract versioning, or one where meta.json was separately
    lost/deleted -- either way it cannot be validated against the current
    run, so it must not be silently treated as a fresh, empty batch.
    """
    if not os.path.exists(meta_path):
        has_existing_samples = bool(
            glob.glob(os.path.join(pairs_dir, '*.wav'))
            or glob.glob(os.path.join(pairs_dir, '*.json'))
        )
        if has_existing_samples:
            raise DatasetContractError(
                f"--resume refused: {pairs_dir} already contains sample "
                f"file(s) but no meta.json exists at {meta_path}, so they "
                "cannot be validated against the current run's contract/"
                "config. This batch either predates contract versioning, "
                "or meta.json was lost/deleted separately from pairs/. "
                "Move pairs/ aside and inspect manually (or discard it) "
                "before starting a fresh --output directory."
            )
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
    if problems:
        raise DatasetContractError(
            "--resume refused: this output directory's meta.json does not "
            "match the current run (would silently mix two distributions "
            "into one dataset):\n  " + "\n  ".join(problems) +
            "\nFix config.ini/--sample-rate to match the original batch, "
            "or pick a new --output directory."
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


def _list_complete_sample_indices(pairs_dir):
    """Return (complete_indices_sorted, orphan_wav_indices, orphan_json_indices).

    A "complete" sample has BOTH NNNNNN.wav and NNNNNN.json. Orphan lists
    are sorted indices for a caller to report and, with --repair-resume (or
    manually), delete. `tmp.NNNNNN.wav`/`tmp.NNNNNN.json` (an in-progress or
    crashed write -- see _tmp_path) never match the `*.wav`/`*.json` globs'
    digit-only-stem filter below, so they are correctly ignored here, not
    misread as orphans. Shared by both gen_dataset.py's own --resume
    bookkeeping (_scan_existing_samples below, which only needs the max
    index) and pack_dataset.py (which needs the full list, to pack exactly
    the complete, contiguous set of samples a batch actually produced --
    never a stray temp/orphan file a naive `glob('*.wav')` would also
    match).
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
    complete = sorted(wav_indices & json_indices)
    orphan_wavs = sorted(wav_indices - json_indices)
    orphan_jsons = sorted(json_indices - wav_indices)
    return complete, orphan_wavs, orphan_jsons


def _scan_existing_samples(pairs_dir):
    """Return (max_complete_index, orphan_wav_indices, orphan_json_indices).

    Only complete (WAV+JSON) samples count toward max_complete_index -- an
    orphan WAV's metadata is missing, so it must never be treated as
    "already generated" (that would permanently lose that sample's
    metadata on --resume). See _list_complete_sample_indices for the full
    scan this wraps.
    """
    complete, orphan_wavs, orphan_jsons = _list_complete_sample_indices(pairs_dir)
    max_complete = complete[-1] if complete else -1
    return max_complete, orphan_wavs, orphan_jsons


def _repair_orphans(pairs_dir, orphan_wavs, orphan_jsons):
    """Delete every orphan WAV/JSON --repair-resume identified. An orphan
    WAV's sample was never actually completed (its metadata is missing);
    an orphan JSON's sample's audio is missing. Neither is salvageable, so
    deleting it is always the correct repair -- but the caller must still
    re-check index contiguity afterwards: if genuinely complete samples
    exist at higher indices too, deleting an INTERIOR orphan leaves a gap
    that --resume's own "continue from max(complete)+1" logic can never
    backfill, and gen_dataset() refuses to proceed in that case rather
    than silently leaving it (see the caller's post-repair check)."""
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

    pairs_dir = os.path.join(args.output, 'pairs')
    meta_path = os.path.join(args.output, 'meta.json')
    os.makedirs(pairs_dir, exist_ok=True)

    # ---- Resume/contract validation runs BEFORE constructing DNS4Dataset
    # (RIR glob/cache I/O) and the one-sample profiling pass below -- a
    # config/contract mismatch must exit immediately, not after paying for
    # both. Everything this needs (pairs_dir, meta_path, config_hash,
    # generation_sr) is already available at this point. ----
    # 起始檔名編號 base_idx 是本批的「地板」:
    #   --resume: 掃 pairs/ 取「完整 pair (wav+json) 最大編號 +1」，但不低於 base_idx。
    #   無 --resume: 直接從 base_idx 起 (擴增, 不掃碟; start_idx 已保證不洗舊檔)。
    sample_count = base_idx
    done = 0
    max_idx = -1
    existing_meta = None
    if args.resume:
        existing_meta = _validate_resume_contract(
            meta_path, pairs_dir, AINR_DATASET_CONTRACT_VERSION, config_hash,
            generation_sr,
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
        # Re-derive from the FULL complete-index list, not just max_idx:
        # --repair-resume above only deletes orphans, it does not by itself
        # guarantee the remaining complete indices are gap-free. An orphan
        # that (however it arose -- see AINR_DATASET_CONTRACT_VERSION's
        # changelog) had genuinely COMPLETE samples after it would otherwise
        # be silently and permanently unrecoverable: resuming from
        # max_idx+1 only ever walks FORWARD from the highest complete
        # index, so a gap anywhere before it can never be revisited once
        # generation resumes past it.
        complete_indices, _, _ = _list_complete_sample_indices(pairs_dir)
        if complete_indices:
            expected = list(range(complete_indices[0], complete_indices[-1] + 1))
            if complete_indices != expected:
                missing = sorted(set(expected) - set(complete_indices))
                raise DatasetContractError(
                    f"--resume refused: {pairs_dir} has a gap in its "
                    f"complete sample indices -- missing {missing} "
                    f"(range {complete_indices[0]}..{complete_indices[-1]}, "
                    f"{len(complete_indices)} present). Resuming would "
                    "silently skip past this gap forever (the next sample "
                    "written continues from the highest COMPLETE index, "
                    "never backfills a hole before it). This should not be "
                    "reachable through normal --resume use -- inspect "
                    f"{pairs_dir} manually before proceeding."
                )
        max_idx = complete_indices[-1] if complete_indices else -1
        sample_count = max(max_idx + 1, base_idx)
        done = sample_count - base_idx
    else:
        # A non-empty pairs_dir without --resume must never be silently
        # extended or partially overwritten -- see
        # AINR_DATASET_CONTRACT_VERSION's changelog for the exact
        # corruption this closes: forgetting --resume and generating a
        # smaller batch with a DIFFERENT config into an existing directory
        # used to silently overwrite only the low indices (leaving a stale
        # tail from the OLD config behind) while meta.json recorded only
        # the NEW config -- a distribution mix pack_dataset.py had no way
        # to detect either. --start-idx remains available, but only for a
        # genuinely fresh/empty --output directory (e.g. numbering a new
        # shard so it cannot collide with a DIFFERENT, unrelated batch
        # elsewhere) -- extending THIS directory's own batch must go
        # through --resume --hours <new TOTAL, not an increment>.
        existing_max_idx, existing_orphan_wavs, existing_orphan_jsons = (
            _scan_existing_samples(pairs_dir)
        )
        # Also refuse when meta.json exists even if pairs_dir is currently
        # EMPTY -- e.g. a crash right after _save_meta(0) but before the
        # first sample's write (see _save_meta(0)'s own call site above).
        # Silently overwriting that meta.json here would discard its
        # generation_history/contract record with nothing to show for it,
        # the same "old record silently lost" failure mode this whole
        # check exists to prevent, just with zero samples at stake instead
        # of a whole batch.
        if (existing_max_idx >= 0 or existing_orphan_wavs or existing_orphan_jsons
                or os.path.isfile(meta_path)):
            raise DatasetContractError(
                f"refused: {pairs_dir} already contains sample file(s), or "
                f"{meta_path} already exists, but --resume was not passed. "
                "Re-run with --resume (and --hours set to the batch's new "
                "TOTAL target, not an increment) to safely extend it, or "
                "pick an empty --output directory (with no meta.json "
                "either) for an unrelated batch."
            )
        if base_idx > 0:
            print(f"New shard starting at {base_idx:06d}.wav (--start-idx, "
                  f"fresh output directory)")

    # One entry appended per gen_dataset() invocation, NEVER overwritten by
    # a later --resume (see AINR_DATASET_CONTRACT_VERSION's v3 changelog).
    # Without this, meta.json's top-level seed/effective_seed/seed_source
    # only ever described the MOST RECENT invocation -- every --resume with
    # the default seed=-1 draws a fresh OS-random seed and _save_meta()
    # below used to clobber the previous run's record with it, so meta.json
    # claimed to describe the whole accumulated batch (n_samples IS
    # cumulative) but could no longer actually reproduce any of it.
    generation_history = list((existing_meta or {}).get('generation_history', []))
    generation_history.append({
        'seed': run_seed,
        'effective_seed': effective_seed,
        'seed_source': seed_source,
        'base_idx': base_idx,
        'sample_count_at_run_start': sample_count,
    })

    dataset = DNS4Dataset(cfg, return_raw=True, return_metadata=True)
    SR = dataset.sr
    pass_size = len(dataset)
    if pass_size <= 0:
        raise ValueError("DNS4Dataset contains no samples")

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

    start_round = done // pass_size
    pass_start = done % pass_size
    if args.resume:
        if done > 0:
            print(f"Resuming: 本批已完成 {done} 筆 "
                  f"(max={max_idx:06d}.wav), "
                  f"從 {sample_count:06d}.wav 接續 "
                  f"(round {start_round + 1}, pass idx {pass_start})...")
        else:
            print(f"Resume: pairs/ 無 >= {base_idx:06d} 的完整 pair "
                  f"→ 從 {base_idx:06d}.wav 開始")

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
            'generation_history': generation_history,
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
        # Atomic (temp + os.replace), same discipline as the WAV/JSON
        # sidecar writes -- a crash mid-write must never leave a truncated,
        # unparseable meta.json at the final path.
        tmp_path = _tmp_path(meta_path)
        with open(tmp_path, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_path, meta_path)

    # Written immediately, before the first sample of this run -- a crash
    # partway through generating the FIRST round (previously the earliest
    # point meta.json could appear at all: only after a full round, which
    # can be thousands of samples for a large pass_size) now still leaves a
    # valid contract record on disk for the next --resume to check against.
    _save_meta(0)

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
                             '.json，或只有 .json 缺 .wav) 而非直接拒絕接續。'
                             '若刪除後在既有完整編號中間留下缺口 (缺口前後都'
                             '有真正完整的樣本)，仍會拒絕接續 (該缺口無法被 '
                             '"從最大完整編號+1 繼續" 的接續邏輯補回)。')
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
