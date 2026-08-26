#!/usr/bin/env python3
"""Recompute AIAEC's final stem from existing acoustic stem WAVs.

This does not rerender speech/noise/RIR mixtures. It concatenates each parent
sequence in ``chunk_index`` order, runs one fresh Python PBFDKF over that full
waveform, then atomically rewrites every chunk with ``linear_error`` as the
last channel (channel five under the current ``STEM_ORDER``; legacy
``BASE_STEM_ORDER``-only inputs have four).

Input is the rendered audio and ``--config``, nothing else -- the corpus has no
sidecars and no run meta.json (see gen_aec_dataset.py). Sequences and chunk
order come from the ``SSSSSS_CCC.wav`` filenames; rate, length and the current
channel count come from the WAV headers.

Sequences are independent by construction -- each gets a fresh PBFDKF -- and
each writes only its own chunks, so ``--jobs N`` fans them across processes
and scales approximately with N until CPU or memory bandwidth saturates.
That is the only lever worth pulling: measured on a 16 kHz corpus the Python
PBFDKF is 99.8% of the run and all file I/O is 0.1%, at roughly 3.3x realtime
per process. Nothing about the WAV handling is worth optimizing, and the
read-back check after every write costs nothing measurable.

The output does not depend on N: structurally there is no shared state and no
random source for it to depend on, and tests/test_rematerialize_linear_aec.py
compares every sample of --jobs 1 against --jobs 3.

⚠ If you check that yourself, compare AUDIO, not file bytes. libsndfile stamps
a PEAK chunk with the wall-clock time when it writes a float WAV, so any two
runs seconds apart differ in one byte at offset 61 -- at any --jobs, including
two serial runs.

``--resume`` skips a sequence only when THIS contract already wrote it and
that sequence's chunks still match on disk -- the ledger (see linear_error_ledger) records what was WRITTEN, not what survived. A full pass empties
the ledger first, so it cannot inherit a claim it did not make, and
pack_aec_dataset refuses a ledger that names only some of the corpus. A ledger written
by a different contract is ignored wholesale rather than partially trusted, so
a resumed run after a [linear_aec] change redoes everything instead of leaving
a corpus with two frontends mixed into it.
"""

from __future__ import annotations

import argparse
import configparser
import multiprocessing
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import tqdm


if __package__ in (None, ""):
    _AUDIO_ALG = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, _AUDIO_ALG)
    __package__ = "AIAEC.dataset_gen"

from .aec_features import BASE_STEM_ORDER, STEM_ORDER  # noqa: E402
from .linear_aec import (  # noqa: E402
    LinearAecContract,
    linear_aec_contract_from_config,
    materialize_linear_error,
)
from .linear_error_ledger import load_ledger, save_ledger  # noqa: E402
from .seq_layout import scan_chunks  # noqa: E402


WAV_ENCODINGS = {
    "float32": dict(encoding="PCM_F", bits_per_sample=32),
    "int16": dict(encoding="PCM_S", bits_per_sample=16),
}


def _encoding_of(path: str) -> str:
    """Which WAV_ENCODINGS entry this file already uses, for --wav-encoding auto."""
    info = torchaudio.info(path)
    for name, spec in WAV_ENCODINGS.items():
        if (info.encoding == spec['encoding']
                and info.bits_per_sample == spec['bits_per_sample']):
            return name
    raise ValueError(
        f"{path}: {info.encoding}/{info.bits_per_sample}-bit is neither "
        f"float32 nor int16; pass --wav-encoding explicitly")


def _load_sequence(
    wav_paths: List[str], expected_sr: int, expected_t: int
) -> Tuple[List[torch.Tensor], int]:
    chunks = []
    channels_seen = set()
    for path in wav_paths:
        audio, sr = torchaudio.load(path)
        if sr != expected_sr:
            raise ValueError(f"{path}: sr={sr}, expected {expected_sr}")
        if audio.shape[1] != expected_t:
            raise ValueError(f"{path}: T={audio.shape[1]}, expected {expected_t}")
        if audio.shape[0] not in (len(BASE_STEM_ORDER), len(STEM_ORDER)):
            raise ValueError(
                f"{path}: expected {len(BASE_STEM_ORDER)} legacy or "
                f"{len(STEM_ORDER)} current channels, got {audio.shape[0]}"
            )
        # A killed prior re-materialization may leave a mixture of four- and
        # five-channel files. Always recover from the first four stems.
        channels_seen.add(int(audio.shape[0]))
        chunks.append(audio[:len(BASE_STEM_ORDER)].float())
    return chunks, max(channels_seen)


def _sequence_is_current(wav_paths: List[str], expected_sr: int,
                         expected_t: int) -> bool:
    """Every chunk already five-channel and the right shape.

    A shape check, and deliberately only that: WHICH contract wrote the fifth
    channel is the ledger's job. This answers the other half -- whether the
    files the ledger is talking about are still the files it saw.
    """
    for path in wav_paths:
        info = torchaudio.info(path)
        if (
            info.num_channels != len(STEM_ORDER)
            or info.sample_rate != expected_sr
            or info.num_frames != expected_t
        ):
            return False
    return True


def _rewrite_sequence(job) -> int:
    """Rematerialize one sequence and rewrite its chunks. Returns its id.

    Deliberately takes only picklable arguments and touches only this
    sequence's own files, which is what makes --jobs safe: a fresh PBFDKF per
    sequence (materialize_linear_error builds one), no shared mutable state,
    and no two workers writing the same path.
    """
    (seqs_dir, sequence_id, wav_paths, sr, chunk_samples,
     contract, encoding) = job

    expected = [
        os.path.join(seqs_dir, f"{sequence_id:06d}_{index:03d}.wav")
        for index in range(len(wav_paths))
    ]
    if wav_paths != expected:
        raise FileNotFoundError(
            f"sequence {sequence_id:06d} has a gap in its chunk numbering; "
            f"the PBFDKF has to run over the whole sequence in order, so "
            f"there is nothing sensible to rematerialize from")

    chunks, _old_channels = _load_sequence(wav_paths, sr, chunk_samples)
    far = torch.cat([
        chunk[BASE_STEM_ORDER.index("far_render")] for chunk in chunks
    ])
    mic = torch.cat([
        chunk[BASE_STEM_ORDER.index("mic_postclip")] for chunk in chunks
    ])
    linear_error, _echo_estimate = materialize_linear_error(mic, far, contract)

    for chunk_index, (path, base) in enumerate(zip(wav_paths, chunks)):
        at = chunk_index * chunk_samples
        error_chunk = linear_error[at:at + chunk_samples].unsqueeze(0)
        full = torch.cat([base, error_chunk], dim=0).contiguous()
        tmp = os.path.join(os.path.dirname(path),
                           f"tmp.{os.path.basename(path)}")
        torchaudio.save(tmp, full, sr, **WAV_ENCODINGS[encoding])
        check, check_sr = torchaudio.load(tmp)
        if check_sr != sr or check.shape != full.shape:
            raise RuntimeError(
                f"{tmp}: failed five-channel round-trip validation"
            )
        os.replace(tmp, path)
    return sequence_id


def _worker_init() -> None:
    """One compute thread per worker.

    Not a correctness measure -- the result is identical at any thread count,
    measured -- but N processes each spawning N threads oversubscribes the
    machine and makes --jobs slower than serial past a small N.
    """
    torch.set_num_threads(1)


def rematerialize(args) -> None:
    seqs_dir = args.input
    if os.path.isdir(os.path.join(seqs_dir, "seqs")):
        seqs_dir = os.path.join(seqs_dir, "seqs")
    sequences = scan_chunks(seqs_dir)
    if not sequences:
        raise FileNotFoundError(f"no SSSSSS_CCC.wav chunk files under {seqs_dir}")

    first = sequences[min(sequences)][0]
    info = torchaudio.info(first)
    sr = int(info.sample_rate)
    chunk_samples = int(info.num_frames)

    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")
    if cfg.getint("signal", "sr") != sr:
        raise ValueError(
            f"config sr={cfg.getint('signal', 'sr')} but the corpus is {sr} Hz"
        )
    contract: LinearAecContract = linear_aec_contract_from_config(cfg)
    if chunk_samples % contract.hop_size:
        raise ValueError(
            f"chunk_samples={chunk_samples} is not divisible by PBFDKF "
            f"hop={contract.hop_size}"
        )
    contract_hash = contract.fingerprint()

    encoding = (_encoding_of(first) if args.wav_encoding == "auto"
                else args.wav_encoding)
    if encoding not in WAV_ENCODINGS:
        raise ValueError(f"unsupported WAV encoding {encoding!r}")

    jobs = int(args.jobs)
    if jobs < 1:
        raise ValueError(f"--jobs must be at least 1, got {jobs}")

    if args.resume:
        claimed = load_ledger(seqs_dir, contract_hash)
        # The ledger records what this contract WROTE, not what is still on
        # disk. A file deleted, truncated or half-restored since then would be
        # skipped on its say-so, so every claim is re-checked against the
        # actual chunks before it is honoured. One torchaudio.info() per chunk,
        # measured at 67 us warm: about 5 s to re-check a whole 200-hour corpus,
        # and zero on a full pass, which never enters this branch.
        done = {sid for sid in claimed
                if sid in sequences
                and _sequence_is_current(sequences[sid], sr, chunk_samples)}
        dropped = len(claimed) - len(done)
        print(f"resume: {len(done)} sequence(s) already written by this "
              f"contract ({contract_hash[:12]})"
              + (f"; {dropped} re-queued -- their chunks no longer match"
                 if dropped else ""))
        if dropped:
            save_ledger(seqs_dir, contract_hash, done, contract.as_dict())
    else:
        done = set()
        # Start from an empty ledger rather than whatever a previous run left.
        # Otherwise a full-pass run that dies before its first sequence lands
        # leaves the old ledger intact, and the next --resume trusts a claim
        # this run never made.
        save_ledger(seqs_dir, contract_hash, done, contract.as_dict())

    pending = [sid for sid in sorted(sequences) if sid not in done]
    work = [
        (seqs_dir, sid, sequences[sid], sr, chunk_samples, contract, encoding)
        for sid in pending
    ]

    rewritten = 0
    pool = None
    progress = tqdm.tqdm(total=len(work), desc="linear-aec")
    try:
        if jobs == 1:
            results = (_rewrite_sequence(job) for job in work)
        else:
            # SPAWN, explicitly. Linux defaults to fork, and forking a process
            # that has already imported torch inherits its OpenMP thread state
            # into a child that never ran pthread_atfork -- a documented way to
            # deadlock. spawn costs a fresh interpreter per worker, which is
            # nothing against a run measured in hours.
            pool = multiprocessing.get_context("spawn").Pool(
                jobs, initializer=_worker_init)
            results = pool.imap_unordered(_rewrite_sequence, work)
        for sequence_id in results:
            done.add(sequence_id)
            rewritten += 1
            # Recorded only once the sequence's own chunks are all on disk,
            # so an interrupted run never claims one it did not finish.
            save_ledger(seqs_dir, contract_hash, done, contract.as_dict())
            progress.update(1)
    except BaseException:
        # terminate(), not close(): close() only stops NEW tasks being queued,
        # so the ones already dispatched would keep rewriting WAVs for minutes
        # after the failure that is on its way up the stack. KeyboardInterrupt
        # is caught here for the same reason, which is why this is BaseException.
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        progress.close()

    print(
        f"Done: {rewritten} sequence(s) rewritten; contract={contract_hash[:12]}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="A generated split directory, or its seqs/ subdirectory",
    )
    parser.add_argument(
        "--config", required=True,
        help="Dataset config containing [signal] and optional [linear_aec]",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip sequences this contract already wrote, per the ledger "
             "beside the corpus. A ledger from another contract is ignored "
             "whole, so this cannot leave two frontends mixed in one corpus",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Sequences to rematerialize in parallel (default 1). Sequences "
             "are independent, so the output does not depend on this",
    )
    parser.add_argument(
        "--wav-encoding", default="auto",
        choices=("auto", "float32", "int16"),
        help="Default: keep whatever encoding the existing chunks use",
    )
    return parser


if __name__ == "__main__":
    rematerialize(build_parser().parse_args())
