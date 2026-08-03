# -*- coding: utf-8 -*-
"""Pack rendered stem WAVs into .pt shards.

Same motivation as ``AINR/dataset_gen/pack_dataset.py``: per-file I/O dominates
training on a corpus of this size.  The shard layout is the packed format every
AEC model project consumes:

    {
      'stems': ['far_render','near_speech','near_target',
                'mic_postclip','linear_error'],  # channel order, fixed
      'data' : float32 tensor (N, 5, T),
      'sr'   : int,
      'meta' : list of N dicts,
      'generator_commit': str,
      'config_hash'     : str,
    }

⚠ Chunks are packed in ``(sequence_id, chunk_index)`` order and a sequence is
never split across shards. Training later randomizes global chunk indices, but
the physical order is retained for full-sequence reconstruction and streaming
evaluation.

Usage:
    python3 pack_aec_dataset.py --input data_aec/train \\
        --output data_aec/packed/train --shard-clips 512
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import torch
import torchaudio
import tqdm

if __package__ in (None, ''):
    _AUDIO_ALG = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, _AUDIO_ALG)
    __package__ = 'AIAEC.dataset_gen'

from .aec_features import STEM_ORDER  # noqa: E402
from .linear_aec import LinearAecContract  # noqa: E402


def _read_run_meta(input_dir: str) -> dict:
    path = os.path.join(input_dir, 'meta.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Point --input at a split directory produced by "
            f"gen_aec_dataset.py (the one containing meta.json and seqs/).")
    with open(path, 'r') as handle:
        return json.load(handle)


def _collect(seqs_dir: str, n_sequences: int) -> Dict[int, List[dict]]:
    """sequence_id -> ordered chunk metadata. Every id in range must be complete.

    ``n_sequences`` (from meta.json, this run's own plan) bounds which
    sidecars are eligible: without the upper bound below, generating a large
    corpus and then re-running with a smaller --hours into the SAME --output
    would leave the earlier, now out-of-range sequence files on disk, and a
    later pack of that directory would silently include content the current
    run never asked for. Those are the ONLY ids allowed to be missing --
    meta.json declares this run produced ids 0..n_sequences-1, so a gap or a
    partial sequence anywhere in that range means the corpus on disk is not
    the one meta.json describes (an interrupted run, a partial rsync, a
    sidecar deleted after the fact), and packing it into a shard anyway would
    silently ship a shrunk corpus that still claims meta.json's full size.
    """
    sequences: Dict[int, List[dict]] = {}
    for meta_path in sorted(glob.glob(os.path.join(seqs_dir, '[0-9]*.json'))):
        sequence_id = int(os.path.splitext(os.path.basename(meta_path))[0])
        if sequence_id >= n_sequences:
            print(f"  ⚠ sequence {sequence_id:06d}: outside this run's "
                  f"n_sequences={n_sequences} (stale from a larger prior "
                  f"generation?), skipped")
            continue
        with open(meta_path, 'r') as handle:
            chunk_meta = json.load(handle)
        wavs = [
            os.path.join(seqs_dir, f"{sequence_id:06d}_{index:03d}.wav")
            for index in range(len(chunk_meta))
        ]
        missing = [path for path in wavs if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"sequence {sequence_id:06d}: {len(missing)} chunk wav(s) "
                f"missing (e.g. {missing[0]}), but this id is within this "
                f"run's n_sequences={n_sequences}. Re-run gen_aec_dataset.py "
                f"--resume to complete it before packing -- packing a "
                f"shrunk sequence would silently ship it as if it were whole."
            )
        for index, meta in enumerate(chunk_meta):
            meta['_wav'] = wavs[index]
            if meta.get('chunk_index') != index:
                raise ValueError(
                    f"{meta_path}: chunk_index {meta.get('chunk_index')} at "
                    f"position {index}; the sidecar is not in chunk order")
        sequences[sequence_id] = chunk_meta
    missing_ids = sorted(set(range(n_sequences)) - set(sequences))
    if missing_ids:
        raise FileNotFoundError(
            f"{len(missing_ids)} of {n_sequences} sequence(s) declared by "
            f"meta.json have no sidecar under {seqs_dir} at all (e.g. "
            f"sequence {missing_ids[0]:06d}). Re-run gen_aec_dataset.py "
            f"--resume to fill the gap before packing."
        )
    return sequences


def _save_shard(clips, metas, shard_index, args, run_meta) -> str:
    data = torch.stack(clips).to(
        torch.float16 if args.dtype == 'float16' else torch.float32)
    path = os.path.join(args.output, f"shard_{shard_index:05d}.pt")
    torch.save({
        'stems': list(STEM_ORDER),
        'data': data,
        'sr': int(run_meta['sr']),
        'meta': metas,
        'generator_commit': run_meta.get('generator_commit', 'unknown'),
        'config_hash': run_meta.get('config_hash', 'unknown'),
        'manifest_version': run_meta['manifest_version'],
        'manifest_seed': run_meta.get('manifest_seed'),
        'linear_aec': run_meta['linear_aec'],
        'linear_aec_contract_hash': run_meta['linear_aec_contract_hash'],
    }, path)
    return path


def pack(args):
    run_meta = _read_run_meta(args.input)
    manifest_version = run_meta.get('manifest_version')
    if not isinstance(manifest_version, str) or not manifest_version:
        raise ValueError(
            f"{args.input}/meta.json has no valid manifest_version; "
            "re-render it with the current generator"
        )
    declared_stems = run_meta.get('stems')
    if declared_stems is None or list(declared_stems) != list(STEM_ORDER):
        raise ValueError(
            f"{args.input}/meta.json declares stems {declared_stems}, "
            f"but this code packs {list(STEM_ORDER)}; the channel order is the "
            f"one thing a consumer cannot detect being wrong.")
    linear_aec = LinearAecContract.from_dict(run_meta.get('linear_aec'))
    contract_hash = linear_aec.fingerprint()
    if run_meta.get('linear_aec_contract_hash') != contract_hash:
        raise ValueError(
            f"{args.input}/meta.json linear_aec_contract_hash does not match "
            "its linear_aec contract"
        )

    sequences = _collect(os.path.join(args.input, 'seqs'), int(run_meta['n_sequences']))
    total_chunks = sum(len(chunks) for chunks in sequences.values())
    if total_chunks != int(run_meta['n_chunks']):
        # _collect already enforces every declared sequence_id is present and
        # self-consistent; this is a cheap belt-and-suspenders check against
        # meta.json's own declared total in case that per-sequence agreement
        # somehow still summed to the wrong aggregate.
        raise ValueError(
            f"{args.input}/meta.json declares n_chunks={run_meta['n_chunks']}, "
            f"but the collected sequences total {total_chunks}"
        )
    os.makedirs(args.output, exist_ok=True)

    bytes_per = 2 if args.dtype == 'float16' else 4
    approx = total_chunks * len(STEM_ORDER) * int(run_meta['chunk_samples']) * bytes_per
    print(f"{len(sequences)} sequences / {total_chunks} chunks -> {args.output}")
    print(f"  sr={run_meta['sr']}, T={run_meta['chunk_samples']}, "
          f"dtype={args.dtype}, ~{approx / 1024 ** 3:.1f} GB")

    clips: List[torch.Tensor] = []
    metas: List[dict] = []
    shard_index = 0
    written: List[str] = []
    expected_t = int(run_meta['chunk_samples'])

    progress = tqdm.tqdm(total=total_chunks, desc="Packing")
    for sequence_id in sorted(sequences):
        chunk_meta = sequences[sequence_id]
        # Flush BEFORE adding, so a sequence is never split across shards.
        if clips and len(clips) + len(chunk_meta) > args.shard_clips:
            written.append(_save_shard(clips, metas, shard_index, args, run_meta))
            shard_index += 1
            clips, metas = [], []

        for meta in chunk_meta:
            if meta.get('manifest_version') != manifest_version:
                raise ValueError(
                    f"sequence {sequence_id} chunk {meta.get('chunk_index')} "
                    "was materialized with a different or missing manifest_version"
                )
            if meta.get('config_hash') != run_meta.get('config_hash'):
                raise ValueError(
                    f"sequence {sequence_id} chunk {meta.get('chunk_index')} "
                    "was materialized with a different or missing config_hash "
                    "than meta.json declares for this run"
                )
            if meta.get('linear_aec_contract_hash') != contract_hash:
                raise ValueError(
                    f"sequence {sequence_id} chunk {meta.get('chunk_index')} "
                    "was materialized with a different or missing linear AEC contract"
                )
            audio, sr = torchaudio.load(meta.pop('_wav'))
            if sr != run_meta['sr']:
                raise ValueError(f"sequence {sequence_id}: wav sr={sr}, "
                                 f"meta.json says {run_meta['sr']}")
            if audio.shape[0] != len(STEM_ORDER):
                raise ValueError(f"sequence {sequence_id}: {audio.shape[0]} "
                                 f"channels, expected {len(STEM_ORDER)}")
            if audio.shape[1] != expected_t:
                raise ValueError(f"sequence {sequence_id}: T={audio.shape[1]}, "
                                 f"expected {expected_t}")
            if not torch.isfinite(audio).all():
                raise ValueError(
                    f"sequence {sequence_id} chunk {meta.get('chunk_index')}: "
                    "wav contains NaN or Inf"
                )
            clips.append(audio)
            metas.append(meta)
            progress.update(1)
    progress.close()

    if clips:
        written.append(_save_shard(clips, metas, shard_index, args, run_meta))

    index = {
        'stems': list(STEM_ORDER),
        'sr': int(run_meta['sr']),
        'chunk_samples': expected_t,
        'split': run_meta.get('split'),
        'n_sequences': len(sequences),
        'n_chunks': total_chunks,
        'dtype': args.dtype,
        'shards': [os.path.basename(path) for path in written],
        'generator_commit': run_meta.get('generator_commit', 'unknown'),
        'config_hash': run_meta.get('config_hash', 'unknown'),
        'manifest_version': manifest_version,
        'manifest_seed': run_meta.get('manifest_seed'),
        'linear_aec': linear_aec.as_dict(),
        'linear_aec_contract_hash': contract_hash,
    }
    with open(os.path.join(args.output, 'index.json'), 'w') as handle:
        json.dump(index, handle, indent=2, sort_keys=True)

    print(f"Done: {len(written)} shard(s), {total_chunks} chunks -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Pack rendered AEC stem WAVs into .pt shards')
    parser.add_argument('--input', required=True,
                        help='A split directory from gen_aec_dataset.py '
                             '(contains meta.json and seqs/)')
    parser.add_argument('--output', required=True, help='Shard output directory')
    parser.add_argument('--shard-clips', type=int, default=512,
                        help='Soft cap on chunks per shard; a sequence is never '
                             'split, so a shard may exceed it by one sequence')
    parser.add_argument('--dtype', default='float32',
                        choices=['float32', 'float16'],
                        help='⚠ float16 halves the size and costs ~3 decimal '
                             'digits; the stem-sum identity then holds to ~1e-3')
    return parser


if __name__ == '__main__':
    pack(build_parser().parse_args())
