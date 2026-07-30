# -*- coding: utf-8 -*-
"""Pack rendered stem WAVs into .pt shards.

Same motivation as ``dataset_gen/pack_dataset.py``: per-file I/O dominates
training on a corpus of this size.  The shard layout is the packed format every
AEC model project consumes:

    {
      'stems': ['far_render','echo','near_speech','local_noise',
                'mic_preclip','mic_postclip'],   # channel order, fixed
      'data' : float32 tensor (N, 6, T),
      'sr'   : int,
      'meta' : list of N dicts,
      'generator_commit': str,
      'config_hash'     : str,
    }

⚠ Chunks are packed in (sequence_id, chunk_index) order and a sequence is never
split across shards.  ``SequenceChunkSampler`` reconstructs lanes from the
metadata, so out-of-order packing would not crash -- it would just feed a
sequence backwards, which reads as a convergence failure rather than a data bug.
Keeping the order is the cheap half of that guarantee; the sampler asserts the
other half.

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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    __package__ = 'dataset_gen.aec'

from .aec_features import STEM_ORDER  # noqa: E402


def _read_run_meta(input_dir: str) -> dict:
    path = os.path.join(input_dir, 'meta.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Point --input at a split directory produced by "
            f"gen_aec_dataset.py (the one containing meta.json and seqs/).")
    with open(path, 'r') as handle:
        return json.load(handle)


def _collect(seqs_dir: str) -> Dict[int, List[dict]]:
    """sequence_id -> ordered chunk metadata, for sequences that are complete."""
    sequences: Dict[int, List[dict]] = {}
    for meta_path in sorted(glob.glob(os.path.join(seqs_dir, '[0-9]*.json'))):
        sequence_id = int(os.path.splitext(os.path.basename(meta_path))[0])
        with open(meta_path, 'r') as handle:
            chunk_meta = json.load(handle)
        wavs = [
            os.path.join(seqs_dir, f"{sequence_id:06d}_{index:03d}.wav")
            for index in range(len(chunk_meta))
        ]
        missing = [path for path in wavs if not os.path.isfile(path)]
        if missing:
            # A sidecar exists but its audio does not: an interrupted run that
            # was resumed with a different chunk length, or a partial copy.
            # Dropping it silently would shrink the corpus with no record.
            print(f"  ⚠ sequence {sequence_id:06d}: {len(missing)} chunk(s) "
                  f"missing, sequence skipped")
            continue
        for index, meta in enumerate(chunk_meta):
            meta['_wav'] = wavs[index]
            if meta.get('chunk_index') != index:
                raise ValueError(
                    f"{meta_path}: chunk_index {meta.get('chunk_index')} at "
                    f"position {index}; the sidecar is not in chunk order")
        sequences[sequence_id] = chunk_meta
    if not sequences:
        raise FileNotFoundError(f"no complete sequences under {seqs_dir}")
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
    }, path)
    return path


def pack(args):
    run_meta = _read_run_meta(args.input)
    if list(run_meta.get('stems', STEM_ORDER)) != list(STEM_ORDER):
        raise ValueError(
            f"{args.input}/meta.json declares stems {run_meta.get('stems')}, "
            f"but this code packs {list(STEM_ORDER)}; the channel order is the "
            f"one thing a consumer cannot detect being wrong.")

    sequences = _collect(os.path.join(args.input, 'seqs'))
    total_chunks = sum(len(chunks) for chunks in sequences.values())
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
