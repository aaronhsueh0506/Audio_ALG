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
      'meta' : list of N {'sequence_id', 'chunk_index'},
      'linear_aec'              : the frozen PBFDKF contract, from --config,
      'linear_aec_contract_hash': its fingerprint,
    }

Input is rendered audio only: ``<split>/seqs/SSSSSS_CCC.wav``. No meta.json, no
per-sequence sidecar, no index.json on the way out either -- the packer
discovers sequences and chunk order from the filenames, and takes T, sr and the
channel count from the WAVs themselves.  A corpus therefore stays packable
after being copied, rsynced or trimmed by hand.

⚠ ``--config`` is the one non-audio input, and it is required: the frozen
linear-AEC contract that produced the ``linear_error`` stem cannot be recovered
from a WAV, and inference has to construct the SAME one to reproduce ``D_hat``
(see ``AIAEC.training_common.LinearAecEngine``). Pass the config the corpus was
generated with. Nothing cross-checks that claim any more, so passing a
different one silently mislabels the corpus.

⚠ Chunks are packed in ``(sequence_id, chunk_index)`` order and a sequence is
never split across shards. Training later randomizes global chunk indices, but
the physical order is retained for full-sequence reconstruction and streaming
evaluation.

Usage:
    python3 pack_aec_dataset.py --config config.ini --input data_aec/all \\
        --output data_aec/packed/all --shard-clips 512
"""

import argparse
import configparser
import glob
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
from .linear_aec import linear_aec_contract_from_config  # noqa: E402
from .seq_layout import scan_chunks, stale_temp_files  # noqa: E402


def _resolve_seqs_dir(input_dir: str) -> str:
    """Accept either a split directory or its seqs/ subdirectory.

    gen_aec_dataset.py prints the split directory in its "Next:" line, but
    pointing straight at seqs/ is just as sensible now that there is nothing
    else in the split directory to read.
    """
    if os.path.isdir(os.path.join(input_dir, 'seqs')):
        return os.path.join(input_dir, 'seqs')
    return input_dir


def _collect(seqs_dir: str) -> Dict[int, List[str]]:
    """``sequence_id -> [chunk wav path, ...]`` in chunk order.

    Every sequence present must have chunks 0..n-1 with no hole: a gap means
    the render was interrupted or a file was deleted, and packing around it
    would silently ship a sequence that is shorter than it looks while its
    remaining chunks keep their original indices.
    """
    sequences = scan_chunks(seqs_dir)
    if not sequences:
        raise FileNotFoundError(
            f"no SSSSSS_CCC.wav chunk files under {seqs_dir}. Point --input at "
            f"a split directory from gen_aec_dataset.py (or directly at its "
            f"seqs/).")
    for sequence_id, paths in sequences.items():
        expected = [
            os.path.join(seqs_dir, f"{sequence_id:06d}_{index:03d}.wav")
            for index in range(len(paths))
        ]
        if paths != expected:
            have = [os.path.basename(p) for p in paths]
            raise FileNotFoundError(
                f"sequence {sequence_id:06d} has a gap in its chunk numbering: "
                f"{have}. Re-run gen_aec_dataset.py --resume to complete it "
                f"before packing -- packing a sequence with a hole in it would "
                f"silently ship it as if it were whole.")
    return sequences


def _save_shard(clips, metas, shard_index, args, header) -> tuple:
    """Write one unpublished shard and return ``(temporary, final)``.

    The final name is only made visible after every input chunk has been
    validated and every shard has been serialized. A failed pack therefore
    cannot leave a truncated ``shard_*.pt`` that a trainer mistakes for a
    complete corpus.
    """
    data = torch.stack(clips).to(
        torch.float16 if args.dtype == 'float16' else torch.float32)
    path = os.path.join(args.output, f"shard_{shard_index:05d}.pt")
    temporary = path + '.tmp'
    torch.save({
        'stems': list(STEM_ORDER),
        'data': data,
        'meta': metas,
        **header,
    }, temporary)
    return temporary, path


def pack(args):
    if args.shard_clips <= 0:
        raise ValueError(f"--shard-clips must be positive, got {args.shard_clips}")
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(
            f"config not found: {args.config}. The frozen linear-AEC contract "
            f"is rebuilt from it (see this file's docstring); pass the config "
            f"the corpus was generated with.")
    linear_aec = linear_aec_contract_from_config(cfg)

    seqs_dir = _resolve_seqs_dir(args.input)
    sequences = _collect(seqs_dir)
    total_chunks = sum(len(paths) for paths in sequences.values())

    stale = stale_temp_files(seqs_dir)
    if stale:
        print(f"  ⚠ ignoring {len(stale)} leftover tmp.*.wav (interrupted "
              f"write; not packed)")

    # A shard directory that already holds shards from an earlier, differently
    # configured pack would be loaded as one corpus by PackedAecDataset, which
    # globs shard_*.pt. There is no index file to disambiguate them, so stop
    # here rather than quietly mix two packs.
    os.makedirs(args.output, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(args.output, 'shard_*.pt')))
    if existing:
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output} already contains {len(existing)} shard(s), e.g. "
                f"{os.path.basename(existing[0])}. Loading a directory takes "
                f"every shard_*.pt in it, so a leftover from an earlier pack "
                f"would silently join this corpus. Pass --overwrite to replace "
                f"them after a new pack succeeds, or use an empty directory.")
        print(f"  --overwrite: {len(existing)} pre-existing shard(s) will be "
              "replaced only after the new pack finishes successfully")
    stale_pack_temps = sorted(glob.glob(
        os.path.join(args.output, 'shard_*.pt.tmp')
    ))
    for path in stale_pack_temps:
        os.remove(path)
    if stale_pack_temps:
        print(f"  removed {len(stale_pack_temps)} stale temporary shard(s)")

    # Geometry comes from the audio, since nothing declares it any more.
    first = sequences[min(sequences)][0]
    info = torchaudio.info(first)
    sr = int(info.sample_rate)
    expected_t = int(info.num_frames)
    if info.num_channels != len(STEM_ORDER):
        raise ValueError(
            f"{first}: {info.num_channels} channels, expected "
            f"{len(STEM_ORDER)} ({list(STEM_ORDER)}). This does not look like "
            f"a rendered AEC corpus.")
    if sr != linear_aec.sample_rate:
        raise ValueError(
            f"{first} is {sr} Hz but --config's linear AEC grid is "
            f"{linear_aec.sample_rate} Hz -- wrong config for this corpus.")

    header = {
        'sr': sr,
        'linear_aec': linear_aec.as_dict(),
        'linear_aec_contract_hash': linear_aec.fingerprint(),
    }

    bytes_per = 2 if args.dtype == 'float16' else 4
    approx = total_chunks * len(STEM_ORDER) * expected_t * bytes_per
    print(f"{len(sequences)} sequences / {total_chunks} chunks -> {args.output}")
    print(f"  sr={sr}, T={expected_t}, dtype={args.dtype}, "
          f"~{approx / 1024 ** 3:.1f} GB")

    clips: List[torch.Tensor] = []
    metas: List[dict] = []
    shard_index = 0
    staged: List[tuple] = []

    progress = tqdm.tqdm(total=total_chunks, desc="Packing")
    try:
        for sequence_id in sorted(sequences):
            paths = sequences[sequence_id]
            # Flush BEFORE adding, so a sequence is never split across shards.
            if clips and len(clips) + len(paths) > args.shard_clips:
                staged.append(_save_shard(
                    clips, metas, shard_index, args, header
                ))
                shard_index += 1
                clips, metas = [], []

            for chunk_index, path in enumerate(paths):
                audio, chunk_sr = torchaudio.load(path)
                if chunk_sr != sr:
                    raise ValueError(f"{path}: sr={chunk_sr}, but this corpus is "
                                     f"{sr} Hz")
                if audio.shape[0] != len(STEM_ORDER):
                    raise ValueError(f"{path}: {audio.shape[0]} channels, expected "
                                     f"{len(STEM_ORDER)}")
                if audio.shape[1] != expected_t:
                    raise ValueError(f"{path}: T={audio.shape[1]}, expected "
                                     f"{expected_t}")
                if not torch.isfinite(audio).all():
                    raise ValueError(f"{path}: contains NaN or Inf")
                clips.append(audio)
                metas.append({'sequence_id': int(sequence_id),
                              'chunk_index': int(chunk_index)})
                progress.update(1)

        if clips:
            staged.append(_save_shard(
                clips, metas, shard_index, args, header
            ))
    except BaseException:
        for temporary, _final in staged:
            if os.path.exists(temporary):
                os.remove(temporary)
        raise
    finally:
        progress.close()

    # Publish only after the complete input inventory has passed validation.
    # Existing shards are kept throughout validation/serialization, so a
    # failure in either phase leaves the previous usable pack intact.
    for path in existing:
        os.remove(path)
    for temporary, final in staged:
        os.replace(temporary, final)

    print(f"Done: {len(staged)} shard(s), {total_chunks} chunks -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Pack rendered AEC stem WAVs into .pt shards')
    parser.add_argument('--config', default='config.ini',
                        help='The config the corpus was generated with. Only '
                             'the frozen linear-AEC contract is taken from it '
                             '(it cannot be recovered from a WAV, and '
                             'inference needs the same one)')
    parser.add_argument('--input', required=True,
                        help='A split directory from gen_aec_dataset.py, or '
                             'its seqs/ subdirectory')
    parser.add_argument('--output', required=True, help='Shard output directory')
    parser.add_argument('--shard-clips', type=int, default=512,
                        help='Soft cap on chunks per shard; a sequence is never '
                             'split, so a shard may exceed it by one sequence')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace shard_*.pt already in --output after the '
                             'new pack validates and serializes completely '
                             '(otherwise an existing shard is an error)')
    parser.add_argument('--dtype', default='float32',
                        choices=['float32', 'float16'],
                        help='⚠ float16 halves the size and costs ~3 decimal '
                             'digits; the stem-sum identity then holds to ~1e-3')
    return parser


if __name__ == '__main__':
    pack(build_parser().parse_args())
