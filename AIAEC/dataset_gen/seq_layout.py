# -*- coding: utf-8 -*-
"""The on-disk layout of a rendered AEC corpus, in one place.

A rendered split is a directory of chunk WAVs and nothing else:

    <split>/seqs/SSSSSS_CCC.wav      5-channel, channels = STEM_ORDER

``SSSSSS`` is the parent sequence id and ``CCC`` its chunk index. That filename
is the entire contract -- there is no sidecar and no run manifest -- so the
generator, the packer and the re-materializer all have to agree on exactly what
counts as a chunk file and how a killed run is made invisible. They agree by
importing from here.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Optional, Tuple

import torchaudio


# SSSSSS_CCC.wav, both parts digits-only. Anything else in seqs/ (a note, a
# rendering of one clip someone dropped in, a `tmp.` file from a killed write)
# is not a chunk and is ignored rather than packed.
CHUNK_RE = re.compile(r'^(\d+)_(\d+)\.wav$')


def chunk_path(seqs_dir: str, sequence_id: int, chunk_index: int) -> str:
    return os.path.join(seqs_dir, f"{sequence_id:06d}_{chunk_index:03d}.wav")


def parse_chunk_name(path: str) -> Optional[Tuple[int, int]]:
    """``(sequence_id, chunk_index)``, or None if this is not a chunk file."""
    match = CHUNK_RE.match(os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def sequence_chunk_paths(seqs_dir: str, sequence_id: int) -> List[str]:
    """Every existing chunk of one sequence, in chunk order."""
    found = []
    for path in glob.glob(os.path.join(seqs_dir, f"{sequence_id:06d}_*.wav")):
        parsed = parse_chunk_name(path)
        if parsed is not None and parsed[0] == sequence_id:
            found.append((parsed[1], path))
    return [path for _, path in sorted(found)]


def scan_chunks(seqs_dir: str) -> dict:
    """``{sequence_id: [path, ...]}`` with each list in chunk order.

    The chunk indices are NOT checked for gaps here -- the packer reports that
    itself, with the sequence named, rather than having it silently absorbed
    into a scan helper.
    """
    by_sequence: dict = {}
    for path in glob.glob(os.path.join(seqs_dir, '*.wav')):
        parsed = parse_chunk_name(path)
        if parsed is None:
            continue
        sequence_id, chunk_index = parsed
        by_sequence.setdefault(sequence_id, []).append((chunk_index, path))
    return {
        sequence_id: [path for _, path in sorted(entries)]
        for sequence_id, entries in sorted(by_sequence.items())
    }


def chunk_indices(seqs_dir: str, sequence_id: int) -> List[int]:
    return [
        parse_chunk_name(path)[1]
        for path in sequence_chunk_paths(seqs_dir, sequence_id)
    ]


def tmp_path(final_path: str) -> str:
    """`tmp.<name>` beside the real file.

    A prefix, not a suffix, for the same two reasons as the NR generator's:
    torchaudio's soundfile backend picks the format from the path's final
    extension (so the temp name must still end in `.wav`), and `tmp.SSSSSS_CCC`
    fails CHUNK_RE, so a temp file is invisible to every scan above.
    """
    return os.path.join(os.path.dirname(final_path),
                        f"tmp.{os.path.basename(final_path)}")


def save_chunk_atomic(seqs_dir: str, sequence_id: int, chunk_index: int,
                      chunk, sample_rate: int, encoding: dict) -> str:
    """Write one chunk through a temp name and rename it into place.

    With the sidecar gone this rename is the ONLY completion marker a chunk
    has: a run killed mid-write leaves a `tmp.` file that no scan sees, rather
    than a truncated WAV that looks like a finished chunk.
    """
    final = chunk_path(seqs_dir, sequence_id, chunk_index)
    temp = tmp_path(final)
    torchaudio.save(temp, chunk, sample_rate, **encoding)
    os.replace(temp, final)
    return final


def stale_temp_files(seqs_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(seqs_dir, 'tmp.*.wav')))
