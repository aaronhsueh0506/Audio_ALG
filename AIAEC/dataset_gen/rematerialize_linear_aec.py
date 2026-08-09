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

⚠ ``--resume`` can only see that a sequence already HAS five channels, not
which contract produced that fifth channel. Re-running after a [linear_aec]
config edit therefore needs a full pass (omit --resume), or the corpus will
silently keep a mix of two contracts.
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from typing import List, Tuple

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

    This is a SHAPE check only -- see the module docstring on what --resume
    can no longer prove.
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

    rewritten = 0
    for sequence_id in tqdm.tqdm(sorted(sequences), desc="linear-aec"):
        wav_paths = sequences[sequence_id]
        expected = [
            os.path.join(seqs_dir, f"{sequence_id:06d}_{index:03d}.wav")
            for index in range(len(wav_paths))
        ]
        if wav_paths != expected:
            raise FileNotFoundError(
                f"sequence {sequence_id:06d} has a gap in its chunk numbering; "
                f"the PBFDKF has to run over the whole sequence in order, so "
                f"there is nothing sensible to rematerialize from")
        if args.resume and _sequence_is_current(wav_paths, sr, chunk_samples):
            continue

        chunks, _old_channels = _load_sequence(wav_paths, sr, chunk_samples)
        far = torch.cat([
            chunk[BASE_STEM_ORDER.index("far_render")] for chunk in chunks
        ])
        mic = torch.cat([
            chunk[BASE_STEM_ORDER.index("mic_postclip")] for chunk in chunks
        ])
        linear_error, _echo_estimate = materialize_linear_error(
            mic, far, contract
        )

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
        rewritten += 1

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
        help="Skip sequences already stored with five channels (shape check "
             "only -- cannot tell WHICH contract wrote that fifth channel)",
    )
    parser.add_argument(
        "--wav-encoding", default="auto",
        choices=("auto", "float32", "int16"),
        help="Default: keep whatever encoding the existing chunks use",
    )
    return parser


if __name__ == "__main__":
    rematerialize(build_parser().parse_args())
