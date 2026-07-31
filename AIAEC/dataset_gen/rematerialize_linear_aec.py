#!/usr/bin/env python3
"""Recompute AIAEC stem six from existing acoustic stem WAVs.

This does not rerender speech/noise/RIR mixtures. It concatenates each parent
sequence in ``chunk_index`` order, runs one fresh Python PBFDKF over that full
waveform, then atomically rewrites every chunk with ``linear_error`` as channel
six. The sequence JSON sidecar is updated last and is the resume marker.
"""

from __future__ import annotations

import argparse
import configparser
import glob
import json
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


WAV_ENCODINGS = {
    "float32": dict(encoding="PCM_F", bits_per_sample=32),
    "int16": dict(encoding="PCM_S", bits_per_sample=16),
}


def _load_json(path: str):
    with open(path, "r") as handle:
        return json.load(handle)


def _atomic_json(path: str, value) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _wav_paths(seqs_dir: str, sequence_id: int, n_chunks: int) -> List[str]:
    return [
        os.path.join(seqs_dir, f"{sequence_id:06d}_{index:03d}.wav")
        for index in range(n_chunks)
    ]


def _load_sequence(
    wav_paths: List[str], expected_sr: int, expected_t: int
) -> Tuple[List[torch.Tensor], int]:
    chunks = []
    channels_seen = set()
    for path in wav_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
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
        # A killed prior re-materialization may leave a mixture of five- and
        # six-channel files while the sidecar still correctly marks the
        # sequence incomplete. Always recover from the first five stems.
        channels_seen.add(int(audio.shape[0]))
        chunks.append(audio[:len(BASE_STEM_ORDER)].float())
    return chunks, max(channels_seen)


def _sequence_is_current(
    chunk_meta: List[dict], wav_paths: List[str], contract_hash: str,
    expected_sr: int, expected_t: int,
) -> bool:
    if not chunk_meta:
        return False
    for chunk_index, (meta, path) in enumerate(zip(chunk_meta, wav_paths)):
        if (
            meta.get("chunk_index") != chunk_index
            or meta.get("linear_aec_contract_hash") != contract_hash
        ):
            return False
        if not os.path.isfile(path):
            return False
        info = torchaudio.info(path)
        if (
            info.num_channels != len(STEM_ORDER)
            or info.sample_rate != expected_sr
            or info.num_frames != expected_t
        ):
            return False
    return True


def rematerialize(args) -> None:
    run_meta_path = os.path.join(args.input, "meta.json")
    run_meta = _load_json(run_meta_path)
    sr = int(run_meta["sr"])
    chunk_samples = int(run_meta["chunk_samples"])
    declared = tuple(run_meta.get("stems", ()))
    if declared not in (BASE_STEM_ORDER, STEM_ORDER):
        raise ValueError(
            f"{run_meta_path}: stems={declared}, expected legacy "
            f"{BASE_STEM_ORDER} or current {STEM_ORDER}"
        )

    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")
    if cfg.getint("signal", "sr") != sr:
        raise ValueError(
            f"config sr={cfg.getint('signal', 'sr')} but dataset sr={sr}"
        )
    contract: LinearAecContract = linear_aec_contract_from_config(cfg)
    if chunk_samples % contract.hop_size:
        raise ValueError(
            f"chunk_samples={chunk_samples} is not divisible by PBFDKF "
            f"hop={contract.hop_size}"
        )
    contract_hash = contract.fingerprint()

    encoding = (
        run_meta.get("wav_encoding", "float32")
        if args.wav_encoding == "auto" else args.wav_encoding
    )
    if encoding not in WAV_ENCODINGS:
        raise ValueError(f"unsupported WAV encoding {encoding!r}")

    seqs_dir = os.path.join(args.input, "seqs")
    meta_paths = sorted(glob.glob(os.path.join(seqs_dir, "[0-9]*.json")))
    if not meta_paths:
        raise FileNotFoundError(f"no sequence metadata under {seqs_dir}")

    rewritten = 0
    for meta_path in tqdm.tqdm(meta_paths, desc="linear-aec"):
        sequence_id = int(os.path.splitext(os.path.basename(meta_path))[0])
        chunk_meta = _load_json(meta_path)
        wav_paths = _wav_paths(seqs_dir, sequence_id, len(chunk_meta))
        if args.resume and _sequence_is_current(
            chunk_meta, wav_paths, contract_hash, sr, chunk_samples
        ):
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
            six = torch.cat([base, error_chunk], dim=0).contiguous()
            tmp = path + ".tmp.wav"
            torchaudio.save(tmp, six, sr, **WAV_ENCODINGS[encoding])
            check, check_sr = torchaudio.load(tmp)
            if check_sr != sr or check.shape != six.shape:
                raise RuntimeError(
                    f"{tmp}: failed six-channel round-trip validation"
                )
            os.replace(tmp, path)

        for meta in chunk_meta:
            meta["linear_aec_contract_hash"] = contract_hash
        _atomic_json(meta_path, chunk_meta)
        rewritten += 1

    run_meta["stems"] = list(STEM_ORDER)
    run_meta["linear_aec"] = contract.as_dict()
    run_meta["linear_aec_contract_hash"] = contract_hash
    run_meta["wav_encoding"] = encoding
    run_meta["linear_aec_rematerialized_sequences"] = rewritten
    _atomic_json(run_meta_path, run_meta)
    print(
        f"Done: {rewritten} sequence(s) rewritten; contract={contract_hash[:12]}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="Generated split directory containing meta.json and seqs/",
    )
    parser.add_argument(
        "--config", required=True,
        help="Dataset config containing [signal] and optional [linear_aec]",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip sequences whose six-channel WAVs and contract markers match",
    )
    parser.add_argument(
        "--wav-encoding", default="auto",
        choices=("auto", "float32", "int16"),
        help="Default: preserve meta.json wav_encoding",
    )
    return parser


if __name__ == "__main__":
    rematerialize(build_parser().parse_args())
