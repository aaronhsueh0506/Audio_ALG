"""Which sequences the CURRENT linear-AEC contract has already rewritten.

Not in seq_layout (a split is chunk WAVs and nothing else) nor in linear_aec
(the DSP contract); the sidecar sits beside the corpus root, not in ``seqs/``,
so scan_chunks() stays a pure ``*.wav`` scan. ⚠ A sidecar can be separated from
its data: ``--resume`` re-checks every claim against the chunks on disk, and
the packer treats an absent ledger as "no claim".
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Set

LEDGER_NAME = "linear_error.done.json"


def ledger_path(seqs_dir: str) -> str:
    """Beside the corpus root, not inside seqs/, so a glob for chunks cannot
    pick it up and scan_chunks() stays a pure *.wav scan."""
    return os.path.join(os.path.dirname(os.path.abspath(seqs_dir)), LEDGER_NAME)


def _read(seqs_dir: str):
    """The ledger as a dict, or ``None`` if absent or unparseable."""
    try:
        with open(ledger_path(seqs_dir), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def recorded_contract(seqs_dir: str):
    """The contract fingerprint this ledger was written under, or ``None``.

    ``None`` covers both "no ledger" and "not readable as one"; a caller that
    needs to tell those apart checks ``ledger_path`` itself. Exists so the
    packer's gate can ask WHICH contract wrote the corpus -- a question
    ``load_ledger`` answers only as "not yours", which is the same answer it
    gives for a ledger that claims nothing.
    """
    data = _read(seqs_dir)
    if data is None:
        return None
    recorded = data.get("contract")
    return recorded if isinstance(recorded, str) else None


def recorded_identity(seqs_dir: str):
    """The contract DICT this ledger was written under, or ``None``.

    The fingerprint is one-way, so a refusal keyed on it can only show two
    opaque hashes; this returns the producing contract (frontend geometry plus
    `aec_behavior_hash`/`aec_commit`/`aec_source_hash`) so a gate can name the
    build and compare field by field. ``None`` also covers a legacy ledger that
    did not record it; callers say so rather than guess.
    """
    data = _read(seqs_dir)
    if data is None:
        return None
    identity = data.get("linear_aec")
    return identity if isinstance(identity, dict) else None


def load_ledger(seqs_dir: str, contract_hash: str) -> set:
    """Sequence ids this exact contract has already written.

    A ledger from a DIFFERENT contract is discarded whole. Trusting part of it
    is what produces a corpus with two frontends in it, which is the failure
    this file exists to make impossible.
    """
    data = _read(seqs_dir)
    if data is None or data.get("contract") != contract_hash:
        return set()
    return {int(x) for x in data.get("sequences", ())}


def save_ledger(seqs_dir: str, contract_hash: str, done: set,
                linear_aec: dict = None) -> None:
    """Rewritten after every sequence, atomically: a run killed mid-write
    leaves the previous ledger intact, so the worst case is redoing in-flight
    sequences, never recording one that did not finish.

    ``linear_aec`` is the producing contract as a dict, recorded next to the
    one-way fingerprint so a mismatch can say WHICH contract wrote the corpus;
    omitting it writes the legacy shape, which every reader accepts."""
    payload = {"contract": contract_hash,
               "sequences": sorted(int(x) for x in done)}
    if linear_aec is not None:
        payload["linear_aec"] = dict(linear_aec)
    path = ledger_path(seqs_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)



def claimed_vs_present(seqs_dir: str, contract_hash: str,
                       present: Iterable[int]) -> tuple:
    """``(claimed, missing, extra)`` for this contract against ``present``.

    The one place that compares a ledger to a corpus, because both the resume
    path and the packer's gate need the same comparison and had started to
    grow their own.
    """
    present = set(int(x) for x in present)
    claimed = load_ledger(seqs_dir, contract_hash)
    return claimed, sorted(present - claimed), sorted(claimed - present)
