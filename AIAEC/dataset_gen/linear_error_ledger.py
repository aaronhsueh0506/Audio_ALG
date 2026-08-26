"""Which sequences the CURRENT linear-AEC contract has already rewritten.

Deliberately not in seq_layout: that module's whole claim is that a rendered
split is chunk WAVs "and nothing else -- there is no sidecar and no run
manifest". This IS a sidecar. It is also not in linear_aec, which is the DSP
contract itself and has no business doing disk bookkeeping.

The file lives beside the corpus root rather than inside ``seqs/``, so a scan
for chunk files cannot pick it up and scan_chunks() stays a pure ``*.wav``
scan.

⚠ A sidecar can be separated from the data it describes. Copy, rsync or trim a
corpus by hand and the ledger over-claims; delete it and nothing is claimed at
all. Both are handled -- ``--resume`` re-checks every claim against the chunks
on disk, and the packer treats an absent ledger as "no claim" rather than an
error -- but neither would be necessary if each chunk carried its own
provenance. See the packer's gate for what that costs today.
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

    The fingerprint above is one-way: it says two contracts differ but never
    how, so a refusal keyed on it can only show the operator two opaque
    hashes. This returns the producing contract itself -- frontend geometry
    plus `aec_behavior_hash`/`aec_commit`/`aec_source_hash` -- so a gate can
    name the build that wrote the corpus and compare field by field.

    ``None`` also covers a LEGACY ledger, written before this was recorded.
    That is not an error and must never be treated as one: the in-flight runs
    this shipped into are writing legacy-shaped ledgers right now, and they
    stay readable by every path here. A caller that needs the distinction
    tells the operator so explicitly rather than guessing an identity.
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
    """Rewritten after every sequence, atomically. A run killed mid-write
    leaves the previous ledger intact, so the worst case is redoing the
    sequences that were in flight -- never recording one that did not finish.

    ``linear_aec`` is the producing contract as a dict. Recorded next to the
    fingerprint, not instead of it: the fingerprint is what ``load_ledger``
    keys on and is one-way, so a ledger carrying only that can say a corpus
    was written by a different contract but never WHICH -- which leaves an
    operator comparing two opaque hashes with no way to learn what the other
    one was. Omitting it writes the legacy shape, which every reader here
    still accepts; a corpus whose ledger predates this field is identified by
    reconstruction instead (see MIGRATED_SOURCE_PROVENANCE).

    Yes, this rewrites the whole file each time, which is O(N^2) over a run.
    Measured before dismissing it: at the 28,800 sequences a 200-hour corpus
    produces, one call is 6.5 ms and the file is 186 KB, so the whole run
    spends 94 s here against roughly 61 CPU-hours of PBFDKF -- 0.043%. It
    stays under 1% until about 4,800 hours of audio. In the parallel case it
    runs in the parent while it would otherwise be idle waiting on workers,
    and only saturates past --jobs 1170. The contract dict adds a fixed ~600
    bytes to a file whose cost is dominated by the sequence list.

    An append-only log would be O(1) but needs a header line for the contract
    key, tolerance for a torn last line, and a compaction path for the
    re-queue rewrite -- three new crash-recovery edge cases in the one
    component whose entire job is crash safety, to buy back 94 seconds."""
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
