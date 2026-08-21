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


def load_ledger(seqs_dir: str, contract_hash: str) -> set:
    """Sequence ids this exact contract has already written.

    A ledger from a DIFFERENT contract is discarded whole. Trusting part of it
    is what produces a corpus with two frontends in it, which is the failure
    this file exists to make impossible.
    """
    try:
        with open(ledger_path(seqs_dir), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return set()
    if data.get("contract") != contract_hash:
        return set()
    return {int(x) for x in data.get("sequences", ())}


def save_ledger(seqs_dir: str, contract_hash: str, done: set) -> None:
    """Rewritten after every sequence, atomically. A run killed mid-write
    leaves the previous ledger intact, so the worst case is redoing the
    sequences that were in flight -- never recording one that did not finish.

    Yes, this rewrites the whole file each time, which is O(N^2) over a run.
    Measured before dismissing it: at the 28,800 sequences a 200-hour corpus
    produces, one call is 6.5 ms and the file is 186 KB, so the whole run
    spends 94 s here against roughly 61 CPU-hours of PBFDKF -- 0.043%. It
    stays under 1% until about 4,800 hours of audio. In the parallel case it
    runs in the parent while it would otherwise be idle waiting on workers,
    and only saturates past --jobs 1170.

    An append-only log would be O(1) but needs a header line for the contract
    key, tolerance for a torn last line, and a compaction path for the
    re-queue rewrite -- three new crash-recovery edge cases in the one
    component whose entire job is crash safety, to buy back 94 seconds."""
    path = ledger_path(seqs_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"contract": contract_hash,
                   "sequences": sorted(int(x) for x in done)}, handle)
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
