"""The packer must refuse a corpus whose fifth channel was only partly rebuilt.

This is the one failure with no downstream detector: once packed, the fifth
channel is just samples, and a shard set labelled with one contract while
holding two looks exactly like a correct one.

⚠ The cases below that call the guard DIRECTLY are testing the guard's own
logic and nothing else. They will pass whether or not pack() actually calls
it -- which is not hypothetical: renaming the guard once left the call site
pointing at the old name, and every one of these stayed green while any real
pack raised NameError. test_pack_itself_runs_the_guard is the one that closes
that gap, and it is the reason to reach for pack() rather than the helper
whenever a test can afford to.
"""
import argparse
import dataclasses
import configparser
import glob
import json
import os

import pytest
import torch
import torchaudio

from AIAEC.dataset_gen import pack_aec_dataset as P
from AIAEC.dataset_gen import rematerialize_linear_aec as R
from AIAEC.dataset_gen.linear_aec import (
    RETIRED_BEHAVIOR_HASHES,
    linear_aec_contract_from_config,
    migrated_ledger_fingerprints,
)
from AIAEC.dataset_gen.linear_error_ledger import LEDGER_NAME, save_ledger

from ledger_corpus import CONFIG, build_corpus, shipped_contract

N_SEQ, N_CHUNK = 3, 2


def _corpus(root):
    return build_corpus(root, N_SEQ, N_CHUNK)


def _gate(root):
    seqs = os.path.join(root, "seqs")
    P._require_complete_rematerialization(
        seqs, P._collect(seqs), shipped_contract())


# The identity of the builds the 200-hour corpus was materialized under.
# ACCEPTED_BEHAVIOR_HASH_MIGRATIONS carried it forward until 2026-09-04; it is
# now RETIRED, so a ledger from any of these revisions must be refused whether
# it is legacy-shaped or records its contract. Spelled out rather than read
# from the tables on purpose: a test fed its inputs from the table it checks
# would still pass if the table were emptied.
#
# A behaviour hash is insensitive to comments and formatting, so it survives
# every docs-only lib/aec commit while `aec_commit` moves under it -- four
# revisions carried this one, measured, and each wrote a DIFFERENT ledger
# fingerprint because `fingerprint()` folds in the commit.
MIGRATED_FROM_HASH = (
    "37ed5ad9b75ce42902361d8195fcf04a650b940744ec036a16c8736dec9d5061")
_SOURCE_HASH_EARLY = (
    "2d5b119d8a69126b94acb98ebfe44d6982aeb12c41bdd46b51ddf0ae20843faf")
_SOURCE_HASH_LATE = (
    "9380c512bca01b8da842e22426c66c016335d33ab4b232f78bafd9cd1efe39fe")
MIGRATED_FROM_REVISIONS = (
    ("2f66c17c03b1fc2c96dd9cd74b15c543824cc757", _SOURCE_HASH_EARLY),
    ("fc5103cff82e7add40325bc44031a9cb0048ccf0", _SOURCE_HASH_LATE),
    ("631eb78665eda1e5e0d06f17e046f448e8938328", _SOURCE_HASH_LATE),
    ("d5193ad6b58efc54f13cbc71980a3e5659c7388d", _SOURCE_HASH_LATE),
)
MIGRATED_FROM_COMMIT, MIGRATED_FROM_SOURCE_HASH = MIGRATED_FROM_REVISIONS[-1]

# The revision immediately BEFORE the range: a real lib/aec commit that carried
# a different behaviour hash. Used as the unrelated build, so the refusal is
# tested against something plausible rather than against 64 zeros.
UNCOVERED_COMMIT = "070787b33d53c10ff586f12b705dcc56b675d737"


def _source_contract(commit, source_hash, behavior=MIGRATED_FROM_HASH):
    """The contract a build with that provenance made of this config."""
    return dataclasses.replace(
        shipped_contract(),
        aec_behavior_hash=behavior,
        aec_commit=commit,
        aec_source_hash=source_hash,
    )


def _migrated_source_fingerprint(commit=None, source_hash=None):
    """What a migrated-from build wrote into the ledger for this config."""
    if commit is None:
        commit, source_hash = MIGRATED_FROM_COMMIT, MIGRATED_FROM_SOURCE_HASH
    return _source_contract(commit, source_hash).fingerprint()


def test_no_ledger_is_not_an_error(tmp_path):
    """A corpus generated in one pass never had a ledger; there is nothing to
    contradict. The guard is against a PARTIAL claim, not an absent one."""
    root = _corpus(str(tmp_path / "c"))
    assert not os.path.exists(os.path.join(root, LEDGER_NAME))
    _gate(root)


def test_a_complete_ledger_passes(tmp_path):
    root = _corpus(str(tmp_path / "c"))
    R.rematerialize(argparse.Namespace(
        input=root, config=CONFIG, resume=False, wav_encoding="auto", jobs=1))
    _gate(root)


def test_an_interrupted_rematerialization_is_refused(tmp_path):
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, shipped_contract().fingerprint(), {0})     # 1 and 2 never finished
    with pytest.raises(ValueError, match="absent from"):
        _gate(root)


def test_a_ledger_from_another_contract_is_refused(tmp_path):
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, "b" * 64, set(range(N_SEQ)))
    with pytest.raises(ValueError, match="DIFFERENT linear-AEC contract"):
        _gate(root)


def test_a_ledger_claiming_absent_sequences_is_refused(tmp_path):
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, shipped_contract().fingerprint(), set(range(N_SEQ + 2)))
    with pytest.raises(ValueError, match="not in the corpus"):
        _gate(root)


@pytest.mark.parametrize("commit,source_hash", MIGRATED_FROM_REVISIONS)
def test_a_legacy_ledger_from_the_retired_corpus_frontend_is_refused(
        tmp_path, commit, source_hash):
    """The remote flow after the frontend moved: the corpus's ledger still
    names a build whose `linear_error` this build does not reproduce. There
    is no bridge for a retired identity -- a legacy ledger records only a
    fingerprint, so the refusal is the generic contract mismatch, and it must
    not carry the frontend-equivalent warning. Run for every revision that
    carried the retired behaviour, because each wrote a different fingerprint.
    """
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, _migrated_source_fingerprint(commit, source_hash),
                set(range(N_SEQ)))
    with pytest.raises(ValueError, match="linear-AEC contract|frontend identity"):
        _gate(root)


def test_a_ledger_from_an_uncovered_revision_is_refused(tmp_path):
    """A legacy ledger from a revision no table records is refused on the
    generic contract mismatch."""
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs,
                _migrated_source_fingerprint(UNCOVERED_COMMIT,
                                             MIGRATED_FROM_SOURCE_HASH),
                set(range(N_SEQ)))
    with pytest.raises(ValueError, match="DIFFERENT linear-AEC contract"):
        _gate(root)


def test_the_ledger_bridge_is_not_consulted_backwards(tmp_path):
    """A build running the OLD frontend gets no candidates for a ledger written
    by the new one. The migration is one-way, and the reconstruction reads the
    table in one direction only."""
    old_build = _source_contract(MIGRATED_FROM_COMMIT,
                                 MIGRATED_FROM_SOURCE_HASH)
    assert migrated_ledger_fingerprints(old_build) == ()


def test_a_refused_legacy_ledger_says_how_to_identify_the_corpus(tmp_path):
    """The operator is not left comparing two opaque hashes.

    A legacy ledger records only a fingerprint, which is one-way -- so when the
    guard refuses, it cannot name the build that wrote the corpus. It has to
    say so, name what THIS build is, and give the operator something to run on
    the machine that produced the corpus.
    """
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, "b" * 64, set(range(N_SEQ)))
    with pytest.raises(ValueError) as caught:
        _gate(root)
    message = str(caught.value)
    assert "records no frontend identity" in message
    assert "git -C lib/aec rev-parse HEAD" in message
    assert shipped_contract().aec_behavior_hash[:12] in message
    assert str(shipped_contract().sample_rate) in message


# ── ledgers that record the frontend that wrote them ────────────────────────
#
# The route above reconstructs candidate fingerprints because a legacy ledger
# says nothing else. A ledger that records the producing contract needs none of
# that: the comparison is exact, config-independent, and can report which field
# disagrees.

def _identity_ledger(root, contract, sequences):
    """A ledger keyed and identified by ``contract``."""
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, contract.fingerprint(), sequences, contract.as_dict())
    return seqs


def test_rematerialize_records_the_producing_frontend(tmp_path):
    """The real writer, not a hand-built ledger: the corpus tool has to record
    the identity or nothing downstream can read one."""
    root = _corpus(str(tmp_path / "c"))
    R.rematerialize(argparse.Namespace(
        input=root, config=CONFIG, resume=False, wav_encoding="auto", jobs=1))
    with open(os.path.join(root, LEDGER_NAME), encoding="utf-8") as fh:
        recorded = json.load(fh)
    assert recorded["linear_aec"] == shipped_contract().as_dict()
    assert recorded["contract"] == shipped_contract().fingerprint()


def test_a_legacy_shaped_ledger_is_still_honoured(tmp_path):
    """⚠ The in-flight 200-hour run is writing this shape right now.

    A ledger with only `contract` and `sequences` -- no identity -- must keep
    passing every reader unchanged. Adding the identity field may not
    invalidate a corpus that is mid-rematerialization.
    """
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, shipped_contract().fingerprint(), set(range(N_SEQ)))
    with open(os.path.join(root, LEDGER_NAME), encoding="utf-8") as fh:
        assert "linear_aec" not in json.load(fh)
    _gate(root)


def test_an_identity_ledger_from_a_comment_only_rebuild_is_accepted(tmp_path):
    """The general case the reconstruction cannot cover: a lib/aec commit that
    moved nothing but comments. Behaviour is unchanged, so the data is the
    same, but the fingerprint moved and the corpus would otherwise be stranded
    -- and no table has to be extended for it."""
    root = _corpus(str(tmp_path / "c"))
    reflowed = dataclasses.replace(
        shipped_contract(), aec_commit=UNCOVERED_COMMIT,
        aec_source_hash="c" * 64)
    _identity_ledger(root, reflowed, set(range(N_SEQ)))
    with pytest.warns(RuntimeWarning, match="the ledger is honoured") as caught:
        _gate(root)
    assert UNCOVERED_COMMIT[:12] in "\n".join(str(w.message) for w in caught)


def test_an_identity_ledger_from_the_retired_corpus_frontend_is_refused(tmp_path):
    """The identity route sees the behaviour hash the ledger states, so the
    corpus frontend is refused by name with the rematerialize instruction --
    not accepted through a bridge that no longer exists."""
    root = _corpus(str(tmp_path / "c"))
    _identity_ledger(root, _source_contract(UNCOVERED_COMMIT, "c" * 64),
                     set(range(N_SEQ)))
    with pytest.raises(ValueError) as caught:
        _gate(root)
    message = str(caught.value)
    assert MIGRATED_FROM_HASH[:12] in message
    assert "different linear_error" in message
    assert "without --resume" in message


def test_an_identity_ledger_from_an_unrelated_frontend_is_refused(tmp_path):
    """Exact, not lenient. The refusal names the field that disagrees and both
    frontends, which is the whole reason to record the identity."""
    root = _corpus(str(tmp_path / "c"))
    other = shipped_contract()
    other = dataclasses.replace(
        other, filter_length=other.filter_length + other.hop_size,
        aec_commit=UNCOVERED_COMMIT)
    _identity_ledger(root, other, set(range(N_SEQ)))
    with pytest.raises(ValueError) as caught:
        _gate(root)
    message = str(caught.value)
    assert "filter_length" in message
    assert UNCOVERED_COMMIT[:12] in message
    assert shipped_contract().aec_behavior_hash[:12] in message


def test_an_identity_ledger_from_a_retired_frontend_is_refused(tmp_path):
    """A retired identity says rematerialize, through the ledger gate too --
    not a bare fingerprint mismatch."""
    root = _corpus(str(tmp_path / "c"))
    retired = sorted(RETIRED_BEHAVIOR_HASHES)[0]
    _identity_ledger(
        root, _source_contract(UNCOVERED_COMMIT, "c" * 64, behavior=retired),
        set(range(N_SEQ)))
    with pytest.raises(ValueError) as caught:
        _gate(root)
    message = str(caught.value)
    # Wording only the retired branch produces, so this cannot pass on the
    # generic "different contract" refusal.
    assert "different linear_error" in message
    assert "without --resume" in message


def test_an_identity_ledger_that_contradicts_its_own_fingerprint_is_refused(tmp_path):
    """Both fields describe the same run, so they cannot disagree. One that
    does has been edited or spliced from two runs, and neither half can be
    trusted to say what wrote the corpus."""
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, "b" * 64, set(range(N_SEQ)),
                shipped_contract().as_dict())
    with pytest.raises(ValueError, match="internally inconsistent"):
        _gate(root)


def test_an_identity_ledger_that_is_not_a_contract_is_refused(tmp_path):
    """Junk in the identity field is refused, not silently ignored down to the
    legacy path -- a ledger that carries an unreadable identity cannot say what
    wrote the corpus any more than a truncated one can."""
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, "b" * 64, set(range(N_SEQ)), {"engine": "nonsense"})
    with pytest.raises(ValueError, match="does not read as one"):
        _gate(root)


def test_an_unreadable_ledger_is_refused(tmp_path):
    """Distinct from "no ledger". A ledger that exists but cannot be parsed
    cannot say the corpus is complete, and treating that as "no claim" would
    let a truncated one through."""
    root = _corpus(str(tmp_path / "c"))
    with open(os.path.join(root, LEDGER_NAME), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    with pytest.raises(ValueError, match="does not read as a ledger"):
        _gate(root)


def test_pack_itself_runs_the_guard(tmp_path):
    """Through the real entry point, so the guard being WIRED IN is tested.

    Every other case here calls the guard directly and would pass with the
    call site deleted, misspelled, or placed after the expensive work.
    """
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    save_ledger(seqs, shipped_contract().fingerprint(), {0})   # 1 and 2 unfinished

    with pytest.raises(ValueError, match="absent from"):
        P.pack(argparse.Namespace(
            input=root, config=CONFIG, output=str(tmp_path / "shards"),
            shard_clips=512, overwrite=False, dtype="float32"))

    # And it must refuse BEFORE doing the expensive work, not after.
    assert not os.path.exists(os.path.join(str(tmp_path / "shards"), "shard_00000.pt"))


def test_rematerialize_then_pack_produces_loadable_shards(tmp_path):
    """The whole path a real run takes, in order, through both entry points.

    Two things this closes that nothing else did. First, the negative case
    above has to be able to pass, or it proves nothing. Second, "pack did not
    raise" is not "pack produced something usable" -- the shards are opened
    and checked, because a corpus that packs into unusable shards fails at
    training time, hours later, with nothing pointing back here.

    Runs the rematerializer at --jobs 2 deliberately: the parallel path is the
    one a real run uses, and it is the one that writes the ledger the packer
    then reads.
    """
    from AIAEC.dataset_gen.pack_aec_dataset import PACKED_STEM_ORDER

    root = _corpus(str(tmp_path / "c"))
    shards = str(tmp_path / "shards")
    R.rematerialize(argparse.Namespace(
        input=root, config=CONFIG, resume=False, wav_encoding="auto", jobs=2))
    P.pack(argparse.Namespace(
        input=root, config=CONFIG, output=shards,
        shard_clips=512, overwrite=False, dtype="float32"))

    written = sorted(glob.glob(os.path.join(shards, "shard_*.pt")))
    assert written, "pack produced no shards"
    assert not glob.glob(os.path.join(shards, "*.tmp")), "a temporary shard survived"

    contract = shipped_contract().fingerprint()
    seen = 0
    for path in written:
        shard = torch.load(path, weights_only=False)
        assert shard["stems"] == list(PACKED_STEM_ORDER)
        # The header must name the contract that actually rebuilt the fifth
        # channel -- the whole point of the guard that let this pack proceed.
        assert shard["linear_aec_contract_hash"] == contract
        assert shard["data"].shape[1] == len(PACKED_STEM_ORDER)
        assert torch.isfinite(shard["data"]).all()
        seen += shard["data"].shape[0]
    assert seen == N_SEQ * N_CHUNK, f"packed {seen} clips, corpus has {N_SEQ * N_CHUNK}"


def test_a_corpus_from_the_retired_frontend_does_not_pack(tmp_path):
    """The remote flow end to end after the frontend moved.

    A corpus is rematerialized, lib/aec then moves to a revision that renders
    a different `linear_error` (simulated by re-keying the ledger to the
    retired corpus build, which is exactly what that machine's ledger holds),
    and the operator packs. The first gate has to refuse: shards stamped with
    THIS build's contract over that audio would train a checkpoint on a stem
    this build does not produce.
    """
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    shards = str(tmp_path / "shards")
    R.rematerialize(argparse.Namespace(
        input=root, config=CONFIG, resume=False, wav_encoding="auto", jobs=1))
    save_ledger(seqs, _migrated_source_fingerprint(), set(range(N_SEQ)))

    with pytest.raises(ValueError, match="linear-AEC contract|frontend identity"):
        P.pack(argparse.Namespace(
            input=root, config=CONFIG, output=shards,
            shard_clips=512, overwrite=False, dtype="float32"))
    assert not os.path.exists(shards) or not os.listdir(shards)
