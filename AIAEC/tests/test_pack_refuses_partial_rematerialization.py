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
import configparser
import glob
import os

import pytest
import torch
import torchaudio

from AIAEC.dataset_gen import pack_aec_dataset as P
from AIAEC.dataset_gen import rematerialize_linear_aec as R
from AIAEC.dataset_gen.linear_aec import linear_aec_contract_from_config
from AIAEC.dataset_gen.linear_error_ledger import LEDGER_NAME, save_ledger

from conftest import CONFIG, build_corpus, shipped_contract

N_SEQ, N_CHUNK = 3, 2


def _corpus(root):
    return build_corpus(root, N_SEQ, N_CHUNK)


def _gate(root):
    seqs = os.path.join(root, "seqs")
    P._require_complete_rematerialization(
        seqs, P._collect(seqs), shipped_contract().fingerprint())


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
