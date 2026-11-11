"""Both supported 16 kHz grids must survive the real corpus/training path.

The acoustic stems are identical between grids, but their materialized linear
errors and contracts are separate artifacts. No user corpus or AEC mock is
used: the shared tiny corpus runs through the actual rematerializer and packer.
"""

import argparse
import configparser
import json
import shutil

import pytest
import torch
import torchaudio

import AIAEC.training_common as training_common
from AIAEC.dataset_gen import MODEL_TASKS
from AIAEC.dataset_gen import pack_aec_dataset as P
from AIAEC.dataset_gen import rematerialize_linear_aec as R
from AIAEC.dataset_gen.aec_features import PACKED_STEM_ORDER
from AIAEC.dataset_gen.linear_aec import linear_aec_contract_from_config
from AIAEC.dataset_gen.linear_error_ledger import ledger_path, load_ledger
from AIAEC.dataset_gen.packed_aec_dataset import PackedAecDataset
from AIAEC.training_common import (
    build_plain_loaders,
    make_checkpoint_contract,
    read_grids,
    require_checkpoint_contract,
    require_checkpoint_linear_aec,
)

from ledger_corpus import CHUNK, build_corpus


GRIDS = ((512, 256), (256, 128))


def _write_config(path, frame, hop):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "signal": {
            "sr": "16000", "n_fft": str(frame),
            "win_len": str(frame), "hop_len": str(hop),
        },
        "linear_aec": {"preset": "balanced", "frame_size": str(frame)},
        "data": {
            "batch_size": "1", "num_workers": "0", "val_fraction": "0.5",
        },
    })
    with path.open("w", encoding="utf-8") as handle:
        cfg.write(handle)
    return cfg


def _rematerialize(root, config, *, resume=False, jobs=1):
    R.rematerialize(argparse.Namespace(
        input=str(root), config=str(config), resume=resume,
        wav_encoding="auto", jobs=jobs,
    ))


def _pack(root, config, output):
    P.pack(argparse.Namespace(
        input=str(root), config=str(config), output=str(output),
        shard_clips=8, overwrite=False, dtype="float32",
    ))


@pytest.fixture(scope="module")
def prepared_grids(tmp_path_factory):
    """Two chunks give the real loader a non-empty train/validation split."""
    workspace = tmp_path_factory.mktemp("linear_aec_grids")
    prepared = {}
    for frame, hop in GRIDS:
        directory = workspace / f"{frame}_{hop}"
        directory.mkdir()
        config = directory / "config.ini"
        cfg = _write_config(config, frame, hop)
        root = directory / "corpus"
        build_corpus(str(root), n_seq=1, n_chunk=2)
        before = {
            path.name: torchaudio.load(str(path))[0]
            for path in sorted((root / "seqs").glob("*.wav"))
        }
        _rematerialize(root, config)
        shards = directory / "packed"
        _pack(root, config, shards)
        prepared[frame] = {
            "root": root, "config": config, "cfg": cfg, "shards": shards,
            "linear": linear_aec_contract_from_config(cfg), "before": before,
        }
    return prepared


@pytest.mark.parametrize("frame,hop", GRIDS)
def test_both_grids_reach_loaders_and_checkpoint_contracts(
        prepared_grids, frame, hop):
    artifact = prepared_grids[frame]
    linear = artifact["linear"]
    assert (linear.sample_rate, linear.frame_size, linear.hop_size) == (
        16000, frame, hop,
    )
    seqs = artifact["root"] / "seqs"
    with open(ledger_path(str(seqs)), encoding="utf-8") as handle:
        ledger = json.load(handle)
    assert ledger["linear_aec"] == linear.as_dict()
    assert ledger["contract"] == linear.fingerprint()
    assert load_ledger(str(seqs), linear.fingerprint()) == {0}

    dataset = PackedAecDataset(str(artifact["shards"]), expected_sr=16000,
                               verbose=False)
    assert len(dataset) == 2
    assert dataset.stems == PACKED_STEM_ORDER
    assert dataset.linear_aec_contract == linear
    assert dataset.linear_aec_contract_hash == linear.fingerprint()
    for index, path in enumerate(sorted(seqs.glob("*.wav"))):
        audio, sr = torchaudio.load(str(path))
        assert sr == 16000 and audio.shape == (5, CHUNK)
        assert torch.equal(audio[:4], artifact["before"][path.name][:4])
        assert torch.isfinite(audio[4]).all() and audio[4].abs().max() > 0
        stems, meta = dataset[index]
        # Pin the WAV -> packed channel mapping independently of the packer.
        assert torch.equal(stems, audio[[0, 3, 4, 2]])
        assert meta == {"sequence_id": 0, "chunk_index": index}

    aec_grid, model_grid = read_grids(artifact["cfg"])
    train, val, data = build_plain_loaders(
        artifact["cfg"], aec_grid, seed=7,
        packed_dir=str(artifact["shards"]),
    )
    assert data["linear_aec"] == linear.as_dict()
    assert data["linear_aec_contract_hash"] == linear.fingerprint()
    assert data["dataset_fingerprint"] == dataset.fingerprint()
    assert sorted(data["train_indices"] + data["val_indices"]) == [0, 1]
    assert set(data["train_indices"]).isdisjoint(data["val_indices"])
    assert val is not None
    for loader in (train, val):
        stems, meta = next(iter(loader))
        assert stems.shape == (1, 4, CHUNK)
        assert torch.isfinite(stems).all()
        assert len(meta) == 1 and meta[0]["sequence_id"] == 0

    checkpoint = make_checkpoint_contract(
        model_name="Align_ULCNet", task=MODEL_TASKS["Align_ULCNet"],
        grid=model_grid, model_kwargs={}, loss_version="grid_pipeline_test",
        data_contract=data,
    )
    assert (checkpoint["sr"], checkpoint["n_fft"], checkpoint["hop_len"]) == (
        16000, frame, hop,
    )
    require_checkpoint_contract({"contract": checkpoint}, checkpoint)
    assert require_checkpoint_linear_aec(checkpoint, model_grid) == linear.as_dict()

    other_frame = 256 if frame == 512 else 512
    _, other_grid = read_grids(prepared_grids[other_frame]["cfg"])
    with pytest.raises(ValueError, match="model/PBFDKF grid mismatch"):
        require_checkpoint_linear_aec(checkpoint, other_grid)
    with pytest.raises(ValueError, match=r"contract\.n_fft"):
        require_checkpoint_contract(
            {"contract": checkpoint},
            {**checkpoint, "n_fft": other_grid.n_fft,
             "win_len": other_grid.win_len, "hop_len": other_grid.hop_len},
        )


@pytest.mark.parametrize("source_frame,target_frame", [(512, 256), (256, 512)])
def test_resume_with_another_grid_rewrites_instead_of_skipping(
        prepared_grids, tmp_path, monkeypatch, source_frame, target_frame):
    source = prepared_grids[source_frame]
    target = prepared_grids[target_frame]
    root = tmp_path / "resumed"
    shutil.copytree(source["root"], root)
    assert source["linear"].fingerprint() != target["linear"].fingerprint()
    assert load_ledger(str(root / "seqs"), target["linear"].fingerprint()) == set()

    rewritten = []
    real_rewrite = R._rewrite_sequence

    def record_rewrite(job):
        rewritten.append(job[1])
        return real_rewrite(job)

    monkeypatch.setattr(R, "_rewrite_sequence", record_rewrite)
    _rematerialize(root, target["config"], resume=True)
    assert rewritten == [0], "a ledger for another grid must not skip its sequence"
    with open(ledger_path(str(root / "seqs")), encoding="utf-8") as handle:
        ledger = json.load(handle)
    assert ledger["contract"] == target["linear"].fingerprint()
    assert ledger["linear_aec"] == target["linear"].as_dict()
    for path in sorted((root / "seqs").glob("*.wav")):
        actual, _ = torchaudio.load(str(path))
        expected, _ = torchaudio.load(str(target["root"] / "seqs" / path.name))
        assert torch.equal(actual, expected)


def test_small_grid_spawn_workers_match_serial(prepared_grids, tmp_path):
    artifact = prepared_grids[256]
    root = tmp_path / "parallel"
    shutil.copytree(artifact["root"], root)
    # A full pass rewrites even an already-materialized copy. Spawn workers
    # must accept the 256/128 table entry and its serialized contract too.
    _rematerialize(root, artifact["config"], jobs=2)
    assert load_ledger(str(root / "seqs"), artifact["linear"].fingerprint()) == {0}
    for path in sorted((root / "seqs").glob("*.wav")):
        parallel, parallel_sr = torchaudio.load(str(path))
        serial, serial_sr = torchaudio.load(
            str(artifact["root"] / "seqs" / path.name))
        assert parallel_sr == serial_sr == 16000
        assert torch.equal(parallel, serial), path.name


@pytest.mark.parametrize("source_frame,target_frame", [(512, 256), (256, 512)])
def test_pack_refuses_config_from_another_grid(
        prepared_grids, tmp_path, source_frame, target_frame):
    output = tmp_path / "wrong_grid"
    with pytest.raises(ValueError, match="frame_size|hop_size|frame/hop"):
        _pack(prepared_grids[source_frame]["root"],
              prepared_grids[target_frame]["config"], output)
    assert not list(output.glob("shard_*.pt"))
    assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize("source_frame,target_frame", [(512, 256), (256, 512)])
def test_training_loader_refuses_other_grid(
        prepared_grids, monkeypatch, source_frame, target_frame):
    source = prepared_grids[source_frame]
    target = prepared_grids[target_frame]
    aec_grid, _model_grid = read_grids(target["cfg"])

    def refuse_loader_construction(*args, **kwargs):
        pytest.fail("a mismatched corpus reached DataLoader construction")

    monkeypatch.setattr(training_common, "DataLoader", refuse_loader_construction)
    with pytest.raises(ValueError,
                       match="training dataset model/PBFDKF grid mismatch"):
        build_plain_loaders(
            target["cfg"], aec_grid, packed_dir=str(source["shards"]),
        )


def test_packed_dataset_refuses_shards_from_both_grids(prepared_grids, tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    for index, (frame, _hop) in enumerate(GRIDS):
        shards = sorted(prepared_grids[frame]["shards"].glob("shard_*.pt"))
        assert len(shards) == 1
        shutil.copyfile(shards[0], mixed / f"shard_{index:05d}.pt")
    with pytest.raises(ValueError, match="linear_aec contract differs"):
        PackedAecDataset(str(mixed), expected_sr=16000, verbose=False)
