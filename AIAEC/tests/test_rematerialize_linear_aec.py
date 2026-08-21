"""--jobs must not change the corpus, and --resume must not mix contracts.

Both properties are structural -- a fresh PBFDKF per sequence, no shared
mutable state, no random source -- but structural arguments are what these
tests exist to stop anyone from having to trust.

⚠ Compare AUDIO, never file bytes. libsndfile stamps a PEAK chunk with the
wall-clock time when it writes a float WAV, so two runs of ANY configuration
seconds apart differ in one byte at offset 61. Byte-comparing the files would
make this test fail for a reason that has nothing to do with what it checks.
"""
import argparse
import json
import os

import pytest
import torch
import torchaudio

from AIAEC.dataset_gen import rematerialize_linear_aec as R

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "dataset_gen", "config.ini")
CHUNK = 256 * 32     # 0.5 s at 16 kHz; must be a multiple of the PBFDKF hop
N_SEQ, N_CHUNK = 3, 3


def _corpus(root):
    seqs = os.path.join(root, "seqs")
    os.makedirs(seqs, exist_ok=True)
    torch.manual_seed(0)
    for sequence in range(N_SEQ):
        for index in range(N_CHUNK):
            far = torch.randn(CHUNK) * 0.1
            mic = torch.roll(far, 800) * 0.5 + torch.randn(CHUNK) * 0.02
            zero = torch.zeros(CHUNK)
            torchaudio.save(
                os.path.join(seqs, f"{sequence:06d}_{index:03d}.wav"),
                torch.stack([far, zero, zero, mic, zero]), 16000,
                encoding="PCM_F", bits_per_sample=32)
    return root


def _read_config():
    import configparser
    parser = configparser.ConfigParser()
    parser.read(CONFIG)
    return parser


def _run(root, jobs=1, resume=False):
    R.rematerialize(argparse.Namespace(
        input=root, config=CONFIG, resume=resume, wav_encoding="auto",
        jobs=jobs))


def _audio(root):
    seqs = os.path.join(root, "seqs")
    return {f: torchaudio.load(os.path.join(seqs, f))[0]
            for f in sorted(os.listdir(seqs)) if f.endswith(".wav")}


def test_jobs_does_not_change_a_single_sample(tmp_path):
    serial = _corpus(str(tmp_path / "serial"))
    parallel = _corpus(str(tmp_path / "parallel"))
    _run(serial, jobs=1)
    _run(parallel, jobs=3)

    a, b = _audio(serial), _audio(parallel)
    assert set(a) == set(b) and len(a) == N_SEQ * N_CHUNK
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs between jobs=1 and jobs=3"


def test_the_fifth_channel_is_actually_rewritten(tmp_path):
    """Guards the comparison above: two runs that both wrote nothing would
    also agree."""
    root = _corpus(str(tmp_path / "c"))
    before = _audio(root)
    _run(root, jobs=1)
    after = _audio(root)
    for name in before:
        assert after[name].shape[0] == 5
        assert torch.equal(before[name][:4], after[name][:4]), "acoustic stems moved"
        assert after[name][4].abs().max() > 0, "linear_error still zero"
        assert torch.isfinite(after[name][4]).all()


def test_resume_skips_only_what_this_contract_wrote(tmp_path):
    root = _corpus(str(tmp_path / "c"))
    _run(root, jobs=1)
    ledger = R._ledger_path(os.path.join(root, "seqs"))
    assert os.path.exists(ledger)
    with open(ledger, encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert sorted(recorded["sequences"]) == list(range(N_SEQ))

    after_first = _audio(root)
    _run(root, jobs=1, resume=True)          # everything already done
    assert all(torch.equal(after_first[k], v) for k, v in _audio(root).items())


def test_a_ledger_from_another_contract_is_ignored_whole(tmp_path):
    """The failure this guards is a corpus carrying two frontends. Trusting
    part of a foreign ledger is exactly how that happens."""
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    _run(root, jobs=1)
    with open(R._ledger_path(seqs), encoding="utf-8") as handle:
        real = json.load(handle)

    R._save_ledger(seqs, "a" * 64, set(range(N_SEQ)))
    assert R._load_ledger(seqs, real["contract"]) == set()
    assert R._load_ledger(seqs, "a" * 64) == set(range(N_SEQ))


def test_an_unfinished_sequence_is_never_recorded(tmp_path):
    """A killed run must redo a sequence, not skip it. The ledger is written
    only after _rewrite_sequence returns, so a sequence that raised is absent."""
    root = _corpus(str(tmp_path / "c"))
    seqs = os.path.join(root, "seqs")
    # A MIDDLE chunk: dropping the last one just makes a shorter sequence,
    # which is not a gap and must not be treated as one.
    os.remove(os.path.join(seqs, f"{1:06d}_{1:03d}.wav"))
    with pytest.raises(FileNotFoundError):
        _run(root, jobs=1)

    # Read with the contract that actually wrote the ledger. Asking for any
    # other hash returns an empty set by design, which would make this pass
    # whether or not the sequence was wrongly recorded.
    contract = R.linear_aec_contract_from_config(
        _read_config()).fingerprint()
    recorded = R._load_ledger(seqs, contract)
    assert 1 not in recorded, "an unfinished sequence was recorded as done"
    # ...and the ledger is not simply empty, which would also satisfy the line
    # above without proving anything: sequence 0 completed before 1 raised.
    assert 0 in recorded
