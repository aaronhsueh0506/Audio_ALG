"""The synthetic corpus the ledger/rematerialization tests share.

A module of its own, not ``conftest.py``: pytest gives every ``conftest.py``
the same top-level module name, so collecting AIAEC together with a sibling
package that has one (``AINR/conftest.py``) makes ``from conftest import ...``
resolve to whichever was imported first and the run dies at collection; a
uniquely named module cannot be shadowed
(``tests/test_cross_package_collection.py`` holds the gate). Both suites need
the SAME corpus shape, so there is one definition. It does not reuse
dataset_gen/tests/test_aec_dataset.py's module-scoped `corpus` fixture: these
tests need per-test mutable roots (they delete a chunk mid-sequence, truncate a
channel and build two corpora side by side).
"""
import configparser
import os

import torch
import torchaudio

from AIAEC.dataset_gen.linear_aec import linear_aec_contract_from_config

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "dataset_gen", "config.ini")
CHUNK = 256 * 32          # 0.5 s at 16 kHz; must be a multiple of the PBFDKF hop


def shipped_contract():
    """The contract the shipped config builds -- the one a real run would use."""
    parser = configparser.ConfigParser()
    parser.read(CONFIG)
    return linear_aec_contract_from_config(parser)


def build_corpus(root, n_seq, n_chunk):
    """A deterministic corpus in the SSSSSS_CCC.wav layout.

    Channel 5 starts at exact zero so "was it rewritten" is decidable, and the
    echo is a plain delayed copy so the PBFDKF has something real to converge
    on without needing a rendered room.
    """
    seqs = os.path.join(root, "seqs")
    os.makedirs(seqs, exist_ok=True)
    torch.manual_seed(0)
    for sequence in range(n_seq):
        for index in range(n_chunk):
            far = torch.randn(CHUNK) * 0.1
            mic = torch.roll(far, 800) * 0.5 + torch.randn(CHUNK) * 0.02
            zero = torch.zeros(CHUNK)
            torchaudio.save(
                os.path.join(seqs, f"{sequence:06d}_{index:03d}.wav"),
                torch.stack([far, zero, zero, mic, zero]), 16000,
                encoding="PCM_F", bits_per_sample=32)
    return root
