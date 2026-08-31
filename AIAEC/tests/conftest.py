"""Fixture wrappers over the shared ledger corpus.

The corpus itself is in ``ledger_corpus.py``: the two ledger suites import
those names directly, and a ``conftest.py`` is not safely importable by name --
see that module's docstring. Only the fixtures live here.
"""
import pytest

from ledger_corpus import CONFIG, shipped_contract, build_corpus


@pytest.fixture(scope="session")
def config_path():
    return CONFIG


@pytest.fixture(scope="session")
def contract():
    return shipped_contract()


@pytest.fixture
def make_corpus():
    """Factory, not a corpus: these tests build two side by side to compare,
    and mutate them independently."""
    return build_corpus
