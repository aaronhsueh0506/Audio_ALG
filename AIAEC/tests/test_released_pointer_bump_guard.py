"""Staleness guard for RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY.

Opus review item C2 (2026-10-20): an agent bumping that dict's single entry
ASSUMED the pointer's prior value was the hash immediately before their own
change, bumped it to their new hash, and in doing so silently re-admitted
roughly a month and 21 unrelated lib/aec commits of un-vetted drift that had
accumulated on the pointer in between. The pre-existing test
(``test_released_checkpoint_exception_is_separate_and_targets_this_build`` in
``test_linear_aec_behavior_migration.py``) only asserts the pointer equals
the LIVE hash right now; it cannot notice that the gap between two bumps was
never individually vetted.

This module tests ``released_pointer_bump_guard.py``'s two layers (see that
module's docstring for the full design), and -- the point of this file --
proves each layer can actually fail:

  * ``test_chain_checker_rejects_*`` / ``test_head_check_rejects_*`` feed the
    pure checker functions the EXACT bad input the opus bug produced (a row
    naming the wrong "old" value, or no row at all) and assert it is
    rejected, before asserting the correct form passes.
  * ``test_bump_table_chains_to_the_live_pointer_value`` and
    ``test_live_worktree_bump_is_documented_against_head`` run the real
    checkers against this repo's real ``linear_aec.py`` (the latter via
    ``git show HEAD:...``), so a future bump that skips updating
    ``RELEASED_INFERENCE_POINTER_BUMPS`` fails `make test` / this file here,
    not just a convention someone has to remember.

Run:
    python3 -m pytest AIAEC/tests/test_released_pointer_bump_guard.py
"""
from __future__ import annotations

from pathlib import Path

from AIAEC.dataset_gen import linear_aec
from AIAEC.dataset_gen.released_pointer_bump_guard import (
    check_bump_against_head,
    check_chain,
    check_file_against_head,
)

# AIAEC/dataset_gen/released_pointer_bump_guard.py -> dataset_gen -> AIAEC ->
# Audio_ALG (the git root this guard's `git show HEAD:...` runs against).
_AUDIO_ALG_ROOT = Path(linear_aec.__file__).resolve().parents[2]
_REL_PATH = "AIAEC/dataset_gen/linear_aec.py"

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
# A real-looking but WRONG value: close to the real pre-bump value in
# spelling (it is the prefix/suffix the opus-bug comment actually names
# elsewhere in linear_aec.py: "428aa9b7...562d") but not literally the hash
# HEAD has. 64 hex chars, deliberately not any real hash here.
_PLAUSIBLE_WRONG = "428aa9b796c2" + ("0" * (64 - len("428aa9b796c2") - 4)) + "562d"
assert len(_PLAUSIBLE_WRONG) == 64


def _live_pointer_value():
    compat = linear_aec.RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY
    assert len(compat) == 1, (
        "this guard and these tests assume exactly one tracked checkpoint "
        "pointer; a second entry needs this file revisited")
    return next(iter(compat.values()))


# --- Layer 1: check_chain -- stateless, survives any future commit ---------

def test_chain_checker_rejects_a_row_whose_old_does_not_match_the_priors_new():
    """Can-fail proof. Reproduces the opus bug shape: row 1's `old` is a
    plausible-looking value that is NOT row 0's `new` (the author assumed/
    recomputed it instead of reading it off the chain)."""
    bad = (
        ("2026-09-22", _A, _B),
        ("2026-10-20", _C, _D),  # WRONG: should chain from _B, not _C
    )
    violations = check_chain(bad, _D)
    assert any("does not equal row 0's `new`" in v for v in violations), violations


def test_chain_checker_rejects_an_abbreviated_hash():
    """Can-fail proof. The real comment in linear_aec.py abbreviates hashes
    as 'prefix...suffix' -- the chain table must reject that form, since an
    elided value cannot be compared literally against git's."""
    bad = (("2026-10-20", "5fe64e4648...3f4", "419f473e...ff8fa"),)
    violations = check_chain(bad, "419f473e...ff8fa")
    assert any("not a literal 64-hex-char hash" in v for v in violations), violations


def test_chain_checker_rejects_a_table_stale_against_the_live_dict():
    """Can-fail proof. The table's last `new` must track the dict's current
    value; a table nobody updated after a later, undocumented bump is caught
    here even if every row it does have chains correctly."""
    stale = (("2026-09-22", _A, _B),)
    violations = check_chain(stale, _D)  # live value moved on to _D undocumented
    assert any("out of date with the dict entry" in v for v in violations), violations


def test_chain_checker_accepts_a_correctly_chained_table():
    good = (
        ("2026-09-22", _A, _B),
        ("2026-10-20", _B, _D),
    )
    assert check_chain(good, _D) == []


def test_bump_table_chains_to_the_live_pointer_value():
    """The real table in linear_aec.py, checked against the real live dict
    value. This is what makes the guard run forever, not just at bump time:
    any future bump that forgets to append a row, or appends one whose `old`
    is not literally the previous row's `new`, fails here."""
    violations = check_chain(
        linear_aec.RELEASED_INFERENCE_POINTER_BUMPS, _live_pointer_value())
    assert violations == [], violations


# --- Layer 2: check_bump_against_head -- the bump-time catch ---------------

def test_head_check_rejects_a_bump_with_the_wrong_assumed_old_value():
    """Can-fail proof. THE exact failure mode from the opus review: the
    pointer is bumped, and a row IS added, but that row's `old` is not what
    HEAD actually had -- it is a value the author assumed or independently
    recomputed, not one copied from `git show HEAD:...`."""
    head_value = "5fe64e46480c7e2a75c90718f1b64c09ca29f611d09713ea4ec2a9ef61c813f4"
    work_value = "419f473e609ec0c3fd645a9dcb7ccd53a742fb8c22d17a45578e6cc6df2ff8fa"
    bumps = (("2026-10-20", _PLAUSIBLE_WRONG, work_value),)
    violations = check_bump_against_head(head_value, work_value, bumps)
    assert violations != [], "a wrong `old` value must be rejected"
    assert head_value in violations[0] and work_value in violations[0]


def test_head_check_rejects_a_bump_with_no_row_at_all():
    """Can-fail proof. The value moved, and the table has no row for it --
    the original bug before any table existed."""
    violations = check_bump_against_head(_A, _B, ())
    assert violations != []


def test_head_check_accepts_a_correctly_documented_bump():
    assert check_bump_against_head(_A, _B, (("2026-10-20", _A, _B),)) == []


def test_head_check_is_silent_when_the_pointer_did_not_move():
    """Pre-commit-style gate: nothing to vet when there is no pending bump."""
    assert check_bump_against_head(_A, _A, ()) == []
    assert check_bump_against_head(None, _A, ()) == []


# --- Layer 3: wired to the real repo, the live pre-commit-style gate -------

def test_live_worktree_bump_is_documented_against_head():
    """Runs both layers against this repo's actual linear_aec.py: the
    working tree's own chain, and, if its pointer value differs from
    `git show HEAD:...`, that the difference is documented with HEAD's real
    value. This is the test that fails `make test` if a future bump forgets
    to add the row described above the table in linear_aec.py."""
    violations = check_file_against_head(_AUDIO_ALG_ROOT, _REL_PATH)
    assert violations == [], violations
