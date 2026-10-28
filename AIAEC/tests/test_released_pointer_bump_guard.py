"""Staleness guard for RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY.

A bump of that dict's single entry must not assume the pointer's prior value
was the hash immediately before the author's own change, or unvetted drift
accumulated on the pointer is silently re-admitted. The dict is currently
EMPTY (the released checkpoint is retired, see
``test_released_checkpoint_pointer_is_retired_not_repointed`` in
``test_linear_aec_behavior_migration.py``); the guard then checks that the bump
table is retired with it and that no pointer is (re)introduced against HEAD.

This module tests the two layers of ``released_pointer_bump_guard.py`` and
proves each can fail:

  * ``test_chain_checker_rejects_*`` / ``test_head_check_rejects_*`` feed the
    pure checkers the EXACT bad input (a row naming the wrong "old" value, or no
    row at all) and assert it is rejected before the correct form passes.
  * ``test_bump_table_chains_to_the_live_pointer_value`` and
    ``test_live_worktree_bump_is_documented_against_head`` run the real checkers
    against this repo's ``linear_aec.py`` (the latter via ``git show HEAD:...``).

Run:
    python3 -m pytest AIAEC/tests/test_released_pointer_bump_guard.py
"""
from __future__ import annotations

from pathlib import Path

from AIAEC.dataset_gen import linear_aec
from AIAEC.dataset_gen.released_pointer_bump_guard import (
    bumps_from_source,
    check_bump_against_head,
    check_chain,
    check_file_against_head,
    check_pointer_introduction,
    live_pointer_value_from_source,
)

# AIAEC/dataset_gen/released_pointer_bump_guard.py -> dataset_gen -> AIAEC ->
# Audio_ALG (the git root this guard's `git show HEAD:...` runs against).
_AUDIO_ALG_ROOT = Path(linear_aec.__file__).resolve().parents[2]
_REL_PATH = "AIAEC/dataset_gen/linear_aec.py"

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
# A real-looking but WRONG value: close to the real pre-bump value in spelling
# but not literally the hash HEAD has. 64 hex chars, deliberately not any real
# hash here.
_PLAUSIBLE_WRONG = "428aa9b796c2" + ("0" * (64 - len("428aa9b796c2") - 4)) + "562d"
assert len(_PLAUSIBLE_WRONG) == 64


def _live_pointer_value():
    """The single value a populated table holds, or None when it is retired."""
    compat = linear_aec.RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY
    assert len(compat) <= 1, (
        "this guard and these tests assume at most one tracked checkpoint "
        "pointer; a second entry needs this file revisited")
    return next(iter(compat.values()), None)


# --- Layer 1: check_chain -- stateless, survives any future commit ---------

def test_chain_checker_rejects_a_row_whose_old_does_not_match_the_priors_new():
    """Can-fail proof. Row 1's `old` is a plausible-looking value that is NOT
    row 0's `new` (assumed or recomputed instead of read off the chain)."""
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
    is not literally the previous row's `new`, fails here. With the pointer
    retired the table is empty with it."""
    live = _live_pointer_value()
    if live is None:
        assert linear_aec.RELEASED_INFERENCE_POINTER_BUMPS == ()
        return
    violations = check_chain(linear_aec.RELEASED_INFERENCE_POINTER_BUMPS, live)
    assert violations == [], violations


# --- Layer 2: check_bump_against_head -- the bump-time catch ---------------

def test_head_check_rejects_a_bump_with_the_wrong_assumed_old_value():
    """Can-fail proof. The pointer is bumped and a row IS added, but that row's
    `old` is not what HEAD had: a value the author assumed or recomputed, not
    one copied from `git show HEAD:...`."""
    head_value = "5fe64e46480c7e2a75c90718f1b64c09ca29f611d09713ea4ec2a9ef61c813f4"
    work_value = "419f473e609ec0c3fd645a9dcb7ccd53a742fb8c22d17a45578e6cc6df2ff8fa"
    bumps = (("2026-10-20", _PLAUSIBLE_WRONG, work_value),)
    violations = check_bump_against_head(head_value, work_value, bumps)
    assert violations != [], "a wrong `old` value must be rejected"
    assert head_value in violations[0] and work_value in violations[0]


def test_head_check_rejects_a_bump_with_no_row_at_all():
    """Can-fail proof. The value moved and the table has no row for it."""
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


# --- Retired pointer: no live value, no table, no silent re-introduction ----

_RETIRED_SOURCE = (
    "RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY = {}\n"
    "RELEASED_INFERENCE_POINTER_BUMPS = ()\n")
_POPULATED_SOURCE = (
    "RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY = {'%s': '%s'}\n"
    "RELEASED_INFERENCE_POINTER_BUMPS = ()\n" % (_A, _B))


def test_an_empty_pointer_table_has_no_live_value():
    assert live_pointer_value_from_source(_RETIRED_SOURCE) is None
    assert bumps_from_source(_RETIRED_SOURCE) == ()
    assert live_pointer_value_from_source(_POPULATED_SOURCE) == _B


def test_introduction_check_rejects_a_pointer_where_head_has_none():
    """Can-fail proof. A retired pointer coming back (the silent re-point this
    change forbids) is reported even though no bump row can document a HEAD
    value that does not exist."""
    violations = check_pointer_introduction(None, _B)
    assert violations != [] and _B in violations[0]


def test_introduction_check_accepts_retirement_and_the_unchanged_cases():
    assert check_pointer_introduction(_A, None) == []      # retiring is allowed
    assert check_pointer_introduction(None, None) == []    # stays retired
    assert check_pointer_introduction(_A, _B) == []        # a bump: other layer


def _git_repo_with(tmp_path, head_source, work_source):
    import subprocess
    repo = tmp_path / "repo"
    path = repo / "AIAEC" / "dataset_gen"
    path.mkdir(parents=True)
    target = path / "linear_aec.py"
    target.write_text(head_source)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": __import__("os").environ["PATH"], "HOME": str(tmp_path)}
    for cmd in (["init", "-q"], ["add", "."],
                ["-c", "commit.gpgsign=false", "commit", "-q", "-m", "head"]):
        subprocess.run(["git", "-C", str(repo)] + cmd, check=True, env=env)
    target.write_text(work_source)
    return repo


def test_file_check_flags_a_reintroduced_pointer_against_a_retired_head(tmp_path):
    repo = _git_repo_with(tmp_path, _RETIRED_SOURCE, _POPULATED_SOURCE)
    violations = check_file_against_head(repo, _REL_PATH)
    assert any("where HEAD has none" in v for v in violations), violations


def test_file_check_allows_retiring_a_pointer_and_staying_retired(tmp_path):
    retire = _git_repo_with(tmp_path / "a", _POPULATED_SOURCE, _RETIRED_SOURCE)
    assert check_file_against_head(retire, _REL_PATH) == []
    stay = _git_repo_with(tmp_path / "b", _RETIRED_SOURCE, _RETIRED_SOURCE)
    assert check_file_against_head(stay, _REL_PATH) == []


def test_file_check_flags_bump_rows_that_outlive_the_pointer(tmp_path):
    orphan = (
        "RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY = {}\n"
        "RELEASED_INFERENCE_POINTER_BUMPS = (('2026-10-21', '%s', '%s'),)\n"
        % (_A, _B))
    repo = _git_repo_with(tmp_path, _RETIRED_SOURCE, orphan)
    violations = check_file_against_head(repo, _REL_PATH)
    assert any("retires together" in v for v in violations), violations
