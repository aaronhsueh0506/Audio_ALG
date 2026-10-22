"""Staleness guard for the released-checkpoint hash-compatibility pointer.

See the comment block directly above ``RELEASED_INFERENCE_POINTER_BUMPS`` in
``linear_aec.py`` for the invariant this exists to enforce and the bug that
motivated it (2026-10-20 opus review item C2): an agent bumping
``RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY``'s single entry ASSUMED the
pointer's prior value was the hash immediately before their own change,
bumped it to their new hash, and in doing so silently re-admitted roughly a
month and 21 unrelated commits of un-vetted drift that had accumulated on the
pointer in between. The old test
(``test_released_checkpoint_exception_is_separate_and_targets_this_build``)
only asserted the pointer equals the LIVE hash right now; it had no way to
notice that the gap between two consecutive bumps was never individually
vetted.

Two independent layers, both pure functions (no git, no filesystem) plus one
thin git-backed wrapper:

``check_chain(bumps, live_value)``
    Stateless; runs in any checkout, forever, including long after the bump
    that introduced a row has been committed. Verifies the rows in
    ``RELEASED_INFERENCE_POINTER_BUMPS`` form one unbroken chain -- row i's
    ``old`` equals row (i-1)'s ``new`` -- and the last row's ``new`` equals
    the dict entry's current live value. A row whose ``old`` does not match
    the previous row's ``new`` is exactly "the author assumed/recomputed the
    old value instead of reading it off the previous row."

``check_bump_against_head(head_value, work_value, bumps)``
    Needs the file's value at git HEAD and in the working tree (or index).
    If they differ, the working tree's ``bumps`` must contain a row whose
    ``old`` is LITERALLY ``head_value`` and whose ``new`` is literally
    ``work_value``. This is what makes "I assumed the old value was X"
    structurally visible: X is checked against what git actually has at
    HEAD, not against the author's claim. If the two values are equal
    (nothing changed), this layer is silent.

``check_file_against_head(repo_root, rel_path, rev="HEAD")``
    Wires both layers to a real git checkout: reads ``rev:rel_path`` with
    ``git show``, reads the working-tree file off disk, extracts the dict
    and the bump table from both with ``ast.literal_eval`` (never ``eval``
    or ``ast.dump`` -- see ``reference_ast_dump_not_version_stable``), and
    returns the combined violation list.

Only single-hop bumps are modelled, matching this file's existing
"single hop by construction" convention for ``ACCEPTED_BEHAVIOR_HASH_MIGRATIONS``:
one commit is expected to add exactly one row connecting the value it found
at HEAD to the value it leaves in the working tree. A diff that jumps through
an intermediate value never recorded at any HEAD is out of scope.

Run standalone (also usable as a pre-commit-style check, exit 1 on failure):

    python3 -m AIAEC.dataset_gen.released_pointer_bump_guard [repo_root] [rel_path]

``make test`` / ``pytest AIAEC/tests`` also runs this guard, via
``AIAEC/tests/test_released_pointer_bump_guard.py``.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BumpRow = Tuple[str, str, str]  # (date, old_hash, new_hash)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_DICT_NAME = "RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY"
DEFAULT_TABLE_NAME = "RELEASED_INFERENCE_POINTER_BUMPS"
DEFAULT_REL_PATH = "AIAEC/dataset_gen/linear_aec.py"


def _is_full_hash(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


def check_chain(bumps: Sequence[BumpRow], live_value: str) -> List[str]:
    """Verifies ``bumps`` forms one unbroken chain ending at ``live_value``.

    Returns a list of violation strings; an empty list means the chain is
    sound. Pure function: no git, no filesystem, deterministic.
    """
    violations: List[str] = []
    for i, row in enumerate(bumps):
        if not (isinstance(row, (tuple, list)) and len(row) == 3):
            violations.append(
                f"row {i}: expected a (date, old, new) triple, got {row!r}")
            continue
        _date, old, new = row
        if not _is_full_hash(old):
            violations.append(
                f"row {i}: `old` is not a literal 64-hex-char hash "
                f"(abbreviations like 'ab12...cd34' are not accepted -- "
                f"the whole point is literal, un-elided matching): {old!r}")
        if not _is_full_hash(new):
            violations.append(
                f"row {i}: `new` is not a literal 64-hex-char hash: {new!r}")
        if i > 0:
            prev_new = bumps[i - 1][2]
            if old != prev_new:
                violations.append(
                    f"row {i}: `old` ({old!r}) does not equal row {i - 1}'s "
                    f"`new` ({prev_new!r}) -- rows must chain to the "
                    f"immediately preceding row, not to an independently "
                    f"assumed or recomputed value")
    if bumps:
        last_new = bumps[-1][2]
        if last_new != live_value:
            violations.append(
                f"the last row's `new` ({last_new!r}) does not equal the "
                f"live pointer value ({live_value!r}) -- the bump table is "
                f"out of date with the dict entry it documents")
    return violations


def check_bump_against_head(head_value: Optional[str], work_value: str,
                             bumps: Sequence[BumpRow]) -> List[str]:
    """Verifies that if the pointer's value changed since HEAD, the working
    tree's ``bumps`` documents exactly that hop with HEAD's real value.

    Silent (returns ``[]``) when ``head_value is None`` (no HEAD copy to
    compare against, e.g. a brand-new file) or when ``head_value ==
    work_value`` (nothing changed -- a pre-commit-style check with no
    pending bump to vet). Pure function: no git, no filesystem.
    """
    if head_value is None or head_value == work_value:
        return []
    matching = [row for row in bumps
                if len(row) == 3 and row[1] == head_value and row[2] == work_value]
    if not matching:
        return [
            f"the pointer's value changed from {head_value!r} (HEAD) to "
            f"{work_value!r} (working tree), but no row in "
            f"RELEASED_INFERENCE_POINTER_BUMPS has old=={head_value!r} and "
            f"new=={work_value!r}. Add one -- and copy `old` from "
            f"`git show HEAD:...`, do not assume or recompute it."
        ]
    return []


def _extract_module_level_value(tree: ast.Module, name: str):
    """Returns the literal value assigned to module-level ``name``, or
    ``None`` if no such assignment exists. Handles both plain ``Assign`` and
    annotated ``AnnAssign`` (``NAME: Type = value``, used throughout
    linear_aec.py)."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return ast.literal_eval(node.value)
    return None


def live_pointer_value_from_source(
        source: str, dict_name: str = DEFAULT_DICT_NAME) -> Optional[str]:
    """The single value held by the ``dict_name`` dict in ``source``.

    Raises ``ValueError`` if the dict exists but does not hold exactly one
    entry -- this guard models a single tracked checkpoint pointer, matching
    the dict's own current shape; a second entry needs this function (and
    the guard's single-row-per-bump assumption) revisited, not silently
    picking one of several values.
    """
    tree = ast.parse(source)
    value = _extract_module_level_value(tree, dict_name)
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(
            f"{dict_name} is expected to hold exactly one entry; got "
            f"{len(value) if isinstance(value, dict) else type(value)!r}")
    return next(iter(value.values()))


def bumps_from_source(source: str,
                       table_name: str = DEFAULT_TABLE_NAME) -> Tuple[BumpRow, ...]:
    """The bump rows recorded in ``source``'s ``table_name`` tuple, or ``()``
    if the table is absent (e.g. at a HEAD revision before this guard
    existed)."""
    tree = ast.parse(source)
    value = _extract_module_level_value(tree, table_name)
    return tuple(value) if value else ()


def read_git_blob(repo_root: Path, rel_path: str, rev: str) -> Optional[str]:
    """``git show <rev>:<rel_path>`` from ``repo_root``, or ``None`` if that
    revision/path does not exist (e.g. a brand-new untracked file, or no git
    at all)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{rev}:{rel_path}"],
            capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout


def check_file_against_head(
        repo_root: Path,
        rel_path: str = DEFAULT_REL_PATH,
        rev: str = "HEAD",
        dict_name: str = DEFAULT_DICT_NAME,
        table_name: str = DEFAULT_TABLE_NAME) -> List[str]:
    """Runs both layers against a real file: the working tree's own chain
    (``check_chain``), plus, if the pointer moved since ``rev``, that the
    move is documented against ``rev``'s real value (``check_bump_against_head``).
    """
    repo_root = Path(repo_root)
    work_path = repo_root / rel_path
    work_source = work_path.read_text()

    work_live = live_pointer_value_from_source(work_source, dict_name)
    work_bumps = bumps_from_source(work_source, table_name)

    violations: List[str] = []
    if work_live is not None:
        violations += check_chain(work_bumps, work_live)

    head_source = read_git_blob(repo_root, rel_path, rev)
    if head_source is not None and work_live is not None:
        head_live = live_pointer_value_from_source(head_source, dict_name)
        violations += check_bump_against_head(head_live, work_live, work_bumps)

    return violations


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path(argv[0]) if len(argv) >= 1 else Path(__file__).resolve().parents[2]
    rel_path = argv[1] if len(argv) >= 2 else DEFAULT_REL_PATH

    violations = check_file_against_head(repo_root, rel_path)
    if violations:
        print(f"released-pointer staleness guard: FAILED ({rel_path})",
              file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print(f"released-pointer staleness guard: OK ({rel_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
