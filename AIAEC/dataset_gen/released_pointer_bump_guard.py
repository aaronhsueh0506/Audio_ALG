"""Staleness guard for the released-checkpoint hash-compatibility pointer.

See the comment block above ``RELEASED_INFERENCE_POINTER_BUMPS`` in
``linear_aec.py``. A bump of ``RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY``
must not assume the pointer's prior value was the hash immediately before the
author's own change, or unvetted drift accumulated on the pointer is silently
re-admitted; asserting that the pointer equals the live hash cannot see that.

Two pure layers (no git, no filesystem) plus one git-backed wrapper:

``check_chain(bumps, live_value)``
    Stateless. The rows of ``RELEASED_INFERENCE_POINTER_BUMPS`` form one
    unbroken chain (row i's ``old`` equals row i-1's ``new``) and the last
    row's ``new`` equals the entry's live value.

``check_bump_against_head(head_value, work_value, bumps)``
    If the entry differs between git HEAD and the working tree, ``bumps`` must
    hold a row whose ``old`` is literally HEAD's value and whose ``new`` is
    literally the working tree's. Silent when nothing changed.

``check_file_against_head(repo_root, rel_path, rev="HEAD")``
    Wires both layers to a git checkout: reads ``rev:rel_path`` with ``git
    show``, extracts the dict and the bump table from both copies with
    ``ast.literal_eval`` (never ``eval``/``ast.dump``) and returns the combined
    violations.

Only single-hop bumps are modelled: one commit adds exactly one row connecting
HEAD's value to the working tree's.

The dict may be EMPTY (every pointer retired, as it is now): then there is no
live value, the bump table must be empty, and a pointer that appears where HEAD
had none is reported by ``check_pointer_introduction`` instead of being waved
through as "nothing to compare".

Run standalone (exit 1 on failure):

    python3 -m AIAEC.dataset_gen.released_pointer_bump_guard [repo_root] [rel_path]

It also runs via ``AIAEC/tests/test_released_pointer_bump_guard.py``.
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


def check_pointer_introduction(head_value: Optional[str],
                               work_value: Optional[str]) -> List[str]:
    """Verifies a pointer is not (re)introduced where ``HEAD`` has none.

    A retired pointer must not come back as a side effect of an unrelated edit:
    the retirement tests, not this guard, decide whether a new entry is
    admissible, so the introduction itself is reported until HEAD carries it.
    Pure function: no git, no filesystem.
    """
    if head_value is None and work_value is not None:
        return [
            f"the working tree carries a released-inference pointer to "
            f"{work_value!r} where HEAD has none. The table is retired on "
            f"purpose; re-adding an entry needs its own evidence and its own "
            f"refusal/warning tests (tests/test_linear_aec_behavior_migration.py) "
            f"before this check is relaxed."
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
    """The single value held by the ``dict_name`` dict in ``source``, or
    ``None`` when the dict is absent or empty (no pointer).

    Raises ``ValueError`` if the dict holds more than one entry -- this guard models a single tracked checkpoint pointer, matching
    the dict's own current shape; a second entry needs this function (and
    the guard's single-row-per-bump assumption) revisited, not silently
    picking one of several values.
    """
    tree = ast.parse(source)
    value = _extract_module_level_value(tree, dict_name)
    if value is None or value == {}:
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
    elif work_bumps:
        violations.append(
            f"{table_name} has {len(work_bumps)} row(s) but {dict_name} holds "
            f"no pointer; the table retires together with its entry")

    head_source = read_git_blob(repo_root, rel_path, rev)
    if head_source is not None:
        head_live = live_pointer_value_from_source(head_source, dict_name)
        violations += check_pointer_introduction(head_live, work_live)
        if work_live is not None:
            violations += check_bump_against_head(
                head_live, work_live, work_bumps)

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
