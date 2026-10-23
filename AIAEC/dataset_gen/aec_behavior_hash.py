"""Behaviour identity of the Python AEC sources, with NO heavy dependencies.

Split out of ``linear_aec`` so it can be imported by a bare interpreter. The
cross-version parity test has to run this under every CPython on the machine,
and most of those do not have torch or numpy installed; importing
``linear_aec`` there would fail on line 1 for reasons that have nothing to do
with the hash. Keep this module free of third-party imports.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import os


_AUDIO_ALG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_AEC_ROOT = os.path.join(_AUDIO_ALG_ROOT, "lib", "aec")
if not os.path.isdir(os.path.join(_AEC_ROOT, "python")):
    # Deployment/training copies carry only AINR/, AIAEC/ and the standalone
    # AEC/ checkout side by side (no git tree, so no lib/aec submodule): fall
    # back to a sibling AEC/ beside this root, then to the SE-tree layout one
    # level up. In a full checkout lib/aec exists and nothing changes.
    for _candidate in (
        os.path.join(_AUDIO_ALG_ROOT, "AEC"),
        os.path.join(os.path.dirname(_AUDIO_ALG_ROOT), "AEC"),
    ):
        if os.path.isdir(os.path.join(_candidate, "python")):
            _AEC_ROOT = _candidate
            break
_AEC_PYTHON = os.path.join(_AEC_ROOT, "python")


def _aec_python_files() -> list:
    """Every .py under lib/aec/python -- the PROVENANCE scope."""
    paths = []
    for root, _dirs, files in os.walk(_AEC_PYTHON):
        for name in files:
            if name.endswith(".py"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


# Directories under lib/aec/python whose contents cannot affect the materialized
# signal: diagnostics, tests and bench/eval tooling. Editing a test or a golden
# generator must not invalidate a trained checkpoint.
_NON_SIGNAL_DIRS = ("diag", "tests", "test", "bench", "eval", "tools")


def _aec_signal_path_files() -> list:
    """The subset of lib/aec/python that can change the produced signal.

    ``aec.py`` (the entry point) plus everything under ``modules/``. Anything
    else -- diag/, tests/, bench and eval tooling -- is excluded: those are
    reachable only from developer entry points, never from
    ``LinearAecProcessor``, so a change there cannot alter ``linear_error``.
    """
    keep = []
    for path in _aec_python_files():
        rel = os.path.relpath(path, _AEC_PYTHON).replace(os.sep, "/")
        top = rel.split("/")[0]
        if rel == "aec.py" or top == "modules":
            keep.append(path)
        elif top in _NON_SIGNAL_DIRS:
            continue
    if not keep:
        raise RuntimeError(
            "aec_behavior_hash found no lib/aec python sources under "
            f"{_AEC_PYTHON!r}; a trimmed checkout must carry the AEC/ "
            "checkout beside AIAEC/ (see the fallback list in this module)"
        )
    return keep


def _strip_version_assignments(tree):
    """Drop module-level ``__version__ = ...`` so a release bump is not a
    behaviour change. The version string is metadata; it never reaches the
    filter."""
    if not isinstance(tree, ast.Module):
        return tree
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__version__"
                for t in node.targets):
            continue
        kept.append(node)
    tree.body = kept or [ast.Pass()]
    return tree


# Identifies the CANONICALIZATION RULE used by `_canon_ast()`, independent of
# the AEC sources it is applied to. Recorded in the contract so that changing
# the serializer produces a readable "schema canon-ast-1 vs canon-ast-2"
# rejection instead of an opaque hash mismatch that looks like a code change.
# It is also folded into the digest, so the two can never disagree.
#
# Bump this whenever `_canon_ast()`'s output changes for unchanged input.
BEHAVIOR_HASH_SCHEMA = "canon-ast-1"


def _canon_constant(value) -> str:
    """Type-tagged repr, so 1, 1.0, True and '1' stay distinct.

    ``repr`` alone would collide ``True``/``1`` only via ``==``, not textually,
    but the tag also pins 1 vs 1.0 -- a real behaviour difference in filter
    coefficients. float repr round-trips exactly in Python 3.
    """
    return type(value).__name__ + ":" + repr(value)


def _canon_ast(node) -> str:
    """Serialize an AST deterministically and IDENTICALLY on every CPython.

    ``ast.dump()`` cannot be used: Python 3.13 omits fields equal to their
    default, so one source would hash differently across interpreters. This
    applies that omission (drop a field that is ``None`` or an empty list)
    uniformly on EVERY version. Fail-closed: a genuinely used new field is
    non-empty and DOES change the hash. Fields are emitted sorted by name.
    """
    if isinstance(node, ast.Constant):
        # `value` bypasses the empty-field rule: the literal `None` is a real
        # constant, not an absent field, and must not vanish.
        fields = ["value=" + _canon_constant(node.value)]
        fields += _canon_fields(node, skip=("value",))
        return "Constant(" + ",".join(fields) + ")"
    if isinstance(node, ast.AST):
        return type(node).__name__ + "(" + ",".join(_canon_fields(node)) + ")"
    if isinstance(node, list):
        return "[" + ",".join(_canon_ast(item) for item in node) + "]"
    if node is None:
        # Only reachable as a LIST ELEMENT (`kw_defaults=[None, ...]`,
        # `Dict.keys=[None]` for `**x`), where the position is significant.
        # A None-valued field is dropped by _canon_fields before getting here.
        return "None"
    return _canon_constant(node)


def _canon_fields(node, skip=()) -> list:
    out = []
    for name in sorted(node._fields):
        if name in skip:
            continue
        value = getattr(node, name, None)
        if value is None or (isinstance(value, list) and not value):
            continue  # the cross-version stability rule; see _canon_ast
        out.append(name + "=" + _canon_ast(value))
    return out


def _strip_docstrings(tree):
    """Drop docstring expression statements, in place."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


@functools.lru_cache(maxsize=1)
def aec_python_behavior_hash() -> str:
    """Hash the AEC sources' CODE, insensitive to comments and formatting.

    The COMPATIBILITY condition: it hashes the normalized AST, so a comment
    reflow keeps shards and checkpoints valid while any change to an
    expression, constant or control flow does not (the fix is to
    rematerialize). A mechanical hash rather than a hand-kept version integer,
    which could never differ between the two sides of a comparison.
    """
    digest = hashlib.sha256()
    digest.update(BEHAVIOR_HASH_SCHEMA.encode("utf-8"))
    digest.update(b"\0")
    for path in _aec_signal_path_files():
        rel = os.path.relpath(path, _AEC_PYTHON).replace(os.sep, "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        try:
            normalized = _canon_ast(
                _strip_version_assignments(
                    _strip_docstrings(ast.parse(source, filename=rel))))
        except SyntaxError:
            # Unparseable file: fall back to raw text rather than silently
            # dropping it from the hash.
            normalized = source
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def aec_python_source_hash() -> str:
    """Raw-text hash of the Python AEC sources, including uncommitted edits.

    PROVENANCE ONLY: it records which bytes produced a dataset but is NOT a
    compatibility condition, since a comment reflow changes it. Compatibility
    uses ``aec_python_behavior_hash()``. Computed once per process.
    """
    digest = hashlib.sha256()
    for path in _aec_python_files():
        rel = os.path.relpath(path, _AEC_PYTHON).replace(os.sep, "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            digest.update(handle.read())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "BEHAVIOR_HASH_SCHEMA",
    "aec_python_behavior_hash",
    "aec_python_source_hash",
]
