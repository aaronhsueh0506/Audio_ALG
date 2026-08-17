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

    ``ast.dump()`` cannot be used here. Python 3.13 changed it to omit fields
    that equal their default, so the same source hashes differently before and
    after: the 48 signal-path files digest to 89b866cd on 3.9 and 402acc1a on
    3.14 with no code difference at all (verified 2026-08-06). A dataset built
    under one interpreter would then be refused by training under another.

    The rule here is that same omission -- drop a field whose value is ``None``
    or an empty list -- applied uniformly on EVERY version, by this function
    rather than by the stdlib. That also absorbs fields which simply do not
    exist on older versions: ``type_params`` (PEP 695, added to FunctionDef/
    AsyncFunctionDef/ClassDef in 3.12) is absent on 3.9 and ``[]`` on 3.14, so
    both sides emit nothing. Those three are the ONLY fields 3.14 adds to a
    node type that 3.9 already has; everything else new (match statements, PEP
    695 declarations, t-strings) is a wholly new node type whose class name
    appears in the output the moment it is used.

    Fail-closed by construction: fields are read from the live ``_fields``, so
    a genuinely-used new field (``type_params`` on ``class Foo[T]:``) is
    non-empty, is emitted, and DOES change the hash. Nothing is silently
    dropped -- only provably-empty values are, and an empty value carries no
    behaviour.

    Fields are emitted in sorted-name order rather than ``_fields`` order,
    which is not guaranteed stable across versions when a field is appended.
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

    This is the COMPATIBILITY condition. It hashes the normalized AST rather
    than raw text, so reflowing a comment, rewording a docstring or reindenting
    a block does NOT invalidate existing shards or checkpoints, while any change
    to an actual expression, constant or control-flow path does -- and fails
    closed.

    Why a hash and not a hand-maintained ``behavior_version`` integer: every
    other field of ``LinearAecContract`` is either a literal pinned by
    ``__post_init__`` or echoed out of the recorded contract by both call sites,
    so a version integer would be the same constant on both sides of every
    comparison and could never differ. That is exactly the tautology that made
    this check vacuous once already (2026-08-06). A mechanical hash cannot be
    forgotten the way a manual bump can.

    Conservative by construction: a pure refactor (renaming a local, reordering
    independent statements) also changes this hash. Refusing to load is the safe
    direction; the fix is to rematerialize, not to loosen the check.

    The single exception is ``ACCEPTED_BEHAVIOR_HASH_MIGRATIONS`` in
    ``linear_aec``: an explicit table of recorded/current hash PAIRS shown by
    measurement to render byte-identical ``linear_error``, so an unchanged
    corpus and an already-trained checkpoint stay usable across a provably inert
    lib/aec change. It is a per-pair, one-way, single-hop exemption that applies
    only when this hash is the sole differing contract field -- not a way to
    accept old hashes generally, and no reason to weaken what is hashed here.
    Adding an entry does NOT bump ``BEHAVIOR_HASH_SCHEMA``: the canonicalizer is
    unchanged, only the sources it runs over.

    Stable across CPython versions -- see `_canon_ast()`. It must be: a dataset
    generated under one interpreter is routinely trained against under another,
    and an interpreter-dependent hash would refuse that pairing even though the
    AEC sources are byte-identical. `test_linear_aec_behavior_hash.py` pins this
    against every interpreter it can find on the machine.
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

    PROVENANCE ONLY. It records exactly which bytes produced a dataset, which is
    what resume/integrity wants, but it is deliberately NOT a compatibility
    condition: it changes when a comment is reflowed, which would invalidate
    byte-identical data and -- worse -- refuse an already-trained checkpoint
    that no rematerialization can repair. Compatibility uses
    ``aec_python_behavior_hash()``.

    Computed once per process. A long-lived process that edits the AEC sources
    after the first call keeps reporting the original hash.
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
