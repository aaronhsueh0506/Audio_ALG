"""The AEC behaviour hash must be stable across CPython versions.

Why this file exists: the first implementation used ``ast.dump()``. Python 3.13
changed ``ast.dump`` to omit fields equal to their default, so the SAME 48
signal-path source files digested to ``89b866cd`` under 3.9 and ``402acc1a``
under 3.14 with no code difference whatsoever. That is not a cosmetic problem.
The behaviour hash is the compatibility gate: a dataset generated under one
interpreter would have been refused by training under another, and a trained
checkpoint would have been unloadable by an interpreter upgrade alone -- a
failure no rematerialization can repair.

Two independent properties are pinned here, and both are needed:

  * STABILITY -- the same sources hash the same under every interpreter on the
    machine. Tested against the real corpus, in subprocesses.
  * SENSITIVITY -- a real code change still changes the hash. A canonicalizer
    that returned a constant would be perfectly stable and completely useless,
    so stability is only meaningful next to a mutation test.

The stability test also runs a CONTROL that reproduces the old ``ast.dump``
behaviour under the same interpreters and asserts it DISAGREES. Without that,
the test would pass vacuously on a machine with only one Python installed, and
would keep passing if the canonicalizer were quietly reverted.
"""

import ast
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from AIAEC.dataset_gen.aec_behavior_hash import (
    BEHAVIOR_HASH_SCHEMA,
    _canon_ast,
    _strip_docstrings,
    _strip_version_assignments,
    aec_python_behavior_hash,
)

_DATASET_GEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _canon(source: str) -> str:
    """Canonical form of a source string, with the same stripping the real
    hash applies."""
    return _canon_ast(
        _strip_version_assignments(_strip_docstrings(ast.parse(source)))
    )


# ── interpreter discovery ────────────────────────────────────────────────────

def _interpreters() -> list:
    """Every distinct CPython on this machine, by resolved realpath.

    Deliberately not a pinned list of versions: the point is to catch drift
    against whatever the developer actually has, and a hardcoded list silently
    tests nothing on a machine that lacks those exact versions.
    """
    found = {}
    candidates = [sys.executable]
    for minor in range(8, 20):
        candidates.append(f"python3.{minor}")
    for name in candidates:
        path = shutil.which(name) if not os.path.isabs(name) else name
        if not path or not os.path.exists(path):
            continue
        real = os.path.realpath(path)
        found.setdefault(real, path)
    return sorted(found.values())


_WORKER = textwrap.dedent(
    """
    import hashlib, os, sys
    sys.path.insert(0, {gen!r})
    import ast
    import aec_behavior_hash as M

    mode = sys.argv[1]
    digest = hashlib.sha256()
    if mode == "canonical":
        print(M.aec_python_behavior_hash())
    else:
        # CONTROL: the pre-fix ast.dump() path, reproduced exactly.
        for path in M._aec_signal_path_files():
            rel = os.path.relpath(path, M._AEC_PYTHON).replace(os.sep, "/")
            digest.update(rel.encode("utf-8")); digest.update(b"\\0")
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
            normalized = ast.dump(
                M._strip_version_assignments(
                    M._strip_docstrings(ast.parse(source, filename=rel))),
                annotate_fields=True, include_attributes=False)
            digest.update(normalized.encode("utf-8")); digest.update(b"\\0")
        print(digest.hexdigest())
    """
).format(gen=_DATASET_GEN)


def _run(interp: str, mode: str) -> str:
    result = subprocess.run(
        [interp, "-c", _WORKER, mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{interp} ({mode}) failed: {result.stderr.decode()[-2000:]}"
        )
    return result.stdout.decode().strip()


def test_behavior_hash_is_identical_across_interpreters():
    interps = _interpreters()
    if len(interps) < 2:
        pytest.skip(f"need >=2 CPython interpreters, found {interps}")

    canonical = {i: _run(i, "canonical") for i in interps}
    assert len(set(canonical.values())) == 1, (
        "behaviour hash is interpreter-dependent -- a dataset built under one "
        f"Python would be refused under another: {canonical}"
    )


def test_control_ast_dump_really_does_differ():
    """Proves the stability test above is not vacuous.

    If the interpreters on this machine happened to agree under plain
    ``ast.dump`` too, the test above would pass without the canonicalizer doing
    any work. This asserts the hazard is real HERE, on this machine, with these
    interpreters -- so a revert of ``_canon_ast`` cannot pass silently.
    """
    interps = _interpreters()
    if len(interps) < 2:
        pytest.skip(f"need >=2 CPython interpreters, found {interps}")

    control = {i: _run(i, "astdump") for i in interps}
    if len(set(control.values())) == 1:
        pytest.skip(
            "all interpreters here agree under raw ast.dump too (too close in "
            f"version to exhibit the hazard): {sorted(set(control.values()))}"
        )
    canonical = {i: _run(i, "canonical") for i in interps}
    assert len(set(canonical.values())) == 1
    assert len(set(control.values())) > 1


# ── sensitivity: the hash must still notice real changes ─────────────────────

_BASE = """
__version__ = "1.2.3"

ALPHA = 0.9


def step(x, y=1, *rest, key, **kw):
    '''Docstring.'''
    # a comment
    total = 0
    for i in range(3):
        total += ALPHA * x[i]
    return total
"""


@pytest.mark.parametrize("variant,label", [
    (_BASE.replace("# a comment", "# a totally different comment"), "comment"),
    (_BASE.replace("'''Docstring.'''", "'''Reworded.'''"), "docstring"),
    (_BASE.replace('__version__ = "1.2.3"', '__version__ = "4.0.0rc1"'), "version"),
    (_BASE.replace("    total = 0\n", "    total  =  0\n"), "whitespace"),
])
def test_non_behavioural_edits_do_not_change_canonical_form(variant, label):
    assert _canon(variant) == _canon(_BASE), f"{label} changed the hash"


@pytest.mark.parametrize("variant,label", [
    (_BASE.replace("ALPHA = 0.9", "ALPHA = 0.85"), "constant value"),
    (_BASE.replace("ALPHA = 0.9", "ALPHA = 0.90000001"), "tiny constant delta"),
    (_BASE.replace("range(3)", "range(4)"), "loop bound"),
    (_BASE.replace("total += ALPHA * x[i]", "total -= ALPHA * x[i]"), "operator"),
    (_BASE.replace("y=1", "y=1.0"), "int vs float default"),
    (_BASE.replace("y=1", "y=True"), "int vs bool default"),
    (_BASE.replace("return total", "return total\n"
                                   "\n\ndef extra():\n    return 1"), "new function"),
])
def test_behavioural_edits_do_change_canonical_form(variant, label):
    assert _canon(variant) != _canon(_BASE), f"{label} did NOT change the hash"


def test_empty_field_rule_does_not_conflate_distinct_code():
    """The stability rule drops None/empty-list fields. Prove that dropping
    them cannot merge two genuinely different programs."""
    pairs = [
        ("try:\n    f()\nexcept E:\n    pass\n",
         "try:\n    f()\nexcept E:\n    pass\nelse:\n    g()\n"),
        ("class K: pass\n", "class K(Base): pass\n"),
        ("def f(): pass\n", "@deco\ndef f(): pass\n"),
        ("def f(a): pass\n", "def f(a, /): pass\n"),
        ("f(a)\n", "f(a, b=1)\n"),
        ("x = None\n", "x = 0\n"),
        ("x = None\n", "x = ()\n"),
        ("x = []\n", "x = None\n"),
    ]
    for left, right in pairs:
        assert _canon(left) != _canon(right), (left, right)


def test_literal_none_survives_canonicalization():
    """`Constant.value` bypasses the empty-field rule; `x = None` must not
    canonicalize to the same thing as a missing value."""
    assert "None" in _canon("x = None\n")
    assert _canon("x = None\n") != _canon("x = False\n")


# ── the schema tag ───────────────────────────────────────────────────────────

def test_schema_is_folded_into_the_digest():
    """Changing the canonicalization rule must change the hash, not just the
    advertised schema string -- otherwise the two could disagree."""
    from AIAEC.dataset_gen import aec_behavior_hash as M

    before = aec_python_behavior_hash()
    original = M.BEHAVIOR_HASH_SCHEMA
    try:
        M.BEHAVIOR_HASH_SCHEMA = original + "-mutated"
        aec_python_behavior_hash.cache_clear()
        after = aec_python_behavior_hash()
    finally:
        M.BEHAVIOR_HASH_SCHEMA = original
        aec_python_behavior_hash.cache_clear()

    assert before != after
    assert aec_python_behavior_hash() == before


def test_contract_records_and_compares_the_schema():
    from AIAEC.dataset_gen.linear_aec import (
        LinearAecContract, make_linear_aec_contract,
    )

    contract = make_linear_aec_contract(16000)
    assert contract.behavior_hash_schema == BEHAVIOR_HASH_SCHEMA
    assert "behavior_hash_schema" in contract.compatibility_dict()

    stale = contract.as_dict()
    stale["behavior_hash_schema"] = "canon-ast-0"
    with pytest.raises(ValueError, match="canon-ast-0"):
        LinearAecContract.from_dict(stale)


def test_v2_contract_still_rejected_after_schema_field_added():
    from AIAEC.dataset_gen.linear_aec import (
        LinearAecContract, make_linear_aec_contract,
    )

    v2 = make_linear_aec_contract(16000).as_dict()
    v2.pop("aec_behavior_hash")
    v2.pop("behavior_hash_schema")
    with pytest.raises(ValueError, match="v2"):
        LinearAecContract.from_dict(v2)
