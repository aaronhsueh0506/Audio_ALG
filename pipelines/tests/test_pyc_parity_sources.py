"""The sources pyc_parity.py ties a result to exist, and cover what its C driver includes.

pyc_parity.py hashes SOURCES into the JSON only after every scenario has run, so a path that went stale
(a file moved or deleted) used to surface at the end of a multi-hour run. This reads the list straight
from the script's text and resolves it the way the script does, in milliseconds.
"""
import ast
import os
import pathlib
import re


_REPO = pathlib.Path(__file__).resolve().parents[2]            # Audio_ALG
_PARITY = _REPO / 'pipelines' / 'tests' / 'pyc_parity'


def _literal(name):
    tree = ast.parse((_PARITY / 'pyc_parity.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('%s is not a literal assignment of pyc_parity.py' % name)


def _aec_dir():
    """AEC_DIR as lib/deps.mk resolves it: the environment, else the sibling AEC checkout's c_impl."""
    repo = pathlib.Path(os.environ.get('AEC_REPO_DIR') or _REPO.parent / 'AEC')
    return pathlib.Path(os.environ.get('AEC_DIR') or repo / 'c_impl')


def _resolve(rel, prefix):
    if rel.startswith(prefix):
        return _aec_dir() / rel[len(prefix):]
    return _REPO.parent / rel


def test_every_hashed_source_exists():
    prefix = _literal('AEC_SOURCE_PREFIX')
    sources = _literal('SOURCES')
    assert len(sources) == len(set(sources))
    missing = [rel for rel in sources if not _resolve(rel, prefix).is_file()]
    assert not missing, 'pyc_parity.py SOURCES names files that do not exist: %s' % missing


def test_hashed_sources_include_the_driver_headers_of_this_repo():
    """A pipelines header that pyc_drv.c includes is judged code and has to be hashed."""
    sources = set(_literal('SOURCES'))
    driver = (_PARITY / 'pyc_drv.c').read_text(encoding='utf-8')
    local = [m for m in re.findall(r'^#\s*include\s+"([^"]+)"', driver, re.MULTILINE)
             if list((_REPO / 'pipelines').rglob(pathlib.Path(m).name))]
    assert local, 'pyc_drv.c includes no pipelines header any more; this check no longer sees anything'
    for name in local:
        hits = [p for p in (_REPO / 'pipelines').rglob(pathlib.Path(name).name)
                if not {'obj', 'bin'} & set(p.relative_to(_REPO).parts)]
        rels = {'Audio_ALG/' + p.relative_to(_REPO).as_posix() for p in hits}
        assert rels & sources, '%s is included by pyc_drv.c but not in SOURCES' % name
