"""Guard for ``docs/c_python_function_map.md`` (the C <-> Python function map).

The map is a hand-maintained index: for every C function in the covered files
it names the Python symbol the function mirrors (or says there is none).  A map
that nobody checks rots silently, so this module pins the parts that can be
checked without running any DSP:

* the document parses under the strict cell grammar written in its header
  (a header or column change must fail here, never pass vacuously);
* every Python symbol it names still exists -- resolved with ``ast`` on the
  source file, nothing is imported, so no torch / numpy / checkpoint is needed;
* every C function it names is still defined in the named file, and every
  function defined in a covered file is accounted for in exactly one row;
* the coverage statistics in the header equal the numbers recomputed from the
  table rows;
* every gap id (S-xx / D-xx / T-xx) a row cites is defined in the document's
  glossary, and every repo path a row cites as its parity tool exists.

It checks that the map still points at something real.  It does NOT check that
the two sides still agree numerically -- that is the job of the parity
scripts and the C-parity pytests the "tool" column points at.

Environment overrides (used to prove the checks can fail, see the document's
"how to maintain" section):

``C_PY_MAP_MD``  path of the document to check (default: docs/ next to this
                 tree)
``C_PY_MAP_ROOT``  Audio_ALG root used to resolve repo-relative paths
``C_PY_MAP_AC``  audiolib checkout (default: ``<root>/../audiolib/common``)

Run from the Audio_ALG directory::

    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
        pipelines/tests/test_c_python_map.py
"""
from __future__ import annotations

import ast
import os
import pathlib
import re

import pytest

HERE = pathlib.Path(__file__).resolve()
ROOT = pathlib.Path(os.environ.get('C_PY_MAP_ROOT') or HERE.parents[2])
COMMON_DIR = pathlib.Path(os.environ.get('C_PY_MAP_AC') or (ROOT.parent / 'audiolib' / 'common'))
# AUDIOLIB_DIR covers the rest of the sibling audiolib checkout (doa/bf/VAD),
# not just its common/ subtree; COMMON_DIR keeps its own override for
# backward compatibility with C_PY_MAP_AC callers that already point it at
# a common/ directory directly.
AUDIOLIB_DIR = pathlib.Path(os.environ.get('C_PY_MAP_AUDIOLIB') or (ROOT.parent / 'audiolib'))
DOC = pathlib.Path(os.environ.get('C_PY_MAP_MD') or (ROOT / 'docs' / 'c_python_function_map.md'))

RELATIONS = ('mirror', 'partial', 'C-only', 'Python-only', '未能判斷')
NONE_CELL = '—'
INFERRED_PREFIX = '（推論）'

_ID_RE = re.compile(r'^[A-Z]+(?:-[A-Z0-9]+)+-\d+$')
_C_SPAN_RE = re.compile(
    r'`([A-Za-z0-9_./+-]+\.[ch]):\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*`')
_PY_SPAN_RE = re.compile(
    r'`([A-Za-z0-9_./+-]+\.py):([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)`')
_GAP_RE = re.compile(r'\b([SDT]-\d{2})\b')
_PATH_TOKEN_RE = re.compile(r'`((?:pipelines|AINR|AIAEC|lib|audiolib|docs)/[A-Za-z0-9_./+-]+)(?:::\w+)?`')


# ---------------------------------------------------------------------------
# C function extraction (comment/string stripped brace walker)
# ---------------------------------------------------------------------------

def _strip_c(src):
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r'[^\n]', ' ', src[i:j]))
            i = j
        elif src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
        elif c in ('"', "'"):
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == '\\' else 1
            out.append(c + ' ' * (j - i - 1) + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def _head_name(head):
    """Function name of a definition head (text since the previous ``;``,
    ``}`` or preprocessor line): the identifier before the parenthesis group
    that closes the head, skipping trailing ``__attribute__((..))``."""
    h = head.rstrip()
    while True:
        if not h.endswith(')'):
            return None
        depth = 0
        j = len(h) - 1
        while j >= 0:
            if h[j] == ')':
                depth += 1
            elif h[j] == '(':
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        if j < 0:
            return None
        m = re.search(r'([A-Za-z_]\w*)\s*$', h[:j])
        if not m:
            return None
        name = m.group(1)
        if name in ('__attribute__', '__declspec'):
            h = h[:m.start()].rstrip()
            continue
        if '=' in h[:m.start()]:
            return None
        return name


def c_functions(path):
    """``{name: is_static}`` of every function *defined* (with a body) at file
    scope in ``path``.  Names defined twice (``#if`` variants) count once; the
    result is static only if every definition is."""
    s = _strip_c(pathlib.Path(path).read_text(encoding='utf-8', errors='replace'))
    found = {}
    depth = 0
    wrappers = 0                       # open ``extern "C" {`` blocks
    start = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '{' and depth == 0 and re.search(r'extern\s*"\s*"\s*$', s[start:i]):
            wrappers += 1
            start = i + 1
        elif c == '}' and depth == 0 and wrappers:
            wrappers -= 1
            start = i + 1
        elif c == '{':
            if depth == 0:
                head = s[start:i]
                name = _head_name(head)
                if (name and name not in ('if', 'while', 'for', 'switch')
                        and not re.match(r'\s*(struct|union|enum|typedef)\b', head)):
                    static = 'static' in head[:head.rfind(name)].split()
                    found[name] = found.get(name, True) and static
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                start = i + 1
        elif c == ';' and depth == 0:
            start = i + 1
        elif c == '#' and depth == 0:
            j = i
            while j < n and s[j] != '\n':
                j += 2 if (s[j] == '\\' and j + 1 < n and s[j + 1] == '\n') else 1
            i = j
            start = i
            continue
        i += 1
    return found


# ---------------------------------------------------------------------------
# Python symbol resolution (ast only, never imports)
# ---------------------------------------------------------------------------

def _flatten(body):
    """Statements of a body, looking through module-level ``if`` / ``try`` /
    ``except`` blocks (a def or assignment guarded by an import fallback)."""
    for node in body:
        if isinstance(node, ast.If):
            yield from _flatten(node.body)
            yield from _flatten(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _flatten(node.body)
            for handler in node.handlers:
                yield from _flatten(handler.body)
            yield from _flatten(node.orelse)
            yield from _flatten(node.finalbody)
        else:
            yield node


def _names_in(body):
    names = {}
    for node in _flatten(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.setdefault(sub.id, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.setdefault(node.target.id, node)
    return names


_TREE_CACHE = {}


def _tree(path):
    key = str(path)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = ast.parse(pathlib.Path(path).read_text(encoding='utf-8'),
                                     filename=key)
    return _TREE_CACHE[key]


def python_symbol_exists(path, symbol):
    """True when ``symbol`` (``name`` or ``Class.member``) is DEFINED in
    ``path``: a def / class / module-level assignment, or, after a class, a
    method / class-body assignment (dataclass field).  A name that is merely
    imported does not count, so rows point at definitions."""
    scope = _names_in(_tree(path).body)
    parts = symbol.split('.')
    for index, part in enumerate(parts):
        node = scope.get(part)
        if node is None:
            return False
        if index < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                return False
            scope = _names_in(node.body)
    return True


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

class MapError(Exception):
    pass


def _region(text, name):
    begin = '<!-- map:%s:begin -->' % name
    end = '<!-- map:%s:end -->' % name
    if begin not in text or end not in text:
        raise MapError('marker pair %s / %s missing from the document' % (begin, end))
    body = text.split(begin, 1)[1].split(end, 1)[0]
    return body


def _table_rows(body):
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        # split on unescaped pipes only; a literal pipe inside a cell is written \|
        cells = [c.strip().replace('\\|', '|')
                 for c in re.split(r'(?<!\\)\|', line[1:-1] if line.endswith('|') else line[1:])]
        if all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
            continue
        rows.append(cells)
    return rows


def _spans(cell, regex, what, row_id):
    """All spans of a cell; the cell must consist of nothing but spans joined by
    ``<br>`` (or be the dash).  Returns a list of regex match groups."""
    if cell == NONE_CELL:
        return []
    rest = regex.sub('', cell)
    rest = re.sub(r'<br\s*/?>', '', rest).strip()
    if rest:
        raise MapError('row %s: %s cell has text outside `file:symbol` spans: %r'
                       % (row_id, what, rest))
    found = regex.findall(cell)
    if not found:
        raise MapError('row %s: %s cell %r holds no span (use %s for none)'
                       % (row_id, what, cell, NONE_CELL))
    return found


class Row:
    def __init__(self, cells):
        if len(cells) != 7:
            raise MapError('map row has %d cells, expected 7: %r' % (len(cells), cells[:2]))
        self.id, c_cell, py_cell, self.relation, gaps, tools, self.note = cells
        if not _ID_RE.match(self.id):
            raise MapError('bad row id %r' % self.id)
        if self.relation not in RELATIONS:
            raise MapError('row %s: relation %r is not one of %s'
                           % (self.id, self.relation, ', '.join(RELATIONS)))
        self.c = []                     # [(file, name)]
        for path, names in _spans(c_cell, _C_SPAN_RE, 'C', self.id):
            for name in re.split(r'\s*,\s*', names.strip()):
                self.c.append((path, name))
        self.py = [(p, s) for p, s in _spans(py_cell, _PY_SPAN_RE, 'Python', self.id)]
        self.gaps = sorted(set(_GAP_RE.findall(gaps)))
        self.tools_text = tools
        self.inferred = self.note.startswith(INFERRED_PREFIX)
        if self.relation == 'Python-only':
            if self.c or not self.py:
                raise MapError('row %s: Python-only needs a dash in the C cell and a Python symbol' % self.id)
        elif self.relation in ('mirror', 'partial'):
            if not self.c or not self.py:
                raise MapError('row %s: %s needs both a C function and a Python symbol' % (self.id, self.relation))
        else:                           # C-only / 未能判斷
            if not self.c or self.py:
                raise MapError('row %s: %s needs a C function and a dash in the Python cell'
                               % (self.id, self.relation))

    @property
    def area(self):
        return self.id.split('-')[0]


def parse_document(text):
    rows = [Row(cells) for cells in _table_rows(_region(text, 'rows'))
            if cells[0] != 'ID']
    if not rows:
        raise MapError('no map rows found between the map:rows markers')
    seen = set()
    for row in rows:
        if row.id in seen:
            raise MapError('duplicate row id %s' % row.id)
        seen.add(row.id)
    coverage = {}
    for cells in _table_rows(_region(text, 'coverage')):
        if len(cells) != 3 or cells[0] in ('檔案',):
            continue
        path = cells[0].strip('`')
        if cells[1] not in ('all', 'extern'):
            raise MapError('coverage row %s: scope %r is not all/extern' % (path, cells[1]))
        coverage[path] = (cells[1], int(cells[2]))
    if not coverage:
        raise MapError('no coverage rows found between the map:coverage markers')
    stats = {}
    for cells in _table_rows(_region(text, 'stats')):
        if len(cells) != 9 or cells[0] == '範圍':
            continue
        key = re.search(r'`(\w+)`', cells[0])
        key = key.group(1) if key else cells[0]
        stats[key] = [int(c) for c in cells[1:]]
    if not stats:
        raise MapError('no statistics rows found between the map:stats markers')
    glossary = set(_GAP_RE.findall(_region(text, 'gaps')))
    return rows, coverage, stats, glossary


def _load():
    if not DOC.is_file():
        raise MapError('document %s not found' % DOC)
    return parse_document(DOC.read_text(encoding='utf-8'))


def _resolve_c(path):
    """Repo-relative C path -> file on disk, or None when audiolib is
    absent (standalone Audio_ALG checkout)."""
    if path.startswith('audiolib/common/'):
        if not COMMON_DIR.is_dir():
            return None
        return COMMON_DIR / path[len('audiolib/common/'):]
    if path.startswith('audiolib/'):
        if not AUDIOLIB_DIR.is_dir():
            return None
        return AUDIOLIB_DIR / path[len('audiolib/'):]
    return ROOT / path


@pytest.fixture(scope='module')
def doc():
    try:
        return _load()
    except MapError as exc:
        pytest.fail('c_python_function_map.md does not parse: %s' % exc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_document_parses_and_is_not_vacuous(doc):
    rows, coverage, stats, glossary = doc
    assert len(rows) >= 100, 'a function map this small cannot cover the covered files'
    symbols = sum(len(r.py) for r in rows)
    functions = sum(len(r.c) for r in rows)
    assert symbols >= 50 and functions >= 500, (symbols, functions)
    assert {r.relation for r in rows} >= {'mirror', 'partial', 'C-only', 'Python-only'}
    assert len(coverage) >= 40


def test_every_python_symbol_exists(doc):
    rows, _, _, _ = doc
    missing = []
    checked = 0
    for row in rows:
        for rel, symbol in row.py:
            path = ROOT / rel
            checked += 1
            if not path.is_file():
                missing.append('%s: %s (file missing)' % (row.id, rel))
            elif not python_symbol_exists(path, symbol):
                missing.append('%s: %s:%s (symbol not found)' % (row.id, rel, symbol))
    assert checked > 0, 'no Python symbol was checked'
    assert not missing, 'Python symbols named by the map no longer exist:\n  ' + '\n  '.join(missing)


def test_every_listed_c_function_is_defined(doc):
    rows, _, _, _ = doc
    problems = []
    checked = 0
    cache = {}
    for row in rows:
        for rel, name in row.c:
            path = _resolve_c(rel)
            if path is None:
                continue
            if not path.is_file():
                problems.append('%s: %s (file missing)' % (row.id, rel))
                continue
            if rel not in cache:
                cache[rel] = c_functions(path)
            checked += 1
            if name not in cache[rel]:
                problems.append('%s: %s:%s (no such function definition)' % (row.id, rel, name))
    assert checked > 0, 'no C function was checked (audiolib checkout missing and nothing else?)'
    assert not problems, 'C functions named by the map no longer exist:\n  ' + '\n  '.join(problems)


def test_every_c_function_in_a_covered_file_is_listed_exactly_once(doc):
    rows, coverage, _, _ = doc
    listed = {}
    duplicates = []
    for row in rows:
        for key in row.c:
            if key in listed:
                duplicates.append('%s and %s: %s:%s' % (listed[key], row.id, key[0], key[1]))
            listed[key] = row.id
    assert not duplicates, 'functions listed in more than one row:\n  ' + '\n  '.join(duplicates)

    problems = []
    audited = 0
    for rel, (scope, count) in sorted(coverage.items()):
        path = _resolve_c(rel)
        if path is None:
            continue
        if not path.is_file():
            problems.append('%s: covered file is missing' % rel)
            continue
        defined = c_functions(path)
        wanted = {n for n, static in defined.items() if scope == 'all' or not static}
        audited += 1
        if len(wanted) != count:
            problems.append('%s: coverage table says %d functions (%s), the file defines %d'
                            % (rel, count, scope, len(wanted)))
        in_rows = {name for (f, name) in listed if f == rel}
        for name in sorted(wanted - in_rows):
            problems.append('%s:%s is defined but not in any row' % (rel, name))
        for name in sorted(in_rows - wanted):
            problems.append('%s:%s is in a row but outside the %s scope of the coverage table'
                            % (rel, name, scope))
    stray = sorted({f for (f, _n) in listed} - set(coverage))
    problems += ['%s: used by a row but absent from the coverage table' % f for f in stray]
    assert audited > 0, 'no covered file was audited'
    assert not problems, 'map and covered files disagree:\n  ' + '\n  '.join(problems)


def test_statistics_match_the_table(doc):
    rows, _, stats, _ = doc
    computed = compute_statistics(rows)
    assert set(stats) == set(computed), (sorted(stats), sorted(computed))
    wrong = ['%s: header %s, table %s' % (k, stats[k], computed[k])
             for k in computed if stats[k] != computed[k]]
    assert not wrong, ('statistics block of the document is stale '
                       '(columns: total, mapped, mirror, partial, C-only, undetermined, '
                       'Python-only symbols, inferred; recompute with --stats):\n  '
                       + '\n  '.join(wrong))


def test_cited_gap_ids_are_defined_in_the_glossary(doc):
    rows, _, _, glossary = doc
    assert glossary, 'the gap glossary region is empty'
    missing = sorted({(r.id, g) for r in rows for g in r.gaps if g not in glossary})
    assert not missing, 'gap ids cited by rows but not defined in the glossary: %s' % missing


def test_cited_parity_tool_paths_exist(doc):
    rows, _, _, _ = doc
    missing = []
    checked = 0
    for row in rows:
        for token in _PATH_TOKEN_RE.findall(row.tools_text):
            checked += 1
            if token.startswith('audiolib/common/'):
                if not COMMON_DIR.is_dir():
                    continue
                path = COMMON_DIR / token[len('audiolib/common/'):]
            elif token.startswith('audiolib/'):
                if not AUDIOLIB_DIR.is_dir():
                    continue
                path = AUDIOLIB_DIR / token[len('audiolib/'):]
            else:
                path = ROOT / token
            if not path.exists():
                missing.append('%s: %s' % (row.id, token))
    assert checked > 0, 'no parity-tool path is cited by any row'
    assert not missing, 'parity tools cited by the map do not exist:\n  ' + '\n  '.join(missing)


def test_symbol_resolver_distinguishes_definitions_from_imports(tmp_path):
    """The resolver must be able to say no: a renamed def, a missing member and
    a merely imported name are all refused; a method, a class-body field and a
    guarded definition are found."""
    source = tmp_path / 'mod.py'
    source.write_text(
        'import os\n'
        'from collections import OrderedDict\n'
        'try:\n'
        '    from fast import helper\n'
        'except ImportError:\n'
        '    def helper():\n'
        '        return 1\n'
        'LIMIT = 3\n'
        'def top(): pass\n'
        'class Box:\n'
        '    size: int = 0\n'
        '    def grow(self): pass\n',
        encoding='utf-8')
    found = ('top', 'LIMIT', 'Box', 'Box.grow', 'Box.size', 'helper')
    refused = ('top_renamed', 'Box.shrink', 'Box.grow.inner', 'os', 'OrderedDict',
               'LIMIT.x', 'missing')
    assert [name for name in found if not python_symbol_exists(source, name)] == []
    assert [name for name in refused if python_symbol_exists(source, name)] == []


def test_parser_rejects_a_malformed_row():
    """The grammar must be able to fail: a Python cell with free text, a
    relation outside the vocabulary, and a C-only row naming a Python symbol
    are all refused."""
    good = ['PL-X-01', '`pipelines/a.c: f`', '`pipelines/a.py:g`', 'mirror', '', '', '']
    Row(good)
    for bad in (
        good[:2] + ['`pipelines/a.py:g` and more'] + good[3:],
        good[:3] + ['same'] + good[4:],
        good[:3] + ['C-only'] + good[4:],
        good[:2] + [NONE_CELL] + good[3:],
        ['pl-x-1'] + good[1:],
    ):
        with pytest.raises(MapError):
            Row(bad)


def compute_statistics(rows):
    """``{area: [total, mapped, mirror, partial, C-only, undetermined,
    Python-only symbols, inferred]}`` including a ``TOTAL`` entry."""
    result = {}
    for area in sorted({r.area for r in rows}) + ['TOTAL']:
        sel = [r for r in rows if area == 'TOTAL' or r.area == area]

        def count(*relations):
            return sum(len(r.c) for r in sel if r.relation in relations)

        result[area] = [
            count(*RELATIONS),
            count('mirror', 'partial'), count('mirror'), count('partial'),
            count('C-only'), count('未能判斷'),
            sum(len(r.py) for r in sel if r.relation == 'Python-only'),
            sum(len(r.c) for r in sel if r.relation in ('mirror', 'partial') and r.inferred),
        ]
    return result


if __name__ == '__main__':          # maintenance helper: recompute the statistics rows
    import sys
    if '--stats' not in sys.argv:
        sys.exit('usage: python3 pipelines/tests/test_c_python_map.py --stats')
    for key, numbers in compute_statistics(_load()[0]).items():
        print('| `%s` | %s |' % (key, ' | '.join(str(n) for n in numbers)))
