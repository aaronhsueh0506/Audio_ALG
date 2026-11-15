"""The comfort-noise scaling floor is one value everywhere.

The AEC library owns the floor (``AEC_CNG_NR_GAIN_FLOOR`` in aec.h) and applies it inside
``aec_apply_cng``; the example programs reach it through that call and keep no copy. What remains
outside the library are the literals that only exist to be compared with it: the oracle chains and
the parity driver of the C tests, the Python twin, and the two witnesses that measure the scaling.
Nothing in the build ties those to the header, so this reads every site straight from the source
text and refuses a retune that reaches only some of them.
"""
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]      # Audio_ALG/pipelines


def _aec_include():
    """The AEC include directory as lib/deps.mk resolves it: AEC_DIR from the environment, else the
    sibling AEC checkout's c_impl."""
    repo = pathlib.Path(os.environ.get('AEC_REPO_DIR') or ROOT.parents[1] / 'AEC')
    return pathlib.Path(os.environ.get('AEC_DIR') or repo / 'c_impl') / 'include'


C_DEFINE = re.compile(r'^#define\s+(\w*CNG_NR_GAIN_FLOOR)\s+([0-9.]+)f\b', re.MULTILINE)
LIB_DEFINE = re.compile(r'^#define\s+AEC_CNG_NR_GAIN_FLOOR\s+([0-9.]+)f\b', re.MULTILINE)

# Oracle chains and the parity driver: each keeps one literal of its own, so a site that disappears
# fails the test instead of being skipped.
LOCAL_FLOOR_SITES = {
    '4ch/tests/conv_chain_4ch.h': 'CONV_4CH_CNG_NR_GAIN_FLOOR',
    'mono/aec_nr_res/tests/conv_chain_mono.h': 'CONV_MONO_CNG_NR_GAIN_FLOOR',
    'tests/pyc_parity/pyc_drv.c': 'PYC_CNG_NR_GAIN_FLOOR',
}

# Programs that draw comfort noise through the library's aec_apply_cng and so must not define a floor.
LIBRARY_OWNED_SITES = [
    '4ch/array_sample_post.h',
    '4ch/array_sample_unit.h',
    'mono/aec_nr_res/main.c',
    'mono/aec_nr_res/example_board_adapter.c',
    'mono/aec_nr_res/tests/test_no_stdio_stack.c',
    'mono/aec_dfn_res/main.c',
    'tests/test_readme_flow.c',
]

OTHER_SITES = {
    'python twin': (ROOT / 'mono' / 'aec_nr_res' / 'pipeline.py',
                    r'^CNG_NR_GAIN_FLOOR\s*=\s*([0-9.]+)\s*$'),
    'mono witness': (ROOT / 'mono' / 'aec_nr_res' / 'tests' / 'test_mono_chain.c',
                     r'const double floor_amp = ([0-9.]+),'),
    '4ch witness': (ROOT / '4ch' / 'tests' / 'test_4ch_chain.c',
                    r'const double floor_amp = ([0-9.]+),'),
}


def _declared(path, pattern):
    matches = re.findall(pattern, path.read_text(encoding='utf-8'), re.MULTILINE)
    assert len(matches) == 1, f'{path.name}: expected one CNG_NR_GAIN_FLOOR site, found {len(matches)}'
    return matches[0]


def _library_floor():
    return _declared(_aec_include() / 'aec.h', LIB_DEFINE.pattern)


def _local_defines():
    """Every ``*CNG_NR_GAIN_FLOOR`` literal defined in a pipelines C file: {relative path: {name: value}}."""
    found = {}
    for path in sorted(ROOT.rglob('*.[ch]')):
        rel = path.relative_to(ROOT)
        if {'obj', 'bin'} & set(rel.parts):
            continue
        defines = C_DEFINE.findall(path.read_text(encoding='utf-8', errors='replace'))
        if defines:
            found[rel.as_posix()] = dict(defines)
    return found


def test_cng_floor_is_one_value_everywhere():
    lib = _library_floor()
    values = {'aec.h': lib}
    defined = _local_defines()
    for rel, name in LOCAL_FLOOR_SITES.items():
        assert rel in defined and name in defined[rel], f'{rel}: {name} is not defined'
        values[rel] = defined[rel][name]
    values.update({name: _declared(path, pattern) for name, (path, pattern) in OTHER_SITES.items()})
    assert len(set(values.values())) == 1, values


def test_no_pipeline_file_keeps_an_unlisted_floor():
    """A new copy of the floor has to be added to LOCAL_FLOOR_SITES (and so compared), not slip in."""
    defined = _local_defines()
    unlisted = {rel: names for rel, names in defined.items()
                if set(names) - {LOCAL_FLOOR_SITES.get(rel)}}
    assert not unlisted, f'floor literals outside LOCAL_FLOOR_SITES: {unlisted}'
    for rel in LIBRARY_OWNED_SITES:
        assert rel not in defined, f'{rel}: the library owns the floor (aec_apply_cng), no local define'


def test_cng_floor_is_minus_ten_db():
    value = float(_library_floor())
    assert abs(value - 10.0 ** (-10.0 / 20.0)) < 1e-7
