"""The comfort-noise scaling floor is one value everywhere.

Each C program, test and oracle that fills comfort noise defines the floor
locally (``*CNG_NR_GAIN_FLOOR``), the Python twin mirrors it, and the two C
witnesses pin the contract with a literal of their own. Nothing in the build
ties these together, so this reads every site straight from the source text
and refuses a retune that reaches only some of them.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]

C_DEFINE = re.compile(r'^#define\s+\w*CNG_NR_GAIN_FLOOR\s+([0-9.]+)f\s*$', re.MULTILINE)

# Files that must each define the floor exactly once (a site that disappears fails the test).
C_SITES = [
    '4ch_aec_bf_nr_res/mcab_sample_post.h',
    '4ch_aec_bf_nr_res/tests/conv_chain_mcab.h',
    'mono_aec_nr_res/main.c',
    'mono_aec_nr_res/example_board_adapter.c',
    'mono_aec_nr_res/tests/test_mono_chain.c',
    'mono_aec_nr_res/tests/test_no_stdio_stack.c',
    'mono_aec_nr_res/tests/conv_chain_mono.h',
    'dfn/mono_aec_dfn_res/main.c',
    'dfn/mono_aec_dfn_res/tests/test_mono_dfn.c',
    'tests/test_readme_flow.c',
    'tests/pyc_parity/pyc_drv.c',
]

OTHER_SITES = {
    'python twin': (ROOT / 'mono_aec_nr_res' / 'pipeline.py',
                    r'^CNG_NR_GAIN_FLOOR\s*=\s*([0-9.]+)\s*$'),
    'mono witness': (ROOT / 'mono_aec_nr_res' / 'tests' / 'test_mono_chain.c',
                     r'const double floor_amp = ([0-9.]+),'),
    '4ch witness': (ROOT / '4ch_aec_bf_nr_res' / 'tests' / 'test_mcab.c',
                    r'const double floor_amp = ([0-9.]+),'),
}


def _declared(path, pattern):
    matches = re.findall(pattern, path.read_text(encoding='utf-8'), re.MULTILINE)
    assert len(matches) == 1, f'{path.name}: expected one CNG_NR_GAIN_FLOOR site, found {len(matches)}'
    return matches[0]


def _c_values():
    values = {}
    for rel in C_SITES:
        values[rel] = _declared(ROOT / rel, C_DEFINE.pattern)
    for path in sorted(ROOT.rglob('*.[ch]')):
        if {'obj', 'bin'} & set(path.relative_to(ROOT).parts):
            continue
        for value in C_DEFINE.findall(path.read_text(encoding='utf-8', errors='replace')):
            values.setdefault(str(path.relative_to(ROOT)), value)
            assert value == values[str(path.relative_to(ROOT))], path
    return values


def test_cng_floor_is_one_value_everywhere():
    values = _c_values()
    values.update({name: _declared(path, pattern) for name, (path, pattern) in OTHER_SITES.items()})
    assert len(set(values.values())) == 1, values


def test_cng_floor_is_minus_ten_db():
    value = float(_declared(ROOT / '4ch_aec_bf_nr_res' / 'mcab_sample_post.h', C_DEFINE.pattern))
    assert abs(value - 10.0 ** (-10.0 / 20.0)) < 1e-7
