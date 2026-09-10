"""The comfort-noise scaling floor is one value in five places.

Each pipeline keeps its own copy of ``CNG_NR_GAIN_FLOOR`` (the pipelines are
deliberately self-contained), the Python twin mirrors it, and the two C
witnesses pin the contract with a literal of their own. Nothing in the build
ties them together, so this reads all five straight from the source text and
refuses a retune that reaches only some of them.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

SITES = {
    'mono pipeline': (ROOT / 'mono_aec_nr_res' / 'audio_pipeline.c',
                      r'^#define\s+CNG_NR_GAIN_FLOOR\s+([0-9.]+)f\s*$'),
    '4ch core': (ROOT / '4ch_aec_bf_nr_res' / '4aec_nr_res.c',
                 r'^#define\s+CNG_NR_GAIN_FLOOR\s+([0-9.]+)f\s*$'),
    'python twin': (ROOT / 'aec_nr_pipeline.py',
                    r'^CNG_NR_GAIN_FLOOR\s*=\s*([0-9.]+)\s*$'),
    'mono witness': (ROOT / 'mono_aec_nr_res' / 'tests' / 'test_audio_pipeline.c',
                     r'const double floor_amp = ([0-9.]+),'),
    '4ch witness': (ROOT / '4ch_aec_bf_nr_res' / 'tests' / 'test_4aec_nr_res.c',
                    r'const double floor_amp = ([0-9.]+),'),
}


def _declared(path, pattern):
    matches = re.findall(pattern, path.read_text(encoding='utf-8'), re.MULTILINE)
    assert len(matches) == 1, f'{path.name}: expected one CNG_NR_GAIN_FLOOR site, found {len(matches)}'
    return matches[0]


def test_cng_floor_is_one_value_everywhere():
    values = {name: _declared(path, pattern) for name, (path, pattern) in SITES.items()}
    assert len(set(values.values())) == 1, values


def test_cng_floor_is_minus_ten_db():
    value = float(_declared(*SITES['mono pipeline']))
    assert abs(value - 10.0 ** (-10.0 / 20.0)) < 1e-7
