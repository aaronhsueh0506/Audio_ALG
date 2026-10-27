"""The Python comfort-noise generator must be the C generator, not a look-alike."""
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.common.cng_aec3 import LUT, SEED, Aec3ComfortNoise, lcg_step  # noqa: E402


def _c_lut_literals():
    header = (ROOT / 'lib' / 'aec' / 'c_impl' / 'include'
              / 'aec_balanced_config.h').read_text(encoding='utf-8')
    match = re.search(r'AECB_SQRT2_SIN_LUT\[32\]\s*=\s*\{([^}]*)\}', header)
    assert match, 'AECB_SQRT2_SIN_LUT not found in the AEC header'
    values = [float(v.strip().rstrip('f')) for v in match.group(1).split(',')]
    return np.array(values, dtype=np.float32)


# Every C file that starts a comfort-noise generator defines the seed locally; each must define it once.
C_SEED_SITES = [
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
C_SEED = re.compile(r'^#define\s+\w*CNG_RNG_SEED\s+0x([0-9a-fA-F]+)u\s*$', re.MULTILINE)


def _c_seeds():
    seeds = {}
    for rel in C_SEED_SITES:
        found = C_SEED.findall((ROOT / 'pipelines' / rel).read_text(encoding='utf-8'))
        assert len(found) == 1, f'{rel}: expected one CNG_RNG_SEED definition, found {len(found)}'
        seeds[rel] = int(found[0], 16)
    for path in sorted((ROOT / 'pipelines').rglob('*.[ch]')):
        if {'obj', 'bin'} & set(path.parts):
            continue
        for value in C_SEED.findall(path.read_text(encoding='utf-8', errors='replace')):
            seeds.setdefault(str(path), int(value, 16))
    return seeds


def test_lut_and_seed_are_the_c_literals():
    assert np.array_equal(LUT, _c_lut_literals())
    seeds = _c_seeds()
    assert set(seeds.values()) == {SEED}, seeds


def test_vectorised_draw_matches_the_scalar_recurrence():
    gen = Aec3ComfortNoise()
    state = SEED
    expected = []
    for _ in range(3 * 511 + 7):
        state = lcg_step(state)
        expected.append(state >> 26)
    got = np.concatenate([gen.draw(511), gen.draw(511), gen.draw(511), gen.draw(7)])
    assert got.tolist() == expected
    assert gen.state == state


def test_add_touches_only_interior_bins_with_quarter_turn_imag():
    gen = Aec3ComfortNoise()
    n = 513
    spec = np.zeros(n, dtype=np.complex64)
    amplitude = np.full(n, 0.5, dtype=np.float32)
    gen.add(spec, amplitude)
    assert spec[0] == 0 and spec[n - 1] == 0
    check = Aec3ComfortNoise()
    ix = check.draw(n - 2)
    assert np.array_equal(spec[1:n - 1].real, (np.float32(0.5) * LUT[ix]).astype(np.float32))
    assert np.array_equal(spec[1:n - 1].imag,
                          (np.float32(0.5) * LUT[(ix + 8) & 31]).astype(np.float32))


def test_reset_replays_the_same_sequence():
    gen = Aec3ComfortNoise()
    first = gen.draw(100)
    gen.reset()
    assert np.array_equal(gen.draw(100), first)
