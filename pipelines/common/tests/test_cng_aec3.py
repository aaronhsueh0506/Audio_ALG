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


# Every C file that starts a comfort-noise generator takes the seed from the AEC library (AEC_CNG_SEED,
# aec_post.h) -- directly or through a one-line alias -- and defines no seed of its own.
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
# A private numeric seed (any `#define ...CNG_RNG_SEED <number>`) anywhere; the retired 0x9e3779b9 literal outside tests.
C_PRIVATE_SEED = re.compile(r'^#define\s+\w*CNG_RNG_SEED\s+(?:0x[0-9a-fA-F]+|\d+)u?\b', re.MULTILINE)
RETIRED_SEED = re.compile(r'0x9e3779b9', re.IGNORECASE)
LIB_SEED = re.compile(r'^#define\s+AEC_CNG_SEED\s+(\d+)u\s*$', re.MULTILINE)


def _lib_seed():
    header = (ROOT / 'lib' / 'aec' / 'c_impl' / 'include' / 'aec_post.h').read_text(encoding='utf-8')
    found = LIB_SEED.findall(header)
    assert len(found) == 1, 'AEC_CNG_SEED must be defined once in aec_post.h'
    return int(found[0])


def test_lut_and_seed_are_the_c_literals():
    assert np.array_equal(LUT, _c_lut_literals())
    assert SEED == _lib_seed()


def test_no_c_site_keeps_a_private_seed():
    for rel in C_SEED_SITES:
        text = (ROOT / 'pipelines' / rel).read_text(encoding='utf-8')
        assert 'AEC_CNG_SEED' in text or 'MCAB_SAMPLE_CNG_RNG_SEED' in text, f'{rel}: does not start from AEC_CNG_SEED'
    for path in sorted((ROOT / 'pipelines').rglob('*.[ch]')):
        if {'obj', 'bin'} & set(path.parts):
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        assert not C_PRIVATE_SEED.search(text), f'{path}: defines a private comfort-noise seed'
        if 'tests' not in path.parts:       # tests plant the retired seed as a negative control
            assert not RETIRED_SEED.search(text), f'{path}: still carries the retired 0x9e3779b9 seed'


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


def test_fill_is_the_c_helper_arithmetic():
    """fill() is aec_cng_fill: interior bins only, draws in bin order, sqrt(1-g^2) * (sqrt(cn) * lut)."""
    n = 513
    rs = np.random.RandomState(3)
    cn = (np.abs(rs.randn(n)) * 1e5).astype(np.float32)
    g = rs.rand(n).astype(np.float32)
    gen = Aec3ComfortNoise()
    spec = np.zeros(n, dtype=np.complex64)
    gen.fill(spec, cn, g)
    assert spec[0] == 0 and spec[n - 1] == 0
    ix = Aec3ComfortNoise().draw(n - 2)
    nf = np.sqrt(np.maximum(cn[1:n - 1] / np.float32(32768.0 ** 2), 0.0)).astype(np.float32)
    ng = np.sqrt(np.maximum(np.float32(1.0) - g[1:n - 1] * g[1:n - 1], 0.0)).astype(np.float32)
    assert np.array_equal(spec[1:n - 1].real, (ng * (nf * LUT[ix])).astype(np.float32))
    assert np.array_equal(spec[1:n - 1].imag, (ng * (nf * LUT[(ix + 8) & 31])).astype(np.float32))
    # a denoiser gain scales the amplitude by max(g_nr, floor)
    spec2 = np.zeros(n, dtype=np.complex64)
    gnr = np.full(n, 0.1, np.float32)
    Aec3ComfortNoise().fill(spec2, cn, g, gnr, 0.25)
    ng_nr = (ng * np.float32(0.25)).astype(np.float32)
    assert np.array_equal(spec2.real[1:n - 1], (ng_nr * (nf * LUT[ix])).astype(np.float32))


def test_reset_replays_the_same_sequence():
    gen = Aec3ComfortNoise()
    first = gen.draw(100)
    gen.reset()
    assert np.array_equal(gen.draw(100), first)
