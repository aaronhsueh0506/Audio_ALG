"""The Python comfort-noise generator must be the C generator, not a look-alike."""
import os
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.common.cng_aec3 import LUT, SEED, Aec3ComfortNoise, lcg_step  # noqa: E402


def _aec_include():
    """The AEC include directory as lib/deps.mk resolves it: AEC_DIR from the environment, else the
    sibling AEC checkout's c_impl."""
    repo = pathlib.Path(os.environ.get('AEC_REPO_DIR') or ROOT.parent / 'AEC')
    return pathlib.Path(os.environ.get('AEC_DIR') or repo / 'c_impl') / 'include'


def _c_lut_literals():
    header = (_aec_include() / 'aec_balanced_config.h').read_text(encoding='utf-8')
    match = re.search(r'AECB_SQRT2_SIN_LUT\[32\]\s*=\s*\{([^}]*)\}', header)
    assert match, 'AECB_SQRT2_SIN_LUT not found in the AEC header'
    values = [float(v.strip().rstrip('f')) for v in match.group(1).split(',')]
    return np.array(values, dtype=np.float32)


# The library owns the generator behind aec_apply_cng and starts it at AEC_CNG_SEED (aec_post.h) after
# every hard post-chain reset, so the examples hold no seed. The C files below keep a generator of their
# own (an oracle chain, a replay driver, a planted-defect rig) and drive it through the explicit-array forms
# aec_apply_cng_spec / aec_cng_fill; each must start it from AEC_CNG_SEED and define no seed of its own.
C_OWN_GENERATOR_SITES = [
    '4ch/tests/conv_chain_4ch.h',
    'mono/aec_nr_res/tests/conv_chain_mono.h',
    'mono/aec_nr_res/tests/test_mono_chain.c',
    'mono/aec_dfn_res/tests/test_mono_dfn.c',
    'tests/pyc_parity/pyc_drv.c',
]
# Programs that reach the generator only through the library's aec_apply_cng.
C_LIBRARY_OWNED_SITES = [
    '4ch/array_sample_post.h',
    '4ch/array_sample_unit.h',
    '4ch/4ch_raw.c',
    '4ch/4ch_static.c',
    '4ch/4ch_external_static.c',
    'mono/aec_nr_res/main.c',
    'mono/aec_nr_res/example_board_adapter.c',
    'mono/aec_nr_res/tests/test_no_stdio_stack.c',
    'mono/aec_dfn_res/main.c',
    'tests/test_readme_flow.c',
]
# A call that hands the generator state in by address: aec_apply_cng_spec(&x->rng, ...) / aec_cng_fill(&rng, ...).
C_EXPLICIT_GENERATOR_CALL = re.compile(r'\b(?:aec_apply_cng_spec|aec_cng_fill)\s*\(\s*&')
C_GENERATOR_STATE = re.compile(r'\brng\b')
# A reseed of the generator state kept in a struct field: `c->rng = <seed>;`.
C_GENERATOR_FIELD_ASSIGN = re.compile(r'(?:->|\.)rng\s*=(?!=)\s*([^;]+);')
C_SEED_NAMES = {'AEC_CNG_SEED', 'PYC_CNG_RNG_SEED', 'RETIRED_CNG_SEED'}
# A private numeric seed (any `#define ...CNG_RNG_SEED <number>`) anywhere; the retired 0x9e3779b9 literal outside tests.
C_PRIVATE_SEED = re.compile(r'^#define\s+\w*CNG_RNG_SEED\s+(?:0x[0-9a-fA-F]+|\d+)u?\b', re.MULTILINE)
RETIRED_SEED = re.compile(r'0x9e3779b9', re.IGNORECASE)
LIB_SEED = re.compile(r'^#define\s+AEC_CNG_SEED\s+(\d+)u\s*$', re.MULTILINE)


def _lib_seed():
    header = (_aec_include() / 'aec_post.h').read_text(encoding='utf-8')
    found = LIB_SEED.findall(header)
    assert len(found) == 1, 'AEC_CNG_SEED must be defined once in aec_post.h'
    return int(found[0])


def test_lut_and_seed_are_the_c_literals():
    assert np.array_equal(LUT, _c_lut_literals())
    assert SEED == _lib_seed()


def _pipeline_c_files():
    for path in sorted((ROOT / 'pipelines').rglob('*.[ch]')):
        if not {'obj', 'bin'} & set(path.parts):
            yield path


def _code(text):
    """C source without comments."""
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.DOTALL)


def test_own_generator_sites_start_from_the_library_seed():
    """Every reseed of a site's generator state is AEC_CNG_SEED (or the planted retired seed in the
    defect rig of test_mono_chain.c); a numeric literal there is a private seed."""
    for rel in C_OWN_GENERATOR_SITES:
        code = _code((ROOT / 'pipelines' / rel).read_text(encoding='utf-8'))
        assert C_EXPLICIT_GENERATOR_CALL.search(code), f'{rel}: no longer drives a generator of its own; update the list'
        assigned = C_GENERATOR_FIELD_ASSIGN.findall(code)
        assert assigned, f'{rel}: no reseed of the generator state found'
        for rhs in assigned:
            idents = set(re.findall(r'[A-Za-z_]\w*', rhs))
            assert idents & C_SEED_NAMES, f'{rel}: seeds the generator with {rhs!r}, not AEC_CNG_SEED'
            assert not re.search(r'\b(?:0[xX][0-9a-fA-F]+|\d+)[uU]?\b', rhs), f'{rel}: numeric seed in {rhs!r}'
            if 'RETIRED_CNG_SEED' in idents:
                assert rel == 'mono/aec_nr_res/tests/test_mono_chain.c', f'{rel}: plants the retired seed'
    drv = (ROOT / 'pipelines' / 'tests' / 'pyc_parity' / 'pyc_drv.c').read_text(encoding='utf-8')
    assert re.search(r'^#define\s+PYC_CNG_RNG_SEED\s+AEC_CNG_SEED\s*$', drv, re.MULTILINE), \
        'pyc_drv.c: PYC_CNG_RNG_SEED must be an alias of AEC_CNG_SEED'


def test_every_explicit_generator_is_listed():
    """A new file that drives its own generator must be listed, which puts it under the seed check."""
    found = set()
    for path in _pipeline_c_files():
        text = _code(path.read_text(encoding='utf-8', errors='replace'))
        if C_EXPLICIT_GENERATOR_CALL.search(text):
            found.add(path.relative_to(ROOT / 'pipelines').as_posix())
    assert found == set(C_OWN_GENERATOR_SITES), sorted(found ^ set(C_OWN_GENERATOR_SITES))


def test_library_owned_sites_keep_no_generator():
    for rel in C_LIBRARY_OWNED_SITES:
        text = (ROOT / 'pipelines' / rel).read_text(encoding='utf-8')
        code = _code(text)
        assert not C_EXPLICIT_GENERATOR_CALL.search(code), f'{rel}: draws comfort noise with a generator of its own'
        assert not C_GENERATOR_STATE.search(code), f'{rel}: keeps comfort-noise state; the library owns it'
        assert 'AEC_CNG_SEED' not in code, f'{rel}: seeds a generator; the library owns it'


def test_no_c_site_keeps_a_private_seed():
    for path in _pipeline_c_files():
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
