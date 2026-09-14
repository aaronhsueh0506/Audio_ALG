"""The Python comfort-noise generator must be the C generator, not a look-alike."""
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.cng_aec3 import LUT, SEED, Aec3ComfortNoise, lcg_step  # noqa: E402


def _c_lut_literals():
    header = (ROOT / 'lib' / 'aec' / 'c_impl' / 'include'
              / 'aec3_balanced_config.h').read_text(encoding='utf-8')
    match = re.search(r'AEC3B_SQRT2_SIN_LUT\[32\]\s*=\s*\{([^}]*)\}', header)
    assert match, 'AEC3B_SQRT2_SIN_LUT not found in the AEC header'
    values = [float(v.strip().rstrip('f')) for v in match.group(1).split(',')]
    return np.array(values, dtype=np.float32)


def _c_seed():
    source = (ROOT / 'pipelines' / 'mono_aec_nr_res' / 'audio_pipeline.c'
              ).read_text(encoding='utf-8')
    match = re.search(r'#define\s+AUDIO_PIPELINE_RNG_SEED\s+0x([0-9a-fA-F]+)u', source)
    assert match, 'AUDIO_PIPELINE_RNG_SEED not found'
    return int(match.group(1), 16)


def test_lut_and_seed_are_the_c_literals():
    assert np.array_equal(LUT, _c_lut_literals())
    assert SEED == _c_seed()


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
