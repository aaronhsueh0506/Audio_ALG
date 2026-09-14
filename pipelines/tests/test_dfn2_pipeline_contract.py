"""End-to-end contract of the mono DFN2 reference against the conventional twin.

With the identity model the DFN2 pipeline must be exactly the conventional
pipeline with NR disabled: S1 = E * G_res, comfort noise off, same WOLA.  The
scenario drives the AEC into real suppression (a far-end echo through a short
room response) and asserts so, because with G_res == 1 everywhere S1 == E and
an estimate/apply swap would be invisible.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AEC_PY = ROOT / 'lib' / 'aec' / 'python'
if str(AEC_PY) not in sys.path:
    sys.path.insert(0, str(AEC_PY))

pytest.importorskip('torch')

from lib.aec.python.aec import AecConfig, AecMode, AecPreset  # noqa: E402
from pipelines.aec_dfn2_res_pipeline import make_stage, run_dfn2_res  # noqa: E402
from pipelines.aec_nr_pipeline import run_aec_linear, run_res  # noqa: E402
from pipelines.dfn2_stage import HOP, MODEL_LOOKAHEAD, load_dfn2  # noqa: E402

SR = 48000
SECONDS = 2.5


def _scenario(seed=0):
    rng = np.random.RandomState(seed)
    n = int(SR * SECONDS)
    far = (rng.randn(n) * 0.2).astype(np.float32)
    far *= (np.sin(2 * np.pi * 3.0 * np.arange(n) / SR) > 0).astype(np.float32)
    rir = np.zeros(2400, np.float32)
    rir[480] = 0.6
    rir[900] = -0.3
    rir[1500:1600] = rng.randn(100) * 0.02
    echo = np.convolve(far, rir)[:n].astype(np.float32)
    near = (rng.randn(n) * 0.01).astype(np.float32)
    mic = (echo + near).astype(np.float32)
    return mic, far


@pytest.fixture(scope='module')
def linear():
    mic, ref = _scenario()
    config = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=SR, frame_size=2 * HOP, hop_size=HOP,
        mode=AecMode.PBFDKF, mu=0.3, enable_res=True, enable_cng=False)
    aec_out, contexts = run_aec_linear(mic, ref, config)
    return mic, config, aec_out, contexts


@pytest.fixture(scope='module')
def assets():
    return load_dfn2(seed=0)


def test_scenario_actually_suppresses(linear):
    _mic, _config, _aec_out, contexts = linear
    floors = np.array([np.min(c.res_gain) for c in contexts])
    assert floors.min() < 0.9, 'G_res never cut: the identity gate would be vacuous'


def test_identity_model_equals_conventional_without_nr(linear, assets):
    mic, config, aec_out, contexts = linear
    n_frames = len(contexts)
    tail = np.zeros(len(mic), np.float32)
    tail[:len(aec_out)] = aec_out
    conventional = run_res(tail, np.ones((n_frames, HOP + 1), np.float32), contexts,
                           config, use_nr=False, use_res=True, combine='min')
    stage = make_stage(assets, identity=True)
    dfn = run_dfn2_res(contexts, config, stage, enable_cng=False,
                       output_length=len(mic))
    span = n_frames * HOP
    assert np.array_equal(dfn[:span], conventional[:span])


def test_realtime_timing_is_a_pure_two_hop_delay(linear, assets):
    mic, config, _aec_out, contexts = linear
    stage = make_stage(assets, identity=True)
    aligned = run_dfn2_res(contexts, config, stage, enable_cng=False,
                           output_length=len(mic))
    delayed = run_dfn2_res(contexts, config, stage, enable_cng=False,
                           realtime_timing=True, output_length=len(mic))
    shift = MODEL_LOOKAHEAD * HOP
    span = len(contexts) * HOP
    assert np.array_equal(delayed[:shift], np.zeros(shift, np.float32))
    assert np.array_equal(delayed[shift:shift + span], aligned[:span])


def test_estimate_source_changes_output_only_through_the_network(linear, assets):
    mic, config, _aec_out, contexts = linear
    outputs = {}
    for source in ('pre_res', 'post_res', 'mic'):
        stage = make_stage(assets)
        outputs[source] = run_dfn2_res(contexts, config, stage, estimate_source=source,
                                       enable_cng=False, output_length=len(mic))
    assert not np.array_equal(outputs['pre_res'], outputs['post_res'])
    assert not np.array_equal(outputs['pre_res'], outputs['mic'])
    for out in outputs.values():
        assert np.all(np.isfinite(out))
