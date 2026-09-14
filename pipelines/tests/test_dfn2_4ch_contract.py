"""The 4-channel DFN2 tail against the conventional tail on the same fused
contexts: with the identity model the two are the same spectrum-to-hop path
minus the NR gain, so they must agree exactly (comfort noise off)."""
import importlib.util
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


def _load_twin():
    path = ROOT / 'pipelines' / '4ch_aec_bf_dfn_res' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location('four_channel_dfn_twin', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['four_channel_dfn_twin'] = module
    spec.loader.exec_module(module)
    return module


twin = _load_twin()

from pipelines.aec_dfn2_res_pipeline import make_stage  # noqa: E402
from pipelines.aec_nr_pipeline import run_res  # noqa: E402
from pipelines.dfn2_stage import HOP, load_dfn2  # noqa: E402

SR = 48000
SECONDS = 1.5


def _scenario(seed=0):
    rng = np.random.RandomState(seed)
    n = int(SR * SECONDS)
    far = (rng.randn(n) * 0.2).astype(np.float32)
    far *= (np.sin(2 * np.pi * 2.0 * np.arange(n) / SR) > 0).astype(np.float32)
    mics = np.zeros((n, 4), np.float32)
    for ch in range(4):
        rir = np.zeros(2000, np.float32)
        rir[400 + 37 * ch] = 0.5
        rir[1200 + 11 * ch] = -0.2
        echo = np.convolve(far, rir)[:n]
        mics[:, ch] = echo + rng.randn(n) * 0.01
    return mics, far


@pytest.fixture(scope='module')
def fused():
    mics, far = _scenario()
    config = twin.FourChannelAecConfig(sample_rate=SR)
    return twin.run_front_end(mics, far, config)


def test_front_end_produces_fused_contexts_with_real_suppression(fused):
    assert len(fused) > 20
    floors = np.array([np.min(c.res_gain) for c in fused])
    assert floors.min() < 0.95


def test_identity_dfn2_tail_equals_conventional_tail_without_nr(fused):
    stage = make_stage(load_dfn2(seed=0), identity=True)
    dfn = twin.post_dfn2(fused, SR, stage, enable_cng=False)
    tail = np.zeros(len(fused) * HOP, np.float32)
    conventional = run_res(tail, np.ones((len(fused), HOP + 1), np.float32), fused,
                           twin._post_config(SR, HOP, enable_cng=False),
                           use_nr=False, use_res=True, combine='min')
    assert np.array_equal(dfn, conventional)


def test_conventional_tail_runs(fused):
    out = twin.post_conventional(fused, SR, enable_cng=False)
    assert out.shape == (len(fused) * HOP,)
    assert np.all(np.isfinite(out))
