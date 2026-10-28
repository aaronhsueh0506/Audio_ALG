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

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AEC_PY = ROOT / 'lib' / 'aec' / 'python'
if str(AEC_PY) not in sys.path:
    sys.path.insert(0, str(AEC_PY))

pytest.importorskip('torch')

from lib.aec.python.aec import AecConfig, AecMode, AecPreset  # noqa: E402
from pipelines.dfn.mono_aec_dfn_res.pipeline import make_stage, run_dfn2_res  # noqa: E402
from pipelines.mono_aec_nr_res.pipeline import run_aec_linear, run_res  # noqa: E402
from pipelines.dfn.dfn2_stage import HOP, load_dfn2  # noqa: E402
from pipelines.dfn.tests.dfn2_pairs import PAIRS, SHIPPED, pair_config, pair_id  # noqa: E402

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


@pytest.fixture(scope='module', params=PAIRS, ids=pair_id)
def pair_assets(request, tmp_path_factory):
    if request.param == SHIPPED:
        return load_dfn2(seed=0)
    return load_dfn2(config_path=pair_config(tmp_path_factory.mktemp('dfn2_pair'), request.param),
                     seed=0)


def test_scenario_actually_suppresses(linear):
    _mic, _config, _aec_out, contexts = linear
    floors = np.array([np.min(c.res_gain) for c in contexts])
    assert floors.min() < 0.9, 'G_res never cut: the identity gate would be vacuous'


def test_identity_model_equals_conventional_without_nr(linear, pair_assets):
    mic, config, aec_out, contexts = linear
    n_frames = len(contexts)
    tail = np.zeros(len(mic), np.float32)
    tail[:len(aec_out)] = aec_out
    conventional = run_res(tail, np.ones((n_frames, HOP + 1), np.float32), contexts,
                           config, use_nr=False, use_res=True, combine='min')
    stage = make_stage(pair_assets, identity=True)
    dfn = run_dfn2_res(contexts, config, stage, enable_cng=False,
                       output_length=len(mic))
    span = n_frames * HOP
    assert np.array_equal(dfn[:span], conventional[:span])


def _events(contexts):
    serials = [c.post_reset_serial for c in contexts]
    return (sum(int(c.guard_active) for c in contexts),
            sum(1 for a, b in zip([0] + serials[:-1], serials) if a != b))


@pytest.fixture(scope='module')
def linear_cng():
    """The same scenario with the library's comfort noise on."""
    mic, ref = _scenario()
    config = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=SR, frame_size=2 * HOP, hop_size=HOP,
        mode=AecMode.PBFDKF, mu=0.3, enable_res=True, enable_cng=True)
    aec_out, contexts = run_aec_linear(mic, ref, config)
    return mic, config, aec_out, contexts


def test_scenario_moves_the_reset_serial(linear_cng):
    """The identity gates below exercise the hard-reset event (and not only a steady stream)."""
    _guard, resets = _events(linear_cng[3])
    assert resets >= 1


def test_identity_model_with_comfort_noise_and_reset_event(linear_cng, assets):
    """P carries the comfort noise from the rendering base through the network; a hard post-chain reset
    reseeds the generator at its hop and drops the synthesis tail when that frame is synthesised (the stage
    delays every frame by its lookahead, so the event travels with it): the identity model still equals the
    conventional chain with NR off, bit for bit.  Planted defects (an ignored event) must not."""
    mic, config, aec_out, contexts = linear_cng
    n_frames = len(contexts)
    tail = np.zeros(len(mic), np.float32)
    tail[:len(aec_out)] = aec_out
    conventional = run_res(tail, np.ones((n_frames, HOP + 1), np.float32), contexts,
                           config, use_nr=False, use_res=True, combine='min')
    span = n_frames * HOP
    stage = make_stage(assets, identity=True)
    dfn = run_dfn2_res(contexts, config, stage, enable_cng=True, output_length=len(mic))
    assert np.array_equal(dfn[:span], conventional[:span])
    assert np.any(conventional[:span] != 0)
    blind = [c.__class__(**{**c.__dict__, 'post_reset_serial': 0}) for c in contexts]
    stage = make_stage(assets, identity=True)
    ignored = run_dfn2_res(blind, config, stage, enable_cng=True, output_length=len(mic))
    assert not np.array_equal(ignored[:span], conventional[:span])


SR16 = 16000
HOP16 = 128


@pytest.fixture(scope='module')
def linear16_cng():
    """A 16 kHz stream (fft 256) whose echo path jumps half-way, comfort noise on: the AEC hard-resets its post chain."""
    rng = np.random.RandomState(0)
    n = int(SR16 * 4.0)
    far = (rng.randn(n) * 0.2).astype(np.float32)
    far *= (np.sin(2 * np.pi * 3.0 * np.arange(n) / SR16) > 0).astype(np.float32)

    def rir(delay):
        r = np.zeros(800, np.float32)
        r[delay] = 0.6
        r[delay + 140] = -0.3
        r[delay + 330:delay + 370] = rng.randn(40) * 0.02
        return r

    half = n // 2
    echo = np.zeros(n, np.float32)
    echo[:half] = np.convolve(far, rir(160))[:half]
    echo[half:] = np.convolve(far, rir(380))[half:n]
    mic = (echo + (rng.randn(n) * 0.01)).astype(np.float32)
    config = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=SR16, frame_size=2 * HOP16, hop_size=HOP16,
        mode=AecMode.PBFDKF, mu=0.3, enable_res=True, enable_cng=True)
    aec_out, contexts = run_aec_linear(mic, far, config)
    return mic, config, aec_out, contexts


def _bridge_host_hops(monkeypatch, contexts, config, assets):
    """run_dfn2_res through the rate bridge with a log of the hops on which reset_synthesis() was called and
    of the apply-lane native hops the bridge synthesised."""
    from pipelines.dfn.dfn2_rate_adapter import Dfn2RateBridge
    log = {'resets': [], 'apply': []}
    push, reset_synthesis = Dfn2RateBridge.push, Dfn2RateBridge.reset_synthesis

    def logged_push(self, estimate_spec, apply_spec):
        out = push(self, estimate_spec, apply_spec)
        log['apply'].append(np.array(self.host_hops[1], np.float32))
        return out

    def logged_reset(self):
        log['resets'].append(len(log['apply']))
        reset_synthesis(self)

    monkeypatch.setattr(Dfn2RateBridge, 'push', logged_push)
    monkeypatch.setattr(Dfn2RateBridge, 'reset_synthesis', logged_reset)
    run_dfn2_res(contexts, config, make_stage(assets, identity=True), enable_cng=True,
                 output_length=len(contexts) * config.hop_size)
    return log


def test_the_rate_bridge_clears_its_synthesis_on_the_hop_of_the_reset_event(linear16_cng, assets, monkeypatch):
    """Below 48 kHz the host-grid synthesis is the bridge's: its native-grid hops equal the conventional chain
    (RES on, NR off, comfort noise on, the AEC's window table, so within the window's last-bit rounding) on every
    hop, the reset hops included, because reset_synthesis() is called on the hop the serial moves, no later.
    A stream whose events are hidden from the pipeline must not."""
    mic, config, aec_out, contexts = linear16_cng
    n_frames = len(contexts)
    serials = [c.post_reset_serial for c in contexts]
    moved = [i for i, (a, b) in enumerate(zip([0] + serials[:-1], serials)) if a != b]
    assert moved, 'the scenario never hard-resets the post chain'
    tail = np.zeros(len(mic), np.float32)
    tail[:len(aec_out)] = aec_out
    conventional = run_res(tail, np.ones((n_frames, HOP16 + 1), np.float32), contexts,
                           config, use_nr=False, use_res=True, combine='min')[:n_frames * HOP16]
    log = _bridge_host_hops(monkeypatch, contexts, config, assets)
    assert log['resets'] == moved
    host = np.concatenate(log['apply'][:n_frames])
    assert np.max(np.abs(host - conventional)) < 1e-5
    blind = [c.__class__(**{**c.__dict__, 'post_reset_serial': 0}) for c in contexts]
    ignored = _bridge_host_hops(monkeypatch, blind, config, assets)
    assert ignored['resets'] == []
    assert np.max(np.abs(np.concatenate(ignored['apply'][:n_frames]) - conventional)) > 1e-3


def test_realtime_timing_is_a_pure_lookahead_delay(linear, pair_assets):
    mic, config, _aec_out, contexts = linear
    stage = make_stage(pair_assets, identity=True)
    aligned = run_dfn2_res(contexts, config, stage, enable_cng=False,
                           output_length=len(mic))
    delayed = run_dfn2_res(contexts, config, stage, enable_cng=False,
                           realtime_timing=True, output_length=len(mic))
    shift = (pair_assets.mask_lookahead + pair_assets.df_lookahead) * HOP
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


@pytest.mark.parametrize('limit', ['nan', 'inf', '-inf', '12dB'])
def test_the_cli_refuses_a_non_finite_attenuation_limit(tmp_path, limit):
    """argparse refuses the value (exit 2) before any input is read."""
    import subprocess
    run = subprocess.run(
        [sys.executable, '-m', 'pipelines.dfn.mono_aec_dfn_res.pipeline',
         '--mic', str(tmp_path / 'absent_mic.wav'), '--ref', str(tmp_path / 'absent_ref.wav'),
         '--output', str(tmp_path / 'out.wav'), '--atten-lim', limit],
        cwd=str(ROOT), capture_output=True, text=True)
    assert run.returncode == 2, run.stderr
    assert '--atten-lim' in run.stderr
    assert not (tmp_path / 'out.wav').exists()
