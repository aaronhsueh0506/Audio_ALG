"""Contract tests for the dual-input DFN2 stage (pipelines/dfn/dfn2_stage.py).

A randomly initialised DFN2 (no checkpoint) is enough for every structural
claim here; the numbers a trained network produces are irrelevant to whether
estimation and application stay disjoint, whether the delay is exactly the
pair's ``mask_lookahead + df_lookahead`` frames, or whether a failed head
evaluation leaves state untouched.  The timing claims run for the shipped pair
and four others (``PAIRS``), each with a model built for that pair from a
``config.ini`` copy.
"""
import configparser
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip('torch')

from pipelines.dfn import dfn2_stage as stage_mod  # noqa: E402
from pipelines.dfn.dfn2_stage import (  # noqa: E402
    DF_BINS, DF_ORDER, MASK_LOOKAHEAD, DF_LOOKAHEAD, N_BINS, N_ERB, SCALE_IN,
    SCALE_OUT, Dfn2Heads, Dfn2Stage, IdentityHeads, expand_erb_mask,
    erb_inv_is_partition_of_unity, load_dfn2, pair_geometry,
)
from pipelines.dfn.tests.dfn2_pairs import PAIRS, pair_config, pair_id  # noqa: E402

@pytest.fixture(scope='module')
def assets():
    return load_dfn2(seed=0)


@pytest.fixture(scope='module', params=PAIRS, ids=pair_id)
def pair_assets(request, tmp_path_factory):
    path = pair_config(tmp_path_factory.mktemp('dfn2_pair'), request.param)
    return load_dfn2(config_path=path, seed=0)


def test_dfn2_modules_are_dfn2s_even_after_a_sibling_project_imported_first(monkeypatch):
    """AIAEC and the other AINR projects each define a top-level ``train``,
    ``model`` and ``export_onnx``; once a sibling's is in ``sys.modules`` a
    bare import hands it back and the stage fails at setup (seen only in a
    combined run)."""
    for name in stage_mod._AINR_BARE_NAMES:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))   # a sibling's, attribute-free
    monkeypatch.setattr(stage_mod, '_DFN2_MODULES', None)
    train, model, export = stage_mod.dfn2_modules()
    for module in (train, model, export):
        assert pathlib.Path(module.__file__).parent == pathlib.Path(stage_mod._DFN2_DIR)
    assert callable(train.read_model_config)


def _frames(count, seed, scale=1.0):
    rng = np.random.RandomState(seed)
    re = rng.randn(count, N_BINS).astype(np.float32) * scale
    im = rng.randn(count, N_BINS).astype(np.float32) * scale
    im[:, 0] = 0.0
    im[:, -1] = 0.0
    return (re + 1j * im).astype(np.complex64)


def _run(stage, a_frames, b_frames):
    outputs = []
    for a, b in zip(a_frames, b_frames):
        outputs.append(stage.push(a, b))
    outputs.extend(stage.flush())
    return outputs


class CountingHeads:
    """Stand-in whose mask depends on the call index; taps zero, alpha 0."""

    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def __call__(self, erb_window, spec_window):
        mask = np.full(N_ERB, 0.25 + 0.001 * self.calls, np.float32)
        self.calls += 1
        return mask, np.zeros((DF_BINS, DF_ORDER, 2), np.float32), 0.0


class FailingHeads:
    """Wraps real heads and raises on the listed call indices."""

    def __init__(self, inner, fail_on):
        self.inner = inner
        self.fail_on = set(fail_on)
        self.calls = 0

    def reset(self):
        self.inner.reset()
        self.calls = 0

    def __call__(self, erb_window, spec_window):
        index = self.calls
        self.calls += 1
        if index in self.fail_on:
            raise RuntimeError('accelerator failed on purpose')
        return self.inner(erb_window, spec_window)


def test_load_dfn2_refuses_a_checkpoint_from_another_model_contract(tmp_path):
    """A checkpoint whose recorded model/feature/loss version differs from the
    code's must be refused, exactly as inference.py refuses it -- the tensor
    shapes can match while the network is a different one."""
    torch.manual_seed(0)
    assets = load_dfn2(seed=0)
    stale = {
        'state_dict': assets.model.state_dict(),
        'model_version': 'some_other_model_contract_v0',
        'feature_version': 'some_other_feature_contract_v0',
        'loss_version': 'some_other_loss_contract_v0',
        'contract': {},
    }
    path = tmp_path / 'stale.pth'
    torch.save(stale, path)
    with pytest.raises(ValueError):
        load_dfn2(checkpoint=str(path))


def test_scale_constants_round_trip_exactly():
    assert SCALE_IN * SCALE_OUT == np.float32(1.0)
    x = np.array([1.0, -3.0e-3, 12345.678, 2.0 ** -100], np.float32)
    assert np.array_equal((x * SCALE_IN) * SCALE_OUT, x)


def test_model_erb_inv_is_a_partition_of_unity(assets):
    assert erb_inv_is_partition_of_unity(assets.erb_inv)
    broken = assets.erb_inv.copy()
    broken[3] *= np.float32(1.0 + 1e-4)
    assert not erb_inv_is_partition_of_unity(broken)
    with pytest.raises(ValueError):
        Dfn2Stage(stage_mod.Dfn2Assets(assets.model, assets.feature_cfg,
                                       assets.erb_fb, broken), heads=IdentityHeads())


def test_the_stage_takes_its_geometry_from_the_model_pair(pair_assets):
    stage = Dfn2Stage(pair_assets, heads=IdentityHeads())
    mask, df = pair_assets.mask_lookahead, pair_assets.df_lookahead
    assert (stage.mask_lookahead, stage.df_lookahead) == (mask, df)
    assert (stage.df_history, stage.df_ring, stage.model_lookahead) == pair_geometry(mask, df)
    assert stage.df_history == DF_ORDER - df - 1
    assert stage.df_ring == max(DF_ORDER, mask + df + 1)
    assert stage.output_delay_frames == mask + df


def test_identity_heads_is_an_exact_lookahead_delay_of_b(pair_assets):
    frames = 12
    a = _frames(frames, seed=1)
    b = _frames(frames, seed=2, scale=0.37)
    stage = Dfn2Stage(pair_assets, heads=IdentityHeads())
    delay = stage.model_lookahead
    outputs = _run(stage, a, b)
    assert all(out is None for out in outputs[:delay])
    emitted = outputs[delay:]
    assert len(emitted) == frames
    for index, out in enumerate(emitted):
        assert out.frame_index == index
        assert np.array_equal(out.spectrum, b[index]), index
        assert np.array_equal(out.bin_gain, np.ones(N_BINS, np.float32))


def test_estimation_and_application_are_disjoint(assets):
    frames = 10
    a1 = _frames(frames, seed=3)
    a2 = _frames(frames, seed=4)
    b1 = _frames(frames, seed=5)
    b2 = _frames(frames, seed=6)

    # Identity heads: A cannot reach the output at all.
    delay = Dfn2Stage(assets, heads=IdentityHeads()).model_lookahead
    out_a1 = _run(Dfn2Stage(assets, heads=IdentityHeads()), a1, b1)
    out_a2 = _run(Dfn2Stage(assets, heads=IdentityHeads()), a2, b1)
    for x, y in zip(out_a1[delay:], out_a2[delay:]):
        assert np.array_equal(x.spectrum, y.spectrum)

    # Real heads: B reaches the output, and the feature state follows A only.
    s1 = Dfn2Stage(assets, heads=Dfn2Heads(assets.model))
    s2 = Dfn2Stage(assets, heads=Dfn2Heads(assets.model))
    o1 = _run(s1, a1, b1)
    o2 = _run(s2, a1, b2)
    assert any(not np.array_equal(x.spectrum, y.spectrum)
               for x, y in zip(o1[delay:], o2[delay:]))
    assert torch.equal(s1.ema_state['erb'], s2.ema_state['erb'])
    assert torch.equal(s1.ema_state['spec'], s2.ema_state['spec'])
    for x, y in zip(o1[delay:], o2[delay:]):
        assert np.array_equal(x.bin_gain, y.bin_gain)   # gains come from A


def test_output_gain_is_the_expansion_of_that_frames_mask(pair_assets):
    frames = 9
    heads = CountingHeads()
    stage = Dfn2Stage(pair_assets, heads=heads)
    outputs = _run(stage, _frames(frames, seed=7), _frames(frames, seed=8))
    assert heads.calls == frames + stage.model_lookahead - pair_assets.mask_lookahead
    for out in outputs[stage.model_lookahead:]:
        # head frame f is served by heads call number f (calls start once
        # frame mask_lookahead has been pushed)
        expected = expand_erb_mask(
            np.full(N_ERB, 0.25 + 0.001 * out.frame_index, np.float32),
            pair_assets.erb_inv)
        assert np.array_equal(out.bin_gain, expected), out.frame_index


def test_failed_heads_keep_state_and_advance_clocks(assets):
    frames = 12
    a = _frames(frames, seed=9)
    b = _frames(frames, seed=10)
    inner = Dfn2Heads(assets.model)
    heads = FailingHeads(inner, fail_on={4, 5})
    stage = Dfn2Stage(assets, heads=heads)
    states = []
    for t in range(frames):
        before = tuple(s.clone() for s in inner.state)
        out = stage.push(a[t], b[t])
        after = inner.state
        states.append((before, after, out))
    # Calls 4 and 5 are made at pushes 5 and 6 (call index = push - 1): the
    # recurrent state must not move on those pushes, every push still emits.
    for t in (5, 6):
        before, after, out = states[t]
        assert all(torch.equal(x, y) for x, y in zip(before, after))
        assert out is not None and np.all(np.isfinite(out.spectrum))
    for t in (4, 7):
        before, after, _out = states[t]
        assert not all(torch.equal(x, y) for x, y in zip(before, after))
    assert stage.skips == 2
    assert stage.frames_in == frames
    assert stage.stream_frame_index == frames
    # Head frames 4 and 5 (served by the failed calls) received the unit mask;
    # their neighbours received the network's mask.
    by_frame = {s[2].frame_index: s[2] for s in states if s[2] is not None}
    for f in (4, 5):
        assert np.array_equal(by_frame[f].bin_gain, np.ones(N_BINS, np.float32)), f
    for f in (3, 6):
        assert not np.array_equal(by_frame[f].bin_gain, np.ones(N_BINS, np.float32)), f
    # An always-failing run is exactly the identity run.
    always = Dfn2Stage(assets, heads=FailingHeads(Dfn2Heads(assets.model),
                                                   fail_on=set(range(frames + 2))))
    ident = Dfn2Stage(assets, heads=IdentityHeads())
    for x, y in zip(_run(always, a, b), _run(ident, a, b)):
        if x is None:
            assert y is None
        else:
            assert np.array_equal(x.spectrum, y.spectrum)


def test_flush_drains_the_last_lookahead_source_frames(pair_assets):
    frames = 7
    b = _frames(frames, seed=11)
    stage = Dfn2Stage(pair_assets, heads=IdentityHeads())
    delay = stage.model_lookahead
    live = [stage.push(np.zeros(N_BINS, np.complex64), b[t]) for t in range(frames)]
    tail = stage.flush()
    assert [o.frame_index for o in live if o is not None] == list(range(frames - delay))
    assert [o.frame_index for o in tail] == list(range(frames - delay, frames))
    for out in tail:
        assert np.array_equal(out.spectrum, b[out.frame_index])


def test_atten_lim_mixes_toward_b(assets):
    frames = 6
    b = _frames(frames, seed=12)

    class Silence:
        def reset(self):
            pass

        def __call__(self, erb_window, spec_window):
            return (np.zeros(N_ERB, np.float32),
                    np.zeros((DF_BINS, DF_ORDER, 2), np.float32), 0.0)

    delay = Dfn2Stage(assets, heads=IdentityHeads()).model_lookahead
    muted = _run(Dfn2Stage(assets, heads=Silence(), atten_lim_db=0.0), b, b)
    for out in muted[delay:]:
        assert np.array_equal(out.spectrum, np.zeros(N_BINS, np.complex64))
    limited = _run(Dfn2Stage(assets, heads=Silence(), atten_lim_db=20.0), b, b)
    lim = np.float32(10.0 ** (-20.0 / 20.0))
    for out in limited[delay:]:
        expected = ((((b[out.frame_index].real * SCALE_IN).astype(np.float32) * lim)
                     * SCALE_OUT).astype(np.float32)
                    + 1j * (((b[out.frame_index].imag * SCALE_IN).astype(np.float32) * lim)
                            * SCALE_OUT).astype(np.float32)).astype(np.complex64)
        assert np.array_equal(out.spectrum, expected)


def test_reset_replays_identically(pair_assets):
    frames = 8
    a = _frames(frames, seed=13)
    b = _frames(frames, seed=14)
    stage = Dfn2Stage(pair_assets, heads=Dfn2Heads(pair_assets.model))
    first = _run(stage, a, b)
    stage.reset()
    second = _run(stage, a, b)
    for x, y in zip(first, second):
        if x is None:
            assert y is None
        else:
            assert np.array_equal(x.spectrum, y.spectrum)


def test_a_pair_the_model_refuses_is_refused(tmp_path):
    """The accepted set is the model constructor's, not a copy kept here."""
    for pair in ((3, 0), (0, DF_ORDER), (-1, 1)):
        with pytest.raises(ValueError):
            load_dfn2(config_path=pair_config(tmp_path, pair), seed=0)


def test_a_checkpoint_of_another_pair_is_refused(tmp_path):
    """Same tensor shapes, different streaming contract: refused."""
    shipped = load_dfn2(seed=0)
    train, _model, _export = stage_mod.dfn2_modules()
    cfg = configparser.ConfigParser()
    cfg.read(pathlib.Path(stage_mod._DFN2_DIR) / 'config.ini')
    path = tmp_path / 'shipped_pair.pth'
    other = (0, 1) if (MASK_LOOKAHEAD, DF_LOOKAHEAD) != (0, 1) else (1, 1)
    model_cfg = train.read_model_config(cfg)
    sr, hop = cfg.getint('signal', 'sr'), cfg.getint('signal', 'hop_len', fallback=1024 // 2)
    contract = train.make_checkpoint_contract(
        sr, cfg.getint('signal', 'n_fft'), cfg.getint('signal', 'win_len', fallback=1024), hop,
        model_cfg['n_erb'], model_cfg['df_bins'], model_cfg['df_order'],
        model_cfg['mask_lookahead'], model_cfg['df_lookahead'], model_cfg['mask_pf'],
        model_cfg['pf_beta'], train.read_feature_config(cfg, sr, hop), train.read_loss_config(cfg))
    torch.save({'state_dict': shipped.model.state_dict(), 'model_version': train.MODEL_VERSION,
                'feature_version': train.FEATURE_VERSION, 'contract': contract}, path)
    load_dfn2(checkpoint=str(path))                       # its own pair loads
    with pytest.raises(ValueError):
        load_dfn2(checkpoint=str(path), config_path=pair_config(tmp_path, other))


@pytest.mark.parametrize('limit', [float('nan'), float('inf'), float('-inf')])
def test_a_non_finite_attenuation_limit_is_refused_at_construction(assets, limit):
    """The same check as set_atten_lim and the C stage's resolve()."""
    with pytest.raises(ValueError):
        Dfn2Stage(assets, heads=IdentityHeads(), atten_lim_db=limit)
