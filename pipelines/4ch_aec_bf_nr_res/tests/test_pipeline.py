"""Python four-channel orchestration contract tests."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from _scripted_delay import ScriptedDelay as _ScriptedDelay

_pipeline = import_module("pipelines.4ch_aec_bf_nr_res.pipeline")
BeamformerFrame = _pipeline.BeamformerFrame
EqualWeightBeamformer = _pipeline.EqualWeightBeamformer
FourChannelAecConfig = _pipeline.FourChannelAecConfig
FourChannelAecPipeline = _pipeline.FourChannelAecPipeline
PostBeamResidualSuppressor = _pipeline.PostBeamResidualSuppressor
SharedMatchedDelayEstimator = _pipeline.SharedMatchedDelayEstimator
AecPreset = _pipeline.AecPreset


def _external_equal_weight(pre):
    return EqualWeightBeamformer().process(pre.linear_hops, pre.contexts)


class _ZeroSumBeamformer:
    """Valid spatial weights that deliberately sum to zero."""

    def process(self, linear_hops, contexts):
        n_freqs = contexts[0].far_spec.size
        weights = np.zeros((4, n_freqs), dtype=np.complex64)
        weights[0] = 1.0
        weights[1] = -1.0
        return BeamformerFrame(
            samples=np.asarray(linear_hops[0] - linear_hops[1], dtype=np.float32),
            weights=weights,
        )


def test_resource_boundary_is_one_matcher_and_four_linear_filters():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    assert pipeline.matched_filter_instance_count == 1
    assert pipeline.linear_filter_instance_count == 4
    assert not pipeline.beamformer_configured


def test_post_beam_res_is_owned_by_the_pipeline():
    """One suppressor, built at config.aec_preset; none with enable_res off."""
    pipeline = _new_pipeline()
    assert isinstance(pipeline.post_beam_res, PostBeamResidualSuppressor)
    assert not hasattr(pipeline, "set_aec_preset")
    assert _new_pipeline(enable_res=False).post_beam_res is None
    seeded = _new_pipeline(aec_preset=AecPreset.AGGRESSIVE).post_beam_res
    reference = _new_suppressor(AecPreset.AGGRESSIVE)
    assert _live_floor(seeded) == _live_floor(reference)
    assert _dt_floor(seeded) == _dt_floor(reference)
    assert _live_floor(seeded) != _live_floor(_new_pipeline().post_beam_res)


def test_one_hop_is_finite_and_exports_post_beam_context():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    rng = np.random.default_rng(7)
    render = (rng.standard_normal(128) * 0.01).astype(np.float32)
    microphones = np.stack(
        [np.roll(render, channel + 1) for channel in range(4)], axis=1
    )
    pre = pipeline.front_process(microphones, render)
    result = pipeline.fuse(pre, _external_equal_weight(pre))
    assert pre.linear_channels.shape == (128, 4)
    assert len(pre.contexts) == 4
    assert pre.aligned_render.shape == (128,)
    assert result.frame_index == pre.frame_index == 0
    assert result.linear_channels.shape == (128, 4)
    assert result.beamformed.shape == (128,)
    assert result.context.error_spec.shape == (129,)
    assert result.context.r2.shape == (129,)
    assert np.all(np.isfinite(result.linear_channels))
    assert np.all(np.isfinite(result.beamformed))
    assert np.all(np.isfinite(result.context.error_spec))
    assert result.context.res_gain.shape == (129,)
    assert result.spec_frame().g_res_ctx is result.context.res_gain


def _echo_hops(n_hops, seed=7):
    rng = np.random.default_rng(seed)
    hops = []
    for _ in range(n_hops):
        render = (rng.standard_normal(128) * 0.01).astype(np.float32)
        microphones = np.stack(
            [np.roll(render, channel + 1) for channel in range(4)], axis=1
        )
        hops.append((microphones, render))
    return hops


def _fused_contexts(pipeline, hops):
    out = []
    for microphones, render in hops:
        pre = pipeline.front_process(microphones, render)
        out.append((pre, pipeline.fuse(pre, _external_equal_weight(pre)).context))
    return out


def test_res_off_leaves_the_gain_absent():
    pipeline = _new_pipeline(enable_res=False)
    for _, context in _fused_contexts(pipeline, _echo_hops(3)):
        # A reader that expects a RES gain must fail, not read a silent unity.
        assert context.res_gain is None


def test_owned_suppressor_equals_a_separate_one_over_the_same_frames():
    """enable_res = 1 is enable_res = 0 plus one suppressor at the same preset
    over the fused context and the aligned reference, bit for bit."""
    hops = _echo_hops(40)
    on = _fused_contexts(_new_pipeline(aec_preset=AecPreset.AGGRESSIVE), hops)
    off = _fused_contexts(_new_pipeline(enable_res=False), hops)
    separate = _new_suppressor(AecPreset.AGGRESSIVE)
    cut = False
    for (_, with_res), (pre, without) in zip(on, off):
        expected = separate.process(without, pre.aligned_render)
        assert with_res.res_gain.dtype == np.float32
        assert np.all((0.0 <= with_res.res_gain) & (with_res.res_gain <= 1.0))
        np.testing.assert_array_equal(with_res.res_gain, expected.res_gain)
        # The suppressor adds the gain and nothing else.
        np.testing.assert_array_equal(with_res.error_spec, without.error_spec)
        np.testing.assert_array_equal(with_res.r2, without.r2)
        np.testing.assert_array_equal(with_res.comfort_noise, without.comfort_noise)
        cut = cut or bool(np.any(with_res.res_gain < 0.9))
    assert cut, "the stimulus never made the suppressor cut"


def _nonlinear_echo_hops(n_hops, seed):
    """Broadband render through a nonlinear echo path, so a residual (and the
    suppressor's temporal state) survives the linear filters."""
    rng = np.random.default_rng(seed)
    hops = []
    for _ in range(n_hops):
        render = (rng.standard_normal(128) * 0.1).astype(np.float32)
        echo = render + 1.5 * render ** 3
        microphones = np.stack(
            [0.4 * np.roll(echo, channel + 1) + 0.003 * rng.standard_normal(128)
             for channel in range(4)], axis=1).astype(np.float32)
        hops.append((microphones, render))
    return hops


def test_reset_restarts_the_suppressor():
    """After reset() the pipeline, suppressor included, equals a fresh one."""
    hops = _nonlinear_echo_hops(60, seed=7)
    used = _new_pipeline()
    _fused_contexts(used, _nonlinear_echo_hops(120, seed=3))
    used.reset()
    after = _fused_contexts(used, hops)
    fresh = _fused_contexts(_new_pipeline(), hops)
    for (_, a), (_, b) in zip(after, fresh):
        np.testing.assert_array_equal(a.res_gain, b.res_gain)


@pytest.mark.parametrize(
    "sample_rate,frame_size", [(16000, 256), (16000, 512), (48000, 1024)]
)
def test_pre_beamformer_context_is_reconstructing_wola(sample_rate, frame_size):
    hop = frame_size // 2
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop,
        )
    )
    index = np.arange(frame_size, dtype=np.float64)
    window = np.sqrt(
        0.5 * (1.0 - np.cos(2.0 * np.pi * index / float(frame_size)))
    ).astype(np.float32)
    previous = np.zeros((4, hop), dtype=np.float32)
    rng = np.random.default_rng(23)

    for _ in range(12):
        render = (0.03 * rng.standard_normal(hop)).astype(np.float32)
        microphones = np.stack(
            [
                (0.35 + 0.05 * channel) * render
                + (0.002 * rng.standard_normal(hop)).astype(np.float32)
                for channel in range(4)
            ],
            axis=1,
        ).astype(np.float32)
        pre = pipeline.front_process(microphones, render)
        if pre.delay.changed:
            previous.fill(0.0)
        for channel, context in enumerate(pre.contexts):
            formed = np.asarray(context.formed_output, dtype=np.float32)
            expected = np.fft.rfft(
                np.concatenate((previous[channel], formed)) * window,
                n=frame_size,
            ).astype(np.complex64)
            np.testing.assert_allclose(
                context.error_spec, expected, rtol=0, atol=1e-6
            )
            np.testing.assert_allclose(
                context.near_spec,
                context.error_spec + context.echo_spec,
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_array_equal(pre.linear_hops[channel], formed)
            previous[channel] = formed.copy()


def test_shared_far_reference_survives_zero_sum_beamformer_weights():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    rng = np.random.default_rng(11)
    render = (rng.standard_normal(128) * 0.05).astype(np.float32)
    microphones = np.stack([render * (0.2 + 0.1 * i) for i in range(4)], axis=1)
    pre = pipeline.front_process(microphones, render)
    beamformed = _ZeroSumBeamformer().process(pre.linear_hops, pre.contexts)
    result = pipeline.fuse(pre, beamformed)
    assert float(np.linalg.norm(result.context.far_spec)) > 0.0
    assert result.context.far_power > 0.0


def _contexts_for_scalar_fusion(sample_rate, frame_size):
    hop = frame_size // 2
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(
            sample_rate=sample_rate, frame_size=frame_size, hop_size=hop
        )
    )
    rng = np.random.default_rng(29)
    render = (rng.standard_normal(hop) * 0.02).astype(np.float32)
    microphones = np.stack([render * (0.2 + 0.1 * i) for i in range(4)], axis=1)
    return pipeline.front_process(microphones, render)


@pytest.mark.parametrize("sample_rate,frame_size", [(16000, 256), (16000, 512), (48000, 1024)])
@pytest.mark.parametrize("active_lane", range(4))
def test_zero_weight_lanes_do_not_drive_control_scalars(sample_rate, frame_size, active_lane):
    """Match the C core's contributor mask without changing its DT policy."""
    pre = _contexts_for_scalar_fusion(sample_rate, frame_size)
    contexts = [
        replace(c, filter_converged=False, dt_indicator=0.9, saturation_level=0.9)
        for c in pre.contexts
    ]
    contexts[active_lane] = replace(
        contexts[active_lane], filter_converged=True,
        dt_indicator=0.1, saturation_level=0.2,
    )
    weights = np.zeros((4, frame_size // 2 + 1), dtype=np.complex64)
    weights[active_lane] = 1.0
    fused = _pipeline._fuse_contexts(
        contexts, BeamformerFrame(samples=pre.linear_hops[active_lane].copy(), weights=weights)
    )
    assert fused.filter_converged
    assert fused.dt_indicator == pytest.approx(0.1)
    assert fused.saturation_level == pytest.approx(0.2)
    np.testing.assert_array_equal(fused.error_spec, contexts[active_lane].error_spec)


@pytest.mark.parametrize("sample_rate,frame_size", [(16000, 256), (16000, 512), (48000, 1024)])
def test_all_contributing_lanes_keep_existing_control_reductions(sample_rate, frame_size):
    pre = _contexts_for_scalar_fusion(sample_rate, frame_size)
    contexts = [
        replace(c, filter_converged=(i != 2), dt_indicator=0.1 * i,
                saturation_level=0.2 * i)
        for i, c in enumerate(pre.contexts)
    ]
    weights = np.full((4, frame_size // 2 + 1), 0.25, dtype=np.complex64)
    fused = _pipeline._fuse_contexts(
        contexts, BeamformerFrame(samples=pre.linear_hops.mean(axis=0), weights=weights)
    )
    assert fused.filter_converged == all(c.filter_converged for c in contexts)
    assert fused.dt_indicator == max(c.dt_indicator for c in contexts)
    assert fused.saturation_level == max(c.saturation_level for c in contexts)


@pytest.mark.parametrize("sample_rate,frame_size", [(16000, 256), (16000, 512), (48000, 1024)])
def test_single_nonzero_bin_still_votes_on_control_scalars(sample_rate, frame_size):
    pre = _contexts_for_scalar_fusion(sample_rate, frame_size)
    contexts = [
        replace(c, filter_converged=True, dt_indicator=0.1, saturation_level=0.1)
        for c in pre.contexts
    ]
    contexts[3] = replace(
        contexts[3], filter_converged=False, dt_indicator=0.7, saturation_level=0.8
    )
    weights = np.zeros((4, frame_size // 2 + 1), dtype=np.complex64)
    weights[0] = 1.0
    # C tests the complex coefficient directly, not a normalized energy or
    # magnitude that could underflow or round a genuine contributor to zero.
    weights[3, -1] = complex(0.0, float(np.nextafter(np.float32(0), np.float32(1))))
    fused = _pipeline._fuse_contexts(
        contexts, BeamformerFrame(samples=pre.linear_hops[0].copy(), weights=weights)
    )
    assert not fused.filter_converged
    assert fused.dt_indicator == pytest.approx(0.7)
    assert fused.saturation_level == pytest.approx(0.8)


@pytest.mark.parametrize("sample_rate,frame_size", [(16000, 256), (16000, 512), (48000, 1024)])
def test_zero_weights_remain_rejected(sample_rate, frame_size):
    pre = _contexts_for_scalar_fusion(sample_rate, frame_size)
    weights = np.zeros((4, frame_size // 2 + 1), dtype=np.complex64)
    with pytest.raises(ValueError, match="zero weight"):
        _pipeline._fuse_contexts(
            pre.contexts, BeamformerFrame(samples=pre.linear_hops[0].copy(), weights=weights)
        )


def test_default_grids_are_no_padding_power_of_two():
    assert FourChannelAecConfig(sample_rate=16000).resolved_grid() == (256, 128)
    assert FourChannelAecConfig(sample_rate=48000).resolved_grid() == (1024, 512)
    assert FourChannelAecConfig(sample_rate=16000, frame_size=512).resolved_grid() == (
        512,
        256,
    )
    with pytest.raises(ValueError, match="256, 512"):
        FourChannelAecConfig(sample_rate=16000, frame_size=320).resolved_grid()
    # 128 is a power of two but outside the C core's whitelist (16 kHz: 256/512,
    # 4aec_nr_res.c derive_dims_and_configs); it must be rejected here too.
    with pytest.raises(ValueError, match="256, 512"):
        FourChannelAecConfig(sample_rate=16000, frame_size=128).resolved_grid()
    with pytest.raises(ValueError, match="1024"):
        FourChannelAecConfig(sample_rate=48000, frame_size=512).resolved_grid()


def test_48k_wrapper_feeds_inner_estimator_raw_native_rate_hops(monkeypatch):
    """The wrapper does no decimation of its own (a stride pick would alias):
    the inner estimator is built at the true native sample_rate and fed every
    raw sample; EchoPathDelayEstimator owns the anti-alias + decimate-by-3
    sidechain (as delay_aec.c's DaResample48)."""
    estimator = SharedMatchedDelayEstimator(sample_rate=48000, hop_size=512)
    assert estimator._estimator._sample_rate == 48000
    seen = []

    def record(near, far):
        seen.append((near.copy(), far.copy()))
        return False

    monkeypatch.setattr(estimator._estimator, "accumulate", record)
    x0 = np.arange(512, dtype=np.float32)
    x1 = np.arange(512, 1024, dtype=np.float32)
    estimator.accumulate(x0, x0)
    estimator.accumulate(x1, x1)
    joined = np.concatenate([pair[0] for pair in seen])
    # No decimation at this wrapper layer: every raw 48kHz sample
    # reaches the inner estimator unchanged.
    np.testing.assert_array_equal(joined, np.arange(0, 1024, dtype=np.float32))


def test_shape_and_nonfinite_rejection():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    with pytest.raises(ValueError, match="shape"):
        pipeline.front_process(
            np.zeros((128, 3), np.float32), np.zeros(128, np.float32)
        )
    bad = np.zeros((128, 4), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.front_process(bad, np.zeros(128, np.float32))


def test_one_call_path_requires_an_explicit_test_or_offline_adapter():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    with pytest.raises(RuntimeError, match="front_process"):
        pipeline.process(np.zeros((128, 4), np.float32), np.zeros(128, np.float32))
    with pytest.raises(RuntimeError, match="explicit test/offline beamformer"):
        pipeline.process_signal(
            np.zeros((128, 4), np.float32), np.zeros(128, np.float32)
        )


def test_explicit_adapter_keeps_legacy_offline_convenience_path():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128),
        beamformer=EqualWeightBeamformer(),
    )
    result = pipeline.process(
        np.zeros((128, 4), np.float32), np.zeros(128, np.float32)
    )
    assert pipeline.beamformer_configured
    assert result.frame_index == 0


def test_queued_pre_frames_are_snapshots_and_post_results_must_be_ordered():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    rng = np.random.default_rng(19)
    render0 = (rng.standard_normal(128) * 0.01).astype(np.float32)
    render1 = (rng.standard_normal(128) * 0.01).astype(np.float32)
    mics0 = np.stack([np.roll(render0, i + 1) for i in range(4)], axis=1)
    mics1 = np.stack([np.roll(render1, i + 2) for i in range(4)], axis=1)

    pre0 = pipeline.front_process(mics0, render0)
    saved = pre0.contexts[0].error_spec.copy()
    pre1 = pipeline.front_process(mics1, render1)
    np.testing.assert_array_equal(pre0.contexts[0].error_spec, saved)

    with pytest.raises(ValueError, match="out of order"):
        pipeline.fuse(pre1, _external_equal_weight(pre1))
    result0 = pipeline.fuse(pre0, _external_equal_weight(pre0))
    result1 = pipeline.fuse(pre1, _external_equal_weight(pre1))
    assert (result0.frame_index, result1.frame_index) == (0, 1)


def test_reset_invalidates_external_frames_that_are_still_in_flight():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    pre = pipeline.front_process(
        np.zeros((128, 4), np.float32), np.zeros(128, np.float32)
    )
    beamformed = _external_equal_weight(pre)
    pipeline.reset()
    with pytest.raises(ValueError, match="invalidated by reset"):
        pipeline.fuse(pre, beamformed)


def test_pre_frame_cannot_resume_on_a_different_pipeline_instance():
    config = FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    source = FourChannelAecPipeline(config)
    other = FourChannelAecPipeline(config)
    pre = source.front_process(
        np.zeros((128, 4), np.float32), np.zeros(128, np.float32)
    )
    with pytest.raises(ValueError, match="different pipeline"):
        other.fuse(pre, _external_equal_weight(pre))


# ---------------------------------------------------------------------------
# An admitted shared-delay change REALIGNS the four lanes; it does not reset
# them (a reset would wipe four converged filters and restart four WOLA
# sequences).  See lib/aec ``AEC.apply_external_realign`` and its
# test_external_realign.py.
# ---------------------------------------------------------------------------

_REALIGN_HOP = 128
_REALIGN_DELAY = 300        # inside the lanes' 896-sample tap span at this grid
_REALIGN_CONVERGE = 160     # hops served at the raw alignment first
_REALIGN_WATCH = 8          # hops watched after the change


def _realign_scene():
    """Four mics carrying the same echo at a fixed bulk delay, and a script
    that moves the published alignment from 0 to that delay once the lanes
    have converged against it."""
    total = _REALIGN_CONVERGE + _REALIGN_WATCH + 2
    rng = np.random.default_rng(0x4CEA)
    far = (rng.standard_normal((total + 4) * _REALIGN_HOP) * 0.25).astype(np.float32)
    mic = np.zeros_like(far)
    mic[_REALIGN_DELAY:] = 0.5 * far[:-_REALIGN_DELAY]
    script = [(0, True)] * _REALIGN_CONVERGE
    script += [(_REALIGN_DELAY, True)] * (total - _REALIGN_CONVERGE)
    return far, mic, script


def _run_realign_scene():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=2 * _REALIGN_HOP,
                             hop_size=_REALIGN_HOP)
    )
    far, mic, script = _realign_scene()
    pipeline._shared_delay._estimator = _ScriptedDelay(script)
    rows = []
    for index in range(len(script)):
        lo, hi = index * _REALIGN_HOP, (index + 1) * _REALIGN_HOP
        mics = np.repeat(mic[lo:hi, None], 4, axis=1)
        pre = pipeline.front_process(mics, far[lo:hi])
        rows.append({
            "changed": bool(pre.delay.changed),
            "delay": int(pre.delay.delay_samples),
            "warm": pipeline.realign_warm_lane_count,
            "soft": pipeline.realign_soft_lane_count,
            "frames": tuple(lane._frame_count for lane in pipeline._lanes),
            "residual": float(np.sqrt(np.mean(
                np.square(pre.linear_channels.astype(np.float64))))),
            "echo": float(np.sqrt(np.mean(
                np.square(mic[lo:hi].astype(np.float64))))),
        })
    return pipeline, rows


def test_admitted_delay_change_realigns_four_lanes_instead_of_resetting_them():
    pipeline, rows = _run_realign_scene()

    changed = [i for i, row in enumerate(rows) if row["changed"]]
    assert changed[0] == 0, "the first acquisition must land on hop 0"
    assert len(changed) == 2, "the scene must acquire once and re-lock once"
    relock = changed[1]
    assert rows[relock]["delay"] == _REALIGN_DELAY

    # The acquisition sweep is four zero-delta no-ops, counted as soft.
    assert (rows[0]["warm"], rows[0]["soft"]) == (0, 4)

    before = rows[relock - 1]
    assert before["residual"] < 0.5 * before["echo"], (
        "the lanes must be cancelling before the change, or nothing below "
        "measures a realign"
    )

    # 1. Every lane was realigned, and every one of them kept its cancellation
    #    (the warm tap-transfer path).  A lane.reset() sweep reaches neither
    #    counter at all.
    assert rows[relock]["warm"] == before["warm"] + 4
    assert rows[relock]["soft"] == before["soft"]

    # 2. No lane restarted its framing: _frame_count is the hop counter a
    #    reset() zeroes, so it is the cheapest witness that the WOLA sequence
    #    and the analysis frames continued across the boundary.
    assert rows[relock]["frames"] == tuple(
        n + 1 for n in before["frames"]
    ), "a lane restarted its frame sequence across the realign"
    assert all(row["frames"] == (i + 1,) * 4 for i, row in enumerate(rows)), (
        "every lane must advance exactly one frame per hop for the whole scene"
    )

    # 3. The cancellation actually survived: no re-exposed echo on any hop
    #    after the change.  This is the audible half -- 1 and 2 can be
    #    satisfied by a realign that shifts the taps the wrong way.
    for row in rows[relock:]:
        assert row["residual"] < 0.5 * row["echo"], (
            f"echo re-exposed after the realign: {row['residual']:.4f} vs "
            f"echo {row['echo']:.4f}"
        )

    # 4. The counters share the lanes' reset epoch, so an instrumentation
    #    total can never span two of them (matches four_aec_nr_res_reset()).
    pipeline.reset()
    assert pipeline.realign_warm_lane_count == 0
    assert pipeline.realign_soft_lane_count == 0


# ── runtime strength retarget ────────────────────────────────────────────
#
# The strength is carried by the pipeline's PostBeamResidualSuppressor
# (pipeline.post_beam_res); the four lanes compute no gain of their own.

def _live_floor(suppressor):
    return suppressor._gain._split_floor_far_active_live


def _dt_floor(suppressor):
    return suppressor._gain._split_floor_dt


def _new_pipeline(**overrides):
    return FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128,
                             **overrides)
    )


def _new_suppressor(preset=None):
    return PostBeamResidualSuppressor(16000, 128, 129, preset)


def test_preset_retarget_moves_the_suppressor_not_the_lanes():
    """A retarget on the suppressor must not reach the lanes.

    The four lanes run with spatial_linear_context and never compute a gain, so
    the strength that shapes the output lives in the ONE post-beam suppressor.
    """
    pipeline = _new_pipeline()
    suppressor = pipeline.post_beam_res
    before_post = _live_floor(suppressor)
    before_lanes = [
        lane._aec3_sg._split_floor_far_active_live for lane in pipeline._lanes
    ]

    suppressor.set_preset("aggressive")
    assert _live_floor(suppressor) != before_post
    assert [
        lane._aec3_sg._split_floor_far_active_live for lane in pipeline._lanes
    ] == before_lanes


def test_preset_retarget_survives_reset():
    """reset() rebuilds the suppressor from a freshly constructed AecConfig.

    Without the suppressor storing the chosen floor, that rebuild would
    silently put balanced back.
    """
    suppressor = _new_suppressor()
    suppressor.set_preset("aggressive")
    after_set = _live_floor(suppressor)

    suppressor.reset()
    assert _live_floor(suppressor) == after_set

    # And the mutation that would break it: clearing the stored value makes the
    # rebuild revert, so the assertion above is not vacuous.
    suppressor._split_floor_far_active_db = None
    suppressor.reset()
    assert _live_floor(suppressor) != after_set


def test_preset_retarget_moves_the_dt_floor_too():
    """set_preset must also move the DT-protect floor, not just far-active.

    aggressive deepens min_gain_floor_dt_db to -30 dB (mild/balanced share the
    dataclass default -20 dB), so switching to aggressive must change the
    live DT floor the suppressor actually applies during double-talk.
    """
    suppressor = _new_suppressor()
    before_dt = _dt_floor(suppressor)

    suppressor.set_preset("aggressive")
    assert _dt_floor(suppressor) != before_dt


def test_preset_retarget_dt_floor_survives_reset():
    """Mirrors test_preset_retarget_survives_reset for the DT floor.

    Without the suppressor storing the chosen DT floor alongside far-active,
    reset()'s fresh AecConfig would silently put balanced's -20 dB back.
    """
    suppressor = _new_suppressor()
    suppressor.set_preset("aggressive")
    after_set = _dt_floor(suppressor)

    suppressor.reset()
    assert _dt_floor(suppressor) == after_set

    # The mutation that would break it: clearing the stored value makes the
    # rebuild revert, so the assertion above is not vacuous.
    suppressor._split_floor_dt_db = None
    suppressor.reset()
    assert _dt_floor(suppressor) != after_set


def test_construction_default_preset_leaves_split_floor_enabled():
    """Default BALANCED must not perturb default behavior.

    BALANCED's from_preset() override dict is empty, so this is a no-op
    value-wise: the point is that it stays True (the dataclass default),
    not that construction changed it.
    """
    assert _new_suppressor(AecPreset.BALANCED)._gain._split_floor_enabled is True
    assert _new_suppressor()._gain._split_floor_enabled is True


def test_construction_time_no_floor_disables_split_floor():
    """Building with NO_FLOOR must start with the split-floor mechanism OFF
    and keep it off across a reset() rebuild.

    Construction-time only here: this does NOT exercise
    set_preset("no_floor") on an already-running instance -- that runtime
    retarget path is covered separately, by
    test_preset_retarget_moves_split_floor_enabled_too and
    test_construction_time_no_floor_matches_runtime_retarget below.
    """
    suppressor = _new_suppressor(AecPreset.NO_FLOOR)
    assert suppressor._gain._split_floor_enabled is False
    suppressor.reset()
    assert suppressor._gain._split_floor_enabled is False


def test_construction_time_preset_matches_runtime_retarget():
    """Building with AGGRESSIVE must match balanced-then-retarget.

    The C construction path (se_derive -> aec_config_from_preset) applies both
    floor fields from the preset before aec_res_init(), the same two fields
    set_preset() retargets; a seed that left them at balanced's values would
    diverge from the C side.
    """
    built_aggressive = _new_suppressor(AecPreset.AGGRESSIVE)
    retargeted = _new_suppressor(AecPreset.BALANCED)
    retargeted.set_preset("aggressive")

    assert _live_floor(built_aggressive) == _live_floor(retargeted)
    assert _dt_floor(built_aggressive) == _dt_floor(retargeted)
    # And it must actually differ from balanced's own values, so the
    # assertions above are not vacuously comparing two balanced instances.
    assert _live_floor(built_aggressive) != _live_floor(_new_suppressor())
    assert _dt_floor(built_aggressive) != _dt_floor(_new_suppressor())


def _gate(suppressor):
    return suppressor._gain._split_floor_enabled


def test_preset_retarget_moves_split_floor_enabled_too():
    """set_preset must also retarget the split-floor gate, not just the
    two floor dB values (the runtime half of NO_FLOOR).

    Mirrors the C twin's aec_res_set_preset(), which mirrors the gate into
    r->sg_cfg.split_floor_enabled on every call.
    """
    suppressor = _new_suppressor()
    assert _gate(suppressor) is True

    suppressor.set_preset("no_floor")
    assert _gate(suppressor) is False

    # And it must switch back on too -- this is a retarget, not a one-way latch.
    suppressor.set_preset("aggressive")
    assert _gate(suppressor) is True


def test_preset_retarget_split_floor_enabled_survives_reset():
    """Mirrors test_preset_retarget_survives_reset for the split-floor gate.

    Without the suppressor storing the chosen gate value, reset()'s fresh
    AecConfig would silently put BALANCED's True back.
    """
    suppressor = _new_suppressor()
    suppressor.set_preset("no_floor")
    after_set = _gate(suppressor)
    assert after_set is False

    suppressor.reset()
    assert _gate(suppressor) == after_set

    # The mutation that would break it: clearing the stored value makes the
    # rebuild revert, so the assertion above is not vacuous.
    suppressor._split_floor_enabled = None
    suppressor.reset()
    assert _gate(suppressor) != after_set


def test_construction_time_no_floor_matches_runtime_retarget():
    """Building with NO_FLOOR must match balanced-then-retarget.

    Same property as test_construction_time_preset_matches_runtime_retarget,
    for the boolean gate: a fresh instance and a retargeted one must agree on
    all three fields NO_FLOOR touches.
    """
    built_no_floor = _new_suppressor(AecPreset.NO_FLOOR)
    retargeted = _new_suppressor()
    retargeted.set_preset("no_floor")

    assert _gate(built_no_floor) == _gate(retargeted) is False
    assert _live_floor(built_no_floor) == _live_floor(retargeted)
    assert _dt_floor(built_no_floor) == _dt_floor(retargeted)


def _no_floor_scene_context(nf, hop):
    """A hop where the split floor actually binds: converged, far-active
    (enough render energy to fire the latch on this very hop), no
    double-talk, and residual echo (r2) that dwarfs the near-end spectrum --
    so the AEC3-native min_gain this hop would clamp suppression far below
    the far-active floor if nothing lifted it back up.
    """
    from lib.aec.python.modules.dataclasses import AecResContext

    rng = np.random.default_rng(11)
    # render std 0.1 -> after the *32768 int16 scaling inside process(), mean
    # square ~= (0.1*32768)**2 ~= 1.07e7, comfortably over the 1e6 latch
    # threshold (min_gain_far_latch_power) on the very first hop.
    render = (rng.standard_normal(hop) * 0.1).astype(np.float32)
    tiny = (1e-4 * (rng.standard_normal(nf) + 1j * rng.standard_normal(nf))).astype(
        np.complex64
    )
    strong_r2 = np.full(nf, 1.0e9, dtype=np.float32)
    zc = np.zeros(nf, dtype=np.complex64)
    z = np.zeros(hop, dtype=np.float32)
    ctx = AecResContext(
        raw_output=z, formed_output=z, echo_spec=zc, far_power=1.0e6, far_spec=zc,
        near_spec=tiny, filter_converged=True, erle_factor=1.0, dt_indicator=0.0,
        divergence=0.0, over_sub=1.0, saturation_level=0.0, error_spec=tiny,
        res_gain=np.ones(nf, np.float32), comfort_noise=np.zeros(nf, np.float32),
        r2=strong_r2, usable_linear=True,
    )
    return ctx, render


def test_runtime_no_floor_switch_changes_suppressor_gain_output():
    """The whole point of this retarget: it must move what PostBeamResidual-
    Suppressor.process() actually outputs on a scene where the split floor
    matters, not just an internal flag.

    BALANCED (floor ON) must clamp the gain up to at least the far-active
    floor once the far-active latch fires; NO_FLOOR (switched at runtime on
    a second, otherwise-identical suppressor -- no rebuild) must NOT, so its
    gain is strictly lower on this deep-suppression scene.
    """
    suppressor = _new_suppressor()
    ctx, render = _no_floor_scene_context(suppressor.n_freqs, suppressor.hop_size)

    gain_balanced = suppressor.process(ctx, render).res_gain.copy()
    far_active_floor_power = suppressor._gain._split_floor_far_active_live

    # Sanity: the floor actually bound on at least one bin this hop (not a
    # vacuous scene where AEC3's own gain was already above the floor).
    assert np.any(gain_balanced >= far_active_floor_power), (
        "scene did not exercise the floor at all -- strengthen r2/near_spec"
    )

    suppressor2 = _new_suppressor()
    suppressor2.set_preset("no_floor")
    assert _gate(suppressor2) is False
    gain_no_floor = suppressor2.process(ctx, render).res_gain

    # Every bin the floor touched above must now be free to go lower.
    assert np.all(gain_no_floor <= gain_balanced)
    assert np.any(gain_no_floor < gain_balanced), (
        "NO_FLOOR produced an identical gain to BALANCED on a scene built "
        "to exercise the floor -- the runtime gate is not actually reaching "
        "the suppressor's gain computation"
    )


def test_preset_retarget_rejects_bad_arguments_without_writing():
    suppressor = _new_suppressor()
    before = _live_floor(suppressor)
    for bad in (99, None, "gentle"):
        with pytest.raises(ValueError):
            suppressor.set_preset(bad)
        assert _live_floor(suppressor) == before
    with pytest.raises(ValueError):
        suppressor.set_preset("aggressive", ramp_ms=-1.0)
    assert _live_floor(suppressor) == before


def test_preset_ramp_walks_and_lands():
    suppressor = _new_suppressor()
    start = _live_floor(suppressor)
    suppressor.set_preset("aggressive", ramp_ms=100.0)
    gain = suppressor._gain
    target = gain._split_floor_far_active
    assert _live_floor(suppressor) == start, "the setter itself must not move it"
    prev = start
    for _ in range(64):
        gain._advance_split_floor_ramp()
        cur = _live_floor(suppressor)
        assert cur <= prev
        assert cur >= target
        prev = cur
    assert _live_floor(suppressor) == target
