"""Python four-channel orchestration contract tests."""

from importlib import import_module

import numpy as np
import pytest

from _scripted_delay import ScriptedDelay as _ScriptedDelay

_pipeline = import_module("pipelines.4ch_aec_bf_nr_res.pipeline")
BeamformerFrame = _pipeline.BeamformerFrame
EqualWeightBeamformer = _pipeline.EqualWeightBeamformer
FourChannelAecConfig = _pipeline.FourChannelAecConfig
FourChannelAecPipeline = _pipeline.FourChannelAecPipeline
SharedMatchedDelayEstimator = _pipeline.SharedMatchedDelayEstimator


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
    assert pipeline.residual_suppressor_instance_count == 1
    assert not pipeline.beamformer_configured


def test_one_hop_is_finite_and_exports_post_beam_context():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    rng = np.random.default_rng(7)
    render = (rng.standard_normal(128) * 0.01).astype(np.float32)
    microphones = np.stack(
        [np.roll(render, channel + 1) for channel in range(4)], axis=1
    )
    pre = pipeline.process_pre_beamformer(microphones, render)
    result = pipeline.process_post_beamformer(pre, _external_equal_weight(pre))
    assert pre.linear_channels.shape == (128, 4)
    assert len(pre.contexts) == 4
    assert pre.aligned_render.shape == (128,)
    assert result.frame_index == pre.frame_index == 0
    assert result.linear_channels.shape == (128, 4)
    assert result.beamformed.shape == (128,)
    assert result.context.error_spec.shape == (129,)
    assert result.context.r2.shape == (129,)
    assert result.context.res_gain.shape == (129,)
    assert np.all(np.isfinite(result.linear_channels))
    assert np.all(np.isfinite(result.beamformed))
    assert np.all(np.isfinite(result.context.error_spec))
    assert np.all((0.0 <= result.context.res_gain) & (result.context.res_gain <= 1.0))


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
        pre = pipeline.process_pre_beamformer(microphones, render)
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
    pre = pipeline.process_pre_beamformer(microphones, render)
    beamformed = _ZeroSumBeamformer().process(pre.linear_hops, pre.contexts)
    result = pipeline.process_post_beamformer(pre, beamformed)
    assert float(np.linalg.norm(result.context.far_spec)) > 0.0
    assert result.context.far_power > 0.0


def test_default_grids_are_no_padding_power_of_two():
    assert FourChannelAecConfig(sample_rate=16000).resolved_grid() == (256, 128)
    assert FourChannelAecConfig(sample_rate=48000).resolved_grid() == (1024, 512)
    assert FourChannelAecConfig(sample_rate=16000, frame_size=512).resolved_grid() == (
        512,
        256,
    )
    with pytest.raises(ValueError, match="256, 512"):
        FourChannelAecConfig(sample_rate=16000, frame_size=320).resolved_grid()
    # frame_size=128 is a power of two but outside the C core's exact
    # whitelist (16 kHz only supports 256/512) -- must be rejected here too,
    # so the Python reference can't silently diverge from the C acceptance
    # gate (see 4aec_nr_res.c's derive_dims_and_configs).
    with pytest.raises(ValueError, match="256, 512"):
        FourChannelAecConfig(sample_rate=16000, frame_size=128).resolved_grid()
    with pytest.raises(ValueError, match="1024"):
        FourChannelAecConfig(sample_rate=48000, frame_size=512).resolved_grid()


def test_48k_wrapper_feeds_inner_estimator_raw_native_rate_hops(monkeypatch):
    """2026-08-03 fix: the wrapper no longer does its own external 48kHz
    stride-pick decimation (which had no anti-alias filter and aliased).
    The inner estimator is now constructed at the TRUE native sample_rate
    and fed every raw sample; EchoPathDelayEstimator owns the anti-alias +
    decimate-by-3 sidechain internally (mirrors delay_aec3.c's
    DaResample48)."""
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
    # No decimation at this wrapper layer any more: every raw 48kHz sample
    # reaches the inner estimator unchanged.
    np.testing.assert_array_equal(joined, np.arange(0, 1024, dtype=np.float32))


def test_shape_and_nonfinite_rejection():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    with pytest.raises(ValueError, match="shape"):
        pipeline.process_pre_beamformer(
            np.zeros((128, 3), np.float32), np.zeros(128, np.float32)
        )
    bad = np.zeros((128, 4), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.process_pre_beamformer(bad, np.zeros(128, np.float32))


def test_one_call_path_requires_an_explicit_test_or_offline_adapter():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    with pytest.raises(RuntimeError, match="process_pre_beamformer"):
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

    pre0 = pipeline.process_pre_beamformer(mics0, render0)
    saved = pre0.contexts[0].error_spec.copy()
    pre1 = pipeline.process_pre_beamformer(mics1, render1)
    np.testing.assert_array_equal(pre0.contexts[0].error_spec, saved)

    with pytest.raises(ValueError, match="out of order"):
        pipeline.process_post_beamformer(pre1, _external_equal_weight(pre1))
    result0 = pipeline.process_post_beamformer(pre0, _external_equal_weight(pre0))
    result1 = pipeline.process_post_beamformer(pre1, _external_equal_weight(pre1))
    assert (result0.frame_index, result1.frame_index) == (0, 1)


def test_reset_invalidates_external_frames_that_are_still_in_flight():
    pipeline = FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )
    pre = pipeline.process_pre_beamformer(
        np.zeros((128, 4), np.float32), np.zeros(128, np.float32)
    )
    beamformed = _external_equal_weight(pre)
    pipeline.reset()
    with pytest.raises(ValueError, match="invalidated by reset"):
        pipeline.process_post_beamformer(pre, beamformed)


def test_pre_frame_cannot_resume_on_a_different_pipeline_instance():
    config = FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    source = FourChannelAecPipeline(config)
    other = FourChannelAecPipeline(config)
    pre = source.process_pre_beamformer(
        np.zeros((128, 4), np.float32), np.zeros(128, np.float32)
    )
    with pytest.raises(ValueError, match="different pipeline"):
        other.process_post_beamformer(pre, _external_equal_weight(pre))


# ---------------------------------------------------------------------------
# An admitted shared-delay change REALIGNS the four lanes; it does not reset
# them.  A reset per lane -- what this used to be -- wipes four converged
# filters and restarts four WOLA sequences on one hop: the echo comes back for
# dozens of hops and the output drops a near-silent one.  See
# lib/aec ``AEC.apply_external_realign`` and its test_external_realign.py.
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
        pre = pipeline.process_pre_beamformer(mics, far[lo:hi])
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

    # The acquisition sweep is four zero-delta no-ops reported as soft, which
    # is what `realign_soft_lane_count` documents and the only reason the
    # counter is not read as "four filters were restarted".
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

def _live_floor(pipeline):
    return pipeline._post_beam_res._gain._split_floor_far_active_live


def _lane_floors(pipeline):
    return [lane._aec3_sg._split_floor_far_active_live for lane in pipeline._lanes]


def _new_pipeline():
    return FourChannelAecPipeline(
        FourChannelAecConfig(sample_rate=16000, frame_size=256, hop_size=128)
    )


def test_preset_retarget_moves_the_shared_post_stage_not_the_lanes():
    """The four lanes run with spatial_linear_context and never compute a gain.

    So the strength that shapes this pipeline's output lives in the ONE shared
    post-beam suppressor. A setter that looped over the lanes instead would
    leave this assertion unmoved -- which is exactly the failure the C twin's
    test pins as well.
    """
    pipeline = _new_pipeline()
    before_post = _live_floor(pipeline)
    before_lanes = _lane_floors(pipeline)

    pipeline.set_aec_preset("aggressive")
    assert _live_floor(pipeline) != before_post
    assert _lane_floors(pipeline) == before_lanes


def test_preset_retarget_survives_reset():
    """reset() rebuilds the suppressor from a freshly constructed AecConfig.

    Without the pipeline storing the chosen floor, that rebuild would silently
    put balanced back -- the same two-copy hazard the C core has with
    post_sg_cfg.
    """
    pipeline = _new_pipeline()
    pipeline.set_aec_preset("aggressive")
    after_set = _live_floor(pipeline)

    pipeline.reset()
    assert _live_floor(pipeline) == after_set

    # And the mutation that would break it: clearing the stored value makes the
    # rebuild revert, so the assertion above is not vacuous.
    pipeline._post_beam_res._split_floor_far_active_db = None
    pipeline.reset()
    assert _live_floor(pipeline) != after_set


def test_preset_retarget_rejects_bad_arguments_without_writing():
    pipeline = _new_pipeline()
    before = _live_floor(pipeline)
    for bad in (99, None, "gentle"):
        with pytest.raises(ValueError):
            pipeline.set_aec_preset(bad)
        assert _live_floor(pipeline) == before
    with pytest.raises(ValueError):
        pipeline.set_aec_preset("aggressive", ramp_ms=-1.0)
    assert _live_floor(pipeline) == before


def test_preset_ramp_walks_and_lands():
    pipeline = _new_pipeline()
    start = _live_floor(pipeline)
    pipeline.set_aec_preset("aggressive", ramp_ms=100.0)
    gain = pipeline._post_beam_res._gain
    target = gain._split_floor_far_active
    assert _live_floor(pipeline) == start, "the setter itself must not move it"
    prev = start
    for _ in range(64):
        gain._advance_split_floor_ramp()
        cur = _live_floor(pipeline)
        assert cur <= prev
        assert cur >= target
        prev = cur
    assert _live_floor(pipeline) == target
