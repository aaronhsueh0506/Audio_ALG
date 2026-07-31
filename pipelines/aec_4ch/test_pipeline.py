import numpy as np
import pytest

from pipelines.aec_4ch.pipeline import (
    BeamformerFrame,
    EqualWeightBeamformer,
    FourChannelAecConfig,
    FourChannelAecPipeline,
    SharedMatchedDelayEstimator,
)


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
    assert FourChannelAecConfig(sample_rate=16000).resolved_grid() == (512, 256)
    assert FourChannelAecConfig(sample_rate=48000).resolved_grid() == (1024, 512)
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


def test_48k_decimator_keeps_continuous_phase_across_512_sample_hops(monkeypatch):
    estimator = SharedMatchedDelayEstimator(sample_rate=48000, hop_size=512)
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
    np.testing.assert_array_equal(joined, np.arange(0, 1024, 3, dtype=np.float32))


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
