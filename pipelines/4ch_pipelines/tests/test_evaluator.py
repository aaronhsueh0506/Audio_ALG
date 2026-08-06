"""Tests for the bundled-recording evaluator contract."""

from importlib import import_module

import numpy as np
import pytest

_evaluator = import_module("pipelines.4ch_pipelines.evaluate_recordings")
estimate_file_offset = _evaluator.estimate_file_offset
place_on_timeline = _evaluator.place_on_timeline
validate_recording_contract = _evaluator.validate_recording_contract


def _valid_result():
    return {
        "case": "synthetic",
        "finite": True,
        "matched_filter_instances": 1,
        "linear_filter_instances": 4,
        "residual_suppressor_instances": 1,
        "beamformer_configured": False,
        "final_shared_delay": {"solid": True},
        "first_nonzero_delay_frame": 10,
        "final_delay_error_samples": -7,
        "hop_size": 256,
        "sample_rate": 16000,
        "processed_samples": 4096,
    }


def test_fixture_alignment_places_source_without_erasing_a_second_path_delay():
    rng = np.random.default_rng(9)
    source = rng.standard_normal(256).astype(np.float32)
    capture = place_on_timeline(source, 73, 512)
    lag, correlation = estimate_file_offset(capture, source)
    assert lag == 73
    assert correlation > 0.999


def test_recording_contract_accepts_expected_resource_and_delay_boundary():
    validate_recording_contract(_valid_result())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("finite", False, "NaN or Inf"),
        ("linear_filter_instances", 1, "linear_filter_instances"),
        ("beamformer_configured", True, "beamformer_configured"),
        ("final_delay_error_samples", 129, "half-hop"),
        ("first_nonzero_delay_frame", None, "nonzero delay"),
    ],
)
def test_recording_contract_rejects_regressions(field, value, message):
    result = _valid_result()
    result[field] = value
    with pytest.raises(RuntimeError, match=message):
        validate_recording_contract(result)
