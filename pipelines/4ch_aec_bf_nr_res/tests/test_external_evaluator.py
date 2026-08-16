"""Tests for external-recording acceptance validation."""

from importlib import import_module

import pytest

validate = import_module(
    "pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings"
).validate


def _valid_result():
    return {
        "case": "synthetic",
        "finite": True,
        "sample_rate": 16000,
        "frame_size": 256,
        "fft_size": 256,
        "hop_size": 128,
        "final_delay_error_samples": 0,
        "delay_tolerance_samples": 80,
        "c_pipeline": {
            "sample_rate": 16000,
            "frame_size": 256,
            "fft_size": 256,
            "hop": 128,
            "n_freqs": 129,
            "doa_sample_rate": 16000,
            "doa_frame_size": 256,
            "doa_hop_size": 128,
            "doa_fft_size": 256,
            "gsc_sample_rate": 16000,
            "gsc_frame_size": 256,
            "gsc_hop_size": 128,
            "gsc_fft_size": 256,
            "matched_filters": 1,
            "linear_aecs": 4,
            "nr": 1,
            "post_res": 1,
            "final_delay_solid": 1,
            "doa_analysis_frames": 1,
            "doa_update_frames": 1,
            "gsc_adaptive_frames": 1,
        },
    }


def test_external_recording_contract_accepts_shared_spatial_grid():
    validate(_valid_result())


@pytest.mark.parametrize(
    "field",
    ["doa_hop_size", "gsc_fft_size", "n_freqs"],
)
def test_external_recording_contract_rejects_spatial_grid_skew(field):
    result = _valid_result()
    result["c_pipeline"][field] += 1
    with pytest.raises(RuntimeError, match=field):
        validate(result)
