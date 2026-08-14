import argparse

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet.sweep_delay_depth import (
    _alignment_summary,
    _delay_summary,
    _write_alignment_trace,
    parse_depths,
    run_streaming_frames,
)
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen.measure_align_residual import EngineRun


GRID = SignalGrid(16000, 512, 512, 256)


def _spec(frames, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(1, frames, GRID.n_freqs, generator=generator),
        torch.randn(1, frames, GRID.n_freqs, generator=generator),
    )


def test_parse_depths_preserves_order_and_removes_duplicates():
    assert parse_depths('64, 16,8,16') == [64, 16, 8]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_depths('64,zero')
    with pytest.raises(argparse.ArgumentTypeError):
        parse_depths('8,0')
    with pytest.raises(argparse.ArgumentTypeError):
        parse_depths(' , ')


def test_streaming_sweep_uses_requested_depth_and_reports_state_memory():
    torch.manual_seed(0)
    large = AlignULCNet(GRID, max_delay_frames=8).eval()
    small = AlignULCNet(GRID, max_delay_frames=4).eval()
    small.load_state_dict(large.state_dict(), strict=True)
    error = _spec(12, 1)
    far = _spec(12, 2)

    result_large = run_streaming_frames(large, error, far)
    result_small = run_streaming_frames(small, error, far)

    assert result_large.enhanced.shape == error.shape
    assert result_small.enhanced.shape == error.shape
    assert result_large.delay_distribution.shape == (1, 12, 8)
    assert result_small.delay_distribution.shape == (1, 12, 4)
    assert result_large.state_bytes > result_small.state_bytes > 0
    assert result_large.elapsed_seconds >= 0.0
    assert torch.isfinite(result_small.enhanced.real).all()

    summary = _delay_summary(
        result_small.delay_distribution, depth=4,
        hop_seconds=GRID.hop_len / GRID.sample_rate,
    )
    assert summary['evaluated_frames'] == 9.0
    assert 0.0 <= summary['argmax_at_max_depth_rate'] <= 1.0
    assert 0.0 <= summary['max_depth_probability_mean'] <= 1.0


def test_alignment_trace_reports_applied_delay_and_acquisition(tmp_path):
    run = EngineRun(
        sample_rate=16000,
        hop_size=256,
        error=np.zeros(4 * 256, dtype=np.float32),
        echo_estimate=np.zeros(4 * 256, dtype=np.float32),
        aligned_far=np.zeros(4 * 256, dtype=np.float32),
        delay_samples=np.asarray([-1, 7936, 7936, 8000], dtype=np.int64),
        confidence=np.asarray([0.0, 0.5, 1.0, 1.0], dtype=np.float64),
    )

    summary = _alignment_summary(run)
    assert summary['aec_delay_acquired'] is True
    assert summary['aec_first_acquired_ms'] == 32.0
    assert summary['aec_initial_delay_samples'] == 7936
    assert summary['aec_final_delay_samples'] == 8000
    assert summary['aec_final_delay_ms'] == 500.0
    assert summary['aec_delay_change_events'] == 1
    assert summary['aec_final_confidence'] == 1.0

    path = tmp_path / 'aec_alignment.csv'
    _write_alignment_trace(path, run)
    rows = path.read_text(encoding='utf-8').splitlines()
    assert rows[0].startswith('hop,time_ms,applied_delay_samples')
    assert rows[1].split(',')[2:] == ['-1', 'nan', '0.000000000', '0']
    assert rows[-1].split(',')[1:4] == ['64.000000', '8000', '500.000000']


def test_alignment_summary_reports_never_acquired():
    run = EngineRun(
        sample_rate=16000,
        hop_size=256,
        error=np.zeros(512, dtype=np.float32),
        echo_estimate=np.zeros(512, dtype=np.float32),
        aligned_far=np.zeros(512, dtype=np.float32),
        delay_samples=np.asarray([-1, -1], dtype=np.int64),
        confidence=np.asarray([0.0, 0.0], dtype=np.float64),
    )
    summary = _alignment_summary(run)
    assert summary['aec_delay_acquired'] is False
    assert summary['aec_final_delay_samples'] == -1
    assert np.isnan(summary['aec_first_acquired_ms'])
    assert np.isnan(summary['aec_final_delay_ms'])
