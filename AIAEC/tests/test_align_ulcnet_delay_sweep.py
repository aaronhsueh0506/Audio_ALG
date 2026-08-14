import argparse

import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet.sweep_delay_depth import (
    _delay_summary,
    parse_depths,
    run_streaming_frames,
)
from AIAEC.aiaec_common import SignalGrid


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
