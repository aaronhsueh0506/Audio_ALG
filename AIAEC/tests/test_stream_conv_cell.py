"""StreamConv2dCell keeps the FULL input window as its state.

The state is ``(kt - 1) * dt + 1`` frames, oldest first. A step cuts the
oldest frame off, appends the new one, convolves, and hands back the whole
new window, so a graph built from the cell can bind the state output to the
state input's address. The oldest frame is only there to keep the two shapes
equal; the convolution never reads it.
"""

import pytest
import torch

from AIAEC.aiaec_common import CausalConv2d
from AIAEC.aiaec_streaming import StreamConv2dCell


def _cell(kt=4, dt=1, kf=3, in_channels=3, out_channels=5, seed=0):
    torch.manual_seed(seed)
    module = CausalConv2d(in_channels, out_channels, (kt, kf),
                          dilation=(dt, 1), bias=False).eval()
    return module, StreamConv2dCell.from_causal(module)


@pytest.mark.parametrize('kt,dt', [(4, 1), (5, 1), (5, 2), (2, 1)])
def test_state_is_the_full_window(kt, dt):
    _module, cell = _cell(kt, dt)
    x = torch.randn(1, 3, 1, 9)
    cell.step(x)
    state = cell.state_tensors()['window']
    assert state.shape == (1, 3, (kt - 1) * dt + 1, 9)
    assert cell.window == (kt - 1) * dt + 1
    # The newest frame is the last slot; everything before it is zero until
    # the stream has filled the window.
    assert torch.equal(state[:, :, -1:], x)
    assert not state[:, :, :-1].any()


@pytest.mark.parametrize('kt,dt', [(4, 1), (5, 2)])
def test_stream_equals_offline_and_window_tracks_the_input(kt, dt):
    module, cell = _cell(kt, dt)
    x = torch.randn(1, 3, 24, 9)
    with torch.no_grad():
        offline = module(x)
        streamed = []
        for t in range(24):
            streamed.append(cell.step(x[:, :, t:t + 1]))
            window = cell.state_tensors()['window']
            lo = t + 1 - cell.window
            expected = (x[:, :, lo:t + 1] if lo >= 0 else torch.cat(
                (torch.zeros(1, 3, -lo, 9), x[:, :, :t + 1]), dim=2))
            assert torch.equal(window, expected)
    assert torch.allclose(torch.cat(streamed, dim=2), offline,
                          rtol=1e-6, atol=1e-6)


def test_several_new_frames_in_one_step_match_one_frame_steps():
    module, cell_one = _cell(4, 1)
    _module, cell_many = _cell(4, 1)
    x = torch.randn(1, 3, 12, 9)
    with torch.no_grad():
        singles = torch.cat(
            [cell_one.step(x[:, :, t:t + 1]) for t in range(12)], dim=2)
        grouped = torch.cat(
            [cell_many.step(x[:, :, 0:5]), cell_many.step(x[:, :, 5:12])],
            dim=2)
    assert torch.allclose(singles, grouped, rtol=1e-6, atol=1e-6)
    assert torch.equal(cell_one.state_tensors()['window'],
                       cell_many.state_tensors()['window'])


def test_oldest_window_frame_is_never_read():
    """Poisoning slot 0 of the state before each step must change nothing."""
    _module, clean = _cell()
    _module, poisoned = _cell()
    with torch.no_grad():
        for _ in range(8):
            frame = torch.randn(1, 3, 1, 9)
            expected = clean.step(frame)
            if poisoned._window is not None:
                poisoned._window[:, :, 0] = float('nan')
            assert torch.equal(poisoned.step(frame), expected)
    # Slot 0 of the next state is the previous slot 1, so the poison never
    # reaches a slot the convolution reads.
    assert torch.isfinite(poisoned._window[:, :, 1:]).all()


def test_lookahead_is_refused():
    module = CausalConv2d(3, 5, (4, 3), bias=False)
    with pytest.raises(ValueError, match='lookahead'):
        StreamConv2dCell(module.conv, 4, 1, 1, 1, lookahead=1)


def test_a_conv_without_temporal_context_needs_no_cell():
    module = CausalConv2d(3, 5, (1, 3), bias=False)
    with pytest.raises(ValueError, match='no cell'):
        StreamConv2dCell.from_causal(module)


def test_reset_forgets_the_window():
    _module, cell = _cell()
    cell.step(torch.randn(1, 3, 1, 9))
    assert cell.state_tensors()
    cell.reset()
    assert cell.state_tensors() == {}
