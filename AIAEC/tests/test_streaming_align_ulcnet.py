"""Frame-by-frame streaming equivalence for Align-ULCNet.

The offline forward is the reference; ``forward_stream`` must replay it one
STFT frame at a time.  Tolerances below were MEASURED on this fixture (worst
over three input pairs: enhanced 4.1e-8, mask 1.2e-7, magnitude_mask 6.0e-8,
delay 4.0e-7 -- pure GEMM-batching noise from running the GRUs stepwise) and
pinned with ~10x headroom.
"""

import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.aiaec_common import SignalGrid

GRID = SignalGrid(16000, 512, 512, 256)
# Longer than max_delay_frames (64 on this grid) so both delay rings wrap.
T = 96

ENHANCED_TOL = 5e-7
MASK_TOL = 2e-6
MAGMASK_TOL = 1e-6
DELAY_TOL = 5e-6


def _spec(batch, frames, bins, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(batch, frames, bins, generator=g),
                         torch.randn(batch, frames, bins, generator=g))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = AlignULCNet(GRID).eval()
    # At default init the softmax over 64 delay slots is near-uniform and the
    # aligned feature is almost invariant to a one-frame far shift, which
    # would make the can-fail test below vacuous.  Widening only the
    # alignment pathway makes the delay head timing-sensitive (a one-frame
    # far shift moves delay_distribution by ~0.18) without touching the
    # offline/streaming contract under test.
    with torch.no_grad():
        for module in (model.align.query, model.align.key,
                       model.align.score.conv):
            for parameter in module.parameters():
                parameter.mul_(4.0)
    return model


def _stream(model, error, far, state=None):
    if state is None:
        state = model.create_stream_state()
    enhanced, mask, magmask, delay = [], [], [], []
    with torch.no_grad():
        for t in range(error.shape[1]):
            out = model.forward_stream(error[:, t:t + 1], far[:, t:t + 1],
                                       state)
            enhanced.append(out.enhanced)
            mask.append(out.mask)
            magmask.append(out.auxiliary["magnitude_mask"])
            delay.append(out.delay_distribution)
    return (torch.cat(enhanced, 1), torch.cat(mask, 1),
            torch.cat(magmask, 1), torch.cat(delay, 1))


def _offline(model, error, far):
    with torch.no_grad():
        return model(linear_error=error, far_end=far)


def test_stream_matches_offline(model):
    assert model.stream_output_delay == 0
    error = _spec(1, T, GRID.n_freqs, seed=1)
    far = _spec(1, T, GRID.n_freqs, seed=2)
    ref = _offline(model, error, far)
    enhanced, mask, magmask, delay = _stream(model, error, far)
    assert (enhanced - ref.enhanced).abs().max() <= ENHANCED_TOL
    assert (mask - ref.mask).abs().max() <= MASK_TOL
    assert (magmask - ref.auxiliary["magnitude_mask"]).abs().max() <= MAGMASK_TOL
    assert (delay - ref.delay_distribution).abs().max() <= DELAY_TOL


def test_shifted_far_is_detected(model):
    # CAN-FAIL check: stream a far reference delayed by one frame against the
    # unshifted offline run.  The delay head is the output that encodes far
    # timing; at this fixture's init the one-frame shift moves it by ~0.18.
    # (enhanced moves only ~1e-6 here: the near-uniform delay softmax averages
    # a one-frame far shift out of the aligned feature at random init.)
    error = _spec(1, T, GRID.n_freqs, seed=3)
    far = _spec(1, T, GRID.n_freqs, seed=4)
    ref = _offline(model, error, far)
    shifted = torch.cat((torch.zeros_like(far[:, :1]), far[:, :-1]), dim=1)
    _, _, _, delay = _stream(model, error, shifted)
    assert (delay - ref.delay_distribution).abs().max() > 1e-3


def test_fresh_states_do_not_cross_contaminate(model):
    error_a = _spec(1, T, GRID.n_freqs, seed=5)
    far_a = _spec(1, T, GRID.n_freqs, seed=6)
    error_b = _spec(1, T, GRID.n_freqs, seed=7)
    far_b = _spec(1, T, GRID.n_freqs, seed=8)
    ref_a = _offline(model, error_a, far_a)
    ref_b = _offline(model, error_b, far_b)

    # Interleave the two utterances frame-by-frame through two fresh states;
    # each must still reproduce its own offline run.
    state_a = model.create_stream_state()
    state_b = model.create_stream_state()
    enhanced_a, enhanced_b = [], []
    with torch.no_grad():
        for t in range(T):
            enhanced_a.append(model.forward_stream(
                error_a[:, t:t + 1], far_a[:, t:t + 1], state_a).enhanced)
            enhanced_b.append(model.forward_stream(
                error_b[:, t:t + 1], far_b[:, t:t + 1], state_b).enhanced)
    enhanced_a = torch.cat(enhanced_a, 1)
    enhanced_b = torch.cat(enhanced_b, 1)
    assert (enhanced_a - ref_a.enhanced).abs().max() <= ENHANCED_TOL
    assert (enhanced_b - ref_b.enhanced).abs().max() <= ENHANCED_TOL


def test_stream_state_requires_eval_mode(model):
    model.train()
    try:
        with pytest.raises(RuntimeError):
            model.create_stream_state()
    finally:
        model.eval()
