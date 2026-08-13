"""Frame-by-frame streaming equivalence tests for CAGCRN.

``forward_stream`` must replay ``forward`` exactly, one STFT frame at a time,
with all time context carried in the state returned by
``create_stream_state`` (``stream_output_delay`` is 0 for this model).
"""

import pytest
import torch

from AIAEC.CAGCRN import CAGCRN
from AIAEC.aiaec_common import SignalGrid

GRID = SignalGrid(16000, 512, 512, 256)

# Measured on random weights (seeds 0/1, B=2, T=96): the largest offline vs
# streamed deviation over enhanced/mask/delay/erb_mask was 5.3e-7 -- pure
# float32 accumulation-order noise.  Pinned with ~10x headroom.
TOL = 5e-6


def _spec(batch, frames, bins, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(batch, frames, bins, generator=generator),
        torch.randn(batch, frames, bins, generator=generator),
    )


def _make_model(seed=0):
    torch.manual_seed(seed)
    return CAGCRN(GRID).eval()


def _run_stream(model, mic, far, far_shift_frames=0):
    """Stream every frame; optionally delay the streamed far input only."""
    state = model.create_stream_state()
    enhanced, masks, delays, erb_masks = [], [], [], []
    frames = mic.shape[1]
    with torch.no_grad():
        for t in range(frames):
            s = t - far_shift_frames
            far_frame = (far[:, s:s + 1] if 0 <= s < frames
                         else torch.zeros_like(far[:, :1]))
            out = model.forward_stream(mic[:, t:t + 1], far_frame, state)
            enhanced.append(out.enhanced)
            masks.append(out.mask)
            delays.append(out.delay_distribution)
            erb_masks.append(out.auxiliary["erb_mask"])
    return (torch.cat(enhanced, dim=1), torch.cat(masks, dim=2),
            torch.cat(delays, dim=1), torch.cat(erb_masks, dim=2))


def test_stream_matches_offline():
    # T=96 > max_delay_frames=63, so the CATA ring wraps its full span.
    model = _make_model()
    assert model.stream_output_delay == 0
    mic = _spec(2, 96, GRID.n_freqs, seed=10)
    far = _spec(2, 96, GRID.n_freqs, seed=20)
    with torch.no_grad():
        offline = model(microphone=mic, far_end=far)
    enhanced, mask, delay, erb_mask = _run_stream(model, mic, far)
    assert (offline.enhanced - enhanced).abs().max() < TOL
    assert (offline.mask - mask).abs().max() < TOL
    assert (offline.delay_distribution - delay).abs().max() < TOL
    assert (offline.auxiliary["erb_mask"] - erb_mask).abs().max() < TOL


def test_far_shifted_by_one_frame_is_detected():
    # Can-fail control: with the streamed reference one frame late, the
    # comparison above must break -- otherwise it proves nothing.
    model = _make_model()
    mic = _spec(1, 90, GRID.n_freqs, seed=30)
    far = _spec(1, 90, GRID.n_freqs, seed=40)
    with torch.no_grad():
        offline = model(microphone=mic, far_end=far)
    enhanced, _, _, _ = _run_stream(model, mic, far, far_shift_frames=1)
    assert (offline.enhanced - enhanced).abs().max() > 1e-3


def test_fresh_states_do_not_cross_contaminate():
    model = _make_model()
    utterances = [
        (_spec(1, 90, GRID.n_freqs, seed=50), _spec(1, 90, GRID.n_freqs, seed=51)),
        (_spec(1, 90, GRID.n_freqs, seed=52), _spec(1, 90, GRID.n_freqs, seed=53)),
    ]
    for mic, far in utterances:
        with torch.no_grad():
            offline = model(microphone=mic, far_end=far)
        enhanced, _, _, _ = _run_stream(model, mic, far)
        assert (offline.enhanced - enhanced).abs().max() < TOL


def test_streaming_refuses_training_mode():
    model = CAGCRN(GRID).train()
    with pytest.raises(RuntimeError, match="eval"):
        model.create_stream_state()
