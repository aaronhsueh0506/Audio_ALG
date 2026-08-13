"""Frame-by-frame streaming equivalence for Align-CRUSE.

The offline forward is the reference; forward_stream must reproduce it from
explicit state.  The two intended nonequivalences (paper_global alignment,
utterances shorter than the delay search window) are pinned by tests of their
own rather than hidden behind loose tolerances.
"""

import pytest
import torch

from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.aiaec_common import SignalGrid

G16 = SignalGrid(16000, 512, 512, 256)

# Measured max-abs stream-vs-offline difference on the reference run was
# 4.9e-7 (enhanced), 1.8e-7 (mask), 3.3e-7 (delay); pinned with ~10x headroom.
ATOL = 5e-6


def _spec(batch, frames, bins, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(batch, frames, bins, generator=generator),
        torch.randn(batch, frames, bins, generator=generator),
    )


def _make_model(seed=0):
    torch.manual_seed(seed)
    return AlignCRUSE(G16).eval()


def _stream(model, mic, far):
    state = model.create_stream_state()
    enhanced, mask, delay = [], [], []
    with torch.no_grad():
        for t in range(mic.shape[1]):
            out = model.forward_stream(mic[:, t:t + 1], far[:, t:t + 1], state)
            enhanced.append(out.enhanced)
            mask.append(out.mask)
            delay.append(out.delay_distribution)
    return (torch.cat(enhanced, dim=1), torch.cat(mask, dim=1),
            torch.cat(delay, dim=1))


def test_stream_matches_offline_on_long_utterance():
    model = _make_model()
    frames = 90
    assert frames >= model.max_delay_frames  # equivalence regime (T >= D)
    assert model.stream_output_delay == 0    # outputs align frame-for-frame
    mic = _spec(1, frames, G16.n_freqs, 1)
    far = _spec(1, frames, G16.n_freqs, 2)
    with torch.no_grad():
        offline = model(mic, far)
    enhanced, mask, delay = _stream(model, mic, far)
    assert (enhanced - offline.enhanced).abs().max() <= ATOL
    assert (mask - offline.mask).abs().max() <= ATOL
    assert (delay - offline.delay_distribution).abs().max() <= ATOL


def test_shifted_far_stream_differs_from_offline():
    """CAN-FAIL check: the comparison has teeth against a one-frame slip."""
    model = _make_model()
    frames = 90
    mic = _spec(1, frames, G16.n_freqs, 1)
    far = _spec(1, frames, G16.n_freqs, 2)
    with torch.no_grad():
        offline = model(mic, far)
    enhanced, _, _ = _stream(model, mic, torch.roll(far, 1, dims=1))
    assert (enhanced - offline.enhanced).abs().max() > 1e-3


def test_fresh_states_do_not_cross_contaminate():
    model = _make_model()
    frames = 70
    mic_a = _spec(1, frames, G16.n_freqs, 11)
    far_a = _spec(1, frames, G16.n_freqs, 12)
    mic_b = _spec(1, frames, G16.n_freqs, 13)
    far_b = _spec(1, frames, G16.n_freqs, 14)
    with torch.no_grad():
        offline_a = model(mic_a, far_a)
        offline_b = model(mic_b, far_b)
    enhanced_a, _, _ = _stream(model, mic_a, far_a)
    enhanced_b, _, _ = _stream(model, mic_b, far_b)
    assert (enhanced_a - offline_a.enhanced).abs().max() <= ATOL
    assert (enhanced_b - offline_b.enhanced).abs().max() <= ATOL


def test_paper_global_alignment_cannot_stream():
    torch.manual_seed(0)
    model = AlignCRUSE(G16, alignment_mode="paper_global").eval()
    with pytest.raises(ValueError, match="paper_global"):
        model.create_stream_state()


def test_short_utterance_below_delay_window_differs():
    """Documented caveat: offline masks unobservable delays with the FINAL
    utterance length; a stream cannot, so T < max_delay_frames diverges."""
    model = _make_model()
    frames = 30
    assert frames < model.max_delay_frames
    mic = _spec(1, frames, G16.n_freqs, 3)
    far = _spec(1, frames, G16.n_freqs, 4)
    with torch.no_grad():
        offline = model(mic, far)
    enhanced, _, _ = _stream(model, mic, far)
    assert (enhanced - offline.enhanced).abs().max() > 1e-3
