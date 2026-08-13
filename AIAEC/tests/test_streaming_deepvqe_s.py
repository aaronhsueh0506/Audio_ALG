"""Frame-by-frame streaming equivalence for DeepVQE-S.

The model is random-weight but eval-mode, so BatchNorm uses running stats and
the offline forward is the exact reference for ``forward_stream``.
"""

import torch

from AIAEC.DeepVQE_S.model import DeepVQES
from AIAEC.aiaec_common import SignalGrid

GRID = SignalGrid(16000, 512, 512, 256)
FRAMES = 96  # > max_delay_frames (63): the delay rings wrap during the test

# Measured on CPU float32: enhanced 1.5e-6, ccm_taps 4.5e-7, delay 3.7e-9
# (seed 0/1, 96 frames).  Pinned with ~10x headroom; the only arithmetic
# divergence is the GRU's batched-vs-stepped kernel path.
TOL_ENHANCED = 2e-5
TOL_AUX = 5e-6


def _model(seed: int = 0) -> DeepVQES:
    torch.manual_seed(seed)
    model = DeepVQES(GRID)
    model.eval()
    return model


def _random_pair(seed: int, frames: int = FRAMES, batch: int = 1):
    g = torch.Generator().manual_seed(seed)
    shape = (batch, frames, GRID.n_freqs)
    def spec():
        return torch.complex(torch.randn(*shape, generator=g),
                             torch.randn(*shape, generator=g))
    return spec(), spec()


def _run_offline(model, mic, far):
    with torch.no_grad():
        return model(microphone=mic, far_end=far)


def _run_stream(model, mic, far):
    state = model.create_stream_state()
    enhanced, delays, taps = [], [], []
    with torch.no_grad():
        for t in range(mic.shape[1]):
            out = model.forward_stream(mic[:, t:t + 1], far[:, t:t + 1], state)
            enhanced.append(out.enhanced)
            delays.append(out.delay_distribution)
            taps.append(out.auxiliary["ccm_taps"])
    assert model.stream_output_delay == 0  # frames align 1:1 with offline
    return (torch.cat(enhanced, dim=1), torch.cat(delays, dim=1),
            torch.cat(taps, dim=1))


def test_stream_matches_offline():
    model = _model()
    mic, far = _random_pair(1)
    off = _run_offline(model, mic, far)
    enhanced, delays, taps = _run_stream(model, mic, far)
    assert (enhanced - off.enhanced).abs().max().item() < TOL_ENHANCED
    assert (delays - off.delay_distribution).abs().max().item() < TOL_AUX
    assert (taps - off.auxiliary["ccm_taps"]).abs().max().item() < TOL_AUX


def test_comparison_can_fail_on_shifted_far():
    # Streaming a one-frame-late far-end against the unshifted offline run
    # must produce a visible difference, or the equivalence test is vacuous.
    model = _model()
    mic, far = _random_pair(2)
    off = _run_offline(model, mic, far)
    far_late = torch.cat((torch.zeros_like(far[:, :1]), far[:, :-1]), dim=1)
    enhanced, _, _ = _run_stream(model, mic, far_late)
    assert (enhanced - off.enhanced).abs().max().item() > 1e-3


def test_fresh_states_do_not_cross_contaminate():
    model = _model()
    pairs = [_random_pair(3), _random_pair(4)]
    for mic, far in pairs:
        off = _run_offline(model, mic, far)
        enhanced, _, _ = _run_stream(model, mic, far)
        assert (enhanced - off.enhanced).abs().max().item() < TOL_ENHANCED
