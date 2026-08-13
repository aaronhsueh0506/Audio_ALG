"""Frame-by-frame streaming equivalence for GTCRN-AENR.

The streaming decomposition ports the upstream GTCRN stream reference onto the
shared aiaec_streaming cells: per-block depthwise-conv input rings (2/4/10
frames), TRA attention GRU hiddens and DPGRNN inter GRU hiddens.  The decoder's
stride-1 depthwise transposed convs stream through an equivalent causal conv
with a flipped kernel, so tolerances are float-accumulation noise, not layout
error: measured max-abs 1.1e-6 over 96 random frames, pinned at 2e-5.
"""

import pytest
import torch

from AIAEC.GTCRN_AENR import GTCRNAENR
from AIAEC.aiaec_common import SignalGrid

GRID = SignalGrid(16000, 512, 512, 256)
TOL = 2e-5


def _spec(batch, frames, bins, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(batch, frames, bins, generator=gen),
                         torch.randn(batch, frames, bins, generator=gen))


def _model(seed=0):
    torch.manual_seed(seed)
    return GTCRNAENR(GRID).eval()


def _run_stream(model, err, far):
    state = model.create_stream_state()
    enhanced, masks, erb = [], [], []
    with torch.no_grad():
        for t in range(err.shape[1]):
            out = model.forward_stream(err[:, t:t + 1], far[:, t:t + 1], state)
            enhanced.append(out.enhanced)
            masks.append(out.mask)
            erb.append(out.auxiliary["erb_complex_mask"])
    return (torch.cat(enhanced, dim=1), torch.cat(masks, dim=1),
            torch.cat(erb, dim=2))


def test_stream_matches_offline():
    model = _model()
    assert model.stream_output_delay == 0
    err = _spec(1, 96, GRID.n_freqs, 1)
    far = _spec(1, 96, GRID.n_freqs, 2)
    with torch.no_grad():
        ref = model(linear_error=err, far_end=far)
    enh, mask, erb = _run_stream(model, err, far)
    assert (enh - ref.enhanced).abs().max().item() < TOL
    assert (mask - ref.mask).abs().max().item() < TOL
    assert (erb - ref.auxiliary["erb_complex_mask"]).abs().max().item() < TOL


def test_shifted_far_end_diverges():
    # CAN-FAIL check: a one-frame far-end shift must be visible, otherwise the
    # equivalence assertions above compare nothing.
    model = _model()
    err = _spec(1, 90, GRID.n_freqs, 3)
    far = _spec(1, 90, GRID.n_freqs, 4)
    with torch.no_grad():
        ref = model(linear_error=err, far_end=far)
    far_shifted = torch.cat((torch.zeros_like(far[:, :1]), far[:, :-1]), dim=1)
    enh, _, _ = _run_stream(model, err, far_shifted)
    assert (enh - ref.enhanced).abs().max().item() > 1e-3


def test_fresh_state_per_utterance():
    # Two utterances through two fresh states must each match their own
    # offline run -- no cross-contamination through module-held state.
    model = _model()
    for seed, frames in ((10, 48), (20, 56)):
        err = _spec(1, frames, GRID.n_freqs, seed)
        far = _spec(1, frames, GRID.n_freqs, seed + 1)
        with torch.no_grad():
            ref = model(linear_error=err, far_end=far)
        enh, _, _ = _run_stream(model, err, far)
        assert (enh - ref.enhanced).abs().max().item() < TOL


def test_streaming_refuses_training_mode():
    model = _model().train()
    with pytest.raises(RuntimeError):
        model.create_stream_state()
