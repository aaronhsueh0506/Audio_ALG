"""Tracked GTCRN explicit-state wrapper must replay the offline model."""

import pytest
import torch

from AINR.GTCRN.model import GTCRN
from AINR.GTCRN.stream_model import StreamGTCRN, initial_inputs


def _run_stream(model, spectrum, poison_conv=False):
    stream = StreamGTCRN(model).eval()
    _, conv, tra, inter = initial_inputs(model)
    frames = []
    with torch.no_grad():
        for index in range(spectrum.shape[2]):
            output, conv, tra, inter = stream(
                spectrum[:, :, index:index + 1], conv, tra, inter)
            frames.append(output)
            if poison_conv:
                conv = torch.roll(conv, 1, dims=3)
    return torch.cat(frames, dim=2), (conv, tra, inter)


# The cache extents are identical on both grids: everything behind the ERB
# split sees erb_subband_1 + erb_subband_2 = 129 features regardless of
# n_fft, so only the spectrum width changes.
@pytest.mark.parametrize('nfft,bins', [(512, 257), (256, 129)])
def test_stream_wrapper_matches_offline_random_weights(nfft, bins):
    torch.manual_seed(41)
    model = GTCRN(65, 64, nfft=nfft, fs=16000).eval()
    assert initial_inputs(model)[0].shape == (1, bins, 1, 2)
    spectrum = torch.randn(1, bins, 12, 2)
    with torch.no_grad():
        offline = model(spectrum)
    online, (conv, tra, inter) = _run_stream(model, spectrum)
    assert (offline - online).abs().max().item() < 2e-5
    assert conv.shape == (2, 1, 16, 16, 33)
    assert tra.shape == (2, 3, 1, 1, 16)
    assert inter.shape == (2, 1, 33, 16)


def test_equivalence_gate_fails_on_a_mispositioned_cache():
    """The comparison above must bite: a cache rolled by one slot between
    frames has to break parity, or the gate proves nothing."""
    torch.manual_seed(41)
    model = GTCRN(65, 64, nfft=512, fs=16000).eval()
    spectrum = torch.randn(1, 257, 12, 2)
    with torch.no_grad():
        offline = model(spectrum)
    online, _ = _run_stream(model, spectrum, poison_conv=True)
    assert (offline - online).abs().max().item() > 2e-5


def test_cache_outputs_are_live_state_not_constant_zeros():
    model = GTCRN(65, 64, nfft=512, fs=16000).eval()
    stream = StreamGTCRN(model).eval()
    spectrum, conv, tra, inter = initial_inputs(model)
    with torch.no_grad():
        _, conv_out, tra_out, inter_out = stream(
            spectrum, conv, tra, inter)
    assert conv_out.abs().max().item() > 0
    assert tra_out.abs().max().item() > 0
    assert inter_out.abs().max().item() > 0
