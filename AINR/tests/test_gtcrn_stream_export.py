"""Tracked GTCRN explicit-state wrapper must replay the offline model."""

import torch

from AINR.GTCRN.model import GTCRN
from AINR.GTCRN.stream_model import StreamGTCRN, initial_inputs


def test_stream_wrapper_matches_offline_random_weights():
    torch.manual_seed(41)
    model = GTCRN(65, 64, nfft=512, fs=16000).eval()
    stream = StreamGTCRN(model).eval()
    spectrum = torch.randn(1, 257, 12, 2)
    with torch.no_grad():
        offline = model(spectrum)
        _, conv, tra, inter = initial_inputs()
        frames = []
        for index in range(spectrum.shape[2]):
            output, conv, tra, inter = stream(
                spectrum[:, :, index:index + 1], conv, tra, inter)
            frames.append(output)
        online = torch.cat(frames, dim=2)
    assert (offline - online).abs().max().item() < 2e-5
    assert conv.shape == (2, 1, 16, 16, 33)
    assert tra.shape == (2, 3, 1, 1, 16)
    assert inter.shape == (2, 1, 33, 16)


def test_cache_outputs_are_live_state_not_constant_zeros():
    model = GTCRN(65, 64, nfft=512, fs=16000).eval()
    stream = StreamGTCRN(model).eval()
    spectrum, conv, tra, inter = initial_inputs()
    with torch.no_grad():
        _, conv_out, tra_out, inter_out = stream(
            spectrum, conv, tra, inter)
    assert conv_out.abs().max().item() > 0
    assert tra_out.abs().max().item() > 0
    assert inter_out.abs().max().item() > 0
