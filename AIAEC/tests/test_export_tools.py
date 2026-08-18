"""Tensor-boundary tests for AIAEC export and ERB utilities."""

import soundfile as sf
import torch

from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid, stft
from AIAEC.Align_CRUSE.model import AlignCRUSE
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.CAGCRN.model import CAGCRN
from AIAEC.DeepVQE_S.model import DeepVQES
from AIAEC.CAGCRN.export_erb_matrix import extract_matrices
from AIAEC._calibration_common import blocks_from_pair
from AIAEC._export_common import (
    alignment_depth,
    dummy_inputs,
    make_export_wrapper,
    set_alignment_depth,
)


GRID = SignalGrid(16000, 512, 512, 256)
AEC_GRID = AecGrid(16000, 512, 512, 256)


def _models():
    torch.manual_seed(31)
    return (
        ('Align_CRUSE', AlignCRUSE(GRID)),
        ('Align_ULCNet', AlignULCNet(GRID)),
        ('DeepVQE_S', DeepVQES(GRID)),
        ('CAGCRN', CAGCRN(GRID)),
    )


def test_default_exports_only_required_control_outputs():
    for name, model in _models():
        model.eval()
        wrapper, names = make_export_wrapper(name, model)
        inputs, _ = dummy_inputs(name, model, 3)
        with torch.no_grad():
            outputs = wrapper(*inputs)
        assert len(names) == len(outputs) == 1
        assert all(torch.isfinite(output).all() for output in outputs)


def test_debug_output_switch_is_explicit():
    model = CAGCRN(GRID).eval()
    _, normal = make_export_wrapper('CAGCRN', model)
    _, debug = make_export_wrapper('CAGCRN', model, include_debug_outputs=True)
    assert normal == ('complex_mask_ri',)
    assert debug == ('complex_mask_ri', 'delay_distribution', 'cata_attention')


def test_alignment_depth_is_read_from_model():
    for name, model in _models():
        depth = alignment_depth(model)
        assert depth > 0


def test_alignment_depth_override_changes_no_weight_shape():
    model = AlignULCNet(GRID).eval()
    shapes = {name: tuple(value.shape)
              for name, value in model.state_dict().items()}
    old = set_alignment_depth(model, 8)
    assert old == 64
    assert alignment_depth(model) == model.max_delay_frames == 8
    assert shapes == {name: tuple(value.shape)
                      for name, value in model.state_dict().items()}


def test_erb_export_shapes_match_model_boundaries():
    forward, inverse, _ = extract_matrices(CAGCRN(GRID))
    assert forward.shape[0] == GRID.n_freqs
    assert inverse.shape == (forward.shape[1], GRID.n_freqs)
    assert torch.from_numpy(forward).isfinite().all()
    assert torch.from_numpy(inverse).isfinite().all()


def test_erb_export_values_match_each_model_codec():
    torch.manual_seed(32)
    model = CAGCRN(GRID)
    forward, inverse, _ = extract_matrices(model)
    bins = torch.rand(1, 3, 2, GRID.n_freqs)
    expected_features = model.erb.merge(bins)
    features = bins @ torch.from_numpy(forward)
    expected_expanded = model.erb.split(expected_features)
    expanded = expected_features @ torch.from_numpy(inverse)
    assert torch.allclose(features, expected_features, atol=1e-6)
    assert torch.allclose(expanded, expected_expanded, atol=1e-6)


def test_calibration_block_removes_only_the_runtime_batch_axis(tmp_path):
    waveform = torch.randn(GRID.sample_rate).numpy()
    primary = tmp_path / 'primary.wav'
    far = tmp_path / 'far.wav'
    sf.write(primary, waveform, GRID.sample_rate, subtype='FLOAT')
    sf.write(far, waveform, GRID.sample_rate, subtype='FLOAT')
    model = CAGCRN(GRID).eval()
    block = next(blocks_from_pair(
        'CAGCRN', model, AEC_GRID, primary, far, frames=4))
    assert block['mic'].shape == (4, GRID.n_freqs, 2)
    assert block['far'].shape == (4, GRID.n_freqs, 2)


def test_align_ulcnet_calibration_far_is_the_raw_input_waveform(tmp_path):
    samples = torch.arange(GRID.sample_rate, dtype=torch.float32)
    primary_wave = torch.zeros_like(samples)
    far_wave = 0.25 * torch.sin(2.0 * torch.pi * 733.0 * samples / GRID.sample_rate)
    primary = tmp_path / 'linear_error.wav'
    far = tmp_path / 'raw_far.wav'
    sf.write(primary, primary_wave.numpy(), GRID.sample_rate, subtype='FLOAT')
    sf.write(far, far_wave.numpy(), GRID.sample_rate, subtype='FLOAT')

    model = AlignULCNet(GRID).eval()
    block = next(blocks_from_pair(
        'Align_ULCNet', model, AEC_GRID, primary, far, frames=4
    ))
    expected = torch.view_as_real(
        stft(far_wave.unsqueeze(0), AEC_GRID).transpose(-2, -1)
    )[0, :4]
    torch.testing.assert_close(
        torch.from_numpy(block['far']), expected,
        rtol=0.0, atol=2e-6,
    )
