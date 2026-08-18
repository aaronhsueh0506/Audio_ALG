"""Explicit delta-state export contract for streaming Align-ULCNet."""

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet.export_onnx import (
    AlignUlcnetStreamingExport,
    host_output,
    stream_features,
    GRU_HIDDEN,
    GRU_LAYERS,
    COMPRESSION_EXPONENT,
    INPUT_NAMES,
    SIGNAL_INPUTS,
    MAX_DELAY_DEPTH,
    MIN_DELAY_DEPTH,
    OUTPUT_NAMES,
    SCORE_HISTORY_FRAMES,
    STATE_LAYOUT_VERSION,
    TA_BINS,
    TA_CHANNELS,
    _write_metadata,
    dummy_inputs,
    export_graph,
    next_state,
    state_shapes,
)
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.aiaec_common import SignalGrid
from AIAEC.training_common import FAR_INPUT_MODE_C_VALUES


GRID = SignalGrid(16000, 512, 512, 256)
D = 8


def _complex_frame(generator):
    return torch.complex(
        torch.randn(1, 1, GRID.n_freqs, generator=generator),
        torch.randn(1, 1, GRID.n_freqs, generator=generator),
    )


def _ri(value):
    return torch.stack((value.real, value.imag), dim=-1)


def test_streaming_export_shapes_are_fixed_and_delta_only():
    shapes = state_shapes(D)
    assert shapes['key_history'] == (1, 32, D - 1, 26)
    assert shapes['value_history'] == (1, 32, D - 1, 26)
    assert shapes['logit_history'] == (1, 32, 4, D)
    assert shapes['h_gru0'] == (2, 1, 128)
    assert INPUT_NAMES == (
        'error_mag', 'far_mag', 'error_cos', 'error_sin', 'error_ri',
        'key_history', 'value_history', 'logit_history', 'h_gru0', 'h_gru1',
    )
    assert OUTPUT_NAMES == (
        'output', 'key_now', 'value_now', 'logit_now',
        'h_gru0_out', 'h_gru1_out',
    )


def test_c_descriptor_constants_match_export_contract():
    header = (
        Path(__file__).resolve().parents[1]
        / 'Align_ULCNet' / 'ulcnet_model_io.h'
    ).read_text(encoding='utf-8')
    expected = {
        'ULCNET_MODEL_IO_LAYOUT_VERSION': STATE_LAYOUT_VERSION,
        'ULCNET_MODEL_IO_MIN_D': MIN_DELAY_DEPTH,
        'ULCNET_MODEL_IO_MAX_D': MAX_DELAY_DEPTH,
        'ULCNET_MODEL_IO_TA_CHANNELS': TA_CHANNELS,
        'ULCNET_MODEL_IO_TA_BINS': TA_BINS,
        'ULCNET_MODEL_IO_SCORE_HISTORY': SCORE_HISTORY_FRAMES,
        'ULCNET_MODEL_IO_GRU_LAYERS': GRU_LAYERS,
        'ULCNET_MODEL_IO_GRU_HIDDEN': GRU_HIDDEN,
    }
    for name, value in expected.items():
        match = re.search(
            r'^#define\s+%s\s+(\d+)u?\s*$' % re.escape(name),
            header,
            flags=re.MULTILINE,
        )
        assert match is not None, name
        assert int(match.group(1)) == value

    # The far-input enumeration is a two-sided contract: the exporter writes
    # the numeric value into the metadata, the descriptor carries the same
    # field, and both pipelines compare them. Pin the enumerators against the
    # single Python table rather than restating the numbers here.
    for name, value in FAR_INPUT_MODE_C_VALUES.items():
        enumerator = 'ULCNET_FAR_' + (
            'RAW' if name == 'raw_far' else name.replace('_far', '').upper()
        )
        match = re.search(
            r'^\s+%s\s*=\s*(\d+)\s*,?' % re.escape(enumerator),
            header,
            flags=re.MULTILINE,
        )
        assert match is not None, enumerator
        assert int(match.group(1)) == value, enumerator
    assert 'int far_input_mode;' in header

    # v5 moved the compression exponent out of the graph; the C define and
    # the exporter constant can now drift silently, so pin them here (the
    # exporter separately refuses checkpoints with any other exponent).
    match = re.search(
        r'^#define\s+ULCNET_MODEL_IO_COMPRESSION_EXP\s+([0-9.]+)f\s*$',
        header,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert float(match.group(1)) == COMPRESSION_EXPONENT


def test_export_refuses_foreign_compression_exponent(tmp_path):
    model = AlignULCNet(
        GRID, max_delay_frames=D, compression_exponent=0.5
    ).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    torch.save({'contract': {}, 'state_dict': model.state_dict()},
               checkpoint)
    with pytest.raises(ValueError, match='compression_exponent'):
        export_graph(model, str(checkpoint), str(tmp_path / 'graph.onnx'))


def test_metadata_separates_training_provenance_from_deployment(tmp_path):
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    checkpoint = tmp_path / 'ckpt.pt'
    checkpoint.write_bytes(b'not a real checkpoint, only hashed')
    inputs = dummy_inputs(D)
    outputs = AlignUlcnetStreamingExport(model).eval()(*inputs)

    metadata = _write_metadata(
        str(tmp_path / 'model.onnx'), str(checkpoint), model,
        {'far_input_mode': 'raw_far'}, inputs, outputs,
    )
    assert metadata['training_far_input_mode'] == 'raw_far'
    assert metadata['far_input_mode'] == 'aligned_far'
    assert metadata['far_input_mode_c_value'] == FAR_INPUT_MODE_C_VALUES[
        'aligned_far'
    ]
    assert metadata['state_layout_version'] == STATE_LAYOUT_VERSION

    # Legacy checkpoints retain raw-far provenance while deployment remains
    # fixed to aligned far.
    legacy = _write_metadata(
        str(tmp_path / 'legacy.onnx'), str(checkpoint), model, {},
        inputs, outputs,
    )
    assert legacy['training_far_input_mode'] == 'raw_far'
    assert legacy['far_input_mode'] == 'aligned_far'
    assert legacy['far_input_mode_c_value'] == 1


def test_delta_state_wrapper_matches_forward_stream_frame_by_frame():
    torch.manual_seed(20260816)
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    explicit = tuple(value.clone() for value in dummy_inputs(D)[SIGNAL_INPUTS:])
    reference = model.create_stream_state()
    generator = torch.Generator().manual_seed(73)

    with torch.no_grad():
        for _ in range(2 * D + 5):
            error = _complex_frame(generator)
            far = _complex_frame(generator)
            signals = stream_features(model, _ri(error), _ri(far))
            outputs = wrapper(*signals, *explicit)
            expected = model.forward_stream(error, far, reference)

            # The graph output is the COMPRESSED estimate; the host applies
            # the inverse signed power (host_output), completing the chain.
            assert torch.allclose(
                host_output(model, outputs[0]), _ri(expected.enhanced),
                atol=5e-7, rtol=1e-6
            )
            explicit = next_state(explicit, outputs, D)

            align = reference['align']
            assert torch.equal(
                explicit[0], align.key_ring._ring[:, :, :D - 1]
            )
            assert torch.equal(
                explicit[1], align.value_ring._ring[:, :, :D - 1]
            )
            assert torch.equal(
                explicit[2], align.score_cell._history
            )
            assert torch.equal(
                explicit[3], reference['subband_gru0']._hidden
            )
            assert torch.equal(
                explicit[4], reference['subband_gru1']._hidden
            )


def test_reset_is_all_zero_external_state():
    inputs = dummy_inputs(D)
    for state in inputs[SIGNAL_INPUTS:]:
        assert torch.count_nonzero(state) == 0


def test_streaming_onnx_runtime_matches_pytorch(tmp_path):
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    depth = 4
    torch.manual_seed(17)
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    initial = dummy_inputs(depth)
    output_path = tmp_path / 'align_ulcnet_stream.onnx'

    torch.onnx.export(
        wrapper,
        initial,
        str(output_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=17,
        do_constant_folding=True,
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(output_path), providers=['CPUExecutionProvider']
    )
    assert tuple(item.name for item in session.get_inputs()) == INPUT_NAMES
    assert tuple(item.name for item in session.get_outputs()) == OUTPUT_NAMES

    state = tuple(value.clone() for value in initial[SIGNAL_INPUTS:])
    generator = torch.Generator().manual_seed(23)
    worst = 0.0
    with torch.no_grad():
        for _ in range(2 * depth + 4):
            current = stream_features(
                model,
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
            ) + state
            expected = wrapper(*current)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(INPUT_NAMES, current)
            })
            for got, want in zip(actual, expected):
                worst = max(worst, float(np.max(
                    np.abs(got - want.numpy())
                )))
            state = next_state(state, expected, depth)
    assert worst <= 3e-4


def test_streaming_export_rejects_a_non_verified_grid():
    """The C boundary guard must fire before any shape can go wrong.

    SamFR width and the 257-bin dummies in this exporter are written for the
    16 kHz / 512-FFT boundary; a checkpoint from another grid has to be
    rejected here with an actionable message, never crash deep in a matmul.
    """
    model = AlignULCNet(SignalGrid(16000, 256, 256, 128),
                        max_delay_frames=D).eval()
    with pytest.raises(ValueError, match='16 kHz / FFT 512'):
        AlignUlcnetStreamingExport(model)
