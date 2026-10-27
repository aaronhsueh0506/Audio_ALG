import json
from pathlib import Path

import numpy as np
import pytest
import torch

from calibration_io import (
    calibration_report_path,
    capture_calibration_inputs,
    sibling_onnx_path,
    write_calibration_artifact,
)


def test_capture_preserves_complete_onnx_input_shape(tmp_path):
    captured = {}
    graph_input = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    capture_calibration_inputs(captured, ('feature',), (graph_input,))
    capture_calibration_inputs(captured, ('feature',), (graph_input + 1,))

    arrays = {'feature': np.stack(captured['feature'])}
    assert arrays['feature'].shape == (2, 1, 3, 4)

    output = tmp_path / 'calib'
    write_calibration_artifact(
        output, arrays, {'frames': 2}, 'bin'
    )
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['binary_tensors']['feature']['frame_shape'] == [1, 3, 4]
    first = np.fromfile(output / 'feature' / 'feature_0000.bin', '<f4')
    assert np.array_equal(first.reshape(1, 3, 4), graph_input.numpy())


def test_capture_rejects_nonfinite_before_partial_append():
    captured = {'a': [], 'b': []}
    with pytest.raises(RuntimeError, match='non-finite.*b'):
        capture_calibration_inputs(
            captured, ('a', 'b'),
            (torch.zeros(1, 1), torch.tensor([[float('nan')]])),
        )
    assert captured == {'a': [], 'b': []}


def test_capture_rejects_schema_length_mismatch():
    with pytest.raises(ValueError, match='length mismatch'):
        capture_calibration_inputs({}, ('a',), ())


def test_npz_report_does_not_take_the_graph_sidecar_name(tmp_path):
    """``foo.npz`` pairs with ``foo.onnx``, whose exporter writes ``foo.json``.

    The calibration report must land elsewhere, or the later of the two
    writers silently replaces the other's file.
    """
    output = tmp_path / 'foo.npz'
    graph_sidecar = Path(sibling_onnx_path(output)).with_suffix('.json')
    assert graph_sidecar == tmp_path / 'foo.json'
    graph_sidecar.write_text('{"graph": true}\n')

    arrays = {'feature': np.zeros((2, 1, 3), dtype=np.float32)}
    report_path = write_calibration_artifact(
        output, arrays, {'frames': 2}, 'npz'
    )

    assert report_path == str(tmp_path / 'foo.calib.json')
    assert report_path == calibration_report_path(output, 'npz')
    assert graph_sidecar.read_text() == '{"graph": true}\n'
    assert json.loads((tmp_path / 'foo.calib.json').read_text())['frames'] == 2


def test_bin_report_stays_inside_its_directory(tmp_path):
    output = tmp_path / 'foo'
    arrays = {'feature': np.zeros((1, 1, 3), dtype=np.float32)}
    report_path = write_calibration_artifact(
        output, arrays, {'frames': 1}, 'bin'
    )
    assert report_path == str(output / 'manifest.json')
    assert report_path == calibration_report_path(output, 'bin')
