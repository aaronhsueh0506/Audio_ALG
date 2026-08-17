import json

import numpy as np
import pytest
import torch

from calibration_io import (
    capture_calibration_inputs,
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
    first = np.fromfile(output / 'feature' / 'feature_1.bin', '<f4')
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
