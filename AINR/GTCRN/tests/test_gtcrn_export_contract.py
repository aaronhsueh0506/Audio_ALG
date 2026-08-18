"""The GTCRN export metadata and gtcrn_process.h are one state contract."""

import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = pathlib.Path(ROOT)
sys.path.insert(0, ROOT)

# Each model project has its own top-level ``train.py``/``model.py``/
# ``export_onnx.py``.  Under a single pytest session the first one imported
# wins ``sys.modules``, so dropping the cached entries is what makes this file
# exercise GTCRN's code rather than a sibling project's.
for _stale in ('train', 'inference', 'model', 'checkpoint_utils', 'export_onnx'):
    sys.modules.pop(_stale, None)


from model import GTCRN  # noqa: E402
from stream_model import StreamGTCRN, initial_inputs  # noqa: E402
from export_onnx import (  # noqa: E402
    INPUT_NAMES,
    OUTPUT_NAMES,
    STATE_LAYOUT_VERSION,
    build_metadata,
    build_stream_model,
)


GRID = {'sr': 16000, 'n_fft': 512, 'win_len': 512, 'hop_len': 256}


def c_macro(name):
    header = (ROOT_PATH / 'gtcrn_process.h').read_text(encoding='utf-8')
    match = re.search(
        r'^#define\s+%s\s+(\d+)u?\s*(?:/\*.*)?$' % re.escape(name),
        header,
        flags=re.MULTILINE,
    )
    assert match is not None, name
    return int(match.group(1))


def _stream_model():
    torch.manual_seed(43)
    return StreamGTCRN(GTCRN(65, 64, nfft=512, fs=16000).eval()).eval()


def _metadata(tmp_path):
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'not a real checkpoint, only hashed')
    stream = _stream_model()
    inputs = initial_inputs(stream.model)
    with torch.no_grad():
        outputs = stream(*(tensor.clone() for tensor in inputs))
    return build_metadata(str(checkpoint), GRID, inputs, outputs)


def test_state_layout_version_is_pinned_to_the_c_header(tmp_path):
    """A board reads this out of the graph to decide whether its
    ``GTCRNModelState`` still matches. Asserting the Python constant alone
    would not catch the metadata key being dropped, so this goes through the
    same builder ``main`` uses.
    """
    metadata = _metadata(tmp_path)
    assert metadata['state_layout_version'] == c_macro(
        'GTCRN_MODEL_LAYOUT_VERSION'
    )
    assert STATE_LAYOUT_VERSION == metadata['state_layout_version']


def test_input_schema_shapes_match_the_c_cache_struct(tmp_path):
    """The schema must come from the real tensors, not a typed-in string.

    gtcrn_process.h sizes its three caches from these extents. While the
    schema was a hand-written literal, a stream-model cache-shape change would
    have left the metadata describing the old graph and nothing would have
    disagreed.
    """
    metadata = _metadata(tmp_path)
    schema = metadata['input_schema']
    assert set(schema) == set(INPUT_NAMES)
    assert schema['conv_cache'] == [
        c_macro('GTCRN_MODEL_CONV_SIDES'),
        c_macro('GTCRN_MODEL_CONV_CHANNELS'),
        c_macro('GTCRN_MODEL_CONV_TIME'),
        c_macro('GTCRN_MODEL_CONV_FREQ'),
    ]
    h_tra = [name for name in INPUT_NAMES if name.startswith('h_tra_')]
    assert len(h_tra) == c_macro('GTCRN_MODEL_TRA_GRUS')
    for name in h_tra:
        assert schema[name] == [
            1, 1, c_macro('GTCRN_MODEL_TRA_HIDDEN'),
        ], name
    h_dpgrnn = [name for name in INPUT_NAMES if name.startswith('h_dpgrnn')]
    assert len(h_dpgrnn) == c_macro('GTCRN_MODEL_DPGRNN_GRUS')
    for name in h_dpgrnn:
        assert schema[name] == [
            1, c_macro('GTCRN_MODEL_DPGRNN_FREQ'),
            c_macro('GTCRN_MODEL_DPGRNN_HIDDEN'),
        ], name

    # Every state output must hand back exactly the shape its input slot
    # expects, or the caller cannot copy one into the other.
    output_schema = metadata['output_schema']
    assert set(output_schema) == set(OUTPUT_NAMES)
    for state_in, state_out in metadata['state_handoff'].items():
        assert output_schema[state_out] == schema[state_in]


def test_calibration_frame_shapes_equal_the_exported_graph_inputs(tmp_path):
    """One recorded calibration frame must BE one graph invocation.

    ``capture_calibration_inputs`` keeps the graph's batch dimension and
    ``np.stack`` adds the calibration-frame axis on top of it.  Dropping the
    batch axis would leave every ``.bin`` one rank short of the input the
    accelerator binds it to, and nothing downstream re-derives the shape --
    the manifest is what a quantizer reads.  So the per-frame shapes are
    compared against the ONNX graph this exporter really produces rather than
    against a literal that could be edited to match a regression.
    """
    onnx = pytest.importorskip('onnx')
    from calibration_io import (
        capture_calibration_inputs,
        write_calibration_artifact,
    )

    stream = _stream_model()
    inputs = initial_inputs(stream.model)
    graph_path = os.fspath(tmp_path / 'gtcrn_stream.onnx')
    torch.onnx.export(
        stream, tuple(tensor.clone() for tensor in inputs), graph_path,
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        opset_version=17, do_constant_folding=True,
    )
    graph_shapes = {
        value.name: [int(dim.dim_value)
                     for dim in value.type.tensor_type.shape.dim]
        for value in onnx.load(graph_path).graph.input
    }
    assert set(graph_shapes) == set(INPUT_NAMES)

    # Two invocations, recorded exactly the way calibration_main records them.
    captured = {}
    first, *state = initial_inputs(stream.model)
    with torch.no_grad():
        for _ in range(2):
            frame_inputs = (first,) + tuple(state)
            capture_calibration_inputs(captured, INPUT_NAMES, frame_inputs)
            state = list(stream(*frame_inputs)[1:])
    arrays = {name: np.stack(values).astype(np.float32, copy=False)
              for name, values in captured.items()}
    artifact = tmp_path / 'calib'
    write_calibration_artifact(artifact, arrays, {'frames': 2}, 'bin')
    manifest = json.loads((artifact / 'manifest.json').read_text())

    for name in INPUT_NAMES:
        assert manifest['binary_tensors'][name]['frame_shape'] == (
            graph_shapes[name]
        ), name
        # And the bytes on disk really hold one whole graph input.
        blob = np.fromfile(artifact / name / ('%s_0000.bin' % name), '<f4')
        assert blob.size == int(np.prod(graph_shapes[name])), name


def _grid_config(tmp_path, n_fft):
    config = tmp_path / 'config.ini'
    config.write_text(
        '[signal]\nsr = 16000\nn_fft = %d\nwin_len = %d\nhop_len = %d\n'
        '[model]\nerb_subband_1 = 65\nerb_subband_2 = 64\n'
        % (n_fft, n_fft, n_fft // 2),
        encoding='utf-8',
    )
    return os.fspath(config)


def test_build_stream_model_follows_the_config_grid(tmp_path):
    """Shapes derive from the config's training grid, not from a default.

    A checkpoint from a different grid must fail loudly at strict
    ``load_state_dict`` (the ERB matrices change extent with n_fft), never
    deep inside a matmul on mismatched dummy inputs.
    """
    checkpoint = os.fspath(tmp_path / 'gtcrn_256.pth')
    torch.save(
        {'state_dict': GTCRN(65, 64, nfft=256, fs=16000).state_dict()},
        checkpoint,
    )
    stream, grid = build_stream_model(_grid_config(tmp_path, 256), checkpoint)
    assert grid['n_fft'] == 256
    assert initial_inputs(stream.model)[0].shape == (1, 129, 1, 2)

    with pytest.raises(RuntimeError, match='size mismatch'):
        build_stream_model(_grid_config(tmp_path, 512), checkpoint)


def test_exported_graph_declares_fully_static_io(tmp_path):
    """Every declared graph input AND output dim must be a concrete integer.

    Shape inference leaves symbolic dim_params after GRU/Slice/Concat even
    though static inputs make every output extent fixed; accelerator
    toolchains reject symbolic I/O, so the exporter pins the declared output
    shapes to the traced tensors' real shapes.
    """
    onnx = pytest.importorskip('onnx')
    from export_onnx import export_graph

    checkpoint = os.fspath(tmp_path / 'ckpt.pth')
    torch.save(
        {'state_dict': GTCRN(65, 64, nfft=512, fs=16000).state_dict()},
        checkpoint,
    )
    stream, grid = build_stream_model(_grid_config(tmp_path, 512), checkpoint)
    graph_path = os.fspath(tmp_path / 'gtcrn_static.onnx')
    export_graph(stream, grid, checkpoint, graph_path)
    graph = onnx.load(graph_path)
    for value in list(graph.graph.input) + list(graph.graph.output):
        for dim in value.type.tensor_type.shape.dim:
            assert dim.HasField('dim_value'), value.name
