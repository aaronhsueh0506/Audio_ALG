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


from model import GTCRN, SFE  # noqa: E402
from stream_model import (  # noqa: E402
    CombinedStateGTCRN,
    StreamGTCRN,
    _FrequencyNeighborhood,
    initial_inputs,
    pack_state,
)
from export_onnx import (  # noqa: E402
    COMBINED_STATE_LAYOUT_VERSION,
    INPUT_NAMES,
    OUTPUT_NAMES,
    SIGNAL_INPUT_NAMES,
    STATE_LAYOUTS,
    STATE_LAYOUT_VERSION,
    build_metadata,
    build_stream_model,
    export_graph,
    layout_of,
    resolve_state_layout,
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


@pytest.mark.parametrize('channels', (3, 8))
def test_grouped_conv_sfe_matches_training_unfold(channels):
    """The deployment lowering must preserve SFE values and channel order."""
    generator = torch.Generator().manual_seed(20260819 + channels)
    value = torch.randn(2, channels, 3, 17, generator=generator)
    expected = SFE()(value)
    actual = _FrequencyNeighborhood(channels)(value)
    assert torch.equal(actual, expected)


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
    assert metadata['erb_boundary'] == 'host_prepost'
    header_version = c_macro('GTCRN_MODEL_LAYOUT_VERSION')
    assert metadata['state_layout_version'] == header_version
    assert STATE_LAYOUT_VERSION == metadata['state_layout_version']
    # gtcrn_process.h's prose reserves the combined layout's number. Stated as
    # an assertion so a C-side bump onto it fails here rather than shipping a
    # header that means two different things to two different graphs.
    assert header_version != COMBINED_STATE_LAYOUT_VERSION, (
        'GTCRN_MODEL_LAYOUT_VERSION %d collides with the combined layout'
        % header_version
    )


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
    conv = [name for name in INPUT_NAMES if name.startswith('conv_')]
    assert conv == [
        'conv_enc0', 'conv_enc1', 'conv_enc2',
        'conv_dec0', 'conv_dec1', 'conv_dec2',
    ]
    times = [
        c_macro('GTCRN_MODEL_CONV_TIME_0'),
        c_macro('GTCRN_MODEL_CONV_TIME_1'),
        c_macro('GTCRN_MODEL_CONV_TIME_2'),
    ]
    for index, name in enumerate(conv[:3]):
        assert schema[name] == [
            1, c_macro('GTCRN_MODEL_CONV_CHANNELS'), times[index],
            c_macro('GTCRN_MODEL_CONV_FREQ'),
        ], name
    for index, name in enumerate(conv[3:]):
        assert schema[name] == [
            1, c_macro('GTCRN_MODEL_CONV_CHANNELS'), times[2 - index],
            c_macro('GTCRN_MODEL_CONV_FREQ'),
        ], name
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
    inputs0 = initial_inputs(stream.model)
    signals = tuple(inputs0[:3])
    state = list(inputs0[3:])
    with torch.no_grad():
        for _ in range(2):
            frame_inputs = signals + tuple(state)
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


def _checkpoint(tmp_path, n_fft=512):
    """A throwaway checkpoint on the grid ``_grid_config`` describes."""
    path = os.fspath(tmp_path / ('gtcrn_%d.pth' % n_fft))
    torch.save(
        {'state_dict': GTCRN(65, 64, nfft=n_fft, fs=16000).state_dict()},
        path,
    )
    return path


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
    checkpoint = _checkpoint(tmp_path, 256)
    stream, grid = build_stream_model(_grid_config(tmp_path, 256), checkpoint)
    assert grid['n_fft'] == 256
    assert initial_inputs(stream.model)[0].shape == (1, 129, 1)

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

    checkpoint = _checkpoint(tmp_path)
    stream, grid = build_stream_model(_grid_config(tmp_path, 512), checkpoint)
    graph_path = os.fspath(tmp_path / 'gtcrn_static.onnx')
    export_graph(stream, grid, checkpoint, graph_path)
    graph = onnx.load(graph_path)
    op_types = [node.op_type for node in graph.graph.node]
    assert 'Gather' not in op_types
    # Leave optimizer-version headroom while still catching the former
    # 323-node Unfold/packed-state graph.
    assert len(op_types) <= 300
    for value in (list(graph.graph.input) + list(graph.graph.output)
                  + list(graph.graph.value_info)):
        for dim in value.type.tensor_type.shape.dim:
            assert dim.HasField('dim_value'), value.name


def test_runtime_erb_bins_match_the_constructor(tmp_path):
    """The exported .bin matrices must equal the constructor's buffers.

    The C host consumes caller-loaded pointers in these exact layouts
    (forward bin-major, inverse band-major, float32 LE); the deployment
    loader owns the files, so this round-trip is the single value pin.
    """
    # The flat names may belong to a sibling model by the time this test
    # runs; re-bind them to THIS model's files exactly like the file top.
    for stale in ('train', 'inference', 'model', 'checkpoint_utils', 'export_onnx',
                  'stream_model', 'export_erb_matrix'):
        sys.modules.pop(stale, None)
    sys.path.insert(0, ROOT)
    from export_erb_matrix import write_runtime_bins

    out = write_runtime_bins(os.path.join(ROOT, 'config.ini'),
                             os.fspath(tmp_path))
    fwd = np.fromfile(os.path.join(out, 'erb_fwd.bin'), '<f4')
    inv = np.fromfile(os.path.join(out, 'erb_inv.bin'), '<f4')
    erb = GTCRN(65, 64, nfft=512, fs=16000).erb
    ref_fwd = erb.erb_fc.weight.detach().numpy().T.astype(np.float32).ravel()
    ref_inv = erb.ierb_fc.weight.detach().numpy().T.astype(np.float32).ravel()
    assert fwd.shape == ref_fwd.shape and inv.shape == ref_inv.shape
    assert np.array_equal(fwd, ref_fwd)
    assert np.array_equal(inv, ref_inv)


def test_combined_state_groups_the_sixteen_slots_by_shape():
    """The combined boundary must regroup state without reshaping it.

    Every group is concatenated along an axis its members already have -- the
    convolution histories along depth, the hiddens along their layer axis --
    so no tensor gains a dimension and the four-dimensional ceiling holds.
    """
    layout = resolve_state_layout('combined')
    assert layout.state_names == ('conv_cache', 'h_tra', 'h_dpgrnn')
    assert layout.input_names == ('mag', 'real', 'imag') + layout.state_names
    assert layout.output_names == (
        'output', 'conv_cache_out', 'h_tra_out', 'h_dpgrnn_out')
    assert layout.layout_version == COMBINED_STATE_LAYOUT_VERSION
    assert layout.layout_version != STATE_LAYOUT_VERSION, (
        'a different state layout must not claim the shipped version'
    )

    model = _stream_model().model
    packed = layout.graph_inputs(model)
    assert len(packed) == len(layout.input_names)
    split = initial_inputs(model)
    assert [tuple(value.shape) for value in packed[:3]] == [
        tuple(value.shape) for value in split[:3]
    ], 'the feature inputs are untouched by the state regrouping'
    conv_cache, h_tra, h_dpgrnn = packed[len(SIGNAL_INPUT_NAMES):]
    # Two rows (encoder, decoder) x channels x summed depth x frequency.
    assert conv_cache.shape == (2, 16, 16, 33)
    assert h_tra.shape == (6, 1, 16)
    assert h_dpgrnn.shape == (4, 33, 8)
    for tensor in packed[len(SIGNAL_INPUT_NAMES):]:
        assert tensor.dim() <= 4
    assert sum(tensor.numel() for tensor in packed[len(SIGNAL_INPUT_NAMES):]) == sum(
        tensor.numel() for tensor in split[len(SIGNAL_INPUT_NAMES):]
    ), 'regrouping must not change the number of state values'


def test_combined_and_split_state_layouts_compute_the_same_frames():
    """Both boundaries wrap one streaming graph, so they must agree exactly."""
    torch.manual_seed(19)
    split = _stream_model()
    model = split.model
    combined = CombinedStateGTCRN(model).eval()
    assert layout_of(split) is STATE_LAYOUTS['split']
    assert layout_of(combined) is STATE_LAYOUTS['combined']

    start = initial_inputs(model)
    bands = start[0].shape[1]
    split_state = start[len(SIGNAL_INPUT_NAMES):]
    combined_state = pack_state(model, split_state)
    generator = torch.Generator().manual_seed(23)
    observed_nonzero = False
    with torch.no_grad():
        for _ in range(5):
            signals = (
                torch.randn(1, bands, 1, generator=generator).abs(),
                torch.randn(1, bands, 1, generator=generator),
                torch.randn(1, bands, 1, generator=generator),
            )
            expected = split(*(signals + split_state))
            actual = combined(*(signals + combined_state))
            assert torch.equal(actual[0], expected[0])
            split_state = expected[1:]
            combined_state = actual[1:]
            for packed, reference in zip(combined_state,
                                         pack_state(model, split_state)):
                assert torch.equal(packed, reference)
            observed_nonzero = observed_nonzero or bool(
                combined_state[0].abs().max() > 0)
    # Written so it can FAIL: zero state would satisfy every comparison above.
    assert observed_nonzero, 'state never left its zero initialisation'


def test_combined_state_graph_exports_and_matches_the_runtime(tmp_path):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    checkpoint = _checkpoint(tmp_path)
    stream, grid = build_stream_model(
        _grid_config(tmp_path, 512), checkpoint, state_layout='combined')
    path = tmp_path / 'gtcrn_combined.onnx'
    metadata = export_graph(stream, grid, checkpoint, os.fspath(path),
                            verify=True)
    assert metadata['state_layout'] == 'combined'
    assert metadata['state_layout_version'] == COMBINED_STATE_LAYOUT_VERSION
    graph = onnx.load(os.fspath(path))
    declared = [entry.name for entry in graph.graph.input]
    assert declared == list(STATE_LAYOUTS['combined'].input_names)
    for name in INPUT_NAMES[len(SIGNAL_INPUT_NAMES):]:
        assert name not in declared, (
            'combined layout must not also declare %s' % name
        )
