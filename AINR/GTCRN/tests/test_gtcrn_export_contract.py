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
    StreamGTCRN,
    _FrequencyNeighborhood,
    initial_inputs,
)
from export_onnx import (  # noqa: E402
    CONV_STATE_NAMES,
    GRU_STATE_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    RETIRED_LAYOUT_VERSIONS,
    SIGNAL_INPUT_NAMES,
    STATE_BINDING,
    STATE_HANDOFF,
    STATE_LAYOUT_VERSION,
    build_metadata,
    build_stream_model,
    export_graph,
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
    # Retired numbers are never reused: a C-side bump onto one must fail here
    # rather than ship a header that means two different things to two
    # different graphs.
    assert RETIRED_LAYOUT_VERSIONS == frozenset({3, 4, 5, 6, 7})
    assert header_version not in RETIRED_LAYOUT_VERSIONS, (
        'GTCRN_MODEL_LAYOUT_VERSION %d collides with a retired layout'
        % header_version
    )
    assert header_version == 8
    assert metadata['state_binding'] == STATE_BINDING == 'in_place_same_address'
    assert metadata['state_handoff'] == STATE_HANDOFF


def test_input_schema_shapes_match_the_c_cache_struct(tmp_path):
    """The schema must come from the real tensors, not a typed-in string.

    gtcrn_process.h sizes its state from these extents. With a hand-written
    literal, a stream-model cache-shape change would leave the metadata
    describing the old graph and nothing would disagree.
    """
    metadata = _metadata(tmp_path)
    schema = metadata['input_schema']
    assert set(schema) == set(INPUT_NAMES)
    conv = list(CONV_STATE_NAMES)
    assert conv == [
        'conv_enc0', 'conv_enc1', 'conv_enc2',
        'conv_dec0', 'conv_dec1', 'conv_dec2',
    ]
    windows = [
        c_macro('GTCRN_MODEL_CONV_WINDOW_0'),
        c_macro('GTCRN_MODEL_CONV_WINDOW_1'),
        c_macro('GTCRN_MODEL_CONV_WINDOW_2'),
    ]
    for index, name in enumerate(conv[:3]):
        assert schema[name] == [
            1, c_macro('GTCRN_MODEL_CONV_CHANNELS'), windows[index],
            c_macro('GTCRN_MODEL_CONV_FREQ'),
        ], name
    for index, name in enumerate(conv[3:]):
        assert schema[name] == [
            1, c_macro('GTCRN_MODEL_CONV_CHANNELS'), windows[2 - index],
            c_macro('GTCRN_MODEL_CONV_FREQ'),
        ], name
    assert list(GRU_STATE_NAMES) == ['h_tra', 'h_dpgrnn']
    assert schema['h_tra'] == [
        1, c_macro('GTCRN_MODEL_TRA_GRUS'), 1,
        c_macro('GTCRN_MODEL_TRA_HIDDEN'),
    ]
    assert schema['h_dpgrnn'] == [
        1, c_macro('GTCRN_MODEL_DPGRNN_GRUS'),
        c_macro('GTCRN_MODEL_DPGRNN_FREQ'),
        c_macro('GTCRN_MODEL_DPGRNN_HIDDEN'),
    ]

    # Every state output must hand back exactly the shape its input slot
    # expects, or the caller cannot copy one into the other.
    output_schema = metadata['output_schema']
    assert set(output_schema) == set(OUTPUT_NAMES)
    for state_in, state_out in metadata['state_handoff'].items():
        assert output_schema[state_out] == schema[state_in]
        assert len(schema[state_in]) == 4, state_in


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
    # Leave optimizer-version headroom while still catching an
    # Unfold/packed-state graph (323 nodes).
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


def test_state_rows_follow_the_graph_order():
    """Each row of ``h_tra`` / ``h_dpgrnn`` must reach its own GRU.

    The recurrent state of block i feeds block i's attention and everything
    computed after it, never anything before it, so perturbing row i moves row
    i and possibly later rows, never an earlier one. The two DPGRNN inter GRUs
    of one layer run in parallel: neither sees the other's hidden state. A
    slice landing on the wrong GRU, or a reversed row order, breaks this.
    """
    torch.manual_seed(31)
    stream = _stream_model()
    start = initial_inputs(stream.model)
    signals = tuple(torch.randn_like(value) for value in start[:3])
    windows = tuple(torch.randn_like(value) for value in start[3:9])
    base_tra = torch.randn_like(start[9])
    base_dp = torch.randn_like(start[10])

    def run(tra, dp):
        with torch.no_grad():
            return stream(*signals, *windows, tra, dp)

    reference = run(base_tra, base_dp)

    def moved(output_index, rows_changed):
        return [row for row in range(reference[output_index].shape[1])
                if not torch.equal(reference[output_index][0, row],
                                   rows_changed[output_index][0, row])]

    for row in range(6):
        bumped = base_tra.clone()
        bumped[:, row] += 0.5
        rows = moved(len(OUTPUT_NAMES) - 2, run(bumped, base_dp))
        assert row in rows and min(rows) == row, (row, rows)
    for row, forbidden in ((0, {1}), (1, {0}), (2, {0, 1, 3}),
                           (3, {0, 1, 2})):
        bumped = base_dp.clone()
        bumped[:, row] += 0.5
        rows = moved(len(OUTPUT_NAMES) - 1, run(base_tra, bumped))
        assert row in rows and not (set(rows) & forbidden), (row, rows)


def test_stream_replay_keeps_state_windows_overlapping():
    """The window the graph returns is the next call's input, unchanged: the
    newest four frames of one window are the oldest four of the next."""
    torch.manual_seed(37)
    stream = _stream_model()
    state = list(initial_inputs(stream.model))
    generator = torch.Generator().manual_seed(5)
    previous = None
    with torch.no_grad():
        for _ in range(14):
            signals = [torch.randn(1, 129, 1, generator=generator).abs(),
                       torch.randn(1, 129, 1, generator=generator),
                       torch.randn(1, 129, 1, generator=generator)]
            outputs = stream(*signals, *state[3:])
            windows = outputs[1:7]
            if previous is not None:
                for before, after in zip(previous, windows):
                    assert torch.equal(before[:, :, 1:], after[:, :, :-1])
            previous = windows
            state[3:] = list(outputs[1:])
    assert previous is not None and float(previous[0].abs().max()) > 0.0


def test_exported_state_boundary_is_in_place_and_sliced_on_the_input(tmp_path):
    """What the board binds: rank-4 state pairs of equal shape, every window
    consumed only by one explicit Slice, the stacked GRU states only by Slices
    on dim 1 that tile them, and no graph output produced by a Slice."""
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from onnx import numpy_helper

    checkpoint = _checkpoint(tmp_path)
    stream, grid = build_stream_model(_grid_config(tmp_path, 512), checkpoint)
    path = os.fspath(tmp_path / 'gtcrn_boundary.onnx')
    metadata = export_graph(stream, grid, checkpoint, path, verify=True)
    assert metadata['state_binding'] == 'in_place_same_address'
    graph = onnx.load(path)
    shapes = {value.name: [dim.dim_value
                           for dim in value.type.tensor_type.shape.dim]
              for value in list(graph.graph.input) + list(graph.graph.output)}
    for state, back in STATE_HANDOFF.items():
        assert len(shapes[state]) == 4
        assert shapes[state] == shapes[back], (state, back)
    assert shapes['h_tra'] == [1, 6, 1, 16]
    assert shapes['h_dpgrnn'] == [1, 4, 33, 8]
    constants = {item.name: numpy_helper.to_array(item).tolist()
                 for item in graph.graph.initializer}
    for name in CONV_STATE_NAMES:
        users = [node for node in graph.graph.node if name in node.input]
        assert [node.op_type for node in users] == ['Slice'], name
        starts, ends, axes, steps = (constants[x] for x in users[0].input[1:])
        assert (starts, ends, axes, steps) == (
            [1], [shapes[name][2]], [2], [1]), name
    for name, rows in (('h_tra', 6), ('h_dpgrnn', 4)):
        users = [node for node in graph.graph.node if name in node.input]
        assert len(users) == rows
        assert all(node.op_type == 'Slice' for node in users)
        assert sorted(constants[node.input[1]][0] for node in users) == (
            list(range(rows)))
    produced = {out: node for node in graph.graph.node
                for out in node.output}
    for output in graph.graph.output:
        assert produced[output.name].op_type != 'Slice', output.name


def _drop_the_other_end(model, name):
    """Point the Slice that reads window ``name`` at the opposite end of the
    window, keeping its length: [1:K) becomes [0:K-1) and back."""
    from onnx import numpy_helper

    constants = {item.name: numpy_helper.to_array(item)
                 for item in model.graph.initializer}
    node = next(item for item in model.graph.node if name in item.input)
    start, end = (int(constants[node.input[index]].reshape(-1)[0])
                  for index in (1, 2))
    new = (0, end - 1) if start == 1 else (1, end + 1)
    for index, value in ((1, new[0]), (2, new[1])):
        label = 'flipped_%s_%d' % (name, index)
        model.graph.initializer.append(numpy_helper.from_array(
            np.array([value], dtype=np.int64), label))
        node.input[index] = label
    return model


def test_a_window_that_drops_the_wrong_end_is_refused(tmp_path):
    """Every conv window is time ordered: the graph drops its first frame, the
    metadata says so, and a graph (or a declaration) with the other end is
    refused by the validator the exporter ran."""
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from onnx_streaming_contract import validate_inplace_state_from_metadata

    checkpoint = _checkpoint(tmp_path)
    stream, grid = build_stream_model(_grid_config(tmp_path, 512), checkpoint)
    path = os.fspath(tmp_path / 'gtcrn_direction.onnx')
    export_graph(stream, grid, checkpoint, path)
    graph = onnx.load(path)
    props = {prop.key: prop.value for prop in graph.metadata_props}
    slicing = json.loads(props['state_slicing'])
    assert {name: slicing[name]['drop'] for name in CONV_STATE_NAMES} == {
        name: 'first' for name in CONV_STATE_NAMES}
    assert validate_inplace_state_from_metadata(graph) == len(STATE_HANDOFF)
    for name in CONV_STATE_NAMES:
        mutant = _drop_the_other_end(onnx.load(path), name)
        with pytest.raises(RuntimeError, match='does not drop the first slot'):
            validate_inplace_state_from_metadata(mutant)
        mutant = onnx.load(path)
        for prop in mutant.metadata_props:
            if prop.key == 'state_slicing':
                declared = json.loads(prop.value)
                declared[name]['drop'] = 'last'
                prop.value = json.dumps(declared)
        with pytest.raises(RuntimeError, match='does not drop the last slot'):
            validate_inplace_state_from_metadata(mutant)
