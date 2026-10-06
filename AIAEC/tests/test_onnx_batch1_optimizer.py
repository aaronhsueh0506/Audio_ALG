"""The batch-1 ONNX cleanup: view merging, rank lowering, frequency-pad folding.

Every pass is a re-labelling of axes, so the optimised graph must reproduce the
original in onnxruntime EXACTLY (difference 0), not approximately.
"""

import copy

import numpy as np
import pytest
import torch

onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

import onnx_batch1_optimizer as b1  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _const(name, array):
    return numpy_helper.from_array(np.asarray(array), name=name)


def _model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(
        nodes, 'g',
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
         for n, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
         for n, s in outputs],
        list(initializers))
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8   # what the pinned onnxruntime loads
    return model


def _optimise(model, **kwargs):
    original = copy.deepcopy(model)
    report = b1.optimize_batch1(model, **kwargs)
    return original, report


def _ops(model):
    return [node.op_type for node in model.graph.node]


def test_view_chain_collapses_and_an_identity_chain_disappears():
    nodes = [
        helper.make_node('Unsqueeze', ['x', 'ax'], ['a']),
        helper.make_node('Squeeze', ['a', 'ax'], ['b']),
        helper.make_node('Unsqueeze', ['b', 'ax2'], ['c']),
        helper.make_node('Relu', ['c'], ['d']),
        helper.make_node('Squeeze', ['d', 'ax2'], ['y']),
    ]
    model = _model(nodes, [('x', [1, 3, 8])], [('y', [1, 3, 8])],
                   [_const('ax', np.asarray([1], np.int64)),
                    _const('ax2', np.asarray([2], np.int64))])
    original, report = _optimise(model)
    # Unsqueeze/Squeeze/Unsqueeze cancel to one view into Relu, and the
    # trailing Squeeze is the only other view left.
    assert _ops(model).count('Reshape') <= 2
    assert report['nodes_after'] < report['nodes_before']
    assert b1.verify_identical(original, model) == 0.0


def test_a_transpose_that_moves_only_unit_axes_is_a_view_but_a_real_one_is_not():
    unit = helper.make_node('Transpose', ['x'], ['y'], perm=[0, 2, 1, 3])
    real = helper.make_node('Transpose', ['x'], ['y'], perm=[0, 3, 1, 2])
    for node, shape_in, shape_out, expect in (
            (unit, [1, 1, 5, 4], [1, 5, 1, 4], False),
            (unit, [1, 3, 1, 4], [1, 1, 3, 4], False),
            (real, [1, 3, 5, 4], [1, 4, 3, 5], True)):
        model = _model([node], [('x', shape_in)], [('y', shape_out)])
        original, _ = _optimise(model)
        assert ('Transpose' in _ops(model)) == expect
        assert b1.verify_identical(original, model) == 0.0


def test_gather_on_a_unit_axis_becomes_a_view():
    nodes = [helper.make_node('Gather', ['x', 'i'], ['g'], axis=1),
             helper.make_node('Relu', ['g'], ['y'])]
    model = _model(nodes, [('x', [2, 1, 6])], [('y', [2, 6])],
                   [_const('i', np.asarray(0, np.int64))])
    original, _ = _optimise(model)
    assert 'Gather' not in _ops(model)
    assert b1.verify_identical(original, model) == 0.0


def test_rank_five_attention_pattern_is_lowered_to_rank_four():
    """Unsqueeze -> broadcast Mul -> ReduceSum, the shape of the DeepVQE and
    ULCNet attention."""
    nodes = [
        helper.make_node('Unsqueeze', ['q', 'a2'], ['q5']),    # [1,4,1,1,65]
        helper.make_node('Unsqueeze', ['k', 'a2'], ['k5']),    # [1,4,1,63,65]
        helper.make_node('Mul', ['q5', 'k5'], ['m']),
        helper.make_node('ReduceSum', ['m', 'ax'], ['s'], keepdims=0),
        helper.make_node('Softmax', ['s'], ['y'], axis=-1),
    ]
    model = _model(
        nodes, [('q', [1, 4, 1, 65]), ('k', [1, 4, 63, 65])],
        [('y', [1, 4, 1, 63])],
        [_const('a2', np.asarray([2], np.int64)),
         _const('ax', np.asarray([4], np.int64))])
    original, report = _optimise(model)
    assert report['high_rank_left'] == []
    assert report['blocked'] == []
    assert report['lowered'] >= 2
    assert b1.verify_identical(original, model) == 0.0


def test_grouped_linear_matmul_drops_its_leading_unit_axes():
    nodes = [
        helper.make_node('Reshape', ['x', 's5'], ['a']),       # [1,1,G,1,K]
        helper.make_node('MatMul', ['a', 'w'], ['m']),         # [1,1,G,1,N]
        helper.make_node('Reshape', ['m', 's3'], ['y']),
    ]
    weights = np.random.default_rng(1).standard_normal((16, 32, 8)).astype(
        np.float32)
    model = _model(
        nodes, [('x', [1, 1, 512])], [('y', [1, 1, 128])],
        [_const('s5', np.asarray([1, 1, 16, 1, 32], np.int64)),
         _const('s3', np.asarray([1, 1, 128], np.int64)),
         _const('w', weights)])
    original, report = _optimise(model)
    assert report['high_rank_left'] == []
    assert report['nodes_after'] <= report['nodes_before']
    assert b1.verify_identical(original, model) == 0.0


def test_an_op_it_cannot_lower_is_reported_and_left_correct():
    nodes = [helper.make_node('Unsqueeze', ['x', 'a'], ['u']),
             helper.make_node('Concat', ['u', 'u'], ['c'], axis=2),
             helper.make_node('ReduceSum', ['c', 'r'], ['y'], keepdims=0)]
    model = _model(nodes, [('x', [1, 4, 3, 5])], [('y', [1, 4, 3, 5])],
                   [_const('a', np.asarray([2], np.int64)),
                    _const('r', np.asarray([2], np.int64))])
    original, report = _optimise(model)
    assert any(op == 'Concat' for op, _ in report['blocked'])
    assert b1.verify_identical(original, model) == 0.0


def _pad_conv(pads, out_shape, conv_op='Conv'):
    weight = np.ones((1, 1, 1, 3), np.float32)
    nodes = [helper.make_node('Pad', ['x', 'p'], ['padded']),
             helper.make_node(conv_op, ['padded', 'w'], ['y'])]
    return _model(nodes, [('x', [1, 1, 2, 6])], [('y', out_shape)],
                  [_const('p', np.asarray(pads, np.int64)),
                   _const('w', weight)])


def test_frequency_pad_folds_into_conv_and_temporal_pad_does_not():
    folded = _pad_conv([0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 2, 6])
    original, report = _optimise(folded)
    assert report['pads_folded'] == 1 and 'Pad' not in _ops(folded)
    assert b1.verify_identical(original, folded) == 0.0

    temporal = _pad_conv([0, 0, 1, 0, 0, 0, 1, 0], [1, 1, 4, 4])
    _, report = _optimise(temporal)
    assert report['pads_folded'] == 0 and 'Pad' in _ops(temporal)

    transposed = _pad_conv([0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 2, 10],
                           conv_op='ConvTranspose')
    _, report = _optimise(transposed)
    assert report['pads_folded'] == 0


def test_the_check_can_fail_a_real_transpose_treated_as_a_view(monkeypatch):
    """Mutation: if a real Transpose were merged as a view the exact compare
    must catch it."""
    node = helper.make_node('Transpose', ['x'], ['t'], perm=[0, 3, 1, 2])
    model = _model([node, helper.make_node('Relu', ['t'], ['y'])],
                   [('x', [1, 3, 5, 4])], [('y', [1, 4, 3, 5])])
    original = copy.deepcopy(model)
    real_is_view = b1._is_view
    monkeypatch.setattr(
        b1, '_is_view',
        lambda info, n: real_is_view(info, n) or n.op_type == 'Transpose')
    b1.optimize_batch1(model)
    assert b1.verify_identical(original, model) != 0.0


def test_a_reshape_that_copies_an_input_dimension_is_not_rewired():
    """With a dynamic batch the second Reshape's 0 means "copy my input's first
    dimension"; reading through the first Reshape would change it."""
    nodes = [helper.make_node('Reshape', ['x', 's1'], ['a']),
             helper.make_node('Reshape', ['a', 's2'], ['y'])]
    model = _model(nodes, [('x', ['N', 2, 3])], [('y', ['M', 12])],
                   [_const('s1', np.asarray([2, -1, 3], np.int64)),
                    _const('s2', np.asarray([0, -1], np.int64))])
    b1.optimize_batch1(model)
    assert [list(n.input)[0] for n in model.graph.node] == ['x', 'a']


def test_subgraphs_are_refused_and_a_wrong_pass_is_caught_by_verify(
        monkeypatch):
    body = helper.make_graph(
        [helper.make_node('Relu', ['x'], ['r'])], 'body', [],
        [helper.make_tensor_value_info('r', TensorProto.FLOAT, [1])])
    branch = helper.make_node('If', ['c'], ['y'], then_branch=body,
                              else_branch=body)
    model = _model([branch], [('c', [1])], [('y', [1])])
    with pytest.raises(ValueError, match='subgraphs'):
        b1.optimize_batch1(model)

    node = helper.make_node('Transpose', ['x'], ['t'], perm=[0, 3, 1, 2])
    model = _model([node, helper.make_node('Relu', ['t'], ['y'])],
                   [('x', [1, 3, 5, 4])], [('y', [1, 4, 3, 5])])
    real_is_view = b1._is_view
    monkeypatch.setattr(
        b1, '_is_view',
        lambda info, n: real_is_view(info, n) or n.op_type == 'Transpose')
    with pytest.raises(RuntimeError, match='changed the outputs'):
        b1.optimize_batch1(model, verify=True)


def test_graph_boundary_and_metadata_survive():
    node = helper.make_node('Relu', ['x'], ['y'])
    model = _model([node], [('x', [1, 4])], [('y', [1, 4])])
    model.metadata_props.add(key='k', value='v')
    b1.optimize_batch1(model)
    assert [(p.key, p.value) for p in model.metadata_props] == [('k', 'v')]
    assert [i.name for i in model.graph.input] == ['x']


# ---- the real exported graphs ------------------------------------------


def _fold_constants(path):
    """The exporters' own onnxruntime constant folding, without the batch-1
    pass, so the tool can be run on its input."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(path) + '.fold'
    options.log_severity_level = 3
    ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    return onnx.load(str(path) + '.fold')


def _assert_exact_rank_four(wrapper, inputs, names, outputs, tmp_path):
    """Export, constant-fold, clean up: no rank > 4 tensor and no blocked op
    left, fewer nodes, and outputs identical to the folded graph."""
    path = tmp_path / 'raw.onnx'
    torch.onnx.export(wrapper, inputs, str(path), input_names=names,
                      output_names=outputs, opset_version=17,
                      do_constant_folding=True)
    model = _fold_constants(path)
    original = copy.deepcopy(model)
    report = b1.optimize_batch1(model)
    assert report['high_rank_left'] == [] and report['blocked'] == []
    assert report['nodes_after'] < report['nodes_before']
    assert b1.verify_identical(original, model) == 0.0


def test_deepvqe_graph_is_exactly_preserved_and_rank_four(tmp_path):
    from AIAEC.aiaec_common import SignalGrid
    from AIAEC.DeepVQE_S import DeepVQES
    from AIAEC._streaming_export import _build

    torch.manual_seed(97)
    wrapper, inputs, names, outputs, _ = _build(
        'DeepVQE_S', DeepVQES(SignalGrid(16000, 512, 512, 256)).eval())
    _assert_exact_rank_four(wrapper, inputs, names, outputs, tmp_path)


def test_ulcnet_graph_is_exactly_preserved_and_rank_four(tmp_path):
    from AIAEC.aiaec_common import SignalGrid
    from AIAEC.Align_ULCNet.model import AlignULCNet
    from AIAEC.Align_ULCNet.export_onnx import (
        AlignUlcnetStreamingExport, OUTPUT_NAMES, INPUT_NAMES, dummy_inputs)

    torch.manual_seed(98)
    grid = SignalGrid(16000, 512, 512, 256)
    model_ = AlignULCNet(grid, max_delay_frames=8).eval()
    wrapper = AlignUlcnetStreamingExport(model_).eval()
    inputs = dummy_inputs(8, grid.n_freqs, 26)
    _assert_exact_rank_four(
        wrapper, inputs, INPUT_NAMES, OUTPUT_NAMES, tmp_path)
