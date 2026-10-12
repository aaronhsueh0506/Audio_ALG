"""Gates for the in-place state boundary validator.

``validate_inplace_state_graph`` is what keeps a state tensor bindable to one
address for input and output. Every rule it states has a graph here that
breaks only that rule, and the well-formed graph passes, so a rule that
stopped being checked would turn one of these red.
"""

import json
import os
import sys

import pytest

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_streaming_contract import (  # noqa: E402
    STATE_BINDING,
    validate_inplace_state_from_metadata,
    validate_inplace_state_graph,
)
import check_state_quantization as quant  # noqa: E402

K = 5
HANDOFF = {'win': 'win_out', 'rows': 'rows_out'}
WINDOWS = {'win': 2}
SLICES = {'rows': 1}


def _ints(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def build(window_shape=(1, 2, K, 3), out_window_shape=None,
          rows_shape=(1, 5, 1, 4), rows_out_shape=None,
          start=1, end=K, step=1, axis=2, extra_window_user=None,
          rows=((0, 1), (1, 3), (3, 5)), rows_axis=1, head_slice=False,
          state_slice=False, binding=STATE_BINDING, window_user_op='Slice',
          row_user_extra=False, dynamic_window=False):
    """A small graph shaped like the exporters' state handling."""
    frames = window_shape[2]
    nodes = []
    initializers = [
        _ints('w_start', [start]), _ints('w_end', [end]),
        _ints('w_axis', [axis]), _ints('w_step', [step]),
        _ints('zero', [0]),
    ]
    new_shape = list(window_shape)
    new_shape[2] = 1
    if window_user_op == 'Slice':
        nodes.append(helper.make_node(
            'Slice', ['win', 'w_start', 'w_end', 'w_axis', 'w_step'], ['hist'],
            name='drop_oldest'))
    else:
        nodes.append(helper.make_node('Identity', ['win'], ['hist']))
    appended = 'win_long' if state_slice else 'win_out'
    nodes.append(helper.make_node('Concat', ['hist', 'new'], [appended],
                                  axis=2, name='append'))
    if state_slice:
        initializers += [_ints('s_s', [0]), _ints('s_e', [window_shape[2]]),
                         _ints('s_a', [2])]
        nodes.append(helper.make_node(
            'Slice', ['win_long', 's_s', 's_e', 's_a'], ['win_out']))
    if extra_window_user:
        nodes.append(helper.make_node('Neg', ['win'], ['unused_neg']))
    pieces = []
    for index, (lo, hi) in enumerate(rows):
        initializers += [_ints('r%d_s' % index, [lo]),
                         _ints('r%d_e' % index, [hi]),
                         _ints('r%d_a' % index, [rows_axis])]
        nodes.append(helper.make_node(
            'Slice', ['rows', 'r%d_s' % index, 'r%d_e' % index,
                      'r%d_a' % index], ['piece%d' % index]))
        pieces.append('piece%d' % index)
    if row_user_extra:
        nodes.append(helper.make_node('Neg', ['rows'], ['unused_rows']))
    nodes.append(helper.make_node('Concat', pieces, ['rows_out'], axis=1))
    graph_outputs = [
        helper.make_tensor_value_info(
            'win_out', TensorProto.FLOAT,
            list(out_window_shape or window_shape)),
        helper.make_tensor_value_info(
            'rows_out', TensorProto.FLOAT,
            list(rows_out_shape or rows_shape)),
    ]
    if head_slice:
        initializers += [_ints('o_s', [0]), _ints('o_e', [1]),
                         _ints('o_a', [2])]
        nodes.append(helper.make_node(
            'Slice', ['win_out', 'o_s', 'o_e', 'o_a'], ['head']))
        graph_outputs.append(helper.make_tensor_value_info(
            'head', TensorProto.FLOAT, [window_shape[0], window_shape[1], 1,
                                        window_shape[3]]))
    win_dims = list(window_shape)
    if dynamic_window:
        win_dims[2] = 'T'
    graph = helper.make_graph(
        nodes, 'g',
        [helper.make_tensor_value_info('win', TensorProto.FLOAT, win_dims),
         helper.make_tensor_value_info('new', TensorProto.FLOAT, new_shape),
         helper.make_tensor_value_info('rows', TensorProto.FLOAT,
                                       list(rows_shape))],
        graph_outputs, initializer=initializers)
    del frames
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 17)])
    props = {
        'state_handoff': json.dumps(HANDOFF),
        'state_slicing': json.dumps({
            'win': {'kind': 'window', 'axis': 2},
            'rows': {'kind': 'rows', 'axis': 1},
        }),
    }
    if binding is not None:
        props['state_binding'] = binding
    helper.set_model_props(model, props)
    return model


def check(model):
    return validate_inplace_state_graph(model, HANDOFF, windows=WINDOWS,
                                        slices=SLICES)


def test_well_formed_graph_passes():
    assert check(build()) == 2


@pytest.mark.parametrize('kwargs, message', [
    (dict(extra_window_user=True), 'exactly one consumer'),
    (dict(window_user_op='Identity'), 'exactly one consumer'),
    (dict(end=2 ** 31 - 1), 'explicit range'),
    (dict(end=4), 'does not drop the oldest'),
    (dict(start=0), 'does not drop the oldest'),
    (dict(step=2), 'step 1'),
    (dict(step=-1), 'step 1'),
    (dict(axis=1), 'axis 2'),
    (dict(out_window_shape=(1, 2, K - 1, 3)), 'changes shape'),
    (dict(rows_out_shape=(1, 5, 2, 4)), 'changes shape'),
    (dict(rows_shape=(5, 1, 4), rows_out_shape=(5, 1, 4)), 'rank'),
    (dict(state_slice=True), 'produced by a Slice'),
    (dict(binding=None), 'state_binding'),
    (dict(binding='ping_pong'), 'state_binding'),
    (dict(rows=((0, 1), (1, 3))), 'tile'),
    (dict(rows=((0, 2), (1, 3), (3, 5))), 'tile'),
    (dict(rows_axis=0), 'axis 1'),
    (dict(row_user_extra=True), 'only by Slice'),
    (dict(dynamic_window=True), 'non-static'),
])
def test_each_rule_has_a_graph_that_breaks_only_it(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        check(build(**kwargs))


def test_a_head_output_may_be_a_slice():
    """Only state outputs are restricted; a head cut from a state by a Slice
    does not break the in-place binding."""
    assert check(build(head_slice=True)) == 2


def test_a_missing_state_pair_is_refused():
    with pytest.raises(RuntimeError, match='not a graph input and output'):
        validate_inplace_state_graph(
            build(), {'win': 'nope'}, windows=WINDOWS, slices=SLICES)


def test_attribute_style_slice_is_read_too():
    """Opset < 10 Slice carries starts/ends/axes as attributes."""
    model = build()
    graph = model.graph
    old = next(node for node in graph.node if node.name == 'drop_oldest')
    new = helper.make_node('Slice', ['win'], ['hist'], name='drop_oldest',
                           starts=[1], ends=[K], axes=[2])
    index = list(graph.node).index(old)
    del graph.node[index]
    graph.node.insert(index, new)
    assert check(model) == 2


def test_newest_first_window_needs_the_either_order():
    """A ring that keeps slots 0..K-2 is a valid in-place window only when
    the caller allows either end to be dropped."""
    model = build(start=0, end=K - 1)
    with pytest.raises(RuntimeError, match='does not drop the oldest'):
        validate_inplace_state_graph(model, HANDOFF, windows=WINDOWS,
                                     slices=SLICES)
    assert validate_inplace_state_graph(
        model, HANDOFF, windows=WINDOWS, slices=SLICES,
        window_drop='either') == 2
    with pytest.raises(RuntimeError, match='drops exactly one'):
        validate_inplace_state_graph(
            build(start=0, end=K), HANDOFF, windows=WINDOWS, slices=SLICES,
            window_drop='either')


def test_window_and_row_names_must_be_state_inputs():
    with pytest.raises(RuntimeError, match='not state inputs'):
        validate_inplace_state_graph(
            build(), {'win': 'win_out'}, windows=WINDOWS, slices=SLICES)


def test_metadata_entry_point_runs_the_same_check():
    assert validate_inplace_state_from_metadata(build()) == 2
    with pytest.raises(RuntimeError, match='tile'):
        validate_inplace_state_from_metadata(build(rows=((0, 1), (1, 3))))
    model = build()
    kept = [prop for prop in model.metadata_props
            if prop.key != 'state_slicing']
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    with pytest.raises(RuntimeError, match='state_slicing'):
        validate_inplace_state_from_metadata(model)


def _report(overrides=None):
    entry = {'dtype': 'int8', 'scale': 0.0125, 'zero_point': 0}
    tensors = {
        'win': dict(entry, shape=[1, 2, K, 3]),
        'win_out': dict(entry, shape=[1, 2, K, 3]),
        'rows': dict(entry, scale=[0.1, 0.2], zero_point=[0, 0],
                     shape=[1, 5, 1, 4]),
        'rows_out': dict(entry, scale=[0.1, 0.2], zero_point=[0, 0],
                         shape=[1, 5, 1, 4]),
    }
    for (name, field), value in (overrides or {}).items():
        tensors[name][field] = value
    return tensors


def test_quantized_pairs_that_agree_pass():
    assert quant.check_quantized_state_pairs(HANDOFF, _report()) == []


@pytest.mark.parametrize('override', [
    {('win_out', 'scale'): 0.025},
    {('win_out', 'zero_point'): 3},
    {('win_out', 'dtype'): 'uint8'},
    {('win_out', 'shape'): [1, 2, K - 1, 3]},
    {('rows_out', 'scale'): [0.1, 0.25]},
    {('rows', 'scale'): 0.1},
])
def test_each_quantized_property_that_differs_is_reported(override):
    problems = quant.check_quantized_state_pairs(HANDOFF, _report(override))
    assert len(problems) == 1, problems


def test_a_state_tensor_missing_from_the_report_is_reported():
    tensors = _report()
    del tensors['rows_out']
    problems = quant.check_quantized_state_pairs(HANDOFF, tensors)
    assert problems and 'missing' in problems[0]
    tensors = _report()
    del tensors['win']['zero_point']
    assert 'lacks zero_point' in quant.check_quantized_state_pairs(
        HANDOFF, tensors)[0]


def test_command_line_exit_codes(tmp_path, capsys):
    graph = tmp_path / 'g.onnx'
    onnx.save(build(), str(graph))
    good = tmp_path / 'good.json'
    good.write_text(json.dumps({'tensors': _report()}))
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'tensors': _report(
        {('win_out', 'scale'): 0.5})}))
    broken = tmp_path / 'broken.json'
    broken.write_text('{"tensors": ')
    assert quant.main([str(graph)]) == 0
    assert 'NOT_RUN' in capsys.readouterr().out
    assert quant.main([str(graph), '--report', str(good)]) == 0
    assert quant.main([str(graph), '--report', str(bad)]) == 1
    assert 'scale differs' in capsys.readouterr().out
    assert quant.main([str(graph), '--report', str(broken)]) == 2
    bad_graph = tmp_path / 'bad_graph.onnx'
    onnx.save(build(window_user_op='Identity'), str(bad_graph))
    assert quant.main([str(bad_graph), '--report', str(good)]) == 1


@pytest.mark.parametrize('scale', [float('nan'), float('inf'), 0.0, -0.5])
def test_an_unusable_scale_is_reported_even_when_both_sides_agree(scale):
    tensors = _report({('win', 'scale'): scale, ('win_out', 'scale'): scale})
    problems = quant.check_quantized_state_pairs(HANDOFF, tensors)
    assert any('finite positive' in item for item in problems), problems


def test_a_non_numeric_zero_point_is_reported():
    tensors = _report({('win', 'zero_point'): 'x',
                       ('win_out', 'zero_point'): 'x'})
    problems = quant.check_quantized_state_pairs(HANDOFF, tensors)
    assert any('zero_point' in item for item in problems), problems


@pytest.mark.parametrize('damage', ['bad_json', 'missing_axis', 'bad_kind'])
def test_malformed_state_metadata_fails_the_graph_check(tmp_path, capsys,
                                                       damage):
    model = build()
    slicing = {'win': {'kind': 'window', 'axis': 2},
               'rows': {'kind': 'rows', 'axis': 1}}
    if damage == 'missing_axis':
        del slicing['win']['axis']
    if damage == 'bad_kind':
        slicing['win']['kind'] = 'ring'
    text = '{"win": ' if damage == 'bad_json' else json.dumps(slicing)
    for prop in model.metadata_props:
        if prop.key == 'state_slicing':
            prop.value = text
    graph = tmp_path / 'g.onnx'
    onnx.save(model, str(graph))
    assert quant.main([str(graph)]) == 1
    assert 'FAIL graph' in capsys.readouterr().out
