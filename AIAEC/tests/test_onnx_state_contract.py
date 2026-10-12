"""validate_inplace_state_contract on hand-built graphs.

The exporters call the validator on every file they write, so a gate that
could not fail would let a regressed boundary ship. Each violation below is
built on its own, on top of a graph that passes, and must be named.
"""

import numpy as np
import pytest

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from AIAEC._onnx_contract import (  # noqa: E402
    STATE_BINDING,
    validate_inplace_state_contract,
)

INT64_MAX = 2 ** 63 - 1


def _value(name, shape, elem=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, elem, list(shape))


def _ints(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def _model(*, window=(1, 2, 4, 3), start=1, end=4, axes=(2,), steps=(1,),
           extra_consumer=None, out_shape=None, out_producer=None,
           binding=STATE_BINDING, rank=4, dynamic_starts=False,
           out_is_input=False):
    shape = tuple(window)[:rank] if rank != 4 else tuple(window)
    frame = list(shape)
    frame[2 if rank == 4 else 0] = 1
    nodes = []
    initializers = [_ints('starts', [start]), _ints('ends', [end]),
                    _ints('axes', list(axes))]
    slice_inputs = ['w', 'starts', 'ends', 'axes']
    if steps is not None:
        initializers.append(_ints('steps', list(steps)))
        slice_inputs.append('steps')
    inputs = [_value('w', shape), _value('x', frame)]
    if dynamic_starts:
        inputs.append(_value('dyn_starts', [1], TensorProto.INT64))
        slice_inputs[1] = 'dyn_starts'
    nodes.append(helper.make_node('Slice', slice_inputs, ['h']))
    time_axis = 2 if rank == 4 else 0
    nodes.append(helper.make_node('Concat', ['h', 'x'], ['joined'],
                                  axis=time_axis))
    outputs = [_value('w_out', out_shape or shape)]
    if out_producer == 'slice':
        # The state output is a Slice of a longer tensor: forbidden.
        nodes.append(helper.make_node(
            'Concat', ['w', 'x'], ['longer'], axis=time_axis))
        initializers += [_ints('s2', [1]), _ints('e2', [shape[time_axis] + 1]),
                         _ints('a2', [time_axis])]
        nodes.append(helper.make_node(
            'Slice', ['longer', 's2', 'e2', 'a2'], ['w_out']))
    elif out_is_input:
        nodes.append(helper.make_node('Identity', ['joined'], ['sink']))
        outputs = [_value('w', shape)]
    else:
        nodes.append(helper.make_node('Identity', ['joined'], ['w_out']))
    if extra_consumer == 'tap':
        nodes.append(helper.make_node('Identity', ['w'], ['tap']))
        outputs.append(_value('tap', shape))
    elif extra_consumer == 'split':
        nodes = [n for n in nodes if n.op_type != 'Slice']
        nodes.insert(0, helper.make_node(
            'Split', ['w'], ['h', 'dropped'], axis=2, split=[3, 1]))
        initializers = []
        outputs.append(_value('dropped', [1, 2, 1, 3]))
    elif extra_consumer == 'gather':
        nodes = [n for n in nodes if n.op_type != 'Slice']
        initializers = [_ints('idx', [1, 2, 3])]
        nodes.insert(0, helper.make_node('Gather', ['w', 'idx'], ['h'],
                                         axis=2))
    graph = helper.make_graph(nodes, 'g', inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    if binding is not None:
        helper.set_model_props(model, {'state_binding': binding})
    return model


def _validate(model, **kwargs):
    return validate_inplace_state_contract(
        model, ['w'], kwargs.pop('outputs', ['w_out']),
        window_inputs=kwargs.pop('window', ['w']), **kwargs)


def test_good_graph_passes_and_counts_its_windows():
    assert _validate(_model()) == 1


def test_newest_first_window_slice_is_accepted():
    # K/V rings keep slots 0..K-2: start 0, end K-1.
    assert _validate(_model(start=0, end=3)) == 1


def test_recurrent_state_that_is_not_a_window_needs_no_slice():
    assert _validate(_model(extra_consumer='tap'), window=[]) == 0


@pytest.mark.parametrize('kwargs,message', [
    (dict(extra_consumer='tap'), 'has 2 consumers'),
    (dict(extra_consumer='split'), 'not Slice'),
    (dict(extra_consumer='gather'), 'not Slice'),
    (dict(axes=(1,)), 'must cut axis 2'),
    (dict(steps=(2,)), 'step 1'),
    (dict(steps=(-1,), start=3, end=0), 'step 1'),
    (dict(end=INT64_MAX), 'outside the extent'),
    (dict(start=1, end=3), 'drops exactly one'),
    (dict(start=0, end=4), 'drops exactly one'),
    (dict(start=-1, end=3), 'outside the extent'),
    (dict(out_producer='slice'), 'produced by a Slice'),
    (dict(out_shape=(1, 2, 5, 3)), 'changes shape'),
    (dict(binding=None), 'state_binding'),
    (dict(binding='copy'), 'state_binding'),
    (dict(dynamic_starts=True), 'non-constant starts'),
    (dict(out_is_input=True), 'is a graph input'),
])
def test_each_violation_is_named(kwargs, message):
    window = kwargs.pop('window', ['w'])
    if kwargs.get('out_producer') == 'slice':
        # the output rule is checked independently of the window rule
        window = []
    outputs = ['w'] if kwargs.get('out_is_input') else ['w_out']
    model = _model(**kwargs)
    with pytest.raises(RuntimeError, match=message):
        _validate(model, window=window, outputs=outputs)


def test_rank_three_state_is_refused_unless_allowed():
    model = _model(window=(4, 2, 3), rank=3, start=1, end=4, axes=(0,))
    with pytest.raises(RuntimeError, match='rank 3, not 4'):
        _validate(model, window=[])
    assert _validate(model, window=[], require_rank4=False) == 0


def test_element_type_change_is_refused():
    model = _model()
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    with pytest.raises(RuntimeError, match='element type'):
        _validate(model)


def test_unknown_names_are_refused():
    model = _model()
    with pytest.raises(RuntimeError, match='not a graph input and output'):
        validate_inplace_state_contract(model, ['nope'], ['w_out'])
    with pytest.raises(RuntimeError, match='not state inputs'):
        validate_inplace_state_contract(model, ['w'], ['w_out'],
                                        window_inputs=['x'])
