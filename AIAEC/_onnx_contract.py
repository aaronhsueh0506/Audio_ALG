"""ONNX contract checks for AIAEC's stateless streaming graphs.

``validate_nctf_no_temporal_padding`` is the shared padding validator. A full
checkout carries ``onnx_streaming_contract`` at the repository root. A
standalone AIAEC copy has ``AINR/`` beside it instead, and ``AINR`` ships its
own self-contained copy of the same module, so fall back to that one.

``validate_inplace_state_contract`` is defined here and has no dependency on
either copy. It checks the boundary that lets an accelerator bind every state
output to the address of the matching state input: each state tensor comes back
in full with its input's shape, history windows are shifted by a Slice on the
graph input, and the new frame is concatenated behind (or in front of) the
kept frames, so no state output is produced by a Slice.
"""

import os
import sys

try:
    from onnx_streaming_contract import validate_nctf_no_temporal_padding
except ModuleNotFoundError as error:
    _AINR_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AINR')
    if error.name != 'onnx_streaming_contract' or not os.path.isdir(_AINR_ROOT):
        raise
    # Appended, not inserted: only this one module is wanted from AINR, and the
    # names AIAEC already resolves must keep resolving where they did.
    sys.path.append(_AINR_ROOT)
    from onnx_streaming_contract import validate_nctf_no_temporal_padding

STATE_BINDING = 'in_place_same_address'
WINDOW_TIME_AXIS = 2


def _constant_arrays(model):
    from onnx import numpy_helper

    values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != 'Constant' or not node.output:
            continue
        for attribute in node.attribute:
            if attribute.name == 'value':
                values[node.output[0]] = numpy_helper.to_array(attribute.t)
    return values


def _static_dims(value_info):
    shape = value_info.type.tensor_type.shape
    dims = []
    for dim in shape.dim:
        if not dim.HasField('dim_value'):
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _slice_parameters(node, constants):
    """(starts, ends, axes, steps) of a Slice; each None when absent, and a
    RuntimeError when present but not a compile-time constant."""
    import onnx

    names = ('starts', 'ends', 'axes', 'steps')
    attributes = {a.name: list(onnx.helper.get_attribute_value(a))
                  for a in node.attribute}
    result = []
    for index, name in enumerate(names):
        if name in attributes:
            result.append(attributes[name])
            continue
        position = index + 1
        if position >= len(node.input) or not node.input[position]:
            result.append(None)
        elif node.input[position] in constants:
            result.append(constants[node.input[position]].reshape(-1).tolist())
        else:
            raise RuntimeError(
                'Slice %s has a non-constant %s' %
                (node.name or node.output[0], name))
    return tuple(result)


def validate_inplace_state_contract(model, state_inputs, state_outputs,
                                    window_inputs=(), require_rank4=True):
    """Check the same-address state boundary of a streaming graph.

    ``state_inputs`` and ``state_outputs`` are the graph's state tensor names,
    pairwise in the same order. ``window_inputs`` is the subset that are
    history windows. For every pair the declared shapes must be identical,
    static, and rank 4 when ``require_rank4``. Every window input must have
    exactly one consumer, a Slice over the time axis with constant
    non-negative start, an explicit integer end inside the extent, step 1, and
    one frame fewer kept than the window holds. No state output may be
    produced by a Slice or be a graph input itself, and the model metadata
    must name ``state_binding = in_place_same_address``.

    Returns the number of window inputs checked. Raises RuntimeError on the
    first violation.
    """
    if len(state_inputs) != len(state_outputs):
        raise RuntimeError('state input/output name lists differ in length')
    unknown = set(window_inputs) - set(state_inputs)
    if unknown:
        raise RuntimeError('window inputs that are not state inputs: %s'
                           % sorted(unknown))
    graph = model.graph
    graph_inputs = {v.name: v for v in graph.input}
    graph_outputs = {v.name: v for v in graph.output}
    constants = _constant_arrays(model)

    for name_in, name_out in zip(state_inputs, state_outputs):
        if name_in not in graph_inputs or name_out not in graph_outputs:
            raise RuntimeError('state pair %s -> %s is not a graph input and '
                               'output' % (name_in, name_out))
        dims_in = _static_dims(graph_inputs[name_in])
        dims_out = _static_dims(graph_outputs[name_out])
        if dims_in is None or dims_out is None:
            raise RuntimeError('state pair %s -> %s has a non-static shape'
                               % (name_in, name_out))
        if dims_in != dims_out:
            raise RuntimeError('state %s changes shape %s -> %s'
                               % (name_in, dims_in, dims_out))
        if require_rank4 and len(dims_in) != 4:
            raise RuntimeError('state %s has rank %d, not 4'
                               % (name_in, len(dims_in)))
        type_in = graph_inputs[name_in].type.tensor_type.elem_type
        type_out = graph_outputs[name_out].type.tensor_type.elem_type
        if type_in != type_out:
            raise RuntimeError('state %s changes element type' % name_in)

    consumers = {}
    producers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
        for name in node.output:
            producers[name] = node

    for name in window_inputs:
        users = consumers.get(name, [])
        if len(users) != 1:
            raise RuntimeError(
                'window input %s has %d consumers, expected exactly one Slice'
                % (name, len(users)))
        node = users[0]
        if node.op_type != 'Slice':
            raise RuntimeError('window input %s is consumed by %s, not Slice'
                               % (name, node.op_type))
        if list(node.input).count(name) != 1 or node.input[0] != name:
            raise RuntimeError('window input %s is not the Slice data input'
                               % name)
        starts, ends, axes, steps = _slice_parameters(node, constants)
        extent = _static_dims(graph_inputs[name])[WINDOW_TIME_AXIS]
        if starts is None or ends is None or axes is None:
            raise RuntimeError('Slice of %s must give starts, ends and axes'
                               % name)
        if axes != [WINDOW_TIME_AXIS] or len(starts) != 1 or len(ends) != 1:
            raise RuntimeError('Slice of %s must cut axis %d only, got %s'
                               % (name, WINDOW_TIME_AXIS, axes))
        if steps is not None and steps != [1]:
            raise RuntimeError('Slice of %s must have step 1, got %s'
                               % (name, steps))
        start, end = int(starts[0]), int(ends[0])
        if not 0 <= start < end <= extent:
            raise RuntimeError(
                'Slice of %s has start %d, end %d outside the extent %d '
                '(an integer end inside the window is required)'
                % (name, start, end, extent))
        if end - start != extent - 1:
            raise RuntimeError(
                'Slice of %s keeps %d of %d frames; a window drops exactly '
                'one' % (name, end - start, extent))

    for name in state_outputs:
        if name in graph_inputs:
            raise RuntimeError('state output %s is a graph input' % name)
        producer = producers.get(name)
        if producer is None:
            raise RuntimeError('state output %s has no producer' % name)
        if producer.op_type == 'Slice':
            raise RuntimeError('state output %s is produced by a Slice' % name)

    props = {p.key: p.value for p in model.metadata_props}
    if props.get('state_binding') != STATE_BINDING:
        raise RuntimeError('metadata state_binding is %r, expected %r'
                           % (props.get('state_binding'), STATE_BINDING))
    return len(tuple(window_inputs))


__all__ = [
    'STATE_BINDING',
    'validate_inplace_state_contract',
    'validate_nctf_no_temporal_padding',
]
