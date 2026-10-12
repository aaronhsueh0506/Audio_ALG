"""Validate stateless streaming ONNX graphs.

Two checks live here: no temporal padding inside the graph, and the
in-place state boundary (``validate_inplace_state_graph``).

Learned 2-D feature maps in these graphs use NCTF layout.  The caller either
supplies the complete temporal kernel window or carries older frames as an
explicit state tensor, so an exported graph must never add time-axis padding.
Frequency padding remains valid and is deliberately not restricted here.

This module lives inside ``AINR`` because that directory is a standalone
release unit used on training and quantization machines.
"""

from __future__ import annotations


def _attribute(node, name):
    import onnx

    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return None


def _constant_arrays(graph):
    from onnx import numpy_helper

    values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.graph.initializer
    }
    for node in graph.graph.node:
        if node.op_type != 'Constant' or not node.output:
            continue
        tensor = _attribute(node, 'value')
        if tensor is not None:
            values[node.output[0]] = numpy_helper.to_array(tensor)
    return values


def validate_nctf_no_temporal_padding(graph, require_static=False):
    """Reject temporal padding in a static NCTF streaming graph.

    ONNX represents a rank-4 Pad by eight integers::

        [N_begin, C_begin, T_begin, F_begin,
         N_end,   C_end,   T_end,   F_end]

    Seeing an 8-element ``pads`` tensor therefore does not mean "pad by 8".
    This check requires ``T_begin == T_end == 0`` while allowing frequency
    padding.  It also checks padding fused into Conv/ConvTranspose attributes.
    The exporters call it after their normal constant-folding pass.  With
    ``require_static=True`` a Pad whose values are still dynamic is rejected;
    ordinary non-``--verify`` exports keep working without onnxruntime's
    optional folding pass, while every statically visible pad is still
    checked.
    """
    constants = _constant_arrays(graph)
    checked = 0
    for node in graph.graph.node:
        if node.op_type == 'Pad':
            if len(node.input) < 2 or node.input[1] not in constants:
                if require_static:
                    raise RuntimeError(
                        'streaming ONNX has a Pad with non-static pads: %s' %
                        (node.name or node.output[0])
                    )
                continue
            pads = constants[node.input[1]].reshape(-1).tolist()
            if len(pads) != 8:
                # Only rank-4 NCTF tensors are governed by this contract.
                continue
            checked += 1
            if int(pads[2]) != 0 or int(pads[6]) != 0:
                raise RuntimeError(
                    'streaming ONNX pads the time axis at %s: %s' %
                    (node.name or node.output[0], pads)
                )
        elif node.op_type in ('Conv', 'ConvTranspose'):
            kernel = _attribute(node, 'kernel_shape')
            pads = _attribute(node, 'pads')
            if kernel is None or len(kernel) != 2:
                continue
            checked += 1
            auto_pad = _attribute(node, 'auto_pad')
            if auto_pad not in (None, b'', b'NOTSET', b'VALID'):
                raise RuntimeError(
                    'streaming ONNX %s uses implicit auto_pad at %s: %r' %
                    (node.op_type, node.name or node.output[0], auto_pad)
                )
            if pads is None:
                continue
            if len(pads) != 4 or int(pads[0]) != 0 or int(pads[2]) != 0:
                raise RuntimeError(
                    'streaming ONNX %s pads the time axis at %s: %s' %
                    (node.op_type, node.name or node.output[0], list(pads))
                )
    return checked


STATE_BINDING = 'in_place_same_address'


def _dims(value_info):
    """Static dims of a graph input/output, or None when any is symbolic."""
    shape = value_info.type.tensor_type.shape
    dims = []
    for dim in shape.dim:
        if not dim.HasField('dim_value') or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def _slice_parameters(node, constants):
    """``(starts, ends, axes, steps)`` of a Slice as integer lists.

    Opset >= 10 passes them as inputs, earlier opsets as attributes. A
    parameter that is not a graph constant comes back as None; ``steps``
    defaults to ones when absent, ``axes`` to None.
    """
    def operand(index, attribute):
        if len(node.input) > index and node.input[index]:
            value = constants.get(node.input[index])
            return None if value is None else [
                int(item) for item in value.reshape(-1).tolist()]
        found = _attribute(node, attribute)
        return None if found is None else [int(item) for item in found]

    starts = operand(1, 'starts')
    ends = operand(2, 'ends')
    axes = operand(3, 'axes')
    steps = operand(4, 'steps')
    if steps is None and starts is not None:
        steps = [1] * len(starts)
    return starts, ends, axes, steps


def _single_axis_slice(node, constants, name, axis, size):
    """Check one Slice reading a state input; return ``(start, stop)``."""
    where = 'state input %s consumer %s' % (name, node.name or node.output[0])
    starts, ends, axes, steps = _slice_parameters(node, constants)
    if starts is None or ends is None:
        raise RuntimeError('%s: Slice bounds are not constants' % where)
    if axes is None or len(axes) != 1 or axes[0] != axis:
        raise RuntimeError('%s: Slice must slice axis %d only, got axes=%s'
                           % (where, axis, axes))
    if len(starts) != 1 or len(ends) != 1 or steps != [1]:
        raise RuntimeError(
            '%s: Slice must be one range with step 1, got starts=%s ends=%s '
            'steps=%s' % (where, starts, ends, steps))
    start, stop = starts[0], ends[0]
    if not 0 <= start < stop <= size:
        raise RuntimeError(
            '%s: Slice range [%d:%d) is not an explicit range inside the '
            'axis of %d' % (where, start, stop, size))
    return start, stop


def validate_inplace_state_graph(graph, state_handoff, windows=None,
                                 slices=None):
    """Check that a graph's state boundary can be bound in place.

    ``state_handoff`` maps each state input name to the output that hands it
    back. The runtime reads every input before it writes any output at the
    same address, so the pair must be interchangeable byte for byte.

    * every state input and its output exist, are rank 4, fully static, and
      have the same shape and element type;
    * ``windows`` maps a state input to the axis of its sliding window. The
      input has exactly one consumer, a Slice of that axis with an explicit
      range ``[start:size)`` (start >= 1) and step 1: it drops the oldest
      slot, and the new slot is concatenated after it;
    * ``slices`` maps a state input to the axis along which it is cut into
      per-GRU rows. Every consumer is a Slice of that axis with an explicit
      range and step 1, and the ranges tile the axis exactly once;
    * no graph output is produced by a Slice;
    * the metadata records ``state_binding = in_place_same_address``.

    Returns the number of state pairs checked.
    """
    windows = dict(windows or {})
    slices = dict(slices or {})
    inputs = {item.name: item for item in graph.graph.input}
    outputs = {item.name: item for item in graph.graph.output}
    constants = _constant_arrays(graph)
    producers = {}
    consumers = {}
    for node in graph.graph.node:
        for output in node.output:
            producers[output] = node
        for operand in node.input:
            consumers.setdefault(operand, []).append(node)

    for name, back in state_handoff.items():
        if name not in inputs or back not in outputs:
            raise RuntimeError('state pair %s -> %s is not in the graph'
                               % (name, back))
        shape_in, shape_out = _dims(inputs[name]), _dims(outputs[back])
        if shape_in is None or shape_out is None:
            raise RuntimeError('state pair %s -> %s has a non-static shape'
                               % (name, back))
        if len(shape_in) != 4:
            raise RuntimeError('state %s has rank %d, expected 4'
                               % (name, len(shape_in)))
        if shape_in != shape_out:
            raise RuntimeError('state pair %s -> %s changes shape: %s vs %s'
                               % (name, back, shape_in, shape_out))
        if (inputs[name].type.tensor_type.elem_type !=
                outputs[back].type.tensor_type.elem_type):
            raise RuntimeError('state pair %s -> %s changes element type'
                               % (name, back))

    for name, axis in windows.items():
        size = _dims(inputs[name])[axis]
        users = consumers.get(name, [])
        if len(users) != 1 or users[0].op_type != 'Slice':
            raise RuntimeError(
                'window input %s must have exactly one consumer and it must '
                'be a Slice, got %s' % (name, [u.op_type for u in users]))
        start, stop = _single_axis_slice(users[0], constants, name, axis, size)
        if start < 1 or stop != size:
            raise RuntimeError(
                'window input %s: Slice [%d:%d) of %d does not drop the '
                'oldest slot' % (name, start, stop, size))

    for name, axis in slices.items():
        size = _dims(inputs[name])[axis]
        users = consumers.get(name, [])
        if not users or any(user.op_type != 'Slice' for user in users):
            raise RuntimeError(
                'state input %s must be consumed only by Slice nodes, got %s'
                % (name, [u.op_type for u in users]))
        covered = [0] * size
        for user in users:
            start, stop = _single_axis_slice(user, constants, name, axis, size)
            for index in range(start, stop):
                covered[index] += 1
        if any(count != 1 for count in covered):
            raise RuntimeError(
                'state input %s: Slices do not tile axis %d exactly once '
                '(coverage %s)' % (name, axis, covered))

    for item in graph.graph.output:
        producer = producers.get(item.name)
        if producer is not None and producer.op_type == 'Slice':
            raise RuntimeError('graph output %s is produced by a Slice'
                               % item.name)

    props = {prop.key: prop.value for prop in graph.metadata_props}
    if props.get('state_binding') != STATE_BINDING:
        raise RuntimeError('metadata state_binding must be %r, got %r'
                           % (STATE_BINDING, props.get('state_binding')))
    return len(state_handoff)


def validate_inplace_state_from_metadata(graph):
    """Run ``validate_inplace_state_graph`` with the parameters the exporter
    wrote into the graph's own metadata.

    ``state_handoff`` maps each state input to its output and
    ``state_slicing`` maps it to ``{"kind": "window" | "rows", "axis": n}``.
    A graph without both entries is refused: the check cannot be run from a
    graph that does not describe its state boundary.
    """
    import json

    props = {prop.key: prop.value for prop in graph.metadata_props}
    for key in ('state_handoff', 'state_slicing'):
        if key not in props:
            raise RuntimeError('metadata lacks %s' % key)
    handoff = json.loads(props['state_handoff'])
    slicing = json.loads(props['state_slicing'])
    if set(slicing) != set(handoff):
        raise RuntimeError('state_slicing and state_handoff name different '
                           'tensors')
    windows = {name: int(item['axis']) for name, item in slicing.items()
               if item['kind'] == 'window'}
    rows = {name: int(item['axis']) for name, item in slicing.items()
            if item['kind'] == 'rows'}
    if len(windows) + len(rows) != len(slicing):
        raise RuntimeError('state_slicing kind must be window or rows')
    return validate_inplace_state_graph(graph, handoff, windows=windows,
                                        slices=rows)
