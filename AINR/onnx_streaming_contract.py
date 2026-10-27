"""Validate stateless streaming ONNX graphs.

Three checks live here: no temporal padding inside the graph, the
in-place state boundary (``validate_inplace_state_graph``), and the agreement
of a multi-frame graph's ``frames_per_call`` with its shapes
(``validate_frames_per_call``). The input-window
alignment that a ``mask_lookahead`` implies (``lookahead_window_offsets``) is
defined here once for the families that share the lookahead contract.

Learned 2-D feature maps in these graphs use NCTF layout.  The caller either
supplies the complete temporal kernel window or carries older frames as an
explicit state tensor, so an exported graph must never add time-axis padding.
Frequency padding remains valid and is deliberately not restricted here.

This is the only implementation. It lives inside ``AINR`` because that
directory is a standalone release unit used on training and quantization
machines; the AIAEC exporters import it through ``AIAEC/_onnx_contract.py``.
"""

from __future__ import annotations

import numbers

# Stated once, shared by the exporters and the trainers' checkpoint contract
# check, so every refusal of a lookahead pair gives the same instruction.
LOOKAHEAD_CONTRACT_NOTE = (
    'mask_lookahead and df_lookahead are part of the model contract (they fix '
    'the feature-window alignment and the end-to-end latency): another value '
    'needs a fresh training run and a C host built with the same pair, not an '
    'edit of config.ini.'
)


def lookahead_window_offsets(mask_lookahead, input_frames):
    """Frame offsets, relative to ``t``, of the window behind ``heads[t]``.

    A graph that takes ``input_frames`` consecutive feature frames and emits
    one frame is fed ``[t-(input_frames-1-L), ..., t+L]`` for
    ``L = mask_lookahead``: ``(-2, -1, 0)``, ``(-1, 0, 1)`` and ``(0, 1, 2)``
    for ``L`` = 0, 1, 2 at three frames. Raises ``ValueError`` for an ``L``
    the window cannot express; a lookahead is a model contract, never a
    free choice.
    """
    if (isinstance(mask_lookahead, bool)
            or not isinstance(mask_lookahead, numbers.Integral)
            or not 0 <= mask_lookahead < input_frames):
        raise ValueError(
            'mask_lookahead=%r is not an integer in 0..%d, the range a '
            '%d-frame input window covers. %s'
            % (mask_lookahead, input_frames - 1, input_frames,
               LOOKAHEAD_CONTRACT_NOTE)
        )
    last = int(mask_lookahead)
    return tuple(range(last + 1 - input_frames, last + 1))


def lookahead_window_alignment(mask_lookahead, input_frames):
    """``'[t-1,t,t+1] -> heads[t]'``-style string of the offsets above."""
    names = ['t' if offset == 0 else 't%+d' % offset
             for offset in lookahead_window_offsets(mask_lookahead,
                                                    input_frames)]
    return '[%s] -> heads[t]' % ','.join(names)


def validate_frames_per_call(graph, window_inputs, head_outputs,
                             input_frames):
    """Check a graph's ``frames_per_call`` metadata against its shapes.

    A graph that emits N heads per invocation records ``frames_per_call = N``
    (N >= 2); a one-frame graph records nothing, so an explicit 1 is refused.
    ``window_inputs`` maps each feature-window input to its time axis, which
    must hold ``input_frames - 1 + N`` frames; ``head_outputs`` maps each
    head output to its time axis, which must hold N. When present,
    ``input_feature_frames`` and ``output_frames_per_invocation`` must state
    the same two numbers. Returns N; raises RuntimeError on the first
    mismatch.
    """
    props = {prop.key: prop.value for prop in graph.metadata_props}
    frames = 1
    if 'frames_per_call' in props:
        try:
            frames = int(props['frames_per_call'])
        except ValueError:
            frames = 0
        if str(frames) != props['frames_per_call'] or frames < 2:
            raise RuntimeError(
                'metadata frames_per_call must be an integer >= 2 (a '
                'one-frame graph records none), got %r'
                % props['frames_per_call'])
    span = input_frames - 1 + frames
    for key, expected in (('input_feature_frames', span),
                          ('output_frames_per_invocation', frames)):
        if key in props and props[key] != str(expected):
            raise RuntimeError('metadata %s is %s, not %d for frames_per_call '
                               '%d' % (key, props[key], expected, frames))
    inputs = {item.name: item for item in graph.graph.input}
    outputs = {item.name: item for item in graph.graph.output}
    for table, values, expected, kind in (
            (window_inputs, inputs, span, 'input'),
            (head_outputs, outputs, frames, 'output')):
        for name, axis in table.items():
            if name not in values:
                raise RuntimeError('graph has no %s %s' % (kind, name))
            dims = _dims(values[name])
            if dims is None or not 0 <= axis < len(dims):
                raise RuntimeError('%s %s has no static axis %d'
                                   % (kind, name, axis))
            if dims[axis] != expected:
                raise RuntimeError(
                    '%s %s has %d frames on axis %d, not %d for '
                    'frames_per_call %d'
                    % (kind, name, dims[axis], axis, expected, frames))
    return frames


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
    padding. It also checks padding fused into Conv/ConvTranspose attributes.
    With ``require_static=True`` a Pad whose values are still dynamic is
    rejected; ordinary non-``--verify`` exports keep working without
    onnxruntime's optional folding pass, while every statically visible pad is
    still checked.
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

# Which end of a window the graph drops before it appends the new frame.
# A time-ordered window (oldest first, ``cat(W[1:K], new)``) drops the first
# slot; a newest-first ring (``cat(new, W[0:K-1])``) drops the last one.
WINDOW_DROPS = ('first', 'last')


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
                                 slices=None, require_rank4=True):
    """Check that a graph's state boundary can be bound in place.

    ``state_handoff`` maps each state input name to the output that hands it
    back. The runtime reads every input before it writes any output at the
    same address, so the pair must be interchangeable byte for byte.

    * every state input and its output exist, have the same static shape and
      element type, and are rank 4 when ``require_rank4``;
    * ``windows`` maps a state input to ``(axis, drop)``: the axis of its
      sliding window and the end the graph drops, one of ``WINDOW_DROPS``.
      The input has exactly one consumer, a Slice of that axis that reads it
      as the data input, with constant bounds, an explicit range inside the
      axis and step 1. The range must be exactly ``[1:size)`` for ``'first'``
      (oldest-first window, the new frame is concatenated after) or
      ``[0:size-1)`` for ``'last'`` (newest-first ring, the new frame is
      concatenated before); the other end is refused;
    * ``slices`` maps a state input to the axis along which it is cut into
      per-GRU rows. Every consumer is a Slice of that axis with an explicit
      range and step 1, and the ranges tile the axis exactly once;
    * no state output is produced by a Slice or is a graph input, and every
      state output has a producer (other outputs are unrestricted);
    * the metadata records ``state_binding = in_place_same_address``.

    Returns the number of state pairs checked. Raises RuntimeError on the
    first violation.
    """
    windows = dict(windows or {})
    for name, (axis, drop) in windows.items():
        if drop not in WINDOW_DROPS:
            raise ValueError('window %s: drop must be one of %s, got %r'
                             % (name, WINDOW_DROPS, drop))
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
            raise RuntimeError('state pair %s -> %s is not a graph input and '
                               'output' % (name, back))
        shape_in, shape_out = _dims(inputs[name]), _dims(outputs[back])
        if shape_in is None or shape_out is None:
            raise RuntimeError('state pair %s -> %s has a non-static shape'
                               % (name, back))
        if require_rank4 and len(shape_in) != 4:
            raise RuntimeError('state %s has rank %d, not 4'
                               % (name, len(shape_in)))
        if shape_in != shape_out:
            raise RuntimeError('state pair %s -> %s changes shape: %s vs %s'
                               % (name, back, shape_in, shape_out))
        if (inputs[name].type.tensor_type.elem_type !=
                outputs[back].type.tensor_type.elem_type):
            raise RuntimeError('state pair %s -> %s changes element type'
                               % (name, back))

    unknown = (set(windows) | set(slices)) - set(state_handoff)
    if unknown:
        raise RuntimeError(
            'windows or slices name inputs that are not state inputs: %s'
            % sorted(unknown))

    for name, (axis, drop) in windows.items():
        size = _dims(inputs[name])[axis]
        users = consumers.get(name, [])
        if len(users) != 1 or users[0].op_type != 'Slice':
            raise RuntimeError(
                'window input %s has %d consumers; it must have exactly one '
                'consumer and it must be a Slice, got %s'
                % (name, len(users), [u.op_type for u in users]))
        if list(users[0].input).count(name) != 1 or users[0].input[0] != name:
            raise RuntimeError('window input %s is not the Slice data input'
                               % name)
        start, stop = _single_axis_slice(users[0], constants, name, axis, size)
        expected = (1, size) if drop == 'first' else (0, size - 1)
        if (start, stop) != expected:
            raise RuntimeError(
                'window input %s: Slice [%d:%d) of %d does not drop the %s '
                'slot (expected [%d:%d))'
                % (name, start, stop, size, drop, expected[0], expected[1]))

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

    for back in state_handoff.values():
        if back in inputs:
            raise RuntimeError('state output %s is a graph input' % back)
        if back not in producers:
            raise RuntimeError('state output %s has no producer' % back)
        if producers[back].op_type == 'Slice':
            raise RuntimeError('state output %s is produced by a Slice'
                               % back)

    props = {prop.key: prop.value for prop in graph.metadata_props}
    if props.get('state_binding') != STATE_BINDING:
        raise RuntimeError('metadata state_binding must be %r, got %r'
                           % (STATE_BINDING, props.get('state_binding')))
    return len(state_handoff)


def validate_inplace_state_from_metadata(graph):
    """Run ``validate_inplace_state_graph`` with the parameters the exporter
    wrote into the graph's own metadata.

    ``state_handoff`` maps each state input to its output and
    ``state_slicing`` maps it to ``{"kind": "window", "axis": n, "drop":
    "first" | "last"}`` or ``{"kind": "rows", "axis": n}``. A graph without
    both entries, or a window without its ``drop``, is refused: the check
    cannot be run from a graph that does not describe its state boundary.
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
    windows = {}
    for name, item in slicing.items():
        if item['kind'] == 'window':
            if item.get('drop') not in WINDOW_DROPS:
                raise RuntimeError(
                    'state_slicing of %s: a window declares drop as one of '
                    '%s, got %r' % (name, WINDOW_DROPS, item.get('drop')))
            windows[name] = (int(item['axis']), item['drop'])
    rows = {name: int(item['axis']) for name, item in slicing.items()
            if item['kind'] == 'rows'}
    if len(windows) + len(rows) != len(slicing):
        raise RuntimeError('state_slicing kind must be window or rows')
    return validate_inplace_state_graph(graph, handoff, windows=windows,
                                        slices=rows)
