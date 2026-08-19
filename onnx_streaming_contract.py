"""Validation shared by the stateless streaming ONNX exporters.

Learned 2-D feature maps in these graphs use NCTF layout.  The caller either
supplies the complete temporal kernel window or carries older frames as an
explicit state tensor, so an exported graph must never add time-axis padding.
Frequency padding remains valid and is deliberately not restricted here.
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
