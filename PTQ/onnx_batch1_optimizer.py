#!/usr/bin/env python3
"""Batch-1 graph cleanup for stateless streaming ONNX graphs.

The streaming exporters trace at batch 1 with T frames per call (T = 1 for the
one-frame graphs, T = N for ``--frames-per-call N``), so most tensors carry
size-1 axes (batch, time when T = 1, a group of one) that the tracer turns into
chains of Squeeze/Unsqueeze/Reshape/Transpose and into 5-D and 6-D
intermediates. With every shape static those axes can be removed, and nothing
here changes a number: every pass is a re-labelling of axes, never a different
arithmetic. The layout is ``[1, C, T, F]``: batch stays 1 and the frames sit on
axis 2, as in the Conv paths; the passes must not move T onto axis 0.

Passes, in the order ``optimize_batch1`` runs them:

``fold_frequency_pad_into_conv``
    A zero Pad on the frequency axis that feeds one Conv becomes that Conv's
    ``pads`` attribute (the temporal axis is never padded).

``merge_views``
    Squeeze, Unsqueeze, Flatten, Reshape, a Transpose that only moves size-1
    axes and a Gather on a size-1 axis are all the same data in the same
    order. Each becomes one Reshape, a chain of them collapses to a single
    Reshape, and a chain whose entry and exit shapes are equal disappears.

``lower_rank``
    Tensors above ``target_rank`` (default 4) are brought down by dropping
    size-1 axes (the leading ones, unless that would put a longer axis in the
    batch position and another unit axis can go instead): the op is run on the reduced operands and
    a Reshape restores the original shape for the consumers, so ``merge_views``
    can then cancel the Reshape pairs. Supported: elementwise ops,
    Transpose, MatMul, ReduceSum/Mean/Max/Min, Softmax. A node it cannot
    lower is left as it is and reported.

    A grouped linear over T frames (``[1,T,G,1,K] @ [G,K,O]``, graphs that
    emit several frames per call) has no spare unit axis, so dropping the
    leading one would leave T on axis 0, which an accelerator reads as the
    batch. There T is moved into the MatMul's row axis instead
    (``[1,G,T,K] @ [G,K,O]``, a Transpose on the rank-4 tensor before and
    after, nothing leaves rank 4 or batch 1); the numbers are the same.
    Tensors that still have an axis 0 other than 1 are listed in the report
    (``batch_axis_left``).

Graph inputs, outputs, their names and shapes, the opset and the metadata are
untouched; the result is checked for that before it is returned. Semantic
rewrites (a different formulation of an op) belong to the model's exporter,
not here.

Usage::

    python3 onnx_batch1_optimizer.py model.onnx -o model_b1.onnx --verify

``--verify`` runs the original and the optimised graph in onnxruntime (graph
optimisation off) on random inputs and requires the outputs to be identical.
The one exception is a grouped linear regrouped over T frames: it is the same
dot products, but onnxruntime may pick another matrix-multiply kernel for T
rows than for one row, which changes the float32 summation order. A graph with
such a MatMul is compared with a tolerance of ``REGROUP_RTOL`` times its output
scale (about a thousand times above that rounding, far below any wrong
rewrite); every other graph still has to match exactly.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys

import numpy as np

_ELEMENTWISE = frozenset({
    'Elu', 'Relu', 'PRelu', 'Sigmoid', 'Tanh', 'Neg', 'Abs', 'Sqrt', 'Exp',
    'Log', 'Clip', 'LeakyRelu', 'Identity', 'Cast', 'Floor', 'Ceil', 'Sign',
    'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Max', 'Min',
})
_STANDARD_DOMAINS = ('', 'ai.onnx')
_REDUCTIONS = frozenset({'ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin'})


def _inferred(model):
    """A copy of ``model`` with fresh shape inference. Stale value_info (the
    passes rename and drop tensors) is discarded first."""
    import onnx
    clean = onnx.ModelProto()
    clean.CopyFrom(model)
    del clean.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(clean)


class _Graph(object):
    """Shapes, producers and consumers of one ModelProto, rebuilt on demand."""

    def __init__(self, model):
        self.model = model
        inferred = _inferred(model)
        self.shapes = {}
        for value in (list(inferred.graph.value_info)
                      + list(inferred.graph.input)
                      + list(inferred.graph.output)):
            dims = value.type.tensor_type.shape.dim
            shape = tuple(d.dim_value if d.HasField('dim_value') else None
                          for d in dims)
            self.shapes[value.name] = shape
        for init in model.graph.initializer:
            self.shapes[init.name] = tuple(init.dims)
        self.refresh()

    def refresh(self):
        graph = self.model.graph
        self.initializers = {i.name: i for i in graph.initializer}
        self.producer = {}
        self.consumers = {}
        for node in graph.node:
            for name in node.output:
                self.producer[name] = node
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)
        self.graph_outputs = {v.name for v in graph.output}

    def shape(self, name):
        shape = self.shapes.get(name)
        if shape is None or any(d is None for d in shape):
            return None
        return shape


def _attr(node, name, default=None):
    from onnx import helper
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _perm(node, rank):
    """A Transpose's permutation (reversal when the attribute is absent)."""
    return list(_attr(node, 'perm', list(range(rank))[::-1]))


def _set_attr(node, name, value):
    from onnx import helper
    kept = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def _replace(repeated, items):
    """Overwrite a protobuf repeated field with ``items``."""
    items = list(items)
    del repeated[:]
    repeated.extend(items)


def _const_ints(info, name):
    from onnx import numpy_helper
    init = info.initializers.get(name)
    if init is None:
        return None
    return [int(v) for v in numpy_helper.to_array(init).reshape(-1)]


class _Namer(object):
    def __init__(self, model):
        self.used = {n for node in model.graph.node
                     for n in list(node.input) + list(node.output)}
        self.used |= {i.name for i in model.graph.initializer}
        self.count = 0

    def new(self, hint):
        while True:
            self.count += 1
            name = '%s_b1_%d' % (hint, self.count)
            if name not in self.used:
                self.used.add(name)
                return name


def _reshape(namer, model, source, target, shape):
    """A Reshape node (and its int64 shape initializer) source -> target."""
    from onnx import helper, numpy_helper
    shape_name = namer.new('shape')
    model.graph.initializer.append(numpy_helper.from_array(
        np.asarray(shape, dtype=np.int64), name=shape_name))
    return helper.make_node('Reshape', [source, shape_name], [target],
                            name=namer.new('Reshape'))


# --------------------------------------------------------------- frequency pad


def fold_frequency_pad_into_conv(model):
    """Fold an explicit zero Pad that feeds exactly one Conv into the Conv's
    ``pads`` attribute, so the padding costs no accelerator op.

    A zero Pad followed by an unpadded Conv and the same Conv with ``pads`` are
    the same arithmetic. Only constant-mode, zero-valued, non-negative
    frequency pads with static amounts qualify; anything else is left as it
    is. Returns the number of Pad nodes removed.
    """
    from onnx import numpy_helper

    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    graph_outputs = {value.name for value in graph.output}
    removed = 0
    for pad in list(graph.node):
        if (pad.op_type != 'Pad' or pad.output[0] in graph_outputs
                or len(pad.input) < 2 or pad.input[1] not in initializers
                or any(pad.input[2:])):
            continue
        mode = _attr(pad, 'mode', b'constant')
        amounts = numpy_helper.to_array(initializers[pad.input[1]]).tolist()
        # [N, C, T, F] begins then ends: only frequency may be padded.
        if (mode != b'constant' or len(amounts) != 8 or min(amounts) < 0
                or any(amounts[i] for i in (0, 1, 2, 4, 5, 6))):
            continue
        users = consumers.get(pad.output[0], [])
        if (len(users) != 1 or users[0].op_type != 'Conv'
                or users[0].input[0] != pad.output[0]):
            continue
        conv = users[0]
        if (_attr(conv, 'auto_pad', b'NOTSET') not in (b'NOTSET', b'')
                or any(_attr(conv, 'pads', (0, 0, 0, 0)))):
            continue
        conv.input[0] = pad.input[0]
        _set_attr(conv, 'pads', [0, amounts[3], 0, amounts[7]])
        graph.node.remove(pad)
        removed += 1
    _drop_unused_initializers(model)
    return removed


# ----------------------------------------------------------------- view merge


def _is_view(info, node):
    """True when ``node`` only re-labels axes: same elements, same order."""
    in_shape = info.shape(node.input[0]) if node.input else None
    out_shape = info.shape(node.output[0])
    if (node.domain not in _STANDARD_DOMAINS or in_shape is None
            or out_shape is None):
        return False
    if node.op_type in ('Squeeze', 'Unsqueeze', 'Flatten'):
        return True
    if node.op_type == 'Reshape':
        return (len(node.input) == 2 and node.input[1] in info.initializers
                and math.prod(in_shape) == math.prod(out_shape))
    if node.op_type == 'Transpose':
        perm = _perm(node, len(in_shape))
        moved = [axis for axis in perm if in_shape[axis] != 1]
        return moved == sorted(moved)
    if node.op_type == 'Gather':
        axis = _attr(node, 'axis', 0)
        return (len(node.input) == 2 and in_shape[axis] == 1
                and math.prod(in_shape) == math.prod(out_shape))
    return False


def merge_views(model):
    """Turn every view into a Reshape, collapse chains, drop identities.

    Returns the number of nodes removed.
    """
    before = len(model.graph.node)
    info = _Graph(model)
    namer = _Namer(model)
    graph = model.graph

    # 1. Every view becomes "Reshape(input, explicit static output shape)"
    #    (a Reshape that already is one is kept; 0 and -1 entries would
    #    depend on the input shape that merging changes).
    rewritten = []
    for node in graph.node:
        if not _is_view(info, node):
            rewritten.append(node)
            continue
        out_shape = info.shape(node.output[0])
        if (node.op_type == 'Reshape'
                and _const_ints(info, node.input[1]) == list(out_shape)):
            rewritten.append(node)
        else:
            rewritten.append(_reshape(namer, model, node.input[0],
                                      node.output[0], out_shape))
    _replace(graph.node, rewritten)
    info.refresh()

    # 2. A Reshape reads straight through a Reshape producer (whatever else
    #    uses that producer), then Reshapes nobody uses any more go.
    for node in graph.node:
        # A 0 in the shape means "copy that dimension of my input", and the
        # input is about to change.
        if (node.op_type != 'Reshape' or node.domain not in _STANDARD_DOMAINS
                or 0 in (_const_ints(info, node.input[1]) or [0])):
            continue
        producer = info.producer.get(node.input[0])
        while producer is not None and producer.op_type == 'Reshape':
            node.input[0] = producer.input[0]
            producer = info.producer.get(node.input[0])
    live = {v.name for v in graph.output}
    kept = []
    for node in reversed(graph.node):
        if node.op_type == 'Reshape' and node.output[0] not in live:
            continue
        live.update(node.input)
        kept.append(node)
    _replace(graph.node, reversed(kept))
    info.refresh()

    # 3. A Reshape that changes nothing is bypassed.
    identity = {}
    for node in graph.node:
        if node.op_type != 'Reshape' or node.output[0] in info.graph_outputs:
            continue
        shape = info.shape(node.input[0])
        if shape is not None and shape == info.shape(node.output[0]):
            identity[node.output[0]] = node.input[0]
    if identity:
        for node in graph.node:
            for index, name in enumerate(node.input):
                while name in identity:
                    name = identity[name]
                node.input[index] = name
        _replace(graph.node, (n for n in graph.node
                              if not (n.op_type == 'Reshape'
                                      and n.output[0] in identity)))

    _drop_unused_initializers(model)
    return before - len(model.graph.node)


def _drop_unused_initializers(model):
    used = {name for node in model.graph.node for name in node.input}
    used |= {v.name for v in model.graph.output}
    _replace(model.graph.initializer,
             (i for i in model.graph.initializer if i.name in used))


# ------------------------------------------------------------------ lower rank


def _drop_axes(shape, target_rank, keep=()):
    """Size-1 axes (ascending) to remove so ``shape`` reaches ``target_rank``,
    or None when there are too few. Axes in ``keep`` stay.

    The leading axes go first. But when that would leave a longer axis in the
    batch position (``[1,32,1,8,26]`` -> ``[32,1,8,26]``) and a later unit
    axis can go instead, that one goes and axis 0 stays 1 (``[1,32,8,26]``):
    an accelerator reads axis 0 as the batch, and a broadcast along it is not
    supported. With no spare unit axis the leading ones go regardless.
    """
    need = len(shape) - target_rank
    if need <= 0:
        return []
    ones = [i for i, d in enumerate(shape) if d == 1 and i not in keep]
    if len(ones) < need:
        return None
    drop = ones[:need]
    first = next(i for i in range(len(shape)) if i not in drop)
    if shape[first] != 1 and 0 in ones and len(ones) > need:
        drop = [i for i in ones if i != 0][:need]
    return drop


def _without(shape, axes):
    """``shape`` with the dimensions at ``axes`` removed."""
    return tuple(d for i, d in enumerate(shape) if i not in axes)


def _renumber(axis, drop):
    """``axis`` after the axes in ``drop`` are removed."""
    return axis - sum(1 for d in drop if d < axis)


def _reduction_axes(node, info):
    """Reduction axes as ints (constant tensor input from opset 13, else the
    attribute), or None when they are not static."""
    if len(node.input) > 1 and node.input[1]:
        return _const_ints(info, node.input[1])
    axes = _attr(node, 'axes')
    return None if axes is None else [int(a) for a in axes]


class _Lowering(object):
    """Rewrite one high-rank node onto reduced operands.

    ``operand`` reshapes an input down (initializers are reshaped in place of
    a node); ``build`` returns the replacement nodes, or None when the node
    cannot be lowered.
    """

    def __init__(self, model, info, namer, target_rank):
        self.opset = next((o.version for o in model.opset_import
                           if o.domain in _STANDARD_DOMAINS), 0)
        self.model = model
        self.info = info
        self.namer = namer
        self.target = target_rank
        self.regrouped = 0

    def operand(self, name, nodes, drop, rank):
        """(name, shape) of an input with the node's axes ``drop`` removed,
        or None. ``rank`` is the rank those axes are numbered in; a shorter
        operand covers the trailing axes (broadcasting), so only its share
        of ``drop`` applies."""
        from onnx import numpy_helper
        shape = self.info.shape(name)
        if shape is None:
            return None
        offset = rank - len(shape)
        local = {a - offset for a in drop if a >= offset}
        if any(shape[i] != 1 for i in local):
            return None
        reduced = _without(shape, local)
        if len(reduced) > self.target:
            return None
        if not local:
            return name, shape
        init = self.info.initializers.get(name)
        out = self.namer.new('lo')
        if init is not None:
            self.model.graph.initializer.append(numpy_helper.from_array(
                numpy_helper.to_array(init).reshape(reduced), name=out))
        else:
            nodes.append(_reshape(self.namer, self.model, name, out, reduced))
        return out, reduced

    def frames_into_rows(self, node, out_shape, drop):
        """A grouped linear run over T frames: ``[1,T,G,1,K] @ [G,K,O]``.

        Dropping the leading unit axis would put T on axis 0, which an
        accelerator reads as the batch. The unit axis in front of K is the
        MatMul's row axis, so T goes there instead, and every tensor stays
        rank 4 with axis 0 equal to 1::

            [1,T,G,1,K] -> [1,T,G,K] -> Transpose(0,2,1,3) -> [1,G,T,K]
            [1,G,T,K] @ [G,K,O] -> [1,G,T,O] -> Transpose(0,2,1,3)
            -> [1,T,G,O] -> [1,T,G,1,O]

        Returns the nodes, or None when the node is not of this form (the
        caller then takes the generic path). Each output element is the same
        dot product; onnxruntime may sum it in another order for T rows, so
        ``--verify`` compares such a graph with a tolerance.
        """
        from onnx import helper
        rank = len(out_shape)
        a = self.info.shape(node.input[0])
        b = self.info.shape(node.input[1])
        if (rank != 5 or a is None or b is None or len(a) != 5
                or len(b) != 3 or list(drop) != [0] or a[0] != 1
                or a[1] <= 1 or a[3] != 1 or out_shape[3] != 1
                or tuple(a[2:3]) != tuple(b[:1])
                or tuple(out_shape[:3]) != tuple(a[:3])
                or self.info.initializers.get(node.input[1]) is None):
            return None
        t, g, k = a[1], a[2], a[4]
        o = out_shape[4]
        self.regrouped += 1
        names = [self.namer.new('lo') for _ in range(4)]
        mat = copy.deepcopy(node)
        del mat.input[:]
        mat.input.extend([names[1], node.input[1]])
        mat.output[0] = names[2]
        return [
            _reshape(self.namer, self.model, node.input[0], names[0],
                     (1, t, g, k)),
            helper.make_node('Transpose', [names[0]], [names[1]],
                             perm=[0, 2, 1, 3],
                             name=self.namer.new('Transpose')),
            mat,
            helper.make_node('Transpose', [names[2]], [names[3]],
                             perm=[0, 2, 1, 3],
                             name=self.namer.new('Transpose')),
            _reshape(self.namer, self.model, names[3], node.output[0],
                     out_shape),
        ]

    def build(self, node):
        op = node.op_type
        out_shape = self.info.shape(node.output[0])
        rank = len(out_shape)
        pre = []
        new = copy.deepcopy(node)
        del new.input[:]
        reduced_out = self.namer.new('lo')
        new.output[0] = reduced_out

        if op in _ELEMENTWISE:
            drop = _drop_axes(out_shape, self.target)
            if drop is None:
                return None
            for name in node.input:
                if not name:
                    new.input.append(name)
                    continue
                reduced = self.operand(name, pre, drop, rank)
                if reduced is None:
                    return None
                new.input.append(reduced[0])
        elif op == 'Transpose':
            drop = _drop_axes(out_shape, self.target)
            in_rank = len(self.info.shape(node.input[0]))
            perm = _perm(node, in_rank)
            if drop is None:
                return None
            gone = sorted(perm[j] for j in drop)
            reduced = self.operand(node.input[0], pre, gone, in_rank)
            if reduced is None:
                return None
            new.input.append(reduced[0])
            _set_attr(new, 'perm', [_renumber(p, gone) for j, p in
                                    enumerate(perm) if j not in drop])
        elif op == 'MatMul':
            drop = _drop_axes(out_shape, self.target,
                              keep=(rank - 2, rank - 1))
            if drop is None:
                return None
            rows = self.frames_into_rows(node, out_shape, drop)
            if rows is not None:
                return rows
            a = self.operand(node.input[0], pre, drop, rank)
            b = self.operand(node.input[1], pre, drop, rank)
            if a is None or b is None or min(len(a[1]), len(b[1])) < 2:
                return None
            batch = np.broadcast_shapes(a[1][:-2], b[1][:-2])
            if (tuple(batch) + (a[1][-2], b[1][-1])
                    != _without(out_shape, drop)):
                return None
            new.input.extend([a[0], b[0]])
        elif op in _REDUCTIONS:
            shape = self.info.shape(node.input[0])
            axes = _reduction_axes(node, self.info)
            if axes is None:
                return None
            axes = [a % len(shape) for a in axes]
            drop = _drop_axes(shape, self.target, keep=axes)
            if drop is None:
                return None
            reduced = self.operand(node.input[0], pre, drop, len(shape))
            if reduced is None:
                return None
            new.input.append(reduced[0])
            axes = [_renumber(a, drop) for a in axes]
            if len(node.input) > 1 and node.input[1]:
                from onnx import numpy_helper
                axes_name = self.namer.new('axes')
                self.model.graph.initializer.append(numpy_helper.from_array(
                    np.asarray(axes, dtype=np.int64), name=axes_name))
                new.input.append(axes_name)
            else:
                _set_attr(new, 'axes', axes)
            # The reduced run keeps the reduced axes (rank stays at target);
            # the Reshape below gives the original shape back.
            _set_attr(new, 'keepdims', 1)
        elif op == 'Softmax' and self.opset >= 13:
            axis = _attr(node, 'axis', -1) % rank
            drop = _drop_axes(out_shape, self.target, keep=(axis,))
            if drop is None:
                return None
            reduced = self.operand(node.input[0], pre, drop, rank)
            if reduced is None:
                return None
            new.input.append(reduced[0])
            _set_attr(new, 'axis', _renumber(axis, drop))
        else:
            return None

        restore = _reshape(self.namer, self.model, reduced_out,
                           node.output[0], out_shape)
        return pre + [new, restore]


def lower_rank(model, target_rank=4):
    """Bring every tensor above ``target_rank`` down by dropping size-1
    axes. Returns ``(lowered, blocked, regrouped)``: the number of nodes
    rewritten, the ``(op_type, name)`` of those that could not be, and how
    many grouped linears were regrouped over T frames."""
    info = _Graph(model)
    lowering = _Lowering(model, info, _Namer(model), target_rank)
    result, blocked, lowered = [], [], 0
    for node in model.graph.node:
        names = [n for n in list(node.input) + list(node.output) if n]
        shapes = [info.shape(n) for n in names]
        high = any(s is not None and len(s) > target_rank for s in shapes)
        if not high or node.op_type == 'Reshape':
            result.append(node)
            continue
        replacement = (None if node.domain not in _STANDARD_DOMAINS
                       or any(s is None for s in shapes)
                       else lowering.build(node))
        if replacement is None:
            blocked.append((node.op_type, node.name))
            result.append(node)
        else:
            result.extend(replacement)
            lowered += 1
    _replace(model.graph.node, result)
    _drop_unused_initializers(model)
    return lowered, blocked, lowering.regrouped


# ------------------------------------------------------------------ the driver


def _boundary(model):
    """Names and shapes of the graph inputs and outputs."""
    def dims(value):
        return tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
    return ([(v.name, dims(v)) for v in model.graph.input],
            [(v.name, dims(v)) for v in model.graph.output])


def _high_rank(model, target_rank):
    inferred = _inferred(model)
    return [(v.name, len(v.type.tensor_type.shape.dim))
            for v in inferred.graph.value_info
            if len(v.type.tensor_type.shape.dim) > target_rank]


def _batch_axis_left(model, target_rank):
    """Names of tensors whose axis 0 is not 1: an accelerator reads that axis
    as the batch. Checked: every rank-``target_rank`` tensor and every tensor
    of rank 3 or more that this optimizer created (``lo_b1_*``). A GRU's Y
    output is ``[seq, directions, batch, hidden]``, not a batch axis, and is
    skipped; the GRU state tensors keep the exporter's own names."""
    inferred = _inferred(model)
    skip = {o for node in model.graph.node if node.op_type == 'GRU'
            for o in node.output}
    left = []
    for v in inferred.graph.value_info:
        dims = v.type.tensor_type.shape.dim
        if v.name in skip or not dims or dims[0].dim_value == 1:
            continue
        if len(dims) == target_rank or (len(dims) >= 3
                                        and v.name.startswith('lo_b1_')):
            left.append(v.name)
    return left


def optimize_batch1(model, target_rank=4, verify=False):
    """Run the passes on ``model`` in place; return a report dict.

    Raises if the graph has subgraphs (not handled), if its inputs, outputs
    or opset would change, if the result is not a valid, shape-consistent
    model, or -- with ``verify`` -- if onnxruntime does not give identical
    outputs for the original and the result.
    """
    import onnx
    from onnx import AttributeProto
    if any(a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
           for node in model.graph.node for a in node.attribute):
        raise ValueError('graphs with subgraphs (If/Loop/Scan) are not '
                         'supported')
    original = copy.deepcopy(model) if verify else None
    boundary = _boundary(model)
    opsets = [(o.domain, o.version) for o in model.opset_import]
    report = {'nodes_before': len(model.graph.node)}
    report['pads_folded'] = fold_frequency_pad_into_conv(model)
    report['views_removed'] = merge_views(model)
    (report['lowered'], report['blocked'],
     report['frames_regrouped']) = lower_rank(model, target_rank)
    report['views_removed'] += merge_views(model)
    del model.graph.value_info[:]
    report['nodes_after'] = len(model.graph.node)
    report['high_rank_left'] = _high_rank(model, target_rank)
    report['batch_axis_left'] = _batch_axis_left(model, target_rank)
    if (_boundary(model) != boundary
            or [(o.domain, o.version) for o in model.opset_import] != opsets):
        raise RuntimeError('batch-1 optimizer changed the graph boundary')
    onnx.checker.check_model(model, full_check=True)
    if verify:
        report['max_abs_diff'] = check_outputs(
            original, model, tolerant=report['frames_regrouped'] > 0)
    return report


def _random_inputs(model, rng):
    from onnx import helper
    feeds = {}
    initializers = {i.name for i in model.graph.initializer}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        dtype = helper.tensor_dtype_to_np_dtype(
            value.type.tensor_type.elem_type)
        shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if np.issubdtype(dtype, np.floating):
            feeds[value.name] = rng.standard_normal(shape).astype(dtype)
        else:
            feeds[value.name] = np.zeros(shape, dtype=dtype)
    return feeds


REGROUP_RTOL = 1e-4


def output_differences(original, optimized, runs=3, seed=0):
    """Run both graphs in onnxruntime (graph optimisation off) on random
    inputs; return ``(output name, largest absolute difference, largest
    absolute value of the original)`` for each output."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    sessions = [
        ort.InferenceSession(m.SerializeToString(), options,
                             providers=['CPUExecutionProvider'])
        for m in (original, optimized)]
    rng = np.random.default_rng(seed)
    names = [o.name for o in original.graph.output]
    worst = [0.0] * len(names)
    scale = [0.0] * len(names)
    for _ in range(runs):
        feeds = _random_inputs(original, rng)
        want = sessions[0].run(None, feeds)
        got = sessions[1].run(None, feeds)
        for i, (a, b) in enumerate(zip(want, got)):
            if a.size:
                with np.errstate(invalid='ignore'):
                    scale[i] = max(scale[i], float(np.nanmax(np.abs(a)))
                                   if np.isfinite(a).any() else 0.0)
            if np.array_equal(a, b, equal_nan=True):
                continue
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            worst[i] = max(worst[i], float('inf') if np.isnan(diff).any()
                           else float(diff.max()))
    return list(zip(names, worst, scale))


def verify_identical(original, optimized, runs=3, seed=0):
    """The largest absolute output difference between the two graphs."""
    return max((d for _, d, _ in output_differences(
        original, optimized, runs, seed)), default=0.0)


def check_outputs(original, optimized, tolerant=False, runs=3, seed=0):
    """Return the largest output difference; raise when it is not allowed:
    any difference at all, or with ``tolerant`` more than ``REGROUP_RTOL``
    times the output scale (at least 1)."""
    diffs = output_differences(original, optimized, runs, seed)
    worst = max((d for _, d, _ in diffs), default=0.0)
    for name, diff, scale in diffs:
        allowed = REGROUP_RTOL * max(1.0, scale) if tolerant else 0.0
        if diff > allowed or math.isnan(diff):
            raise RuntimeError(
                'batch-1 optimizer changed the outputs (max abs diff %g; '
                'output %s differs by %g, allowed %g%s)'
                % (worst, name, diff, allowed,
                   ' -- tolerance applies because a grouped linear was '
                   'regrouped over frames' if tolerant else ''))
    return worst


def format_report(report):
    """One summary line (plus any blocked ops / leftover high-rank tensors)."""
    lines = ['%d -> %d nodes (pads folded %d, views removed %d, rank '
             'lowered %d)' % (report['nodes_before'], report['nodes_after'],
                              report['pads_folded'], report['views_removed'],
                              report['lowered'])]
    lines += ['blocked: %s %s' % item for item in report['blocked']]
    if report['high_rank_left']:
        lines.append('rank above target left: %s' % report['high_rank_left'])
    if report['batch_axis_left']:
        lines.append('axis 0 is not 1 on: %s' % report['batch_axis_left'])
    return '\n'.join(lines)


def annotate_static_shapes(model):
    """Re-declare the shape of every internal tensor (the passes drop the old
    annotations), keeping only fully static ones. Returns the annotated model."""
    annotated = _inferred(model)
    kept = [info for info in annotated.graph.value_info
            if all(d.HasField('dim_value')
                   for d in info.type.tensor_type.shape.dim)]
    del annotated.graph.value_info[:]
    annotated.graph.value_info.extend(kept)
    return annotated


def optimize_file(path, output=None, target_rank=4, verify=False):
    """Read ``path``, optimise, write ``output`` (default: in place)."""
    import onnx
    model = onnx.load(path)
    report = optimize_batch1(model, target_rank, verify)
    onnx.save(annotate_static_shapes(model), output or path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('model')
    parser.add_argument('-o', '--output', help='default: rewrite in place')
    parser.add_argument('--target-rank', type=int, default=4)
    parser.add_argument('--verify', action='store_true',
                        help='require identical outputs in onnxruntime')
    args = parser.parse_args(argv)
    report = optimize_file(args.model, args.output, args.target_rank,
                           args.verify)
    print(format_report(report))
    if args.verify:
        print('max abs diff vs original: %g (%s)' % (
            report['max_abs_diff'],
            'tolerance: %d grouped linear(s) regrouped over frames'
            % report['frames_regrouped'] if report['frames_regrouped']
            else 'exact'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
