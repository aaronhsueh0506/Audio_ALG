#!/usr/bin/env python3
"""Batch-1 graph cleanup for stateless streaming ONNX graphs.

The streaming exporters trace one frame at batch 1, so most tensors carry
size-1 axes (batch, time, a group of one) that the tracer turns into chains of
Squeeze/Unsqueeze/Reshape/Transpose and into 5-D and 6-D intermediates. With
every shape static those axes can be removed, and nothing here changes a
number: every pass is a pure re-labelling of axes (a view), never a different
arithmetic.

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
    their LEADING size-1 axes: the op is run on the reduced operands and a
    Reshape restores the original shape for the consumers, so ``merge_views``
    can then cancel the Reshape pairs. Supported: elementwise ops,
    Transpose, MatMul, ReduceSum/Mean/Max/Min, Softmax. A node it cannot
    lower is left as it is and reported.

Graph inputs, outputs, their names and shapes, the opset and the metadata are
untouched; the result is checked for that before it is returned. Semantic
rewrites (a different formulation of an op) belong to the model's exporter,
not here.

Usage::

    python3 onnx_batch1_optimizer.py model.onnx -o model_b1.onnx --verify

``--verify`` runs the original and the optimised graph in onnxruntime (graph
optimisation off) on random inputs and requires the outputs to be identical.
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
        perm = list(_attr(node, 'perm', list(range(len(in_shape)))[::-1]))
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


def _leading_drop(shape, target_rank):
    """Leading axes to drop to reach ``target_rank`` (all must be size 1),
    or None when they are not."""
    drop = len(shape) - target_rank
    if drop <= 0:
        return 0
    return drop if all(d == 1 for d in shape[:drop]) else None


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

    def operand(self, name, nodes):
        """(name, shape) of an input reduced to the target rank, or None."""
        from onnx import numpy_helper
        shape = self.info.shape(name)
        drop = None if shape is None else _leading_drop(shape, self.target)
        if drop is None:
            return None
        if drop == 0:
            return name, shape
        reduced = shape[drop:]
        init = self.info.initializers.get(name)
        out = self.namer.new('lo')
        if init is not None:
            self.model.graph.initializer.append(numpy_helper.from_array(
                numpy_helper.to_array(init).reshape(reduced), name=out))
        else:
            nodes.append(_reshape(self.namer, self.model, name, out, reduced))
        return out, reduced

    def build(self, node):
        op = node.op_type
        out_shape = self.info.shape(node.output[0])
        pre = []
        new = copy.deepcopy(node)
        del new.input[:]
        reduced_out = self.namer.new('lo')
        new.output[0] = reduced_out

        if op in _ELEMENTWISE:
            drop = _leading_drop(out_shape, self.target)
            if drop is None:
                return None
            for name in node.input:
                shape = self.info.shape(name) if name else ()
                if shape is None:
                    return None
                if len(shape) == len(out_shape):
                    reduced = self.operand(name, pre)
                    if reduced is None:
                        return None
                    new.input.append(reduced[0])
                elif len(shape) <= self.target:
                    new.input.append(name)
                else:
                    return None
        elif op == 'Transpose':
            drop = _leading_drop(out_shape, self.target)
            perm = list(_attr(node, 'perm', []))
            if drop is None or perm[:drop] != list(range(drop)):
                return None
            reduced = self.operand(node.input[0], pre)
            if reduced is None:
                return None
            new.input.append(reduced[0])
            _set_attr(new, 'perm', [p - drop for p in perm[drop:]])
        elif op == 'MatMul':
            drop = _leading_drop(out_shape, self.target)
            a = self.operand(node.input[0], pre)
            b = self.operand(node.input[1], pre)
            if drop is None or a is None or b is None or min(
                    len(a[1]), len(b[1])) < 2:
                return None
            batch = np.broadcast_shapes(a[1][:-2], b[1][:-2])
            if tuple(batch) + (a[1][-2], b[1][-1]) != out_shape[drop:]:
                return None
            new.input.extend([a[0], b[0]])
        elif op in _REDUCTIONS:
            shape = self.info.shape(node.input[0])
            axes = _reduction_axes(node, self.info)
            drop = None if shape is None else _leading_drop(shape,
                                                            self.target)
            if axes is None or drop is None:
                return None
            axes = [a % len(shape) for a in axes]
            if any(a < drop for a in axes):
                return None
            reduced = self.operand(node.input[0], pre)
            new.input.append(reduced[0])
            if len(node.input) > 1 and node.input[1]:
                from onnx import numpy_helper
                axes_name = self.namer.new('axes')
                self.model.graph.initializer.append(numpy_helper.from_array(
                    np.asarray([a - drop for a in axes], dtype=np.int64),
                    name=axes_name))
                new.input.append(axes_name)
            else:
                _set_attr(new, 'axes', [a - drop for a in axes])
            # The reduced run keeps the reduced axes (rank stays at target);
            # the Reshape below gives the original shape back.
            _set_attr(new, 'keepdims', 1)
        elif op == 'Softmax' and self.opset >= 13:
            drop = _leading_drop(out_shape, self.target)
            axis = _attr(node, 'axis', -1) % len(out_shape)
            reduced = self.operand(node.input[0], pre)
            if drop is None or axis < drop or reduced is None:
                return None
            new.input.append(reduced[0])
            _set_attr(new, 'axis', axis - drop)
        else:
            return None

        restore = _reshape(self.namer, self.model, reduced_out,
                           node.output[0], out_shape)
        return pre + [new, restore]


def lower_rank(model, target_rank=4):
    """Bring every tensor above ``target_rank`` down by dropping leading
    size-1 axes. Returns ``(lowered, blocked)``: the number of nodes
    rewritten and the ``(op_type, name)`` of those that could not be."""
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
    return lowered, blocked


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
    report['lowered'], report['blocked'] = lower_rank(model, target_rank)
    report['views_removed'] += merge_views(model)
    del model.graph.value_info[:]
    report['nodes_after'] = len(model.graph.node)
    report['high_rank_left'] = _high_rank(model, target_rank)
    if (_boundary(model) != boundary
            or [(o.domain, o.version) for o in model.opset_import] != opsets):
        raise RuntimeError('batch-1 optimizer changed the graph boundary')
    onnx.checker.check_model(model, full_check=True)
    if verify:
        report['max_abs_diff'] = verify_identical(original, model)
        if report['max_abs_diff'] != 0.0:
            raise RuntimeError(
                'batch-1 optimizer changed the outputs (max abs diff %g)'
                % report['max_abs_diff'])
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


def verify_identical(original, optimized, runs=3, seed=0):
    """Run both graphs in onnxruntime (graph optimisation off) on random
    inputs; return the largest absolute output difference."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    sessions = [
        ort.InferenceSession(m.SerializeToString(), options,
                             providers=['CPUExecutionProvider'])
        for m in (original, optimized)]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(runs):
        feeds = _random_inputs(original, rng)
        want = sessions[0].run(None, feeds)
        got = sessions[1].run(None, feeds)
        for a, b in zip(want, got):
            if np.array_equal(a, b, equal_nan=True):
                continue
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            worst = max(worst, float('inf') if np.isnan(diff).any()
                        else float(diff.max()))
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
    return '\n'.join(lines)


def optimize_file(path, output=None, target_rank=4, verify=False):
    """Read ``path``, optimise, write ``output`` (default: in place)."""
    import onnx
    model = onnx.load(path)
    report = optimize_batch1(model, target_rank, verify)
    onnx.save(model, output or path)
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
        print('max abs diff vs original: %g' % report['max_abs_diff'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
