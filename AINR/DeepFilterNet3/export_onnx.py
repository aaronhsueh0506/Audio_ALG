#!/usr/bin/env python3
"""Export DFN3 heads as a stateless streaming ONNX graph.

The accelerator owns no persistent state. Each invocation receives the three
feature frames required by the input kernel, one combined recurrent-state
tensor and the five-frame deep-filter pathway window. It emits the heads for
the centre feature frame together with every updated state tensor.

STFT, feature normalisation, the three-frame feature window, deep-filter
composition and WOLA remain in the host C code.

Window alignment
----------------

The graph is the same for every ``mask_lookahead`` L in ``[0, 2]``: it takes a
complete three-frame window and pads only frequency. What L fixes is which
frames the host puts in that window for the heads of frame ``t``:
``[t-(2-L), ..., t+L]`` (``[t-2,t-1,t]``, ``[t-1,t,t+1]``, ``[t,t+1,t+2]``),
the receptive field ``LookaheadConv2d`` gives the same frame offline. L and
``df_lookahead`` are part of the trained contract; the metadata records both
and the host (``DFN3_MASK_LOOKAHEAD`` / ``DFN3_DF_LOOKAHEAD`` in
``dfn3_process.h``) must be built with the same pair.

State boundary
--------------

Every state tensor is rank 4 and has the same shape at its input and at its
``*_out`` output, so a runtime binds both to one address and the updated state
is the next call's input with no copy (``state_binding`` in the metadata).

``h_gru`` ``(1, 5, 1, 256)``
    The five GRU hidden rows in ``dfn3_model_io.h`` order: encoder (row 0),
    ERB decoder (rows 1-2), deep-filter decoder (rows 3-4). Each GRU reads its
    rows through a Slice on dim 1 of the graph input followed by a squeeze,
    and the next state leaves as one Concat on dim 1. ONNX has no stacked GRU
    op, so the two-layer stacks are still two GRU nodes each.

``df_convp_history`` ``(1, 64, 5, 96)``
    The full causal window of the deep-filter pathway convolution. The graph
    drops the oldest frame with a Slice on the input, concatenates the new
    frame, convolves, and returns the whole concatenation as the next window.
    The Slice is the only consumer of the input, has explicit integer bounds
    and step 1, and no state output is produced by a Slice.

Run from ``AINR/DeepFilterNet3``::

    python3 export_onnx.py --model output/dfn3_best.pth \
        --output output/dfn3_stream.onnx --verify

Generate deployment calibration beside this model with::

    python3 inference.py calib --model output/dfn3_best.pth \
        --wav-dir /path/to/noisy --frames 8192 --format bin \
        --output calib/dfn3

``calib`` exports the graph itself, from the same model instance in the same
process.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AINR_ROOT = os.path.dirname(_SCRIPT_DIR)
if _AINR_ROOT not in sys.path:
    sys.path.insert(0, _AINR_ROOT)

from onnx_streaming_contract import (
    lookahead_window_alignment,
    lookahead_window_offsets,
    validate_inplace_state_from_metadata,
    validate_nctf_no_temporal_padding,
)

INPUT_FRAMES = 3

# Kept numerically equal to DFN3_MODEL_IO_LAYOUT_VERSION in dfn3_model_io.h,
# which declares the caller-owned state struct this graph consumes and emits.
# tests/test_dfn3_contract.py pins the two together. Tensor names are part of
# this contract: runtimes bind by name.
STATE_LAYOUT_VERSION = 9

# Layout numbers that are never reused. test_dfn3_contract.py rejects a
# STATE_LAYOUT_VERSION (and the header constant) that lands on one of them.
RETIRED_LAYOUT_VERSIONS = frozenset({4, 5, 6, 7, 8})

# The stateful GRUs, in row order of the combined tensor. Names label the
# calibration slices; the graph itself publishes only GRU_STATE_NAME.
GRU_STATE_NAMES = (
    'h_encoder',
    'h_erb',
    'h_df',
)

GRU_STATE_NAME = 'h_gru'
PATHWAY_STATE_NAME = 'df_convp_history'

CONTENT_INPUT_NAMES = ('erb', 'spec')
HEAD_OUTPUT_NAMES = ('erb_mask', 'df_coefs')
STATE_NAMES = (GRU_STATE_NAME, PATHWAY_STATE_NAME)

INPUT_NAMES = CONTENT_INPUT_NAMES + STATE_NAMES
OUTPUT_NAMES = HEAD_OUTPUT_NAMES + tuple(name + '_out' for name in STATE_NAMES)

# Which output hands each state input back. Zipped, not indexed.
STATE_HANDOFF = dict(zip(STATE_NAMES, OUTPUT_NAMES[len(HEAD_OUTPUT_NAMES):]))

# Both state tensors are read and written at one address.
STATE_BINDING = 'in_place_same_address'

# How each state input is consumed inside the graph (see the module
# docstring); the post-quantization check reads this back from the metadata.
# A window drops its first (oldest) frame and the new frame is appended.
STATE_SLICING = {
    GRU_STATE_NAME: {'kind': 'rows', 'axis': 1},
    PATHWAY_STATE_NAME: {'kind': 'window', 'axis': 2, 'drop': 'first'},
}


def _schema(names, tensors):
    return {
        name: [int(size) for size in tensor.shape]
        for name, tensor in zip(names, tensors)
    }


def set_onnx_metadata(graph, metadata):
    """Write a Python metadata mapping into an ONNX graph's model props.

    ONNX model properties are string-valued, so structured entries are JSON
    encoded (containers and bools) and scalars use ``str``; a consumer must
    see one rule everywhere.
    """
    import onnx
    onnx.helper.set_model_props(graph, {
        key: (json.dumps(value, sort_keys=True)
              if isinstance(value, (dict, list, bool)) else str(value))
        for key, value in metadata.items()
    })


def optimize_graph_file(path):
    """onnxoptimizer cleanup: drop the tracer's Identity/Constant/dead-end
    noise so the deployed graph carries only real ops. Skipped when the
    package is absent -- the graph is then correct but unoptimized."""
    try:
        import onnxoptimizer
        from onnxoptimizer import (
            get_available_passes,
            get_fuse_and_elimination_passes,
        )
    except ImportError:
        print('[skip] onnxoptimizer not installed; graph left unoptimized')
        return
    import onnx
    wanted = {
        'eliminate_nop_pad', 'eliminate_nop_transpose', 'eliminate_identity',
        'eliminate_deadend', 'eliminate_unused_initializer',
        'fuse_consecutive_transposes', 'fuse_consecutive_squeezes',
        'fuse_consecutive_unsqueezes', 'fuse_matmul_add_bias_into_gemm',
        'fuse_add_bias_into_conv',
    }
    passes = list(set(get_fuse_and_elimination_passes())
                  | (wanted & set(get_available_passes())))
    graph = onnx.load(path)
    before = len(graph.graph.node)
    graph = onnxoptimizer.optimize(graph, passes, fixed_point=True)
    onnx.save(graph, path)
    print('[onnxoptimizer] %d -> %d nodes' % (before, len(graph.graph.node)))

    try:
        import onnxruntime as ort
    except ImportError:
        print('[skip] onnxruntime not installed; constant folding skipped')
        return
    # onnxoptimizer only fuses/eliminates; the ConstantOfShape/Shape/Gather
    # chains the tracer leaves need actual constant EVALUATION. onnxruntime's
    # BASIC offline level is generic graph optimization (constant folding
    # included, no runtime-specific ops) and every export environment already
    # carries onnxruntime for the parity check.
    folded = path + '.fold'
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
    options.optimized_model_filepath = folded
    options.log_severity_level = 3
    ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])
    os.replace(folded, path)
    graph = onnx.load(path)
    print('[ort-fold] -> %d nodes' % len(graph.graph.node))


def _pin_static_output_shapes(graph, output_names, outputs):
    """Declare every graph output with its concrete traced shape.

    Shape inference cannot prove some extents static after GRU/Slice/Concat
    and leaves symbolic dim_params, but with every input static the true
    output shapes ARE static -- they are exactly the reference forward's
    tensor shapes. Accelerator toolchains require static I/O declarations,
    so the symbolic dims are overwritten with the measured ones.
    """
    shapes = {name: tuple(int(size) for size in tensor.shape)
              for name, tensor in zip(output_names, outputs)}
    for value_info in graph.graph.output:
        tensor_shape = value_info.type.tensor_type.shape
        tensor_shape.ClearField('dim')
        for size in shapes[value_info.name]:
            tensor_shape.dim.add().dim_value = size


def _resolve_internal_shapes(model):
    """Make every internal value_info static, or drop the annotation.

    torch's shape inference labels edges it cannot prove with symbolic
    unk__N dims even though static inputs fix every extent. onnxruntime's
    symbolic shape inference folds those to integers; any annotation it
    still cannot prove is removed -- a missing annotation is re-inferred by
    the consumer, a symbolic one is read as dynamic.
    """
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,
        )
        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    except Exception as error:
        print('[skip] symbolic shape inference unavailable: %s' % error)
    kept = [
        value_info for value_info in model.graph.value_info
        if all(dim.HasField('dim_value')
               for dim in value_info.type.tensor_type.shape.dim)
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    return model


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _lookahead_centre(module, window):
    """Run a LookaheadConv2d on its complete temporal window, emitting frame t."""
    if window.shape[2] != INPUT_FRAMES:
        raise ValueError('DFN3 expects exactly three input feature frames')
    layers = list(module.children())
    conv = layers[1]
    if conv.kernel_size[0] != INPUT_FRAMES:
        raise ValueError('unsupported DFN3 input temporal kernel')
    pad = layers[0]
    if not isinstance(pad, nn.ConstantPad2d):
        raise ValueError('unsupported DFN3 lookahead padding module')
    # The host supplied the complete temporal receptive field, so only the
    # frequency part of LookaheadConv2d's original padding remains here.
    output = F.pad(window, (pad.padding[0], pad.padding[1], 0, 0))
    for layer in layers[1:]:
        output = layer(output)
    if output.shape[2] != 1:
        raise RuntimeError('DFN3 centre convolution did not emit one frame')
    return output


def _squeezed_gru(module, value, hidden):
    raw = value
    value = module.linear_in(value)
    value, hidden_next = module.gru(value, hidden)
    value = module.linear_out(value)
    if module.gru_skip is not None:
        value = value + module.gru_skip(raw)
    return value, hidden_next


def _df_pathway_step(module, current, window):
    """Run the causal kernel-5 DF residual and return its next window.

    ``window`` is the previous call's full concatenation. Its oldest frame is
    dropped by a Slice on the graph input with explicit integer bounds, the
    new frame is appended, and the whole result is the next window.
    """
    kernel = module.conv.kernel_size[0]
    if window.shape[2] != kernel:
        raise ValueError('df_convp_history has the wrong temporal depth')
    combined = torch.cat((window[:, :, 1:kernel], current), dim=2)
    output = module.conv(combined)
    output = module.pw(output)
    output = module.bn(output)
    output = module.act(output)
    if output.shape[2] != 1:
        raise RuntimeError('DF pathway did not emit one frame')
    return output, combined


def window_frames(mask_lookahead):
    """Frame offsets, relative to ``t``, of the window behind ``heads[t]``.

    The one place the alignment is derived: ``t-(2-L) .. t+L``, which is the
    temporal padding ``LookaheadConv2d`` was built with
    (``INPUT_FRAMES - 1 - L`` frames before, ``L`` after).
    """
    return lookahead_window_offsets(mask_lookahead, INPUT_FRAMES)


def window_alignment(mask_lookahead):
    """``'[t-1,t,t+1] -> heads[t]'`` for the shipped ``mask_lookahead=1``."""
    return lookahead_window_alignment(mask_lookahead, INPUT_FRAMES)


def feature_windows(feature, mask_lookahead):
    """Return every window of a ``(B,C,T,F)`` feature stream, ``[t-(2-L)..t+L]``.

    The stream is zero-padded ``2 - L`` frames before and ``L`` after
    (``L = mask_lookahead``), so window ``t`` is exactly what the host holds in
    ``dfn3_model_io_push_features`` once frame ``t+L`` has been pushed and the
    graph turns into ``heads[t]``.  Every replay site -- calibration capture
    and the contract tests -- must slide the window the same way, so
    ``mask_lookahead`` has no default and is checked when this is called, not
    when the first window is taken.
    """
    offsets = window_frames(mask_lookahead)
    padded = F.pad(feature, (0, 0, -offsets[0], offsets[-1]))
    return (padded[:, :, index:index + INPUT_FRAMES]
            for index in range(feature.shape[2]))


class StatelessDFN3Heads(nn.Module):
    """Functional one-output-frame twin of ``DeepFilterNet3.heads``.

    ``batch_size`` is a fixed graph contract, not a dynamic ONNX axis.  Each
    batch element is an independent streaming lane with its own recurrent and
    convolutional state.  The normal deployment graph keeps the default of
    one; ``inference_batch.py`` uses larger values for NPU profiling.
    """

    def __init__(self, model, batch_size=1):
        super().__init__()
        self.model = model
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError('batch_size must be an integer')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.batch_size = batch_size
        # The host window for heads[t] is [t-(2-L), ..., t+L]; the model's own
        # input convolutions must have been built for that same L.
        offsets = window_frames(model.mask_lookahead)
        for name in ('erb_conv0', 'df_conv0'):
            module = getattr(model.encoder, name)
            pad, conv = module[0], module[1]
            if conv.kernel_size[0] != INPUT_FRAMES:
                raise ValueError('this exporter requires input temporal kernels of 3')
            if not isinstance(pad, nn.ConstantPad2d):
                raise ValueError('unsupported DFN3 lookahead padding module')
            if tuple(pad.padding[2:]) != (-offsets[0], offsets[-1]):
                raise ValueError(
                    '%s pads time by %r, not the window of mask_lookahead=%d' %
                    (name, pad.padding[2:], model.mask_lookahead)
                )
        # Only the input window and df_convp have explicit convolutional
        # history in this state layout. Intermediate convolutions run one
        # frame at a time, so a larger temporal kernel would silently replace
        # their past frames with zeros on every invocation.
        with_history = ('encoder.erb_conv0.', 'encoder.df_conv0.',
                        'df_dec.df_convp.')
        for name, layer in model.named_modules():
            if (isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d))
                    and layer.kernel_size[0] != 1
                    and not name.startswith(with_history)):
                raise ValueError(
                    'stateless export requires intermediate temporal kernels '
                    'of 1; %s has kernel %s and no exported history' %
                    (name, layer.kernel_size)
                )
        if model.erb_dec.emb_gru.gru.num_layers != 2:
            raise ValueError('this exporter requires two ERB decoder GRU layers')
        if model.df_dec.df_gru.gru.num_layers != 2:
            raise ValueError('this exporter requires two DF decoder GRU layers')
        self.gru_state_slices = gru_state_slices(model)
        self.gru_state_shape = gru_state_shape(model, self.batch_size)

    @property
    def mask_lookahead(self):
        return self.model.mask_lookahead

    @property
    def df_lookahead(self):
        return self.model.df_lookahead

    @property
    def input_names(self):
        return INPUT_NAMES

    @property
    def output_names(self):
        return OUTPUT_NAMES

    @property
    def state_layout_version(self):
        return STATE_LAYOUT_VERSION

    def initial_inputs(self):
        """This instance's dummy inputs."""
        return initial_inputs(self.model, batch_size=self.batch_size)

    def forward(self, feat_erb_window, feat_spec_window, gru_state,
                df_convp_history):
        # One Slice per GRU off the shared rank-4 input, on dim 1 (the layer
        # axis), then the layer rows become nn.GRU's own (layers, batch,
        # hidden). A profiling graph with B>1 carries the lanes on dim 0.
        if self.batch_size == 1:
            encoder_gru_hidden, erb_gru_hidden, df_gru_hidden = (
                gru_state[:, start:stop].squeeze(0)
                for _name, start, stop in self.gru_state_slices
            )
        else:
            encoder_gru_hidden, erb_gru_hidden, df_gru_hidden = (
                gru_state[:, start:stop].squeeze(2).permute(1, 0, 2)
                for _name, start, stop in self.gru_state_slices
            )
        model = self.model
        enc = model.encoder

        e0 = _lookahead_centre(enc.erb_conv0, feat_erb_window)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0 = _lookahead_centre(enc.df_conv0, feat_spec_window)
        c1 = enc.df_conv1(c0)

        batch = e3.shape[0]
        e3_flat = e3.permute(0, 2, 3, 1).reshape(batch, 1, -1)
        c1_flat = c1.permute(0, 2, 3, 1).reshape(batch, 1, -1)
        df_embedding = enc.df_fc_emb(c1_flat)
        if enc.enc_concat:
            embedding = torch.cat((e3_flat, df_embedding), dim=-1)
        else:
            embedding = e3_flat + df_embedding
        embedding, encoder_hidden_next = _squeezed_gru(
            enc.emb_gru, embedding, encoder_gru_hidden
        )

        erb = model.erb_dec
        erb_value, erb_hidden_next = _squeezed_gru(
            erb.emb_gru, embedding, erb_gru_hidden
        )
        erb_value = erb_value.reshape(
            batch, 1, erb.n_erb_4, erb.enc_ch
        ).permute(0, 3, 1, 2)
        erb_value = erb.convt3(erb.conv3p(e3) + erb_value)
        erb_value = erb.convt2(erb.conv2p(e2) + erb_value)
        erb_value = erb.convt1(erb.conv1p(e1) + erb_value)
        erb_mask = erb.conv0_out(erb.conv0p(e0) + erb_value)

        decoder = model.df_dec
        df_value, df_hidden_next = _squeezed_gru(
            decoder.df_gru, embedding, df_gru_hidden
        )
        if decoder.df_skip is not None:
            df_value = df_value + decoder.df_skip(embedding)
        c0_residual, pathway_window_next = _df_pathway_step(
            decoder.df_convp, c0, df_convp_history
        )
        c0_residual = c0_residual.permute(0, 2, 3, 1)
        coefficients = decoder.df_out(df_value).view(
            batch, 1, decoder.df_bins, decoder.df_order * 2
        )
        coefficients = coefficients + c0_residual
        heads = (erb_mask, coefficients)
        hidden_next = (encoder_hidden_next, erb_hidden_next, df_hidden_next)
        # Concatenated in the order the input was sliced, which is also
        # DFN3ModelIOState's memory order.
        if self.batch_size == 1:
            gru_state_next = torch.cat(
                [hidden.unsqueeze(0) for hidden in hidden_next], dim=1
            )
        else:
            gru_state_next = torch.cat(
                [hidden.permute(1, 0, 2).unsqueeze(2)
                 for hidden in hidden_next], dim=1
            )
        return heads + (gru_state_next, pathway_window_next)


def _state_grus(model):
    """The stateful GRUs, in GRU_STATE_NAMES order."""
    return (
        model.encoder.emb_gru.gru,
        model.erb_dec.emb_gru.gru,
        model.df_dec.df_gru.gru,
    )


def gru_state_slices(model):
    """``(name, start, stop)`` per GRU along the combined state's layer axis.

    Read off the module tree rather than written down, and shared by the
    graph's own slicing and the calibration report, so a stack depth can never
    be described one way in the graph and another in the report.
    """
    slices = []
    at = 0
    for name, gru in zip(GRU_STATE_NAMES, _state_grus(model)):
        slices.append((name, at, at + gru.num_layers))
        at += gru.num_layers
    return tuple(slices)


def gru_state_shape(model, batch_size=1):
    """Shape of the combined recurrent state: ``(B, layers, 1, hidden)``.

    With one lane this is ``(1, layers, gru_batch, hidden)`` read the other
    way round, so the batch-one tensor is nn.GRU's own hidden stack with a
    leading singleton.
    """
    grus = _state_grus(model)
    widths = {gru.hidden_size for gru in grus}
    if len(widths) != 1:
        raise ValueError(
            'the combined recurrent state requires one hidden width '
            'for every GRU; got %s' % sorted(widths)
        )
    return (batch_size, sum(gru.num_layers for gru in grus), 1,
            widths.pop())


def initial_inputs(model, batch_size=1):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError('batch_size must be an integer')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    enc_ch = model.encoder.erb_conv0[1].out_channels
    pathway_window = model.df_dec.df_convp.conv.kernel_size[0]
    return (
        torch.randn(batch_size, 1, INPUT_FRAMES, model.n_erb),
        torch.randn(batch_size, 2, INPUT_FRAMES, model.df_bins),
        torch.zeros(gru_state_shape(model, batch_size)),
        torch.zeros(
            batch_size, enc_ch, pathway_window, model.df_bins
        ),
    )


def gru_state_slice_report(combined, stats, slices):
    """What the single shared PTQ scale of ``h_gru`` costs each GRU.

    ``combined`` is the captured ``(frames, graph_batch, layers, gru_batch,
    hidden)`` array, ``stats`` produces one per-tensor range dict and
    ``slices`` comes from ``gru_state_slices``, so the per-slice entries match
    those the calibration report publishes for every other input. The bits
    figure assumes a symmetric max-abs int8 scale: the widest slice sets it, so
    a narrower one keeps 127 * own/shared levels and gives up log2(shared/own)
    of resolution (it does not apply to a percentile-clipped policy, hence the
    name rather than a bare `bits_lost`).
    """
    per_layer = {}
    for name, start, stop in slices:
        entry = stats(combined[:, :, start:stop])
        # max-abs from the range already computed, not a second pass.
        entry['max_abs'] = max(abs(entry['min']), abs(entry['max']))
        per_layer[name] = entry
    shared_max_abs = max(entry['max_abs'] for entry in per_layer.values())
    for entry in per_layer.values():
        own = entry['max_abs']
        entry['max_abs_symmetric_scale_bits_lost'] = (
            float(math.log2(shared_max_abs / own))
            if own > 0.0 else None
        )
    return {
        'order': [name for name, _start, _stop in slices],
        'shared_max_abs': shared_max_abs,
        'worst_bits_lost': max(
            (entry['max_abs_symmetric_scale_bits_lost']
             for entry in per_layer.values()
             if entry['max_abs_symmetric_scale_bits_lost'] is not None),
            default=None,
        ),
        'per_layer': per_layer,
    }


def build_metadata(checkpoint_path, params, inputs, outputs,
                   mask_lookahead, df_lookahead):
    """The exported graph's metadata.

    Separate from ``main`` so the contract test can compare
    ``state_layout_version`` against dfn3_model_io.h without an ONNX export.
    ``mask_lookahead`` and ``df_lookahead`` are the pair the checkpoint was
    trained for; the host is built with the same pair.
    """
    return {
        'model_family': 'DeepFilterNet3',
        'composition': 'dfn3_parallel_split',
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'boundary': 'stateless_streaming_heads_explicit_state',
        'state_layout_version': STATE_LAYOUT_VERSION,
        'sample_rate': params['SR'],
        'n_fft': params['N_FFT'],
        'win_len': params['WIN_LEN'],
        'hop_len': params['HOP_LEN'],
        'input_feature_frames': INPUT_FRAMES,
        'output_frames_per_invocation': 1,
        'fixed_batch_size': int(inputs[0].shape[0]),
        'batch_semantics': 'independent_streaming_lanes',
        'mask_lookahead': int(mask_lookahead),
        'df_lookahead': int(df_lookahead),
        'input_window_alignment': window_alignment(mask_lookahead),
        'temporal_padding_inside_graph': False,
        'frequency_padding_inside_graph': True,
        'accelerator_persistent_state': False,
        'host_updates_feature_window': True,
        'state_binding': STATE_BINDING,
        'state_handoff': STATE_HANDOFF,
        'state_slicing': STATE_SLICING,
        'c_prepost': 'dfn3_process.c/dfn3_process.h',
        'c_model_io': 'dfn3_model_io.c/dfn3_model_io.h',
        'input_schema': _schema(INPUT_NAMES, inputs),
        'output_schema': _schema(OUTPUT_NAMES, outputs),
    }


def export_graph(wrapper, params, checkpoint_path, output_path,
                 opset=17, verify=False):
    """Write the ONNX graph plus its metadata JSON; optionally verify parity.

    Shared by the export CLI and ``inference.py calib``, so the calibration
    tensors and the graph they bind to always come from the same model
    instance in the same process.
    """
    input_names = INPUT_NAMES
    output_names = OUTPUT_NAMES
    inputs = wrapper.initial_inputs()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )
    optimize_graph_file(output_path)

    import onnx
    graph = onnx.shape_inference.infer_shapes(onnx.load(output_path))
    onnx.checker.check_model(graph)
    validate_nctf_no_temporal_padding(graph, require_static=verify)
    metadata = build_metadata(checkpoint_path, params, inputs, outputs,
                              wrapper.mask_lookahead, wrapper.df_lookahead)
    set_onnx_metadata(graph, metadata)
    _pin_static_output_shapes(graph, output_names, outputs)
    graph = _resolve_internal_shapes(graph)
    validate_inplace_state_from_metadata(graph)
    onnx.save(graph, output_path)
    with open(os.path.splitext(output_path)[0] + '.json', 'w',
              encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')

    if verify:
        import onnxruntime as ort
        session = ort.InferenceSession(
            output_path, providers=['CPUExecutionProvider']
        )
        actual = session.run(None, {
            name: value.detach().numpy()
            for name, value in zip(input_names, inputs)
        })
        worst = max(
            float(np.max(np.abs(got - want.detach().numpy())))
            for got, want in zip(actual, outputs)
        )
        if worst > 2e-4:
            raise RuntimeError(
                'ONNX parity failed: max abs error %.6g' % worst
            )
        print('ONNX parity max_abs=%.6g' % worst)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        default=os.path.join(_SCRIPT_DIR, 'config.ini'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    try:
        from .inference import load_model
    except ImportError:  # direct ``python export_onnx.py`` execution
        from inference import load_model
    model, params = load_model(SimpleNamespace(
        config=args.config, model=args.model
    ))
    wrapper = StatelessDFN3Heads(model).eval()
    export_graph(wrapper, params, args.model, args.output,
                 opset=args.opset, verify=args.verify)
    print(args.output)


if __name__ == '__main__':
    main()
