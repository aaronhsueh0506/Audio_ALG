#!/usr/bin/env python3
"""Export DFN2 heads as a stateless streaming ONNX graph.

The accelerator owns no persistent state. Each invocation receives the three
feature frames required by the input kernel, the GRU hidden state, and the
four-frame causal history used by ``df_convp``. It emits the heads for the
centre feature frame together with every updated state tensor.

STFT, feature normalisation, the three-frame feature window, deep-filter
composition and WOLA remain in the host C code.

Recurrent-state layouts
-----------------------

``--gru-state-layout`` selects how the five GRU hidden states are presented at
the graph boundary. The graph maths is identical either way; only the boundary
differs, and the exported metadata records which layout a file uses.

``split`` (default, state layout version 5, the shipped contract)
    One tensor per GRU in its native stacked form: ``h_encoder`` (1 layer),
    ``h_erb`` (2) and ``h_df`` (2), each ``(layers, 1, hidden)``. This is
    PyTorch's own hidden shape, and it matches ``DFN2ModelIOState``'s three
    ``[layers][hidden]`` arrays one for one, so a runtime binds each tensor
    to one contiguous field. It is what ``dfn2_model_io.h`` binds and what a
    board runs.

    ONNX has no stacked GRU op, so each two-layer stack still becomes two GRU
    nodes fed by per-layer Slices off its own input -- five GRU nodes in all.
    The trade this layout makes is therefore per-GRU, not per-layer, PTQ
    scales: the two layers of ``h_erb`` share one input tensor and one scale.
    Layout version 3 bought per-layer scales instead, by cloning the stacks
    into one-layer modules and publishing five tensors; this contract chooses
    the native shape and the exact match to the C struct.

``combined`` (state layout version 7, EXPERIMENTAL -- no C runtime binds it)
    All three as one ``(1, 5, 1, hidden)`` tensor named ``h_gru``, ordered
    encoder, erb, df -- ``DFN2ModelIOState``'s own memory order, since those
    three arrays are adjacent and equally wide, so a runtime binds the
    combined tensor to the same bytes with no gather.

    It exists to answer one question with numbers instead of argument: what
    does a single shared quantization scale cost? ``inference.py calib``
    publishes that per GRU under ``gru_state_slices``. One caveat before
    reading a verdict off it: the three states now reach their GRU nodes
    through Slices off one input, so an integer compiler may assign all three
    one scale -- which is precisely what the split layout keeps it from
    doing.

    Adopting it is a contract change, not a flag flip: it contradicts the
    one-tensor-per-GRU naming convention, and ``dfn2_model_io.h``, its commit
    API, the I/O tables and the C tests would all have to move with it.

Run from ``AINR/DeepFilterNet2``::

    python3 export_onnx.py --model output/dfn2_best.pth \
        --output output/dfn2_stream.onnx --verify

Generate deployment calibration beside this model with::

    python3 inference.py calib --model output/dfn2_best.pth \
        --wav-dir /path/to/noisy --frames 8192 --format bin \
        --output calib/dfn2

``calib`` exports the graph itself, from the same model instance in the same
process, so pass ``--gru-state-layout`` there rather than exporting twice.
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

from onnx_streaming_contract import validate_nctf_no_temporal_padding

INPUT_FRAMES = 3

# Kept numerically equal to DFN2_MODEL_IO_LAYOUT_VERSION in dfn2_model_io.h,
# which declares the caller-owned state struct this graph consumes and emits.
# tests/test_dfn2_contract.py pins the two together. Tensor names are part of
# this contract: runtimes bind by name.
#
# Version 3 published one tensor per recurrent LAYER, which needed the stacked
# GRUs cloned into one-layer modules; version 5 publishes one tensor per GRU in
# PyTorch's native stacked shape instead, which is also DFN2ModelIOState's own
# layout. ONNX has no stacked GRU op, so a two-layer stack still traces to two
# GRU nodes fed by per-layer Slices -- what moved is the PTQ scale, now one per
# GRU rather than one per layer. Version 4 was burned by the layout below.
STATE_LAYOUT_VERSION = 5

# EXPERIMENTAL, no C implementation: the combined-GRU-state layout below
# publishes a different state layout, so it must not claim version 5. Version 7
# is reserved for it in dfn2_model_io.h so a later C-side bump cannot take the
# same number. It exists to measure what a single recurrent tensor costs in PTQ
# accuracy before the contract is committed to.
COMBINED_STATE_LAYOUT_VERSION = 7

# Numbers no layout may claim again. 4 was the first combined layout, published
# before the split layout moved to per-GRU tensors; 6 was the combined
# ``(5,1,hidden)`` layout. Graphs exist that carry both.
# Stated here rather than only in dfn2_model_io.h's prose so a bump onto one
# fails a test instead of a review.
RETIRED_LAYOUT_VERSIONS = frozenset({4, 6})

# One entry per GRU, not per layer: each is that GRU's native stacked hidden
# ``(num_layers, 1, hidden)``, the shape PyTorch and DFN2ModelIOState both
# already use. The deep-filter pathway history stays a separate cache in every
# layout because it is one convolutional state with one distribution. See the
# module docstring for what each layout trades.
GRU_STATE_NAMES = (
    'h_encoder',
    'h_erb',
    'h_df',
)

COMBINED_GRU_STATE_NAME = 'h_gru'

CONTENT_INPUT_NAMES = ('erb', 'spec')
HEAD_OUTPUT_NAMES = ('erb_mask', 'df_coefs', 'df_alpha')
PATHWAY_STATE_NAME = 'df_convp_history'


class GruStateLayout(object):
    """One way of presenting the recurrent state at the graph boundary.

    A named layout rather than a boolean because a third one is already
    foreseeable: DFN2's five hidden states happen to share a shape, so they
    stack, but GTCRN's ten do not (six (1,1,16) plus four (1,33,8)), and a
    rollout there needs a flatten or a per-group scheme rather than a second
    boolean and a 2x2 of impossible states.

    The layout owns its own version, so a name set and the version that
    describes it cannot be passed separately and disagree.
    """

    def __init__(self, label, state_names, layout_version, combined):
        self.label = label
        self.state_names = state_names
        self.layout_version = layout_version
        self.combined = combined
        state = state_names + (PATHWAY_STATE_NAME,)
        self.input_names = CONTENT_INPUT_NAMES + state
        self.output_names = HEAD_OUTPUT_NAMES + tuple(
            name + '_out' for name in state
        )

    @property
    def state_handoff(self):
        """Which output hands each state input back. Zipped, not indexed."""
        return dict(zip(
            self.input_names[len(CONTENT_INPUT_NAMES):],
            self.output_names[len(HEAD_OUTPUT_NAMES):],
        ))


GRU_STATE_LAYOUTS = {
    'split': GruStateLayout(
        'split', GRU_STATE_NAMES, STATE_LAYOUT_VERSION, combined=False),
    'combined': GruStateLayout(
        'combined', (COMBINED_GRU_STATE_NAME,),
        COMBINED_STATE_LAYOUT_VERSION, combined=True),
}

DEFAULT_GRU_STATE_LAYOUT = 'split'


def resolve_gru_state_layout(layout):
    """Accept a layout, or its name, and return the layout."""
    if isinstance(layout, GruStateLayout):
        return layout
    try:
        return GRU_STATE_LAYOUTS[layout]
    except KeyError:
        raise ValueError('unknown gru_state_layout %r; expected one of %s'
                         % (layout, sorted(GRU_STATE_LAYOUTS)))


INPUT_NAMES = GRU_STATE_LAYOUTS[DEFAULT_GRU_STATE_LAYOUT].input_names
OUTPUT_NAMES = GRU_STATE_LAYOUTS[DEFAULT_GRU_STATE_LAYOUT].output_names


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
    """Run a LookaheadConv2d on [t-1,t,t+1] and emit only frame t."""
    if window.shape[2] != INPUT_FRAMES:
        raise ValueError('DFN2 expects exactly three input feature frames')
    layers = list(module.children())
    conv = layers[1]
    if conv.kernel_size[0] != INPUT_FRAMES:
        raise ValueError('unsupported DFN2 input temporal kernel')
    pad = layers[0]
    if not isinstance(pad, nn.ConstantPad2d):
        raise ValueError('unsupported DFN2 lookahead padding module')
    # The host supplied the complete temporal receptive field, so only the
    # frequency part of LookaheadConv2d's original padding remains here.
    output = F.pad(window, (pad.padding[0], pad.padding[1], 0, 0))
    for layer in layers[1:]:
        output = layer(output)
    if output.shape[2] != 1:
        raise RuntimeError('DFN2 centre convolution did not emit one frame')
    return output


def _squeezed_gru(module, value, hidden):
    raw = value
    value = module.linear_in(value)
    value, hidden_next = module.gru(value, hidden)
    value = module.linear_out(value)
    if module.gru_skip is not None:
        value = value + module.gru_skip(raw)
    return value, hidden_next


def _df_pathway_step(module, current, history):
    """Run the causal kernel-5 DF residual and return its next history."""
    kernel = module.conv.kernel_size[0]
    if history.shape[2] != kernel - 1:
        raise ValueError('df_convp_history has the wrong temporal depth')
    combined = torch.cat((history, current), dim=2)
    output = module.conv(combined)
    output = module.pw(output)
    output = module.bn(output)
    output = module.act(output)
    if output.shape[2] != 1:
        raise RuntimeError('DF pathway did not emit one frame')
    return output, combined[:, :, 1:]


def feature_windows(feature):
    """Yield every ``[t-1,t,t+1]`` window of a ``(B,C,T,F)`` feature stream.

    The stream is zero-padded on both ends, so window ``t`` is exactly what
    the host pushes into ``dfn2_model_io_push_features`` for frame ``t`` and
    the graph turns into ``heads[t]``.  Every replay site -- calibration
    capture and the contract tests -- must slide the window the same way, or
    a test can agree with itself while disagreeing with the exporter.
    """
    zero = torch.zeros_like(feature[:, :, :1])
    padded = torch.cat((zero, feature, zero), dim=2)
    for index in range(feature.shape[2]):
        yield padded[:, :, index:index + INPUT_FRAMES]


class StatelessDFN2Heads(nn.Module):
    """Functional one-output-frame twin of ``DeepFilterNet2.heads``."""

    def __init__(self, model, gru_state_layout=DEFAULT_GRU_STATE_LAYOUT):
        super().__init__()
        self.model = model
        # The wrapper is the single source of truth for which recurrent-state
        # layout this instance exports: every caller reads the layout back off
        # it rather than re-deriving one, so a graph cannot be written with one
        # layout and calibrated with the other.
        self.gru_state_layout = resolve_gru_state_layout(gru_state_layout)
        if model.mask_lookahead != 1:
            raise ValueError(
                'this three-frame export requires mask_lookahead=1'
            )
        if model.encoder.erb_conv0[1].kernel_size[0] != INPUT_FRAMES:
            raise ValueError('this exporter requires a temporal kernel of 3')
        if model.erb_dec.emb_gru.gru.num_layers != 2:
            raise ValueError('this exporter requires two ERB decoder GRU layers')
        if model.df_dec.df_gru.gru.num_layers != 2:
            raise ValueError('this exporter requires two DF decoder GRU layers')
        self.gru_state_slices = gru_state_slices(model)
        if self.gru_state_layout.combined:
            # Checked here beside the other constructor preconditions, not at
            # first use: a wrapper that cannot export must not construct.
            widths = {gru.hidden_size for gru in _state_grus(model)}
            if len(widths) != 1:
                raise ValueError(
                    'the combined recurrent state requires one hidden width '
                    'for every GRU; got %s' % sorted(widths)
                )
            self.combined_gru_state_shape = (
                1,
                sum(gru.num_layers for gru in _state_grus(model)),
                1,
                widths.pop(),
            )

    @property
    def input_names(self):
        return self.gru_state_layout.input_names

    @property
    def output_names(self):
        return self.gru_state_layout.output_names

    @property
    def state_layout_version(self):
        return self.gru_state_layout.layout_version

    def initial_inputs(self):
        """This instance's dummy inputs, in its own layout."""
        return initial_inputs(self.model, self.gru_state_layout)

    def forward(self, feat_erb_window, feat_spec_window, *state):
        if self.gru_state_layout.combined:
            combined_hidden, df_convp_history = state
            # The leading dimension is the graph invocation batch. Remove it
            # before feeding PyTorch's native (layers, batch, hidden) GRU
            # state; the exported combined ABI is (1, layers, 1, hidden).
            native_hidden = combined_hidden.squeeze(0)
            # One Slice per GRU off the shared input, in GRU_STATE_NAMES
            # order. Views, not copies: each still arrives at its own GRU node
            # carrying that GRU's whole stack.
            encoder_gru_hidden, erb_gru_hidden, df_gru_hidden = (
                native_hidden[start:stop]
                for _name, start, stop in self.gru_state_slices
            )
        else:
            (encoder_gru_hidden, erb_gru_hidden, df_gru_hidden,
             df_convp_history) = state
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
        c0_residual, pathway_history_next = _df_pathway_step(
            decoder.df_convp, c0, df_convp_history
        )
        c0_residual = c0_residual.permute(0, 2, 3, 1)
        alpha = decoder.df_fc_a(df_value)
        coefficients = decoder.df_out(df_value).view(
            batch, 1, decoder.df_bins, decoder.df_order * 2
        )
        coefficients = coefficients + c0_residual
        hidden_next = (encoder_hidden_next, erb_hidden_next, df_hidden_next)
        heads = (erb_mask, coefficients, alpha)
        if self.gru_state_layout.combined:
            # Concatenated in the same order the input was sliced, which is
            # also DFN2ModelIOState's memory order. Restore the graph batch
            # dimension at the boundary.
            combined_next = torch.cat(hidden_next, dim=0).reshape(
                self.combined_gru_state_shape
            )
            return heads + (combined_next, pathway_history_next)
        return heads + hidden_next + (pathway_history_next,)


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


def _split_hidden_inputs(model):
    """One native stacked zero hidden per GRU, in GRU_STATE_NAMES order."""
    return tuple(
        torch.zeros(gru.num_layers, 1, gru.hidden_size)
        for gru in _state_grus(model)
    )


def initial_inputs(model, gru_state_layout=DEFAULT_GRU_STATE_LAYOUT):
    enc_ch = model.encoder.erb_conv0[1].out_channels
    pathway_history = model.df_dec.df_convp.conv.kernel_size[0] - 1
    hidden = _split_hidden_inputs(model)
    if resolve_gru_state_layout(gru_state_layout).combined:
        hidden = (torch.cat(hidden, dim=0).unsqueeze(0),)
    return (
        torch.randn(1, 1, INPUT_FRAMES, model.n_erb),
        torch.randn(1, 2, INPUT_FRAMES, model.df_bins),
        *hidden,
        torch.zeros(1, enc_ch, pathway_history, model.df_bins),
    )


def gru_state_slice_report(combined, stats, slices):
    """What one shared PTQ scale costs each GRU of a combined state tensor.

    ``combined`` is the captured
    ``(frames, graph_batch, layers, gru_batch, hidden)`` array,
    ``stats`` produces one per-tensor range dict and ``slices`` comes from
    ``gru_state_slices``, so the per-slice entries are the same shape the
    calibration report publishes for every other input -- in the split layout
    those three entries exist already, and this restores them.

    The bits figure is named for the policy it assumes. A symmetric max-abs
    int8 scale is set by the widest slice, so a narrower one keeps only
    127 * own/shared levels and gives up log2(shared/own) of resolution. Under
    a percentile-clipped policy -- which the same report anticipates by
    publishing p001/p999 -- the number does not apply, hence the name rather
    than a bare `bits_lost`.

    Lives here beside the layout it describes so a test can reach it without a
    checkpoint and a wav directory.
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
                   gru_state_layout=DEFAULT_GRU_STATE_LAYOUT):
    """The exported graph's metadata.

    Separate from ``main`` so the contract test can compare
    ``state_layout_version`` against dfn2_model_io.h without an ONNX export.

    Takes the layout, not its parts: names, version and label all come from
    one object, so metadata that names ``h_gru`` while claiming the split
    version -- exactly the lying descriptor dfn2_model_io.h's version check
    exists to make a board refuse -- is unrepresentable.
    """
    layout = resolve_gru_state_layout(gru_state_layout)
    return {
        'model_family': 'DeepFilterNet2',
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'boundary': 'stateless_streaming_heads_explicit_state',
        'state_layout_version': layout.layout_version,
        'gru_state_layout': layout.label,
        'sample_rate': params['SR'],
        'n_fft': params['N_FFT'],
        'win_len': params['WIN_LEN'],
        'hop_len': params['HOP_LEN'],
        'input_feature_frames': INPUT_FRAMES,
        'output_frames_per_invocation': 1,
        'input_window_alignment': '[t-1,t,t+1] -> heads[t]',
        'temporal_padding_inside_graph': False,
        'frequency_padding_inside_graph': True,
        'accelerator_persistent_state': False,
        'host_updates_feature_window': True,
        'state_handoff': layout.state_handoff,
        'c_prepost': 'dfn2_process.c/dfn2_process.h',
        'c_model_io': 'dfn2_model_io.c/dfn2_model_io.h',
        'input_schema': _schema(layout.input_names, inputs),
        'output_schema': _schema(layout.output_names, outputs),
    }


def export_graph(wrapper, params, checkpoint_path, output_path,
                 opset=17, verify=False):
    """Write the ONNX graph plus its metadata JSON; optionally verify parity.

    Shared by the export CLI and ``inference.py calib``, so the calibration
    tensors and the graph they bind to always come from the same model
    instance in the same process.
    """
    layout = wrapper.gru_state_layout
    input_names = layout.input_names
    output_names = layout.output_names
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
                              gru_state_layout=layout)
    set_onnx_metadata(graph, metadata)
    _pin_static_output_shapes(graph, output_names, outputs)
    graph = _resolve_internal_shapes(graph)
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
    parser.add_argument(
        '--gru-state-layout', choices=sorted(GRU_STATE_LAYOUTS),
        default=DEFAULT_GRU_STATE_LAYOUT,
        help="'split' (default, layout version %d, the shipped contract) "
             "exports one tensor per GRU in its native stacked shape; "
             "'combined' (EXPERIMENTAL, layout version %d, no C runtime "
             "binds it) exports all three as one (1,5,1,hidden) tensor, "
             "so the three share one quantization scale. See the module "
             "docstring for the trade."
             % (STATE_LAYOUT_VERSION, COMBINED_STATE_LAYOUT_VERSION))
    args = parser.parse_args()

    try:
        from .inference import load_model
    except ImportError:  # direct ``python export_onnx.py`` execution
        from inference import load_model
    model, params = load_model(SimpleNamespace(
        config=args.config, model=args.model
    ))
    wrapper = StatelessDFN2Heads(
        model, gru_state_layout=args.gru_state_layout).eval()
    export_graph(wrapper, params, args.model, args.output,
                 opset=args.opset, verify=args.verify)
    print(args.output)


if __name__ == '__main__':
    main()
