#!/usr/bin/env python3
"""Align-ULCNet ONNX export -- stateless, one-frame streaming inference.

Run from the ``Audio_ALG`` directory.

Export with the delay depth stored in the checkpoint::

    python3 AIAEC/Align_ULCNet/export_onnx.py \
        --checkpoint output/align_ulcnet_best.pth \
        --output output/align_ulcnet_stream.onnx \
        --verify

Export a product graph with an explicit delay depth (D=4/8/16 each produces a
different graph and state layout)::

    python3 AIAEC/Align_ULCNet/export_onnx.py \
        --checkpoint output/align_ulcnet_best.pth \
        --max-delay-frames 8 \
        --output output/align_ulcnet_d8_stream.onnx \
        --verify

Capture PTQ calibration inputs through this model's inference entry::

    python3 AIAEC/Align_ULCNet/inference.py calib \
        --checkpoint output/align_ulcnet_best.pth \
        --primary-dir calibration/linear_error \
        --far-dir calibration/raw_far \
        --frames 8192 \
        --max-delay-frames 8 \
        --output AIAEC/Align_ULCNet/calib/align_ulcnet_d8.npz

To write one raw binary per input tensor per streaming frame instead::

    python3 AIAEC/Align_ULCNet/inference.py calib \
        --checkpoint output/align_ulcnet_best.pth \
        --primary-dir calibration/linear_error \
        --far-dir calibration/raw_far \
        --frames 8192 \
        --max-delay-frames 8 \
        --format bin \
        --output AIAEC/Align_ULCNet/calib/align_ulcnet_d8

Calibration straight from RAW microphone recordings (the checkpoint-matched
frozen PBFDKF derives the linear-error stems in-process and persists them
beside the artifact)::

    python3 AIAEC/Align_ULCNet/inference.py calib \\
        --checkpoint output/align_ulcnet_best.pth \\
        --primary-dir calibration/raw_mic \\
        --far-dir calibration/raw_far \\
        --primary-is-mic \\
        --frames 8192 \\
        --max-delay-frames 8 \\
        --output AIAEC/Align_ULCNet/calib/align_ulcnet_d8.npz

``--primary-dir`` and ``--far-dir`` must contain matching relative WAV
paths; differently-tokened names (mic_001.wav vs lpb_001.wav) pair via an
explicit ``--pair-replace mic:lpb`` rule.
Calibration intentionally uses raw far-end audio; deployment supplies aligned
far-end audio (the graph's ``far`` input is the AEC aligned-far seam on the
board -- raw far before acquisition, aligned far afterward).  The calibration and ONNX commands must use the same
``--max-delay-frames`` value.  NPZ output writes a sibling JSON contract;
binary output writes ``manifest.json`` inside its output directory.

The accelerator retains no state.  The CPU supplies past K/V features,
attention-score history and temporal-GRU hidden tensors on every invocation.
The graph returns only the new K/V/logit entries plus next GRU hidden tensors;
the CPU updates its caller-owned rings.  STFT/WOLA and PBFDKF stay outside the
graph in ``ulcnet_process.c`` and the AEC library respectively.  The exported
graph boundary follows the checkpoint and supports 16 kHz / FFT-window 512 /
hop 256 or 48 kHz / 1024 / 512.  The C pre/post must be compiled for the same
grid; one binary serves one grid.

Graph boundary layouts
----------------------

Two independent switches choose the boundary; the graph maths is identical
under all four combinations, and the exported metadata records the pair under
``feature_layout``/``gru_state_layout`` plus the version in
``LAYOUT_VERSIONS``.

``--feature-layout`` -- where the fixed front/back ends run.

``host`` (default, the shipped contract)
    Five signal inputs: ``error_mag``, ``far_mag``, ``error_cos``,
    ``error_sin``, ``error_ri``. Signed-power compression, magnitudes and the
    compressed-domain phase run outside the graph (``stream_features``;
    C: ``ulcnet_model_io_prepare``), the inverse power on the way back
    (``host_output``; C: ``ulcnet_model_io_commit``). Each feature keeps its
    own quantization scale, and the unlearned sqrt/atan2/pow never enter the
    quantized domain. This is what ``ulcnet_model_io.h`` binds.

``graph``
    Two signal inputs: ``error`` and ``far``, the raw RI spectra, with that
    same fixed math back inside the graph. Fewer tensors to bind and no host
    front end, paid for by putting sqrt/atan2/pow into the quantized domain
    and by ``error_cos``/``error_sin`` sharing whatever scale the compiler
    derives for them. It exists to measure that trade. It reproduces the
    pre-host-front-end boundary in every respect except the recurrent-state
    rank, so it carries its own version rather than the retired one that
    boundary once had.

``--gru-state-layout`` -- how the two subband GRU hiddens are presented.

``split`` (default, the shipped contract)
    ``h_gru0``/``h_gru1``, one tensor per subband GRU, each
    ``(1, GRU_LAYERS, 1, GRU_HIDDEN)``.

``combined``
    One ``(1, 2*GRU_LAYERS, 1, GRU_HIDDEN)`` tensor named ``h_gru``, h_gru0
    then h_gru1, stacked along dim 1. The attention caches (key/value/logit history) stay separate in
    both layouts: they are structural histories, not recurrent hidden state,
    and they do not share a distribution with the hiddens.

    It exists to measure what a single shared quantization scale costs. As on
    DFN2, the two hiddens reach their GRUs through Slices off one input, so
    an integer compiler may assign them one scale -- which is the whole
    question.

No C runtime binds anything but ``host``/``split``. Adopting another pair is a
contract change: ``ulcnet_model_io.h``, its prepare/commit API and the I/O
tables move with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from onnx_streaming_contract import validate_nctf_no_temporal_padding

from AIAEC.Align_ULCNet.inference import load_model
from AIAEC.training_common import (
    checkpoint_far_input_mode,
    far_input_mode_c_value,
)


# Version history. Version 3 fixed production wiring to aligned far while
# retaining the checkpoint's original far-input provenance separately;
# version 4 renamed the tensors (error/far inputs, output head, h_gru0/h_gru1
# hiddens, *_out states) -- runtimes bind by name; version 5 moved the fixed
# front/back ends (signed-power compression, magnitudes, phase cos/sin,
# inverse power) out of the graph onto the host.
#
# Versions 8-11 present every recurrent hidden as rank-4 NCHW, matching the
# three attention caches, which were already NCHW. The target platform's
# quantization toolchain requires rank-4 tensors; element count, row-major
# order and the per-tensor quantization scales are all unchanged by the rank,
# so this costs a version number and nothing else.
# The five SIGNAL inputs are deliberately not touched here: error_mag/far_mag/
# error_cos/error_sin are [1,1,n_freqs] and only error_ri is rank-4. If the
# toolchain turns out to reject those too, they move in their own bump --
# folding them in here would not have made that re-export any cheaper.
#
# The boundary is the PAIR (feature layout, recurrent-state layout), so the
# version belongs to the pair and is written down once rather than derived
# from either half. The rank change reaches every pair -- both split hiddens
# and the combined tensor -- so all four boundaries moved together and
# versions 3-7 are retired. No later contract may reclaim them: a number that
# denoted a rank-3 boundary must never also denote a rank-4 one.
#
# ULCNET_MODEL_IO_LAYOUT_VERSION in ulcnet_model_io.h is the deployed pair's
# version; test_export_streaming_ulcnet.py pins the two together. The other
# three are reserved there so a later C-side bump cannot take their numbers.
# Stated here rather than only in ulcnet_model_io.h's prose so a bump onto one
# fails a test instead of a review. 3, 4 and 5 were shipped rank-3 boundaries;
# 6 and 7 were rank-3 pairs reachable from the CLI and stamped into exported
# metadata, so they were allocated, not merely reserved.
RETIRED_LAYOUT_VERSIONS = frozenset(range(3, 8))
LAYOUT_VERSIONS = {
    ('host', 'split'): 8,
    ('host', 'combined'): 9,
    ('graph', 'split'): 10,
    ('graph', 'combined'): 11,
}
# The deployed pair's version, named because ulcnet_model_io.h pins it.
STATE_LAYOUT_VERSION = LAYOUT_VERSIONS[('host', 'split')]
# The deployed C front/back end hardcodes this exponent
# (ULCNET_MODEL_IO_COMPRESSION_EXP); export_graph refuses any checkpoint
# whose model carries a different value, because nothing downstream of the
# graph could detect the mismatch.
# The two product grids. SignalGrid admits more than these -- 16 kHz / 256 is
# constructible and the model runs on it -- but the deployed boundary is
# defined only for the pair the corpus is materialized on, so anything else is
# refused here rather than left to fail deep in a matmul or, worse, to export a
# graph no C build can bind.
SUPPORTED_GRIDS = ((16000, 512), (48000, 1024))
COMPRESSION_EXPONENT = 0.3
MIN_DELAY_DEPTH = 2
MAX_DELAY_DEPTH = 64
TA_CHANNELS = 32
SCORE_HISTORY_FRAMES = 4
GRU_LAYERS = 2
GRU_HIDDEN = 128

# Five separate feature inputs so every tensor keeps its own quantization
# scale; the fixed front end (signed-power compression, magnitude, phase
# cos/sin) runs on the HOST -- see stream_features, mirrored in C by
# ulcnet_model_io_prepare() --
# and the graph starts at the learned reorient/encoder compute.
HOST_SIGNAL_INPUT_NAMES = (
    'error_mag',
    'far_mag',
    'error_cos',
    'error_sin',
    'error_ri',
)
# The raw RI spectra the host front end consumes. Binding these instead makes
# the graph run that same fixed math itself, so there is no host front end and
# nothing to keep in step with ulcnet_model_io_prepare().
GRAPH_SIGNAL_INPUT_NAMES = ('error', 'far')
# The two subband GRU hiddens. Both are (1, GRU_LAYERS, 1, GRU_HIDDEN), so
# they stack along dim 1 into one (1, 2*GRU_LAYERS, 1, GRU_HIDDEN) tensor --
# see GRU_STATE_LAYOUTS. dim 0 is the singleton N of the shared NCHW
# convention, so it is never the stacking axis. The attention caches stay
# separate in both layouts: they are structural histories, not recurrent
# hidden state.
GRU_STATE_NAMES = ('h_gru0', 'h_gru1')
COMBINED_GRU_STATE_NAME = 'h_gru'
CACHE_STATE_NAMES = ('key_history', 'value_history', 'logit_history')
HEAD_OUTPUT_NAMES = ('output',)
CACHE_OUTPUT_NAMES = ('key_now', 'value_now', 'logit_now')


class FeatureLayout(object):
    """Which signal tensors the graph binds, and hence where the fixed
    front/back ends run. See the module docstring for the trade."""

    def __init__(self, label, signal_names, in_graph):
        self.label = label
        self.signal_names = signal_names
        self.in_graph = in_graph


FEATURE_LAYOUTS = {
    'host': FeatureLayout('host', HOST_SIGNAL_INPUT_NAMES, in_graph=False),
    'graph': FeatureLayout('graph', GRAPH_SIGNAL_INPUT_NAMES, in_graph=True),
}
DEFAULT_FEATURE_LAYOUT = 'host'


class GruStateLayout(object):
    """One way of presenting the recurrent state at the graph boundary.

    Named rather than boolean for the same reason as DFN2's: the models in
    this stack do not all stack the same way (GTCRN's ten hiddens carry two
    different shapes), so a third layout is foreseeable and a second boolean
    would be a 2x2 of impossible states.
    """

    def __init__(self, label, gru_names, combined):
        self.label = label
        self.gru_names = gru_names
        self.combined = combined
        self.state_names = CACHE_STATE_NAMES + gru_names


GRU_STATE_LAYOUTS = {
    'split': GruStateLayout('split', GRU_STATE_NAMES, combined=False),
    'combined': GruStateLayout(
        'combined', (COMBINED_GRU_STATE_NAME,), combined=True),
}
DEFAULT_GRU_STATE_LAYOUT = 'split'


class GraphLayout(object):
    """One complete graph boundary: the feature/state layout pair.

    The two axes are independent but the *version* is not a property of
    either half, so it is looked up for the pair. Everything downstream --
    tensor names, arity, metadata, the C version pin -- reads it off here, so
    a graph cannot be written under one pair and calibrated under another.
    """

    def __init__(self, feature, gru):
        self.feature = feature
        self.gru = gru
        self.layout_version = LAYOUT_VERSIONS[(feature.label, gru.label)]
        self.state_names = gru.state_names
        self.input_names = feature.signal_names + gru.state_names
        self.output_names = (
            HEAD_OUTPUT_NAMES + CACHE_OUTPUT_NAMES
            + tuple(name + '_out' for name in gru.gru_names)
        )

    @property
    def signal_inputs(self):
        return len(self.feature.signal_names)

    def unpack(self, tensors):
        """One flat ``input_names``-ordered tuple -> its three groups.

        The only place the boundary's arity is turned back into structure, so
        adding a cache or a feature moves one name tuple and nothing else.
        """
        caches = self.signal_inputs + len(CACHE_STATE_NAMES)
        return (tensors[:self.signal_inputs],
                tensors[self.signal_inputs:caches],
                tensors[caches:])

    @property
    def in_graph_features(self):
        return self.feature.in_graph

    @property
    def combined(self):
        return self.gru.combined


# Built once from the version table, so the set of valid boundaries is
# enumerated in exactly one place and every caller that resolves the same pair
# gets the same object.
GRAPH_LAYOUTS = {
    pair: GraphLayout(FEATURE_LAYOUTS[pair[0]], GRU_STATE_LAYOUTS[pair[1]])
    for pair in LAYOUT_VERSIONS
}


def resolve_layout(feature_layout=DEFAULT_FEATURE_LAYOUT,
                   gru_state_layout=DEFAULT_GRU_STATE_LAYOUT):
    """Accept a GraphLayout, or the two axis names, and return the pair."""
    if isinstance(feature_layout, GraphLayout):
        return feature_layout
    try:
        return GRAPH_LAYOUTS[(feature_layout, gru_state_layout)]
    except KeyError:
        raise ValueError('unknown layout %r; expected one of %s'
                         % ((feature_layout, gru_state_layout),
                            sorted(GRAPH_LAYOUTS)))


DEFAULT_LAYOUT = resolve_layout()
INPUT_NAMES = DEFAULT_LAYOUT.input_names
SIGNAL_INPUTS = DEFAULT_LAYOUT.signal_inputs

OUTPUT_NAMES = DEFAULT_LAYOUT.output_names


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


def _signed_power(value: Tensor, exponent: float) -> Tensor:
    return value.sign() * value.abs().pow(exponent)


class AlignUlcnetStreamingExport(nn.Module):
    """Functional, explicit-state twin of ``AlignULCNet.forward_stream``.

    K/V history is newest-first: slot zero is t-1.  Logit history is
    chronological: slot zero is t-4 and slot three is t-1, matching the
    causal score convolution's input order.
    """

    def __init__(self, model: nn.Module,
                 feature_layout=DEFAULT_FEATURE_LAYOUT,
                 gru_state_layout=DEFAULT_GRU_STATE_LAYOUT):
        super().__init__()
        self.model = model
        # The wrapper is the single source of truth for the boundary layout:
        # every caller reads it back off the wrapper rather than re-deriving
        # one, so a graph cannot be written with one layout and calibrated
        # with the other.
        self.layout = resolve_layout(feature_layout, gru_state_layout)
        self.delay_depth = int(model.max_delay_frames)
        if not MIN_DELAY_DEPTH <= self.delay_depth <= MAX_DELAY_DEPTH:
            # The common way to land here is a config that leaves
            # max_delay_frames unset. The 16 kHz grid pins 64 (the paper's
            # Dmax), but every other grid falls through to one physical
            # second, which is 94 frames at 48 kHz. Such a run trains to
            # completion and only fails here, so the message names the
            # remedy rather than just the bound.
            raise ValueError(
                'streaming export delay depth must be in [%d, %d], got %d. '
                'Set [model] max_delay_frames explicitly -- the one-second '
                'default overshoots this bound on the 48 kHz grid.' %
                (MIN_DELAY_DEPTH, MAX_DELAY_DEPTH, self.delay_depth)
            )
        grid = (model.grid.sample_rate, model.grid.n_fft)
        if grid not in SUPPORTED_GRIDS:
            raise ValueError(
                'streaming export supports %s, got %r' %
                (' and '.join('%d Hz / FFT %d' % g for g in SUPPORTED_GRIDS),
                 grid)
            )
        # win/hop need no test of their own: SignalGrid already requires
        # win_len == n_fft and hop_len == n_fft/2. What is NOT implied by the
        # grid is the C-SamFR sampling pair -- gamma and subband_bins change
        # reorient.width, and hence the K/V feature width, independently of
        # n_freqs.
        if (model.reorient.gamma, model.reorient.subband_bins) != (5, 2):
            raise ValueError(
                'streaming C boundary requires C-SamFR (gamma, subband_bins) '
                '= (5, 2), got (%r, %r)'
                % (model.reorient.gamma, model.reorient.subband_bins)
            )
        self.n_freqs = int(model.grid.n_freqs)
        self.ta_bins = ta_bins_for(model)

    def forward(self, *tensors: Tensor) -> Tuple[Tensor, ...]:
        layout = self.layout
        model = self.model
        signals, caches, hidden = layout.unpack(tensors)
        key_history, value_history, logit_history = caches
        if layout.combined:
            # One Slice per subband GRU off the shared input, in
            # GRU_STATE_NAMES order. Views, not copies.
            combined_hidden, = hidden
            # dim 1, not dim 0: dim 0 is the singleton N. Slicing dim 0 here
            # would return the whole tensor and an empty one WITHOUT raising.
            h_gru0 = combined_hidden[:, :GRU_LAYERS]
            h_gru1 = combined_hidden[:, GRU_LAYERS:]
        else:
            h_gru0, h_gru1 = hidden

        if layout.in_graph_features:
            # The same fixed math the host layout runs outside, run here
            # instead -- one implementation, so the two layouts cannot drift.
            error_mag, far_mag, error_cos, error_sin, error_ri = (
                stream_features(model, *signals)
            )
        else:
            error_mag, far_mag, error_cos, error_sin, error_ri = signals
        error_real = error_ri[..., 0]
        error_imag = error_ri[..., 1]

        error_feature = model.error_encoder(
            model.reorient(error_mag.unsqueeze(1))
        )
        far_feature = model.far_encoder(
            model.reorient(far_mag.unsqueeze(1))
        )

        attention = model.align
        query_now = attention.query(error_feature)
        key_now = attention.key(far_feature)
        value_now = attention.value(far_feature)

        # [B,C,1,D,F], delay slot zero is the current frame.
        key_candidates = torch.cat((key_now, key_history), dim=2).unsqueeze(2)
        value_candidates = torch.cat(
            (value_now, value_history), dim=2
        ).unsqueeze(2)
        logit_now = (
            query_now.unsqueeze(3) * key_candidates
        ).sum(dim=-1)

        # Reproduce StreamConv2dCell: four chronological history frames plus
        # current logits, frequency-axis padding only, then the raw Conv2d.
        score_input = torch.cat((logit_history, logit_now), dim=2)
        frequency_total = (attention.score.kf - 1) * attention.score.df
        frequency_left = frequency_total // 2
        score = attention.score.conv(F.pad(
            score_input,
            (frequency_left, frequency_total - frequency_left, 0, 0),
        )).squeeze(1)
        distribution = torch.softmax(score, dim=-1)
        aligned = (
            value_candidates * distribution[:, None, :, :, None]
        ).sum(dim=3)

        features = model.joint2(model.joint1(torch.cat(
            (error_feature, aligned), dim=1
        )))
        features = model.fgru(features)

        pieces = []
        hidden_next = []
        at = 0
        # nn.GRU's h_0 is strictly rank-3, so the boundary's singleton N is
        # dropped on the way in and restored on the way out. squeeze/unsqueeze
        # rather than reshape: reshape would silently reinterpret a tensor that
        # had the same element count with the axes in another order, which is
        # the failure this boundary convention exists to make impossible.
        hidden_inputs = (h_gru0.squeeze(0), h_gru1.squeeze(0))
        for index, width in enumerate(model.subband_widths):
            piece = features[..., at:at + width]
            batch, channels, _time, bins = piece.shape
            sequence = piece.permute(0, 2, 1, 3).reshape(
                batch, 1, channels * bins
            )
            output, hidden = model.subband_grus[index].gru(
                sequence, hidden_inputs[index]
            )
            pieces.append(output)
            hidden_next.append(hidden.unsqueeze(0))
            at += width

        joined = torch.cat(pieces, dim=-1)
        magnitude_mask = torch.sigmoid(model.mask_fc2(
            model.mask_act(model.mask_fc1(joined))
        ))
        intermediate = torch.stack((
            magnitude_mask * error_cos,
            magnitude_mask * error_sin,
        ), dim=1)
        stage2 = model.stage2_act(model.stage2_norm1(
            model.stage2_conv1(intermediate)
        ))
        stage2 = model.stage2_act(model.stage2_norm2(
            model.stage2_conv2(stage2)
        ))
        mask = model.complex_mask(stage2)

        mask_real = mask[:, 0]
        mask_imag = mask[:, 1]
        estimate_real = error_real * mask_real - error_imag * mask_imag
        estimate_imag = error_real * mask_imag + error_imag * mask_real
        output = torch.stack((estimate_real, estimate_imag), dim=-1)
        if layout.in_graph_features:
            output = host_output(model, output)
        # Otherwise a COMPRESSED-domain estimate: the fixed inverse signed
        # power runs on the host (host_output; C: ulcnet_model_io_commit).

        heads = (output, key_now, value_now, logit_now)
        if layout.combined:
            # Concatenated in the order it was sliced, which is also the
            # order ulcnet_model_io.h stores the two hiddens in.
            return heads + (torch.cat(tuple(hidden_next), dim=1),)
        return heads + tuple(hidden_next)


def ta_bins_for(model) -> int:
    """The K/V feature width the encoder emits.

    NOT a constant -- it follows n_freqs, so it is 26 on the 16 kHz grid and
    52 on the 48 kHz one. Read off the encoder rather than re-derived, so
    there is nothing here to drift from model.py. The C side derives the
    same number from its compiled grid as (BINS + 9) / 10, which is equal to
    this for the (gamma, subband_bins) pair the export guard enforces;
    test_c_ta_bins_matches_the_python_derivation evaluates both and compares
    them at each supported grid.

    ⚠ Not model.reorient.width. That is 52 at 16 kHz -- the same number this
    returns at 48 kHz -- so the wrong derivation looks right in a 48 kHz
    trace and is off by 2x on the shipped grid.
    """
    return model.encoded_width


def state_shapes(delay_depth: int, ta_bins: int) -> Dict[str, Tuple[int, ...]]:
    if not MIN_DELAY_DEPTH <= delay_depth <= MAX_DELAY_DEPTH:
        raise ValueError(
            'delay depth must be in [%d, %d]' %
            (MIN_DELAY_DEPTH, MAX_DELAY_DEPTH)
        )
    return {
        'key_history': (1, TA_CHANNELS, delay_depth - 1, ta_bins),
        'value_history': (1, TA_CHANNELS, delay_depth - 1, ta_bins),
        'logit_history': (
            1, TA_CHANNELS, SCORE_HISTORY_FRAMES, delay_depth
        ),
        'h_gru0': (1, GRU_LAYERS, 1, GRU_HIDDEN),
        'h_gru1': (1, GRU_LAYERS, 1, GRU_HIDDEN),
        COMBINED_GRU_STATE_NAME: (1, 2 * GRU_LAYERS, 1, GRU_HIDDEN),
    }


def stream_features(model, error_ri: Tensor, far_ri: Tensor):
    """Host-side fixed front end from RAW RI spectra ((1, 1, n_freqs, 2) each).

    Signed-power compression, magnitudes, and the compressed-domain phase as
    cos/sin -- everything unlearned, in fp32 (C: ulcnet_model_io_prepare).
    Returns
    the five graph signal inputs in INPUT_NAMES order; the graph starts at
    the learned reorient/encoder compute.
    """
    exponent = model.compression_exponent
    e_re = _signed_power(error_ri[..., 0], exponent)
    e_im = _signed_power(error_ri[..., 1], exponent)
    f_re = _signed_power(far_ri[..., 0], exponent)
    f_im = _signed_power(far_ri[..., 1], exponent)
    error_mag = (e_re.square() + e_im.square() + 1e-12).sqrt()
    far_mag = (f_re.square() + f_im.square() + 1e-12).sqrt()
    phase = torch.atan2(e_im, e_re)
    return (error_mag, far_mag, torch.cos(phase), torch.sin(phase),
            torch.stack((e_re, e_im), dim=-1))


def host_output(model, compressed_ri: Tensor) -> Tensor:
    """Host-side fixed back end: the inverse signed power the graph no
    longer applies (C: ulcnet_model_io_commit)."""
    inverse = 1.0 / model.compression_exponent
    return torch.stack((
        _signed_power(compressed_ri[..., 0], inverse),
        _signed_power(compressed_ri[..., 1], inverse),
    ), dim=-1)


def graph_signals(model, error_ri: Tensor, far_ri: Tensor,
                  layout) -> Tuple[Tensor, ...]:
    """The graph's signal inputs for one frame, from RAW RI spectra.

    The single place that knows how much of the fixed front end a layout
    leaves outside the graph, so the parity check and the calibration
    recorder cannot disagree about it.
    """
    if layout.in_graph_features:
        return (error_ri, far_ri)
    return stream_features(model, error_ri, far_ri)


def dummy_inputs(delay_depth: int, n_freqs: int, ta_bins: int,
                 layout=DEFAULT_LAYOUT) -> Tuple[Tensor, ...]:
    """Trace inputs for one frame. n_freqs and ta_bins have no defaults on
    purpose: a default would let a caller that forgot them trace one grid's
    graph while believing it had another."""
    layout = resolve_layout(layout)
    shapes = state_shapes(delay_depth, ta_bins)
    if layout.in_graph_features:
        signals = (
            torch.randn(1, 1, n_freqs, 2),         # error (raw RI)
            torch.randn(1, 1, n_freqs, 2),         # far (raw RI)
        )
    else:
        signals = (
            torch.randn(1, 1, n_freqs).abs(),          # error_mag
            torch.randn(1, 1, n_freqs).abs(),          # far_mag
            torch.randn(1, 1, n_freqs).clamp(-1, 1),   # error_cos
            torch.randn(1, 1, n_freqs).clamp(-1, 1),   # error_sin
            torch.randn(1, 1, n_freqs, 2),             # error_ri (compressed)
        )
    return signals + (
        torch.zeros(shapes['key_history']),
        torch.zeros(shapes['value_history']),
        torch.zeros(shapes['logit_history']),
    ) + tuple(
        torch.zeros(shapes[name]) for name in layout.gru.gru_names
    )


def next_state(
    current: Sequence[Tensor], outputs: Sequence[Tensor], delay_depth: int
) -> Tuple[Tensor, ...]:
    """Advance the caller-held state one frame, in whichever layout it is in.

    Layout-agnostic by construction: the caches are always the first three
    entries and the recurrent tail is whatever the graph returned, so this
    needs no branch and cannot fall out of step with GRU_STATE_LAYOUTS.
    """
    key_history, value_history, logit_history = current[:3]
    _enhanced, key_now, value_now, logit_now = outputs[:4]
    hidden_next = tuple(outputs[4:])
    if delay_depth > 1:
        key_history = torch.cat(
            (key_now, key_history[:, :, :delay_depth - 2]), dim=2
        )
        value_history = torch.cat(
            (value_now, value_history[:, :, :delay_depth - 2]), dim=2
        )
    logit_history = torch.cat(
        (logit_history[:, :, 1:], logit_now), dim=2
    )
    return (key_history, value_history, logit_history) + hidden_next


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _schema(names: Iterable[str], values: Iterable[Tensor]) -> Dict[str, list]:
    return {
        name: [int(size) for size in value.shape]
        for name, value in zip(names, values)
    }


def _write_metadata(
    output_path: str,
    checkpoint: str,
    model,
    contract: Dict,
    inputs: Sequence[Tensor],
    outputs: Sequence[Tensor],
    layout=DEFAULT_LAYOUT,
) -> Dict:
    layout = resolve_layout(layout)
    training_far_input_mode = checkpoint_far_input_mode(contract)
    deployed_far_input_mode = 'aligned_far'
    metadata = {
        'model_family': 'Align_ULCNet',
        'boundary': 'stateless_one_frame_delta_state',
        'state_layout_version': layout.layout_version,
        'feature_layout': layout.feature.label,
        'gru_state_layout': layout.gru.label,
        'compression_exponent': COMPRESSION_EXPONENT,
        'checkpoint_sha256': file_sha256(checkpoint),
        'sample_rate': model.grid.sample_rate,
        'n_fft': model.grid.n_fft,
        'win_len': model.grid.win_len,
        'hop_len': model.grid.hop_len,
        'frames_per_invocation': 1,
        'temporal_padding_inside_graph': False,
        'temporal_context': 'explicit_kv_logit_and_gru_state',
        'max_delay_frames': model.max_delay_frames,
        'training_far_input_mode': training_far_input_mode,
        'far_input_mode': deployed_far_input_mode,
        # Fixed production contract. Raw/aligned comparison remains available
        # only in sweep_delay_depth.py.
        'far_input_mode_c_value': far_input_mode_c_value(
            deployed_far_input_mode
        ),
        'accelerator_persistent_state': False,
        'cpu_delta_state_update': True,
        'tensor_dtype': 'float32',
        'complex_tensor_policy': 'real_imag_last_dimension',
        'input_schema': _schema(layout.input_names, inputs),
        'output_schema': _schema(layout.output_names, outputs),
        'k_v_history_order': 'newest_first',
        'logit_history_order': 'oldest_first',
        'production_streaming_equivalent': True,
        'c_frontend_postprocess': 'Align_ULCNet/ulcnet_process.c',
        'c_model_io_state': 'Align_ULCNet/ulcnet_model_io.c',
    }
    json_path = os.path.splitext(output_path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return metadata


def _set_onnx_metadata(graph, metadata: Dict) -> None:
    import onnx
    props = {
        key: (json.dumps(value, sort_keys=True)
              if isinstance(value, (dict, list, bool)) else str(value))
        for key, value in metadata.items()
    }
    onnx.helper.set_model_props(graph, props)


def _verify_onnx(output_path: str, wrapper: nn.Module,
                 inputs: Sequence[Tensor]) -> float:
    import onnxruntime as ort
    session = ort.InferenceSession(
        output_path, providers=['CPUExecutionProvider']
    )
    layout = wrapper.layout
    state = tuple(value.clone() for value in inputs[layout.signal_inputs:])
    worst = 0.0
    generator = torch.Generator().manual_seed(20260816)
    with torch.no_grad():
        for _ in range(2 * wrapper.delay_depth + 5):
            signals = graph_signals(
                wrapper.model,
                torch.randn(1, 1, wrapper.n_freqs, 2, generator=generator),
                torch.randn(1, 1, wrapper.n_freqs, 2, generator=generator),
                layout,
            )
            torch_inputs = signals + state
            expected = wrapper(*torch_inputs)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(layout.input_names, torch_inputs)
            })
            for got, want in zip(actual, expected):
                worst = max(worst, float(np.max(
                    np.abs(got - want.detach().numpy())
                )))
            state = next_state(state, expected, wrapper.delay_depth)
    return worst


def export_graph(model, checkpoint_path, output_path, opset=17,
                 verify=False, feature_layout=DEFAULT_FEATURE_LAYOUT,
                 gru_state_layout=DEFAULT_GRU_STATE_LAYOUT):
    """Write the streaming graph plus its metadata; optionally verify parity.

    Shared by the export CLI and the calib recorder, so the calibration
    tensors and the graph they bind to always come from the same wrapper in
    the same process. Returns the wrapper, its dummy inputs and the graph
    metadata for reuse.
    """
    if float(model.compression_exponent) != COMPRESSION_EXPONENT:
        raise ValueError(
            'checkpoint compression_exponent %r != %r: the deployed C '
            'front/back end (ULCNET_MODEL_IO_COMPRESSION_EXP in '
            'ulcnet_model_io.h) is fixed, and a graph exported for another '
            'exponent would deploy silently wrong'
            % (float(model.compression_exponent), COMPRESSION_EXPONENT)
        )
    wrapper = AlignUlcnetStreamingExport(
        model, feature_layout=feature_layout,
        gru_state_layout=gru_state_layout).eval()
    layout = wrapper.layout
    inputs = dummy_inputs(
        wrapper.delay_depth, wrapper.n_freqs, wrapper.ta_bins, layout)
    with torch.no_grad():
        outputs = wrapper(*inputs)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError('PyTorch streaming export reference is non-finite')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=list(layout.input_names),
        output_names=list(layout.output_names),
        opset_version=opset,
        do_constant_folding=True,
    )
    optimize_graph_file(output_path)

    import onnx
    graph = onnx.shape_inference.infer_shapes(onnx.load(output_path))
    onnx.checker.check_model(graph)
    validate_nctf_no_temporal_padding(graph, require_static=verify)
    checkpoint_data = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    metadata = _write_metadata(
        output_path,
        checkpoint_path,
        model,
        checkpoint_data['contract'],
        inputs,
        outputs,
        layout=layout,
    )
    _set_onnx_metadata(graph, metadata)
    _pin_static_output_shapes(graph, layout.output_names, outputs)
    graph = _resolve_internal_shapes(graph)
    onnx.save(graph, output_path)

    if verify:
        worst = _verify_onnx(output_path, wrapper, inputs)
        if worst > 3e-4:
            raise RuntimeError(
                'streaming ONNX parity failed: max_abs=%.6g' % worst
            )
        print('streaming ONNX parity max_abs=%.6g' % worst)
    return wrapper, inputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-delay-frames', type=int, default=None)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument(
        '--feature-layout', choices=sorted(FEATURE_LAYOUTS),
        default=DEFAULT_FEATURE_LAYOUT,
        help="'host' (default, the shipped contract) binds the five "
             "features the host front end computes; 'graph' binds the two "
             "raw RI spectra and runs that front end inside the graph "
             "instead. See LAYOUT_VERSIONS for the resulting version.")
    parser.add_argument(
        '--gru-state-layout', choices=sorted(GRU_STATE_LAYOUTS),
        default=DEFAULT_GRU_STATE_LAYOUT,
        help="'split' (default, the shipped contract) exports one tensor "
             "per subband GRU; 'combined' stacks both into one "
             "(1, 2*layers, 1, hidden) tensor. Every recurrent state is "
             "rank-4 NCHW either way, matching the attention caches, which "
             "stay separate in both layouts.")
    args = parser.parse_args()

    model, _grid, _linear_contract = load_model(
        args.checkpoint, 'cpu', max_delay_frames=args.max_delay_frames
    )
    model.eval()
    export_graph(model, args.checkpoint, args.output,
                 opset=args.opset, verify=args.verify,
                 feature_layout=args.feature_layout,
                 gru_state_layout=args.gru_state_layout)
    print(args.output)


if __name__ == '__main__':
    main()
