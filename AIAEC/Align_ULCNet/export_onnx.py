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

Capture PTQ calibration inputs through this model's inference entry (add
``--format bin`` for one raw binary per input tensor per streaming frame)::

    python3 AIAEC/Align_ULCNet/inference.py calib \
        --checkpoint output/align_ulcnet_best.pth \
        --primary-dir calibration/linear_error \
        --far-dir calibration/raw_far \
        --frames 8192 \
        --max-delay-frames 8 \
        --output AIAEC/Align_ULCNet/calib/align_ulcnet_d8.npz

For RAW microphone recordings pass ``--primary-dir calibration/raw_mic
--primary-is-mic``: the checkpoint-matched frozen PBFDKF derives the
linear-error stems in-process and persists them beside the artifact.
``--primary-dir`` and ``--far-dir`` must contain matching relative WAV paths;
differently-tokened names (mic_001.wav vs lpb_001.wav) pair via an explicit
``--pair-replace mic:lpb`` rule.

Calibration and deployment both use raw far-end audio: the linear AEC aligns
its private copy only to form ``linear_error`` and the graph's TA block aligns
raw far against that error. The calibration and ONNX commands must use the same
``--max-delay-frames``. NPZ output writes a sibling JSON contract; binary
output writes ``manifest.json`` inside its output directory.

STFT/WOLA and PBFDKF stay outside the graph (``ulcnet_process.c`` and the AEC
library). The graph boundary follows the checkpoint and supports 16 kHz / FFT
window 512 / hop 256 or 48 kHz / 1024 / 512; the C pre/post must be compiled
for the same grid (one binary serves one grid).

Graph boundary (state layout version ``LAYOUT_VERSION``)
--------------------------------------------------------

The accelerator retains no state. Four host-computed signal inputs
(``error_mag``, ``far_mag``, ``error_cos``, ``error_sin``, each ``[1,1,F]``;
signed-power compression, magnitudes and the compressed-domain phase run on
the host, see ``stream_features``) are followed by four rank-4 state inputs,
and every state input has an ``*_out`` output of exactly the same shape:

    key_history    [1, 32, D, TA]   newest frame first (slot 0 is the newest)
    value_history  [1, 32, D, TA]   newest frame first
    logit_history  [1, 32, 5, D]    oldest frame first (slot 4 is the newest)
    h_gru          [1, 4, 1, 128]   subband GRU 0 layers, then subband GRU 1

Each ``*_out`` is the FULL next window. The shift runs in the graph: the
history is cut by a Slice on the graph input (K/V keep slots ``0..D-2``, the
logit window keeps frames ``1..4``) and the new frame is concatenated behind or
in front of it, so the Concat result is the output and no output is produced by
a Slice. A runtime can bind each ``*_out`` to the address of its input; it must
finish reading every state input before it writes any state output. The graph
ends at the ``complex_mask`` 1x1 Conv and returns the learned mask itself,
``output`` = ``[1, 2, 1, n_freqs]`` (plane 0 real, plane 1 imaginary); the
fixed back end (complex multiply of the compressed error by that mask, then
the inverse power) runs on the host (``host_back_end``; C:
``ulcnet_model_io_commit``).

``export_graph`` validates the written file with
``validate_inplace_state_contract`` and records ``state_binding`` and
``state_window_drop`` in its metadata.
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

from AIAEC._onnx_contract import (
    STATE_BINDING,
    validate_inplace_state_contract,
    validate_nctf_no_temporal_padding,
)

from AIAEC.Align_ULCNet.inference import load_model
from AIAEC.dataset_gen.linear_aec import LinearAecContract
from AIAEC.training_common import (
    DEPLOYED_FAR_INPUT_MODE,
    checkpoint_far_input_mode,
    far_input_mode_c_value,
    require_not_retired_behavior,
)


# The state boundary's version. ULCNET_MODEL_IO_LAYOUT_VERSION in
# ulcnet_model_io.h carries the same number (test_export_streaming_ulcnet.py
# pins the two). Versions 3-17 are never reused: a number must not denote two
# boundaries.
LAYOUT_VERSION = 18
RETIRED_LAYOUT_VERSIONS = frozenset(range(3, 18))
STATE_LAYOUT_VERSION = LAYOUT_VERSION
# The deployed C front/back end hardcodes this exponent
# (ULCNET_MODEL_IO_COMPRESSION_EXP); export_graph refuses a checkpoint whose
# model carries a different value, since nothing downstream could detect it.
# The two product grids: SignalGrid admits more (16 kHz / 256 is constructible)
# but the deployed boundary is defined only for the pair the corpus is
# materialized on, so anything else is refused here rather than failing deep in
# a matmul or exporting a graph no C build can bind.
SUPPORTED_GRIDS = ((16000, 512), (48000, 1024))
COMPRESSION_EXPONENT = 0.3
MIN_DELAY_DEPTH = 2
MAX_DELAY_DEPTH = 64
TA_CHANNELS = 32
SCORE_WINDOW_FRAMES = 5
GRU_SUBBANDS = 2
GRU_LAYERS = 2
GRU_HIDDEN = 128

# Four separate feature inputs so every tensor keeps its own quantization
# scale; the fixed front end (signed-power compression, magnitude, phase
# cos/sin) runs on the HOST -- see stream_features, mirrored in C by
# ulcnet_model_io_prepare() --
# and the graph starts at the learned reorient/encoder compute. The compressed
# error spectrum stream_features also returns is not a graph input: only the
# host back end (host_back_end) multiplies by it.
SIGNAL_INPUT_NAMES = (
    'error_mag',
    'far_mag',
    'error_cos',
    'error_sin',
)
# The two subband GRU hiddens are stacked along dim 1 of one rank-4 tensor,
# h_gru0 then h_gru1; dim 0 is the singleton N of the shared NCHW convention,
# so it is never the stacking axis.
GRU_STATE_NAME = 'h_gru'
# The end each window drops before the new frame joins: the key and value
# histories are newest first, the logit history is time ordered.
WINDOW_STATE_DROPS = {
    'key_history': 'last',
    'value_history': 'last',
    'logit_history': 'first',
}
WINDOW_STATE_NAMES = tuple(WINDOW_STATE_DROPS)
STATE_NAMES = WINDOW_STATE_NAMES + (GRU_STATE_NAME,)
HEAD_OUTPUT_NAMES = ('output',)
STATE_OUTPUT_NAMES = tuple(name + '_out' for name in STATE_NAMES)

INPUT_NAMES = SIGNAL_INPUT_NAMES + STATE_NAMES
OUTPUT_NAMES = HEAD_OUTPUT_NAMES + STATE_OUTPUT_NAMES
SIGNAL_INPUTS = len(SIGNAL_INPUT_NAMES)


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
    print('[ort-fold] -> %d nodes' % len(onnx.load(path).graph.node))


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

    K/V windows are newest-first: slot zero is the current frame, slot ``d`` is
    ``t - d``.  The logit window is chronological: slot four is the current
    frame and slot zero is ``t - 4``, matching the causal score convolution's
    input order.
    """

    input_names = INPUT_NAMES
    output_names = OUTPUT_NAMES
    signal_inputs = SIGNAL_INPUTS
    layout_version = LAYOUT_VERSION

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
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

    def forward(self, error_mag: Tensor, far_mag: Tensor, error_cos: Tensor,
                error_sin: Tensor, key_history: Tensor,
                value_history: Tensor, logit_history: Tensor,
                h_gru: Tensor) -> Tuple[Tensor, ...]:
        model = self.model
        depth = self.delay_depth
        # One Slice per subband GRU off the shared input. dim 1, not dim 0:
        # dim 0 is the singleton N. Slicing dim 0 here would return the whole
        # tensor and an empty one WITHOUT raising.
        h_gru0 = h_gru[:, :GRU_LAYERS]
        h_gru1 = h_gru[:, GRU_LAYERS:GRU_SUBBANDS * GRU_LAYERS]

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

        # [B,C,D,F], delay slot zero is the current frame. The Slice keeps
        # the D-1 newest slots of the input window and the Concat result is
        # both the attention candidates and the next window.
        key_window = torch.cat((key_now, key_history[:, :, :depth - 1]), dim=2)
        value_window = torch.cat(
            (value_now, value_history[:, :, :depth - 1]), dim=2)
        key_candidates = key_window.unsqueeze(2)
        value_candidates = value_window.unsqueeze(2)
        logit_now = (
            query_now.unsqueeze(3) * key_candidates
        ).sum(dim=-1)

        # Reproduce StreamConv2dCell: the four newest chronological frames of
        # the window plus current logits, frequency-axis padding only, then
        # the raw Conv2d. The Concat result is the next window.
        score_input = torch.cat(
            (logit_history[:, :, 1:SCORE_WINDOW_FRAMES], logit_now), dim=2)
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
        # The graph ends at the mask Conv: the host multiplies the compressed
        # error by it and applies the inverse signed power (host_back_end; C:
        # ulcnet_model_io_commit).
        output = model.complex_mask(stage2)

        # Concatenated in the order it was sliced, which is also the order
        # ulcnet_model_io.h stores the two hiddens in.
        return (output, key_window, value_window, score_input,
                torch.cat(tuple(hidden_next), dim=1))


def ta_bins_for(model) -> int:
    """The K/V feature width the encoder emits.

    NOT a constant: it follows n_freqs (26 on the 16 kHz grid, 52 on the 48 kHz
    one), so it is read off the encoder. The C side derives it from its
    compiled grid as (BINS + 9) / 10 (pinned by
    test_c_ta_bins_matches_the_python_derivation). ⚠ Not model.reorient.width,
    which is 52 at 16 kHz.
    """
    return model.encoded_width


def state_shapes(delay_depth: int, ta_bins: int) -> Dict[str, Tuple[int, ...]]:
    if not MIN_DELAY_DEPTH <= delay_depth <= MAX_DELAY_DEPTH:
        raise ValueError(
            'delay depth must be in [%d, %d]' %
            (MIN_DELAY_DEPTH, MAX_DELAY_DEPTH)
        )
    return {
        'key_history': (1, TA_CHANNELS, delay_depth, ta_bins),
        'value_history': (1, TA_CHANNELS, delay_depth, ta_bins),
        'logit_history': (
            1, TA_CHANNELS, SCORE_WINDOW_FRAMES, delay_depth
        ),
        GRU_STATE_NAME: (1, GRU_SUBBANDS * GRU_LAYERS, 1, GRU_HIDDEN),
    }


def stream_features(model, error_ri: Tensor, far_ri: Tensor):
    """Host-side fixed front end from RAW RI spectra ((1, 1, n_freqs, 2) each).

    Signed-power compression, magnitudes and the compressed-domain phase as
    cos/sin, all unlearned and in fp32 (C: ulcnet_model_io_prepare). Returns
    ``(error_mag, far_mag, error_cos, error_sin, error_ri)``: the first four
    are the graph's signal inputs in INPUT_NAMES order, ``error_ri`` is the
    compressed error spectrum the host keeps for host_back_end.
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


def apply_mask_ri(error_ri: Tensor, mask: Tensor) -> Tensor:
    """Complex multiply of the compressed error ``[1,1,F,2]`` by the planar
    mask ``[1,2,1,F]``, giving the compressed estimate ``[1,1,F,2]``."""
    error_real, error_imag = error_ri[..., 0], error_ri[..., 1]
    mask_real, mask_imag = mask[:, 0], mask[:, 1]
    return torch.stack((
        error_real * mask_real - error_imag * mask_imag,
        error_real * mask_imag + error_imag * mask_real,
    ), dim=-1)


def host_output(model, compressed_ri: Tensor) -> Tensor:
    """The inverse signed power of a compressed estimate."""
    inverse = 1.0 / model.compression_exponent
    return torch.stack((
        _signed_power(compressed_ri[..., 0], inverse),
        _signed_power(compressed_ri[..., 1], inverse),
    ), dim=-1)


def host_back_end(model, error_ri: Tensor, mask: Tensor) -> Tensor:
    """Host-side fixed back end: mask multiply, then inverse signed power
    (C: ulcnet_model_io_commit)."""
    return host_output(model, apply_mask_ri(error_ri, mask))


def graph_signals(model, error_ri: Tensor, far_ri: Tensor) -> Tuple[Tensor, ...]:
    """The graph's four signal inputs for one frame, from RAW RI spectra
    ``(1, 1, n_freqs, 2)``: the part of ``stream_features`` the graph binds.
    The single place the parity check and the calibration recorder derive
    them, so the two cannot disagree."""
    return stream_features(model, error_ri, far_ri)[:SIGNAL_INPUTS]


def dummy_inputs(delay_depth: int, n_freqs: int, ta_bins: int
                 ) -> Tuple[Tensor, ...]:
    """Trace inputs for one frame. n_freqs and ta_bins have no defaults on
    purpose: a default would let a caller that forgot them trace one grid's
    graph while believing it had another."""
    shapes = state_shapes(delay_depth, ta_bins)
    signals = (
        torch.randn(1, 1, n_freqs).abs(),          # error_mag
        torch.randn(1, 1, n_freqs).abs(),          # far_mag
        torch.randn(1, 1, n_freqs).clamp(-1, 1),   # error_cos
        torch.randn(1, 1, n_freqs).clamp(-1, 1),   # error_sin
    )
    return signals + tuple(torch.zeros(shapes[name]) for name in STATE_NAMES)


def next_state(outputs: Sequence[Tensor]) -> Tuple[Tensor, ...]:
    """The caller-held state for the next frame.

    The graph returns every state tensor in full, in the order the state
    inputs are bound, so the state is the output tail unchanged -- the same
    assignment an in-place runtime performs by binding each ``*_out`` to its
    input's address. It names no tensor.
    """
    return tuple(outputs[len(HEAD_OUTPUT_NAMES):])


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


# Which spectrum feeds the graph's error input: the analysis seam of lib/aec
# (formed_hop / beam.spectrum), the real selected linear residual. The
# rendering base (aec_get_output_base) is never fed to the model.
ERROR_INPUT_SEAM = 'analysis_formed'


def _checkpoint_linear_aec(contract: Dict) -> LinearAecContract:
    """The front-end contract the checkpoint was trained against.

    Parsed through LinearAecContract.from_dict, so a retired or foreign version
    is refused here as well, not only in load_model; a retired behaviour hash
    under the current version is refused too (from_dict does not look at it).
    """
    if 'linear_aec' not in contract:
        raise ValueError(
            'checkpoint contract has no linear_aec: the graph metadata must '
            'record the front end the weights were trained against'
        )
    linear_aec = LinearAecContract.from_dict(contract['linear_aec'])
    require_not_retired_behavior(linear_aec, 'checkpoint contract')
    return linear_aec


def _write_metadata(
    output_path: str,
    checkpoint: str,
    model,
    contract: Dict,
    inputs: Sequence[Tensor],
    outputs: Sequence[Tensor],
) -> Dict:
    training_far_input_mode = checkpoint_far_input_mode(contract)
    deployed_far_input_mode = DEPLOYED_FAR_INPUT_MODE
    if training_far_input_mode != deployed_far_input_mode:
        raise ValueError(
            'checkpoint/deployment far-input mismatch: training used '
            f'{training_far_input_mode!r}, deployment requires '
            f'{deployed_far_input_mode!r}'
        )
    linear_aec = _checkpoint_linear_aec(contract)
    metadata = {
        'model_family': 'Align_ULCNet',
        'boundary': 'stateless_one_frame_full_state',
        'state_layout_version': LAYOUT_VERSION,
        'state_binding': STATE_BINDING,
        'state_window_drop': WINDOW_STATE_DROPS,
        'feature_layout': 'host',
        'gru_state_layout': 'combined',
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
        # only in sweep_delay_depth.py; export refuses a training mismatch.
        'far_input_mode_c_value': far_input_mode_c_value(
            deployed_far_input_mode
        ),
        # Front end the weights were trained on, copied from the checkpoint.
        # The integrator may compare or log linear_aec_contract_version against
        # ULCNET_MODEL_IO_FRONTEND_CONTRACT; C init never refuses by front end.
        # Enforcement is on the Python side (export, inference, calibration)
        # and in the committed tests.
        'linear_aec_contract_version': linear_aec.version,
        'aec_behavior_hash': linear_aec.aec_behavior_hash,
        'error_input_seam': ERROR_INPUT_SEAM,
        'accelerator_persistent_state': False,
        'cpu_delta_state_update': False,
        'tensor_dtype': 'float32',
        'complex_tensor_policy': 'real_imag_last_dimension',
        'input_schema': _schema(INPUT_NAMES, inputs),
        'output_schema': _schema(OUTPUT_NAMES, outputs),
        'k_v_history_order': 'newest_first',
        'logit_history_order': 'oldest_first',
        'score_window_frames': SCORE_WINDOW_FRAMES,
        'production_streaming_equivalent': True,
        'c_frontend_postprocess': 'Align_ULCNet/ulcnet_process.c',
        'c_model_io_state': 'Align_ULCNet/ulcnet_model_io.c',
    }
    _write_sidecar(output_path, metadata)
    return metadata


def _write_sidecar(output_path: str, metadata: Dict) -> None:
    json_path = os.path.splitext(output_path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')


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
    state = tuple(value.clone() for value in inputs[SIGNAL_INPUTS:])
    worst = 0.0
    generator = torch.Generator().manual_seed(20260816)
    with torch.no_grad():
        for _ in range(2 * wrapper.delay_depth + 5):
            signals = graph_signals(
                wrapper.model,
                torch.randn(1, 1, wrapper.n_freqs, 2, generator=generator),
                torch.randn(1, 1, wrapper.n_freqs, 2, generator=generator),
            )
            torch_inputs = signals + state
            expected = wrapper(*torch_inputs)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(INPUT_NAMES, torch_inputs)
            })
            for got, want in zip(actual, expected):
                worst = max(worst, float(np.max(
                    np.abs(got - want.detach().numpy())
                )))
            state = next_state(expected)
    return worst


def export_graph(model, checkpoint_path, output_path, opset=17,
                 verify=False):
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
    checkpoint_data = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    # Refused before anything is written: the metadata step below would also
    # refuse, but only after torch.onnx.export has put a graph on disk.
    _checkpoint_linear_aec(checkpoint_data['contract'])
    wrapper = AlignUlcnetStreamingExport(model).eval()
    inputs = dummy_inputs(
        wrapper.delay_depth, wrapper.n_freqs, wrapper.ta_bins)
    with torch.no_grad():
        outputs = wrapper(*inputs)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError('PyTorch streaming export reference is non-finite')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
        do_constant_folding=True,
    )
    optimize_graph_file(output_path)

    import onnx
    graph = onnx.shape_inference.infer_shapes(onnx.load(output_path))
    onnx.checker.check_model(graph)
    validate_nctf_no_temporal_padding(graph, require_static=verify)
    metadata = _write_metadata(
        output_path,
        checkpoint_path,
        model,
        checkpoint_data['contract'],
        inputs,
        outputs,
    )
    _set_onnx_metadata(graph, metadata)
    _pin_static_output_shapes(graph, OUTPUT_NAMES, outputs)
    graph = _resolve_internal_shapes(graph)
    validate_inplace_state_contract(
        graph, STATE_NAMES, STATE_OUTPUT_NAMES,
        window_drops=metadata['state_window_drop'])
    onnx.save(graph, output_path)
    # Identity of the deployed graph bytes, for the deployment gate
    # (frontend_gate.py). It cannot be one of the graph's own props (writing it
    # would change the bytes), so only the JSON sidecar carries it, written after
    # the final save.
    metadata['onnx_sha256'] = file_sha256(output_path)
    _write_sidecar(output_path, metadata)

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
    args = parser.parse_args()

    model, _grid, _linear_contract = load_model(
        args.checkpoint, 'cpu', max_delay_frames=args.max_delay_frames
    )
    model.eval()
    export_graph(model, args.checkpoint, args.output,
                 opset=args.opset, verify=args.verify)
    print(args.output)


if __name__ == '__main__':
    main()
