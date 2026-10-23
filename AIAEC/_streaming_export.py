#!/usr/bin/env python3
"""Internal stateless ONNX backend for model-local AIAEC exporters.

One new STFT frame is supplied per invocation for causal models; every temporal
convolution history, attention ring and recurrent hidden tensor is an ordinary
graph input/output. Align-ULCNet keeps its specialised exporter because its
full-state ABI is already paired with ``ulcnet_model_io.c``.

State boundary
--------------

Every state tensor is rank 4 and comes back in full with its input's shape, so
each ``*_out`` can be bound to the address of its input.

* Time-extended convolutions keep their whole input window (oldest first): the
  graph cuts the history off with a Slice and concatenates the new frame.
* Delay rings (newest first) keep slots ``0..D-2`` through a Slice and
  concatenate the new frame in front.
* A GRU hidden ``[layers, N, H]`` crosses as ``[1, layers, N, H]``. Several GRU
  hiddens (one shape) are stacked on dim 1 into one ``h_gru`` placed after the
  other states; a single hidden is already called ``h_gru``.
* Vector and scalar accumulators are lifted to ``[1, 1, N, D]`` / ``[1, 1, 1, 1]``.

``export_graph`` validates the written file with
``validate_inplace_state_contract`` and records ``state_binding`` and
``state_window_drop`` in its metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AIAEC.aiaec_streaming import (
    DelayRingCell,
    FrameDelayAttentionCell,
    GlobalDelayAttentionCell,
    StreamConv2dCell,
    StreamGRUCell,
    StreamModuleCell,
)
from AIAEC.aiaec_common import fit_frequency, three_vector_complex
from AIAEC._onnx_contract import (
    STATE_BINDING,
    validate_inplace_state_contract,
    validate_nctf_no_temporal_padding,
)
from AIAEC.CAGCRN.model import (
    _stream_cata,
    _stream_decoder_block,
    _stream_encoder_block,
    _stream_tfag,
    _stream_tfgru,
)
from AIAEC._export_common import (
    _compressed_ri,
    _mag_ri,
    alignment_depth,
    file_sha256,
    load_checkpoint_model,
    set_alignment_depth,
)


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


# The candidates this shared stateless exporter serves: everything except
# Align-ULCNet, which keeps its own exporter (see the module docstring).
GENERIC_MODEL_NAMES = (
    'Align_CRUSE',
    'DeepVQE_S',
    'CAGCRN',
)

_C_BOUNDARIES = {
    'Align_CRUSE': 'aiaec_process.c/aiaec_process.h',
    'DeepVQE_S': ('aiaec_process.c/aiaec_process.h+'
                  'DeepVQE_S/deepvqe_process.c/.h'),
    'CAGCRN': 'aiaec_process.c/aiaec_process.h',
}

_CONTROL_SEMANTICS = {
    'Align_CRUSE': 'real_mask_for_microphone_spectrum',
    'DeepVQE_S': 'complex_ccm_taps_host_applies_spectrum_ring',
    'CAGCRN': 'complex_mask_for_microphone_spectrum',
}
# The graph's `mic`/`far` inputs are the host-compressed [1,2,1,F] planar
# spectra (see host_signal_inputs). DEEPVQE_PREPOST_LAYOUT_VERSION in
# deepvqe_prepost.h carries the same number. Versions 1-3 are RETIRED.
DEEPVQE_C_LAYOUT_VERSION = 4
# The host compresses the inputs with this exponent (C:
# DEEPVQE_COMPRESSION_EXP); a checkpoint trained with another is refused.
DEEPVQE_COMPRESSION_EXPONENT = 0.3


def _deepvqe_c_descriptor(built, outputs) -> Dict[str, int]:
    """DeepVqePrepostDescriptor (DeepVQE_S/deepvqe_prepost.h), MEASURED off
    the graph just built rather than restated by hand: the head's tap axes,
    the conv and score window depths, the GRU shape, the alignment depth and
    the state count all come from the boundary tensors, and the grid from the
    model's own. A topology change therefore surfaces as a
    deepvqe_prepost_descriptor_validate() refusal on the board, never as a
    silently wrong binding."""
    wrapper, inputs, input_names, _output_names, split = built
    grid = wrapper.model.grid
    shapes = {
        name: tuple(int(size) for size in tensor.shape)
        for name, tensor in zip(input_names, inputs)
    }
    head = tuple(int(size) for size in outputs[0].shape)
    # The CCM taps cross the accelerator boundary as one rank-4 tensor.  Its
    # packed channel keeps the model's row-major [time][frequency][RI] order,
    # so the C postprocessor can consume it without a transpose or copy.
    time_order = int(wrapper.model.time_order)
    freq_taps = 2 * int(wrapper.model.freq_radius) + 1
    packed_width = time_order * freq_taps * 2
    if head != (1, 1, grid.n_freqs, packed_width):
        raise ValueError(
            'DeepVQE_S head is not packed [1,1,F,T*K*2]=[1,1,%d,%d]: %r'
            % (grid.n_freqs, packed_width, head))
    h_gru = shapes['h_gru']                       # [1, layers, 1, hidden]
    conv = shapes['state_mic1_history']           # [1, C, window, F]
    key_ring = shapes['state_align_key_ring']     # [1, sim, D, F2]
    score = shapes['state_align_score_history']   # [1, sim, window, D]
    # D is the one deployment parameter: both rings must agree on it, and
    # with the depth the export was asked for.
    depth = key_ring[2]
    if score[3] != depth or alignment_depth(wrapper.model) != depth:
        raise ValueError(
            'DeepVQE_S alignment depth is inconsistent: key ring %d, score '
            'history %d, model %d'
            % (depth, score[3], alignment_depth(wrapper.model)))
    return {
        'layout_version': DEEPVQE_C_LAYOUT_VERSION,
        'delay_depth': depth,
        'sample_rate': int(grid.sample_rate),
        'fft_size': int(grid.n_fft),
        'hop_size': int(grid.hop_len),
        'spectrum_bins': int(grid.n_freqs),
        'time_order': time_order,
        'freq_taps': freq_taps,
        'conv_window_frames': conv[2],
        'score_window_frames': score[2],
        'gru_layers': h_gru[1],
        'gru_hidden': h_gru[3],
        'state_tensor_count': len(input_names) - split.signal_inputs,
    }


# Models with a C pre/post class, mapped to the function that measures that
# class's descriptor. The layout version inside it is the class header's
# *_PREPOST_LAYOUT_VERSION (the header is the source of truth; the class test
# pins the two). A model absent here has no C class yet -- only the shared
# aiaec_process.c helpers -- and its artifact carries neither
# `state_layout_version` nor `c_descriptor`: ABSENT means "nothing on the C
# side pins this boundary", and a present value is always real JSON (a None
# would reach the ONNX props as the string "None").
_C_DESCRIPTORS = {
    'DeepVQE_S': _deepvqe_c_descriptor,
}


def c_descriptor(model_name: str, built, outputs) -> Optional[Dict[str, int]]:
    """The compiled-ABI descriptor the model's C class validates, or None
    for a model with no C class."""
    measure = _C_DESCRIPTORS.get(model_name)
    return None if measure is None else measure(built, outputs)


def state_precision_policy(model_name: str) -> Dict[str, str]:
    """State tensors that must stay outside integer PTQ.

    Align-CRUSE's running score is an undecayed accumulator, so no finite
    calibration recording can establish a deployment-horizon-independent
    int8 range.  The absolute frame counter is an integer control value rather
    than a quantized activation.  Other state tensors may follow the target
    accelerator's normal calibration policy.
    """
    if model_name != 'Align_CRUSE':
        return {}
    return {
        'state_align_score_sum': 'float32_no_ptq',
        'state_align_frame_index': 'int64_no_ptq',
    }


def requires_contiguous_calibration(model_name: str) -> bool:
    """Whether calibration must come from ONE uninterrupted source stream.

    The cumulative Align-CRUSE state that ``state_precision_policy`` keeps out
    of integer PTQ also makes stitched calibration meaningless: joining short
    files resets ``score_sum`` at every boundary. Skipping short sources, the
    failure message and the recorded frame count all read this one predicate.
    """
    return model_name == 'Align_CRUSE'


_WINDOW_DROP = {'window': 'first', 'ring': 'last'}


class _StateSlot:
    """One cell tensor and how it crosses the graph boundary.

    ``kind`` decides the boundary form: windows and rings are already rank 4;
    a GRU hidden ``[layers, N, H]`` is lifted to ``[1, layers, N, H]``; a
    vector accumulator ``[N, D]`` to ``[1, 1, N, D]``; a scalar counter to
    ``[1, 1, 1, 1]``. ``lift``/``lower`` convert between the cell's own tensor
    and the boundary tensor.
    """

    WINDOW = 'window'
    RING = 'ring'
    GRU = 'gru'
    VECTOR = 'vector'
    SCALAR = 'scalar'

    def __init__(self, path: str, owner, attribute: str, kind: str):
        self.path = path
        self.owner = owner
        self.attribute = attribute
        self.kind = kind

    @property
    def is_window(self) -> bool:
        """Whether the graph shifts it with a Slice on the input."""
        return self.kind in (self.WINDOW, self.RING)

    @property
    def window_drop(self) -> Optional[str]:
        """The end a window drops per frame: a time-ordered conv window
        keeps its oldest frame first and drops ``'first'``; a newest-first
        delay ring drops ``'last'``. None for a state that is no window."""
        return _WINDOW_DROP.get(self.kind)

    def get(self) -> Tensor:
        value = getattr(self.owner, self.attribute)
        if not isinstance(value, Tensor):
            raise RuntimeError('stream state %s is not initialised' % self.path)
        return value

    def set(self, value: Tensor) -> None:
        setattr(self.owner, self.attribute, value)

    def lift(self, value: Tensor) -> Tensor:
        if self.kind == self.GRU:
            return value.unsqueeze(0)
        if self.kind == self.VECTOR:
            return value.reshape(1, 1, *value.shape)
        if self.kind == self.SCALAR:
            return value.reshape(1, 1, 1, 1)
        return value

    def lower(self, value: Tensor) -> Tensor:
        if self.kind == self.GRU:
            return value.squeeze(0)
        if self.kind == self.VECTOR:
            return value.reshape(value.shape[2:])
        if self.kind == self.SCALAR:
            return value.reshape(())
        return value


def _state_slots(state: Dict[str, object]) -> List[_StateSlot]:
    slots: List[_StateSlot] = []

    def visit(path: str, cell) -> None:
        if isinstance(cell, StreamConv2dCell):
            slots.append(_StateSlot(path + '.history', cell, '_window',
                                    _StateSlot.WINDOW))
        elif isinstance(cell, StreamGRUCell):
            slots.append(_StateSlot(path + '.hidden', cell, '_hidden',
                                    _StateSlot.GRU))
        elif isinstance(cell, DelayRingCell):
            slots.append(_StateSlot(path + '.ring', cell, '_ring',
                                    _StateSlot.RING))
        elif isinstance(cell, StreamModuleCell):
            visit(path, cell.conv_cell)
        elif isinstance(cell, FrameDelayAttentionCell):
            visit(path + '.key', cell.key_ring)
            visit(path + '.value', cell.value_ring)
            visit(path + '.score', cell.score_cell)
        elif isinstance(cell, GlobalDelayAttentionCell):
            visit(path + '.key', cell.key_ring)
            visit(path + '.value', cell.value_ring)
            slots.append(_StateSlot(path + '.score_sum', cell, '_score_sum',
                                    _StateSlot.VECTOR))
            slots.append(_StateSlot(path + '.frame_index', cell,
                                    'frame_index', _StateSlot.SCALAR))
        elif hasattr(cell, 'conv_cell'):
            # DeepVQE frequency-upsample and residual streaming cells.
            visit(path, cell.conv_cell)
        elif isinstance(cell, Tensor):
            # Inference constants such as CAGCRN's learned-window gate are
            # derived from weights and are not caller-owned temporal state.
            return
        else:
            raise TypeError('unsupported stream state cell at %s: %s' %
                            (path, type(cell).__name__))

    for name in sorted(state):
        visit(name, state[name])
    paths = [slot.path for slot in slots]
    if len(paths) != len(set(paths)):
        raise RuntimeError('duplicate streaming state names')
    return slots


def _onnx_name(path: str) -> str:
    return 'state_' + _sanitize(path)


def _sanitize(path: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', path).strip('_')


def _hidden_name(path: str) -> str:
    """GRU hiddens name themselves ``h_<module>``; ``h_`` already says
    hidden, so the trailing ``.hidden`` path segment is dropped."""
    if path.endswith('.hidden'):
        path = path[:-len('.hidden')]
    return 'h_' + _sanitize(path)


def _fold_three_vector_conv(conv: nn.Conv2d, pairs: int) -> nn.Conv2d:
    """Fold DeepVQE's three-vector -> complex map into the last conv.

    ``conv`` is the bias-free conv under ``ccm_up`` whose output channel ``(p *
    3 + v) * 2 + s`` carries tap ``p``, vector component ``v`` and sub-pixel
    phase ``s``. Everything after it (shuffle, frequency fit, ``re = w0 - w1/2
    - w2/2``, ``im = sqrt(3)/2 * (w1 - w2)``) is linear and per position, so
    the map is applied to the weights once; the model's own conv is untouched.
    """
    weight = conv.weight.detach()
    if conv.bias is not None or weight.shape[0] != pairs * 3 * 2:
        raise ValueError('ccm_up conv does not have the three-vector layout')
    tail = weight.shape[1:]
    weight = weight.reshape(pairs, 3, 2, *tail)
    # The map itself is three_vector_complex's: row r of `fold` is its
    # (re, im) response to each unit vector.
    unit = three_vector_complex(torch.eye(3, dtype=weight.dtype))
    fold = torch.stack((unit.real, unit.imag))
    folded = torch.einsum('rv,pvs...->spr...', fold, weight)
    out = copy.deepcopy(conv)
    out.out_channels = 2 * pairs * 2
    out.weight = nn.Parameter(
        folded.reshape(out.out_channels, *tail).contiguous(),
        requires_grad=False,
    )
    return out


def _fold_batch_norm_into_upsample_conv(conv: nn.Conv2d,
                                        norm: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold an eval-mode BatchNorm that follows a sub-pixel shuffle into the
    bias-free conv before it.

    The conv emits ``2 * C`` channels read as ``(c, phase)``; the norm is a
    per-``c`` affine map that commutes with the shuffle and a frequency crop,
    so both phases of channel ``c`` get the same weight scale and the same
    shift as their new bias. The model's own conv is untouched.
    """
    if conv.bias is not None or norm.training or not norm.track_running_stats:
        raise ValueError('upsample norm cannot be folded: need a bias-free '
                         'conv and an eval-mode BatchNorm with running stats')
    channels = norm.num_features
    if conv.out_channels != 2 * channels:
        raise ValueError('upsample conv is not the 2 x C sub-pixel layout')
    gain = norm.weight.detach() / (norm.running_var + norm.eps).sqrt()
    shift = norm.bias.detach() - norm.running_mean * gain
    gain = gain.repeat_interleave(2)
    shift = shift.repeat_interleave(2)
    out = copy.deepcopy(conv)
    out.weight = nn.Parameter(
        (conv.weight.detach() * gain.reshape(-1, 1, 1, 1)).contiguous(),
        requires_grad=False)
    out.bias = nn.Parameter(shift.contiguous(), requires_grad=False)
    return out


class StatelessOneFrameAIAEC(nn.Module):
    """Bind explicit tensors to an existing model's streaming reference."""

    #: Name of the stacked recurrent tensor of a model with several hiddens.
    COMBINED_GRU_STATE_NAME = 'h_gru'

    def __init__(self, model_name: str, model: nn.Module):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.stream_state = model.create_stream_state()
        frequency = model.grid.n_freqs
        dummy = torch.complex(
            torch.randn(1, 1, frequency), torch.randn(1, 1, frequency)
        )
        with torch.no_grad():
            model.forward_stream(dummy, dummy, self.stream_state)
        if model_name == 'DeepVQE_S':
            # The deployment graph emits CCM taps; the host applies them and
            # owns the raw-spectrum ring. It is not model-internal state.
            self.stream_state.pop('spec_ring')
            self._fold_ccm_head()
            self._fold_upsample_norms()
        self.slots = _state_slots(self.stream_state)
        for slot in self.slots:
            slot.get()
        self._gru_slots = [
            index for index, slot in enumerate(self.slots)
            if slot.kind == _StateSlot.GRU
        ]
        gru_slots = set(self._gru_slots)
        self._other_slots = [
            index for index in range(len(self.slots))
            if index not in gru_slots
        ]
        # Several hiddens are stacked on dim 1 into one tensor. One hidden
        # already is one tensor (Align_CRUSE and DeepVQE_S: h_gru).
        if self.combines_gru_state:
            widths = {tuple(self.slots[i].lift(self.slots[i].get()).shape)
                      for i in self._gru_slots}
            if len(widths) != 1:
                raise ValueError(
                    '%s: the combined recurrent state requires one shape for '
                    'every GRU hidden; got %s'
                    % (model_name, sorted(widths))
                )
        # Sanitizing every slot path is regex work, and the boundary is fixed
        # once the slots are; the per-frame forward() must not redo it.
        slot_names = self._split_state_names()
        self._state_names = self._regroup(
            slot_names, self.COMBINED_GRU_STATE_NAME)
        self._window_state_drops = {
            name: slot.window_drop
            for name, slot in zip(slot_names, self.slots) if slot.is_window}

    def _fold_ccm_head(self) -> None:
        """Make the DeepVQE graph end in one conv plus a rank-4 reshape.

        Every size the head uses is read here, as a Python int, so the traced
        graph carries no shape arithmetic.
        """
        model = self.model
        block = model.ccm_up
        if not (isinstance(block.norm, nn.Identity)
                and isinstance(block.act, nn.Identity)):
            raise ValueError('ccm_up must be a bare conv plus sub-pixel '
                             'shuffle to fold the three-vector map into it')
        cell = self.stream_state['ccm_up'].conv_cell
        if cell._window.shape[0] != 1:
            raise ValueError('the one-frame graph is exported at batch 1')
        pairs = model.time_order * (2 * model.freq_radius + 1)
        cell.conv = _fold_three_vector_conv(cell.conv, pairs)
        # The window holds the conv's input frames: its width is the width
        # of the pre-shuffle frequency axis.
        self._ccm_pre_shuffle_width = int(cell._window.shape[-1])

    _UPSAMPLE_NAMES = ('up3', 'up2', 'up1')

    def _fold_upsample_norms(self) -> None:
        """Move each decoder upsample's BatchNorm into its conv, so the graph
        carries no BatchNorm after a Slice (see _up_step)."""
        for name in self._UPSAMPLE_NAMES:
            cell = self.stream_state[name]
            cell.conv_cell.conv = _fold_batch_norm_into_upsample_conv(
                cell.conv_cell.conv, cell.block.norm)

    def _up_step(self, name: str, value: Tensor, target: int) -> Tensor:
        """``_StreamFreqUpsampleCell.step`` with the norm already folded into
        the conv: shuffle, frequency crop, activation. The crop must not pad,
        because the norm's shift is only exact on real columns."""
        cell = self.stream_state[name]
        block = cell.block
        value = cell.conv_cell.step(value)
        batch, _, time, frequency = value.shape
        value = value.reshape(batch, block.out_channels, 2, time, frequency)
        value = value.permute(0, 1, 3, 4, 2).reshape(
            batch, block.out_channels, time, 2 * frequency)
        if value.shape[-1] < target:
            raise ValueError('%s would zero-pad %d -> %d bins: the folded '
                             'norm shift is not exact there'
                             % (name, value.shape[-1], target))
        return block.act(value[..., :target])

    @property
    def combines_gru_state(self) -> bool:
        return len(self._gru_slots) > 1

    def _split_state_names(self) -> Tuple[str, ...]:
        return tuple(
            _hidden_name(slot.path)
            if slot.kind == _StateSlot.GRU
            else _onnx_name(slot.path)
            for slot in self.slots
        )

    def _regroup(self, per_slot, combined_tail):
        """Slot-ordered values -> the boundary order.

        Caches keep their original order; with several hiddens the one
        stacked recurrent entry goes last, so the boundary stays readable and
        the caches' positions are unchanged relative to each other. The rule
        lives here alone -- names, initial state and per-frame state must all
        use it, or a graph would be written in one order and fed in another.
        """
        if not self.combines_gru_state:
            return tuple(per_slot)
        return tuple(per_slot[i] for i in self._other_slots) + (
            combined_tail,
        )

    @property
    def state_names(self) -> Tuple[str, ...]:
        return self._state_names

    @property
    def state_output_names(self) -> Tuple[str, ...]:
        return tuple(name + '_out' for name in self._state_names)

    @property
    def window_state_drops(self) -> Dict[str, str]:
        """State inputs the graph shifts with a Slice (windows and rings),
        each with the end it drops: ``'first'`` or ``'last'``."""
        return dict(self._window_state_drops)

    @property
    def window_state_names(self) -> Tuple[str, ...]:
        return tuple(self._window_state_drops)

    def initial_state(self) -> Tuple[Tensor, ...]:
        return self._collapse_state(
            [torch.zeros_like(slot.get()) for slot in self.slots])

    def _expand_state(self, state_tensors):
        """Boundary tensors -> one cell tensor per slot, in slot order."""
        if not self.combines_gru_state:
            lifted = state_tensors
        else:
            *others, combined = state_tensors
            lifted = [None] * len(self.slots)
            for position, index in enumerate(self._other_slots):
                lifted[index] = others[position]
            step = combined.shape[1] // len(self._gru_slots)
            for position, index in enumerate(self._gru_slots):
                lifted[index] = combined[:, position * step:(position + 1) * step]
        return tuple(slot.lower(value)
                     for slot, value in zip(self.slots, lifted))

    def _collapse_state(self, per_slot):
        """One cell tensor per slot -> boundary tensors."""
        lifted = [slot.lift(value) for slot, value in zip(self.slots, per_slot)]
        if not self.combines_gru_state:
            return tuple(lifted)
        return self._regroup(lifted, torch.cat(
            [lifted[i] for i in self._gru_slots], dim=1))

    def forward(self, primary_ri: Tensor, far_end_ri: Tensor,
                *state_tensors: Tensor):
        # DeepVQE_S takes the two signal inputs already compressed
        # (host_signal_inputs); the other models take raw RI.
        if len(state_tensors) != len(self._state_names):
            raise ValueError('wrong number of explicit stream-state tensors')
        for slot, value in zip(self.slots, self._expand_state(state_tensors)):
            slot.set(value)
        if self.model_name == 'Align_CRUSE':
            learned = self._align_cruse(primary_ri, far_end_ri)
        elif self.model_name == 'DeepVQE_S':
            learned = self._deepvqe(primary_ri, far_end_ri)
        elif self.model_name == 'CAGCRN':
            learned = self._cagcrn(primary_ri, far_end_ri)
        else:
            raise RuntimeError('unsupported one-frame model')
        return (learned,) + self._collapse_state(
            [slot.get() for slot in self.slots]
        )

    @staticmethod
    def _log_power(value: Tensor) -> Tensor:
        power = value[..., 0].square() + value[..., 1].square()
        return power.clamp_min(1e-12).log().unsqueeze(1)

    def _align_cruse(self, primary: Tensor, far: Tensor) -> Tensor:
        model = self.model
        state = self.stream_state
        m1 = state['mic1'].step(self._log_power(primary))
        m2 = state['mic2'].step(m1)
        f1 = state['far1'].step(self._log_power(far))
        f2 = state['far2'].step(f1)
        aligned, _delay = state['align'].step(m2, f2)
        m3 = state['mic3'].step(torch.cat((m2, aligned), dim=1))
        m4 = state['mic4'].step(m3)
        batch, channels, _time, frequency = m4.shape
        sequence = m4.permute(0, 2, 1, 3).reshape(
            batch, 1, channels * frequency
        )
        sequence = state['gru'].step(sequence)
        value = model.gru_out(sequence).reshape(
            batch, 1, channels, frequency
        ).permute(0, 2, 1, 3)
        value = model.up3(value + model.skip4(m4), m3.shape[-1])
        value = model.up2(value + model.skip3(m3), m2.shape[-1])
        value = model.up1(value + model.skip2(m2), m1.shape[-1])
        logits = model.mask_up(
            value + model.skip1(m1), model.grid.n_freqs
        ).squeeze(1)
        return (torch.sigmoid(fit_frequency(logits, model.grid.n_freqs))
                * model.mask_gain.clamp_min(0.0))

    def _deepvqe(self, primary: Tensor, far: Tensor) -> Tensor:
        model = self.model
        state = self.stream_state
        m1 = state['mic1'].step(primary)
        m2 = state['mic2'].step(m1)
        f1 = state['far1'].step(far)
        f2 = state['far2'].step(f1)
        aligned, _delay = state['align'].step(m2, f2)
        m3 = state['mic3'].step(torch.cat((m2, aligned), dim=1))
        m4 = state['mic4'].step(m3)
        batch, channels, time, frequency = m4.shape
        value = m4.permute(0, 2, 1, 3).reshape(
            batch, time, channels * frequency
        )
        value = state['gru'].step(value)
        value = model.gru_out(value).reshape(
            batch, time, channels, frequency
        ).permute(0, 2, 1, 3)
        value = self._up_step('up3', value + model.skip4(m4), m3.shape[-1])
        value = self._up_step(
            'up2', state['res3'].step(value + model.skip3(m3)), m2.shape[-1]
        )
        value = self._up_step(
            'up1', state['res2'].step(value + model.skip2(m2)), m1.shape[-1]
        )
        packed = state['ccm_up'].conv_cell.step(value + model.skip1(m1))
        # [1,36,1,W] -> [1,W,1,36] -> [1,1,2W,18]: phase-major channels make
        # the flatten of (W, phase) the shuffled bin order, so the layout is
        # exactly the CCM memory order deepvqe_ccm_process consumes.
        taps = packed.permute(0, 3, 2, 1).reshape(
            1, 1, 2 * self._ccm_pre_shuffle_width, -1
        )
        extra = 2 * self._ccm_pre_shuffle_width - model.grid.n_freqs
        if extra > 0:
            return taps[:, :, :model.grid.n_freqs]
        return torch.nn.functional.pad(taps, (0, 0, 0, -extra))

    def _cagcrn(self, primary: Tensor, far: Tensor) -> Tensor:
        model = self.model
        state = self.stream_state
        mic = model.erb.merge(_mag_ri(primary))
        ref = model.erb.merge(_mag_ri(far))
        mic_skips = []
        far_skips = []
        mic = _stream_encoder_block(
            model.mic_encoder[0], state['mic_enc0_group'], mic
        )
        ref = _stream_encoder_block(
            model.far_encoder[0], state['far_enc0_group'], ref
        )
        mic_skips.append(mic)
        ref, _delay, _attention = _stream_cata(
            model.cata, state['cata_query'], state['cata_key'],
            state['cata_ref_kv'], state['cata_ring'],
            state['cata_gate_log'], mic, ref,
        )
        far_skips.append(ref)
        for index in range(1, 4):
            mic = _stream_encoder_block(
                model.mic_encoder[index],
                state['mic_enc%d_group' % index], mic,
            )
            ref = _stream_encoder_block(
                model.far_encoder[index],
                state['far_enc%d_group' % index], ref,
            )
            mic_skips.append(mic)
            far_skips.append(ref)
        mic = _stream_tfgru(
            model.mic_tfgru, state['mic_tfgru_time'], mic
        )
        ref = _stream_tfgru(
            model.far_tfgru, state['far_tfgru_time'], ref
        )
        value = _stream_tfag(
            model.tfag, state['tfag_time1'], state['tfag_time2'], ref, mic
        )
        for index, block in enumerate(model.decoder):
            value = (
                value
                + fit_frequency(
                    model.skip_mic[index](mic_skips[-1 - index]),
                    value.shape[-1],
                )
                + fit_frequency(
                    model.skip_far[index](far_skips[-1 - index]),
                    value.shape[-1],
                )
            )
            target = (
                mic_skips[-2 - index].shape[-1]
                if index < 3 else model.erb.compressed_bins
            )
            value = _stream_decoder_block(
                block, state['dec%d_group' % index], value, target
            )
        mask_erb = model.mask(value)
        mask = fit_frequency(
            model.erb.split(mask_erb), model.grid.n_freqs
        )
        return mask.permute(0, 2, 3, 1)

class GraphSplit(NamedTuple):
    """Where the signal tensors end and the explicit state begins.

    ``_build`` already knows both counts when it names the tensors, so every
    consumer takes them from here. Re-deriving them downstream would let a
    future input-arity change leave the
    graph exporting correctly while the calibration recorder silently fed
    signals into state slots.
    """

    signal_inputs: int
    head_outputs: int


def host_signal_inputs(model_name: str, model, signals):
    """The graph's signal inputs for one frame, from RAW RI spectra
    ``[1,1,F,2]``.

    DeepVQE_S's power-law compression runs on the host (C:
    deepvqe_prepost_frame_inputs), so its graph starts at the first
    convolution and binds ``[1,2,1,F]`` planar tensors (real plane, then
    imaginary plane). The other models bind the raw spectra unchanged.
    """
    if model_name == 'DeepVQE_S':
        return tuple(_compressed_ri(signal, model.compression_exponent)
                     for signal in signals)
    return tuple(signals)


def _build(model_name: str, model):
    if (model_name == 'DeepVQE_S'
            and model.compression_exponent != DEEPVQE_COMPRESSION_EXPONENT):
        raise ValueError(
            'DeepVQE_S host front end is compiled for compression exponent '
            '%r, checkpoint has %r'
            % (DEEPVQE_COMPRESSION_EXPONENT, model.compression_exponent))
    wrapper = StatelessOneFrameAIAEC(model_name, model).eval()
    shape = ((1, 2, 1, model.grid.n_freqs) if model_name == 'DeepVQE_S'
             else (1, 1, model.grid.n_freqs, 2))
    inputs = (torch.randn(shape), torch.randn(shape)) + wrapper.initial_state()
    input_names = ('mic', 'far') + wrapper.state_names
    output_names = ('output',) + tuple(
        name + '_out' for name in wrapper.state_names
    )
    split = GraphSplit(signal_inputs=2, head_outputs=1)
    return wrapper, inputs, input_names, output_names, split


def _verify_onnx(path, wrapper, inputs, input_names, split: GraphSplit,
                 steps: int) -> float:
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    state = tuple(value.clone() for value in inputs[split.signal_inputs:])
    generator = torch.Generator().manual_seed(20260817)
    worst = 0.0
    with torch.no_grad():
        for _ in range(steps):
            signal = tuple(torch.randn(
                value.shape, generator=generator
            ) for value in inputs[:split.signal_inputs])
            current = signal + state
            expected = wrapper(*current)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(input_names, current)
            })
            for got, want in zip(actual, expected):
                worst = max(worst, float(np.max(
                    np.abs(got - want.detach().numpy())
                )))
            state = tuple(expected[split.head_outputs:])
    return worst


def export_graph(grid, built, checkpoint_path, output_path, checkpoint_depth,
                 opset=17, verify=False):
    """Write the streaming graph plus its metadata JSON; optionally verify.

    Shared by the export CLI and the calib recorder, so the calibration tensors
    and the graph they bind to come from the same wrapper in the same process.
    ``checkpoint_depth`` is the alignment depth the checkpoint recorded,
    captured by the caller BEFORE any deployment override. Returns the metadata
    written into the graph.
    """
    wrapper, inputs, input_names, output_names, split = built
    model_name = wrapper.model_name
    model = wrapper.model
    with torch.no_grad():
        outputs = wrapper(*inputs)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError('streaming export reference contains NaN or Inf')

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
    metadata = {
        'model_family': model_name,
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'boundary': 'stateless_streaming_explicit_state',
        'sample_rate': int(grid.sr),
        'n_fft': int(grid.n_fft),
        'win_len': int(grid.win_len),
        'hop_len': int(grid.hop_len),
        'input_feature_frames': 1,
        'output_frames_per_invocation': 1,
        'temporal_padding_inside_graph': False,
        'temporal_context': 'explicit_state_inputs_outputs',
        'accelerator_persistent_state': False,
        'c_prepost': _C_BOUNDARIES[model_name],
        'learned_control_semantics': _CONTROL_SEMANTICS[model_name],
        'gru_state_layout': 'combined',
        'state_binding': STATE_BINDING,
        'input_schema': _schema(input_names, inputs),
        'output_schema': _schema(output_names, outputs),
        'max_delay_frames': alignment_depth(model),
        'checkpoint_max_delay_frames': checkpoint_depth,
        'production_streaming_equivalent': True,
    }
    descriptor = c_descriptor(model_name, built, outputs)
    if descriptor is not None:
        metadata['state_layout_version'] = descriptor['layout_version']
        metadata['c_descriptor'] = descriptor
    metadata['state_handoff'] = {
        input_name: output_name
        for input_name, output_name in zip(
            input_names[split.signal_inputs:],
            output_names[split.head_outputs:],
        )
    }
    metadata['state_window_drop'] = wrapper.window_state_drops
    metadata['state_dtypes'] = {
        name: str(value.detach().cpu().numpy().dtype)
        for name, value in zip(
            input_names[split.signal_inputs:], inputs[split.signal_inputs:]
        )
    }
    metadata['state_precision_policy'] = state_precision_policy(model_name)
    set_onnx_metadata(graph, metadata)
    _pin_static_output_shapes(graph, output_names, outputs)
    graph = _resolve_internal_shapes(graph)
    validate_inplace_state_contract(
        graph, input_names[split.signal_inputs:],
        output_names[split.head_outputs:],
        window_drops=metadata['state_window_drop'])
    onnx.save(graph, output_path)
    with open(os.path.splitext(output_path)[0] + '.json', 'w',
              encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')

    if verify:
        worst = _verify_onnx(
            output_path, wrapper, inputs, input_names, split,
            max(4, 2 * alignment_depth(model) + 1),
        )
        if worst > 3e-4:
            raise RuntimeError(
                'streaming ONNX parity failed: max_abs=%.6g' % worst
            )
        print('streaming ONNX parity max_abs=%.6g' % worst)
    return metadata


def main(model_name: str) -> None:
    """Run one model's export CLI; each model's export_onnx.py names itself."""
    if model_name not in GENERIC_MODEL_NAMES:
        raise ValueError('unsupported AIAEC model: %s' % model_name)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-delay-frames', type=int, default=None)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    args.model_name = model_name

    model, grid = load_checkpoint_model(args.model_name, args.checkpoint)
    checkpoint_depth = alignment_depth(model)
    if args.max_delay_frames is not None:
        set_alignment_depth(model, args.max_delay_frames)
    model.eval()
    built = _build(args.model_name, model)
    export_graph(grid, built, args.checkpoint, args.output, checkpoint_depth,
                 opset=args.opset, verify=args.verify)
    print(args.output)
