#!/usr/bin/env python3
"""Internal stateless ONNX backend for model-local AIAEC exporters.

One new STFT frame is supplied per invocation for causal models. Every
temporal convolution history, attention ring and recurrent hidden tensor is
an ordinary graph input/output.

Align-ULCNet keeps its specialised exporter because its delta-state ABI is
already paired with ``ulcnet_model_io.c``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, NamedTuple, Tuple

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
from AIAEC.aiaec_common import fit_frequency
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

    The same cumulative Align-CRUSE state that ``state_precision_policy``
    keeps out of integer PTQ also makes stitched calibration meaningless:
    joining short files resets ``score_sum`` at every boundary, so the
    recorded range describes a session length no deployment will ever see.
    Every calibration decision that depends on that -- skipping short
    sources, the failure message, the recorded frame count -- reads this one
    predicate rather than re-testing the model name.
    """
    return model_name == 'Align_CRUSE'


class _StateSlot:
    def __init__(self, path: str, owner, attribute: str):
        self.path = path
        self.owner = owner
        self.attribute = attribute

    def get(self) -> Tensor:
        value = getattr(self.owner, self.attribute)
        if not isinstance(value, Tensor):
            raise RuntimeError('stream state %s is not initialised' % self.path)
        return value

    def set(self, value: Tensor) -> None:
        setattr(self.owner, self.attribute, value)


def _state_slots(state: Dict[str, object]) -> List[_StateSlot]:
    slots: List[_StateSlot] = []

    def visit(path: str, cell) -> None:
        if isinstance(cell, StreamConv2dCell):
            slots.append(_StateSlot(path + '.history', cell, '_history'))
        elif isinstance(cell, StreamGRUCell):
            slots.append(_StateSlot(path + '.hidden', cell, '_hidden'))
        elif isinstance(cell, DelayRingCell):
            slots.append(_StateSlot(path + '.ring', cell, '_ring'))
        elif isinstance(cell, StreamModuleCell):
            visit(path, cell.conv_cell)
        elif isinstance(cell, FrameDelayAttentionCell):
            visit(path + '.key', cell.key_ring)
            visit(path + '.value', cell.value_ring)
            visit(path + '.score', cell.score_cell)
        elif isinstance(cell, GlobalDelayAttentionCell):
            visit(path + '.key', cell.key_ring)
            visit(path + '.value', cell.value_ring)
            slots.append(_StateSlot(path + '.score_sum', cell, '_score_sum'))
            slots.append(_StateSlot(path + '.frame_index', cell,
                                    'frame_index'))
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


class StatelessOneFrameAIAEC(nn.Module):
    """Bind explicit tensors to an existing model's streaming reference."""

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
        self.slots = _state_slots(self.stream_state)
        for slot in self.slots:
            slot.get()

    @property
    def state_names(self) -> Tuple[str, ...]:
        return tuple(
            _hidden_name(slot.path)
            if isinstance(slot.owner, StreamGRUCell)
            else _onnx_name(slot.path)
            for slot in self.slots
        )

    def initial_state(self) -> Tuple[Tensor, ...]:
        return tuple(torch.zeros_like(slot.get()) for slot in self.slots)

    def forward(self, primary_ri: Tensor, far_end_ri: Tensor,
                *state_tensors: Tensor):
        if len(state_tensors) != len(self.slots):
            raise ValueError('wrong number of explicit stream-state tensors')
        for slot, value in zip(self.slots, state_tensors):
            slot.set(value)
        if self.model_name == 'Align_CRUSE':
            learned = self._align_cruse(primary_ri, far_end_ri)
        elif self.model_name == 'DeepVQE_S':
            learned = self._deepvqe(primary_ri, far_end_ri)
        elif self.model_name == 'CAGCRN':
            learned = self._cagcrn(primary_ri, far_end_ri)
        else:
            raise RuntimeError('unsupported one-frame model')
        return (learned,) + tuple(slot.get() for slot in self.slots)

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
        m1 = state['mic1'].step(_compressed_ri(
            primary, model.compression_exponent
        ))
        m2 = state['mic2'].step(m1)
        f1 = state['far1'].step(_compressed_ri(
            far, model.compression_exponent
        ))
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
        value = state['up3'].step(
            value + model.skip4(m4), m3.shape[-1]
        )
        value = state['up2'].step(
            state['res3'].step(value + model.skip3(m3)), m2.shape[-1]
        )
        value = state['up1'].step(
            state['res2'].step(value + model.skip2(m2)), m1.shape[-1]
        )
        raw = state['ccm_up'].step(
            value + model.skip1(m1), model.grid.n_freqs
        )
        raw = raw.permute(0, 2, 3, 1).reshape(
            batch, time, model.grid.n_freqs,
            model.time_order, 2 * model.freq_radius + 1, 3,
        )
        tap_real = raw[..., 0] - 0.5 * raw[..., 1] - 0.5 * raw[..., 2]
        tap_imag = (3.0 ** 0.5 / 2.0) * (raw[..., 1] - raw[..., 2])
        return torch.stack((tap_real, tap_imag), dim=-1)

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


def _build(model_name: str, model):
    wrapper = StatelessOneFrameAIAEC(model_name, model).eval()
    shape = (1, 1, model.grid.n_freqs, 2)
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

    Shared by the export CLI and the calib recorder, so the calibration
    tensors and the graph they bind to always come from the same wrapper in
    the same process. ``checkpoint_depth`` is the alignment depth the
    checkpoint recorded, captured by the caller BEFORE any deployment
    override -- it cannot be recovered from the model afterwards. Returns
    the metadata written into the graph.
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
        'accelerator_persistent_state': False,
        'c_prepost': _C_BOUNDARIES[model_name],
        'learned_control_semantics': _CONTROL_SEMANTICS[model_name],
        'input_schema': _schema(input_names, inputs),
        'output_schema': _schema(output_names, outputs),
        'max_delay_frames': alignment_depth(model),
        'checkpoint_max_delay_frames': checkpoint_depth,
        'production_streaming_equivalent': True,
    }
    metadata['state_handoff'] = {
        input_name: output_name
        for input_name, output_name in zip(
            input_names[split.signal_inputs:],
            output_names[split.head_outputs:],
        )
    }
    metadata['state_dtypes'] = {
        name: str(value.detach().cpu().numpy().dtype)
        for name, value in zip(
            input_names[split.signal_inputs:], inputs[split.signal_inputs:]
        )
    }
    metadata['state_precision_policy'] = state_precision_policy(model_name)
    set_onnx_metadata(graph, metadata)
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
