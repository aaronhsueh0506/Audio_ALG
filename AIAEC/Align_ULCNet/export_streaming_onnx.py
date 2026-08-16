#!/usr/bin/env python3
"""Export Align-ULCNet as a stateless one-frame ONNX graph.

The accelerator retains no state.  The CPU supplies past K/V features,
attention-score history and temporal-GRU hidden tensors on every invocation.
The graph returns only the new K/V/logit entries plus next GRU hidden tensors;
the CPU updates its caller-owned rings.  STFT/WOLA and PBFDKF stay outside the
graph in ``ulcnet_process.c`` and the AEC library respectively.
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

from AIAEC.Align_ULCNet.denoise import load_model
from AIAEC.training_common import (
    checkpoint_far_input_mode,
    far_input_mode_c_value,
)


# Kept numerically equal to ULCNET_MODEL_IO_LAYOUT_VERSION in
# AIAEC/Align_ULCNet/ulcnet_model_io.h; test_export_streaming_ulcnet.py pins
# the two together. Version 2 added far_input_mode to the C descriptor and
# far_input_mode_c_value to the metadata below.
STATE_LAYOUT_VERSION = 2
MIN_DELAY_DEPTH = 2
MAX_DELAY_DEPTH = 64
TA_CHANNELS = 32
TA_BINS = 26
SCORE_HISTORY_FRAMES = 4
GRU_LAYERS = 2
GRU_HIDDEN = 128

INPUT_NAMES = (
    'linear_error_ri',
    'far_end_ri',
    'key_history',
    'value_history',
    'logit_history',
    'gru0_hidden',
    'gru1_hidden',
)

OUTPUT_NAMES = (
    'enhanced_ri',
    'key_now',
    'value_now',
    'logit_now',
    'gru0_hidden_next',
    'gru1_hidden_next',
)


def _signed_power(value: Tensor, exponent: float) -> Tensor:
    return value.sign() * value.abs().pow(exponent)


class AlignUlcnetStreamingExport(nn.Module):
    """Functional, explicit-state twin of ``AlignULCNet.forward_stream``.

    K/V history is newest-first: slot zero is t-1.  Logit history is
    chronological: slot zero is t-4 and slot three is t-1, matching the
    causal score convolution's input order.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.delay_depth = int(model.max_delay_frames)
        if not MIN_DELAY_DEPTH <= self.delay_depth <= MAX_DELAY_DEPTH:
            raise ValueError(
                'streaming export delay depth must be in [%d, %d]' %
                (MIN_DELAY_DEPTH, MAX_DELAY_DEPTH)
            )
        if model.grid.sample_rate != 16000 or model.grid.n_fft != 512:
            raise ValueError(
                'streaming C boundary currently requires 16 kHz / FFT 512'
            )
        if model.grid.win_len != 512 or model.grid.hop_len != 256:
            raise ValueError(
                'streaming C boundary currently requires window/hop 512/256'
            )
        if model.reorient.width != 52:
            raise ValueError('unexpected C-SamFR width')

    def forward(
        self,
        linear_error_ri: Tensor,
        far_end_ri: Tensor,
        key_history: Tensor,
        value_history: Tensor,
        logit_history: Tensor,
        gru0_hidden: Tensor,
        gru1_hidden: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        model = self.model
        exponent = model.compression_exponent

        error_real = _signed_power(linear_error_ri[..., 0], exponent)
        error_imag = _signed_power(linear_error_ri[..., 1], exponent)
        far_real = _signed_power(far_end_ri[..., 0], exponent)
        far_imag = _signed_power(far_end_ri[..., 1], exponent)
        error_magnitude = (
            error_real.square() + error_imag.square() + 1e-12
        ).sqrt()
        far_magnitude = (
            far_real.square() + far_imag.square() + 1e-12
        ).sqrt()
        error_phase = torch.atan2(error_imag, error_real)

        error_feature = model.error_encoder(
            model.reorient(error_magnitude.unsqueeze(1))
        )
        far_feature = model.far_encoder(
            model.reorient(far_magnitude.unsqueeze(1))
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
        hidden_inputs = (gru0_hidden, gru1_hidden)
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
            hidden_next.append(hidden)
            at += width

        joined = torch.cat(pieces, dim=-1)
        magnitude_mask = torch.sigmoid(model.mask_fc2(
            model.mask_act(model.mask_fc1(joined))
        ))
        intermediate = torch.stack((
            magnitude_mask * torch.cos(error_phase),
            magnitude_mask * torch.sin(error_phase),
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
        inverse_exponent = 1.0 / exponent
        enhanced_ri = torch.stack((
            _signed_power(estimate_real, inverse_exponent),
            _signed_power(estimate_imag, inverse_exponent),
        ), dim=-1)

        return (
            enhanced_ri,
            key_now,
            value_now,
            logit_now,
            hidden_next[0],
            hidden_next[1],
        )


def state_shapes(delay_depth: int) -> Dict[str, Tuple[int, ...]]:
    if not MIN_DELAY_DEPTH <= delay_depth <= MAX_DELAY_DEPTH:
        raise ValueError(
            'delay depth must be in [%d, %d]' %
            (MIN_DELAY_DEPTH, MAX_DELAY_DEPTH)
        )
    return {
        'key_history': (1, TA_CHANNELS, delay_depth - 1, TA_BINS),
        'value_history': (1, TA_CHANNELS, delay_depth - 1, TA_BINS),
        'logit_history': (
            1, TA_CHANNELS, SCORE_HISTORY_FRAMES, delay_depth
        ),
        'gru0_hidden': (GRU_LAYERS, 1, GRU_HIDDEN),
        'gru1_hidden': (GRU_LAYERS, 1, GRU_HIDDEN),
    }


def dummy_inputs(delay_depth: int) -> Tuple[Tensor, ...]:
    shapes = state_shapes(delay_depth)
    return (
        torch.randn(1, 1, 257, 2),
        torch.randn(1, 1, 257, 2),
        torch.zeros(shapes['key_history']),
        torch.zeros(shapes['value_history']),
        torch.zeros(shapes['logit_history']),
        torch.zeros(shapes['gru0_hidden']),
        torch.zeros(shapes['gru1_hidden']),
    )


def next_state(
    current: Sequence[Tensor], outputs: Sequence[Tensor], delay_depth: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    key_history, value_history, logit_history, _gru0, _gru1 = current
    _enhanced, key_now, value_now, logit_now, gru0_next, gru1_next = outputs
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
    return (
        key_history,
        value_history,
        logit_history,
        gru0_next,
        gru1_next,
    )


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
) -> Dict:
    far_input_mode = checkpoint_far_input_mode(contract)
    metadata = {
        'model_family': 'Align_ULCNet',
        'boundary': 'stateless_one_frame_delta_state',
        'state_layout_version': STATE_LAYOUT_VERSION,
        'checkpoint_sha256': file_sha256(checkpoint),
        'sample_rate': model.grid.sample_rate,
        'n_fft': model.grid.n_fft,
        'win_len': model.grid.win_len,
        'hop_len': model.grid.hop_len,
        'frames_per_invocation': 1,
        'max_delay_frames': model.max_delay_frames,
        'far_input_mode': far_input_mode,
        # The same choice as the C UlcnetModelIoDescriptor.far_input_mode
        # field carries, so board init can compare this metadata against the
        # compiled descriptor (and thus against the pipeline's far branch)
        # with an integer ==, without keeping its own name->value table.
        'far_input_mode_c_value': far_input_mode_c_value(far_input_mode),
        'accelerator_persistent_state': False,
        'cpu_delta_state_update': True,
        'tensor_dtype': 'float32',
        'complex_tensor_policy': 'real_imag_last_dimension',
        'input_schema': _schema(INPUT_NAMES, inputs),
        'output_schema': _schema(OUTPUT_NAMES, outputs),
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
    state = tuple(value.clone() for value in inputs[2:])
    worst = 0.0
    generator = torch.Generator().manual_seed(20260816)
    with torch.no_grad():
        for _ in range(2 * wrapper.delay_depth + 5):
            audio = (
                torch.randn(1, 1, 257, 2, generator=generator),
                torch.randn(1, 1, 257, 2, generator=generator),
            )
            torch_inputs = audio + state
            expected = wrapper(*torch_inputs)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(INPUT_NAMES, torch_inputs)
            })
            for got, want in zip(actual, expected):
                worst = max(worst, float(np.max(
                    np.abs(got - want.detach().numpy())
                )))
            state = next_state(state, expected, wrapper.delay_depth)
    return worst


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
    wrapper = AlignUlcnetStreamingExport(model).eval()
    inputs = dummy_inputs(wrapper.delay_depth)
    with torch.no_grad():
        outputs = wrapper(*inputs)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError('PyTorch streaming export reference is non-finite')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        args.output,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    import onnx
    graph = onnx.shape_inference.infer_shapes(onnx.load(args.output))
    onnx.checker.check_model(graph)
    checkpoint_data = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False
    )
    metadata = _write_metadata(
        args.output,
        args.checkpoint,
        model,
        checkpoint_data['contract'],
        inputs,
        outputs,
    )
    _set_onnx_metadata(graph, metadata)
    onnx.save(graph, args.output)

    if args.verify:
        worst = _verify_onnx(args.output, wrapper, inputs)
        if worst > 3e-4:
            raise RuntimeError(
                'streaming ONNX parity failed: max_abs=%.6g' % worst
            )
        print('streaming ONNX parity max_abs=%.6g' % worst)
    print(args.output)


if __name__ == '__main__':
    main()
