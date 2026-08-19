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
graph boundary is fixed at 16 kHz, FFT/window 512 and hop 256.
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


# Kept numerically equal to ULCNET_MODEL_IO_LAYOUT_VERSION in
# AIAEC/Align_ULCNet/ulcnet_model_io.h; test_export_streaming_ulcnet.py pins
# the two together. Version 3 fixed production wiring to aligned far while
# retaining the checkpoint's original far-input provenance separately;
# version 4 renamed the tensors (error/far inputs, output head,
# h_gru0/h_gru1 hiddens, *_out states) -- runtimes bind by name;
# version 5 moved the fixed front/back ends (signed-power compression,
# magnitudes, phase cos/sin, inverse power) out of the graph onto the host.
STATE_LAYOUT_VERSION = 5
# The deployed C front/back end hardcodes this exponent
# (ULCNET_MODEL_IO_COMPRESSION_EXP); export_graph refuses any checkpoint
# whose model carries a different value, because nothing downstream of the
# graph could detect the mismatch.
COMPRESSION_EXPONENT = 0.3
MIN_DELAY_DEPTH = 2
MAX_DELAY_DEPTH = 64
TA_CHANNELS = 32
TA_BINS = 26
SCORE_HISTORY_FRAMES = 4
GRU_LAYERS = 2
GRU_HIDDEN = 128

# Five separate feature inputs so every tensor keeps its own quantization
# scale; the fixed front end (signed-power compression, magnitude, phase
# cos/sin) runs on the HOST -- see stream_features, mirrored in C by
# ulcnet_model_io_prepare() --
# and the graph starts at the learned reorient/encoder compute.
SIGNAL_INPUT_NAMES = (
    'error_mag',
    'far_mag',
    'error_cos',
    'error_sin',
    'error_ri',
)
STATE_INPUT_NAMES = (
    'key_history',
    'value_history',
    'logit_history',
    'h_gru0',
    'h_gru1',
)
INPUT_NAMES = SIGNAL_INPUT_NAMES + STATE_INPUT_NAMES
SIGNAL_INPUTS = len(SIGNAL_INPUT_NAMES)

OUTPUT_NAMES = (
    'output',
    'key_now',
    'value_now',
    'logit_now',
    'h_gru0_out',
    'h_gru1_out',
)


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
        error_mag: Tensor,
        far_mag: Tensor,
        error_cos: Tensor,
        error_sin: Tensor,
        error_ri: Tensor,
        key_history: Tensor,
        value_history: Tensor,
        logit_history: Tensor,
        h_gru0: Tensor,
        h_gru1: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        model = self.model

        # All fixed input math (signed-power compression, magnitude, phase)
        # already ran on the host; the graph holds learned compute only.
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
        hidden_inputs = (h_gru0, h_gru1)
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
        # COMPRESSED-domain estimate: the fixed inverse signed power runs on
        # the host (host_output; C: ulcnet_model_io_commit), not in the
        # graph.
        output = torch.stack((estimate_real, estimate_imag), dim=-1)

        return (
            output,
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
        'h_gru0': (GRU_LAYERS, 1, GRU_HIDDEN),
        'h_gru1': (GRU_LAYERS, 1, GRU_HIDDEN),
    }


def stream_features(model, error_ri: Tensor, far_ri: Tensor):
    """Host-side fixed front end from RAW RI spectra ((1, 1, 257, 2) each).

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


def dummy_inputs(delay_depth: int) -> Tuple[Tensor, ...]:
    shapes = state_shapes(delay_depth)
    return (
        torch.randn(1, 1, 257).abs(),          # error_mag
        torch.randn(1, 1, 257).abs(),          # far_mag
        torch.randn(1, 1, 257).clamp(-1, 1),   # error_cos
        torch.randn(1, 1, 257).clamp(-1, 1),   # error_sin
        torch.randn(1, 1, 257, 2),             # error_ri (compressed)
        torch.zeros(shapes['key_history']),
        torch.zeros(shapes['value_history']),
        torch.zeros(shapes['logit_history']),
        torch.zeros(shapes['h_gru0']),
        torch.zeros(shapes['h_gru1']),
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
    training_far_input_mode = checkpoint_far_input_mode(contract)
    deployed_far_input_mode = 'aligned_far'
    metadata = {
        'model_family': 'Align_ULCNet',
        'boundary': 'stateless_one_frame_delta_state',
        'state_layout_version': STATE_LAYOUT_VERSION,
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
    state = tuple(value.clone() for value in inputs[SIGNAL_INPUTS:])
    worst = 0.0
    generator = torch.Generator().manual_seed(20260816)
    with torch.no_grad():
        for _ in range(2 * wrapper.delay_depth + 5):
            signals = stream_features(
                wrapper.model,
                torch.randn(1, 1, 257, 2, generator=generator),
                torch.randn(1, 1, 257, 2, generator=generator),
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
            state = next_state(state, expected, wrapper.delay_depth)
    return worst


def export_graph(model, checkpoint_path, output_path, opset=17, verify=False):
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
    wrapper = AlignUlcnetStreamingExport(model).eval()
    inputs = dummy_inputs(wrapper.delay_depth)
    with torch.no_grad():
        outputs = wrapper(*inputs)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError('PyTorch streaming export reference is non-finite')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
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
    )
    _set_onnx_metadata(graph, metadata)
    _pin_static_output_shapes(graph, OUTPUT_NAMES, outputs)
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
