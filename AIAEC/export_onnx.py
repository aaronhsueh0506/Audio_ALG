#!/usr/bin/env python3
"""Export an AIAEC candidate's learned block to fixed-frame real-valued ONNX.

ONNX/NPU boundaries use real/imag pairs and never ONNX complex tensors.  The
graph emits the learned control object (mask, CCM taps, or DFN heads); fixed
STFT/WOLA and model-specific composition stay in the accompanying C code.

This is a fixed-block export.  Recurrent and attention state resets at every
invocation, so ``--frames`` is part of the deployment/checkpoint validation
contract.  The existing ``forward_stream`` implementations remain the
reference for a future explicit-state graph; this exporter does not pretend a
one-frame state-reset graph is streaming-equivalent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AIAEC.aiaec_common import fit_frequency


MODEL_NAMES = (
    'Align_ULCNet', 'Align_CRUSE', 'DeepVQE_S', 'CAGCRN',
    'GTCRN_AENR', 'DeepFilterNet_AENR',
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ri_mag(x: Tensor) -> Tensor:
    return (x[..., 0].square() + x[..., 1].square() + 1e-12).sqrt()


def _compressed_ri(x: Tensor, exponent: float) -> Tensor:
    scale = _ri_mag(x).clamp_min(1e-12).pow(exponent - 1.0)
    return torch.stack((x[..., 0] * scale, x[..., 1] * scale), dim=1)


def _mag_ri(x: Tensor) -> Tensor:
    return torch.stack((_ri_mag(x), x[..., 0], x[..., 1]), dim=1)


def _delay_stack(x: Tensor, delays: int) -> Tensor:
    """Export-friendly fixed-shape twin of causal_delay_stack.

    PyTorch's legacy ONNX lowering cannot export ``Pad -> Unfold`` even when
    every dimension is fixed.  Explicit slices preserve the exact slot order
    while producing only standard Pad/Concat/Unsqueeze nodes.
    """
    time = x.shape[2]
    slots = []
    for delay in range(delays):
        if delay == 0:
            shifted = x
        elif delay < time:
            shifted = torch.cat((x.new_zeros(x.shape[0], x.shape[1], delay,
                                             x.shape[3]),
                                 x[:, :, :time-delay]), dim=2)
        else:
            shifted = torch.zeros_like(x)
        slots.append(shifted)
    return torch.stack(slots, dim=3)


def _frame_attention(attention, mic: Tensor, far: Tensor):
    q = attention.query(mic)
    k_delayed = _delay_stack(attention.key(far), attention.max_delay_frames)
    logits = (q.unsqueeze(3) * k_delayed).sum(dim=-1)
    distribution = torch.softmax(attention.score(logits).squeeze(1), dim=-1)
    values = _delay_stack(attention.value(far), attention.max_delay_frames)
    aligned = (values * distribution[:, None, :, :, None]).sum(dim=3)
    return aligned, distribution


def _global_attention(attention, mic: Tensor, far: Tensor):
    b, _, time, _ = mic.shape
    q = attention.query(attention.mic_pool(mic).permute(0, 2, 1, 3).reshape(b, time, -1))
    k = attention.key(attention.far_pool(far).permute(0, 2, 1, 3).reshape(b, time, -1))
    delayed = _delay_stack(k.transpose(1, 2).unsqueeze(-1),
                           attention.max_delay_frames).squeeze(-1).permute(0, 2, 3, 1)
    frame = torch.arange(time, device=mic.device)[None, :, None]
    delay_index = torch.arange(attention.max_delay_frames,
                               device=mic.device)[None, None, :]
    valid = (frame >= delay_index).to(q.dtype)
    scores = (q.unsqueeze(2) * delayed).sum(-1)
    observable = torch.arange(attention.max_delay_frames,
                              device=mic.device) < time
    if attention.mode == 'paper_global':
        logits = (scores * valid).sum(dim=1)
        distribution = torch.softmax(logits.masked_fill(
            ~observable[None], float('-inf')), dim=-1)
        weights = distribution[:, None, None, :, None]
    else:
        logits = (scores * valid).cumsum(dim=1)
        distribution = torch.softmax(logits.masked_fill(
            ~observable[None, None], float('-inf')), dim=-1)
        weights = distribution[:, None, :, :, None]
    aligned = (_delay_stack(far, attention.max_delay_frames) * weights).sum(dim=3)
    return aligned, distribution


def _cata(attention, mic: Tensor, far: Tensor):
    q = attention.mic_query(mic)
    k_mic = attention.mic_key(q)
    mic_distribution = torch.softmax(q * k_mic, dim=-1)
    y_mic = attention.mic_value(mic) * mic_distribution
    delayed = _delay_stack(attention.ref_kv(far), attention.max_delay_frames)
    logits = q.unsqueeze(3) * delayed
    boundary = 1.0 + (attention.max_delay_frames - 1.0) * torch.sigmoid(attention.raw_window)
    delay = torch.arange(attention.max_delay_frames, device=mic.device,
                         dtype=mic.dtype)
    gate = torch.sigmoid((boundary - delay) / attention.window_temperature).clamp_min(1e-6)
    distribution = torch.softmax(
        logits + gate.log()[None, None, None, :, None], dim=3)
    y_ref = (distribution * delayed).sum(dim=3)
    fused = attention.fuse(torch.cat((y_mic, y_ref), dim=1))
    return fused, distribution.mean(dim=(1, 4)), distribution


class AlignCruseExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, microphone, far_end):
        m = self.m
        m0 = (microphone[..., 0].square() + microphone[..., 1].square()).clamp_min(1e-12).log().unsqueeze(1)
        f0 = (far_end[..., 0].square() + far_end[..., 1].square()).clamp_min(1e-12).log().unsqueeze(1)
        m1 = m.mic1(m0); m2 = m.mic2(m1)
        f1 = m.far1(f0); f2 = m.far2(f1)
        aligned, delay = _global_attention(m.align, m2, f2)
        m3 = m.mic3(torch.cat((m2, aligned), dim=1)); m4 = m.mic4(m3)
        b, c, t, f = m4.shape
        x, _ = m.gru(m4.permute(0, 2, 1, 3).reshape(b, t, c * f))
        x = m.gru_out(x).reshape(b, t, c, f).permute(0, 2, 1, 3)
        x = m.up3(x + m.skip4(m4), m3.shape[-1])
        x = m.up2(x + m.skip3(m3), m2.shape[-1])
        x = m.up1(x + m.skip2(m2), m1.shape[-1])
        logits = m.mask_up(x + m.skip1(m1), m.grid.n_freqs).squeeze(1)
        mask = torch.sigmoid(fit_frequency(logits, m.grid.n_freqs))
        return mask * m.mask_gain.clamp_min(0.0), delay


class AlignUlcnetExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    @staticmethod
    def signed_power(x, exponent):
        return x.sign() * x.abs().pow(exponent)

    def forward(self, linear_error, far_end):
        m = self.m; exponent = m.compression_exponent
        zr = self.signed_power(linear_error[..., 0], exponent)
        zi = self.signed_power(linear_error[..., 1], exponent)
        yr = self.signed_power(far_end[..., 0], exponent)
        yi = self.signed_power(far_end[..., 1], exponent)
        zmag = (zr.square() + zi.square() + 1e-12).sqrt()
        ymag = (yr.square() + yi.square() + 1e-12).sqrt()
        zphase = torch.atan2(zi, zr)
        e = m.error_encoder(m.reorient(zmag.unsqueeze(1)))
        f = m.far_encoder(m.reorient(ymag.unsqueeze(1)))
        aligned, delay = _frame_attention(m.align, e, f)
        x = m.joint2(m.joint1(torch.cat((e, aligned), dim=1))); x = m.fgru(x)
        pieces = []; at = 0
        for width, block in zip(m.subband_widths, m.subband_grus):
            pieces.append(block(x[..., at:at + width])); at += width
        features = torch.cat(pieces, dim=-1)
        magnitude_mask = torch.sigmoid(m.mask_fc2(m.mask_act(m.mask_fc1(features))))
        intermediate = torch.stack((magnitude_mask * torch.cos(zphase),
                                    magnitude_mask * torch.sin(zphase)), dim=1)
        stage2 = m.stage2_act(m.stage2_norm1(m.stage2_conv1(intermediate)))
        stage2 = m.stage2_act(m.stage2_norm2(m.stage2_conv2(stage2)))
        return m.complex_mask(stage2).permute(0, 2, 3, 1), delay, magnitude_mask


class DeepVqeExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, microphone, far_end):
        m = self.m
        mic_feat = _compressed_ri(microphone, m.compression_exponent)
        far_feat = _compressed_ri(far_end, m.compression_exponent)
        m1 = m.mic1(mic_feat); m2 = m.mic2(m1)
        f1 = m.far1(far_feat); f2 = m.far2(f1)
        aligned, delay = _frame_attention(m.align, m2, f2)
        m3 = m.mic3(torch.cat((m2, aligned), dim=1)); m4 = m.mic4(m3)
        b, c, t, f = m4.shape
        x, _ = m.gru(m4.permute(0, 2, 1, 3).reshape(b, t, c * f))
        x = m.gru_out(x).reshape(b, t, c, f).permute(0, 2, 1, 3)
        x = m.up3(x + m.skip4(m4), m3.shape[-1])
        x = m.up2(m.res3(x + m.skip3(m3)), m2.shape[-1])
        x = m.up1(m.res2(x + m.skip2(m2)), m1.shape[-1])
        raw = m.ccm_up(x + m.skip1(m1), m.grid.n_freqs)
        raw = raw.permute(0, 2, 3, 1).reshape(
            b, t, m.grid.n_freqs, m.time_order, 2 * m.freq_radius + 1, 3)
        tap_re = raw[..., 0] - 0.5 * raw[..., 1] - 0.5 * raw[..., 2]
        tap_im = (math.sqrt(3.0) / 2.0) * (raw[..., 1] - raw[..., 2])
        return torch.stack((tap_re, tap_im), dim=-1), delay


class CagcrnExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, microphone, far_end):
        m = self.m
        mic = m.erb.merge(_mag_ri(microphone)); far = m.erb.merge(_mag_ri(far_end))
        mic_skips = []; far_skips = []
        mic = m.mic_encoder[0](mic); far = m.far_encoder[0](far); mic_skips.append(mic)
        far, delay, full_attention = _cata(m.cata, mic, far); far_skips.append(far)
        for index in range(1, 4):
            mic = m.mic_encoder[index](mic); far = m.far_encoder[index](far)
            mic_skips.append(mic); far_skips.append(far)
        mic = m.mic_tfgru(mic); far = m.far_tfgru(far); x = m.tfag(far, mic)
        for index, block in enumerate(m.decoder):
            x = (x + fit_frequency(m.skip_mic[index](mic_skips[-1-index]), x.shape[-1])
                 + fit_frequency(m.skip_far[index](far_skips[-1-index]), x.shape[-1]))
            target = mic_skips[-2-index].shape[-1] if index < 3 else m.erb.compressed_bins
            x = block(x, target)
        mask_erb = m.mask(x)
        mask_full = fit_frequency(m.erb.split(mask_erb), m.grid.n_freqs)
        return mask_full.permute(0, 2, 3, 1), delay, full_attention


class GtcrnAenrExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, linear_error, far_end):
        m = self.m
        err = m.sfe(m.erb.bm(_mag_ri(linear_error)))
        far = m.sfe(m.erb.bm(_mag_ri(far_end)))
        feat, skips = m.encoder(torch.cat((err, far), dim=1))
        feat = m.dpgrnn2(m.dpgrnn1(feat))
        mask_erb = m.decoder(feat, skips)
        mask_full = m.erb.bs(mask_erb)
        return mask_full.permute(0, 2, 3, 1), mask_erb.permute(0, 2, 3, 1)


class DfnAenrExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model

    def forward(self, error_erb, error_spec, far_erb, far_spec):
        feat_erb, feat_spec = self.m.condition_features(
            error_erb, error_spec, far_erb, far_spec)
        return self.m.heads(feat_erb, feat_spec)


WRAPPERS = {
    'Align_CRUSE': (AlignCruseExport, ('mask', 'delay_distribution')),
    'Align_ULCNet': (AlignUlcnetExport, ('complex_mask_ri', 'delay_distribution', 'magnitude_mask')),
    'DeepVQE_S': (DeepVqeExport, ('ccm_taps_ri', 'delay_distribution')),
    'CAGCRN': (CagcrnExport, ('complex_mask_ri', 'delay_distribution', 'cata_attention')),
    'GTCRN_AENR': (GtcrnAenrExport, ('complex_mask_ri', 'erb_complex_mask_ri')),
    'DeepFilterNet_AENR': (DfnAenrExport, ('erb_mask', 'df_coefs', 'df_alpha')),
}

REQUIRED_OUTPUT_INDICES = {
    'Align_CRUSE': (0,),
    'Align_ULCNet': (0,),
    'DeepVQE_S': (0,),
    'CAGCRN': (0,),
    'GTCRN_AENR': (0,),
    'DeepFilterNet_AENR': (0, 1, 2),
}

COMPOSITION = {
    'Align_CRUSE': 'aiaec_apply_real_mask',
    'Align_ULCNet': 'aiaec_apply_ulcnet_compressed_mask',
    'DeepVQE_S': 'deepvqe_ccm_process',
    'CAGCRN': 'aiaec_apply_complex_mask',
    'GTCRN_AENR': 'aiaec_apply_complex_mask',
    'DeepFilterNet_AENR': 'dfn_aenr_compose_stream',
}


class SelectOutputs(nn.Module):
    def __init__(self, model, indices):
        super().__init__()
        self.model = model
        self.indices = tuple(indices)

    def forward(self, *inputs):
        outputs = self.model(*inputs)
        return tuple(outputs[index] for index in self.indices)


def make_export_wrapper(model_name: str, model: nn.Module,
                        include_debug_outputs: bool = False):
    cls, all_names = WRAPPERS[model_name]
    indices = (tuple(range(len(all_names))) if include_debug_outputs
               else REQUIRED_OUTPUT_INDICES[model_name])
    names = tuple(all_names[index] for index in indices)
    return SelectOutputs(cls(model).eval(), indices).eval(), names


def load_checkpoint_model(model_name: str, checkpoint: str):
    module = importlib.import_module('AIAEC.%s.denoise' % model_name)
    result = module.load_model(checkpoint, 'cpu')
    return result[0], result[1]


def dummy_inputs(model_name: str, model, frames: int):
    if model_name == 'DeepFilterNet_AENR':
        return (
            torch.randn(1, 1, frames, model.n_erb),
            torch.randn(1, 2, frames, model.df_bins),
            torch.randn(1, 1, frames, model.n_erb),
            torch.randn(1, 2, frames, model.df_bins),
        ), ('error_erb', 'error_spec', 'far_erb', 'far_spec')
    shape = (1, frames, model.grid.n_freqs, 2)
    primary = 'microphone_ri' if model_name in ('Align_CRUSE', 'DeepVQE_S', 'CAGCRN') else 'linear_error_ri'
    return (torch.randn(shape), torch.randn(shape)), (primary, 'far_end_ri')


def alignment_depth(model):
    if hasattr(model, 'align'):
        return int(model.align.max_delay_frames)
    if hasattr(model, 'cata'):
        return int(model.cata.max_delay_frames)
    return 0


def set_alignment_depth(model, depth: int):
    """Apply a shape-independent deployment override and return old depth."""
    if depth <= 0:
        raise ValueError('alignment depth must be positive')
    old = alignment_depth(model)
    if old == 0:
        raise ValueError('this model has no delay-alignment depth')
    if hasattr(model, 'align'):
        model.align.max_delay_frames = int(depth)
    else:
        model.cata.max_delay_frames = int(depth)
    if hasattr(model, 'max_delay_frames'):
        model.max_delay_frames = int(depth)
    return old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_name', choices=MODEL_NAMES)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--max-delay-frames', type=int, default=None,
                        help='deployment-only D override for alignment models; '
                             'weight shapes are unchanged, numerical output is not')
    parser.add_argument('--include-debug-outputs', action='store_true',
                        help='also export delay/attention/intermediate tensors')
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    model, grid = load_checkpoint_model(args.model_name, args.checkpoint)
    checkpoint_delay_depth = alignment_depth(model)
    if args.max_delay_frames is not None:
        try:
            set_alignment_depth(model, args.max_delay_frames)
        except ValueError as error:
            parser.error(str(error))
    delay_depth = alignment_depth(model)
    if delay_depth and args.frames < delay_depth:
        parser.error('--frames=%d is shorter than this checkpoint alignment '
                     'depth D=%d; use a larger block or export a checkpoint '
                     'with smaller D' % (args.frames, delay_depth))
    wrapper, output_names = make_export_wrapper(
        args.model_name, model, args.include_debug_outputs)
    inputs, input_names = dummy_inputs(args.model_name, model, args.frames)
    with torch.no_grad():
        expected = wrapper(*inputs)
    if not all(torch.isfinite(value).all() for value in expected):
        raise RuntimeError('PyTorch export reference contains NaN or Inf')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(wrapper, inputs, args.output, input_names=input_names,
                      output_names=output_names, opset_version=args.opset,
                      do_constant_folding=True)
    import onnx
    graph = onnx.shape_inference.infer_shapes(onnx.load(args.output))
    onnx.checker.check_model(graph)
    metadata: Dict[str, str] = {
        'model_family': args.model_name, 'boundary': 'learned_control_outputs',
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'sample_rate': str(grid.sr), 'n_fft': str(grid.n_fft),
        'win_len': str(grid.win_len), 'hop_len': str(grid.hop_len),
        'frames_per_invocation': str(args.frames),
        'recurrent_state': 'internal_reset_each_invocation',
        'production_streaming_equivalent': 'false',
        'input_schema': ';'.join(
            '%s:%s' % (name, 'x'.join(str(int(size)) for size in value.shape))
            for name, value in zip(input_names, inputs)),
        'output_schema': ';'.join(
            '%s:%s' % (name, 'x'.join(str(int(size)) for size in value.shape))
            for name, value in zip(output_names, expected)),
        'debug_outputs_included': str(args.include_debug_outputs).lower(),
        'max_delay_frames': str(delay_depth),
        'checkpoint_max_delay_frames': str(checkpoint_delay_depth),
        'delay_depth_overridden': str(
            delay_depth != checkpoint_delay_depth).lower(),
        'c_composition_function': COMPOSITION[args.model_name],
        'complex_tensor_policy': 'real_imag_last_dimension',
        'c_prepost': ('DeepFilterNet_AENR/dfn_aenr_process.c' if args.model_name == 'DeepFilterNet_AENR'
                      else ('DeepVQE_S/deepvqe_process.c+AIAEC/aiaec_process.c'
                            if args.model_name == 'DeepVQE_S' else 'AIAEC/aiaec_process.c')),
    }
    onnx.helper.set_model_props(graph, metadata); onnx.save(graph, args.output)
    with open(os.path.splitext(args.output)[0] + '.json', 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True); fp.write('\n')
    if args.verify:
        import onnxruntime as ort
        session = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        actual = session.run(None, {name: value.numpy() for name, value in zip(input_names, inputs)})
        worst = max(float(np.max(np.abs(a - b.detach().numpy()))) for a, b in zip(actual, expected))
        if worst > 3e-4: raise RuntimeError('ONNX parity failed: %.6g' % worst)
        print('ONNX parity max_abs=%.6g' % worst)
    print(args.output)


if __name__ == '__main__':
    main()
