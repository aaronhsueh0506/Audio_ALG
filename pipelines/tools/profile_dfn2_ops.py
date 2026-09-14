#!/usr/bin/env python3
"""Reproducible structural MAC budget for the shipped streaming DFN2 graph.

Counts convolution, transposed-convolution, linear, grouped-linear and GRU
multiply-accumulates for one batch-one graph invocation. Elementwise ops,
normalization, activation, host feature/compose DSP and memory traffic are not
included, so this is an accelerator admission budget rather than a cycle
prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.dfn2_stage import HOP, SAMPLE_RATE, dfn2_modules, load_dfn2  # noqa: E402


def count_model():
    assets = load_dfn2(seed=0)
    _train, model_module, export_module = dfn2_modules()
    wrapper = export_module.StatelessDFN2Heads(assets.model).eval()
    totals = {
        'conv2d': 0,
        'conv_transpose2d': 0,
        'linear': 0,
        'grouped_linear': 0,
        'gru': 0,
    }
    handles = []

    def record(kind):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(module, torch.nn.Conv2d):
                macs = (value.numel() * (module.in_channels // module.groups)
                        * module.kernel_size[0] * module.kernel_size[1])
            elif isinstance(module, torch.nn.ConvTranspose2d):
                source = inputs[0]
                macs = (source.numel() * (module.out_channels // module.groups)
                        * module.kernel_size[0] * module.kernel_size[1])
            elif isinstance(module, torch.nn.Linear):
                macs = value.numel() * module.in_features
            elif isinstance(module, model_module.GroupedLinearEinsum):
                source = inputs[0]
                vectors = source.numel() // module.input_size
                macs = (vectors * module.input_size * module.hidden_size
                        // module.groups)
            elif isinstance(module, torch.nn.GRU):
                source = inputs[0]
                batch = source.shape[0] if module.batch_first else source.shape[1]
                steps = source.shape[1] if module.batch_first else source.shape[0]
                directions = 2 if module.bidirectional else 1
                macs = 0
                for layer in range(module.num_layers):
                    width = (module.input_size if layer == 0 else
                             module.hidden_size * directions)
                    macs += (batch * steps * directions * 3
                             * (width * module.hidden_size
                                + module.hidden_size * module.hidden_size))
            else:  # pragma: no cover - registration below is exhaustive
                return
            totals[kind] += int(macs)
        return hook

    for module in wrapper.modules():
        # GroupedLinearEinsum contains no child Linear, so this ordering does
        # not double-count its broadcast matmul.
        if isinstance(module, model_module.GroupedLinearEinsum):
            handles.append(module.register_forward_hook(record('grouped_linear')))
        elif isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(record('conv2d')))
        elif isinstance(module, torch.nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(
                record('conv_transpose2d')))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(record('linear')))
        elif isinstance(module, torch.nn.GRU):
            handles.append(module.register_forward_hook(record('gru')))

    with torch.no_grad():
        wrapper(*wrapper.initial_inputs())
    for handle in handles:
        handle.remove()

    per_frame = sum(totals.values())
    frames_per_second = SAMPLE_RATE / HOP
    params = sum(parameter.numel() for parameter in assets.model.parameters())
    return {
        'scope': ('major model MACs only; excludes elementwise/BN/activation, '
                  'host DSP and memory traffic'),
        'sample_rate': SAMPLE_RATE,
        'hop': HOP,
        'frames_per_second': frames_per_second,
        'trainable_parameters': params,
        'weights_mib': {
            'int8': params / 2 ** 20,
            'fp16': 2 * params / 2 ** 20,
            'fp32': 4 * params / 2 ** 20,
        },
        'macs_per_frame_by_op': totals,
        'major_macs_per_frame': per_frame,
        'major_macs_per_second': per_frame * frames_per_second,
    }


def resampler_budget(native_rate: int):
    if native_rate == SAMPLE_RATE:
        one_up = one_down = 0
    else:
        # audio_common's current polyphase layout gives 33 MAC/output for
        # every supported native->48k pair. Downstream taps differ by rate.
        import math
        def macs(rate_in, rate_out):
            gcd = math.gcd(rate_in, rate_out)
            up, down = rate_out // gcd, rate_in // gcd
            ratio = max(up, down)
            length = 32 * ratio + 1
            taps = (length + up - 1) // up
            return taps * rate_out

        one_up = macs(native_rate, SAMPLE_RATE)
        one_down = macs(SAMPLE_RATE, native_rate)
    return {
        'native_rate': native_rate,
        'one_channel_native_to_48k_macs_per_second': one_up,
        'one_channel_48k_to_native_macs_per_second': one_down,
        'mono_whole_pipeline_rate_boundary_macs_per_second':
            2 * one_up + one_down,
        'four_channel_whole_pipeline_rate_boundary_macs_per_second':
            5 * one_up + one_down,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-rate', type=int, default=16000,
                        choices=(8000, 16000, 24000, 32000, 48000))
    args = parser.parse_args()
    print(json.dumps({
        'model': count_model(),
        'rate_boundary': resampler_budget(args.native_rate),
    }, indent=2))


if __name__ == '__main__':
    main()
