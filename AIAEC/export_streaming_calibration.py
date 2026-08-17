#!/usr/bin/env python3
"""Capture representative inputs for stateless AIAEC streaming ONNX graphs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AIAEC.export_calibration import (
    _dfn_feature_config,
    blocks_from_pair,
    discover_pairs,
    far_mode_provenance,
)
from AIAEC.export_onnx import (
    alignment_depth,
    file_sha256,
    load_checkpoint_model,
    set_alignment_depth,
)
from AIAEC.export_streaming_onnx import (
    MODEL_NAMES as GENERIC_MODEL_NAMES,
    GraphSplit,
    _build,
    requires_contiguous_calibration,
    state_precision_policy,
)
from AIAEC.Align_ULCNet.denoise import load_model as load_ulcnet_model
from AIAEC.Align_ULCNet.export_streaming_onnx import (
    INPUT_NAMES as ULCNET_INPUT_NAMES,
    STATE_LAYOUT_VERSION as ULCNET_STATE_LAYOUT_VERSION,
    AlignUlcnetStreamingExport,
    dummy_inputs as ulcnet_dummy_inputs,
    next_state as ulcnet_next_state,
)
from AINR.DeepFilterNet2.export_onnx import (
    INPUT_FRAMES as DFN_INPUT_FRAMES,
    feature_windows,
)


MODEL_NAMES = ('Align_ULCNet',) + GENERIC_MODEL_NAMES

# Align-ULCNet's graph emits one head plus delta-state; its state slots are
# rebuilt by next_state() rather than sliced, so only signal_inputs is read.
_ULCNET_SPLIT = GraphSplit(signal_inputs=2, head_outputs=1)


def input_range_report(arrays, policy):
    """Per-tensor calibration ranges, minus the tensors PTQ must not touch.

    A tensor named in ``state_precision_policy`` carries its policy marker
    instead of a range. Emitting both would be worse than useless: the range
    of an undecayed accumulator is a function of how long this capture
    happened to run, and float percentiles over the int64 frame counter
    describe nothing at all -- yet a quantizer reading the ``inputs`` block has
    no way to know it was meant to skip those entries.
    """
    report = {}
    for name, value in arrays.items():
        entry = {'shape': list(value.shape), 'dtype': str(value.dtype)}
        if name in policy:
            entry['precision'] = policy[name]
        else:
            entry.update({
                'min': float(value.min()),
                'max': float(value.max()),
                'p001': float(np.percentile(value, 0.1)),
                'p999': float(np.percentile(value, 99.9)),
            })
        report[name] = entry
    return report


def _capture(captured, names, tensors):
    for name, value in zip(names, tensors):
        captured.setdefault(name, []).append(
            value.detach().cpu().numpy().copy()
        )


def _dfn_windows(blocks, name):
    """Slide the exporter's own zero-padded window over the captured frames."""
    frames = torch.cat(
        [torch.from_numpy(block[name]).unsqueeze(0) for block in blocks],
        dim=2,
    )
    return feature_windows(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_name', choices=MODEL_NAMES)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--primary-dir', required=True)
    parser.add_argument('--far-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=256,
                        help='number of streaming invocations to capture')
    parser.add_argument('--max-delay-frames', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dfn-config', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'DeepFilterNet_AENR', 'config.ini'))
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    if not args.output.lower().endswith('.npz'):
        parser.error('--output must end in .npz')

    if args.model_name == 'Align_ULCNet':
        model, grid, _linear_contract = load_ulcnet_model(
            args.checkpoint, 'cpu',
            max_delay_frames=args.max_delay_frames,
        )
        wrapper = AlignUlcnetStreamingExport(model).eval()
        dummy = ulcnet_dummy_inputs(wrapper.delay_depth)
        input_names = ULCNET_INPUT_NAMES
        split = _ULCNET_SPLIT
    else:
        model, grid = load_checkpoint_model(args.model_name, args.checkpoint)
        if args.max_delay_frames is not None:
            set_alignment_depth(model, args.max_delay_frames)
        model.eval()
        wrapper, dummy, input_names, _output_names, split = _build(
            args.model_name, model
        )
    pairs = discover_pairs(args.primary_dir, args.far_dir)
    random.Random(args.seed).shuffle(pairs)
    feature_config = (
        _dfn_feature_config(args.dfn_config, grid)
        if args.model_name == 'DeepFilterNet_AENR' else None
    )

    captured = {}
    source_files = []
    with torch.no_grad():
        for relative, primary_path, far_path in pairs:
            blocks = list(blocks_from_pair(
                args.model_name,
                model,
                grid,
                primary_path,
                far_path,
                1,
                feature_config,
            ))
            if not blocks:
                continue
            if (requires_contiguous_calibration(args.model_name)
                    and len(blocks) < args.frames):
                continue
            state = tuple(
                value.clone() for value in dummy[split.signal_inputs:]
            )
            used = False
            if args.model_name == 'DeepFilterNet_AENR':
                windows = zip(
                    _dfn_windows(blocks, 'error_erb'),
                    _dfn_windows(blocks, 'error_spec'),
                    _dfn_windows(blocks, 'far_erb'),
                    _dfn_windows(blocks, 'far_spec'),
                )
                for signal in windows:
                    inputs = signal + state
                    _capture(captured, input_names, inputs)
                    outputs = wrapper(*inputs)
                    state = tuple(outputs[split.head_outputs:])
                    used = True
                    if len(next(iter(captured.values()))) >= args.frames:
                        break
            else:
                signal_names = input_names[:split.signal_inputs]
                for block in blocks:
                    signal = tuple(
                        torch.from_numpy(block[name]).unsqueeze(0)
                        for name in signal_names
                    )
                    inputs = signal + state
                    _capture(captured, input_names, inputs)
                    outputs = wrapper(*inputs)
                    if args.model_name == 'Align_ULCNet':
                        state = ulcnet_next_state(
                            state, outputs, wrapper.delay_depth
                        )
                    else:
                        state = tuple(outputs[split.head_outputs:])
                    used = True
                    if len(next(iter(captured.values()))) >= args.frames:
                        break
            if used:
                source_files.append(relative)
            if captured and len(next(iter(captured.values()))) >= args.frames:
                break
    if not captured and requires_contiguous_calibration(args.model_name):
        raise RuntimeError(
            '%s calibration requires one uninterrupted source with at least '
            '%d frames; cumulative state ranges must not be assembled from '
            'state-reset clips' % (args.model_name, args.frames)
        )
    if not captured:
        raise RuntimeError('no calibration frames were produced')

    arrays = {}
    for name, values in captured.items():
        value = np.stack(values[:args.frames])
        # Align-CRUSE's absolute frame counter is an int64 graph input. Keep
        # state dtypes exact; only floating tensors are normalised to fp32.
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32, copy=False)
        arrays[name] = value
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    policy = state_precision_policy(args.model_name)
    report = {
        'schema': 'aiaec-stateless-stream-calibration-v1',
        'model_family': args.model_name,
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'sample_rate': int(grid.sr),
        'n_fft': int(grid.n_fft),
        'win_len': int(grid.win_len),
        'hop_len': int(grid.hop_len),
        'input_feature_frames': (
            DFN_INPUT_FRAMES
            if args.model_name == 'DeepFilterNet_AENR' else 1
        ),
        'calibration_far_input_mode': far_mode_provenance(
            args.model_name)[0],
        'deployment_far_input_mode': far_mode_provenance(
            args.model_name)[1],
        'max_delay_frames': alignment_depth(model),
        # The exported graph records the same field; a board that pairs a
        # calibration set with a graph compares the two before trusting the
        # ranges. Only Align-ULCNet's state layout is versioned today.
        'state_layout_version': (
            ULCNET_STATE_LAYOUT_VERSION
            if args.model_name == 'Align_ULCNet' else None
        ),
        'frames': int(next(iter(arrays.values())).shape[0]),
        'seed': args.seed,
        'source_files': source_files,
        'state_precision_policy': policy,
        'contiguous_state_frames': (
            int(next(iter(arrays.values())).shape[0])
            if requires_contiguous_calibration(args.model_name) else None
        ),
        'inputs': input_range_report(arrays, policy),
    }
    with open(os.path.splitext(args.output)[0] + '.json', 'w',
              encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('%s: %d streaming frames' % (args.output, report['frames']))


if __name__ == '__main__':
    main()
