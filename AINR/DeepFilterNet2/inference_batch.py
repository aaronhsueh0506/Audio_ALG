#!/usr/bin/env python3
"""Export and calibrate a fixed-batch DFN2 streaming graph.

The batch axis represents independent streaming lanes.  It must never be used
to place consecutive frames from one stream into one model invocation: each
lane owns a separate GRU state and ``df_convp`` history.

Export a batch graph::

    python3 inference_batch.py export \
        --model output/dfn2_best.pth --batch-size 4 \
        --gru-state-layout combined \
        --output output/dfn2_stream_b4.onnx --verify

Generate the matching ONNX graph and per-invocation calibration BIN files::

    python3 inference_batch.py calib \
        --model output/dfn2_best.pth --wav-dir /path/to/noisy_wavs \
        --batch-size 4 --batches 1000 --gru-state-layout combined \
        --format bin --output calib/dfn2_b4

Every BIN file contains one complete NPU invocation, including all batch
elements, and keeps the exact static shape declared by the ONNX input.  For
example, combined GRU state uses ``(B, 5, 1, hidden)``.  ``--format npz`` is
also supported.  The shipped C runtime remains a batch-one/split-state API;
this tool is for NPU profiling and compiler evaluation.
"""

import argparse
import glob
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torchaudio

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AINR_DIR = os.path.dirname(_SCRIPT_DIR)
if _AINR_DIR not in sys.path:
    sys.path.insert(0, _AINR_DIR)

try:
    from .inference import load_model
    from .train import extract_dfn2_features
    from .export_onnx import (
        COMBINED_GRU_STATE_NAME,
        DEFAULT_GRU_STATE_LAYOUT,
        GRU_STATE_LAYOUTS,
        GRU_STATE_NAMES,
        HEAD_OUTPUT_NAMES,
        INPUT_FRAMES,
        StatelessDFN2Heads,
        export_graph,
        feature_windows,
        gru_state_slice_report,
    )
    from ..calibration_io import (
        CALIBRATION_FORMATS,
        capture_calibration_inputs,
        resolve_calibration_format,
        sibling_onnx_path,
        write_calibration_artifact,
    )
except ImportError:  # direct ``python inference_batch.py`` execution
    from inference import load_model
    from train import extract_dfn2_features
    from export_onnx import (
        COMBINED_GRU_STATE_NAME,
        DEFAULT_GRU_STATE_LAYOUT,
        GRU_STATE_LAYOUTS,
        GRU_STATE_NAMES,
        HEAD_OUTPUT_NAMES,
        INPUT_FRAMES,
        StatelessDFN2Heads,
        export_graph,
        feature_windows,
        gru_state_slice_report,
    )
    from calibration_io import (
        CALIBRATION_FORMATS,
        capture_calibration_inputs,
        resolve_calibration_format,
        sibling_onnx_path,
        write_calibration_artifact,
    )


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return value


def _tensor_stats(value):
    low, high = np.percentile(value, [0.1, 99.9])
    return {
        'shape': [int(size) for size in value.shape],
        'dtype': str(value.dtype),
        'min': float(value.min()),
        'max': float(value.max()),
        'p001': float(low),
        'p999': float(high),
    }


def _merge_independent_samples(names, samples, combined):
    """Merge batch-one state snapshots into one fixed-batch invocation.

    Feature tensors, convolution history, and combined state carry their
    graph batch on axis 0.  Split GRU states stay in PyTorch's native
    ``(layers, batch, hidden)`` order and therefore concatenate on axis 1.
    """
    if not samples:
        raise ValueError('cannot merge an empty batch')
    if any(len(sample) != len(names) for sample in samples):
        raise ValueError('sample tensor count disagrees with graph inputs')
    merged = []
    for index, name in enumerate(names):
        axis = 1 if not combined and name in GRU_STATE_NAMES else 0
        merged.append(torch.cat(
            tuple(sample[index] for sample in samples), dim=axis
        ))
    return tuple(merged)


def _wav_feature_stream(path, model, params):
    wave, source_rate = sf.read(path, dtype='float32', always_2d=True)
    wave = torch.from_numpy(wave.mean(axis=1))
    if source_rate != params['SR']:
        wave = torchaudio.functional.resample(
            wave, source_rate, params['SR']
        )
    window = torch.hann_window(params['WIN_LEN']).sqrt()
    spectrum = torch.stft(
        wave.unsqueeze(0),
        params['N_FFT'],
        params['HOP_LEN'],
        params['WIN_LEN'],
        window=window,
        normalized=True,
        return_complex=True,
    )
    _, erb, spec, _ = extract_dfn2_features(
        spectrum,
        model.erb_fb,
        model.df_bins,
        feature_cfg=params['FEATURE_CFG'],
    )
    return erb, spec


def _capture_fixed_batches(model, params, files, batch_wrapper, batches):
    """Capture valid state snapshots, then pack them into independent lanes."""
    single = StatelessDFN2Heads(
        model,
        gru_state_layout=batch_wrapper.gru_state_layout,
        batch_size=1,
    ).eval()
    names = batch_wrapper.input_names
    captured = {name: [] for name in names}
    pending = []
    source_files = []

    with torch.no_grad():
        for path in files:
            erb, spec = _wav_feature_stream(path, model, params)
            state = single.initial_inputs()[2:]
            used = False
            for erb_window, spec_window in zip(
                    feature_windows(erb), feature_windows(spec)):
                inputs = (erb_window, spec_window) + state
                # Clone before the next invocation.  These snapshots are
                # independent calibration lanes even if they came from
                # adjacent points on the same real trajectory.
                pending.append(tuple(value.detach().clone()
                                     for value in inputs))
                outputs = single(*inputs)
                state = tuple(outputs[len(HEAD_OUTPUT_NAMES):])
                used = True
                if len(pending) == batch_wrapper.batch_size:
                    merged = _merge_independent_samples(
                        names, pending,
                        batch_wrapper.gru_state_layout.combined,
                    )
                    capture_calibration_inputs(captured, names, merged)
                    pending = []
                    if len(captured[names[0]]) >= batches:
                        break
            if used:
                source_files.append(path)
            if len(captured[names[0]]) >= batches:
                break

    produced = len(captured[names[0]])
    if produced == 0:
        raise RuntimeError(
            'not enough source frames to form one batch of %d' %
            batch_wrapper.batch_size
        )
    return captured, source_files, produced, len(pending)


def _load(args):
    return load_model(SimpleNamespace(config=args.config, model=args.model))


def export_main(args):
    model, params = _load(args)
    wrapper = StatelessDFN2Heads(
        model,
        gru_state_layout=args.gru_state_layout,
        batch_size=args.batch_size,
    ).eval()
    export_graph(
        wrapper,
        params,
        args.model,
        args.output,
        opset=args.opset,
        verify=args.verify,
    )
    print('%s: fixed batch %d' % (args.output, args.batch_size))


def calibration_main(args):
    try:
        artifact_format = resolve_calibration_format(
            args.output, args.format
        )
    except ValueError as error:
        raise SystemExit(str(error))
    files = sorted(glob.glob(
        os.path.join(args.wav_dir, '**', '*.wav'), recursive=True
    ))
    if not files:
        raise FileNotFoundError('no WAV files under %s' % args.wav_dir)
    random.Random(args.seed).shuffle(files)

    model, params = _load(args)
    wrapper = StatelessDFN2Heads(
        model,
        gru_state_layout=args.gru_state_layout,
        batch_size=args.batch_size,
    ).eval()
    onnx_path = sibling_onnx_path(args.output, args.onnx)
    graph_metadata = export_graph(
        wrapper,
        params,
        args.model,
        onnx_path,
        opset=args.opset,
        verify=True,
    )
    captured, source_paths, produced, discarded = _capture_fixed_batches(
        model, params, files, wrapper, args.batches
    )
    arrays = {
        name: np.stack(values[:args.batches]).astype(
            np.float32, copy=False
        )
        for name, values in captured.items()
    }
    report = {
        'schema': 'dfn2-fixed-batch-stream-calibration-v1',
        'gru_state_layout': graph_metadata['gru_state_layout'],
        'checkpoint_sha256': graph_metadata['checkpoint_sha256'],
        'graph': os.path.basename(onnx_path),
        'sample_rate': params['SR'],
        'n_fft': params['N_FFT'],
        'win_len': params['WIN_LEN'],
        'hop_len': params['HOP_LEN'],
        'input_feature_frames': INPUT_FRAMES,
        # calibration_io calls the leading axis "frames"; here one entry is
        # one NPU invocation containing batch_size independent frame states.
        'frames': produced,
        'batch_invocations': produced,
        'fixed_batch_size': args.batch_size,
        'source_frame_snapshots': produced * args.batch_size,
        'batch_semantics': 'independent_streaming_lanes',
        'discarded_incomplete_batch_samples': discarded,
        'seed': args.seed,
        'source_files': [os.path.relpath(path, args.wav_dir)
                         for path in source_paths],
        'inputs': {name: _tensor_stats(value)
                   for name, value in arrays.items()},
    }
    if wrapper.gru_state_layout.combined:
        report['gru_state_slices'] = gru_state_slice_report(
            arrays[COMBINED_GRU_STATE_NAME],
            _tensor_stats,
            wrapper.gru_state_slices,
        )
    write_calibration_artifact(
        args.output, arrays, report, artifact_format
    )
    print('%s: %d invocations x batch %d, graph %s' % (
        args.output, produced, args.batch_size, onnx_path
    ))


def _common_parser(parser):
    parser.add_argument(
        '--config', default=os.path.join(_SCRIPT_DIR, 'config.ini')
    )
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch-size', type=_positive_int, required=True)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument(
        '--gru-state-layout', choices=sorted(GRU_STATE_LAYOUTS),
        default=DEFAULT_GRU_STATE_LAYOUT,
    )


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    export_parser = commands.add_parser(
        'export', help='write a fixed-batch ONNX graph'
    )
    _common_parser(export_parser)
    export_parser.add_argument('--output', required=True)
    export_parser.add_argument('--verify', action='store_true')
    export_parser.set_defaults(run=export_main)

    calib_parser = commands.add_parser(
        'calib', help='write matching batch ONNX and calibration tensors'
    )
    _common_parser(calib_parser)
    calib_parser.add_argument('--wav-dir', required=True)
    calib_parser.add_argument('--output', required=True)
    calib_parser.add_argument('--format', choices=CALIBRATION_FORMATS,
                              default=None)
    calib_parser.add_argument(
        '--batches', '--frames', dest='batches', type=_positive_int,
        default=256,
        help='number of NPU invocations to capture; each contains '
             '--batch-size independent frame/state snapshots (default: 256)',
    )
    calib_parser.add_argument('--seed', type=int, default=42)
    calib_parser.add_argument(
        '--onnx', default=None,
        help='graph path (default: <output>.onnx)',
    )
    calib_parser.set_defaults(run=calibration_main)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    cli()
