#!/usr/bin/env python3
"""Internal calibration backend used by each model's ``inference.py``.

NumPy archive::

    python3 AIAEC/Align_ULCNet/inference.py calib \
        --checkpoint checkpoint.pth \
        --primary-dir calibration/linear_error \
        --far-dir calibration/raw_far \
        --frames 8192 --max-delay-frames 8 \
        --output AIAEC/Align_ULCNet/calib/align_ulcnet_d8.npz

Per-frame, per-input raw binary files::

    python3 AIAEC/Align_ULCNet/inference.py calib \
        --checkpoint checkpoint.pth \
        --primary-dir calibration/linear_error \
        --far-dir calibration/raw_far \
        --frames 8192 --max-delay-frames 8 \
        --format bin \
        --output AIAEC/Align_ULCNet/calib/align_ulcnet_d8

Align-ULCNet may instead take RAW microphone WAVs: pass ``--primary-is-mic``
and the checkpoint-matched frozen PBFDKF derives the linear-error stems
in-process, persisting them beside the artifact (``<output>_linear_error/``).

Every calibration run also exports and parity-checks the ONNX graph the
tensors bind to (default ``<output>.onnx``, override with ``--onnx``).

Binary output uses one directory per ONNX input and one file per invocation,
for example ``h_gru0/h_gru0_0000.bin``.  ``manifest.json`` records each
frame shape, dtype, byte order, D and state contract.  The output directory
must not already exist, preventing stale files from an older capture.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_AINR_DIR = os.path.join(_REPO_ROOT, 'AINR')  # home of calibration_io
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _AINR_DIR not in sys.path:
    sys.path.insert(0, _AINR_DIR)

from calibration_io import (  # noqa: E402
    CALIBRATION_FORMATS,
    capture_calibration_inputs,
    resolve_calibration_format,
    sibling_onnx_path,
    write_calibration_artifact,
)

from AIAEC._calibration_common import (
    blocks_from_pair,
    discover_pairs,
    far_mode_provenance,
)
from AIAEC._export_common import (
    ALL_MODEL_NAMES,
    alignment_depth,
    load_checkpoint_model,
    set_alignment_depth,
)
from AIAEC._streaming_export import (
    GraphSplit,
    _build,
    export_graph,
    requires_contiguous_calibration,
    state_precision_policy,
)
from AIAEC.Align_ULCNet.inference import load_model as load_ulcnet_model
from AIAEC.Align_ULCNet.export_onnx import (
    INPUT_NAMES as ULCNET_INPUT_NAMES,
    STATE_LAYOUT_VERSION as ULCNET_STATE_LAYOUT_VERSION,
    export_graph as ulcnet_export_graph,
    next_state as ulcnet_next_state,
    stream_features as ulcnet_stream_features,
    SIGNAL_INPUTS as ULCNET_SIGNAL_INPUTS,
)


# Unlike the exporters, this recorder serves every candidate: Align-ULCNet's
# own graph is driven through its exporter's helpers below.
#
# Align-ULCNet's graph emits one head plus delta-state; its state slots are
# rebuilt by next_state() rather than sliced, so only signal_inputs is read.
_ULCNET_SPLIT = GraphSplit(
    signal_inputs=ULCNET_SIGNAL_INPUTS, head_outputs=1
)


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


def _materialize_linear_error(pairs, linear_contract, grid, output_path):
    """Derive linear-error stems from RAW microphone pairs, persistently.

    Runs the checkpoint-matched frozen PBFDKF (the exact engine deployment
    runs in C) hop by hop over every pair and writes the error stems beside
    the calibration artifact, so the recorded tensors stay auditable and the
    derived stems are reusable. Returns the pair list rewritten to use the
    derived stems as primary, plus the directory they were written to.
    """
    import soundfile as sf

    from AIAEC.inference_common import load_mic_far
    from AIAEC.training_common import LinearAecEngine

    root = str(output_path) + '_linear_error'
    derived = []
    for relative, primary_path, far_path in pairs:
        mic, far, _rates = load_mic_far(primary_path, far_path, grid.sr)
        engine = LinearAecEngine(n_lanes=1, sample_rate=grid.sr,
                                 contract=linear_contract)
        hops = []
        length = mic.shape[-1]
        for start in range(0, length, grid.hop_len):
            stop = min(start + grid.hop_len, length)
            error_hop, _ = engine(mic[:, start:stop], far[:, start:stop],
                                  grid.sr)
            hops.append(error_hop)
        error = torch.cat(hops, dim=-1)[:, :length]
        out_path = os.path.join(root, relative)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        sf.write(out_path, error.squeeze(0).cpu().numpy(), grid.sr,
                 subtype='FLOAT')
        derived.append((relative, out_path, far_path))
    return derived, root


def main(model_name: str) -> None:
    """Run one model's calib CLI; each model's inference.py names itself."""
    if model_name not in ALL_MODEL_NAMES:
        raise ValueError('unsupported AIAEC model: %s' % model_name)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--primary-dir', required=True)
    parser.add_argument('--far-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--format', choices=CALIBRATION_FORMATS, default=None,
                        help='bin or npz; inferred from --output when omitted')
    parser.add_argument('--frames', type=int, default=256,
                        help='number of streaming invocations to capture')
    parser.add_argument('--max-delay-frames', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--onnx', default=None,
                        help='where to write the graph these tensors bind to '
                             '(default: <output>.onnx)')
    parser.add_argument('--pair-replace', default=None,
                        metavar='PRIMARY_TOKEN:FAR_TOKEN',
                        help='pair mic_001.wav with lpb_001.wav by replacing '
                             'the token in primary names when looking up the '
                             'far tree (identical names still match first)')
    parser.add_argument('--primary-is-mic', action='store_true',
                        help='Align_ULCNet only: --primary-dir holds RAW '
                             'microphone WAVs; the checkpoint-matched frozen '
                             'PBFDKF derives the linear-error stems here and '
                             'persists them beside the artifact')
    args = parser.parse_args()
    args.model_name = model_name
    if args.frames <= 0:
        parser.error('--frames must be positive')
    try:
        artifact_format = resolve_calibration_format(args.output, args.format)
    except ValueError as error:
        parser.error(str(error))

    if args.primary_is_mic and args.model_name != 'Align_ULCNet':
        parser.error('--primary-is-mic applies only to Align_ULCNet; the '
                     'end-to-end models take microphone WAVs natively')

    # Pairs are discovered before the model load and export, so a bad
    # --primary-dir/--far-dir fails on its own actionable error first.
    pair_replace = None
    if args.pair_replace:
        pieces = args.pair_replace.split(':')
        if len(pieces) != 2 or not pieces[0]:
            parser.error('--pair-replace takes PRIMARY_TOKEN:FAR_TOKEN')
        pair_replace = (pieces[0], pieces[1])
    pairs = discover_pairs(args.primary_dir, args.far_dir, pair_replace)
    random.Random(args.seed).shuffle(pairs)

    # The graph is exported (and parity-checked) in the same process, from
    # the same wrapper the tensors below are recorded against, so the two
    # deployment artifacts cannot drift apart.
    onnx_path = sibling_onnx_path(args.output, args.onnx)
    if args.model_name == 'Align_ULCNet':
        model, grid, linear_contract = load_ulcnet_model(
            args.checkpoint, 'cpu',
            max_delay_frames=args.max_delay_frames,
        )
        wrapper, dummy, graph_metadata = ulcnet_export_graph(
            model, args.checkpoint, onnx_path, verify=True
        )
        input_names = ULCNET_INPUT_NAMES
        split = _ULCNET_SPLIT

        # The block carries RAW error/far RI; the host front end (the same
        # fixed math the deployment runs in C) produces the five feature
        # inputs the graph binds. State is rebuilt by next_state() rather
        # than sliced from the outputs.
        def make_signal(block):
            return ulcnet_stream_features(
                model,
                torch.from_numpy(block['error']).unsqueeze(0),
                torch.from_numpy(block['far']).unsqueeze(0),
            )

        def advance_state(state, outputs):
            return ulcnet_next_state(state, outputs, wrapper.delay_depth)

        if args.primary_is_mic:
            pairs, _ = _materialize_linear_error(
                pairs, linear_contract, grid, args.output
            )
    else:
        model, grid = load_checkpoint_model(args.model_name, args.checkpoint)
        checkpoint_depth = alignment_depth(model)
        if args.max_delay_frames is not None:
            set_alignment_depth(model, args.max_delay_frames)
        model.eval()
        built = _build(args.model_name, model)
        graph_metadata = export_graph(grid, built, args.checkpoint, onnx_path,
                                      checkpoint_depth, verify=True)
        wrapper, dummy, input_names, _output_names, split = built
        signal_names = input_names[:split.signal_inputs]

        def make_signal(block):
            return tuple(
                torch.from_numpy(block[name]).unsqueeze(0)
                for name in signal_names
            )

        def advance_state(state, outputs):
            return tuple(outputs[split.head_outputs:])
    feature_config = None

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
            for block in blocks:
                inputs = make_signal(block) + state
                capture_calibration_inputs(captured, input_names, inputs)
                outputs = wrapper(*inputs)
                state = advance_state(state, outputs)
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
    policy = state_precision_policy(args.model_name)
    report = {
        'schema': 'aiaec-stateless-stream-calibration-v1',
        'model_family': args.model_name,
        'checkpoint_sha256': graph_metadata['checkpoint_sha256'],
        'graph': os.path.basename(onnx_path),
        'sample_rate': int(grid.sr),
        'n_fft': int(grid.n_fft),
        'win_len': int(grid.win_len),
        'hop_len': int(grid.hop_len),
        'input_feature_frames': 1,
        'primary_source': (
            'raw_mic_via_frozen_pbfdkf' if args.primary_is_mic
            else 'materialized_primary_wavs'),
        'derived_linear_error_dir': (
            os.path.basename(str(args.output)) + '_linear_error'
            if args.primary_is_mic else None),
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
    write_calibration_artifact(
        args.output, arrays, report, artifact_format
    )
    print('%s: %d streaming frames, graph %s' %
          (args.output, report['frames'], onnx_path))
