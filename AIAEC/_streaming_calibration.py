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
Without it, ``--primary-dir`` must hold stems stamped with the checkpoint's own
linear-AEC contract (that directory's ``linear_error.contract.json``, or a
``linear_error.done.json`` corpus ledger beside it); an unstamped, retired or
mismatching directory is refused before anything is written, unless
``--allow-unstamped-linear-error`` is given for the unstamped case (the report
records the override as ``linear_error_stamp``). The Align-ULCNet report and
manifest are stamped with the checkpoint's ``linear_aec_contract_version``,
``aec_behavior_hash``, ``linear_aec_contract_hash`` and ``error_input_seam``;
``PTQ/calib_analyze.py`` refuses an artifact whose stamp is missing, from
another contract version, or retired. ``--primary-is-mic`` stamps itself: the
derived directory gets its ``linear_error.contract.json`` from the same
checkpoint contract.

Every calibration run also exports and parity-checks the ONNX graph the
tensors bind to (default ``<output>.onnx``, override with ``--onnx``).

Binary output uses one directory per ONNX input and one file per invocation,
for example ``h_gru/h_gru_0000.bin``.  ``manifest.json`` records each
frame shape, dtype, byte order, D and state contract.  The output directory
must not already exist, preventing stale files from an older capture.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings

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
    host_signal_inputs,
    export_graph,
    requires_contiguous_calibration,
    state_precision_policy,
)
from AIAEC.Align_ULCNet.inference import load_model as load_ulcnet_model
from AIAEC.dataset_gen.linear_aec import (
    LINEAR_AEC_CONTRACT_VERSION,
    RETIRED_BEHAVIOR_HASHES,
    LinearAecContract,
    require_linear_aec_contract,
)
from AIAEC.dataset_gen.linear_error_ledger import (
    claimed_vs_present,
    ledger_path,
    recorded_contract,
    recorded_identity,
)
from AIAEC.dataset_gen.seq_layout import scan_chunks
from AIAEC.Align_ULCNet.export_onnx import (
    ERROR_INPUT_SEAM as ULCNET_ERROR_INPUT_SEAM,
    INPUT_NAMES as ULCNET_INPUT_NAMES,
    LAYOUT_VERSION as ULCNET_LAYOUT_VERSION,
    SIGNAL_INPUTS as ULCNET_SIGNAL_INPUTS,
    export_graph as ulcnet_export_graph,
    graph_signals as ulcnet_graph_signals,
    next_state as ulcnet_next_state,
)


# Unlike the exporters, this recorder serves every candidate: Align-ULCNet's
# own graph is driven through its exporter's helpers below.


def input_range_report(arrays, policy):
    """Per-tensor calibration ranges, minus the tensors PTQ must not touch.

    A tensor named in ``state_precision_policy`` carries its policy marker
    instead of a range: the range of an undecayed accumulator depends on how
    long the capture ran, and percentiles over the int64 frame counter describe
    nothing, yet a quantizer reading the ``inputs`` block could not know to
    skip them.
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


LINEAR_ERROR_SIDECAR = 'linear_error.contract.json'
LINEAR_ERROR_SIDECAR_SCHEMA = 'aiaec-linear-error-stems-v1'


def linear_error_sidecar(linear_contract):
    """Self-description of a derived linear-error directory.

    ``linear_contract`` is the checkpoint's recorded linear-AEC contract dict.
    The sidecar repeats it with its fingerprint, the front-end contract version
    and the AEC behaviour hash, plus the seam the stems were taken from, so a
    later run that reuses the directory as ``--primary-dir`` can check the stems
    against its own checkpoint instead of trusting the directory name.
    """
    contract = LinearAecContract.from_dict(linear_contract)
    return {
        'schema': LINEAR_ERROR_SIDECAR_SCHEMA,
        'linear_aec_contract_version': contract.version,
        'aec_behavior_hash': contract.aec_behavior_hash,
        'linear_aec_contract_hash': contract.fingerprint(),
        'error_input_seam': ULCNET_ERROR_INPUT_SEAM,
        'sample_rate': contract.sample_rate,
        'linear_aec': contract.as_dict(),
    }


def _contract_from(recorded, path):
    """``LinearAecContract.from_dict`` with the stamp file named in the error
    (it refuses retired versions and malformed contracts)."""
    try:
        return LinearAecContract.from_dict(recorded)
    except ValueError as error:
        raise ValueError('%s: %s' % (path, error)) from error


def _stamp_from_sidecar(path):
    """The linear-AEC contract a derived-stems sidecar vouches for."""
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            data = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError('%s does not read as JSON (%s)' % (path, error))
    if (not isinstance(data, dict)
            or data.get('schema') != LINEAR_ERROR_SIDECAR_SCHEMA
            or not isinstance(data.get('linear_aec'), dict)):
        raise ValueError('%s is not a %s sidecar' % (
            path, LINEAR_ERROR_SIDECAR_SCHEMA))
    stamp = _contract_from(data['linear_aec'], path)
    if data.get('linear_aec_contract_hash') != stamp.fingerprint():
        raise ValueError(
            '%s is internally inconsistent: its linear_aec contract '
            'fingerprints to %s, the file claims %s; it was edited or merged '
            'from two runs' % (path, stamp.fingerprint()[:12],
                               str(data.get('linear_aec_contract_hash'))[:12]))
    if data.get('error_input_seam') != ULCNET_ERROR_INPUT_SEAM:
        raise ValueError(
            '%s was taken from seam %r, the graph consumes %r' % (
                path, data.get('error_input_seam'), ULCNET_ERROR_INPUT_SEAM))
    return stamp


def _stamp_from_ledger(root):
    """The contract a corpus ledger beside ``root`` records, or ``None``.

    ``None`` covers no ledger and a legacy one without a recorded identity: the
    fingerprint alone cannot say which front end wrote the data. A ledger that
    does not name every chunk sequence present is a corpus mid-rematerialization
    (part old front end, part new) and is refused.
    """
    identity = recorded_identity(root)
    if identity is None:
        return None
    path = ledger_path(root)
    recorded = recorded_contract(root)
    stamp = _contract_from(identity, path)
    if stamp.fingerprint() != recorded:
        raise ValueError(
            '%s is internally inconsistent: the contract it records '
            'fingerprints to %s, the ledger claims %s' % (
                path, stamp.fingerprint()[:12], str(recorded)[:12]))
    _, missing, _ = claimed_vs_present(root, recorded, scan_chunks(root))
    if missing:
        raise ValueError(
            '%s does not claim %d of the chunk sequence(s) under %s (first: '
            '%06d): the corpus is part new front end and part old' % (
                path, len(missing), root, missing[0]))
    return stamp


def linear_error_provenance(primary_dir, linear_contract,
                            allow_unstamped=False):
    """Refuse pre-made linear-error stems from another front end than the model's.

    ``--primary-dir`` is taken as-is by the default Align-ULCNet path, so a
    directory written before the analysis-seam front end would calibrate
    silently. The stems must be stamped, by the sidecar of a derived
    directory or by a corpus ledger recording its contract, and the stamp must
    be the checkpoint's own linear-AEC contract (``linear_contract``) and not a
    retired version or behaviour hash. A matching retired stamp is refused too:
    the checkpoint check alone would let both sides agree on the old front end.

    No stamp refuses, unless ``allow_unstamped`` is set; that is recorded as
    such in the report. Returns the provenance dict the report carries.
    """
    root = os.path.abspath(primary_dir)
    sidecar = os.path.join(root, LINEAR_ERROR_SIDECAR)
    if os.path.isfile(sidecar):
        origin, path, stamp = 'sidecar', sidecar, _stamp_from_sidecar(sidecar)
    else:
        origin, path = 'corpus_ledger', ledger_path(root)
        stamp = _stamp_from_ledger(root)
    if stamp is None:
        if not allow_unstamped:
            raise ValueError(
                'linear-error stems in %s carry no front-end stamp (no %s, '
                'and no corpus ledger recording its contract at %s), so '
                'nothing says which linear-AEC front end wrote them. Derive '
                'them with --primary-is-mic from raw microphone WAVs, or '
                'pass --allow-unstamped-linear-error if you vouch for the '
                'data.' % (root, LINEAR_ERROR_SIDECAR, ledger_path(root)))
        warnings.warn(
            'calibrating on unstamped linear-error stems from %s; their '
            'front end is not verified against the checkpoint' % root,
            RuntimeWarning, stacklevel=2)
        return {'origin': 'unstamped_override', 'path': None}
    if stamp.aec_behavior_hash in RETIRED_BEHAVIOR_HASHES:
        raise ValueError(
            '%s records AEC behaviour %s, which is retired: it produced a '
            'different linear_error from this build (%s). These stems must '
            'not be calibrated on; recompute them with --primary-is-mic or '
            'rematerialize the corpus.' % (
                path, stamp.aec_behavior_hash, LINEAR_AEC_CONTRACT_VERSION))
    # The recorded side is the stamp, as for a corpus ledger at pack time.
    require_linear_aec_contract(
        linear_contract, stamp.as_dict(),
        'linear-error stems in %s (%s, expected) against the checkpoint '
        '(got)' % (root, origin))
    return {
        'origin': origin,
        'path': path,
        'linear_aec_contract_version': stamp.version,
        'aec_behavior_hash': stamp.aec_behavior_hash,
    }


def _materialize_linear_error(pairs, linear_contract, grid, output_path):
    """Derive linear-error stems from RAW microphone pairs, persistently.

    Runs the checkpoint-matched frozen PBFDKF (the exact engine deployment
    runs in C) hop by hop over every pair and writes the error stems beside
    the calibration artifact, so the recorded tensors stay auditable and the
    derived stems are reusable; ``linear_error.contract.json`` there records
    which contract produced them. Returns the pair list rewritten to use the
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
    # Written last: a run that stopped part-way leaves stems without a sidecar,
    # never a sidecar that vouches for an incomplete set.
    with open(os.path.join(root, LINEAR_ERROR_SIDECAR), 'w',
              encoding='utf-8') as stream:
        json.dump(linear_error_sidecar(linear_contract), stream, indent=2,
                  sort_keys=True)
        stream.write('\n')
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
    parser.add_argument('--allow-unstamped-linear-error', action='store_true',
                        help='Align_ULCNet only: calibrate on --primary-dir '
                             'linear-error WAVs that carry no front-end '
                             'stamp (no linear_error.contract.json, no '
                             'corpus ledger). Without it such a directory is '
                             'refused; the report records the override')
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
    if args.allow_unstamped_linear_error and (
            args.primary_is_mic or args.model_name != 'Align_ULCNet'):
        parser.error('--allow-unstamped-linear-error applies only to '
                     'Align_ULCNet --primary-dir linear-error WAVs, not to '
                     '--primary-is-mic or the microphone-input models')

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
    frontend_stamp = {}
    linear_error_stamp = None
    if args.model_name == 'Align_ULCNet':
        model, grid, linear_contract = load_ulcnet_model(
            args.checkpoint, 'cpu',
            max_delay_frames=args.max_delay_frames,
        )
        # Before the export writes anything: stems from another front end
        # must not reach the capture, and a refusal leaves no artifact.
        linear_error_stamp = (
            {'origin': 'derived_in_process', 'path': None}
            if args.primary_is_mic else linear_error_provenance(
                args.primary_dir, linear_contract,
                args.allow_unstamped_linear_error))
        wrapper, dummy, graph_metadata = ulcnet_export_graph(
            model, args.checkpoint, onnx_path, verify=True,
        )
        input_names = ULCNET_INPUT_NAMES
        # One head plus full next-state; the state slots are taken from
        # next_state() rather than sliced, so only signal_inputs is read.
        split = GraphSplit(
            signal_inputs=ULCNET_SIGNAL_INPUTS, head_outputs=1
        )
        state_layout_version = ULCNET_LAYOUT_VERSION
        # Which linear-AEC front end produced the error tensors these ranges
        # describe: the checkpoint's own contract, the same fields the graph
        # sidecar carries. Only Align-ULCNet consumes a linear error; the
        # end-to-end models take the raw microphone and carry no stamp.
        frontend_stamp = {
            'linear_aec_contract_version': linear_contract['version'],
            'aec_behavior_hash': linear_contract['aec_behavior_hash'],
            'linear_aec_contract_hash': LinearAecContract.from_dict(
                linear_contract).fingerprint(),
            'error_input_seam': ULCNET_ERROR_INPUT_SEAM,
        }

        # The block carries RAW error/far RI; graph_signals applies the host
        # front end. State is rebuilt by next_state() rather than sliced.
        def make_signal(block):
            return ulcnet_graph_signals(
                model,
                torch.from_numpy(block['error']).unsqueeze(0),
                torch.from_numpy(block['far']).unsqueeze(0),
            )

        def advance_state(state, outputs):
            return ulcnet_next_state(outputs)

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
        state_layout_version = None

        def make_signal(block):
            return host_signal_inputs(args.model_name, model, tuple(
                torch.from_numpy(block[name]).unsqueeze(0)
                for name in signal_names
            ))

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
        # Where the primary linear-error stems were vouched for (Align_ULCNet
        # only): the sidecar or corpus ledger that matched the checkpoint, the
        # in-process derivation, or an explicit unstamped override.
        'linear_error_stamp': linear_error_stamp,
        'calibration_far_input_mode': far_mode_provenance(
            args.model_name)[0],
        'deployment_far_input_mode': far_mode_provenance(
            args.model_name)[1],
        'max_delay_frames': alignment_depth(model),
        # The exported graph records the same field; a board that pairs a
        # calibration set with a graph compares the two before trusting the
        # ranges. Only models with a C boundary class are versioned.
        'state_layout_version': state_layout_version,
        'frames': int(next(iter(arrays.values())).shape[0]),
        'seed': args.seed,
        'source_files': source_files,
        'state_precision_policy': policy,
        'contiguous_state_frames': (
            int(next(iter(arrays.values())).shape[0])
            if requires_contiguous_calibration(args.model_name) else None
        ),
        'inputs': input_range_report(arrays, policy),
        **frontend_stamp,
    }
    write_calibration_artifact(
        args.output, arrays, report, artifact_format
    )
    print('%s: %d streaming frames, graph %s' %
          (args.output, report['frames'], onnx_path))
