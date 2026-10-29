"""What the calibration recorder writes down about the graph it recorded.

The calibration set and the exported graph are shipped as a pair, and every
field here exists so a consumer can refuse a mismatched pair instead of
quantizing against ranges measured on a different D or a different state
layout. These tests run the two tools and compare their JSON side by side.
"""

import contextlib
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC.aiaec_common import SignalGrid
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC._streaming_calibration import (
    far_mode_provenance,
    input_range_report,
)
from AIAEC._streaming_export import state_precision_policy
from calibration_io import (
    calibration_report_path,
    sibling_onnx_path,
    write_calibration_artifact,
)


GRID = SignalGrid(16000, 512, 512, 256)
DEPTH = 3


def _write_ulcnet_checkpoint(path, model):
    from AIAEC.dataset_gen import make_linear_aec_contract

    linear = make_linear_aec_contract(GRID.sample_rate)
    contract = {
        'model_name': 'Align_ULCNet',
        'task': AlignULCNet.task,
        'sr': GRID.sample_rate, 'n_fft': GRID.n_fft,
        'win_len': GRID.win_len, 'hop_len': GRID.hop_len,
        'loss_version': 'test',
        'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        'ctor_max_delay_frames': DEPTH,
    }
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)


def _write_stamp(directory, contract=None):
    """A ``linear_error.contract.json`` for ``directory``, written by hand.

    Spelled out here rather than through the recorder's writer, so the gate is
    tested against the documented file and not against its own producer.
    ``contract`` may be a retired or foreign ``LinearAecContract``.
    """
    from AIAEC.dataset_gen import make_linear_aec_contract

    contract = contract or make_linear_aec_contract(GRID.sample_rate)
    sidecar = {
        'schema': 'aiaec-linear-error-stems-v1',
        'linear_aec_contract_version': contract.version,
        'aec_behavior_hash': contract.aec_behavior_hash,
        'linear_aec_contract_hash': contract.fingerprint(),
        'error_input_seam': 'analysis_formed',
        'sample_rate': contract.sample_rate,
        'linear_aec': contract.as_dict(),
    }
    (directory / 'linear_error.contract.json').write_text(json.dumps(sidecar))


def _write_pair(root, seconds=0.75, seed=5, stamp=True):
    generator = torch.Generator().manual_seed(seed)
    samples = int(seconds * GRID.sample_rate)
    primary_dir = root / 'primary'
    far_dir = root / 'far'
    primary_dir.mkdir()
    far_dir.mkdir()
    wave = 0.2 * torch.randn(samples, generator=generator)
    sf.write(primary_dir / 'a.wav', wave.numpy(),
             GRID.sample_rate, subtype='FLOAT')
    sf.write(far_dir / 'a.wav', wave.roll(64).numpy(),
             GRID.sample_rate, subtype='FLOAT')
    if stamp:
        _write_stamp(primary_dir)
    return primary_dir, far_dir


@contextlib.contextmanager
def _argv(*arguments):
    saved = sys.argv
    sys.argv = list(arguments)
    try:
        yield
    finally:
        sys.argv = saved


@pytest.fixture(scope='module')
def ulcnet_pair(tmp_path_factory):
    """Export a tiny ULCNet graph and record calibration against it."""
    pytest.importorskip('onnx')
    from AIAEC.Align_ULCNet import export_onnx as ulcnet_export
    from AIAEC import _streaming_calibration as calibration

    work = tmp_path_factory.mktemp('ulcnet_dchain')
    torch.manual_seed(59)
    model = AlignULCNet(GRID, max_delay_frames=DEPTH).eval()
    checkpoint = work / 'ckpt.pth'
    _write_ulcnet_checkpoint(str(checkpoint), model)
    primary_dir, far_dir = _write_pair(work)

    graph = work / 'model.onnx'
    with _argv('export_onnx.py',
               '--checkpoint', str(checkpoint),
               '--output', str(graph)):
        ulcnet_export.main()

    capture = work / 'calibration.npz'
    # The public entry is ``inference.py calib``, which names its own model;
    # drive that fixed-name path rather than a positional model argument.
    with _argv('inference.py',
               '--checkpoint', str(checkpoint),
               '--primary-dir', str(primary_dir),
               '--far-dir', str(far_dir),
               '--output', str(capture),
               '--frames', '8'):
        calibration.main('Align_ULCNet')

    def _read(path):
        with open(path) as stream:
            return json.load(stream)

    # The run exported its own graph next to the archive; that graph's
    # sidecar must have survived beside the calibration report.
    own_graph = _read(os.path.splitext(str(capture))[0] + '.json')
    assert 'input_schema' in own_graph and 'seed' not in own_graph
    return (_read(os.path.splitext(str(graph))[0] + '.json'),
            _read(calibration_report_path(capture, 'npz')))


def test_calibration_and_graph_agree_on_the_d_chain(ulcnet_pair):
    """D and the state layout must match, or the ranges describe another graph.

    The alignment depth changes the state tensor shapes AND the numerical
    output, so a calibration set recorded at a different D is not merely
    stale -- it is measured on a different function.
    """
    graph_report, capture_report = ulcnet_pair
    assert graph_report['max_delay_frames'] == DEPTH
    assert capture_report['max_delay_frames'] == graph_report[
        'max_delay_frames'
    ]
    assert capture_report['state_layout_version'] == graph_report[
        'state_layout_version'
    ]


def test_calibration_records_the_full_window_inputs(ulcnet_pair):
    """The recorded state tensors are the graph's full windows, not the
    history one frame short of them."""
    graph_report, capture_report = ulcnet_pair
    assert capture_report['state_layout_version'] == 18
    schema = graph_report['input_schema']
    assert schema['key_history'][2] == DEPTH
    assert schema['logit_history'][2] == 5
    assert schema['h_gru'] == [1, 4, 1, 128]
    for name, shape in schema.items():
        assert capture_report['inputs'][name]['shape'][1:] == shape, name


def test_calibration_report_records_both_far_seams(ulcnet_pair):
    """The report records both provenance fields even when both are raw far.

    Asserting this on the report dict rather than on ``far_mode_provenance``
    is the point: the helper being right does not put the fields in the file
    a quantizer actually reads.
    """
    graph_report, capture_report = ulcnet_pair
    calibration_mode, deployment_mode = far_mode_provenance('Align_ULCNet')
    assert capture_report['calibration_far_input_mode'] == calibration_mode
    assert capture_report['deployment_far_input_mode'] == deployment_mode
    assert calibration_mode == deployment_mode == 'raw_far'
    # The graph stamps the seam a board wires up; the calibration set has to
    # name the same one.
    assert capture_report['deployment_far_input_mode'] == graph_report[
        'far_input_mode'
    ]


@pytest.mark.parametrize('artifact_format', ['npz', 'bin'])
def test_calibration_report_stamps_the_checkpoint_front_end(
        tmp_path, artifact_format):
    """The report names the linear-AEC front end of the error tensors.

    The ranges describe a linear error produced by one specific front end; a
    report without that identity cannot be told apart from one recorded on a
    retired corpus. The stamp must equal the checkpoint's own contract, in the
    NPZ sidecar and in the binary manifest alike.
    """
    from AIAEC import _streaming_calibration as calibration

    torch.manual_seed(59)
    model = AlignULCNet(GRID, max_delay_frames=DEPTH).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    _write_ulcnet_checkpoint(str(checkpoint), model)
    primary_dir, far_dir = _write_pair(tmp_path)
    capture = tmp_path / ('stamp.npz' if artifact_format == 'npz' else 'stamp')
    with _argv('inference.py',
               '--checkpoint', str(checkpoint),
               '--primary-dir', str(primary_dir),
               '--far-dir', str(far_dir),
               '--output', str(capture),
               '--frames', '8'):
        calibration.main('Align_ULCNet')

    with open(calibration_report_path(capture, artifact_format)) as stream:
        report = json.load(stream)
    contract = torch.load(str(checkpoint), weights_only=False)['contract']
    linear = contract['linear_aec']
    assert report['linear_aec_contract_version'] == linear['version']
    assert report['aec_behavior_hash'] == linear['aec_behavior_hash']
    assert report['linear_aec_contract_hash'] == contract[
        'linear_aec_contract_hash']
    assert report['error_input_seam'] == 'analysis_formed'
    # Not vacuous: the identity is the live build's, not an empty string.
    assert report['linear_aec_contract_version'].startswith('aiaec-linear-error-')
    assert len(report['aec_behavior_hash']) == 64
    # The graph written beside it is the one the stamp is paired with.
    with open(os.path.splitext(sibling_onnx_path(capture))[0]
              + '.json') as stream:
        graph = json.load(stream)
    for key in ('linear_aec_contract_version', 'aec_behavior_hash',
                'error_input_seam'):
        assert report[key] == graph[key], key


def test_unrestricted_tensors_keep_their_recorded_ranges(ulcnet_pair):
    """Align-ULCNet has no excluded tensor, so every entry carries a range."""
    _graph_report, capture_report = ulcnet_pair
    assert capture_report['state_precision_policy'] == {}
    assert capture_report['inputs']
    required = {'shape', 'dtype', 'min', 'max', 'p001', 'p999'}
    for name, entry in capture_report['inputs'].items():
        assert required <= set(entry), name
        assert 'precision' not in entry, name


def test_policy_tensors_carry_a_marker_instead_of_numeric_ranges(tmp_path):
    """Excluded tensors must lose min/max/p001/p999 entirely.

    Leaving them in would hand a quantizer a range for the one tensor whose
    range is meaningless -- and computing float percentiles over the int64
    frame counter produces a number that looks usable and is not.
    """
    policy = state_precision_policy('Align_CRUSE')
    arrays = {
        'state_align_score_sum': np.full((8, 1), 3.5, dtype=np.float32),
        'state_align_frame_index': np.arange(8, dtype=np.int64),
        'error': np.random.RandomState(3).randn(
            8, 1, 257, 2
        ).astype(np.float32),
    }
    report = input_range_report(arrays, policy)

    for name in policy:
        entry = report[name]
        assert entry['precision'] == policy[name]
        assert not {'min', 'max', 'p001', 'p999'} & set(entry), name
        # The shape/dtype record stays: a consumer still has to allocate it.
        assert entry['shape'] and entry['dtype']

    plain = report['error']
    assert 'precision' not in plain
    assert plain['min'] <= plain['p001'] <= plain['p999'] <= plain['max']

    # With an empty policy the same tensors DO get ranges, so the absence
    # above is caused by the policy and not by the tensor names or dtypes.
    unrestricted = input_range_report(arrays, {})
    for name in policy:
        assert 'min' in unrestricted[name], name

    # The binary form preserves dtype and splits every invocation into the
    # exact per-input file layout consumed by the accelerator calibrator.
    output = tmp_path / 'calibration_bin'
    write_calibration_artifact(output, arrays, {
        'schema': 'test',
        'frames': 8,
        'inputs': report,
    }, 'bin')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['binary_frame_index_base'] == 0
    assert manifest['binary_byte_order'] == 'little'
    for name, value in arrays.items():
        first = output / name / ('%s_0000.bin' % name)
        last = output / name / ('%s_0007.bin' % name)
        assert first.is_file() and last.is_file()
        loaded = np.fromfile(first, dtype=value.dtype.newbyteorder('<'))
        assert np.array_equal(loaded.reshape(value.shape[1:]), value[0])
    with pytest.raises(FileExistsError):
        write_calibration_artifact(output, arrays, {'frames': 8}, 'bin')


def test_primary_is_mic_derives_and_persists_linear_error(tmp_path):
    """--primary-is-mic runs the frozen PBFDKF on RAW mic pairs.

    The derived linear-error stems must be written beside the artifact
    (recorded tensors stay auditable and reusable) and the manifest must say
    where its primary came from, so a consumer can tell derived stems from
    materialized ones. The derived stem must actually BE the engine's error: on
    this echo-only scene it carries less energy than the raw microphone.
    """
    from AIAEC import _streaming_calibration as calibration
    from AIAEC.Align_ULCNet import export_onnx as ulcnet_export  # noqa: F401

    work = tmp_path
    torch.manual_seed(59)
    model = AlignULCNet(GRID, max_delay_frames=DEPTH).eval()
    checkpoint = work / 'ckpt.pth'
    _write_ulcnet_checkpoint(str(checkpoint), model)
    # A genuinely cancellable pair: the microphone IS a delayed, attenuated
    # copy of far, so the frozen PBFDKF must remove energy and the derived
    # stem is distinguishable from a pass-through by RMS alone.
    generator = torch.Generator().manual_seed(7)
    samples = int(1.5 * GRID.sample_rate)
    primary_dir = work / 'mic'
    far_dir = work / 'farraw'
    primary_dir.mkdir()
    far_dir.mkdir()
    far_wave = 0.25 * torch.randn(samples, generator=generator)
    mic_wave = 0.3 * torch.roll(far_wave, 64)
    sf.write(primary_dir / 'a.wav', mic_wave.numpy(), GRID.sample_rate,
             subtype='FLOAT')
    sf.write(far_dir / 'a.wav', far_wave.numpy(), GRID.sample_rate,
             subtype='FLOAT')

    capture = work / 'rawmic_calibration'
    with _argv('inference.py',
               '--checkpoint', str(checkpoint),
               '--primary-dir', str(primary_dir),
               '--far-dir', str(far_dir),
               '--primary-is-mic',
               '--output', str(capture),
               '--frames', '8'):
        calibration.main('Align_ULCNet')

    manifest = json.loads((capture / 'manifest.json').read_text())
    assert manifest['primary_source'] == 'raw_mic_via_frozen_pbfdkf'
    assert manifest['derived_linear_error_dir'] == 'rawmic_calibration_linear_error'
    derived = work / 'rawmic_calibration_linear_error' / 'a.wav'
    assert derived.is_file()
    error, sr = sf.read(derived)
    mic, _ = sf.read(primary_dir / 'a.wav')
    assert sr == GRID.sample_rate
    assert len(error) == len(mic)
    import numpy as np
    assert float(np.sqrt((error ** 2).mean())) < float(
        np.sqrt((mic ** 2).mean()))

    # The derived directory describes itself: its sidecar is the checkpoint's
    # own contract, so the stems can be reused through --primary-dir and
    # checked against a checkpoint without trusting the directory name.
    from AIAEC.dataset_gen.linear_aec import LinearAecContract
    from AIAEC.Align_ULCNet.export_onnx import ERROR_INPUT_SEAM

    sidecar = json.loads((work / 'rawmic_calibration_linear_error'
                          / 'linear_error.contract.json').read_text())
    recorded = torch.load(str(checkpoint), map_location='cpu',
                          weights_only=False)['contract']
    assert sidecar['linear_aec'] == recorded['linear_aec']
    assert (sidecar['linear_aec_contract_hash']
            == recorded['linear_aec_contract_hash']
            == LinearAecContract.from_dict(sidecar['linear_aec']).fingerprint())
    assert sidecar['linear_aec_contract_version'] == recorded['linear_aec']['version']
    assert sidecar['aec_behavior_hash'] == recorded['linear_aec']['aec_behavior_hash']
    assert sidecar['error_input_seam'] == ERROR_INPUT_SEAM
    assert sidecar['sample_rate'] == GRID.sample_rate

    # Reuse: the derived directory is itself a valid --primary-dir for a
    # second run (this is the path the provenance gate must keep open).
    reuse = work / 'reuse_calibration'
    with _argv('inference.py',
               '--checkpoint', str(checkpoint),
               '--primary-dir', str(work / 'rawmic_calibration_linear_error'),
               '--far-dir', str(far_dir),
               '--output', str(reuse),
               '--frames', '8'):
        calibration.main('Align_ULCNet')
    second = json.loads((reuse / 'manifest.json').read_text())
    assert second['primary_source'] == 'materialized_primary_wavs'
    assert second['linear_aec_contract_hash'] == sidecar['linear_aec_contract_hash']


# ---- provenance of pre-made linear-error stems ------------------------------

RETIRED_HASH = (
    '9fa7a2dec85cef8f123cc4dd02e7eb3c5dc0c1b45d7ec0f32b766cf9cda245e5')


def _calibrate_primary(tmp_path, primary_dir, far_dir, *extra):
    """Run ``inference.py calib`` on pre-made stems; return the output dir."""
    from AIAEC import _streaming_calibration as calibration

    torch.manual_seed(59)
    model = AlignULCNet(GRID, max_delay_frames=DEPTH).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    _write_ulcnet_checkpoint(str(checkpoint), model)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with _argv('inference.py',
               '--checkpoint', str(checkpoint),
               '--primary-dir', str(primary_dir),
               '--far-dir', str(far_dir),
               '--output', str(out_dir / 'cal.npz'),
               '--frames', '8', *extra):
        calibration.main('Align_ULCNet')
    return out_dir


def _foreign_contracts():
    """Stamps the checkpoint (current build) must not accept, with the text
    each refusal has to carry."""
    import dataclasses

    from AIAEC.dataset_gen import make_linear_aec_contract

    current = make_linear_aec_contract(GRID.sample_rate)
    return {
        'v3_version': (dataclasses.replace(
            current, version='aiaec-linear-error-v3',
            aec_behavior_hash=RETIRED_HASH), 'retired'),
        # Current version string, retired behaviour: from_dict alone accepts it.
        'retired_hash': (dataclasses.replace(
            current, aec_behavior_hash=RETIRED_HASH), 'retired'),
        'foreign_hash': (dataclasses.replace(
            current, aec_behavior_hash='ab' * 32), 'mismatch'),
        'other_filter': (dataclasses.replace(
            current, filter_length=current.filter_length * 2), 'mismatch'),
    }


@pytest.mark.parametrize('case', ['v3_version', 'retired_hash',
                                  'foreign_hash', 'other_filter'])
def test_calibration_refuses_a_foreign_or_retired_stamp(tmp_path, case):
    """Stems stamped with another front end never reach the capture.

    The checkpoint here is the current build's; every stamp below differs from
    it, two of them by being retired outright. Refused before the graph export,
    so the output directory stays empty.
    """
    pytest.importorskip('onnx')
    contract, message = _foreign_contracts()[case]
    primary_dir, far_dir = _write_pair(tmp_path, stamp=False)
    _write_stamp(primary_dir, contract)
    out_dir = tmp_path / 'out'
    with pytest.raises(ValueError, match=message):
        _calibrate_primary(tmp_path, primary_dir, far_dir)
    assert list(out_dir.iterdir()) == []


def test_calibration_records_where_matching_stems_were_stamped(tmp_path):
    """A matching stamp passes and the report names its origin."""
    pytest.importorskip('onnx')
    primary_dir, far_dir = _write_pair(tmp_path)
    out_dir = _calibrate_primary(tmp_path, primary_dir, far_dir)
    with open(calibration_report_path(out_dir / 'cal.npz', 'npz')) as stream:
        report = json.load(stream)
    assert report['primary_source'] == 'materialized_primary_wavs'
    stamp = report['linear_error_stamp']
    assert stamp['origin'] == 'sidecar'
    assert stamp['path'] == str(primary_dir.resolve()
                                / 'linear_error.contract.json')
    assert stamp['linear_aec_contract_version'] == report[
        'linear_aec_contract_version']
    assert stamp['aec_behavior_hash'] == report['aec_behavior_hash']


def test_calibration_refuses_unstamped_stems_unless_overridden(tmp_path):
    """No stamp, no calibration; the override is explicit and on the record."""
    pytest.importorskip('onnx')
    primary_dir, far_dir = _write_pair(tmp_path, stamp=False)
    out_dir = tmp_path / 'out'
    with pytest.raises(ValueError, match='no front-end stamp'):
        _calibrate_primary(tmp_path, primary_dir, far_dir)
    assert list(out_dir.iterdir()) == []

    out_dir.rmdir()
    with pytest.warns(RuntimeWarning, match='unstamped'):
        _calibrate_primary(tmp_path, primary_dir, far_dir,
                           '--allow-unstamped-linear-error')
    with open(calibration_report_path(out_dir / 'cal.npz', 'npz')) as stream:
        assert json.load(stream)['linear_error_stamp'] == {
            'origin': 'unstamped_override', 'path': None}


def test_corpus_ledger_stamps_stems_and_refuses_partial_claims(tmp_path):
    """The ledger beside a corpus vouches for it only if it names the contract
    AND every chunk sequence present; the sidecar needs no ledger."""
    import dataclasses

    from AIAEC._streaming_calibration import linear_error_provenance
    from AIAEC.dataset_gen import make_linear_aec_contract
    from AIAEC.dataset_gen.linear_error_ledger import save_ledger

    current = make_linear_aec_contract(GRID.sample_rate)
    seqs = tmp_path / 'seqs'
    seqs.mkdir()
    for sequence in (0, 1):
        (seqs / ('%06d_000.wav' % sequence)).write_bytes(b'')

    with pytest.raises(ValueError, match='no front-end stamp'):
        linear_error_provenance(seqs, current.as_dict())

    save_ledger(str(seqs), current.fingerprint(), {0}, current.as_dict())
    with pytest.raises(ValueError, match='part new front end'):
        linear_error_provenance(seqs, current.as_dict())

    save_ledger(str(seqs), current.fingerprint(), {0, 1}, current.as_dict())
    stamp = linear_error_provenance(seqs, current.as_dict())
    assert stamp['origin'] == 'corpus_ledger'
    assert stamp['aec_behavior_hash'] == current.aec_behavior_hash

    old = dataclasses.replace(current, aec_behavior_hash=RETIRED_HASH)
    save_ledger(str(seqs), old.fingerprint(), {0, 1}, old.as_dict())
    with pytest.raises(ValueError, match='retired'):
        linear_error_provenance(seqs, current.as_dict())
    # Agreeing with a retired checkpoint is no excuse either.
    with pytest.raises(ValueError, match='retired'):
        linear_error_provenance(seqs, old.as_dict())
