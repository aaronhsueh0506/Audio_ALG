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
from calibration_io import write_calibration_artifact


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


def _write_pair(root, seconds=0.75, seed=5):
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
        with open(os.path.splitext(str(path))[0] + '.json') as stream:
            return json.load(stream)

    return _read(graph), _read(capture)


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


def test_calibration_report_records_both_far_seams(ulcnet_pair):
    """The recorded far and the deployed far are different signals.

    Asserting this on the report dict rather than on ``far_mode_provenance``
    is the point: the helper being right does not put the fields in the file
    a quantizer actually reads.
    """
    graph_report, capture_report = ulcnet_pair
    calibration_mode, deployment_mode = far_mode_provenance('Align_ULCNet')
    assert capture_report['calibration_far_input_mode'] == calibration_mode
    assert capture_report['deployment_far_input_mode'] == deployment_mode
    assert calibration_mode != deployment_mode
    # The graph stamps the seam a board wires up; the calibration set has to
    # name the same one.
    assert capture_report['deployment_far_input_mode'] == graph_report[
        'far_input_mode'
    ]


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
        'linear_error_ri': np.random.RandomState(3).randn(
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

    plain = report['linear_error_ri']
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
    assert manifest['binary_frame_index_base'] == 1
    assert manifest['binary_byte_order'] == 'little'
    for name, value in arrays.items():
        first = output / name / ('%s_1.bin' % name)
        last = output / name / ('%s_8.bin' % name)
        assert first.is_file() and last.is_file()
        loaded = np.fromfile(first, dtype=value.dtype.newbyteorder('<'))
        assert np.array_equal(loaded.reshape(value.shape[1:]), value[0])
    with pytest.raises(FileExistsError):
        write_calibration_artifact(output, arrays, {'frames': 8}, 'bin')
