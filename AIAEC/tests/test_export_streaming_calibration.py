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
from AIAEC.export_streaming_calibration import (
    far_mode_provenance,
    input_range_report,
)
from AIAEC.export_streaming_onnx import state_precision_policy


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
    from AIAEC.Align_ULCNet import export_streaming_onnx as ulcnet_export
    from AIAEC import export_streaming_calibration as calibration

    work = tmp_path_factory.mktemp('ulcnet_dchain')
    torch.manual_seed(59)
    model = AlignULCNet(GRID, max_delay_frames=DEPTH).eval()
    checkpoint = work / 'ckpt.pth'
    _write_ulcnet_checkpoint(str(checkpoint), model)
    primary_dir, far_dir = _write_pair(work)

    graph = work / 'model.onnx'
    with _argv('export_streaming_onnx.py',
               '--checkpoint', str(checkpoint),
               '--output', str(graph)):
        ulcnet_export.main()

    capture = work / 'calibration.npz'
    with _argv('export_streaming_calibration.py', 'Align_ULCNet',
               '--checkpoint', str(checkpoint),
               '--primary-dir', str(primary_dir),
               '--far-dir', str(far_dir),
               '--output', str(capture),
               '--frames', '8'):
        calibration.main()

    # The legacy fixed-block recorder reads the same two directories, so it
    # has to describe the far seam the same way.
    from AIAEC import export_calibration as legacy_calibration

    legacy = work / 'legacy.npz'
    with _argv('export_calibration.py', 'Align_ULCNet',
               '--checkpoint', str(checkpoint),
               '--primary-dir', str(primary_dir),
               '--far-dir', str(far_dir),
               '--output', str(legacy),
               '--frames', '4', '--blocks', '2'):
        legacy_calibration.main()

    def _read(path):
        with open(os.path.splitext(str(path))[0] + '.json') as stream:
            return json.load(stream)

    return _read(graph), _read(capture), _read(legacy)


def test_calibration_and_graph_agree_on_the_d_chain(ulcnet_pair):
    """D and the state layout must match, or the ranges describe another graph.

    The alignment depth changes the state tensor shapes AND the numerical
    output, so a calibration set recorded at a different D is not merely
    stale -- it is measured on a different function.
    """
    graph_report, capture_report, _legacy = ulcnet_pair
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
    graph_report, capture_report, _legacy = ulcnet_pair
    calibration_mode, deployment_mode = far_mode_provenance('Align_ULCNet')
    assert capture_report['calibration_far_input_mode'] == calibration_mode
    assert capture_report['deployment_far_input_mode'] == deployment_mode
    assert calibration_mode != deployment_mode
    # The graph stamps the seam a board wires up; the calibration set has to
    # name the same one.
    assert capture_report['deployment_far_input_mode'] == graph_report[
        'far_input_mode'
    ]


def test_legacy_fixed_block_report_records_the_far_seams_too(ulcnet_pair):
    """A fixed-block set is still a set someone can hand to a quantizer.

    It was the only recorder that said nothing about which far it measured,
    so a stale npz was indistinguishable from one recorded on the deployed
    seam.
    """
    _graph_report, capture_report, legacy_report = ulcnet_pair
    assert legacy_report['calibration_far_input_mode'] == capture_report[
        'calibration_far_input_mode'
    ]
    assert legacy_report['deployment_far_input_mode'] == capture_report[
        'deployment_far_input_mode'
    ]


def test_unrestricted_tensors_keep_their_recorded_ranges(ulcnet_pair):
    """Align-ULCNet has no excluded tensor, so every entry carries a range."""
    _graph_report, capture_report, _legacy = ulcnet_pair
    assert capture_report['state_precision_policy'] == {}
    assert capture_report['inputs']
    required = {'shape', 'dtype', 'min', 'max', 'p001', 'p999'}
    for name, entry in capture_report['inputs'].items():
        assert required <= set(entry), name
        assert 'precision' not in entry, name


def test_dfn_windows_are_the_exporters_own_zero_padded_slide():
    """The recorder must feed the graph the window the exporter defines.

    It captures one frame at a time and reassembles the stream, so this is the
    one place the shared slide could be applied to a differently shaped
    buffer and still look plausible.
    """
    from AIAEC.export_streaming_calibration import _dfn_windows

    blocks = [
        {'error_erb': np.full((1, 1, 4), float(index + 1), dtype=np.float32)}
        for index in range(3)
    ]
    windows = list(_dfn_windows(blocks, 'error_erb'))
    assert len(windows) == len(blocks)
    assert all(tuple(window.shape) == (1, 1, 3, 4) for window in windows)
    # Frame 0 sees [pad, f0, f1]; the final frame sees [f1, f2, pad].
    assert [float(windows[0][0, 0, index, 0]) for index in range(3)] == [
        0.0, 1.0, 2.0
    ]
    assert [float(windows[-1][0, 0, index, 0]) for index in range(3)] == [
        2.0, 3.0, 0.0
    ]


def test_policy_tensors_carry_a_marker_instead_of_numeric_ranges():
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
