import json
from pathlib import Path

import numpy as np
import pytest
import torch

from calibration_io import (
    calibration_report_path,
    capture_calibration_inputs,
    sibling_onnx_path,
    write_calibration_artifact,
)


def test_capture_preserves_complete_onnx_input_shape(tmp_path):
    captured = {}
    graph_input = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    capture_calibration_inputs(captured, ('feature',), (graph_input,))
    capture_calibration_inputs(captured, ('feature',), (graph_input + 1,))

    arrays = {'feature': np.stack(captured['feature'])}
    assert arrays['feature'].shape == (2, 1, 3, 4)

    output = tmp_path / 'calib'
    write_calibration_artifact(
        output, arrays, {'frames': 2}, 'bin'
    )
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['binary_tensors']['feature']['frame_shape'] == [1, 3, 4]
    first = np.fromfile(output / 'feature' / 'feature_0000.bin', '<f4')
    assert np.array_equal(first.reshape(1, 3, 4), graph_input.numpy())


def test_capture_rejects_nonfinite_before_partial_append():
    captured = {'a': [], 'b': []}
    with pytest.raises(RuntimeError, match='non-finite.*b'):
        capture_calibration_inputs(
            captured, ('a', 'b'),
            (torch.zeros(1, 1), torch.tensor([[float('nan')]])),
        )
    assert captured == {'a': [], 'b': []}


def test_capture_rejects_schema_length_mismatch():
    with pytest.raises(ValueError, match='length mismatch'):
        capture_calibration_inputs({}, ('a',), ())


def test_npz_report_does_not_take_the_graph_sidecar_name(tmp_path):
    """``foo.npz`` pairs with ``foo.onnx``, whose exporter writes ``foo.json``.

    The calibration report must land elsewhere, or the later of the two
    writers silently replaces the other's file.
    """
    output = tmp_path / 'foo.npz'
    graph_sidecar = Path(sibling_onnx_path(output)).with_suffix('.json')
    assert graph_sidecar == tmp_path / 'foo.json'
    graph_sidecar.write_text('{"graph": true}\n')

    arrays = {'feature': np.zeros((2, 1, 3), dtype=np.float32)}
    report_path = write_calibration_artifact(
        output, arrays, {'frames': 2}, 'npz'
    )

    assert report_path == str(tmp_path / 'foo.calib.json')
    assert report_path == calibration_report_path(output, 'npz')
    assert graph_sidecar.read_text() == '{"graph": true}\n'
    assert json.loads((tmp_path / 'foo.calib.json').read_text())['frames'] == 2


def test_bin_report_stays_inside_its_directory(tmp_path):
    output = tmp_path / 'foo'
    arrays = {'feature': np.zeros((1, 1, 3), dtype=np.float32)}
    report_path = write_calibration_artifact(
        output, arrays, {'frames': 1}, 'bin'
    )
    assert report_path == str(output / 'manifest.json')
    assert report_path == calibration_report_path(output, 'bin')


# --- the capture's selection options and its --jobs processes ---------------

import argparse  # noqa: E402

import calibration_io  # noqa: E402


def _parser(jobs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--frames', type=int, default=None)
    calibration_io.add_selection_arguments(parser, jobs)
    return parser


def test_selection_options_fall_back_and_refuse_what_cannot_work(capsys):
    parser = _parser()
    args = parser.parse_args(['--output', 'o.npz'])
    calibration_io.check_selection_arguments(parser, args)
    assert args.frames == calibration_io.DEFAULT_FRAMES
    for argv, message in ((['--frames', '0'], '--frames must be positive'),
                          (['--select', '4', '--frames', '9'],
                           'leave out --frames'),
                          (['--jobs', '2'], 'go with --select'),
                          (['--select', '0'], 'at least 1')):
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit):
            calibration_io.check_selection_arguments(parser, args)
        assert message in capsys.readouterr().err
    # A capture that cannot split its recordings takes neither option.
    with pytest.raises(SystemExit):
        _parser(jobs=False).parse_args(['--shard', '0/2:x'])


def test_the_options_a_jobs_process_must_not_inherit_are_removed():
    flags = ('--jobs', '--report', '--output', '--onnx', '--shard',
             '--primary-dir', '--primary-is-mic')
    valued = flags[:-1]
    argv = ['--checkpoint', 'c.pth', '--out', 'x.npz', '--jobs=4',
            '--primary-is', '--primary-d', 'mics', '--rep', 'r.html',
            '--select', '9', '--onnx', 'g.onnx']
    assert calibration_io._without(argv, flags, valued) == [
        '--checkpoint', 'c.pth', '--select', '9']


def test_the_report_goes_beside_the_output():
    assert calibration_io._report_path('a/sel.npz', None) == \
        'a/sel.report.html'
    assert calibration_io._report_path('a/sel_bin/', None) == \
        'a/sel_bin.report.html'
    assert calibration_io._report_path('a/x', 'b/page.html') == 'b/page.html'


def test_a_jobs_process_whose_share_gave_no_frame_hands_back_nothing(
        tmp_path):
    args = _parser().parse_args(['--output', str(tmp_path / 'o.npz'),
                                 '--select', '5', '--shard',
                                 '1/2:%s' % (tmp_path / 'part')])
    frames = calibration_io.CalibrationFrames(args, ['x'], None)
    assert list(frames.sources(['a.wav', 'b.wav', 'c.wav'])) == ['b.wav']
    assert not frames.empty()          # the merged run is checked instead
    with pytest.raises(SystemExit) as done:
        frames.write(str(tmp_path / 'o.npz'), 'npz', lambda arrays: {})
    assert done.value.code == 0
    assert not (tmp_path / 'part').exists()
    whole = calibration_io.CalibrationFrames(
        _parser().parse_args(['--output', 'o.npz', '--select', '5']), ['x'],
        None)
    assert whole.empty()
