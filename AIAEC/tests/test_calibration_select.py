"""``inference.py calib --select``: the capture scans every recording and
writes the frames ``PTQ/calib_analyze.py analyze`` would choose from the whole
capture; ``--jobs`` splits the recordings between processes and changes
nothing."""

import contextlib
import io
import warnings

import numpy as np
import pytest
import soundfile as sf
import torch

import calib_analyze
from AIAEC.DeepVQE_S import DeepVQES
from test_export_streaming_calibration import LAYOUT_GRID, _argv

COUNT = 120     # above the frames every tensor's extremes need, below the capture


@pytest.fixture(scope='module')
def work(tmp_path_factory):
    pytest.importorskip('onnxruntime')
    root = tmp_path_factory.mktemp('select')
    grid = LAYOUT_GRID
    torch.manual_seed(3)
    model = DeepVQES(grid, max_delay_seconds=0.5).eval()
    torch.save({'contract': {
        'model_name': 'DeepVQE_S', 'task': DeepVQES.task, 'sr': grid.sample_rate,
        'n_fft': grid.n_fft, 'win_len': grid.n_fft, 'hop_len': grid.hop_len,
        'loss_version': 'test', 'ctor_max_delay_seconds': 0.5,
        'ctor_similarity_channels': 4, 'ctor_gru_hidden': 192,
        'ctor_compression_exponent': 0.3,
    }, 'state_dict': model.state_dict()}, root / 'dv.pth')
    rng = np.random.RandomState(4)
    for name in ('primary', 'far'):
        (root / name).mkdir()
        for k in range(3):
            sf.write(root / name / ('w%d.wav' % k),
                     0.1 * rng.randn(grid.sample_rate + 300 * k).astype('f'),
                     grid.sample_rate)
    return root


def _calib(root, extra, name, model='DeepVQE_S'):
    from AIAEC import _streaming_calibration as calibration

    output = root / name
    with _argv('calib', '--checkpoint', str(root / 'dv.pth'),
               '--primary-dir', str(root / 'primary'),
               '--far-dir', str(root / 'far'), '--output', str(output),
               *extra), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter('ignore')
        calibration.main(model)
    return calib_analyze.load_artifact(str(output))


def _same(a, b):
    return a.arrays.keys() == b.arrays.keys() and all(
        np.array_equal(a.arrays[name], b.arrays[name]) for name in a.arrays)


def test_select_writes_what_analyze_chooses_from_the_whole_capture(work):
    selected = _calib(work, ['--select', str(COUNT)], 'sel.npz')
    whole = _calib(work, ['--frames', '100000'], 'all.npz')
    with contextlib.redirect_stdout(io.StringIO()):
        calib_analyze.main(['analyze', whole.path, '--count', str(COUNT),
                            '--out', str(work / 'all.html'),
                            '--output', str(work / 'analyzed.npz')])
    analyzed = calib_analyze.load_artifact(str(work / 'analyzed.npz'))
    assert selected.frames == COUNT < whole.frames
    assert _same(selected, analyzed)
    assert selected.report['selection']['parent_frames'] == whole.frames
    assert selected.report['selection']['parent'] == (
        '3 recordings scanned by the capture')
    assert selected.report['model_family'] == 'DeepVQE_S'
    assert 'state_align_key_ring' in selected.report['state_inputs']
    assert set(selected.report['source_files']) <= set(
        whole.report['source_files'])
    for name, value in whole.arrays.items():
        assert selected.arrays[name].min() == value.min()
        assert selected.arrays[name].max() == value.max()
    assert (work / 'sel.report.html').exists()


def test_jobs_split_the_recordings_and_change_nothing(work):
    one = _calib(work, ['--select', str(COUNT)], 'one.npz')
    two = _calib(work, ['--select', str(COUNT), '--jobs', '2',
                        '--report', str(work / 'two.html')], 'two.npz')
    assert _same(one, two)
    # The page names this run's graph, not a process's temporary one.
    page = (work / 'two.html').read_text(encoding='utf-8')
    assert '"graph":"%s"' % (work / 'two.onnx') in page
    assert '.calib-jobs' not in page and 'part0' not in page
    assert '無法確認兩者出自同一個 checkpoint' not in page
    assert not [path for path in work.iterdir()
                if path.name.startswith('.calib-jobs')]
    # A BIN artifact in a directory that does not exist yet (the graph goes
    # elsewhere, so nothing else makes it), its page in another one.
    stored = _calib(work, ['--select', str(COUNT), '--jobs', '2',
                           '--format', 'bin',
                           '--onnx', str(work / 'graphs' / 'bin.onnx'),
                           '--report', str(work / 'pages' / 'bin.html')],
                    'fresh/bin')
    assert stored.fmt == 'bin' and _same(one, stored)
    assert (work / 'pages' / 'bin.html').exists()
    assert not [path for path in (work / 'fresh').iterdir()
                if path.name.startswith('.calib-jobs')]


def test_jobs_over_one_recording_name_it(work):
    for name in ('primary', 'far'):
        (work / (name + '1')).mkdir(exist_ok=True)
        (work / (name + '1') / 'w0.wav').write_bytes(
            (work / name / 'w0.wav').read_bytes())
    one_dir = ['--primary-dir', str(work / 'primary1'),
               '--far-dir', str(work / 'far1'), '--select', '30']
    alone = _calib(work, one_dir, 'alone.npz')
    shared = _calib(work, one_dir + ['--jobs', '2'], 'alone2.npz')
    assert _same(alone, shared)
    assert shared.report['source_files'] == alone.report['source_files']
    assert len(shared.report['source_files']) == 1


def test_jobs_refuse_a_stem_directory_an_earlier_run_left(work, capsys):
    (work / 'stale.npz_linear_error').mkdir()
    with pytest.raises(SystemExit):
        _calib(work, ['--select', '10', '--jobs', '2', '--primary-is-mic'],
               'stale.npz', 'Align_ULCNet')
    assert 'stale.npz_linear_error exists' in capsys.readouterr().err


@pytest.mark.parametrize('extra,model,message', [
    (['--select', '10', '--frames', '5'], 'DeepVQE_S', 'leave out --frames'),
    (['--jobs', '2'], 'DeepVQE_S', 'go with --select'),
    (['--report', 'x.html'], 'DeepVQE_S', 'go with --select'),
    (['--select', '0'], 'DeepVQE_S', 'at least 1'),
    (['--select', '10', '--jobs', '0'], 'DeepVQE_S', 'at least 1'),
    (['--select', '10'], 'Align_CRUSE', 'one uninterrupted stream'),
    (['--select', '10', '--jobs', '2', '--events', 'e.json'], 'Align_ULCNet',
     'leave out --jobs'),
])
def test_combinations_that_cannot_work_are_refused(work, capsys, extra, model,
                                                   message):
    with pytest.raises(SystemExit):
        _calib(work, extra, 'refused.npz', model)
    assert message in capsys.readouterr().err
    assert not (work / 'refused.npz').exists()
