"""DeepFilterNet ``inference.py calib --select`` / ``inference_batch.py calib
--select``: the capture writes the frames ``PTQ/calib_analyze.py analyze``
would choose from the whole capture; ``--jobs`` changes nothing."""

import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pytest

import dfn_pairs

pytest.importorskip('soundfile')
pytest.importorskip('onnxruntime')

_PTQ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'PTQ')
if _PTQ not in sys.path:
    sys.path.insert(0, _PTQ)

import calib_analyze  # noqa: E402


@pytest.fixture(scope='module', params=dfn_pairs.FAMILIES)
def work(request, tmp_path_factory):
    import soundfile

    family = request.param
    root = tmp_path_factory.mktemp(family)
    config, checkpoint, wavs = dfn_pairs.write_small_model(
        root, family, dfn_pairs.shipped_pair(family), seconds=1.0)
    rate = soundfile.info(str(next(wavs.glob('*.wav')))).samplerate
    rng = np.random.RandomState(9)
    for k in range(2):
        soundfile.write(str(wavs / ('extra%d.wav' % k)),
                        (0.05 * rng.randn(int(rate * (0.6 + 0.2 * k))))
                        .astype('f'), rate)
    return family, root, ['--config', str(config), '--model', str(checkpoint),
                          '--wav-dir', str(wavs)]


def _run(family, module, argv):
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tool = dfn_pairs.import_family_module(family, module)
            tool.calibration_main() if module == 'inference' else tool.cli()
    finally:
        sys.argv = old


def _analyzed(root, whole, count, name):
    with contextlib.redirect_stdout(io.StringIO()):
        calib_analyze.main(['analyze', str(whole), '--count', str(count),
                            '--out', str(root / (name + '.html')),
                            '--output', str(root / (name + '.npz'))])
    return calib_analyze.load_artifact(str(root / (name + '.npz')))


def _same(a, b):
    return a.arrays.keys() == b.arrays.keys() and all(
        np.array_equal(a.arrays[name], b.arrays[name]) for name in a.arrays)


def test_select_writes_what_analyze_chooses_and_jobs_change_nothing(work):
    family, root, base = work
    _run(family, 'inference', ['inference.py', *base, '--select', '25',
                               '--output', str(root / 'sel.npz')])
    _run(family, 'inference', ['inference.py', *base, '--select', '25',
                               '--jobs', '2', '--output', str(root / 'two.npz')])
    _run(family, 'inference', ['inference.py', *base, '--frames', '100000',
                               '--output', str(root / 'all.npz')])
    selected = calib_analyze.load_artifact(str(root / 'sel.npz'))
    whole = calib_analyze.load_artifact(str(root / 'all.npz'))
    assert selected.frames == 25 < whole.frames
    assert _same(selected, _analyzed(root, root / 'all.npz', 25, 'analyzed'))
    assert _same(selected, calib_analyze.load_artifact(str(root / 'two.npz')))
    assert selected.report['frames'] == 25
    assert selected.report['inputs']['erb']['shape'][0] == 25
    assert len(selected.report['source_files']) <= 3
    assert (root / 'sel.report.html').exists()


def test_a_fixed_batch_capture_selects_its_invocations(work):
    family, root, base = work
    batch = ['inference_batch.py', 'calib', *base, '--batch-size', '2']
    _run(family, 'inference_batch',
         batch + ['--select', '6', '--output', str(root / 'bsel.npz')])
    _run(family, 'inference_batch',
         batch + ['--batches', '100000', '--output', str(root / 'ball.npz')])
    selected = calib_analyze.load_artifact(str(root / 'bsel.npz'))
    assert selected.frames == 6
    assert _same(selected, _analyzed(root, root / 'ball.npz', 6, 'banalyzed'))
    assert selected.report['batch_invocations'] == 6
    assert selected.report['source_frame_snapshots'] == 12
    # The lanes of one invocation come from adjacent recordings: no --jobs.
    with pytest.raises(SystemExit):
        _run(family, 'inference_batch',
             batch + ['--select', '6', '--jobs', '2',
                      '--output', str(root / 'bjobs.npz')])


def test_combinations_that_cannot_work_are_refused(work, capsys):
    family, root, base = work
    for extra, message in ((['--select', '5', '--frames', '9'],
                            'leave out --frames'),
                           (['--jobs', '2'], 'go with --select')):
        with pytest.raises(SystemExit):
            _run(family, 'inference', ['inference.py', *base, *extra,
                                       '--output', str(root / 'no.npz')])
        assert message in capsys.readouterr().err
    assert not (root / 'no.npz').exists()
