#!/usr/bin/env python3
"""Both calibrators must actually execute, not merely import.

These are standalone scripts with no other caller, so nothing exercised them:
the DeepFilterNet2 one shipped reading ``feat['erb_norm_init_lo_db']`` when its
own ``read_feature_config`` returns that value under ``erb_init_lo_db`` (the
two projects spell the key differently -- config.ini has the ``norm_`` infix,
the parsed dict does not).  ``--help`` passed, so the KeyError only appeared on
a real run.

This drives each script end to end over a tiny synthetic corpus and checks it
reaches its output.  The numbers are meaningless -- white noise, a handful of
clips -- so nothing here asserts values; it asserts that the pipeline runs and
emits the four config lines.
"""

import pathlib
import subprocess
import sys
import tempfile

import torch


AINR = pathlib.Path(__file__).resolve().parents[1]

# sr, project directory, clip seconds.  Kept short: the point is coverage of
# the script body, not statistical quality.
CASES = [
    (16000, 'RNNoise-ERB', 1.0),
    (48000, 'DeepFilterNet2', 1.0),
]

EXPECTED_LINES = (
    'erb_norm_init_lo_db =',
    'erb_norm_init_hi_db =',
    'spec_norm_init_lo =',
    'spec_norm_init_hi =',
)


def _write_packed(directory, sr, seconds, n_clips=6, seed=0):
    """A minimal pack_dataset.py-shaped file: {'data': (N,2,T), 'sr': sr}."""
    torch.manual_seed(seed)
    samples = int(sr * seconds)
    clean = torch.randn(n_clips, 1, samples) * 0.05
    noisy = clean + torch.randn(n_clips, 1, samples) * 0.05
    path = pathlib.Path(directory) / 'packed.pt'
    torch.save({'data': torch.cat([noisy, clean], dim=1), 'sr': sr}, path)
    return directory


def _run_one(sr, project, seconds):
    with tempfile.TemporaryDirectory(prefix=f'calib-{project}-') as tmp:
        _write_packed(tmp, sr, seconds)
        proc = subprocess.run(
            [sys.executable, 'calibrate_norm_init.py',
             '--packed-dir', tmp, '--clips', '4'],
            cwd=AINR / project, capture_output=True, text=True,
        )
    if proc.returncode != 0:
        raise AssertionError(
            f'{project}/calibrate_norm_init.py exited {proc.returncode}\n'
            f'--- stdout ---\n{proc.stdout[-2000:]}\n'
            f'--- stderr ---\n{proc.stderr[-2000:]}')
    for line in EXPECTED_LINES:
        assert line in proc.stdout, (
            f'{project}: missing {line!r} in output\n{proc.stdout[-1500:]}')
    # The training-split restriction is load-bearing (calibrating on held-out
    # clips leaks the validation set into an initialisation constant).
    assert 'training split only' in proc.stdout, project
    return proc.stdout


def test_rnnoise_erb_calibrator_runs():
    _run_one(*CASES[0])


def test_deepfilternet2_calibrator_runs():
    _run_one(*CASES[1])


def test_calibrators_do_not_share_endpoints():
    """The two must be calibrated separately, so they must not agree by accident.

    Different sample rate, FFT size and band count give different band widths
    and therefore a different per-band offset; if these ever came out identical
    it would mean one of them is not measuring its own grid.
    """
    outs = {}
    for sr, project, seconds in CASES:
        text = _run_one(sr, project, seconds)
        line = next(l for l in text.splitlines()
                    if l.startswith('erb_norm_init_lo_db ='))
        outs[project] = float(line.split('=')[1])
    assert outs['RNNoise-ERB'] != outs['DeepFilterNet2'], outs


if __name__ == '__main__':
    tests = [
        test_rnnoise_erb_calibrator_runs,
        test_deepfilternet2_calibrator_runs,
        test_calibrators_do_not_share_endpoints,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
