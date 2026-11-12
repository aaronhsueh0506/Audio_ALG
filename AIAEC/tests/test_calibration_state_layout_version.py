"""The calibration manifest carries the exported graph's state_layout_version.

Both come off the same export in one process; a delivery side matches graph
and calibration set on this field, so a null manifest value is a defect.
"""

import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC import _streaming_calibration as calibration
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.DeepVQE_S import DeepVQES
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import make_linear_aec_contract
from calibration_io import calibration_report_path, sibling_onnx_path

pytest.importorskip('onnx')

SR, N_FFT, HOP = 16000, 256, 128
GRID = SignalGrid(SR, N_FFT, N_FFT, HOP)


@pytest.fixture(scope='module')
def work(tmp_path_factory):
    root = tmp_path_factory.mktemp('layout_version')
    torch.manual_seed(3)
    dv = DeepVQES(GRID, max_delay_seconds=0.5).eval()
    torch.save({'contract': {
        'model_name': 'DeepVQE_S', 'task': DeepVQES.task, 'sr': SR,
        'n_fft': N_FFT, 'win_len': N_FFT, 'hop_len': HOP,
        'loss_version': 'test', 'ctor_max_delay_seconds': 0.5,
        'ctor_similarity_channels': 4, 'ctor_gru_hidden': 192,
        'ctor_compression_exponent': 0.3,
    }, 'state_dict': dv.state_dict()}, root / 'dv.pth')
    al = AlignULCNet(GRID, max_delay_seconds=0.5).eval()
    linear = make_linear_aec_contract(SR, frame_size=N_FFT)
    torch.save({'contract': {
        'model_name': 'Align_ULCNet', 'task': AlignULCNet.task, 'sr': SR,
        'n_fft': N_FFT, 'win_len': N_FFT, 'hop_len': HOP,
        'loss_version': 'test', 'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        'ctor_max_delay_seconds': 0.5,
    }, 'state_dict': al.state_dict()}, root / 'al.pth')
    rng = np.random.RandomState(4)
    for name in ('primary', 'far'):
        (root / name).mkdir()
        sf.write(root / name / 'a.wav',
                 0.1 * rng.randn(SR * 3 + 10).astype('f'), SR)
    return root


def _calib(work, model_name, checkpoint, n, fmt):
    output = work / ('%s_n%d' % (model_name, n) + ('.npz' if fmt == 'npz' else '_bin'))
    argv = ['calib', '--checkpoint', str(work / checkpoint),
            '--primary-dir', str(work / 'primary'),
            '--far-dir', str(work / 'far'), '--frames', '4',
            '--format', fmt, '--frames-per-call', str(n),
            '--output', str(output)]
    if model_name == 'Align_ULCNet':
        argv.append('--allow-unstamped-linear-error')
    saved, sys.argv = sys.argv, argv
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            calibration.main(model_name)
    finally:
        sys.argv = saved
    with open(calibration_report_path(output, fmt)) as stream:
        manifest = json.load(stream)
    graph_json = os.path.splitext(sibling_onnx_path(output))[0] + '.json'
    with open(graph_json) as stream:
        graph = json.load(stream)
    return manifest, graph


@pytest.mark.parametrize('fmt', ['bin', 'npz'])
@pytest.mark.parametrize('n', [1, 8])
@pytest.mark.parametrize('model_name,checkpoint', [
    ('DeepVQE_S', 'dv.pth'), ('Align_ULCNet', 'al.pth')])
def test_manifest_layout_version_equals_graph_metadata(
        work, model_name, checkpoint, n, fmt):
    manifest, graph = _calib(work, model_name, checkpoint, n, fmt)
    assert graph['state_layout_version'] is not None
    assert manifest['state_layout_version'] == graph['state_layout_version']
