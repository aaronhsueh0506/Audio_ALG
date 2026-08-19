"""User-facing model scripts must work from their own directories."""

from pathlib import Path
import os
import subprocess
import sys

import pytest


AINR_ROOT = Path(__file__).parents[1]


def _help(model, script, *arguments):
    result = subprocess.run(
        [sys.executable, script, *arguments, '--help'],
        cwd=AINR_ROOT / model,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


@pytest.mark.parametrize('model', ('RNNoise-ERB', 'DeepFilterNet2',
                                   'DeepFilterNet3', 'GTCRN'))
def test_inference_help_runs_from_model_directory(model):
    _help(model, 'inference.py')


# DeepFilterNet3 is absent below on purpose: it ships neither a calib
# subcommand nor export_onnx.py.
@pytest.mark.parametrize('model', ('DeepFilterNet2', 'GTCRN'))
def test_calibration_help_runs_from_model_directory(model):
    _help(model, 'inference.py', 'calib')


@pytest.mark.parametrize('model', ('RNNoise-ERB', 'DeepFilterNet2', 'GTCRN'))
def test_export_help_runs_from_model_directory(model):
    _help(model, 'export_onnx.py')


@pytest.mark.parametrize('model', ('RNNoise-ERB', 'DeepFilterNet2', 'GTCRN'))
def test_exporter_does_not_import_from_audio_alg_parent(model):
    """AINR exporters must remain usable when AINR is released alone."""
    source = (AINR_ROOT / model / 'export_onnx.py').read_text()
    assert '_AUDIO_ALG_ROOT' not in source
    assert '_AINR_ROOT' in source

    result = subprocess.run(
        [sys.executable, '-c',
         'import onnx_streaming_contract; print(onnx_streaming_contract.__file__)'],
        cwd=AINR_ROOT / model,
        env={'PATH': os.environ.get('PATH', ''),
             'PYTHONPATH': str(AINR_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (
        AINR_ROOT / 'onnx_streaming_contract.py'
    ).resolve()
