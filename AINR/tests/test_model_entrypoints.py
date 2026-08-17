"""User-facing model scripts must work from their own directories."""

from pathlib import Path
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
