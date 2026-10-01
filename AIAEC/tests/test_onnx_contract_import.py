"""The exporters' padding validator resolves in a full checkout and in a
standalone AIAEC copy that has AINR/ beside it but no repository-root module."""

import shutil
import subprocess
import sys
from pathlib import Path

AIAEC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AIAEC_ROOT.parent

_PROBE = r'''
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "_onnx_contract", sys.argv[1] + "/AIAEC/_onnx_contract.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.validate_nctf_no_temporal_padding.__code__.co_filename)
'''


def _probe(root, extra_path=()):
    return subprocess.run(
        [sys.executable, '-c', _PROBE, str(root)],
        capture_output=True, text=True, cwd=str(root),
        env={'PYTHONPATH': ':'.join(str(p) for p in extra_path),
             'PYTHONDONTWRITEBYTECODE': '1'},
    )


def _standalone_copy(tmp_path, with_ainr=True):
    (tmp_path / 'AIAEC').mkdir()
    shutil.copy(AIAEC_ROOT / '_onnx_contract.py', tmp_path / 'AIAEC')
    if with_ainr:
        (tmp_path / 'AINR').mkdir()
        shutil.copy(REPO_ROOT / 'AINR' / 'onnx_streaming_contract.py',
                    tmp_path / 'AINR')
    return tmp_path


def test_full_checkout_uses_the_repository_root_module():
    result = _probe(REPO_ROOT, extra_path=[REPO_ROOT])
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT / 'onnx_streaming_contract.py'


def test_standalone_aiaec_copy_falls_back_to_the_ainr_module(tmp_path):
    root = _standalone_copy(tmp_path)
    result = _probe(root)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == root / 'AINR' / 'onnx_streaming_contract.py'


def test_missing_module_still_fails_loudly_without_ainr(tmp_path):
    root = _standalone_copy(tmp_path, with_ainr=False)
    result = _probe(root)
    assert result.returncode != 0
    assert "No module named 'onnx_streaming_contract'" in result.stderr
