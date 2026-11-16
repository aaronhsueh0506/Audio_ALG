"""Two modes of the gate scripts that no other test enters: audit_fp_contract.sh under AEC_NO_ARRAY=1 (the array and
4ch rows are skipped, the single-channel rows are audited) and the default work directory of pyc_parity_pipelines.sh
(created private; refused when it is a symlink, belongs to someone else or is writable by others)."""

import os
import pathlib
import shutil
import subprocess

import pytest

_PIPELINES = pathlib.Path(__file__).resolve().parents[1]
_ALG = _PIPELINES.parent


def _resolved_dirs():
    out = subprocess.run(['make', '-s', '-f', str(_ALG / 'lib' / 'deps.mk'), 'print-deps-sh'],
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split('=', 1) for line in out.splitlines() if '=' in line)


def _need_tree():
    if shutil.which('make') is None or shutil.which('cc') is None:
        pytest.skip('no make / C compiler available')
    dirs = _resolved_dirs()
    if not all(os.path.isdir(dirs.get(key, '')) for key in ('AEC_DIR', 'NR_DIR', 'COMMON_DIR')):
        pytest.skip('the AEC, NR and audiolib checkouts are not available')


def test_fp_contract_audit_without_the_array_skips_its_rows():
    _need_tree()
    run = subprocess.run(['make', '-C', str(_PIPELINES), 'AEC_NO_ARRAY=1', 'test-fp-contract'],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-600:] + run.stderr[-600:]
    assert 'SKIPPED' in run.stdout and 'AEC_NO_ARRAY=1' in run.stdout
    assert 'objects audited: 5 ' in run.stdout and 'aec_array.o' not in run.stdout


@pytest.mark.parametrize('how', ['world_writable', 'group_writable', 'symlink'])
def test_pyc_parity_default_work_directory_must_be_private(tmp_path, how):
    _need_tree()
    work = tmp_path / ('pyc_parity_pipelines.%d' % os.getuid())
    if how == 'symlink':
        (tmp_path / 'elsewhere').mkdir()
        work.symlink_to(tmp_path / 'elsewhere')
    else:
        work.mkdir()
        os.chmod(work, 0o777 if how == 'world_writable' else 0o775)
    env = dict(os.environ, TMPDIR=str(tmp_path))
    env.pop('PYC_WORK', None)
    run = subprocess.run(['bash', str(_PIPELINES / 'tests' / 'pyc_parity_pipelines.sh')], env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 2, run.stdout[-300:] + run.stderr[-300:]
    assert 'closed to group and others' in run.stderr
