"""Golden digests of the Align-ULCNet (16 and 48 kHz) and DeepVQE-S pre/post
chains. test_prepost_golden.c drives each chain through its public
frequency-domain class with a seeded generator and compares one digest per
delay depth with the recorded constant; this module builds it for the native
and the forced-scalar configuration, which must reproduce the same constants,
and runs it. A host whose libm differs from the recorded one makes the test
report SKIP rather than a verdict.
"""

import os
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AIAEC_DIR = os.path.dirname(_THIS_DIR)
_ULCNET_DIR = os.path.join(_AIAEC_DIR, 'Align_ULCNet')
_DEEPVQE_DIR = os.path.join(_AIAEC_DIR, 'DeepVQE_S')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audio_common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')
_TEST_SOURCE = os.path.join(_THIS_DIR, 'test_prepost_golden.c')

_ULCNET_SOURCES = tuple(os.path.join(_ULCNET_DIR, name) for name in (
    'ulcnet_process.c', 'ulcnet_model_io.c', 'ulcnet_prepost.c'))
_DEEPVQE_SOURCES = (
    os.path.join(_DEEPVQE_DIR, 'deepvqe_process.c'),
    os.path.join(_AIAEC_DIR, 'aiaec_process.c'),
    os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.c'),
)

_GRID_48K = ('-DULCNET_MODEL_IO_SR=48000', '-DULCNET_MODEL_IO_N_FFT=1024')
_SCALAR = ('-DSIMD_KERNELS_FORCE_SCALAR',)
_ULCNET = ('-DGOLDEN_ULCNET',)
_DEEPVQE = ('-DGOLDEN_DEEPVQE',)

_CASES = {
    'ulcnet': (_ULCNET, _ULCNET_SOURCES),
    'ulcnet_scalar': (_ULCNET + _SCALAR, _ULCNET_SOURCES),
    'ulcnet_48k': (_ULCNET + _GRID_48K, _ULCNET_SOURCES),
    'ulcnet_48k_scalar': (_ULCNET + _GRID_48K + _SCALAR, _ULCNET_SOURCES),
    'deepvqe': (_DEEPVQE, _DEEPVQE_SOURCES),
    'deepvqe_scalar': (_DEEPVQE + _SCALAR, _DEEPVQE_SOURCES),
}


@pytest.fixture(scope='module')
def audio_common_lib():
    if shutil.which('make') is None:
        pytest.skip('no make available')
    subprocess.run(['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'lib'],
                   check=True, capture_output=True)
    out = subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'print-lib-path'],
        check=True, capture_output=True, text=True)
    lib = out.stdout.strip().splitlines()[-1]
    assert os.path.isfile(lib), lib
    return lib


@pytest.mark.parametrize('case', sorted(_CASES))
def test_chain_reproduces_the_recorded_digests(
        case, tmp_path, audio_common_lib):
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    defines, sources = _CASES[case]
    executable = tmp_path / case
    subprocess.run(
        [cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
         '-ffp-contract=off', '-fno-math-errno', *defines,
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR, '-I', _DEEPVQE_DIR,
         '-I', _AC_INCLUDE, _TEST_SOURCE, *sources,
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True)
    done = subprocess.run([str(executable)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    if 'SKIP' in done.stdout:
        pytest.skip('host libm differs from the recorded one')
    assert 'ok' in done.stdout
