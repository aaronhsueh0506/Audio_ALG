"""Bit-exactness gate for the model-local NEON kernels of Align-ULCNet and
DeepVQE-S: the front and back end of ulcnet_model_io.c, the finite predicate
over strided state rows, and DeepVQE-S's compression. test_prepost_kernels.c
compares each kernel with its scalar twin and with a plain statement of the
contract on random and special values; this module builds it for the native
and the forced-scalar configuration (and for the 48 kHz Align-ULCNet grid) and
runs it.
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
_TEST_SOURCE = os.path.join(_THIS_DIR, 'test_prepost_kernels.c')

_DEEPVQE_SOURCES = (
    os.path.join(_DEEPVQE_DIR, 'deepvqe_process.c'),
    os.path.join(_AIAEC_DIR, 'aiaec_process.c'),
    os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
)

_GRID_48K = ('-DULCNET_MODEL_IO_SR=48000', '-DULCNET_MODEL_IO_N_FFT=1024')

_CASES = {
    'ulcnet': (('-DKERNEL_TEST_ULCNET',), ()),
    'ulcnet_scalar': (('-DKERNEL_TEST_ULCNET', '-DSIMD_KERNELS_FORCE_SCALAR'), ()),
    'ulcnet_48k': (('-DKERNEL_TEST_ULCNET',) + _GRID_48K, ()),
    'ulcnet_48k_scalar': (('-DKERNEL_TEST_ULCNET', '-DSIMD_KERNELS_FORCE_SCALAR')
                          + _GRID_48K, ()),
    'deepvqe': (('-DKERNEL_TEST_DEEPVQE',), _DEEPVQE_SOURCES),
    'deepvqe_scalar': (('-DKERNEL_TEST_DEEPVQE', '-DSIMD_KERNELS_FORCE_SCALAR'),
                       _DEEPVQE_SOURCES),
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
def test_kernels_match_their_twins_and_the_contract(
        case, tmp_path, audio_common_lib):
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    defines, extra_sources = _CASES[case]
    executable = tmp_path / case
    subprocess.run(
        [cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
         '-ffp-contract=off', '-fno-math-errno', *defines,
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR, '-I', _DEEPVQE_DIR,
         '-I', _AC_INCLUDE, _TEST_SOURCE, *extra_sources,
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True)
    done = subprocess.run([str(executable)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert 'ok' in done.stdout
