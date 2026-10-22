"""Whole-chain digests of the Align-ULCNet (16 and 48 kHz) and DeepVQE-S
pre/post chains. test_prepost_chains.c drives each chain through its public
frequency-domain class with a seeded generator, including the non-finite-head,
non-finite-state, skipped-frame and unwritten-frame schedules, and prints one
digest per delay depth. This module builds it for the native and the
forced-scalar configuration and requires the two to print the same digests; a
different generator seed must change the digest, so the comparison can fail.
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
    os.path.join(_THIS_DIR, '..', '..', '..', 'audiolib', 'common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')
_TEST_SOURCE = os.path.join(_THIS_DIR, 'test_prepost_chains.c')

_ULCNET_SOURCES = tuple(os.path.join(_ULCNET_DIR, name) for name in (
    'ulcnet_process.c', 'ulcnet_model_io.c', 'ulcnet_prepost.c'))
_DEEPVQE_SOURCES = (
    os.path.join(_DEEPVQE_DIR, 'deepvqe_process.c'),
    os.path.join(_AIAEC_DIR, 'aiaec_process.c'),
    os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.c'),
)

_GRID_48K = ('-DULCNET_MODEL_IO_SR=48000', '-DULCNET_MODEL_IO_N_FFT=1024')
_SCALAR = ('-DSIMD_KERNELS_FORCE_SCALAR',)
_ULCNET = ('-DCHAIN_ULCNET',)
_DEEPVQE = ('-DCHAIN_DEEPVQE',)

_CHAINS = {
    'ulcnet': (_ULCNET, _ULCNET_SOURCES),
    'ulcnet_48k': (_ULCNET + _GRID_48K, _ULCNET_SOURCES),
    'deepvqe': (_DEEPVQE, _DEEPVQE_SOURCES),
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


def _digest(chain, extra, tmp_path, audio_common_lib):
    """Build the chain with `extra` compiler flags, run it, return its digest."""
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    defines, sources = _CHAINS[chain]
    executable = tmp_path / ('_'.join((chain,) + tuple(
        c.strip('-D=').replace('=', '') for c in extra)) or chain)
    subprocess.run(
        [cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
         '-ffp-contract=off', '-fno-math-errno', *defines, *extra,
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR, '-I', _DEEPVQE_DIR,
         '-I', _AC_INCLUDE, '-I', _AC_DIR, _TEST_SOURCE, *sources,
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True)
    done = subprocess.run([str(executable)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    line = done.stdout.strip().splitlines()[-1]
    assert 'digest=' in line, line
    return line.split('digest=', 1)[1]


@pytest.mark.parametrize('chain', sorted(_CHAINS))
def test_native_and_scalar_chains_agree_bit_for_bit(
        chain, tmp_path, audio_common_lib):
    native = _digest(chain, (), tmp_path, audio_common_lib)
    scalar = _digest(chain, _SCALAR, tmp_path, audio_common_lib)
    assert native == scalar


def test_a_different_seed_changes_the_digest(tmp_path, audio_common_lib):
    reference = _digest('ulcnet', (), tmp_path, audio_common_lib)
    changed = _digest('ulcnet', ('-DRNG_SEED=0x6b1d5eeeu',), tmp_path,
                      audio_common_lib)
    assert changed != reference
