"""Build helpers for the tests that compile process.c against audiolib."""

import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _common_dir():
    return pathlib.Path(os.environ.get(
        'COMMON_DIR', ROOT.parents[2] / 'audiolib' / 'common')).resolve()


def compile_c(sources, output, shared=False):
    """Compile `sources` (paths or C text already written to files) with process.c and audiolib.

    RNNOISE_TEST_SIMD=0 selects the scalar build; COMMON_DIR the audiolib tree; EXTRA_CFLAGS is added to both builds.
    """
    cc = os.environ.get('CC') or shutil.which('cc')
    if not cc:
        raise RuntimeError('C compiler not found (set CC or install cc)')
    common = _common_dir()
    simd = os.environ.get('RNNOISE_TEST_SIMD', '1')
    if simd not in ('0', '1'):
        raise ValueError('RNNOISE_TEST_SIMD must be 0 or 1')
    common_make = ['make', '-s', '--no-print-directory', '-C', str(common),
                   f'SIMD={simd}', f'CC={cc}']
    subprocess.run(common_make + ['lib'], check=True, stdout=subprocess.DEVNULL)
    archive = subprocess.check_output(common_make + ['print-lib-path'], text=True).strip()
    flags = ['-O2', '-ffp-contract=off', '-fno-math-errno', *os.environ.get('EXTRA_CFLAGS', '').split()]
    if simd == '0':
        flags.append('-DSIMD_KERNELS_FORCE_SCALAR')
    if shared:
        flags += ['-dynamiclib' if sys.platform == 'darwin' else '-shared', '-fPIC']
    subprocess.run([
        cc, *flags, '-I', str(ROOT), '-I', str(ROOT / 'test'),
        '-I', str(common / 'include'), '-I', str(common),
        *map(str, sources), str(ROOT / 'process.c'), archive,
        '-o', str(output), '-lm',
    ], check=True)
