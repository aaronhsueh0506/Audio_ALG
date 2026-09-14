"""Build a small C shim against audio_common (and extra sources) and load it
through ctypes.  Shared by the C-parity tests, which differ only in the shim
text, the sources it needs and the argtypes they bind."""
import ctypes
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AC = ROOT.parent / 'audio_common'


def build_shim(work, name, shim_source, extra_sources=(), extra_includes=(),
               defines=()):
    """Compile `shim_source` plus `extra_sources` into a shared library in
    `work` and return the loaded ctypes library (skips the test when no C
    toolchain is available)."""
    cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if cc is None or shutil.which('make') is None:
        pytest.skip('C build tools are unavailable')
    subprocess.run(['make', '-s', '-C', str(AC), 'BACKEND=ne10', 'lib'],
                   check=True, capture_output=True)
    archive = subprocess.run(
        ['make', '-s', '-C', str(AC), 'BACKEND=ne10', 'print-lib-path'],
        check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    shim = pathlib.Path(work) / f'{name}.c'
    shim.write_text(shim_source, encoding='utf-8')
    library = pathlib.Path(work) / (f'{name}.dylib' if sys.platform == 'darwin' else f'{name}.so')
    shared = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
    command = [cc, shared, '-fPIC', '-O2', '-std=c11', '-ffp-contract=off',
               '-fno-math-errno']
    for define in defines:
        command.append('-D' + define)
    for include in (AC / 'include', *extra_includes):
        command += ['-I', str(include)]
    command += [str(shim), *[str(s) for s in extra_sources], archive, '-lm',
                '-o', str(library)]
    subprocess.run(command, check=True, capture_output=True)
    return ctypes.CDLL(str(library))


def float_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
