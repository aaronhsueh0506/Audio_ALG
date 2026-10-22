"""DFN3 restored C public lifecycle and Python feature/compose parity.

Each test runs for both scalar and SIMD builds. The process-only shared library
proves the DSP remains independently linkable; the separate executable exercises
its public pool/accelerator boundary and shared audiolib framing.
"""
import configparser
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

AINR = Path(__file__).resolve().parents[1]
ROOT = AINR.parent.parent
AC = ROOT / 'audiolib' / 'common'
DFN = AINR / 'DeepFilterNet3'
sys.path.insert(0, str(AINR.parent))
from AINR.DeepFilterNet3 import model as dfn_model  # noqa: E402
from AINR.DeepFilterNet3 import train as dfn_train  # noqa: E402


_WRAPPER = r'''
#include <stdlib.h>
#include "dfn3_process.h"
void *parity_create(const float *fwd, const float *inv) {
    DFN3State *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    dfn3_state_init(s, NULL, NULL);
    dfn3_set_erb_matrices(s, fwd, inv);
    return s;
}
void parity_destroy(void *s) { free(s); }
void parity_features(void *s, const float *re, const float *im,
                     float *erb, float *spec) {
    dfn3_compute_features(s, re, im, erb, spec);
}
int parity_compose(void *s, const float *re, const float *im,
                   const float *mask, const float *coefs,
                   float *out_re, float *out_im) {
    return dfn3_compose(s, re, im, mask, coefs, out_re, out_im);
}
int parity_stream(void *s, const float *re, const float *im, int valid,
                  const float *mask, const float *coefs, float limit,
                  float *out_re, float *out_im, long long *index) {
    return dfn3_compose_stream(s, re, im, valid, mask, coefs, limit,
                               out_re, out_im, index);
}
'''


def _fp(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


@pytest.fixture(scope='module', params=[0, 1], ids=['scalar', 'simd'])
def c_dfn3(request, tmp_path_factory):
    cc = shutil.which('cc')
    if cc is None or shutil.which('make') is None:
        pytest.skip('C build tools unavailable')
    simd = request.param
    make = ['make', '-s', '-C', str(AC), 'BACKEND=ne10', f'SIMD={simd}']
    subprocess.run(make + ['lib'], check=True, capture_output=True)
    archive = subprocess.run(make + ['print-lib-path'], check=True,
                             capture_output=True, text=True).stdout.strip().splitlines()[-1]
    work = tmp_path_factory.mktemp(f'dfn3_c_{simd}')
    flags = [cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Wpedantic',
             '-Werror', '-ffp-contract=off', '-fno-math-errno',
             '-I', str(AINR), '-I', str(DFN), '-I', str(AC / 'include'), '-I', str(AC)]
    if not simd:
        flags += ['-DSIMD_KERNELS_FORCE_SCALAR']
    driver = work / 'dfn3_stream'
    subprocess.run(flags + [str(AINR / 'tests/test_dfn3_stream.c')]
                   + [str(DFN / name) for name in
                      ['dfn3_process.c', 'dfn3_model_io.c', 'dfn3_prepost.c']]
                   + [archive, '-lm', '-o', str(driver)],
                   check=True, capture_output=True)
    wrapper = work / 'wrapper.c'
    wrapper.write_text(_WRAPPER)
    library = work / ('dfn3.dylib' if sys.platform == 'darwin' else 'dfn3.so')
    subprocess.run(flags + ['-dynamiclib' if sys.platform == 'darwin' else '-shared',
                            '-fPIC', str(wrapper), str(DFN / 'dfn3_process.c'),
                            archive, '-lm', '-o', str(library)],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.parity_create.argtypes = [fp, fp]
    lib.parity_create.restype = ctypes.c_void_p
    lib.parity_destroy.argtypes = [ctypes.c_void_p]
    lib.parity_features.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
    lib.parity_compose.argtypes = [ctypes.c_void_p, fp, fp, fp, fp, fp, fp]
    lib.parity_compose.restype = ctypes.c_int
    lib.parity_stream.argtypes = [ctypes.c_void_p, fp, fp, ctypes.c_int,
                                  fp, fp, ctypes.c_float, fp, fp,
                                  ctypes.POINTER(ctypes.c_longlong)]
    lib.parity_stream.restype = ctypes.c_int
    return lib, driver


def test_dfn3_c_public_stream_contract(c_dfn3):
    result = subprocess.run([str(c_dfn3[1])], check=True,
                            capture_output=True, text=True)
    assert 'identity delay=1024 samples PASS' in result.stdout


@pytest.mark.parametrize('atten_lim', [0.0, -20.0])
def test_dfn3_features_and_stream_compose_match_python(c_dfn3, atten_lim):
    lib = c_dfn3[0]
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    feature_cfg = dfn_train.read_feature_config(cfg, 48000, 512)
    erb_fb, erb_inv = dfn_model._build_erb_fb(1024, 48000, 32)
    fwd = np.ascontiguousarray(erb_fb.T.numpy(), dtype=np.float32)
    inv = np.ascontiguousarray(erb_inv.numpy(), dtype=np.float32)
    rng = np.random.default_rng(937)
    frames = 17  # wraps the five-frame raw-spectrum ring more than twice
    real = rng.normal(0, 0.15, (frames, 513)).astype(np.float32)
    imag = rng.normal(0, 0.15, (frames, 513)).astype(np.float32)
    spectrum = torch.complex(torch.from_numpy(real.T)[None],
                             torch.from_numpy(imag.T)[None])
    _, py_erb, py_spec, _ = dfn_train.extract_dfn3_features(
        spectrum, erb_fb, 96, feature_cfg)
    state = lib.parity_create(_fp(fwd), _fp(inv))
    assert state
    try:
        c_erb = np.empty((frames, 32), np.float32)
        c_spec = np.empty((frames, 192), np.float32)
        for t in range(frames):
            lib.parity_features(state, _fp(real[t]), _fp(imag[t]),
                                _fp(c_erb[t]), _fp(c_spec[t]))
        np.testing.assert_allclose(c_erb, py_erb[0, 0].numpy(),
                                   rtol=2e-5, atol=3e-6)
        expected_spec = py_spec[0].permute(1, 0, 2).reshape(frames, 192).numpy()
        np.testing.assert_allclose(c_spec, expected_spec, rtol=2e-5, atol=3e-6)
    finally:
        lib.parity_destroy(state)

    # Run actual restored heads, then cross the C host boundary with their
    # t-1 timing. No trained checkpoint is necessary for arithmetic parity.
    torch.manual_seed(54)
    model = dfn_model.DeepFilterNet3(**dfn_train.read_model_config(cfg)).eval()
    with torch.no_grad():
        masks, coefficients = model.heads(py_erb, py_spec)
        expected = model.compose(spectrum, masks, coefficients)[0]
        if atten_lim:
            mix = 10 ** (-abs(atten_lim) / 20)
            expected = spectrum[0] * mix + expected * (1 - mix)
    masks = masks[0, 0].numpy().copy()
    coefficients = coefficients[0].numpy().copy()
    state = lib.parity_create(_fp(fwd), _fp(inv))
    assert state
    zero = np.zeros(513, np.float32)
    output = []
    try:
        for t in range(frames + 1):
            r, i = (real[t], imag[t]) if t < frames else (zero, zero)
            out_r, out_i = np.empty(513, np.float32), np.empty(513, np.float32)
            index = ctypes.c_longlong(-77)
            valid = lib.parity_stream(
                state, _fp(r), _fp(i), int(t > 0),
                _fp(masks[t-1]) if t else None,
                _fp(coefficients[t-1]) if t else None,
                atten_lim, _fp(out_r), _fp(out_i), ctypes.byref(index))
            assert valid == int(t > 0)
            if valid:
                assert index.value == t - 1
                output.append(out_r + 1j * out_i)
    finally:
        lib.parity_destroy(state)
    np.testing.assert_allclose(np.stack(output, axis=1), expected.numpy(),
                               rtol=4e-5, atol=8e-6)


def test_dfn3_aligned_compose_matches_python(c_dfn3):
    lib = c_dfn3[0]
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    model = dfn_model.DeepFilterNet3(**dfn_train.read_model_config(cfg)).eval()
    rng = np.random.default_rng(23)
    frames = 13
    real = rng.normal(0, 0.15, (frames, 513)).astype(np.float32)
    imag = rng.normal(0, 0.15, (frames, 513)).astype(np.float32)
    masks = rng.uniform(0.05, 0.95, (frames, 32)).astype(np.float32)
    coefficients = rng.normal(0, 0.1, (frames, 96, 10)).astype(np.float32)
    spectrum = torch.complex(torch.from_numpy(real.T)[None],
                             torch.from_numpy(imag.T)[None])
    with torch.no_grad():
        expected = model.compose(spectrum, torch.from_numpy(masks)[None, None],
                                 torch.from_numpy(coefficients)[None])[0]
    fwd = np.ascontiguousarray(model.erb_fb.T.numpy())
    inv = np.ascontiguousarray(model.erb_inv.numpy())
    state = lib.parity_create(_fp(fwd), _fp(inv))
    output = []
    try:
        for t in range(frames):
            out_r, out_i = np.empty(513, np.float32), np.empty(513, np.float32)
            valid = lib.parity_compose(state, _fp(real[t]), _fp(imag[t]),
                                       _fp(masks[t]), _fp(coefficients[t]),
                                       _fp(out_r), _fp(out_i))
            assert valid == int(t > 0)
            if valid:
                output.append(out_r + 1j * out_i)
    finally:
        lib.parity_destroy(state)
    np.testing.assert_allclose(np.stack(output, axis=1), expected[:, :-1].numpy(),
                               rtol=4e-5, atol=8e-6)
