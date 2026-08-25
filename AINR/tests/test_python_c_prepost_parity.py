"""Numeric parity between shipped AINR C pre/post-processing and Python.

The accelerator owns only learned operators. These tests cross the actual
host boundary: DFN2 features and composition, plus GTCRN ERB features and CRM
synthesis. The existing C executable remains responsible for lifecycle,
state-commit, WOLA, and SIMD/scalar checks.
"""

import configparser
import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch


AINR = Path(__file__).resolve().parents[1]
ROOT = AINR.parent.parent
AC = ROOT / 'audio_common'
DFN = AINR / 'DeepFilterNet2'
GT = AINR / 'GTCRN'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dfn_modules():
    model = _load('dfn2_parity_model', DFN / 'model.py')
    old_model = sys.modules.get('model')
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(DFN))
        sys.modules['model'] = model
        train = _load('dfn2_parity_train', DFN / 'train.py')
    finally:
        sys.path[:] = old_path
        if old_model is None:
            sys.modules.pop('model', None)
        else:
            sys.modules['model'] = old_model
    return model, train


_WRAPPER = r'''
#include <stdlib.h>
#include "dfn2_process.h"
#include "gtcrn_process.h"

void *parity_dfn_create(const float *fwd, const float *inv) {
    DFN2State *state = (DFN2State *)calloc(1, sizeof(*state));
    if (!state) return NULL;
    dfn2_state_init(state, NULL);
    dfn2_set_erb_matrices(state, fwd, inv);
    return state;
}
void parity_dfn_destroy(void *state) { free(state); }
void parity_dfn_features(void *state, const float *re, const float *im,
                         float *erb, float *spec) {
    dfn2_compute_features((DFN2State *)state, re, im, erb, spec);
}
int parity_dfn_compose(void *state, const float *re, const float *im,
                       const float *mask, const float *coefs, float alpha,
                       float *out_re, float *out_im) {
    return dfn2_compose((DFN2State *)state, re, im, mask, coefs, alpha,
                        out_re, out_im);
}
void parity_gt_input(const float *spectrum, const float *fwd,
                     float *mag, float *re, float *im) {
    gtcrn_model_input((const float (*)[2])spectrum, fwd, mag, re, im);
}
void parity_gt_output(const float *mask, const float *inv,
                      const float *spectrum, float *enhanced) {
    gtcrn_model_output((const float (*)[2])mask, inv,
                       (const float (*)[2])spectrum,
                       (float (*)[2])enhanced);
}
'''


@pytest.fixture(scope='module')
def c_prepost(tmp_path_factory):
    cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if cc is None or shutil.which('make') is None:
        pytest.skip('C build tools are unavailable')
    subprocess.run(
        ['make', '-s', '-C', str(AC), 'BACKEND=ne10', 'lib'],
        check=True, capture_output=True,
    )
    archive = subprocess.run(
        ['make', '-s', '-C', str(AC), 'BACKEND=ne10', 'print-lib-path'],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[-1]

    work = tmp_path_factory.mktemp('ainr_c_python')
    wrapper = work / 'wrapper.c'
    wrapper.write_text(_WRAPPER, encoding='utf-8')
    library = work / ('prepost.dylib' if sys.platform == 'darwin'
                      else 'prepost.so')
    shared_flag = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
    subprocess.run(
        [cc, shared_flag, '-fPIC', '-O2', '-std=c11', '-ffp-contract=off',
         '-I', str(DFN), '-I', str(GT), '-I', str(AC / 'include'),
         str(wrapper), str(DFN / 'dfn2_process.c'),
         str(GT / 'gtcrn_process.c'), archive, '-lm', '-o', str(library)],
        check=True, capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.parity_dfn_create.argtypes = [fp, fp]
    lib.parity_dfn_create.restype = ctypes.c_void_p
    lib.parity_dfn_destroy.argtypes = [ctypes.c_void_p]
    lib.parity_dfn_features.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
    lib.parity_dfn_compose.argtypes = [
        ctypes.c_void_p, fp, fp, fp, fp, ctypes.c_float, fp, fp]
    lib.parity_dfn_compose.restype = ctypes.c_int
    lib.parity_gt_input.argtypes = [fp, fp, fp, fp, fp]
    lib.parity_gt_output.argtypes = [fp, fp, fp, fp]
    return lib


def _fp(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def test_dfn2_features_and_compose_match_python(c_prepost):
    dfn_model, dfn_train = _load_dfn_modules()
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    feature_cfg = dfn_train.read_feature_config(cfg, 48000, 512)
    erb_fb, erb_inv = dfn_model._build_erb_fb(1024, 48000, 32)
    fwd = np.ascontiguousarray(erb_fb.T.numpy(), dtype=np.float32)
    inv = np.ascontiguousarray(erb_inv.numpy(), dtype=np.float32)

    rng = np.random.default_rng(41)
    frames = 9
    real = rng.normal(0, 0.15, (frames, 513)).astype(np.float32)
    imag = rng.normal(0, 0.15, (frames, 513)).astype(np.float32)
    spectrum = torch.complex(torch.from_numpy(real.T)[None],
                             torch.from_numpy(imag.T)[None])
    _, py_erb, py_spec, _ = dfn_train.extract_dfn2_features(
        spectrum, erb_fb, 96, feature_cfg)

    state = c_prepost.parity_dfn_create(_fp(fwd), _fp(inv))
    assert state
    try:
        c_erb = np.empty((frames, 32), np.float32)
        c_spec = np.empty((frames, 192), np.float32)
        for frame in range(frames):
            c_prepost.parity_dfn_features(
                state, _fp(real[frame]), _fp(imag[frame]),
                _fp(c_erb[frame]), _fp(c_spec[frame]))
    finally:
        c_prepost.parity_dfn_destroy(state)

    np.testing.assert_allclose(c_erb, py_erb[0, 0].numpy(),
                               rtol=2e-5, atol=3e-6)
    expected_spec = py_spec[0].permute(1, 0, 2).reshape(frames, 192).numpy()
    np.testing.assert_allclose(c_spec, expected_spec,
                               rtol=2e-5, atol=3e-6)

    model = dfn_model.DeepFilterNet2(
        **dfn_train.read_model_config(cfg)).eval()
    mask = torch.sigmoid(torch.randn(1, 1, frames, 32))
    coefs = torch.randn(1, frames, 96, 10) * 0.1
    alpha = torch.sigmoid(torch.randn(1, frames, 1))
    with torch.no_grad():
        expected = model.compose(spectrum, mask, coefs, alpha)[0]

    state = c_prepost.parity_dfn_create(_fp(fwd), _fp(inv))
    assert state
    c_out = []
    try:
        for frame in range(frames):
            out_re = np.empty(513, np.float32)
            out_im = np.empty(513, np.float32)
            valid = c_prepost.parity_dfn_compose(
                state, _fp(real[frame]), _fp(imag[frame]),
                _fp(np.ascontiguousarray(mask[0, 0, frame].numpy())),
                _fp(np.ascontiguousarray(coefs[0, frame].numpy())),
                ctypes.c_float(alpha[0, frame, 0].item()),
                _fp(out_re), _fp(out_im))
            if valid:
                c_out.append(out_re + 1j * out_im)
    finally:
        c_prepost.parity_dfn_destroy(state)
    got = np.stack(c_out, axis=1)
    np.testing.assert_allclose(got, expected[:, :frames - 1].numpy(),
                               rtol=4e-5, atol=8e-6)


def test_gtcrn_host_boundary_matches_python(c_prepost):
    gt_model = _load('gtcrn_parity_model', GT / 'model.py')
    gt_stream = _load('gtcrn_parity_stream', GT / 'stream_model.py')
    model = gt_model.GTCRN().eval()
    fwd = np.ascontiguousarray(
        model.erb.erb_fc.weight.detach().numpy().T, dtype=np.float32)
    inv = np.ascontiguousarray(
        model.erb.ierb_fc.weight.detach().numpy().T, dtype=np.float32)
    rng = np.random.default_rng(73)
    spectrum = rng.normal(0, 0.2, (257, 2)).astype(np.float32)
    spectrum_t = torch.from_numpy(spectrum)[None, :, None, :]
    with torch.no_grad():
        py_mag, py_re, py_im = gt_stream.stream_features(model, spectrum_t)

    c_mag = np.empty(129, np.float32)
    c_re = np.empty(129, np.float32)
    c_im = np.empty(129, np.float32)
    c_prepost.parity_gt_input(
        _fp(spectrum), _fp(fwd), _fp(c_mag), _fp(c_re), _fp(c_im))
    np.testing.assert_allclose(c_mag, py_mag[0, :, 0].numpy(),
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(c_re, py_re[0, :, 0].numpy(),
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(c_im, py_im[0, :, 0].numpy(),
                               rtol=2e-5, atol=2e-6)

    mask = rng.normal(0, 0.4, (129, 2)).astype(np.float32)
    with torch.no_grad():
        expected = gt_stream.host_synthesis(
            model, torch.from_numpy(mask)[None, :, None, :], spectrum_t)
    enhanced = np.empty((257, 2), np.float32)
    c_prepost.parity_gt_output(
        _fp(mask), _fp(inv), _fp(spectrum), _fp(enhanced))
    np.testing.assert_allclose(enhanced, expected[0, :, 0].numpy(),
                               rtol=3e-5, atol=3e-6)
