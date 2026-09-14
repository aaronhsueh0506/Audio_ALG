"""The Python stage's compose and feature branches against the C they mirror.

``dfn2_stage.Dfn2Stage`` claims to be ``dfn2_compute_features`` (on A) plus
``dfn2_compose_stream`` (on B) with the C's rounding sequence.  This drives the
real C functions from ``AINR/DeepFilterNet2/dfn2_process.c`` through a tiny
shim and compares frame by frame with stand-in heads, so the comparison is
about the arithmetic, not about a network.

* compose: bit-exact (``np.array_equal``) on every emitted frame -- the stage
  performs each multiply and add as its own float32 operation, in the C order;
* features: the existing AINR parity band (rtol 2e-5 / atol 3e-6), because the
  torch reference the stage reuses is not the C's op order.
"""
import ctypes
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip('torch')

from pipelines.dfn2_stage import (  # noqa: E402
    DF_BINS, DF_ORDER, N_BINS, N_ERB, SCALE_IN, SCALE_OUT, Dfn2Stage, load_dfn2,
)
from pipelines.tests.c_shim import build_shim, float_pointer as _fp  # noqa: E402

DFN = ROOT / 'AINR' / 'DeepFilterNet2'

_SHIM = r'''
#include <stdlib.h>
#include "dfn2_process.h"

void *stage_parity_create(const float *fwd, const float *inv) {
    DFN2State *state = (DFN2State *)calloc(1, sizeof(*state));
    if (!state) return NULL;
    dfn2_state_init(state, NULL);
    dfn2_set_erb_matrices(state, fwd, inv);
    return state;
}
void stage_parity_destroy(void *state) { free(state); }
void stage_parity_features(void *state, const float *re, const float *im,
                           float *erb, float *spec) {
    dfn2_compute_features((DFN2State *)state, re, im, erb, spec);
}
int stage_parity_compose_stream(void *state, const float *re, const float *im,
                                int heads_valid, const float *mask,
                                const float *coefs, float alpha,
                                float atten_lim_db, float *out_re,
                                float *out_im, long long *frame) {
    return dfn2_compose_stream((DFN2State *)state, re, im, heads_valid,
                               heads_valid ? mask : NULL,
                               heads_valid ? coefs : NULL, alpha,
                               atten_lim_db, out_re, out_im, frame);
}
'''



@pytest.fixture(scope='module')
def assets():
    return load_dfn2(seed=0)

@pytest.fixture(scope='module')
def c_lib(tmp_path_factory):
    lib = build_shim(tmp_path_factory.mktemp('dfn2_stage_c_parity'), 'shim', _SHIM,
                     extra_sources=[DFN / 'dfn2_process.c'], extra_includes=[DFN])
    fp = ctypes.POINTER(ctypes.c_float)
    lib.stage_parity_create.argtypes = [fp, fp]
    lib.stage_parity_create.restype = ctypes.c_void_p
    lib.stage_parity_destroy.argtypes = [ctypes.c_void_p]
    lib.stage_parity_features.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
    lib.stage_parity_compose_stream.argtypes = [
        ctypes.c_void_p, fp, fp, ctypes.c_int, fp, fp, ctypes.c_float,
        ctypes.c_float, fp, fp, ctypes.POINTER(ctypes.c_longlong)]
    lib.stage_parity_compose_stream.restype = ctypes.c_int
    return lib


class ScriptedHeads:
    """Deterministic, frame-varying heads shared by both sides."""

    def __init__(self, seed):
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.RandomState(self.seed)
        self.calls = 0

    def __call__(self, erb_window=None, spec_window=None):
        self.calls += 1
        mask = self.rng.uniform(0.05, 1.0, N_ERB).astype(np.float32)
        coefs = (self.rng.randn(DF_BINS, DF_ORDER, 2) * 0.3).astype(np.float32)
        alpha = float(self.rng.uniform(0.0, 1.0))
        return mask, coefs, alpha


def _spectra(count, seed, scale):
    rng = np.random.RandomState(seed)
    re = (rng.randn(count, N_BINS) * scale).astype(np.float32)
    im = (rng.randn(count, N_BINS) * scale).astype(np.float32)
    im[:, 0] = 0.0
    im[:, -1] = 0.0
    return re, im


@pytest.mark.parametrize('atten_lim_db', [0.0, 20.0])
def test_compose_stream_is_bit_exact_against_c(c_lib, assets, atten_lim_db):
    erb_fwd = np.ascontiguousarray(assets.erb_fb.numpy().T, np.float32)  # [bins][bands]
    erb_inv = np.ascontiguousarray(assets.erb_inv, np.float32)           # [bands][bins]
    state = c_lib.stage_parity_create(_fp(erb_fwd), _fp(erb_inv))
    assert state
    frames = 40
    a_re, a_im = _spectra(frames, seed=21, scale=0.7)
    b_re, b_im = _spectra(frames, seed=22, scale=0.2)

    py_heads = ScriptedHeads(seed=5)
    c_heads = ScriptedHeads(seed=5)
    stage = Dfn2Stage(assets, heads=py_heads, atten_lim_db=atten_lim_db)

    out_re = np.zeros(N_BINS, np.float32)
    out_im = np.zeros(N_BINS, np.float32)
    frame = ctypes.c_longlong(-1)
    compared = 0
    try:
        for t in range(frames):
            # The stage takes audio-scale spectra and scales by 2^-5 itself;
            # feed the C the same normalised values.
            b_scaled_re = (b_re[t] * SCALE_IN).astype(np.float32)
            b_scaled_im = (b_im[t] * SCALE_IN).astype(np.float32)
            heads_valid = 1 if t >= 1 else 0
            if heads_valid:
                mask, coefs, alpha = c_heads()
            else:
                mask = np.ones(N_ERB, np.float32)
                coefs = np.zeros((DF_BINS, DF_ORDER, 2), np.float32)
                alpha = 0.0
            rc = c_lib.stage_parity_compose_stream(
                state, _fp(b_scaled_re), _fp(b_scaled_im), heads_valid,
                _fp(np.ascontiguousarray(mask)), _fp(np.ascontiguousarray(coefs)),
                ctypes.c_float(alpha), ctypes.c_float(atten_lim_db),
                _fp(out_re), _fp(out_im), ctypes.byref(frame))
            assert rc >= 0
            py = stage.push((a_re[t] + 1j * a_im[t]).astype(np.complex64),
                            (b_re[t] + 1j * b_im[t]).astype(np.complex64))
            if rc == 0:
                assert py is None
                continue
            assert py is not None
            assert py.frame_index == frame.value
            expected = ((out_re * SCALE_OUT).astype(np.float32)
                        + 1j * (out_im * SCALE_OUT).astype(np.float32)).astype(np.complex64)
            assert np.array_equal(py.spectrum, expected), t
            compared += 1
    finally:
        c_lib.stage_parity_destroy(state)
    assert compared == frames - 2
    assert py_heads.calls == c_heads.calls == frames - 1


def test_features_match_c_within_the_ainr_parity_band(c_lib, assets):
    erb_fwd = np.ascontiguousarray(assets.erb_fb.numpy().T, np.float32)
    erb_inv = np.ascontiguousarray(assets.erb_inv, np.float32)
    state = c_lib.stage_parity_create(_fp(erb_fwd), _fp(erb_inv))
    assert state
    frames = 30
    a_re, a_im = _spectra(frames, seed=31, scale=0.5)
    stage = Dfn2Stage(assets, heads=ScriptedHeads(seed=9))
    c_erb = np.zeros(N_ERB, np.float32)
    c_spec = np.zeros(2 * DF_BINS, np.float32)
    try:
        for t in range(frames):
            sr = (a_re[t] * SCALE_IN).astype(np.float32)
            si = (a_im[t] * SCALE_IN).astype(np.float32)
            c_lib.stage_parity_features(state, _fp(sr), _fp(si), _fp(c_erb), _fp(c_spec))
            py_erb, py_spec = stage._features(sr, si)
            np.testing.assert_allclose(py_erb, c_erb, rtol=2e-5, atol=3e-6)
            np.testing.assert_allclose(py_spec.reshape(-1), c_spec, rtol=2e-5, atol=3e-6)
    finally:
        c_lib.stage_parity_destroy(state)
