"""DFN3 restored C public lifecycle and Python feature/compose parity.

Each test runs for both scalar and SIMD builds and for every (mask_lookahead,
df_lookahead) pair in ``dfn_pairs.c_build_pairs``: the pair is a build-time
constant of the C host, so each one is its own compile (-DDFN3_MASK_LOOKAHEAD,
-DDFN3_DF_LOOKAHEAD; the shipped pair is the build with no override). The
shared library (process + model I/O) proves the DSP remains independently
linkable; the separate executable exercises its public pool/accelerator
boundary and shared audiolib framing. Compiler flags are the Makefile's.
"""
import configparser
import ctypes
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import dfn_pairs

FAMILY = 'DeepFilterNet3'
LOOKAHEAD_PAIRS = dfn_pairs.c_build_pairs(FAMILY)
SHIPPED_PAIR = dfn_pairs.shipped_pair(FAMILY)

AINR = dfn_pairs.AINR
DFN = dfn_pairs.family_dir(FAMILY)
sys.path.insert(0, str(AINR.parent))
from AINR.DeepFilterNet3 import export_onnx as dfn_export  # noqa: E402
from AINR.DeepFilterNet3 import model as dfn_model  # noqa: E402
from AINR.DeepFilterNet3 import train as dfn_train  # noqa: E402


_WRAPPER = r'''
#include <stdlib.h>
#include <string.h>
#include "DeepFilterNet3/dfn3_model_io.h"
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
void *parity_io_create(void) {
    DFN3ModelIOState *io = calloc(1, sizeof(*io));
    if (io) dfn3_model_io_reset(io);
    return io;
}
int parity_io_push(void *io, const float *erb, const float *spec) {
    return dfn3_model_io_push_features(io, erb, (const float (*)[DFN3_DF_BINS])spec);
}
void parity_io_window(void *io, float *erb_window, float *spec_window) {
    DFN3ModelIOState *state = io;
    memcpy(erb_window, state->erb_window, sizeof(state->erb_window));
    memcpy(spec_window, state->spec_window, sizeof(state->spec_window));
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


@pytest.fixture(scope='module', params=[
    (simd, pair) for pair in LOOKAHEAD_PAIRS for simd in (0, 1)],
    ids=lambda case: ('simd' if case[0] else 'scalar') + (
        '' if case[1] == SHIPPED_PAIR else '-' + dfn_pairs.pair_id(case[1])))
def c_dfn3(request, tmp_path_factory):
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    simd, pair = request.param
    archive = dfn_pairs.audio_common_lib(simd)
    work = tmp_path_factory.mktemp(f'dfn3_c_{simd}_{pair[0]}{pair[1]}')
    flags = ['cc', *dfn_pairs.house_cflags(simd), *dfn_pairs.defines(FAMILY, pair)]
    driver = work / 'dfn3_stream'
    subprocess.run(flags + [str(AINR / 'tests/test_dfn3_stream.c')]
                   + [str(dfn_pairs.source_path(FAMILY, kind)) for kind in
                      ('process', 'model_io', 'prepost')]
                   + [archive, '-lm', '-o', str(driver)],
                   check=True, capture_output=True)
    wrapper = work / 'wrapper.c'
    wrapper.write_text(_WRAPPER)
    library = work / ('dfn3.dylib' if sys.platform == 'darwin' else 'dfn3.so')
    subprocess.run(flags + ['-dynamiclib' if sys.platform == 'darwin' else '-shared',
                            '-fPIC', str(wrapper), str(DFN / 'dfn3_process.c'),
                            str(DFN / 'dfn3_model_io.c'), archive, '-lm', '-o', str(library)],
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
    lib.parity_io_create.restype = ctypes.c_void_p
    lib.parity_io_push.argtypes = [ctypes.c_void_p, fp, fp]
    lib.parity_io_push.restype = ctypes.c_int
    lib.parity_io_window.argtypes = [ctypes.c_void_p, fp, fp]
    return lib, driver, pair


def test_dfn3_c_public_stream_contract(c_dfn3):
    mask, df = c_dfn3[2]
    result = subprocess.run([str(c_dfn3[1])], check=True,
                            capture_output=True, text=True)
    # One hop of framing plus the model lookahead max(mask, df) (1024 samples
    # when shipped).
    assert f'mask={mask} df={df}:' in result.stdout
    assert (f'identity delay={(1 + max(mask, df)) * 512} samples PASS'
            in result.stdout)


def test_c_feature_windows_equal_the_exporters_windows(c_dfn3):
    """The host's window for heads[t] is the window the exporter and every
    replay helper slide (``export_onnx.feature_windows``): the C readiness and
    zero-fill rule against the Python one, with random features and a flush of
    ``mask_lookahead`` zero frames."""
    lib = c_dfn3[0]
    mask = c_dfn3[2][0]
    frames = 9
    rng = np.random.default_rng(11)
    erb = rng.normal(size=(1, 1, frames, 32)).astype(np.float32)
    spec = rng.normal(size=(1, 2, frames, 96)).astype(np.float32)
    wanted = list(zip(
        dfn_export.feature_windows(torch.from_numpy(erb), mask),
        dfn_export.feature_windows(torch.from_numpy(spec), mask)))
    io = lib.parity_io_create()
    assert io
    try:
        seen = []
        for n in range(frames + mask):
            e = erb[0, 0, n] if n < frames else np.zeros(32, np.float32)
            c = spec[0, :, n] if n < frames else np.zeros((2, 96), np.float32)
            ready = lib.parity_io_push(io, _fp(np.ascontiguousarray(e)),
                                       _fp(np.ascontiguousarray(c)))
            assert ready == int(n >= mask)
            if ready:
                got_erb = np.empty((3, 32), np.float32)
                got_spec = np.empty((2, 3, 96), np.float32)
                lib.parity_io_window(io, _fp(got_erb), _fp(got_spec))
                seen.append((got_erb, got_spec))
    finally:
        lib.parity_destroy(io)
    assert len(seen) == len(wanted) == frames
    for t, ((got_erb, got_spec), (want_erb, want_spec)) in enumerate(zip(seen, wanted)):
        assert np.array_equal(got_erb, want_erb[0, 0].numpy()), t
        assert np.array_equal(got_spec, want_spec[0].numpy()), t


@pytest.mark.parametrize('std', ['c11', 'gnu99'])
@pytest.mark.parametrize('pair', [(3, 0), (3, 3), (-1, 0), (0, -1), (1, 5),
                                  (2, 5)])
def test_dfn3_header_refuses_pairs_the_trainer_would_not_accept(
        tmp_path, pair, std):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    refused = dfn_pairs.header_compiles(FAMILY, pair, std, tmp_path)
    assert refused.returncode != 0
    assert 'DFN3_' in refused.stderr and 'LOOKAHEAD must be' in refused.stderr
    # The same translation unit accepts the shipped pair.
    assert dfn_pairs.header_compiles(
        FAMILY, SHIPPED_PAIR, std, tmp_path).returncode == 0


@pytest.mark.parametrize('atten_lim', [0.0, -20.0])
def test_dfn3_features_and_stream_compose_match_python(c_dfn3, atten_lim):
    lib = c_dfn3[0]
    mask_lookahead, df_lookahead = c_dfn3[2]
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    feature_cfg = dfn_train.read_feature_config(cfg, 48000, 512)
    erb_fb, erb_inv = dfn_model._build_erb_fb(1024, 48000, 32)
    fwd = np.ascontiguousarray(erb_fb.T.numpy(), dtype=np.float32)
    inv = np.ascontiguousarray(erb_inv.numpy(), dtype=np.float32)
    rng = np.random.default_rng(937)
    frames = 17  # wraps the raw-spectrum ring (5 to 7 slots) more than twice
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
    # t-MASK timing. No trained checkpoint is necessary for arithmetic parity.
    torch.manual_seed(54)
    model = dfn_model.DeepFilterNet3(**{
        **dfn_train.read_model_config(cfg),
        'mask_lookahead': mask_lookahead, 'df_lookahead': df_lookahead,
    }).eval()
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
    flush_mask = np.ones(32, np.float32)
    flush_coefficients = np.zeros_like(coefficients[0])
    delay = max(mask_lookahead, df_lookahead)
    output = []
    try:
        # Zero spectra flush the last `delay` frames, as the zero padding of
        # training does; the heads of those padded frames feed no real output.
        for t in range(frames + delay):
            r, i = (real[t], imag[t]) if t < frames else (zero, zero)
            out_r, out_i = np.empty(513, np.float32), np.empty(513, np.float32)
            index = ctypes.c_longlong(-77)
            ready = t >= mask_lookahead
            head = t - mask_lookahead
            real_head = ready and head < frames
            valid = lib.parity_stream(
                state, _fp(r), _fp(i), int(ready),
                (_fp(masks[head]) if real_head else _fp(flush_mask))
                if ready else None,
                (_fp(coefficients[head]) if real_head
                 else _fp(flush_coefficients)) if ready else None,
                atten_lim, _fp(out_r), _fp(out_i), ctypes.byref(index))
            assert valid == int(t >= delay)
            if valid:
                assert index.value == t - delay
                output.append(out_r + 1j * out_i)
        assert len(output) == frames
    finally:
        lib.parity_destroy(state)
    np.testing.assert_allclose(np.stack(output, axis=1), expected.numpy(),
                               rtol=4e-5, atol=8e-6)


def test_dfn3_aligned_compose_matches_python(c_dfn3):
    lib = c_dfn3[0]
    mask_lookahead, df_lookahead = c_dfn3[2]
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    model = dfn_model.DeepFilterNet3(**{
        **dfn_train.read_model_config(cfg),
        'mask_lookahead': mask_lookahead, 'df_lookahead': df_lookahead,
    }).eval()
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
            assert valid == int(t >= df_lookahead)
            if valid:
                output.append(out_r + 1j * out_i)
    finally:
        lib.parity_destroy(state)
    # Output is the frame df_lookahead behind the newest input.
    np.testing.assert_allclose(np.stack(output, axis=1),
                               expected[:, :frames - df_lookahead].numpy(),
                               rtol=4e-5, atol=8e-6)


@pytest.fixture(scope='module')
def pool_drivers(tmp_path_factory):
    """The stream driver for every C-built pair (one SIMD setting: the pool
    identity does not depend on it)."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    archive = dfn_pairs.audio_common_lib(1)
    drivers = {}
    for pair in LOOKAHEAD_PAIRS:
        work = tmp_path_factory.mktemp('dfn3_pool_%d%d' % pair)
        driver = work / 'dfn3_stream'
        subprocess.run(
            ['cc', *dfn_pairs.house_cflags(1),
             *dfn_pairs.defines(FAMILY, pair),
             str(AINR / 'tests/test_dfn3_stream.c'),
             *[str(dfn_pairs.source_path(FAMILY, kind))
               for kind in ('process', 'model_io', 'prepost')],
             archive, '-lm', '-o', str(driver)],
            check=True, capture_output=True)
        drivers[pair] = driver
    return drivers


def test_a_pool_identity_from_another_lookahead_pair_is_refused(
        tmp_path, pool_drivers):
    """The pool identity (``build_flags_hash``) carries the lookahead pair, and
    ``init_ex`` acts on it: a host built for one pair refuses the memory
    request of a build for another, outright and also when the dumped request
    is given this build's size and alignment (identity alone). Pairs whose
    rings differ in size differ in ``bytes`` too; the hash is what separates
    those that do not."""
    dumps, hashes = {}, set()
    for pair, exe in pool_drivers.items():
        path = tmp_path / ('req_%d%d.bin' % pair)
        out = subprocess.run([str(exe), 'req_dump', str(path)], check=True,
                             capture_output=True, text=True)
        dumps[pair] = path
        hashes.add(out.stdout.split('hash=')[1].split()[0])
    assert len(hashes) == len(pool_drivers)
    for host, exe in pool_drivers.items():
        for pool, path in dumps.items():
            for mode in ((1,) if host == pool else (0, 2)):
                done = subprocess.run(
                    [str(exe), 'req_check', str(path), str(mode)],
                    capture_output=True, text=True)
                assert done.returncode == 0, (host, pool, mode, done.stderr)
