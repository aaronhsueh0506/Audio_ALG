"""DFN3 C public lifecycle and Python feature/compose parity.

Each test runs for both scalar and SIMD builds. The lookahead pair is a run-time
field of the grid, so one shared library (process + model I/O behind a thin
wrapper) serves every pair per SIMD setting, and its grid is built from
``config.ini`` here. The stream driver ``test_dfn3_stream.c`` takes the pair as a
compile-time preset (-DDFN3_MASK_LOOKAHEAD, -DDFN3_DF_LOOKAHEAD; the shipped pair is
also its default), so it is one compile per pair in ``dfn_pairs.c_build_pairs``: it
exercises the public pool/accelerator boundary and audiolib framing. Compiler flags
are the Makefile's.
"""
import configparser
import ctypes
import re
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
# dfn3_*.c/.h moved to lib/ainr/dfn3/ in the single-model-release restructuring; DFN (still
# AINR/DeepFilterNet3, the Python training dir) now holds only config.ini and a forwarding stub.
DFN3_C_DIR = AINR.parent / 'lib' / 'ainr' / 'dfn3'
sys.path.insert(0, str(AINR.parent))
from AINR.DeepFilterNet3 import export_onnx as dfn_export  # noqa: E402
from AINR.DeepFilterNet3 import model as dfn_model  # noqa: E402
from AINR.DeepFilterNet3 import train as dfn_train  # noqa: E402


_WRAPPER = r'''
#include <stdlib.h>
#include <string.h>
#include "DeepFilterNet3/dfn3_prepost.h"

typedef struct { DFN3Grid grid; DFN3State state; void *block; } ParityState;
typedef struct { DFN3Grid grid; DFN3ModelIOState io; void *block; } ParityIo;

size_t parity_sizeof_spec(void) { return sizeof(DFN3GridSpec); }
int parity_grid_status(const DFN3GridSpec *spec) {
    DFN3Grid grid;
    return dfn3_grid_make(&grid, spec);
}
void *parity_create(const DFN3GridSpec *spec, const float *fwd, const float *inv) {
    ParityState *s = calloc(1, sizeof(*s));
    size_t bytes;
    if (!s || dfn3_grid_make(&s->grid, spec) != DFN3_GRID_OK) { free(s); return NULL; }
    bytes = dfn3_state_mem_size(&s->grid);
    s->block = aligned_alloc(16u, bytes);
    if (!s->block || dfn3_state_init(&s->state, s->block, bytes, &s->grid) != 0) {
        free(s->block); free(s); return NULL;
    }
    dfn3_set_erb_matrices(&s->state, fwd, inv);
    return s;
}
void parity_destroy(void *p) {
    ParityState *s = p;
    if (s) { free(s->block); free(s); }
}
void parity_features(void *p, const float *re, const float *im,
                     float *erb, float *spec) {
    dfn3_compute_features(&((ParityState *)p)->state, re, im, erb, spec);
}
int parity_compose(void *p, const float *re, const float *im,
                   const float *mask, const float *coefs,
                   float *out_re, float *out_im) {
    return dfn3_enhance_spectrum(&((ParityState *)p)->state, re, im, mask, coefs, out_re, out_im);
}
void *parity_io_create(const DFN3GridSpec *spec) {
    ParityIo *s = calloc(1, sizeof(*s));
    size_t bytes;
    if (!s || dfn3_grid_make(&s->grid, spec) != DFN3_GRID_OK) { free(s); return NULL; }
    bytes = dfn3_model_io_mem_size(&s->grid);
    s->block = aligned_alloc(16u, bytes);
    if (!s->block || dfn3_model_io_init(&s->io, s->block, bytes, &s->grid,
                                        NULL, 0, NULL, 0) != 0) {
        free(s->block); free(s); return NULL;
    }
    return s;
}
void parity_io_destroy(void *p) {
    ParityIo *s = p;
    if (s) { free(s->block); free(s); }
}
int parity_io_push(void *p, const float *erb, const float *spec) {
    return dfn3_model_io_push_features(&((ParityIo *)p)->io, erb, spec);
}
void parity_io_window(void *p, float *erb_window, float *spec_window) {
    ParityIo *s = p;
    memcpy(erb_window, s->io.erb_window,
           dfn3_prepost_erb_window_elements(&s->grid) * sizeof(float));
    memcpy(spec_window, s->io.spec_window,
           dfn3_prepost_spec_window_elements(&s->grid) * sizeof(float));
}
int parity_stream(void *p, const float *re, const float *im, int valid,
                  const float *mask, const float *coefs, float limit,
                  float *out_re, float *out_im, long long *index) {
    return dfn3_enhance_spectrum_stream(&((ParityState *)p)->state, re, im, valid, mask, coefs,
                                        limit, out_re, out_im, index);
}
'''


class _GridSpec(ctypes.Structure):
    """DFN3GridSpec, field for field."""
    _fields_ = [('sample_rate', ctypes.c_int), ('n_fft', ctypes.c_int),
                ('n_erb', ctypes.c_int), ('df_bins', ctypes.c_int),
                ('df_order', ctypes.c_int), ('mask_lookahead', ctypes.c_int),
                ('df_lookahead', ctypes.c_int),
                ('erb_norm_tau_sec', ctypes.c_float),
                ('erb_norm_init_lo_db', ctypes.c_float),
                ('erb_norm_init_hi_db', ctypes.c_float),
                ('erb_norm_scale_db', ctypes.c_float),
                ('spec_norm_tau_sec', ctypes.c_float),
                ('spec_norm_init_lo', ctypes.c_float),
                ('spec_norm_init_hi', ctypes.c_float)]


def _grid_spec(pair):
    """The shipped model's grid from config.ini, with ``pair`` as its lookahead."""
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    sr = cfg.getint('signal', 'sr')
    model = dfn_train.read_model_config(cfg)
    feature = dfn_train.read_feature_config(cfg, sr, cfg.getint('signal', 'hop_len'))
    # The tests below size their arrays for this grid.
    assert (cfg.getint('signal', 'n_fft'), model['n_erb'], model['df_bins'],
            model['df_order']) == (1024, 32, 96, 5)
    return _GridSpec(
        sample_rate=sr, n_fft=cfg.getint('signal', 'n_fft'),
        n_erb=model['n_erb'], df_bins=model['df_bins'],
        df_order=model['df_order'], mask_lookahead=pair[0],
        df_lookahead=pair[1], erb_norm_tau_sec=feature['erb_tau_sec'],
        erb_norm_init_lo_db=feature['erb_init_lo_db'],
        erb_norm_init_hi_db=feature['erb_init_hi_db'],
        erb_norm_scale_db=feature['erb_scale_db'],
        spec_norm_tau_sec=feature['spec_tau_sec'],
        spec_norm_init_lo=feature['spec_init_lo'],
        spec_norm_init_hi=feature['spec_init_hi'])


def _fp(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _pair_flags(pair):
    """The compile-time lookahead preset of test_dfn3_stream.c."""
    return [f'-DDFN3_MASK_LOOKAHEAD={pair[0]}', f'-DDFN3_DF_LOOKAHEAD={pair[1]}']


_LIBRARIES = {}


def _wrapper_library(simd, tmp_path_factory):
    """The wrapper library, built once per SIMD setting: the pair is an argument
    of its grid, not of the build."""
    if simd in _LIBRARIES:
        return _LIBRARIES[simd]
    archive = dfn_pairs.audio_common_lib(simd)
    work = tmp_path_factory.mktemp(f'dfn3_wrapper_{simd}')
    wrapper = work / 'wrapper.c'
    wrapper.write_text(_WRAPPER)
    library = work / ('dfn3.dylib' if sys.platform == 'darwin' else 'dfn3.so')
    subprocess.run(['cc', *dfn_pairs.house_cflags(simd),
                    '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                    '-fPIC', str(wrapper), str(DFN3_C_DIR / 'dfn3_process.c'),
                    str(DFN3_C_DIR / 'dfn3_model_io.c'), archive, '-lm',
                    '-o', str(library)],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    spec = ctypes.POINTER(_GridSpec)
    vp = ctypes.c_void_p
    lib.parity_sizeof_spec.restype = ctypes.c_size_t
    lib.parity_grid_status.argtypes = [spec]
    lib.parity_grid_status.restype = ctypes.c_int
    lib.parity_create.argtypes = [spec, fp, fp]
    lib.parity_create.restype = vp
    lib.parity_destroy.argtypes = [vp]
    lib.parity_features.argtypes = [vp, fp, fp, fp, fp]
    lib.parity_compose.argtypes = [vp, fp, fp, fp, fp, fp, fp]
    lib.parity_compose.restype = ctypes.c_int
    lib.parity_stream.argtypes = [vp, fp, fp, ctypes.c_int, fp, fp,
                                  ctypes.c_float, fp, fp,
                                  ctypes.POINTER(ctypes.c_longlong)]
    lib.parity_stream.restype = ctypes.c_int
    lib.parity_io_create.argtypes = [spec]
    lib.parity_io_create.restype = vp
    lib.parity_io_destroy.argtypes = [vp]
    lib.parity_io_push.argtypes = [vp, fp, fp]
    lib.parity_io_push.restype = ctypes.c_int
    lib.parity_io_window.argtypes = [vp, fp, fp]
    # A field-order slip in the ctypes mirror would be silent undefined behaviour.
    assert ctypes.sizeof(_GridSpec) == lib.parity_sizeof_spec()
    _LIBRARIES[simd] = lib
    return lib


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
    flags = ['cc', *dfn_pairs.house_cflags(simd), *_pair_flags(pair)]
    driver = work / 'dfn3_stream'
    subprocess.run(flags + [str(AINR / 'tests/test_dfn3_stream.c')]
                   + [str(dfn_pairs.source_path(FAMILY, kind)) for kind in
                      ('process', 'model_io', 'prepost')]
                   + [archive, '-lm', '-o', str(driver)],
                   check=True, capture_output=True)
    return _wrapper_library(simd, tmp_path_factory), driver, pair


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
    io = lib.parity_io_create(ctypes.byref(_grid_spec(c_dfn3[2])))
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
        lib.parity_io_destroy(io)
    assert len(seen) == len(wanted) == frames
    for t, ((got_erb, got_spec), (want_erb, want_spec)) in enumerate(zip(seen, wanted)):
        assert np.array_equal(got_erb, want_erb[0, 0].numpy()), t
        assert np.array_equal(got_spec, want_spec[0].numpy()), t


def _grid_status_codes():
    """``{'OK': 0, 'ERR_MASK_LOOKAHEAD': -7, ...}`` read from the library's own
    ``DFN3_GRID_*`` enum, so a test names a refusal instead of a number."""
    text = (DFN3_C_DIR / 'dfn3_process.h').read_text()
    codes = {name: int(value) for name, value in re.findall(
        r'DFN3_GRID_(\w+)\s*=\s*(-?\d+)', text)}
    assert {'OK', 'ERR_MASK_LOOKAHEAD', 'ERR_DF_LOOKAHEAD'} <= set(codes), codes
    return codes


@pytest.mark.parametrize('pair, reason', [
    ((3, 0), 'ERR_MASK_LOOKAHEAD'), ((3, 3), 'ERR_MASK_LOOKAHEAD'),
    ((-1, 0), 'ERR_MASK_LOOKAHEAD'), ((0, -1), 'ERR_DF_LOOKAHEAD'),
    ((1, 5), 'ERR_DF_LOOKAHEAD'), ((2, 5), 'ERR_DF_LOOKAHEAD')])
def test_dfn3_grid_refuses_pairs_the_trainer_would_not_accept(
        tmp_path_factory, pair, reason):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    lib = _wrapper_library(1, tmp_path_factory)
    codes = _grid_status_codes()
    assert lib.parity_grid_status(ctypes.byref(_grid_spec(pair))) == codes[reason]
    # The same grid accepts the shipped pair.
    assert lib.parity_grid_status(
        ctypes.byref(_grid_spec(SHIPPED_PAIR))) == codes['OK']


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
    spec = _grid_spec(c_dfn3[2])
    state = lib.parity_create(ctypes.byref(spec), _fp(fwd), _fp(inv))
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
    state = lib.parity_create(ctypes.byref(spec), _fp(fwd), _fp(inv))
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
    state = lib.parity_create(ctypes.byref(_grid_spec(c_dfn3[2])), _fp(fwd), _fp(inv))
    assert state
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
             *_pair_flags(pair),
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
