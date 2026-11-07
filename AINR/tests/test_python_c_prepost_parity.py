"""Numeric parity between shipped AINR C pre/post-processing and Python.

The accelerator owns only learned operators. These tests cross the actual
host boundary: DFN2 features and composition, plus GTCRN ERB features and CRM
synthesis. The existing C executable remains responsible for lifecycle,
state-commit, WOLA, and SIMD/scalar checks.

The DFN2 library holds no grid: the wrapper below builds a ``DFN2Grid`` from the
numbers Python hands it (the order, the lookahead pair and the feature
normaliser settings of ``config.ini``), so the same compiled library is compared
with the Python reference on the shipped grid, on the 32 kHz and 16 kHz grids,
and on several lookahead pairs. The GTCRN wrapper builds its grid from
``GTCRN/config.ini`` the same way.

The scale of the framing is the application's: the PCM-start cases at the end of
the file drive audiolib's AcStft / AcIstft at ``grid.stft_scale`` /
``grid.stft_unscale`` from one PCM clip and compare with
``torch.stft(normalized=True)`` and ``torch.istft(normalized=True)``, for DFN2 and
DFN3.
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

import dfn_pairs


AINR = Path(__file__).resolve().parents[1]
ROOT = AINR.parent.parent
AC = ROOT / 'audiolib' / 'common'
DFN = AINR / 'DeepFilterNet2'
GT = AINR / 'GTCRN'
# dfn2_process.c/gtcrn_process.c moved to lib/ainr/{dfn2,gtcrn}/ in the single-model-release
# restructuring; DFN/GT (still the Python training dirs) now hold only config.ini/model.py and
# forwarding stubs for the headers.
DFN2_C_DIR = AINR.parent / 'lib' / 'ainr' / 'dfn2'
GTCRN_C_DIR = AINR.parent / 'lib' / 'ainr' / 'gtcrn'


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

typedef struct { DFN2State state; void *block; } ParityDfn;

/* The grid is an argument: the library holds none. NULL when the library
 * refuses it (dfn2_grid_make or dfn2_state_init). */
void *parity_dfn_create(const float *fwd, const float *inv, int sample_rate,
                        int n_fft, int n_erb, int df_bins, int df_order,
                        int mask_lookahead, int df_lookahead,
                        float erb_tau, float erb_lo, float erb_hi, float erb_scale,
                        float spec_tau, float spec_lo, float spec_hi) {
    DFN2GridSpec spec = {
        .sample_rate = sample_rate, .n_fft = n_fft, .n_erb = n_erb,
        .df_bins = df_bins, .df_order = df_order,
        .mask_lookahead = mask_lookahead, .df_lookahead = df_lookahead,
        .frames_per_call = 1,
        .erb_norm_tau_sec = erb_tau, .erb_norm_init_lo_db = erb_lo,
        .erb_norm_init_hi_db = erb_hi, .erb_norm_scale_db = erb_scale,
        .spec_norm_tau_sec = spec_tau, .spec_norm_init_lo = spec_lo,
        .spec_norm_init_hi = spec_hi,
    };
    DFN2Grid grid;
    ParityDfn *host;
    size_t bytes;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) return NULL;
    host = (ParityDfn *)calloc(1, sizeof(*host));
    if (!host) return NULL;
    bytes = dfn2_state_mem_size(&grid);
    host->block = aligned_alloc(16, bytes);
    if (!host->block || dfn2_state_init(&host->state, host->block, bytes, &grid) != 0) {
        free(host->block);
        free(host);
        return NULL;
    }
    dfn2_set_erb_matrices(&host->state, fwd, inv);
    return host;
}
void parity_dfn_destroy(void *host) {
    if (!host) return;
    free(((ParityDfn *)host)->block);
    free(host);
}
void parity_dfn_features(void *host, const float *re, const float *im,
                         float *erb, float *spec) {
    dfn2_compute_features(&((ParityDfn *)host)->state, re, im, erb, spec);
}
int parity_dfn_compose(void *host, const float *re, const float *im,
                       const float *mask, const float *coefs, float alpha,
                       float *out_re, float *out_im) {
    return dfn2_enhance_spectrum(&((ParityDfn *)host)->state, re, im, mask, coefs,
                                 alpha, out_re, out_im);
}

typedef struct { GTCRNProcess *p; void *pool; } ParityGt;

/* No state registered: model_input / model_output need the grid and the tables only. */
void *parity_gt_create(const float *fwd, const float *inv, int n_fft, int erb_low,
                       int erb_high_bands) {
    GTCRNGridSpec spec = { .n_fft = n_fft, .erb_low_bins = erb_low,
                           .erb_high_bands = erb_high_bands };
    GTCRNGrid grid;
    GTCRNProcessConfig cfg;
    GTCRNProcessMemReq req;
    ParityGt *host;
    if (gtcrn_grid_make(&grid, &spec) != GTCRN_GRID_OK ||
        gtcrn_process_config_defaults(&cfg, &grid) != 0) return NULL;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    if (gtcrn_process_get_mem_size(&cfg, &req) != 0) return NULL;
    host = (ParityGt *)calloc(1, sizeof(*host));
    if (!host) return NULL;
    host->pool = aligned_alloc(16, ((size_t)req.bytes + 15u) & ~(size_t)15u);
    host->p = host->pool ? gtcrn_process_init(host->pool, (size_t)req.bytes, &cfg) : NULL;
    if (!host->p) {
        free(host->pool);
        free(host);
        return NULL;
    }
    return host;
}
void parity_gt_destroy(void *host) {
    if (!host) return;
    free(((ParityGt *)host)->pool);
    free(host);
}
void parity_gt_input(void *host, const float *spectrum, float *mag, float *re,
                     float *im) {
    gtcrn_model_input(((ParityGt *)host)->p, spectrum, mag, re, im);
}
void parity_gt_output(void *host, const float *mask, const float *spectrum,
                      float *enhanced) {
    gtcrn_model_output(((ParityGt *)host)->p, mask, spectrum, enhanced);
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
         '-I', str(DFN), '-I', str(GT), '-I', str(AC / 'include'), '-I', str(AC),
         str(wrapper), str(DFN2_C_DIR / 'dfn2_process.c'),
         str(GTCRN_C_DIR / 'gtcrn_process.c'), archive, '-lm', '-o', str(library)],
        check=True, capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.parity_dfn_create.argtypes = ([fp, fp] + [ctypes.c_int] * 7 +
                                      [ctypes.c_float] * 7)
    lib.parity_dfn_create.restype = ctypes.c_void_p
    lib.parity_dfn_destroy.argtypes = [ctypes.c_void_p]
    lib.parity_dfn_features.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
    lib.parity_dfn_compose.argtypes = [
        ctypes.c_void_p, fp, fp, fp, fp, ctypes.c_float, fp, fp]
    lib.parity_dfn_compose.restype = ctypes.c_int
    lib.parity_gt_create.argtypes = [fp, fp] + [ctypes.c_int] * 3
    lib.parity_gt_create.restype = ctypes.c_void_p
    lib.parity_gt_destroy.argtypes = [ctypes.c_void_p]
    lib.parity_gt_input.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
    lib.parity_gt_output.argtypes = [ctypes.c_void_p, fp, fp, fp]
    return lib


def _fp(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _shipped_config():
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    return cfg


_FEATURE_KEYS = ('erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db',
                 'erb_norm_scale_db', 'spec_norm_tau_sec', 'spec_norm_init_lo',
                 'spec_norm_init_hi')


def _spec_tail(cfg):
    """The grid spec's df_order and the seven normaliser fields, from the config
    the Python reference is built with: the C side runs the numbers the trainer
    reads."""
    return (cfg.getint('model', 'df_order'),
            *[cfg.getfloat('feature', key) for key in _FEATURE_KEYS])


def _shipped_grid():
    """The grid config.ini says: (sr, n_fft, n_erb, df_bins, pair)."""
    cfg = _shipped_config()
    return (cfg.getint('signal', 'sr'), cfg.getint('signal', 'n_fft'),
            cfg.getint('model', 'n_erb'), cfg.getint('model', 'df_bins'),
            (cfg.getint('model', 'mask_lookahead'),
             cfg.getint('model', 'df_lookahead')))


# (sr, n_fft, n_erb, df_bins, (mask_lookahead, df_lookahead)): the shipped grid
# from config.ini, the 32 kHz and 16 kHz grids of the test grid presets
# (tests/dfn2_test_grid.h), and lookahead pairs on the shipped grid that move the
# deep-filter window (df_lookahead 0, 2, 4).
_SHIPPED = _shipped_grid()
DFN2_CASES = [
    pytest.param(*_SHIPPED, id='shipped'),
    pytest.param(32000, 512, 28, 96, (1, 1), id='32k'),
    pytest.param(16000, 256, 24, 96, (1, 1), id='16k'),
    pytest.param(*_SHIPPED[:4], (0, 0), id='48k-pair0_0'),
    pytest.param(*_SHIPPED[:4], (2, 4), id='48k-pair2_4'),
    pytest.param(16000, 256, 24, 96, (1, 2), id='16k-pair1_2'),
]


@pytest.mark.parametrize('sr,n_fft,n_erb,df_bins,pair', DFN2_CASES)
def test_dfn2_features_and_compose_match_python(c_prepost, sr, n_fft, n_erb,
                                                df_bins, pair):
    dfn_model, dfn_train = _load_dfn_modules()
    cfg = dfn_pairs.dfn2_grid_config(sr, n_fft, n_erb, df_bins, pair)
    n_bins = n_fft // 2 + 1
    feature_cfg = dfn_train.read_feature_config(cfg, sr, n_fft // 2)
    erb_fb, erb_inv = dfn_model._build_erb_fb(n_fft, sr, n_erb)
    fwd = np.ascontiguousarray(erb_fb.T.numpy(), dtype=np.float32)
    inv = np.ascontiguousarray(erb_inv.numpy(), dtype=np.float32)

    order, *normalisers = _spec_tail(cfg)

    def create():
        return c_prepost.parity_dfn_create(
            _fp(fwd), _fp(inv), sr, n_fft, n_erb, df_bins, order, *pair,
            *normalisers)

    rng = np.random.default_rng(41)
    frames = 9
    real = rng.normal(0, 0.15, (frames, n_bins)).astype(np.float32)
    imag = rng.normal(0, 0.15, (frames, n_bins)).astype(np.float32)
    spectrum = torch.complex(torch.from_numpy(real.T)[None],
                             torch.from_numpy(imag.T)[None])
    _, py_erb, py_spec, _ = dfn_train.extract_dfn2_features(
        spectrum, erb_fb, df_bins, feature_cfg)

    state = create()
    assert state
    try:
        c_erb = np.empty((frames, n_erb), np.float32)
        c_spec = np.empty((frames, 2 * df_bins), np.float32)
        for frame in range(frames):
            c_prepost.parity_dfn_features(
                state, _fp(real[frame]), _fp(imag[frame]),
                _fp(c_erb[frame]), _fp(c_spec[frame]))
    finally:
        c_prepost.parity_dfn_destroy(state)

    np.testing.assert_allclose(c_erb, py_erb[0, 0].numpy(),
                               rtol=2e-5, atol=3e-6)
    expected_spec = py_spec[0].permute(1, 0, 2).reshape(
        frames, 2 * df_bins).numpy()
    np.testing.assert_allclose(c_spec, expected_spec,
                               rtol=2e-5, atol=3e-6)

    model = dfn_model.DeepFilterNet2(
        **dfn_train.read_model_config(cfg)).eval()
    mask = torch.sigmoid(torch.randn(1, 1, frames, n_erb))
    coefs = torch.randn(1, frames, df_bins, 10) * 0.1
    alpha = torch.sigmoid(torch.randn(1, frames, 1))
    with torch.no_grad():
        expected = model.compose(spectrum, mask, coefs, alpha)[0]

    state = create()
    assert state
    c_out = []
    try:
        for frame in range(frames):
            out_re = np.empty(n_bins, np.float32)
            out_im = np.empty(n_bins, np.float32)
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
    # The aligned entry emits the frame df_lookahead behind the newest input.
    np.testing.assert_allclose(
        got, expected[:, :frames - model.df_lookahead].numpy(),
        rtol=4e-5, atol=8e-6)


def test_dfn2_library_refuses_a_grid_it_cannot_run(c_prepost):
    """The grid is run-time data, so a bad one is refused by the library
    (dfn2_grid_make) rather than by the build. The same call with a valid grid
    succeeds, so the refusals are not an artefact of the wrapper; sizes the
    library has no reason to bound are accepted."""
    sr, n_fft, n_erb, df_bins, pair = _SHIPPED
    order, *normalisers = _spec_tail(_shipped_config())
    n_bins = n_fft // 2 + 1
    # The tables are borrowed and read at the grid's extent: room for the widest
    # band count tried below.
    widest = 40
    fwd = np.zeros((n_bins, widest), np.float32)
    inv = np.zeros((widest, n_bins), np.float32)

    def create(**changes):
        grid = dict(sr=sr, n_fft=n_fft, n_erb=n_erb, df_bins=df_bins,
                    order=order, mask=pair[0], df=pair[1])
        grid.update(changes)
        state = c_prepost.parity_dfn_create(
            _fp(fwd), _fp(inv), grid['sr'], grid['n_fft'], grid['n_erb'],
            grid['df_bins'], grid['order'], grid['mask'], grid['df'],
            *normalisers)
        if state:
            c_prepost.parity_dfn_destroy(state)
        return bool(state)

    assert create()
    assert not create(n_erb=n_erb + 2)             # not a multiple of four
    assert not create(n_fft=n_fft + 2)             # not a power of two
    assert not create(df_bins=n_bins)              # the deep filter owns every bin
    assert not create(mask=3)
    assert not create(df=order)                    # deep-filter lookahead < order
    assert not create(sr=0)
    assert create(n_erb=36)                        # no band-count capacity
    assert create(order=7, df=5)                   # the order is the spec's


def test_gtcrn_host_boundary_matches_python(c_prepost):
    gt_model = _load('gtcrn_parity_model', GT / 'model.py')
    gt_stream = _load('gtcrn_parity_stream', GT / 'stream_model.py')
    cfg = configparser.ConfigParser()
    cfg.read(GT / 'config.ini')
    n_fft = cfg.getint('signal', 'n_fft')
    erb_low = cfg.getint('model', 'erb_subband_1')
    erb_high = cfg.getint('model', 'erb_subband_2')
    n_bins = n_fft // 2 + 1
    model = gt_model.GTCRN(erb_low, erb_high, nfft=n_fft).eval()
    fwd = np.ascontiguousarray(
        model.erb.erb_fc.weight.detach().numpy().T, dtype=np.float32)
    inv = np.ascontiguousarray(
        model.erb.ierb_fc.weight.detach().numpy().T, dtype=np.float32)
    rng = np.random.default_rng(73)
    spectrum = rng.normal(0, 0.2, (n_bins, 2)).astype(np.float32)
    spectrum_t = torch.from_numpy(spectrum)[None, :, None, :]
    with torch.no_grad():
        py_mag, py_re, py_im = gt_stream.stream_features(model, spectrum_t)

    bands = erb_low + erb_high
    host = c_prepost.parity_gt_create(_fp(fwd), _fp(inv), n_fft, erb_low,
                                      erb_high)
    assert host
    c_mag = np.empty(bands, np.float32)
    c_re = np.empty(bands, np.float32)
    c_im = np.empty(bands, np.float32)
    c_prepost.parity_gt_input(
        host, _fp(spectrum), _fp(c_mag), _fp(c_re), _fp(c_im))
    np.testing.assert_allclose(c_mag, py_mag[0, :, 0].numpy(),
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(c_re, py_re[0, :, 0].numpy(),
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(c_im, py_im[0, :, 0].numpy(),
                               rtol=2e-5, atol=2e-6)

    mask = rng.normal(0, 0.4, (bands, 2)).astype(np.float32)
    with torch.no_grad():
        expected = gt_stream.host_synthesis(
            model, torch.from_numpy(mask)[None, :, None, :], spectrum_t)
    enhanced = np.empty((n_bins, 2), np.float32)
    c_prepost.parity_gt_output(host, _fp(mask), _fp(spectrum), _fp(enhanced))
    c_prepost.parity_gt_destroy(host)
    np.testing.assert_allclose(enhanced, expected[0, :, 0].numpy(),
                               rtol=3e-5, atol=3e-6)


def test_dfn2_process_compiles_alone_without_the_class_symbols(tmp_path):
    """dfn2_process.c is the parity unit and must stay linkable on its own:
    the accelerator callback boundary (DFN2Model, dfn2_model_run_frame) lives
    in dfn2_prepost, and a future move of it into dfn2_process.h would drag
    dfn2_model_io/dfn2_prepost into this build. The object's undefined
    symbols must not name either sibling."""
    cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if cc is None or shutil.which('nm') is None:
        pytest.skip('C build tools are unavailable')
    obj = tmp_path / 'dfn2_process.o'
    subprocess.run(
        [cc, '-c', '-O2', '-std=c11', '-ffp-contract=off', '-fno-math-errno',
         '-I', str(DFN), '-I', str(AC / 'include'), '-I', str(AC), str(DFN2_C_DIR / 'dfn2_process.c'),
         '-o', str(obj)],
        check=True, capture_output=True)
    undefined = subprocess.run(['nm', '-u', str(obj)], check=True,
                               capture_output=True, text=True).stdout
    assert 'dfn2_prepost' not in undefined
    assert 'dfn2_model_io' not in undefined
    assert 'DFN2Model' not in undefined


# ---- PCM start: the framing scale is the application's ----------------------
# The libraries hold no STFT. Everything above shares one spectrum between the two
# sides, so a wrong spec_scale / ispec_scale in the application would pass it.
# These cases start from PCM: AcStft / AcIstft (tests/ainr_test_hop.h) at the
# scales the grid reports, against torch.stft / torch.istft with normalized=True.

_FRAMING_PROLOGUE = r'''
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_prepost.h"
#include "DeepFilterNet3/dfn3_prepost.h"
#include "tests/ainr_test_hop.h"
'''

# @N@ = 2 or 3; @FRAMES@ is DFN2's frames_per_call field. `g` = {sample_rate, n_fft,
# n_erb, df_bins, df_order, mask_lookahead, df_lookahead}, `norm` = the seven
# [feature] settings.
_FRAMING_TEMPLATE = r'''
typedef struct {
    DFN@N@Grid grid;
    DFN@N@Prepost *p;
    AinrTestHop hop;
    float *re, *im;
} Framing@N@;

void framing_dfn@N@_destroy(void *host) {
    Framing@N@ *f = (Framing@N@ *)host;
    if (!f) return;
    dfn@N@_prepost_destroy(f->p);
    ainr_test_hop_close(&f->hop);
    free(f->re);
    free(f);
}

/* NULL when the library refuses the grid or the instance. */
void *framing_dfn@N@_create(const int *g, const float *norm, const float *fwd,
                            const float *inv) {
    DFN@N@GridSpec spec = {
        .sample_rate = g[0], .n_fft = g[1], .n_erb = g[2], .df_bins = g[3],
        .df_order = g[4], .mask_lookahead = g[5], .df_lookahead = g[6],
@FRAMES@        .erb_norm_tau_sec = norm[0], .erb_norm_init_lo_db = norm[1],
        .erb_norm_init_hi_db = norm[2], .erb_norm_scale_db = norm[3],
        .spec_norm_tau_sec = norm[4], .spec_norm_init_lo = norm[5],
        .spec_norm_init_hi = norm[6],
    };
    DFN@N@PrepostConfig cfg;
    Framing@N@ *f = (Framing@N@ *)calloc(1, sizeof(*f));
    if (!f) return NULL;
    if (dfn@N@_grid_make(&f->grid, &spec) != DFN@N@_GRID_OK ||
        dfn@N@_prepost_config_defaults(&cfg, &f->grid) != 0) goto fail;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    f->p = dfn@N@_prepost_create(&cfg);
    f->re = (float *)malloc(2u * (size_t)f->grid.n_bins * sizeof(float));
    if (!f->p || !f->re) goto fail;
    f->im = f->re + f->grid.n_bins;
    if (ainr_test_hop_open(&f->hop, f->grid.n_fft, f->grid.stft_scale,
                           f->grid.stft_unscale) != 0) goto fail;
    return f;
fail:
    framing_dfn@N@_destroy(f);
    return NULL;
}

/* One hop of PCM through AcStft and pre_process_freq: its count, or -1 on a
 * refusal. When a frame opens, the graph's two windows are copied out and the
 * frame is closed. */
int framing_dfn@N@_push(void *host, const float *pcm_hop, float *erb_window,
                        float *spec_window) {
    Framing@N@ *f = (Framing@N@ *)host;
    DFN@N@PrepostInputs in;
    DFN@N@PrepostOutputs out;
    int n;
    ainr_test_hop_analysis(&f->hop, pcm_hop, f->re, f->im);
    n = dfn@N@_prepost_pre_process_freq(f->p, f->re, f->im);
    if (n != 1) return n;
    if (dfn@N@_prepost_frame_open(f->p, &in, &out) != 0 ||
        in.erb_window_elements != 3u * (size_t)f->grid.n_erb ||
        in.spec_window_elements != 6u * (size_t)f->grid.df_bins) return -1;
    memcpy(erb_window, in.erb_window, in.erb_window_elements * sizeof(float));
    memcpy(spec_window, in.spec_window, in.spec_window_elements * sizeof(float));
    return dfn@N@_prepost_frame_bypass(f->p) == 0 ? 1 : -1;
}

/* One spectrum through AcIstft: one hop of samples. */
void framing_dfn@N@_synthesize(void *host, const float *re, const float *im,
                               float *out_hop) {
    ainr_test_hop_synthesis(&((Framing@N@ *)host)->hop, re, im, out_hop);
}
'''


def _framing_wrapper():
    body = []
    for n, frames in ((2, '        .frames_per_call = 1,\n'), (3, '')):
        body.append(_FRAMING_TEMPLATE.replace('@N@', str(n))
                    .replace('@FRAMES@', frames))
    return _FRAMING_PROLOGUE + ''.join(body)


@pytest.fixture(scope='module', params=(0, 1),
                ids=lambda simd: 'simd' if simd else 'scalar')
def c_framing(request, tmp_path_factory):
    """DFN2 and DFN3 pre/post classes plus the AcStft/AcIstft hop, once per SIMD
    setting, driven through ctypes."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    simd = request.param
    archive = dfn_pairs.audio_common_lib(simd)
    work = tmp_path_factory.mktemp(f'framing_{simd}')
    wrapper = work / 'framing.c'
    wrapper.write_text(_framing_wrapper(), encoding='utf-8')
    library = work / ('framing.dylib' if sys.platform == 'darwin'
                      else 'framing.so')
    sources = [str(dfn_pairs.source_path(family, kind))
               for family in dfn_pairs.FAMILIES
               for kind in ('process', 'model_io', 'prepost')]
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(simd),
         '-dynamiclib' if sys.platform == 'darwin' else '-shared', '-fPIC',
         str(wrapper), *sources, archive, '-lm', '-o', str(library)],
        check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    fp, ip, vp = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                  ctypes.c_void_p)
    for n in (2, 3):
        create = getattr(lib, f'framing_dfn{n}_create')
        create.argtypes = [ip, fp, fp, fp]
        create.restype = vp
        getattr(lib, f'framing_dfn{n}_destroy').argtypes = [vp]
        push = getattr(lib, f'framing_dfn{n}_push')
        push.argtypes = [vp, fp, fp, fp]
        push.restype = ctypes.c_int
        getattr(lib, f'framing_dfn{n}_synthesize').argtypes = [vp, fp, fp, fp]
    return lib


def _family_config(family, sr, n_fft, n_erb, df_bins, pair):
    """The family's config.ini on another signal grid: the Python reference is
    built for it as the trainer would."""
    cfg = configparser.ConfigParser()
    cfg.read(dfn_pairs.family_dir(family) / 'config.ini')
    for section, key, value in (
            ('signal', 'sr', sr), ('signal', 'n_fft', n_fft),
            ('signal', 'win_len', n_fft), ('signal', 'hop_len', n_fft // 2),
            ('model', 'n_erb', n_erb), ('model', 'df_bins', df_bins),
            ('model', 'mask_lookahead', pair[0]),
            ('model', 'df_lookahead', pair[1])):
        cfg.set(section, key, str(value))
    return cfg


def _framing_cases():
    """The shipped 48 kHz row of each family's config.ini and the 16 kHz row of
    the board samples (lookahead 0/0)."""
    cases = []
    for family in dfn_pairs.FAMILIES:
        cfg = configparser.ConfigParser()
        cfg.read(dfn_pairs.family_dir(family) / 'config.ini')
        shipped = (cfg.getint('signal', 'sr'), cfg.getint('signal', 'n_fft'),
                   cfg.getint('model', 'n_erb'), cfg.getint('model', 'df_bins'),
                   dfn_pairs.shipped_pair(family))
        for row in (shipped, (16000, 256, 24, 96, (0, 0))):
            cases.append(pytest.param(
                family, *row, id=f'dfn{family[-1]}-{row[0] // 1000}k'))
    return cases


FRAMING_CASES = _framing_cases()
_FRAMING_HOPS = 12


class _FramingReference:
    """The family's Python path for one grid, and the C instance built from the
    same numbers. `host` is open inside the `with`."""

    def __init__(self, lib, family, sr, n_fft, n_erb, df_bins, pair):
        self.lib, self.n_fft, self.n_erb, self.df_bins = lib, n_fft, n_erb, df_bins
        self.n = int(family[-1])
        self.hop, self.n_bins = n_fft // 2, n_fft // 2 + 1
        cfg = _family_config(family, sr, n_fft, n_erb, df_bins, pair)
        self.model = dfn_pairs.import_family_module(family, 'model')
        self.train = dfn_pairs.import_family_module(family, 'train')
        self.feature_cfg = self.train.read_feature_config(cfg, sr, self.hop)
        erb_fb, erb_inv = self.model._build_erb_fb(n_fft, sr, n_erb)
        self.erb_fb = erb_fb
        self.fwd = np.ascontiguousarray(erb_fb.T.numpy(), dtype=np.float32)
        self.inv = np.ascontiguousarray(erb_inv.numpy(), dtype=np.float32)
        order, *normalisers = _spec_tail(cfg)
        self.g = np.array([sr, n_fft, n_erb, df_bins, order, *pair], np.int32)
        self.norm = np.array(normalisers, np.float32)
        self.window = torch.hann_window(n_fft).sqrt()

    def __enter__(self):
        self.host = getattr(self.lib, f'framing_dfn{self.n}_create')(
            self.g.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            _fp(self.norm), _fp(self.fwd), _fp(self.inv))
        assert self.host
        return self

    def __exit__(self, *exc):
        getattr(self.lib, f'framing_dfn{self.n}_destroy')(self.host)

    def push(self, pcm_hop):
        erb = np.empty(3 * self.n_erb, np.float32)
        spec = np.empty(6 * self.df_bins, np.float32)
        count = getattr(self.lib, f'framing_dfn{self.n}_push')(
            self.host, _fp(pcm_hop), _fp(erb), _fp(spec))
        return count, erb.reshape(3, self.n_erb), spec.reshape(2, 3, self.df_bins)

    def synthesize(self, re, im):
        out = np.empty(self.hop, np.float32)
        getattr(self.lib, f'framing_dfn{self.n}_synthesize')(
            self.host, _fp(re), _fp(im), _fp(out))
        return out


@pytest.mark.parametrize('family,sr,n_fft,n_erb,df_bins,pair', FRAMING_CASES)
def test_pcm_through_audiolib_stft_matches_torch_stft_features(
        c_framing, family, sr, n_fft, n_erb, df_bins, pair):
    """PCM -> AcStft(spec_scale = grid.stft_scale) -> pre_process_freq: the graph
    windows equal the family's features of torch.stft(normalized=True). The C
    frame t covers pcm[(t-1)*hop:(t+1)*hop], so the reference is center=False on
    the clip behind one hop of zeros."""
    ref = _FramingReference(c_framing, family, sr, n_fft, n_erb, df_bins, pair)
    pcm = np.random.default_rng(29).normal(
        0, 0.1, _FRAMING_HOPS * ref.hop).astype(np.float32)
    spectrum = torch.stft(
        torch.cat([torch.zeros(ref.hop), torch.from_numpy(pcm)])[None],
        n_fft, ref.hop, n_fft, window=ref.window, center=False,
        normalized=True, return_complex=True)
    assert spectrum.shape[-1] == _FRAMING_HOPS
    extract = getattr(ref.train, f'extract_dfn{ref.n}_features')
    _, py_erb, py_spec, _ = extract(spectrum, ref.erb_fb, df_bins,
                                    ref.feature_cfg)
    counts = []
    with ref:
        for t in range(_FRAMING_HOPS):
            count, erb, spec = ref.push(pcm[t * ref.hop:(t + 1) * ref.hop])
            counts.append(count)
            if count != 1:
                continue
            # Window row j holds the features of frame t - 2 + j.
            first = max(0, 2 - t)
            np.testing.assert_allclose(
                erb[first:], py_erb[0, 0, t - 2 + first:t + 1].numpy(),
                rtol=2e-5, atol=3e-6)
            np.testing.assert_allclose(
                spec[:, first:], py_spec[0, :, t - 2 + first:t + 1].numpy(),
                rtol=2e-5, atol=3e-6)
    # Every hop opens a frame (one frame per call): the first mask_lookahead are warm-up frames
    # whose windows are compared above like any other (zero rows before the stream start).
    assert counts == [1] * _FRAMING_HOPS


@pytest.mark.parametrize('family,sr,n_fft,n_erb,df_bins,pair', FRAMING_CASES)
def test_audiolib_istft_matches_torch_istft_normalized(
        c_framing, family, sr, n_fft, n_erb, df_bins, pair):
    """A spectrum -> AcIstft(ispec_scale = grid.stft_unscale): the hops equal
    torch.istft(normalized=True). center=True drops the first half window, so
    the C hops 1.. are the reference (hop 0 has one frame of overlap)."""
    ref = _FramingReference(c_framing, family, sr, n_fft, n_erb, df_bins, pair)
    rng = np.random.default_rng(31)
    re = rng.normal(0, 0.2, (_FRAMING_HOPS, ref.n_bins)).astype(np.float32)
    im = rng.normal(0, 0.2, (_FRAMING_HOPS, ref.n_bins)).astype(np.float32)
    im[:, 0] = 0.0
    im[:, -1] = 0.0
    expected = torch.istft(
        torch.complex(torch.from_numpy(re.T), torch.from_numpy(im.T))[None],
        n_fft, ref.hop, n_fft, window=ref.window, center=True,
        normalized=True, length=(_FRAMING_HOPS - 1) * ref.hop)[0].numpy()
    with ref:
        hops = [ref.synthesize(re[t], im[t]) for t in range(_FRAMING_HOPS)]
    np.testing.assert_allclose(np.concatenate(hops[1:]), expected,
                               rtol=4e-5, atol=8e-6)
