"""Parity test between train.py / inference.py and process.c on fixed vectors, on the shipped
grid and two others.

The C side is what an application builds: the two raw ERB matrix files read in place, a pool-based
state, audiolib's AcStft/AcIstft owned by the caller, flat float arrays."""

import atexit
import configparser
import ctypes
import math
import os
import pathlib
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Each of the three model projects has its own top-level ``train.py``
# (and ``inference.py``/``model.py``). Under a single pytest session the first one
# imported wins ``sys.modules``, so a sibling project's tests would silently
# exercise the wrong code.  Dropping the cached entries forces the re-import
# to resolve against the ROOT just inserted above.
for _stale in ('train', 'inference', 'model', 'checkpoint_utils'):
    sys.modules.pop(_stale, None)


from train import (  # noqa: E402
    compute_erb_matrix,
    erb_bandborder,
    extract_model_features,
    make_norm_alpha,
    read_feature_config,
    istft,
    stft,
)
from inference import apply_atten_lim  # noqa: E402
from erb_fixture import erb_files  # noqa: E402
from rnnoise_c_build import compile_c  # noqa: E402

DEFAULTS = dict(erb_norm_tau_sec=1.0, erb_norm_init_lo_db=-20.0, erb_norm_init_hi_db=-45.0,
                erb_norm_scale_db=40.0, spec_norm_tau_sec=1.0, spec_norm_init_lo=0.04,
                spec_norm_init_hi=0.008, spec_norm_eps=1e-12)
MOVED = dict(erb_norm_tau_sec=0.5, erb_norm_init_lo_db=-25.0, erb_norm_init_hi_db=-50.0,
             erb_norm_scale_db=35.0, spec_norm_tau_sec=2.0, spec_norm_init_lo=0.03,
             spec_norm_init_hi=0.006, spec_norm_eps=1e-10)

# name, sample rate, n_fft, n_bands, min_bins_per_band, lookahead, spec_max_hz, feature values;
# the first grid is config.ini's own.
GRIDS = [
    ('config.ini', None, None, None, None, None, None, None),
    ('16k/256 moved features', 16000, 256, 18, 2, 0, 4000, MOVED),
    ('48k/1024', 48000, 1024, 32, 2, 1, 4000, DEFAULTS),
]

WRAPPER = r"""
#define _POSIX_C_SOURCE 200809L
#include "rnnoise_test_support.h"

typedef struct { RNNoiseGrid grid; RnnErb erb; RnnPool pool; RnnFraming framing; } Handle;

/* Releases whatever br_create opened; also valid on a partly opened handle. */
void br_destroy(void *p) {
    Handle *h = (Handle *)p;
    rnn_framing_close(&h->framing);
    free(h->pool.pool);
    rnn_erb_free(&h->erb);
    free(h);
}

/* info: status, n_bins, n_bands, spec_bins, hop, pool bytes. NULL on any refusal. */
void *br_create(const char *fwd_path, const char *inv_path, const int *iv, const double *dv,
                int *info) {
    RNNoiseGridSpec spec = {
        .sample_rate = iv[0], .n_fft = iv[1], .n_bands = iv[2], .min_bins_per_band = iv[3],
        .lookahead_frames = iv[4], .spec_max_hz = iv[5],
        .erb_norm_tau_sec = dv[0], .erb_norm_init_lo_db = (float)dv[1],
        .erb_norm_init_hi_db = (float)dv[2], .erb_norm_scale_db = (float)dv[3],
        .spec_norm_tau_sec = dv[4], .spec_norm_init_lo = (float)dv[5],
        .spec_norm_init_hi = (float)dv[6], .spec_norm_eps = (float)dv[7],
    };
    Handle *h = (Handle *)calloc(1, sizeof(*h));
    AcStftConfig sc;
    int status;
    if (!h) return NULL;
    info[0] = rnnoise_grid_make(&h->grid, &spec);
    if (info[0] != 0) { free(h); return NULL; }
    info[1] = h->grid.n_bins; info[2] = h->grid.n_bands;
    info[3] = h->grid.spec_bins; info[4] = h->grid.hop;
    status = rnn_erb_read_grid(fwd_path, inv_path, &h->grid, &h->erb);
    if (status != RNN_ERB_OK) { info[0] = 1000 - status; free(h); return NULL; }
    rnnoise_config_defaults(&h->pool.cfg, &h->grid);
    h->pool.cfg.erb_fwd = h->erb.fwd;
    h->pool.cfg.erb_inv = h->erb.inv;
    if (rnn_pool_open(&h->pool) != 0) { info[0] = -100; br_destroy(h); return NULL; }
    info[5] = (int)h->pool.req.bytes;
    rnn_framing_config(&sc, &h->grid);
    if (rnn_framing_open(&h->framing, &sc) != 0) { info[0] = -101; br_destroy(h); return NULL; }
    return h;
}
void br_alphas(void *p, float *out) {
    Handle *h = (Handle *)p;
    out[0] = h->grid.erb_norm_alpha; out[1] = h->grid.spec_norm_alpha;
    out[2] = h->grid.stft_scale; out[3] = h->grid.stft_unscale;
}
void br_reset(void *p) {
    Handle *h = (Handle *)p;
    rnnoise_reset(h->pool.st); ac_stft_reset(h->framing.stft); ac_istft_reset(h->framing.istft);
}
void br_analysis(void *p, const float *frame, float *re, float *im) {
    ac_stft_frame_planar(((Handle *)p)->framing.stft, frame, re, im);
}
int br_step(void *p, const float *re, const float *im, float *erb, float *spec) {
    return rnnoise_compute_features(((Handle *)p)->pool.st, re, im, erb, spec);
}
void br_states(void *p, float *erb, float *spec) {
    Handle *h = (Handle *)p;
    memcpy(erb, h->pool.st->erb_norm_state, sizeof(float) * (size_t)h->grid.n_bands);
    memcpy(spec, h->pool.st->spec_norm_state, sizeof(float) * (size_t)h->grid.spec_bins);
}
int br_atten(void *p, float *gains, float db) {
    return rnnoise_apply_atten_lim(((Handle *)p)->pool.st, gains, db);
}
int br_expand(void *p, const float *gains, float *bin) {
    return rnnoise_expand_gains(((Handle *)p)->pool.st, gains, bin);
}
int br_apply(void *p, float *re, float *im, const float *bin) {
    return rnnoise_apply_gains(((Handle *)p)->pool.st, re, im, bin);
}
int br_synth(void *p, const float *re, const float *im, float *out) {
    return ac_istft_push_planar(((Handle *)p)->framing.istft, re, im, out);
}
"""

_BRIDGE = {}
fp = ctypes.POINTER(ctypes.c_float)


def build_bridge():
    """Compile the C bridge once per process and keep it mapped.

    The build directory survives until interpreter exit: tearing it down while the shared
    library is still loaded through ctypes would crash the next module import with SIGTRAP
    under pytest, where the process lives on.
    """
    if 'lib' not in _BRIDGE:
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix='rnnoise-feature-parity-'))
        atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
        wrapper = tmpdir / 'feature_bridge.c'
        wrapper.write_text(WRAPPER)
        suffix = '.dylib' if sys.platform == 'darwin' else '.so'
        library = tmpdir / f'feature_bridge{suffix}'
        compile_c([wrapper], library, shared=True)
        lib = ctypes.CDLL(str(library))
        lib.br_create.restype = ctypes.c_void_p
        lib.br_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
        lib.br_destroy.argtypes = [ctypes.c_void_p]
        lib.br_alphas.argtypes = [ctypes.c_void_p, fp]
        lib.br_reset.argtypes = [ctypes.c_void_p]
        lib.br_analysis.argtypes = [ctypes.c_void_p, fp, fp, fp]
        lib.br_step.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
        lib.br_step.restype = ctypes.c_int
        lib.br_states.argtypes = [ctypes.c_void_p, fp, fp]
        lib.br_atten.argtypes = [ctypes.c_void_p, fp, ctypes.c_float]
        lib.br_atten.restype = ctypes.c_int
        lib.br_expand.argtypes = [ctypes.c_void_p, fp, fp]
        lib.br_expand.restype = ctypes.c_int
        lib.br_apply.argtypes = [ctypes.c_void_p, fp, fp, fp]
        lib.br_apply.restype = ctypes.c_int
        lib.br_synth.argtypes = [ctypes.c_void_p, fp, fp, fp]
        lib.br_synth.restype = ctypes.c_int
        _BRIDGE['lib'] = lib
        _BRIDGE['dir'] = tmpdir
    return _BRIDGE['lib']


def P(array):
    return array.ctypes.data_as(fp)


def grid_config(row):
    """(sr, n_fft, n_bands, min_bins, lookahead, spec_max_hz, feature values, ConfigParser)."""
    name, sr, n_fft, n_bands, min_bins, lookahead, spec_max_hz, values = row
    cfg = configparser.ConfigParser()
    if values is None:
        if not cfg.read(ROOT / 'config.ini'):
            raise FileNotFoundError(ROOT / 'config.ini')
        sr = cfg.getint('signal', 'sr')
        n_fft = cfg.getint('signal', 'n_fft')
        n_bands = cfg.getint('signal', 'n_bands')
        min_bins = cfg.getint('signal', 'min_bins_per_band', fallback=2)
        lookahead = cfg.getint('signal', 'lookahead_frames')
        spec_max_hz = cfg.getint('feature', 'spec_max_hz')
        values = {k: cfg.getfloat('feature', k) for k in DEFAULTS}
    else:
        cfg.read_dict({'feature': {k: str(v) for k, v in values.items()},
                       'signal': {'lookahead_frames': str(lookahead), 'n_bands': str(n_bands)}})
        cfg['feature']['spec_max_hz'] = str(spec_max_hz)
    return name, sr, n_fft, n_bands, min_bins, lookahead, spec_max_hz, values, cfg


def deterministic_spectra(n_bins, n_frames=12):
    k = np.arange(n_bins, dtype=np.float32)
    frames = []
    for t in range(n_frames):
        scale = np.float32(0.65 + 0.07 * t)
        re = scale * (0.0008 + 0.00005 * ((k * 17 + t * 3) % 31))
        im = scale * (-0.0004 + 0.00004 * ((k * 11 + t * 5) % 23))
        im[0] = 0.0
        im[-1] = 0.0
        frames.append(re.astype(np.float32) + 1j * im.astype(np.float32))
    return np.stack(frames, axis=0).astype(np.complex64)


def open_case(lib, row, workdir):
    """One grid row: train.py's matrices, their two files and the C handle (caller destroys it)."""
    name, sr, n_fft, n_bands, min_bins, lookahead, spec_max_hz, values, cfg = grid_config(row)
    hop = n_fft // 2
    feature_cfg = read_feature_config(cfg, sr, hop, n_fft, n_fft)
    assert feature_cfg['n_bands'] == n_bands, (name, feature_cfg['n_bands'])
    borders = erb_bandborder(n_bands, sr, n_fft, min_bins)
    fwd = np.ascontiguousarray(compute_erb_matrix(borders, n_fft, mode=0), dtype=np.float32)
    inv = np.ascontiguousarray(compute_erb_matrix(borders, n_fft, mode=1), dtype=np.float32)
    fwd_path = workdir / f'erb_fwd_{sr}_{n_fft}.bin'
    inv_path = workdir / f'erb_inv_{sr}_{n_fft}.bin'
    fwd_bytes, inv_bytes = erb_files(fwd, inv)
    fwd_path.write_bytes(fwd_bytes)
    inv_path.write_bytes(inv_bytes)

    ints = (ctypes.c_int * 6)(sr, n_fft, n_bands, min_bins, lookahead, spec_max_hz)
    doubles = (ctypes.c_double * 8)(*[values[k] for k in DEFAULTS])
    info = (ctypes.c_int * 6)()
    handle = lib.br_create(os.fsencode(fwd_path), os.fsencode(inv_path), ints, doubles, info)
    assert handle, (name, 'create refused', list(info))
    return types.SimpleNamespace(
        name=name, sr=sr, n_fft=n_fft, n_bands=n_bands, hop=hop, n_bins=n_fft // 2 + 1,
        spec_bins=feature_cfg['spec_bins'], values=values, feature_cfg=feature_cfg,
        fwd=fwd, inv=inv, handle=handle, info=info)


def run_grid(lib, row, workdir):
    case = open_case(lib, row, workdir)
    name, sr, n_fft, n_bands, hop, n_bins = (
        case.name, case.sr, case.n_fft, case.n_bands, case.hop, case.n_bins)
    spec_bins, values, feature_cfg = case.spec_bins, case.values, case.feature_cfg
    fwd, inv, handle, info = case.fwd, case.inv, case.handle, case.info
    try:
        assert (info[1], info[2], info[3], info[4]) == (n_bins, n_bands, spec_bins, hop), (
            name, list(info), spec_bins)

        # The normaliser coefficients are train.py's make_norm_alpha, as float32.
        alphas = np.empty(4, dtype=np.float32)
        lib.br_alphas(handle, P(alphas))
        assert alphas[0] == np.float32(make_norm_alpha(sr, hop, values['erb_norm_tau_sec'])), name
        assert alphas[1] == np.float32(make_norm_alpha(sr, hop, values['spec_norm_tau_sec'])), name
        assert alphas[0] == np.float32(feature_cfg['erb_alpha']), name
        assert alphas[1] == np.float32(feature_cfg['spec_alpha']), name
        scale = np.float32(math.sqrt(n_fft))
        np.testing.assert_allclose(alphas[2], 1.0 / math.sqrt(n_fft), rtol=1e-6)
        np.testing.assert_allclose(alphas[3], math.sqrt(n_fft), rtol=1e-6)

        # One center=False analysis frame: audiolib's root-Hann + the application's
        # 1/sqrt(n_fft) scale must match the tensor entering Python feature extraction.
        window = torch.sqrt(torch.hann_window(n_fft))
        n = np.arange(n_fft, dtype=np.float32)
        frame = (0.03 * np.sin(2 * np.pi * 17 * n / n_fft) +
                 0.01 * np.cos(2 * np.pi * 53 * n / n_fft)).astype(np.float32)
        c_re = np.empty(n_bins, dtype=np.float32)
        c_im = np.empty(n_bins, dtype=np.float32)
        lib.br_analysis(handle, P(frame), P(c_re), P(c_im))
        py_fft = torch.fft.rfft(torch.from_numpy(frame) * window, norm='ortho')
        np.testing.assert_allclose(c_re, py_fft.real.numpy(), rtol=3e-5, atol=3e-6, err_msg=name)
        np.testing.assert_allclose(c_im, py_fft.imag.numpy(), rtol=3e-5, atol=3e-6, err_msg=name)

        # Features and normaliser states: the C library frame by frame against train.py.
        spectra = deterministic_spectra(n_bins)
        py_erb, py_spec, py_state, _ = extract_model_features(
            torch.from_numpy(spectra.T).unsqueeze(0), torch.from_numpy(fwd), feature_cfg)
        c_erb = np.empty((3, n_bands), dtype=np.float32)
        c_spec = np.empty((3, 2, spec_bins), dtype=np.float32)
        for t, spectrum in enumerate(spectra):
            re = np.ascontiguousarray(spectrum.real, dtype=np.float32)
            im = np.ascontiguousarray(spectrum.imag, dtype=np.float32)
            ready = lib.br_step(handle, P(re), P(im), P(c_erb), P(c_spec))
            assert ready == int(t >= 2), (name, t, ready)
            if ready:
                np.testing.assert_allclose(
                    c_erb, py_erb[0, t - 2:t + 1].numpy(), rtol=3e-5, atol=3e-5, err_msg=name)
                np.testing.assert_allclose(
                    c_spec, py_spec[0, t - 2:t + 1].numpy(), rtol=3e-5, atol=3e-5, err_msg=name)
        c_erb_state = np.empty(n_bands, dtype=np.float32)
        c_spec_state = np.empty(spec_bins, dtype=np.float32)
        lib.br_states(handle, P(c_erb_state), P(c_spec_state))
        np.testing.assert_allclose(
            c_erb_state, py_state['erb'][0, 0].numpy(), rtol=3e-5, atol=3e-5, err_msg=name)
        np.testing.assert_allclose(
            c_spec_state, py_state['spec'][0, 0].numpy(), rtol=3e-5, atol=3e-5, err_msg=name)

        # rnnoise_apply_atten_lim vs inference.py apply_atten_lim, for the disabled (<=0)
        # no-op path and the active-mixing path.
        py_gains = torch.linspace(0.0, 1.0, n_bands, dtype=torch.float32)
        for atten_lim_db in (0.0, -3.0, 6.0, 12.0, 40.0):
            c_gains = py_gains.numpy().copy().astype(np.float32)
            assert lib.br_atten(handle, P(c_gains), atten_lim_db) == 0
            py_out = apply_atten_lim(py_gains.clone(), atten_lim_db).numpy()
            np.testing.assert_allclose(c_gains, py_out, rtol=1e-6, atol=1e-6, err_msg=name)

        # Band gains to bin gains through the loaded inverse, and its partition of unity.
        rng = np.random.default_rng(20261021 + sr + n_fft)
        c_bin = np.empty(n_bins, dtype=np.float32)
        for gains in rng.uniform(0.05, 1.0, (4, n_bands)).astype(np.float32):
            gains = np.ascontiguousarray(gains)
            assert lib.br_expand(handle, P(gains), P(c_bin)) == 0
            np.testing.assert_allclose(
                c_bin, (torch.from_numpy(gains) @ torch.from_numpy(inv).t()).numpy(),
                rtol=1e-5, atol=1e-6, err_msg=name)
        ones = np.ones(n_bands, dtype=np.float32)
        assert lib.br_expand(handle, P(ones), P(c_bin)) == 0
        np.testing.assert_allclose(c_bin, np.ones(n_bins), rtol=1e-6, atol=1e-6, err_msg=name)

        # One full caller frame per analysis call; synthesis is the gain product plus the
        # application's AcIstft, against an independent torch istft.
        waveform = rng.uniform(-0.2, 0.2, n_fft + 8 * hop).astype(np.float32)
        c_frames = []
        for offset in range(0, len(waveform) - n_fft + 1, hop):
            frame = np.ascontiguousarray(waveform[offset:offset + n_fft])
            lib.br_analysis(handle, P(frame), P(c_re), P(c_im))
            reference = torch.fft.rfft(torch.from_numpy(frame) * window, norm='ortho')
            np.testing.assert_allclose(c_re, reference.real.numpy(), rtol=3e-5, atol=3e-6, err_msg=name)
            np.testing.assert_allclose(c_im, reference.imag.numpy(), rtol=3e-5, atol=3e-6, err_msg=name)
            c_frames.append(c_re.copy() + 1j * c_im.copy())
        spectra = np.stack(c_frames)

        def synthesize(spectra, gains):
            lib.br_reset(handle)
            emitted = []
            for spectrum, gain in zip(spectra, gains):
                re = np.ascontiguousarray(spectrum.real)
                im = np.ascontiguousarray(spectrum.imag)
                gain = np.ascontiguousarray(gain)
                expected_re, expected_im = re * (gain * scale), im * (gain * scale)
                out = np.empty(hop, dtype=np.float32)
                assert lib.br_apply(handle, P(re), P(im), P(gain)) == 0
                # gain * sqrt(n_fft) in one product, bit for bit
                np.testing.assert_array_equal(re.view(np.uint32), expected_re.view(np.uint32))
                np.testing.assert_array_equal(im.view(np.uint32), expected_im.view(np.uint32))
                assert lib.br_synth(handle, P(re), P(im), P(out)) == hop
                emitted.append(out)
            return np.concatenate(emitted)

        gains = rng.uniform(0.05, 1.0, spectra.shape).astype(np.float32)
        c_wave = synthesize(spectra, gains)
        reference = istft(torch.from_numpy(spectra * gains).T, n_fft, hop, n_fft, window).numpy()
        assert len(c_wave) == len(spectra) * hop
        np.testing.assert_allclose(c_wave, reference[:len(c_wave)], rtol=3e-5, atol=3e-6, err_msg=name)
        np.testing.assert_array_equal(synthesize(spectra, gains), c_wave)   # reset is complete
        identity = synthesize(spectra, np.ones_like(gains))
        np.testing.assert_allclose(identity[hop:], waveform[hop:len(identity)],
                                   rtol=3e-5, atol=3e-6, err_msg=name)
        # A wrong whole-frame shift or extra analysis delay cannot hide behind a periodic test
        # tone: compare the untrimmed first hop separately.
        first_hop = waveform[:hop] * window[:hop].numpy() ** 2
        np.testing.assert_allclose(identity[:hop], first_hop, rtol=3e-5, atol=3e-6, err_msg=name)
    finally:
        lib.br_destroy(handle)
    return f'{name}: sr={sr} n_fft={n_fft} bands={n_bands} spec_bins={spec_bins}'


def pcm_chain(lib, row, workdir):
    """The framing scales end to end, from PCM: AcStft at spec_scale = stft_scale into the C
    features, and the gain product into AcIstft at ispec_scale 1.0, against train.py's stft,
    extract_model_features and istft of the same PCM."""
    case = open_case(lib, row, workdir)
    name, n_fft, hop, handle = case.name, case.n_fft, case.hop, case.handle
    try:
        window = torch.sqrt(torch.hann_window(n_fft))
        rng = np.random.default_rng(20261106 + case.sr + n_fft)
        pcm = rng.uniform(-0.2, 0.2, n_fft + 11 * hop).astype(np.float32)
        frames = [np.ascontiguousarray(pcm[start:start + n_fft])
                  for start in range(0, len(pcm) - n_fft + 1, hop)]
        py_spec = stft(torch.from_numpy(pcm), n_fft, hop, n_fft, window)          # (n_bins, T)
        assert py_spec.shape[-1] == len(frames), (name, py_spec.shape, len(frames))
        py_erb, py_feat, _, _ = extract_model_features(
            py_spec.unsqueeze(0), torch.from_numpy(case.fwd), case.feature_cfg)

        c_re = np.empty(case.n_bins, dtype=np.float32)
        c_im = np.empty(case.n_bins, dtype=np.float32)
        c_erb = np.empty((3, case.n_bands), dtype=np.float32)
        c_spec = np.empty((3, 2, case.spec_bins), dtype=np.float32)
        for t, frame in enumerate(frames):
            lib.br_analysis(handle, P(frame), P(c_re), P(c_im))
            ready = lib.br_step(handle, P(c_re), P(c_im), P(c_erb), P(c_spec))
            assert ready == int(t >= 2), (name, t, ready)
            if ready:
                np.testing.assert_allclose(c_erb, py_erb[0, t - 2:t + 1].numpy(), rtol=3e-5,
                                           atol=3e-5, err_msg=f'{name} PCM erb features, frame {t}')
                np.testing.assert_allclose(c_spec, py_feat[0, t - 2:t + 1].numpy(), rtol=3e-5,
                                           atol=3e-5, err_msg=f'{name} PCM complex features, frame {t}')

        lib.br_reset(handle)
        gains = rng.uniform(0.05, 1.0, (len(frames), case.n_bins)).astype(np.float32)
        emitted = []
        for frame, gain in zip(frames, gains):
            out = np.empty(hop, dtype=np.float32)
            lib.br_analysis(handle, P(frame), P(c_re), P(c_im))
            assert lib.br_apply(handle, P(c_re), P(c_im), P(np.ascontiguousarray(gain))) == 0
            assert lib.br_synth(handle, P(c_re), P(c_im), P(out)) == hop
            emitted.append(out)
        c_wave = np.concatenate(emitted)
        reference = istft(py_spec * torch.from_numpy(gains).T, n_fft, hop, n_fft, window).numpy()
        np.testing.assert_allclose(c_wave, reference[:len(c_wave)], rtol=3e-5, atol=3e-6,
                                   err_msg=f'{name} PCM synthesis: gain product and ispec_scale')
    finally:
        lib.br_destroy(handle)
    return name


def test_pcm_chain():
    lib = build_bridge()
    with tempfile.TemporaryDirectory(prefix='rnnoise-pcm-chain-') as tmp:
        for row in GRIDS:
            name = pcm_chain(lib, row, pathlib.Path(tmp))
            print('PASS: PCM features and synthesis agree with train.py on', name)


def main():
    lib = build_bridge()
    with tempfile.TemporaryDirectory(prefix='rnnoise-feature-erb-') as tmp:
        for row in GRIDS:
            print('PASS: train.py and process.c agree on', run_grid(lib, row, pathlib.Path(tmp)))
    print('PASS: normaliser alphas, STFT/features/states, attenuation limit, expansion, '
          'gain product, synthesis hop, reset and latency')


def test_main():
    """Expose main() to pytest.

    This file is script-shaped (the Makefile runs it directly); without a ``test_``-prefixed
    entry point pytest would collect zero items from it and report the run green.
    """
    main()


if __name__ == '__main__':
    main()
    test_pcm_chain()
