"""Golden-vector parity test between train.py and process.c feature extraction."""

import atexit
import configparser
import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Each of the three model projects has its own top-level ``train.py`` (and
# ``inference.py``/``model.py``). Under a single pytest session the first one
# imported wins ``sys.modules``, so a sibling project's tests would silently
# exercise the wrong code.  Dropping the cached entries forces the re-import
# to resolve against the ROOT just inserted above.
for _stale in ('train', 'inference', 'model', 'checkpoint_utils'):
    sys.modules.pop(_stale, None)


from train import (  # noqa: E402
    compute_erb_matrix,
    erb_bandborder,
    extract_model_features,
    read_feature_config,
)
from inference import apply_atten_lim  # noqa: E402


N_BINS = 257
N_BANDS = 22
SPEC_BINS = 129


WRAPPER = r"""
#include "process.h"
#include <stddef.h>
#include <string.h>

size_t test_state_size(void) { return sizeof(RNNoiseState); }

void test_state_init(void *memory) {
    rnnoise_state_init((RNNoiseState *)memory);
}

void test_analysis(void *memory, const float *frame, float *re, float *im) {
    rnnoise_analysis((RNNoiseState *)memory, frame, re, im);
}

int test_step(void *memory, const float *re, const float *im,
              float *out_erb, float *out_spec) {
    return rnnoise_compute_features(
        (RNNoiseState *)memory, re, im,
        (float (*)[RNNOISE_N_BANDS])out_erb,
        (float (*)[2][RNNOISE_SPEC_BINS])out_spec);
}

void test_copy_erb_state(void *memory, float *out) {
    RNNoiseState *state = (RNNoiseState *)memory;
    memcpy(out, state->erb_norm_state, sizeof(state->erb_norm_state));
}

void test_copy_spec_state(void *memory, float *out) {
    RNNoiseState *state = (RNNoiseState *)memory;
    memcpy(out, state->spec_norm_state, sizeof(state->spec_norm_state));
}

void test_apply_atten_lim(float *band_gains, float atten_lim_db) {
    rnnoise_apply_atten_lim(band_gains, atten_lim_db);
}
"""


_BRIDGE = {}


def build_bridge():
    """Compile the C bridge once per process and keep it mapped.

    It used to be built inside a ``TemporaryDirectory`` that was torn down
    while the .dylib was still loaded through ctypes.  Running this file
    directly hid the problem -- the process exited immediately afterwards --
    but under pytest the process lives on and the next module import crashed
    with SIGTRAP.  The build directory now survives until interpreter exit.
    """
    if 'lib' not in _BRIDGE:
        tmpdir = tempfile.mkdtemp(prefix='rnnoise-feature-parity-')
        atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
        _BRIDGE['lib'] = _compile_bridge(tmpdir)
    return _BRIDGE['lib']


def _compile_bridge(tmpdir):
    cc = os.environ.get('CC') or shutil.which('cc')
    if not cc:
        raise RuntimeError('C compiler not found (set CC or install cc)')
    wrapper = pathlib.Path(tmpdir) / 'feature_bridge.c'
    wrapper.write_text(WRAPPER)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    library = pathlib.Path(tmpdir) / f'feature_bridge{suffix}'
    shared_flag = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
    subprocess.run([
        cc, shared_flag, '-fPIC', '-O2', '-ffp-contract=off',
        '-I', str(ROOT), str(wrapper), str(ROOT / 'process.c'),
        '-o', str(library), '-lm',
    ], check=True)
    return ctypes.CDLL(str(library))


def deterministic_spectra(n_frames=12):
    k = np.arange(N_BINS, dtype=np.float32)
    frames = []
    for t in range(n_frames):
        scale = np.float32(0.65 + 0.07 * t)
        re = scale * (0.0008 + 0.00005 * ((k * 17 + t * 3) % 31))
        im = scale * (-0.0004 + 0.00004 * ((k * 11 + t * 5) % 23))
        im[0] = 0.0
        im[-1] = 0.0
        frames.append(re.astype(np.float32) + 1j * im.astype(np.float32))
    return np.stack(frames, axis=0).astype(np.complex64)


def main():
    # Read the feature constants rather than restating them.  This file used to
    # hardcode alpha and the four init values, so pinning erb_norm_alpha in
    # config.ini + process.h left the test comparing C-at-0.99 against
    # Python-at-0.984 and reporting a parity failure that was purely its own.
    # C-vs-Python parity is what this test exists to check; the constants both
    # sides use are test_feature_contract.py's job.
    cfg = configparser.ConfigParser()
    if not cfg.read(ROOT / 'config.ini'):
        raise FileNotFoundError(ROOT / 'config.ini')
    feature_cfg = read_feature_config(cfg, 16000, 256, 512, 512)
    assert feature_cfg['spec_bins'] == SPEC_BINS, feature_cfg['spec_bins']
    assert feature_cfg['n_bands'] == N_BANDS, feature_cfg['n_bands']
    spectra = deterministic_spectra()
    borders = erb_bandborder(
        N_BANDS, 16000, 512,
        cfg.getint('signal', 'min_bins_per_band', fallback=2))
    erb_matrix = torch.from_numpy(compute_erb_matrix(borders, 512, mode=0))
    spec_torch = torch.from_numpy(spectra.T).unsqueeze(0)
    py_erb, py_spec, py_state, _ = extract_model_features(
        spec_torch, erb_matrix, feature_cfg)

    lib = build_bridge()
    float_p = ctypes.POINTER(ctypes.c_float)
    lib.test_state_size.restype = ctypes.c_size_t
    lib.test_state_init.argtypes = [ctypes.c_void_p]
    lib.test_analysis.argtypes = [ctypes.c_void_p, float_p, float_p, float_p]
    lib.test_step.argtypes = [ctypes.c_void_p, float_p, float_p, float_p, float_p]
    lib.test_step.restype = ctypes.c_int
    lib.test_copy_erb_state.argtypes = [ctypes.c_void_p, float_p]
    lib.test_copy_spec_state.argtypes = [ctypes.c_void_p, float_p]
    lib.test_apply_atten_lim.argtypes = [float_p, ctypes.c_float]

    memory = ctypes.create_string_buffer(lib.test_state_size())
    lib.test_state_init(memory)

    # One center=False analysis frame: C root-Hann + normalized FFT must
    # match the exact tensor entering Python feature extraction.
    n = np.arange(512, dtype=np.float32)
    frame = (0.03 * np.sin(2 * np.pi * 17 * n / 512) +
             0.01 * np.cos(2 * np.pi * 53 * n / 512)).astype(np.float32)
    c_re = np.empty(N_BINS, dtype=np.float32)
    c_im = np.empty(N_BINS, dtype=np.float32)
    lib.test_analysis(
        memory, frame.ctypes.data_as(float_p),
        c_re.ctypes.data_as(float_p), c_im.ctypes.data_as(float_p))
    window = torch.sqrt(torch.hann_window(512))
    py_fft = torch.fft.rfft(torch.from_numpy(frame) * window, norm='ortho')
    np.testing.assert_allclose(c_re, py_fft.real.numpy(), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(c_im, py_fft.imag.numpy(), rtol=3e-5, atol=3e-6)

    c_erb = np.empty((3, N_BANDS), dtype=np.float32)
    c_spec = np.empty((3, 2, SPEC_BINS), dtype=np.float32)
    for t, frame in enumerate(spectra):
        re = np.ascontiguousarray(frame.real, dtype=np.float32)
        im = np.ascontiguousarray(frame.imag, dtype=np.float32)
        ready = lib.test_step(
            memory,
            re.ctypes.data_as(float_p),
            im.ctypes.data_as(float_p),
            c_erb.ctypes.data_as(float_p),
            c_spec.ctypes.data_as(float_p),
        )
        assert ready == int(t >= 2), (t, ready)
        if ready:
            np.testing.assert_allclose(
                c_erb, py_erb[0, t - 2:t + 1].numpy(), rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(
                c_spec, py_spec[0, t - 2:t + 1].numpy(), rtol=3e-5, atol=3e-5)

    c_erb_state = np.empty(N_BANDS, dtype=np.float32)
    c_spec_state = np.empty(SPEC_BINS, dtype=np.float32)
    lib.test_copy_erb_state(memory, c_erb_state.ctypes.data_as(float_p))
    lib.test_copy_spec_state(memory, c_spec_state.ctypes.data_as(float_p))
    np.testing.assert_allclose(
        c_erb_state, py_state['erb'][0, 0].numpy(), rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        c_spec_state, py_state['spec'][0, 0].numpy(), rtol=3e-5, atol=3e-5)

    # rnnoise_apply_atten_lim must agree with inference.py's apply_atten_lim
    # (attenuation-limit gain mix, ported from Rikorose/DeepFilterNet
    # enhance.py) bit-for-bit-equivalent within float32 tolerance, both
    # for the disabled (<=0) no-op path and the active-mixing path.
    # MUST be N_BANDS long: rnnoise_apply_atten_lim writes
    # RNNOISE_N_BANDS entries unconditionally.  This used to pass a
    # 5-element array, overflowing the heap by 17 floats on every call
    # with atten_lim_db > 0.  The process exited before anything noticed;
    # under pytest the corruption surfaced later as a malloc abort inside
    # an unrelated torch op.
    py_gains = torch.linspace(0.0, 1.0, N_BANDS, dtype=torch.float32)
    for atten_lim_db in (0.0, -3.0, 6.0, 12.0, 40.0):
        c_gains = py_gains.numpy().copy().astype(np.float32)
        lib.test_apply_atten_lim(c_gains.ctypes.data_as(float_p),
                                 ctypes.c_float(atten_lim_db))
        py_out = apply_atten_lim(py_gains.clone(), atten_lim_db).numpy()
        np.testing.assert_allclose(c_gains, py_out, rtol=1e-6, atol=1e-6)

    print('PASS: train.py and process.c STFT/features/states agree')
    print('PASS: rnnoise_apply_atten_lim matches inference.py apply_atten_lim')


def test_main():
    """Expose main() to pytest.

    This file is script-shaped (Makefile runs it directly), so without a
    ``test_``-prefixed entry point pytest collected ZERO items from it and
    still reported the run green -- the C/Python parity and feature-contract
    guards were simply absent from any `pytest` invocation.
    """
    main()


if __name__ == '__main__':
    main()
