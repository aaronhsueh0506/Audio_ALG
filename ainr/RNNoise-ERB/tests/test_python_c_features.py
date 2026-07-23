"""Golden-vector parity test between train.py and process.c feature extraction."""

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

from train import (  # noqa: E402
    compute_erb_matrix,
    erb_bandborder,
    extract_model_features,
    make_norm_alpha,
)


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
"""


def build_bridge(tmpdir):
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
    alpha = make_norm_alpha(16000, 256, 1.0)
    feature_cfg = dict(
        erb_alpha=alpha,
        erb_norm_init_lo_db=-60.0,
        erb_norm_init_hi_db=-90.0,
        erb_norm_scale_db=40.0,
        erb_norm_clip=5.0,
        spec_bins=SPEC_BINS,
        spec_alpha=alpha,
        spec_norm_init_lo=0.001,
        spec_norm_init_hi=0.0001,
        spec_norm_eps=1e-12,
        spec_clip=10.0,
    )
    spectra = deterministic_spectra()
    borders = erb_bandborder(N_BANDS, 16000, 512)
    erb_matrix = torch.from_numpy(compute_erb_matrix(borders, 512, mode=0))
    spec_torch = torch.from_numpy(spectra.T).unsqueeze(0)
    py_erb, py_spec, py_state, _ = extract_model_features(
        spec_torch, erb_matrix, feature_cfg)

    with tempfile.TemporaryDirectory(prefix='rnnoise-feature-parity-') as tmpdir:
        lib = build_bridge(tmpdir)
        float_p = ctypes.POINTER(ctypes.c_float)
        lib.test_state_size.restype = ctypes.c_size_t
        lib.test_state_init.argtypes = [ctypes.c_void_p]
        lib.test_analysis.argtypes = [ctypes.c_void_p, float_p, float_p, float_p]
        lib.test_step.argtypes = [ctypes.c_void_p, float_p, float_p, float_p, float_p]
        lib.test_step.restype = ctypes.c_int
        lib.test_copy_erb_state.argtypes = [ctypes.c_void_p, float_p]
        lib.test_copy_spec_state.argtypes = [ctypes.c_void_p, float_p]

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

    print('PASS: train.py and process.c STFT/features/states agree')


if __name__ == '__main__':
    main()
