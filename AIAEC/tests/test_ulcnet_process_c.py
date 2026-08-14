"""Parity gate for the Align-ULCNet C pre/post-processing.

Compiles AIAEC/Align_ULCNet/ulcnet_process.c with a tiny file-driven test
main (written to a temp dir, never to the repo), runs it on random audio and
compares, frame by frame and sample by sample, against the Python reference
(aiaec_streaming.StreamSTFT/StreamISTFT), which is itself the bit-exact twin
of torch.stft/istft(center=True).

The C side now transforms through audio_common's fft_wrapper (caller-owned
FftHandle, real RFFT/IRFFT, backend-switched by BACKEND=kiss/ne10), so the
fixture first builds audio_common's KISS static lib the way the AEC C tests
document it (make -s -C ../audio_common BACKEND=kiss lib + print-lib-path)
and links it into the driver. The driver deliberately shares ONE FftHandle
across the analysis and the synthesis (strictly sequential use), exercising
the same sharing contract the pipeline variants rely on.

Expected agreement is float-ULP class, not bit-exact: the C side uses the
f32 KISS FFT while torch computes with double twiddles. Measured ~2e-5
absolute on unit-scale audio; tolerances are pinned at ~10x that. A
one-frame misalignment shows up as O(1) error, so the comparison has teeth
by construction (asserted below).
"""

import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT
from AIAEC.dataset_gen.aec_features import sqrt_hann_window

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ULCNET_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'Align_ULCNet')
_AIAEC_DIR = os.path.dirname(_THIS_DIR)
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audio_common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

N_FFT, HOP, BINS = 512, 256, 257

_DRIVER = r'''
#include <stdio.h>
#include <stdlib.h>
#include "ulcnet_process.h"

/* argv: mode in.f32 out.f32
 * mode "analysis":  in = raw samples -> out = frames (re[257] im[257] each)
 * mode "roundtrip": in = raw samples -> out = analysis->synthesis samples
 * One shared FftHandle drives BOTH the analysis and the synthesis (their
 * use is strictly sequential), matching the pipelines' sharing contract. */
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *fi = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb");
    if (!fi || !fo) return 3;
    static UlcnetAnalysis an;
    static UlcnetSynthesis sy;
    static float win[ULCNET_N_FFT];
    FftHandle *fft = fft_create(ULCNET_N_FFT);  /* heap OK in a test driver */
    if (!fft) return 4;
    ulcnet_make_window(win);
    if (ulcnet_analysis_init(&an, fft, win) != 0) return 5;
    if (ulcnet_synthesis_init(&sy, fft, win) != 0) return 5;
    int roundtrip = argv[1][0] == 'r';
    float hop[ULCNET_HOP];
    float fre[2][ULCNET_BINS], fim[2][ULCNET_BINS];
    float out[ULCNET_N_FFT];
    while (fread(hop, sizeof(float), ULCNET_HOP, fi) == ULCNET_HOP) {
        int n = ulcnet_analysis_push(&an, hop, fre, fim);
        for (int f = 0; f < n; ++f) {
            if (roundtrip) {
                int m = ulcnet_synthesis_push(&sy, fre[f], fim[f], out);
                fwrite(out, sizeof(float), (size_t)m, fo);
            } else {
                fwrite(fre[f], sizeof(float), ULCNET_BINS, fo);
                fwrite(fim[f], sizeof(float), ULCNET_BINS, fo);
            }
        }
    }
    int n = ulcnet_analysis_flush(&an, fre, fim);
    for (int f = 0; f < n; ++f) {
        if (roundtrip) {
            int m = ulcnet_synthesis_push(&sy, fre[f], fim[f], out);
            fwrite(out, sizeof(float), (size_t)m, fo);
        } else {
            fwrite(fre[f], sizeof(float), ULCNET_BINS, fo);
            fwrite(fim[f], sizeof(float), ULCNET_BINS, fo);
        }
    }
    if (roundtrip) {
        int m = ulcnet_synthesis_flush(&sy, out);
        fwrite(out, sizeof(float), (size_t)m, fo);
    }
    fclose(fi); fclose(fo);
    fft_destroy(fft);
    return 0;
}
'''


@pytest.fixture(scope='module')
def audio_common_lib():
    """audio_common's KISS static lib, built + located the way the AEC C
    tests document it (make -s -C ../audio_common BACKEND=kiss lib, then
    print-lib-path for this invocation's exact archive path)."""
    if shutil.which('make') is None:
        pytest.skip('no make available')
    subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=kiss', 'lib'],
        check=True, capture_output=True,
    )
    out = subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=kiss', 'print-lib-path'],
        check=True, capture_output=True, text=True,
    )
    lib = out.stdout.strip().splitlines()[-1]
    assert os.path.isfile(lib), lib
    return lib


@pytest.fixture(scope='module')
def driver(tmp_path_factory, audio_common_lib):
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp('ulcnet_c')
    main_c = work / 'driver.c'
    main_c.write_text(_DRIVER)
    exe = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _ULCNET_DIR, '-I', _AC_INCLUDE,
         str(main_c), os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
         audio_common_lib, '-lm', '-o', str(exe)],
        check=True, capture_output=True,
    )
    return work, exe


def _run(driver_fx, mode, samples):
    work, exe = driver_fx
    fin = work / 'in.f32'
    fout = work / 'out.f32'
    samples.astype(np.float32).tofile(fin)
    subprocess.run([str(exe), mode, str(fin), str(fout)], check=True)
    return np.fromfile(fout, dtype=np.float32)


def _python_frames(x):
    st = StreamSTFT(N_FFT, HOP, sqrt_hann_window(N_FFT))
    frames = []
    for s in range(0, x.numel(), HOP):
        frames += st.push(x[None, s:s + HOP])
    frames += st.flush()
    return torch.stack(frames, dim=1)[0]          # [T, 257] complex


def test_analysis_matches_python_reference(driver):
    torch.manual_seed(3)
    x = torch.randn(HOP * 40)
    ref = _python_frames(x)
    raw = _run(driver, 'analysis', x.numpy())
    got = raw.reshape(-1, 2, BINS)
    assert got.shape[0] == ref.shape[0]           # L/hop + 1 frames
    c = torch.complex(torch.from_numpy(got[:, 0]), torch.from_numpy(got[:, 1]))
    diff = (c - ref).abs().max().item()
    assert diff < 2e-4, diff                      # f32 FFT vs torch, ~1e-5 class


def test_analysis_would_catch_a_frame_slip(driver):
    """The gate has teeth: comparing against a one-frame-shifted reference
    must blow far past the tolerance."""
    torch.manual_seed(4)
    x = torch.randn(HOP * 40)
    ref = _python_frames(x)
    raw = _run(driver, 'analysis', x.numpy())
    got = raw.reshape(-1, 2, BINS)
    c = torch.complex(torch.from_numpy(got[:, 0]), torch.from_numpy(got[:, 1]))
    slipped = (c[1:] - ref[:-1]).abs().max().item()
    assert slipped > 1.0, slipped


def test_roundtrip_matches_python_and_input(driver):
    torch.manual_seed(5)
    length = HOP * 40
    x = torch.randn(length)

    raw = _run(driver, 'roundtrip', x.numpy())
    got = torch.from_numpy(raw[:length])

    # Python reference roundtrip (bit-exact vs torch stft->istft).
    stft = StreamSTFT(N_FFT, HOP, sqrt_hann_window(N_FFT))
    istft = StreamISTFT(N_FFT, HOP, sqrt_hann_window(N_FFT))
    frames = []
    for s in range(0, length, HOP):
        frames += stft.push(x[None, s:s + HOP])
    frames += stft.flush()
    pieces = [istft.push(f) for f in frames]
    emitted = sum(p.shape[-1] for p in pieces)
    pieces.append(istft.flush(length=length, already_emitted=emitted))
    ref = torch.cat(pieces, dim=-1)[0, :length]

    assert got.shape[-1] == length
    assert (got - ref).abs().max().item() < 2e-4
    # COLA identity: the whole chain reconstructs the input.
    assert (got - x).abs().max().item() < 2e-4


def test_emission_timing_contract(driver):
    """0 frames on the first hop, 2 on the second, 1 per hop after; flush
    adds exactly one -- total L/hop + 1."""
    x = torch.randn(HOP * 7)
    raw = _run(driver, 'analysis', x.numpy())
    assert raw.size == (7 + 1) * 2 * BINS


def test_compression_helpers_roundtrip(driver, audio_common_lib):
    """signed |x|^0.3 then |x|^(1/0.3) is identity to f32 accuracy; checked
    through the Python-side formula to pin the exponent convention."""
    work, exe = driver
    # helper functions are exercised via a tiny dedicated main
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    src = work / 'comp.c'
    src.write_text(r'''
#include <math.h>
#include <stdio.h>
#include "ulcnet_process.h"
int main(void) {
    float re[ULCNET_BINS], im[ULCNET_BINS], zr[ULCNET_BINS], zi[ULCNET_BINS];
    float rr[ULCNET_BINS], ri[ULCNET_BINS];
    for (int k = 0; k < ULCNET_BINS; ++k) {
        re[k] = (k % 3 == 0 ? -1.0f : 1.0f) * (0.001f + 0.01f * (float)k);
        im[k] = (k % 2 == 0 ? 1.0f : -1.0f) * (0.002f + 0.005f * (float)k);
    }
    ulcnet_compress_frame(re, im, zr, zi);
    ulcnet_expand_frame(zr, zi, rr, ri);
    float worst = 0.0f;
    for (int k = 0; k < ULCNET_BINS; ++k) {
        float dr = fabsf(rr[k] - re[k]) / fabsf(re[k]);
        float di = fabsf(ri[k] - im[k]) / fabsf(im[k]);
        if (dr > worst) worst = dr;
        if (di > worst) worst = di;
    }
    printf("%g\n", worst);
    return worst < 5e-5f ? 0 : 1;
}
''')
    exe2 = work / 'comp'
    subprocess.run([cc, '-O2', '-std=c99', '-ffp-contract=off',
                    '-I', _ULCNET_DIR, '-I', _AC_INCLUDE, str(src),
                    os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
                    audio_common_lib, '-lm', '-o', str(exe2)],
                   check=True, capture_output=True)
    subprocess.run([str(exe2)], check=True)


def test_ulcnet_compressed_mask_helper_matches_python(driver, audio_common_lib):
    work, _ = driver
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    source = work / 'ulc_mask.c'
    source.write_text(r'''
#include <stdio.h>
#include "aiaec_process.h"
int main(int argc, char **argv) {
    float in[4][AIAEC_N_BINS], out[2][AIAEC_N_BINS];
    FILE *fi, *fo;
    if (argc != 3) return 2;
    fi = fopen(argv[1], "rb"); fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 3;
    if (fread(in, sizeof(float), 4 * AIAEC_N_BINS, fi) !=
        4 * AIAEC_N_BINS) return 4;
    aiaec_apply_ulcnet_compressed_mask(
        in[0], in[1], in[2], in[3], out[0], out[1]);
    fwrite(out, sizeof(float), 2 * AIAEC_N_BINS, fo);
    fclose(fi); fclose(fo); return 0;
}
''')
    executable = work / 'ulc_mask'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR, '-I', _AC_INCLUDE,
         str(source), os.path.join(_AIAEC_DIR, 'aiaec_process.c'),
         os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True)
    torch.manual_seed(21)
    values = torch.randn(4, BINS).clamp(-2.0, 2.0)
    input_path = work / 'ulc_mask_in.f32'
    output_path = work / 'ulc_mask_out.f32'
    values.numpy().astype(np.float32).tofile(input_path)
    subprocess.run([str(executable), str(input_path), str(output_path)],
                   check=True)
    got = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(2, BINS)
    exponent = 0.3
    compressed = values[:2].sign() * values[:2].abs().pow(exponent)
    masked = torch.stack((
        compressed[0] * values[2] - compressed[1] * values[3],
        compressed[1] * values[2] + compressed[0] * values[3],
    ))
    expected = masked.sign() * masked.abs().pow(1.0 / exponent)
    assert (got - expected).abs().max().item() < 2e-5


def test_deepvqe_ccm_helper_matches_python(tmp_path):
    from AIAEC.aiaec_common import apply_causal_tf_filter

    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    source = tmp_path / 'ccm.c'
    source.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include "deepvqe_process.h"
int main(int argc, char **argv) {
    DeepVqeCcmState state;
    float re[AIAEC_N_BINS], im[AIAEC_N_BINS];
    float taps[AIAEC_N_BINS][DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2];
    float out_re[AIAEC_N_BINS], out_im[AIAEC_N_BINS];
    FILE *fi, *fo;
    if (argc != 4) return 2;
    fi = fopen(argv[1], "rb"); fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 3;
    deepvqe_ccm_init(&state);
    for (int frame = 0; frame < atoi(argv[3]); ++frame) {
        if (fread(re, sizeof(float), AIAEC_N_BINS, fi) != AIAEC_N_BINS) return 4;
        if (fread(im, sizeof(float), AIAEC_N_BINS, fi) != AIAEC_N_BINS) return 4;
        if (fread(taps, sizeof(float), AIAEC_N_BINS * 3 * 3 * 2, fi) !=
            AIAEC_N_BINS * 3 * 3 * 2) return 4;
        deepvqe_ccm_process(&state, re, im, taps, out_re, out_im);
        fwrite(out_re, sizeof(float), AIAEC_N_BINS, fo);
        fwrite(out_im, sizeof(float), AIAEC_N_BINS, fo);
    }
    fclose(fi); fclose(fo); return 0;
}
''')
    executable = tmp_path / 'ccm'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR,
         '-I', os.path.join(_AIAEC_DIR, 'DeepVQE_S'), '-I', _AC_INCLUDE,
         str(source), os.path.join(_AIAEC_DIR, 'DeepVQE_S', 'deepvqe_process.c'),
         '-o', str(executable)], check=True, capture_output=True)
    torch.manual_seed(22)
    frames = 6
    spectrum = torch.complex(torch.randn(1, frames, BINS),
                             torch.randn(1, frames, BINS))
    taps = torch.complex(torch.randn(1, frames, BINS, 3, 3),
                         torch.randn(1, frames, BINS, 3, 3))
    raw = []
    for frame in range(frames):
        raw.extend((spectrum.real[0, frame].numpy(),
                    spectrum.imag[0, frame].numpy(),
                    torch.view_as_real(taps[0, frame]).numpy().reshape(-1)))
    input_path = tmp_path / 'ccm_in.f32'
    output_path = tmp_path / 'ccm_out.f32'
    np.concatenate(raw).astype(np.float32).tofile(input_path)
    subprocess.run([str(executable), str(input_path), str(output_path), str(frames)],
                   check=True)
    got = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(
        frames, 2, BINS)
    got = torch.complex(got[:, 0], got[:, 1])
    expected = apply_causal_tf_filter(spectrum, taps, 3, 1)[0]
    assert (got - expected).abs().max().item() < 2e-5
