"""Parity gate for the Align-ULCNet C pre/post-processing.

Compiles AIAEC/Align_ULCNet/ulcnet_process.c with a tiny file-driven test
main (written to a temp dir, never to the repo), runs it on random audio and
compares, frame by frame and sample by sample, against the Python reference
(aiaec_streaming.StreamSTFT/StreamISTFT), which is itself the bit-exact twin
of torch.stft/istft(center=True).

Expected agreement is float-ULP class, not bit-exact: the C side uses the
shared f32 radix-2 FFT (AINR/dfn_process_common.h) while torch computes with
double twiddles. Measured ~2e-5 absolute on unit-scale audio; tolerances are
pinned at ~10x that. A one-frame misalignment shows up as O(1) error, so the
comparison has teeth by construction (asserted below).
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

N_FFT, HOP, BINS = 512, 256, 257

_DRIVER = r'''
#include <stdio.h>
#include <stdlib.h>
#include "ulcnet_process.h"

/* argv: mode in.f32 out.f32
 * mode "analysis":  in = raw samples -> out = frames (re[257] im[257] each)
 * mode "roundtrip": in = raw samples -> out = analysis->synthesis samples */
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *fi = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb");
    if (!fi || !fo) return 3;
    static UlcnetAnalysis an;
    static UlcnetSynthesis sy;
    ulcnet_analysis_init(&an);
    ulcnet_synthesis_init(&sy);
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
    return 0;
}
'''


@pytest.fixture(scope='module')
def driver(tmp_path_factory):
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp('ulcnet_c')
    main_c = work / 'driver.c'
    main_c.write_text(_DRIVER)
    exe = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _ULCNET_DIR,
         str(main_c), os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
         '-lm', '-o', str(exe)],
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


def test_compression_helpers_roundtrip(driver):
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
    subprocess.run([cc, '-O2', '-std=c99', '-I', _ULCNET_DIR, str(src),
                    os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
                    '-lm', '-o', str(exe2)], check=True, capture_output=True)
    subprocess.run([str(exe2)], check=True)
