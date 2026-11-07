"""Parity gate for the Align-ULCNet framing contract and the mask helpers.

The framing is the application's, built from audiolib directly (the contract
the weights are trained against is stated in ulcnet_process.h). A tiny
file-driven test main (written to a temp dir, never to the repo) drives
audiolib's AcStft / AcIstft in that configuration (sqrt-Hann, centered
analysis, envelope-normalised synthesis) on random audio and compares, frame by
frame, against the Python reference (aiaec_streaming.StreamSTFT/StreamISTFT,
the bit-exact twin of torch.stft/istft(center=True)). One program serves both
grids: n_fft is an argument. The fixture first builds audiolib's NE10 static lib
the way the AEC C tests document it. The driver shares ONE FftHandle across
analysis and synthesis (strictly sequential), the sharing contract the pipeline
variants rely on. The mask helpers (aiaec_process.c) and the DeepVQE-S CCM
kernel are checked against their Python statements at the same widths.

Agreement is float-ULP class, not bit-exact (the C side uses the f32 NE10 FFT,
torch double twiddles): tolerances sit at ~10x the ~2e-5 absolute error on
unit-scale audio, and a one-frame misalignment shows up as O(1) error, so the
comparison has teeth by construction.
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
_AIAEC_DIR = os.path.dirname(_THIS_DIR)
_ULCNET_DIR = os.path.join(_AIAEC_DIR, '..', 'lib', 'aiaec', 'ulcnet')
_CAGCRN_DIR = os.path.join(_AIAEC_DIR, '..', 'lib', 'aiaec', 'cagcrn')
_ALIGN_CRUSE_DIR = os.path.join(_AIAEC_DIR, '..', 'lib', 'aiaec', 'align_cruse')
_DEEPVQE_S_DIR = os.path.join(_AIAEC_DIR, '..', 'lib', 'aiaec', 'deepvqe_s')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audiolib', 'common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

N_FFT, HOP, BINS = 512, 256, 257

_DRIVER = r'''
#include <stdio.h>
#include <stdlib.h>
#include "audio_stft.h"

#define N_FFT_MAX 1024
#define BINS_MAX 513

/* argv: mode in.f32 out.f32 n_fft
 * mode "analysis":  in = raw samples -> out = frames (re[bins] im[bins] each)
 * mode "roundtrip": in = raw samples -> out = analysis->synthesis samples
 * One shared FftHandle and window table drive BOTH the centered analysis
 * and the synthesis (their use is strictly sequential), matching the
 * pipelines' sharing contract. */
static void *block(size_t bytes) {
    void *mem = NULL;
    if (posix_memalign(&mem, 16, bytes) != 0) exit(6);
    return mem;
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    FILE *fi = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb");
    if (!fi || !fo) return 3;
    const int n_fft = atoi(argv[4]);
    const int n_hop = n_fft / 2, n_bins = n_fft / 2 + 1;
    if (n_fft < 16 || n_fft > N_FFT_MAX) return 2;
    FftHandle *fft = fft_create(n_fft);  /* heap OK in a test driver */
    static float win[N_FFT_MAX];
    AcStftConfig cfg;
    if (!fft) return 4;
    ac_window_build(win, n_fft);
    ac_stft_config_default(&cfg, n_fft);
    cfg.center = 1;
    cfg.ola = AC_OLA_ENV_NORM;
    cfg.fft = fft;
    cfg.window = win;
    size_t an_bytes = ac_stft_get_mem_size(&cfg);
    size_t sy_bytes = ac_istft_get_mem_size(&cfg);
    if (an_bytes == 0 || sy_bytes == 0) return 5;
    AcStft *an = ac_stft_init(block(an_bytes), an_bytes, &cfg);
    AcIstft *sy = ac_istft_init(block(sy_bytes), sy_bytes, &cfg);
    if (!an || !sy) return 5;
    int roundtrip = argv[1][0] == 'r';
    float hop[N_FFT_MAX / 2];
    float fre[2 * BINS_MAX], fim[2 * BINS_MAX];
    float out[N_FFT_MAX];
    while (fread(hop, sizeof(float), (size_t)n_hop, fi) == (size_t)n_hop) {
        int n = ac_stft_push_planar(an, hop, fre, fim);
        for (int f = 0; f < n; ++f) {
            const float *re = fre + f * n_bins, *im = fim + f * n_bins;
            if (roundtrip) {
                int m = ac_istft_push_planar(sy, re, im, out);
                fwrite(out, sizeof(float), (size_t)m, fo);
            } else {
                fwrite(re, sizeof(float), (size_t)n_bins, fo);
                fwrite(im, sizeof(float), (size_t)n_bins, fo);
            }
        }
    }
    if (ac_stft_flush_planar(an, fre, fim) == 1) {
        if (roundtrip) {
            int m = ac_istft_push_planar(sy, fre, fim, out);
            fwrite(out, sizeof(float), (size_t)m, fo);
        } else {
            fwrite(fre, sizeof(float), (size_t)n_bins, fo);
            fwrite(fim, sizeof(float), (size_t)n_bins, fo);
        }
    }
    if (roundtrip) {
        int m = ac_istft_flush(sy, out);
        fwrite(out, sizeof(float), (size_t)m, fo);
    }
    fclose(fi); fclose(fo);
    fft_destroy(fft);
    return 0;
}
'''


@pytest.fixture(scope='module')
def audio_common_lib():
    """audiolib's NE10 static lib (default backend), built + located the
    way the AEC C tests document it (make -s -C ../audiolib/common BACKEND=ne10
    lib, then print-lib-path for this invocation's exact archive path)."""
    if shutil.which('make') is None:
        pytest.skip('no make available')
    subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'lib'],
        check=True, capture_output=True,
    )
    out = subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'print-lib-path'],
        check=True, capture_output=True, text=True,
    )
    lib = out.stdout.strip().splitlines()[-1]
    assert os.path.isfile(lib), lib
    return lib


@pytest.fixture(scope='module')
def driver(tmp_path_factory, audio_common_lib):
    """The framing driver: one program for every grid (n_fft is an argument)."""
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp('ulcnet_c')
    main_c = work / 'driver.c'
    main_c.write_text(_DRIVER)
    exe = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _ULCNET_DIR, '-I', _AC_INCLUDE, '-I', _AC_DIR,
         str(main_c), audio_common_lib, '-lm', '-o', str(exe)],
        check=True, capture_output=True,
    )
    return work, exe


def _run(driver_fx, mode, samples, n_fft=N_FFT):
    work, exe = driver_fx
    fin = work / 'in.f32'
    fout = work / 'out.f32'
    samples.astype(np.float32).tofile(fin)
    subprocess.run([str(exe), mode, str(fin), str(fout), str(n_fft)],
                   check=True)
    return np.fromfile(fout, dtype=np.float32)


def _python_frames(x, n_fft=N_FFT, hop=HOP):
    st = StreamSTFT(n_fft, hop, sqrt_hann_window(n_fft))
    frames = []
    for start in range(0, x.numel(), hop):
        frames += st.push(x[None, start:start + hop])
    frames += st.flush()
    return torch.stack(frames, dim=1)[0]          # [T, n_fft/2+1] complex


def test_48k_analysis_and_roundtrip_match_python(driver):
    """Numerical parity for the complete STFT/WOLA framing contract at 48 kHz.

    The pipeline build smoke catches dimensions and linking; this test also
    catches a correct-sized but incorrectly framed 513-bin implementation.
    """
    n_fft, hop = 1024, 512
    bins = n_fft // 2 + 1
    torch.manual_seed(31)
    x = torch.randn(hop * 24)
    ref_frames = _python_frames(x, n_fft, hop)

    raw = _run(driver, 'analysis', x.numpy(), n_fft)
    got = raw.reshape(-1, 2, bins)
    assert got.shape[0] == ref_frames.shape[0]
    got_complex = torch.complex(
        torch.from_numpy(got[:, 0]), torch.from_numpy(got[:, 1]))
    assert (got_complex - ref_frames).abs().max().item() < 4e-4

    roundtrip = torch.from_numpy(
        _run(driver, 'roundtrip', x.numpy(), n_fft)[:x.numel()])
    assert roundtrip.shape == x.shape
    assert (roundtrip - x).abs().max().item() < 4e-4


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


def test_ulcnet_compressed_mask_helper_matches_python(driver, audio_common_lib):
    work, _ = driver
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    source = work / 'ulc_mask.c'
    source.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include "aiaec_process.h"
#define BINS_MAX 513
int main(int argc, char **argv) {
    static float in[4][BINS_MAX], out[2][BINS_MAX];
    FILE *fi, *fo;
    int n_bins;
    if (argc != 4) return 2;
    n_bins = atoi(argv[3]);
    if (n_bins < 1 || n_bins > BINS_MAX) return 2;
    fi = fopen(argv[1], "rb"); fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 3;
    for (int row = 0; row < 4; ++row)
        if (fread(in[row], sizeof(float), (size_t)n_bins, fi) != (size_t)n_bins) return 4;
    aiaec_apply_ulcnet_compressed_mask(
        in[0], in[1], in[2], in[3], out[0], out[1], n_bins);
    for (int row = 0; row < 2; ++row)
        fwrite(out[row], sizeof(float), (size_t)n_bins, fo);
    fclose(fi); fclose(fo); return 0;
}
''')
    executable = work / 'ulc_mask'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR, '-I', _AC_INCLUDE, '-I', _AC_DIR,
         str(source), os.path.join(_ULCNET_DIR, 'ulcnet_mask_process.c'),
         os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True)
    torch.manual_seed(21)
    values = torch.randn(4, BINS).clamp(-2.0, 2.0)
    input_path = work / 'ulc_mask_in.f32'
    output_path = work / 'ulc_mask_out.f32'
    values.numpy().astype(np.float32).tofile(input_path)
    subprocess.run([str(executable), str(input_path), str(output_path),
                    str(BINS)], check=True)
    got = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(2, BINS)
    exponent = 0.3
    compressed = values[:2].sign() * values[:2].abs().pow(exponent)
    masked = torch.stack((
        compressed[0] * values[2] - compressed[1] * values[3],
        compressed[1] * values[2] + compressed[0] * values[3],
    ))
    expected = masked.sign() * masked.abs().pow(1.0 / exponent)
    assert (got - expected).abs().max().item() < 2e-5


def test_shared_real_and_complex_masks_match_python(driver, audio_common_lib):
    """Pins the host post-processing used by Align-CRUSE and CAGCRN."""
    work, _ = driver
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    source = work / 'shared_masks.c'
    source.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include "aiaec_process.h"
#define BINS_MAX 513
int main(int argc, char **argv) {
    static float in[5][BINS_MAX], out[4][BINS_MAX];
    FILE *fi, *fo;
    int n_bins;
    if (argc != 4) return 2;
    n_bins = atoi(argv[3]);
    if (n_bins < 1 || n_bins > BINS_MAX) return 2;
    fi = fopen(argv[1], "rb"); fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 3;
    for (int row = 0; row < 5; ++row)
        if (fread(in[row], sizeof(float), (size_t)n_bins, fi) != (size_t)n_bins) return 4;
    aiaec_apply_real_mask(in[0], in[1], in[2], out[0], out[1], n_bins);
    aiaec_apply_complex_mask(in[0], in[1], in[3], in[4], out[2], out[3], n_bins);
    for (int row = 0; row < 4; ++row)
        fwrite(out[row], sizeof(float), (size_t)n_bins, fo);
    fclose(fi); fclose(fo); return 0;
}
''')
    executable = work / 'shared_masks'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _AIAEC_DIR, '-I', _CAGCRN_DIR, '-I', _ALIGN_CRUSE_DIR, '-I', _AC_INCLUDE, '-I', _AC_DIR,
         str(source), os.path.join(_CAGCRN_DIR, 'cagcrn_process.c'),
         os.path.join(_ALIGN_CRUSE_DIR, 'align_cruse_process.c'),
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True)

    torch.manual_seed(24)
    values = torch.randn(5, BINS, dtype=torch.float32)
    input_path = work / 'shared_masks_in.f32'
    output_path = work / 'shared_masks_out.f32'
    values.numpy().tofile(input_path)
    subprocess.run([str(executable), str(input_path), str(output_path),
                    str(BINS)], check=True)
    got = torch.from_numpy(
        np.fromfile(output_path, dtype=np.float32)).reshape(4, BINS)
    real_expected = torch.stack((values[0] * values[2],
                                 values[1] * values[2]))
    complex_expected = torch.stack((
        values[0] * values[3] - values[1] * values[4],
        values[1] * values[3] + values[0] * values[4],
    ))
    torch.testing.assert_close(got[:2], real_expected, rtol=0, atol=0)
    torch.testing.assert_close(got[2:], complex_expected, rtol=0, atol=0)


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
#define BINS_MAX 513
/* The spectrum rings are the caller's: TIME_ORDER spectra each. */
int main(int argc, char **argv) {
    DeepVqeCcmState state;
    static float ring_re[DEEPVQE_TIME_ORDER * BINS_MAX], ring_im[DEEPVQE_TIME_ORDER * BINS_MAX];
    static float re[BINS_MAX], im[BINS_MAX];
    static float taps[BINS_MAX][DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2];
    static float out_re[BINS_MAX], out_im[BINS_MAX];
    FILE *fi, *fo;
    int n_bins;
    if (argc != 5) return 2;
    n_bins = atoi(argv[4]);
    if (n_bins < 3 || n_bins > BINS_MAX) return 2;
    fi = fopen(argv[1], "rb"); fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 3;
    if (deepvqe_ccm_init(&state, n_bins, ring_re, ring_im) != 0) return 5;
    for (int frame = 0; frame < atoi(argv[3]); ++frame) {
        const size_t n = (size_t)n_bins;
        if (fread(re, sizeof(float), n, fi) != n) return 4;
        if (fread(im, sizeof(float), n, fi) != n) return 4;
        if (fread(taps, sizeof(float), n * 3 * 3 * 2, fi) != n * 3 * 3 * 2) return 4;
        deepvqe_ccm_process(&state, re, im, (const float *)taps, out_re, out_im);
        fwrite(out_re, sizeof(float), n, fo);
        fwrite(out_im, sizeof(float), n, fo);
    }
    fclose(fi); fclose(fo); return 0;
}
''')
    executable = tmp_path / 'ccm'
    subprocess.run(
        [cc, '-O2', '-std=c99', '-ffp-contract=off',
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR,
         '-I', _DEEPVQE_S_DIR, '-I', _AC_INCLUDE, '-I', _AC_DIR,
         str(source), os.path.join(_DEEPVQE_S_DIR, 'deepvqe_process.c'),
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
    subprocess.run([str(executable), str(input_path), str(output_path),
                    str(frames), str(BINS)], check=True)
    got = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(
        frames, 2, BINS)
    got = torch.complex(got[:, 0], got[:, 1])
    expected = apply_causal_tf_filter(spectrum, taps, 3, 1)[0]
    assert (got - expected).abs().max().item() < 2e-5
