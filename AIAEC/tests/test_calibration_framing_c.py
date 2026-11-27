"""The calibration captures frame audio as the deployed streams do.

The DeepVQE-S, Align-ULCNet and DeepFilterNet streams analyse with audiolib's
``AcStft`` at its default configuration (center 0): every hop pushed yields
the frame over the last ``n_fft`` samples, zeros before the stream's first
sample. A tiny test main (written to a temp dir, never to the repo) runs that
configuration, a fresh ``AcStft`` per recording as a stream starts fresh, and
pushes a file's last partial hop completed with zeros, as a host reading the
file does; the frames ``iter_blocks`` (AIAEC) and ``calibration_io.stream_stft``
(DeepFilterNet, normalised) hand to the graph must match it, the first and
last frame of each recording included. The training's ``torch.stft
(center=True)`` reflects the start into the first frame (and the end into the
last, adding a frame when the file is a whole number of hops); the test shows
the frames differ, so the comparison has teeth.
"""

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC._calibration_common import iter_blocks
from AIAEC.dataset_gen import AecGrid
from AIAEC.dataset_gen.aec_features import sqrt_hann_window
from calibration_io import stream_stft
from test_aiaec_framing_backend_c import _AC_DIR, _AC_INCLUDE, \
    audio_common_lib  # noqa: F401  (a fixture)

_DRIVER = r'''
#include <stdio.h>
#include <stdlib.h>
#include "audio_stft.h"

/* argv: in.f32 out.f32 n_fft. The streams' configuration: the defaults,
 * the window carved by AcStft itself. */
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 3;
    const int n_fft = atoi(argv[3]);
    const int n_hop = n_fft / 2, n_bins = n_fft / 2 + 1;
    AcStftConfig cfg;
    ac_stft_config_default(&cfg, n_fft);
    size_t bytes = ac_stft_get_mem_size(&cfg);
    void *mem = NULL;
    if (bytes == 0 || posix_memalign(&mem, 16, bytes) != 0) return 4;
    AcStft *an = ac_stft_init(mem, bytes, &cfg);
    if (!an) return 5;
    float *hop = malloc(sizeof(float) * (size_t)n_hop);
    float *re = malloc(sizeof(float) * 2 * (size_t)n_bins);
    float *im = malloc(sizeof(float) * 2 * (size_t)n_bins);
    size_t got;
    while ((got = fread(hop, sizeof(float), (size_t)n_hop, fi)) > 0) {
        for (size_t i = got; i < (size_t)n_hop; ++i) hop[i] = 0.0f;
        int n = ac_stft_push_planar(an, hop, re, im);
        for (int f = 0; f < n; ++f) {
            fwrite(re + f * n_bins, sizeof(float), (size_t)n_bins, fo);
            fwrite(im + f * n_bins, sizeof(float), (size_t)n_bins, fo);
        }
    }
    fclose(fi); fclose(fo);
    return 0;
}
'''


@pytest.fixture(scope='module')
def board(tmp_path_factory, audio_common_lib):  # noqa: F811
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp('stream_framing')
    (work / 'driver.c').write_text(_DRIVER)
    exe = work / 'driver'
    subprocess.run([cc, '-O2', '-std=c99', '-ffp-contract=off',
                    '-I', _AC_INCLUDE, '-I', _AC_DIR, str(work / 'driver.c'),
                    audio_common_lib, '-lm', '-o', str(exe)],
                   check=True, capture_output=True)

    def frames(samples, n_fft):
        """The stream's frames of one recording: [frames, bins] complex."""
        samples.astype(np.float32).tofile(work / 'in.f32')
        subprocess.run([str(exe), str(work / 'in.f32'), str(work / 'out.f32'),
                        str(n_fft)], check=True)
        raw = np.fromfile(work / 'out.f32', dtype=np.float32)
        raw = raw.reshape(-1, 2, n_fft // 2 + 1)
        return raw[:, 0] + 1j * raw[:, 1]

    return frames


def _recording(seed, samples):
    """Audio whose start is far from silent and not symmetric, so a reflected
    first frame and a zero-padded one differ."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.6, -0.2, samples)
    return (ramp + 0.1 * rng.standard_normal(samples)).astype(np.float32)


@pytest.mark.parametrize('tail', [37, 0])
@pytest.mark.parametrize('n_fft', [256, 512, 1024])
def test_the_capture_frames_each_recording_as_a_fresh_stream_does(board,
                                                                   n_fft, tail):
    hop = n_fft // 2
    window = sqrt_hann_window(n_fft)
    for seed in (1, 2):              # a second recording: a fresh stream
        x = _recording(seed, hop * 20 + tail)
        want = board(x, n_fft)
        got = stream_stft(torch.from_numpy(x), n_fft, hop, n_fft,
                          window).T.numpy()
        assert got.shape == want.shape == (-(-len(x) // hop), n_fft // 2 + 1)
        assert np.abs(got[0] - want[0]).max() < 4e-4      # first frame
        assert np.abs(got[1] - want[1]).max() < 4e-4      # second frame
        assert np.abs(got[-1] - want[-1]).max() < 4e-4    # last frame
        assert np.abs(got - want).max() < 4e-4
        # Normalised, as the DeepFilterNet captures take it.
        normed = stream_stft(torch.from_numpy(x), n_fft, hop, n_fft, window,
                             normalized=True).T.numpy()
        assert np.abs(normed * np.sqrt(n_fft) - want).max() < 4e-4
        # The training's centred framing is not the board's: the first frame
        # differs, the frames between are the same, and a whole number of
        # hops gets one frame more.
        centred = torch.stft(torch.from_numpy(x), n_fft, hop, n_fft,
                             window=window, center=True,
                             return_complex=True).T.numpy()
        assert centred.shape[0] == want.shape[0] + (1 if tail == 0 else 0)
        assert np.abs(centred[0] - want[0]).max() > 0.1
        assert np.abs(centred[1:want.shape[0] - 1]
                      - want[1:-1]).max() < 4e-4


def test_iter_blocks_hands_the_graph_the_stream_frames(board, tmp_path):
    grid = AecGrid(16000, 512, 512, 256)
    mic, far = _recording(3, 256 * 12 + 50), _recording(4, 256 * 12 + 50)
    sf.write(tmp_path / 'mic.wav', mic, 16000, subtype='FLOAT')
    sf.write(tmp_path / 'far.wav', far, 16000, subtype='FLOAT')
    blocks = [block for block, _ in iter_blocks(
        'DeepVQE_S', None, grid, tmp_path / 'mic.wav', tmp_path / 'far.wav', 1)]
    for name, x in (('mic', mic), ('far', far)):
        want = board(x, 512)
        got = np.stack([block[name][0, :, 0] + 1j * block[name][0, :, 1]
                        for block in blocks])
        assert got.shape == want.shape
        assert np.abs(got - want).max() < 4e-4
