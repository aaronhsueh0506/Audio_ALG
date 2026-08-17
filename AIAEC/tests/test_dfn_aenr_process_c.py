"""C/Python parity for DeepFilterNet-AENR's centered 48 kHz frontend."""

import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT
from AIAEC.dataset_gen.aec_features import sqrt_hann_window


_HERE = os.path.dirname(os.path.abspath(__file__))
_AIAEC = os.path.dirname(_HERE)
_DFN_AENR = os.path.join(_AIAEC, 'DeepFilterNet_AENR')
_DFN2 = os.path.abspath(os.path.join(_AIAEC, '..', 'AINR', 'DeepFilterNet2'))
_AUDIO_COMMON = os.path.abspath(os.path.join(_AIAEC, '..', '..', 'audio_common'))
_AC_INCLUDE = os.path.join(_AUDIO_COMMON, 'include')

N_FFT, HOP, BINS = 1024, 512, 513

_DRIVER = r'''
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "dfn_aenr_process.h"

int main(int argc, char **argv) {
    static DfnAenrProcessState state;
    static DfnAenrModelIOState model_io;
    static DfnAenrModelIOState committed_model_io;
    static float encoder_next[DFN2_MODEL_ENCODER_GRU_LAYERS]
                             [DFN2_MODEL_GRU_HIDDEN];
    static float erb_next[DFN2_MODEL_ERB_GRU_LAYERS]
                         [DFN2_MODEL_GRU_HIDDEN];
    static float df_next[DFN2_MODEL_DF_GRU_LAYERS]
                        [DFN2_MODEL_GRU_HIDDEN];
    static float pathway_next[DFN2_MODEL_ENCODER_CHANNELS]
                              [DFN2_MODEL_DF_PATHWAY_HISTORY]
                              [DFN2_DF_BINS];
    static float window[DFN2_N_FFT];
    float hop[DFN2_HOP_LEN];
    float error_erb[DFN2_N_ERB] = {0};
    float far_erb[DFN2_N_ERB] = {0};
    float error_spec[2][DFN2_DF_BINS] = {{0}};
    float far_spec[2][DFN2_DF_BINS] = {{0}};
    float er[2][DFN2_N_BINS], ei[2][DFN2_N_BINS];
    float fr[2][DFN2_N_BINS], fi[2][DFN2_N_BINS];
    float output[DFN2_N_FFT];
    FftHandle *fft;
    FILE *input, *result;
    int roundtrip;
    if (argc != 4) return 2;
    dfn_aenr_model_io_init(&model_io);
    error_erb[3] = 1.0f; far_erb[3] = 2.0f;
    error_spec[1][4] = 3.0f; far_spec[1][4] = 4.0f;
    if (dfn_aenr_model_io_push_features(
            &model_io, error_erb, error_spec, far_erb, far_spec) != 0)
        return 20;
    error_erb[3] = 5.0f; far_erb[3] = 6.0f;
    if (dfn_aenr_model_io_push_features(
            &model_io, error_erb, error_spec, far_erb, far_spec) != 1)
        return 21;
    if (model_io.error_erb_window[1][3] != 1.0f ||
        model_io.error_erb_window[2][3] != 5.0f ||
        model_io.far_erb_window[1][3] != 2.0f ||
        model_io.far_erb_window[2][3] != 6.0f ||
        model_io.error_spec_window[1][2][4] != 3.0f ||
        model_io.far_spec_window[1][2][4] != 4.0f)
        return 22;
    encoder_next[0][0] = 7.0f; erb_next[0][0] = 8.0f;
    df_next[0][0] = 9.0f; pathway_next[0][0][0] = 10.0f;
    if (dfn_aenr_model_io_commit_state(
            &model_io, encoder_next, erb_next, df_next, pathway_next) != 0 ||
        model_io.encoder_gru_hidden[0][0] != 7.0f ||
        model_io.erb_gru_hidden[0][0] != 8.0f ||
        model_io.df_gru_hidden[0][0] != 9.0f ||
        model_io.df_convp_history[0][0][0] != 10.0f)
        return 23;
    committed_model_io = model_io;
    pathway_next[DFN2_MODEL_ENCODER_CHANNELS - 1]
                [DFN2_MODEL_DF_PATHWAY_HISTORY - 1]
                [DFN2_DF_BINS - 1] = NAN;
    if (dfn_aenr_model_io_commit_state(
            &model_io, encoder_next, erb_next, df_next, pathway_next) == 0 ||
        memcmp(&model_io, &committed_model_io, sizeof(model_io)) != 0)
        return 25;
    /* DFN-AENR owns four feature windows but inherits every DFN2 state
     * extent; the shared dfn2_model_io helpers write through these arrays, so
     * a silent re-declaration on either side would corrupt state rather than
     * fail to compile. */
    if (DFN_AENR_MODEL_IO_LAYOUT_VERSION != DFN2_MODEL_IO_LAYOUT_VERSION ||
        sizeof(model_io.error_erb_window) !=
            sizeof(((DFN2ModelIOState *)0)->erb_window) ||
        sizeof(model_io.far_erb_window) !=
            sizeof(((DFN2ModelIOState *)0)->erb_window) ||
        sizeof(model_io.error_spec_window) !=
            sizeof(((DFN2ModelIOState *)0)->spec_window) ||
        sizeof(model_io.far_spec_window) !=
            sizeof(((DFN2ModelIOState *)0)->spec_window) ||
        sizeof(model_io.encoder_gru_hidden) !=
            sizeof(((DFN2ModelIOState *)0)->encoder_gru_hidden) ||
        sizeof(model_io.erb_gru_hidden) !=
            sizeof(((DFN2ModelIOState *)0)->erb_gru_hidden) ||
        sizeof(model_io.df_gru_hidden) !=
            sizeof(((DFN2ModelIOState *)0)->df_gru_hidden) ||
        sizeof(model_io.df_convp_history) !=
            sizeof(((DFN2ModelIOState *)0)->df_convp_history))
        return 24;
    input = fopen(argv[2], "rb"); result = fopen(argv[3], "wb");
    if (!input || !result) return 3;
    fft = fft_create(DFN2_N_FFT);
    if (!fft) return 4;
    dfn_aenr_make_window(window);
    if (dfn_aenr_process_init(&state, fft, window) != 0) return 5;
    roundtrip = argv[1][0] == 'r';
    while (fread(hop, sizeof(float), DFN2_HOP_LEN, input) == DFN2_HOP_LEN) {
        int count = dfn_aenr_analysis_push(
            &state, hop, hop, er, ei, fr, fi);
        if (count < 0) return 6;
        for (int frame = 0; frame < count; ++frame) {
            if (roundtrip) {
                int samples = dfn_aenr_synthesis_push(
                    &state, er[frame], ei[frame], output);
                if (samples < 0) return 7;
                fwrite(output, sizeof(float), (size_t)samples, result);
            } else {
                fwrite(er[frame], sizeof(float), DFN2_N_BINS, result);
                fwrite(ei[frame], sizeof(float), DFN2_N_BINS, result);
            }
        }
    }
    {
        int count = dfn_aenr_analysis_flush(&state, er, ei, fr, fi);
        if (count < 0) return 8;
        for (int frame = 0; frame < count; ++frame) {
            if (roundtrip) {
                int samples = dfn_aenr_synthesis_push(
                    &state, er[frame], ei[frame], output);
                fwrite(output, sizeof(float), (size_t)samples, result);
            } else {
                fwrite(er[frame], sizeof(float), DFN2_N_BINS, result);
                fwrite(ei[frame], sizeof(float), DFN2_N_BINS, result);
            }
        }
    }
    if (roundtrip) {
        int samples = dfn_aenr_synthesis_flush(&state, output);
        fwrite(output, sizeof(float), (size_t)samples, result);
    }
    fclose(input); fclose(result); fft_destroy(fft); return 0;
}
'''


@pytest.fixture(scope='module')
def driver(tmp_path_factory):
    compiler = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if compiler is None or shutil.which('make') is None:
        pytest.skip('C build tools are unavailable')
    subprocess.run(['make', '-s', '-C', _AUDIO_COMMON,
                    'BACKEND=kiss', 'lib'], check=True)
    result = subprocess.run(
        ['make', '-s', '-C', _AUDIO_COMMON, 'BACKEND=kiss',
         'print-lib-path'], check=True, capture_output=True, text=True)
    library = result.stdout.strip().splitlines()[-1]
    work = tmp_path_factory.mktemp('dfn_aenr_c')
    source = work / 'driver.c'
    source.write_text(_DRIVER)
    executable = work / 'driver'
    subprocess.run([
        compiler, '-O2', '-std=c11', '-ffp-contract=off',
        '-I', _DFN_AENR, '-I', _DFN2, '-I', _AC_INCLUDE,
        str(source), os.path.join(_DFN_AENR, 'dfn_aenr_process.c'),
        os.path.join(_DFN2, 'dfn2_process.c'),
        os.path.join(_DFN2, 'dfn2_model_io.c'), library, '-lm',
        '-o', str(executable),
    ], check=True, capture_output=True)
    return work, executable


def _run(driver, mode, samples):
    work, executable = driver
    source = work / 'input.f32'
    output = work / 'output.f32'
    samples.numpy().astype(np.float32).tofile(source)
    subprocess.run([str(executable), mode, str(source), str(output)],
                   check=True)
    return np.fromfile(output, dtype=np.float32)


def _reference_frames(samples):
    stream = StreamSTFT(N_FFT, HOP, sqrt_hann_window(N_FFT))
    frames = []
    for start in range(0, samples.numel(), HOP):
        frames.extend(stream.push(samples[None, start:start + HOP]))
    frames.extend(stream.flush())
    return torch.stack(frames, dim=1)[0] * (N_FFT ** -0.5)


def test_centered_normalized_analysis_matches_python(driver):
    torch.manual_seed(71)
    samples = torch.randn(HOP * 20)
    reference = _reference_frames(samples)
    raw = _run(driver, 'analysis', samples).reshape(-1, 2, BINS)
    actual = torch.complex(torch.from_numpy(raw[:, 0]),
                           torch.from_numpy(raw[:, 1]))
    assert actual.shape == reference.shape
    assert (actual - reference).abs().max().item() < 3e-5
    assert (actual[1:] - reference[:-1]).abs().max().item() > 0.1


def test_centered_normalized_roundtrip_matches_input(driver):
    torch.manual_seed(72)
    samples = torch.randn(HOP * 20)
    actual = torch.from_numpy(_run(driver, 'roundtrip', samples))[
        :samples.numel()]

    analysis = StreamSTFT(N_FFT, HOP, sqrt_hann_window(N_FFT))
    synthesis = StreamISTFT(N_FFT, HOP, sqrt_hann_window(N_FFT))
    frames = []
    for start in range(0, samples.numel(), HOP):
        frames.extend(analysis.push(samples[None, start:start + HOP]))
    frames.extend(analysis.flush())
    pieces = [synthesis.push(frame) for frame in frames]
    emitted = sum(piece.shape[-1] for piece in pieces)
    pieces.append(synthesis.flush(length=samples.numel(),
                                  already_emitted=emitted))
    reference = torch.cat(pieces, dim=-1)[0, :samples.numel()]
    assert (actual - reference).abs().max().item() < 3e-4
    assert (actual - samples).abs().max().item() < 3e-4
