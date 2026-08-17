#include "dfn_aenr_process.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void dfn_aenr_model_io_init(DfnAenrModelIOState *state) {
    if (state) memset(state, 0, sizeof(*state));
}

int dfn_aenr_model_io_push_features(
    DfnAenrModelIOState *state,
    const float error_erb[DFN2_N_ERB],
    const float error_spec[2][DFN2_DF_BINS],
    const float far_erb[DFN2_N_ERB],
    const float far_spec[2][DFN2_DF_BINS]) {
    if (!state || !error_erb || !error_spec || !far_erb || !far_spec)
        return -1;
    dfn2_model_io_push_erb_window(state->error_erb_window, error_erb);
    dfn2_model_io_push_spec_window(state->error_spec_window, error_spec);
    dfn2_model_io_push_erb_window(state->far_erb_window, far_erb);
    dfn2_model_io_push_spec_window(state->far_spec_window, far_spec);
    if (state->feature_frames_seen < 2U) ++state->feature_frames_seen;
    return state->feature_frames_seen == 2U ? 1 : 0;
}

int dfn_aenr_model_io_commit_state(
    DfnAenrModelIOState *state,
    const float encoder_hidden_next[DFN2_MODEL_ENCODER_GRU_LAYERS]
                                   [DFN2_MODEL_GRU_HIDDEN],
    const float erb_hidden_next[DFN2_MODEL_ERB_GRU_LAYERS]
                               [DFN2_MODEL_GRU_HIDDEN],
    const float df_hidden_next[DFN2_MODEL_DF_GRU_LAYERS]
                              [DFN2_MODEL_GRU_HIDDEN],
    const float pathway_history_next[DFN2_MODEL_ENCODER_CHANNELS]
                                    [DFN2_MODEL_DF_PATHWAY_HISTORY]
                                    [DFN2_DF_BINS]) {
    if (!state) return -1;
    return dfn2_model_io_commit_arrays(
        state->encoder_gru_hidden, state->erb_gru_hidden,
        state->df_gru_hidden, state->df_convp_history,
        encoder_hidden_next, erb_hidden_next, df_hidden_next,
        pathway_history_next);
}

void dfn_aenr_make_window(float window[DFN2_N_FFT]) {
    int index;
    if (!window) return;
    for (index = 0; index < DFN2_N_FFT; ++index) {
        window[index] = sqrtf(0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)index / (float)DFN2_N_FFT));
    }
}

static int analysis_init(DfnAenrAnalysis *analysis, FftHandle *fft,
                         const float *window) {
    if (!analysis || !fft || !window ||
        fft_get_n_freqs(fft) != DFN2_N_BINS) return -1;
    memset(analysis, 0, sizeof(*analysis));
    analysis->fft = fft;
    analysis->window = window;
    return 0;
}

static void analysis_transform(
    DfnAenrAnalysis *analysis, const float *input,
    float real[DFN2_N_BINS], float imag[DFN2_N_BINS]) {
    const float scale = 1.0f / sqrtf((float)DFN2_N_FFT);
    int index;
    for (index = 0; index < DFN2_N_FFT; ++index)
        analysis->segment[index] = input[index] * analysis->window[index];
    fft_forward_scratch(analysis->fft, analysis->segment,
                        analysis->spectrum);
    for (index = 0; index < DFN2_N_BINS; ++index) {
        real[index] = analysis->spectrum[index].r * scale;
        imag[index] = analysis->spectrum[index].i * scale;
    }
}

static int analysis_push(
    DfnAenrAnalysis *analysis, const float input[DFN2_HOP_LEN],
    float real[2][DFN2_N_BINS], float imag[2][DFN2_N_BINS]) {
    int index;
    if (!analysis || !input || !real || !imag) return -1;
    memmove(analysis->history, analysis->history + DFN2_HOP_LEN,
            (DFN2_N_FFT - DFN2_HOP_LEN) * sizeof(float));
    memcpy(analysis->history + DFN2_N_FFT - DFN2_HOP_LEN, input,
           DFN2_HOP_LEN * sizeof(float));
    if (analysis->hops_seen < 3) ++analysis->hops_seen;
    if (analysis->hops_seen == 1) return 0;
    if (analysis->hops_seen == 2) {
        for (index = 0; index < DFN2_HOP_LEN; ++index)
            analysis->segment[index] =
                analysis->history[DFN2_HOP_LEN - index];
        memcpy(analysis->segment + DFN2_HOP_LEN, analysis->history,
               DFN2_HOP_LEN * sizeof(float));
        analysis_transform(analysis, analysis->segment, real[0], imag[0]);
        analysis_transform(analysis, analysis->history, real[1], imag[1]);
        return 2;
    }
    analysis_transform(analysis, analysis->history, real[0], imag[0]);
    return 1;
}

static int analysis_flush(
    DfnAenrAnalysis *analysis,
    float real[2][DFN2_N_BINS], float imag[2][DFN2_N_BINS]) {
    int index;
    if (!analysis || !real || !imag) return -1;
    if (analysis->hops_seen < 2) return 0;
    memcpy(analysis->segment,
           analysis->history + DFN2_N_FFT - DFN2_HOP_LEN,
           DFN2_HOP_LEN * sizeof(float));
    for (index = 0; index < DFN2_HOP_LEN; ++index)
        analysis->segment[DFN2_HOP_LEN + index] =
            analysis->history[DFN2_N_FFT - 2 - index];
    analysis_transform(analysis, analysis->segment, real[0], imag[0]);
    return 1;
}

int dfn_aenr_process_init(DfnAenrProcessState *state, FftHandle *fft,
                          const float window[DFN2_N_FFT]) {
    if (!state || !fft || !window ||
        fft_get_n_freqs(fft) != DFN2_N_BINS) return -1;
    memset(state, 0, sizeof(*state));
    dfn2_state_init(&state->error);
    dfn2_state_init(&state->far);
    if (analysis_init(&state->error_analysis, fft, window) != 0 ||
        analysis_init(&state->far_analysis, fft, window) != 0) return -1;
    state->synthesis.fft = fft;
    state->synthesis.window = window;
    return 0;
}

int dfn_aenr_analysis_push(
    DfnAenrProcessState *state,
    const float error_hop[DFN2_HOP_LEN],
    const float far_hop[DFN2_HOP_LEN],
    float error_re[2][DFN2_N_BINS],
    float error_im[2][DFN2_N_BINS],
    float far_re[2][DFN2_N_BINS],
    float far_im[2][DFN2_N_BINS]) {
    int error_count, far_count;
    if (!state) return -1;
    error_count = analysis_push(&state->error_analysis, error_hop,
                                error_re, error_im);
    far_count = analysis_push(&state->far_analysis, far_hop,
                              far_re, far_im);
    return error_count >= 0 && error_count == far_count ? error_count : -1;
}

int dfn_aenr_analysis_flush(
    DfnAenrProcessState *state,
    float error_re[2][DFN2_N_BINS],
    float error_im[2][DFN2_N_BINS],
    float far_re[2][DFN2_N_BINS],
    float far_im[2][DFN2_N_BINS]) {
    int error_count, far_count;
    if (!state) return -1;
    error_count = analysis_flush(&state->error_analysis, error_re, error_im);
    far_count = analysis_flush(&state->far_analysis, far_re, far_im);
    return error_count >= 0 && error_count == far_count ? error_count : -1;
}

void dfn_aenr_compute_features(DfnAenrProcessState *state,
                               const float error_re[DFN2_N_BINS],
                               const float error_im[DFN2_N_BINS],
                               const float far_re[DFN2_N_BINS],
                               const float far_im[DFN2_N_BINS],
                               float error_erb[DFN2_N_ERB],
                               float error_spec[2 * DFN2_DF_BINS],
                               float far_erb[DFN2_N_ERB],
                               float far_spec[2 * DFN2_DF_BINS]) {
    if (!state) return;
    dfn2_compute_features(&state->error, error_re, error_im,
                          error_erb, error_spec);
    dfn2_compute_features(&state->far, far_re, far_im,
                          far_erb, far_spec);
}

int dfn_aenr_compose_stream(DfnAenrProcessState *state,
                            const float error_re[DFN2_N_BINS],
                            const float error_im[DFN2_N_BINS],
                            int heads_valid,
                            const float erb_mask[DFN2_N_ERB],
                            const float coefs[DFN2_DF_BINS * DFN2_DF_ORDER * 2],
                            float alpha, float atten_lim_db,
                            float output_re[DFN2_N_BINS],
                            float output_im[DFN2_N_BINS],
                            long long *output_frame_index) {
    if (!state) return -1;
    return dfn2_compose_stream(&state->error, error_re, error_im,
                               heads_valid, erb_mask, coefs, alpha,
                               atten_lim_db, output_re, output_im,
                               output_frame_index);
}

int dfn_aenr_synthesis_push(DfnAenrProcessState *state,
                            const float real[DFN2_N_BINS],
                            const float imag[DFN2_N_BINS],
                            float output[DFN2_HOP_LEN]) {
    DfnAenrSynthesis *synthesis;
    const float scale = sqrtf((float)DFN2_N_FFT);
    int index;
    if (!state || !real || !imag || !output) return -1;
    synthesis = &state->synthesis;
    for (index = 0; index < DFN2_N_BINS; ++index) {
        synthesis->spectrum[index].r = real[index];
        synthesis->spectrum[index].i = imag[index];
    }
    fft_inverse_scratch(synthesis->fft, synthesis->spectrum,
                        synthesis->time);
    for (index = 0; index < DFN2_N_FFT; ++index) {
        const float window = synthesis->window[index];
        synthesis->overlap[index] +=
            synthesis->time[index] * scale * window;
        synthesis->envelope[index] += window * window;
    }
    if (synthesis->frames_seen < 2) ++synthesis->frames_seen;
    if (synthesis->frames_seen > 1) {
        for (index = 0; index < DFN2_HOP_LEN; ++index) {
            const float envelope = synthesis->envelope[index];
            output[index] = synthesis->overlap[index] /
                (envelope > 1e-11f ? envelope : 1e-11f);
        }
    }
    memmove(synthesis->overlap,
            synthesis->overlap + DFN2_HOP_LEN,
            (DFN2_N_FFT - DFN2_HOP_LEN) * sizeof(float));
    memset(synthesis->overlap + DFN2_N_FFT - DFN2_HOP_LEN, 0,
           DFN2_HOP_LEN * sizeof(float));
    memmove(synthesis->envelope,
            synthesis->envelope + DFN2_HOP_LEN,
            (DFN2_N_FFT - DFN2_HOP_LEN) * sizeof(float));
    memset(synthesis->envelope + DFN2_N_FFT - DFN2_HOP_LEN, 0,
           DFN2_HOP_LEN * sizeof(float));
    return synthesis->frames_seen > 1 ? DFN2_HOP_LEN : 0;
}

int dfn_aenr_synthesis_flush(DfnAenrProcessState *state,
                             float output[DFN2_N_FFT]) {
    DfnAenrSynthesis *synthesis;
    int index;
    if (!state || !output) return -1;
    synthesis = &state->synthesis;
    if (synthesis->frames_seen == 0) return 0;
    for (index = 0; index < DFN2_N_FFT - DFN2_HOP_LEN; ++index) {
        const float envelope = synthesis->envelope[index];
        output[index] = synthesis->overlap[index] /
            (envelope > 1e-11f ? envelope : 1e-11f);
    }
    return DFN2_N_FFT - DFN2_HOP_LEN;
}
