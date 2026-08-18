#include "gtcrn_process.h"

#include <math.h>
#include <string.h>

#include "../dfn_process_common.h"

void gtcrn_process_init(GTCRNProcessState* state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    df_common_make_root_hann(state->window, GTCRN_WIN_LEN);
}

/* Band one already-computed full-band channel: low bins pass through, the
 * high 192 bins accumulate into 64 bands via the committed forward table
 * (bin-major, so the inner loop runs contiguously over bands and
 * auto-vectorizes). */
static void erb_band_channel(const float* full, const float* erb_fwd,
                             float* banded)
{
    for (int k = 0; k < GTCRN_MODEL_ERB_KEPT; ++k) banded[k] = full[k];
    for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b)
        banded[GTCRN_MODEL_ERB_KEPT + b] = 0.0f;
    for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
        float value = full[GTCRN_MODEL_ERB_KEPT + k];
        const float* row = erb_fwd + (size_t)k * GTCRN_MODEL_ERB_HIGH_BANDS;
        float* out = banded + GTCRN_MODEL_ERB_KEPT;
        for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b)
            out[b] += row[b] * value;
    }
}

void gtcrn_model_input(const float spectrum[GTCRN_N_BINS][2],
                       const float* erb_fwd,
                       float mag[GTCRN_MODEL_ERB_BANDS],
                       float real_part[GTCRN_MODEL_ERB_BANDS],
                       float imag_part[GTCRN_MODEL_ERB_BANDS])
{
    float full_mag[GTCRN_N_BINS];
    float full_re[GTCRN_N_BINS];
    float full_im[GTCRN_N_BINS];
    if (!spectrum || !erb_fwd || !mag || !real_part || !imag_part) return;
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        float re = spectrum[k][0];
        float im = spectrum[k][1];
        full_mag[k] = sqrtf(re * re + im * im + 1e-12f);
        full_re[k] = re;
        full_im[k] = im;
    }
    erb_band_channel(full_mag, erb_fwd, mag);
    erb_band_channel(full_re, erb_fwd, real_part);
    erb_band_channel(full_im, erb_fwd, imag_part);
}

void gtcrn_model_output(const float mask_erb[GTCRN_MODEL_ERB_BANDS][2],
                        const float* erb_inv,
                        const float spectrum[GTCRN_N_BINS][2],
                        float enhanced[GTCRN_N_BINS][2])
{
    float mask_re[GTCRN_N_BINS];
    float mask_im[GTCRN_N_BINS];
    if (!mask_erb || !erb_inv || !spectrum || !enhanced) return;
    for (int k = 0; k < GTCRN_MODEL_ERB_KEPT; ++k) {
        mask_re[k] = mask_erb[k][0];
        mask_im[k] = mask_erb[k][1];
    }
    for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
        mask_re[GTCRN_MODEL_ERB_KEPT + k] = 0.0f;
        mask_im[GTCRN_MODEL_ERB_KEPT + k] = 0.0f;
    }
    /* Inverse table is band-major: the inner loop runs contiguously over
     * the 192 high bins and auto-vectorizes. */
    for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b) {
        float band_re = mask_erb[GTCRN_MODEL_ERB_KEPT + b][0];
        float band_im = mask_erb[GTCRN_MODEL_ERB_KEPT + b][1];
        const float* row = erb_inv + (size_t)b * GTCRN_MODEL_ERB_HIGH_BINS;
        for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
            mask_re[GTCRN_MODEL_ERB_KEPT + k] += row[k] * band_re;
            mask_im[GTCRN_MODEL_ERB_KEPT + k] += row[k] * band_im;
        }
    }
    /* Complex ratio mask, mirroring model.py's Mask exactly. */
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        float sr = spectrum[k][0], si = spectrum[k][1];
        float mr = mask_re[k], mi = mask_im[k];
        enhanced[k][0] = sr * mr - si * mi;
        enhanced[k][1] = si * mr + sr * mi;
    }
}

void gtcrn_model_state_init(GTCRNModelState* state)
{
    if (state != NULL) memset(state, 0, sizeof(*state));
}

static int all_finite(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

int gtcrn_model_state_commit(GTCRNModelState* state,
                             const float* conv_cache_out,
                             const float* const h_tra_out[GTCRN_MODEL_TRA_GRUS],
                             const float* const h_dpgrnn_out[GTCRN_MODEL_DPGRNN_GRUS])
{
    if (state == NULL || conv_cache_out == NULL || h_tra_out == NULL ||
        h_dpgrnn_out == NULL) return -1;
    for (int i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
        if (h_tra_out[i] == NULL) return -1;
    }
    for (int i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
        if (h_dpgrnn_out[i] == NULL) return -1;
    }
    /* Validate every state tensor BEFORE writing any of them. A partial
     * commit would leave the recurrent state a mix of two invocations, which
     * the next call cannot distinguish from a healthy state; refusing the
     * whole batch keeps the last good state replayable. */
    if (!all_finite(conv_cache_out,
                    sizeof(state->conv_cache) / sizeof(float))) return -1;
    for (int i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
        if (!all_finite(h_tra_out[i],
                        sizeof(state->h_tra[0]) / sizeof(float))) return -1;
    }
    for (int i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
        if (!all_finite(h_dpgrnn_out[i],
                        sizeof(state->h_dpgrnn[0]) / sizeof(float))) return -1;
    }
    memcpy(state->conv_cache, conv_cache_out, sizeof(state->conv_cache));
    for (int i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
        memcpy(state->h_tra[i], h_tra_out[i], sizeof(state->h_tra[0]));
    }
    for (int i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
        memcpy(state->h_dpgrnn[i], h_dpgrnn_out[i],
               sizeof(state->h_dpgrnn[0]));
    }
    return 0;
}

void gtcrn_analysis(GTCRNProcessState* state, const float* input,
                    float output[GTCRN_N_BINS][2])
{
    if (!state || !input || !output) return;
    df_common_analysis(state->analysis_buf, state->window,
                       state->scratch_re, state->scratch_im, input,
                       GTCRN_N_FFT, GTCRN_HOP_LEN, 1.0f,
                       state->scratch_re, state->scratch_im);
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        output[k][0] = state->scratch_re[k];
        output[k][1] = state->scratch_im[k];
    }
}

void gtcrn_synthesis(GTCRNProcessState* state,
                     const float input[GTCRN_N_BINS][2],
                     float* output)
{
    if (!state || !input || !output) return;
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        state->scratch_re[k] = input[k][0];
        state->scratch_im[k] = input[k][1];
    }
    df_common_synthesis(state->synthesis_buf, state->window,
                        state->scratch_re, state->scratch_im,
                        state->scratch_re, state->scratch_im,
                        GTCRN_N_FFT, GTCRN_HOP_LEN,
                        1.0f, output);
}

const char* gtcrn_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
