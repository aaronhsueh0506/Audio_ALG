#include "gtcrn_process.h"

#include <math.h>
#include <string.h>

#include "simd_kernel_nn.h"


/* Model-local DSP kernels (FFT / root-Hann / STFT / WOLA).
 * Deliberately NOT shared across models: porting is single-model,
 * so each model directory carries every kernel it runs. */

#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
#include <arm_neon.h>
#define DF_COMMON_HAVE_NEON 1
#else
#define DF_COMMON_HAVE_NEON 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void df_common_make_root_hann(float *window, int win_len) {
    for (int i = 0; i < win_len; ++i) {
        window[i] = sqrtf(0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)i / (float)win_len));
    }
}

static inline void df_common_analysis(FftHandle* fft, float *analysis_buf,
                                      const float *window,
                                      float *scratch_time,
                                      Complex *scratch_freq,
                                      const float *new_samples, int n_fft,
                                      int hop, float norm,
                                      float out[][2]) {
    memmove(analysis_buf, analysis_buf + hop,
            (size_t)(n_fft - hop) * sizeof(float));
    memcpy(analysis_buf + n_fft - hop, new_samples,
           (size_t)hop * sizeof(float));
    {
        int i = 0;
#if DF_COMMON_HAVE_NEON
        for (; i + 4 <= n_fft; i += 4) {
            vst1q_f32(scratch_time + i,
                      vmulq_f32(vld1q_f32(analysis_buf + i),
                                vld1q_f32(window + i)));
        }
#endif
        for (; i < n_fft; ++i) scratch_time[i] = analysis_buf[i] * window[i];
    }
    fft_forward(fft, scratch_time, scratch_freq);
    {
        int bins = n_fft / 2 + 1;
        for (int k = 0; k < bins; ++k) {
            out[k][0] = scratch_freq[k].r * norm;
            out[k][1] = scratch_freq[k].i * norm;
        }
    }
}

static inline void df_common_synthesis(FftHandle* fft,
                                       float *synthesis_buf,
                                       const float *window,
                                       float *scratch_time,
                                       Complex *scratch_freq,
                                       const float spec[][2],
                                       int n_fft, int hop, float inv_norm,
                                       float *output) {
    int bins = n_fft / 2 + 1;
    for (int k = 0; k < bins; ++k) {
        scratch_freq[k].r = spec[k][0] * inv_norm;
        scratch_freq[k].i = spec[k][1] * inv_norm;
    }
    fft_inverse(fft, scratch_freq, scratch_time);
    {
        int i = 0;
#if DF_COMMON_HAVE_NEON
        for (; i + 4 <= n_fft; i += 4) {
            vst1q_f32(scratch_time + i,
                      vmulq_f32(vld1q_f32(scratch_time + i),
                                vld1q_f32(window + i)));
        }
        for (i = 0; i + 4 <= hop; i += 4) {
            vst1q_f32(output + i,
                      vaddq_f32(vld1q_f32(synthesis_buf + i),
                                vld1q_f32(scratch_time + i)));
        }
#else
        for (; i < n_fft; ++i) scratch_time[i] *= window[i];
#endif
#if DF_COMMON_HAVE_NEON
        for (; i < hop; ++i)
            output[i] = synthesis_buf[i] + scratch_time[i];
#else
        for (i = 0; i < hop; ++i)
            output[i] = synthesis_buf[i] + scratch_time[i];
#endif
    }
    memcpy(synthesis_buf, scratch_time + hop,
           (size_t)(n_fft - hop) * sizeof(float));
    memset(synthesis_buf + n_fft - hop, 0, (size_t)hop * sizeof(float));
}

void gtcrn_process_init(GTCRNProcessState* state, FftHandle* fft)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->fft = fft;
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
    return skn_all_finite_f32(values, count);
}

#define GTCRN_MODEL_STATE_TENSORS \
    (GTCRN_MODEL_CONV_STATES + GTCRN_MODEL_TRA_GRUS + GTCRN_MODEL_DPGRNN_GRUS)

/* The sixteen state tensors in graph order (conv, h_tra, h_dpgrnn) with their
 * element counts; together they tile the struct. */
static void state_tensors(GTCRNModelState* state,
                          float* base[GTCRN_MODEL_STATE_TENSORS],
                          size_t count[GTCRN_MODEL_STATE_TENSORS])
{
    int n = 0;
    base[n] = &state->conv_enc0[0][0][0];
    count[n++] = sizeof(state->conv_enc0) / sizeof(float);
    base[n] = &state->conv_enc1[0][0][0];
    count[n++] = sizeof(state->conv_enc1) / sizeof(float);
    base[n] = &state->conv_enc2[0][0][0];
    count[n++] = sizeof(state->conv_enc2) / sizeof(float);
    base[n] = &state->conv_dec0[0][0][0];
    count[n++] = sizeof(state->conv_dec0) / sizeof(float);
    base[n] = &state->conv_dec1[0][0][0];
    count[n++] = sizeof(state->conv_dec1) / sizeof(float);
    base[n] = &state->conv_dec2[0][0][0];
    count[n++] = sizeof(state->conv_dec2) / sizeof(float);
    for (int i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
        base[n] = &state->h_tra[i][0][0][0];
        count[n++] = sizeof(state->h_tra[0]) / sizeof(float);
    }
    for (int i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
        base[n] = &state->h_dpgrnn[i][0][0][0];
        count[n++] = sizeof(state->h_dpgrnn[0]) / sizeof(float);
    }
}

int gtcrn_model_state_validate(GTCRNModelState* state)
{
    float* base[GTCRN_MODEL_STATE_TENSORS];
    size_t count[GTCRN_MODEL_STATE_TENSORS];
    if (state == NULL) return -1;
    state_tensors(state, base, count);
    for (int i = 0; i < GTCRN_MODEL_STATE_TENSORS; ++i) {
        if (!all_finite(base[i], count[i])) {
            gtcrn_model_state_init(state);
            return -1;
        }
    }
    return 0;
}

int gtcrn_model_state_inherit(
    GTCRNModelState* state,
    const float* const conv_out[GTCRN_MODEL_CONV_STATES],
    const float* const h_tra_out[GTCRN_MODEL_TRA_GRUS],
    const float* const h_dpgrnn_out[GTCRN_MODEL_DPGRNN_GRUS])
{
    float* base[GTCRN_MODEL_STATE_TENSORS];
    size_t count[GTCRN_MODEL_STATE_TENSORS];
    const float* src[GTCRN_MODEL_STATE_TENSORS];
    int n = 0;
    if (state == NULL || conv_out == NULL || h_tra_out == NULL ||
        h_dpgrnn_out == NULL) return -1;
    for (int i = 0; i < GTCRN_MODEL_CONV_STATES; ++i) src[n++] = conv_out[i];
    for (int i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) src[n++] = h_tra_out[i];
    for (int i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
        src[n++] = h_dpgrnn_out[i];
    }
    for (int i = 0; i < GTCRN_MODEL_STATE_TENSORS; ++i) {
        if (src[i] == NULL) return -1;
    }
    state_tensors(state, base, count);
    /* Validate every tensor that would be copied before touching any, so a
     * refusal leaves the state byte-identical. A tensor the runtime wrote in
     * place (same address) is skipped: it is not checked here and copying a
     * range onto itself is undefined. */
    for (int i = 0; i < GTCRN_MODEL_STATE_TENSORS; ++i) {
        if (src[i] != base[i] && !all_finite(src[i], count[i])) return -1;
    }
    for (int i = 0; i < GTCRN_MODEL_STATE_TENSORS; ++i) {
        if (src[i] != base[i]) {
            memcpy(base[i], src[i], count[i] * sizeof(float));
        }
    }
    return 0;
}

void gtcrn_analysis(GTCRNProcessState* state, const float* input,
                    float output[GTCRN_N_BINS][2])
{
    if (!state || !input || !output) return;
    df_common_analysis(state->fft, state->analysis_buf, state->window,
                       state->scratch_time, state->scratch_freq, input,
                       GTCRN_N_FFT, GTCRN_HOP_LEN, 1.0f, output);
}

void gtcrn_synthesis(GTCRNProcessState* state,
                     const float input[GTCRN_N_BINS][2],
                     float* output)
{
    if (!state || !input || !output) return;
    df_common_synthesis(state->fft, state->synthesis_buf, state->window,
                        state->scratch_time, state->scratch_freq, input,
                        GTCRN_N_FFT, GTCRN_HOP_LEN,
                        1.0f, output);
}

const char* gtcrn_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
