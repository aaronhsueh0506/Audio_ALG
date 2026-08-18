#include "gtcrn_process.h"

#include <math.h>
#include <string.h>


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

static inline void df_common_fft(float *re, float *im, int n, int inverse) {
    int j = 0;
    for (int i = 0; i < n; ++i) {
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
        {
            int m = n >> 1;
            while (m >= 1 && j >= m) { j -= m; m >>= 1; }
            j += m;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float angle = (inverse ? 1.0f : -1.0f) *
            2.0f * (float)M_PI / (float)len;
        float step_re = cosf(angle);
        float step_im = sinf(angle);
        for (int base = 0; base < n; base += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            int k = 0;
#if DF_COMMON_HAVE_NEON
            for (; k + 4 <= len / 2; k += 4) {
                float tw_re[4], tw_im[4];
                for (int lane = 0; lane < 4; ++lane) {
                    float next_re, next_im;
                    tw_re[lane] = cur_re;
                    tw_im[lane] = cur_im;
                    next_re = cur_re * step_re - cur_im * step_im;
                    next_im = cur_re * step_im + cur_im * step_re;
                    cur_re = next_re;
                    cur_im = next_im;
                }
                {
                    int u = base + k, v = u + len / 2;
                    float32x4_t ur = vld1q_f32(re + u);
                    float32x4_t ui = vld1q_f32(im + u);
                    float32x4_t vr = vld1q_f32(re + v);
                    float32x4_t vi = vld1q_f32(im + v);
                    float32x4_t wr = vld1q_f32(tw_re);
                    float32x4_t wi = vld1q_f32(tw_im);
                    float32x4_t tr = vsubq_f32(
                        vmulq_f32(vr, wr), vmulq_f32(vi, wi));
                    float32x4_t ti = vaddq_f32(
                        vmulq_f32(vr, wi), vmulq_f32(vi, wr));
                    vst1q_f32(re + v, vsubq_f32(ur, tr));
                    vst1q_f32(im + v, vsubq_f32(ui, ti));
                    vst1q_f32(re + u, vaddq_f32(ur, tr));
                    vst1q_f32(im + u, vaddq_f32(ui, ti));
                }
            }
#endif
            for (; k < len / 2; ++k) {
                int u = base + k, v = u + len / 2;
                float tr = re[v] * cur_re - im[v] * cur_im;
                float ti = re[v] * cur_im + im[v] * cur_re;
                float next_re, next_im;
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] += tr;
                im[u] += ti;
                next_re = cur_re * step_re - cur_im * step_im;
                next_im = cur_re * step_im + cur_im * step_re;
                cur_re = next_re;
                cur_im = next_im;
            }
        }
    }
    if (inverse) {
        int i = 0;
#if DF_COMMON_HAVE_NEON
        float32x4_t vn = vdupq_n_f32((float)n);
        for (; i + 4 <= n; i += 4) {
            vst1q_f32(re + i, vdivq_f32(vld1q_f32(re + i), vn));
            vst1q_f32(im + i, vdivq_f32(vld1q_f32(im + i), vn));
        }
#endif
        for (; i < n; ++i) { re[i] /= n; im[i] /= n; }
    }
}

static inline void df_common_make_root_hann(float *window, int win_len) {
    for (int i = 0; i < win_len; ++i) {
        window[i] = sqrtf(0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)i / (float)win_len));
    }
}

static inline void df_common_analysis(float *analysis_buf, const float *window,
                                      float *scratch_re, float *scratch_im,
                                      const float *new_samples, int n_fft,
                                      int hop, float norm,
                                      float *out_re, float *out_im) {
    memmove(analysis_buf, analysis_buf + hop,
            (size_t)(n_fft - hop) * sizeof(float));
    memcpy(analysis_buf + n_fft - hop, new_samples,
           (size_t)hop * sizeof(float));
    memset(scratch_im, 0, (size_t)n_fft * sizeof(float));
    {
        int i = 0;
#if DF_COMMON_HAVE_NEON
        for (; i + 4 <= n_fft; i += 4) {
            vst1q_f32(scratch_re + i,
                      vmulq_f32(vld1q_f32(analysis_buf + i),
                                vld1q_f32(window + i)));
        }
#endif
        for (; i < n_fft; ++i) scratch_re[i] = analysis_buf[i] * window[i];
    }
    df_common_fft(scratch_re, scratch_im, n_fft, 0);
    {
        int i = 0, bins = n_fft / 2 + 1;
#if DF_COMMON_HAVE_NEON
        float32x4_t vnorm = vdupq_n_f32(norm);
        for (; i + 4 <= bins; i += 4) {
            vst1q_f32(out_re + i,
                      vmulq_f32(vld1q_f32(scratch_re + i), vnorm));
            vst1q_f32(out_im + i,
                      vmulq_f32(vld1q_f32(scratch_im + i), vnorm));
        }
#endif
        for (; i < bins; ++i) {
            out_re[i] = scratch_re[i] * norm;
            out_im[i] = scratch_im[i] * norm;
        }
    }
}

static inline void df_common_synthesis(float *synthesis_buf,
                                       const float *window,
                                       float *scratch_re, float *scratch_im,
                                       const float *spec_re,
                                       const float *spec_im,
                                       int n_fft, int hop, float inv_norm,
                                       float *output) {
    int bins = n_fft / 2 + 1;
    int k = 0;
#if DF_COMMON_HAVE_NEON
    {
        float32x4_t norm = vdupq_n_f32(inv_norm);
        for (; k + 4 <= bins; k += 4) {
            vst1q_f32(scratch_re + k,
                      vmulq_f32(vld1q_f32(spec_re + k), norm));
            vst1q_f32(scratch_im + k,
                      vmulq_f32(vld1q_f32(spec_im + k), norm));
        }
    }
#endif
    for (; k < bins; ++k) {
        scratch_re[k] = spec_re[k] * inv_norm;
        scratch_im[k] = spec_im[k] * inv_norm;
    }
    for (k = 1; k < n_fft / 2; ++k) {
        scratch_re[n_fft - k] = scratch_re[k];
        scratch_im[n_fft - k] = -scratch_im[k];
    }
    df_common_fft(scratch_re, scratch_im, n_fft, 1);
    {
        int i = 0;
#if DF_COMMON_HAVE_NEON
        for (; i + 4 <= n_fft; i += 4) {
            vst1q_f32(scratch_re + i,
                      vmulq_f32(vld1q_f32(scratch_re + i),
                                vld1q_f32(window + i)));
        }
        for (i = 0; i + 4 <= hop; i += 4) {
            vst1q_f32(output + i,
                      vaddq_f32(vld1q_f32(synthesis_buf + i),
                                vld1q_f32(scratch_re + i)));
        }
#else
        for (; i < n_fft; ++i) scratch_re[i] *= window[i];
#endif
#if DF_COMMON_HAVE_NEON
        for (; i < hop; ++i)
            output[i] = synthesis_buf[i] + scratch_re[i];
#else
        for (i = 0; i < hop; ++i)
            output[i] = synthesis_buf[i] + scratch_re[i];
#endif
    }
    memcpy(synthesis_buf, scratch_re + hop,
           (size_t)(n_fft - hop) * sizeof(float));
    memset(synthesis_buf + n_fft - hop, 0, (size_t)hop * sizeof(float));
}

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
