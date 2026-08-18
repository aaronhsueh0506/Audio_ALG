#include "dfn2_process.h"

#include <math.h>
#include <string.h>


/* Model-local DSP kernels (FFT / root-Hann / STFT / WOLA /
 * features / mask expansion / attenuation limit / post-filter).
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

/* erb_fwd: caller-loaded exported matrix, raw float32, bin-major
 * [n_bins][n_bands] -- the exact buffer the model trained with (see
 * export_erb_matrix.py --runtime-bins). The library never derives a
 * filterbank; the loader owns the file and can swap it at runtime. */
static inline void df_common_features(
    const float *spec_re, const float *spec_im,
    const float *erb_fwd, int n_bins, int n_bands, int df_bins,
    float analysis_scale, float log_floor,
    float erb_alpha, float erb_scale, float *erb_state,
    float spec_alpha, float spec_eps, float *spec_state,
    float *power, float *erb_work, float *feat_erb, float *feat_spec) {
    float scale2 = analysis_scale * analysis_scale;
    memset(erb_work, 0, (size_t)n_bands * sizeof(float));
    for (int k = 0; k < n_bins; ++k) {
        float p = (spec_re[k] * spec_re[k] +
                   spec_im[k] * spec_im[k]) * scale2;
        const float *row = erb_fwd + (size_t)k * n_bands;
        power[k] = p;
        for (int b = 0; b < n_bands; ++b)
            erb_work[b] += p * row[b];
    }
    for (int b = 0; b < n_bands; ++b) {
        float db = 10.0f * log10f(erb_work[b] + log_floor);
        float mean = erb_alpha * erb_state[b] + (1.0f - erb_alpha) * db;
        erb_state[b] = mean;
        feat_erb[b] = (db - mean) / erb_scale;
    }
    {
        int k = 0;
#if DF_COMMON_HAVE_NEON
        float32x4_t va = vdupq_n_f32(spec_alpha);
        float32x4_t vb = vdupq_n_f32(1.0f - spec_alpha);
        float32x4_t veps = vdupq_n_f32(spec_eps);
        float32x4_t vscale = vdupq_n_f32(analysis_scale);
        for (; k + 4 <= df_bins; k += 4) {
            float32x4_t magnitude = vsqrtq_f32(vld1q_f32(power + k));
            float32x4_t state = vaddq_f32(
                vmulq_f32(va, vld1q_f32(spec_state + k)),
                vmulq_f32(vb, magnitude));
            float32x4_t denom = vsqrtq_f32(vaddq_f32(state, veps));
            vst1q_f32(spec_state + k, state);
            vst1q_f32(feat_spec + k,
                      vdivq_f32(vmulq_f32(vld1q_f32(spec_re + k), vscale),
                                denom));
            vst1q_f32(feat_spec + df_bins + k,
                      vdivq_f32(vmulq_f32(vld1q_f32(spec_im + k), vscale),
                                denom));
        }
#endif
        for (; k < df_bins; ++k) {
            float state = spec_alpha * spec_state[k] +
                (1.0f - spec_alpha) * sqrtf(power[k]);
            float denom = sqrtf(state + spec_eps);
            spec_state[k] = state;
            feat_spec[k] = spec_re[k] * analysis_scale / denom;
            feat_spec[df_bins + k] = spec_im[k] * analysis_scale / denom;
        }
    }
}

/* erb_inv: caller-loaded exported matrix, band-major [n_bands][n_bins]
 * (the model's mask-expansion buffer); the inner loop runs contiguously
 * over bins and auto-vectorizes. */
static inline void df_common_expand_mask(const float *band_gain,
                                         const float *erb_inv,
                                         int n_bins, int n_bands,
                                         float *bin_gain) {
    memset(bin_gain, 0, (size_t)n_bins * sizeof(float));
    for (int b = 0; b < n_bands; ++b) {
        float gain = band_gain[b];
        const float *row = erb_inv + (size_t)b * n_bins;
        for (int k = 0; k < n_bins; ++k)
            bin_gain[k] += row[k] * gain;
    }
}

static inline void df_common_atten_lim(const float *noisy_re,
                                       const float *noisy_im,
                                       float *enh_re, float *enh_im, int bins,
                                       float atten_lim_db) {
    float lim, mix;
    int k = 0;
    if (atten_lim_db == 0.0f) return;
    lim = powf(10.0f, -fabsf(atten_lim_db) / 20.0f);
    mix = 1.0f - lim;
#if DF_COMMON_HAVE_NEON
    {
        float32x4_t vl = vdupq_n_f32(lim), vm = vdupq_n_f32(mix);
        for (; k + 4 <= bins; k += 4) {
            vst1q_f32(enh_re + k, vaddq_f32(
                vmulq_f32(vld1q_f32(noisy_re + k), vl),
                vmulq_f32(vld1q_f32(enh_re + k), vm)));
            vst1q_f32(enh_im + k, vaddq_f32(
                vmulq_f32(vld1q_f32(noisy_im + k), vl),
                vmulq_f32(vld1q_f32(enh_im + k), vm)));
        }
    }
#endif
    for (; k < bins; ++k) {
        enh_re[k] = noisy_re[k] * lim + enh_re[k] * mix;
        enh_im[k] = noisy_im[k] * lim + enh_im[k] * mix;
    }
}

static inline void df_common_post_filter(const float *spec_re,
                                         const float *spec_im,
                                         float *enh_re, float *enh_im,
                                         int bins, float beta) {
    const float eps = 1e-12f;
    if (!(beta > 0.0f)) return;
    for (int k = 0; k < bins; ++k) {
        float noisy_mag = sqrtf(spec_re[k] * spec_re[k] +
                                spec_im[k] * spec_im[k]);
        float enh_mag = sqrtf(enh_re[k] * enh_re[k] +
                              enh_im[k] * enh_im[k]);
        float mask = enh_mag / (noisy_mag + eps);
        float mask_sin, ratio, pf;
        if (mask < eps) mask = eps;
        if (mask > 1.0f) mask = 1.0f;
        mask_sin = mask * sinf((float)M_PI * mask * 0.5f);
        if (mask_sin < eps) mask_sin = eps;
        ratio = mask / mask_sin;
        pf = (1.0f + beta) / (1.0f + beta * ratio * ratio);
        enh_re[k] *= pf;
        enh_im[k] *= pf;
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

void dfn2_set_erb_matrices(DFN2State* st,
                             const float* erb_fwd,
                             const float* erb_inv)
{
    if (!st) return;
    st->erb_fwd = erb_fwd;
    st->erb_inv = erb_inv;
}

void dfn2_state_init(DFN2State* st)
{
    if (!st) return;
    memset(st, 0, sizeof(*st));
    /* ERB matrices arrive via dfn2_set_erb_matrices(): caller-loaded
     * erb_fwd.bin / erb_inv.bin from export_erb_matrix.py --runtime-bins. */
    df_common_make_root_hann(st->window, DFN2_WIN_LEN);
    for (int b = 0; b < DFN2_N_ERB; ++b) {
        float position = (float)b / (float)(DFN2_N_ERB - 1);
        st->erb_norm_state[b] = DFN2_ERB_NORM_INIT_LO_DB + position *
            (DFN2_ERB_NORM_INIT_HI_DB - DFN2_ERB_NORM_INIT_LO_DB);
    }
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        float position = (float)k / (float)(DFN2_DF_BINS - 1);
        st->spec_norm_state[k] = DFN2_SPEC_NORM_INIT_LO + position *
            (DFN2_SPEC_NORM_INIT_HI - DFN2_SPEC_NORM_INIT_LO);
    }
}

void dfn2_analysis(DFN2State* st, const float* frame,
                   float* out_re, float* out_im)
{
    const float normalization = 1.0f / sqrtf((float)DFN2_N_FFT);
    if (!st || !frame || !out_re || !out_im) return;
    df_common_analysis(st->analysis_buf, st->window,
                       st->scratch_re, st->scratch_im, frame,
                       DFN2_N_FFT, DFN2_HOP_LEN, normalization,
                       out_re, out_im);
}

void dfn2_compute_features(DFN2State* st,
                           const float* spec_re, const float* spec_im,
                           float* feat_erb, float* feat_spec)
{
    if (!st || !spec_re || !spec_im || !feat_erb || !feat_spec) return;
    df_common_features(
        spec_re, spec_im, st->erb_fwd,
        DFN2_N_BINS, DFN2_N_ERB, DFN2_DF_BINS,
        DFN2_ANALYSIS_SCALE, DFN2_ERB_LOG_FLOOR,
        DFN2_ERB_NORM_ALPHA, DFN2_ERB_NORM_SCALE_DB,
        st->erb_norm_state,
        DFN2_SPEC_NORM_ALPHA, DFN2_SPEC_NORM_EPS,
        st->spec_norm_state,
        st->scratch_power, st->scratch_erb_db, feat_erb, feat_spec);
}

void dfn2_apply_atten_lim(const float* noisy_re, const float* noisy_im,
                          float* enh_re, float* enh_im,
                          float atten_lim_db)
{
    if (!noisy_re || !noisy_im || !enh_re || !enh_im ||
        !isfinite(atten_lim_db) || atten_lim_db == 0.0f) return;
    df_common_atten_lim(noisy_re, noisy_im, enh_re, enh_im,
                        DFN2_N_BINS, atten_lim_db);
}

int dfn2_compose(DFN2State* st,
                 const float* spec_re, const float* spec_im,
                 const float* erb_mask, const float* coefs, float alpha,
                 float* out_re, float* out_im)
{
    int slot;
    int target;
    if (!st || !spec_re || !spec_im || !erb_mask || !coefs ||
        !out_re || !out_im || !isfinite(alpha)) return 0;
    df_common_expand_mask(erb_mask, st->erb_inv,
                          DFN2_N_BINS, DFN2_N_ERB,
                          st->scratch_bin_gain);
    slot = st->df_ring_idx;
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        st->df_ring_re[slot][k] = spec_re[k] * st->scratch_bin_gain[k];
        st->df_ring_im[slot][k] = spec_im[k] * st->scratch_bin_gain[k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        st->hi_delay_re[slot][high] =
            spec_re[k] * st->scratch_bin_gain[k];
        st->hi_delay_im[slot][high] =
            spec_im[k] * st->scratch_bin_gain[k];
    }
    memcpy(st->coef_ring[slot], coefs,
           sizeof(st->coef_ring[slot]));
    memcpy(st->noisy_ring_re[slot], spec_re,
           sizeof(st->noisy_ring_re[slot]));
    memcpy(st->noisy_ring_im[slot], spec_im,
           sizeof(st->noisy_ring_im[slot]));
    st->alpha_ring[slot] = alpha;
    st->df_ring_idx = (slot + 1) % DFN2_DF_RING;
    if (st->df_ring_count < DFN2_DF_RING) ++st->df_ring_count;
    if (st->df_ring_count <= DFN2_DF_LOOKAHEAD) return 0;

    target = (slot - DFN2_DF_LOOKAHEAD + DFN2_DF_RING) % DFN2_DF_RING;
    alpha = st->alpha_ring[target];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        float filtered_re = 0.0f;
        float filtered_im = 0.0f;
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
            int source = (slot + tap - (DFN2_DF_ORDER - 1) +
                          DFN2_DF_RING) % DFN2_DF_RING;
            float xr = st->df_ring_re[source][k];
            float xi = st->df_ring_im[source][k];
            float cr = st->coef_ring[target][k][tap][0];
            float ci = st->coef_ring[target][k][tap][1];
            filtered_re += xr * cr - xi * ci;
            filtered_im += xi * cr + xr * ci;
        }
        out_re[k] = alpha * filtered_re +
                    (1.0f - alpha) * st->df_ring_re[target][k];
        out_im[k] = alpha * filtered_im +
                    (1.0f - alpha) * st->df_ring_im[target][k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        out_re[k] = st->hi_delay_re[target][high];
        out_im[k] = st->hi_delay_im[target][high];
    }
#if DFN2_MASK_PF
    dfn2_post_filter(st->noisy_ring_re[target], st->noisy_ring_im[target],
                     out_re, out_im, DFN2_PF_BETA);
#endif
    return 1;
}

int dfn2_compose_stream(DFN2State* st,
                        const float* current_spec_re,
                        const float* current_spec_im,
                        int heads_valid,
                        const float* erb_mask,
                        const float* coefs,
                        float alpha,
                        float atten_lim_db,
                        float* out_re,
                        float* out_im,
                        long long* output_frame_index)
{
    long long current;
    long long head_frame;
    long long target_frame;
    int current_slot;
    int head_slot;
    int target_slot;

    if (!st || !current_spec_re || !current_spec_im || !out_re || !out_im)
        return -1;
    current = st->stream_frame_index;
    if (current < DFN2_MASK_LOOKAHEAD) {
        if (heads_valid) return -1;
    } else if (!heads_valid || !erb_mask || !coefs || !isfinite(alpha) ||
               !isfinite(atten_lim_db)) {
        return -1;
    }
    ++st->stream_frame_index;
    current_slot = (int)(current % DFN2_DF_RING);
    memcpy(st->noisy_ring_re[current_slot], current_spec_re,
           sizeof(st->noisy_ring_re[current_slot]));
    memcpy(st->noisy_ring_im[current_slot], current_spec_im,
           sizeof(st->noisy_ring_im[current_slot]));

    /* A lookahead network cannot return frame 0's heads until input frame
     * MASK_LOOKAHEAD has arrived.  Enforce this alignment here: silently
     * treating a returned frame-(n-L) mask as frame n is audible but finite,
     * so ordinary NaN/output-smoke tests cannot catch the mistake. */
    if (current < DFN2_MASK_LOOKAHEAD) {
        return 0;
    }

    head_frame = current - DFN2_MASK_LOOKAHEAD;
    head_slot = (int)(head_frame % DFN2_DF_RING);
    df_common_expand_mask(erb_mask, st->erb_inv,
                          DFN2_N_BINS, DFN2_N_ERB,
                          st->scratch_bin_gain);
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        st->df_ring_re[head_slot][k] =
            st->noisy_ring_re[head_slot][k] * st->scratch_bin_gain[k];
        st->df_ring_im[head_slot][k] =
            st->noisy_ring_im[head_slot][k] * st->scratch_bin_gain[k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        st->hi_delay_re[head_slot][high] =
            st->noisy_ring_re[head_slot][k] * st->scratch_bin_gain[k];
        st->hi_delay_im[head_slot][high] =
            st->noisy_ring_im[head_slot][k] * st->scratch_bin_gain[k];
    }
    memcpy(st->coef_ring[head_slot], coefs,
           sizeof(st->coef_ring[head_slot]));
    st->alpha_ring[head_slot] = alpha;

    /* In a cascade the newest usable masked source is head_frame, hence the
     * output target is one DF lookahead behind that head. */
    target_frame = head_frame - DFN2_DF_LOOKAHEAD;
    if (target_frame < 0) return 0;
    target_slot = (int)(target_frame % DFN2_DF_RING);
    alpha = st->alpha_ring[target_slot];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        float filtered_re = 0.0f;
        float filtered_im = 0.0f;
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
            long long source_frame =
                target_frame - DFN2_DF_HISTORY + tap;
            float xr = 0.0f;
            float xi = 0.0f;
            if (source_frame >= 0) {
                int source_slot = (int)(source_frame % DFN2_DF_RING);
                xr = st->df_ring_re[source_slot][k];
                xi = st->df_ring_im[source_slot][k];
            }
            {
                float cr = st->coef_ring[target_slot][k][tap][0];
                float ci = st->coef_ring[target_slot][k][tap][1];
                filtered_re += xr * cr - xi * ci;
                filtered_im += xi * cr + xr * ci;
            }
        }
        out_re[k] = alpha * filtered_re +
                    (1.0f - alpha) * st->df_ring_re[target_slot][k];
        out_im[k] = alpha * filtered_im +
                    (1.0f - alpha) * st->df_ring_im[target_slot][k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        out_re[k] = st->hi_delay_re[target_slot][high];
        out_im[k] = st->hi_delay_im[target_slot][high];
    }
#if DFN2_MASK_PF
    dfn2_post_filter(st->noisy_ring_re[target_slot],
                     st->noisy_ring_im[target_slot],
                     out_re, out_im, DFN2_PF_BETA);
#endif
    dfn2_apply_atten_lim(st->noisy_ring_re[target_slot],
                         st->noisy_ring_im[target_slot],
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = target_frame;
    return 1;
}

void dfn2_post_filter(const float* spec_re, const float* spec_im,
                      float* enh_re, float* enh_im, float beta)
{
    if (!spec_re || !spec_im || !enh_re || !enh_im) return;
    df_common_post_filter(
        spec_re, spec_im, enh_re, enh_im, DFN2_N_BINS, beta);
}

void dfn2_synthesis(DFN2State* st,
                    const float* spec_re, const float* spec_im,
                    float* out_frame)
{
    const float normalization = sqrtf((float)DFN2_N_FFT);
    if (!st || !spec_re || !spec_im || !out_frame) return;
    df_common_synthesis(st->synthesis_buf, st->window,
                        st->scratch_re, st->scratch_im,
                        spec_re, spec_im, DFN2_N_FFT, DFN2_HOP_LEN,
                        normalization, out_frame);
}

const char* dfn2_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
