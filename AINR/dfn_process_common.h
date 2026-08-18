#ifndef AINR_DFN_PROCESS_COMMON_H
#define AINR_DFN_PROCESS_COMMON_H

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

#endif /* AINR_DFN_PROCESS_COMMON_H */
