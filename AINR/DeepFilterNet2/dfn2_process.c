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

#include "simd_kernel_nn.h"

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

/* The deep filter's taps before the stream start read this row. */
static const float df_zero_row[DFN2_DF_BINS];

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
                                      float *out_re, float *out_im) {
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
            out_re[k] = scratch_freq[k].r * norm;
            out_im[k] = scratch_freq[k].i * norm;
        }
    }
}

/* erb_work[b] = sum over bins k in [k0[b], k1[b]) ascending of
 * power[k] * erb_fwd[k * n_bands + b]; erb_work is zeroed on entry and
 * n_bands is a multiple of 4. Four adjacent bands (their ranges are about the
 * same length) advance together, each with its own register accumulator: the
 * bin-major walk updates the accumulators in memory and every update waits for
 * the previous one's store. Every power must be finite (a zero weight outside
 * a range then adds nothing). */
static inline void erb_forward_banded(const float *power, const float *erb_fwd,
                                      const uint16_t *k0, const uint16_t *k1,
                                      int n_bands, float *erb_work) {
    for (int b = 0; b < n_bands; b += 4) {
        const int a0 = k0[b], a1 = k0[b + 1], a2 = k0[b + 2], a3 = k0[b + 3];
        const int n0 = k1[b] - a0, n1 = k1[b + 1] - a1;
        const int n2 = k1[b + 2] - a2, n3 = k1[b + 3] - a3;
        int common = n0 < n1 ? n0 : n1;
        const float *w0 = erb_fwd + (size_t)a0 * n_bands + b;
        const float *w1 = erb_fwd + (size_t)a1 * n_bands + b + 1;
        const float *w2 = erb_fwd + (size_t)a2 * n_bands + b + 2;
        const float *w3 = erb_fwd + (size_t)a3 * n_bands + b + 3;
        float s0 = erb_work[b], s1 = erb_work[b + 1];
        float s2 = erb_work[b + 2], s3 = erb_work[b + 3];
        if (n2 < common) common = n2;
        if (n3 < common) common = n3;
        int j;
        for (j = 0; j < common; ++j) {
            s0 += power[a0 + j] * w0[(size_t)j * n_bands];
            s1 += power[a1 + j] * w1[(size_t)j * n_bands];
            s2 += power[a2 + j] * w2[(size_t)j * n_bands];
            s3 += power[a3 + j] * w3[(size_t)j * n_bands];
        }
        for (j = common; j < n0; ++j) s0 += power[a0 + j] * w0[(size_t)j * n_bands];
        for (j = common; j < n1; ++j) s1 += power[a1 + j] * w1[(size_t)j * n_bands];
        for (j = common; j < n2; ++j) s2 += power[a2 + j] * w2[(size_t)j * n_bands];
        for (j = common; j < n3; ++j) s3 += power[a3 + j] * w3[(size_t)j * n_bands];
        erb_work[b] = s0;
        erb_work[b + 1] = s1;
        erb_work[b + 2] = s2;
        erb_work[b + 3] = s3;
    }
}

/* erb_fwd: caller-loaded exported matrix, raw float32, bin-major
 * [n_bins][n_bands] -- the exact buffer the model trained with (see
 * export_erb_matrix.py --runtime-bins). The library never derives a
 * filterbank; the loader owns the file and can swap it at runtime. */
static inline void df_common_features(
    const float *spec_re, const float *spec_im,
    const float *erb_fwd, const uint16_t *fwd_lo, const uint16_t *fwd_hi,
    const uint16_t *band_k0, const uint16_t *band_k1,
    int n_bins, int n_bands, int df_bins,
    float analysis_scale, float log_floor,
    float erb_alpha, float erb_scale, float *erb_state,
    float spec_alpha, float spec_eps, float *spec_state,
    float *power, float *erb_work, float *feat_erb, float *feat_spec) {
    float scale2 = analysis_scale * analysis_scale;
    memset(erb_work, 0, (size_t)n_bands * sizeof(float));
    skn_power_scale_f32(power, spec_re, spec_im, (size_t)n_bins, scale2);
    /* Each band accumulates its bins in ascending k over its nonzero range
     * (see DFN2State). A non-finite power takes its whole row, since Inf/NaN
     * times 0 is NaN; that case walks bin by bin, every other frame runs
     * band by band with the accumulators in registers. Both add each band's
     * nonzero products in the same order. */
    if (skn_all_finite_f32(power, (size_t)n_bins)) {
        _Static_assert(DFN2_N_ERB % 4 == 0, "erb_forward_banded runs bands in fours");
        erb_forward_banded(power, erb_fwd, band_k0, band_k1, n_bands,
                           erb_work);
    } else {
        for (int k = 0; k < n_bins; ++k) {
            float p = power[k];
            const float *row = erb_fwd + (size_t)k * n_bands;
            int finite = isfinite(p);
            int hi = finite ? fwd_hi[k] : n_bands;
            for (int b = finite ? fwd_lo[k] : 0; b < hi; ++b)
                erb_work[b] += p * row[b];
        }
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
 * over the band's nonzero bins, or the whole row for a non-finite gain. */
static inline void df_common_expand_mask(const float *band_gain,
                                         const float *erb_inv,
                                         const uint16_t *inv_lo,
                                         const uint16_t *inv_hi,
                                         int n_bins, int n_bands,
                                         float *bin_gain) {
    memset(bin_gain, 0, (size_t)n_bins * sizeof(float));
    for (int b = 0; b < n_bands; ++b) {
        float gain = band_gain[b];
        const float *row = erb_inv + (size_t)b * n_bins;
        int finite = isfinite(gain);
        int hi = finite ? inv_hi[b] : n_bins;
        for (int k = finite ? inv_lo[b] : 0; k < hi; ++k)
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

static inline void df_common_synthesis(FftHandle* fft,
                                       float *synthesis_buf,
                                       const float *window,
                                       float *scratch_time,
                                       Complex *scratch_freq,
                                       const float *spec_re,
                                       const float *spec_im,
                                       int n_fft, int hop, float inv_norm,
                                       float *output) {
    int bins = n_fft / 2 + 1;
    for (int k = 0; k < bins; ++k) {
        scratch_freq[k].r = spec_re[k] * inv_norm;
        scratch_freq[k].i = spec_im[k] * inv_norm;
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

/* One frame of the deep filter on bins [0, DFN2_DF_BINS): each bin sums
 * DFN2_DF_ORDER complex taps over the source rows (one per tap, oldest
 * first), then mixes the filtered value with the unfiltered `mix` frame by
 * alpha. out_* must not alias a source row or the coefficients. */
static inline void df_common_deep_filter(
    const float *const *src_re, const float *const *src_im,
    const float *coefs, float alpha,
    const float *mix_re, const float *mix_im,
    float *out_re, float *out_im) {
    const float beta = 1.0f - alpha;
    skn_ctaps_mac_f32(out_re, out_im, DFN2_DF_BINS, DFN2_DF_ORDER,
                      src_re, src_im, coefs);
    sk_ema_f32(out_re, mix_re, alpha, beta, DFN2_DF_BINS);
    sk_ema_f32(out_im, mix_im, alpha, beta, DFN2_DF_BINS);
}

/* Where each of `lines` lines of a matrix is nonzero: line i holds
 * m[i * line_step + j * elem_step] for j < len, and its span is the first nonzero j
 * and one past the last (an all-zero line is the empty range [0, 0); without
 * a matrix every line is full). Rows of a [rows][cols] matrix: (rows, cols,
 * cols, 1); columns: (cols, rows, 1, cols). */
static void erb_spans(const float *m, int lines, int len, int line_step,
                      int elem_step, uint16_t *lo, uint16_t *hi)
{
    for (int i = 0; i < lines; ++i) {
        int first = 0, last = len;
        if (m) {
            const float *line = m + (size_t)i * line_step;
            while (first < last && line[(size_t)first * elem_step] == 0.0f)
                ++first;
            while (last > first && line[(size_t)(last - 1) * elem_step] == 0.0f)
                --last;
            if (first == last) first = last = 0;
        }
        lo[i] = (uint16_t)first;
        hi[i] = (uint16_t)last;
    }
}

void dfn2_set_erb_matrices(DFN2State* st,
                             const float* erb_fwd,
                             const float* erb_inv)
{
    if (!st) return;
    st->erb_fwd = erb_fwd;
    st->erb_inv = erb_inv;
    erb_spans(erb_fwd, DFN2_N_BINS, DFN2_N_ERB, DFN2_N_ERB, 1,
              st->erb_fwd_lo, st->erb_fwd_hi);
    erb_spans(erb_fwd, DFN2_N_ERB, DFN2_N_BINS, 1, DFN2_N_ERB,
              st->erb_fwd_k0, st->erb_fwd_k1);
    erb_spans(erb_inv, DFN2_N_ERB, DFN2_N_BINS, DFN2_N_BINS, 1,
              st->erb_inv_lo, st->erb_inv_hi);
}

void dfn2_state_init(DFN2State* st, FftHandle* fft)
{
    if (!st) return;
    memset(st, 0, sizeof(*st));
    st->fft = fft;
    /* ERB matrices arrive via dfn2_set_erb_matrices(): caller-loaded
     * erb_fwd.bin / erb_inv.bin from export_erb_matrix.py --runtime-bins.
     * Until then the matrices are NULL and their spans full. */
    dfn2_set_erb_matrices(st, NULL, NULL);
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
    df_common_analysis(st->fft, st->analysis_buf, st->window,
                       st->scratch_time, st->scratch_freq, frame,
                       DFN2_N_FFT, DFN2_HOP_LEN, normalization,
                       out_re, out_im);
}

void dfn2_compute_features(DFN2State* st,
                           const float* spec_re, const float* spec_im,
                           float* feat_erb, float* feat_spec)
{
    if (!st || !spec_re || !spec_im || !feat_erb || !feat_spec) return;
    df_common_features(
        spec_re, spec_im, st->erb_fwd, st->erb_fwd_lo, st->erb_fwd_hi,
        st->erb_fwd_k0, st->erb_fwd_k1, DFN2_N_BINS, DFN2_N_ERB, DFN2_DF_BINS,
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
    const float *src_re[DFN2_DF_ORDER];
    const float *src_im[DFN2_DF_ORDER];
    if (!st || !spec_re || !spec_im || !erb_mask || !coefs ||
        !out_re || !out_im || !isfinite(alpha)) return 0;
    df_common_expand_mask(erb_mask, st->erb_inv,
                          st->erb_inv_lo, st->erb_inv_hi,
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
    for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
        int source = (slot + tap - (DFN2_DF_ORDER - 1) +
                      DFN2_DF_RING) % DFN2_DF_RING;
        src_re[tap] = st->df_ring_re[source];
        src_im[tap] = st->df_ring_im[source];
    }
    df_common_deep_filter(src_re, src_im, &st->coef_ring[target][0][0][0],
                          alpha, st->df_ring_re[target],
                          st->df_ring_im[target], out_re, out_im);
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
    const float *src_re[DFN2_DF_ORDER];
    const float *src_im[DFN2_DF_ORDER];

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
                          st->erb_inv_lo, st->erb_inv_hi,
                          DFN2_N_BINS, DFN2_N_ERB,
                          st->scratch_bin_gain);
    skn_mul_f32(st->df_ring_re[head_slot], st->noisy_ring_re[head_slot],
                st->scratch_bin_gain, DFN2_DF_BINS);
    skn_mul_f32(st->df_ring_im[head_slot], st->noisy_ring_im[head_slot],
                st->scratch_bin_gain, DFN2_DF_BINS);
    skn_mul_f32(st->hi_delay_re[head_slot],
                st->noisy_ring_re[head_slot] + DFN2_DF_BINS,
                st->scratch_bin_gain + DFN2_DF_BINS,
                DFN2_N_BINS - DFN2_DF_BINS);
    skn_mul_f32(st->hi_delay_im[head_slot],
                st->noisy_ring_im[head_slot] + DFN2_DF_BINS,
                st->scratch_bin_gain + DFN2_DF_BINS,
                DFN2_N_BINS - DFN2_DF_BINS);
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

    /* The taps' source rows are per frame, not per bin: resolve them once.
     * A source frame before the stream start is a row of 0.0f, multiplied
     * through like any other. */
    for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
        long long source_frame = target_frame - DFN2_DF_HISTORY + tap;
        int slot = source_frame >= 0 ? (int)(source_frame % DFN2_DF_RING)
                                     : -1;
        src_re[tap] = slot >= 0 ? st->df_ring_re[slot] : df_zero_row;
        src_im[tap] = slot >= 0 ? st->df_ring_im[slot] : df_zero_row;
    }
    df_common_deep_filter(src_re, src_im,
                          &st->coef_ring[target_slot][0][0][0], alpha,
                          st->df_ring_re[target_slot],
                          st->df_ring_im[target_slot], out_re, out_im);
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
    df_common_synthesis(st->fft, st->synthesis_buf, st->window,
                        st->scratch_time, st->scratch_freq,
                        spec_re, spec_im, DFN2_N_FFT, DFN2_HOP_LEN,
                        normalization, out_frame);
}

const char* dfn2_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
