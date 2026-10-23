#include "dfn2_process.h"

#include <math.h>
#include <string.h>


/* Model-local DSP kernels (features / mask expansion / attenuation limit).
 * The STFT and ISTFT framing is audiolib's AcStft and
 * AcIstft, owned by the caller and bound with dfn2_state_init(). */

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "utility/simd_kernel_nn.h"

#if SK_HAVE_NEON
#include <arm_neon.h>
#endif
#define DF_COMMON_HAVE_NEON SK_HAVE_NEON

/* The deep filter's taps before the stream start read this row. */
static const float df_zero_row[DFN2_DF_BINS];

/* erb_fwd: caller-loaded matrix, raw float32, bin-major [n_bins][n_bands]
 * (export_erb_matrix.py --runtime-bins). The library never derives a filterbank. */
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
    /* Each band accumulates its bins in ascending k over its nonzero range. A
     * non-finite power takes its whole row (Inf/NaN times 0 is NaN), walked bin by
     * bin; other frames run band by band with accumulators in registers. */
    if (skn_all_finite_f32(power, (size_t)n_bins)) {
        _Static_assert(DFN2_N_ERB % 4 == 0, "skn_erb_forward_banded_f32 runs bands in fours");
        skn_erb_forward_banded_f32(power, erb_fwd, band_k0, band_k1, n_bands,
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

/* Scalar twin of a NEON kernel; it is the whole kernel in a SIMD=0 build. */
#define DFN2_TWIN static inline __attribute__((unused))

/* erb_inv: caller-loaded matrix, band-major [n_bands][n_bins]; the inner loop
 * runs over the band's nonzero bins, or the whole row for a non-finite gain. */
DFN2_TWIN void df_common_expand_mask_scalar(const float *band_gain,
                                            const float *erb_inv,
                                            const uint16_t *inv_lo,
                                            const uint16_t *inv_hi,
                                            int n_bins, int n_bands,
                                            float *bin_gain) {
    skn_erb_expand_mask_f32_scalar(band_gain, erb_inv, inv_lo, inv_hi, n_bins,
                                   n_bands, bin_gain);
}

#if DF_COMMON_HAVE_NEON
/* The same sums four bins per step. A band's last step may run up to three bins
 * past its span, where the weights are zero (a non-finite gain takes the whole
 * row, as in the twin). */
static inline void df_common_expand_mask_neon(const float *band_gain,
                                              const float *erb_inv,
                                              const uint16_t *inv_lo,
                                              const uint16_t *inv_hi,
                                              int n_bins, int n_bands,
                                              float *bin_gain) {
    skn_erb_expand_mask_f32_neon(band_gain, erb_inv, inv_lo, inv_hi, n_bins,
                                 n_bands, bin_gain);
}
#endif

static inline void df_common_expand_mask(const float *band_gain,
                                         const float *erb_inv,
                                         const uint16_t *inv_lo,
                                         const uint16_t *inv_hi,
                                         int n_bins, int n_bands,
                                         float *bin_gain) {
#if DF_COMMON_HAVE_NEON
    df_common_expand_mask_neon(band_gain, erb_inv, inv_lo, inv_hi, n_bins,
                               n_bands, bin_gain);
#else
    df_common_expand_mask_scalar(band_gain, erb_inv, inv_lo, inv_hi, n_bins,
                                 n_bands, bin_gain);
#endif
}

/* enh = noisy*lim + enh*mix, expressed as sk_ema_f32 (state=enh, x=noisy,
 * alpha=mix, beta=lim). */
static inline void df_common_atten_lim(const float *noisy_re,
                                       const float *noisy_im,
                                       float *enh_re, float *enh_im, int bins,
                                       float atten_lim_db) {
    float lim, mix;
    if (atten_lim_db == 0.0f) return;
    lim = powf(10.0f, -fabsf(atten_lim_db) / 20.0f);
    mix = 1.0f - lim;
    sk_ema_f32(enh_re, noisy_re, mix, lim, bins);
    sk_ema_f32(enh_im, noisy_im, mix, lim, bins);
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

/* Nonzero span of each of `lines` matrix lines: line i holds
 * m[i*line_step + j*elem_step] for j < len; the span is the first nonzero j to one
 * past the last (all-zero line: [0, 0); no matrix: full). Rows of [rows][cols]:
 * (rows, cols, cols, 1); columns: (cols, rows, 1, cols). */
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

void dfn2_framing_config(AcStftConfig* cfg, FftHandle* fft,
                          const float* window)
{
    if (!cfg) return;
    ac_stft_config_default(cfg, DFN2_N_FFT);
    cfg->spec_scale = 1.0f / sqrtf((float)DFN2_N_FFT);
    cfg->ispec_scale = sqrtf((float)DFN2_N_FFT);
    cfg->fft = fft;
    cfg->window = window;
}

void dfn2_state_init(DFN2State* st, AcStft* stft, AcIstft* istft)
{
    if (!st) return;
    memset(st, 0, sizeof(*st));
    st->stft = stft;
    st->istft = istft;
    ac_stft_reset(stft);
    ac_istft_reset(istft);
    /* ERB matrices arrive via dfn2_set_erb_matrices(); until then they are NULL
     * and their spans full. */
    dfn2_set_erb_matrices(st, NULL, NULL);
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
    if (!st || !st->stft || !frame || !out_re || !out_im) return;
    ac_stft_push_planar(st->stft, frame, out_re, out_im);
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

    /* A lookahead network cannot return frame 0's heads before input frame
     * MASK_LOOKAHEAD has arrived. This alignment is enforced because a misaligned
     * mask is audible but finite and passes NaN/smoke checks. */
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

    /* In a cascade the newest usable masked source is head_frame, so the output
     * target is one DF lookahead behind it. */
    target_frame = head_frame - DFN2_DF_LOOKAHEAD;
    if (target_frame < 0) return 0;
    target_slot = (int)(target_frame % DFN2_DF_RING);
    alpha = st->alpha_ring[target_slot];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    /* The taps' source rows are per frame, not per bin: resolve them once. A
     * source frame before the stream start is a row of 0.0f. */
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
    dfn2_apply_atten_lim(st->noisy_ring_re[target_slot],
                         st->noisy_ring_im[target_slot],
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = target_frame;
    return 1;
}

void dfn2_synthesis(DFN2State* st,
                    const float* spec_re, const float* spec_im,
                    float* out_frame)
{
    if (!st || !st->istft || !spec_re || !spec_im || !out_frame) return;
    ac_istft_push_planar(st->istft, spec_re, spec_im, out_frame);
}

const char* dfn2_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
