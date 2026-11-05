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

/* The variant and capacities this object was built for, read by dfn2_model_io.c and
 * dfn2_prepost.c (dfn2_process.h): objects of another layout fail to link. */
const unsigned char DFN2_VARIANT_SYMBOL = 0;

/* The deep filter's taps before the stream start read this row. */
#ifndef DFN2_MASK_ONLY
static const float df_zero_row[DFN2_MAX_DF_BINS];
#endif

/* erb_fwd: caller-loaded matrix, raw float32, bin-major [n_bins][n_bands]
 * (export_erb_matrix.py --runtime-bins). The library never derives a filterbank. */
#if defined(__GNUC__) && !defined(__clang__)
/* gcc on 32-bit ARM warns that the scalar tail after the NEON loop cannot reach iteration 2^30 (an error under -Werror);
 * the tail is correct for every df_bins. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif
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
        /* n_bands is a multiple of 4 (dfn2_grid_make): the kernel runs bands in fours. */
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
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

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

#ifndef DFN2_MASK_ONLY
/* One frame of the deep filter on bins [0, df_bins): each bin sums
 * DFN2_DF_ORDER complex taps over the source rows (one per tap, oldest
 * first), then mixes the filtered value with the unfiltered `mix` frame by
 * alpha. out_* must not alias a source row or the coefficients. */
static inline void df_common_deep_filter(
    const float *const *src_re, const float *const *src_im,
    const float *coefs, float alpha,
    const float *mix_re, const float *mix_im,
    float *out_re, float *out_im, int df_bins) {
    const float beta = 1.0f - alpha;
    skn_ctaps_mac_f32(out_re, out_im, (size_t)df_bins, DFN2_DF_ORDER,
                      src_re, src_im, coefs);
    sk_ema_f32(out_re, mix_re, alpha, beta, df_bins);
    sk_ema_f32(out_im, mix_im, alpha, beta, df_bins);
}
#endif

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

/* The normaliser starting points this object was compiled with, for the pre/post pool hash
 * (dfn2_prepost.c reads them from here so the hash describes what dfn2_state_init() uses). */
const float dfn2_norm_starts[DFN2_NORM_STARTS] = {
    DFN2_ERB_NORM_INIT_LO_DB, DFN2_ERB_NORM_INIT_HI_DB, DFN2_ERB_NORM_SCALE_DB,
    DFN2_SPEC_NORM_INIT_LO, DFN2_SPEC_NORM_INIT_HI,
};

/* ---- Grid ---- */

static int is_power_of_two(int v)
{
    return v > 0 && (v & (v - 1)) == 0;
}

/* Python's round(x, digits) for 0 < x < 1: the decimal with `digits` places nearest the
 * exact binary value of x, a tie going to the even last digit, as the nearest double of
 * that decimal. x * scale is first rounded to a double, so the sign of x * scale - (t +
 * 1/2) is taken from fma(), which rounds once. */
static double round_decimals(double x, int digits)
{
    double scale = 1.0;
    double t;
    double diff;
    for (int i = 0; i < digits; ++i) scale *= 10.0;   /* exact up to 1e22 */
    t = floor(x * scale);
    if (fma(x, scale, -t) < 0.0) t -= 1.0;
    else if (fma(x, scale, -(t + 1.0)) >= 0.0) t += 1.0;
    diff = fma(x, scale, -(t + 0.5));
    if (diff > 0.0 || (diff == 0.0 && fmod(t, 2.0) != 0.0)) t += 1.0;
    return t / scale;
}

float dfn2_norm_alpha(int sample_rate, int hop, double tau_sec)
{
    /* train.py make_norm_alpha: three decimals, one more while the result reaches 1.0.
     * The loop is bounded: a hop of a few samples at a high rate never gets there. */
    double exact;
    double alpha = 1.0;
    if (sample_rate <= 0 || hop <= 0 || !(tau_sec > 0.0)) return 0.0f;
    exact = exp(-((double)hop / (double)sample_rate) / tau_sec);
    for (int digits = 3; alpha >= 1.0 && digits <= 17; ++digits)
        alpha = round_decimals(exact, digits);
    return (float)alpha;
}

int dfn2_grid_make(DFN2Grid* grid, int sample_rate, int n_fft, int n_erb,
                   int df_bins, int mask_lookahead, int df_lookahead)
{
    DFN2Grid g;
    int n_bins;
    int noisy_ring;
    if (!grid) return DFN2_GRID_ERR_NULL;
    if (sample_rate <= 0) return DFN2_GRID_ERR_SAMPLE_RATE;
    if (n_fft < 16 || n_fft > 8192 || !is_power_of_two(n_fft))
        return DFN2_GRID_ERR_N_FFT;
    n_bins = n_fft / 2 + 1;
    if (n_erb <= 0 || n_erb % 4 != 0 || n_erb > n_bins / 2)   /* 2 * n_erb <= n_bins */
        return DFN2_GRID_ERR_N_ERB;
#ifdef DFN2_MASK_ONLY
    if (df_bins < 2 || df_bins > n_bins) return DFN2_GRID_ERR_DF_BINS;
#else
    if (df_bins < 2 || df_bins >= n_bins) return DFN2_GRID_ERR_DF_BINS;
#endif
    if (mask_lookahead < 0 || mask_lookahead > DFN2_MAX_MASK_LOOKAHEAD)
        return DFN2_GRID_ERR_MASK_LOOKAHEAD;
#ifdef DFN2_MASK_ONLY
    if (df_lookahead != 0) return DFN2_GRID_ERR_DF_LOOKAHEAD;
#else
    if (df_lookahead < 0 || df_lookahead > DFN2_MAX_DF_LOOKAHEAD)
        return DFN2_GRID_ERR_DF_LOOKAHEAD;
#endif

    memset(&g, 0, sizeof(g));
    g.sample_rate = sample_rate;
    g.n_fft = n_fft;
    g.hop = n_fft / 2;
    g.n_bins = n_bins;
    g.n_erb = n_erb;
    g.df_bins = df_bins;
    g.mask_lookahead = mask_lookahead;
    g.df_lookahead = df_lookahead;
    g.df_history = DFN2_DF_ORDER - df_lookahead - 1;
    noisy_ring = mask_lookahead + df_lookahead + 1;
#ifdef DFN2_MASK_ONLY
    g.ring = noisy_ring;   /* only the noisy ring is kept */
#else
    g.ring = DFN2_DF_ORDER > noisy_ring ? DFN2_DF_ORDER : noisy_ring;
#endif
    if (n_fft > DFN2_MAX_N_FFT || n_erb > DFN2_MAX_N_ERB ||
        df_bins > DFN2_MAX_DF_BINS || g.ring > DFN2_MAX_DF_RING)
        return DFN2_GRID_ERR_CAPACITY;
    g.erb_norm_alpha = dfn2_norm_alpha(sample_rate, g.hop, 1.0);
    g.spec_norm_alpha = g.erb_norm_alpha;
    /* An EMA coefficient of 0 or 1 means a rate the framing cannot run (a hop that is far
     * below or above one sample at this rate); dfn2_grid_validate() refuses it too. */
    if (!(g.erb_norm_alpha > 0.0f && g.erb_norm_alpha < 1.0f))
        return DFN2_GRID_ERR_SAMPLE_RATE;
    /* float32 division and square root are correctly rounded, so these are the float32
     * values of 1/sqrt(n_fft) and sqrt(n_fft) on every compiler (exact powers of two at
     * 16, 64, 256, 1024, 4096). */
    g.stft_scale = 1.0f / sqrtf((float)n_fft);
    g.stft_unscale = sqrtf((float)n_fft);
    *grid = g;
    return DFN2_GRID_OK;
}

int dfn2_grid_validate(const DFN2Grid* grid)
{
    DFN2Grid want;
    int status;
    if (!grid) return DFN2_GRID_ERR_NULL;
    status = dfn2_grid_make(&want, grid->sample_rate, grid->n_fft, grid->n_erb,
                            grid->df_bins, grid->mask_lookahead,
                            grid->df_lookahead);
    if (status != DFN2_GRID_OK) return status;
    if (grid->hop != want.hop || grid->n_bins != want.n_bins ||
        grid->df_history != want.df_history || grid->ring != want.ring ||
        grid->stft_scale != want.stft_scale ||
        grid->stft_unscale != want.stft_unscale ||
        /* The alphas are the model's (a tau other than 1.0 sets them): finite, in (0, 1). */
        !(grid->erb_norm_alpha > 0.0f && grid->erb_norm_alpha < 1.0f) ||
        !(grid->spec_norm_alpha > 0.0f && grid->spec_norm_alpha < 1.0f))
        return DFN2_GRID_ERR_INCONSISTENT;
    return DFN2_GRID_OK;
}

const char* dfn2_grid_status_text(int status)
{
    switch (status) {
    case DFN2_GRID_OK:                  return "ok";
    case DFN2_GRID_ERR_NULL:            return "null pointer";
    case DFN2_GRID_ERR_SAMPLE_RATE:     return "sample rate must be positive and give a norm alpha in (0, 1)";
    case DFN2_GRID_ERR_N_FFT:           return "n_fft must be a power of two in [16, 8192]";
    case DFN2_GRID_ERR_N_ERB:           return "n_erb must be a positive multiple of 4 with at least two bins per band";
    case DFN2_GRID_ERR_DF_BINS:         return "df_bins must be in [2, n_fft/2 + 1) (up to n_fft/2 + 1 when mask-only)";
    case DFN2_GRID_ERR_MASK_LOOKAHEAD:  return "mask_lookahead must be in [0, 2]";
    case DFN2_GRID_ERR_DF_LOOKAHEAD:    return "df_lookahead must be in [0, 4] (0 when mask-only)";
    case DFN2_GRID_ERR_CAPACITY:        return "grid exceeds the library capacity (DFN2_MAX_N_FFT / N_ERB / DF_BINS / DF_RING)";
    case DFN2_GRID_ERR_INCONSISTENT:    return "derived grid fields do not match the given ones";
    default:                            return "unknown status";
    }
}

/* ---- State ---- */

/* A state is usable once dfn2_state_init() recorded a grid. */
static int state_ready(const DFN2State* st)
{
    return st && st->grid.n_bins > 0;
}

void dfn2_set_erb_matrices(DFN2State* st,
                             const float* erb_fwd,
                             const float* erb_inv)
{
    int n_bins;
    int n_erb;
    if (!state_ready(st)) return;
    n_bins = st->grid.n_bins;
    n_erb = st->grid.n_erb;
    st->erb_fwd = erb_fwd;
    st->erb_inv = erb_inv;
    erb_spans(erb_fwd, n_bins, n_erb, n_erb, 1,
              st->erb_fwd_lo, st->erb_fwd_hi);
    erb_spans(erb_fwd, n_erb, n_bins, 1, n_erb,
              st->erb_fwd_k0, st->erb_fwd_k1);
    erb_spans(erb_inv, n_erb, n_bins, n_bins, 1,
              st->erb_inv_lo, st->erb_inv_hi);
}

void dfn2_framing_config(AcStftConfig* cfg, const DFN2Grid* grid,
                         FftHandle* fft, const float* window)
{
    if (!cfg) return;
    if (dfn2_grid_validate(grid) != DFN2_GRID_OK) {
        memset(cfg, 0, sizeof(*cfg));   /* fft_size 0: the framing objects refuse it */
        return;
    }
    ac_stft_config_default(cfg, grid->n_fft);
    cfg->spec_scale = grid->stft_scale;
    cfg->ispec_scale = grid->stft_unscale;
    cfg->fft = fft;
    cfg->window = window;
}

int dfn2_state_init(DFN2State* st, const DFN2Grid* grid, AcStft* stft,
                    AcIstft* istft)
{
    if (!st) return -1;
    memset(st, 0, sizeof(*st));
    if (dfn2_grid_validate(grid) != DFN2_GRID_OK) return -1;
    st->grid = *grid;
    st->stft = stft;
    st->istft = istft;
    ac_stft_reset(stft);
    ac_istft_reset(istft);
    /* ERB matrices arrive via dfn2_set_erb_matrices(); until then they are NULL
     * and their spans full. */
    dfn2_set_erb_matrices(st, NULL, NULL);
    for (int b = 0; b < grid->n_erb; ++b) {
        float position = (float)b / (float)(grid->n_erb - 1);
        st->erb_norm_state[b] = DFN2_ERB_NORM_INIT_LO_DB + position *
            (DFN2_ERB_NORM_INIT_HI_DB - DFN2_ERB_NORM_INIT_LO_DB);
    }
    for (int k = 0; k < grid->df_bins; ++k) {
        float position = (float)k / (float)(grid->df_bins - 1);
        st->spec_norm_state[k] = DFN2_SPEC_NORM_INIT_LO + position *
            (DFN2_SPEC_NORM_INIT_HI - DFN2_SPEC_NORM_INIT_LO);
    }
    return 0;
}

void dfn2_analysis(DFN2State* st, const float* frame,
                   float* out_re, float* out_im)
{
    if (!state_ready(st) || !st->stft || !frame || !out_re || !out_im) return;
    ac_stft_push_planar(st->stft, frame, out_re, out_im);
}

void dfn2_compute_features(DFN2State* st,
                           const float* spec_re, const float* spec_im,
                           float* feat_erb, float* feat_spec)
{
    if (!state_ready(st) || !spec_re || !spec_im || !feat_erb || !feat_spec)
        return;
    df_common_features(
        spec_re, spec_im, st->erb_fwd, st->erb_fwd_lo, st->erb_fwd_hi,
        st->erb_fwd_k0, st->erb_fwd_k1, st->grid.n_bins, st->grid.n_erb,
        st->grid.df_bins,
        st->grid.stft_scale, DFN2_ERB_LOG_FLOOR,
        st->grid.erb_norm_alpha, DFN2_ERB_NORM_SCALE_DB,
        st->erb_norm_state,
        st->grid.spec_norm_alpha, DFN2_SPEC_NORM_EPS,
        st->spec_norm_state,
        st->scratch_power, st->scratch_erb_db, feat_erb, feat_spec);
}

void dfn2_apply_atten_lim(int n_bins,
                          const float* noisy_re, const float* noisy_im,
                          float* enh_re, float* enh_im,
                          float atten_lim_db)
{
    if (n_bins <= 0 || !noisy_re || !noisy_im || !enh_re || !enh_im ||
        !isfinite(atten_lim_db) || atten_lim_db == 0.0f) return;
    df_common_atten_lim(noisy_re, noisy_im, enh_re, enh_im,
                        n_bins, atten_lim_db);
}

#ifdef DFN2_MASK_ONLY
/* Stage 1: out = in * (mask expanded over the full band). */
static void df_mask_spectrum(DFN2State *st, const float *erb_mask,
                             const float *in_re, const float *in_im,
                             float *out_re, float *out_im) {
    const int n_bins = st->grid.n_bins;
    df_common_expand_mask(erb_mask, st->erb_inv, st->erb_inv_lo,
                          st->erb_inv_hi, n_bins, st->grid.n_erb,
                          st->scratch_bin_gain);
    skn_mul_f32(out_re, in_re, st->scratch_bin_gain, (size_t)n_bins);
    skn_mul_f32(out_im, in_im, st->scratch_bin_gain, (size_t)n_bins);
}
#endif

int dfn2_compose(DFN2State* st,
                 const float* spec_re, const float* spec_im,
                 const float* erb_mask, const float* coefs, float alpha,
                 float* out_re, float* out_im)
{
#ifdef DFN2_MASK_ONLY
    (void)coefs;
    (void)alpha;
    if (!state_ready(st) || !spec_re || !spec_im || !erb_mask || !out_re ||
        !out_im)
        return 0;
    df_mask_spectrum(st, erb_mask, spec_re, spec_im, out_re, out_im);
    return 1;
#else
    int slot;
    int target;
    const float *src_re[DFN2_DF_ORDER];
    const float *src_im[DFN2_DF_ORDER];
    if (!state_ready(st) || !spec_re || !spec_im || !erb_mask || !coefs ||
        !out_re || !out_im || !isfinite(alpha)) return 0;
    const int n_bins = st->grid.n_bins;
    const int df_bins = st->grid.df_bins;
    const int ring = st->grid.ring;
    df_common_expand_mask(erb_mask, st->erb_inv,
                          st->erb_inv_lo, st->erb_inv_hi,
                          n_bins, st->grid.n_erb,
                          st->scratch_bin_gain);
    slot = st->df_ring_idx;
    for (int k = 0; k < n_bins; ++k) {
        st->masked_re[slot][k] = spec_re[k] * st->scratch_bin_gain[k];
        st->masked_im[slot][k] = spec_im[k] * st->scratch_bin_gain[k];
    }
    memcpy(st->coef_ring[slot], coefs,
           (size_t)df_bins * DFN2_DF_ORDER * 2 * sizeof(float));
    memcpy(st->noisy_ring_re[slot], spec_re, (size_t)n_bins * sizeof(float));
    memcpy(st->noisy_ring_im[slot], spec_im, (size_t)n_bins * sizeof(float));
    st->alpha_ring[slot] = alpha;
    st->df_ring_idx = (slot + 1) % ring;
    if (st->df_ring_count < ring) ++st->df_ring_count;
    if (st->df_ring_count <= st->grid.df_lookahead) return 0;

    target = (slot - st->grid.df_lookahead + ring) % ring;
    alpha = st->alpha_ring[target];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
        int source = (slot + tap - (DFN2_DF_ORDER - 1) + ring) % ring;
        src_re[tap] = st->masked_re[source];
        src_im[tap] = st->masked_im[source];
    }
    df_common_deep_filter(src_re, src_im, &st->coef_ring[target][0][0][0],
                          alpha, st->masked_re[target],
                          st->masked_im[target], out_re, out_im, df_bins);
    for (int k = df_bins; k < n_bins; ++k) {
        out_re[k] = st->masked_re[target][k];
        out_im[k] = st->masked_im[target][k];
    }
    return 1;
#endif
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
    int current_slot;
    int head_slot;

    if (!state_ready(st) || !current_spec_re || !current_spec_im || !out_re ||
        !out_im)
        return -1;
    const int n_bins = st->grid.n_bins;
    const int ring = st->grid.ring;
    const int mask_lookahead = st->grid.mask_lookahead;
#ifdef DFN2_MASK_ONLY
    (void)coefs;
    (void)alpha;
    current = st->stream_frame_index;
    if (current < mask_lookahead) {
        if (heads_valid) return -1;
    } else if (!heads_valid || !erb_mask || !isfinite(atten_lim_db)) {
        return -1;
    }
    ++st->stream_frame_index;
    current_slot = (int)(current % ring);
    memcpy(st->noisy_ring_re[current_slot], current_spec_re,
           (size_t)n_bins * sizeof(float));
    memcpy(st->noisy_ring_im[current_slot], current_spec_im,
           (size_t)n_bins * sizeof(float));
    if (current < mask_lookahead) return 0;

    head_frame = current - mask_lookahead;
    head_slot = (int)(head_frame % ring);
    df_mask_spectrum(st, erb_mask, st->noisy_ring_re[head_slot],
                     st->noisy_ring_im[head_slot], out_re, out_im);
    dfn2_apply_atten_lim(n_bins, st->noisy_ring_re[head_slot],
                         st->noisy_ring_im[head_slot],
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = head_frame;
    return 1;
#else
    long long target_frame;
    int target_slot;
    const int df_bins = st->grid.df_bins;
    const int df_lookahead = st->grid.df_lookahead;
    const int df_history = st->grid.df_history;
    const float *src_re[DFN2_DF_ORDER];
    const float *src_im[DFN2_DF_ORDER];

    current = st->stream_frame_index;
    if (current < mask_lookahead) {
        if (heads_valid) return -1;
    } else if (!heads_valid || !erb_mask || !coefs || !isfinite(alpha) ||
               !isfinite(atten_lim_db)) {
        return -1;
    }
    ++st->stream_frame_index;
    current_slot = (int)(current % ring);
    memcpy(st->noisy_ring_re[current_slot], current_spec_re,
           (size_t)n_bins * sizeof(float));
    memcpy(st->noisy_ring_im[current_slot], current_spec_im,
           (size_t)n_bins * sizeof(float));

    /* A lookahead network cannot return frame 0's heads before input frame
     * mask_lookahead has arrived. This alignment is enforced because a misaligned
     * mask is audible but finite and passes NaN/smoke checks. */
    if (current < mask_lookahead) {
        return 0;
    }

    head_frame = current - mask_lookahead;
    head_slot = (int)(head_frame % ring);
    df_common_expand_mask(erb_mask, st->erb_inv,
                          st->erb_inv_lo, st->erb_inv_hi,
                          n_bins, st->grid.n_erb,
                          st->scratch_bin_gain);
    skn_mul_f32(st->masked_re[head_slot], st->noisy_ring_re[head_slot],
                st->scratch_bin_gain, (size_t)n_bins);
    skn_mul_f32(st->masked_im[head_slot], st->noisy_ring_im[head_slot],
                st->scratch_bin_gain, (size_t)n_bins);
    memcpy(st->coef_ring[head_slot], coefs,
           (size_t)df_bins * DFN2_DF_ORDER * 2 * sizeof(float));
    st->alpha_ring[head_slot] = alpha;

    /* In a cascade the newest usable masked source is head_frame, so the output
     * target is one DF lookahead behind it. */
    target_frame = head_frame - df_lookahead;
    if (target_frame < 0) return 0;
    target_slot = (int)(target_frame % ring);
    alpha = st->alpha_ring[target_slot];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    /* The taps' source rows are per frame, not per bin: resolve them once. A
     * source frame before the stream start is a row of 0.0f. */
    for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
        long long source_frame = target_frame - df_history + tap;
        int slot = source_frame >= 0 ? (int)(source_frame % ring) : -1;
        src_re[tap] = slot >= 0 ? st->masked_re[slot] : df_zero_row;
        src_im[tap] = slot >= 0 ? st->masked_im[slot] : df_zero_row;
    }
    df_common_deep_filter(src_re, src_im,
                          &st->coef_ring[target_slot][0][0][0], alpha,
                          st->masked_re[target_slot],
                          st->masked_im[target_slot], out_re, out_im, df_bins);
    for (int k = df_bins; k < n_bins; ++k) {
        out_re[k] = st->masked_re[target_slot][k];
        out_im[k] = st->masked_im[target_slot][k];
    }
    dfn2_apply_atten_lim(n_bins, st->noisy_ring_re[target_slot],
                         st->noisy_ring_im[target_slot],
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = target_frame;
    return 1;
#endif
}

void dfn2_synthesis(DFN2State* st,
                    const float* spec_re, const float* spec_im,
                    float* out_frame)
{
    if (!state_ready(st) || !st->istft || !spec_re || !spec_im || !out_frame)
        return;
    ac_istft_push_planar(st->istft, spec_re, spec_im, out_frame);
}

const char* dfn2_simd_backend(void)
{
    return SK_NEON_BACKEND_NAME;
}
