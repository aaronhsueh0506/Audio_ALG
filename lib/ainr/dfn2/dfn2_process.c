/* DeepFilterNet2 signal path outside the network (contract: dfn2_process.h): the grid, the ERB
 * and complex features, mask expansion, the deep filter and the attenuation limit, on spectra.
 * The framing is audiolib's, built by the application. */

#include "dfn2_process.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "audio_stft.h"            /* ac_stft_get_mem_size(): the n_fft rule is audiolib's */
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#if SK_HAVE_NEON
#include <arm_neon.h>
#endif
#define DF_COMMON_HAVE_NEON SK_HAVE_NEON

/* The variant symbol the other objects reference (dfn2_process.h). */
const unsigned char DFN2_VARIANT_SYMBOL = 0;

/* ERB and complex features of one spectrum. erb_fwd: caller-loaded matrix, raw float32,
 * bin-major [n_bins][n_bands]. */
#if defined(__GNUC__) && !defined(__clang__)
/* gcc on 32-bit ARM warns that the scalar tail after the NEON loop cannot reach iteration
 * 2^30 (an error under -Werror); the tail is correct for every df_bins. */
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
/* One frame of the deep filter on bins [0, df_bins): each bin sums `taps` complex taps over the
 * source rows (one per tap, oldest first), then mixes with the unfiltered `mix` frame by alpha.
 * out_* must not alias a source row or the coefficients. */
static inline void df_common_deep_filter(
    const float *const *src_re, const float *const *src_im, int taps,
    const float *coefs, float alpha,
    const float *mix_re, const float *mix_im,
    float *out_re, float *out_im, int df_bins) {
    const float beta = 1.0f - alpha;
    skn_ctaps_mac_f32(out_re, out_im, (size_t)df_bins, (size_t)taps,
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

/* ---- Grid ---- */

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

float dfn2_norm_alpha(int sample_rate, int hop, float tau_sec)
{
    /* train.py make_norm_alpha: three decimals, one more while the result reaches 1.0.
     * The loop is bounded: a hop of a few samples at a high rate never gets there. */
    double exact;
    double alpha = 1.0;
    if (sample_rate <= 0 || hop <= 0 || !(tau_sec > 0.0f)) return 0.0f;
    exact = exp(-((double)hop / (double)sample_rate) / (double)tau_sec);
    for (int digits = 3; alpha >= 1.0 && digits <= 17; ++digits)
        alpha = round_decimals(exact, digits);
    return (float)alpha;
}

/* An EMA coefficient of 0 or 1 is a tau (or a rate) the filter cannot run. */
static int norm_alpha_ok(float alpha)
{
    return alpha > 0.0f && alpha < 1.0f;
}

int dfn2_grid_make(DFN2Grid* grid, const DFN2GridSpec* spec)
{
    DFN2Grid g;
    AcStftConfig framing;
    int n_bins;
    int noisy_ring;
    int df_order;
    if (!grid || !spec) return DFN2_GRID_ERR_NULL;
    if (spec->sample_rate <= 0) return DFN2_GRID_ERR_SAMPLE_RATE;
    /* a size the framing cannot build has no memory size; the state's ERB spans are 16-bit */
    ac_stft_config_default(&framing, spec->n_fft);
    if (ac_stft_get_mem_size(&framing) == 0u) return DFN2_GRID_ERR_N_FFT;
    n_bins = spec->n_fft / 2 + 1;
    if (n_bins > UINT16_MAX) return DFN2_GRID_ERR_N_FFT;
    if (spec->n_erb <= 0 || spec->n_erb % 4 != 0 ||
        spec->n_erb > n_bins / 2)   /* 2 * n_erb <= n_bins */
        return DFN2_GRID_ERR_N_ERB;
#ifdef DFN2_MASK_ONLY
    if (spec->df_bins < 2 || spec->df_bins > n_bins) return DFN2_GRID_ERR_DF_BINS;
    if (spec->df_order < 0 || spec->df_order > DFN2_GRID_DF_ORDER_MAX)
        return DFN2_GRID_ERR_DF_ORDER;
    df_order = 0;   /* no deep filter to size */
#else
    if (spec->df_bins < 2 || spec->df_bins >= n_bins) return DFN2_GRID_ERR_DF_BINS;
    if (spec->df_order < 1 || spec->df_order > DFN2_GRID_DF_ORDER_MAX)
        return DFN2_GRID_ERR_DF_ORDER;
    df_order = spec->df_order;
#endif
    if (spec->mask_lookahead < 0 || spec->mask_lookahead > DFN2_GRID_MASK_LOOKAHEAD_MAX)
        return DFN2_GRID_ERR_MASK_LOOKAHEAD;
#ifdef DFN2_MASK_ONLY
    if (spec->df_lookahead != 0) return DFN2_GRID_ERR_DF_LOOKAHEAD;
#else
    if (spec->df_lookahead < 0 || spec->df_lookahead > df_order - 1)
        return DFN2_GRID_ERR_DF_LOOKAHEAD;
#endif
    if (spec->frames_per_call < 1 ||
        spec->frames_per_call > DFN2_MODEL_MAX_FRAMES_PER_CALL)
        return DFN2_GRID_ERR_FRAMES_PER_CALL;

    memset(&g, 0, sizeof(g));
    g.sample_rate = spec->sample_rate;
    g.n_fft = spec->n_fft;
    g.hop = spec->n_fft / 2;
    g.n_bins = n_bins;
    g.n_erb = spec->n_erb;
    g.df_bins = spec->df_bins;
    g.df_order = df_order;
    g.mask_lookahead = spec->mask_lookahead;
    g.df_lookahead = spec->df_lookahead;
    g.frames_per_call = spec->frames_per_call;
    g.input_frames = DFN2_MODEL_INPUT_FRAMES - 1 + spec->frames_per_call;
    noisy_ring = spec->mask_lookahead + spec->df_lookahead + 1;
#ifdef DFN2_MASK_ONLY
    g.df_history = 0;
    g.ring = noisy_ring;   /* only the noisy ring is kept */
#else
    g.df_history = df_order - spec->df_lookahead - 1;
    g.ring = df_order > noisy_ring ? df_order : noisy_ring;
#endif

    /* The alphas are derived, never given. */
    if (!isfinite(spec->erb_norm_tau_sec) || !(spec->erb_norm_tau_sec > 0.0f) ||
        !isfinite(spec->erb_norm_init_lo_db) || !isfinite(spec->erb_norm_init_hi_db) ||
        !isfinite(spec->erb_norm_scale_db) || !(spec->erb_norm_scale_db > 0.0f))
        return DFN2_GRID_ERR_ERB_NORM;
    g.erb_norm_tau_sec = spec->erb_norm_tau_sec;
    g.erb_norm_init_lo_db = spec->erb_norm_init_lo_db;
    g.erb_norm_init_hi_db = spec->erb_norm_init_hi_db;
    g.erb_norm_scale_db = spec->erb_norm_scale_db;
    g.erb_norm_alpha = dfn2_norm_alpha(spec->sample_rate, g.hop, spec->erb_norm_tau_sec);
    if (!norm_alpha_ok(g.erb_norm_alpha)) return DFN2_GRID_ERR_ERB_NORM;
    if (!isfinite(spec->spec_norm_tau_sec) || !(spec->spec_norm_tau_sec > 0.0f) ||
        !(spec->spec_norm_init_lo > 0.0f) || !isfinite(spec->spec_norm_init_lo) ||
        !(spec->spec_norm_init_hi > 0.0f) || !isfinite(spec->spec_norm_init_hi))
        return DFN2_GRID_ERR_SPEC_NORM;
    g.spec_norm_tau_sec = spec->spec_norm_tau_sec;
    g.spec_norm_init_lo = spec->spec_norm_init_lo;
    g.spec_norm_init_hi = spec->spec_norm_init_hi;
    g.spec_norm_alpha = dfn2_norm_alpha(spec->sample_rate, g.hop, spec->spec_norm_tau_sec);
    if (!norm_alpha_ok(g.spec_norm_alpha)) return DFN2_GRID_ERR_SPEC_NORM;

    /* correctly rounded float32 division and square root: the same values on every compiler */
    g.stft_scale = 1.0f / sqrtf((float)spec->n_fft);
    g.stft_unscale = sqrtf((float)spec->n_fft);
    *grid = g;
    return DFN2_GRID_OK;
}

int dfn2_grid_validate(const DFN2Grid* grid)
{
    DFN2GridSpec spec;
    DFN2Grid want;
    int status;
    if (!grid) return DFN2_GRID_ERR_NULL;
    spec.sample_rate = grid->sample_rate;
    spec.n_fft = grid->n_fft;
    spec.n_erb = grid->n_erb;
    spec.df_bins = grid->df_bins;
    spec.df_order = grid->df_order;
    spec.mask_lookahead = grid->mask_lookahead;
    spec.df_lookahead = grid->df_lookahead;
    spec.frames_per_call = grid->frames_per_call;
    spec.erb_norm_tau_sec = grid->erb_norm_tau_sec;
    spec.erb_norm_init_lo_db = grid->erb_norm_init_lo_db;
    spec.erb_norm_init_hi_db = grid->erb_norm_init_hi_db;
    spec.erb_norm_scale_db = grid->erb_norm_scale_db;
    spec.spec_norm_tau_sec = grid->spec_norm_tau_sec;
    spec.spec_norm_init_lo = grid->spec_norm_init_lo;
    spec.spec_norm_init_hi = grid->spec_norm_init_hi;
    status = dfn2_grid_make(&want, &spec);
    if (status != DFN2_GRID_OK) return status;
    if (grid->hop != want.hop || grid->n_bins != want.n_bins ||
        grid->df_order != want.df_order ||
        grid->df_history != want.df_history || grid->ring != want.ring ||
        grid->input_frames != want.input_frames ||
        grid->erb_norm_alpha != want.erb_norm_alpha ||
        grid->spec_norm_alpha != want.spec_norm_alpha ||
        grid->stft_scale != want.stft_scale ||
        grid->stft_unscale != want.stft_unscale)
        return DFN2_GRID_ERR_INCONSISTENT;
    return DFN2_GRID_OK;
}

const char* dfn2_grid_status_text(int status)
{
    switch (status) {
    case DFN2_GRID_OK:                  return "ok";
    case DFN2_GRID_ERR_NULL:            return "null pointer";
    case DFN2_GRID_ERR_SAMPLE_RATE:     return "sample rate must be positive";
    case DFN2_GRID_ERR_N_FFT:           return "n_fft is not a size the audiolib framing builds (or has too many bins for the 16-bit ERB spans)";
    case DFN2_GRID_ERR_N_ERB:           return "n_erb must be a positive multiple of 4 with at least two bins per band";
    case DFN2_GRID_ERR_DF_BINS:         return "df_bins must be in [2, n_fft/2 + 1) (up to n_fft/2 + 1 when mask-only)";
    case DFN2_GRID_ERR_DF_ORDER:        return "df_order must be in [1, 16] (any value in [0, 16] when mask-only)";
    case DFN2_GRID_ERR_MASK_LOOKAHEAD:  return "mask_lookahead must be in [0, 2]";
    case DFN2_GRID_ERR_DF_LOOKAHEAD:    return "df_lookahead must be in [0, df_order - 1] (0 when mask-only)";
    case DFN2_GRID_ERR_FRAMES_PER_CALL: return "frames_per_call must be in [1, 64]";
    case DFN2_GRID_ERR_ERB_NORM:        return "erb normaliser: tau must be finite and > 0 with an alpha in (0, 1), starts finite, scale finite and > 0";
    case DFN2_GRID_ERR_SPEC_NORM:       return "spec normaliser: tau must be finite and > 0 with an alpha in (0, 1), starts finite and > 0";
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

/* Reserve `count` elements of `elem` bytes at *cursor, 16-byte aligned, saturating on overflow;
 * base==NULL sizes only. */
static int state_carve(unsigned char* base, size_t* cursor, size_t count,
                       size_t elem, void** out)
{
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, ck_mul_size(count, elem));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

/* The block's one walk: dfn2_state_mem_size() runs it with base==NULL, dfn2_state_init()
 * with the block and the header, so size and carve cannot disagree. */
static int state_layout(DFN2State* st, unsigned char* base, const DFN2Grid* g,
                        size_t* total)
{
    const size_t n_bins = (size_t)g->n_bins;
    const size_t n_erb = (size_t)g->n_erb;
    const size_t ring_bins = (size_t)g->ring * n_bins;
    size_t cursor = 0;
    void* ptr = NULL;
#define STATE_CARVE(field, count, type)                                        \
    do {                                                                       \
        if (state_carve(base, &cursor, (count), sizeof(type), &ptr) != 0)     \
            return -1;                                                         \
        if (st) st->field = ptr;                                               \
    } while (0)
    STATE_CARVE(erb_norm_state, n_erb, float);
    STATE_CARVE(spec_norm_state, (size_t)g->df_bins, float);
#ifndef DFN2_MASK_ONLY
    STATE_CARVE(masked_re, ring_bins, float);
    STATE_CARVE(masked_im, ring_bins, float);
    STATE_CARVE(coef_ring, (size_t)g->ring * (size_t)g->df_bins *
                               (size_t)g->df_order * 2u, float);
    STATE_CARVE(alpha_ring, (size_t)g->ring, float);
    STATE_CARVE(df_zero_row, (size_t)g->df_bins, float);
    STATE_CARVE(src_re, (size_t)g->df_order, const float*);
    STATE_CARVE(src_im, (size_t)g->df_order, const float*);
#endif
    STATE_CARVE(noisy_ring_re, ring_bins, float);
    STATE_CARVE(noisy_ring_im, ring_bins, float);
    STATE_CARVE(scratch_power, n_bins, float);
    STATE_CARVE(scratch_erb_db, n_erb, float);
    STATE_CARVE(scratch_bin_gain, n_bins, float);
    STATE_CARVE(erb_fwd_lo, n_bins, uint16_t);
    STATE_CARVE(erb_fwd_hi, n_bins, uint16_t);
    STATE_CARVE(erb_fwd_k0, n_erb, uint16_t);
    STATE_CARVE(erb_fwd_k1, n_erb, uint16_t);
    STATE_CARVE(erb_inv_lo, n_erb, uint16_t);
    STATE_CARVE(erb_inv_hi, n_erb, uint16_t);
#undef STATE_CARVE
    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

size_t dfn2_state_mem_size(const DFN2Grid* grid)
{
    size_t total = 0;
    if (dfn2_grid_validate(grid) != DFN2_GRID_OK) return 0;
    if (state_layout(NULL, NULL, grid, &total) != 0) return 0;
    return total;
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

/* The linspace seed of one bin: a line from lo to hi held between the two ends, so a span wide enough to
 * cancel in float (lo + (hi - lo) rounding to 0) cannot drop the last bin below its configured end. */
static float norm_seed_linear(float lo, float hi, float position)
{
    float v = lo + position * (hi - lo);
    float vmin = lo < hi ? lo : hi, vmax = lo < hi ? hi : lo;
    return v < vmin ? vmin : (v > vmax ? vmax : v);
}

/* The state at its start: normalisers at their linspace initial values, ERB spans derived from
 * the matrices the state holds (none after init: spans full; a reset keeps the pointers). */
static void state_begin(DFN2State* st)
{
    const DFN2Grid* grid = &st->grid;
    dfn2_set_erb_matrices(st, st->erb_fwd, st->erb_inv);
    for (int b = 0; b < grid->n_erb; ++b) {
        float position = (float)b / (float)(grid->n_erb - 1);
        st->erb_norm_state[b] = grid->erb_norm_init_lo_db + position *
            (grid->erb_norm_init_hi_db - grid->erb_norm_init_lo_db);
    }
    for (int k = 0; k < grid->df_bins; ++k) {
        float position = (float)k / (float)(grid->df_bins - 1);
        st->spec_norm_state[k] = norm_seed_linear(grid->spec_norm_init_lo, grid->spec_norm_init_hi, position);
    }
}

int dfn2_state_init(DFN2State* st, void* block, size_t bytes, const DFN2Grid* grid)
{
    size_t need;
    size_t total = 0;
    if (!st) return -1;
    memset(st, 0, sizeof(*st));
    need = dfn2_state_mem_size(grid);
    if (need == 0 || !block || bytes < need || !MEM_IS_ALIGNED16(block)) return -1;
    memset(block, 0, need);
    st->grid = *grid;
    if (state_layout(st, (unsigned char*)block, grid, &total) != 0 || total != need) {
        memset(st, 0, sizeof(*st));
        return -1;
    }
    st->block = block;
    st->block_bytes = need;
    state_begin(st);
    return 0;
}

void dfn2_state_reset(DFN2State* st)
{
    if (!state_ready(st)) return;
    memset(st->block, 0, st->block_bytes);
    st->stream_frame_index = 0;
#ifndef DFN2_MASK_ONLY
    st->df_ring_idx = 0;
    st->df_ring_count = 0;
#endif
    state_begin(st);
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
        st->grid.erb_norm_alpha, st->grid.erb_norm_scale_db,
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

int dfn2_enhance_spectrum(DFN2State* st,
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
    if (!state_ready(st) || !spec_re || !spec_im || !erb_mask || !coefs ||
        !out_re || !out_im || !isfinite(alpha)) return 0;
    const int n_bins = st->grid.n_bins;
    const int df_bins = st->grid.df_bins;
    const int df_order = st->grid.df_order;
    const int ring = st->grid.ring;
    const size_t row = (size_t)n_bins;
    const size_t frame_coefs = (size_t)df_bins * (size_t)df_order * 2u;
    df_common_expand_mask(erb_mask, st->erb_inv,
                          st->erb_inv_lo, st->erb_inv_hi,
                          n_bins, st->grid.n_erb,
                          st->scratch_bin_gain);
    slot = st->df_ring_idx;
    {
        float *masked_re = st->masked_re + (size_t)slot * row;
        float *masked_im = st->masked_im + (size_t)slot * row;
        for (int k = 0; k < n_bins; ++k) {
            masked_re[k] = spec_re[k] * st->scratch_bin_gain[k];
            masked_im[k] = spec_im[k] * st->scratch_bin_gain[k];
        }
    }
    memcpy(st->coef_ring + (size_t)slot * frame_coefs, coefs,
           frame_coefs * sizeof(float));
    memcpy(st->noisy_ring_re + (size_t)slot * row, spec_re, row * sizeof(float));
    memcpy(st->noisy_ring_im + (size_t)slot * row, spec_im, row * sizeof(float));
    st->alpha_ring[slot] = alpha;
    st->df_ring_idx = (slot + 1) % ring;
    if (st->df_ring_count < ring) ++st->df_ring_count;
    if (st->df_ring_count <= st->grid.df_lookahead) return 0;

    target = (slot - st->grid.df_lookahead + ring) % ring;
    alpha = st->alpha_ring[target];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    for (int tap = 0; tap < df_order; ++tap) {
        int source = (slot + tap - (df_order - 1) + ring) % ring;
        st->src_re[tap] = st->masked_re + (size_t)source * row;
        st->src_im[tap] = st->masked_im + (size_t)source * row;
    }
    {
        const float *target_re = st->masked_re + (size_t)target * row;
        const float *target_im = st->masked_im + (size_t)target * row;
        df_common_deep_filter(st->src_re, st->src_im, df_order,
                              st->coef_ring + (size_t)target * frame_coefs,
                              alpha, target_re, target_im, out_re, out_im, df_bins);
        for (int k = df_bins; k < n_bins; ++k) {
            out_re[k] = target_re[k];
            out_im[k] = target_im[k];
        }
    }
    return 1;
#endif
}

int dfn2_enhance_spectrum_stream(DFN2State* st,
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
    const size_t row = (size_t)n_bins;
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
    memcpy(st->noisy_ring_re + (size_t)current_slot * row, current_spec_re,
           row * sizeof(float));
    memcpy(st->noisy_ring_im + (size_t)current_slot * row, current_spec_im,
           row * sizeof(float));
    if (current < mask_lookahead) return 0;

    head_frame = current - mask_lookahead;
    head_slot = (int)(head_frame % ring);
    df_mask_spectrum(st, erb_mask, st->noisy_ring_re + (size_t)head_slot * row,
                     st->noisy_ring_im + (size_t)head_slot * row, out_re, out_im);
    dfn2_apply_atten_lim(n_bins, st->noisy_ring_re + (size_t)head_slot * row,
                         st->noisy_ring_im + (size_t)head_slot * row,
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = head_frame;
    return 1;
#else
    long long target_frame;
    int target_slot;
    const int df_bins = st->grid.df_bins;
    const int df_order = st->grid.df_order;
    const int df_lookahead = st->grid.df_lookahead;
    const int df_history = st->grid.df_history;
    const size_t frame_coefs = (size_t)df_bins * (size_t)df_order * 2u;

    current = st->stream_frame_index;
    if (current < mask_lookahead) {
        if (heads_valid) return -1;
    } else if (!heads_valid || !erb_mask || !coefs || !isfinite(alpha) ||
               !isfinite(atten_lim_db)) {
        return -1;
    }
    ++st->stream_frame_index;
    current_slot = (int)(current % ring);
    memcpy(st->noisy_ring_re + (size_t)current_slot * row, current_spec_re,
           row * sizeof(float));
    memcpy(st->noisy_ring_im + (size_t)current_slot * row, current_spec_im,
           row * sizeof(float));

    /* The heads of frame 0 cannot arrive before input frame mask_lookahead; a misaligned mask is
     * audible but finite, so the alignment is enforced. */
    if (current < mask_lookahead) {
        return 0;
    }

    head_frame = current - mask_lookahead;
    head_slot = (int)(head_frame % ring);
    df_common_expand_mask(erb_mask, st->erb_inv,
                          st->erb_inv_lo, st->erb_inv_hi,
                          n_bins, st->grid.n_erb,
                          st->scratch_bin_gain);
    skn_mul_f32(st->masked_re + (size_t)head_slot * row,
                st->noisy_ring_re + (size_t)head_slot * row,
                st->scratch_bin_gain, (size_t)n_bins);
    skn_mul_f32(st->masked_im + (size_t)head_slot * row,
                st->noisy_ring_im + (size_t)head_slot * row,
                st->scratch_bin_gain, (size_t)n_bins);
    memcpy(st->coef_ring + (size_t)head_slot * frame_coefs, coefs,
           frame_coefs * sizeof(float));
    st->alpha_ring[head_slot] = alpha;

    /* cascade: the newest masked source is head_frame, the output target is df_lookahead behind */
    target_frame = head_frame - df_lookahead;
    if (target_frame < 0) return 0;
    target_slot = (int)(target_frame % ring);
    alpha = st->alpha_ring[target_slot];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    /* the taps' source rows are per frame: resolve once; before the stream start a row of 0.0f */
    for (int tap = 0; tap < df_order; ++tap) {
        long long source_frame = target_frame - df_history + tap;
        int slot = source_frame >= 0 ? (int)(source_frame % ring) : -1;
        st->src_re[tap] = slot >= 0 ? st->masked_re + (size_t)slot * row
                                    : st->df_zero_row;
        st->src_im[tap] = slot >= 0 ? st->masked_im + (size_t)slot * row
                                    : st->df_zero_row;
    }
    {
        const float *target_re = st->masked_re + (size_t)target_slot * row;
        const float *target_im = st->masked_im + (size_t)target_slot * row;
        df_common_deep_filter(st->src_re, st->src_im, df_order,
                              st->coef_ring + (size_t)target_slot * frame_coefs,
                              alpha, target_re, target_im, out_re, out_im, df_bins);
        for (int k = df_bins; k < n_bins; ++k) {
            out_re[k] = target_re[k];
            out_im[k] = target_im[k];
        }
    }
    dfn2_apply_atten_lim(n_bins, st->noisy_ring_re + (size_t)target_slot * row,
                         st->noisy_ring_im + (size_t)target_slot * row,
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = target_frame;
    return 1;
#endif
}

const char* dfn2_simd_backend(void)
{
    return SK_NEON_BACKEND_NAME;
}
