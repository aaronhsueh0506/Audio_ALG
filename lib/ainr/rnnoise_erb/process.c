/* RNNoise-ERB feature/gain kernels and state (contract: process.h). Requires
 * -ffp-contract=off. No heap, no globals, no stdio. */

#include "process.h"

#include <math.h>
#include <string.h>

#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"
#include "utility/simd_kernels.h"

#if SK_HAVE_NEON
#include <arm_neon.h>
#endif

#define LOG_FLOOR 1e-10f

/* ---- Grid ---- */

/* The sizes audiolib's AcStft accepts are the only n_fft range. */
static int fft_size_valid(int n_fft)
{
    AcStftConfig sc;
    ac_stft_config_default(&sc, n_fft);
    return ac_stft_get_mem_size(&sc) != 0u;
}

/* Python's round(x, digits) for 0 < x < 1: the decimal nearest the exact binary value of x,
 * a tie to the even last digit, as the nearest double. x * scale rounds as a double, so the
 * tests against t and t + 1/2 use fma(), which rounds once. */
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

float rnnoise_norm_alpha(int sample_rate, int hop, double tau_sec)
{
    /* train.py make_norm_alpha: three decimals, one more while it reaches 1.0 (at most 17). */
    double exact;
    double alpha = 1.0;
    if (sample_rate <= 0 || hop <= 0 || !(tau_sec > 0.0) || !isfinite(tau_sec)) return 0.0f;
    exact = exp(-((double)hop / (double)sample_rate) / tau_sec);
    for (int digits = 3; alpha >= 1.0 && digits <= 17; ++digits)
        alpha = round_decimals(exact, digits);
    return (float)alpha;
}

int rnnoise_grid_make(RNNoiseGrid *grid, const RNNoiseGridSpec *spec)
{
    RNNoiseGrid g;
    int n_bins;
    if (!grid || !spec) return RNNOISE_GRID_ERR_NULL;
    if (spec->sample_rate <= 0) return RNNOISE_GRID_ERR_SAMPLE_RATE;
    if (!fft_size_valid(spec->n_fft)) return RNNOISE_GRID_ERR_N_FFT;
    n_bins = spec->n_fft / 2 + 1;
    if (spec->min_bins_per_band < 1) return RNNOISE_GRID_ERR_MIN_BINS;
    if (spec->n_bands < 2 ||
        (long long)(spec->n_bands - 1) * spec->min_bins_per_band > (long long)n_bins)
        return RNNOISE_GRID_ERR_N_BANDS;
    if (spec->lookahead_frames < 0 || spec->lookahead_frames > 2)
        return RNNOISE_GRID_ERR_LOOKAHEAD;
    if (spec->spec_max_hz <= 0 || spec->spec_max_hz > spec->sample_rate / 2)
        return RNNOISE_GRID_ERR_SPEC_MAX_HZ;
    if (!(spec->erb_norm_tau_sec > 0.0) || !isfinite(spec->erb_norm_tau_sec) ||
        !isfinite(spec->erb_norm_init_lo_db) || !isfinite(spec->erb_norm_init_hi_db) ||
        !(spec->erb_norm_scale_db > 0.0f) || !isfinite(spec->erb_norm_scale_db))
        return RNNOISE_GRID_ERR_ERB_NORM;
    if (!(spec->spec_norm_tau_sec > 0.0) || !isfinite(spec->spec_norm_tau_sec) ||
        !(spec->spec_norm_init_lo > 0.0f) || !isfinite(spec->spec_norm_init_lo) ||
        !(spec->spec_norm_init_hi > 0.0f) || !isfinite(spec->spec_norm_init_hi) ||
        !(spec->spec_norm_eps > 0.0f) || !isfinite(spec->spec_norm_eps))
        return RNNOISE_GRID_ERR_SPEC_NORM;

    memset(&g, 0, sizeof(g));
    g.sample_rate = spec->sample_rate;
    g.n_fft = spec->n_fft;
    g.hop = spec->n_fft / 2;
    g.n_bins = n_bins;
    g.n_bands = spec->n_bands;
    g.min_bins_per_band = spec->min_bins_per_band;
    g.lookahead_frames = spec->lookahead_frames;
    g.conv_delay = 2 - spec->lookahead_frames;
    g.spec_max_hz = spec->spec_max_hz;
    g.spec_bins = (int)((long long)spec->spec_max_hz * spec->n_fft / spec->sample_rate) + 1;
    g.erb_norm_tau_sec = spec->erb_norm_tau_sec;
    g.spec_norm_tau_sec = spec->spec_norm_tau_sec;
    g.erb_norm_alpha = rnnoise_norm_alpha(spec->sample_rate, g.hop, spec->erb_norm_tau_sec);
    g.erb_norm_init_lo_db = spec->erb_norm_init_lo_db;
    g.erb_norm_init_hi_db = spec->erb_norm_init_hi_db;
    g.erb_norm_scale_db = spec->erb_norm_scale_db;
    g.spec_norm_alpha = rnnoise_norm_alpha(spec->sample_rate, g.hop, spec->spec_norm_tau_sec);
    g.spec_norm_init_lo = spec->spec_norm_init_lo;
    g.spec_norm_init_hi = spec->spec_norm_init_hi;
    g.spec_norm_eps = spec->spec_norm_eps;
    /* EMA coefficient 0 or 1: the hop is far longer (or absurdly shorter) than tau. */
    if (!(g.erb_norm_alpha > 0.0f && g.erb_norm_alpha < 1.0f))
        return RNNOISE_GRID_ERR_ERB_NORM;
    if (!(g.spec_norm_alpha > 0.0f && g.spec_norm_alpha < 1.0f))
        return RNNOISE_GRID_ERR_SPEC_NORM;
    /* float32 division and square root are correctly rounded, so these are the float32
     * values of 1/sqrt(n_fft) and sqrt(n_fft) on every compiler. */
    g.stft_scale = 1.0f / sqrtf((float)spec->n_fft);
    g.stft_unscale = sqrtf((float)spec->n_fft);
    *grid = g;
    return RNNOISE_GRID_OK;
}

int rnnoise_grid_validate(const RNNoiseGrid *grid)
{
    RNNoiseGridSpec spec;
    RNNoiseGrid want;
    int status;
    if (!grid) return RNNOISE_GRID_ERR_NULL;
    spec.sample_rate = grid->sample_rate;
    spec.n_fft = grid->n_fft;
    spec.n_bands = grid->n_bands;
    spec.min_bins_per_band = grid->min_bins_per_band;
    spec.lookahead_frames = grid->lookahead_frames;
    spec.spec_max_hz = grid->spec_max_hz;
    spec.erb_norm_tau_sec = grid->erb_norm_tau_sec;
    spec.erb_norm_init_lo_db = grid->erb_norm_init_lo_db;
    spec.erb_norm_init_hi_db = grid->erb_norm_init_hi_db;
    spec.erb_norm_scale_db = grid->erb_norm_scale_db;
    spec.spec_norm_tau_sec = grid->spec_norm_tau_sec;
    spec.spec_norm_init_lo = grid->spec_norm_init_lo;
    spec.spec_norm_init_hi = grid->spec_norm_init_hi;
    spec.spec_norm_eps = grid->spec_norm_eps;
    status = rnnoise_grid_make(&want, &spec);
    if (status != RNNOISE_GRID_OK) return status;
    if (grid->hop != want.hop || grid->n_bins != want.n_bins ||
        grid->conv_delay != want.conv_delay || grid->spec_bins != want.spec_bins ||
        grid->erb_norm_alpha != want.erb_norm_alpha ||
        grid->spec_norm_alpha != want.spec_norm_alpha ||
        grid->stft_scale != want.stft_scale ||
        grid->stft_unscale != want.stft_unscale)
        return RNNOISE_GRID_ERR_INCONSISTENT;
    return RNNOISE_GRID_OK;
}

const char *rnnoise_grid_status_text(int status)
{
    switch (status) {
    case RNNOISE_GRID_OK:               return "ok";
    case RNNOISE_GRID_ERR_NULL:         return "null argument";
    case RNNOISE_GRID_ERR_SAMPLE_RATE:  return "invalid sample_rate";
    case RNNOISE_GRID_ERR_N_FFT:        return "n_fft refused by audiolib";
    case RNNOISE_GRID_ERR_N_BANDS:      return "invalid n_bands";
    case RNNOISE_GRID_ERR_MIN_BINS:     return "invalid min_bins_per_band";
    case RNNOISE_GRID_ERR_LOOKAHEAD:    return "lookahead_frames outside 0..2";
    case RNNOISE_GRID_ERR_SPEC_MAX_HZ:  return "spec_max_hz outside (0, sample_rate/2]";
    case RNNOISE_GRID_ERR_ERB_NORM:     return "invalid ERB normaliser setting";
    case RNNOISE_GRID_ERR_SPEC_NORM:    return "invalid spectrum normaliser setting";
    case RNNOISE_GRID_ERR_INCONSISTENT: return "derived grid fields inconsistent";
    default:                            return "unknown status";
    }
}

/* ---- Pool arithmetic ---- */

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(RNNOISE_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* Reserve `count` elements of `elem` bytes at *cursor, aligned; base==NULL sizes only. An
 * overflow saturates the cursor to SIZE_MAX (audiolib's helpers), caught by MEM_SIZE_INVALID. */
static int st_carve(unsigned char *base, size_t *cursor, size_t count, size_t elem,
                    void **out)
{
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, ck_mul_size(count, elem));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

/* The block's one walk, shared by rnnoise_get_mem_size() (base==NULL) and rnnoise_init():
 * header, ERB span arrays, then the mutable arrays that rnnoise_reset() clears. */
static int state_layout(RNNoiseState *st, unsigned char *base, const RNNoiseGrid *g,
                        size_t *total)
{
    const size_t n_bands = (size_t)g->n_bands;
    const size_t frames = (size_t)RNNOISE_MODEL_INPUT_FRAMES;
    size_t cursor = 0;
    size_t mutable_start;
    void *ptr = NULL;

#define STATE_CARVE(member, count, type)                                        \
    do {                                                                        \
        if (st_carve(base, &cursor, (count), sizeof(type), &ptr) != 0) return -1; \
        if (base && st) st->member = (type *)ptr;                               \
    } while (0)

    if (st_carve(base, &cursor, 1u, sizeof(RNNoiseState), &ptr) != 0) return -1;
    STATE_CARVE(fwd_k0, n_bands, uint16_t);
    STATE_CARVE(fwd_k1, n_bands, uint16_t);
    STATE_CARVE(inv_lo, n_bands, uint16_t);
    STATE_CARVE(inv_hi, n_bands, uint16_t);
    mutable_start = ck_align16_size(cursor);
    STATE_CARVE(erb_feat_buf, ck_mul_size(frames, n_bands), float);
    STATE_CARVE(spec_feat_buf, ck_mul_size(ck_mul_size(frames, 2u), (size_t)g->spec_bins), float);
    STATE_CARVE(erb_norm_state, n_bands, float);
    STATE_CARVE(spec_norm_state, (size_t)g->spec_bins, float);
    STATE_CARVE(scratch_power, (size_t)g->n_bins, float);
    STATE_CARVE(scratch_erb_db, n_bands, float);
#undef STATE_CARVE

    *total = ck_align16_size(cursor);
    if (MEM_SIZE_INVALID(*total)) return -1;
    if (base && st) {
        st->mutable_base = base + mutable_start;
        st->mutable_bytes = *total - mutable_start;
    }
    return 0;
}

/* ---- Build identity ---- */

static uint32_t fnv1a_u32(uint32_t h, uint32_t v)
{
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

static uint32_t fnv1a_f32(uint32_t h, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    return fnv1a_u32(h, bits);
}

static uint32_t fnv1a_f64(uint32_t h, double v)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    h = fnv1a_u32(h, (uint32_t)(bits & 0xffffffffu));
    return fnv1a_u32(h, (uint32_t)(bits >> 32));
}

static uint32_t fnv1a_str(uint32_t h, const char *s)
{
    while (*s) {
        h ^= (uint32_t)(unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

/* Everything a pool depends on: the whole grid, the layout, carve and feature versions and the
 * graph's input window, so a pool from another grid or checkpoint contract is refused. The
 * state registration is not in it: the buffers are borrowed and change no pool byte. */
static uint32_t build_hash(const RNNoiseGrid *g)
{
    uint32_t h = 2166136261u;
    h = fnv1a_u32(h, (uint32_t)g->sample_rate);
    h = fnv1a_u32(h, (uint32_t)g->n_fft);
    h = fnv1a_u32(h, (uint32_t)g->hop);
    h = fnv1a_u32(h, (uint32_t)g->n_bins);
    h = fnv1a_u32(h, (uint32_t)g->n_bands);
    h = fnv1a_u32(h, (uint32_t)g->min_bins_per_band);
    h = fnv1a_u32(h, (uint32_t)g->lookahead_frames);
    h = fnv1a_u32(h, (uint32_t)g->conv_delay);
    h = fnv1a_u32(h, (uint32_t)g->spec_max_hz);
    h = fnv1a_u32(h, (uint32_t)g->spec_bins);
    h = fnv1a_f64(h, g->erb_norm_tau_sec);
    h = fnv1a_f64(h, g->spec_norm_tau_sec);
    h = fnv1a_f32(h, g->erb_norm_alpha);
    h = fnv1a_f32(h, g->erb_norm_init_lo_db);
    h = fnv1a_f32(h, g->erb_norm_init_hi_db);
    h = fnv1a_f32(h, g->erb_norm_scale_db);
    h = fnv1a_f32(h, g->spec_norm_alpha);
    h = fnv1a_f32(h, g->spec_norm_init_lo);
    h = fnv1a_f32(h, g->spec_norm_init_hi);
    h = fnv1a_f32(h, g->spec_norm_eps);
    h = fnv1a_f32(h, g->stft_scale);
    h = fnv1a_f32(h, g->stft_unscale);
    h = fnv1a_u32(h, (uint32_t)RNNOISE_MODEL_IO_LAYOUT_VERSION);
    h = fnv1a_u32(h, (uint32_t)RNNOISE_PREPOST_LAYOUT_VERSION);
    h = fnv1a_u32(h, (uint32_t)RNNOISE_CARVE_VERSION);
    h = fnv1a_u32(h, (uint32_t)RNNOISE_MODEL_INPUT_FRAMES);
    h = fnv1a_str(h, RNNOISE_FEATURE_VERSION);
    return ac_memreq_abi_hash(h);
}

/* ---- Config / sizing ---- */

int rnnoise_config_defaults(RNNoiseConfig *cfg, const RNNoiseGrid *grid)
{
    if (!cfg) return -1;
    if (rnnoise_grid_validate(grid) != RNNOISE_GRID_OK) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->grid = *grid;
    return 0;
}

/* All three (pointer, nonzero count) pairs or none (process.h). */
static int registration_ok(const RNNoiseConfig *cfg)
{
    const int any = cfg->h1 != NULL || cfg->h1_elements != 0u ||
                    cfg->h2 != NULL || cfg->h2_elements != 0u ||
                    cfg->h3 != NULL || cfg->h3_elements != 0u;
    const int all = cfg->h1 != NULL && cfg->h1_elements != 0u &&
                    cfg->h2 != NULL && cfg->h2_elements != 0u &&
                    cfg->h3 != NULL && cfg->h3_elements != 0u;
    return !any || all;
}

static int check_config(const RNNoiseConfig *cfg)
{
    if (!cfg) return -1;
    if (rnnoise_grid_validate(&cfg->grid) != RNNOISE_GRID_OK) return -1;
    if (!cfg->erb_fwd || !cfg->erb_inv) return -1;
    if (!registration_ok(cfg)) return -1;
    return 0;
}

int rnnoise_get_mem_size(const RNNoiseConfig *cfg, RNNoiseMemReq *req)
{
    size_t total = 0;
    if (!req || check_config(cfg) != 0) return -1;
    if (state_layout(NULL, NULL, &cfg->grid, &total) != 0) return -1;
    memset(req, 0, sizeof(*req));
    req->descriptor_version = RNNOISE_MEM_REQ_VERSION;
    req->layout_version = (uint32_t)RNNOISE_MODEL_IO_LAYOUT_VERSION;
    req->build_flags_hash = build_hash(&cfg->grid);
    req->alignment = (uint32_t)RNNOISE_ALIGNMENT;
    req->bytes = (uint64_t)total;
    return 0;
}

/* ---- Lifecycle ---- */

/* Nonzero span [lo, hi) of each of `lines` matrix lines, line i being m[i*line_step +
 * j*elem_step], j < len (all-zero line: [0, 0)). Columns of the bin-major forward matrix:
 * (bands, bins, 1, bands); rows of the band-major inverse: (bands, bins, bins, 1). */
static void erb_spans(const float *m, int lines, int len, int line_step,
                      int elem_step, uint16_t *lo, uint16_t *hi)
{
    for (int i = 0; i < lines; ++i) {
        int first = 0, last = len;
        const float *line = m + (size_t)i * (size_t)line_step;
        while (first < last && line[(size_t)first * (size_t)elem_step] == 0.0f)
            ++first;
        while (last > first && line[(size_t)(last - 1) * (size_t)elem_step] == 0.0f)
            --last;
        if (first == last) first = last = 0;
        lo[i] = (uint16_t)first;
        hi[i] = (uint16_t)last;
    }
}

/* The linspace seed of one bin: a line from lo to hi held between the two ends, so a span wide enough to
 * cancel in float (lo + (hi - lo) rounding to 0) cannot drop the last bin below its configured end. */
static float norm_seed_linear(float lo, float hi, float position)
{
    float v = lo + position * (hi - lo);
    float vmin = lo < hi ? lo : hi, vmax = lo < hi ? hi : lo;
    return v < vmin ? vmin : (v > vmax ? vmax : v);
}

/* Normaliser start values: a line from lo to hi over the bands (dB) and over the complex bins. */
static void seed_norm_states(RNNoiseState *st)
{
    const RNNoiseGrid *g = &st->grid;
    for (int b = 0; b < g->n_bands; ++b) {
        float pos = (float)b / (float)(g->n_bands - 1);
        st->erb_norm_state[b] = g->erb_norm_init_lo_db +
            pos * (g->erb_norm_init_hi_db - g->erb_norm_init_lo_db);
    }
    for (int k = 0; k < g->spec_bins; ++k) {
        float pos = g->spec_bins > 1 ? (float)k / (float)(g->spec_bins - 1) : 0.0f;
        st->spec_norm_state[k] = norm_seed_linear(g->spec_norm_init_lo, g->spec_norm_init_hi, pos);
    }
}

static void zero_registered(RNNoiseState *st)
{
    if (st->h1) memset(st->h1, 0, st->h1_elements * sizeof(float));
    if (st->h2) memset(st->h2, 0, st->h2_elements * sizeof(float));
    if (st->h3) memset(st->h3, 0, st->h3_elements * sizeof(float));
}

RNNoiseState *rnnoise_init_ex(void *pool, size_t bytes, const RNNoiseConfig *cfg,
                              const RNNoiseMemReq *expected)
{
    RNNoiseMemReq req;
    RNNoiseState *st;
    size_t total = 0;

    if (!pool || rnnoise_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }

    memset(pool, 0, (size_t)req.bytes);
    st = (RNNoiseState *)pool;
    st->grid = cfg->grid;
    st->erb_fwd = cfg->erb_fwd;
    st->erb_inv = cfg->erb_inv;
    if (state_layout(st, (unsigned char *)pool, &st->grid, &total) != 0) return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    erb_spans(st->erb_fwd, st->grid.n_bands, st->grid.n_bins, 1, st->grid.n_bands,
              st->fwd_k0, st->fwd_k1);
    erb_spans(st->erb_inv, st->grid.n_bands, st->grid.n_bins, st->grid.n_bins, 1,
              st->inv_lo, st->inv_hi);
    seed_norm_states(st);

    st->h1 = cfg->h1;
    st->h1_elements = cfg->h1_elements;
    st->h2 = cfg->h2;
    st->h2_elements = cfg->h2_elements;
    st->h3 = cfg->h3;
    st->h3_elements = cfg->h3_elements;
    zero_registered(st);
    st->state_generation = 1u;
    return st;
}

RNNoiseState *rnnoise_init(void *pool, size_t bytes, const RNNoiseConfig *cfg)
{
    return rnnoise_init_ex(pool, bytes, cfg, NULL);
}

void rnnoise_reset(RNNoiseState *st)
{
    if (!st) return;
    if (st->mutable_base) memset(st->mutable_base, 0, st->mutable_bytes);
    st->feat_idx = 0;
    st->feat_count = 0;
    seed_norm_states(st);
    zero_registered(st);
    ++st->state_generation;
}

unsigned int rnnoise_state_generation(const RNNoiseState *st)
{
    return st ? st->state_generation : 0u;
}

/* ---- Pre-processing ---- */

/* erb_db[b] = sum over k in [k0[b], k1[b]) ascending of power[k] * fwd[k * n_bands + b],
 * from 0.0f: the sum of the dense product with its exact zeros left out. Four adjacent
 * bands advance together with local accumulators; the remaining bands run one at a time.
 * Every power must be finite (a non-finite one takes the dense walk below). */
static void erb_forward_banded(const float *power, const float *fwd,
                               const uint16_t *k0, const uint16_t *k1, int n_bands,
                               float *erb_db)
{
    const size_t stride = (size_t)n_bands;
    int b = 0;
    for (; b + 4 <= n_bands; b += 4) {
        const int a0 = k0[b], a1 = k0[b + 1], a2 = k0[b + 2], a3 = k0[b + 3];
        const int n0 = k1[b] - a0, n1 = k1[b + 1] - a1;
        const int n2 = k1[b + 2] - a2, n3 = k1[b + 3] - a3;
        int common = n0 < n1 ? n0 : n1;
        const float *w0 = fwd + (size_t)a0 * stride + b;
        const float *w1 = fwd + (size_t)a1 * stride + b + 1;
        const float *w2 = fwd + (size_t)a2 * stride + b + 2;
        const float *w3 = fwd + (size_t)a3 * stride + b + 3;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int j;
        if (n2 < common) common = n2;
        if (n3 < common) common = n3;
        for (j = 0; j < common; ++j) {
            s0 += power[a0 + j] * w0[(size_t)j * stride];
            s1 += power[a1 + j] * w1[(size_t)j * stride];
            s2 += power[a2 + j] * w2[(size_t)j * stride];
            s3 += power[a3 + j] * w3[(size_t)j * stride];
        }
        for (j = common; j < n0; ++j) s0 += power[a0 + j] * w0[(size_t)j * stride];
        for (j = common; j < n1; ++j) s1 += power[a1 + j] * w1[(size_t)j * stride];
        for (j = common; j < n2; ++j) s2 += power[a2 + j] * w2[(size_t)j * stride];
        for (j = common; j < n3; ++j) s3 += power[a3 + j] * w3[(size_t)j * stride];
        erb_db[b] = s0;
        erb_db[b + 1] = s1;
        erb_db[b + 2] = s2;
        erb_db[b + 3] = s3;
    }
    for (; b < n_bands; ++b) {
        const int a = k0[b], n = k1[b] - a;
        const float *w = fwd + (size_t)a * stride + b;
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += power[a + j] * w[(size_t)j * stride];
        erb_db[b] = s;
    }
}

int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float *out_erb, float *out_spec)
{
    if (!st || !spec_re || !spec_im || !out_erb || !out_spec) return -1;
    const RNNoiseGrid *g = &st->grid;
    const int n_bins = g->n_bins;
    const int n_bands = g->n_bands;
    const int spec_bins = g->spec_bins;

    float *power = st->scratch_power;
    int i = 0;
#if SK_HAVE_NEON
    for (; i + 4 <= n_bins; i += 4) {
        float32x4_t re = vld1q_f32(spec_re + i);
        float32x4_t im = vld1q_f32(spec_im + i);
        vst1q_f32(power + i,
                  vaddq_f32(vmulq_f32(re, re), vmulq_f32(im, im)));
    }
#endif
    for (; i < n_bins; i++) {
        power[i] = spec_re[i] * spec_re[i] + spec_im[i] * spec_im[i];
    }

    /* ERB band energy in dB. A non-finite power takes the dense walk: Inf/NaN times a zero
     * weight is NaN, so it reaches every band. */
    float *erb_db = st->scratch_erb_db;
    if (skn_all_finite_f32(power, (size_t)n_bins)) {
        erb_forward_banded(power, st->erb_fwd, st->fwd_k0, st->fwd_k1, n_bands, erb_db);
    } else {
        memset(erb_db, 0, (size_t)n_bands * sizeof(float));
        for (int k = 0; k < n_bins; ++k) {
            const float p = power[k];
            const float *row = st->erb_fwd + (size_t)k * (size_t)n_bands;
            for (int b = 0; b < n_bands; ++b) erb_db[b] += p * row[b];
        }
    }
    for (int b = 0; b < n_bands; ++b) {
        erb_db[b] = 10.0f * log10f(erb_db[b] + LOG_FLOOR);
    }

    /* ERB branch (band_mean_norm_erb): the EMA takes this frame before the subtraction. */
    const int idx = st->feat_idx;
    const float erb_a = g->erb_norm_alpha;
    float *erb_feat = st->erb_feat_buf + (size_t)idx * (size_t)n_bands;
    int b = 0;
#if SK_HAVE_NEON
    {
        const float32x4_t va = vdupq_n_f32(erb_a);
        const float32x4_t vb = vdupq_n_f32(1.0f - erb_a);
        const float32x4_t scale = vdupq_n_f32(g->erb_norm_scale_db);
        for (; b + 4 <= n_bands; b += 4) {
            float32x4_t db = vld1q_f32(erb_db + b);
            float32x4_t mean = vaddq_f32(
                vmulq_f32(va, vld1q_f32(st->erb_norm_state + b)),
                vmulq_f32(vb, db));
            vst1q_f32(st->erb_norm_state + b, mean);
            vst1q_f32(erb_feat + b, vdivq_f32(vsubq_f32(db, mean), scale));
        }
    }
#endif
    for (; b < n_bands; b++) {
        float mean = erb_a * st->erb_norm_state[b] + (1.0f - erb_a) * erb_db[b];
        float feat = (erb_db[b] - mean) / g->erb_norm_scale_db;
        st->erb_norm_state[b] = mean;
        erb_feat[b] = feat;
    }

    /* Complex branch (band_unit_norm): the magnitude EMA takes this frame before the division. */
    const float norm_a = g->spec_norm_alpha;
    float *spec_feat_re = st->spec_feat_buf + (size_t)idx * 2u * (size_t)spec_bins;
    float *spec_feat_im = spec_feat_re + spec_bins;
    int k = 0;
#if SK_HAVE_NEON
    {
        const float32x4_t va = vdupq_n_f32(norm_a);
        const float32x4_t vb = vdupq_n_f32(1.0f - norm_a);
        const float32x4_t eps = vdupq_n_f32(g->spec_norm_eps);
        for (; k + 4 <= spec_bins; k += 4) {
            float32x4_t state = vaddq_f32(
                vmulq_f32(va, vld1q_f32(st->spec_norm_state + k)),
                vmulq_f32(vb, vsqrtq_f32(vld1q_f32(power + k))));
            float32x4_t denom = vsqrtq_f32(vaddq_f32(state, eps));
            vst1q_f32(st->spec_norm_state + k, state);
            vst1q_f32(spec_feat_re + k,
                      vdivq_f32(vld1q_f32(spec_re + k), denom));
            vst1q_f32(spec_feat_im + k,
                      vdivq_f32(vld1q_f32(spec_im + k), denom));
        }
    }
#endif
    for (; k < spec_bins; k++) {
        float magnitude = sqrtf(power[k]);
        float state = norm_a * st->spec_norm_state[k] +
            (1.0f - norm_a) * magnitude;
        float denom = sqrtf(state + g->spec_norm_eps);
        float re = spec_re[k] / denom;
        float im = spec_im[k] / denom;
        st->spec_norm_state[k] = state;
        spec_feat_re[k] = re;
        spec_feat_im[k] = im;
    }
    st->feat_idx = (idx + 1) % RNNOISE_MODEL_INPUT_FRAMES;
    /* Only readiness matters; saturation avoids signed overflow on long streams. */
    if (st->feat_count < RNNOISE_MODEL_INPUT_FRAMES) ++st->feat_count;

    if (st->feat_count < RNNOISE_MODEL_INPUT_FRAMES) return 0;

    int oldest = st->feat_idx;  /* feat_idx is the next write slot, i.e. the oldest */
    for (int f = 0; f < RNNOISE_MODEL_INPUT_FRAMES; f++) {
        int src = (oldest + f) % RNNOISE_MODEL_INPUT_FRAMES;
        memcpy(out_erb + (size_t)f * (size_t)n_bands,
               st->erb_feat_buf + (size_t)src * (size_t)n_bands,
               sizeof(float) * (size_t)n_bands);
        memcpy(out_spec + (size_t)f * 2u * (size_t)spec_bins,
               st->spec_feat_buf + (size_t)src * 2u * (size_t)spec_bins,
               sizeof(float) * 2u * (size_t)spec_bins);
    }
    return 1;
}

/* ---- Accelerator boundary ---- */

/* The registered tensors, whole buffers: all finite, or nothing registered. */
static int registered_finite(const RNNoiseState *st)
{
    return (st->h1 == NULL || skn_all_finite_f32(st->h1, st->h1_elements)) &&
           (st->h2 == NULL || skn_all_finite_f32(st->h2, st->h2_elements)) &&
           (st->h3 == NULL || skn_all_finite_f32(st->h3, st->h3_elements));
}

int rnnoise_frame_finish(RNNoiseState *st, const float *band_gains)
{
    if (!st || !band_gains) return -1;
    if (skn_all_finite_f32(band_gains, (size_t)st->grid.n_bands) &&
        registered_finite(st))
        return 0;
    /* Zero is the only state a stream can restart from after an in-place overwrite. */
    zero_registered(st);
    ++st->state_generation;
    return -1;
}

/* ---- Post-processing ---- */

int rnnoise_apply_atten_lim(const RNNoiseState *st, float *band_gains,
                            float atten_lim_db)
{
    if (!st || !band_gains || !isfinite(atten_lim_db)) return -1;
    if (atten_lim_db <= 0.0f) return 0;
    const int n_bands = st->grid.n_bands;
    float lim = powf(10.0f, -atten_lim_db / 20.0f);
    int b = 0;
#if SK_HAVE_NEON
    {
        float32x4_t vlim = vdupq_n_f32(lim);
        float32x4_t mix = vdupq_n_f32(1.0f - lim);
        for (; b + 4 <= n_bands; b += 4) {
            vst1q_f32(band_gains + b,
                      vaddq_f32(vlim,
                                vmulq_f32(vld1q_f32(band_gains + b), mix)));
        }
    }
#endif
    for (; b < n_bands; b++) {
        band_gains[b] = lim + band_gains[b] * (1.0f - lim);
    }
    return 0;
}

int rnnoise_expand_gains(const RNNoiseState *st, const float *band_gains,
                         float *bin_gains)
{
    if (!st || !band_gains || !bin_gains) return -1;
    /* bin_gains = gains @ erb_inv (mode=1, partition of unity): per bin the bands add in
     * ascending order over each band's nonzero span. */
    skn_erb_expand_mask_f32(band_gains, st->erb_inv, st->inv_lo, st->inv_hi,
                            st->grid.n_bins, st->grid.n_bands, bin_gains);
    return 0;
}

int rnnoise_apply_gains(const RNNoiseState *st, float *spec_re, float *spec_im,
                        const float *bin_gains)
{
    if (!st || !spec_re || !spec_im || !bin_gains) return -1;
    const int n_bins = st->grid.n_bins;
    const float unscale = st->grid.stft_unscale;
    int i = 0;
#if SK_HAVE_NEON
    for (; i + 4 <= n_bins; i += 4) {
        float32x4_t gain = vmulq_n_f32(vld1q_f32(bin_gains + i), unscale);
        vst1q_f32(spec_re + i,
                  vmulq_f32(vld1q_f32(spec_re + i), gain));
        vst1q_f32(spec_im + i,
                  vmulq_f32(vld1q_f32(spec_im + i), gain));
    }
#endif
    for (; i < n_bins; i++) {
        spec_re[i] *= bin_gains[i] * unscale;
        spec_im[i] *= bin_gains[i] * unscale;
    }
    return 0;
}

const char *rnnoise_simd_backend(void)
{
    return SK_NEON_BACKEND_NAME;
}
