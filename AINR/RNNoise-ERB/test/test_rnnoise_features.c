#define _POSIX_C_SOURCE 200809L
/* Independent C reference for the RNNOISE_FEATURE_VERSION feature contract (dense product with
 * the loaded forward matrix against the library's banded kernel), and the call contracts: the
 * registered recurrent state, a non-finite spectrum, the attenuation limit and the pool
 * descriptor. Usage: test_rnnoise_features ERB_FWD_FILE ERB_INV_FILE */

#include "rnnoise_test_support.h"

#include <float.h>
#include <math.h>

#include "utility/mem_align.h"

#define TEST_FRAMES 4096
#define TOL 2e-5f
#define H_ELEMS 128          /* any nonzero count works; the shipped graph's GRU width */
#define N_STATE 3

typedef struct {
    float erb_norm[RNN_TEST_N_BANDS];
    float spec_norm[RNN_TEST_SPEC_BINS];
    float erb_history[3][RNN_TEST_N_BANDS];
    float spec_history[3][2][RNN_TEST_SPEC_BINS];
    int index;
    int count;
} RefState;

static int g_checks, g_fail;
#define CHECK(name, cond) do { ++g_checks; if (!(cond)) { ++g_fail; \
    printf("FAIL [%s] line %d: %s\n", (name), __LINE__, #cond); } } while (0)

static void fill_stationary_spectrum(float *re, float *im) {
    for (int k = 0; k < RNN_TEST_N_BINS; ++k) {
        re[k] = 0.0007f * (float)(1 + (k * 17) % 29);
        im[k] = 0.0003f * (float)((k * 11) % 23);
    }
    im[0] = 0.0f;
    im[RNN_TEST_N_BINS - 1] = 0.0f;
}

static int close_enough(float got, float want, const char *what,
                        int frame, int index) {
    float scale = fmaxf(1.0f, fmaxf(fabsf(got), fabsf(want)));
    if (fabsf(got - want) <= TOL * scale) return 1;
    printf("FAIL: %s frame=%d index=%d got=%.9g want=%.9g\n",
           what, frame, index, (double)got, (double)want);
    return 0;
}

static void ref_init(RefState *st, const RNNoiseGrid *g) {
    memset(st, 0, sizeof(*st));
    for (int b = 0; b < RNN_TEST_N_BANDS; ++b) {
        float pos = (float)b / (float)(RNN_TEST_N_BANDS - 1);
        st->erb_norm[b] = g->erb_norm_init_lo_db +
            pos * (g->erb_norm_init_hi_db - g->erb_norm_init_lo_db);
    }
    for (int k = 0; k < RNN_TEST_SPEC_BINS; ++k) {
        float pos = (float)k / (float)(RNN_TEST_SPEC_BINS - 1);
        st->spec_norm[k] = g->spec_norm_init_lo +
            pos * (g->spec_norm_init_hi - g->spec_norm_init_lo);
    }
}

static int ref_step(RefState *st, const RNNoiseGrid *g, const float *fwd,
                    const float *re, const float *im,
                    float out_erb[3][RNN_TEST_N_BANDS],
                    float out_spec[3][2][RNN_TEST_SPEC_BINS]) {
    const float erb_alpha = g->erb_norm_alpha;
    const float spec_alpha = g->spec_norm_alpha;
    const int idx = st->index;

    for (int b = 0; b < RNN_TEST_N_BANDS; ++b) {
        float energy = 0.0f;
        for (int k = 0; k < RNN_TEST_N_BINS; ++k) {
            float power = re[k] * re[k] + im[k] * im[k];
            energy += power * fwd[k * RNN_TEST_N_BANDS + b];
        }
        float erb_db = 10.0f * log10f(energy + 1e-10f);
        float mean = erb_alpha * st->erb_norm[b] +
            (1.0f - erb_alpha) * erb_db;
        float feat = (erb_db - mean) / g->erb_norm_scale_db;
        st->erb_norm[b] = mean;
        st->erb_history[idx][b] = feat;
    }

    for (int k = 0; k < RNN_TEST_SPEC_BINS; ++k) {
        float magnitude = sqrtf(re[k] * re[k] + im[k] * im[k]);
        float state = spec_alpha * st->spec_norm[k] +
            (1.0f - spec_alpha) * magnitude;
        float denom = sqrtf(state + g->spec_norm_eps);
        float re_norm = re[k] / denom;
        float im_norm = im[k] / denom;
        st->spec_norm[k] = state;
        st->spec_history[idx][0][k] = re_norm;
        st->spec_history[idx][1][k] = im_norm;
    }

    st->index = (idx + 1) % 3;
    if (st->count < 3) ++st->count;
    if (st->count < 3) return 0;

    for (int f = 0; f < 3; ++f) {
        int src = (st->index + f) % 3;
        memcpy(out_erb[f], st->erb_history[src], sizeof(out_erb[f]));
        memcpy(out_spec[f], st->spec_history[src], sizeof(out_spec[f]));
    }
    return 1;
}

/* A state in its pool with the three recurrent buffers registered (or none: the board form). */
typedef struct {
    RnnPool pool;
    float   h[N_STATE][H_ELEMS];
} Reg;

static int reg_open(Reg *r, const RNNoiseGrid *grid, const float *fwd, const float *inv,
                    int registered)
{
    memset(&r->pool, 0, sizeof(r->pool));
    for (int i = 0; i < N_STATE; ++i)
        for (int k = 0; k < H_ELEMS; ++k) r->h[i][k] = 0.25f + (float)(i * H_ELEMS + k);
    if (rnnoise_config_defaults(&r->pool.cfg, grid) != 0) return -1;
    r->pool.cfg.erb_fwd = fwd;
    r->pool.cfg.erb_inv = inv;
    if (registered) {
        r->pool.cfg.h1 = r->h[0]; r->pool.cfg.h1_elements = H_ELEMS;
        r->pool.cfg.h2 = r->h[1]; r->pool.cfg.h2_elements = H_ELEMS;
        r->pool.cfg.h3 = r->h[2]; r->pool.cfg.h3_elements = H_ELEMS;
    }
    return rnn_pool_open(&r->pool);
}

static void reg_fill(Reg *r, float base)
{
    for (int i = 0; i < N_STATE; ++i)
        for (int k = 0; k < H_ELEMS; ++k) r->h[i][k] = base + (float)(i * H_ELEMS + k);
}

static int reg_all_zero(const Reg *r)
{
    for (int i = 0; i < N_STATE; ++i)
        for (int k = 0; k < H_ELEMS; ++k)
            if (r->h[i][k] != 0.0f) return 0;
    return 1;
}

/* The pool block as carved (header and every array) plus the registered buffers. */
typedef struct {
    unsigned char *block;
    size_t         bytes;
    float          h[N_STATE][H_ELEMS];
} Snap;

static void snap_take(Snap *s, const Reg *r)
{
    s->bytes = (size_t)r->pool.req.bytes;
    s->block = (unsigned char *)malloc(s->bytes);
    if (!s->block) abort();
    memcpy(s->block, r->pool.pool, s->bytes);
    memcpy(s->h, r->h, sizeof(s->h));
}

/* The pool block equals the snapshot, the generation being the one header field a refusal may
 * change (`generation_plus` of it). */
static int snap_block_same(const Snap *s, const Reg *r, unsigned int generation_plus)
{
    unsigned char *expect = (unsigned char *)malloc(s->bytes);
    int same;
    if (!expect) abort();
    memcpy(expect, s->block, s->bytes);
    ((RNNoiseState *)(void *)expect)->state_generation += generation_plus;
    same = memcmp(expect, r->pool.pool, s->bytes) == 0;
    free(expect);
    return same;
}

static int snap_buffers_same(const Snap *s, const Reg *r)
{
    return memcmp(s->h, r->h, sizeof(s->h)) == 0;
}

static void snap_free(Snap *s) { free(s->block); s->block = NULL; }

static void feed(Reg *r, int frames)
{
    static float re[RNN_TEST_N_BINS], im[RNN_TEST_N_BINS];
    static float erb[3][RNN_TEST_N_BANDS], spec[3][2][RNN_TEST_SPEC_BINS];
    fill_stationary_spectrum(re, im);
    for (int t = 0; t < frames; ++t)
        rnnoise_compute_features(r->pool.st, re, im, &erb[0][0], &spec[0][0][0]);
}

static void gains_fill(float *gains, int n_bands, float v)
{
    for (int b = 0; b < n_bands; ++b) gains[b] = v;
}

static void registered_state_gates(const RNNoiseGrid *grid, const float *fwd, const float *inv)
{
    static Reg reg, fresh, board;
    static float gains[RNN_TEST_N_BANDS];
    static const float bad_value[3] = {NAN, INFINITY, -INFINITY};
    const int where[3] = {0, H_ELEMS / 2, H_ELEMS - 1};
    Snap snap;

    gains_fill(gains, grid->n_bands, 0.5f);

    /* init: dirty buffers are zeroed, the pointers and counts are kept */
    CHECK("reg.open", reg_open(&reg, grid, fwd, inv, 1) == 0);
    CHECK("reg.init_zeroes", reg_all_zero(&reg));
    CHECK("reg.init_generation", rnnoise_state_generation(reg.pool.st) == 1u);
    CHECK("reg.init_pointers", reg.pool.st->h1 == reg.h[0] && reg.pool.st->h2 == reg.h[1] &&
          reg.pool.st->h3 == reg.h[2]);
    CHECK("reg.init_counts", reg.pool.st->h1_elements == H_ELEMS &&
          reg.pool.st->h2_elements == H_ELEMS && reg.pool.st->h3_elements == H_ELEMS);

    /* finite gains and finite state: nothing in the pool or the buffers changes */
    feed(&reg, 5);
    reg_fill(&reg, 0.5f);
    snap_take(&snap, &reg);
    CHECK("finish.finite", rnnoise_frame_finish(reg.pool.st, gains) == 0);
    CHECK("finish.finite_untouched",
          snap_block_same(&snap, &reg, 0u) && snap_buffers_same(&snap, &reg));
    snap_free(&snap);

    /* a non-finite element anywhere in any buffer: -1, all three zeroed, generation + 1 */
    for (int i = 0; i < N_STATE; ++i) {
        for (int w = 0; w < 3; ++w) {
            for (int v = 0; v < 3; ++v) {
                unsigned int before = rnnoise_state_generation(reg.pool.st);
                reg_fill(&reg, 0.5f);
                reg.h[i][where[w]] = bad_value[v];
                snap_take(&snap, &reg);
                CHECK("finish.bad_state", rnnoise_frame_finish(reg.pool.st, gains) == -1);
                CHECK("finish.bad_state_zeroed", reg_all_zero(&reg));
                CHECK("finish.bad_state_generation",
                      rnnoise_state_generation(reg.pool.st) == before + 1u);
                CHECK("finish.bad_state_block_unchanged", snap_block_same(&snap, &reg, 1u));
                snap_free(&snap);
                /* per call, not a latch: the zeroed state finishes cleanly */
                before = rnnoise_state_generation(reg.pool.st);
                CHECK("finish.after_refusal", rnnoise_frame_finish(reg.pool.st, gains) == 0 &&
                      rnnoise_state_generation(reg.pool.st) == before);
            }
        }
    }

    /* a non-finite gain with a finite state: the same refusal */
    for (int w = 0; w < 3; ++w) {
        const int band = w == 0 ? 0 : (w == 1 ? grid->n_bands / 2 : grid->n_bands - 1);
        for (int v = 0; v < 3; ++v) {
            unsigned int before = rnnoise_state_generation(reg.pool.st);
            reg_fill(&reg, 0.5f);
            gains[band] = bad_value[v];
            CHECK("finish.bad_gain", rnnoise_frame_finish(reg.pool.st, gains) == -1);
            CHECK("finish.bad_gain_zeroed", reg_all_zero(&reg));
            CHECK("finish.bad_gain_generation",
                  rnnoise_state_generation(reg.pool.st) == before + 1u);
            gains[band] = 0.5f;
        }
    }

    /* NULL arguments change nothing, not even the generation */
    reg_fill(&reg, 0.5f);
    snap_take(&snap, &reg);
    CHECK("finish.null_state", rnnoise_frame_finish(NULL, gains) == -1);
    CHECK("finish.null_gains", rnnoise_frame_finish(reg.pool.st, NULL) == -1);
    CHECK("finish.null_untouched",
          snap_block_same(&snap, &reg, 0u) && snap_buffers_same(&snap, &reg));
    snap_free(&snap);

    /* reset: registered state zeroed, generation + 1, the rest as a fresh init */
    CHECK("reset.open_fresh", reg_open(&fresh, grid, fwd, inv, 1) == 0);
    feed(&reg, 7);
    reg_fill(&reg, 0.5f);
    {
        unsigned int before = rnnoise_state_generation(reg.pool.st);
        rnnoise_reset(reg.pool.st);
        CHECK("reset.zeroes", reg_all_zero(&reg));
        CHECK("reset.generation", rnnoise_state_generation(reg.pool.st) == before + 1u);
        CHECK("reset.mutable_block", reg.pool.st->mutable_bytes == fresh.pool.st->mutable_bytes &&
              memcmp(reg.pool.st->mutable_base, fresh.pool.st->mutable_base,
                     reg.pool.st->mutable_bytes) == 0);
        CHECK("reset.ring", reg.pool.st->feat_idx == 0 && reg.pool.st->feat_count == 0);
    }

    /* board form: nothing registered; the library holds no state but still reports the refusal */
    CHECK("board.open", reg_open(&board, grid, fwd, inv, 0) == 0);
    CHECK("board.pointers", board.pool.st->h1 == NULL && board.pool.st->h2 == NULL &&
          board.pool.st->h3 == NULL && board.pool.st->h1_elements == 0u);
    CHECK("board.init_generation", rnnoise_state_generation(board.pool.st) == 1u);
    reg_fill(&board, 0.5f);
    snap_take(&snap, &board);
    CHECK("board.finish_finite", rnnoise_frame_finish(board.pool.st, gains) == 0);
    CHECK("board.finish_untouched",
          snap_block_same(&snap, &board, 0u) && snap_buffers_same(&snap, &board));
    {
        unsigned int before = rnnoise_state_generation(board.pool.st);
        gains[0] = NAN;
        CHECK("board.finish_bad_gain", rnnoise_frame_finish(board.pool.st, gains) == -1);
        CHECK("board.finish_generation", rnnoise_state_generation(board.pool.st) == before + 1u);
        CHECK("board.buffers_not_ours", snap_buffers_same(&snap, &board));
        gains[0] = 0.5f;
    }
    {
        unsigned int before = rnnoise_state_generation(board.pool.st);
        rnnoise_reset(board.pool.st);
        CHECK("board.reset_generation", rnnoise_state_generation(board.pool.st) == before + 1u);
        rnnoise_reset(NULL);
        CHECK("generation.null", rnnoise_state_generation(NULL) == 0u);
    }
    snap_free(&snap);
}

/* A registration must be all three (pointer, nonzero count) pairs or none. */
static void registration_gates(const RNNoiseGrid *grid, const float *fwd, const float *inv)
{
    static float h[N_STATE][H_ELEMS];
    RNNoiseConfig cfg, plain;
    RNNoiseMemReq req, want;
    unsigned char *pool;
    unsigned char *copy;
    size_t bytes;
    int variant;

    rnnoise_config_defaults(&plain, grid);
    plain.erb_fwd = fwd;
    plain.erb_inv = inv;
    CHECK("regcfg.plain", rnnoise_get_mem_size(&plain, &want) == 0);
    bytes = (size_t)want.bytes;
    pool = (unsigned char *)rnn_test_aligned(bytes);
    copy = (unsigned char *)malloc(bytes);
    CHECK("regcfg.alloc", pool && copy);
    if (!pool || !copy) { free(copy); free(pool); return; }
    memset(pool, 0xA5, bytes);
    memcpy(copy, pool, bytes);

    cfg = plain;
    cfg.h1 = h[0]; cfg.h1_elements = H_ELEMS;
    cfg.h2 = h[1]; cfg.h2_elements = H_ELEMS;
    cfg.h3 = h[2]; cfg.h3_elements = H_ELEMS;
    CHECK("regcfg.full", rnnoise_get_mem_size(&cfg, &req) == 0);
    CHECK("regcfg.same_pool", memcmp(&req, &want, sizeof(req)) == 0);   /* bytes and hash */

    for (variant = 0; variant < 9; ++variant) {
        cfg = plain;
        switch (variant) {
        case 0: cfg.h1 = h[0]; break;                                    /* pointer, no count */
        case 1: cfg.h1_elements = H_ELEMS; break;                        /* count, no pointer */
        case 2: cfg.h1 = h[0]; cfg.h1_elements = H_ELEMS; break;         /* one pair of three */
        case 3: cfg.h1 = h[0]; cfg.h1_elements = H_ELEMS;
                cfg.h2 = h[1]; cfg.h2_elements = H_ELEMS; break;         /* two pairs of three */
        case 4: cfg.h1 = h[0]; cfg.h1_elements = H_ELEMS;
                cfg.h2 = h[1]; cfg.h2_elements = H_ELEMS;
                cfg.h3 = h[2]; break;                                    /* third without count */
        case 5: cfg.h1 = h[0]; cfg.h1_elements = H_ELEMS;
                cfg.h2 = h[1]; cfg.h2_elements = H_ELEMS;
                cfg.h3_elements = H_ELEMS; break;                        /* third without pointer */
        case 6: cfg.h1_elements = H_ELEMS; cfg.h2_elements = H_ELEMS;
                cfg.h3_elements = H_ELEMS; break;                        /* counts only */
        case 7: cfg.h1 = h[0]; cfg.h2 = h[1]; cfg.h3 = h[2]; break;      /* pointers only */
        default: cfg.erb_fwd = NULL; break;                              /* missing matrix */
        }
        memset(&req, 0x5A, sizeof(req));
        {
            RNNoiseMemReq junk = req;
            CHECK("regcfg.size_refused", rnnoise_get_mem_size(&cfg, &req) == -1);
            CHECK("regcfg.size_untouched", memcmp(&req, &junk, sizeof(req)) == 0);
        }
        CHECK("regcfg.init_refused", rnnoise_init(pool, bytes, &cfg) == NULL);
        CHECK("regcfg.pool_untouched", memcmp(pool, copy, bytes) == 0);   /* whole block */
    }
    CHECK("regcfg.init_ok", rnnoise_init(pool, bytes, &plain) != NULL);
    free(copy);
    free(pool);
}

/* ---- non-finite spectrum ---- */

typedef struct {
    float erb[3][RNN_TEST_N_BANDS];
    float spec[3][2][RNN_TEST_SPEC_BINS];
} FeatWindow;

static void reg_close(Reg *r)
{
    free(r->pool.pool);
    memset(r, 0, sizeof(*r));
}

static void fill_spectrum(float *re, float *im, int t)
{
    const float scale = 0.65f + 0.07f * (float)t;
    for (int k = 0; k < RNN_TEST_N_BINS; ++k) {
        re[k] = scale * (0.0008f + 0.00005f * (float)((k * 17 + t * 3) % 31));
        im[k] = scale * (-0.0004f + 0.00004f * (float)((k * 11 + t * 5) % 23));
    }
    im[0] = 0.0f;
    im[RNN_TEST_N_BINS - 1] = 0.0f;
}

/* Frame t of a fixed stream; bad_bin >= 0 overwrites that bin of the real or imaginary plane. */
static int feed_frame(Reg *r, int t, int bad_bin, int bad_imag, float bad, FeatWindow *out)
{
    static float re[RNN_TEST_N_BINS], im[RNN_TEST_N_BINS];
    fill_spectrum(re, im, t);
    if (bad_bin >= 0) (bad_imag ? im : re)[bad_bin] = bad;
    return rnnoise_compute_features(r->pool.st, re, im, &out->erb[0][0], &out->spec[0][0][0]);
}

static int none_finite(const float *x, int n)
{
    for (int i = 0; i < n; ++i)
        if (isfinite(x[i])) return 0;
    return 1;
}

static int all_finite(const float *x, int n)
{
    for (int i = 0; i < n; ++i)
        if (!isfinite(x[i])) return 0;
    return 1;
}

/* A non-finite bin reaches every band of the newest ERB window (process.h). Observed
 * behaviour, not in process.h: the window stays non-finite on later finite frames until
 * rnnoise_reset(); after the reset the stream is a fresh instance's, bit for bit. */
static void nonfinite_spectrum_gates(const RNNoiseGrid *grid, const float *fwd, const float *inv)
{
    static Reg bad, fresh;
    static FeatWindow got, want;
    static const float value[3] = {NAN, INFINITY, -INFINITY};
    /* first bin, last complex-branch bin, above the complex branch, the last bin */
    const int bin[4] = {0, RNN_TEST_SPEC_BINS - 1, 200, RNN_TEST_N_BINS - 1};

    CHECK("nonfinite.open", reg_open(&bad, grid, fwd, inv, 0) == 0);
    for (int plane = 0; plane < 2; ++plane) {
        for (int i = 0; i < 4; ++i) {
            for (int v = 0; v < 3; ++v) {
                int ready = 0;
                rnnoise_reset(bad.pool.st);
                for (int t = 0; t < 3; ++t) ready = feed_frame(&bad, t, -1, 0, 0.0f, &got);
                CHECK("nonfinite.window_ready", ready == 1);
                ready = feed_frame(&bad, 3, bin[i], plane, value[v], &got);
                CHECK("nonfinite.every_band", ready == 1 && none_finite(got.erb[2], grid->n_bands));
                ready = feed_frame(&bad, 4, -1, 0, 0.0f, &got);
                CHECK("nonfinite.until_reset", ready == 1 && none_finite(got.erb[2], grid->n_bands));

                rnnoise_reset(bad.pool.st);
                CHECK("nonfinite.fresh_open", reg_open(&fresh, grid, fwd, inv, 0) == 0);
                for (int t = 0; t < 6; ++t) {
                    const int r_bad = feed_frame(&bad, t, -1, 0, 0.0f, &got);
                    const int r_fresh = feed_frame(&fresh, t, -1, 0, 0.0f, &want);
                    CHECK("nonfinite.reset_ready", r_bad == r_fresh);
                    CHECK("nonfinite.reset_output",
                          r_bad == 0 || memcmp(&got, &want, sizeof(got)) == 0);
                }
                CHECK("nonfinite.reset_state",
                      bad.pool.st->mutable_bytes == fresh.pool.st->mutable_bytes &&
                      memcmp(bad.pool.st->mutable_base, fresh.pool.st->mutable_base,
                             bad.pool.st->mutable_bytes) == 0);
                reg_close(&fresh);
            }
        }
    }
    reg_close(&bad);
}

/* ---- attenuation limit ---- */

/* -1 only for a NULL argument or a non-finite limit, with the gains and the state untouched;
 * <= 0 disables the mixture and is accepted. */
static void atten_lim_gates(const RNNoiseGrid *grid, const float *fwd, const float *inv)
{
    static Reg reg;
    static const float bad_db[3] = {NAN, INFINITY, -INFINITY};
    static const float off_db[3] = {0.0f, -3.0f, -FLT_MAX};
    float gains[RNN_TEST_N_BANDS], keep[RNN_TEST_N_BANDS];
    Snap snap;
    float lim;

    for (int b = 0; b < grid->n_bands; ++b) keep[b] = 0.05f + 0.04f * (float)b;
    CHECK("atten.open", reg_open(&reg, grid, fwd, inv, 1) == 0);
    feed(&reg, 5);
    reg_fill(&reg, 0.5f);
    snap_take(&snap, &reg);

    for (int v = 0; v < 3; ++v) {
        memcpy(gains, keep, sizeof(gains));
        CHECK("atten.nonfinite_refused",
              rnnoise_apply_atten_lim(reg.pool.st, gains, bad_db[v]) == -1);
        CHECK("atten.nonfinite_gains_untouched", memcmp(gains, keep, sizeof(gains)) == 0);
    }
    memcpy(gains, keep, sizeof(gains));
    CHECK("atten.null_state", rnnoise_apply_atten_lim(NULL, gains, 6.0f) == -1);
    CHECK("atten.null_gains", rnnoise_apply_atten_lim(reg.pool.st, NULL, 6.0f) == -1);
    CHECK("atten.null_gains_untouched", memcmp(gains, keep, sizeof(gains)) == 0);
    CHECK("atten.refused_state_untouched",
          snap_block_same(&snap, &reg, 0u) && snap_buffers_same(&snap, &reg));
    snap_free(&snap);

    for (int v = 0; v < 3; ++v) {
        memcpy(gains, keep, sizeof(gains));
        CHECK("atten.disabled_status",
              rnnoise_apply_atten_lim(reg.pool.st, gains, off_db[v]) == 0);
        CHECK("atten.disabled_gains_untouched", memcmp(gains, keep, sizeof(gains)) == 0);
    }
    /* finite but huge: the limit underflows to 0, so the mixture returns the gains */
    memcpy(gains, keep, sizeof(gains));
    CHECK("atten.huge_status", rnnoise_apply_atten_lim(reg.pool.st, gains, 1e30f) == 0);
    CHECK("atten.huge_gains_unchanged", memcmp(gains, keep, sizeof(gains)) == 0);

    /* after the refusals the call still mixes: lim + gain * (1 - lim) */
    memcpy(gains, keep, sizeof(gains));
    CHECK("atten.mix_status", rnnoise_apply_atten_lim(reg.pool.st, gains, 6.0f) == 0);
    lim = powf(10.0f, -6.0f / 20.0f);
    for (int b = 0; b < grid->n_bands; ++b) {
        CHECK("atten.mix_value", fabsf(gains[b] - (lim + keep[b] * (1.0f - lim))) <= 1e-6f);
        CHECK("atten.mix_moves", fabsf(gains[b] - keep[b]) > 0.01f);
    }
    reg_close(&reg);
}

/* ---- pool descriptor ---- */

static uint32_t fnv_u32(uint32_t h, uint32_t v)
{
    for (int i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

static uint32_t fnv_f32(uint32_t h, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    return fnv_u32(h, bits);
}

static uint32_t fnv_f64(uint32_t h, double v)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    return fnv_u32(fnv_u32(h, (uint32_t)bits), (uint32_t)(bits >> 32));
}

/* build_flags_hash restated: FNV-1a-32 over the grid fields in declaration order, the layout,
 * carve and input-window constants and the feature tag, then audiolib's ABI token. Equality
 * with the library pins every fold, the constants included. */
static uint32_t mirror_hash(const RNNoiseGrid *g)
{
    uint32_t h = 2166136261u;
    const char *tag = RNNOISE_FEATURE_VERSION;
    h = fnv_u32(h, (uint32_t)g->sample_rate);
    h = fnv_u32(h, (uint32_t)g->n_fft);
    h = fnv_u32(h, (uint32_t)g->hop);
    h = fnv_u32(h, (uint32_t)g->n_bins);
    h = fnv_u32(h, (uint32_t)g->n_bands);
    h = fnv_u32(h, (uint32_t)g->min_bins_per_band);
    h = fnv_u32(h, (uint32_t)g->lookahead_frames);
    h = fnv_u32(h, (uint32_t)g->conv_delay);
    h = fnv_u32(h, (uint32_t)g->spec_max_hz);
    h = fnv_u32(h, (uint32_t)g->spec_bins);
    h = fnv_f64(h, g->erb_norm_tau_sec);
    h = fnv_f64(h, g->spec_norm_tau_sec);
    h = fnv_f32(h, g->erb_norm_alpha);
    h = fnv_f32(h, g->erb_norm_init_lo_db);
    h = fnv_f32(h, g->erb_norm_init_hi_db);
    h = fnv_f32(h, g->erb_norm_scale_db);
    h = fnv_f32(h, g->spec_norm_alpha);
    h = fnv_f32(h, g->spec_norm_init_lo);
    h = fnv_f32(h, g->spec_norm_init_hi);
    h = fnv_f32(h, g->spec_norm_eps);
    h = fnv_f32(h, g->stft_scale);
    h = fnv_f32(h, g->stft_unscale);
    h = fnv_u32(h, (uint32_t)RNNOISE_MODEL_IO_LAYOUT_VERSION);
    h = fnv_u32(h, (uint32_t)RNNOISE_PREPOST_LAYOUT_VERSION);
    h = fnv_u32(h, (uint32_t)RNNOISE_CARVE_VERSION);
    h = fnv_u32(h, (uint32_t)RNNOISE_MODEL_INPUT_FRAMES);
    while (*tag) {
        h ^= (uint32_t)(unsigned char)*tag++;
        h *= 16777619u;
    }
    return ac_memreq_abi_hash(h);
}

/* The descriptor `spec` would get. Sizing only tests that the matrices are present. */
static int descriptor_of(const RNNoiseGridSpec *spec, const float *fwd, const float *inv,
                         RNNoiseGrid *grid_out, RNNoiseMemReq *req)
{
    RNNoiseConfig cfg;
    if (rnnoise_grid_make(grid_out, spec) != RNNOISE_GRID_OK) return -1;
    if (rnnoise_config_defaults(&cfg, grid_out) != 0) return -1;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    return rnnoise_get_mem_size(&cfg, req);
}

/* Each field of the spec, moved to another valid value. */
#define N_SPEC_FIELDS 14
static void move_field(RNNoiseGridSpec *s, int field)
{
    switch (field) {
    case 0:  s->sample_rate = 32000; break;
    case 1:  s->n_fft = 256; break;
    case 2:  s->n_bands = 20; break;
    case 3:  s->min_bins_per_band = 3; break;
    case 4:  s->lookahead_frames = 0; break;
    case 5:  s->spec_max_hz = 3000; break;
    case 6:  s->erb_norm_tau_sec = 0.5; break;
    case 7:  s->erb_norm_init_lo_db = -21.0f; break;
    case 8:  s->erb_norm_init_hi_db = -46.0f; break;
    case 9:  s->erb_norm_scale_db = 35.0f; break;
    case 10: s->spec_norm_tau_sec = 2.0; break;
    case 11: s->spec_norm_init_lo = 0.05f; break;
    case 12: s->spec_norm_init_hi = 0.009f; break;
    default: s->spec_norm_eps = 1e-11f; break;
    }
}

/* rnnoise_init_ex() on the running pool: refused, with the carved block and the registered
 * buffers unchanged. */
static int gate_refuses(const Reg *r, const RNNoiseMemReq *expected, const Snap *snap)
{
    return rnnoise_init_ex(r->pool.pool, (size_t)r->pool.req.bytes, &r->pool.cfg, expected) == NULL &&
           snap_block_same(snap, r, 0u) && snap_buffers_same(snap, r);
}

static void pool_descriptor_gates(const RNNoiseGrid *grid, const float *fwd, const float *inv)
{
    static Reg live;
    static FeatWindow win;
    RNNoiseGridSpec spec;
    RNNoiseGrid other_grid;
    RNNoiseMemReq req, other, tweak;
    RNNoiseState *st;
    Snap snap;

    CHECK("desc.open", reg_open(&live, grid, fwd, inv, 1) == 0);
    req = live.pool.req;
    CHECK("desc.fields", req.descriptor_version == RNNOISE_MEM_REQ_VERSION &&
          req.layout_version == RNNOISE_MODEL_IO_LAYOUT_VERSION &&
          req.alignment == RNNOISE_ALIGNMENT && req.bytes > 0u);
    CHECK("desc.hash_mirror", req.build_flags_hash == mirror_hash(grid));

    /* every field of the spec moves the hash, and the hash stays the mirror's */
    for (int f = 0; f < N_SPEC_FIELDS; ++f) {
        int ok;
        spec = RNN_TEST_SHIPPED;
        move_field(&spec, f);
        ok = descriptor_of(&spec, fwd, inv, &other_grid, &other) == 0;
        CHECK("hash.variant_valid", ok);
        if (!ok) continue;
        CHECK("hash.field_moves_hash", other.build_flags_hash != req.build_flags_hash);
        CHECK("hash.variant_mirror", other.build_flags_hash == mirror_hash(&other_grid));
    }

    /* the gate, on a pool that holds a running state */
    feed(&live, 5);
    reg_fill(&live, 0.5f);
    snap_take(&snap, &live);

    spec = RNN_TEST_SHIPPED;
    spec.lookahead_frames = 0;                  /* same pool size, other hash */
    CHECK("gate.same_size_other_grid", descriptor_of(&spec, fwd, inv, &other_grid, &other) == 0 &&
          other.bytes == req.bytes && other.build_flags_hash != req.build_flags_hash);
    CHECK("gate.hash_only_refused", gate_refuses(&live, &other, &snap));
    spec = RNN_TEST_SHIPPED;
    spec.n_fft = 256;
    spec.n_bands = 18;
    spec.lookahead_frames = 0;                  /* other size and hash */
    CHECK("gate.other_size_grid", descriptor_of(&spec, fwd, inv, &other_grid, &other) == 0 &&
          other.bytes != req.bytes);
    CHECK("gate.other_grid_refused", gate_refuses(&live, &other, &snap));

    tweak = req; tweak.descriptor_version += 1u;
    CHECK("gate.descriptor_version_refused", gate_refuses(&live, &tweak, &snap));
    tweak = req; tweak.layout_version += 1u;
    CHECK("gate.layout_version_refused", gate_refuses(&live, &tweak, &snap));
    tweak = req; tweak.build_flags_hash ^= 1u;
    CHECK("gate.hash_bit_refused", gate_refuses(&live, &tweak, &snap));
    tweak = req; tweak.alignment = 2u * RNNOISE_ALIGNMENT;
    CHECK("gate.alignment_refused", gate_refuses(&live, &tweak, &snap));
    tweak = req; tweak.bytes += 16u;
    CHECK("gate.bytes_more_refused", gate_refuses(&live, &tweak, &snap));
    tweak = req; tweak.bytes -= 16u;
    CHECK("gate.bytes_fewer_refused", gate_refuses(&live, &tweak, &snap));
    snap_free(&snap);

    /* the exact descriptor re-initialises the pool, and the state then runs */
    st = rnnoise_init_ex(live.pool.pool, (size_t)req.bytes, &live.pool.cfg, &req);
    CHECK("gate.exact_accepted", st == live.pool.st);
    CHECK("gate.exact_fresh", st && rnnoise_state_generation(st) == 1u && reg_all_zero(&live) &&
          st->feat_count == 0);
    CHECK("gate.exact_runs", feed_frame(&live, 0, -1, 0, 0.0f, &win) == 0 &&
          feed_frame(&live, 1, -1, 0, 0.0f, &win) == 0 &&
          feed_frame(&live, 2, -1, 0, 0.0f, &win) == 1 &&
          all_finite(&win.erb[0][0], 3 * grid->n_bands));
    /* no descriptor: the same as rnnoise_init() */
    reg_fill(&live, 0.5f);
    st = rnnoise_init_ex(live.pool.pool, (size_t)req.bytes, &live.pool.cfg, NULL);
    CHECK("gate.null_expected", st == live.pool.st && reg_all_zero(&live) &&
          rnnoise_state_generation(st) == 1u);
    reg_close(&live);
}

/* A span wide enough to cancel in float (lo + (hi - lo) rounding to 0) keeps both configured ends of
 * the spec normaliser, from init and from reset, whichever end is the larger. */
static void norm_seed_gates(const float *fwd, const float *inv)
{
    for (int e = 0; e < 2; ++e) {
        const float lo = e ? 1e-8f : 1.0f, hi = e ? 1.0f : 1e-8f;
        RNNoiseGridSpec spec = RNN_TEST_SHIPPED;
        RNNoiseGrid grid;
        RnnPool pool;
        int ends_kept = 1;
        spec.spec_norm_init_lo = lo;
        spec.spec_norm_init_hi = hi;
        memset(&pool, 0, sizeof pool);
        CHECK("norm_seed.open", rnnoise_grid_make(&grid, &spec) == RNNOISE_GRID_OK &&
              rnnoise_config_defaults(&pool.cfg, &grid) == 0);
        pool.cfg.erb_fwd = fwd;
        pool.cfg.erb_inv = inv;
        CHECK("norm_seed.init", rnn_pool_open(&pool) == 0);
        if (!pool.st) { free(pool.pool); continue; }
        for (int pass = 0; pass < 2; ++pass) {
            const float *s = pool.st->spec_norm_state;
            ends_kept = ends_kept && s[0] == lo && s[grid.spec_bins - 1] == hi;
            for (int k = 0; k < grid.spec_bins; ++k)
                ends_kept = ends_kept && s[k] > 0.0f && s[k] >= (lo < hi ? lo : hi) &&
                            s[k] <= (lo < hi ? hi : lo);
            memset(pool.st->spec_norm_state, 0, sizeof(float) * (size_t)grid.spec_bins);
            rnnoise_reset(pool.st);
        }
        CHECK("norm_seed.ends", ends_kept);
        free(pool.pool);
    }
}

int main(int argc, char **argv) {
    RNNoiseGrid grid;
    RnnErb mats;
    RnnPool pool;
    RefState ref;
    static float re[RNN_TEST_N_BINS], im[RNN_TEST_N_BINS];
    static float got_erb[3][RNN_TEST_N_BANDS], want_erb[3][RNN_TEST_N_BANDS];
    static float got_spec[3][2][RNN_TEST_SPEC_BINS];
    static float want_spec[3][2][RNN_TEST_SPEC_BINS];
    int ok = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: %s ERB_FWD_FILE ERB_INV_FILE\n", argv[0]);
        return 2;
    }
    if (rnnoise_grid_make(&grid, &RNN_TEST_SHIPPED) != RNNOISE_GRID_OK) return 1;
    if (grid.n_bins != RNN_TEST_N_BINS || grid.n_bands != RNN_TEST_N_BANDS ||
        grid.spec_bins != RNN_TEST_SPEC_BINS) {
        fprintf(stderr, "FAIL: shipped grid differs from the test's array sizes\n");
        return 1;
    }
    if (rnn_erb_read_grid(argv[1], argv[2], &grid, &mats) != RNN_ERB_OK) {
        fprintf(stderr, "FAIL: cannot read matching ERB matrix files: %s %s\n", argv[1], argv[2]);
        return 1;
    }

    registered_state_gates(&grid, mats.fwd, mats.inv);
    registration_gates(&grid, mats.fwd, mats.inv);
    nonfinite_spectrum_gates(&grid, mats.fwd, mats.inv);
    atten_lim_gates(&grid, mats.fwd, mats.inv);
    pool_descriptor_gates(&grid, mats.fwd, mats.inv);
    norm_seed_gates(mats.fwd, mats.inv);
    printf("INFO: %d call-contract checks, %d failed\n", g_checks, g_fail);
    if (g_fail != 0) ok = 0;

    memset(&pool, 0, sizeof(pool));
    if (rnnoise_config_defaults(&pool.cfg, &grid) != 0) return 1;
    pool.cfg.erb_fwd = mats.fwd;
    pool.cfg.erb_inv = mats.inv;
    if (rnn_pool_open(&pool) != 0) return 1;

    ref_init(&ref, &grid);
    fill_stationary_spectrum(re, im);

    for (int t = 0; ok && t < TEST_FRAMES; ++t) {
        int got_ready = rnnoise_compute_features(
            pool.st, re, im, &got_erb[0][0], &got_spec[0][0][0]);
        int want_ready = ref_step(&ref, &grid, mats.fwd, re, im, want_erb, want_spec);
        if (got_ready != want_ready) {
            printf("FAIL: ready mismatch at frame %d: got=%d want=%d\n",
                   t, got_ready, want_ready);
            ok = 0;
            break;
        }
        if (got_ready) {
            for (int f = 0; f < 3; ++f) {
                for (int b = 0; b < RNN_TEST_N_BANDS; ++b) {
                    if (!close_enough(got_erb[f][b], want_erb[f][b],
                                      "erb", t, b)) ok = 0;
                }
                for (int c = 0; c < 2; ++c) {
                    for (int k = 0; k < RNN_TEST_SPEC_BINS; ++k) {
                        if (!close_enough(got_spec[f][c][k], want_spec[f][c][k],
                                          "complex", t, k)) ok = 0;
                    }
                }
            }
        }
        for (int k = 0; k < RNN_TEST_SPEC_BINS; ++k) {
            if (!close_enough(pool.st->spec_norm_state[k], ref.spec_norm[k],
                              "spec_norm_state", t, k)) ok = 0;
        }
        for (int b = 0; b < RNN_TEST_N_BANDS; ++b) {
            if (!close_enough(pool.st->erb_norm_state[b], ref.erb_norm[b],
                              "erb_norm_state", t, b)) ok = 0;
        }
    }

    if (ok) {
        float erb_abs_max = fabsf(got_erb[2][0]);
        float complex_abs_sum = 0.0f;
        for (int b = 1; b < RNN_TEST_N_BANDS; ++b) {
            if (fabsf(got_erb[2][b]) > erb_abs_max) {
                erb_abs_max = fabsf(got_erb[2][b]);
            }
        }
        for (int c = 0; c < 2; ++c) {
            for (int k = 0; k < RNN_TEST_SPEC_BINS; ++k) {
                complex_abs_sum += fabsf(got_spec[2][c][k]);
            }
        }
        if (!(erb_abs_max < 2e-5f)) {
            printf("FAIL: stationary ERB mean norm did not converge to zero: %.9g\n",
                   (double)erb_abs_max);
            ok = 0;
        }
        if (!(complex_abs_sum / (2.0f * RNN_TEST_SPEC_BINS) > 0.01f)) {
            printf("FAIL: stationary complex feature collapsed\n");
            ok = 0;
        }
    }

    if (ok) {
        printf("PASS: %s matches independent reference; stationary ERB mean "
               "converges to zero while complex features remain observable after %d frames\n",
               RNNOISE_FEATURE_VERSION, TEST_FRAMES);
    }
    free(pool.pool);
    rnn_erb_free(&mats);
    return ok ? 0 : 1;
}
