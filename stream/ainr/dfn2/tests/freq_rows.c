/* The spectrum entries of dfn2_stream (memory mode, freq_only context). Rows:
 *   latency   16 kHz / 256, 32 kHz / 512 and 48 kHz / 1024, every lookahead pair the build accepts: with the accelerator not
 *             started and with heads that are the identity on the unit ERB partition, the output of hop h is the application
 *             spectrum of hop h - (mask + df) bit for bit (zero and flagged invalid before), and hops before mask_lookahead
 *             have no job; the descriptor of one pair or grid is refused by the others;
 *   gate      every field of the stale-pool descriptor, a short, a misaligned and a NULL pool are refused with the pool
 *             bytes unchanged, and the same pool then initialises and runs, for a full and a freq_only context;
 *   refusal   a bin beyond the limit (or NaN) in any of the four spectra, or a second begin, is refused and leaves the
 *             stream as if it had never been offered: windows and outputs of the next 80 hops equal a stream that never saw it;
 *   warm-up   a device that zeroes its state on a generation change, one that starts zeroed, and one that starts at its
 *             first job give the same output; hops without a job never move the generation; a failed first job and a
 *             reset advance it once each, and after a reset the first mask_lookahead hops have no job again. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "erb_fixture.h"

typedef struct GridRow { int sr, fft, n_erb; } GridRow;
static const GridRow GRIDS[] = { { 16000, 256, 24 }, { 32000, 512, 28 }, { 48000, 1024, 32 } };
#define N_GRIDS 3

#ifdef DFN2_MASK_ONLY
#define MAX_DF 0
#else
#define MAX_DF 4
#endif
#define MAX_MASK 2

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Ctx {
    DFN2Grid grid;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    float *fwd, *inv;
    void *pool;
    DFN2Stream *s;
} Ctx;

static DFN2Grid make_grid(const GridRow *g, int mask_lookahead, int df_lookahead) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = g->sr;
    spec.n_fft = g->fft;
    spec.n_erb = g->n_erb;
    spec.df_bins = 96;
    spec.mask_lookahead = mask_lookahead;
#ifndef DFN2_MASK_ONLY
    spec.df_order = 5;
    spec.df_lookahead = df_lookahead;
#else
    (void)df_lookahead;
#endif
    spec.frames_per_call = 1;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) {
        fprintf(stderr, "grid_make: %d Hz / %d, pair %d/%d refused\n", g->sr, g->fft, mask_lookahead, df_lookahead);
        exit(2);
    }
    return grid;
}

/* Prepares cfg, req and the ERB tables; opens the stream when `open`. */
static int ctx_init(Ctx *c, const DFN2Grid *grid, int freq_only, int open) {
    size_t n = (size_t)grid->n_bins * (size_t)grid->n_erb, i;
    memset(c, 0, sizeof *c);
    c->grid = *grid;
    c->fwd = (float *)malloc(n * sizeof(float));
    c->inv = (float *)malloc(n * sizeof(float));
    if (!c->fwd || !c->inv) exit(2);
    for (i = 0; i < n; ++i) c->fwd[i] = 1.0f / (float)grid->n_erb;
    erb_fixture_unit_inv(c->inv, grid->n_bins, grid->n_erb);
    c->cfg.grid = *grid;
    c->cfg.erb_fwd = c->fwd;
    c->cfg.erb_inv = c->inv;
    c->cfg.freq_only = freq_only;
    if (dfn2_stream_get_mem_size(&c->cfg, &c->req) != 0) return -1;
    if (posix_memalign(&c->pool, (size_t)c->req.alignment, (size_t)c->req.bytes + 64u) != 0) exit(2);
    if (!open) return 0;
    c->s = dfn2_stream_init_ex(&c->cfg, c->pool, (size_t)c->req.bytes, &c->req);
    return c->s ? 0 : -1;
}

static void ctx_free(Ctx *c) {
    free(c->pool); free(c->fwd); free(c->inv);
    memset(c, 0, sizeof *c);
}

static float lcg_unit(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(int32_t)(*state >> 8) / (float)(1 << 23) - 1.0f;   /* [-1, 1) */
}

typedef struct Spec { float *est_re, *est_im, *app_re, *app_im; int n; } Spec;

static void spec_alloc(Spec *p, int n) {
    float *m = (float *)malloc((size_t)n * 4u * sizeof(float));
    if (!m) exit(2);
    p->est_re = m; p->est_im = m + n; p->app_re = m + 2 * n; p->app_im = m + 3 * n; p->n = n;
}

static void spec_fill(Spec *p, uint32_t *rng) {
    int k;
    for (k = 0; k < p->n; ++k) {
        p->est_re[k] = 40.0f * lcg_unit(rng); p->est_im[k] = 40.0f * lcg_unit(rng);
        p->app_re[k] = 40.0f * lcg_unit(rng); p->app_im[k] = 40.0f * lcg_unit(rng);
    }
}

static void write_state(DFN2PrepostOutputs *out) {
    size_t i;
    for (i = 0; i < out->gru_hidden_elements; ++i) out->gru_hidden_next[i] = 0.0f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->df_convp_window_elements; ++i) out->df_convp_window_next[i] = 0.0f;
#endif
}

/* Heads that are the identity on the unit partition: mask 1, deep filter 0. */
static void write_identity(DFN2PrepostOutputs *out) {
    size_t i;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 1.0f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.0f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.0f;
#endif
    write_state(out);
}

static uint32_t fnv32(uint32_t h, const float *x, size_t n) {
    const unsigned char *c = (const unsigned char *)x;
    size_t i;
    for (i = 0; i < n * sizeof(float); ++i) { h ^= c[i]; h *= 16777619u; }
    return h;
}

/* Heads that depend on the windows the model is shown, so a corrupt feature normaliser changes the output. */
static void write_window_heads(const DFN2HopIO *io, DFN2PrepostOutputs *out) {
    uint32_t h = fnv32(fnv32(2166136261u, io->inputs.erb_window, io->inputs.erb_window_elements),
                       io->inputs.spec_window, io->inputs.spec_window_elements);
    size_t i;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.1f + 0.8f * (float)((h >> (i % 24u)) & 255u) / 255.0f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.01f * (float)((h >> (i % 29u)) & 7u) - 0.03f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.25f;
#endif
    write_state(out);
}

typedef struct HopOut { float *re, *im; int valid; } HopOut;

/* The spectrum of one hop, offered and closed. mode 0: not started, 1: identity heads, 2: window heads. The hop's
 * DFN2HopIO is left in *io. dup_rc not NULL: a second begin is offered while the hop is open and its result stored. */
static void run_hop(DFN2Stream *s, const Spec *sp, int mode, HopOut *o, DFN2HopIO *io, int *pre_rc, int *post_rc, int *dup_rc) {
    int status = STREAM_RUN_NOT_STARTED;
    *pre_rc = dfn2_stream_pre_process_freq_dual(s, sp->est_re, sp->est_im, sp->app_re, sp->app_im, io);
    if (*pre_rc != 1) { *post_rc = 99; return; }
    if (dup_rc) {
        DFN2HopIO second;
        *dup_rc = dfn2_stream_pre_process_freq_dual(s, sp->est_re, sp->est_im, sp->app_re, sp->app_im, &second);
    }
    if (io->action != STREAM_WORK_SUBMIT) status = STREAM_RUN_FAILED;   /* a hop without a job ignores the status */
    if (io->action == STREAM_WORK_SUBMIT && mode == 1) { write_identity(&io->outputs); status = 0; }
    if (io->action == STREAM_WORK_SUBMIT && mode == 2) { write_window_heads(io, &io->outputs); status = 0; }
    *post_rc = dfn2_stream_post_process_freq(s, status, o->re, o->im, &o->valid);
}

/* ---- latency --------------------------------------------------------------------------------- */

static void row_latency(const GridRow *g, int m, int d, int mode, DFN2StreamMemReq *prev) {
    DFN2Grid grid = make_grid(g, m, d);
    Ctx c;
    Spec sp;
    HopOut o;
    float *hist_re, *hist_im;
    uint32_t rng = 12345u + (uint32_t)(m * 8 + d);
    const int lat = m + d, hops = lat + 12, nb = grid.n_bins;
    unsigned gen0;
    int h, k, bad = 0;

    if (ctx_init(&c, &grid, 1, 1) != 0) { CHECK(0, "%d Hz pair %d/%d: init", g->sr, m, d); return; }
    gen0 = dfn2_stream_state_generation(c.s);
    spec_alloc(&sp, nb);
    o.re = (float *)malloc((size_t)nb * 2u * sizeof(float)); o.im = o.re + nb;
    hist_re = (float *)malloc((size_t)nb * (size_t)hops * 2u * sizeof(float)); hist_im = hist_re + (size_t)nb * (size_t)hops;
    for (h = 0; h < hops; ++h) {
        DFN2HopIO io;
        int pre, post;
        spec_fill(&sp, &rng);
        memcpy(hist_re + (size_t)h * (size_t)nb, sp.app_re, (size_t)nb * sizeof(float));
        memcpy(hist_im + (size_t)h * (size_t)nb, sp.app_im, (size_t)nb * sizeof(float));
        run_hop(c.s, &sp, mode, &o, &io, &pre, &post, NULL);
        if (pre != 1 || post != 0 || (io.action == STREAM_WORK_SUBMIT) != (h >= m)) ++bad;
        if (io.state_generation != gen0 || dfn2_stream_state_generation(c.s) != gen0) ++bad;   /* not started and identity runs keep the state */
        if (h < lat) {
            for (k = 0; k < nb; ++k) if (o.re[k] != 0.0f || o.im[k] != 0.0f) { ++bad; break; }
            if (o.valid) ++bad;
        } else {
            if (!o.valid) ++bad;
            if (memcmp(o.re, hist_re + (size_t)(h - lat) * (size_t)nb, (size_t)nb * sizeof(float)) != 0 ||
                memcmp(o.im, hist_im + (size_t)(h - lat) * (size_t)nb, (size_t)nb * sizeof(float)) != 0) ++bad;
        }
    }
    CHECK(bad == 0, "%d Hz / %d pair %d/%d %s: %d hops differ from the application spectrum delayed %d hops", g->sr, g->fft,
          m, d, mode == 1 ? "identity heads" : "not started", bad, lat);
    /* the descriptor of the previous pair (or grid) does not initialise this one */
    if (mode == 0) {
        CHECK(!prev->bytes || dfn2_stream_init_ex(&c.cfg, c.pool, (size_t)c.req.bytes, prev) == NULL,
              "%d Hz / %d pair %d/%d: the previous descriptor is accepted", g->sr, g->fft, m, d);
        *prev = c.req;
    }
    free(o.re); free(hist_re); free(sp.est_re);
    ctx_free(&c);
}

/* ---- the stale-pool gate -------------------------------------------------------------------- */

static int refused_untouched(const Ctx *c, void *pool, size_t pool_bytes, const DFN2StreamMemReq *expected,
                             const unsigned char *snap) {
    DFN2Stream *s = dfn2_stream_init_ex(&c->cfg, pool, pool_bytes, expected);
    return s == NULL && memcmp(snap, c->pool, (size_t)c->req.bytes + 64u) == 0;
}

static void row_gate(int freq_only) {
    DFN2Grid grid = make_grid(&GRIDS[0], 0, 0);
    Ctx c;
    DFN2StreamMemReq bad;
    unsigned char *snap;
    size_t total;
    Spec sp;
    HopOut o;
    DFN2HopIO io;
    int pre, post;
    uint32_t rng = 7u;
    const char *kind = freq_only ? "freq_only" : "full";

    if (ctx_init(&c, &grid, freq_only, 0) != 0) { CHECK(0, "gate %s: sizing", kind); return; }
    total = (size_t)c.req.bytes + 64u;
    snap = (unsigned char *)malloc(total);
    if (!snap) exit(2);
    memset(c.pool, 0xA5, total);
    memcpy(snap, c.pool, total);

#define GATE(label, field_edit) do { bad = c.req; field_edit; \
        CHECK(refused_untouched(&c, c.pool, (size_t)c.req.bytes, &bad, snap), "gate %s: %s accepted or pool written", kind, label); } while (0)
    GATE("descriptor_version + 1", bad.descriptor_version += 1u);
    GATE("layout_version + 1", bad.layout_version += 1u);
    GATE("reserved0 != 0", bad.reserved0 = 1u);
    GATE("build_flags_hash ^ 1", bad.build_flags_hash ^= 1u);
    GATE("alignment 0", bad.alignment = 0u);
    GATE("alignment 32", bad.alignment = 32u);
    GATE("reserved != 0", bad.reserved = 1u);
    GATE("bytes - 16", bad.bytes -= 16u);
    GATE("bytes + 16", bad.bytes += 16u);
#undef GATE
    CHECK(refused_untouched(&c, c.pool, (size_t)c.req.bytes - 16u, &c.req, snap), "gate %s: a pool 16 bytes short is accepted", kind);
    CHECK(refused_untouched(&c, (unsigned char *)c.pool + 4, (size_t)c.req.bytes, &c.req, snap), "gate %s: a misaligned pool is accepted", kind);
    CHECK(refused_untouched(&c, NULL, (size_t)c.req.bytes, &c.req, snap), "gate %s: a NULL pool is accepted", kind);

    /* the same pool then initialises and runs */
    c.s = dfn2_stream_init_ex(&c.cfg, c.pool, (size_t)c.req.bytes, &c.req);
    CHECK(c.s != NULL, "gate %s: the pool does not initialise after the refusals", kind);
    if (c.s) {
        spec_alloc(&sp, grid.n_bins);
        o.re = (float *)malloc((size_t)grid.n_bins * 2u * sizeof(float)); o.im = o.re + grid.n_bins;
        spec_fill(&sp, &rng);
        run_hop(c.s, &sp, 1, &o, &io, &pre, &post, NULL);
        CHECK(pre == 1 && post == 0 && io.action == STREAM_WORK_SUBMIT, "gate %s: the first hop after the refusals", kind);
        free(o.re); free(sp.est_re);
    }
    free(snap);
    ctx_free(&c);
}

/* ---- refusals leave no trace ------------------------------------------------------------------ */

static void row_refusal(void) {
    static const float BAD[2] = { 1.0e21f, (float)NAN };
    DFN2Grid grid = make_grid(&GRIDS[0], 1, 1);
    const int nb = grid.n_bins, after = 80, warm = 30;
    int which, v, h;

    for (which = 0; which < 6; ++which)           /* 0..3: a bad bin in one of the four arrays, 4 and 5: a second begin */
        for (v = 0; v < (which < 4 ? 2 : 1); ++v) {
            /* a second begin is offered on hop 0 (no job: no frame is open, only the stream knows a hop is) or on hop `warm` (a frame is open) */
            const int at = which == 4 ? 0 : warm;
            Ctx a, b;
            Spec sp;
            HopOut oa, ob;
            DFN2HopIO ia, ib;
            uint32_t rng = 99u;
            int pre, post, dup, bad = 0;
            if (ctx_init(&a, &grid, 1, 1) != 0 || ctx_init(&b, &grid, 1, 1) != 0) { CHECK(0, "refusal: init"); return; }
            spec_alloc(&sp, nb);
            oa.re = (float *)malloc((size_t)nb * 4u * sizeof(float)); oa.im = oa.re + nb;
            ob.re = oa.re + 2 * nb; ob.im = ob.re + nb;
            for (h = 0; h < warm + after; ++h) {
                spec_fill(&sp, &rng);
                if (h == at && which < 4) {
                    DFN2HopIO junk;
                    float *arr[4];
                    float save;
                    unsigned gen = dfn2_stream_state_generation(a.s);
                    arr[0] = sp.est_re; arr[1] = sp.est_im; arr[2] = sp.app_re; arr[3] = sp.app_im;
                    save = arr[which][nb / 2];
                    arr[which][nb / 2] = BAD[v];
                    if (dfn2_stream_pre_process_freq_dual(a.s, sp.est_re, sp.est_im, sp.app_re, sp.app_im, &junk) != -1) ++bad;
                    arr[which][nb / 2] = save;
                    if (dfn2_stream_state_generation(a.s) != gen) ++bad;
                }
                dup = 0;
                run_hop(a.s, &sp, 2, &oa, &ia, &pre, &post, h == at && which >= 4 ? &dup : NULL);
                if (pre != 1 || post != 0 || (h == at && which >= 4 && dup != -1)) ++bad;
                run_hop(b.s, &sp, 2, &ob, &ib, &pre, &post, NULL);
                if (pre != 1 || post != 0) ++bad;
                if (ia.action != ib.action || ia.state_generation != ib.state_generation || oa.valid != ob.valid) ++bad;
                if (ia.action == STREAM_WORK_SUBMIT &&
                    (ia.inputs.erb_window_elements != ib.inputs.erb_window_elements ||
                     ia.inputs.spec_window_elements != ib.inputs.spec_window_elements ||
                     memcmp(ia.inputs.erb_window, ib.inputs.erb_window, ia.inputs.erb_window_elements * sizeof(float)) != 0 ||
                     memcmp(ia.inputs.spec_window, ib.inputs.spec_window, ia.inputs.spec_window_elements * sizeof(float)) != 0))
                    ++bad;
                if (memcmp(oa.re, ob.re, (size_t)nb * sizeof(float)) != 0 || memcmp(oa.im, ob.im, (size_t)nb * sizeof(float)) != 0) ++bad;
            }
            CHECK(bad == 0, "refusal %s %d: %d differences from a stream that never saw it",
                  which < 4 ? "bad bin in array" : "second begin at hop index", which < 4 ? which : at, bad);
            free(oa.re); free(sp.est_re);
            ctx_free(&a); ctx_free(&b);
        }
}

/* ---- warm-up, generation, reset ---------------------------------------------------------------- */

typedef struct Device {
    int mode;            /* 0: zeroes on a generation change, 1: zeroed at start only, 2: starts at its first job */
    int state, started, calls;
    unsigned seen;
} Device;

static void device_run(Device *dv, DFN2HopIO *io, int fail) {
    DFN2PrepostOutputs *out = &io->outputs;
    size_t i;
    if (dv->mode == 2 && !dv->started) { dv->state = 0; dv->started = 1; dv->seen = io->state_generation; }
    if (dv->mode == 0 && io->state_generation != dv->seen) { dv->state = 0; dv->seen = io->state_generation; }
    ++dv->calls;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.1f + 0.001f * (float)((dv->state + (int)i) % 97);
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.002f * (float)((dv->state + (int)i) % 13);
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.2f;
#endif
    write_state(out);
    dv->state = fail ? 1000 : dv->state + 1;
}

/* `hops` hops through one device; fail_first: the first job reports a failure; reset_at >= 0: a reset before that hop.
 * out receives hops * n_bins re values and as many im values. Returns the number of contract violations. */
static int drive(Ctx *c, Device *dv, int hops, int fail_first, int reset_at, float *out_re, float *out_im, long long *gen_moves) {
    const int nb = c->grid.n_bins, m = c->grid.mask_lookahead;
    int h, bad = 0, jobs = 0, since_reset = 0;
    uint32_t rng = 4242u;
    Spec sp;
    spec_alloc(&sp, nb);
    for (h = 0; h < hops; ++h) {
        DFN2HopIO io;
        HopOut o;
        unsigned g0, g1;
        int status = STREAM_RUN_FAILED, pre, post;   /* a hop without a job ignores the status */
        if (h == reset_at) {
            unsigned before = dfn2_stream_state_generation(c->s);
            dfn2_stream_reset(c->s);
            if (dfn2_stream_state_generation(c->s) != before + 1u) ++bad;
            since_reset = 0; rng = 4242u;
            ++*gen_moves;
        }
        spec_fill(&sp, &rng);
        g0 = dfn2_stream_state_generation(c->s);
        pre = dfn2_stream_pre_process_freq_dual(c->s, sp.est_re, sp.est_im, sp.app_re, sp.app_im, &io);
        if (pre != 1) { ++bad; continue; }
        if ((io.action == STREAM_WORK_SUBMIT) != (since_reset >= m) || io.state_generation != g0) ++bad;
        if (io.action == STREAM_WORK_SUBMIT) {
            int fail = fail_first && jobs == 0;
            device_run(dv, &io, fail);
            status = fail ? STREAM_RUN_FAILED : 0;
            ++jobs;
        } else if (io.state_generation != g0) {
            ++bad;
        }
        o.re = out_re + (size_t)h * (size_t)nb; o.im = out_im + (size_t)h * (size_t)nb;
        post = dfn2_stream_post_process_freq(c->s, status, o.re, o.im, &o.valid);
        if (post != 0) ++bad;
        g1 = dfn2_stream_state_generation(c->s);
        if (io.action != STREAM_WORK_SUBMIT && g1 != g0) ++bad;                  /* a hop without a job never moves the generation */
        if (io.action == STREAM_WORK_SUBMIT && g1 != g0 + (status != 0 ? 1u : 0u)) ++bad;
        if (g1 != g0) ++*gen_moves;
        ++since_reset;
    }
    free(sp.est_re);
    return bad;
}

static void row_warmup(int m, int d) {
    DFN2Grid grid = make_grid(&GRIDS[0], m, d);
    const int hops = m + 40, nb = grid.n_bins;
    float *r[3], *im[3], *rr, *ri;
    int v, bad;
    Ctx c;
    Device dv;
    long long moves;
    const size_t cells = (size_t)(2 * hops) * (size_t)nb;

    for (v = 0; v < 3; ++v) { r[v] = (float *)malloc(cells * sizeof(float)); im[v] = r[v] + (size_t)hops * (size_t)nb; }
    rr = (float *)malloc(cells * sizeof(float)); ri = rr + (size_t)hops * (size_t)nb;
    for (v = 0; v < 3; ++v) {
        if (ctx_init(&c, &grid, 1, 1) != 0) { CHECK(0, "warm-up: init"); return; }
        memset(&dv, 0, sizeof dv);
        dv.mode = v; dv.seen = dfn2_stream_state_generation(c.s);
        moves = 0;
        bad = drive(&c, &dv, hops, 0, -1, r[v], im[v], &moves);
        CHECK(bad == 0 && moves == 0, "warm-up mask %d, device %d: %d violations, generation moved %lld times", m, v, bad, moves);
        CHECK(dv.calls == hops - m, "warm-up mask %d, device %d: %d model runs, want %d", m, v, dv.calls, hops - m);
        CHECK(dfn2_stream_state_generation(c.s) == 1u, "warm-up mask %d, device %d: generation %u", m, v, dfn2_stream_state_generation(c.s));
        ctx_free(&c);
    }
    CHECK(memcmp(r[0], r[1], cells * sizeof(float)) == 0 && memcmp(r[0], r[2], cells * sizeof(float)) == 0,
          "warm-up mask %d: the three devices give different audio", m);

    /* a failed first job advances the generation once; the device that follows the generation then restarts from zero */
    if (ctx_init(&c, &grid, 1, 1) != 0) { CHECK(0, "warm-up: init"); return; }
    memset(&dv, 0, sizeof dv);
    dv.seen = dfn2_stream_state_generation(c.s);
    moves = 0;
    bad = drive(&c, &dv, hops, 1, -1, rr, ri, &moves);
    CHECK(bad == 0 && moves == 1 && dfn2_stream_state_generation(c.s) == 2u,
          "warm-up mask %d, failed first job: %d violations, %lld generation moves, generation %u", m, bad, moves, dfn2_stream_state_generation(c.s));
    ctx_free(&c);

    /* a reset advances it once, restarts the hops without a job, and the run after it equals a fresh stream's */
    if (ctx_init(&c, &grid, 1, 1) != 0) { CHECK(0, "warm-up: init"); return; }
    memset(&dv, 0, sizeof dv);
    dv.seen = dfn2_stream_state_generation(c.s);
    {
        float *two = (float *)malloc(cells * 2u * sizeof(float));
        float *two_im = two + (size_t)(2 * hops) * (size_t)nb;
        moves = 0;
        bad = drive(&c, &dv, 2 * hops, 0, hops, two, two_im, &moves);
        CHECK(bad == 0 && moves == 1 && dfn2_stream_state_generation(c.s) == 2u && dv.calls == 2 * (hops - m),
              "warm-up mask %d, reset: %d violations, %lld generation moves, %d model runs", m, bad, moves, dv.calls);
        CHECK(memcmp(two + (size_t)hops * (size_t)nb, r[0], (size_t)hops * (size_t)nb * sizeof(float)) == 0 &&
              memcmp(two_im + (size_t)hops * (size_t)nb, im[0], (size_t)hops * (size_t)nb * sizeof(float)) == 0,
              "warm-up mask %d: the run after a reset differs from a fresh stream's", m);
        free(two);
    }
    ctx_free(&c);
    for (v = 0; v < 3; ++v) free(r[v]);
    free(rr);
}

int main(void) {
    DFN2StreamMemReq prev;
    int gi, m, d, mode;

    memset(&prev, 0, sizeof prev);
    for (gi = 0; gi < N_GRIDS; ++gi)
        for (m = 0; m <= MAX_MASK; ++m)
            for (d = 0; d <= MAX_DF; ++d)
                for (mode = 0; mode < 2; ++mode) row_latency(&GRIDS[gi], m, d, mode, &prev);
    row_gate(0);
    row_gate(1);
    row_refusal();
    for (m = 0; m <= MAX_MASK; ++m) row_warmup(m, MAX_DF ? 1 : 0);
    printf("freq_rows: %d grids x %d pairs, gate, refusals, warm-up -- %s\n", N_GRIDS, (MAX_MASK + 1) * (MAX_DF + 1),
           g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
