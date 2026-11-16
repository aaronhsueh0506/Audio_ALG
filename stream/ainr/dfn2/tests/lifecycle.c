/* The per-call lifecycle of dfn2_stream on a freq_only context, N = 1 and N > 1, L = 0..2, D = 0..4: the call gates (pre twice, post
 * without pre, setters while pending, wrong entry family, reset with a call open), a refused run per tensor (every frame of the call
 * becomes the identity, the generation moves once, the next call equals a stream that never saw the fault), the generation table
 * (commit, NOT_STARTED, FAILED, an odd status, a BYPASS hop, reset) and the warm-up hops. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "erb_fixture.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Row { int sr, fft, n_erb, L, D, N; } Row;

static const Row ROWS[] = {
    { 16000, 256, 24, 0, 0, 1 }, { 16000, 256, 24, 0, 1, 1 }, { 16000, 256, 24, 1, 0, 1 }, { 16000, 256, 24, 2, 3, 1 },
    { 16000, 256, 24, 0, 4, 1 }, { 16000, 256, 24, 0, 0, 4 }, { 16000, 256, 24, 0, 2, 3 }, { 48000, 1024, 32, 1, 1, 1 },
};

typedef struct Ctx {
    DFN2Grid g;
    DFN2Stream *s;
    void *pool;
    float *x;   /* est re, im, app re, app im, out re, out im: N * n_bins each */
    size_t plane;
} Ctx;

/* The heads of a call: a mask from the first window frame of each head, coefficients from the second. kind: 0 write all, 1 write nothing,
 * 2/3 NaN / huge erb_mask in the last frame, 4/5 NaN / huge coefs in the last frame, 6 NaN alpha in the last frame (full). */
static void heads(const DFN2HopIO *io, const DFN2Grid *g, int kind) {
    const int n = g->frames_per_call, ne = g->n_erb;
    size_t i, k;
    if (kind == 1) return;
    for (i = 0; i < (size_t)n; ++i) {
        const float w = io->inputs.erb_window[i * (size_t)ne] * 0.01f;
        for (k = 0; k < (size_t)ne; ++k) io->outputs.erb_mask[i * (size_t)ne + k] = 0.5f + 0.25f * sinf(w + 0.1f * (float)k);
#ifndef DFN2_MASK_ONLY
        {
            const size_t per = io->outputs.coefs_elements / (size_t)n;
            for (k = 0; k < per; ++k) io->outputs.coefs[i * per + k] = 0.02f * cosf(w + 0.3f * (float)k);
            io->outputs.alpha[i] = 0.5f;
        }
#endif
    }
    if (kind == 2) io->outputs.erb_mask[(size_t)(n - 1) * (size_t)ne + 1] = (float)NAN;
    if (kind == 3) io->outputs.erb_mask[(size_t)(n - 1) * (size_t)ne + 1] = 2.0e6f;
#ifndef DFN2_MASK_ONLY
    {
        const size_t per = io->outputs.coefs_elements / (size_t)n;
        if (kind == 4) io->outputs.coefs[(size_t)(n - 1) * per + 7] = (float)NAN;
        if (kind == 5) io->outputs.coefs[(size_t)(n - 1) * per + 7] = -3.0e6f;
        if (kind == 6) io->outputs.alpha[n - 1] = (float)NAN;
    }
#endif
}

static int open_ctx(Ctx *c, const Row *r, const float *fwd, const float *inv) {
    DFN2GridSpec spec;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = r->sr; spec.n_fft = r->fft; spec.n_erb = r->n_erb; spec.df_bins = 96;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
    if (r->D) return 1;
#else
    spec.df_order = 5; spec.df_lookahead = r->D;
#endif
    spec.mask_lookahead = r->L; spec.frames_per_call = r->N;
    if (dfn2_grid_make(&c->g, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid\n"); exit(2); }
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = c->g; cfg.freq_only = 1; cfg.erb_fwd = fwd; cfg.erb_inv = inv;
    if (dfn2_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&c->pool, req.alignment, (size_t)req.bytes) != 0) exit(2);
    c->s = dfn2_stream_init_ex(&cfg, c->pool, (size_t)req.bytes, &req);
    if (!c->s) { fprintf(stderr, "init\n"); exit(2); }
    c->plane = (size_t)c->g.n_bins * (size_t)r->N;
    c->x = (float *)calloc(c->plane * 6u, sizeof(float));
    return 0;
}

static void close_ctx(Ctx *c) { free(c->x); free(c->pool); }

static float *ew(Ctx *c, int k) { return c->x + c->plane * (size_t)k; }   /* 0 est re, 1 est im, 2 app re, 3 app im, 4 out re, 5 out im */

static void fill(Ctx *c, int hop) {
    size_t i;
    for (i = 0; i < c->plane; ++i) {
        ew(c, 0)[i] = 0.4f * sinf(0.013f * (float)((size_t)hop * c->plane + i)) + 0.05f;
        ew(c, 1)[i] = 0.3f * cosf(0.017f * (float)((size_t)hop * c->plane + i));
        ew(c, 2)[i] = hop % 2 ? ew(c, 0)[i] : 0.5f * ew(c, 0)[i];
        ew(c, 3)[i] = hop % 2 ? ew(c, 1)[i] : 0.5f * ew(c, 1)[i];
    }
}

static int pre(Ctx *c, DFN2HopIO *io) {
    return c->g.frames_per_call == 1 ? dfn2_stream_pre_process_freq_dual(c->s, ew(c, 0), ew(c, 1), ew(c, 2), ew(c, 3), io)
                                     : dfn2_stream_pre_process_packet_freq_dual(c->s, ew(c, 0), ew(c, 1), ew(c, 2), ew(c, 3), io);
}
static int post(Ctx *c, int rc, int *valid) {
    return c->g.frames_per_call == 1 ? dfn2_stream_post_process_freq(c->s, rc, ew(c, 4), ew(c, 5), valid)
                                     : dfn2_stream_post_process_packet_freq(c->s, rc, ew(c, 4), ew(c, 5), valid);
}

/* One hop: pre, heads of `kind`, post with `status`. Returns the valid count, -9 on a contract error. */
static int hop(Ctx *c, int h, int kind, int status, DFN2HopIO *io_out) {
    DFN2HopIO io;
    int valid = -1;
    fill(c, h);
    if (pre(c, &io) != 1) return -9;
    if (io.action == STREAM_WORK_SUBMIT) heads(&io, &c->g, kind);
    if (post(c, status, &valid) != 0) return -9;
    if (io_out) *io_out = io;
    return valid;
}

static void gates(const Row *r, const float *fwd, const float *inv) {
    Ctx c;
    DFN2HopIO io;
    int valid;
    unsigned gen;
    if (open_ctx(&c, r, fwd, inv)) return;
    fill(&c, 0);
    CHECK(post(&c, STREAM_RUN_OK, &valid) == -1, "N=%d: post without pre accepted", r->N);
    CHECK(pre(&c, &io) == 1, "N=%d: pre", r->N);
    CHECK(pre(&c, &io) == -1, "N=%d: a second pre with the call pending", r->N);
    CHECK(dfn2_stream_set_atten_lim(c.s, 6.0f) == -1, "N=%d: set_atten_lim while pending", r->N);
    if (io.action == STREAM_WORK_SUBMIT) heads(&io, &c.g, 0);
    CHECK(post(&c, STREAM_RUN_OK, &valid) == 0, "N=%d: the pending call was lost by the refused second pre", r->N);
    CHECK(dfn2_stream_set_atten_lim(c.s, 6.0f) == 0 && dfn2_stream_set_atten_lim(c.s, 0.0f) == 0, "N=%d: set_atten_lim between calls", r->N);
    CHECK(dfn2_stream_set_atten_lim(c.s, (float)NAN) == -1, "N=%d: a NaN limit accepted", r->N);
    /* the other entry family is refused and moves nothing */
    gen = dfn2_stream_state_generation(c.s);
    if (r->N == 1) CHECK(dfn2_stream_pre_process_packet_freq_dual(c.s, ew(&c, 0), ew(&c, 1), ew(&c, 2), ew(&c, 3), &io) == -1, "packet pre on N=1");
    else CHECK(dfn2_stream_pre_process_freq_dual(c.s, ew(&c, 0), ew(&c, 1), ew(&c, 2), ew(&c, 3), &io) == -1, "hop pre on N=%d", r->N);
    CHECK(dfn2_stream_state_generation(c.s) == gen, "N=%d: a refused entry moved the generation", r->N);
    /* reset with a call open drops it */
    CHECK(pre(&c, &io) == 1, "N=%d: pre after the refused entry", r->N);
    dfn2_stream_reset(c.s);
    CHECK(dfn2_stream_state_generation(c.s) == gen + 1u, "N=%d: reset generation", r->N);
    CHECK(post(&c, STREAM_RUN_OK, &valid) == -1, "N=%d: post after a reset that dropped the call", r->N);
    CHECK(pre(&c, &io) == 1, "N=%d: pre after reset", r->N);
    close_ctx(&c);
}

/* A refused run: every frame of the call is the identity of the NOT_STARTED twin, the generation moves exactly once, and the next call
 * equals the twin's (the state is zeroed by the restart, not by the twin: the next hop is compared on output only). */
static void refusals(const Row *r, const float *fwd, const float *inv) {
    static const int kinds[] = { 1, 2, 3
#ifndef DFN2_MASK_ONLY
        , 4, 5, 6
#endif
    };
    size_t k;
    int warm, h;
    for (k = 0; k < sizeof kinds / sizeof kinds[0]; ++k) {
        Ctx a, b;
        if (open_ctx(&a, r, fwd, inv)) return;
        if (open_ctx(&b, r, fwd, inv)) return;
        warm = r->L + 2;
        for (h = 0; h < warm; ++h) CHECK(hop(&a, h, 0, STREAM_RUN_OK, NULL) >= 0 && hop(&b, h, 0, STREAM_RUN_OK, NULL) >= 0, "warm hop %d", h);
        {
            const unsigned ga = dfn2_stream_state_generation(a.s), gb = dfn2_stream_state_generation(b.s);
            int va, vb;
            va = hop(&a, warm, (int)kinds[k], STREAM_RUN_OK, NULL);
            vb = hop(&b, warm, 0, STREAM_RUN_NOT_STARTED, NULL);
            CHECK(va == vb && va >= 0, "L=%d D=%d N=%d kind %d: valid %d vs %d", r->L, r->D, r->N, kinds[k], va, vb);
            CHECK(dfn2_stream_state_generation(a.s) == ga + 1u, "L=%d N=%d kind %d: generation %u -> %u, want +1", r->L, r->N, kinds[k],
                  ga, dfn2_stream_state_generation(a.s));
            CHECK(dfn2_stream_state_generation(b.s) == gb, "NOT_STARTED moved the generation");
            CHECK(memcmp(ew(&a, 4), ew(&b, 4), a.plane * sizeof(float)) == 0 && memcmp(ew(&a, 5), ew(&b, 5), a.plane * sizeof(float)) == 0,
                  "L=%d D=%d N=%d kind %d: the refused call is not the identity", r->L, r->D, r->N, kinds[k]);
            /* the refused heads were never read: both streams produce the same call afterwards */
            va = hop(&a, warm + 1, 0, STREAM_RUN_OK, NULL);
            vb = hop(&b, warm + 1, 0, STREAM_RUN_OK, NULL);
            CHECK(va == vb, "L=%d N=%d kind %d: valid after %d vs %d", r->L, r->N, kinds[k], va, vb);
        }
        close_ctx(&a); close_ctx(&b);
    }
}

static void generations(const Row *r, const float *fwd, const float *inv) {
    static const int statuses[] = { STREAM_RUN_OK, STREAM_RUN_NOT_STARTED, STREAM_RUN_FAILED, 777 };
    Ctx c;
    size_t k;
    int h;
    unsigned gen;
    DFN2HopIO io;
    if (open_ctx(&c, r, fwd, inv)) return;
    /* a BYPASS hop (N == 1, the first L hops) ignores every status and moves nothing */
    for (h = 0; h < r->L && r->N == 1; ++h) {
        fill(&c, h);
        gen = dfn2_stream_state_generation(c.s);
        CHECK(pre(&c, &io) == 1 && io.action == STREAM_WORK_BYPASS && io.state_generation == gen, "L=%d: hop %d is not a BYPASS at the generation", r->L, h);
        CHECK(post(&c, STREAM_RUN_FAILED, NULL) == 0 && dfn2_stream_state_generation(c.s) == gen, "L=%d: a BYPASS hop moved the generation", r->L);
    }
    for (k = 0; k < sizeof statuses / sizeof statuses[0]; ++k) {
        const int want = statuses[k] == STREAM_RUN_OK || statuses[k] == STREAM_RUN_NOT_STARTED ? 0 : 1;
        gen = dfn2_stream_state_generation(c.s);
        CHECK(hop(&c, 40 + (int)k, 0, statuses[k], &io) >= -1, "hop");
        CHECK(dfn2_stream_state_generation(c.s) == gen + (unsigned)want, "L=%d N=%d status %d: generation %u -> %u, want +%d", r->L, r->N,
              statuses[k], gen, dfn2_stream_state_generation(c.s), want);
        CHECK(io.state_generation == gen, "L=%d: the tensors carry generation %u, want %u", r->L, io.state_generation, gen);
    }
    gen = dfn2_stream_state_generation(c.s);
    dfn2_stream_reset(c.s);
    CHECK(dfn2_stream_state_generation(c.s) == gen + 1u, "reset generation");
    close_ctx(&c);
}

/* N == 1: the first L hops have no job and no spectrum, hop L carries the job, the first spectrum comes out at hop L + D. */
static void warmup(const Row *r, const float *fwd, const float *inv) {
    Ctx c;
    int h;
    if (r->N != 1 || open_ctx(&c, r, fwd, inv)) return;
    for (h = 0; h < r->L + r->D + 3; ++h) {
        DFN2HopIO io;
        int valid = -1;
        size_t i;
        fill(&c, h);
        CHECK(pre(&c, &io) == 1, "pre hop %d", h);
        CHECK((io.action == STREAM_WORK_SUBMIT) == (h >= r->L), "L=%d: hop %d action %d", r->L, h, (int)io.action);
        if (io.action == STREAM_WORK_SUBMIT) heads(&io, &c.g, 0);
        CHECK(post(&c, STREAM_RUN_OK, &valid) == 0, "post hop %d", h);
        CHECK(valid == (h >= r->L + r->D ? 1 : 0), "L=%d D=%d: hop %d valid %d", r->L, r->D, h, valid);
        if (h < r->L + r->D) {
            int zero = 1;
            for (i = 0; i < c.plane; ++i) zero &= ew(&c, 4)[i] == 0.0f && ew(&c, 5)[i] == 0.0f;
            CHECK(zero, "L=%d D=%d: hop %d has output before the first spectrum", r->L, r->D, h);
        }
    }
    close_ctx(&c);
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof ROWS / sizeof ROWS[0]; ++i) {
        const Row *r = &ROWS[i];
        const int nb = r->fft / 2 + 1;
        float *fwd = (float *)malloc((size_t)nb * (size_t)r->n_erb * sizeof(float));
        float *inv = (float *)malloc((size_t)nb * (size_t)r->n_erb * sizeof(float));
        size_t k;
        for (k = 0; k < (size_t)nb * (size_t)r->n_erb; ++k) fwd[k] = (float)(k % 17) * 0.01f;
        erb_fixture_split_inv_band_major(inv, nb, r->n_erb);
        gates(r, fwd, inv);
        refusals(r, fwd, inv);
        generations(r, fwd, inv);
        warmup(r, fwd, inv);
        free(fwd); free(inv);
    }
    if (g_fail) { printf("lifecycle: %d check(s) FAILED\n", g_fail); return 1; }
    printf("lifecycle: %d grids, call gates, refused runs per tensor, generation table, warm-up -- PASS\n", (int)(sizeof ROWS / sizeof ROWS[0]));
    return 0;
}
