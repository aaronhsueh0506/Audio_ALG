/* The per-call lifecycle of deepvqe_stream (the library keeps no per-call latch): the open-call guards, the failure table
 * (generation effect and output of every run status), the NaN-filled head before every run, the range refusal of the
 * _freq entries and the all-N-frames-or-none output of a call whose enhanced spectrum overflows in a later frame.
 * Per grid row (N = 1 and a batched N), on a full stream (PCM and _freq entries) and on a freq_only stream:
 *   - pre twice, post without pre, post twice and a reset with a call open are -1 and change nothing;
 *   - a clean run commits: generation unchanged; STREAM_RUN_NOT_STARTED leaves the state and the generation alone and
 *     gives N zero frames; any other status, a refused validate_run and a non-finite enhanced bin each restart the
 *     registered state and advance the generation exactly once, and give N zero frames;
 *   - the frames of a committed call equal hand-driven CCM calls in slot order over the raw mic spectra passed in;
 *   - an overflow in frame k > 0 of a PCM call gives the PCM of a twin whose call failed (no frame < k reaches the ISTFT);
 *   - the head is NaN before every run, whatever the previous run wrote;
 *   - a NaN, +Inf or -Inf bin in any of the four _freq planes (first and last position) is DEEPVQE_STREAM_ERR_RANGE with
 *     the pool, the generation and the io untouched, and the stream goes on equal to a twin that never saw the call. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepvqe_stream.h"
#include "stream_run.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

#define ELEMS 6

typedef struct Row { int sr, fft, depth, n; } Row;
static const Row ROWS[] = { { 16000, 256, 63, 1 }, { 16000, 256, 63, 8 }, { 48000, 1024, 47, 2 } };

#define GUARD 256u
#define CANARY 0xC3

typedef struct Rig {
    DeepVqeStream *s;
    unsigned char *raw;   /* the pool between two canary zones */
    void *pool;
    size_t pool_bytes;
    DeepVqeGrid g;
    int freq_only;
    float *reg[DEEPVQE_STATE_COUNT];
    size_t plane;
} Rig;

static uint32_t lcg_next(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float lcg_f(uint32_t *s) { return (float)(lcg_next(s) >> 9) / 4194304.0f - 1.0f; }

static int rig_make(Rig *r, const Row *row, int freq_only) {
    DeepVqeGridSpec spec;
    DeepVqeStreamConfig cfg;
    DeepVqeStreamMemReq req;
    int id;
    memset(r, 0, sizeof *r);
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = row->sr; spec.n_fft = row->fft; spec.delay_depth = row->depth; spec.frames_per_call = row->n;
    if (deepvqe_grid_make(&r->g, &spec) != DEEPVQE_GRID_OK) return -1;
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = r->g;
    cfg.freq_only = freq_only;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        r->reg[id] = (float *)calloc(ELEMS, sizeof(float));
        if (!r->reg[id]) return -1;
        cfg.state[id] = r->reg[id];
        cfg.state_elements[id] = ELEMS;
    }
    if (deepvqe_stream_get_mem_size(&cfg, &req) != 0) return -1;
    if (posix_memalign((void **)&r->raw, 16, (size_t)req.bytes + 2u * GUARD) != 0) return -1;
    memset(r->raw, CANARY, (size_t)req.bytes + 2u * GUARD);
    r->pool = r->raw + GUARD;
    r->pool_bytes = (size_t)req.bytes;
    r->s = deepvqe_stream_init_ex(&cfg, r->pool, r->pool_bytes, &req);
    r->freq_only = freq_only;
    r->plane = (size_t)r->g.n_bins * (size_t)row->n;
    return r->s ? 0 : -1;
}

static void rig_free(Rig *r) {
    int id;
    size_t k;
    int intact = 1;
    for (k = 0; k < GUARD; ++k) intact &= r->raw[k] == CANARY && r->raw[GUARD + r->pool_bytes + k] == CANARY;
    CHECK(intact, "a stream wrote outside its pool");
    free(r->raw);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(r->reg[id]);
}

static int state_is_zero(const Rig *r) {
    int id;
    size_t k;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        for (k = 0; k < ELEMS; ++k) if (r->reg[id][k] != 0.0f) return 0;
    return 1;
}

/* Device kinds: 0 clean, 1 a NaN tap, 2 a NaN in one registered buffer, 3 an overflowing enhanced bin in frame `frame`,
 * 4 nothing written, 5 as 3 with the overflow only in the imaginary plane. A clean run leaves a non-zero registered state.
 * The overflow is a taps value of 3e38 on PCM input and a mic bin of 3e38 under a tap of 2 on _freq input (make_call); kind 5
 * has the _freq input only: the mic bin is imaginary, so the real plane stays finite. */
enum { DEV_OK, DEV_NAN_TAP, DEV_NAN_STATE, DEV_OVERFLOW, DEV_UNWRITTEN, DEV_OVERFLOW_IM };

static void dev(const Rig *r, DeepVqeHopIO *io, int kind, int frame, int pcm) {
    const size_t taps = io->outputs.taps_elements;
    const size_t per_bin = (size_t)DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2u;
    size_t i, k;
    int id;
    if (kind == DEV_UNWRITTEN) return;
    for (i = 0; i < taps; ++i) io->outputs.taps[i] = 0.05f * (float)((long)(i * 13u % 19u) - 9) / 9.0f + (i % per_bin == 8u ? 0.5f : 0.0f);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        for (k = 0; k < io->outputs.state_elements[id]; ++k) io->outputs.state_out[id][k] = 0.25f + 0.01f * (float)id;
    if (kind == DEV_NAN_TAP) io->outputs.taps[taps / 2u] = NAN;
    if (kind == DEV_NAN_STATE) io->outputs.state_out[DEEPVQE_STATE_H_GRU][1] = NAN;
    if (kind == DEV_OVERFLOW || kind == DEV_OVERFLOW_IM) {
        const size_t base = ((size_t)frame * (size_t)r->g.n_bins + 3u) * per_bin;
        for (i = 0; i < 9u; ++i) io->outputs.taps[base + 2u * i] = pcm ? 3.0e38f : 2.0f;
    }
}

typedef struct Call { float *mr, *mi, *fr, *fi; int16_t *mpcm, *fpcm; } Call;

/* The inputs of call `c`: spectra for the _freq entries, PCM for the packet entries. */
static void make_call(const Rig *r, Call *c, int idx, int huge_frame) {
    uint32_t seed = 2463534242u + (uint32_t)idx * 977u;
    const size_t hop = (size_t)r->g.hop * (size_t)(r->plane / (size_t)r->g.n_bins);
    size_t k;
    for (k = 0; k < r->plane; ++k) {
        c->mr[k] = 0.3f * lcg_f(&seed); c->mi[k] = 0.3f * lcg_f(&seed);
        c->fr[k] = 0.3f * lcg_f(&seed); c->fi[k] = 0.3f * lcg_f(&seed);
    }
    if (huge_frame >= 0) c->mr[(size_t)huge_frame * (size_t)r->g.n_bins + 3u] = 3.0e38f;
    for (k = 0; c->mpcm && k < hop; ++k) {
        c->mpcm[k] = (int16_t)(6000.0f * lcg_f(&seed));
        c->fpcm[k] = (int16_t)(1500.0f * lcg_f(&seed));
    }
}

static int pre(Rig *r, const Call *c, int pcm, DeepVqeHopIO *io) {
    return pcm ? deepvqe_stream_pre_process_packet(r->s, c->mpcm, c->fpcm, io)
               : deepvqe_stream_pre_process_packet_freq(r->s, c->mr, c->mi, c->fr, c->fi, io);
}

typedef struct Out { float *re, *im; int16_t *pcm; } Out;

static int post(Rig *r, int pcm, int status, Out *o) {
    return pcm ? deepvqe_stream_post_process_packet(r->s, status, o->pcm)
               : deepvqe_stream_post_process_packet_freq(r->s, status, o->re, o->im);
}

static int all_zero_f(const float *x, size_t n) { size_t k; for (k = 0; k < n; ++k) if (x[k] != 0.0f) return 0; return 1; }

static void guards(const Row *row, int freq_only, int pcm, const char *label) {
    Rig r;
    Call c;
    Out o;
    DeepVqeHopIO io, io2;
    const size_t hop = (size_t)row->fft / 2u * (size_t)row->n, nb = (size_t)row->fft / 2u + 1u;
    float *buf = (float *)calloc(nb * (size_t)row->n * 6u, sizeof(float));
    int16_t *pc = (int16_t *)calloc(hop * 3u, sizeof(int16_t));
    uint32_t gen;
    if (!buf || !pc || rig_make(&r, row, freq_only) != 0) exit(2);
    c.mr = buf; c.mi = buf + r.plane; c.fr = buf + 2u * r.plane; c.fi = buf + 3u * r.plane;
    c.mpcm = pc; c.fpcm = pc + hop;
    o.re = buf + 4u * r.plane; o.im = buf + 5u * r.plane; o.pcm = pc + 2u * hop;
    make_call(&r, &c, 1, -1);

    CHECK(post(&r, pcm, STREAM_RUN_OK, &o) == -1, "%s: post without pre", label);
    CHECK(pre(&r, &c, pcm, &io) == 1, "%s: pre", label);
    gen = deepvqe_stream_state_generation(r.s);
    CHECK(io.state_generation == gen, "%s: io generation", label);
    CHECK(pre(&r, &c, pcm, &io2) == -1, "%s: pre twice", label);
    CHECK(deepvqe_stream_state_generation(r.s) == gen, "%s: a refused pre moved the generation", label);
    dev(&r, &io, DEV_OK, 0, pcm);
    CHECK(post(&r, pcm, STREAM_RUN_OK, &o) == 1, "%s: post", label);
    CHECK(post(&r, pcm, STREAM_RUN_OK, &o) == -1, "%s: post twice", label);
    CHECK(deepvqe_stream_state_generation(r.s) == gen, "%s: a clean run moved the generation", label);
    CHECK(!state_is_zero(&r), "%s: a clean run left no state", label);

    CHECK(pre(&r, &c, pcm, &io) == 1, "%s: pre before the reset", label);
    deepvqe_stream_reset(r.s);
    CHECK(deepvqe_stream_state_generation(r.s) == gen + 1u, "%s: reset generation", label);
    CHECK(state_is_zero(&r), "%s: reset left state", label);
    CHECK(post(&r, pcm, STREAM_RUN_OK, &o) == -1, "%s: post after a reset dropped the call", label);
    CHECK(pre(&r, &c, pcm, &io) == 1, "%s: pre after the reset", label);
    dev(&r, &io, DEV_OK, 0, pcm);
    CHECK(post(&r, pcm, STREAM_RUN_OK, &o) == 1, "%s: the stream went on", label);
    rig_free(&r);
    free(buf); free(pc);
}

/* One call with run status `status` and device kind `kind`: returns the generation delta; fills the output. */
static long one_call(Rig *r, const Call *c, int pcm, int status, int kind, int frame, Out *o, int *rc) {
    DeepVqeHopIO io;
    const uint32_t before = deepvqe_stream_state_generation(r->s);
    size_t k;
    int nan_head = 1;
    if (pre(r, c, pcm, &io) != 1) { *rc = -9; return 0; }
    for (k = 0; k < io.outputs.taps_elements; ++k) nan_head &= isnan(io.outputs.taps[k]) != 0;
    CHECK(nan_head, "the head is not NaN before the run");
    if (status != STREAM_RUN_NOT_STARTED) dev(r, &io, kind, frame, pcm);
    *rc = post(r, pcm, status, o);
    return (long)deepvqe_stream_state_generation(r->s) - (long)before;
}

static void failure_table(const Row *row, int freq_only, int pcm, const char *label) {
    static const struct { int status, kind, delta, zero_out, state_zero; const char *what; } T[] = {
        { STREAM_RUN_OK, DEV_OK, 0, 0, 0, "clean run" },
        { STREAM_RUN_OK, DEV_NAN_TAP, 1, 1, 1, "refused: a NaN tap" },
        { STREAM_RUN_OK, DEV_NAN_STATE, 1, 1, 1, "refused: a NaN state element" },
        { STREAM_RUN_OK, DEV_UNWRITTEN, 1, 1, 1, "refused: nothing written (the head stayed NaN)" },
        { STREAM_RUN_OK, DEV_OVERFLOW, 1, 1, 1, "refused: an enhanced bin overflows" },
        { STREAM_RUN_OK, DEV_OVERFLOW_IM, 1, 1, 1, "refused: an enhanced bin overflows in the imaginary plane only" },
        { STREAM_RUN_NOT_STARTED, DEV_OK, 0, 1, 0, "not started" },
        { STREAM_RUN_FAILED, DEV_OK, 1, 1, 1, "failed" },
        { 12345, DEV_OK, 1, 1, 1, "odd status" },
        { STREAM_RUN_FAILED, DEV_NAN_TAP, 1, 1, 1, "failed with a NaN tap: one restart" }
    };
    Rig r;
    Call c;
    Out o;
    const size_t hop = (size_t)row->fft / 2u * (size_t)row->n, nb = (size_t)row->fft / 2u + 1u;
    float *buf = (float *)calloc(nb * (size_t)row->n * 6u, sizeof(float));
    int16_t *pc = (int16_t *)calloc(hop * 3u, sizeof(int16_t));
    int t, warm;
    if (!buf || !pc || rig_make(&r, row, freq_only) != 0) exit(2);
    c.mr = buf; c.mi = buf + r.plane; c.fr = buf + 2u * r.plane; c.fi = buf + 3u * r.plane;
    c.mpcm = pc; c.fpcm = pc + hop;
    o.re = buf + 4u * r.plane; o.im = buf + 5u * r.plane; o.pcm = pc + 2u * hop;

    for (t = 0; t < (int)(sizeof T / sizeof T[0]); ++t) {
        const int frame = row->n > 1 ? row->n - 2 : 0;   /* a later frame of a batched call */
        int rc;
        long delta;
        if (T[t].kind == DEV_OVERFLOW_IM && pcm) continue;
        for (warm = 0; warm < 4; ++warm) {   /* a warm state: the zeroing is visible */
            make_call(&r, &c, 10 + warm, -1);
            (void)one_call(&r, &c, pcm, STREAM_RUN_OK, DEV_OK, 0, &o, &rc);
        }
        CHECK(!state_is_zero(&r), "%s %s: no state to restart", label, T[t].what);
        make_call(&r, &c, 100 + t, T[t].kind == DEV_OVERFLOW && !pcm ? frame : -1);
        if (T[t].kind == DEV_OVERFLOW_IM) c.mi[(size_t)frame * (size_t)r.g.n_bins + 3u] = 3.0e38f;
        delta = one_call(&r, &c, pcm, T[t].status, T[t].kind, frame, &o, &rc);
        CHECK(rc == 1, "%s %s: post rc %d", label, T[t].what, rc);
        CHECK(delta == (long)T[t].delta, "%s %s: generation moved by %ld, expected %d", label, T[t].what, delta, T[t].delta);
        CHECK(state_is_zero(&r) == T[t].state_zero, "%s %s: registered state zero = %d, expected %d", label, T[t].what,
              state_is_zero(&r), T[t].state_zero);
        if (!pcm) CHECK((all_zero_f(o.re, r.plane) && all_zero_f(o.im, r.plane)) == T[t].zero_out, "%s %s: output zero = %d", label,
                        T[t].what, !T[t].zero_out);
        if (T[t].delta) {   /* the stream goes on from a cold state; a second restart is not owed to this call */
            long d2;
            make_call(&r, &c, 200 + t, -1);
            d2 = one_call(&r, &c, pcm, STREAM_RUN_OK, DEV_OK, 0, &o, &rc);
            CHECK(rc == 1 && d2 == 0, "%s %s: the next clean call moved the generation", label, T[t].what);
        }
        deepvqe_stream_reset(r.s);
    }
    rig_free(&r);
    free(buf); free(pc);
}

/* The N output frames of a committed call equal N hand-driven CCM calls, in slot order, over the raw mic spectra the caller
 * passed in (the ring carries across calls): a stream that ran the slots out of order, or took another spectrum as the raw
 * mic, differs. */
static void commit_equals_ccm(const Row *row, int freq_only, const char *label) {
    Rig r;
    Call c;
    Out o;
    DeepVqeCcmState ccm;
    const size_t nb = (size_t)row->fft / 2u + 1u, taps_per_frame = nb * 18u;
    float *buf = (float *)calloc(nb * (size_t)row->n * 8u, sizeof(float));
    float *ring = (float *)calloc(2u * 3u * nb, sizeof(float));
    int call, k, real = 0;
    if (!buf || !ring || rig_make(&r, row, freq_only) != 0) exit(2);
    memset(&c, 0, sizeof c);
    c.mr = buf; c.mi = buf + r.plane; c.fr = buf + 2u * r.plane; c.fi = buf + 3u * r.plane;
    o.re = buf + 4u * r.plane; o.im = buf + 5u * r.plane; o.pcm = NULL;
    if (deepvqe_ccm_init(&ccm, r.g.n_bins, ring, ring + 3u * nb) != 0) exit(2);
    for (call = 0; call < 4; ++call) {
        DeepVqeHopIO io;
        float *want_re = buf + 6u * r.plane, *want_im = buf + 7u * r.plane;
        make_call(&r, &c, 500 + call, -1);
        CHECK(pre(&r, &c, 0, &io) == 1, "%s: pre", label);
        dev(&r, &io, DEV_OK, 0, 0);
        for (k = 0; k < row->n; ++k)
            deepvqe_ccm_process(&ccm, c.mr + (size_t)k * nb, c.mi + (size_t)k * nb, io.outputs.taps + (size_t)k * taps_per_frame,
                                want_re + (size_t)k * nb, want_im + (size_t)k * nb);
        CHECK(post(&r, 0, STREAM_RUN_OK, &o) == 1, "%s: post", label);
        CHECK(memcmp(o.re, want_re, r.plane * sizeof(float)) == 0 && memcmp(o.im, want_im, r.plane * sizeof(float)) == 0,
              "%s: call %d differs from the hand-driven CCM", label, call);
        real += !all_zero_f(o.re, r.plane);
    }
    CHECK(real == 4, "%s: the reference is silent", label);
    rig_free(&r);
    free(buf); free(ring);
}

/* PCM: an overflow in frame k of a batched call gives the PCM of a twin whose call failed, and so does every call after it. */
static void overflow_all_or_none(const Row *row, const char *label) {
    Rig a, b;
    Call ca, cb;
    Out oa, ob;
    const size_t hop = (size_t)row->fft / 2u * (size_t)row->n, nb = (size_t)row->fft / 2u + 1u;
    float *buf = (float *)calloc(nb * (size_t)row->n * 12u, sizeof(float));
    int16_t *pc = (int16_t *)calloc(hop * 6u, sizeof(int16_t));
    int w, rc, rcb;
    if (!buf || !pc || rig_make(&a, row, 0) != 0 || rig_make(&b, row, 0) != 0) exit(2);
    ca.mr = buf; ca.mi = buf + a.plane; ca.fr = buf + 2u * a.plane; ca.fi = buf + 3u * a.plane;
    ca.mpcm = pc; ca.fpcm = pc + hop; oa.pcm = pc + 2u * hop; oa.re = buf + 4u * a.plane; oa.im = buf + 5u * a.plane;
    cb = ca; cb.mr += 6u * a.plane; cb.mi += 6u * a.plane; cb.fr += 6u * a.plane; cb.fi += 6u * a.plane;
    cb.mpcm = pc + 3u * hop; cb.fpcm = pc + 4u * hop; ob = oa; ob.pcm = pc + 5u * hop; ob.re = cb.mr + 4u * a.plane; ob.im = cb.mr + 5u * a.plane;
    for (w = 0; w < 6; ++w) {
        const int overflow = w == 3;
        const int status_b = overflow ? STREAM_RUN_FAILED : STREAM_RUN_OK;
        make_call(&a, &ca, 300 + w, -1);
        make_call(&b, &cb, 300 + w, -1);
        (void)one_call(&a, &ca, 1, STREAM_RUN_OK, overflow ? DEV_OVERFLOW : DEV_OK, row->n - 1, &oa, &rc);
        (void)one_call(&b, &cb, 1, status_b, DEV_OK, 0, &ob, &rcb);
        CHECK(rc == 1 && rcb == 1, "%s: post rc %d / %d", label, rc, rcb);
        CHECK(memcmp(oa.pcm, ob.pcm, hop * sizeof(int16_t)) == 0, "%s: call %d: PCM differs from the failed twin%s", label, w,
              overflow ? " (a frame before the overflow reached the ISTFT)" : "");
        CHECK(deepvqe_stream_state_generation(a.s) == deepvqe_stream_state_generation(b.s), "%s: call %d: generations differ", label, w);
    }
    CHECK(deepvqe_stream_state_generation(a.s) == 1u, "%s: the overflow restarted %u times", label, deepvqe_stream_state_generation(a.s));
    rig_free(&a); rig_free(&b);
    free(buf); free(pc);
}

/* _freq: a non-finite bin is -2, nothing moves, and the stream equals a twin that never saw the call. */
static void range_refusal(const Row *row, int freq_only, const char *label) {
    Rig a, b;
    Call c;
    Out o;
    const size_t nb = (size_t)row->fft / 2u + 1u;
    float *buf = (float *)calloc(nb * (size_t)row->n * 6u, sizeof(float));
    int plane_id, which, pos, rc_a, rc_b, seen = 0;
    static const float bad[3] = { NAN, INFINITY, -INFINITY };
    if (!buf || rig_make(&a, row, freq_only) != 0 || rig_make(&b, row, freq_only) != 0) exit(2);
    memset(&c, 0, sizeof c); memset(&o, 0, sizeof o);
    c.mr = buf; c.mi = buf + a.plane; c.fr = buf + 2u * a.plane; c.fi = buf + 3u * a.plane;
    o.re = buf + 4u * a.plane; o.im = buf + 5u * a.plane;
    for (plane_id = 0; plane_id < 4; ++plane_id)
        for (which = 0; which < 3; ++which)
            for (pos = 0; pos < 2; ++pos) {
                float *pl[4];
                DeepVqeHopIO io, before;
                unsigned char *snap = (unsigned char *)malloc(a.pool_bytes);
                size_t at;
                const uint32_t gen = deepvqe_stream_state_generation(a.s);
                float keep;
                pl[0] = c.mr; pl[1] = c.mi; pl[2] = c.fr; pl[3] = c.fi;
                make_call(&a, &c, 400 + seen, -1);
                at = pos ? a.plane - 1u : 0u;
                keep = pl[plane_id][at];
                pl[plane_id][at] = bad[which];
                memset(&io, 0x5A, sizeof io);
                before = io;
                memcpy(snap, a.pool, a.pool_bytes);
                CHECK(deepvqe_stream_pre_process_packet_freq(a.s, c.mr, c.mi, c.fr, c.fi, &io) == DEEPVQE_STREAM_ERR_RANGE,
                      "%s: plane %d bin %d at %zu accepted", label, plane_id, which, at);
                if (row->n == 1) CHECK(deepvqe_stream_pre_process_freq(a.s, c.mr, c.mi, c.fr, c.fi, &io) == DEEPVQE_STREAM_ERR_RANGE,
                                       "%s: the one-hop entry accepted", label);
                CHECK(memcmp(&io, &before, sizeof io) == 0 && memcmp(snap, a.pool, a.pool_bytes) == 0 &&
                      deepvqe_stream_state_generation(a.s) == gen, "%s: a refused call changed the stream", label);
                CHECK(deepvqe_stream_post_process_packet_freq(a.s, STREAM_RUN_OK, o.re, o.im) == -1, "%s: a call was left open", label);
                pl[plane_id][at] = keep;
                free(snap);
                /* the same good call on both streams: equal jobs and outputs, a twin that never saw the refused one */
                {
                    DeepVqeHopIO ia, ib;
                    float *ra = o.re, *rb = (float *)malloc(a.plane * 2u * sizeof(float));
                    if (!rb) exit(2);
                    rc_a = deepvqe_stream_pre_process_packet_freq(a.s, c.mr, c.mi, c.fr, c.fi, &ia);
                    rc_b = deepvqe_stream_pre_process_packet_freq(b.s, c.mr, c.mi, c.fr, c.fi, &ib);
                    CHECK(rc_a == 1 && rc_b == 1 && ia.state_generation == ib.state_generation &&
                          memcmp(ia.inputs.mic, ib.inputs.mic, ia.inputs.spectrum_ri_elements * sizeof(float)) == 0 &&
                          memcmp(ia.inputs.far, ib.inputs.far, ia.inputs.spectrum_ri_elements * sizeof(float)) == 0, "%s: the twin's job differs", label);
                    dev(&a, &ia, DEV_OK, 0, 0);
                    dev(&b, &ib, DEV_OK, 0, 0);
                    CHECK(deepvqe_stream_post_process_packet_freq(a.s, STREAM_RUN_OK, ra, ra + a.plane) == 1 &&
                          deepvqe_stream_post_process_packet_freq(b.s, STREAM_RUN_OK, rb, rb + a.plane) == 1 &&
                          memcmp(ra, rb, a.plane * sizeof(float)) == 0 && memcmp(ra + a.plane, rb + a.plane, a.plane * sizeof(float)) == 0 &&
                          deepvqe_stream_state_generation(a.s) == deepvqe_stream_state_generation(b.s), "%s: the twin's output differs", label);
                    free(rb);
                }
                ++seen;
            }
    CHECK(seen == 24, "%s: %d refusal rows", label, seen);
    rig_free(&a); rig_free(&b);
    free(buf);
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof ROWS / sizeof ROWS[0]; ++i) {
        char label[96];
        int fo;
        for (fo = 0; fo < 2; ++fo) {
            int pcm;
            for (pcm = 0; pcm < 2; ++pcm) {
                if (fo && pcm) continue;
                snprintf(label, sizeof label, "%d/%d N=%d %s %s", ROWS[i].sr, ROWS[i].fft, ROWS[i].n, fo ? "freq_only" : "full", pcm ? "pcm" : "freq");
                guards(&ROWS[i], fo, pcm, label);
                failure_table(&ROWS[i], fo, pcm, label);
            }
            snprintf(label, sizeof label, "%d/%d N=%d %s", ROWS[i].sr, ROWS[i].fft, ROWS[i].n, fo ? "freq_only" : "full");
            range_refusal(&ROWS[i], fo, label);
            commit_equals_ccm(&ROWS[i], fo, label);
        }
        if (ROWS[i].n > 1) {
            snprintf(label, sizeof label, "%d/%d N=%d pcm", ROWS[i].sr, ROWS[i].fft, ROWS[i].n);
            overflow_all_or_none(&ROWS[i], label);
        }
    }
    if (g_fail) { printf("lifecycle: %d check(s) FAILED\n", g_fail); return 1; }
    printf("lifecycle: %d grid(s) -- PASS\n", (int)(sizeof ROWS / sizeof ROWS[0]));
    return 0;
}
