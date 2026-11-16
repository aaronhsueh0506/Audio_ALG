/* The one-hop spectrum entries of ulcnet_stream (pre_process_freq / post_process_freq) in absolute terms: which hops
 * open a frame, what the accelerator is shown, what each hop hands back, when the state generation moves, the call
 * order, reset and the pool plan. Every expectation is built here: the edge rule, the front end and the mask
 * arithmetic are written again in this file (scalar, unfused), so an accepted hop is checked bit for bit against
 * arithmetic that does not come from the library, and an identity hop against the error spectrum handed in. The
 * stand-in accelerator writes a mask that is a hash of the tensors it is shown. Runs four grids. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet_stream.h"
#include "stream_run.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); ++g_fail; } } while (0)

enum { BINS_MAX = 513, EXTRA = 64 };

static UlcnetGrid g_grid;
#define BINS (g_grid.n_bins)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float urand(uint32_t *s) { return (float)(int)(lcg(s) >> 9) / 4194304.0f - 1.0f; }

static void *pool_of(size_t bytes, int fill) {
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes ? bytes : 16) != 0) return NULL;
    memset(p, fill, bytes);
    return p;
}

static int same_bits(const float *a, const float *b, size_t n) { return memcmp(a, b, n * sizeof(float)) == 0; }

/* ---- the graph's arithmetic, written again ---- */

static void front_end_of(const float *er, const float *ei, const float *fr, const float *fi,
                         float *emag, float *fmag, float *ecos, float *esin) {
    int k;
    for (k = 0; k < BINS; ++k) {
        const float c_re = ulcnet_model_io_signed_pow(er[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float c_im = ulcnet_model_io_signed_pow(ei[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float f_re = ulcnet_model_io_signed_pow(fr[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float f_im = ulcnet_model_io_signed_pow(fi[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float norm = hypotf(c_re, c_im);
        emag[k] = sqrtf(c_re * c_re + c_im * c_im + 1e-12f);
        fmag[k] = sqrtf(f_re * f_re + f_im * f_im + 1e-12f);
        ecos[k] = norm > 0.0f ? c_re / norm : 1.0f;
        esin[k] = norm > 0.0f ? c_im / norm : 0.0f;
    }
}

static void enhanced_of(const float *er, const float *ei, const float *mask, float *out_re, float *out_im) {
    const float inverse = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
    int k;
    for (k = 0; k < BINS; ++k) {
        const float c_re = ulcnet_model_io_signed_pow(er[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float c_im = ulcnet_model_io_signed_pow(ei[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        out_re[k] = ulcnet_model_io_signed_pow(c_re * mask[k] - c_im * mask[BINS + k], inverse);
        out_im[k] = ulcnet_model_io_signed_pow(c_re * mask[BINS + k] + c_im * mask[k], inverse);
    }
}

/* ---- the stand-in accelerator ---- */

/* What a run does: write the whole mask, report failure (a started run, or one that never started) after writing it,
 * report success with nothing or half of it written, or write it with one bad value. */
enum { K_PLAIN, K_FAIL, K_NOT_STARTED, K_UNWRITTEN, K_HALF, K_NAN, K_INF, K_HUGE_RE, K_HUGE_IM };

typedef struct Fake {
    long calls;
    float mask[2 * BINS_MAX];   /* the mask of the last run, as written */
} Fake;

static int fake_run(Fake *f, int kind, const UlcnetModelIoInputs *in, UlcnetModelIoOutputs *out) {
    const size_t bins = out->spectrum_ri_elements / 2u;
    uint32_t h = (uint32_t)f->calls * 2654435761u, bits;
    const float *planes[4];
    size_t i;
    planes[0] = in->error_mag; planes[1] = in->far_mag; planes[2] = in->error_cos; planes[3] = in->error_sin;
    for (i = 0; i < 4u * in->spectrum_bins_elements; ++i) {
        memcpy(&bits, &planes[i / in->spectrum_bins_elements][i % in->spectrum_bins_elements], sizeof bits);
        h = (h ^ bits) * 16777619u;
    }
    f->calls++;
    if (kind == K_UNWRITTEN) return 0;
    for (i = 0; i < bins; ++i) {
        lcg(&h); f->mask[i] = 0.2f + 0.9f * (float)((h >> 8) & 0xffff) / 65535.0f;
        lcg(&h); f->mask[bins + i] = 0.3f * ((float)((h >> 8) & 0xffff) / 32768.0f - 1.0f);
    }
    if (kind == K_HALF) {
        memcpy(out->output, f->mask, (bins / 2u) * sizeof(float));
        memcpy(out->output + bins, f->mask + bins, (bins / 2u) * sizeof(float));
        return 0;
    }
    if (kind == K_NAN) f->mask[(2u * bins) / 3u] = NAN;
    if (kind == K_INF) f->mask[bins / 3u] = -INFINITY;
    if (kind == K_HUGE_RE) f->mask[5] = 3.0e30f;
    if (kind == K_HUGE_IM) f->mask[bins] = 3.0e30f;
    memcpy(out->output, f->mask, 2u * bins * sizeof(float));
    return kind == K_FAIL ? STREAM_RUN_FAILED : (kind == K_NOT_STARTED ? STREAM_RUN_NOT_STARTED : STREAM_RUN_OK);
}

/* ---- a stream in an exact pool with a guard behind it ---- */

typedef struct Rig {
    UlcnetStream *s;
    unsigned char *pool;
    size_t bytes;
    Fake fake;
} Rig;

static int rig_init(Rig *r, int fill) {
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    memset(r, 0, sizeof *r);
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = g_grid;
    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0) return -1;
    r->bytes = (size_t)req.bytes;
    r->pool = (unsigned char *)pool_of(r->bytes + EXTRA, fill);
    if (!r->pool) return -1;
    memset(r->pool + r->bytes, 0xA5, EXTRA);
    r->s = ulcnet_stream_init_ex(&cfg, r->pool, r->bytes, &req);
    return r->s ? 0 : -1;
}

static int rig_guard_intact(const Rig *r) {
    size_t i;
    for (i = 0; i < EXTRA; ++i) if (r->pool[r->bytes + i] != 0xA5) return 0;
    return 1;
}

static void rig_free(Rig *r) { free(r->pool); memset(r, 0, sizeof *r); }

static void fill_spectra(uint32_t *seed, int h, float *er, float *ei, float *fr, float *fi) {
    const float level = (h % 31 == 7) ? 4.0f : (h % 9 == 0) ? 1e-4f : 0.2f;
    int k;
    for (k = 0; k < BINS; ++k) {
        er[k] = level * urand(seed); ei[k] = (k == 0 || k == BINS - 1) ? 0.0f : level * urand(seed);
        fr[k] = 0.3f * urand(seed);  fi[k] = (k == 0 || k == BINS - 1) ? 0.0f : 0.3f * urand(seed);
    }
}

static UlcnetStreamDelay ev(int changed, int solid) {
    UlcnetStreamDelay d;
    d.changed = changed;
    d.solid = solid;
    return d;
}

/* ---- one scripted run against the oracle ---- */

typedef struct Sched { int changed_at[8]; int solid_from, solid_off_at, solid_back_at; } Sched;

static void sched_clear(Sched *sc) {
    int k;
    memset(sc, 0, sizeof *sc);
    for (k = 0; k < 8; ++k) sc->changed_at[k] = -1;
    sc->solid_off_at = -1;
    sc->solid_back_at = -1;
}

static UlcnetStreamDelay sched_event(const Sched *s, int h) {
    int k, changed = 0, solid;
    for (k = 0; k < 8; ++k) if (s->changed_at[k] == h) changed = 1;
    solid = h >= s->solid_from;
    if (s->solid_off_at >= 0 && h >= s->solid_off_at && h < s->solid_back_at) solid = 0;
    return ev(changed, solid);
}

typedef struct Tally {
    long calls, edges, refused, failed, enhanced;
    int bad_pre;          /* pre_process_freq did not return 1 on a model hop / 0 on an edge hop */
    int bad_io;           /* a frame's views: tensors, NaN staging, state views, generation; a reprime hop's zeroed io */
    int bad_identity;     /* an identity hop that is not the error spectrum bit for bit */
    int bad_enhanced;     /* an accepted hop that is not the mask arithmetic bit for bit, or equals the error spectrum */
    int bad_generation;   /* the generation is not edges + resets + refusals after the hop */
    uint32_t generation;  /* expected, running */
} Tally;

/* The kind of run number `call` (0 based) in a fail-open row: every `period`-th run, offset by 3, is `kind`. */
static int kind_of_call(int kind, int period, long call) {
    return period > 0 && call % period == 3 ? kind : K_PLAIN;
}

/* `n` hops of schedule `sc` through r; at hop `nan_err_at` (-1: none) the stream is first offered the hop with a NaN
 * in the error spectrum and must refuse it without moving; outs gets 2 * BINS floats per hop when given. The tally starts from *t (so a run can continue after a reset). */
static int run_hops(Rig *r, const Sched *sc, int n, int kind, int period, int nan_err_at, uint32_t seed, Tally *t,
                    float *outs) {
    static float er[BINS_MAX], ei[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX], re[BINS_MAX], im[BINS_MAX];
    static float want_re[BINS_MAX], want_im[BINS_MAX], emag[BINS_MAX], fmag[BINS_MAX], ecos[BINS_MAX], esin[BINS_MAX];
    int h, k, had_hop = 0, prev_solid = 0;

    for (h = 0; h < n; ++h) {
        const UlcnetStreamDelay d = sched_event(sc, h);
        const int edge = d.changed || (had_hop && !prev_solid && d.solid);   /* the rule, written here */
        UlcnetHopIO io;
        int pre, rc = -1, run_kind = K_PLAIN, identity, refused = 0;

        had_hop = 1; prev_solid = d.solid;
        fill_spectra(&seed, h, er, ei, fr, fi);
        if (h == nan_err_at) {
            float *probe = (float *)malloc((size_t)BINS * sizeof(float));
            UlcnetHopIO bad;
            int rc_bad;
            if (!probe) return -1;
            memcpy(probe, er, (size_t)BINS * sizeof(float));
            probe[7] = NAN;
            memset(&bad, 0x5A, sizeof bad);
            rc_bad = ulcnet_stream_pre_process_freq(r->s, probe, ei, fr, fi, d, &bad);
            free(probe);
            if (rc_bad != ULCNET_STREAM_ERR_RANGE || ulcnet_stream_post_process_freq(r->s, 0, re, im) != -1 ||
                ulcnet_stream_state_generation(r->s) != t->generation)
                t->bad_pre++;
        }
        if (edge) { t->edges++; t->generation++; }

        memset(&io, 0x5A, sizeof io);
        pre = ulcnet_stream_pre_process_freq(r->s, er, ei, fr, fi, d, &io);
        if (pre != (edge ? 0 : 1)) { t->bad_pre++; if (pre < 0) return -1; }
        if (pre == 1) {
            int ok = io.state_generation == t->generation && io.inputs.spectrum_bins_elements == (size_t)BINS &&
                     io.outputs.spectrum_ri_elements == 2u * (size_t)BINS && io.outputs.output != NULL &&
                     io.inputs.key_history == NULL && io.inputs.value_history == NULL &&
                     io.inputs.logit_history == NULL && io.inputs.h_gru == NULL && io.inputs.frame_reset == NULL &&
                     io.outputs.key_history_out == NULL && io.outputs.h_gru_out == NULL;
            for (k = 0; ok && k < 2 * BINS; ++k) ok = isnan(io.outputs.output[k]);
            if (ok) {
                front_end_of(er, ei, fr, fi, emag, fmag, ecos, esin);
                ok = same_bits(io.inputs.error_mag, emag, (size_t)BINS) && same_bits(io.inputs.far_mag, fmag, (size_t)BINS) &&
                     same_bits(io.inputs.error_cos, ecos, (size_t)BINS) && same_bits(io.inputs.error_sin, esin, (size_t)BINS);
            }
            if (!ok) t->bad_io++;
            run_kind = kind_of_call(kind, period, r->fake.calls);
            rc = fake_run(&r->fake, run_kind, &io.inputs, &io.outputs);
            t->calls++;
        } else {
            UlcnetHopIO zero;
            memset(&zero, 0, sizeof zero);
            if (memcmp(&io, &zero, sizeof io) != 0) t->bad_io++;
        }
        if (ulcnet_stream_post_process_freq(r->s, rc, re, im) != 1) return -1;

        if (pre != 1) identity = 1;
        else if (run_kind == K_FAIL) { identity = 1; t->failed++; t->generation++; }
        else if (run_kind == K_NOT_STARTED) { identity = 1; t->failed++; }
        else if (run_kind != K_PLAIN) { identity = 1; refused = 1; }
        else identity = 0;
        if (refused) { t->refused++; t->generation++; }
        if (identity) {
            if (!same_bits(re, er, (size_t)BINS) || !same_bits(im, ei, (size_t)BINS)) t->bad_identity++;
        } else {
            enhanced_of(er, ei, r->fake.mask, want_re, want_im);
            t->enhanced++;
            if (!same_bits(re, want_re, (size_t)BINS) || !same_bits(im, want_im, (size_t)BINS) ||
                (same_bits(re, er, (size_t)BINS) && same_bits(im, ei, (size_t)BINS)))
                t->bad_enhanced++;
        }
        if (ulcnet_stream_state_generation(r->s) != t->generation) t->bad_generation++;
        if (outs) {
            memcpy(outs + (size_t)h * 2u * (size_t)BINS, re, (size_t)BINS * sizeof(float));
            memcpy(outs + (size_t)h * 2u * (size_t)BINS + (size_t)BINS, im, (size_t)BINS * sizeof(float));
        }
    }
    return 0;
}

static int tally_clean(const Tally *t) {
    return !t->bad_pre && !t->bad_io && !t->bad_identity && !t->bad_enhanced && !t->bad_generation;
}

/* ---- rows ---- */

/* Which hops open a frame: one run per hop except the one identity hop after each alignment boundary. */
static void row_alignment_events(void) {
    enum { N = 60 };
    Rig r;
    Tally t;
    Sched sc;

#define EVENT_CASE(msg, want_edges)                                                                      \
    do {                                                                                                 \
        memset(&t, 0, sizeof t);                                                                         \
        CHECK(rig_init(&r, 0) == 0 && run_hops(&r, &sc, N, K_PLAIN, 0, -1, 5u, &t, NULL) == 0, msg ": run"); \
        CHECK(tally_clean(&t), msg ": every hop is what the schedule says");                             \
        CHECK(t.edges == (want_edges) && t.calls == N - (want_edges) * (long)ULCNET_STREAM_REPRIME_FRAMES && \
              r.fake.calls == t.calls && t.enhanced == t.calls && t.refused == 0 &&                      \
              ulcnet_stream_state_generation(r.s) == (uint32_t)(want_edges), msg);                       \
        rig_free(&r);                                                                                    \
    } while (0)

    sched_clear(&sc); sc.changed_at[0] = 10; sc.changed_at[1] = 30; sc.solid_from = 10;
    EVENT_CASE("events: two generation changes restart the model twice, one identity hop each", 2);
    sched_clear(&sc); sc.solid_from = 0;
    EVENT_CASE("events: a stream that is solid from hop #0 never restarts", 0);
    sc.solid_from = 15;
    EVENT_CASE("events: the first transition to solid after hop #0 is a boundary", 1);
    sc.solid_from = 0; sc.solid_off_at = 20; sc.solid_back_at = 35;
    EVENT_CASE("events: a relock after an unlocked stretch is a boundary", 1);
    sc.solid_from = N + 5; sc.solid_off_at = -1;
    EVENT_CASE("events: an unlocked stream runs the model on every hop", 0);
    sc.solid_from = 0; sc.changed_at[0] = 0;
    EVENT_CASE("events: a generation change on hop #0 is a boundary", 1);
    sched_clear(&sc); sc.changed_at[0] = 40; sc.changed_at[1] = 41; sc.solid_from = 20; sc.solid_off_at = 50;
    sc.solid_back_at = 55;
    EVENT_CASE("events: consecutive changes and a relock are four boundaries", 4);
#undef EVENT_CASE
}

/* What a hop hands back when the run failed or wrote a mask the stream refuses, and that the next run is applied. */
static void row_fail_open(void) {
    enum { N = 50 };
    /* refused: counted as a refused mask (else as a failed run); steps: generation steps per bad hop (0 or 1) */
    static const struct { int kind; int refused; int steps; const char *name; } rows[] = {
        { K_FAIL, 0, 1, "fail-open: a run that reports failure after writing a finite mask" },
        { K_NOT_STARTED, 0, 0, "fail-open: a run that never started" },
        { K_UNWRITTEN, 1, 1, "fail-open: a mask left unwritten (the NaN staging)" },
        { K_HALF, 1, 1, "fail-open: a mask half written" },
        { K_NAN, 1, 1, "fail-open: a NaN mask bin" },
        { K_INF, 1, 1, "fail-open: an Inf mask bin" },
        { K_HUGE_RE, 1, 1, "fail-open: a 3.0e30 mask bin, real plane" },
        { K_HUGE_IM, 1, 1, "fail-open: a 3.0e30 mask bin, imaginary plane" },
    };
    size_t i;
    Sched sc;
    sched_clear(&sc);
    sc.solid_from = 0;
    sc.changed_at[0] = 12;
    for (i = 0; i < sizeof rows / sizeof rows[0]; ++i) {
        Rig r;
        Tally t;
        long bad;
        memset(&t, 0, sizeof t);
        CHECK(rig_init(&r, 0) == 0 && run_hops(&r, &sc, N, rows[i].kind, 7, -1, 71u, &t, NULL) == 0, rows[i].name);
        bad = rows[i].refused ? t.refused : t.failed;
        CHECK(tally_clean(&t), rows[i].name);
        CHECK(bad == 7 && (rows[i].refused ? t.failed : t.refused) == 0 && t.enhanced == N - 1 - 7 &&
              ulcnet_stream_state_generation(r.s) == (uint32_t)(1 + 7 * rows[i].steps),
              "fail-open: seven such hops are the error spectrum, each restarts the state once unless the run never started, the rest are applied");
        CHECK(rig_guard_intact(&r), "fail-open: nothing is written past the pool");
        rig_free(&r);
    }
    {   /* a non-finite error bin: the entry refuses the offer (ULCNET_STREAM_ERR_RANGE), nothing moves, the hop that follows
         * (an alignment edge included) runs as if it had not been offered */
        Rig r;
        Tally t;
        memset(&t, 0, sizeof t);
        CHECK(rig_init(&r, 0) == 0 && run_hops(&r, &sc, N, K_PLAIN, 0, 30, 71u, &t, NULL) == 0,
              "range: NaN error bin run");
        CHECK(tally_clean(&t) && t.refused == 0 && t.enhanced == N - 1,
              "range: a NaN error bin is refused with no state change and the next hop is applied");
        rig_free(&r);
    }
}

/* The call order and the refusals, none of which may change the stream. */
static void row_sequence(void) {
    Rig r;
    static float er[BINS_MAX], ei[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX], re[BINS_MAX], im[BINS_MAX], keep[BINS_MAX];
    unsigned char *before;
    UlcnetHopIO io, io2;
    uint32_t seed = 9u;
    int k;

    CHECK(rig_init(&r, 0) == 0, "sequence: setup");
    if (!r.s) return;
    before = (unsigned char *)pool_of(r.bytes, 0);
    CHECK(before != NULL, "sequence: snapshot buffer");
    if (!before) { rig_free(&r); return; }
    fill_spectra(&seed, 1, er, ei, fr, fi);
    for (k = 0; k < BINS; ++k) { re[k] = 123.0f; keep[k] = 123.0f; }

    memcpy(before, r.pool, r.bytes);
    CHECK(ulcnet_stream_post_process_freq(r.s, 0, re, im) == -1, "sequence: post before any pre is refused");
    CHECK(same_bits(re, keep, (size_t)BINS), "sequence: a refused post writes nothing");
    CHECK(ulcnet_stream_pre_process_freq(NULL, er, ei, fr, fi, ev(0, 0), &io) == -1 &&
          ulcnet_stream_pre_process_freq(r.s, NULL, ei, fr, fi, ev(0, 0), &io) == -1 &&
          ulcnet_stream_pre_process_freq(r.s, er, NULL, fr, fi, ev(0, 0), &io) == -1 &&
          ulcnet_stream_pre_process_freq(r.s, er, ei, NULL, fi, ev(0, 0), &io) == -1 &&
          ulcnet_stream_pre_process_freq(r.s, er, ei, fr, NULL, ev(0, 0), &io) == -1 &&
          ulcnet_stream_pre_process_freq(r.s, er, ei, fr, fi, ev(0, 0), NULL) == -1,
          "sequence: NULL arguments of pre are refused");
    CHECK(ulcnet_stream_post_process_freq(NULL, 0, re, im) == -1, "sequence: NULL stream of post is refused");
    CHECK(ulcnet_stream_pre_process_freq(r.s, er, ei, fr, fi, ev(1, 1), NULL) == -1,
          "sequence: a refused pre with a change event");
    CHECK(memcmp(before, r.pool, r.bytes) == 0 && ulcnet_stream_state_generation(r.s) == 0u,
          "sequence: refused calls leave the pool bytes and the generation unchanged");

    CHECK(ulcnet_stream_pre_process_freq(r.s, er, ei, fr, fi, ev(0, 0), &io) == 1, "sequence: pre opens the frame");
    memcpy(before, r.pool, r.bytes);
    CHECK(ulcnet_stream_pre_process_freq(r.s, er, ei, fr, fi, ev(1, 1), &io2) == -1,
          "sequence: a second pre in the same hop is refused");
    CHECK(ulcnet_stream_post_process_freq(r.s, 0, NULL, im) == -1 && ulcnet_stream_post_process_freq(r.s, 0, re, NULL) == -1,
          "sequence: post with a NULL plane is refused");
    CHECK(memcmp(before, r.pool, r.bytes) == 0, "sequence: those refusals leave the open hop as it was");
    CHECK(fake_run(&r.fake, K_PLAIN, &io.inputs, &io.outputs) == 0 && ulcnet_stream_post_process_freq(r.s, 0, re, im) == 1,
          "sequence: the open hop still closes");
    CHECK(!same_bits(re, er, (size_t)BINS), "sequence: and it is the accepted frame");
    CHECK(ulcnet_stream_post_process_freq(r.s, 0, re, im) == -1, "sequence: a second post in the same hop is refused");

    /* a hop closed without a run is the error spectrum and leaves the stream usable */
    CHECK(ulcnet_stream_pre_process_freq(r.s, er, ei, fr, fi, ev(0, 0), &io) == 1 &&
          ulcnet_stream_post_process_freq(r.s, STREAM_RUN_NOT_STARTED, re, im) == 1 && same_bits(re, er, (size_t)BINS) &&
          same_bits(im, ei, (size_t)BINS) && ulcnet_stream_state_generation(r.s) == 0u,
          "sequence: a hop closed as a run that never started is the error spectrum; the generation stays");
    CHECK(ulcnet_stream_pre_process_freq(r.s, er, ei, fr, fi, ev(0, 0), &io) == 1 &&
          ulcnet_stream_post_process_freq(r.s, 7, re, im) == 1 && same_bits(re, er, (size_t)BINS) &&
          same_bits(im, ei, (size_t)BINS) && ulcnet_stream_state_generation(r.s) == 1u,
          "sequence: a hop closed as a failed run (any other status) is the error spectrum; the generation advances");
    CHECK(rig_guard_intact(&r), "sequence: nothing is written past the pool");
    free(before);
    rig_free(&r);
}

/* Reset: generation, a pending reprime, the edge memory, and the replay against a new stream. */
static void row_reset(void) {
    enum { N = 24 };
    Rig a, b;
    Tally ta, tb;
    Sched sc;
    static float er[BINS_MAX], ei[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX], re[BINS_MAX], im[BINS_MAX];
    static float first[N * 2 * BINS_MAX], replay[N * 2 * BINS_MAX], fresh[N * 2 * BINS_MAX];
    UlcnetHopIO io;
    uint32_t seed = 3u;

    CHECK(rig_init(&a, 0) == 0 && rig_init(&b, 0) == 0, "reset: setup");
    if (!a.s || !b.s) return;
    fill_spectra(&seed, 1, er, ei, fr, fi);

    /* a reset between hops leaves no edge memory: solid rising on the next hop is a first hop, not a boundary */
    CHECK(ulcnet_stream_pre_process_freq(a.s, er, ei, fr, fi, ev(1, 0), &io) == 0 &&
          ulcnet_stream_post_process_freq(a.s, 0, re, im) == 1 && ulcnet_stream_state_generation(a.s) == 1u,
          "reset: a change on hop #0 is an identity hop and one generation step");
    ulcnet_stream_reset(a.s);
    CHECK(ulcnet_stream_state_generation(a.s) == 2u, "reset: one generation step");
    CHECK(ulcnet_stream_pre_process_freq(a.s, er, ei, fr, fi, ev(0, 1), &io) == 1 && io.state_generation == 2u,
          "reset: the next hop opens a frame although solid rises (no edge memory)");
    ulcnet_stream_reset(a.s);   /* with the hop open */
    CHECK(ulcnet_stream_state_generation(a.s) == 3u && ulcnet_stream_post_process_freq(a.s, 0, re, im) == -1,
          "reset: an open hop is dropped");
    ulcnet_stream_reset(NULL);
    CHECK(ulcnet_stream_state_generation(NULL) == 0u && ulcnet_stream_hop_size(NULL) == -1, "reset: NULL stream accessors");

    /* replay: the same hops after a reset equal the first pass and a new stream, outputs and frames alike */
    rig_free(&a);
    CHECK(rig_init(&a, 0) == 0, "reset: replay setup");
    if (!a.s) { rig_free(&b); return; }
    sched_clear(&sc);
    sc.solid_from = 0;
    sc.changed_at[0] = 6;
    memset(&ta, 0, sizeof ta);
    memset(&tb, 0, sizeof tb);
    CHECK(run_hops(&a, &sc, N, K_PLAIN, 0, -1, 77u, &ta, first) == 0 && tally_clean(&ta), "reset: first pass");
    ulcnet_stream_reset(a.s);
    ta.generation++;
    a.fake.calls = 0;
    ta.calls = 0;
    CHECK(run_hops(&a, &sc, N, K_PLAIN, 0, -1, 77u, &ta, replay) == 0 && tally_clean(&ta), "reset: replay");
    CHECK(run_hops(&b, &sc, N, K_PLAIN, 0, -1, 77u, &tb, fresh) == 0 && tally_clean(&tb), "reset: new stream");
    CHECK(memcmp(replay, first, (size_t)N * 2u * (size_t)BINS * sizeof(float)) == 0 &&
          memcmp(replay, fresh, (size_t)N * 2u * (size_t)BINS * sizeof(float)) == 0 && ta.calls == tb.calls,
          "reset: the replay after a reset equals the first pass and a new stream bit for bit, runs included");
    CHECK(ulcnet_stream_state_generation(a.s) == ulcnet_stream_state_generation(b.s) + 2u,
          "reset: the replay's generation is the new stream's plus the first pass's boundary and the reset");
    rig_free(&a);
    rig_free(&b);
}

/* The grid and the pool plan: what is refused, that a refusal leaves the block alone, and that a stale block
 * initialises like a zeroed one. */
static void row_pools(void) {
    enum { N = 40 };
    UlcnetStreamConfig cfg, bad;
    UlcnetStreamMemReq req, other;
    static float out_a[N * 2 * BINS_MAX], out_b[N * 2 * BINS_MAX];
    Sched sc;
    int variant;

    memset(&cfg, 0, sizeof cfg);
    cfg.grid = g_grid;
    CHECK(ulcnet_stream_get_mem_size(&cfg, &req) == 0 && req.bytes > 0 && req.bytes % 16u == 0 &&
          req.alignment == ULCNET_STREAM_ALIGNMENT && req.layout_version == ULCNET_STREAM_LAYOUT_VERSION &&
          req.descriptor_version == ULCNET_STREAM_DESCRIPTOR_VERSION && req.reserved0 == 0 && req.reserved == 0,
          "pools: a valid grid has a pool and a filled descriptor");
    printf("  stream pool %llu bytes (%d Hz / fft %d, D %d)\n", (unsigned long long)req.bytes, g_grid.sample_rate,
           g_grid.n_fft, g_grid.delay_depth);

    memset(&other, 0x5A, sizeof other);
    CHECK(ulcnet_stream_get_mem_size(NULL, &other) == -1 && ulcnet_stream_get_mem_size(&cfg, NULL) == -1,
          "grid: NULL arguments have no pool");
#define BAD_GRID(edit, msg)                                                                           \
    do {                                                                                              \
        UlcnetStreamMemReq untouched;                                                                 \
        memset(&untouched, 0x5A, sizeof untouched);                                                   \
        bad = cfg; edit;                                                                              \
        CHECK(ulcnet_stream_get_mem_size(&bad, &other) == -1 && memcmp(&other, &untouched, sizeof other) == 0, msg); \
    } while (0)
    BAD_GRID(bad.grid.sample_rate = 0, "grid: sample_rate 0 has no pool");
    BAD_GRID(bad.grid.n_fft = 500, "grid: an n_fft that is not a power of two has no pool");
    BAD_GRID(bad.grid.hop = g_grid.hop + 1, "grid: an inconsistent hop has no pool");
    BAD_GRID(bad.grid.n_bins = g_grid.n_bins - 1, "grid: an inconsistent n_bins has no pool");
    BAD_GRID(bad.grid.delay_depth = ULCNET_MODEL_IO_MAX_D + 1, "grid: a delay depth out of range has no pool");
    BAD_GRID(bad.frames_per_call = -1, "grid: a negative frames_per_call has no pool");
    BAD_GRID(bad.frames_per_call = ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL + 1, "grid: a frames_per_call past the maximum has no pool");
#undef BAD_GRID

    /* a refused init leaves the block bytes exactly as they were */
    for (variant = 0; variant < 7; ++variant) {
        const size_t full = (size_t)req.bytes;
        unsigned char *raw = (unsigned char *)pool_of(full + 32, 0xCD);
        unsigned char *snap = (unsigned char *)pool_of(full + 32, 0);
        unsigned char *base = raw;
        const UlcnetStreamConfig *use = &cfg;
        const UlcnetStreamMemReq *expected = &req;
        size_t bytes = full;
        if (!raw || !snap) { CHECK(0, "pools: refusal setup"); free(raw); free(snap); continue; }
        memcpy(snap, raw, full + 32);
        other = req;
        if (variant == 0) bytes = full - 16;
        if (variant == 1) base = raw + 8;
        if (variant == 2) use = NULL;
        if (variant == 3) { bad = cfg; bad.grid.n_fft = 500; use = &bad; }
        if (variant == 4) base = NULL;
        if (variant == 5) { bad = cfg; bad.grid.delay_depth = 1; use = &bad; }
        if (variant == 6) { other.build_flags_hash ^= 1u; expected = &other; }
        CHECK(ulcnet_stream_init_ex(use, base, bytes, expected) == NULL,
              "pools: a short, misaligned or NULL block, a NULL or invalid config and a stale descriptor are refused");
        CHECK(memcmp(snap, raw, full + 32) == 0, "pools: a refused init leaves the block bytes unchanged");
        free(raw); free(snap);
    }
    {   /* a pool sized for another delay depth or frame count is another build */
        UlcnetStreamMemReq q;
        bad = cfg; bad.grid.delay_depth = g_grid.delay_depth == 8 ? 32 : 8;
        CHECK(ulcnet_stream_get_mem_size(&bad, &q) == 0 && q.build_flags_hash != req.build_flags_hash,
              "pools: another delay depth has another build hash");
        bad = cfg; bad.frames_per_call = 8;
        CHECK(ulcnet_stream_get_mem_size(&bad, &q) == 0 && q.build_flags_hash != req.build_flags_hash && q.bytes > req.bytes,
              "pools: a packet stream has another build hash and a larger pool");
        bad = cfg; bad.frames_per_call = 1;
        CHECK(ulcnet_stream_get_mem_size(&bad, &q) == 0 && memcmp(&q, &req, sizeof q) == 0,
              "pools: frames_per_call 1 is the one-hop stream of frames_per_call 0");
    }

    /* a stale block initialises like a zeroed one, and the stream keeps its own copy of the grid */
    {
        Rig a, b;
        Tally ta, tb;
        sched_clear(&sc);
        sc.solid_from = 20;
        sc.changed_at[0] = 20; sc.changed_at[1] = 33;
        memset(&ta, 0, sizeof ta);
        memset(&tb, 0, sizeof tb);
        CHECK(rig_init(&a, 0) == 0 && rig_init(&b, 0xCD) == 0, "pools: twin setup");
        if (a.s && b.s) {
            CHECK(run_hops(&a, &sc, N, K_NAN, 9, -1, 600u, &ta, out_a) == 0 && tally_clean(&ta) &&
                  run_hops(&b, &sc, N, K_NAN, 9, -1, 600u, &tb, out_b) == 0 && tally_clean(&tb), "pools: twin runs");
            CHECK(memcmp(out_a, out_b, (size_t)N * 2u * (size_t)BINS * sizeof(float)) == 0 && ta.calls == tb.calls &&
                  ulcnet_stream_state_generation(a.s) == ulcnet_stream_state_generation(b.s) && ta.refused > 0,
                  "pools: a stale (0xCD) pool initialises like a zeroed one");
            CHECK(rig_guard_intact(&a) && rig_guard_intact(&b), "pools: nothing past the planned bytes is written");
        }
        rig_free(&a);
        rig_free(&b);
    }
    {
        UlcnetStreamConfig mine = cfg;
        unsigned char *pool = (unsigned char *)pool_of((size_t)req.bytes, 0);
        UlcnetStream *s = pool ? ulcnet_stream_init(&mine, pool, (size_t)req.bytes) : NULL;
        memset(&mine, 0, sizeof mine);
        CHECK(s && ulcnet_stream_hop_size(s) == g_grid.hop, "grid: the stream keeps its own copy of the grid");
        free(pool);
    }
    {   /* the one-hop entries belong to a one-hop stream */
        UlcnetStreamMemReq q;
        unsigned char *pool;
        UlcnetStream *s;
        static float z[8 * BINS_MAX];
        UlcnetStreamDelay d8[8];
        UlcnetHopIO io;
        Rig one;
        bad = cfg; bad.frames_per_call = 8;
        memset(d8, 0, sizeof d8);
        CHECK(ulcnet_stream_get_mem_size(&bad, &q) == 0, "packet: pool query");
        pool = (unsigned char *)pool_of((size_t)q.bytes, 0);
        s = pool ? ulcnet_stream_init_ex(&bad, pool, (size_t)q.bytes, &q) : NULL;
        CHECK(s && ulcnet_stream_pre_process_freq(s, z, z, z, z, ev(1, 1), &io) == -1 &&
              ulcnet_stream_state_generation(s) == 0u,
              "packet: a packet stream refuses the one-hop spectrum entry before its change event restarts anything");
        CHECK(rig_init(&one, 0) == 0 && ulcnet_stream_pre_process_packet_freq(one.s, z, z, z, z, d8, &io) == -1,
              "packet: a one-hop stream refuses the packet spectrum entry");
        rig_free(&one);
        free(pool);
    }
}

int main(void) {
    static const UlcnetGridSpec rows[] = {
        { .sample_rate = 16000, .n_fft = 256,  .delay_depth = 8 },
        { .sample_rate = 16000, .n_fft = 512,  .delay_depth = 8 },
        { .sample_rate = 16000, .n_fft = 512,  .delay_depth = 32 },
        { .sample_rate = 48000, .n_fft = 1024, .delay_depth = 8 },
    };
    size_t g;
    for (g = 0; g < sizeof rows / sizeof rows[0]; ++g) {
        if (ulcnet_grid_make(&g_grid, &rows[g]) != ULCNET_GRID_OK || g_grid.n_bins > BINS_MAX) {
            printf("FAIL: grid row %zu does not build or exceeds the test buffers\n", g);
            return 1;
        }
        printf("grid %d Hz / fft %d / %d bins / D %d\n", g_grid.sample_rate, g_grid.n_fft, g_grid.n_bins, g_grid.delay_depth);
        row_alignment_events();
        row_fail_open();
        row_sequence();
        row_reset();
        row_pools();
    }
    if (g_fail) { printf("test_ulcnet_stream_freq: %d check(s) FAILED\n", g_fail); return 1; }
    printf("test_ulcnet_stream_freq: ALL PASS\n");
    return 0;
}
