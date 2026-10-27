/* Stage-level tests of ulcnet/ulcnet_nn_stage.h: the hop order, the identity paths, the
 * model-call policy (reprime, alignment events, fail-open) and the descriptor gate, driven by
 * synthetic alignment events and deterministic models. Expected outputs come from an
 * independently built STFT/WOLA chain. Build and run: `make test` in pipelines/. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet/ulcnet_nn_stage.h"
#include "utility/mem_align.h"

#define HOP ULCNET_HOP
#define BINS ULCNET_BINS

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); ++g_fail; } } while (0)

static uint32_t g_lcg;
static float lcg_unit(void) {   /* [-0.5, 0.5) */
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (float)(g_lcg >> 8) / 16777216.0f - 0.5f;
}

static void* pool_of(size_t bytes, int fill) {
    void* p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) return NULL;
    memset(p, fill, bytes);
    return p;
}

static const UlcnetModelIoDescriptor* io_descriptor(void) {
    static UlcnetModelIoDescriptor d;
    static int ready;
    if (!ready && ulcnet_model_io_descriptor_default(8, &d) == 0) ready = 1;
    return ready ? &d : NULL;
}

/* ---- models ---- */

typedef struct ModelState {
    int infer_calls, reset_calls;
    int saw_nan_prefill;      /* every out bin was NaN at the call */
    int saw_far_mismatch;     /* far spectrum differs from the expected one */
    const float* want_far_re; const float* want_far_im;
} ModelState;

typedef enum { M_COPY, M_FAIL, M_HALF_WRITE, M_NAN, M_SCALE } ModelKind;
static ModelKind g_kind;

static int model_infer(void* user, const float er[BINS], const float ei[BINS],
                       const float fr[BINS], const float fi[BINS], float orr[BINS], float oi[BINS]) {
    ModelState* st = (ModelState*)user;
    int k, all_nan = 1;
    st->infer_calls++;
    for (k = 0; k < BINS; ++k) if (!isnan(orr[k]) || !isnan(oi[k])) all_nan = 0;
    if (!all_nan) st->saw_nan_prefill = 0;
    if (st->want_far_re && (memcmp(fr, st->want_far_re, sizeof(float) * BINS) != 0 ||
                            memcmp(fi, st->want_far_im, sizeof(float) * BINS) != 0))
        st->saw_far_mismatch = 1;
    switch (g_kind) {
        case M_FAIL: return -1;
        case M_HALF_WRITE:
            memcpy(orr, er, sizeof(float) * (BINS / 2));
            memcpy(oi, ei, sizeof(float) * (BINS / 2));
            return 0;
        case M_NAN:
            memcpy(orr, er, sizeof(float) * BINS);
            memcpy(oi, ei, sizeof(float) * BINS);
            orr[BINS / 3] = NAN;
            return 0;
        case M_SCALE:
            for (k = 0; k < BINS; ++k) { orr[k] = 0.5f * er[k]; oi[k] = 0.5f * ei[k]; }
            return 0;
        default:
            memcpy(orr, er, sizeof(float) * BINS);
            memcpy(oi, ei, sizeof(float) * BINS);
            return 0;
    }
}

static void model_reset(void* user) { ((ModelState*)user)->reset_calls++; }

static UlcnetModel make_model(ModelState* st) {
    UlcnetModel m;
    memset(&m, 0, sizeof m);
    m.user = st;
    m.infer = model_infer;
    m.reset = model_reset;
    m.io_descriptor = io_descriptor();
    return m;
}

/* ---- an independent STFT/WOLA chain: own FFT handles, own windows ---- */

typedef struct Ref {
    AcStft* stft;
    AcIstft* istft;
    void *m1, *m2;
} Ref;

static int ref_init(Ref* r) {
    AcStftConfig sc;
    size_t b1, b2;
    ulcnet_framing_config(&sc, NULL, NULL, 0);
    b1 = ac_stft_get_mem_size(&sc);
    b2 = ac_istft_get_mem_size(&sc);
    r->m1 = pool_of(b1, 0);
    r->m2 = pool_of(b2, 0);
    r->stft = r->m1 ? ac_stft_init(r->m1, b1, &sc) : NULL;
    r->istft = r->m2 ? ac_istft_init(r->m2, b2, &sc) : NULL;
    return r->stft && r->istft ? 0 : -1;
}

static void ref_free(Ref* r) { free(r->m1); free(r->m2); }

/* A stage and the hops it is fed. */
static UlcnetNnStage* make_stage(int err_analysis, void** pool_out, int fill) {
    size_t bytes = ulcnet_nn_stage_mem_size(err_analysis);
    void* pool = pool_of(bytes, fill);
    UlcnetNnStage* s = pool ? ulcnet_nn_stage_init(pool, bytes, err_analysis) : NULL;
    *pool_out = pool;
    return s;
}

static void fill_hop(float* x, int n) {
    int i;
    for (i = 0; i < n; ++i) x[i] = lcg_unit() * 0.5f;
}

static void interleave(Complex* dst, const float* re, const float* im) {
    int k;
    for (k = 0; k < BINS; ++k) { dst[k].r = re[k]; dst[k].i = im[k]; }
}

static UlcnetNnDelay ev(int changed, int solid) {
    UlcnetNnDelay d;
    d.changed = changed;
    d.solid = solid;
    return d;
}

/* one hop through the phased calls with the installed model, like ulcnet_nn_stage_process() */
static int phased_hop(UlcnetNnStage* s, const UlcnetModel* m, float* out) {
    UlcnetNnJob job;
    if (m->infer && ulcnet_nn_stage_nn_begin(s, &job) == 1)
        if (ulcnet_nn_stage_nn_end(s, m->infer(m->user, job.err_re, job.err_im, job.far_re, job.far_im,
                                               job.out_re, job.out_im)) != 0) return -1;
    return ulcnet_nn_stage_end(s, out);
}

/* ------------------------------------------------------------------------------------- */

static void test_identity_all_entries(void) {
    enum { N = 40 };
    void *p1, *p2, *p3;
    UlcnetNnStage *planar = make_stage(0, &p1, 0xCD), *cplx = make_stage(0, &p2, 0xCD),
                  *timed = make_stage(1, &p3, 0xCD);
    Ref ref;
    float err[HOP], far[HOP], re[BINS], im[BINS], want[HOP];
    float o1[HOP], o2[HOP], o3[HOP];
    Complex c[BINS];
    int h, i, bad_planar = 0, bad_complex = 0, bad_timed = 0, hop0_zero = 1;
    double energy = 0.0;
    CHECK(planar && cplx && timed && ref_init(&ref) == 0, "identity: setup");
    if (!planar || !cplx || !timed) { free(p1); free(p2); free(p3); return; }
    g_lcg = 11u;
    for (h = 0; h < N; ++h) {
        fill_hop(err, HOP);
        fill_hop(far, HOP);
        (void)ac_stft_push_planar(ref.stft, err, re, im);
        interleave(c, re, im);
        /* with no model the frame is the identity: the expected hop is the reference WOLA */
        {
            int w = ac_istft_push_planar(ref.istft, re, im, want);
            if (w == 0) memset(want, 0, sizeof want);
        }
        if (ulcnet_nn_stage_begin(planar, re, im, far, ev(0, 0)) != 0 ||
            ulcnet_nn_stage_process(planar, o1) != 0) bad_planar++;
        if (ulcnet_nn_stage_begin_complex(cplx, c, far, ev(0, 0)) != 0 ||
            ulcnet_nn_stage_process(cplx, o2) != 0) bad_complex++;
        if (ulcnet_nn_stage_begin_time(timed, err, far, ev(0, 0)) != 0 ||
            ulcnet_nn_stage_process(timed, o3) != 0) bad_timed++;
        if (memcmp(o1, want, sizeof want) != 0) bad_planar++;
        if (memcmp(o2, want, sizeof want) != 0) bad_complex++;
        if (memcmp(o3, want, sizeof want) != 0) bad_timed++;
        if (h == 0) for (i = 0; i < HOP; ++i) if (o1[i] != 0.0f || o2[i] != 0.0f || o3[i] != 0.0f) hop0_zero = 0;
        if (h > 0) for (i = 0; i < HOP; ++i) energy += fabs((double)o1[i]);
    }
    CHECK(bad_planar == 0, "identity: planar entry equals the independent STFT/WOLA chain bit for bit");
    CHECK(bad_complex == 0, "identity: interleaved entry equals it too");
    CHECK(bad_timed == 0, "identity: time-domain entry (own error STFT) equals it too");
    CHECK(hop0_zero, "identity: hop #0 is exactly zero");
    CHECK(energy > 1.0, "identity: the output carries signal");
    ref_free(&ref);
    free(p1); free(p2); free(p3);
}

/* One run of N hops with a delay-event schedule; returns the model's counters and fills outs. */
typedef struct Sched { int changed_at[8]; int solid_from; int solid_off_at; int solid_back_at; } Sched;

static UlcnetNnDelay sched_event(const Sched* s, int h) {
    int k, changed = 0, solid;
    for (k = 0; k < 8; ++k) if (s->changed_at[k] == h) changed = 1;
    solid = h >= s->solid_from;
    if (s->solid_off_at >= 0 && h >= s->solid_off_at && h < s->solid_back_at) solid = 0;
    return ev(changed, solid);
}

static int run_schedule(const Sched* sc, int n, ModelKind kind, ModelState* st, float* outs /* n*HOP or NULL */) {
    void* pool;
    UlcnetNnStage* s = make_stage(0, &pool, 0);
    UlcnetModel m = make_model(st);
    float err_re[BINS], err_im[BINS], far[HOP], out[HOP];
    int h;
    if (!s) { free(pool); return -1; }
    g_kind = kind;
    memset(st, 0, sizeof *st);
    st->saw_nan_prefill = 1;
    if (ulcnet_nn_stage_set_model(s, &m) != 0) { free(pool); return -1; }
    g_lcg = 5u;
    for (h = 0; h < n; ++h) {
        UlcnetNnDelay d = sched_event(sc, h);
        int k;
        for (k = 0; k < BINS; ++k) { err_re[k] = lcg_unit(); err_im[k] = lcg_unit(); }
        fill_hop(far, HOP);
        if (ulcnet_nn_stage_begin(s, err_re, err_im, far, d) != 0 || ulcnet_nn_stage_process(s, out) != 0) {
            free(pool);
            return -1;
        }
        if (outs) memcpy(outs + (size_t)h * HOP, out, sizeof out);
    }
    free(pool);
    return 0;
}

static void test_counting_and_events(void) {
    enum { N = 60 };
    ModelState st;
    Sched sc;
    int k;
    memset(&sc, 0, sizeof sc);
    for (k = 0; k < 8; ++k) sc.changed_at[k] = -1;
    sc.solid_off_at = -1;
    sc.solid_back_at = -1;

    /* generation events at hops 10 and 30 (solid from hop 10): two resets, one identity frame each */
    sc.changed_at[0] = 10; sc.changed_at[1] = 30; sc.solid_from = 10;
    CHECK(run_schedule(&sc, N, M_COPY, &st, NULL) == 0, "events: run");
    CHECK(st.infer_calls == N - 2 * ULCNET_NN_STAGE_REPRIME_FRAMES,
          "events: one inference per hop except the reprime frames after each generation change");
    CHECK(st.reset_calls == 2, "events: model reset once per generation change");

    /* no events, solid from the first hop: no reset, no skipped frame */
    memset(&sc, 0, sizeof sc);
    for (k = 0; k < 8; ++k) sc.changed_at[k] = -1;
    sc.solid_off_at = -1; sc.solid_back_at = -1; sc.solid_from = 0;
    CHECK(run_schedule(&sc, N, M_COPY, &st, NULL) == 0 && st.infer_calls == N && st.reset_calls == 0,
          "events: a stream that is solid from hop #0 never resets");

    /* a mode without generation events: the first transition to solid after hop #0 counts */
    sc.solid_from = 15;
    CHECK(run_schedule(&sc, N, M_COPY, &st, NULL) == 0 && st.reset_calls == 1 &&
          st.infer_calls == N - ULCNET_NN_STAGE_REPRIME_FRAMES,
          "events: first transition to solid resets once and arms the reprime");

    /* solid dropping and rising again is another boundary */
    sc.solid_from = 0; sc.solid_off_at = 20; sc.solid_back_at = 35;
    CHECK(run_schedule(&sc, N, M_COPY, &st, NULL) == 0 && st.reset_calls == 1 &&
          st.infer_calls == N - ULCNET_NN_STAGE_REPRIME_FRAMES,
          "events: a relock after an unlocked stretch is a boundary");

    /* never solid, never changed: nothing resets, the model runs every hop */
    sc.solid_from = N + 5; sc.solid_off_at = -1;
    CHECK(run_schedule(&sc, N, M_COPY, &st, NULL) == 0 && st.reset_calls == 0 && st.infer_calls == N,
          "events: an unlocked stream runs the model on every hop");

    /* a change on hop #0 is a boundary too */
    sc.solid_from = 0;
    sc.changed_at[0] = 0;
    CHECK(run_schedule(&sc, N, M_COPY, &st, NULL) == 0 && st.reset_calls == 1 &&
          st.infer_calls == N - ULCNET_NN_STAGE_REPRIME_FRAMES,
          "events: a generation change on hop #0 resets and arms the reprime");
}

static void test_model_policy(void) {
    enum { N = 50 };
    static float ident[N * HOP], mod[N * HOP];
    ModelState st;
    Sched sc;
    int k, h, differs;
    memset(&sc, 0, sizeof sc);
    for (k = 0; k < 8; ++k) sc.changed_at[k] = -1;
    sc.solid_off_at = -1; sc.solid_back_at = -1; sc.solid_from = 0;
    sc.changed_at[0] = 12;

    CHECK(run_schedule(&sc, N, M_COPY, &st, ident) == 0, "policy: copy model run");
    {   /* the identity reference: no model at all */
        void* pool;
        UlcnetNnStage* s = make_stage(0, &pool, 0);
        float err_re[BINS], err_im[BINS], far[HOP], out[HOP];
        int bad = 0;
        g_lcg = 5u;
        for (h = 0; h < N; ++h) {
            for (k = 0; k < BINS; ++k) { err_re[k] = lcg_unit(); err_im[k] = lcg_unit(); }
            fill_hop(far, HOP);
            if (ulcnet_nn_stage_begin(s, err_re, err_im, far, sched_event(&sc, h)) != 0 ||
                ulcnet_nn_stage_process(s, out) != 0 || memcmp(out, ident + (size_t)h * HOP, sizeof out) != 0)
                bad++;
        }
        CHECK(bad == 0, "policy: a model that copies its input is the identity (equals no model)");
        free(pool);
    }
    CHECK(run_schedule(&sc, N, M_FAIL, &st, mod) == 0 && memcmp(mod, ident, sizeof mod) == 0,
          "policy: a failing model falls open to the identity");
    CHECK(run_schedule(&sc, N, M_HALF_WRITE, &st, mod) == 0 && memcmp(mod, ident, sizeof mod) == 0,
          "policy: a model that leaves bins unwritten (NaN staging) falls open");
    CHECK(run_schedule(&sc, N, M_NAN, &st, mod) == 0 && memcmp(mod, ident, sizeof mod) == 0,
          "policy: a non-finite output bin falls open");
    CHECK(run_schedule(&sc, N, M_SCALE, &st, mod) == 0 && st.saw_nan_prefill, "policy: scale model run; out staging was NaN-filled");
    differs = 0;
    for (h = 2; h < N; ++h)
        if (memcmp(mod + (size_t)h * HOP, ident + (size_t)h * HOP, HOP * sizeof(float)) != 0) differs = 1;
    CHECK(differs, "policy: an accepted model output reaches the synthesis");
}

static void test_model_sees_the_hop(void) {
    enum { N = 12 };
    void* pool;
    UlcnetNnStage* s = make_stage(0, &pool, 0);
    ModelState st;
    UlcnetModel m = make_model(&st);
    Ref far_ref;
    float err_re[BINS], err_im[BINS], far[HOP], fre[BINS], fim[BINS], out[HOP];
    int h, k, bad_far = 0;
    CHECK(s && ref_init(&far_ref) == 0, "inputs: setup");
    if (!s) { free(pool); return; }
    memset(&st, 0, sizeof st);
    st.saw_nan_prefill = 1;
    g_kind = M_COPY;
    CHECK(ulcnet_nn_stage_set_model(s, &m) == 0, "inputs: model installed");
    g_lcg = 321u;
    for (h = 0; h < N; ++h) {
        fill_hop(far, HOP);
        (void)ac_stft_push_planar(far_ref.stft, far, fre, fim);
        for (k = 0; k < BINS; ++k) { err_re[k] = lcg_unit(); err_im[k] = lcg_unit(); }
        st.want_far_re = fre;
        st.want_far_im = fim;
        if (ulcnet_nn_stage_begin(s, err_re, err_im, far, ev(0, 0)) != 0 || ulcnet_nn_stage_process(s, out) != 0)
            bad_far++;
    }
    CHECK(st.infer_calls == N && !st.saw_far_mismatch && bad_far == 0,
          "inputs: the model gets the STFT of the same hop's raw far samples, one call per hop from hop #0");
    CHECK(st.saw_nan_prefill, "inputs: output staging is NaN-filled before every call");
    ref_free(&far_ref);
    free(pool);
}

static void test_reset_and_sequence(void) {
    void* pool;
    UlcnetNnStage* s = make_stage(0, &pool, 0);
    ModelState st;
    UlcnetModel m = make_model(&st);
    float re[BINS], im[BINS], far[HOP], out[HOP], first[HOP];
    UlcnetNnJob job;
    int h, k;
    CHECK(s != NULL, "sequence: setup");
    if (!s) { free(pool); return; }
    memset(&st, 0, sizeof st);
    g_kind = M_COPY;
    CHECK(ulcnet_nn_stage_set_model(s, &m) == 0, "sequence: model installed");
    for (k = 0; k < BINS; ++k) { re[k] = 0.01f * (float)k; im[k] = -0.02f * (float)k; }
    fill_hop(far, HOP);

    CHECK(ulcnet_nn_stage_nn_begin(s, &job) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: nn_begin before begin");
    CHECK(ulcnet_nn_stage_nn_end(s, 0) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: nn_end without an open frame");
    CHECK(ulcnet_nn_stage_end(s, out) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: end before begin");
    CHECK(ulcnet_nn_stage_begin(NULL, re, im, far, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_begin(s, NULL, im, far, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_begin(s, re, NULL, far, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_begin(s, re, im, NULL, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT,
          "sequence: NULL arguments");
    CHECK(ulcnet_nn_stage_begin_time(s, far, far, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT,
          "sequence: the time-domain entry needs a stage built with err_analysis");
    CHECK(st.reset_calls == 0 && st.infer_calls == 0, "sequence: refused calls touch nothing");

    CHECK(ulcnet_nn_stage_begin(s, re, im, far, ev(0, 0)) == 0, "sequence: begin");
    CHECK(ulcnet_nn_stage_begin(s, re, im, far, ev(0, 0)) == ULCNET_NN_STAGE_SEQUENCE_ERROR,
          "sequence: a second begin in the same hop");
    CHECK(ulcnet_nn_stage_nn_begin(s, &job) == 1, "sequence: the frame opens");
    CHECK(ulcnet_nn_stage_nn_begin(s, &job) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: nn_begin while the frame is open");
    CHECK(ulcnet_nn_stage_nn_end(s, 0) == 0, "sequence: nn_end");
    CHECK(ulcnet_nn_stage_nn_begin(s, &job) == 0, "sequence: no second frame in a hop");
    CHECK(ulcnet_nn_stage_end(s, first) == 0, "sequence: end (hop #0)");
    for (k = 0; k < HOP; ++k) if (first[k] != 0.0f) break;
    CHECK(k == HOP, "sequence: hop #0 is zero");

    /* a frame left open, and a frame never fetched, are the identity and leave the stage usable */
    for (h = 0; h < 3; ++h) {
        CHECK(ulcnet_nn_stage_begin(s, re, im, far, ev(0, 0)) == 0, "sequence: begin (open frame case)");
        if (h == 1) CHECK(ulcnet_nn_stage_nn_begin(s, &job) == 1, "sequence: open the frame and leave it");
        CHECK(ulcnet_nn_stage_end(s, out) == 0, "sequence: end with a frame left open or never fetched");
    }
    CHECK(st.infer_calls == 0, "sequence: process() is what calls the model, the phased calls do not");

    /* reset: hop #0 again, the model is reset once, a pending reprime is dropped */
    CHECK(ulcnet_nn_stage_begin(s, re, im, far, ev(1, 1)) == 0 && ulcnet_nn_stage_end(s, out) == 0,
          "reset: a generation change arms the reprime");
    st.reset_calls = 0;
    ulcnet_nn_stage_reset(s);
    CHECK(st.reset_calls == 1, "reset: model reset called once");
    CHECK(ulcnet_nn_stage_begin(s, re, im, far, ev(0, 0)) == 0, "reset: begin after reset");
    CHECK(ulcnet_nn_stage_nn_begin(s, &job) == 1, "reset: the pending reprime was dropped, the model runs");
    CHECK(ulcnet_nn_stage_nn_end(s, 0) == 0 && ulcnet_nn_stage_end(s, out) == 0, "reset: finish the hop");
    for (k = 0; k < HOP; ++k) if (out[k] != 0.0f) break;
    CHECK(k == HOP, "reset: the next hop is hop #0 (zeros)");
    ulcnet_nn_stage_reset(NULL);
    free(pool);
}

static void test_phased_twin_and_pools(void) {
    enum { N = 70 };
    void *pa, *pb, *pc;
    UlcnetNnStage *a = make_stage(0, &pa, 0), *b = make_stage(0, &pb, 0xCD), *c = make_stage(0, &pc, 0);
    ModelState sa, sb, sc_;
    UlcnetModel ma = make_model(&sa), mb = make_model(&sb), mc = make_model(&sc_);
    float re[BINS], im[BINS], far[HOP], oa[HOP], ob[HOP], oc[HOP];
    int h, k, bad_twin = 0, bad_pool = 0;
    CHECK(a && b && c, "twin: setup");
    if (!a || !b || !c) { free(pa); free(pb); free(pc); return; }
    memset(&sa, 0, sizeof sa); memset(&sb, 0, sizeof sb); memset(&sc_, 0, sizeof sc_);
    g_kind = M_SCALE;
    CHECK(ulcnet_nn_stage_set_model(a, &ma) == 0 && ulcnet_nn_stage_set_model(b, &mb) == 0 &&
          ulcnet_nn_stage_set_model(c, &mc) == 0, "twin: models installed");
    g_lcg = 600u;
    for (h = 0; h < N; ++h) {
        UlcnetNnDelay d = ev(h == 20 || h == 45, h >= 20);
        for (k = 0; k < BINS; ++k) { re[k] = lcg_unit(); im[k] = lcg_unit(); }
        fill_hop(far, HOP);
        if (ulcnet_nn_stage_begin(a, re, im, far, d) != 0 || ulcnet_nn_stage_process(a, oa) != 0) bad_twin++;
        if (ulcnet_nn_stage_begin(b, re, im, far, d) != 0 || phased_hop(b, &mb, ob) != 0) bad_twin++;
        if (ulcnet_nn_stage_begin(c, re, im, far, d) != 0 || ulcnet_nn_stage_process(c, oc) != 0) bad_pool++;
        if (memcmp(oa, ob, sizeof oa) != 0) bad_twin++;
        if (memcmp(oa, oc, sizeof oa) != 0) bad_pool++;
    }
    CHECK(bad_twin == 0 && sa.infer_calls == sb.infer_calls && sa.reset_calls == sb.reset_calls,
          "twin: process() and the phased calls agree bit for bit, calls and resets included");
    CHECK(bad_pool == 0, "pools: a stale (0xCD) pool initialises like a zeroed one");
    {
        void* tiny = pool_of(ulcnet_nn_stage_mem_size(0), 0);
        size_t full = ulcnet_nn_stage_mem_size(0);
        CHECK(full > 0 && ulcnet_nn_stage_mem_size(1) > full, "pools: the error STFT adds bytes");
        CHECK(tiny && ulcnet_nn_stage_init(tiny, full - 16, 0) == NULL, "pools: short pool refused");
        CHECK(tiny && ulcnet_nn_stage_init((uint8_t*)tiny + 8, full, 0) == NULL, "pools: misaligned pool refused");
        CHECK(ulcnet_nn_stage_init(NULL, full, 0) == NULL, "pools: NULL pool refused");
        free(tiny);
    }
    free(pa); free(pb); free(pc);
}

static void test_model_gate(void) {
    void* pool;
    UlcnetNnStage* s = make_stage(0, &pool, 0);
    ModelState st;
    UlcnetModel m = make_model(&st);
    UlcnetModelIoDescriptor aligned;
    UlcnetModel bad;
    float re[BINS], im[BINS], far[HOP], out[HOP];
    int k;
    CHECK(s && io_descriptor(), "gate: setup");
    if (!s) { free(pool); return; }
    memset(&st, 0, sizeof st);
    g_kind = M_SCALE;
    CHECK(ulcnet_nn_stage_set_model(NULL, &m) == -1, "gate: NULL stage");
    CHECK(ulcnet_nn_stage_set_model(s, &m) == 0, "gate: a published raw-far descriptor is accepted");
    bad = m;
    bad.io_descriptor = NULL;
    CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a model that infers must publish a descriptor");
    aligned = *io_descriptor();
    aligned.far_input_mode = ULCNET_FAR_ALIGNED;
    bad = m;
    bad.io_descriptor = &aligned;
    CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: an aligned-far descriptor is refused");
    aligned = *io_descriptor();
    aligned.delay_depth = 1;
    bad.io_descriptor = &aligned;
    CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a descriptor with D out of range is refused");
    /* the refused models left the installed one in place */
    for (k = 0; k < BINS; ++k) { re[k] = 0.5f; im[k] = 0.25f; }
    fill_hop(far, HOP);
    CHECK(ulcnet_nn_stage_begin(s, re, im, far, ev(0, 0)) == 0 && ulcnet_nn_stage_process(s, out) == 0 &&
          st.infer_calls == 1, "gate: a refused model leaves the installed one running");
    {   /* an identity model may omit the descriptor */
        UlcnetModel idle;
        memset(&idle, 0, sizeof idle);
        CHECK(ulcnet_nn_stage_set_model(s, &idle) == 0 && ulcnet_nn_stage_set_model(s, NULL) == 0,
              "gate: the all-zero model and NULL install the identity");
        CHECK(ulcnet_nn_stage_begin(s, re, im, far, ev(0, 0)) == 0 && ulcnet_nn_stage_process(s, out) == 0 &&
              st.infer_calls == 1, "gate: the model no longer runs");
    }
    free(pool);
}

int main(void) {
    test_identity_all_entries();
    test_counting_and_events();
    test_model_policy();
    test_model_sees_the_hop();
    test_reset_and_sequence();
    test_phased_twin_and_pools();
    test_model_gate();
    if (g_fail) { printf("%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
