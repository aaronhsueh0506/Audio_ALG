/* Stage-level tests of ulcnet/ulcnet_nn_stage.h: the hop order, the identity paths, the
 * model-call policy (reprime, alignment events, fail-open), the grid and pool plan and the
 * descriptor gate, driven by synthetic alignment events and deterministic models. The stage takes
 * and returns spectra; every test wraps it in the application's own AcStft / AcIstft (struct Rig)
 * and compares the output hops with an independently built STFT/WOLA chain, with a direct-DFT
 * reference. One binary runs every grid of the table below. Build and run: `make test` in
 * pipelines/. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet/ulcnet_nn_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

/* Largest row of the grid table: 48 kHz / fft 1024. */
enum { HOP_MAX = 512, BINS_MAX = 513 };

static const UlcnetGridSpec k_grids[] = {
    { .sample_rate = 16000, .n_fft = 512,  .delay_depth = 8 },
    { .sample_rate = 48000, .n_fft = 1024, .delay_depth = 8 },
};

static UlcnetGrid g_grid;   /* the grid under test */
#define HOP (g_grid.hop)
#define BINS (g_grid.n_bins)

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
    return ulcnet_model_io_descriptor_default(&g_grid, &d) == 0 ? &d : NULL;
}

/* ---- models ---- */

typedef struct ModelState {
    int infer_calls, reset_calls;
    int saw_nan_prefill;      /* every out bin was NaN at the call */
    int saw_far_mismatch;     /* far spectrum differs from the expected one */
    const float* want_far_re; const float* want_far_im;
} ModelState;

typedef enum { M_COPY, M_FAIL, M_HALF_WRITE, M_NAN, M_SCALE, M_BAND } ModelKind;
static ModelKind g_kind;

/* The band mask M_BAND applies: a real gain that varies from bin to bin. */
static float band_gain(int k) { return 0.25f + 0.75f * (float)(k % 9) / 8.0f; }

static int model_infer(void* user, const float* er, const float* ei,
                       const float* fr, const float* fi, float* orr, float* oi) {
    ModelState* st = (ModelState*)user;
    int k, all_nan = 1;
    st->infer_calls++;
    for (k = 0; k < BINS; ++k) if (!isnan(orr[k]) || !isnan(oi[k])) all_nan = 0;
    if (!all_nan) st->saw_nan_prefill = 0;
    if (st->want_far_re && (memcmp(fr, st->want_far_re, sizeof(float) * BINS) != 0 ||
                            memcmp(fi, st->want_far_im, sizeof(float) * BINS) != 0))
        st->saw_far_mismatch = 1;
    switch (g_kind) {
        case M_FAIL:   /* finite garbage written, failure reported: the stage must discard it */
            for (k = 0; k < BINS; ++k) { orr[k] = 2.0f * er[k]; oi[k] = 2.0f * ei[k]; }
            return -1;
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
        case M_BAND:
            for (k = 0; k < BINS; ++k) { orr[k] = band_gain(k) * er[k]; oi[k] = band_gain(k) * ei[k]; }
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
/* (overlap-add synthesis: each hop is this frame's first half plus the previous frame's second
 * half, nothing held back -- hop #0 is already output) */

typedef struct Ref {
    AcStft* stft;
    AcIstft* istft;
    void *m1, *m2;
} Ref;

static int ref_init(Ref* r) {
    AcStftConfig sc;
    size_t b1, b2;
    ac_stft_config_default(&sc, g_grid.n_fft);
    sc.ola = AC_OLA_ACCUMULATE;
    b1 = ac_stft_get_mem_size(&sc);
    b2 = ac_istft_get_mem_size(&sc);
    r->m1 = pool_of(b1, 0);
    r->m2 = pool_of(b2, 0);
    r->stft = r->m1 ? ac_stft_init(r->m1, b1, &sc) : NULL;
    r->istft = r->m2 ? ac_istft_init(r->m2, b2, &sc) : NULL;
    return r->stft && r->istft ? 0 : -1;
}

static void ref_free(Ref* r) { free(r->m1); free(r->m2); }

/* ---- the application side: the stage plus the caller-owned AcStft / AcIstft ---- */

typedef struct Rig {
    UlcnetNnStage* stage;
    void *stage_pool, *m_far, *m_err, *m_syn;
    size_t stage_bytes;
    AcStft *far_stft, *err_stft;
    AcIstft* istft;
    Complex far_spec[BINS_MAX], err_spec[BINS_MAX], out_spec[BINS_MAX];
} Rig;

static int rig_init(Rig* r, int stage_fill) {
    AcStftConfig sc;
    size_t b_stft, b_istft;
    memset(r, 0, sizeof *r);
    r->stage_bytes = ulcnet_nn_stage_mem_size(&g_grid);
    r->stage_pool = pool_of(r->stage_bytes, stage_fill);
    r->stage = r->stage_pool ? ulcnet_nn_stage_init(r->stage_pool, r->stage_bytes, &g_grid) : NULL;
    ac_stft_config_default(&sc, g_grid.n_fft);
    sc.ola = AC_OLA_ACCUMULATE;
    b_stft = ac_stft_get_mem_size(&sc);
    b_istft = ac_istft_get_mem_size(&sc);
    r->m_far = pool_of(b_stft, 0);
    r->m_err = pool_of(b_stft, 0);
    r->m_syn = pool_of(b_istft, 0);
    r->far_stft = r->m_far ? ac_stft_init(r->m_far, b_stft, &sc) : NULL;
    r->err_stft = r->m_err ? ac_stft_init(r->m_err, b_stft, &sc) : NULL;
    r->istft = r->m_syn ? ac_istft_init(r->m_syn, b_istft, &sc) : NULL;
    return r->stage && r->far_stft && r->err_stft && r->istft ? 0 : -1;
}

static void rig_free(Rig* r) { free(r->stage_pool); free(r->m_far); free(r->m_err); free(r->m_syn); }

/* The stage and the application's transforms are reset together. */
static void rig_reset(Rig* r) {
    ulcnet_nn_stage_reset(r->stage);
    ac_stft_reset(r->far_stft);
    ac_stft_reset(r->err_stft);
    ac_istft_reset(r->istft);
}

/* The error comes as a time-domain hop (the mono lane's formed hop) */
static int rig_begin_time(Rig* r, const float* err_hop, const float* far_hop, UlcnetNnDelay d) {
    (void)ac_stft_push(r->far_stft, far_hop, r->far_spec);
    (void)ac_stft_push(r->err_stft, err_hop, r->err_spec);
    return ulcnet_nn_stage_begin_spec(r->stage, r->err_spec, r->far_spec, d);
}

/* The error comes as a spectrum (the 4-channel beam spectrum) */
static int rig_begin_spec(Rig* r, const Complex* err_spec, const float* far_hop, UlcnetNnDelay d) {
    memcpy(r->err_spec, err_spec, sizeof(Complex) * (size_t)BINS);
    (void)ac_stft_push(r->far_stft, far_hop, r->far_spec);
    return ulcnet_nn_stage_begin_spec(r->stage, r->err_spec, r->far_spec, d);
}

static int rig_synth(Rig* r, float* out) { return ac_istft_push(r->istft, r->out_spec, out) == HOP ? 0 : -1; }

static int rig_end(Rig* r, float* out) {
    int rc = ulcnet_nn_stage_end_spec(r->stage, r->out_spec);
    return rc != 0 ? rc : rig_synth(r, out);
}

static int rig_process(Rig* r, float* out) {
    int rc = ulcnet_nn_stage_process(r->stage, r->out_spec);
    return rc != 0 ? rc : rig_synth(r, out);
}

/* one hop through the phased calls with the installed model, like ulcnet_nn_stage_process() */
static int phased_hop(Rig* r, const UlcnetModel* m, float* out) {
    UlcnetNnJob job;
    if (m->infer && ulcnet_nn_stage_nn_begin(r->stage, &job) == 1)
        if (ulcnet_nn_stage_nn_end(r->stage, m->infer(m->user, job.err_re, job.err_im, job.far_re, job.far_im,
                                                      job.out_re, job.out_im)) != 0) return -1;
    return rig_end(r, out);
}

static void fill_hop(float* x, int n) {
    int i;
    for (i = 0; i < n; ++i) x[i] = lcg_unit() * 0.5f;
}

static void fill_spec(Complex* c) {
    int k;
    for (k = 0; k < BINS; ++k) { c[k].r = lcg_unit(); c[k].i = lcg_unit(); }
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

/* ------------------------------------------------------------------------------------- */

static void test_identity_all_entries(void) {
    enum { N = 40 };
    Rig *spec_rig = (Rig*)calloc(1, sizeof(Rig)), *time_rig = (Rig*)calloc(1, sizeof(Rig));
    Ref ref;
    float err[HOP_MAX], far[HOP_MAX], re[BINS_MAX], im[BINS_MAX], want[HOP_MAX];
    float o1[HOP_MAX], o2[HOP_MAX];
    Complex c[BINS_MAX];
    int h, i, bad_spec = 0, bad_time = 0, bad_bits = 0;
    size_t hop_bytes = (size_t)HOP * sizeof(float);
    double energy = 0.0;
    CHECK(spec_rig && time_rig && rig_init(spec_rig, 0xCD) == 0 && rig_init(time_rig, 0xCD) == 0 &&
          ref_init(&ref) == 0, "identity: setup");
    if (!spec_rig || !time_rig || !spec_rig->stage || !time_rig->stage) return;
    g_lcg = 11u;
    for (h = 0; h < N; ++h) {
        fill_hop(err, HOP);
        fill_hop(far, HOP);
        (void)ac_stft_push_planar(ref.stft, err, re, im);
        interleave(c, re, im);
        /* with no model the frame is the identity: the expected hop is the reference WOLA */
        if (ac_istft_push_planar(ref.istft, re, im, want) != HOP) bad_spec++;
        if (rig_begin_spec(spec_rig, c, far, ev(0, 0)) != 0 || rig_process(spec_rig, o1) != 0) bad_spec++;
        if (rig_begin_time(time_rig, err, far, ev(0, 0)) != 0 || rig_process(time_rig, o2) != 0) bad_time++;
        if (memcmp(spec_rig->out_spec, c, sizeof(Complex) * (size_t)BINS) != 0) bad_bits++;
        if (memcmp(time_rig->out_spec, c, sizeof(Complex) * (size_t)BINS) != 0) bad_bits++;
        if (memcmp(o1, want, hop_bytes) != 0) bad_spec++;
        if (memcmp(o2, want, hop_bytes) != 0) bad_time++;
        if (h > 0) for (i = 0; i < HOP; ++i) energy += fabs((double)o1[i]);
    }
    CHECK(bad_bits == 0, "identity: end_spec writes the error spectrum bit for bit when no model ran");
    CHECK(bad_spec == 0, "identity: spectrum entry + caller AcIstft equals the independent STFT/WOLA chain bit for bit");
    CHECK(bad_time == 0, "identity: caller-analysed time-domain error equals it too");
    CHECK(energy > 1.0, "identity: the output carries signal");
    ref_free(&ref);
    rig_free(spec_rig); rig_free(time_rig);
    free(spec_rig); free(time_rig);
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
    Rig* r = (Rig*)calloc(1, sizeof(Rig));
    UlcnetModel m = make_model(st);
    Complex err[BINS_MAX];
    float far[HOP_MAX], out[HOP_MAX];
    int h;
    if (!r || rig_init(r, 0) != 0) { if (r) rig_free(r); free(r); return -1; }
    g_kind = kind;
    memset(st, 0, sizeof *st);
    st->saw_nan_prefill = 1;
    if (ulcnet_nn_stage_set_model(r->stage, &m) != 0) { rig_free(r); free(r); return -1; }
    g_lcg = 5u;
    for (h = 0; h < n; ++h) {
        UlcnetNnDelay d = sched_event(sc, h);
        fill_spec(err);
        fill_hop(far, HOP);
        if (rig_begin_spec(r, err, far, d) != 0 || rig_process(r, out) != 0) {
            rig_free(r); free(r);
            return -1;
        }
        if (outs) memcpy(outs + (size_t)h * HOP, out, (size_t)HOP * sizeof(float));
    }
    rig_free(r); free(r);
    return 0;
}

static void sched_clear(Sched* sc) {
    int k;
    memset(sc, 0, sizeof *sc);
    for (k = 0; k < 8; ++k) sc->changed_at[k] = -1;
    sc->solid_off_at = -1;
    sc->solid_back_at = -1;
}

static void test_counting_and_events(void) {
    enum { N = 60 };
    ModelState st;
    Sched sc;

    /* generation events at hops 10 and 30 (solid from hop 10): two resets, one identity frame each */
    sched_clear(&sc);
    sc.changed_at[0] = 10; sc.changed_at[1] = 30; sc.solid_from = 10;
    CHECK(run_schedule(&sc, N, M_COPY, &st, NULL) == 0, "events: run");
    CHECK(st.infer_calls == N - 2 * ULCNET_NN_STAGE_REPRIME_FRAMES,
          "events: one inference per hop except the reprime frames after each generation change");
    CHECK(st.reset_calls == 2, "events: model reset once per generation change");

    /* no events, solid from the first hop: no reset, no skipped frame */
    sched_clear(&sc);
    sc.solid_from = 0;
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
    static float ident[N * HOP_MAX], mod[N * HOP_MAX];
    size_t run_bytes = (size_t)N * (size_t)HOP * sizeof(float);
    ModelState st;
    Sched sc;
    int h, differs;
    sched_clear(&sc);
    sc.solid_from = 0;
    sc.changed_at[0] = 12;

    CHECK(run_schedule(&sc, N, M_COPY, &st, ident) == 0, "policy: copy model run");
    {   /* the identity reference: no model at all */
        Rig* r = (Rig*)calloc(1, sizeof(Rig));
        Complex err[BINS_MAX];
        float far[HOP_MAX], out[HOP_MAX];
        int bad = 0;
        CHECK(r && rig_init(r, 0) == 0, "policy: identity rig");
        if (r && r->stage) {
            g_lcg = 5u;
            for (h = 0; h < N; ++h) {
                fill_spec(err);
                fill_hop(far, HOP);
                if (rig_begin_spec(r, err, far, sched_event(&sc, h)) != 0 || rig_process(r, out) != 0 ||
                    memcmp(out, ident + (size_t)h * HOP, (size_t)HOP * sizeof(float)) != 0)
                    bad++;
            }
            CHECK(bad == 0, "policy: a model that copies its input is the identity (equals no model)");
        }
        if (r) rig_free(r);
        free(r);
    }
    CHECK(run_schedule(&sc, N, M_FAIL, &st, mod) == 0 && memcmp(mod, ident, run_bytes) == 0,
          "policy: a failing model falls open to the identity");
    CHECK(run_schedule(&sc, N, M_HALF_WRITE, &st, mod) == 0 && memcmp(mod, ident, run_bytes) == 0,
          "policy: a model that leaves bins unwritten (NaN staging) falls open");
    CHECK(run_schedule(&sc, N, M_NAN, &st, mod) == 0 && memcmp(mod, ident, run_bytes) == 0,
          "policy: a non-finite output bin falls open");
    CHECK(run_schedule(&sc, N, M_SCALE, &st, mod) == 0 && st.saw_nan_prefill, "policy: scale model run; out staging was NaN-filled");
    differs = 0;
    for (h = 2; h < N; ++h)
        if (memcmp(mod + (size_t)h * HOP, ident + (size_t)h * HOP, (size_t)HOP * sizeof(float)) != 0) differs = 1;
    CHECK(differs, "policy: an accepted model output reaches the synthesis");
}

/* What end_spec hands the caller: the accepted model spectrum, else the error spectrum, bit for bit. */
static void test_end_spec_content(void) {
    enum { N = 30 };
    static const ModelKind kinds[] = { M_SCALE, M_FAIL, M_NAN, M_HALF_WRITE };
    size_t kd;
    for (kd = 0; kd < sizeof kinds / sizeof kinds[0]; ++kd) {
        Rig* r = (Rig*)calloc(1, sizeof(Rig));
        ModelState st;
        UlcnetModel m = make_model(&st);
        Complex err[BINS_MAX];
        float far[HOP_MAX], out[HOP_MAX];
        int h, k, bad_model = 0, bad_err = 0, n_model = 0, n_err = 0;
        CHECK(r && rig_init(r, 0) == 0, "end_spec: setup");
        if (!r || !r->stage) { if (r) rig_free(r); free(r); return; }
        memset(&st, 0, sizeof st);
        g_kind = kinds[kd];
        CHECK(ulcnet_nn_stage_set_model(r->stage, &m) == 0, "end_spec: model installed");
        g_lcg = 71u;
        for (h = 0; h < N; ++h) {
            UlcnetNnDelay d = ev(h == 12, 1);
            int before = st.infer_calls;
            fill_spec(err);
            fill_hop(far, HOP);
            if (rig_begin_spec(r, err, far, d) != 0 || rig_process(r, out) != 0) { bad_err++; continue; }
            if (st.infer_calls > before && kinds[kd] == M_SCALE) {   /* accepted: 0.5 * err, exactly */
                n_model++;
                for (k = 0; k < BINS; ++k)
                    if (r->out_spec[k].r != 0.5f * err[k].r || r->out_spec[k].i != 0.5f * err[k].i) { bad_model++; break; }
            } else {   /* the reprime hop, or a frame the stage rejected: the error spectrum */
                n_err++;
                if (memcmp(r->out_spec, err, sizeof(Complex) * (size_t)BINS) != 0) bad_err++;
            }
        }
        CHECK(kinds[kd] == M_SCALE ? (n_model == N - ULCNET_NN_STAGE_REPRIME_FRAMES && n_err == ULCNET_NN_STAGE_REPRIME_FRAMES)
                                   : (n_model == 0 && n_err == N), "end_spec: which frames reached the model spectrum");
        CHECK(bad_model == 0, "end_spec: an accepted frame is the model spectrum bin for bin");
        CHECK(bad_err == 0, "end_spec: reprime, failed, non-finite and half-written frames are the error spectrum bit for bit");
        rig_free(r); free(r);
    }
}

static void test_model_sees_the_hop(void) {
    enum { N = 12 };
    Rig* r = (Rig*)calloc(1, sizeof(Rig));
    ModelState st;
    UlcnetModel m = make_model(&st);
    Ref far_ref;
    Complex err[BINS_MAX];
    float far[HOP_MAX], fre[BINS_MAX], fim[BINS_MAX], out[HOP_MAX];
    int h, bad_far = 0;
    CHECK(r && rig_init(r, 0) == 0 && ref_init(&far_ref) == 0, "inputs: setup");
    if (!r || !r->stage) { if (r) rig_free(r); free(r); return; }
    memset(&st, 0, sizeof st);
    st.saw_nan_prefill = 1;
    g_kind = M_COPY;
    CHECK(ulcnet_nn_stage_set_model(r->stage, &m) == 0, "inputs: model installed");
    g_lcg = 321u;
    for (h = 0; h < N; ++h) {
        fill_hop(far, HOP);
        (void)ac_stft_push_planar(far_ref.stft, far, fre, fim);
        fill_spec(err);
        st.want_far_re = fre;
        st.want_far_im = fim;
        if (rig_begin_spec(r, err, far, ev(0, 0)) != 0 || rig_process(r, out) != 0) bad_far++;
    }
    CHECK(st.infer_calls == N && !st.saw_far_mismatch && bad_far == 0,
          "inputs: the model gets the far spectrum the caller handed in, one call per hop from hop #0");
    CHECK(st.saw_nan_prefill, "inputs: output staging is NaN-filled before every call");
    ref_free(&far_ref);
    rig_free(r); free(r);
}

static void test_reset_and_sequence(void) {
    Rig* r = (Rig*)calloc(1, sizeof(Rig));
    void* before = NULL;
    ModelState st;
    UlcnetModel m = make_model(&st);
    Complex e[BINS_MAX], tmp[BINS_MAX];
    float far[HOP_MAX], out[HOP_MAX], first[HOP_MAX];
    UlcnetNnJob job;
    int h, k;
    CHECK(r && rig_init(r, 0) == 0, "sequence: setup");
    if (!r || !r->stage) { if (r) rig_free(r); free(r); return; }
    before = pool_of(r->stage_bytes, 0);
    CHECK(before != NULL, "sequence: snapshot buffer");
    if (!before) { rig_free(r); free(r); return; }
    memset(&st, 0, sizeof st);
    g_kind = M_COPY;
    CHECK(ulcnet_nn_stage_set_model(r->stage, &m) == 0, "sequence: model installed");
    for (k = 0; k < BINS; ++k) { e[k].r = 0.01f * (float)k; e[k].i = -0.02f * (float)k; }
    fill_hop(far, HOP);
    (void)ac_stft_push(r->far_stft, far, r->far_spec);   /* the caller's far spectrum for the direct calls */
    ac_stft_reset(r->far_stft);

    memcpy(before, r->stage_pool, r->stage_bytes);
    CHECK(ulcnet_nn_stage_nn_begin(r->stage, &job) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: nn_begin before begin_spec");
    CHECK(ulcnet_nn_stage_nn_end(r->stage, 0) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: nn_end without an open frame");
    CHECK(ulcnet_nn_stage_end_spec(r->stage, r->out_spec) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: end_spec before begin_spec");
    CHECK(ulcnet_nn_stage_process(r->stage, r->out_spec) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: process before begin_spec");
    CHECK(ulcnet_nn_stage_begin_spec(NULL, e, r->far_spec, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_begin_spec(r->stage, NULL, r->far_spec, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_begin_spec(r->stage, e, NULL, ev(0, 0)) == ULCNET_NN_STAGE_INVALID_ARGUMENT,
          "sequence: NULL arguments");
    CHECK(ulcnet_nn_stage_end_spec(NULL, r->out_spec) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_nn_begin(r->stage, NULL) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_nn_begin(NULL, &job) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_nn_end(NULL, 0) == ULCNET_NN_STAGE_INVALID_ARGUMENT &&
          ulcnet_nn_stage_process(NULL, r->out_spec) == ULCNET_NN_STAGE_INVALID_ARGUMENT,
          "sequence: NULL stage / job arguments");
    CHECK(st.reset_calls == 0 && st.infer_calls == 0, "sequence: refused calls touch nothing");
    CHECK(memcmp(before, r->stage_pool, r->stage_bytes) == 0, "sequence: refused calls leave the stage block bytes unchanged");

    CHECK(ulcnet_nn_stage_begin_spec(r->stage, e, r->far_spec, ev(0, 0)) == 0, "sequence: begin_spec");
    CHECK(ulcnet_nn_stage_end_spec(r->stage, NULL) == ULCNET_NN_STAGE_INVALID_ARGUMENT,
          "sequence: end_spec with a NULL output is refused and leaves the hop open");
    CHECK(ulcnet_nn_stage_begin_spec(r->stage, e, r->far_spec, ev(0, 0)) == ULCNET_NN_STAGE_SEQUENCE_ERROR,
          "sequence: a second begin_spec in the same hop");
    CHECK(ulcnet_nn_stage_nn_begin(r->stage, &job) == 1, "sequence: the frame opens");
    CHECK(ulcnet_nn_stage_nn_begin(r->stage, &job) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: nn_begin while the frame is open");
    CHECK(ulcnet_nn_stage_begin_spec(r->stage, e, r->far_spec, ev(0, 0)) == ULCNET_NN_STAGE_SEQUENCE_ERROR,
          "sequence: begin_spec while the frame is open");
    CHECK(ulcnet_nn_stage_nn_end(r->stage, 0) == 0, "sequence: nn_end");
    CHECK(ulcnet_nn_stage_nn_end(r->stage, 0) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: a second nn_end");
    CHECK(ulcnet_nn_stage_nn_begin(r->stage, &job) == 0, "sequence: no second frame in a hop");
    CHECK(rig_end(r, first) == 0, "sequence: end_spec + synthesis (hop #0)");
    CHECK(ulcnet_nn_stage_end_spec(r->stage, tmp) == ULCNET_NN_STAGE_SEQUENCE_ERROR, "sequence: a second end_spec in the same hop");
    {   /* hop #0 is real output: this spectrum's first synthesis half over a zero history */
        double en = 0.0;
        for (k = 0; k < HOP; ++k) en += fabs((double)first[k]);
        CHECK(en > 1e-3, "sequence: hop #0 carries the first half of the first frame's synthesis");
    }
    /* a frame left open, and a frame never fetched, are the identity and leave the stage usable */
    for (h = 0; h < 3; ++h) {
        CHECK(rig_begin_spec(r, e, far, ev(0, 0)) == 0, "sequence: begin_spec (open frame case)");
        if (h == 1) CHECK(ulcnet_nn_stage_nn_begin(r->stage, &job) == 1, "sequence: open the frame and leave it");
        CHECK(rig_end(r, out) == 0, "sequence: end_spec with a frame left open or never fetched");
        CHECK(memcmp(r->out_spec, e, sizeof(Complex) * (size_t)BINS) == 0, "sequence: that frame is the error spectrum");
    }
    CHECK(st.infer_calls == 0, "sequence: process() is what calls the model, the phased calls do not");

    /* reset: hop #0 again, the model is reset once, a pending reprime is dropped */
    CHECK(rig_begin_spec(r, e, far, ev(1, 1)) == 0 && rig_end(r, out) == 0,
          "reset: a generation change arms the reprime");
    st.reset_calls = 0;
    rig_reset(r);
    CHECK(st.reset_calls == 1, "reset: model reset called once");
    CHECK(rig_begin_spec(r, e, far, ev(0, 0)) == 0, "reset: begin_spec after reset");
    CHECK(ulcnet_nn_stage_nn_begin(r->stage, &job) == 1, "reset: the pending reprime was dropped, the model runs");
    CHECK(ulcnet_nn_stage_nn_end(r->stage, 0) == 0 && rig_end(r, out) == 0, "reset: finish the hop");
    CHECK(memcmp(out, first, (size_t)HOP * sizeof(float)) == 0,
          "reset: the next hop is hop #0 again (same input, same output as the first hop of a new stage)");

    /* the stage keeps no transform history: its own reset leaves the caller's AcIstft overlap alone */
    st.reset_calls = 0;
    ulcnet_nn_stage_reset(r->stage);
    CHECK(st.reset_calls == 1, "reset: stage-only reset calls the model reset");
    ac_stft_reset(r->far_stft);
    CHECK(rig_begin_spec(r, e, far, ev(0, 0)) == 0 && ulcnet_nn_stage_nn_begin(r->stage, &job) == 1 &&
          ulcnet_nn_stage_nn_end(r->stage, 0) == 0 && rig_end(r, out) == 0, "reset: hop after a stage-only reset");
    CHECK(memcmp(out, first, (size_t)HOP * sizeof(float)) != 0,
          "reset: without the caller's AcIstft reset the overlap of the previous frame still shows (the caller owns it)");
    ulcnet_nn_stage_reset(NULL);
    free(before);
    rig_free(r); free(r);
}

static void test_phased_twin_and_pools(void) {
    enum { N = 70 };
    Rig *a = (Rig*)calloc(1, sizeof(Rig)), *b = (Rig*)calloc(1, sizeof(Rig)), *c = (Rig*)calloc(1, sizeof(Rig));
    ModelState sa, sb, sc_;
    UlcnetModel ma = make_model(&sa), mb = make_model(&sb), mc = make_model(&sc_);
    Complex e[BINS_MAX];
    float far[HOP_MAX], oa[HOP_MAX], ob[HOP_MAX], oc[HOP_MAX];
    size_t hop_bytes = (size_t)HOP * sizeof(float);
    int h, bad_twin = 0, bad_pool = 0;
    CHECK(a && b && c && rig_init(a, 0) == 0 && rig_init(b, 0) == 0 && rig_init(c, 0xCD) == 0, "twin: setup");
    if (!a || !b || !c || !a->stage || !b->stage || !c->stage) {
        if (a) rig_free(a);
        if (b) rig_free(b);
        if (c) rig_free(c);
        free(a); free(b); free(c);
        return;
    }
    memset(&sa, 0, sizeof sa); memset(&sb, 0, sizeof sb); memset(&sc_, 0, sizeof sc_);
    g_kind = M_SCALE;
    CHECK(ulcnet_nn_stage_set_model(a->stage, &ma) == 0 && ulcnet_nn_stage_set_model(b->stage, &mb) == 0 &&
          ulcnet_nn_stage_set_model(c->stage, &mc) == 0, "twin: models installed");
    g_lcg = 600u;
    for (h = 0; h < N; ++h) {
        UlcnetNnDelay d = ev(h == 20 || h == 45, h >= 20);
        fill_spec(e);
        fill_hop(far, HOP);
        if (rig_begin_spec(a, e, far, d) != 0 || rig_process(a, oa) != 0) bad_twin++;
        if (rig_begin_spec(b, e, far, d) != 0 || phased_hop(b, &mb, ob) != 0) bad_twin++;
        if (rig_begin_spec(c, e, far, d) != 0 || rig_process(c, oc) != 0) bad_pool++;
        if (memcmp(oa, ob, hop_bytes) != 0) bad_twin++;
        if (memcmp(oa, oc, hop_bytes) != 0) bad_pool++;
    }
    CHECK(bad_twin == 0 && sa.infer_calls == sb.infer_calls && sa.reset_calls == sb.reset_calls,
          "twin: process() and the phased calls agree bit for bit, calls and resets included");
    CHECK(bad_pool == 0, "pools: a stale (0xCD) pool initialises like a zeroed one");
    rig_free(a); rig_free(b); rig_free(c);
    free(a); free(b); free(c);
}

/* The grid and the pool plan: what the stage refuses, that a refusal leaves the block bytes
 * alone, and that the plan is exactly what init carves. */
static void test_grid_and_pool_plan(void) {
    size_t full = ulcnet_nn_stage_mem_size(&g_grid);
    UlcnetGrid bad;
    UlcnetNnStage* s;
    int variant;

    printf("  stage pool %zu bytes (%d Hz / fft %d)\n", full, g_grid.sample_rate, g_grid.n_fft);
    CHECK(full > 0, "pools: a valid grid has a pool");
    CHECK(full % 16 == 0, "pools: the plan is a multiple of 16 bytes");
    CHECK(full < 6u * (size_t)BINS * sizeof(float) + 512, "pools: the plan is the six spectra plus the state (no transform objects)");

    /* invalid grids are sized 0 and refused by init */
    CHECK(ulcnet_nn_stage_mem_size(NULL) == 0, "grid: NULL grid has no pool");
    bad = g_grid; bad.sample_rate = 0;
    CHECK(ulcnet_nn_stage_mem_size(&bad) == 0, "grid: sample_rate 0 has no pool");
    bad = g_grid; bad.n_fft = 500;
    CHECK(ulcnet_nn_stage_mem_size(&bad) == 0, "grid: an n_fft that is not a power of two has no pool");
    bad = g_grid; bad.hop = g_grid.hop + 1;
    CHECK(ulcnet_nn_stage_mem_size(&bad) == 0, "grid: an inconsistent hop has no pool");
    bad = g_grid; bad.n_bins = g_grid.n_bins - 1;
    CHECK(ulcnet_nn_stage_mem_size(&bad) == 0, "grid: an inconsistent n_bins has no pool");
    bad = g_grid; bad.delay_depth = ULCNET_MODEL_IO_MAX_D + 1;
    CHECK(ulcnet_nn_stage_mem_size(&bad) == 0, "grid: a delay depth out of range has no pool");

    /* a refused init leaves the block bytes exactly as they were */
    for (variant = 0; variant < 6; ++variant) {
        void* raw = pool_of(full + 32, 0xCD);
        void* snap = pool_of(full + 32, 0);
        uint8_t* base = (uint8_t*)raw;
        const UlcnetGrid* grid = &g_grid;
        size_t bytes = full;
        UlcnetNnStage* r;
        if (!raw || !snap) { CHECK(0, "pools: refusal setup"); free(raw); free(snap); continue; }
        memcpy(snap, raw, full + 32);
        if (variant == 0) bytes = full - 16;
        if (variant == 1) base += 8;
        if (variant == 2) grid = NULL;
        if (variant == 3) { bad = g_grid; bad.n_fft = 500; grid = &bad; }
        if (variant == 4) base = NULL;
        if (variant == 5) { bad = g_grid; bad.delay_depth = 1; grid = &bad; }
        r = ulcnet_nn_stage_init(base, bytes, grid);
        CHECK(r == NULL, "pools: short, misaligned, NULL and invalid-grid init is refused");
        CHECK(memcmp(snap, raw, full + 32) == 0, "pools: a refused init leaves the block bytes unchanged");
        free(raw); free(snap);
    }

    /* the plan is exact: a stage placed at the end of a poisoned block never writes past it,
     * and the last bytes of the plan are written */
    {
        uint8_t* raw = (uint8_t*)pool_of(full + 64, 0xA5);
        size_t hw = 0, i;
        Complex e[BINS_MAX], f[BINS_MAX], o[BINS_MAX];
        ModelState st;
        UlcnetModel m = make_model(&st);
        CHECK(raw != NULL, "pools: exactness setup");
        if (raw) {
            s = ulcnet_nn_stage_init(raw, full, &g_grid);
            CHECK(s != NULL, "pools: the exact block initialises");
            if (s) {
                int h;
                memset(&st, 0, sizeof st);
                g_kind = M_SCALE;
                CHECK(ulcnet_nn_stage_set_model(s, &m) == 0, "pools: exactness model");
                g_lcg = 99u;
                for (h = 0; h < 6; ++h) {
                    fill_spec(e);
                    fill_spec(f);
                    (void)ulcnet_nn_stage_begin_spec(s, e, f, ev(0, 0));
                    (void)ulcnet_nn_stage_process(s, o);
                }
            }
            for (i = 0; i < full; ++i) if (raw[i] != 0xA5) hw = i + 1;
            for (i = full; i < full + 64; ++i) if (raw[i] != 0xA5) break;
            CHECK(i == full + 64, "pools: nothing past the planned bytes is written");
            CHECK(hw + 16 > full, "pools: the plan has no unused tail (within one 16-byte slot)");
            free(raw);
        }
    }

    /* the stage keeps its own copy of the grid */
    {
        void* pool;
        UlcnetGrid copy = g_grid;
        size_t bytes = ulcnet_nn_stage_mem_size(&copy);
        pool = pool_of(bytes, 0);
        s = pool ? ulcnet_nn_stage_init(pool, bytes, &copy) : NULL;
        copy.n_fft = 0;
        CHECK(s && ulcnet_nn_stage_grid(s) && memcmp(ulcnet_nn_stage_grid(s), &g_grid, sizeof g_grid) == 0,
              "grid: the stage keeps its own copy of the grid");
        CHECK(ulcnet_nn_stage_grid(NULL) == NULL, "grid: NULL stage has no grid");
        free(pool);
    }
}

static void test_model_gate(void) {
    Rig* r = (Rig*)calloc(1, sizeof(Rig));
    UlcnetNnStage* s;
    ModelState st;
    UlcnetModel m = make_model(&st);
    UlcnetModelIoDescriptor aligned;
    UlcnetModel bad;
    Complex e[BINS_MAX];
    float far[HOP_MAX], out[HOP_MAX];
    int k;
    CHECK(r && rig_init(r, 0) == 0 && io_descriptor(), "gate: setup");
    if (!r || !r->stage) { if (r) rig_free(r); free(r); return; }
    s = r->stage;
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
    {   /* a descriptor of another grid is refused, field by field */
        UlcnetModelIoDescriptor other;
        UlcnetGrid og;
        UlcnetGridSpec spec = k_grids[0];
        if (g_grid.n_fft == k_grids[0].n_fft) spec = k_grids[1];
        CHECK(ulcnet_grid_make(&og, &spec) == ULCNET_GRID_OK &&
              ulcnet_model_io_descriptor_default(&og, &other) == 0, "gate: the other grid's descriptor builds");
        bad = m;
        bad.io_descriptor = &other;
        CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a descriptor of the other grid is refused");
        aligned = *io_descriptor();
        aligned.sample_rate = other.sample_rate;
        bad.io_descriptor = &aligned;
        CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a descriptor with another sample rate is refused");
        aligned = *io_descriptor();
        aligned.fft_size = other.fft_size;
        bad.io_descriptor = &aligned;
        CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a descriptor with another fft size is refused");
        aligned = *io_descriptor();
        aligned.hop_size += 1;
        bad.io_descriptor = &aligned;
        CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a descriptor with another hop is refused");
        aligned = *io_descriptor();
        aligned.spectrum_bins += 1;
        bad.io_descriptor = &aligned;
        CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a descriptor with another bin count is refused");
    }
    {   /* a packet descriptor (layout 19, N > 1) validates against the grid but is not a one-frame graph: refused */
        UlcnetModelIoDescriptor packet;
        static const int counts[] = { 2, 8, ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL };
        size_t c;
        for (c = 0; c < sizeof counts / sizeof counts[0]; ++c) {
            CHECK(ulcnet_model_io_descriptor_default_packet(&g_grid, counts[c], &packet) == 0 &&
                  packet.layout_version == ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION &&
                  ulcnet_model_io_descriptor_validate(&g_grid, &packet) == 0 &&
                  ulcnet_model_io_descriptor_frames(&packet) == counts[c],
                  "gate: a layout-19 descriptor builds and validates for N = 2, 8 and the maximum");
            bad = m;
            bad.io_descriptor = &packet;
            CHECK(ulcnet_nn_stage_set_model(s, &bad) == -1, "gate: a packet descriptor (layout 19, N > 1) is refused");
        }
    }
    /* the refused models left the installed one in place */
    for (k = 0; k < BINS; ++k) { e[k].r = 0.5f; e[k].i = 0.25f; }
    fill_hop(far, HOP);
    CHECK(rig_begin_spec(r, e, far, ev(0, 0)) == 0 && rig_process(r, out) == 0 &&
          st.infer_calls == 1, "gate: a refused model leaves the installed one running");
    {   /* an identity model may omit the descriptor */
        UlcnetModel idle;
        memset(&idle, 0, sizeof idle);
        CHECK(ulcnet_nn_stage_set_model(s, &idle) == 0 && ulcnet_nn_stage_set_model(s, NULL) == 0,
              "gate: the all-zero model and NULL install the identity");
        CHECK(rig_begin_spec(r, e, far, ev(0, 0)) == 0 && rig_process(r, out) == 0 &&
              st.infer_calls == 1, "gate: the model no longer runs");
    }
    rig_free(r); free(r);
}


/* ---- independent per-sample accumulate reference: direct inverse DFT in double, own window ---- */

typedef struct AccRef {
    double saved[HOP_MAX];   /* second half of the previous frame's windowed IFFT */
    double win[2 * HOP_MAX], cs[2 * HOP_MAX], sn[2 * HOP_MAX];
} AccRef;

static void accref_init(AccRef* a) {
    int n, N = g_grid.n_fft;
    const double pi = 3.14159265358979323846;
    memset(a, 0, sizeof *a);
    for (n = 0; n < N; ++n) {
        a->win[n] = sqrt(0.5 * (1.0 - cos(2.0 * pi * (double)n / (double)N)));   /* periodic sqrt-Hann */
        a->cs[n] = cos(2.0 * pi * (double)n / (double)N);
        a->sn[n] = sin(2.0 * pi * (double)n / (double)N);
    }
}

/* hop = frame's first half + saved; saved = frame's second half. The spectrum is unnormalised
 * (the inverse carries 1/N); DC and Nyquist take their real parts. */
static void accref_hop(AccRef* a, const float* re, const float* im, float* out) {
    int n, k, N = g_grid.n_fft, H = HOP;
    double y[2 * HOP_MAX];
    for (n = 0; n < N; ++n) {
        double acc = (double)re[0] + ((n & 1) ? -1.0 : 1.0) * (double)re[N / 2];
        for (k = 1; k < N / 2; ++k) {
            int idx = (int)(((long long)k * n) % N);
            acc += 2.0 * ((double)re[k] * a->cs[idx] - (double)im[k] * a->sn[idx]);
        }
        y[n] = a->win[n] * acc / (double)N;
    }
    for (n = 0; n < H; ++n) {
        out[n] = (float)(y[n] + a->saved[n]);
        a->saved[n] = y[H + n];
    }
}

/* Drives a time-domain-error rig with the band-mask model and, hop for hop, an independent
 * analysis + (model applied where the stage actually ran it) + (a) the audiolib accumulate chain
 * and (b) the direct-DFT accumulate reference. replay != 0 resets the rig (stage and the caller's
 * transforms) and the references after hop n and feeds the same hops again; both passes are compared
 * with the references, and the second pass bit for bit with the first. */
typedef struct RefRun {
    double max_err;            /* worst |stage - direct-DFT reference| over every hop */
    double peak0;              /* peak |hop #0| */
    int chain_bad;             /* hops not bit-equal to the audiolib accumulate chain */
    int replay_bad;            /* hops of the replay not bit-equal to the first pass */
    int infer_calls[2], reset_calls[2];
    int inferred_at[2][96];
} RefRun;

static int run_vs_reference(const Sched* sc, int n, int replay, RefRun* r) {
    Rig* rig = (Rig*)calloc(1, sizeof(Rig));
    ModelState st;
    UlcnetModel m = make_model(&st);
    static float first_pass[96 * HOP_MAX];
    Ref ref;
    AccRef acc;
    float err[HOP_MAX], far[HOP_MAX], out[HOP_MAX], want[HOP_MAX], dft[HOP_MAX];
    float re[BINS_MAX], im[BINS_MAX], mre[BINS_MAX], mim[BINS_MAX];
    int pass, h, k, passes = replay ? 2 : 1;
    memset(r, 0, sizeof *r);
    if (!rig || n > 96 || rig_init(rig, 0) != 0 || ref_init(&ref) != 0) {
        if (rig) rig_free(rig);
        free(rig);
        return -1;
    }
    accref_init(&acc);
    memset(&st, 0, sizeof st);
    g_kind = M_BAND;
    if (ulcnet_nn_stage_set_model(rig->stage, &m) != 0) { rig_free(rig); free(rig); ref_free(&ref); return -1; }
    for (pass = 0; pass < passes; ++pass) {
        if (pass == 1) {
            rig_reset(rig);
            ref_free(&ref);
            if (ref_init(&ref) != 0) { rig_free(rig); free(rig); return -1; }
            accref_init(&acc);
            r->infer_calls[0] = st.infer_calls; r->reset_calls[0] = st.reset_calls;
            st.infer_calls = 0; st.reset_calls = 0;
        }
        g_lcg = 77u;
        for (h = 0; h < n; ++h) {
            int before = st.infer_calls, ran;
            fill_hop(err, HOP);
            fill_hop(far, HOP);
            if (rig_begin_time(rig, err, far, sched_event(sc, h)) != 0 ||
                rig_process(rig, out) != 0) { rig_free(rig); free(rig); ref_free(&ref); return -1; }
            ran = st.infer_calls > before;
            r->inferred_at[pass][h] = ran;
            (void)ac_stft_push_planar(ref.stft, err, re, im);
            if (ran) for (k = 0; k < BINS; ++k) { mre[k] = band_gain(k) * re[k]; mim[k] = band_gain(k) * im[k]; }
            else memcpy(mre, re, sizeof(float) * BINS), memcpy(mim, im, sizeof(float) * BINS);
            if (ac_istft_push_planar(ref.istft, mre, mim, want) != HOP ||
                memcmp(want, out, (size_t)HOP * sizeof(float)) != 0) r->chain_bad++;
            accref_hop(&acc, mre, mim, dft);
            for (k = 0; k < HOP; ++k) {
                double d = fabs((double)out[k] - (double)dft[k]);
                if (d > r->max_err) r->max_err = d;
            }
            if (pass == 0 && h == 0)
                for (k = 0; k < HOP; ++k) if (fabs((double)out[k]) > r->peak0) r->peak0 = fabs((double)out[k]);
            if (pass == 0) memcpy(first_pass + (size_t)h * HOP, out, (size_t)HOP * sizeof(float));
            else if (memcmp(first_pass + (size_t)h * HOP, out, (size_t)HOP * sizeof(float)) != 0) r->replay_bad++;
        }
    }
    r->infer_calls[passes - 1] = st.infer_calls; r->reset_calls[passes - 1] = st.reset_calls;
    ref_free(&ref);
    rig_free(rig); free(rig);
    return 0;
}

static void test_accumulate_reference(void) {
    enum { N = 40 };
    RefRun r;
    Sched sc;
    int h, ev_ok;
    const double tol = 2e-6;

    /* first hop and steady state: solid from hop #0, no event -- one inference on every hop,
     * including the first, and a first hop that is real output */
    sched_clear(&sc);
    sc.solid_from = 0;
    CHECK(run_vs_reference(&sc, N, 0, &r) == 0, "accumulate: first-hop run");
    printf("  accumulate reference: first hop peak %.6f, worst |stage - direct DFT| %.3g over %d hops\n",
           r.peak0, r.max_err, N);
    CHECK(r.peak0 > 0.01, "accumulate: hop #0 is real output (peak well above zero)");
    CHECK(r.max_err < tol, "accumulate: every hop equals the direct-DFT per-sample accumulate reference");
    CHECK(r.chain_bad == 0, "accumulate: every hop equals the audiolib accumulate chain bit for bit");
    CHECK(r.infer_calls[0] == N && r.reset_calls[0] == 0 && r.inferred_at[0][0] == 1,
          "accumulate: the first hop sends the model one frame; one inference per hop after it");

    /* mid-stream events: a generation change (hop 10) and solid dropping then rising again (20..30).
     * The event hops run the identity, the overlap of the previous frame carries through them. */
    sched_clear(&sc);
    sc.solid_from = 0; sc.solid_off_at = 20; sc.solid_back_at = 30;
    sc.changed_at[0] = 10;
    CHECK(run_vs_reference(&sc, N, 0, &r) == 0, "events: run");
    CHECK(r.max_err < tol && r.chain_bad == 0,
          "events: PCM through a generation change and a relock equals the continuous accumulate reference");
    CHECK(r.reset_calls[0] == 2 && r.infer_calls[0] == N - 2 * ULCNET_NN_STAGE_REPRIME_FRAMES,
          "events: one model reset and no inference on each event hop");
    ev_ok = 1;
    for (h = 10; h < 10 + ULCNET_NN_STAGE_REPRIME_FRAMES; ++h) if (r.inferred_at[0][h]) ev_ok = 0;
    for (h = 30; h < 30 + ULCNET_NN_STAGE_REPRIME_FRAMES; ++h) if (r.inferred_at[0][h]) ev_ok = 0;
    CHECK(ev_ok && r.inferred_at[0][10 + ULCNET_NN_STAGE_REPRIME_FRAMES] == 1,
          "events: the event hop bypasses the model and the next hop runs it");

    /* full reset (stage plus the caller's transforms) after non-zero overlap: replaying the same hops
     * equals the first pass bit for bit and equals references restarted from zero */
    sched_clear(&sc);
    sc.solid_from = 0;
    sc.changed_at[0] = 6;
    CHECK(run_vs_reference(&sc, 24, 1, &r) == 0, "reset: replay run");
    CHECK(r.max_err < tol && r.chain_bad == 0, "reset: both passes equal the references (the second restarted from zero)");
    CHECK(r.replay_bad == 0, "reset: the replay after the stage and transform reset equals the first pass bit for bit");
    CHECK(r.infer_calls[0] == r.infer_calls[1] && r.reset_calls[1] >= 1 + 0 && r.inferred_at[1][0] == r.inferred_at[0][0],
          "reset: the replay sends the model the same frames as a new stage");
}

int main(void) {
    size_t g;
    for (g = 0; g < sizeof k_grids / sizeof k_grids[0]; ++g) {
        if (ulcnet_grid_make(&g_grid, &k_grids[g]) != ULCNET_GRID_OK) {
            printf("FAIL: grid %zu does not build\n", g);
            return 1;
        }
        CHECK(g_grid.hop <= HOP_MAX && g_grid.n_bins <= BINS_MAX, "the table row fits the test buffers");
        printf("grid %d Hz / fft %d / hop %d / %d bins\n", g_grid.sample_rate, g_grid.n_fft,
               g_grid.hop, g_grid.n_bins);
        test_identity_all_entries();
        test_accumulate_reference();
        test_counting_and_events();
        test_model_policy();
        test_end_spec_content();
        test_model_sees_the_hop();
        test_reset_and_sequence();
        test_phased_twin_and_pools();
        test_grid_and_pool_plan();
        test_model_gate();
    }
    if (g_fail) { printf("%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
