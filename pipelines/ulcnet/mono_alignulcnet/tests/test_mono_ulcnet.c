/**
 * Composition tests of the mono linear AEC feeding the Align-ULCNet stage (lib/aec's Aec +
 * ulcnet/ulcnet_nn_stage.h), wired as ulcnet/mono_alignulcnet/main.c wires them: one-hop STFT/WOLA
 * latency, delay-event identity reprime on the real alignment state and the model-callback
 * policy on real framing. A contract test, not a DSP-quality test. The stage alone (synthetic
 * alignment events) is tested in ulcnet/tests/test_ulcnet_nn_stage.c; what needs the real AEC
 * (FIXED and MATCHED delay states, the framing straddle derivation, acquisition of a known
 * delay) stays here.
 * Stimuli: a zero-delay echo (the delay state stays UNLOCKED) and a bulk-delay echo
 * (ECHO_DELAY is beyond the filter reach, so the matched filter locks and a model is
 * applied). Each test banner states its case. Build and run: `make test` in pipelines/.
 */
#include "aec.h"
#include "ulcnet/ulcnet_nn_stage.h"
#include "utility/mem_align.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <time.h>
#include <math.h>

/* Every test model that actually infers must publish a descriptor: the
 * pipeline refuses infer-without-descriptor, because a shape disagreement
 * between the graph and the host-side rings is undetectable downstream (the
 * finite guard catches an unwritten output, never a wrong-shaped one). One
 * shared descriptor for the whole file; D is the example default. */
static const UlcnetModelIoDescriptor* test_io_descriptor(void) {
    static UlcnetModelIoDescriptor d;
    static int ready = 0;
    if (!ready) {
        if (ulcnet_model_io_descriptor_default(8, &d) != 0) return NULL;
        ready = 1;
    }
    return &d;
}

/* ---------------------------------------------------------------------------------------
 * The composition under test: Aec (linear context only) + UlcnetNnStage, as main.c builds
 * them. A small test-local shim gives the pieces the shape the tests below drive.
 * --------------------------------------------------------------------------------------- */

typedef struct MonoUlcnetConfig {
    int sample_rate, fft_size, filter_length;
    AecDelayMode delay_mode;
    int delay_num_filters, fixed_delay_samples;
    AecPreset aec_preset;
    UlcnetModel model;            /* all-zero: identity */
} MonoUlcnetConfig;

typedef struct MonoUlcnetMemReq { uint64_t bytes; } MonoUlcnetMemReq;

typedef struct MonoUlcnet {
    Aec* aec;
    UlcnetNnStage* stage;
    Aec* heap_aec;                /* separate-allocation construction only */
    void* heap_stage_mem;
} MonoUlcnet;

static MonoUlcnetConfig mu_default_config(int sample_rate) {
    MonoUlcnetConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = ULCNET_N_FFT;           /* compiled checkpoint grid */
    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = 5;
    cfg.fixed_delay_samples = -1;
    cfg.aec_preset = AEC_PRESET_BALANCED;
    return cfg;
}

/* The lane on the compiled grid, linear context only; -1 for a request off that grid or an
 * out-of-enum preset (this harness's policy; libaec refuses the rest when sizing). */
static int mu_lane(const MonoUlcnetConfig* cfg, AecConfig* aec_cfg) {
    if (!cfg || cfg->sample_rate != ULCNET_SR ||
        (cfg->fft_size != 0 && cfg->fft_size != ULCNET_N_FFT))
        return -1;
    if (cfg->aec_preset != AEC_PRESET_MILD && cfg->aec_preset != AEC_PRESET_BALANCED &&
        cfg->aec_preset != AEC_PRESET_AGGRESSIVE && cfg->aec_preset != AEC_PRESET_NO_FLOOR)
        return -1;
    aec_config_from_preset(aec_cfg, cfg->aec_preset, ULCNET_SR);
    aec_cfg->fft_size = ULCNET_N_FFT;
    if (cfg->filter_length != 0) aec_cfg->filter_length = cfg->filter_length;
    aec_cfg->delay_mode = cfg->delay_mode;
    aec_cfg->delay_num_filters = cfg->delay_num_filters;
    aec_cfg->fixed_delay_samples = cfg->fixed_delay_samples;
    aec_cfg->enable_delay_est = cfg->delay_mode == AEC_DELAY_MATCHED;
    aec_cfg->enable_res = 0;
    aec_cfg->return_res_context = 1;
    aec_cfg->spatial_linear_context = AEC_CONTEXT_LINEAR_ONLY;
    return 0;
}

/* Pool bytes of the AEC and the stage (with its error analysis), 0 for a refused config. */
static int mu_get_mem_requirements(const MonoUlcnetConfig* cfg, MonoUlcnetMemReq* req) {
    AecConfig aec_cfg;
    size_t aec_bytes, stage_bytes = ulcnet_nn_stage_mem_size(1);
    if (!req || mu_lane(cfg, &aec_cfg) != 0) return -1;
    aec_bytes = aec_get_mem_size(&aec_cfg);
    if (aec_bytes == 0 || stage_bytes == 0) return -1;
    req->bytes = (uint64_t)(ALIGN16(aec_bytes) + ALIGN16(stage_bytes));
    return 0;
}

static MonoUlcnet* mu_wrap(Aec* aec, void* stage_mem, size_t stage_bytes, const MonoUlcnetConfig* cfg,
                           MonoUlcnet* p) {
    p->aec = aec;
    p->stage = aec ? ulcnet_nn_stage_init(stage_mem, stage_bytes, 1) : NULL;
    if (!p->stage || ulcnet_nn_stage_set_model(p->stage, &cfg->model) != 0) return NULL;
    return p;
}

/* A fresh instance in one block of `bytes` at `pool` (AEC first, then the stage). */
static MonoUlcnet* mu_init(void* pool, size_t bytes, const MonoUlcnetConfig* cfg) {
    AecConfig aec_cfg;
    MonoUlcnetMemReq req;
    MonoUlcnet* p;
    size_t aec_bytes;
    if (!pool || mu_get_mem_requirements(cfg, &req) != 0 || bytes < req.bytes) return NULL;
    if (mu_lane(cfg, &aec_cfg) != 0) return NULL;
    aec_bytes = aec_get_mem_size(&aec_cfg);
    p = (MonoUlcnet*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    if (!mu_wrap(aec_init(pool, aec_bytes, &aec_cfg), (uint8_t*)pool + ALIGN16(aec_bytes),
                 ulcnet_nn_stage_mem_size(1), cfg, p)) {
        free(p);
        return NULL;
    }
    return p;
}

/* A fresh instance from separate allocations (the heap path). */
static MonoUlcnet* mu_create(const MonoUlcnetConfig* cfg) {
    AecConfig aec_cfg;
    MonoUlcnet* p;
    size_t stage_bytes = ulcnet_nn_stage_mem_size(1);
    if (mu_lane(cfg, &aec_cfg) != 0) return NULL;
    p = (MonoUlcnet*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->heap_aec = (Aec*)calloc(1, sizeof(Aec));
    if (!p->heap_aec || aec_create(p->heap_aec, &aec_cfg) != 0 ||
        posix_memalign(&p->heap_stage_mem, 16, stage_bytes) != 0) {
        p->heap_stage_mem = NULL;
        free(p->heap_aec);
        free(p);
        return NULL;
    }
    memset(p->heap_stage_mem, 0x5A, stage_bytes);
    if (!mu_wrap(p->heap_aec, p->heap_stage_mem, stage_bytes, cfg, p)) {
        aec_destroy(p->heap_aec);
        free(p->heap_aec);
        free(p->heap_stage_mem);
        free(p);
        return NULL;
    }
    return p;
}

static void mu_destroy(MonoUlcnet* p) {
    if (!p) return;
    if (p->aec) aec_destroy(p->aec);
    free(p->heap_aec);
    free(p->heap_stage_mem);
    free(p);
}

static int mu_hop_size(const MonoUlcnet* p) { return p ? ULCNET_HOP : -1; }
static Aec* mu_get_aec(const MonoUlcnet* p) { return p ? p->aec : NULL; }

static void mu_reset(MonoUlcnet* p) {
    if (!p) return;
    aec_reset(p->aec);
    ulcnet_nn_stage_reset(p->stage);
}

/* AEC, then the stage's begin with the lane's formed hop, the raw reference and the alignment
 * events (changed = CHANGED, solid = not UNLOCKED). */
static int mu_process_begin(MonoUlcnet* p, const float* mic, const float* ref) {
    AecResContext rctx;
    AecLinearContext lctx;
    UlcnetNnDelay delay;
    if (!p || !mic || !ref) return -1;
    aec_process_context(p->aec, mic, ref);
    aec_get_res_context(p->aec, &rctx);
    aec_get_linear_context(p->aec, &lctx);
    delay.changed = lctx.delay_state == AEC_LINEAR_DELAY_CHANGED;
    delay.solid = lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
    return ulcnet_nn_stage_begin_time(p->stage, rctx.formed_hop, ref, delay) == 0 ? 0 : -1;
}

static int mu_nn_begin(MonoUlcnet* p, UlcnetNnJob* job) {
    int r;
    if (!p || !job) return -1;
    r = ulcnet_nn_stage_nn_begin(p->stage, job);
    return r < 0 ? -1 : r;
}

static int mu_nn_end(MonoUlcnet* p, int model_rc) {
    return p && ulcnet_nn_stage_nn_end(p->stage, model_rc) == 0 ? 0 : -1;
}

static int mu_process_end(MonoUlcnet* p, float* out) {
    return p && out && ulcnet_nn_stage_end(p->stage, out) == 0 ? 0 : -1;
}

/* The whole hop with the installed model (identity when unset). */
static int mu_process(MonoUlcnet* p, const float* mic, const float* ref, float* out) {
    if (!p || !mic || !ref || !out) return -1;
    if (mu_process_begin(p, mic, ref) != 0) return -1;
    return ulcnet_nn_stage_process(p->stage, out) == 0 ? 0 : -1;
}


#define HOP            ULCNET_HOP     /* pinned by the compiled grid        */
#define ECHO_DELAY     2000           /* samples; > 832-tap filter reach    */
#define MAX_LOCK_HOPS  150            /* acquisition bound (measured: ~15)  */

static int g_failures = 0;
#define CHECK(cond, msg) do { \
        if (cond) { printf("PASS: %s\n", (msg)); } \
        else      { fprintf(stderr, "FAIL: %s\n", (msg)); g_failures++; } \
    } while (0)

static char g_msgbuf[256];
static const char* fmt_msg(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_msgbuf, sizeof(g_msgbuf), fmt, ap);
    va_end(ap);
    return g_msgbuf;
}

/* DelayAec searches a 16 kHz-equivalent sidechain and scales the selected
 * delay back to native samples. Test stimuli and bounds therefore scale by
 * exactly three on the 48 kHz product grid. */
#define DELAY_RATE_FACTOR (ULCNET_SR / 16000)

/* ---- LCG synthetic generator (as the other mono tests) ---- */
static uint32_t lcg_state;
static float lcg_sample(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

/* Deterministic echo simulator. delay==0: mic = 0.3*ref + 0.05*noise (the
 * zero-delay shape). delay>0: echo goes through a `delay`-sample line with
 * 0.02*noise near-end (the bulk-delay shape). Reseed via echo_sim_init. */
typedef struct {
    float dline[ECHO_DELAY];
    int   dpos;
    int   delay;
} EchoSim;

static void echo_sim_init(EchoSim* s, int delay, uint32_t seed) {
    memset(s, 0, sizeof(*s));
    s->delay = delay;
    lcg_state = seed;
}

static void echo_sim_hop(EchoSim* s, float* mic, float* ref) {
    for (int i = 0; i < HOP; i++) {
        float r = lcg_sample();
        ref[i] = r;
        if (s->delay > 0) {
            float delayed = s->dline[s->dpos];
            s->dline[s->dpos] = r;
            s->dpos = (s->dpos + 1) % s->delay;
            mic[i] = 0.3f * delayed + 0.02f * lcg_sample();
        } else {
            mic[i] = 0.3f * r + 0.05f * lcg_sample();
        }
    }
}

static AecLinearDelayState pipeline_delay_state(const MonoUlcnet* p) {
    AecLinearContext lctx;
    aec_get_linear_context(mu_get_aec(p), &lctx);
    return lctx.delay_state;
}

/* Frames emitted at hop index h: the rolling (center=False) analysis emits
 * exactly one frame per hop from hop #0 (kept as a function so the
 * schedule stays in one place). */
static int frames_at_hop(int h) { (void)h; return 1; }

/* Spend an armed identity-reprime budget over one hop's `frames` emitted
 * frames, and report how many of them the reprime covered. The budget is
 * armed at the TOP of the CHANGED hop, so that hop's own frames are already
 * spending it. */
static int reprime_take(int* armed, int frames) {
    int skipped = *armed < frames ? *armed : frames;
    *armed -= skipped;
    return skipped;
}

/* =========================================================================
 * 1. identity E2E: NULL model, zero-delay echo. Every hop, BEFORE the next
 *    process call, ctx.formed_hop is recorded; out[hop p] must match
 *    formed_history[p-1] within 2e-4 and hop 0 must be exactly zeros.
 * ========================================================================= */
static void test_identity_e2e(void) {
    enum { N = 100 };
    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    MonoUlcnet* p = mu_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for identity E2E test\n"); g_failures++; return; }

    CHECK(mu_hop_size(p) == HOP,
          fmt_msg("hop_size accessor reports the compiled ULCNet hop (%d)", HOP));

    float mic[HOP], ref[HOP];
    static float out_hist[N][HOP], formed_hist[N][HOP];
    EchoSim sim;
    echo_sim_init(&sim, 0, 0xC0FFEEu);

    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        mu_process(p, mic, ref, out_hist[h]);
        /* Read the tap AFTER process, BEFORE the next AEC call — the seam
         * pointers alias AEC-internal buffers valid until then. */
        AecResContext rctx;
        aec_get_res_context(mu_get_aec(p), &rctx);
        memcpy(formed_hist[h], rctx.formed_hop, sizeof(formed_hist[h]));
    }

    int hop0_zero = 1;
    for (int i = 0; i < HOP; i++) if (out_hist[0][i] != 0.0f) hop0_zero = 0;
    CHECK(hop0_zero, "hop #0 output is exactly all zeros (nothing emitted yet)");

    double max_err = 0.0, max_sig = 0.0;
    for (int h = 1; h < N; h++) {
        for (int i = 0; i < HOP; i++) {
            double d = fabs((double)out_hist[h][i] - (double)formed_hist[h - 1][i]);
            if (d > max_err) max_err = d;
            double s = fabs((double)out_hist[h][i]);
            if (s > max_sig) max_sig = s;
        }
    }
    CHECK(max_err <= 2e-4,
          fmt_msg("identity E2E: out[hop p] == formed_history[p-1] within 2e-4 "
                  "over %d hops (max err %.3e) — one-hop latency + tap choice", N, max_err));
    CHECK(max_sig > 1e-4,
          fmt_msg("identity E2E carries real signal (max |out| = %.3e), not silence", max_sig));

    mu_destroy(p);
}

/* =========================================================================
 * 2. counting model, bulk-delay echo (UNLOCKED -> CHANGED -> LOCKED). infer is stepped
 *    for every emitted frame except the ULCNET_NN_STAGE_REPRIME_FRAMES identity
 *    frames after each CHANGED hop; model->reset fires once per CHANGED hop plus once
 *    per pipeline reset. The expectation is built hop by hop from that policy, so a
 *    missing or an over-long reprime fails.
 * ========================================================================= */
typedef struct {
    int infer_calls;
    int reset_calls;
} CountingModelState;

static int counting_infer(void* user,
                          const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                          const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                          float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    CountingModelState* st = (CountingModelState*)user;
    (void)far_re; (void)far_im;
    st->infer_calls++;
    memcpy(out_re, err_re, ULCNET_BINS * sizeof(float));
    memcpy(out_im, err_im, ULCNET_BINS * sizeof(float));
    return 0;
}

static void counting_reset(void* user) {
    ((CountingModelState*)user)->reset_calls++;
}

static void test_counting_model(void) {
    enum { N = 200 };
    CountingModelState st = {0, 0};

    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    cfg.model.user  = &st;
    cfg.model.infer = counting_infer;
    cfg.model.io_descriptor = test_io_descriptor();
    cfg.model.reset = counting_reset;

    MonoUlcnet* p = mu_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for counting-model test\n"); g_failures++; return; }

    float mic[HOP], ref[HOP], out[HOP];
    EchoSim sim;
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);

    int calls_after_hop1 = -1;
    int pattern_ok = 1, first_bad_hop = -1;
    int n_changed = 0, n_locked = 0;
    int expected_calls = 0, armed = 0, skipped_total = 0;
    for (int h = 0; h < N; h++) {
        int frames, skipped_here;
        echo_sim_hop(&sim, mic, ref);
        mu_process(p, mic, ref, out);
        AecLinearDelayState ds = pipeline_delay_state(p);
        if (ds == AEC_LINEAR_DELAY_CHANGED)
            armed = ULCNET_NN_STAGE_REPRIME_FRAMES;
        frames = frames_at_hop(h);
        skipped_here = reprime_take(&armed, frames);
        skipped_total += skipped_here;
        expected_calls += frames - skipped_here;
        if (h == 1) calls_after_hop1 = st.infer_calls;
        if (st.infer_calls != expected_calls && pattern_ok) {
            pattern_ok = 0;
            first_bad_hop = h;
        }
        if (ds == AEC_LINEAR_DELAY_CHANGED) n_changed++;
        if (ds == AEC_LINEAR_DELAY_LOCKED)  n_locked++;
    }

    CHECK(calls_after_hop1 == 2,
          fmt_msg("counting model: 2 infer calls after hop #1 (one per hop from "
                  "hop #0; got %d)", calls_after_hop1));
    CHECK(pattern_ok && st.infer_calls == N - skipped_total,
          fmt_msg("counting model: one infer per emitted frame at every hop except the "
                  "%d reprime frames per alignment generation "
                  "(%d calls after %d hops, %d skipped, first bad hop %d [-1=none])",
                  (int)ULCNET_NN_STAGE_REPRIME_FRAMES,
                  st.infer_calls, N, skipped_total, first_bad_hop));
    CHECK(skipped_total == n_changed * ULCNET_NN_STAGE_REPRIME_FRAMES &&
          skipped_total > 0,
          fmt_msg("counting model: the reprime really ran (%d frames skipped over %d "
                  "generations) -- compute DROPS on those frames, never doubles",
                  skipped_total, n_changed));
    CHECK(n_changed >= 1 && n_locked >= 1,
          fmt_msg("delay actually acquired during the run (%d CHANGED, %d LOCKED hops)",
                  n_changed, n_locked));
    CHECK(st.reset_calls == n_changed,
          fmt_msg("model->reset fired exactly once per CHANGED hop (%d resets, %d CHANGED)",
                  st.reset_calls, n_changed));

    int resets_before = st.reset_calls;
    mu_reset(p);
    CHECK(st.reset_calls == resets_before + 1,
          "mu_reset invokes model->reset exactly once");

    int calls_before = st.infer_calls;
    echo_sim_hop(&sim, mic, ref);
    mu_process(p, mic, ref, out);
    int hop0_zero = 1;
    for (int i = 0; i < HOP; i++) if (out[i] != 0.0f) hop0_zero = 0;
    CHECK(st.infer_calls == calls_before + 1 && hop0_zero,
          "after pipeline reset the next hop is hop #0 again: one inference "
          "(like every hop), all-zero output (synthesis preamble)");

    mu_destroy(p);
}

/* FIXED has no estimator generation event: its seam is raw while the alignment ring
 * fills, then aligned. The composition must still reset the model at that boundary.
 * The far branch carries real noise, so "UNLOCKED means the RAW far" is asserted
 * byte-exactly against the caller's far history on every fill hop, and the shifted
 * content the same way from the first LOCKED hop on. */
static void test_fixed_first_alignment_resets_model(void) {
    enum { FIXED_DELAY = 2 * HOP, N = 8 };
    MonoUlcnetConfig cfg =
        mu_default_config(ULCNET_SR);
    CountingModelState st = {0, 0};
    MonoUlcnet* p;
    float mic[HOP] = {0};
    float ref[HOP];
    float out[HOP];
    static float ref_hist[N][HOP];
    int first_locked = -1;
    int changed = 0;
    int raw_hops = 0;
    int shifted_hops = 0;
    int raw_mismatch = -1;
    int shifted_mismatch = -1;

    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = FIXED_DELAY;
    cfg.model.user = &st;
    cfg.model.infer = counting_infer;
    cfg.model.io_descriptor = test_io_descriptor();
    cfg.model.reset = counting_reset;
    p = mu_create(&cfg);
    if (!p) {
        fprintf(stderr, "FAIL: setup fixed-delay transition test\n");
        g_failures++;
        return;
    }

    lcg_state = 0xA11FEEDu;
    for (int h = 0; h < N; ++h) {
        AecLinearDelayState state;
        AecLinearContext lctx;
        for (int i = 0; i < HOP; ++i) ref[i] = lcg_sample();
        memcpy(ref_hist[h], ref, sizeof(ref));
        CHECK(mu_process(p, mic, ref, out) == 0,
              "FIXED transition frame processes");
        state = pipeline_delay_state(p);
        /* The far tap aliases AEC-internal memory valid only until the next
         * aec call, so it is read here, right after process. */
        aec_get_linear_context(mu_get_aec(p), &lctx);
        if (state == AEC_LINEAR_DELAY_CHANGED) changed++;
        if (state == AEC_LINEAR_DELAY_LOCKED && first_locked < 0)
            first_locked = h;
        if (state == AEC_LINEAR_DELAY_LOCKED) {
            /* FIXED_DELAY is exactly 2 hops, so the shifted content is the
             * far hop from 2 hops back -- an independently held reference. */
            if (memcmp(lctx.aligned_far_hop, ref_hist[h - 2], sizeof(ref)) != 0
                && shifted_mismatch < 0)
                shifted_mismatch = h;
            shifted_hops++;
        } else {
            if (memcmp(lctx.aligned_far_hop, ref_hist[h], sizeof(ref)) != 0
                && raw_mismatch < 0)
                raw_mismatch = h;
            raw_hops++;
        }
    }

    CHECK(first_locked == 2,
          fmt_msg("FIXED aligned far becomes usable after exactly delay+hop "
                  "samples (first locked hop %d, expected 2)", first_locked));
    CHECK(changed == 0,
          "lib/aec FIXED exposes UNLOCKED->LOCKED without CHANGED");
    CHECK(raw_mismatch < 0 && raw_hops == 2,
          fmt_msg("every UNLOCKED fill hop serves the RAW far byte for byte "
                  "(%d fill hops, first mismatch %d [-1=none])",
                  raw_hops, raw_mismatch));
    CHECK(shifted_mismatch < 0 && shifted_hops == N - 2,
          fmt_msg("every LOCKED hop serves the far delayed by exactly "
                  "fixed_delay_samples (%d locked hops, first mismatch %d "
                  "[-1=none])", shifted_hops, shifted_mismatch));
    CHECK(st.reset_calls == 1,
          fmt_msg("the model is reset exactly once when FIXED switches from "
                  "raw to aligned far (got %d)", st.reset_calls));
    /* The transition suppresses inference for the identity reprime and for
     * nothing else: N hops emit N frames, of which exactly
     * ULCNET_NN_STAGE_REPRIME_FRAMES straddle the switch. */
    CHECK(st.infer_calls == N - ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("FIXED transition suppresses inference for exactly the %d "
                  "straddling frames and no others (%d calls over %d emitted "
                  "frames)", (int)ULCNET_NN_STAGE_REPRIME_FRAMES,
                  st.infer_calls, N));
    mu_destroy(p);
}

/* =========================================================================
 * 2b. A relock on the SAME delay still resets the model (ALIGNED mode keeps stepping
 *     infer() while unlocked, so its recurrent state spans an unvouched reference):
 *     LOCKED(V) -> forced UNLOCKED for several hops -> LOCKED(V) issues exactly one
 *     model->reset, on the relock hop. Same contract as the 4ch twin test.
 *     MUTATION: removing the `delay_state == AEC_LINEAR_DELAY_CHANGED` model reset in
 *     ulcnet/ulcnet_nn_stage.c turns this test red.
 * ========================================================================= */
static int pipeline_delay_samples(const MonoUlcnet* p) {
    AecLinearContext lctx;
    aec_get_linear_context(mu_get_aec(p), &lctx);
    return lctx.delay_samples;
}

static void test_relock_same_delay_resets_model(void) {
    enum { WARM = 90, RELOCK = 90 };
    const int bulk_delay = 64 * DELAY_RATE_FACTOR;
    CountingModelState st = {0, 0};

    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    cfg.model.user  = &st;
    cfg.model.infer = counting_infer;
    cfg.model.io_descriptor = test_io_descriptor();
    cfg.model.reset = counting_reset;

    MonoUlcnet* p = mu_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for mono relock test\n"); g_failures++; return; }

    float mic[HOP], ref[HOP], out[HOP];
    EchoSim sim;
    int acquire_hop = -1, relock_hop = -1;
    int acquire_delay = -1, relock_delay = -2;
    int resets_at_acquire = -1, resets_at_relock = -1;
    int resets_before_reset = -1, resets_at_pipeline_reset = -1;
    int unlocked_hops_after_reset = 0, resets_during_unlock = -1;
    int applied_after_relock = 0;

    echo_sim_init(&sim, bulk_delay, 0xC0FFEEu);
    for (int h = 0; h < WARM + RELOCK; h++) {
        if (h == WARM) {
            /* Forced unlock: the pipeline reset runs aec_reset(), which is
             * lib/aec's only in-processing path back to current_delay == -1
             * (the estimator's REFINED-confidence latch never drops on its
             * own). The stimulus continues unchanged, so the AEC re-acquires
             * the very same bulk delay. */
            resets_before_reset = st.reset_calls;
            mu_reset(p);
            resets_at_pipeline_reset = st.reset_calls;
            echo_sim_init(&sim, bulk_delay, 0xC0FFEEu);
        }
        echo_sim_hop(&sim, mic, ref);
        mu_process(p, mic, ref, out);

        AecLinearDelayState ds = pipeline_delay_state(p);
        int locked = (ds == AEC_LINEAR_DELAY_LOCKED ||
                      ds == AEC_LINEAR_DELAY_CHANGED);
        if (h < WARM) {
            if (locked && acquire_hop < 0) {
                acquire_hop = h;
                acquire_delay = pipeline_delay_samples(p);
                resets_at_acquire = st.reset_calls;
            }
        } else if (!locked) {
            unlocked_hops_after_reset++;
            resets_during_unlock = st.reset_calls - resets_at_pipeline_reset;
        } else if (relock_hop < 0) {
            relock_hop = h;
            relock_delay = pipeline_delay_samples(p);
            resets_at_relock = st.reset_calls;
        } else {
            applied_after_relock++;
        }
    }

    CHECK(acquire_hop >= 0 && relock_hop >= 0,
          fmt_msg("mono relock: both acquisitions happened (acquire hop %d, relock hop %d)",
                  acquire_hop, relock_hop));
    CHECK(relock_delay == acquire_delay,
          fmt_msg("mono relock: re-acquired the SAME applied delay (%d then %d samples)",
                  acquire_delay, relock_delay));
    CHECK(resets_at_acquire == 1,
          fmt_msg("mono relock: first acquisition fired model->reset exactly once (%d)",
                  resets_at_acquire));
    CHECK(resets_at_pipeline_reset == resets_before_reset + 1,
          "mono relock: the pipeline reset itself fires model->reset once");
    CHECK(unlocked_hops_after_reset >= 2,
          fmt_msg("mono relock: the forced unlock lasted several hops (%d, not a vacuous pass)",
                  unlocked_hops_after_reset));
    CHECK(resets_during_unlock == 0,
          fmt_msg("mono relock: no model->reset while merely unlocked (%d)",
                  resets_during_unlock));
    CHECK(resets_at_relock == resets_at_pipeline_reset + 1,
          fmt_msg("mono relock: the same-delay relock fires model->reset exactly once, "
                  "on the relock hop itself (%d after %d)",
                  resets_at_relock, resets_at_pipeline_reset));
    CHECK(st.reset_calls == 3,
          fmt_msg("mono relock: 3 model resets total -- acquisition, pipeline reset, "
                  "same-delay relock (got %d)", st.reset_calls));
    CHECK(applied_after_relock >= 10,
          fmt_msg("mono relock: the stream goes on running locked after the relock (%d hops)",
                  applied_after_relock));

    mu_destroy(p);
}

/* =========================================================================
 * 3. fail-open model callback. Model A halves the spectrum on success (rc=0) and
 *    doubles it on scheduled failing hops (rc!=0); pipeline B has a NULL model.
 *    Output hop p mixes the frames pushed at hops p-1 and p (50% WOLA overlap), so
 *    A == B bitwise exactly when both contributing frames took the identity path.
 * ========================================================================= */
typedef struct {
    int fail_now;   /* set by the test before each process call */
} FailModelState;

static int failing_infer(void* user,
                         const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                         const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                         float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    FailModelState* st = (FailModelState*)user;
    (void)far_re; (void)far_im;
    float scale = st->fail_now ? 2.0f : 0.5f;
    for (int k = 0; k < ULCNET_BINS; k++) {
        out_re[k] = scale * err_re[k];
        out_im[k] = scale * err_im[k];
    }
    /* On failure the model has ALREADY written garbage (2x) into out_re/
     * out_im — the pipeline must discard it, not "fail because the buffer
     * was left untouched". */
    return st->fail_now ? 1 : 0;
}

static void test_fail_open_and_delay_gating(void) {
    enum { N = 200, FAIL_LEN = 8, FAIL_GAP = 10 };
    FailModelState st = {0};

    MonoUlcnetConfig cfg_a = mu_default_config(ULCNET_SR);
    cfg_a.model.user  = &st;
    cfg_a.model.infer = failing_infer;
    cfg_a.model.io_descriptor = test_io_descriptor();
    MonoUlcnetConfig cfg_b = mu_default_config(ULCNET_SR); /* NULL model */

    MonoUlcnet* pa = mu_create(&cfg_a);
    MonoUlcnet* pb = mu_create(&cfg_b);
    if (!pa || !pb) {
        fprintf(stderr, "FAIL: setup (create) for fail-open test\n");
        g_failures++;
        if (pa) mu_destroy(pa);
        if (pb) mu_destroy(pb);
        return;
    }

    float mic[HOP], ref[HOP], out_a[HOP], out_b[HOP];
    EchoSim sim;
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);

    static int ident[N];       /* frame(s) pushed at hop h took identity path */
    int lock_hop = -1;         /* first non-UNLOCKED hop */
    int fail_start = -1;       /* scheduled after lock is observed */
    int equal_ok = 1, differ_ok = 1, states_ok = 1;
    int first_bad_equal = -1, first_bad_differ = -1;
    int n_equal_hops = 0, n_differ_hops = 0, n_failing = 0, n_unlocked = 0;
    int reprime_left = 0, n_reprime_hops = 0;

    for (int h = 0; h < N; h++) {
        st.fail_now = (fail_start >= 0 && h >= fail_start && h < fail_start + FAIL_LEN);
        if (st.fail_now) n_failing++;

        echo_sim_hop(&sim, mic, ref);
        mu_process(pa, mic, ref, out_a);
        mu_process(pb, mic, ref, out_b);

        AecLinearDelayState ds_a = pipeline_delay_state(pa);
        AecLinearDelayState ds_b = pipeline_delay_state(pb);
        if (ds_a != ds_b) states_ok = 0;   /* model must not feed back into the AEC */
        if (ds_a == AEC_LINEAR_DELAY_UNLOCKED) n_unlocked++;
        if (lock_hop < 0 && ds_a != AEC_LINEAR_DELAY_UNLOCKED) {
            lock_hop = h;
            fail_start = h + FAIL_GAP;   /* schedule the failure window post-lock */
        }
        /* The identity reprime after an alignment generation puts its own
         * frames on the identity path too. Fold it into the same
         * bookkeeping so the 50%-overlap rule below stays exact. The
         * acquisition always lands well inside the steady 1-frame-per-hop
         * region (lock_hop >= 2, asserted below), so one reprime frame is
         * consumed per hop here. */
        if (ds_a == AEC_LINEAR_DELAY_CHANGED)
            reprime_left = ULCNET_NN_STAGE_REPRIME_FRAMES;
        {
            int reprimed = reprime_take(&reprime_left, 1);
            ident[h] = st.fail_now || reprimed > 0;
            n_reprime_hops += reprimed;
        }

        /* equal_expected: hop 0 both zero; hop p>=1 mixes the frames pushed
         * at hops p-1 and p (one frame per hop from hop #0). */
        int equal_expected;
        if (h == 0)      equal_expected = 1;
        else             equal_expected = ident[h - 1] && ident[h];

        int bitwise_equal = memcmp(out_a, out_b, sizeof(out_a)) == 0;
        if (equal_expected) {
            n_equal_hops++;
            if (!bitwise_equal && equal_ok) { equal_ok = 0; first_bad_equal = h; }
        } else {
            n_differ_hops++;
            if (bitwise_equal && differ_ok) { differ_ok = 0; first_bad_differ = h; }
        }
    }

    CHECK(lock_hop >= 2 && lock_hop <= MAX_LOCK_HOPS,
          fmt_msg("delay acquired at hop %d (2..%d) so both gated and applied phases ran "
                  "and the reprime lands in the steady 1-frame-per-hop region",
                  lock_hop, MAX_LOCK_HOPS));
    CHECK(n_unlocked >= 2 && n_failing == FAIL_LEN &&
          n_reprime_hops == ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("coverage: %d UNLOCKED hops, %d scheduled infer failures, %d reprime "
                  "identity frames", n_unlocked, n_failing, n_reprime_hops));
    CHECK(states_ok, "model presence does not perturb the AEC (identical delay states A vs B)");
    CHECK(equal_ok,
          fmt_msg("fail-open: output is BIT-IDENTICAL to the NULL-model pipeline on all %d "
                  "identity-path hops (discarded rc!=0 output; "
                  "first bad hop %d [-1=none])", n_equal_hops, first_bad_equal));
    CHECK(differ_ok,
          fmt_msg("applied path: output DIFFERS from the NULL-model pipeline on all %d hops "
                  "with a successful model frame (first bad hop %d [-1=none])",
                  n_differ_hops, first_bad_differ));

    mu_destroy(pa);
    mu_destroy(pb);
}

/* =========================================================================
 * 10. far-input contract gate: a model that publishes a model-I/O
 * descriptor may only be wired to a stage whose far branch matches the
 * checkpoint the descriptor describes. Both directions are checked, on
 * every construction path of the composition.
 * ========================================================================= */
static int gate_infer_identity(void* user,
                               const float er[ULCNET_BINS], const float ei[ULCNET_BINS],
                               const float fr[ULCNET_BINS], const float fi[ULCNET_BINS],
                               float or_[ULCNET_BINS], float oi[ULCNET_BINS]) {
    (void)user; (void)fr; (void)fi;
    memcpy(or_, er, sizeof(float) * ULCNET_BINS);
    memcpy(oi, ei, sizeof(float) * ULCNET_BINS);
    return 0;
}

static void test_raw_descriptor_gate(void) {
    UlcnetModelIoDescriptor raw_desc, aligned_desc;
    MonoUlcnetMemReq req;
    void* pool = NULL;
    MonoUlcnet* p;

    if (ulcnet_model_io_descriptor_default(8, &raw_desc) != 0 ||
        ulcnet_model_io_descriptor_default(8, &aligned_desc) != 0) {
        fprintf(stderr, "FAIL: setup (descriptor_default) for far-mode gate test\n");
        g_failures++;
        return;
    }
    aligned_desc.far_input_mode = ULCNET_FAR_ALIGNED;
    CHECK(raw_desc.far_input_mode == ULCNET_FAR_RAW,
          "descriptor_default publishes the fixed raw-far contract");

    /* An undescribed model remains valid for the identity/test boundary. */
    MonoUlcnetConfig undescribed = mu_default_config(ULCNET_SR);
    undescribed.model.user = NULL;
    undescribed.model.infer = gate_infer_identity;
    undescribed.model.reset = NULL;
    undescribed.model.io_descriptor = NULL;
    /* An INFERRING model must publish a descriptor: its delay depth and
     * history shapes are what the host-side rings are carved from, and a
     * disagreement with the graph is undetectable downstream. An identity
     * model -- no infer callback -- has no shapes to agree about. */
    p = mu_create(&undescribed);
    CHECK(p == NULL, "an INFERRING model with io_descriptor == NULL is refused");
    mu_destroy(p);
    {
        MonoUlcnetConfig identity_only = undescribed;
        identity_only.model.infer = NULL;
        p = mu_create(&identity_only);
        CHECK(p != NULL, "an identity model (no infer) may still leave the descriptor NULL");
        mu_destroy(p);
    }
    undescribed.model.io_descriptor = test_io_descriptor();

    MonoUlcnetConfig raw_model = undescribed;
    raw_model.model.io_descriptor = &raw_desc;
    p = mu_create(&raw_model);
    CHECK(p != NULL, "raw descriptor is accepted");
    mu_destroy(p);

    MonoUlcnetConfig aligned_model = undescribed;
    aligned_model.model.io_descriptor = &aligned_desc;
    p = mu_create(&aligned_model);
    CHECK(p == NULL, "aligned-far descriptor is rejected");
    mu_destroy(p);

    /* The one-block construction shares the gate, so the mismatch is fail-fast on
     * both construction paths. */
    if (mu_get_mem_requirements(&raw_model, &req) != 0 ||
        posix_memalign(&pool, 16, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for far-mode gate test\n");
        g_failures++;
        return;
    }
    CHECK(mu_init(pool, (size_t)req.bytes, &aligned_model) == NULL,
          "mu_init rejects aligned descriptor too");

    p = mu_init(pool, (size_t)req.bytes, &raw_model);
    CHECK(p != NULL, "raw descriptor initializes in the same pool");
    mu_destroy(p);
    free(pool);
}

/* =========================================================================
 * The grid and the pool plan of the composition: the lane refuses what is not the compiled grid, the delay request reaches the
 * AEC pool size, and the stage and AEC blocks are refused when misaligned or short.
 * ========================================================================= */
static void test_grid_and_pool_plan(void) {
    MonoUlcnetMemReq req;
    MonoUlcnetConfig good = mu_default_config(ULCNET_SR);
    static const int bad_rates[] = {8000, 44100, ULCNET_OTHER_SR};

    CHECK(good.sample_rate == ULCNET_SR && good.fft_size == ULCNET_N_FFT,
          fmt_msg("default config is the compiled %d Hz / frame-FFT %d / hop %d grid",
                  ULCNET_SR, ULCNET_N_FFT, ULCNET_HOP));
    CHECK(mu_get_mem_requirements(NULL, &req) == -1 && mu_get_mem_requirements(&good, NULL) == -1,
          "the plan rejects NULL arguments");
    for (int i = 0; i < 3; i++) {
        MonoUlcnetConfig c = mu_default_config(bad_rates[i]);
        CHECK(mu_get_mem_requirements(&c, &req) == -1,
              fmt_msg("the lane rejects sample_rate=%d (compiled ULCNet rate differs)", bad_rates[i]));
    }
    {
        MonoUlcnetConfig bad_fft = mu_default_config(ULCNET_SR);
        bad_fft.fft_size = ULCNET_OTHER_N_FFT;   /* legal elsewhere -- NOT on this grid */
        CHECK(mu_get_mem_requirements(&bad_fft, &req) == -1,
              fmt_msg("the lane rejects fft_size=%d (compiled ULCNet grid is %d)",
                      ULCNET_OTHER_N_FFT, ULCNET_N_FFT));
    }
    {
        MonoUlcnetConfig bad_preset = mu_default_config(ULCNET_SR);
        bad_preset.aec_preset = (AecPreset)99;
        CHECK(mu_get_mem_requirements(&bad_preset, &req) == -1,
              "the lane rejects an out-of-enum aec_preset");
    }
    {
        MonoUlcnetConfig matched2 = mu_default_config(ULCNET_SR);
        MonoUlcnetConfig external = matched2;
        MonoUlcnetMemReq matched_req, external_req;
        matched2.delay_num_filters = 2;
        external.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        external.delay_num_filters = 5;
        CHECK(mu_get_mem_requirements(&matched2, &matched_req) == 0 &&
              mu_get_mem_requirements(&external, &external_req) == 0 &&
              matched_req.bytes > external_req.bytes,
              "the AEC delay configuration reaches the pool size");
    }
    {
        MonoUlcnetMemReq req0, req_explicit;
        MonoUlcnetConfig c0 = mu_default_config(ULCNET_SR);
        c0.fft_size = 0;  /* 0 resolves to the compiled grid */
        CHECK(mu_get_mem_requirements(&c0, &req0) == 0 &&
              mu_get_mem_requirements(&good, &req_explicit) == 0 &&
              req0.bytes == req_explicit.bytes && req0.bytes > 0,
              fmt_msg("fft_size=0 resolves to the same pool as the explicit %d", ULCNET_N_FFT));
    }

    /* Pool plan: a misaligned or short block is refused, the exact block accepted, and the
     * block stays usable after the refusals. */
    {
        void* pool = NULL;
        MonoUlcnet* p;
        CHECK(mu_get_mem_requirements(&good, &req) == 0 &&
              posix_memalign(&pool, 16, (size_t)req.bytes + 16) == 0 && pool,
              "pool plan: the block allocates");
        if (pool) {
            memset(pool, 0xA5, (size_t)req.bytes + 16);
            CHECK(mu_init((uint8_t*)pool + 1, (size_t)req.bytes, &good) == NULL,
                  "init rejects a misaligned (base+1) pool");
            CHECK(mu_init(pool, (size_t)req.bytes - 1, &good) == NULL,
                  "init rejects an undersized (bytes-1) pool");
            p = mu_init(pool, (size_t)req.bytes, &good);
            CHECK(p != NULL, "init accepts a correctly aligned/sized pool");
            mu_destroy(p);
            free(pool);
        }
    }
}

/* =========================================================================
 * 4e. create-vs-init byte parity on a 0xA5-poisoned pool (NULL model;
 *     bulk-delay echo so the parity run crosses the CHANGED/LOCKED
 *     transition too). Direct proof of the explicit-zeroing claim.
 * ========================================================================= */
static void test_create_vs_init_parity(void) {
    enum { N = 300 };
    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    MonoUlcnetMemReq req;
    if (mu_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup for create-vs-init parity test\n");
        g_failures++;
        return;
    }

    MonoUlcnet* p_heap = mu_create(&cfg);
    void* pool = NULL;
    if (!p_heap || posix_memalign(&pool, 16, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: setup alloc for create-vs-init parity test\n");
        g_failures++;
        if (p_heap) mu_destroy(p_heap);
        return;
    }
    memset(pool, 0xA5, (size_t)req.bytes);   /* dirty pool: init must not rely on zeros */
    MonoUlcnet* p_pool = mu_init(pool, (size_t)req.bytes, &cfg);
    if (!p_pool) {
        fprintf(stderr, "FAIL: init on a poisoned (0xA5) pool\n");
        g_failures++;
        mu_destroy(p_heap);
        free(pool);
        return;
    }

    float mic[HOP], ref[HOP];
    static float out_heap[N][HOP], out_pool[N][HOP];
    EchoSim sim;

    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        mu_process(p_heap, mic, ref, out_heap[h]);
    }
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        mu_process(p_pool, mic, ref, out_pool[h]);
    }

    CHECK(memcmp(out_heap, out_pool, sizeof(out_heap)) == 0,
          fmt_msg("create (heap) == init (0xA5-poisoned pool), %d hops, byte-for-byte", N));

    int finite = 1;
    for (int h = 0; h < N && finite; h++)
        for (int i = 0; i < HOP; i++)
            if (out_heap[h][i] != out_heap[h][i]) { finite = 0; break; }
    CHECK(finite, fmt_msg("%d-hop synthetic run produces no NaN in the output", N));

    mu_destroy(p_heap);
    mu_destroy(p_pool);
    free(pool);
}

/* =========================================================================
 * 5. NULL model == identity model, bit-identical (bulk-delay echo: covers
 *    UNLOCKED, CHANGED and LOCKED phases — the identity model's applied
 *    output must be byte-equal to the bypass path in every one of them).
 * ========================================================================= */
static void test_null_model_equals_identity_model(void) {
    enum { N = 200 };
    CountingModelState st = {0, 0};

    MonoUlcnetConfig cfg_null = mu_default_config(ULCNET_SR);
    MonoUlcnetConfig cfg_id   = mu_default_config(ULCNET_SR);
    cfg_id.model.user  = &st;
    cfg_id.model.infer = counting_infer;   /* copies err -> out, returns 0 */
    cfg_id.model.io_descriptor = test_io_descriptor();
    cfg_id.model.reset = counting_reset;

    MonoUlcnet* pn = mu_create(&cfg_null);
    MonoUlcnet* pi = mu_create(&cfg_id);
    if (!pn || !pi) {
        fprintf(stderr, "FAIL: setup (create) for NULL-vs-identity test\n");
        g_failures++;
        if (pn) mu_destroy(pn);
        if (pi) mu_destroy(pi);
        return;
    }

    float mic[HOP], ref[HOP], out_n[HOP], out_i[HOP];
    EchoSim sim;
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);

    int equal = 1, first_bad = -1, saw_locked = 0, n_changed = 0;
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        mu_process(pn, mic, ref, out_n);
        mu_process(pi, mic, ref, out_i);
        if (memcmp(out_n, out_i, sizeof(out_n)) != 0 && equal) { equal = 0; first_bad = h; }
        if (pipeline_delay_state(pi) != AEC_LINEAR_DELAY_UNLOCKED) saw_locked = 1;
        if (pipeline_delay_state(pi) == AEC_LINEAR_DELAY_CHANGED) n_changed++;
    }

    CHECK(saw_locked, "NULL-vs-identity run reached the locked (model-applied) phase");
    CHECK(equal,
          fmt_msg("NULL model output == identity (err->out copy) model output, "
                  "bit-identical over %d hops (first mismatch %d [-1=none])", N, first_bad));
    /* Stepped on every emitted frame except the reprime frames -- which are
     * identity frames too, so the bit-identity above is unaffected by them. */
    CHECK(st.infer_calls ==
          N - n_changed * ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("identity model was stepped on every frame except the %d reprime "
                  "frames of its %d alignment generations (%d calls)",
                  (int)ULCNET_NN_STAGE_REPRIME_FRAMES, n_changed,
                  st.infer_calls));

    mu_destroy(pn);
    mu_destroy(pi);
}

/* =========================================================================
 * 6. far-timestamp after fixed-delay acquisition. A far-passthrough model, silence on
 *    the mic and one unit impulse in far at sample T: the mono far tap is same-hop
 *    with the error tap and the chain lags the input by one hop, so the impulse must
 *    land at exactly T + HOP. Feeding the AEC's aligned far would move it by
 *    FIXED_DELAY.
 * ========================================================================= */
static int passthrough_far_infer(void* user,
                                 const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                                 const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                                 float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    (void)user; (void)err_re; (void)err_im;
    memcpy(out_re, far_re, ULCNET_BINS * sizeof(float));
    memcpy(out_im, far_im, ULCNET_BINS * sizeof(float));
    return 0;
}

static void test_raw_far_timestamp_after_fixed_delay_lock(void) {
    enum { N = 40, IMP_HOP = 8, IMP_OFF = 37, FIXED_DELAY = 2 * HOP };
    const int imp_index = IMP_HOP * HOP + IMP_OFF;
    const int expect_index = imp_index + HOP;   /* same-hop far tap + 1-hop chain */

    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = FIXED_DELAY;
    cfg.model.infer = passthrough_far_infer;
    cfg.model.io_descriptor = test_io_descriptor();
    MonoUlcnet* p = mu_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for far-timestamp test\n"); g_failures++; return; }

    float mic[HOP], ref[HOP];
    static float out_hist[N][HOP];
    for (int h = 0; h < N; h++) {
        memset(mic, 0, sizeof(mic));
        memset(ref, 0, sizeof(ref));
        if (h == IMP_HOP) ref[IMP_OFF] = 1.0f;
        mu_process(p, mic, ref, out_hist[h]);
    }

    int   found_index = -1;
    float peak = 0.0f;
    for (int h = 0; h < N; h++) {
        for (int i = 0; i < HOP; i++) {
            float a = fabsf(out_hist[h][i]);
            if (a > peak) { peak = a; found_index = h * HOP + i; }
        }
    }

    CHECK(found_index == expect_index,
          fmt_msg("raw-far timestamp after fixed-delay lock: impulse at far[%d] lands at out[%d] "
                  "(expected %d = impulse + 1 hop; offset %+d samples)",
                  imp_index, found_index, expect_index, found_index - expect_index));
    CHECK(peak > 0.9f,
          fmt_msg("far impulse reconstructed at ~unit amplitude (peak %.4f)", peak));

    mu_destroy(p);
}

/* =========================================================================
 * 7. Production does not gate on the delay lock: the 0.5x model's output is
 *    APPLIED from the first emitted frame (hop #1), while the delay is
 *    still UNLOCKED -- the model pipeline must differ from the
 *    NULL-model pipeline on every hop from #1 on (the bulk-delay near-end
 *    noise makes the error nonzero from the start).
 * ========================================================================= */
static int halving_infer(void* user,
                         const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                         const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                         float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    (void)user; (void)far_re; (void)far_im;
    for (int k = 0; k < ULCNET_BINS; k++) {
        out_re[k] = 0.5f * err_re[k];
        out_im[k] = 0.5f * err_im[k];
    }
    return 0;
}

static void test_model_applies_unlocked(void) {
    enum { N = 30 };
    MonoUlcnetConfig cfg_raw = mu_default_config(ULCNET_SR);
    cfg_raw.model.infer = halving_infer;
    cfg_raw.model.io_descriptor = test_io_descriptor();
    MonoUlcnetConfig cfg_null = mu_default_config(ULCNET_SR);

    MonoUlcnet* pr = mu_create(&cfg_raw);
    MonoUlcnet* pn = mu_create(&cfg_null);
    if (!pr || !pn) {
        fprintf(stderr, "FAIL: setup (create) for unlocked-model test\n");
        g_failures++;
        if (pr) mu_destroy(pr);
        if (pn) mu_destroy(pn);
        return;
    }

    float mic[HOP], ref[HOP], out_r[HOP], out_n[HOP];
    EchoSim sim;
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);

    int applied_from_hop1 = 1, first_equal_hop = -1, all_unlocked_seen = 1;
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        mu_process(pr, mic, ref, out_r);
        mu_process(pn, mic, ref, out_n);
        if (h == 0) {
            int zero = 1;
            for (int i = 0; i < HOP; i++) if (out_r[i] != 0.0f) zero = 0;
            CHECK(zero, "hop #0 still emits zeros (timing preamble unchanged)");
        } else if (memcmp(out_r, out_n, sizeof(out_r)) == 0 && applied_from_hop1) {
            applied_from_hop1 = 0;
            first_equal_hop = h;
        }
        /* N is small enough that the ECHO_DELAY acquisition (~hop 15 with
         * this seed) may occur late in the run; the claim only needs some
         * genuinely-UNLOCKED applied hops, which hop 1 always is. */
        if (h <= 2 && pipeline_delay_state(pr) != AEC_LINEAR_DELAY_UNLOCKED)
            all_unlocked_seen = 0;
    }
    CHECK(all_unlocked_seen, "delay really is UNLOCKED on the early applied hops");
    CHECK(applied_from_hop1,
          fmt_msg("production applies the model from the FIRST emitted frame with no "
                  "delay lock (output != NULL-model on every hop >= 1; first equal "
                  "hop %d [-1=none])", first_equal_hop));

    mu_destroy(pr);
    mu_destroy(pn);
}

/* =========================================================================
 * 8. NaN guard. On scheduled hops the model returns rc==0 but poisons one bin with
 *    NaN and another with +Inf; otherwise it halves the spectrum. As in test 3,
 *    output hop p is bit-identical to a NULL-model pipeline exactly when both
 *    contributing frames were poisoned and discarded, and differs once a clean frame
 *    contributes (so recovery is proven). No NaN may reach the output.
 * ========================================================================= */
typedef struct {
    int poison_now;   /* set by the test before each process call */
} NanModelState;

static int nan_infer(void* user,
                     const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                     const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                     float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    NanModelState* st = (NanModelState*)user;
    (void)far_re; (void)far_im;
    for (int k = 0; k < ULCNET_BINS; k++) {
        out_re[k] = 0.5f * err_re[k];
        out_im[k] = 0.5f * err_im[k];
    }
    if (st->poison_now) {
        /* rc stays 0: only the NaN/Inf guard can catch this frame. Poison
         * mid-array values so a partial scan would miss them. */
        out_re[100] = nanf("");
        out_im[200] = INFINITY;
    }
    return 0;
}

static void test_nan_guard(void) {
    enum { N = 60 };
    NanModelState st = {0};

    MonoUlcnetConfig cfg_a = mu_default_config(ULCNET_SR);
    cfg_a.model.user  = &st;
    cfg_a.model.infer = nan_infer;
    cfg_a.model.io_descriptor = test_io_descriptor();
    MonoUlcnetConfig cfg_b = mu_default_config(ULCNET_SR);

    MonoUlcnet* pa = mu_create(&cfg_a);
    MonoUlcnet* pb = mu_create(&cfg_b);
    if (!pa || !pb) {
        fprintf(stderr, "FAIL: setup (create) for NaN-guard test\n");
        g_failures++;
        if (pa) mu_destroy(pa);
        if (pb) mu_destroy(pb);
        return;
    }

    float mic[HOP], ref[HOP], out_a[HOP], out_b[HOP];
    EchoSim sim;
    echo_sim_init(&sim, 0, 0xC0FFEEu);   /* zero-delay: error nonzero from hop 0 */

    static int poisoned[N];
    int equal_ok = 1, differ_ok = 1, no_nan = 1;
    int first_bad_equal = -1, first_bad_differ = -1;
    int n_equal = 0, n_differ = 0, n_changed = 0;

    for (int h = 0; h < N; h++) {
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        st.poison_now = (h >= 20 && h < 28) || (h >= 40 && h < 48);
        poisoned[h] = st.poison_now;

        echo_sim_hop(&sim, mic, ref);
        mu_process(pa, mic, ref, out_a);
        mu_process(pb, mic, ref, out_b);
        if (pipeline_delay_state(pa) == AEC_LINEAR_DELAY_CHANGED) n_changed++;

        for (int i = 0; i < HOP; i++)
            if (!isfinite(out_a[i])) no_nan = 0;

        int equal_expected;
        if (h == 0)      equal_expected = 1;               /* both all-zero */
        else             equal_expected = poisoned[h - 1] && poisoned[h];

        int bitwise_equal = memcmp(out_a, out_b, sizeof(out_a)) == 0;
        if (equal_expected) {
            n_equal++;
            if (!bitwise_equal && equal_ok) { equal_ok = 0; first_bad_equal = h; }
        } else if (h >= 1) {
            n_differ++;
            if (bitwise_equal && differ_ok) { differ_ok = 0; first_bad_differ = h; }
        }
    }

    CHECK(no_nan, "NaN guard: no NaN/Inf ever reaches the pipeline output");
    /* The zero-delay shape never acquires, so the poison schedule is the ONLY
     * source of identity frames here -- proven, not assumed, because a
     * reprime frame would silently join the identity set and blunt the
     * "differs" half of the mixing rule below. */
    CHECK(n_changed == 0,
          fmt_msg("NaN-guard run crosses no alignment generation, so no identity "
                  "reprime frame joins the schedule (%d CHANGED hops)", n_changed));
    CHECK(n_equal >= 10 && n_differ >= 10,
          fmt_msg("coverage: %d NaN-identity hops and %d applied hops compared", n_equal, n_differ));
    CHECK(equal_ok,
          fmt_msg("rc==0 NaN frames are discarded: output BIT-IDENTICAL to the "
                  "NULL-model pipeline on all %d fully-poisoned hops (first bad "
                  "hop %d [-1=none])", n_equal, first_bad_equal));
    CHECK(differ_ok,
          fmt_msg("clean frames recover: output DIFFERS from the NULL-model "
                  "pipeline on all %d hops with a clean applied frame (first bad "
                  "hop %d [-1=none])", n_differ, first_bad_differ));

    mu_destroy(pa);
    mu_destroy(pb);
}

/* =========================================================================
 * 9. full-write contract. On scheduled hops the model writes only the first 100 bins
 *    but returns rc==0. The pipeline pre-fills its staging with NaN before every
 *    infer call, so the unwritten bins stay non-finite and the frame takes the
 *    identity path (same mixing rule as test 8); the next fully-written frame is
 *    applied again. MUTATION: removing the pre-fill leaves stale finite values in
 *    the unwritten bins and this test goes red.
 * ========================================================================= */
typedef struct {
    int partial_now;   /* set by the test before each process call */
} PartialModelState;

static int partial_write_infer(void* user,
                               const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                               const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                               float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    PartialModelState* st = (PartialModelState*)user;
    (void)far_re; (void)far_im;
    int n = st->partial_now ? 100 : ULCNET_BINS;
    for (int k = 0; k < n; k++) {
        out_re[k] = 0.5f * err_re[k];
        out_im[k] = 0.5f * err_im[k];
    }
    /* rc stays 0 even for the partial write: only the pipeline's NaN
     * pre-fill + finite guard can catch the 157 unwritten bins. */
    return 0;
}

static void test_partial_write_guard(void) {
    enum { N = 60 };
    PartialModelState st = {0};

    MonoUlcnetConfig cfg_a = mu_default_config(ULCNET_SR);
    cfg_a.model.user  = &st;
    cfg_a.model.infer = partial_write_infer;
    cfg_a.model.io_descriptor = test_io_descriptor();
    MonoUlcnetConfig cfg_b = mu_default_config(ULCNET_SR);

    MonoUlcnet* pa = mu_create(&cfg_a);
    MonoUlcnet* pb = mu_create(&cfg_b);
    if (!pa || !pb) {
        fprintf(stderr, "FAIL: setup (create) for partial-write test\n");
        g_failures++;
        if (pa) mu_destroy(pa);
        if (pb) mu_destroy(pb);
        return;
    }

    float mic[HOP], ref[HOP], out_a[HOP], out_b[HOP];
    EchoSim sim;
    echo_sim_init(&sim, 0, 0xC0FFEEu);   /* zero-delay: error nonzero from hop 0 */

    static int partial[N];
    int equal_ok = 1, differ_ok = 1, no_nan = 1;
    int first_bad_equal = -1, first_bad_differ = -1;
    int n_equal = 0, n_differ = 0, n_changed = 0;

    for (int h = 0; h < N; h++) {
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        st.partial_now = (h >= 20 && h < 28) || (h >= 40 && h < 48);
        partial[h] = st.partial_now;

        echo_sim_hop(&sim, mic, ref);
        mu_process(pa, mic, ref, out_a);
        mu_process(pb, mic, ref, out_b);
        if (pipeline_delay_state(pa) == AEC_LINEAR_DELAY_CHANGED) n_changed++;

        for (int i = 0; i < HOP; i++)
            if (!isfinite(out_a[i])) no_nan = 0;

        int equal_expected;
        if (h == 0)      equal_expected = 1;               /* both all-zero */
        else             equal_expected = partial[h - 1] && partial[h];

        int bitwise_equal = memcmp(out_a, out_b, sizeof(out_a)) == 0;
        if (equal_expected) {
            n_equal++;
            if (!bitwise_equal && equal_ok) { equal_ok = 0; first_bad_equal = h; }
        } else if (h >= 1) {
            n_differ++;
            if (bitwise_equal && differ_ok) { differ_ok = 0; first_bad_differ = h; }
        }
    }

    CHECK(no_nan, "partial-write guard: no NaN/Inf ever reaches the pipeline output");
    /* Same premise as the NaN test: no alignment generation, so the partial
     * schedule is the only source of identity frames. */
    CHECK(n_changed == 0,
          fmt_msg("partial-write run crosses no alignment generation, so no identity "
                  "reprime frame joins the schedule (%d CHANGED hops)", n_changed));
    CHECK(n_equal >= 10 && n_differ >= 10,
          fmt_msg("coverage: %d partial-identity hops and %d applied hops compared", n_equal, n_differ));
    CHECK(equal_ok,
          fmt_msg("rc==0 partial-write frames (first 100 bins only) are discarded: "
                  "output BIT-IDENTICAL to the NULL-model pipeline on all %d "
                  "fully-partial hops (NaN pre-fill catches the unwritten bins; "
                  "first bad hop %d [-1=none])", n_equal, first_bad_equal));
    CHECK(differ_ok,
          fmt_msg("fully-written frames recover: output DIFFERS from the NULL-model "
                  "pipeline on all %d hops with a clean applied frame (first bad "
                  "hop %d [-1=none])", n_differ, first_bad_differ));

    mu_destroy(pa);
    mu_destroy(pb);
}

/* 11. Identity-reprime straddle derivation: per branch, how many frames emitted at or after
 *     an alignment boundary at hop T still hold a pre-boundary hop in their two-hop window.
 *     One unit impulse at the middle of hop T-1 (window weight 0.707 in both covering
 *     frames) is marked on the error or the far branch in separate runs. The count comes
 *     from a boundary-free control (FIXED delay 0, no reprime armed). The boundary run
 *     (FIXED delay = T hops) must skip exactly ULCNET_NN_STAGE_REPRIME_FRAMES frames,
 *     at hops T.., and the first frame the model then sees must be marker-free on both
 *     branches. MUTATION: no reprime, or one frame too few, turns that red. */
enum {
    RP_T        = 8,            /* boundary hop under test                 */
    RP_MARK_HOP = RP_T - 1,     /* last hop pushed before the boundary     */
    RP_MARK_POS = HOP / 2,      /* impulse position inside that hop        */
    RP_RUN      = 14,           /* hops per probe run                      */
    RP_WINDOW   = 4,            /* hops examined from RP_T on              */
    RP_MAXCALL  = 32
};
/* A windowed unit impulse reaches every bin at |w| = 0.707; the residue a
 * marker-free frame can still carry (mic-HPF ring on the error branch, plain
 * zero on the far branch) is orders below. The margin is asserted, so this
 * floor can never silently become a classifier of noise. */
static const float RP_FLOOR = 0.05f;

typedef struct {
    int   hop;                      /* current hop; set by the test        */
    int   calls;
    int   call_hop[RP_MAXCALL];
    float err_peak[RP_MAXCALL];
    float far_peak[RP_MAXCALL];
    int   resets;
    int   overflow;
} ProbeModelState;

static float spec_peak(const float* re, const float* im) {
    float m = 0.0f;
    for (int k = 0; k < ULCNET_BINS; k++) {
        float a = fabsf(re[k]);
        if (a > m) m = a;
        a = fabsf(im[k]);
        if (a > m) m = a;
    }
    return m;
}

static int probe_infer(void* user,
                       const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                       const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                       float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    ProbeModelState* st = (ProbeModelState*)user;
    if (st->calls < RP_MAXCALL) {
        st->call_hop[st->calls] = st->hop;
        st->err_peak[st->calls] = spec_peak(err_re, err_im);
        st->far_peak[st->calls] = spec_peak(far_re, far_im);
        st->calls++;
    } else {
        st->overflow = 1;
    }
    memcpy(out_re, err_re, ULCNET_BINS * sizeof(float));
    memcpy(out_im, err_im, ULCNET_BINS * sizeof(float));
    return 0;
}

static void probe_reset(void* user) { ((ProbeModelState*)user)->resets++; }

/* One probe run. fixed_delay == 0 is the boundary-free control;
 * fixed_delay == RP_T*HOP puts the FIXED raw->aligned switch on hop RP_T.
 * mark_far selects which branch carries the impulse. */
static int reprime_probe_run(int fixed_delay, int mark_far, ProbeModelState* st) {
    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    MonoUlcnet* p;
    float mic[HOP], ref[HOP], out[HOP];

    memset(st, 0, sizeof(*st));
    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = fixed_delay;
    cfg.model.user  = st;
    cfg.model.infer = probe_infer;
    cfg.model.io_descriptor = test_io_descriptor();
    cfg.model.reset = probe_reset;
    p = mu_create(&cfg);
    if (!p) return -1;

    for (int h = 0; h < RP_RUN; h++) {
        memset(mic, 0, sizeof(mic));
        memset(ref, 0, sizeof(ref));
        if (h == RP_MARK_HOP) {
            if (mark_far) ref[RP_MARK_POS] = 1.0f;
            else          mic[RP_MARK_POS] = 1.0f;
        }
        st->hop = h;
        mu_process(p, mic, ref, out);
    }
    mu_destroy(p);
    return 0;
}

/* Frames emitted at hops [RP_T, RP_T+RP_WINDOW) that still carry the marker, plus the
 * classification margin (smallest marked peak / largest unmarked peak). On the far
 * branch the memory is exactly the window straddle, which derives REPRIME_FRAMES. On
 * the error branch the marker passes the AEC filter first, so its memory is the
 * straddle plus the filter's causal tail; that branch only bounds the straddle from
 * above. */
static int reprime_straddle_count(const ProbeModelState* st, int far_branch,
                                  float* dirty_min, float* clean_max) {
    int n = 0;
    *dirty_min = 1e30f;
    *clean_max = 0.0f;
    for (int c = 0; c < st->calls; c++) {
        float peak = far_branch ? st->far_peak[c] : st->err_peak[c];
        if (st->call_hop[c] < RP_T || st->call_hop[c] >= RP_T + RP_WINDOW)
            continue;
        if (peak > RP_FLOOR) {
            n++;
            if (peak < *dirty_min) *dirty_min = peak;
        } else if (peak > *clean_max) {
            *clean_max = peak;
        }
    }
    return n;
}

static void test_reprime_straddle_derivation(void) {
    ProbeModelState err_ctl, far_ctl, err_bnd, far_bnd;
    float dmin, cmax;
    int straddle_err, straddle_far;
    int expected_frames = 0;
    int first_visible_err = -1, first_visible_far = -1;
    float first_visible_err_peak = -1.0f, first_visible_far_peak = -1.0f;
    int stepped_inside_reprime = 0;

    for (int h = 0; h < RP_RUN; h++) expected_frames += frames_at_hop(h);

    if (reprime_probe_run(0, 0, &err_ctl) != 0 ||
        reprime_probe_run(0, 1, &far_ctl) != 0 ||
        reprime_probe_run(RP_T * HOP, 0, &err_bnd) != 0 ||
        reprime_probe_run(RP_T * HOP, 1, &far_bnd) != 0) {
        fprintf(stderr, "FAIL: setup (create) for reprime derivation test\n");
        g_failures++;
        return;
    }

    /* ---- the control runs really are boundary-free and reprime-free ---- */
    CHECK(err_ctl.resets == 0 && far_ctl.resets == 0 &&
          err_ctl.calls == expected_frames && far_ctl.calls == expected_frames &&
          !err_ctl.overflow && !far_ctl.overflow,
          fmt_msg("derivation control (FIXED delay 0) has no alignment boundary: "
                  "0 model resets and all %d emitted frames stepped "
                  "(err %d calls/%d resets, far %d calls/%d resets)",
                  expected_frames, err_ctl.calls, err_ctl.resets,
                  far_ctl.calls, far_ctl.resets));

    /* ---- derive the straddle count, per branch ---- */
    straddle_err = reprime_straddle_count(&err_ctl, 0, &dmin, &cmax);
    printf("reprime derivation (mono, ERROR branch): %d straddling frames at hops "
           "%d..%d; marked peak >= %.4f, unmarked peak <= %.3e\n",
           straddle_err, RP_T, RP_T + RP_WINDOW - 1, dmin, cmax);
    CHECK(straddle_err >= ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("mono ERROR branch: chain memory of %d frames from hop %d on is at "
                  "least ULCNET_NN_STAGE_REPRIME_FRAMES (%d) -- it carries the "
                  "AEC filter tail on top of the straddle, so it bounds rather than "
                  "derives it", straddle_err, RP_T,
                  (int)ULCNET_NN_STAGE_REPRIME_FRAMES));
    /* No absolute floor on dmin here: the impulse arrives spread by the AEC
     * filter, so its peak is grid-dependent. The separation ratio is what
     * makes the classification sound. */
    CHECK(straddle_err > 0 && dmin > 20.0f * cmax,
          fmt_msg("mono ERROR branch marker is unambiguous (marked %.4f vs unmarked "
                  "%.3e, ratio %.1fx)", dmin, cmax,
                  cmax > 0.0f ? (double)(dmin / cmax) : 1e30));

    straddle_far = reprime_straddle_count(&far_ctl, 1, &dmin, &cmax);
    printf("reprime derivation (mono, FAR branch):   %d straddling frames at hops "
           "%d..%d; marked peak >= %.4f, unmarked peak <= %.3e\n",
           straddle_far, RP_T, RP_T + RP_WINDOW - 1, dmin, cmax);
    /* The unfiltered branch IS the derivation: no filter tail, so its chain
     * memory is the window straddle and nothing else. */
    CHECK(straddle_far == ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("mono FAR branch: %d emitted frames from hop %d on still contain "
                  "pre-switch samples == ULCNET_NN_STAGE_REPRIME_FRAMES (%d)",
                  straddle_far, RP_T, (int)ULCNET_NN_STAGE_REPRIME_FRAMES));
    CHECK(straddle_far > 0 && dmin > 20.0f * cmax && dmin > 0.5f,
          fmt_msg("mono FAR branch marker is unambiguous (marked %.4f vs unmarked "
                  "%.3e, ratio %.1fx)", dmin, cmax,
                  cmax > 0.0f ? (double)(dmin / cmax) : 1e30));

    /* ---- the boundary runs: the stage skips exactly those frames ---- */
    CHECK(err_bnd.resets == 1 && far_bnd.resets == 1,
          fmt_msg("boundary run: FIXED raw->aligned switch fired model->reset once "
                  "(err %d, far %d)", err_bnd.resets, far_bnd.resets));
    for (int c = 0; c < err_bnd.calls; c++) {
        if (err_bnd.call_hop[c] >= RP_T &&
            err_bnd.call_hop[c] < RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES)
            stepped_inside_reprime++;
        if (err_bnd.call_hop[c] >= RP_T && first_visible_err < 0) {
            first_visible_err = err_bnd.call_hop[c];
            first_visible_err_peak = err_bnd.err_peak[c];
        }
    }
    for (int c = 0; c < far_bnd.calls; c++) {
        if (far_bnd.call_hop[c] >= RP_T &&
            far_bnd.call_hop[c] < RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES)
            stepped_inside_reprime++;
        if (far_bnd.call_hop[c] >= RP_T && first_visible_far < 0) {
            first_visible_far = far_bnd.call_hop[c];
            first_visible_far_peak = far_bnd.far_peak[c];
        }
    }
    printf("reprime resume (mono): first model-visible frame after the boundary is "
           "hop %d; err peak %.3e, far peak %.3e (floor %.3f)\n",
           first_visible_err, first_visible_err_peak, first_visible_far_peak,
           RP_FLOOR);
    /* The primary claim first, then the bookkeeping that follows from it.
     * Only the far branch can carry it: the error branch's first visible
     * frame legitimately holds AEC filter tail, which is filtered echo of
     * pre-switch content rather than a pre-switch SAMPLE, and no reprime
     * length can remove it. */
    CHECK(first_visible_far_peak >= 0.0f && first_visible_far_peak <= RP_FLOOR,
          fmt_msg("the first model-visible frame after the reprime contains NO "
                  "pre-switch sample on the far branch (far peak %.3e <= %.3f; "
                  "err peak %.3e is AEC filter tail, not a straddling sample)",
                  first_visible_far_peak, RP_FLOOR, first_visible_err_peak));
    CHECK(stepped_inside_reprime == 0,
          fmt_msg("boundary run: the model is not stepped on any hop in [%d, %d) "
                  "(%d stray calls)", RP_T,
                  RP_T + (int)ULCNET_NN_STAGE_REPRIME_FRAMES,
                  stepped_inside_reprime));
    CHECK(first_visible_err == RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES &&
          first_visible_far == RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("boundary run: stepping resumes on hop %d (err) / %d (far), "
                  "expected %d", first_visible_err, first_visible_far,
                  RP_T + (int)ULCNET_NN_STAGE_REPRIME_FRAMES));
    CHECK(err_bnd.calls == expected_frames - ULCNET_NN_STAGE_REPRIME_FRAMES &&
          far_bnd.calls == expected_frames - ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("boundary run: exactly %d of %d emitted frames skipped inference "
                  "(err %d calls, far %d calls)",
                  (int)ULCNET_NN_STAGE_REPRIME_FRAMES, expected_frames,
                  err_bnd.calls, far_bnd.calls));
}

/* =========================================================================
 * 12. Identity-reprime behaviour: exactly REPRIME_FRAMES identity frames per alignment
 *     generation, for a mid-stream MATCHED CHANGED hop and for FIXED ring-fill
 *     completion. The test predicts the infer delta per hop from its own copy of the
 *     policy and compares with the counting model; reset stays once per generation.
 * ========================================================================= */
static void test_reprime_behavior(void) {
    enum { N = 60, FIXED_HOPS = 8 };
    CountingModelState st = {0, 0};
    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    MonoUlcnet* p;
    float mic[HOP], ref[HOP], out[HOP];
    EchoSim sim;
    int armed = 0, n_changed = 0, skipped_total = 0;
    int pattern_ok = 1, first_bad_hop = -1;
    int boundary_hop = -1, resumed_hop = -1;
    int calls_before, delta, frames, skipped_here, expected_delta;

    /* ---- (a) mid-stream MATCHED CHANGED ---- */
    cfg.model.user  = &st;
    cfg.model.infer = counting_infer;
    cfg.model.io_descriptor = test_io_descriptor();
    cfg.model.reset = counting_reset;
    p = mu_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for reprime behaviour test\n"); g_failures++; return; }

    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);
    for (int h = 0; h < N; h++) {
        calls_before = st.infer_calls;
        echo_sim_hop(&sim, mic, ref);
        mu_process(p, mic, ref, out);
        if (pipeline_delay_state(p) == AEC_LINEAR_DELAY_CHANGED) {
            armed = ULCNET_NN_STAGE_REPRIME_FRAMES;
            n_changed++;
            if (boundary_hop < 0) boundary_hop = h;
        }
        frames = frames_at_hop(h);
        skipped_here = reprime_take(&armed, frames);
        skipped_total += skipped_here;
        expected_delta = frames - skipped_here;
        delta = st.infer_calls - calls_before;
        if (delta != expected_delta && pattern_ok) {
            pattern_ok = 0;
            first_bad_hop = h;
        }
        if (boundary_hop >= 0 && resumed_hop < 0 && delta > 0 && h > boundary_hop)
            resumed_hop = h;
    }

    CHECK(n_changed >= 1,
          fmt_msg("reprime behaviour: the run really crossed an alignment generation (%d)",
                  n_changed));
    CHECK(pattern_ok,
          fmt_msg("reprime behaviour (MATCHED): infer is stepped once per emitted frame "
                  "EXCEPT the %d reprime frames after each CHANGED hop "
                  "(first mismatching hop %d [-1=none])",
                  (int)ULCNET_NN_STAGE_REPRIME_FRAMES, first_bad_hop));
    CHECK(skipped_total == n_changed * ULCNET_NN_STAGE_REPRIME_FRAMES &&
          st.infer_calls == N - skipped_total,
          fmt_msg("reprime behaviour (MATCHED): %d frames skipped inference over %d "
                  "generations (%d infer calls in %d hops)",
                  skipped_total, n_changed, st.infer_calls, N));
    CHECK(resumed_hop == boundary_hop + ULCNET_NN_STAGE_REPRIME_FRAMES,
          fmt_msg("reprime behaviour (MATCHED): stepping stops on the boundary hop %d "
                  "and resumes on hop %d (expected %d)", boundary_hop, resumed_hop,
                  boundary_hop + (int)ULCNET_NN_STAGE_REPRIME_FRAMES));
    CHECK(st.reset_calls == n_changed,
          fmt_msg("reprime behaviour (MATCHED): still exactly one model->reset per "
                  "generation (%d resets, %d generations)", st.reset_calls, n_changed));
    mu_destroy(p);

    /* ---- (b) FIXED ring-fill completion ---- */
    {
        CountingModelState fst = {0, 0};
        MonoUlcnetConfig fcfg = mu_default_config(ULCNET_SR);
        int fixed_calls_at_boundary = -1, fixed_calls_after = -1;
        int fixed_boundary = -1;

        fcfg.delay_mode = AEC_DELAY_FIXED;
        fcfg.fixed_delay_samples = FIXED_HOPS * HOP;
        fcfg.model.user  = &fst;
        fcfg.model.infer = counting_infer;
        fcfg.model.io_descriptor = test_io_descriptor();
        fcfg.model.reset = counting_reset;
        p = mu_create(&fcfg);
        if (!p) { fprintf(stderr, "FAIL: setup (create) for FIXED reprime test\n"); g_failures++; return; }

        lcg_state = 0xA11FEEDu;
        armed = 0;
        pattern_ok = 1;
        first_bad_hop = -1;
        skipped_total = 0;
        for (int h = 0; h < N; h++) {
            AecLinearDelayState prev = pipeline_delay_state(p);
            calls_before = fst.infer_calls;
            memset(mic, 0, sizeof(mic));
            for (int i = 0; i < HOP; i++) ref[i] = lcg_sample();
            mu_process(p, mic, ref, out);
            if (h != 0 && prev == AEC_LINEAR_DELAY_UNLOCKED &&
                pipeline_delay_state(p) == AEC_LINEAR_DELAY_LOCKED) {
                armed = ULCNET_NN_STAGE_REPRIME_FRAMES;
                fixed_boundary = h;
            }
            frames = frames_at_hop(h);
            skipped_here = reprime_take(&armed, frames);
            skipped_total += skipped_here;
            delta = fst.infer_calls - calls_before;
            if (delta != frames - skipped_here && pattern_ok) {
                pattern_ok = 0;
                first_bad_hop = h;
            }
            if (fixed_boundary >= 0 && h == fixed_boundary)
                fixed_calls_at_boundary = delta;
            if (fixed_boundary >= 0 &&
                h == fixed_boundary + ULCNET_NN_STAGE_REPRIME_FRAMES)
                fixed_calls_after = delta;
        }

        CHECK(fixed_boundary == FIXED_HOPS,
              fmt_msg("reprime behaviour (FIXED): the raw->aligned switch is on hop %d "
                      "(expected %d)", fixed_boundary, FIXED_HOPS));
        CHECK(pattern_ok && skipped_total == ULCNET_NN_STAGE_REPRIME_FRAMES,
              fmt_msg("reprime behaviour (FIXED): exactly %d frames skipped inference at "
                      "the fill completion (%d skipped, first mismatching hop %d [-1=none])",
                      (int)ULCNET_NN_STAGE_REPRIME_FRAMES, skipped_total,
                      first_bad_hop));
        CHECK(fixed_calls_at_boundary == 0 && fixed_calls_after == 1,
              fmt_msg("reprime behaviour (FIXED): the boundary hop steps the model 0 times "
                      "and hop boundary+%d steps it once (%d then %d)",
                      (int)ULCNET_NN_STAGE_REPRIME_FRAMES,
                      fixed_calls_at_boundary, fixed_calls_after));
        CHECK(fst.reset_calls == 1,
              fmt_msg("reprime behaviour (FIXED): exactly one model->reset (%d)",
                      fst.reset_calls));
        CHECK(fst.infer_calls == N - ULCNET_NN_STAGE_REPRIME_FRAMES,
              fmt_msg("reprime behaviour (FIXED): %d infer calls over %d hops "
                      "(%d emitted frames minus the reprime)",
                      fst.infer_calls, N, N));
        mu_destroy(p);
    }
}

/* =========================================================================
 * Known-delay profile check (product delay gate, not an audio-quality run): a
 * two-path echo with an exactly known bulk delay, checked against ground truth for
 * acquisition, coverage boundary, mislock detectability and cost. Mono has one AEC,
 * so the whole per-filter pool cost lands in its pool.
 * ========================================================================= */

/* A bank of n filters reaches (n-1)*DA_FILTER_INTRA_SHIFT + (DA_FILTER_SIZE - 11)
 * downsampled samples (the -11 is the `lag < filter_size - 10` reliability cut),
 * decimation DA_DOWN_SAMPLING_FACTOR. */
#define KD_RELIABLE_SAMPLES(n) \
    ((((n) - 1) * DA_FILTER_INTRA_SHIFT * DA_DOWN_SAMPLING_FACTOR + \
      (DA_FILTER_SIZE - 11) * DA_DOWN_SAMPLING_FACTOR) * DELAY_RATE_FACTOR)
/* Applied alignment may be EARLY of the true echo (PBFDKF then models a
 * positive residual) but never LATE; measured shortfall is 64-80 samples. */
#define KD_MAX_UNDERSHOOT (128 * DELAY_RATE_FACTOR)
/* 960 ms acquisition budget in wall-clock time on any grid. */
#define KD_MAX_LOCK_HOPS \
    ((960 * ULCNET_SR + 1000 * HOP - 1) / (1000 * HOP))

typedef struct {
    int locked;
    int lock_hop;
    int applied_delay;
    int changed_hops;
    double us_per_hop;
} MonoKnownDelayRun;

/* Two-path synthetic echo (dominant + optional weaker early path) through a
 * MATCHED mono pipeline with the given bank size. */
static void mono_known_delay_run(int num_filters, int hops,
                                 int dominant_delay, float dominant_gain,
                                 int early_delay, float early_gain,
                                 MonoKnownDelayRun* out) {
    const int kd_pad = 16384 * DELAY_RATE_FACTOR;
    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    MonoUlcnet* p;
    float* far_hist;
    float mic[HOP], ref[HOP], outbuf[HOP];
    clock_t t0, t1;
    int hop, i;

    memset(out, 0, sizeof(*out));
    out->lock_hop = -1;
    out->applied_delay = -1;

    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = num_filters;
    p = mu_create(&cfg);
    if (!p) return;

    far_hist = (float*)malloc((size_t)(hops * HOP + kd_pad) * sizeof(float));
    if (!far_hist) { mu_destroy(p); return; }
    lcg_state = 0xC0FFEEu;
    for (i = 0; i < hops * HOP + kd_pad; i++) far_hist[i] = lcg_sample();

    t0 = clock();
    for (hop = 0; hop < hops; hop++) {
        int base = hop * HOP + kd_pad;
        for (i = 0; i < HOP; i++) {
            int t = base + i;
            ref[i] = far_hist[t];
            mic[i] = dominant_gain * far_hist[t - dominant_delay] +
                     early_gain * far_hist[t - early_delay];
        }
        mu_process(p, mic, ref, outbuf);
        {
            AecLinearDelayState ds = pipeline_delay_state(p);
            if (ds == AEC_LINEAR_DELAY_CHANGED) out->changed_hops++;
            if (!out->locked && (ds == AEC_LINEAR_DELAY_LOCKED ||
                                 ds == AEC_LINEAR_DELAY_CHANGED)) {
                out->locked = 1;
                out->lock_hop = hop;
                out->applied_delay = pipeline_delay_samples(p);
            }
        }
    }
    t1 = clock();
    out->us_per_hop = hops > 0
        ? (double)(t1 - t0) * 1e6 / (double)CLOCKS_PER_SEC / (double)hops
        : 0.0;

    free(far_hist);
    mu_destroy(p);
}

static void test_known_delay_profile(void) {
    static const int inside_ms[6] = { 0, 125, 221, 317, 413, 509 };
    MonoKnownDelayRun mislock, control, cost;
    MonoUlcnetMemReq req[6];
    int n;
    int mem_ok = 1;
    int mislock_error;

    printf("known-delay acquisition (mono, %d Hz, hop %d, synthetic echo):\n",
           ULCNET_SR, HOP);
    for (n = 1; n <= 5; n++) {
        MonoKnownDelayRun in_range, out_of_range;
        int inside = inside_ms[n] * ULCNET_SR / 1000;
        int outside = (n == 5) ? 9271 * DELAY_RATE_FACTOR
                               : inside_ms[n + 1] * ULCNET_SR / 1000;

        mono_known_delay_run(n, 200, inside, 0.6f, inside, 0.0f, &in_range);
        mono_known_delay_run(n, 200, outside, 0.6f, outside, 0.0f, &out_of_range);

        printf("  n=%d ceiling %d samples (%.2f ms): in-range %d ms -> lock hop %d "
               "applied %d (short by %d); out-of-range %d ms -> %s\n",
               n, KD_RELIABLE_SAMPLES(n),
               KD_RELIABLE_SAMPLES(n) * 1000.0 / ULCNET_SR,
               inside_ms[n], in_range.lock_hop, in_range.applied_delay,
               in_range.locked ? inside - in_range.applied_delay : -1,
               outside * 1000 / ULCNET_SR,
               out_of_range.locked ? "LOCKED" : "no lock");

        CHECK(in_range.locked && in_range.lock_hop >= 0 &&
              in_range.lock_hop < KD_MAX_LOCK_HOPS,
              fmt_msg("mono n=%d acquires a %d ms bulk delay within %d hops "
                      "(lock hop %d, inside its %.2f ms ceiling)",
                      n, inside_ms[n], KD_MAX_LOCK_HOPS, in_range.lock_hop,
                      KD_RELIABLE_SAMPLES(n) * 1000.0 / ULCNET_SR));
        CHECK(in_range.locked &&
              inside - in_range.applied_delay >= 0 &&
              inside - in_range.applied_delay <= KD_MAX_UNDERSHOOT,
              fmt_msg("mono n=%d applied delay is early-or-exact and short by at "
                      "most %d samples (true %d, applied %d)",
                      n, KD_MAX_UNDERSHOOT, inside, in_range.applied_delay));
        CHECK(in_range.changed_hops == 1,
              fmt_msg("mono n=%d reports exactly one alignment generation for a "
                      "static delay (%d)", n, in_range.changed_hops));
        CHECK(outside > KD_RELIABLE_SAMPLES(n) && !out_of_range.locked,
              fmt_msg("mono n=%d does NOT acquire a %d ms bulk delay (beyond its "
                      "%.2f ms ceiling)", n, outside * 1000 / ULCNET_SR,
                      KD_RELIABLE_SAMPLES(n) * 1000.0 / ULCNET_SR));
    }

    /* Mislock detectability: an out-of-range dominant path plus an in-range early
     * reflection makes the estimator lock onto the wrong path with a LOCKED seam state.
     * Only a comparison against the independently known delay catches it. */
    mono_known_delay_run(5, 200, 9271 * DELAY_RATE_FACTOR, 0.6f,
                         512 * DELAY_RATE_FACTOR, 0.5f, &mislock);
    mono_known_delay_run(5, 200, 3536 * DELAY_RATE_FACTOR, 0.6f,
                         512 * DELAY_RATE_FACTOR, 0.5f, &control);
    mislock_error = mislock.locked
        ? 9271 * DELAY_RATE_FACTOR - mislock.applied_delay : -1;
    printf("known-delay mislock (mono): dominant 9271 (579.44 ms) + early 512 "
           "(32.00 ms) -> lock hop %d applied %d (%.2f ms), wrong by %d samples "
           "(%.2f ms)\n",
           mislock.lock_hop, mislock.applied_delay,
           mislock.applied_delay * 1000.0 / ULCNET_SR,
           mislock_error, mislock_error * 1000.0 / ULCNET_SR);
    CHECK(mislock.locked,
          "mono: an in-range early path makes an out-of-range bulk delay lock anyway");
    CHECK(mislock.locked && mislock_error > KD_MAX_UNDERSHOOT,
          fmt_msg("mono: ground-truth comparison FLAGS the mislock (applied short "
                  "by %d samples = %.2f ms, far past the %d-sample alignment "
                  "contract); the LOCKED seam state alone cannot",
                  mislock_error, mislock_error * 1000.0 / ULCNET_SR,
                  KD_MAX_UNDERSHOOT));
    CHECK(control.locked &&
          3536 * DELAY_RATE_FACTOR - control.applied_delay >= 0 &&
          3536 * DELAY_RATE_FACTOR - control.applied_delay <= KD_MAX_UNDERSHOOT,
          fmt_msg("mono control: with the dominant path in range the SAME check "
                  "stays quiet (applied %d, short by %d)",
                  control.applied_delay,
                  control.locked
                      ? 3536 * DELAY_RATE_FACTOR - control.applied_delay : -1));

    /* RAM: the per-filter pool cost lands in this single AEC instance. */
    for (n = 1; n <= 5; n++) {
        MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
        cfg.delay_num_filters = n;
        if (mu_get_mem_requirements(&cfg, &req[n]) != 0)
            mem_ok = 0;
    }
    CHECK(mem_ok, "mono pool query answers for every n=1..5");
    if (mem_ok) {
        printf("mono pool vs matched-filter bank size (%d Hz, fft %d):\n",
               ULCNET_SR, ULCNET_N_FFT);
        for (n = 1; n <= 5; n++) {
            printf("  n=%d  total %llu", n, (unsigned long long)req[n].bytes);
            if (n > 1)
                printf("  (+%lld vs n=%d)",
                       (long long)req[n].bytes - (long long)req[n - 1].bytes, n - 1);
            printf("\n");
        }
        for (n = 2; n <= 5; n++) {
            CHECK(req[n].bytes > req[n - 1].bytes,
                  fmt_msg("mono n=%d costs strictly more than n=%d (%llu > %llu)",
                          n, n - 1, (unsigned long long)req[n].bytes,
                          (unsigned long long)req[n - 1].bytes));
            CHECK((long long)req[n].bytes - (long long)req[n - 1].bytes ==
                  (long long)req[2].bytes - (long long)req[1].bytes,
                  fmt_msg("mono n=%d adds the same per-filter cost as every other "
                          "step (%lld bytes)", n,
                          (long long)req[n].bytes - (long long)req[n - 1].bytes));
        }
    }

    /* Rough per-hop CPU, recorded rather than tuned. The budget is the wall
     * time one hop of audio occupies on the compiled grid, so it retimes with
     * the grid instead of assuming 16 ms. Host measurement, liveness guard. */
    const double hop_budget_us = (double)HOP * 1e6 / (double)ULCNET_SR;
    mono_known_delay_run(5, 400, 3536 * DELAY_RATE_FACTOR, 0.6f,
                         3536 * DELAY_RATE_FACTOR, 0.0f, &cost);
    printf("known-delay cost (mono, n=5): %.1f us/hop, %.4f x real time "
           "(hop = %.2f ms of audio)\n",
           cost.us_per_hop, cost.us_per_hop / hop_budget_us, hop_budget_us / 1000.0);
    CHECK(cost.us_per_hop > 0.0 && cost.us_per_hop < hop_budget_us,
          fmt_msg("mono pipeline runs faster than real time on the host "
                  "(%.1f us/hop vs a %.0f us budget)",
                  cost.us_per_hop, hop_budget_us));
}

/* The phased hop (process_begin / nn_begin / nn_end / process_end) is
 * mu_process() with the model call moved out: a twin driven
 * phase by phase, with the same network run by the caller, reproduces the
 * callback-driven instance byte for byte -- across an alignment boundary (the
 * echo path is delayed, so the reprime hops are part of the claim). The
 * fail-open rows pin the ways a frame ends as the identity: a failed run, an
 * output with a non-finite bin (only nn_end's finite guard can catch it), a
 * frame left open, a frame never fetched. */
enum { PH_RUN, PH_FAIL, PH_NAN, PH_OPEN, PH_UNFETCHED };

static int phased_twin_matches(int mode, int hops) {
    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    MonoUlcnetConfig cb_cfg = cfg;
    NanModelState nan_state;
    FailModelState fail_state;
    MonoUlcnet *ph, *cb;
    EchoSim sim;
    float mic[ULCNET_HOP], ref[ULCNET_HOP], a[ULCNET_HOP], b[ULCNET_HOP];
    UlcnetNnJob job;
    int h, ok = 1, nonzero = 0, ran = 0;

    memset(&nan_state, 0, sizeof(nan_state));
    nan_state.poison_now = 1;
    memset(&fail_state, 0, sizeof(fail_state));
    fail_state.fail_now = 1;
    if (mode == PH_RUN || mode == PH_FAIL || mode == PH_NAN) {
        cb_cfg.model.infer = mode == PH_RUN ? halving_infer
                           : mode == PH_FAIL ? failing_infer : nan_infer;
        cb_cfg.model.user = mode == PH_NAN ? (void*)&nan_state
                          : mode == PH_FAIL ? (void*)&fail_state : NULL;
        cb_cfg.model.io_descriptor = test_io_descriptor();
    }
    ph = mu_create(&cfg);
    cb = mu_create(&cb_cfg);
    if (!ph || !cb) {
        mu_destroy(ph);
        mu_destroy(cb);
        return 0;
    }
    echo_sim_init(&sim, ECHO_DELAY, 0xBEEFu + (uint32_t)mode);
    for (h = 0; h < hops && ok; ++h) {
        int r = 0;
        echo_sim_hop(&sim, mic, ref);
        ok = mu_process(cb, mic, ref, b) == 0 &&
             mu_process_begin(ph, mic, ref) == 0;
        if (!ok) break;
        if (mode != PH_UNFETCHED) {
            r = mu_nn_begin(ph, &job);
            if (r < 0) { ok = 0; break; }
        }
        if (r == 1) {
            ran++;
            if (mode == PH_RUN)
                ok = halving_infer(NULL, job.err_re, job.err_im, job.far_re, job.far_im,
                                   job.out_re, job.out_im) == 0 &&
                     mu_nn_end(ph, 0) == 0;
            else if (mode == PH_FAIL)
                ok = mu_nn_end(ph, -1) == 0;
            else if (mode == PH_NAN)
                ok = nan_infer(&nan_state, job.err_re, job.err_im, job.far_re, job.far_im,
                               job.out_re, job.out_im) == 0 &&
                     mu_nn_end(ph, 0) == 0;
            /* PH_OPEN: the frame stays open into process_end */
        }
        ok = ok && mu_process_end(ph, a) == 0;
        if (!ok) break;
        if (memcmp(a, b, sizeof(a)) != 0) ok = 0;
        for (int i = 0; i < ULCNET_HOP; ++i) if (a[i] != 0.0f) nonzero = 1;
    }
    mu_destroy(ph);
    mu_destroy(cb);
    /* mode PH_UNFETCHED never fetches; every other mode must have run frames
     * (the first hops after the boundary are reprime and run none). */
    return ok && nonzero && (mode == PH_UNFETCHED || ran > 0);
}

static void test_phased_hop(void) {
    MonoUlcnetConfig cfg = mu_default_config(ULCNET_SR);
    MonoUlcnet* p = mu_create(&cfg);
    UlcnetNnJob job;
    float mic[ULCNET_HOP] = {0}, ref[ULCNET_HOP] = {0}, out[ULCNET_HOP];

    CHECK(phased_twin_matches(PH_RUN, 120),
          "phased hop with the caller's network equals the callback-driven pipeline byte for byte");
    CHECK(phased_twin_matches(PH_FAIL, 120), "phased: a failed run is the identity");
    CHECK(phased_twin_matches(PH_NAN, 120), "phased: a non-finite output bin is the identity");
    CHECK(phased_twin_matches(PH_OPEN, 120), "phased: a frame left open is the identity");
    CHECK(phased_twin_matches(PH_UNFETCHED, 120), "phased: a frame never fetched is the identity");

    CHECK(p != NULL, "phased: instance creates");
    if (!p) return;
    CHECK(mu_nn_begin(p, &job) == -1 &&
          mu_nn_end(p, 0) == -1 &&
          mu_process_end(p, out) == -1,
          "phased: nn/end calls before process_begin are refused");
    CHECK(mu_process_begin(NULL, mic, ref) == -1 &&
          mu_process_begin(p, NULL, ref) == -1 &&
          mu_process_begin(p, mic, NULL) == -1,
          "phased: NULL arguments are refused");
    CHECK(mu_process_begin(p, mic, ref) == 0 &&
          mu_process_begin(p, mic, ref) == -1,
          "phased: a second process_begin before process_end is refused");
    CHECK(mu_nn_begin(p, &job) == 1 &&
          mu_nn_begin(p, &job) == -1,
          "phased: one frame per hop; a second nn_begin is refused while it is open");
    CHECK(mu_nn_end(p, 0) == 0 &&
          mu_nn_end(p, 0) == -1 &&
          mu_nn_begin(p, &job) == 0,
          "phased: nn_end closes the frame; none is left in this hop");
    CHECK(mu_process_end(p, out) == 0 &&
          mu_process_end(p, out) == -1,
          "phased: process_end closes the hop; a second one is refused");
    mu_destroy(p);
}

int main(void) {
    printf("=== mono_alignulcnet: identity E2E (one-hop latency) ===\n");
    test_identity_e2e();

    printf("\n=== mono_alignulcnet: counting model (stepping + reset policy) ===\n");
    test_counting_model();

    printf("\n=== mono_alignulcnet: same-delay relock resets the model ===\n");
    test_relock_same_delay_resets_model();

    /* Grid-relative (2 * HOP), so it runs everywhere. */
    printf("\n=== mono_alignulcnet: FIXED first alignment resets the model ===\n");
    test_fixed_first_alignment_resets_model();

    /* The far branch derives REPRIME_FRAMES exactly; the error branch only bounds it
     * (its marker carries the AEC filter's causal tail). */
    printf("\n=== mono_alignulcnet: identity-reprime straddle derivation ===\n");
    test_reprime_straddle_derivation();

    printf("\n=== mono_alignulcnet: identity-reprime behaviour ===\n");
    test_reprime_behavior();

    printf("\n=== mono_alignulcnet: fail-open callback handling ===\n");
    test_fail_open_and_delay_gating();

    printf("\n=== mono_alignulcnet: grid and pool plan ===\n");
    test_grid_and_pool_plan();

    printf("\n=== mono_alignulcnet: create-vs-init parity (poisoned pool) ===\n");
    test_create_vs_init_parity();

    printf("\n=== mono_alignulcnet: NULL model == identity model ===\n");
    test_null_model_equals_identity_model();

    printf("\n=== mono_alignulcnet: raw-far timestamp after fixed-delay lock ===\n");
    test_raw_far_timestamp_after_fixed_delay_lock();

    printf("\n=== mono_alignulcnet: model applies while UNLOCKED ===\n");
    test_model_applies_unlocked();

    printf("\n=== mono_alignulcnet: NaN/Inf guard ===\n");
    test_nan_guard();

    printf("\n=== mono_alignulcnet: full-write contract (partial-write guard) ===\n");
    test_partial_write_guard();

    printf("\n=== mono_alignulcnet: phased hop (begin / nn / end) ===\n");
    test_phased_hop();

    printf("\n=== mono_alignulcnet: raw-far descriptor gate ===\n");
    test_raw_descriptor_gate();

    printf("\n=== mono_alignulcnet: known-delay profile (acquisition/coverage/mislock/cost) ===\n");
    test_known_delay_profile();

    if (g_failures) {
        fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
