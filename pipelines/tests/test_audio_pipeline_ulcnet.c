/**
 * tests/test_audio_pipeline_ulcnet.c — acceptance tests for
 * pipelines/audio_pipeline_ulcnet.h (mono AEC(linear) -> Align-ULCNet
 * neural post-filter pipeline).
 *
 * Style mirrors tests/test_audio_pipeline.c (CHECK macro, LCG synthetic
 * input, 0xA5-poisoned pool, PASS/FAIL prints, nonzero exit on failure).
 * Not a DSP-quality test — a contract test for the API surface, the
 * one-hop-latency STFT/WOLA integration, and the model-callback policy.
 *
 * Two synthetic signal shapes drive everything:
 *   - "zero-delay echo" (mic = 0.3*ref + noise): the AEC's linear filter
 *     converges before the matched filter acquires, so the Path-A
 *     already-cancelling protection keeps delay_state UNLOCKED for the
 *     whole run — used where a stable identity path is wanted (test 1).
 *   - "bulk-delay echo" (echo delayed ECHO_DELAY=2000 samples, beyond the
 *     832-tap filter reach): the filter cannot converge unaligned, so the
 *     matched filter acquires deterministically (~hop 15 with this seed):
 *     one CHANGED hop, then LOCKED for the rest of the run — used wherever
 *     the model must actually be APPLIED (tests 2/3/5).
 *
 * Cases:
 *   1. identity E2E (NULL model): out[hop p] ~= formed_history[p-1] within
 *      2e-4 (hop 0 exactly zeros) — proves the AecResContext.formed_hop tap
 *      choice, the one-hop algorithmic latency, and the STFT/WOLA closure.
 *   2. counting model: infer is stepped for every emitted frame (constant
 *      compute path): cumulative calls after hop #p == p+1 — 2 on hop #1
 *      (the 0/2/1 emission), 1 per hop after; model->reset fires exactly
 *      once per CHANGED hop during the run, plus exactly once more on
 *      audio_pipeline_ulcnet_reset(); a post-reset hop #0 emits 0 frames.
 *   3. fail-open + delay gating (ULCNET_FAR_ALIGNED): a model whose output
 *      halves the spectrum (rc=0) but doubles it on scheduled failing hops
 *      (rc!=0) — output is bit-identical to the NULL-model pipeline's
 *      exactly where the policy says identity must hold (every UNLOCKED
 *      hop, every failing frame) and differs exactly where the model
 *      output must be applied.
 *   4. pool rejection / reject-first config validation / init_ex 8-point
 *      descriptor gate / destroy idempotence / create-vs-init byte parity
 *      on a 0xA5-poisoned pool (patterns copied from test_audio_pipeline.c).
 *   5. NULL model == identity (copy err->out, rc=0) model: bit-identical
 *      output across UNLOCKED, CHANGED and LOCKED phases.
 *   6. far-timestamp (ULCNET_FAR_RAW, the default): far-passthrough model,
 *      silence on mic, one unit impulse in far at a known sample index —
 *      the impulse must land in the output at EXACTLY impulse_index + 256:
 *      the mono far tap is SAME-HOP with the error tap (no wrapper-side far
 *      compensation exists or is needed), and the centered ULCNet chain
 *      lags the input by exactly one hop. Documents the mono timing.
 *   7. RAW mode never gates on the delay lock: a 0.5x model in RAW mode
 *      diverges from the NULL-model pipeline from the FIRST emitted frame
 *      (hop #1), long before the delay acquires.
 *   8. NaN guard: a model returning rc==0 but a NaN-poisoned spectrum on
 *      scheduled hops — those frames take the identity path BITWISE (equal
 *      to the NULL-model pipeline under the same 50%-overlap mixing rule
 *      as test 3), the next clean frame is applied again, and no NaN ever
 *      reaches the output.
 *
 * Build (from pipelines/): `make test` (kiss backend). Standalone:
 *   cc -O2 -std=gnu99 -ffp-contract=off -I. -I../lib/aec/c_impl/include \
 *      -I../lib/aec/c_impl/example -I../../audio_common/include \
 *      -I../../audio_common/lib/kiss_fft -I../AIAEC/Align_ULCNet \
 *      -DAUDIO_PIPELINE_BACKEND_STR=\"kiss\" \
 *      tests/test_audio_pipeline_ulcnet.c audio_pipeline_ulcnet.c \
 *      ../AIAEC/Align_ULCNet/ulcnet_process.c \
 *      <libaec.a> <libaudio_common.a> -lm -o /tmp/tapu && /tmp/tapu
 */
#include "audio_pipeline_ulcnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>

#define HOP            ULCNET_HOP     /* 256 — pinned by the compiled grid  */
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

/* ---- LCG synthetic generator (mirrors test_audio_pipeline.c) ---- */
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

static AecLinearDelayState pipeline_delay_state(const AudioPipelineUlcnet* p) {
    AecLinearContext lctx;
    aec_get_linear_context(audio_pipeline_ulcnet_get_aec(p), &lctx);
    return lctx.delay_state;
}

/* Frames emitted by the 0/2/1 analysis contract at hop index h. */
static int frames_at_hop(int h) { return (h == 0) ? 0 : (h == 1) ? 2 : 1; }

/* =========================================================================
 * 1. identity E2E: NULL model, zero-delay echo. Every hop, BEFORE the next
 *    process call, ctx.formed_hop is recorded; out[hop p] must match
 *    formed_history[p-1] within 2e-4 and hop 0 must be exactly zeros.
 * ========================================================================= */
static void test_identity_e2e(void) {
    enum { N = 100 };
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnet* p = audio_pipeline_ulcnet_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for identity E2E test\n"); g_failures++; return; }

    CHECK(audio_pipeline_ulcnet_hop_size(p) == HOP,
          "hop_size accessor reports the compiled ULCNet hop (256)");

    float mic[HOP], ref[HOP];
    static float out_hist[N][HOP], formed_hist[N][HOP];
    EchoSim sim;
    echo_sim_init(&sim, 0, 0xC0FFEEu);

    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(p, mic, ref, out_hist[h]);
        /* Read the tap AFTER process, BEFORE the next AEC call — the seam
         * pointers alias AEC-internal buffers valid until then. */
        AecResContext rctx;
        aec_get_res_context(audio_pipeline_ulcnet_get_aec(p), &rctx);
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

    audio_pipeline_ulcnet_destroy(p);
}

/* =========================================================================
 * 2. counting model. Bulk-delay echo so the run crosses UNLOCKED ->
 *    CHANGED -> LOCKED. infer is stepped for EVERY emitted frame (constant
 *    compute), so cumulative calls after hop #p == p+1 (2 on hop #1);
 *    model->reset fires once per CHANGED hop + once per pipeline reset.
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

    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    cfg.model.user  = &st;
    cfg.model.infer = counting_infer;
    cfg.model.reset = counting_reset;

    AudioPipelineUlcnet* p = audio_pipeline_ulcnet_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for counting-model test\n"); g_failures++; return; }

    float mic[HOP], ref[HOP], out[HOP];
    EchoSim sim;
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);

    int calls_after_hop1 = -1;
    int pattern_ok = 1, first_bad_hop = -1;
    int n_changed = 0, n_locked = 0;
    int expected_calls = 0;
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(p, mic, ref, out);
        expected_calls += frames_at_hop(h);   /* == h+1 for h >= 1 */
        if (h == 1) calls_after_hop1 = st.infer_calls;
        if (st.infer_calls != expected_calls && pattern_ok) {
            pattern_ok = 0;
            first_bad_hop = h;
        }
        AecLinearDelayState ds = pipeline_delay_state(p);
        if (ds == AEC_LINEAR_DELAY_CHANGED) n_changed++;
        if (ds == AEC_LINEAR_DELAY_LOCKED)  n_locked++;
    }

    CHECK(calls_after_hop1 == 2,
          fmt_msg("counting model: 2 infer calls after hop #1 (0/2/1 emission; got %d)",
                  calls_after_hop1));
    CHECK(pattern_ok && st.infer_calls == N,
          fmt_msg("counting model: cumulative infer calls == hop_index+1 at every hop "
                  "(%d after %d hops, first bad hop %d [-1=none])",
                  st.infer_calls, N, first_bad_hop));
    CHECK(n_changed >= 1 && n_locked >= 1,
          fmt_msg("delay actually acquired during the run (%d CHANGED, %d LOCKED hops)",
                  n_changed, n_locked));
    CHECK(st.reset_calls == n_changed,
          fmt_msg("model->reset fired exactly once per CHANGED hop (%d resets, %d CHANGED)",
                  st.reset_calls, n_changed));

    int resets_before = st.reset_calls;
    audio_pipeline_ulcnet_reset(p);
    CHECK(st.reset_calls == resets_before + 1,
          "audio_pipeline_ulcnet_reset invokes model->reset exactly once");

    int calls_before = st.infer_calls;
    echo_sim_hop(&sim, mic, ref);
    audio_pipeline_ulcnet_process(p, mic, ref, out);
    int hop0_zero = 1;
    for (int i = 0; i < HOP; i++) if (out[i] != 0.0f) hop0_zero = 0;
    CHECK(st.infer_calls == calls_before && hop0_zero,
          "after pipeline reset the next hop is hop #0 again: 0 infer calls, all-zero output");

    audio_pipeline_ulcnet_destroy(p);
}

/* =========================================================================
 * 3. fail-open + delay gating, in ULCNET_FAR_ALIGNED (the mode that gates
 *    application on the lock; RAW never does -- see test 7). Model A halves
 *    the spectrum on success (rc=0) and doubles it on scheduled failing
 *    hops (rc!=0); pipeline B has a NULL model. Output hop p mixes the
 *    frames pushed at hops p-1 and p (50% WOLA overlap), so A == B bitwise
 *    exactly when BOTH contributing frames took the identity path
 *    (UNLOCKED or rc!=0), and A != B when any applied (locked, rc==0)
 *    frame contributed.
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

    AudioPipelineUlcnetConfig cfg_a = audio_pipeline_ulcnet_default_config(16000);
    cfg_a.model.user  = &st;
    cfg_a.model.infer = failing_infer;
    cfg_a.far_input_mode = ULCNET_FAR_ALIGNED;   /* the lock-gated mode */
    AudioPipelineUlcnetConfig cfg_b = audio_pipeline_ulcnet_default_config(16000); /* NULL model */
    cfg_b.far_input_mode = ULCNET_FAR_ALIGNED;

    AudioPipelineUlcnet* pa = audio_pipeline_ulcnet_create(&cfg_a);
    AudioPipelineUlcnet* pb = audio_pipeline_ulcnet_create(&cfg_b);
    if (!pa || !pb) {
        fprintf(stderr, "FAIL: setup (create) for fail-open test\n");
        g_failures++;
        if (pa) audio_pipeline_ulcnet_destroy(pa);
        if (pb) audio_pipeline_ulcnet_destroy(pb);
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

    for (int h = 0; h < N; h++) {
        st.fail_now = (fail_start >= 0 && h >= fail_start && h < fail_start + FAIL_LEN);
        if (st.fail_now) n_failing++;

        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(pa, mic, ref, out_a);
        audio_pipeline_ulcnet_process(pb, mic, ref, out_b);

        AecLinearDelayState ds_a = pipeline_delay_state(pa);
        AecLinearDelayState ds_b = pipeline_delay_state(pb);
        if (ds_a != ds_b) states_ok = 0;   /* model must not feed back into the AEC */
        if (ds_a == AEC_LINEAR_DELAY_UNLOCKED) n_unlocked++;
        if (lock_hop < 0 && ds_a != AEC_LINEAR_DELAY_UNLOCKED) {
            lock_hop = h;
            fail_start = h + FAIL_GAP;   /* schedule the failure window post-lock */
        }
        ident[h] = (ds_a == AEC_LINEAR_DELAY_UNLOCKED) || st.fail_now;

        /* equal_expected: hop 0 both zero; hop 1 mixes only frames pushed at
         * hop 1; hop p>=2 mixes frames pushed at hops p-1 and p. */
        int equal_expected;
        if (h == 0)      equal_expected = 1;
        else if (h == 1) equal_expected = ident[1];
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

    CHECK(lock_hop >= 1 && lock_hop <= MAX_LOCK_HOPS,
          fmt_msg("delay acquired at hop %d (<= %d) so both gated and applied phases ran",
                  lock_hop, MAX_LOCK_HOPS));
    CHECK(n_unlocked >= 2 && n_failing == FAIL_LEN,
          fmt_msg("coverage: %d UNLOCKED hops and %d scheduled infer failures", n_unlocked, n_failing));
    CHECK(states_ok, "model presence does not perturb the AEC (identical delay states A vs B)");
    CHECK(equal_ok,
          fmt_msg("fail-open: output is BIT-IDENTICAL to the NULL-model pipeline on all %d "
                  "identity-path hops (UNLOCKED bypass + discarded rc!=0 output; "
                  "first bad hop %d [-1=none])", n_equal_hops, first_bad_equal));
    CHECK(differ_ok,
          fmt_msg("applied path: output DIFFERS from the NULL-model pipeline on all %d hops "
                  "with a locked, successful model frame (first bad hop %d [-1=none])",
                  n_differ_hops, first_bad_differ));

    audio_pipeline_ulcnet_destroy(pa);
    audio_pipeline_ulcnet_destroy(pb);
}

/* =========================================================================
 * 4a. reject-first config validation (mirrors test_config_validation_rejects)
 * ========================================================================= */
static void test_config_validation_rejects(void) {
    AudioPipelineUlcnetMemReq req;

    CHECK(audio_pipeline_ulcnet_get_mem_requirements(NULL, &req) == -1,
          "get_mem_requirements rejects a NULL config");

    AudioPipelineUlcnetConfig good = audio_pipeline_ulcnet_default_config(16000);
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&good, NULL) == -1,
          "get_mem_requirements rejects a NULL out-param");

    static const int bad_rates[] = {8000, 44100, 48000};
    for (int i = 0; i < 3; i++) {
        AudioPipelineUlcnetConfig c = audio_pipeline_ulcnet_default_config(bad_rates[i]);
        CHECK(audio_pipeline_ulcnet_get_mem_requirements(&c, &req) == -1,
              fmt_msg("get_mem_requirements rejects sample_rate=%d (ULCNet grid is 16 kHz only)",
                      bad_rates[i]));
    }

    AudioPipelineUlcnetConfig bad_fft = audio_pipeline_ulcnet_default_config(16000);
    bad_fft.fft_size = 256;   /* the generic 16 kHz rate default — NOT this grid */
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&bad_fft, &req) == -1,
          "get_mem_requirements rejects fft_size=256 (compiled ULCNet grid is 512)");

    AudioPipelineUlcnetConfig bad_preset = audio_pipeline_ulcnet_default_config(16000);
    bad_preset.aec_preset = (AecPreset)99;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&bad_preset, &req) == -1,
          "get_mem_requirements rejects an out-of-enum aec_preset");

    CHECK(good.far_input_mode == ULCNET_FAR_RAW,
          "default config far_input_mode is ULCNET_FAR_RAW (checkpoint-compatible)");
    AudioPipelineUlcnetConfig bad_mode = audio_pipeline_ulcnet_default_config(16000);
    bad_mode.far_input_mode = (UlcnetFarInputMode)99;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&bad_mode, &req) == -1,
          "get_mem_requirements rejects an out-of-enum far_input_mode");
    AudioPipelineUlcnetConfig aligned_mode = audio_pipeline_ulcnet_default_config(16000);
    aligned_mode.far_input_mode = ULCNET_FAR_ALIGNED;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&aligned_mode, &req) == 0,
          "get_mem_requirements accepts ULCNET_FAR_ALIGNED");

    AudioPipelineUlcnetMemReq req0, req512;
    AudioPipelineUlcnetConfig c0 = audio_pipeline_ulcnet_default_config(16000);   /* fft 0 */
    AudioPipelineUlcnetConfig c512 = audio_pipeline_ulcnet_default_config(16000);
    c512.fft_size = 512;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&c0, &req0) == 0 &&
          audio_pipeline_ulcnet_get_mem_requirements(&c512, &req512) == 0 &&
          req0.bytes == req512.bytes && req0.bytes > 0,
          "fft_size=0 resolves to the same descriptor as the explicit 512");
    printf("       (descriptor: descriptor_version=%u bytes=%llu alignment=%u "
           "layout_version=%u backend_id=%u build_flags_hash=0x%08x)\n",
           req0.descriptor_version, (unsigned long long)req0.bytes, req0.alignment,
           req0.layout_version, req0.backend_id, req0.build_flags_hash);

    /* Same rejections on the init() entry point (one shared gate). */
    void* pool = NULL;
    if (posix_memalign(&pool, 16, (size_t)req0.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for config-validation init test\n");
        g_failures++;
        return;
    }
    AudioPipelineUlcnetConfig bad_init = audio_pipeline_ulcnet_default_config(16000);
    bad_init.fft_size = 256;
    CHECK(audio_pipeline_ulcnet_init(pool, (size_t)req0.bytes, &bad_init) == NULL,
          "audio_pipeline_ulcnet_init rejects fft_size=256 too");
    AudioPipelineUlcnetConfig bad_init_rate = audio_pipeline_ulcnet_default_config(48000);
    CHECK(audio_pipeline_ulcnet_init(pool, (size_t)req0.bytes, &bad_init_rate) == NULL,
          "audio_pipeline_ulcnet_init rejects sample_rate=48000 too");

    AudioPipelineUlcnet* p_ok = audio_pipeline_ulcnet_init(pool, (size_t)req0.bytes, &good);
    CHECK(p_ok != NULL, "pool is still usable via a valid config after rejected init() attempts");
    if (p_ok) audio_pipeline_ulcnet_destroy(p_ok);
    free(pool);
}

/* =========================================================================
 * 4b. pool rejection (mirrors test_pool_rejection)
 * ========================================================================= */
static void test_pool_rejection(void) {
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnetMemReq req;
    if (audio_pipeline_ulcnet_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for pool-rejection test\n");
        g_failures++;
        return;
    }

    /* +16 headroom so the misaligned base+1 probe is never an OOB setup. */
    void* pool = NULL;
    if (posix_memalign(&pool, 16, (size_t)req.bytes + 16) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for pool-rejection test\n");
        g_failures++;
        return;
    }

    CHECK(audio_pipeline_ulcnet_init((uint8_t*)pool + 1, (size_t)req.bytes, &cfg) == NULL,
          "audio_pipeline_ulcnet_init rejects a misaligned (base+1) pool");
    CHECK(audio_pipeline_ulcnet_init(pool, (size_t)req.bytes - 1, &cfg) == NULL,
          "audio_pipeline_ulcnet_init rejects an undersized (bytes-1) pool");

    AudioPipelineUlcnet* p_ok = audio_pipeline_ulcnet_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p_ok != NULL, "audio_pipeline_ulcnet_init accepts a correctly aligned/sized pool");
    if (p_ok) audio_pipeline_ulcnet_destroy(p_ok);
    free(pool);
}

/* =========================================================================
 * 4c. init_ex 8-point descriptor gate (mirrors test_init_ex_descriptor)
 * ========================================================================= */
static void test_init_ex_descriptor(void) {
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnetMemReq req;
    if (audio_pipeline_ulcnet_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for init_ex descriptor test\n");
        g_failures++;
        return;
    }
    void* pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for init_ex descriptor test\n");
        g_failures++;
        return;
    }

    AudioPipelineUlcnet* p_ok = audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p_ok != NULL, "init_ex accepts a correct/current descriptor");
    if (p_ok) audio_pipeline_ulcnet_destroy(p_ok);

    AudioPipelineUlcnet* p_null = audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, NULL);
    CHECK(p_null != NULL, "init_ex(expected=NULL) accepts, same as init");
    if (p_null) audio_pipeline_ulcnet_destroy(p_null);

    AudioPipelineUlcnetMemReq bad;

    bad = req; bad.descriptor_version = req.descriptor_version + 1;
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &bad) == NULL,
          "init_ex rejects a tampered descriptor_version");

    bad = req; bad.layout_version = req.layout_version + 1;
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &bad) == NULL,
          "init_ex rejects a tampered layout_version");

    bad = req; bad.backend_id = 99u;
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &bad) == NULL,
          "init_ex rejects a tampered backend_id");

    bad = req; bad.build_flags_hash = req.build_flags_hash ^ 0xFFFFFFFFu;
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &bad) == NULL,
          "init_ex rejects a tampered build_flags_hash");

    bad = req; bad.alignment = req.alignment * 2u;
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &bad) == NULL,
          "init_ex rejects a tampered alignment");

    bad = req; bad.reserved = 1u;
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &bad) == NULL,
          "init_ex rejects a nonzero reserved field");

    bad = req; bad.bytes = req.bytes - 1u;
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &bad) == NULL,
          "init_ex rejects expected->bytes smaller than the current requirement");

    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes - 1u, &cfg, &req) == NULL,
          "init_ex rejects an undersized pool even with a correct descriptor");

    AudioPipelineUlcnet* p_final = audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p_final != NULL, "pool is still usable via init_ex after rejected attempts");
    if (p_final) audio_pipeline_ulcnet_destroy(p_final);
    free(pool);
}

/* =========================================================================
 * 4d. destroy idempotence (pool instance) + pool reuse
 * ========================================================================= */
static void test_destroy_idempotence(void) {
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnetMemReq req;
    if (audio_pipeline_ulcnet_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup for destroy-idempotence test\n");
        g_failures++;
        return;
    }
    void* pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for destroy-idempotence test\n");
        g_failures++;
        return;
    }
    AudioPipelineUlcnet* p = audio_pipeline_ulcnet_init(pool, (size_t)req.bytes, &cfg);
    if (!p) {
        fprintf(stderr, "FAIL: init for destroy-idempotence test\n");
        g_failures++;
        free(pool);
        return;
    }
    audio_pipeline_ulcnet_destroy(p);
    audio_pipeline_ulcnet_destroy(p);      /* second call on the same pool instance */
    audio_pipeline_ulcnet_destroy(NULL);   /* NULL-safe */
    printf("PASS: audio_pipeline_ulcnet_destroy is idempotent (2x) and NULL-safe on a "
           "pool-resident instance\n");

    AudioPipelineUlcnet* p2 = audio_pipeline_ulcnet_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p2 != NULL, "pool is reusable via a fresh init after destroy");
    if (p2) audio_pipeline_ulcnet_destroy(p2);
    free(pool);
}

/* =========================================================================
 * 4e. create-vs-init byte parity on a 0xA5-poisoned pool (NULL model;
 *     bulk-delay echo so the parity run crosses the CHANGED/LOCKED
 *     transition too). Direct proof of the explicit-zeroing claim.
 * ========================================================================= */
static void test_create_vs_init_parity(void) {
    enum { N = 300 };
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnetMemReq req;
    if (audio_pipeline_ulcnet_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup for create-vs-init parity test\n");
        g_failures++;
        return;
    }

    AudioPipelineUlcnet* p_heap = audio_pipeline_ulcnet_create(&cfg);
    void* pool = NULL;
    if (!p_heap || posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: setup alloc for create-vs-init parity test\n");
        g_failures++;
        if (p_heap) audio_pipeline_ulcnet_destroy(p_heap);
        return;
    }
    memset(pool, 0xA5, (size_t)req.bytes);   /* dirty pool: init must not rely on zeros */
    AudioPipelineUlcnet* p_pool = audio_pipeline_ulcnet_init(pool, (size_t)req.bytes, &cfg);
    if (!p_pool) {
        fprintf(stderr, "FAIL: init on a poisoned (0xA5) pool\n");
        g_failures++;
        audio_pipeline_ulcnet_destroy(p_heap);
        free(pool);
        return;
    }

    float mic[HOP], ref[HOP];
    static float out_heap[N][HOP], out_pool[N][HOP];
    EchoSim sim;

    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(p_heap, mic, ref, out_heap[h]);
    }
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(p_pool, mic, ref, out_pool[h]);
    }

    CHECK(memcmp(out_heap, out_pool, sizeof(out_heap)) == 0,
          fmt_msg("create (heap) == init (0xA5-poisoned pool), %d hops, byte-for-byte", N));

    int finite = 1;
    for (int h = 0; h < N && finite; h++)
        for (int i = 0; i < HOP; i++)
            if (out_heap[h][i] != out_heap[h][i]) { finite = 0; break; }
    CHECK(finite, fmt_msg("%d-hop synthetic run produces no NaN in the output", N));

    audio_pipeline_ulcnet_destroy(p_heap);
    audio_pipeline_ulcnet_destroy(p_pool);
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

    AudioPipelineUlcnetConfig cfg_null = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnetConfig cfg_id   = audio_pipeline_ulcnet_default_config(16000);
    cfg_id.model.user  = &st;
    cfg_id.model.infer = counting_infer;   /* copies err -> out, returns 0 */
    cfg_id.model.reset = counting_reset;

    AudioPipelineUlcnet* pn = audio_pipeline_ulcnet_create(&cfg_null);
    AudioPipelineUlcnet* pi = audio_pipeline_ulcnet_create(&cfg_id);
    if (!pn || !pi) {
        fprintf(stderr, "FAIL: setup (create) for NULL-vs-identity test\n");
        g_failures++;
        if (pn) audio_pipeline_ulcnet_destroy(pn);
        if (pi) audio_pipeline_ulcnet_destroy(pi);
        return;
    }

    float mic[HOP], ref[HOP], out_n[HOP], out_i[HOP];
    EchoSim sim;
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);

    int equal = 1, first_bad = -1, saw_locked = 0;
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(pn, mic, ref, out_n);
        audio_pipeline_ulcnet_process(pi, mic, ref, out_i);
        if (memcmp(out_n, out_i, sizeof(out_n)) != 0 && equal) { equal = 0; first_bad = h; }
        if (pipeline_delay_state(pi) != AEC_LINEAR_DELAY_UNLOCKED) saw_locked = 1;
    }

    CHECK(saw_locked, "NULL-vs-identity run reached the locked (model-applied) phase");
    CHECK(equal,
          fmt_msg("NULL model output == identity (err->out copy) model output, "
                  "bit-identical over %d hops (first mismatch %d [-1=none])", N, first_bad));
    CHECK(st.infer_calls == N,
          fmt_msg("identity model really was stepped every frame (%d calls)", st.infer_calls));

    audio_pipeline_ulcnet_destroy(pn);
    audio_pipeline_ulcnet_destroy(pi);
}

/* =========================================================================
 * 6. far-timestamp (ULCNET_FAR_RAW, the default). Far-passthrough model
 *    (copies far_ri -> out_ri, ignores err), silence on mic, one unit
 *    impulse in far at sample index T. The mono far tap is SAME-HOP with
 *    the error tap (this hop's raw ref feeds the far analysis beside this
 *    hop's formed error -- no wrapper-side far compensation exists or is
 *    needed), and the centered ULCNet chain output lags its input by
 *    exactly one hop (hop #p carries input hop p-1). So the reconstructed
 *    impulse must land at EXACTLY T + HOP. RAW mode applies the model from
 *    the first emitted frame (no delay lock ever happens on a silent mic),
 *    which is also what makes this test able to see the far branch at all.
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

static void test_far_timestamp_raw(void) {
    enum { N = 40, IMP_HOP = 8, IMP_OFF = 37 };
    const int imp_index = IMP_HOP * HOP + IMP_OFF;
    const int expect_index = imp_index + HOP;   /* same-hop far tap + 1-hop chain */

    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    cfg.model.infer = passthrough_far_infer;    /* far_input_mode: RAW default */
    AudioPipelineUlcnet* p = audio_pipeline_ulcnet_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for far-timestamp test\n"); g_failures++; return; }

    float mic[HOP], ref[HOP];
    static float out_hist[N][HOP];
    for (int h = 0; h < N; h++) {
        memset(mic, 0, sizeof(mic));
        memset(ref, 0, sizeof(ref));
        if (h == IMP_HOP) ref[IMP_OFF] = 1.0f;
        audio_pipeline_ulcnet_process(p, mic, ref, out_hist[h]);
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
          fmt_msg("far timestamp (mono RAW): impulse at far[%d] lands at out[%d] "
                  "(expected %d = impulse + 1 hop; offset %+d samples)",
                  imp_index, found_index, expect_index, found_index - expect_index));
    CHECK(peak > 0.9f,
          fmt_msg("far impulse reconstructed at ~unit amplitude (peak %.4f)", peak));

    audio_pipeline_ulcnet_destroy(p);
}

/* =========================================================================
 * 7. RAW mode never gates on the delay lock: the 0.5x model's output is
 *    APPLIED from the first emitted frame (hop #1), while the delay is
 *    still UNLOCKED -- the RAW-mode pipeline must differ from the
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

static void test_raw_mode_applies_unlocked(void) {
    enum { N = 30 };
    AudioPipelineUlcnetConfig cfg_raw = audio_pipeline_ulcnet_default_config(16000);
    cfg_raw.model.infer = halving_infer;        /* far_input_mode: RAW default */
    AudioPipelineUlcnetConfig cfg_null = audio_pipeline_ulcnet_default_config(16000);

    AudioPipelineUlcnet* pr = audio_pipeline_ulcnet_create(&cfg_raw);
    AudioPipelineUlcnet* pn = audio_pipeline_ulcnet_create(&cfg_null);
    if (!pr || !pn) {
        fprintf(stderr, "FAIL: setup (create) for RAW-mode gating test\n");
        g_failures++;
        if (pr) audio_pipeline_ulcnet_destroy(pr);
        if (pn) audio_pipeline_ulcnet_destroy(pn);
        return;
    }

    float mic[HOP], ref[HOP], out_r[HOP], out_n[HOP];
    EchoSim sim;
    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);

    int applied_from_hop1 = 1, first_equal_hop = -1, all_unlocked_seen = 1;
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(pr, mic, ref, out_r);
        audio_pipeline_ulcnet_process(pn, mic, ref, out_n);
        if (h == 0) {
            int zero = 1;
            for (int i = 0; i < HOP; i++) if (out_r[i] != 0.0f) zero = 0;
            CHECK(zero, "RAW mode hop #0 still emits zeros (timing preamble unchanged)");
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
          fmt_msg("RAW mode applies the model from the FIRST emitted frame with no "
                  "delay lock (output != NULL-model on every hop >= 1; first equal "
                  "hop %d [-1=none])", first_equal_hop));

    audio_pipeline_ulcnet_destroy(pr);
    audio_pipeline_ulcnet_destroy(pn);
}

/* =========================================================================
 * 8. NaN guard. Model returns rc==0 but poisons one bin with NaN (and
 *    another with +Inf) on scheduled hops; otherwise it halves the
 *    spectrum. RAW mode (applied from frame 1). Pipeline B has a NULL
 *    model. Same 50%-overlap mixing rule as test 3: output hop p is
 *    bit-identical to B exactly when BOTH contributing frames took the
 *    identity path (here: were NaN-poisoned and discarded), and differs
 *    when any clean (applied, 0.5x) frame contributed -- which also proves
 *    the next clean frame after a NaN window recovers. No NaN may ever
 *    appear in the output.
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

    AudioPipelineUlcnetConfig cfg_a = audio_pipeline_ulcnet_default_config(16000);
    cfg_a.model.user  = &st;
    cfg_a.model.infer = nan_infer;              /* far_input_mode: RAW default */
    AudioPipelineUlcnetConfig cfg_b = audio_pipeline_ulcnet_default_config(16000);

    AudioPipelineUlcnet* pa = audio_pipeline_ulcnet_create(&cfg_a);
    AudioPipelineUlcnet* pb = audio_pipeline_ulcnet_create(&cfg_b);
    if (!pa || !pb) {
        fprintf(stderr, "FAIL: setup (create) for NaN-guard test\n");
        g_failures++;
        if (pa) audio_pipeline_ulcnet_destroy(pa);
        if (pb) audio_pipeline_ulcnet_destroy(pb);
        return;
    }

    float mic[HOP], ref[HOP], out_a[HOP], out_b[HOP];
    EchoSim sim;
    echo_sim_init(&sim, 0, 0xC0FFEEu);   /* zero-delay: error nonzero from hop 0 */

    static int poisoned[N];
    int equal_ok = 1, differ_ok = 1, no_nan = 1;
    int first_bad_equal = -1, first_bad_differ = -1;
    int n_equal = 0, n_differ = 0;

    for (int h = 0; h < N; h++) {
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        st.poison_now = (h >= 20 && h < 28) || (h >= 40 && h < 48);
        poisoned[h] = st.poison_now;

        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(pa, mic, ref, out_a);
        audio_pipeline_ulcnet_process(pb, mic, ref, out_b);

        for (int i = 0; i < HOP; i++)
            if (!isfinite(out_a[i])) no_nan = 0;

        int equal_expected;
        if (h == 0)      equal_expected = 1;               /* both all-zero */
        else if (h == 1) equal_expected = poisoned[1];
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

    audio_pipeline_ulcnet_destroy(pa);
    audio_pipeline_ulcnet_destroy(pb);
}

int main(void) {
    printf("=== audio_pipeline_ulcnet: identity E2E (one-hop latency) ===\n");
    test_identity_e2e();

    printf("\n=== audio_pipeline_ulcnet: counting model (stepping + reset policy) ===\n");
    test_counting_model();

    printf("\n=== audio_pipeline_ulcnet: fail-open + delay gating ===\n");
    test_fail_open_and_delay_gating();

    printf("\n=== audio_pipeline_ulcnet: config validation ===\n");
    test_config_validation_rejects();

    printf("\n=== audio_pipeline_ulcnet: pool rejection ===\n");
    test_pool_rejection();

    printf("\n=== audio_pipeline_ulcnet: init_ex descriptor gate ===\n");
    test_init_ex_descriptor();

    printf("\n=== audio_pipeline_ulcnet: destroy idempotence ===\n");
    test_destroy_idempotence();

    printf("\n=== audio_pipeline_ulcnet: create-vs-init parity (poisoned pool) ===\n");
    test_create_vs_init_parity();

    printf("\n=== audio_pipeline_ulcnet: NULL model == identity model ===\n");
    test_null_model_equals_identity_model();

    printf("\n=== audio_pipeline_ulcnet: far timestamp (RAW mode) ===\n");
    test_far_timestamp_raw();

    printf("\n=== audio_pipeline_ulcnet: RAW mode applies while UNLOCKED ===\n");
    test_raw_mode_applies_unlocked();

    printf("\n=== audio_pipeline_ulcnet: NaN/Inf guard ===\n");
    test_nan_guard();

    if (g_failures) {
        fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
