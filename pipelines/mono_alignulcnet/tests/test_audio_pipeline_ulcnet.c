/**
 * tests/test_audio_pipeline_ulcnet.c — acceptance tests for
 * pipelines/mono_alignulcnet/audio_pipeline_ulcnet.h (mono AEC(linear) -> Align-ULCNet
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
 *      compute path) EXCEPT the identity-reprime frames after an alignment
 *      generation: cumulative calls after hop #p == p+1 minus the reprime
 *      frames consumed so far — 2 on hop #1 (the 0/2/1 emission), 1 per hop
 *      after; model->reset fires exactly once per CHANGED hop during the
 *      run, plus exactly once more on audio_pipeline_ulcnet_reset(); a
 *      post-reset hop #0 emits 0 frames.
 *   3. fail-open callback handling: a model whose output
 *      halves the spectrum (rc=0) but doubles it on scheduled failing hops
 *      (rc!=0) — output is bit-identical to the NULL-model pipeline's
 *      exactly where the policy says identity must hold (every failing
 *      frame) and differs on successful model frames, including while the
 *      AEC alignment state is UNLOCKED.
 *   4. pool rejection / reject-first config validation / init_ex 8-point
 *      descriptor gate / destroy idempotence / create-vs-init byte parity
 *      on a 0xA5-poisoned pool (patterns copied from test_audio_pipeline.c).
 *   5. NULL model == identity (copy err->out, rc=0) model: bit-identical
 *      output across UNLOCKED, CHANGED and LOCKED phases.
 *   6. far-timestamp before matched-delay acquisition: far-passthrough model,
 *      silence on mic, one unit impulse in far at a known sample index —
 *      the impulse must land in the output at EXACTLY impulse_index + 256:
 *      the mono far tap is SAME-HOP with the error tap (no wrapper-side far
 *      compensation exists or is needed), and the centered ULCNet chain
 *      lags the input by exactly one hop. Documents the mono timing.
 *   7. Production never gates on the delay lock: a 0.5x model
 *      diverges from the NULL-model pipeline from the FIRST emitted frame
 *      (hop #1), long before the delay acquires.
 *   8. NaN guard: a model returning rc==0 but a NaN-poisoned spectrum on
 *      scheduled hops — those frames take the identity path BITWISE (equal
 *      to the NULL-model pipeline under the same 50%-overlap mixing rule
 *      as test 3), the next clean frame is applied again, and no NaN ever
 *      reaches the output.
 *   9. full-write contract: a model returning rc==0 after writing only the
 *      FIRST 100 bins on scheduled hops — the pipeline's NaN pre-fill of
 *      the staging buffers leaves the unwritten bins non-finite, so those
 *      frames take the identity path BITWISE (same mixing rule as test 8)
 *      and the next fully-written frame recovers. MUTATION: removing the
 *      pre-fill in audio_pipeline_ulcnet.c leaks stale finite values into
 *      the unwritten bins, the partial frames get applied, and this test
 *      goes red.
 *  10. fixed deployment contract: a model publishing an aligned-far
 *      descriptor is accepted; a raw-far descriptor is rejected by
 *      get_mem_requirements/init/init_ex/create. A NULL descriptor remains
 *      supported for identity and board bring-up callbacks.
 *  11. identity-reprime straddle DERIVATION: a unit impulse in the middle of
 *      the last pre-boundary hop measures, branch by branch, how many
 *      emitted frames still have a pre-switch hop inside their analysis
 *      window. Measured in a boundary-FREE control run (so the reprime never
 *      measures itself) and asserted equal to
 *      AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES; the boundary run then proves
 *      the first model-visible frame after the reprime is marker-free on
 *      BOTH branches. MUTATION: no reprime, or one frame too few, goes red.
 *  12. identity-reprime BEHAVIOUR: exactly that many identity frames after a
 *      mid-stream MATCHED CHANGED hop and after the FIXED fill completion,
 *      infer frozen for exactly those frames and resuming right after, one
 *      model->reset per generation.
 *
 * Build (from pipelines/): `make test` (ne10 default; BACKEND=kiss for the
 * portable reference). Standalone (kiss reference variant shown):
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
#include <time.h>
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
 *    CHANGED -> LOCKED. infer is stepped for EVERY emitted frame EXCEPT the
 *    AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES straddling frames after each
 *    CHANGED hop (the identity reprime), so cumulative calls after hop #p ==
 *    p+1 minus the reprime frames consumed so far (2 on hop #1);
 *    model->reset fires once per CHANGED hop + once per pipeline reset.
 *
 *    Original intent kept: the per-hop compute is CONSTANT everywhere else
 *    -- one infer per emitted frame, never two, and never a hidden
 *    delay-gated skip. The reprime is the one documented exception and it
 *    only ever REMOVES work (those frames emit the identity without any
 *    inference); nothing anywhere doubles the per-hop model cost. The
 *    expectation below is built from the policy hop by hop, so both a
 *    missing and an over-long reprime fail here.
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
    int expected_calls = 0, armed = 0, skipped_total = 0;
    for (int h = 0; h < N; h++) {
        int frames, skipped_here;
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(p, mic, ref, out);
        AecLinearDelayState ds = pipeline_delay_state(p);
        if (ds == AEC_LINEAR_DELAY_CHANGED)
            armed = AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES;
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
          fmt_msg("counting model: 2 infer calls after hop #1 (0/2/1 emission; got %d)",
                  calls_after_hop1));
    CHECK(pattern_ok && st.infer_calls == N - skipped_total,
          fmt_msg("counting model: one infer per emitted frame at every hop except the "
                  "%d reprime frames per alignment generation "
                  "(%d calls after %d hops, %d skipped, first bad hop %d [-1=none])",
                  (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
                  st.infer_calls, N, skipped_total, first_bad_hop));
    CHECK(skipped_total == n_changed * AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES &&
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

/* FIXED has no estimator generation event: its seam is raw while the
 * alignment ring fills, then becomes aligned. The wrapper must still reset
 * external recurrent state exactly at that input-distribution boundary.
 *
 * The far branch is driven with real noise (not silence) so the seam's
 * content claim is checkable rather than vacuous: "UNLOCKED means the
 * content is the RAW far" is asserted byte-exactly on every fill hop
 * against the caller's own far history, and the shifted content is asserted
 * the same way from the first LOCKED hop on. The ref path applies no HPF and
 * no soft-clip, so equality is exact rather than approximate. */
static void test_fixed_first_alignment_resets_model(void) {
    enum { FIXED_DELAY = 2 * HOP, N = 8 };
    AudioPipelineUlcnetConfig cfg =
        audio_pipeline_ulcnet_default_config(16000);
    CountingModelState st = {0, 0};
    AudioPipelineUlcnet* p;
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
    cfg.model.reset = counting_reset;
    p = audio_pipeline_ulcnet_create(&cfg);
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
        CHECK(audio_pipeline_ulcnet_process(p, mic, ref, out) == 0,
              "FIXED transition frame processes");
        state = pipeline_delay_state(p);
        /* The far tap aliases AEC-internal memory valid only until the next
         * aec call, so it is read here, right after process. */
        aec_get_linear_context(audio_pipeline_ulcnet_get_aec(p), &lctx);
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
          fmt_msg("wrapper resets model exactly once when FIXED switches from "
                  "raw to aligned far (got %d)", st.reset_calls));
    /* The transition suppresses inference for the identity reprime and for
     * nothing else: N hops emit N frames, of which exactly
     * AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES straddle the switch. */
    CHECK(st.infer_calls == N - AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
          fmt_msg("FIXED transition suppresses inference for exactly the %d "
                  "straddling frames and no others (%d calls over %d emitted "
                  "frames)", (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
                  st.infer_calls, N));
    audio_pipeline_ulcnet_destroy(p);
}

/* =========================================================================
 * 2b. relock on the SAME delay still resets the model (ALIGNED).
 *
 * The mono twin of test_relock_same_delay_resets_model() in
 * 4ch_alignulcnet/tests/. Same scenario, same contract:
 *
 *   LOCKED(V) -> forced UNLOCKED for several hops -> LOCKED(V), same V
 *
 * must issue exactly one model->reset, on the relock hop, because ALIGNED
 * mode keeps STEPPING infer() while unlocked (constant per-hop compute) and
 * therefore carries recurrent state built over a reference the estimator had
 * not vouched for.
 *
 * The two families reach this from opposite directions and the test records
 * which. lib/aec spells "nothing accepted yet" as current_delay == -1, so its
 * first acquisition after a reset always crosses that sentinel and bumps
 * delay_generation -> AEC_LINEAR_DELAY_CHANGED, whatever value it relocks on;
 * mono was already correct. The 4ch core keeps a plain non-negative
 * accepted_delay with no sentinel, so its same-value relock had to be fixed
 * by tracking the not-usable -> usable transition explicitly. This test
 * pins the mono half of that shared contract so the two cannot drift apart.
 *
 * MUTATION: deleting the `delay_state == AEC_LINEAR_DELAY_CHANGED` model
 * reset in audio_pipeline_ulcnet.c turns both the acquisition and the relock
 * reset off and this test goes red.
 * ========================================================================= */
static int pipeline_delay_samples(const AudioPipelineUlcnet* p) {
    AecLinearContext lctx;
    aec_get_linear_context(audio_pipeline_ulcnet_get_aec(p), &lctx);
    return lctx.delay_samples;
}

static void test_relock_same_delay_resets_model(void) {
    enum { WARM = 90, RELOCK = 90, BULK_DELAY = 64 };
    CountingModelState st = {0, 0};

    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    cfg.model.user  = &st;
    cfg.model.infer = counting_infer;
    cfg.model.reset = counting_reset;

    AudioPipelineUlcnet* p = audio_pipeline_ulcnet_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for mono relock test\n"); g_failures++; return; }

    float mic[HOP], ref[HOP], out[HOP];
    EchoSim sim;
    int acquire_hop = -1, relock_hop = -1;
    int acquire_delay = -1, relock_delay = -2;
    int resets_at_acquire = -1, resets_at_relock = -1;
    int resets_before_reset = -1, resets_at_pipeline_reset = -1;
    int unlocked_hops_after_reset = 0, resets_during_unlock = -1;
    int applied_after_relock = 0;

    echo_sim_init(&sim, BULK_DELAY, 0xC0FFEEu);
    for (int h = 0; h < WARM + RELOCK; h++) {
        if (h == WARM) {
            /* Forced unlock: the pipeline reset runs aec_reset(), which is
             * lib/aec's only in-processing path back to current_delay == -1
             * (the estimator's REFINED-confidence latch never drops on its
             * own). The stimulus continues unchanged, so the AEC re-acquires
             * the very same bulk delay. */
            resets_before_reset = st.reset_calls;
            audio_pipeline_ulcnet_reset(p);
            resets_at_pipeline_reset = st.reset_calls;
            echo_sim_init(&sim, BULK_DELAY, 0xC0FFEEu);
        }
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(p, mic, ref, out);

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

    audio_pipeline_ulcnet_destroy(p);
}

/* =========================================================================
 * 3. fail-open model callback. Model A halves
 *    the spectrum on success (rc=0) and doubles it on scheduled failing
 *    hops (rc!=0); pipeline B has a NULL model. Output hop p mixes the
 *    frames pushed at hops p-1 and p (50% WOLA overlap), so A == B bitwise
 *    exactly when BOTH contributing frames took the identity path (rc!=0),
 *    and A != B when any successful
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
    AudioPipelineUlcnetConfig cfg_b = audio_pipeline_ulcnet_default_config(16000); /* NULL model */

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
    int reprime_left = 0, n_reprime_hops = 0;

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
        /* The identity reprime after an alignment generation puts its own
         * frames on the identity path too. Fold it into the same
         * bookkeeping so the 50%-overlap rule below stays exact. The
         * acquisition always lands well inside the steady 1-frame-per-hop
         * region (lock_hop >= 2, asserted below), so one reprime frame is
         * consumed per hop here. */
        if (ds_a == AEC_LINEAR_DELAY_CHANGED)
            reprime_left = AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES;
        {
            int reprimed = reprime_take(&reprime_left, 1);
            ident[h] = st.fail_now || reprimed > 0;
            n_reprime_hops += reprimed;
        }

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

    CHECK(lock_hop >= 2 && lock_hop <= MAX_LOCK_HOPS,
          fmt_msg("delay acquired at hop %d (2..%d) so both gated and applied phases ran "
                  "and the reprime lands in the steady 1-frame-per-hop region",
                  lock_hop, MAX_LOCK_HOPS));
    CHECK(n_unlocked >= 2 && n_failing == FAIL_LEN &&
          n_reprime_hops == AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
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
    CHECK(good.sample_rate == 16000 && good.fft_size == 512,
          "default config is the trained 16 kHz / frame-FFT 512 / hop 256 grid");
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

    AudioPipelineUlcnetConfig matched2 =
        audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnetConfig external = matched2;
    AudioPipelineUlcnetMemReq matched_req, external_req;
    matched2.delay_num_filters = 2;
    external.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    external.delay_num_filters = 5;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&matched2, &matched_req) == 0 &&
          audio_pipeline_ulcnet_get_mem_requirements(&external, &external_req) == 0 &&
          matched_req.bytes > external_req.bytes,
          "ULCNet wrapper passes AEC delay configuration into pool sizing");

    AudioPipelineUlcnetMemReq req0, req512;
    AudioPipelineUlcnetConfig c0 = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnetConfig c512 = audio_pipeline_ulcnet_default_config(16000);
    c0.fft_size = 0;  /* compatibility spelling for the same fixed grid */
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
 * 10. far-input contract gate: a model that publishes a model-I/O
 * descriptor may only be wired to a pipeline whose far branch matches the
 * checkpoint the descriptor describes. Both directions are checked, on
 * every entry point that funnels through the shared config gate.
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

static void test_aligned_descriptor_gate(void) {
    UlcnetModelIoDescriptor raw_desc, aligned_desc;
    AudioPipelineUlcnetMemReq req;
    void* pool = NULL;

    if (ulcnet_model_io_descriptor_default(8, &raw_desc) != 0 ||
        ulcnet_model_io_descriptor_default(8, &aligned_desc) != 0) {
        fprintf(stderr, "FAIL: setup (descriptor_default) for far-mode gate test\n");
        g_failures++;
        return;
    }
    raw_desc.far_input_mode = ULCNET_FAR_RAW;
    CHECK(aligned_desc.far_input_mode == ULCNET_FAR_ALIGNED,
          "descriptor_default publishes the fixed aligned-far contract");

    /* An undescribed model remains valid for the identity/test boundary. */
    AudioPipelineUlcnetConfig undescribed = audio_pipeline_ulcnet_default_config(16000);
    undescribed.model.user = NULL;
    undescribed.model.infer = gate_infer_identity;
    undescribed.model.reset = NULL;
    undescribed.model.io_descriptor = NULL;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&undescribed, &req) == 0,
          "a model with io_descriptor == NULL publishes no contract and is not gated");

    AudioPipelineUlcnetConfig aligned_aligned = undescribed;
    aligned_aligned.model.io_descriptor = &aligned_desc;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&aligned_aligned, &req) == 0,
          "aligned descriptor is accepted");

    AudioPipelineUlcnetConfig raw_model = undescribed;
    raw_model.model.io_descriptor = &raw_desc;
    CHECK(audio_pipeline_ulcnet_get_mem_requirements(&raw_model, &req) == -1,
          "raw-far production descriptor is rejected");

    /* init/init_ex/create share the gate, so the mismatch is fail-fast on
     * every construction path, not only on the sizing query. */
    if (audio_pipeline_ulcnet_get_mem_requirements(&aligned_aligned, &req) != 0 ||
        posix_memalign(&pool, 16, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for far-mode gate test\n");
        g_failures++;
        return;
    }
    CHECK(audio_pipeline_ulcnet_init(pool, (size_t)req.bytes, &raw_model) == NULL,
          "audio_pipeline_ulcnet_init rejects raw descriptor too");
    CHECK(audio_pipeline_ulcnet_init_ex(pool, (size_t)req.bytes, &raw_model, NULL) == NULL,
          "audio_pipeline_ulcnet_init_ex rejects raw descriptor too");
    CHECK(audio_pipeline_ulcnet_create(&raw_model) == NULL,
          "audio_pipeline_ulcnet_create rejects raw descriptor too");

    AudioPipelineUlcnet* p_ok =
        audio_pipeline_ulcnet_init(pool, (size_t)req.bytes, &aligned_aligned);
    CHECK(p_ok != NULL, "aligned descriptor initializes in the same pool");
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

    int equal = 1, first_bad = -1, saw_locked = 0, n_changed = 0;
    for (int h = 0; h < N; h++) {
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(pn, mic, ref, out_n);
        audio_pipeline_ulcnet_process(pi, mic, ref, out_i);
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
          N - n_changed * AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
          fmt_msg("identity model was stepped on every frame except the %d reprime "
                  "frames of its %d alignment generations (%d calls)",
                  (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES, n_changed,
                  st.infer_calls));

    audio_pipeline_ulcnet_destroy(pn);
    audio_pipeline_ulcnet_destroy(pi);
}

/* =========================================================================
 * 6. far-timestamp before matched-delay acquisition. Far-passthrough model
 *    (copies far_ri -> out_ri, ignores err), silence on mic, one unit
 *    impulse in far at sample index T. The mono far tap is SAME-HOP with
 *    the error tap (this hop's raw ref feeds the far analysis beside this
 *    hop's formed error -- no wrapper-side far compensation exists or is
 *    needed), and the centered ULCNet chain output lags its input by
 *    exactly one hop (hop #p carries input hop p-1). So the reconstructed
 *    impulse must land at EXACTLY T + HOP. The model runs from the first
 *    emitted frame (no delay lock ever happens on a silent mic),
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

static void test_far_timestamp_before_acquisition(void) {
    enum { N = 40, IMP_HOP = 8, IMP_OFF = 37 };
    const int imp_index = IMP_HOP * HOP + IMP_OFF;
    const int expect_index = imp_index + HOP;   /* same-hop far tap + 1-hop chain */

    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    cfg.model.infer = passthrough_far_infer;
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
          fmt_msg("far timestamp before acquisition: impulse at far[%d] lands at out[%d] "
                  "(expected %d = impulse + 1 hop; offset %+d samples)",
                  imp_index, found_index, expect_index, found_index - expect_index));
    CHECK(peak > 0.9f,
          fmt_msg("far impulse reconstructed at ~unit amplitude (peak %.4f)", peak));

    audio_pipeline_ulcnet_destroy(p);
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
    AudioPipelineUlcnetConfig cfg_raw = audio_pipeline_ulcnet_default_config(16000);
    cfg_raw.model.infer = halving_infer;
    AudioPipelineUlcnetConfig cfg_null = audio_pipeline_ulcnet_default_config(16000);

    AudioPipelineUlcnet* pr = audio_pipeline_ulcnet_create(&cfg_raw);
    AudioPipelineUlcnet* pn = audio_pipeline_ulcnet_create(&cfg_null);
    if (!pr || !pn) {
        fprintf(stderr, "FAIL: setup (create) for unlocked-model test\n");
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

    audio_pipeline_ulcnet_destroy(pr);
    audio_pipeline_ulcnet_destroy(pn);
}

/* =========================================================================
 * 8. NaN guard. Model returns rc==0 but poisons one bin with NaN (and
 *    another with +Inf) on scheduled hops; otherwise it halves the
 *    spectrum. The model is applied from frame 1. Pipeline B has a NULL
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
    cfg_a.model.infer = nan_infer;
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
    int n_equal = 0, n_differ = 0, n_changed = 0;

    for (int h = 0; h < N; h++) {
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        st.poison_now = (h >= 20 && h < 28) || (h >= 40 && h < 48);
        poisoned[h] = st.poison_now;

        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(pa, mic, ref, out_a);
        audio_pipeline_ulcnet_process(pb, mic, ref, out_b);
        if (pipeline_delay_state(pa) == AEC_LINEAR_DELAY_CHANGED) n_changed++;

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

    audio_pipeline_ulcnet_destroy(pa);
    audio_pipeline_ulcnet_destroy(pb);
}

/* =========================================================================
 * 9. full-write contract. Model writes only the FIRST 100 bins (0.5x) on
 *    scheduled hops but still returns rc==0; otherwise it writes all bins
 *    (0.5x). The pipeline pre-fills the mdl_re/mdl_im staging with NaN
 *    before every infer call (ulcnet_process.h's FULL-WRITE CONTRACT), so
 *    the 157 unwritten bins stay NaN and the finite guard discards the
 *    frame: output is bit-identical to the NULL-model pipeline exactly
 *    when BOTH contributing frames were partial (same 50%-overlap mixing
 *    rule as test 8), and the next fully-written frame is applied again.
 *    MUTATION PROOF: removing the pre-fill loop in audio_pipeline_ulcnet.c
 *    leaves the previous frame's stale FINITE values in bins 100..256 --
 *    the guard cannot catch them, the partial frames get applied, and the
 *    bit-identical check below goes red.
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

    AudioPipelineUlcnetConfig cfg_a = audio_pipeline_ulcnet_default_config(16000);
    cfg_a.model.user  = &st;
    cfg_a.model.infer = partial_write_infer;
    AudioPipelineUlcnetConfig cfg_b = audio_pipeline_ulcnet_default_config(16000);

    AudioPipelineUlcnet* pa = audio_pipeline_ulcnet_create(&cfg_a);
    AudioPipelineUlcnet* pb = audio_pipeline_ulcnet_create(&cfg_b);
    if (!pa || !pb) {
        fprintf(stderr, "FAIL: setup (create) for partial-write test\n");
        g_failures++;
        if (pa) audio_pipeline_ulcnet_destroy(pa);
        if (pb) audio_pipeline_ulcnet_destroy(pb);
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
        audio_pipeline_ulcnet_process(pa, mic, ref, out_a);
        audio_pipeline_ulcnet_process(pb, mic, ref, out_b);
        if (pipeline_delay_state(pa) == AEC_LINEAR_DELAY_CHANGED) n_changed++;

        for (int i = 0; i < HOP; i++)
            if (!isfinite(out_a[i])) no_nan = 0;

        int equal_expected;
        if (h == 0)      equal_expected = 1;               /* both all-zero */
        else if (h == 1) equal_expected = partial[1];
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

    audio_pipeline_ulcnet_destroy(pa);
    audio_pipeline_ulcnet_destroy(pb);
}

/* =========================================================================
 * 11. Identity-reprime straddle DERIVATION (marker/timestamp test).
 *
 * Measures the quantity the reprime constant is supposed to be: how many
 * frames emitted at or after an alignment boundary at hop T still have a
 * PRE-boundary hop inside their 512-sample analysis window, on the error
 * branch and on the far branch separately.
 *
 * Instrument: total silence except ONE unit impulse at the MIDDLE of input
 * hop T-1 -- the last hop pushed before the boundary. The middle is the
 * position the sqrt-Hann window weights at 0.707 in BOTH frames that cover
 * that hop (a marker at a hop edge would be multiplied by the window's zero
 * and hide itself). A frame's window therefore contains a pre-boundary hop
 * exactly when the model sees a spectrum whose peak is ~0.707 instead of the
 * silence floor. The two branches are marked in SEPARATE runs: the error
 * branch through the mic (far kept silent so the linear filter has no far
 * history to leak, leaving only the mic HPF's own short ring), the far
 * branch through the reference (which the seam carries byte-exactly).
 *
 * The count is derived from a CONTROL run that has NO boundary at all
 * (FIXED delay 0: the ring can serve from hop #0, so the wrapper never sees
 * a raw->aligned transition and never arms a reprime -- asserted, not
 * assumed). The reprime logic therefore takes no part in measuring its own
 * length, and the measurement is pure framing/latency: which emitted frames
 * still reach back across hop T. A boundary would only replace the content
 * of those straddling slots (mono clears nothing at all), never move them.
 *
 * The boundary run (FIXED delay = T hops, so the raw->aligned switch lands
 * exactly on hop T) then checks the implementation against that number:
 * exactly REPRIME frames are skipped, they are the frames at hops
 * T..T+REPRIME-1, and the first frame the model DOES see afterwards carries
 * no pre-switch sample on either branch.
 *
 * MUTATION: dropping the reprime counter (stepping always) makes the first
 * model-visible frame after the boundary the straddling one and the
 * marker-free assertion goes red; arming it one frame too short does the
 * same.
 * ========================================================================= */
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
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnet* p;
    float mic[HOP], ref[HOP], out[HOP];

    memset(st, 0, sizeof(*st));
    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = fixed_delay;
    cfg.model.user  = st;
    cfg.model.infer = probe_infer;
    cfg.model.reset = probe_reset;
    p = audio_pipeline_ulcnet_create(&cfg);
    if (!p) return -1;

    for (int h = 0; h < RP_RUN; h++) {
        memset(mic, 0, sizeof(mic));
        memset(ref, 0, sizeof(ref));
        if (h == RP_MARK_HOP) {
            if (mark_far) ref[RP_MARK_POS] = 1.0f;
            else          mic[RP_MARK_POS] = 1.0f;
        }
        st->hop = h;
        audio_pipeline_ulcnet_process(p, mic, ref, out);
    }
    audio_pipeline_ulcnet_destroy(p);
    return 0;
}

/* Frames emitted at hops [RP_T, RP_T+RP_WINDOW) that still carry the marker,
 * plus the classification margin (smallest marked peak / largest unmarked
 * peak inside that window). */
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
    CHECK(straddle_err == AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
          fmt_msg("mono ERROR branch: %d emitted frames from hop %d on still contain "
                  "pre-switch samples == AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES (%d)",
                  straddle_err, RP_T, (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES));
    CHECK(straddle_err > 0 && dmin > 20.0f * cmax && dmin > 0.5f,
          fmt_msg("mono ERROR branch marker is unambiguous (marked %.4f vs unmarked "
                  "%.3e, ratio %.1fx)", dmin, cmax,
                  cmax > 0.0f ? (double)(dmin / cmax) : 1e30));

    straddle_far = reprime_straddle_count(&far_ctl, 1, &dmin, &cmax);
    printf("reprime derivation (mono, FAR branch):   %d straddling frames at hops "
           "%d..%d; marked peak >= %.4f, unmarked peak <= %.3e\n",
           straddle_far, RP_T, RP_T + RP_WINDOW - 1, dmin, cmax);
    CHECK(straddle_far == AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
          fmt_msg("mono FAR branch: %d emitted frames from hop %d on still contain "
                  "pre-switch samples == AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES (%d)",
                  straddle_far, RP_T, (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES));
    CHECK(straddle_far > 0 && dmin > 20.0f * cmax && dmin > 0.5f,
          fmt_msg("mono FAR branch marker is unambiguous (marked %.4f vs unmarked "
                  "%.3e, ratio %.1fx)", dmin, cmax,
                  cmax > 0.0f ? (double)(dmin / cmax) : 1e30));

    /* ---- the boundary runs: the wrapper skips exactly those frames ---- */
    CHECK(err_bnd.resets == 1 && far_bnd.resets == 1,
          fmt_msg("boundary run: FIXED raw->aligned switch fired model->reset once "
                  "(err %d, far %d)", err_bnd.resets, far_bnd.resets));
    for (int c = 0; c < err_bnd.calls; c++) {
        if (err_bnd.call_hop[c] >= RP_T &&
            err_bnd.call_hop[c] < RP_T + AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES)
            stepped_inside_reprime++;
        if (err_bnd.call_hop[c] >= RP_T && first_visible_err < 0) {
            first_visible_err = err_bnd.call_hop[c];
            first_visible_err_peak = err_bnd.err_peak[c];
        }
    }
    for (int c = 0; c < far_bnd.calls; c++) {
        if (far_bnd.call_hop[c] >= RP_T &&
            far_bnd.call_hop[c] < RP_T + AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES)
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
    /* The primary claim first, then the bookkeeping that follows from it. */
    CHECK(first_visible_err_peak >= 0.0f && first_visible_err_peak <= RP_FLOOR &&
          first_visible_far_peak >= 0.0f && first_visible_far_peak <= RP_FLOOR,
          fmt_msg("the first model-visible frame after the reprime contains NO "
                  "pre-switch sample on either branch (err peak %.3e, far peak "
                  "%.3e <= %.3f)", first_visible_err_peak, first_visible_far_peak,
                  RP_FLOOR));
    CHECK(stepped_inside_reprime == 0,
          fmt_msg("boundary run: the model is not stepped on any hop in [%d, %d) "
                  "(%d stray calls)", RP_T,
                  RP_T + (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
                  stepped_inside_reprime));
    CHECK(first_visible_err == RP_T + AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES &&
          first_visible_far == RP_T + AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
          fmt_msg("boundary run: stepping resumes on hop %d (err) / %d (far), "
                  "expected %d", first_visible_err, first_visible_far,
                  RP_T + (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES));
    CHECK(err_bnd.calls == expected_frames - AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES &&
          far_bnd.calls == expected_frames - AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
          fmt_msg("boundary run: exactly %d of %d emitted frames skipped inference "
                  "(err %d calls, far %d calls)",
                  (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES, expected_frames,
                  err_bnd.calls, far_bnd.calls));
}

/* =========================================================================
 * 12. Identity-reprime BEHAVIOUR: exactly REPRIME identity frames per
 *     alignment generation, on both event shapes the wrapper recognises --
 *     a mid-stream MATCHED CHANGED hop and the FIXED ring-fill completion.
 *     Per hop the test predicts the infer delta from its OWN copy of the
 *     policy (arm REPRIME at the boundary hop, consume one per emitted
 *     frame) and compares against the counting model, so both a missing and
 *     an over-long reprime fail. reset stays exactly one per generation.
 * ========================================================================= */
static void test_reprime_behavior(void) {
    enum { N = 60, FIXED_HOPS = 8 };
    CountingModelState st = {0, 0};
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnet* p;
    float mic[HOP], ref[HOP], out[HOP];
    EchoSim sim;
    int armed = 0, n_changed = 0, skipped_total = 0;
    int pattern_ok = 1, first_bad_hop = -1;
    int boundary_hop = -1, resumed_hop = -1;
    int calls_before, delta, frames, skipped_here, expected_delta;

    /* ---- (a) mid-stream MATCHED CHANGED ---- */
    cfg.model.user  = &st;
    cfg.model.infer = counting_infer;
    cfg.model.reset = counting_reset;
    p = audio_pipeline_ulcnet_create(&cfg);
    if (!p) { fprintf(stderr, "FAIL: setup (create) for reprime behaviour test\n"); g_failures++; return; }

    echo_sim_init(&sim, ECHO_DELAY, 0xC0FFEEu);
    for (int h = 0; h < N; h++) {
        calls_before = st.infer_calls;
        echo_sim_hop(&sim, mic, ref);
        audio_pipeline_ulcnet_process(p, mic, ref, out);
        if (pipeline_delay_state(p) == AEC_LINEAR_DELAY_CHANGED) {
            armed = AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES;
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
                  (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES, first_bad_hop));
    CHECK(skipped_total == n_changed * AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES &&
          st.infer_calls == N - skipped_total,
          fmt_msg("reprime behaviour (MATCHED): %d frames skipped inference over %d "
                  "generations (%d infer calls in %d hops)",
                  skipped_total, n_changed, st.infer_calls, N));
    CHECK(resumed_hop == boundary_hop + AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
          fmt_msg("reprime behaviour (MATCHED): stepping stops on the boundary hop %d "
                  "and resumes on hop %d (expected %d)", boundary_hop, resumed_hop,
                  boundary_hop + (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES));
    CHECK(st.reset_calls == n_changed,
          fmt_msg("reprime behaviour (MATCHED): still exactly one model->reset per "
                  "generation (%d resets, %d generations)", st.reset_calls, n_changed));
    audio_pipeline_ulcnet_destroy(p);

    /* ---- (b) FIXED ring-fill completion ---- */
    {
        CountingModelState fst = {0, 0};
        AudioPipelineUlcnetConfig fcfg = audio_pipeline_ulcnet_default_config(16000);
        int fixed_calls_at_boundary = -1, fixed_calls_after = -1;
        int fixed_boundary = -1;

        fcfg.delay_mode = AEC_DELAY_FIXED;
        fcfg.fixed_delay_samples = FIXED_HOPS * HOP;
        fcfg.model.user  = &fst;
        fcfg.model.infer = counting_infer;
        fcfg.model.reset = counting_reset;
        p = audio_pipeline_ulcnet_create(&fcfg);
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
            audio_pipeline_ulcnet_process(p, mic, ref, out);
            if (h != 0 && prev == AEC_LINEAR_DELAY_UNLOCKED &&
                pipeline_delay_state(p) == AEC_LINEAR_DELAY_LOCKED) {
                armed = AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES;
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
                h == fixed_boundary + AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES)
                fixed_calls_after = delta;
        }

        CHECK(fixed_boundary == FIXED_HOPS,
              fmt_msg("reprime behaviour (FIXED): the raw->aligned switch is on hop %d "
                      "(expected %d)", fixed_boundary, FIXED_HOPS));
        CHECK(pattern_ok && skipped_total == AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
              fmt_msg("reprime behaviour (FIXED): exactly %d frames skipped inference at "
                      "the fill completion (%d skipped, first mismatching hop %d [-1=none])",
                      (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES, skipped_total,
                      first_bad_hop));
        CHECK(fixed_calls_at_boundary == 0 && fixed_calls_after == 1,
              fmt_msg("reprime behaviour (FIXED): the boundary hop steps the model 0 times "
                      "and hop boundary+%d steps it once (%d then %d)",
                      (int)AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
                      fixed_calls_at_boundary, fixed_calls_after));
        CHECK(fst.reset_calls == 1,
              fmt_msg("reprime behaviour (FIXED): exactly one model->reset (%d)",
                      fst.reset_calls));
        CHECK(fst.infer_calls == N - AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES,
              fmt_msg("reprime behaviour (FIXED): %d infer calls over %d hops "
                      "(%d emitted frames minus the reprime)",
                      fst.infer_calls, N, N));
        audio_pipeline_ulcnet_destroy(p);
    }
}

/* =========================================================================
 * Known-delay profile verification (product delay gate, NOT an audio-quality
 * run -- see docs/align_ulcnet_delay_profile_plan_zh_TW.md §5.1/§6).
 *
 * The mono twin of test_4aec_nr_res.c's known-delay block: same synthetic
 * two-path echo with an exactly known bulk delay, same four questions
 * (acquisition, coverage boundary, mislock detectability, cost), checked
 * against ground truth rather than against a score. The one structural
 * difference is where n lives -- mono has ONE AEC instance and the whole
 * "5,728 B per matched filter" contract lands in its pool, whereas 4ch pays
 * it once in a shared estimator that the four lanes do not multiply.
 * ========================================================================= */

/* Same geometry as the 4ch twin, spelled with lib/aec's own constants: a
 * bank of n filters reaches (n-1)*DA_FILTER_INTRA_SHIFT +
 * (DA_FILTER_SIZE - 11) downsampled samples (the -11 is the
 * `lag < filter_size - 10` reliability cut), decimation
 * DA_DOWN_SAMPLING_FACTOR. */
#define KD_RELIABLE_SAMPLES(n) \
    (((n) - 1) * DA_FILTER_INTRA_SHIFT * DA_DOWN_SAMPLING_FACTOR + \
     (DA_FILTER_SIZE - 11) * DA_DOWN_SAMPLING_FACTOR)
/* Applied alignment may be EARLY of the true echo (PBFDKF then models a
 * positive residual) but never LATE; measured shortfall is 64-80 samples. */
#define KD_MAX_UNDERSHOOT 128

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
    enum { KD_PAD = 16384 };
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
    AudioPipelineUlcnet* p;
    float* far_hist;
    float mic[HOP], ref[HOP], outbuf[HOP];
    clock_t t0, t1;
    int hop, i;

    memset(out, 0, sizeof(*out));
    out->lock_hop = -1;
    out->applied_delay = -1;

    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = num_filters;
    p = audio_pipeline_ulcnet_create(&cfg);
    if (!p) return;

    far_hist = (float*)malloc((size_t)(hops * HOP + KD_PAD) * sizeof(float));
    if (!far_hist) { audio_pipeline_ulcnet_destroy(p); return; }
    lcg_state = 0xC0FFEEu;
    for (i = 0; i < hops * HOP + KD_PAD; i++) far_hist[i] = lcg_sample();

    t0 = clock();
    for (hop = 0; hop < hops; hop++) {
        int base = hop * HOP + KD_PAD;
        for (i = 0; i < HOP; i++) {
            int t = base + i;
            ref[i] = far_hist[t];
            mic[i] = dominant_gain * far_hist[t - dominant_delay] +
                     early_gain * far_hist[t - early_delay];
        }
        audio_pipeline_ulcnet_process(p, mic, ref, outbuf);
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
    audio_pipeline_ulcnet_destroy(p);
}

static void test_known_delay_profile(void) {
    static const int inside_ms[6] = { 0, 125, 221, 317, 413, 509 };
    MonoKnownDelayRun mislock, control, cost;
    AudioPipelineUlcnetMemReq req[6];
    int n;
    int mem_ok = 1;
    int mislock_error;

    printf("known-delay acquisition (mono, 16 kHz, hop 256, synthetic echo):\n");
    for (n = 1; n <= 5; n++) {
        MonoKnownDelayRun in_range, out_of_range;
        int inside = inside_ms[n] * 16;
        int outside = (n == 5) ? 9271 : inside_ms[n + 1] * 16;

        mono_known_delay_run(n, 200, inside, 0.6f, inside, 0.0f, &in_range);
        mono_known_delay_run(n, 200, outside, 0.6f, outside, 0.0f, &out_of_range);

        printf("  n=%d ceiling %d samples (%.2f ms): in-range %d ms -> lock hop %d "
               "applied %d (short by %d); out-of-range %d ms -> %s\n",
               n, KD_RELIABLE_SAMPLES(n), KD_RELIABLE_SAMPLES(n) / 16.0,
               inside_ms[n], in_range.lock_hop, in_range.applied_delay,
               in_range.locked ? inside - in_range.applied_delay : -1,
               outside / 16, out_of_range.locked ? "LOCKED" : "no lock");

        CHECK(in_range.locked && in_range.lock_hop >= 0 && in_range.lock_hop < 60,
              fmt_msg("mono n=%d acquires a %d ms bulk delay within 60 hops "
                      "(lock hop %d, inside its %.2f ms ceiling)",
                      n, inside_ms[n], in_range.lock_hop,
                      KD_RELIABLE_SAMPLES(n) / 16.0));
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
                      "%.2f ms ceiling)", n, outside / 16,
                      KD_RELIABLE_SAMPLES(n) / 16.0));
    }

    /* Mislock detectability. Identical construction to the 4ch twin: an
     * out-of-range dominant path plus an in-range early reflection makes the
     * estimator lock, with a LOCKED seam state, onto the wrong path. Nothing
     * in AecLinearContext distinguishes this from a correct lock, so only a
     * comparison against an independently known delay catches it -- that
     * comparison is what is asserted. This does not bless the behaviour. */
    mono_known_delay_run(5, 200, 9271, 0.6f, 512, 0.5f, &mislock);
    mono_known_delay_run(5, 200, 3536, 0.6f, 512, 0.5f, &control);
    mislock_error = mislock.locked ? 9271 - mislock.applied_delay : -1;
    printf("known-delay mislock (mono): dominant 9271 (579.44 ms) + early 512 "
           "(32.00 ms) -> lock hop %d applied %d (%.2f ms), wrong by %d samples "
           "(%.2f ms)\n",
           mislock.lock_hop, mislock.applied_delay, mislock.applied_delay / 16.0,
           mislock_error, mislock_error / 16.0);
    CHECK(mislock.locked,
          "mono: an in-range early path makes an out-of-range bulk delay lock anyway");
    CHECK(mislock.locked && mislock_error > KD_MAX_UNDERSHOOT,
          fmt_msg("mono: ground-truth comparison FLAGS the mislock (applied short "
                  "by %d samples = %.2f ms, far past the %d-sample alignment "
                  "contract); the LOCKED seam state alone cannot",
                  mislock_error, mislock_error / 16.0, KD_MAX_UNDERSHOOT));
    CHECK(control.locked && 3536 - control.applied_delay >= 0 &&
          3536 - control.applied_delay <= KD_MAX_UNDERSHOOT,
          fmt_msg("mono control: with the dominant path in range the SAME check "
                  "stays quiet (applied %d, short by %d)",
                  control.applied_delay,
                  control.locked ? 3536 - control.applied_delay : -1));

    /* RAM. Mono has ONE AEC, so the whole per-filter cost lands in its pool
     * -- this is where the 5,728 B/filter contract is spent per instance. */
    for (n = 1; n <= 5; n++) {
        AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
        cfg.delay_num_filters = n;
        if (audio_pipeline_ulcnet_get_mem_requirements(&cfg, &req[n]) != 0)
            mem_ok = 0;
    }
    CHECK(mem_ok, "mono pool query answers for every n=1..5");
    if (mem_ok) {
        printf("mono pool vs matched-filter bank size (16 kHz, fft 512):\n");
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

    /* Rough per-hop CPU, recorded rather than tuned. One hop is 256 samples =
     * 16 ms of audio at 16 kHz. Host measurement, liveness guard only. */
    mono_known_delay_run(5, 400, 3536, 0.6f, 3536, 0.0f, &cost);
    printf("known-delay cost (mono, n=5): %.1f us/hop, %.4f x real time "
           "(hop = 16.00 ms of audio)\n",
           cost.us_per_hop, cost.us_per_hop / 16000.0);
    CHECK(cost.us_per_hop > 0.0 && cost.us_per_hop < 16000.0,
          fmt_msg("mono pipeline runs faster than real time on the host "
                  "(%.1f us/hop vs a 16000 us budget)", cost.us_per_hop));
}

int main(void) {
    printf("=== audio_pipeline_ulcnet: identity E2E (one-hop latency) ===\n");
    test_identity_e2e();

    printf("\n=== audio_pipeline_ulcnet: counting model (stepping + reset policy) ===\n");
    test_counting_model();

    printf("\n=== audio_pipeline_ulcnet: same-delay relock resets the model ===\n");
    test_relock_same_delay_resets_model();

    printf("\n=== audio_pipeline_ulcnet: FIXED first alignment resets the model ===\n");
    test_fixed_first_alignment_resets_model();

    printf("\n=== audio_pipeline_ulcnet: identity-reprime straddle derivation ===\n");
    test_reprime_straddle_derivation();

    printf("\n=== audio_pipeline_ulcnet: identity-reprime behaviour ===\n");
    test_reprime_behavior();

    printf("\n=== audio_pipeline_ulcnet: fail-open callback handling ===\n");
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

    printf("\n=== audio_pipeline_ulcnet: far timestamp before acquisition ===\n");
    test_far_timestamp_before_acquisition();

    printf("\n=== audio_pipeline_ulcnet: model applies while UNLOCKED ===\n");
    test_model_applies_unlocked();

    printf("\n=== audio_pipeline_ulcnet: NaN/Inf guard ===\n");
    test_nan_guard();

    printf("\n=== audio_pipeline_ulcnet: full-write contract (partial-write guard) ===\n");
    test_partial_write_guard();

    printf("\n=== audio_pipeline_ulcnet: aligned-far descriptor gate ===\n");
    test_aligned_descriptor_gate();

    printf("\n=== audio_pipeline_ulcnet: known-delay profile (acquisition/coverage/mislock/cost) ===\n");
    test_known_delay_profile();

    if (g_failures) {
        fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
