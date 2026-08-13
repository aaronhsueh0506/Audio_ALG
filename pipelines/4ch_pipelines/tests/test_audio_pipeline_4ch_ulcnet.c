/**
 * tests/test_audio_pipeline_4ch_ulcnet.c — Align-ULCNet 4ch pipeline tests.
 *
 * Mirrors test_audio_pipeline_4ch.c's public-API acceptance style. Covers:
 *   1. identity E2E: the 2-hop timing contract (hops 0..1 zero; out[hop p]
 *      equals the beamformed-error accessor value captured at hop p-1)
 *      through the WOLA reconstruction + ULCNet chain in fail-open mode.
 *   2. counting model (ULCNET_FAR_ALIGNED): infer STEPPED on every emitted
 *      frame (constant per-hop compute, matching the mono variant) while
 *      its output is APPLIED only on locked frames; model->reset fired on
 *      every delay change AND on pipeline reset; fail-open on infer()
 *      error (output stays identity, calls continue).
 *   3. core PreFrame extension: pre.aligned_ref non-NULL, byte-exact against
 *      an independently maintained delayed-far reference, delay lock on a
 *      delayed far, and the abandon_pre token protocol.
 *   4. pool rejection / 8-point descriptor gate / destroy idempotence /
 *      pool reuse, plus a short heap-vs-pool byte-equal run.
 *   5. far-timestamp (ULCNET_FAR_RAW, the default): far-passthrough model,
 *      silence on all mics, one unit impulse in far at a known index -- it
 *      must land in the output at EXACTLY impulse + 2 hops (512 samples):
 *      the wrapper's one-hop far-compensation buffer (matching the beam
 *      WOLA's one-hop lag, so err/far frame pairs are same-hop) plus the
 *      one-hop centered ULCNet chain. Applied delay contributes 0 here
 *      (the shared delay never acquires on silent mics). Goes red by
 *      exactly 256 samples if the far compensation buffer is removed.
 *   6. RAW mode never gates on the delay lock: the 0.5x model's output is
 *      applied from the FIRST emitted frame, before the delay is solid.
 *   7. NaN guard: rc==0 frames poisoned with NaN/Inf are discarded
 *      (bit-identical to the NULL-model pipeline under the 50%-overlap
 *      mixing rule), the next clean frame recovers, no NaN in the output.
 *   8. full-write contract: rc==0 frames that wrote only the FIRST 100
 *      bins are discarded bitwise (the pipeline's NaN pre-fill of the
 *      staging buffers leaves the unwritten bins non-finite, so the
 *      finite guard catches the partial write), and the next fully
 *      written frame recovers. MUTATION: removing the pre-fill in
 *      audio_pipeline_4ch_ulcnet.c leaks stale finite values into the
 *      unwritten bins and this test goes red.
 *   9. far-input mode switch lock: set_far_input_mode succeeds before any
 *      hop is processed, is REJECTED (nonzero, mode unchanged -- verified
 *      via the far_input_mode getter) once a hop has been processed, and
 *      succeeds again after audio_pipeline_4ch_ulcnet_reset().
 */

#include "audio_pipeline_4ch_ulcnet.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            return 0;                                                  \
        }                                                              \
    } while (0)

/* Deterministic noise for delay-lock stimuli (xorshift32). */
static uint32_t g_rng = 0x1234567u;
static float frand(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return ((float)(x >> 8) * (1.0f / 16777216.0f)) - 0.5f;
}

/* ============================================================================
 * 1. Identity E2E: WOLA + ULCNet chain + 2-hop latency contract
 * ========================================================================== */

static int test_identity_e2e(void) {
    enum { FRAMES = 60 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    float* microphones;
    float* far;
    float* out;
    float* beam_hist;
    float max_beam = 0.0f;
    float max_err = 0.0f;
    float max_preamble = 0.0f;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.4f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create ULCNet 4ch pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);
    CHECK(hop == ULCNET_HOP, "hop matches the ULCNet compile-time hop");
    CHECK(audio_pipeline_4ch_ulcnet_fft_size(p) == ULCNET_N_FFT &&
          audio_pipeline_4ch_ulcnet_n_freqs(p) == ULCNET_BINS &&
          audio_pipeline_4ch_ulcnet_sample_rate(p) == ULCNET_SR,
          "grid pinned to 16 kHz / 512 / 257");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    beam_hist = (float*)malloc((size_t)FRAMES * hop * sizeof(float));
    CHECK(microphones && far && out && beam_hist,
          "allocate identity-test buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        const float* beam;
        for (int i = 0; i < hop; ++i) {
            int64_t absolute = (int64_t)frame * hop + i;
            float phase = 2.0f * (float)M_PI * 700.0f *
                          (float)absolute / (float)ULCNET_SR;
            /* frames 0..1 silent so the 2-hop preamble is provably zero. */
            float echo = frame >= 2 ? 0.08f * sinf(phase) : 0.0f;
            float near = frame >= 30
                ? 0.025f * sinf(phase * 1.73f + 0.2f) : 0.0f;
            far[i] = echo;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) + near * (1.0f + 0.02f * ch);
            }
        }
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, frame >= 30, out) ==
                  FOUR_AEC_NR_RES_OK,
              "identity frame processes");

        beam = audio_pipeline_4ch_ulcnet_last_beamformed_error(p);
        CHECK(beam != NULL, "beamformed-error accessor non-NULL");
        memcpy(beam_hist + (size_t)frame * hop, beam,
               (size_t)hop * sizeof(float));

        for (int i = 0; i < hop; ++i) {
            float b = fabsf(beam_hist[(size_t)frame * hop + i]);
            if (b > max_beam) max_beam = b;
            CHECK(isfinite(out[i]), "identity output finite");
            if (frame <= 1) {
                float a = fabsf(out[i]);
                if (a > max_preamble) max_preamble = a;
            } else {
                float d = fabsf(
                    out[i] - beam_hist[(size_t)(frame - 1) * hop + i]);
                if (d > max_err) max_err = d;
            }
        }
    }
    CHECK(max_preamble <= 2e-4f,
          "hops 0..1 emit zeros (2-hop latency preamble)");
    CHECK(max_err <= 2e-4f,
          "out[hop p] equals beamformed[p-1] (identity chain, one extra hop)");
    /* The comparison must not be vacuous: the beamformed error itself has
     * to carry real signal for this test to be able to fail. */
    CHECK(max_beam > 1e-3f, "beamformed error is non-trivial");

    audio_pipeline_4ch_ulcnet_destroy(p);
    free(beam_hist);
    free(out);
    free(far);
    free(microphones);
    return 1;
}

/* ============================================================================
 * 2. Counting model: policy wiring (lock gating, delay-change reset,
 *    pipeline reset, fail-open on infer error)
 * ========================================================================== */

typedef struct CountingModel {
    long infer_calls;
    long reset_calls;
    int fail;       /* nonzero => infer reports an error */
    float scale;    /* marker gain written to the output spectrum */
} CountingModel;

static int counting_infer(
    void* user,
    const float* err_re, const float* err_im,
    const float* far_re, const float* far_im,
    float* out_re, float* out_im) {
    CountingModel* m = (CountingModel*)user;
    (void)far_re;
    (void)far_im;
    m->infer_calls += 1;
    for (int k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = m->scale * err_re[k];
        out_im[k] = m->scale * err_im[k];
    }
    return m->fail;
}

static void counting_reset(void* user) {
    ((CountingModel*)user)->reset_calls += 1;
}

static int test_counting_model_policy(void) {
    enum { FRAMES = 240, TRUE_DELAY = 400, FAIL_FRAMES = 8 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    CountingModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out;
    float* far_hist;   /* full far stream for the echo path */
    float* beam_hist;
    long expected_infer = 0;
    long changed_count = 0;
    int locked_frames = 0;
    int scaled_checks = 0;
    int identity_checks = 0;
    int prev_scaled = 0;
    float max_scaled_err = 0.0f;
    float max_identity_err = 0.0f;
    int hop;
    int total_frames = FRAMES + FAIL_FRAMES;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create counting-model pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.reset = counting_reset;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install counting model");
    CHECK(audio_pipeline_4ch_ulcnet_set_far_input_mode(
              p, (UlcnetFarInputMode)99) != 0,
          "set_far_input_mode rejects an undefined mode value");
    CHECK(audio_pipeline_4ch_ulcnet_set_far_input_mode(
              p, ULCNET_FAR_ALIGNED) == 0,
          "select ULCNET_FAR_ALIGNED (the lock-gated mode)");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)total_frames * hop, sizeof(float));
    beam_hist = (float*)calloc((size_t)total_frames * hop, sizeof(float));
    CHECK(microphones && far && out && far_hist && beam_hist,
          "allocate counting-model buffers");

    for (int frame = 0; frame < total_frames; ++frame) {
        const float* beam;
        int frames_in_hop = frame == 0 ? 0 : (frame == 1 ? 2 : 1);
        int scaled;

        if (frame == FRAMES) {
            /* Flip to the error-reporting regime: infer keeps being called
             * but its (deliberately wrong, 3x-scaled) output must be
             * ignored -- fail-open identity. */
            m.fail = 1;
            m.scale = 3.0f;
        }

        for (int i = 0; i < hop; ++i) {
            int64_t t = (int64_t)frame * hop + i;
            float noise = 0.25f * frand();
            float echo;
            far_hist[t] = noise;
            far[i] = noise;
            echo = t >= TRUE_DELAY
                ? 0.6f * far_hist[t - TRUE_DELAY] : 0.0f;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.02f * ch);
            }
        }
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "counting-model frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(p, &delay) == 0,
              "delay-state accessor works");

        beam = audio_pipeline_4ch_ulcnet_last_beamformed_error(p);
        memcpy(beam_hist + (size_t)frame * hop, beam,
               (size_t)hop * sizeof(float));

        if (delay.changed) changed_count += 1;
        scaled = delay.solid && delay.delay_samples >= 0;
        if (scaled) locked_frames += 1;
        /* Step-consistency (matches the mono variant): infer runs for EVERY
         * emitted frame -- locked or not -- so the per-hop compute budget
         * is constant and the runtime state stays continuous; the lock only
         * gates whether its output is APPLIED (checked further below). */
        expected_infer += frames_in_hop;
        CHECK(m.infer_calls == expected_infer,
              "infer stepped on every emitted frame (lock gates application only)");

        /* Output relation: both contributing synthesis frames (hop p and
         * hop p-1) must have used the same per-frame gain for the emitted
         * hop to be a pure scaled copy; skip the blend frames. Once
         * m.fail is set the expected gain is identity again. */
        if (frame >= 2) {
            float expected_gain =
                (scaled && !m.fail) ? m.scale : 1.0f;
            int comparable =
                (scaled == prev_scaled) &&
                (frame < FRAMES || frame >= FRAMES + 2);
            if (comparable) {
                float e = 0.0f;
                for (int i = 0; i < hop; ++i) {
                    float d = fabsf(
                        out[i] - expected_gain *
                        beam_hist[(size_t)(frame - 1) * hop + i]);
                    if (d > e) e = d;
                }
                if (scaled && !m.fail) {
                    if (e > max_scaled_err) max_scaled_err = e;
                    scaled_checks += 1;
                } else {
                    if (e > max_identity_err) max_identity_err = e;
                    identity_checks += 1;
                }
            }
        }
        prev_scaled = scaled;
    }

    CHECK(locked_frames > 20, "delay actually locks on the delayed far");
    CHECK(changed_count >= 1, "at least one delay change event fired");
    CHECK(m.reset_calls == changed_count,
          "model->reset fired exactly once per delay change");
    CHECK(scaled_checks > 10,
          "enough locked frames were compared in scaled mode");
    CHECK(max_scaled_err <= 2e-4f,
          "locked frames carry the model's 0.5x output (infer wired in)");
    CHECK(identity_checks >= FAIL_FRAMES - 2,
          "fail-open frames were compared");
    CHECK(max_identity_err <= 2e-4f,
          "infer error => fail-open identity (3x marker ignored)");

    /* Pipeline reset must also reset the runtime and restart the 2-hop
     * preamble. */
    audio_pipeline_4ch_ulcnet_reset(p);
    CHECK(m.reset_calls == changed_count + 1,
          "pipeline reset fires model->reset once");
    {
        long infer_before = m.infer_calls;
        float max_out = 0.0f;
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "post-reset frame processes");
        for (int i = 0; i < hop; ++i) {
            float a = fabsf(out[i]);
            if (a > max_out) max_out = a;
        }
        CHECK(max_out == 0.0f, "post-reset hop#0 emits exact zeros");
        CHECK(m.infer_calls == infer_before,
              "post-reset hop#0 emits no frames to the model");
    }

    audio_pipeline_4ch_ulcnet_destroy(p);
    free(beam_hist);
    free(far_hist);
    free(out);
    free(far);
    free(microphones);
    return 1;
}

/* ============================================================================
 * 3. Core PreFrame extension: aligned_ref + delay lock + abandon protocol
 * ========================================================================== */

static int test_core_aligned_ref_and_abandon(void) {
    enum { FRAMES = 240, TRUE_DELAY = 400 };
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
    FourAecNrRes* core;
    FourAecNrResPreFrame pre;
    FourAecNrResFrameToken first_token;
    float* microphones;
    float* far;
    float* far_hist;
    int hop;
    int locked = 0;
    int aligned_checked = 0;
    int have_first_token = 0;

    cfg.fft_size = 512;
    cfg.enable_cng = 0;
    core = four_aec_nr_res_create(&cfg);
    CHECK(core != NULL, "create 512/256 core");
    hop = four_aec_nr_res_hop_size(core);
    CHECK(hop == 256, "core hop is 256");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && far_hist, "allocate core-test buffers");

    /* No frame pending yet: abandoning must be a sequence error. */
    memset(&first_token, 0, sizeof(first_token));
    CHECK(four_aec_nr_res_abandon_pre(core, &first_token) ==
              FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "abandon with no pending frame is rejected");
    CHECK(four_aec_nr_res_abandon_pre(core, NULL) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "abandon with NULL token is rejected");

    for (int frame = 0; frame < FRAMES; ++frame) {
        for (int i = 0; i < hop; ++i) {
            int64_t t = (int64_t)frame * hop + i;
            float noise = 0.25f * frand();
            float echo;
            far_hist[t] = noise;
            far[i] = noise;
            echo = t >= TRUE_DELAY
                ? 0.6f * far_hist[t - TRUE_DELAY] : 0.0f;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.02f * ch);
            }
        }
        /* Zero the whole out-struct first: if process_pre ever stopped
         * assigning aligned_ref, the non-NULL check below would fail
         * instead of reading leftover stack garbage. */
        memset(&pre, 0, sizeof(pre));
        CHECK(four_aec_nr_res_process_pre(
                  core, microphones, far, &pre) == FOUR_AEC_NR_RES_OK,
              "core pre processes");
        CHECK(pre.aligned_ref != NULL,
              "pre.aligned_ref is assigned (non-NULL)");

        /* Byte-exact aligned-far check against our own far history: the
         * ring path is a pure copy, so equality is exact, on every frame
         * and for whatever delay the estimator currently applies. */
        CHECK(pre.delay.delay_samples >= 0, "applied delay is non-negative");
        for (int i = 0; i < hop; ++i) {
            int64_t t = (int64_t)frame * hop + i;
            float expected =
                t >= (int64_t)pre.delay.delay_samples
                    ? far_hist[t - pre.delay.delay_samples]
                    : 0.0f;
            if (pre.aligned_ref[i] != expected) {
                CHECK(0, "aligned_ref content matches the delayed far");
            }
            aligned_checked += 1;
        }

        if (pre.delay.solid && pre.delay.delay_samples > 0) locked = 1;

        if (!have_first_token) {
            first_token = pre.token;
            have_first_token = 1;
        } else {
            /* A stale (older-frame) token must not release this frame. */
            CHECK(four_aec_nr_res_abandon_pre(core, &first_token) ==
                      FOUR_AEC_NR_RES_SEQUENCE_ERROR,
                  "stale token cannot abandon the pending frame");
        }
        CHECK(four_aec_nr_res_abandon_pre(core, &pre.token) ==
                  FOUR_AEC_NR_RES_OK,
              "abandon releases the pending frame");
        CHECK(four_aec_nr_res_abandon_pre(core, &pre.token) ==
                  FOUR_AEC_NR_RES_SEQUENCE_ERROR,
              "double abandon is rejected");
    }

    CHECK(locked,
          "delayed far eventually locks with positive delay_samples");
    CHECK(aligned_checked == FRAMES * hop,
          "aligned_ref was checked on every sample");

    free(far_hist);
    free(far);
    free(microphones);
    four_aec_nr_res_destroy(core);
    return 1;
}

/* ============================================================================
 * 4. Pool rejection / descriptor gate / destroy idempotence
 * ========================================================================== */

static int test_pool_and_descriptor_gate(void) {
    enum { EXTRA = 32, FRAMES = 12 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnetMemReq req;
    AudioPipeline4ChUlcnetMemReq stale;
    AudioPipeline4ChUlcnet* stat;
    AudioPipeline4ChUlcnet* heap;
    unsigned char* pool = NULL;
    float* microphones;
    float* far;
    float* heap_out;
    float* stat_out;
    size_t pool_bytes;
    size_t k;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.35f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    CHECK(audio_pipeline_4ch_ulcnet_get_mem_requirements(&cfg, &req) == 0,
          "memory requirement query succeeds");
    CHECK(req.bytes <= (uint64_t)SIZE_MAX, "requirement fits size_t");
    pool_bytes = (size_t)req.bytes + (size_t)EXTRA;

    CHECK(posix_memalign(
              (void**)&pool, (size_t)req.alignment, pool_bytes) == 0 && pool,
          "aligned caller pool allocates");
    memset(pool, 0xa5, pool_bytes);

    CHECK(audio_pipeline_4ch_ulcnet_init(
              pool + 1, (size_t)req.bytes, &cfg) == NULL,
          "init rejects a misaligned pool");
    CHECK(audio_pipeline_4ch_ulcnet_init(
              pool, (size_t)req.bytes - 1u, &cfg) == NULL,
          "init rejects an undersized pool");

    stale = req;
    stale.descriptor_version += 1u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects a stale descriptor ABI");
    stale = req;
    stale.layout_version += 1u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects a stale carve layout");
    stale = req;
    stale.backend_id = 99u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects a backend mismatch");
    stale = req;
    stale.build_flags_hash ^= 0xffffffffu;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects a build-layout mismatch");
    stale = req;
    stale.alignment *= 2u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects an alignment mismatch");
    stale = req;
    stale.reserved = 1u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects a corrupt reserved field");
    stale = req;
    stale.bytes -= 1u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects stale descriptor bytes");
    stale = req;
    stale.layout_version -= 1u;
    stale.bytes += 4096u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects a stale layout even with larger cached bytes");

    stat = audio_pipeline_4ch_ulcnet_init_ex(
        pool, (size_t)req.bytes, &cfg, &req);
    heap = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(stat != NULL && heap != NULL,
          "poisoned caller pool and heap construction both succeed");

    for (k = 0; k < (size_t)EXTRA; ++k) {
        if (pool[(size_t)req.bytes + k] != 0xa5) break;
    }
    CHECK(k == (size_t)EXTRA, "init stays inside the queried pool");

    hop = audio_pipeline_4ch_ulcnet_hop_size(stat);
    CHECK(hop == audio_pipeline_4ch_ulcnet_hop_size(heap) &&
              hop == ULCNET_HOP,
          "heap and pool instances agree on hop size");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    heap_out = (float*)malloc((size_t)hop * sizeof(float));
    stat_out = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far && heap_out && stat_out,
          "parity buffers allocate");

    for (int frame = 0; frame < FRAMES; ++frame) {
        for (int i = 0; i < hop; ++i) {
            int64_t absolute = (int64_t)frame * hop + i;
            float phase = 2.0f * (float)M_PI * 700.0f *
                          (float)absolute / (float)ULCNET_SR;
            float echo = 0.08f * sinf(phase);
            far[i] = echo;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch);
            }
        }
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  heap, microphones, far, 0, heap_out) ==
                  FOUR_AEC_NR_RES_OK,
              "heap instance processes");
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  stat, microphones, far, 0, stat_out) ==
                  FOUR_AEC_NR_RES_OK,
              "pool instance processes");
        CHECK(memcmp(heap_out, stat_out,
                     (size_t)hop * sizeof(float)) == 0,
              "heap/pool output is byte-identical");
    }

    audio_pipeline_4ch_ulcnet_destroy(stat);
    CHECK(audio_pipeline_4ch_ulcnet_hop_size(stat) == -1,
          "destroy marks a caller-pool instance inactive");
    CHECK(audio_pipeline_4ch_ulcnet_last_beamformed_error(stat) == NULL,
          "destroyed instance returns NULL beamformed accessor");
    audio_pipeline_4ch_ulcnet_destroy(stat);
    CHECK(audio_pipeline_4ch_ulcnet_hop_size(stat) == -1,
          "caller-pool destroy is idempotent");
    stat = audio_pipeline_4ch_ulcnet_init_ex(
        pool, (size_t)req.bytes, &cfg, &req);
    CHECK(stat != NULL, "caller pool is reusable after destroy");

    audio_pipeline_4ch_ulcnet_destroy(stat);
    audio_pipeline_4ch_ulcnet_destroy(heap);
    free(stat_out);
    free(heap_out);
    free(far);
    free(microphones);
    free(pool);
    return 1;
}

/* ============================================================================
 * 5. Far-timestamp (ULCNET_FAR_RAW default): the model's far branch must
 *    carry the SAME input hop as its error branch. Far-passthrough model,
 *    silence on all mics, a single unit impulse in far at sample T. The
 *    expected output position is derived, not measured:
 *      + 0    applied delay (silent mics -> the shared delay never
 *             acquires; process_pre applies delay 0 -- and RAW mode feeds
 *             the caller's far directly anyway)
 *      + 256  the wrapper's one-hop far-compensation buffer (far frames
 *             are delayed one hop to match the beam WOLA's one-hop lag)
 *      + 256  the centered ULCNet chain (hop #p output = chain input hop
 *             p-1) closed by its WOLA
 *      = T + 512.
 *    MUTATION PROOF: pushing this hop's far directly (removing the
 *    one-hop far buffer) moves the reconstructed impulse to T + 256 --
 *    this check then fails with offset -256.
 * ========================================================================== */

static int passthrough_far_infer(
    void* user,
    const float* err_re, const float* err_im,
    const float* far_re, const float* far_im,
    float* out_re, float* out_im) {
    (void)user;
    (void)err_re;
    (void)err_im;
    memcpy(out_re, far_re, ULCNET_BINS * sizeof(float));
    memcpy(out_im, far_im, ULCNET_BINS * sizeof(float));
    return 0;
}

static int test_far_timestamp_raw(void) {
    enum { FRAMES = 40, IMP_FRAME = 8, IMP_OFF = 37 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    UlcnetModel model;
    float* microphones;
    float* far;
    float* out_hist;
    int hop;
    int imp_index;
    int expect_index;
    int found_index = -1;
    float peak = 0.0f;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.4f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create far-timestamp pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);
    imp_index = IMP_FRAME * hop + IMP_OFF;
    expect_index = imp_index + 2 * hop;

    memset(&model, 0, sizeof(model));
    model.infer = passthrough_far_infer;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install far-passthrough model");
    /* far_input_mode stays the ULCNET_FAR_RAW default: applied without a
     * delay lock (none ever happens on silent mics). */

    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    out_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out_hist,
          "allocate far-timestamp buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        memset(far, 0, (size_t)hop * sizeof(float));
        if (frame == IMP_FRAME) far[IMP_OFF] = 1.0f;
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0,
                  out_hist + (size_t)frame * hop) == FOUR_AEC_NR_RES_OK,
              "far-timestamp frame processes");
    }

    for (int t = 0; t < FRAMES * hop; ++t) {
        float a = fabsf(out_hist[t]);
        if (a > peak) {
            peak = a;
            found_index = t;
        }
    }
    printf("far timestamp (4ch RAW): impulse at far[%d] -> out[%d] "
           "(expected %d, offset %+d samples, peak %.4f)\n",
           imp_index, found_index, expect_index,
           found_index - expect_index, peak);
    CHECK(found_index == expect_index,
          "far impulse lands at impulse + 2 hops (1-hop far buffer + "
          "1-hop ULCNet chain; no err/far skew)");
    CHECK(peak > 0.9f, "far impulse reconstructed at ~unit amplitude");

    audio_pipeline_4ch_ulcnet_destroy(p);
    free(out_hist);
    free(far);
    free(microphones);
    return 1;
}

/* ============================================================================
 * 6. RAW mode never gates application on the delay lock: the 0.5x model's
 *    output must appear from the FIRST emitted frame, while delay.solid is
 *    still 0 (the paper contract does not depend on lock).
 * ========================================================================== */

static int test_raw_mode_applies_unlocked(void) {
    enum { FRAMES = 60, TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    CountingModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out;
    float* far_hist;
    float* beam_hist;
    long expected_infer = 0;
    int unsolid_checks = 0;
    int first_solid_frame = -1;
    float max_err = 0.0f;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create RAW-mode pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.reset = counting_reset;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install 0.5x model (RAW default mode)");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    beam_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out && far_hist && beam_hist,
          "allocate RAW-mode buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        const float* beam;
        int frames_in_hop = frame == 0 ? 0 : (frame == 1 ? 2 : 1);
        for (int i = 0; i < hop; ++i) {
            int64_t t = (int64_t)frame * hop + i;
            float noise = 0.25f * frand();
            float echo;
            far_hist[t] = noise;
            far[i] = noise;
            echo = t >= TRUE_DELAY
                ? 0.6f * far_hist[t - TRUE_DELAY] : 0.0f;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.02f * ch);
            }
        }
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "RAW-mode frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(p, &delay) == 0,
              "RAW-mode delay accessor works");
        if (first_solid_frame < 0 && delay.solid) first_solid_frame = frame;

        beam = audio_pipeline_4ch_ulcnet_last_beamformed_error(p);
        memcpy(beam_hist + (size_t)frame * hop, beam,
               (size_t)hop * sizeof(float));

        expected_infer += frames_in_hop;
        CHECK(m.infer_calls == expected_infer,
              "RAW mode also steps infer on every emitted frame");

        /* Applied from the very first emitted frames: out[p] must be the
         * 0.5x-scaled beam stream on EVERY frame >= 2 (skip delay-change
         * boundary frames where the beam itself is discontinuous). */
        if (frame >= 2 && !delay.changed) {
            float e = 0.0f;
            for (int i = 0; i < hop; ++i) {
                float d = fabsf(
                    out[i] - m.scale *
                    beam_hist[(size_t)(frame - 1) * hop + i]);
                if (d > e) e = d;
            }
            if (e > max_err) max_err = e;
            if (!delay.solid) unsolid_checks += 1;
        }
    }

    CHECK(unsolid_checks > 5,
          "RAW-mode application was verified on genuinely unlocked frames");
    (void)first_solid_frame;
    CHECK(max_err <= 2e-4f,
          "RAW mode applies the model from the first emitted frame "
          "(no delay-lock gating of application)");

    audio_pipeline_4ch_ulcnet_destroy(p);
    free(beam_hist);
    free(far_hist);
    free(out);
    free(far);
    free(microphones);
    return 1;
}

/* ============================================================================
 * 7. NaN guard: infer() returns rc==0 but poisons the spectrum with
 *    NaN/Inf on scheduled hops. Those frames must take the identity path
 *    BITWISE (equal to a NULL-model pipeline fed identical input, under
 *    the 50%-overlap mixing rule: output hop p mixes the frames pushed at
 *    hops p-1 and p), the next clean frame must be applied again, and no
 *    NaN may ever reach the WOLA/output.
 * ========================================================================== */

typedef struct NanModel {
    int poison;   /* set by the test before each process call */
} NanModel;

static int nan_infer(
    void* user,
    const float* err_re, const float* err_im,
    const float* far_re, const float* far_im,
    float* out_re, float* out_im) {
    NanModel* m = (NanModel*)user;
    (void)far_re;
    (void)far_im;
    for (int k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = 0.5f * err_re[k];
        out_im[k] = 0.5f * err_im[k];
    }
    if (m->poison) {
        /* rc stays 0: only the finite-output guard can catch this frame.
         * Mid-array positions so a partial scan would miss them. */
        out_re[100] = nanf("");
        out_im[200] = INFINITY;
    }
    return 0;
}

static int test_nan_guard(void) {
    enum { FRAMES = 60, TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* pa;
    AudioPipeline4ChUlcnet* pb;
    NanModel m;
    UlcnetModel model;
    float* microphones;
    float* far;
    float* out_a;
    float* out_b;
    float* far_hist;
    int poisoned[FRAMES];
    int n_equal = 0;
    int n_differ = 0;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    pa = audio_pipeline_4ch_ulcnet_create(&cfg);
    pb = audio_pipeline_4ch_ulcnet_create(&cfg);   /* NULL model: identity */
    CHECK(pa != NULL && pb != NULL, "create NaN-guard pipeline pair");
    hop = audio_pipeline_4ch_ulcnet_hop_size(pa);

    memset(&m, 0, sizeof(m));
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = nan_infer;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(pa, &model) == 0,
          "install NaN-poisoning model (RAW default mode)");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out_a = (float*)malloc((size_t)hop * sizeof(float));
    out_b = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out_a && out_b && far_hist,
          "allocate NaN-guard buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        int equal_expected;
        int bitwise_equal;
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        m.poison = (frame >= 20 && frame < 28) || (frame >= 40 && frame < 48);
        poisoned[frame] = m.poison;

        for (int i = 0; i < hop; ++i) {
            int64_t t = (int64_t)frame * hop + i;
            float noise = 0.25f * frand();
            float echo;
            far_hist[t] = noise;
            far[i] = noise;
            echo = t >= TRUE_DELAY
                ? 0.6f * far_hist[t - TRUE_DELAY] : 0.0f;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.02f * ch);
            }
        }
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pa, microphones, far, 0, out_a) == FOUR_AEC_NR_RES_OK,
              "NaN-guard model frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pb, microphones, far, 0, out_b) == FOUR_AEC_NR_RES_OK,
              "NaN-guard reference frame processes");

        for (int i = 0; i < hop; ++i) {
            CHECK(isfinite(out_a[i]),
                  "no NaN/Inf ever reaches the pipeline output");
        }

        /* 50%-overlap mixing rule (same as the mono test's): hop 0 emits
         * exact zeros on both sides; hop 1 mixes ONLY the two frames
         * pushed at hop 1 (and the beam is NOT exactly zero there -- the
         * AEC lanes leak ~1e-12 residual from the far noise even on
         * silent mics, so an applied 0.5x frame already differs); hop
         * p >= 2 mixes the frames pushed at hops p-1 and p. */
        if (frame == 0) {
            equal_expected = 1;
        } else if (frame == 1) {
            equal_expected = poisoned[1];
        } else {
            equal_expected = poisoned[frame - 1] && poisoned[frame];
        }
        bitwise_equal =
            memcmp(out_a, out_b, (size_t)hop * sizeof(float)) == 0;
        if (equal_expected) {
            n_equal += 1;
            CHECK(bitwise_equal,
                  "rc==0 NaN frames are discarded bitwise (identity path)");
        } else if (frame >= 3) {
            /* frame 2's beam may still be zero (echo starts at t=400 which
             * lands in beam hop 1, reconstructed at hop 2 -- keep a one-
             * frame margin); from frame 3 the 0.5x scale must show. */
            n_differ += 1;
            CHECK(!bitwise_equal,
                  "clean frames after a NaN window are applied again");
        }
    }
    CHECK(n_equal >= 10 && n_differ >= 10,
          "NaN-guard coverage: both regimes actually compared");

    audio_pipeline_4ch_ulcnet_destroy(pa);
    audio_pipeline_4ch_ulcnet_destroy(pb);
    free(far_hist);
    free(out_b);
    free(out_a);
    free(far);
    free(microphones);
    return 1;
}

/* ============================================================================
 * 8. Full-write contract: infer() writes only the FIRST 100 bins (0.5x)
 *    on scheduled hops but still returns rc==0. The pipeline pre-fills the
 *    enh_re/enh_im staging with NaN before every infer call
 *    (ulcnet_process.h's FULL-WRITE CONTRACT), so the 157 unwritten bins
 *    stay NaN and the finite guard discards the frame: bit-identical to
 *    the NULL-model pipeline under the same 50%-overlap mixing rule as
 *    test 7, and the next fully-written frame is applied again.
 *    MUTATION PROOF: removing the pre-fill loop in
 *    audio_pipeline_4ch_ulcnet.c leaves the previous frame's stale FINITE
 *    values in bins 100..256 -- the guard cannot catch them, the partial
 *    frames get applied, and the bit-identical check goes red.
 * ========================================================================== */

typedef struct PartialModel {
    int partial;   /* set by the test before each process call */
} PartialModel;

static int partial_write_infer(
    void* user,
    const float* err_re, const float* err_im,
    const float* far_re, const float* far_im,
    float* out_re, float* out_im) {
    PartialModel* m = (PartialModel*)user;
    int n = m->partial ? 100 : ULCNET_BINS;
    (void)far_re;
    (void)far_im;
    for (int k = 0; k < n; ++k) {
        out_re[k] = 0.5f * err_re[k];
        out_im[k] = 0.5f * err_im[k];
    }
    /* rc stays 0 even for the partial write: only the pipeline's NaN
     * pre-fill + finite guard can catch the 157 unwritten bins. */
    return 0;
}

static int test_partial_write_guard(void) {
    enum { FRAMES = 60, TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* pa;
    AudioPipeline4ChUlcnet* pb;
    PartialModel m;
    UlcnetModel model;
    float* microphones;
    float* far;
    float* out_a;
    float* out_b;
    float* far_hist;
    int partial[FRAMES];
    int n_equal = 0;
    int n_differ = 0;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    pa = audio_pipeline_4ch_ulcnet_create(&cfg);
    pb = audio_pipeline_4ch_ulcnet_create(&cfg);   /* NULL model: identity */
    CHECK(pa != NULL && pb != NULL, "create partial-write pipeline pair");
    hop = audio_pipeline_4ch_ulcnet_hop_size(pa);

    memset(&m, 0, sizeof(m));
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = partial_write_infer;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(pa, &model) == 0,
          "install partial-write model (RAW default mode)");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out_a = (float*)malloc((size_t)hop * sizeof(float));
    out_b = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out_a && out_b && far_hist,
          "allocate partial-write buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        int equal_expected;
        int bitwise_equal;
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        m.partial = (frame >= 20 && frame < 28) || (frame >= 40 && frame < 48);
        partial[frame] = m.partial;

        for (int i = 0; i < hop; ++i) {
            int64_t t = (int64_t)frame * hop + i;
            float noise = 0.25f * frand();
            float echo;
            far_hist[t] = noise;
            far[i] = noise;
            echo = t >= TRUE_DELAY
                ? 0.6f * far_hist[t - TRUE_DELAY] : 0.0f;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.02f * ch);
            }
        }
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pa, microphones, far, 0, out_a) == FOUR_AEC_NR_RES_OK,
              "partial-write model frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pb, microphones, far, 0, out_b) == FOUR_AEC_NR_RES_OK,
              "partial-write reference frame processes");

        for (int i = 0; i < hop; ++i) {
            CHECK(isfinite(out_a[i]),
                  "no NaN/Inf ever reaches the output (partial-write test)");
        }

        /* Same 50%-overlap mixing rule as test 7. */
        if (frame == 0) {
            equal_expected = 1;
        } else if (frame == 1) {
            equal_expected = partial[1];
        } else {
            equal_expected = partial[frame - 1] && partial[frame];
        }
        bitwise_equal =
            memcmp(out_a, out_b, (size_t)hop * sizeof(float)) == 0;
        if (equal_expected) {
            n_equal += 1;
            CHECK(bitwise_equal,
                  "rc==0 partial-write frames are discarded bitwise "
                  "(NaN pre-fill catches the unwritten bins)");
        } else if (frame >= 3) {
            /* Same one-frame margin as test 7 (echo reaches the beam at
             * frame 3). */
            n_differ += 1;
            CHECK(!bitwise_equal,
                  "fully-written frames after a partial window are applied "
                  "again");
        }
    }
    CHECK(n_equal >= 10 && n_differ >= 10,
          "partial-write coverage: both regimes actually compared");

    audio_pipeline_4ch_ulcnet_destroy(pa);
    audio_pipeline_4ch_ulcnet_destroy(pb);
    free(far_hist);
    free(out_b);
    free(out_a);
    free(far);
    free(microphones);
    return 1;
}

/* ============================================================================
 * 9. Far-input mode switch lock: allowed only while no hop has been
 *    processed; rejected mid-stream (mode unchanged, verified via the
 *    getter); allowed again after reset.
 * ========================================================================== */

static int test_far_mode_switch_rejected_midstream(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    float* microphones;
    float* far;
    float* out;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create mode-switch pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);

    CHECK(audio_pipeline_4ch_ulcnet_far_input_mode(NULL) == -1,
          "far_input_mode getter returns -1 for NULL");
    CHECK(audio_pipeline_4ch_ulcnet_far_input_mode(p) == ULCNET_FAR_RAW,
          "instances start in ULCNET_FAR_RAW");

    /* Before any hop: switching (both directions) succeeds. */
    CHECK(audio_pipeline_4ch_ulcnet_set_far_input_mode(
              p, ULCNET_FAR_ALIGNED) == 0,
          "pre-stream switch to ALIGNED succeeds");
    CHECK(audio_pipeline_4ch_ulcnet_far_input_mode(p) == ULCNET_FAR_ALIGNED,
          "getter reflects the pre-stream switch");
    CHECK(audio_pipeline_4ch_ulcnet_set_far_input_mode(
              p, ULCNET_FAR_RAW) == 0,
          "pre-stream switch back to RAW succeeds");

    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far && out, "allocate mode-switch buffers");

    CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
              p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
          "one hop processes before the mid-stream switch attempt");

    /* Mid-stream: rejected, mode unchanged. */
    CHECK(audio_pipeline_4ch_ulcnet_set_far_input_mode(
              p, ULCNET_FAR_ALIGNED) != 0,
          "mid-stream switch is rejected after one processed hop");
    CHECK(audio_pipeline_4ch_ulcnet_far_input_mode(p) == ULCNET_FAR_RAW,
          "rejected mid-stream switch leaves the mode unchanged");

    /* Undefined values keep being rejected too (reject-first ordering). */
    CHECK(audio_pipeline_4ch_ulcnet_set_far_input_mode(
              p, (UlcnetFarInputMode)99) != 0,
          "undefined mode value still rejected mid-stream");

    /* After reset (frame_index back to 0) switching is allowed again. */
    audio_pipeline_4ch_ulcnet_reset(p);
    CHECK(audio_pipeline_4ch_ulcnet_set_far_input_mode(
              p, ULCNET_FAR_ALIGNED) == 0,
          "switch succeeds again after reset");
    CHECK(audio_pipeline_4ch_ulcnet_far_input_mode(p) == ULCNET_FAR_ALIGNED,
          "getter reflects the post-reset switch");

    /* NOTE: no post-destroy getter probe here -- this is a create() (heap)
     * instance, so destroy() frees the memory containing `p` itself; the
     * destroyed-accessor contract is covered on the caller-pool instance
     * in test 4. */
    audio_pipeline_4ch_ulcnet_destroy(p);
    free(out);
    free(far);
    free(microphones);
    return 1;
}

/* ============================================================================
 * Driver
 * ========================================================================== */

static int run_all_tests(void) {
    /* Grid gate: this wrapper is 16 kHz / 512 only. */
    {
        AudioPipeline4ChConfig invalid =
            audio_pipeline_4ch_default_config(48000);
        AudioPipeline4ChUlcnetMemReq req;
        CHECK(audio_pipeline_4ch_ulcnet_get_mem_requirements(
                  &invalid, &req) != 0,
              "48 kHz config is rejected");
        CHECK(audio_pipeline_4ch_ulcnet_create(&invalid) == NULL,
              "48 kHz create is rejected");
        invalid = audio_pipeline_4ch_ulcnet_default_config();
        invalid.core.fft_size = 256;
        CHECK(audio_pipeline_4ch_ulcnet_create(&invalid) == NULL,
              "core fft 256 is rejected (ULCNet grid is 512/256)");
        invalid = audio_pipeline_4ch_ulcnet_default_config();
        invalid.core.fft_size = 0;
        {
            AudioPipeline4ChUlcnet* forced =
                audio_pipeline_4ch_ulcnet_create(&invalid);
            CHECK(forced != NULL, "core fft 0 is accepted");
            CHECK(audio_pipeline_4ch_ulcnet_fft_size(forced) == 512,
                  "core fft 0 is forced to 512, not the core's 256 default");
            audio_pipeline_4ch_ulcnet_destroy(forced);
        }
    }
    CHECK(test_identity_e2e(),
          "identity E2E / 2-hop timing contract");
    CHECK(test_counting_model_policy(),
          "counting model policy (stepping, lock gating, resets, fail-open)");
    CHECK(test_core_aligned_ref_and_abandon(),
          "core PreFrame aligned_ref + abandon protocol");
    CHECK(test_pool_and_descriptor_gate(),
          "pool rejection / descriptor gate / destroy idempotence");
    CHECK(test_far_timestamp_raw(),
          "far timestamp: err/far frame pairs are same-hop (RAW mode)");
    CHECK(test_raw_mode_applies_unlocked(),
          "RAW mode applies without a delay lock");
    CHECK(test_nan_guard(),
          "NaN guard: non-finite model output never reaches the WOLA");
    CHECK(test_partial_write_guard(),
          "full-write contract: partial rc==0 frames are discarded");
    CHECK(test_far_mode_switch_rejected_midstream(),
          "far-input mode switch is locked once streaming has started");
    printf("All audio_pipeline_4ch_ulcnet tests passed\n");
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
