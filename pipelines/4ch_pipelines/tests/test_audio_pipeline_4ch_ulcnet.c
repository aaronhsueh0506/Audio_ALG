/**
 * tests/test_audio_pipeline_4ch_ulcnet.c — Align-ULCNet 4ch pipeline tests.
 *
 * Mirrors test_audio_pipeline_4ch.c's public-API acceptance style. Covers:
 *   1. identity E2E: the 2-hop timing contract (hops 0..1 zero; out[hop p]
 *      equals the beamformed-error accessor value captured at hop p-1)
 *      through the WOLA reconstruction + ULCNet chain in fail-open mode.
 *   2. counting model: infer called exactly on locked, non-bypassed frames;
 *      model->reset fired on every delay change AND on pipeline reset;
 *      fail-open on infer() error (output stays identity, calls continue).
 *   3. core PreFrame extension: pre.aligned_ref non-NULL, byte-exact against
 *      an independently maintained delayed-far reference, delay lock on a
 *      delayed far, and the abandon_pre token protocol.
 *   4. pool rejection / 8-point descriptor gate / destroy idempotence /
 *      pool reuse, plus a short heap-vs-pool byte-equal run.
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
        if (scaled) {
            locked_frames += 1;
            expected_infer += frames_in_hop;
        }
        CHECK(m.infer_calls == expected_infer,
              "infer called exactly on locked, non-bypassed frames");

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
          "counting model policy (lock gating, resets, fail-open)");
    CHECK(test_core_aligned_ref_and_abandon(),
          "core PreFrame aligned_ref + abandon protocol");
    CHECK(test_pool_and_descriptor_gate(),
          "pool rejection / descriptor gate / destroy idempotence");
    printf("All audio_pipeline_4ch_ulcnet tests passed\n");
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
