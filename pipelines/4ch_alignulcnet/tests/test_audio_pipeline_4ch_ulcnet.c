/**
 * tests/test_audio_pipeline_4ch_ulcnet.c — Align-ULCNet 4ch pipeline tests.
 *
 * Mirrors test_audio_pipeline_4ch.c's public-API acceptance style. Covers:
 *   1. identity E2E: the 2-hop timing contract (hops 0..1 zero; out[hop p]
 *      equals the beamformed-error accessor value captured at hop p-1)
 *      through the WOLA reconstruction + ULCNet chain in fail-open mode.
 *   2. counting model: infer STEPPED and applied on every emitted frame
 *      except the identity-reprime frames after an alignment generation;
 *      model->reset fired on
 *      every delay change AND on pipeline reset; fail-open on infer()
 *      error (output stays identity, calls continue).
 *  2b. relock on the SAME delay: LOCKED(0) -> forced UNLOCKED ->
 *      LOCKED(0) must still fire model->reset exactly once, on the relock
 *      hop. Applied delay 0 is the one value a value-only `changed` test
 *      cannot tell apart from "nothing accepted yet"; see the test's own
 *      header for the mutation that turns it red.
 *   3. core PreFrame extension: pre.aligned_ref non-NULL, byte-exact against
 *      an independently maintained delayed-far reference, delay lock on a
 *      delayed far, and the abandon_pre token protocol.
 *   4. pool rejection / 8-point descriptor gate / destroy idempotence /
 *      pool reuse, plus a short heap-vs-pool byte-equal run.
 *   5. far-timestamp before acquisition: far-passthrough model,
 *      silence on all mics, one unit impulse in far at a known index -- it
 *      must land in the output at EXACTLY impulse + 2 hops (512 samples):
 *      the wrapper's one-hop far-compensation buffer (matching the beam
 *      WOLA's one-hop lag, so err/far frame pairs are same-hop) plus the
 *      one-hop centered ULCNet chain. Applied delay contributes 0 here
 *      (the shared delay never acquires on silent mics). Goes red by
 *      exactly 256 samples if the far compensation buffer is removed.
 *   6. Production never gates on the delay lock: the 0.5x model's output is
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
 *   9. identity-reprime straddle DERIVATION: a unit impulse in the middle of
 *      the last pre-boundary input hop measures, branch by branch, how many
 *      emitted frames still have a pre-switch hop inside their analysis
 *      window. Measured in a boundary-FREE control run (so the reprime never
 *      measures itself, and the boundary's own buffer clearing cannot hide
 *      the straddling slot) and asserted equal to
 *      AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES; the boundary run then
 *      proves the first model-visible frame after the reprime is marker-free
 *      on BOTH branches. MUTATION: no reprime, or one frame too few, goes
 *      red.
 *  10. identity-reprime BEHAVIOUR: exactly that many identity frames after a
 *      mid-stream delay change and after the FIXED fill completion, infer
 *      frozen for exactly those frames and resuming right after, one
 *      model->reset per generation.
 *  11. fixed deployment contract: an aligned-far descriptor is accepted and
 *      a raw-far descriptor is rejected; no runtime mode setter exists.
 */

#include "audio_pipeline_4ch_ulcnet.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Every test model that actually infers must publish a descriptor: the
 * wrapper refuses infer-without-descriptor, because a shape disagreement
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

/* One hop of the shared synthetic scene every delay-lock test drives: fresh
 * far noise appended to far_hist, and a true_delay-delayed 0.6x echo fanned
 * out to the four mics with a per-channel taper. Only the delay differs
 * between tests. */
static void fill_echo_hop(int frame, int hop, int true_delay,
                          float* far_hist, float* far, float* microphones) {
    for (int i = 0; i < hop; ++i) {
        int64_t t = (int64_t)frame * hop + i;
        float noise = 0.25f * frand();
        float echo;
        far_hist[t] = noise;
        far[i] = noise;
        echo = t >= true_delay ? 0.6f * far_hist[t - true_delay] : 0.0f;
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                echo * (1.0f - 0.02f * ch);
        }
    }
}

/* Frames the 0/2/1 analysis contract emits at hop index `frame`. */
static int frames_in_hop(int frame) {
    return frame == 0 ? 0 : (frame == 1 ? 2 : 1);
}

/* Spend an armed identity-reprime budget over one hop's `frames` emitted
 * frames, and report how many of them the reprime covered. The budget is
 * armed at the TOP of the hop that reports the alignment generation, so the
 * arming hop's own frames are already spending it. */
static int reprime_take(int* armed, int frames) {
    int skipped = *armed < frames ? *armed : frames;
    *armed -= skipped;
    return skipped;
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

    CHECK(cfg.core.sample_rate == 16000 && cfg.core.fft_size == 512,
          "default config is the trained 16 kHz / frame-FFT 512 / hop 256 grid");
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
 * 2. Counting model: policy wiring (continuous stepping, delay-change reset,
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
    long skipped_total = 0;
    int reprime_armed = 0;
    int cur_reprime = 0;
    int prev_reprime = 0;
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
    model.io_descriptor = test_io_descriptor();
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
        int frames = frames_in_hop(frame);
        int scaled;

        if (frame == FRAMES) {
            /* Flip to the error-reporting regime: infer keeps being called
             * but its (deliberately wrong, 3x-scaled) output must be
             * ignored -- fail-open identity. */
            m.fail = 1;
            m.scale = 3.0f;
        }

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "counting-model frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(p, &delay) == 0,
              "delay-state accessor works");

        beam = audio_pipeline_4ch_ulcnet_last_beamformed_error(p);
        memcpy(beam_hist + (size_t)frame * hop, beam,
               (size_t)hop * sizeof(float));

        if (delay.changed) {
            changed_count += 1;
            reprime_armed = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
        }
        scaled = !m.fail;
        if (delay.solid && delay.delay_samples >= 0) locked_frames += 1;
        /* infer runs and applies for every emitted frame, locked or not --
         * the ONLY exception being the identity-reprime frames after an
         * alignment generation, which skip inference entirely (compute drops
         * for those frames; it never doubles anywhere). */
        {
            int skipped_here = reprime_take(&reprime_armed, frames);
            skipped_total += skipped_here;
            expected_infer += frames - skipped_here;
            prev_reprime = cur_reprime;
            cur_reprime = skipped_here > 0;
        }
        CHECK(m.infer_calls == expected_infer,
              "infer stepped on every emitted frame except the identity "
              "reprime, independent of delay lock");

        /* Output relation: both contributing synthesis frames (hop p and
         * hop p-1) must have used the same per-frame gain for the emitted
         * hop to be a pure scaled copy; skip the blend frames. Once
         * m.fail is set the expected gain is identity again, and a reprime
         * frame is an identity frame too, so hops either of whose two
         * contributing frames was one are not comparable at a single gain. */
        if (frame >= 2) {
            float expected_gain = scaled ? m.scale : 1.0f;
            int comparable =
                (scaled == prev_scaled) && !cur_reprime && !prev_reprime &&
                (frame < FRAMES || frame >= FRAMES + 2);
            if (comparable) {
                float e = 0.0f;
                for (int i = 0; i < hop; ++i) {
                    float d = fabsf(
                        out[i] - expected_gain *
                        beam_hist[(size_t)(frame - 1) * hop + i]);
                    if (d > e) e = d;
                }
                if (scaled) {
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
    CHECK(skipped_total ==
              changed_count * AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES &&
          skipped_total > 0,
          "the identity reprime really ran: REPRIME skipped frames per "
          "generation and no others");
    CHECK(scaled_checks > 10,
          "enough locked frames were compared in scaled mode");
    CHECK(max_scaled_err <= 2e-4f,
          "successful frames carry the model's 0.5x output (infer wired in)");
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

/* FIXED does not have an estimator generation, but its aligned-reference
 * seam still changes after the ring fills. Reset the external runtime at
 * that boundary, and skip inference for exactly the straddling frames the
 * identity reprime covers -- no more. */
static int test_fixed_first_alignment_resets_model(void) {
    enum { FIXED_DELAY = 2 * ULCNET_HOP, N = 8 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    CountingModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float microphones[FOUR_AEC_NR_RES_CHANNELS * ULCNET_HOP] = {0};
    float far[ULCNET_HOP] = {0};
    float out[ULCNET_HOP];
    int first_solid = -1;
    int changed = 0;

    cfg.core.delay_mode = AEC_DELAY_FIXED;
    cfg.core.fixed_delay_samples = FIXED_DELAY;
    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create 4ch FIXED transition pipeline");
    if (!p) return 0;

    memset(&m, 0, sizeof(m));
    m.scale = 1.0f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install model for 4ch FIXED transition");

    for (int frame = 0; frame < N; ++frame) {
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "4ch FIXED transition frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(p, &delay) == 0,
              "4ch FIXED transition exposes delay state");
        if (delay.changed) changed++;
        if (delay.solid && first_solid < 0) first_solid = frame;
    }

    CHECK(first_solid == 2,
          "4ch FIXED reports aligned far on the same hop the ring can serve it");
    CHECK(changed == 0,
          "4ch FIXED has no estimator-generated changed event");
    CHECK(m.reset_calls == 1,
          "4ch wrapper resets model once at FIXED raw-to-aligned transition");
    /* N hops emit N frames (0/2/1); exactly the straddling ones are skipped. */
    CHECK(m.infer_calls == N - AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
          "4ch FIXED transition suppresses inference for exactly the "
          "straddling reprime frames and no others");
    audio_pipeline_4ch_ulcnet_destroy(p);
    return 1;
}

/* ============================================================================
 * 2c. FIXED pre-lock seam content: RAW far, byte for byte
 *
 * "UNLOCKED means the content is the RAW far" is the seam's central promise,
 * and under FIXED the unlocked stretch is the whole ring fill -- up to half a
 * second of audio at a realistic loudspeaker delay. Zero-filling it would
 * hand the linear filters and any external far branch a silent reference for
 * that entire window while the microphone already carries the echo.
 *
 * Checked byte-exactly on both sides of the switch and against
 * INDEPENDENTLY maintained references (the caller's own far history), not
 * against anything the pipeline produced:
 *   - every pre-solid hop equals the RAW far hop of the SAME hop,
 *   - the first solid hop onward equals the far hop delayed by exactly
 *     fixed_delay_samples,
 *   - the switch happens on hop ceil(fixed / hop) and nowhere else, and it
 *     is whole-hop: the 320-sample case would splice raw and shifted audio
 *     inside hop 1 under a per-sample rule and go red here.
 * ========================================================================== */

static int fixed_seam_case(int fixed_delay, int frames, int expect_first_solid) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
    FourAecNrRes* core;
    FourAecNrResPreFrame pre;
    float* microphones;
    float* far;
    float* far_hist;
    float* expected;
    int hop;
    int first_solid = -1;
    int raw_hops = 0;
    int aligned_hops = 0;
    int changed = 0;
    char label[192];

    cfg.fft_size = 512;
    cfg.enable_cng = 0;
    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = fixed_delay;
    core = four_aec_nr_res_create(&cfg);
    CHECK(core != NULL, "create FIXED seam core");
    hop = four_aec_nr_res_hop_size(core);

    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)frames * hop, sizeof(float));
    expected = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far && far_hist && expected,
          "allocate FIXED seam buffers");

    for (int frame = 0; frame < frames; ++frame) {
        int64_t hop_start = (int64_t)frame * hop;
        for (int i = 0; i < hop; ++i) {
            float noise = 0.25f * frand();
            far_hist[hop_start + i] = noise;
            far[i] = noise;
        }
        memset(&pre, 0, sizeof(pre));
        CHECK(four_aec_nr_res_process_pre(
                  core, microphones, far, &pre) == FOUR_AEC_NR_RES_OK,
              "FIXED seam hop processes");
        CHECK(pre.aligned_ref != NULL, "FIXED seam exposes aligned_ref");
        CHECK(pre.delay.delay_samples == fixed_delay,
              "FIXED seam reports the configured delay on every hop");
        if (pre.delay.changed) changed += 1;

        if (!pre.delay.solid) {
            /* Pre-lock: byte-equal to the raw far hop we just handed in. */
            if (memcmp(pre.aligned_ref, far, (size_t)hop * sizeof(float))
                != 0) {
                snprintf(label, sizeof(label),
                         "fixed=%d hop %d (pre-lock) serves the RAW far hop "
                         "byte for byte", fixed_delay, frame);
                CHECK(0, label);
            }
            raw_hops += 1;
        } else {
            if (first_solid < 0) first_solid = frame;
            /* Locked: byte-equal to an independently delayed reference. */
            for (int i = 0; i < hop; ++i) {
                int64_t t = hop_start + i - fixed_delay;
                expected[i] = t >= 0 ? far_hist[t] : 0.0f;
            }
            if (memcmp(pre.aligned_ref, expected,
                       (size_t)hop * sizeof(float)) != 0) {
                snprintf(label, sizeof(label),
                         "fixed=%d hop %d (locked) serves the far delayed by "
                         "exactly %d samples", fixed_delay, frame,
                         fixed_delay);
                CHECK(0, label);
            }
            aligned_hops += 1;
        }
        CHECK(four_aec_nr_res_abandon_pre(core, &pre.token) ==
                  FOUR_AEC_NR_RES_OK,
              "FIXED seam pre frame releases");
    }

    snprintf(label, sizeof(label),
             "fixed=%d switches to the shifted far on hop %d (expected %d)",
             fixed_delay, first_solid, expect_first_solid);
    CHECK(first_solid == expect_first_solid, label);
    snprintf(label, sizeof(label),
             "fixed=%d covers both sides of the switch (%d raw hops, %d "
             "aligned hops)", fixed_delay, raw_hops, aligned_hops);
    CHECK(raw_hops == expect_first_solid && aligned_hops > 0, label);
    CHECK(changed == 0, "FIXED never reports an alignment generation event");

    free(expected);
    free(far_hist);
    free(far);
    free(microphones);
    four_aec_nr_res_destroy(core);
    return 1;
}

static int test_fixed_prelock_seam_is_raw_far(void) {
    g_rng = 0x5EED01u;
    /* 8 hops of warm-up: the plain multiple-of-hop case. */
    CHECK(fixed_seam_case(8 * ULCNET_HOP, 12, 8), "fixed=8 hops");
    /* Not a multiple of the hop: a per-sample rule would splice raw and
     * shifted audio inside hop 1. */
    CHECK(fixed_seam_case(320, 8, 2), "fixed=320 samples");
    /* 500 ms at 16 kHz -- the realistic loudspeaker-path case, and the one
     * where zero-fill would have silenced half a second of reference. */
    CHECK(fixed_seam_case(8000, 40, 32), "fixed=500 ms");
    return 1;
}

/* ============================================================================
 * 2b. Relock on the SAME delay must still reset the model
 *
 * The scenario the shared-delay `changed` flag exists for, in the one variant
 * a value-only comparison cannot see:
 *
 *   LOCKED(applied delay 0) -> UNLOCKED for several hops -> LOCKED(applied
 *   delay 0 again, the very same value)
 *
 * The model keeps running across the unlocked stretch on the seam's raw far,
 * so its recurrent state advances over a different delay regime. Unless the
 * relock hop reports `changed`, the wrapper never resets that history before
 * consuming the newly aligned far.
 *
 * The applied delay is deliberately pinned to 0 here. 0 is a legal applied
 * delay AND the value the core's accepted_delay holds when nothing has been
 * accepted yet (init, and every four_aec_nr_res_reset()), so "no alignment
 * yet" and "aligned at 0" are the one pair a value-only `changed` test cannot
 * distinguish -- every other value relocks through a visible 0 -> V step and
 * would pass even with the bug present. TRUE_DELAY 64 minus the estimator's
 * 32-sample headroom lands exactly there; the test asserts delay_samples == 0
 * rather than assuming it, so a headroom change fails loudly instead of
 * quietly turning this into a vacuous pass.
 *
 * MUTATION: restoring the old `state.changed = eligible && estimated !=
 * p->accepted_delay;` in 4aec_nr_res.c's update_shared_delay() drops BOTH the
 * first acquisition and the relock, and this test goes red on the acquisition
 * assertion.
 * ========================================================================== */

static int test_relock_same_delay_resets_model(void) {
    enum { WARM_FRAMES = 60, RELOCK_FRAMES = 60, TRUE_DELAY = 64 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    CountingModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out;
    float* far_hist;
    int total_frames = WARM_FRAMES + RELOCK_FRAMES;
    int hop;
    int acquire_frame = -1;      /* first frame reporting solid, phase A */
    int acquired_delay = -1;     /* what phase A settled on, for phase B */
    int relock_frame = -1;       /* first frame reporting solid, phase B */
    long reset_at_acquire = -1;
    long reset_at_relock = -1;
    long reset_before_relock = -1;
    int unlocked_hops_after_reset = 0;
    long resets_during_unlock = 0;
    long reset_at_pipeline_reset = 0;
    int applied_after_relock = 0;
    int changed_events = 0;
    long reset_before_hop = 0;
    int resets_without_acceptance = 0;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create relock pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install counting model");
    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)total_frames * hop, sizeof(float));
    CHECK(microphones && far && out && far_hist, "allocate relock buffers");

    g_rng = 0x1234567u;
    for (int frame = 0; frame < total_frames; ++frame) {
        if (frame == WARM_FRAMES) {
            /* The forced unlock. This is the only lever the core exposes:
             * four_aec_nr_res_reset() (which audio_pipeline_4ch_ulcnet_reset
             * calls) is the sole caller of delay_aec3_reset(), and the
             * estimator's REFINED-confidence latch cannot otherwise drop
             * once set. It also puts accepted_delay back to 0 -- which is
             * precisely why a same-value relock at 0 is invisible to a
             * value-only test. */
            reset_before_relock = m.reset_calls;
            audio_pipeline_4ch_ulcnet_reset(p);
            reset_at_pipeline_reset = m.reset_calls;
        }

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        reset_before_hop = m.reset_calls;
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "relock frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(p, &delay) == 0,
              "delay-state accessor works");
        /* A model reset flushes recurrent state against the applied delay,
         * so it may only fire on a hop where an alignment generation was
         * actually ACCEPTED -- never on a hop where the estimator is merely
         * confident (confidence >= 1.0 is delay_aec3_is_solid()) but the
         * wrapper's acceptance conditions are not all met yet. Recomputed
         * from the seam's own estimator fields, so it is independent of the
         * `solid`/`changed` flags being tested.
         *
         * MUTATION: adding a raw-confidence arm to this file's own
         * raw->aligned reset gate in audio_pipeline_4ch_ulcnet.c -- e.g.
         * `|| (p->frame_index != 0 && p->last_delay.confidence < 0.5f &&
         * pre.delay.confidence >= 0.5f)` -- fires a reset on the first
         * COARSE hop, long before anything is accepted, and this counter
         * goes non-zero. */
        if (m.reset_calls != reset_before_hop &&
            !(delay.confidence >= 1.0f && delay.estimator_updates >= 3))
            resets_without_acceptance += 1;
        if (delay.changed) changed_events += 1;

        if (frame < WARM_FRAMES) {
            if (delay.solid && acquire_frame < 0) {
                acquire_frame = frame;
                reset_at_acquire = m.reset_calls;
                acquired_delay = delay.delay_samples;
                /* Pinned as a distance from the truth, not as a literal: the
                 * estimator resolves the delay on a 64-sample grid, so the
                 * exact landing value is a property of that quantization
                 * rather than of this pipeline. What this test is about is
                 * that phase B relocks on the SAME value -- see below. */
                {
                    char why[160];
                    snprintf(why, sizeof why,
                             "phase A locks within one grid step of the true "
                             "delay (applied %d, true %d)",
                             acquired_delay, (int)TRUE_DELAY);
                    CHECK(acquired_delay >= 0 &&
                          acquired_delay <= (int)TRUE_DELAY &&
                          (int)TRUE_DELAY - acquired_delay <= 64, why);
                }
                CHECK(delay.changed,
                      "acquisition hop reports a new alignment generation");
            }
        } else {
            if (!delay.solid) {
                /* Still re-acquiring: the model is stepped but nothing may
                 * be applied, and nothing may be flushed either. */
                unlocked_hops_after_reset += 1;
                resets_during_unlock =
                    m.reset_calls - reset_at_pipeline_reset;
            } else if (relock_frame < 0) {
                relock_frame = frame;
                reset_at_relock = m.reset_calls;
                /* The subject of the whole test: the value does not change
                 * across the forced unlock, so a value-only `changed` test
                 * could not tell this relock from "nothing accepted yet" --
                 * which is why `changed` has to carry a generation. */
                {
                    char why[160];
                    snprintf(why, sizeof why,
                             "phase B relocks on the SAME applied delay "
                             "(phase A %d, phase B %d)",
                             acquired_delay, delay.delay_samples);
                    CHECK(delay.delay_samples == acquired_delay, why);
                }
                CHECK(delay.changed,
                      "relock on an unchanged delay value still reports a "
                      "new alignment generation");
            } else {
                applied_after_relock += 1;
            }
        }
    }

    CHECK(resets_without_acceptance == 0,
          "no model->reset fires on a hop where the estimator is confident "
          "but no alignment generation has been accepted");
    CHECK(acquire_frame >= 0, "phase A actually acquired the delay");
    CHECK(relock_frame >= 0, "phase B actually re-acquired the delay");
    CHECK(reset_before_relock == 1,
          "exactly one model->reset up to the pipeline reset (the "
          "acquisition)");
    CHECK(reset_at_pipeline_reset == reset_before_relock + 1,
          "the pipeline reset itself fires model->reset once");
    CHECK(unlocked_hops_after_reset >= 2,
          "the forced unlock lasted several hops (not a vacuous pass)");
    CHECK(resets_during_unlock == 0,
          "no model->reset while merely unlocked -- only a usable alignment "
          "starts a new generation");
    CHECK(reset_at_relock == reset_at_pipeline_reset + 1,
          "the same-delay relock fires model->reset exactly once, on the "
          "relock hop itself (before that hop's frames are applied)");
    CHECK(m.reset_calls == 3,
          "3 model resets total: acquisition, pipeline reset, same-delay "
          "relock");
    CHECK(changed_events == 2,
          "2 alignment generations: acquisition and same-delay relock");
    CHECK(applied_after_relock >= 10,
          "the stream really goes on applying model output after the relock");
    CHECK(reset_at_acquire == 1, "acquisition reset counted once");

    audio_pipeline_4ch_ulcnet_destroy(p);
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
        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
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
         * and for whatever delay the estimator currently applies. The
         * shift is a WHOLE-hop decision -- until the samples already seen
         * cover the applied delay the seam serves the RAW far hop -- so the
         * expectation is chosen per hop, never per sample. */
        CHECK(pre.delay.delay_samples >= 0, "applied delay is non-negative");
        {
            int64_t hop_start = (int64_t)frame * hop;
            int servable = hop_start >= (int64_t)pre.delay.delay_samples;
            for (int i = 0; i < hop; ++i) {
                int64_t t = hop_start + i;
                float expected = servable
                    ? far_hist[t - pre.delay.delay_samples]
                    : far_hist[t];
                if (pre.aligned_ref[i] != expected) {
                    CHECK(0, "aligned_ref content matches the delayed far");
                }
                aligned_checked += 1;
            }
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
    /* The immediately superseded version is spelled out rather than derived.
     * Its byte count is left at the CURRENT figure so the only mismatch is the
     * layout this wrapper descriptor carries. Bump this literal with
     * AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION. */
    stale = req;
    stale.layout_version = 12u;
    CHECK(audio_pipeline_4ch_ulcnet_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "init_ex rejects a descriptor from the superseded layout even "
          "when its byte count exactly covers the current pool");
    CHECK(req.layout_version == AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION &&
          AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION == 13u,
          "the queried descriptor publishes the current carve layout");

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
 * 5. Far-timestamp before acquisition: the model's far branch must
 *    carry the SAME input hop as its error branch. Far-passthrough model,
 *    silence on all mics, a single unit impulse in far at sample T. The
 *    expected output position is derived, not measured:
 *      + 0    applied delay (silent mics -> the shared delay never
 *             acquires; pre.aligned_ref therefore carries raw far)
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

static int test_far_timestamp_before_acquisition(void) {
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
    model.io_descriptor = test_io_descriptor();
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install far-passthrough model");
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
    printf("far timestamp (4ch before acquisition): impulse at far[%d] -> out[%d] "
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
 * 6. Production never gates application on the delay lock: the 0.5x model's
 *    output must appear from the FIRST emitted frame, while delay.solid is
 *    still 0 (the paper contract does not depend on lock).
 * ========================================================================== */

static int test_model_applies_unlocked(void) {
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
    int reprime_armed = 0;
    int cur_reprime = 0;
    int prev_reprime = 0;
    int unsolid_checks = 0;
    int first_solid_frame = -1;
    float max_err = 0.0f;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create unlocked-model pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install 0.5x model");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    beam_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out && far_hist && beam_hist,
          "allocate unlocked-model buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        const float* beam;
        int frames = frames_in_hop(frame);
        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "unlocked-model frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(p, &delay) == 0,
              "unlocked-model delay accessor works");
        if (first_solid_frame < 0 && delay.solid) first_solid_frame = frame;

        beam = audio_pipeline_4ch_ulcnet_last_beamformed_error(p);
        memcpy(beam_hist + (size_t)frame * hop, beam,
               (size_t)hop * sizeof(float));

        /* Stepped on every emitted frame except the identity-reprime frames
         * armed at an alignment generation. */
        if (delay.changed)
            reprime_armed = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
        {
            int skipped_here = reprime_take(&reprime_armed, frames);
            expected_infer += frames - skipped_here;
            prev_reprime = cur_reprime;
            cur_reprime = skipped_here > 0;
        }
        CHECK(m.infer_calls == expected_infer,
              "model steps infer on every emitted frame except the identity "
              "reprime");

        /* Applied from the very first emitted frames: out[p] must be the
         * 0.5x-scaled beam stream on EVERY frame >= 2 (skip delay-change
         * boundary frames where the beam itself is discontinuous, and the
         * reprime frames, which are identity by policy). */
        if (frame >= 2 && !delay.changed && !cur_reprime && !prev_reprime) {
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
          "model application was verified on genuinely unlocked frames");
    (void)first_solid_frame;
    CHECK(max_err <= 2e-4f,
          "production applies the model from the first emitted frame "
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
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out_a;
    float* out_b;
    float* far_hist;
    int poisoned[FRAMES];
    int reprime_left = 0;
    int n_reprime = 0;
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
    model.io_descriptor = test_io_descriptor();
    CHECK(audio_pipeline_4ch_ulcnet_set_model(pa, &model) == 0,
          "install NaN-poisoning model");

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

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pa, microphones, far, 0, out_a) == FOUR_AEC_NR_RES_OK,
              "NaN-guard model frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pb, microphones, far, 0, out_b) == FOUR_AEC_NR_RES_OK,
              "NaN-guard reference frame processes");

        /* Both pipelines cross the same alignment generations, and an
         * identity-reprime frame is an identity frame exactly like a
         * discarded poisoned one -- fold it into the same schedule so the
         * 50%-overlap rule below stays exact. The acquisition lands well
         * inside the steady one-frame-per-hop region. */
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(pa, &delay) == 0,
              "NaN-guard delay accessor works");
        if (delay.changed)
            reprime_left = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
        /* One emitted frame per hop this far into the run, so a hop spends
         * at most one frame of the budget. */
        if (reprime_take(&reprime_left, 1) > 0) {
            poisoned[frame] = 1;
            n_reprime += 1;
        }

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
    CHECK(n_reprime % AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES == 0,
          "NaN-guard: reprime frames accounted for in whole generations");

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
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out_a;
    float* out_b;
    float* far_hist;
    int partial[FRAMES];
    int reprime_left = 0;
    int n_reprime = 0;
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
    model.io_descriptor = test_io_descriptor();
    CHECK(audio_pipeline_4ch_ulcnet_set_model(pa, &model) == 0,
          "install partial-write model");

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

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pa, microphones, far, 0, out_a) == FOUR_AEC_NR_RES_OK,
              "partial-write model frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  pb, microphones, far, 0, out_b) == FOUR_AEC_NR_RES_OK,
              "partial-write reference frame processes");

        /* Same reprime bookkeeping as the NaN test: an identity-reprime
         * frame is an identity frame exactly like a discarded partial one. */
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(pa, &delay) == 0,
              "partial-write delay accessor works");
        if (delay.changed)
            reprime_left = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
        if (reprime_take(&reprime_left, 1) > 0) {
            partial[frame] = 1;
            n_reprime += 1;
        }

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
    CHECK(n_reprime % AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES == 0,
          "partial-write: reprime frames accounted for in whole generations");

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
 * 9. Identity-reprime straddle DERIVATION (marker/timestamp test).
 *
 * Measures the quantity the reprime constant is supposed to be: how many
 * frames emitted at or after an alignment boundary at hop T still have a
 * PRE-boundary hop inside their 512-sample analysis window, on the error
 * branch and on the far branch separately.
 *
 * Instrument: total silence except ONE unit impulse at the MIDDLE of input
 * hop T-1 -- the last input hop that reaches the chain from the pre-boundary
 * side. The middle is the position the sqrt-Hann window weights at 0.707 in
 * BOTH frames covering the hop it lands in (a marker at a hop edge would be
 * multiplied by the window's zero and hide itself). The branches are marked
 * in SEPARATE runs: the error branch through the four mics (far kept silent,
 * so the AEC lanes have no far history to leak and the fixed beamformer is
 * time-invariant), the far branch through the reference.
 *
 * The count is derived from a CONTROL run with NO boundary at all (FIXED
 * delay 0: the shared ring can serve from hop #0, so the wrapper never sees
 * a raw->aligned transition and never arms a reprime -- asserted, not
 * assumed). The reprime logic therefore takes no part in measuring its own
 * length, and the measurement is pure framing/latency: which emitted frames
 * still reach back across hop T. This also matters for the far branch: at a
 * real boundary the wrapper CLEARS the saved far hop (and the beam OLA), so
 * the boundary run itself could not see that slot's pre-switch content --
 * clearing replaces what the straddling slot holds, it does not move the
 * slot.
 *
 * Both branches come out at 2 here versus 1 for the mono wrapper, and for a
 * measured reason: this wrapper pushes both branches one hop behind the
 * input (beam WOLA lag + the matching one-hop far compensation), so the slot
 * pushed at hop T still belongs to input hop T-1.
 *
 * The boundary run (FIXED delay = T hops, so the raw->aligned switch lands
 * exactly on hop T) then checks the implementation: exactly REPRIME frames
 * skipped, at hops T..T+REPRIME-1, and the first frame the model DOES see
 * afterwards carries no pre-switch sample on either branch.
 *
 * MUTATION: dropping the reprime counter (stepping always) makes the first
 * model-visible frame after the boundary a straddling one and the
 * marker-free assertion goes red; arming it one frame too short does the
 * same.
 * ========================================================================== */

enum {
    RP_T        = 8,             /* boundary hop under test                */
    RP_MARK_HOP = RP_T - 1,      /* last input hop before the boundary     */
    RP_MARK_POS = ULCNET_HOP / 2,
    RP_RUN      = 14,
    RP_WINDOW   = 4,             /* hops examined from RP_T on             */
    RP_MAXCALL  = 32
};
/* A windowed unit impulse reaches every bin at |w| = 0.707; what a
 * marker-free frame can still carry (mic-HPF ring plus the fixed
 * beamformer's own sinc tails on the error branch, plain zero on the far
 * branch) is orders below. The margin is asserted, so this floor can never
 * silently become a classifier of noise. */
#define RP_FLOOR 0.05f

typedef struct ProbeModel {
    int   hop;                   /* current hop; set by the test           */
    int   calls;
    int   call_hop[RP_MAXCALL];
    float err_peak[RP_MAXCALL];
    float far_peak[RP_MAXCALL];
    int   resets;
    int   overflow;
} ProbeModel;

static float spec_peak(const float* re, const float* im) {
    float m = 0.0f;
    for (int k = 0; k < ULCNET_BINS; ++k) {
        float a = fabsf(re[k]);
        if (a > m) m = a;
        a = fabsf(im[k]);
        if (a > m) m = a;
    }
    return m;
}

static int probe_infer(
    void* user,
    const float* err_re, const float* err_im,
    const float* far_re, const float* far_im,
    float* out_re, float* out_im) {
    ProbeModel* st = (ProbeModel*)user;
    if (st->calls < RP_MAXCALL) {
        st->call_hop[st->calls] = st->hop;
        st->err_peak[st->calls] = spec_peak(err_re, err_im);
        st->far_peak[st->calls] = spec_peak(far_re, far_im);
        st->calls += 1;
    } else {
        st->overflow = 1;
    }
    memcpy(out_re, err_re, ULCNET_BINS * sizeof(float));
    memcpy(out_im, err_im, ULCNET_BINS * sizeof(float));
    return 0;
}

static void probe_reset(void* user) { ((ProbeModel*)user)->resets += 1; }

/* One probe run. fixed_delay == 0 is the boundary-free control;
 * fixed_delay == RP_T*ULCNET_HOP puts the FIXED raw->aligned switch on hop
 * RP_T. mark_far selects which branch carries the impulse. */
static int reprime_probe_run(int fixed_delay, int mark_far, ProbeModel* st) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    UlcnetModel model;
    float microphones[FOUR_AEC_NR_RES_CHANNELS * ULCNET_HOP];
    float far[ULCNET_HOP];
    float out[ULCNET_HOP];

    memset(st, 0, sizeof(*st));
    /* Frozen fixed beamformer (mu 0 and vad_external 1 both freeze GSC
     * adaptation) so the beam path is time-invariant and the marker cannot
     * be smeared by an adapting filter. */
    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.0f;
    cfg.core.enable_cng = 0;
    cfg.core.delay_mode = AEC_DELAY_FIXED;
    cfg.core.fixed_delay_samples = fixed_delay;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    if (!p) return 0;
    memset(&model, 0, sizeof(model));
    model.user = st;
    model.infer = probe_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = probe_reset;
    if (audio_pipeline_4ch_ulcnet_set_model(p, &model) != 0) {
        audio_pipeline_4ch_ulcnet_destroy(p);
        return 0;
    }

    for (int frame = 0; frame < RP_RUN; ++frame) {
        memset(microphones, 0, sizeof(microphones));
        memset(far, 0, sizeof(far));
        if (frame == RP_MARK_HOP) {
            if (mark_far) {
                far[RP_MARK_POS] = 1.0f;
            } else {
                for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                    microphones[RP_MARK_POS * FOUR_AEC_NR_RES_CHANNELS + ch] =
                        1.0f;
            }
        }
        st->hop = frame;
        if (audio_pipeline_4ch_ulcnet_process_with_activity(
                p, microphones, far, 1, out) != FOUR_AEC_NR_RES_OK) {
            audio_pipeline_4ch_ulcnet_destroy(p);
            return 0;
        }
    }
    audio_pipeline_4ch_ulcnet_destroy(p);
    return 1;
}

/* Frames emitted at hops [RP_T, RP_T+RP_WINDOW) that still carry the marker,
 * plus the classification margin (smallest marked peak / largest unmarked
 * peak inside that window). */
static int reprime_straddle_count(const ProbeModel* st, int far_branch,
                                  float* dirty_min, float* clean_max) {
    int n = 0;
    *dirty_min = 1e30f;
    *clean_max = 0.0f;
    for (int c = 0; c < st->calls; ++c) {
        float peak = far_branch ? st->far_peak[c] : st->err_peak[c];
        if (st->call_hop[c] < RP_T || st->call_hop[c] >= RP_T + RP_WINDOW)
            continue;
        if (peak > RP_FLOOR) {
            n += 1;
            if (peak < *dirty_min) *dirty_min = peak;
        } else if (peak > *clean_max) {
            *clean_max = peak;
        }
    }
    return n;
}

static int test_reprime_straddle_derivation(void) {
    ProbeModel err_ctl, far_ctl, err_bnd, far_bnd;
    float dmin, cmax;
    int straddle_err, straddle_far;
    int expected_frames = 0;
    int first_visible_err = -1, first_visible_far = -1;
    float first_visible_err_peak = -1.0f, first_visible_far_peak = -1.0f;
    int stepped_inside_reprime = 0;

    for (int frame = 0; frame < RP_RUN; ++frame)
        expected_frames += frames_in_hop(frame);

    CHECK(reprime_probe_run(0, 0, &err_ctl), "run error-branch control probe");
    CHECK(reprime_probe_run(0, 1, &far_ctl), "run far-branch control probe");
    CHECK(reprime_probe_run(RP_T * ULCNET_HOP, 0, &err_bnd),
          "run error-branch boundary probe");
    CHECK(reprime_probe_run(RP_T * ULCNET_HOP, 1, &far_bnd),
          "run far-branch boundary probe");

    /* The control runs really are boundary-free and reprime-free. */
    CHECK(err_ctl.resets == 0 && far_ctl.resets == 0 &&
          err_ctl.calls == expected_frames &&
          far_ctl.calls == expected_frames &&
          !err_ctl.overflow && !far_ctl.overflow,
          "derivation control (FIXED delay 0) has no alignment boundary: no "
          "model reset and every emitted frame stepped");

    straddle_err = reprime_straddle_count(&err_ctl, 0, &dmin, &cmax);
    printf("reprime derivation (4ch, ERROR branch): %d straddling frames at hops "
           "%d..%d; marked peak >= %.4f, unmarked peak <= %.3e\n",
           straddle_err, RP_T, RP_T + RP_WINDOW - 1, (double)dmin, (double)cmax);
    CHECK(straddle_err == AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
          "4ch ERROR branch: the emitted frames that still contain pre-switch "
          "samples number exactly AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES");
    CHECK(straddle_err > 0 && dmin > 20.0f * cmax && dmin > 0.2f,
          "4ch ERROR branch marker is unambiguous (marked peak far above the "
          "unmarked residue)");

    straddle_far = reprime_straddle_count(&far_ctl, 1, &dmin, &cmax);
    printf("reprime derivation (4ch, FAR branch):   %d straddling frames at hops "
           "%d..%d; marked peak >= %.4f, unmarked peak <= %.3e\n",
           straddle_far, RP_T, RP_T + RP_WINDOW - 1, (double)dmin, (double)cmax);
    CHECK(straddle_far == AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
          "4ch FAR branch: the emitted frames that still contain pre-switch "
          "samples number exactly AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES");
    CHECK(straddle_far > 0 && dmin > 20.0f * cmax && dmin > 0.2f,
          "4ch FAR branch marker is unambiguous (marked peak far above the "
          "unmarked residue)");

    /* The boundary runs: the wrapper skips exactly those frames. */
    CHECK(err_bnd.resets == 1 && far_bnd.resets == 1,
          "boundary run: the FIXED raw->aligned switch fired model->reset once");

    for (int c = 0; c < err_bnd.calls; ++c) {
        if (err_bnd.call_hop[c] >= RP_T &&
            err_bnd.call_hop[c] <
                RP_T + AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES)
            stepped_inside_reprime += 1;
        if (err_bnd.call_hop[c] >= RP_T && first_visible_err < 0) {
            first_visible_err = err_bnd.call_hop[c];
            first_visible_err_peak = err_bnd.err_peak[c];
        }
    }
    for (int c = 0; c < far_bnd.calls; ++c) {
        if (far_bnd.call_hop[c] >= RP_T &&
            far_bnd.call_hop[c] <
                RP_T + AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES)
            stepped_inside_reprime += 1;
        if (far_bnd.call_hop[c] >= RP_T && first_visible_far < 0) {
            first_visible_far = far_bnd.call_hop[c];
            first_visible_far_peak = far_bnd.far_peak[c];
        }
    }
    printf("reprime resume (4ch): first model-visible frame after the boundary is "
           "hop %d; err peak %.3e, far peak %.3e (floor %.3f)\n",
           first_visible_err, (double)first_visible_err_peak,
           (double)first_visible_far_peak, (double)RP_FLOOR);
    /* The primary claim, checked before the bookkeeping so a mutation cannot
     * be masked by a count assertion firing first. */
    CHECK(first_visible_err_peak >= 0.0f &&
          first_visible_err_peak <= RP_FLOOR &&
          first_visible_far_peak >= 0.0f &&
          first_visible_far_peak <= RP_FLOOR,
          "the first model-visible frame after the reprime contains NO "
          "pre-switch sample on either branch");
    CHECK(stepped_inside_reprime == 0,
          "boundary run: the model is not stepped on any reprime hop");
    CHECK(first_visible_err ==
              RP_T + AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES &&
          first_visible_far ==
              RP_T + AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
          "boundary run: stepping resumes exactly REPRIME frames after the "
          "boundary hop");
    CHECK(err_bnd.calls ==
              expected_frames - AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES &&
          far_bnd.calls ==
              expected_frames - AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
          "boundary run: exactly REPRIME emitted frames skipped inference");
    return 1;
}

/* ============================================================================
 * 10. Identity-reprime BEHAVIOUR: exactly REPRIME identity frames per
 *     alignment generation, on both event shapes the wrapper recognises -- a
 *     mid-stream delay change and the FIXED ring-fill completion. Per hop the
 *     test predicts the infer delta from its OWN copy of the policy (arm
 *     REPRIME at the boundary hop, consume one per emitted frame) and
 *     compares against the counting model, so both a missing and an over-long
 *     reprime fail. reset stays exactly one per generation.
 * ========================================================================== */

static int test_reprime_behavior(void) {
    enum { FRAMES = 120, TRUE_DELAY = 400, FIXED_HOPS = 8 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    CountingModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out;
    float* far_hist;
    long changed_count = 0;
    long skipped_total = 0;
    int armed = 0;
    int boundary_frame = -1;
    int resumed_frame = -1;
    int hop;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;

    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create reprime-behaviour pipeline");
    hop = audio_pipeline_4ch_ulcnet_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &model) == 0,
          "install counting model (reprime behaviour)");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out && far_hist,
          "allocate reprime-behaviour buffers");

    g_rng = 0x1234567u;
    for (int frame = 0; frame < FRAMES; ++frame) {
        int frames = frames_in_hop(frame);
        long calls_before = m.infer_calls;
        int skipped_here;
        long delta;

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "reprime-behaviour frame processes");
        CHECK(audio_pipeline_4ch_ulcnet_last_delay(p, &delay) == 0,
              "reprime-behaviour delay accessor works");

        if (delay.changed) {
            armed = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
            changed_count += 1;
            if (boundary_frame < 0) boundary_frame = frame;
        }
        skipped_here = reprime_take(&armed, frames);
        skipped_total += skipped_here;
        delta = m.infer_calls - calls_before;
        CHECK(delta == frames - skipped_here,
              "infer is stepped once per emitted frame except during the "
              "identity reprime");
        if (boundary_frame >= 0 && resumed_frame < 0 && delta > 0 &&
            frame > boundary_frame)
            resumed_frame = frame;
    }

    CHECK(changed_count >= 1,
          "reprime behaviour: the run really crossed an alignment generation");
    CHECK(skipped_total ==
              changed_count * AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
          "reprime behaviour: REPRIME frames skipped per generation and no "
          "others -- compute DROPS on those frames, it never doubles");
    CHECK(resumed_frame ==
              boundary_frame + AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
          "reprime behaviour: stepping stops on the boundary hop and resumes "
          "REPRIME frames later");
    CHECK(m.reset_calls == changed_count,
          "reprime behaviour: still exactly one model->reset per generation");
    audio_pipeline_4ch_ulcnet_destroy(p);
    free(far_hist);
    free(out);
    free(far);
    free(microphones);

    /* ---- FIXED ring-fill completion ---- */
    {
        AudioPipeline4ChConfig fcfg =
            audio_pipeline_4ch_ulcnet_default_config();
        AudioPipeline4ChUlcnet* fp;
        CountingModel fm;
        UlcnetModel fmodel;
        float mics[FOUR_AEC_NR_RES_CHANNELS * ULCNET_HOP];
        float ffar[ULCNET_HOP];
        float fout[ULCNET_HOP];
        int fixed_boundary = -1;
        int calls_at_boundary = -1;
        int calls_after_reprime = -1;
        long fskipped = 0;
        int farmed = 0;

        fcfg.core.delay_mode = AEC_DELAY_FIXED;
        fcfg.core.fixed_delay_samples = FIXED_HOPS * ULCNET_HOP;
        fcfg.core.enable_cng = 0;
        fp = audio_pipeline_4ch_ulcnet_create(&fcfg);
        CHECK(fp != NULL, "create FIXED reprime-behaviour pipeline");
        memset(&fm, 0, sizeof(fm));
        fm.scale = 1.0f;
        memset(&fmodel, 0, sizeof(fmodel));
        fmodel.user = &fm;
        fmodel.infer = counting_infer;
        fmodel.io_descriptor = test_io_descriptor();
        fmodel.reset = counting_reset;
        CHECK(audio_pipeline_4ch_ulcnet_set_model(fp, &fmodel) == 0,
              "install counting model (FIXED reprime behaviour)");

        memset(mics, 0, sizeof(mics));
        for (int frame = 0; frame < 20; ++frame) {
            FourAecNrResDelayState fdelay;
            int frames = frames_in_hop(frame);
            long calls_before = fm.infer_calls;
            int solid_before;
            int skipped_here;
            long delta;

            CHECK(audio_pipeline_4ch_ulcnet_last_delay(fp, &fdelay) == 0,
                  "FIXED reprime: delay accessor works before the hop");
            solid_before = fdelay.solid;
            for (int i = 0; i < ULCNET_HOP; ++i) ffar[i] = 0.25f * frand();
            CHECK(audio_pipeline_4ch_ulcnet_process_with_activity(
                      fp, mics, ffar, 0, fout) == FOUR_AEC_NR_RES_OK,
                  "FIXED reprime frame processes");
            CHECK(audio_pipeline_4ch_ulcnet_last_delay(fp, &fdelay) == 0,
                  "FIXED reprime: delay accessor works after the hop");
            if (frame != 0 && !solid_before && fdelay.solid) {
                farmed = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
                fixed_boundary = frame;
            }
            skipped_here = reprime_take(&farmed, frames);
            fskipped += skipped_here;
            delta = fm.infer_calls - calls_before;
            CHECK(delta == frames - skipped_here,
                  "FIXED reprime: infer stepped once per emitted frame except "
                  "during the reprime");
            if (fixed_boundary >= 0 && frame == fixed_boundary)
                calls_at_boundary = (int)delta;
            if (fixed_boundary >= 0 &&
                frame == fixed_boundary +
                             AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES)
                calls_after_reprime = (int)delta;
        }

        CHECK(fixed_boundary == FIXED_HOPS,
              "FIXED reprime: the raw->aligned switch is on the expected hop");
        CHECK(fskipped == AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES,
              "FIXED reprime: exactly REPRIME frames skipped at the fill "
              "completion");
        CHECK(calls_at_boundary == 0 && calls_after_reprime == 1,
              "FIXED reprime: the boundary hop steps the model 0 times and "
              "stepping is back to one per frame after the reprime");
        CHECK(fm.reset_calls == 1,
              "FIXED reprime: exactly one model->reset");
        audio_pipeline_4ch_ulcnet_destroy(fp);
    }
    return 1;
}

/* ============================================================================
 * 11. Fixed deployment contract: production accepts only the aligned-far
 * descriptor. The raw enumerator remains useful for negative ABI tests and
 * the offline sweep, but there is no runtime far-mode setter.
 * ========================================================================== */
static int gate_infer_identity(
    void* user,
    const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
    const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
    float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    (void)user; (void)far_re; (void)far_im;
    memcpy(out_re, err_re, sizeof(float) * ULCNET_BINS);
    memcpy(out_im, err_im, sizeof(float) * ULCNET_BINS);
    return 0;
}

static int test_aligned_descriptor_gate(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    AudioPipeline4ChUlcnet* p;
    UlcnetModelIoDescriptor raw_desc;
    UlcnetModelIoDescriptor aligned_desc;
    UlcnetModel raw_model;
    UlcnetModel aligned_model;
    UlcnetModel undescribed_model;

    CHECK(ulcnet_model_io_descriptor_default(8, &raw_desc) == 0 &&
          ulcnet_model_io_descriptor_default(8, &aligned_desc) == 0,
          "build model-I/O descriptors");
    raw_desc.far_input_mode = ULCNET_FAR_RAW;
    CHECK(aligned_desc.far_input_mode == ULCNET_FAR_ALIGNED,
          "descriptor_default publishes fixed aligned far");

    memset(&undescribed_model, 0, sizeof(undescribed_model));
    undescribed_model.infer = gate_infer_identity;
    undescribed_model.io_descriptor = test_io_descriptor();
    raw_model = undescribed_model;
    raw_model.io_descriptor = &raw_desc;
    aligned_model = undescribed_model;
    aligned_model.io_descriptor = &aligned_desc;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.3f;
    cfg.gsc_mu = 0.02f;
    cfg.core.enable_cng = 0;
    p = audio_pipeline_4ch_ulcnet_create(&cfg);
    CHECK(p != NULL, "create aligned-descriptor gate pipeline");

    /* Baseline: an undescribed model publishes no contract, so the
     * rejections below cannot be passing for want of a model. */
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &undescribed_model) == 0,
          "a model with io_descriptor == NULL remains supported");

    /* Production accepts only an aligned-far descriptor. */
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &raw_model) != 0,
          "raw descriptor is rejected");
    CHECK(audio_pipeline_4ch_ulcnet_set_model(p, &aligned_model) == 0,
          "aligned descriptor is accepted");

    audio_pipeline_4ch_ulcnet_destroy(p);
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
        /* The pre-only contract: every post-only field must keep the value
         * the default config returns, and is REJECTED otherwise. With
         * enable_post = 0 the core builds no denoiser, no suppressor, no
         * comfort noise and no post iFFT, so any of these having been
         * accepted would have meant a caller configuring something that
         * cannot exist. One row per field, each proved to fire. */
        invalid = audio_pipeline_4ch_ulcnet_default_config();
        CHECK(invalid.core.enable_post == 0 && invalid.core.enable_cng == 0,
              "the default config itself states the pre-only profile");
        {
            AudioPipeline4ChUlcnet* ok_default =
                audio_pipeline_4ch_ulcnet_create(&invalid);
            CHECK(ok_default != NULL,
                  "the unmodified default config is accepted");
            audio_pipeline_4ch_ulcnet_destroy(ok_default);
        }
#define REJECT_POST_ONLY(mutate, what)                                       \
        do {                                                                 \
            AudioPipeline4ChConfig bad =                                     \
                audio_pipeline_4ch_ulcnet_default_config();                  \
            mutate;                                                          \
            CHECK(audio_pipeline_4ch_ulcnet_create(&bad) == NULL,            \
                  "post-only field rejected: " what);                        \
        } while (0)
        REJECT_POST_ONLY(bad.core.enable_post = 1, "enable_post");
        REJECT_POST_ONLY(bad.core.enable_cng = 1, "enable_cng");
        REJECT_POST_ONLY(bad.core.legacy_amin = 1, "legacy_amin");
        REJECT_POST_ONLY(bad.core.nr_mode = MMSE_LSA_NR_AGGRESSIVE,
                         "nr_mode (BALANCED is a required sentinel)");
        REJECT_POST_ONLY(bad.auto_vad_threshold_dbfs = -40.0f,
                         "auto_vad_threshold_dbfs");
        REJECT_POST_ONLY(bad.auto_vad_snr_ratio = 4.0f, "auto_vad_snr_ratio");
        REJECT_POST_ONLY(bad.auto_vad_hangover_frames = 4,
                         "auto_vad_hangover_frames");
#undef REJECT_POST_ONLY
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
          "counting model policy (stepping, resets, fail-open)");
    CHECK(test_fixed_first_alignment_resets_model(),
          "FIXED first aligned far resets the model without gating inference");
    CHECK(test_fixed_prelock_seam_is_raw_far(),
          "FIXED pre-lock seam serves the raw far byte for byte");
    CHECK(test_relock_same_delay_resets_model(),
          "relock on the same delay still resets the model (ALIGNED)");
    CHECK(test_core_aligned_ref_and_abandon(),
          "core PreFrame aligned_ref + abandon protocol");
    CHECK(test_pool_and_descriptor_gate(),
          "pool rejection / descriptor gate / destroy idempotence");
    CHECK(test_far_timestamp_before_acquisition(),
          "far timestamp: err/far frame pairs are same-hop before acquisition");
    CHECK(test_model_applies_unlocked(),
          "model applies without a delay lock");
    CHECK(test_nan_guard(),
          "NaN guard: non-finite model output never reaches the WOLA");
    CHECK(test_partial_write_guard(),
          "full-write contract: partial rc==0 frames are discarded");
    CHECK(test_reprime_straddle_derivation(),
          "identity-reprime length derived from the straddling frames");
    CHECK(test_reprime_behavior(),
          "identity reprime: exactly REPRIME identity frames per generation");
    CHECK(test_aligned_descriptor_gate(),
          "production accepts only aligned-far descriptors");
    printf("All audio_pipeline_4ch_ulcnet tests passed\n");
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
