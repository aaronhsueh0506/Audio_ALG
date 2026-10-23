/**
 * Composition tests of the 4-channel linear AEC + beamformer unit (audio_pipeline_4ch.h,
 * core.enable_post = 0) feeding the Align-ULCNet stage (samples/ulcnet_nn_stage.h), wired as
 * 4ch_alignulcnet/main.c wires them: one-hop STFT/WOLA latency, delay-event identity reprime on
 * the real alignment state, the model-callback policy on real framing, and recovery by resetting
 * unit and stage together. A contract test, not a DSP-quality test. The stage alone (synthetic
 * alignment events) is tested in samples/tests/test_ulcnet_nn_stage.c; the unit's split calls
 * are tested in test_audio_pipeline_4ch.c. What needs the real lanes (FIXED and MATCHED delay
 * states, the framing straddle derivation, acquisition of a known delay) stays here. Each test
 * banner states its case:
 *   0. grid of the example unit config                1. identity E2E and the 1-hop timing contract
 *   2. counting model (stepping, reset, fail-open)    2b. same-delay relock resets the model
 *   2c. FIXED pre-lock seam is the RAW far            4. caller pool vs heap construction
 *   5/5b. far / error-branch impulse timestamps       6. production never gates on the delay lock
 *   7. NaN guard   8. full-write contract             9. identity-reprime straddle derivation
 *   10. identity-reprime behaviour                    11. fixed deployment contract
 *   12. phased hop                                    13. recovery by paired reset equals a fresh pair
 * Build and run: `make test_4ch_ulcnet` in 4ch_aec_bf_nr_res/.
 */

#include "audio_pipeline_4ch.h"
#include "ulcnet_nn_stage.h"
#include "utility/mem_align.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The unit on the compiled grid, linear context only: what 4ch_alignulcnet/main.c builds. */
static AudioPipeline4ChConfig ulcnet_example_unit_config(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(ULCNET_SR);
    cfg.core.fft_size = ULCNET_N_FFT;
    cfg.core.enable_post = 0;
    return cfg;
}

/* ---------------------------------------------------------------------------------------
 * The composition under test: the unit plus the stage (err_analysis 0: the beamformed
 * spectrum is the error frame), as main.c builds them. A small test-local shim gives the
 * pieces the shape the tests below drive. The unit is always called through its split
 * (front + beam) or one-call forms with the activity flag in both slots and no mask.
 * --------------------------------------------------------------------------------------- */

typedef struct Ul4 {
    AudioPipeline4Ch* unit;
    UlcnetNnStage* stage;
    void* stage_mem;
    FourAecNrResDelayState last_delay;   /* alignment of the most recent successful hop */
} Ul4;

/* The stage takes ULCNET_BINS bins per hop, so a unit of another grid cannot feed it. */
static int unit_matches_stage(const AudioPipeline4Ch* unit) {
    return audio_pipeline_4ch_hop_size(unit) == ULCNET_HOP &&
           audio_pipeline_4ch_n_freqs(unit) == ULCNET_BINS &&
           audio_pipeline_4ch_fft_size(unit) == ULCNET_N_FFT;
}

static Ul4* ul4_wrap(AudioPipeline4Ch* unit, void* stage_mem, size_t stage_bytes) {
    Ul4* p = (Ul4*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->unit = unit;
    p->stage_mem = stage_mem;
    if (unit && stage_mem && unit_matches_stage(unit))
        p->stage = ulcnet_nn_stage_init(stage_mem, stage_bytes, 0);
    if (!p->stage) {
        if (unit) audio_pipeline_4ch_destroy(unit);
        free(stage_mem);
        free(p);
        return NULL;
    }
    return p;
}

/* A fresh instance from separate heap allocations; the stage block starts stale. */
static Ul4* ul4_create(const AudioPipeline4ChConfig* cfg) {
    size_t stage_bytes = ulcnet_nn_stage_mem_size(0);
    void* stage_mem = NULL;
    if (!cfg || stage_bytes == 0 || posix_memalign(&stage_mem, 16, stage_bytes) != 0)
        return NULL;
    memset(stage_mem, 0x5A, stage_bytes);
    return ul4_wrap(audio_pipeline_4ch_create(cfg), stage_mem, stage_bytes);
}

static void ul4_destroy(Ul4* p) {
    if (!p) return;
    audio_pipeline_4ch_destroy(p->unit);
    free(p->stage_mem);
    free(p);
}

static int ul4_set_model(Ul4* p, const UlcnetModel* model) {
    return p ? ulcnet_nn_stage_set_model(p->stage, model) : -1;
}

static int ul4_hop_size(const Ul4* p) { return p ? audio_pipeline_4ch_hop_size(p->unit) : -1; }
static int ul4_fft_size(const Ul4* p) { return p ? audio_pipeline_4ch_fft_size(p->unit) : -1; }
static int ul4_n_freqs(const Ul4* p) { return p ? audio_pipeline_4ch_n_freqs(p->unit) : -1; }
static int ul4_sample_rate(const Ul4* p) { return p ? audio_pipeline_4ch_sample_rate(p->unit) : -1; }

/* Unit and stage together: the recovery after any refused call, and the stream restart. */
static void ul4_reset(Ul4* p) {
    if (!p) return;
    audio_pipeline_4ch_reset(p->unit);
    ulcnet_nn_stage_reset(p->stage);
    memset(&p->last_delay, 0, sizeof(p->last_delay));
}

static int ul4_last_delay(const Ul4* p, FourAecNrResDelayState* out) {
    if (!p || !out) return -1;
    *out = p->last_delay;
    return 0;
}

static UlcnetNnDelay delay_events(const FourAecNrResDelayState* d) {
    UlcnetNnDelay e;
    e.changed = d->changed;
    e.solid = d->solid;
    return e;
}

/* Unit (split calls), then the stage's begin with the beamformed spectrum, the raw far and the
 * alignment events of the front. A refused unit hop takes both down. */
static int ul4_process_begin(Ul4* p, const float* mic, const float* far, int vad) {
    FourAecFront front;
    AudioPipeline4ChBeam beam;
    int rc;
    if (!p || !mic || !far) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    rc = audio_pipeline_4ch_process_front(p->unit, mic, far, &front);
    if (rc == FOUR_AEC_NR_RES_OK)
        rc = audio_pipeline_4ch_process_beam(p->unit, &front, vad, vad, NULL, &beam, NULL);
    if (rc != FOUR_AEC_NR_RES_OK) {
        if (rc == FOUR_AEC_NR_RES_DSP_ERROR) ul4_reset(p);
        return rc;
    }
    rc = ulcnet_nn_stage_begin_complex(p->stage, beam.spectrum, far, delay_events(&front.delay));
    if (rc == ULCNET_NN_STAGE_OK) p->last_delay = front.delay;
    return rc;
}

static int ul4_nn_begin(Ul4* p, UlcnetNnJob* job) {
    return p ? ulcnet_nn_stage_nn_begin(p->stage, job) : -1;
}
static int ul4_nn_end(Ul4* p, int model_rc) {
    return p ? ulcnet_nn_stage_nn_end(p->stage, model_rc) : -1;
}
static int ul4_process_end(Ul4* p, float* out) {
    return p ? ulcnet_nn_stage_end(p->stage, out) : -1;
}

/* The whole hop with the installed model (identity when unset), the unit's one-call form. */
static int ul4_process_with_activity(Ul4* p, const float* mic, const float* far, int vad,
                                     float* out) {
    AudioPipeline4ChBeam beam;
    AudioPipeline4ChFrameInfo info;
    int rc;
    if (!p || !mic || !far || !out) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    rc = audio_pipeline_4ch_process_with_activity(p->unit, mic, far, vad, vad, NULL, &beam, &info);
    if (rc != FOUR_AEC_NR_RES_OK) {
        if (rc == FOUR_AEC_NR_RES_DSP_ERROR) ul4_reset(p);
        return rc;
    }
    rc = ulcnet_nn_stage_begin_complex(p->stage, beam.spectrum, far, delay_events(&info.delay));
    if (rc != ULCNET_NN_STAGE_OK) return rc;
    p->last_delay = info.delay;
    return ulcnet_nn_stage_process(p->stage, out);
}

/* Every test model that actually infers must publish a descriptor: the
 * stage refuses infer-without-descriptor, because a shape disagreement
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


/* DelayAec searches a 16 kHz-equivalent sidechain and scales the selected
 * delay back to native samples. Echo-delay stimuli therefore scale by
 * exactly three on the 48 kHz product grid. */
#define DELAY_RATE_FACTOR (ULCNET_SR / 16000)

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
 * 1. Identity E2E: direct BF -> model -> synthesis chain and the 1-hop latency contract.
 *    The reference is synthesised by the TEST from the spectra the pipeline hands the model
 *    (a capture model that also returns the identity) through its own AcIstft: bit-identical
 *    output proves the chain adds nothing between the model's frame and the WOLA, and the
 *    call count proves one inference per hop from hop #0. FIXED delay 0 keeps the run
 *    boundary-free (asserted), so no reprime frame bypasses the model.
 * ========================================================================== */

typedef struct CaptureModel {
    float re[ULCNET_BINS];
    float im[ULCNET_BINS];
    long calls;
    long resets;
} CaptureModel;

static int capture_infer(
    void* user,
    const float* err_re, const float* err_im,
    const float* far_re, const float* far_im,
    float* out_re, float* out_im) {
    CaptureModel* m = (CaptureModel*)user;
    (void)far_re;
    (void)far_im;
    memcpy(m->re, err_re, ULCNET_BINS * sizeof(float));
    memcpy(m->im, err_im, ULCNET_BINS * sizeof(float));
    memcpy(out_re, err_re, ULCNET_BINS * sizeof(float));
    memcpy(out_im, err_im, ULCNET_BINS * sizeof(float));
    m->calls += 1;
    return 0;
}

static void capture_reset(void* user) { ((CaptureModel*)user)->resets += 1; }

static int test_identity_e2e(void) {
    enum { FRAMES = 60 };
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
    CaptureModel m;
    UlcnetModel model;
    AcIstft* ref_synth;
    void* ref_mem = NULL;
    float* microphones;
    float* far;
    float* out;
    float* ref;
    float max_ref = 0.0f;
    float max_preamble = 0.0f;
    int identical_hops = 0;
    int hop;

    CHECK(cfg.core.sample_rate == ULCNET_SR &&
          cfg.core.fft_size == ULCNET_N_FFT,
          "default config matches the compiled ULCNet deployment grid");
    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.4f;
    cfg.bf_mu = 0.02f;
    cfg.core.delay_mode = AEC_DELAY_FIXED;
    cfg.core.fixed_delay_samples = 0;

    p = ul4_create(&cfg);
    CHECK(p != NULL, "create ULCNet 4ch pipeline");
    hop = ul4_hop_size(p);
    CHECK(hop == ULCNET_HOP, "hop matches the ULCNet compile-time hop");
    CHECK(ul4_fft_size(p) == ULCNET_N_FFT &&
          ul4_n_freqs(p) == ULCNET_BINS &&
          ul4_sample_rate(p) == ULCNET_SR,
          "runtime dimensions match the compiled ULCNet deployment grid");

    memset(&m, 0, sizeof(m));
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = capture_infer;
    model.reset = capture_reset;
    model.io_descriptor = test_io_descriptor();
    CHECK(ul4_set_model(p, &model) == 0,
          "install capture model");

    /* The test's own synthesis: its own handle and window table. */
    {
        AcStftConfig sc;
        size_t ref_bytes;
        ulcnet_framing_config(&sc, NULL, NULL, 0);
        ref_bytes = ac_istft_get_mem_size(&sc);
        CHECK(ref_bytes != 0 && posix_memalign(&ref_mem, 16, ref_bytes) == 0,
              "allocate reference synthesis");
        ref_synth = ac_istft_init(ref_mem, ref_bytes, &sc);
        CHECK(ref_synth != NULL, "init reference synthesis");
    }

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far && out && ref, "allocate identity-test buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        int ref_wrote;
        /* Poison the output buffer first, so "hop 0 is exact zeros" and the
         * memcmp below prove the pipeline WROTE the hop, not that the
         * allocator handed over zeroed memory. */
        for (int i = 0; i < hop; ++i) out[i] = 1.0f;
        for (int i = 0; i < hop; ++i) {
            int64_t absolute = (int64_t)frame * hop + i;
            float phase = 2.0f * (float)M_PI * 700.0f *
                          (float)absolute / (float)ULCNET_SR;
            /* frame 0 silent so the 1-hop preamble is provably zero. */
            float echo = frame >= 1 ? 0.08f * sinf(phase) : 0.0f;
            float near = frame >= 30
                ? 0.025f * sinf(phase * 1.73f + 0.2f) : 0.0f;
            far[i] = echo;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) + near * (1.0f + 0.02f * ch);
            }
        }
        CHECK(ul4_process_with_activity(
                  p, microphones, far, frame >= 30, out) ==
                  FOUR_AEC_NR_RES_OK,
              "identity frame processes");
        CHECK(m.calls == (long)frame + 1,
              "exactly one inference per hop, from hop #0");

        /* The reference: the frame the model saw, through the test's own
         * synthesis. Same arithmetic on the same history -> bit-identical. */
        ref_wrote = ac_istft_push_planar(ref_synth, m.re, m.im, ref);
        if (frame == 0) {
            CHECK(ref_wrote == 0, "reference emits nothing on hop #0");
            for (int i = 0; i < hop; ++i) {
                float a = fabsf(out[i]);
                if (a > max_preamble) max_preamble = a;
            }
        } else {
            CHECK(ref_wrote == hop, "reference emits a full hop from hop #1");
            for (int i = 0; i < hop; ++i) {
                float a = fabsf(ref[i]);
                if (a > max_ref) max_ref = a;
                CHECK(isfinite(out[i]), "identity output finite");
            }
            CHECK(memcmp(out, ref, (size_t)hop * sizeof(float)) == 0,
                  "out[hop p] is bit-identical to the synthesis of the "
                  "spectrum the model saw at hop p (identity chain, 1-hop "
                  "latency)");
            identical_hops += 1;
        }
    }
    CHECK(m.resets == 0,
          "FIXED delay 0 keeps the identity run boundary-free (no reprime "
          "frame ever bypassed the capture model)");
    CHECK(max_preamble == 0.0f, "hop 0 emits exact zeros (1-hop preamble)");
    CHECK(identical_hops == FRAMES - 1, "every later hop was compared");
    /* The comparison must not be vacuous: the reference has to carry real
     * signal for this test to be able to fail. */
    CHECK(max_ref > 1e-3f, "beamformed error is non-trivial");

    ul4_destroy(p);
    free(ref_mem);
    free(ref);
    free(out);
    free(far);
    free(microphones);
    return 1;
}

/* Identity twin: a second pipeline built from the same config with NO model,
 * stepped in lockstep with the pipeline under test, so a test can compare
 * against what that pipeline WOULD emit without a model (identity frames
 * throughout, including its own reprime frames). */
typedef struct IdentityTwin {
    Ul4* p;
    float* out;
} IdentityTwin;

static int twin_create(IdentityTwin* t, const AudioPipeline4ChConfig* cfg) {
    t->p = ul4_create(cfg);
    t->out = NULL;
    if (t->p != NULL) {
        t->out = (float*)malloc(
            (size_t)ul4_hop_size(t->p) * sizeof(float));
    }
    return t->p != NULL && t->out != NULL;
}

static int twin_step(IdentityTwin* t, const float* microphones,
                     const float* far) {
    return ul4_process_with_activity(
               t->p, microphones, far, 0, t->out) == FOUR_AEC_NR_RES_OK;
}

static void twin_destroy(IdentityTwin* t) {
    ul4_destroy(t->p);
    free(t->out);
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
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
    CountingModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out;
    float* far_hist;   /* full far stream for the echo path */
    IdentityTwin twin;   /* NULL-model twin: the identity reference */
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

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;

    p = ul4_create(&cfg);
    CHECK(p != NULL && twin_create(&twin, &cfg),
          "create counting-model pipeline pair");
    hop = ul4_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(ul4_set_model(p, &model) == 0,
          "install counting model");
    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)total_frames * hop, sizeof(float));
    CHECK(microphones && far && out && far_hist,
          "allocate counting-model buffers");

    for (int frame = 0; frame < total_frames; ++frame) {
        const int frames = 1;   /* one frame per hop, from hop #0 */
        int scaled;

        if (frame == FRAMES) {
            /* Flip to the error-reporting regime: infer keeps being called
             * but its (deliberately wrong, 3x-scaled) output must be
             * ignored -- fail-open identity. */
            m.fail = 1;
            m.scale = 3.0f;
        }

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(ul4_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "counting-model frame processes");
        CHECK(ul4_last_delay(p, &delay) == 0,
              "delay-state accessor works");
        CHECK(twin_step(&twin, microphones, far),
              "counting-model reference frame processes");

        if (delay.changed) {
            changed_count += 1;
            reprime_armed = ULCNET_NN_STAGE_REPRIME_FRAMES;
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
                    float d = fabsf(out[i] - expected_gain * twin.out[i]);
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
              changed_count * ULCNET_NN_STAGE_REPRIME_FRAMES &&
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

    /* Pipeline reset must also reset the runtime and restart the 1-hop
     * preamble. */
    ul4_reset(p);
    CHECK(m.reset_calls == changed_count + 1,
          "pipeline reset fires model->reset once");
    {
        long infer_before = m.infer_calls;
        float max_out = 0.0f;
        CHECK(ul4_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "post-reset frame processes");
        for (int i = 0; i < hop; ++i) {
            float a = fabsf(out[i]);
            if (a > max_out) max_out = a;
        }
        CHECK(max_out == 0.0f, "post-reset hop#0 emits exact zeros");
        CHECK(m.infer_calls == infer_before + 1,
              "post-reset hop#0 is one inference, like every hop");
    }

    ul4_destroy(p);
    twin_destroy(&twin);
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
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
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
    p = ul4_create(&cfg);
    CHECK(p != NULL, "create 4ch FIXED transition pipeline");
    if (!p) return 0;

    memset(&m, 0, sizeof(m));
    m.scale = 1.0f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(ul4_set_model(p, &model) == 0,
          "install model for 4ch FIXED transition");

    for (int frame = 0; frame < N; ++frame) {
        CHECK(ul4_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "4ch FIXED transition frame processes");
        CHECK(ul4_last_delay(p, &delay) == 0,
              "4ch FIXED transition exposes delay state");
        if (delay.changed) changed++;
        if (delay.solid && first_solid < 0) first_solid = frame;
    }

    CHECK(first_solid == 2,
          "4ch FIXED reports aligned far on the same hop the ring can serve it");
    CHECK(changed == 0,
          "4ch FIXED has no estimator-generated changed event");
    CHECK(m.reset_calls == 1,
          "4ch composition resets model once at FIXED raw-to-aligned transition");
    /* N hops, one frame each; exactly the straddling ones are skipped. */
    CHECK(m.infer_calls == N - ULCNET_NN_STAGE_REPRIME_FRAMES,
          "4ch FIXED transition suppresses inference for exactly the "
          "straddling reprime frames and no others");
    ul4_destroy(p);
    return 1;
}

/* ============================================================================
 * 2c. FIXED pre-lock seam content: RAW far, byte for byte. Under FIXED the unlocked
 * stretch is the whole ring fill (up to half a second at a realistic loudspeaker delay), and
 * zero-filling it would give the linear filters a silent reference while the mic carries the
 * echo. Checked against the caller's own far history: every pre-solid hop equals the raw far
 * of the same hop, the first solid hop onward equals the far delayed by fixed_delay_samples,
 * and the switch happens on hop ceil(fixed / hop) only, as a whole-hop decision.
 * ========================================================================== */

/* This case exercises the 4ch CORE's FIXED seam, so it pins the core's own
 * 16 kHz / FFT 512 grid rather than the compiled ULCNet grid. Delays handed
 * to it must therefore be expressed in FIXED_SEAM_HOP, not in ULCNET_HOP:
 * the switch hop is ceil(fixed / FIXED_SEAM_HOP). */
#define FIXED_SEAM_SR   16000
#define FIXED_SEAM_FFT  512
#define FIXED_SEAM_HOP  (FIXED_SEAM_FFT / 2)

static int fixed_seam_case(int fixed_delay, int frames, int expect_first_solid) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(FIXED_SEAM_SR);
    FourAecNrRes* core;
    FourAecFront pre;
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

    cfg.fft_size = FIXED_SEAM_FFT;
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
        CHECK(four_aec_front_process(
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
    CHECK(fixed_seam_case(8 * FIXED_SEAM_HOP, 12, 8), "fixed=8 hops");
    /* Not a multiple of the hop: a per-sample rule would splice raw and
     * shifted audio inside hop 1. */
    CHECK(fixed_seam_case(320, 8, 2), "fixed=320 samples");
    /* 500 ms at 16 kHz -- the realistic loudspeaker-path case, and the one
     * where zero-fill would have silenced half a second of reference. */
    CHECK(fixed_seam_case(8000, 40, 32), "fixed=500 ms");
    return 1;
}

/* 2b. Relock on the SAME delay must still reset the model: LOCKED(applied delay 0) ->
 * UNLOCKED for several hops -> LOCKED(0 again). The model keeps running across the unlocked
 * stretch on the seam's raw far, so its recurrent state spans a different delay regime and
 * the relock hop must report `changed`. Delay 0 is pinned because it is also the core's
 * accepted_delay when nothing is accepted, the one pair a value-only `changed` test cannot
 * tell apart (TRUE_DELAY 64 minus the estimator's 32-sample headroom lands there; the test
 * asserts delay_samples == 0). MUTATION: a value-only `state.changed = eligible &&
 * estimated != p->accepted_delay;` in update_shared_delay() turns the acquisition red. */

static int test_relock_same_delay_resets_model(void) {
    enum { WARM_FRAMES = 60, RELOCK_FRAMES = 60 };
    const int true_delay = 64 * DELAY_RATE_FACTOR;
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
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

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;

    p = ul4_create(&cfg);
    CHECK(p != NULL, "create relock pipeline");
    hop = ul4_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(ul4_set_model(p, &model) == 0,
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
             * four_aec_nr_res_reset() (which the unit's reset calls) is the
             * sole caller of delay_aec_reset(), and the
             * estimator's REFINED-confidence latch cannot otherwise drop
             * once set. It also puts accepted_delay back to 0 -- which is
             * precisely why a same-value relock at 0 is invisible to a
             * value-only test. */
            reset_before_relock = m.reset_calls;
            ul4_reset(p);
            reset_at_pipeline_reset = m.reset_calls;
        }

        fill_echo_hop(frame, hop, true_delay, far_hist, far, microphones);
        reset_before_hop = m.reset_calls;
        CHECK(ul4_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "relock frame processes");
        CHECK(ul4_last_delay(p, &delay) == 0,
              "delay-state accessor works");
        /* A model reset flushes recurrent state against the applied delay, so it may only fire on
         * a hop where an alignment generation was ACCEPTED, never where the estimator is merely
         * confident (confidence >= 1.0 is delay_aec_is_solid()). Recomputed from the seam's own
         * estimator fields, independent of the `solid`/`changed` flags under test.
         * MUTATION: handing the stage `solid = confidence >= 0.5f` instead of the front's
         * delay.solid fires a reset on the first COARSE hop and this counter goes non-zero. */
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
                             acquired_delay, true_delay);
                    CHECK(acquired_delay >= 0 &&
                          acquired_delay <= true_delay &&
                          true_delay - acquired_delay <=
                              64 * DELAY_RATE_FACTOR, why);
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

    ul4_destroy(p);
    free(far_hist);
    free(out);
    free(far);
    free(microphones);
    return 1;
}

/* 5. Far-timestamp after fixed-delay acquisition: the model's far branch must carry the
 *    SAME input hop as its error branch. Far-passthrough model, silence on all mics, one
 *    unit impulse in far at sample T. Expected output position T + one hop: production feeds
 *    the caller's raw far even after the FIXED delay is solid (+0), the far branch is framed
 *    from the current hop like the error branch (+0), and the ULCNet synthesis adds one hop
 *    (output hop #p = chain input hop p-1).
 *    MUTATION: delaying the far branch by a hop moves the impulse to T + 2 hops. */

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

/* Shared shell of the two impulse-timestamp tests (5 and 5b): FRAMES hops
 * of silence with one unit impulse on ONE branch at (IMP_FRAME, IMP_OFF),
 * through `model` (NULL = identity). Reports where the output peaks and how
 * much of its energy sits within +-8 samples of the expected position; the
 * assertions differ per branch and stay in the callers. */
typedef struct ImpulseRun {
    int imp_index;
    int expect_index;    /* imp_index + 1 hop: the synthesis' one hop */
    int found_index;
    float peak;
    double near_ratio;   /* energy within +-8 of expect_index / total */
} ImpulseRun;

static int run_impulse(const AudioPipeline4ChConfig* cfg,
                       const UlcnetModel* model, int impulse_on_mics,
                       int activity, ImpulseRun* r) {
    enum { FRAMES = 40, IMP_FRAME = 8, IMP_OFF = 37 };
    Ul4* p;
    float* microphones;
    float* far;
    float* out_hist;
    int hop;
    double energy_total = 0.0;
    double energy_near = 0.0;

    p = ul4_create(cfg);
    CHECK(p != NULL, "create impulse-timestamp pipeline");
    hop = ul4_hop_size(p);
    if (model != NULL) {
        CHECK(ul4_set_model(p, model) == 0,
              "install impulse-timestamp model");
    }
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    out_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out_hist,
          "allocate impulse-timestamp buffers");

    memset(r, 0, sizeof(*r));
    r->imp_index = IMP_FRAME * hop + IMP_OFF;
    r->expect_index = r->imp_index + hop;
    r->found_index = -1;

    for (int frame = 0; frame < FRAMES; ++frame) {
        memset(far, 0, (size_t)hop * sizeof(float));
        memset(microphones, 0,
               (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
        if (frame == IMP_FRAME) {
            if (impulse_on_mics) {
                for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                    microphones[IMP_OFF * FOUR_AEC_NR_RES_CHANNELS + ch] = 1.0f;
            } else {
                far[IMP_OFF] = 1.0f;
            }
        }
        CHECK(ul4_process_with_activity(
                  p, microphones, far, activity,
                  out_hist + (size_t)frame * hop) == FOUR_AEC_NR_RES_OK,
              "impulse-timestamp frame processes");
    }

    for (int t = 0; t < FRAMES * hop; ++t) {
        float a = fabsf(out_hist[t]);
        energy_total += (double)a * (double)a;
        if (a > r->peak) {
            r->peak = a;
            r->found_index = t;
        }
    }
    for (int t = r->expect_index - 8; t <= r->expect_index + 8; ++t) {
        if (t >= 0 && t < FRAMES * hop)
            energy_near += (double)out_hist[t] * (double)out_hist[t];
    }
    r->near_ratio = energy_total > 0.0 ? energy_near / energy_total : 0.0;

    ul4_destroy(p);
    free(out_hist);
    free(far);
    free(microphones);
    return 1;
}

static int test_raw_far_timestamp_after_fixed_delay_lock(void) {
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    UlcnetModel model;
    ImpulseRun r;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.4f;
    cfg.bf_mu = 0.02f;
    cfg.core.delay_mode = AEC_DELAY_FIXED;
    cfg.core.fixed_delay_samples = 2 * ULCNET_HOP;
    memset(&model, 0, sizeof(model));
    model.infer = passthrough_far_infer;
    model.io_descriptor = test_io_descriptor();

    CHECK(run_impulse(&cfg, &model, 0, 0, &r), "far-timestamp run");
    printf("raw-far timestamp (4ch after fixed-delay lock): impulse at far[%d] -> out[%d] "
           "(expected %d, offset %+d samples, peak %.4f)\n",
           r.imp_index, r.found_index, r.expect_index,
           r.found_index - r.expect_index, r.peak);
    CHECK(r.found_index == r.expect_index,
          "far impulse lands at impulse + 1 hop (the ULCNet synthesis only; "
          "no err/far skew)");
    CHECK(r.peak > 0.9f, "far impulse reconstructed at ~unit amplitude");
    return 1;
}

/* 5b. Error-branch timestamp and content: the mirror of test 5 on the other branch, read
 *    against the INPUT. Silence on far, one unit impulse on all four mics, a frozen fixed
 *    beamformer (mu 0, vad 1), FIXED delay 0, NULL model. The impulse must come out at
 *    exactly impulse + 1 hop and AS an impulse: a dropped imaginary part or an error branch
 *    delayed by a hop is self-consistent in every model-output test and is caught only here.
 *    MUTATION: a hop of delay moves the peak by +hop; zeroing err_im mirrors the impulse and
 *    raises the energy outside the peak neighbourhood to about half. */

static int test_error_timestamp_and_content(void) {
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    ImpulseRun r;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.0f;
    cfg.core.delay_mode = AEC_DELAY_FIXED;
    cfg.core.fixed_delay_samples = 0;

    /* Mics carry the impulse, far is silent, NULL model = identity. */
    CHECK(run_impulse(&cfg, NULL, 1, 1, &r), "error-timestamp run");
    printf("error timestamp (4ch): impulse at mic[%d] -> out[%d] (expected %d, "
           "offset %+d samples, peak %.4f, near-peak energy %.3f of total)\n",
           r.imp_index, r.found_index, r.expect_index,
           r.found_index - r.expect_index, r.peak, r.near_ratio);
    /* The frozen beamformer's fractional steering delays and the mic HPF
     * spread the impulse over a few samples and can move its peak sample by
     * one or two (measured: +0 at 16 kHz, -1 at 48 kHz) and scale it to
     * about 0.2-0.25 of unit; +-8 covers the spread and nothing else. A
     * mirrored copy (imaginary part dropped) lands a hop-scale distance away
     * and halves the ratio (measured 0.48 vs 0.91/0.96). */
    CHECK(r.found_index >= r.expect_index - 2 &&
          r.found_index <= r.expect_index + 2,
          "mic impulse lands at impulse + 1 hop (within the beamformer's "
          "steering spread) through the direct BF -> synthesis path");
    CHECK(r.peak > 0.1f,
          "mic impulse comes through the frozen beamformer (spread, not "
          "attenuated away)");
    CHECK(r.near_ratio > 0.8,
          "the impulse comes out as an impulse: the spectrum reached the "
          "synthesis with both parts intact (no mirrored copy, no smear)");
    return 1;
}

/* ============================================================================
 * 6. Production never gates application on the delay lock: the 0.5x model's
 *    output must appear from the FIRST emitted frame, while delay.solid is
 *    still 0 (the paper contract does not depend on lock).
 * ========================================================================== */

static int test_model_applies_unlocked(void) {
    enum { FRAMES = 60, TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
    CountingModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out;
    float* far_hist;
    IdentityTwin twin;   /* NULL-model twin: the identity reference */
    long expected_infer = 0;
    int reprime_armed = 0;
    int cur_reprime = 0;
    int prev_reprime = 0;
    int unsolid_checks = 0;
    int first_solid_frame = -1;
    float max_err = 0.0f;
    int hop;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;

    p = ul4_create(&cfg);
    CHECK(p != NULL && twin_create(&twin, &cfg),
          "create unlocked-model pipeline pair");
    hop = ul4_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(ul4_set_model(p, &model) == 0,
          "install 0.5x model");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out && far_hist,
          "allocate unlocked-model buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        const int frames = 1;   /* one frame per hop, from hop #0 */
        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(ul4_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "unlocked-model frame processes");
        CHECK(ul4_last_delay(p, &delay) == 0,
              "unlocked-model delay accessor works");
        if (first_solid_frame < 0 && delay.solid) first_solid_frame = frame;
        CHECK(twin_step(&twin, microphones, far),
              "unlocked-model reference frame processes");

        /* Stepped on every emitted frame except the identity-reprime frames
         * armed at an alignment generation. */
        if (delay.changed)
            reprime_armed = ULCNET_NN_STAGE_REPRIME_FRAMES;
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
         * 0.5x-scaled identity stream on EVERY frame >= 2 (skip delay-change
         * boundary frames where the beam itself is discontinuous, and the
         * reprime frames, which are identity by policy). */
        if (frame >= 2 && !delay.changed && !cur_reprime && !prev_reprime) {
            float e = 0.0f;
            for (int i = 0; i < hop; ++i) {
                float d = fabsf(out[i] - m.scale * twin.out[i]);
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

    ul4_destroy(p);
    twin_destroy(&twin);
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
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* pa;
    IdentityTwin twin;   /* NULL model: identity */
    NanModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out_a;
    float* far_hist;
    int poisoned[FRAMES];
    int reprime_left = 0;
    int n_reprime = 0;
    int n_equal = 0;
    int n_differ = 0;
    int hop;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;

    pa = ul4_create(&cfg);
    CHECK(pa != NULL && twin_create(&twin, &cfg),
          "create NaN-guard pipeline pair");
    hop = ul4_hop_size(pa);

    memset(&m, 0, sizeof(m));
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = nan_infer;
    model.io_descriptor = test_io_descriptor();
    CHECK(ul4_set_model(pa, &model) == 0,
          "install NaN-poisoning model");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out_a = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out_a && far_hist,
          "allocate NaN-guard buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        int equal_expected;
        int bitwise_equal;
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        m.poison = (frame >= 20 && frame < 28) || (frame >= 40 && frame < 48);
        poisoned[frame] = m.poison;

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(ul4_process_with_activity(
                  pa, microphones, far, 0, out_a) == FOUR_AEC_NR_RES_OK,
              "NaN-guard model frame processes");
        CHECK(twin_step(&twin, microphones, far),
              "NaN-guard reference frame processes");

        /* Both pipelines cross the same alignment generations, and an
         * identity-reprime frame is an identity frame exactly like a
         * discarded poisoned one -- fold it into the same schedule so the
         * 50%-overlap rule below stays exact. The acquisition lands well
         * inside the steady one-frame-per-hop region. */
        CHECK(ul4_last_delay(pa, &delay) == 0,
              "NaN-guard delay accessor works");
        if (delay.changed)
            reprime_left = ULCNET_NN_STAGE_REPRIME_FRAMES;
        /* One frame per hop, so a hop spends at most one frame of the
         * budget. */
        if (reprime_take(&reprime_left, 1) > 0) {
            poisoned[frame] = 1;
            n_reprime += 1;
        }

        for (int i = 0; i < hop; ++i) {
            CHECK(isfinite(out_a[i]),
                  "no NaN/Inf ever reaches the pipeline output");
        }

        /* 50%-overlap mixing rule: hop 0 emits exact zeros on both sides
         * (frame #0 lies inside the synthesis's trimmed half window); hop
         * p >= 1 mixes the frames pushed at hops p-1 and p, so it is
         * bit-identical only when BOTH were identity frames. */
        if (frame == 0) {
            equal_expected = 1;
        } else {
            equal_expected = poisoned[frame - 1] && poisoned[frame];
        }
        bitwise_equal =
            memcmp(out_a, twin.out, (size_t)hop * sizeof(float)) == 0;
        if (equal_expected) {
            n_equal += 1;
            CHECK(bitwise_equal,
                  "rc==0 NaN frames are discarded bitwise (identity path)");
        } else if (frame >= 3) {
            /* frame 2's beam may still be near zero (echo starts at t=400,
             * i.e. hop 1, whose frame reaches the output at hop 2 -- keep a
             * one-frame margin); from frame 3 the 0.5x scale must show. */
            n_differ += 1;
            CHECK(!bitwise_equal,
                  "clean frames after a NaN window are applied again");
        }
    }
    CHECK(n_equal >= 10 && n_differ >= 10,
          "NaN-guard coverage: both regimes actually compared");
    CHECK(n_reprime % ULCNET_NN_STAGE_REPRIME_FRAMES == 0,
          "NaN-guard: reprime frames accounted for in whole generations");

    ul4_destroy(pa);
    twin_destroy(&twin);
    free(far_hist);
    free(out_a);
    free(far);
    free(microphones);
    return 1;
}

/* 8. Full-write contract: infer() writes only the FIRST 100 bins (0.5x) on scheduled hops
 *    but returns rc==0. The stage pre-fills the enh_re/enh_im staging with NaN before
 *    every infer call (ulcnet_process.h's FULL-WRITE CONTRACT), so the unwritten bins stay
 *    NaN and the finite guard discards the frame: bit-identical to the NULL-model pipeline
 *    under the mixing rule of test 7, and the next fully-written frame is applied again.
 *    MUTATION: removing the NaN pre-fill in ulcnet_nn_stage_nn_begin() leaves stale finite
 *    values in bins 100..256 and the bit-identical check goes red. */

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
    /* rc stays 0 even for the partial write: only the stage's NaN
     * pre-fill + finite guard can catch the 157 unwritten bins. */
    return 0;
}

static int test_partial_write_guard(void) {
    enum { FRAMES = 60, TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* pa;
    IdentityTwin twin;   /* NULL model: identity */
    PartialModel m;
    UlcnetModel model;
    FourAecNrResDelayState delay;
    float* microphones;
    float* far;
    float* out_a;
    float* far_hist;
    int partial[FRAMES];
    int reprime_left = 0;
    int n_reprime = 0;
    int n_equal = 0;
    int n_differ = 0;
    int hop;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;

    pa = ul4_create(&cfg);
    CHECK(pa != NULL && twin_create(&twin, &cfg),
          "create partial-write pipeline pair");
    hop = ul4_hop_size(pa);

    memset(&m, 0, sizeof(m));
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = partial_write_infer;
    model.io_descriptor = test_io_descriptor();
    CHECK(ul4_set_model(pa, &model) == 0,
          "install partial-write model");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out_a = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && out_a && far_hist,
          "allocate partial-write buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        int equal_expected;
        int bitwise_equal;
        /* Two windows so recovery is proven twice: 20..27 and 40..47. */
        m.partial = (frame >= 20 && frame < 28) || (frame >= 40 && frame < 48);
        partial[frame] = m.partial;

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(ul4_process_with_activity(
                  pa, microphones, far, 0, out_a) == FOUR_AEC_NR_RES_OK,
              "partial-write model frame processes");
        CHECK(twin_step(&twin, microphones, far),
              "partial-write reference frame processes");

        /* Same reprime bookkeeping as the NaN test: an identity-reprime
         * frame is an identity frame exactly like a discarded partial one. */
        CHECK(ul4_last_delay(pa, &delay) == 0,
              "partial-write delay accessor works");
        if (delay.changed)
            reprime_left = ULCNET_NN_STAGE_REPRIME_FRAMES;
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
        } else {
            equal_expected = partial[frame - 1] && partial[frame];
        }
        bitwise_equal =
            memcmp(out_a, twin.out, (size_t)hop * sizeof(float)) == 0;
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
    CHECK(n_reprime % ULCNET_NN_STAGE_REPRIME_FRAMES == 0,
          "partial-write: reprime frames accounted for in whole generations");

    ul4_destroy(pa);
    twin_destroy(&twin);
    free(far_hist);
    free(out_a);
    free(far);
    free(microphones);
    return 1;
}

/* 9. Identity-reprime straddle derivation: per branch, how many frames emitted at or after
 *    an alignment boundary at hop T still have a pre-boundary hop in their two-hop window.
 *    One unit impulse at the middle of input hop T-1 (window weight 0.707 in both covering
 *    frames) is marked through the mics (error branch) or the reference (far branch) in
 *    separate runs. The count comes from a boundary-free control (FIXED delay 0, no reprime
 *    armed, asserted) and is 1 on both branches. The boundary run (FIXED delay = T hops)
 *    must skip exactly REPRIME frames, at hops T.., and the first frame the model then sees
 *    must be marker-free on both. MUTATION: no reprime, or one frame too few, turns that red. */

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
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
    UlcnetModel model;
    float microphones[FOUR_AEC_NR_RES_CHANNELS * ULCNET_HOP];
    float far[ULCNET_HOP];
    float out[ULCNET_HOP];

    memset(st, 0, sizeof(*st));
    /* Frozen fixed beamformer (mu 0 and vad_external 1 both freeze BF
     * adaptation) so the beam path is time-invariant and the marker cannot
     * be smeared by an adapting filter. */
    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.0f;
    cfg.core.delay_mode = AEC_DELAY_FIXED;
    cfg.core.fixed_delay_samples = fixed_delay;

    p = ul4_create(&cfg);
    if (!p) return 0;
    memset(&model, 0, sizeof(model));
    model.user = st;
    model.infer = probe_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = probe_reset;
    if (ul4_set_model(p, &model) != 0) {
        ul4_destroy(p);
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
        if (ul4_process_with_activity(
                p, microphones, far, 1, out) != FOUR_AEC_NR_RES_OK) {
            ul4_destroy(p);
            return 0;
        }
    }
    ul4_destroy(p);
    return 1;
}

/* Frames emitted at hops [RP_T, RP_T+RP_WINDOW) that still carry the marker, plus the
 * classification margin (smallest marked peak / largest unmarked peak). Per branch: far, the
 * marker reaches the analysis untouched, so the memory is exactly the window straddle (the
 * derivation of REPRIME_FRAMES); error, the marker passes the AEC linear filter first, so
 * the memory is the straddle plus the filter's causal tail and can only bound the straddle
 * from above. The tail is why the branches agree at 16 kHz and differ at 48 kHz:
 * filter_length is 52 ms (3.25 hops) at 16 kHz but 64 ms (6.00 hops) at 48 kHz. */
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

    expected_frames = RP_RUN;   /* one frame per hop, from hop #0 */

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
    CHECK(straddle_err >= ULCNET_NN_STAGE_REPRIME_FRAMES,
          "4ch ERROR branch: chain memory is at least "
          "ULCNET_NN_STAGE_REPRIME_FRAMES -- it carries the AEC "
          "filter tail on top of the straddle, so it bounds rather than "
          "derives it");
    /* No absolute floor on dmin: the impulse arrives spread by the AEC
     * filter, so its peak is grid-dependent. The separation ratio is what
     * makes the classification sound. */
    CHECK(straddle_err > 0 && dmin > 20.0f * cmax,
          "4ch ERROR branch marker is unambiguous (marked peak far above the "
          "unmarked residue)");

    straddle_far = reprime_straddle_count(&far_ctl, 1, &dmin, &cmax);
    printf("reprime derivation (4ch, FAR branch):   %d straddling frames at hops "
           "%d..%d; marked peak >= %.4f, unmarked peak <= %.3e\n",
           straddle_far, RP_T, RP_T + RP_WINDOW - 1, (double)dmin, (double)cmax);
    CHECK(straddle_far == ULCNET_NN_STAGE_REPRIME_FRAMES,
          "4ch FAR branch: the emitted frames that still contain pre-switch "
          "samples number exactly ULCNET_NN_STAGE_REPRIME_FRAMES");
    CHECK(straddle_far > 0 && dmin > 20.0f * cmax && dmin > 0.2f,
          "4ch FAR branch marker is unambiguous (marked peak far above the "
          "unmarked residue)");

    /* The boundary runs: the stage skips exactly those frames. */
    CHECK(err_bnd.resets == 1 && far_bnd.resets == 1,
          "boundary run: the FIXED raw->aligned switch fired model->reset once");

    for (int c = 0; c < err_bnd.calls; ++c) {
        if (err_bnd.call_hop[c] >= RP_T &&
            err_bnd.call_hop[c] <
                RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES)
            stepped_inside_reprime += 1;
        if (err_bnd.call_hop[c] >= RP_T && first_visible_err < 0) {
            first_visible_err = err_bnd.call_hop[c];
            first_visible_err_peak = err_bnd.err_peak[c];
        }
    }
    for (int c = 0; c < far_bnd.calls; ++c) {
        if (far_bnd.call_hop[c] >= RP_T &&
            far_bnd.call_hop[c] <
                RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES)
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
     * be masked by a count assertion firing first. Only the far branch can
     * carry it: the error branch's first visible frame legitimately holds
     * AEC filter tail, which is filtered echo of pre-switch content rather
     * than a pre-switch SAMPLE, and no reprime length removes it. */
    CHECK(first_visible_far_peak >= 0.0f &&
          first_visible_far_peak <= RP_FLOOR,
          "the first model-visible frame after the reprime contains NO "
          "pre-switch sample on the far branch (the error branch's residue "
          "is AEC filter tail, not a straddling sample)");
    CHECK(stepped_inside_reprime == 0,
          "boundary run: the model is not stepped on any reprime hop");
    CHECK(first_visible_err ==
              RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES &&
          first_visible_far ==
              RP_T + ULCNET_NN_STAGE_REPRIME_FRAMES,
          "boundary run: stepping resumes exactly REPRIME frames after the "
          "boundary hop");
    CHECK(err_bnd.calls ==
              expected_frames - ULCNET_NN_STAGE_REPRIME_FRAMES &&
          far_bnd.calls ==
              expected_frames - ULCNET_NN_STAGE_REPRIME_FRAMES,
          "boundary run: exactly REPRIME emitted frames skipped inference");
    return 1;
}

/* ============================================================================
 * 10. Identity-reprime behaviour: exactly REPRIME identity frames per alignment generation,
 *     for a mid-stream delay change and for the FIXED ring-fill completion. The test
 *     predicts the infer delta per hop from its own copy of the policy and compares with the
 *     counting model; reset stays once per generation.
 * ========================================================================== */

static int test_reprime_behavior(void) {
    enum { FRAMES = 120, TRUE_DELAY = 400, FIXED_HOPS = 8 };
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
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

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;

    p = ul4_create(&cfg);
    CHECK(p != NULL, "create reprime-behaviour pipeline");
    hop = ul4_hop_size(p);

    memset(&m, 0, sizeof(m));
    m.scale = 0.5f;
    memset(&model, 0, sizeof(model));
    model.user = &m;
    model.infer = counting_infer;
    model.io_descriptor = test_io_descriptor();
    model.reset = counting_reset;
    CHECK(ul4_set_model(p, &model) == 0,
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
        const int frames = 1;   /* one frame per hop, from hop #0 */
        long calls_before = m.infer_calls;
        int skipped_here;
        long delta;

        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        CHECK(ul4_process_with_activity(
                  p, microphones, far, 0, out) == FOUR_AEC_NR_RES_OK,
              "reprime-behaviour frame processes");
        CHECK(ul4_last_delay(p, &delay) == 0,
              "reprime-behaviour delay accessor works");

        if (delay.changed) {
            armed = ULCNET_NN_STAGE_REPRIME_FRAMES;
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
              changed_count * ULCNET_NN_STAGE_REPRIME_FRAMES,
          "reprime behaviour: REPRIME frames skipped per generation and no "
          "others -- compute DROPS on those frames, it never doubles");
    CHECK(resumed_frame ==
              boundary_frame + ULCNET_NN_STAGE_REPRIME_FRAMES,
          "reprime behaviour: stepping stops on the boundary hop and resumes "
          "REPRIME frames later");
    CHECK(m.reset_calls == changed_count,
          "reprime behaviour: still exactly one model->reset per generation");
    ul4_destroy(p);
    free(far_hist);
    free(out);
    free(far);
    free(microphones);

    /* ---- FIXED ring-fill completion ---- */
    {
        AudioPipeline4ChConfig fcfg =
            ulcnet_example_unit_config();
        Ul4* fp;
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
        fp = ul4_create(&fcfg);
        CHECK(fp != NULL, "create FIXED reprime-behaviour pipeline");
        memset(&fm, 0, sizeof(fm));
        fm.scale = 1.0f;
        memset(&fmodel, 0, sizeof(fmodel));
        fmodel.user = &fm;
        fmodel.infer = counting_infer;
        fmodel.io_descriptor = test_io_descriptor();
        fmodel.reset = counting_reset;
        CHECK(ul4_set_model(fp, &fmodel) == 0,
              "install counting model (FIXED reprime behaviour)");

        memset(mics, 0, sizeof(mics));
        for (int frame = 0; frame < 20; ++frame) {
            FourAecNrResDelayState fdelay;
            const int frames = 1;   /* one frame per hop, from hop #0 */
            long calls_before = fm.infer_calls;
            int solid_before;
            int skipped_here;
            long delta;

            CHECK(ul4_last_delay(fp, &fdelay) == 0,
                  "FIXED reprime: delay accessor works before the hop");
            solid_before = fdelay.solid;
            for (int i = 0; i < ULCNET_HOP; ++i) ffar[i] = 0.25f * frand();
            CHECK(ul4_process_with_activity(
                      fp, mics, ffar, 0, fout) == FOUR_AEC_NR_RES_OK,
                  "FIXED reprime frame processes");
            CHECK(ul4_last_delay(fp, &fdelay) == 0,
                  "FIXED reprime: delay accessor works after the hop");
            if (frame != 0 && !solid_before && fdelay.solid) {
                farmed = ULCNET_NN_STAGE_REPRIME_FRAMES;
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
                             ULCNET_NN_STAGE_REPRIME_FRAMES)
                calls_after_reprime = (int)delta;
        }

        CHECK(fixed_boundary == FIXED_HOPS,
              "FIXED reprime: the raw->aligned switch is on the expected hop");
        CHECK(fskipped == ULCNET_NN_STAGE_REPRIME_FRAMES,
              "FIXED reprime: exactly REPRIME frames skipped at the fill "
              "completion");
        CHECK(calls_at_boundary == 0 && calls_after_reprime == 1,
              "FIXED reprime: the boundary hop steps the model 0 times and "
              "stepping is back to one per frame after the reprime");
        CHECK(fm.reset_calls == 1,
              "FIXED reprime: exactly one model->reset");
        ul4_destroy(fp);
    }
    return 1;
}

/* ============================================================================
 * 11. Fixed deployment contract: production accepts only the raw-far descriptor. The
 * aligned enumerator is used only for negative ABI tests; there is no runtime far-mode
 * setter.
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

static int test_raw_descriptor_gate(void) {
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4* p;
    UlcnetModelIoDescriptor raw_desc;
    UlcnetModelIoDescriptor aligned_desc;
    UlcnetModel raw_model;
    UlcnetModel aligned_model;
    UlcnetModel undescribed_model;

    CHECK(ulcnet_model_io_descriptor_default(8, &raw_desc) == 0 &&
          ulcnet_model_io_descriptor_default(8, &aligned_desc) == 0,
          "build model-I/O descriptors");
    aligned_desc.far_input_mode = ULCNET_FAR_ALIGNED;
    CHECK(raw_desc.far_input_mode == ULCNET_FAR_RAW,
          "descriptor_default publishes fixed raw far");

    memset(&undescribed_model, 0, sizeof(undescribed_model));
    undescribed_model.infer = gate_infer_identity;
    undescribed_model.io_descriptor = test_io_descriptor();
    raw_model = undescribed_model;
    raw_model.io_descriptor = &raw_desc;
    aligned_model = undescribed_model;
    aligned_model.io_descriptor = &aligned_desc;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;
    p = ul4_create(&cfg);
    CHECK(p != NULL, "create aligned-descriptor gate pipeline");

    /* Baseline: an undescribed model publishes no contract, so the
     * rejections below cannot be passing for want of a model. */
    CHECK(ul4_set_model(p, &undescribed_model) == 0,
          "a model with io_descriptor == NULL remains supported");

    /* Production accepts only a raw-far descriptor. */
    CHECK(ul4_set_model(p, &aligned_model) != 0,
          "aligned descriptor is rejected");
    CHECK(ul4_set_model(p, &raw_model) == 0,
          "raw descriptor is accepted");

    ul4_destroy(p);
    return 1;
}

/* ============================================================================
 * 12. Phased hop
 * ========================================================================== */

/* The phased hop (process_begin / nn_begin / nn_end / process_end) is
 * process_with_activity() with the model call moved out: a twin driven phase
 * by phase, with the same network run by the caller, reproduces the
 * callback-driven instance byte for byte (an echo path appears mid-run, so
 * the alignment boundary and its reprime hops are part of the claim). The
 * fail-open rows pin the ways a frame ends as the identity: a failed run, an
 * output with a non-finite bin, a frame left open, a frame never fetched. */
enum { PH_RUN, PH_FAIL, PH_NAN, PH_OPEN, PH_UNFETCHED };

static int phased_half_infer(void* user,
                             const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                             const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                             float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    int poison = user != NULL;
    (void)far_re; (void)far_im;
    for (int k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = 0.5f * err_re[k];
        out_im[k] = 0.5f * err_im[k];
    }
    if (poison) { out_re[100] = (float)NAN; out_im[200] = (float)INFINITY; }
    return 0;
}

static int phased_fail_infer(void* user,
                             const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                             const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                             float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    (void)user; (void)far_re; (void)far_im;
    for (int k = 0; k < ULCNET_BINS; ++k) { out_re[k] = 2.0f * err_re[k]; out_im[k] = 2.0f * err_im[k]; }
    return 1;   /* garbage written, failure reported: the pipeline must discard it */
}

static int phased_twin_matches(int mode, int frames) {
    enum { TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4 *ph, *cb;
    UlcnetModel model;
    UlcnetNnJob job;
    float *mics, *far, *far_hist, *a, *b;
    int hop, ran = 0, nonzero = 0, ok = 1;
    int poison_tag = 1;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.3f;
    cfg.bf_mu = 0.02f;
    ph = ul4_create(&cfg);
    cb = ul4_create(&cfg);
    if (!ph || !cb) return 0;
    memset(&model, 0, sizeof(model));
    if (mode == PH_RUN || mode == PH_FAIL || mode == PH_NAN) {
        model.infer = mode == PH_FAIL ? phased_fail_infer : phased_half_infer;
        model.user = mode == PH_NAN ? &poison_tag : NULL;
        model.io_descriptor = test_io_descriptor();
        if (ul4_set_model(cb, &model) != 0) return 0;
    }
    hop = ul4_hop_size(ph);
    mics = (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)frames * hop, sizeof(float));
    a = (float*)malloc((size_t)hop * sizeof(float));
    b = (float*)malloc((size_t)hop * sizeof(float));
    for (int f = 0; f < frames && ok; ++f) {
        int r = 0;
        fill_echo_hop(f, hop, TRUE_DELAY, far_hist, far, mics);
        ok = ul4_process_with_activity(cb, mics, far, 0, b) == FOUR_AEC_NR_RES_OK &&
             ul4_process_begin(ph, mics, far, 0) == FOUR_AEC_NR_RES_OK;
        if (!ok) break;
        if (mode != PH_UNFETCHED) {
            r = ul4_nn_begin(ph, &job);
            if (r < 0) { ok = 0; break; }
        }
        if (r == 1) {
            ran++;
            if (mode == PH_RUN)
                ok = phased_half_infer(NULL, job.err_re, job.err_im, job.far_re, job.far_im,
                                       job.out_re, job.out_im) == 0 &&
                     ul4_nn_end(ph, 0) == FOUR_AEC_NR_RES_OK;
            else if (mode == PH_FAIL)
                ok = ul4_nn_end(ph, -1) == FOUR_AEC_NR_RES_OK;
            else if (mode == PH_NAN)
                ok = phased_half_infer(&poison_tag, job.err_re, job.err_im, job.far_re, job.far_im,
                                       job.out_re, job.out_im) == 0 &&
                     ul4_nn_end(ph, 0) == FOUR_AEC_NR_RES_OK;
        }
        ok = ok && ul4_process_end(ph, a) == FOUR_AEC_NR_RES_OK;
        if (!ok) break;
        if (memcmp(a, b, (size_t)hop * sizeof(float)) != 0) ok = 0;
        for (int i = 0; i < hop; ++i) if (a[i] != 0.0f) nonzero = 1;
    }
    free(mics); free(far); free(far_hist); free(a); free(b);
    ul4_destroy(ph);
    ul4_destroy(cb);
    return ok && nonzero && (mode == PH_UNFETCHED || ran > 0);
}

static int test_phased_hop(void) {
    CHECK(phased_twin_matches(PH_RUN, 80),
          "phased hop with the caller's network equals the callback-driven pipeline byte for byte");
    CHECK(phased_twin_matches(PH_FAIL, 80), "phased: a failed run is the identity");
    CHECK(phased_twin_matches(PH_NAN, 80), "phased: a non-finite output bin is the identity");
    CHECK(phased_twin_matches(PH_OPEN, 80), "phased: a frame left open is the identity");
    CHECK(phased_twin_matches(PH_UNFETCHED, 80), "phased: a frame never fetched is the identity");
    return 1;
}

/* ============================================================================
 * 0. The example unit config is the stage's grid: the compiled ULCNet rate and FFT, linear
 *    context only (fuse is the post-beam stages' entry and is refused). A unit of the other
 *    grid builds, but the stage's bin count does not fit it, which is what the example's grid
 *    guard (unit_matches_stage) catches.
 * ========================================================================== */

static int test_example_unit_config_grid(void) {
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    AudioPipeline4ChConfig other = audio_pipeline_4ch_default_config(ULCNET_OTHER_SR);
    AudioPipeline4Ch* unit;
    SeSpecFrame frame;

    CHECK(cfg.core.sample_rate == ULCNET_SR && cfg.core.fft_size == ULCNET_N_FFT &&
          cfg.core.enable_post == 0,
          "example unit config: compiled ULCNet grid, enable_post = 0");
    unit = audio_pipeline_4ch_create(&cfg);
    CHECK(unit != NULL && unit_matches_stage(unit),
          "example unit: hop, bins and FFT size are the stage's");
    CHECK(audio_pipeline_4ch_sample_rate(unit) == ULCNET_SR,
          "example unit: sample rate is the compiled one");
    CHECK(audio_pipeline_4ch_fuse(unit, &frame) == FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "example unit: fuse is refused without the post lanes (enable_post = 0)");
    audio_pipeline_4ch_destroy(unit);

    other.core.fft_size = ULCNET_OTHER_N_FFT;
    other.core.enable_post = 0;
    unit = audio_pipeline_4ch_create(&other);
    CHECK(unit != NULL, "a unit of the other grid builds");
    CHECK(unit != NULL && !unit_matches_stage(unit),
          "the other grid's unit does not fit the stage");
    audio_pipeline_4ch_destroy(unit);
    CHECK(ul4_create(&other) == NULL, "the composition refuses the other grid");
    return 1;
}

/* ============================================================================
 * 4. Caller-pool construction: the unit through its queried descriptor (init_ex) and the
 *    stage in a stale block, each in its own pool, against the heap construction. Same
 *    stream, byte-identical output, nothing written past either pool, and the pools are
 *    reusable after destroy.
 * ========================================================================== */

static Ul4* ul4_in_pools(void* unit_pool, size_t unit_bytes, const AudioPipeline4ChConfig* cfg,
                         const AudioPipeline4ChMemReq* req, void* stage_pool,
                         size_t stage_bytes) {
    Ul4* p = (Ul4*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->unit = audio_pipeline_4ch_init_ex(unit_pool, unit_bytes, cfg, req);
    p->stage = ulcnet_nn_stage_init(stage_pool, stage_bytes, 0);
    if (!p->unit || !p->stage) {
        free(p);
        return NULL;
    }
    return p;
}

static int test_create_vs_pool_parity(void) {
    enum { EXTRA = 32, FRAMES = 40, TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    AudioPipeline4ChMemReq req;
    UlcnetModel model;
    Ul4 *heap, *pooled;
    unsigned char *unit_pool = NULL, *stage_pool = NULL;
    size_t stage_bytes = ulcnet_nn_stage_mem_size(0);
    float *mics, *far, *far_hist, *oh, *op;
    size_t k;
    int hop;

    cfg.bf_mu = 0.02f;
    CHECK(audio_pipeline_4ch_get_mem_requirements(&cfg, &req) == 0 && stage_bytes != 0,
          "pool queries succeed");
    CHECK(posix_memalign((void**)&unit_pool, req.alignment, (size_t)req.bytes + EXTRA) == 0 &&
              posix_memalign((void**)&stage_pool, 16, stage_bytes + EXTRA) == 0,
          "aligned caller pools allocate");
    memset(unit_pool, 0xa5, (size_t)req.bytes + EXTRA);
    memset(stage_pool, 0xa5, stage_bytes + EXTRA);

    heap = ul4_create(&cfg);
    pooled = ul4_in_pools(unit_pool, (size_t)req.bytes, &cfg, &req, stage_pool, stage_bytes);
    CHECK(heap != NULL && pooled != NULL,
          "heap and poisoned caller-pool construction both succeed");
    memset(&model, 0, sizeof(model));
    model.infer = phased_half_infer;
    model.io_descriptor = test_io_descriptor();
    CHECK(ul4_set_model(heap, &model) == 0 && ul4_set_model(pooled, &model) == 0,
          "the model installs on both");

    hop = ul4_hop_size(heap);
    mics = (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    oh = (float*)malloc((size_t)hop * sizeof(float));
    op = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(mics && far && far_hist && oh && op, "parity buffers allocate");
    g_rng = 0x7A11E5u;
    for (int frame = 0; frame < FRAMES; ++frame) {
        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, mics);
        CHECK(ul4_process_with_activity(heap, mics, far, 0, oh) == FOUR_AEC_NR_RES_OK &&
              ul4_process_with_activity(pooled, mics, far, 0, op) == FOUR_AEC_NR_RES_OK,
              "heap and pool instances process");
        CHECK(memcmp(oh, op, (size_t)hop * sizeof(float)) == 0,
              "heap/pool output is byte-identical");
    }
    for (k = 0; k < (size_t)EXTRA; ++k)
        if (unit_pool[(size_t)req.bytes + k] != 0xa5 || stage_pool[stage_bytes + k] != 0xa5) break;
    CHECK(k == (size_t)EXTRA, "init stays inside both queried pools");

    audio_pipeline_4ch_destroy(pooled->unit);
    free(pooled);
    pooled = ul4_in_pools(unit_pool, (size_t)req.bytes, &cfg, &req, stage_pool, stage_bytes);
    CHECK(pooled != NULL, "the caller pools are reusable after destroy");
    audio_pipeline_4ch_destroy(pooled->unit);
    free(pooled);
    ul4_destroy(heap);
    free(oh); free(op); free(far_hist); free(far); free(mics);
    free(unit_pool); free(stage_pool);
    return 1;
}

/* ============================================================================
 * 13. Recovery: any refused call is answered by resetting unit and stage together (what
 *     4ch_alignulcnet/main.c does). After that the stream restarts as on a fresh pair, byte
 *     for byte, with the model reset once by the recovery and then by its own alignment
 *     events only. A model whose gain depends on the calls since its last reset makes any
 *     state the recovery leaves behind visible in the output. The three refusals: the stage
 *     is begun twice in one hop (sequence error), the unit gets a non-finite sample
 *     (invalid argument), and a model frame is left open. The activity pattern keeps SRP and
 *     BF adapting.
 * ========================================================================== */

typedef struct StatefulModel {
    long calls_since_reset, infer_calls, resets;
} StatefulModel;

static int stateful_infer(
    void* user,
    const float* err_re, const float* err_im,
    const float* far_re, const float* far_im,
    float* out_re, float* out_im) {
    StatefulModel* m = (StatefulModel*)user;
    float g = 0.5f + 0.5f * expf(-(float)m->calls_since_reset / 50.0f);
    (void)far_re;
    (void)far_im;
    m->infer_calls += 1;
    m->calls_since_reset += 1;
    for (int k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = g * err_re[k];
        out_im[k] = g * err_im[k];
    }
    return 0;
}

static void stateful_reset(void* user) {
    StatefulModel* m = (StatefulModel*)user;
    m->calls_since_reset = 0;
    m->resets += 1;
}

enum { RC_STAGE_SEQUENCE, RC_UNIT_NONFINITE, RC_OPEN_FRAME };

/* The same scripted scene on every call: frames 0..frames-1, echo delay TRUE_DELAY. */
static int run_scene(Ul4* p, int frames, float* out_all) {
    enum { TRUE_DELAY = 400 };
    int hop = ul4_hop_size(p);
    float* mics = (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    float* far = (float*)malloc((size_t)hop * sizeof(float));
    float* far_hist = (float*)calloc((size_t)frames * hop, sizeof(float));
    int ok = mics && far && far_hist;
    g_rng = 0xABCDEF1u;
    for (int f = 0; ok && f < frames; ++f) {
        fill_echo_hop(f, hop, TRUE_DELAY, far_hist, far, mics);
        ok = ul4_process_with_activity(p, mics, far, (f / 7) % 2,
                                       out_all + (size_t)f * hop) == FOUR_AEC_NR_RES_OK;
    }
    free(mics); free(far); free(far_hist);
    return ok;
}

static int recovery_case(int refusal) {
    enum { PRE = 70, POST = 80 };
    AudioPipeline4ChConfig cfg = ulcnet_example_unit_config();
    Ul4 *a, *b;
    StatefulModel ma, mb;
    UlcnetModel model;
    UlcnetNnJob job;
    float *oa, *ob, *mics, *far;
    long resets_after_recovery, calls_after_recovery;
    int hop, equal = 1, nonzero = 0;

    a = ul4_create(&cfg);
    b = ul4_create(&cfg);
    CHECK(a != NULL && b != NULL, "recovery: both pairs create");
    hop = ul4_hop_size(a);
    memset(&ma, 0, sizeof(ma));
    memset(&mb, 0, sizeof(mb));
    memset(&model, 0, sizeof(model));
    model.infer = stateful_infer;
    model.reset = stateful_reset;
    model.io_descriptor = test_io_descriptor();
    model.user = &ma;
    CHECK(ul4_set_model(a, &model) == 0, "recovery: model installs on the recovering pair");
    model.user = &mb;
    CHECK(ul4_set_model(b, &model) == 0, "recovery: model installs on the fresh pair");
    oa = (float*)malloc((size_t)(PRE > POST ? PRE : POST) * hop * sizeof(float));
    ob = (float*)malloc((size_t)POST * hop * sizeof(float));
    mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    CHECK(oa && ob && mics && far, "recovery: buffers allocate");

    CHECK(run_scene(a, PRE, oa), "recovery: the first stretch runs");
    CHECK(ma.infer_calls > 0 && ma.resets >= 1,
          "recovery: the first stretch used the model and an alignment event reset it");

    switch (refusal) {
    case RC_STAGE_SEQUENCE:
        CHECK(ul4_process_begin(a, mics, far, 0) == FOUR_AEC_NR_RES_OK &&
              ul4_process_begin(a, mics, far, 0) == ULCNET_NN_STAGE_SEQUENCE_ERROR,
              "recovery: a second stage begin in one hop is refused");
        break;
    case RC_UNIT_NONFINITE:
        mics[3] = NAN;
        CHECK(ul4_process_with_activity(a, mics, far, 0, oa) == FOUR_AEC_NR_RES_INVALID_ARGUMENT,
              "recovery: a non-finite sample is refused by the unit");
        break;
    default:
        CHECK(ul4_process_begin(a, mics, far, 0) == FOUR_AEC_NR_RES_OK &&
              ul4_nn_begin(a, &job) == 1,
              "recovery: a model frame is opened and left open");
        break;
    }

    ul4_reset(a);
    resets_after_recovery = ma.resets;
    calls_after_recovery = ma.infer_calls;
    CHECK(ma.calls_since_reset == 0, "recovery: the paired reset reset the model");

    CHECK(run_scene(a, POST, oa) && run_scene(b, POST, ob),
          "recovery: the stream restarts on both pairs");
    equal = memcmp(oa, ob, (size_t)POST * hop * sizeof(float)) == 0;
    for (int i = 0; i < POST * hop; ++i) if (ob[i] != 0.0f) nonzero = 1;
    CHECK(equal, "recovery: output after the paired reset equals a fresh pair byte for byte");
    CHECK(nonzero, "recovery: the compared output is not silence");
    CHECK(ma.resets - resets_after_recovery == mb.resets &&
          ma.infer_calls - calls_after_recovery == mb.infer_calls,
          "recovery: the model sees the same resets and inferences as on a fresh pair");

    free(oa); free(ob); free(mics); free(far);
    ul4_destroy(a);
    ul4_destroy(b);
    return 1;
}

static int test_recovery_equals_fresh_pair(void) {
    CHECK(recovery_case(RC_STAGE_SEQUENCE), "recovery after a stage sequence error");
    CHECK(recovery_case(RC_UNIT_NONFINITE), "recovery after a refused unit hop");
    CHECK(recovery_case(RC_OPEN_FRAME), "recovery from an open model frame");
    return 1;
}

static int run_all_tests(void) {
    CHECK(test_example_unit_config_grid(),
          "example unit config: the stage's grid, linear context only");
    CHECK(test_identity_e2e(),
          "identity E2E / 1-hop timing contract");
    CHECK(test_counting_model_policy(),
          "counting model policy (stepping, resets, fail-open)");
    CHECK(test_fixed_first_alignment_resets_model(),
          "FIXED first aligned far resets the model without gating inference");
    CHECK(test_fixed_prelock_seam_is_raw_far(),
          "FIXED pre-lock seam serves the raw far byte for byte");
    CHECK(test_relock_same_delay_resets_model(),
          "relock on the same delay still resets the model (ALIGNED)");
    CHECK(test_create_vs_pool_parity(),
          "caller pools vs heap construction");
    CHECK(test_raw_far_timestamp_after_fixed_delay_lock(),
          "far timestamp: model receives raw far after AEC alignment");
    CHECK(test_error_timestamp_and_content(),
          "error timestamp: the mic impulse lands at +1 hop, as an impulse");
    CHECK(test_model_applies_unlocked(),
          "model applies without a delay lock");
    CHECK(test_nan_guard(),
          "NaN guard: non-finite model output never reaches the WOLA");
    CHECK(test_partial_write_guard(),
          "full-write contract: partial rc==0 frames are discarded");
    /* Runs on every grid. The far branch derives REPRIME_FRAMES exactly; the error branch only
     * bounds it, because its marker arrives through the AEC filter and carries a causal tail
     * that no reprime length can remove. See reprime_straddle_count's comment. */
    CHECK(test_reprime_straddle_derivation(),
          "identity-reprime length derived from the straddling frames");
    CHECK(test_reprime_behavior(),
          "identity reprime: exactly REPRIME identity frames per generation");
    CHECK(test_raw_descriptor_gate(),
          "production accepts only raw-far descriptors");
    CHECK(test_phased_hop(), "phased hop (begin / nn / end)");
    CHECK(test_recovery_equals_fresh_pair(),
          "recovery: the paired reset restarts the stream as a fresh pair");
    printf("All 4ch_ulcnet tests passed\n");
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
