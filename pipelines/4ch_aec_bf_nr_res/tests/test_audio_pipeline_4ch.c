/**
 * tests/test_audio_pipeline_4ch.c — complete four-channel pipeline tests.
 *
 * Mirrors test_audio_pipeline.c's public-API acceptance style while exercising
 * the additional SRP-PHAT/GSC stage at all three production signal grids.
 */

#include "audio_pipeline_4ch.h"
#include "conv_chain_4ch.h"

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

static int run_grid(int sample_rate, int fft_size) {
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(sample_rate);
    AudioPipeline4Ch* p;
    AudioPipeline4ChFrameInfo info;
    float* microphones;
    float* far;
    float* output;
    float phase = 0.0f;
    int hop;
    int doa_analysis_total = 0;

    cfg.core.fft_size = fft_size;
    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.4f;
    cfg.gsc_mu = 0.02f;
    p = audio_pipeline_4ch_create(&cfg);
    CHECK(p != NULL, "create complete 4ch spatial pipeline");
    hop = audio_pipeline_4ch_hop_size(p);
    CHECK(hop == fft_size / 2,
          "rate-specific hop");
    CHECK(audio_pipeline_4ch_frame_size(p) == fft_size &&
          audio_pipeline_4ch_fft_size(p) == fft_size,
          "zero-padding-free frame/FFT contract");
    CHECK(audio_pipeline_4ch_n_freqs(p) == fft_size / 2 + 1,
          "spatial bin count follows selected FFT");
    CHECK(
        audio_pipeline_4ch_doa_sample_rate(p) ==
            sample_rate &&
        audio_pipeline_4ch_doa_frame_size(p) == fft_size &&
        audio_pipeline_4ch_doa_hop_size(p) == fft_size / 2 &&
        audio_pipeline_4ch_doa_fft_size(p) == fft_size,
        "DOA grid follows selected main AEC/NR/RES grid");
    CHECK(
        audio_pipeline_4ch_gsc_sample_rate(p) == sample_rate &&
        audio_pipeline_4ch_gsc_frame_size(p) == fft_size &&
        audio_pipeline_4ch_gsc_hop_size(p) == fft_size / 2 &&
        audio_pipeline_4ch_gsc_fft_size(p) == fft_size,
        "GSC grid follows selected main AEC/NR/RES grid");
    CHECK(audio_pipeline_4ch_matched_filter_count(p) == 1,
          "one shared matcher");
    CHECK(audio_pipeline_4ch_linear_aec_count(p) ==
              FOUR_AEC_NR_RES_CHANNELS,
          "four linear AEC lanes");
    CHECK(audio_pipeline_4ch_nr_count(p) == 1,
          "one post-beam NR");
    CHECK(audio_pipeline_4ch_post_res_count(p) == 1,
          "one post-beam RES");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    output = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far && output, "allocate complete-pipeline buffers");

    for (int frame = 0; frame < 80; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float echo = 0.08f * sinf(phase);
            float near = frame >= 30
                ? 0.025f * sinf(phase * 1.73f + 0.2f) : 0.0f;
            far[i] = echo;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) +
                    near * (1.0f + 0.02f * ch);
            }
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(audio_pipeline_4ch_process_with_activity(
                  p, microphones, far,
                  frame >= 30, frame >= 30, NULL, output, &info) ==
                  FOUR_AEC_NR_RES_OK,
              "complete explicit-VAD frame");
        CHECK(info.frame_index == (uint64_t)frame,
              "monotonic spatial frame index");
        CHECK(info.doa_analysis_frames >= 0 &&
              info.doa_analysis_frames <= 1,
              "bounded DOA analyses per main hop");
        doa_analysis_total += info.doa_analysis_frames;
        CHECK(isfinite(info.doa_used_rad), "fixed GSC DOA must be finite");
        for (int i = 0; i < hop; ++i) {
            CHECK(isfinite(output[i]), "complete pipeline output finite");
        }
    }
    CHECK(doa_analysis_total == 80,
          "same-grid DOA consumes every main analysis frame");

    audio_pipeline_4ch_reset(p);
    CHECK(audio_pipeline_4ch_process(
              p, microphones, far, output, &info) ==
              FOUR_AEC_NR_RES_OK,
          "fallback-VAD processing after reset");
    CHECK(info.frame_index == 0, "reset restarts wrapper frame index");
    CHECK(info.doa_analysis_frames == 1,
          "same-grid DOA resumes immediately after reset");

    free(output);
    free(far);
    free(microphones);
    audio_pipeline_4ch_destroy(p);
    return 1;
}

/* Contract test: in fixed-notebook mode
 * (gsc_fixed_mode && gsc_fixed_align_notebook), gsc_create() forces the
 * actual GSC RLS update cadence to 1 hop regardless of the configured
 * gsc_adapt_interval -- kept for baseline-matching against the reference
 * notebook. The lambda retime scaling computed in
 * audio_pipeline_4ch_create() MUST use that same forced cadence, not the
 * raw pre-forced gsc_adapt_interval, or lambda ends up calibrated for a
 * slower wall-clock update period than what is actually running (too
 * aggressive/timid RLS forgetting relative to the intended design). */
static int test_gsc_fixed_notebook_lambda_matches_forced_cadence(void) {
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(16000);
    AudioPipeline4Ch* p;
    int hop_size;
    int actual_interval;
    float actual_lambda;
    float expected_lambda;
    float buggy_lambda;

    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_align_notebook = 1;
    cfg.gsc_adapt_interval = 4; /* caller asks for a slower cadence... */
    cfg.gsc_lambda = 0.98f;

    p = audio_pipeline_4ch_create(&cfg);
    CHECK(p != NULL, "fixed-notebook cadence test: create pipeline");

    hop_size = audio_pipeline_4ch_gsc_hop_size(p);
    actual_interval = audio_pipeline_4ch_gsc_effective_adapt_interval(p);
    actual_lambda = audio_pipeline_4ch_gsc_lambda(p);

    /* ...but fixed_align_notebook forces the real GSC cadence to every hop
     * (1), regardless of the requested 4. */
    CHECK(actual_interval == 1,
          "fixed-notebook mode forces the actual GSC cadence to 1");

    /* Lambda must be retimed assuming that SAME forced interval (1), i.e.
     * exactly anr_retime_alpha(gsc_lambda, sr, hop_size*1) -- not
     * hop_size*4, which scaling with the raw pre-forced gsc_adapt_interval
     * would give. */
    expected_lambda = anr_retime_alpha(
        cfg.gsc_lambda, cfg.core.sample_rate, hop_size * 1);
    CHECK(fabsf(actual_lambda - expected_lambda) < 1e-9f,
          "lambda retime uses the forced (not the requested) adapt_interval");

    /* Sanity check that this test can fail: the lambda scaled against the
     * pre-forced interval of 4 must visibly differ from the correct one. */
    buggy_lambda = anr_retime_alpha(
        cfg.gsc_lambda, cfg.core.sample_rate, hop_size * 4);
    CHECK(fabsf(actual_lambda - buggy_lambda) > 1e-4f,
          "forced-cadence lambda differs from the old mis-scaled lambda");

    audio_pipeline_4ch_destroy(p);
    return 1;
}

/* Companion check: outside fixed-notebook mode, a configured
 * gsc_adapt_interval > 1 is NOT forced down, so the effective interval used
 * for both the actual GSC cadence and the lambda retime must equal the
 * requested value itself (guards the other branch of
 * gsc_effective_adapt_interval() from regressing). */
static int test_gsc_auto_mode_respects_configured_interval(void) {
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(16000);
    AudioPipeline4Ch* p;
    int hop_size;
    float expected_lambda;

    cfg.gsc_adapt_interval = 4;
    cfg.gsc_lambda = 0.98f;

    p = audio_pipeline_4ch_create(&cfg);
    CHECK(p != NULL, "auto-mode cadence test: create pipeline");

    hop_size = audio_pipeline_4ch_gsc_hop_size(p);
    CHECK(audio_pipeline_4ch_gsc_effective_adapt_interval(p) == 4,
          "auto mode keeps the requested adapt_interval unforced");

    expected_lambda = anr_retime_alpha(
        cfg.gsc_lambda, cfg.core.sample_rate, hop_size * 4);
    CHECK(fabsf(audio_pipeline_4ch_gsc_lambda(p) - expected_lambda) < 1e-9f,
          "lambda retime uses the requested adapt_interval in auto mode");

    audio_pipeline_4ch_destroy(p);
    return 1;
}

/*
 * audio_pipeline_4ch_process() must reject a bad call before auto_vad() --
 * which unconditionally folds its input into the noise_power EMA -- can see
 * it: NULL far_reference/output and non-finite input must not reach the VAD
 * state. A NaN noise_power would make `power >= noise_power * ratio` false
 * for every future frame (any comparison against NaN is false), silently
 * disabling VAD forever. Checked here by running two identically-fed
 * pipeline instances side by side: only one ("attacked") also receives a
 * batch of calls that must be rejected (NULL far_reference, NULL output, a
 * NaN mic buffer); if any of those leaks into VAD state, "attacked" diverges
 * from "clean" on every frame afterward despite otherwise-identical input.
 */
static int test_rejected_calls_preserve_vad_state(void) {
    const int sample_rate = 16000;
    const int fft_size = 256;
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(sample_rate);
    AudioPipeline4Ch *clean, *attacked;
    int hop;
    float *mic, *far, *out_clean, *out_attacked, *nan_mic, *nan_far, *inf_far,
        *quiet_mic;
    float phase = 0.0f;
    AudioPipeline4ChFrameInfo info_clean, info_attacked, throwaway;

    cfg.core.fft_size = fft_size;
    clean = audio_pipeline_4ch_create(&cfg);
    attacked = audio_pipeline_4ch_create(&cfg);
    CHECK(clean && attacked, "create clean/attacked pipeline pair");
    hop = audio_pipeline_4ch_hop_size(clean);

    mic = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    out_clean = (float*)malloc((size_t)hop * sizeof(float));
    out_attacked = (float*)malloc((size_t)hop * sizeof(float));
    nan_mic = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    nan_far = (float*)malloc((size_t)hop * sizeof(float));
    inf_far = (float*)malloc((size_t)hop * sizeof(float));
    quiet_mic = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    CHECK(mic && far && out_clean && out_attacked && nan_mic
              && nan_far && inf_far && quiet_mic,
          "allocate VAD-contamination test buffers");

    /* Quiet warm-up so noise_power settles away from its init value
     * identically on both instances before the attack. */
    for (int frame = 0; frame < 5; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(audio_pipeline_4ch_process(
                  clean, mic, far, out_clean, &info_clean) ==
                  FOUR_AEC_NR_RES_OK,
              "clean warm-up frame");
        CHECK(audio_pipeline_4ch_process(
                  attacked, mic, far, out_attacked, &info_attacked) ==
                  FOUR_AEC_NR_RES_OK,
              "attacked warm-up frame");
    }

    /* One loud frame on BOTH instances arms vad_hangover to the same
     * nonzero value on each -- a quiet warm-up alone never exercises the
     * `if (speech) p->vad_hangover -= 1;` branch a leaking rejected call
     * would still execute (hangover stays 0 the whole time otherwise, so
     * `speech` there is always false and that line never runs). */
    for (int i = 0; i < hop; ++i) {
        float v = 0.05f * sinf(phase);
        far[i] = 0.0f;
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(audio_pipeline_4ch_process(
              clean, mic, far, out_clean, &info_clean) ==
              FOUR_AEC_NR_RES_OK,
          "clean hangover-arming frame");
    CHECK(audio_pipeline_4ch_process(
              attacked, mic, far, out_attacked, &info_attacked) ==
              FOUR_AEC_NR_RES_OK,
          "attacked hangover-arming frame");
    CHECK(info_clean.vad_raw == 1 && info_attacked.vad_raw == 1,
          "loud tone actually arms the hangover on both instances");

    memcpy(nan_mic, mic,
           (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    nan_mic[0] = NAN;
    memcpy(nan_far, far, (size_t)hop * sizeof(float));
    nan_far[0] = NAN;
    memcpy(inf_far, far, (size_t)hop * sizeof(float));
    inf_far[0] = (float)INFINITY;
    CHECK(audio_pipeline_4ch_process(
              attacked, mic, NULL, out_attacked, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "NULL far_reference is rejected");
    CHECK(audio_pipeline_4ch_process(
              attacked, mic, far, NULL, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "NULL output is rejected");
    CHECK(audio_pipeline_4ch_process(
              attacked, nan_mic, far, out_attacked, &throwaway) !=
              FOUR_AEC_NR_RES_OK,
          "NaN mic input is rejected, not silently processed");
    /* far_reference is never read by auto_vad() itself (only
     * microphones_interleaved is), so on perfectly finite mic data a
     * non-finite far passes auto_vad()'s own guard entirely and would only
     * be caught by the deeper finite check inside four_aec_front_process()
     * -- by which point auto_vad() would already have mutated
     * noise_power/vad_hangover from real, finite mic energy. Both NaN and
     * Inf are exercised since they're rejected by different mechanisms
     * downstream (isfinite() catches both, but this pins the check at the
     * source rather than relying on that).
     *
     * Deliberately QUIET mic content for these two calls, NOT the loud
     * arming signal still sitting in `mic`: auto_vad() on the loud signal
     * would just re-arm vad_hangover to the exact value it already has
     * (indistinguishable from not running at all), so a missing
     * far_reference check would go unnoticed. Quiet
     * content instead takes the EMA's "silence" branch, a real, different
     * noise_power update if auto_vad() incorrectly runs. */
    for (int i = 0; i < hop; ++i) {
        float v = 0.003f * sinf(phase + 1.7f);
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            quiet_mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
    }
    CHECK(audio_pipeline_4ch_process(
              attacked, quiet_mic, nan_far, out_attacked, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "NaN far_reference is rejected before touching VAD state");
    CHECK(audio_pipeline_4ch_process(
              attacked, quiet_mic, inf_far, out_attacked, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "Inf far_reference is rejected before touching VAD state");

    /* Resume byte-identical valid input on both -- QUIET this time, so the
     * hangover counter actually counts down to 0 across the run instead of
     * being re-armed to the same fixed value on every hop (a loud frame
     * here would mask a 1-frame hangover discrepancy the moment it hits,
     * since `speech` re-arms hangover unconditionally on the speech
     * branch). A leak in any of the three rejected calls above -- including
     * a hangover off-by-one -- desyncs "attacked" from "clean" somewhere in
     * this walk-down, even though both see exactly the same frames. */
    for (int frame = 0; frame < 40; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(audio_pipeline_4ch_process(
                  clean, mic, far, out_clean, &info_clean) ==
                  FOUR_AEC_NR_RES_OK,
              "clean post-attack frame");
        CHECK(audio_pipeline_4ch_process(
                  attacked, mic, far, out_attacked, &info_attacked) ==
                  FOUR_AEC_NR_RES_OK,
              "attacked post-attack frame");
        CHECK(info_clean.vad_raw == info_attacked.vad_raw &&
              info_clean.vad_out == info_attacked.vad_out,
              "rejected calls must not desync VAD decisions from an "
              "identically-fed clean instance");
        CHECK(memcmp(out_clean, out_attacked,
                     (size_t)hop * sizeof(float)) == 0,
              "rejected calls must not desync output from an "
              "identically-fed clean instance");
    }

    /* Positive check, not just clean==attacked: guards against both
     * instances being identically broken (e.g. if noise_power corruption
     * were somehow shared or the loud tone were mistuned). */
    for (int i = 0; i < hop; ++i) {
        float v = 0.05f * sinf(phase);
        far[i] = 0.0f;
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(audio_pipeline_4ch_process(
              clean, mic, far, out_clean, &info_clean) ==
              FOUR_AEC_NR_RES_OK,
          "clean final loud frame");
    CHECK(info_clean.vad_raw == 1,
          "loud tone is actually detected as speech");

    free(mic); free(far); free(out_clean); free(out_attacked); free(nan_mic);
    free(nan_far); free(inf_far); free(quiet_mic);
    audio_pipeline_4ch_destroy(clean);
    audio_pipeline_4ch_destroy(attacked);
    return 1;
}

/*
 * Static-memory pool coverage for the descriptor-tier composition layer
 * (Phase A.3): poison/bounds, descriptor-staleness rejection, and a
 * heap-vs-pool byte-equal run. Mirrors 4ch_aec_bf_nr_res/test_4aec_nr_res.c's
 * run_static_parity() -- same 0xa5 poison + EXTRA trailing-byte bounds check,
 * same 8-point descriptor mutation block (descriptor_version/layout_version/
 * backend_id/build_flags_hash/alignment/reserved/bytes/"stale layout despite
 * larger cached bytes"), same destroy-then-reinit pool-reuse check -- plus a
 * heap vs. pool side-by-side run across many synthetic hops comparing both
 * the mono output hop and every scalar field of AudioPipeline4ChFrameInfo.
 *
 * Deliberately does NOT scan the whole poisoned pool for "rejected calls
 * never touch it" (test_spatial_third_party.c's smaller SRP/GSC pools do;
 * this composite pool can be several MB at 48 kHz/1024, and
 * test_4aec_nr_res.c's own run_static_parity() -- the direct precedent for
 * this descriptor tier -- likewise only checks the EXTRA trailing bytes
 * after the ACCEPTING init, not a full-pool scan around every rejection).
 */
static int run_static_parity(int sample_rate, int fft_size) {
    enum { EXTRA = 32, FRAMES = 40 };
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(sample_rate);
    AudioPipeline4ChMemReq req;
    AudioPipeline4ChMemReq stale;
    AudioPipeline4Ch* heap;
    AudioPipeline4Ch* stat;
    unsigned char* pool = NULL;
    float* microphones;
    float* far;
    float* heap_out;
    float* stat_out;
    float phase_step;
    size_t pool_bytes;
    size_t k;
    int hop;
    int frame;

    cfg.core.fft_size = fft_size;
    cfg.gsc_fixed_mode = 1;
    cfg.gsc_fixed_doa_rad = 0.35f;
    cfg.gsc_mu = 0.02f;

    CHECK(audio_pipeline_4ch_get_mem_requirements(&cfg, &req) == 0,
          "static memory requirement query succeeds");
    CHECK(req.layout_version == AUDIO_PIPELINE_4CH_LAYOUT_VERSION &&
              AUDIO_PIPELINE_4CH_LAYOUT_VERSION == 15u,
          "the queried descriptor publishes the current carve layout");
    CHECK(req.bytes <= (uint64_t)SIZE_MAX,
          "static memory requirement fits size_t");
    pool_bytes = (size_t)req.bytes + (size_t)EXTRA;

    CHECK(posix_memalign(
              (void**)&pool, (size_t)req.alignment, pool_bytes) == 0 && pool,
          "aligned caller pool allocates");
    memset(pool, 0xa5, pool_bytes);

    CHECK(audio_pipeline_4ch_init(pool + 1, (size_t)req.bytes, &cfg) == NULL,
          "static init rejects a misaligned pool");
    CHECK(audio_pipeline_4ch_init(pool, (size_t)req.bytes - 1u, &cfg) == NULL,
          "static init rejects an undersized pool");

    stale = req;
    stale.descriptor_version += 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a stale descriptor ABI");
    stale = req;
    stale.layout_version += 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a stale carve layout");
    stale = req;
    stale.backend_id = 99u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a backend mismatch");
    stale = req;
    stale.build_flags_hash ^= 0xffffffffu;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a build-layout mismatch");
    stale = req;
    stale.alignment *= 2u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects an alignment mismatch");
    stale = req;
    stale.reserved = 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a corrupt reserved field");
    stale = req;
    stale.bytes -= 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects stale descriptor bytes");
    stale = req;
    stale.layout_version -= 1u;
    stale.bytes += 4096u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a stale layout even when its cached bytes "
          "are larger than current (byte count fitting must never "
          "substitute for layout/hash agreement)");

    stat = audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    heap = audio_pipeline_4ch_create(&cfg);
    CHECK(stat != NULL && heap != NULL,
          "poisoned caller pool and heap construction both succeed");

    for (k = 0; k < (size_t)EXTRA; ++k) {
        if (pool[(size_t)req.bytes + k] != 0xa5) break;
    }
    CHECK(k == (size_t)EXTRA, "static init stays inside the queried pool");

    hop = audio_pipeline_4ch_hop_size(stat);
    CHECK(hop == audio_pipeline_4ch_hop_size(heap),
          "heap and pool instances agree on hop size");
    CHECK(hop == fft_size / 2, "rate-specific hop");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    heap_out = (float*)malloc((size_t)hop * sizeof(float));
    stat_out = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far && heap_out && stat_out,
          "static parity buffers allocate");

    phase_step = (float)(2.0 * M_PI * 700.0 / sample_rate);
    for (frame = 0; frame < FRAMES; ++frame) {
        AudioPipeline4ChFrameInfo heap_info;
        AudioPipeline4ChFrameInfo stat_info;
        int vad = frame >= 15;
        int i;
        int ch;

        memset(&heap_info, 0, sizeof(heap_info));
        memset(&stat_info, 0, sizeof(stat_info));

        for (i = 0; i < hop; ++i) {
            int64_t absolute = (int64_t)frame * hop + i;
            float phase = phase_step * (float)absolute;
            float echo = 0.08f * sinf(phase);
            float near = vad ? 0.02f * sinf(phase * 1.6f + 0.3f) : 0.0f;
            far[i] = echo;
            for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) + near * (1.0f + 0.02f * ch);
            }
        }

        CHECK(audio_pipeline_4ch_process_with_activity(
                  heap, microphones, far, vad, vad, NULL,
                  heap_out, &heap_info) == FOUR_AEC_NR_RES_OK,
              "heap instance processes a synthetic hop");
        CHECK(audio_pipeline_4ch_process_with_activity(
                  stat, microphones, far, vad, vad, NULL,
                  stat_out, &stat_info) == FOUR_AEC_NR_RES_OK,
              "pool instance processes a synthetic hop");
        CHECK(memcmp(heap_out, stat_out, (size_t)hop * sizeof(float)) == 0,
              "heap/pool mono output is byte-identical");
        CHECK(heap_info.frame_index == stat_info.frame_index,
              "heap/pool frame_index matches");
        CHECK(heap_info.delay.delay_samples ==
                      stat_info.delay.delay_samples &&
                  heap_info.delay.confidence == stat_info.delay.confidence &&
                  heap_info.delay.solid == stat_info.delay.solid &&
                  heap_info.delay.changed == stat_info.delay.changed &&
                  heap_info.delay.estimator_calls ==
                      stat_info.delay.estimator_calls &&
                  heap_info.delay.estimator_updates ==
                      stat_info.delay.estimator_updates,
              "heap/pool delay state matches");
        CHECK((isnan(heap_info.doa_raw_rad) &&
                   isnan(stat_info.doa_raw_rad)) ||
                  heap_info.doa_raw_rad == stat_info.doa_raw_rad,
              "heap/pool doa_raw_rad matches");
        CHECK((isnan(heap_info.doa_smooth_rad) &&
                   isnan(stat_info.doa_smooth_rad)) ||
                  heap_info.doa_smooth_rad == stat_info.doa_smooth_rad,
              "heap/pool doa_smooth_rad matches");
        CHECK((isnan(heap_info.doa_used_rad) &&
                   isnan(stat_info.doa_used_rad)) ||
                  heap_info.doa_used_rad == stat_info.doa_used_rad,
              "heap/pool doa_used_rad matches");
        CHECK(heap_info.vad_raw == stat_info.vad_raw &&
                  heap_info.vad_out == stat_info.vad_out &&
                  heap_info.gsc_adaptive == stat_info.gsc_adaptive &&
                  heap_info.doa_analysis_frames ==
                      stat_info.doa_analysis_frames,
              "heap/pool remaining frame info fields match");
    }

    audio_pipeline_4ch_destroy(stat);
    CHECK(audio_pipeline_4ch_hop_size(stat) == -1,
          "destroy marks a caller-pool instance inactive");
    audio_pipeline_4ch_destroy(stat);
    CHECK(audio_pipeline_4ch_hop_size(stat) == -1,
          "caller-pool destroy is idempotent");
    stat = audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(stat != NULL, "caller pool is reusable after destroy");

    audio_pipeline_4ch_destroy(stat);
    audio_pipeline_4ch_destroy(heap);
    free(stat_out);
    free(heap_out);
    free(far);
    free(microphones);
    free(pool);
    return 1;
}

/* CHECK() early-returns 0 from whichever function it is lexically inside.
 * All top-level assertions therefore run inside this helper (not directly in
 * main()) so a failure returns 0 here -- a value main() explicitly turns
 * into a nonzero process exit -- instead of returning 0 from main() itself,
 * which the C runtime reports to the shell as a SUCCESSFUL exit despite the
 * "FAIL: ..." line already printed to stderr. */
/* ---------------------------------------------------------------------------
 * Post-beam stages carved by this wrapper: strength setters, sizing, and the
 * equivalence of the wrapper's chain with the reference chain written over the
 * front end (conv_chain_4ch.h), on a bypassed GSC so both see the same error
 * spectrum (lane 0) and the identity weights.
 * ------------------------------------------------------------------------ */

static void feed_hops(AudioPipeline4Ch* p, int hops, int first, float* out_last) {
    int hop = audio_pipeline_4ch_hop_size(p);
    int sr = audio_pipeline_4ch_sample_rate(p);
    float* mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    float* far = (float*)calloc((size_t)hop, sizeof(float));
    float* out = (float*)calloc((size_t)hop, sizeof(float));
    int h, i, ch;
    for (h = 0; h < hops; ++h) {
        for (i = 0; i < hop; ++i) {
            float t = (float)((first + h) * hop + i);
            float r = 0.6f * sinf(0.06f * t);
            far[i] = r;
            for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + (size_t)ch] =
                    0.5f * r + 0.02f * sinf(0.017f * t);
        }
        (void)sr;
        if (audio_pipeline_4ch_process_with_activity(p, mics, far, 0, 0, NULL, out, NULL) !=
            FOUR_AEC_NR_RES_OK)
            break;
    }
    if (out_last) memcpy(out_last, out, (size_t)hop * sizeof(float));
    free(mics); free(far); free(out);
}

static int test_runtime_strength(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
    AudioPipeline4Ch* p = audio_pipeline_4ch_create(&cfg);
    float live0 = 0.0f, target0 = 0.0f, live1 = 0.0f, target1 = 0.0f;

    CHECK(p != NULL, "strength: pipeline created");
    feed_hops(p, 30, 0, NULL);
    CHECK(audio_pipeline_4ch_post_split_floor(p, &live0, &target0) == 0,
          "strength: the post-beam floor is readable");

    /* The setter moves the post-beam suppressor -- the gain that actually
     * multiplies this pipeline's output. */
    CHECK(audio_pipeline_4ch_set_aec_preset(p, AEC_PRESET_AGGRESSIVE, 0.0f) == 0,
          "strength: aggressive accepted");
    CHECK(audio_pipeline_4ch_post_split_floor(p, &live1, &target1) == 0 &&
              live1 != live0 && live1 == target1,
          "strength: the post-beam floor moved and landed immediately");

    /* It survives a reset: a setter that changed only the live floor would be
     * silently reverted right here. */
    audio_pipeline_4ch_reset(p);
    {
        float live2 = 0.0f, target2 = 0.0f;
        CHECK(audio_pipeline_4ch_post_split_floor(p, &live2, &target2) == 0 &&
                  live2 == live1 && target2 == target1,
              "strength: the new floor survives a reset");
    }

    /* A ramp is genuinely in flight before it lands. */
    feed_hops(p, 20, 0, NULL);
    CHECK(audio_pipeline_4ch_set_aec_preset(p, AEC_PRESET_MILD, 100.0f) == 0,
          "strength: ramped retarget accepted");
    {
        float lv = 0.0f, tg = 0.0f;
        feed_hops(p, 2, 20, NULL);
        audio_pipeline_4ch_post_split_floor(p, &lv, &tg);
        CHECK(lv != tg, "strength: the ramp is mid-flight after 2 hops");
        feed_hops(p, 60, 22, NULL);
        audio_pipeline_4ch_post_split_floor(p, &lv, &tg);
        CHECK(lv == tg, "strength: the ramp lands");
    }

    CHECK(audio_pipeline_4ch_set_aec_preset(NULL, AEC_PRESET_MILD, 0.0f) == -1,
          "strength: NULL pipeline refused");
    CHECK(audio_pipeline_4ch_set_aec_preset(p, (AecPreset)77, 0.0f) == -1,
          "strength: out-of-enum preset refused");
    CHECK(audio_pipeline_4ch_set_aec_preset(p, AEC_PRESET_MILD, 60001.0f) == -1,
          "strength: out-of-range ramp_ms refused");
    CHECK(audio_pipeline_4ch_set_nr_mode(p, (AnrMode)42) == -1,
          "strength: out-of-enum NR mode refused");
    /* The NR setter recomposes THIS pipeline's configuration: the canonical
     * convenience setter would be refused on this instance (its L differs). */
    CHECK(audio_pipeline_4ch_set_nr_mode(p, ANR_MODE_AGGRESSIVE) == 0,
          "strength: NR mode change accepted through the recomposed target");
    audio_pipeline_4ch_destroy(p);
    return 1;
}

static size_t pool_bytes(const AudioPipeline4ChConfig* cfg) {
    AudioPipeline4ChMemReq req;
    return audio_pipeline_4ch_get_mem_requirements(cfg, &req) == 0 ? (size_t)req.bytes : 0;
}

static int test_stage_sizing_and_switches(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g;
    for (g = 0; g < 3; ++g) {
        AudioPipeline4ChConfig both = audio_pipeline_4ch_default_config(grids[g][0]);
        AudioPipeline4ChConfig no_nr, no_res, neither;
        AecConfig res_cfg;
        AnrConfig nr_cfg;
        size_t full, res_b, nr_b, unity_b;
        int n_freqs;
        AudioPipeline4Ch *p;

        both.core.fft_size = grids[g][1];
        no_nr = both; no_nr.core.enable_nr = 0;
        no_res = both; no_res.core.enable_res = 0;
        neither = both; neither.core.enable_nr = 0; neither.core.enable_res = 0;
        CHECK(four_aec_nr_res_stage_configs(&both.core, &res_cfg, &nr_cfg) == 0,
              "sizing: the stage configurations derive");
        res_b = (aec_res_get_mem_size(&res_cfg) + 15u) & ~(size_t)15u;
        nr_b = (se_anr_stage_mem_size(&nr_cfg) + 15u) & ~(size_t)15u;
        n_freqs = grids[g][1] / 2 + 1;
        unity_b = (((size_t)n_freqs * sizeof(float)) + 15u) & ~(size_t)15u;
        full = pool_bytes(&both);
        CHECK(full > 0 && full - pool_bytes(&no_nr) == nr_b - unity_b,
              "sizing: NR off omits exactly the aligned denoiser and adds the unity gain vector");
        CHECK(full - pool_bytes(&no_res) == res_b - unity_b,
              "sizing: RES off omits exactly the aligned AecRes and adds the unity gain vector");
        CHECK(full - pool_bytes(&neither) == res_b + nr_b - unity_b,
              "sizing: both off omit both blocks and carve the unity gain vector once");

        p = audio_pipeline_4ch_create(&no_nr);
        CHECK(p && audio_pipeline_4ch_nr_count(p) == 0 && audio_pipeline_4ch_post_res_count(p) == 1 &&
              audio_pipeline_4ch_set_nr_mode(p, ANR_MODE_AGGRESSIVE) == -1,
              "switches: NR off leaves the RES and refuses the NR setter");
        audio_pipeline_4ch_destroy(p);
        p = audio_pipeline_4ch_create(&no_res);
        CHECK(p && audio_pipeline_4ch_nr_count(p) == 1 && audio_pipeline_4ch_post_res_count(p) == 0 &&
              audio_pipeline_4ch_set_aec_preset(p, AEC_PRESET_MILD, 0.0f) == -1 &&
              audio_pipeline_4ch_post_split_floor(p, NULL, NULL) == -1,
              "switches: RES off leaves the denoiser and refuses the suppressor entry points");
        audio_pipeline_4ch_destroy(p);
    }
    return 1;
}

/* With the GSC bypassed the beamformed error is lane 0's error spectrum and
 * the effective weights are [1,0,0,0]; the reference chain fed the same two
 * must reproduce the wrapper bit for bit, for every combination of the three
 * switches the post stages offer. */
static int test_wrapper_equals_reference_chain(void) {
    int mode, grid;
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    for (grid = 0; grid < 3; ++grid) {
        for (mode = 0; mode < 8; ++mode) {
            AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(grids[grid][0]);
            AudioPipeline4Ch* wrap;
            ConvChain* ref;
            Complex* weights;
            float *mics, *far, *out_w, *out_r;
            int hop, n, h, i, ch, k, identical = 1, nonzero = 0, ok = 1;
            char label[128];

            cfg.core.fft_size = grids[grid][1];
            cfg.core.enable_nr = mode & 1;
            cfg.core.enable_res = (mode >> 1) & 1;
            cfg.core.enable_cng = (mode >> 2) & 1;
            cfg.gsc_enable = 0;
            wrap = audio_pipeline_4ch_create(&cfg);
            ref = conv_create(&cfg.core);
            snprintf(label, sizeof(label), "reference chain: %d Hz/%d nr=%d res=%d cng=%d build",
                     grids[grid][0], grids[grid][1], mode & 1, (mode >> 1) & 1, (mode >> 2) & 1);
            CHECK(wrap && ref, label);
            if (!wrap || !ref) { audio_pipeline_4ch_destroy(wrap); conv_destroy(ref); continue; }
            hop = audio_pipeline_4ch_hop_size(wrap);
            n = audio_pipeline_4ch_n_freqs(wrap);
            mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
            far = (float*)calloc((size_t)hop, sizeof(float));
            out_w = (float*)calloc((size_t)hop, sizeof(float));
            out_r = (float*)calloc((size_t)hop, sizeof(float));
            weights = (Complex*)calloc((size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n, sizeof(Complex));
            for (k = 0; k < n; ++k) weights[k].r = 1.0f;
            for (h = 0; h < 120 && ok; ++h) {
                for (i = 0; i < hop; ++i) {
                    float t = (float)(h * hop + i);
                    float r = 0.5f * sinf(0.05f * t) + 0.1f * sinf(0.31f * t);
                    far[i] = r;
                    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                        mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + (size_t)ch] =
                            0.4f * r + 0.03f * sinf(0.017f * t + (float)ch);
                }
                if (audio_pipeline_4ch_process_with_activity(wrap, mics, far, 0, 0, NULL, out_w, NULL) !=
                        FOUR_AEC_NR_RES_OK ||
                    four_aec_front_process(ref->core, mics, far, &ref->front) != FOUR_AEC_NR_RES_OK ||
                    conv_post(ref, weights, ref->front.linear_spectra[0], out_r) != FOUR_AEC_NR_RES_OK) {
                    ok = 0;
                    break;
                }
                if (memcmp(out_w, out_r, (size_t)hop * sizeof(float)) != 0) identical = 0;
                for (i = 0; i < hop; ++i) if (out_r[i] != 0.0f) nonzero = 1;
            }
            snprintf(label, sizeof(label),
                     "reference chain: %d Hz/%d nr=%d res=%d cng=%d wrapper output is bit-identical",
                     grids[grid][0], grids[grid][1], mode & 1, (mode >> 1) & 1, (mode >> 2) & 1);
            CHECK(ok && identical && nonzero, label);
            free(mics); free(far); free(out_w); free(out_r); free(weights);
            audio_pipeline_4ch_destroy(wrap);
            conv_destroy(ref);
        }
    }
    return 1;
}

static int run_all_tests(void) {
    AudioPipeline4ChConfig invalid =
        audio_pipeline_4ch_default_config(16000);
    invalid.core.fft_size = 1024;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "16 kHz rejects cross-rate FFT 1024");
    invalid = audio_pipeline_4ch_default_config(48000);
    invalid.core.fft_size = 512;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "48 kHz rejects cross-rate FFT 512");
    invalid = audio_pipeline_4ch_default_config(16000);
    invalid.core.enable_post = 0;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "standard wrapper rejects the core-only pre profile");
    invalid = audio_pipeline_4ch_default_config(16000);
    invalid.core.enable_nr = 0;
    {
        AudioPipeline4Ch* res_only = audio_pipeline_4ch_create(&invalid);
        CHECK(res_only != NULL,
              "standard wrapper accepts the RES-only post profile");
        audio_pipeline_4ch_destroy(res_only);
    }
    invalid = audio_pipeline_4ch_default_config(16000);
    invalid.core.enable_res = 0;
    {
        AudioPipeline4Ch* nr_only = audio_pipeline_4ch_create(&invalid);
        CHECK(nr_only != NULL,
              "standard wrapper accepts the NR-only post profile");
        audio_pipeline_4ch_destroy(nr_only);
    }
    invalid = audio_pipeline_4ch_default_config(16000);
    invalid.gsc_lambda = 1.0f + 1e-6f;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "GSC forgetting factor above 1.0 is rejected");
    invalid.gsc_lambda = 1.0f;
    {
        AudioPipeline4Ch* boundary = audio_pipeline_4ch_create(&invalid);
        CHECK(boundary != NULL,
              "GSC forgetting factor of exactly 1.0 is still accepted");
        audio_pipeline_4ch_destroy(boundary);
    }
    CHECK(run_grid(16000, 256),
          "16 kHz 256/128 complete spatial pipeline");
    CHECK(run_grid(16000, 512),
          "16 kHz 512/256 complete spatial pipeline");
    CHECK(run_grid(48000, 1024),
          "48 kHz 1024/512 complete spatial pipeline");
    CHECK(test_gsc_fixed_notebook_lambda_matches_forced_cadence(),
          "GSC fixed-notebook lambda/forced-cadence match test");
    CHECK(test_gsc_auto_mode_respects_configured_interval(),
          "GSC auto-mode configured-interval test");
    CHECK(test_rejected_calls_preserve_vad_state(),
          "rejected calls (NULL args, NaN input) must not contaminate VAD state");
    CHECK(run_static_parity(16000, 256),
          "16 kHz 256/128 static-memory pool parity");
    CHECK(run_static_parity(16000, 512),
          "16 kHz 512/256 static-memory pool parity");
    CHECK(run_static_parity(48000, 1024),
          "48 kHz 1024/512 static-memory pool parity");
    CHECK(test_runtime_strength(), "post-beam strength setters");
    CHECK(test_stage_sizing_and_switches(),
          "post-beam stage sizing and switch behaviour");
    CHECK(test_wrapper_equals_reference_chain(),
          "wrapper chain equals the reference chain over the front end");
    printf("All audio_pipeline_4ch tests passed (spatial=%s)\n",
           audio_pipeline_4ch_spatial_backend());
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
