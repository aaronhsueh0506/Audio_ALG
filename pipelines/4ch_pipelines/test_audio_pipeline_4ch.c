/**
 * test_audio_pipeline_4ch.c — complete four-channel pipeline tests.
 *
 * Mirrors test_audio_pipeline.c's public-API acceptance style while exercising
 * the additional SRP-PHAT/GSC stage at all three production signal grids.
 */

#include "audio_pipeline_4ch.h"

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

/* Regression test: in fixed-notebook mode
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
     * exactly mmse_lsa_retime_alpha(gsc_lambda, sr, hop_size*1) -- not
     * hop_size*4, which is what the pre-fix code effectively assumed by
     * scaling with the raw pre-forced gsc_adapt_interval. */
    expected_lambda = mmse_lsa_retime_alpha(
        cfg.gsc_lambda, cfg.core.sample_rate, hop_size * 1);
    CHECK(fabsf(actual_lambda - expected_lambda) < 1e-9f,
          "lambda retime uses the forced (not the requested) adapt_interval");

    /* Sanity check that this test would actually have caught the original
     * bug: the mis-scaled lambda (computed against the pre-forced interval
     * of 4) must visibly differ from the correct one. */
    buggy_lambda = mmse_lsa_retime_alpha(
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

    expected_lambda = mmse_lsa_retime_alpha(
        cfg.gsc_lambda, cfg.core.sample_rate, hop_size * 4);
    CHECK(fabsf(audio_pipeline_4ch_gsc_lambda(p) - expected_lambda) < 1e-9f,
          "lambda retime uses the requested adapt_interval in auto mode");

    audio_pipeline_4ch_destroy(p);
    return 1;
}

/*
 * Static-memory pool coverage for the descriptor-tier composition layer
 * (Phase A.3): poison/bounds, descriptor-staleness rejection, and a
 * heap-vs-pool byte-equal run. Mirrors 4ch_pipelines/test_4aec_nr_res.c's
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
    CHECK(run_static_parity(16000, 256),
          "16 kHz 256/128 static-memory pool parity");
    CHECK(run_static_parity(16000, 512),
          "16 kHz 512/256 static-memory pool parity");
    CHECK(run_static_parity(48000, 1024),
          "48 kHz 1024/512 static-memory pool parity");
    printf("All audio_pipeline_4ch tests passed (spatial=%s)\n",
           audio_pipeline_4ch_spatial_backend());
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
