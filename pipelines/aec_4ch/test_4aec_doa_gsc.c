/**
 * test_4aec_doa_gsc.c — complete four-channel spatial wrapper tests.
 *
 * Mirrors test_audio_pipeline.c's public-API acceptance style while exercising
 * the additional SRP-PHAT/GSC stage at all three production signal grids.
 */

#include "4aec_doa_gsc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            return 0;                                                  \
        }                                                              \
    } while (0)

static int run_grid(int sample_rate, int fft_size) {
    FourAecDoaGscConfig cfg =
        four_aec_doa_gsc_default_config(sample_rate);
    FourAecDoaGsc* p;
    FourAecDoaGscFrameInfo info;
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
    p = four_aec_doa_gsc_create(&cfg);
    CHECK(p != NULL, "create complete 4ch spatial pipeline");
    hop = four_aec_doa_gsc_hop_size(p);
    CHECK(hop == fft_size / 2,
          "rate-specific hop");
    CHECK(four_aec_doa_gsc_frame_size(p) == fft_size &&
          four_aec_doa_gsc_fft_size(p) == fft_size,
          "zero-padding-free frame/FFT contract");
    CHECK(four_aec_doa_gsc_n_freqs(p) == fft_size / 2 + 1,
          "spatial bin count follows selected FFT");
    CHECK(
        four_aec_doa_gsc_doa_sample_rate(p) ==
            sample_rate &&
        four_aec_doa_gsc_doa_frame_size(p) == fft_size &&
        four_aec_doa_gsc_doa_hop_size(p) == fft_size / 2 &&
        four_aec_doa_gsc_doa_fft_size(p) == fft_size,
        "DOA grid follows selected main AEC/NR/RES grid");
    CHECK(
        four_aec_doa_gsc_gsc_sample_rate(p) == sample_rate &&
        four_aec_doa_gsc_gsc_frame_size(p) == fft_size &&
        four_aec_doa_gsc_gsc_hop_size(p) == fft_size / 2 &&
        four_aec_doa_gsc_gsc_fft_size(p) == fft_size,
        "GSC grid follows selected main AEC/NR/RES grid");
    CHECK(four_aec_doa_gsc_matched_filter_count(p) == 1,
          "one shared matcher");
    CHECK(four_aec_doa_gsc_linear_aec_count(p) ==
              FOUR_AEC_NR_RES_CHANNELS,
          "four linear AEC lanes");
    CHECK(four_aec_doa_gsc_nr_count(p) == 1,
          "one post-beam NR");
    CHECK(four_aec_doa_gsc_post_res_count(p) == 1,
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
        CHECK(four_aec_doa_gsc_process_with_activity(
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

    four_aec_doa_gsc_reset(p);
    CHECK(four_aec_doa_gsc_process(
              p, microphones, far, output, &info) ==
              FOUR_AEC_NR_RES_OK,
          "fallback-VAD processing after reset");
    CHECK(info.frame_index == 0, "reset restarts wrapper frame index");
    CHECK(info.doa_analysis_frames == 1,
          "same-grid DOA resumes immediately after reset");

    free(output);
    free(far);
    free(microphones);
    four_aec_doa_gsc_destroy(p);
    return 1;
}

/* CHECK() early-returns 0 from whichever function it is lexically inside.
 * All top-level assertions therefore run inside this helper (not directly in
 * main()) so a failure returns 0 here -- a value main() explicitly turns
 * into a nonzero process exit -- instead of returning 0 from main() itself,
 * which the C runtime reports to the shell as a SUCCESSFUL exit despite the
 * "FAIL: ..." line already printed to stderr. */
static int run_all_tests(void) {
    FourAecDoaGscConfig invalid =
        four_aec_doa_gsc_default_config(16000);
    invalid.core.fft_size = 1024;
    CHECK(four_aec_doa_gsc_create(&invalid) == NULL,
          "16 kHz rejects cross-rate FFT 1024");
    invalid = four_aec_doa_gsc_default_config(48000);
    invalid.core.fft_size = 512;
    CHECK(four_aec_doa_gsc_create(&invalid) == NULL,
          "48 kHz rejects cross-rate FFT 512");
    invalid = four_aec_doa_gsc_default_config(16000);
    invalid.gsc_lambda = 1.0f + 1e-6f;
    CHECK(four_aec_doa_gsc_create(&invalid) == NULL,
          "GSC forgetting factor above 1.0 is rejected");
    invalid.gsc_lambda = 1.0f;
    {
        FourAecDoaGsc* boundary = four_aec_doa_gsc_create(&invalid);
        CHECK(boundary != NULL,
              "GSC forgetting factor of exactly 1.0 is still accepted");
        four_aec_doa_gsc_destroy(boundary);
    }
    CHECK(run_grid(16000, 256),
          "16 kHz 256/128 complete spatial pipeline");
    CHECK(run_grid(16000, 512),
          "16 kHz 512/256 complete spatial pipeline");
    CHECK(run_grid(48000, 1024),
          "48 kHz 1024/512 complete spatial pipeline");
    printf("All 4aec DOA/GSC tests passed (spatial=%s)\n",
           four_aec_doa_gsc_spatial_backend());
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
