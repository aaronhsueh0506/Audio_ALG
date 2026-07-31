/**
 * Structural and lifecycle tests for 4aec_nr_res.h.
 *
 * The test intentionally uses equal weights as an external-beamformer stand
 * in. The library itself must never choose those weights in production.
 */

#include "4aec_nr_res.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, label)                                            \
    do {                                                                   \
        if (condition) {                                                   \
            printf("PASS: %s\n", label);                                   \
        } else {                                                           \
            printf("FAIL: %s\n", label);                                   \
            failures += 1;                                                 \
        }                                                                  \
    } while (0)

static int all_finite(const float* values, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static void fill_inputs(float* microphones, float* ref, int hop,
                        int sample_rate, int frame_index) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        int64_t absolute = (int64_t)frame_index * hop + i;
        float phase = 2.0f * 3.14159265358979323846f *
                      440.0f * (float)absolute / (float)sample_rate;
        ref[i] = 0.08f * sinf(phase);
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            float near = 0.01f * sinf(
                phase * (1.0f + 0.05f * (float)ch));
            microphones[
                i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                ref[i] * (0.4f + 0.05f * (float)ch) + near;
        }
    }
}

static int run_rate(int sample_rate) {
    FourAecNrResConfig cfg;
    FourAecNrRes* pipeline;
    FourAecNrResPreFrame pre;
    FourAecNrResFrameToken stale;
    float* microphones;
    float* ref;
    float* out;
    Complex* weights;
    int hop;
    int n_freqs;
    int frame;
    int ch;
    int k;
    int rc;

    four_aec_nr_res_config_defaults(&cfg, sample_rate);
    cfg.enable_cng = 0;
    pipeline = four_aec_nr_res_create(&cfg);
    CHECK(pipeline != NULL, "create supported rate");
    if (!pipeline) return 0;

    hop = four_aec_nr_res_hop_size(pipeline);
    n_freqs = four_aec_nr_res_n_freqs(pipeline);
    CHECK(hop == (sample_rate == 16000 ? 256 : 512),
          "rate-specific hop");
    CHECK(four_aec_nr_res_fft_size(pipeline) == 2 * hop,
          "FFT is two hops");
    CHECK(four_aec_nr_res_sample_rate(pipeline) == sample_rate,
          "sample-rate accessor");
    CHECK(four_aec_nr_res_matched_filter_count(pipeline) == 1,
          "one shared matcher");
    CHECK(four_aec_nr_res_linear_aec_count(pipeline) == 4,
          "four linear AEC filters");
    CHECK(four_aec_nr_res_nr_count(pipeline) == 1,
          "one mono NR");
    CHECK(four_aec_nr_res_post_res_count(pipeline) == 1,
          "one post-beam RES");

    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs,
        sizeof(Complex));
    CHECK(microphones && ref && out && weights, "test buffers allocate");
    if (!microphones || !ref || !out || !weights) {
        free(microphones);
        free(ref);
        free(out);
        free(weights);
        four_aec_nr_res_destroy(pipeline);
        return 0;
    }

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k) {
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
        }
    }

    for (frame = 0; frame < 3; ++frame) {
        fill_inputs(microphones, ref, hop, sample_rate, frame);
        rc = four_aec_nr_res_process_pre(
            pipeline, microphones, ref, &pre);
        CHECK(rc == FOUR_AEC_NR_RES_OK, "pre stage succeeds");
        if (rc != FOUR_AEC_NR_RES_OK) break;
        CHECK(pre.linear_interleaved != NULL &&
              pre.hop_size == hop &&
              pre.n_channels == FOUR_AEC_NR_RES_CHANNELS,
              "pre frame shape contract");
        CHECK(all_finite(
                  pre.linear_interleaved,
                  hop * FOUR_AEC_NR_RES_CHANNELS),
              "four linear outputs finite");

        if (frame == 0) {
            FourAecNrResPreFrame duplicate;
            CHECK(four_aec_nr_res_process_pre(
                      pipeline, microphones, ref, &duplicate) ==
                      FOUR_AEC_NR_RES_SEQUENCE_ERROR,
                  "second pre frame is rejected while one is pending");
        }

        stale = pre.token;
        rc = four_aec_nr_res_process_post(
            pipeline, &pre.token, weights, out);
        CHECK(rc == FOUR_AEC_NR_RES_OK, "post NR/RES stage succeeds");
        CHECK(all_finite(out, hop), "mono output is finite");
        CHECK(four_aec_nr_res_process_post(
                  pipeline, &stale, weights, out) ==
                  FOUR_AEC_NR_RES_SEQUENCE_ERROR,
              "post token cannot be replayed");
    }

    fill_inputs(microphones, ref, hop, sample_rate, 4);
    CHECK(four_aec_nr_res_process_pre(
              pipeline, microphones, ref, &pre) == FOUR_AEC_NR_RES_OK,
          "pre before reset succeeds");
    stale = pre.token;
    four_aec_nr_res_reset(pipeline);
    CHECK(four_aec_nr_res_process_post(
              pipeline, &stale, weights, out) ==
              FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "reset invalidates in-flight token");

    fill_inputs(microphones, ref, hop, sample_rate, 5);
    CHECK(four_aec_nr_res_process_pre(
              pipeline, microphones, ref, &pre) == FOUR_AEC_NR_RES_OK,
          "pre after reset succeeds");
    memset(weights, 0,
           (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs *
           sizeof(Complex));
    CHECK(four_aec_nr_res_process_post(
              pipeline, &pre.token, weights, out) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "all-zero beamformer weights are rejected");
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k) {
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
        }
    }
    CHECK(four_aec_nr_res_process_post(
              pipeline, &pre.token, weights, out) ==
              FOUR_AEC_NR_RES_OK,
          "pending frame may retry after invalid weights");

    free(microphones);
    free(ref);
    free(out);
    free(weights);
    four_aec_nr_res_destroy(pipeline);
    return 1;
}

static void test_invalid_configs(void) {
    FourAecNrResConfig cfg;

    four_aec_nr_res_config_defaults(&cfg, 8000);
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "8 kHz is outside the 4-channel contract");

    four_aec_nr_res_config_defaults(&cfg, 16000);
    cfg.fft_size = 1024;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "cross-rate FFT grid is rejected");

    four_aec_nr_res_config_defaults(&cfg, 16000);
    cfg.capture_proxy_channel = 4;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "invalid capture proxy is rejected");
}

int main(void) {
    test_invalid_configs();
    run_rate(16000);
    run_rate(48000);

    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All 4aec_nr_res tests passed\n");
    return 0;
}
