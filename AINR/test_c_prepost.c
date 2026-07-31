#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_process.h"
#include "DeepFilterNet3/dfn3_process.h"
#include "GTCRN/gtcrn_process.h"

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
        return 0;                                                           \
    }                                                                       \
} while (0)

static uint64_t hash_bytes(uint64_t h, const void* data, size_t bytes)
{
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static float signal_sample(int64_t index, int sample_rate)
{
    if (index < 0) return 0.0f;
    return 0.31f * sinf((float)(2.0 * M_PI * 437.0 * index / sample_rate)) +
           0.11f * cosf((float)(2.0 * M_PI * 1733.0 * index / sample_rate));
}

static int all_finite(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static float stream_spec_re(int frame, int bin)
{
    return 0.01f * (float)(frame + 1) + 0.00003f * (float)bin;
}

static float stream_spec_im(int frame, int bin)
{
    return -0.006f * (float)(frame + 1) + 0.00002f * (float)bin;
}

static float stream_mask(int frame)
{
    return 0.35f + 0.03f * (float)(frame % 5);
}

static float stream_alpha(int frame)
{
    return 0.25f + 0.05f * (float)(frame % 4);
}

static float stream_tap(int tap)
{
    static const float taps[5] = {0.11f, -0.07f, 0.43f, 0.29f, 0.17f};
    return taps[tap];
}

static int test_dfn_stream_alignment(void)
{
    static DFN2State dfn2;
    static DFN3State dfn3;
    float spec2_re[DFN2_N_BINS], spec2_im[DFN2_N_BINS];
    float spec3_re[DFN3_N_BINS], spec3_im[DFN3_N_BINS];
    float out2_re[DFN2_N_BINS], out2_im[DFN2_N_BINS];
    float out3_re[DFN3_N_BINS], out3_im[DFN3_N_BINS];
    float mask2[DFN2_N_ERB] = {0}, mask3[DFN3_N_ERB] = {0};
    float coef2[DFN2_DF_BINS][DFN2_DF_ORDER][2] = {{{0}}};
    float coef3[DFN3_DF_BINS][DFN3_DF_ORDER][2] = {{{0}}};
    long long output_frame = -1;

    dfn2_state_init(&dfn2);
    dfn3_state_init(&dfn3);
    for (int k = 0; k < DFN2_DF_BINS; ++k)
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap)
            coef2[k][tap][0] = stream_tap(tap);
    for (int k = 0; k < DFN3_DF_BINS; ++k)
        for (int tap = 0; tap < DFN3_DF_ORDER; ++tap)
            coef3[k][tap][0] = stream_tap(tap);

    memset(spec2_re, 0, sizeof(spec2_re));
    memset(spec2_im, 0, sizeof(spec2_im));
    CHECK(dfn2_compose_stream(
              &dfn2, spec2_re, spec2_im, 1, mask2,
              &coef2[0][0][0], 0.5f, 0.0f,
              out2_re, out2_im, NULL) == -1,
          "DFN2 rejects a head before model lookahead is satisfied");

    for (int wall = 0; wall < 14; ++wall) {
        int head = wall - DFN2_MASK_LOOKAHEAD;
        int expected_target = head - DFN2_DF_LOOKAHEAD;
        for (int k = 0; k < DFN2_N_BINS; ++k) {
            spec2_re[k] = stream_spec_re(wall, k);
            spec2_im[k] = stream_spec_im(wall, k);
        }
        if (head >= 0)
            for (int b = 0; b < DFN2_N_ERB; ++b)
                mask2[b] = stream_mask(head);
        {
            int valid = dfn2_compose_stream(
                &dfn2, spec2_re, spec2_im, head >= 0,
                head >= 0 ? mask2 : NULL,
                head >= 0 ? &coef2[0][0][0] : NULL,
                head >= 0 ? stream_alpha(head) : 0.0f,
                0.0f, out2_re, out2_im, &output_frame);
            CHECK(valid == (expected_target >= 0),
                  "DFN2 warmup equals mask+DF lookahead");
            if (valid == 1) {
                CHECK(output_frame == expected_target,
                      "DFN2 reports the delayed target frame");
                for (int k = 0; k < DFN2_N_BINS; ++k) {
                    float expected_re;
                    float expected_im;
                    float target_mask = stream_mask(expected_target);
                    if (k < DFN2_DF_BINS) {
                        float filtered_re = 0.0f;
                        float filtered_im = 0.0f;
                        float alpha = stream_alpha(expected_target);
                        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
                            int source = expected_target - DFN2_DF_HISTORY + tap;
                            if (source >= 0) {
                                float source_mask = stream_mask(source);
                                filtered_re += stream_spec_re(source, k) *
                                               source_mask * stream_tap(tap);
                                filtered_im += stream_spec_im(source, k) *
                                               source_mask * stream_tap(tap);
                            }
                        }
                        expected_re = alpha * filtered_re + (1.0f - alpha) *
                            stream_spec_re(expected_target, k) * target_mask;
                        expected_im = alpha * filtered_im + (1.0f - alpha) *
                            stream_spec_im(expected_target, k) * target_mask;
                    } else {
                        expected_re = stream_spec_re(expected_target, k) * target_mask;
                        expected_im = stream_spec_im(expected_target, k) * target_mask;
                    }
                    CHECK(fabsf(out2_re[k] - expected_re) < 3e-6f &&
                          fabsf(out2_im[k] - expected_im) < 3e-6f,
                          "DFN2 heads align with their cascade spectra");
                }
            }
        }
    }

    for (int wall = 0; wall < 14; ++wall) {
        int expected_target = wall - DFN3_MASK_LOOKAHEAD;
        for (int k = 0; k < DFN3_N_BINS; ++k) {
            spec3_re[k] = stream_spec_re(wall, k);
            spec3_im[k] = stream_spec_im(wall, k);
        }
        if (expected_target >= 0)
            for (int b = 0; b < DFN3_N_ERB; ++b)
                mask3[b] = stream_mask(expected_target);
        {
            int valid = dfn3_compose_stream(
                &dfn3, spec3_re, spec3_im, expected_target >= 0,
                expected_target >= 0 ? mask3 : NULL,
                expected_target >= 0 ? &coef3[0][0][0] : NULL,
                0.0f, out3_re, out3_im, &output_frame);
            CHECK(valid == (expected_target >= 0),
                  "DFN3 warmup equals parallel-branch lookahead");
            if (valid == 1) {
                CHECK(output_frame == expected_target,
                      "DFN3 reports the delayed target frame");
                for (int k = 0; k < DFN3_N_BINS; ++k) {
                    float expected_re = 0.0f;
                    float expected_im = 0.0f;
                    if (k < DFN3_DF_BINS) {
                        for (int tap = 0; tap < DFN3_DF_ORDER; ++tap) {
                            int source = expected_target - DFN3_DF_HISTORY + tap;
                            if (source >= 0) {
                                expected_re += stream_spec_re(source, k) *
                                               stream_tap(tap);
                                expected_im += stream_spec_im(source, k) *
                                               stream_tap(tap);
                            }
                        }
                    } else {
                        expected_re = stream_spec_re(expected_target, k) *
                                      stream_mask(expected_target);
                        expected_im = stream_spec_im(expected_target, k) *
                                      stream_mask(expected_target);
                    }
                    CHECK(fabsf(out3_re[k] - expected_re) < 3e-6f &&
                          fabsf(out3_im[k] - expected_im) < 3e-6f,
                          "DFN3 heads align with their parallel spectra");
                }
            }
        }
    }
    return 1;
}

static int test_dfn2(uint64_t* digest)
{
    static DFN2State state;
    float input[DFN2_HOP_LEN];
    float spec_re[DFN2_N_BINS], spec_im[DFN2_N_BINS];
    float previous_re[DFN2_N_BINS] = {0}, previous_im[DFN2_N_BINS] = {0};
    float enhanced_re[DFN2_N_BINS], enhanced_im[DFN2_N_BINS];
    float output[DFN2_HOP_LEN];
    float erb[DFN2_N_ERB], feature_spec[2 * DFN2_DF_BINS];
    float mask[DFN2_N_ERB];
    float coefs[DFN2_DF_BINS][DFN2_DF_ORDER][2] = {{{0}}};
    float max_spectral_error = 0.0f;

    dfn2_state_init(&state);
    for (int b = 0; b < DFN2_N_ERB; ++b) mask[b] = 1.0f;
    for (int k = 0; k < DFN2_DF_BINS; ++k)
        coefs[k][DFN2_DF_HISTORY][0] = 1.0f;

    for (int frame = 0; frame < 24; ++frame) {
        for (int i = 0; i < DFN2_HOP_LEN; ++i)
            input[i] = signal_sample((int64_t)frame * DFN2_HOP_LEN + i,
                                     DFN2_SR);
        dfn2_analysis(&state, input, spec_re, spec_im);
        dfn2_compute_features(&state, spec_re, spec_im, erb, feature_spec);
        CHECK(all_finite(erb, DFN2_N_ERB), "DFN2 finite ERB features");
        CHECK(all_finite(feature_spec, 2 * DFN2_DF_BINS),
              "DFN2 finite complex features");
        if (dfn2_compose(&state, spec_re, spec_im, mask,
                         &coefs[0][0][0], 1.0f,
                         enhanced_re, enhanced_im)) {
            for (int k = 0; k < DFN2_N_BINS; ++k) {
                float er = fabsf(enhanced_re[k] - previous_re[k]);
                float ei = fabsf(enhanced_im[k] - previous_im[k]);
                if (er > max_spectral_error) max_spectral_error = er;
                if (ei > max_spectral_error) max_spectral_error = ei;
            }
            dfn2_apply_atten_lim(previous_re, previous_im,
                                 enhanced_re, enhanced_im, -100.0f);
            dfn2_synthesis(&state, enhanced_re, enhanced_im, output);
            CHECK(all_finite(output, DFN2_HOP_LEN), "DFN2 finite WOLA output");
            *digest = hash_bytes(*digest, erb, sizeof(erb));
            *digest = hash_bytes(*digest, feature_spec, sizeof(feature_spec));
            *digest = hash_bytes(*digest, output, sizeof(output));
        }
        memcpy(previous_re, spec_re, sizeof(spec_re));
        memcpy(previous_im, spec_im, sizeof(spec_im));
    }
    CHECK(max_spectral_error < 2e-6f,
          "DFN2 lookahead ring returns the target spectrum");
    return 1;
}

static int test_dfn3(uint64_t* digest)
{
    static DFN3State state;
    float input[DFN3_HOP_LEN];
    float spec_re[DFN3_N_BINS], spec_im[DFN3_N_BINS];
    float previous_re[DFN3_N_BINS] = {0}, previous_im[DFN3_N_BINS] = {0};
    float enhanced_re[DFN3_N_BINS], enhanced_im[DFN3_N_BINS];
    float output[DFN3_HOP_LEN];
    float erb[DFN3_N_ERB], feature_spec[2 * DFN3_DF_BINS];
    float mask[DFN3_N_ERB];
    float coefs[DFN3_DF_BINS][DFN3_DF_ORDER][2] = {{{0}}};
    float max_spectral_error = 0.0f;

    dfn3_state_init(&state);
    for (int b = 0; b < DFN3_N_ERB; ++b) mask[b] = 1.0f;
    for (int k = 0; k < DFN3_DF_BINS; ++k)
        coefs[k][DFN3_DF_HISTORY][0] = 1.0f;

    for (int frame = 0; frame < 24; ++frame) {
        for (int i = 0; i < DFN3_HOP_LEN; ++i)
            input[i] = signal_sample((int64_t)frame * DFN3_HOP_LEN + i,
                                     DFN3_SR);
        dfn3_analysis(&state, input, spec_re, spec_im);
        dfn3_compute_features(&state, spec_re, spec_im, erb, feature_spec);
        CHECK(all_finite(erb, DFN3_N_ERB), "DFN3 finite ERB features");
        CHECK(all_finite(feature_spec, 2 * DFN3_DF_BINS),
              "DFN3 finite complex features");
        if (dfn3_compose(&state, spec_re, spec_im, mask,
                         &coefs[0][0][0], enhanced_re, enhanced_im)) {
            for (int k = 0; k < DFN3_N_BINS; ++k) {
                float er = fabsf(enhanced_re[k] - previous_re[k]);
                float ei = fabsf(enhanced_im[k] - previous_im[k]);
                if (er > max_spectral_error) max_spectral_error = er;
                if (ei > max_spectral_error) max_spectral_error = ei;
            }
            dfn3_apply_atten_lim(previous_re, previous_im,
                                 enhanced_re, enhanced_im, -100.0f);
            dfn3_synthesis(&state, enhanced_re, enhanced_im, output);
            CHECK(all_finite(output, DFN3_HOP_LEN), "DFN3 finite WOLA output");
            *digest = hash_bytes(*digest, erb, sizeof(erb));
            *digest = hash_bytes(*digest, feature_spec, sizeof(feature_spec));
            *digest = hash_bytes(*digest, output, sizeof(output));
        }
        memcpy(previous_re, spec_re, sizeof(spec_re));
        memcpy(previous_im, spec_im, sizeof(spec_im));
    }
    CHECK(max_spectral_error < 2e-6f,
          "DFN3 lookahead ring returns the target spectrum");
    return 1;
}

static int test_gtcrn(uint64_t* digest)
{
    GTCRNProcessState state;
    float input[GTCRN_HOP_LEN], previous[GTCRN_HOP_LEN] = {0};
    float output[GTCRN_HOP_LEN];
    float spectrum[GTCRN_N_BINS][2];
    float max_error = 0.0f;

    gtcrn_process_init(&state);
    for (int frame = 0; frame < 24; ++frame) {
        for (int i = 0; i < GTCRN_HOP_LEN; ++i)
            input[i] = signal_sample((int64_t)frame * GTCRN_HOP_LEN + i,
                                     GTCRN_SR);
        gtcrn_analysis(&state, input, spectrum);
        gtcrn_synthesis(&state, spectrum, output);
        CHECK(all_finite(&spectrum[0][0], 2 * GTCRN_N_BINS),
              "GTCRN finite spectrum");
        CHECK(all_finite(output, GTCRN_HOP_LEN), "GTCRN finite WOLA output");
        if (frame >= 2) {
            for (int i = 0; i < GTCRN_HOP_LEN; ++i) {
                float error = fabsf(output[i] - previous[i]);
                if (error > max_error) max_error = error;
            }
        }
        *digest = hash_bytes(*digest, spectrum, sizeof(spectrum));
        *digest = hash_bytes(*digest, output, sizeof(output));
        memcpy(previous, input, sizeof(input));
    }
    CHECK(max_error < 2e-4f, "GTCRN analysis/synthesis steady-state unity");
    return 1;
}

static int run_all_tests(void)
{
    uint64_t digest = UINT64_C(1469598103934665603);
    CHECK(test_dfn_stream_alignment(), "DFN streaming head alignment");
    CHECK(test_dfn2(&digest), "DFN2 C pre/post");
    CHECK(test_dfn3(&digest), "DFN3 C pre/post");
    CHECK(test_gtcrn(&digest), "GTCRN C pre/post");
    printf("backend=%s/%s/%s digest=%016llx\n",
           dfn2_simd_backend(), dfn3_simd_backend(), gtcrn_simd_backend(),
           (unsigned long long)digest);
    return 1;
}

int main(int argc, char** argv)
{
    if (argc == 2 && strcmp(argv[1], "--stream-only") == 0)
        return test_dfn_stream_alignment() ? EXIT_SUCCESS : EXIT_FAILURE;
    return run_all_tests() ? EXIT_SUCCESS : EXIT_FAILURE;
}
