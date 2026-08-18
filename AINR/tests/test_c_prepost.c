/* C pre/post-processing contract test. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_process.h"
#include "DeepFilterNet2/dfn2_model_io.h"
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

/* Reference ERB matrices for the DFN tests, built from the shipped
 * 48-kHz/1024/32 border table with the exact triangular construction the
 * runtime used to derive on-device. The runtime now only consumes
 * caller-loaded matrices (erb_fwd.bin/erb_inv.bin); reproducing the
 * construction HERE keeps every existing golden value valid while pinning
 * the new pointer plumbing. */
static const int dfn_test_borders[32] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
    50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
    325, 378, 440, 513
};
static float dfn_test_fwd[513][32];
static float dfn_test_inv[32][513];
static void dfn_test_build_erb(void)
{
    int segment = 0;
    memset(dfn_test_fwd, 0, sizeof(dfn_test_fwd));
    memset(dfn_test_inv, 0, sizeof(dfn_test_inv));
    for (int k = 0; k < 513; ++k) {
        int lo, hi, width, offset;
        float right, left, fleft, fright;
        while (segment + 1 < 31 && k >= dfn_test_borders[segment + 1])
            ++segment;
        lo = dfn_test_borders[segment];
        hi = dfn_test_borders[segment + 1];
        width = hi - lo;
        offset = k - lo;
        right = (float)offset / (float)width;
        left = 1.0f - right;
        fleft = left; fright = right;
        if (segment == 0) fleft *= 2.0f;
        if (segment + 1 == 31) fright *= 2.0f;
        dfn_test_fwd[k][segment] = fleft;
        dfn_test_fwd[k][segment + 1] = fright;
        dfn_test_inv[segment][k] = left;
        dfn_test_inv[segment + 1][k] = right;
    }
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

    dfn_test_build_erb();
    {
        static FftHandle* fft2;
        static FftHandle* fft3;
        if (!fft2) fft2 = fft_create(DFN2_N_FFT);
        if (!fft3) fft3 = fft_create(DFN3_N_FFT);
        dfn2_state_init(&dfn2, fft2);
        dfn3_state_init(&dfn3, fft3);
    }
    dfn2_set_erb_matrices(&dfn2, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    dfn3_set_erb_matrices(&dfn3, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
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

static int test_dfn2_model_io(void)
{
    static DFN2ModelIOState state;
    static DFN2ModelIOState committed;
    static float encoder_next[DFN2_MODEL_ENCODER_GRU_LAYERS]
                             [DFN2_MODEL_GRU_HIDDEN];
    static float erb_next[DFN2_MODEL_ERB_GRU_LAYERS]
                         [DFN2_MODEL_GRU_HIDDEN];
    static float df_next[DFN2_MODEL_DF_GRU_LAYERS]
                        [DFN2_MODEL_GRU_HIDDEN];
    static float pathway_next[DFN2_MODEL_ENCODER_CHANNELS]
                              [DFN2_MODEL_DF_PATHWAY_HISTORY]
                              [DFN2_DF_BINS];
    float erb[DFN2_N_ERB];
    float spec[2][DFN2_DF_BINS];

    dfn2_model_io_init(&state);
    for (int frame = 0; frame < 3; ++frame) {
        for (int band = 0; band < DFN2_N_ERB; ++band)
            erb[band] = (float)(100 * frame + band);
        for (int channel = 0; channel < 2; ++channel)
            for (int bin = 0; bin < DFN2_DF_BINS; ++bin)
                spec[channel][bin] =
                    (float)(1000 * frame + 100 * channel + bin);
        CHECK(dfn2_model_io_push_features(&state, erb, spec) == (frame != 0),
              "DFN2 model window warms up for one lookahead frame");
    }
    CHECK(state.erb_window[0][7] == 7.0f &&
          state.erb_window[1][7] == 107.0f &&
          state.erb_window[2][7] == 207.0f,
          "DFN2 ERB model window keeps [t-1,t,t+1]");
    CHECK(state.spec_window[1][0][9] == 109.0f &&
          state.spec_window[1][2][9] == 2109.0f,
          "DFN2 complex model window preserves channel-major layout");

    memset(encoder_next, 0x3c, sizeof(encoder_next));
    memset(erb_next, 0x4d, sizeof(erb_next));
    memset(df_next, 0x5e, sizeof(df_next));
    memset(pathway_next, 0x6f, sizeof(pathway_next));
    CHECK(dfn2_model_io_commit_state(&state, encoder_next, erb_next, df_next,
                                     pathway_next) == 0 &&
          memcmp(state.encoder_gru_hidden, encoder_next,
                 sizeof(encoder_next)) == 0 &&
          memcmp(state.erb_gru_hidden, erb_next, sizeof(erb_next)) == 0 &&
          memcmp(state.df_gru_hidden, df_next, sizeof(df_next)) == 0 &&
          memcmp(state.df_convp_history, pathway_next,
                 sizeof(pathway_next)) == 0,
          "DFN2 model state outputs become the next invocation inputs");

    committed = state;
    memset(encoder_next, 0x41, sizeof(encoder_next));
    memset(erb_next, 0x41, sizeof(erb_next));
    memset(df_next, 0x41, sizeof(df_next));
    memset(pathway_next, 0x41, sizeof(pathway_next));
    pathway_next[DFN2_MODEL_ENCODER_CHANNELS - 1]
                [DFN2_MODEL_DF_PATHWAY_HISTORY - 1]
                [DFN2_DF_BINS - 1] = NAN;
    CHECK(dfn2_model_io_commit_state(&state, encoder_next, erb_next, df_next,
                                     pathway_next) != 0,
          "DFN2 commit refuses a non-finite state batch");
    CHECK(memcmp(&state, &committed, sizeof(state)) == 0,
          "DFN2 refusal preserves every previously committed state byte");
    CHECK(dfn2_model_io_commit_state(NULL, encoder_next, erb_next, df_next,
                                     pathway_next) != 0,
          "DFN2 commit refuses a null destination");
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

    dfn_test_build_erb();
    {
        static FftHandle* fft_handle;
        if (!fft_handle) fft_handle = fft_create(DFN2_N_FFT);
        dfn2_state_init(&state, fft_handle);
    }
    dfn2_set_erb_matrices(&state, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
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

    dfn_test_build_erb();
    {
        static FftHandle* fft_handle;
        if (!fft_handle) fft_handle = fft_create(DFN3_N_FFT);
        dfn3_state_init(&state, fft_handle);
    }
    dfn3_set_erb_matrices(&state, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
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

    {
        static FftHandle* fft_handle;
        if (!fft_handle) fft_handle = fft_create(GTCRN_N_FFT);
        gtcrn_process_init(&state, fft_handle);
    }
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

static int test_gtcrn_model_state(void)
{
    static GTCRNModelState state;
    static GTCRNModelState next;
    static GTCRNModelState committed;
    const float* h_tra[GTCRN_MODEL_TRA_GRUS];
    const float* h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS];
    const float* broken[GTCRN_MODEL_TRA_GRUS];
    int i;
    for (i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
        h_tra[i] = &next.h_tra[i][0][0][0];
    }
    for (i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
        h_dpgrnn[i] = &next.h_dpgrnn[i][0][0][0];
    }
    {
        static float spec_in[GTCRN_N_BINS][2];
        static float mag[GTCRN_MODEL_ERB_BANDS];
        static float re_b[GTCRN_MODEL_ERB_BANDS];
        static float im_b[GTCRN_MODEL_ERB_BANDS];
        static float mask_erb[GTCRN_MODEL_ERB_BANDS][2];
        static float enhanced[GTCRN_N_BINS][2];
        static float fwd[GTCRN_MODEL_ERB_HIGH_BINS]
                        [GTCRN_MODEL_ERB_HIGH_BANDS];
        static float inv[GTCRN_MODEL_ERB_HIGH_BANDS]
                        [GTCRN_MODEL_ERB_HIGH_BINS];
        int b;
        /* Synthetic caller-loaded matrices: the values live in the .bin the
         * loader owns, so this test pins the WIRING (which row scales which
         * band/bin), while the exporter's round-trip test pins the values. */
        for (b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b)
            fwd[35][b] = 0.25f + 0.001f * (float)b;
        for (b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b)
            inv[b][35] = 0.125f + 0.002f * (float)b;
        memset(spec_in, 0, sizeof(spec_in));
        spec_in[10][0] = 3.0f; spec_in[10][1] = 4.0f;   /* low: passthrough */
        spec_in[GTCRN_MODEL_ERB_KEPT + 35][0] = 2.0f;   /* one high bin     */
        gtcrn_model_input(spec_in, &fwd[0][0], mag, re_b, im_b);
        CHECK(fabsf(mag[10] - 5.0f) < 1e-6f &&
              re_b[10] == 3.0f && im_b[10] == 4.0f &&
              mag[0] == sqrtf(1e-12f),
              "GTCRN host feature: low bins pass [mag, re, im] through");
        {
            int wired = 1;
            for (b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b) {
                if (fabsf(re_b[GTCRN_MODEL_ERB_KEPT + b] -
                          fwd[35][b] * 2.0f) > 1e-6f) wired = 0;
            }
            CHECK(wired,
                  "GTCRN host feature: high bins band through the "
                  "caller-loaded forward matrix");
        }
        memset(mask_erb, 0, sizeof(mask_erb));
        mask_erb[10][0] = 1.0f;                          /* unity low mask  */
        mask_erb[GTCRN_MODEL_ERB_KEPT + 7][0] = 0.5f;    /* one high band   */
        gtcrn_model_output(mask_erb, &inv[0][0], spec_in, enhanced);
        CHECK(enhanced[10][0] == 3.0f && enhanced[10][1] == 4.0f,
              "GTCRN host output: unity low-band CRM reproduces the "
              "spectrum bin");
        CHECK(fabsf(enhanced[GTCRN_MODEL_ERB_KEPT + 35][0] -
                    2.0f * 0.5f * inv[7][35]) < 1e-6f,
              "GTCRN host output: high bands expand through the "
              "caller-loaded inverse matrix");
    }
    gtcrn_model_state_init(&state);
    CHECK(state.conv_cache[1][15][15][32] == 0.0f &&
          state.h_tra[GTCRN_MODEL_TRA_GRUS - 1][0][0][15] == 0.0f &&
          state.h_dpgrnn[1][0][32][15] == 0.0f,
          "GTCRN model state starts at zero");
    memset(&next, 0x5a, sizeof(next));
    CHECK(gtcrn_model_state_commit(&state, &next.conv_cache[0][0][0][0],
                                   h_tra, h_dpgrnn) == 0,
          "GTCRN commit accepts a finite state");
    CHECK(memcmp(&state, &next, sizeof(state)) == 0,
          "GTCRN state outputs become the next invocation inputs");

    /* Every byte of the rejected batch DIFFERS from the committed state, so a
     * non-transactional implementation that writes the good elements before
     * reaching the bad one leaves a visible difference. With a byte pattern
     * equal to what is already stored, a partial writeback would be
     * indistinguishable from a clean refusal. */
    committed = state;
    memset(&next, 0x41, sizeof(next));
    next.h_dpgrnn[1][0][32][15] = NAN;
    CHECK(gtcrn_model_state_commit(&state, &next.conv_cache[0][0][0][0],
                                   h_tra, h_dpgrnn) != 0,
          "GTCRN commit refuses a NaN state");
    CHECK(memcmp(&state, &committed, sizeof(state)) == 0,
          "GTCRN NaN refusal leaves the previous state byte-identical");

    /* The bad element sits in the FIRST tensor here, so the two cases
     * together cover refusal before and after the earlier ones validate. */
    memset(&next, 0x41, sizeof(next));
    next.conv_cache[0][0][0][0] = INFINITY;
    CHECK(gtcrn_model_state_commit(&state, &next.conv_cache[0][0][0][0],
                                   h_tra, h_dpgrnn) != 0,
          "GTCRN commit refuses an Inf state");
    CHECK(memcmp(&state, &committed, sizeof(state)) == 0,
          "GTCRN Inf refusal leaves the previous state byte-identical");

    for (i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
        broken[i] = h_tra[i];
    }
    broken[3] = NULL;
    CHECK(gtcrn_model_state_commit(NULL, &next.conv_cache[0][0][0][0],
                                   h_tra, h_dpgrnn) != 0 &&
          gtcrn_model_state_commit(&state, NULL, h_tra, h_dpgrnn) != 0 &&
          gtcrn_model_state_commit(&state, &next.conv_cache[0][0][0][0],
                                   broken, h_dpgrnn) != 0,
          "GTCRN commit refuses null arguments and null h elements");
    CHECK(memcmp(&state, &committed, sizeof(state)) == 0,
          "GTCRN preserved state survives every refused commit");

    /* A finite batch must still be accepted after the refusals, or the guard
     * could be a permanent latch rather than a per-call check. */
    memset(&next, 0x41, sizeof(next));
    CHECK(gtcrn_model_state_commit(&state, &next.conv_cache[0][0][0][0],
                                   h_tra, h_dpgrnn) == 0 &&
          memcmp(&state, &next, sizeof(state)) == 0,
          "GTCRN commit still accepts a finite state after a refusal");
    return 1;
}

static int run_all_tests(void)
{
    uint64_t digest = UINT64_C(1469598103934665603);
    CHECK(test_dfn2_model_io(), "DFN2 stateless model I/O");
    CHECK(test_dfn_stream_alignment(), "DFN streaming head alignment");
    CHECK(test_dfn2(&digest), "DFN2 C pre/post");
    CHECK(test_dfn3(&digest), "DFN3 C pre/post");
    CHECK(test_gtcrn_model_state(), "GTCRN stateless model I/O");
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
