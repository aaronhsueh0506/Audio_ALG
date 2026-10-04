/* C pre/post-processing contract test. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_process.h"
#include "DeepFilterNet2/dfn2_model_io.h"
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
    float spec2_re[DFN2_N_BINS], spec2_im[DFN2_N_BINS];
    float out2_re[DFN2_N_BINS], out2_im[DFN2_N_BINS];
    float mask2[DFN2_N_ERB] = {0};
    float coef2[DFN2_DF_BINS][DFN2_DF_ORDER][2] = {{{0}}};
    long long output_frame = -1;

    dfn_test_build_erb();
    {
        static FftHandle* fft2;
        if (!fft2) fft2 = fft_create(DFN2_N_FFT);
        dfn2_state_init(&dfn2, fft2);
    }
    dfn2_set_erb_matrices(&dfn2, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    for (int k = 0; k < DFN2_DF_BINS; ++k)
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap)
            coef2[k][tap][0] = stream_tap(tap);

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
    return 1;
}

static int all_zero(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (values[i] != 0.0f) return 0;
    }
    return 1;
}

static int test_dfn2_model_io(void)
{
    static DFN2ModelIOState state;
    static DFN2ModelIOState healthy;
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

    /* The accelerator writes the four recurrent arrays in place. */
    memset(state.encoder_gru_hidden, 0x3c, sizeof(state.encoder_gru_hidden));
    memset(state.erb_gru_hidden, 0x4d, sizeof(state.erb_gru_hidden));
    memset(state.df_gru_hidden, 0x5e, sizeof(state.df_gru_hidden));
    memset(state.df_convp_history, 0x6f, sizeof(state.df_convp_history));
    healthy = state;
    CHECK(dfn2_model_io_validate_state(&state) == 0 &&
          memcmp(&state, &healthy, sizeof(state)) == 0,
          "DFN2 validating a finite state leaves every byte in place");

    state.erb_gru_hidden[1][DFN2_MODEL_GRU_HIDDEN - 1] = NAN;
    CHECK(dfn2_model_io_validate_state(&state) != 0,
          "DFN2 validate checks the second ERB stack layer");
    CHECK(all_zero(&state.encoder_gru_hidden[0][0],
                   DFN2_MODEL_ENCODER_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN) &&
          all_zero(&state.erb_gru_hidden[0][0],
                   DFN2_MODEL_ERB_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN) &&
          all_zero(&state.df_gru_hidden[0][0],
                   DFN2_MODEL_DF_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN) &&
          all_zero(&state.df_convp_history[0][0][0],
                   (size_t)DFN2_MODEL_ENCODER_CHANNELS *
                       DFN2_MODEL_DF_PATHWAY_HISTORY * DFN2_DF_BINS),
          "DFN2 non-finite state zeroes all four recurrent arrays");
    CHECK(memcmp(state.erb_window, healthy.erb_window,
                 sizeof(state.erb_window)) == 0 &&
          memcmp(state.spec_window, healthy.spec_window,
                 sizeof(state.spec_window)) == 0 &&
          state.feature_frames_seen == healthy.feature_frames_seen,
          "DFN2 state refusal leaves the feature windows and counter alone");

    state = healthy;
    state.df_convp_history[DFN2_MODEL_ENCODER_CHANNELS - 1]
                          [DFN2_MODEL_DF_PATHWAY_HISTORY - 1]
                          [DFN2_DF_BINS - 1] = INFINITY;
    CHECK(dfn2_model_io_validate_state(&state) != 0 &&
          all_zero(&state.encoder_gru_hidden[0][0],
                   DFN2_MODEL_ENCODER_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN),
          "DFN2 refuses a non-finite pathway history and zeroes the batch");

    CHECK(dfn2_model_io_validate_state(NULL) != 0,
          "DFN2 validate refuses a null state");
    state = healthy;
    CHECK(dfn2_model_io_validate_arrays(NULL, state.erb_gru_hidden,
                                        state.df_gru_hidden,
                                        state.df_convp_history) != 0 &&
          dfn2_model_io_validate_arrays(state.encoder_gru_hidden,
                                        state.erb_gru_hidden,
                                        state.df_gru_hidden, NULL) != 0 &&
          memcmp(&state, &healthy, sizeof(state)) == 0,
          "DFN2 validate_arrays refuses a null array and touches nothing");
    CHECK(dfn2_model_io_validate_arrays(state.encoder_gru_hidden,
                                        state.erb_gru_hidden,
                                        state.df_gru_hidden,
                                        state.df_convp_history) == 0,
          "DFN2 validate_arrays accepts finite arrays");

    /* Copy path: the runtime wrote its *_next tensors into its own buffers. */
    {
        static DFN2ModelIOState runtime;
        static DFN2ModelIOState before;
        const size_t convp_count = (size_t)DFN2_MODEL_ENCODER_CHANNELS *
                                   DFN2_MODEL_DF_PATHWAY_HISTORY * DFN2_DF_BINS;

        memset(runtime.encoder_gru_hidden, 0x11,
               sizeof(runtime.encoder_gru_hidden));
        memset(runtime.erb_gru_hidden, 0x22, sizeof(runtime.erb_gru_hidden));
        memset(runtime.df_gru_hidden, 0x33, sizeof(runtime.df_gru_hidden));
        memset(runtime.df_convp_history, 0x44,
               sizeof(runtime.df_convp_history));

        state = healthy;
        CHECK(dfn2_model_io_inherit_state(
                  &state, runtime.encoder_gru_hidden, runtime.erb_gru_hidden,
                  runtime.df_gru_hidden, runtime.df_convp_history) == 0 &&
              memcmp(state.encoder_gru_hidden, runtime.encoder_gru_hidden,
                     sizeof(state.encoder_gru_hidden)) == 0 &&
              memcmp(state.erb_gru_hidden, runtime.erb_gru_hidden,
                     sizeof(state.erb_gru_hidden)) == 0 &&
              memcmp(state.df_gru_hidden, runtime.df_gru_hidden,
                     sizeof(state.df_gru_hidden)) == 0 &&
              memcmp(state.df_convp_history, runtime.df_convp_history,
                     sizeof(state.df_convp_history)) == 0,
              "DFN2 inherit_state copies all four recurrent arrays");
        CHECK(memcmp(state.erb_window, healthy.erb_window,
                     sizeof(state.erb_window)) == 0 &&
              memcmp(state.spec_window, healthy.spec_window,
                     sizeof(state.spec_window)) == 0 &&
              state.feature_frames_seen == healthy.feature_frames_seen,
              "DFN2 inherit_state leaves the feature windows and counter alone");

        /* One non-finite element in any one source copies nothing, including
         * the arrays that precede it, and leaves the state intact. */
        for (int which = 0; which < 4; ++which) {
            float* poison =
                which == 0 ? &runtime.encoder_gru_hidden[0][5] :
                which == 1 ? &runtime.erb_gru_hidden[1][0] :
                which == 2 ? &runtime.df_gru_hidden[1][DFN2_MODEL_GRU_HIDDEN - 1] :
                             &runtime.df_convp_history
                                   [DFN2_MODEL_ENCODER_CHANNELS - 1]
                                   [DFN2_MODEL_DF_PATHWAY_HISTORY - 1]
                                   [DFN2_DF_BINS - 1];
            const float saved = *poison;
            state = healthy;
            before = state;
            *poison = (which & 1) ? INFINITY : NAN;
            CHECK(dfn2_model_io_inherit_state(
                      &state, runtime.encoder_gru_hidden,
                      runtime.erb_gru_hidden, runtime.df_gru_hidden,
                      runtime.df_convp_history) == -1 &&
                  memcmp(&state, &before, sizeof(state)) == 0,
                  "DFN2 inherit_state refuses a non-finite source and copies nothing");
            CHECK(dfn2_model_io_inherit_arrays(
                      state.encoder_gru_hidden, state.erb_gru_hidden,
                      state.df_gru_hidden, state.df_convp_history,
                      runtime.encoder_gru_hidden, runtime.erb_gru_hidden,
                      runtime.df_gru_hidden, runtime.df_convp_history) == -1 &&
                  memcmp(&state, &before, sizeof(state)) == 0,
                  "DFN2 inherit_arrays refuses a non-finite source and copies nothing");
            *poison = saved;
        }

        /* Sources at the destination's own address were written in place:
         * skipped, never copied onto themselves, never checked here. */
        state = healthy;
        before = state;
        CHECK(dfn2_model_io_inherit_state(
                  &state, state.encoder_gru_hidden, state.erb_gru_hidden,
                  state.df_gru_hidden, state.df_convp_history) == 0 &&
              memcmp(&state, &before, sizeof(state)) == 0,
              "DFN2 inherit_state with every source aliased is a no-op");
        state.df_gru_hidden[0][3] = NAN;
        before = state;
        CHECK(dfn2_model_io_inherit_state(
                  &state, state.encoder_gru_hidden, state.erb_gru_hidden,
                  state.df_gru_hidden, state.df_convp_history) == 0 &&
              memcmp(&state, &before, sizeof(state)) == 0 &&
              dfn2_model_io_validate_state(&state) == -1,
              "DFN2 inherit leaves an in-place value to validate_state");

        /* Mixed binding: only the tensors that moved are copied. */
        state = healthy;
        CHECK(dfn2_model_io_inherit_arrays(
                  state.encoder_gru_hidden, state.erb_gru_hidden,
                  state.df_gru_hidden, state.df_convp_history,
                  state.encoder_gru_hidden, runtime.erb_gru_hidden,
                  state.df_gru_hidden, runtime.df_convp_history) == 0 &&
              memcmp(state.encoder_gru_hidden, healthy.encoder_gru_hidden,
                     sizeof(state.encoder_gru_hidden)) == 0 &&
              memcmp(state.erb_gru_hidden, runtime.erb_gru_hidden,
                     sizeof(state.erb_gru_hidden)) == 0 &&
              memcmp(state.df_gru_hidden, healthy.df_gru_hidden,
                     sizeof(state.df_gru_hidden)) == 0 &&
              memcmp(state.df_convp_history, runtime.df_convp_history,
                     convp_count * sizeof(float)) == 0,
              "DFN2 inherit_arrays copies only the arrays that moved");

        /* NULL anywhere is refused and touches nothing. */
        state = healthy;
        before = state;
        CHECK(dfn2_model_io_inherit_state(
                  NULL, runtime.encoder_gru_hidden, runtime.erb_gru_hidden,
                  runtime.df_gru_hidden, runtime.df_convp_history) == -1 &&
              dfn2_model_io_inherit_state(
                  &state, NULL, runtime.erb_gru_hidden,
                  runtime.df_gru_hidden, runtime.df_convp_history) == -1 &&
              dfn2_model_io_inherit_state(
                  &state, runtime.encoder_gru_hidden, runtime.erb_gru_hidden,
                  runtime.df_gru_hidden, NULL) == -1 &&
              dfn2_model_io_inherit_arrays(
                  NULL, state.erb_gru_hidden, state.df_gru_hidden,
                  state.df_convp_history, runtime.encoder_gru_hidden,
                  runtime.erb_gru_hidden, runtime.df_gru_hidden,
                  runtime.df_convp_history) == -1 &&
              dfn2_model_io_inherit_arrays(
                  state.encoder_gru_hidden, state.erb_gru_hidden,
                  state.df_gru_hidden, NULL, runtime.encoder_gru_hidden,
                  runtime.erb_gru_hidden, runtime.df_gru_hidden,
                  runtime.df_convp_history) == -1 &&
              memcmp(&state, &before, sizeof(state)) == 0,
              "DFN2 inherit refuses a NULL argument and touches nothing");
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
    static GTCRNModelState zeros;
    static GTCRNModelState expect;
    float* conv[GTCRN_MODEL_CONV_STATES];
    float* h_tra[GTCRN_MODEL_TRA_GRUS];
    float* h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS];
    size_t conv_n[GTCRN_MODEL_CONV_STATES];
    const size_t tra_n = sizeof(state.h_tra[0]) / sizeof(float);
    const size_t dpgrnn_n = sizeof(state.h_dpgrnn[0]) / sizeof(float);
    int i;
    int c;
    CHECK(sizeof(state) == 72192u,
          "GTCRN v6 keeps the v5 caller-owned state byte budget");
    /* The first element of every state tensor: the board binds the graph's
     * state input and *_out output to these addresses. */
    conv[0] = &state.conv_enc0[0][0][0];
    conv[1] = &state.conv_enc1[0][0][0];
    conv[2] = &state.conv_enc2[0][0][0];
    conv[3] = &state.conv_dec0[0][0][0];
    conv[4] = &state.conv_dec1[0][0][0];
    conv[5] = &state.conv_dec2[0][0][0];
    conv_n[0] = sizeof(state.conv_enc0) / sizeof(float);
    conv_n[1] = sizeof(state.conv_enc1) / sizeof(float);
    conv_n[2] = sizeof(state.conv_enc2) / sizeof(float);
    conv_n[3] = sizeof(state.conv_dec0) / sizeof(float);
    conv_n[4] = sizeof(state.conv_dec1) / sizeof(float);
    conv_n[5] = sizeof(state.conv_dec2) / sizeof(float);
    for (i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
        h_tra[i] = &state.h_tra[i][0][0][0];
    }
    for (i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
        h_dpgrnn[i] = &state.h_dpgrnn[i][0][0][0];
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
    CHECK(state.conv_dec0[15][GTCRN_MODEL_CONV_TIME_2 - 1][32] == 0.0f &&
          state.h_tra[GTCRN_MODEL_TRA_GRUS - 1][0][0][15] == 0.0f &&
          state.h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS - 1][0][32]
                           [GTCRN_MODEL_DPGRNN_HIDDEN - 1] == 0.0f,
          "GTCRN model state starts at zero");
    /* The struct is the state buffer the graph writes its *_out tensors
     * into, so a healthy state validates in place and is left untouched. */
    memset(&state, 0x5a, sizeof(state));
    memset(&expect, 0x5a, sizeof(expect));
    CHECK(gtcrn_model_state_validate(&state) == 0,
          "GTCRN validate accepts a finite state");
    CHECK(memcmp(&state, &expect, sizeof(state)) == 0,
          "GTCRN validate leaves a healthy state untouched");

    CHECK(gtcrn_model_state_validate(NULL) == -1,
          "GTCRN validate refuses a null state");

    /* The sixteen tensors tile the struct exactly, so checking each one
     * covers every state byte. */
    {
        size_t total = 0;
        for (i = 0; i < GTCRN_MODEL_CONV_STATES; ++i) total += conv_n[i];
        total += (size_t)GTCRN_MODEL_TRA_GRUS * tra_n;
        total += (size_t)GTCRN_MODEL_DPGRNN_GRUS * dpgrnn_n;
        CHECK(total * sizeof(float) == sizeof(state),
              "GTCRN state tensors tile the whole state struct");
    }

    /* One NaN or Inf at the first or last element of any single tensor
     * zeroes the WHOLE state. The state is nonzero everywhere else (0x41
     * pattern), so a partial reset or an unchecked tensor is visible. */
    for (c = 0; c < 4; ++c) {
        const float bad = (c & 1) ? INFINITY : NAN;
        int cls_ok = 1;
        for (i = 0; i < GTCRN_MODEL_CONV_STATES; ++i) {
            memset(&state, 0x41, sizeof(state));
            conv[i][(c < 2) ? 0 : conv_n[i] - 1] = bad;
            cls_ok &= gtcrn_model_state_validate(&state) == -1 &&
                      memcmp(&state, &zeros, sizeof(state)) == 0;
        }
        CHECK(cls_ok, "GTCRN non-finite conv element returns -1 and "
                      "zeroes the whole state");
        cls_ok = 1;
        for (i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
            memset(&state, 0x41, sizeof(state));
            h_tra[i][(c < 2) ? 0 : tra_n - 1] = bad;
            cls_ok &= gtcrn_model_state_validate(&state) == -1 &&
                      memcmp(&state, &zeros, sizeof(state)) == 0;
        }
        CHECK(cls_ok, "GTCRN non-finite h_tra element returns -1 and "
                      "zeroes the whole state");
        cls_ok = 1;
        for (i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
            memset(&state, 0x41, sizeof(state));
            h_dpgrnn[i][(c < 2) ? 0 : dpgrnn_n - 1] = bad;
            cls_ok &= gtcrn_model_state_validate(&state) == -1 &&
                      memcmp(&state, &zeros, sizeof(state)) == 0;
        }
        CHECK(cls_ok, "GTCRN non-finite h_dpgrnn element returns -1 and "
                      "zeroes the whole state");
    }

    /* A finite state must still validate after a refusal, or the guard could
     * be a permanent latch rather than a per-call check. */
    memset(&state, 0x41, sizeof(state));
    memset(&expect, 0x41, sizeof(expect));
    CHECK(gtcrn_model_state_validate(&state) == 0 &&
          memcmp(&state, &expect, sizeof(state)) == 0,
          "GTCRN validate still accepts a finite state after a refusal");

    /* Copy path: the runtime's own *_out tensors are finite-checked and
     * copied into the state by gtcrn_model_state_inherit. */
    {
        enum { INHERIT_FRAMES = 24 };
        static GTCRNModelState in_place;
        static GTCRNModelState copied;
        static GTCRNModelState before;
        static GTCRNModelState source;
        /* The runtime's private output tensors: a second struct whose
         * fields are laid out like the state's. */
        const float* cs[GTCRN_MODEL_CONV_STATES];
        const float* ts[GTCRN_MODEL_TRA_GRUS];
        const float* ds[GTCRN_MODEL_DPGRNN_GRUS];
        const float* cself[GTCRN_MODEL_CONV_STATES];
        const float* tself[GTCRN_MODEL_TRA_GRUS];
        const float* dself[GTCRN_MODEL_DPGRNN_GRUS];
        float* sc[GTCRN_MODEL_CONV_STATES];
        float* st[GTCRN_MODEL_TRA_GRUS];
        float* sd[GTCRN_MODEL_DPGRNN_GRUS];
        enum { N_TENSORS = GTCRN_MODEL_CONV_STATES + GTCRN_MODEL_TRA_GRUS +
                           GTCRN_MODEL_DPGRNN_GRUS };
        int stream_ok = 1;
        int alias_ok = 1;
        int nan_ok = 1;
        int t;
        size_t k;
        sc[0] = &source.conv_enc0[0][0][0];
        sc[1] = &source.conv_enc1[0][0][0];
        sc[2] = &source.conv_enc2[0][0][0];
        sc[3] = &source.conv_dec0[0][0][0];
        sc[4] = &source.conv_dec1[0][0][0];
        sc[5] = &source.conv_dec2[0][0][0];
        for (i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i)
            st[i] = &source.h_tra[i][0][0][0];
        for (i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i)
            sd[i] = &source.h_dpgrnn[i][0][0][0];
        cself[0] = &copied.conv_enc0[0][0][0];
        cself[1] = &copied.conv_enc1[0][0][0];
        cself[2] = &copied.conv_enc2[0][0][0];
        cself[3] = &copied.conv_dec0[0][0][0];
        cself[4] = &copied.conv_dec1[0][0][0];
        cself[5] = &copied.conv_dec2[0][0][0];
        for (i = 0; i < GTCRN_MODEL_CONV_STATES; ++i) cs[i] = sc[i];
        for (i = 0; i < GTCRN_MODEL_TRA_GRUS; ++i) {
            ts[i] = st[i];
            tself[i] = &copied.h_tra[i][0][0][0];
        }
        for (i = 0; i < GTCRN_MODEL_DPGRNN_GRUS; ++i) {
            ds[i] = sd[i];
            dself[i] = &copied.h_dpgrnn[i][0][0][0];
        }

        /* In-place state A has each frame written straight into the struct;
         * copy state B goes through the private tensors and inherit. Every
         * frame they must be byte-identical, and B must equal the source. */
        gtcrn_model_state_init(&in_place);
        gtcrn_model_state_init(&copied);
        for (t = 0; t < INHERIT_FRAMES; ++t) {
            float* a = (float*)&in_place;
            float* s = (float*)&source;
            for (k = 0; k < sizeof(source) / sizeof(float); ++k) {
                float v = sinf(0.29f * (float)(t + 1) + 0.0007f * (float)k);
                a[k] = v;
                s[k] = v;
            }
            stream_ok &= gtcrn_model_state_validate(&in_place) == 0 &&
                         gtcrn_model_state_inherit(&copied, cs, ts, ds) == 0 &&
                         memcmp(&in_place, &copied, sizeof(copied)) == 0 &&
                         memcmp(&copied, &source, sizeof(copied)) == 0;
        }
        CHECK(stream_ok,
              "GTCRN inherit stream is byte-identical to the in-place state");

        /* Every pointer equal to its field: the runtime wrote in place, so
         * nothing is copied or checked and the state is unchanged (a
         * non-finite value included: validate owns that case). */
        memset(&copied, 0x41, sizeof(copied));
        copied.conv_enc0[0][0][0] = NAN;
        before = copied;
        alias_ok &= gtcrn_model_state_inherit(&copied, cself, tself, dself)
                        == 0 &&
                    memcmp(&copied, &before, sizeof(copied)) == 0;
        /* Mixed: one aliased tensor is skipped, the others are copied. */
        memset(&copied, 0x41, sizeof(copied));
        cs[2] = cself[2];
        expect = source;
        memcpy(&expect.conv_enc2, &copied.conv_enc2, sizeof(expect.conv_enc2));
        alias_ok &= gtcrn_model_state_inherit(&copied, cs, ts, ds) == 0 &&
                    memcmp(&copied, &expect, sizeof(copied)) == 0;
        cs[2] = sc[2];
        CHECK(alias_ok,
              "GTCRN inherit leaves aliased tensors alone and copies the "
              "rest");

        /* One NaN or Inf at the first or last element of any one of the
         * sixteen tensors: -1, nothing copied, state byte-identical. Source
         * and state differ everywhere else, so a partial copy is visible. */
        for (c = 0; c < 4; ++c) {
            const float bad = (c & 1) ? INFINITY : NAN;
            for (i = 0; i < N_TENSORS; ++i) {
                float* victim;
                size_t n;
                if (i < GTCRN_MODEL_CONV_STATES) {
                    victim = sc[i];
                    n = conv_n[i];
                } else if (i < GTCRN_MODEL_CONV_STATES +
                               GTCRN_MODEL_TRA_GRUS) {
                    victim = st[i - GTCRN_MODEL_CONV_STATES];
                    n = tra_n;
                } else {
                    victim = sd[i - GTCRN_MODEL_CONV_STATES -
                                GTCRN_MODEL_TRA_GRUS];
                    n = dpgrnn_n;
                }
                memset(&source, 0x3d, sizeof(source));
                victim[(c < 2) ? 0 : n - 1] = bad;
                memset(&copied, 0x41, sizeof(copied));
                before = copied;
                nan_ok &= gtcrn_model_state_inherit(&copied, cs, ts, ds)
                              == -1 &&
                          memcmp(&copied, &before, sizeof(copied)) == 0;
            }
        }
        CHECK(nan_ok,
              "GTCRN inherit refuses a non-finite element in any of the "
              "sixteen tensors and leaves the state byte-identical");

        memset(&source, 0x3d, sizeof(source));
        memset(&copied, 0x41, sizeof(copied));
        before = copied;
        {
            const float* null_c[GTCRN_MODEL_CONV_STATES];
            const float* null_t[GTCRN_MODEL_TRA_GRUS];
            const float* null_d[GTCRN_MODEL_DPGRNN_GRUS];
            int null_ok;
            memcpy(null_c, cs, sizeof(null_c));
            memcpy(null_t, ts, sizeof(null_t));
            memcpy(null_d, ds, sizeof(null_d));
            null_ok = gtcrn_model_state_inherit(NULL, cs, ts, ds) == -1 &&
                      gtcrn_model_state_inherit(&copied, NULL, ts, ds) == -1 &&
                      gtcrn_model_state_inherit(&copied, cs, NULL, ds) == -1 &&
                      gtcrn_model_state_inherit(&copied, cs, ts, NULL) == -1;
            for (i = 0; i < N_TENSORS; ++i) {
                const float** slot;
                const float* saved;
                if (i < GTCRN_MODEL_CONV_STATES) {
                    slot = &null_c[i];
                } else if (i < GTCRN_MODEL_CONV_STATES +
                               GTCRN_MODEL_TRA_GRUS) {
                    slot = &null_t[i - GTCRN_MODEL_CONV_STATES];
                } else {
                    slot = &null_d[i - GTCRN_MODEL_CONV_STATES -
                                   GTCRN_MODEL_TRA_GRUS];
                }
                saved = *slot;
                *slot = NULL;
                null_ok &= gtcrn_model_state_inherit(&copied, null_c, null_t,
                                                     null_d) == -1;
                *slot = saved;
            }
            CHECK(null_ok && memcmp(&copied, &before, sizeof(copied)) == 0,
                  "GTCRN inherit refuses a NULL state, array or element and "
                  "leaves the state intact");
        }
    }
    return 1;
}

/* The estimation branch (dfn2_compute_features) and the application branch
 * (dfn2_compose_stream) of one DFN2State must be independent: the compose
 * output is a function of the applied spectrum and the heads alone, and the
 * features are a function of the estimated spectrum alone.  Proved by
 * running two states that differ in only one branch's input and comparing
 * the other branch's output byte for byte. */
static int test_dfn_dual_state_disjoint(void)
{
    static DFN2State st_a, st_b;
    static float est_re[DFN2_N_BINS], est_im[DFN2_N_BINS];
    static float est2_re[DFN2_N_BINS], est2_im[DFN2_N_BINS];
    static float app_re[DFN2_N_BINS], app_im[DFN2_N_BINS];
    static float app2_re[DFN2_N_BINS], app2_im[DFN2_N_BINS];
    static float feat_erb_a[DFN2_N_ERB], feat_spec_a[2 * DFN2_DF_BINS];
    static float feat_erb_b[DFN2_N_ERB], feat_spec_b[2 * DFN2_DF_BINS];
    static float out_a_re[DFN2_N_BINS], out_a_im[DFN2_N_BINS];
    static float out_b_re[DFN2_N_BINS], out_b_im[DFN2_N_BINS];
    float mask[DFN2_N_ERB];
    float coef[DFN2_DF_BINS][DFN2_DF_ORDER][2] = {{{0}}};
    int features_differed = 0;

    dfn_test_build_erb();
    dfn2_state_init(&st_a, NULL);
    dfn2_state_init(&st_b, NULL);
    dfn2_set_erb_matrices(&st_a, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    dfn2_set_erb_matrices(&st_b, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    for (int k = 0; k < DFN2_DF_BINS; ++k)
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap)
            coef[k][tap][0] = stream_tap(tap);

    /* Same applied spectrum, different estimated spectra: the compose
     * outputs must agree while the features must differ. */
    for (int wall = 0; wall < 14; ++wall) {
        int head = wall - DFN2_MASK_LOOKAHEAD;
        int va, vb;
        for (int k = 0; k < DFN2_N_BINS; ++k) {
            est_re[k] = stream_spec_re(wall, k);
            est_im[k] = stream_spec_im(wall, k);
            est2_re[k] = 0.37f * est_re[k] + 0.01f;
            est2_im[k] = -0.5f * est_im[k];
            app_re[k] = stream_spec_re(wall + 5, k);
            app_im[k] = stream_spec_im(wall + 5, k);
        }
        dfn2_compute_features(&st_a, est_re, est_im, feat_erb_a, feat_spec_a);
        dfn2_compute_features(&st_b, est2_re, est2_im, feat_erb_b, feat_spec_b);
        if (memcmp(feat_erb_a, feat_erb_b, sizeof(feat_erb_a)) != 0)
            features_differed = 1;
        for (int b = 0; b < DFN2_N_ERB; ++b)
            mask[b] = head >= 0 ? stream_mask(head) : 0.0f;
        va = dfn2_compose_stream(&st_a, app_re, app_im, head >= 0,
                                 head >= 0 ? mask : NULL,
                                 head >= 0 ? &coef[0][0][0] : NULL,
                                 head >= 0 ? stream_alpha(head) : 0.0f,
                                 0.0f, out_a_re, out_a_im, NULL);
        vb = dfn2_compose_stream(&st_b, app_re, app_im, head >= 0,
                                 head >= 0 ? mask : NULL,
                                 head >= 0 ? &coef[0][0][0] : NULL,
                                 head >= 0 ? stream_alpha(head) : 0.0f,
                                 0.0f, out_b_re, out_b_im, NULL);
        CHECK(va == vb, "DFN2 compose clocks agree across states");
        if (va == 1) {
            CHECK(memcmp(out_a_re, out_b_re, sizeof(out_a_re)) == 0 &&
                  memcmp(out_a_im, out_b_im, sizeof(out_a_im)) == 0,
                  "DFN2 compose output does not depend on the estimated spectrum");
        }
    }
    CHECK(features_differed, "DFN2 features do depend on the estimated spectrum");

    /* Same estimated spectrum, different applied spectra: the features must
     * agree byte for byte. */
    dfn2_state_init(&st_a, NULL);
    dfn2_state_init(&st_b, NULL);
    dfn2_set_erb_matrices(&st_a, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    dfn2_set_erb_matrices(&st_b, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    for (int wall = 0; wall < 14; ++wall) {
        int head = wall - DFN2_MASK_LOOKAHEAD;
        for (int k = 0; k < DFN2_N_BINS; ++k) {
            est_re[k] = stream_spec_re(wall, k);
            est_im[k] = stream_spec_im(wall, k);
            app_re[k] = stream_spec_re(wall + 5, k);
            app_im[k] = stream_spec_im(wall + 5, k);
            app2_re[k] = 0.11f * app_re[k] - 0.02f;
            app2_im[k] = 2.0f * app_im[k];
        }
        for (int b = 0; b < DFN2_N_ERB; ++b)
            mask[b] = head >= 0 ? stream_mask(head) : 0.0f;
        (void)dfn2_compose_stream(&st_a, app_re, app_im, head >= 0,
                                  head >= 0 ? mask : NULL,
                                  head >= 0 ? &coef[0][0][0] : NULL,
                                  head >= 0 ? stream_alpha(head) : 0.0f,
                                  0.0f, out_a_re, out_a_im, NULL);
        (void)dfn2_compose_stream(&st_b, app2_re, app2_im, head >= 0,
                                  head >= 0 ? mask : NULL,
                                  head >= 0 ? &coef[0][0][0] : NULL,
                                  head >= 0 ? stream_alpha(head) : 0.0f,
                                  0.0f, out_b_re, out_b_im, NULL);
        dfn2_compute_features(&st_a, est_re, est_im, feat_erb_a, feat_spec_a);
        dfn2_compute_features(&st_b, est_re, est_im, feat_erb_b, feat_spec_b);
        CHECK(memcmp(feat_erb_a, feat_erb_b, sizeof(feat_erb_a)) == 0 &&
              memcmp(feat_spec_a, feat_spec_b, sizeof(feat_spec_a)) == 0,
              "DFN2 features do not depend on the applied spectrum");
    }
    return 1;
}

/* The ERB feature sum and the mask expansion visit only the matrices'
 * nonzero spans (DFN2State). Checked here against the plain dense sums,
 * bit for bit, including the cases the spans must not change: powers that
 * overflow to Inf, NaN spectra, non-finite band gains, and matrices with
 * NaN/Inf entries, an all-zero row and zero holes inside a span. */
static int test_dfn_erb_spans_exact(void)
{
    static DFN2State st;
    static float fwd[513][32], inv[32][513], coefs[DFN2_DF_BINS * DFN2_DF_ORDER * 2];
    float re[513], im[513], fe[32], fs[2 * DFN2_DF_BINS], mask[32];
    float out_re[513], out_im[513], ref_state[32], erb[32], ref_gain[513];
    const float scale2 = DFN2_ANALYSIS_SCALE * DFN2_ANALYSIS_SCALE;
    uint32_t lcg = 12345u;

    dfn_test_build_erb();
    memset(coefs, 0, sizeof(coefs));
    for (int variant = 0; variant < 3; ++variant) {
        memcpy(fwd, dfn_test_fwd, sizeof(fwd));
        memcpy(inv, dfn_test_inv, sizeof(inv));
        if (variant == 1) {
            fwd[10][5] = NAN;
            fwd[300][0] = INFINITY;
            memset(fwd[7], 0, sizeof(fwd[7]));
            fwd[200][20] = 0.0f;       /* a hole inside a span */
            inv[3][40] = -0.0f;
            inv[20][0] = 0.5f;         /* a far outlier widens the span */
        } else if (variant == 2) {
            for (int k = 0; k < 513; ++k) fwd[k][(k * 7) % 32] += 0.125f;
            inv[31][3] = 0.25f;
        }
        dfn2_state_init(&st, NULL);
        dfn2_set_erb_matrices(&st, &fwd[0][0], &inv[0][0]);
        memcpy(ref_state, st.erb_norm_state, sizeof(ref_state));
        for (int frame = 0; frame < 24; ++frame) {
            for (int k = 0; k < 513; ++k) {
                lcg = lcg * 1664525u + 1013904223u;
                re[k] = (float)(int32_t)lcg * 4.6566e-10f;
                lcg = lcg * 1664525u + 1013904223u;
                im[k] = (float)(int32_t)lcg * 4.6566e-10f;
            }
            if (frame % 4 == 1) re[37 + frame] = 1e30f;   /* power -> Inf */
            if (frame % 6 == 3) im[250] = NAN;
            if (frame % 5 == 2) re[512] = -INFINITY;

            dfn2_compute_features(&st, re, im, fe, fs);
            memset(erb, 0, sizeof(erb));
            for (int k = 0; k < 513; ++k) {
                float p = (re[k] * re[k] + im[k] * im[k]) * scale2;
                for (int b = 0; b < 32; ++b) erb[b] += p * fwd[k][b];
            }
            for (int b = 0; b < 32; ++b) {
                float db = 10.0f * log10f(erb[b] + DFN2_ERB_LOG_FLOOR);
                float mean = DFN2_ERB_NORM_ALPHA * ref_state[b] +
                             (1.0f - DFN2_ERB_NORM_ALPHA) * db;
                float ref = (db - mean) / DFN2_ERB_NORM_SCALE_DB;
                ref_state[b] = mean;
                CHECK(memcmp(&fe[b], &ref, sizeof(ref)) == 0,
                      "DFN2 ERB features over the nonzero spans equal the dense sum");
            }

            for (int b = 0; b < 32; ++b) mask[b] = 0.3f + 0.02f * (float)b;
            if (frame % 3 == 0) mask[frame % 32] = (frame & 1) ? INFINITY : NAN;
            if (frame % 7 == 2) mask[0] = -INFINITY;
            if (frame % 10 == 9) mask[31] = -0.0f;
            (void)dfn2_compose(&st, re, im, mask, coefs, 0.5f, out_re, out_im);
            memset(ref_gain, 0, sizeof(ref_gain));
            for (int b = 0; b < 32; ++b)
                for (int k = 0; k < 513; ++k) ref_gain[k] += inv[b][k] * mask[b];
            CHECK(memcmp(st.scratch_bin_gain, ref_gain, sizeof(ref_gain)) == 0,
                  "DFN2 mask expansion over the nonzero spans equals the dense sum");
        }
    }
    return 1;
}

static int run_all_tests(void)
{
    uint64_t digest = UINT64_C(1469598103934665603);
    CHECK(test_dfn_erb_spans_exact(),
          "DFN2 ERB nonzero spans are exact");
    CHECK(test_dfn2_model_io(), "DFN2 stateless model I/O");
    CHECK(test_dfn_stream_alignment(), "DFN streaming head alignment");
    CHECK(test_dfn_dual_state_disjoint(),
          "DFN2 estimation/application branches are disjoint");
    CHECK(test_dfn2(&digest), "DFN2 C pre/post");
    CHECK(test_gtcrn_model_state(), "GTCRN stateless model I/O");
    CHECK(test_gtcrn(&digest), "GTCRN C pre/post");
    printf("backend=%s/%s digest=%016llx\n",
           dfn2_simd_backend(), gtcrn_simd_backend(),
           (unsigned long long)digest);
    return 1;
}

int main(int argc, char** argv)
{
    if (argc == 2 && strcmp(argv[1], "--stream-only") == 0)
        return test_dfn_stream_alignment() ? EXIT_SUCCESS : EXIT_FAILURE;
    return run_all_tests() ? EXIT_SUCCESS : EXIT_FAILURE;
}
