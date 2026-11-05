#define _POSIX_C_SOURCE 200809L
/* C pre/post-processing contract test.
 *
 * The DeepFilterNet2 grid and lookahead pair this program runs are chosen when it
 * is built (dfn2_test_grid.h: -DDFN2_GRID_KHZ=16|32|48, -DDFN2_MASK_LOOKAHEAD,
 * -DDFN2_DF_LOOKAHEAD); the library is handed the same grid at run time. The
 * digest printed for the shipped 48 kHz 1/1 grid is a recorded value; the other
 * grids print their own. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_process.h"
#include "DeepFilterNet2/dfn2_model_io.h"
#include "GTCRN/gtcrn_process.h"
#include "dfn2_test_grid.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
        return 0;                                                           \
    }                                                                       \
} while (0)

/* Framing objects the way a host builds them: one FFT handle, one window
 * table (carved by the analysis object, borrowed by the synthesis object),
 * each object in its own 16-byte aligned block. */
static void* aligned_block(size_t bytes)
{
    void* block = NULL;
    if (posix_memalign(&block, 16, bytes) != 0) return NULL;
    memset(block, 0xA5, bytes);
    return block;
}

static int dfn2_framing(AcStft** stft, AcIstft** istft)
{
    AcStftConfig cfg;
    size_t bytes;
    FftHandle* fft = fft_create(DFN2_N_FFT);
    if (!fft) return 0;
    dfn2_framing_config(&cfg, dfn2_test_grid(), fft, NULL);
    bytes = ac_stft_get_mem_size(&cfg);
    *stft = ac_stft_init(aligned_block(bytes), bytes, &cfg);
    dfn2_framing_config(&cfg, dfn2_test_grid(), fft,
                        *stft ? ac_stft_window(*stft) : NULL);
    bytes = ac_istft_get_mem_size(&cfg);
    *istft = ac_istft_init(aligned_block(bytes), bytes, &cfg);
    return *stft != NULL && *istft != NULL;
}

static int gtcrn_framing(AcStft** stft, AcIstft** istft)
{
    AcStftConfig cfg;
    size_t bytes;
    FftHandle* fft = fft_create(GTCRN_N_FFT);
    if (!fft) return 0;
    gtcrn_framing_config(&cfg, fft, NULL);
    bytes = ac_stft_get_mem_size(&cfg);
    *stft = ac_stft_init(aligned_block(bytes), bytes, &cfg);
    gtcrn_framing_config(&cfg, fft, *stft ? ac_stft_window(*stft) : NULL);
    bytes = ac_istft_get_mem_size(&cfg);
    *istft = ac_istft_init(aligned_block(bytes), bytes, &cfg);
    return *stft != NULL && *istft != NULL;
}

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

/* Reference ERB matrices for the DFN tests, built with the exact triangular
 * construction over a band border table (dfn2_test_band_borders(), dfn2_test_grid.h):
 * the shipped table on the 48 kHz / 1024 / 32 grid, a generated one on any other grid.
 * The runtime consumes only caller-loaded matrices (erb_fwd.bin/erb_inv.bin);
 * the tests build them here and hand them in through the same pointers,
 * so each golden value is reproduced from the table, not from a stored
 * copy of the matrices. */

static float dfn_test_fwd[DFN2_N_BINS][DFN2_N_ERB];
static float dfn_test_inv[DFN2_N_ERB][DFN2_N_BINS];
static void dfn_test_build_erb(void)
{
    const int last = DFN2_N_ERB - 1;
    int borders[DFN2_N_ERB];
    int segment = 0;
    dfn2_test_band_borders(borders);
    if (borders[0] != 0 || borders[last] != DFN2_N_BINS) exit(2);
    for (int b = 0; b < last; ++b)
        if (borders[b + 1] - borders[b] < 2) exit(2);
    memset(dfn_test_fwd, 0, sizeof(dfn_test_fwd));
    memset(dfn_test_inv, 0, sizeof(dfn_test_inv));
    for (int k = 0; k < DFN2_N_BINS; ++k) {
        int lo, hi, width, offset;
        float right, left, fleft, fright;
        while (segment + 1 < last && k >= borders[segment + 1])
            ++segment;
        lo = borders[segment];
        hi = borders[segment + 1];
        width = hi - lo;
        offset = k - lo;
        right = (float)offset / (float)width;
        left = 1.0f - right;
        fleft = left; fright = right;
        if (segment == 0) fleft *= 2.0f;
        if (segment + 1 == last) fright *= 2.0f;
        dfn_test_fwd[k][segment] = fleft;
        dfn_test_fwd[k][segment + 1] = fright;
        dfn_test_inv[segment][k] = left;
        dfn_test_inv[segment + 1][k] = right;
    }
}

/* The buffers of this program are sized by the DFN2_* macros of dfn2_test_grid.h
 * while the library runs on the DFN2Grid built from the same values: the two must
 * agree, or a buffer is smaller than what the library reads or writes. Then the
 * run-time refusals that stand where the grid and the lookahead pair used to be
 * compile-time constants: an invalid grid never starts an object, and a
 * descriptor or a pair of another grid is refused. */
static int test_dfn2_grid_contract(void)
{
    const DFN2Grid* g = dfn2_test_grid();
    static DFN2State state;
    static DFN2ModelIOState model_io;
    float erb[DFN2_N_ERB] = {0};
    float spec[2 * DFN2_DF_BINS] = {0};
    DFN2ModelIoDescriptor descriptor;
    DFN2Grid other, bad;
    /* Pairs that differ from the test grid's in one lookahead and stay within
     * the ring capacity (the other lookahead is 0 or 1 at most). */
    const int other_mask = DFN2_MASK_LOOKAHEAD == 0 ? 1 : 0;
    const int other_df = DFN2_DF_LOOKAHEAD == 0 ? 1 : 0;

    CHECK(dfn2_grid_validate(g) == DFN2_GRID_OK, "DFN2 test grid validates");
    CHECK(g->sample_rate == DFN2_SR && g->n_fft == DFN2_N_FFT &&
          g->hop == DFN2_HOP_LEN && g->n_bins == DFN2_N_BINS &&
          g->n_erb == DFN2_N_ERB && g->df_bins == DFN2_DF_BINS,
          "DFN2 test grid has the signal grid the buffers are sized for");
    CHECK(g->mask_lookahead == DFN2_MASK_LOOKAHEAD &&
          g->df_lookahead == DFN2_DF_LOOKAHEAD &&
          g->df_history == DFN2_DF_HISTORY && g->ring == DFN2_DF_RING,
          "DFN2 test grid has the lookahead pair and ring the test expects");
    CHECK(g->stft_scale == DFN2_NORMALIZED_STFT_SCALE &&
          g->stft_unscale == DFN2_NORMALIZED_STFT_UNSCALE,
          "DFN2 test grid has the analysis and synthesis scale of the framing");
#ifdef DFN2_GRID_KHZ
    /* train.py make_norm_alpha(sr, hop, 1.0): the literals a model trained on
     * these grids carries (config.ini [feature]). */
    CHECK(g->erb_norm_alpha == (DFN2_GRID_KHZ == 48 ? 0.989f : 0.992f) &&
          g->spec_norm_alpha == g->erb_norm_alpha,
          "DFN2 test grid has the normaliser coefficient of its model");
#endif

    /* An invalid or missing grid never starts an object. */
    bad = *g;
    bad.hop += 1;
    CHECK(dfn2_grid_validate(&bad) == DFN2_GRID_ERR_INCONSISTENT &&
          dfn2_state_init(&state, &bad, NULL, NULL) == -1 &&
          dfn2_state_init(&state, NULL, NULL, NULL) == -1,
          "DFN2 objects refuse a hand-edited or missing grid");
    CHECK(dfn2_model_io_init(&model_io, &bad) == -1 &&
          dfn2_model_io_ready_frames(&model_io) == 0u &&
          dfn2_model_io_push_features(&model_io, erb, spec) == -1,
          "DFN2 a model I/O state without a grid fails every call");
    CHECK(dfn2_model_io_init(NULL, g) == -1, "DFN2 model I/O init refuses NULL");

    /* The capacities and the lookahead range are refused by dfn2_grid_make. */
    CHECK(dfn2_grid_make(&other, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB + 1,
                         DFN2_DF_BINS, DFN2_MASK_LOOKAHEAD,
                         DFN2_DF_LOOKAHEAD) == DFN2_GRID_ERR_N_ERB &&
          dfn2_grid_make(&other, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB, DFN2_DF_BINS,
                         DFN2_MAX_MASK_LOOKAHEAD + 1,
                         DFN2_DF_LOOKAHEAD) == DFN2_GRID_ERR_MASK_LOOKAHEAD &&
          dfn2_grid_make(&other, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB, DFN2_DF_BINS,
                         DFN2_MASK_LOOKAHEAD,
                         DFN2_MAX_DF_LOOKAHEAD + 1) == DFN2_GRID_ERR_DF_LOOKAHEAD,
          "DFN2 grid refuses a band count or a lookahead the graph cannot take");
    if (DFN2_N_BINS > DFN2_MAX_DF_BINS + 1)
        CHECK(dfn2_grid_make(&other, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB,
                             DFN2_MAX_DF_BINS + 1, DFN2_MASK_LOOKAHEAD,
                             DFN2_DF_LOOKAHEAD) == DFN2_GRID_ERR_CAPACITY,
              "DFN2 grid refuses more deep-filter bins than the capacity");

    /* The descriptor is the graph contract for this grid and no other. */
    CHECK(dfn2_model_io_descriptor_default(g, &descriptor) == 0 &&
          dfn2_model_io_descriptor_validate(g, &descriptor) == 0,
          "DFN2 the default descriptor validates against its own grid");
    CHECK(dfn2_grid_make(&other, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB, DFN2_DF_BINS,
                         other_mask, 0) == DFN2_GRID_OK &&
          dfn2_model_io_descriptor_validate(&other, &descriptor) == -1,
          "DFN2 a descriptor of another mask lookahead is refused");
    CHECK(dfn2_grid_make(&other, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB, DFN2_DF_BINS,
                         DFN2_MASK_LOOKAHEAD, other_df) == DFN2_GRID_OK &&
          dfn2_model_io_descriptor_validate(&other, &descriptor) == -1,
          "DFN2 a descriptor of another deep-filter lookahead is refused");
    descriptor.erb_bands += 4;
    CHECK(dfn2_model_io_descriptor_validate(g, &descriptor) == -1,
          "DFN2 a descriptor with another band count is refused");
    descriptor.erb_bands -= 4;
    descriptor.sample_rate += 1;
    CHECK(dfn2_model_io_descriptor_validate(g, &descriptor) == -1,
          "DFN2 a descriptor with another sample rate is refused");
    CHECK(dfn2_model_io_descriptor_validate(&bad, &descriptor) == -1 &&
          dfn2_model_io_descriptor_default(&bad, &descriptor) == -1,
          "DFN2 a descriptor is neither made nor validated for an invalid grid");
    return 1;
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
    CHECK(dfn2_state_init(&dfn2, dfn2_test_grid(), NULL, NULL) == 0,
          "DFN2 state starts for the test grid");
    dfn2_set_erb_matrices(&dfn2, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    for (int k = 0; k < DFN2_DF_BINS; ++k)
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap)
            coef2[k][tap][0] = stream_tap(tap);

    memset(spec2_re, 0, sizeof(spec2_re));
    memset(spec2_im, 0, sizeof(spec2_im));
    /* With mask lookahead 0 the first frame already has heads, so there is
     * nothing to reject. */
    if (DFN2_MASK_LOOKAHEAD > 0)
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
    /* The three windows are flat runs packed at the grid's stride inside
     * capacity-sized storage: erb_window[frame][n_erb], spec_window[channel]
     * [frame][df_bins] and df_convp_window[channel][tap][df_bins]. */
    enum { WF = DFN2_MODEL_WINDOW_FRAMES };
    const DFN2Grid* grid = dfn2_test_grid();
    const size_t convp_n = (size_t)DFN2_MODEL_ENCODER_CHANNELS *
                           DFN2_MODEL_DF_PATHWAY_WINDOW * DFN2_DF_BINS;
    static DFN2ModelIOState state;
    static DFN2ModelIOState healthy;
    float erb[DFN2_N_ERB];
    float spec[2][DFN2_DF_BINS];

    CHECK(dfn2_model_io_init(&state, grid) == 0,
          "DFN2 model I/O state starts for the test grid");
    CHECK(dfn2_model_io_ready_frames(&state) ==
              (unsigned)(DFN2_MASK_LOOKAHEAD + DFN2_FRAMES_PER_CALL) &&
          dfn2_model_io_ready_frames(NULL) == 0u,
          "DFN2 the first call needs mask lookahead + frames-per-call frames");
    for (int frame = 0; frame < 3; ++frame) {
        for (int band = 0; band < DFN2_N_ERB; ++band)
            erb[band] = (float)(100 * frame + band);
        for (int channel = 0; channel < 2; ++channel)
            for (int bin = 0; bin < DFN2_DF_BINS; ++bin)
                spec[channel][bin] =
                    (float)(1000 * frame + 100 * channel + bin);
        CHECK(dfn2_model_io_push_features(&state, erb, &spec[0][0]) ==
                  (frame >= DFN2_MASK_LOOKAHEAD),
              "DFN2 model window warms up for the mask lookahead frames");
    }
    CHECK(state.erb_window[0 * DFN2_N_ERB + 7] == 7.0f &&
          state.erb_window[1 * DFN2_N_ERB + 7] == 107.0f &&
          state.erb_window[2 * DFN2_N_ERB + 7] == 207.0f,
          "DFN2 ERB model window keeps the last three pushed frames");
    CHECK(state.spec_window[(1 * WF + 0) * DFN2_DF_BINS + 9] == 109.0f &&
          state.spec_window[(1 * WF + 2) * DFN2_DF_BINS + 9] == 2109.0f,
          "DFN2 complex model window preserves channel-major layout");
    CHECK(all_zero(state.erb_window + WF * DFN2_N_ERB,
                   sizeof(state.erb_window) / sizeof(float) -
                       (size_t)WF * DFN2_N_ERB) &&
          all_zero(state.spec_window + 2 * WF * DFN2_DF_BINS,
                   sizeof(state.spec_window) / sizeof(float) -
                       (size_t)2 * WF * DFN2_DF_BINS),
          "DFN2 feature windows leave the storage past the grid's extent zero");

    /* The first complete window is [t-(2-L) .. t+L] for t = 0, L the mask
     * lookahead: frames before the stream start read as zero, whatever L is. */
    CHECK(dfn2_model_io_init(&state, grid) == 0,
          "DFN2 model I/O state starts again for the test grid");
    for (int frame = 0; frame <= DFN2_MASK_LOOKAHEAD; ++frame) {
        for (int band = 0; band < DFN2_N_ERB; ++band)
            erb[band] = (float)(10 + frame);
        for (int channel = 0; channel < 2; ++channel)
            for (int bin = 0; bin < DFN2_DF_BINS; ++bin)
                spec[channel][bin] = (float)(20 + frame);
        CHECK(dfn2_model_io_push_features(&state, erb, &spec[0][0]) ==
                  (frame == DFN2_MASK_LOOKAHEAD),
              "DFN2 heads[0] is ready exactly when frame L has been pushed");
    }
    for (int slot = 0; slot < DFN2_MODEL_INPUT_FRAMES; ++slot) {
        int source = slot - (DFN2_MODEL_INPUT_FRAMES - 1 - DFN2_MASK_LOOKAHEAD);
        float want_erb = source >= 0 ? (float)(10 + source) : 0.0f;
        float want_spec = source >= 0 ? (float)(20 + source) : 0.0f;
        CHECK(state.erb_window[slot * DFN2_N_ERB + 3] == want_erb &&
              state.spec_window[(0 * WF + slot) * DFN2_DF_BINS + 5] == want_spec &&
              state.spec_window[(1 * WF + slot) * DFN2_DF_BINS + 5] == want_spec,
              "DFN2 first window is [t-(2-L)..t+L] with zeros before the start");
    }

    CHECK(dfn2_model_io_state_generation(&state) == 1u &&
          dfn2_model_io_state_generation(NULL) == 0u,
          "DFN2 init starts the state generation at 1");
    CHECK(DFN2_MODEL_GRU_ROW_ERB == DFN2_MODEL_GRU_ROW_ENCODER +
              DFN2_MODEL_ENCODER_GRU_LAYERS &&
          DFN2_MODEL_GRU_ROW_DF == DFN2_MODEL_GRU_ROW_ERB +
              DFN2_MODEL_ERB_GRU_LAYERS &&
          DFN2_MODEL_GRU_ROWS == DFN2_MODEL_GRU_ROW_DF +
              DFN2_MODEL_DF_GRU_LAYERS &&
          sizeof(state.gru_hidden) ==
              (size_t)DFN2_MODEL_GRU_ROWS * DFN2_MODEL_GRU_HIDDEN *
                  sizeof(float),
          "DFN2 h_gru rows tile the three stacks in order");
    CHECK(dfn2_prepost_convp_window_elements(grid) == convp_n &&
          sizeof(state.df_convp_window) >= convp_n * sizeof(float),
          "DFN2 pathway window is the full kernel depth");

    /* The accelerator writes the two recurrent arrays in place. */
    memset(state.gru_hidden, 0x3c, sizeof(state.gru_hidden));
    memset(state.df_convp_window, 0x6f, sizeof(state.df_convp_window));
    healthy = state;
    CHECK(dfn2_model_io_validate_state(&state) == 0 &&
          memcmp(&state, &healthy, sizeof(state)) == 0,
          "DFN2 validating a finite state leaves every byte in place");

    state.gru_hidden[DFN2_MODEL_GRU_ROW_ERB + 1][DFN2_MODEL_GRU_HIDDEN - 1] =
        NAN;
    CHECK(dfn2_model_io_validate_state(&state) != 0,
          "DFN2 validate checks the second ERB stack layer");
    CHECK(all_zero(&state.gru_hidden[0][0],
                   (size_t)DFN2_MODEL_GRU_ROWS * DFN2_MODEL_GRU_HIDDEN) &&
          all_zero(state.df_convp_window, convp_n),
          "DFN2 non-finite state zeroes both recurrent arrays");
    CHECK(memcmp(state.erb_window, healthy.erb_window,
                 sizeof(state.erb_window)) == 0 &&
          memcmp(state.spec_window, healthy.spec_window,
                 sizeof(state.spec_window)) == 0 &&
          state.feature_frames_seen == healthy.feature_frames_seen,
          "DFN2 state refusal leaves the feature windows and counter alone");
    CHECK(dfn2_model_io_state_generation(&state) ==
              healthy.state_generation + 1u,
          "DFN2 a refused validation advances the state generation once");

    state = healthy;
    state.df_convp_window[convp_n - 1] = INFINITY;   /* last element of the packed run */
    CHECK(dfn2_model_io_validate_state(&state) != 0 &&
          all_zero(&state.gru_hidden[0][0],
                   (size_t)DFN2_MODEL_GRU_ROWS * DFN2_MODEL_GRU_HIDDEN),
          "DFN2 refuses a non-finite pathway window and zeroes the batch");

    CHECK(dfn2_model_io_validate_state(NULL) != 0,
          "DFN2 validate refuses a null state");
    state = healthy;
    CHECK(dfn2_model_io_validate_arrays(NULL, state.df_convp_window,
                                        DFN2_DF_BINS) != 0 &&
          dfn2_model_io_validate_arrays(state.gru_hidden, NULL,
                                        DFN2_DF_BINS) != 0 &&
          memcmp(&state, &healthy, sizeof(state)) == 0,
          "DFN2 validate_arrays refuses a null array and touches nothing");
    CHECK(dfn2_model_io_validate_arrays(state.gru_hidden,
                                        state.df_convp_window,
                                        DFN2_DF_BINS) == 0,
          "DFN2 validate_arrays accepts finite arrays");

    /* reset and zero_recurrent each advance the generation exactly once. */
    state = healthy;
    dfn2_model_io_zero_recurrent(&state);
    CHECK(all_zero(&state.gru_hidden[0][0],
                   (size_t)DFN2_MODEL_GRU_ROWS * DFN2_MODEL_GRU_HIDDEN) &&
          memcmp(state.erb_window, healthy.erb_window,
                 sizeof(state.erb_window)) == 0 &&
          state.state_generation == healthy.state_generation + 1u,
          "DFN2 zero_recurrent clears the recurrent arrays and counts");
    state = healthy;
    dfn2_model_io_reset(&state);
    CHECK(all_zero(state.erb_window, sizeof(state.erb_window) / sizeof(float)) &&
          state.feature_frames_seen == 0u &&
          state.state_generation == healthy.state_generation + 1u,
          "DFN2 reset returns to init values and counts");
    CHECK(state.n_erb == DFN2_N_ERB && state.df_bins == DFN2_DF_BINS &&
          state.mask_lookahead == DFN2_MASK_LOOKAHEAD &&
          dfn2_model_io_ready_frames(&state) ==
              (unsigned)(DFN2_MASK_LOOKAHEAD + DFN2_FRAMES_PER_CALL),
          "DFN2 reset keeps the grid dimensions of the state");
    dfn2_model_io_reset(NULL);
    dfn2_model_io_zero_recurrent(NULL);

    /* The generation counts modulo 2^32, whichever call zeroes the state. */
    state = healthy;
    state.state_generation = 0xffffffffu;
    dfn2_model_io_reset(&state);
    CHECK(dfn2_model_io_state_generation(&state) == 0u,
          "DFN2 reset wraps the generation modulo 2^32");
    state.state_generation = 0xffffffffu;
    dfn2_model_io_zero_recurrent(&state);
    CHECK(dfn2_model_io_state_generation(&state) == 0u,
          "DFN2 zero_recurrent wraps the generation modulo 2^32");
    state = healthy;
    state.state_generation = 0xffffffffu;
    state.gru_hidden[0][0] = NAN;
    CHECK(dfn2_model_io_validate_state(&state) != 0 &&
          dfn2_model_io_state_generation(&state) == 0u,
          "DFN2 a refused validation wraps the generation modulo 2^32");
    return 1;
}

static int test_dfn2(uint64_t* digest)
{
    static DFN2State state;
    float input[DFN2_HOP_LEN];
    float spec_re[DFN2_N_BINS], spec_im[DFN2_N_BINS];
    enum { FRAMES = 24 };
    static float history_re[FRAMES][DFN2_N_BINS], history_im[FRAMES][DFN2_N_BINS];
    float enhanced_re[DFN2_N_BINS], enhanced_im[DFN2_N_BINS];
    float output[DFN2_HOP_LEN];
    float erb[DFN2_N_ERB], feature_spec[2 * DFN2_DF_BINS];
    float mask[DFN2_N_ERB];
    float coefs[DFN2_DF_BINS][DFN2_DF_ORDER][2] = {{{0}}};
    float max_spectral_error = 0.0f;
    int valid_frames = 0;

    dfn_test_build_erb();
    {
        AcStft* stft;
        AcIstft* istft;
        CHECK(dfn2_framing(&stft, &istft), "DFN2 framing objects");
        CHECK(dfn2_state_init(&state, dfn2_test_grid(), stft, istft) == 0,
              "DFN2 state starts for the test grid");
    }
    dfn2_set_erb_matrices(&state, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    for (int b = 0; b < DFN2_N_ERB; ++b) mask[b] = 1.0f;
    for (int k = 0; k < DFN2_DF_BINS; ++k)
        coefs[k][DFN2_DF_HISTORY][0] = 1.0f;

    for (int frame = 0; frame < FRAMES; ++frame) {
        int valid;
        for (int i = 0; i < DFN2_HOP_LEN; ++i)
            input[i] = signal_sample((int64_t)frame * DFN2_HOP_LEN + i,
                                     DFN2_SR);
        dfn2_analysis(&state, input, spec_re, spec_im);
        memcpy(history_re[frame], spec_re, sizeof(spec_re));
        memcpy(history_im[frame], spec_im, sizeof(spec_im));
        dfn2_compute_features(&state, spec_re, spec_im, erb, feature_spec);
        CHECK(all_finite(erb, DFN2_N_ERB), "DFN2 finite ERB features");
        CHECK(all_finite(feature_spec, 2 * DFN2_DF_BINS),
              "DFN2 finite complex features");
        valid = dfn2_compose(&state, spec_re, spec_im, mask,
                             &coefs[0][0][0], 1.0f,
                             enhanced_re, enhanced_im);
        /* The aligned entry emits from call DFN2_DF_LOOKAHEAD on: its target
         * is that many frames behind the newest spectrum. */
        CHECK(valid == (frame >= DFN2_DF_LOOKAHEAD),
              "DFN2 aligned compose is valid from call DF_LOOKAHEAD on");
        valid_frames += valid;
        if (valid) {
            /* The target is the spectrum DFN2_DF_LOOKAHEAD frames back. */
            const float* previous_re = history_re[frame - DFN2_DF_LOOKAHEAD];
            const float* previous_im = history_im[frame - DFN2_DF_LOOKAHEAD];
            for (int k = 0; k < DFN2_N_BINS; ++k) {
                float er = fabsf(enhanced_re[k] - previous_re[k]);
                float ei = fabsf(enhanced_im[k] - previous_im[k]);
                if (er > max_spectral_error) max_spectral_error = er;
                if (ei > max_spectral_error) max_spectral_error = ei;
            }
            dfn2_apply_atten_lim(DFN2_N_BINS, previous_re, previous_im,
                                 enhanced_re, enhanced_im, -100.0f);
            dfn2_synthesis(&state, enhanced_re, enhanced_im, output);
            CHECK(all_finite(output, DFN2_HOP_LEN), "DFN2 finite WOLA output");
            *digest = hash_bytes(*digest, erb, sizeof(erb));
            *digest = hash_bytes(*digest, feature_spec, sizeof(feature_spec));
            *digest = hash_bytes(*digest, output, sizeof(output));
        }
    }
    CHECK(valid_frames == FRAMES - DFN2_DF_LOOKAHEAD,
          "DFN2 aligned compose drops exactly DF_LOOKAHEAD warm-up frames");
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
        AcStft* stft;
        AcIstft* istft;
        CHECK(gtcrn_framing(&stft, &istft), "GTCRN framing objects");
        gtcrn_process_init(&state, stft, istft);
    }
    for (int frame = 0; frame < 24; ++frame) {
        for (int i = 0; i < GTCRN_HOP_LEN; ++i)
            input[i] = signal_sample((int64_t)frame * GTCRN_HOP_LEN + i,
                                     GTCRN_SR);
        gtcrn_analysis(&state, input, spectrum);
        gtcrn_synthesis(&state, (const float (*)[2])spectrum, output);
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
    const size_t tra_n = sizeof(state.h_tra) / sizeof(float);
    const size_t dpgrnn_n = sizeof(state.h_dpgrnn) / sizeof(float);
    float* tensor[GTCRN_MODEL_CONV_STATES + 2];
    size_t tensor_n[GTCRN_MODEL_CONV_STATES + 2];
    size_t conv_n[GTCRN_MODEL_CONV_STATES];
    int i;
    int c;
    CHECK(sizeof(state) == 84864u + sizeof(unsigned int),
          "GTCRN caller-owned state is the eight tensors plus the counter");
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
    for (i = 0; i < GTCRN_MODEL_CONV_STATES; ++i) {
        tensor[i] = conv[i];
        tensor_n[i] = conv_n[i];
    }
    tensor[GTCRN_MODEL_CONV_STATES] = &state.h_tra[0][0][0];
    tensor_n[GTCRN_MODEL_CONV_STATES] = tra_n;
    tensor[GTCRN_MODEL_CONV_STATES + 1] = &state.h_dpgrnn[0][0][0];
    tensor_n[GTCRN_MODEL_CONV_STATES + 1] = dpgrnn_n;
    CHECK(tra_n == (size_t)GTCRN_MODEL_TRA_GRUS * GTCRN_MODEL_TRA_HIDDEN &&
          dpgrnn_n == (size_t)GTCRN_MODEL_DPGRNN_GRUS *
                          GTCRN_MODEL_DPGRNN_FREQ * GTCRN_MODEL_DPGRNN_HIDDEN,
          "GTCRN h_tra and h_dpgrnn are the rank-4 graph tensors");
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
        gtcrn_model_input((const float (*)[2])spec_in, &fwd[0][0], mag, re_b, im_b);
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
        gtcrn_model_output((const float (*)[2])mask_erb, &inv[0][0],
                           (const float (*)[2])spec_in, enhanced);
        CHECK(enhanced[10][0] == 3.0f && enhanced[10][1] == 4.0f,
              "GTCRN host output: unity low-band CRM reproduces the "
              "spectrum bin");
        CHECK(fabsf(enhanced[GTCRN_MODEL_ERB_KEPT + 35][0] -
                    2.0f * 0.5f * inv[7][35]) < 1e-6f,
              "GTCRN host output: high bands expand through the "
              "caller-loaded inverse matrix");
    }
    gtcrn_model_state_init(&state);
    CHECK(state.conv_dec0[15][GTCRN_MODEL_CONV_WINDOW_2 - 1][32] == 0.0f &&
          state.h_tra[GTCRN_MODEL_TRA_GRUS - 1][0][15] == 0.0f &&
          state.h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS - 1][32]
                        [GTCRN_MODEL_DPGRNN_HIDDEN - 1] == 0.0f,
          "GTCRN model state starts at zero");
    CHECK(gtcrn_model_state_generation(&state) == 1u &&
          gtcrn_model_state_generation(NULL) == 0u,
          "GTCRN init starts the state generation at 1");
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

    /* The eight tensors tile the struct up to the counter, so checking each
     * one covers every state byte. */
    {
        size_t total = 0;
        for (i = 0; i < GTCRN_MODEL_CONV_STATES + 2; ++i) total += tensor_n[i];
        CHECK(total * sizeof(float) + sizeof(unsigned int) == sizeof(state),
              "GTCRN state tensors tile the whole state struct");
    }

    /* One NaN or Inf at the first or last element of any single tensor
     * zeroes the WHOLE state and advances the generation by one. The state is
     * nonzero everywhere else (0x41 pattern), so a partial reset or an
     * unchecked tensor is visible. */
    for (c = 0; c < 4; ++c) {
        const float bad = (c & 1) ? INFINITY : NAN;
        int cls_ok = 1;
        for (i = 0; i < GTCRN_MODEL_CONV_STATES + 2; ++i) {
            const unsigned int generation = 7u;
            memset(&state, 0x41, sizeof(state));
            state.state_generation = generation;
            tensor[i][(c < 2) ? 0 : tensor_n[i] - 1] = bad;
            cls_ok &= gtcrn_model_state_validate(&state) == -1 &&
                      state.state_generation == generation + 1u &&
                      memcmp(&state, &zeros, offsetof(GTCRNModelState,
                                                      state_generation)) == 0;
        }
        CHECK(cls_ok, "GTCRN non-finite element in any tensor returns -1, "
                      "zeroes the whole state and counts once");
    }

    /* A finite state must still validate after a refusal, or the guard could
     * be a permanent latch rather than a per-call check, and a healthy
     * validation must not touch the generation. */
    memset(&state, 0x41, sizeof(state));
    memset(&expect, 0x41, sizeof(expect));
    CHECK(gtcrn_model_state_validate(&state) == 0 &&
          memcmp(&state, &expect, sizeof(state)) == 0,
          "GTCRN validate still accepts a finite state after a refusal");

    memset(&state, 0x41, sizeof(state));
    state.state_generation = 41u;
    gtcrn_model_state_reset(&state);
    CHECK(memcmp(&state, &zeros, offsetof(GTCRNModelState,
                                          state_generation)) == 0 &&
          state.state_generation == 42u &&
          gtcrn_model_state_generation(&state) == 42u,
          "GTCRN reset zeroes the tensors and advances the generation");
    gtcrn_model_state_reset(NULL);

    /* The generation counts modulo 2^32. */
    state.state_generation = 0xffffffffu;
    gtcrn_model_state_reset(&state);
    CHECK(gtcrn_model_state_generation(&state) == 0u,
          "GTCRN reset wraps the generation modulo 2^32");
    state.state_generation = 0xffffffffu;
    state.h_tra[0][0][0] = NAN;
    CHECK(gtcrn_model_state_validate(&state) == -1 &&
          gtcrn_model_state_generation(&state) == 0u,
          "GTCRN a refused validation wraps the generation modulo 2^32");
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
    CHECK(dfn2_state_init(&st_a, dfn2_test_grid(), NULL, NULL) == 0 &&
          dfn2_state_init(&st_b, dfn2_test_grid(), NULL, NULL) == 0,
          "DFN2 states start for the test grid");
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
    CHECK(dfn2_state_init(&st_a, dfn2_test_grid(), NULL, NULL) == 0 &&
          dfn2_state_init(&st_b, dfn2_test_grid(), NULL, NULL) == 0,
          "DFN2 states start again for the test grid");
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
    enum { BINS = DFN2_N_BINS, BANDS = DFN2_N_ERB };
#if DFN2_N_ERB < 24 || DFN2_N_BINS < 129
#error "test_dfn_erb_spans_exact addresses fixed bands (up to 20) and bins (up to 58): it needs n_erb >= 24 and n_bins >= 129"
#endif
    /* Bin positions of the shipped 513-bin grid, scaled to this grid's width
     * (unchanged on the 48 kHz grid). */
#define BIN_AT(k513) ((k513) * BINS / 513)
    static DFN2State st;
    static float fwd[BINS][BANDS], inv[BANDS][BINS], coefs[DFN2_DF_BINS * DFN2_DF_ORDER * 2];
    float re[BINS], im[BINS], fe[BANDS], fs[2 * DFN2_DF_BINS], mask[BANDS];
    float out_re[BINS], out_im[BINS], ref_state[BANDS], erb[BANDS], ref_gain[BINS];
    const DFN2Grid* grid = dfn2_test_grid();
    const float scale2 = grid->stft_scale * grid->stft_scale;
    const float alpha = grid->erb_norm_alpha;
    uint32_t lcg = 12345u;

    dfn_test_build_erb();
    memset(coefs, 0, sizeof(coefs));
    for (int variant = 0; variant < 3; ++variant) {
        memcpy(fwd, dfn_test_fwd, sizeof(fwd));
        memcpy(inv, dfn_test_inv, sizeof(inv));
        if (variant == 1) {
            fwd[10][5] = NAN;
            fwd[BIN_AT(300)][0] = INFINITY;
            memset(fwd[7], 0, sizeof(fwd[7]));
            for (int k = 0; k < BINS; ++k) fwd[k][12] = 0.0f;   /* an empty band */
            fwd[BIN_AT(200)][20] = 0.0f;       /* a hole inside a span */
            inv[3][40] = -0.0f;
            inv[20][0] = 0.5f;         /* a far outlier widens the span */
        } else if (variant == 2) {
            for (int k = 0; k < BINS; ++k) fwd[k][(k * 7) % BANDS] += 0.125f;
            for (int k = 0; k < BINS; ++k) fwd[k][3] = 0.0f;
            fwd[BINS - 1][3] = 0.5f;   /* a band fed by the last bin only */
            inv[BANDS - 1][3] = 0.25f;
        }
        CHECK(dfn2_state_init(&st, grid, NULL, NULL) == 0,
              "DFN2 state starts for the test grid");
        dfn2_set_erb_matrices(&st, &fwd[0][0], &inv[0][0]);
        memcpy(ref_state, st.erb_norm_state, sizeof(ref_state));
        for (int frame = 0; frame < 24; ++frame) {
            for (int k = 0; k < BINS; ++k) {
                lcg = lcg * 1664525u + 1013904223u;
                re[k] = (float)(int32_t)lcg * 4.6566e-10f;
                lcg = lcg * 1664525u + 1013904223u;
                im[k] = (float)(int32_t)lcg * 4.6566e-10f;
            }
            if (frame % 4 == 1) re[37 + frame] = 1e30f;   /* power -> Inf */
            if (frame % 6 == 3) im[BINS / 2 - 6] = NAN;
            if (frame % 5 == 2) re[BINS - 1] = -INFINITY;

            dfn2_compute_features(&st, re, im, fe, fs);
            memset(erb, 0, sizeof(erb));
            for (int k = 0; k < BINS; ++k) {
                float p = (re[k] * re[k] + im[k] * im[k]) * scale2;
                for (int b = 0; b < BANDS; ++b) erb[b] += p * fwd[k][b];
            }
            for (int b = 0; b < BANDS; ++b) {
                float db = 10.0f * log10f(erb[b] + DFN2_ERB_LOG_FLOOR);
                float mean = alpha * ref_state[b] + (1.0f - alpha) * db;
                float ref = (db - mean) / DFN2_ERB_NORM_SCALE_DB;
                ref_state[b] = mean;
                CHECK(memcmp(&fe[b], &ref, sizeof(ref)) == 0,
                      "DFN2 ERB features over the nonzero spans equal the dense sum");
            }

            for (int b = 0; b < BANDS; ++b) mask[b] = 0.3f + 0.02f * (float)b;
            if (frame % 3 == 0) mask[frame % BANDS] = (frame & 1) ? INFINITY : NAN;
            if (frame % 7 == 2) mask[0] = -INFINITY;
            if (frame % 10 == 9) mask[BANDS - 1] = -0.0f;
            (void)dfn2_compose(&st, re, im, mask, coefs, 0.5f, out_re, out_im);
            memset(ref_gain, 0, sizeof(ref_gain));
            for (int b = 0; b < BANDS; ++b)
                for (int k = 0; k < BINS; ++k) ref_gain[k] += inv[b][k] * mask[b];
            CHECK(memcmp(st.scratch_bin_gain, ref_gain, sizeof(ref_gain)) == 0,
                  "DFN2 mask expansion over the nonzero spans equals the dense sum");
        }
    }
#undef BIN_AT
    return 1;
}

/* State evolution through the in-place boundary. A deterministic stand-in for
 * the accelerator rewrites the caller-owned state the way the graph does
 * (recurrent rows from their own previous values, conv windows shifted by one
 * frame, all inputs read before any output is written) through the struct's
 * typed fields; a reference on plain strided vectors must agree byte for byte
 * after every hop (commit, refused commit, reset). The arrays are hashed into
 * the digest that test-simd compares across builds. */

static float standin_row(float previous, int hop, int row, int column)
{
    return previous * 0.5f + 0.003f * (float)(hop + 1) +
           0.001f * (float)(row + 1) + 0.00001f * (float)column;
}

static float standin_frame(int hop, int tensor, int channel, int column)
{
    return 0.02f * (float)(hop + 1) + 0.001f * (float)(tensor + 1) +
           0.0001f * (float)channel + 0.000001f * (float)column;
}

/* Reference window update on a flat [channels][k][width] vector. */
static void reference_window(float* flat, int channels, int k, int width,
                             int hop, int tensor)
{
    for (int c = 0; c < channels; ++c) {
        float* base = flat + (size_t)c * k * width;
        memmove(base, base + width, (size_t)(k - 1) * width * sizeof(float));
        for (int column = 0; column < width; ++column)
            base[(size_t)(k - 1) * width + column] =
                standin_frame(hop, tensor, c, column);
    }
}

static int test_dfn2_state_evolution(uint64_t* digest)
{
    enum {
        ROWS = DFN2_MODEL_GRU_ROWS, HIDDEN = DFN2_MODEL_GRU_HIDDEN,
        CH = DFN2_MODEL_ENCODER_CHANNELS, K = DFN2_MODEL_DF_PATHWAY_WINDOW,
        BINS = DFN2_DF_BINS, HOPS = 14
    };
    /* The pathway window is the packed run [CH][K][BINS] at the start of the
     * state's storage. */
    static DFN2ModelIOState state;
    static float ref_gru[ROWS * HIDDEN];
    static float ref_win[CH * K * BINS];
    static float previous[ROWS][HIDDEN];
    unsigned int generation;
    int zeroings = 0;

    CHECK(dfn2_model_io_init(&state, dfn2_test_grid()) == 0,
          "DFN2 model I/O state starts for the test grid");
    CHECK(sizeof(state.gru_hidden) == sizeof(ref_gru) &&
          sizeof(state.df_convp_window) >= sizeof(ref_win) &&
          dfn2_prepost_convp_window_elements(dfn2_test_grid()) ==
              sizeof(ref_win) / sizeof(float),
          "DFN2 state arrays hold exactly the graph tensors");
    generation = dfn2_model_io_state_generation(&state);
    for (int hop = 0; hop < HOPS; ++hop) {
        int expect_refusal = 0;
        /* The stand-in reads every state input before it writes any output. */
        memcpy(previous, state.gru_hidden, sizeof(previous));
        for (int row = 0; row < ROWS; ++row)
            for (int j = 0; j < HIDDEN; ++j)
                state.gru_hidden[row][j] =
                    standin_row(previous[row][j], hop, row, j);
        for (int c = 0; c < CH; ++c) {
            float* channel = state.df_convp_window + (size_t)c * K * BINS;
            for (int t = 0; t + 1 < K; ++t)
                memcpy(channel + (size_t)t * BINS, channel + (size_t)(t + 1) * BINS,
                       BINS * sizeof(float));
            for (int b = 0; b < BINS; ++b)
                channel[(size_t)(K - 1) * BINS + b] = standin_frame(hop, 0, c, b);
        }
        for (int row = 0; row < ROWS; ++row)
            for (int j = 0; j < HIDDEN; ++j)
                ref_gru[row * HIDDEN + j] =
                    standin_row(ref_gru[row * HIDDEN + j], hop, row, j);
        reference_window(ref_win, CH, K, BINS, hop, 0);

        if (hop == 6) {
            state.df_convp_window[(size_t)CH * K * BINS - 1] = (float)NAN;
            expect_refusal = 1;
        }
        if (hop == 9) {
            state.gru_hidden[DFN2_MODEL_GRU_ROW_DF + 1][HIDDEN - 1] =
                (float)INFINITY;
            expect_refusal = 1;
        }
        CHECK((dfn2_model_io_validate_state(&state) != 0) == expect_refusal,
              "DFN2 validate refuses exactly the non-finite hops");
        if (expect_refusal) {
            ++zeroings;
            memset(ref_gru, 0, sizeof(ref_gru));
            memset(ref_win, 0, sizeof(ref_win));
        }
        if (hop == 11) {
            dfn2_model_io_zero_recurrent(&state);
            memset(ref_gru, 0, sizeof(ref_gru));
            memset(ref_win, 0, sizeof(ref_win));
            ++zeroings;
        }
        CHECK(memcmp(&state.gru_hidden[0][0], ref_gru, sizeof(ref_gru)) == 0,
              "DFN2 h_gru equals the flat reference after the hop");
        CHECK(memcmp(state.df_convp_window, ref_win, sizeof(ref_win)) == 0,
              "DFN2 pathway window equals the flat reference after the hop");
        CHECK(dfn2_model_io_state_generation(&state) ==
                  generation + (unsigned)zeroings,
              "DFN2 generation counts the zeroings");
        *digest = hash_bytes(*digest, state.gru_hidden,
                             sizeof(state.gru_hidden));
        *digest = hash_bytes(*digest, state.df_convp_window, sizeof(ref_win));
    }
    /* The row offsets address the rows the exporter slices: encoder GRU at
     * row 0, the two ERB decoder layers from row 1, the two DF decoder layers
     * from row 3. */
    for (int row = 0; row < ROWS; ++row)
        for (int j = 0; j < HIDDEN; ++j)
            state.gru_hidden[row][j] = standin_row(0.0f, 0, row, j);
    CHECK(state.gru_hidden[DFN2_MODEL_GRU_ROW_ENCODER][5] ==
              standin_row(0.0f, 0, 0, 5) &&
          state.gru_hidden[DFN2_MODEL_GRU_ROW_ERB][5] ==
              standin_row(0.0f, 0, 1, 5) &&
          state.gru_hidden[DFN2_MODEL_GRU_ROW_DF][5] ==
              standin_row(0.0f, 0, 3, 5),
          "DFN2 GRU row offsets address rows 0, 1 and 3");
    return 1;
}

static int test_gtcrn_state_evolution(uint64_t* digest)
{
    enum { HOPS = 14, CH = GTCRN_MODEL_CONV_CHANNELS,
           F = GTCRN_MODEL_CONV_FREQ };
    static const int window_of[GTCRN_MODEL_CONV_STATES] = {
        GTCRN_MODEL_CONV_WINDOW_0, GTCRN_MODEL_CONV_WINDOW_1,
        GTCRN_MODEL_CONV_WINDOW_2, GTCRN_MODEL_CONV_WINDOW_2,
        GTCRN_MODEL_CONV_WINDOW_1, GTCRN_MODEL_CONV_WINDOW_0};
    static GTCRNModelState state;
    static float ref_conv[GTCRN_MODEL_CONV_STATES][CH * 11 * F];
    static float ref_tra[GTCRN_MODEL_TRA_GRUS * GTCRN_MODEL_TRA_HIDDEN];
    static float ref_dp[GTCRN_MODEL_DPGRNN_GRUS * GTCRN_MODEL_DPGRNN_FREQ *
                        GTCRN_MODEL_DPGRNN_HIDDEN];
    static float previous_tra[GTCRN_MODEL_TRA_GRUS][1]
                             [GTCRN_MODEL_TRA_HIDDEN];
    static float previous_dp[GTCRN_MODEL_DPGRNN_GRUS][GTCRN_MODEL_DPGRNN_FREQ]
                            [GTCRN_MODEL_DPGRNN_HIDDEN];
    float* conv[GTCRN_MODEL_CONV_STATES];
    size_t conv_n[GTCRN_MODEL_CONV_STATES];
    unsigned int generation;
    int zeroings = 0;

    gtcrn_model_state_init(&state);
    conv[0] = &state.conv_enc0[0][0][0];
    conv[1] = &state.conv_enc1[0][0][0];
    conv[2] = &state.conv_enc2[0][0][0];
    conv[3] = &state.conv_dec0[0][0][0];
    conv[4] = &state.conv_dec1[0][0][0];
    conv[5] = &state.conv_dec2[0][0][0];
    conv_n[0] = sizeof(state.conv_enc0);
    conv_n[1] = sizeof(state.conv_enc1);
    conv_n[2] = sizeof(state.conv_enc2);
    conv_n[3] = sizeof(state.conv_dec0);
    conv_n[4] = sizeof(state.conv_dec1);
    conv_n[5] = sizeof(state.conv_dec2);
    for (int i = 0; i < GTCRN_MODEL_CONV_STATES; ++i)
        CHECK(conv_n[i] == (size_t)CH * window_of[i] * F * sizeof(float),
              "GTCRN conv field holds the graph window");
    CHECK(sizeof(state.h_tra) == sizeof(ref_tra) &&
          sizeof(state.h_dpgrnn) == sizeof(ref_dp),
          "GTCRN recurrent fields hold the graph tensors");
    generation = gtcrn_model_state_generation(&state);

#define GTCRN_STANDIN_WINDOW(field, tensor, hop_, k_)                       \
    do {                                                                    \
        for (int c = 0; c < CH; ++c) {                                      \
            for (int t = 0; t + 1 < (k_); ++t)                              \
                memcpy(state.field[c][t], state.field[c][t + 1],            \
                       sizeof(state.field[c][t]));                          \
            for (int f = 0; f < F; ++f)                                     \
                state.field[c][(k_) - 1][f] =                               \
                    standin_frame((hop_), (tensor), c, f);                  \
        }                                                                   \
    } while (0)

    for (int hop = 0; hop < HOPS; ++hop) {
        int expect_refusal = 0;
        memcpy(previous_tra, state.h_tra, sizeof(previous_tra));
        memcpy(previous_dp, state.h_dpgrnn, sizeof(previous_dp));
        GTCRN_STANDIN_WINDOW(conv_enc0, 0, hop, GTCRN_MODEL_CONV_WINDOW_0);
        GTCRN_STANDIN_WINDOW(conv_enc1, 1, hop, GTCRN_MODEL_CONV_WINDOW_1);
        GTCRN_STANDIN_WINDOW(conv_enc2, 2, hop, GTCRN_MODEL_CONV_WINDOW_2);
        GTCRN_STANDIN_WINDOW(conv_dec0, 3, hop, GTCRN_MODEL_CONV_WINDOW_2);
        GTCRN_STANDIN_WINDOW(conv_dec1, 4, hop, GTCRN_MODEL_CONV_WINDOW_1);
        GTCRN_STANDIN_WINDOW(conv_dec2, 5, hop, GTCRN_MODEL_CONV_WINDOW_0);
        for (int row = 0; row < GTCRN_MODEL_TRA_GRUS; ++row)
            for (int j = 0; j < GTCRN_MODEL_TRA_HIDDEN; ++j)
                state.h_tra[row][0][j] =
                    standin_row(previous_tra[row][0][j], hop, row, j);
        for (int row = 0; row < GTCRN_MODEL_DPGRNN_GRUS; ++row)
            for (int f = 0; f < GTCRN_MODEL_DPGRNN_FREQ; ++f)
                for (int j = 0; j < GTCRN_MODEL_DPGRNN_HIDDEN; ++j)
                    state.h_dpgrnn[row][f][j] = standin_row(
                        previous_dp[row][f][j], hop, row,
                        f * GTCRN_MODEL_DPGRNN_HIDDEN + j);

        for (int i = 0; i < GTCRN_MODEL_CONV_STATES; ++i)
            reference_window(ref_conv[i], CH, window_of[i], F, hop, i);
        for (int row = 0; row < GTCRN_MODEL_TRA_GRUS; ++row)
            for (int j = 0; j < GTCRN_MODEL_TRA_HIDDEN; ++j)
                ref_tra[row * GTCRN_MODEL_TRA_HIDDEN + j] = standin_row(
                    ref_tra[row * GTCRN_MODEL_TRA_HIDDEN + j], hop, row, j);
        for (int row = 0; row < GTCRN_MODEL_DPGRNN_GRUS; ++row)
            for (int column = 0; column < GTCRN_MODEL_DPGRNN_FREQ *
                                          GTCRN_MODEL_DPGRNN_HIDDEN;
                 ++column) {
                float* cell = &ref_dp[row * GTCRN_MODEL_DPGRNN_FREQ *
                                          GTCRN_MODEL_DPGRNN_HIDDEN + column];
                *cell = standin_row(*cell, hop, row, column);
            }

        if (hop == 5) {
            state.h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS - 1][32][7] = (float)NAN;
            expect_refusal = 1;
        }
        if (hop == 9) {
            state.conv_dec1[CH - 1][GTCRN_MODEL_CONV_WINDOW_1 - 1][F - 1] =
                (float)INFINITY;
            expect_refusal = 1;
        }
        CHECK((gtcrn_model_state_validate(&state) != 0) == expect_refusal,
              "GTCRN validate refuses exactly the non-finite hops");
        if (expect_refusal) {
            ++zeroings;
            memset(ref_conv, 0, sizeof(ref_conv));
            memset(ref_tra, 0, sizeof(ref_tra));
            memset(ref_dp, 0, sizeof(ref_dp));
        }
        if (hop == 11) {
            gtcrn_model_state_reset(&state);
            ++zeroings;
            memset(ref_conv, 0, sizeof(ref_conv));
            memset(ref_tra, 0, sizeof(ref_tra));
            memset(ref_dp, 0, sizeof(ref_dp));
        }
        for (int i = 0; i < GTCRN_MODEL_CONV_STATES; ++i) {
            CHECK(memcmp(conv[i], ref_conv[i], conv_n[i]) == 0,
                  "GTCRN conv window equals the flat reference");
            *digest = hash_bytes(*digest, conv[i], conv_n[i]);
        }
        CHECK(memcmp(&state.h_tra[0][0][0], ref_tra, sizeof(ref_tra)) == 0,
              "GTCRN h_tra equals the flat reference");
        CHECK(memcmp(&state.h_dpgrnn[0][0][0], ref_dp, sizeof(ref_dp)) == 0,
              "GTCRN h_dpgrnn equals the flat reference");
        CHECK(gtcrn_model_state_generation(&state) ==
                  generation + (unsigned)zeroings,
              "GTCRN generation counts the zeroings");
        *digest = hash_bytes(*digest, state.h_tra, sizeof(state.h_tra));
        *digest = hash_bytes(*digest, state.h_dpgrnn, sizeof(state.h_dpgrnn));
    }
#undef GTCRN_STANDIN_WINDOW
    return 1;
}

static int run_all_tests(void)
{
    uint64_t digest = UINT64_C(1469598103934665603);
    CHECK(test_dfn2_grid_contract(), "DFN2 test grid contract");
    CHECK(test_dfn_erb_spans_exact(),
          "DFN2 ERB nonzero spans are exact");
    CHECK(test_dfn2_model_io(), "DFN2 stateless model I/O");
    CHECK(test_dfn2_state_evolution(&digest), "DFN2 state evolution");
    CHECK(test_dfn_stream_alignment(), "DFN streaming head alignment");
    CHECK(test_dfn_dual_state_disjoint(),
          "DFN2 estimation/application branches are disjoint");
    CHECK(test_dfn2(&digest), "DFN2 C pre/post");
    CHECK(test_gtcrn_model_state(), "GTCRN stateless model I/O");
    CHECK(test_gtcrn_state_evolution(&digest), "GTCRN state evolution");
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
    if (argc == 2 && strcmp(argv[1], "--state-only") == 0) {
        uint64_t digest = UINT64_C(1469598103934665603);
        return test_dfn2_state_evolution(&digest) &&
                       test_gtcrn_state_evolution(&digest)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    return run_all_tests() ? EXIT_SUCCESS : EXIT_FAILURE;
}
