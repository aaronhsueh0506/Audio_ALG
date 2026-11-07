#define _POSIX_C_SOURCE 200809L
/* C pre/post-processing contract test.
 *
 * The DeepFilterNet2 grid and lookahead pair this program runs are chosen when it
 * is built (dfn2_test_grid.h: -DDFN2_GRID_KHZ=16|32|48, -DDFN2_MASK_LOOKAHEAD,
 * -DDFN2_DF_LOOKAHEAD); the library is handed the same grid at run time. The
 * digest printed for the shipped 48 kHz 1/1 grid is a recorded value; the other
 * grids print their own. The recurrent state is the host's: float buffers
 * registered with the library (DFN2: dfn2_test_grid.h, GTCRN: gtcrn_test_grid.h),
 * and every "nothing changed" check compares those buffers and the carved block
 * byte for byte, never an object header alone. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_process.h"
#include "DeepFilterNet2/dfn2_model_io.h"
#include "GTCRN/gtcrn_process.h"
#include "ainr_test_hop.h"
#include "dfn2_test_grid.h"
#include "gtcrn_test_grid.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
        return 0;                                                           \
    }                                                                       \
} while (0)

/* A 16-byte aligned block filled with a poison pattern, so an init that leaves a
 * byte it should zero is visible. */
static void* aligned_block(size_t bytes)
{
    void* block = NULL;
    if (posix_memalign(&block, 16, bytes ? bytes : 16u) != 0) return NULL;
    memset(block, 0xA5, bytes);
    return block;
}

static void* duplicate(const void* data, size_t bytes)
{
    void* copy = malloc(bytes ? bytes : 1u);
    if (copy) memcpy(copy, data, bytes);
    return copy;
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
 * run-time refusals: an invalid grid never starts an object, and a descriptor or
 * a spec of another grid is refused. */
static int test_dfn2_grid_contract(void)
{
    const DFN2Grid* g = dfn2_test_grid();
    static DFN2State state;
    static DFN2ModelIOState model_io;
    const size_t state_bytes = dfn2_state_mem_size(g);
    const size_t io_bytes = dfn2_model_io_mem_size(g);
    void* state_block = aligned_block(state_bytes);
    void* io_block = aligned_block(io_bytes);
    float erb[DFN2_N_ERB] = {0};
    float spec[2 * DFN2_DF_BINS] = {0};
    DFN2ModelIoDescriptor descriptor;
    DFN2GridSpec spec_other;
    DFN2Grid other, bad;
    /* Pairs that differ from the test grid's in one lookahead. */
    const int other_mask = DFN2_MASK_LOOKAHEAD == 0 ? 1 : 0;
    const int other_df = DFN2_DF_LOOKAHEAD == 0 ? 1 : 0;

    CHECK(state_block != NULL && io_block != NULL, "DFN2 blocks allocate");
    CHECK(dfn2_grid_validate(g) == DFN2_GRID_OK, "DFN2 test grid validates");
    CHECK(g->sample_rate == DFN2_SR && g->n_fft == DFN2_N_FFT &&
          g->hop == DFN2_HOP_LEN && g->n_bins == DFN2_N_BINS &&
          g->n_erb == DFN2_N_ERB && g->df_bins == DFN2_DF_BINS,
          "DFN2 test grid has the signal grid the buffers are sized for");
    CHECK(g->mask_lookahead == DFN2_MASK_LOOKAHEAD &&
          g->df_lookahead == DFN2_DF_LOOKAHEAD &&
          g->df_history == DFN2_DF_HISTORY && g->ring == DFN2_DF_RING &&
          g->df_order == DFN2_DF_ORDER && g->frames_per_call == 1,
          "DFN2 test grid has the lookahead pair, order and ring the test expects");
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
          dfn2_state_init(&state, state_block, state_bytes, &bad) == -1 &&
          dfn2_state_init(&state, state_block, state_bytes, NULL) == -1,
          "DFN2 objects refuse a hand-edited or missing grid");
    CHECK(dfn2_model_io_init(&model_io, io_block, io_bytes, &bad, NULL, 0, NULL, 0) == -1 &&
          dfn2_model_io_ready_frames(&model_io) == 0u &&
          dfn2_model_io_push_features(&model_io, erb, spec) == -1,
          "DFN2 a model I/O state without a grid fails every call");
    CHECK(dfn2_model_io_init(NULL, io_block, io_bytes, g, NULL, 0, NULL, 0) == -1,
          "DFN2 model I/O init refuses NULL");

    /* The band count and the lookahead range are refused by dfn2_grid_make. */
    spec_other = dfn2_test_grid_spec();
    spec_other.n_erb = DFN2_N_ERB + 1;
    CHECK(dfn2_grid_make(&other, &spec_other) == DFN2_GRID_ERR_N_ERB,
          "DFN2 grid refuses a band count the graph cannot take");
    spec_other = dfn2_test_grid_spec();
    spec_other.mask_lookahead = DFN2_GRID_MASK_LOOKAHEAD_MAX + 1;
    CHECK(dfn2_grid_make(&other, &spec_other) == DFN2_GRID_ERR_MASK_LOOKAHEAD,
          "DFN2 grid refuses a mask lookahead the graph cannot take");
    spec_other = dfn2_test_grid_spec();
    spec_other.df_lookahead = DFN2_DF_ORDER;
    CHECK(dfn2_grid_make(&other, &spec_other) == DFN2_GRID_ERR_DF_LOOKAHEAD,
          "DFN2 grid refuses a deep-filter lookahead the order cannot take");

    /* A span wide enough to cancel in float keeps both configured ends of the spec normaliser,
     * from init and from reset, whichever end is the larger. */
    for (int e = 0; e < 2; ++e) {
        const float lo = e ? 1e-8f : 1.0f, hi = e ? 1.0f : 1e-8f;
        int ends_kept = 1;
        spec_other = dfn2_test_grid_spec();
        spec_other.spec_norm_init_lo = lo;
        spec_other.spec_norm_init_hi = hi;
        CHECK(dfn2_grid_make(&other, &spec_other) == DFN2_GRID_OK &&
              dfn2_state_mem_size(&other) == state_bytes &&
              dfn2_state_init(&state, state_block, state_bytes, &other) == 0,
              "DFN2 state starts on a wide spec normaliser span");
        for (int pass = 0; pass < 2; ++pass) {
            ends_kept = ends_kept && state.spec_norm_state[0] == lo &&
                        state.spec_norm_state[other.df_bins - 1] == hi;
            for (int k = 0; k < other.df_bins; ++k)
                ends_kept = ends_kept && state.spec_norm_state[k] > 0.0f &&
                            state.spec_norm_state[k] >= (lo < hi ? lo : hi) &&
                            state.spec_norm_state[k] <= (lo < hi ? hi : lo);
            memset(state.spec_norm_state, 0, sizeof(float) * (size_t)other.df_bins);
            dfn2_state_reset(&state);
        }
        CHECK(ends_kept, "DFN2 spec normaliser keeps both configured ends, at init and after reset");
    }

    /* The descriptor is the graph contract for this grid and no other. */
    CHECK(dfn2_model_io_descriptor_default(g, &descriptor) == 0 &&
          dfn2_model_io_descriptor_validate(g, &descriptor) == 0,
          "DFN2 the default descriptor validates against its own grid");
    spec_other = dfn2_test_grid_spec();
    spec_other.mask_lookahead = other_mask;
    spec_other.df_lookahead = 0;
    CHECK(dfn2_grid_make(&other, &spec_other) == DFN2_GRID_OK &&
          dfn2_model_io_descriptor_validate(&other, &descriptor) == -1,
          "DFN2 a descriptor of another mask lookahead is refused");
    spec_other = dfn2_test_grid_spec();
    spec_other.df_lookahead = other_df;
    CHECK(dfn2_grid_make(&other, &spec_other) == DFN2_GRID_OK &&
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
    free(state_block);
    free(io_block);
    return 1;
}

static int test_dfn_stream_alignment(void)
{
    static DFN2State dfn2;
    const size_t state_bytes = dfn2_state_mem_size(dfn2_test_grid());
    void* state_block = aligned_block(state_bytes);
    float spec2_re[DFN2_N_BINS], spec2_im[DFN2_N_BINS];
    float out2_re[DFN2_N_BINS], out2_im[DFN2_N_BINS];
    float mask2[DFN2_N_ERB] = {0};
    float coef2[DFN2_DF_BINS][DFN2_DF_ORDER][2] = {{{0}}};
    long long output_frame = -1;

    dfn_test_build_erb();
    CHECK(state_block != NULL &&
          dfn2_state_init(&dfn2, state_block, state_bytes, dfn2_test_grid()) == 0,
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
        CHECK(dfn2_enhance_spectrum_stream(
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
            int valid = dfn2_enhance_spectrum_stream(
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
    free(state_block);
    return 1;
}

static int all_zero(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (values[i] != 0.0f) return 0;
    }
    return 1;
}

/* Everything dfn2_model_io_validate_state() and friends may touch, captured so a
 * "nothing changed" check compares the carved block and both registered buffers
 * byte for byte, and a restore puts them back. */
typedef struct IoSnapshot {
    DFN2ModelIOState header;
    void* block;
    float* gru;
    float* convp;
} IoSnapshot;

static int io_capture(IoSnapshot* snap, const DFN2ModelIOState* state, const float* gru,
                      const float* convp)
{
    snap->header = *state;
    snap->block = duplicate(state->block, state->block_bytes);
    snap->gru = duplicate(gru, DFN2_TEST_GRU_ELEMENTS * sizeof(float));
    snap->convp = duplicate(convp, DFN2_TEST_CONVP_ELEMENTS * sizeof(float));
    return snap->block != NULL && snap->gru != NULL && snap->convp != NULL;
}

static void io_restore(const IoSnapshot* snap, DFN2ModelIOState* state, float* gru,
                       float* convp)
{
    memcpy(state->block, snap->block, snap->header.block_bytes);
    memcpy(gru, snap->gru, DFN2_TEST_GRU_ELEMENTS * sizeof(float));
    memcpy(convp, snap->convp, DFN2_TEST_CONVP_ELEMENTS * sizeof(float));
    *state = snap->header;
}

static int io_block_unchanged(const IoSnapshot* snap, const DFN2ModelIOState* state)
{
    return memcmp(state->block, snap->block, snap->header.block_bytes) == 0;
}

static int io_buffers_unchanged(const IoSnapshot* snap, const float* gru,
                                const float* convp)
{
    return memcmp(gru, snap->gru, DFN2_TEST_GRU_ELEMENTS * sizeof(float)) == 0 &&
           memcmp(convp, snap->convp, DFN2_TEST_CONVP_ELEMENTS * sizeof(float)) == 0;
}

static int test_dfn2_model_io(void)
{
    /* The windows are flat runs packed at the grid's stride inside the carved block:
     * erb_window[frame][n_erb] and spec_window[channel][frame][df_bins]; the recurrent
     * state is two registered buffers of the host. */
    const DFN2Grid* grid = dfn2_test_grid();
    const int wf = grid->input_frames;
    const size_t gru_n = DFN2_TEST_GRU_ELEMENTS;
    const size_t convp_n = DFN2_TEST_CONVP_ELEMENTS;
    const size_t block_bytes = dfn2_model_io_mem_size(grid);
    static DFN2ModelIOState state;
    static float gru[DFN2_TEST_GRU_ELEMENTS];
    static float convp[DFN2_TEST_CONVP_ELEMENTS];
    IoSnapshot healthy;
    void* block = aligned_block(block_bytes);
    float erb[DFN2_N_ERB];
    float spec[2][DFN2_DF_BINS];

    CHECK(block != NULL, "DFN2 model I/O block allocates");
    memset(gru, 0x5a, sizeof(gru));
    memset(convp, 0x5a, sizeof(convp));
    CHECK(dfn2_model_io_init(&state, block, block_bytes, grid, gru, gru_n, convp,
                             convp_n) == 0,
          "DFN2 model I/O state starts for the test grid");
    CHECK(all_zero(gru, gru_n) && all_zero(convp, convp_n),
          "DFN2 init zeroes the registered recurrent buffers");
    CHECK(dfn2_model_io_state_generation(&state) == 1u &&
          dfn2_model_io_state_generation(NULL) == 0u,
          "DFN2 init starts the state generation at 1");
    CHECK(dfn2_model_io_ready_frames(&state) ==
              (unsigned)(DFN2_MASK_LOOKAHEAD + grid->frames_per_call) &&
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
    CHECK(state.spec_window[(1 * wf + 0) * DFN2_DF_BINS + 9] == 109.0f &&
          state.spec_window[(1 * wf + 2) * DFN2_DF_BINS + 9] == 2109.0f,
          "DFN2 complex model window preserves channel-major layout");

    /* The first complete window is [t-(2-L) .. t+L] for t = 0, L the mask
     * lookahead: frames before the stream start read as zero, whatever L is. */
    dfn2_model_io_reset(&state);
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
              state.spec_window[(0 * wf + slot) * DFN2_DF_BINS + 5] == want_spec &&
              state.spec_window[(1 * wf + slot) * DFN2_DF_BINS + 5] == want_spec,
              "DFN2 first window is [t-(2-L)..t+L] with zeros before the start");
    }

    CHECK(state.gru_hidden == gru && state.gru_hidden_elements == gru_n &&
          state.df_convp_window == convp && state.df_convp_window_elements == convp_n,
          "DFN2 the registered buffers are the graph tensors the host owns");

    /* The accelerator writes the two recurrent arrays in place. */
    memset(gru, 0x3c, sizeof(gru));
    memset(convp, 0x6f, sizeof(convp));
    CHECK(io_capture(&healthy, &state, gru, convp), "DFN2 snapshot allocates");
    CHECK(dfn2_model_io_validate_state(&state) == 0 &&
          io_block_unchanged(&healthy, &state) && io_buffers_unchanged(&healthy, gru, convp) &&
          memcmp(&state, &healthy.header, sizeof(state)) == 0,
          "DFN2 validating a finite state leaves every byte in place");

    gru[(size_t)(DFN2_TEST_GRU_ROW_ERB + 1) * DFN2_TEST_GRU_HIDDEN +
        DFN2_TEST_GRU_HIDDEN - 1] = NAN;
    CHECK(dfn2_model_io_validate_state(&state) != 0,
          "DFN2 validate checks the second ERB stack layer");
    CHECK(all_zero(gru, gru_n) && all_zero(convp, convp_n),
          "DFN2 non-finite state zeroes both recurrent arrays");
    CHECK(io_block_unchanged(&healthy, &state) &&
          state.feature_frames_seen == healthy.header.feature_frames_seen,
          "DFN2 state refusal leaves the feature windows and counter alone");
    CHECK(dfn2_model_io_state_generation(&state) ==
              healthy.header.state_generation + 1u,
          "DFN2 a refused validation advances the state generation once");

    io_restore(&healthy, &state, gru, convp);
    convp[convp_n - 1] = INFINITY;   /* last element of the packed run */
    CHECK(dfn2_model_io_validate_state(&state) != 0 && all_zero(gru, gru_n),
          "DFN2 refuses a non-finite pathway window and zeroes the batch");

    CHECK(dfn2_model_io_validate_state(NULL) != 0,
          "DFN2 validate refuses a null state");
    io_restore(&healthy, &state, gru, convp);
    CHECK(dfn2_model_io_validate_arrays(NULL, gru_n, convp, convp_n) != 0 &&
          dfn2_model_io_validate_arrays(gru, 0u, convp, convp_n) != 0 &&
          dfn2_model_io_validate_arrays(gru, gru_n, NULL, convp_n) != 0 &&
          io_block_unchanged(&healthy, &state) && io_buffers_unchanged(&healthy, gru, convp) &&
          memcmp(&state, &healthy.header, sizeof(state)) == 0,
          "DFN2 validate_arrays refuses a null array and touches nothing");
    CHECK(dfn2_model_io_validate_arrays(gru, gru_n, convp, convp_n) == 0,
          "DFN2 validate_arrays accepts finite arrays");

    /* reset and zero_recurrent each advance the generation exactly once. */
    io_restore(&healthy, &state, gru, convp);
    dfn2_model_io_zero_recurrent(&state);
    CHECK(all_zero(gru, gru_n) && io_block_unchanged(&healthy, &state) &&
          state.state_generation == healthy.header.state_generation + 1u,
          "DFN2 zero_recurrent clears the recurrent arrays and counts");
    io_restore(&healthy, &state, gru, convp);
    dfn2_model_io_reset(&state);
    CHECK(all_zero(state.erb_window, (size_t)wf * DFN2_N_ERB) &&
          all_zero(gru, gru_n) && all_zero(convp, convp_n) &&
          state.feature_frames_seen == 0u &&
          state.state_generation == healthy.header.state_generation + 1u,
          "DFN2 reset returns to init values and counts");
    CHECK(state.n_erb == DFN2_N_ERB && state.df_bins == DFN2_DF_BINS &&
          state.mask_lookahead == DFN2_MASK_LOOKAHEAD &&
          dfn2_model_io_ready_frames(&state) ==
              (unsigned)(DFN2_MASK_LOOKAHEAD + grid->frames_per_call),
          "DFN2 reset keeps the grid dimensions of the state");
    dfn2_model_io_reset(NULL);
    dfn2_model_io_zero_recurrent(NULL);

    /* The generation counts modulo 2^32, whichever call zeroes the state. */
    io_restore(&healthy, &state, gru, convp);
    state.state_generation = 0xffffffffu;
    dfn2_model_io_reset(&state);
    CHECK(dfn2_model_io_state_generation(&state) == 0u,
          "DFN2 reset wraps the generation modulo 2^32");
    state.state_generation = 0xffffffffu;
    dfn2_model_io_zero_recurrent(&state);
    CHECK(dfn2_model_io_state_generation(&state) == 0u,
          "DFN2 zero_recurrent wraps the generation modulo 2^32");
    io_restore(&healthy, &state, gru, convp);
    state.state_generation = 0xffffffffu;
    gru[0] = NAN;
    CHECK(dfn2_model_io_validate_state(&state) != 0 &&
          dfn2_model_io_state_generation(&state) == 0u,
          "DFN2 a refused validation wraps the generation modulo 2^32");
    free(healthy.block);
    free(healthy.gru);
    free(healthy.convp);
    free(block);
    return 1;
}

static int test_dfn2(uint64_t* digest)
{
    static DFN2State state;
    const DFN2Grid* grid = dfn2_test_grid();
    const size_t state_bytes = dfn2_state_mem_size(grid);
    void* state_block = aligned_block(state_bytes);
    AinrTestHop hop_io;
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
    CHECK(ainr_test_hop_open(&hop_io, DFN2_N_FFT, grid->stft_scale, grid->stft_unscale) == 0,
          "DFN2 framing objects");
    CHECK(state_block != NULL &&
          dfn2_state_init(&state, state_block, state_bytes, grid) == 0,
          "DFN2 state starts for the test grid");
    dfn2_set_erb_matrices(&state, &dfn_test_fwd[0][0], &dfn_test_inv[0][0]);
    for (int b = 0; b < DFN2_N_ERB; ++b) mask[b] = 1.0f;
    for (int k = 0; k < DFN2_DF_BINS; ++k)
        coefs[k][DFN2_DF_HISTORY][0] = 1.0f;

    for (int frame = 0; frame < FRAMES; ++frame) {
        int valid;
        for (int i = 0; i < DFN2_HOP_LEN; ++i)
            input[i] = signal_sample((int64_t)frame * DFN2_HOP_LEN + i,
                                     DFN2_SR);
        ainr_test_hop_analysis(&hop_io, input, spec_re, spec_im);
        memcpy(history_re[frame], spec_re, sizeof(spec_re));
        memcpy(history_im[frame], spec_im, sizeof(spec_im));
        dfn2_compute_features(&state, spec_re, spec_im, erb, feature_spec);
        CHECK(all_finite(erb, DFN2_N_ERB), "DFN2 finite ERB features");
        CHECK(all_finite(feature_spec, 2 * DFN2_DF_BINS),
              "DFN2 finite complex features");
        valid = dfn2_enhance_spectrum(&state, spec_re, spec_im, mask,
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
            ainr_test_hop_synthesis(&hop_io, enhanced_re, enhanced_im, output);
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
    ainr_test_hop_close(&hop_io);
    free(state_block);
    return 1;
}

static int test_gtcrn(uint64_t* digest)
{
    AinrTestHop hop_io;
    float input[GTCRN_TEST_HOP_LEN], previous[GTCRN_TEST_HOP_LEN] = {0};
    float output[GTCRN_TEST_HOP_LEN];
    float spectrum[GTCRN_TEST_N_BINS][2];
    float max_error = 0.0f;

    CHECK(ainr_test_hop_open(&hop_io, GTCRN_TEST_N_FFT, 1.0f, 1.0f) == 0,
          "GTCRN framing objects");
    for (int frame = 0; frame < 24; ++frame) {
        for (int i = 0; i < GTCRN_TEST_HOP_LEN; ++i)
            input[i] = signal_sample((int64_t)frame * GTCRN_TEST_HOP_LEN + i,
                                     GTCRN_TEST_SR);
        ainr_test_hop_analysis_il(&hop_io, input, &spectrum[0][0]);
        ainr_test_hop_synthesis_il(&hop_io, &spectrum[0][0], output);
        CHECK(all_finite(&spectrum[0][0], 2 * GTCRN_TEST_N_BINS),
              "GTCRN finite spectrum");
        CHECK(all_finite(output, GTCRN_TEST_HOP_LEN), "GTCRN finite WOLA output");
        if (frame >= 2) {
            for (int i = 0; i < GTCRN_TEST_HOP_LEN; ++i) {
                float error = fabsf(output[i] - previous[i]);
                if (error > max_error) max_error = error;
            }
        }
        *digest = hash_bytes(*digest, spectrum, sizeof(spectrum));
        *digest = hash_bytes(*digest, output, sizeof(output));
        memcpy(previous, input, sizeof(input));
    }
    CHECK(max_error < 2e-4f, "GTCRN analysis/synthesis steady-state unity");
    ainr_test_hop_close(&hop_io);
    return 1;
}

/* A GTCRN instance on the test grid: the matrices, the eight tensors registered as
 * slices of `state` and a pool from get_mem_size(), poisoned before init. */
typedef struct GtcrnHost {
    GTCRNProcess* p;
    void* pool;
    size_t pool_bytes;
} GtcrnHost;

static int gtcrn_host_open(GtcrnHost* host, const float* fwd, const float* inv,
                           float* state)
{
    GTCRNProcessConfig cfg;
    GTCRNProcessMemReq req;
    host->p = NULL;
    host->pool = NULL;
    if (gtcrn_process_config_defaults(&cfg, gtcrn_test_grid()) != 0) return 0;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    gtcrn_test_register(&cfg, state);
    if (gtcrn_process_get_mem_size(&cfg, &req) != 0) return 0;
    host->pool_bytes = (size_t)req.bytes;
    host->pool = aligned_block(host->pool_bytes);
    if (host->pool == NULL) return 0;
    host->p = gtcrn_process_init(host->pool, host->pool_bytes, &cfg);
    return host->p != NULL;
}

static int test_gtcrn_model_state(void)
{
    enum { N = GTCRN_TEST_STATE_FLOATS, BANDS = GTCRN_TEST_ERB_BANDS,
           KEPT = GTCRN_TEST_ERB_LOW_BINS, HIGH_BANDS = GTCRN_TEST_ERB_HIGH_BANDS,
           HIGH_BINS = GTCRN_TEST_ERB_HIGH_BINS };
    static float state[N];
    static float expect[N];
    static float fwd[HIGH_BINS * HIGH_BANDS];
    static float inv[HIGH_BANDS * HIGH_BINS];
    static float spec_in[GTCRN_TEST_N_BINS][2];
    static float enhanced[GTCRN_TEST_N_BINS][2];
    float mag[BANDS], re_b[BANDS], im_b[BANDS], mask_erb[BANDS][2];
    GtcrnHost host;
    void* pool_before;
    size_t total = 0;
    unsigned int generation;
    int i;
    int c;

    for (i = 0; i < GTCRN_STATE_COUNT; ++i) total += gtcrn_test_state_elements[i];
    CHECK(total == N,
          "GTCRN the eight tensors of the host table are the flat state array");
    CHECK(gtcrn_test_state_elements[GTCRN_STATE_H_TRA] ==
              (size_t)GTCRN_TEST_TRA_GRUS * GTCRN_TEST_TRA_HIDDEN &&
          gtcrn_test_state_elements[GTCRN_STATE_H_DPGRNN] ==
              (size_t)GTCRN_TEST_DPGRNN_GRUS * GTCRN_TEST_DPGRNN_FREQ *
                  GTCRN_TEST_DPGRNN_HIDDEN,
          "GTCRN h_tra and h_dpgrnn are the rank-4 graph tensors");

    /* Synthetic caller-loaded matrices: the values live in the .bin the
     * loader owns, so this test pins the WIRING (which row scales which
     * band/bin), while the exporter's round-trip test pins the values. */
    for (i = 0; i < HIGH_BANDS; ++i) fwd[35 * HIGH_BANDS + i] = 0.25f + 0.001f * (float)i;
    for (i = 0; i < HIGH_BANDS; ++i) inv[i * HIGH_BINS + 35] = 0.125f + 0.002f * (float)i;
    memset(state, 0x5a, sizeof(state));
    CHECK(gtcrn_host_open(&host, fwd, inv, state), "GTCRN process starts");
    {
        int zero = 1;
        for (i = 0; i < N; ++i) zero &= state[i] == 0.0f;
        CHECK(zero, "GTCRN init zeroes the registered state");
    }
    memset(spec_in, 0, sizeof(spec_in));
    spec_in[10][0] = 3.0f; spec_in[10][1] = 4.0f;   /* low: passthrough */
    spec_in[KEPT + 35][0] = 2.0f;                   /* one high bin     */
    gtcrn_model_input(host.p, &spec_in[0][0], mag, re_b, im_b);
    CHECK(fabsf(mag[10] - 5.0f) < 1e-6f &&
          re_b[10] == 3.0f && im_b[10] == 4.0f &&
          mag[0] == sqrtf(1e-12f),
          "GTCRN host feature: low bins pass [mag, re, im] through");
    {
        int wired = 1;
        for (i = 0; i < HIGH_BANDS; ++i) {
            if (fabsf(re_b[KEPT + i] - fwd[35 * HIGH_BANDS + i] * 2.0f) > 1e-6f)
                wired = 0;
        }
        CHECK(wired,
              "GTCRN host feature: high bins band through the "
              "caller-loaded forward matrix");
    }
    memset(mask_erb, 0, sizeof(mask_erb));
    mask_erb[10][0] = 1.0f;                          /* unity low mask  */
    mask_erb[KEPT + 7][0] = 0.5f;                    /* one high band   */
    gtcrn_model_output(host.p, &mask_erb[0][0], &spec_in[0][0], &enhanced[0][0]);
    CHECK(enhanced[10][0] == 3.0f && enhanced[10][1] == 4.0f,
          "GTCRN host output: unity low-band CRM reproduces the "
          "spectrum bin");
    CHECK(fabsf(enhanced[KEPT + 35][0] - 2.0f * 0.5f * inv[7 * HIGH_BINS + 35]) < 1e-6f,
          "GTCRN host output: high bands expand through the "
          "caller-loaded inverse matrix");
    CHECK(gtcrn_process_state_generation(host.p) == 1u &&
          gtcrn_process_state_generation(NULL) == 0u,
          "GTCRN init starts the state generation at 1");

    /* The registered buffers are the state buffers the graph writes its *_out
     * tensors into, so a healthy state validates in place and leaves them, the
     * pool and the generation untouched. */
    memset(state, 0x5a, sizeof(state));
    memset(expect, 0x5a, sizeof(expect));
    pool_before = duplicate(host.pool, host.pool_bytes);
    CHECK(pool_before != NULL, "GTCRN snapshot allocates");
    CHECK(gtcrn_process_state_validate(host.p) == 0,
          "GTCRN validate accepts a finite state");
    CHECK(memcmp(state, expect, sizeof(state)) == 0 &&
          memcmp(host.pool, pool_before, host.pool_bytes) == 0,
          "GTCRN validate leaves a healthy state untouched");

    CHECK(gtcrn_process_state_validate(NULL) == -1,
          "GTCRN validate refuses a null state");

    /* One NaN or Inf at the first or last element of any single tensor
     * zeroes ALL eight buffers and advances the generation by one. The state is
     * nonzero everywhere else (0x41 pattern), so a partial reset or an
     * unchecked tensor is visible. */
    for (c = 0; c < 4; ++c) {
        const float bad = (c & 1) ? INFINITY : NAN;
        int cls_ok = 1;
        size_t offset = 0;
        for (i = 0; i < GTCRN_STATE_COUNT; ++i) {
            memset(state, 0x41, sizeof(state));
            generation = gtcrn_process_state_generation(host.p);
            state[offset + ((c < 2) ? 0 : gtcrn_test_state_elements[i] - 1)] = bad;
            cls_ok &= gtcrn_process_state_validate(host.p) == -1 &&
                      gtcrn_process_state_generation(host.p) == generation + 1u &&
                      all_zero(state, N);
            offset += gtcrn_test_state_elements[i];
        }
        CHECK(cls_ok, "GTCRN non-finite element in any tensor returns -1, "
                      "zeroes the whole state and counts once");
    }

    /* A finite state must still validate after a refusal, or the guard could
     * be a permanent latch rather than a per-call check, and a healthy
     * validation must not touch the generation. */
    memset(state, 0x41, sizeof(state));
    memset(expect, 0x41, sizeof(expect));
    generation = gtcrn_process_state_generation(host.p);
    CHECK(gtcrn_process_state_validate(host.p) == 0 &&
          memcmp(state, expect, sizeof(state)) == 0 &&
          gtcrn_process_state_generation(host.p) == generation,
          "GTCRN validate still accepts a finite state after a refusal");

    memset(state, 0x41, sizeof(state));
    generation = gtcrn_process_state_generation(host.p);
    gtcrn_process_reset(host.p);
    CHECK(all_zero(state, N) &&
          gtcrn_process_state_generation(host.p) == generation + 1u,
          "GTCRN reset zeroes the tensors and advances the generation");
    gtcrn_process_reset(NULL);
    free(pool_before);
    free(host.pool);
    return 1;
}

/* ---- GTCRN refusals: grid, config, pool, registration, mask ---- */

/* Zero matrices big enough for every grid below. */
static const float gtcrn_zero_matrix[GTCRN_TEST_N_BINS * (GTCRN_TEST_ERB_HIGH_BANDS + 1)];

/* Config for the grid of `spec`, zero matrices; the eight tensors are registered as slices
 * of `state` when it is non-NULL. */
static int gtcrn_make_config(GTCRNProcessConfig* cfg, const GTCRNGridSpec* spec, float* state)
{
    GTCRNGrid grid;
    if (gtcrn_grid_make(&grid, spec) != GTCRN_GRID_OK) return 0;
    if (gtcrn_process_config_defaults(cfg, &grid) != 0) return 0;
    cfg->erb_fwd = gtcrn_zero_matrix;
    cfg->erb_inv = gtcrn_zero_matrix;
    if (state) gtcrn_test_register(cfg, state);
    return 1;
}

/* One frame through a live instance: bin 10 (a passed-through low bin) reaches the features
 * and a unity mask on it returns the input bin. */
static int gtcrn_frame_runs(GTCRNProcess* p)
{
    static float spec[GTCRN_TEST_N_BINS][2], enhanced[GTCRN_TEST_N_BINS][2];
    float mag[GTCRN_TEST_ERB_BANDS], re[GTCRN_TEST_ERB_BANDS], im[GTCRN_TEST_ERB_BANDS];
    float mask[GTCRN_TEST_ERB_BANDS][2];
    memset(spec, 0, sizeof(spec));
    memset(enhanced, 0, sizeof(enhanced));
    memset(mask, 0, sizeof(mask));
    spec[10][0] = 3.0f;
    spec[10][1] = 4.0f;
    mask[10][0] = 1.0f;
    gtcrn_model_input(p, &spec[0][0], mag, re, im);
    gtcrn_model_output(p, &mask[0][0], &spec[0][0], &enhanced[0][0]);
    return fabsf(mag[10] - 5.0f) < 1e-6f && re[10] == 3.0f && im[10] == 4.0f &&
           enhanced[10][0] == 3.0f && enhanced[10][1] == 4.0f;
}

/* Exactly one aligned unsigned-int word of the pool differs from its snapshot, and it holds
 * `value`. */
static int only_word_moved(const void* pool, const void* snapshot, size_t bytes, unsigned value)
{
    const unsigned char* x = (const unsigned char*)pool;
    const unsigned char* y = (const unsigned char*)snapshot;
    size_t word = (size_t)-1;
    unsigned held;
    for (size_t i = 0; i < bytes; ++i) {
        if (x[i] == y[i]) continue;
        if (word == (size_t)-1) word = i / sizeof(unsigned int);
        else if (i / sizeof(unsigned int) != word) return 0;
    }
    if (word == (size_t)-1) return 0;
    memcpy(&held, x + word * sizeof(unsigned int), sizeof held);
    return held == value;
}

/* The pool and the registered state are byte-identical to their snapshots. */
static int gtcrn_untouched(const void* pool, const void* pool_copy, size_t pool_bytes,
                           const float* state, const float* state_copy)
{
    return memcmp(pool, pool_copy, pool_bytes) == 0 &&
           memcmp(state, state_copy, GTCRN_TEST_STATE_FLOATS * sizeof(float)) == 0;
}

static int test_gtcrn_grid_make_refusals(void)
{
    const struct { GTCRNGridSpec spec; int status; } refused[] = {
        { .spec = { .n_fft = 0,     .erb_low_bins = 65,  .erb_high_bands = 64 },
          .status = GTCRN_GRID_ERR_N_FFT },
        { .spec = { .n_fft = 24,    .erb_low_bins = 65,  .erb_high_bands = 64 },
          .status = GTCRN_GRID_ERR_N_FFT },
        { .spec = { .n_fft = 16384, .erb_low_bins = 65,  .erb_high_bands = 64 },
          .status = GTCRN_GRID_ERR_N_FFT },
        { .spec = { .n_fft = 512,   .erb_low_bins = -1,  .erb_high_bands = 64 },
          .status = GTCRN_GRID_ERR_ERB_LOW_BINS },
        { .spec = { .n_fft = 512,   .erb_low_bins = 257, .erb_high_bands = 1 },
          .status = GTCRN_GRID_ERR_ERB_LOW_BINS },
        { .spec = { .n_fft = 512,   .erb_low_bins = 65,  .erb_high_bands = 0 },
          .status = GTCRN_GRID_ERR_ERB_HIGH_BANDS },
        { .spec = { .n_fft = 512,   .erb_low_bins = 65,  .erb_high_bands = -3 },
          .status = GTCRN_GRID_ERR_ERB_HIGH_BANDS },
        { .spec = { .n_fft = 512,   .erb_low_bins = 65,  .erb_high_bands = 193 },
          .status = GTCRN_GRID_ERR_ERB_HIGH_BANDS },
        { .spec = { .n_fft = 24,    .erb_low_bins = -1,  .erb_high_bands = 0 },
          .status = GTCRN_GRID_ERR_N_FFT },
        { .spec = { .n_fft = 512,   .erb_low_bins = 257, .erb_high_bands = 0 },
          .status = GTCRN_GRID_ERR_ERB_LOW_BINS },
    };
    const GTCRNGridSpec edge[] = {
        { .n_fft = 512, .erb_low_bins = 0,   .erb_high_bands = 1 },
        { .n_fft = 512, .erb_low_bins = 0,   .erb_high_bands = 257 },
        { .n_fft = 512, .erb_low_bins = 256, .erb_high_bands = 1 },
        { .n_fft = 512, .erb_low_bins = 65,  .erb_high_bands = 192 },
    };
    const GTCRNGridSpec shipped = gtcrn_test_grid_spec();
    const char* text[7];
    GTCRNGrid grid, poison;

    memset(&poison, 0xA5, sizeof(poison));
    for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i) {
        memcpy(&grid, &poison, sizeof(grid));
        CHECK(gtcrn_grid_make(&grid, &refused[i].spec) == refused[i].status &&
              memcmp(&grid, &poison, sizeof(grid)) == 0,
              "GTCRN grid_make refuses each spec with its status and writes nothing");
    }
    memcpy(&grid, &poison, sizeof(grid));
    CHECK(gtcrn_grid_make(NULL, &shipped) == GTCRN_GRID_ERR_NULL &&
          gtcrn_grid_make(&grid, NULL) == GTCRN_GRID_ERR_NULL &&
          memcmp(&grid, &poison, sizeof(grid)) == 0 &&
          gtcrn_grid_validate(NULL) == GTCRN_GRID_ERR_NULL,
          "GTCRN grid_make and grid_validate refuse NULL arguments");

    /* The ends of every accepted range, with the derived fields. */
    for (size_t i = 0; i < sizeof(edge) / sizeof(edge[0]); ++i) {
        const int n_bins = edge[i].n_fft / 2 + 1;
        CHECK(gtcrn_grid_make(&grid, &edge[i]) == GTCRN_GRID_OK &&
              grid.n_fft == edge[i].n_fft && grid.hop == edge[i].n_fft / 2 &&
              grid.n_bins == n_bins && grid.erb_low_bins == edge[i].erb_low_bins &&
              grid.erb_high_bands == edge[i].erb_high_bands &&
              grid.erb_bands == edge[i].erb_low_bins + edge[i].erb_high_bands &&
              grid.erb_high_bins == n_bins - edge[i].erb_low_bins &&
              gtcrn_grid_validate(&grid) == GTCRN_GRID_OK,
              "GTCRN grid_make accepts the ends of each range and derives the fields");
    }

    for (int status = 0; status >= GTCRN_GRID_ERR_INCONSISTENT; --status) {
        text[-status] = gtcrn_grid_status_text(status);
        CHECK(text[-status] != NULL && text[-status][0] != '\0',
              "GTCRN every grid status has a text");
        for (int other = 0; other < -status; ++other)
            CHECK(strcmp(text[-status], text[other]) != 0,
                  "GTCRN grid status texts are distinct");
    }
    CHECK(gtcrn_grid_status_text(7) != NULL && gtcrn_grid_status_text(-99) != NULL,
          "GTCRN an unknown grid status still has a text");
    return 1;
}

/* A hand-edited grid is refused by validate and by every consumer of a grid. */
static int test_gtcrn_grid_validate_refusals(void)
{
    static float state[GTCRN_TEST_STATE_FLOATS];
    const GTCRNGridSpec shipped = gtcrn_test_grid_spec();
    GTCRNProcessConfig cfg, cfg_poison;
    GTCRNProcessMemReq req, req_poison;

    memset(&req_poison, 0xA5, sizeof(req_poison));
    memset(&cfg_poison, 0xA5, sizeof(cfg_poison));
    for (int edit = 0; edit < 7; ++edit) {
        GTCRNGrid bad = *gtcrn_test_grid();
        int want = GTCRN_GRID_ERR_INCONSISTENT;
        switch (edit) {
        case 0: bad.hop += 1; break;
        case 1: bad.n_bins += 1; break;
        case 2: bad.erb_bands += 1; break;
        case 3: bad.erb_high_bins -= 1; break;
        case 4: bad.n_fft = 24; want = GTCRN_GRID_ERR_N_FFT; break;
        case 5: bad.erb_low_bins = -1; want = GTCRN_GRID_ERR_ERB_LOW_BINS; break;
        default: bad.erb_high_bands = 0; want = GTCRN_GRID_ERR_ERB_HIGH_BANDS; break;
        }
        CHECK(gtcrn_grid_validate(&bad) == want, "GTCRN grid_validate names the failing field");

        memcpy(&cfg, &cfg_poison, sizeof(cfg));
        CHECK(gtcrn_process_config_defaults(&cfg, &bad) == -1 &&
              memcmp(&cfg, &cfg_poison, sizeof(cfg)) == 0,
              "GTCRN config_defaults refuses an edited grid and writes nothing");

        CHECK(gtcrn_make_config(&cfg, &shipped, state), "GTCRN config on the shipped grid");
        cfg.grid = bad;
        memcpy(&req, &req_poison, sizeof(req));
        CHECK(gtcrn_process_get_mem_size(&cfg, &req) == -1 &&
              memcmp(&req, &req_poison, sizeof(req)) == 0,
              "GTCRN get_mem_size refuses an edited grid and writes nothing");
    }
    memcpy(&cfg, &cfg_poison, sizeof(cfg));
    CHECK(gtcrn_process_config_defaults(NULL, gtcrn_test_grid()) == -1 &&
          gtcrn_process_config_defaults(&cfg, NULL) == -1 &&
          memcmp(&cfg, &cfg_poison, sizeof(cfg)) == 0,
          "GTCRN config_defaults refuses NULL arguments");
    return 1;
}

static int test_gtcrn_config_and_pool_refusals(void)
{
    enum { SLACK = 32 };
    static float state[GTCRN_TEST_STATE_FLOATS], state_copy[GTCRN_TEST_STATE_FLOATS];
    const GTCRNGridSpec shipped = gtcrn_test_grid_spec();
    GTCRNProcessConfig cfg, bad;
    GTCRNProcessMemReq req, probe, poison;
    GTCRNProcess* p;
    unsigned char* block;
    void* before;
    size_t bytes;

    CHECK(gtcrn_make_config(&cfg, &shipped, state) &&
          gtcrn_process_get_mem_size(&cfg, &req) == 0, "GTCRN config and size");
    bytes = (size_t)req.bytes;
    memset(&poison, 0xA5, sizeof(poison));

    /* get_mem_size: a missing argument or matrix writes nothing. */
    bad = cfg;
    bad.erb_fwd = NULL;
    memcpy(&probe, &poison, sizeof(probe));
    CHECK(gtcrn_process_get_mem_size(&bad, &probe) == -1 &&
          memcmp(&probe, &poison, sizeof(probe)) == 0, "GTCRN size refuses a missing erb_fwd");
    bad = cfg;
    bad.erb_inv = NULL;
    CHECK(gtcrn_process_get_mem_size(&bad, &probe) == -1 &&
          memcmp(&probe, &poison, sizeof(probe)) == 0, "GTCRN size refuses a missing erb_inv");
    CHECK(gtcrn_process_get_mem_size(NULL, &probe) == -1 &&
          gtcrn_process_get_mem_size(&cfg, NULL) == -1 &&
          memcmp(&probe, &poison, sizeof(probe)) == 0, "GTCRN size refuses NULL arguments");

    /* init: every refusal leaves the pool and the registered state as they were. */
    block = (unsigned char*)aligned_block(bytes + SLACK);
    CHECK(block != NULL, "GTCRN pool allocates");
    memset(state, 0x5a, sizeof(state));
    memcpy(state_copy, state, sizeof(state));
    before = duplicate(block, bytes + SLACK);
    CHECK(before != NULL, "GTCRN snapshot allocates");
    CHECK(gtcrn_process_init(NULL, bytes, &cfg) == NULL &&
          gtcrn_process_init(block, 0u, &cfg) == NULL &&
          gtcrn_process_init(block, bytes - 1u, &cfg) == NULL &&
          gtcrn_process_init_ex(block, bytes - 1u, &cfg, &req) == NULL &&
          gtcrn_process_init(block + 8, bytes, &cfg) == NULL &&
          gtcrn_process_init(block, bytes, NULL) == NULL &&
          gtcrn_process_init(block, bytes, &bad) == NULL &&
          gtcrn_untouched(block, before, bytes + SLACK, state, state_copy),
          "GTCRN init refuses a NULL, short or misaligned pool and a bad config, writes nothing");

    /* The exact size is enough; the instance is used, and the bytes past it are not written. */
    p = gtcrn_process_init(block, bytes, &cfg);
    CHECK(p != NULL && gtcrn_process_state_generation(p) == 1u &&
          memcmp(gtcrn_process_grid(p), &cfg.grid, sizeof(GTCRNGrid)) == 0 &&
          all_zero(state, GTCRN_TEST_STATE_FLOATS) &&
          memcmp(block + bytes, (unsigned char*)before + bytes, SLACK) == 0 &&
          gtcrn_frame_runs(p),
          "GTCRN init accepts an exact-size pool and the instance runs a frame");
    free(before);
    free(block);
    return 1;
}

static int test_gtcrn_stale_pool_gate(void)
{
    static float state[GTCRN_TEST_STATE_FLOATS], state_copy[GTCRN_TEST_STATE_FLOATS];
    const GTCRNGridSpec shipped = gtcrn_test_grid_spec();
    /* The first three have the shipped pool size; the third also has the shipped erb_bands, so
     * only the split of the bands between low bins and high bands tells it apart. */
    const GTCRNGridSpec other[] = {
        { .n_fft = 512, .erb_low_bins = 64, .erb_high_bands = 64 },
        { .n_fft = 512, .erb_low_bins = 65, .erb_high_bands = 63 },
        { .n_fft = 512, .erb_low_bins = 64, .erb_high_bands = 65 },
        { .n_fft = 256, .erb_low_bins = 65, .erb_high_bands = 64 },
    };
    GTCRNProcessConfig cfg, cfg_other;
    GTCRNProcessMemReq req, req_other, tampered;
    GTCRNProcess* p;
    void *block, *before;
    size_t bytes;

    CHECK(gtcrn_make_config(&cfg, &shipped, state) &&
          gtcrn_process_get_mem_size(&cfg, &req) == 0, "GTCRN config and size");
    bytes = (size_t)req.bytes;
    block = aligned_block(bytes);
    CHECK(block != NULL, "GTCRN pool allocates");
    p = gtcrn_process_init_ex(block, bytes, &cfg, NULL);
    CHECK(p != NULL && gtcrn_process_state_generation(p) == 1u && gtcrn_frame_runs(p),
          "GTCRN init_ex without a descriptor behaves as init");
    memset(block, 0xA5, bytes);
    p = gtcrn_process_init_ex(block, bytes, &cfg, &req);
    CHECK(p != NULL && gtcrn_frame_runs(p), "GTCRN init_ex accepts its own descriptor");

    /* A live instance: a second generation and a nonzero state to lose. */
    gtcrn_process_reset(p);
    gtcrn_process_reset(p);
    memset(state, 0x41, sizeof(state));
    memcpy(state_copy, state, sizeof(state));
    before = duplicate(block, bytes);
    CHECK(before != NULL && gtcrn_process_state_generation(p) == 3u, "GTCRN live instance");

    for (size_t i = 0; i < sizeof(other) / sizeof(other[0]); ++i) {
        CHECK(gtcrn_make_config(&cfg_other, &other[i], state) &&
              gtcrn_process_get_mem_size(&cfg_other, &req_other) == 0 &&
              req_other.build_flags_hash != req.build_flags_hash &&
              (i == 3 ? req_other.bytes != req.bytes : req_other.bytes == req.bytes),
              "GTCRN the other grid differs in hash (and in size only for another n_fft)");
        CHECK(gtcrn_process_init_ex(block, bytes, &cfg_other, &req) == NULL &&
              gtcrn_untouched(block, before, bytes, state, state_copy) &&
              gtcrn_process_state_generation(p) == 3u,
              "GTCRN init_ex refuses a pool described for another grid, writes nothing");
    }
    for (int field = 0; field < 8; ++field) {
        tampered = req;
        switch (field) {
        case 0: tampered.descriptor_version += 1u; break;
        case 1: tampered.layout_version -= 1u; break;
        case 2: tampered.reserved0 = 1u; break;
        case 3: tampered.build_flags_hash ^= 1u; break;
        case 4: tampered.alignment = 32u; break;
        case 5: tampered.reserved1 = 1u; break;
        case 6: tampered.bytes += 16u; break;
        default: tampered.bytes -= 16u; break;
        }
        CHECK(gtcrn_process_init_ex(block, bytes, &cfg, &tampered) == NULL &&
              gtcrn_untouched(block, before, bytes, state, state_copy) &&
              gtcrn_process_state_generation(p) == 3u,
              "GTCRN init_ex refuses each altered descriptor field, writes nothing");
    }
    CHECK(gtcrn_frame_runs(p), "GTCRN the live instance still runs after the refusals");

    /* The right descriptor re-initialises the same pool for the other grid. */
    CHECK(gtcrn_make_config(&cfg_other, &other[1], state) &&
          gtcrn_process_get_mem_size(&cfg_other, &req_other) == 0, "GTCRN other config");
    p = gtcrn_process_init_ex(block, bytes, &cfg_other, &req_other);
    CHECK(p != NULL && gtcrn_process_grid(p)->erb_bands == 128 &&
          gtcrn_process_state_generation(p) == 1u && all_zero(state, GTCRN_TEST_STATE_FLOATS) &&
          gtcrn_frame_runs(p),
          "GTCRN init_ex re-initialises the pool for the grid its descriptor names");
    free(before);
    free(block);
    return 1;
}

static int test_gtcrn_registration_all_or_none(void)
{
    static float state[GTCRN_TEST_STATE_FLOATS], state_copy[GTCRN_TEST_STATE_FLOATS];
    const GTCRNGridSpec shipped = gtcrn_test_grid_spec();
    GTCRNProcessConfig full, none, cfg;
    GTCRNProcessMemReq req_full, req_none, req, poison;
    GTCRNProcess* p;
    void *block, *before;
    size_t bytes;

    CHECK(gtcrn_make_config(&full, &shipped, state) && gtcrn_make_config(&none, &shipped, NULL) &&
          gtcrn_process_get_mem_size(&full, &req_full) == 0 &&
          gtcrn_process_get_mem_size(&none, &req_none) == 0,
          "GTCRN registered and unregistered configs");
    CHECK(memcmp(&req_full, &req_none, sizeof(req_full)) == 0,
          "GTCRN the registration is in neither the pool size nor the hash");
    bytes = (size_t)req_none.bytes;
    block = aligned_block(bytes);
    CHECK(block != NULL, "GTCRN pool allocates");
    p = gtcrn_process_init_ex(block, bytes, &full, &req_none);
    CHECK(p != NULL && gtcrn_process_state_generation(p) == 1u && gtcrn_frame_runs(p),
          "GTCRN all eight pairs are accepted and the instance runs");
    p = gtcrn_process_init_ex(block, bytes, &none, &req_full);
    CHECK(p != NULL && gtcrn_process_state_generation(p) == 1u && gtcrn_frame_runs(p),
          "GTCRN none of the pairs is accepted and the instance runs");

    memset(block, 0xA5, bytes);
    memset(state, 0x41, sizeof(state));
    memcpy(state_copy, state, sizeof(state));
    before = duplicate(block, bytes);
    memset(&poison, 0xA5, sizeof(poison));
    CHECK(before != NULL, "GTCRN snapshot allocates");

    for (int base = 0; base < 2; ++base) {
        for (int i = 0; i < GTCRN_STATE_COUNT; ++i) {
            for (int variant = 0; variant < 4; ++variant) {
                cfg = base ? none : full;
                if (base == 0) {          /* one pair of a full registration is wrong */
                    if (variant == 0) { cfg.state[i] = NULL; cfg.state_elements[i] = 0u; }
                    if (variant == 1) cfg.state_elements[i] = 0u;
                    if (variant == 2) cfg.state[i] = NULL;
                } else {                  /* one pair registered, the other seven not */
                    cfg.state[i] = variant == 2 ? NULL : state;
                    cfg.state_elements[i] = variant == 1 ? 0u : 1u;
                }
                if (variant == 3) {       /* the byte size does not fit */
                    cfg.state[i] = state;
                    cfg.state_elements[i] = SIZE_MAX / sizeof(float) + 1u;
                }
                memcpy(&req, &poison, sizeof(req));
                CHECK(gtcrn_process_get_mem_size(&cfg, &req) == -1 &&
                      memcmp(&req, &poison, sizeof(req)) == 0 &&
                      gtcrn_process_init(block, bytes, &cfg) == NULL &&
                      gtcrn_process_init_ex(block, bytes, &cfg, &req_none) == NULL &&
                      gtcrn_untouched(block, before, bytes, state, state_copy),
                      "GTCRN a partial or malformed registration is refused, writes nothing");
            }
        }
    }
    free(before);
    free(block);
    return 1;
}

static int test_gtcrn_mask_validate(void)
{
    enum { N = GTCRN_TEST_STATE_FLOATS, BANDS = GTCRN_TEST_ERB_BANDS };
    static float state[N], expect[N];
    static float spec[GTCRN_TEST_N_BINS][2], enhanced[GTCRN_TEST_N_BINS][2];
    const struct { int index; float value; } bad_at[] = {
        { .index = 0,             .value = NAN },
        { .index = BANDS + 1,     .value = -INFINITY },
        { .index = 2 * BANDS - 1, .value = INFINITY },
    };
    float mask[2 * BANDS + 1], bad_mask[2 * BANDS + 1];
    GtcrnHost host;
    void* pool_copy;
    unsigned int generation;

    CHECK(gtcrn_host_open(&host, gtcrn_zero_matrix, gtcrn_zero_matrix, state), "GTCRN starts");
    memset(mask, 0, sizeof(mask));
    mask[2 * 10] = 1.0f;
    mask[2 * BANDS] = NAN;                 /* one past the mask: never read */
    memset(spec, 0, sizeof(spec));
    spec[10][0] = 3.0f;
    spec[10][1] = 4.0f;
    memset(state, 0x41, sizeof(state));
    memcpy(expect, state, sizeof(state));
    pool_copy = duplicate(host.pool, host.pool_bytes);
    CHECK(pool_copy != NULL, "GTCRN snapshot allocates");
    generation = gtcrn_process_state_generation(host.p);

    CHECK(gtcrn_process_mask_validate(host.p, NULL) == -1 &&
          gtcrn_process_mask_validate(NULL, mask) == -1 &&
          gtcrn_untouched(host.pool, pool_copy, host.pool_bytes, state, expect) &&
          gtcrn_process_state_generation(host.p) == generation,
          "GTCRN mask_validate refuses NULL arguments and changes nothing");
    CHECK(gtcrn_process_mask_validate(host.p, mask) == 0 &&
          gtcrn_untouched(host.pool, pool_copy, host.pool_bytes, state, expect) &&
          gtcrn_process_state_generation(host.p) == generation,
          "GTCRN mask_validate accepts a finite mask and changes nothing");
    gtcrn_model_output(host.p, mask, &spec[0][0], &enhanced[0][0]);
    CHECK(enhanced[10][0] == 3.0f && enhanced[10][1] == 4.0f,
          "GTCRN the accepted mask is applied by model_output");

    for (size_t i = 0; i < sizeof(bad_at) / sizeof(bad_at[0]); ++i) {
        memcpy(bad_mask, mask, sizeof(mask));
        bad_mask[bad_at[i].index] = bad_at[i].value;
        memset(state, 0x41, sizeof(state));
        generation = gtcrn_process_state_generation(host.p);
        CHECK(gtcrn_process_mask_validate(host.p, bad_mask) == -1 &&
              all_zero(state, N) && gtcrn_process_state_generation(host.p) == generation + 1u,
              "GTCRN a non-finite mask element zeroes all eight tensors and counts once");
        CHECK(gtcrn_process_state_validate(host.p) == 0 &&
              gtcrn_process_state_generation(host.p) == generation + 1u,
              "GTCRN mask_validate then state_validate advance the generation once");
        CHECK(gtcrn_process_mask_validate(host.p, mask) == 0 &&
              gtcrn_process_state_generation(host.p) == generation + 1u,
              "GTCRN a finite mask validates again after a refusal");
    }
    free(pool_copy);
    free(host.pool);
    return 1;
}

/* Nothing registered: the generation still counts the refusals and no pool byte but the counter
 * moves. */
static int test_gtcrn_board_path_registers_nothing(void)
{
    enum { BANDS = GTCRN_TEST_ERB_BANDS };
    const GTCRNGridSpec shipped = gtcrn_test_grid_spec();
    float mask[2 * BANDS];
    GTCRNProcessConfig cfg;
    GTCRNProcessMemReq req;
    GTCRNProcess* p;
    void *pool, *pool_copy;
    size_t bytes;

    CHECK(gtcrn_make_config(&cfg, &shipped, NULL) && gtcrn_process_get_mem_size(&cfg, &req) == 0,
          "GTCRN board config");
    for (int i = 0; i < GTCRN_STATE_COUNT; ++i)
        CHECK(cfg.state[i] == NULL && cfg.state_elements[i] == 0u,
              "GTCRN the default config registers nothing");
    bytes = (size_t)req.bytes;
    pool = aligned_block(bytes);
    p = pool ? gtcrn_process_init(pool, bytes, &cfg) : NULL;
    CHECK(p != NULL && gtcrn_process_state_generation(p) == 1u, "GTCRN board init");
    pool_copy = duplicate(pool, bytes);
    CHECK(pool_copy != NULL, "GTCRN snapshot allocates");
    memset(mask, 0, sizeof(mask));

    CHECK(gtcrn_process_state_validate(p) == 0 && gtcrn_process_mask_validate(p, mask) == 0 &&
          gtcrn_process_state_generation(p) == 1u && memcmp(pool, pool_copy, bytes) == 0,
          "GTCRN validation of nothing registered passes and changes nothing");
    mask[0] = NAN;
    CHECK(gtcrn_process_mask_validate(p, mask) == -1 && gtcrn_process_state_generation(p) == 2u &&
          only_word_moved(pool, pool_copy, bytes, 2u),
          "GTCRN a refused mask advances the generation although nothing is registered");
    CHECK(gtcrn_process_state_validate(p) == 0 && gtcrn_process_state_generation(p) == 2u,
          "GTCRN state_validate after a refused mask advances nothing");
    gtcrn_process_reset(p);
    mask[0] = 0.0f;
    mask[2 * BANDS - 1] = INFINITY;
    CHECK(gtcrn_process_state_generation(p) == 3u &&
          gtcrn_process_mask_validate(p, mask) == -1 && gtcrn_process_state_generation(p) == 4u &&
          only_word_moved(pool, pool_copy, bytes, 4u),
          "GTCRN reset and a refused mask each advance the generation by one");
    CHECK(gtcrn_frame_runs(p), "GTCRN the board instance runs the next frame");
    free(pool_copy);
    free(pool);
    return 1;
}

/* A refused set_erb_matrices keeps both matrices in use; an accepted one is used by the next
 * frame. */
static int test_gtcrn_set_erb_matrices(void)
{
    enum { BANDS = GTCRN_TEST_ERB_BANDS, KEPT = GTCRN_TEST_ERB_LOW_BINS,
           HIGH_BANDS = GTCRN_TEST_ERB_HIGH_BANDS, HIGH_BINS = GTCRN_TEST_ERB_HIGH_BINS };
    static float fwd_a[HIGH_BINS * HIGH_BANDS], inv_a[HIGH_BANDS * HIGH_BINS];
    static float fwd_b[HIGH_BINS * HIGH_BANDS], inv_b[HIGH_BANDS * HIGH_BINS];
    static float state[GTCRN_TEST_STATE_FLOATS];
    static float spec[GTCRN_TEST_N_BINS][2], enhanced[GTCRN_TEST_N_BINS][2];
    float mag[BANDS], re[BANDS], im[BANDS], mask[BANDS][2];
    GtcrnHost host;

    fwd_a[35 * HIGH_BANDS + 3] = 0.5f;
    inv_a[3 * HIGH_BINS + 35] = 0.25f;
    fwd_b[35 * HIGH_BANDS + 3] = 2.0f;
    inv_b[3 * HIGH_BINS + 35] = 4.0f;
    memset(mask, 0, sizeof(mask));
    spec[KEPT + 35][0] = 2.0f;
    mask[KEPT + 3][0] = 1.0f;
    CHECK(gtcrn_host_open(&host, fwd_a, inv_a, state), "GTCRN starts");

    gtcrn_model_input(host.p, &spec[0][0], mag, re, im);
    gtcrn_model_output(host.p, &mask[0][0], &spec[0][0], &enhanced[0][0]);
    CHECK(re[KEPT + 3] == 1.0f && enhanced[KEPT + 35][0] == 0.5f,
          "GTCRN the configured matrices are used");
    CHECK(gtcrn_process_set_erb_matrices(NULL, fwd_b, inv_b) == -1 &&
          gtcrn_process_set_erb_matrices(host.p, NULL, inv_b) == -1 &&
          gtcrn_process_set_erb_matrices(host.p, fwd_b, NULL) == -1,
          "GTCRN set_erb_matrices refuses a NULL argument");
    gtcrn_model_input(host.p, &spec[0][0], mag, re, im);
    gtcrn_model_output(host.p, &mask[0][0], &spec[0][0], &enhanced[0][0]);
    CHECK(re[KEPT + 3] == 1.0f && enhanced[KEPT + 35][0] == 0.5f,
          "GTCRN a refused set_erb_matrices keeps both matrices");
    CHECK(gtcrn_process_set_erb_matrices(host.p, fwd_b, inv_b) == 0, "GTCRN set_erb_matrices");
    gtcrn_model_input(host.p, &spec[0][0], mag, re, im);
    gtcrn_model_output(host.p, &mask[0][0], &spec[0][0], &enhanced[0][0]);
    CHECK(re[KEPT + 3] == 4.0f && enhanced[KEPT + 35][0] == 8.0f,
          "GTCRN the next frame uses the replaced matrices");
    free(host.pool);
    return 1;
}

/* The estimation branch (dfn2_compute_features) and the application branch
 * (dfn2_enhance_spectrum_stream) of one DFN2State must be independent: the compose
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
    const size_t state_bytes = dfn2_state_mem_size(dfn2_test_grid());
    void* block_a = aligned_block(state_bytes);
    void* block_b = aligned_block(state_bytes);
    int features_differed = 0;

    dfn_test_build_erb();
    CHECK(block_a != NULL && block_b != NULL &&
          dfn2_state_init(&st_a, block_a, state_bytes, dfn2_test_grid()) == 0 &&
          dfn2_state_init(&st_b, block_b, state_bytes, dfn2_test_grid()) == 0,
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
        va = dfn2_enhance_spectrum_stream(&st_a, app_re, app_im, head >= 0,
                                 head >= 0 ? mask : NULL,
                                 head >= 0 ? &coef[0][0][0] : NULL,
                                 head >= 0 ? stream_alpha(head) : 0.0f,
                                 0.0f, out_a_re, out_a_im, NULL);
        vb = dfn2_enhance_spectrum_stream(&st_b, app_re, app_im, head >= 0,
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
    CHECK(dfn2_state_init(&st_a, block_a, state_bytes, dfn2_test_grid()) == 0 &&
          dfn2_state_init(&st_b, block_b, state_bytes, dfn2_test_grid()) == 0,
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
        (void)dfn2_enhance_spectrum_stream(&st_a, app_re, app_im, head >= 0,
                                  head >= 0 ? mask : NULL,
                                  head >= 0 ? &coef[0][0][0] : NULL,
                                  head >= 0 ? stream_alpha(head) : 0.0f,
                                  0.0f, out_a_re, out_a_im, NULL);
        (void)dfn2_enhance_spectrum_stream(&st_b, app2_re, app2_im, head >= 0,
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
    free(block_a);
    free(block_b);
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
    const size_t state_bytes = dfn2_state_mem_size(grid);
    void* block = aligned_block(state_bytes);
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
        CHECK(block != NULL && dfn2_state_init(&st, block, state_bytes, grid) == 0,
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
                float ref = (db - mean) / grid->erb_norm_scale_db;
                ref_state[b] = mean;
                CHECK(memcmp(&fe[b], &ref, sizeof(ref)) == 0,
                      "DFN2 ERB features over the nonzero spans equal the dense sum");
            }

            for (int b = 0; b < BANDS; ++b) mask[b] = 0.3f + 0.02f * (float)b;
            if (frame % 3 == 0) mask[frame % BANDS] = (frame & 1) ? INFINITY : NAN;
            if (frame % 7 == 2) mask[0] = -INFINITY;
            if (frame % 10 == 9) mask[BANDS - 1] = -0.0f;
            (void)dfn2_enhance_spectrum(&st, re, im, mask, coefs, 0.5f, out_re, out_im);
            memset(ref_gain, 0, sizeof(ref_gain));
            for (int b = 0; b < BANDS; ++b)
                for (int k = 0; k < BINS; ++k) ref_gain[k] += inv[b][k] * mask[b];
            CHECK(memcmp(st.scratch_bin_gain, ref_gain, sizeof(ref_gain)) == 0,
                  "DFN2 mask expansion over the nonzero spans equals the dense sum");
        }
    }
#undef BIN_AT
    free(block);
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
        ROWS = DFN2_TEST_GRU_ROWS, HIDDEN = DFN2_TEST_GRU_HIDDEN,
        CH = DFN2_TEST_CONVP_CHANNELS, K = DFN2_TEST_CONVP_WINDOW,
        BINS = DFN2_DF_BINS, HOPS = 14
    };
    /* The recurrent state is two host buffers registered with the model I/O state:
     * h_gru [ROWS][HIDDEN] and the pathway window [CH][K][BINS]. */
    static DFN2ModelIOState state;
    static float gru_hidden[ROWS * HIDDEN];
    static float df_convp_window[CH * K * BINS];
    static float ref_gru[ROWS * HIDDEN];
    static float ref_win[CH * K * BINS];
    static float previous[ROWS][HIDDEN];
    const DFN2Grid* grid = dfn2_test_grid();
    const size_t block_bytes = dfn2_model_io_mem_size(grid);
    void* block = aligned_block(block_bytes);
    unsigned int generation;
    int zeroings = 0;

    CHECK(block != NULL &&
          dfn2_model_io_init(&state, block, block_bytes, grid, gru_hidden,
                             (size_t)ROWS * HIDDEN, df_convp_window,
                             (size_t)CH * K * BINS) == 0,
          "DFN2 model I/O state starts for the test grid");
    CHECK(state.gru_hidden_elements == sizeof(ref_gru) / sizeof(float) &&
          state.df_convp_window_elements == sizeof(ref_win) / sizeof(float) &&
          state.gru_hidden_elements == DFN2_TEST_GRU_ELEMENTS &&
          state.df_convp_window_elements == DFN2_TEST_CONVP_ELEMENTS,
          "DFN2 state arrays hold exactly the graph tensors");
    generation = dfn2_model_io_state_generation(&state);
    for (int hop = 0; hop < HOPS; ++hop) {
        int expect_refusal = 0;
        /* The stand-in reads every state input before it writes any output. */
        memcpy(previous, gru_hidden, sizeof(previous));
        for (int row = 0; row < ROWS; ++row)
            for (int j = 0; j < HIDDEN; ++j)
                gru_hidden[row * HIDDEN + j] =
                    standin_row(previous[row][j], hop, row, j);
        for (int c = 0; c < CH; ++c) {
            float* channel = df_convp_window + (size_t)c * K * BINS;
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
            df_convp_window[(size_t)CH * K * BINS - 1] = (float)NAN;
            expect_refusal = 1;
        }
        if (hop == 9) {
            gru_hidden[(DFN2_TEST_GRU_ROW_DF + 1) * HIDDEN + HIDDEN - 1] =
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
        CHECK(memcmp(gru_hidden, ref_gru, sizeof(ref_gru)) == 0,
              "DFN2 h_gru equals the flat reference after the hop");
        CHECK(memcmp(df_convp_window, ref_win, sizeof(ref_win)) == 0,
              "DFN2 pathway window equals the flat reference after the hop");
        CHECK(dfn2_model_io_state_generation(&state) ==
                  generation + (unsigned)zeroings,
              "DFN2 generation counts the zeroings");
        *digest = hash_bytes(*digest, gru_hidden, sizeof(gru_hidden));
        *digest = hash_bytes(*digest, df_convp_window, sizeof(ref_win));
    }
    /* The row offsets address the rows the exporter slices: encoder GRU at
     * row 0, the two ERB decoder layers from row 1, the two DF decoder layers
     * from row 3. */
    for (int row = 0; row < ROWS; ++row)
        for (int j = 0; j < HIDDEN; ++j)
            gru_hidden[row * HIDDEN + j] = standin_row(0.0f, 0, row, j);
    CHECK(gru_hidden[DFN2_TEST_GRU_ROW_ENCODER * HIDDEN + 5] ==
              standin_row(0.0f, 0, 0, 5) &&
          gru_hidden[DFN2_TEST_GRU_ROW_ERB * HIDDEN + 5] ==
              standin_row(0.0f, 0, 1, 5) &&
          gru_hidden[DFN2_TEST_GRU_ROW_DF * HIDDEN + 5] ==
              standin_row(0.0f, 0, 3, 5),
          "DFN2 GRU row offsets address rows 0, 1 and 3");
    free(block);
    return 1;
}

static int test_gtcrn_state_evolution(uint64_t* digest)
{
    enum { HOPS = 14, CH = GTCRN_TEST_CONV_CHANNELS, F = GTCRN_TEST_CONV_FREQ,
           TRA = GTCRN_TEST_TRA_GRUS * GTCRN_TEST_TRA_HIDDEN,
           DP_ROW = GTCRN_TEST_DPGRNN_FREQ * GTCRN_TEST_DPGRNN_HIDDEN,
           DP = GTCRN_TEST_DPGRNN_GRUS * DP_ROW };
    static float state[GTCRN_TEST_STATE_FLOATS];
    static float fwd[GTCRN_TEST_ERB_HIGH_BINS * GTCRN_TEST_ERB_HIGH_BANDS];
    static float inv[GTCRN_TEST_ERB_HIGH_BANDS * GTCRN_TEST_ERB_HIGH_BINS];
    static float ref_conv[6][CH * 11 * F];
    static float ref_tra[TRA];
    static float ref_dp[DP];
    static float previous_tra[TRA];
    static float previous_dp[DP];
    float* conv[6];
    float* tra;
    float* dp;
    size_t conv_n[6];
    GtcrnHost host;
    unsigned int generation;
    int zeroings = 0;
    size_t offset = 0;

    CHECK(gtcrn_host_open(&host, fwd, inv, state), "GTCRN process starts");
    for (int i = 0; i < 6; ++i) {
        conv[i] = state + offset;
        conv_n[i] = gtcrn_test_state_elements[i];
        CHECK(conv_n[i] == (size_t)CH * gtcrn_test_conv_window[i] * F,
              "GTCRN conv tensor holds the graph window");
        offset += conv_n[i];
    }
    tra = state + offset;
    dp = tra + TRA;
    CHECK(gtcrn_test_state_elements[GTCRN_STATE_H_TRA] == (size_t)TRA &&
          gtcrn_test_state_elements[GTCRN_STATE_H_DPGRNN] == (size_t)DP &&
          offset + TRA + DP == GTCRN_TEST_STATE_FLOATS,
          "GTCRN recurrent tensors hold the graph tensors");
    generation = gtcrn_process_state_generation(host.p);

    for (int hop = 0; hop < HOPS; ++hop) {
        int expect_refusal = 0;
        memcpy(previous_tra, tra, sizeof(previous_tra));
        memcpy(previous_dp, dp, sizeof(previous_dp));
        for (int i = 0; i < 6; ++i) {
            const int k = gtcrn_test_conv_window[i];
            for (int c = 0; c < CH; ++c) {
                float* base = conv[i] + (size_t)c * k * F;
                for (int t = 0; t + 1 < k; ++t)
                    memcpy(base + (size_t)t * F, base + (size_t)(t + 1) * F,
                           F * sizeof(float));
                for (int f = 0; f < F; ++f)
                    base[(size_t)(k - 1) * F + f] = standin_frame(hop, i, c, f);
            }
        }
        for (int row = 0; row < GTCRN_TEST_TRA_GRUS; ++row)
            for (int j = 0; j < GTCRN_TEST_TRA_HIDDEN; ++j)
                tra[row * GTCRN_TEST_TRA_HIDDEN + j] = standin_row(
                    previous_tra[row * GTCRN_TEST_TRA_HIDDEN + j], hop, row, j);
        for (int row = 0; row < GTCRN_TEST_DPGRNN_GRUS; ++row)
            for (int column = 0; column < DP_ROW; ++column)
                dp[row * DP_ROW + column] = standin_row(
                    previous_dp[row * DP_ROW + column], hop, row, column);

        for (int i = 0; i < 6; ++i)
            reference_window(ref_conv[i], CH, gtcrn_test_conv_window[i], F, hop, i);
        for (int row = 0; row < GTCRN_TEST_TRA_GRUS; ++row)
            for (int j = 0; j < GTCRN_TEST_TRA_HIDDEN; ++j)
                ref_tra[row * GTCRN_TEST_TRA_HIDDEN + j] = standin_row(
                    ref_tra[row * GTCRN_TEST_TRA_HIDDEN + j], hop, row, j);
        for (int row = 0; row < GTCRN_TEST_DPGRNN_GRUS; ++row)
            for (int column = 0; column < DP_ROW; ++column) {
                float* cell = &ref_dp[row * DP_ROW + column];
                *cell = standin_row(*cell, hop, row, column);
            }

        if (hop == 5) {
            dp[((size_t)(GTCRN_TEST_DPGRNN_GRUS - 1) * GTCRN_TEST_DPGRNN_FREQ + 32) *
                   GTCRN_TEST_DPGRNN_HIDDEN + 7] = (float)NAN;
            expect_refusal = 1;
        }
        if (hop == 9) {
            conv[4][((size_t)(CH - 1) * gtcrn_test_conv_window[4] +
                     (gtcrn_test_conv_window[4] - 1)) * F + (F - 1)] = (float)INFINITY;
            expect_refusal = 1;
        }
        CHECK((gtcrn_process_state_validate(host.p) != 0) == expect_refusal,
              "GTCRN validate refuses exactly the non-finite hops");
        if (expect_refusal) {
            ++zeroings;
            memset(ref_conv, 0, sizeof(ref_conv));
            memset(ref_tra, 0, sizeof(ref_tra));
            memset(ref_dp, 0, sizeof(ref_dp));
        }
        if (hop == 11) {
            gtcrn_process_reset(host.p);
            ++zeroings;
            memset(ref_conv, 0, sizeof(ref_conv));
            memset(ref_tra, 0, sizeof(ref_tra));
            memset(ref_dp, 0, sizeof(ref_dp));
        }
        for (int i = 0; i < 6; ++i) {
            CHECK(memcmp(conv[i], ref_conv[i], conv_n[i] * sizeof(float)) == 0,
                  "GTCRN conv window equals the flat reference");
            *digest = hash_bytes(*digest, conv[i], conv_n[i] * sizeof(float));
        }
        CHECK(memcmp(tra, ref_tra, sizeof(ref_tra)) == 0,
              "GTCRN h_tra equals the flat reference");
        CHECK(memcmp(dp, ref_dp, sizeof(ref_dp)) == 0,
              "GTCRN h_dpgrnn equals the flat reference");
        CHECK(gtcrn_process_state_generation(host.p) ==
                  generation + (unsigned)zeroings,
              "GTCRN generation counts the zeroings");
        *digest = hash_bytes(*digest, tra, sizeof(ref_tra));
        *digest = hash_bytes(*digest, dp, sizeof(ref_dp));
    }
    free(host.pool);
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
    CHECK(test_gtcrn_grid_make_refusals(), "GTCRN grid_make refusals");
    CHECK(test_gtcrn_grid_validate_refusals(), "GTCRN edited grids are refused everywhere");
    CHECK(test_gtcrn_config_and_pool_refusals(), "GTCRN config and pool refusals");
    CHECK(test_gtcrn_stale_pool_gate(), "GTCRN stale-pool gate");
    CHECK(test_gtcrn_registration_all_or_none(), "GTCRN state registration is all or none");
    CHECK(test_gtcrn_mask_validate(), "GTCRN mask_validate");
    CHECK(test_gtcrn_board_path_registers_nothing(), "GTCRN board path registers nothing");
    CHECK(test_gtcrn_set_erb_matrices(), "GTCRN set_erb_matrices");
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
