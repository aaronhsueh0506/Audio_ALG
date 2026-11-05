#ifndef DFN2_MULTI_FRAME_HOST
#define DFN2_MULTI_FRAME_HOST 1
#endif
#include "dfn2_model_io.h"

#include <math.h>
#include <string.h>

#include "utility/simd_kernel_nn.h"

/* The frames per call this object was built for (dfn2_model_io.h): objects
 * of another N fail to link. */
const unsigned char DFN2_FRAMES_PER_CALL_SYMBOL = 0;

int dfn2_model_io_init(DFN2ModelIOState *state, const DFN2Grid *grid)
{
    if (state == NULL) return -1;
    memset(state, 0, sizeof(*state));
    if (dfn2_grid_validate(grid) != DFN2_GRID_OK) return -1;
    state->n_erb = grid->n_erb;
    state->df_bins = grid->df_bins;
    state->mask_lookahead = grid->mask_lookahead;
    state->state_generation = 1u;
    return 0;
}

unsigned int dfn2_model_io_ready_frames(const DFN2ModelIOState *state)
{
    if (state == NULL || state->n_erb == 0) return 0u;
    return (unsigned int)state->mask_lookahead + DFN2_FRAMES_PER_CALL;
}

#ifdef DFN2_MASK_ONLY
#define DFN2_MODEL_IO_STATE_TENSORS 1
#else
#define DFN2_MODEL_IO_STATE_TENSORS 2
#endif

/* One list of (field, contract value) drives both the default and the validator,
 * so they cannot diverge. `g` is the grid. */
#define DFN2_MODEL_IO_DESCRIPTOR_FIELDS(X)                            \
    X(layout_version, (unsigned int)DFN2_MODEL_IO_LAYOUT_VERSION)     \
    X(sample_rate, g->sample_rate)                                    \
    X(fft_size, g->n_fft)                                             \
    X(hop_size, g->hop)                                               \
    X(spectrum_bins, g->n_bins)                                       \
    X(erb_bands, g->n_erb)                                            \
    X(df_bins, g->df_bins)                                            \
    X(df_order, DFN2_DF_ORDER)                                        \
    X(mask_lookahead, g->mask_lookahead)                              \
    X(df_lookahead, g->df_lookahead)                                  \
    X(input_frames, DFN2_MODEL_WINDOW_FRAMES)                         \
    X(encoder_gru_layers, DFN2_MODEL_ENCODER_GRU_LAYERS)              \
    X(erb_gru_layers, DFN2_MODEL_ERB_GRU_LAYERS)                      \
    X(df_gru_layers, DFN2_MODEL_DF_GRU_LAYERS)                        \
    X(gru_hidden, DFN2_MODEL_GRU_HIDDEN)                              \
    X(encoder_channels, DFN2_MODEL_ENCODER_CHANNELS)                  \
    X(df_pathway_window, DFN2_MODEL_DF_PATHWAY_WINDOW)                \
    X(state_tensor_count, DFN2_MODEL_IO_STATE_TENSORS)

/* Every descriptor field is in the list, so validate() checks all of them. */
#define DFN2_COUNT_FIELD(field, value) + 1
enum { DFN2_MODEL_IO_DESCRIPTOR_FIELD_COUNT =
           0 DFN2_MODEL_IO_DESCRIPTOR_FIELDS(DFN2_COUNT_FIELD) };
#undef DFN2_COUNT_FIELD
#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN2_MODEL_IO_DESCRIPTOR_FIELD_COUNT == 18 &&
                   sizeof(DFN2ModelIoDescriptor) ==
                       DFN2_MODEL_IO_DESCRIPTOR_FIELD_COUNT * sizeof(int),
               "the field list covers the 18-field descriptor");
#endif
#endif

int dfn2_model_io_descriptor_default(const DFN2Grid *g,
                                     DFN2ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
    if (dfn2_grid_validate(g) != DFN2_GRID_OK) return -1;
#define DFN2_SET_FIELD(field, value) descriptor->field = (value);
    DFN2_MODEL_IO_DESCRIPTOR_FIELDS(DFN2_SET_FIELD)
#undef DFN2_SET_FIELD
    return 0;
}

int dfn2_model_io_descriptor_validate(const DFN2Grid *g,
                                      const DFN2ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL || dfn2_grid_validate(g) != DFN2_GRID_OK) return -1;
#define DFN2_CHECK_FIELD(field, value) \
    if (descriptor->field != (value)) return -1;
    DFN2_MODEL_IO_DESCRIPTOR_FIELDS(DFN2_CHECK_FIELD)
#undef DFN2_CHECK_FIELD
    return 0;
}

void dfn2_model_io_push_erb_window(float *window, const float *frame, int n_erb)
{
    if (window == NULL || frame == NULL || n_erb <= 0) return;
    memmove(window, window + n_erb,
            (DFN2_MODEL_WINDOW_FRAMES - 1) * (size_t)n_erb * sizeof(float));
    memcpy(window + (DFN2_MODEL_WINDOW_FRAMES - 1) * (size_t)n_erb, frame,
           (size_t)n_erb * sizeof(float));
}

void dfn2_model_io_push_spec_window(float *window, const float *frame,
                                    int df_bins)
{
    if (window == NULL || frame == NULL || df_bins <= 0) return;
    for (int channel = 0; channel < 2; ++channel) {
        float *plane = window + (size_t)channel * DFN2_MODEL_WINDOW_FRAMES *
                                    (size_t)df_bins;
        memmove(plane, plane + df_bins,
                (DFN2_MODEL_WINDOW_FRAMES - 1) * (size_t)df_bins *
                    sizeof(float));
        memcpy(plane + (DFN2_MODEL_WINDOW_FRAMES - 1) * (size_t)df_bins,
               frame + (size_t)channel * df_bins,
               (size_t)df_bins * sizeof(float));
    }
}

static int all_finite(const float *values, size_t count)
{
    return skn_all_finite_f32(values, count);
}

#ifdef DFN2_MASK_ONLY
int dfn2_model_io_validate_arrays(
    float gru_hidden[DFN2_MODEL_GRU_ROWS][DFN2_MODEL_GRU_HIDDEN])
{
    const size_t gru_count = (size_t)DFN2_MODEL_GRU_ROWS *
                             DFN2_MODEL_GRU_HIDDEN;
    if (gru_hidden == NULL) return -1;
    if (all_finite(&gru_hidden[0][0], gru_count)) return 0;
    /* Zero is the only state a stream can restart from after an in-place overwrite. */
    memset(gru_hidden, 0, gru_count * sizeof(float));
    return -1;
}
#else
int dfn2_model_io_validate_arrays(
    float gru_hidden[DFN2_MODEL_GRU_ROWS][DFN2_MODEL_GRU_HIDDEN],
    float *df_convp_window, int df_bins)
{
    const size_t gru_count = (size_t)DFN2_MODEL_GRU_ROWS *
                             DFN2_MODEL_GRU_HIDDEN;
    const size_t convp_count = (size_t)DFN2_MODEL_ENCODER_CHANNELS *
                               DFN2_MODEL_DF_PATHWAY_WINDOW *
                               (size_t)(df_bins > 0 ? df_bins : 0);
    if (gru_hidden == NULL || df_convp_window == NULL || df_bins <= 0 ||
        df_bins > DFN2_MAX_DF_BINS) return -1;
    if (all_finite(&gru_hidden[0][0], gru_count) &&
        all_finite(df_convp_window, convp_count)) return 0;
    /* Zero is the only state a stream can restart from after an in-place overwrite. */
    memset(gru_hidden, 0, gru_count * sizeof(float));
    memset(df_convp_window, 0, convp_count * sizeof(float));
    return -1;
}
#endif

int dfn2_model_io_push_features(DFN2ModelIOState *state, const float *erb,
                                const float *spec)
{
    const unsigned int ready = dfn2_model_io_ready_frames(state);
    DFN2_REQUIRE_VARIANT_LINK();
    if (state == NULL || erb == NULL || spec == NULL || ready == 0u) return -1;
    dfn2_model_io_push_erb_window(state->erb_window, erb, state->n_erb);
    dfn2_model_io_push_spec_window(state->spec_window, spec, state->df_bins);
    /* The count runs to L + N and, after each complete call, steps back by N,
     * so it stays at L + 1 .. L + N once the warm-up is over. */
    if (state->feature_frames_seen == ready)
        state->feature_frames_seen -= DFN2_FRAMES_PER_CALL;
    if (state->feature_frames_seen < ready)
        ++state->feature_frames_seen;
    return state->feature_frames_seen == ready ? 1 : 0;
}

int dfn2_model_io_frame_slot(const DFN2ModelIOState *state)
{
    unsigned long long seen;
    if (state == NULL) return -1;
    seen = state->feature_frames_seen;
    if (seen <= (unsigned long long)state->mask_lookahead ||
        seen > dfn2_model_io_ready_frames(state))
        return -1;
    return (int)(seen - (unsigned long long)state->mask_lookahead - 1U);
}

int dfn2_model_io_validate_state(DFN2ModelIOState *state)
{
    if (state == NULL) return -1;
#ifdef DFN2_MASK_ONLY
    if (dfn2_model_io_validate_arrays(state->gru_hidden) == 0) return 0;
#else
    if (dfn2_model_io_validate_arrays(state->gru_hidden, state->df_convp_window,
                                      state->df_bins) == 0)
        return 0;
#endif
    ++state->state_generation;
    return -1;
}

void dfn2_model_io_reset(DFN2ModelIOState *state)
{
    unsigned int generation;
    int n_erb, df_bins, mask_lookahead;
    if (state == NULL) return;
    n_erb = state->n_erb;
    df_bins = state->df_bins;
    mask_lookahead = state->mask_lookahead;
    generation = state->state_generation + 1u;
    memset(state, 0, sizeof(*state));
    state->n_erb = n_erb;
    state->df_bins = df_bins;
    state->mask_lookahead = mask_lookahead;
    state->state_generation = generation;
}

void dfn2_model_io_zero_recurrent(DFN2ModelIOState *state)
{
    if (state == NULL) return;
    memset(state->gru_hidden, 0, sizeof(state->gru_hidden));
#ifndef DFN2_MASK_ONLY
    memset(state->df_convp_window, 0, sizeof(state->df_convp_window));
#endif
    ++state->state_generation;
}

unsigned int dfn2_model_io_state_generation(const DFN2ModelIOState *state)
{
    return state == NULL ? 0u : state->state_generation;
}
