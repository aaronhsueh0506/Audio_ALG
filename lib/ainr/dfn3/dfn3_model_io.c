#include "dfn3_model_io.h"

#include <math.h>
#include <string.h>

#include "dfn3_model_io_priv.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#define DFN3_MODEL_IO_STATE_TENSORS 2

/* Reserve `count` floats at *cursor, 16-byte aligned, saturating on overflow. base==NULL
 * sizes only. */
static int io_carve(unsigned char *base, size_t *cursor, size_t count, float **out)
{
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, ck_mul_size(count, sizeof(float)));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = (float *)(base + aligned);
    *cursor = next;
    return 0;
}

/* The block's one walk, shared by dfn3_model_io_mem_size() (base==NULL) and
 * dfn3_model_io_init(). */
static int io_layout(DFN3ModelIOState *state, unsigned char *base, const DFN3Grid *g,
                     size_t *total)
{
    size_t cursor = 0;
    float *erb = NULL;
    float *spec = NULL;
    if (io_carve(base, &cursor, (size_t)DFN3_MODEL_INPUT_FRAMES * (size_t)g->n_erb, &erb) != 0 ||
        io_carve(base, &cursor,
                 (size_t)2 * DFN3_MODEL_INPUT_FRAMES * (size_t)g->df_bins, &spec) != 0)
        return -1;
    if (state) {
        state->erb_window = erb;
        state->spec_window = spec;
    }
    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

size_t dfn3_model_io_mem_size(const DFN3Grid *grid)
{
    size_t total = 0;
    if (dfn3_grid_validate(grid) != DFN3_GRID_OK) return 0;
    if (io_layout(NULL, NULL, grid, &total) != 0) return 0;
    return total;
}

/* Zero the registered arrays (none: no-op). */
static void zero_registered(DFN3ModelIOState *state)
{
    if (state->gru_hidden)
        memset(state->gru_hidden, 0, state->gru_hidden_elements * sizeof(float));
    if (state->df_convp_window)
        memset(state->df_convp_window, 0,
               state->df_convp_window_elements * sizeof(float));
}

int dfn3_model_io_init(DFN3ModelIOState *state, void *block, size_t bytes,
                       const DFN3Grid *grid, float *gru_hidden,
                       size_t gru_hidden_elements, float *df_convp_history,
                       size_t df_convp_history_elements)
{
    size_t need;
    size_t total = 0;
    if (state == NULL) return -1;
    memset(state, 0, sizeof(*state));
    need = dfn3_model_io_mem_size(grid);
    if (need == 0 || block == NULL || bytes < need || !MEM_IS_ALIGNED16(block)) return -1;
    if (dfn3_registration_check(gru_hidden, gru_hidden_elements, df_convp_history,
                                df_convp_history_elements) != 0)
        return -1;
    memset(block, 0, need);
    if (io_layout(state, (unsigned char *)block, grid, &total) != 0 || total != need) {
        memset(state, 0, sizeof(*state));
        return -1;
    }
    state->block = block;
    state->block_bytes = need;
    state->n_erb = grid->n_erb;
    state->df_bins = grid->df_bins;
    state->mask_lookahead = grid->mask_lookahead;
    state->state_generation = 1u;
    state->gru_hidden = gru_hidden;
    state->gru_hidden_elements = gru_hidden_elements;
    state->df_convp_window = df_convp_history;
    state->df_convp_window_elements = df_convp_history_elements;
    zero_registered(state);
    return 0;
}

unsigned int dfn3_model_io_ready_frames(const DFN3ModelIOState *state)
{
    if (state == NULL || state->n_erb == 0) return 0u;
    return (unsigned int)state->mask_lookahead + 1u;
}

/* One list of (field, contract value) drives both the default and the validator, so they
 * cannot diverge. `g` is the grid. The external fields are what the C code relies on; the
 * interior ones are the model's and are only carried. */
#define DFN3_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(X)                   \
    X(layout_version, (unsigned int)DFN3_MODEL_IO_LAYOUT_VERSION)     \
    X(sample_rate, g->sample_rate)                                    \
    X(fft_size, g->n_fft)                                             \
    X(hop_size, g->hop)                                               \
    X(spectrum_bins, g->n_bins)                                       \
    X(erb_bands, g->n_erb)                                            \
    X(df_bins, g->df_bins)                                            \
    X(df_order, g->df_order)                                          \
    X(mask_lookahead, g->mask_lookahead)                              \
    X(df_lookahead, g->df_lookahead)                                  \
    X(input_frames, DFN3_MODEL_INPUT_FRAMES)                          \
    X(state_tensor_count, DFN3_MODEL_IO_STATE_TENSORS)

#define DFN3_MODEL_IO_DESCRIPTOR_INTERIOR_FIELDS(X)                   \
    X(encoder_gru_layers)                                             \
    X(erb_gru_layers)                                                 \
    X(df_gru_layers)                                                  \
    X(gru_hidden)                                                     \
    X(encoder_channels)                                               \
    X(df_pathway_window)

/* Every descriptor field is in one of the two lists. */
#define DFN3_COUNT_FIELD(field, value) + 1
#define DFN3_COUNT_INTERIOR(field) + 1
enum { DFN3_MODEL_IO_DESCRIPTOR_FIELD_COUNT =
           0 DFN3_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(DFN3_COUNT_FIELD)
             DFN3_MODEL_IO_DESCRIPTOR_INTERIOR_FIELDS(DFN3_COUNT_INTERIOR) };
#undef DFN3_COUNT_FIELD
#undef DFN3_COUNT_INTERIOR
#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN3_MODEL_IO_DESCRIPTOR_FIELD_COUNT == 18 &&
                   sizeof(DFN3ModelIoDescriptor) ==
                       DFN3_MODEL_IO_DESCRIPTOR_FIELD_COUNT * sizeof(int),
               "the field lists cover the 18-field descriptor");
#endif
#endif

int dfn3_model_io_descriptor_default(const DFN3Grid *g,
                                     DFN3ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
    if (dfn3_grid_validate(g) != DFN3_GRID_OK) return -1;
#define DFN3_SET_FIELD(field, value) descriptor->field = (value);
    DFN3_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(DFN3_SET_FIELD)
#undef DFN3_SET_FIELD
    return 0;
}

int dfn3_model_io_descriptor_validate(const DFN3Grid *g,
                                      const DFN3ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL || dfn3_grid_validate(g) != DFN3_GRID_OK) return -1;
#define DFN3_CHECK_FIELD(field, value) \
    if (descriptor->field != (value)) return -1;
    DFN3_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(DFN3_CHECK_FIELD)
#undef DFN3_CHECK_FIELD
    return 0;
}

void dfn3_model_io_push_erb_window(float *window, const float *frame, int n_erb)
{
    if (window == NULL || frame == NULL || n_erb <= 0) return;
    memmove(window, window + n_erb,
            (DFN3_MODEL_INPUT_FRAMES - 1) * (size_t)n_erb * sizeof(float));
    memcpy(window + (DFN3_MODEL_INPUT_FRAMES - 1) * (size_t)n_erb, frame,
           (size_t)n_erb * sizeof(float));
}

void dfn3_model_io_push_spec_window(float *window, const float *frame, int df_bins)
{
    if (window == NULL || frame == NULL || df_bins <= 0) return;
    for (int channel = 0; channel < 2; ++channel) {
        float *plane = window + (size_t)channel * DFN3_MODEL_INPUT_FRAMES * (size_t)df_bins;
        memmove(plane, plane + df_bins,
                (DFN3_MODEL_INPUT_FRAMES - 1) * (size_t)df_bins * sizeof(float));
        memcpy(plane + (DFN3_MODEL_INPUT_FRAMES - 1) * (size_t)df_bins,
               frame + (size_t)channel * df_bins, (size_t)df_bins * sizeof(float));
    }
}

static int all_finite(const float *values, size_t count)
{
    return skn_all_finite_f32(values, count);
}

int dfn3_model_io_validate_arrays(float *gru_hidden, size_t gru_count,
                                  float *df_convp_history, size_t convp_count)
{
    if (gru_hidden == NULL || gru_count == 0u || df_convp_history == NULL ||
        convp_count == 0u) return -1;
    if (all_finite(gru_hidden, gru_count) &&
        all_finite(df_convp_history, convp_count)) return 0;
    /* Zero is the only state a stream can restart from after an in-place overwrite. */
    memset(gru_hidden, 0, gru_count * sizeof(float));
    memset(df_convp_history, 0, convp_count * sizeof(float));
    return -1;
}

int dfn3_model_io_push_features(DFN3ModelIOState *state, const float *erb,
                                const float *spec)
{
    const unsigned int ready = dfn3_model_io_ready_frames(state);
    if (state == NULL || erb == NULL || spec == NULL || ready == 0u) return -1;
    dfn3_model_io_push_erb_window(state->erb_window, erb, state->n_erb);
    dfn3_model_io_push_spec_window(state->spec_window, spec, state->df_bins);
    if (state->feature_frames_seen < ready)
        ++state->feature_frames_seen;
    return state->feature_frames_seen == ready ? 1 : 0;
}

int dfn3_model_io_validate_state(DFN3ModelIOState *state)
{
    if (state == NULL) return -1;
    if (state->gru_hidden == NULL) return 0;   /* nothing registered: nothing to scan */
    if (dfn3_model_io_validate_arrays(state->gru_hidden, state->gru_hidden_elements,
                                      state->df_convp_window,
                                      state->df_convp_window_elements) == 0)
        return 0;
    ++state->state_generation;
    return -1;
}

void dfn3_model_io_reset(DFN3ModelIOState *state)
{
    if (state == NULL) return;
    if (state->block) memset(state->block, 0, state->block_bytes);
    zero_registered(state);
    state->feature_frames_seen = 0u;
    ++state->state_generation;
}

void dfn3_model_io_zero_recurrent(DFN3ModelIOState *state)
{
    if (state == NULL) return;
    zero_registered(state);
    ++state->state_generation;
}

unsigned int dfn3_model_io_state_generation(const DFN3ModelIOState *state)
{
    return state == NULL ? 0u : state->state_generation;
}
