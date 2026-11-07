#include "dfn2_model_io.h"
#include "dfn2_model_io_priv.h"

#include <math.h>
#include <string.h>

#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

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

/* The block's one walk, shared by dfn2_model_io_mem_size() (base==NULL) and
 * dfn2_model_io_init(). */
static int io_layout(DFN2ModelIOState *state, unsigned char *base, const DFN2Grid *g,
                     size_t *total)
{
    size_t cursor = 0;
    float *erb = NULL;
    float *spec = NULL;
    if (io_carve(base, &cursor, (size_t)g->input_frames * (size_t)g->n_erb, &erb) != 0 ||
        io_carve(base, &cursor,
                 (size_t)2 * (size_t)g->input_frames * (size_t)g->df_bins, &spec) != 0)
        return -1;
    if (state) {
        state->erb_window = erb;
        state->spec_window = spec;
    }
    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

size_t dfn2_model_io_mem_size(const DFN2Grid *grid)
{
    size_t total = 0;
    if (dfn2_grid_validate(grid) != DFN2_GRID_OK) return 0;
    if (io_layout(NULL, NULL, grid, &total) != 0) return 0;
    return total;
}

/* Zero the registered arrays (none: no-op). */
static void zero_registered(DFN2ModelIOState *state)
{
    if (state->gru_hidden)
        memset(state->gru_hidden, 0, state->gru_hidden_elements * sizeof(float));
#ifndef DFN2_MASK_ONLY
    if (state->df_convp_window)
        memset(state->df_convp_window, 0,
               state->df_convp_window_elements * sizeof(float));
#endif
}

int dfn2_model_io_init(DFN2ModelIOState *state, void *block, size_t bytes,
                       const DFN2Grid *grid, float *gru_hidden,
                       size_t gru_hidden_elements, float *df_convp_history,
                       size_t df_convp_history_elements)
{
    size_t need;
    size_t total = 0;
    if (state == NULL) return -1;
    memset(state, 0, sizeof(*state));
    need = dfn2_model_io_mem_size(grid);
    if (need == 0 || block == NULL || bytes < need || !MEM_IS_ALIGNED16(block)) return -1;
    if (dfn2_registration_check(gru_hidden, gru_hidden_elements, df_convp_history,
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
    state->frames_per_call = grid->frames_per_call;
    state->input_frames = grid->input_frames;
    state->state_generation = 1u;
    state->gru_hidden = gru_hidden;
    state->gru_hidden_elements = gru_hidden_elements;
#ifndef DFN2_MASK_ONLY
    state->df_convp_window = df_convp_history;
    state->df_convp_window_elements = df_convp_history_elements;
#else
    (void)df_convp_history_elements;
#endif
    zero_registered(state);
    return 0;
}

unsigned int dfn2_model_io_ready_frames(const DFN2ModelIOState *state)
{
    if (state == NULL || state->n_erb == 0) return 0u;
    return (unsigned int)state->mask_lookahead + (unsigned int)state->frames_per_call;
}

#ifdef DFN2_MASK_ONLY
#define DFN2_MODEL_IO_STATE_TENSORS 1
#else
#define DFN2_MODEL_IO_STATE_TENSORS 2
#endif

/* One list of (field, contract value) per kind drives both the default and the validator,
 * so they cannot diverge. `g` is the grid. The external fields are what the C code relies
 * on; the interior ones are the model's and are only carried. */
#define DFN2_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(X)                   \
    X(layout_version, (unsigned int)DFN2_MODEL_IO_LAYOUT_VERSION)     \
    X(sample_rate, g->sample_rate)                                    \
    X(fft_size, g->n_fft)                                             \
    X(hop_size, g->hop)                                               \
    X(spectrum_bins, g->n_bins)                                       \
    X(erb_bands, g->n_erb)                                            \
    X(df_bins, g->df_bins)                                            \
    X(df_order, g->df_order)                                          \
    X(mask_lookahead, g->mask_lookahead)                              \
    X(df_lookahead, g->df_lookahead)                                  \
    X(input_frames, g->input_frames)                                  \
    X(state_tensor_count, DFN2_MODEL_IO_STATE_TENSORS)

#define DFN2_MODEL_IO_DESCRIPTOR_INTERIOR_FIELDS(X)                   \
    X(encoder_gru_layers)                                             \
    X(erb_gru_layers)                                                 \
    X(df_gru_layers)                                                  \
    X(gru_hidden)                                                     \
    X(encoder_channels)                                               \
    X(df_pathway_window)

/* Every descriptor field is in one of the two lists. */
#define DFN2_COUNT_FIELD(field, value) + 1
#define DFN2_COUNT_INTERIOR(field) + 1
enum { DFN2_MODEL_IO_DESCRIPTOR_FIELD_COUNT =
           0 DFN2_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(DFN2_COUNT_FIELD)
             DFN2_MODEL_IO_DESCRIPTOR_INTERIOR_FIELDS(DFN2_COUNT_INTERIOR) };
#undef DFN2_COUNT_FIELD
#undef DFN2_COUNT_INTERIOR
#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN2_MODEL_IO_DESCRIPTOR_FIELD_COUNT == 18 &&
                   sizeof(DFN2ModelIoDescriptor) ==
                       DFN2_MODEL_IO_DESCRIPTOR_FIELD_COUNT * sizeof(int),
               "the field lists cover the 18-field descriptor");
#endif
#endif

int dfn2_model_io_descriptor_default(const DFN2Grid *g,
                                     DFN2ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
    if (dfn2_grid_validate(g) != DFN2_GRID_OK) return -1;
#define DFN2_SET_FIELD(field, value) descriptor->field = (value);
    DFN2_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(DFN2_SET_FIELD)
#undef DFN2_SET_FIELD
    return 0;
}

int dfn2_model_io_descriptor_validate(const DFN2Grid *g,
                                      const DFN2ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL || dfn2_grid_validate(g) != DFN2_GRID_OK) return -1;
#define DFN2_CHECK_FIELD(field, value) \
    if (descriptor->field != (value)) return -1;
    DFN2_MODEL_IO_DESCRIPTOR_EXTERNAL_FIELDS(DFN2_CHECK_FIELD)
#undef DFN2_CHECK_FIELD
    return 0;
}

void dfn2_model_io_push_erb_window(float *window, const float *frame, int n_erb,
                                   int window_frames)
{
    if (window == NULL || frame == NULL || n_erb <= 0 || window_frames <= 0) return;
    memmove(window, window + n_erb,
            (size_t)(window_frames - 1) * (size_t)n_erb * sizeof(float));
    memcpy(window + (size_t)(window_frames - 1) * (size_t)n_erb, frame,
           (size_t)n_erb * sizeof(float));
}

void dfn2_model_io_push_spec_window(float *window, const float *frame, int df_bins,
                                    int window_frames)
{
    if (window == NULL || frame == NULL || df_bins <= 0 || window_frames <= 0) return;
    for (int channel = 0; channel < 2; ++channel) {
        float *plane = window + (size_t)channel * (size_t)window_frames *
                                    (size_t)df_bins;
        memmove(plane, plane + df_bins,
                (size_t)(window_frames - 1) * (size_t)df_bins * sizeof(float));
        memcpy(plane + (size_t)(window_frames - 1) * (size_t)df_bins,
               frame + (size_t)channel * df_bins,
               (size_t)df_bins * sizeof(float));
    }
}

static int all_finite(const float *values, size_t count)
{
    return skn_all_finite_f32(values, count);
}

int dfn2_model_io_validate_arrays(float *gru_hidden, size_t gru_count,
                                  float *df_convp_history, size_t convp_count)
{
    if (gru_hidden == NULL || gru_count == 0u) return -1;
#ifdef DFN2_MASK_ONLY
    if (df_convp_history != NULL || convp_count != 0u) return -1;
#else
    if (convp_count != 0u && df_convp_history == NULL) return -1;
#endif
    if (all_finite(gru_hidden, gru_count) &&
        (convp_count == 0u || all_finite(df_convp_history, convp_count))) return 0;
    /* Zero is the only state a stream can restart from after an in-place overwrite. */
    memset(gru_hidden, 0, gru_count * sizeof(float));
    if (convp_count != 0u) memset(df_convp_history, 0, convp_count * sizeof(float));
    return -1;
}

int dfn2_model_io_push_features(DFN2ModelIOState *state, const float *erb,
                                const float *spec)
{
    const unsigned int ready = dfn2_model_io_ready_frames(state);
    DFN2_REQUIRE_VARIANT_LINK();
    if (state == NULL || erb == NULL || spec == NULL || ready == 0u) return -1;
    dfn2_model_io_push_erb_window(state->erb_window, erb, state->n_erb,
                                  state->input_frames);
    dfn2_model_io_push_spec_window(state->spec_window, spec, state->df_bins,
                                   state->input_frames);
    /* The count runs to L + N and, after each complete call, steps back by N,
     * so it stays at L + 1 .. L + N once the warm-up is over. */
    if (state->feature_frames_seen == ready)
        state->feature_frames_seen -= (unsigned int)state->frames_per_call;
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
    if (state->gru_hidden == NULL) return 0;   /* nothing registered: nothing to scan */
#ifdef DFN2_MASK_ONLY
    if (dfn2_model_io_validate_arrays(state->gru_hidden, state->gru_hidden_elements,
                                      NULL, 0u) == 0)
        return 0;
#else
    if (dfn2_model_io_validate_arrays(state->gru_hidden, state->gru_hidden_elements,
                                      state->df_convp_window,
                                      state->df_convp_window_elements) == 0)
        return 0;
#endif
    ++state->state_generation;
    return -1;
}

void dfn2_model_io_reset(DFN2ModelIOState *state)
{
    if (state == NULL) return;
    if (state->block) memset(state->block, 0, state->block_bytes);
    zero_registered(state);
    state->feature_frames_seen = 0u;
    ++state->state_generation;
}

void dfn2_model_io_zero_recurrent(DFN2ModelIOState *state)
{
    if (state == NULL) return;
    zero_registered(state);
    ++state->state_generation;
}

unsigned int dfn2_model_io_state_generation(const DFN2ModelIOState *state)
{
    return state == NULL ? 0u : state->state_generation;
}
