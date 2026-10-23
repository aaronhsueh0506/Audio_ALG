#include "dfn3_model_io.h"

#include <math.h>
#include <string.h>

#include "utility/simd_kernel_nn.h"

/* The exported graph consumes the fixed [t-1,t,t+1] window; a different head
 * delay needs a different graph window and readiness contract, not only
 * different ring constants. */
_Static_assert(DFN3_MASK_LOOKAHEAD == 1 && DFN3_MODEL_INPUT_FRAMES == 3,
               "DFN3 layout9 requires a three-frame window and mask lookahead1");

#define DFN3_MODEL_IO_STATE_TENSORS 2

/* One list of (field, contract value) drives both the default and the validator,
 * so they cannot diverge. */
#define DFN3_MODEL_IO_DESCRIPTOR_FIELDS(X)                            \
    X(layout_version, (unsigned int)DFN3_MODEL_IO_LAYOUT_VERSION)     \
    X(sample_rate, DFN3_SR)                                           \
    X(fft_size, DFN3_N_FFT)                                           \
    X(hop_size, DFN3_HOP_LEN)                                         \
    X(spectrum_bins, DFN3_N_BINS)                                     \
    X(erb_bands, DFN3_N_ERB)                                          \
    X(df_bins, DFN3_DF_BINS)                                          \
    X(df_order, DFN3_DF_ORDER)                                        \
    X(mask_lookahead, DFN3_MASK_LOOKAHEAD)                            \
    X(df_lookahead, DFN3_DF_LOOKAHEAD)                                \
    X(input_frames, DFN3_MODEL_INPUT_FRAMES)                          \
    X(encoder_gru_layers, DFN3_MODEL_ENCODER_GRU_LAYERS)              \
    X(erb_gru_layers, DFN3_MODEL_ERB_GRU_LAYERS)                      \
    X(df_gru_layers, DFN3_MODEL_DF_GRU_LAYERS)                        \
    X(gru_hidden, DFN3_MODEL_GRU_HIDDEN)                              \
    X(encoder_channels, DFN3_MODEL_ENCODER_CHANNELS)                  \
    X(df_pathway_window, DFN3_MODEL_DF_PATHWAY_WINDOW)              \
    X(state_tensor_count, DFN3_MODEL_IO_STATE_TENSORS)

int dfn3_model_io_descriptor_default(DFN3ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
#define DFN3_SET_FIELD(field, value) descriptor->field = (value);
    DFN3_MODEL_IO_DESCRIPTOR_FIELDS(DFN3_SET_FIELD)
#undef DFN3_SET_FIELD
    return 0;
}

int dfn3_model_io_descriptor_validate(
    const DFN3ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL) return -1;
#define DFN3_CHECK_FIELD(field, value) \
    if (descriptor->field != (value)) return -1;
    DFN3_MODEL_IO_DESCRIPTOR_FIELDS(DFN3_CHECK_FIELD)
#undef DFN3_CHECK_FIELD
    return 0;
}

void dfn3_model_io_push_erb_window(
    float window[DFN3_MODEL_INPUT_FRAMES][DFN3_N_ERB],
    const float frame[DFN3_N_ERB])
{
    if (window == NULL || frame == NULL) return;
    memmove(window[0], window[1],
            (DFN3_MODEL_INPUT_FRAMES - 1) * sizeof(window[0]));
    memcpy(window[DFN3_MODEL_INPUT_FRAMES - 1], frame, sizeof(window[0]));
}

void dfn3_model_io_push_spec_window(
    float window[2][DFN3_MODEL_INPUT_FRAMES][DFN3_DF_BINS],
    const float frame[2][DFN3_DF_BINS])
{
    if (window == NULL || frame == NULL) return;
    for (int channel = 0; channel < 2; ++channel) {
        memmove(window[channel][0], window[channel][1],
                (DFN3_MODEL_INPUT_FRAMES - 1) *
                    sizeof(window[channel][0]));
        memcpy(window[channel][DFN3_MODEL_INPUT_FRAMES - 1],
               frame[channel], sizeof(window[channel][0]));
    }
}

static int all_finite(const float *values, size_t count)
{
    return skn_all_finite_f32(values, count);
}

int dfn3_model_io_validate_arrays(
    float gru_hidden[DFN3_MODEL_GRU_ROWS][DFN3_MODEL_GRU_HIDDEN],
    float df_convp_window[DFN3_MODEL_ENCODER_CHANNELS]
                         [DFN3_MODEL_DF_PATHWAY_WINDOW]
                         [DFN3_DF_BINS])
{
    const size_t gru_count = (size_t)DFN3_MODEL_GRU_ROWS *
                             DFN3_MODEL_GRU_HIDDEN;
    const size_t convp_count = (size_t)DFN3_MODEL_ENCODER_CHANNELS *
                               DFN3_MODEL_DF_PATHWAY_WINDOW * DFN3_DF_BINS;
    if (gru_hidden == NULL || df_convp_window == NULL) return -1;
    if (all_finite(&gru_hidden[0][0], gru_count) &&
        all_finite(&df_convp_window[0][0][0], convp_count)) return 0;
    /* Zero is the only state a stream can restart from after an in-place overwrite. */
    memset(gru_hidden, 0, gru_count * sizeof(float));
    memset(df_convp_window, 0, convp_count * sizeof(float));
    return -1;
}

int dfn3_model_io_push_features(DFN3ModelIOState *state,
                                const float erb[DFN3_N_ERB],
                                const float spec[2][DFN3_DF_BINS])
{
    if (state == NULL || erb == NULL || spec == NULL) return -1;
    dfn3_model_io_push_erb_window(state->erb_window, erb);
    dfn3_model_io_push_spec_window(state->spec_window, spec);
    if (state->feature_frames_seen < 2U) ++state->feature_frames_seen;
    return state->feature_frames_seen == 2U ? 1 : 0;
}

int dfn3_model_io_validate_state(DFN3ModelIOState *state)
{
    if (state == NULL) return -1;
    if (dfn3_model_io_validate_arrays(state->gru_hidden,
                                      state->df_convp_window) == 0)
        return 0;
    ++state->state_generation;
    return -1;
}

void dfn3_model_io_reset(DFN3ModelIOState *state)
{
    unsigned int generation;
    if (state == NULL) return;
    generation = state->state_generation + 1u;
    memset(state, 0, sizeof(*state));
    state->state_generation = generation;
}

void dfn3_model_io_zero_recurrent(DFN3ModelIOState *state)
{
    if (state == NULL) return;
    memset(state->gru_hidden, 0, sizeof(state->gru_hidden));
    memset(state->df_convp_window, 0, sizeof(state->df_convp_window));
    ++state->state_generation;
}

unsigned int dfn3_model_io_state_generation(const DFN3ModelIOState *state)
{
    return state == NULL ? 0u : state->state_generation;
}
