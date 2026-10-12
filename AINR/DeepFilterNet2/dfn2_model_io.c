#include "dfn2_model_io.h"

#include <math.h>
#include <string.h>

#include "simd_kernel_nn.h"

void dfn2_model_io_init(DFN2ModelIOState *state)
{
    if (state == NULL) return;
    memset(state, 0, sizeof(*state));
    state->state_generation = 1u;
}

#define DFN2_MODEL_IO_STATE_TENSORS 2

/* One list of (field, contract value) drives both the default and the
 * validator, so validate(default()) == 0 holds by construction and a new
 * geometry constant cannot be added to one without the other. */
#define DFN2_MODEL_IO_DESCRIPTOR_FIELDS(X)                            \
    X(layout_version, (unsigned int)DFN2_MODEL_IO_LAYOUT_VERSION)     \
    X(sample_rate, DFN2_SR)                                           \
    X(fft_size, DFN2_N_FFT)                                           \
    X(hop_size, DFN2_HOP_LEN)                                         \
    X(spectrum_bins, DFN2_N_BINS)                                     \
    X(erb_bands, DFN2_N_ERB)                                          \
    X(df_bins, DFN2_DF_BINS)                                          \
    X(df_order, DFN2_DF_ORDER)                                        \
    X(mask_lookahead, DFN2_MASK_LOOKAHEAD)                            \
    X(df_lookahead, DFN2_DF_LOOKAHEAD)                                \
    X(input_frames, DFN2_MODEL_INPUT_FRAMES)                          \
    X(encoder_gru_layers, DFN2_MODEL_ENCODER_GRU_LAYERS)              \
    X(erb_gru_layers, DFN2_MODEL_ERB_GRU_LAYERS)                      \
    X(df_gru_layers, DFN2_MODEL_DF_GRU_LAYERS)                        \
    X(gru_hidden, DFN2_MODEL_GRU_HIDDEN)                              \
    X(encoder_channels, DFN2_MODEL_ENCODER_CHANNELS)                  \
    X(df_pathway_window, DFN2_MODEL_DF_PATHWAY_WINDOW)              \
    X(state_tensor_count, DFN2_MODEL_IO_STATE_TENSORS)

int dfn2_model_io_descriptor_default(DFN2ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
#define DFN2_SET_FIELD(field, value) descriptor->field = (value);
    DFN2_MODEL_IO_DESCRIPTOR_FIELDS(DFN2_SET_FIELD)
#undef DFN2_SET_FIELD
    return 0;
}

int dfn2_model_io_descriptor_validate(
    const DFN2ModelIoDescriptor *descriptor)
{
    if (descriptor == NULL) return -1;
#define DFN2_CHECK_FIELD(field, value) \
    if (descriptor->field != (value)) return -1;
    DFN2_MODEL_IO_DESCRIPTOR_FIELDS(DFN2_CHECK_FIELD)
#undef DFN2_CHECK_FIELD
    return 0;
}

void dfn2_model_io_push_erb_window(
    float window[DFN2_MODEL_INPUT_FRAMES][DFN2_N_ERB],
    const float frame[DFN2_N_ERB])
{
    if (window == NULL || frame == NULL) return;
    memmove(window[0], window[1],
            (DFN2_MODEL_INPUT_FRAMES - 1) * sizeof(window[0]));
    memcpy(window[DFN2_MODEL_INPUT_FRAMES - 1], frame, sizeof(window[0]));
}

void dfn2_model_io_push_spec_window(
    float window[2][DFN2_MODEL_INPUT_FRAMES][DFN2_DF_BINS],
    const float frame[2][DFN2_DF_BINS])
{
    if (window == NULL || frame == NULL) return;
    for (int channel = 0; channel < 2; ++channel) {
        memmove(window[channel][0], window[channel][1],
                (DFN2_MODEL_INPUT_FRAMES - 1) *
                    sizeof(window[channel][0]));
        memcpy(window[channel][DFN2_MODEL_INPUT_FRAMES - 1],
               frame[channel], sizeof(window[channel][0]));
    }
}

static int all_finite(const float *values, size_t count)
{
    return skn_all_finite_f32(values, count);
}

int dfn2_model_io_validate_arrays(
    float gru_hidden[DFN2_MODEL_GRU_ROWS][DFN2_MODEL_GRU_HIDDEN],
    float df_convp_window[DFN2_MODEL_ENCODER_CHANNELS]
                         [DFN2_MODEL_DF_PATHWAY_WINDOW]
                         [DFN2_DF_BINS])
{
    const size_t gru_count = (size_t)DFN2_MODEL_GRU_ROWS *
                             DFN2_MODEL_GRU_HIDDEN;
    const size_t convp_count = (size_t)DFN2_MODEL_ENCODER_CHANNELS *
                               DFN2_MODEL_DF_PATHWAY_WINDOW * DFN2_DF_BINS;
    if (gru_hidden == NULL || df_convp_window == NULL) return -1;
    if (all_finite(&gru_hidden[0][0], gru_count) &&
        all_finite(&df_convp_window[0][0][0], convp_count)) return 0;
    /* The accelerator wrote these arrays in place, so the previous values are
     * gone: zero is the only state a stream can restart from. */
    memset(gru_hidden, 0, gru_count * sizeof(float));
    memset(df_convp_window, 0, convp_count * sizeof(float));
    return -1;
}

int dfn2_model_io_push_features(DFN2ModelIOState *state,
                                const float erb[DFN2_N_ERB],
                                const float spec[2][DFN2_DF_BINS])
{
    if (state == NULL || erb == NULL || spec == NULL) return -1;
    dfn2_model_io_push_erb_window(state->erb_window, erb);
    dfn2_model_io_push_spec_window(state->spec_window, spec);
    if (state->feature_frames_seen < 2U) ++state->feature_frames_seen;
    return state->feature_frames_seen == 2U ? 1 : 0;
}

int dfn2_model_io_validate_state(DFN2ModelIOState *state)
{
    if (state == NULL) return -1;
    if (dfn2_model_io_validate_arrays(state->gru_hidden,
                                      state->df_convp_window) == 0)
        return 0;
    ++state->state_generation;
    return -1;
}

void dfn2_model_io_reset(DFN2ModelIOState *state)
{
    unsigned int generation;
    if (state == NULL) return;
    generation = state->state_generation + 1u;
    memset(state, 0, sizeof(*state));
    state->state_generation = generation;
}

void dfn2_model_io_zero_recurrent(DFN2ModelIOState *state)
{
    if (state == NULL) return;
    memset(state->gru_hidden, 0, sizeof(state->gru_hidden));
    memset(state->df_convp_window, 0, sizeof(state->df_convp_window));
    ++state->state_generation;
}

unsigned int dfn2_model_io_state_generation(const DFN2ModelIOState *state)
{
    return state == NULL ? 0u : state->state_generation;
}
