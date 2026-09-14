#include "dfn2_model_io.h"

#include <math.h>
#include <string.h>

void dfn2_model_io_init(DFN2ModelIOState *state)
{
    if (state != NULL) memset(state, 0, sizeof(*state));
}

#define DFN2_MODEL_IO_STATE_TENSORS 4

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
    X(df_pathway_history, DFN2_MODEL_DF_PATHWAY_HISTORY)              \
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
    size_t index;
    for (index = 0; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

int dfn2_model_io_commit_arrays(
    float encoder_gru_hidden[DFN2_MODEL_ENCODER_GRU_LAYERS]
                            [DFN2_MODEL_GRU_HIDDEN],
    float erb_gru_hidden[DFN2_MODEL_ERB_GRU_LAYERS][DFN2_MODEL_GRU_HIDDEN],
    float df_gru_hidden[DFN2_MODEL_DF_GRU_LAYERS][DFN2_MODEL_GRU_HIDDEN],
    float df_convp_history[DFN2_MODEL_ENCODER_CHANNELS]
                          [DFN2_MODEL_DF_PATHWAY_HISTORY]
                          [DFN2_DF_BINS],
    const float encoder_hidden_next[DFN2_MODEL_ENCODER_GRU_LAYERS]
                                   [DFN2_MODEL_GRU_HIDDEN],
    const float erb_hidden_next[DFN2_MODEL_ERB_GRU_LAYERS]
                               [DFN2_MODEL_GRU_HIDDEN],
    const float df_hidden_next[DFN2_MODEL_DF_GRU_LAYERS]
                              [DFN2_MODEL_GRU_HIDDEN],
    const float pathway_history_next[DFN2_MODEL_ENCODER_CHANNELS]
                                    [DFN2_MODEL_DF_PATHWAY_HISTORY]
                                    [DFN2_DF_BINS])
{
    if (encoder_gru_hidden == NULL || erb_gru_hidden == NULL ||
        df_gru_hidden == NULL || df_convp_history == NULL ||
        encoder_hidden_next == NULL || erb_hidden_next == NULL ||
        df_hidden_next == NULL || pathway_history_next == NULL) return -1;
    /* Validate the complete accelerator result before the first write. */
    if (!all_finite(&encoder_hidden_next[0][0],
                    DFN2_MODEL_ENCODER_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN) ||
        !all_finite(&erb_hidden_next[0][0],
                    DFN2_MODEL_ERB_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN) ||
        !all_finite(&df_hidden_next[0][0],
                    DFN2_MODEL_DF_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN) ||
        !all_finite(&pathway_history_next[0][0][0],
                    DFN2_MODEL_ENCODER_CHANNELS *
                    DFN2_MODEL_DF_PATHWAY_HISTORY * DFN2_DF_BINS)) return -1;
    memcpy(encoder_gru_hidden, encoder_hidden_next,
           DFN2_MODEL_ENCODER_GRU_LAYERS * sizeof(encoder_gru_hidden[0]));
    memcpy(erb_gru_hidden, erb_hidden_next,
           DFN2_MODEL_ERB_GRU_LAYERS * sizeof(erb_gru_hidden[0]));
    memcpy(df_gru_hidden, df_hidden_next,
           DFN2_MODEL_DF_GRU_LAYERS * sizeof(df_gru_hidden[0]));
    memcpy(df_convp_history, pathway_history_next,
           DFN2_MODEL_ENCODER_CHANNELS * sizeof(df_convp_history[0]));
    return 0;
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

int dfn2_model_io_commit_state(
    DFN2ModelIOState *state,
    const float encoder_hidden_next[DFN2_MODEL_ENCODER_GRU_LAYERS]
                                   [DFN2_MODEL_GRU_HIDDEN],
    const float erb_hidden_next[DFN2_MODEL_ERB_GRU_LAYERS]
                               [DFN2_MODEL_GRU_HIDDEN],
    const float df_hidden_next[DFN2_MODEL_DF_GRU_LAYERS]
                              [DFN2_MODEL_GRU_HIDDEN],
    const float pathway_history_next[DFN2_MODEL_ENCODER_CHANNELS]
                                    [DFN2_MODEL_DF_PATHWAY_HISTORY]
                                    [DFN2_DF_BINS])
{
    if (state == NULL) return -1;
    return dfn2_model_io_commit_arrays(
        state->encoder_gru_hidden, state->erb_gru_hidden,
        state->df_gru_hidden, state->df_convp_history,
        encoder_hidden_next, erb_hidden_next, df_hidden_next,
        pathway_history_next);
}
