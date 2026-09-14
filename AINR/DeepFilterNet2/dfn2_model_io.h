#ifndef DFN2_MODEL_IO_H
#define DFN2_MODEL_IO_H

#include "dfn2_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Must match export_onnx.py and the shipped DFN2 config. Version 3 exposed
 * one graph tensor per recurrent LAYER (h_erb_0/h_erb_1, h_df_0/h_df_1) so
 * PTQ could scale each layer independently; version 5 exposes one per GRU in
 * its native stacked shape (h_erb, h_df, each (layers, 1, hidden)), which is
 * exactly how the arrays below are already laid out, so a runtime binds each
 * tensor to one contiguous field. The graph still holds one GRU node per
 * layer -- ONNX has no stacked GRU op -- so the layers of one stack now
 * share that stack's quantization scale.
 *
 * ⚠ Versions 4, 6 and 7 are TAKEN, not free -- see export_onnx.py's
 * RETIRED_LAYOUT_VERSIONS and COMBINED_STATE_LAYOUT_VERSION, which
 * test_dfn2_contract.py asserts this constant against. Nothing here binds the
 * combined layout, so a board built against this header refuses such a graph:
 * that is the intent, not an oversight. */
#define DFN2_MODEL_IO_LAYOUT_VERSION       5
#define DFN2_MODEL_INPUT_FRAMES            3
#define DFN2_MODEL_ENCODER_GRU_LAYERS      1
#define DFN2_MODEL_ERB_GRU_LAYERS          2
#define DFN2_MODEL_DF_GRU_LAYERS           2
#define DFN2_MODEL_GRU_HIDDEN               256
#define DFN2_MODEL_ENCODER_CHANNELS         64
#define DFN2_MODEL_DF_PATHWAY_HISTORY       4

/* The graph contract a runtime was built against, as a value a host can
 * compare field by field. Every field is a compiled constant of this build;
 * there is no free deployment parameter here (unlike Align-ULCNet's
 * delay_depth), so validate(default()) == 0 by construction and the struct's
 * only job is refusing a graph exported against another layout version or
 * geometry -- a mismatch the NaN-prefill and finite checks cannot see,
 * because they catch an unwritten output, never a wrong-shaped one. */
typedef struct DFN2ModelIoDescriptor {
    unsigned int layout_version;   /* DFN2_MODEL_IO_LAYOUT_VERSION         */
    int sample_rate;               /* DFN2_SR                              */
    int fft_size;                  /* DFN2_N_FFT                           */
    int hop_size;                  /* DFN2_HOP_LEN                         */
    int spectrum_bins;             /* DFN2_N_BINS                          */
    int erb_bands;                 /* DFN2_N_ERB                           */
    int df_bins;                   /* DFN2_DF_BINS                         */
    int df_order;                  /* DFN2_DF_ORDER                        */
    int mask_lookahead;            /* DFN2_MASK_LOOKAHEAD                  */
    int df_lookahead;              /* DFN2_DF_LOOKAHEAD                    */
    int input_frames;              /* DFN2_MODEL_INPUT_FRAMES              */
    int encoder_gru_layers;        /* DFN2_MODEL_ENCODER_GRU_LAYERS        */
    int erb_gru_layers;            /* DFN2_MODEL_ERB_GRU_LAYERS            */
    int df_gru_layers;             /* DFN2_MODEL_DF_GRU_LAYERS             */
    int gru_hidden;                /* DFN2_MODEL_GRU_HIDDEN                */
    int encoder_channels;          /* DFN2_MODEL_ENCODER_CHANNELS          */
    int df_pathway_history;        /* DFN2_MODEL_DF_PATHWAY_HISTORY        */
    int state_tensor_count;        /* 4: h_encoder, h_erb, h_df, df_convp  */
} DFN2ModelIoDescriptor;

/* Fill with this build's contract. Returns 0, or -1 on NULL. */
int dfn2_model_io_descriptor_default(DFN2ModelIoDescriptor *descriptor);
/* 0 when every field equals this build's contract, -1 otherwise (NULL
 * included). Field by field, never memcmp, so padding cannot decide. */
int dfn2_model_io_descriptor_validate(
    const DFN2ModelIoDescriptor *descriptor);

/* Caller-owned state for a stateless accelerator. The graph receives these
 * arrays as ordinary inputs and returns their *_next values as outputs. */
typedef struct {
    float erb_window[DFN2_MODEL_INPUT_FRAMES][DFN2_N_ERB];
    float spec_window[2][DFN2_MODEL_INPUT_FRAMES][DFN2_DF_BINS];
    float encoder_gru_hidden[DFN2_MODEL_ENCODER_GRU_LAYERS]
                            [DFN2_MODEL_GRU_HIDDEN];
    float erb_gru_hidden[DFN2_MODEL_ERB_GRU_LAYERS]
                        [DFN2_MODEL_GRU_HIDDEN];
    float df_gru_hidden[DFN2_MODEL_DF_GRU_LAYERS]
                       [DFN2_MODEL_GRU_HIDDEN];
    float df_convp_history[DFN2_MODEL_ENCODER_CHANNELS]
                           [DFN2_MODEL_DF_PATHWAY_HISTORY]
                           [DFN2_DF_BINS];
    unsigned long long feature_frames_seen;
} DFN2ModelIOState;

void dfn2_model_io_init(DFN2ModelIOState *state);

/* Slide a [t-1,t,t+1] feature window by one frame and write the newest frame
 * into the last slot. Exported rather than inlined so an external dual-input
 * consumer keeping its own windows reuses this one: a second copy of the
 * memmove/memcpy pair is a place for the window order to drift silently,
 * which no shape check would catch. */
void dfn2_model_io_push_erb_window(
    float window[DFN2_MODEL_INPUT_FRAMES][DFN2_N_ERB],
    const float frame[DFN2_N_ERB]);
void dfn2_model_io_push_spec_window(
    float window[2][DFN2_MODEL_INPUT_FRAMES][DFN2_DF_BINS],
    const float frame[2][DFN2_DF_BINS]);

/* Commit the explicit-state graph outputs into caller-owned arrays.
 * Takes the arrays instead of a state struct so an external dual-input
 * consumer whose struct differs reuses this commit discipline without
 * changing its own field layout. */
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
                                    [DFN2_DF_BINS]);

/* Push one newly computed feature frame. Returns 1 when [t-1,t,t+1] is
 * available and the graph may emit heads[t], or 0 after the first frame.
 * To flush the final source frame, push one all-zero feature frame. */
int dfn2_model_io_push_features(DFN2ModelIOState *state,
                                const float erb[DFN2_N_ERB],
                                const float spec[2][DFN2_DF_BINS]);

/* Commit all explicit-state graph outputs after a successful invocation.
 * Returns 0 on success. NULL or non-finite output returns -1 and leaves every
 * previously committed state array untouched. */
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
                                    [DFN2_DF_BINS]);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_MODEL_IO_H */
