#ifndef DFN3_MODEL_IO_H
#define DFN3_MODEL_IO_H

#include "dfn3_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Must match export_onnx.py. Graph state tensors (rank 4; each *_out output has
 * the shape and address of its input):
 *   h_gru             [1, DFN3_MODEL_GRU_ROWS, 1, DFN3_MODEL_GRU_HIDDEN]; rows:
 *                     encoder GRU, ERB decoder GRU (2), DF decoder GRU (2), at
 *                     the DFN3_MODEL_GRU_ROW_* offsets
 *   df_convp_history  [1, DFN3_MODEL_ENCODER_CHANNELS,
 *                      DFN3_MODEL_DF_PATHWAY_WINDOW, DFN3_DF_BINS]; the full
 *                     causal window of the pathway convolution, returned whole
 * Layout versions 4-8 are never reused. */
#define DFN3_MODEL_IO_LAYOUT_VERSION       9
#define DFN3_MODEL_INPUT_FRAMES            3
#define DFN3_MODEL_ENCODER_GRU_LAYERS      1
#define DFN3_MODEL_ERB_GRU_LAYERS          2
#define DFN3_MODEL_DF_GRU_LAYERS           2
#define DFN3_MODEL_GRU_ROWS                5   /* 1 + 2 + 2                */
#define DFN3_MODEL_GRU_ROW_ENCODER         0
#define DFN3_MODEL_GRU_ROW_ERB             1
#define DFN3_MODEL_GRU_ROW_DF              3
#define DFN3_MODEL_GRU_HIDDEN               256
#define DFN3_MODEL_ENCODER_CHANNELS         64
#define DFN3_MODEL_DF_PATHWAY_WINDOW        5
/* The exported graph consumes a full three-frame window whatever the mask
 * lookahead; the lookahead only decides which frames the host puts in it
 * ([t-(2-MASK), ..., t+MASK]) and when the first one is complete: the window for
 * heads[t] ends at frame t+MASK, so heads[0] needs MASK + 1 pushed frames. */
#define DFN3_MODEL_READY_FRAMES             (DFN3_MASK_LOOKAHEAD + 1U)

#if DFN3_MODEL_READY_FRAMES > DFN3_MODEL_INPUT_FRAMES
#error "the first window must fit in the three-frame model input"
#endif

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN3_MODEL_GRU_ROWS == DFN3_MODEL_ENCODER_GRU_LAYERS +
                   DFN3_MODEL_ERB_GRU_LAYERS + DFN3_MODEL_DF_GRU_LAYERS,
               "h_gru rows are the sum of the three GRU stacks");
_Static_assert(DFN3_MODEL_GRU_ROW_ERB == DFN3_MODEL_GRU_ROW_ENCODER +
                   DFN3_MODEL_ENCODER_GRU_LAYERS &&
               DFN3_MODEL_GRU_ROW_DF == DFN3_MODEL_GRU_ROW_ERB +
                   DFN3_MODEL_ERB_GRU_LAYERS,
               "row offsets follow the stack order");
#endif
#endif

/* The graph contract a runtime was built against, compared field by field. Every
 * field is a compiled constant, so a mismatch means a graph exported against
 * another layout version or geometry. */
typedef struct DFN3ModelIoDescriptor {
    unsigned int layout_version;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int erb_bands;
    int df_bins;
    int df_order;
    int mask_lookahead;
    int df_lookahead;
    int input_frames;
    int encoder_gru_layers;
    int erb_gru_layers;
    int df_gru_layers;
    int gru_hidden;
    int encoder_channels;
    int df_pathway_window;
    int state_tensor_count;        /* 2: h_gru, df_convp_history           */
} DFN3ModelIoDescriptor;

/* Fill with this build's contract. Returns 0, or -1 on NULL. */
int dfn3_model_io_descriptor_default(DFN3ModelIoDescriptor *descriptor);
/* 0 when every field equals this build's contract, -1 otherwise (NULL included). */
int dfn3_model_io_descriptor_validate(
    const DFN3ModelIoDescriptor *descriptor);

/* Caller-owned state for a stateless accelerator. The graph reads the two state
 * arrays and writes the updated ones to outputs bound to the same addresses
 * (read all state inputs before writing any state output). The feature windows
 * and feature_frames_seen are CPU-owned and read-only to the accelerator.
 * state_generation advances whenever the recurrent arrays are zeroed (reset or
 * refused validation); a runtime holding a fixed-point copy of the state refills
 * it with quantized zeros when dfn3_model_io_state_generation() changes. */
typedef struct {
    float erb_window[DFN3_MODEL_INPUT_FRAMES][DFN3_N_ERB];
    float spec_window[2][DFN3_MODEL_INPUT_FRAMES][DFN3_DF_BINS];
    float gru_hidden[DFN3_MODEL_GRU_ROWS][DFN3_MODEL_GRU_HIDDEN];
    float df_convp_window[DFN3_MODEL_ENCODER_CHANNELS]
                         [DFN3_MODEL_DF_PATHWAY_WINDOW]
                         [DFN3_DF_BINS];
    unsigned long long feature_frames_seen;
    unsigned int state_generation;
} DFN3ModelIOState;

/* Slide the feature window by one frame and write the newest frame into the
 * last slot. After frame n is pushed the window holds [n-2, n-1, n], which is
 * the window of heads[n-MASK_LOOKAHEAD] -- [t-1,t,t+1] for the shipped
 * mask_lookahead 1; frames before the first stay zero, as in training. */
void dfn3_model_io_push_erb_window(
    float window[DFN3_MODEL_INPUT_FRAMES][DFN3_N_ERB],
    const float frame[DFN3_N_ERB]);
void dfn3_model_io_push_spec_window(
    float window[2][DFN3_MODEL_INPUT_FRAMES][DFN3_DF_BINS],
    const float frame[2][DFN3_DF_BINS]);

/* Validate the two recurrent arrays after the accelerator wrote them in place.
 * Returns 0 when all elements are finite; otherwise zeroes both arrays and
 * returns -1 (the previous values cannot be restored). NULL returns -1 and
 * touches nothing. */
int dfn3_model_io_validate_arrays(
    float gru_hidden[DFN3_MODEL_GRU_ROWS][DFN3_MODEL_GRU_HIDDEN],
    float df_convp_window[DFN3_MODEL_ENCODER_CHANNELS]
                         [DFN3_MODEL_DF_PATHWAY_WINDOW]
                         [DFN3_DF_BINS]);

/* Push one newly computed feature frame. Returns 1 when the window of
 * heads[t] is available and the graph may emit it, or 0 while fewer than
 * DFN3_MODEL_READY_FRAMES frames were pushed (no run: the first
 * DFN3_MASK_LOOKAHEAD frames). To flush the final source frames, push
 * DFN3_MASK_LOOKAHEAD all-zero feature frames. */
int dfn3_model_io_push_features(DFN3ModelIOState *state,
                                const float erb[DFN3_N_ERB],
                                const float spec[2][DFN3_DF_BINS]);

/* validate_arrays() over the state's own recurrent arrays; call after each
 * invocation. Returns -1 on NULL or non-finite (arrays zeroed, generation
 * advances). The feature windows are untouched. */
int dfn3_model_io_validate_state(DFN3ModelIOState *state);

/* Back to the init values (windows, recurrent arrays, frame count); the
 * generation advances. NULL is ignored. */
void dfn3_model_io_reset(DFN3ModelIOState *state);

/* Zero the two recurrent arrays only; the generation advances. NULL is
 * ignored. */
void dfn3_model_io_zero_recurrent(DFN3ModelIOState *state);

/* The generation: 1 after init, plus one for each reset, refused validation
 * and zero_recurrent, modulo 2^32. 0 for NULL. */
unsigned int dfn3_model_io_state_generation(const DFN3ModelIOState *state);

#ifdef __cplusplus
}
#endif

#endif /* DFN3_MODEL_IO_H */
