#ifndef DFN2_MODEL_IO_H
#define DFN2_MODEL_IO_H

#include "dfn2_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Must match export_onnx.py and the shipped DFN2 config.
 *
 * Graph state tensors (all rank 4, input and *_out output of one tensor have
 * the same shape and are bound to the same address):
 *   h_gru             [1, DFN2_MODEL_GRU_ROWS, 1, DFN2_MODEL_GRU_HIDDEN]
 *                     rows: encoder GRU, ERB decoder GRU (2 layers), DF
 *                     decoder GRU (2 layers), see the DFN2_MODEL_GRU_ROW_*
 *                     offsets
 *   df_convp_history  [1, DFN2_MODEL_ENCODER_CHANNELS,
 *                      DFN2_MODEL_DF_PATHWAY_WINDOW, DFN2_DF_BINS]
 *                     the full causal window of the pathway convolution: the
 *                     graph drops the oldest frame, appends the new one and
 *                     returns the whole window
 *
 * Layout versions 4, 5, 6 and 7 are retired (RETIRED_LAYOUT_VERSIONS in
 * export_onnx.py); test_dfn2_contract.py asserts this constant against it. */
#define DFN2_MODEL_IO_LAYOUT_VERSION       8
#define DFN2_MODEL_INPUT_FRAMES            3
#define DFN2_MODEL_ENCODER_GRU_LAYERS      1
#define DFN2_MODEL_ERB_GRU_LAYERS          2
#define DFN2_MODEL_DF_GRU_LAYERS           2
#define DFN2_MODEL_GRU_ROWS                5   /* 1 + 2 + 2                */
#define DFN2_MODEL_GRU_ROW_ENCODER         0
#define DFN2_MODEL_GRU_ROW_ERB             1
#define DFN2_MODEL_GRU_ROW_DF              3
#define DFN2_MODEL_GRU_HIDDEN               256
#define DFN2_MODEL_ENCODER_CHANNELS         64
#define DFN2_MODEL_DF_PATHWAY_WINDOW        5

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN2_MODEL_GRU_ROWS == DFN2_MODEL_ENCODER_GRU_LAYERS +
                   DFN2_MODEL_ERB_GRU_LAYERS + DFN2_MODEL_DF_GRU_LAYERS,
               "h_gru rows are the sum of the three GRU stacks");
_Static_assert(DFN2_MODEL_GRU_ROW_ERB == DFN2_MODEL_GRU_ROW_ENCODER +
                   DFN2_MODEL_ENCODER_GRU_LAYERS &&
               DFN2_MODEL_GRU_ROW_DF == DFN2_MODEL_GRU_ROW_ERB +
                   DFN2_MODEL_ERB_GRU_LAYERS,
               "row offsets follow the stack order");
#endif
#endif

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
    int df_pathway_window;         /* DFN2_MODEL_DF_PATHWAY_WINDOW         */
    int state_tensor_count;        /* 2: h_gru, df_convp_history           */
} DFN2ModelIoDescriptor;

/* Fill with this build's contract. Returns 0, or -1 on NULL. */
int dfn2_model_io_descriptor_default(DFN2ModelIoDescriptor *descriptor);
/* 0 when every field equals this build's contract, -1 otherwise (NULL
 * included). Field by field, never memcmp, so padding cannot decide. */
int dfn2_model_io_descriptor_validate(
    const DFN2ModelIoDescriptor *descriptor);

/* Caller-owned state for a stateless accelerator. The graph reads the two
 * state arrays as inputs and writes the updated ones to outputs bound to the
 * SAME addresses, so the output of one invocation is the input of the next
 * with no copy. The runtime must read every state input before it writes any
 * state output.
 *
 * The two feature windows and feature_frames_seen are CPU-owned: the
 * accelerator reads the windows and never writes them.
 *
 * state_generation advances each time this library zeroes the recurrent
 * arrays (reset, or a refused validation). The accelerator's fixed-point copy of the
 * state is not visible here, so a runtime refills it with quantized zeros
 * whenever dfn2_model_io_state_generation() differs from the value it saw
 * last. */
typedef struct {
    float erb_window[DFN2_MODEL_INPUT_FRAMES][DFN2_N_ERB];
    float spec_window[2][DFN2_MODEL_INPUT_FRAMES][DFN2_DF_BINS];
    float gru_hidden[DFN2_MODEL_GRU_ROWS][DFN2_MODEL_GRU_HIDDEN];
    float df_convp_window[DFN2_MODEL_ENCODER_CHANNELS]
                         [DFN2_MODEL_DF_PATHWAY_WINDOW]
                         [DFN2_DF_BINS];
    unsigned long long feature_frames_seen;
    unsigned int state_generation;
} DFN2ModelIOState;

/* First construction: zero the whole object and start the generation at 1.
 * Not a reset -- a state that is already in use goes through
 * dfn2_model_io_reset() so its generation keeps advancing. The caller fills
 * the accelerator's state with zeros once after this. */
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

/* Validate the two recurrent arrays after the accelerator wrote them in
 * place. Takes the arrays instead of a state struct so an external dual-input
 * consumer whose struct differs reuses this discipline without changing its
 * own field layout.
 *
 * Returns 0 when every element is finite. On any non-finite element it zeroes
 * both arrays and returns -1: the previous values were overwritten by the
 * accelerator and cannot be restored, so a stream restarts from zero state.
 * NULL returns -1 and touches nothing. A partial or mixed write that is still
 * finite is not detectable here. */
int dfn2_model_io_validate_arrays(
    float gru_hidden[DFN2_MODEL_GRU_ROWS][DFN2_MODEL_GRU_HIDDEN],
    float df_convp_window[DFN2_MODEL_ENCODER_CHANNELS]
                         [DFN2_MODEL_DF_PATHWAY_WINDOW]
                         [DFN2_DF_BINS]);

/* Push one newly computed feature frame. Returns 1 when [t-1,t,t+1] is
 * available and the graph may emit heads[t], or 0 after the first frame.
 * To flush the final source frame, push one all-zero feature frame. */
int dfn2_model_io_push_features(DFN2ModelIOState *state,
                                const float erb[DFN2_N_ERB],
                                const float spec[2][DFN2_DF_BINS]);

/* validate_arrays() over the state's own recurrent arrays; call it after
 * each invocation. Returns 0 when they are all finite; NULL or a non-finite
 * element returns -1, and in the non-finite case the arrays are zeroed and the
 * generation advances. The feature windows and feature_frames_seen are never
 * touched. */
int dfn2_model_io_validate_state(DFN2ModelIOState *state);

/* Back to the init values (windows, recurrent arrays, frame count); the
 * generation advances. NULL is ignored. */
void dfn2_model_io_reset(DFN2ModelIOState *state);

/* Zero the two recurrent arrays only; the generation advances. NULL is
 * ignored. */
void dfn2_model_io_zero_recurrent(DFN2ModelIOState *state);

/* The generation: 1 after init, plus one for each reset, refused validation
 * and zero_recurrent, modulo 2^32. 0 for NULL. */
unsigned int dfn2_model_io_state_generation(const DFN2ModelIOState *state);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_MODEL_IO_H */
