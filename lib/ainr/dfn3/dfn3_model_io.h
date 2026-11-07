#ifndef DFN3_MODEL_IO_H
#define DFN3_MODEL_IO_H

#include "dfn3_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Graph contract; must match export_onnx.py. State tensors, each *_out output at the shape and address of its
 * input: h_gru [1, rows, 1, hidden], df_convp_history [1, channels, window, df_bins]. The application owns them and
 * the library knows none of rows, hidden, channels or window: on a board they live in the accelerator's buffers; a
 * float host registers them in dfn3_model_io_init() and the library zeroes and finite-checks them. */
#define DFN3_MODEL_IO_LAYOUT_VERSION  9   /* versions 4-8 are never reused */
/* Frames of the graph's input window; for mask lookahead L the window of heads[t] is [t-(2-L) .. t+L]. */
#define DFN3_MODEL_INPUT_FRAMES       3

/* A graph's external contract, compared field by field against the DFN3Grid the instance runs. The six interior
 * fields are the adapter's own: 0 from _default(), never validated. */
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
    int input_frames;              /* DFN3_MODEL_INPUT_FRAMES */
    int encoder_gru_layers;        /* interior, caller-filled */
    int erb_gru_layers;
    int df_gru_layers;
    int gru_hidden;
    int encoder_channels;
    int df_pathway_window;
    int state_tensor_count;        /* 2: h_gru, df_convp_history */
} DFN3ModelIoDescriptor;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DFN3ModelIoDescriptor) == 72,
               "DFN3ModelIoDescriptor must stay 72 bytes");
#endif
#endif

/* Fill the external fields for `grid`, zero the interior ones. Returns 0, or -1 on NULL or an invalid grid. */
int dfn3_model_io_descriptor_default(const DFN3Grid *grid,
                                     DFN3ModelIoDescriptor *descriptor);
/* 0 when every external field equals the contract for `grid`, -1 otherwise. */
int dfn3_model_io_descriptor_validate(const DFN3Grid *grid,
                                      const DFN3ModelIoDescriptor *descriptor);

/* Feature windows (carved by dfn3_model_io_init()) and the registered recurrent state, which the graph updates in
 * place (read every state input before writing any state output). state_generation advances whenever the recurrent
 * state is zeroed, registered or not; a fixed-point copy loads quantized zeros at start and on every change. */
typedef struct DFN3ModelIOState {
    float *erb_window;              /* packed [3][n_erb]; read-only to the accelerator */
    float *spec_window;             /* packed [2][3][df_bins]; read-only to the accelerator */
    float *gru_hidden;              /* registered state, borrowed; NULL and 0 when nothing is registered */
    size_t gru_hidden_elements;
    float *df_convp_window;
    size_t df_convp_window_elements;
    void  *block;                   /* the block the windows live in, for reset */
    size_t block_bytes;
    unsigned long long feature_frames_seen;
    unsigned int state_generation;
    int n_erb;                      /* from the grid; 0 before init */
    int df_bins;
    int mask_lookahead;
} DFN3ModelIOState;

/* Bytes of the block dfn3_model_io_init() carves, a multiple of 16; 0 for a NULL or invalid grid. */
size_t dfn3_model_io_mem_size(const DFN3Grid *grid);

/* Carve the windows out of `block` (16-byte aligned, `bytes` >= dfn3_model_io_mem_size(grid)), zero them and start
 * the generation at 1. Registration is two (pointer, element count) pairs, borrowed for the instance's life: both
 * set, or all four values NULL/0; a pointer with a count of 0 or a count with a NULL pointer is refused.
 * Registered buffers are zeroed. Returns 0, or -1 on NULL, an invalid grid or registration, or a short or misaligned
 * block (the header is then zeroed; the block and the buffers are untouched). */
int dfn3_model_io_init(DFN3ModelIOState *state, void *block, size_t bytes,
                       const DFN3Grid *grid, float *gru_hidden,
                       size_t gru_hidden_elements, float *df_convp_history,
                       size_t df_convp_history_elements);

/* mask_lookahead + 1: the feature frames the first accelerator call needs. 0 for NULL or no grid. */
unsigned int dfn3_model_io_ready_frames(const DFN3ModelIOState *state);

/* Slide a window by one frame and append the newest: after frame n it holds [n-2, n-1, n], earlier slots zero as in
 * training. `window` is the packed [3][n_erb] run, or [2][3][df_bins] with frame[2][df_bins]. */
void dfn3_model_io_push_erb_window(float *window, const float *frame, int n_erb);
void dfn3_model_io_push_spec_window(float *window, const float *frame, int df_bins);

/* Check the recurrent arrays the accelerator wrote in place (element counts): 0 when all are finite, else both arrays
 * are zeroed and -1 is returned. A NULL array or a count of 0 returns -1 and touches nothing. */
int dfn3_model_io_validate_arrays(float *gru_hidden, size_t gru_count,
                                  float *df_convp_history, size_t convp_count);

/* Push one feature frame: erb[n_erb] and spec[2][df_bins] (re, then im). Returns 1 when the window of
 * heads[t] is complete and the graph may emit it, 0 for the first mask_lookahead frames, -1 on NULL or no grid.
 * To flush the final heads, push mask_lookahead all-zero frames. */
int dfn3_model_io_push_features(DFN3ModelIOState *state, const float *erb,
                                const float *spec);

/* validate_arrays() over the registered state; call after each invocation. Returns 0 with nothing registered, -1
 * on NULL or non-finite (the registered arrays zeroed, generation advances). The windows are untouched. */
int dfn3_model_io_validate_state(DFN3ModelIOState *state);

/* Windows, registered state, frame count back to init values; generation advances; pointers stay. NULL: no-op. */
void dfn3_model_io_reset(DFN3ModelIOState *state);

/* Zero the registered recurrent arrays only; the generation advances. NULL: no-op. */
void dfn3_model_io_zero_recurrent(DFN3ModelIOState *state);

/* 1 after init, plus one per reset, refused validation and zero_recurrent, modulo 2^32; 0 for NULL. */
unsigned int dfn3_model_io_state_generation(const DFN3ModelIOState *state);

#ifdef __cplusplus
}
#endif

#endif /* DFN3_MODEL_IO_H */
