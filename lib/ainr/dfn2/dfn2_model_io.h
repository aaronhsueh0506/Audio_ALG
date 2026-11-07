#ifndef DFN2_MODEL_IO_H
#define DFN2_MODEL_IO_H

#include "dfn2_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layout version of the graph's state tensors (h_gru and, in the full variant, df_convp_history); it does
 * not change with the frames per call, and 4-7 are never reused. -DDFN2_MASK_ONLY (every translation unit)
 * has no df_convp_history and its own version, so a graph of the other kind is refused. */
#define DFN2_MODEL_IO_LAYOUT_VERSION_FULL        8
#define DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY   9
#ifndef DFN2_MASK_ONLY
#define DFN2_MODEL_IO_LAYOUT_VERSION       DFN2_MODEL_IO_LAYOUT_VERSION_FULL
#else
#define DFN2_MODEL_IO_LAYOUT_VERSION       DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY
#endif

/* Heads per call: N = frames_per_call, L = mask_lookahead. The graph reads N + 2 feature frames and returns
 * N heads plus the state after the last; N > 1 means readiness every N hops, N heads per commit and N - 1
 * more hops of latency. A call's window is frames [t-(2-L) .. t+N-1+L] for heads[t .. t+N-1] (head i from
 * the three frames starting at slot i); frames before the stream start read as zero. */

/* A graph's external contract, compared field by field. The six interior fields (GRU layers and width, encoder
 * channels, pathway window) are caller bookkeeping: 0 by default, never read or validated here. */
typedef struct DFN2ModelIoDescriptor {
    unsigned int layout_version;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int erb_bands;
    int df_bins;
    int df_order;                  /* the grid's (0 when mask-only)          */
    int mask_lookahead;
    int df_lookahead;
    int input_frames;              /* N + 2 (metadata input_feature_frames)  */
    int encoder_gru_layers;
    int erb_gru_layers;
    int df_gru_layers;
    int gru_hidden;
    int encoder_channels;
    int df_pathway_window;
    int state_tensor_count;        /* 2: h_gru, df_convp_history; 1 with DFN2_MASK_ONLY */
} DFN2ModelIoDescriptor;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DFN2ModelIoDescriptor) == 72,
               "DFN2ModelIoDescriptor must stay 72 bytes");
#endif
#endif

/* Fill the external fields for `grid`, zero the interior ones; -1 on NULL or an invalid grid. */
int dfn2_model_io_descriptor_default(const DFN2Grid *grid,
                                     DFN2ModelIoDescriptor *descriptor);
/* 0 if every external field equals the contract for `grid`, -1 otherwise (NULL, invalid grid). */
int dfn2_model_io_descriptor_validate(
    const DFN2Grid *grid, const DFN2ModelIoDescriptor *descriptor);

/* Feature windows (CPU-owned, read-only to the accelerator) carved by init, plus the optional registration of
 * the application's recurrent state: borrowed, never allocated here; each *_out output aliases its input (read
 * all state inputs before writing any state output). state_generation advances whenever the state is zeroed,
 * registered or not; a fixed-point runtime copy loads quantized zeros at start and on every change. */
typedef struct DFN2ModelIOState {
    float *erb_window;     /* [input_frames][n_erb] */
    float *spec_window;    /* [2][input_frames][df_bins] */
    float *gru_hidden;     /* registered recurrent state: NULL and 0 when nothing is registered */
    size_t gru_hidden_elements;
#ifndef DFN2_MASK_ONLY
    float *df_convp_window;
    size_t df_convp_window_elements;
#endif
    void  *block;          /* the carved block, kept for dfn2_model_io_reset() */
    size_t block_bytes;
    unsigned long long feature_frames_seen;
    unsigned int state_generation;
    int n_erb;             /* this and the next four: from the grid given to init; 0 before it */
    int df_bins;
    int mask_lookahead;
    int frames_per_call;
    int input_frames;
} DFN2ModelIOState;

size_t dfn2_model_io_mem_size(const DFN2Grid *grid);   /* init's block bytes, a multiple of 16; 0 if invalid */

/* Carve the windows from `block` (16-byte aligned, `bytes` >= dfn2_model_io_mem_size(grid)), zero them and
 * register the recurrent state: borrowed (pointer, count) pairs for h_gru and df_convp_history, all set or
 * all NULL/0 (refused: a pointer with count 0, a count without a pointer, h_gru alone in the full variant,
 * any df_convp_history with DFN2_MASK_ONLY); the buffers are zeroed. Returns 0, or -1 on NULL, an invalid
 * grid or registration, or a short or misaligned block (header zeroed, no grid; block and buffers
 * untouched). */
int dfn2_model_io_init(DFN2ModelIOState *state, void *block, size_t bytes,
                       const DFN2Grid *grid, float *gru_hidden,
                       size_t gru_hidden_elements, float *df_convp_history,
                       size_t df_convp_history_elements);

/* Feature frames the first call needs: mask_lookahead + frames_per_call; 0 for NULL or no grid. */
unsigned int dfn2_model_io_ready_frames(const DFN2ModelIOState *state);

/* Slide a feature window by one frame, the newest in the last slot. `window` is [window_frames][n_erb] or
 * [2][window_frames][df_bins] (frame[2][df_bins]); window_frames is the grid's input_frames. */
void dfn2_model_io_push_erb_window(float *window, const float *frame, int n_erb,
                                   int window_frames);
void dfn2_model_io_push_spec_window(float *window, const float *frame, int df_bins,
                                    int window_frames);

/* Check the recurrent arrays after the accelerator wrote them in place (element counts; df_convp_history
 * count 0 = absent): 0 when finite, else zeroes them and returns -1. -1 with nothing touched: NULL or empty
 * h_gru, NULL df_convp_history with a count, or any df_convp_history with DFN2_MASK_ONLY. */
int dfn2_model_io_validate_arrays(float *gru_hidden, size_t gru_count,
                                  float *df_convp_history, size_t convp_count);

/* Push feature frame f: erb[n_erb], spec[2][df_bins] (re, then im). Returns 1 when the window of
 * heads[f-L-N+1 .. f-L] is complete (after L + N pushes, then every N), 0 otherwise, -1 on NULL or no grid.
 * Flush: N = 1 push L zero frames, else push zero frames until the call holding the last head has run. */
int dfn2_model_io_push_features(DFN2ModelIOState *state, const float *erb,
                                const float *spec);

/* Index among the N heads of the next call of the head the latest push completes (N - 1 exactly when
 * push_features returned 1); -1 for the first L frames, before any push, and for NULL. */
int dfn2_model_io_frame_slot(const DFN2ModelIOState *state);

/* Call after each invocation: validate_arrays() on the registered state (none: 0); -1 on NULL or non-finite. */
int dfn2_model_io_validate_state(DFN2ModelIOState *state);

/* reset: back to the init values (windows, registered state, frame count), keeping header, registration and
 * pointers (restarts a state in use); zero_recurrent: the registered arrays only. Both advance the generation, as
 * does a refused validate_state() (1 after init, modulo 2^32; 0 for NULL); NULL is ignored. */
void dfn2_model_io_reset(DFN2ModelIOState *state);
void dfn2_model_io_zero_recurrent(DFN2ModelIOState *state);
unsigned int dfn2_model_io_state_generation(const DFN2ModelIOState *state);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_MODEL_IO_H */
