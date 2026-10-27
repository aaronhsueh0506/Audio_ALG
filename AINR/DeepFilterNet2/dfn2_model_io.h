#ifndef DFN2_MODEL_IO_H
#define DFN2_MODEL_IO_H

#include "dfn2_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Must match export_onnx.py. Graph state tensors (rank 4; each *_out output has
 * the shape and address of its input):
 *   h_gru             [1, DFN2_MODEL_GRU_ROWS, 1, DFN2_MODEL_GRU_HIDDEN]; rows:
 *                     encoder GRU, ERB decoder GRU (2), DF decoder GRU (2), at
 *                     the DFN2_MODEL_GRU_ROW_* offsets
 *   df_convp_history  [1, DFN2_MODEL_ENCODER_CHANNELS,
 *                      DFN2_MODEL_DF_PATHWAY_WINDOW, DFN2_DF_BINS]; the full
 *                     causal window of the pathway convolution, returned whole
 * Layout versions 4-7 are never reused. The layout version names the state
 * tensors, which DFN2_FRAMES_PER_CALL leaves unchanged; the frame count of the
 * feature windows and heads is the descriptor's frames_per_call. */
#define DFN2_MODEL_IO_LAYOUT_VERSION       8
/* Feature frames behind one head (the input convolution's temporal kernel). */
#define DFN2_MODEL_INPUT_FRAMES            3

/* Heads per accelerator invocation (N): a graph exported with
 * export_onnx.py --frames-per-call N reads N + 2 feature frames and returns the
 * heads of N consecutive frames with the state after the last one. A build
 * for N > 1 passes -DDFN2_FRAMES_PER_CALL=<N> (plain decimal) on every
 * translation unit, like the lookahead pair; objects built for different N do
 * not link (DFN2_REQUIRE_FRAMES_PER_CALL_LINK below). N > 1 changes the
 * per-hop contract (readiness every N hops, N heads per commit, N - 1 more
 * hops of latency), so a unit written for one frame per call must not compile
 * against it: a unit that drives N frames defines DFN2_MULTI_FRAME_HOST before
 * including this header. */
#ifndef DFN2_FRAMES_PER_CALL
#define DFN2_FRAMES_PER_CALL               1
#endif
#define DFN2_MODEL_MAX_FRAMES_PER_CALL     64

#if DFN2_FRAMES_PER_CALL < 1 || DFN2_FRAMES_PER_CALL > DFN2_MODEL_MAX_FRAMES_PER_CALL
#error "DFN2_FRAMES_PER_CALL must be in [1, 64]: heads per accelerator invocation"
#endif
#if DFN2_FRAMES_PER_CALL != 1 && !defined(DFN2_MULTI_FRAME_HOST)
#error "DFN2_FRAMES_PER_CALL > 1: this unit was written for one frame per call; a unit that drives N frames per call defines DFN2_MULTI_FRAME_HOST"
#endif

/* Feature frames per invocation: N three-frame windows overlapping by two. */
#define DFN2_MODEL_WINDOW_FRAMES \
    (DFN2_MODEL_INPUT_FRAMES - 1 + DFN2_FRAMES_PER_CALL)

/* Link-time N check, like DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK: dfn2_model_io.c
 * defines dfn2_frames_per_call<N> and dfn2_prepost.c references it. */
#define DFN2_FRAMES_PER_CALL_SYMBOL_JOIN_(frames) dfn2_frames_per_call##frames
#define DFN2_FRAMES_PER_CALL_SYMBOL_JOIN(frames) \
    DFN2_FRAMES_PER_CALL_SYMBOL_JOIN_(frames)
#define DFN2_FRAMES_PER_CALL_SYMBOL \
    DFN2_FRAMES_PER_CALL_SYMBOL_JOIN(DFN2_FRAMES_PER_CALL)
extern const unsigned char DFN2_FRAMES_PER_CALL_SYMBOL;
#define DFN2_REQUIRE_FRAMES_PER_CALL_LINK() \
    ((void)*(const volatile unsigned char *)&DFN2_FRAMES_PER_CALL_SYMBOL)

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
/* The window that yields heads[t] is frames [t-(2-L) .. t+L] for
 * L = DFN2_MASK_LOOKAHEAD: it is complete once frame t+L has been pushed, so the
 * first heads (t = 0) need L + 1 pushed frames. Frames before the stream start
 * read as zero (the window is zeroed at init); at the end of a stream, L zero
 * frames flush the last L heads.
 * With N = DFN2_FRAMES_PER_CALL the window of one call is
 * [t-(2-L) .. t+N-1+L] and yields heads[t .. t+N-1], head i from the three
 * frames starting at slot i. The first call (t = 0) needs L + N pushed frames
 * and every later one N more; to flush, push zero frames until the call that
 * holds the last head has run (L more, plus enough to complete that call). */
#define DFN2_MODEL_READY_FRAMES \
    (DFN2_MASK_LOOKAHEAD + 0U + DFN2_FRAMES_PER_CALL)

#if DFN2_MODEL_READY_FRAMES > DFN2_MODEL_WINDOW_FRAMES
#error "the first window must fit in the model input"
#endif

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

/* The graph contract a runtime was built against, compared field by field. Every
 * field is a compiled constant, so a mismatch means a graph exported against
 * another layout version or geometry. */
typedef struct DFN2ModelIoDescriptor {
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
    int input_frames;              /* DFN2_MODEL_WINDOW_FRAMES (metadata
                                      input_feature_frames)                */
    int encoder_gru_layers;
    int erb_gru_layers;
    int df_gru_layers;
    int gru_hidden;
    int encoder_channels;
    int df_pathway_window;
    int state_tensor_count;        /* 2: h_gru, df_convp_history           */
    int frames_per_call;           /* DFN2_FRAMES_PER_CALL (metadata
                                      frames_per_call; 1 when absent)      */
} DFN2ModelIoDescriptor;

/* Fill with this build's contract. Returns 0, or -1 on NULL. */
int dfn2_model_io_descriptor_default(DFN2ModelIoDescriptor *descriptor);
/* 0 when every field equals this build's contract, -1 otherwise (NULL included). */
int dfn2_model_io_descriptor_validate(
    const DFN2ModelIoDescriptor *descriptor);

/* Caller-owned state for a stateless accelerator. The graph reads the two state
 * arrays and writes the updated ones to outputs bound to the same addresses
 * (read all state inputs before writing any state output). The feature windows
 * and feature_frames_seen are CPU-owned and read-only to the accelerator.
 * state_generation advances whenever the recurrent arrays are zeroed (reset or
 * refused validation); a runtime holding a fixed-point copy of the state refills
 * it with quantized zeros when dfn2_model_io_state_generation() changes. */
typedef struct {
    float erb_window[DFN2_MODEL_WINDOW_FRAMES][DFN2_N_ERB];
    float spec_window[2][DFN2_MODEL_WINDOW_FRAMES][DFN2_DF_BINS];
    float gru_hidden[DFN2_MODEL_GRU_ROWS][DFN2_MODEL_GRU_HIDDEN];
    float df_convp_window[DFN2_MODEL_ENCODER_CHANNELS]
                         [DFN2_MODEL_DF_PATHWAY_WINDOW]
                         [DFN2_DF_BINS];
    unsigned long long feature_frames_seen;
    unsigned int state_generation;
} DFN2ModelIOState;

/* First construction: zero the object and start the generation at 1. A state in
 * use goes through dfn2_model_io_reset(). The caller zeroes the accelerator
 * state once afterwards. */
void dfn2_model_io_init(DFN2ModelIOState *state);

/* Slide a feature window by one frame and write the newest frame into the last
 * slot. Slot 2 holds frame f, slot 1 f-1, slot 0 f-2: with f = t + L that is
 * [t-(2-L) .. t+L], the alignment export_onnx.py publishes as
 * input_window_alignment. With N frames per call the window holds the last
 * N + 2 frames the same way (chunk_window_alignment). */
void dfn2_model_io_push_erb_window(
    float window[DFN2_MODEL_WINDOW_FRAMES][DFN2_N_ERB],
    const float frame[DFN2_N_ERB]);
void dfn2_model_io_push_spec_window(
    float window[2][DFN2_MODEL_WINDOW_FRAMES][DFN2_DF_BINS],
    const float frame[2][DFN2_DF_BINS]);

/* Validate the two recurrent arrays after the accelerator wrote them in place.
 * Returns 0 when all elements are finite; otherwise zeroes both arrays and
 * returns -1 (the previous values cannot be restored). NULL returns -1 and
 * touches nothing. */
int dfn2_model_io_validate_arrays(
    float gru_hidden[DFN2_MODEL_GRU_ROWS][DFN2_MODEL_GRU_HIDDEN],
    float df_convp_window[DFN2_MODEL_ENCODER_CHANNELS]
                         [DFN2_MODEL_DF_PATHWAY_WINDOW]
                         [DFN2_DF_BINS]);

/* Push one newly computed feature frame f. Returns 1 when the window
 * [t-(2-L) .. t+L] of t = f - L is available and the graph may emit heads[t],
 * or 0 for the first DFN2_MASK_LOOKAHEAD frames (nothing to run yet). To flush
 * the final heads, push DFN2_MASK_LOOKAHEAD all-zero feature frames.
 * With N frames per call it returns 1 when the window of heads[f-L-N+1 .. f-L]
 * is complete (pushes L + N, L + 2N, ...) and 0 otherwise. */
int dfn2_model_io_push_features(DFN2ModelIOState *state,
                                const float erb[DFN2_N_ERB],
                                const float spec[2][DFN2_DF_BINS]);

/* Position, among the N heads of the next call, of the head the most recently
 * pushed frame completes: 0 .. DFN2_FRAMES_PER_CALL - 1 (N - 1 exactly when
 * push_features returned 1), or -1 while that frame was one of the first
 * DFN2_MASK_LOOKAHEAD (no head yet), before any push, and for NULL. Always 0
 * after the warm-up when N = 1. */
int dfn2_model_io_frame_slot(const DFN2ModelIOState *state);

/* validate_arrays() over the state's own recurrent arrays; call after each
 * invocation. Returns -1 on NULL or non-finite (arrays zeroed, generation
 * advances). The feature windows are untouched. */
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
