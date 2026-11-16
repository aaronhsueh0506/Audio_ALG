/* DeepVQE-S pre/post class: per-frame functions and the model state between the caller's spectra and
 * the accelerator's tensors, frequency domain only. One call carries N = grid.frames_per_call frames
 * and one accelerator run; the caller (the stream) sequences it:
 *
 *     deepvqe_prepost_pre_frame(p, k, mic_re, mic_im, far_re, far_im);   // k = 0 .. N-1
 *     deepvqe_prepost_views(p, &in, &out);        // after the last pre_frame: NaN-filled head, tensor views
 *     my_npu_run(&in, &out);                      // fills `out` (N taps)
 *     deepvqe_prepost_validate_run(p);            // after a run with status OK; -1: state restarted
 *     deepvqe_prepost_post_frame(p, k, mic_re, mic_im, out_re, out_im);  // k = 0 .. N-1, in order
 *
 * pre_frame writes frame k into the compressed model windows. post_frame applies the taps of frame k
 * to the CCM ring of the last TIME_ORDER raw microphone spectra, which the caller keeps from
 * pre_frame until post_frame (the ring advances once per post_frame). A run that did not succeed
 * is closed with deepvqe_prepost_zero_state() (or nothing when the device state was not touched).
 * The class holds no per-call latch: the caller owns the order. Nothing in it names the stream.
 *
 * Framing: one AcStft per stream and one AcIstft for the output, all from
 * ac_stft_config_default(&sc, n_fft) (sqrt-Hann, center 0, hop n_fft / 2, spec_scale =
 * ispec_scale = 1) with sc.ola = AC_OLA_ACCUMULATE (analysis ignores ola); one frame per hop from
 * the first hop, no latency beyond it. The class holds no window and no FFT handle. */

#ifndef DEEPVQE_PREPOST_H
#define DEEPVQE_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_process.h"   /* DeepVqeCcmState, DEEPVQE_TIME_ORDER, DEEPVQE_FREQ_TAPS */

/* DeepVqeGrid/DeepVqeGridSpec's frames_per_call field: N == 1 unchanged, N > 1 batches N frames
 * per accelerator call (see the lifecycle comment above). */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_PREPOST_DESCRIPTOR_VERSION 1u

/* Accelerator boundary version (two signal inputs, the CCM-tap head, sixteen rank-4 state tensors,
 * each returned in full); any rename, reshape, reorder or change of the tensor count bumps it.
 * frames_per_call does not: the signal/head tensors' time axis grows with it without changing
 * their names, count, or the (unaffected) state tensors -- deepvqe_grid_validate() and the pool's
 * build_flags_hash (which folds frames_per_call in directly) are what refuse a mismatched N,
 * same split DFN2 uses between its model_io layout version and frames_per_call. */
#define DEEPVQE_PREPOST_LAYOUT_VERSION 4u

/* Input compression exponent (model.compression_exponent); the exporter refuses others. */
#define DEEPVQE_COMPRESSION_EXP 0.3f
#define DEEPVQE_PREPOST_ALIGNMENT      16u
/* Folded into build_flags_hash; bumped on any change of pool size, carve order or descriptor
 * that ISN'T already a function of frames_per_call (which is folded into the hash directly). */
#define DEEPVQE_PREPOST_CARVE_VERSION  9u

/* Sanity bounds for the delay depth D (exporter max_delay_frames). D has no default: the
 * application supplies the value its model is exported with; it sizes nothing here. */
#define DEEPVQE_PREPOST_MIN_D  1
#define DEEPVQE_PREPOST_MAX_D  256

/* frames_per_call upper bound, same as the exporter's (AIAEC/_streaming_export.py
 * MAX_FRAMES_PER_CALL) and DFN2's DFN2_MODEL_MAX_FRAMES_PER_CALL. */
#define DEEPVQE_MODEL_MAX_FRAMES_PER_CALL 64

typedef struct DeepVqeGridSpec {
    int sample_rate;
    int n_fft;
    int delay_depth;   /* D, as exported */
    int frames_per_call;   /* heads per accelerator call, [1, DEEPVQE_MODEL_MAX_FRAMES_PER_CALL] */
} DeepVqeGridSpec;

typedef struct DeepVqeGrid {
    int sample_rate;
    int n_fft;
    int hop;
    int n_bins;
    int delay_depth;
    int frames_per_call;
} DeepVqeGrid;

enum {
    DEEPVQE_GRID_OK = 0,
    DEEPVQE_GRID_ERR_NULL = -1,
    DEEPVQE_GRID_ERR_SAMPLE_RATE = -2,
    DEEPVQE_GRID_ERR_N_FFT = -3,
    DEEPVQE_GRID_ERR_DELAY_DEPTH = -4,
    DEEPVQE_GRID_ERR_INCONSISTENT = -5,
    DEEPVQE_GRID_ERR_FRAMES_PER_CALL = -6
};

/* Build the run-time grid from a spec filled with what the model is exported for (hop = n_fft / 2,
 * n_bins = n_fft / 2 + 1). Returns DEEPVQE_GRID_OK (grid written), or the first failing check
 * (grid untouched): sample_rate > 0; n_fft a size audiolib's framing builds (ac_stft_get_mem_size()
 * of the default config is not 0); delay_depth within [DEEPVQE_PREPOST_MIN_D, MAX_D];
 * frames_per_call within [1, DEEPVQE_MODEL_MAX_FRAMES_PER_CALL]. */
int deepvqe_grid_make(DeepVqeGrid *grid, const DeepVqeGridSpec *spec);

/* _make's checks plus consistent hop and n_bins. Returns DEEPVQE_GRID_OK or the failing check. */
int deepvqe_grid_validate(const DeepVqeGrid *grid);

/* CCM taps [1,N,BINS,TIME_ORDER*FREQ_TAPS*2] (N = grid.frames_per_call, oldest first), last axis
 * [time][frequency][RI]. */
static inline size_t deepvqe_prepost_taps_elements(const DeepVqeGrid *grid)
{
    return (size_t)grid->frames_per_call * (size_t)grid->n_bins *
           DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2u;
}

/* Graph input order after the two signal inputs (outputs: after the head). */
typedef enum DeepVqeStateId {
    DEEPVQE_STATE_ALIGN_KEY_RING = 0,
    DEEPVQE_STATE_ALIGN_VALUE_RING,
    DEEPVQE_STATE_ALIGN_SCORE_HISTORY,
    DEEPVQE_STATE_CCM_UP_HISTORY,
    DEEPVQE_STATE_FAR1_HISTORY,
    DEEPVQE_STATE_FAR2_HISTORY,
    DEEPVQE_STATE_H_GRU,
    DEEPVQE_STATE_MIC1_HISTORY,
    DEEPVQE_STATE_MIC2_HISTORY,
    DEEPVQE_STATE_MIC3_HISTORY,
    DEEPVQE_STATE_MIC4_HISTORY,
    DEEPVQE_STATE_RES2_HISTORY,
    DEEPVQE_STATE_RES3_HISTORY,
    DEEPVQE_STATE_UP1_HISTORY,
    DEEPVQE_STATE_UP2_HISTORY,
    DEEPVQE_STATE_UP3_HISTORY,
    DEEPVQE_STATE_COUNT
} DeepVqeStateId;

typedef struct DeepVqePrepostConfig {
    DeepVqeGrid grid;    /* deepvqe_grid_make(); copied at init */
    /* The application's recurrent state by DeepVqeStateId: borrowed, must outlive the instance;
     * shapes are the application's, only element counts are given. A float host registers its
     * sixteen buffers (zeroed at init and reset, finite-checked whole at every validate_run); a
     * board whose accelerator owns the state registers nothing and binds each state input and
     * its `_out` to one accelerator buffer. All or none: every pointer non-NULL with a nonzero
     * count, or all NULL/0, else refused. Not in the pool size or hash. */
    float *state[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostConfig;

/* The model ABI for validating a graph's ONNX/JSON metadata. The library owns layout_version,
 * delay_depth (range-checked only), the grid, time_order, freq_taps and state_tensor_count; the
 * conv/score window and GRU fields are the model's: zero from _descriptor_default(), never used. */
typedef struct DeepVqePrepostDescriptor {
    uint32_t layout_version;
    int delay_depth;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int time_order;
    int freq_taps;
    int conv_window_frames;
    int score_window_frames;
    int gru_layers;
    int gru_hidden;
    int state_tensor_count;
} DeepVqePrepostDescriptor;

/* Fixed 32-byte descriptor; a pool sized for another grid, D or build is refused by _init_ex. */
typedef struct DeepVqePrepostMemReq {
    uint32_t descriptor_version;  /* DEEPVQE_PREPOST_DESCRIPTOR_VERSION     */
    uint32_t layout_version;      /* DEEPVQE_PREPOST_LAYOUT_VERSION         */
    uint32_t reserved0;           /* 0 */
    uint32_t build_flags_hash;    /* FNV-1a-32 over sample rate, n_fft, D, N and versions */
    uint32_t alignment;
    uint32_t reserved1;           /* 0 */
    uint64_t bytes;
} DeepVqePrepostMemReq;

typedef struct DeepVqePrepost DeepVqePrepost;

/* Read-only accelerator views: `mic`/`far` are the compressed planar spectra, N = grid.frames_per_call
 * frames packed real-plane-then-imaginary (BINS = grid.n_bins, each plane N*BINS, oldest frame
 * first within a plane); `state[]` is the registered buffers by DeepVqeStateId, NULL with 0
 * elements when none (unaffected by N -- see deepvqe_prepost.h's frames_per_call note). The
 * pointers are constant for the instance's life. */
typedef struct DeepVqePrepostInputs {
    const float *mic;                            /* [1,2,N,BINS] */
    const float *far;                            /* [1,2,N,BINS] */
    size_t       spectrum_ri_elements;           /* 2 * N * BINS */
    const float *state[DEEPVQE_STATE_COUNT];
    size_t       state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostInputs;

/* Accelerator-writable outputs; the graph returns the full next value of every state tensor.
 * `state_out[id]` has the address of `inputs.state[id]` (both NULL when nothing is registered;
 * a board binds each pair to one accelerator buffer): read every state input before writing any
 * state output. Only `taps` is NaN-prefilled, by views(). */
typedef struct DeepVqePrepostOutputs {
    float *taps;   /* [1,N,BINS,TIME_ORDER*FREQ_TAPS*2], packed CCM taps, oldest frame first */
    size_t taps_elements;
    float *state_out[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostOutputs;

/* Exact graph tensor names; static literals, NULL for an unknown id. */
const char *deepvqe_prepost_state_name(int state_id);      /* graph input  */
const char *deepvqe_prepost_state_name_out(int state_id);  /* graph output */

/* Fill a config for `grid` (copied), nothing registered. Returns 0, or -1 on NULL/invalid grid. */
int deepvqe_prepost_config_defaults(DeepVqePrepostConfig *cfg, const DeepVqeGrid *grid);

/* Fill the external fields of the model ABI for `grid`; the interior ones are 0. Returns 0, or
 * -1 on NULL or an invalid grid. */
int deepvqe_prepost_descriptor_default(const DeepVqeGrid *grid,
                                       DeepVqePrepostDescriptor *descriptor);

/* Every external field except delay_depth (range-checked only) must equal the contract for `grid`.
 * Returns 0 on match, -1 otherwise (NULL and an invalid grid included). */
int deepvqe_prepost_descriptor_validate(
    const DeepVqeGrid *grid, const DeepVqePrepostDescriptor *descriptor);

/* Exact pool size: a function of n_bins, not of D. The caller supplies the pool and _init
 * carves it. Returns -1 (req untouched) on NULL, an invalid grid, or an inconsistent state
 * registration. */
int deepvqe_prepost_get_mem_size(const DeepVqePrepostConfig *cfg,
                                 DeepVqePrepostMemReq *req);

/* Construct inside caller memory; starts reset (the registered state, if any, is zeroed).
 * NULL on a bad config, misaligned or undersized pool. */
DeepVqePrepost *deepvqe_prepost_init(void *pool, size_t bytes,
                                     const DeepVqePrepostConfig *cfg);

/* As _init, plus the stale-pool gate; `expected` NULL behaves as _init. */
DeepVqePrepost *deepvqe_prepost_init_ex(void *pool, size_t bytes,
                                        const DeepVqePrepostConfig *cfg,
                                        const DeepVqePrepostMemReq *expected);

/* get_mem_size + aligned allocation + init (host path). */
DeepVqePrepost *deepvqe_prepost_create(const DeepVqePrepostConfig *cfg);

/* Frees only what _create allocated, never the registered state. NULL-safe; no-op for _init. */
void deepvqe_prepost_destroy(DeepVqePrepost *p);

/* Zero the registered state and the CCM ring and clear the model windows; config, every pointer and
 * the pool stay. Advances the state generation. NULL is ignored. */
void deepvqe_prepost_reset(DeepVqePrepost *p);

/* A state restart: zero the registered state, empty the CCM ring, advance the generation; the
 * windows stay. NULL is ignored. */
void deepvqe_prepost_zero_state(DeepVqePrepost *p);

/* Counter of state restarts (reset, zero_state, a refusing validate_run); 0 after
 * init and for NULL; kept whether or not a state is registered. Every advance also empties the CCM
 * ring, so the next frame is a cold start. A fixed-point accelerator loads quantized zeros into its
 * own copy of the state at start and on every change. */
uint32_t deepvqe_prepost_state_generation(const DeepVqePrepost *p);

const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(
    const DeepVqePrepost *p);

enum {
    DEEPVQE_PREPOST_OK = 0,
    DEEPVQE_PREPOST_ERR_ARG = -1,         /* NULL or a slot outside [0, N); nothing changed */
    DEEPVQE_PREPOST_ERR_NONFINITE = -2    /* post_frame: a non-finite enhanced bin; the ring advanced */
};

/* Frame `slot` of the call: compress the n_bins-bin mic and far spectra into the planar [1,2,N,BINS]
 * windows. Moves no model state. Returns OK or ERR_ARG. */
int deepvqe_prepost_pre_frame(DeepVqePrepost *p, int slot,
                              const float *mic_re, const float *mic_im,
                              const float *far_re, const float *far_im);

/* NaN-fill the head (`taps`; the state buffers are the accelerator's input too and stay) and fill the
 * accelerator boundary; once per run, after the last pre_frame. Idempotent. Returns 0, or -1 on NULL. */
int deepvqe_prepost_views(DeepVqePrepost *p, DeepVqePrepostInputs *inputs,
                          DeepVqePrepostOutputs *outputs);

/* After a run with status OK: every tap and every element of every registered state buffer (whole
 * buffers, older frames of a history window included) must be finite. Returns 0 (nothing written), or
 * -1 on a non-finite value: the registered buffers are zeroed, the CCM ring emptied and the state
 * generation advanced once. -1 on NULL with nothing touched. */
int deepvqe_prepost_validate_run(DeepVqePrepost *p);

/* Frame `slot`: push the raw mic spectrum (n_bins bins, kept by the caller since pre_frame) into the CCM
 * ring, convolve the taps of the frame over (t, t-1, t-2) and write n_bins bins to out_re / out_im, which
 * do not alias the inputs. Slots run 0 .. N-1 in order. Returns OK; ERR_ARG with nothing advanced and
 * out untouched; ERR_NONFINITE when an enhanced bin is not finite (the ring advanced: the caller then
 * calls zero_state once). */
int deepvqe_prepost_post_frame(DeepVqePrepost *p, int slot,
                               const float *mic_re, const float *mic_im,
                               float *out_re, float *out_im);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_PREPOST_H */
