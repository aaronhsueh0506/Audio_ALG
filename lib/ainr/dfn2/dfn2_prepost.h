/* DeepFilterNet2 per-frame functions and model state: features, windows, enhance rings, output and state validation, zero state.
 * Spectra are normalized=True (scale by dfn2_stft_scale(n_fft) in, dfn2_stft_unscale(n_fft) out); L, D, N = mask_lookahead,
 * df_lookahead, frames_per_call. The pool is the caller's and holds no recurrent state. The per-call sequence is dfn2_stream.h's.
 *
 *     pre_frame x N -> views -> accelerator -> validate_run -> post_frame / post_frame_identity x N
 */

#ifndef DFN2_PREPOST_H
#define DFN2_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_model_io.h"   /* DFN2ModelIOState, dfn2_process.h (grid, DFN2State) */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN2_PREPOST_DESCRIPTOR_VERSION 1u
/* In build_flags_hash; bump on any change of pool size, carve order or descriptor content. */
#define DFN2_PREPOST_CARVE_VERSION      13u

#define DFN2_PREPOST_ALIGNMENT 16u   /* the one alignment of every module (audiolib mem_align.h) */

/* Frames the output trails the input by, on top of the framing's hop: L + D, whatever N is. */
static inline int dfn2_prepost_latency_frames(const DFN2Grid *grid)
{
    return grid->mask_lookahead + grid->df_lookahead;
}

/* Element counts at the accelerator boundary; the heads hold frames_per_call frames, oldest first. */
static inline size_t dfn2_prepost_erb_window_elements(const DFN2Grid *grid)
{
    return (size_t)grid->input_frames * (size_t)grid->n_erb;
}
static inline size_t dfn2_prepost_spec_window_elements(const DFN2Grid *grid)
{
    return (size_t)2 * (size_t)grid->input_frames * (size_t)grid->df_bins;
}
static inline size_t dfn2_prepost_erb_mask_elements(const DFN2Grid *grid)
{
    return (size_t)grid->frames_per_call * (size_t)grid->n_erb;
}
#ifndef DFN2_MASK_ONLY
static inline size_t dfn2_prepost_coefs_elements(const DFN2Grid *grid)
{
    return (size_t)grid->frames_per_call * (size_t)grid->df_bins *
           (size_t)grid->df_order * 2u;
}
static inline size_t dfn2_prepost_alpha_elements(const DFN2Grid *grid)
{
    return (size_t)grid->frames_per_call;
}
#endif

typedef struct DFN2PrepostConfig {
    DFN2Grid     grid;          /* the grid the instance runs (dfn2_grid_make); copied at init */
    const float *erb_fwd;       /* required, borrowed, bin-major [n_bins][n_erb] */
    const float *erb_inv;       /* required, borrowed, band-major [n_erb][n_bins] */
    float        atten_lim_db;  /* |x| dB limit, sign ignored; 0 disables  */
    /* Optional recurrent-state registration (borrowed, element counts) under the rules of dfn2_model_io_init():
     * all pairs or none, a pointer with count 0 refused, df_convp_history NULL/0 with DFN2_MASK_ONLY. Zeroed at
     * init and reset, finite-checked at every commit; not in the pool size or hash. */
    float       *gru_hidden;
    size_t       gru_hidden_elements;
    float       *df_convp_history;
    size_t       df_convp_history_elements;
} DFN2PrepostConfig;

/* Fixed 32-byte descriptor; a pool sized for another grid or build is refused by _init_ex. */
typedef struct DFN2PrepostMemReq {
    uint32_t descriptor_version;  /* DFN2_PREPOST_DESCRIPTOR_VERSION        */
    uint32_t layout_version;      /* DFN2_MODEL_IO_LAYOUT_VERSION           */
    uint32_t reserved0;           /* must be 0: _init_ex refuses another value */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, variant, layout and feature versions */
    uint32_t alignment;
    uint32_t reserved;            /* 0 */
    uint64_t bytes;
} DFN2PrepostMemReq;

typedef struct DFN2Prepost DFN2Prepost;

/* Accelerator inputs, pointers into the pool valid for the instance's life. State tensors are the registered
 * buffers (NULL and 0 when none), never moving; the `*_next` outputs alias them (read every state input
 * before writing any state output). The windows are read-only. */
typedef struct DFN2PrepostInputs {
    const float *erb_window;        /* 'erb'  (1,1,input_frames,n_erb) */
    const float *spec_window;       /* 'spec' (1,2,input_frames,df_bins) */
    const float *gru_hidden;        /* 'h_gru' (1,rows,1,hidden): the registered buffer */
#ifndef DFN2_MASK_ONLY
    const float *df_convp_window;   /* 'df_convp_history' (1,channels,window,df_bins): ditto */
#endif
    size_t erb_window_elements;
    size_t spec_window_elements;
    size_t gru_hidden_elements;
#ifndef DFN2_MASK_ONLY
    size_t df_convp_window_elements;
#endif
} DFN2PrepostInputs;

/* Accelerator-writable outputs: the heads (NaN-filled by views() so validate_run() detects a partial write;
 * N frames each, frame-major, oldest first) and the next-state tensors aliasing the inputs' state. */
typedef struct DFN2PrepostOutputs {
    float *erb_mask;                /* 'erb_mask'  [N][n_erb]               */
#ifndef DFN2_MASK_ONLY
    float *coefs;                   /* 'df_coefs'  [N][bins][order][2]      */
    float *alpha;                   /* 'df_alpha'  [N]                      */
#endif
    float *gru_hidden_next;         /* h_gru_out: the registered buffer     */
#ifndef DFN2_MASK_ONLY
    float *df_convp_window_next;    /* df_convp_history_out (the registered buffer) */
#endif
    size_t erb_mask_elements;
#ifndef DFN2_MASK_ONLY
    size_t coefs_elements;
    size_t alpha_elements;
#endif
    size_t gru_hidden_elements;
#ifndef DFN2_MASK_ONLY
    size_t df_convp_window_elements;
#endif
} DFN2PrepostOutputs;

/* Fill a config for `grid` (copied), the rest NULL/0; -1 on NULL or an invalid grid. */
int dfn2_prepost_config_defaults(DFN2PrepostConfig *cfg, const DFN2Grid *grid);

/* Exact pool size; -1 (req untouched) on NULL, missing ERB matrices, non-finite atten_lim_db, an invalid grid or state
 * registration, or mask_lookahead > 0 with frames_per_call > 1. */
int dfn2_prepost_get_mem_size(const DFN2PrepostConfig *cfg,
                              DFN2PrepostMemReq *req);

/* Construct in caller memory (16-byte aligned, `bytes` >= the reported size), starting reset. NULL on a bad
 * config, ERB tables failing dfn2_erb_tables_valid(), or a misaligned or undersized pool (pool and registered buffers untouched). */
DFN2Prepost *dfn2_prepost_init(void *pool, size_t bytes,
                               const DFN2PrepostConfig *cfg);

/* As _init, plus the stale-pool gate; `expected` NULL behaves as _init. */
DFN2Prepost *dfn2_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN2PrepostConfig *cfg,
                                  const DFN2PrepostMemReq *expected);

/* get_mem_size + aligned allocation + init (host path). */
DFN2Prepost *dfn2_prepost_create(const DFN2PrepostConfig *cfg);

/* Frees only what _create allocated: a no-op on an _init instance, never the caller's matrices or state. */
void dfn2_prepost_destroy(DFN2Prepost *p);

/* Back to the start: rings, windows and registered state zeroed, clocks and normalisers restarted, generation advanced;
 * config, borrowed pointers and pool stay. Reset your own AcStft / AcIstft too. */
void dfn2_prepost_reset(DFN2Prepost *p);

/* Generation of the recurrent state (dfn2_model_io_state_generation()): 1 after init, plus one per reset, per
 * refused validate_run() and per zero_state(); 0 for NULL. A runtime with its own state copy reloads quantized zeros
 * on every change. */
unsigned int dfn2_prepost_state_generation(const DFN2Prepost *p);

const DFN2Grid *dfn2_prepost_grid(const DFN2Prepost *p);   /* the grid it runs; NULL for NULL */

/* Point at caller-loaded ERB matrices, borrowed and never copied. Between calls only; a swap consistent per
 * source frame needs a reset first. -1 on NULL or tables failing dfn2_erb_tables_valid(). */
int dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd,
                                  const float *erb_inv);

/* |atten_lim_db| dB attenuation limit (sign ignored, 0 disables), between calls only; -1 on NULL or non-finite.
 * With N frames per call the enhance stage runs at the commit, so the limit applies to the whole next committed call. */
int dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db);

/* Pre, one frame: features of est (n_bins bins, MODEL scale) and the graph windows advance. Bins are not checked: each must be
 * finite with |bin| <= 2^63, else the normaliser is corrupt until reset (the stream checks). 1 when the window of the call's
 * heads is complete (a job is due), 0 not yet (the first L frames at N == 1), -1 on NULL or no grid. */
int dfn2_prepost_pre_frame(DFN2Prepost *p, const float *est_re, const float *est_im);

/* Publish the accelerator boundary once per job, after the last pre_frame: window and head pointers into the pool, state
 * pointers the registered buffers (NULL when none); heads NaN-prefilled, state not. -1 on NULL. */
int dfn2_prepost_views(DFN2Prepost *p, DFN2PrepostInputs *inputs, DFN2PrepostOutputs *outputs);

/* Largest |value| of an erb_mask or coefs element that validate_run() accepts (a sigmoid mask is at most 1). */
#define DFN2_PREPOST_HEAD_LIMIT 1.0e6f

/* Check the accelerator's output for the whole call: every head finite and within DFN2_PREPOST_HEAD_LIMIT (alpha finite), then the
 * registered state. 0 accepted, nothing written. -1 refused: the registered tensors are zeroed and the generation advanced once;
 * -1 on NULL, nothing touched. */
int dfn2_prepost_validate_run(DFN2Prepost *p);

/* Post, one frame, with head `slot` (of N) of the lib-owned heads and app = this frame's application spectrum (n_bins bins, MODEL
 * scale); out (n_bins, never aliasing app) is the enhanced frame at the input's scale. 1: emitted. 0: no spectrum yet (the first
 * L + D frames), out zero-filled; while the clock is below L the head is discarded. -1: NULL, slot outside [0, N) or a non-finite
 * alpha or attenuation limit, nothing advanced. Advances the rings, the clock and output_frame_index. */
int dfn2_prepost_post_frame(DFN2Prepost *p, int slot, const float *app_re, const float *app_im,
                            float *out_re, float *out_im);

/* Post, one frame, identity heads (unit band mask, zero taps, alpha 0): the rings and the clock advance as post_frame(), so later
 * deep-filter taps read the frame; while the clock is below L the frame is queued without heads. Same returns and out as post_frame(). */
int dfn2_prepost_post_frame_identity(DFN2Prepost *p, const float *app_re, const float *app_im,
                                     float *out_re, float *out_im);

/* Zero the registered recurrent tensors and advance the generation (a run that started may have written them); the buffers
 * are untouched when none are registered. NULL is ignored. */
void dfn2_prepost_zero_state(DFN2Prepost *p);

/* Source frame (from 0) of the last emission, written to *frame; -1 on NULL or before the first emission. */
int dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame);

/* Accelerator callbacks: infer() gets the job's views and returns the run status (stream_run.h): 0 once every output is written,
 * STREAM_RUN_NOT_STARTED when it ran nothing, any other value for a run that started and failed; reset() optional; a model that
 * infers must publish an io_descriptor passing dfn2_model_io_descriptor_validate(). */
typedef struct DFN2Model {
    void *user;
    int (*infer)(void *user, const DFN2PrepostInputs *inputs,
                 DFN2PrepostOutputs *outputs);
    void (*reset)(void *user);
    const DFN2ModelIoDescriptor *io_descriptor;   /* NULL = not published */
} DFN2Model;

#ifdef __cplusplus
}
#endif

#endif /* DFN2_PREPOST_H */
