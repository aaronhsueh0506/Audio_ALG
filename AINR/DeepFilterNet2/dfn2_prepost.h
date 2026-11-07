/* DeepFilterNet2 pre/post class: one object between the caller's spectra and the accelerator's tensors.
 * Frequency domain only: the application frames with audiolib as dfn2_process.h states; spectra are
 * normalized=True (unnormalised: scale by grid.stft_scale in, grid.stft_unscale out). L, D, N = mask_lookahead,
 * df_lookahead, frames_per_call (as the graph's). The pool is the caller's and holds no recurrent state.
 *
 *     ac_stft_push_planar(stft, in_samples, re, im);
 *     dfn2_prepost_pre_process_freq(p, re, im);        // 1 on every hop when N == 1
 *     dfn2_prepost_frame_open(p, &in, &out);           // views into the pool
 *     my_npu_run(&in, &out);                           // NPU fills `out`
 *     dfn2_prepost_frame_finish(p);                    // or _frame_bypass()
 *     dfn2_prepost_post_process_freq(p, out_re, out_im, NULL);
 *     ac_istft_push_planar(istft, out_re, out_im, out_samples);   // every hop (AC_OLA_ACCUMULATE)
 *
 * N == 1 with L >= 1: the first L frames after init or reset are warm-up frames (below); N > 1:
 * pre_process_freq returns 1 at the zero-indexed hop L + N - 1, then every N hops after (L == 0 included,
 * with no warm-up frames). */

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
#define DFN2_PREPOST_CARVE_VERSION      11u

#define DFN2_PREPOST_ALIGNMENT 16u   /* the one alignment of every module (audiolib mem_align.h) */

/* Frames the output trails the input by, on top of the framing's hop: L + D + N - 1. */
static inline int dfn2_prepost_latency_frames(const DFN2Grid *grid)
{
    return grid->mask_lookahead + grid->df_lookahead + grid->frames_per_call - 1;
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
    float        atten_lim_db;  /* 0 disables the attenuation limit         */
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

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DFN2PrepostMemReq) == 32,
               "DFN2PrepostMemReq must stay 32 bytes");
#endif
#endif

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

/* Accelerator-writable outputs: the heads (NaN-filled by frame_open() so frame_finish() detects a partial
 * write; N frames each, frame-major, oldest first) and the next-state tensors aliasing the inputs' state. */
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

/* Exact pool size; -1 (req untouched) on NULL, missing ERB matrices, non-finite atten_lim_db, an invalid grid
 * or state registration. */
int dfn2_prepost_get_mem_size(const DFN2PrepostConfig *cfg,
                              DFN2PrepostMemReq *req);

/* Construct in caller memory (16-byte aligned, `bytes` >= the reported size), starting reset. NULL on a bad
 * config or a misaligned or undersized pool (pool and registered buffers untouched). */
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

/* Back to the start: rings and registered state zeroed, open frame dropped, clocks and normalisers restarted,
 * generation advanced, warm-up frames (N == 1) due again; config, borrowed pointers and pool stay. Reset your
 * own AcStft / AcIstft too. */
void dfn2_prepost_reset(DFN2Prepost *p);

/* Generation of the recurrent state (dfn2_model_io_state_generation()): 1 after init, plus one per reset, per
 * warm-up frame and per refused frame_finish(); 0 for NULL. A runtime with its own state copy reloads quantized
 * zeros on every change. */
unsigned int dfn2_prepost_state_generation(const DFN2Prepost *p);

int dfn2_prepost_hop_size(const DFN2Prepost *p);   /* or -1 */
int dfn2_prepost_num_bins(const DFN2Prepost *p);   /* or -1 */
const DFN2Grid *dfn2_prepost_grid(const DFN2Prepost *p);   /* the grid it runs; NULL for NULL */
int dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);   /* latency_frames(); -1 on NULL */
int dfn2_prepost_layout_version(const DFN2Prepost *p);   /* the graph contract bound; -1 on NULL */

/* Point at caller-loaded ERB matrices, borrowed and never copied. Between hops only; a swap consistent per
 * source frame needs a reset first. -1 on NULL or with a frame open. */
int dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd,
                                  const float *erb_inv);

/* Attenuation limit in dB (0 disables), between hops only; -1 on NULL, a non-finite value or with a frame open.
 * With N frames per call the enhance stage runs at the commit, so the limit (and a new inverse ERB matrix)
 * applies to the whole next committed call, frames pushed earlier included. */
int dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db);

/* One analysed frame (n_bins bins), no transform. Returns the accelerator invocations needed: N == 1: 1 on every
 * frame, the first L being warm-up frames (their heads describe a frame before the stream start and are
 * discarded by frame_finish / frame_bypass); N > 1: 1 at the zero-indexed hop L + N - 1, then every N hops
 * after (L == 0 included, with no warm-up frames), 0 otherwise. -1 on NULL args or with a frame open. Flush:
 * push zero spectra and run the accelerator until the last real frame is emitted. */
int dfn2_prepost_pre_process_freq(DFN2Prepost *p, const float *spec_re,
                                  const float *spec_im);

/* Dual input: `est` feeds the estimation branch (features, graph window, heads), `app` the application branch
 * (what the mask, deep filter and attenuation limit act on; kept per frame); else as pre_process_freq(). */
int dfn2_prepost_pre_process_freq_dual(DFN2Prepost *p, const float *est_re,
                                       const float *est_im,
                                       const float *app_re,
                                       const float *app_im);

/* Publish the open frame's accelerator boundary: window and head pointers into the pool, state pointers the
 * registered buffers (NULL when none); heads NaN-prefilled, state not. -1 if no frame is open. */
int dfn2_prepost_frame_open(DFN2Prepost *p, DFN2PrepostInputs *inputs,
                              DFN2PrepostOutputs *outputs);

/* Check the accelerator's output, then build the enhanced spectrum once per frame of the call (oldest
 * first); needs frame_open(). A non-finite head or registered state zeroes the registered tensors, advances
 * the generation, leaves the frame open and returns -1: call frame_bypass() or redo via frame_open().
 * A warm-up frame (N == 1) reads no head: it zeroes the registered state, advances the generation by one,
 * emits nothing and returns 0, so the first real frame starts from zero state. */
int dfn2_prepost_frame_finish(DFN2Prepost *p);

/* Give up on the open frame and pass the noisy spectrum through for every frame of the call (unit ERB mask,
 * alpha 0); clocks advance, recurrent state untouched. -1 if no frame is open. A warm-up frame (N == 1)
 * is closed as by frame_finish: state zeroed, generation + 1, nothing emitted. */
int dfn2_prepost_frame_bypass(DFN2Prepost *p);

/* The enhanced spectrum (n_bins bins) of the frame just emitted, at the input's scale: zeros while `*valid`
 * (may be NULL) reports 0 (the first latency_frames() calls). -1 on NULL or with a frame open. Pushing it
 * into an AcIstft every hop, valid or not, needs AC_OLA_ACCUMULATE: a zero spectrum then yields zeros. */
int dfn2_prepost_post_process_freq(DFN2Prepost *p, float *re, float *im,
                                   int *valid);

/* Source frame (from 0) of the last emission, written to *frame; -1 on NULL or before the first emission. */
int dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame);

/* Accelerator callbacks: infer() gets the open frame's views, returns 0 once every output is written; reset()
 * optional; a model that infers must publish an io_descriptor passing dfn2_model_io_descriptor_validate(). */
typedef struct DFN2Model {
    void *user;
    int (*infer)(void *user, const DFN2PrepostInputs *inputs,
                 DFN2PrepostOutputs *outputs);
    void (*reset)(void *user);
    const DFN2ModelIoDescriptor *io_descriptor;   /* NULL = not published */
} DFN2Model;

/* Drive one open frame: open, infer(), finish; a nonzero infer() or refused commit takes frame_bypass() (clocks
 * advance once). 1 committed, 0 skipped, -1 contract error (NULL args, no infer, bad io_descriptor, no frame
 * open). A warm-up frame (N == 1) also returns 1 when infer() succeeds, although its heads are discarded. */
int dfn2_model_run_frame(const DFN2Model *model, DFN2Prepost *p);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_PREPOST_H */
