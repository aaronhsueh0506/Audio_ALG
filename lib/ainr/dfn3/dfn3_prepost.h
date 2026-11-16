/* DeepFilterNet3 pre/post class: the spectra-to-tensors path around the accelerator, for the DFN3Grid in its config.
 * Frequency domain only: the application owns the AcStft/AcIstft, set up as the framing contract of dfn3_process.h
 * states (spec_scale = grid.stft_scale, ispec_scale = grid.stft_unscale); a wrong scale is not detectable here. The
 * caller supplies the pool get_mem_size() reports; the recurrent state is the application's, optionally registered.
 * M = grid.mask_lookahead. Every hop runs the same flow:
 *     ac_stft_push_planar(stft, in_samples, re, im);
 *     if (dfn3_prepost_pre_process_freq(p, re, im) == 1) {   // 0 for the first M hops: no job, nothing open
 *         dfn3_prepost_frame_open(p, &in, &out);             // views into the pool
 *         my_npu_run(&in, &out);                             // fills `out`
 *         dfn3_prepost_frame_finish(p);                      // -1: dfn3_prepost_frame_bypass(p)
 *     }                                                      // dfn3_model_run_frame() wraps the three calls
 *     dfn3_prepost_post_process_freq(p, out_re, out_im, NULL);
 *     ac_istft_push_planar(istft, out_re, out_im, out_samples);   // every hop (AC_OLA_ACCUMULATE)
 * The first M hops after init or reset have no complete graph window: the accelerator is not called for them and
 * the post-processed spectrum is zero with valid 0. M == 0: every hop runs. */

#ifndef DFN3_PREPOST_H
#define DFN3_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "dfn3_model_io.h"   /* graph geometry, DFN3ModelIOState, dfn3_process.h (grid, DFN3State) */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN3_PREPOST_DESCRIPTOR_VERSION 1u
#define DFN3_PREPOST_CARVE_VERSION      9u   /* bump on any pool size, carve order or descriptor change */

#define DFN3_PREPOST_ALIGNMENT 16u   /* the shared carve alignment (audiolib mem_align.h) */

static inline size_t dfn3_prepost_erb_window_elements(const DFN3Grid *grid)
{
    return (size_t)DFN3_MODEL_INPUT_FRAMES * (size_t)grid->n_erb;
}
static inline size_t dfn3_prepost_spec_window_elements(const DFN3Grid *grid)
{
    return (size_t)2 * DFN3_MODEL_INPUT_FRAMES * (size_t)grid->df_bins;
}
static inline size_t dfn3_prepost_erb_mask_elements(const DFN3Grid *grid)
{
    return (size_t)grid->n_erb;
}
static inline size_t dfn3_prepost_coefs_elements(const DFN3Grid *grid)
{
    return (size_t)grid->df_bins * (size_t)grid->df_order * 2u;
}

/* The ERB matrices and the registered state are borrowed pointers: they must outlive the instance. */
typedef struct DFN3PrepostConfig {
    DFN3Grid     grid;          /* from dfn3_grid_make(); copied at init */
    const float *erb_fwd;       /* required, bin-major [n_bins][n_erb] */
    const float *erb_inv;       /* required, band-major [n_erb][n_bins] */
    float        atten_lim_db;  /* 0 disables the attenuation limit */
    /* Optional registration of the recurrent state (h_gru, df_convp_history): both pairs set, or all four values
     * NULL/0 (a pointer with a count of 0 is refused). Zeroed at init and reset, finite-checked at every commit. */
    float       *gru_hidden;
    size_t       gru_hidden_elements;
    float       *df_convp_history;
    size_t       df_convp_history_elements;
} DFN3PrepostConfig;

/* Fixed 32-byte descriptor; _init_ex refuses a pool sized for another grid or build (registration is not part). */
typedef struct DFN3PrepostMemReq {
    uint32_t descriptor_version;  /* DFN3_PREPOST_DESCRIPTOR_VERSION */
    uint32_t layout_version;      /* DFN3_MODEL_IO_LAYOUT_VERSION */
    uint32_t reserved0;           /* must be 0: _init_ex refuses another value */
    uint32_t build_flags_hash;    /* FNV-1a-32 over the grid, layout, carve and feature versions */
    uint32_t alignment;
    uint32_t reserved;            /* 0 */
    uint64_t bytes;
} DFN3PrepostMemReq;

typedef struct DFN3Prepost DFN3Prepost;

/* Accelerator inputs at stable addresses: read-only pool windows and the registered state (NULL and 0 if none; the
 * counts are the application's). `*_next` outputs alias the state: read each state input before any state output. */
typedef struct DFN3PrepostInputs {
    const float *erb_window;        /* 'erb'  (1,1,3,n_erb) */
    const float *spec_window;       /* 'spec' (1,2,3,df_bins) */
    const float *gru_hidden;        /* 'h_gru' (1,rows,1,hidden) */
    const float *df_convp_window;   /* 'df_convp_history' (1,channels,window,df_bins) */
    size_t erb_window_elements;
    size_t spec_window_elements;
    size_t gru_hidden_elements;
    size_t df_convp_window_elements;
} DFN3PrepostInputs;

/* Accelerator-writable outputs: the two heads (NaN-filled by frame_open()) and the next-state tensors. */
typedef struct DFN3PrepostOutputs {
    float *erb_mask;                /* 'erb_mask'  [n_erb] */
    float *coefs;                   /* 'df_coefs'  [df_bins][df_order][2]: bin, tap, re/im */
    float *gru_hidden_next;         /* h_gru_out */
    float *df_convp_window_next;    /* df_convp_history_out */
    size_t erb_mask_elements;
    size_t coefs_elements;
} DFN3PrepostOutputs;

/* Config for `grid`: no ERB matrices or registration, atten_lim_db 0. Returns 0, or -1 on NULL or an invalid grid. */
int dfn3_prepost_config_defaults(DFN3PrepostConfig *cfg, const DFN3Grid *grid);

/* Pool size; -1 (req untouched) on NULL, missing ERB matrices, non-finite atten_lim_db, bad grid or registration. */
int dfn3_prepost_get_mem_size(const DFN3PrepostConfig *cfg, DFN3PrepostMemReq *req);

/* Construct in caller memory (16-byte aligned, `bytes` >= the reported size). NULL on a bad config or an undersized
 * or misaligned pool, leaving the pool and the registered buffers untouched. Starts reset. */
DFN3Prepost *dfn3_prepost_init(void *pool, size_t bytes, const DFN3PrepostConfig *cfg);

/* As _init, plus the stale-pool gate against `expected` (NULL = none). */
DFN3Prepost *dfn3_prepost_init_ex(void *pool, size_t bytes, const DFN3PrepostConfig *cfg,
                                  const DFN3PrepostMemReq *expected);

/* Host path: get_mem_size + aligned allocation + init. */
DFN3Prepost *dfn3_prepost_create(const DFN3PrepostConfig *cfg);

/* Frees only what _create allocated; NULL-safe, a no-op on an _init instance. */
void dfn3_prepost_destroy(DFN3Prepost *p);

/* Zero every ring and the registered state, drop any open frame, restart the normalisers and the enhance clock;
 * advances the generation. The application resets its own AcStft / AcIstft. */
void dfn3_prepost_reset(DFN3Prepost *p);

/* State generation (see DFN3ModelIOState): 1 after init, +1 per reset and per refused frame_finish(); 0 for NULL.
 * A runtime with its own state copy reloads quantized zeros on every change. */
unsigned int dfn3_prepost_state_generation(const DFN3Prepost *p);

/* Use other ERB matrices (borrowed; call again after an in-place rewrite). A swap consistent per source frame needs a
 * reset first. Returns 0, or -1 on NULL or frame open. */
int dfn3_prepost_set_erb_matrices(DFN3Prepost *p, const float *erb_fwd, const float *erb_inv);

/* Attenuation limit in dB (0 disables) for the enhance stage. Returns 0, or -1 on NULL, non-finite or frame open. */
int dfn3_prepost_set_atten_lim(DFN3Prepost *p, float atten_lim_db);

/* One analysed frame (n_bins bins). Returns 1 with a frame open from hop M on, 0 for the first M hops (no frame
 * open: the clock advances without heads, no job runs), -1 on NULL args, a non-finite spectrum or with a frame open.
 * Through an AcStft / AcIstft pair the audio trails the input by output_delay + 1 hops (frame t is ready after hop t +
 * output_delay, plus one framing hop). */
int dfn3_prepost_pre_process_freq(DFN3Prepost *p, const float *spec_re, const float *spec_im);

/* Dual input: `est` feeds the features and heads, `app` the noisy ring the mask, deep filter and attenuation limit
 * act on (est == app is pre_process_freq()). Returns as there. */
int dfn3_prepost_pre_process_freq_dual(DFN3Prepost *p, const float *est_re, const float *est_im,
                                       const float *app_re, const float *app_im);

/* Publish the open frame's accelerator boundary and NaN-fill the heads. Returns 0, or -1 if no frame is open. */
int dfn3_prepost_frame_open(DFN3Prepost *p, DFN3PrepostInputs *inputs,
                            DFN3PrepostOutputs *outputs);

/* Check heads and registered state for finiteness, then build the enhanced spectrum; after frame_open(). A non-finite
 * value zeroes the registered state, advances the generation and returns -1, frame left open: frame_bypass() or
 * frame_open() again. */
int dfn3_prepost_frame_finish(DFN3Prepost *p);

/* Give up on the open frame and pass the noisy spectrum through (unit ERB gains, a unit tap on the raw low bins).
 * Clocks advance, the recurrent state is left as is. Returns 0, or -1 if no frame is open. */
int dfn3_prepost_frame_bypass(DFN3Prepost *p);

/* The enhanced spectrum (n_bins bins) of the frame just emitted, at the input's scale. The output trails the input by
 * grid.output_delay frames; until then `*valid` (may be NULL) is 0 and the spectrum zero. Pushing it into an AcIstft
 * every hop, valid or not, needs AC_OLA_ACCUMULATE: a zero spectrum then yields zeros. Returns 0, or -1. */
int dfn3_prepost_post_process_freq(DFN3Prepost *p, float *re, float *im, int *valid);

/* Source frame (from 0) of the last emitted output: 0 and *frame after the first emission since reset, else -1. */
int dfn3_prepost_output_frame_index(const DFN3Prepost *p, long long *frame);

/* Accelerator callback: `infer` writes every output of the open frame and returns 0; `io_descriptor` is the graph
 * contract the runtime was built against. */
typedef struct DFN3Model {
    void *user;
    int (*infer)(void *user, const DFN3PrepostInputs *inputs,
                 DFN3PrepostOutputs *outputs);
    const DFN3ModelIoDescriptor *io_descriptor;   /* NULL = not published */
} DFN3Model;

/* Run the open frame: frame_open(), infer(), frame_finish(); a nonzero infer() or a refused commit takes
 * frame_bypass(), so the clocks advance once. Returns 1 when committed, 0 when bypassed, -1 on a contract error (NULL
 * arguments or infer, no frame open, an io_descriptor that is NULL or disagrees with the grid or layout version). */
int dfn3_model_run_frame(const DFN3Model *model, DFN3Prepost *p);

#ifdef __cplusplus
}
#endif

#endif /* DFN3_PREPOST_H */
