/* DeepFilterNet2 pre/post-processing class: one object owning everything between
 * the caller's audio and the accelerator's tensors.
 *
 *     if (dfn2_prepost_pre_process(p, in_hop) == 1) {
 *         dfn2_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *         my_npu_run(&in, &out);                     // NPU fills `out`
 *         dfn2_prepost_frame_commit(p);              // or _frame_skip()
 *     }
 *     dfn2_prepost_post_process(p, out_hop, &written);
 *
 * With L = DFN2_MASK_LOOKAHEAD and D = DFN2_DF_LOOKAHEAD, pre_process returns 0
 * on the first L hops (the graph window [t-(2-L) .. t+L] is not complete yet;
 * do not run the accelerator) and 1 afterwards. Latency is 1 + L + D hops: one
 * of framing plus L + D of model lookahead (3 hops, 1536 samples, 32 ms at
 * 48 kHz for the shipped L = D = 1); post_process reports written = 0 for the
 * first L + D hops. Flush the tail with L + D more zero hops and run the
 * accelerator on them. L and D are build-time constants (dfn2_process.h) and
 * must equal the pair the graph was trained and exported for.
 *
 * DFN2_IO_FREQ spectra are torch.stft(normalized=True): the rfft of the
 * sqrt-Hann-windowed block times 1/sqrt(DFN2_N_FFT). AIAEC spectra are
 * unnormalised: scale by 1.0f / sqrtf(DFN2_N_FFT) into DFN2 and by
 * sqrtf(DFN2_N_FFT) out of it (a wrong scale is not detectable here).
 * The FftHandle, window and ERB matrices are caller-owned; window == NULL carves
 * a sqrt-Hann table in the pool. */

#ifndef DFN2_PREPOST_H
#define DFN2_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_model_io.h"   /* model geometry, DFN2ModelIOState, dfn2_process.h (grid, DFN2State, FftHandle) */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN2_PREPOST_DESCRIPTOR_VERSION 1u
/* Folded into build_flags_hash; bumped on any change of pool size, carve order or
 * descriptor content. */
#define DFN2_PREPOST_CARVE_VERSION      7u

/* One shared alignment for every module (audiolib mem_align.h). */
#define DFN2_PREPOST_ALIGNMENT 16u

/* Flat element counts at the accelerator boundary, for binding tensors by count. */
#define DFN2_PREPOST_ERB_WINDOW_ELEMENTS \
    (DFN2_MODEL_INPUT_FRAMES * DFN2_N_ERB)
#define DFN2_PREPOST_SPEC_WINDOW_ELEMENTS \
    (2 * DFN2_MODEL_INPUT_FRAMES * DFN2_DF_BINS)
#define DFN2_PREPOST_GRU_HIDDEN_ELEMENTS \
    (DFN2_MODEL_GRU_ROWS * DFN2_MODEL_GRU_HIDDEN)
#define DFN2_PREPOST_CONVP_WINDOW_ELEMENTS \
    (DFN2_MODEL_ENCODER_CHANNELS * DFN2_MODEL_DF_PATHWAY_WINDOW * \
     DFN2_DF_BINS)
#define DFN2_PREPOST_ERB_MASK_ELEMENTS   DFN2_N_ERB
#define DFN2_PREPOST_COEFS_ELEMENTS      (DFN2_DF_BINS * DFN2_DF_ORDER * 2)
#define DFN2_PREPOST_ALPHA_ELEMENTS      1

/* Which side of the transform the caller works on; fixed at init because it
 * decides the pool size (DFN2_IO_TIME also carves the output staging and the
 * framing objects). */
typedef enum DFN2IoMode {
    /* Spectrum in, spectrum out; the caller owns the transform. `fft` may be NULL. */
    DFN2_IO_FREQ = 0,
    /* Hop in, hop out; the class owns framing and calls the caller's FftHandle. */
    DFN2_IO_TIME = 1
} DFN2IoMode;

typedef struct DFN2PrepostConfig {
    int          io_mode;       /* a DFN2IoMode                             */
    FftHandle   *fft;           /* DFN2_IO_TIME: required, borrowed, sized for DFN2_N_FFT */
    const float *window;        /* optional, DFN2_WIN_LEN entries, borrowed (DFN2_IO_TIME); NULL = sqrt-Hann table carved in the pool */
    const float *erb_fwd;       /* required, borrowed, bin-major [DFN2_N_BINS][DFN2_N_ERB] */
    const float *erb_inv;       /* required, borrowed, band-major [DFN2_N_ERB][DFN2_N_BINS] */
    float        atten_lim_db;  /* 0 disables the attenuation limit         */
} DFN2PrepostConfig;

/* Fixed 32-byte descriptor; a pool sized for another io_mode, grid or build is
 * refused by _init_ex. */
typedef struct DFN2PrepostMemReq {
    uint32_t descriptor_version;  /* DFN2_PREPOST_DESCRIPTOR_VERSION        */
    uint32_t layout_version;      /* DFN2_MODEL_IO_LAYOUT_VERSION           */
    uint32_t io_mode;             /* the resolved DFN2IoMode                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, geometry, feature version, io_mode */
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

/* Accelerator inputs, pointing into the pool for the life of the instance. The
 * state tensors never change address and the matching `*_next` outputs alias them
 * (read every state input before writing any state output). The feature windows
 * are read-only. Graph input names are given for binding by name. */
typedef struct DFN2PrepostInputs {
    /* 'erb'  (1,1,DFN2_MODEL_INPUT_FRAMES,DFN2_N_ERB) */
    const float (*erb_window)[DFN2_N_ERB];
    /* 'spec' (1,2,DFN2_MODEL_INPUT_FRAMES,DFN2_DF_BINS) */
    const float (*spec_window)[DFN2_MODEL_INPUT_FRAMES][DFN2_DF_BINS];
    /* 'h_gru' (1,DFN2_MODEL_GRU_ROWS,1,DFN2_MODEL_GRU_HIDDEN); rows start at
     * DFN2_MODEL_GRU_ROW_ENCODER / _ERB / _DF */
    const float (*gru_hidden)[DFN2_MODEL_GRU_HIDDEN];
    /* 'df_convp_history'
     * (1,DFN2_MODEL_ENCODER_CHANNELS,DFN2_MODEL_DF_PATHWAY_WINDOW,
     *  DFN2_DF_BINS) */
    const float (*df_convp_window)[DFN2_MODEL_DF_PATHWAY_WINDOW]
                                  [DFN2_DF_BINS];
    size_t erb_window_elements;
    size_t spec_window_elements;
    size_t gru_hidden_elements;
    size_t df_convp_window_elements;
} DFN2PrepostInputs;

/* Accelerator-writable outputs: the three heads and the two next-state tensors.
 * frame_inputs() NaN-fills the heads so frame_commit() detects a partial write.
 * The next-state tensors alias the inputs' state (nothing copied, not prefilled).
 * `coefs` is (DFN2_DF_BINS, DFN2_DF_ORDER, 2): bin, tap, re/im. `alpha` is a
 * one-element tensor. */
typedef struct DFN2PrepostOutputs {
    float *erb_mask;                /* 'erb_mask'  [DFN2_N_ERB]             */
    float *coefs;                   /* 'df_coefs'  [bins][order][2]         */
    float *alpha;                   /* 'df_alpha'  [1]                      */
    float (*gru_hidden_next)[DFN2_MODEL_GRU_HIDDEN];          /* h_gru_out */
    float (*df_convp_window_next)[DFN2_MODEL_DF_PATHWAY_WINDOW]
                                 [DFN2_DF_BINS];  /* df_convp_history_out  */
    size_t erb_mask_elements;
    size_t coefs_elements;
    size_t alpha_elements;
    size_t gru_hidden_elements;
    size_t df_convp_window_elements;
} DFN2PrepostOutputs;

/* ---- config / sizing ------------------------------------------------- */

/* Fill a config with this build's defaults. Returns 0, or -1 on NULL or an
 * unknown io_mode. fft, window, erb_fwd and erb_inv are left NULL;
 * atten_lim_db is 0 (disabled). */
int dfn2_prepost_config_defaults(DFN2PrepostConfig *cfg, int io_mode);

/* Exact pool size. Returns -1 (req untouched) on NULL, unknown io_mode, missing
 * ERB matrices, non-finite atten_lim_db, or DFN2_IO_TIME without an FftHandle
 * sized for DFN2_N_FFT. */
int dfn2_prepost_get_mem_size(const DFN2PrepostConfig *cfg,
                              DFN2PrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config or a misaligned or
 * undersized pool. Starts reset. */
DFN2Prepost *dfn2_prepost_init(void *pool, size_t bytes,
                               const DFN2PrepostConfig *cfg);

/* As _init, plus the stale-pool gate; `expected` NULL behaves as _init. */
DFN2Prepost *dfn2_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN2PrepostConfig *cfg,
                                  const DFN2PrepostMemReq *expected);

/* get_mem_size + aligned allocation + init (host path). */
DFN2Prepost *dfn2_prepost_create(const DFN2PrepostConfig *cfg);

/* Frees only what _create allocated; NULL-safe, a no-op on an _init instance.
 * Never touches the caller's fft, window or ERB matrices. */
void dfn2_prepost_destroy(DFN2Prepost *p);

/* Zero every recurrent/ring/hidden tensor, drop any open frame and reset the
 * framing, normaliser and compose-clock states. Config, borrowed pointers and
 * pool are untouched. Advances the state generation. */
void dfn2_prepost_reset(DFN2Prepost *p);

/* Generation of the recurrent state: 1 after init, plus one per reset and per
 * refused frame_commit(), modulo 2^32; 0 for NULL. A runtime holding a
 * fixed-point copy of the state refills it with quantized zeros when it changes. */
unsigned int dfn2_prepost_state_generation(const DFN2Prepost *p);

/* ---- accessors ------------------------------------------------------- */

int dfn2_prepost_hop_size(const DFN2Prepost *p);   /* or -1 */
int dfn2_prepost_num_bins(const DFN2Prepost *p);   /* or -1 */
int dfn2_prepost_io_mode(const DFN2Prepost *p);    /* or -1 */
/* Frames of end-to-end lookahead the model costs on top of the framing:
 * DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD. -1 on NULL. */
int dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);
/* The graph contract this build binds. -1 on NULL. */
int dfn2_prepost_layout_version(const DFN2Prepost *p);

/* Point the instance at caller-loaded ERB matrices (erb_fwd.bin / erb_inv.bin from
 * export_erb_matrix.py --runtime-bins); borrowed, never copied. Between hops only
 * (refused with a frame open); a swap consistent per source frame needs a reset
 * first. Returns 0, or -1 on NULL or with a frame open. */
int dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd,
                                  const float *erb_inv);

/* Runtime attenuation limit in dB (0 disables), applied by the compose stage.
 * Between hops only. Returns 0, or -1 on NULL, a non-finite value, or with a
 * frame open. */
int dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db);

/* ---- per-hop stages -------------------------------------------------- */

/* DFN2_IO_TIME. One hop of noisy input (DFN2 is single-stream): runs the
 * analysis and feature front end, slides the graph window and advances the
 * compose clock. Returns the number of accelerator invocations this hop needs (0
 * on the first DFN2_MASK_LOOKAHEAD hops, 1 after), or -1 on NULL args, in
 * DFN2_IO_FREQ, or with a frame open. */
int dfn2_prepost_pre_process(DFN2Prepost *p, const float in_hop[DFN2_HOP_LEN]);

/* DFN2_IO_FREQ. One analysed frame at normalized=True scale: rfft of a
 * sqrt-Hann-windowed DFN2_N_FFT block on the DFN2_HOP_LEN grid. No transform is
 * done. Returns 0 on the first DFN2_MASK_LOOKAHEAD frames, 1 after, or -1. */
int dfn2_prepost_pre_process_freq(DFN2Prepost *p,
                                  const float spec_re[DFN2_N_BINS],
                                  const float spec_im[DFN2_N_BINS]);

/* DFN2_IO_FREQ, dual input. `est` feeds the estimation branch (features, graph
 * window, heads); `app` feeds the application branch (the noisy ring that the
 * mask, deep filter and attenuation limit act on). Same frame clock and scale;
 * est == app equals dfn2_prepost_pre_process_freq(). Returns 0 on the first
 * DFN2_MASK_LOOKAHEAD frames, 1 after, or -1 on NULL args, in DFN2_IO_TIME, or
 * with a frame open. */
int dfn2_prepost_pre_process_freq_dual(DFN2Prepost *p,
                                       const float est_re[DFN2_N_BINS],
                                       const float est_im[DFN2_N_BINS],
                                       const float app_re[DFN2_N_BINS],
                                       const float app_im[DFN2_N_BINS]);

/* Publish the current frame's accelerator boundary. The three heads are
 * NaN-prefilled so an unwritten head is caught by frame_commit; the recurrent
 * state is not. Pointers are into the pool and the state pointers are constant.
 * Returns 0, or -1 if no frame is open. */
int dfn2_prepost_frame_inputs(DFN2Prepost *p, DFN2PrepostInputs *inputs,
                              DFN2PrepostOutputs *outputs);

/* Validate the heads and both recurrent-state tensors (all finite), then run the
 * compose stage and, in DFN2_IO_TIME, the synthesis. Requires a frame published
 * by frame_inputs(). On a non-finite value the recurrent tensors are zeroed, the
 * state generation advances, the frame stays open and -1 is returned; call
 * dfn2_prepost_frame_skip() or re-run via a fresh frame_inputs(). */
int dfn2_prepost_frame_commit(DFN2Prepost *p);

/* Take the identity for the current frame (unit ERB mask, alpha 0: the noisy
 * spectrum passes through). Clocks advance and the recurrent state is left as is.
 * Used after a failed accelerator run. Returns 0, or -1 if no frame is open. */
int dfn2_prepost_frame_skip(DFN2Prepost *p);

/* DFN2_IO_TIME. Emit this hop's output: DFN2_HOP_LEN samples, zero-filled during
 * the first DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD warm-up hops; `*written`
 * (may be NULL) reports 0 or DFN2_HOP_LEN. Returns 0, or -1. */
int dfn2_prepost_post_process(DFN2Prepost *p, float out_hop[DFN2_HOP_LEN],
                              int *written);

/* DFN2_IO_FREQ. The enhanced spectrum of the frame just emitted, at the input's
 * normalized=True scale. Zeros during the DFN2_MASK_LOOKAHEAD +
 * DFN2_DF_LOOKAHEAD warm-up frames, where `*valid` (may be NULL) reports 0.
 * Returns 0, or -1. */
int dfn2_prepost_post_process_freq(DFN2Prepost *p, float re[DFN2_N_BINS],
                                   float im[DFN2_N_BINS], int *valid);

/* Source frame (counted from 0) of the last emitted output: returns 0 and writes
 * *frame once an output has been emitted since reset; -1 on NULL or before the
 * first emission. */
int dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame);

/* ---- Accelerator callback boundary ----
 * A hosting pipeline owns a DFN2Model: `infer` receives the published tensor views
 * of the open frame and returns 0 when it wrote every output; `reset` is optional;
 * `io_descriptor` is the graph contract the runtime was built against (a model
 * that infers must publish one, and the host refuses one that
 * dfn2_model_io_descriptor_validate() rejects). */
typedef struct DFN2Model {
    void *user;
    int (*infer)(void *user, const DFN2PrepostInputs *inputs,
                 DFN2PrepostOutputs *outputs);
    void (*reset)(void *user);
    const DFN2ModelIoDescriptor *io_descriptor;   /* NULL = not published */
} DFN2Model;

/* Drive one open frame: frame_inputs(), infer(), frame_commit(). A nonzero infer()
 * result or a refused commit takes the frame with frame_skip(), so the clocks
 * advance exactly once. Returns 1 when the heads were committed, 0 when skipped,
 * -1 on a contract error (NULL model or instance, no infer callback, invalid
 * io_descriptor -- NULL, or another graph or lookahead pair -- or no frame open). */
int dfn2_model_run_frame(const DFN2Model *model, DFN2Prepost *p);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_PREPOST_H */
