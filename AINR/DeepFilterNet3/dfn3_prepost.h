/* DeepFilterNet3 pre/post-processing class: one object owning everything between
 * the caller's audio and the accelerator's tensors.
 *
 *     if (dfn3_prepost_pre_process(p, in_hop) == 1) {
 *         dfn3_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *         my_npu_run(&in, &out);                     // NPU fills `out`
 *         dfn3_prepost_frame_commit(p);              // or _frame_skip()
 *     }
 *     dfn3_prepost_post_process(p, out_hop, &written);
 *
 * pre_process returns 0 on the first DFN3_MASK_LOOKAHEAD hops (the graph window
 * [t-(2-MASK), ..., t+MASK] of the first head is not complete; do not run the
 * accelerator) and 1 afterwards. Latency is 1 + DFN3_OUTPUT_DELAY hops: 1 of
 * framing plus the model lookahead max(DFN3_MASK_LOOKAHEAD, DFN3_DF_LOOKAHEAD),
 * 2 hops (1024 samples, 21.33 ms at 48 kHz) for the shipped 1/1. post_process
 * reports written = 0 on the first DFN3_OUTPUT_DELAY hops only (with
 * DF_LOOKAHEAD > MASK_LOOKAHEAD the accelerator runs on some hops that still
 * emit nothing). Flush the tail with DFN3_OUTPUT_DELAY zero hops while running
 * the accelerator on every hop after the first MASK; an exact finite-clip flush
 * must zero-pad the model features (as training does), not recompute normalised
 * features from a zero spectrum.
 *
 * DFN3_IO_FREQ spectra are torch.stft(normalized=True): the rfft of the
 * sqrt-Hann-windowed block times 1/sqrt(DFN3_N_FFT). AIAEC spectra are
 * unnormalised: scale by 1.0f / sqrtf(DFN3_N_FFT) into DFN3 and by
 * sqrtf(DFN3_N_FFT) out of it (a wrong scale is not detectable here).
 * The FftHandle, window, ERB matrices and (DFN3_IO_TIME) the synthesis AcIstft are
 * caller-owned; window == NULL carves a sqrt-Hann table in the pool for the
 * analysis object.
 *
 * Inverse FFT and overlap-add are audiolib's AcIstft, never this class's. Two ways
 * to chain it:
 *
 *     // DFN3_IO_TIME: hand the class the caller's AcIstft; commit pushes into it.
 *     AcStftConfig sc;  dfn3_framing_config(&sc, fft, NULL);
 *     AcIstft *istft = ac_istft_init(mem, ac_istft_get_mem_size(&sc), &sc);
 *     cfg.istft = istft;                       // ... dfn3_prepost_init(...)
 *     dfn3_prepost_post_process(p, out_hop, &written);   // hop out
 *
 *     // DFN3_IO_FREQ: spectrum out, the caller pushes its own AcIstft.
 *     dfn3_prepost_post_process_freq(p, re, im, &valid);
 *     if (valid) ac_istft_push_planar(istft, re, im, out_hop);
 *
 * The class resets a borrowed AcIstft together with its own state. */

#ifndef DFN3_PREPOST_H
#define DFN3_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "dfn3_model_io.h"   /* model geometry, DFN3ModelIOState, dfn3_process.h (grid, DFN3State, FftHandle) */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN3_PREPOST_DESCRIPTOR_VERSION 1u
/* Folded into build_flags_hash; bumped on any change of pool size, carve order or
 * descriptor content. */
#define DFN3_PREPOST_CARVE_VERSION      8u

/* One shared alignment for every module (audiolib mem_align.h). */
#define DFN3_PREPOST_ALIGNMENT 16u

/* Flat element counts at the accelerator boundary, for binding tensors by count. */
#define DFN3_PREPOST_ERB_WINDOW_ELEMENTS \
    (DFN3_MODEL_INPUT_FRAMES * DFN3_N_ERB)
#define DFN3_PREPOST_SPEC_WINDOW_ELEMENTS \
    (2 * DFN3_MODEL_INPUT_FRAMES * DFN3_DF_BINS)
#define DFN3_PREPOST_GRU_HIDDEN_ELEMENTS \
    (DFN3_MODEL_GRU_ROWS * DFN3_MODEL_GRU_HIDDEN)
#define DFN3_PREPOST_CONVP_WINDOW_ELEMENTS \
    (DFN3_MODEL_ENCODER_CHANNELS * DFN3_MODEL_DF_PATHWAY_WINDOW * \
     DFN3_DF_BINS)
#define DFN3_PREPOST_ERB_MASK_ELEMENTS   DFN3_N_ERB
#define DFN3_PREPOST_COEFS_ELEMENTS      (DFN3_DF_BINS * DFN3_DF_ORDER * 2)

/* Which side of the transform the caller works on; fixed at init because it
 * decides the pool size (DFN3_IO_TIME also carves the output staging and the
 * analysis object). */
typedef enum DFN3IoMode {
    /* Spectrum in, spectrum out; the caller owns the transform. `fft` may be NULL. */
    DFN3_IO_FREQ = 0,
    /* Hop in, hop out; the class owns the analysis framing and calls the caller's
     * FftHandle, and synthesises through the caller's AcIstft. */
    DFN3_IO_TIME = 1
} DFN3IoMode;

typedef struct DFN3PrepostConfig {
    int          io_mode;       /* a DFN3IoMode                             */
    FftHandle   *fft;           /* DFN3_IO_TIME: required, borrowed, sized for DFN3_N_FFT */
    const float *window;        /* optional, DFN3_WIN_LEN entries, borrowed (DFN3_IO_TIME); NULL = sqrt-Hann table carved in the pool */
    AcIstft     *istft;         /* DFN3_IO_TIME: required, borrowed, built from dfn3_framing_config() (same fft size, window values, ola, ispec_scale); ignored in DFN3_IO_FREQ */
    const float *erb_fwd;       /* required, borrowed, bin-major [DFN3_N_BINS][DFN3_N_ERB] */
    const float *erb_inv;       /* required, borrowed, band-major [DFN3_N_ERB][DFN3_N_BINS] */
    float        atten_lim_db;  /* 0 disables the attenuation limit         */
} DFN3PrepostConfig;

/* Fixed 32-byte descriptor; a pool sized for another io_mode, grid or build is
 * refused by _init_ex. */
typedef struct DFN3PrepostMemReq {
    uint32_t descriptor_version;  /* DFN3_PREPOST_DESCRIPTOR_VERSION        */
    uint32_t layout_version;      /* DFN3_MODEL_IO_LAYOUT_VERSION           */
    uint32_t io_mode;             /* the resolved DFN3IoMode                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, geometry, feature version, io_mode */
    uint32_t alignment;
    uint32_t reserved;            /* 0 */
    uint64_t bytes;
} DFN3PrepostMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DFN3PrepostMemReq) == 32,
               "DFN3PrepostMemReq must stay 32 bytes");
#endif
#endif

typedef struct DFN3Prepost DFN3Prepost;

/* Accelerator inputs, pointing into the pool for the life of the instance. The
 * state tensors never change address and the matching `*_next` outputs alias them
 * (read every state input before writing any state output). The feature windows
 * are read-only. Graph input names are given for binding by name. */
typedef struct DFN3PrepostInputs {
    /* 'erb'  (1,1,DFN3_MODEL_INPUT_FRAMES,DFN3_N_ERB) */
    const float (*erb_window)[DFN3_N_ERB];
    /* 'spec' (1,2,DFN3_MODEL_INPUT_FRAMES,DFN3_DF_BINS) */
    const float (*spec_window)[DFN3_MODEL_INPUT_FRAMES][DFN3_DF_BINS];
    /* 'h_gru' (1,DFN3_MODEL_GRU_ROWS,1,DFN3_MODEL_GRU_HIDDEN); rows start at
     * DFN3_MODEL_GRU_ROW_ENCODER / _ERB / _DF */
    const float (*gru_hidden)[DFN3_MODEL_GRU_HIDDEN];
    /* 'df_convp_history'
     * (1,DFN3_MODEL_ENCODER_CHANNELS,DFN3_MODEL_DF_PATHWAY_WINDOW,
     *  DFN3_DF_BINS) */
    const float (*df_convp_window)[DFN3_MODEL_DF_PATHWAY_WINDOW]
                                  [DFN3_DF_BINS];
    size_t erb_window_elements;
    size_t spec_window_elements;
    size_t gru_hidden_elements;
    size_t df_convp_window_elements;
} DFN3PrepostInputs;

/* Accelerator-writable outputs: the two heads and the two next-state tensors.
 * frame_inputs() NaN-fills the heads so frame_commit() detects a partial write.
 * The next-state tensors alias the inputs' state (nothing copied, not prefilled).
 * `coefs` is (DFN3_DF_BINS, DFN3_DF_ORDER, 2): bin, tap, re/im. There is no alpha
 * head. */
typedef struct DFN3PrepostOutputs {
    float *erb_mask;                /* 'erb_mask'  [DFN3_N_ERB]             */
    float *coefs;                   /* 'df_coefs'  [bins][order][2]         */
    float (*gru_hidden_next)[DFN3_MODEL_GRU_HIDDEN];          /* h_gru_out */
    float (*df_convp_window_next)[DFN3_MODEL_DF_PATHWAY_WINDOW]
                                 [DFN3_DF_BINS];  /* df_convp_history_out  */
    size_t erb_mask_elements;
    size_t coefs_elements;
    size_t gru_hidden_elements;
    size_t df_convp_window_elements;
} DFN3PrepostOutputs;

/* ---- config / sizing ------------------------------------------------- */

/* Fill a config with this build's defaults. Returns 0, or -1 on NULL or an
 * unknown io_mode. fft, window, istft, erb_fwd and erb_inv are left NULL;
 * atten_lim_db is 0 (disabled). */
int dfn3_prepost_config_defaults(DFN3PrepostConfig *cfg, int io_mode);

/* Exact pool size. Returns -1 (req untouched) on NULL, unknown io_mode, missing
 * ERB matrices, non-finite atten_lim_db, or DFN3_IO_TIME without an FftHandle
 * sized for DFN3_N_FFT or without an AcIstft that matches dfn3_framing_config(). */
int dfn3_prepost_get_mem_size(const DFN3PrepostConfig *cfg,
                              DFN3PrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config (a borrowed AcIstft whose
 * window differs from the analysis window included) or a misaligned or
 * undersized pool. Starts reset. */
DFN3Prepost *dfn3_prepost_init(void *pool, size_t bytes,
                               const DFN3PrepostConfig *cfg);

/* As _init, plus the stale-pool gate; `expected` NULL behaves as _init. */
DFN3Prepost *dfn3_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN3PrepostConfig *cfg,
                                  const DFN3PrepostMemReq *expected);

/* get_mem_size + aligned allocation + init (host path). */
DFN3Prepost *dfn3_prepost_create(const DFN3PrepostConfig *cfg);

/* Frees only what _create allocated; NULL-safe, a no-op on an _init instance.
 * Never frees the caller's fft, window, istft or ERB matrices. */
void dfn3_prepost_destroy(DFN3Prepost *p);

/* Zero every recurrent/ring/hidden tensor, drop any open frame and reset the
 * framing (a borrowed AcIstft included), normaliser and compose-clock states.
 * Config, borrowed pointers and pool are untouched. Advances the state
 * generation. */
void dfn3_prepost_reset(DFN3Prepost *p);

/* Generation of the recurrent state: 1 after init, plus one per reset and per
 * refused frame_commit(), modulo 2^32; 0 for NULL. A runtime holding a
 * fixed-point copy of the state refills it with quantized zeros when it changes. */
unsigned int dfn3_prepost_state_generation(const DFN3Prepost *p);

/* ---- accessors ------------------------------------------------------- */

int dfn3_prepost_hop_size(const DFN3Prepost *p);   /* or -1 */
int dfn3_prepost_num_bins(const DFN3Prepost *p);   /* or -1 */
int dfn3_prepost_io_mode(const DFN3Prepost *p);    /* or -1 */
/* Frames of end-to-end lookahead the model costs on top of the framing:
 * DFN3_OUTPUT_DELAY = max(DFN3_MASK_LOOKAHEAD, DFN3_DF_LOOKAHEAD). -1 on NULL. */
int dfn3_prepost_model_lookahead_frames(const DFN3Prepost *p);
/* The graph contract this build binds. -1 on NULL. */
int dfn3_prepost_layout_version(const DFN3Prepost *p);

/* Point the instance at caller-loaded ERB matrices (erb_fwd.bin / erb_inv.bin from
 * export_erb_matrix.py --runtime-bins); borrowed, never copied. Between hops only
 * (refused with a frame open); a swap consistent per source frame needs a reset
 * first. Returns 0, or -1 on NULL or with a frame open. */
int dfn3_prepost_set_erb_matrices(DFN3Prepost *p, const float *erb_fwd,
                                  const float *erb_inv);

/* Runtime attenuation limit in dB (0 disables), applied by the compose stage.
 * Between hops only. Returns 0, or -1 on NULL, a non-finite value, or with a
 * frame open. */
int dfn3_prepost_set_atten_lim(DFN3Prepost *p, float atten_lim_db);

/* ---- per-hop stages -------------------------------------------------- */

/* DFN3_IO_TIME. One hop of noisy input (DFN3 is single-stream): runs the
 * analysis and feature front end, slides the graph window and advances the
 * compose clock. Returns the number of accelerator invocations this hop needs (0
 * on the first DFN3_MASK_LOOKAHEAD hops, 1 after), or -1 on NULL args, in
 * DFN3_IO_FREQ, or with a frame open. */
int dfn3_prepost_pre_process(DFN3Prepost *p, const float in_hop[DFN3_HOP_LEN]);

/* DFN3_IO_FREQ. One analysed frame at normalized=True scale: rfft of a
 * sqrt-Hann-windowed DFN3_N_FFT block on the DFN3_HOP_LEN grid. No transform is
 * done. Returns 0 on the first DFN3_MASK_LOOKAHEAD frames, 1 after, or -1. */
int dfn3_prepost_pre_process_freq(DFN3Prepost *p,
                                  const float spec_re[DFN3_N_BINS],
                                  const float spec_im[DFN3_N_BINS]);

/* DFN3_IO_FREQ, dual input. `est` feeds the estimation branch (features, graph
 * window, heads); `app` feeds the application branch (the noisy ring that the
 * mask, deep filter and attenuation limit act on). Same frame clock and scale;
 * est == app equals dfn3_prepost_pre_process_freq(). Returns 0 on the first
 * DFN3_MASK_LOOKAHEAD frames, 1 after, or -1 on NULL args, in DFN3_IO_TIME, or
 * with a frame open. */
int dfn3_prepost_pre_process_freq_dual(DFN3Prepost *p,
                                       const float est_re[DFN3_N_BINS],
                                       const float est_im[DFN3_N_BINS],
                                       const float app_re[DFN3_N_BINS],
                                       const float app_im[DFN3_N_BINS]);

/* Publish the current frame's accelerator boundary. The two heads are
 * NaN-prefilled so an unwritten head is caught by frame_commit; the recurrent
 * state is not. Pointers are into the pool and the state pointers are constant.
 * Returns 0, or -1 if no frame is open. */
int dfn3_prepost_frame_inputs(DFN3Prepost *p, DFN3PrepostInputs *inputs,
                              DFN3PrepostOutputs *outputs);

/* Validate the heads and both recurrent-state tensors (all finite), then run the
 * compose stage and, in DFN3_IO_TIME, the synthesis. Requires a frame published
 * by frame_inputs(). On a non-finite value the recurrent tensors are zeroed, the
 * state generation advances, the frame stays open and -1 is returned; call
 * dfn3_prepost_frame_skip() or re-run via a fresh frame_inputs(). */
int dfn3_prepost_frame_commit(DFN3Prepost *p);

/* Take the identity for the current frame: unit ERB gains on the high bins and a
 * unit center DF tap on the raw low bins, so every band passes through at the same
 * source frame. The lookahead clock advances and the recurrent state is left as
 * is. Returns 0, or -1 if no frame is open. */
int dfn3_prepost_frame_skip(DFN3Prepost *p);

/* DFN3_IO_TIME. Emit this hop's output: DFN3_HOP_LEN samples, zero-filled during
 * the DFN3_OUTPUT_DELAY warm-up hops; `*written` (may be NULL) reports 0 or
 * DFN3_HOP_LEN.
 * Returns 0, or -1. */
int dfn3_prepost_post_process(DFN3Prepost *p, float out_hop[DFN3_HOP_LEN],
                              int *written);

/* DFN3_IO_FREQ. The enhanced spectrum of the frame just emitted, at the input's
 * normalized=True scale. Zeros during the DFN3_OUTPUT_DELAY warm-up frames,
 * where `*valid` (may be NULL) reports 0. Returns 0, or -1. */
int dfn3_prepost_post_process_freq(DFN3Prepost *p, float re[DFN3_N_BINS],
                                   float im[DFN3_N_BINS], int *valid);

/* Source frame (counted from 0) of the last emitted output: returns 0 and writes
 * *frame once an output has been emitted since reset; -1 on NULL or before the
 * first emission. */
int dfn3_prepost_output_frame_index(const DFN3Prepost *p, long long *frame);

/* ---- Accelerator callback boundary ----
 * A hosting pipeline owns a DFN3Model: `infer` receives the published tensor views
 * of the open frame and returns 0 when it wrote every output; `reset` is optional;
 * `io_descriptor` is the graph contract the runtime was built against (a model
 * that infers must publish one, and the host refuses one that
 * dfn3_model_io_descriptor_validate() rejects). */
typedef struct DFN3Model {
    void *user;
    int (*infer)(void *user, const DFN3PrepostInputs *inputs,
                 DFN3PrepostOutputs *outputs);
    void (*reset)(void *user);
    const DFN3ModelIoDescriptor *io_descriptor;   /* NULL = not published */
} DFN3Model;

/* Drive one open frame: frame_inputs(), infer(), frame_commit(). A nonzero infer()
 * result or a refused commit takes the frame with frame_skip(), so the clocks
 * advance exactly once. Returns 1 when the heads were committed, 0 when skipped,
 * -1 on a contract error (NULL model or instance, no infer callback, invalid
 * io_descriptor, or no frame open). */
int dfn3_model_run_frame(const DFN3Model *model, DFN3Prepost *p);

#ifdef __cplusplus
}
#endif

#endif /* DFN3_PREPOST_H */
