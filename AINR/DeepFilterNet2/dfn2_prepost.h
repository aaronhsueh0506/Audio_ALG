/* ============================================================
 * DeepFilterNet2 pre/post-processing class.
 *
 * ONE object owning everything between the caller's audio and the
 * accelerator's tensors, so an integrator never touches the framing state,
 * the ERB/complex feature front end, the deep-filter rings or the recurrent
 * caches directly:
 *
 *     if (dfn2_prepost_pre_process(p, in_hop) == 1) {
 *         dfn2_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *         my_npu_run(&in, &out);                     // NPU fills `out`
 *         dfn2_prepost_frame_commit(p);              // or _frame_skip()
 *     }
 *     dfn2_prepost_post_process(p, out_hop, &written);
 *
 * pre_process returns 1 on every hop but the very first, where it returns 0:
 * DFN2's analysis is center=False streaming, so one hop in is exactly one
 * frame, but the graph's [t-1, t, t+1] window has no right-hand neighbour
 * on hop 0 and MUST NOT be invoked there. Every later hop is one inference.
 *
 * WHY A THIRD TRANSLATION UNIT, not an addition to one of the two it uses:
 * dfn2_process.c and dfn2_model_io.c are deliberately independently linkable
 * -- AINR/tests/test_python_c_prepost_parity.py compiles dfn2_process.c
 * WITHOUT dfn2_model_io.c. A class placed in either would put an undefined
 * symbol in the other's parity build. This file composes them and neither of
 * them changes.
 *
 * Memory: both house lifecycles (audio_common/include/fft_wrapper.h,
 * "Static memory support").
 *   dfn2_prepost_get_mem_size(&cfg, &req);
 *   DFN2Prepost *p = dfn2_prepost_init(pool, req.bytes, &cfg);   // board
 *   DFN2Prepost *p = dfn2_prepost_create(&cfg);                  // host
 *   dfn2_prepost_destroy(p);   // frees only what create() allocated
 *
 * ============================================================
 * ⚠ SPECTRUM SCALE AT THE DFN2_IO_FREQ BOUNDARY -- READ BEFORE CHAINING
 * ============================================================
 * The spectrum this class consumes and produces in DFN2_IO_FREQ is
 * torch.stft(normalized=True), i.e. the rfft of the sqrt-Hann-windowed
 * DFN2_N_FFT block multiplied by 1/sqrt(DFN2_N_FFT) = 1/32. dfn2_analysis()
 * applies that factor and dfn2_synthesis() removes it with sqrt(DFN2_N_FFT).
 *
 * The AIAEC models (Align-ULCNet and friends) hand over the UNNORMALISED
 * rfft. Chaining an AIAEC spectrum straight into dfn2_prepost_pre_process_freq
 * therefore over-drives DFN2 by exactly sqrt(1024) = 32x, and the reverse
 * under-drives it by 32x. Nothing in either class can detect this: both sides
 * are dimensionally valid float spectra, the ERB features are EMA-normalised
 * so the mask still looks plausible, and only the output level is wrong.
 * A caller crossing that boundary must scale explicitly:
 *     aiaec_spectrum -> dfn2 :  multiply by 1.0f / sqrtf(DFN2_N_FFT)
 *     dfn2 -> aiaec_spectrum :  multiply by sqrtf(DFN2_N_FFT)
 * They are also on DIFFERENT GRIDS (48 kHz / 1024 / 512 here versus
 * 16 kHz / 512), so a resampler sits between them anyway -- do the scaling
 * where that resampler lives.
 *
 * ⚠ DFN2_ANALYSIS_SCALE is NOT this factor. That constant is internal to the
 * FEATURE branch (libDF's wnorm residual, applied only to the copy that feeds
 * the normalisers) and never touches the spectrum on the masking/ISTFT path.
 * Do not apply it at this boundary and do not conflate the two.
 *
 * ============================================================
 * ⚠ END-TO-END LATENCY
 * ============================================================
 * Two sources, and they add:
 *   1. WOLA framing. center=False streaming analysis/synthesis at
 *      win 1024 / hop 512.
 *   2. Model lookahead. DFN2 is a cascade: the mask head is centered on
 *      t-DFN2_MASK_LOOKAHEAD, and the deep filter needs the MASKED frame
 *      t+DFN2_DF_LOOKAHEAD before it can emit t. So the emitted frame is
 *      current - MASK_LOOKAHEAD - DF_LOOKAHEAD = current - 2, NOT
 *      current - max(1,1).
 *
 * Measured on the identity path (unit mask, alpha 0 -> perfect
 * reconstruction): the newest output sample trails the newest consumed input
 * sample by 3 hops = 1536 samples = 32.0 ms at 48 kHz. Of that, 1 hop
 * (10.67 ms) is the WOLA framing and 2 hops (21.33 ms) is the model
 * lookahead. dfn2_prepost_post_process() reports `written` = 0 for the first
 * two hops, which is where those 2 lookahead hops are paid.
 *
 * Flushing the tail: keep calling pre_process with a zero hop (or a zero
 * spectrum) for MASK_LOOKAHEAD + DF_LOOKAHEAD = 2 more hops. The accelerator
 * must run on those hops too -- that is exactly the zero padding the trainer
 * used, and this class will not fabricate heads on its behalf.
 *
 * ============================================================
 * ⚠ WINDOW OWNERSHIP -- a documented deviation from the house rule
 * ============================================================
 * The FftHandle and the ERB matrices are CALLER-OWNED and BORROWED: this
 * class never creates or destroys either, and the ERB pointers may be swapped
 * between hops with dfn2_prepost_set_erb_matrices().
 *
 * The analysis/synthesis window CANNOT be borrowed here. DFN2State embeds its
 * window as `float window[DFN2_WIN_LEN]` BY VALUE and dfn2_state_init() fills
 * it with the sqrt-Hann table (dfn2_process.h, which carries a live parity
 * gate and is not ours to change). So `cfg.window`, when non-NULL, is COPIED
 * over that table at init and at every reset. The class still allocates and
 * frees nothing: the caller's table stays the caller's. Leave it NULL to use
 * the parity-tested built-in sqrt-Hann, which is what every shipped path
 * wants; override it only to share a table you have proved identical.
 * ============================================================ */

#ifndef DFN2_PREPOST_H
#define DFN2_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_model_io.h"   /* model geometry, DFN2ModelIOState, and
                              * (transitively) dfn2_process.h: the signal
                              * grid, DFN2State and FftHandle              */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN2_PREPOST_DESCRIPTOR_VERSION 1u

/* One shared alignment for every module (audio_common mem_align.h). */
#define DFN2_PREPOST_ALIGNMENT 16u

/* Flat element counts at the accelerator boundary. Stated as macros because
 * a caller wiring a runtime binds tensors by element count, and a count
 * recomputed at the call site is a place for a shape to drift silently. */
#define DFN2_PREPOST_ERB_WINDOW_ELEMENTS \
    (DFN2_MODEL_INPUT_FRAMES * DFN2_N_ERB)
#define DFN2_PREPOST_SPEC_WINDOW_ELEMENTS \
    (2 * DFN2_MODEL_INPUT_FRAMES * DFN2_DF_BINS)
#define DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS \
    (DFN2_MODEL_ENCODER_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN)
#define DFN2_PREPOST_ERB_HIDDEN_ELEMENTS \
    (DFN2_MODEL_ERB_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN)
#define DFN2_PREPOST_DF_HIDDEN_ELEMENTS \
    (DFN2_MODEL_DF_GRU_LAYERS * DFN2_MODEL_GRU_HIDDEN)
#define DFN2_PREPOST_CONVP_HISTORY_ELEMENTS \
    (DFN2_MODEL_ENCODER_CHANNELS * DFN2_MODEL_DF_PATHWAY_HISTORY * \
     DFN2_DF_BINS)
#define DFN2_PREPOST_ERB_MASK_ELEMENTS   DFN2_N_ERB
#define DFN2_PREPOST_COEFS_ELEMENTS      (DFN2_DF_BINS * DFN2_DF_ORDER * 2)
#define DFN2_PREPOST_ALPHA_ELEMENTS      1

/* Which side of the transform the caller works on. FIXED AT INIT because it
 * decides the pool size.
 *
 * ⚠ Honest note on how much it decides: DFN2State embeds its analysis
 * overlap, window and synthesis OLA buffers by value, and the compose stage
 * lives in the same struct, so a DFN2_IO_FREQ instance cannot shed them
 * without editing dfn2_process.h. The two modes therefore differ by only the
 * output hop staging. The enum is still fixed at init -- it is the pool's
 * identity, checked by _init_ex -- but do not expect the AIAEC models'
 * proportional saving here. */
typedef enum DFN2IoMode {
    /* Spectrum in, spectrum out. NO framing, windowing, FFT, iFFT or
     * overlap-add -- the caller owns the transform. This is the mode for
     * chaining (mind the scale note above), and the mode in which `fft` may
     * be NULL. */
    DFN2_IO_FREQ = 0,
    /* Hop in, hop out. The class owns framing, windowing, overlap-add and
     * calls the CALLER'S FftHandle for the transforms. */
    DFN2_IO_TIME = 1
} DFN2IoMode;

typedef struct DFN2PrepostConfig {
    int          io_mode;       /* a DFN2IoMode                             */
    FftHandle   *fft;           /* DFN2_IO_TIME: required, borrowed, sized
                                 * for DFN2_N_FFT                           */
    const float *window;        /* optional, DFN2_WIN_LEN entries, COPIED;
                                 * NULL = built-in sqrt-Hann (see header)   */
    const float *erb_fwd;       /* required, borrowed, bin-major
                                 * [DFN2_N_BINS][DFN2_N_ERB]                */
    const float *erb_inv;       /* required, borrowed, band-major
                                 * [DFN2_N_ERB][DFN2_N_BINS]                */
    float        atten_lim_db;  /* 0 disables the attenuation limit         */
} DFN2PrepostConfig;

/* Fixed 32-byte shape, same staleness-gate discipline as the pipelines'
 * memory descriptors: a pool sized for another io_mode, grid or build is
 * refused by _init_ex rather than reinterpreted. */
typedef struct DFN2PrepostMemReq {
    uint32_t descriptor_version;  /* DFN2_PREPOST_DESCRIPTOR_VERSION        */
    uint32_t layout_version;      /* DFN2_MODEL_IO_LAYOUT_VERSION           */
    uint32_t io_mode;             /* the resolved DFN2IoMode                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid + geometry +
                                   * feature version + io_mode              */
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

/* Read-only accelerator inputs. All pointers are into this instance's pool
 * and stay valid until the next pre_process. Graph input names of the shipped
 * split layout (export_onnx.py, DFN2_MODEL_IO_LAYOUT_VERSION 5) are given so
 * a runtime binds by name without reading the exporter. */
typedef struct DFN2PrepostInputs {
    /* 'erb'  (1,1,DFN2_MODEL_INPUT_FRAMES,DFN2_N_ERB) */
    const float (*erb_window)[DFN2_N_ERB];
    /* 'spec' (1,2,DFN2_MODEL_INPUT_FRAMES,DFN2_DF_BINS) */
    const float (*spec_window)[DFN2_MODEL_INPUT_FRAMES][DFN2_DF_BINS];
    /* 'h_encoder' (DFN2_MODEL_ENCODER_GRU_LAYERS,1,DFN2_MODEL_GRU_HIDDEN) */
    const float (*encoder_gru_hidden)[DFN2_MODEL_GRU_HIDDEN];
    /* 'h_erb' */
    const float (*erb_gru_hidden)[DFN2_MODEL_GRU_HIDDEN];
    /* 'h_df' */
    const float (*df_gru_hidden)[DFN2_MODEL_GRU_HIDDEN];
    /* 'df_convp_history'
     * (1,DFN2_MODEL_ENCODER_CHANNELS,DFN2_MODEL_DF_PATHWAY_HISTORY,
     *  DFN2_DF_BINS) */
    const float (*df_convp_history)[DFN2_MODEL_DF_PATHWAY_HISTORY]
                                   [DFN2_DF_BINS];
    size_t erb_window_elements;
    size_t spec_window_elements;
    size_t encoder_gru_hidden_elements;
    size_t erb_gru_hidden_elements;
    size_t df_gru_hidden_elements;
    size_t df_convp_history_elements;
} DFN2PrepostInputs;

/* Accelerator-writable outputs: the three heads and the four next-state
 * tensors. frame_inputs() fills every element with NaN so frame_commit()
 * detects a partial write instead of committing the previous frame's values.
 *
 * `coefs` is (DFN2_DF_BINS, DFN2_DF_ORDER, 2) with bin outermost, tap next
 * and re/im innermost -- exactly model.py's layout and exactly what
 * dfn2_compose_stream() expects. `alpha` is a one-element tensor, not a
 * scalar field, because the graph emits it as a tensor. */
typedef struct DFN2PrepostOutputs {
    float *erb_mask;                /* 'erb_mask'  [DFN2_N_ERB]             */
    float *coefs;                   /* 'df_coefs'  [bins][order][2]         */
    float *alpha;                   /* 'df_alpha'  [1]                      */
    float (*encoder_gru_hidden_next)[DFN2_MODEL_GRU_HIDDEN];  /* h_encoder_out */
    float (*erb_gru_hidden_next)[DFN2_MODEL_GRU_HIDDEN];      /* h_erb_out     */
    float (*df_gru_hidden_next)[DFN2_MODEL_GRU_HIDDEN];       /* h_df_out      */
    float (*df_convp_history_next)[DFN2_MODEL_DF_PATHWAY_HISTORY]
                                  [DFN2_DF_BINS];   /* df_convp_history_out */
    size_t erb_mask_elements;
    size_t coefs_elements;
    size_t alpha_elements;
    size_t encoder_gru_hidden_elements;
    size_t erb_gru_hidden_elements;
    size_t df_gru_hidden_elements;
    size_t df_convp_history_elements;
} DFN2PrepostOutputs;

/* ---- config / sizing ------------------------------------------------- */

/* Fill a config with this build's defaults. Returns 0, or -1 on NULL or an
 * unknown io_mode. `fft`, `window`, `erb_fwd` and `erb_inv` are left NULL for
 * the caller to set; atten_lim_db is left at 0 (disabled). */
int dfn2_prepost_config_defaults(DFN2PrepostConfig *cfg, int io_mode);

/* Exact pool size. Reject-first: -1 with *req untouched on a NULL argument,
 * an unknown io_mode, missing ERB matrices, a non-finite atten_lim_db, or
 * DFN2_IO_TIME without an FftHandle sized for DFN2_N_FFT. */
int dfn2_prepost_get_mem_size(const DFN2PrepostConfig *cfg,
                              DFN2PrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config, or a misaligned or
 * undersized pool. Starts reset. */
DFN2Prepost *dfn2_prepost_init(void *pool, size_t bytes,
                               const DFN2PrepostConfig *cfg);

/* As _init, plus refusing a pool whose recorded requirements differ from this
 * build's -- the stale-pool gate. `expected` NULL behaves as _init. */
DFN2Prepost *dfn2_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN2PrepostConfig *cfg,
                                  const DFN2PrepostMemReq *expected);

/* get_mem_size + aligned allocation + init. Host tools and tests; the board
 * path is _init on a caller pool. */
DFN2Prepost *dfn2_prepost_create(const DFN2PrepostConfig *cfg);

/* Frees only what _create allocated. NULL-safe. For an _init instance it is
 * a genuine no-op that may be called repeatedly -- the caller's pool stays
 * the caller's; a _create instance is gone after the first call, so calling
 * it twice on one is a use-after-free like any other double free. Never
 * touches the caller's fft, window or ERB matrices. */
void dfn2_prepost_destroy(DFN2Prepost *p);

/* Zero every recurrent/ring/hidden tensor, drop any open frame, and reset the
 * framing, normaliser and compose-clock states to their init values. Config,
 * borrowed pointers and pool are untouched. */
void dfn2_prepost_reset(DFN2Prepost *p);

/* ---- accessors ------------------------------------------------------- */

int dfn2_prepost_hop_size(const DFN2Prepost *p);   /* or -1 */
int dfn2_prepost_num_bins(const DFN2Prepost *p);   /* or -1 */
int dfn2_prepost_io_mode(const DFN2Prepost *p);    /* or -1 */
/* Frames of end-to-end lookahead the model costs on top of the framing:
 * DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD. -1 on NULL. */
int dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);
/* The graph contract this build binds. -1 on NULL. */
int dfn2_prepost_layout_version(const DFN2Prepost *p);

/* Point the instance at caller-loaded ERB matrices (erb_fwd.bin/erb_inv.bin
 * from export_erb_matrix.py --runtime-bins). Borrowed, never copied. BETWEEN
 * HOPS ONLY: with a frame open the call is refused (-1), so a swap can never
 * land inside one transaction -- between the features just taken through
 * erb_fwd and the mask expansion still pending through erb_inv. Be precise
 * about what that buys: it is atomicity per hop, not consistency per source
 * frame. The model's lookahead means the mask expanded in hop t belongs to
 * source frame t-1, and the graph's [t-1, t, t+1] feature window spans
 * three hops, so an accepted between-hop swap still straddles both for a
 * few hops. A swap that must be consistent per source frame is a stream
 * boundary: reset, then swap. Returns 0, or -1 on a NULL argument or with
 * a frame open. */
int dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd,
                                  const float *erb_inv);

/* Runtime attenuation limit in dB, applied by the compose stage against the
 * correctly delayed noisy target. 0 disables it. Between hops only, like the
 * ERB pair, and for the same reason -- an open transaction composes with the
 * limit it was opened under; the limit is applied at emission, which trails
 * a frame's features by the model lookahead, so this is atomicity per hop,
 * not a per-source-frame guarantee. Returns 0, or -1 on NULL, a non-finite
 * value, or with a frame open. */
int dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db);

/* ---- per-hop stages -------------------------------------------------- */

/* DFN2_IO_TIME. One hop of noisy input -- DFN2 is single-stream.
 *
 * Runs the analysis, computes the ERB and complex features, slides the
 * [t-1,t,t+1] graph window and advances the compose clock. Returns the number
 * of accelerator invocations this hop needs: 0 on the first hop, 1 thereafter.
 * -1 on NULL args, in DFN2_IO_FREQ, or with a frame still open. */
int dfn2_prepost_pre_process(DFN2Prepost *p, const float in_hop[DFN2_HOP_LEN]);

/* DFN2_IO_FREQ. One already-analysed frame, in the model's own framing:
 * the rfft of a sqrt-Hann-windowed DFN2_N_FFT block on the DFN2_HOP_LEN grid,
 * at normalized=True scale (⚠ see the scale note at the top of this file).
 * Does NO transform. Returns 0 on the first frame, 1 thereafter, or -1. */
int dfn2_prepost_pre_process_freq(DFN2Prepost *p,
                                  const float spec_re[DFN2_N_BINS],
                                  const float spec_im[DFN2_N_BINS]);

/* Publish the current frame's accelerator boundary. Every writable output is
 * NaN-prefilled, so a partial write is caught by frame_commit rather than
 * leaking the previous frame's values. Pointers are into this instance's pool
 * and stay valid until the next pre_process. Returns 0, or -1 if no frame is
 * open. */
int dfn2_prepost_frame_inputs(DFN2Prepost *p, DFN2PrepostInputs *inputs,
                              DFN2PrepostOutputs *outputs);

/* Transactional: validates that the accelerator wrote every head and every
 * next-state tensor with finite values, commits the recurrent state, then
 * runs the compose stage (ERB mask expansion, deep filter, alpha blend,
 * attenuation limit) and, in DFN2_IO_TIME, the synthesis.
 *
 * Requires a frame opened by pre_process AND published by frame_inputs():
 * a commit with no frame_inputs() behind it is refused, so an accelerator
 * that never ran cannot pass untouched buffers off as a result.
 *
 * On failure NOTHING moves -- persistent state is byte-identical, the frame
 * stays open, and -1 is returned. The caller then either calls
 * dfn2_prepost_frame_skip() to keep the framing schedule intact, or re-runs
 * the accelerator through a fresh frame_inputs(). */
int dfn2_prepost_frame_commit(DFN2Prepost *p);

/* Take the identity for the current frame: a unit ERB mask with alpha 0, so
 * the noisy spectrum passes through the cascade unchanged. This is the same
 * output the model itself produces when it asks for no suppression, and it is
 * exact rather than approximate: export_erb_matrix.py refuses to write an
 * erb_inv whose rows do not sum to 1 per bin, so a unit band mask expands to
 * unit bin gain. The recurrent state is NOT stepped, and the framing and
 * compose clocks still advance. This is what a failed accelerator run needs.
 * Returns 0, or -1 if no frame is open. */
int dfn2_prepost_frame_skip(DFN2Prepost *p);

/* DFN2_IO_TIME. Emit this hop's output. `out_hop` is always fully written:
 * DFN2_HOP_LEN samples, zero-filled during the two lookahead warm-up hops.
 * `*written` (may be NULL) reports how many are meaningful -- 0 or
 * DFN2_HOP_LEN. Returns 0, or -1. */
int dfn2_prepost_post_process(DFN2Prepost *p, float out_hop[DFN2_HOP_LEN],
                              int *written);

/* DFN2_IO_FREQ. The enhanced spectrum of the frame the compose stage just
 * emitted, at the same normalized=True scale as the input. Always fully
 * written: zeros during the two lookahead warm-up frames, where `*valid`
 * (may be NULL) reports 0. Returns 0, or -1. */
int dfn2_prepost_post_process_freq(DFN2Prepost *p, float re[DFN2_N_BINS],
                                   float im[DFN2_N_BINS], int *valid);

/* Which source frame the last emitted output belongs to, counting analysis
 * frames from 0. Returns 0 and writes *frame when an output has been emitted
 * since the last reset; -1 on NULL or before the first emission. Use it to
 * pair the output stream with anything else clocked on the same frames. */
int dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_PREPOST_H */
