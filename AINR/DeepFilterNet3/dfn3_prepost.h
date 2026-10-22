/* ============================================================
 * DeepFilterNet3 pre/post-processing class.
 *
 * ONE object owning everything between the caller's audio and the
 * accelerator's tensors, so an integrator never touches the framing state,
 * the ERB/complex feature front end, the deep-filter rings or the recurrent
 * caches directly:
 *
 *     if (dfn3_prepost_pre_process(p, in_hop) == 1) {
 *         dfn3_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *         my_npu_run(&in, &out);                     // NPU fills `out`
 *         dfn3_prepost_frame_commit(p);              // or _frame_skip()
 *     }
 *     dfn3_prepost_post_process(p, out_hop, &written);
 *
 * pre_process returns 1 on every hop but the very first, where it returns 0:
 * DFN3's analysis is center=False streaming, so one hop in is exactly one
 * frame, but the graph's [t-1, t, t+1] window has no right-hand neighbour
 * on hop 0 and MUST NOT be invoked there. Every later hop is one inference.
 *
 * WHY A THIRD TRANSLATION UNIT, not an addition to one of the two it uses:
 * dfn3_process.c and dfn3_model_io.c are deliberately independently linkable
 * -- AINR/tests/test_dfn3_stream_c.py compiles dfn3_process.c
 * WITHOUT dfn3_model_io.c. A class placed in either would put an undefined
 * symbol in the other's parity build. This file composes them and neither of
 * them changes.
 *
 * Memory: both house lifecycles (audiolib/common/include/fft_wrapper.h,
 * "Static memory support").
 *   dfn3_prepost_get_mem_size(&cfg, &req);
 *   DFN3Prepost *p = dfn3_prepost_init(pool, req.bytes, &cfg);   // board
 *   DFN3Prepost *p = dfn3_prepost_create(&cfg);                  // host
 *   dfn3_prepost_destroy(p);   // frees only what create() allocated
 *
 * ============================================================
 * ⚠ SPECTRUM SCALE AT THE DFN3_IO_FREQ BOUNDARY -- READ BEFORE CHAINING
 * ============================================================
 * The spectrum this class consumes and produces in DFN3_IO_FREQ is
 * torch.stft(normalized=True), i.e. the rfft of the sqrt-Hann-windowed
 * DFN3_N_FFT block multiplied by 1/sqrt(DFN3_N_FFT) = 1/32. dfn3_analysis()
 * applies that factor and dfn3_synthesis() removes it with sqrt(DFN3_N_FFT).
 *
 * The AIAEC models (Align-ULCNet and friends) hand over the UNNORMALISED
 * rfft. Chaining an AIAEC spectrum straight into dfn3_prepost_pre_process_freq
 * therefore over-drives DFN3 by exactly sqrt(1024) = 32x, and the reverse
 * under-drives it by 32x. Nothing in either class can detect this: both sides
 * are dimensionally valid float spectra, the ERB features are EMA-normalised
 * so the mask still looks plausible, and only the output level is wrong.
 * A caller crossing that boundary must scale explicitly:
 *     aiaec_spectrum -> dfn3 :  multiply by 1.0f / sqrtf(DFN3_N_FFT)
 *     dfn3 -> aiaec_spectrum :  multiply by sqrtf(DFN3_N_FFT)
 * They are also on DIFFERENT GRIDS (48 kHz / 1024 / 512 here versus
 * 16 kHz / 512), so a resampler sits between them anyway -- do the scaling
 * where that resampler lives.
 *
 * ⚠ DFN3_ANALYSIS_SCALE is NOT this factor. That constant is internal to the
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
 *   2. Model lookahead. Low-bin DF uses RAW future spectra; high-bin ERB
 *      masking uses the target frame. They overlap, so model lookahead is
 *      max(MASK_LOOKAHEAD, DF_LOOKAHEAD) = 1 hop for this fixed graph.
 *
 * Identity-path measured delay is 2 hops = 1024 samples = 21.33 ms at
 * 48 kHz: 1 hop WOLA plus 1 hop model lookahead. post_process() reports
 * written=0 on the first hop only. A unit center DF tap and unit ERB gains
 * produce identity, without an alpha head.
 *
 * Tail flush: feed zero hops while running the accelerator; one model
 * lookahead hop plus the WOLA tail must be drained for time-domain output.
 * An exact finite-clip flush must zero-pad the MODEL FEATURES (as training
 * does), not recompute normalised features from a zero spectrum. The public
 * entry points continue a live stream; callers exporting finite clips must
 * supply the matching padded graph inputs.
 *
 * ============================================================
 * BORROWED TABLES
 * ============================================================
 * The FftHandle, the window table and the ERB matrices are CALLER-OWNED and
 * BORROWED: this class never creates or destroys them, a borrowed window must
 * stay unchanged and alive as long as the class, and the ERB pointers may be
 * swapped between hops with dfn3_prepost_set_erb_matrices(). Leave `window`
 * NULL to let the analysis object carve the sqrt-Hann table in the pool.
 * ============================================================ */

#ifndef DFN3_PREPOST_H
#define DFN3_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "dfn3_model_io.h"   /* model geometry, DFN3ModelIOState, and
                              * (transitively) dfn3_process.h: the signal
                              * grid, DFN3State and FftHandle              */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN3_PREPOST_DESCRIPTOR_VERSION 1u
/* Folded into build_flags_hash. Any change of pool size, carve order or
 * descriptor content bumps it, so a pool recorded by another carve is refused
 * on the hash, not only on `bytes`. The carve covers DFN3State (including the
 * ERB matrices' nonzero ranges and each band's nonzero bin range), the
 * recurrent state of DFN3ModelIOState (h_gru and the pathway window) and, in
 * DFN3_IO_TIME, the output hop staging and the STFT / ISTFT objects; in
 * DFN3_IO_FREQ the second staging pair (apply_re/apply_im) of the dual-input
 * entry point. */
#define DFN3_PREPOST_CARVE_VERSION      7u

/* One shared alignment for every module (audiolib mem_align.h). */
#define DFN3_PREPOST_ALIGNMENT 16u

/* Flat element counts at the accelerator boundary. Stated as macros because
 * a caller wiring a runtime binds tensors by element count, and a count
 * recomputed at the call site is a place for a shape to drift silently. */
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

/* Which side of the transform the caller works on. FIXED AT INIT because it
 * decides the pool size: DFN3_IO_TIME also carves the output hop staging and
 * the two framing objects (STFT, ISTFT), DFN3_IO_FREQ carves none of them. */
typedef enum DFN3IoMode {
    /* Spectrum in, spectrum out. NO framing, windowing, FFT, iFFT or
     * overlap-add -- the caller owns the transform. This is the mode for
     * chaining (mind the scale note above), and the mode in which `fft` may
     * be NULL. */
    DFN3_IO_FREQ = 0,
    /* Hop in, hop out. The class owns framing, windowing, overlap-add and
     * calls the CALLER'S FftHandle for the transforms. */
    DFN3_IO_TIME = 1
} DFN3IoMode;

typedef struct DFN3PrepostConfig {
    int          io_mode;       /* a DFN3IoMode                             */
    FftHandle   *fft;           /* DFN3_IO_TIME: required, borrowed, sized
                                 * for DFN3_N_FFT                           */
    const float *window;        /* optional, DFN3_WIN_LEN entries, BORROWED
                                 * (DFN3_IO_TIME); NULL = a sqrt-Hann table
                                 * carved in the pool                        */
    const float *erb_fwd;       /* required, borrowed, bin-major
                                 * [DFN3_N_BINS][DFN3_N_ERB]                */
    const float *erb_inv;       /* required, borrowed, band-major
                                 * [DFN3_N_ERB][DFN3_N_BINS]                */
    float        atten_lim_db;  /* 0 disables the attenuation limit         */
} DFN3PrepostConfig;

/* Fixed 32-byte shape, same staleness-gate discipline as the pipelines'
 * memory descriptors: a pool sized for another io_mode, grid or build is
 * refused by _init_ex rather than reinterpreted. */
typedef struct DFN3PrepostMemReq {
    uint32_t descriptor_version;  /* DFN3_PREPOST_DESCRIPTOR_VERSION        */
    uint32_t layout_version;      /* DFN3_MODEL_IO_LAYOUT_VERSION           */
    uint32_t io_mode;             /* the resolved DFN3IoMode                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid + geometry +
                                   * feature version + io_mode              */
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

/* Accelerator inputs. All pointers are into this instance's pool and stay
 * valid for the life of the instance; the two recurrent-state tensors never
 * change address. The outputs' `*_next` state tensors point at the same
 * memory (see DFN3PrepostOutputs): the runtime binds each state input and its
 * output to one address and reads every state input before it writes any
 * state output. The feature windows are read-only: the accelerator must not
 * write them. Graph input names of the current layout (export_onnx.py,
 * DFN3_MODEL_IO_LAYOUT_VERSION) are given so a runtime binds by name without
 * reading the exporter. */
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

/* Accelerator-writable outputs: the two heads and the two next-state
 * tensors. frame_inputs() fills every head element with NaN so frame_commit()
 * detects a partial write instead of committing the previous frame's values.
 * The runtime binds its outputs to the pointers in this struct. The
 * next-state tensors alias the inputs' state tensors -- the same two buffers
 * at the same addresses on every frame, nothing copied -- so the runtime must
 * read every state input before it writes any state output at that address.
 * The next-state tensors are not NaN-prefilled: a partial or mixed state
 * write is not detectable, only a non-finite value is.
 *
 * `coefs` is (DFN3_DF_BINS, DFN3_DF_ORDER, 2) with bin outermost, tap next
 * and re/im innermost -- exactly model.py's layout and what
 * dfn3_compose_stream() expects. There is no alpha head in DFN3. */
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
 * unknown io_mode. `fft`, `window`, `erb_fwd` and `erb_inv` are left NULL for
 * the caller to set; atten_lim_db is left at 0 (disabled). */
int dfn3_prepost_config_defaults(DFN3PrepostConfig *cfg, int io_mode);

/* Exact pool size. Reject-first: -1 with *req untouched on a NULL argument,
 * an unknown io_mode, missing ERB matrices, a non-finite atten_lim_db, or
 * DFN3_IO_TIME without an FftHandle sized for DFN3_N_FFT. */
int dfn3_prepost_get_mem_size(const DFN3PrepostConfig *cfg,
                              DFN3PrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config, or a misaligned or
 * undersized pool. Starts reset. */
DFN3Prepost *dfn3_prepost_init(void *pool, size_t bytes,
                               const DFN3PrepostConfig *cfg);

/* As _init, plus refusing a pool whose recorded requirements differ from this
 * build's -- the stale-pool gate. `expected` NULL behaves as _init. */
DFN3Prepost *dfn3_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN3PrepostConfig *cfg,
                                  const DFN3PrepostMemReq *expected);

/* get_mem_size + aligned allocation + init. Host tools and tests; the board
 * path is _init on a caller pool. */
DFN3Prepost *dfn3_prepost_create(const DFN3PrepostConfig *cfg);

/* Frees only what _create allocated. NULL-safe. For an _init instance it is
 * a genuine no-op that may be called repeatedly -- the caller's pool stays
 * the caller's; a _create instance is gone after the first call, so calling
 * it twice on one is a use-after-free like any other double free. Never
 * touches the caller's fft, window or ERB matrices. */
void dfn3_prepost_destroy(DFN3Prepost *p);

/* Zero every recurrent/ring/hidden tensor, drop any open frame, and reset the
 * framing, normaliser and compose-clock states to their init values. Config,
 * borrowed pointers and pool are untouched. Advances the state generation. */
void dfn3_prepost_reset(DFN3Prepost *p);

/* Generation of the recurrent state (h_gru and the pathway window): 1 after
 * init, plus one for each dfn3_prepost_reset() and each refused
 * dfn3_prepost_frame_commit(), modulo 2^32; 0 for NULL. The accelerator keeps
 * its own fixed-point copy of the state, which neither of those events
 * reaches: a runtime compares this value with the one it saw at the previous
 * frame and refills its state tensors with quantized zeros when it changed. */
unsigned int dfn3_prepost_state_generation(const DFN3Prepost *p);

/* ---- accessors ------------------------------------------------------- */

int dfn3_prepost_hop_size(const DFN3Prepost *p);   /* or -1 */
int dfn3_prepost_num_bins(const DFN3Prepost *p);   /* or -1 */
int dfn3_prepost_io_mode(const DFN3Prepost *p);    /* or -1 */
/* Frames of end-to-end lookahead the model costs on top of the framing:
 * max(DFN3_MASK_LOOKAHEAD, DFN3_DF_LOOKAHEAD). -1 on NULL. */
int dfn3_prepost_model_lookahead_frames(const DFN3Prepost *p);
/* The graph contract this build binds. -1 on NULL. */
int dfn3_prepost_layout_version(const DFN3Prepost *p);

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
int dfn3_prepost_set_erb_matrices(DFN3Prepost *p, const float *erb_fwd,
                                  const float *erb_inv);

/* Runtime attenuation limit in dB, applied by the compose stage against the
 * correctly delayed noisy target. 0 disables it. Between hops only, like the
 * ERB pair, and for the same reason -- an open transaction composes with the
 * limit it was opened under; the limit is applied at emission, which trails
 * a frame's features by the model lookahead, so this is atomicity per hop,
 * not a per-source-frame guarantee. Returns 0, or -1 on NULL, a non-finite
 * value, or with a frame open. */
int dfn3_prepost_set_atten_lim(DFN3Prepost *p, float atten_lim_db);

/* ---- per-hop stages -------------------------------------------------- */

/* DFN3_IO_TIME. One hop of noisy input -- DFN3 is single-stream.
 *
 * Runs the analysis, computes the ERB and complex features, slides the
 * [t-1,t,t+1] graph window and advances the compose clock. Returns the number
 * of accelerator invocations this hop needs: 0 on the first hop, 1 thereafter.
 * -1 on NULL args, in DFN3_IO_FREQ, or with a frame still open. */
int dfn3_prepost_pre_process(DFN3Prepost *p, const float in_hop[DFN3_HOP_LEN]);

/* DFN3_IO_FREQ. One already-analysed frame, in the model's own framing:
 * the rfft of a sqrt-Hann-windowed DFN3_N_FFT block on the DFN3_HOP_LEN grid,
 * at normalized=True scale (⚠ see the scale note at the top of this file).
 * Does NO transform. Returns 0 on the first frame, 1 thereafter, or -1. */
int dfn3_prepost_pre_process_freq(DFN3Prepost *p,
                                  const float spec_re[DFN3_N_BINS],
                                  const float spec_im[DFN3_N_BINS]);

/* DFN3_IO_FREQ, dual input. `est` feeds the ESTIMATION branch (the ERB and
 * complex features, hence the graph window and the heads); `app` feeds the
 * APPLICATION branch (the noisy ring the mask, deep filter and attenuation
 * limit are applied to). Both are one already-analysed frame in the model's
 * own framing and at normalized=True scale, on the same frame clock. The two
 * branches touch disjoint state inside dfn3_process.c, so with est == app
 * this call is exactly dfn3_prepost_pre_process_freq(). Named est/app rather
 * than a/b because two same-typed pairs is where an argument swap hides.
 * Returns 0 on the first frame, 1 thereafter, or -1 on NULL arguments, in
 * DFN3_IO_TIME, or with a frame still open. */
int dfn3_prepost_pre_process_freq_dual(DFN3Prepost *p,
                                       const float est_re[DFN3_N_BINS],
                                       const float est_im[DFN3_N_BINS],
                                       const float app_re[DFN3_N_BINS],
                                       const float app_im[DFN3_N_BINS]);

/* Publish the current frame's accelerator boundary. The two heads are
 * NaN-prefilled, so an unwritten head is caught by frame_commit rather than
 * leaking the previous frame's values; the recurrent state is not, because it
 * is the accelerator's input. Pointers are into this instance's pool and the
 * state pointers are the same on every call. Returns 0, or -1 if no frame is
 * open. */
int dfn3_prepost_frame_inputs(DFN3Prepost *p, DFN3PrepostInputs *inputs,
                              DFN3PrepostOutputs *outputs);

/* Validates the result -- the two heads and the two recurrent-state
 * tensors must be finite -- then runs the compose stage (ERB mask expansion,
 * deep filter, attenuation limit) and, in DFN3_IO_TIME, the
 * synthesis. Nothing is copied here: the state is the pool's own tensors, as
 * written in place.
 *
 * Requires a frame opened by pre_process AND published by frame_inputs():
 * a commit with no frame_inputs() behind it is refused, so an accelerator
 * that never ran cannot pass untouched buffers off as a result.
 *
 * A non-finite head or state element cannot be rolled back, because the
 * accelerator overwrote the previous state: the two recurrent tensors are
 * zeroed, the state generation advances, the transaction is disarmed without
 * composing, and -1 is returned. The frame stays open and the framing, compose state and feature windows are
 * untouched. The caller then either calls dfn3_prepost_frame_skip() to keep
 * the framing schedule intact, or re-runs the accelerator through a fresh
 * frame_inputs(). A finite but partial or mixed state write passes the check;
 * only non-finite values are detectable. */
int dfn3_prepost_frame_commit(DFN3Prepost *p);

/* Identity heads: unit ERB gains on highs, unit center DF tap on raw
 * lows. Keeps the model lookahead clock and recurrent state unchanged.
 * Returns 0, or -1 if no frame is open. */
int dfn3_prepost_frame_skip(DFN3Prepost *p);

/* DFN3_IO_TIME. Emit this hop's output. `out_hop` is always fully written:
 * DFN3_HOP_LEN samples, zero-filled during the one lookahead warm-up hops.
 * `*written` (may be NULL) reports how many are meaningful -- 0 or
 * DFN3_HOP_LEN. Returns 0, or -1. */
int dfn3_prepost_post_process(DFN3Prepost *p, float out_hop[DFN3_HOP_LEN],
                              int *written);

/* DFN3_IO_FREQ. The enhanced spectrum of the frame the compose stage just
 * emitted, at the same normalized=True scale as the input. Always fully
 * written: zeros during the one lookahead warm-up frames, where `*valid`
 * (may be NULL) reports 0. Returns 0, or -1. */
int dfn3_prepost_post_process_freq(DFN3Prepost *p, float re[DFN3_N_BINS],
                                   float im[DFN3_N_BINS], int *valid);

/* Which source frame the last emitted output belongs to, counting analysis
 * frames from 0. Returns 0 and writes *frame when an output has been emitted
 * since the last reset; -1 on NULL or before the first emission. Use it to
 * pair the output stream with anything else clocked on the same frames. */
int dfn3_prepost_output_frame_index(const DFN3Prepost *p, long long *frame);

/* ---- the accelerator callback boundary ------------------------------
 *
 * A pipeline that hosts this class never owns an accelerator runtime; it
 * owns a DFN3Model, four plain fields the board fills in. `infer` receives
 * the published tensor views of the open frame (the same
 * DFN3PrepostInputs/DFN3PrepostOutputs frame_inputs() hands out, so a
 * runtime binds by name and element count and no flat argument order can
 * drift) and returns 0 when it wrote every output. `reset` (optional) is
 * called when the hosting stage resets. `io_descriptor` is the graph
 * contract the runtime was built against; a model that infers must publish
 * one, and the host refuses a descriptor dfn3_model_io_descriptor_validate()
 * rejects, because the finite checks catch an unwritten or non-finite
 * output, never a wrong-shaped one.
 *
 * This boundary lives here, not in dfn3_process.h: it names the class's
 * I/O views, and dfn3_process.c must stay compilable on its own for the
 * parity builds that link nothing else. */
typedef struct DFN3Model {
    void *user;
    int (*infer)(void *user, const DFN3PrepostInputs *inputs,
                 DFN3PrepostOutputs *outputs);
    void (*reset)(void *user);
    const DFN3ModelIoDescriptor *io_descriptor;   /* NULL = not published */
} DFN3Model;

/* Drive one open frame through the model: frame_inputs() (heads NaN-
 * prefilled), infer(), frame_commit(); on a nonzero infer() result or a
 * refused commit (a non-finite or unwritten output) the frame is taken with
 * frame_skip() instead, so the framing and compose clocks always advance
 * exactly once. The callback binds `outputs` directly and writes in place. A
 * nonzero infer() result is trusted not to have written the state, which
 * stays as it was; a refused commit has already zeroed it.
 * The finiteness rule is not re-implemented here -- frame_commit() owns it.
 * Returns 1 when the heads were committed, 0 when the frame was skipped,
 * -1 on a contract error (NULL model or instance, no infer callback, or no
 * frame open). Unlike the AIAEC identity policy, a skipped frame still emits
 * the source frame one hop back: the identity is applied inside the
 * raw-low/high-mask composition, the timing never changes. */
int dfn3_model_run_frame(const DFN3Model *model, DFN3Prepost *p);

#ifdef __cplusplus
}
#endif

#endif /* DFN3_PREPOST_H */
