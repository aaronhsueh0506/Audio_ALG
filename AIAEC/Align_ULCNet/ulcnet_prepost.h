/* ============================================================
 * Align-ULCNet pre/post-processing class.
 *
 * ONE object owning everything between the caller's audio and the
 * accelerator's tensors, so an integrator never touches the framing
 * states, the feature front end or the recurrent rings directly:
 *
 *     ulcnet_prepost_pre_process(p, err_hop, far_hop);
 *     ulcnet_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *     my_npu_run(&in, &out);                       // NPU fills `out`
 *     ulcnet_prepost_frame_commit(p);              // or _frame_skip()
 *     ulcnet_prepost_post_process(p, out_hop, &written);
 *
 * ONE HOP IN, ONE INFERENCE, ONE HOP OUT -- always, from the very first hop,
 * in both I/O modes. There is no variable emission count to loop over.
 *
 * FRAMING: this class analyses at center=False -- a plain rolling window over
 * the last ULCNET_N_FFT samples, one frame per hop. It is NOT
 * ulcnet_analysis_push's centered (center=True) schedule, which emits 0
 * frames on the first hop and TWO on the second to carry a reflect prefix.
 *
 * That is a re-labelling, not a change of signal. At this geometry
 * (win == n_fft, hop == n_fft/2, one shared sqrt-Hann table) centered frame k
 * and non-centered frame k-1 cover exactly the same samples with exactly the
 * same window, so they are bit-identical -- measured, not assumed. The only
 * difference is the head: the centered schedule additionally emits one
 * reflect-prefix frame, which this class does not. That also makes the class
 * agree with the AEC/GSC seam, whose spectra are center=False and arrive one
 * per hop from hop 0.
 *
 * The one-hop algorithmic latency is unchanged: it comes from the synthesis
 * half-window trim, not from the analysis schedule.
 *
 * WHY A THIRD TRANSLATION UNIT, not an addition to one of the two it uses:
 * ulcnet_process.c and ulcnet_model_io.c are deliberately independently
 * linkable -- AIAEC/tests/test_ulcnet_process_c.py compiles the first
 * WITHOUT the second, and test_ulcnet_model_io_c.py the second without the
 * first. A class in either would put an undefined symbol in the other's
 * parity build. This file composes them and neither of them changes.
 *
 * Memory: both house lifecycles (audio_common/include/fft_wrapper.h,
 * "Static memory support").
 *   size_t/req  ulcnet_prepost_get_mem_size(&cfg, &req);
 *   UlcnetPrepost *p = ulcnet_prepost_init(pool, req.bytes, &cfg);   // board
 *   UlcnetPrepost *p = ulcnet_prepost_create(&cfg);                  // host
 *   ulcnet_prepost_destroy(p);   // frees only what create() allocated
 *
 * The FftHandle and the sqrt-Hann window table are CALLER-OWNED and may be
 * shared with anything else on the same grid -- this class never creates or
 * destroys either. They are required in ULCNET_IO_TIME and unused in
 * ULCNET_IO_FREQ.
 * ============================================================ */

#ifndef ULCNET_PREPOST_H
#define ULCNET_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_process.h"   /* grid, UlcnetAnalysis/Synthesis, FftHandle,
                               * and (transitively) ulcnet_model_io.h       */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_PREPOST_DESCRIPTOR_VERSION 1u

/* Which side of the transform the caller works on. FIXED AT INIT because it
 * decides the pool size: ULCNET_IO_TIME additionally carves two rolling
 * analysis histories, the shared transform scratch, one synthesis state and
 * the output hop staging. At D=4 that is larger than the recurrent state
 * itself, so a runtime switch would make every frequency-domain instance
 * pay for machinery it never runs. */
typedef enum UlcnetIoMode {
    /* Spectrum in, spectrum out. NO framing, windowing, FFT, iFFT or
     * overlap-add -- the caller owns the transform. This is the mode for
     * chaining, and the mode in which `fft`/`window` may be NULL. */
    ULCNET_IO_FREQ = 0,
    /* Hop in, hop out. The class owns framing, windowing, overlap-add and
     * calls the CALLER'S FftHandle for the transforms. */
    ULCNET_IO_TIME = 1
} UlcnetIoMode;

typedef struct UlcnetPrepostConfig {
    int          io_mode;      /* a UlcnetIoMode                            */
    int          delay_depth;  /* D, as exported; ULCNET_MODEL_IO_MIN/MAX_D */
    FftHandle   *fft;          /* ULCNET_IO_TIME: required, borrowed        */
    const float *window;       /* ULCNET_IO_TIME: required, borrowed,
                                * ULCNET_N_FFT entries from
                                * ulcnet_make_window()                      */
} UlcnetPrepostConfig;

/* Fixed 32-byte shape, same staleness-gate discipline as the pipelines'
 * memory descriptors: a pool sized for another io_mode, delay depth or build
 * is refused by _init_ex rather than reinterpreted. */
typedef struct UlcnetPrepostMemReq {
    uint32_t descriptor_version;  /* ULCNET_PREPOST_DESCRIPTOR_VERSION      */
    uint32_t layout_version;      /* ULCNET_MODEL_IO_LAYOUT_VERSION         */
    uint32_t io_mode;             /* the resolved UlcnetIoMode              */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid + io_mode + D      */
    uint32_t alignment;
    uint32_t reserved;            /* 0 */
    uint64_t bytes;
} UlcnetPrepostMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(UlcnetPrepostMemReq) == 32,
               "UlcnetPrepostMemReq must stay 32 bytes");
#endif
#endif

typedef struct UlcnetPrepost UlcnetPrepost;

/* Fill a config with this build's grid defaults. Returns 0, or -1 on NULL /
 * an io_mode or delay_depth outside its range. `fft`/`window` are left NULL
 * for the caller to set in ULCNET_IO_TIME. */
int ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg,
                                   int io_mode, int delay_depth);

/* Exact pool size. Reject-first: -1 with *req untouched on a NULL argument,
 * an unknown io_mode, a D out of range, or ULCNET_IO_TIME without a usable
 * fft/window. */
int ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg,
                                UlcnetPrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config, a misaligned or
 * undersized pool, or a wrong-size FftHandle. Starts reset. */
UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes,
                                   const UlcnetPrepostConfig *cfg);

/* As _init, plus refusing a pool whose recorded requirements differ from
 * this build's -- the stale-pool gate. `expected` NULL behaves as _init. */
UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes,
                                      const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected);

/* get_mem_size + aligned allocation + init. Host tools and tests; the board
 * path is _init on a caller pool. */
UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg);

/* Frees only what _create allocated. NULL-safe. For an _init instance it is
 * a genuine no-op that may be called repeatedly -- the caller's pool stays
 * the caller's; a _create instance is gone after the first call, so calling
 * it twice on one is a use-after-free like any other double free. Never
 * touches the caller's fft or window. */
void ulcnet_prepost_destroy(UlcnetPrepost *p);

/* Zero every recurrent/ring/hidden tensor, drop any open frame, and reset
 * the framing states. Config and pool are untouched. */
void ulcnet_prepost_reset(UlcnetPrepost *p);

int ulcnet_prepost_hop_size(const UlcnetPrepost *p);   /* or -1 */
int ulcnet_prepost_num_bins(const UlcnetPrepost *p);   /* or -1 */
int ulcnet_prepost_io_mode(const UlcnetPrepost *p);    /* or -1 */
const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p);

/* ---- per-hop stages -------------------------------------------------- */

/* ULCNET_IO_TIME. One hop of the linear-AEC error and one of the aligned
 * far, BOTH FOR THE SAME INPUT HOP -- this class applies no internal skew.
 * A caller whose error source lags its far source (an upstream WOLA, say)
 * aligns them before calling.
 *
 * Transforms both streams (center=False rolling window). ALWAYS returns 1 --
 * one hop is one accelerator invocation, from the first hop onward. Returns
 * -1 on NULL args, in ULCNET_IO_FREQ, or while the previous frame is still
 * open (neither committed nor skipped): a hop is refused rather than
 * silently stacked on top of an unfinished one. */
int ulcnet_prepost_pre_process(UlcnetPrepost *p,
                               const float err_hop[ULCNET_HOP],
                               const float far_hop[ULCNET_HOP]);

/* ULCNET_IO_FREQ. One already-analysed frame per stream, in the model's own
 * framing (unnormalised rfft of a sqrt-Hann-windowed ULCNET_N_FFT block on
 * the hop grid, center=False -- see FRAMING above; the AEC's
 * AecResContext.error_spec is already in exactly this convention). Does NO
 * transform. Always returns 1; -1 on NULL args, in ULCNET_IO_TIME, or while
 * the previous frame is still open. */
int ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                    const float err_re[ULCNET_BINS],
                                    const float err_im[ULCNET_BINS],
                                    const float far_re[ULCNET_BINS],
                                    const float far_im[ULCNET_BINS]);

/* Publish the current frame's accelerator boundary. Every writable output
 * is NaN-prefilled, so a partial write is caught by frame_commit rather
 * than leaking the previous frame's values. Pointers are into this
 * instance's pool and stay valid until the next pre_process. Returns 0, or
 * -1 if no frame is open. */
int ulcnet_prepost_frame_inputs(UlcnetPrepost *p,
                                UlcnetModelIoInputs *inputs,
                                UlcnetModelIoOutputs *outputs);

/* Transactional: validates that the accelerator wrote every output, applies
 * the inverse compression, advances the K/V/logit rings and swaps the GRU
 * hidden tensors, then feeds the enhanced spectrum to the synthesis
 * (ULCNET_IO_TIME) or stages it for post_process_freq.
 *
 * Requires a frame opened by pre_process AND published by frame_inputs():
 * a commit with no frame_inputs() behind it is refused, so an accelerator
 * that never ran cannot pass untouched buffers off as a result.
 *
 * On failure NOTHING moves -- persistent state is byte-identical, the frame
 * stays open, and -1 is returned. The caller then either calls
 * ulcnet_prepost_frame_skip() to keep the framing schedule intact, or
 * re-runs the accelerator through a fresh frame_inputs(). */
int ulcnet_prepost_frame_commit(UlcnetPrepost *p);

/* Take the identity for the current frame: the error spectrum passes
 * through unenhanced, the model's recurrent state is NOT stepped, and the
 * framing schedule still advances. This is what a failed accelerator run
 * and an alignment-boundary reprime both need. Returns 0, or -1. */
int ulcnet_prepost_frame_skip(UlcnetPrepost *p);

/* ULCNET_IO_TIME. Emit this hop's output. `out_hop` is always fully
 * written: ULCNET_HOP samples, zero-filled during warm-up. `*written` (may
 * be NULL) reports how many are meaningful. Returns 0, or -1. */
int ulcnet_prepost_post_process(UlcnetPrepost *p,
                                float out_hop[ULCNET_HOP], int *written);

/* ULCNET_IO_FREQ. The enhanced spectrum of the frame just committed or
 * skipped. Returns 0, or -1. */
int ulcnet_prepost_post_process_freq(UlcnetPrepost *p,
                                     float re[ULCNET_BINS],
                                     float im[ULCNET_BINS]);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_PREPOST_H */
