/* Align-ULCNet pre/post-processing class: one object owning everything between
 * the caller's audio and the accelerator's tensors.
 *
 *     ulcnet_prepost_pre_process(p, err_hop, far_hop);
 *     ulcnet_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *     my_npu_run(&in, &out);                       // NPU fills `out`
 *     ulcnet_prepost_frame_commit(p);              // or _frame_skip()
 *     ulcnet_prepost_post_process(p, out_hop, &written);
 *
 * One hop in, one inference, one hop out, from the first hop, in both I/O modes.
 * Analysis is a rolling window over the last ULCNET_N_FFT samples (AcStft with
 * center 0), one frame per hop, the same convention as the AEC spectra.
 *
 * Memory: init() on a caller pool (board) or create() (host). The FftHandle,
 * sqrt-Hann window and synthesis AcIstft are caller-owned, required in
 * ULCNET_IO_TIME only.
 *
 * Inverse FFT and overlap-add are audiolib's AcIstft, never this class's. Two ways
 * to chain it:
 *
 *     // ULCNET_IO_TIME: hand the class the caller's AcIstft; commit pushes into it.
 *     AcStftConfig sc;  ulcnet_framing_config(&sc, fft, window, 0);
 *     AcIstft *istft = ac_istft_init(mem, ac_istft_get_mem_size(&sc), &sc);
 *     cfg.istft = istft;                       // ... ulcnet_prepost_init(...)
 *     ulcnet_prepost_post_process(p, out_hop, &written);   // hop out
 *
 *     // ULCNET_IO_FREQ: spectrum out, the caller pushes its own AcIstft.
 *     ulcnet_prepost_post_process_freq(p, re, im);
 *     ac_istft_push_planar(istft, re, im, out_hop);
 *
 * The class resets a borrowed AcIstft together with its own state.
 *
 * Error-input domain: the error stream is the AEC analysis seam, the signal the model is trained on
 * (aiaec-linear-error-v4). ULCNET_IO_TIME: err_hop is AecResContext.formed_hop. ULCNET_IO_FREQ:
 * err_re/err_im are AecResContext.error_spec, which pairs with the formed hop on this center-0
 * sqrt-Hann grid. Never AecOutputBase.base_spec from aec_get_output_base(): on a guard hop it does
 * not pair with formed_hop and is not the training signal. The class has no rendering base and
 * applies no output guard (see frame_skip). */

#ifndef ULCNET_PREPOST_H
#define ULCNET_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_process.h"   /* grid, AcStft/AcIstft, FftHandle, ulcnet_framing_config, ulcnet_model_io.h */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_PREPOST_DESCRIPTOR_VERSION 1u
/* Folded into build_flags_hash. Bumped on any change of pool size, carve order
 * or descriptor content, so a pool carved differently is refused by _init_ex. */
#define ULCNET_PREPOST_CARVE_VERSION 7u

/* Which side of the transform the caller works on. Fixed at init because it
 * decides the pool size: ULCNET_IO_TIME also carves the analysis states and the
 * output hop staging. */
typedef enum UlcnetIoMode {
    /* Spectrum in, spectrum out; the caller owns the transform. fft/window may be NULL. */
    ULCNET_IO_FREQ = 0,
    /* Hop in, hop out; the class owns the analysis framing and calls the
     * caller's FftHandle, and synthesises through the caller's AcIstft. */
    ULCNET_IO_TIME = 1
} UlcnetIoMode;

typedef struct UlcnetPrepostConfig {
    int          io_mode;      /* a UlcnetIoMode                            */
    int          delay_depth;  /* D, as exported; ULCNET_MODEL_IO_MIN/MAX_D */
    FftHandle   *fft;          /* ULCNET_IO_TIME: required, borrowed */
    const float *window;       /* ULCNET_IO_TIME: required, borrowed; ULCNET_N_FFT entries from ac_window_build() */
    AcIstft     *istft;        /* ULCNET_IO_TIME: required, borrowed, built from ulcnet_framing_config() over the same window; ignored in ULCNET_IO_FREQ */
    uint32_t model_layout_version; /* 0 = ULCNET_MODEL_IO_LAYOUT_VERSION; otherwise must equal it and the exported graph metadata */
} UlcnetPrepostConfig;

/* Fixed 32-byte descriptor; a pool sized for another io_mode, delay depth or
 * build is refused by _init_ex. */
typedef struct UlcnetPrepostMemReq {
    uint32_t descriptor_version;  /* ULCNET_PREPOST_DESCRIPTOR_VERSION      */
    uint32_t layout_version;      /* ULCNET_MODEL_IO_LAYOUT_VERSION         */
    uint32_t io_mode;             /* the resolved UlcnetIoMode              */
    uint32_t build_flags_hash;    /* FNV-1a-32: grid + layout + carve + io_mode + D */
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
 * an io_mode or delay_depth outside its range. `fft`/`window`/`istft` are left
 * NULL for the caller to set in ULCNET_IO_TIME. */
int ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg,
                                   int io_mode, int delay_depth);

/* Exact pool size. Reject-first: -1 with *req untouched on a NULL argument,
 * an unknown io_mode, a D out of range, or ULCNET_IO_TIME without a usable
 * fft/window/istft (the AcIstft must match ulcnet_framing_config()). */
int ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg,
                                UlcnetPrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config, a misaligned or
 * undersized pool, or a wrong-size FftHandle. Starts reset. */
UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes,
                                   const UlcnetPrepostConfig *cfg);

/* As _init, plus the stale-pool gate: refuses a pool whose recorded
 * requirements differ from this build's. `expected` NULL behaves as _init. */
UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes,
                                      const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected);

/* get_mem_size + aligned allocation + init (host path). */
UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg);

/* Frees only what _create allocated; NULL-safe, a no-op on an _init instance.
 * Never frees the caller's fft, window or istft. */
void ulcnet_prepost_destroy(UlcnetPrepost *p);

/* Zero every recurrent window/hidden tensor, drop any open frame, and reset
 * the framing states (a borrowed AcIstft included). Config and pool are
 * untouched. */
void ulcnet_prepost_reset(UlcnetPrepost *p);

/* Counter of state restarts: advances on ulcnet_prepost_reset() and when
 * frame_commit() refused a frame; 0 for a fresh instance or NULL. A fixed-point
 * accelerator refills its state with quantized zeros when it changes. */
uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p);

int ulcnet_prepost_hop_size(const UlcnetPrepost *p);   /* or -1 */
int ulcnet_prepost_num_bins(const UlcnetPrepost *p);   /* or -1 */
int ulcnet_prepost_io_mode(const UlcnetPrepost *p);    /* or -1 */
const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p);

/* ---- per-hop stages -------------------------------------------------- */

/* ULCNET_IO_TIME. One hop of the linear-AEC error (the analysis seam, formed_hop) and one
 * of the aligned far, both for the same input hop (no internal skew). Returns 1 (one hop is one
 * accelerator invocation), or -1 on NULL args, in ULCNET_IO_FREQ, or while the
 * previous frame is still open (neither committed nor skipped). */
int ulcnet_prepost_pre_process(UlcnetPrepost *p,
                               const float err_hop[ULCNET_HOP],
                               const float far_hop[ULCNET_HOP]);

/* ULCNET_IO_FREQ. The error spectrum is AecResContext.error_spec (not base_spec). One
 * analysed frame per stream: unnormalised rfft of a
 * sqrt-Hann-windowed ULCNET_N_FFT block on the hop grid (center=False). No transform is done. Returns 1; -1 on NULL args,
 * in ULCNET_IO_TIME, or while the previous frame is open. */
int ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                    const float err_re[ULCNET_BINS],
                                    const float err_im[ULCNET_BINS],
                                    const float far_re[ULCNET_BINS],
                                    const float far_im[ULCNET_BINS]);

/* Publish the current frame's accelerator boundary. `output` is NaN-prefilled so
 * an unwritten frame is caught by frame_commit. Each state output has the same
 * address as its state input: bind the returned pointers and read every state
 * input before writing any state output. Pointers are into the pool and stable
 * for its life. Returns 0, or -1 if no frame is open. */
int ulcnet_prepost_frame_inputs(UlcnetPrepost *p,
                                UlcnetModelIoInputs *inputs,
                                UlcnetModelIoOutputs *outputs);

/* Validate that the accelerator wrote a finite mask and state, apply the mask
 * and inverse compression, and feed the enhanced spectrum to the synthesis
 * (ULCNET_IO_TIME) or stage it for post_process_freq. Requires a frame opened by
 * pre_process and published by frame_inputs(); otherwise it is refused.
 *
 * On failure the K/V/logit windows and hidden restart from zero,
 * ulcnet_prepost_state_generation() advances, the frame stays open and -1 is
 * returned; the caller then calls ulcnet_prepost_frame_skip() or re-runs via a
 * fresh frame_inputs(). */
int ulcnet_prepost_frame_commit(UlcnetPrepost *p);

/* Take the identity for the current frame: the error spectrum passes through
 * unenhanced and the framing schedule still advances. Used after a failed
 * accelerator run or an alignment-boundary reprime. The output is the analysis
 * residual without the AEC output guard: on a guard hop (unusable estimate, residual
 * louder than the capture) it can exceed the capture. Returns 0, or -1. */
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
