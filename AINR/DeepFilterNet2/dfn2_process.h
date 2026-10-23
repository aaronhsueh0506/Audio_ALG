/* DeepFilterNet2 signal path outside the network (C twin of model.py
 * heads()/compose(); Python reference: inference.py).
 *
 *   C  : frame -> STFT -> ERB features + complex features
 *   NN : (feat_erb, feat_spec) -> (erb_mask, coefs, alpha)   <- the learned part
 *   C  : erb_mask/coefs/alpha -> full-band mask -> low-band deep filter ->
 *        alpha blend -> atten_lim -> ISTFT + OLA -> frame
 *
 * - STFT: sqrt-Hann, normalized=True (x N_FFT^-0.5); the feature branch also
 *   scales by DFN2_ANALYSIS_SCALE before the normalisers.
 * - ERB matrices are caller-loaded (erb_fwd.bin / erb_inv.bin): forward doubles
 *   the edge columns, inverse is a partition of unity.
 * - ERB feature: band energy in dB, causal EMA mean norm per band, divided by 40.
 * - Complex feature: real/imag of bins 0..df_bins with per-bin magnitude EMA
 *   unit norm.
 * - Cascade: ERB mask over the full band, then a DF_ORDER-tap complex FIR on the
 *   lowest DF_BINS bins; low band = alpha*DF(masked) + (1-alpha)*masked, high
 *   band keeps masked.
 * - ISTFT: normalized=True (x N_FFT^+0.5), sqrt-Hann, 50% OLA.
 *
 * Known differences from the training graph:
 * 1) Framing is streaming (center=False, zero-filled warm-up) instead of
 *    center=True reflect padding; only the first frame of a clip differs.
 * 2) DFN2_SPEC_NORM_EPS is kept inside the unit-norm sqrt: where denormals flush
 *    to zero the EMA can reach 0 and x/sqrt(0) would be NaN. */

#ifndef DFN2_PROCESS_H
#define DFN2_PROCESS_H

#include "audio_stft.h"

#include <stddef.h>
#include <stdint.h>

/* ---- Signal grid (= config.ini [signal]) ---- */
#define DFN2_SR             48000
#define DFN2_N_FFT          1024
#define DFN2_N_BINS         513   /* N_FFT/2 + 1 */
#define DFN2_WIN_LEN        1024  /* analysis window length (= N_FFT) */
#define DFN2_HOP_LEN        512   /* hop (= WIN_LEN/2, COLA) */
#define DFN2_OVL_LEN        (DFN2_WIN_LEN - DFN2_HOP_LEN)

/* The grid is 1024/512, not 960/480: 960 is not a power of two. The DF cutoff is
 * therefore 4500 Hz instead of 4800 Hz (config.ini [model] df_bins). */

/* ---- Model geometry (= config.ini [model]) ---- */
#define DFN2_N_ERB          32
#define DFN2_DF_BINS        96    /* bins owned by the deep filter (0..95) */
#define DFN2_DF_ORDER       5     /* FIR taps */
#define DFN2_DF_LOOKAHEAD   1     /* future taps */
#define DFN2_MASK_LOOKAHEAD 1

/* The deep-filter window is [t-(ORDER-LOOKAHEAD-1) .. t+LOOKAHEAD], i.e.
 * [t-3 .. t+1] for the shipped 5/1: ring depth 5, one frame of output delay.
 * DFN2 is a cascade (masked(t+1) waits for mask head(t+1), whose conv lookahead
 * is 1), so the streaming head-to-audio latency is MASK_LOOKAHEAD + DF_LOOKAHEAD
 * = 2 frames. See dfn2_compose_stream(). */
#define DFN2_DF_HISTORY     (DFN2_DF_ORDER - DFN2_DF_LOOKAHEAD - 1)  /* 3 */
#define DFN2_DF_RING        DFN2_DF_ORDER                            /* 5 */

/* ---- Feature normalisation (= config.ini [feature]) ----
 * FEATURE_VERSION must equal train.py's, otherwise the checkpoint and this front
 * end are different contracts. */
#define DFN2_FEATURE_VERSION  "dfn2_libdf_wnorm_upstream_init_no_eps_v5"

/* alpha is derived, not configured: exp(-(hop/sr)/tau) rounded to 3 decimals
 * (train.py make_norm_alpha), tau = 1.0 s, hop 512 @ 48 kHz -> 0.989 (effective
 * tau 0.9644 s). Re-derive it if sr or hop change. */
#define DFN2_ERB_NORM_ALPHA         0.989f
#define DFN2_ERB_NORM_INIT_LO_DB  (-60.0f)   /* = libDF MEAN_NORM_INIT */
#define DFN2_ERB_NORM_INIT_HI_DB  (-90.0f)
#define DFN2_ERB_NORM_SCALE_DB      40.0f
#define DFN2_ERB_LOG_FLOOR          1e-10f   /* = libDF lib.rs literal */

/* scale_db = |log floor - init_lo| (10*log10(1e-10) = -100 dB, |-100 - (-60)| =
 * 40), so numerically silent bands are bounded at exactly -1.00. The three
 * constants are mutually consistent; change them together. */

#define DFN2_SPEC_NORM_ALPHA        0.989f
#define DFN2_SPEC_NORM_INIT_LO      0.001f   /* = libDF UNIT_NORM_INIT */
#define DFN2_SPEC_NORM_INIT_HI      0.0001f
#define DFN2_SPEC_NORM_EPS          1e-12f   /* guards x/sqrt(0) where denormals flush to zero */

/* libDF analysis scale: the reference multiplies the spectrum by
 * wnorm = 2*hop/win^2 (1/1024) while torch.stft(normalized=True) applies
 * 1/sqrt(n_fft) (1/32), leaving a residual of 1/32. It matters on the DF path
 * (band_unit_norm is x/sqrt(EMA|x|), so an input scale persists as sqrt(c)).
 * Applied only to the copy feeding the normalisers; the spectrum used for
 * masking and ISTFT keeps its scale. */
#define DFN2_ANALYSIS_SCALE   0.03125f       /* = 2*512/1024^2 * sqrt(1024) */

/* ---- State (caller-allocated, persists across frames) ---- */
typedef struct {
    /* Framing objects, caller-owned and borrowed (see dfn2_framing_config); NULL
     * where the caller works on spectra only. */
    AcStft  *stft;
    AcIstft *istft;
    /* Caller-loaded ERB matrices: erb_fwd bin-major [DFN2_N_BINS][DFN2_N_ERB],
     * erb_inv band-major [DFN2_N_ERB][DFN2_N_BINS]; raw float32. The loader owns
     * the memory and may swap it between hops. */
    const float *erb_fwd;
    const float *erb_inv;

    /* per-band causal log-ERB mean EMA */
    float erb_norm_state[DFN2_N_ERB];

    /* per-bin complex-magnitude EMA (DF bins only) */
    float spec_norm_state[DFN2_DF_BINS];

    /* Deep-filter rings hold the lowest DF_BINS bins after the ERB mask; the
     * unmasked spectrum must not be pushed here (cascade). */
    float df_ring_re[DFN2_DF_RING][DFN2_DF_BINS];
    float df_ring_im[DFN2_DF_RING][DFN2_DF_BINS];
    float coef_ring[DFN2_DF_RING][DFN2_DF_BINS][DFN2_DF_ORDER][2];
    float alpha_ring[DFN2_DF_RING];
    float noisy_ring_re[DFN2_DF_RING][DFN2_N_BINS];
    float noisy_ring_im[DFN2_DF_RING][DFN2_N_BINS];
    int   df_ring_idx;     /* next write position */
    int   df_ring_count;   /* frames accumulated; warm-up ends when full */

    /* Streaming accelerator handoff: dfn2_compose_stream() numbers each pushed
     * spectrum and the returned heads must describe current_frame-MASK_LOOKAHEAD.
     * Do not mix the aligned and streaming compose APIs without dfn2_state_init(). */
    long long stream_frame_index;

    /* The high band (bin >= DF_BINS) is delayed by the same number of frames so
     * both halves of the band split come from the same instant. */
    float hi_delay_re[DFN2_DF_RING][DFN2_N_BINS - DFN2_DF_BINS];
    float hi_delay_im[DFN2_DF_RING][DFN2_N_BINS - DFN2_DF_BINS];

    /* scratch: not state, fully overwritten before each use */
    float scratch_power[DFN2_N_BINS];
    float scratch_erb_db[DFN2_N_ERB];
    float scratch_bin_gain[DFN2_N_BINS];

    /* Nonzero spans of the ERB matrices, derived from their contents by
     * dfn2_set_erb_matrices(): bin k's forward row is nonzero only in bands
     * [erb_fwd_lo[k], erb_fwd_hi[k]), band b's inverse row only in bins
     * [erb_inv_lo[b], erb_inv_hi[b]), and band b is fed only by bins
     * [erb_fwd_k0[b], erb_fwd_k1[b]). The sums visit only these spans; skipped
     * terms are finite values times +-0, so the result is unchanged. A non-finite
     * power or band gain takes its whole row. A caller that rewrites a matrix in
     * place must re-publish it through dfn2_set_erb_matrices(). */
    uint16_t erb_fwd_lo[DFN2_N_BINS];
    uint16_t erb_fwd_hi[DFN2_N_BINS];
    uint16_t erb_fwd_k0[DFN2_N_ERB];
    uint16_t erb_fwd_k1[DFN2_N_ERB];
    uint16_t erb_inv_lo[DFN2_N_ERB];
    uint16_t erb_inv_hi[DFN2_N_ERB];
} DFN2State;

/* Config of the two framing objects, for ac_stft/ac_istft get_mem_size and init:
 * 1024-point sqrt-Hann, hop 512, no centering, torch normalized=True scaling
 * (analysis x N_FFT^-0.5, synthesis x N_FFT^+0.5). `fft` is a DFN2_N_FFT handle
 * shared by both objects; `window` is a shared DFN2_WIN_LEN table, or NULL for the
 * analysis object to carve one that the synthesis object borrows. Borrowed tables
 * stay alive and unchanged as long as the objects. */
void dfn2_framing_config(AcStftConfig *cfg, FftHandle *fft,
                         const float *window);

/* Zero the state and set the two normalisers' linspace initial values. stft/istft
 * are the objects built per dfn2_framing_config (may be NULL when only spectra
 * are used) and are reset too, so calling this on a state in use resets it. */
void dfn2_state_init(DFN2State *st, AcStft *stft, AcIstft *istft);

/* Point the state at the caller-loaded ERB matrices and record their nonzero
 * spans. Call before feature extraction or mask expansion, and again after any
 * change to a matrix's contents. The library never reads files. */
void dfn2_set_erb_matrices(DFN2State *st, const float *erb_fwd,
                             const float *erb_inv);

/* frame (HOP_LEN new samples; the analysis overlap is kept internally) ->
 * complex spectrum of N_BINS bins, at the original scale used for masking and
 * ISTFT. */
void dfn2_analysis(DFN2State *st, const float *frame,
                   float *out_re, float *out_im);

/* Spectrum -> the network's two inputs: feat_erb [N_ERB] and feat_spec
 * [2*DF_BINS] laid out as [re..., im...] (two segments). DFN2_ANALYSIS_SCALE is
 * applied internally; callers must not pre-scale. */
void dfn2_compute_features(DFN2State *st,
                           const float *spec_re, const float *spec_im,
                           float *feat_erb, float *feat_spec);

/* atten_lim mixes into the final complex spectrum (the DF output is not a real
 * mask times noisy, so it cannot be applied in the band-gain domain). */
void dfn2_apply_atten_lim(const float *noisy_re, const float *noisy_im,
                          float *enh_re, float *enh_im,
                          float atten_lim_db);

/* Network outputs -> enhanced spectrum (C twin of model.py compose()).
 *   erb_mask: [N_ERB], sigmoid output in [0,1]
 *   coefs   : [DF_BINS * DF_ORDER * 2], model.py's (df_bins, df_order, 2) layout
 *   alpha   : one sigmoid blend weight; 0 = masked residual, 1 = deep filter
 *   spec_re/spec_im: current frame, original scale; pushed into the ring
 *   out_re/out_im  : enhanced spectrum delayed by DF_LOOKAHEAD frames
 * Returns 1 when out is valid, 0 during warm-up (those frames must not go to the
 * ISTFT). */
int dfn2_compose(DFN2State *st,
                 const float *spec_re, const float *spec_im,
                 const float *erb_mask, const float *coefs, float alpha,
                 float *out_re, float *out_im);

/* Streaming/hardware handoff: each call pushes one CURRENT spectrum. With
 * heads_valid=1, erb_mask/coefs/alpha must be the heads the accelerator returned
 * for current_frame-MASK_LOOKAHEAD; heads_valid=0 is for the left warm-up only
 * and the three head pointers may be NULL.
 *
 * Because of the cascade, a valid output corresponds to
 * current_frame - MASK_LOOKAHEAD - DF_LOOKAHEAD (2 hops = 21.33 ms at 48 kHz for
 * the shipped 1/1, plus the STFT framing). output_frame_index may be NULL;
 * otherwise it reports the frame id of out.
 *
 * To flush on the right, the accelerator produces the remaining heads with the
 * training zero padding and the caller keeps calling with zero spectra; C does
 * not invent a flush. atten_lim_db=0 disables the limit; otherwise it mixes
 * against the correctly delayed noisy target internally.
 * Returns 1 = valid out, 0 = warm-up, -1 = pointer or timing contract error. */
int dfn2_compose_stream(DFN2State *st,
                        const float *current_spec_re,
                        const float *current_spec_im,
                        int heads_valid,
                        const float *erb_mask,
                        const float *coefs,
                        float alpha,
                        float atten_lim_db,
                        float *out_re,
                        float *out_im,
                        long long *output_frame_index);

/* Enhanced spectrum -> HOP_LEN output samples (sqrt-Hann synthesis + OLA). No
 * window-envelope division: sqrt-Hann analysis x sqrt-Hann synthesis = Hann,
 * which is COLA at 50% overlap. */
void dfn2_synthesis(DFN2State *st,
                    const float *spec_re, const float *spec_im,
                    float *out_frame);

const char *dfn2_simd_backend(void);

#endif /* DFN2_PROCESS_H */
