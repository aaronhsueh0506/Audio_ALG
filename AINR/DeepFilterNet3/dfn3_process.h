/* DeepFilterNet3 signal path outside the network (C twin of model.py
 * heads()/compose(); Python reference: inference.py).
 *
 *   C  : frame -> STFT -> ERB features + complex features
 *   NN : (feat_erb, feat_spec) -> (erb_mask, coefs)   <- the learned part
 *   C  : erb_mask/coefs -> mask expansion + deep filter + band split ->
 *        ISTFT + OLA -> frame
 *
 * - STFT: sqrt-Hann, normalized=True (x N_FFT^-0.5); the feature branch also
 *   scales by DFN3_ANALYSIS_SCALE before the normalisers.
 * - ERB matrices are caller-loaded (erb_fwd.bin / erb_inv.bin): forward doubles
 *   the edge columns, inverse is a partition of unity.
 * - ERB feature: band energy in dB, causal EMA mean norm per band, divided by 40.
 * - Complex feature: real/imag of bins 0..df_bins with per-bin magnitude EMA
 *   unit norm.
 * - Band gain -> bin gain: erb_mask @ ERB_inv^T.
 * - Deep filter: DF_ORDER-tap complex FIR on the lowest DF_BINS bins.
 * - Band split: the DF owns bins < DF_BINS and the mask owns bins >= DF_BINS
 *   (parallel, not cascade).
 * - ISTFT: normalized=True (x N_FFT^+0.5), sqrt-Hann, 50% OLA.
 *
 * Known differences from the training graph:
 * 1) Framing is streaming (center=False, zero-filled warm-up) instead of
 *    center=True reflect padding; only the first frame of a clip differs.
 * 2) DFN3_SPEC_NORM_EPS is kept inside the unit-norm sqrt: where denormals flush
 *    to zero the EMA can reach 0 and x/sqrt(0) would be NaN. */

#ifndef DFN3_PROCESS_H
#define DFN3_PROCESS_H

#include "audio_stft.h"
#include <stdint.h>

#include <stddef.h>

/* ---- Signal grid (= config.ini [signal]) ---- */
#define DFN3_SR             48000
#define DFN3_N_FFT          1024
#define DFN3_N_BINS         513   /* N_FFT/2 + 1 */
#define DFN3_WIN_LEN        1024  /* analysis window length (= N_FFT) */
#define DFN3_HOP_LEN        512   /* hop (= WIN_LEN/2, COLA) */
#define DFN3_OVL_LEN        (DFN3_WIN_LEN - DFN3_HOP_LEN)

/* The grid is 1024/512, not 960/480: 960 is not a power of two. The DF cutoff is
 * therefore 4500 Hz instead of 4800 Hz (config.ini [model] df_bins). */

/* ---- Model geometry (= config.ini [model]) ---- */
#define DFN3_N_ERB          32
#define DFN3_DF_BINS        96    /* bins owned by the deep filter (0..95) */
#define DFN3_DF_ORDER       5     /* FIR taps */

/* The lookahead pair is part of the trained model: a build must carry the pair
 * its checkpoint was trained with (config.ini [model] mask_lookahead /
 * df_lookahead, recorded in the export metadata). Override the shipped 1/1 with
 * -DDFN3_MASK_LOOKAHEAD=<0..2> -DDFN3_DF_LOOKAHEAD=<0..DF_ORDER-1>, plain
 * decimals, on every translation unit of the build; the two are independent.
 * dfn3_model_io_descriptor_validate() and the pre/post pool hash refuse a host
 * built for another pair, and objects built for different pairs do not link
 * (see DFN3_REQUIRE_LOOKAHEAD_PAIR_LINK below). */
#ifndef DFN3_DF_LOOKAHEAD
#define DFN3_DF_LOOKAHEAD   1     /* future taps */
#endif
#ifndef DFN3_MASK_LOOKAHEAD
#define DFN3_MASK_LOOKAHEAD 1     /* future feature frames behind each head */
#endif

#if DFN3_MASK_LOOKAHEAD < 0 || DFN3_MASK_LOOKAHEAD > 2
#error "DFN3_MASK_LOOKAHEAD must be in [0, 2]: the graph window holds three frames"
#endif
#if DFN3_DF_LOOKAHEAD < 0 || DFN3_DF_LOOKAHEAD >= DFN3_DF_ORDER
#error "DFN3_DF_LOOKAHEAD must be in [0, DFN3_DF_ORDER - 1]: the filter keeps at least its current tap"
#endif

/* The deep-filter window is [t-(ORDER-LOOKAHEAD-1) .. t+LOOKAHEAD], i.e.
 * [t-3 .. t+1] for the shipped 5/1. DFN3 is a parallel split: the taps read the
 * raw spectrum, so output frame t needs the heads of t (input up to
 * t+MASK_LOOKAHEAD) and the spectrum up to t+DF_LOOKAHEAD, and the output is
 * delayed by DFN3_OUTPUT_DELAY = max(MASK, DF) frames (10.67 ms each at
 * 48 kHz; 1 for the shipped pair where the reference 2/2 would cost about
 * 21.3 ms). When DF > MASK the heads arrive before their output frame and wait
 * in the ring (DFN3_HEAD_QUEUE).
 * The ring keeps every frame from the oldest tap of the delayed frame,
 * HISTORY frames before it, to the newest write, OUTPUT_DELAY frames after it:
 * HISTORY + OUTPUT_DELAY + 1 = ORDER + OUTPUT_DELAY - DF slots (5 for the
 * shipped pair). */
#define DFN3_OUTPUT_DELAY \
    (DFN3_MASK_LOOKAHEAD > DFN3_DF_LOOKAHEAD ? DFN3_MASK_LOOKAHEAD \
                                             : DFN3_DF_LOOKAHEAD)
#define DFN3_HEAD_QUEUE     (DFN3_DF_LOOKAHEAD > DFN3_MASK_LOOKAHEAD)
#define DFN3_DF_HISTORY     (DFN3_DF_ORDER - DFN3_DF_LOOKAHEAD - 1)
#define DFN3_DF_RING        (DFN3_DF_ORDER + DFN3_OUTPUT_DELAY - DFN3_DF_LOOKAHEAD)

/* Link-time pair check. dfn3_process.c defines a one-byte symbol named after
 * the pair it was built for (dfn3_lookahead_mask1_df1 for the shipped 1/1);
 * dfn3_model_io.c and dfn3_prepost.c execute DFN3_REQUIRE_LOOKAHEAD_PAIR_LINK()
 * and so reference the symbol of their own pair. An object built for another
 * pair than dfn3_process.c fails to link with that symbol undefined, instead of
 * running a window, ring or pool of one pair against a graph of another (the
 * state size depends on the ring). The read is volatile so it survives
 * optimisation and section garbage collection together with the code that holds
 * it; no other code, allocation or I/O. */
#define DFN3_LOOKAHEAD_PAIR_SYMBOL_JOIN_(mask, df) \
    dfn3_lookahead_mask##mask##_df##df
#define DFN3_LOOKAHEAD_PAIR_SYMBOL_JOIN(mask, df) \
    DFN3_LOOKAHEAD_PAIR_SYMBOL_JOIN_(mask, df)
#define DFN3_LOOKAHEAD_PAIR_SYMBOL \
    DFN3_LOOKAHEAD_PAIR_SYMBOL_JOIN(DFN3_MASK_LOOKAHEAD, DFN3_DF_LOOKAHEAD)
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned char DFN3_LOOKAHEAD_PAIR_SYMBOL;
#ifdef __cplusplus
}
#endif
#define DFN3_REQUIRE_LOOKAHEAD_PAIR_LINK() \
    ((void)*(const volatile unsigned char *)&DFN3_LOOKAHEAD_PAIR_SYMBOL)

/* ---- Feature normalisation (= config.ini [feature]) ----
 * FEATURE_VERSION must equal train.py's, otherwise the checkpoint and this front
 * end are different contracts. */
#define DFN3_FEATURE_VERSION  "dfn2_libdf_wnorm_upstream_init_no_eps_v5"

/* alpha is derived, not configured: exp(-(hop/sr)/tau) rounded to 3 decimals
 * (train.py make_norm_alpha), tau = 1.0 s, hop 512 @ 48 kHz -> 0.989 (effective
 * tau 0.9644 s). Re-derive it if sr or hop change. */
#define DFN3_ERB_NORM_ALPHA         0.989f
#define DFN3_ERB_NORM_INIT_LO_DB  (-60.0f)   /* = libDF MEAN_NORM_INIT */
#define DFN3_ERB_NORM_INIT_HI_DB  (-90.0f)
#define DFN3_ERB_NORM_SCALE_DB      40.0f
#define DFN3_ERB_LOG_FLOOR          1e-10f   /* = libDF lib.rs literal */

/* scale_db = |log floor - init_lo| (10*log10(1e-10) = -100 dB, |-100 - (-60)| =
 * 40), so numerically silent bands are bounded at exactly -1.00. The three
 * constants are mutually consistent; change them together. */

#define DFN3_SPEC_NORM_ALPHA        0.989f
#define DFN3_SPEC_NORM_INIT_LO      0.001f   /* = libDF UNIT_NORM_INIT */
#define DFN3_SPEC_NORM_INIT_HI      0.0001f
#define DFN3_SPEC_NORM_EPS          1e-12f   /* guards x/sqrt(0) where denormals flush to zero */

/* libDF analysis scale: the reference multiplies the spectrum by
 * wnorm = 2*hop/win^2 (1/1024) while torch.stft(normalized=True) applies
 * 1/sqrt(n_fft) (1/32), leaving a residual of 1/32. It matters on the DF path
 * (band_unit_norm is x/sqrt(EMA|x|), so an input scale persists as sqrt(c)).
 * Applied only to the copy feeding the normalisers; the spectrum used for
 * masking and ISTFT keeps its scale. */
#define DFN3_ANALYSIS_SCALE   0.03125f       /* = 2*512/1024^2 * sqrt(1024) */

/* ---- State (caller-allocated, persists across frames) ---- */
typedef struct {
    /* Framing objects, caller-owned and borrowed (see dfn3_framing_config); NULL
     * where the caller works on spectra only. */
    AcStft  *stft;
    AcIstft *istft;
    /* Caller-loaded ERB matrices: erb_fwd bin-major [DFN3_N_BINS][DFN3_N_ERB],
     * erb_inv band-major [DFN3_N_ERB][DFN3_N_BINS]; raw float32. The loader owns
     * the memory and may swap it between hops. */
    const float *erb_fwd;
    const float *erb_inv;

    /* per-band causal log-ERB mean EMA */
    float erb_norm_state[DFN3_N_ERB];

    /* per-bin complex-magnitude EMA (DF bins only) */
    float spec_norm_state[DFN3_DF_BINS];

    /* Low-bin ring holds the raw noisy spectrum: ERB gain applies only to the high
     * bins, so future low taps never wait for future masks. */
    float df_ring_re[DFN3_DF_RING][DFN3_DF_BINS];
    float df_ring_im[DFN3_DF_RING][DFN3_DF_BINS];
    float coef_ring[DFN3_DF_RING][DFN3_DF_BINS][DFN3_DF_ORDER][2];
    float noisy_ring_re[DFN3_DF_RING][DFN3_N_BINS];
    float noisy_ring_im[DFN3_DF_RING][DFN3_N_BINS];
    int   df_ring_idx;     /* next write position */
    int   df_ring_count;   /* frames accumulated; warm-up ends when full */

    /* Streaming accelerator handoff: dfn3_compose_stream() numbers each pushed
     * spectrum and the returned heads must describe current_frame-MASK_LOOKAHEAD.
     * Do not mix the aligned and streaming compose APIs without dfn3_state_init(). */
    long long stream_frame_index;

    /* The high band (bin >= DF_BINS) is delayed by the same number of frames so
     * both halves of the band split come from the same instant. */
    float hi_delay_re[DFN3_DF_RING][DFN3_N_BINS - DFN3_DF_BINS];
    float hi_delay_im[DFN3_DF_RING][DFN3_N_BINS - DFN3_DF_BINS];

    /* scratch: not state, fully overwritten before each use */
    float scratch_power[DFN3_N_BINS];
    float scratch_erb_db[DFN3_N_ERB];
    float scratch_bin_gain[DFN3_N_BINS];

    /* Nonzero spans of the ERB matrices, derived from their contents by
     * dfn3_set_erb_matrices(): bin k's forward row is nonzero only in bands
     * [erb_fwd_lo[k], erb_fwd_hi[k]), band b's inverse row only in bins
     * [erb_inv_lo[b], erb_inv_hi[b]), and band b is fed only by bins
     * [erb_fwd_k0[b], erb_fwd_k1[b]). The sums visit only these spans; skipped
     * terms are finite values times +-0, so the result is unchanged. A non-finite
     * power or band gain takes its whole row. A caller that rewrites a matrix in
     * place must re-publish it through dfn3_set_erb_matrices(). */
    uint16_t erb_fwd_lo[DFN3_N_BINS];
    uint16_t erb_fwd_hi[DFN3_N_BINS];
    uint16_t erb_fwd_k0[DFN3_N_ERB];
    uint16_t erb_fwd_k1[DFN3_N_ERB];
    uint16_t erb_inv_lo[DFN3_N_ERB];
    uint16_t erb_inv_hi[DFN3_N_ERB];
} DFN3State;

/* Config of the two framing objects, for ac_stft/ac_istft get_mem_size and init:
 * 1024-point sqrt-Hann, hop 512, no centering, torch normalized=True scaling
 * (analysis x N_FFT^-0.5, synthesis x N_FFT^+0.5). `fft` is a DFN3_N_FFT handle
 * shared by both objects; `window` is a shared DFN3_WIN_LEN table, or NULL for the
 * analysis object to carve one that the synthesis object borrows. Borrowed tables
 * stay alive and unchanged as long as the objects. */
void dfn3_framing_config(AcStftConfig *cfg, FftHandle *fft,
                         const float *window);

/* Zero the state and set the two normalisers' linspace initial values. stft/istft
 * are the objects built per dfn3_framing_config (may be NULL when only spectra
 * are used) and are reset too, so calling this on a state in use resets it. */
void dfn3_state_init(DFN3State *st, AcStft *stft, AcIstft *istft);

/* Point the state at the caller-loaded ERB matrices and record their nonzero
 * spans. Call before feature extraction or mask expansion, and again after any
 * change to a matrix's contents. The library never reads files. */
void dfn3_set_erb_matrices(DFN3State *st, const float *erb_fwd,
                             const float *erb_inv);

/* frame (HOP_LEN new samples; the analysis overlap is kept internally) ->
 * complex spectrum of N_BINS bins, at the original scale used for masking and
 * ISTFT. */
void dfn3_analysis(DFN3State *st, const float *frame,
                   float *out_re, float *out_im);

/* Spectrum -> the network's two inputs: feat_erb [N_ERB] and feat_spec
 * [2*DF_BINS] laid out as [re..., im...] (two segments). DFN3_ANALYSIS_SCALE is
 * applied internally; callers must not pre-scale. */
void dfn3_compute_features(DFN3State *st,
                           const float *spec_re, const float *spec_im,
                           float *feat_erb, float *feat_spec);

/* atten_lim mixes noisy and enhanced in the final complex spectrum. */
void dfn3_apply_atten_lim(const float *noisy_re, const float *noisy_im,
                          float *enh_re, float *enh_im,
                          float atten_lim_db);

/* Network outputs -> enhanced spectrum (C twin of model.py compose()).
 *   erb_mask: [N_ERB], sigmoid output in [0,1]
 *   coefs   : [DF_BINS * DF_ORDER * 2], model.py's (df_bins, df_order, 2) layout
 *   spec_re/spec_im: current frame, original scale; pushed into the ring
 *   out_re/out_im  : enhanced spectrum delayed by DF_LOOKAHEAD frames
 * Returns 1 when out is valid, 0 during warm-up (those frames must not go to the
 * ISTFT).
 *
 * Offline alignment: the heads are those of the spectrum passed in the same
 * call (heads[t] with spectrum t), and there is no MASK_LOOKAHEAD offset. A
 * live accelerator delivers heads[t] only after frame t+MASK_LOOKAHEAD has been
 * captured, so this entry fits a live stream only when MASK_LOOKAHEAD == 0, or
 * when the caller holds each spectrum back by MASK_LOOKAHEAD frames itself to
 * pair it with its heads (and then the output is DF_LOOKAHEAD + MASK_LOOKAHEAD
 * frames behind the newest capture). dfn3_compose_stream() does that pairing
 * from the ring and is the path for the accelerator handoff. */
int dfn3_compose(DFN3State *st,
                 const float *spec_re, const float *spec_im,
                 const float *erb_mask, const float *coefs,
                 float *out_re, float *out_im);

/* Streaming/hardware handoff: each call pushes one CURRENT spectrum. With
 * heads_valid=1, erb_mask/coefs must be the heads for
 * current_frame-MASK_LOOKAHEAD; the MASK_LOOKAHEAD warm-up frames pass
 * heads_valid=0 and may pass NULL heads.
 *
 * DFN3 is a parallel band split: the DF's future taps read the raw spectrum and
 * never wait for a future mask, so the latency is DFN3_OUTPUT_DELAY =
 * max(MASK_LOOKAHEAD, DF_LOOKAHEAD) = 1 hop = 10.67 ms at 48 kHz for the
 * shipped 1/1. The first MASK_LOOKAHEAD calls take no heads and return 0; with
 * DF_LOOKAHEAD > MASK_LOOKAHEAD the following calls take heads, keep them and
 * still return 0 until the first output frame is complete.
 * output_frame_index may be NULL; otherwise it reports the frame id of out.
 * atten_lim_db=0 disables the complex attenuation-limit mix.
 *
 * To flush on the right, the accelerator produces the remaining heads with the
 * training zero padding and the caller keeps calling with zero spectra. Every
 * spectrum and active head must be finite; invalid input is refused before any
 * ring, clock or output changes, so the same frame can be retried.
 * Returns 1 = valid out, 0 = warm-up, -1 = invalid input or clock contract. */
int dfn3_compose_stream(DFN3State *st,
                        const float *current_spec_re,
                        const float *current_spec_im,
                        int heads_valid,
                        const float *erb_mask,
                        const float *coefs,
                        float atten_lim_db,
                        float *out_re,
                        float *out_im,
                        long long *output_frame_index);

/* Enhanced spectrum -> HOP_LEN output samples (sqrt-Hann synthesis + OLA). No
 * window-envelope division: sqrt-Hann analysis x sqrt-Hann synthesis = Hann,
 * which is COLA at 50% overlap. */
void dfn3_synthesis(DFN3State *st,
                    const float *spec_re, const float *spec_im,
                    float *out_frame);

const char *dfn3_simd_backend(void);

#endif /* DFN3_PROCESS_H */
