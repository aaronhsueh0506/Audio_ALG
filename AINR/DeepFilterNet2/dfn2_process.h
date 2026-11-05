/* DeepFilterNet2 signal path outside the network (C twin of model.py
 * heads()/compose(); Python reference: inference.py).
 *
 *   C  : frame -> STFT -> ERB features + complex features
 *   NN : (feat_erb, feat_spec) -> (erb_mask, coefs, alpha)   <- the learned part
 *   C  : erb_mask/coefs/alpha -> full-band mask -> low-band deep filter ->
 *        alpha blend -> atten_lim -> ISTFT + OLA -> frame
 *
 * - STFT: sqrt-Hann, normalized=True (x n_fft^-0.5); the feature branch also
 *   scales by the grid's analysis scale before the normalisers.
 * - ERB matrices are caller-loaded (erb_fwd.bin / erb_inv.bin): forward doubles
 *   the edge columns, inverse is a partition of unity.
 * - ERB feature: band energy in dB, causal EMA mean norm per band, divided by 40.
 * - Complex feature: real/imag of bins 0..df_bins with per-bin magnitude EMA
 *   unit norm.
 * - Cascade: ERB mask over the full band, then a DF_ORDER-tap complex FIR on the
 *   lowest df_bins bins; low band = alpha*DF(masked) + (1-alpha)*masked, high
 *   band keeps masked.
 * - ISTFT: normalized=True (x n_fft^+0.5), sqrt-Hann, 50% OLA.
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

/* ---- Grid ----
 * Nothing about the signal grid, the model geometry or the lookahead pair is a compile-time
 * constant of this library: a DFN2Grid (dfn2_grid.h) is built by the application with
 * dfn2_grid_make() and passed to dfn2_state_init() and to every other object. The grid
 * must equal the one the checkpoint was trained and exported for (config.ini [signal]
 * sr / n_fft, [model] n_erb / df_bins / mask_lookahead / df_lookahead): it is part of the
 * model-I/O descriptor and of the pre/post pool hash, so a host built for one grid refuses a
 * graph description or a pool of another.
 *
 * The shipped 48 kHz grid is 1024/512, not 960/480 (960 is not a power of two), so its DF
 * cutoff is 4500 Hz instead of 4800 Hz. */
#include "dfn2_grid.h"

#ifdef DFN2_MASK_ONLY
/* -DDFN2_MASK_ONLY builds stage 1 alone (ERB mask, no deep filter), the model
 * `export_onnx.py --mask-only` writes. Every translation unit of the build passes it. There
 * is no deep filter to look ahead for, so the grid's df_lookahead is 0; the deep-filter
 * rings, the coefficient/alpha heads and the pathway window do not exist, and
 * dfn2_compose*() take the masked noisy spectrum as the output. */
#endif

/* The deep-filter window is [t-(ORDER-LOOKAHEAD-1) .. t+LOOKAHEAD], i.e. [t-3 .. t+1] for the
 * 5/1 pair of the 48 kHz model.
 * DFN2 is a cascade (masked(t+DF) waits for mask head(t+DF), which needs feature frame
 * t+DF+MASK_LOOKAHEAD), so the streaming head-to-audio latency is MASK_LOOKAHEAD +
 * DF_LOOKAHEAD frames (2 for 1/1, up to 6). See dfn2_compose_stream().
 *
 * Ring depth (DFN2Grid.ring). At stream frame `current` the heads are those of
 * head = current - MASK_LOOKAHEAD and the output is target = head - DF_LOOKAHEAD:
 *   noisy spectra   written at current, read at head and target
 *                   -> MASK_LOOKAHEAD + DF_LOOKAHEAD + 1 frames live
 *   masked spectra  written at head, read at the ORDER taps target-HISTORY..head
 *                   -> ORDER frames live
 *   coefs/alpha     written at head, read at target -> DF_LOOKAHEAD + 1
 * so the ring is the larger of the first two: 5 up to MASK + DF = 4, 7 for 2/4. Mask-only
 * keeps only the noisy ring: MASK_LOOKAHEAD + 1. The state holds DFN2_MAX_DF_RING frames. */

/* Link-time variant check. dfn2_process.c defines a one-byte symbol named after what the
 * state and boundary structs are built on -- the deep-filter variant and the capacities --
 * and dfn2_model_io.c and dfn2_prepost.c execute DFN2_REQUIRE_VARIANT_LINK() and so
 * reference the symbol of their own build. Objects built with a different layout than
 * dfn2_process.c (-DDFN2_MASK_ONLY on some units only, other DFN2_MAX_* values) fail to link
 * with that symbol undefined, instead of running a state of one size against code of
 * another. The grid is not in it: one build serves every grid within its capacities. The
 * read is volatile so it survives optimisation and section garbage collection together
 * with the code that holds it; no other code, allocation or I/O. The capacities must be
 * plain decimal integers (they are pasted into the name). */
#ifdef __cplusplus
extern "C" {
#endif
#ifdef DFN2_MASK_ONLY
#define DFN2_VARIANT_NAME_ dfn2_variant_mask_only
#else
#define DFN2_VARIANT_NAME_ dfn2_variant_full
#endif
#define DFN2_VARIANT_SYMBOL_JOIN_(base, fft, erb, df, ring) \
    base##_n##fft##_e##erb##_d##df##_r##ring
#define DFN2_VARIANT_SYMBOL_JOIN(base, fft, erb, df, ring) \
    DFN2_VARIANT_SYMBOL_JOIN_(base, fft, erb, df, ring)
#define DFN2_VARIANT_SYMBOL DFN2_VARIANT_SYMBOL_JOIN(DFN2_VARIANT_NAME_, \
    DFN2_MAX_N_FFT, DFN2_MAX_N_ERB, DFN2_MAX_DF_BINS, DFN2_MAX_DF_RING)
extern const unsigned char DFN2_VARIANT_SYMBOL;
#ifdef __cplusplus
}
#endif
#define DFN2_REQUIRE_VARIANT_LINK() \
    ((void)*(const volatile unsigned char *)&DFN2_VARIANT_SYMBOL)

/* ---- Feature normalisation (= config.ini [feature]) ----
 * FEATURE_VERSION must equal train.py's, otherwise the checkpoint and this front
 * end are different contracts. */
#define DFN2_FEATURE_VERSION  "dfn2_libdf_wnorm_upstream_init_no_eps_v5"

/* The two EMA coefficients and the analysis scale belong to the grid (dfn2_grid.h). The
 * normaliser starting points below are libDF's own literals and are the same for every
 * grid; a model trained with others defines them on the build line (every translation
 * unit). They are folded into the pre/post pool hash. */
#ifndef DFN2_ERB_NORM_INIT_LO_DB
#define DFN2_ERB_NORM_INIT_LO_DB  (-60.0f)   /* = libDF MEAN_NORM_INIT */
#endif
#ifndef DFN2_ERB_NORM_INIT_HI_DB
#define DFN2_ERB_NORM_INIT_HI_DB  (-90.0f)
#endif
#ifndef DFN2_ERB_NORM_SCALE_DB
#define DFN2_ERB_NORM_SCALE_DB      40.0f
#endif
#define DFN2_ERB_LOG_FLOOR          1e-10f   /* = libDF lib.rs literal */

/* scale_db = |log floor - init_lo| (10*log10(1e-10) = -100 dB, |-100 - (-60)| =
 * 40), so numerically silent bands are bounded at exactly -1.00. The three
 * constants are mutually consistent; change them together. */

#ifndef DFN2_SPEC_NORM_INIT_LO
#define DFN2_SPEC_NORM_INIT_LO      0.001f   /* = libDF UNIT_NORM_INIT */
#endif
#ifndef DFN2_SPEC_NORM_INIT_HI
#define DFN2_SPEC_NORM_INIT_HI      0.0001f
#endif
#define DFN2_SPEC_NORM_EPS          1e-12f   /* guards x/sqrt(0) where denormals flush to zero */

/* The five starting points above as compiled into dfn2_process.c: ERB lo, ERB hi, ERB scale,
 * spec lo, spec hi. The pre/post pool hash folds these, so a unit built with other values
 * than dfn2_process.c does not describe the state that is actually initialised. */
#define DFN2_NORM_STARTS 5
#ifdef __cplusplus
extern "C" {
#endif
extern const float dfn2_norm_starts[DFN2_NORM_STARTS];
#ifdef __cplusplus
}
#endif

/* ---- State (caller-allocated, persists across frames) ----
 * Sized for the capacities of dfn2_grid.h; a grid uses the first n_erb / df_bins / n_bins
 * entries of each array (a ring slot is one capacity-sized row, its first n values the
 * frame). dfn2_state_init() records the grid. */
typedef struct {
    DFN2Grid grid;
    /* Framing objects, caller-owned and borrowed (see dfn2_framing_config); NULL
     * where the caller works on spectra only. */
    AcStft  *stft;
    AcIstft *istft;
    /* Caller-loaded ERB matrices: erb_fwd bin-major [n_bins][n_erb],
     * erb_inv band-major [n_erb][n_bins]; raw float32. The loader owns
     * the memory and may swap it between hops. */
    const float *erb_fwd;
    const float *erb_inv;

    /* per-band causal log-ERB mean EMA */
    float erb_norm_state[DFN2_MAX_N_ERB];

    /* per-bin complex-magnitude EMA (DF bins only) */
    float spec_norm_state[DFN2_MAX_DF_BINS];

#ifndef DFN2_MASK_ONLY
    /* Masked spectra: the full band after the ERB mask. The deep filter reads the lowest
     * df_bins of ORDER consecutive rows and keeps the rest of the head row as the high
     * band, so both halves of the band split come from the same instant. The unmasked
     * spectrum must not be pushed here (cascade). */
    float masked_re[DFN2_MAX_DF_RING][DFN2_MAX_N_BINS];
    float masked_im[DFN2_MAX_DF_RING][DFN2_MAX_N_BINS];
    float coef_ring[DFN2_MAX_DF_RING][DFN2_MAX_DF_BINS][DFN2_DF_ORDER][2];
    float alpha_ring[DFN2_MAX_DF_RING];
#endif
    float noisy_ring_re[DFN2_MAX_DF_RING][DFN2_MAX_N_BINS];
    float noisy_ring_im[DFN2_MAX_DF_RING][DFN2_MAX_N_BINS];
#ifndef DFN2_MASK_ONLY
    int   df_ring_idx;     /* next write position */
    int   df_ring_count;   /* frames accumulated; warm-up ends when full */
#endif

    /* Streaming accelerator handoff: dfn2_compose_stream() numbers each pushed
     * spectrum and the returned heads must describe current_frame-MASK_LOOKAHEAD.
     * Do not mix the aligned and streaming compose APIs without dfn2_state_init(). */
    long long stream_frame_index;

    /* scratch: not state, fully overwritten before each use */
    float scratch_power[DFN2_MAX_N_BINS];
    float scratch_erb_db[DFN2_MAX_N_ERB];
    float scratch_bin_gain[DFN2_MAX_N_BINS];

    /* Nonzero spans of the ERB matrices, derived from their contents by
     * dfn2_set_erb_matrices(): bin k's forward row is nonzero only in bands
     * [erb_fwd_lo[k], erb_fwd_hi[k]), band b's inverse row only in bins
     * [erb_inv_lo[b], erb_inv_hi[b]), and band b is fed only by bins
     * [erb_fwd_k0[b], erb_fwd_k1[b]). The sums visit only these spans; skipped
     * terms are finite values times +-0, so the result is unchanged. A non-finite
     * power or band gain takes its whole row. A caller that rewrites a matrix in
     * place must re-publish it through dfn2_set_erb_matrices(). */
    uint16_t erb_fwd_lo[DFN2_MAX_N_BINS];
    uint16_t erb_fwd_hi[DFN2_MAX_N_BINS];
    uint16_t erb_fwd_k0[DFN2_MAX_N_ERB];
    uint16_t erb_fwd_k1[DFN2_MAX_N_ERB];
    uint16_t erb_inv_lo[DFN2_MAX_N_ERB];
    uint16_t erb_inv_hi[DFN2_MAX_N_ERB];
} DFN2State;

/* Config of the two framing objects, for ac_stft/ac_istft get_mem_size and init:
 * grid->n_fft-point sqrt-Hann, hop grid->hop, no centering, torch normalized=True scaling
 * (analysis x n_fft^-0.5, synthesis x n_fft^+0.5). `fft` is an n_fft handle shared by both
 * objects; `window` is a shared n_fft-entry table, or NULL for the analysis object to carve
 * one that the synthesis object borrows. Borrowed tables stay alive and unchanged as long
 * as the objects. A NULL grid or an invalid one (dfn2_grid_validate) zeroes cfg, so the two
 * get_mem_size calls report 0 bytes. */
void dfn2_framing_config(AcStftConfig *cfg, const DFN2Grid *grid, FftHandle *fft,
                         const float *window);

/* Zero the state, record the grid and set the two normalisers' linspace initial values.
 * stft/istft are the objects built per dfn2_framing_config (may be NULL when only spectra
 * are used) and are reset too, so calling this on a state in use resets it. Returns 0, or
 * -1 on NULL or an invalid grid (dfn2_grid_validate; the state is then zeroed with no grid
 * and every other call on it is a no-op or an error). */
int dfn2_state_init(DFN2State *st, const DFN2Grid *grid, AcStft *stft, AcIstft *istft);

/* Point the state at the caller-loaded ERB matrices and record their nonzero
 * spans. Call before feature extraction or mask expansion, and again after any
 * change to a matrix's contents. The library never reads files. */
void dfn2_set_erb_matrices(DFN2State *st, const float *erb_fwd,
                             const float *erb_inv);

/* frame (hop new samples; the analysis overlap is kept internally) -> complex spectrum of
 * n_bins bins, at the original scale used for masking and ISTFT. */
void dfn2_analysis(DFN2State *st, const float *frame,
                   float *out_re, float *out_im);

/* Spectrum -> the network's two inputs: feat_erb [n_erb] and feat_spec [2*df_bins] laid
 * out as [re..., im...] (two segments). The grid's stft_scale (= analysis scale) is applied
 * internally; callers must not pre-scale. */
void dfn2_compute_features(DFN2State *st,
                           const float *spec_re, const float *spec_im,
                           float *feat_erb, float *feat_spec);

/* atten_lim mixes into the final complex spectrum (the DF output is not a real
 * mask times noisy, so it cannot be applied in the band-gain domain). n_bins is the
 * grid's spectrum size. */
void dfn2_apply_atten_lim(int n_bins,
                          const float *noisy_re, const float *noisy_im,
                          float *enh_re, float *enh_im,
                          float atten_lim_db);

/* Network outputs -> enhanced spectrum (C twin of model.py compose()). With
 * DFN2_MASK_ONLY it is compose_mask_only(): out = noisy * full-band mask, with no
 * delay and no warm-up; `coefs` and `alpha` are ignored (pass NULL and 0).
 *   erb_mask: [n_erb], sigmoid output in [0,1]
 *   coefs   : [df_bins * DF_ORDER * 2], model.py's (df_bins, df_order, 2) layout
 *   alpha   : one sigmoid blend weight; 0 = masked residual, 1 = deep filter
 *   spec_re/spec_im: current frame, original scale; pushed into the ring
 *   out_re/out_im  : enhanced spectrum delayed by df_lookahead frames
 * Returns 1 when out is valid, 0 during warm-up (those frames must not go to the
 * ISTFT). */
int dfn2_compose(DFN2State *st,
                 const float *spec_re, const float *spec_im,
                 const float *erb_mask, const float *coefs, float alpha,
                 float *out_re, float *out_im);

/* Streaming/hardware handoff: each call pushes one CURRENT spectrum. With
 * DFN2_MASK_ONLY the output is the masked noisy spectrum of head_frame
 * (current - mask_lookahead), `coefs` and `alpha` are ignored, and the latency
 * is mask_lookahead frames. With
 * heads_valid=1, erb_mask/coefs/alpha must be the heads the accelerator returned
 * for current_frame-mask_lookahead; heads_valid=0 is for the left warm-up only
 * (the first mask_lookahead frames, none when it is 0) and the three head
 * pointers may be NULL.
 *
 * Because of the cascade, a valid output corresponds to
 * current_frame - mask_lookahead - df_lookahead (2 hops = 21.33 ms at the 48 kHz grid
 * for 1/1, plus the STFT framing). output_frame_index may be NULL;
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

/* Enhanced spectrum -> hop output samples (sqrt-Hann synthesis + OLA). No
 * window-envelope division: sqrt-Hann analysis x sqrt-Hann synthesis = Hann,
 * which is COLA at 50% overlap. */
void dfn2_synthesis(DFN2State *st,
                    const float *spec_re, const float *spec_im,
                    float *out_frame);

const char *dfn2_simd_backend(void);

#endif /* DFN2_PROCESS_H */
