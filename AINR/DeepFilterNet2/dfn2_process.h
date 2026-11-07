/* DeepFilterNet2 signal path outside the network, on spectra (C twin of model.py).
 * Framing contract: the library holds no window, FFT or STFT object; the application frames with audiolib
 * (AcStft / AcIstft) as trained: sqrt-Hann (audiolib's default window), center 0, hop = n_fft / 2, plain
 * overlap-add (AC_OLA_ACCUMULATE), spec_scale = grid.stft_scale (1/sqrt(n_fft)), ispec_scale =
 * grid.stft_unscale (sqrt(n_fft)). A wrong window or scale is not detectable here. Against the training
 * graph's center=True padding only the first frame of a clip differs. */

#ifndef DFN2_PROCESS_H
#define DFN2_PROCESS_H

#include <stddef.h>
#include <stdint.h>

/* A DFN2Grid (dfn2_grid_make()) goes to every object. */
#include "dfn2_grid.h"

#ifdef DFN2_MASK_ONLY
/* -DDFN2_MASK_ONLY (every translation unit): stage 1 alone (export_onnx.py --mask-only), no deep filter. */
#endif

/* The deep filter reads frames [t-(df_order-df_lookahead-1) .. t+df_lookahead]; DFN2Grid.ring is
 * the larger of the noisy history, mask + df_lookahead + 1 frames, and the df_order masked taps. */

/* Mixed -DDFN2_MASK_ONLY units fail to link: each object references its variant's symbol. */
#ifdef __cplusplus
extern "C" {
#endif
#ifdef DFN2_MASK_ONLY
#define DFN2_VARIANT_SYMBOL dfn2_variant_mask_only
#else
#define DFN2_VARIANT_SYMBOL dfn2_variant_full
#endif
extern const unsigned char DFN2_VARIANT_SYMBOL;
#ifdef __cplusplus
}
#endif
#define DFN2_REQUIRE_VARIANT_LINK() \
    ((void)*(const volatile unsigned char *)&DFN2_VARIANT_SYMBOL)

#define DFN2_FEATURE_VERSION  "dfn2_libdf_wnorm_upstream_init_no_eps_v5"   /* = train.py's */

#define DFN2_ERB_LOG_FLOOR          1e-10f   /* = libDF lib.rs literal */
#define DFN2_SPEC_NORM_EPS          1e-12f   /* guards x/sqrt(0) where denormals flush */

/* A header over arrays dfn2_state_init() carves out of a caller block, which must stay alive and unchanged
 * while used. Rings are flat: slot s at s * n_bins (coef_ring: s * df_bins * df_order * 2; alpha_ring: s). */
typedef struct DFN2State {
    DFN2Grid grid;
    const float *erb_fwd;     /* borrowed raw float32, bin-major [n_bins][n_erb] */
    const float *erb_inv;     /* borrowed raw float32, band-major [n_erb][n_bins] */

    float *erb_norm_state;    /* [n_erb] causal log-ERB mean EMA */
    float *spec_norm_state;   /* [df_bins] per-bin complex-magnitude EMA */

#ifndef DFN2_MASK_ONLY
    /* Full-band masked spectra [ring][n_bins]; never push the unmasked spectrum here (cascade). */
    float *masked_re;
    float *masked_im;
    float *coef_ring;      /* [ring][df_bins][df_order][2] */
    float *alpha_ring;     /* [ring] */
    float *df_zero_row;    /* [df_bins] zeros: the taps before the stream start read it */
    const float **src_re;  /* [df_order] source-row pointers of one filtered frame */
    const float **src_im;
#endif
    float *noisy_ring_re;  /* [ring][n_bins] */
    float *noisy_ring_im;

    float *scratch_power;     /* [n_bins] scratch, not state: overwritten before each use */
    float *scratch_erb_db;    /* [n_erb] scratch */
    float *scratch_bin_gain;  /* [n_bins] scratch */

    /* Half-open nonzero spans of the ERB matrices; the sums visit only them (skipped terms are
     * finite * +-0, so results are unchanged; a non-finite power or gain takes its whole row). */
    uint16_t *erb_fwd_lo;  /* [n_bins] bands of bin k's forward row */
    uint16_t *erb_fwd_hi;
    uint16_t *erb_fwd_k0;  /* [n_erb] bins feeding band b */
    uint16_t *erb_fwd_k1;
    uint16_t *erb_inv_lo;  /* [n_erb] bins of band b's inverse row */
    uint16_t *erb_inv_hi;

    void   *block;         /* the carved block, kept for dfn2_state_reset() */
    size_t  block_bytes;

    long long stream_frame_index;   /* stream API counter; mixing APIs needs dfn2_state_reset() */
#ifndef DFN2_MASK_ONLY
    int   df_ring_idx;     /* next write position */
    int   df_ring_count;   /* frames accumulated; warm-up ends when full */
#endif
} DFN2State;

/* Bytes dfn2_state_init() carves for `grid`, a multiple of 16; 0 on NULL or an invalid grid. */
size_t dfn2_state_mem_size(const DFN2Grid *grid);

/* Carve the arrays from `block` (16-byte aligned, `bytes` >= dfn2_state_mem_size(grid)), zero them
 * and set the normaliser starts. Returns 0, or -1 on NULL, an invalid grid or a short or misaligned
 * block (the header is then zeroed with no grid: any other call on it is a no-op or an error). */
int dfn2_state_init(DFN2State *st, void *block, size_t bytes, const DFN2Grid *grid);

void dfn2_state_reset(DFN2State *st);   /* init values; grid, pointers and borrowed matrices stay */

/* Borrow the caller-loaded ERB matrices (alive while used; the inverse a partition of unity) and
 * record their nonzero spans. Call before features or mask expansion, and after changing one. */
void dfn2_set_erb_matrices(DFN2State *st, const float *erb_fwd,
                             const float *erb_inv);

/* Spectrum -> the network's inputs feat_erb [n_erb] and feat_spec [2*df_bins] as [re..., im...].
 * Takes a normalized=True spectrum; train.py's analysis_scale is applied here. */
void dfn2_compute_features(DFN2State *st,
                           const float *spec_re, const float *spec_im,
                           float *feat_erb, float *feat_spec);

/* Mix noisy into enh by atten_lim_db (0 or non-finite: no-op), on the final complex spectrum. */
void dfn2_apply_atten_lim(int n_bins,
                          const float *noisy_re, const float *noisy_im,
                          float *enh_re, float *enh_im,
                          float atten_lim_db);

/* Network outputs -> enhanced spectrum, model.py compose() (mask-only: compose_mask_only(), noisy *
 * mask, no delay or warm-up; `coefs`, `alpha` ignored). erb_mask [n_erb] in [0,1]; coefs [df_bins *
 * df_order * 2] as model.py's (df_bins, df_order, 2); alpha 0 = masked residual, 1 = deep filter;
 * out_*: delayed by df_lookahead frames. Returns 1 when out is valid, 0 during warm-up (not for the
 * ISTFT) or on a NULL pointer or non-finite alpha. */
int dfn2_enhance_spectrum(DFN2State *st,
                 const float *spec_re, const float *spec_im,
                 const float *erb_mask, const float *coefs, float alpha,
                 float *out_re, float *out_im);

/* Streaming form: one CURRENT spectrum per call plus the heads as they arrive. heads_valid = 1: the
 * heads are those of frame current - mask_lookahead; 0 only for the first mask_lookahead frames
 * (head pointers may then be NULL). A valid out is frame current - mask_lookahead - df_lookahead,
 * reported by output_frame_index (may be NULL). Mask-only: the masked noisy head frame. No right
 * flush: keep calling with zero spectra and the accelerator's heads for them. atten_lim_db 0 = off,
 * else mixed against the delayed noisy target. Returns 1 = valid out, 0 = warm-up, -1 = pointer,
 * timing or non-finite alpha / atten_lim_db error. */
int dfn2_enhance_spectrum_stream(DFN2State *st,
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

const char *dfn2_simd_backend(void);

#endif /* DFN2_PROCESS_H */
