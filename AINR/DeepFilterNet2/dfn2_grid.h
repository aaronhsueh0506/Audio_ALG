/* DeepFilterNet2 grid: the signal grid and model geometry one build is made for.
 *
 * The grid is a compile-time contract between this library, the trained model and
 * its ERB matrices. It is chosen for the whole build (every translation unit of the
 * library and of the application passes the same choice) in one of two ways.
 *
 * 1) A built-in preset: -DDFN2_GRID_KHZ=16 | 32 | 48 (48 when nothing is chosen, the
 *    shipped model).
 *
 *      preset   sr      n_fft  hop   bins  n_erb  df_bins  alpha
 *      48       48000   1024   512   513   32     96       0.989
 *      32       32000   512    256   257   28     96       0.992   (not trained yet)
 *      16       16000   256    128   129   24     96       0.992
 *
 * 2) A grid defined outside the library, for a model trained on another grid:
 *    define the macros below on the build line (-DDFN2_SR=24000 ...), or put them
 *    in a header and name it (-DDFN2_GRID_HEADER='"my_grid.h"'). Nothing is
 *    defaulted for such a grid except where noted, so a value taken from another
 *    grid cannot sneak in; a missing one stops the build and names itself.
 *
 *      DFN2_SR                 sample rate in Hz                 (config.ini [signal] sr)
 *      DFN2_N_FFT              FFT size, power of two 16..8192   ([signal] n_fft)
 *      DFN2_N_ERB              ERB bands, a multiple of 4        ([model] n_erb)
 *      DFN2_DF_BINS            bins owned by the deep filter     ([model] df_bins)
 *      DFN2_ERB_NORM_ALPHA     ERB-feature EMA coefficient       (see below)
 *      DFN2_SPEC_NORM_ALPHA    complex-feature EMA coefficient   (see below)
 *    Plain decimal integers for the four integers: they are pasted into the link
 *    symbol (dfn2_process.h), so an expression stops the build.
 *
 * hop = n_fft/2 and win = n_fft are fixed by the framing (audiolib audio_stft.h:
 * 50% overlap, sqrt-Hann, no zero padding), so they are derived, not configured.
 *
 * Both alphas are train.py make_norm_alpha(sr, hop, tau): exp(-(hop/sr)/tau)
 * rounded to three decimals (one more digit if that reaches 1.0). tau = 1.0 s gives
 * 0.989 at hop 512 @ 48 kHz and 0.992 at hop 128 @ 16 kHz or hop 256 @ 32 kHz. They
 * are written out, not computed, so they cannot depend on the compiler; the check
 * against train.py is AINR/tests/test_dfn2_grid.py. Change them with tau.
 *
 * The lookahead pair, DFN2_MASK_ONLY and the normaliser starting points
 * (DFN2_ERB_NORM_INIT_*, DFN2_ERB_NORM_SCALE_DB, DFN2_SPEC_NORM_INIT_*) stay where
 * they are (dfn2_process.h); they are separate build choices and are overridable on
 * the build line. A build that overrides a starting point is not the shipped grid,
 * even on the 48 preset. */

#ifndef DFN2_GRID_H
#define DFN2_GRID_H

#include <math.h>

#ifdef DFN2_GRID_HEADER
#include DFN2_GRID_HEADER
#endif

#ifdef DFN2_SR
/* ---- A grid defined outside the library ---- */
#ifdef DFN2_GRID_KHZ
#error "define either DFN2_GRID_KHZ (a built-in preset) or the grid macros (DFN2_SR, ...), not both"
#endif
#ifndef DFN2_N_FFT
#error "DFN2 grid: DFN2_N_FFT is not defined (config.ini [signal] n_fft)"
#endif
#ifndef DFN2_N_ERB
#error "DFN2 grid: DFN2_N_ERB is not defined (config.ini [model] n_erb)"
#endif
#ifndef DFN2_DF_BINS
#error "DFN2 grid: DFN2_DF_BINS is not defined (config.ini [model] df_bins)"
#endif
#ifndef DFN2_ERB_NORM_ALPHA
#error "DFN2 grid: DFN2_ERB_NORM_ALPHA is not defined (exp(-(hop/sr)/tau) rounded to 3 decimals, e.g. 0.992f)"
#endif
#ifndef DFN2_SPEC_NORM_ALPHA
#error "DFN2 grid: DFN2_SPEC_NORM_ALPHA is not defined (exp(-(hop/sr)/tau) rounded to 3 decimals, e.g. 0.992f)"
#endif
/* Even with the shipped integers, a grid defined here is not the shipped one: its alphas
 * and starting points are the definer's, so it gets its own link symbol and pool hash. */
#define DFN2_GRID_IS_SHIPPED 0
#else
/* ---- Built-in presets (48 first: it is the shipped grid and the default) ---- */
#ifndef DFN2_GRID_KHZ
#define DFN2_GRID_KHZ 48
#endif
#if defined(DFN2_N_FFT) || defined(DFN2_N_ERB) || defined(DFN2_DF_BINS) || \
    defined(DFN2_ERB_NORM_ALPHA) || defined(DFN2_SPEC_NORM_ALPHA)
#error "a DFN2_GRID_KHZ preset is complete: to change one value define the whole grid, starting with DFN2_SR"
#endif
#if DFN2_GRID_KHZ == 48
#define DFN2_SR                 48000
#define DFN2_N_FFT              1024
#define DFN2_N_ERB              32
#define DFN2_DF_BINS            96    /* bins owned by the deep filter (0..95) */
#define DFN2_ERB_NORM_ALPHA     0.989f
#define DFN2_SPEC_NORM_ALPHA    0.989f
/* Only this preset is the shipped grid, and only while the build line leaves the
 * normaliser starting points alone: its link symbol and pool hash are the
 * historical ones. */
#if defined(DFN2_ERB_NORM_INIT_LO_DB) || defined(DFN2_ERB_NORM_INIT_HI_DB) || \
    defined(DFN2_ERB_NORM_SCALE_DB) || defined(DFN2_SPEC_NORM_INIT_LO) || \
    defined(DFN2_SPEC_NORM_INIT_HI)
#define DFN2_GRID_IS_SHIPPED    0
#else
#define DFN2_GRID_IS_SHIPPED    1
#endif
#elif DFN2_GRID_KHZ == 32
/* Provisional: no 32 kHz model is trained yet. 28 bands is a multiple of 4, which is all train.py and this library ask for. */
#define DFN2_SR                 32000
#define DFN2_N_FFT              512
#define DFN2_N_ERB              28
#define DFN2_DF_BINS            96
#define DFN2_ERB_NORM_ALPHA     0.992f
#define DFN2_SPEC_NORM_ALPHA    0.992f
#define DFN2_GRID_IS_SHIPPED    0
#elif DFN2_GRID_KHZ == 16
#define DFN2_SR                 16000
#define DFN2_N_FFT              256
#define DFN2_N_ERB              24
#define DFN2_DF_BINS            96
#define DFN2_ERB_NORM_ALPHA     0.992f
#define DFN2_SPEC_NORM_ALPHA    0.992f
#define DFN2_GRID_IS_SHIPPED    0
#else
#error "DFN2_GRID_KHZ must be 16, 32 or 48 (or define the grid macros for another grid)"
#endif
#endif

/* ---- Derived: the framing fixes these from n_fft ---- */
#define DFN2_N_BINS         ((DFN2_N_FFT) / 2 + 1)
#define DFN2_WIN_LEN        (DFN2_N_FFT)       /* analysis window length (= N_FFT) */
#define DFN2_HOP_LEN        ((DFN2_N_FFT) / 2) /* hop (= WIN_LEN/2, COLA) */
#define DFN2_OVL_LEN        (DFN2_WIN_LEN - DFN2_HOP_LEN)

/* torch.stft(normalized=True) multiplies by 1/sqrt(n_fft); a spectrum on the audio
 * scale is taken to the model's scale by DFN2_NORMALIZED_STFT_SCALE and back by
 * DFN2_NORMALIZED_STFT_UNSCALE. float32 division and square root are correctly
 * rounded, so these are the float32 values of 1/sqrt(n_fft) and sqrt(n_fft) on every
 * compiler (exact powers of two at 16, 64, 256, 1024, 4096). The framing accepts
 * n_fft a power of two in [16, 8192] (audiolib audio_stft.h). */
#if DFN2_N_FFT < 16 || DFN2_N_FFT > 8192 || ((DFN2_N_FFT) & ((DFN2_N_FFT) - 1)) != 0
#error "DFN2_N_FFT must be a power of two in [16, 8192] (audiolib audio_stft.h)"
#endif
#define DFN2_NORMALIZED_STFT_SCALE    (1.0f / sqrtf((float)(DFN2_N_FFT)))
#define DFN2_NORMALIZED_STFT_UNSCALE  sqrtf((float)(DFN2_N_FFT))

/* libDF analysis scale: the reference multiplies its spectrum by wnorm = 2*hop/win^2
 * while torch.stft(normalized=True) applies 1/sqrt(n_fft); the residual that feeds
 * the normalisers is 2*hop/win^2*sqrt(n_fft) (train.py analysis_scale). With
 * win = n_fft and hop = n_fft/2 that is exactly 1/sqrt(n_fft): 1/32 at 1024, 1/16 at
 * 256. It matters on the DF path (band_unit_norm is x/sqrt(EMA|x|), so an input
 * scale persists as sqrt(c)). Applied only to the copy feeding the normalisers; the
 * spectrum used for masking and ISTFT keeps its scale. */
#define DFN2_ANALYSIS_SCALE   DFN2_NORMALIZED_STFT_SCALE

/* ---- Guards: what the C path itself depends on ---- */
#if DFN2_SR <= 0
#error "DFN2_SR must be positive"
#endif
#if (DFN2_N_ERB) <= 0 || ((DFN2_N_ERB) % 4) != 0
#error "DFN2_N_ERB must be a positive multiple of 4: the ERB kernel runs bands in fours and the encoder halves the band axis twice"
#endif
#if 2 * (DFN2_N_ERB) > DFN2_N_BINS
#error "DFN2_N_ERB: every ERB band needs at least two bins (2 * n_erb <= n_fft/2 + 1)"
#endif
#if (DFN2_DF_BINS) < 2 || (DFN2_DF_BINS) > DFN2_N_BINS
#error "DFN2_DF_BINS must be in [2, n_fft/2 + 1]: the spectrum normaliser's start values are spread over bins 0..df_bins-1"
#endif
#if !defined(DFN2_MASK_ONLY) && (DFN2_DF_BINS) >= DFN2_N_BINS
#error "DFN2_DF_BINS must be below n_fft/2 + 1: bins above the deep filter keep the masked spectrum"
#endif

#endif /* DFN2_GRID_H */
