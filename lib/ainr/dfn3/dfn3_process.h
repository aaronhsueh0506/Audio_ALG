/* DeepFilterNet3 signal path outside the network, on spectra (C twin of model.py heads()/compose(); reference
 * inference.py): ERB and complex features, mask expansion, deep filter, band split, attenuation limit.
 * Framing contract: spectra of grid.n_bins bins, one per hop, from the application's audiolib AcStft/AcIstft on the
 * default sqrt-Hann window, center 0, hop n_fft/2, accumulating overlap-add; analysis spec_scale = grid.stft_scale
 * (1/sqrt(n_fft)), synthesis ispec_scale = grid.stft_unscale (sqrt(n_fft)). A wrong scale or window is not detectable
 * here. The streaming framing differs from the training graph's center=True in the first frame only. */

#ifndef DFN3_PROCESS_H
#define DFN3_PROCESS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Validation bounds of dfn3_grid_make(); nothing is sized by them. */
#define DFN3_GRID_MASK_LOOKAHEAD_MAX  2    /* the graph's input window holds three frames */
#define DFN3_GRID_DF_ORDER_MAX        16

#define DFN3_FEATURE_VERSION  "dfn2_libdf_wnorm_upstream_init_no_eps_v5"   /* train.py's feature-contract tag */

#define DFN3_ERB_LOG_FLOOR    1e-10f   /* added to the band power before log10, as in train.py */
#define DFN3_SPEC_NORM_EPS    1e-12f   /* C only: keeps x/sqrt(0) finite where denormals flush to zero */

/* What a trained and exported model fixes (config.ini [signal], [model], [feature]). ERB starts are a linspace
 * over the bands (dB), spec starts a linspace over bins 0..df_bins-1 (magnitudes). */
typedef struct DFN3GridSpec {
    int   sample_rate;
    int   n_fft;
    int   n_erb;
    int   df_bins;
    int   df_order;
    int   mask_lookahead;
    int   df_lookahead;
    float erb_norm_tau_sec;       /* > 0 */
    float erb_norm_init_lo_db;    /* band 0 */
    float erb_norm_init_hi_db;    /* last band */
    float erb_norm_scale_db;      /* > 0 */
    float spec_norm_tau_sec;      /* > 0 */
    float spec_norm_init_lo;      /* bin 0, > 0 */
    float spec_norm_init_hi;      /* bin df_bins-1, > 0 */
} DFN3GridSpec;

typedef struct DFN3Grid {
    int   sample_rate;
    int   n_fft;
    int   hop;               /* n_fft / 2 */
    int   n_bins;            /* n_fft / 2 + 1 */
    int   n_erb;
    int   df_bins;
    int   df_order;
    int   mask_lookahead;
    int   df_lookahead;
    int   output_delay;      /* max(mask_lookahead, df_lookahead) frames */
    int   df_history;        /* df_order - df_lookahead - 1 past taps */
    int   ring;              /* df_order + output_delay - df_lookahead frames */
    float erb_norm_tau_sec;
    float erb_norm_init_lo_db;
    float erb_norm_init_hi_db;
    float erb_norm_scale_db;
    float spec_norm_tau_sec;
    float spec_norm_init_lo;
    float spec_norm_init_hi;
    float erb_norm_alpha;    /* dfn3_norm_alpha() of erb_norm_tau_sec */
    float spec_norm_alpha;   /* dfn3_norm_alpha() of spec_norm_tau_sec */
    float stft_scale;        /* 1 / sqrt(n_fft) */
    float stft_unscale;      /* sqrt(n_fft) */
} DFN3Grid;

enum {
    DFN3_GRID_OK = 0,
    DFN3_GRID_ERR_NULL = -1,
    DFN3_GRID_ERR_SAMPLE_RATE = -2,
    DFN3_GRID_ERR_N_FFT = -3,
    DFN3_GRID_ERR_N_ERB = -4,
    DFN3_GRID_ERR_DF_BINS = -5,
    DFN3_GRID_ERR_DF_ORDER = -6,
    DFN3_GRID_ERR_MASK_LOOKAHEAD = -7,
    DFN3_GRID_ERR_DF_LOOKAHEAD = -8,
    DFN3_GRID_ERR_ERB_NORM = -9,
    DFN3_GRID_ERR_SPEC_NORM = -10,
    DFN3_GRID_ERR_INCONSISTENT = -11
};

/* Derive a grid from the spec and validate it: DFN3_GRID_OK (grid written) or the first failing check (grid
 * untouched). sample_rate > 0; n_fft a size audiolib's framing builds, with bins that fit 16-bit ERB spans; n_erb a
 * positive multiple of 4 (the ERB kernel needs 4; the trainer requires 8) with at least two bins per band; df_bins in
 * [2, n_bins); df_order in [1, DFN3_GRID_DF_ORDER_MAX]; mask_lookahead in [0, DFN3_GRID_MASK_LOOKAHEAD_MAX] and
 * df_lookahead in [0, df_order - 1] (any pair, both run-time); each normaliser a finite tau > 0 with an alpha
 * strictly in (0, 1) and finite starts (spec starts and ERB scale > 0). */
int dfn3_grid_make(DFN3Grid *grid, const DFN3GridSpec *spec);

/* The same checks on an existing grid, whose derived fields must equal what dfn3_grid_make() derives. */
int dfn3_grid_validate(const DFN3Grid *grid);

/* Same value as train.py make_norm_alpha(): exp(-(hop / sample_rate) / tau_sec) rounded half to even to three
 * decimals, one more digit while that reaches 1.0, as float32. A tau that is not a float32 value (0.3) enters as its
 * float32 value. 0 for a non-positive argument. */
float dfn3_norm_alpha(int sample_rate, int hop, float tau_sec);

const char *dfn3_grid_status_text(int status);   /* never NULL */

/* ---- State ----
 * A header over arrays that dfn3_state_init() carves out of a caller block of dfn3_state_mem_size(grid) bytes,
 * kept alive and unchanged while the state is used. Rows are flat floats of the lengths noted per field. */
typedef struct DFN3State {
    DFN3Grid grid;
    const float *erb_fwd;       /* borrowed, bin-major [n_bins][n_erb]; kept by dfn3_state_reset() */
    const float *erb_inv;       /* borrowed, band-major [n_erb][n_bins] */

    float *erb_norm_state;      /* [n_erb] */
    float *spec_norm_state;     /* [df_bins] */

    /* Rings of `ring` frames; the low-bin rings hold the raw noisy spectrum (the ERB gain acts on high bins only). */
    float *df_ring_re;          /* [ring][df_bins] */
    float *df_ring_im;
    float *coef_ring;           /* [ring][df_bins][df_order][2] */
    float *noisy_ring_re;       /* [ring][n_bins] */
    float *noisy_ring_im;
    float *hi_delay_re;         /* [ring][n_bins - df_bins] masked high band */
    float *hi_delay_im;
    float *df_zero_row;         /* [df_bins], all zero: the taps before the stream start */
    const float **src_re;       /* [df_order] source rows of one filtered frame */
    const float **src_im;

    float *scratch_power;       /* [n_bins]; scratch, overwritten before each use */
    float *scratch_erb_db;      /* [n_erb] */

    /* Nonzero spans of the ERB matrices, derived by dfn3_set_erb_matrices(): bin k's forward row lies in bands
     * [erb_fwd_lo[k], erb_fwd_hi[k]), band b's inverse row in bins [erb_inv_lo[b], erb_inv_hi[b]), band b is fed by
     * bins [erb_fwd_k0[b], erb_fwd_k1[b]). Sums visit only these spans (skipped terms are finite values times +-0);
     * a non-finite power or band gain takes its whole row. Re-publish a matrix rewritten in place. */
    uint16_t *erb_fwd_lo;       /* [n_bins] */
    uint16_t *erb_fwd_hi;       /* [n_bins] */
    uint16_t *erb_fwd_k0;       /* [n_erb] */
    uint16_t *erb_fwd_k1;       /* [n_erb] */
    uint16_t *erb_inv_lo;       /* [n_erb] */
    uint16_t *erb_inv_hi;       /* [n_erb] */

    void   *block;              /* the carved block, kept for dfn3_state_reset() */
    size_t  block_bytes;

    long long stream_frame_index;   /* frames pushed through dfn3_enhance_spectrum_stream() */
    int   df_ring_idx;              /* dfn3_enhance_spectrum(): next write position */
    int   df_ring_count;            /* frames held, up to ring; output starts once it exceeds df_lookahead */
} DFN3State;

/* Bytes of the block dfn3_state_init() carves, a multiple of 16; 0 for a NULL or invalid grid. */
size_t dfn3_state_mem_size(const DFN3Grid *grid);

/* Carve the arrays out of `block` (16-byte aligned, `bytes` >= dfn3_state_mem_size(grid)), zero them, record
 * the grid and set the normalisers' initial values. Returns 0, or -1 on NULL, an invalid grid, or a short or
 * misaligned block (the header is then zeroed and every other call on it is a no-op or an error). */
int dfn3_state_init(DFN3State *st, void *block, size_t bytes, const DFN3Grid *grid);

/* Back to the init values: arrays zeroed, normalisers restarted. The header, grid, array pointers and ERB
 * matrices stay (the spans are derived again from them). A state without a grid is ignored. */
void dfn3_state_reset(DFN3State *st);

/* Point the state at the caller-loaded ERB matrices and record their nonzero spans. Call before feature
 * extraction or mask expansion, and again after any change to a matrix's contents. */
void dfn3_set_erb_matrices(DFN3State *st, const float *erb_fwd, const float *erb_inv);

/* Spectrum -> the network's inputs: feat_erb [n_erb] and feat_spec [2*df_bins] as [re..., im...]. Takes the framing
 * contract's normalized spectrum unchanged; the feature branch's libDF scale (train.py analysis_scale, equal to
 * grid.stft_scale for window length n_fft and hop n_fft / 2) is applied here. */
void dfn3_compute_features(DFN3State *st,
                           const float *spec_re, const float *spec_im,
                           float *feat_erb, float *feat_spec);

/* In place on enh: enh = lim * noisy + (1 - lim) * enh, lim = 10^(-|atten_lim_db| / 20); 0 or non-finite: no-op. */
void dfn3_apply_atten_lim(int n_bins,
                          const float *noisy_re, const float *noisy_im,
                          float *enh_re, float *enh_im,
                          float atten_lim_db);

/* Network heads -> enhanced spectrum (C twin of model.py compose()): the deep filter on the raw spectrum owns bins
 * below df_bins, the ERB mask the rest. erb_mask [n_erb] in [0,1]; coefs [df_bins][df_order][2]; spec_re/spec_im the
 * current frame at the framing scale, pushed into the ring; out_re/out_im the enhanced spectrum, df_lookahead frames
 * behind. Returns 1 when out is valid, 0 during warm-up (skip the ISTFT) or on invalid input. The heads are those of
 * the spectrum passed in the same call, but a live accelerator delivers heads[t] only after frame t + mask_lookahead:
 * on a live stream use this entry only with mask_lookahead 0, else dfn3_enhance_spectrum_stream(). */
int dfn3_enhance_spectrum(DFN3State *st,
                          const float *spec_re, const float *spec_im,
                          const float *erb_mask, const float *coefs,
                          float *out_re, float *out_im);

/* Streaming handoff: each call pushes one CURRENT spectrum. With heads_valid = 1, erb_mask/coefs are the heads for
 * current_frame - mask_lookahead; the first mask_lookahead calls pass heads_valid = 0 (heads may be NULL). The
 * latency is grid.output_delay frames; with df_lookahead > mask_lookahead the calls after the warm-up keep their
 * heads and return 0 until the first output frame is complete. output_frame_index may be NULL; atten_lim_db = 0
 * disables the mix. To flush, the accelerator produces the remaining heads from zero padding while the caller keeps
 * pushing zero spectra. Every spectrum and active head must be finite: invalid input is refused before any ring,
 * clock or output changes, so the frame can be retried. Do not mix with dfn3_enhance_spectrum() without
 * dfn3_state_reset(). Returns 1 = valid out, 0 = warm-up, -1 = invalid input or clock contract. */
int dfn3_enhance_spectrum_stream(DFN3State *st,
                                 const float *current_spec_re,
                                 const float *current_spec_im,
                                 int heads_valid,
                                 const float *erb_mask,
                                 const float *coefs,
                                 float atten_lim_db,
                                 float *out_re,
                                 float *out_im,
                                 long long *output_frame_index);

const char *dfn3_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* DFN3_PROCESS_H */
