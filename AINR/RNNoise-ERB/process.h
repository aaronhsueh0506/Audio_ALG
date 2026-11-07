#ifndef RNNOISE_PROCESS_H
#define RNNOISE_PROCESS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RNNoise-ERB pre/post-processing, spectrum domain only: the C twin of inference.py, equal to
 * float32 ULP (not bitwise: matmul accumulation order differs). The application owns audiolib's
 * AcStft/AcIstft, the recurrent state and the ERB matrices (loaded data; no file I/O here).
 * Framing contract (nothing here wraps it): ac_stft_config_default(&sc, grid.n_fft) with
 *   sc.spec_scale = grid.stft_scale, one full n_fft frame per hop (hop = n_fft / 2), and the
 *   defaults kept: center 0, periodic sqrt-Hann window on both sides, sc.ola =
 *   AC_OLA_ACCUMULATE (no envelope division), sc.ispec_scale = 1.0f, because
 *   rnnoise_apply_gains() already multiplies by sqrt(n_fft). */

/* Per frame: rnnoise_compute_features() (1 once a 3-frame window exists), the accelerator
 * (erb, spec, h1..h3 -> gains, h1..h3, same buffers), rnnoise_frame_finish(), optionally
 * rnnoise_apply_atten_lim(), rnnoise_expand_gains(), rnnoise_apply_gains(), the AcIstft. */

/* Feature tag: keep identical to config.ini [feature] and train.py's checkpoint validation. */
#define RNNOISE_FEATURE_VERSION          "log_erb_dfn_mean_cplx_unit_0_4k_v8"

#define RNNOISE_PREPOST_LAYOUT_VERSION   2
#define RNNOISE_MODEL_IO_LAYOUT_VERSION  1
#define RNNOISE_MODEL_INPUT_FRAMES       3   /* temporal kernel of the graph's input convolution */
#define RNNOISE_MEM_REQ_VERSION          1u
#define RNNOISE_CARVE_VERSION            1u  /* bump on any change of pool size, carve order, hash */
#define RNNOISE_ALIGNMENT                16u /* shared by every module (audiolib mem_align.h) */

/* The config.ini [signal] and [feature] values the C math uses, by key name (sample_rate is sr).
 * Norm start values are checkpoint-specific (calibrate_norm_init.py): recalibrate per checkpoint. */
typedef struct RNNoiseGridSpec {
    int    sample_rate;
    int    n_fft;               /* validated by ac_stft_get_mem_size() */
    int    n_bands;
    int    min_bins_per_band;   /* shaped the loaded matrix; checked and hashed only */
    int    lookahead_frames;    /* 0..2 */
    int    spec_max_hz;         /* highest frequency of the complex branch */
    double erb_norm_tau_sec;
    float  erb_norm_init_lo_db; /* start of band 0 */
    float  erb_norm_init_hi_db; /* start of the last band */
    float  erb_norm_scale_db;
    double spec_norm_tau_sec;
    float  spec_norm_init_lo;   /* start of bin 0 */
    float  spec_norm_init_hi;   /* start of the last complex bin */
    float  spec_norm_eps;
} RNNoiseGridSpec;

typedef struct RNNoiseGrid {
    int    sample_rate;
    int    n_fft;
    int    hop;                 /* n_fft / 2 */
    int    n_bins;              /* n_fft / 2 + 1 */
    int    n_bands;
    int    min_bins_per_band;
    int    lookahead_frames;
    /* The gain of the newest window [t-2, t-1, t] belongs to spectrum t - lookahead_frames,
     * window slot conv_delay (0 = oldest): the caller keeps and delays that spectrum. */
    int    conv_delay;          /* 2 - lookahead_frames */
    int    spec_max_hz;
    int    spec_bins;           /* spec_max_hz * n_fft / sample_rate + 1 (train.py) */
    double erb_norm_tau_sec;
    double spec_norm_tau_sec;
    float  erb_norm_alpha;      /* rnnoise_norm_alpha(sample_rate, hop, tau) */
    float  erb_norm_init_lo_db;
    float  erb_norm_init_hi_db;
    float  erb_norm_scale_db;
    float  spec_norm_alpha;
    float  spec_norm_init_lo;
    float  spec_norm_init_hi;
    float  spec_norm_eps;
    float  stft_scale;          /* 1 / sqrt(n_fft): the analysis spec_scale */
    float  stft_unscale;        /* sqrt(n_fft): folded into rnnoise_apply_gains() */
} RNNoiseGrid;

enum {
    RNNOISE_GRID_OK = 0,
    RNNOISE_GRID_ERR_NULL = -1,
    RNNOISE_GRID_ERR_SAMPLE_RATE = -2,   /* needs sample_rate > 0 */
    RNNOISE_GRID_ERR_N_FFT = -3,         /* needs n_fft accepted by ac_stft_get_mem_size() */
    RNNOISE_GRID_ERR_N_BANDS = -4,       /* needs n_bands >= 2, (n_bands-1)*min_bins_per_band <= n_bins */
    RNNOISE_GRID_ERR_MIN_BINS = -5,      /* needs min_bins_per_band >= 1 */
    RNNOISE_GRID_ERR_LOOKAHEAD = -6,     /* needs lookahead_frames in [0, 2] */
    RNNOISE_GRID_ERR_SPEC_MAX_HZ = -7,   /* needs spec_max_hz in (0, sample_rate / 2] */
    RNNOISE_GRID_ERR_ERB_NORM = -8,      /* needs tau, scale finite > 0; starts finite; alpha in (0, 1) */
    RNNOISE_GRID_ERR_SPEC_NORM = -9,     /* needs tau, starts, eps finite > 0; alpha in (0, 1) */
    RNNOISE_GRID_ERR_INCONSISTENT = -10  /* validate only: a derived field differs from make's */
};

/* Derive and validate a grid: RNNOISE_GRID_OK (written) or the first failing status (untouched). */
int rnnoise_grid_make(RNNoiseGrid *grid, const RNNoiseGridSpec *spec);

/* The same checks on an existing grid, and every derived field must equal what
 * rnnoise_grid_make() derives (a hand-edited hop, alpha or scale is refused). */
int rnnoise_grid_validate(const RNNoiseGrid *grid);

/* train.py make_norm_alpha(): exp(-(hop / sample_rate) / tau_sec) rounded to 3 decimals (one
 * more digit while it reaches 1.0) as float32, half to even on the exact value. 0 if invalid. */
float rnnoise_norm_alpha(int sample_rate, int hop, double tau_sec);

/* Short text for a status code, for logs. Never NULL. */
const char *rnnoise_grid_status_text(int status);

/* Lengths of out_erb and out_spec (flat), for binding tensors by count. */
static inline size_t rnnoise_erb_window_elements(const RNNoiseGrid *grid)
{
    return (size_t)RNNOISE_MODEL_INPUT_FRAMES * (size_t)grid->n_bands;
}
static inline size_t rnnoise_spec_window_elements(const RNNoiseGrid *grid)
{
    return (size_t)RNNOISE_MODEL_INPUT_FRAMES * 2u * (size_t)grid->spec_bins;
}

typedef struct RNNoiseConfig {
    RNNoiseGrid  grid;      /* rnnoise_grid_make(); copied at init */
    const float *erb_fwd;   /* required, borrowed; train.py mode 0, bin-major [n_bins][n_bands] */
    const float *erb_inv;   /* required, borrowed; train.py mode 1, band-major [n_bands][n_bins] */
    /* Optional registration of the application's recurrent state (borrowed; element counts):
     * init and reset zero it, every rnnoise_frame_finish() finite-checks it whole. The graph
     * binds each h*_in and h*_out to one buffer and reads every state input before writing any
     * state output. A board whose accelerator owns the state registers nothing. All three
     * pairs or none (a pointer with count 0 is refused); not part of the pool size or hash. */
    float       *h1;
    size_t       h1_elements;
    float       *h2;
    size_t       h2_elements;
    float       *h3;
    size_t       h3_elements;
} RNNoiseConfig;

/* Fixed 24-byte pool descriptor; rnnoise_init_ex() refuses a pool of another grid or build. */
typedef struct RNNoiseMemReq {
    uint32_t descriptor_version;  /* RNNOISE_MEM_REQ_VERSION              */
    uint32_t layout_version;      /* RNNOISE_MODEL_IO_LAYOUT_VERSION      */
    uint32_t build_flags_hash;    /* FNV-1a-32 of the grid, layout, carve and feature versions */
    uint32_t alignment;
    uint64_t bytes;
} RNNoiseMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(RNNoiseMemReq) == 24,
               "RNNoiseMemReq must stay 24 bytes");
#endif
#endif

/* The pool's header. Its arrays are carved from the pool; the ERB matrices and the
 * registered tensors are borrowed, never created or released here. */
typedef struct RNNoiseState {
    RNNoiseGrid  grid;

    const float *erb_fwd;
    const float *erb_inv;
    /* Nonzero bin span of each band, derived at init; skipping the zeros outside changes no sum. */
    uint16_t    *fwd_k0;              /* [n_bands]: forward band b covers [fwd_k0[b], fwd_k1[b]) */
    uint16_t    *fwd_k1;
    uint16_t    *inv_lo;              /* [n_bands]: inverse band b covers [inv_lo[b], inv_hi[b]) */
    uint16_t    *inv_hi;

    float       *erb_feat_buf;        /* [3][n_bands], ring */
    float       *spec_feat_buf;       /* [3][2][spec_bins], ring */
    float       *erb_norm_state;      /* [n_bands]: per-band causal log-ERB mean EMA */
    float       *spec_norm_state;     /* [spec_bins]: per-bin complex-magnitude EMA (libDF) */
    float       *scratch_power;       /* [n_bins], per-call scratch */
    float       *scratch_erb_db;      /* [n_bands], per-call scratch */

    int          feat_idx;            /* next ring write position (0..2) */
    int          feat_count;          /* frames accumulated, saturates at 3 */

    /* Registered recurrent state; NULL/0 when none. */
    float       *h1;
    size_t       h1_elements;
    float       *h2;
    size_t       h2_elements;
    float       *h3;
    size_t       h3_elements;
    unsigned int state_generation;

    void        *mutable_base;        /* cleared by rnnoise_reset() */
    size_t       mutable_bytes;
} RNNoiseState;

/* Fill a config for `grid` (copied); the matrices and the state registration stay NULL/0.
 * Returns 0, or -1 on NULL or an invalid grid. */
int rnnoise_config_defaults(RNNoiseConfig *cfg, const RNNoiseGrid *grid);

/* Exact pool size (header and arrays, each 16-byte aligned). Returns 0, or -1 (req
 * untouched) on NULL, an invalid grid, missing ERB matrices or an inconsistent registration. */
int rnnoise_get_mem_size(const RNNoiseConfig *cfg, RNNoiseMemReq *req);

/* Construct in caller memory (16-byte aligned, at least req.bytes); the state is the pool's
 * first bytes. NULL on a bad config or a misaligned or undersized pool. */
RNNoiseState *rnnoise_init(void *pool, size_t bytes, const RNNoiseConfig *cfg);

/* As rnnoise_init(), plus the stale-pool gate; `expected` NULL behaves as rnnoise_init(). */
RNNoiseState *rnnoise_init_ex(void *pool, size_t bytes, const RNNoiseConfig *cfg,
                              const RNNoiseMemReq *expected);

/* Clear the feature ring, restore the normaliser start values, zero the registered state and
 * advance the generation (config and borrowed pointers kept). Reset the AcStft/AcIstft too. */
void rnnoise_reset(RNNoiseState *st);

/* Generation of the recurrent state: 1 after init, plus one per reset and per refused
 * rnnoise_frame_finish(), modulo 2^32; 0 for NULL. When it changes the application refills
 * the state the library does not hold: unregistered accelerator buffers with zeros, a
 * fixed-point copy with quantized zeros. */
unsigned int rnnoise_state_generation(const RNNoiseState *st);

/* Dual-branch features (libDF band_mean_norm_erb / band_unit_norm, no clamps) of one
 * normalized spectrum (n_bins floats each; rfft of the sqrt-Hann block x grid.stft_scale).
 * out_erb [3][n_bands] and out_spec [3][2][spec_bins] (real plane, then imaginary, per
 * frame) hold the last 3 frames, oldest first. Returns 1 when available (3 frames), 0
 * otherwise (outputs unwritten), -1 on a NULL argument. A non-finite power reaches every band. */
int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float *out_erb, float *out_spec);

/* Close this frame's accelerator call: band_gains and the registered state must be all finite.
 * Otherwise the registered tensors are zeroed, the state generation advances and -1 is returned
 * (use unit gains). Returns 0 when finite, -1 on a NULL argument (nothing changed) or a refusal.
 * NaN-fill band_gains first to detect an unwritten output; a finite partial write is not detectable. */
int rnnoise_frame_finish(RNNoiseState *st, const float *band_gains);

/* band_gains[b] = lim + gain[b]*(1-lim), lim = 10^(-atten_lim_db/20) (inference.py's
 * attenuation-limit mixture); <= 0 disables. Returns 0, or -1 on a NULL argument or a
 * non-finite atten_lim_db (band_gains untouched). */
int rnnoise_apply_atten_lim(const RNNoiseState *st, float *band_gains,
                            float atten_lim_db);

/* Expand band gains to bin gains through the mode 1 inverse matrix (partition of unity:
 * gains = 1 gives bin_gains = 1). Returns 0, or -1 on a NULL argument. */
int rnnoise_expand_gains(const RNNoiseState *st, const float *band_gains,
                         float *bin_gains);

/* spec[k] *= bin_gains[k] * grid.stft_unscale, in place: the gain and the x sqrt(n_fft) of
 * torch.istft(normalized=True) in one product, so the AcIstft must not scale again.
 * Returns 0, or -1 on a NULL argument (spectra untouched). */
int rnnoise_apply_gains(const RNNoiseState *st, float *spec_re, float *spec_im,
                        const float *bin_gains);

/* Name of the compiled SIMD backend: "aarch64-neon", "aarch32-neon" or "scalar". */
const char *rnnoise_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* RNNOISE_PROCESS_H */
