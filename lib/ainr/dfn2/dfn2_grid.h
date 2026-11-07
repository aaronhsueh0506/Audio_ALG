/* DeepFilterNet2 grid: the signal grid, model geometry, lookahead pair and normaliser settings one
 * instance runs, as a run-time value. The application builds a DFN2Grid with dfn2_grid_make() and
 * passes it to every object; it must equal the grid the checkpoint was trained and exported for.
 * Which (sample rate, fft size) maps to which geometry and settings is the application's table. */

#ifndef DFN2_GRID_H
#define DFN2_GRID_H

#ifdef __cplusplus
extern "C" {
#endif

/* Temporal kernel of the graph's input convolution: N heads need N + DFN2_MODEL_INPUT_FRAMES - 1
 * feature frames. */
#define DFN2_MODEL_INPUT_FRAMES         3

/* Validation bounds of dfn2_grid_make(): they reject what the model cannot build. */
#define DFN2_GRID_MASK_LOOKAHEAD_MAX    2      /* the mask path reads a three-frame window */
#define DFN2_GRID_DF_ORDER_MAX          16     /* no Python bound; keeps the rings sensible */
#define DFN2_MODEL_MAX_FRAMES_PER_CALL  64     /* export_onnx.py --frames-per-call */

/* What a trained and exported model fixes (config.ini [signal], [model], [feature]). df_order is
 * only range-checked (stored 0) with -DDFN2_MASK_ONLY; frames_per_call = heads per accelerator call. The
 * normaliser fields must equal the training values: time constants in seconds, linspace starts over
 * the bands (ERB, dB) and the deep-filter bins (spec, magnitude), the ERB divisor in dB. */
typedef struct DFN2GridSpec {
    int    sample_rate;
    int    n_fft;
    int    n_erb;
    int    df_bins;
    int    df_order;
    int    mask_lookahead;
    int    df_lookahead;
    int    frames_per_call;
    float  erb_norm_tau_sec;      /* > 0                    */
    float  erb_norm_init_lo_db;   /* band 0 start           */
    float  erb_norm_init_hi_db;   /* last band start        */
    float  erb_norm_scale_db;     /* > 0                    */
    float  spec_norm_tau_sec;     /* > 0                    */
    float  spec_norm_init_lo;     /* bin 0 start, > 0       */
    float  spec_norm_init_hi;     /* bin df_bins-1 start, > 0 */
} DFN2GridSpec;

typedef struct DFN2Grid {
    float  erb_norm_tau_sec;
    float  spec_norm_tau_sec;
    int    sample_rate;
    int    n_fft;           /* frame size = analysis window length */
    int    hop;             /* n_fft / 2 */
    int    n_bins;          /* n_fft / 2 + 1 */
    int    n_erb;
    int    df_bins;         /* bins owned by the deep filter (the lowest ones) */
    int    df_order;        /* deep-filter taps; 0 when mask-only */
    int    mask_lookahead;  /* future frames of the graph's input window (0..2) */
    int    df_lookahead;    /* future deep-filter taps (0..df_order-1; 0 when mask-only) */
    int    df_history;      /* df_order - df_lookahead - 1 past taps; 0 when mask-only */
    int    ring;            /* frames the state rings hold (dfn2_process.h) */
    int    frames_per_call; /* heads per accelerator call */
    int    input_frames;    /* frames_per_call + DFN2_MODEL_INPUT_FRAMES - 1 */
    float  erb_norm_init_lo_db;
    float  erb_norm_init_hi_db;
    float  erb_norm_scale_db;
    float  spec_norm_init_lo;
    float  spec_norm_init_hi;
    float  erb_norm_alpha;  /* dfn2_norm_alpha() of erb_norm_tau_sec */
    float  spec_norm_alpha; /* dfn2_norm_alpha() of spec_norm_tau_sec */
    float  stft_scale;      /* 1 / sqrt(n_fft): normalized=True analysis scale */
    float  stft_unscale;    /* sqrt(n_fft): the synthesis scale */
} DFN2Grid;

/* Status codes; -8 is unassigned and must not be reused. */
enum {
    DFN2_GRID_OK = 0,
    DFN2_GRID_ERR_NULL = -1,
    DFN2_GRID_ERR_SAMPLE_RATE = -2,
    DFN2_GRID_ERR_N_FFT = -3,
    DFN2_GRID_ERR_N_ERB = -4,
    DFN2_GRID_ERR_DF_BINS = -5,
    DFN2_GRID_ERR_MASK_LOOKAHEAD = -6,
    DFN2_GRID_ERR_DF_LOOKAHEAD = -7,
    DFN2_GRID_ERR_INCONSISTENT = -9,
    DFN2_GRID_ERR_DF_ORDER = -10,
    DFN2_GRID_ERR_FRAMES_PER_CALL = -11,
    DFN2_GRID_ERR_ERB_NORM = -12,
    DFN2_GRID_ERR_SPEC_NORM = -13
};

/* Derive a grid from the spec and validate it: DFN2_GRID_OK (grid written), or the first failing
 * check (grid untouched):
 *   sample_rate > 0; n_fft a size audiolib's framing builds (ac_stft_get_mem_size() is not 0) with
 *     n_bins <= 65535 (16-bit ERB spans)
 *   n_erb a positive multiple of 4 (the ERB kernel runs bands in fours, the encoder halves the band
 *     axis twice), every band at least two bins: 2 * n_erb <= n_bins
 *   df_bins in [2, n_bins], below n_bins unless -DDFN2_MASK_ONLY
 *   df_order in [1, 16]; with -DDFN2_MASK_ONLY any value in [0, 16], stored as 0
 *   mask_lookahead in [0, 2]; df_lookahead in [0, df_order - 1], 0 with -DDFN2_MASK_ONLY
 *   frames_per_call in [1, 64]
 *   each normaliser: a finite tau > 0 giving an alpha strictly in (0, 1), finite starts, and for
 *     the ERB one a finite scale > 0; the spec starts must be > 0 (the feature divides by their root) */
int dfn2_grid_make(DFN2Grid *grid, const DFN2GridSpec *spec);

/* The same checks on an existing grid, plus every derived field must equal what dfn2_grid_make()
 * derives (hand-edited hop, ring, scale or alpha: refused). Run by every entry point on a grid. */
int dfn2_grid_validate(const DFN2Grid *grid);

/* train.py make_norm_alpha(): exp(-(hop / sample_rate) / tau_sec) rounded to three decimals (one
 * more while it reaches 1.0), as float32; the exact binary value rounds half to even, as Python's
 * round(). A tau that is not a float32 value (0.3) enters as its float32 value, so a rare tau may
 * land one rounding step from make_norm_alpha on the decimal. */
float dfn2_norm_alpha(int sample_rate, int hop, float tau_sec);

/* A short name for a status, for a log line. Never NULL. */
const char *dfn2_grid_status_text(int status);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_GRID_H */
