/* DeepFilterNet2 grid: the signal grid, model geometry and lookahead pair one instance
 * runs, as a run-time value.
 *
 * The library holds no grid. The application (a board sample, a test, a pipeline) decides
 * which grid it runs, builds a DFN2Grid with dfn2_grid_make() and passes it to every object
 * it creates (dfn2_state_init, dfn2_model_io_init, dfn2_prepost_config_defaults, the model-io
 * descriptor, dfn2_framing_config). The same binary therefore runs any grid whose model and
 * ERB matrices it is given: which (sample rate, fft size) pair maps to which n_erb, df_bins
 * and lookahead pair is the application's table, because it is a property of the trained
 * model, not of this code.
 *
 * dfn2_grid_make() takes the six values a trained model fixes and derives the rest by
 * formula:
 *
 *   sample_rate, n_fft, n_erb, df_bins, mask_lookahead, df_lookahead   (given)
 *   hop      = n_fft / 2          50% overlap sqrt-Hann framing (audiolib audio_stft.h)
 *   n_bins   = n_fft / 2 + 1
 *   stft_scale   = 1 / sqrt(n_fft)   torch.stft(normalized=True) analysis scale, also the
 *                                    scale of the feature branch (train.py analysis_scale)
 *   stft_unscale = sqrt(n_fft)       the synthesis scale
 *   erb_norm_alpha = spec_norm_alpha = train.py make_norm_alpha(sample_rate, hop, tau=1.0):
 *                    exp(-(hop / sample_rate) / tau) rounded to three decimals, one more
 *                    digit while that reaches 1.0 (0.989 for hop 512 at 48 kHz, 0.992 for
 *                    hop 128 at 16 kHz and hop 256 at 32 kHz). A model trained with another
 *                    tau overwrites the two fields (dfn2_norm_alpha() computes them).
 *   df_history, ring               deep-filter window and ring depth, from the lookahead pair
 *
 * What stays a compile-time constant of the library is what the state layout and the graph
 * are built on: DFN2_DF_ORDER, the capacities below, DFN2_MASK_ONLY, DFN2_FRAMES_PER_CALL
 * (dfn2_model_io.h) and the graph geometry (GRU hidden size, encoder channels). */

#ifndef DFN2_GRID_H
#define DFN2_GRID_H

#ifdef __cplusplus
extern "C" {
#endif

/* The deep filter has five taps: the graph's pathway window, the state layout and every ring
 * depth are built for it, so it is not a grid choice. */
#ifdef DFN2_DF_ORDER
#error "DFN2_DF_ORDER is fixed at 5 (the graph's pathway window and the state layout); it is not a grid choice"
#endif
#define DFN2_DF_ORDER       5     /* FIR taps */

/* ---- Capacity ------------------------------------------------------------
 * The state and the accelerator-boundary arrays are plain structs sized for the largest
 * grid the build serves; a grid runs in the first part of each array. The defaults hold
 * every grid up to 1024-point frames, 32 ERB bands, 96 deep-filter bins and a ring of five
 * frames (every lookahead pair with mask + df <= 4: ring = max(DF_ORDER, mask + df + 1), so
 * 1/4 and 2/3 need 6 and 2/4 needs 7). A
 * grid above a capacity is refused by dfn2_grid_make() (DFN2_GRID_ERR_CAPACITY); serving
 * it takes one definition on every translation unit of the build, e.g.
 * -DDFN2_MAX_N_FFT=2048, and costs that RAM for every grid. They are capacities, never
 * the grid a program runs. With -DDFN2_MASK_ONLY there is no deep-filter ring, only the
 * noisy ring of mask + 1 frames. */
#ifndef DFN2_MAX_N_FFT
#define DFN2_MAX_N_FFT      1024
#endif
#ifndef DFN2_MAX_N_ERB
#define DFN2_MAX_N_ERB      32
#endif
#ifndef DFN2_MAX_DF_BINS
#define DFN2_MAX_DF_BINS    96
#endif
#ifndef DFN2_MAX_DF_RING
#ifdef DFN2_MASK_ONLY
#define DFN2_MAX_DF_RING    2
#else
#define DFN2_MAX_DF_RING    5
#endif
#endif
#define DFN2_MAX_N_BINS     ((DFN2_MAX_N_FFT) / 2 + 1)

/* Lookahead bounds the graph allows: the mask path reads a three-frame window, and the deep
 * filter keeps at least its current tap. */
#define DFN2_MAX_MASK_LOOKAHEAD  2
#define DFN2_MAX_DF_LOOKAHEAD    (DFN2_DF_ORDER - 1)

typedef struct DFN2Grid {
    int   sample_rate;
    int   n_fft;           /* frame size = analysis window length */
    int   hop;             /* n_fft / 2 */
    int   n_bins;          /* n_fft / 2 + 1 */
    int   n_erb;
    int   df_bins;         /* bins owned by the deep filter (the lowest ones) */
    int   mask_lookahead;  /* future frames of the graph's input window (0..2) */
    int   df_lookahead;    /* future deep-filter taps (0..DF_ORDER-1; 0 when mask-only) */
    int   df_history;      /* DF_ORDER - df_lookahead - 1 past taps */
    int   ring;            /* frames the state rings hold */
    float erb_norm_alpha;
    float spec_norm_alpha;
    float stft_scale;      /* 1 / sqrt(n_fft) */
    float stft_unscale;    /* sqrt(n_fft) */
} DFN2Grid;

enum {
    DFN2_GRID_OK = 0,
    DFN2_GRID_ERR_NULL = -1,
    DFN2_GRID_ERR_SAMPLE_RATE = -2,
    DFN2_GRID_ERR_N_FFT = -3,
    DFN2_GRID_ERR_N_ERB = -4,
    DFN2_GRID_ERR_DF_BINS = -5,
    DFN2_GRID_ERR_MASK_LOOKAHEAD = -6,
    DFN2_GRID_ERR_DF_LOOKAHEAD = -7,
    DFN2_GRID_ERR_CAPACITY = -8,
    DFN2_GRID_ERR_INCONSISTENT = -9
};

/* Derive a grid from the six values a trained model fixes and validate it. Returns
 * DFN2_GRID_OK (the grid is written), or the first failing check (grid untouched):
 *   sample_rate > 0, and giving a norm alpha strictly between 0 and 1
 *   n_fft a power of two in [16, 8192] (audiolib audio_stft.h)
 *   n_erb a positive multiple of 4 (the ERB kernel runs bands in fours and the encoder
 *     halves the band axis twice) with every band at least two bins: 2 * n_erb <= n_bins
 *   df_bins in [2, n_bins] (the spectrum normaliser's start values are spread over bins
 *     0..df_bins-1), below n_bins unless -DDFN2_MASK_ONLY (bins above the deep filter keep
 *     the masked spectrum)
 *   mask_lookahead in [0, 2]; df_lookahead in [0, DF_ORDER - 1], 0 with -DDFN2_MASK_ONLY
 *     (no deep filter to look ahead for)
 *   n_fft, n_erb, df_bins and the ring depth within the capacities above
 * The alphas use tau = 1.0 s, the train.py default. */
int dfn2_grid_make(DFN2Grid *grid, int sample_rate, int n_fft, int n_erb,
                   int df_bins, int mask_lookahead, int df_lookahead);

/* The same checks on an existing grid: the derived fields must equal what
 * dfn2_grid_make() derives from the given ones (a hand-edited hop, ring or scale is
 * refused), the alphas must be finite and in (0, 1). Every library entry point that
 * takes a grid runs it. Returns DFN2_GRID_OK or the failing check. */
int dfn2_grid_validate(const DFN2Grid *grid);

/* train.py make_norm_alpha(): exp(-(hop / sample_rate) / tau_sec) rounded to three
 * decimals, one more digit while the rounded value reaches 1.0, as float32. Rounds the
 * exact binary value half to even as Python's round() does. */
float dfn2_norm_alpha(int sample_rate, int hop, double tau_sec);

/* A short name for a status, for a log line. Never NULL. */
const char *dfn2_grid_status_text(int status);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_GRID_H */
