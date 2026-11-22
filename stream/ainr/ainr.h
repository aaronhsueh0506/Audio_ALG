/* AINR customer interface (DeepFilterNet2 full / lite). The customer sets {sr, fft, hop_in} and the grid's ERB tables;
 * the product build binds the model, whose grid (dfn2_grids.h) and N = hop_in / (fft / 2) init resolves. Files (graph,
 * ERB matrices) and the accelerator (open graph, move tensors, run, read back) stay the sample's:
 *
 *     ...load fb_<sr>_<n_erb>.bin (graph_io_dfn2_erb_file(), dfn2_erb_file.h) into erb_fwd / erb_inv...
 *     AinrConfig cfg = { sr, fft, hop_in, erb_fwd, erb_inv, n_erb };
 *     ainr_get_mem_size(&cfg, &req);  ...allocate req.bytes at req.alignment...
 *     ainr_init(&cfg, pool, req.bytes, &ctx, &info);   // host only: pool, no SDK, no file
 *     ...open the file of the graph_io_dfn2.h row (sr, fft, N, info.mask_lookahead), check its ports against that row,
 *        seed the state sync from ainr_state_generation()...
 *     for each block of hop_in samples:
 *         ainr_preprocess(ctx, pcm_in, &work);          // work.action: SUBMIT or BYPASS
 *         ...SUBMIT: run the accelerator on work (state sync and readback recovery are the sample's);
 *            BYPASS: no accelerator call...
 *         ainr_postprocess(ctx, run_status, pcm_out);   // hop_in valid samples, BYPASS block included
 *     the output trails the input by info.queue_frames + 1 model hops
 *
 * Run status and state generation: STREAM_RUN_* in stream_run.h. The accelerator's state is zero when the generation
 * is seeded after init. The status of a BYPASS block is not read.
 *
 * A block whose run did not succeed still calls ainr_postprocess() with its run status. A config without a graph_io row
 * is refused by the sample before any SDK call. Every successful preprocess pairs with one postprocess, a SUBMIT block
 * also with one run. A BYPASS block is N == 1 with mask_lookahead > 0 and only one of the first mask_lookahead
 * blocks after init or reset: no job, its work is all zero (state_generation included), its output is silence.
 * N > 1 is one packet at any mask_lookahead: the N hops of a block are one SUBMIT job (never a BYPASS block) whose graph reads
 * N + 2 feature frames (the block's N and the two before) and returns N heads, for the frames mask_lookahead earlier. The output
 * has the latency of N == 1; with mask_lookahead > 0 the first block's first mask_lookahead heads are before the stream start
 * (silence), so the accelerator's state has seen that many more frames than at N == 1.
 * StreamWork (stream_run.h) has the same size in the full and the lite archive: a client linked to
 * the wrong variant sees another input_count/output_count, not an overrun. A context is used from one thread at a time. */

#ifndef AINR_H
#define AINR_H

#include <stddef.h>
#include <stdint.h>

#include "stream_mem.h"    /* StreamMemReq; include roots: stream/common and audiolib common/ */
#include "stream_run.h"    /* StreamWork, STREAM_RUN_*: the per-block run contract */

#ifdef __cplusplus
extern "C" {
#endif

#define AINR_ALIGNMENT 16u

typedef struct AinrConfig {
    int sr;        /* Hz */
    int fft;       /* FFT size; model_hop = fft / 2 */
    int hop_in;    /* new PCM samples per call: a positive multiple of model_hop, N = hop_in / model_hop */
    /* The grid's ERB tables, borrowed for the context's life (get_mem_size does not read them): erb_fwd bin-major
     * [n_bins][n_erb], erb_inv band-major [n_erb][n_bins], n_bins = fft / 2 + 1, as dfn2_erb_file.h loads them.
     * n_erb is the tables' band count; init refuses one other than the grid's. */
    const float *erb_fwd;
    const float *erb_inv;
    int          n_erb;
} AinrConfig;

enum {
    AINR_OK = 0,
    AINR_ERR_NULL = -1,
    AINR_ERR_MODEL_ID = -2,                     /* the bound model is not of this archive variant */
    AINR_ERR_GRID = -3,                         /* the bound model has no grid for (sr, fft) */
    AINR_ERR_IN_HOP = -4,                       /* hop_in is not a positive multiple of model_hop */
    AINR_ERR_FRAMES_PER_CALL = -6,              /* N beyond DFN2_MODEL_MAX_FRAMES_PER_CALL (64) */
    AINR_ERR_POOL = -7,                         /* pool too small or misaligned */
    AINR_ERR_ERB_TABLES = -8,                   /* erb_fwd / erb_inv NULL, n_erb not the grid's, a non-finite or negative forward value, or an inverse that is not a partition of unity */
    AINR_ERR_SEQUENCE = -9,                     /* preprocess/postprocess not called in pairs */
    AINR_ERR_NONFINITE_INPUT = -10              /* _freq bin NaN/Inf or beyond 2^63 * sqrt(fft); NaN/Inf atten_lim */
};

/* Short text for a status code, for a log line. Never NULL. */
const char *ainr_status_text(int code);

/* Result of one init, the same for both families of a config. */
typedef struct AinrInfo {
    int          queue_frames;         /* mask_lookahead + df_lookahead, for every N: the final PCM trails the input by queue_frames + 1
                                              model hops (_freq: after the caller's own ISTFT) */
    int          mask_lookahead;       /* of the bound model at this grid: the graph_io_dfn2.h key of its graph */
} AinrInfo;

/* ---- PCM family ----
 * int16 in and out (/ 32768 in, clipped out): a float caller pays one 16-bit requantization, about 1 LSB, per side. */

typedef struct AinrCtx AinrCtx;

/* Pool size and alignment for cfg: AINR_OK (req filled) or an AINR_ERR_* (req untouched). Checked in order:
 * model vs variant, grid, hop_in divisibility, N range. */
int ainr_get_mem_size(const AinrConfig *cfg, StreamMemReq *req);

/* Constructs in the pool (AINR_ALIGNMENT-aligned, >= ainr_get_mem_size() bytes) over cfg's ERB tables. AINR_OK sets
 * *ctx and *info; an AINR_ERR_* leaves *ctx NULL and *info untouched, and a small or misaligned pool or missing tables
 * leave the pool unwritten. */
int ainr_init(const AinrConfig *cfg, void *pool, size_t pool_bytes, AinrCtx **ctx, AinrInfo *info);

/* Back to the start: state zeroed, generation advanced, an open call dropped; ERB tables kept. */
void ainr_reset(AinrCtx *ctx);

/* Attenuation limit of |atten_lim_db| dB (sign ignored; 0 disables, the default), between blocks only.
 * AINR_ERR_SEQUENCE while a preprocess awaits its postprocess, AINR_ERR_NONFINITE_INPUT for NaN/Inf,
 * AINR_ERR_NULL for NULL; else 0. Kept by ainr_reset(). The _freq variant is the same. */
int ainr_set_atten_lim(AinrCtx *ctx, float atten_lim_db);

/* Current state generation (0 for NULL): seeds the sample's sync after init or reset; see the run-status rule above.
 * All-zero bytes are quantized zero: the port check refuses a fixed-point state port with zero_point != 0. */
unsigned int ainr_state_generation(const AinrCtx *ctx);

/* One block in: exactly cfg->hop_in samples. AINR_OK: work SUBMIT holds the block's one job (run it, then ainr_postprocess()),
 * work BYPASS holds none (skip the accelerator, call ainr_postprocess() all the same). AINR_ERR_SEQUENCE (no postprocess
 * since the last call) or AINR_ERR_NULL: no block is open. */
int ainr_preprocess(AinrCtx *ctx, const int16_t *pcm_in, StreamWork *work);

/* Closes the block with its run status: 0 commits (full: df_alpha clamped to [0, 1]), anything else bypasses with a
 * near-identity pass-through (unit ERB mask, alpha 0). Returns 0 with hop_in samples in pcm_out either way;
 * AINR_ERR_SEQUENCE without a matching preprocess (pcm_out untouched). */
int ainr_postprocess(AinrCtx *ctx, int run_status, int16_t *pcm_out);

/* ---- Spectrum -> spectrum ("_freq") family ----
 * Same config, AinrInfo and StreamWork as the PCM family; only the data crossing the block boundary differs.
 * Spectra are AUDIO scale (AcStft/AcIstft with scale 1.0, ac_stft_config_default(), the PCM scaled as
 * int16 / 32768), frame-major, oldest first, N frames of n_bins. The caller owns the analysis and synthesis:
 * it supplies an estimate spectrum E (features, mask, deep filter) and an apply spectrum P (what mask, deep
 * filter and attenuation limit act on) and does the ISTFT/OLA of the result; E == P (same pointers) is the
 * single-spectrum case. After ainr_reset_freq() reset the analyser too (ac_stft_reset()). A bypass delivers
 * P-aligned near-identity. Direct (non-bridge) caller: an upstream restart at input frame f belongs to the output
 * frame f + queue_frames (position f + queue_frames of the delivered frames, block (f + queue_frames) / N); clear the
 * caller's synthesis (ac_istft_reset()) just before pushing it. Through AinrBridge,
 * ainr_bridge_reset_synthesis() on the hop of the event is that same point; the model keeps running. */

typedef struct AinrFreqCtx AinrFreqCtx;

/* As ainr_get_mem_size(), for the spectrum -> spectrum family: a smaller pool, with no analysis, synthesis or PCM
 * buffer in it. */
int ainr_get_mem_size_freq(const AinrConfig *cfg, StreamMemReq *req);

/* As ainr_init(), for the spectrum -> spectrum family. */
int ainr_init_freq(const AinrConfig *cfg, void *pool, size_t pool_bytes, AinrFreqCtx **ctx, AinrInfo *info);

/* As ainr_reset(), for the spectrum -> spectrum family. */
void ainr_reset_freq(AinrFreqCtx *ctx);

/* As ainr_set_atten_lim(), for the spectrum -> spectrum family. */
int ainr_set_atten_lim_freq(AinrFreqCtx *ctx, float atten_lim_db);

/* As ainr_state_generation(), for the spectrum -> spectrum family. */
unsigned int ainr_state_generation_freq(const AinrFreqCtx *ctx);

/* As ainr_preprocess() for N frames each of est/app re/im. A bin that is NaN/Inf or beyond 2^63 * sqrt(fft) (its feature
 * power would overflow) refuses the block (AINR_ERR_NONFINITE_INPUT) before any frame is consumed: state untouched, no
 * block open, so no run and no postprocess_freq; queued frames come out a block late. */
int ainr_preprocess_freq(AinrFreqCtx *ctx, const float *est_re, const float *est_im,
                         const float *app_re, const float *app_im, StreamWork *work);

/* Closes the block like ainr_postprocess(), writing N frames of enhanced spectrum (out_re/out_im: N * n_bins
 * floats each). */
int ainr_postprocess_freq(AinrFreqCtx *ctx, int run_status, float *out_re, float *out_im);

#ifdef __cplusplus
}
#endif

#endif /* AINR_H */
