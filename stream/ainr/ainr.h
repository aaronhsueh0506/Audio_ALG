/* AINR customer interface (DeepFilterNet2 full / lite). The customer sets {sr, fft, hop_in}; the product
 * build binds the model; ainr.c's catalog resolves the grid, N = hop_in / (fft / 2), the graph path and
 * the ERB file. The accelerator (open graph, move tensors, run, read back) stays the sample's:
 *
 *     AinrConfig cfg = { sr, fft, hop_in };
 *     ainr_get_mem_size(&cfg, &req);  ...allocate req.bytes at req.alignment...
 *     ainr_init(&cfg, pool, req.bytes, &ctx, &info);   // host only: ERB, pool, no SDK
 *     ...open info.graph_path, check its ports against the sample's model config for the same graph,
 *        seed the state sync from ainr_state_generation()...
 *     for each block of hop_in samples:
 *         ainr_preprocess(ctx, pcm_in, &work);          // always carries one job
 *         ...run the accelerator on work (state sync and readback recovery are the sample's)...
 *         ainr_postprocess(ctx, ai_rc, pcm_out);        // hop_in valid samples
 *     after the last real block (zero-padded to hop_in): info.tail_zero_blocks all-zero blocks through
 *         the same sequence, then drop info.trim_samples from the start of the output
 *
 * A failed run or readback still calls ainr_postprocess() with the failure status. An unregistered
 * config is refused before any SDK call. Every successful preprocess pairs with one run and one
 * postprocess. AinrWork has the same size in the full and the lite archive: a client linked to the
 * wrong variant sees another input_count/output_count, not an overrun.
 * File mode only: the pool size assumes the ERB tables load from a file. Under -DDFN2_STREAM_NO_STDIO
 * no init function is compiled in; use dfn2_stream_init() (dfn2/dfn2_stream.h) directly. */

#ifndef AINR_H
#define AINR_H

#include <stddef.h>
#include <stdint.h>

#include "stream_mem.h"    /* StreamMemReq */

#ifdef __cplusplus
extern "C" {
#endif

#define AINR_ALIGNMENT 16u

typedef struct AinrConfig {
    int sr;        /* Hz */
    int fft;       /* FFT size; model_hop = fft / 2 */
    int hop_in;    /* new PCM samples per call: a positive multiple of model_hop, N = hop_in / model_hop */
} AinrConfig;

enum {
    AINR_OK = 0,
    AINR_ERR_NULL = -1,
    AINR_ERR_MODEL_ID = -2,                     /* the bound model is not of this archive variant */
    AINR_ERR_GRID = -3,                         /* no catalog entry for (sr, fft) */
    AINR_ERR_IN_HOP = -4,                       /* hop_in is not a positive multiple of model_hop */
    AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED = -5,  /* mask_lookahead > 0 with N > 1 */
    AINR_ERR_FRAMES_PER_CALL = -6,              /* N has no delivered graph */
    AINR_ERR_POOL = -7,                         /* pool too small or misaligned */
    AINR_ERR_ERB_FILE = -8,                     /* ERB file missing, short or of another size */
    AINR_ERR_SEQUENCE = -9,                     /* preprocess/postprocess not called in pairs */
    AINR_ERR_NONFINITE_INPUT = -10              /* _freq bin NaN/Inf or beyond 2^63 * sqrt(fft); NaN/Inf atten_lim */
};

/* Short text for a status code, for a log line. Never NULL. */
const char *ainr_status_text(int code);

#define AINR_MAX_INPUT_TENSORS  2   /* erb window, spec window */
#define AINR_MAX_OUTPUT_TENSORS 3   /* full: erb_mask, coefs, alpha; lite: erb_mask */

typedef struct AinrTensorIn {
    const float *data;
    size_t elements;
    unsigned int w, h, c, b;   /* tensor shape, w innermost (the ONNX logical [b,c,h,w]) */
} AinrTensorIn;

typedef struct AinrTensorOut {
    float *data;
    size_t elements;
    unsigned int w, h, c, b;
} AinrTensorOut;

/* One block's tensors: inputs[i] is signal input i of the graph, outputs[j] head output j; the accelerator keeps the
 * recurrent state, state_generation says when to reload it with quantized zero. The data pointers are stream-owned,
 * valid until the matching postprocess: do not free them or write the inputs (the accelerator fills the outputs). */
typedef struct AinrWork {
    AinrTensorIn  inputs[AINR_MAX_INPUT_TENSORS];
    int           input_count;
    AinrTensorOut outputs[AINR_MAX_OUTPUT_TENSORS];
    int           output_count;
    /* Sampled every block, as DFN2HopIO.state_generation. */
    unsigned int  state_generation;
} AinrWork;

/* Tail: AC_OLA_ACCUMULATE has no flush. After the last real block push info.tail_zero_blocks zero blocks
 * (_spec: also through the caller's ISTFT), drop info.trim_samples from the start, keep the real length.
 * _freq: feed the upstream analyser the same zero blocks first, then drain the queue with the same count. */

/* Result of one init, the same for the three families of a config. */
typedef struct AinrInfo {
    const char  *graph_path;           /* model.bin of this configuration */
    int          trim_samples;         /* samples dropped from the start of the final PCM: (queue_frames + 1) * model_hop */
    int          queue_frames;         /* mask_lookahead + df_lookahead + N - 1 */
    int          tail_zero_blocks;     /* zero hop_in blocks after the last real block: ceil(trim_samples / hop_in) */
} AinrInfo;

/* ---- PCM family ---- */

typedef struct AinrCtx AinrCtx;

/* Pool size and alignment for cfg: AINR_OK (req filled) or an AINR_ERR_* (req untouched). Checked in order:
 * model vs variant, grid, hop_in divisibility, lookahead/N rule, shipped graph for N. */
int ainr_get_mem_size(const AinrConfig *cfg, StreamMemReq *req);

/* Constructs in the pool (AINR_ALIGNMENT-aligned, >= ainr_get_mem_size() bytes) and loads the grid's ERB
 * tables. AINR_OK sets *ctx and *info; an AINR_ERR_* leaves *ctx NULL and *info untouched, and a small or
 * misaligned pool is not written. */
int ainr_init(const AinrConfig *cfg, void *pool, size_t pool_bytes, AinrCtx **ctx, AinrInfo *info);

/* Back to the start: state zeroed, generation advanced, a partly gathered block dropped; ERB tables kept. */
void ainr_reset(AinrCtx *ctx);

/* Attenuation limit in dB (0 disables, the default), between blocks only. AINR_ERR_SEQUENCE while a
 * preprocess awaits its postprocess, AINR_ERR_NONFINITE_INPUT for NaN/Inf, AINR_ERR_NULL for NULL; else
 * 0. Kept by ainr_reset(). The _spec/_freq variants are the same. */
int ainr_set_atten_lim(AinrCtx *ctx, float atten_lim_db);

/* Current state generation (0 for NULL): seeds the sample's sync after init or reset. It advances on reset,
 * on each warm-up hop and on a refused commit (non-finite heads), not on a caller-signaled bypass. Reload
 * quantized zero on a change and after any failed run or readback. */
unsigned int ainr_state_generation(const AinrCtx *ctx);

/* One block in: exactly cfg->hop_in samples. AINR_OK: work holds the block's one job; run the accelerator,
 * then ainr_postprocess(). AINR_ERR_SEQUENCE (no postprocess since the last call) or AINR_ERR_NULL: no
 * block is open. */
int ainr_preprocess(AinrCtx *ctx, const int16_t *pcm_in, AinrWork *work);

/* Closes the block: infer_status 0 commits, anything else bypasses with a near-identity pass-through (unit
 * ERB mask, alpha 0). Returns 0 with hop_in samples in pcm_out either way; AINR_ERR_SEQUENCE without a
 * matching preprocess (pcm_out untouched). */
int ainr_postprocess(AinrCtx *ctx, int infer_status, int16_t *pcm_out);

/* ---- PCM -> spectrum ("_spec") and spectrum -> spectrum ("_freq") families ----
 * Same config, AinrInfo and AinrWork as the PCM family; only the data crossing the block boundary differs.
 * Spectra are AUDIO scale (AcStft/AcIstft with scale 1.0), frame-major, oldest first, N frames of n_bins.
 * "_spec": PCM in, enhanced spectrum out (the caller does the ISTFT/OLA). "_freq": the caller supplies an
 * estimate spectrum E (features, mask, deep filter) and an apply spectrum P (what mask, deep filter and
 * attenuation limit act on); E == P (same pointers) is the single-spectrum case. A bypass delivers P-aligned
 * near-identity. */

typedef struct AinrSpecCtx AinrSpecCtx;

/* As ainr_get_mem_size(), for the PCM -> spectrum family. */
int ainr_get_mem_size_spec(const AinrConfig *cfg, StreamMemReq *req);

/* As ainr_init(), for the PCM -> spectrum family. */
int ainr_init_spec(const AinrConfig *cfg, void *pool, size_t pool_bytes, AinrSpecCtx **ctx, AinrInfo *info);

/* As ainr_reset(), for the PCM -> spectrum family. */
void ainr_reset_spec(AinrSpecCtx *ctx);

/* As ainr_set_atten_lim(), for the PCM -> spectrum family. */
int ainr_set_atten_lim_spec(AinrSpecCtx *ctx, float atten_lim_db);

/* As ainr_state_generation(), for the PCM -> spectrum family. */
unsigned int ainr_state_generation_spec(const AinrSpecCtx *ctx);

/* As ainr_preprocess(). */
int ainr_preprocess_spec(AinrSpecCtx *ctx, const int16_t *pcm_in, AinrWork *work);

/* As ainr_postprocess(), writing N frames of enhanced spectrum (out_re/out_im: N * n_bins floats each). */
int ainr_postprocess_spec(AinrSpecCtx *ctx, int infer_status, float *out_re, float *out_im);

typedef struct AinrFreqCtx AinrFreqCtx;

/* As ainr_get_mem_size(), for the spectrum -> spectrum family. */
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
                         const float *app_re, const float *app_im, AinrWork *work);

/* As ainr_postprocess_spec(). */
int ainr_postprocess_freq(AinrFreqCtx *ctx, int infer_status, float *out_re, float *out_im);

#ifdef __cplusplus
}
#endif

#endif /* AINR_H */
