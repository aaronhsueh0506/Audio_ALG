/* DeepFilterNet2 PCM stream: optional HPF, AcStft/AcIstft (AC_OLA_ACCUMULATE) and DFN2Prepost around int16
 * PCM, one hop in and one hop out per call, grid.frames_per_call N in [1, 64]. N == 1: the accelerator runs
 * every hop. N > 1: once every N hops (io.job_ready); pre/post are still called every hop.
 *
 *     ctx = dfn2_stream_init_from_erb_file(&cfg, erb_path, pool, pool_bytes);
 *     while (have a hop of int16 pcm_in) {
 *         DFN2HopIO io;
 *         if (dfn2_stream_pre_process(ctx, pcm_in, &io) != 1) break;   // -1 is a contract error
 *         int ai_rc = 0;
 *         if (io.job_ready) {
 *             device_state_sync(io.state_generation);      // reload accelerator state on change
 *             ai_rc = run_accelerator(&io.inputs, &io.outputs);
 *             if (ai_rc != 0) device_state_refill_zero();   // partial write: reload now, same hop
 *         }
 *         if (dfn2_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *     }
 *
 * device_state_sync/refill_zero are the caller's code on accelerator memory this stream never touches;
 * skipping either leaves stale or garbage state live. io.state_generation is sampled every hop.
 * Warm-up (N == 1: mask_lookahead hops after init/reset; N > 1: none) runs the same loop and emits silence.
 * Tail: push mask_lookahead + df_lookahead + N zero hops after the last real one. */

#ifndef DFN2_STREAM_H
#define DFN2_STREAM_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "dfn2_prepost.h"   /* DFN2Grid, DFN2PrepostInputs/Outputs */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN2_STREAM_DESCRIPTOR_VERSION 1u
#define DFN2_STREAM_LAYOUT_VERSION     3u   /* bump with build_flags_hash on any pool/carve/descriptor change */
#define DFN2_STREAM_ALIGNMENT          16u   /* the one alignment of every module (audiolib mem_align.h) */

/* Largest accepted |bin| of a MODEL-scale spectrum, as IEEE bits: 2^63, so the feature power re^2 + im^2 stays
 * <= 2^127 < FLT_MAX. A larger or non-finite bin would put Inf/NaN into the feature normaliser until reset. */
#define DFN2_STREAM_BIN_LIMIT_BITS 0x5F000000u

/* 1 when every x[k] * scale (float product, scale 1.0f for MODEL scale) is finite and within the limit above. */
static inline int dfn2_stream_bins_in_range(const float *x, size_t n, float scale)
{
    size_t k;
    for (k = 0; k < n; ++k) {
        float m = x[k] * scale;
        uint32_t bits;
        memcpy(&bits, &m, sizeof bits);
        if ((bits & 0x7FFFFFFFu) > DFN2_STREAM_BIN_LIMIT_BITS) return 0;
    }
    return 1;
}

typedef struct DFN2StreamConfig {
    DFN2Grid grid;             /* dfn2_grid_make(); frames_per_call in [1, 64] */
    float    atten_lim_db;     /* 0 disables the attenuation limit, as dfn2_prepost */
    int      hpf_enable;       /* optional input high-pass, applied before analysis only */
    float    hpf_cutoff_hz;    /* > 0, read only when hpf_enable */
    /* Memory-init mode (dfn2_stream_init): both required, borrowed for the stream's life; erb_fwd bin-major
     * [n_bins][n_erb], erb_inv band-major [n_erb][n_bins] (transpose the on-disk table). Both NULL: file-init mode,
     * storage sized from the grid. */
    const float *erb_fwd;
    const float *erb_inv;
} DFN2StreamConfig;

/* Fixed 32-byte descriptor; a pool sized for another grid, mode or build is refused by init. */
typedef struct DFN2StreamMemReq {
    uint32_t descriptor_version;  /* DFN2_STREAM_DESCRIPTOR_VERSION          */
    uint32_t layout_version;      /* DFN2_STREAM_LAYOUT_VERSION              */
    uint32_t reserved0;           /* must be 0                                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, mode, variant and the folded DFN2Prepost hash */
    uint32_t alignment;           /* DFN2_STREAM_ALIGNMENT                    */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} DFN2StreamMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DFN2StreamMemReq) == 32,
               "DFN2StreamMemReq must stay 32 bytes");
#endif
#endif

typedef struct DFN2Stream DFN2Stream;

/* DFN2HopIO's size depends on DFN2_MASK_ONLY, so every entry point is suffixed _mask_only under it: a
 * mismatched pairing fails to link instead of overrunning DFN2HopIO. */
#ifdef DFN2_MASK_ONLY
#define dfn2_stream_get_mem_size            dfn2_stream_get_mem_size_mask_only
#define dfn2_stream_init                    dfn2_stream_init_mask_only
#define dfn2_stream_init_ex                 dfn2_stream_init_ex_mask_only
#define dfn2_stream_init_from_erb_file      dfn2_stream_init_from_erb_file_mask_only
#define dfn2_stream_init_from_erb_file_ex   dfn2_stream_init_from_erb_file_ex_mask_only
#define dfn2_stream_reset                   dfn2_stream_reset_mask_only
#define dfn2_stream_state_generation        dfn2_stream_state_generation_mask_only
#define dfn2_stream_set_atten_lim           dfn2_stream_set_atten_lim_mask_only
#define dfn2_stream_grid                    dfn2_stream_grid_mask_only
#define dfn2_stream_hop_size                dfn2_stream_hop_size_mask_only
#define dfn2_stream_layout_version          dfn2_stream_layout_version_mask_only
#define dfn2_stream_pre_process             dfn2_stream_pre_process_mask_only
#define dfn2_stream_post_process            dfn2_stream_post_process_mask_only
#define dfn2_stream_pre_process_freq_dual   dfn2_stream_pre_process_freq_dual_mask_only
#define dfn2_stream_post_process_freq       dfn2_stream_post_process_freq_mask_only
#endif

/* Exact pool size for the mode erb_fwd/erb_inv imply (both NULL: file mode; both set: memory mode). -1
 * (req untouched) on NULL, an invalid grid, non-finite atten_lim_db, bad hpf_cutoff_hz or exactly one table. */
int dfn2_stream_get_mem_size(const DFN2StreamConfig *cfg, DFN2StreamMemReq *req);

/* Constructs in the pool (16-byte aligned, >= get_mem_size()): HPF (if enabled), STFT/ISTFT and DFN2Prepost,
 * no heap; starts reset. NULL on a bad config, a misaligned or small pool, or erb_fwd/erb_inv not both set. */
DFN2Stream *dfn2_stream_init(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes);

/* As dfn2_stream_init, plus the stale-pool gate: a pool carved for another grid, mode, variant or layout
 * is refused. `expected` NULL: as dfn2_stream_init. */
DFN2Stream *dfn2_stream_init_ex(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes,
                                const DFN2StreamMemReq *expected);

/* As dfn2_stream_init, loading the ERB tables from `erb_path` into pool storage (cfg->erb_fwd/erb_inv NULL): int32 n_bins,
 * n_erb, then n_bins*n_erb float32 forward and inverse, both bin-major. NULL (pool not live) on a header not matching
 * cfg->grid or a short file. The only file I/O; out under -DDFN2_STREAM_NO_STDIO. */
DFN2Stream *dfn2_stream_init_from_erb_file(const DFN2StreamConfig *cfg, const char *erb_path,
                                           void *pool, size_t pool_bytes);

/* As dfn2_stream_init_from_erb_file, plus the same stale-pool gate as dfn2_stream_init_ex. */
DFN2Stream *dfn2_stream_init_from_erb_file_ex(const DFN2StreamConfig *cfg, const char *erb_path,
                                              void *pool, size_t pool_bytes,
                                              const DFN2StreamMemReq *expected);

/* Back to the start: HPF, STFT/ISTFT and DFN2Prepost reset (state zeroed, generation advanced, warm-up due
 * again); ERB tables kept. */
void dfn2_stream_reset(DFN2Stream *s);

/* Generation of the recurrent state (dfn2_prepost_state_generation()); 0 for NULL. */
unsigned int dfn2_stream_state_generation(const DFN2Stream *s);

/* Attenuation limit in dB (0 disables), between hops only; 0, or -1 on NULL, a non-finite value or an open
 * frame. Kept by reset. */
int dfn2_stream_set_atten_lim(DFN2Stream *s, float atten_lim_db);

const DFN2Grid *dfn2_stream_grid(const DFN2Stream *s);   /* NULL for NULL */
int dfn2_stream_hop_size(const DFN2Stream *s);           /* samples per PCM hop; -1 on NULL */
/* dfn2_prepost_layout_version(): the graph contract of this instance. -1 on NULL. */
int dfn2_stream_layout_version(const DFN2Stream *s);

/* The tensor views of one open hop and the generation the accelerator's state must match; valid until
 * dfn2_stream_post_process(). */
typedef struct DFN2HopIO {
    int                job_ready;   /* 1: run the accelerator. 0 (N > 1 only): no job, inputs/outputs stale */
    DFN2PrepostInputs  inputs;      /* valid only when job_ready */
    DFN2PrepostOutputs outputs;     /* valid only when job_ready */
    /* Sampled every hop; reload the accelerator's state with quantized zero if it differs. */
    unsigned int       state_generation;
} DFN2HopIO;

/* One PCM hop in (dfn2_stream_hop_size(s) samples): int16 -> float, HPF, STFT, pre_process_freq, frame_open
 * when io->job_ready. 1 with *io filled (warm-up included), or -1 on NULL args or no preceding post_process. */
int dfn2_stream_pre_process(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io);

/* Closes the hop, once per pre_process: with io->job_ready ai_rc 0 commits, else bypasses (otherwise ignored); one hop
 * is then drained to pcm_out through the ISTFT. Returns the spectral-validity flag (1/0) of
 * dfn2_prepost_post_process_freq(), -1 on error; it is not an alignment: count L + D + N hops. */
int dfn2_stream_post_process(DFN2Stream *s, int ai_rc, int16_t *pcm_out);

/* As dfn2_stream_pre_process() on spectra: est and app are n_bins floats each, MODEL scale; est drives features and
 * heads, app is what mask, deep filter and attenuation limit act on (est == app: one spectrum). -1, state untouched, on
 * NULL args, no preceding post_process, or a bin that is non-finite or beyond DFN2_STREAM_BIN_LIMIT_BITS. */
int dfn2_stream_pre_process_freq_dual(DFN2Stream *s, const float *est_re, const float *est_im,
                                      const float *app_re, const float *app_im, DFN2HopIO *io);

/* As dfn2_stream_post_process(), delivering the enhanced spectrum (re/im n_bins floats each, MODEL scale)
 * and *valid (may be NULL) instead of an ISTFT. 0, or -1 on NULL args or no pending pre_process. */
int dfn2_stream_post_process_freq(DFN2Stream *s, int ai_rc, float *re, float *im, int *valid);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_STREAM_H */
