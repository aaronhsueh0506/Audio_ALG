/* DeepFilterNet2 PCM stream: optional HPF, AcStft/AcIstft (AC_OLA_ACCUMULATE) and the dfn2_prepost per-frame functions
 * around int16 PCM, grid.frames_per_call N in [1, 64]. The stream sequences a call (N pre_frame, one job, validate_run,
 * N post_frame or post_frame_identity) and holds the application-spectrum staging of the call. N == 1: one hop per call, a
 * job on every hop from hop L (io.action == STREAM_WORK_SUBMIT; the first mask_lookahead hops after init or reset are
 * STREAM_WORK_BYPASS). N > 1: one packet of N hops per call and one job at any L, every packet a SUBMIT
 * (dfn2_stream_{pre,post}_process_packet, below); the per-hop entries refuse it.
 *
 *     ctx = dfn2_stream_init_from_erb_file(&cfg, erb_path, pool, pool_bytes);
 *     while (have a hop of int16 pcm_in) {
 *         DFN2HopIO io;
 *         if (dfn2_stream_pre_process(ctx, pcm_in, &io) != 1) break;   // -1 is a contract error
 *         int ai_rc = 0;
 *         if (io.action == STREAM_WORK_SUBMIT) {
 *             device_state_sync(io.state_generation);      // zero the accelerator state when the generation changed
 *             ai_rc = run_accelerator(&io.inputs, &io.outputs);
 *         }
 *         if (dfn2_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *     }
 *
 * device_state_sync is the caller's code on accelerator memory this stream never touches; skipping it leaves
 * stale or garbage state live. io.state_generation is sampled every hop.
 * Run status and state generation: STREAM_RUN_* in stream_run.h.
 * A BYPASS hop emits silence and the status passed to post_process is ignored.
 * The output trails the input by mask_lookahead + df_lookahead hops for every N; there is nothing to flush.
 *
 * A caller with its own analysis and synthesis uses dfn2_stream_pre_process_freq_dual() /
 * dfn2_stream_post_process_freq() instead; cfg.freq_only then keeps the HPF, STFT/ISTFT and PCM buffers out of the pool. */

#ifndef DFN2_STREAM_H
#define DFN2_STREAM_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "dfn2_prepost.h"   /* DFN2Grid, DFN2PrepostInputs/Outputs */
#include "stream_run.h"     /* StreamWorkAction, STREAM_RUN_OK, STREAM_RUN_NOT_STARTED, STREAM_RUN_FAILED */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN2_STREAM_DESCRIPTOR_VERSION 1u
#define DFN2_STREAM_LAYOUT_VERSION     6u   /* bump with build_flags_hash on any pool/carve/descriptor change */
#define DFN2_STREAM_ALIGNMENT          16u   /* the one alignment of every module (audiolib mem_align.h) */

/* Largest accepted |bin| of a MODEL-scale spectrum, as IEEE bits: 2^63, so the feature power re^2 + im^2 stays
 * <= 2^127 < FLT_MAX. A larger or non-finite bin would put Inf/NaN into the feature normaliser until reset. */
#define DFN2_STREAM_BIN_LIMIT_BITS 0x5F000000u

/* Returned by the _freq pre entries for a bin that is non-finite or beyond the limit above; nothing moved. Other contract errors are -1. */
#define DFN2_STREAM_ERR_RANGE (-2)

typedef struct DFN2StreamConfig {
    DFN2Grid grid;             /* dfn2_grid_make(); frames_per_call in [1, 64] */
    float    atten_lim_db;     /* |x| dB limit, sign ignored; 0 disables, as dfn2_prepost */
    int      hpf_enable;       /* optional input high-pass, applied before analysis only */
    float    hpf_cutoff_hz;    /* > 0, read only when hpf_enable */
    /* Non-zero: a spectrum-only context. Only the _freq entries run (the PCM entries return -1) and the pool holds no
     * FFT, STFT/ISTFT or PCM buffer; hpf_enable must be 0. 0: both entry families. */
    int      freq_only;
    /* Memory-init mode (dfn2_stream_init): both required, borrowed for the stream's life; erb_fwd bin-major
     * [n_bins][n_erb], erb_inv band-major [n_erb][n_bins] (transpose the on-disk table). Both NULL: file-init mode,
     * storage sized from the grid. */
    const float *erb_fwd;
    const float *erb_inv;
} DFN2StreamConfig;

/* Fixed 32-byte descriptor; a pool sized for another grid, mode (ERB source, freq_only) or build is refused by init. */
typedef struct DFN2StreamMemReq {
    uint32_t descriptor_version;  /* DFN2_STREAM_DESCRIPTOR_VERSION          */
    uint32_t layout_version;      /* DFN2_STREAM_LAYOUT_VERSION              */
    uint32_t reserved0;           /* must be 0                                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, mode, freq_only, variant and the folded DFN2Prepost hash */
    uint32_t alignment;           /* DFN2_STREAM_ALIGNMENT                    */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} DFN2StreamMemReq;

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
#define dfn2_stream_hop_size                dfn2_stream_hop_size_mask_only
#define dfn2_stream_pre_process             dfn2_stream_pre_process_mask_only
#define dfn2_stream_post_process            dfn2_stream_post_process_mask_only
#define dfn2_stream_pre_process_freq_dual   dfn2_stream_pre_process_freq_dual_mask_only
#define dfn2_stream_post_process_freq       dfn2_stream_post_process_freq_mask_only
#define dfn2_stream_pre_process_packet      dfn2_stream_pre_process_packet_mask_only
#define dfn2_stream_post_process_packet     dfn2_stream_post_process_packet_mask_only
#define dfn2_stream_pre_process_packet_freq_dual dfn2_stream_pre_process_packet_freq_dual_mask_only
#define dfn2_stream_post_process_packet_freq dfn2_stream_post_process_packet_freq_mask_only
#endif

/* Exact pool size for the mode erb_fwd/erb_inv imply (both NULL: file mode; both set: memory mode) and for freq_only.
 * -1 (req untouched) on NULL, an invalid grid, non-finite atten_lim_db, bad hpf_cutoff_hz, exactly one table, or
 * freq_only with hpf_enable. */
int dfn2_stream_get_mem_size(const DFN2StreamConfig *cfg, DFN2StreamMemReq *req);

/* Constructs in the pool (16-byte aligned, >= get_mem_size()): HPF (if enabled), STFT/ISTFT and DFN2Prepost
 * (freq_only: DFN2Prepost and the spectrum staging), no heap; starts reset. NULL on a bad config, a misaligned or small pool, erb_fwd/erb_inv not both set, or tables that fail
 * dfn2_erb_tables_valid() (finite erb_fwd >= 0; erb_inv in [0, 1] summing to exactly 1.0f in every bin, band-ascending in float32). */
DFN2Stream *dfn2_stream_init(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes);

/* As dfn2_stream_init, plus the stale-pool gate: a pool carved for another grid, mode, variant or layout
 * is refused. `expected` NULL: as dfn2_stream_init. */
DFN2Stream *dfn2_stream_init_ex(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes,
                                const DFN2StreamMemReq *expected);

/* As dfn2_stream_init, loading the ERB tables from `erb_path` into pool storage (cfg->erb_fwd/erb_inv NULL): int32 n_bins, n_erb,
 * then n_bins*n_erb float32 forward and inverse, both bin-major, nothing after. NULL (pool not live) on another grid's header, a short
 * or longer file, a non-finite forward value or a non-partition inverse (as dfn2_stream_init). Out under -DDFN2_STREAM_NO_STDIO. */
DFN2Stream *dfn2_stream_init_from_erb_file(const DFN2StreamConfig *cfg, const char *erb_path,
                                           void *pool, size_t pool_bytes);

/* As dfn2_stream_init_from_erb_file, plus the same stale-pool gate as dfn2_stream_init_ex. */
DFN2Stream *dfn2_stream_init_from_erb_file_ex(const DFN2StreamConfig *cfg, const char *erb_path,
                                              void *pool, size_t pool_bytes,
                                              const DFN2StreamMemReq *expected);

/* Back to the start: HPF, STFT/ISTFT (when the context has them) and DFN2Prepost reset (state zeroed, generation
 * advanced); ERB tables kept. */
void dfn2_stream_reset(DFN2Stream *s);

/* Generation of the recurrent state (dfn2_prepost_state_generation()); 0 for NULL. */
unsigned int dfn2_stream_state_generation(const DFN2Stream *s);

/* Attenuation limit of |atten_lim_db| dB (sign ignored; 0 disables), between hops only; 0, or -1 on NULL, a
 * non-finite value or a hop awaiting its post_process (a hop without a job included). Kept by reset. */
int dfn2_stream_set_atten_lim(DFN2Stream *s, float atten_lim_db);

int dfn2_stream_hop_size(const DFN2Stream *s);           /* samples per PCM hop; -1 on NULL */

/* The tensor views of one open hop and the generation the accelerator's state must match; valid until
 * dfn2_stream_post_process(). */
typedef struct DFN2HopIO {
    StreamWorkAction   action;      /* SUBMIT: run the accelerator. BYPASS (N == 1, the first mask_lookahead hops): no job, inputs/outputs not set */
    DFN2PrepostInputs  inputs;      /* valid only on SUBMIT */
    DFN2PrepostOutputs outputs;     /* valid only on SUBMIT */
    /* Sampled every hop; reload the accelerator's state with quantized zero if it differs. */
    unsigned int       state_generation;
} DFN2HopIO;

/* One PCM hop in (dfn2_stream_hop_size(s) samples), frames_per_call == 1: int16 -> float, HPF, STFT, pre_frame, views
 * on a SUBMIT hop. 1 with *io filled (inputs/outputs only on SUBMIT), or -1 on NULL args, no preceding
 * post_process, a freq_only context or frames_per_call > 1. */
int dfn2_stream_pre_process(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io);

/* Closes the hop, once per pre_process: on a SUBMIT hop ai_rc 0 commits, STREAM_RUN_NOT_STARTED bypasses, any other
 * value bypasses and advances the generation (otherwise ai_rc is ignored); one hop is then drained to pcm_out through
 * the ISTFT. Returns the spectral-validity flag (1/0) of the hop's post_frame, -1 on error (a freq_only
 * context included); it is not an alignment: count L + D hops. */
int dfn2_stream_post_process(DFN2Stream *s, int ai_rc, int16_t *pcm_out);

/* As dfn2_stream_pre_process() on spectra, for either kind of context: est and app are n_bins floats each, MODEL scale; est drives features and
 * heads, app is what mask, deep filter and attenuation limit act on (est == app: one spectrum). State untouched on a refusal, which
 * is DFN2_STREAM_ERR_RANGE for a bin that is non-finite or beyond DFN2_STREAM_BIN_LIMIT_BITS, else -1 (NULL args, no preceding post_process). */
int dfn2_stream_pre_process_freq_dual(DFN2Stream *s, const float *est_re, const float *est_im,
                                      const float *app_re, const float *app_im, DFN2HopIO *io);

/* As dfn2_stream_post_process(), delivering the enhanced spectrum (re/im n_bins floats each, MODEL scale)
 * and *valid (may be NULL) instead of an ISTFT. 0, or -1 on NULL args or no pending pre_process. */
int dfn2_stream_post_process_freq(DFN2Stream *s, int ai_rc, float *re, float *im, int *valid);

/* Packet entries, N > 1: N hops of PCM (N * hop_size samples) in and out per call, one job. pre returns 1 with *io filled,
 * -1 on NULL args, a pending packet, a freq_only context or N == 1; post closes the packet as per hop and returns the number
 * of hops with a spectrum (0..N), -1 on error. Output hop t is input hop t - L - D, as for N == 1. */
int dfn2_stream_pre_process_packet(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io);
int dfn2_stream_post_process_packet(DFN2Stream *s, int ai_rc, int16_t *pcm_out);

/* The packet entries on spectra (N * n_bins floats per plane, frame-major, MODEL scale): state untouched on a refusal, which is
 * DFN2_STREAM_ERR_RANGE for a bad bin in any frame, else -1 (NULL args, a pending packet, N == 1). post returns 0 with *valid_frames (may be NULL) set, -1 on error. */
int dfn2_stream_pre_process_packet_freq_dual(DFN2Stream *s, const float *est_re, const float *est_im,
                                             const float *app_re, const float *app_im, DFN2HopIO *io);
int dfn2_stream_post_process_packet_freq(DFN2Stream *s, int ai_rc, float *re, float *im, int *valid_frames);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_STREAM_H */
