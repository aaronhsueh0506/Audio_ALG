/* DeepFilterNet2 PCM stream: wraps the optional HPF, audiolib AcStft/AcIstft
 * (AC_OLA_ACCUMULATE) and lib/ainr/dfn2's DFN2Prepost around int16 PCM, one hop per call.
 * frames_per_call must be 1 (refused at init); N > 1 is core-level only
 * (lib/ainr/dfn2/dfn2_process.h / dfn2_model_io.h). Frequency-domain callers should use
 * lib/ainr/dfn2/dfn2_prepost.h directly.
 *
 *     ctx = dfn2_stream_init_from_erb_file(&cfg, erb_path, pool, pool_bytes);
 *     while (have a hop of int16 pcm_in) {
 *         DFN2HopIO io;
 *         if (dfn2_stream_pre_process(ctx, pcm_in, &io) != 1) break;   // -1 is a contract error
 *         device_state_sync(io.state_generation);      // reload accelerator state on change
 *         int ai_rc = run_accelerator(&io.inputs, &io.outputs);
 *         if (ai_rc != 0) device_state_refill_zero();   // partial write -> reload now, same hop
 *         if (dfn2_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *     }
 *
 * device_state_sync/refill_zero are the adapter's own code operating accelerator memory this
 * stream never touches (it only tracks/reports io.state_generation); skipping sync leaves a
 * stale state copy live across a generation change (every sample of the next hop wrong),
 * skipping refill leaves a failed run's garbage write live until the next change (both
 * confirmed; the pattern matches alg_ai3_dfn.c's per-hop loop).
 *
 * Warm-up (mask_lookahead hops after init/reset) runs this same loop, emitting silence. Tail
 * flush: push mask_lookahead + df_lookahead + 1 zero hops after the last real one, to drain the
 * core's L+D latency plus this stream's own one hop of STFT/ISTFT framing. */

#ifndef DFN2_STREAM_H
#define DFN2_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_prepost.h"   /* DFN2Grid, DFN2PrepostInputs/Outputs */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN2_STREAM_DESCRIPTOR_VERSION 1u
#define DFN2_STREAM_LAYOUT_VERSION     1u   /* bump with build_flags_hash on any pool/carve/descriptor change */
#define DFN2_STREAM_ALIGNMENT          16u   /* the one alignment of every module (audiolib mem_align.h) */

typedef struct DFN2StreamConfig {
    DFN2Grid grid;             /* dfn2_grid_make(); frames_per_call must be 1 */
    float    atten_lim_db;     /* 0 disables the attenuation limit, as dfn2_prepost */
    int      hpf_enable;       /* optional input high-pass, applied before analysis only */
    float    hpf_cutoff_hz;    /* > 0, read only when hpf_enable */
    /* Memory-init mode only (dfn2_stream_init): required, borrowed, alive for the stream's life.
     * erb_fwd is bin-major [n_bins][n_erb] (matches lib/ainr/dfn2/dfn2_process.h's DFN2State);
     * erb_inv is band-major [n_erb][n_bins] -- NOT the same layout as erb_fwd, and not the raw
     * on-disk (bin-major) format load_erb_matrix() reads; a caller must transpose first (see
     * tests/check_erb_file.c). Leave both NULL for file-init mode
     * (dfn2_stream_init_from_erb_file): get_mem_size() then reserves table storage from the grid
     * instead, and both fields are ignored. */
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

/* DFN2HopIO's size depends on DFN2_MASK_ONLY (deep-filter output members omitted under it): an
 * archive built with one flag linked against a caller built with the other would otherwise
 * still link under the same symbol names, then overrun/misread DFN2HopIO on the first hop. Every
 * entry point below is therefore variant-suffixed by this header, keyed off the caller's own
 * -DDFN2_MASK_ONLY -- a mismatched pairing now fails to link instead of running. Same mechanism
 * as dfn2_process.h's DFN2_VARIANT_SYMBOL, extended here to this SDK's own public API (does not
 * reach dfn2_prepost.h's own entry points, which still rely on that weaker link-check-only
 * protection). */
#ifdef DFN2_MASK_ONLY
#define dfn2_stream_get_mem_size            dfn2_stream_get_mem_size_mask_only
#define dfn2_stream_init                    dfn2_stream_init_mask_only
#define dfn2_stream_init_ex                 dfn2_stream_init_ex_mask_only
#define dfn2_stream_init_from_erb_file      dfn2_stream_init_from_erb_file_mask_only
#define dfn2_stream_init_from_erb_file_ex   dfn2_stream_init_from_erb_file_ex_mask_only
#define dfn2_stream_reset                   dfn2_stream_reset_mask_only
#define dfn2_stream_state_generation        dfn2_stream_state_generation_mask_only
#define dfn2_stream_grid                    dfn2_stream_grid_mask_only
#define dfn2_stream_hop_size                dfn2_stream_hop_size_mask_only
#define dfn2_stream_layout_version          dfn2_stream_layout_version_mask_only
#define dfn2_stream_pre_process             dfn2_stream_pre_process_mask_only
#define dfn2_stream_post_process            dfn2_stream_post_process_mask_only
#endif

/* Exact pool size for the mode cfg->erb_fwd/erb_inv implies (both NULL: file mode, sized from
 * cfg->grid; both non-NULL: memory mode, no table storage). -1 (req untouched) on NULL, an
 * invalid grid, frames_per_call != 1, non-finite atten_lim_db, an out-of-range hpf_cutoff_hz
 * (checked against hpf_init()'s own bounds so init can't fail later on it), or exactly one of
 * erb_fwd/erb_inv set. */
int dfn2_stream_get_mem_size(const DFN2StreamConfig *cfg, DFN2StreamMemReq *req);

/* Construct in caller memory (16-byte aligned, >= get_mem_size()'s bytes): carves the HPF (if
 * enabled, hpf_init()'d -- no heap after init), the STFT/ISTFT pair and DFN2Prepost from `pool`;
 * starts reset. NULL on a bad config, misaligned/undersized pool, or erb_fwd/erb_inv not both set. */
DFN2Stream *dfn2_stream_init(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes);

/* As dfn2_stream_init, plus the stale-pool gate: a pool carved for another grid, mode, variant or
 * layout is refused. `expected` NULL behaves as dfn2_stream_init. */
DFN2Stream *dfn2_stream_init_ex(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes,
                                const DFN2StreamMemReq *expected);

/* As dfn2_stream_init, but loads ERB tables from `erb_path` into pool-owned storage instead of
 * borrowing cfg->erb_fwd/erb_inv (both must be NULL): int32 n_bins, n_erb, then n_bins*n_erb
 * float32 forward + inverse, both bin-major on disk (inverse transposed to band-major on load) --
 * the same format load_erb_matrix() in the board samples reads. Rejects a header mismatching
 * cfg->grid or a short file; on any failure the file is closed and NULL returned (a partial
 * table write left in the pool is harmless -- treat NULL as total failure, never reuse the pool
 * as live). Only this entry point does file I/O; omit dfn2_stream_erb_file.o for NO_STDIO. */
DFN2Stream *dfn2_stream_init_from_erb_file(const DFN2StreamConfig *cfg, const char *erb_path,
                                           void *pool, size_t pool_bytes);

/* As dfn2_stream_init_from_erb_file, plus the same stale-pool gate as dfn2_stream_init_ex. */
DFN2Stream *dfn2_stream_init_from_erb_file_ex(const DFN2StreamConfig *cfg, const char *erb_path,
                                              void *pool, size_t pool_bytes,
                                              const DFN2StreamMemReq *expected);

/* Back to the start: HPF, STFT/ISTFT and DFN2Prepost all reset (recurrent state zeroed, generation
 * advanced, warm-up frames due again); ERB tables (borrowed or pool-owned) kept. */
void dfn2_stream_reset(DFN2Stream *s);

/* Generation of the recurrent state (dfn2_prepost_state_generation()); 0 for NULL. */
unsigned int dfn2_stream_state_generation(const DFN2Stream *s);

const DFN2Grid *dfn2_stream_grid(const DFN2Stream *s);   /* NULL for NULL */
int dfn2_stream_hop_size(const DFN2Stream *s);           /* samples per PCM hop; -1 on NULL */
/* dfn2_prepost_layout_version(): the graph contract this instance is bound to (a board sample's
 * own `_Static_assert(DFN2_MODEL_IO_LAYOUT_VERSION == ...)` needs dfn2_model_io.h's constant
 * directly; this is the run-time counterpart). -1 on NULL. */
int dfn2_stream_layout_version(const DFN2Stream *s);

/* The model's tensor views of one open hop (DFN2PrepostInputs/Outputs; dfn2_prepost.h), plus the
 * generation the accelerator's own state copy must match before this call. Valid until
 * dfn2_stream_post_process(). */
typedef struct DFN2HopIO {
    DFN2PrepostInputs  inputs;
    DFN2PrepostOutputs outputs;
    /* Sampled right after frame_open(), before the accelerator runs -- reflects the previous
     * hop's warm-up/refusal. Reload the accelerator's own state copy with quantized zero first
     * if it doesn't match. */
    unsigned int       state_generation;
} DFN2HopIO;

/* One PCM hop in: int16 -> float, HPF (if enabled), analysis STFT, pre_process_freq + frame_open.
 * Returns 1 with *io filled (every hop, warm-up included -- run the accelerator, then call post),
 * or -1 on a contract error (NULL args, or called again before the matching post_process -- the
 * stream tracks open/closed itself, independent of frame_open()'s own refusal, since the
 * analysis STFT is pushed before frame_open() is even attempted). pcm_in holds exactly
 * dfn2_stream_hop_size(s) samples. */
int dfn2_stream_pre_process(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io);

/* Closes the hop opened by pre_process: ai_rc == 0 commits (frame_finish), anything else bypasses
 * (frame_bypass); either way post_process_freq() then runs once. On its ordinary result (0),
 * ac_istft_push_planar() always writes a full hop (AC_OLA_ACCUMULATE) and pcm_out is saturated
 * from it. On a post_process_freq() contract error (defensive only, unreached in correct use)
 * pcm_out is zeroed and synthesis is skipped, matching the board samples' own fallback.
 *
 * Return value (1/0/-1) is dfn2_prepost_post_process_freq()'s own spectral-validity flag, NOT
 * "pcm_out is this hop delayed by L+D" -- this stream's STFT/ISTFT framing adds one more hop
 * (confirmed: at 48kHz/1024/full/L=D=1, post_process already returns 1 at t=2, but pcm_out is
 * still warm-up filler; the hop-0-aligned real PCM appears only at t=3). A caller needing
 * sample-accurate alignment must count hops itself (L + D + 1), not branch on this return value.
 * Call once per successful pre_process, never more than once per hop. */
int dfn2_stream_post_process(DFN2Stream *s, int ai_rc, int16_t *pcm_out);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_STREAM_H */
