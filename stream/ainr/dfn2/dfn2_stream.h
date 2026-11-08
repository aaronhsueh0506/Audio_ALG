/* DeepFilterNet2 PCM stream: the one header a board sample includes. Wraps the HPF (optional), the
 * analysis/synthesis STFT (audiolib AcStft/AcIstft, AC_OLA_ACCUMULATE) and lib/ainr/dfn2's
 * DFN2Prepost around int16 PCM, one hop per call. Frequency-domain callers with their own framing
 * should use lib/ainr/dfn2/dfn2_prepost.h directly instead. N == 1 (frames_per_call) only; a grid
 * with frames_per_call != 1 is refused at init. pipelines/dfn/dfn_res_stage.h is NOT the N > 1
 * path either -- it refuses frames_per_call != 1 for the same reason (one frame per call). N > 1
 * is only supported at the core level directly: lib/ainr/dfn2/dfn2_process.h / dfn2_model_io.h.
 *
 *     ctx = dfn2_stream_init_from_erb_file(&cfg, erb_path, pool, pool_bytes);   // or dfn2_stream_init()
 *     while (have a hop of int16 pcm_in) {
 *         DFN2HopIO io;
 *         if (dfn2_stream_pre_process(ctx, pcm_in, &io) != 1) break;           // -1 is a contract error
 *         device_state_sync(io.state_generation);    // reload the accelerator's OWN state copy
 *                                                     // with quantized zero if this changed since
 *                                                     // the last hop -- see below, this is not optional
 *         int ai_rc = run_accelerator(&io.inputs, &io.outputs);                // 0 = every output written
 *         if (ai_rc != 0) device_state_refill_zero(); // the run may have written PART of the state
 *                                                     // before failing; restart it from zero now,
 *                                                     // do not wait for the next hop's sync above
 *         if (dfn2_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *         // pcm_out is always a full hop; write/play it unconditionally.
 *     }
 *
 * The two device_state_* calls above are not this stream's own functions -- they are the adapter's
 * own code, operating the accelerator's state buffers directly (this stream only tracks and reports
 * io.state_generation; it has no access to accelerator memory). Skipping the first leaves the
 * accelerator holding a stale state copy across a reset/warm-up/refused-commit generation change
 * (confirmed: on a state-dependent accelerator, the first hop after a generation change this way
 * produces a PCM hop that is completely wrong, every sample differing); skipping the second leaves
 * it holding a failed run's partial/garbage write at the next sync, since that sync only reloads on
 * a generation CHANGE, and a plain bypass does not change the generation (confirmed: the next hop
 * then reads the stale write instead of the quantized zero a correct caller would see). The real
 * board sample does exactly this, in alg_ai3_dfn.c's per-hop loop: a state_generation_sync() call
 * right before the NPU run, and a state_refill_zero() call immediately after a failed run, inside
 * the same hop, before falling through to frame_bypass -- not a new pattern, a pattern this
 * stream's own minimal example above had been omitting.
 *
 * Warm-up (mask_lookahead hops after init/reset): pre_process still returns 1 and opens a job, the
 * accelerator still runs; the emitted hop is silence (the library discards the heads and advances
 * io.state_generation) -- same loop, no special case. Tail flush: after the last real PCM hop, push
 * mask_lookahead + df_lookahead + 1 all-zero PCM hops through the same loop to drain the last real
 * source frame (the stream's STFT/ISTFT framing adds one hop on top of the core's own L + D
 * latency); the first of those zero hops' STFT output still reflects the previous hop's window
 * history, as audiolib's analysis window overlaps by construction. */

#ifndef DFN2_STREAM_H
#define DFN2_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_prepost.h"   /* DFN2Grid, DFN2PrepostInputs/Outputs */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN2_STREAM_DESCRIPTOR_VERSION 1u
/* In build_flags_hash; bump on any change of pool size, carve order, or descriptor content,
 * the DFN2Prepost pool and hash included. */
#define DFN2_STREAM_LAYOUT_VERSION     1u
#define DFN2_STREAM_ALIGNMENT          16u   /* the one alignment of every module (audiolib mem_align.h) */

typedef struct DFN2StreamConfig {
    DFN2Grid grid;             /* dfn2_grid_make(); frames_per_call must be 1 */
    float    atten_lim_db;     /* 0 disables the attenuation limit, as dfn2_prepost */
    int      hpf_enable;       /* optional input high-pass, applied before analysis only */
    float    hpf_cutoff_hz;    /* > 0, read only when hpf_enable */
    /* Memory-init mode only (dfn2_stream_init): required, borrowed, alive for the stream's life.
     * NOT the same layout for both: erb_fwd is bin-major [n_bins][n_erb] (matching
     * lib/ainr/dfn2/dfn2_process.h's own DFN2State.erb_fwd doc), erb_inv is band-major
     * [n_erb][n_bins] -- the file-init entry point reads both from erb_path in on-disk,
     * bin-major order and transposes the inverse table to band-major on load, so a caller
     * feeding its own in-memory tables here must already have erb_inv band-major, not pass the
     * raw bin-major file layout straight through. Confirmed against tests/check_erb_file.c, which
     * builds and compares both an on-disk file (board-sample format) and the equivalent
     * already-transposed in-memory call. Leave both NULL for file-init mode
     * (dfn2_stream_init_from_erb_file): get_mem_size() then reserves the stream's own table
     * storage from the grid instead, and both fields are ignored by every entry point. */
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

/* DFN2HopIO's size depends on DFN2_MASK_ONLY (DFN2PrepostOutputs' deep-filter members are omitted
 * under it): an archive built with one flag linked against a caller compiled with the other would
 * otherwise still link -- same public names either way -- and then overrun/misread DFN2HopIO on
 * the first hop. Every public entry point below is therefore variant-suffixed by this header, so
 * the *caller's own* -DDFN2_MASK_ONLY choice picks which symbol its calls resolve to; a mismatched
 * pairing fails to link instead of running. Same mechanism as dfn2_process.h's
 * DFN2_VARIANT_SYMBOL/DFN2_REQUIRE_VARIANT_LINK, extended here to this SDK's own public API (this
 * does not reach lib/ainr/dfn2/dfn2_prepost.h's own entry points, which keep relying on that
 * weaker, link-check-only protection -- out of scope for this header). */
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

/* Exact pool size for the mode cfg->erb_fwd/erb_inv implies (both NULL: file mode, reserving the
 * stream's own ERB table storage sized from cfg->grid; both non-NULL: memory mode, no table
 * storage). -1 (req untouched) on NULL, an invalid grid, frames_per_call != 1, a non-finite
 * atten_lim_db, an hpf_enable'd hpf_cutoff_hz outside hpf_init()'s own accepted range (finite,
 * > 0, < 0.45 * grid.sample_rate -- checked here so a later dfn2_stream_init() can't fail on a
 * cutoff this call already approved), or exactly one of erb_fwd/erb_inv set. */
int dfn2_stream_get_mem_size(const DFN2StreamConfig *cfg, DFN2StreamMemReq *req);

/* Construct in caller memory (16-byte aligned, >= get_mem_size()'s bytes): the HPF (if enabled,
 * carved and hpf_init()'d into the pool -- never hpf_create(), there is no heap after init), the
 * analysis/synthesis STFT pair and the DFN2Prepost class, all carved from `pool`; starts reset.
 * NULL on a bad config, a misaligned/undersized pool, or cfg->erb_fwd/erb_inv not both set. */
DFN2Stream *dfn2_stream_init(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes);

/* As dfn2_stream_init, plus the stale-pool gate: a pool carved for another grid, mode, variant or
 * layout is refused. `expected` NULL behaves as dfn2_stream_init. */
DFN2Stream *dfn2_stream_init_ex(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes,
                                const DFN2StreamMemReq *expected);

/* As dfn2_stream_init, but loads the ERB tables from `erb_path` into pool-owned storage instead of
 * borrowing cfg->erb_fwd/erb_inv (which must both be NULL): int32 n_bins, int32 n_erb, then
 * n_bins*n_erb float32 forward (bin-major) and n_bins*n_erb float32 inverse (bin-major, transposed
 * to band-major on load) -- the same format and byte order load_erb_matrix() in the board samples
 * reads. Rejects a header that doesn't match cfg->grid and a short/truncated file; on any failure
 * the file is closed and NULL returned -- no DFN2Stream is ever constructed, though the pool's
 * ERB table region may hold a partial read (harmless: treat a NULL return as total failure and
 * either discard the pool or retry into it, never reuse it as if it were live). Only this entry
 * point performs file I/O; link without dfn2_stream_erb_file.o for a NO_STDIO build that supplies
 * its own matrices via dfn2_stream_init(). */
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
/* dfn2_prepost_layout_version(): the graph contract this instance is bound to. A board sample's
 * own compile-time `_Static_assert(DFN2_MODEL_IO_LAYOUT_VERSION == ...)` (its NPU tensor table and
 * state sizes are graph artifacts the sample owns, not part of this stream's contract) still needs
 * dfn2_model_io.h directly for that constant; this accessor is the run-time counterpart. -1 on NULL. */
int dfn2_stream_layout_version(const DFN2Stream *s);

/* The model's tensor views of one open hop (DFN2PrepostInputs/Outputs; dfn2_prepost.h), plus the
 * generation the accelerator's own state copy must match before this call (reload quantized zero
 * first if it doesn't). Valid until dfn2_stream_post_process(). */
typedef struct DFN2HopIO {
    DFN2PrepostInputs  inputs;
    DFN2PrepostOutputs outputs;
    /* Sampled right after frame_open(), before the accelerator runs: it already reflects the
     * previous hop's warm-up or refusal (dfn2_prepost_frame_finish() advances the generation on
     * its way out, not on the way in). An accelerator-side state copy must match this value
     * before it runs this hop -- reload quantized zero first if it doesn't. */
    unsigned int       state_generation;
} DFN2HopIO;

/* One PCM hop in: int16 -> float, HPF (if enabled), analysis STFT, dfn2_prepost pre_process_freq +
 * frame_open. Returns 1 with *io filled (every hop, warm-up included: run the accelerator, then
 * call post), or -1 on a contract error (NULL args, or pre_process called again before the
 * matching post_process -- the stream tracks open/closed itself, independent of frame_open()'s own
 * refusal, because the analysis STFT is pushed before frame_open() is even attempted). pcm_in
 * holds exactly dfn2_stream_hop_size(s) samples. */
int dfn2_stream_pre_process(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io);

/* Closes the hop opened by pre_process: ai_rc == 0 commits (frame_finish), anything else bypasses
 * (frame_bypass). Either way, post_process_freq() is then called once; on its ordinary result
 * (0) one ac_istft_push_planar() runs (AC_OLA_ACCUMULATE: always a full hop, pcm_out is always
 * fully written either way) and pcm_out is saturated from its output. On a post_process_freq()
 * contract error -- defensive only, unreached in correct use since the hop was just closed --
 * pcm_out is zeroed directly and the synthesis is NOT pushed, mirroring the board samples' own
 * fallback exactly (they treat this as unreachable too).
 *
 * The return value (1 valid / 0 still latency filler / -1 contract error, no open hop) is
 * dfn2_prepost_post_process_freq()'s own core-level spectral validity flag -- it is NOT, by
 * itself, "pcm_out's samples are this call's own input hop, through the model, delayed by
 * exactly L+D hops." This stream's own analysis/synthesis STFT framing adds one more hop on top
 * of that core latency (documented at the top of this file for the tail-flush case), so the
 * return value turns 1 one hop BEFORE pcm_out's content is actually the audio aligned with the
 * hop that was input L+D hops ago -- confirmed: at 48 kHz/1024/full/mask_lookahead=df_lookahead=1,
 * post_process already returns 1 at the hop-2 call (t=2), but pcm_out there is still a full hop
 * of near-zero warm-up filler; the hop-0-input-aligned, non-filler PCM only appears at the hop-3
 * call (t=3). A caller that needs sample-accurate alignment must count hops itself (L + D + the
 * one stream-framing hop), not branch on this return value. Always call this once after a
 * successful pre_process, never more than once per hop. */
int dfn2_stream_post_process(DFN2Stream *s, int ai_rc, int16_t *pcm_out);

#ifdef __cplusplus
}
#endif

#endif /* DFN2_STREAM_H */
