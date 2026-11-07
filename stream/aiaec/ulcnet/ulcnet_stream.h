/* Align-ULCNet PCM stream: the one header a board sample includes. Wraps two analysis AcStft
 * (linear-error, raw far), one synthesis AcIstft (AC_OLA_ENV_NORM) and lib/aiaec/ulcnet's
 * UlcnetPrepost frame_open/finish/bypass per-hop lifecycle around int16 PCM, one hop per call
 * (no HPF -- the board sample has none for this model), plus the alignment-driven reset/reprime
 * bookkeeping the board sample does itself
 * (alg_ai3_alignulcnet.c's stream_reset() + g_reprime_frames). This model is a post-filter
 * BEHIND a linear AEC: its two inputs are the AEC's linear-error hop and the raw far hop, never
 * raw mic -- see pipelines/ulcnet/ulcnet_nn_stage.h's own doc for why (trained on the analysis
 * seam, not the AEC's guarded rendering base). `pipelines/ulcnet/ulcnet_nn_stage.{c,h}` is a
 * different, higher-level component (a synchronous "installed model" callback consumer used by
 * the 4-channel pipeline) built on raw complex spectra, not on this model's real NPU tensor
 * boundary (compressed features + history/GRU state); it is untouched by this stream and stays
 * the right entry point for that caller. Frequency-domain callers with their own framing should
 * use lib/aiaec/ulcnet/ulcnet_prepost.h directly instead.
 *
 *     ctx = ulcnet_stream_init(&cfg, pool, pool_bytes);
 *     while (have a hop of int16 err_in, far_in) {
 *         UlcnetHopIO io;
 *         UlcnetStreamDelay delay = { .changed = ..., .solid = ... };   // from the caller's own AEC
 *         int pre_rc = ulcnet_stream_pre_process(ctx, err_in, far_in, delay, &io);
 *         if (pre_rc < 0) break;                                // contract error
 *         int ai_rc = 0;
 *         if (pre_rc == 1) ai_rc = run_accelerator(&io.inputs, &io.outputs);   // 0 = mask written
 *         if (ulcnet_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *         // pcm_out is always a full hop; write/play it unconditionally. ai_rc is ignored by
 *         // post_process when pre_rc was 0 (a reprime hop): the frame was already bypassed
 *         // inside pre_process, before any accelerator run was due.
 *     }
 *
 * Alignment / reprime: `delay.changed` (a new alignment generation starts this hop) or the first
 * hop where `delay.solid` turns true after a prior hop where it was false triggers, at the START
 * of that hop's pre_process (before this hop's own analysis): a full reset (both analysis STFTs,
 * the synthesis ISTFT and UlcnetPrepost together -- ulcnet_stream_reset(), the same function a
 * caller can invoke directly) and arms ULCNET_STREAM_REPRIME_FRAMES identity hops. During a
 * reprime hop, pre_process still runs this hop's own STFT analysis and pre_process_freq (the
 * framing must advance every hop), but bypasses the frame directly (frame_bypass(), no
 * frame_open() attempted) and returns 0: there is no accelerator job this hop. Only the
 * solid-transition half of this edge is suppressed on the very first hop after init/reset (there
 * is no prior solid value yet to compare against, same as the board sample's own processed-hop
 * guard); `delay.changed == 1` on that same first hop still triggers the reset, exactly as the
 * board sample's unguarded `changed` check does -- a caller whose AEC already reports a fresh
 * alignment generation on hop 0 gets a reset there too, which is correct, not a bug.
 *
 * Fail-OPEN, not fail-closed (the opposite of DeepVQE-S): a bypassed frame -- a reprime hop, a
 * refused accelerator run (ai_rc != 0), or frame_finish's own non-finite rejection -- passes the
 * linear-error spectrum through unchanged (ulcnet_prepost.h's frame_bypass(): "the error spectrum
 * passes through without the AEC output guard, so on a guard hop it can exceed the capture").
 * This is deliberate: the error is already the AEC's own residual, and a model failure should not
 * make the signal worse than running without the model at all.
 *
 * First hop: AC_OLA_ENV_NORM's synthesis emits no samples on its very first push after
 * init/reset/reprime-triggered-reset (audio_stft.h); that one hop's pcm_out is zero-filled by
 * this stream, mirroring the board sample's own `if (emitted != hop) memset(...)` fallback.
 * post_process distinguishes this expected case (returns 0) from a post_process_freq() contract
 * error (returns -1, pcm_out still zero-filled). */

#ifndef ULCNET_STREAM_H
#define ULCNET_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_prepost.h"   /* UlcnetGrid, UlcnetModelIoInputs/Outputs */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_STREAM_DESCRIPTOR_VERSION 1u
/* In build_flags_hash; bump on any change of pool size, carve order, or descriptor content, the
 * UlcnetPrepost pool and hash included. */
#define ULCNET_STREAM_LAYOUT_VERSION     1u
#define ULCNET_STREAM_ALIGNMENT          16u   /* the one alignment of every module */
/* Identity hops after an alignment boundary; matches ULCNET_NN_STAGE_REPRIME_FRAMES
 * (pipelines/ulcnet/ulcnet_nn_stage.h) and the board sample's own ALIGNULCNET_REPRIME_FRAMES. */
#define ULCNET_STREAM_REPRIME_FRAMES     1u

/* Alignment events of the hop, from the caller's own linear AEC. A 4-channel unit passes its
 * front frame's delay.changed/solid; a mono lane maps AecLinearContext.delay_state
 * (changed = CHANGED, solid = not UNLOCKED) -- same mapping pipelines/ulcnet/ulcnet_nn_stage.h's
 * UlcnetNnDelay documents, reused here verbatim since it is the model-agnostic signal, not a
 * stage-specific one. */
typedef struct UlcnetStreamDelay {
    int changed;
    int solid;
} UlcnetStreamDelay;

typedef struct UlcnetStreamConfig {
    UlcnetGrid grid;   /* ulcnet_grid_make() */
    /* Optional recurrent-state registration (borrowed, all four or none; see ulcnet_prepost.h's
     * UlcnetPrepostConfig): every pointer non-NULL with a nonzero element count, or all four
     * NULL/0 (the accelerator owns its own state copy instead). The pointer values and element
     * counts themselves are not in the pool size or hash (UlcnetPrepostConfig's own contract);
     * only the on/off registration *mode* is folded into build_flags_hash below, so switching a
     * caller between registered and accelerator-owned state is still caught as a stale-pool
     * mismatch by init_ex(). */
    float  *key_history;
    size_t  key_history_elements;
    float  *value_history;
    size_t  value_history_elements;
    float  *logit_history;
    size_t  logit_history_elements;
    float  *h_gru;
    size_t  gru_hidden_elements;
} UlcnetStreamConfig;

/* Fixed 32-byte descriptor; a pool sized for another grid, state-registration mode or build is
 * refused by init. */
typedef struct UlcnetStreamMemReq {
    uint32_t descriptor_version;  /* ULCNET_STREAM_DESCRIPTOR_VERSION        */
    uint32_t layout_version;      /* ULCNET_STREAM_LAYOUT_VERSION            */
    uint32_t reserved0;           /* must be 0                                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, state-registration mode and the folded
                                    * UlcnetPrepost hash */
    uint32_t alignment;           /* ULCNET_STREAM_ALIGNMENT                  */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} UlcnetStreamMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(UlcnetStreamMemReq) == 32,
               "UlcnetStreamMemReq must stay 32 bytes");
#endif
#endif

typedef struct UlcnetStream UlcnetStream;

/* Exact pool size for cfg. -1 (req untouched) on NULL, an invalid grid, or an inconsistent state
 * registration (some but not all of the four pointer/count pairs set). */
int ulcnet_stream_get_mem_size(const UlcnetStreamConfig *cfg, UlcnetStreamMemReq *req);

/* Construct in caller memory (16-byte aligned, >= get_mem_size()'s bytes): the two analysis
 * AcStft (error, far), the synthesis AcIstft and the UlcnetPrepost class, all carved from `pool`;
 * starts reset (the first hop never triggers the alignment-edge reset, same as a fresh board
 * sample run). No heap, no stdio. NULL on a bad config or a misaligned/undersized pool. */
UlcnetStream *ulcnet_stream_init(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes);

/* As ulcnet_stream_init, plus the stale-pool gate: a pool carved for another grid, state-
 * registration mode or layout is refused. `expected` NULL behaves as ulcnet_stream_init. */
UlcnetStream *ulcnet_stream_init_ex(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes,
                                    const UlcnetStreamMemReq *expected);

/* Back to the start: both analysis STFTs, the synthesis ISTFT and UlcnetPrepost all reset
 * (registered state zeroed if any, generation advances); any pending reprime countdown is
 * cleared (the next hop is exactly like the first hop after init). This is the same operation
 * the alignment-edge detector inside pre_process runs automatically; call it directly for a
 * caller-driven full reset (e.g. a stream restart unrelated to alignment). */
void ulcnet_stream_reset(UlcnetStream *s);

/* Generation of the recurrent state (ulcnet_prepost_state_generation()); 0 for NULL. */
uint32_t ulcnet_stream_state_generation(const UlcnetStream *s);

const UlcnetGrid *ulcnet_stream_grid(const UlcnetStream *s);   /* NULL for NULL */
int ulcnet_stream_hop_size(const UlcnetStream *s);             /* samples per PCM hop; -1 on NULL */
/* ulcnet_prepost_descriptor(s->prepost)->layout_version: the graph contract this instance is
 * bound to. -1 on NULL. */
int ulcnet_stream_layout_version(const UlcnetStream *s);

/* The model's tensor views of one open hop (UlcnetModelIoInputs/Outputs; ulcnet_model_io.h via
 * ulcnet_prepost.h), plus the generation the accelerator's own state copy must match before this
 * call (reload quantized zero first if it doesn't; sampled right after frame_open(), before the
 * accelerator runs). Valid, and meaningful, only when pre_process returned 1 -- on a reprime hop
 * (pre_process returned 0) the fields are left zeroed and must not be used. */
typedef struct UlcnetHopIO {
    UlcnetModelIoInputs  inputs;
    UlcnetModelIoOutputs outputs;
    uint32_t             state_generation;
} UlcnetHopIO;

/* One PCM hop in: int16 -> float for both err_in and far_in (each ulcnet_stream_hop_size(s)
 * samples -- no HPF), the alignment-edge check (see the file header above; may trigger a full
 * reset), two analysis STFT pushes, ulcnet_prepost pre_process_freq, then either frame_bypass()
 * directly (a reprime hop: no frame_open attempted, *io left zeroed, returns 0 -- call
 * post_process anyway, with any ai_rc, it is ignored) or frame_open() (returns 1 with *io filled:
 * run the accelerator, then call post_process with its result). Returns -1 on a contract error
 * (NULL args, or pre_process called again before the matching post_process -- the stream tracks
 * open/closed itself, independent of frame_open()'s own refusal, because both analysis pushes
 * and the alignment check happen before frame_open() is ever attempted). */
int ulcnet_stream_pre_process(UlcnetStream *s, const int16_t *err_in, const int16_t *far_in,
                              UlcnetStreamDelay delay, UlcnetHopIO *io);

/* Closes the hop opened by pre_process. When pre_process returned 1 (a real accelerator job was
 * due): ai_rc == 0 commits (frame_finish), anything else bypasses (frame_bypass) -- fail OPEN
 * (the linear-error spectrum passes through), not closed; frame_finish's own non-finite
 * rejection falls through to the same bypass. When pre_process returned 0 (a reprime hop): the
 * frame was already bypassed inside pre_process, so ai_rc is ignored and no further frame_finish/
 * frame_bypass dispatch happens here. Either way, post_process_freq() then stages the spectrum;
 * on its ordinary result (0) one ac_istft_push_planar() runs. pcm_out is always a full hop:
 *   - the push returned the full hop (the ordinary case): pcm_out is saturated from it, return 1.
 *   - the push returned 0 (the very first push after init/reset/reprime-triggered-reset,
 *     AC_OLA_ENV_NORM's own documented startup behavior): pcm_out is zeroed, return 0 -- expected,
 *     not an error.
 *   - post_process_freq() itself returned nonzero (defensive only, unreached in correct use since
 *     the hop was just closed): pcm_out is zeroed, return -1, same as any other contract error.
 * Returns -1 also on a contract error with no pending hop (no pcm_out write in that case). Always
 * call this once after every pre_process call (both its 1 and its 0 return), never more than
 * once per hop. */
int ulcnet_stream_post_process(UlcnetStream *s, int ai_rc, int16_t *pcm_out);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_STREAM_H */
