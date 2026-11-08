/* DeepVQE-S PCM stream: the one header a board sample includes. Wraps two analysis AcStft (mic,
 * far), one synthesis AcIstft (AC_OLA_ENV_NORM) and lib/aiaec/deepvqe_s's DeepVqePrepost
 * frame_open/finish/bypass per-hop lifecycle around int16 PCM, one hop per call. No HPF (the
 * board sample has none for this model); no latency beyond the framing itself -- a frame is
 * ready from the very first hop (deepvqe_prepost.h: "one frame per hop ... no latency beyond
 * it"). Frequency-domain callers with their own framing should use
 * lib/aiaec/deepvqe_s/deepvqe_prepost.h directly instead.
 *
 *     ctx = deepvqe_stream_init(&cfg, pool, pool_bytes);
 *     while (have a hop of int16 mic_in, far_in) {
 *         DeepVqeHopIO io;
 *         if (deepvqe_stream_pre_process(ctx, mic_in, far_in, &io) != 1) break;   // -1 is a contract error
 *         device_state_sync(io.state_generation);    // reload the accelerator's OWN state copy
 *                                                     // with quantized zero if this changed since
 *                                                     // the last hop -- not optional, see below
 *         int ai_rc = run_accelerator(&io.inputs, &io.outputs);    // 0 = taps + every state written
 *         if (ai_rc != 0) device_state_refill_zero(); // the run may have written PART of the
 *                                                     // state before failing; restart it from zero
 *                                                     // now, do not wait for the next hop's sync
 *         if (deepvqe_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *         // pcm_out is always a full hop; write/play it unconditionally.
 *     }
 *
 * The two device_state_* calls above are not this stream's own functions -- they are the adapter's
 * own code operating the accelerator's state buffers directly (this stream only tracks and reports
 * io.state_generation). Skipping the sync leaves a state-dependent accelerator holding a stale
 * copy across a generation change (every sample of that hop then wrong); skipping the post-failure
 * refill leaves it holding a failed run's partial write, since a plain bypass does not itself
 * change the generation the next hop's sync would react to. Same pattern as DFN2's stream (see its
 * own header comment for the exact board-sample line numbers); this model has no registered-state
 * config fields to begin with (DeepVqeStreamConfig.state[]/state_elements[] -- all-NULL registers
 * nothing, same as the smoke test here), but a caller that DOES register still owns this sync.
 *
 * Bypass (ai_rc != 0): fails CLOSED, not through -- the primary input is the raw microphone with
 * no canceller in front of it, so this hop's own contribution to the enhanced spectrum is zeroed,
 * not a passthrough of mic_in (see deepvqe_prepost.h's frame_bypass()). This is a spectral-layer
 * guarantee, not a PCM one: AC_OLA_ENV_NORM's synthesis still overlap-adds the PRECEDING hop's
 * real output on top, so a single bypassed hop's pcm_out is not fully zero (confirmed: after a
 * normal hop, one forced-fail hop still writes a mostly-nonzero PCM hop, tapering off); pcm_out
 * only reaches true silence after enough CONSECUTIVE bypassed hops to flush that one hop of OLA
 * history (50% overlap: two). A caller must not assume a single bypass hop's pcm_out is silent.
 * The CCM ring and any registered state are left exactly as
 * they were on an ai_rc != 0 bypass -- bypass touches neither. frame_finish's own non-finite
 * rejection is different: it zeroes the *registered state* (not the CCM ring, which it also
 * leaves untouched -- "the ring is not pushed") and advances the generation, then this stream
 * still calls frame_bypass() to actually close the hop. There is no DFN2-style warm-up concept
 * here: a frame is ready from hop 0. The CCM ring itself starts zero-filled, though, so the taps
 * the first DEEPVQE_TIME_ORDER-1 hops apply run over a partially-zero raw-mic history -- that is
 * the model's own startup behavior, not a stream artifact, and does not gate pre/post_process's
 * return values the way DFN2's mask_lookahead does.
 *
 * First hop: AC_OLA_ENV_NORM's synthesis emits no samples on its very first push after
 * init/reset (audio_stft.h); that one hop's pcm_out is zero-filled by this stream, mirroring the
 * board sample's own `if (written != hop) memset(...)` fallback. post_process distinguishes this
 * expected case (returns 0) from a post_process_freq() contract error (returns -1, pcm_out still
 * zero-filled) -- see post_process's own doc below. */

#ifndef DEEPVQE_STREAM_H
#define DEEPVQE_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_prepost.h"   /* DeepVqeGrid, DeepVqePrepostInputs/Outputs, DeepVqeStateId */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_STREAM_DESCRIPTOR_VERSION 1u
/* In build_flags_hash; bump on any change of pool size, carve order, or descriptor content, the
 * DeepVqePrepost pool and hash included. */
#define DEEPVQE_STREAM_LAYOUT_VERSION     1u
#define DEEPVQE_STREAM_ALIGNMENT          16u   /* the one alignment of every module */

typedef struct DeepVqeStreamConfig {
    DeepVqeGrid grid;   /* deepvqe_grid_make() */
    /* Optional recurrent-state registration (borrowed, all-or-none, by DeepVqeStateId; see
     * deepvqe_prepost.h's DeepVqePrepostConfig): every pointer non-NULL with a nonzero element
     * count, or all NULL/0 (the accelerator owns its own state copy instead). Not in the pool
     * size or hash. */
    float  *state[DEEPVQE_STATE_COUNT];
    size_t  state_elements[DEEPVQE_STATE_COUNT];
} DeepVqeStreamConfig;

/* Fixed 32-byte descriptor; a pool sized for another grid, state-registration mode or build is
 * refused by init. */
typedef struct DeepVqeStreamMemReq {
    uint32_t descriptor_version;  /* DEEPVQE_STREAM_DESCRIPTOR_VERSION       */
    uint32_t layout_version;      /* DEEPVQE_STREAM_LAYOUT_VERSION           */
    uint32_t reserved0;           /* must be 0                                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, state-registration mode and the folded
                                    * DeepVqePrepost hash */
    uint32_t alignment;           /* DEEPVQE_STREAM_ALIGNMENT                 */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} DeepVqeStreamMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DeepVqeStreamMemReq) == 32,
               "DeepVqeStreamMemReq must stay 32 bytes");
#endif
#endif

typedef struct DeepVqeStream DeepVqeStream;

/* Exact pool size for cfg. -1 (req untouched) on NULL, an invalid grid, or an inconsistent state
 * registration (exactly some of cfg->state[]/state_elements[] set). */
int deepvqe_stream_get_mem_size(const DeepVqeStreamConfig *cfg, DeepVqeStreamMemReq *req);

/* Construct in caller memory (16-byte aligned, >= get_mem_size()'s bytes): the two analysis
 * AcStft (mic, far), the synthesis AcIstft and the DeepVqePrepost class, all carved from `pool`;
 * starts reset. No heap, no stdio. NULL on a bad config or a misaligned/undersized pool. */
DeepVqeStream *deepvqe_stream_init(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes);

/* As deepvqe_stream_init, plus the stale-pool gate: a pool carved for another grid, state-
 * registration mode or layout is refused. `expected` NULL behaves as deepvqe_stream_init. */
DeepVqeStream *deepvqe_stream_init_ex(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes,
                                     const DeepVqeStreamMemReq *expected);

/* Back to the start: both analysis STFTs, the synthesis ISTFT and DeepVqePrepost all reset
 * (registered state zeroed if any, generation advances, CCM ring cleared); the next hop is
 * exactly like the first hop after init (AC_OLA_ENV_NORM's one silent hop included). */
void deepvqe_stream_reset(DeepVqeStream *s);

/* Generation of the recurrent state (deepvqe_prepost_state_generation()); 0 for NULL. */
uint32_t deepvqe_stream_state_generation(const DeepVqeStream *s);

const DeepVqeGrid *deepvqe_stream_grid(const DeepVqeStream *s);   /* NULL for NULL */
int deepvqe_stream_hop_size(const DeepVqeStream *s);              /* samples per PCM hop; -1 on NULL */
/* deepvqe_prepost_descriptor(s->prepost)->layout_version: the graph contract this instance is
 * bound to. -1 on NULL. */
int deepvqe_stream_layout_version(const DeepVqeStream *s);

/* The model's tensor views of one open hop (DeepVqePrepostInputs/Outputs; deepvqe_prepost.h),
 * plus the generation the accelerator's own state copy must match before this call (reload
 * quantized zero first if it doesn't; sampled right after frame_open(), before the accelerator
 * runs, so it already reflects the previous hop's bypass/refusal). Valid until
 * deepvqe_stream_post_process(). */
typedef struct DeepVqeHopIO {
    DeepVqePrepostInputs  inputs;
    DeepVqePrepostOutputs outputs;
    uint32_t              state_generation;
} DeepVqeHopIO;

/* One PCM hop in: int16 -> float for both mic_in and far_in (each deepvqe_stream_hop_size(s)
 * samples -- no HPF), two analysis STFT pushes, deepvqe_prepost pre_process_freq + frame_open. Returns 1
 * with *io filled (every hop, from the first one: there is no warm-up here), or -1 on a contract
 * error (NULL args, or pre_process called again before the matching post_process -- the stream
 * tracks open/closed itself, independent of frame_open()'s own refusal, because both analysis
 * pushes happen before frame_open() is even attempted). */
int deepvqe_stream_pre_process(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in,
                               DeepVqeHopIO *io);

/* Closes the hop opened by pre_process: ai_rc == 0 commits (frame_finish), anything else bypasses
 * (frame_bypass) -- fail CLOSED (silence), not a passthrough of mic_in; frame_finish's own
 * non-finite rejection falls through to the same bypass. Either way, post_process_freq() then
 * stages the enhanced (or silent) spectrum; on its ordinary result (0) one
 * ac_istft_push_planar() runs. pcm_out is always a full hop:
 *   - the push returned the full hop (the ordinary case): pcm_out is saturated from it, return 1.
 *   - the push returned 0 (only ever the very first push after init/reset, AC_OLA_ENV_NORM's own
 *     documented startup behavior): pcm_out is zeroed, return 0 -- an expected, not an error,
 *     case; it happens exactly once per init/reset, never again until the next reset.
 *   - post_process_freq() itself returned nonzero (defensive only, unreached in correct use since
 *     the hop was just closed): pcm_out is zeroed, return -1, same as any other contract error.
 * Returns -1 also on a contract error with no open hop (no pcm_out write in that case). Always
 * call this once after a successful pre_process, never more than once per hop. */
int deepvqe_stream_post_process(DeepVqeStream *s, int ai_rc, int16_t *pcm_out);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_STREAM_H */
