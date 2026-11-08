/* DeepVQE-S PCM stream: wraps two analysis AcStft (mic, far), one synthesis AcIstft
 * (AC_OLA_ENV_NORM) and lib/aiaec/deepvqe_s's DeepVqePrepost frame_open/finish/bypass per-hop
 * lifecycle around int16 PCM, one hop per call. No HPF; no latency beyond the framing itself --
 * a frame is ready from the very first hop. Frequency-domain callers should use
 * lib/aiaec/deepvqe_s/deepvqe_prepost.h directly.
 *
 *     ctx = deepvqe_stream_init(&cfg, pool, pool_bytes);
 *     while (have a hop of int16 mic_in, far_in) {
 *         DeepVqeHopIO io;
 *         if (deepvqe_stream_pre_process(ctx, mic_in, far_in, &io) != 1) break;   // -1 is a contract error
 *         device_state_sync(io.state_generation);      // reload accelerator state on change
 *         int ai_rc = run_accelerator(&io.inputs, &io.outputs);
 *         if (ai_rc != 0) device_state_refill_zero();   // partial write -> reload now, same hop
 *         if (deepvqe_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *     }
 *
 * device_state_sync/refill_zero are the adapter's own code operating accelerator memory this
 * stream never touches (it only tracks/reports io.state_generation); skipping sync leaves a
 * stale state copy live across a generation change, skipping refill leaves a failed run's
 * garbage write live until the next change -- same pattern as DFN2's stream. This model has no
 * registered-state config fields by default (DeepVqeStreamConfig.state[]/state_elements[]
 * all-NULL registers nothing), but a caller that does register still owns this sync.
 *
 * Bypass (ai_rc != 0) fails CLOSED: the primary input is the raw microphone with no canceller in
 * front of it, so this hop's own contribution to the enhanced spectrum is zeroed, not a
 * passthrough of mic_in. This is a spectral-layer guarantee only, not a PCM one -- AC_OLA_ENV_NORM's
 * synthesis still overlap-adds the PRECEDING hop's real output, so a single bypassed hop's
 * pcm_out is not fully zero (confirmed: one forced-fail hop after a normal one still writes
 * mostly-nonzero PCM, tapering off); true silence needs enough CONSECUTIVE bypassed hops to flush
 * that OLA history (50% overlap: two). frame_finish's own non-finite rejection zeroes the
 * registered state and advances the generation, then still falls through to frame_bypass() to
 * close the hop; bypass itself touches neither the CCM ring nor registered state. There is no
 * DFN2-style warm-up here -- a frame is ready from hop 0, though the CCM ring starts zero-filled
 * so the first DEEPVQE_TIME_ORDER-1 hops' taps run over a partially-zero history (the model's own
 * startup behavior, not gated by pre/post_process's return values).
 *
 * First hop: AC_OLA_ENV_NORM's synthesis emits no samples on its very first push after
 * init/reset; that hop's pcm_out is zero-filled here, mirroring the board sample's own
 * `if (written != hop) memset(...)` fallback. post_process distinguishes this expected case
 * (returns 0) from a post_process_freq() contract error (returns -1). */

#ifndef DEEPVQE_STREAM_H
#define DEEPVQE_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_prepost.h"   /* DeepVqeGrid, DeepVqePrepostInputs/Outputs, DeepVqeStateId */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_STREAM_DESCRIPTOR_VERSION 1u
#define DEEPVQE_STREAM_LAYOUT_VERSION     1u   /* bump with build_flags_hash on any pool/carve/descriptor change */
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
 * plus the generation the accelerator's own state copy must match before this call (sampled
 * right after frame_open(), before the accelerator runs). Valid until
 * deepvqe_stream_post_process(). */
typedef struct DeepVqeHopIO {
    DeepVqePrepostInputs  inputs;
    DeepVqePrepostOutputs outputs;
    uint32_t              state_generation;
} DeepVqeHopIO;

/* One PCM hop in: int16 -> float for both mic_in and far_in (each deepvqe_stream_hop_size(s)
 * samples -- no HPF), two analysis STFT pushes, pre_process_freq + frame_open. Returns 1 with
 * *io filled (every hop, from the first one: there is no warm-up here), or -1 on a contract
 * error (NULL args, or called again before the matching post_process -- the stream tracks
 * open/closed itself, independent of frame_open()'s own refusal, since both analysis pushes
 * happen before frame_open() is even attempted). */
int deepvqe_stream_pre_process(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in,
                               DeepVqeHopIO *io);

/* Closes the hop opened by pre_process: ai_rc == 0 commits (frame_finish), anything else bypasses
 * (frame_bypass) -- fail CLOSED (silence), not a passthrough of mic_in; frame_finish's own
 * non-finite rejection falls through to the same bypass. Either way, post_process_freq() then
 * stages the enhanced (or silent) spectrum; on its ordinary result (0) one
 * ac_istft_push_planar() runs. pcm_out is always a full hop:
 *   - the push returned the full hop (the ordinary case): pcm_out is saturated from it, return 1.
 *   - the push returned 0 (only ever the very first push after init/reset, AC_OLA_ENV_NORM's own
 *     documented startup behavior): pcm_out is zeroed, return 0 -- expected, not an error; it
 *     happens exactly once per init/reset.
 *   - post_process_freq() itself returned nonzero (defensive only, unreached in correct use since
 *     the hop was just closed): pcm_out is zeroed, return -1, same as any other contract error.
 * Returns -1 also on a contract error with no open hop (no pcm_out write in that case). Call once
 * per successful pre_process, never more than once per hop. */
int deepvqe_stream_post_process(DeepVqeStream *s, int ai_rc, int16_t *pcm_out);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_STREAM_H */
