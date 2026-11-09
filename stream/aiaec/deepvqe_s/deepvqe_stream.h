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
 * device_state_sync/refill_zero are the caller's own code on accelerator memory this stream
 * never touches (it only tracks/reports io.state_generation). This model has no registered-state
 * config fields by default (DeepVqeStreamConfig.state[]/state_elements[] all-NULL), but a caller
 * that does register still owns this sync.
 *
 * Bypass (ai_rc != 0) fails CLOSED: the primary input is the raw microphone with no canceller in
 * front of it, so this hop's contribution to the enhanced spectrum is zeroed, not a passthrough
 * of mic_in -- a spectral-layer guarantee only, not a PCM one (AC_OLA_ENV_NORM's synthesis still
 * overlap-adds the preceding hop's real output, so a single bypassed hop's pcm_out is not fully
 * zero). There is no DFN2-style warm-up here -- a frame is ready from hop 0, though the CCM ring
 * starts zero-filled so the first DEEPVQE_TIME_ORDER-1 hops' taps run over a partially-zero
 * history.
 *
 * First hop: AC_OLA_ENV_NORM's synthesis emits no samples on its very first push after
 * init/reset; that hop's pcm_out is zero-filled here (post_process returns 0, not an error). */

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

/* Construct in caller memory (16-byte aligned, >= get_mem_size()'s bytes), carved from `pool`;
 * starts reset. No heap, no stdio. NULL on a bad config or a misaligned/undersized pool. */
DeepVqeStream *deepvqe_stream_init(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes);

/* As deepvqe_stream_init, plus the stale-pool gate: a pool carved for another grid, state-
 * registration mode or layout is refused. `expected` NULL behaves as deepvqe_stream_init. */
DeepVqeStream *deepvqe_stream_init_ex(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes,
                                     const DeepVqeStreamMemReq *expected);

/* Back to the start: framing and DeepVqePrepost reset (registered state zeroed if any, generation
 * advances, CCM ring cleared); the next hop is exactly like the first hop after init. */
void deepvqe_stream_reset(DeepVqeStream *s);

/* Generation of the recurrent state (deepvqe_prepost_state_generation()); 0 for NULL. */
uint32_t deepvqe_stream_state_generation(const DeepVqeStream *s);

const DeepVqeGrid *deepvqe_stream_grid(const DeepVqeStream *s);   /* NULL for NULL */
int deepvqe_stream_hop_size(const DeepVqeStream *s);              /* samples per PCM hop; -1 on NULL */
/* deepvqe_prepost_descriptor(s->prepost)->layout_version: the graph contract this instance is
 * bound to. -1 on NULL. */
int deepvqe_stream_layout_version(const DeepVqeStream *s);

/* The model's tensor views of one open hop, plus the generation the accelerator's own state
 * copy must match before this call. Valid until deepvqe_stream_post_process(). */
typedef struct DeepVqeHopIO {
    DeepVqePrepostInputs  inputs;
    DeepVqePrepostOutputs outputs;
    uint32_t              state_generation;
} DeepVqeHopIO;

/* One PCM hop in: int16 -> float for both mic_in and far_in (each deepvqe_stream_hop_size(s)
 * samples, no HPF), two analysis STFT pushes, pre_process_freq + frame_open. Returns 1 with *io
 * filled (every hop, from the first one -- there is no warm-up here), or -1 on a contract error
 * (NULL args, or called again before the matching post_process). */
int deepvqe_stream_pre_process(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in,
                               DeepVqeHopIO *io);

/* Closes the hop opened by pre_process: ai_rc == 0 commits (frame_finish), anything else bypasses
 * (frame_bypass) -- fail CLOSED (silence), not a passthrough of mic_in. pcm_out is always a full
 * hop: the ordinary push is saturated from it (return 1); the very first push after init/reset
 * emits nothing, so pcm_out is zeroed (return 0, expected); a post_process_freq() contract error
 * also zeroes pcm_out (return -1). Call once per successful pre_process, never more than once
 * per hop. */
int deepvqe_stream_post_process(DeepVqeStream *s, int ai_rc, int16_t *pcm_out);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_STREAM_H */
