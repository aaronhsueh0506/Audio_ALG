/* Align-ULCNet PCM stream: two analysis AcStft (linear-error, raw far), one synthesis AcIstft
 * (AC_OLA_ENV_NORM), lib/aiaec/ulcnet's UlcnetPrepost per-hop lifecycle, and alignment-driven
 * reset/reprime, around int16 PCM, one hop per call, no HPF. Post-filter BEHIND a linear AEC:
 * inputs are the AEC's linear-error hop and raw far, never raw mic (see
 * pipelines/ulcnet/ulcnet_nn_stage.h for why). That file is a different, higher-level component
 * for the 4-channel pipeline and stays the right entry point for that caller. Frequency-domain
 * callers with their own framing should use lib/aiaec/ulcnet/ulcnet_prepost.h directly.
 *
 *     ctx = ulcnet_stream_init(&cfg, pool, pool_bytes);
 *     while (have a hop of int16 err_in, far_in) {
 *         UlcnetHopIO io;
 *         UlcnetStreamDelay delay = { .changed = ..., .solid = ... };   // from the caller's own AEC
 *         int pre_rc = ulcnet_stream_pre_process(ctx, err_in, far_in, delay, &io);
 *         if (pre_rc < 0) break;                                // contract error
 *         int ai_rc = 0;
 *         if (pre_rc == 1) {
 *             device_state_sync(io.state_generation);      // reload accelerator state on change
 *             ai_rc = run_accelerator(&io.inputs, &io.outputs);
 *             if (ai_rc != 0) device_state_refill_zero(); // partial write -> reload now, same hop
 *         }
 *         if (ulcnet_stream_post_process(ctx, ai_rc, pcm_out) < 0) break;
 *     }
 *
 * Alignment / reprime: `delay.changed`, or the first hop where `delay.solid` turns true, triggers
 * a full reset (ulcnet_stream_reset()) and arms ULCNET_STREAM_REPRIME_FRAMES identity hops;
 * pre_process still runs this hop's STFT/pre_process_freq but bypasses the frame and returns 0 --
 * no accelerator job that hop.
 *
 * Fail-OPEN, not fail-closed (the opposite of DeepVQE-S): a bypassed frame passes the
 * linear-error spectrum through unchanged, since the error is already the AEC's own residual and
 * a model failure should not make the signal worse than running without the model. */

#ifndef ULCNET_STREAM_H
#define ULCNET_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_prepost.h"   /* UlcnetGrid, UlcnetModelIoInputs/Outputs */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_STREAM_DESCRIPTOR_VERSION 1u
#define ULCNET_STREAM_LAYOUT_VERSION     1u   /* bump with build_flags_hash on any pool/carve/descriptor change */
#define ULCNET_STREAM_ALIGNMENT          16u   /* the one alignment of every module */
/* Identity hops after an alignment boundary; matches ULCNET_NN_STAGE_REPRIME_FRAMES
 * (pipelines/ulcnet/ulcnet_nn_stage.h). */
#define ULCNET_STREAM_REPRIME_FRAMES     1u

/* Alignment events of the hop, from the caller's own linear AEC (same mapping
 * pipelines/ulcnet/ulcnet_nn_stage.h's UlcnetNnDelay documents). */
typedef struct UlcnetStreamDelay {
    int changed;
    int solid;
} UlcnetStreamDelay;

typedef struct UlcnetStreamConfig {
    UlcnetGrid grid;   /* ulcnet_grid_make() */
    /* Optional recurrent-state registration (borrowed, all four or none; see ulcnet_prepost.h's
     * UlcnetPrepostConfig): every pointer non-NULL with a nonzero element count, or all four
     * NULL/0 (the accelerator owns its own state copy instead). Only the on/off mode is folded
     * into build_flags_hash below. */
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

/* Construct in caller memory (16-byte aligned, >= get_mem_size()'s bytes), carved from `pool`;
 * starts reset. No heap, no stdio. NULL on a bad config or a misaligned/undersized pool. */
UlcnetStream *ulcnet_stream_init(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes);

/* As ulcnet_stream_init, plus the stale-pool gate: a pool carved for another grid, state-
 * registration mode or layout is refused. `expected` NULL behaves as ulcnet_stream_init. */
UlcnetStream *ulcnet_stream_init_ex(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes,
                                    const UlcnetStreamMemReq *expected);

/* Back to the start: framing and UlcnetPrepost reset (registered state zeroed if any, generation
 * advances), any pending reprime countdown cleared. Same operation pre_process's alignment-edge
 * detector runs automatically; call directly for a caller-driven full reset. */
void ulcnet_stream_reset(UlcnetStream *s);

/* Generation of the recurrent state (ulcnet_prepost_state_generation()); 0 for NULL. */
uint32_t ulcnet_stream_state_generation(const UlcnetStream *s);

const UlcnetGrid *ulcnet_stream_grid(const UlcnetStream *s);   /* NULL for NULL */
int ulcnet_stream_hop_size(const UlcnetStream *s);             /* samples per PCM hop; -1 on NULL */
/* ulcnet_prepost_descriptor(s->prepost)->layout_version: the graph contract this instance is
 * bound to. -1 on NULL. */
int ulcnet_stream_layout_version(const UlcnetStream *s);

/* The model's tensor views of one open hop, plus the generation the accelerator's own state
 * copy must match before this call. Valid only when pre_process returned 1 -- on a reprime hop
 * (returned 0) the fields are left zeroed and must not be used. */
typedef struct UlcnetHopIO {
    UlcnetModelIoInputs  inputs;
    UlcnetModelIoOutputs outputs;
    uint32_t             state_generation;
} UlcnetHopIO;

/* One hop in, as full-scale (~=1.0) float for both err_in and far_in (each
 * ulcnet_stream_hop_size(s) samples, no HPF): the alignment-edge check (may trigger a full
 * reset), two analysis STFT pushes, pre_process_freq, then either frame_bypass() directly (a
 * reprime hop: *io left zeroed, returns 0) or frame_open() (returns 1 with *io filled). Returns
 * -1 on a contract error (NULL args, or called again before the matching post_process).
 *
 * Use this when the caller's own err/far lanes are already float (e.g. a linear AEC's float
 * error output, AecResContext.formed_hop) -- routing a float lane through
 * ulcnet_stream_pre_process() below instead round-trips it through int16 for no reason, and is
 * measurably not the same signal (up to 4 LSB of output PCM differs). */
int ulcnet_stream_pre_process_f32(UlcnetStream *s, const float *err_in, const float *far_in,
                                  UlcnetStreamDelay delay, UlcnetHopIO *io);

/* int16 convenience wrapper: converts to float (/32768.0f), then runs
 * ulcnet_stream_pre_process_f32()'s path. Same return-value contract. */
int ulcnet_stream_pre_process(UlcnetStream *s, const int16_t *err_in, const int16_t *far_in,
                              UlcnetStreamDelay delay, UlcnetHopIO *io);

/* Closes the hop opened by pre_process. When pre_process returned 1: ai_rc == 0 commits
 * (frame_finish), anything else bypasses (frame_bypass) -- fail OPEN, not closed. When
 * pre_process returned 0 (a reprime hop): already bypassed, ai_rc is ignored. pcm_out is always
 * a full hop: the ordinary push is saturated from it (return 1); the very first push after
 * init/reset emits nothing, so pcm_out is zeroed (return 0, expected); a post_process_freq()
 * contract error also zeroes pcm_out (return -1). Call once per pre_process call, never more
 * than once per hop. */
int ulcnet_stream_post_process(UlcnetStream *s, int ai_rc, int16_t *pcm_out);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_STREAM_H */
