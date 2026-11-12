/* Product front end for Align-ULCNet: the linear AEC and delay policy that turn the microphone and far end into what
 * aiaec_preprocess() takes (the AEC's float residual, the raw far of the same hop, one alignment event per model hop).
 * Its own unit, libaiaec_front.a: libaiaec.a has no AEC dependency, this one links the AEC library. A standalone sample
 * sets the same {sr, fft, hop_in} as for aiaec_init() and nothing else; the delay policy is internal (BALANCED, online
 * matched-filter estimate, 5 filters).
 *
 *     AiaecConfig cfg = { sr, fft, hop_in };
 *     bytes = aiaec_front_get_mem_size(&cfg);                     // 0: refused
 *     f = aiaec_front_init(pool, bytes, &cfg, &err);              // pool: AIAEC_FRONT_ALIGNMENT
 *     per block: aiaec_front_preprocess(f, ctx, mic, far, &work); // the linear AEC, then the block pre
 *                ...SUBMIT: the sample's AI3 run; BYPASS: none...
 *                aiaec_postprocess(ctx, ai_rc, pcm_out);          // exactly once per successful pre
 *
 * The AEC runs context-only (enable_res = 0, return_res_context = 1); the residual is the analysis seam
 * (AecResContext.formed_hop). int16 input is scaled by 1/32768 once; float input is used as is; no int16 round trip
 * on the way to the block entry. A pipeline with its own linear AEC passes its residual, raw far and events to
 * aiaec_preprocess() directly. */

#ifndef AIAEC_FRONT_H
#define AIAEC_FRONT_H

#include <stddef.h>
#include <stdint.h>

#include "aiaec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AIAEC_FRONT_ALIGNMENT 16u

enum {
    AIAEC_FRONT_ERR_DELAY = -201,   /* illegal delay policy (reachable only through aiaec_front_test.h) */
    AIAEC_FRONT_ERR_AEC = -202      /* the AEC library refused the resolved configuration */
};

/* One block's output; pointers are front-owned (raw_far of a float call is the caller's far buffer), valid until the
 * next process or reset call. events: one per model hop, oldest first; event_count == hop_in / (fft / 2). */
typedef struct AiaecFrontOut {
    const float           *formed_hop;
    const float           *raw_far;
    const AiaecAlignEvent *events;
    int                    event_count;
    int                    samples;
} AiaecFrontOut;

typedef struct AiaecFront AiaecFront;

/* Pool bytes, 0 for a refused grid (reason from aiaec_front_init). */
size_t aiaec_front_get_mem_size(const AiaecConfig *grid);

/* grid: the AiaecConfig given to aiaec_init(). Statuses: AIAEC_ERR_NULL / _GRID / _IN_HOP / _POOL, AIAEC_FRONT_ERR_AEC. */
AiaecFront *aiaec_front_init(void *pool, size_t pool_bytes, const AiaecConfig *grid, int *err);

/* One block, mic and far int16 of grid.hop_in samples each: the linear AEC over the model hops, the events, then
 * aiaec_preprocess() (same returns and work). ctx: same AiaecConfig, no block open. A refusal zeroes work, leaves AEC and ctx
 * untouched and owes no postprocess; success owes one. AIAEC_FRONT_ERR_AEC owes none, but the AEC may have advanced. */
int aiaec_front_preprocess(AiaecFront *f, AiaecCtx *ctx, const int16_t *mic, const int16_t *far,
                           AiaecWork *work);

/* Lower-level steps of aiaec_front_preprocess; exactly grid.hop_in samples per call, else AIAEC_ERR_INPUT_SIZE. */
int aiaec_front_process_i16(AiaecFront *f, const int16_t *mic, const int16_t *far, int samples,
                            AiaecFrontOut *out);
int aiaec_front_process_f32(AiaecFront *f, const float *mic, const float *far, int samples,
                            AiaecFrontOut *out);

/* Back to the initial AEC state. The block context is not touched. */
void aiaec_front_reset(AiaecFront *f);

/* Short text for any status the front or the block entry returns (AIAEC_FRONT_ERR_* here, the rest as aiaec_status_text()).
 * Never NULL. Defined with aiaec_front_preprocess(), so it needs libaiaec.a at link time. */
const char *aiaec_front_status_text(int code);

/* AecLinearDelayState (UNLOCKED 0, LOCKED 1, CHANGED 2) -> event: CHANGED -> changed and solid, LOCKED -> solid, else
 * neither (CHANGED is solid so the LOCKED hop after it is not a second event). */
AiaecAlignEvent aiaec_front_event_from_delay_state(int delay_state);

/* AIAEC_INPUT_ALIGN_ULCNET arm of AiaecInput from one front output. */
void aiaec_front_fill_align_input(const AiaecFrontOut *out, AiaecInput *in);

#ifdef __cplusplus
}
#endif

#endif /* AIAEC_FRONT_H */
