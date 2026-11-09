/** Sample-level builder of one mcab flow (not part of the mcab module): the application-side
 * init sequence of mcab.h in one place, for samples and tests that do not need to show it
 * themselves. One caller pool holds, each 16-byte aligned and in this order: the four lanes
 * (mcab_lane_config() + aec_init()), the shared estimator object and its pool (MATCHED only,
 * delay_aec_init()) and mcab's own pool (mcab_init_ex()). A beamformer helper
 * (mcab_sample_beam.h) is built by the caller on top of `flow`. */
#ifndef MCAB_SAMPLE_FLOW_H
#define MCAB_SAMPLE_FLOW_H

#include <stddef.h>

#include "mcab.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct McabSampleFlow {
    Mcab* flow;
    Aec* lanes[MCAB_CHANNELS];
    DelayAec* delay;              /* in the pool; NULL outside MATCHED */
    void* owned_heap;             /* mcab_sample_flow_create() only */
} McabSampleFlow;

/** Bytes of the whole pool for cfg, a multiple of 16. 0, or -1 for an invalid config. */
int mcab_sample_flow_get_mem_size(const McabConfig* cfg, size_t* bytes);

/** Builds every object in `mem` (16-byte aligned, >= the queried bytes, dirty is fine).
 * 0, or -1 (nothing usable; `f` zeroed) on any refusal. */
int mcab_sample_flow_init(McabSampleFlow* f, void* mem, size_t bytes, const McabConfig* cfg);

/** The same in one heap block owned by `f`. */
int mcab_sample_flow_create(McabSampleFlow* f, const McabConfig* cfg);

/** The application's teardown order: mcab_destroy(), then the lanes; frees the heap block of
 * mcab_sample_flow_create(). Idempotent. */
void mcab_sample_flow_destroy(McabSampleFlow* f);

#ifdef __cplusplus
}
#endif
#endif /* MCAB_SAMPLE_FLOW_H */
