/* Test-only access to the delay policy of aiaec_front, for the other delay modes and the policy gates; not part of
 * the customer interface. */

#ifndef AIAEC_FRONT_TEST_H
#define AIAEC_FRONT_TEST_H

#include <stddef.h>

#include "aiaec_front.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum AiaecFrontDelayMode {
    AIAEC_FRONT_DELAY_MATCHED  = 0,   /* online matched-filter estimate; needs delay_num_filters 1..5 */
    AIAEC_FRONT_DELAY_FIXED    = 1,   /* caller-measured delay; needs fixed_delay_samples >= 0 */
    AIAEC_FRONT_DELAY_EXTERNAL = 2    /* far already aligned to the mic; no estimator, always solid */
} AiaecFrontDelayMode;

typedef struct AiaecFrontConfig {
    AiaecConfig grid;              /* sr, fft, hop_in; AEC hop = fft / 2, hop_in a multiple of it */
    int delay_mode;                /* AiaecFrontDelayMode */
    int delay_num_filters;         /* MATCHED only, 1..5 */
    int fixed_delay_samples;       /* FIXED only, >= 0; -1 otherwise */
} AiaecFrontConfig;

/* The product's policy: MATCHED, delay_num_filters 5, fixed_delay_samples -1. */
void aiaec_front_config_defaults(AiaecFrontConfig *fc, const AiaecConfig *grid);

size_t aiaec_front_get_mem_size_cfg(const AiaecFrontConfig *fc);
AiaecFront *aiaec_front_init_cfg(void *pool, size_t pool_bytes, const AiaecFrontConfig *fc, int *err);

#ifdef __cplusplus
}
#endif

#endif /* AIAEC_FRONT_TEST_H */
