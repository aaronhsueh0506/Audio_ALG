/* The unit ERB partition: band 0 owns every bin, a partition of unity by
 * construction. It lets the DFN2 fail-open path run without the model's
 * exported matrices (board smoke runs, tests); it is NOT the model's ERB
 * analysis and must not be used with a real callback. */
#ifndef DFN_UNIT_PARTITION_H
#define DFN_UNIT_PARTITION_H

#include <string.h>

#include "dfn2_process.h"

static inline void dfn_unit_partition(float *erb_fwd, float *erb_inv) {
    int k;
    memset(erb_fwd, 0, (size_t)DFN2_N_BINS * DFN2_N_ERB * sizeof(float));
    memset(erb_inv, 0, (size_t)DFN2_N_ERB * DFN2_N_BINS * sizeof(float));
    for (k = 0; k < DFN2_N_BINS; ++k) {
        erb_fwd[(size_t)k * DFN2_N_ERB] = 1.0f;
        erb_inv[k] = 1.0f;
    }
}

#endif /* DFN_UNIT_PARTITION_H */
