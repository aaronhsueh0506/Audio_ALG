/* Unit ERB partition (band 0 owns every bin): lets the DFN2 fail-open path run
 * without the model's exported matrices; not the model's ERB analysis. */
#ifndef DFN_UNIT_PARTITION_H
#define DFN_UNIT_PARTITION_H

#include <stddef.h>
#include <string.h>

#include "dfn2_grid.h"

/* Fills erb_fwd (bin-major [n_bins][n_erb]) and erb_inv (band-major [n_erb][n_bins]) for `grid`. */
static inline void dfn_unit_partition(float *erb_fwd, float *erb_inv, const DFN2Grid *grid) {
    const size_t n_bins = (size_t)grid->n_bins, n_erb = (size_t)grid->n_erb;
    size_t k;
    memset(erb_fwd, 0, n_bins * n_erb * sizeof(float));
    memset(erb_inv, 0, n_erb * n_bins * sizeof(float));
    for (k = 0; k < n_bins; ++k) {
        erb_fwd[k * n_erb] = 1.0f;
        erb_inv[k] = 1.0f;
    }
}

#endif /* DFN_UNIT_PARTITION_H */
