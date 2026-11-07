#include "ulcnet_process.h"

#include <math.h>

#include "utility/simd_kernel_nn.h"

/* Finite check of a model callback's output (contract: ulcnet_process.h). */
int ulcnet_frame_is_finite(const float *re, const float *im, int n_bins) {
    if (!re || !im || n_bins < 1) return 0;
    return skn_all_finite_f32(re, (size_t)n_bins) &&
           skn_all_finite_f32(im, (size_t)n_bins);
}
