#ifndef STREAM_COMMON_RANGE_H
#define STREAM_COMMON_RANGE_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "utility/simd_kernel_nn.h"

/* 1 when every |x[k]| <= the float whose IEEE bits are limit_bits; NaN and Inf are above any finite limit. */
static inline int stream_plane_within(const float *x, size_t n, uint32_t limit_bits)
{
    float limit;
    memcpy(&limit, &limit_bits, sizeof limit);
    return skn_all_within_f32(x, n, limit);
}
#endif
