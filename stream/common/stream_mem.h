/* Model-agnostic memory and PCM helpers for the stream/ modules: StreamMemReq, the 16-byte-aligned
 * pool carve, the FNV-1a-32 fold for each module's build_flags_hash, and the float-to-int16 PCM
 * saturate. Header-only, so including it adds no link dependency. */

#ifndef STREAM_COMMON_MEM_H
#define STREAM_COMMON_MEM_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "utility/mem_align.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pool size and alignment a library needs for one config. */
typedef struct StreamMemReq {
    uint64_t bytes;
    uint32_t alignment;
} StreamMemReq;

/* Reserves `bytes` at *cursor, 16-byte aligned; base NULL only sizes (out stays NULL). -1 on size
 * overflow, with *cursor and *out untouched. */
static inline int stream_carve(unsigned char *base, size_t *cursor, size_t bytes, void **out) {
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, bytes);
    if (MEM_SIZE_INVALID(next)) return -1;
    if (out) *out = base ? (void *)(base + aligned) : NULL;
    *cursor = next;
    return 0;
}

/* One FNV-1a-32 fold of `v`'s four bytes into `h`: once per field, seeded with 2166136261u. */
static inline uint32_t stream_fnv1a_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* float PCM (full scale 1.0f) -> int16: scale by 32768, clip to the asymmetric int16 range. NaN (an
 * extreme accelerator output) is 0, since the cast would be undefined; +-Inf clip. */
static inline int16_t stream_saturate_s16(float v) {
    v *= 32768.0f;
    if (isnan(v)) return 0;   /* only NaN is UB here; +-Inf clip below */
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    return (int16_t)v;
}

#ifdef __cplusplus
}
#endif

#endif /* STREAM_COMMON_MEM_H */
