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
    if (ck_carve(base, cursor, bytes, 1u, out) != 0) return -1;
    if (out && !base) *out = NULL;
    return 0;
}

/* One FNV-1a-32 fold of `v`'s four bytes into `h`: once per field, seeded with AC_FNV1A_SEED. */
static inline uint32_t stream_fnv1a_u32(uint32_t h, uint32_t v) {
    return ac_fnv1a_u32(h, v);
}

/* float PCM (full scale 1.0f) -> int16: scale by 32768, clip to the asymmetric int16 range, round half away from zero
 * exactly (truncate, then compare the exact remainder with 0.5; no float add of 0.5, which would carry the float just
 * below 0.5 LSB up to 1). Same rule as audiolib wav_io.h's writer. NaN is 0, since the cast would be undefined; +-Inf clip. */
static inline int16_t stream_saturate_s16(float v) {
    int32_t t;
    float d;
    v *= 32768.0f;
    if (isnan(v)) return 0;   /* only NaN is UB here; +-Inf clip below */
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    t = (int32_t)v;           /* toward zero; v - t is exact for |v| < 2^23 */
    d = v - (float)t;
    return (int16_t)(t + (d >= 0.5f) - (d <= -0.5f));
}

#ifdef __cplusplus
}
#endif

#endif /* STREAM_COMMON_MEM_H */
