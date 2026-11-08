/* Shared, model-agnostic helpers for the stream/ PCM modules (stream/ainr/dfn2,
 * stream/aiaec/deepvqe_s, stream/aiaec/ulcnet): the 16-byte-aligned pool carve, the FNV-1a-32
 * fold used to build each module's own build_flags_hash, and the float-to-int16 PCM saturate.
 * static inline, header-only: including this adds no link dependency and does not compromise
 * any one model's ability to ship its own stream/ directory independently of the others -- these
 * three functions carry no model-specific content.
 *
 * This duplication across the three stream modules was flagged by the simplify pass; audiolib's
 * own utility/mem_align.h would be the architecturally deeper home (it already hosts
 * ac_memreq_abi_hash(), the same FNV-1a-32 fold, and six-plus per-model prepost source files
 * already hand-roll an equivalent carve+hash pair), but that header lives in the separate
 * audiolib repo and editing it needs its own commit+pin-bump cycle -- out of scope for this pass.
 * This header is the same-repo fix; promoting it into mem_align.h is a reasonable follow-up, not
 * done here. */

#ifndef STREAM_COMMON_MEM_H
#define STREAM_COMMON_MEM_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "utility/mem_align.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reserve `bytes` at *cursor, 16-byte aligned; base NULL sizes only (out stays NULL). Saturating:
 * an overflow pins *cursor to SIZE_MAX, which MEM_SIZE_INVALID catches on the next check. */
static inline int stream_carve(unsigned char *base, size_t *cursor, size_t bytes, void **out) {
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, bytes);
    if (MEM_SIZE_INVALID(next)) return -1;
    if (out) *out = base ? (void *)(base + aligned) : NULL;
    *cursor = next;
    return 0;
}

/* One FNV-1a-32 fold of `v`'s four bytes into `h`; call once per field building a
 * build_flags_hash, seeded from FNV offset basis 2166136261u. */
static inline uint32_t stream_fnv1a_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* float PCM (full-scale ~=1.0f) -> saturated int16, the convention every stream module's
 * post_process uses: scale by 32768 first, then clip to the asymmetric int16 range. A NaN input
 * (confirmed reachable: a finite but extreme accelerator output -- a large mask or CCM tap, both
 * accepted since they are finite -- can still overflow to NaN in the DSP between the accelerator
 * and this saturate, e.g. in the synthesis ISTFT's own math) would otherwise compare false
 * against both clip bounds, leave v as NaN, and reach `(int16_t)v`, which is undefined behavior
 * for a non-finite float (confirmed under UBSan: "nan is outside the range of representable
 * values of type 'short'") -- not merely unspecified rounding, an actual UB cast the compiler is
 * free to miscompile. +-Inf do not need this: they already compare true against a clip bound and
 * saturate correctly. Defined here as silence (0), matching this codebase's convention elsewhere
 * that a non-finite result is treated as invalid rather than propagated. */
static inline int16_t stream_saturate_s16(float v) {
    v *= 32768.0f;
    if (isnan(v)) return 0;   /* only NaN is UB here; +-Inf already clip correctly below */
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    return (int16_t)v;
}

#ifdef __cplusplus
}
#endif

#endif /* STREAM_COMMON_MEM_H */
