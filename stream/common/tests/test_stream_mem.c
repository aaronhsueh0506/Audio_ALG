/* stream_mem.h unit test: StreamMemReq layout, stream_carve, stream_fnv1a_u32, stream_saturate_s16. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "stream_bind.h"
#include "stream_mem.h"

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    StreamMemReq r;
    size_t cur = 0;
    void *p = NULL, *q = NULL;
    unsigned char base[256];

    r.bytes = (uint64_t)1 << 40; r.alignment = 16u;
    CHECK(r.bytes == ((uint64_t)1 << 40) && r.alignment == 16u, "StreamMemReq fields");
    CHECK(offsetof(StreamMemReq, bytes) == 0u && offsetof(StreamMemReq, alignment) == sizeof(uint64_t) &&
          sizeof(StreamMemReq) <= 2u * sizeof(uint64_t), "StreamMemReq layout");

    CHECK(stream_carve(NULL, &cur, 10, &p) == 0 && p == NULL && cur == 10, "size-only carve");
    CHECK(stream_carve(base, &cur, 8, &q) == 0 && q == (void *)(base + 16) && cur == 24, "aligned carve");
    cur = 24;
    CHECK(stream_carve(NULL, &cur, (size_t)-1, &p) == -1 && cur == 24, "overflow leaves cursor");

    CHECK(stream_fnv1a_u32(2166136261u, 0u) == 0x4b95f515u, "fnv1a of 0 (got %08x)", stream_fnv1a_u32(2166136261u, 0u));
    CHECK(stream_saturate_s16(0.5f) == 16384 && stream_saturate_s16(2.0f) == 32767 &&
          stream_saturate_s16(-2.0f) == -32768 && stream_saturate_s16(NAN) == 0 &&
          stream_saturate_s16(INFINITY) == 32767 && stream_saturate_s16(-INFINITY) == -32768, "saturate");

    if (g_fail) { printf("test_stream_mem: %d of %d check(s) FAILED\n", g_fail, g_checks); return 1; }
    printf("test_stream_mem: PASS (%d checks)\n", g_checks);
    return 0;
}
