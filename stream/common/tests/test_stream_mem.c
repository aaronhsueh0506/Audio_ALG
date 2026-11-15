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
    {   /* round half away from zero, in LSB: not truncation (0.6), not ties to even (2.5), not floor(v + 0.5) (-0.5) */
        static const float lsb[][2] = {
            { 0.4f, 0 }, { 0.5f, 1 }, { 0.6f, 1 }, { 1.5f, 2 }, { 2.5f, 3 }, { -0.4f, 0 }, { -0.5f, -1 }, { -0.6f, -1 },
            { -1.5f, -2 }, { -2.5f, -3 }, { 32766.4f, 32766 }, { 32766.5f, 32767 }, { 32767.4f, 32767 }, { 32767.6f, 32767 },
            { -32767.4f, -32767 }, { -32767.5f, -32768 }, { -32768.6f, -32768 },
        };
        size_t i;
        int x, bad = 0;
        for (i = 0; i < sizeof lsb / sizeof lsb[0]; ++i)
            CHECK(stream_saturate_s16(lsb[i][0] / 32768.0f) == (int16_t)lsb[i][1], "saturate(%g LSB) = %d, want %d",
                  (double)lsb[i][0], stream_saturate_s16(lsb[i][0] / 32768.0f), (int)lsb[i][1]);
        /* int16 -> float -> int16 is exact, also one float step either side of the exact value */
        for (x = -32768; x <= 32767; ++x) {
            const float f = (float)x / 32768.0f;
            bad += stream_saturate_s16(f) != x || stream_saturate_s16(nextafterf(f, 0.0f)) != x ||
                   stream_saturate_s16(nextafterf(f, x < 0 ? -1.0f : 1.0f)) != x;
        }
        CHECK(bad == 0, "int16 round trip: %d of 65536 values not restored", bad);
    }

    if (g_fail) { printf("test_stream_mem: %d of %d check(s) FAILED\n", g_fail, g_checks); return 1; }
    printf("test_stream_mem: PASS (%d checks)\n", g_checks);
    return 0;
}
