/* stream_range.h: stream_plane_within against the bit-pattern reference |x| <= limit over the limits the three
 * streams use, lengths around the vector widths, and every special value at every position. A limit moved by one
 * ulp must change at least one answer, so the comparison is able to fail. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "stream_range.h"

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int ref_within(const float *x, size_t n, uint32_t lim) {
    size_t k;
    for (k = 0; k < n; ++k) {
        uint32_t b;
        memcpy(&b, &x[k], 4);
        if ((b & 0x7FFFFFFFu) > lim) return 0;
    }
    return 1;
}

static float f_of(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

int main(void) {
    const uint32_t lims[] = {0x5F000000u, 0x7F7FFFFFu, 0x77800000u};
    const size_t lens[] = {0, 1, 3, 4, 5, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1025};
    long mutated_diff = 0;
    size_t li, ni, pos, s, k;

    for (li = 0; li < sizeof lims / sizeof lims[0]; ++li) {
        const uint32_t L = lims[li];
        const float specials[] = {0.0f, -0.0f, 1.0f, -1.0f, f_of(L), -f_of(L), f_of(L + 1u), -f_of(L + 1u), f_of(L - 1u),
                                  nanf(""), -nanf(""), INFINITY, -INFINITY, f_of(0x00000001u), f_of(0x7FC00001u),
                                  f_of(0xFFFFFFFFu), 3.0e38f, f_of(0x7F800001u)};
        for (ni = 0; ni < sizeof lens / sizeof lens[0]; ++ni) {
            const size_t n = lens[ni];
            float *x = (float *)calloc(n + 1u, sizeof(float));
            if (x == NULL) return 1;
            for (s = 0; s < sizeof specials / sizeof specials[0]; ++s) {
                for (pos = 0; pos < (n ? n : 1u); ++pos) {
                    int got, want;
                    for (k = 0; k < n; ++k) x[k] = 0.25f * (float)(k % 7);
                    if (n) x[pos] = specials[s];
                    got = stream_plane_within(x, n, L);
                    want = ref_within(x, n, L);
                    CHECK(got == want, "lim=%08x n=%zu pos=%zu special=%zu got=%d want=%d", (unsigned)L, n, pos, s, got, want);
                    mutated_diff += (stream_plane_within(x, n, L - 1u) != want);
                }
            }
            free(x);
        }
    }
    CHECK(mutated_diff > 0, "a limit one ulp lower changed no answer: the comparison cannot fail");
    printf("test_stream_range: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail != 0;
}
