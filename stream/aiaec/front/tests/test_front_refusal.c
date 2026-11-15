/* A non-finite sample (the AEC answers AEC_ERR_ARG) or a finite one that overflows the hop (3e38: AEC_ERR_DSP) in hop 0 or hop 1
 * of a block is refused (AIAEC_FRONT_ERR_AEC). The refusal resets the AEC, so the next block equals a fresh front's output even
 * when an earlier hop of the refused block had already advanced it. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aec.h"
#include "aiaec_front.h"
#include "aiaec_front_test.h"

enum { N = 256 };
static AiaecFront *mk(void **pool) {
    AiaecConfig g; AiaecFrontConfig fc; int err; size_t b;
    g.sr = 16000; g.fft = 256; g.hop_in = N;
    aiaec_front_config_defaults(&fc, &g);
    b = aiaec_front_get_mem_size_cfg(&fc);
    if (posix_memalign(pool, 16, b) != 0) abort();
    return aiaec_front_init_cfg(*pool, b, &fc, &err);
}
static void sig(float *m, float *f, int blk) {
    for (int i = 0; i < N; ++i) {
        unsigned s = (unsigned)(blk * N + i) * 2654435761u;
        f[i] = ((s >> 8) & 0xffff) / 65536.f - .5f;
        m[i] = 0.5f * f[i] + 0.1f * (((s * 7) >> 9) & 0xff) / 256.f;
    }
}
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); ++fails; } } while (0)

int main(void) {
    static const int bad_at[] = { 3, N / 2 + 3 };   /* hop 0 and hop 1 of a two-hop block */
    static const float bad_value[] = { NAN, 3.0e38f };
    for (int w = 0; w < 4; ++w) {
        void *p1, *p2; AiaecFront *a = mk(&p1), *fresh = mk(&p2);
        AiaecFrontOut oa, of; float m[N], f[N], bad[N];
        for (int k = 0; k < 30; ++k) { sig(m, f, k); CHECK(aiaec_front_process_f32(a, m, f, N, &oa) == AIAEC_OK, "warm-up"); }
        sig(m, f, 30); memcpy(bad, m, sizeof m); bad[bad_at[w % 2]] = bad_value[w / 2];
        CHECK(aiaec_front_process_f32(a, bad, f, N, &oa) == AIAEC_FRONT_ERR_AEC, "case %d: the bad mic sample is refused", w);
        CHECK(oa.formed_hop == NULL && oa.event_count == 0, "out cleared");
        sig(m, f, 31);
        CHECK(aiaec_front_process_f32(a, m, f, N, &oa) == AIAEC_OK, "block after the refusal");
        CHECK(aiaec_front_process_f32(fresh, m, f, N, &of) == AIAEC_OK, "fresh block");
        CHECK(memcmp(oa.formed_hop, of.formed_hop, N * sizeof(float)) == 0, "case %d: residual differs from a fresh front's", w);
        CHECK(memcmp(oa.events, of.events, 2 * sizeof(AiaecAlignEvent)) == 0, "case %d: events differ from a fresh front's", w);
        free(p1); free(p2);
    }
    puts(fails ? "test_front_refusal: FAIL" : "test_front_refusal: PASS");
    return fails != 0;
}
