/* aiaec_front_preprocess() against the three calls it replaces. Two independent (front, block
 * context) pairs see the same scene: pair A through the one-shot entry, pair B through
 * aiaec_front_process_i16 + aiaec_front_fill_align_input + aiaec_preprocess. Every block the two
 * work structs must carry the same action, restart mask, state generation and, tensor by tensor, the same
 * shapes and bit-identical float contents; after a deterministic non-identity accelerator the
 * postprocess PCM must be byte-equal. Run at N = 1 and N = 8 on the int16 path. The scene's delay change
 * makes one block an event (BYPASS at N = 1, a frame_reset bit at N = 8), so the events are in play. The
 * failure gates (NULL arguments, a context of another grid, a second pre before the post) must leave
 * the pair usable, with no postprocess owed, and leave the AEC where it was. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_front.h"
#include "scene.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

#define BINS 129
#define HOP 128
#define BLOCKS 504   /* hops; a multiple of 8 */

typedef struct Pair {
    AiaecConfig cfg;
    AiaecCtx *ctx;
    AiaecFront *front;
    void *bpool, *fpool;
} Pair;

static void pair_init(Pair *p, int n) {
    StreamMemReq req;
    AiaecInfo info;
    size_t fbytes;
    int err = 0;
    memset(p, 0, sizeof(*p));
    p->cfg.sr = 16000; p->cfg.fft = 256; p->cfg.hop_in = HOP * n;
    CHECK(aiaec_get_mem_size(&p->cfg, &req) == AIAEC_OK, "N=%d block mem size", n);
    if (posix_memalign(&p->bpool, req.alignment, (size_t)req.bytes) != 0) abort();
    err = aiaec_init(&p->cfg, p->bpool, (size_t)req.bytes, &p->ctx, &info);
    CHECK(err == AIAEC_OK && p->ctx != NULL, "N=%d block init err=%d", n, err);
    fbytes = aiaec_front_get_mem_size(&p->cfg);
    CHECK(fbytes > 0, "N=%d front size", n);
    if (posix_memalign(&p->fpool, AIAEC_FRONT_ALIGNMENT, fbytes) != 0) abort();
    p->front = aiaec_front_init(p->fpool, fbytes, &p->cfg, &err);
    CHECK(p->front != NULL, "N=%d front init err=%d", n, err);
}

static void pair_free(Pair *p) { free(p->bpool); free(p->fpool); }

static int pre_three(Pair *p, const int16_t *mic, const int16_t *far, AiaecWork *w) {
    AiaecFrontOut fo;
    AiaecInput in;
    int rc = aiaec_front_process_i16(p->front, mic, far, p->cfg.hop_in, &fo);
    if (rc != AIAEC_OK) return rc;
    aiaec_front_fill_align_input(&fo, &in);
    return aiaec_preprocess(p->ctx, &in, w);
}

/* Same everything except pointers into the two contexts. */
static int same_work(const AiaecWork *a, const AiaecWork *b) {
    int i;
    if (a->action != b->action || a->input_count != b->input_count || a->output_count != b->output_count ||
        a->state_generation != b->state_generation || a->synthesis_restart_mask != b->synthesis_restart_mask ||
        a->synthesis_restarted != b->synthesis_restarted)
        return 0;
    for (i = 0; i < a->input_count; ++i) {
        const AiaecTensorIn *x = &a->inputs[i], *y = &b->inputs[i];
        if (x->elements != y->elements || x->w != y->w || x->h != y->h || x->c != y->c || x->b != y->b) return 0;
        if (x->elements && memcmp(x->data, y->data, x->elements * sizeof(float)) != 0) return 0;
    }
    for (i = 0; i < a->output_count; ++i) {
        const AiaecTensorOut *x = &a->outputs[i], *y = &b->outputs[i];
        if (x->elements != y->elements || x->w != y->w || x->h != y->h || x->c != y->c || x->b != y->b) return 0;
    }
    return 1;
}

/* Deterministic non-identity mask, identical for both pairs. */
static void fill_mask(const AiaecWork *w, int n, int blk) {
    float *out = w->outputs[0].data;
    int k, b;
    for (k = 0; k < n; ++k)
        for (b = 0; b < BINS; ++b) {
            out[(size_t)k * BINS + b] = 0.3f + 0.7f * (float)((b + blk) % 9) / 8.0f;
            out[((size_t)n + (size_t)k) * BINS + b] = 0.05f * (float)((b + k) % 5);
        }
}

static void run_equiv(const Scene *sc, int n, int *events, int *discriminating) {
    static int16_t pa[BLOCKS * HOP], pb[BLOCKS * HOP];
    const int hop_in = HOP * n, packets = BLOCKS / n;
    Pair A, B;
    int b;

    pair_init(&A, n);
    pair_init(&B, n);
    if (!A.ctx || !A.front || !B.ctx || !B.front) return;
    *events = 0;
    for (b = 0; b < packets && !g_fail; ++b) {
        const int16_t *mic = sc->mic + (size_t)b * hop_in, *far = sc->far + (size_t)b * hop_in;
        AiaecWork wa, wb;
        int ra = aiaec_front_preprocess(A.front, A.ctx, mic, far, &wa);
        int rb = pre_three(&B, mic, far, &wb);
        CHECK(ra == AIAEC_OK && rb == AIAEC_OK, "N=%d block %d: rc one-shot %d, three-call %d", n, b, ra, rb);
        if (ra != AIAEC_OK || rb != AIAEC_OK) break;
        if (!same_work(&wa, &wb)) {
            CHECK(0, "N=%d block %d: work differs (action %d/%d, mask %llx/%llx, generation %u/%u)", n, b,
                  (int)wa.action, (int)wb.action, (unsigned long long)wa.synthesis_restart_mask,
                  (unsigned long long)wb.synthesis_restart_mask, wa.state_generation, wb.state_generation);
            break;
        }
        if (wa.synthesis_restart_mask) ++*events;
        if (wa.action == AIAEC_SUBMIT) {
            fill_mask(&wa, n, b);
            fill_mask(&wb, n, b);
        }
        CHECK(aiaec_postprocess(A.ctx, 0, pa + (size_t)b * hop_in) == AIAEC_OK, "N=%d block %d: post A", n, b);
        CHECK(aiaec_postprocess(B.ctx, 0, pb + (size_t)b * hop_in) == AIAEC_OK, "N=%d block %d: post B", n, b);
    }
    CHECK(memcmp(pa, pb, sizeof(pa)) == 0, "N=%d: postprocess PCM differs between the one-shot and the three calls", n);
    {
        int i;
        for (i = 4 * HOP; i < BLOCKS * HOP; ++i)
            if (abs((int)pa[i] - (int)sc->mic[i - HOP]) > 500) ++*discriminating;
    }
    pair_free(&A);
    pair_free(&B);
}

static void run_gates(const Scene *sc) {
    static int16_t pa[HOP], pb[HOP];
    Pair A, B, C;
    AiaecWork wa, wb, wz;
    AiaecConfig cfg512 = { 16000, 512, 256 };   /* same N = 1, another grid: only the sample count differs */
    StreamMemReq req512;
    AiaecCtx *ctx512 = NULL;
    void *pool512 = NULL;
    int b, rc;

    pair_init(&A, 1);
    pair_init(&B, 1);
    pair_init(&C, 8);   /* a context of another grid */
    if (!A.ctx || !A.front || !B.ctx || !B.front || !C.ctx) return;
    CHECK(aiaec_get_mem_size(&cfg512, &req512) == AIAEC_OK, "16k/512 block mem size");
    if (posix_memalign(&pool512, req512.alignment, (size_t)req512.bytes) != 0) abort();
    {
        AiaecInfo info512;
        CHECK(aiaec_init(&cfg512, pool512, (size_t)req512.bytes, &ctx512, &info512) == AIAEC_OK && ctx512 != NULL, "16k/512 block init");
    }

    memset(&wz, 0, sizeof(wz));
    for (b = 0; b < 40 && !g_fail; ++b) {
        const int16_t *mic = sc->mic + (size_t)b * HOP, *far = sc->far + (size_t)b * HOP;
        if (b == 10) {
            /* NULL arguments: AIAEC_ERR_NULL, work zeroed, nothing advanced, no postprocess owed. */
            memset(&wa, 0x5A, sizeof(wa));
            rc = aiaec_front_preprocess(NULL, A.ctx, mic, far, &wa);
            CHECK(rc == AIAEC_ERR_NULL && memcmp(&wa, &wz, sizeof(wa)) == 0, "NULL front: rc %d or work not zeroed", rc);
            memset(&wa, 0x5A, sizeof(wa));
            rc = aiaec_front_preprocess(A.front, NULL, mic, far, &wa);
            CHECK(rc == AIAEC_ERR_NULL && memcmp(&wa, &wz, sizeof(wa)) == 0, "NULL ctx: rc %d or work not zeroed", rc);
            memset(&wa, 0x5A, sizeof(wa));
            rc = aiaec_front_preprocess(A.front, A.ctx, NULL, far, &wa);
            CHECK(rc == AIAEC_ERR_NULL && memcmp(&wa, &wz, sizeof(wa)) == 0, "NULL mic: rc %d or work not zeroed", rc);
            rc = aiaec_front_preprocess(A.front, A.ctx, mic, NULL, &wa);
            CHECK(rc == AIAEC_ERR_NULL, "NULL far: rc %d", rc);
            rc = aiaec_front_preprocess(A.front, A.ctx, mic, far, NULL);
            CHECK(rc == AIAEC_ERR_NULL, "NULL work: rc %d", rc);
            /* a block context of another grid: refused before the AEC runs, so B needs no skip */
            rc = aiaec_front_preprocess(A.front, C.ctx, mic, far, &wa);
            CHECK(rc == AIAEC_ERR_INPUT_SIZE && memcmp(&wa, &wz, sizeof(wa)) == 0, "wrong-grid ctx: rc %d", rc);
            memset(&wa, 0x5A, sizeof(wa));
            rc = aiaec_front_preprocess(A.front, ctx512, mic, far, &wa);
            CHECK(rc == AIAEC_ERR_INPUT_SIZE && memcmp(&wa, &wz, sizeof(wa)) == 0, "same-N ctx of another grid: rc %d", rc);
            continue;
        }
        rc = aiaec_front_preprocess(A.front, A.ctx, mic, far, &wa);
        CHECK(rc == AIAEC_OK, "block %d after the gates: rc %d (no postprocess may be owed)", b, rc);
        CHECK(pre_three(&B, mic, far, &wb) == AIAEC_OK, "block %d: three-call rc", b);
        if (rc != AIAEC_OK) break;
        if (!same_work(&wa, &wb)) { CHECK(0, "block %d after the gates: work differs", b); break; }
        if (wa.action == AIAEC_SUBMIT) { fill_mask(&wa, 1, b); fill_mask(&wb, 1, b); }
        CHECK(aiaec_postprocess(A.ctx, 0, pa) == AIAEC_OK && aiaec_postprocess(B.ctx, 0, pb) == AIAEC_OK, "post %d", b);
        CHECK(memcmp(pa, pb, sizeof(pa)) == 0, "block %d after the gates: PCM differs", b);
    }
    pair_free(&A);
    pair_free(&B);
    pair_free(&C);
    free(pool512);
}

/* A refused call leaves the AEC and the context untouched: stream B gets a second pre at packet 20 while the
 * first is still open (AIAEC_ERR_SEQUENCE), stream A does not; every packet from 20 on, work and PCM, is
 * bit-identical between the two. */
static void run_duplicate(const Scene *sc, int n) {
    static int16_t pa[BLOCKS * HOP], pb[BLOCKS * HOP];
    const int hop_in = HOP * n, packets = BLOCKS / n, dup_at = 20 < packets ? 20 : packets / 2;
    Pair A, B;
    int b;

    pair_init(&A, n);
    pair_init(&B, n);
    if (!A.ctx || !A.front || !B.ctx || !B.front) return;
    for (b = 0; b < packets && !g_fail; ++b) {
        const int16_t *mic = sc->mic + (size_t)b * hop_in, *far = sc->far + (size_t)b * hop_in;
        AiaecWork wa, wb, wd, wz;
        int ra = aiaec_front_preprocess(A.front, A.ctx, mic, far, &wa);
        int rb = aiaec_front_preprocess(B.front, B.ctx, mic, far, &wb);
        CHECK(ra == AIAEC_OK && rb == AIAEC_OK, "dup N=%d block %d: rc %d/%d", n, b, ra, rb);
        if (ra != AIAEC_OK || rb != AIAEC_OK) break;
        if (b == dup_at) {
            const int16_t *m2 = sc->mic + (size_t)(b + 1) * hop_in, *f2 = sc->far + (size_t)(b + 1) * hop_in;
            memset(&wz, 0, sizeof(wz));
            memset(&wd, 0x5A, sizeof(wd));
            rb = aiaec_front_preprocess(B.front, B.ctx, m2, f2, &wd);
            CHECK(rb == AIAEC_ERR_SEQUENCE && memcmp(&wd, &wz, sizeof(wd)) == 0, "dup N=%d: rc %d or work not zeroed", n, rb);
        }
        if (!same_work(&wa, &wb)) { CHECK(0, "dup N=%d block %d: work differs", n, b); break; }
        if (wa.action == AIAEC_SUBMIT) { fill_mask(&wa, n, b); fill_mask(&wb, n, b); }
        CHECK(aiaec_postprocess(A.ctx, 0, pa + (size_t)b * hop_in) == AIAEC_OK &&
              aiaec_postprocess(B.ctx, 0, pb + (size_t)b * hop_in) == AIAEC_OK, "dup N=%d block %d: post", n, b);
    }
    CHECK(memcmp(pa, pb, sizeof(pa)) == 0, "dup N=%d: PCM differs from the control after the refused call", n);
    pair_free(&A);
    pair_free(&B);
}

int main(void) {
    Scene sc = scene_make(BLOCKS * HOP, 480);
    int ev1 = 0, ev8 = 0, d1 = 0, d8 = 0;

    run_equiv(&sc, 1, &ev1, &d1);
    run_equiv(&sc, 8, &ev8, &d8);
    run_gates(&sc);
    run_duplicate(&sc, 1);
    run_duplicate(&sc, 8);
    CHECK(ev1 >= 1 && ev8 >= 1, "the scene made no alignment event (N=1 %d, N=8 %d): the equivalence would not cover events", ev1, ev8);
    CHECK(d1 > 1000 && d8 > 1000, "PCM never differs from the mic (%d, %d): the comparison cannot tell the paths apart", d1, d8);
    scene_free(&sc);
    if (g_fail) { fprintf(stderr, "test_front_oneshot: %d failure(s)\n", g_fail); return 1; }
    printf("test_front_oneshot: PASS (N=1 and N=8 bit-identical to the three calls over %d hops; event blocks N=1 %d, N=8 %d)\n", BLOCKS, ev1, ev8);
    return 0;
}
