/* aiaec_front feeding the real Align-ULCNet block entry (test catalog): the events the front derives
 * from the AEC drive exactly the block's own restart edges. One delay change is one CHANGED hop and
 * must cost exactly one event: one BYPASS block at N = 1, one frame_reset bit and one
 * synthesis_restart_mask bit at N = 8 -- the LOCKED hop after the CHANGED hop is an ordinary hop.
 * The accelerator is an identity mask, so the PCM the block returns is the front's float residual,
 * one hop late, except the event block, which is exactly silent (the PCM entry restarts its own
 * STFT/ISTFT at an event hop); N = 8 returns the same PCM as N = 1, sample for sample. */

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

typedef struct Pass {
    int16_t pcm[BLOCKS * HOP];
    float resid[BLOCKS * HOP];
    int edge[BLOCKS];        /* per hop, the consumers' edge rule applied to the front's events */
    int changed;             /* CHANGED hops */
    int bypass_blocks;       /* BYPASS blocks */
    int submit_blocks;
    int reset_ones;          /* frame_reset values equal to 1 (N = 8) */
    int mask_bits;           /* synthesis_restart_mask bits (N = 8 and N = 1) */
} Pass;

/* The accelerator of the graph with an identity mask: mask real = 1, imaginary = 0, for every frame. */
static void identity_mask(const AiaecWork *w, int n) {
    float *out = w->outputs[0].data;
    int k, b;
    CHECK(w->outputs[0].elements == (size_t)2 * n * BINS, "mask elements %zu", w->outputs[0].elements);
    for (k = 0; k < n; ++k)
        for (b = 0; b < BINS; ++b) {
            out[(size_t)k * BINS + b] = 1.0f;
            out[((size_t)n + (size_t)k) * BINS + b] = 0.0f;
        }
}

static int popcount64(uint64_t m) {
    int c = 0;
    for (; m; m >>= 1) c += (int)(m & 1u);
    return c;
}

static void run_pass(const Scene *sc, int n, Pass *p) {
    const int hop_in = HOP * n, packets = BLOCKS / n;
    AiaecConfig cfg;
    StreamMemReq req;
    AiaecInfo info;
    AiaecCtx *ctx = NULL;
    AiaecFront *front;
    void *bpool = NULL, *fpool = NULL;
    size_t fbytes;
    int err = 0, b, k, had_hop = 0, prev_solid = 0;

    memset(p, 0, sizeof(*p));
    cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = hop_in;
    CHECK(aiaec_get_mem_size(&cfg, &req) == AIAEC_OK, "N=%d block mem size", n);
    if (posix_memalign(&bpool, req.alignment, (size_t)req.bytes) != 0) abort();
    err = aiaec_init(&cfg, bpool, (size_t)req.bytes, &ctx, &info);
    CHECK(err == AIAEC_OK && ctx != NULL, "N=%d block init err=%d", n, err);

    fbytes = aiaec_front_get_mem_size(&cfg);
    if (posix_memalign(&fpool, AIAEC_FRONT_ALIGNMENT, fbytes) != 0) abort();
    front = aiaec_front_init(fpool, fbytes, &cfg, &err);
    CHECK(front != NULL, "N=%d front init err=%d", n, err);
    if (!ctx || !front) return;

    for (b = 0; b < packets && !g_fail; ++b) {
        AiaecFrontOut fo;
        AiaecInput in;
        AiaecWork work;
        int rc = aiaec_front_process_i16(front, sc->mic + (size_t)b * hop_in, sc->far + (size_t)b * hop_in, hop_in, &fo);
        CHECK(rc == AIAEC_OK, "N=%d front rc=%d", n, rc);
        if (rc != AIAEC_OK) break;
        memcpy(p->resid + (size_t)b * hop_in, fo.formed_hop, sizeof(float) * (size_t)hop_in);

        for (k = 0; k < n; ++k) {
            const AiaecAlignEvent *e = &fo.events[k];
            p->edge[b * n + k] = e->changed || (had_hop && !prev_solid && e->solid);
            had_hop = 1;
            prev_solid = e->solid;
            p->changed += e->changed;
        }

        aiaec_front_fill_align_input(&fo, &in);
        rc = aiaec_preprocess(ctx, &in, &work);
        CHECK(rc == AIAEC_OK, "N=%d preprocess rc=%d (%s)", n, rc, aiaec_front_status_text(rc));
        if (rc != AIAEC_OK) break;
        p->mask_bits += popcount64(work.synthesis_restart_mask);
        if (n == 1) {
            CHECK((work.action == AIAEC_BYPASS) == (p->edge[b] != 0), "block %d: action %s but the events say edge=%d", b,
                  work.action == AIAEC_BYPASS ? "BYPASS" : "SUBMIT", p->edge[b]);
            CHECK(work.synthesis_restart_mask == (uint64_t)(p->edge[b] != 0), "block %d: restart mask %llx", b,
                  (unsigned long long)work.synthesis_restart_mask);
        } else {
            uint64_t want = 0;
            const float *reset = work.inputs[4].data;
            CHECK(work.action == AIAEC_SUBMIT && work.input_count == 5 && work.inputs[4].elements == (size_t)n,
                  "packet %d: action %d inputs %d", b, (int)work.action, work.input_count);
            for (k = 0; k < n; ++k) {
                if (p->edge[b * n + k]) want |= (uint64_t)1u << k;
                CHECK(reset[k] == (p->edge[b * n + k] ? 1.0f : 0.0f), "packet %d frame %d: frame_reset %g, edge %d", b, k,
                      (double)reset[k], p->edge[b * n + k]);
                p->reset_ones += reset[k] == 1.0f;
            }
            CHECK(work.synthesis_restart_mask == want, "packet %d: restart mask %llx want %llx", b,
                  (unsigned long long)work.synthesis_restart_mask, (unsigned long long)want);
        }
        if (work.action == AIAEC_SUBMIT) {
            identity_mask(&work, n);
            ++p->submit_blocks;
        } else {
            ++p->bypass_blocks;
        }
        rc = aiaec_postprocess(ctx, 0, p->pcm + (size_t)b * hop_in);
        CHECK(rc == AIAEC_OK, "N=%d postprocess rc=%d", n, rc);
    }
    free(bpool);
    free(fpool);
}

int main(void) {
    static Pass p1, p8;
    Scene sc = scene_make(BLOCKS * HOP, 480);
    int b, i, compared = 0, discriminating = 0, edges1 = 0, edges8 = 0;

    run_pass(&sc, 1, &p1);
    run_pass(&sc, 8, &p8);
    if (g_fail) return 1;

    for (b = 0; b < BLOCKS; ++b) { edges1 += p1.edge[b]; edges8 += p8.edge[b]; }
    CHECK(p1.changed >= 1 && p1.changed == p8.changed && edges1 == edges8, "N=1 changed %d edges %d, N=8 changed %d edges %d",
          p1.changed, edges1, p8.changed, edges8);
    for (b = 0; b < BLOCKS; ++b) CHECK(p1.edge[b] == p8.edge[b], "hop %d: edge N=1 %d N=8 %d", b, p1.edge[b], p8.edge[b]);

    /* One delay change is one event: the CHANGED hop and nothing else. */
    CHECK(p1.bypass_blocks == p1.changed && edges1 == p1.changed, "N=1: %d BYPASS blocks, %d edges for %d CHANGED hops",
          p1.bypass_blocks, edges1, p1.changed);
    CHECK(p8.reset_ones == p8.changed && p8.mask_bits == p8.changed && p8.bypass_blocks == 0,
          "N=8: %d frame_reset ones, %d restart bits, %d BYPASS blocks for %d CHANGED hops", p8.reset_ones, p8.mask_bits,
          p8.bypass_blocks, p8.changed);
    CHECK(p1.mask_bits == p1.changed, "N=1: %d restart bits for %d CHANGED hops", p1.mask_bits, p1.changed);
    for (b = 1; b < BLOCKS; ++b) CHECK(!(p1.edge[b] && p1.edge[b - 1]), "hops %d and %d are both events", b - 1, b);
    CHECK(p1.submit_blocks == BLOCKS - p1.bypass_blocks, "N=1 submit blocks %d", p1.submit_blocks);

    /* Every hop from 4 on is compared, the event hop included: an event hop returns exactly silence, every other hop the
     * residual one hop late. */
    for (b = 4; b < BLOCKS; ++b) {
        for (i = 0; i < HOP; ++i) {
            const float want = p1.edge[b] ? 0.0f : p1.resid[(size_t)(b - 1) * HOP + i] * 32768.0f;
            const float got = (float)p1.pcm[(size_t)b * HOP + i];
            CHECK(p1.edge[b] ? got == 0.0f : fabsf(got - want) <= 3.0f, "hop %d (edge=%d) sample %d: pcm %.0f vs %.2f", b, p1.edge[b], i, got, want);
            if (!p1.edge[b] && fabsf((float)sc.mic[(size_t)(b - 1) * HOP + i] - want) > 500.0f) ++discriminating;
        }
        ++compared;
        if (g_fail) break;
    }
    CHECK(compared > 100, "only %d hops compared", compared);
    CHECK(discriminating > 1000, "residual never differs from the mic (%d samples): the test cannot tell them apart", discriminating);
    CHECK(memcmp(p1.pcm, p8.pcm, sizeof(p1.pcm)) == 0, "PCM at N=8 differs from N=1");
    CHECK(memcmp(p1.resid, p8.resid, sizeof(p1.resid)) == 0, "front residual at N=8 differs from N=1");

    {
        const int codes[] = { AIAEC_OK, AIAEC_ERR_GRID, AIAEC_ERR_INPUT_SIZE, AIAEC_FRONT_ERR_DELAY, AIAEC_FRONT_ERR_AEC, -999 };
        const char *texts[6];
        int a, b;
        for (a = 0; a < 6; ++a) {
            texts[a] = aiaec_front_status_text(codes[a]);
            CHECK(texts[a] && texts[a][0], "status text for %d is empty", codes[a]);
            for (b = 0; b < a; ++b) CHECK(strcmp(texts[a], texts[b]) != 0, "status %d and %d share the text '%s'", codes[a], codes[b], texts[a]);
        }
        CHECK(strcmp(texts[2], aiaec_status_text(AIAEC_ERR_INPUT_SIZE)) == 0, "a core status is not delegated to aiaec_status_text()");
        CHECK(strcmp(texts[5], aiaec_status_text(-999)) == 0, "an unknown status is not delegated to aiaec_status_text()");
    }

    scene_free(&sc);
    if (g_fail) { fprintf(stderr, "test_front_block: %d failure(s)\n", g_fail); return 1; }
    printf("test_front_block: PASS (CHANGED hops %d: N=1 bypass=%d, N=8 frame_reset=%d; %d hops compared, %d discriminating samples)\n",
           p1.changed, p1.bypass_blocks, p8.reset_ones, compared, discriminating);
    return 0;
}
