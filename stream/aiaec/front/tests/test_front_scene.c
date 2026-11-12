/* aiaec_front against an independent use of the AEC API with the same configuration, on a synthetic
 * echo scene: the residual, the raw far and every per-hop event must be bit-equal, for the int16 and
 * the float entry, on several grids, block sizes and delay modes. The scene must also make the delay
 * policy actually move (changed / solid hops are counted), and a reset must replay bit-identically. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aiaec_front.h"
#include "aiaec_front_test.h"
#include "scene.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Case {
    const char *name;
    int sr, fft, n_frames, mode, filters, fixed, delay;
} Case;

typedef struct Counts { int changed, solid, hops, edges; } Counts;

static void *alloc16(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 16, n) != 0) abort();
    return p;
}

/* The reference: the AEC API driven directly, config written out field by field. */
static Aec *ref_aec(const Case *c, void **pool) {
    AecConfig a;
    size_t bytes;
    aec_config_from_level(&a, AEC_LEVEL_BALANCED, c->sr);
    a.fft_size = c->fft;
    a.delay_mode = c->mode == 0 ? AEC_DELAY_MATCHED : c->mode == 1 ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
    a.delay_num_filters = c->filters;
    a.fixed_delay_samples = c->fixed;
    a.enable_delay_est = c->mode == 0;
    a.enable_res = 0;
    a.return_res_context = 1;
    bytes = aec_get_mem_size(&a);
    *pool = alloc16(bytes);
    return aec_init(*pool, bytes, &a);
}

static AiaecFront *make_front(const Case *c, void **pool) {
    AiaecFrontConfig fc;
    AiaecConfig g;
    size_t bytes;
    int err = -99;
    AiaecFront *f;
    g.sr = c->sr; g.fft = c->fft; g.hop_in = c->fft / 2 * c->n_frames;
    aiaec_front_config_defaults(&fc, &g);
    fc.delay_mode = c->mode;
    fc.delay_num_filters = c->filters;
    fc.fixed_delay_samples = c->fixed;
    bytes = aiaec_front_get_mem_size_cfg(&fc);
    CHECK(bytes > 0, "%s: mem size 0", c->name);
    *pool = alloc16(bytes);
    f = aiaec_front_init_cfg(*pool, bytes, &fc, &err);
    CHECK(f != NULL && err == AIAEC_OK, "%s: init err=%d", c->name, err);
    return f;
}

static int events_equal(const AiaecAlignEvent *a, const AiaecAlignEvent *b) {
    return a->changed == b->changed && a->solid == b->solid;
}

/* One pass over the scene through `front`, compared hop by hop with the reference AEC. `is_f32`
 * picks the entry. Optionally records the first `keep` blocks' residual into keep_buf. */
static void run_case(const Case *c, const Scene *sc, int is_f32, Counts *cnt, float *keep_buf, int keep_blocks) {
    void *fpool, *rpool;
    AiaecFront *f = make_front(c, &fpool);
    Aec *r = ref_aec(c, &rpool);
    const int hop = c->fft / 2, hop_in = hop * c->n_frames;
    const int blocks = sc->n / hop_in;
    float *mic_f = (float *)alloc16((size_t)sc->n * sizeof(float));
    float *far_f = (float *)alloc16((size_t)sc->n * sizeof(float));
    int b, h, i, had_hop = 0, prev_solid = 0;

    if (!f || !r) { g_fail++; return; }
    for (i = 0; i < sc->n; ++i) {
        mic_f[i] = (float)sc->mic[i] / 32768.0f;
        far_f[i] = (float)sc->far[i] / 32768.0f;
    }
    memset(cnt, 0, sizeof(*cnt));
    for (b = 0; b < blocks; ++b) {
        AiaecFrontOut fo;
        int rc = is_f32 ? aiaec_front_process_f32(f, mic_f + (size_t)b * hop_in, far_f + (size_t)b * hop_in, hop_in, &fo)
                        : aiaec_front_process_i16(f, sc->mic + (size_t)b * hop_in, sc->far + (size_t)b * hop_in, hop_in, &fo);
        CHECK(rc == AIAEC_OK, "%s: process rc=%d", c->name, rc);
        if (rc != AIAEC_OK) return;
        CHECK(fo.event_count == c->n_frames && fo.samples == hop_in, "%s: out shape", c->name);
        CHECK(memcmp(fo.raw_far, far_f + (size_t)b * hop_in, (size_t)hop_in * sizeof(float)) == 0,
              "%s: raw_far differs from the converted far (block %d)", c->name, b);
        if (is_f32) CHECK(fo.raw_far == far_f + (size_t)b * hop_in, "%s: float raw_far is not the caller's buffer", c->name);
        for (h = 0; h < c->n_frames; ++h) {
            AecResContext rc_ctx;
            AecLinearContext lc;
            AiaecAlignEvent want;
            size_t o = (size_t)(b * c->n_frames + h) * hop;
            aec_process_context(r, mic_f + o, far_f + o);
            aec_get_res_context(r, &rc_ctx);
            aec_get_linear_context(r, &lc);
            /* the AEC's own contract: CHANGED is a compensated hop, so solid is "not UNLOCKED" */
            want.changed = lc.delay_state == AEC_LINEAR_DELAY_CHANGED;
            want.solid = lc.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
            CHECK(memcmp(fo.formed_hop + (size_t)h * hop, rc_ctx.formed_hop, (size_t)hop * sizeof(float)) == 0,
                  "%s: residual differs at block %d hop %d", c->name, b, h);
            CHECK(events_equal(&fo.events[h], &want), "%s: event differs at block %d hop %d (%d/%d vs %d/%d)",
                  c->name, b, h, fo.events[h].changed, fo.events[h].solid, want.changed, want.solid);
            /* the consumers' edge rule: one edge per delay change, never a second one on the LOCKED hop after it */
            cnt->edges += fo.events[h].changed || (had_hop && !prev_solid && fo.events[h].solid);
            had_hop = 1;
            prev_solid = fo.events[h].solid;
            cnt->changed += want.changed;
            cnt->solid += want.solid;
            cnt->hops++;
        }
        if (keep_buf && b < keep_blocks) memcpy(keep_buf + (size_t)b * hop_in, fo.formed_hop, (size_t)hop_in * sizeof(float));
        if (g_fail) return;
    }
    /* Reset replay: the same blocks again must give the same residual and events. */
    if (keep_buf) {
        aiaec_front_reset(f);
        for (b = 0; b < keep_blocks; ++b) {
            AiaecFrontOut fo;
            int rc = aiaec_front_process_f32(f, mic_f + (size_t)b * hop_in, far_f + (size_t)b * hop_in, hop_in, &fo);
            CHECK(rc == AIAEC_OK, "%s: replay rc=%d", c->name, rc);
            if (rc != AIAEC_OK) return;
            CHECK(memcmp(fo.formed_hop, keep_buf + (size_t)b * hop_in, (size_t)hop_in * sizeof(float)) == 0,
                  "%s: reset replay differs at block %d", c->name, b);
        }
    }
    free(mic_f); free(far_f); free(fpool); free(rpool);
}

int main(void) {
    static const Case cases[] = {
        { "16k/256 N=1 matched",   16000, 256, 1, 0, 5, -1, 480 },
        { "16k/256 N=1 matched-1", 16000, 256, 1, 0, 1, -1, 480 },
        { "16k/256 N=2 matched",   16000, 256, 2, 0, 5, -1, 480 },
        { "16k/512 N=1 matched",   16000, 512, 1, 0, 5, -1, 480 },
        { "48k/1024 N=1 matched",  48000, 1024, 1, 0, 5, -1, 1440 },
        { "16k/256 N=1 fixed",     16000, 256, 1, 1, 5, 480, 480 },
        { "16k/256 N=1 external",  16000, 256, 1, 2, 5, -1, 0 },
    };
    size_t k;
    for (k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
        const Case *c = &cases[k];
        const int hop_in = c->fft / 2 * c->n_frames;
        const int hops_total = c->sr == 48000 ? 300 : 500;
        Scene sc = scene_make(hops_total * c->fft / 2, c->delay);
        float *keep = (float *)calloc((size_t)hop_in * 40, sizeof(float));
        Counts ci, cf;
        int errs = g_fail;
        run_case(c, &sc, 0, &ci, keep, 40);
        run_case(c, &sc, 1, &cf, NULL, 0);
        CHECK(ci.changed == cf.changed && ci.solid == cf.solid, "%s: int16 and float entries disagree", c->name);
        if (c->mode == 0) CHECK(ci.changed >= 1 && ci.solid >= 1, "%s: delay policy never moved (changed=%d solid=%d)", c->name, ci.changed, ci.solid);
        if (c->mode == 0) CHECK(ci.edges == ci.changed, "%s: %d edges for %d CHANGED hops", c->name, ci.edges, ci.changed);
        if (c->mode == 1) CHECK(ci.changed == 0 && ci.solid >= 1, "%s: fixed mode changed=%d solid=%d", c->name, ci.changed, ci.solid);
        if (c->mode == 1) CHECK(ci.edges == 1, "%s: fixed mode has %d edges (UNLOCKED -> LOCKED once)", c->name, ci.edges);
        if (c->mode == 2) CHECK(ci.changed == 0 && ci.solid == ci.hops, "%s: external mode changed=%d solid=%d of %d", c->name, ci.changed, ci.solid, ci.hops);
        if (c->mode == 2) CHECK(ci.edges == 0, "%s: external mode has %d edges", c->name, ci.edges);
        printf("%-24s hops=%d changed=%d solid=%d edges=%d %s\n", c->name, ci.hops, ci.changed, ci.solid, ci.edges, g_fail == errs ? "PASS" : "FAIL");
        free(keep);
        scene_free(&sc);
    }
    if (g_fail) { fprintf(stderr, "test_front_scene: %d failure(s)\n", g_fail); return 1; }
    printf("test_front_scene: PASS\n");
    return 0;
}
