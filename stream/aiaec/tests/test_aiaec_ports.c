/* The AiaecWork signal and head views of real pre calls, for every registered 16k grid / N of the product
 * model, in the PCM, _spec and _freq families. Built once per product with -DAIAEC_TEST_CATALOG so every row
 * has a graph path.
 *
 * The expected shapes are literals taken from the exporters' graphs (ONNX [b,c,h,w], 129 bins at FFT 256, 257
 * at FFT 512, 18 CCM taps), not recomputed from library constants:
 *   DeepVQE-S:    mic, far [1,2,N,bins]; CCM taps [1,N,bins,18].
 *   Align-ULCNet: error_mag, far_mag, error_cos, error_sin [1,1,N,bins]; at N > 1 frame_reset [1,1,1,N] as the
 *                 fifth and last signal; complex mask [1,2,N,bins].
 * The recurrent state is not a work view. An Align-ULCNet N = 1 event hop is a BYPASS block (zero counts); at N = 8
 * the event hop goes into frame_reset (00100000 for an event at hop 2). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"

#define PRODUCT_DV (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S)

/* AiaecInfo is the init's slim result (graph, trim, queue, tail): no port list can sit inside it. */
typedef char info_stays_slim[sizeof(AiaecInfo) <= sizeof(const char *) + 4 * sizeof(int) ? 1 : -1];

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Shape { unsigned b, c, h, w; } Shape;

typedef struct Expect {
    int fft, n;
    int n_in;
    Shape in[5];
    Shape out;
    int trim, queue;
} Expect;

#if PRODUCT_DV
static const Expect EXPECT[] = {
    { 256, 1, 2, { { 1, 2, 1, 129 }, { 1, 2, 1, 129 } }, { 1, 1, 129, 18 }, 128, 0 },
    { 256, 8, 2, { { 1, 2, 8, 129 }, { 1, 2, 8, 129 } }, { 1, 8, 129, 18 }, 1024, 7 },
    { 512, 1, 2, { { 1, 2, 1, 257 }, { 1, 2, 1, 257 } }, { 1, 1, 257, 18 }, 256, 0 },
};
#else
static const Expect EXPECT[] = {
    { 256, 1, 4, { { 1, 1, 1, 129 }, { 1, 1, 1, 129 }, { 1, 1, 1, 129 }, { 1, 1, 1, 129 } }, { 1, 2, 1, 129 }, 128, 0 },
    { 256, 8, 5, { { 1, 1, 8, 129 }, { 1, 1, 8, 129 }, { 1, 1, 8, 129 }, { 1, 1, 8, 129 }, { 1, 1, 1, 8 } },
      { 1, 2, 8, 129 }, 128, 0 },
    { 512, 1, 4, { { 1, 1, 1, 257 }, { 1, 1, 1, 257 }, { 1, 1, 1, 257 }, { 1, 1, 1, 257 } }, { 1, 2, 1, 257 }, 256, 0 },
};
#endif
#define N_EXPECT ((int)(sizeof(EXPECT) / sizeof(EXPECT[0])))

enum { FAM_PCM = 0, FAM_SPEC = 1, FAM_FREQ = 2, FAMS = 3 };
static const char *FAM_NAME[FAMS] = { "pcm", "spec", "freq" };

#define MAX_PLANE (8 * 257)
#if PRODUCT_DV
static int16_t g_pcm[2048];
#else
static float g_pcm_f[2048];
#endif
static float g_plane[MAX_PLANE];
static AiaecAlignEvent g_ev[8];

static size_t shape_elements(const Shape *s) { return (size_t)s->b * s->c * s->h * s->w; }

static void check_in(const AiaecTensorIn *t, const Shape *s, const char *what) {
    CHECK(t->data != NULL && t->b == s->b && t->c == s->c && t->h == s->h && t->w == s->w && t->elements == shape_elements(s),
          "%s: [%u,%u,%u,%u] (%zu) want [%u,%u,%u,%u]", what, t->b, t->c, t->h, t->w, t->elements, s->b, s->c, s->h, s->w);
}

static int disjoint(const void *a, size_t na, const void *b, size_t nb) {
    const unsigned char *pa = (const unsigned char *)a, *pb = (const unsigned char *)b;
    return pa + na * sizeof(float) <= pb || pb + nb * sizeof(float) <= pa;
}

/* Views of one SUBMIT block against the table row: shapes, non-NULL and pairwise-disjoint buffers. */
static void check_views(const AiaecWork *w, const Expect *e, const char *what) {
    char lab[96];
    int i, j;

    CHECK(w->action == AIAEC_SUBMIT && w->input_count == e->n_in && w->output_count == 1,
          "%s: action %d counts %d/%d want SUBMIT %d/1", what, (int)w->action, w->input_count, w->output_count, e->n_in);
    for (i = 0; i < e->n_in && i < 5; ++i) {
        snprintf(lab, sizeof(lab), "%s: inputs[%d]", what, i);
        check_in(&w->inputs[i], &e->in[i], lab);
    }
    for (i = e->n_in; i < 5; ++i)
        CHECK(w->inputs[i].data == NULL && w->inputs[i].elements == 0u && w->inputs[i].w == 0u,
              "%s: inputs[%d] beyond the %d signals is not empty", what, i, e->n_in);
    CHECK(w->outputs[0].data != NULL && w->outputs[0].b == e->out.b && w->outputs[0].c == e->out.c && w->outputs[0].h == e->out.h &&
          w->outputs[0].w == e->out.w && w->outputs[0].elements == shape_elements(&e->out),
          "%s: outputs[0] [%u,%u,%u,%u] (%zu) want [%u,%u,%u,%u]", what, w->outputs[0].b, w->outputs[0].c, w->outputs[0].h,
          w->outputs[0].w, w->outputs[0].elements, e->out.b, e->out.c, e->out.h, e->out.w);
    for (i = 0; i < e->n_in; ++i) {
        for (j = i + 1; j < e->n_in; ++j)
            CHECK(disjoint(w->inputs[i].data, w->inputs[i].elements, w->inputs[j].data, w->inputs[j].elements),
                  "%s: inputs[%d] and inputs[%d] overlap", what, i, j);
        CHECK(disjoint(w->inputs[i].data, w->inputs[i].elements, w->outputs[0].data, w->outputs[0].elements),
              "%s: inputs[%d] overlaps outputs[0]", what, i);
    }
}

static void input_pcm(AiaecInput *in, int n, int hop, const AiaecAlignEvent *ev) {
    memset(in, 0, sizeof(*in));
    in->samples = n * hop;
#if PRODUCT_DV
    in->kind = AIAEC_INPUT_DEEPVQE_S;
    in->u.deepvqe_s.mic_in = g_pcm; in->u.deepvqe_s.far_in = g_pcm;
    (void)ev;
#else
    in->kind = AIAEC_INPUT_ALIGN_ULCNET;
    in->u.align_ulcnet.residual_in = g_pcm_f; in->u.align_ulcnet.far_in = g_pcm_f;
    in->u.align_ulcnet.events = ev; in->u.align_ulcnet.event_count = n;
#endif
}

static void input_freq(AiaecFreqInput *in, int n, const AiaecAlignEvent *ev) {
    memset(in, 0, sizeof(*in));
    in->frames = n;
#if PRODUCT_DV
    in->kind = AIAEC_INPUT_DEEPVQE_S;
    in->u.deepvqe_s.mic_re = in->u.deepvqe_s.mic_im = in->u.deepvqe_s.far_re = in->u.deepvqe_s.far_im = g_plane;
    (void)ev;
#else
    in->kind = AIAEC_INPUT_ALIGN_ULCNET;
    in->u.align_ulcnet.err_re = in->u.align_ulcnet.err_im = in->u.align_ulcnet.far_re = in->u.align_ulcnet.far_im = g_plane;
    in->u.align_ulcnet.events = ev; in->u.align_ulcnet.event_count = n;
#endif
}

static int same_info(const AiaecInfo *a, const AiaecInfo *b) {
    return a->graph_path == b->graph_path && a->trim_samples == b->trim_samples &&
           a->queue_frames == b->queue_frames && a->tail_zero_blocks == b->tail_zero_blocks;
}

typedef struct Ctxs {
    AiaecCtx *pcm;
    AiaecSpecCtx *spec;
    AiaecFreqCtx *freq;
    void *pool[FAMS];
    AiaecInfo info[FAMS];
} Ctxs;

static int ctxs_make(Ctxs *c, const AiaecConfig *cfg) {
    StreamMemReq r[FAMS];
    int rc, f;

    memset(c, 0, sizeof(*c));
    rc = aiaec_get_mem_size(cfg, &r[0]);
    if (rc != AIAEC_OK) return rc;
    if (aiaec_get_mem_size_spec(cfg, &r[1]) != AIAEC_OK || aiaec_get_mem_size_freq(cfg, &r[2]) != AIAEC_OK) return -99;
    for (f = 0; f < FAMS; ++f)
        if (posix_memalign(&c->pool[f], (size_t)r[f].alignment, (size_t)r[f].bytes) != 0) return -99;
    rc = aiaec_init(cfg, c->pool[0], (size_t)r[0].bytes, &c->pcm, &c->info[0]);
    if (rc != AIAEC_OK) return rc;
    rc = aiaec_init_spec(cfg, c->pool[1], (size_t)r[1].bytes, &c->spec, &c->info[1]);
    if (rc != AIAEC_OK) return rc;
    return aiaec_init_freq(cfg, c->pool[2], (size_t)r[2].bytes, &c->freq, &c->info[2]);
}

static void ctxs_free(Ctxs *c) {
    int f;
    for (f = 0; f < FAMS; ++f) free(c->pool[f]);
}

static void close_block(Ctxs *c, int fam, const AiaecWork *w) {
    static int16_t pcm[2048];
    static float re[MAX_PLANE], im[MAX_PLANE];
    size_t i;
    if (w->action == AIAEC_SUBMIT)
        for (i = 0; i < w->outputs[0].elements; ++i) w->outputs[0].data[i] = 0.0f;
    if (fam == FAM_PCM) CHECK(aiaec_postprocess(c->pcm, 0, pcm) == AIAEC_OK, "postprocess pcm");
    else if (fam == FAM_SPEC) CHECK(aiaec_postprocess_spec(c->spec, 0, re, im) == AIAEC_OK, "postprocess spec");
    else CHECK(aiaec_postprocess_freq(c->freq, 0, re, im) == AIAEC_OK, "postprocess freq");
}

static int pre_family(Ctxs *c, int fam, int n, int hop, const AiaecAlignEvent *ev, AiaecWork *w) {
    AiaecInput in;
    AiaecFreqInput fin;
    if (fam == FAM_FREQ) {
        input_freq(&fin, n, ev);
        return aiaec_preprocess_freq(c->freq, &fin, w);
    }
    input_pcm(&in, n, hop, ev);
    return fam == FAM_PCM ? aiaec_preprocess(c->pcm, &in, w) : aiaec_preprocess_spec(c->spec, &in, w);
}

/* One table row: info, and the first block's views in every family. */
static void run_row(const Expect *e) {
    char label[64];
    AiaecConfig cfg;
    Ctxs c;
    int fam, rc, hop = e->fft / 2;

    snprintf(label, sizeof(label), "%s 16k/%d N=%d", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", e->fft, e->n);
    cfg.sr = 16000; cfg.fft = e->fft; cfg.hop_in = e->n * hop;
    rc = ctxs_make(&c, &cfg);
    CHECK(rc == AIAEC_OK, "%s: init rc=%d (%s)", label, rc, aiaec_status_text(rc));
    if (rc != AIAEC_OK) { ctxs_free(&c); return; }

    CHECK(c.info[0].graph_path != NULL && same_info(&c.info[0], &c.info[1]) && same_info(&c.info[0], &c.info[2]),
          "%s: the three families report one and the same info", label);
    CHECK(c.info[0].trim_samples == e->trim && c.info[0].queue_frames == e->queue && c.info[0].tail_zero_blocks == 1,
          "%s: trim %d queue %d tail %d", label, c.info[0].trim_samples, c.info[0].queue_frames, c.info[0].tail_zero_blocks);

    for (fam = 0; fam < FAMS; ++fam) {
        AiaecWork w;
        char what[96];
        memset(&w, 0x5a, sizeof(w));
        rc = pre_family(&c, fam, e->n, hop, g_ev, &w);
        snprintf(what, sizeof(what), "%s [%s]", label, FAM_NAME[fam]);
        CHECK(rc == AIAEC_OK, "%s: preprocess rc=%d", what, rc);
        if (rc != AIAEC_OK) continue;
        check_views(&w, e, what);
        close_block(&c, fam, &w);
    }
    ctxs_free(&c);
}

#if !PRODUCT_DV
/* Align-ULCNet events: N = 1 event hop is a BYPASS block with nothing to bind; at N = 8 an event at hop 2 is one SUBMIT
 * block whose fifth signal reads 00100000. Every family. */
static void run_events(void) {
    static AiaecAlignEvent ev1[1], ev8[8];
    static const float want[8] = { 0, 0, 1, 0, 0, 0, 0, 0 };
    AiaecConfig cfg;
    Ctxs c;
    int fam, rc;

    ev1[0].changed = 1;
    cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = 128;
    rc = ctxs_make(&c, &cfg);
    CHECK(rc == AIAEC_OK, "events N=1: init rc=%d", rc);
    if (rc == AIAEC_OK)
        for (fam = 0; fam < FAMS; ++fam) {
            AiaecWork w;
            memset(&w, 0x5a, sizeof(w));
            rc = pre_family(&c, fam, 1, 128, ev1, &w);
            CHECK(rc == AIAEC_OK && w.action == AIAEC_BYPASS && w.input_count == 0 && w.output_count == 0 &&
                  w.inputs[0].data == NULL && w.outputs[0].data == NULL && w.outputs[0].elements == 0u && w.state_generation == 0u,
                  "events N=1 [%s]: an event hop is a BYPASS block with no views (rc=%d action=%d)", FAM_NAME[fam], rc, (int)w.action);
            close_block(&c, fam, &w);
        }
    ctxs_free(&c);

    ev8[2].changed = 1;
    cfg.hop_in = 1024;
    rc = ctxs_make(&c, &cfg);
    CHECK(rc == AIAEC_OK, "events N=8: init rc=%d", rc);
    if (rc == AIAEC_OK)
        for (fam = 0; fam < FAMS; ++fam) {
            AiaecWork w;
            memset(&w, 0x5a, sizeof(w));
            rc = pre_family(&c, fam, 8, 128, ev8, &w);
            CHECK(rc == AIAEC_OK && w.action == AIAEC_SUBMIT && w.input_count == 5 && w.inputs[4].data != NULL &&
                  memcmp(w.inputs[4].data, want, sizeof(want)) == 0,
                  "events N=8 [%s]: frame_reset is not 00100000 (rc=%d)", FAM_NAME[fam], rc);
            close_block(&c, fam, &w);
        }
    ctxs_free(&c);
}
#endif

/* The customer arithmetic and refusals (no work view exists for a refused configuration). */
static void run_refusals(void) {
    AiaecConfig cfg;
    AiaecInfo info;
    unsigned char small[64];
    AiaecCtx *ctx = (AiaecCtx *)small;

    cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = 129;
    CHECK(aiaec_init(&cfg, small, sizeof(small), &ctx, &info) == AIAEC_ERR_IN_HOP && ctx == NULL, "hop_in 129 must be refused with AIAEC_ERR_IN_HOP");
    cfg.hop_in = 128; cfg.sr = 44100;
    ctx = (AiaecCtx *)small;
    CHECK(aiaec_init(&cfg, small, sizeof(small), &ctx, &info) == AIAEC_ERR_GRID && ctx == NULL, "unknown grid must be refused with AIAEC_ERR_GRID");
}

int main(void) {
    int i;

    for (i = 0; i < N_EXPECT; ++i) run_row(&EXPECT[i]);
#if !PRODUCT_DV
    run_events();
#endif
    run_refusals();

    if (g_fail) { printf("test_aiaec_ports: %d check(s) FAILED\n", g_fail); return 1; }
    printf("test_aiaec_ports[%s]: PASS\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet");
    return 0;
}
