/* The catalog through the public API, per product: the five delivered grid/N rows construct and have a graph_io row, other N
 * in 1..64 construct with GRAPH_IO_ERR_NO_ROW, and a bad grid, hop_in or N = 65 is refused with its status in that order.
 * Row-by-row work views against the graph rows: test_aiaec_graph_io.c. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"
#include "graph_io.h"
#include "graph_io_alignulcnet.h"
#include "graph_io_deepvqes.h"

#define PRODUCT_DV (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S)
#if PRODUCT_DV
#define FIND(sr, fft, n, mc) graph_io_deepvqes_find(sr, fft, n, mc)
#else
#define FIND(sr, fft, n, mc) graph_io_alignulcnet_find(sr, fft, n, mc)
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static AiaecConfig cfg_of(int sr, int fft, int hop_in) {
    AiaecConfig c;
    c.sr = sr; c.fft = fft; c.hop_in = hop_in;
    return c;
}

/* One config refused with `want` by every entry that validates, with nothing allocated or written. */
static void expect_refused(const AiaecConfig *cfg, int want, const char *what) {
    StreamMemReq req;
    AiaecCtx *ctx;
    AiaecFreqCtx *cf;
    unsigned char pool[256];
    int rc;

    memset(&req, 0xA5, sizeof(req));
    rc = aiaec_get_mem_size(cfg, &req);
    CHECK(rc == want, "%s: get_mem_size rc=%d (%s), want %d", what, rc, aiaec_status_text(rc), want);
    CHECK(req.bytes == 0xA5A5A5A5A5A5A5A5ull, "%s: req touched on a refusal", what);
    ctx = (AiaecCtx *)pool;
    rc = aiaec_init(cfg, pool, sizeof(pool), &ctx);
    CHECK(rc == want && ctx == NULL, "%s: init rc=%d (%s)", what, rc, aiaec_status_text(rc));
    cf = (AiaecFreqCtx *)pool;
    CHECK(aiaec_init_freq(cfg, pool, sizeof(pool), &cf) == want && cf == NULL, "%s: init_freq", what);
}

static void section_refusal_order(void) {
    AiaecConfig c;

    c = cfg_of(32000, 1024, 512);   expect_refused(&c, AIAEC_ERR_GRID, "no (sr,fft) row");
    c = cfg_of(48000, 512, 256);    expect_refused(&c, AIAEC_ERR_GRID, "48k with fft 512 not registered");
    c = cfg_of(16000, 1024, 512);   expect_refused(&c, AIAEC_ERR_GRID, "fft not registered");
    c = cfg_of(16000, 256, 129);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 129 at 16k/256");
    c = cfg_of(16000, 256, 100);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 100");
    c = cfg_of(16000, 256, 0);      expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 0");
    c = cfg_of(16000, 256, -128);   expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in negative");
    c = cfg_of(16000, 512, 255);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 255 at 16k/512");
    /* grid precedes hop_in: an unregistered grid with a bad hop_in is a grid error */
    c = cfg_of(32000, 1024, 129);   expect_refused(&c, AIAEC_ERR_GRID, "grid before hop_in");
    c = cfg_of(48000, 1024, 513);   expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 513 at 48k/1024");
    /* N beyond the model stream's range (64) on every registered grid */
    c = cfg_of(16000, 256, 65 * 128);   expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/256 N=65");
    c = cfg_of(16000, 512, 65 * 256);   expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/512 N=65");
    c = cfg_of(48000, 1024, 65 * 512);  expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "48k/1024 N=65");
    /* N is checked after hop_in: a non-multiple beyond the range is an IN_HOP error */
    c = cfg_of(16000, 256, 65 * 128 + 1);   expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in beyond the range, not a multiple");
    printf("section_refusal_order: done\n");
}

/* One registered row: constructs, has a graph_io row, and its first pre call carries the row's work. */
static void expect_row(int sr, int fft, int hop_in) {
    AiaecConfig cfg = cfg_of(sr, fft, hop_in);
    const unsigned bins = (unsigned)(fft / 2 + 1), nf = (unsigned)(hop_in / (fft / 2));
    StreamMemReq req, req_f;
    AiaecCtx *ctx = NULL;
    AiaecFreqCtx *cf = NULL;
    void *pool = NULL, *pool_f = NULL;
    int rc;

    rc = aiaec_get_mem_size(&cfg, &req);
    CHECK(rc == AIAEC_OK, "row %d/%d: get_mem_size rc=%d (%s)", fft, hop_in, rc, aiaec_status_text(rc));
    if (rc != AIAEC_OK) return;
    CHECK(posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) == 0, "row %d/%d: pool", fft, hop_in);
    rc = aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx);
    CHECK(rc == AIAEC_OK && ctx != NULL, "row %d/%d: init rc=%d (%s)", fft, hop_in, rc, aiaec_status_text(rc));
    if (ctx) {
#if PRODUCT_DV
        static const int16_t zeros16[1024];
#else
        static const float zeros_f[1024];
        AiaecAlignEvent no_event[8];
#endif
        AiaecInput in;
        StreamWork work;

        {
            GraphIoConfig mc;
            int frc = FIND(sr, fft, (int)nf, &mc);
            CHECK(frc == GRAPH_IO_OK, "row %d/%d: graph_io rc %d", fft, hop_in, frc);
        }
        CHECK(aiaec_state_generation(ctx) == 0u, "row %d/%d: generation after init", fft, hop_in);
        memset(&in, 0, sizeof(in));
        in.samples = hop_in;
#if PRODUCT_DV
        in.kind = AIAEC_INPUT_DEEPVQE_S;
        in.u.deepvqe_s.mic_in = zeros16; in.u.deepvqe_s.far_in = zeros16;
#else
        memset(no_event, 0, sizeof(no_event));
        in.kind = AIAEC_INPUT_ALIGN_ULCNET;
        in.u.align_ulcnet.residual_in = zeros_f; in.u.align_ulcnet.far_in = zeros_f;
        in.u.align_ulcnet.events = no_event; in.u.align_ulcnet.event_count = (int)nf;
#endif
        CHECK(aiaec_preprocess(ctx, &in, &work) == AIAEC_OK && work.action == STREAM_WORK_SUBMIT &&
              work.output_count >= 1 && work.inputs[0].w == bins,
              "row %d/%d: work (%d in, %d out)", fft, hop_in, work.input_count, work.output_count);
#if PRODUCT_DV
        /* mic, far [1,2,N,bins]; CCM taps [1,N,bins,18] */
        CHECK(work.input_count == 2 && work.output_count == 1 &&
              work.inputs[0].b == 1u && work.inputs[0].c == 2u && work.inputs[0].h == nf &&
              work.inputs[0].elements == 2u * nf * bins && work.inputs[1].w == bins && work.inputs[1].elements == 2u * nf * bins &&
              work.outputs[0].b == 1u && work.outputs[0].c == nf && work.outputs[0].h == bins && work.outputs[0].w == 18u &&
              work.outputs[0].elements == 18u * nf * bins,
              "row %d/%d: DeepVQE-S work views", fft, hop_in);
#else
        /* four signals [1,1,N,bins], frame_reset [1,1,1,N] last at N > 1; the mask [1,2,N,bins] */
        CHECK(work.input_count == (nf > 1u ? 5 : 4) && work.output_count == 1 &&
              work.inputs[0].h == nf && work.inputs[0].elements == nf * bins &&
              work.outputs[0].c == 2u && work.outputs[0].h == nf && work.outputs[0].elements == 2u * nf * bins &&
              (nf == 1u || (work.inputs[4].h == 1u && work.inputs[4].w == nf)),
              "row %d/%d: Align-ULCNet work views", fft, hop_in);
#endif
        aiaec_reset(ctx);   /* closes the open block and restarts */
        CHECK(aiaec_state_generation(ctx) != 0u, "row %d/%d: a reset advances the generation", fft, hop_in);
    }
    CHECK(aiaec_get_mem_size_freq(&cfg, &req_f) == AIAEC_OK &&
          posix_memalign(&pool_f, (size_t)req_f.alignment, (size_t)req_f.bytes) == 0, "row %d/%d: freq pool", fft, hop_in);
    CHECK(aiaec_init_freq(&cfg, pool_f, (size_t)req_f.bytes, &cf) == AIAEC_OK && cf != NULL, "row %d/%d: the _freq family constructs", fft, hop_in);
    CHECK(aiaec_init(NULL, pool, (size_t)req.bytes, &ctx) == AIAEC_ERR_NULL && ctx == NULL, "row %d/%d: NULL cfg", fft, hop_in);
    CHECK(aiaec_init(&cfg, NULL, (size_t)req.bytes, &ctx) == AIAEC_ERR_NULL, "row %d/%d: NULL pool", fft, hop_in);
    CHECK(aiaec_init(&cfg, pool, (size_t)req.bytes, NULL) == AIAEC_ERR_NULL, "row %d/%d: NULL ctx", fft, hop_in);
    free(pool);
    free(pool_f);
}

static void section_delivered_rows(void) {
    expect_row(16000, 256, 128);
    expect_row(16000, 512, 256);
    expect_row(16000, 512, 1024);
    expect_row(48000, 1024, 512);
    expect_row(48000, 1024, 1024);
    printf("section_delivered_rows: %s 16k/256 N=1, 16k/512 N=1 and N=4, 48k/1024 N=1 and N=2 construct and have a graph_io row\n",
           PRODUCT_DV ? "DeepVQE-S" : "Align-ULCNet");
}

/* An N inside the model's range with no graph_io row: the library constructs both families, graph_io finds no row. */
static void expect_no_row(int sr, int fft, int n) {
    AiaecConfig cfg = cfg_of(sr, fft, n * (fft / 2));
    StreamMemReq req, req_f;
    GraphIoConfig mc;
    AiaecCtx *ctx = NULL;
    AiaecFreqCtx *cf = NULL;
    void *pool = NULL, *pool_f = NULL;
    int rc = aiaec_get_mem_size(&cfg, &req);

    CHECK(rc == AIAEC_OK, "%d/%d N=%d: get_mem_size rc=%d (%s)", sr, fft, n, rc, aiaec_status_text(rc));
    if (rc == AIAEC_OK) {
        CHECK(posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) == 0, "%d/%d N=%d: pool", sr, fft, n);
        rc = aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx);
        CHECK(rc == AIAEC_OK && ctx != NULL, "%d/%d N=%d: init rc=%d (%s)", sr, fft, n, rc, aiaec_status_text(rc));
    }
    rc = aiaec_get_mem_size_freq(&cfg, &req_f);
    if (rc == AIAEC_OK && posix_memalign(&pool_f, (size_t)req_f.alignment, (size_t)req_f.bytes) == 0)
        CHECK(aiaec_init_freq(&cfg, pool_f, (size_t)req_f.bytes, &cf) == AIAEC_OK && cf != NULL, "%d/%d N=%d: init_freq", sr, fft, n);
    else
        CHECK(0, "%d/%d N=%d: freq pool rc=%d", sr, fft, n, rc);
    rc = FIND(sr, fft, n, &mc);
    CHECK(rc == GRAPH_IO_ERR_NO_ROW, "%d/%d N=%d: graph_io rc=%d, want GRAPH_IO_ERR_NO_ROW", sr, fft, n, rc);
    free(pool);
    free(pool_f);
}

static void section_rows_without_graph(void) {
    expect_no_row(16000, 256, 2);
    expect_no_row(16000, 256, 8);
    expect_no_row(16000, 256, 64);
    expect_no_row(16000, 512, 2);
    expect_no_row(16000, 512, 8);
    expect_no_row(48000, 1024, 4);
    expect_no_row(48000, 1024, 8);
    printf("section_rows_without_graph: N in the model's range constructs; graph_io has no row for it\n");
}

int main(void) {
    section_refusal_order();
    section_delivered_rows();
    section_rows_without_graph();
    printf("test_aiaec_catalog[%s]: %s\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
