/* ainr.c bound with -DAINR_PRODUCT_MODEL_ID resolves {sr, fft, hop_in} against that model's grids (-DAINR_TEST_PRODUCT=1 own
 * variant: grids, N 1..64, N = 65 and foreign grids refused; 2 other variant's id: AINR_ERR_MODEL_ID; 3 mask_lookahead 0
 * model, full only). A library-accepted N without a graph_io row is not an error (the rows are the sample's). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr.h"
#include "ainr_product.h"
#include "ainr_test_fixture.h"
#include "dfn2_grids.h"
#include "graph_io_dfn2.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

#if AINR_TEST_PRODUCT == 1 || AINR_TEST_PRODUCT == 3
/* Init of cfg over tables of its grid with a fresh pool; the info is returned, the context and pool are released. */
static int init_info(const AinrConfig *cfg, AinrInfo *info) {
    StreamMemReq req;
    void *pool = NULL;
    AinrCtx *ctx = NULL;
    ErbTables erb;
    const DFN2GridSpec *grid = dfn2_grids_find(AINR_PRODUCT_MODEL_ID, cfg->sr, cfg->fft);
    int rc = ainr_get_mem_size(cfg, &req);
    if (rc != AINR_OK) return rc;
    CHECK(grid != NULL, "%d Hz / %d: no grid", cfg->sr, cfg->fft);
    if (!grid) return AINR_ERR_GRID;
    erb = erb_tables_synthetic(cfg->fft / 2 + 1, grid->n_erb);
    if (posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(1);
    rc = ainr_init_t(cfg, &erb, pool, (size_t)req.bytes, &ctx, info);
    CHECK((rc == AINR_OK) == (ctx != NULL), "init rc=%d with ctx %p", rc, (void *)ctx);
    free(pool);
    erb_tables_free(&erb);
    return rc;
}

typedef struct GridWant { int sr, fft, mask_lookahead, queue_frames; } GridWant;

/* Every N in 1..64 at the grid: sized and, at the N of the sample table's rows, initialised with the grid's info. */
static void check_grid(const GridWant *g, int row_model) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    AinrInfo bi;
    int n, rows = 0, rc;
    cfg.sr = g->sr;
    cfg.fft = g->fft;
    for (n = 1; n <= 64; ++n) {
        cfg.hop_in = n * (g->fft / 2);
        rc = ainr_get_mem_size(&cfg, &req);
        CHECK(rc == AINR_OK, "%d Hz N %d: get_mem_size rc=%d (%s)", g->sr, n, rc, ainr_status_text(rc));
        if (graph_io_dfn2_lookup(row_model, g->sr, g->fft, n, g->mask_lookahead)) {
            ++rows;
            memset(&bi, 0x5a, sizeof(bi));
            CHECK(init_info(&cfg, &bi) == AINR_OK, "%d Hz N %d init", g->sr, n);
            CHECK(bi.mask_lookahead == g->mask_lookahead && bi.queue_frames == g->queue_frames,
                  "%d Hz N %d: mask_lookahead %d queue_frames %d, want %d / %d", g->sr, n, bi.mask_lookahead,
                  bi.queue_frames, g->mask_lookahead, g->queue_frames);
        }
    }
    CHECK(rows > 0, "%d Hz: the graph_io table has no row at mask_lookahead %d", g->sr, g->mask_lookahead);
    cfg.hop_in = 65 * (g->fft / 2);
    rc = ainr_get_mem_size(&cfg, &req);
    CHECK(rc == AINR_ERR_FRAMES_PER_CALL, "%d Hz N 65: rc=%d, want AINR_ERR_FRAMES_PER_CALL", g->sr, rc);
}
#endif

/* A refused init: the code, ctx NULL, info byte-for-byte as it was. */
static void expect_refused(const AinrConfig *cfg, int want, const char *what) {
    _Alignas(16) static unsigned char pool[64];
    AinrInfo info, before;
    AinrCtx *ctx = (AinrCtx *)pool;   /* must be cleared */
    int rc;
    memset(&info, 0x5a, sizeof(info));
    before = info;
    rc = ainr_init(cfg, pool, sizeof(pool), &ctx, &info);
    CHECK(rc == want, "%s: init rc=%d (%s), want %d", what, rc, ainr_status_text(rc), want);
    CHECK(ctx == NULL, "%s: ctx not NULL after refusal", what);
    CHECK(memcmp(&info, &before, sizeof(info)) == 0, "%s: info written on refusal", what);
}

int main(void) {
    AinrConfig n1 = {0};
    StreamMemReq req;
    int rc;
#ifdef DFN2_MASK_ONLY
    n1.sr = 48000; n1.fft = 1024; n1.hop_in = 512;
#else
    n1.sr = 16000; n1.fft = 256; n1.hop_in = 128;
#endif
    rc = ainr_get_mem_size(&n1, &req);

#if AINR_TEST_PRODUCT == 1
    {
#ifdef DFN2_MASK_ONLY
        static const GridWant want[] = { { 48000, 1024, 1, 1 } };
        static const int other_sr[] = { 16000, 32000 }, other_fft[] = { 256, 512 };
#else
        static const GridWant want[] = { { 16000, 256, 0, 0 }, { 48000, 1024, 1, 2 } };
        static const int other_sr[] = { 32000, 44100 }, other_fft[] = { 512, 256 };
#endif
        size_t i;
        CHECK(rc == AINR_OK, "get_mem_size rc=%d (%s)", rc, ainr_status_text(rc));
        for (i = 0; i < sizeof want / sizeof want[0]; ++i) check_grid(&want[i], GRAPH_IO_DFN2_PRODUCT);
        for (i = 0; i < sizeof other_sr / sizeof other_sr[0]; ++i) {
            AinrConfig c = {0};
            c.sr = other_sr[i]; c.fft = other_fft[i]; c.hop_in = other_fft[i] / 2;
            expect_refused(&c, AINR_ERR_GRID, "a grid of another model");
        }
    }
#elif AINR_TEST_PRODUCT == 3
    {
        static const GridWant want[] = { { 16000, 256, 0, 0 }, { 32000, 512, 0, 0 }, { 48000, 1024, 0, 0 } };
        size_t i;
        CHECK(rc == AINR_OK, "L0 16 kHz N 1 get_mem_size rc=%d (%s)", rc, ainr_status_text(rc));
        for (i = 0; i < sizeof want / sizeof want[0]; ++i) check_grid(&want[i], GRAPH_IO_DFN2_FULL);
        {
            AinrConfig c = {0};
            c.sr = 44100; c.fft = 256; c.hop_in = 128;
            expect_refused(&c, AINR_ERR_GRID, "44.1 kHz");
        }
    }
#else
    CHECK(rc == AINR_ERR_MODEL_ID, "other-variant id: get_mem_size rc=%d (%s), want AINR_ERR_MODEL_ID", rc, ainr_status_text(rc));
    expect_refused(&n1, AINR_ERR_MODEL_ID, "other-variant id");
    {
        StreamMemReq freq;
        CHECK(ainr_get_mem_size_freq(&n1, &freq) == AINR_ERR_MODEL_ID, "other-variant id: get_mem_size_freq not AINR_ERR_MODEL_ID");
    }
#endif
    if (g_fail) { printf("test_ainr_product[%d]: FAIL\n", AINR_TEST_PRODUCT); return 1; }
    printf("test_ainr_product[%d]: PASS (AINR_PRODUCT_MODEL_ID=%d)\n", AINR_TEST_PRODUCT, (int)AINR_PRODUCT_MODEL_ID);
    return 0;
}
