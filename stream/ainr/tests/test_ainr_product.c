/* The product binding (ainr_product.h): ainr.c compiled with -DAINR_PRODUCT_MODEL_ID=<id> resolves
 * the same customer config {sr, fft, hop_in} to that model's own catalog rows. One source, five
 * builds selected by -DAINR_TEST_PRODUCT=1..5:
 *   1  quantization alternate 2 (16x8): graph paths differ from the default's (16x16), N=1/N=8 rows
 *      resolve (full).
 *   2  quantization alternate 1 (8x8): not registered -- every config is refused (AINR_ERR_GRID),
 *      by the sizing call and by init, which leaves its outputs untouched.
 *   3  the OTHER archive variant's id (lite id in a full archive, full id in a lite one): every
 *      config is refused with AINR_ERR_MODEL_ID.
 *   4  the mask_lookahead 0 model (id 3, full only), production catalog: 16 kHz N 1 and 8 resolve, the 32 and 48 kHz
 *      graphs (not delivered) are refused with AINR_ERR_FRAMES_PER_CALL.
 *   5  the same model on the test catalog: every row resolves (test:// paths) with queue_frames 0.
 * Not built under NO_STDIO. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_product.h"
#include "ainr_test.h"
#include "../dfn2/tests/erb_fixture.h"   /* erb_fixture_scratch_path() */

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb, i;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *inv = (float *)malloc(elems * sizeof(float));
    int32_t hdr[2];
    FILE *fp;
    if (!fwd || !inv) exit(1);
    for (i = 0; i < elems; ++i) { fwd[i] = (float)(i % 17) * 0.01f; inv[i] = 0.0f; }
    for (i = 0; i < (size_t)n_bins; ++i) {   /* a partition of unity: bin i splits 0.75 / 0.25 over two bands */
        inv[i * (size_t)n_erb + i % (size_t)n_erb] = 0.75f;
        inv[i * (size_t)n_erb + (i + 1) % (size_t)n_erb] = 0.25f;
    }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd); free(inv);
}

#if AINR_TEST_PRODUCT == 1 || AINR_TEST_PRODUCT == 5
/* Init of cfg with a fresh pool; the info is returned, the context and pool are released. */
static int init_info(const AinrConfig *cfg, const char *erb, AinrInfo *info) {
    StreamMemReq req;
    void *pool = NULL;
    AinrCtx *ctx = NULL;
    int rc = ainr_get_mem_size(cfg, &req);
    if (rc != AINR_OK) return rc;
    if (posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(1);
    rc = ainr_init_from_erb_file(cfg, erb, pool, (size_t)req.bytes, &ctx, info);
    CHECK((rc == AINR_OK) == (ctx != NULL), "init rc=%d with ctx %p", rc, (void *)ctx);
    free(pool);
    return rc;
}
#endif

#if AINR_TEST_PRODUCT != 1 && AINR_TEST_PRODUCT != 5
/* A refused init: the code, ctx NULL, info byte-for-byte as it was. */
static void expect_refused(const AinrConfig *cfg, int want, const char *what) {
    _Alignas(16) static unsigned char pool[64];
    AinrInfo info, before;
    AinrCtx *ctx = (AinrCtx *)pool;   /* must be cleared */
    int rc;
    memset(&info, 0x5a, sizeof(info));
    before = info;
    rc = ainr_init_from_erb_file(cfg, "/nonexistent/erb.bin", pool, sizeof(pool), &ctx, &info);
    CHECK(rc == want, "%s: init rc=%d (%s), want %d", what, rc, ainr_status_text(rc), want);
    CHECK(ctx == NULL, "%s: ctx not NULL after refusal", what);
    CHECK(memcmp(&info, &before, sizeof(info)) == 0, "%s: info written on refusal", what);
}
#endif

int main(void) {
    AinrConfig n1;
    StreamMemReq req;
    char erb[96];
    int rc;
#ifdef DFN2_MASK_ONLY
    n1.sr = 48000; n1.fft = 1024; n1.hop_in = 512;
#else
    n1.sr = 16000; n1.fft = 256; n1.hop_in = 128;
#endif
    rc = ainr_get_mem_size(&n1, &req);
    erb_fixture_scratch_path(erb, sizeof(erb), "ainr_product_erb");
#ifdef DFN2_MASK_ONLY
    write_synthetic_erb(erb, 513, 32);
#else
    write_synthetic_erb(erb, 129, 24);
#endif

#if AINR_TEST_PRODUCT == 1
    {
        AinrInfo bi;
        const char *path;
        CHECK(rc == AINR_OK, "alt2 get_mem_size rc=%d (%s)", rc, ainr_status_text(rc));
        CHECK(init_info(&n1, erb, &bi) == AINR_OK, "alt2 init");
        path = bi.graph_path;
        CHECK(path && strstr(path, "16x8") != NULL && strstr(path, "16x16") == NULL,
              "alt2 graph path unexpected: %s", path ? path : "(null)");
#ifndef DFN2_MASK_ONLY
        {
            AinrConfig n8 = n1;
            n8.hop_in = 8 * 128;
            CHECK(init_info(&n8, erb, &bi) == AINR_OK, "alt2 N=8 init");
            path = bi.graph_path;
            CHECK(path && strstr(path, "16x8") != NULL && strstr(path, "n8") != NULL, "alt2 N=8 path: %s", path ? path : "(null)");
        }
#endif
    }
#elif AINR_TEST_PRODUCT == 4
    {
        static const struct { int sr, fft, hop_in; const char *what; } undelivered[] = {
            { 32000, 512, 256, "32 kHz N 1" }, { 32000, 512, 1024, "32 kHz N 4" }, { 32000, 512, 512, "32 kHz N 2" },
            { 48000, 1024, 512, "48 kHz N 1" }, { 48000, 1024, 1024, "48 kHz N 2" },
        };
        AinrInfo bi;
        AinrConfig n8 = n1;
        size_t i;
        CHECK(rc == AINR_OK, "L0 16 kHz N 1 get_mem_size rc=%d (%s)", rc, ainr_status_text(rc));
        n8.hop_in = 8 * 128;
        {
            AinrCtx *ctx = NULL;
            StreamMemReq r8;
            void *pool = NULL;
            CHECK(ainr_get_mem_size(&n8, &r8) == AINR_OK, "L0 16 kHz N 8 sizing");
            if (posix_memalign(&pool, r8.alignment, (size_t)r8.bytes) != 0) exit(1);
            CHECK(ainr_init_from_erb_file(&n8, erb, pool, (size_t)r8.bytes, &ctx, &bi) == AINR_OK && ctx, "L0 16 kHz N 8 init");
            CHECK(bi.graph_path && strstr(bi.graph_path, "16x16") != NULL && strstr(bi.graph_path, "n8") != NULL,
                  "L0 16 kHz N 8 path: %s", bi.graph_path ? bi.graph_path : "(null)");
            free(pool);
        }
        for (i = 0; i < sizeof undelivered / sizeof undelivered[0]; ++i) {
            AinrConfig c;
            c.sr = undelivered[i].sr; c.fft = undelivered[i].fft; c.hop_in = undelivered[i].hop_in;
            expect_refused(&c, AINR_ERR_FRAMES_PER_CALL, undelivered[i].what);
        }
        {
            AinrConfig c = { 44100, 256, 128 };
            expect_refused(&c, AINR_ERR_GRID, "44.1 kHz");
        }
    }
#elif AINR_TEST_PRODUCT == 5
    {
        static const struct { int sr, fft, n_erb, n; const char *marker; } rows[] = {
            { 16000, 256, 24, 1, "16x16" }, { 16000, 256, 24, 8, "16x16" },
            { 32000, 512, 28, 1, "test://ainr/dfn2_l0/32k_fft512_n1" }, { 32000, 512, 28, 4, "test://ainr/dfn2_l0/32k_fft512_n4" },
            { 48000, 1024, 32, 1, "test://ainr/dfn2_l0/48k_fft1024_n1" }, { 48000, 1024, 32, 2, "test://ainr/dfn2_l0/48k_fft1024_n2" },
        };
        size_t i;
        for (i = 0; i < sizeof rows / sizeof rows[0]; ++i) {
            AinrConfig c;
            AinrInfo bi;
            char erb_row[96];
            c.sr = rows[i].sr; c.fft = rows[i].fft; c.hop_in = rows[i].n * (rows[i].fft / 2);
            erb_fixture_scratch_path(erb_row, sizeof(erb_row), "ainr_product_l0_erb");
            write_synthetic_erb(erb_row, rows[i].fft / 2 + 1, rows[i].n_erb);
            CHECK(init_info(&c, erb_row, &bi) == AINR_OK, "L0 %d Hz N %d init", rows[i].sr, rows[i].n);
            CHECK(bi.graph_path && strstr(bi.graph_path, rows[i].marker) != NULL, "L0 %d Hz N %d path: %s", rows[i].sr, rows[i].n,
                  bi.graph_path ? bi.graph_path : "(null)");
            CHECK(bi.queue_frames == 0, "L0 %d Hz N %d: queue_frames %d, want 0", rows[i].sr, rows[i].n, bi.queue_frames);
            unlink(erb_row);
        }
        CHECK(rc == AINR_OK, "L0 16 kHz N 1 get_mem_size rc=%d", rc);
    }
#elif AINR_TEST_PRODUCT == 2
    CHECK(rc == AINR_ERR_GRID, "alt1 get_mem_size rc=%d (%s), want AINR_ERR_GRID", rc, ainr_status_text(rc));
    expect_refused(&n1, AINR_ERR_GRID, "alt1");
#else
    CHECK(rc == AINR_ERR_MODEL_ID, "other-variant id: get_mem_size rc=%d (%s), want AINR_ERR_MODEL_ID", rc, ainr_status_text(rc));
    expect_refused(&n1, AINR_ERR_MODEL_ID, "other-variant id");
#endif
    unlink(erb);
    if (g_fail) { printf("test_ainr_product[%d]: FAIL\n", AINR_TEST_PRODUCT); return 1; }
    printf("test_ainr_product[%d]: PASS (AINR_PRODUCT_MODEL_ID=%d)\n", AINR_TEST_PRODUCT, (int)AINR_PRODUCT_MODEL_ID);
    return 0;
}
