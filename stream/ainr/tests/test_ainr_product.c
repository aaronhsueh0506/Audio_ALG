/* The product binding (ainr_product.h): ainr.c compiled with -DAINR_PRODUCT_MODEL_ID=<id> resolves
 * the same customer config {sr, fft, hop_in} to that model's own catalog rows. One source, three
 * builds selected by -DAINR_TEST_PRODUCT=1|2|3:
 *   1  quantization alternate 2 (16x8): graph paths differ from the default's (16x16), N=1/N=8 rows
 *      resolve (full).
 *   2  quantization alternate 1 (8x8): not registered -- every config is refused (AINR_ERR_GRID),
 *      by the sizing call and by init, which leaves its outputs untouched.
 *   3  the OTHER archive variant's id (lite id in a full archive, full id in a lite one): every
 *      config is refused with AINR_ERR_MODEL_ID.
 * Not built under NO_STDIO. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_product.h"
#include "ainr_test.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb, i;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *inv = (float *)malloc(elems * sizeof(float));
    int32_t hdr[2];
    FILE *fp;
    if (!fwd || !inv) exit(1);
    for (i = 0; i < elems; ++i) { fwd[i] = (float)(i % 17) * 0.01f; inv[i] = (float)((i + 3) % 11) * 0.02f; }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd); free(inv);
}

#if AINR_TEST_PRODUCT == 1
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

#if AINR_TEST_PRODUCT != 1
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
    snprintf(erb, sizeof(erb), "/tmp/ainr_product_erb_%d.bin", (int)getpid());
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
