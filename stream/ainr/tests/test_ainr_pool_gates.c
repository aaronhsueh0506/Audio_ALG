/* The init gates of all three families: init recomputes the size itself. A pool one byte short or
 * misaligned is refused (AINR_ERR_POOL) before anything is written; a larger pool is accepted; NULL
 * arguments are AINR_ERR_NULL; a missing ERB file is AINR_ERR_ERB_FILE. Every refusal leaves *ctx NULL
 * and *info untouched (a pool refusal also leaves the pool bytes unchanged), and the same pool then
 * initialises normally. Not built under NO_STDIO. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"

typedef enum { FAM_PCM = 0, FAM_SPEC, FAM_FREQ } Fam;

static const char *fam_name(Fam f) { return f == FAM_PCM ? "pcm" : f == FAM_SPEC ? "spec" : "freq"; }

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

static int get_size(Fam f, const AinrConfig *cfg, StreamMemReq *req) {
    return f == FAM_PCM ? ainr_get_mem_size(cfg, req)
         : f == FAM_SPEC ? ainr_get_mem_size_spec(cfg, req)
         : ainr_get_mem_size_freq(cfg, req);
}

/* Init through the family's file entry; the context pointer goes to *ctx_out (NULL when refused).
 * The family's context pointer starts as junk, so a path that forgets to clear it is caught. info
 * may be NULL to test the argument check. */
static char g_junk;

static int try_init(Fam f, const AinrConfig *cfg, const char *erb, void *pool, size_t bytes, void **ctx_out, AinrInfo *info) {
    int rc;
    switch (f) {
    case FAM_PCM: {
        AinrCtx *c = (AinrCtx *)(void *)&g_junk;
        rc = ainr_init_from_erb_file(cfg, erb, pool, bytes, &c, info);
        *ctx_out = c;
        break;
    }
    case FAM_SPEC: {
        AinrSpecCtx *c = (AinrSpecCtx *)(void *)&g_junk;
        rc = ainr_init_spec_from_erb_file(cfg, erb, pool, bytes, &c, info);
        *ctx_out = c;
        break;
    }
    default: {
        AinrFreqCtx *c = (AinrFreqCtx *)(void *)&g_junk;
        rc = ainr_init_freq_from_erb_file(cfg, erb, pool, bytes, &c, info);
        *ctx_out = c;
        break;
    }
    }
    return rc;
}

#define EXPECT(got, want, what) do { if ((got) != (want)) { \
    fprintf(stderr, "%s/%s: %s: got %d (%s), want %d (%s)\n", label, fam_name(f), what, (got), \
            ainr_status_text(got), (want), ainr_status_text(want)); return 1; } } while (0)

#define REQUIRE(c, what) do { if (!(c)) { fprintf(stderr, "%s/%s: %s\n", label, fam_name(f), what); return 1; } } while (0)

/* A refused init: the code, ctx NULL, info as it was, the pool's bytes as they were. */
static int refused(Fam f, const AinrConfig *cfg, const char *erb, unsigned char *pool, size_t pool_off,
                   size_t bytes, size_t span, int want, const char *what, const char *label) {
    AinrInfo info, before;
    unsigned char *snap = (unsigned char *)malloc(span);
    void *ctx = NULL;
    int rc;
    if (!snap) exit(1);
    memset(pool, 0xC3, span);
    memcpy(snap, pool, span);
    memset(&info, 0x5a, sizeof(info));
    before = info;
    rc = try_init(f, cfg, erb, pool + pool_off, bytes, &ctx, &info);
    EXPECT(rc, want, what);
    REQUIRE(ctx == NULL, "ctx not NULL after a refusal");
    REQUIRE(memcmp(&info, &before, sizeof(info)) == 0, "info written by a refusal");
    REQUIRE(memcmp(snap, pool, span) == 0, "pool written by a refusal");
    free(snap);
    return 0;
}

static int run_family(Fam f, const AinrConfig *cfg, const char *erb, const char *label) {
    StreamMemReq req;
    unsigned char *buf;
    size_t bytes;
    AinrInfo info;
    void *ctx = NULL;
    int rc;

    if (get_size(f, cfg, &req) != AINR_OK) { fprintf(stderr, "%s/%s: size fail\n", label, fam_name(f)); return 1; }
    REQUIRE(req.alignment == AINR_ALIGNMENT, "reported alignment is not AINR_ALIGNMENT");
    bytes = (size_t)req.bytes;
    if (posix_memalign((void **)&buf, 64, bytes + 128) != 0) exit(1);

    /* exact pool, then a larger one */
    memset(&info, 0, sizeof(info));
    rc = try_init(f, cfg, erb, buf, bytes, &ctx, &info);
    EXPECT(rc, AINR_OK, "exact pool");
    REQUIRE(ctx != NULL && info.graph_path != NULL && info.tail_zero_blocks >= 1, "exact pool: ctx/info not set");
    rc = try_init(f, cfg, erb, buf, bytes + 64, &ctx, &info);
    EXPECT(rc, AINR_OK, "larger pool");
    REQUIRE(ctx != NULL, "larger pool: ctx NULL");

    /* too small, misaligned: refused before any write */
    if (refused(f, cfg, erb, buf, 0, bytes - 1, bytes + 128, AINR_ERR_POOL, "pool one byte short", label)) return 1;
    if (refused(f, cfg, erb, buf, 0, 0, bytes + 128, AINR_ERR_POOL, "zero-byte pool", label)) return 1;
    if (refused(f, cfg, erb, buf, 1, bytes + 64, bytes + 128, AINR_ERR_POOL, "misaligned pool", label)) return 1;
    if (refused(f, cfg, erb, buf, 8, bytes + 64, bytes + 128, AINR_ERR_POOL, "pool 8-byte aligned only", label)) return 1;

    /* NULL arguments */
    {
        AinrInfo i2;
        void *c2 = NULL;
        EXPECT(try_init(f, cfg, erb, NULL, bytes, &c2, &i2), AINR_ERR_NULL, "NULL pool");
        REQUIRE(c2 == NULL, "NULL pool: ctx set");
        EXPECT(try_init(f, cfg, erb, buf, bytes, &c2, NULL), AINR_ERR_NULL, "NULL info");
        REQUIRE(c2 == NULL, "NULL info: ctx set");
        EXPECT(try_init(f, NULL, erb, buf, bytes, &c2, &i2), AINR_ERR_NULL, "NULL cfg");
        REQUIRE(c2 == NULL, "NULL cfg: ctx set");
    }

    /* a missing ERB file: refused after the pool was carved, ctx NULL and info untouched */
    {
        AinrInfo i3, before3;
        void *c3 = NULL;
        memset(&i3, 0x5a, sizeof(i3));
        before3 = i3;
        EXPECT(try_init(f, cfg, "/nonexistent/ainr_erb.bin", buf, bytes, &c3, &i3), AINR_ERR_ERB_FILE, "missing ERB file");
        REQUIRE(c3 == NULL, "missing ERB file: ctx set");
        REQUIRE(memcmp(&i3, &before3, sizeof(i3)) == 0, "missing ERB file: info written");
    }

    /* the refusals left nothing behind: the same pool initialises again */
    rc = try_init(f, cfg, erb, buf, bytes, &ctx, &info);
    EXPECT(rc, AINR_OK, "init after refusals");
    REQUIRE(ctx != NULL, "init after refusals: ctx NULL");
    free(buf);
    printf("%s/%s: pool gates OK\n", label, fam_name(f));
    return 0;
}

static int run_case(int sr, int fft, int n_erb, int n, const char *label) {
    AinrConfig cfg;
    char erb[160];
    int f, rc = 0;
    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = n * (fft / 2);
    snprintf(erb, sizeof(erb), "/tmp/ainr_poolgates_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, fft / 2 + 1, n_erb);
    for (f = FAM_PCM; f <= FAM_FREQ; ++f) rc |= run_family((Fam)f, &cfg, erb, label);
    unlink(erb);
    return rc;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, "poolgates[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, "poolgates[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, "poolgates[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, "poolgates[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_pool_gates: FAIL\n"); return 1; }
    printf("test_ainr_pool_gates: PASS\n");
    return 0;
}
