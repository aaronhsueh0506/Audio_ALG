/* The init gates of both families (PCM, _freq): init recomputes the size itself. A pool one byte short or
 * misaligned is refused (AINR_ERR_POOL) before anything is written; a larger pool is accepted; NULL
 * arguments are AINR_ERR_NULL; a missing ERB file, one with bytes after the tables and one with a non-finite
 * forward value are AINR_ERR_ERB_FILE. Every refusal leaves *ctx NULL
 * and *info untouched (a pool refusal also leaves the pool bytes unchanged), and the same pool then
 * initialises normally. The size of each family is the literal of its case (_freq: no analysis, synthesis or PCM
 * buffer, so the smaller one), and blocks run on a pool of exactly that size leave the bytes after it alone.
 * Not built under NO_STDIO. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "ainr_test_fixture.h"

typedef enum { FAM_PCM = 0, FAM_FREQ } Fam;

static const char *fam_name(Fam f) { return f == FAM_PCM ? "pcm" : "freq"; }

static int get_size(Fam f, const AinrConfig *cfg, StreamMemReq *req) {
    return f == FAM_PCM ? ainr_get_mem_size(cfg, req) : ainr_get_mem_size_freq(cfg, req);
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
    default: {
        AinrFreqCtx *c = (AinrFreqCtx *)(void *)&g_junk;
        rc = ainr_init_freq_from_erb_file(cfg, erb, pool, bytes, &c, info);
        *ctx_out = c;
        break;
    }
    }
    return rc;
}

/* A copy of the ERB file `src` with one defect: 0 a byte appended, 1 the first forward value NaN, 2 the last
 * forward value +Inf. */
static void write_defective_erb(const char *src, const char *dst, int n_bins, int n_erb, int defect) {
    size_t elems = (size_t)n_bins * (size_t)n_erb, n = 8u + 8u * elems;
    unsigned char *buf = (unsigned char *)malloc(n + 1u);
    float v = defect == 1 ? (float)NAN : (float)INFINITY;
    FILE *fp = fopen(src, "rb");
    if (!buf || !fp || fread(buf, 1, n, fp) != n) exit(1);
    fclose(fp);
    if (defect == 1) memcpy(buf + 8, &v, sizeof(v));
    if (defect == 2) memcpy(buf + 8 + 4u * (elems - 1u), &v, sizeof(v));
    buf[n] = 0;
    fp = fopen(dst, "wb");
    if (!fp || fwrite(buf, 1, n + (defect == 0 ? 1u : 0u), fp) != n + (defect == 0 ? 1u : 0u)) exit(1);
    fclose(fp);
    free(buf);
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

/* A few blocks with a reset and a failed run in between, through the family's own entries. */
static int run_blocks(Fam f, void *ctx, const AinrConfig *cfg) {
    const int n = cfg->hop_in / (cfg->fft / 2), n_bins = cfg->fft / 2 + 1;
    const size_t spec = (size_t)n * (size_t)n_bins;
    int16_t *pcm = (int16_t *)calloc((size_t)cfg->hop_in * 2u, sizeof(int16_t));
    float *x = (float *)calloc(spec * 4u, sizeof(float));
    int t, k, j, rc = 0;
    if (!pcm || !x) exit(1);
    for (t = 0; t < 12 && rc == 0; ++t) {
        StreamWork w;
        for (k = 0; k < cfg->hop_in; ++k) pcm[k] = (int16_t)((k * 29 + t * 7) % 2001 - 1000);
        for (k = 0; k < (int)(spec * 2u); ++k) x[k] = (float)((k * 13 + t * 5) % 97) / 97.0f - 0.5f;
        if (t == 5) { if (f == FAM_PCM) ainr_reset((AinrCtx *)ctx); else ainr_reset_freq((AinrFreqCtx *)ctx); }
        rc = f == FAM_PCM ? ainr_preprocess((AinrCtx *)ctx, pcm, &w)
                          : ainr_preprocess_freq((AinrFreqCtx *)ctx, x, x + spec, x, x + spec, &w);
        if (rc != AINR_OK) break;
        for (j = 0; j < w.output_count; ++j)
            for (k = 0; k < (int)w.outputs[j].elements; ++k) w.outputs[j].data[k] = 0.25f;
        rc = f == FAM_PCM ? ainr_postprocess((AinrCtx *)ctx, t == 8, pcm + cfg->hop_in)
                          : ainr_postprocess_freq((AinrFreqCtx *)ctx, t == 8, x + 2u * spec, x + 3u * spec);
    }
    free(pcm);
    free(x);
    return rc;
}

static int run_family(Fam f, const AinrConfig *cfg, const char *erb, int n_erb, uint64_t want_bytes, const char *label) {
    StreamMemReq req;
    unsigned char *buf;
    size_t bytes;
    AinrInfo info;
    void *ctx = NULL;
    int rc;

    if (get_size(f, cfg, &req) != AINR_OK) { fprintf(stderr, "%s/%s: size fail\n", label, fam_name(f)); return 1; }
    REQUIRE(req.alignment == AINR_ALIGNMENT, "reported alignment is not AINR_ALIGNMENT");
    if (req.bytes != want_bytes) {
        fprintf(stderr, "%s/%s: pool is %llu bytes, want %llu\n", label, fam_name(f),
                (unsigned long long)req.bytes, (unsigned long long)want_bytes);
        return 1;
    }
    bytes = (size_t)req.bytes;
    if (posix_memalign((void **)&buf, 64, bytes + 128) != 0) exit(1);

    /* exact pool (blocks run on it stay inside it), then a larger one */
    memset(&info, 0, sizeof(info));
    memset(buf + bytes, 0xC3, 128);
    rc = try_init(f, cfg, erb, buf, bytes, &ctx, &info);
    EXPECT(rc, AINR_OK, "exact pool");
    REQUIRE(ctx != NULL && info.graph_path != NULL, "exact pool: ctx/info not set");
    EXPECT(run_blocks(f, ctx, cfg), AINR_OK, "blocks on the exact pool");
    {
        size_t k;
        for (k = 0; k < 128; ++k) REQUIRE(buf[bytes + k] == 0xC3, "a write past the exact pool");
    }
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

    /* an ERB file whose inverse is not a partition of unity: refused the same way, ctx NULL, info untouched */
    {
        AinrInfo i4, before4;
        void *c4 = NULL;
        char bad[200];
        snprintf(bad, sizeof(bad), "%s.bad", erb);
        write_non_partition_erb(bad, cfg->fft / 2 + 1, n_erb);
        memset(&i4, 0x5a, sizeof(i4));
        before4 = i4;
        EXPECT(try_init(f, cfg, bad, buf, bytes, &c4, &i4), AINR_ERR_ERB_FILE, "non-partition ERB file");
        REQUIRE(c4 == NULL, "non-partition ERB file: ctx set");
        REQUIRE(memcmp(&i4, &before4, sizeof(i4)) == 0, "non-partition ERB file: info written");
        unlink(bad);
    }

    /* bytes after the tables, a NaN first / Inf last forward value: refused the same way */
    {
        static const char *const WHAT[3] = { "ERB file with a byte after the tables", "ERB file with a NaN forward value",
                                             "ERB file with an Inf last forward value" };
        int d;
        for (d = 0; d < 3; ++d) {
            AinrInfo i5, before5;
            void *c5 = NULL;
            char bad[200];
            snprintf(bad, sizeof(bad), "%s.defect", erb);
            write_defective_erb(erb, bad, cfg->fft / 2 + 1, n_erb, d);
            memset(&i5, 0x5a, sizeof(i5));
            before5 = i5;
            EXPECT(try_init(f, cfg, bad, buf, bytes, &c5, &i5), AINR_ERR_ERB_FILE, WHAT[d]);
            REQUIRE(c5 == NULL, "defective ERB file: ctx set");
            REQUIRE(memcmp(&i5, &before5, sizeof(i5)) == 0, "defective ERB file: info written");
            unlink(bad);
        }
    }

    /* the refusals left nothing behind: the same pool initialises again */
    rc = try_init(f, cfg, erb, buf, bytes, &ctx, &info);
    EXPECT(rc, AINR_OK, "init after refusals");
    REQUIRE(ctx != NULL, "init after refusals: ctx NULL");
    free(buf);
    printf("%s/%s: pool gates OK\n", label, fam_name(f));
    return 0;
}

static int run_case(int sr, int fft, int n_erb, int n, uint64_t pcm_bytes, uint64_t freq_bytes, const char *label) {
    AinrConfig cfg;
    char erb[160];
    int f, rc = 0;
    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = n * (fft / 2);
    snprintf(erb, sizeof(erb), "/tmp/ainr_poolgates_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, fft / 2 + 1, n_erb);
    for (f = FAM_PCM; f <= FAM_FREQ; ++f)
        rc |= run_family((Fam)f, &cfg, erb, n_erb, f == FAM_PCM ? pcm_bytes : freq_bytes, label);
    unlink(erb);
    return rc;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, 235360, 171952, "poolgates[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, 91312, 74752, "poolgates[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, 148480, 137360, "poolgates[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, 295728, 232320, "poolgates[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_pool_gates: FAIL\n"); return 1; }
    printf("test_ainr_pool_gates: PASS\n");
    return 0;
}
