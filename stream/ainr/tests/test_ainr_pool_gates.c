/* The init gates of both families (PCM, _freq): init recomputes the size itself. A pool one byte short or
 * misaligned is refused (AINR_ERR_POOL) before anything is written; a larger pool is accepted; NULL
 * arguments are AINR_ERR_NULL; NULL ERB tables (the pool stays unwritten) and tables the init refuses (an
 * inverse that is not a partition of unity, a non-finite or negative forward value) and an n_erb other than the grid's are AINR_ERR_ERB_TABLES. Every refusal leaves *ctx NULL
 * and *info untouched (a pool refusal also leaves the pool bytes unchanged), and the same pool then
 * initialises normally. The size of each family is the literal of its case (_freq: no analysis, synthesis or PCM
 * buffer, so the smaller one), and blocks run on a pool of exactly that size leave the bytes after it alone. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr.h"
#include "ainr_test_fixture.h"

typedef enum { FAM_PCM = 0, FAM_FREQ } Fam;

static const char *fam_name(Fam f) { return f == FAM_PCM ? "pcm" : "freq"; }

static int get_size(Fam f, const AinrConfig *cfg, StreamMemReq *req) {
    return f == FAM_PCM ? ainr_get_mem_size(cfg, req) : ainr_get_mem_size_freq(cfg, req);
}

/* Init through the family's entry over `erb`; the context pointer goes to *ctx_out (NULL when refused).
 * The family's context pointer starts as junk, so a path that forgets to clear it is caught. info
 * may be NULL to test the argument check. */
static char g_junk;

static int try_init(Fam f, const AinrConfig *cfg, const ErbTables *erb, void *pool, size_t bytes, void **ctx_out, AinrInfo *info) {
    int rc;
    switch (f) {
    case FAM_PCM: {
        AinrCtx *c = (AinrCtx *)(void *)&g_junk;
        rc = ainr_init_t(cfg, erb, pool, bytes, &c, info);
        *ctx_out = c;
        break;
    }
    default: {
        AinrFreqCtx *c = (AinrFreqCtx *)(void *)&g_junk;
        rc = ainr_init_freq_t(cfg, erb, pool, bytes, &c, info);
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
static int refused(Fam f, const AinrConfig *cfg, const ErbTables *erb, unsigned char *pool, size_t pool_off,
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

static int run_family(Fam f, const AinrConfig *cfg, const ErbTables *erb, int n_erb, uint64_t want_bytes, const char *label) {
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
    REQUIRE(ctx != NULL && info.mask_lookahead == (cfg->sr == 48000 ? 1 : 0), "exact pool: ctx/info not set");
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

    /* NULL tables, either or both: AINR_ERR_ERB_TABLES, ctx NULL, info untouched, and the pool not written */
    {
        const ErbTables no_fwd = { NULL, erb->inv, erb->n_erb }, no_inv = { erb->fwd, NULL, erb->n_erb },
                         none = { NULL, NULL, erb->n_erb };
        const struct { const ErbTables *t; const char *what; } cases[3] = {
            { &no_fwd, "NULL erb_fwd" }, { &no_inv, "NULL erb_inv" }, { &none, "NULL erb_fwd and erb_inv" } };
        int c;
        for (c = 0; c < 3; ++c)
            if (refused(f, cfg, cases[c].t, buf, 0, bytes, bytes + 128, AINR_ERR_ERB_TABLES, cases[c].what, label)) return 1;
    }

    /* n_erb one below and one above the grid's, over the grid's own tables: the same code, nothing written */
    {
        const int off[2] = { n_erb - 1, n_erb + 1 };
        int c;
        for (c = 0; c < 2; ++c) {
            ErbTables wrong = *erb;
            wrong.n_erb = off[c];
            if (refused(f, cfg, &wrong, buf, 0, bytes, bytes + 128, AINR_ERR_ERB_TABLES,
                        c == 0 ? "n_erb one below the grid's" : "n_erb one above the grid's", label)) return 1;
        }
    }

    /* tables the init refuses (an inverse that is not a partition of unity, a NaN / Inf / negative forward value): the
     * same code, ctx NULL, info untouched */
    {
        static const char *const WHAT[4] = { "non-partition inverse", "NaN first forward value", "Inf last forward value",
                                             "negative forward value" };
        int d;
        for (d = 0; d < 4; ++d) {
            AinrInfo i5, before5;
            void *c5 = NULL;
            ErbTables bad = d == 0 ? erb_tables_non_partition(cfg->fft / 2 + 1, n_erb) : erb_tables_synthetic(cfg->fft / 2 + 1, n_erb);
            size_t elems = (size_t)(cfg->fft / 2 + 1) * (size_t)n_erb;
            if (d == 1) bad.fwd[0] = (float)NAN;
            if (d == 2) bad.fwd[elems - 1u] = (float)INFINITY;
            if (d == 3) bad.fwd[elems / 2u] = -0.5f;
            memset(&i5, 0x5a, sizeof(i5));
            before5 = i5;
            EXPECT(try_init(f, cfg, &bad, buf, bytes, &c5, &i5), AINR_ERR_ERB_TABLES, WHAT[d]);
            REQUIRE(c5 == NULL, "refused tables: ctx set");
            REQUIRE(memcmp(&i5, &before5, sizeof(i5)) == 0, "refused tables: info written");
            erb_tables_free(&bad);
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
    AinrConfig cfg = {0};
    ErbTables erb = erb_tables_synthetic(fft / 2 + 1, n_erb);
    int f, rc = 0;
    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = n * (fft / 2);
    for (f = FAM_PCM; f <= FAM_FREQ; ++f)
        rc |= run_family((Fam)f, &cfg, &erb, n_erb, f == FAM_PCM ? pcm_bytes : freq_bytes, label);
    erb_tables_free(&erb);
    return rc;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, 91504, 32192, "poolgates[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, 63248, 47712, "poolgates[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, 104064, 102976, "poolgates[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, 151888, 92576, "poolgates[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_pool_gates: FAIL\n"); return 1; }
    printf("test_ainr_pool_gates: PASS\n");
    return 0;
}
