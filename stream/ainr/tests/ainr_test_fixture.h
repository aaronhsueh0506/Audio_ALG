/* Host test fixture of the ainr suites: in-memory ERB tables and the aligned
 * pool allocator. Header-only; not part of the library. link_only_ainr.c keeps its own copy (it is a link probe). */

#ifndef AINR_TEST_FIXTURE_H
#define AINR_TEST_FIXTURE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../ainr.h"
#include "../dfn2/tests/erb_fixture.h"   /* erb_fixture_split_inv_band_major() */

/* ERB_PATTERN: fwd[i] = (i % 17) * 0.01 and an inverse that splits bin k 0.75 / 0.25 over bands k % n_erb and
 * (k + 1) % n_erb (ainr_init refuses an inverse that is not a partition of unity). ERB_IDENTITY: every bin in
 * one contiguous band, forward = band mean, inverse = 1, so a unit mask is a unit gain.
 * ERB_NOT_PARTITION: inv at bin-major index i = ((i + 3) % 11) * 0.02, an inverse whose bins do not sum to 1 (refused by init).
 * ERB_BAND0: zero forward table, band 0 owns every bin (a partition of unity). */
typedef enum AinrFixtureErb { ERB_PATTERN = 0, ERB_IDENTITY = 1, ERB_NOT_PARTITION = 2, ERB_BAND0 = 3 } AinrFixtureErb;

/* In-memory ERB tables as AinrConfig takes them: fwd bin-major [n_bins][n_erb], inv band-major [n_erb][n_bins]; n_erb is
 * the band count AinrConfig.n_erb gets (a test may overwrite it to present tables of another count). */
typedef struct ErbTables { float *fwd, *inv; int n_erb; } ErbTables;

static inline ErbTables erb_tables(int n_bins, int n_erb, AinrFixtureErb mode)
{
    size_t elems = (size_t)n_bins * (size_t)n_erb, i;
    int k, e;
    ErbTables t;

    t.n_erb = n_erb;
    t.fwd = (float *)calloc(elems, sizeof(float));
    t.inv = (float *)calloc(elems, sizeof(float));
    if (!t.fwd || !t.inv) { fprintf(stderr, "calloc fail\n"); exit(1); }
    if (mode == ERB_IDENTITY) {
        int *size = (int *)calloc((size_t)n_erb, sizeof(int));
        if (!size) { fprintf(stderr, "calloc fail\n"); exit(1); }
        for (k = 0; k < n_bins; ++k) size[(int)((long long)k * n_erb / n_bins)]++;
        for (k = 0; k < n_bins; ++k) {
            e = (int)((long long)k * n_erb / n_bins);
            t.fwd[(size_t)k * (size_t)n_erb + (size_t)e] = 1.0f / (float)size[e];
            t.inv[(size_t)e * (size_t)n_bins + (size_t)k] = 1.0f;
        }
        free(size);
    } else if (mode == ERB_BAND0) {
        for (k = 0; k < n_bins; ++k) t.inv[k] = 1.0f;
    } else if (mode == ERB_NOT_PARTITION) {
        for (k = 0; k < n_bins; ++k)
            for (e = 0; e < n_erb; ++e) {
                i = (size_t)k * (size_t)n_erb + (size_t)e;
                t.fwd[i] = (float)(i % 17) * 0.01f;
                t.inv[(size_t)e * (size_t)n_bins + (size_t)k] = (float)((i + 3) % 11) * 0.02f;
            }
    } else {
        for (i = 0; i < elems; ++i) t.fwd[i] = (float)(i % 17) * 0.01f;
        erb_fixture_split_inv_band_major(t.inv, n_bins, n_erb);
    }
    return t;
}

static inline ErbTables erb_tables_synthetic(int n_bins, int n_erb) { return erb_tables(n_bins, n_erb, ERB_PATTERN); }
static inline ErbTables erb_tables_identity(int n_bins, int n_erb) { return erb_tables(n_bins, n_erb, ERB_IDENTITY); }
static inline ErbTables erb_tables_non_partition(int n_bins, int n_erb) { return erb_tables(n_bins, n_erb, ERB_NOT_PARTITION); }

static inline void erb_tables_free(ErbTables *t)
{
    free(t->fwd); free(t->inv);
    t->fwd = NULL; t->inv = NULL; t->n_erb = 0;
}

/* cfg with t's tables and band count set (cfg is not modified). */
static inline AinrConfig ainr_cfg_with(const AinrConfig *cfg, const ErbTables *t)
{
    AinrConfig c = *cfg;
    c.erb_fwd = t ? t->fwd : NULL;
    c.erb_inv = t ? t->inv : NULL;
    c.n_erb = t ? t->n_erb : cfg->n_erb;
    return c;
}

static inline int ainr_init_t(const AinrConfig *cfg, const ErbTables *t, void *pool, size_t bytes, AinrCtx **ctx,
                              AinrInfo *info)
{
    AinrConfig c;
    if (!cfg) return ainr_init(NULL, pool, bytes, ctx, info);
    c = ainr_cfg_with(cfg, t);
    return ainr_init(&c, pool, bytes, ctx, info);
}

static inline int ainr_init_freq_t(const AinrConfig *cfg, const ErbTables *t, void *pool, size_t bytes,
                                   AinrFreqCtx **ctx, AinrInfo *info)
{
    AinrConfig c;
    if (!cfg) return ainr_init_freq(NULL, pool, bytes, ctx, info);
    c = ainr_cfg_with(cfg, t);
    return ainr_init_freq(&c, pool, bytes, ctx, info);
}

/* A deterministic non-silent hop of int16 samples; `seed` moves the pattern. */
static inline void fill_pcm(int16_t *pcm, int n, int seed)
{
    int i;
    for (i = 0; i < n; ++i) pcm[i] = (int16_t)((i * 13 + seed * 7) % 211 - 100);
}

static inline void *alloc_aligned(size_t bytes, unsigned int alignment)
{
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
}

#endif /* AINR_TEST_FIXTURE_H */
