/* Host test fixture of the ainr suites: the ERB table writer and the aligned pool allocator. Header-only; not part of
 * the library. link_only_ainr.c keeps its own copy (it is a link probe). */

#ifndef AINR_TEST_FIXTURE_H
#define AINR_TEST_FIXTURE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../dfn2/tests/erb_fixture.h"   /* erb_fixture_scratch_path() */

/* ERB_PATTERN: fwd[i] = (i % 17) * 0.01 and an inverse that splits bin k 0.75 / 0.25 over bands k % n_erb and
 * (k + 1) % n_erb (bin-major both; ainr_init refuses an inverse that is not a partition of unity). ERB_IDENTITY: every bin in
 * one contiguous band, forward = band mean, inverse = 1, so a unit mask is a unit gain.
 * ERB_NOT_PARTITION: inv[i] = ((i + 3) % 11) * 0.02, an inverse whose bins do not sum to 1 (refused by init).
 * ERB_BAND0: zero forward table, band 0 owns every bin (a partition of unity). */
typedef enum AinrFixtureErb { ERB_PATTERN = 0, ERB_IDENTITY = 1, ERB_NOT_PARTITION = 2, ERB_BAND0 = 3 } AinrFixtureErb;

/* Writes the ERB file ainr_init_from_erb_file() reads: int32 n_bins, n_erb, then both tables; exits on failure. */
static inline void write_erb_file(const char *path, int n_bins, int n_erb, AinrFixtureErb mode)
{
    size_t elems = (size_t)n_bins * (size_t)n_erb, i;
    float *fwd = (float *)calloc(elems, sizeof(float));
    float *inv = (float *)calloc(elems, sizeof(float));
    int *size = (int *)calloc((size_t)n_erb, sizeof(int));
    int32_t hdr[2];
    FILE *fp;
    int k, e;

    if (!fwd || !inv || !size) { fprintf(stderr, "calloc fail\n"); exit(1); }
    if (mode == ERB_IDENTITY) {
        for (k = 0; k < n_bins; ++k) { e = (int)((long long)k * n_erb / n_bins); size[e]++; }
        for (k = 0; k < n_bins; ++k) {
            e = (int)((long long)k * n_erb / n_bins);
            fwd[(size_t)k * (size_t)n_erb + (size_t)e] = 1.0f / (float)size[e];
            inv[(size_t)k * (size_t)n_erb + (size_t)e] = 1.0f;
        }
    } else if (mode == ERB_BAND0) {
        for (k = 0; k < n_bins; ++k) inv[(size_t)k * (size_t)n_erb] = 1.0f;
    } else if (mode == ERB_NOT_PARTITION) {
        for (i = 0; i < elems; ++i) {
            fwd[i] = (float)(i % 17) * 0.01f;
            inv[i] = (float)((i + 3) % 11) * 0.02f;
        }
    } else {
        for (i = 0; i < elems; ++i) fwd[i] = (float)(i % 17) * 0.01f;
        for (k = 0; k < n_bins; ++k) {   /* a partition of unity: bin k splits 0.75 / 0.25 over two bands */
            inv[(size_t)k * (size_t)n_erb + (size_t)(k % n_erb)] = 0.75f;
            inv[(size_t)k * (size_t)n_erb + (size_t)((k + 1) % n_erb)] = 0.25f;
        }
    }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd); free(inv); free(size);
}

/* A deterministic non-silent hop of int16 samples; `seed` moves the pattern. */
static inline void fill_pcm(int16_t *pcm, int n, int seed)
{
    int i;
    for (i = 0; i < n; ++i) pcm[i] = (int16_t)((i * 13 + seed * 7) % 211 - 100);
}

static inline void write_synthetic_erb(const char *path, int n_bins, int n_erb)
{
    write_erb_file(path, n_bins, n_erb, ERB_PATTERN);
}

static inline void write_non_partition_erb(const char *path, int n_bins, int n_erb)
{
    write_erb_file(path, n_bins, n_erb, ERB_NOT_PARTITION);
}

static inline void write_identity_erb(const char *path, int n_bins, int n_erb)
{
    write_erb_file(path, n_bins, n_erb, ERB_IDENTITY);
}

static inline void *alloc_aligned(size_t bytes, unsigned int alignment)
{
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
}

#endif /* AINR_TEST_FIXTURE_H */
