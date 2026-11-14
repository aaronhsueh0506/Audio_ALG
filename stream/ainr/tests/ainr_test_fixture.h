/* Host test fixture of the ainr suites: the ERB table writer and the aligned pool allocator. Header-only; not part of
 * the library. link_only_ainr.c keeps its own copy (it is a link probe). */

#ifndef AINR_TEST_FIXTURE_H
#define AINR_TEST_FIXTURE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ERB_PATTERN: fwd[i] = (i % 17) * 0.01, inv[i] = ((i + 3) % 11) * 0.02 (bin-major both). ERB_IDENTITY: every bin in
 * one contiguous band, forward = band mean, inverse = 1, so a unit mask is a unit gain. */
typedef enum AinrFixtureErb { ERB_PATTERN = 0, ERB_IDENTITY = 1 } AinrFixtureErb;

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
    } else {
        for (i = 0; i < elems; ++i) {
            fwd[i] = (float)(i % 17) * 0.01f;
            inv[i] = (float)((i + 3) % 11) * 0.02f;
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

static inline void write_synthetic_erb(const char *path, int n_bins, int n_erb)
{
    write_erb_file(path, n_bins, n_erb, ERB_PATTERN);
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
