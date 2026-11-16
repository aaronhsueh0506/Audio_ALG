/* Inverse ERB tables for the dfn2_stream tests. dfn2_stream_init refuses an erb_inv that is not a
 * partition of unity (every bin's band weights sum to exactly 1.0f), so the tests' stand-in
 * tables are partitions: not the model's ERB matrices. */
#ifndef DFN2_STREAM_TEST_ERB_FIXTURE_H
#define DFN2_STREAM_TEST_ERB_FIXTURE_H

#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* A scratch file name under /tmp: the file is created here with O_EXCL (mode 0600), so a symlink or
 * file planted under a guessed name is refused, never followed; the caller rewrites it and unlinks it. Exits on failure. */
static inline void erb_fixture_scratch_path(char *out, size_t cap, const char *tag)
{
    int attempt;
    for (attempt = 0; attempt < 64; ++attempt) {
        int fd;
        if (snprintf(out, cap, "/tmp/%s_%d_%d.bin", tag, (int)getpid(), attempt) >= (int)cap) break;
        fd = open(out, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) { close(fd); return; }
    }
    fprintf(stderr, "cannot create a scratch file for %s\n", tag);
    exit(2);
}

/* Band 0 owns every bin. band_major [n_erb][n_bins]. */
static inline void erb_fixture_unit_inv(float *inv_band_major, int n_bins, int n_erb)
{
    size_t k, total = (size_t)n_bins * (size_t)n_erb;
    for (k = 0; k < total; ++k) inv_band_major[k] = k < (size_t)n_bins ? 1.0f : 0.0f;
}

/* bin-major [n_bins][n_erb]: bin k splits 0.75 / 0.25 between bands k % n_erb and (k + 1) % n_erb
 * (exact in float). Not symmetric in bins and bands, so a transposed read is a different table. */
static inline void erb_fixture_split_inv_bin_major(float *inv_bin_major, int n_bins, int n_erb)
{
    int k;
    for (k = 0; k < n_bins * n_erb; ++k) inv_bin_major[k] = 0.0f;
    for (k = 0; k < n_bins; ++k) {
        inv_bin_major[(size_t)k * (size_t)n_erb + (size_t)(k % n_erb)] = 0.75f;
        inv_bin_major[(size_t)k * (size_t)n_erb + (size_t)((k + 1) % n_erb)] = 0.25f;
    }
}

/* The same split table, band_major [n_erb][n_bins]. */
static inline void erb_fixture_split_inv_band_major(float *inv_band_major, int n_bins, int n_erb)
{
    int k;
    for (k = 0; k < n_bins * n_erb; ++k) inv_band_major[k] = 0.0f;
    for (k = 0; k < n_bins; ++k) {
        inv_band_major[(size_t)(k % n_erb) * (size_t)n_bins + (size_t)k] = 0.75f;
        inv_band_major[(size_t)((k + 1) % n_erb) * (size_t)n_bins + (size_t)k] = 0.25f;
    }
}

#endif /* DFN2_STREAM_TEST_ERB_FIXTURE_H */
