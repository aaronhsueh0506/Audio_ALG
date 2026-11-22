/* Reads fb_<sample_rate>_<n_erb>.bin, the DFN2 ERB matrices of one grid, into caller buffers for AinrConfig.erb_fwd /
 * erb_inv (or DFN2StreamConfig). Sample-side and header-only: no library archive reads a file. The file is int32 n_bins,
 * int32 n_erb, then n_bins * n_erb float32 forward and inverse, both bin-major, and nothing after. */

#ifndef DFN2_ERB_FILE_H
#define DFN2_ERB_FILE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* erb_fwd gets the forward table bin-major [n_bins][n_erb], erb_inv the inverse band-major [n_erb][n_bins], each
 * n_bins * n_erb floats. 0, or -1 on an open failure, a header of other dimensions, a short read or a byte after the
 * tables. The values are checked by the library's init (dfn2_erb_tables_valid). */
static inline int dfn2_erb_file_load(const char *path, int n_bins, int n_erb, float *erb_fwd, float *erb_inv)
{
    FILE *fp;
    int32_t hdr[2];
    float chunk[64];
    size_t want, done = 0, k;
    int rc = 0;

    if (!path || !erb_fwd || !erb_inv || n_bins <= 0 || n_erb <= 0) return -1;
    want = (size_t)n_bins * (size_t)n_erb;
    fp = fopen(path, "rb");
    if (!fp) return -1;
    if (fread(hdr, sizeof(int32_t), 2, fp) != 2 || hdr[0] != (int32_t)n_bins || hdr[1] != (int32_t)n_erb ||
        fread(erb_fwd, sizeof(float), want, fp) != want)
        rc = -1;
    while (rc == 0 && done < want) {   /* the stored bin-major inverse, transposed in place a chunk at a time */
        size_t n = want - done < 64u ? want - done : 64u;
        if (fread(chunk, sizeof(float), n, fp) != n) { rc = -1; break; }
        for (k = 0; k < n; ++k) {
            size_t e = done + k;
            erb_inv[(e % (size_t)n_erb) * (size_t)n_bins + e / (size_t)n_erb] = chunk[k];
        }
        done += n;
    }
    if (rc == 0 && (fgetc(fp) != EOF || ferror(fp))) rc = -1;
    fclose(fp);
    return rc;
}

#ifdef __cplusplus
}
#endif

#endif /* DFN2_ERB_FILE_H */
