/* The only <stdio.h> TU in stream/ainr/dfn2/: dfn2_stream_init_from_erb_file[_ex]. Mirrors
 * load_erb_matrix() / build_erb_inv_band_major() in the board samples (alg_ai3_dfn.c) exactly:
 * int32 n_bins, int32 n_erb header, then n_bins*n_erb float32 forward (bin-major, kept as read)
 * and n_bins*n_erb float32 inverse (bin-major as the file stores it, transposed to band-major
 * here, as the library's erb_inv argument requires). Link without this file for a NO_STDIO
 * build; dfn2_stream_init() (memory mode) needs nothing from here. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dfn2_stream_internal.h"

/* Reads exactly n_bins*n_erb floats into fwd_out (kept bin-major) and n_bins*n_erb floats into
 * scratch_out (bin-major, as the file stores it -- transposed by the caller). 0, or -1 after
 * closing the file on an open/header/short-read/dimension-mismatch failure. */
static int load_erb_file(const char *path, const DFN2Grid *grid,
                         float *fwd_out, float *scratch_out) {
    FILE *fp;
    int32_t hdr[2];
    size_t want = (size_t)grid->n_bins * (size_t)grid->n_erb;
    size_t got;

    fp = fopen(path, "rb");
    if (!fp) return -1;

    if (fread(hdr, sizeof(int32_t), 2, fp) != 2) {
        fclose(fp);
        return -1;
    }
    if (hdr[0] != (int32_t)grid->n_bins || hdr[1] != (int32_t)grid->n_erb) {
        fclose(fp);
        return -1;
    }
    got = fread(fwd_out, sizeof(float), want, fp);
    if (got != want) {
        fclose(fp);
        return -1;
    }
    got = fread(scratch_out, sizeof(float), want, fp);
    fclose(fp);
    if (got != want) return -1;
    return 0;
}

/* bin-major [n_bins][n_erb] -> band-major [n_erb][n_bins]; mirrors build_erb_inv_band_major(). */
static void transpose_band_major(const float *bin_major, float *band_major,
                                 int n_bins, int n_erb) {
    int bin, band;
    for (bin = 0; bin < n_bins; ++bin) {
        for (band = 0; band < n_erb; ++band) {
            band_major[(size_t)band * (size_t)n_bins + (size_t)bin] =
                bin_major[(size_t)bin * (size_t)n_erb + (size_t)band];
        }
    }
}

DFN2Stream *dfn2_stream_init_from_erb_file_ex(const DFN2StreamConfig *cfg, const char *erb_path,
                                              void *pool, size_t pool_bytes,
                                              const DFN2StreamMemReq *expected) {
    DFN2StreamLayout layout;

    if (!erb_path || !dfn2_stream_is_file_mode(cfg)) return NULL;
    if (dfn2_stream_prepare_pool(cfg, pool, pool_bytes, expected, &layout) != 0) return NULL;

    /* Loaded straight into pool-owned storage; a short read or mismatch below returns NULL with no
     * DFN2Stream ever constructed (dfn2_stream_finish_init() is never reached), though the
     * erb_fwd_storage/erb_scratch bytes themselves may hold a partial read -- harmless, since no
     * live object is returned and a correct retry overwrites them with a full fread(). */
    if (load_erb_file(erb_path, &cfg->grid, layout.erb_fwd_storage, layout.erb_scratch) != 0)
        return NULL;
    transpose_band_major(layout.erb_scratch, layout.erb_inv_storage,
                         cfg->grid.n_bins, cfg->grid.n_erb);

    return dfn2_stream_finish_init(cfg, (unsigned char *)pool, &layout,
                                   layout.erb_fwd_storage, layout.erb_inv_storage);
}

DFN2Stream *dfn2_stream_init_from_erb_file(const DFN2StreamConfig *cfg, const char *erb_path,
                                           void *pool, size_t pool_bytes) {
    return dfn2_stream_init_from_erb_file_ex(cfg, erb_path, pool, pool_bytes, NULL);
}
