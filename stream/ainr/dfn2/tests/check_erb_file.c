/* Checks dfn2_stream_init_from_erb_file against dfn2_stream_init: writes a synthetic ERB file,
 * loads it via the file entry point, and confirms the resulting stream's per-hop PCM is
 * byte-identical to the memory-entry point given the same (correctly transposed) tables -- plus
 * two negative cases (dimension mismatch, truncated file). Needs stdio; not built under NO_STDIO. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"

static DFN2Grid make_grid(void) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 48000;
    spec.n_fft = 1024;
    spec.n_erb = 32;
    spec.df_bins = 96;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
    spec.df_lookahead = 0;
#else
    spec.df_order = 5;
    spec.df_lookahead = 1;
#endif
    spec.mask_lookahead = 1;
    spec.frames_per_call = 1;
    spec.erb_norm_tau_sec = 1.0f;
    spec.erb_norm_init_lo_db = -60.0f;
    spec.erb_norm_init_hi_db = -90.0f;
    spec.erb_norm_scale_db = 40.0f;
    spec.spec_norm_tau_sec = 1.0f;
    spec.spec_norm_init_lo = 0.001f;
    spec.spec_norm_init_hi = 0.0001f;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make fail\n"); exit(1); }
    return grid;
}

static int zero_accel(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    size_t i;
    (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.3f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.0f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.0f;
#endif
    for (i = 0; i < out->gru_hidden_elements; ++i) out->gru_hidden_next[i] = 0.0f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->df_convp_window_elements; ++i) out->df_convp_window_next[i] = 0.0f;
#endif
    return 0;
}

static void run_hops(DFN2Stream *s, int hops, int16_t *digest_out) {
    int hop = dfn2_stream_hop_size(s);
    int16_t *pcm_in = calloc((size_t)hop, sizeof(int16_t));
    int16_t *pcm_out = calloc((size_t)hop, sizeof(int16_t));
    int i, k;
    int16_t acc = 0;
    for (i = 0; i < hops; ++i) {
        DFN2HopIO io;
        int ai_rc;
        for (k = 0; k < hop; ++k) pcm_in[k] = (int16_t)((k * 13 + i * 5) % 211 - 100);
        if (dfn2_stream_pre_process(s, pcm_in, &io) != 1) { fprintf(stderr, "pre fail hop %d\n", i); exit(1); }
        ai_rc = zero_accel(&io.inputs, &io.outputs);
        if (dfn2_stream_post_process(s, ai_rc, pcm_out) < 0) { fprintf(stderr, "post fail hop %d\n", i); exit(1); }
        for (k = 0; k < hop; ++k) acc = (int16_t)(acc * 31 + pcm_out[k]);
    }
    *digest_out = acc;
    free(pcm_in);
    free(pcm_out);
}

int main(void) {
    DFN2Grid grid = make_grid();
    size_t n_bins = (size_t)grid.n_bins, n_erb = (size_t)grid.n_erb;
    size_t elems = n_bins * n_erb;
    float *fwd = malloc(elems * sizeof(float));
    float *inv_bin_major = malloc(elems * sizeof(float));
    float *inv_band_major = malloc(elems * sizeof(float));
    size_t i, bin, band;
    const char *path = "/tmp/dfn2_stream_scratch_erb.bin";
    FILE *fp;
    int32_t hdr[2];

    for (i = 0; i < elems; ++i) { fwd[i] = (float)(i % 17) * 0.01f; inv_bin_major[i] = (float)((i + 3) % 11) * 0.02f; }
    for (bin = 0; bin < n_bins; ++bin)
        for (band = 0; band < n_erb; ++band)
            inv_band_major[band * n_bins + bin] = inv_bin_major[bin * n_erb + band];

    /* Write the synthetic file: header + fwd (bin-major) + inv (bin-major). */
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail\n"); return 1; }
    hdr[0] = (int32_t)grid.n_bins; hdr[1] = (int32_t)grid.n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv_bin_major, sizeof(float), elems, fp);
    fclose(fp);

    {
        DFN2StreamConfig cfg_mem, cfg_file;
        DFN2StreamMemReq req_mem, req_file;
        void *pool_mem, *pool_file;
        DFN2Stream *s_mem, *s_file;
        int16_t digest_mem, digest_file;

        memset(&cfg_mem, 0, sizeof(cfg_mem));
        cfg_mem.grid = grid;
        cfg_mem.erb_fwd = fwd;
        cfg_mem.erb_inv = inv_band_major;
        if (dfn2_stream_get_mem_size(&cfg_mem, &req_mem) != 0) { fprintf(stderr, "get_mem_size mem fail\n"); return 1; }
        posix_memalign(&pool_mem, req_mem.alignment, req_mem.bytes);
        s_mem = dfn2_stream_init_ex(&cfg_mem, pool_mem, req_mem.bytes, &req_mem);
        if (!s_mem) { fprintf(stderr, "init_ex mem fail\n"); return 1; }

        memset(&cfg_file, 0, sizeof(cfg_file));
        cfg_file.grid = grid;
        if (dfn2_stream_get_mem_size(&cfg_file, &req_file) != 0) { fprintf(stderr, "get_mem_size file fail\n"); return 1; }
        posix_memalign(&pool_file, req_file.alignment, req_file.bytes);
        s_file = dfn2_stream_init_from_erb_file_ex(&cfg_file, path, pool_file, req_file.bytes, &req_file);
        if (!s_file) { fprintf(stderr, "init_from_erb_file_ex fail\n"); return 1; }

        run_hops(s_mem, 12, &digest_mem);
        run_hops(s_file, 12, &digest_file);
        printf("digest_mem=%d digest_file=%d %s\n", digest_mem, digest_file,
              digest_mem == digest_file ? "MATCH" : "MISMATCH");
        if (digest_mem != digest_file) return 1;

        /* Negative: dimension mismatch. */
        {
            DFN2Grid bad_grid = grid;
            DFN2StreamConfig cfg_bad;
            DFN2StreamMemReq req_bad;
            void *pool_bad;
            bad_grid.n_erb = grid.n_erb; /* keep grid itself valid; corrupt the FILE header instead */
            memset(&cfg_bad, 0, sizeof(cfg_bad));
            cfg_bad.grid = grid;
            dfn2_stream_get_mem_size(&cfg_bad, &req_bad);
            posix_memalign(&pool_bad, req_bad.alignment, req_bad.bytes);
            {
                FILE *bad_fp = fopen("/tmp/dfn2_stream_scratch_erb_bad.bin", "wb");
                int32_t bad_hdr[2] = { (int32_t)grid.n_bins, (int32_t)grid.n_erb + 1 };
                fwrite(bad_hdr, sizeof(int32_t), 2, bad_fp);
                fclose(bad_fp);
            }
            if (dfn2_stream_init_from_erb_file(&cfg_bad, "/tmp/dfn2_stream_scratch_erb_bad.bin",
                                               pool_bad, req_bad.bytes) != NULL) {
                fprintf(stderr, "FAIL: dimension-mismatch file was accepted\n");
                return 1;
            }
            printf("dimension mismatch correctly refused\n");
            free(pool_bad);
        }

        /* Negative: short/truncated file. */
        {
            DFN2StreamConfig cfg_bad;
            DFN2StreamMemReq req_bad;
            void *pool_bad;
            memset(&cfg_bad, 0, sizeof(cfg_bad));
            cfg_bad.grid = grid;
            dfn2_stream_get_mem_size(&cfg_bad, &req_bad);
            posix_memalign(&pool_bad, req_bad.alignment, req_bad.bytes);
            {
                FILE *short_fp = fopen("/tmp/dfn2_stream_scratch_erb_short.bin", "wb");
                int32_t good_hdr[2] = { (int32_t)grid.n_bins, (int32_t)grid.n_erb };
                fwrite(good_hdr, sizeof(int32_t), 2, short_fp);
                fwrite(fwd, sizeof(float), elems / 2, short_fp);   /* truncated */
                fclose(short_fp);
            }
            if (dfn2_stream_init_from_erb_file(&cfg_bad, "/tmp/dfn2_stream_scratch_erb_short.bin",
                                               pool_bad, req_bad.bytes) != NULL) {
                fprintf(stderr, "FAIL: truncated file was accepted\n");
                return 1;
            }
            printf("short read correctly refused\n");
            free(pool_bad);
        }

        free(pool_mem);
        free(pool_file);
    }

    free(fwd); free(inv_bin_major); free(inv_band_major);
    printf("check_erb_file: ALL PASS\n");
    return 0;
}
