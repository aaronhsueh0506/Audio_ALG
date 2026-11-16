/* Checks dfn2_stream_init_from_erb_file against dfn2_stream_init: writes a synthetic ERB file,
 * loads it via the file entry point, and confirms the resulting stream's per-hop PCM is
 * byte-identical to the memory-entry point given the same (correctly transposed) tables -- plus
 * the refusals: dimension mismatch, truncated file, bytes after the two tables and a forward value that
 * is not finite, each against the same file without the defect, for a full and a freq_only context.
 * Needs stdio; not built under NO_STDIO. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dfn2_stream.h"
#include "erb_fixture.h"

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
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make fail\n"); exit(1); }
    return grid;
}

static int zero_accel(DFN2PrepostOutputs *out) {
    size_t i;
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

/* The test file with `trailing` zero bytes appended and, when at < elems, fwd[at] replaced by value. */
static void write_variant(const char *path, const DFN2Grid *g, const float *fwd, const float *inv_bin,
                          size_t trailing, size_t at, float value) {
    size_t elems = (size_t)g->n_bins * (size_t)g->n_erb, i;
    int32_t hdr[2];
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail\n"); exit(1); }
    hdr[0] = (int32_t)g->n_bins; hdr[1] = (int32_t)g->n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    for (i = 0; i < elems; ++i) {
        float v = i == at ? value : fwd[i];
        fwrite(&v, sizeof(float), 1, fp);
    }
    fwrite(inv_bin, sizeof(float), elems, fp);
    for (i = 0; i < trailing; ++i) fputc(0, fp);
    fclose(fp);
}

/* 1 when the file initialises a stream (full or freq_only) in `pool`. */
static int loads(const DFN2Grid *g, int freq_only, const char *path, void *pool, size_t bytes) {
    DFN2StreamConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = *g;
    cfg.freq_only = freq_only;
    return dfn2_stream_init_from_erb_file(&cfg, path, pool, bytes) != NULL;
}

static int check_defects(const DFN2Grid *g, const float *fwd, const float *inv_bin, const char *path) {
    const size_t elems = (size_t)g->n_bins * (size_t)g->n_erb, none = (size_t)-1;
    const struct { const char *name; size_t trailing, at; float value; int ok; } CASES[] = {
        { "no defect",                    0, none,          0.0f,      1 },
        { "one byte after the tables",    1, none,          0.0f,      0 },
        { "one float after the tables",   4, none,          0.0f,      0 },
        { "NaN first forward value",      0, 0,             NAN,       0 },
        { "+Inf last forward value",      0, elems - 1u,    INFINITY,  0 },
        { "-Inf middle forward value",    0, elems / 2u,    -INFINITY, 0 },
        { "NaN last forward value",       0, elems - 1u,    NAN,       0 },
        { "3e38 forward value (finite)",  0, elems - 1u,    3.0e38f,   1 },
        { "no defect, after refusals",    0, none,          0.0f,      1 },
    };
    int kind, fail = 0;
    size_t c;
    for (kind = 0; kind < 2; ++kind) {
        DFN2StreamConfig cfg;
        DFN2StreamMemReq req;
        void *pool = NULL;
        memset(&cfg, 0, sizeof(cfg));
        cfg.grid = *g;
        cfg.freq_only = kind;
        if (dfn2_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) {
            fprintf(stderr, "defects: sizing fail\n");
            return 1;
        }
        for (c = 0; c < sizeof(CASES) / sizeof(CASES[0]); ++c) {
            int got;
            write_variant(path, g, fwd, inv_bin, CASES[c].trailing, CASES[c].at, CASES[c].value);
            got = loads(g, kind, path, pool, (size_t)req.bytes);
            if (got != CASES[c].ok) {
                fprintf(stderr, "FAIL: %s context, %s: %s\n", kind ? "freq_only" : "full", CASES[c].name,
                        got ? "accepted" : "refused");
                fail = 1;
            }
        }
        remove(path);
        free(pool);
    }
    if (!fail) printf("trailing bytes and non-finite forward values correctly refused (full and freq_only)\n");
    return fail;
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
        ai_rc = io.action == STREAM_WORK_SUBMIT ? zero_accel(&io.outputs) : 0;
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
    /* PID-suffixed, not a fixed name: two concurrent runs of this test (parallel make -jN across
     * sibling model dirs, two CI jobs on one shared build host) must not interleave writes/reads
     * of the same /tmp path and produce a spurious MISMATCH unrelated to the code under test. */
    char path[64], bad_path[64], short_path[64];
    FILE *fp;
    int32_t hdr[2];

    erb_fixture_scratch_path(path, sizeof(path), "dfn2_stream_scratch_erb");
    erb_fixture_scratch_path(bad_path, sizeof(bad_path), "dfn2_stream_scratch_erb_bad");
    erb_fixture_scratch_path(short_path, sizeof(short_path), "dfn2_stream_scratch_erb_short");

    for (i = 0; i < elems; ++i) { fwd[i] = (float)(i % 17) * 0.01f; }
    erb_fixture_split_inv_bin_major(inv_bin_major, (int)n_bins, (int)n_erb);
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
        remove(path);
        if (!s_file) { fprintf(stderr, "init_from_erb_file_ex fail\n"); return 1; }

        run_hops(s_mem, 12, &digest_mem);
        run_hops(s_file, 12, &digest_file);
        printf("digest_mem=%d digest_file=%d %s\n", digest_mem, digest_file,
              digest_mem == digest_file ? "MATCH" : "MISMATCH");
        if (digest_mem != digest_file) return 1;

        /* Negative: dimension mismatch. */
        {
            DFN2StreamConfig cfg_bad;
            DFN2StreamMemReq req_bad;
            void *pool_bad;
            memset(&cfg_bad, 0, sizeof(cfg_bad));
            cfg_bad.grid = grid;
            dfn2_stream_get_mem_size(&cfg_bad, &req_bad);
            posix_memalign(&pool_bad, req_bad.alignment, req_bad.bytes);
            {
                FILE *bad_fp = fopen(bad_path, "wb");
                /* keep grid itself valid; corrupt the FILE header instead */
                int32_t bad_hdr[2] = { (int32_t)grid.n_bins, (int32_t)grid.n_erb + 1 };
                fwrite(bad_hdr, sizeof(int32_t), 2, bad_fp);
                fclose(bad_fp);
            }
            if (dfn2_stream_init_from_erb_file(&cfg_bad, bad_path, pool_bad, req_bad.bytes) != NULL) {
                fprintf(stderr, "FAIL: dimension-mismatch file was accepted\n");
                remove(bad_path);
                return 1;
            }
            remove(bad_path);
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
                FILE *short_fp = fopen(short_path, "wb");
                int32_t good_hdr[2] = { (int32_t)grid.n_bins, (int32_t)grid.n_erb };
                fwrite(good_hdr, sizeof(int32_t), 2, short_fp);
                fwrite(fwd, sizeof(float), elems / 2, short_fp);   /* truncated */
                fclose(short_fp);
            }
            if (dfn2_stream_init_from_erb_file(&cfg_bad, short_path, pool_bad, req_bad.bytes) != NULL) {
                fprintf(stderr, "FAIL: truncated file was accepted\n");
                remove(short_path);
                return 1;
            }
            remove(short_path);
            printf("short read correctly refused\n");
            free(pool_bad);
        }

        if (check_defects(&grid, fwd, inv_bin_major, bad_path)) return 1;

        free(pool_mem);
        free(pool_file);
    }

    free(fwd); free(inv_bin_major); free(inv_band_major);
    printf("check_erb_file: ALL PASS\n");
    return 0;
}
