/* Checks dfn2_erb_file_load (the sample-side fb_*.bin reader) against dfn2_stream_init: the loaded tables are the
 * written ones (inverse transposed to band-major) and a stream over them runs byte-identically to one over tables built
 * in memory. The loader refuses (-1) a missing file, other dimensions, a truncated file and any byte after the tables;
 * values are the init's to judge: a non-finite forward value or a non-partition inverse loads fine and
 * dfn2_stream_init refuses it, for a full and a freq_only context. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../dfn2_erb_file.h"
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

/* The fb_*.bin of the grid (header n_bins_hdr / n_erb_hdr, bin-major tables) with `trailing` zero bytes appended and,
 * when at < elems, fwd[at] replaced by value; `keep` floats of the forward table only when keep < elems (truncated). */
static void write_variant(const char *path, int n_bins_hdr, int n_erb_hdr, const float *fwd, const float *inv_bin,
                          size_t elems, size_t trailing, size_t at, float value, size_t keep) {
    size_t i;
    int32_t hdr[2];
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail\n"); exit(1); }
    hdr[0] = (int32_t)n_bins_hdr; hdr[1] = (int32_t)n_erb_hdr;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    for (i = 0; i < (keep < elems ? keep : elems); ++i) {
        float v = i == at ? value : fwd[i];
        fwrite(&v, sizeof(float), 1, fp);
    }
    if (keep >= elems) fwrite(inv_bin, sizeof(float), elems, fp);
    for (i = 0; i < trailing; ++i) fputc(0, fp);
    fclose(fp);
}

/* -1: the loader refused the file; 0: loaded, dfn2_stream_init refused the tables; 1: loaded and initialised. */
static int load_and_init(const DFN2Grid *g, int freq_only, const char *path, void *pool, size_t bytes,
                         float *fwd, float *inv) {
    DFN2StreamConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = *g;
    cfg.freq_only = freq_only;
    if (dfn2_erb_file_load(path, g->n_bins, g->n_erb, fwd, inv) != 0) return -1;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    return dfn2_stream_init(&cfg, pool, bytes) != NULL;
}

static int check_defects(const DFN2Grid *g, const float *fwd, const float *inv_bin, const char *path) {
    const size_t elems = (size_t)g->n_bins * (size_t)g->n_erb, none = (size_t)-1;
    const struct { const char *name; size_t trailing, at; float value; int expect; } CASES[] = {
        { "no defect",                    0, none,          0.0f,      1 },
        { "one byte after the tables",    1, none,          0.0f,     -1 },
        { "one float after the tables",   4, none,          0.0f,     -1 },
        { "NaN first forward value",      0, 0,             NAN,       0 },
        { "+Inf last forward value",      0, elems - 1u,    INFINITY,  0 },
        { "-Inf middle forward value",    0, elems / 2u,    -INFINITY, 0 },
        { "NaN last forward value",       0, elems - 1u,    NAN,       0 },
        { "3e38 forward value (finite)",  0, elems - 1u,    3.0e38f,   1 },
        { "no defect, after refusals",    0, none,          0.0f,      1 },
    };
    float *lf = malloc(elems * sizeof(float)), *li = malloc(elems * sizeof(float));
    int kind, fail = 0;
    size_t c;
    if (!lf || !li) { fprintf(stderr, "malloc fail\n"); exit(1); }
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
            write_variant(path, g->n_bins, g->n_erb, fwd, inv_bin, elems, CASES[c].trailing, CASES[c].at,
                          CASES[c].value, elems);
            got = load_and_init(g, kind, path, pool, (size_t)req.bytes, lf, li);
            if (got != CASES[c].expect) {
                fprintf(stderr, "FAIL: %s context, %s: got %d, want %d\n", kind ? "freq_only" : "full",
                        CASES[c].name, got, CASES[c].expect);
                fail = 1;
            }
        }
        remove(path);
        free(pool);
    }
    free(lf); free(li);
    if (!fail) printf("trailing bytes refused by the loader, non-finite forward values by init (full and freq_only)\n");
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

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); return 1; } } while (0)

int main(void) {
    DFN2Grid grid = make_grid();
    size_t n_bins = (size_t)grid.n_bins, n_erb = (size_t)grid.n_erb;
    size_t elems = n_bins * n_erb;
    float *fwd = malloc(elems * sizeof(float));
    float *inv_bin_major = malloc(elems * sizeof(float));
    float *inv_band_major = malloc(elems * sizeof(float));
    float *lfwd = malloc(elems * sizeof(float));
    float *linv = malloc(elems * sizeof(float));
    size_t i, bin, band;
    /* PID-suffixed, not a fixed name: concurrent runs must not interleave writes/reads of one /tmp path. */
    char path[64];
    DFN2StreamConfig cfg_mem, cfg_file;
    DFN2StreamMemReq req_mem, req_file;
    void *pool_mem = NULL, *pool_file = NULL;
    DFN2Stream *s_mem, *s_file;
    int16_t digest_mem, digest_file;

    if (!fwd || !inv_bin_major || !inv_band_major || !lfwd || !linv) { fprintf(stderr, "malloc fail\n"); return 1; }
    erb_fixture_scratch_path(path, sizeof(path), "dfn2_stream_scratch_erb");

    for (i = 0; i < elems; ++i) { fwd[i] = (float)(i % 17) * 0.01f; }
    erb_fixture_split_inv_bin_major(inv_bin_major, (int)n_bins, (int)n_erb);
    for (bin = 0; bin < n_bins; ++bin)
        for (band = 0; band < n_erb; ++band)
            inv_band_major[band * n_bins + bin] = inv_bin_major[bin * n_erb + band];

    /* The loaded tables are the written ones: forward as is, inverse transposed to band-major. */
    write_variant(path, grid.n_bins, grid.n_erb, fwd, inv_bin_major, elems, 0, (size_t)-1, 0.0f, elems);
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, lfwd, linv) == 0, "loader refused the good file");
    CHECK(memcmp(lfwd, fwd, elems * sizeof(float)) == 0, "forward table differs from the file");
    CHECK(memcmp(linv, inv_band_major, elems * sizeof(float)) == 0, "inverse is not the band-major transpose");
    printf("loader: tables match (forward as is, inverse transposed)\n");

    /* A stream over the loaded tables runs byte-identically to one over tables built in memory. */
    memset(&cfg_mem, 0, sizeof(cfg_mem));
    cfg_mem.grid = grid;
    cfg_mem.erb_fwd = fwd;
    cfg_mem.erb_inv = inv_band_major;
    CHECK(dfn2_stream_get_mem_size(&cfg_mem, &req_mem) == 0, "get_mem_size mem");
    CHECK(posix_memalign(&pool_mem, req_mem.alignment, req_mem.bytes) == 0, "alloc");
    s_mem = dfn2_stream_init_ex(&cfg_mem, pool_mem, req_mem.bytes, &req_mem);
    CHECK(s_mem != NULL, "init_ex over memory tables");

    memset(&cfg_file, 0, sizeof(cfg_file));
    cfg_file.grid = grid;
    cfg_file.erb_fwd = lfwd;
    cfg_file.erb_inv = linv;
    CHECK(dfn2_stream_get_mem_size(&cfg_file, &req_file) == 0, "get_mem_size file");
    CHECK(posix_memalign(&pool_file, req_file.alignment, req_file.bytes) == 0, "alloc");
    s_file = dfn2_stream_init_ex(&cfg_file, pool_file, req_file.bytes, &req_file);
    CHECK(s_file != NULL, "init_ex over loaded tables");

    run_hops(s_mem, 12, &digest_mem);
    run_hops(s_file, 12, &digest_file);
    printf("digest_mem=%d digest_file=%d %s\n", digest_mem, digest_file, digest_mem == digest_file ? "MATCH" : "MISMATCH");
    if (digest_mem != digest_file) return 1;
    free(pool_mem);
    free(pool_file);

    /* Loader refusals. */
    write_variant(path, grid.n_bins, grid.n_erb + 1, fwd, inv_bin_major, elems, 0, (size_t)-1, 0.0f, elems);
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, lfwd, linv) == -1, "other n_erb accepted");
    write_variant(path, grid.n_bins + 1, grid.n_erb, fwd, inv_bin_major, elems, 0, (size_t)-1, 0.0f, elems);
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, lfwd, linv) == -1, "other n_bins accepted");
    write_variant(path, grid.n_bins, grid.n_erb, fwd, inv_bin_major, elems, 0, (size_t)-1, 0.0f, elems);
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb + 1, lfwd, linv) == -1, "caller dims differ from the file's");
    printf("dimension mismatch correctly refused\n");
    write_variant(path, grid.n_bins, grid.n_erb, fwd, inv_bin_major, elems, 0, (size_t)-1, 0.0f, elems / 2);
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, lfwd, linv) == -1, "truncated forward table accepted");
    write_variant(path, grid.n_bins, grid.n_erb, fwd, inv_bin_major, elems, 0, (size_t)-1, 0.0f, elems);
    {   /* truncated inside the inverse table: drop the last float */
        FILE *fp = fopen(path, "rb+");
        long end;
        CHECK(fp != NULL, "reopen");
        fseek(fp, 0, SEEK_END);
        end = ftell(fp);
        fclose(fp);
        CHECK(truncate(path, (off_t)(end - 4)) == 0, "truncate");
    }
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, lfwd, linv) == -1, "file short by one float accepted");
    printf("short read correctly refused\n");
    remove(path);
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, lfwd, linv) == -1, "missing file accepted");
    CHECK(dfn2_erb_file_load(NULL, grid.n_bins, grid.n_erb, lfwd, linv) == -1, "NULL path accepted");
    CHECK(dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, NULL, linv) == -1, "NULL forward buffer accepted");
    CHECK(dfn2_erb_file_load(path, 0, grid.n_erb, lfwd, linv) == -1, "n_bins 0 accepted");
    printf("missing file and NULL arguments correctly refused\n");

    if (check_defects(&grid, fwd, inv_bin_major, path)) return 1;

    /* An inverse that is not a partition of unity loads fine; dfn2_stream_init refuses it. */
    {
        float *bad_inv = malloc(elems * sizeof(float));
        DFN2StreamConfig cfg;
        DFN2StreamMemReq req;
        void *pool = NULL;
        CHECK(bad_inv != NULL, "malloc");
        for (i = 0; i < elems; ++i) bad_inv[i] = (float)((i + 3) % 11) * 0.02f;
        write_variant(path, grid.n_bins, grid.n_erb, fwd, bad_inv, elems, 0, (size_t)-1, 0.0f, elems);
        memset(&cfg, 0, sizeof(cfg));
        cfg.grid = grid;
        CHECK(dfn2_stream_get_mem_size(&cfg, &req) == 0, "get_mem_size");
        CHECK(posix_memalign(&pool, req.alignment, req.bytes) == 0, "alloc");
        CHECK(load_and_init(&grid, 0, path, pool, (size_t)req.bytes, lfwd, linv) == 0,
              "non-partition inverse: expected loaded then refused by init");
        remove(path);
        free(pool); free(bad_inv);
        printf("non-partition inverse loads, init refuses\n");
    }

    free(fwd); free(inv_bin_major); free(inv_band_major); free(lfwd); free(linv);
    printf("check_erb_file: ALL PASS\n");
    return 0;
}
