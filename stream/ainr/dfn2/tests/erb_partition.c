/* dfn2_stream init refuses an inverse ERB table that is not a partition of unity: a unit band mask
 * must expand to exactly 1.0f in every bin (summed band-ascending in float32), the property the
 * fail-open identity is exact under. Per grid row, in memory mode (and in file mode unless
 * -DDFN2_STREAM_NO_STDIO):
 *   - the two partition tables (band 0 owns every bin; every bin split 0.75 / 0.25) are accepted;
 *   - each broken table is refused by init and init_ex with NULL: every band owning every bin, all
 *     zero, one bin summing to 0 / 1 + ulp / 1 - ulp / 1.5, a NaN, an Inf;
 *   - get_mem_size() is the same for a good and a broken table (the table is not part of the pool
 *     ABI), and the pool a refusal left behind takes a good table at once.
 * With arguments `SAMPLE_RATE FFT N_ERB FILE`, loads FILE (the exporter's erb_matrix.bin layout) by
 * the file entry point instead and reports whether it is accepted (exit 1 when refused). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dfn2_stream.h"
#include "erb_fixture.h"

typedef struct Row { int sr, fft, n_erb; } Row;

static const Row ROWS[] = {
    { 16000, 256, 24 },
    { 32000, 512, 28 },
    { 48000, 1024, 32 },
};

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

static DFN2Grid make_grid(int sr, int fft, int n_erb) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = sr;
    spec.n_fft = fft;
    spec.n_erb = n_erb;
    spec.df_bins = 96;
    spec.mask_lookahead = 1;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
    spec.df_lookahead = 0;
#else
    spec.df_order = 5;
    spec.df_lookahead = 1;
#endif
    spec.frames_per_call = 1;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make fail\n"); exit(2); }
    return grid;
}

enum { BAD_ALL_ONES, BAD_ALL_ZERO, BAD_BIN_ZERO, BAD_BIN_PLUS_ULP, BAD_BIN_MINUS_ULP, BAD_LAST_BAND_HALF,
       BAD_NAN, BAD_INF, BAD_COUNT };
static const char *const BAD_NAME[BAD_COUNT] = {
    "every band owns every bin", "all zero", "last bin sums to 0", "last bin sums to 1 + ulp",
    "first bin sums to 1 - ulp", "bin 5 gains 0.5 in the last band", "NaN", "Inf" };

/* inv: band-major [n_erb][n_bins], filled as a unit partition and then broken one way. */
static void make_bad(float *inv, int n_bins, int n_erb, int kind) {
    size_t i, total = (size_t)n_bins * (size_t)n_erb;
    erb_fixture_unit_inv(inv, n_bins, n_erb);
    switch (kind) {
    case BAD_ALL_ONES: for (i = 0; i < total; ++i) inv[i] = 1.0f; break;
    case BAD_ALL_ZERO: for (i = 0; i < total; ++i) inv[i] = 0.0f; break;
    case BAD_BIN_ZERO: inv[n_bins - 1] = 0.0f; break;
    case BAD_BIN_PLUS_ULP: inv[n_bins - 1] = nextafterf(1.0f, 2.0f); break;
    case BAD_BIN_MINUS_ULP: inv[0] = nextafterf(1.0f, 0.0f); break;
    case BAD_LAST_BAND_HALF: inv[(size_t)(n_erb - 1) * (size_t)n_bins + 5u] = 0.5f; break;
    case BAD_NAN: inv[n_bins / 2] = (float)NAN; break;
    default: inv[n_bins / 2] = (float)INFINITY; break;   /* BAD_INF */
    }
}

static DFN2StreamConfig make_cfg(const DFN2Grid *grid, const float *fwd, const float *inv) {
    DFN2StreamConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = *grid;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    return cfg;
}

#ifndef DFN2_STREAM_NO_STDIO
/* The file entry point's layout: int32 n_bins, n_erb, forward and inverse both bin-major. */
static int write_file(const char *path, const DFN2Grid *grid, const float *inv_band_major) {
    size_t n_bins = (size_t)grid->n_bins, n_erb = (size_t)grid->n_erb, bin, band;
    float *fwd = (float *)calloc(n_bins * n_erb, sizeof(float));
    float *inv = (float *)malloc(n_bins * n_erb * sizeof(float));
    int32_t hdr[2];
    FILE *fp;
    if (!fwd || !inv) exit(2);
    for (bin = 0; bin < n_bins; ++bin)
        for (band = 0; band < n_erb; ++band) inv[bin * n_erb + band] = inv_band_major[band * n_bins + bin];
    fp = fopen(path, "wb");
    if (!fp) exit(2);
    hdr[0] = grid->n_bins; hdr[1] = grid->n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), n_bins * n_erb, fp);
    fwrite(inv, sizeof(float), n_bins * n_erb, fp);
    fclose(fp);
    free(fwd); free(inv);
    return 0;
}
#endif

static void run_row(const Row *r) {
    DFN2Grid grid = make_grid(r->sr, r->fft, r->n_erb);
    size_t n_bins = (size_t)grid.n_bins, n_erb = (size_t)grid.n_erb, elems = n_bins * n_erb, i;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *good_unit = (float *)malloc(elems * sizeof(float));
    float *good_split = (float *)malloc(elems * sizeof(float));
    float *bad = (float *)malloc(elems * sizeof(float));
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req_good, req_bad;
    void *pool = NULL;
    int kind;

    if (!fwd || !good_unit || !good_split || !bad) exit(2);
    for (i = 0; i < elems; ++i) fwd[i] = 1.0f / (float)n_erb;
    erb_fixture_unit_inv(good_unit, grid.n_bins, grid.n_erb);
    erb_fixture_split_inv_band_major(good_split, grid.n_bins, grid.n_erb);

    cfg = make_cfg(&grid, fwd, good_unit);
    CHECK(dfn2_stream_get_mem_size(&cfg, &req_good) == 0, "%d/%d: sizing", r->sr, r->fft);
    if (posix_memalign(&pool, 16, (size_t)req_good.bytes) != 0) exit(2);

    cfg = make_cfg(&grid, fwd, good_unit);
    CHECK(dfn2_stream_init_ex(&cfg, pool, (size_t)req_good.bytes, &req_good) != NULL,
          "%d/%d: the unit partition is refused", r->sr, r->fft);
    cfg = make_cfg(&grid, fwd, good_split);
    CHECK(dfn2_stream_init(&cfg, pool, (size_t)req_good.bytes) != NULL,
          "%d/%d: the 0.75 / 0.25 partition is refused", r->sr, r->fft);

    for (kind = 0; kind < BAD_COUNT; ++kind) {
        make_bad(bad, grid.n_bins, grid.n_erb, kind);
        cfg = make_cfg(&grid, fwd, bad);
        CHECK(dfn2_stream_get_mem_size(&cfg, &req_bad) == 0 && req_bad.bytes == req_good.bytes &&
              req_bad.build_flags_hash == req_good.build_flags_hash,
              "%d/%d (%s): pool descriptor depends on the table", r->sr, r->fft, BAD_NAME[kind]);
        CHECK(dfn2_stream_init(&cfg, pool, (size_t)req_good.bytes) == NULL,
              "%d/%d (%s): init accepted the table", r->sr, r->fft, BAD_NAME[kind]);
        CHECK(dfn2_stream_init_ex(&cfg, pool, (size_t)req_good.bytes, &req_good) == NULL,
              "%d/%d (%s): init_ex accepted the table", r->sr, r->fft, BAD_NAME[kind]);
        cfg = make_cfg(&grid, fwd, good_split);
        CHECK(dfn2_stream_init(&cfg, pool, (size_t)req_good.bytes) != NULL,
              "%d/%d (%s): the pool a refusal left behind is not usable", r->sr, r->fft, BAD_NAME[kind]);
    }

#ifndef DFN2_STREAM_NO_STDIO
    {
        char path[96];
        DFN2StreamConfig fcfg = make_cfg(&grid, NULL, NULL);
        DFN2StreamMemReq freq;
        void *fpool = NULL;
        snprintf(path, sizeof(path), "/tmp/dfn2_erb_partition_%d_%d.bin", r->fft, (int)getpid());
        CHECK(dfn2_stream_get_mem_size(&fcfg, &freq) == 0, "%d/%d: file sizing", r->sr, r->fft);
        if (posix_memalign(&fpool, 16, (size_t)freq.bytes) != 0) exit(2);
        write_file(path, &grid, good_split);
        CHECK(dfn2_stream_init_from_erb_file(&fcfg, path, fpool, (size_t)freq.bytes) != NULL,
              "%d/%d: a partition file is refused", r->sr, r->fft);
        for (kind = 0; kind < BAD_COUNT; ++kind) {
            make_bad(bad, grid.n_bins, grid.n_erb, kind);
            write_file(path, &grid, bad);
            CHECK(dfn2_stream_init_from_erb_file(&fcfg, path, fpool, (size_t)freq.bytes) == NULL,
                  "%d/%d (%s): init_from_erb_file accepted the file", r->sr, r->fft, BAD_NAME[kind]);
            CHECK(dfn2_stream_init_from_erb_file_ex(&fcfg, path, fpool, (size_t)freq.bytes, &freq) == NULL,
                  "%d/%d (%s): init_from_erb_file_ex accepted the file", r->sr, r->fft, BAD_NAME[kind]);
        }
        unlink(path);
        free(fpool);
    }
#endif
    free(pool); free(fwd); free(good_unit); free(good_split); free(bad);
}

#ifndef DFN2_STREAM_NO_STDIO
static int check_file(int sr, int fft, int n_erb, const char *path) {
    DFN2Grid grid = make_grid(sr, fft, n_erb);
    DFN2StreamConfig cfg = make_cfg(&grid, NULL, NULL);
    DFN2StreamMemReq req;
    void *pool = NULL;
    DFN2Stream *s;
    if (dfn2_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&pool, 16, (size_t)req.bytes) != 0) return 2;
    s = dfn2_stream_init_from_erb_file(&cfg, path, pool, (size_t)req.bytes);
    printf("%s (%d Hz / FFT %d / %d bands): %s\n", path, sr, fft, n_erb, s ? "accepted" : "REFUSED");
    free(pool);
    return s ? 0 : 1;
}
#endif

int main(int argc, char **argv) {
    size_t i;
#ifndef DFN2_STREAM_NO_STDIO
    if (argc == 5) return check_file(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4]);
#else
    (void)argc; (void)argv;
#endif
    for (i = 0; i < sizeof(ROWS) / sizeof(ROWS[0]); ++i) run_row(&ROWS[i]);
    if (g_fail) { printf("erb_partition: %d check(s) FAILED\n", g_fail); return 1; }
    printf("erb_partition: %d grid(s), %d broken tables each -- PASS\n", (int)(sizeof(ROWS) / sizeof(ROWS[0])), BAD_COUNT);
    return 0;
}
