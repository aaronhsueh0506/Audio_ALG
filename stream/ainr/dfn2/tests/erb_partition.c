/* dfn2_stream init refuses an inverse ERB table that is not a partition of unity: a unit band mask
 * must expand to exactly 1.0f in every bin (summed band-ascending in float32), the property the
 * fail-open identity is exact under. Per grid row:
 *   - the two partition tables (band 0 owns every bin; every bin split 0.75 / 0.25) are accepted;
 *   - each broken table is refused by init and init_ex with NULL: every band owning every bin, all
 *     zero, one bin summing to 0 / 1 + ulp / 1 - ulp / 1.5, a NaN, an Inf, and two bins that sum to exactly 1.0f
 *     with entries outside [0, 1] (1e30 + -1e30 + 1, 2 + -1);
 *   - a forward table with a NaN, an Inf or a negative entry is refused;
 *   - get_mem_size() is the same for a good and a broken table (the table is not part of the pool
 *     ABI), and the pool a refusal left behind takes a good table at once.
 * With arguments `SAMPLE_RATE FFT N_ERB FILE`, loads FILE (an fb_*.bin) with dfn2_erb_file_load and reports
 * whether init accepts it (exit 1 when refused). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../dfn2_erb_file.h"
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
       BAD_NAN, BAD_INF, BAD_HUGE_PAIR, BAD_OUT_OF_RANGE, BAD_COUNT };
static const char *const BAD_NAME[BAD_COUNT] = {
    "every band owns every bin", "all zero", "last bin sums to 0", "last bin sums to 1 + ulp",
    "first bin sums to 1 - ulp", "bin 5 gains 0.5 in the last band", "NaN", "Inf", "1e30 + -1e30 + 1 in one bin",
    "2 + -1 in one bin" };

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
    case BAD_INF: inv[n_bins / 2] = (float)INFINITY; break;
    case BAD_HUGE_PAIR:
        inv[n_bins / 2] = 1e30f; inv[(size_t)n_bins + (size_t)(n_bins / 2)] = -1e30f;
        inv[2u * (size_t)n_bins + (size_t)(n_bins / 2)] = 1.0f; break;
    default:   /* BAD_OUT_OF_RANGE */
        inv[n_bins / 2] = 2.0f; inv[(size_t)n_bins + (size_t)(n_bins / 2)] = -1.0f; break;
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

    {   /* the forward table: finite and non-negative */
        const float bad_fwd[3] = { (float)NAN, (float)INFINITY, -0.5f };
        for (kind = 0; kind < 3; ++kind) {
            float keep = fwd[elems / 2];
            fwd[elems / 2] = bad_fwd[kind];
            cfg = make_cfg(&grid, fwd, good_unit);
            CHECK(dfn2_stream_init(&cfg, pool, (size_t)req_good.bytes) == NULL,
                  "%d/%d: init accepted a forward table with %g", r->sr, r->fft, (double)bad_fwd[kind]);
            fwd[elems / 2] = keep;
        }
        cfg = make_cfg(&grid, fwd, good_unit);
        CHECK(dfn2_stream_init(&cfg, pool, (size_t)req_good.bytes) != NULL, "%d/%d: the restored table is refused",
              r->sr, r->fft);
    }

    free(pool); free(fwd); free(good_unit); free(good_split); free(bad);
}

static int check_file(int sr, int fft, int n_erb, const char *path) {
    DFN2Grid grid = make_grid(sr, fft, n_erb);
    DFN2StreamConfig cfg = make_cfg(&grid, NULL, NULL);
    DFN2StreamMemReq req;
    float *fwd = (float *)malloc((size_t)grid.n_bins * (size_t)grid.n_erb * sizeof(float));
    float *inv = (float *)malloc((size_t)grid.n_bins * (size_t)grid.n_erb * sizeof(float));
    void *pool = NULL;
    DFN2Stream *s = NULL;
    int loaded;
    if (!fwd || !inv || dfn2_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&pool, 16, (size_t)req.bytes) != 0)
        return 2;
    loaded = dfn2_erb_file_load(path, grid.n_bins, grid.n_erb, fwd, inv) == 0;
    if (loaded) {
        cfg = make_cfg(&grid, fwd, inv);
        s = dfn2_stream_init(&cfg, pool, (size_t)req.bytes);
    }
    printf("%s (%d Hz / FFT %d / %d bands): %s\n", path, sr, fft, n_erb,
           s ? "accepted" : loaded ? "REFUSED by init" : "REFUSED by the loader");
    free(pool); free(fwd); free(inv);
    return s ? 0 : 1;
}

int main(int argc, char **argv) {
    size_t i;
    if (argc == 5) return check_file(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4]);
    for (i = 0; i < sizeof(ROWS) / sizeof(ROWS[0]); ++i) run_row(&ROWS[i]);
    if (g_fail) { printf("erb_partition: %d check(s) FAILED\n", g_fail); return 1; }
    printf("erb_partition: %d grid(s), %d broken tables each -- PASS\n", (int)(sizeof(ROWS) / sizeof(ROWS[0])), BAD_COUNT);
    return 0;
}
