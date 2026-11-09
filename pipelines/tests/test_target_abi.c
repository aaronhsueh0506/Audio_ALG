/* Run on each ABI, then exchange --write files with --reject. The foreign
 * byte budget is normalized so rejection specifically exercises the hash.
 * Files are exchanged between builds with the same DFN2_TEST_GRID_KHZ and lookahead pair. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifdef NDEBUG
#error "target ABI regression requires assertions"
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mcab.h"
#include "dfn/tests/dfn_test_grid.h"
#include "dfn/dfn_res_stage.h"
#include "dfn/dfn_rate_bridge.h"
#include "dfn2_prepost.h"
#include "dfn3_prepost.h"
#include "ulcnet_prepost.h"
#include "deepvqe_prepost.h"
#include "dfn/dfn_unit_partition.h"

static float erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static FILE *write_file, *foreign_file;
static unsigned cases;

#define EXERCISE(prefix, Req, cfg) do { \
    Req req, foreign; \
    void *pool = NULL; \
    assert(sizeof(Req) == 32u); \
    assert(prefix##_get_mem_requirements(&(cfg), &req) == 0); \
    assert(req.bytes <= (uint64_t)SIZE_MAX); \
    assert(posix_memalign(&pool, req.alignment, (size_t)req.bytes) == 0); \
    assert(prefix##_init_ex(pool, (size_t)req.bytes, &(cfg), &req)); \
    if (write_file) assert(fwrite(&req, sizeof(req), 1, write_file) == 1); \
    if (foreign_file) { \
        assert(fread(&foreign, sizeof(foreign), 1, foreign_file) == 1); \
        assert(foreign.build_flags_hash != req.build_flags_hash); \
        foreign.bytes = req.bytes; \
        memset(pool, 0xa5, (size_t)req.bytes); \
        assert(!prefix##_init_ex(pool, (size_t)req.bytes, &(cfg), &foreign)); \
        for (size_t i = 0; i < (size_t)req.bytes; ++i) \
            assert(((const unsigned char *)pool)[i] == 0xa5); \
    } \
    printf("%s ptr%u hash%08x bytes%llu\n", #prefix, \
           (unsigned)sizeof(void *), req.build_flags_hash, \
           (unsigned long long)req.bytes); \
    ++cases; \
    free(pool); \
} while (0)

/* mcab's init takes the objects it drives: one set of lanes and the shared estimator, built
 * once in main() with their own libaec init calls, serves both mcab cases. */
static McabLibs g_mcab_libs;
static Mcab *mcab_case_init_ex(void *mem, size_t bytes, const McabConfig *cfg,
                               const McabMemReq *req) {
    return mcab_init_ex(mem, bytes, cfg, &g_mcab_libs, req);
}
#define mcab_case_get_mem_requirements mcab_get_mem_requirements

static void build_mcab_libs(const McabConfig *cfg) {
    AecConfig lane;
    McabDelayEstimatorConfig dc;
    size_t bytes;
    static DelayAec delay;
    void *mem;
    int ch;
    assert(mcab_lane_config(cfg, &lane) == 0 && mcab_delay_estimator_config(cfg, &dc) == 0);
    bytes = aec_get_mem_size(&lane);
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        assert(posix_memalign(&mem, 16, bytes) == 0);
        assert((g_mcab_libs.lanes[ch] = aec_init(mem, bytes, &lane)) != NULL);
    }
    assert(dc.required);
    bytes = delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters);
    assert(posix_memalign(&mem, 16, bytes) == 0);
    assert(delay_aec_init(&delay, mem, bytes, dc.sample_rate, dc.hop_size, dc.num_filters) == 0);
    g_mcab_libs.delay = &delay;
}

/* The grids the descriptors are taken for: the DFN3 class at the DFN2 test grid's numbers (the same
 * config.ini [feature] values), the AIAEC models at their shipped grid with delay depth 8. */
static void dfn3_target_grid(DFN3Grid *grid) {
    const DFN2Grid *g = dfn_test_grid();
    const DFN3GridSpec spec = {
        .sample_rate          = g->sample_rate,
        .n_fft                = g->n_fft,
        .n_erb                = g->n_erb,
        .df_bins              = g->df_bins,
        .df_order             = g->df_order,
        .mask_lookahead       = g->mask_lookahead,
        .df_lookahead         = g->df_lookahead,
        /* DFN3's own grid takes these explicitly; DFN2's are fixed constants in dfn2_grid.h,
         * the same config.ini [feature] numbers. */
        .erb_norm_tau_sec     = DFN2_ERB_NORM_TAU_SEC,
        .erb_norm_init_lo_db  = DFN2_ERB_NORM_INIT_LO_DB,
        .erb_norm_init_hi_db  = DFN2_ERB_NORM_INIT_HI_DB,
        .erb_norm_scale_db    = DFN2_ERB_NORM_SCALE_DB,
        .spec_norm_tau_sec    = DFN2_SPEC_NORM_TAU_SEC,
        .spec_norm_init_lo    = DFN2_SPEC_NORM_INIT_LO,
        .spec_norm_init_hi    = DFN2_SPEC_NORM_INIT_HI,
    };
    assert(dfn3_grid_make(grid, &spec) == DFN3_GRID_OK);
}

static void ulcnet_target_grid(UlcnetGrid *grid) {
    const UlcnetGridSpec spec = { .sample_rate = 16000, .n_fft = 512, .delay_depth = 8 };
    assert(ulcnet_grid_make(grid, &spec) == ULCNET_GRID_OK);
}

static void deepvqe_target_grid(DeepVqeGrid *grid) {
    const DeepVqeGridSpec spec = { .sample_rate = 16000, .n_fft = 512, .delay_depth = 8 };
    assert(deepvqe_grid_make(grid, &spec) == DEEPVQE_GRID_OK);
}

/* The pre/post APIs call their descriptor query get_mem_size. */
#define dfn2_prepost_get_mem_requirements dfn2_prepost_get_mem_size
#define dfn3_prepost_get_mem_requirements dfn3_prepost_get_mem_size
#define ulcnet_prepost_get_mem_requirements ulcnet_prepost_get_mem_size
#define deepvqe_prepost_get_mem_requirements deepvqe_prepost_get_mem_size

int main(int argc, char **argv) {
    uint32_t pointer_bytes = (uint32_t)sizeof(void *), foreign_pointer_bytes;
    McabConfig flow = mcab_default_config(16000);
    McabConfig front_only = flow;
    DFN2PrepostConfig dfn2;
    DFN3PrepostConfig dfn3;
    UlcnetPrepostConfig ulc;
    DeepVqePrepostConfig deep;
    DFN3Grid dfn3_grid;
    UlcnetGrid ulc_grid;
    DeepVqeGrid deep_grid;
    DfnResStageConfig stage = dfn_res_stage_default_config(dfn_test_grid());
    DfnRateBridgeConfig bridge = dfn_rate_bridge_default_config(16000, dfn_test_grid());

    if (argc == 3 && strcmp(argv[1], "--write") == 0) {
        write_file = fopen(argv[2], "wb");
        assert(write_file);
        assert(fwrite(&pointer_bytes, sizeof(pointer_bytes), 1, write_file) == 1);
    } else if (argc == 3 && strcmp(argv[1], "--reject") == 0) {
        foreign_file = fopen(argv[2], "rb");
        assert(foreign_file);
        assert(fread(&foreign_pointer_bytes, sizeof(foreign_pointer_bytes),
                     1, foreign_file) == 1);
        assert(foreign_pointer_bytes != pointer_bytes);
    } else {
        assert(argc == 1);
    }

    dfn3_target_grid(&dfn3_grid);
    ulcnet_target_grid(&ulc_grid);
    deepvqe_target_grid(&deep_grid);
    dfn_unit_partition(erb_fwd, erb_inv, dfn_test_grid());
    assert(dfn2_prepost_config_defaults(&dfn2, dfn_test_grid()) == 0);
    assert(dfn3_prepost_config_defaults(&dfn3, &dfn3_grid) == 0);
    assert(ulcnet_prepost_config_defaults(&ulc, &ulc_grid) == 0);
    assert(deepvqe_prepost_config_defaults(&deep, &deep_grid) == 0);
    dfn2.erb_fwd = dfn3.erb_fwd = erb_fwd;
    dfn2.erb_inv = dfn3.erb_inv = erb_inv;
    stage.erb_fwd = bridge.stage.erb_fwd = erb_fwd;
    stage.erb_inv = bridge.stage.erb_inv = erb_inv;

    EXERCISE(dfn2_prepost, DFN2PrepostMemReq, dfn2);
    EXERCISE(dfn3_prepost, DFN3PrepostMemReq, dfn3);
    EXERCISE(ulcnet_prepost, UlcnetPrepostMemReq, ulc);
    EXERCISE(deepvqe_prepost, DeepVqePrepostMemReq, deep);
    /* Two carves of the one flow: with the fuse buffers and front only. */
    front_only.enable_post = 0;
    build_mcab_libs(&flow);
    EXERCISE(mcab_case, McabMemReq, flow);
    build_mcab_libs(&front_only);   /* the front-only flow needs front-only lanes */
    EXERCISE(mcab_case, McabMemReq, front_only);
    EXERCISE(dfn_res_stage, DfnResStageMemReq, stage);
    EXERCISE(dfn_rate_bridge, DfnRateBridgeMemReq, bridge);
    assert(cases == 8u);
    if (write_file) assert(fclose(write_file) == 0);
    if (foreign_file) {
        assert(fgetc(foreign_file) == EOF);
        assert(fclose(foreign_file) == 0);
    }
    puts("PASS target ABI descriptors");
    return 0;
}
