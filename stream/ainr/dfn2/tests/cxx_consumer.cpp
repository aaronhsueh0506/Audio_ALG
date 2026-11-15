// A C++ consumer of the DFN2 headers: every function it calls must keep C linkage (check-header-cxx reads its
// undefined symbols).
#include "dfn2_prepost.h"
#include "dfn2_stream.h"

int dfn2_cxx_consumer(const DFN2Grid *grid, DFN2State *st, void *block, size_t bytes, float *f, long long *idx) {
    int rc = 0;
    DFN2_REQUIRE_VARIANT_LINK();
    rc += (int)dfn2_state_mem_size(grid);
    rc += dfn2_state_init(st, block, bytes, grid);
    dfn2_state_reset(st);
    dfn2_set_erb_matrices(st, f, f);
    dfn2_compute_features(st, f, f, f, f);
    dfn2_apply_atten_lim(1, f, f, f, f, 0.0f);
    rc += dfn2_enhance_spectrum(st, f, f, f, f, 0.0f, f, f);
    rc += dfn2_enhance_spectrum_stream(st, f, f, 1, f, f, 0.0f, 0.0f, f, f, idx);
    rc += dfn2_grid_validate(grid) + (int)dfn2_model_io_mem_size(grid);
    dfn2_prepost_reset(NULL);
    dfn2_stream_reset(NULL);
    return rc;
}
