/* DFN2 graph rows (full and lite) for the board samples, one per registered (grid, N, mask_lookahead), from
 * AINR/DeepFilterNet2/export_onnx.py. Inputs: erb, spec, h_gru [, df_convp_history]; outputs: erb_mask [, df_coefs,
 * df_alpha], then the states as <name>_out. Signal windows hold N + 2 frames, heads N. The shapes do not depend on the
 * lookahead pair, so the port check cannot tell the 48 kHz mask_lookahead 1 and 0 graphs apart: mask_lookahead is
 * part of the key, and each has its own file. */

#ifndef GRAPH_IO_DFN2_H
#define GRAPH_IO_DFN2_H

#include "graph_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The model of this build: lite under -DDFN2_MASK_ONLY (the library's own switch), full otherwise. */
#ifdef DFN2_MASK_ONLY
#define GRAPH_IO_DFN2_PRODUCT GRAPH_IO_DFN2_LITE
#else
#define GRAPH_IO_DFN2_PRODUCT GRAPH_IO_DFN2_FULL
#endif

/* Full only: output 2 is df_alpha. */
#define GRAPH_IO_DFN2_HEAD_DF_ALPHA 2

/* Layout versions: DFN2_MODEL_IO_LAYOUT_VERSION_FULL / _MASK_ONLY of dfn2_model_io.h. */
#define GRAPH_IO_DFN2_LAYOUT_FULL      8
#define GRAPH_IO_DFN2_LAYOUT_LITE      9

/* One DFN2 graph: the shared row, the mask_lookahead it was exported with (AinrInfo.mask_lookahead) and the ERB
 * matrix file of its grid, fb_<sample_rate>_<n_erb>.bin, which dfn2_erb_file.h loads. */
typedef struct GraphIoDfn2Row {
    GraphIoRow  row;
    int         mask_lookahead;
    const char *erb_file;
} GraphIoDfn2Row;

#define GRAPH_IO_DFN2_STATES_FULL { {1, 5, 1, 256}, {1, 64, 5, 96} }

/* State rows: h_gru [1, 5 (full) or 3 (lite), 1, 256]; df_convp_history [1, 64, 5, df_bins 96] (full only). The
 * 16 kHz graphs are mask_lookahead 0 in both full models; the dfn2_l0/ files are the mask_lookahead 0 model's own. */
static const GraphIoDfn2Row GRAPH_IO_DFN2_ROWS[] = {
    /* full, 16 kHz / fft 256 (n_erb 24), N = 1 */
    { { GRAPH_IO_DFN2_FULL, 16000, 256, 1, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 3, 24}, {1, 2, 3, 96} },
        { {1, 1, 1, 24}, {1, 1, 96, 10}, {1, 1, 1, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2/nvt_model_16k_n1.bin" }, 0, "fb_16000_24.bin" },
    /* full, 16 kHz / fft 256, N = 8 */
    { { GRAPH_IO_DFN2_FULL, 16000, 256, 8, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 10, 24}, {1, 2, 10, 96} },
        { {1, 1, 8, 24}, {1, 8, 96, 10}, {1, 1, 8, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2/nvt_model_16k_n8.bin" }, 0, "fb_16000_24.bin" },
    /* full, 48 kHz / fft 1024 (n_erb 32), N = 1, mask_lookahead 1 */
    { { GRAPH_IO_DFN2_FULL, 48000, 1024, 1, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 3, 32}, {1, 2, 3, 96} },
        { {1, 1, 1, 32}, {1, 1, 96, 10}, {1, 1, 1, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2/nvt_model_48k_n1.bin" }, 1, "fb_48000_32.bin" },
    /* full, 48 kHz / fft 1024, N = 2, mask_lookahead 1 */
    { { GRAPH_IO_DFN2_FULL, 48000, 1024, 2, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 4, 32}, {1, 2, 4, 96} },
        { {1, 1, 2, 32}, {1, 2, 96, 10}, {1, 1, 2, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2/nvt_model_48k_n2.bin" }, 1, "fb_48000_32.bin" },
    /* full, 48 kHz / fft 1024, N = 1, mask_lookahead 0 */
    { { GRAPH_IO_DFN2_FULL, 48000, 1024, 1, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 3, 32}, {1, 2, 3, 96} },
        { {1, 1, 1, 32}, {1, 1, 96, 10}, {1, 1, 1, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2_l0/nvt_model_48k_n1.bin" }, 0, "fb_48000_32.bin" },
    /* full, 48 kHz / fft 1024, N = 2, mask_lookahead 0 */
    { { GRAPH_IO_DFN2_FULL, 48000, 1024, 2, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 4, 32}, {1, 2, 4, 96} },
        { {1, 1, 2, 32}, {1, 2, 96, 10}, {1, 1, 2, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2_l0/nvt_model_48k_n2.bin" }, 0, "fb_48000_32.bin" },
    /* full, 32 kHz / fft 512 (n_erb 28), N = 1, mask_lookahead 0 */
    { { GRAPH_IO_DFN2_FULL, 32000, 512, 1, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 3, 28}, {1, 2, 3, 96} },
        { {1, 1, 1, 28}, {1, 1, 96, 10}, {1, 1, 1, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2_l0/nvt_model_32k_n1.bin" }, 0, "fb_32000_28.bin" },
    /* full, 32 kHz / fft 512, N = 4, mask_lookahead 0 */
    { { GRAPH_IO_DFN2_FULL, 32000, 512, 4, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
        { {1, 1, 6, 28}, {1, 2, 6, 96} },
        { {1, 1, 4, 28}, {1, 4, 96, 10}, {1, 1, 4, 1} },
        GRAPH_IO_DFN2_STATES_FULL, "dfn2_l0/nvt_model_32k_n4.bin" }, 0, "fb_32000_28.bin" },
    /* lite (--mask-only), 48 kHz / fft 1024, N = 1, mask_lookahead 1 */
    { { GRAPH_IO_DFN2_LITE, 48000, 1024, 1, GRAPH_IO_DFN2_LAYOUT_LITE, 2, 1, 1,
        { {1, 1, 3, 32}, {1, 2, 3, 96} },
        { {1, 1, 1, 32} },
        { {1, 3, 1, 256} }, "dfn2_lite/nvt_model_48k_n1.bin" }, 1, "fb_48000_32.bin" },
};

#undef GRAPH_IO_DFN2_STATES_FULL

#define GRAPH_IO_DFN2_ROW_COUNT ((int)(sizeof(GRAPH_IO_DFN2_ROWS) / sizeof(GRAPH_IO_DFN2_ROWS[0])))

/* The row of model (GRAPH_IO_DFN2_FULL / _LITE) at (sample_rate, fft, n, mask_lookahead), or NULL: a configuration
 * without a row is refused, never derived. */
static inline const GraphIoDfn2Row *graph_io_dfn2_lookup(int model, int sample_rate, int fft, int n,
                                                         int mask_lookahead)
{
    int i;
    for (i = 0; i < GRAPH_IO_DFN2_ROW_COUNT; ++i) {
        const GraphIoDfn2Row *r = &GRAPH_IO_DFN2_ROWS[i];
        if (r->row.model == model && r->row.sample_rate == sample_rate && r->row.fft == fft && r->row.n == n &&
            r->mask_lookahead == mask_lookahead)
            return r;
    }
    return (const GraphIoDfn2Row *)0;
}

/* The expanded config of that row: OK, or ERR_NO_ROW / ERR_NULL / ERR_ROW; cfg is zeroed on failure. */
static inline int graph_io_dfn2_find(int model, int sample_rate, int fft, int n, int mask_lookahead,
                                     GraphIoConfig *cfg)
{
    const GraphIoDfn2Row *r;

    if (!cfg) return GRAPH_IO_ERR_NULL;
    memset(cfg, 0, sizeof *cfg);
    r = graph_io_dfn2_lookup(model, sample_rate, fft, n, mask_lookahead);
    return r ? graph_io_expand(&r->row, cfg) : GRAPH_IO_ERR_NO_ROW;
}

/* The ERB matrix file of model's grid (sample_rate, fft) and its n_erb (may be NULL), shared by every row of the grid;
 * needed before ainr_init, so it takes no N or mask_lookahead. NULL without a row. */
static inline const char *graph_io_dfn2_erb_file(int model, int sample_rate, int fft, int *n_erb)
{
    int i;
    for (i = 0; i < GRAPH_IO_DFN2_ROW_COUNT; ++i) {
        const GraphIoDfn2Row *r = &GRAPH_IO_DFN2_ROWS[i];
        if (r->row.model == model && r->row.sample_rate == sample_rate && r->row.fft == fft) {
            if (n_erb) *n_erb = (int)r->row.signal[0][3];   /* signal 0 is the erb window [1, 1, N + 2, n_erb] */
            return r->erb_file;
        }
    }
    return (const char *)0;
}

/* Row i in table order, expanded; ERR_INDEX outside [0, GRAPH_IO_DFN2_ROW_COUNT). */
static inline int graph_io_dfn2_row(int i, GraphIoConfig *cfg)
{
    if (!cfg) return GRAPH_IO_ERR_NULL;
    if (i < 0 || i >= GRAPH_IO_DFN2_ROW_COUNT) {
        memset(cfg, 0, sizeof *cfg);
        return GRAPH_IO_ERR_INDEX;
    }
    return graph_io_expand(&GRAPH_IO_DFN2_ROWS[i].row, cfg);
}

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_IO_DFN2_H */
