/* DFN2 graph rows (full and lite) for the board samples, one per registered (grid, N), from AINR/DeepFilterNet2/
 * export_onnx.py. Inputs: erb, spec, h_gru [, df_convp_history]; outputs: erb_mask [, df_coefs, df_alpha], then the
 * states as <name>_out. Signal windows hold N + 2 frames, heads N. The shapes do not depend on the lookahead pair, so a
 * row serves every graph of its (grid, N): the 48 kHz / 1024 N = 1 row is the mask_lookahead 1 graph's and the
 * mask_lookahead 0 graph's alike, and the port check cannot tell the two apart. */

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

/* State rows: h_gru [1, 5 (full) or 3 (lite), 1, 256]; df_convp_history [1, 64, 5, df_bins 96] (full only). */
static const GraphIoRow GRAPH_IO_DFN2_ROWS[] = {
    /* full, 16 kHz / fft 256 (n_erb 24), N = 1 */
    { GRAPH_IO_DFN2_FULL, 16000, 256, 1, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 3, 24}, {1, 2, 3, 96} },
      { {1, 1, 1, 24}, {1, 1, 96, 10}, {1, 1, 1, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* full, 16 kHz / fft 256, N = 8 */
    { GRAPH_IO_DFN2_FULL, 16000, 256, 8, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 10, 24}, {1, 2, 10, 96} },
      { {1, 1, 8, 24}, {1, 8, 96, 10}, {1, 1, 8, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* full, 48 kHz / fft 1024 (n_erb 32), N = 1 */
    { GRAPH_IO_DFN2_FULL, 48000, 1024, 1, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 3, 32}, {1, 2, 3, 96} },
      { {1, 1, 1, 32}, {1, 1, 96, 10}, {1, 1, 1, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* full, 48 kHz / fft 1024, N = 2 (mask_lookahead 0) */
    { GRAPH_IO_DFN2_FULL, 48000, 1024, 2, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 4, 32}, {1, 2, 4, 96} },
      { {1, 1, 2, 32}, {1, 2, 96, 10}, {1, 1, 2, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* full, 32 kHz / fft 512 (n_erb 28), N = 1 (mask_lookahead 0) */
    { GRAPH_IO_DFN2_FULL, 32000, 512, 1, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 3, 28}, {1, 2, 3, 96} },
      { {1, 1, 1, 28}, {1, 1, 96, 10}, {1, 1, 1, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* full, 32 kHz / fft 512, N = 4 (mask_lookahead 0) */
    { GRAPH_IO_DFN2_FULL, 32000, 512, 4, GRAPH_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 6, 28}, {1, 2, 6, 96} },
      { {1, 1, 4, 28}, {1, 4, 96, 10}, {1, 1, 4, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* lite (--mask-only), 48 kHz / fft 1024, N = 1 */
    { GRAPH_IO_DFN2_LITE, 48000, 1024, 1, GRAPH_IO_DFN2_LAYOUT_LITE, 2, 1, 1,
      { {1, 1, 3, 32}, {1, 2, 3, 96} },
      { {1, 1, 1, 32} },
      { {1, 3, 1, 256} } },
};

#define GRAPH_IO_DFN2_ROW_COUNT ((int)(sizeof(GRAPH_IO_DFN2_ROWS) / sizeof(GRAPH_IO_DFN2_ROWS[0])))

/* The config of model (GRAPH_IO_DFN2_FULL / _LITE) at (sample_rate, fft, n); see graph_io_find_row(). */
static inline int graph_io_dfn2_find(int model, int sample_rate, int fft, int n, GraphIoConfig *cfg)
{
    if (model != GRAPH_IO_DFN2_FULL && model != GRAPH_IO_DFN2_LITE) {
        if (cfg) memset(cfg, 0, sizeof *cfg);
        return cfg ? GRAPH_IO_ERR_NO_ROW : GRAPH_IO_ERR_NULL;
    }
    return graph_io_find_row(GRAPH_IO_DFN2_ROWS, GRAPH_IO_DFN2_ROW_COUNT, model, sample_rate, fft, n,
                                 cfg);
}

/* Row i in table order, expanded; ERR_INDEX outside [0, GRAPH_IO_DFN2_ROW_COUNT). */
static inline int graph_io_dfn2_row(int i, GraphIoConfig *cfg)
{
    if (i < 0 || i >= GRAPH_IO_DFN2_ROW_COUNT) {
        if (cfg) memset(cfg, 0, sizeof *cfg);
        return cfg ? GRAPH_IO_ERR_INDEX : GRAPH_IO_ERR_NULL;
    }
    return graph_io_expand(&GRAPH_IO_DFN2_ROWS[i], cfg);
}

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_IO_DFN2_H */
