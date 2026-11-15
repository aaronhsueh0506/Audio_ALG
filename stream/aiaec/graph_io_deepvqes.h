/* DeepVQE-S graph rows for the board samples, one per registered (grid, N), from AIAEC/_streaming_export.py (delay
 * depth D = ceil(0.5 s / hop)). Inputs: mic, far, then the 16 states; outputs: output [1, N, bins, 18], then the
 * states as <name>_out. */

#ifndef GRAPH_IO_DEEPVQES_H
#define GRAPH_IO_DEEPVQES_H

#include "graph_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layout version: DEEPVQE_PREPOST_LAYOUT_VERSION of deepvqe_prepost.h. */
#define GRAPH_IO_DEEPVQES_LAYOUT 4

/* State order (named state_<name> except h_gru): align_key_ring, align_value_ring, align_score_history, ccm_up_history,
 * far1_history, far2_history, h_gru, mic1_history .. mic4_history, res2_history, res3_history, up1_history ..
 * up3_history. */
static const GraphIoRow GRAPH_IO_DEEPVQES_ROWS[] = {
    /* 16 kHz / fft 256 (129 bins, D 63), N = 1 */
    { GRAPH_IO_DEEPVQES, 16000, 256, 1, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 1, 129}, {1, 2, 1, 129} },
      { {1, 1, 129, 18} },
      { {1, 4, 63, 33}, {1, 24, 63, 33}, {1, 4, 5, 63}, {1, 32, 4, 65}, {1, 2, 4, 129}, {1, 8, 4, 65},
        {1, 1, 1, 192}, {1, 2, 4, 129}, {1, 16, 4, 65}, {1, 64, 4, 33}, {1, 56, 4, 17}, {1, 32, 4, 33},
        {1, 40, 4, 17}, {1, 32, 4, 33}, {1, 40, 4, 17}, {1, 24, 4, 9} } },
    /* 16 kHz / fft 256, N = 8 */
    { GRAPH_IO_DEEPVQES, 16000, 256, 8, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 8, 129}, {1, 2, 8, 129} },
      { {1, 8, 129, 18} },
      { {1, 4, 63, 33}, {1, 24, 63, 33}, {1, 4, 5, 63}, {1, 32, 4, 65}, {1, 2, 4, 129}, {1, 8, 4, 65},
        {1, 1, 1, 192}, {1, 2, 4, 129}, {1, 16, 4, 65}, {1, 64, 4, 33}, {1, 56, 4, 17}, {1, 32, 4, 33},
        {1, 40, 4, 17}, {1, 32, 4, 33}, {1, 40, 4, 17}, {1, 24, 4, 9} } },
    /* 16 kHz / fft 512 (257 bins, D 32), N = 1 */
    { GRAPH_IO_DEEPVQES, 16000, 512, 1, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 1, 257}, {1, 2, 1, 257} },
      { {1, 1, 257, 18} },
      { {1, 4, 32, 65}, {1, 24, 32, 65}, {1, 4, 5, 32}, {1, 32, 4, 129}, {1, 2, 4, 257}, {1, 8, 4, 129},
        {1, 1, 1, 192}, {1, 2, 4, 257}, {1, 16, 4, 129}, {1, 64, 4, 65}, {1, 56, 4, 33}, {1, 32, 4, 65},
        {1, 40, 4, 33}, {1, 32, 4, 65}, {1, 40, 4, 33}, {1, 24, 4, 17} } },
};

#define GRAPH_IO_DEEPVQES_ROW_COUNT \
    ((int)(sizeof(GRAPH_IO_DEEPVQES_ROWS) / sizeof(GRAPH_IO_DEEPVQES_ROWS[0])))

/* The config at (sample_rate, fft, n); see graph_io_find_row(). */
static inline int graph_io_deepvqes_find(int sample_rate, int fft, int n, GraphIoConfig *cfg)
{
    return graph_io_find_row(GRAPH_IO_DEEPVQES_ROWS, GRAPH_IO_DEEPVQES_ROW_COUNT,
                                 GRAPH_IO_DEEPVQES, sample_rate, fft, n, cfg);
}

/* Row i in table order, expanded; ERR_INDEX outside [0, GRAPH_IO_DEEPVQES_ROW_COUNT). */
static inline int graph_io_deepvqes_row(int i, GraphIoConfig *cfg)
{
    if (i < 0 || i >= GRAPH_IO_DEEPVQES_ROW_COUNT) {
        if (cfg) memset(cfg, 0, sizeof *cfg);
        return cfg ? GRAPH_IO_ERR_INDEX : GRAPH_IO_ERR_NULL;
    }
    return graph_io_expand(&GRAPH_IO_DEEPVQES_ROWS[i], cfg);
}

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_IO_DEEPVQES_H */
