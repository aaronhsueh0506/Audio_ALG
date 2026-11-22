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
 * up3_history. Each row follows the grid's AIAEC/DeepVQE_S/config_<grid>.ini. */
static const GraphIoRow GRAPH_IO_DEEPVQES_ROWS[] = {
    /* 16 kHz / fft 256 (129 bins, D 63), N = 1 */
    { GRAPH_IO_DEEPVQES, 16000, 256, 1, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 1, 129}, {1, 2, 1, 129} },
      { {1, 1, 129, 18} },
      { {1, 8, 63, 33}, {1, 24, 63, 33}, {1, 8, 9, 63}, {1, 32, 4, 65}, {1, 2, 4, 129}, {1, 8, 4, 65},
        {1, 1, 1, 256}, {1, 2, 4, 129}, {1, 16, 4, 65}, {1, 64, 4, 33}, {1, 56, 4, 17}, {1, 32, 4, 33},
        {1, 40, 4, 17}, {1, 32, 4, 33}, {1, 40, 4, 17}, {1, 24, 4, 9} },
      "deepvqes_16k_fft256_n1.bin" },
    /* 16 kHz / fft 512 (257 bins, D 32), N = 1 */
    { GRAPH_IO_DEEPVQES, 16000, 512, 1, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 1, 257}, {1, 2, 1, 257} },
      { {1, 1, 257, 18} },
      { {1, 4, 32, 65}, {1, 24, 32, 65}, {1, 4, 5, 32}, {1, 32, 4, 129}, {1, 2, 4, 257}, {1, 8, 4, 129},
        {1, 1, 1, 192}, {1, 2, 4, 257}, {1, 16, 4, 129}, {1, 64, 4, 65}, {1, 56, 4, 33}, {1, 32, 4, 65},
        {1, 40, 4, 33}, {1, 32, 4, 65}, {1, 40, 4, 33}, {1, 24, 4, 17} },
      "deepvqes_16k_fft512_n1.bin" },
    /* 16 kHz / fft 512, N = 4 */
    { GRAPH_IO_DEEPVQES, 16000, 512, 4, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 4, 257}, {1, 2, 4, 257} },
      { {1, 4, 257, 18} },
      { {1, 4, 32, 65}, {1, 24, 32, 65}, {1, 4, 5, 32}, {1, 32, 4, 129}, {1, 2, 4, 257}, {1, 8, 4, 129},
        {1, 1, 1, 192}, {1, 2, 4, 257}, {1, 16, 4, 129}, {1, 64, 4, 65}, {1, 56, 4, 33}, {1, 32, 4, 65},
        {1, 40, 4, 33}, {1, 32, 4, 65}, {1, 40, 4, 33}, {1, 24, 4, 17} },
      "deepvqes_16k_fft512_n4.bin" },
    /* 48 kHz / fft 1024 (513 bins, D 47), N = 1 */
    { GRAPH_IO_DEEPVQES, 48000, 1024, 1, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 1, 513}, {1, 2, 1, 513} },
      { {1, 1, 513, 18} },
      { {1, 4, 47, 129}, {1, 24, 47, 129}, {1, 4, 5, 47}, {1, 32, 4, 257}, {1, 2, 4, 513}, {1, 8, 4, 257},
        {1, 1, 1, 192}, {1, 2, 4, 513}, {1, 16, 4, 257}, {1, 64, 4, 129}, {1, 56, 4, 65}, {1, 32, 4, 129},
        {1, 40, 4, 65}, {1, 32, 4, 129}, {1, 40, 4, 65}, {1, 24, 4, 33} },
      "deepvqes_48k_fft1024_n1.bin" },
    /* 48 kHz / fft 1024, N = 2 */
    { GRAPH_IO_DEEPVQES, 48000, 1024, 2, GRAPH_IO_DEEPVQES_LAYOUT, 2, 1, 16,
      { {1, 2, 2, 513}, {1, 2, 2, 513} },
      { {1, 2, 513, 18} },
      { {1, 4, 47, 129}, {1, 24, 47, 129}, {1, 4, 5, 47}, {1, 32, 4, 257}, {1, 2, 4, 513}, {1, 8, 4, 257},
        {1, 1, 1, 192}, {1, 2, 4, 513}, {1, 16, 4, 257}, {1, 64, 4, 129}, {1, 56, 4, 65}, {1, 32, 4, 129},
        {1, 40, 4, 65}, {1, 32, 4, 129}, {1, 40, 4, 65}, {1, 24, 4, 33} },
      "deepvqes_48k_fft1024_n2.bin" },
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
