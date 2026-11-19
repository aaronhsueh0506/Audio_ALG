/* Align-ULCNet graph rows for the board samples, from AIAEC/Align_ULCNet/export_onnx.py: one generator per grid for
 * the one-frame graph (layout 18, N = 1) and the packet graph (layout 19, N > 1). bins = fft / 2 + 1, the K/V width
 * ta_bins = (bins + 9) / 10 and the delay depth D = ceil(0.5 s / hop) = ceil(sample_rate / fft), as exported. */

#ifndef GRAPH_IO_ALIGNULCNET_H
#define GRAPH_IO_ALIGNULCNET_H

#include "graph_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layout versions: ULCNET_MODEL_IO_LAYOUT_VERSION (N = 1) and ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION (N > 1). */
#define GRAPH_IO_ALIGNULCNET_LAYOUT_ONE_FRAME 18
#define GRAPH_IO_ALIGNULCNET_LAYOUT_PACKET    19

#define GRAPH_IO_ALIGNULCNET_TA_CHANNELS   32
#define GRAPH_IO_ALIGNULCNET_GRU_ROWS      4
#define GRAPH_IO_ALIGNULCNET_GRU_HIDDEN    128

/* One registered graph; score_window is the logit window W (SCORE_WINDOW_BY_GRID in export_onnx.py). */
typedef struct GraphIoAlignulcnetGrid {
    int sample_rate, fft, n, score_window;
} GraphIoAlignulcnetGrid;

/* Each row follows the grid's AIAEC/Align_ULCNet/config_<grid>.ini. */
static const GraphIoAlignulcnetGrid GRAPH_IO_ALIGNULCNET_GRIDS[] = {
    { 16000, 256, 1, 9 },
    { 16000, 512, 1, 5 },
    { 16000, 512, 4, 5 },
    { 48000, 1024, 1, 5 },
    { 48000, 1024, 2, 5 },
};

#define GRAPH_IO_ALIGNULCNET_ROW_COUNT \
    ((int)(sizeof(GRAPH_IO_ALIGNULCNET_GRIDS) / sizeof(GRAPH_IO_ALIGNULCNET_GRIDS[0])))

/* The row of one grid. Inputs: error_mag, far_mag, error_cos, error_sin [1, 1, N, bins], frame_reset [1, 1, 1, N] last
 * at N > 1, then key_history, value_history [1, 32, D, ta_bins], logit_history [1, 32, W, D], h_gru [1, 4, 1, 128];
 * outputs: output [1, 2, N, bins], then the states as <name>_out. */
static inline void graph_io_alignulcnet_describe(const GraphIoAlignulcnetGrid *g, GraphIoRow *row)
{
    unsigned int n = (unsigned int)g->n, bins = (unsigned int)g->fft / 2u + 1u, ta = (bins + 9u) / 10u;
    unsigned int d = ((unsigned int)g->sample_rate + (unsigned int)g->fft - 1u) / (unsigned int)g->fft;
    int i;

    memset(row, 0, sizeof *row);
    row->model = GRAPH_IO_ALIGNULCNET;
    row->sample_rate = g->sample_rate; row->fft = g->fft; row->n = g->n;
    row->layout_version = g->n > 1 ? GRAPH_IO_ALIGNULCNET_LAYOUT_PACKET
                                   : GRAPH_IO_ALIGNULCNET_LAYOUT_ONE_FRAME;
    for (i = 0; i < 4; ++i) {
        row->signal[i][0] = 1u; row->signal[i][1] = 1u; row->signal[i][2] = n; row->signal[i][3] = bins;
    }
    row->signal_count = 4;
    if (g->n > 1) {
        row->signal[4][0] = 1u; row->signal[4][1] = 1u; row->signal[4][2] = 1u; row->signal[4][3] = n;
        row->signal_count = 5;
    }
    row->head[0][0] = 1u; row->head[0][1] = 2u; row->head[0][2] = n; row->head[0][3] = bins;
    row->head_count = 1;
    for (i = 0; i < 2; ++i) {   /* key_history, value_history */
        row->state[i][0] = 1u; row->state[i][1] = GRAPH_IO_ALIGNULCNET_TA_CHANNELS;
        row->state[i][2] = d; row->state[i][3] = ta;
    }
    row->state[2][0] = 1u; row->state[2][1] = GRAPH_IO_ALIGNULCNET_TA_CHANNELS;
    row->state[2][2] = (unsigned int)g->score_window; row->state[2][3] = d;   /* logit_history */
    row->state[3][0] = 1u; row->state[3][1] = GRAPH_IO_ALIGNULCNET_GRU_ROWS;
    row->state[3][2] = 1u; row->state[3][3] = GRAPH_IO_ALIGNULCNET_GRU_HIDDEN;   /* h_gru */
    row->state_count = 4;
}

/* The config at (sample_rate, fft, n); a grid without an entry is refused, never derived. */
static inline int graph_io_alignulcnet_find(int sample_rate, int fft, int n, GraphIoConfig *cfg)
{
    int i;
    GraphIoRow row;

    if (!cfg) return GRAPH_IO_ERR_NULL;
    memset(cfg, 0, sizeof *cfg);
    for (i = 0; i < GRAPH_IO_ALIGNULCNET_ROW_COUNT; ++i) {
        const GraphIoAlignulcnetGrid *g = &GRAPH_IO_ALIGNULCNET_GRIDS[i];
        if (g->sample_rate == sample_rate && g->fft == fft && g->n == n) {
            graph_io_alignulcnet_describe(g, &row);
            return graph_io_expand(&row, cfg);
        }
    }
    return GRAPH_IO_ERR_NO_ROW;
}

/* Row i in table order, expanded; ERR_INDEX outside [0, GRAPH_IO_ALIGNULCNET_ROW_COUNT). */
static inline int graph_io_alignulcnet_row(int i, GraphIoConfig *cfg)
{
    GraphIoRow row;

    if (!cfg) return GRAPH_IO_ERR_NULL;
    if (i < 0 || i >= GRAPH_IO_ALIGNULCNET_ROW_COUNT) {
        memset(cfg, 0, sizeof *cfg);
        return GRAPH_IO_ERR_INDEX;
    }
    graph_io_alignulcnet_describe(&GRAPH_IO_ALIGNULCNET_GRIDS[i], &row);
    return graph_io_expand(&row, cfg);
}

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_IO_ALIGNULCNET_H */
