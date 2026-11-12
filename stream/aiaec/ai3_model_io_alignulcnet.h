/* Align-ULCNet graph rows for the board samples, from AIAEC/Align_ULCNet/export_onnx.py: one generator per grid for
 * the one-frame graph (layout 18, N = 1) and the packet graph (layout 19, N > 1). D = ceil(0.5 s / hop) and
 * ta_bins = (bins + 9) / 10, as exported. */

#ifndef AI3_MODEL_IO_ALIGNULCNET_H
#define AI3_MODEL_IO_ALIGNULCNET_H

#include "ai3_model_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layout versions: ULCNET_MODEL_IO_LAYOUT_VERSION (N = 1) and ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION (N > 1). */
#define AI3_MODEL_IO_ALIGNULCNET_LAYOUT_ONE_FRAME 18
#define AI3_MODEL_IO_ALIGNULCNET_LAYOUT_PACKET    19

#define AI3_MODEL_IO_ALIGNULCNET_TA_CHANNELS   32
#define AI3_MODEL_IO_ALIGNULCNET_SCORE_WINDOW  5
#define AI3_MODEL_IO_ALIGNULCNET_GRU_ROWS      4
#define AI3_MODEL_IO_ALIGNULCNET_GRU_HIDDEN    128

/* One registered graph: the grid's bins, the K/V width ta_bins and the delay depth D are exporter values. */
typedef struct Ai3ModelIoAlignulcnetGrid {
    int sample_rate, fft, n, bins, ta_bins, depth;
} Ai3ModelIoAlignulcnetGrid;

static const Ai3ModelIoAlignulcnetGrid AI3_MODEL_IO_ALIGNULCNET_GRIDS[] = {
    { 16000, 256, 1, 129, 13, 63 },
    { 16000, 256, 8, 129, 13, 63 },
    { 16000, 512, 1, 257, 26, 32 },
};

#define AI3_MODEL_IO_ALIGNULCNET_ROW_COUNT \
    ((int)(sizeof(AI3_MODEL_IO_ALIGNULCNET_GRIDS) / sizeof(AI3_MODEL_IO_ALIGNULCNET_GRIDS[0])))

/* The row of one grid. Inputs: error_mag, far_mag, error_cos, error_sin [1, 1, N, bins], frame_reset [1, 1, 1, N] last
 * at N > 1, then key_history, value_history [1, 32, D, ta_bins], logit_history [1, 32, 5, D], h_gru [1, 4, 1, 128];
 * outputs: output [1, 2, N, bins], then the states as <name>_out. */
static inline void ai3_model_io_alignulcnet_describe(const Ai3ModelIoAlignulcnetGrid *g, Ai3ModelIoRow *row)
{
    unsigned int n = (unsigned int)g->n, bins = (unsigned int)g->bins, d = (unsigned int)g->depth;
    unsigned int ta = (unsigned int)g->ta_bins;
    int i;

    memset(row, 0, sizeof *row);
    row->model = AI3_MODEL_IO_ALIGN_ULCNET;
    row->sample_rate = g->sample_rate; row->fft = g->fft; row->n = g->n;
    row->layout_version = g->n > 1 ? AI3_MODEL_IO_ALIGNULCNET_LAYOUT_PACKET
                                   : AI3_MODEL_IO_ALIGNULCNET_LAYOUT_ONE_FRAME;
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
        row->state[i][0] = 1u; row->state[i][1] = AI3_MODEL_IO_ALIGNULCNET_TA_CHANNELS;
        row->state[i][2] = d; row->state[i][3] = ta;
    }
    row->state[2][0] = 1u; row->state[2][1] = AI3_MODEL_IO_ALIGNULCNET_TA_CHANNELS;
    row->state[2][2] = AI3_MODEL_IO_ALIGNULCNET_SCORE_WINDOW; row->state[2][3] = d;   /* logit_history */
    row->state[3][0] = 1u; row->state[3][1] = AI3_MODEL_IO_ALIGNULCNET_GRU_ROWS;
    row->state[3][2] = 1u; row->state[3][3] = AI3_MODEL_IO_ALIGNULCNET_GRU_HIDDEN;   /* h_gru */
    row->state_count = 4;
}

/* The config at (sample_rate, fft, n); a grid without an entry is refused, never derived. */
static inline int ai3_model_io_alignulcnet_find(int sample_rate, int fft, int n, Ai3ModelIoConfig *cfg)
{
    int i;
    Ai3ModelIoRow row;

    if (!cfg) return AI3_MODEL_IO_ERR_NULL;
    memset(cfg, 0, sizeof *cfg);
    for (i = 0; i < AI3_MODEL_IO_ALIGNULCNET_ROW_COUNT; ++i) {
        const Ai3ModelIoAlignulcnetGrid *g = &AI3_MODEL_IO_ALIGNULCNET_GRIDS[i];
        if (g->sample_rate == sample_rate && g->fft == fft && g->n == n) {
            ai3_model_io_alignulcnet_describe(g, &row);
            return ai3_model_io_expand(&row, cfg);
        }
    }
    return AI3_MODEL_IO_ERR_NO_ROW;
}

/* Row i in table order, expanded; ERR_INDEX outside [0, AI3_MODEL_IO_ALIGNULCNET_ROW_COUNT). */
static inline int ai3_model_io_alignulcnet_row(int i, Ai3ModelIoConfig *cfg)
{
    Ai3ModelIoRow row;

    if (!cfg) return AI3_MODEL_IO_ERR_NULL;
    if (i < 0 || i >= AI3_MODEL_IO_ALIGNULCNET_ROW_COUNT) {
        memset(cfg, 0, sizeof *cfg);
        return AI3_MODEL_IO_ERR_INDEX;
    }
    ai3_model_io_alignulcnet_describe(&AI3_MODEL_IO_ALIGNULCNET_GRIDS[i], &row);
    return ai3_model_io_expand(&row, cfg);
}

#ifdef __cplusplus
}
#endif

#endif /* AI3_MODEL_IO_ALIGNULCNET_H */
