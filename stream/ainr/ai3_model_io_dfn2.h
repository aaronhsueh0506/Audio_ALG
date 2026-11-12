/* DFN2 graph rows (full and lite) for the board samples, one per registered (grid, N), from AINR/DeepFilterNet2/
 * export_onnx.py. Inputs: erb, spec, h_gru [, df_convp_history]; outputs: erb_mask [, df_coefs, df_alpha], then the
 * states as <name>_out. Signal windows hold N + 2 frames, heads N. */

#ifndef AI3_MODEL_IO_DFN2_H
#define AI3_MODEL_IO_DFN2_H

#include "ai3_model_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The model of this build: lite under -DDFN2_MASK_ONLY (the library's own switch), full otherwise. */
#ifdef DFN2_MASK_ONLY
#define AI3_MODEL_IO_DFN2_PRODUCT AI3_MODEL_IO_DFN2_LITE
#else
#define AI3_MODEL_IO_DFN2_PRODUCT AI3_MODEL_IO_DFN2_FULL
#endif

/* Full only: output 2 is df_alpha. */
#define AI3_MODEL_IO_DFN2_HEAD_DF_ALPHA 2

/* Layout versions: DFN2_MODEL_IO_LAYOUT_VERSION_FULL / _MASK_ONLY of dfn2_model_io.h. */
#define AI3_MODEL_IO_DFN2_LAYOUT_FULL      8
#define AI3_MODEL_IO_DFN2_LAYOUT_LITE      9

/* State rows: h_gru [1, 5 (full) or 3 (lite), 1, 256]; df_convp_history [1, 64, 5, df_bins 96] (full only). */
static const Ai3ModelIoRow AI3_MODEL_IO_DFN2_ROWS[] = {
    /* full, 16 kHz / fft 256 (n_erb 24), N = 1 */
    { AI3_MODEL_IO_DFN2_FULL, 16000, 256, 1, AI3_MODEL_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 3, 24}, {1, 2, 3, 96} },
      { {1, 1, 1, 24}, {1, 1, 96, 10}, {1, 1, 1, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* full, 16 kHz / fft 256, N = 8 */
    { AI3_MODEL_IO_DFN2_FULL, 16000, 256, 8, AI3_MODEL_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 10, 24}, {1, 2, 10, 96} },
      { {1, 1, 8, 24}, {1, 8, 96, 10}, {1, 1, 8, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* full, 48 kHz / fft 1024 (n_erb 32), N = 1 */
    { AI3_MODEL_IO_DFN2_FULL, 48000, 1024, 1, AI3_MODEL_IO_DFN2_LAYOUT_FULL, 2, 3, 2,
      { {1, 1, 3, 32}, {1, 2, 3, 96} },
      { {1, 1, 1, 32}, {1, 1, 96, 10}, {1, 1, 1, 1} },
      { {1, 5, 1, 256}, {1, 64, 5, 96} } },
    /* lite (--mask-only), 48 kHz / fft 1024, N = 1 */
    { AI3_MODEL_IO_DFN2_LITE, 48000, 1024, 1, AI3_MODEL_IO_DFN2_LAYOUT_LITE, 2, 1, 1,
      { {1, 1, 3, 32}, {1, 2, 3, 96} },
      { {1, 1, 1, 32} },
      { {1, 3, 1, 256} } },
};

#define AI3_MODEL_IO_DFN2_ROW_COUNT ((int)(sizeof(AI3_MODEL_IO_DFN2_ROWS) / sizeof(AI3_MODEL_IO_DFN2_ROWS[0])))

/* The config of model (AI3_MODEL_IO_DFN2_FULL / _LITE) at (sample_rate, fft, n); see ai3_model_io_find_row(). */
static inline int ai3_model_io_dfn2_find(int model, int sample_rate, int fft, int n, Ai3ModelIoConfig *cfg)
{
    if (model != AI3_MODEL_IO_DFN2_FULL && model != AI3_MODEL_IO_DFN2_LITE) {
        if (cfg) memset(cfg, 0, sizeof *cfg);
        return cfg ? AI3_MODEL_IO_ERR_NO_ROW : AI3_MODEL_IO_ERR_NULL;
    }
    return ai3_model_io_find_row(AI3_MODEL_IO_DFN2_ROWS, AI3_MODEL_IO_DFN2_ROW_COUNT, model, sample_rate, fft, n,
                                 cfg);
}

/* Row i in table order, expanded; ERR_INDEX outside [0, AI3_MODEL_IO_DFN2_ROW_COUNT). */
static inline int ai3_model_io_dfn2_row(int i, Ai3ModelIoConfig *cfg)
{
    if (i < 0 || i >= AI3_MODEL_IO_DFN2_ROW_COUNT) {
        if (cfg) memset(cfg, 0, sizeof *cfg);
        return cfg ? AI3_MODEL_IO_ERR_INDEX : AI3_MODEL_IO_ERR_NULL;
    }
    return ai3_model_io_expand(&AI3_MODEL_IO_DFN2_ROWS[i], cfg);
}

#ifdef __cplusplus
}
#endif

#endif /* AI3_MODEL_IO_DFN2_H */
