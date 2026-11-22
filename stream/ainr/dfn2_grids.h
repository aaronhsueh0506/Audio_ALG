/* The DFN2 checkpoints' grids: which (sample_rate, n_fft) maps to which n_erb, df_bins, df_order and lookahead pair
 * (AINR/DeepFilterNet2/config.ini [signal], [model]) for each bound model of ainr_product.h. The one table of ainr.c
 * and the pipeline example mains; header-only. frames_per_call is the caller's (1 here). */

#ifndef DFN2_GRIDS_H
#define DFN2_GRIDS_H

#include "ainr_product.h"
#include "dfn2_grid.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dfn2GridEntry {
    int          model_id;   /* AINR_MODEL_ID_* */
    DFN2GridSpec spec;
} Dfn2GridEntry;

/* sample_rate, n_fft, n_erb, df_bins, df_order, mask_lookahead, df_lookahead, frames_per_call. The lite model has no deep
 * filter (df_order 0, df_lookahead 0); the mask_lookahead 0 model's 16 kHz grid is the full model's. */
static const Dfn2GridEntry DFN2_GRIDS[] = {
    { AINR_MODEL_ID_DFN2_FULL,    { 16000, 256,  24, 96, 5, 0, 0, 1 } },
    { AINR_MODEL_ID_DFN2_FULL,    { 48000, 1024, 32, 96, 5, 1, 1, 1 } },
    { AINR_MODEL_ID_DFN2_FULL_L0, { 16000, 256,  24, 96, 5, 0, 0, 1 } },
    { AINR_MODEL_ID_DFN2_FULL_L0, { 32000, 512,  28, 96, 5, 0, 0, 1 } },
    { AINR_MODEL_ID_DFN2_FULL_L0, { 48000, 1024, 32, 96, 5, 0, 0, 1 } },
    { AINR_MODEL_ID_DFN2_LITE,    { 48000, 1024, 32, 96, 0, 1, 0, 1 } },
};

#define DFN2_GRIDS_ROWS ((int)(sizeof(DFN2_GRIDS) / sizeof(DFN2_GRIDS[0])))

/* The grid of model_id at (sample_rate, n_fft), or NULL. */
static inline const DFN2GridSpec *dfn2_grids_find(int model_id, int sample_rate, int n_fft)
{
    int i;
    for (i = 0; i < DFN2_GRIDS_ROWS; ++i)
        if (DFN2_GRIDS[i].model_id == model_id && DFN2_GRIDS[i].spec.sample_rate == sample_rate &&
            DFN2_GRIDS[i].spec.n_fft == n_fft)
            return &DFN2_GRIDS[i].spec;
    return (const DFN2GridSpec *)0;
}

#ifdef __cplusplus
}
#endif

#endif /* DFN2_GRIDS_H */
