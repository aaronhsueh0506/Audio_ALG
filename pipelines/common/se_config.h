/* se_config.h -- configuration shared by the AEC pipeline applications.
 *
 * Derives the signal grid and the linear-stage AecConfig from the fields every
 * application carries; each application states its own rules as an SeRules value.
 * Pure functions: no allocation, no I/O, no state. */
#ifndef PIPELINES_SE_CONFIG_H
#define PIPELINES_SE_CONFIG_H

#include "aec.h"
#include "anr_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* PBFDKF taps accepted from a caller; 0 selects the preset's own length. */
#define SE_FILTER_LENGTH_MAX 4096

/* SeRules.spatial_context: keep the context level the preset carries. */
#define SE_CONTEXT_PRESET (-1)

/* No-padding, 50 % overlap grid. frame size == fft_size. */
typedef struct SeGrid {
    int fft_size;
    int hop_size;     /* fft_size / 2     */
    int n_freqs;      /* fft_size / 2 + 1 */
} SeGrid;

/* What a caller asks for. */
typedef struct SeRequest {
    int          sample_rate;
    int          fft_size;             /* 0 = the rate's default grid          */
    int          filter_length;        /* 0 = preset, else <= SE_FILTER_LENGTH_MAX */
    AecDelayMode delay_mode;
    int          delay_num_filters;
    int          fixed_delay_samples;
    AecPreset    aec_preset;
    AnrMode      nr_mode;              /* checked only when SeRules.with_nr    */
} SeRequest;

/* What an application accepts and how its AEC stage is wired. */
typedef struct SeRules {
    int allow_8k;          /* admit the 8 kHz grid (fft 256)                          */
    int fixed_rate;        /* > 0: the only rate; then fft must be 0 or fixed_fft, allow_8k is ignored */
    int fixed_fft;         /* compiled-grid fft size */
    int check_delay;       /* reject delay settings that contradict delay_mode          */
    int lane_aligned;      /* the AEC consumes an already aligned reference (EXTERNAL_ALIGNED) */
    int spatial_context;   /* AecConfig.spatial_linear_context, or SE_CONTEXT_PRESET    */
    int with_nr;           /* validate nr_mode                                          */
} SeRules;

static inline int se_flag_valid(int v) { return v == 0 || v == 1; }

/* Resolves (sample_rate, requested fft) to a grid: 16 kHz 256|512, 48 kHz 1024,
 * and 8 kHz 256 when allow_8k. Returns 0, or -1 for any other pair. */
int se_grid_resolve(int sample_rate, int requested_fft, int allow_8k, SeGrid* out);

/* The compiled-grid variant: only (rate, fft) with fft requested as 0 or fft. */
int se_grid_fixed(int sample_rate, int requested_fft, int rate, int fft, SeGrid* out);

/* Validates `req` against `rules`, then fills the grid and the linear-stage
 * AecConfig (enable_res 0, return_res_context 1: the caller applies the RES and NR
 * gains). Returns 0, or -1 for the first violated rule (outputs unspecified). */
int se_derive(const SeRequest* req, const SeRules* rules,
              SeGrid* grid, AecConfig* aec);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SE_CONFIG_H */
