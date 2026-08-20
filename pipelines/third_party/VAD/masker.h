#ifndef MASKER_H
#define MASKER_H
#include <stddef.h>
#include "fft_wrapper.h"

typedef struct {

    /* signal format */
    int NFFT;
    int sr;

    /* energy mask */
    float E_alpha_up;
    float E_alpha_down;
    float margin_dB;

    /* band mask */
    float low_freq;
    float high_freq;

    /* spp mask */
    float M_alpha;
    float spp_thr;
    float spp_upd_thr;

    /* smoothing */
    int   enable_freq_smooth;
    int   smooth_size;
    int   enable_time_smooth;
    float T_alpha;

    /* enable flags */
    int enable_energy;
    int enable_spp;
    int enable_band;

} MaskerConfig;



typedef struct {
    /* system */
    int NFFT;
    int F;
    int sr;

    /* config (pointer) */
    MaskerConfig cfg;

    float* power_frame;
    /* 10*log10 of power_frame, materialised so the dB conversion and the
     * asymmetric noise-floor EMA can each run as one array pass. */
    float* energy_frame;

    /* state */
    float* noise_floor;
    float* noise_psd;
    float* spp_time;

    int* band_mask;
    int* energy_mask;
    int* spp_mask;
    int* spp_mask_bin;
    int* spp_mask_f;
    int* mask;

    int initialized;

    /* Non-NULL only on the masker_create() heap path: the single block
     * backing this struct and every array above, freed by masker_destroy().
     * NULL on the masker_init() caller-pool path. */
    void* owned_heap;
} MaskEstimator;



/* Heap constructor: one block sized by masker_get_mem_size(), carved by the
 * same code path as masker_init(). */
MaskEstimator* masker_create(const MaskerConfig* cfg);

/* Caller-pool constructor. `mem` must be 16-byte aligned and `mem_size` at
 * least masker_get_mem_size(cfg), else NULL. The caller owns the memory and
 * masker_destroy() will not free it. */
size_t         masker_get_mem_size(const MaskerConfig* cfg);
MaskEstimator* masker_init(void* mem, size_t mem_size, const MaskerConfig* cfg);

void masker_destroy(MaskEstimator* m);

void masker_step(MaskEstimator* m,
                 const Complex* X_ref);

const int* masker_get_mask(const MaskEstimator* m);

#endif /* MASKER_H */
