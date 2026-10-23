#ifndef NR_GAIN_H
#define NR_GAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Post-NR gain compensation: one smoothed broadcast gain per hop, chosen by
 * whether a DOA estimate exists. The config is taken per call; `nr_enable` is live
 * state, so refresh a cached NrGainConfig every hop. */
typedef struct {
    int enable;                 /* master switch; 0 pins the gain at 1.0 */
    int only_when_nr_enable;    /* skip compensation while NR itself is off */
    int nr_enable;              /* the caller's current NR state */

    float target_gain;          /* applied when a DOA estimate exists */
    float noise_gain;           /* applied when doa_gain is NaN */

    float min_gain;
    float max_gain;

    float attack_alpha;         /* EMA coefficient while the gain rises */
    float release_alpha;        /* ... and while it falls */
} NrGainConfig;

typedef struct NrGain NrGain;

/* Heap constructor: one block sized by nr_gain_get_mem_size(). */
NrGain* nr_gain_create(void);

/* Caller-pool constructor: `mem` 16-byte aligned, `mem_size` >=
 * nr_gain_get_mem_size() (config independent), else NULL. The caller owns the
 * memory; nr_gain_destroy() does not free it. */
size_t  nr_gain_get_mem_size(void);
NrGain* nr_gain_init(void* mem, size_t mem_size);

/* Advances the smoothed gain one hop and scales `x` in place by it; returns the
 * applied gain. `x` NULL or `n` <= 0 advances the smoother only. */
float   nr_gain_process(NrGain* ng,
                        const NrGainConfig* cfg,
                        float* x,
                        int n,
                        float doa_gain);
float   nr_gain_get_gain(const NrGain* ng);
void    nr_gain_destroy(NrGain* ng);

#ifdef __cplusplus
}
#endif

#endif /* NR_GAIN_H */
