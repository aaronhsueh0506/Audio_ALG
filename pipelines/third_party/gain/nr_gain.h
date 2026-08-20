#ifndef NR_GAIN_H
#define NR_GAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Post-NR gain compensation: one smoothed broadcast gain per hop, chosen by
 * whether a DOA estimate exists.
 *
 * The tuning lives in a struct this module owns rather than in an
 * application-wide runtime-config header, so the archive builds standalone
 * exactly like FixGainConfig and PostGainConfig next to it. README.md carries
 * the field-by-field mapping for integrators migrating from the original.
 *
 * The config is taken per call, not at construction, so retuning between hops
 * needs no rebuild -- and `nr_enable` is live state rather than tuning, so a
 * caller caching an NrGainConfig must refresh it each hop or the compensation
 * keeps acting on a stale NR state. */
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

/* Caller-pool constructor. `mem` must be 16-byte aligned and `mem_size` at
 * least nr_gain_get_mem_size(), else NULL. The caller owns the memory and
 * nr_gain_destroy() will not free it. The size does not depend on any config
 * -- the instance is a single smoothed gain -- so the query takes no
 * argument, matching vad_get_mem_size()/hpf_get_mem_size(). */
size_t  nr_gain_get_mem_size(void);
NrGain* nr_gain_init(void* mem, size_t mem_size);

void    nr_gain_reset(NrGain* ng);

/* Advances the smoothed gain one hop and scales `x` in place by it. Returns
 * the gain that was applied. `x` may be NULL (or `n` <= 0) to advance the
 * smoother without touching a signal. */
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
