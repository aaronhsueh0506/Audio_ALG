/**
 * hpf.h - 2nd-order Butterworth IIR High-Pass Filter
 *
 * Removes DC offset, 50/60Hz hum, and low-frequency rumble.
 * 12 dB/octave rolloff. Direct Form II transposed.
 */

#ifndef HPF_H
#define HPF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Hpf Hpf;

/** Create high-pass filter (malloc version) */
Hpf* hpf_create(float cutoff_hz, int sample_rate);

/** Initialize HPF in pre-allocated memory (static version) */
Hpf* hpf_init(void* mem, size_t mem_size, float cutoff_hz, int sample_rate);

/** Get memory required for hpf_init() */
size_t hpf_get_mem_size(void);

/** Destroy HPF (no-op if created via hpf_init) */
void hpf_destroy(Hpf* hpf);

/**
 * Process samples in-place
 *
 * @param hpf Filter handle
 * @param data Sample buffer (modified in-place)
 * @param n Number of samples
 */
void hpf_process(Hpf* hpf, float* data, int n);

/**
 * Reset filter state
 */
void hpf_reset(Hpf* hpf);

#ifdef __cplusplus
}
#endif

#endif /* HPF_H */
