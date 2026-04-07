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

/**
 * Create high-pass filter
 *
 * @param cutoff_hz  Cutoff frequency in Hz (e.g. 80.0)
 * @param sample_rate Sample rate in Hz (e.g. 16000)
 * @return Filter handle, or NULL on error
 */
Hpf* hpf_create(float cutoff_hz, int sample_rate);

/**
 * Destroy high-pass filter
 */
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
