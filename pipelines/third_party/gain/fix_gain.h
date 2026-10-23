#ifndef FIX_GAIN_H
#define FIX_GAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-channel input gain with an optional clip. For a single-gain mono path use
 * AudioPreGain (audio_pre_gain.h); this module adds the per-microphone table. */

typedef struct {
    int channels;
    int enable;

    /* Rejected as a whole config if any gain or clip_value is non-finite. A
     * global_gain of exactly 0 is treated as 1.0; use `enable` to bypass. */
    float global_gain;

    /* length = channels, NULL means 1.0 for all. Read only during construction (the
     * module keeps its own copy). */
    const float* channel_gain;

    int enable_clip;
    float clip_value;           /* usually 1.0 for float audio; <= 0 means 1.0 */
} FixGainConfig;

typedef struct FixGain FixGain;

/* Heap constructor: one block sized by fix_gain_get_mem_size(). */
FixGain* fix_gain_create(const FixGainConfig* cfg);

/* Caller-pool constructor: `mem` 16-byte aligned, `mem_size` >=
 * fix_gain_get_mem_size(cfg), else NULL. The caller owns the memory;
 * fix_gain_destroy() does not free it. Returns 0 for a config it will not
 * construct (channels <= 0, non-finite gain/clip), so it is also the config gate. */
size_t   fix_gain_get_mem_size(const FixGainConfig* cfg);
FixGain* fix_gain_init(const FixGainConfig* cfg, void* mem, size_t mem_size);

/* No state between calls; provided so a pipeline reset can treat gain stages
 * uniformly. */
void     fix_gain_reset(FixGain* fg);

/* Scales `n` samples of channel `ch` in place, then clips to +/-clip_value when
 * enable_clip is set. Out-of-range `ch`, n <= 0 and a disabled instance are no-ops.
 * The config is fixed at construction. */
void     fix_gain_process(FixGain* fg, float* x, int n, int ch);

void     fix_gain_destroy(FixGain* fg);

/* 10^(db/20), no validation: a non-finite `db` gives a non-finite gain, which
 * fix_gain_get_mem_size() refuses. */
float    fix_gain_db_to_linear(float db);

#ifdef __cplusplus
}
#endif

#endif /* FIX_GAIN_H */
