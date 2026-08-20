#ifndef FIX_GAIN_H
#define FIX_GAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-channel input gain with an optional clip.
 *
 * audio_common's AudioPreGain (audio_pre_gain.h) is the canonical single-gain
 * input stage for this stack; this module is the multi-channel variant the
 * integrator's capture path needs -- one gain per microphone plus the clip --
 * and it shapes the signal with the same shared kernels. Prefer AudioPreGain
 * for a mono path; reach for this one when the per-channel table is what you
 * actually need. */

typedef struct {
    int channels;
    int enable;

    /* Rejected as a whole config if any gain or clip_value is non-finite:
     * these multiply the signal directly, so a NaN here becomes NaN audio.
     * A global_gain of exactly 0 is rewritten to 1.0 rather than muting --
     * inherited behaviour, kept so existing tuning files still mean what they
     * meant. Use `enable` to bypass; there is no way to mute through gain. */
    float global_gain;

    /* length = channels, NULL means 1.0 for all. Read only during
     * construction -- the module keeps its own copy, so the caller's array
     * does not have to outlive the FixGain. */
    const float* channel_gain;

    int enable_clip;
    float clip_value;           /* usually 1.0 for float audio; <= 0 means 1.0 */
} FixGainConfig;

typedef struct FixGain FixGain;

/* Heap constructor: one block sized by fix_gain_get_mem_size(), carved by the
 * same code path as fix_gain_init(). */
FixGain* fix_gain_create(const FixGainConfig* cfg);

/* Caller-pool constructor. `mem` must be 16-byte aligned and `mem_size` at
 * least fix_gain_get_mem_size(cfg), else NULL. The caller owns the memory and
 * fix_gain_destroy() will not free it.
 *
 * Argument order is (cfg, mem, mem_size), matching vad_api_init() and the
 * integrator call sites that already drive this module.
 *
 * fix_gain_get_mem_size() returns 0 for a config it will not construct, so it
 * is also the config gate: channels <= 0, or any non-finite gain/clip value. */
size_t   fix_gain_get_mem_size(const FixGainConfig* cfg);
FixGain* fix_gain_init(const FixGainConfig* cfg, void* mem, size_t mem_size);

/* The instance holds no state between calls; provided so a pipeline-level
 * reset can treat the three gain stages uniformly. */
void     fix_gain_reset(FixGain* fg);

/* Scales `n` samples of channel `ch` in place, then clips to +/-clip_value
 * when enable_clip is set. Out-of-range `ch`, n <= 0 and a disabled instance
 * are all no-ops. The config is fixed at construction -- there is no retune
 * path; build a new instance to change gains. */
void     fix_gain_process(FixGain* fg, float* x, int n, int ch);

void     fix_gain_destroy(FixGain* fg);

/* 10^(db/20). Pure conversion with NO validation: a non-finite `db` gives a
 * non-finite gain, which fix_gain_get_mem_size() then refuses. audio_common's
 * audio_pre_gain_set_gain_db() is the validating entry point for the same
 * convention. */
float    fix_gain_db_to_linear(float db);

#ifdef __cplusplus
}
#endif

#endif /* FIX_GAIN_H */
