#ifndef POST_GAIN_H
#define POST_GAIN_H

#include <stddef.h>

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Directional post-gain: keeps bins whose dominant direction agrees with the
 * steered angle and suppresses the rest, smoothed over frequency and time. The
 * spectrum is audiolib's `Complex` ({float r; float i;}); convert other complex
 * types with a cast at the call site. */

typedef struct {
    int F;
    int num_angles;

    int enable;

    float gain_match;
    float gain_suppress;

    int angle_tol;
    float angle_vad_thr;

    int enable_freq_smooth;
    int freq_smooth_radius;

    int enable_time_smooth;
    float attack_alpha;
    float release_alpha;

    float min_gain;
    float max_gain;

    int enable_mask_relax;
    int mask_relax_bins;

} PostGainConfig;

typedef struct {
    int cnt_match;
    int cnt_suppress;
} PostGainStats;

/* Opaque type; layout hidden so `owned_heap` stays out of caller reach. */
typedef struct PostGainState PostGainState;

/* Heap constructor: one block sized by post_gain_get_mem_size(). */
PostGainState* post_gain_create(const PostGainConfig* cfg);

/* Caller-pool constructor: `mem` 16-byte aligned, `mem_size` >=
 * post_gain_get_mem_size(cfg), else NULL. The caller owns the memory;
 * post_gain_destroy() does not free it. */
size_t         post_gain_get_mem_size(const PostGainConfig* cfg);
PostGainState* post_gain_init(const PostGainConfig* cfg,
                              void* mem, size_t mem_size);

void post_gain_destroy(PostGainState* st);

/* Shapes `Y` in place. The config is taken per call and may be retuned between
 * hops, but `F` must not change (a differing `cfg->F` is a no-op). `gain_match`
 * also seeds the smoother at construction. A non-finite `doa_used` means no
 * usable steering angle: all bins are suppressed and `Y` is left untouched. */
void post_gain_apply(PostGainState* st,
                     const PostGainConfig* cfg,
                     Complex* Y,
                     const int* mask,
                     const int* bin_best_idx,
                     float doa_used);

const int* post_gain_get_raw_mask(const PostGainState* st);
const int* post_gain_get_class(const PostGainState* st);
/* Per-bin gain applied on the last frame (the smoother's carry-in). Length F. */
const float* post_gain_get_gain(const PostGainState* st);
const PostGainStats* post_gain_get_stats(const PostGainState* st);

#ifdef __cplusplus
}
#endif

#endif /* POST_GAIN_H */
