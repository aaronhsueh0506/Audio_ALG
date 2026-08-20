#ifndef POST_GAIN_H
#define POST_GAIN_H

#include <stddef.h>

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Directional post-gain: keeps bins whose dominant direction agrees with the
 * steered angle and suppresses the rest, then smooths the result over
 * frequency and time before applying it to the spectrum.
 *
 * The spectrum is audio_common's backend-neutral `Complex`, not a KISS type,
 * so the same header and archive serve BACKEND=kiss and BACKEND=ne10 and the
 * shared complex kernels apply to it directly. Integrators still holding a
 * `kiss_fft_cpx*` are byte-compatible -- both are {float r; float i;} -- but
 * must spell the conversion as a cast at the call site rather than relying on
 * the two names being interchangeable. */

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

/* Opaque, like FixGain and VADApi: every field is reachable through the
 * accessors below, and hiding the layout keeps `owned_heap` -- the flag that
 * decides whether destroy() frees -- out of caller reach, which is the whole
 * point of the pool contract. */
typedef struct PostGainState PostGainState;

/* Heap constructor: one block sized by post_gain_get_mem_size(), carved by
 * the same code path as post_gain_init(). */
PostGainState* post_gain_create(const PostGainConfig* cfg);

/* Caller-pool constructor. `mem` must be 16-byte aligned and `mem_size` at
 * least post_gain_get_mem_size(cfg), else NULL. The caller owns the memory
 * and post_gain_destroy() will not free it. */
size_t         post_gain_get_mem_size(const PostGainConfig* cfg);
PostGainState* post_gain_init(const PostGainConfig* cfg,
                              void* mem, size_t mem_size);

void post_gain_destroy(PostGainState* st);

/* Shapes `Y` in place. The config is taken per call, so a caller may retune
 * between hops without rebuilding -- but `F` must not change (a differing
 * `cfg->F` is a no-op), and `gain_match` is read once at construction to seed
 * the smoother as well as per call, so changing it retunes the target without
 * moving the state the smoother starts from.
 *
 * A `doa_used` that is not finite means "no usable steering angle": the frame
 * is classified all-suppress and `Y` is left untouched. */
void post_gain_apply(PostGainState* st,
                     const PostGainConfig* cfg,
                     Complex* Y,
                     const int* mask,
                     const int* bin_best_idx,
                     float doa_used);

const int* post_gain_get_raw_mask(const PostGainState* st);
const int* post_gain_get_class(const PostGainState* st);
/* The per-bin gain actually applied on the last frame, and the smoother's
 * carry-in for the next one. Length F. */
const float* post_gain_get_gain(const PostGainState* st);
const PostGainStats* post_gain_get_stats(const PostGainState* st);

#ifdef __cplusplus
}
#endif

#endif /* POST_GAIN_H */
