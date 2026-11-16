/* Test-only sequencer for lib-only DeepVQE-S builds: the per-call order the stream (deepvqe_stream.c) runs over
 * the per-frame functions of deepvqe_prepost.h, with the return numbers of the former batch API, so a driver can
 * run a whole call without a stream. The library keeps no latch; the open call, the staged raw mic spectra and
 * the output frames live here, per instance, in a small table keyed by the instance pointer.
 *
 *     dvl_pre_process_freq   N calls; the Nth returns 1, the others 0; -1 on NULL or while a call is open
 *     dvl_frame_open         views; -1 when no call is open
 *     dvl_frame_finish       validate_run, then post_frame x N; -1 leaves the call open for bypass
 *     dvl_frame_bypass       the call's N output frames become zeros
 *     dvl_frame_fail         bypass, after zero_state when the run started
 *     dvl_post_process_freq  the call's frames, one per call, oldest first
 *
 * Not a second implementation under test: stream/aiaec/deepvqe_s/tests/verify_vs_sample.c keeps its own. */
#ifndef DEEPVQE_LIFECYCLE_DRIVER_H
#define DEEPVQE_LIFECYCLE_DRIVER_H

#include <stdlib.h>
#include <string.h>

#include "deepvqe_prepost.h"

#define DVL_MAX_INSTANCES 32

typedef struct DvlCall {
    DeepVqePrepost *p;
    int n, n_bins;
    int pending, deliver, open, prepared;
    float *mic_re, *mic_im, *enh_re, *enh_im;   /* [N * n_bins] */
} DvlCall;

static DvlCall g_dvl[DVL_MAX_INSTANCES];

static inline void dvl_drop(DvlCall *c) {
    free(c->mic_re);
    memset(c, 0, sizeof *c);
}

/* Forget every instance (a case boundary) or one (before its memory goes away). */
static inline void dvl_forget_all(void) {
    int i;
    for (i = 0; i < DVL_MAX_INSTANCES; ++i) dvl_drop(&g_dvl[i]);
}

static inline void dvl_forget(const DeepVqePrepost *p) {
    int i;
    for (i = 0; i < DVL_MAX_INSTANCES; ++i)
        if (g_dvl[i].p == p) dvl_drop(&g_dvl[i]);
}

/* The record of `p`, created on first use from the instance's descriptor sizes. NULL when `p` is NULL. */
static inline DvlCall *dvl_call(DeepVqePrepost *p, int n, int n_bins) {
    int i;
    DvlCall *free_slot = NULL;
    if (!p) return NULL;
    for (i = 0; i < DVL_MAX_INSTANCES; ++i) {
        if (g_dvl[i].p == p) return &g_dvl[i];
        if (!g_dvl[i].p && !free_slot) free_slot = &g_dvl[i];
    }
    if (!free_slot) exit(2);
    free_slot->p = p;
    free_slot->n = n;
    free_slot->n_bins = n_bins;
    free_slot->mic_re = (float *)calloc(4u * (size_t)n * (size_t)n_bins, sizeof(float));
    if (!free_slot->mic_re) exit(2);
    free_slot->mic_im = free_slot->mic_re + (size_t)n * (size_t)n_bins;
    free_slot->enh_re = free_slot->mic_im + (size_t)n * (size_t)n_bins;
    free_slot->enh_im = free_slot->enh_re + (size_t)n * (size_t)n_bins;
    return free_slot;
}

/* The grid of the instance under test: set once by the driver (dvl_use_grid) before the first call. */
static int g_dvl_n = 1, g_dvl_bins = 0;
static inline void dvl_use_grid(int frames_per_call, int n_bins) { g_dvl_n = frames_per_call; g_dvl_bins = n_bins; }
static inline DvlCall *dvl_of(DeepVqePrepost *p) { return dvl_call(p, g_dvl_n, g_dvl_bins); }

static inline int dvl_pre_process_freq(DeepVqePrepost *p, const float *mic_re, const float *mic_im,
                                const float *far_re, const float *far_im) {
    DvlCall *c = dvl_of(p);
    size_t at;
    if (!c || !mic_re || !mic_im || !far_re || !far_im) return -1;
    if (c->open) return -1;
    at = (size_t)c->pending * (size_t)c->n_bins;
    memcpy(c->mic_re + at, mic_re, (size_t)c->n_bins * sizeof(float));
    memcpy(c->mic_im + at, mic_im, (size_t)c->n_bins * sizeof(float));
    if (deepvqe_prepost_pre_frame(p, c->pending, mic_re, mic_im, far_re, far_im) != DEEPVQE_PREPOST_OK) return -1;
    c->pending += 1;
    if (c->pending < c->n) return 0;
    c->pending = 0;
    c->open = 1;
    return 1;
}

static inline int dvl_frame_open(DeepVqePrepost *p, DeepVqePrepostInputs *in, DeepVqePrepostOutputs *out) {
    DvlCall *c = dvl_of(p);
    if (!c || !in || !out || !c->open) return -1;
    if (deepvqe_prepost_views(p, in, out) != 0) return -1;
    c->prepared = 1;
    return 0;
}

static inline int dvl_frame_finish(DeepVqePrepost *p) {
    DvlCall *c = dvl_of(p);
    int k;
    if (!c || !c->prepared) return -1;
    if (deepvqe_prepost_validate_run(p) != 0) {
        c->prepared = 0;
        return -1;
    }
    for (k = 0; k < c->n; ++k) {
        const size_t at = (size_t)k * (size_t)c->n_bins;
        if (deepvqe_prepost_post_frame(p, k, c->mic_re + at, c->mic_im + at, c->enh_re + at, c->enh_im + at) !=
            DEEPVQE_PREPOST_OK) break;
    }
    if (k != c->n) {
        deepvqe_prepost_zero_state(p);
        c->prepared = 0;
        return -1;
    }
    c->deliver = 0;
    c->open = 0;
    c->prepared = 0;
    return 0;
}

static inline int dvl_frame_bypass(DeepVqePrepost *p) {
    DvlCall *c = dvl_of(p);
    size_t bytes;
    if (!c || !c->open) return -1;
    bytes = (size_t)c->n * (size_t)c->n_bins * sizeof(float);
    memset(c->enh_re, 0, bytes);
    memset(c->enh_im, 0, bytes);
    c->deliver = 0;
    c->open = 0;
    c->prepared = 0;
    return 0;
}

static inline int dvl_frame_fail(DeepVqePrepost *p, int run_started) {
    DvlCall *c = dvl_of(p);
    if (!c || !c->open) return -1;
    if (run_started) deepvqe_prepost_zero_state(p);
    return dvl_frame_bypass(p);
}

static inline int dvl_post_process_freq(DeepVqePrepost *p, float *re, float *im) {
    DvlCall *c = dvl_of(p);
    size_t at;
    if (!c || !re || !im || c->open) return -1;
    at = (size_t)c->deliver * (size_t)c->n_bins;
    memcpy(re, c->enh_re + at, (size_t)c->n_bins * sizeof(float));
    memcpy(im, c->enh_im + at, (size_t)c->n_bins * sizeof(float));
    c->deliver = (c->deliver + 1) % c->n;
    return 0;
}

/* deepvqe_prepost_reset() plus the open call dropped, as the former reset did. */
static inline void dvl_reset(DeepVqePrepost *p) {
    DvlCall *c = dvl_of(p);
    deepvqe_prepost_reset(p);
    if (c) { c->pending = c->deliver = c->open = c->prepared = 0; }
}

#endif /* DEEPVQE_LIFECYCLE_DRIVER_H */
