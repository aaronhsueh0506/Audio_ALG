/* Lib-only stand-in for the call lifecycle that stream/aiaec/ulcnet owns: staging of the error spectrum and the events,
 * one pre_frame per slot, views, validate_run + post_frame or a restart, identity frames. It exists so the lib tests can
 * drive UlcnetPrepost without the stream; the stream's own tests never include it. Header-only, no latch: a call
 * sequence is the caller's. */
#ifndef ULCNET_LIFECYCLE_DRIVER_H
#define ULCNET_LIFECYCLE_DRIVER_H

#include <stdlib.h>
#include <string.h>

#include "ulcnet_prepost.h"

typedef struct UlLife {
    UlcnetPrepost *p;
    int frames;
    size_t bins;
    float *err_re, *err_im;   /* [frames * bins] this call's error spectrum */
    float *enh_re, *enh_im;   /* [frames * bins] the frames of the closed call */
    int *events;              /* [frames] */
} UlLife;

static inline int ul_life_init(UlLife *l, UlcnetPrepost *p) {
    const UlcnetModelIoDescriptor *d = ulcnet_prepost_descriptor(p);
    size_t plane;
    memset(l, 0, sizeof *l);
    if (!d) return -1;
    l->p = p;
    l->frames = ulcnet_model_io_descriptor_frames(d);
    l->bins = (size_t)d->spectrum_bins;
    plane = (size_t)l->frames * l->bins;
    l->err_re = (float *)calloc(plane, sizeof(float));
    l->err_im = (float *)calloc(plane, sizeof(float));
    l->enh_re = (float *)calloc(plane, sizeof(float));
    l->enh_im = (float *)calloc(plane, sizeof(float));
    l->events = (int *)calloc((size_t)l->frames, sizeof(int));
    return l->err_re && l->err_im && l->enh_re && l->enh_im && l->events ? 0 : -1;
}

static inline void ul_life_free(UlLife *l) {
    free(l->err_re); free(l->err_im); free(l->enh_re); free(l->enh_im); free(l->events);
    memset(l, 0, sizeof *l);
}

/* Stage the error spectrum and the events (NULL: none; N entries otherwise) and write every slot's features. 1, or -1. */
static inline int ul_life_pre(UlLife *l, const float *er, const float *ei, const float *fr, const float *fi,
                              const int *events) {
    int k;
    const size_t plane = (size_t)l->frames * l->bins;
    memcpy(l->err_re, er, plane * sizeof(float));
    memcpy(l->err_im, ei, plane * sizeof(float));
    for (k = 0; k < l->frames; ++k) l->events[k] = events ? events[k] != 0 : 0;
    for (k = 0; k < l->frames; ++k) {
        const size_t o = (size_t)k * l->bins;
        if (ulcnet_prepost_pre_frame(l->p, k, er + o, ei + o, fr + o, fi + o, l->events[k]) != 0) return -1;
    }
    return 1;
}

static inline int ul_life_views(UlLife *l, UlcnetModelIoInputs *in, UlcnetModelIoOutputs *out) {
    return ulcnet_prepost_views(l->p, in, out);
}

/* The error spectrum of every frame, state and generation untouched. */
static inline int ul_life_bypass(UlLife *l) {
    const size_t plane = (size_t)l->frames * l->bins * sizeof(float);
    memcpy(l->enh_re, l->err_re, plane);
    memcpy(l->enh_im, l->err_im, plane);
    return 0;
}

/* After a successful run: validate_run, then every non-event frame through post_frame (event frames are the error
 * spectrum). 0; -1 when the run was refused or a frame was not finite (one restart, all frames identity). */
static inline int ul_life_finish(UlLife *l) {
    int k, ok = 0;
    if (ulcnet_prepost_validate_run(l->p) == 0) {
        ok = 1;
        for (k = 0; ok && k < l->frames; ++k) {
            const size_t o = (size_t)k * l->bins;
            if (l->frames > 1 && l->events[k]) {
                memcpy(l->enh_re + o, l->err_re + o, l->bins * sizeof(float));
                memcpy(l->enh_im + o, l->err_im + o, l->bins * sizeof(float));
            } else {
                ok = ulcnet_prepost_post_frame(l->p, k, l->enh_re + o, l->enh_im + o) == 0;
            }
        }
        if (!ok) ulcnet_prepost_zero_state(l->p);
    }
    if (ok) return 0;
    ul_life_bypass(l);
    return -1;
}

/* A run that did not succeed: a started run restarts the state, so does a packet with an event frame (its graph restart
 * never ran); the frames are the error spectrum. */
static inline int ul_life_fail(UlLife *l, int run_started) {
    int k, restart = run_started != 0;
    for (k = 0; !restart && l->frames > 1 && k < l->frames; ++k) restart = l->events[k] != 0;
    if (restart) ulcnet_prepost_zero_state(l->p);
    return ul_life_bypass(l);
}

static inline void ul_life_out(const UlLife *l, float *re, float *im) {
    const size_t plane = (size_t)l->frames * l->bins * sizeof(float);
    memcpy(re, l->enh_re, plane);
    memcpy(im, l->enh_im, plane);
}

#endif
