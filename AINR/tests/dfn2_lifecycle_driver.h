/* Test-only call lifecycle over the DFN2 per-frame lib functions for lib-only builds (no dfn2_stream.c): pre, views, accelerator,
 * validate_run, post, with the application-spectrum and output staging kept here. One state record per instance, found by pointer;
 * create / init / init_ex / reset / destroy go through the dfn2_life_* wrappers below, so a pool reused at one address starts clean. */
#ifndef DFN2_LIFECYCLE_DRIVER_H
#define DFN2_LIFECYCLE_DRIVER_H

#include <stdlib.h>
#include <string.h>

#include "../../lib/ainr/dfn2/dfn2_prepost.h"
#include "../../stream/common/stream_run.h"

#define DFN2_LIFE_MAX 32

typedef struct Dfn2Life {
    const DFN2Prepost *p;
    float *spec_re, *spec_im;   /* [N][n_bins] application spectra of the call */
    float *enh_re, *enh_im;     /* [N][n_bins] enhanced frames of the last call */
    int frame_open, prepared, valid_frames;
} Dfn2Life;

static Dfn2Life dfn2_life_table[DFN2_LIFE_MAX];

static inline void dfn2_life_forget(const DFN2Prepost *p)
{
    int i;
    for (i = 0; i < DFN2_LIFE_MAX; ++i) {
        if (dfn2_life_table[i].p != p) continue;
        free(dfn2_life_table[i].spec_re);
        free(dfn2_life_table[i].spec_im);
        free(dfn2_life_table[i].enh_re);
        free(dfn2_life_table[i].enh_im);
        memset(&dfn2_life_table[i], 0, sizeof dfn2_life_table[i]);
    }
}

static inline Dfn2Life *dfn2_life_of(DFN2Prepost *p)
{
    int i, slot = -1;
    size_t count;
    const DFN2Grid *g = dfn2_prepost_grid(p);
    if (!p || !g) return NULL;
    for (i = 0; i < DFN2_LIFE_MAX; ++i) {
        if (dfn2_life_table[i].p == p) return &dfn2_life_table[i];
        if (slot < 0 && !dfn2_life_table[i].p) slot = i;
    }
    if (slot < 0) abort();
    count = (size_t)g->frames_per_call * (size_t)g->n_bins;
    dfn2_life_table[slot].p = p;
    dfn2_life_table[slot].spec_re = (float *)calloc(count, sizeof(float));
    dfn2_life_table[slot].spec_im = (float *)calloc(count, sizeof(float));
    dfn2_life_table[slot].enh_re = (float *)calloc(count, sizeof(float));
    dfn2_life_table[slot].enh_im = (float *)calloc(count, sizeof(float));
    if (!dfn2_life_table[slot].spec_re || !dfn2_life_table[slot].spec_im || !dfn2_life_table[slot].enh_re ||
        !dfn2_life_table[slot].enh_im)
        abort();
    return &dfn2_life_table[slot];
}

static inline DFN2Prepost *dfn2_life_create(const DFN2PrepostConfig *cfg)
{
    DFN2Prepost *p = dfn2_prepost_create(cfg);
    if (p) { dfn2_life_forget(p); (void)dfn2_life_of(p); }
    return p;
}

static inline DFN2Prepost *dfn2_life_init_ex(void *pool, size_t bytes, const DFN2PrepostConfig *cfg, const DFN2PrepostMemReq *expected)
{
    DFN2Prepost *p = dfn2_prepost_init_ex(pool, bytes, cfg, expected);
    if (p) { dfn2_life_forget(p); (void)dfn2_life_of(p); }
    return p;
}

static inline DFN2Prepost *dfn2_life_init(void *pool, size_t bytes, const DFN2PrepostConfig *cfg)
{
    return dfn2_life_init_ex(pool, bytes, cfg, NULL);
}

static inline void dfn2_life_destroy(DFN2Prepost *p)
{
    dfn2_life_forget(p);
    dfn2_prepost_destroy(p);
}

static inline void dfn2_life_reset(DFN2Prepost *p)
{
    Dfn2Life *l = dfn2_life_of(p);
    dfn2_prepost_reset(p);
    if (l) {
        const DFN2Grid *g = dfn2_prepost_grid(p);
        const size_t bytes = (size_t)g->frames_per_call * (size_t)g->n_bins * sizeof(float);
        memset(l->enh_re, 0, bytes);
        memset(l->enh_im, 0, bytes);
        l->frame_open = l->prepared = l->valid_frames = 0;
    }
}

/* One frame of the call: the head of `slot` when committing, else the identity; out is the call's enhanced plane `slot`. */
static inline int dfn2_life_post_one(DFN2Prepost *p, Dfn2Life *l, int commit, int slot, int *valid)
{
    const size_t nb = (size_t)dfn2_prepost_grid(p)->n_bins, off = (size_t)slot * nb;
    const int r = commit ? dfn2_prepost_post_frame(p, slot, l->spec_re + off, l->spec_im + off, l->enh_re + off, l->enh_im + off)
                         : dfn2_prepost_post_frame_identity(p, l->spec_re + off, l->spec_im + off, l->enh_re + off, l->enh_im + off);
    if (r < 0) return -1;
    *valid += r;
    return 0;
}

static inline int dfn2_life_close(DFN2Prepost *p, Dfn2Life *l, int commit)
{
    int i, valid = 0;
    for (i = 0; i < dfn2_prepost_grid(p)->frames_per_call; ++i)
        if (dfn2_life_post_one(p, l, commit, i, &valid) != 0) return -1;
    l->frame_open = 0;
    l->prepared = 0;
    l->valid_frames = valid;
    return 0;
}

/* N == 1: 1 on every hop from hop L (a frame is open), 0 before (the frame is queued, nothing open), -1 on a misuse. */
static inline int dfn2_life_pre_process_freq_dual(DFN2Prepost *p, const float *est_re, const float *est_im, const float *app_re,
                                           const float *app_im)
{
    Dfn2Life *l = dfn2_life_of(p);
    const DFN2Grid *g = dfn2_prepost_grid(p);
    int r;
    if (!l || !est_re || !est_im || !app_re || !app_im || l->frame_open || g->frames_per_call != 1) return -1;
    l->prepared = 0;
    l->valid_frames = 0;
    memcpy(l->spec_re, app_re, (size_t)g->n_bins * sizeof(float));
    memcpy(l->spec_im, app_im, (size_t)g->n_bins * sizeof(float));
    r = dfn2_prepost_pre_frame(p, est_re, est_im);
    if (r < 0) return -1;
    if (r == 0) {
        int valid = 0;
        if (dfn2_life_post_one(p, l, 0, 0, &valid) != 0) return -1;
        return 0;
    }
    l->frame_open = 1;
    return 1;
}

static inline int dfn2_life_pre_process_freq(DFN2Prepost *p, const float *re, const float *im)
{
    return dfn2_life_pre_process_freq_dual(p, re, im, re, im);
}

/* N > 1: one job per call. */
static inline int dfn2_life_pre_process_block_dual(DFN2Prepost *p, const float *est_re, const float *est_im, const float *app_re,
                                            const float *app_im)
{
    Dfn2Life *l = dfn2_life_of(p);
    const DFN2Grid *g = dfn2_prepost_grid(p);
    const size_t nb = p ? (size_t)g->n_bins : 0u;
    int i;
    if (!l || !est_re || !est_im || !app_re || !app_im || l->frame_open || g->frames_per_call < 2) return -1;
    l->prepared = 0;
    l->valid_frames = 0;
    if (l->spec_re != app_re) memcpy(l->spec_re, app_re, nb * (size_t)g->frames_per_call * sizeof(float));
    if (l->spec_im != app_im) memcpy(l->spec_im, app_im, nb * (size_t)g->frames_per_call * sizeof(float));
    for (i = 0; i < g->frames_per_call; ++i)
        if (dfn2_prepost_pre_frame(p, est_re + (size_t)i * nb, est_im + (size_t)i * nb) < 0) return -1;
    l->frame_open = 1;
    return 1;
}

static inline int dfn2_life_block_input(DFN2Prepost *p, float **re, float **im)
{
    Dfn2Life *l = dfn2_life_of(p);
    if (!l || !re || !im || l->frame_open || dfn2_prepost_grid(p)->frames_per_call < 2) return -1;
    *re = l->spec_re;
    *im = l->spec_im;
    return 0;
}

static inline int dfn2_life_block_output(DFN2Prepost *p, const float **re, const float **im)
{
    Dfn2Life *l = dfn2_life_of(p);
    if (!l || !re || !im || l->frame_open || dfn2_prepost_grid(p)->frames_per_call < 2) return -1;
    *re = l->enh_re;
    *im = l->enh_im;
    return 0;
}

static inline int dfn2_life_frame_open(DFN2Prepost *p, DFN2PrepostInputs *in, DFN2PrepostOutputs *out)
{
    Dfn2Life *l = dfn2_life_of(p);
    if (!l || !in || !out || !l->frame_open) return -1;
    if (dfn2_prepost_views(p, in, out) != 0) return -1;
    l->prepared = 1;
    return 0;
}

/* -1 with the frame left open when the run is refused (state zeroed, generation advanced once). */
static inline int dfn2_life_frame_finish(DFN2Prepost *p)
{
    Dfn2Life *l = dfn2_life_of(p);
    if (!l || !l->prepared) return -1;
    if (dfn2_prepost_validate_run(p) != 0) {
        l->prepared = 0;
        return -1;
    }
    return dfn2_life_close(p, l, 1);
}

static inline int dfn2_life_frame_bypass(DFN2Prepost *p)
{
    Dfn2Life *l = dfn2_life_of(p);
    if (!l || !l->frame_open) return -1;
    return dfn2_life_close(p, l, 0);
}

static inline int dfn2_life_frame_fail(DFN2Prepost *p, int run_started)
{
    Dfn2Life *l = dfn2_life_of(p);
    if (!l || !l->frame_open) return -1;
    if (run_started) dfn2_prepost_zero_state(p);
    return dfn2_life_frame_bypass(p);
}

static inline int dfn2_life_post_process_freq(DFN2Prepost *p, float *re, float *im, int *valid)
{
    Dfn2Life *l = dfn2_life_of(p);
    const DFN2Grid *g = dfn2_prepost_grid(p);
    if (!l || !re || !im || l->frame_open || g->frames_per_call != 1) return -1;
    memcpy(re, l->enh_re, (size_t)g->n_bins * sizeof(float));
    memcpy(im, l->enh_im, (size_t)g->n_bins * sizeof(float));
    if (valid) *valid = l->valid_frames;
    return 0;
}

static inline int dfn2_life_post_process_block(DFN2Prepost *p, float *re, float *im, int *valid_frames)
{
    Dfn2Life *l = dfn2_life_of(p);
    const DFN2Grid *g = dfn2_prepost_grid(p);
    size_t bytes;
    if (!l || l->frame_open || g->frames_per_call < 2) return -1;
    bytes = (size_t)g->frames_per_call * (size_t)g->n_bins * sizeof(float);
    if (re) memcpy(re, l->enh_re, bytes);
    if (im) memcpy(im, l->enh_im, bytes);
    if (valid_frames) *valid_frames = l->valid_frames;
    return 0;
}

/* 1 committed, 0 skipped, -1 contract error (NULL args, no infer, bad io_descriptor, no frame open). */
static inline int dfn2_life_model_run_frame(const DFN2Model *model, DFN2Prepost *p)
{
    Dfn2Life *l = dfn2_life_of(p);
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    int run_status;
    if (!model || !model->infer || !l || !l->frame_open ||
        dfn2_model_io_descriptor_validate(dfn2_prepost_grid(p), model->io_descriptor) != 0)
        return -1;
    if (dfn2_life_frame_open(p, &inputs, &outputs) != 0) return -1;
    run_status = model->infer(model->user, &inputs, &outputs);
    if (run_status != STREAM_RUN_OK)
        return dfn2_life_frame_fail(p, run_status != STREAM_RUN_NOT_STARTED) == 0 ? 0 : -1;
    if (dfn2_life_frame_finish(p) != 0) return dfn2_life_frame_bypass(p) == 0 ? 0 : -1;
    return 1;
}

#endif /* DFN2_LIFECYCLE_DRIVER_H */
