/* se_stage.h -- the view exchanged by the frequency-domain stages of the AEC
 * pipelines.
 *
 * The AEC returns its context spectra once per hop; later stages are plain
 * functions over the borrowed SeSpecFrame. A frame is valid until the producer
 * runs again (next process/front call or reset) and is never written through.
 * Scales: e, y, x in audio scale; r2 and cn in int16^2 power scale; gains are
 * amplitudes. */
#ifndef PIPELINES_SE_STAGE_H
#define PIPELINES_SE_STAGE_H

#include <math.h>

#include "aec_res.h"
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Comfort noise follows the denoiser gain of its bin but never below this
 * amplitude (-10 dB), so the residual echo it masks does not surface. */
#define SE_CNG_NR_GAIN_FLOOR 0.31622777f

/* Start state of every application's comfort-noise generator (aec_cng_fill). */
#define SE_CNG_RNG_SEED 0x9e3779b9u

typedef struct SeSpecFrame {
    int n_freqs;              /* bins of every spectrum / gain vector            */
    int hop;                  /* time-domain hop of far_hop / formed_hop         */
    const Complex* e;         /* linear-AEC error E(f); for 4 channels the beamformed error */
    const Complex* y;         /* capture spectrum matching e; NULL when not kept  */
    const Complex* x;         /* far-end spectrum; NULL when not kept             */
    const float* r2;          /* residual-echo PSD R^2(f); NULL when not computed */
    const float* cn;          /* comfort-noise PSD N^2(f); NULL when not computed */
    const float* g_res_ctx;   /* the AEC's own residual gain; NULL when its lanes compute none */
    const float* far_hop;     /* aligned far-end samples of this hop; NULL when not kept */
    const float* formed_hop;  /* time-domain hop underlying e; NULL when not kept */
    float far_power;
    float max_dt;             /* worst double-talk indicator                      */
    float max_sat;            /* worst saturation level                           */
    int all_converged;        /* every contributing linear filter has converged   */
    int usable_linear;        /* r2 derives from a usable linear estimate */
    int clock_drift;
    int delay_changed;        /* this hop starts a new alignment generation       */
    int delay_locked;         /* a usable alignment is in force                   */
} SeSpecFrame;

/* Residual-echo gain over `frame` (needs e, y, r2, cn and far_hop): n_freqs floats
 * owned by `res`, valid until its next call; NULL for a NULL argument or a frame
 * lacking one of those inputs. */
static inline const float* se_res_stage_gain(AecRes* res, const SeSpecFrame* frame) {
    AecResInput in;
    if (!res || !frame) return NULL;
    in.error_spec = frame->e;
    in.near_spec = frame->y;
    in.r2 = frame->r2;
    in.comfort_noise = frame->cn;
    in.render_hop = frame->far_hop;
    in.all_converged = frame->all_converged;
    in.max_dt = frame->max_dt;
    in.max_saturation = frame->max_sat;
    in.clock_drift = frame->clock_drift;
    return aec_res_process(res, &in);
}

/* The SeSpecFrame view of one hop of a mono Aec after aec_process_context(). The lane
 * must carry the seam (enable_res = 0, return_res_context = 1, as se_derive builds it);
 * its own gain is exposed as g_res_ctx. All pointers alias the AEC and are valid until its
 * next processing call or reset. Returns 0, or -1 for a NULL argument or a lane without
 * the seam. */
static inline int se_frame_from_aec(const Aec* aec, SeSpecFrame* frame) {
    AecResContext ctx;
    AecLinearContext lin;
    if (!aec || !frame) return -1;
    aec_get_res_context(aec, &ctx);
    aec_get_linear_context(aec, &lin);
    if (!ctx.error_spec) return -1;
    frame->n_freqs = ctx.n_freqs;
    frame->hop = ctx.hop_size;
    frame->e = ctx.error_spec;
    frame->y = ctx.near_spec;
    frame->x = ctx.far_spec;
    frame->r2 = ctx.r2;
    frame->cn = ctx.comfort_noise;
    frame->g_res_ctx = ctx.res_gain;
    frame->far_hop = lin.aligned_far_hop;
    frame->formed_hop = ctx.formed_hop;
    frame->far_power = ctx.far_power;
    frame->max_dt = ctx.dt_indicator;
    frame->max_sat = ctx.saturation_level;
    frame->all_converged = ctx.filter_converged;
    frame->usable_linear = ctx.usable_linear;
    frame->clock_drift = 0;
    frame->delay_changed = lin.delay_state == AEC_LINEAR_DELAY_CHANGED;
    frame->delay_locked = lin.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
    return 0;
}

/* 1 when all n samples are finite: the check an integrator runs on an emitted hop before
 * trusting it. */
static inline int se_hop_is_finite(const float* x, int n) {
    int k;
    for (k = 0; k < n; ++k)
        if (!isfinite(x[k])) return 0;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SE_STAGE_H */
