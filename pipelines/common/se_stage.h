/* pipelines/common/se_stage.h -- the view exchanged by the frequency-domain
 * stages of the AEC pipelines.
 *
 * Each application analyses once per hop: the AEC takes time-domain mic/ref
 * and returns its context spectra. Every later stage is a plain function over
 * the borrowed SeSpecFrame below; the application writes the combination of
 * their results itself:
 *
 *     gain stages       const float* <stage>_gain(stage, frame)
 *     min of the gains  sk_min_f32
 *     apply             sk_capply_gain_f32
 *     comfort noise     aec_cng_fill
 *     synthesis         ac_istft_push
 *
 * A frame borrows the producer's per-hop buffers: it is valid until the
 * producer runs again (the next process/front call or reset) and is never
 * written through. Scales: e, y, x in audio scale; r2 and cn in int16^2 power
 * scale; gains are amplitudes.
 */
#ifndef PIPELINES_SE_STAGE_H
#define PIPELINES_SE_STAGE_H

#include "aec_res.h"
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Comfort noise is scaled by the denoiser gain of its bin, but never below this
 * amplitude (-10 dB): the fill follows the denoised floor while enough of it
 * remains to keep the residual echo it masks from surfacing. */
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
    int clock_drift;
    int delay_changed;        /* this hop starts a new alignment generation       */
    int delay_locked;         /* a usable alignment is in force                   */
} SeSpecFrame;

/* Residual-echo gain of the post-beam suppressor over `frame` (needs e, y, r2,
 * cn and far_hop). The gain is n_freqs floats owned by `res`, valid until the
 * next call on it; NULL for a NULL argument or a frame lacking one of those
 * inputs (aec_res_process refuses it). */
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

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SE_STAGE_H */
