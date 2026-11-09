/** Sample-level glue of the four-channel programs between mcab_fuse() and the libraries behind it
 * (not part of the mcab module or its archive). Header-only, no ABI: copy and adapt.
 *
 * Per hop, over the McabFrame mcab_fuse() returned:
 *     AecResInput in;  mcab_sample_res_input(&frame, &in);           // analysis: E, Y, R^2, N^2
 *     G_res = aec_res_process(res, &in);
 *     G_nr  = mcab_sample_anr_gain(nr, &frame, scratch, use_r2);       // estimated on E
 *     if (mcab_sample_post_reset_due(&handled, &frame)) { ac_istft_reset(istft); rng = MCAB_SAMPLE_CNG_RNG_SEED; }
 *     P = B * min(G_nr, G_res);  aec_cng_fill(&rng, P, frame.cn, G_res, G_nr,   // B = frame.b
 *                                             MCAB_SAMPLE_CNG_NR_GAIN_FLOOR, n_freqs);
 *     ac_istft_push(istft, P, out);
 * with rng started (and restarted on every reset) at MCAB_SAMPLE_CNG_RNG_SEED and `handled` at 0
 * (on fresh lanes the first frame matches; otherwise the first hop clears a synthesizer that is
 * already clear). mcab.h: analysis vs rendering, post-chain resets. */
#ifndef MCAB_SAMPLE_POST_H
#define MCAB_SAMPLE_POST_H

#include <stddef.h>
#include <string.h>

#include "aec_res.h"
#include "anr.h"
#include "mcab.h"               /* aec.h: AEC_CNG_SEED */
#include "utility/simd_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Comfort noise follows the denoiser gain of its bin but never below this amplitude (-10 dB),
 * so the residual echo it masks does not surface. */
#define MCAB_SAMPLE_CNG_NR_GAIN_FLOOR 0.31622777f

/* Start state of the comfort-noise generator (aec_cng_fill): the library's own seed, so the
 * comfort noise restarts where the lane's would after a post-chain reset. */
#define MCAB_SAMPLE_CNG_RNG_SEED AEC_CNG_SEED

/* The residual-echo powers are on the int16^2 scale, the denoiser's input on the audio scale;
 * the reciprocal is a power of two, so the multiply is exact. */
#define MCAB_SAMPLE_PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))

/** The libaec RES input of one fused hop (pointers alias the frame). A NULL frame yields an
 * all-zero input, which aec_res_process() refuses. */
static inline void mcab_sample_res_input(const McabFrame* f, AecResInput* in) {
    if (!in) return;
    memset(in, 0, sizeof(*in));
    if (!f) return;
    in->error_spec = f->e;
    in->near_spec = f->y;
    in->r2 = f->r2;
    in->comfort_noise = f->cn;
    in->render_hop = f->far_hop;
    in->all_converged = f->all_converged;
    in->max_dt = f->max_dt;
    in->max_saturation = f->max_sat;
    in->clock_drift = f->clock_drift;
}

/** The denoiser gain of one fused hop. With use_r2 the residual-echo powers f->r2 fold into the
 * noise floor as r2 / 32768^2 through `scratch` (n_freqs floats); use_r2 = 0, NULL r2/scratch
 * or f->usable_linear = 0 selects plain noise reduction. Returns n_freqs gains owned by the
 * denoiser, valid until its next call; NULL for a NULL argument or a refused hop. */
static inline const float* mcab_sample_anr_gain(AnrDenoiser* nr, const McabFrame* f,
                                                float* scratch, int use_r2) {
    const float* extra = NULL;
    if (!nr || !f || !f->e) return NULL;
    if (use_r2 && f->r2 && scratch && f->usable_linear) {
        sk_scale_f32(scratch, f->r2, MCAB_SAMPLE_PSD_SCALE_INV, f->n_freqs);
        extra = scratch;
    }
    if (anr_process_gain(nr, f->e, extra, NULL) != 0) return NULL;
    return anr_get_gain(nr, NULL);
}

/** 1 when the frame's post_reset_serial differs from *handled (a lane had a hard post-chain
 * reset since the synthesizer's last hop) and stores it; then, before this hop's comfort-noise
 * draw and inverse transform, the synthesizer clears its overlap-add tail and reseeds its
 * comfort noise to MCAB_SAMPLE_CNG_RNG_SEED. 0 otherwise or for a NULL argument. */
static inline int mcab_sample_post_reset_due(uint32_t* handled, const McabFrame* f) {
    if (!handled || !f || f->post_reset_serial == *handled) return 0;
    *handled = f->post_reset_serial;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* MCAB_SAMPLE_POST_H */
