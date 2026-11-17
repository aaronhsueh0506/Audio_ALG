/** Sample-level glue of the four-channel programs between aec_get_residual() of an array instance and the
 * libraries behind it (not part of a delivered archive). Header-only, no ABI: copy and adapt.
 *
 * Per hop, after aec_process_residual(aec, &beam) and aec_get_residual(aec, &v):
 *     G_nr = array_sample_anr_gain(nr, &v, scratch, use_r2);              // estimated on B (v.base_spec)
 *     if (array_sample_post_reset_due(&handled, &v)) ac_istft_reset(istft);
 *     aec_apply_gain(aec, G_nr, P);                                       // P = B * min(G_nr, G_res), B = v.base_spec
 *     aec_apply_cng(aec, G_nr, P);                                        // comfort noise on the bins the gain cut
 *     ac_istft_push(istft, P, out);
 * with `handled` at 0 (on a fresh array the first frame matches; otherwise the first hop clears a synthesizer that is
 * already clear). The comfort-noise generator is the array's own and restarts at AEC_CNG_SEED by itself. aec.h:
 * aec_post_reset_serial(). */
#ifndef ARRAY_SAMPLE_POST_H
#define ARRAY_SAMPLE_POST_H

#include <stddef.h>
#include <string.h>

#include "aec.h"
#include "anr.h"
#include "utility/simd_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Denoiser gain of one fused hop, estimated on v->base_spec. With use_r2, r2 * AEC_PSD_SCALE_INV_F folds into the
 * noise floor through `scratch` (n_freqs floats); use_r2 = 0, NULL r2/scratch or usable_linear = 0 gives plain NR.
 * Returns n_freqs gains owned by the denoiser until its next call; NULL for a NULL argument or a refused hop. */
static inline const float* array_sample_anr_gain(AnrDenoiser* nr, const AecResidualView* v,
                                                float* scratch, int use_r2) {
    const float* residual_psd_audio = NULL;
    if (!nr || !v || !v->base_spec) return NULL;
    if (use_r2 && v->r2 && scratch && v->usable_linear) {
        sk_scale_f32(scratch, v->r2, AEC_PSD_SCALE_INV_F, v->n_freqs);
        residual_psd_audio = scratch;
    }
    return anr_process_gain_with_residual_psd(nr, v->base_spec, residual_psd_audio);
}

/** 1 when the view's post_reset_serial differs from *handled (a lane had a hard post-chain
 * reset since the synthesizer's last hop) and stores it; then, before this hop's comfort-noise
 * draw and inverse transform, the synthesizer clears its overlap-add tail. 0 otherwise or for a
 * NULL argument. */
static inline int array_sample_post_reset_due(uint32_t* handled, const AecResidualView* v) {
    if (!handled || !v || v->post_reset_serial == *handled) return 0;
    *handled = v->post_reset_serial;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* ARRAY_SAMPLE_POST_H */
