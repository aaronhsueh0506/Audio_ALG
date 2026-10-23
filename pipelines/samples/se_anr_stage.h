/* se_anr_stage.h -- example: the noise-reduction stage call of the AEC pipelines, which folds
 * the AEC residual-echo powers into the denoiser's noise floor. Header-only, no ABI: copy and
 * adapt. The denoiser itself is the library's: anr_config_for_mode_grid() on the AEC grid,
 * anr_get_mem_size() / anr_init(), anr_set_mode() for a strength change. Stateless: the
 * function takes the AnrDenoiser the application carved from its pool. */
#ifndef PIPELINES_SAMPLES_SE_ANR_STAGE_H
#define PIPELINES_SAMPLES_SE_ANR_STAGE_H

#include <stddef.h>

#include "anr.h"
#include "common/se_stage.h"
#include "utility/simd_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* int16 full scale squared: the power scale shared by AEC residual-echo powers and
 * the denoiser input. */
#define SE_PSD_SCALE (32768.0f * 32768.0f)
/* Exact reciprocal (a power of two): x / SE_PSD_SCALE == x * SE_PSD_SCALE_INV. */
#define SE_PSD_SCALE_INV (1.0f / SE_PSD_SCALE)

/* out[k] = r2[k] / SE_PSD_SCALE as an exact reciprocal multiply. */
static inline void se_psd_scale(float* out, const float* r2, int n) {
    sk_scale_f32(out, r2, SE_PSD_SCALE_INV, n);
}

/* One hop over `frame` (needs e). With use_r2 the residual-echo powers frame->r2
 * fold into the noise floor as r2 / SE_PSD_SCALE through `scratch` (n_freqs
 * floats); use_r2 = 0, NULL r2/scratch or frame->usable_linear = 0 selects plain
 * noise reduction. Returns n_freqs gains owned by the denoiser, valid until its
 * next call; NULL for a NULL argument or a refused hop. */
static inline const float* se_anr_stage_gain(AnrDenoiser* nr, const SeSpecFrame* frame,
                                             float* scratch, int use_r2) {
    const float* extra = NULL;
    if (!nr || !frame || !frame->e) return NULL;
    if (use_r2 && frame->r2 && scratch && frame->usable_linear) {
        se_psd_scale(scratch, frame->r2, frame->n_freqs);
        extra = scratch;
    }
    if (anr_process_gain(nr, frame->e, extra, NULL) != 0) return NULL;
    return anr_get_gain(nr, NULL);
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SAMPLES_SE_ANR_STAGE_H */
