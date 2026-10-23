/* se_anr_stage.h -- noise-reduction stage shared by the AEC pipelines. Stateless:
 * functions take the AnrDenoiser the application carved from its pool. */
#ifndef PIPELINES_SE_ANR_STAGE_H
#define PIPELINES_SE_ANR_STAGE_H

#include <stddef.h>

#include "anr.h"
#include "se_stage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* int16 full scale squared: the power scale shared by AEC residual-echo powers and
 * the denoiser input. */
#define SE_PSD_SCALE (32768.0f * 32768.0f)
/* Exact reciprocal (a power of two): x / SE_PSD_SCALE == x * SE_PSD_SCALE_INV. */
#define SE_PSD_SCALE_INV (1.0f / SE_PSD_SCALE)

/* Denoiser configuration on a no-padding 50 % overlap grid: canonical preset for
 * `mode`, L = 150 frames of 10 ms retimed to the hop, alpha_decay = alpha_g,
 * broadband_threshold 1.0 (scene-reset gate off). */
AnrConfig se_anr_compose_config(int sample_rate, int fft_size, int hop_size, AnrMode mode);

/* Bytes of the denoiser block for `cfg` (0 for an invalid config). */
size_t se_anr_stage_mem_size(const AnrConfig* cfg);

/* Initialises the denoiser in a 16-byte aligned block of at least
 * se_anr_stage_mem_size(cfg) bytes. NULL on failure. */
AnrDenoiser* se_anr_stage_init(void* mem, size_t bytes, const AnrConfig* cfg);

/* One hop over `frame` (needs e). With use_r2 the residual-echo powers frame->r2
 * fold into the noise floor as r2 / SE_PSD_SCALE through `scratch` (n_freqs
 * floats); use_r2 = 0, NULL r2/scratch or frame->usable_linear = 0 selects plain
 * noise reduction. Returns n_freqs gains owned by the denoiser, valid until its
 * next call; NULL for a NULL argument or a refused hop. */
const float* se_anr_stage_gain(AnrDenoiser* nr, const SeSpecFrame* frame,
                               float* scratch, int use_r2);

/* Re-targets the strength on the instance's own grid. Returns 0, or -1 for a NULL
 * denoiser, an invalid mode or a rejected target (nothing changes). */
int se_anr_stage_set_mode(AnrDenoiser* nr, int sample_rate, int fft_size,
                          int hop_size, AnrMode mode);

void se_anr_stage_reset(AnrDenoiser* nr);
void se_anr_stage_destroy(AnrDenoiser* nr);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SE_ANR_STAGE_H */
