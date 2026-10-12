/* pipelines/common/se_anr_stage.h -- the noise-reduction stage of the AEC
 * pipelines.
 *
 * Every AEC + NR pipeline builds its denoiser from the same configuration: the
 * canonical strength preset with two overrides on top. The overrides are this
 * stack's product decision, not the denoiser's, so they live here and not in
 * NR's API; anr_set_mode() cannot reproduce them (it composes the bare preset,
 * whose L differs from the instance's, and anr_reconfigure() refuses a target
 * with another L). The strength setter therefore recomposes the full
 * configuration and hands it to anr_reconfigure().
 *
 * The stage keeps no state of its own: functions take the AnrDenoiser the
 * application carved from its pool, so a pipeline instance does not grow.
 */
#ifndef PIPELINES_SE_ANR_STAGE_H
#define PIPELINES_SE_ANR_STAGE_H

#include <stddef.h>

#include "anr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* int16 full scale squared: the power scale AEC's residual-echo estimate shares
 * with the denoiser's input. */
#define SE_PSD_SCALE (32768.0f * 32768.0f)

/* The denoiser configuration of these pipelines on a no-padding 50 % overlap
 * grid: canonical preset for `mode`, L = 150 frames of 10 ms retimed to the
 * hop, alpha_decay = alpha_g. broadband_threshold keeps its canonical 1.0
 * (scene-reset gate off). */
AnrConfig se_anr_compose_config(int sample_rate, int fft_size, int hop_size, AnrMode mode);

/* Bytes of the denoiser block for `cfg` (0 for an invalid config). */
size_t se_anr_stage_mem_size(const AnrConfig* cfg);

/* Initialises the denoiser in a 16-byte aligned block of at least
 * se_anr_stage_mem_size(cfg) bytes. NULL on failure. */
AnrDenoiser* se_anr_stage_init(void* mem, size_t bytes, const AnrConfig* cfg);

/* One hop. `r2` (n_freqs residual-echo powers on the AEC power scale) folds
 * into the denoiser's noise floor as r2 / SE_PSD_SCALE through `scratch`
 * (n_freqs floats); NULL r2 or NULL scratch selects plain noise reduction.
 * Returns 0, or -1 for a NULL denoiser or spectrum. */
int se_anr_stage_process_gain(AnrDenoiser* nr, const Complex* error_spec,
                              const float* r2, float* scratch, int n_freqs);

/* The gain of the last processed hop; borrowed until the next call. */
const float* se_anr_stage_gain(const AnrDenoiser* nr);

/* Re-targets the strength: recomposes the configuration on the instance's own
 * grid and reconfigures. Returns 0, or -1 for a NULL denoiser, an invalid
 * mode or a rejected target; nothing changes on -1. */
int se_anr_stage_set_mode(AnrDenoiser* nr, int sample_rate, int fft_size,
                          int hop_size, AnrMode mode);

void se_anr_stage_reset(AnrDenoiser* nr);
void se_anr_stage_destroy(AnrDenoiser* nr);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SE_ANR_STAGE_H */
