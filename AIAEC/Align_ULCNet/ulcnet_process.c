/* Align-ULCNet C pre/post-processing (contract: ulcnet_process.h). Framing is
 * audiolib's AcStft/AcIstft as configured by ulcnet_framing_config(); this file
 * adds that configuration and the model-callback output check.
 * Requires -ffp-contract=off. */

#include "ulcnet_process.h"

#include <math.h>

#include "utility/simd_kernel_nn.h"

void ulcnet_framing_config(AcStftConfig *cfg, FftHandle *fft,
                           const float *window, int center) {
    if (!cfg) return;
    ac_stft_config_default(cfg, ULCNET_N_FFT);
    cfg->center = center;
    cfg->ola = AC_OLA_ENV_NORM;
    cfg->fft = fft;
    cfg->window = window;
}

/* ---- model callback output check (shared by the pipeline wrappers) ------- */

int ulcnet_frame_is_finite(const float re[ULCNET_BINS], const float im[ULCNET_BINS]) {
    return skn_all_finite_f32(re, ULCNET_BINS) &&
           skn_all_finite_f32(im, ULCNET_BINS);
}
