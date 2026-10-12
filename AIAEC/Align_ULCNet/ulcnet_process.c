/* Align-ULCNet C pre/post-processing. Contract and parity notes: see the
 * header. The STFT / ISTFT framing is audio_common's AcStft / AcIstft in the
 * configuration ulcnet_framing_config() describes; this file adds that
 * configuration and the model-callback step the pipeline wrappers share.
 *
 * Constraint: -ffp-contract=off (all owning Makefiles append it). */

#include "ulcnet_process.h"

#include <math.h>

#include "simd_kernel_nn.h"

void ulcnet_framing_config(AcStftConfig *cfg, FftHandle *fft,
                           const float *window, int center) {
    if (!cfg) return;
    ac_stft_config_default(cfg, ULCNET_N_FFT);
    cfg->center = center;
    cfg->ola = AC_OLA_ENV_NORM;
    cfg->fft = fft;
    cfg->window = window;
}

/* ---- model callback step (shared by the pipeline wrappers) --------------- */

int ulcnet_frame_is_finite(const float re[ULCNET_BINS], const float im[ULCNET_BINS]) {
    return skn_all_finite_f32(re, ULCNET_BINS) &&
           skn_all_finite_f32(im, ULCNET_BINS);
}

int ulcnet_model_run_frame(const UlcnetModel *model,
                           const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                           const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                           float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    skn_fill_f32(out_re, ULCNET_BINS, NAN);
    skn_fill_f32(out_im, ULCNET_BINS, NAN);
    if (model->infer(model->user, err_re, err_im, far_re, far_im, out_re, out_im) != 0)
        return 0;
    return ulcnet_frame_is_finite(out_re, out_im);
}
