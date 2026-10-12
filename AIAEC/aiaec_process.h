/* Shared 16 kHz AIAEC boundary.
 *
 * Align-CRUSE, Align-ULCNet, DeepVQE-S and CAGCRN all use the same
 * 16 kHz / 512 / 256 centered sqrt-Hann framing, which is audio_common's
 * AcStft / AcIstft (audio_stft.h) in the AC_OLA_ENV_NORM configuration; each
 * model's pre/post class builds its own objects. This header holds the
 * constants of that boundary and the complex-mask helpers. Model-specific
 * masks/CCM filters cross the model output boundary, while every
 * recurrent/conv/attention state tensor is caller-owned and returned to the
 * stateless accelerator on the next call. Exact state names and shapes are
 * emitted by Align_ULCNet/export_onnx.py.
 */
#ifndef AIAEC_PROCESS_H
#define AIAEC_PROCESS_H

#include "audio_stft.h"
#include "Align_ULCNet/ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This boundary's own contract, independent of ULCNET_*: Align-ULCNet's grid
 * is a build parameter (ulcnet_model_io.h) that can be pointed at the 48 kHz
 * product grid, while the other models' framing must not move when it is. */
#define AIAEC_SR       16000
#define AIAEC_N_FFT    512
#define AIAEC_HOP      (AIAEC_N_FFT / 2)
#define AIAEC_N_BINS   (AIAEC_N_FFT / 2 + 1)

/* Complex ratio-mask helper used by CAGCRN. */
void aiaec_apply_complex_mask(const float input_re[AIAEC_N_BINS],
                              const float input_im[AIAEC_N_BINS],
                              const float mask_re[AIAEC_N_BINS],
                              const float mask_im[AIAEC_N_BINS],
                              float output_re[AIAEC_N_BINS],
                              float output_im[AIAEC_N_BINS]);
void aiaec_apply_real_mask(const float input_re[AIAEC_N_BINS],
                           const float input_im[AIAEC_N_BINS],
                           const float mask[AIAEC_N_BINS],
                           float output_re[AIAEC_N_BINS],
                           float output_im[AIAEC_N_BINS]);

/* Align-ULCNet's mask is not an ordinary CRM: it multiplies the error after
 * component-wise signed |x|^0.3 compression, then applies signed
 * |x|^(1/0.3) expansion. CAGCRN must use
 * aiaec_apply_complex_mask() instead. */
void aiaec_apply_ulcnet_compressed_mask(
    const float input_re[AIAEC_N_BINS],
    const float input_im[AIAEC_N_BINS],
    const float mask_re[AIAEC_N_BINS],
    const float mask_im[AIAEC_N_BINS],
    float output_re[AIAEC_N_BINS],
    float output_im[AIAEC_N_BINS]);

#ifdef __cplusplus
}
#endif
#endif
