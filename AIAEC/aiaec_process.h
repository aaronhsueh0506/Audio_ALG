/* Shared 16 kHz AIAEC boundary: 512-point FFT, 256 hop, sqrt-Hann
 * framing (audiolib AcStft/AcIstft, AC_OLA_ENV_NORM). Each model's pre/post
 * class owns its own STFT objects. Recurrent, conv and attention state tensors
 * are caller-owned and passed to the stateless accelerator on every call. */
#ifndef AIAEC_PROCESS_H
#define AIAEC_PROCESS_H

#include "audio_stft.h"
#include "Align_ULCNet/ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed framing of this boundary; independent of the ULCNET_* build parameters. */
#define AIAEC_SR       16000
#define AIAEC_N_FFT    512
#define AIAEC_HOP      (AIAEC_N_FFT / 2)
#define AIAEC_N_BINS   (AIAEC_N_FFT / 2 + 1)

/* Complex ratio mask: output = input * mask (CAGCRN). */
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

/* Align-ULCNet mask: signed |x|^0.3 compression, mask multiply, signed
 * |x|^(1/0.3) expansion. */
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
