/* Shared 16 kHz AIAEC STFT/WOLA boundary.
 *
 * Align-CRUSE, Align-ULCNet, DeepVQE-S and CAGCRN all use the
 * same 16 kHz / 512 / 256 centered sqrt-Hann framing.  This API deliberately
 * aliases the already parity-tested Align-ULCNet implementation instead of
 * maintaining four copies. Model-specific masks/CCM filters cross the model
 * output boundary, while every recurrent/conv/attention state tensor is
 * caller-owned and returned to the stateless accelerator on the next call.
 * Exact state names and shapes are emitted by Align_ULCNet/export_onnx.py.
 */
#ifndef AIAEC_PROCESS_H
#define AIAEC_PROCESS_H

#include "Align_ULCNet/ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AIAEC_SR       ULCNET_SR
#define AIAEC_N_FFT    ULCNET_N_FFT
#define AIAEC_HOP      ULCNET_HOP
#define AIAEC_N_BINS   ULCNET_BINS

typedef UlcnetAnalysis AiaecAnalysis;
typedef UlcnetSynthesis AiaecSynthesis;

void aiaec_make_window(float window[AIAEC_N_FFT]);
int aiaec_analysis_init(AiaecAnalysis *state, FftHandle *fft,
                        const float *window);
int aiaec_analysis_push(AiaecAnalysis *state,
                        const float input[AIAEC_HOP],
                        float real[2][AIAEC_N_BINS],
                        float imag[2][AIAEC_N_BINS]);
int aiaec_analysis_flush(AiaecAnalysis *state,
                         float real[2][AIAEC_N_BINS],
                         float imag[2][AIAEC_N_BINS]);
int aiaec_synthesis_init(AiaecSynthesis *state, FftHandle *fft,
                         const float *window);
int aiaec_synthesis_push(AiaecSynthesis *state,
                         const float real[AIAEC_N_BINS],
                         const float imag[AIAEC_N_BINS],
                         float output[AIAEC_HOP]);
int aiaec_synthesis_flush(AiaecSynthesis *state,
                          float output[AIAEC_N_FFT]);

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
