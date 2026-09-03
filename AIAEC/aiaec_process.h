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

/* Held here, NOT aliased from ULCNET_*. Align-ULCNet's grid is a build
 * parameter (ulcnet_model_io.h) so it can be pointed at the 48 kHz product
 * grid; the other three models have no such build and their framing must not
 * move because someone rebuilt Align-ULCNet. These four are this boundary's
 * own contract. */
#define AIAEC_SR       16000
#define AIAEC_N_FFT    512
#define AIAEC_HOP      (AIAEC_N_FFT / 2)
#define AIAEC_N_BINS   (AIAEC_N_FFT / 2 + 1)

/* The typedefs below alias Align-ULCNet's structs, whose buffers are sized by
 * ITS grid, so this boundary is only meaningful while the two agree. A build
 * that moved Align-ULCNet off 16 kHz must not silently reinterpret these. */
#if ULCNET_SR != AIAEC_SR || ULCNET_N_FFT != AIAEC_N_FFT
#error "aiaec_process.h is a 16 kHz boundary; it cannot alias a non-16 kHz Align-ULCNet build"
#endif

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
/* center=False rolling analysis, one frame per push from the first push
 * (ulcnet_analysis_push_frame). */
int aiaec_analysis_push_frame(AiaecAnalysis *state,
                              const float input[AIAEC_HOP],
                              float real[AIAEC_N_BINS],
                              float imag[AIAEC_N_BINS]);
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
