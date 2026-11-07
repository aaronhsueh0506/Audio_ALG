/* Mask application shared by the AIAEC models: the spectra hold n_bins values (the caller's
 * grid); a NULL pointer makes a call a no-op. Python mirrors: aiaec_common.complex_mask,
 * Align_ULCNet/export_onnx.host_back_end, Align_CRUSE/model.py (AlignCRUSE.forward). */
#ifndef AIAEC_PROCESS_H
#define AIAEC_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Complex ratio mask: output = input * mask (CAGCRN). */
void aiaec_apply_complex_mask(const float *input_re, const float *input_im,
                              const float *mask_re, const float *mask_im,
                              float *output_re, float *output_im, int n_bins);
/* Real mask (Align-CRUSE): output = input * mask. */
void aiaec_apply_real_mask(const float *input_re, const float *input_im,
                           const float *mask,
                           float *output_re, float *output_im, int n_bins);

/* Align-ULCNet mask: signed |x|^e compression (e = ULCNET_MODEL_IO_COMPRESSION_EXP), mask
 * multiply, signed |x|^(1/e) expansion. */
void aiaec_apply_ulcnet_compressed_mask(
    const float *input_re, const float *input_im,
    const float *mask_re, const float *mask_im,
    float *output_re, float *output_im, int n_bins);

#ifdef __cplusplus
}
#endif
#endif
