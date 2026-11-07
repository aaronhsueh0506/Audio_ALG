/* Align-ULCNet's compressed-domain mask: the spectra hold n_bins values (the caller's grid); a
 * NULL pointer makes the call a no-op. Python mirror: Align_ULCNet/export_onnx.host_back_end. */
#ifndef ULCNET_MASK_PROCESS_H
#define ULCNET_MASK_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Signed |x|^e compression (e = ULCNET_MODEL_IO_COMPRESSION_EXP), mask multiply, signed
 * |x|^(1/e) expansion. */
void aiaec_apply_ulcnet_compressed_mask(
    const float *input_re, const float *input_im,
    const float *mask_re, const float *mask_im,
    float *output_re, float *output_im, int n_bins);

#ifdef __cplusplus
}
#endif
#endif
