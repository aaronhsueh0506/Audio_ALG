/* Align-CRUSE's post-processing: the spectra hold n_bins values (the caller's grid); a NULL
 * pointer makes the call a no-op. Python mirror: aiaec_common.complex_mask (the real-mask path),
 * Align_CRUSE/model.py (AlignCRUSE.forward). */
#ifndef ALIGN_CRUSE_PROCESS_H
#define ALIGN_CRUSE_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Real mask: output = input * mask (same real gain on both planes). */
void aiaec_apply_real_mask(const float *input_re, const float *input_im,
                           const float *mask,
                           float *output_re, float *output_im, int n_bins);

#ifdef __cplusplus
}
#endif
#endif
