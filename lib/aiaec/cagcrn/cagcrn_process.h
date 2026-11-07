/* CAGCRN's post-processing: the spectra hold n_bins values (the caller's grid); a NULL pointer
 * makes the call a no-op. Python mirror: aiaec_common.complex_mask. */
#ifndef CAGCRN_PROCESS_H
#define CAGCRN_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Complex ratio mask: output = input * mask. */
void aiaec_apply_complex_mask(const float *input_re, const float *input_im,
                              const float *mask_re, const float *mask_im,
                              float *output_re, float *output_im, int n_bins);

#ifdef __cplusplus
}
#endif
#endif
