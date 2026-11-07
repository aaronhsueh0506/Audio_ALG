#include "cagcrn_process.h"

void aiaec_apply_complex_mask(const float *input_re, const float *input_im,
                              const float *mask_re, const float *mask_im,
                              float *output_re, float *output_im, int n_bins) {
    int bin;
    if (!input_re || !input_im || !mask_re || !mask_im ||
        !output_re || !output_im) return;
    for (bin = 0; bin < n_bins; ++bin) {
        const float xr = input_re[bin];
        const float xi = input_im[bin];
        const float mr = mask_re[bin];
        const float mi = mask_im[bin];
        output_re[bin] = xr * mr - xi * mi;
        output_im[bin] = xi * mr + xr * mi;
    }
}
