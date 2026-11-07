#include "align_cruse_process.h"

void aiaec_apply_real_mask(const float *input_re, const float *input_im,
                           const float *mask,
                           float *output_re, float *output_im, int n_bins) {
    int bin;
    if (!input_re || !input_im || !mask || !output_re || !output_im) return;
    for (bin = 0; bin < n_bins; ++bin) {
        output_re[bin] = input_re[bin] * mask[bin];
        output_im[bin] = input_im[bin] * mask[bin];
    }
}
