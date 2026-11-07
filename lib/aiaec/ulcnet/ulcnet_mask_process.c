#include "ulcnet_mask_process.h"

#include "ulcnet_model_io.h"   /* ULCNET_MODEL_IO_COMPRESSION_EXP */

#include <math.h>

static float aiaec_signed_pow(float value, float exponent) {
    const float magnitude = powf(fabsf(value), exponent);
    return value < 0.0f ? -magnitude : magnitude;
}

void aiaec_apply_ulcnet_compressed_mask(
    const float *input_re, const float *input_im,
    const float *mask_re, const float *mask_im,
    float *output_re, float *output_im, int n_bins) {
    const float inverse_exponent = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
    int bin;
    if (!input_re || !input_im || !mask_re || !mask_im ||
        !output_re || !output_im) return;
    for (bin = 0; bin < n_bins; ++bin) {
        const float xr = aiaec_signed_pow(
            input_re[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float xi = aiaec_signed_pow(
            input_im[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float yr = xr * mask_re[bin] - xi * mask_im[bin];
        const float yi = xi * mask_re[bin] + xr * mask_im[bin];
        output_re[bin] = aiaec_signed_pow(yr, inverse_exponent);
        output_im[bin] = aiaec_signed_pow(yi, inverse_exponent);
    }
}
