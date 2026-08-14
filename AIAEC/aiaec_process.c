#include "aiaec_process.h"

#include <math.h>

static float aiaec_signed_pow(float value, float exponent) {
    const float magnitude = powf(fabsf(value), exponent);
    return value < 0.0f ? -magnitude : magnitude;
}

void aiaec_make_window(float window[AIAEC_N_FFT]) {
    ulcnet_make_window(window);
}

int aiaec_analysis_init(AiaecAnalysis *state, FftHandle *fft,
                        const float *window) {
    return ulcnet_analysis_init(state, fft, window);
}

int aiaec_analysis_push(AiaecAnalysis *state,
                        const float input[AIAEC_HOP],
                        float real[2][AIAEC_N_BINS],
                        float imag[2][AIAEC_N_BINS]) {
    return ulcnet_analysis_push(state, input, real, imag);
}

int aiaec_analysis_flush(AiaecAnalysis *state,
                         float real[2][AIAEC_N_BINS],
                         float imag[2][AIAEC_N_BINS]) {
    return ulcnet_analysis_flush(state, real, imag);
}

int aiaec_synthesis_init(AiaecSynthesis *state, FftHandle *fft,
                         const float *window) {
    return ulcnet_synthesis_init(state, fft, window);
}

int aiaec_synthesis_push(AiaecSynthesis *state,
                         const float real[AIAEC_N_BINS],
                         const float imag[AIAEC_N_BINS],
                         float output[AIAEC_HOP]) {
    return ulcnet_synthesis_push(state, real, imag, output);
}

int aiaec_synthesis_flush(AiaecSynthesis *state,
                          float output[AIAEC_N_FFT]) {
    return ulcnet_synthesis_flush(state, output);
}

void aiaec_apply_complex_mask(const float input_re[AIAEC_N_BINS],
                              const float input_im[AIAEC_N_BINS],
                              const float mask_re[AIAEC_N_BINS],
                              const float mask_im[AIAEC_N_BINS],
                              float output_re[AIAEC_N_BINS],
                              float output_im[AIAEC_N_BINS]) {
    int bin;
    if (!input_re || !input_im || !mask_re || !mask_im ||
        !output_re || !output_im) return;
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        const float xr = input_re[bin];
        const float xi = input_im[bin];
        const float mr = mask_re[bin];
        const float mi = mask_im[bin];
        output_re[bin] = xr * mr - xi * mi;
        output_im[bin] = xi * mr + xr * mi;
    }
}

void aiaec_apply_real_mask(const float input_re[AIAEC_N_BINS],
                           const float input_im[AIAEC_N_BINS],
                           const float mask[AIAEC_N_BINS],
                           float output_re[AIAEC_N_BINS],
                           float output_im[AIAEC_N_BINS]) {
    int bin;
    if (!input_re || !input_im || !mask || !output_re || !output_im) return;
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        output_re[bin] = input_re[bin] * mask[bin];
        output_im[bin] = input_im[bin] * mask[bin];
    }
}

void aiaec_apply_ulcnet_compressed_mask(
    const float input_re[AIAEC_N_BINS],
    const float input_im[AIAEC_N_BINS],
    const float mask_re[AIAEC_N_BINS],
    const float mask_im[AIAEC_N_BINS],
    float output_re[AIAEC_N_BINS],
    float output_im[AIAEC_N_BINS]) {
    const float inverse_exponent = 1.0f / ULCNET_COMPRESSION_EXP;
    int bin;
    if (!input_re || !input_im || !mask_re || !mask_im ||
        !output_re || !output_im) return;
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        const float xr = aiaec_signed_pow(
            input_re[bin], ULCNET_COMPRESSION_EXP);
        const float xi = aiaec_signed_pow(
            input_im[bin], ULCNET_COMPRESSION_EXP);
        const float yr = xr * mask_re[bin] - xi * mask_im[bin];
        const float yi = xi * mask_re[bin] + xr * mask_im[bin];
        output_re[bin] = aiaec_signed_pow(yr, inverse_exponent);
        output_im[bin] = aiaec_signed_pow(yi, inverse_exponent);
    }
}
