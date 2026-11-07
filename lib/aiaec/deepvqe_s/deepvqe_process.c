#include "deepvqe_process.h"

#include "utility/simd_kernel_nn.h"

#include <string.h>

int deepvqe_ccm_init(DeepVqeCcmState *state, int n_bins,
                     float *spectrum_re, float *spectrum_im) {
    if (!state || n_bins < 3 || !spectrum_re || !spectrum_im) return -1;
    state->n_bins = n_bins;
    state->spectrum_re = spectrum_re;
    state->spectrum_im = spectrum_im;
    deepvqe_ccm_reset(state);
    return 0;
}

void deepvqe_ccm_reset(DeepVqeCcmState *state) {
    size_t bytes;
    if (!state || !state->spectrum_re || !state->spectrum_im) return;
    bytes = (size_t)DEEPVQE_TIME_ORDER * (size_t)state->n_bins * sizeof(float);
    memset(state->spectrum_re, 0, bytes);
    memset(state->spectrum_im, 0, bytes);
}

/* Edge bins: taps outside the spectrum are skipped, not multiplied by zero (a non-finite tap
 * times zero is NaN). */
static void ccm_edge_bin(const DeepVqeCcmState *state, int bin, const float *taps,
                         float *out_re, float *out_im) {
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    int delay, freq_tap;
    for (delay = 0; delay < DEEPVQE_TIME_ORDER; ++delay) {
        for (freq_tap = 0; freq_tap < DEEPVQE_FREQ_TAPS; ++freq_tap) {
            const int source_bin = bin + freq_tap - 1;
            if (source_bin >= 0 && source_bin < state->n_bins) {
                const size_t row = (size_t)delay * (size_t)state->n_bins;
                const float xr = state->spectrum_re[row + (size_t)source_bin];
                const float xi = state->spectrum_im[row + (size_t)source_bin];
                const float tr = taps[(delay * DEEPVQE_FREQ_TAPS + freq_tap) * 2];
                const float ti = taps[(delay * DEEPVQE_FREQ_TAPS + freq_tap) * 2 + 1];
                sum_re += xr * tr - xi * ti;
                sum_im += xi * tr + xr * ti;
            }
        }
    }
    *out_re = sum_re;
    *out_im = sum_im;
}

void deepvqe_ccm_process(DeepVqeCcmState *state,
                         const float *input_re, const float *input_im,
                         const float *taps,
                         float *output_re, float *output_im) {
    enum { N_TAPS = DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS };
    const float *src_re[N_TAPS];
    const float *src_im[N_TAPS];
    size_t n;
    int delay, freq_tap;
    if (!state || !state->spectrum_re || !state->spectrum_im || !input_re ||
        !input_im || !taps || !output_re || !output_im) return;
    n = (size_t)state->n_bins;
    memmove(state->spectrum_re + n, state->spectrum_re,
            (DEEPVQE_TIME_ORDER - 1) * n * sizeof(float));
    memmove(state->spectrum_im + n, state->spectrum_im,
            (DEEPVQE_TIME_ORDER - 1) * n * sizeof(float));
    memcpy(state->spectrum_re, input_re, n * sizeof(float));
    memcpy(state->spectrum_im, input_im, n * sizeof(float));
    /* Bins 1..n-2 see every tap in range: one source row per tap, shifted one bin per freq tap. */
    for (delay = 0; delay < DEEPVQE_TIME_ORDER; ++delay) {
        for (freq_tap = 0; freq_tap < DEEPVQE_FREQ_TAPS; ++freq_tap) {
            src_re[delay * DEEPVQE_FREQ_TAPS + freq_tap] =
                state->spectrum_re + (size_t)delay * n + (size_t)freq_tap;
            src_im[delay * DEEPVQE_FREQ_TAPS + freq_tap] =
                state->spectrum_im + (size_t)delay * n + (size_t)freq_tap;
        }
    }
    skn_ctaps_mac_f32(output_re + 1, output_im + 1, n - 2, N_TAPS,
                      src_re, src_im, taps + (size_t)N_TAPS * 2u);
    ccm_edge_bin(state, 0, taps, &output_re[0], &output_im[0]);
    ccm_edge_bin(state, state->n_bins - 1, taps + (n - 1) * (size_t)N_TAPS * 2u,
                 &output_re[n - 1], &output_im[n - 1]);
}
