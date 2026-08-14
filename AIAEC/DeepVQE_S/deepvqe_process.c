#include "deepvqe_process.h"

#include <string.h>

void deepvqe_ccm_init(DeepVqeCcmState *state) {
    if (state) memset(state, 0, sizeof(*state));
}

void deepvqe_ccm_process(
    DeepVqeCcmState *state,
    const float input_re[AIAEC_N_BINS],
    const float input_im[AIAEC_N_BINS],
    const float taps[AIAEC_N_BINS][DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2],
    float output_re[AIAEC_N_BINS],
    float output_im[AIAEC_N_BINS]) {
    int bin, delay, freq_tap;
    if (!state || !input_re || !input_im || !taps ||
        !output_re || !output_im) return;
    memmove(state->spectrum_re[1], state->spectrum_re[0],
            (DEEPVQE_TIME_ORDER - 1) * AIAEC_N_BINS * sizeof(float));
    memmove(state->spectrum_im[1], state->spectrum_im[0],
            (DEEPVQE_TIME_ORDER - 1) * AIAEC_N_BINS * sizeof(float));
    memcpy(state->spectrum_re[0], input_re, AIAEC_N_BINS * sizeof(float));
    memcpy(state->spectrum_im[0], input_im, AIAEC_N_BINS * sizeof(float));
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        for (delay = 0; delay < DEEPVQE_TIME_ORDER; ++delay) {
            for (freq_tap = 0; freq_tap < DEEPVQE_FREQ_TAPS; ++freq_tap) {
                const int source_bin = bin + freq_tap - 1;
                if (source_bin >= 0 && source_bin < AIAEC_N_BINS) {
                    const float xr = state->spectrum_re[delay][source_bin];
                    const float xi = state->spectrum_im[delay][source_bin];
                    const float tr = taps[bin][delay][freq_tap][0];
                    const float ti = taps[bin][delay][freq_tap][1];
                    sum_re += xr * tr - xi * ti;
                    sum_im += xi * tr + xr * ti;
                }
            }
        }
        output_re[bin] = sum_re;
        output_im[bin] = sum_im;
    }
}
