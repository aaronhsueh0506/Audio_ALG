#include "deepvqe_process.h"

#include "simd_kernel_nn.h"

#include <string.h>

void deepvqe_ccm_init(DeepVqeCcmState *state) {
    if (state) memset(state, 0, sizeof(*state));
}

/* The two edge bins have taps that fall outside the spectrum; those taps are
 * skipped, which is not the same as multiplying a zero in (a non-finite tap
 * times zero is NaN). */
static void ccm_edge_bin(const DeepVqeCcmState *state, int bin,
                         const float taps[DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2],
                         float *out_re, float *out_im) {
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    int delay, freq_tap;
    for (delay = 0; delay < DEEPVQE_TIME_ORDER; ++delay) {
        for (freq_tap = 0; freq_tap < DEEPVQE_FREQ_TAPS; ++freq_tap) {
            const int source_bin = bin + freq_tap - 1;
            if (source_bin >= 0 && source_bin < AIAEC_N_BINS) {
                const float xr = state->spectrum_re[delay][source_bin];
                const float xi = state->spectrum_im[delay][source_bin];
                const float tr = taps[delay][freq_tap][0];
                const float ti = taps[delay][freq_tap][1];
                sum_re += xr * tr - xi * ti;
                sum_im += xi * tr + xr * ti;
            }
        }
    }
    *out_re = sum_re;
    *out_im = sum_im;
}

void deepvqe_ccm_process(
    DeepVqeCcmState *state,
    const float input_re[AIAEC_N_BINS],
    const float input_im[AIAEC_N_BINS],
    const float taps[AIAEC_N_BINS][DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2],
    float output_re[AIAEC_N_BINS],
    float output_im[AIAEC_N_BINS]) {
    enum { N_TAPS = DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS };
    const float *src_re[N_TAPS];
    const float *src_im[N_TAPS];
    int delay, freq_tap;
    if (!state || !input_re || !input_im || !taps ||
        !output_re || !output_im) return;
    memmove(state->spectrum_re[1], state->spectrum_re[0],
            (DEEPVQE_TIME_ORDER - 1) * AIAEC_N_BINS * sizeof(float));
    memmove(state->spectrum_im[1], state->spectrum_im[0],
            (DEEPVQE_TIME_ORDER - 1) * AIAEC_N_BINS * sizeof(float));
    memcpy(state->spectrum_re[0], input_re, AIAEC_N_BINS * sizeof(float));
    memcpy(state->spectrum_im[0], input_im, AIAEC_N_BINS * sizeof(float));
    /* Bins 1 .. N-2 see all nine taps in range: one source row per tap, the
     * frequency tap moving the row by one bin. */
    for (delay = 0; delay < DEEPVQE_TIME_ORDER; ++delay) {
        for (freq_tap = 0; freq_tap < DEEPVQE_FREQ_TAPS; ++freq_tap) {
            src_re[delay * DEEPVQE_FREQ_TAPS + freq_tap] =
                state->spectrum_re[delay] + freq_tap;
            src_im[delay * DEEPVQE_FREQ_TAPS + freq_tap] =
                state->spectrum_im[delay] + freq_tap;
        }
    }
    skn_ctaps_mac_f32(output_re + 1, output_im + 1, AIAEC_N_BINS - 2, N_TAPS,
                      src_re, src_im, &taps[1][0][0][0]);
    ccm_edge_bin(state, 0, taps[0], &output_re[0], &output_im[0]);
    ccm_edge_bin(state, AIAEC_N_BINS - 1, taps[AIAEC_N_BINS - 1],
                 &output_re[AIAEC_N_BINS - 1], &output_im[AIAEC_N_BINS - 1]);
}
