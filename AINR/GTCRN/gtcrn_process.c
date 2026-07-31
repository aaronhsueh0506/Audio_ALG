#include "gtcrn_process.h"

#include <string.h>

#include "../dfn_process_common.h"

void gtcrn_process_init(GTCRNProcessState* state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    df_common_make_root_hann(state->window, GTCRN_WIN_LEN);
}

void gtcrn_analysis(GTCRNProcessState* state, const float* input,
                    float output[GTCRN_N_BINS][2])
{
    if (!state || !input || !output) return;
    df_common_analysis(state->analysis_buf, state->window,
                       state->scratch_re, state->scratch_im, input,
                       GTCRN_N_FFT, GTCRN_HOP_LEN, 1.0f,
                       state->scratch_re, state->scratch_im);
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        output[k][0] = state->scratch_re[k];
        output[k][1] = state->scratch_im[k];
    }
}

void gtcrn_synthesis(GTCRNProcessState* state,
                     const float input[GTCRN_N_BINS][2],
                     float* output)
{
    if (!state || !input || !output) return;
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        state->scratch_re[k] = input[k][0];
        state->scratch_im[k] = input[k][1];
    }
    df_common_synthesis(state->synthesis_buf, state->window,
                        state->scratch_re, state->scratch_im,
                        state->scratch_re, state->scratch_im,
                        GTCRN_N_FFT, GTCRN_HOP_LEN,
                        1.0f, output);
}

const char* gtcrn_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
