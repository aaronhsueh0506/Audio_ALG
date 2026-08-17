#include "gtcrn_process.h"

#include <math.h>
#include <string.h>

#include "../dfn_process_common.h"

void gtcrn_process_init(GTCRNProcessState* state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    df_common_make_root_hann(state->window, GTCRN_WIN_LEN);
}

void gtcrn_model_state_init(GTCRNModelState* state)
{
    if (state != NULL) memset(state, 0, sizeof(*state));
}

static int all_finite(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

int gtcrn_model_state_commit(GTCRNModelState* state,
                             const float* conv_cache_out,
                             const float* tra_cache_out,
                             const float* inter_cache_out)
{
    if (state == NULL || conv_cache_out == NULL || tra_cache_out == NULL ||
        inter_cache_out == NULL) return -1;
    /* Validate all three caches BEFORE writing any of them. A partial commit
     * would leave the recurrent state a mix of two invocations, which the next
     * call cannot distinguish from a healthy state; refusing the whole batch
     * keeps the last good state replayable. */
    if (!all_finite(conv_cache_out,
                    sizeof(state->conv_cache) / sizeof(float)) ||
        !all_finite(tra_cache_out,
                    sizeof(state->tra_cache) / sizeof(float)) ||
        !all_finite(inter_cache_out,
                    sizeof(state->inter_cache) / sizeof(float))) return -1;
    memcpy(state->conv_cache, conv_cache_out, sizeof(state->conv_cache));
    memcpy(state->tra_cache, tra_cache_out, sizeof(state->tra_cache));
    memcpy(state->inter_cache, inter_cache_out, sizeof(state->inter_cache));
    return 0;
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
