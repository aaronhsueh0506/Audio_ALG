#include "gtcrn_process.h"

#include <math.h>
#include <string.h>

#include "simd_kernel_nn.h"

#if defined(__aarch64__) && defined(__ARM_NEON) && \
    !defined(SIMD_KERNELS_FORCE_SCALAR)
#define DF_COMMON_HAVE_NEON 1
#else
#define DF_COMMON_HAVE_NEON 0
#endif

void gtcrn_framing_config(AcStftConfig* cfg, FftHandle* fft,
                          const float* window)
{
    if (!cfg) return;
    ac_stft_config_default(cfg, GTCRN_N_FFT);
    cfg->fft = fft;
    cfg->window = window;
}

void gtcrn_process_init(GTCRNProcessState* state, AcStft* stft,
                        AcIstft* istft)
{
    if (!state) return;
    state->stft = stft;
    state->istft = istft;
    ac_stft_reset(stft);
    ac_istft_reset(istft);
}

/* Band one already-computed full-band channel: low bins pass through, the
 * high 192 bins accumulate into 64 bands via the committed forward table
 * (bin-major, so the inner loop runs contiguously over bands and
 * auto-vectorizes). */
static void erb_band_channel(const float* full, const float* erb_fwd,
                             float* banded)
{
    for (int k = 0; k < GTCRN_MODEL_ERB_KEPT; ++k) banded[k] = full[k];
    for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b)
        banded[GTCRN_MODEL_ERB_KEPT + b] = 0.0f;
    for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
        float value = full[GTCRN_MODEL_ERB_KEPT + k];
        const float* row = erb_fwd + (size_t)k * GTCRN_MODEL_ERB_HIGH_BANDS;
        float* out = banded + GTCRN_MODEL_ERB_KEPT;
        for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b)
            out[b] += row[b] * value;
    }
}

void gtcrn_model_input(const float spectrum[GTCRN_N_BINS][2],
                       const float* erb_fwd,
                       float mag[GTCRN_MODEL_ERB_BANDS],
                       float real_part[GTCRN_MODEL_ERB_BANDS],
                       float imag_part[GTCRN_MODEL_ERB_BANDS])
{
    float full_mag[GTCRN_N_BINS];
    float full_re[GTCRN_N_BINS];
    float full_im[GTCRN_N_BINS];
    if (!spectrum || !erb_fwd || !mag || !real_part || !imag_part) return;
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        float re = spectrum[k][0];
        float im = spectrum[k][1];
        full_mag[k] = sqrtf(re * re + im * im + 1e-12f);
        full_re[k] = re;
        full_im[k] = im;
    }
    erb_band_channel(full_mag, erb_fwd, mag);
    erb_band_channel(full_re, erb_fwd, real_part);
    erb_band_channel(full_im, erb_fwd, imag_part);
}

void gtcrn_model_output(const float mask_erb[GTCRN_MODEL_ERB_BANDS][2],
                        const float* erb_inv,
                        const float spectrum[GTCRN_N_BINS][2],
                        float enhanced[GTCRN_N_BINS][2])
{
    float mask_re[GTCRN_N_BINS];
    float mask_im[GTCRN_N_BINS];
    if (!mask_erb || !erb_inv || !spectrum || !enhanced) return;
    for (int k = 0; k < GTCRN_MODEL_ERB_KEPT; ++k) {
        mask_re[k] = mask_erb[k][0];
        mask_im[k] = mask_erb[k][1];
    }
    for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
        mask_re[GTCRN_MODEL_ERB_KEPT + k] = 0.0f;
        mask_im[GTCRN_MODEL_ERB_KEPT + k] = 0.0f;
    }
    /* Inverse table is band-major: the inner loop runs contiguously over
     * the 192 high bins and auto-vectorizes. */
    for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b) {
        float band_re = mask_erb[GTCRN_MODEL_ERB_KEPT + b][0];
        float band_im = mask_erb[GTCRN_MODEL_ERB_KEPT + b][1];
        const float* row = erb_inv + (size_t)b * GTCRN_MODEL_ERB_HIGH_BINS;
        for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
            mask_re[GTCRN_MODEL_ERB_KEPT + k] += row[k] * band_re;
            mask_im[GTCRN_MODEL_ERB_KEPT + k] += row[k] * band_im;
        }
    }
    /* Complex ratio mask, mirroring model.py's Mask exactly. */
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        float sr = spectrum[k][0], si = spectrum[k][1];
        float mr = mask_re[k], mi = mask_im[k];
        enhanced[k][0] = sr * mr - si * mi;
        enhanced[k][1] = si * mr + sr * mi;
    }
}

void gtcrn_model_state_init(GTCRNModelState* state)
{
    if (state == NULL) return;
    memset(state, 0, sizeof(*state));
    state->state_generation = 1u;
}

void gtcrn_model_state_reset(GTCRNModelState* state)
{
    unsigned int generation;
    if (state == NULL) return;
    generation = state->state_generation + 1u;
    memset(state, 0, sizeof(*state));
    state->state_generation = generation;
}

unsigned int gtcrn_model_state_generation(const GTCRNModelState* state)
{
    return state == NULL ? 0u : state->state_generation;
}

static int all_finite(const float* values, size_t count)
{
    return skn_all_finite_f32(values, count);
}

#define GTCRN_MODEL_STATE_TENSORS (GTCRN_MODEL_CONV_STATES + 2)

/* The eight state tensors in graph order (six conv windows, h_tra,
 * h_dpgrnn) with their element counts; together they tile the struct up to
 * state_generation. */
static void state_tensors(GTCRNModelState* state,
                          float* base[GTCRN_MODEL_STATE_TENSORS],
                          size_t count[GTCRN_MODEL_STATE_TENSORS])
{
    int n = 0;
    base[n] = &state->conv_enc0[0][0][0];
    count[n++] = sizeof(state->conv_enc0) / sizeof(float);
    base[n] = &state->conv_enc1[0][0][0];
    count[n++] = sizeof(state->conv_enc1) / sizeof(float);
    base[n] = &state->conv_enc2[0][0][0];
    count[n++] = sizeof(state->conv_enc2) / sizeof(float);
    base[n] = &state->conv_dec0[0][0][0];
    count[n++] = sizeof(state->conv_dec0) / sizeof(float);
    base[n] = &state->conv_dec1[0][0][0];
    count[n++] = sizeof(state->conv_dec1) / sizeof(float);
    base[n] = &state->conv_dec2[0][0][0];
    count[n++] = sizeof(state->conv_dec2) / sizeof(float);
    base[n] = &state->h_tra[0][0][0];
    count[n++] = sizeof(state->h_tra) / sizeof(float);
    base[n] = &state->h_dpgrnn[0][0][0];
    count[n++] = sizeof(state->h_dpgrnn) / sizeof(float);
}

int gtcrn_model_state_validate(GTCRNModelState* state)
{
    float* base[GTCRN_MODEL_STATE_TENSORS];
    size_t count[GTCRN_MODEL_STATE_TENSORS];
    if (state == NULL) return -1;
    state_tensors(state, base, count);
    for (int i = 0; i < GTCRN_MODEL_STATE_TENSORS; ++i) {
        if (!all_finite(base[i], count[i])) {
            gtcrn_model_state_reset(state);
            return -1;
        }
    }
    return 0;
}

void gtcrn_analysis(GTCRNProcessState* state, const float* input,
                    float output[GTCRN_N_BINS][2])
{
    if (!state || !state->stft || !input || !output) return;
    ac_stft_push(state->stft, input, (Complex*)(void*)output);
}

void gtcrn_synthesis(GTCRNProcessState* state,
                     const float input[GTCRN_N_BINS][2],
                     float* output)
{
    if (!state || !state->istft || !input || !output) return;
    ac_istft_push(state->istft, (const Complex*)(const void*)input, output);
}

const char* gtcrn_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
