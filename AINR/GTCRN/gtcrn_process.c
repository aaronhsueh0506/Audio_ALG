#include "gtcrn_process.h"

#include <math.h>
#include <string.h>

#include "utility/simd_kernel_nn.h"

#if SK_HAVE_NEON
#include <arm_neon.h>
#endif
#define DF_COMMON_HAVE_NEON SK_HAVE_NEON

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

/* Scalar twin of a NEON kernel; it is the whole kernel in a SIMD=0 build. */
#define GTCRN_TWIN static inline __attribute__((unused))

/* Low bins pass through; the high bins are banded with the shared scalar kernel
 * (ascending-bin, unfused accumulation). */
GTCRN_TWIN void erb_band_channel_scalar(const float* full,
                                        const float* erb_fwd, float* banded)
{
    for (int k = 0; k < GTCRN_MODEL_ERB_KEPT; ++k) banded[k] = full[k];
    skn_erb_band_accumulate_f32_scalar(full + GTCRN_MODEL_ERB_KEPT, erb_fwd,
                                       GTCRN_MODEL_ERB_HIGH_BINS,
                                       GTCRN_MODEL_ERB_HIGH_BANDS,
                                       banded + GTCRN_MODEL_ERB_KEPT);
}

#if DF_COMMON_HAVE_NEON
/* Quads of bins the mask-expansion kernel keeps in registers. */
#define MASK_BIN_QUADS 8
_Static_assert(GTCRN_MODEL_ERB_HIGH_BINS % (4 * MASK_BIN_QUADS) == 0,
               "the bin blocks tile the high bins");

/* The three channels fuse into one table pass; each channel/band keeps its own
 * accumulator in ascending-bin, unfused order. */
static inline void erb_band3_neon(const float* const full[3],
                                  const float* erb_fwd, float* const banded[3])
{
    const float* x3[3];
    float* b3[3];
    for (int c = 0; c < 3; ++c) {
        memcpy(banded[c], full[c], GTCRN_MODEL_ERB_KEPT * sizeof(float));
        x3[c] = full[c] + GTCRN_MODEL_ERB_KEPT;
        b3[c] = banded[c] + GTCRN_MODEL_ERB_KEPT;
    }
    skn_erb_band_accumulate_nch_f32(x3, 3, erb_fwd, GTCRN_MODEL_ERB_HIGH_BINS,
                                    GTCRN_MODEL_ERB_HIGH_BANDS, b3);
}
#endif

static inline void erb_band3(const float* const full[3], const float* erb_fwd,
                             float* const banded[3])
{
#if DF_COMMON_HAVE_NEON
    erb_band3_neon(full, erb_fwd, banded);
#else
    for (int c = 0; c < 3; ++c)
        erb_band_channel_scalar(full[c], erb_fwd, banded[c]);
#endif
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
    const float* const full[3] = { full_mag, full_re, full_im };
    float* const banded[3] = { mag, real_part, imag_part };
    if (!spectrum || !erb_fwd || !mag || !real_part || !imag_part) return;
    skn_deinterleave_cf32((const Complex*)(const void*)spectrum, full_re,
                          full_im, GTCRN_N_BINS);
    skn_mag_eps_f32(full_mag, full_re, full_im, GTCRN_N_BINS, 1e-12f);
    erb_band3(full, erb_fwd, banded);
}

/* mask[KEPT + k] = sum over the high bands b ascending of inv[b][k] * gain[b].
 * The inverse table is band-major: the inner loop runs contiguously over the
 * 192 high bins. */
GTCRN_TWIN void mask_expand_scalar(
    const float mask_erb[GTCRN_MODEL_ERB_BANDS][2], const float* erb_inv,
    float* mask_re, float* mask_im)
{
    for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
        mask_re[GTCRN_MODEL_ERB_KEPT + k] = 0.0f;
        mask_im[GTCRN_MODEL_ERB_KEPT + k] = 0.0f;
    }
    for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b) {
        float band_re = mask_erb[GTCRN_MODEL_ERB_KEPT + b][0];
        float band_im = mask_erb[GTCRN_MODEL_ERB_KEPT + b][1];
        const float* row = erb_inv + (size_t)b * GTCRN_MODEL_ERB_HIGH_BINS;
        for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
            mask_re[GTCRN_MODEL_ERB_KEPT + k] += row[k] * band_re;
            mask_im[GTCRN_MODEL_ERB_KEPT + k] += row[k] * band_im;
        }
    }
}

#if DF_COMMON_HAVE_NEON
/* The same sums with the running values of each bin block in registers, in
 * the scalar twin's band order. */
static inline void mask_expand_neon(
    const float mask_erb[GTCRN_MODEL_ERB_BANDS][2], const float* erb_inv,
    float* mask_re, float* mask_im)
{
    for (int k0 = 0; k0 < GTCRN_MODEL_ERB_HIGH_BINS; k0 += 4 * MASK_BIN_QUADS) {
        float32x4_t acc_re[MASK_BIN_QUADS], acc_im[MASK_BIN_QUADS];
        for (int q = 0; q < MASK_BIN_QUADS; ++q) {
            acc_re[q] = vdupq_n_f32(0.0f);
            acc_im[q] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b) {
            const float* row =
                erb_inv + (size_t)b * GTCRN_MODEL_ERB_HIGH_BINS + k0;
            const float32x4_t gr =
                vdupq_n_f32(mask_erb[GTCRN_MODEL_ERB_KEPT + b][0]);
            const float32x4_t gi =
                vdupq_n_f32(mask_erb[GTCRN_MODEL_ERB_KEPT + b][1]);
            for (int q = 0; q < MASK_BIN_QUADS; ++q) {
                const float32x4_t w = vld1q_f32(row + 4 * q);
                acc_re[q] = vaddq_f32(acc_re[q], vmulq_f32(w, gr));
                acc_im[q] = vaddq_f32(acc_im[q], vmulq_f32(w, gi));
            }
        }
        for (int q = 0; q < MASK_BIN_QUADS; ++q) {
            vst1q_f32(mask_re + GTCRN_MODEL_ERB_KEPT + k0 + 4 * q, acc_re[q]);
            vst1q_f32(mask_im + GTCRN_MODEL_ERB_KEPT + k0 + 4 * q, acc_im[q]);
        }
    }
}
#endif

static inline void mask_expand(const float mask_erb[GTCRN_MODEL_ERB_BANDS][2],
                               const float* erb_inv, float* mask_re,
                               float* mask_im)
{
#if DF_COMMON_HAVE_NEON
    mask_expand_neon(mask_erb, erb_inv, mask_re, mask_im);
#else
    mask_expand_scalar(mask_erb, erb_inv, mask_re, mask_im);
#endif
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
    mask_expand(mask_erb, erb_inv, mask_re, mask_im);
    /* Complex ratio mask (model.py Mask). */
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
