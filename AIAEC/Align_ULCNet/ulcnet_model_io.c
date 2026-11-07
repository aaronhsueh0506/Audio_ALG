#include "ulcnet_model_io.h"

#include "audio_stft.h"            /* ac_stft_get_mem_size(): the n_fft rule is audiolib's */
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Scalar twin of a NEON kernel; it is the whole kernel in a SIMD=0 build. */
#define ULCNET_TWIN static inline __attribute__((unused))

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(ULCNET_MODEL_IO_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

struct UlcnetModelIoState {
    UlcnetModelIoDescriptor descriptor;

    float *output;

    /* The application's recurrent state, borrowed: NULL and 0 when none is registered. */
    float *key_history;
    float *value_history;
    float *logit_history;
    float *h_gru;

    /* Feature tensors computed by prepare(), and the compressed error spectrum kept for enhance_spectrum(). */
    float *error_mag;
    float *far_mag;
    float *error_cos;
    float *error_sin;
    float *error_ri;

    size_t spectrum_ri_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
    uint32_t state_generation;
    int prepared;
};

/* Pool arithmetic uses audiolib's saturating helpers: overflow pins the cursor to SIZE_MAX (MEM_SIZE_INVALID). */

/* Reserve count * elem bytes at *cursor, aligned; base == NULL sizes only. */
static int mio_carve(unsigned char *base, size_t *cursor, size_t count, size_t elem,
                     void **out) {
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, ck_mul_size(count, elem));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

/* Walk the layout once: base == NULL sizes it, otherwise it also carves into `state`; get_mem_requirements and init
 * share this walk so their order agrees. */
static int mio_layout(UlcnetModelIoState *state, unsigned char *base, size_t bins,
                      size_t *total) {
    const size_t counts[] = { bins, bins, bins, bins, 2u * bins, 2u * bins };
    float **slots[sizeof counts / sizeof counts[0]] = { NULL };
    size_t cursor = 0;
    size_t i;
    void *ptr = NULL;

    if (state) {
        slots[0] = &state->error_mag;  slots[1] = &state->far_mag;
        slots[2] = &state->error_cos;  slots[3] = &state->error_sin;
        slots[4] = &state->error_ri;   slots[5] = &state->output;
    }
    if (mio_carve(base, &cursor, 1u, sizeof(UlcnetModelIoState), &ptr) != 0) return -1;
    for (i = 0; i < sizeof counts / sizeof counts[0]; ++i) {
        if (mio_carve(base, &cursor, counts[i], sizeof(float), &ptr) != 0) return -1;
        if (base && slots[i]) *slots[i] = (float *)ptr;
    }
    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

int ulcnet_grid_make(UlcnetGrid *grid, const UlcnetGridSpec *spec) {
    AcStftConfig framing;
    if (!grid || !spec) return ULCNET_GRID_ERR_NULL;
    if (spec->sample_rate <= 0) return ULCNET_GRID_ERR_SAMPLE_RATE;
    ac_stft_config_default(&framing, spec->n_fft);
    if (ac_stft_get_mem_size(&framing) == 0u) return ULCNET_GRID_ERR_N_FFT;
    if (spec->delay_depth < ULCNET_MODEL_IO_MIN_D ||
        spec->delay_depth > ULCNET_MODEL_IO_MAX_D) {
        return ULCNET_GRID_ERR_DELAY_DEPTH;
    }
    grid->sample_rate = spec->sample_rate;
    grid->n_fft = spec->n_fft;
    grid->hop = spec->n_fft / 2;
    grid->n_bins = spec->n_fft / 2 + 1;
    grid->delay_depth = spec->delay_depth;
    return ULCNET_GRID_OK;
}

int ulcnet_grid_validate(const UlcnetGrid *grid) {
    UlcnetGridSpec spec;
    UlcnetGrid want;
    int status;
    if (!grid) return ULCNET_GRID_ERR_NULL;
    spec.sample_rate = grid->sample_rate;
    spec.n_fft = grid->n_fft;
    spec.delay_depth = grid->delay_depth;
    status = ulcnet_grid_make(&want, &spec);
    if (status != ULCNET_GRID_OK) return status;
    return (grid->hop == want.hop && grid->n_bins == want.n_bins)
               ? ULCNET_GRID_OK : ULCNET_GRID_ERR_INCONSISTENT;
}

int ulcnet_model_io_descriptor_default(const UlcnetGrid *grid,
                                       UlcnetModelIoDescriptor *descriptor) {
    if (!descriptor || ulcnet_grid_validate(grid) != ULCNET_GRID_OK) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->layout_version = ULCNET_MODEL_IO_LAYOUT_VERSION;
    descriptor->delay_depth = grid->delay_depth;
    descriptor->sample_rate = grid->sample_rate;
    descriptor->fft_size = grid->n_fft;
    descriptor->hop_size = grid->hop;
    descriptor->spectrum_bins = grid->n_bins;
    descriptor->far_input_mode = ULCNET_FAR_RAW;
    return 0;
}

const char *ulcnet_far_input_mode_name(int mode) {
    switch (mode) {
        case ULCNET_FAR_RAW:     return "raw_far";
        case ULCNET_FAR_ALIGNED: return "aligned_far";
        default:                 return "unknown";
    }
}

int ulcnet_model_io_descriptor_validate(
    const UlcnetGrid *grid, const UlcnetModelIoDescriptor *descriptor) {
    if (!descriptor || ulcnet_grid_validate(grid) != ULCNET_GRID_OK ||
        descriptor->layout_version != ULCNET_MODEL_IO_LAYOUT_VERSION ||
        descriptor->delay_depth < ULCNET_MODEL_IO_MIN_D ||
        descriptor->delay_depth > ULCNET_MODEL_IO_MAX_D ||
        descriptor->sample_rate != grid->sample_rate ||
        descriptor->fft_size != grid->n_fft ||
        descriptor->hop_size != grid->hop ||
        descriptor->spectrum_bins != grid->n_bins ||
        descriptor->far_input_mode != ULCNET_FAR_RAW) {
        return -1;
    }
    return 0;
}

int ulcnet_model_io_descriptor_grid(const UlcnetModelIoDescriptor *descriptor,
                                    UlcnetGrid *grid) {
    UlcnetGridSpec spec;
    UlcnetGrid made;
    if (!descriptor || !grid) return -1;
    spec.sample_rate = descriptor->sample_rate;
    spec.n_fft = descriptor->fft_size;
    spec.delay_depth = descriptor->delay_depth;
    if (ulcnet_grid_make(&made, &spec) != ULCNET_GRID_OK ||
        ulcnet_model_io_descriptor_validate(&made, descriptor) != 0) {
        return -1;
    }
    *grid = made;
    return 0;
}

int ulcnet_model_io_state_buffers_validate(
    const UlcnetModelIoStateBuffers *buffers) {
    int set;
    if (!buffers) return 0;
    set = (buffers->key_history != NULL) + (buffers->value_history != NULL) +
          (buffers->logit_history != NULL) + (buffers->h_gru != NULL);
    if (set == 0 && buffers->key_history_elements == 0u &&
        buffers->value_history_elements == 0u &&
        buffers->logit_history_elements == 0u &&
        buffers->gru_hidden_elements == 0u) {
        return 0;
    }
    /* Registered: every pair set, each a pointer with a nonzero count. */
    return (set == 4 && buffers->key_history_elements != 0u &&
            buffers->value_history_elements != 0u &&
            buffers->logit_history_elements != 0u &&
            buffers->gru_hidden_elements != 0u) ? 0 : -1;
}

int ulcnet_model_io_get_mem_requirements(
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetModelIoMemReq *requirements) {
    UlcnetGrid grid;
    size_t bytes;

    if (!requirements || ulcnet_model_io_descriptor_grid(descriptor, &grid) != 0 ||
        mio_layout(NULL, NULL, (size_t)descriptor->spectrum_bins, &bytes) != 0) {
        return -1;
    }
    requirements->bytes = bytes;
    requirements->alignment = ULCNET_MODEL_IO_ALIGNMENT;
    return 0;
}

/* Zero the registered tensors; with none registered there is nothing to zero. */
static void zero_registered(UlcnetModelIoState *state) {
    if (state->key_history_elements != 0u)
        memset(state->key_history, 0, state->key_history_elements * sizeof(float));
    if (state->value_history_elements != 0u)
        memset(state->value_history, 0, state->value_history_elements * sizeof(float));
    if (state->logit_history_elements != 0u)
        memset(state->logit_history, 0, state->logit_history_elements * sizeof(float));
    if (state->gru_hidden_elements != 0u)
        memset(state->h_gru, 0, state->gru_hidden_elements * sizeof(float));
}

UlcnetModelIoState *ulcnet_model_io_init(
    void *pool,
    size_t pool_bytes,
    const UlcnetModelIoDescriptor *descriptor,
    const UlcnetModelIoStateBuffers *buffers) {
    UlcnetModelIoMemReq requirements;
    UlcnetModelIoState *state;
    size_t total = 0;

    if (!pool || !MEM_IS_ALIGNED16(pool) ||
        ulcnet_model_io_state_buffers_validate(buffers) != 0 ||
        ulcnet_model_io_get_mem_requirements(descriptor, &requirements) != 0 ||
        pool_bytes < requirements.bytes) {
        return NULL;
    }

    memset(pool, 0, requirements.bytes);
    state = (UlcnetModelIoState *)pool;
    state->descriptor = *descriptor;
    state->spectrum_ri_elements = 2u * (size_t)descriptor->spectrum_bins;
    if (buffers) {
        state->key_history = buffers->key_history;
        state->value_history = buffers->value_history;
        state->logit_history = buffers->logit_history;
        state->h_gru = buffers->h_gru;
        state->key_history_elements = buffers->key_history_elements;
        state->value_history_elements = buffers->value_history_elements;
        state->logit_history_elements = buffers->logit_history_elements;
        state->gru_hidden_elements = buffers->gru_hidden_elements;
    }

    if (mio_layout(state, (unsigned char *)pool,
                   (size_t)descriptor->spectrum_bins, &total) != 0 ||
        total != requirements.bytes) {
        return NULL;
    }
    zero_registered(state);
    return state;
}

/* Zero the registered recurrent state and advance the generation. */
static void clear_recurrent_state(UlcnetModelIoState *state) {
    zero_registered(state);
    ++state->state_generation;
}

void ulcnet_model_io_reset(UlcnetModelIoState *state) {
    if (!state) {
        return;
    }
    clear_recurrent_state(state);
    state->prepared = 0;
}

uint32_t ulcnet_model_io_state_generation(const UlcnetModelIoState *state) {
    return state ? state->state_generation : 0u;
}

/* Front end: signed-power compression, magnitudes and compressed-domain phase direction (cos/sin of atan2); hypotf
 * avoids underflow, a zero vector gives cos 1, sin 0. Python reference: export_onnx.stream_features. */
ULCNET_TWIN void front_end_scalar(UlcnetModelIoState *state,
                                  const float *error_re, const float *error_im,
                                  const float *far_re, const float *far_im) {
    int bin;

    for (bin = 0; bin < state->descriptor.spectrum_bins; ++bin) {
        float c_re = ulcnet_model_io_signed_pow(
            error_re[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float c_im = ulcnet_model_io_signed_pow(
            error_im[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float f_re = ulcnet_model_io_signed_pow(
            far_re[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float f_im = ulcnet_model_io_signed_pow(
            far_im[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float norm = hypotf(c_re, c_im);
        state->error_mag[bin] = sqrtf(c_re * c_re + c_im * c_im + 1e-12f);
        state->far_mag[bin] = sqrtf(f_re * f_re + f_im * f_im + 1e-12f);
        state->error_cos[bin] = norm > 0.0f ? c_re / norm : 1.0f;
        state->error_sin[bin] = norm > 0.0f ? c_im / norm : 0.0f;
        state->error_ri[2 * bin] = c_re;
        state->error_ri[2 * bin + 1] = c_im;
    }
}

#if SK_HAVE_NEON
/* Two-pass front end: libm calls run per bin into planes the frame owns, then
 * sqrt, divide and direction run four bins at a time. */
static inline void front_end_tail_one(UlcnetModelIoState *state, int bin) {
    const float c_re = state->error_ri[2 * bin];
    const float c_im = state->error_ri[2 * bin + 1];
    const float f_re = state->output[bin];
    const float f_im = state->output[state->descriptor.spectrum_bins + bin];
    const float norm = state->error_cos[bin];
    state->error_mag[bin] = sqrtf(c_re * c_re + c_im * c_im + 1e-12f);
    state->far_mag[bin] = sqrtf(f_re * f_re + f_im * f_im + 1e-12f);
    state->error_cos[bin] = norm > 0.0f ? c_re / norm : 1.0f;
    state->error_sin[bin] = norm > 0.0f ? c_im / norm : 0.0f;
}

static inline void front_end_neon(UlcnetModelIoState *state,
                                  const float *error_re, const float *error_im,
                                  const float *far_re, const float *far_im) {
    const int bins = state->descriptor.spectrum_bins;
    float *far_plane_re = state->output;
    float *far_plane_im = state->output + bins;
    const float32x4_t eps = vdupq_n_f32(1e-12f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    int bin;

    for (bin = 0; bin < bins; ++bin) {
        float c_re = ulcnet_model_io_signed_pow(
            error_re[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float c_im = ulcnet_model_io_signed_pow(
            error_im[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        far_plane_re[bin] = ulcnet_model_io_signed_pow(
            far_re[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        far_plane_im[bin] = ulcnet_model_io_signed_pow(
            far_im[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        state->error_cos[bin] = hypotf(c_re, c_im);
        state->error_ri[2 * bin] = c_re;
        state->error_ri[2 * bin + 1] = c_im;
    }
    for (bin = 0; bin + 4 <= bins; bin += 4) {
        const float32x4x2_t c = vld2q_f32(state->error_ri + 2 * bin);
        const float32x4_t f_re = vld1q_f32(far_plane_re + bin);
        const float32x4_t f_im = vld1q_f32(far_plane_im + bin);
        const float32x4_t norm = vld1q_f32(state->error_cos + bin);
        const uint32x4_t positive = vcgtq_f32(norm, zero);
        vst1q_f32(state->error_mag + bin, vsqrtq_f32(vaddq_f32(
            vaddq_f32(vmulq_f32(c.val[0], c.val[0]),
                      vmulq_f32(c.val[1], c.val[1])), eps)));
        vst1q_f32(state->far_mag + bin, vsqrtq_f32(vaddq_f32(
            vaddq_f32(vmulq_f32(f_re, f_re), vmulq_f32(f_im, f_im)), eps)));
        vst1q_f32(state->error_cos + bin,
                  vbslq_f32(positive, vdivq_f32(c.val[0], norm), one));
        vst1q_f32(state->error_sin + bin,
                  vbslq_f32(positive, vdivq_f32(c.val[1], norm), zero));
    }
    for (; bin < bins; ++bin) front_end_tail_one(state, bin);
}
#endif

static inline void front_end(UlcnetModelIoState *state, const float *error_re,
                             const float *error_im, const float *far_re,
                             const float *far_im) {
#if SK_HAVE_NEON
    front_end_neon(state, error_re, error_im, far_re, far_im);
#else
    front_end_scalar(state, error_re, error_im, far_re, far_im);
#endif
}

int ulcnet_model_io_prepare(UlcnetModelIoState *state,
                            const float *error_re, const float *error_im,
                            const float *far_re, const float *far_im,
                            UlcnetModelIoInputs *inputs,
                            UlcnetModelIoOutputs *outputs) {
    if (!state || !error_re || !error_im || !far_re || !far_im || !inputs ||
        !outputs) {
        return -1;
    }
    front_end(state, error_re, error_im, far_re, far_im);

    skn_fill_f32(state->output, state->spectrum_ri_elements, NAN);

    inputs->error_mag = state->error_mag;
    inputs->far_mag = state->far_mag;
    inputs->error_cos = state->error_cos;
    inputs->error_sin = state->error_sin;
    inputs->key_history = state->key_history;
    inputs->value_history = state->value_history;
    inputs->logit_history = state->logit_history;
    inputs->h_gru = state->h_gru;
    inputs->spectrum_bins_elements = (size_t)state->descriptor.spectrum_bins;
    inputs->key_history_elements = state->key_history_elements;
    inputs->value_history_elements = state->value_history_elements;
    inputs->logit_history_elements = state->logit_history_elements;
    inputs->gru_hidden_elements = state->gru_hidden_elements;

    outputs->output = state->output;
    outputs->key_history_out = state->key_history;
    outputs->value_history_out = state->value_history;
    outputs->logit_history_out = state->logit_history;
    outputs->h_gru_out = state->h_gru;
    outputs->spectrum_ri_elements = state->spectrum_ri_elements;
    outputs->key_history_elements = state->key_history_elements;
    outputs->value_history_elements = state->value_history_elements;
    outputs->logit_history_elements = state->logit_history_elements;
    outputs->gru_hidden_elements = state->gru_hidden_elements;
    state->prepared = 1;
    return 0;
}

static int all_finite_or_empty(const float *values, size_t count) {
    return count == 0u || skn_all_finite_f32(values, count);
}

/* Finite check over what the accelerator wrote: the mask and the whole of every registered state buffer (a
 * non-finite value anywhere in a window would reach `output` on a later frame). */
static int frame_is_finite(const UlcnetModelIoState *state) {
    return skn_all_finite_f32(state->output, state->spectrum_ri_elements) &&
           all_finite_or_empty(state->h_gru, state->gru_hidden_elements) &&
           all_finite_or_empty(state->key_history, state->key_history_elements) &&
           all_finite_or_empty(state->value_history, state->value_history_elements) &&
           all_finite_or_empty(state->logit_history, state->logit_history_elements);
}

/* Back end: compressed error times the planar complex mask, then inverse signed power. Products stay unfused
 * (-ffp-contract=off). Python reference: export_onnx.apply_mask_ri. */
ULCNET_TWIN void back_end_scalar(const UlcnetModelIoState *state,
                                 float *enhanced_re, float *enhanced_im) {
    const float inverse_exponent = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
    const float *mask_re = state->output;
    const float *mask_im = state->output + state->descriptor.spectrum_bins;
    int bin;

    for (bin = 0; bin < state->descriptor.spectrum_bins; ++bin) {
        const float err_re = state->error_ri[2 * bin];
        const float err_im = state->error_ri[2 * bin + 1];
        enhanced_re[bin] = ulcnet_model_io_signed_pow(
            err_re * mask_re[bin] - err_im * mask_im[bin], inverse_exponent);
        enhanced_im[bin] = ulcnet_model_io_signed_pow(
            err_re * mask_im[bin] + err_im * mask_re[bin], inverse_exponent);
    }
}

#if SK_HAVE_NEON
/* Complex products four bins at a time, then the libm pass in place. */
static inline void back_end_neon(const UlcnetModelIoState *state,
                                 float *enhanced_re, float *enhanced_im) {
    const float inverse_exponent = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
    const int bins = state->descriptor.spectrum_bins;
    const float *mask_re = state->output;
    const float *mask_im = state->output + bins;
    int bin;

    for (bin = 0; bin + 4 <= bins; bin += 4) {
        const float32x4x2_t err = vld2q_f32(state->error_ri + 2 * bin);
        const float32x4_t m_re = vld1q_f32(mask_re + bin);
        const float32x4_t m_im = vld1q_f32(mask_im + bin);
        vst1q_f32(enhanced_re + bin,
                  vsubq_f32(vmulq_f32(err.val[0], m_re),
                            vmulq_f32(err.val[1], m_im)));
        vst1q_f32(enhanced_im + bin,
                  vaddq_f32(vmulq_f32(err.val[0], m_im),
                            vmulq_f32(err.val[1], m_re)));
    }
    for (; bin < bins; ++bin) {
        const float err_re = state->error_ri[2 * bin];
        const float err_im = state->error_ri[2 * bin + 1];
        enhanced_re[bin] = err_re * mask_re[bin] - err_im * mask_im[bin];
        enhanced_im[bin] = err_re * mask_im[bin] + err_im * mask_re[bin];
    }
    for (bin = 0; bin < bins; ++bin) {
        enhanced_re[bin] = ulcnet_model_io_signed_pow(enhanced_re[bin],
                                                      inverse_exponent);
        enhanced_im[bin] = ulcnet_model_io_signed_pow(enhanced_im[bin],
                                                      inverse_exponent);
    }
}
#endif

static inline void back_end(const UlcnetModelIoState *state,
                            float *enhanced_re, float *enhanced_im) {
#if SK_HAVE_NEON
    back_end_neon(state, enhanced_re, enhanced_im);
#else
    back_end_scalar(state, enhanced_re, enhanced_im);
#endif
}

int ulcnet_model_io_enhance_spectrum(UlcnetModelIoState *state,
                                     float *enhanced_re, float *enhanced_im) {
    if (!state || !state->prepared || !enhanced_re || !enhanced_im) {
        return -1;
    }
    /* State is updated in place, so a bad frame cannot be rolled back: restart the recurrent state (the registered
     * buffers; a runtime that holds the state itself sees the generation). */
    if (!frame_is_finite(state)) {
        clear_recurrent_state(state);
        state->prepared = 0;
        return -1;
    }

    back_end(state, enhanced_re, enhanced_im);
    state->prepared = 0;
    return 0;
}

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state) {
    return state ? &state->descriptor : NULL;
}
