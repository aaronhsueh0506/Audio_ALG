#include "ulcnet_model_io.h"

#include "utility/simd_kernel_nn.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* A scalar twin is the reference of its NEON kernel and the whole kernel in a
 * SIMD=0 build; a NEON build keeps it only for the differential test. */
#define ULCNET_TWIN static inline __attribute__((unused))

struct UlcnetModelIoState {
    UlcnetModelIoDescriptor descriptor;

    float *output;

    float *key_history;
    float *value_history;
    float *logit_history;
    float *h_gru;

    /* The four graph feature tensors prepare() computes, and the compressed
     * error spectrum it keeps for commit(). */
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

typedef struct UlcnetModelIoCounts {
    size_t spectrum_ri_elements;
    size_t spectrum_bins_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoCounts;

static int checked_add(size_t left, size_t right, size_t *out) {
    if (!out || left > SIZE_MAX - right) {
        return -1;
    }
    *out = left + right;
    return 0;
}

static int checked_mul(size_t left, size_t right, size_t *out) {
    if (!out || (left != 0u && right > SIZE_MAX / left)) {
        return -1;
    }
    *out = left * right;
    return 0;
}

static int align_up(size_t value, size_t alignment, size_t *out) {
    size_t remainder;

    if (!out || alignment == 0u) {
        return -1;
    }
    remainder = value % alignment;
    if (remainder == 0u) {
        *out = value;
        return 0;
    }
    return checked_add(value, alignment - remainder, out);
}

static int add_float_region(size_t elements, size_t *bytes) {
    size_t region;

    if (checked_mul(elements, sizeof(float), &region) != 0 ||
        align_up(region, ULCNET_MODEL_IO_ALIGNMENT, &region) != 0) {
        return -1;
    }
    return checked_add(*bytes, region, bytes);
}

static int add_float_regions(size_t count, size_t elements, size_t *bytes) {
    size_t index;

    for (index = 0; index < count; ++index) {
        if (add_float_region(elements, bytes) != 0) {
            return -1;
        }
    }
    return 0;
}

static int compute_counts(const UlcnetModelIoDescriptor *descriptor,
                          UlcnetModelIoCounts *counts) {
    size_t channels;
    size_t bins;
    size_t depth;
    size_t one_feature;

    if (!descriptor || !counts ||
        ulcnet_model_io_descriptor_validate(descriptor) != 0) {
        return -1;
    }
    memset(counts, 0, sizeof(*counts));
    channels = (size_t)descriptor->ta_channels;
    bins = (size_t)descriptor->ta_bins;
    depth = (size_t)descriptor->delay_depth;

    counts->spectrum_bins_elements = (size_t)descriptor->spectrum_bins;
    if (checked_mul((size_t)descriptor->spectrum_bins, 2u,
                    &counts->spectrum_ri_elements) != 0 ||
        checked_mul(channels, bins, &one_feature) != 0 ||
        checked_mul(one_feature, depth, &counts->key_history_elements) != 0 ||
        checked_mul(channels, (size_t)descriptor->score_window_frames,
                    &counts->logit_history_elements) != 0 ||
        checked_mul(counts->logit_history_elements, depth,
                    &counts->logit_history_elements) != 0 ||
        checked_mul((size_t)descriptor->gru_layers,
                    (size_t)descriptor->gru_hidden,
                    &counts->gru_hidden_elements) != 0 ||
        checked_mul(counts->gru_hidden_elements,
                    (size_t)ULCNET_MODEL_IO_GRU_SUBBANDS,
                    &counts->gru_hidden_elements) != 0) {
        return -1;
    }
    counts->value_history_elements = counts->key_history_elements;
    return 0;
}

int ulcnet_model_io_descriptor_default(int delay_depth,
                                       UlcnetModelIoDescriptor *descriptor) {
    if (!descriptor || delay_depth < ULCNET_MODEL_IO_MIN_D ||
        delay_depth > ULCNET_MODEL_IO_MAX_D) {
        return -1;
    }
    descriptor->layout_version = ULCNET_MODEL_IO_LAYOUT_VERSION;
    descriptor->delay_depth = delay_depth;
    descriptor->sample_rate = ULCNET_MODEL_IO_SR;
    descriptor->fft_size = ULCNET_MODEL_IO_N_FFT;
    descriptor->hop_size = ULCNET_MODEL_IO_HOP;
    descriptor->spectrum_bins = ULCNET_MODEL_IO_BINS;
    descriptor->ta_channels = ULCNET_MODEL_IO_TA_CHANNELS;
    descriptor->ta_bins = ULCNET_MODEL_IO_TA_BINS;
    descriptor->score_window_frames = ULCNET_MODEL_IO_SCORE_WINDOW;
    descriptor->gru_layers = ULCNET_MODEL_IO_GRU_LAYERS;
    descriptor->gru_hidden = ULCNET_MODEL_IO_GRU_HIDDEN;
    descriptor->far_input_mode = ULCNET_FAR_RAW;
    return 0;
}

int ulcnet_model_io_align_up(size_t value, size_t alignment, size_t *out) {
    return align_up(value, alignment, out);
}

const char *ulcnet_far_input_mode_name(int mode) {
    switch (mode) {
        case ULCNET_FAR_RAW:     return "raw_far";
        case ULCNET_FAR_ALIGNED: return "aligned_far";
        default:                 return "unknown";
    }
}

int ulcnet_model_io_descriptor_validate(
    const UlcnetModelIoDescriptor *descriptor) {
    if (!descriptor ||
        descriptor->layout_version != ULCNET_MODEL_IO_LAYOUT_VERSION ||
        descriptor->delay_depth < ULCNET_MODEL_IO_MIN_D ||
        descriptor->delay_depth > ULCNET_MODEL_IO_MAX_D ||
        descriptor->sample_rate != ULCNET_MODEL_IO_SR ||
        descriptor->fft_size != ULCNET_MODEL_IO_N_FFT ||
        descriptor->hop_size != ULCNET_MODEL_IO_HOP ||
        descriptor->spectrum_bins != ULCNET_MODEL_IO_BINS ||
        descriptor->ta_channels != ULCNET_MODEL_IO_TA_CHANNELS ||
        descriptor->ta_bins != ULCNET_MODEL_IO_TA_BINS ||
        descriptor->score_window_frames != ULCNET_MODEL_IO_SCORE_WINDOW ||
        descriptor->gru_layers != ULCNET_MODEL_IO_GRU_LAYERS ||
        descriptor->gru_hidden != ULCNET_MODEL_IO_GRU_HIDDEN ||
        descriptor->far_input_mode != ULCNET_FAR_RAW) {
        return -1;
    }
    return 0;
}

int ulcnet_model_io_get_mem_requirements(
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetModelIoMemReq *requirements) {
    UlcnetModelIoCounts counts;
    size_t bytes;

    if (!requirements || compute_counts(descriptor, &counts) != 0 ||
        align_up(sizeof(UlcnetModelIoState), ULCNET_MODEL_IO_ALIGNMENT,
                 &bytes) != 0 ||
        /* error_mag, far_mag, error_cos, error_sin */
        add_float_regions(4u, counts.spectrum_bins_elements, &bytes) != 0 ||
        /* error_ri, output */
        add_float_regions(2u, counts.spectrum_ri_elements, &bytes) != 0 ||
        add_float_region(counts.key_history_elements, &bytes) != 0 ||
        add_float_region(counts.value_history_elements, &bytes) != 0 ||
        add_float_region(counts.logit_history_elements, &bytes) != 0 ||
        add_float_region(counts.gru_hidden_elements, &bytes) != 0) {
        return -1;
    }
    requirements->bytes = bytes;
    requirements->alignment = ULCNET_MODEL_IO_ALIGNMENT;
    return 0;
}

static float *carve_float(unsigned char **cursor, size_t elements) {
    float *result;
    size_t bytes;
    size_t aligned;

    if (!cursor || !*cursor ||
        checked_mul(elements, sizeof(float), &bytes) != 0 ||
        align_up(bytes, ULCNET_MODEL_IO_ALIGNMENT, &aligned) != 0) {
        return NULL;
    }
    result = (float *)(void *)*cursor;
    *cursor += aligned;
    return result;
}

UlcnetModelIoState *ulcnet_model_io_init(
    void *pool,
    size_t pool_bytes,
    const UlcnetModelIoDescriptor *descriptor) {
    UlcnetModelIoMemReq requirements;
    UlcnetModelIoCounts counts;
    UlcnetModelIoState *state;
    unsigned char *cursor;
    size_t state_bytes;

    if (!pool || ((uintptr_t)pool % ULCNET_MODEL_IO_ALIGNMENT) != 0u ||
        compute_counts(descriptor, &counts) != 0 ||
        ulcnet_model_io_get_mem_requirements(descriptor, &requirements) != 0 ||
        pool_bytes < requirements.bytes ||
        align_up(sizeof(UlcnetModelIoState), ULCNET_MODEL_IO_ALIGNMENT,
                 &state_bytes) != 0) {
        return NULL;
    }

    memset(pool, 0, requirements.bytes);
    state = (UlcnetModelIoState *)pool;
    state->descriptor = *descriptor;
    state->spectrum_ri_elements = counts.spectrum_ri_elements;
    state->key_history_elements = counts.key_history_elements;
    state->value_history_elements = counts.value_history_elements;
    state->logit_history_elements = counts.logit_history_elements;
    state->gru_hidden_elements = counts.gru_hidden_elements;

    cursor = (unsigned char *)pool + state_bytes;
    state->error_mag = carve_float(&cursor, counts.spectrum_bins_elements);
    state->far_mag = carve_float(&cursor, counts.spectrum_bins_elements);
    state->error_cos = carve_float(&cursor, counts.spectrum_bins_elements);
    state->error_sin = carve_float(&cursor, counts.spectrum_bins_elements);
    state->error_ri = carve_float(&cursor, counts.spectrum_ri_elements);
    state->output = carve_float(&cursor, counts.spectrum_ri_elements);
    state->key_history = carve_float(&cursor, counts.key_history_elements);
    state->value_history = carve_float(&cursor, counts.value_history_elements);
    state->logit_history = carve_float(&cursor, counts.logit_history_elements);
    state->h_gru = carve_float(&cursor, counts.gru_hidden_elements);

    if (!state->error_mag || !state->far_mag || !state->error_cos ||
        !state->error_sin || !state->error_ri ||
        !state->output || !state->key_history || !state->value_history ||
        !state->logit_history || !state->h_gru ||
        (size_t)(cursor - (unsigned char *)pool) != requirements.bytes) {
        return NULL;
    }
    return state;
}

/* Zero every recurrent tensor: the K/V/logit windows and the GRU hidden. */
static void clear_recurrent_state(UlcnetModelIoState *state) {
    memset(state->key_history, 0,
           state->key_history_elements * sizeof(float));
    memset(state->value_history, 0,
           state->value_history_elements * sizeof(float));
    memset(state->logit_history, 0,
           state->logit_history_elements * sizeof(float));
    memset(state->h_gru, 0, state->gru_hidden_elements * sizeof(float));
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

/* The fixed front end: signed-power compression, magnitudes and the
 * compressed-domain phase direction, all in fp32 on the host. cos/sin of
 * atan2(c_im, c_re) are the normalized direction: hypotf avoids underflow for
 * tiny components, and the zero-vector case follows atan2(0,0) == 0 (cos 1,
 * sin 0), matching the torch reference in export_onnx.stream_features to fp32
 * rounding. */
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
/* The same front end in two passes. The libm calls run per bin and leave
 * their results in planes that the frame owns anyway (the compressed error in
 * error_ri, the compressed far spectrum in the not yet filled output planes,
 * the norm in error_cos); the square roots, divides and the direction are
 * then taken four bins at a time. Each lane performs the scalar twin's
 * operations in its order, with the correctly rounded vsqrtq/vdivq. */
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
                            const float error_re[ULCNET_MODEL_IO_BINS],
                            const float error_im[ULCNET_MODEL_IO_BINS],
                            const float far_re[ULCNET_MODEL_IO_BINS],
                            const float far_im[ULCNET_MODEL_IO_BINS],
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

/* The finite predicate over the tensors one frame wrote.  The graph writes a
 * whole window back, but only the frame it just produced is new -- key/value
 * slot 0 of every channel (newest first) and the last logit frame (oldest
 * first); the older frames were checked when they were new, and a non-finite
 * value left in any of them reaches `output` on the next frame, where this
 * same check refuses it.  The window geometry follows from the element counts
 * and the compiled grid. */
static int frame_is_finite(const UlcnetModelIoState *state) {
    const size_t channels = ULCNET_MODEL_IO_TA_CHANNELS;
    const size_t key_stride = state->key_history_elements / channels;
    const size_t logit_stride = state->logit_history_elements / channels;
    const size_t depth = logit_stride / ULCNET_MODEL_IO_SCORE_WINDOW;
    const size_t newest_logit = logit_stride - depth;

    return skn_all_finite_f32(state->output, state->spectrum_ri_elements) &&
           skn_all_finite_f32(state->h_gru, state->gru_hidden_elements) &&
           skn_all_finite_rows_f32(state->key_history, channels, key_stride,
                           ULCNET_MODEL_IO_TA_BINS) &&
           skn_all_finite_rows_f32(state->value_history, channels, key_stride,
                           ULCNET_MODEL_IO_TA_BINS) &&
           skn_all_finite_rows_f32(state->logit_history + newest_logit, channels,
                           logit_stride, depth);
}

/* The fixed back end: the compressed error prepare() kept times the graph's
 * planar complex mask (plane 0 real, plane 1 imaginary), then the inverse
 * signed power. Products stay unfused (the build keeps -ffp-contract=off),
 * matching export_onnx.apply_mask_ri. */
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
/* The complex products four bins at a time into the caller's planes, then the
 * libm pass over them in place; the same operations as the scalar twin. */
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

int ulcnet_model_io_commit(UlcnetModelIoState *state,
                           float enhanced_re[ULCNET_MODEL_IO_BINS],
                           float enhanced_im[ULCNET_MODEL_IO_BINS]) {
    if (!state || !state->prepared || !enhanced_re || !enhanced_im) {
        return -1;
    }
    /* Every state tensor was written in place, so a bad frame cannot leave
     * them as they were: refuse it and restart the recurrent state cold. */
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
