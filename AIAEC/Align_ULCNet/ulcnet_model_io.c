#include "ulcnet_model_io.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

struct UlcnetModelIoState {
    UlcnetModelIoDescriptor descriptor;

    float *output;

    float *key_history;
    float *value_history;
    float *logit_history;
    float *h_gru0;
    float *h_gru1;

    /* The five graph feature tensors prepare() computes. */
    float *error_mag;
    float *far_mag;
    float *error_cos;
    float *error_sin;
    float *error_ri;

    float *key_now;
    float *value_now;
    float *logit_now;
    float *h_gru0_out;
    float *h_gru1_out;

    size_t spectrum_ri_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
    size_t key_now_elements;
    size_t value_now_elements;
    size_t logit_now_elements;
    int prepared;
};

typedef struct UlcnetModelIoCounts {
    size_t spectrum_ri_elements;
    size_t spectrum_bins_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
    size_t key_now_elements;
    size_t value_now_elements;
    size_t logit_now_elements;
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
    size_t history_depth;
    size_t one_feature;

    if (!descriptor || !counts ||
        ulcnet_model_io_descriptor_validate(descriptor) != 0) {
        return -1;
    }
    memset(counts, 0, sizeof(*counts));
    channels = (size_t)descriptor->ta_channels;
    bins = (size_t)descriptor->ta_bins;
    depth = (size_t)descriptor->delay_depth;
    history_depth = depth - 1u;

    counts->spectrum_bins_elements = (size_t)descriptor->spectrum_bins;
    if (checked_mul((size_t)descriptor->spectrum_bins, 2u,
                    &counts->spectrum_ri_elements) != 0 ||
        checked_mul(channels, bins, &one_feature) != 0 ||
        checked_mul(one_feature, history_depth,
                    &counts->key_history_elements) != 0 ||
        checked_mul(channels, (size_t)descriptor->score_history_frames,
                    &counts->logit_history_elements) != 0 ||
        checked_mul(counts->logit_history_elements, depth,
                    &counts->logit_history_elements) != 0 ||
        checked_mul((size_t)descriptor->gru_layers,
                    (size_t)descriptor->gru_hidden,
                    &counts->gru_hidden_elements) != 0 ||
        checked_mul(channels, depth, &counts->logit_now_elements) != 0) {
        return -1;
    }
    counts->value_history_elements = counts->key_history_elements;
    counts->key_now_elements = one_feature;
    counts->value_now_elements = one_feature;
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
    descriptor->sample_rate = 16000;
    descriptor->fft_size = 512;
    descriptor->hop_size = 256;
    descriptor->spectrum_bins = 257;
    descriptor->ta_channels = ULCNET_MODEL_IO_TA_CHANNELS;
    descriptor->ta_bins = ULCNET_MODEL_IO_TA_BINS;
    descriptor->score_history_frames = ULCNET_MODEL_IO_SCORE_HISTORY;
    descriptor->gru_layers = ULCNET_MODEL_IO_GRU_LAYERS;
    descriptor->gru_hidden = ULCNET_MODEL_IO_GRU_HIDDEN;
    descriptor->far_input_mode = ULCNET_FAR_ALIGNED;
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
        descriptor->sample_rate != 16000 || descriptor->fft_size != 512 ||
        descriptor->hop_size != 256 || descriptor->spectrum_bins != 257 ||
        descriptor->ta_channels != ULCNET_MODEL_IO_TA_CHANNELS ||
        descriptor->ta_bins != ULCNET_MODEL_IO_TA_BINS ||
        descriptor->score_history_frames != ULCNET_MODEL_IO_SCORE_HISTORY ||
        descriptor->gru_layers != ULCNET_MODEL_IO_GRU_LAYERS ||
        descriptor->gru_hidden != ULCNET_MODEL_IO_GRU_HIDDEN ||
        descriptor->far_input_mode != ULCNET_FAR_ALIGNED) {
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
        /* h_gru0, h_gru1 */
        add_float_regions(2u, counts.gru_hidden_elements, &bytes) != 0 ||
        add_float_region(counts.key_now_elements, &bytes) != 0 ||
        add_float_region(counts.value_now_elements, &bytes) != 0 ||
        add_float_region(counts.logit_now_elements, &bytes) != 0 ||
        /* h_gru0_out, h_gru1_out */
        add_float_regions(2u, counts.gru_hidden_elements, &bytes) != 0) {
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
    state->key_now_elements = counts.key_now_elements;
    state->value_now_elements = counts.value_now_elements;
    state->logit_now_elements = counts.logit_now_elements;

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
    state->h_gru0 = carve_float(&cursor, counts.gru_hidden_elements);
    state->h_gru1 = carve_float(&cursor, counts.gru_hidden_elements);
    state->key_now = carve_float(&cursor, counts.key_now_elements);
    state->value_now = carve_float(&cursor, counts.value_now_elements);
    state->logit_now = carve_float(&cursor, counts.logit_now_elements);
    state->h_gru0_out = carve_float(&cursor, counts.gru_hidden_elements);
    state->h_gru1_out = carve_float(&cursor, counts.gru_hidden_elements);

    if (!state->error_mag || !state->far_mag || !state->error_cos ||
        !state->error_sin || !state->error_ri ||
        !state->output || !state->key_history || !state->value_history ||
        !state->logit_history || !state->h_gru0 || !state->h_gru1 ||
        !state->key_now || !state->value_now || !state->logit_now ||
        !state->h_gru0_out || !state->h_gru1_out ||
        (size_t)(cursor - (unsigned char *)pool) != requirements.bytes) {
        return NULL;
    }
    return state;
}

void ulcnet_model_io_reset(UlcnetModelIoState *state) {
    if (!state) {
        return;
    }
    memset(state->key_history, 0,
           state->key_history_elements * sizeof(float));
    memset(state->value_history, 0,
           state->value_history_elements * sizeof(float));
    memset(state->logit_history, 0,
           state->logit_history_elements * sizeof(float));
    memset(state->h_gru0, 0,
           state->gru_hidden_elements * sizeof(float));
    memset(state->h_gru1, 0,
           state->gru_hidden_elements * sizeof(float));
    memset(state->h_gru0_out, 0,
           state->gru_hidden_elements * sizeof(float));
    memset(state->h_gru1_out, 0,
           state->gru_hidden_elements * sizeof(float));
    state->prepared = 0;
}

static void fill_nan(float *values, size_t elements) {
    size_t index;

    for (index = 0; index < elements; ++index) {
        values[index] = NAN;
    }
}

int ulcnet_model_io_prepare(UlcnetModelIoState *state,
                            const float error_re[257],
                            const float error_im[257],
                            const float far_re[257],
                            const float far_im[257],
                            UlcnetModelIoInputs *inputs,
                            UlcnetModelIoOutputs *outputs) {
    int bin;

    if (!state || !error_re || !error_im || !far_re || !far_im || !inputs ||
        !outputs) {
        return -1;
    }
    /* The fixed front end: signed-power compression, magnitudes and the
     * compressed-domain phase direction, all in fp32 on the host. */
    for (bin = 0; bin < state->descriptor.spectrum_bins; ++bin) {
        float c_re = ulcnet_model_io_signed_pow(
            error_re[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float c_im = ulcnet_model_io_signed_pow(
            error_im[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float f_re = ulcnet_model_io_signed_pow(
            far_re[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        float f_im = ulcnet_model_io_signed_pow(
            far_im[bin], ULCNET_MODEL_IO_COMPRESSION_EXP);
        /* cos/sin of atan2(c_im, c_re) as the normalized direction: hypotf
         * avoids underflow for tiny components, and the zero-vector case
         * follows atan2(0,0) == 0 (cos 1, sin 0), matching the torch
         * reference in export_onnx.stream_features to fp32 rounding. */
        float norm = hypotf(c_re, c_im);
        state->error_mag[bin] = sqrtf(c_re * c_re + c_im * c_im + 1e-12f);
        state->far_mag[bin] = sqrtf(f_re * f_re + f_im * f_im + 1e-12f);
        state->error_cos[bin] = norm > 0.0f ? c_re / norm : 1.0f;
        state->error_sin[bin] = norm > 0.0f ? c_im / norm : 0.0f;
        state->error_ri[2 * bin] = c_re;
        state->error_ri[2 * bin + 1] = c_im;
    }

    fill_nan(state->output, state->spectrum_ri_elements);
    fill_nan(state->key_now, state->key_now_elements);
    fill_nan(state->value_now, state->value_now_elements);
    fill_nan(state->logit_now, state->logit_now_elements);
    fill_nan(state->h_gru0_out, state->gru_hidden_elements);
    fill_nan(state->h_gru1_out, state->gru_hidden_elements);

    inputs->error_mag = state->error_mag;
    inputs->far_mag = state->far_mag;
    inputs->error_cos = state->error_cos;
    inputs->error_sin = state->error_sin;
    inputs->error_ri = state->error_ri;
    inputs->key_history = state->key_history;
    inputs->value_history = state->value_history;
    inputs->logit_history = state->logit_history;
    inputs->h_gru0 = state->h_gru0;
    inputs->h_gru1 = state->h_gru1;
    inputs->spectrum_ri_elements = state->spectrum_ri_elements;
    inputs->spectrum_bins_elements = (size_t)state->descriptor.spectrum_bins;
    inputs->key_history_elements = state->key_history_elements;
    inputs->value_history_elements = state->value_history_elements;
    inputs->logit_history_elements = state->logit_history_elements;
    inputs->gru_hidden_elements = state->gru_hidden_elements;

    outputs->output = state->output;
    outputs->key_now = state->key_now;
    outputs->value_now = state->value_now;
    outputs->logit_now = state->logit_now;
    outputs->h_gru0_out = state->h_gru0_out;
    outputs->h_gru1_out = state->h_gru1_out;
    outputs->spectrum_ri_elements = state->spectrum_ri_elements;
    outputs->key_now_elements = state->key_now_elements;
    outputs->value_now_elements = state->value_now_elements;
    outputs->logit_now_elements = state->logit_now_elements;
    outputs->gru_hidden_elements = state->gru_hidden_elements;
    state->prepared = 1;
    return 0;
}

static int all_finite(const float *values, size_t elements) {
    size_t index;

    for (index = 0; index < elements; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static void update_feature_history(float *history, const float *current,
                                   int channels, int frames, int bins) {
    int channel;

    if (frames <= 0) {
        return;
    }
    for (channel = 0; channel < channels; ++channel) {
        float *base = history + (size_t)channel * (size_t)frames *
            (size_t)bins;
        const float *now = current + (size_t)channel * (size_t)bins;
        if (frames > 1) {
            memmove(base + bins, base,
                    (size_t)(frames - 1) * (size_t)bins * sizeof(float));
        }
        memcpy(base, now, (size_t)bins * sizeof(float));
    }
}

static void update_logit_history(float *history, const float *current,
                                 int channels, int frames, int depth) {
    int channel;

    for (channel = 0; channel < channels; ++channel) {
        float *base = history + (size_t)channel * (size_t)frames *
            (size_t)depth;
        const float *now = current + (size_t)channel * (size_t)depth;
        if (frames > 1) {
            memmove(base, base + depth,
                    (size_t)(frames - 1) * (size_t)depth * sizeof(float));
        }
        memcpy(base + (size_t)(frames - 1) * (size_t)depth, now,
               (size_t)depth * sizeof(float));
    }
}

int ulcnet_model_io_commit(UlcnetModelIoState *state,
                           float enhanced_re[257],
                           float enhanced_im[257]) {
    const UlcnetModelIoDescriptor *descriptor;
    float *temporary;
    int bin;

    if (!state || !state->prepared || !enhanced_re || !enhanced_im ||
        !all_finite(state->output, state->spectrum_ri_elements) ||
        !all_finite(state->key_now, state->key_now_elements) ||
        !all_finite(state->value_now, state->value_now_elements) ||
        !all_finite(state->logit_now, state->logit_now_elements) ||
        !all_finite(state->h_gru0_out, state->gru_hidden_elements) ||
        !all_finite(state->h_gru1_out, state->gru_hidden_elements)) {
        if (state) {
            state->prepared = 0;
        }
        return -1;
    }

    descriptor = &state->descriptor;
    update_feature_history(state->key_history, state->key_now,
                           descriptor->ta_channels,
                           descriptor->delay_depth - 1,
                           descriptor->ta_bins);
    update_feature_history(state->value_history, state->value_now,
                           descriptor->ta_channels,
                           descriptor->delay_depth - 1,
                           descriptor->ta_bins);
    update_logit_history(state->logit_history, state->logit_now,
                         descriptor->ta_channels,
                         descriptor->score_history_frames,
                         descriptor->delay_depth);

    temporary = state->h_gru0;
    state->h_gru0 = state->h_gru0_out;
    state->h_gru0_out = temporary;
    temporary = state->h_gru1;
    state->h_gru1 = state->h_gru1_out;
    state->h_gru1_out = temporary;

    {
        /* The graph emits the COMPRESSED estimate; the fixed inverse
         * signed power runs here. */
        const float inverse_exponent = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
        for (bin = 0; bin < descriptor->spectrum_bins; ++bin) {
            enhanced_re[bin] = ulcnet_model_io_signed_pow(
                state->output[2 * bin], inverse_exponent);
            enhanced_im[bin] = ulcnet_model_io_signed_pow(
                state->output[2 * bin + 1], inverse_exponent);
        }
    }
    state->prepared = 0;
    return 0;
}

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state) {
    return state ? &state->descriptor : NULL;
}
