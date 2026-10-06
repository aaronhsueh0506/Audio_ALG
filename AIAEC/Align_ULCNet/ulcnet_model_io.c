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
    descriptor->score_history_frames = ULCNET_MODEL_IO_SCORE_HISTORY;
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
        descriptor->score_history_frames != ULCNET_MODEL_IO_SCORE_HISTORY ||
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
        /* h_gru0, h_gru1 */
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

    if (!state->error_mag || !state->far_mag || !state->error_cos ||
        !state->error_sin || !state->error_ri ||
        !state->output || !state->key_history || !state->value_history ||
        !state->logit_history || !state->h_gru0 || !state->h_gru1 ||
        (size_t)(cursor - (unsigned char *)pool) != requirements.bytes) {
        return NULL;
    }
    return state;
}

/* Zero every recurrent tensor: the K/V/logit rings and both GRU hiddens. */
static void clear_recurrent_state(UlcnetModelIoState *state) {
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
}

void ulcnet_model_io_reset(UlcnetModelIoState *state) {
    if (!state) {
        return;
    }
    clear_recurrent_state(state);
    state->prepared = 0;
}

static void fill_nan(float *values, size_t elements) {
    size_t index;

    for (index = 0; index < elements; ++index) {
        values[index] = NAN;
    }
}

int ulcnet_model_io_prepare(UlcnetModelIoState *state,
                            const float error_re[ULCNET_MODEL_IO_BINS],
                            const float error_im[ULCNET_MODEL_IO_BINS],
                            const float far_re[ULCNET_MODEL_IO_BINS],
                            const float far_im[ULCNET_MODEL_IO_BINS],
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

    inputs->error_mag = state->error_mag;
    inputs->far_mag = state->far_mag;
    inputs->error_cos = state->error_cos;
    inputs->error_sin = state->error_sin;
    inputs->key_history = state->key_history;
    inputs->value_history = state->value_history;
    inputs->logit_history = state->logit_history;
    inputs->h_gru0 = state->h_gru0;
    inputs->h_gru1 = state->h_gru1;
    inputs->spectrum_bins_elements = (size_t)state->descriptor.spectrum_bins;
    inputs->key_history_elements = state->key_history_elements;
    inputs->value_history_elements = state->value_history_elements;
    inputs->logit_history_elements = state->logit_history_elements;
    inputs->gru_hidden_elements = state->gru_hidden_elements;

    outputs->output = state->output;
    outputs->key_history_out = state->key_history;
    outputs->value_history_out = state->value_history;
    outputs->logit_history_out = state->logit_history;
    outputs->h_gru0_out = state->h_gru0;
    outputs->h_gru1_out = state->h_gru1;
    outputs->spectrum_ri_elements = state->spectrum_ri_elements;
    outputs->key_history_elements = state->key_history_elements;
    outputs->value_history_elements = state->value_history_elements;
    outputs->logit_history_elements = state->logit_history_elements;
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

/* Finite check of the `length` floats at `offset` inside each of `channels`
 * per-channel blocks of `stride` floats. */
static int all_finite_per_channel(const float *base, int channels,
                                  size_t stride, size_t offset,
                                  size_t length) {
    int channel;

    for (channel = 0; channel < channels; ++channel) {
        if (!all_finite(base + (size_t)channel * stride + offset, length)) {
            return 0;
        }
    }
    return 1;
}

/* The finite predicate over the tensors one frame wrote, shared by commit()
 * (the state's own buffers) and inherit() (the runtime's).  A NULL tensor is
 * skipped.  The graph shifts each ring by one frame, so only the frame it
 * just wrote needs a check -- key/value slot 0 (newest first) and the last
 * logit frame (oldest first); the older slots were checked when they were
 * new, and a non-finite value left in any of them reaches `output` on the
 * next frame, where this same check refuses it.  The ring geometry follows
 * from the element counts and the compiled grid. */
static int frame_is_finite(const float *output, const float *key_history,
                           const float *value_history,
                           const float *logit_history, const float *h_gru0,
                           const float *h_gru1,
                           const UlcnetModelIoOutputs *counts) {
    const size_t key_stride = counts->key_history_elements /
        ULCNET_MODEL_IO_TA_CHANNELS;
    const size_t logit_stride = counts->logit_history_elements /
        ULCNET_MODEL_IO_TA_CHANNELS;
    const size_t depth = logit_stride / ULCNET_MODEL_IO_SCORE_HISTORY;

    return (!output || all_finite(output, counts->spectrum_ri_elements)) &&
        (!key_history ||
         all_finite_per_channel(key_history, ULCNET_MODEL_IO_TA_CHANNELS,
                                key_stride, 0u, ULCNET_MODEL_IO_TA_BINS)) &&
        (!value_history ||
         all_finite_per_channel(value_history, ULCNET_MODEL_IO_TA_CHANNELS,
                                key_stride, 0u, ULCNET_MODEL_IO_TA_BINS)) &&
        (!logit_history ||
         all_finite_per_channel(logit_history, ULCNET_MODEL_IO_TA_CHANNELS,
                                logit_stride, logit_stride - depth, depth)) &&
        (!h_gru0 || all_finite(h_gru0, counts->gru_hidden_elements)) &&
        (!h_gru1 || all_finite(h_gru1, counts->gru_hidden_elements));
}

static int counts_match(const UlcnetModelIoOutputs *left,
                        const UlcnetModelIoOutputs *right) {
    return left->spectrum_ri_elements == right->spectrum_ri_elements &&
        left->key_history_elements == right->key_history_elements &&
        left->value_history_elements == right->value_history_elements &&
        left->logit_history_elements == right->logit_history_elements &&
        left->gru_hidden_elements == right->gru_hidden_elements;
}

/* Copy `elements` floats unless the runtime already wrote them in place. */
static void inherit_tensor(float *destination, const float *source,
                           size_t elements) {
    if (destination != source) {
        memcpy(destination, source, elements * sizeof(float));
    }
}

int ulcnet_model_io_inherit(const UlcnetModelIoOutputs *destination,
                            const UlcnetModelIoOutputs *runtime) {
    if (!destination || !runtime || !destination->output ||
        !destination->key_history_out || !destination->value_history_out ||
        !destination->logit_history_out || !destination->h_gru0_out ||
        !destination->h_gru1_out ||
        !runtime->output || !runtime->key_history_out ||
        !runtime->value_history_out || !runtime->logit_history_out ||
        !runtime->h_gru0_out || !runtime->h_gru1_out ||
        !counts_match(destination, runtime) ||
        destination->key_history_elements % ULCNET_MODEL_IO_TA_CHANNELS != 0u ||
        destination->logit_history_elements %
            (ULCNET_MODEL_IO_TA_CHANNELS * ULCNET_MODEL_IO_SCORE_HISTORY) !=
            0u) {
        return -1;
    }
    /* Refuse before writing anything: a non-finite frame leaves the state
     * as it was.  A tensor the runtime wrote in place is the state itself,
     * so it is neither copied nor checked here; commit() checks it. */
#define ULCNET_COPIED(field) \
    (runtime->field != destination->field ? runtime->field : NULL)
    if (!frame_is_finite(ULCNET_COPIED(output),
                         ULCNET_COPIED(key_history_out),
                         ULCNET_COPIED(value_history_out),
                         ULCNET_COPIED(logit_history_out),
                         ULCNET_COPIED(h_gru0_out),
                         ULCNET_COPIED(h_gru1_out), runtime)) {
        return -1;
    }
#undef ULCNET_COPIED
    inherit_tensor(destination->output, runtime->output,
                   destination->spectrum_ri_elements);
    inherit_tensor(destination->key_history_out, runtime->key_history_out,
                   destination->key_history_elements);
    inherit_tensor(destination->value_history_out,
                   runtime->value_history_out,
                   destination->value_history_elements);
    inherit_tensor(destination->logit_history_out,
                   runtime->logit_history_out,
                   destination->logit_history_elements);
    inherit_tensor(destination->h_gru0_out, runtime->h_gru0_out,
                   destination->gru_hidden_elements);
    inherit_tensor(destination->h_gru1_out, runtime->h_gru1_out,
                   destination->gru_hidden_elements);
    return 0;
}

int ulcnet_model_io_commit(UlcnetModelIoState *state,
                           float enhanced_re[ULCNET_MODEL_IO_BINS],
                           float enhanced_im[ULCNET_MODEL_IO_BINS]) {
    const UlcnetModelIoDescriptor *descriptor;
    UlcnetModelIoOutputs counts;
    int bin;

    if (!state || !state->prepared || !enhanced_re || !enhanced_im) {
        return -1;
    }
    descriptor = &state->descriptor;
    counts.spectrum_ri_elements = state->spectrum_ri_elements;
    counts.key_history_elements = state->key_history_elements;
    counts.value_history_elements = state->value_history_elements;
    counts.logit_history_elements = state->logit_history_elements;
    counts.gru_hidden_elements = state->gru_hidden_elements;
    /* Every state tensor was written in place, so a bad frame cannot leave
     * them as they were: refuse it and restart the recurrent state cold. */
    if (!frame_is_finite(state->output, state->key_history,
                         state->value_history, state->logit_history,
                         state->h_gru0, state->h_gru1, &counts)) {
        clear_recurrent_state(state);
        state->prepared = 0;
        return -1;
    }

    {
        /* The graph emits the planar complex mask.  The fixed back end runs
         * here: the compressed error prepare() kept times the mask, then the
         * inverse signed power.  Products stay unfused (the build keeps
         * -ffp-contract=off), matching export_onnx.apply_mask_ri. */
        const float inverse_exponent = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
        const float *mask_re = state->output;
        const float *mask_im = state->output + descriptor->spectrum_bins;
        for (bin = 0; bin < descriptor->spectrum_bins; ++bin) {
            const float err_re = state->error_ri[2 * bin];
            const float err_im = state->error_ri[2 * bin + 1];
            enhanced_re[bin] = ulcnet_model_io_signed_pow(
                err_re * mask_re[bin] - err_im * mask_im[bin],
                inverse_exponent);
            enhanced_im[bin] = ulcnet_model_io_signed_pow(
                err_re * mask_im[bin] + err_im * mask_re[bin],
                inverse_exponent);
        }
    }
    state->prepared = 0;
    return 0;
}

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state) {
    return state ? &state->descriptor : NULL;
}
