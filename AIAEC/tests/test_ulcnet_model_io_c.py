"""C contract tests for Align-ULCNet in-place state storage."""

import os
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ULCNET_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'Align_ULCNet')

_DRIVER = r'''
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ulcnet_model_io.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

_Alignas(16) static unsigned char pool[1024 * 1024];
_Alignas(16) static unsigned char pool_copy[1024 * 1024];

/* Independent formulation of model.py's _signed_power (deliberately NOT
 * copysignf like the implementation, so agreement is a real check). The
 * exponent itself comes from the header contract define, which the Python
 * suite pins against export_onnx.COMPRESSION_EXPONENT. */
static float signed_power_ref(float value, float exponent) {
    float magnitude = powf(fabsf(value), exponent);
    return value < 0.0f ? -magnitude : magnitude;
}

static int all_zero(const float *values, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (values[index] != 0.0f) return 0;
    return 1;
}

/* The graph's ring shift, performed by the stand-in accelerator: every
 * state input is read (the caller stages `now` from them) before any state
 * output is written, and each *_out is the input's own memory. K/V rings are
 * newest-first, the logit history oldest-first. */
static void push_history(const UlcnetModelIoOutputs *out,
                         const float *key_now, const float *value_now,
                         const float *logit_now, size_t ta_bins,
                         size_t depth) {
    const size_t channels = 32u;
    const size_t score_frames = 4u;
    size_t channel;
    for (channel = 0; channel < channels; ++channel) {
        float *key = out->key_history_out + channel * (depth - 1u) * ta_bins;
        float *value = out->value_history_out +
            channel * (depth - 1u) * ta_bins;
        float *logit = out->logit_history_out +
            channel * score_frames * depth;
        memmove(key + ta_bins, key, (depth - 2u) * ta_bins * sizeof(float));
        memcpy(key, key_now + channel * ta_bins, ta_bins * sizeof(float));
        memmove(value + ta_bins, value,
                (depth - 2u) * ta_bins * sizeof(float));
        memcpy(value, value_now + channel * ta_bins,
               ta_bins * sizeof(float));
        memmove(logit, logit + depth,
                (score_frames - 1u) * depth * sizeof(float));
        memcpy(logit + (score_frames - 1u) * depth,
               logit_now + channel * depth, depth * sizeof(float));
    }
}

/* The planar mask the fake accelerator writes: plane 0 real, plane 1
 * imaginary. Dyadic steps keep the values exact; both planes change with
 * `base`, so two frames never share a mask. */
static float mask_re(float base, size_t bin) {
    return 0.5f + 0.0625f * base + 0.0078125f * (float)bin;
}

static float mask_im(float base, size_t bin) {
    return -0.25f + 0.03125f * base + 0.00390625f * (float)bin;
}

static void write_outputs(UlcnetModelIoOutputs *outputs, float base) {
    static float key_now[32u * ULCNET_MODEL_IO_TA_BINS];
    static float value_now[32u * ULCNET_MODEL_IO_TA_BINS];
    static float logit_now[32u * 8u];
    size_t index;
    for (index = 0; index < ULCNET_MODEL_IO_BINS; ++index) {
        outputs->output[index] = mask_re(base, index);
        outputs->output[ULCNET_MODEL_IO_BINS + index] = mask_im(base, index);
    }
    for (index = 0; index < 32u * ULCNET_MODEL_IO_TA_BINS; ++index) {
        key_now[index] = base + (float)index;
        value_now[index] = base + 10000.0f + (float)index;
    }
    for (index = 0; index < 32u * 8u; ++index)
        logit_now[index] = base + 20000.0f + (float)index;
    push_history(outputs, key_now, value_now, logit_now,
                 ULCNET_MODEL_IO_TA_BINS, 8u);
    for (index = 0; index < outputs->gru_hidden_elements; ++index) {
        outputs->h_gru0_out[index] = base + 30000.0f + (float)index;
        outputs->h_gru1_out[index] = base + 40000.0f + (float)index;
    }
}

int main(void) {
    UlcnetModelIoDescriptor d4, d8, d64, invalid;
    UlcnetModelIoMemReq r4, r8, r64, r8b;
    UlcnetModelIoState *state;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;
    const size_t feature = ULCNET_MODEL_IO_TA_BINS;
    const size_t logit_frame = 8u;
    float error_re[ULCNET_MODEL_IO_BINS], error_im[ULCNET_MODEL_IO_BINS];
    float far_re[ULCNET_MODEL_IO_BINS], far_im[ULCNET_MODEL_IO_BINS];
    float enhanced_re[ULCNET_MODEL_IO_BINS];
    float enhanced_im[ULCNET_MODEL_IO_BINS];
    const float *gru0_address;
    const float *gru1_address;
    size_t index;

    for (index = 0; index < ULCNET_MODEL_IO_BINS; ++index) {
        error_re[index] = (float)index;
        error_im[index] = -(float)index;
        far_re[index] = 1000.0f + (float)index;
        far_im[index] = -1000.0f - (float)index;
        enhanced_re[index] = -7.0f;
        enhanced_im[index] = -8.0f;
    }

    CHECK(ulcnet_model_io_descriptor_default(4, &d4) == 0);
    CHECK(ulcnet_model_io_descriptor_default(8, &d8) == 0);
    CHECK(ulcnet_model_io_descriptor_default(64, &d64) == 0);
    CHECK(ulcnet_model_io_descriptor_default(1, &invalid) != 0);
    CHECK(ulcnet_model_io_descriptor_default(65, &invalid) != 0);
    CHECK(ulcnet_model_io_get_mem_requirements(&d4, &r4) == 0);
    CHECK(ulcnet_model_io_get_mem_requirements(&d8, &r8) == 0);
    CHECK(ulcnet_model_io_get_mem_requirements(&d64, &r64) == 0);
    CHECK(r4.bytes < r8.bytes && r8.bytes < r64.bytes);
    CHECK(r8.alignment == ULCNET_MODEL_IO_ALIGNMENT);
    CHECK(d8.sample_rate == ULCNET_MODEL_IO_SR);
    CHECK(d8.fft_size == ULCNET_MODEL_IO_N_FFT);
    CHECK(d8.hop_size == ULCNET_MODEL_IO_HOP);
    CHECK(d8.spectrum_bins == ULCNET_MODEL_IO_BINS);
    CHECK(d8.ta_bins == ULCNET_MODEL_IO_TA_BINS);

    invalid = d8;
    ++invalid.layout_version;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);
    invalid = d8;
    ++invalid.ta_bins;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);

    /* Production descriptors are fixed to the raw far used for training. */
    CHECK(d4.far_input_mode == ULCNET_FAR_RAW);
    CHECK(d8.far_input_mode == ULCNET_FAR_RAW);
    CHECK(d64.far_input_mode == ULCNET_FAR_RAW);
    invalid = d8;
    invalid.far_input_mode = ULCNET_FAR_ALIGNED;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);
    /* The mode does not change the state size -- it selects which far
     * stream the caller feeds, not how much history is stored. */
    CHECK(ulcnet_model_io_get_mem_requirements(&d8, &r8b) == 0);
    CHECK(r8b.bytes == r8.bytes);
    invalid.far_input_mode = 2;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);
    invalid.far_input_mode = -1;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);
    CHECK(ulcnet_model_io_get_mem_requirements(&invalid, &r8b) != 0);
    CHECK(ulcnet_model_io_init(pool, sizeof(pool), &invalid) == NULL);

    /* The names are exactly the exporter's metadata strings. */
    CHECK(strcmp(ulcnet_far_input_mode_name(ULCNET_FAR_RAW),
                 "raw_far") == 0);
    CHECK(strcmp(ulcnet_far_input_mode_name(ULCNET_FAR_ALIGNED),
                 "aligned_far") == 0);
    CHECK(strcmp(ulcnet_far_input_mode_name(2), "unknown") == 0);
    CHECK(strcmp(ulcnet_far_input_mode_name(-1), "unknown") == 0);

    CHECK(ulcnet_model_io_init(pool + 1, sizeof(pool) - 1, &d8) == NULL);
    CHECK(ulcnet_model_io_init(pool, r8.bytes - 1, &d8) == NULL);
    state = ulcnet_model_io_init(pool, r8.bytes, &d8);
    CHECK(state != NULL);
    CHECK(ulcnet_model_io_descriptor(state)->delay_depth == 8);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);

    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.spectrum_bins_elements == ULCNET_MODEL_IO_BINS);
    CHECK(outputs.spectrum_ri_elements == 2u * ULCNET_MODEL_IO_BINS);
    {
        /* prepare() runs the fixed front end (layout v5): the same fp32
         * expressions are evaluated here so the values are pinned exactly. */
        float c_re = signed_power_ref(error_re[17],
                                      ULCNET_MODEL_IO_COMPRESSION_EXP);
        float c_im = signed_power_ref(error_im[17],
                                      ULCNET_MODEL_IO_COMPRESSION_EXP);
        float f_re = signed_power_ref(far_re[31],
                                      ULCNET_MODEL_IO_COMPRESSION_EXP);
        float f_im = signed_power_ref(far_im[31],
                                      ULCNET_MODEL_IO_COMPRESSION_EXP);
        /* cos/sin are checked against the torch-reference formulation
         * (cos/sin of atan2); the implementation uses the mathematically
         * equal normalized direction, so a small tolerance makes this a
         * cross-formulation agreement gate, not an identity. */
        float phase = atan2f(c_im, c_re);
        CHECK(inputs.error_mag[17] ==
              sqrtf(c_re * c_re + c_im * c_im + 1e-12f));
        CHECK(inputs.far_mag[31] ==
              sqrtf(f_re * f_re + f_im * f_im + 1e-12f));
        CHECK(fabsf(inputs.error_cos[17] - cosf(phase)) <= 1e-6f);
        CHECK(fabsf(inputs.error_sin[17] - sinf(phase)) <= 1e-6f);
        /* Zero-vector convention pins atan2(0,0) == 0: bin 0 carries
         * error_re = error_im = 0. */
        CHECK(inputs.error_cos[0] == 1.0f);
        CHECK(inputs.error_sin[0] == 0.0f);
    }
    CHECK(inputs.key_history_elements ==
          32u * 7u * ULCNET_MODEL_IO_TA_BINS);
    CHECK(inputs.logit_history_elements == 32u * 4u * 8u);
    CHECK(outputs.key_history_elements == inputs.key_history_elements);
    CHECK(outputs.value_history_elements == inputs.value_history_elements);
    CHECK(outputs.logit_history_elements == inputs.logit_history_elements);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.h_gru0, inputs.gru_hidden_elements));
    CHECK(isnan(outputs.output[0]));
    /* Every state tensor is in-place: each output IS its input, at an
     * address that never moves, and prepare() does not touch the contents. */
    CHECK(outputs.key_history_out == inputs.key_history);
    CHECK(outputs.value_history_out == inputs.value_history);
    CHECK(outputs.logit_history_out == inputs.logit_history);
    CHECK(outputs.h_gru0_out == inputs.h_gru0);
    CHECK(outputs.h_gru1_out == inputs.h_gru1);
    CHECK(all_zero(inputs.h_gru1, inputs.gru_hidden_elements));
    gru0_address = inputs.h_gru0;
    gru1_address = inputs.h_gru1;

    write_outputs(&outputs, 1.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    /* The compressed error is not handed to the graph; prepare() keeps it for
     * commit(), which multiplies it by the planar mask, then
     * applies the inverse signed power. The reference spells the complex
     * multiply out with unfused fp32 products, as export_onnx.apply_mask_ri
     * does, so every bin must match bit for bit. */
    for (index = 0; index < ULCNET_MODEL_IO_BINS; ++index) {
        const float c_re = signed_power_ref(
            error_re[index], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float c_im = signed_power_ref(
            error_im[index], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float m_re = mask_re(1.0f, index);
        const float m_im = mask_im(1.0f, index);
        const float p_rr = c_re * m_re, p_ii = c_im * m_im;
        const float p_ri = c_re * m_im, p_ir = c_im * m_re;
        const float inverse = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
        CHECK(enhanced_re[index] == signed_power_ref(p_rr - p_ii, inverse));
        CHECK(enhanced_im[index] == signed_power_ref(p_ri + p_ir, inverse));
    }
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == 1.0f);
    CHECK(inputs.key_history[feature] == 0.0f);
    CHECK(inputs.value_history[0] == 10001.0f);
    /* Logits are chronological: t-4,t-3,t-2,t-1. */
    CHECK(inputs.logit_history[0] == 0.0f);
    CHECK(inputs.logit_history[3u * logit_frame] == 20001.0f);
    CHECK(inputs.h_gru0[0] == 30001.0f);
    CHECK(inputs.h_gru1[0] == 40001.0f);
    CHECK(inputs.h_gru0 == gru0_address && inputs.h_gru1 == gru1_address);

    /* A partial estimate (one element written, the rest still NaN) is
     * refused: the state was already written in place, so every recurrent
     * tensor restarts from zero, and the caller's outputs stay untouched. */
    outputs.output[0] = 7.0f;
    enhanced_re[0] = -7.0f;
    enhanced_im[0] = -8.0f;
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
    CHECK(enhanced_re[0] == -7.0f && enhanced_im[0] == -8.0f);
    /* One prepare permits one commit attempt. */
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.h_gru0, inputs.gru_hidden_elements));
    CHECK(all_zero(inputs.h_gru1, inputs.gru_hidden_elements));

    write_outputs(&outputs, 2.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    write_outputs(&outputs, 3.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == 3.0f);
    CHECK(inputs.key_history[feature] == 2.0f);
    CHECK(inputs.key_history[2u * feature] == 0.0f);
    CHECK(inputs.logit_history[2u * logit_frame] == 20002.0f);
    CHECK(inputs.logit_history[3u * logit_frame] == 20003.0f);
    CHECK(inputs.h_gru0[0] == 30003.0f);
    CHECK(inputs.h_gru0 == gru0_address && inputs.h_gru1 == gru1_address);

    /* A non-finite value in any ONE tensor the graph just wrote, with every
     * other output healthy, is refused and takes the rest of the state with
     * it: the state cannot be rolled back, so nothing recurrent may survive
     * the frame. Newest key/value slot, newest logit frame, both hiddens and
     * the estimate each get their own run. */
    {
        const size_t key_stride = 7u * feature;
        int which;
        for (which = 0; which < 6; ++which) {
            CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re,
                                          far_im, &inputs, &outputs) == 0);
            write_outputs(&outputs, 4.0f);
            switch (which) {
            case 0: outputs.key_history_out[0] = NAN; break;
            case 1: outputs.value_history_out[31u * key_stride] = INFINITY;
                    break;
            case 2: outputs.logit_history_out[31u * 4u * 8u + 3u * 8u + 7u] =
                        NAN;
                    break;
            case 3: outputs.h_gru0_out[0] = NAN; break;
            case 4: outputs.h_gru1_out[outputs.gru_hidden_elements - 1u] =
                        INFINITY;
                    break;
            default: outputs.output[outputs.spectrum_ri_elements - 1u] = NAN;
                     break;
            }
            enhanced_re[0] = -7.0f;
            enhanced_im[0] = -8.0f;
            CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
            CHECK(enhanced_re[0] == -7.0f && enhanced_im[0] == -8.0f);
            CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re,
                                          far_im, &inputs, &outputs) == 0);
            CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
            CHECK(all_zero(inputs.value_history,
                           inputs.value_history_elements));
            CHECK(all_zero(inputs.logit_history,
                           inputs.logit_history_elements));
            CHECK(all_zero(inputs.h_gru0, inputs.gru_hidden_elements));
            CHECK(all_zero(inputs.h_gru1, inputs.gru_hidden_elements));
            /* The state keeps working after a refusal. */
            write_outputs(&outputs, 5.0f);
            CHECK(ulcnet_model_io_commit(state, enhanced_re,
                                         enhanced_im) == 0);
        }
    }
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == 5.0f);
    CHECK(inputs.h_gru0[0] == 30005.0f);

    /* Only the frame the graph just wrote is checked at commit. A
     * non-finite value left in an OLDER ring slot is not seen here; it
     * reaches the estimate on the next frame, where it is refused. */
    write_outputs(&outputs, 6.0f);
    outputs.key_history_out[3u * feature] = NAN;
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);

    ulcnet_model_io_reset(state);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.h_gru0, inputs.gru_hidden_elements));
    CHECK(all_zero(inputs.h_gru1, inputs.gru_hidden_elements));

    /* The copy path: a runtime with its own output tensors hands them to
     * ulcnet_model_io_inherit().  Over a stream it must leave exactly the
     * state the in-place binding leaves. */
    {
        static float p_output[2u * ULCNET_MODEL_IO_BINS];
        static float p_key[32u * 7u * ULCNET_MODEL_IO_TA_BINS];
        static float p_value[32u * 7u * ULCNET_MODEL_IO_TA_BINS];
        static float p_logit[32u * 4u * 8u];
        static float p_gru0[256];
        static float p_gru1[256];
        UlcnetModelIoState *copy_state;
        UlcnetModelIoInputs copy_inputs;
        UlcnetModelIoOutputs copy_outputs, mine;
        float copy_re[ULCNET_MODEL_IO_BINS], copy_im[ULCNET_MODEL_IO_BINS];
        float base;

        ulcnet_model_io_reset(state);
        copy_state = ulcnet_model_io_init(pool_copy, sizeof(pool_copy), &d8);
        CHECK(copy_state != NULL);
        for (base = 1.0f; base <= 6.0f; base += 1.0f) {
            CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re,
                                          far_im, &inputs, &outputs) == 0);
            write_outputs(&outputs, base);
            CHECK(ulcnet_model_io_commit(state, enhanced_re,
                                         enhanced_im) == 0);

            CHECK(ulcnet_model_io_prepare(copy_state, error_re, error_im,
                                          far_re, far_im, &copy_inputs,
                                          &copy_outputs) == 0);
            /* The runtime's tensors start as copies of the inputs it was
             * handed, then it computes its outputs into them. */
            memcpy(p_key, copy_inputs.key_history, sizeof(p_key));
            memcpy(p_value, copy_inputs.value_history, sizeof(p_value));
            memcpy(p_logit, copy_inputs.logit_history, sizeof(p_logit));
            mine = copy_outputs;
            mine.output = p_output;
            mine.key_history_out = p_key;
            mine.value_history_out = p_value;
            mine.logit_history_out = p_logit;
            mine.h_gru0_out = p_gru0;
            mine.h_gru1_out = p_gru1;
            write_outputs(&mine, base);
            CHECK(ulcnet_model_io_inherit(&copy_outputs, &mine) == 0);
            CHECK(ulcnet_model_io_commit(copy_state, copy_re, copy_im) == 0);
            CHECK(memcmp(enhanced_re, copy_re, sizeof(copy_re)) == 0);
            CHECK(memcmp(enhanced_im, copy_im, sizeof(copy_im)) == 0);
        }
        CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re,
                                      far_im, &inputs, &outputs) == 0);
        CHECK(ulcnet_model_io_prepare(copy_state, error_re, error_im, far_re,
                                      far_im, &copy_inputs,
                                      &copy_outputs) == 0);
        CHECK(memcmp(inputs.key_history, copy_inputs.key_history,
                     sizeof(p_key)) == 0);
        CHECK(memcmp(inputs.value_history, copy_inputs.value_history,
                     sizeof(p_value)) == 0);
        CHECK(memcmp(inputs.logit_history, copy_inputs.logit_history,
                     sizeof(p_logit)) == 0);
        CHECK(memcmp(inputs.h_gru0, copy_inputs.h_gru0, sizeof(p_gru0)) == 0);
        CHECK(memcmp(inputs.h_gru1, copy_inputs.h_gru1, sizeof(p_gru1)) == 0);
        CHECK(!all_zero(copy_inputs.key_history,
                        copy_inputs.key_history_elements));

        /* A runtime that wrote everything in place is not copied over
         * itself. */
        CHECK(ulcnet_model_io_inherit(&copy_outputs, &copy_outputs) == 0);
        CHECK(memcmp(inputs.key_history, copy_inputs.key_history,
                     sizeof(p_key)) == 0);

        /* One non-finite value in one runtime tensor: nothing is written and
         * the state is exactly what it was. Newest key/value slot, newest
         * logit frame, both hiddens and the estimate each get a run. */
        {
            int which;
            for (which = 0; which < 6; ++which) {
                mine = copy_outputs;
                mine.output = p_output;
                mine.key_history_out = p_key;
                mine.value_history_out = p_value;
                mine.logit_history_out = p_logit;
                mine.h_gru0_out = p_gru0;
                mine.h_gru1_out = p_gru1;
                memcpy(p_key, copy_inputs.key_history, sizeof(p_key));
                memcpy(p_value, copy_inputs.value_history, sizeof(p_value));
                memcpy(p_logit, copy_inputs.logit_history, sizeof(p_logit));
                write_outputs(&mine, 7.0f);
                switch (which) {
                case 0: p_key[0] = NAN; break;
                case 1: p_value[31u * 7u * ULCNET_MODEL_IO_TA_BINS] = INFINITY;
                        break;
                case 2: p_logit[31u * 4u * 8u + 3u * 8u + 7u] = NAN; break;
                case 3: p_gru0[0] = NAN; break;
                case 4: p_gru1[255] = INFINITY; break;
                default: p_output[2u * ULCNET_MODEL_IO_BINS - 1u] = NAN; break;
                }
                CHECK(ulcnet_model_io_inherit(&copy_outputs, &mine) != 0);
                CHECK(memcmp(inputs.key_history, copy_inputs.key_history,
                             sizeof(p_key)) == 0);
                CHECK(memcmp(inputs.value_history,
                             copy_inputs.value_history, sizeof(p_value)) == 0);
                CHECK(memcmp(inputs.logit_history,
                             copy_inputs.logit_history, sizeof(p_logit)) == 0);
                CHECK(memcmp(inputs.h_gru0, copy_inputs.h_gru0,
                             sizeof(p_gru0)) == 0);
                CHECK(memcmp(inputs.h_gru1, copy_inputs.h_gru1,
                             sizeof(p_gru1)) == 0);
            }
        }

        /* Malformed calls are refused, not half-applied. */
        mine = copy_outputs;
        mine.output = p_output;
        mine.key_history_out = p_key;
        mine.value_history_out = p_value;
        mine.logit_history_out = p_logit;
        mine.h_gru0_out = p_gru0;
        mine.h_gru1_out = p_gru1;
        write_outputs(&mine, 8.0f);
        CHECK(ulcnet_model_io_inherit(NULL, &mine) == -1);
        CHECK(ulcnet_model_io_inherit(&copy_outputs, NULL) == -1);
        {
            float **const runtime_fields[] = {
                &mine.output, &mine.key_history_out, &mine.value_history_out,
                &mine.logit_history_out, &mine.h_gru0_out, &mine.h_gru1_out};
            float **const destination_fields[] = {
                &copy_outputs.output, &copy_outputs.key_history_out,
                &copy_outputs.value_history_out,
                &copy_outputs.logit_history_out, &copy_outputs.h_gru0_out,
                &copy_outputs.h_gru1_out};
            size_t *const counts[] = {
                &mine.spectrum_ri_elements, &mine.key_history_elements,
                &mine.value_history_elements, &mine.logit_history_elements,
                &mine.gru_hidden_elements};
            size_t field;
            for (field = 0; field < 6u; ++field) {
                float *const kept_runtime = *runtime_fields[field];
                float *const kept_destination = *destination_fields[field];
                *runtime_fields[field] = NULL;
                CHECK(ulcnet_model_io_inherit(&copy_outputs, &mine) == -1);
                *runtime_fields[field] = kept_runtime;
                *destination_fields[field] = NULL;
                CHECK(ulcnet_model_io_inherit(&copy_outputs, &mine) == -1);
                *destination_fields[field] = kept_destination;
            }
            for (field = 0; field < 5u; ++field) {
                ++*counts[field];
                CHECK(ulcnet_model_io_inherit(&copy_outputs, &mine) == -1);
                --*counts[field];
            }
        }
        CHECK(ulcnet_model_io_inherit(&copy_outputs, &mine) == 0);
    }
    return 0;
}
'''


@pytest.mark.parametrize('sample_rate,n_fft', [(16000, 512), (48000, 1024)])
def test_ulcnet_model_io_external_state_contract(
        tmp_path, sample_rate, n_fft):
    compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if compiler is None:
        pytest.skip('no C compiler available')
    driver = tmp_path / 'driver.c'
    executable = tmp_path / 'driver'
    driver.write_text(_DRIVER, encoding='utf-8')
    subprocess.run([
        compiler,
        '-O2', '-ffp-contract=off', '-std=c11', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
        '-DULCNET_MODEL_IO_SR=%d' % sample_rate,
        '-DULCNET_MODEL_IO_N_FFT=%d' % n_fft,
        '-I', _ULCNET_DIR,
        str(driver), os.path.join(_ULCNET_DIR, 'ulcnet_model_io.c'),
        '-lm', '-o', str(executable),
    ], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
