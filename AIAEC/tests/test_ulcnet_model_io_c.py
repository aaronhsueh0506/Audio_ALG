"""C contract tests for Align-ULCNet in-place full-window state storage."""

import os
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ULCNET_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'Align_ULCNet')
_AC_DIR = os.path.abspath(os.path.join(
    _THIS_DIR, '..', '..', '..', 'audiolib', 'common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

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

#define DEPTH 8u
#define FEATURE ((size_t)ULCNET_MODEL_IO_TA_BINS)
#define CHANNELS 32u
#define SCORE 5u
#define MAX_HOPS 64u

_Alignas(16) static unsigned char pool[1024 * 1024];

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

/* The planar mask the fake accelerator writes: plane 0 real, plane 1
 * imaginary. Dyadic steps keep the values exact; both planes change with
 * `base`, so two frames never share a mask. */
static float mask_re(float base, size_t bin) {
    return 0.5f + 0.0625f * base + 0.0078125f * (float)bin;
}

static float mask_im(float base, size_t bin) {
    return -0.25f + 0.03125f * base + 0.00390625f * (float)bin;
}

/* What the graph computes for one hop, as pure functions of the hop number
 * `base` (1, 2, ...): the new K/V feature, the new logit row, the hidden. */
static float key_now(float base, size_t channel, size_t bin) {
    return base + 100.0f * (float)channel + (float)bin;
}
static float value_now(float base, size_t channel, size_t bin) {
    return base + 10000.0f + 100.0f * (float)channel + (float)bin;
}
static float logit_now(float base, size_t channel, size_t slot) {
    return base + 20000.0f + 100.0f * (float)channel + (float)slot;
}
static float hidden_value(float base, size_t index) {
    return base + 30000.0f + (float)index;
}

/* Stand-in for the accelerator, bound the way the board binds it: every
 * state output IS the memory of its input. It first reads all state inputs
 * into private copies, then computes the next window of each state exactly
 * as the graph does, and only then writes them back:
 *   K/V      X = cat(new, W[:, :, 0:D-1])    newest frame first
 *   logit    X = cat(W[:, :, 1:5], new)      oldest frame first
 *   h_gru    replaced
 * The expectation below does not use this shift code; see expected_*(). */
static void fake_run(const UlcnetModelIoInputs *in,
                     const UlcnetModelIoOutputs *out, float base) {
    static float key_in[CHANNELS * DEPTH * 52u];
    static float value_in[CHANNELS * DEPTH * 52u];
    static float logit_in[CHANNELS * SCORE * DEPTH];
    size_t channel, slot, bin, index;

    memcpy(key_in, in->key_history, in->key_history_elements * sizeof(float));
    memcpy(value_in, in->value_history,
           in->value_history_elements * sizeof(float));
    memcpy(logit_in, in->logit_history,
           in->logit_history_elements * sizeof(float));

    for (index = 0; index < ULCNET_MODEL_IO_BINS; ++index) {
        out->output[index] = mask_re(base, index);
        out->output[ULCNET_MODEL_IO_BINS + index] = mask_im(base, index);
    }
    for (channel = 0; channel < CHANNELS; ++channel) {
        float *key = out->key_history_out + channel * DEPTH * FEATURE;
        float *value = out->value_history_out + channel * DEPTH * FEATURE;
        float *logit = out->logit_history_out + channel * SCORE * DEPTH;
        const float *key_old = key_in + channel * DEPTH * FEATURE;
        const float *value_old = value_in + channel * DEPTH * FEATURE;
        const float *logit_old = logit_in + channel * SCORE * DEPTH;
        for (bin = 0; bin < FEATURE; ++bin) {
            key[bin] = key_now(base, channel, bin);
            value[bin] = value_now(base, channel, bin);
        }
        memcpy(key + FEATURE, key_old, (DEPTH - 1u) * FEATURE * sizeof(float));
        memcpy(value + FEATURE, value_old,
               (DEPTH - 1u) * FEATURE * sizeof(float));
        memcpy(logit, logit_old + DEPTH, (SCORE - 1u) * DEPTH * sizeof(float));
        for (slot = 0; slot < DEPTH; ++slot)
            logit[(SCORE - 1u) * DEPTH + slot] = logit_now(base, channel, slot);
    }
    for (index = 0; index < out->gru_hidden_elements; ++index)
        out->h_gru_out[index] = hidden_value(base, index);
}

/* Window contents by direct indexing on the hop number: slot d of a K/V
 * window after hop `hop` holds the feature of hop `hop - d` (zero before
 * hop 1); frame f of the logit window holds the row of hop `hop - 4 + f`. */
static float expected_key(size_t hop, size_t channel, size_t slot, size_t bin) {
    return hop > slot ? key_now((float)(hop - slot), channel, bin) : 0.0f;
}
static float expected_value(size_t hop, size_t channel, size_t slot,
                            size_t bin) {
    return hop > slot ? value_now((float)(hop - slot), channel, bin) : 0.0f;
}
static float expected_logit(size_t hop, size_t channel, size_t frame,
                            size_t slot) {
    size_t back = SCORE - 1u - frame;
    return hop > back ? logit_now((float)(hop - back), channel, slot) : 0.0f;
}

static int windows_match_after(const UlcnetModelIoInputs *in, size_t hop) {
    size_t channel, slot, bin, frame;
    for (channel = 0; channel < CHANNELS; ++channel) {
        for (slot = 0; slot < DEPTH; ++slot) {
            for (bin = 0; bin < FEATURE; ++bin) {
                size_t at = (channel * DEPTH + slot) * FEATURE + bin;
                if (in->key_history[at] != expected_key(hop, channel, slot, bin))
                    return 0;
                if (in->value_history[at] !=
                    expected_value(hop, channel, slot, bin))
                    return 0;
            }
        }
        for (frame = 0; frame < SCORE; ++frame) {
            for (slot = 0; slot < DEPTH; ++slot) {
                size_t at = (channel * SCORE + frame) * DEPTH + slot;
                if (in->logit_history[at] !=
                    expected_logit(hop, channel, frame, slot))
                    return 0;
            }
        }
    }
    return hop == 0u ? all_zero(in->h_gru, in->gru_hidden_elements)
                     : in->h_gru[0] == hidden_value((float)hop, 0u);
}

int main(void) {
    UlcnetModelIoDescriptor d4, d8, d64, invalid;
    UlcnetModelIoMemReq r4, r8, r64, r8b;
    UlcnetModelIoState *state;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;
    float error_re[ULCNET_MODEL_IO_BINS], error_im[ULCNET_MODEL_IO_BINS];
    float far_re[ULCNET_MODEL_IO_BINS], far_im[ULCNET_MODEL_IO_BINS];
    float enhanced_re[ULCNET_MODEL_IO_BINS];
    float enhanced_im[ULCNET_MODEL_IO_BINS];
    const float *key_address, *value_address, *logit_address, *gru_address;
    size_t index;
    size_t hop;
    uint32_t generation;

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
    CHECK(d8.layout_version == 18u);
    CHECK(d8.sample_rate == ULCNET_MODEL_IO_SR);
    CHECK(d8.fft_size == ULCNET_MODEL_IO_N_FFT);
    CHECK(d8.hop_size == ULCNET_MODEL_IO_HOP);
    CHECK(d8.spectrum_bins == ULCNET_MODEL_IO_BINS);
    CHECK(d8.ta_bins == ULCNET_MODEL_IO_TA_BINS);
    CHECK(d8.score_window_frames == 5);

    invalid = d8;
    ++invalid.layout_version;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);
    invalid = d8;
    ++invalid.ta_bins;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);
    invalid = d8;
    --invalid.score_window_frames;
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

    /* A fresh state has restarted nothing. */
    CHECK(ulcnet_model_io_state_generation(state) == 0u);
    CHECK(ulcnet_model_io_state_generation(NULL) == 0u);

    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.spectrum_bins_elements == ULCNET_MODEL_IO_BINS);
    CHECK(outputs.spectrum_ri_elements == 2u * ULCNET_MODEL_IO_BINS);
    {
        /* prepare() runs the fixed front end: the same fp32 expressions are
         * evaluated here so the values are pinned exactly. */
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
    /* Full windows: D frames of K/V, 5 of logit, and ONE hidden tensor. */
    CHECK(inputs.key_history_elements == CHANNELS * DEPTH * FEATURE);
    CHECK(inputs.value_history_elements == CHANNELS * DEPTH * FEATURE);
    CHECK(inputs.logit_history_elements == CHANNELS * SCORE * DEPTH);
    CHECK(inputs.gru_hidden_elements == 4u * 128u);
    CHECK(outputs.key_history_elements == inputs.key_history_elements);
    CHECK(outputs.value_history_elements == inputs.value_history_elements);
    CHECK(outputs.logit_history_elements == inputs.logit_history_elements);
    CHECK(outputs.gru_hidden_elements == inputs.gru_hidden_elements);
    CHECK(windows_match_after(&inputs, 0u));
    CHECK(isnan(outputs.output[0]));
    /* Every state tensor is in-place: each output IS its input, at an
     * address that never moves, and prepare() does not touch the contents. */
    CHECK(outputs.key_history_out == inputs.key_history);
    CHECK(outputs.value_history_out == inputs.value_history);
    CHECK(outputs.logit_history_out == inputs.logit_history);
    CHECK(outputs.h_gru_out == inputs.h_gru);
    key_address = inputs.key_history;
    value_address = inputs.value_history;
    logit_address = inputs.logit_history;
    gru_address = inputs.h_gru;

    /* Over a stream longer than every window, the in-place runtime leaves
     * exactly the windows direct indexing predicts -- K/V newest first, logit
     * oldest first -- at the same addresses. */
    for (hop = 1u; hop <= 3u * DEPTH; ++hop) {
        float base = (float)hop;
        CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re,
                                      far_im, &inputs, &outputs) == 0);
        CHECK(windows_match_after(&inputs, hop - 1u));
        fake_run(&inputs, &outputs, base);
        enhanced_re[0] = -7.0f;
        CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
        /* The compressed error is not handed to the graph; prepare() keeps
         * it for commit(), which multiplies it by the planar mask, then
         * applies the inverse signed power. The reference spells the complex
         * multiply out with unfused fp32 products, as
         * export_onnx.apply_mask_ri does, so every bin must match bit for
         * bit. */
        for (index = 0; index < ULCNET_MODEL_IO_BINS; ++index) {
            const float c_re = signed_power_ref(
                error_re[index], ULCNET_MODEL_IO_COMPRESSION_EXP);
            const float c_im = signed_power_ref(
                error_im[index], ULCNET_MODEL_IO_COMPRESSION_EXP);
            const float m_re = mask_re(base, index);
            const float m_im = mask_im(base, index);
            const float p_rr = c_re * m_re, p_ii = c_im * m_im;
            const float p_ri = c_re * m_im, p_ir = c_im * m_re;
            const float inverse = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
            CHECK(enhanced_re[index] ==
                  signed_power_ref(p_rr - p_ii, inverse));
            CHECK(enhanced_im[index] ==
                  signed_power_ref(p_ri + p_ir, inverse));
        }
    }
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(windows_match_after(&inputs, 3u * DEPTH));
    CHECK(inputs.key_history == key_address &&
          inputs.value_history == value_address &&
          inputs.logit_history == logit_address && inputs.h_gru == gru_address);
    /* Newest frame first / oldest frame first, spelled out for one channel. */
    CHECK(inputs.key_history[0] == key_now(24.0f, 0u, 0u));
    CHECK(inputs.key_history[FEATURE] == key_now(23.0f, 0u, 0u));
    CHECK(inputs.key_history[(DEPTH - 1u) * FEATURE] ==
          key_now(17.0f, 0u, 0u));
    CHECK(inputs.logit_history[0] == logit_now(20.0f, 0u, 0u));
    CHECK(inputs.logit_history[(SCORE - 1u) * DEPTH] ==
          logit_now(24.0f, 0u, 0u));
    CHECK(ulcnet_model_io_state_generation(state) == 0u);

    /* A partial mask (one element written, the rest still NaN) is
     * refused: the state was already written in place, so every recurrent
     * tensor restarts from zero, and the caller's outputs stay untouched. */
    outputs.output[0] = 7.0f;
    enhanced_re[0] = -7.0f;
    enhanced_im[0] = -8.0f;
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
    CHECK(enhanced_re[0] == -7.0f && enhanced_im[0] == -8.0f);
    CHECK(ulcnet_model_io_state_generation(state) == 1u);
    /* One prepare permits one commit attempt; the refused retry does not
     * count as another restart. */
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
    CHECK(ulcnet_model_io_state_generation(state) == 1u);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.h_gru, inputs.gru_hidden_elements));
    CHECK(ulcnet_model_io_state_generation(state) == 1u);

    /* A non-finite value in any ONE tensor the graph just wrote, with every
     * other output healthy, is refused and takes the rest of the state with
     * it: the state cannot be rolled back, so nothing recurrent may survive
     * the frame. Newest key/value slot, newest logit frame, the hidden and
     * the mask each get their own run, and each advances the generation. */
    {
        const size_t key_stride = DEPTH * FEATURE;
        int which;
        for (which = 0; which < 5; ++which) {
            CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re,
                                          far_im, &inputs, &outputs) == 0);
            fake_run(&inputs, &outputs, 4.0f);
            switch (which) {
            case 0: outputs.key_history_out[2u * key_stride] = NAN; break;
            case 1: outputs.value_history_out[31u * key_stride + FEATURE - 1u] =
                        INFINITY;
                    break;
            case 2: outputs.logit_history_out[31u * SCORE * DEPTH +
                                              (SCORE - 1u) * DEPTH + 7u] = NAN;
                    break;
            case 3: outputs.h_gru_out[outputs.gru_hidden_elements - 1u] =
                        INFINITY;
                    break;
            default: outputs.output[outputs.spectrum_ri_elements - 1u] = NAN;
                     break;
            }
            enhanced_re[0] = -7.0f;
            enhanced_im[0] = -8.0f;
            generation = ulcnet_model_io_state_generation(state);
            CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
            CHECK(enhanced_re[0] == -7.0f && enhanced_im[0] == -8.0f);
            CHECK(ulcnet_model_io_state_generation(state) == generation + 1u);
            CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re,
                                          far_im, &inputs, &outputs) == 0);
            CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
            CHECK(all_zero(inputs.value_history,
                           inputs.value_history_elements));
            CHECK(all_zero(inputs.logit_history,
                           inputs.logit_history_elements));
            CHECK(all_zero(inputs.h_gru, inputs.gru_hidden_elements));
            /* The state keeps working after a refusal. */
            fake_run(&inputs, &outputs, 5.0f);
            CHECK(ulcnet_model_io_commit(state, enhanced_re,
                                         enhanced_im) == 0);
            CHECK(ulcnet_model_io_state_generation(state) == generation + 1u);
        }
    }
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == key_now(5.0f, 0u, 0u));
    CHECK(inputs.h_gru[0] == hidden_value(5.0f, 0u));

    /* Only the frame the graph just wrote is checked at commit. A
     * non-finite value in an OLDER slot or in the slot the window drops is
     * not seen here; it reaches the mask on a later frame, where it is
     * refused. */
    fake_run(&inputs, &outputs, 6.0f);
    outputs.key_history_out[3u * FEATURE] = NAN;
    outputs.key_history_out[(DEPTH - 1u) * FEATURE] = NAN;
    outputs.logit_history_out[0] = NAN;
    outputs.logit_history_out[SCORE * DEPTH - DEPTH - 1u] = NAN;
    generation = ulcnet_model_io_state_generation(state);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    CHECK(ulcnet_model_io_state_generation(state) == generation);

    /* reset() zeroes the state and advances the generation, whether or not a
     * frame is open. */
    generation = ulcnet_model_io_state_generation(state);
    ulcnet_model_io_reset(state);
    CHECK(ulcnet_model_io_state_generation(state) == generation + 1u);
    ulcnet_model_io_reset(NULL);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(windows_match_after(&inputs, 0u));
    CHECK(all_zero(inputs.h_gru, inputs.gru_hidden_elements));
    CHECK(inputs.key_history == key_address && inputs.h_gru == gru_address);
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
        '-O2', '-ffp-contract=off', '-std=c11',
        '-Wall', '-Wextra', '-Wpedantic', '-Werror',
        '-DULCNET_MODEL_IO_SR=%d' % sample_rate,
        '-DULCNET_MODEL_IO_N_FFT=%d' % n_fft,
        '-I', _ULCNET_DIR, '-I', _AC_INCLUDE, '-I', _AC_DIR,
        str(driver), os.path.join(_ULCNET_DIR, 'ulcnet_model_io.c'),
        '-lm', '-o', str(executable),
    ], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
