"""C contract tests for Align-ULCNet external delta-state storage."""

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

static int all_zero(const float *values, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (values[index] != 0.0f) return 0;
    return 1;
}

static void write_outputs(UlcnetModelIoOutputs *outputs, float base) {
    size_t index;
    for (index = 0; index < outputs->spectrum_ri_elements; ++index)
        outputs->enhanced_ri[index] = base + 50000.0f + (float)index;
    for (index = 0; index < outputs->key_now_elements; ++index)
        outputs->key_now[index] = base + (float)index;
    for (index = 0; index < outputs->value_now_elements; ++index)
        outputs->value_now[index] = base + 10000.0f + (float)index;
    for (index = 0; index < outputs->logit_now_elements; ++index)
        outputs->logit_now[index] = base + 20000.0f + (float)index;
    for (index = 0; index < outputs->gru_hidden_elements; ++index) {
        outputs->gru0_hidden_next[index] = base + 30000.0f + (float)index;
        outputs->gru1_hidden_next[index] = base + 40000.0f + (float)index;
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
    float error_re[257], error_im[257], far_re[257], far_im[257];
    float enhanced_re[257], enhanced_im[257];
    size_t index;

    for (index = 0; index < 257u; ++index) {
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

    invalid = d8;
    ++invalid.layout_version;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);
    invalid = d8;
    ++invalid.ta_bins;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) != 0);

    /* far_input_mode: the checkpoint contract carried in the descriptor.
     * Defaults to RAW (what every current checkpoint trains on), accepts
     * ALIGNED, rejects anything else. */
    CHECK(d4.far_input_mode == ULCNET_FAR_RAW);
    CHECK(d8.far_input_mode == ULCNET_FAR_RAW);
    CHECK(d64.far_input_mode == ULCNET_FAR_RAW);
    invalid = d8;
    invalid.far_input_mode = ULCNET_FAR_ALIGNED;
    CHECK(ulcnet_model_io_descriptor_validate(&invalid) == 0);
    /* The mode does not change the state size -- it selects which far
     * stream the caller feeds, not how much history is stored. */
    CHECK(ulcnet_model_io_get_mem_requirements(&invalid, &r8b) == 0);
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
    CHECK(inputs.spectrum_ri_elements == 514u);
    CHECK(outputs.spectrum_ri_elements == 514u);
    CHECK(inputs.linear_error_ri[2u * 17u] == error_re[17]);
    CHECK(inputs.linear_error_ri[2u * 17u + 1u] == error_im[17]);
    CHECK(inputs.far_end_ri[2u * 31u] == far_re[31]);
    CHECK(inputs.far_end_ri[2u * 31u + 1u] == far_im[31]);
    CHECK(inputs.key_history_elements ==
          32u * 7u * ULCNET_MODEL_IO_TA_BINS);
    CHECK(inputs.logit_history_elements == 32u * 4u * 8u);
    CHECK(outputs.key_now_elements == 32u * ULCNET_MODEL_IO_TA_BINS);
    CHECK(outputs.logit_now_elements == 32u * 8u);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.gru0_hidden, inputs.gru_hidden_elements));
    CHECK(isnan(outputs.enhanced_ri[0]));
    CHECK(isnan(outputs.key_now[0]));
    CHECK(isnan(outputs.gru1_hidden_next[0]));

    write_outputs(&outputs, 1.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    CHECK(enhanced_re[0] == 50001.0f);
    CHECK(enhanced_im[0] == 50002.0f);
    CHECK(enhanced_re[256] == 50513.0f);
    CHECK(enhanced_im[256] == 50514.0f);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == 1.0f);
    CHECK(inputs.key_history[feature] == 0.0f);
    CHECK(inputs.value_history[0] == 10001.0f);
    /* Logits are chronological: t-4,t-3,t-2,t-1. */
    CHECK(inputs.logit_history[0] == 0.0f);
    CHECK(inputs.logit_history[3u * logit_frame] == 20001.0f);
    CHECK(inputs.gru0_hidden[0] == 30001.0f);
    CHECK(inputs.gru1_hidden[0] == 40001.0f);

    /* A partial accelerator write must not advance persistent state. */
    outputs.key_now[0] = 7.0f;
    enhanced_re[0] = -7.0f;
    enhanced_im[0] = -8.0f;
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
    CHECK(enhanced_re[0] == -7.0f && enhanced_im[0] == -8.0f);
    write_outputs(&outputs, 9.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == 1.0f);
    CHECK(inputs.gru0_hidden[0] == 30001.0f);

    write_outputs(&outputs, 2.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == 2.0f);
    CHECK(inputs.key_history[feature] == 1.0f);
    CHECK(inputs.logit_history[2u * logit_frame] == 20001.0f);
    CHECK(inputs.logit_history[3u * logit_frame] == 20002.0f);
    CHECK(inputs.gru0_hidden[0] == 30002.0f);

    ulcnet_model_io_reset(state);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.gru0_hidden, inputs.gru_hidden_elements));
    CHECK(all_zero(inputs.gru1_hidden, inputs.gru_hidden_elements));
    return 0;
}
'''


def test_ulcnet_model_io_external_state_contract(tmp_path):
    compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if compiler is None:
        pytest.skip('no C compiler available')
    driver = tmp_path / 'driver.c'
    executable = tmp_path / 'driver'
    driver.write_text(_DRIVER, encoding='utf-8')
    subprocess.run([
        compiler,
        '-O2', '-std=c11', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
        '-I', _ULCNET_DIR,
        str(driver), os.path.join(_ULCNET_DIR, 'ulcnet_model_io.c'),
        '-lm', '-o', str(executable),
    ], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
