"""C contract tests for Align-ULCNet external delta-state storage."""

import os
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ULCNET_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'Align_ULCNet')

_DRIVER = r'''
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ulcnet_model_io.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

_Alignas(16) static unsigned char pool[1024 * 1024];

/* Independent formulation of model.py's _signed_power (deliberately NOT
 * copysignf like the implementation, so agreement is a real check). The
 * exponent itself comes from the header contract define, which the Python
 * suite pins against export_onnx.COMPRESSION_EXPONENT. */
__attribute__((unused)) static float signed_power_ref(float value, float exponent) {
    float magnitude = powf(fabsf(value), exponent);
    return value < 0.0f ? -magnitude : magnitude;
}

__attribute__((unused)) static int close_fp32(float actual, float expected) {
    float scale = fmaxf(1.0f, fabsf(expected));
    return fabsf(actual - expected) <= 2.0f * FLT_EPSILON * scale;
}

static int all_zero(const float *values, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (values[index] != 0.0f) return 0;
    return 1;
}

static void write_outputs(UlcnetModelIoOutputs *outputs, float base) {
    size_t index;
    for (index = 0; index < outputs->spectrum_ri_elements; ++index)
        outputs->output[index] = base + 50000.0f + (float)index;
    for (index = 0; index < outputs->key_now_elements; ++index)
        outputs->key_now[index] = base + (float)index;
    for (index = 0; index < outputs->value_now_elements; ++index)
        outputs->value_now[index] = base + 10000.0f + (float)index;
    for (index = 0; index < outputs->logit_now_elements; ++index)
        outputs->logit_now[index] = base + 20000.0f + (float)index;
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
    CHECK(inputs.spectrum_ri_elements == 2u * ULCNET_MODEL_IO_BINS);
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
        CHECK(inputs.error_ri[2u * 17u] == c_re);
        CHECK(inputs.error_ri[2u * 17u + 1u] == c_im);
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
    CHECK(outputs.key_now_elements == 32u * ULCNET_MODEL_IO_TA_BINS);
    CHECK(outputs.logit_now_elements == 32u * 8u);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.h_gru0, inputs.gru_hidden_elements));
    CHECK(isnan(outputs.output[0]));
    CHECK(isnan(outputs.key_now[0]));
    CHECK(isnan(outputs.h_gru1_out[0]));

    write_outputs(&outputs, 1.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    /* commit() applies the inverse signed power to the graph's compressed
     * estimate (layout v5). */
    CHECK(close_fp32(enhanced_re[0], signed_power_ref(
        50001.0f, 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP)));
    CHECK(close_fp32(enhanced_im[0], signed_power_ref(
        50002.0f, 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP)));
    CHECK(close_fp32(enhanced_re[ULCNET_MODEL_IO_BINS - 1], signed_power_ref(
        50001.0f + 2.0f * (ULCNET_MODEL_IO_BINS - 1),
        1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP)));
    CHECK(close_fp32(enhanced_im[ULCNET_MODEL_IO_BINS - 1], signed_power_ref(
        50002.0f + 2.0f * (ULCNET_MODEL_IO_BINS - 1),
        1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP)));
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
    CHECK(inputs.h_gru0[0] == 30001.0f);

    write_outputs(&outputs, 2.0f);
    CHECK(ulcnet_model_io_commit(state, enhanced_re, enhanced_im) == 0);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(inputs.key_history[0] == 2.0f);
    CHECK(inputs.key_history[feature] == 1.0f);
    CHECK(inputs.logit_history[2u * logit_frame] == 20001.0f);
    CHECK(inputs.logit_history[3u * logit_frame] == 20002.0f);
    CHECK(inputs.h_gru0[0] == 30002.0f);

    /* A graph delta is contiguous [C,1,F], but the history is [C,H,F].
     * Pin channel strides and both temporal orders, not just channel zero:
     * aliasing the output to history[0] would corrupt these positions. */
    for (size_t channel = 0; channel < 32u; ++channel) {
        for (size_t bin = 0; bin < feature; ++bin) {
            size_t now = channel * feature + bin;
            size_t history = channel * 7u * feature + bin;
            CHECK(inputs.key_history[history] == 2.0f + (float)now);
            CHECK(inputs.key_history[history + feature] == 1.0f + (float)now);
            CHECK(inputs.value_history[history] == 10002.0f + (float)now);
            CHECK(inputs.value_history[history + feature] == 10001.0f + (float)now);
        }
        for (size_t lag = 0; lag < logit_frame; ++lag) {
            size_t now = channel * logit_frame + lag;
            size_t history = channel * 4u * logit_frame + lag;
            CHECK(inputs.logit_history[history + 2u * logit_frame] == 20001.0f + (float)now);
            CHECK(inputs.logit_history[history + 3u * logit_frame] == 20002.0f + (float)now);
        }
    }

    ulcnet_model_io_reset(state);
    CHECK(ulcnet_model_io_prepare(state, error_re, error_im, far_re, far_im,
                                  &inputs, &outputs) == 0);
    CHECK(all_zero(inputs.key_history, inputs.key_history_elements));
    CHECK(all_zero(inputs.value_history, inputs.value_history_elements));
    CHECK(all_zero(inputs.logit_history, inputs.logit_history_elements));
    CHECK(all_zero(inputs.h_gru0, inputs.gru_hidden_elements));
    CHECK(all_zero(inputs.h_gru1, inputs.gru_hidden_elements));
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
        '-O2', '-std=c11', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
        '-DULCNET_MODEL_IO_SR=%d' % sample_rate,
        '-DULCNET_MODEL_IO_N_FFT=%d' % n_fft,
        '-I', _ULCNET_DIR,
        str(driver), os.path.join(_ULCNET_DIR, 'ulcnet_model_io.c'),
        '-lm', '-o', str(executable),
    ], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)


# Compare complete states to the legacy host-shift implementation, not just
# the enhanced head. Test D=2 as well: equal K/V output sizes do not make the
# two ABIs interchangeable (the logit outputs still differ).
_FULL_DRIVER = _DRIVER[:_DRIVER.index('int main(void)')] + r'''
_Alignas(16) static unsigned char full_pool[2 * 1024 * 1024];

static void write_full(const UlcnetModelIoInputs *in,
                       const UlcnetModelIoOutputs *delta,
                       UlcnetModelIoOutputs *full, int depth) {
    int c, t, k;
    const int f = ULCNET_MODEL_IO_TA_BINS;
    for (c = 0; c < ULCNET_MODEL_IO_TA_CHANNELS; ++c) {
        for (t = 0; t < depth - 1; ++t) {
            for (k = 0; k < f; ++k) {
                size_t dst = ((size_t)c * (depth - 1) + t) * f + k;
                size_t src = t ? dst - f : (size_t)c * f + k;
                full->key_history_out[dst] = t ? in->key_history[src] : delta->key_now[src];
                full->value_history_out[dst] = t ? in->value_history[src] : delta->value_now[src];
            }
        }
        for (t = 0; t < ULCNET_MODEL_IO_SCORE_HISTORY; ++t) {
            for (k = 0; k < depth; ++k) {
                size_t dst = ((size_t)c * ULCNET_MODEL_IO_SCORE_HISTORY + t) * depth + k;
                full->logit_history_out[dst] = t + 1 < ULCNET_MODEL_IO_SCORE_HISTORY
                    ? in->logit_history[dst + depth] : delta->logit_now[(size_t)c * depth + k];
            }
        }
    }
    memcpy(full->output, delta->output, full->spectrum_ri_elements * sizeof(float));
    memcpy(full->h_gru0_out, delta->h_gru0_out, full->gru_hidden_elements * sizeof(float));
    memcpy(full->h_gru1_out, delta->h_gru1_out, full->gru_hidden_elements * sizeof(float));
}

/* The full-history state must equal the delta-ring state, value for value. */
static int same_state(const UlcnetModelIoInputs *a, const UlcnetModelIoInputs *b) {
    return memcmp(a->key_history, b->key_history, b->key_history_elements * sizeof(float)) == 0 &&
           memcmp(a->value_history, b->value_history, b->value_history_elements * sizeof(float)) == 0 &&
           memcmp(a->logit_history, b->logit_history, b->logit_history_elements * sizeof(float)) == 0 &&
           memcmp(a->h_gru0, b->h_gru0, b->gru_hidden_elements * sizeof(float)) == 0 &&
           memcmp(a->h_gru1, b->h_gru1, b->gru_hidden_elements * sizeof(float)) == 0;
}

static int test_depth(int depth) {
    UlcnetModelIoDescriptor dd, fd;
    UlcnetModelIoMemReq dr, fr;
    UlcnetModelIoState *ds, *fs;
    UlcnetModelIoInputs di, fi, initial, previous;
    UlcnetModelIoOutputs dout, fout, previous_out;
    float input[ULCNET_MODEL_IO_BINS] = {0};
    float dre[ULCNET_MODEL_IO_BINS], dim[ULCNET_MODEL_IO_BINS];
    float fre[ULCNET_MODEL_IO_BINS], fim[ULCNET_MODEL_IO_BINS];
    int hop, corrupt;
    size_t i;
    CHECK(ulcnet_model_io_descriptor_default(depth, &dd) == 0);
    fd = dd;
    fd.layout_version = ULCNET_MODEL_IO_FULL_HISTORY_VERSION;
    CHECK(ulcnet_model_io_get_mem_requirements(&dd, &dr) == 0);
    CHECK(ulcnet_model_io_get_mem_requirements(&fd, &fr) == 0);
    CHECK(fr.bytes > dr.bytes && fr.bytes <= sizeof(full_pool));
    CHECK(ulcnet_model_io_init(full_pool, fr.bytes - 1, &fd) == NULL);
    ds = ulcnet_model_io_init(pool, sizeof(pool), &dd);
    fs = ulcnet_model_io_init(full_pool, sizeof(full_pool), &fd);
    CHECK(ds && fs);
    memset(&initial, 0, sizeof(initial));
    memset(&previous, 0, sizeof(previous));
    memset(&previous_out, 0, sizeof(previous_out));
    for (hop = 0; hop < depth + 5; ++hop) {
        CHECK(ulcnet_model_io_prepare(ds, input, input, input, input, &di, &dout) == 0);
        CHECK(ulcnet_model_io_prepare(fs, input, input, input, input, &fi, &fout) == 0);
        if (!hop) initial = fi;
        else {
            CHECK(fi.key_history == previous_out.key_history_out);
            CHECK(fi.value_history == previous_out.value_history_out);
            CHECK(fi.logit_history == previous_out.logit_history_out);
            CHECK(fi.h_gru0 == previous_out.h_gru0_out);
            CHECK(fout.key_history_out == previous.key_history);
        }
        CHECK(!dout.key_history_out && !dout.key_history_elements);
        CHECK(!fout.key_now && !fout.value_now && !fout.logit_now);
        CHECK(!fout.key_now_elements && !fout.value_now_elements && !fout.logit_now_elements);
        CHECK(fi.key_history != fout.key_history_out);
        CHECK(fi.value_history != fout.value_history_out);
        CHECK(fi.logit_history != fout.logit_history_out);
        CHECK(fout.key_history_elements == fi.key_history_elements);
        CHECK(fout.value_history_elements == fi.value_history_elements);
        CHECK(fout.logit_history_elements == fi.logit_history_elements);
        CHECK(same_state(&di, &fi));
        write_outputs(&dout, (float)hop);
        /* A partial/non-finite write to ANY output must keep all live state
         * and caller audio untouched. Last slots catch truncated bindings. */
        for (corrupt = 0; corrupt < 6; ++corrupt) {
            float *bad[] = {fout.output, fout.key_history_out, fout.value_history_out,
                            fout.logit_history_out, fout.h_gru0_out, fout.h_gru1_out};
            size_t count[] = {fout.spectrum_ri_elements, fout.key_history_elements,
                             fout.value_history_elements, fout.logit_history_elements,
                             fout.gru_hidden_elements, fout.gru_hidden_elements};
            write_full(&fi, &dout, &fout, depth);
            bad[corrupt][count[corrupt] - 1] = corrupt % 2 ? NAN : INFINITY;
            for (i = 0; i < ULCNET_MODEL_IO_BINS; ++i) fre[i] = fim[i] = -7.0f;
            CHECK(ulcnet_model_io_commit(fs, fre, fim) != 0);
            CHECK(ulcnet_model_io_commit(fs, fre, fim) != 0);
            for (i = 0; i < ULCNET_MODEL_IO_BINS; ++i) CHECK(fre[i] == -7.0f && fim[i] == -7.0f);
            previous = fi;
            CHECK(ulcnet_model_io_prepare(fs, input, input, input, input, &fi, &fout) == 0);
            CHECK(fi.key_history == previous.key_history && fi.h_gru0 == previous.h_gru0);
            CHECK(same_state(&di, &fi));
        }
        previous = fi;
        previous_out = fout;
        write_full(&fi, &dout, &fout, depth);
        CHECK(ulcnet_model_io_commit(ds, dre, dim) == 0);
        CHECK(ulcnet_model_io_commit(fs, fre, fim) == 0);
        CHECK(memcmp(dre, fre, sizeof(dre)) == 0 && memcmp(dim, fim, sizeof(dim)) == 0);
    }
    ulcnet_model_io_reset(fs); /* all tested depths are even -> odd commits */
    CHECK(ulcnet_model_io_prepare(fs, input, input, input, input, &fi, &fout) == 0);
    CHECK(fi.key_history == initial.key_history && fi.value_history == initial.value_history);
    CHECK(fi.logit_history == initial.logit_history && fi.h_gru0 == initial.h_gru0);
    CHECK(all_zero(fi.key_history, fi.key_history_elements));
    CHECK(all_zero(fi.value_history, fi.value_history_elements));
    CHECK(all_zero(fi.logit_history, fi.logit_history_elements));
    CHECK(all_zero(fi.h_gru0, fi.gru_hidden_elements));
    CHECK(all_zero(fi.h_gru1, fi.gru_hidden_elements));
    return 0;
}

int main(void) {
    CHECK(test_depth(2) == 0);
    CHECK(test_depth(8) == 0);
    CHECK(test_depth(64) == 0);
    return 0;
}
'''


@pytest.mark.parametrize('sample_rate,n_fft', [(16000, 512), (48000, 1024)])
def test_full_history_bank_swap_matches_delta_and_is_transactional(tmp_path, sample_rate, n_fft):
    cc = shutil.which('cc')
    if cc is None:
        pytest.skip('no C compiler available')
    source, binary = tmp_path / 'full.c', tmp_path / 'full'
    source.write_text(_FULL_DRIVER, encoding='utf-8')
    subprocess.run([
        cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
        '-DULCNET_MODEL_IO_SR=%d' % sample_rate,
        '-DULCNET_MODEL_IO_N_FFT=%d' % n_fft,
        '-I', _ULCNET_DIR, str(source), os.path.join(_ULCNET_DIR, 'ulcnet_model_io.c'),
        '-lm', '-o', str(binary),
    ], check=True, capture_output=True)
    done = subprocess.run([str(binary)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
