#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TestRuntime {
    int separate_buffers; /* own output tensors + inherit, not in place */
    int poison;           /* separate mode: one NaN in a state tensor   */
    int partial_write;
    int fail_run;         /* write every output, then report failure */
    int calls;
    float stamp;          /* value written into every state tensor  */
    float observed_key0;  /* inputs->key_history[0] seen on entry   */
    float observed_gru0;  /* inputs->h_gru0[0] seen on entry        */
    int ignore_refusal;   /* separate mode: report 0 after a refused inherit */
} TestRuntime;

/* Compression round trip (prepare's signed pow 0.3, commit's inverse) is
 * identity only up to fp32 powf error, so the copy-through model is checked
 * with a tolerance instead of memcmp. */
static int nearly_equal(const float *a, const float *b, size_t elements) {
    size_t index;
    for (index = 0; index < elements; ++index) {
        if (fabsf(a[index] - b[index]) > 1e-5f) return 0;
    }
    return 1;
}

static void fill(float *values, size_t elements, float value) {
    size_t index;
    for (index = 0; index < elements; ++index) values[index] = value;
}

/* The graph's output is the planar complex mask. The identity mask (real
 * plane 1, imaginary plane 0) hands the error spectrum back, so the test
 * model is a pass-through; a partial write sets only the first element. */
static void write_identity_mask(float *mask, const UlcnetModelIoInputs *inputs,
                                int partial) {
    const size_t bins = inputs->spectrum_bins_elements;
    size_t index;
    for (index = 0; index < (partial ? 1u : bins); ++index) {
        mask[index] = 1.0f;
        if (!partial) mask[bins + index] = 0.0f;
    }
}

/* The graph's ring shift for a constant frame: K/V newest-first, logit
 * history oldest-first; D = 8, TA_BINS from the compiled grid. */
static void push_stamp(const UlcnetModelIoOutputs *outputs, float stamp) {
    const size_t depth = 8u;
    const size_t ta_bins = ULCNET_MODEL_IO_TA_BINS;
    size_t channel;
    size_t index;
    for (channel = 0; channel < ULCNET_MODEL_IO_TA_CHANNELS; ++channel) {
        float *key = outputs->key_history_out +
            channel * (depth - 1u) * ta_bins;
        float *value = outputs->value_history_out +
            channel * (depth - 1u) * ta_bins;
        float *logit = outputs->logit_history_out +
            channel * ULCNET_MODEL_IO_SCORE_HISTORY * depth;
        memmove(key + ta_bins, key, (depth - 2u) * ta_bins * sizeof(float));
        memmove(value + ta_bins, value,
                (depth - 2u) * ta_bins * sizeof(float));
        memmove(logit, logit + depth, 3u * depth * sizeof(float));
        for (index = 0; index < ta_bins; ++index) {
            key[index] = stamp;
            value[index] = stamp;
        }
        for (index = 0; index < depth; ++index) logit[3u * depth + index] = stamp;
    }
}

/* The runtime's own output tensors (an ONNX-style runtime that cannot bind a
 * state output to its input's address): every state value is computed from
 * the inputs into these, then handed to ulcnet_model_io_inherit(). D = 8. */
#define PRIVATE_KEY_ELEMENTS \
    (ULCNET_MODEL_IO_TA_CHANNELS * 7u * ULCNET_MODEL_IO_TA_BINS)
#define PRIVATE_LOGIT_ELEMENTS \
    (ULCNET_MODEL_IO_TA_CHANNELS * ULCNET_MODEL_IO_SCORE_HISTORY * 8u)
#define PRIVATE_GRU_ELEMENTS \
    (ULCNET_MODEL_IO_GRU_LAYERS * ULCNET_MODEL_IO_GRU_HIDDEN)

static int run_separate(TestRuntime *runtime,
                        const UlcnetModelIoInputs *inputs,
                        UlcnetModelIoOutputs *outputs) {
    static float output[2u * ULCNET_MODEL_IO_BINS];
    static float key[PRIVATE_KEY_ELEMENTS];
    static float value[PRIVATE_KEY_ELEMENTS];
    static float logit[PRIVATE_LOGIT_ELEMENTS];
    static float gru0[PRIVATE_GRU_ELEMENTS];
    static float gru1[PRIVATE_GRU_ELEMENTS];
    const size_t ta_bins = ULCNET_MODEL_IO_TA_BINS;
    const size_t depth = 8u;
    UlcnetModelIoOutputs mine = *outputs;
    size_t channel;
    size_t index;

    /* Unwritten tensors read as NaN, like the in-place path's prefill. */
    fill(output, outputs->spectrum_ri_elements, NAN);
    write_identity_mask(output, inputs, runtime->partial_write);
    if (runtime->partial_write) {
        mine.output = output;
        mine.key_history_out = key;
        mine.value_history_out = value;
        mine.logit_history_out = logit;
        mine.h_gru0_out = gru0;
        mine.h_gru1_out = gru1;
        fill(key, PRIVATE_KEY_ELEMENTS, 0.0f);
        fill(value, PRIVATE_KEY_ELEMENTS, 0.0f);
        fill(logit, PRIVATE_LOGIT_ELEMENTS, 0.0f);
        fill(gru0, PRIVATE_GRU_ELEMENTS / 2u, 0.0f);
        fill(gru1, PRIVATE_GRU_ELEMENTS / 2u, 0.0f);
        return ulcnet_model_io_inherit(outputs, &mine);
    }
    for (channel = 0; channel < ULCNET_MODEL_IO_TA_CHANNELS; ++channel) {
        const size_t k = channel * (depth - 1u) * ta_bins;
        const size_t l = channel * ULCNET_MODEL_IO_SCORE_HISTORY * depth;
        for (index = 0; index < ta_bins; ++index) {
            key[k + index] = runtime->stamp;
            value[k + index] = runtime->stamp;
        }
        memcpy(key + k + ta_bins, inputs->key_history + k,
               (depth - 2u) * ta_bins * sizeof(float));
        memcpy(value + k + ta_bins, inputs->value_history + k,
               (depth - 2u) * ta_bins * sizeof(float));
        memcpy(logit + l, inputs->logit_history + l + depth,
               3u * depth * sizeof(float));
        for (index = 0; index < depth; ++index) {
            logit[l + 3u * depth + index] = runtime->stamp;
        }
    }
    fill(gru0, outputs->gru_hidden_elements, runtime->stamp);
    fill(gru1, outputs->gru_hidden_elements, 0.0f);
    if (runtime->poison) {
        gru1[outputs->gru_hidden_elements - 1u] = NAN;
    }
    if (runtime->fail_run) {
        return -1;  /* a run that reports failure is taken not to have written */
    }
    mine.output = output;
    mine.key_history_out = key;
    mine.value_history_out = value;
    mine.logit_history_out = logit;
    mine.h_gru0_out = gru0;
    mine.h_gru1_out = gru1;
    {
        const int status = ulcnet_model_io_inherit(outputs, &mine);
        return runtime->ignore_refusal ? 0 : status;
    }
}

static int run(void *user, const UlcnetModelIoInputs *inputs,
               UlcnetModelIoOutputs *outputs) {
    TestRuntime *runtime = (TestRuntime *)user;
    size_t index;
    runtime->calls += 1;
    runtime->observed_key0 = inputs->key_history[0];
    runtime->observed_gru0 = inputs->h_gru0[0];
    if (runtime->separate_buffers) {
        return run_separate(runtime, inputs, outputs);
    }
    /* A partial write leaves all but the first estimate element at NaN. */
    write_identity_mask(outputs->output, inputs, runtime->partial_write);
    if (runtime->partial_write) {
        return 0;
    }
    /* The state tensors are in place: the newest K/V slot of every channel
     * gets the stamp and the older slots shift by one, exactly as the graph
     * shifts them; every input was read above. */
    push_stamp(outputs, runtime->stamp);
    fill(outputs->h_gru0_out, outputs->gru_hidden_elements,
         runtime->stamp);
    fill(outputs->h_gru1_out, outputs->gru_hidden_elements, 0.0f);
    return runtime->fail_run ? -1 : 0;
}

/* One full pass of the adapter contract; the same assertions hold for both
 * state bindings, except where a run that reports failure is concerned. */
static int exercise(int separate_buffers) {
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModel model;
    TestRuntime runtime = {0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f};
    void *pool = NULL;
    size_t bytes;
    size_t alignment;
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoDescriptor invalid_descriptor;
    /* Sized from the compiled grid, not from 257: the adapter reads and
     * writes descriptor->spectrum_bins floats through these, so a fixed
     * 16 kHz width silently overflows the stack on the 48 kHz build. */
    float error_re[ULCNET_MODEL_IO_BINS];
    float error_im[ULCNET_MODEL_IO_BINS];
    float far_re[ULCNET_MODEL_IO_BINS] = {0};
    float far_im[ULCNET_MODEL_IO_BINS] = {0};
    float output_re[ULCNET_MODEL_IO_BINS];
    float output_im[ULCNET_MODEL_IO_BINS];
    int bin;

    runtime.separate_buffers = separate_buffers;
    if (ulcnet_model_io_descriptor_default(8, &descriptor) != 0 ||
        ulcnet_accelerator_adapter_get_mem_size(
            &descriptor, &bytes, &alignment) != 0 ||
        posix_memalign(&pool, alignment, bytes) != 0) {
        return 1;
    }
    /* An aligned-far or undefined deployment descriptor is rejected. */
    invalid_descriptor = descriptor;
    invalid_descriptor.far_input_mode = ULCNET_FAR_ALIGNED;
    if (ulcnet_accelerator_adapter_init(
            pool, bytes, &invalid_descriptor, run, &runtime) != NULL) {
        free(pool);
        return 1;
    }
    invalid_descriptor.far_input_mode = 2;
    if (ulcnet_accelerator_adapter_init(
            pool, bytes, &invalid_descriptor, run, &runtime) != NULL) {
        free(pool);
        return 1;
    }
    adapter = ulcnet_accelerator_adapter_init(
        pool, bytes, &descriptor, run, &runtime);
    if (!adapter ||
        ulcnet_accelerator_adapter_descriptor(adapter)->far_input_mode !=
            ULCNET_FAR_RAW) {
        free(pool);
        return 1;
    }
    model = ulcnet_accelerator_adapter_model(adapter);
    if (!model.infer || !model.reset ||
        /* The model published the adapter's compiled contract, which is what
         * lets a pipeline reject a far branch the checkpoint was not trained
         * on. */
        model.io_descriptor != ulcnet_accelerator_adapter_descriptor(adapter) ||
        model.io_descriptor->far_input_mode != ULCNET_FAR_RAW ||
        model.io_descriptor->delay_depth != 8 ||
        strcmp(ulcnet_far_input_mode_name(
                   model.io_descriptor->far_input_mode), "raw_far") != 0) {
        free(pool);
        return 1;
    }
    for (bin = 0; bin < ULCNET_MODEL_IO_BINS; ++bin) {
        error_re[bin] = (float)bin * 0.001f;
        error_im[bin] = (float)-bin * 0.002f;
    }
    if (model.infer(model.user, error_re, error_im, far_re, far_im,
                    output_re, output_im) != 0 || runtime.calls != 1 ||
        !nearly_equal(error_re, output_re, ULCNET_MODEL_IO_BINS) ||
        !nearly_equal(error_im, output_im, ULCNET_MODEL_IO_BINS)) {
        free(pool);
        return 1;
    }

    runtime.partial_write = 1;
    if (model.infer(model.user, error_re, error_im, far_re, far_im,
                    output_re, output_im) == 0 || runtime.calls != 2) {
        free(pool);
        return 1;
    }
    runtime.partial_write = 0;
    model.reset(model.user);
    if (model.infer(model.user, error_re, error_im, far_re, far_im,
                    output_re, output_im) != 0 || runtime.calls != 3) {
        free(pool);
        return 1;
    }

    /* A run that fills every output and THEN reports failure must not
     * commit: the pipeline discards that frame. Bound in place, the state
     * tensors stay as that run wrote them; with the runtime's own tensors the
     * run never reached inherit, so the state is untouched. Observed through
     * the NEXT run's inputs. */
    model.reset(model.user);
    runtime.fail_run = 1;
    runtime.stamp = 3.5f;
    if (model.infer(model.user, error_re, error_im, far_re, far_im,
                    output_re, output_im) == 0 || runtime.calls != 4) {
        free(pool);
        return 1;
    }
    runtime.fail_run = 0;
    runtime.stamp = 1.25f;
    if (model.infer(model.user, error_re, error_im, far_re, far_im,
                    output_re, output_im) != 0 || runtime.calls != 5 ||
        runtime.observed_key0 != (separate_buffers ? 0.0f : 3.5f) ||
        runtime.observed_gru0 != (separate_buffers ? 0.0f : 3.5f)) {
        free(pool);
        return 1;
    }
    /* ...and a run that succeeds carries its state forward the same way. */
    if (model.infer(model.user, error_re, error_im, far_re, far_im,
                    output_re, output_im) != 0 || runtime.calls != 6 ||
        runtime.observed_key0 != 1.25f || runtime.observed_gru0 != 1.25f) {
        free(pool);
        return 1;
    }

    if (separate_buffers) {
        /* One NaN in one of the runtime's own state tensors: inherit refuses
         * before writing anything, the adapter takes the skip, and the state
         * the next run sees is exactly what the last good run left. */
        runtime.stamp = 2.0f;
        if (model.infer(model.user, error_re, error_im, far_re, far_im,
                        output_re, output_im) != 0) {
            free(pool);
            return 1;
        }
        runtime.stamp = 9.0f;
        runtime.poison = 1;
        if (model.infer(model.user, error_re, error_im, far_re, far_im,
                        output_re, output_im) == 0) {
            free(pool);
            return 1;
        }
        runtime.poison = 0;
        runtime.stamp = 4.0f;
        if (model.infer(model.user, error_re, error_im, far_re, far_im,
                        output_re, output_im) != 0 ||
            runtime.observed_key0 != 2.0f || runtime.observed_gru0 != 2.0f) {
            free(pool);
            return 1;
        }
        /* A callback that reports success after a refused inherit leaves
         * `output` unwritten, so the frame is still refused -- but at commit,
         * which restarts the state from zero instead of keeping it. */
        runtime.stamp = 5.0f;
        runtime.poison = 1;
        runtime.ignore_refusal = 1;
        if (model.infer(model.user, error_re, error_im, far_re, far_im,
                        output_re, output_im) == 0) {
            free(pool);
            return 1;
        }
        runtime.poison = 0;
        runtime.ignore_refusal = 0;
        runtime.stamp = 6.0f;
        if (model.infer(model.user, error_re, error_im, far_re, far_im,
                        output_re, output_im) != 0 ||
            runtime.observed_key0 != 0.0f || runtime.observed_gru0 != 0.0f) {
            free(pool);
            return 1;
        }
    }

    free(pool);
    return 0;
}

int main(void) {
    if (exercise(0) != 0 || exercise(1) != 0) {
        return 1;
    }
    puts("ulcnet_accelerator_adapter: PASS");
    return 0;
}
