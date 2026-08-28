#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TestRuntime {
    int partial_write;
    int fail_run;         /* write every output, then report failure */
    int calls;
    float stamp;          /* value written into h_gru0_out     */
    float observed_gru0;  /* inputs->h_gru0[0] seen on entry    */
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

static int run(void *user, const UlcnetModelIoInputs *inputs,
               UlcnetModelIoOutputs *outputs) {
    TestRuntime *runtime = (TestRuntime *)user;
    size_t index;
    runtime->calls += 1;
    runtime->observed_gru0 = inputs->h_gru0[0];
    for (index = 0; index < outputs->spectrum_ri_elements; ++index) {
        outputs->output[index] = inputs->error_ri[index];
    }
    if (runtime->partial_write) {
        return 0;
    }
    fill(outputs->key_now, outputs->key_now_elements, 0.0f);
    fill(outputs->value_now, outputs->value_now_elements, 0.0f);
    fill(outputs->logit_now, outputs->logit_now_elements, 0.0f);
    fill(outputs->h_gru0_out, outputs->gru_hidden_elements,
         runtime->stamp);
    fill(outputs->h_gru1_out, outputs->gru_hidden_elements, 0.0f);
    return runtime->fail_run ? -1 : 0;
}

int main(void) {
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModel model;
    TestRuntime runtime = {0, 0, 0, 0.0f, 0.0f};
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

    if (ulcnet_model_io_descriptor_default(8, &descriptor) != 0 ||
        ulcnet_accelerator_adapter_get_mem_size(
            &descriptor, &bytes, &alignment) != 0 ||
        posix_memalign(&pool, alignment, bytes) != 0) {
        return 1;
    }
    /* A raw-far or undefined deployment descriptor is rejected. */
    invalid_descriptor = descriptor;
    invalid_descriptor.far_input_mode = ULCNET_FAR_RAW;
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
            ULCNET_FAR_ALIGNED) {
        free(pool);
        return 1;
    }
    model = ulcnet_accelerator_adapter_model(adapter);
    if (!model.infer || !model.reset ||
        /* The model published the adapter's compiled contract, which is what
         * lets a pipeline reject a far branch the checkpoint was not trained
         * on. */
        model.io_descriptor != ulcnet_accelerator_adapter_descriptor(adapter) ||
        model.io_descriptor->far_input_mode != ULCNET_FAR_ALIGNED ||
        model.io_descriptor->delay_depth != 8 ||
        strcmp(ulcnet_far_input_mode_name(
                   model.io_descriptor->far_input_mode), "aligned_far") != 0) {
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
     * advance the persistent state: the pipeline discards that frame, so
     * committing it would step the K/V, logit and GRU rings off a frame that
     * never reached the output. Observed through the NEXT run's inputs. */
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
        runtime.observed_gru0 != 0.0f) {
        free(pool);
        return 1;
    }
    /* ...and a run that succeeds still does advance it. */
    if (model.infer(model.user, error_re, error_im, far_re, far_im,
                    output_re, output_im) != 0 || runtime.calls != 6 ||
        runtime.observed_gru0 != 1.25f) {
        free(pool);
        return 1;
    }

    free(pool);
    puts("ulcnet_accelerator_adapter: PASS");
    return 0;
}
