#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

#define DEPTH 8u

typedef struct TestRuntime {
    int poison;           /* one NaN in the newest K/V slot of the output */
    int partial_write;
    int fail_run;         /* write every output, then report failure */
    int calls;
    float stamp;          /* value written into every state tensor  */
    float observed_key0;  /* inputs->key_history[0] seen on entry   */
    float observed_gru;   /* inputs->h_gru[0] seen on entry         */
} TestRuntime;

/* Compression round trip (prepare's signed pow 0.3, commit's inverse) is
 * identity only up to fp32 powf error, so the pass-through model is checked
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

/* The graph's window shift for a constant frame, in place: K/V windows
 * newest-first (X = cat(new, W[0:D-1])), logit window oldest-first
 * (X = cat(W[1:5], new)); D = 8, TA_BINS from the compiled grid. */
static void push_stamp(const UlcnetModelIoOutputs *outputs, float stamp) {
    const size_t ta_bins = ULCNET_MODEL_IO_TA_BINS;
    const size_t score = ULCNET_MODEL_IO_SCORE_WINDOW;
    size_t channel;
    size_t index;
    for (channel = 0; channel < ULCNET_MODEL_IO_TA_CHANNELS; ++channel) {
        float *key = outputs->key_history_out + channel * DEPTH * ta_bins;
        float *value = outputs->value_history_out + channel * DEPTH * ta_bins;
        float *logit = outputs->logit_history_out + channel * score * DEPTH;
        memmove(key + ta_bins, key, (DEPTH - 1u) * ta_bins * sizeof(float));
        memmove(value + ta_bins, value,
                (DEPTH - 1u) * ta_bins * sizeof(float));
        memmove(logit, logit + DEPTH, (score - 1u) * DEPTH * sizeof(float));
        for (index = 0; index < ta_bins; ++index) {
            key[index] = stamp;
            value[index] = stamp;
        }
        for (index = 0; index < DEPTH; ++index)
            logit[(score - 1u) * DEPTH + index] = stamp;
    }
}

/* A runtime bound the way the board binds it: every state output is the
 * address of its input (the pointers prepare() returned), and every input is
 * read before any output is written. */
static int run(void *user, const UlcnetModelIoInputs *inputs,
               UlcnetModelIoOutputs *outputs) {
    TestRuntime *runtime = (TestRuntime *)user;
    runtime->calls += 1;
    runtime->observed_key0 = inputs->key_history[0];
    runtime->observed_gru = inputs->h_gru[0];
    if (outputs->key_history_out != inputs->key_history ||
        outputs->value_history_out != inputs->value_history ||
        outputs->logit_history_out != inputs->logit_history ||
        outputs->h_gru_out != inputs->h_gru) {
        return -1;
    }
    /* A partial write leaves all but the first mask element at NaN. */
    write_identity_mask(outputs->output, inputs, runtime->partial_write);
    if (runtime->partial_write) {
        return 0;
    }
    push_stamp(outputs, runtime->stamp);
    fill(outputs->h_gru_out, outputs->gru_hidden_elements, runtime->stamp);
    if (runtime->poison) {
        outputs->key_history_out[0] = NAN;
    }
    return runtime->fail_run ? -1 : 0;
}

static int infer(const UlcnetModel *model, const float *error_re,
                 const float *error_im, const float *far_re,
                 const float *far_im, float *output_re, float *output_im) {
    return model->infer(model->user, error_re, error_im, far_re, far_im,
                        output_re, output_im);
}

static int exercise(void) {
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModel model;
    TestRuntime runtime = {0, 0, 0, 0, 0.0f, 0.0f, 0.0f};
    void *pool = NULL;
    size_t bytes;
    size_t alignment;
    uint32_t generation;
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

    CHECK(ulcnet_model_io_descriptor_default(8, &descriptor) == 0);
    CHECK(ulcnet_accelerator_adapter_get_mem_size(
              &descriptor, &bytes, &alignment) == 0);
    CHECK(posix_memalign(&pool, alignment, bytes) == 0);
    /* An aligned-far or undefined deployment descriptor is rejected. */
    invalid_descriptor = descriptor;
    invalid_descriptor.far_input_mode = ULCNET_FAR_ALIGNED;
    CHECK(ulcnet_accelerator_adapter_init(
              pool, bytes, &invalid_descriptor, run, &runtime) == NULL);
    invalid_descriptor.far_input_mode = 2;
    CHECK(ulcnet_accelerator_adapter_init(
              pool, bytes, &invalid_descriptor, run, &runtime) == NULL);
    /* A descriptor of another layout is rejected too. */
    invalid_descriptor = descriptor;
    invalid_descriptor.layout_version = 17u;
    CHECK(ulcnet_accelerator_adapter_init(
              pool, bytes, &invalid_descriptor, run, &runtime) == NULL);
    adapter = ulcnet_accelerator_adapter_init(
        pool, bytes, &descriptor, run, &runtime);
    CHECK(adapter != NULL);
    CHECK(ulcnet_accelerator_adapter_descriptor(adapter)->far_input_mode ==
          ULCNET_FAR_RAW);
    model = ulcnet_accelerator_adapter_model(adapter);
    CHECK(model.infer && model.reset);
    /* The model published the adapter's compiled contract, which is what
     * lets a pipeline reject a far branch the checkpoint was not trained
     * on. */
    CHECK(model.io_descriptor == ulcnet_accelerator_adapter_descriptor(adapter));
    CHECK(model.io_descriptor->far_input_mode == ULCNET_FAR_RAW);
    CHECK(model.io_descriptor->delay_depth == 8);
    CHECK(model.io_descriptor->layout_version == 18u);
    CHECK(strcmp(ulcnet_far_input_mode_name(
                     model.io_descriptor->far_input_mode), "raw_far") == 0);
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) == 0u);
    CHECK(ulcnet_accelerator_adapter_state_generation(NULL) == 0u);

    for (bin = 0; bin < ULCNET_MODEL_IO_BINS; ++bin) {
        error_re[bin] = (float)bin * 0.001f;
        error_im[bin] = (float)-bin * 0.002f;
    }
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) == 0);
    CHECK(runtime.calls == 1);
    CHECK(nearly_equal(error_re, output_re, ULCNET_MODEL_IO_BINS));
    CHECK(nearly_equal(error_im, output_im, ULCNET_MODEL_IO_BINS));
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) == 0u);

    /* A partial mask is refused at commit: the state restarts from zero and
     * the generation advances. */
    runtime.partial_write = 1;
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) != 0);
    CHECK(runtime.calls == 2);
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) == 1u);
    runtime.partial_write = 0;
    model.reset(model.user);
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) == 2u);
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) == 0);
    CHECK(runtime.calls == 3);

    /* A run that fills every output and THEN reports failure must not
     * commit: the pipeline discards that frame. Bound in place, the state
     * tensors stay as that run wrote them (and nothing counts as a restart).
     * Observed through the NEXT run's inputs. */
    model.reset(model.user);
    generation = ulcnet_accelerator_adapter_state_generation(adapter);
    runtime.fail_run = 1;
    runtime.stamp = 3.5f;
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) != 0);
    CHECK(runtime.calls == 4);
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) == generation);
    runtime.fail_run = 0;
    runtime.stamp = 1.25f;
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) == 0);
    CHECK(runtime.calls == 5);
    CHECK(runtime.observed_key0 == 3.5f && runtime.observed_gru == 3.5f);
    /* ...and a run that succeeds carries its state forward the same way. */
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) == 0);
    CHECK(runtime.calls == 6);
    CHECK(runtime.observed_key0 == 1.25f && runtime.observed_gru == 1.25f);

    /* A non-finite value in the newest K/V slot is refused at commit: the
     * frame is dropped, the state restarts from zero, the generation
     * advances once, and the run after it sees zeros. */
    generation = ulcnet_accelerator_adapter_state_generation(adapter);
    runtime.stamp = 9.0f;
    runtime.poison = 1;
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) != 0);
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) ==
          generation + 1u);
    runtime.poison = 0;
    runtime.stamp = 4.0f;
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) == 0);
    CHECK(runtime.observed_key0 == 0.0f && runtime.observed_gru == 0.0f);
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) ==
          generation + 1u);

    free(pool);
    return 0;
}

int main(void) {
    if (exercise() != 0) {
        return 1;
    }
    puts("ulcnet_accelerator_adapter: PASS");
    return 0;
}
