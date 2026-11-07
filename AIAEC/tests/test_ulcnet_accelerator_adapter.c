#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

#define DEPTH 8
#define BINS_MAX 513

/* State table of the shipped Align-ULCNet graph (pinned to the exporter by
 * test_export_streaming_ulcnet.py): the adapter holds none of it, the host does. */
#define STATE_CHANNELS 32
#define STATE_SCORE_FRAMES 5
#define STATE_GRU_ROWS 4
#define STATE_GRU_HIDDEN 128
#define STATE_TA_BINS(n_bins) (((n_bins) + 9) / 10)

/* The four recurrent tensors of the application, registered with the adapter. */
typedef struct HostState {
    float *key, *value, *logit, *gru;
    UlcnetModelIoStateBuffers buffers;
} HostState;

static int host_state_make(HostState *host, const UlcnetGrid *grid) {
    const size_t depth = (size_t)grid->delay_depth;
    const size_t ta_bins = STATE_TA_BINS((size_t)grid->n_bins);
    const size_t kv = (size_t)STATE_CHANNELS * depth * ta_bins;
    const size_t logit = (size_t)STATE_CHANNELS * STATE_SCORE_FRAMES * depth;
    const size_t gru = (size_t)STATE_GRU_ROWS * STATE_GRU_HIDDEN;
    memset(host, 0, sizeof(*host));
    host->key = (float *)malloc(kv * sizeof(float));
    host->value = (float *)malloc(kv * sizeof(float));
    host->logit = (float *)malloc(logit * sizeof(float));
    host->gru = (float *)malloc(gru * sizeof(float));
    if (!host->key || !host->value || !host->logit || !host->gru) return -1;
    host->buffers = (UlcnetModelIoStateBuffers){
        .key_history = host->key, .key_history_elements = kv,
        .value_history = host->value, .value_history_elements = kv,
        .logit_history = host->logit, .logit_history_elements = logit,
        .h_gru = host->gru, .gru_hidden_elements = gru,
    };
    return 0;
}

static void host_state_free(HostState *host) {
    free(host->key);
    free(host->value);
    free(host->logit);
    free(host->gru);
}

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
 * (X = cat(W[1:5], new)). */
static void push_stamp(const UlcnetModelIoOutputs *outputs, size_t ta_bins,
                       float stamp) {
    const size_t depth = DEPTH;
    const size_t score = STATE_SCORE_FRAMES;
    size_t channel;
    size_t index;
    for (channel = 0; channel < STATE_CHANNELS; ++channel) {
        float *key = outputs->key_history_out + channel * depth * ta_bins;
        float *value = outputs->value_history_out + channel * depth * ta_bins;
        float *logit = outputs->logit_history_out + channel * score * depth;
        memmove(key + ta_bins, key, (depth - 1u) * ta_bins * sizeof(float));
        memmove(value + ta_bins, value,
                (depth - 1u) * ta_bins * sizeof(float));
        memmove(logit, logit + depth, (score - 1u) * depth * sizeof(float));
        for (index = 0; index < ta_bins; ++index) {
            key[index] = stamp;
            value[index] = stamp;
        }
        for (index = 0; index < depth; ++index)
            logit[(score - 1u) * depth + index] = stamp;
    }
}

/* A runtime bound the way the board binds it: every state output is the
 * address of its input (the pointers prepare() returned), and every input is
 * read before any output is written. The state views are the registered
 * buffers, so a count that differs from the table fails the run. */
static int run(void *user, const UlcnetModelIoInputs *inputs,
               UlcnetModelIoOutputs *outputs) {
    TestRuntime *runtime = (TestRuntime *)user;
    const size_t ta_bins = STATE_TA_BINS(inputs->spectrum_bins_elements);
    runtime->calls += 1;
    if (inputs->key_history_elements != STATE_CHANNELS * DEPTH * ta_bins ||
        inputs->value_history_elements != STATE_CHANNELS * DEPTH * ta_bins ||
        inputs->logit_history_elements !=
            STATE_CHANNELS * STATE_SCORE_FRAMES * DEPTH ||
        inputs->gru_hidden_elements != STATE_GRU_ROWS * STATE_GRU_HIDDEN ||
        inputs->key_history == NULL || inputs->h_gru == NULL) {
        return -1;
    }
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
    push_stamp(outputs, ta_bins, runtime->stamp);
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

/* The whole transaction on one grid. */
static int exercise(int sample_rate, int n_fft) {
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModel model;
    TestRuntime runtime = {0, 0, 0, 0, 0.0f, 0.0f, 0.0f};
    HostState host;
    void *pool = NULL;
    size_t bytes;
    size_t alignment;
    uint32_t generation;
    UlcnetGrid grid;
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoDescriptor invalid_descriptor;
    /* The adapter reads and writes descriptor->spectrum_bins floats through
     * these: sized for the larger grid. */
    float error_re[BINS_MAX];
    float error_im[BINS_MAX];
    float far_re[BINS_MAX] = {0};
    float far_im[BINS_MAX] = {0};
    float output_re[BINS_MAX];
    float output_im[BINS_MAX];
    const UlcnetGridSpec spec = {
        .sample_rate = sample_rate, .n_fft = n_fft, .delay_depth = DEPTH };
    int bin;

    CHECK(ulcnet_grid_make(&grid, &spec) == ULCNET_GRID_OK);
    CHECK(grid.n_bins <= BINS_MAX);
    CHECK(host_state_make(&host, &grid) == 0);
    CHECK(ulcnet_model_io_descriptor_default(&grid, &descriptor) == 0);
    CHECK(ulcnet_accelerator_adapter_get_mem_size(
              &descriptor, &bytes, &alignment) == 0);
    CHECK(posix_memalign(&pool, alignment, bytes) == 0);
    /* An aligned-far or undefined deployment descriptor is rejected. */
    invalid_descriptor = descriptor;
    invalid_descriptor.far_input_mode = ULCNET_FAR_ALIGNED;
    CHECK(ulcnet_accelerator_adapter_init(
              pool, bytes, &invalid_descriptor, &host.buffers, run,
              &runtime) == NULL);
    invalid_descriptor.far_input_mode = 2;
    CHECK(ulcnet_accelerator_adapter_init(
              pool, bytes, &invalid_descriptor, &host.buffers, run,
              &runtime) == NULL);
    /* A descriptor of another layout is rejected too. */
    invalid_descriptor = descriptor;
    invalid_descriptor.layout_version = 17u;
    CHECK(ulcnet_accelerator_adapter_init(
              pool, bytes, &invalid_descriptor, &host.buffers, run,
              &runtime) == NULL);
    adapter = ulcnet_accelerator_adapter_init(
        pool, bytes, &descriptor, &host.buffers, run, &runtime);
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
    CHECK(model.io_descriptor->delay_depth == DEPTH);
    CHECK(model.io_descriptor->layout_version == 18u);
    CHECK(strcmp(ulcnet_far_input_mode_name(
                     model.io_descriptor->far_input_mode), "raw_far") == 0);
    CHECK(ulcnet_accelerator_adapter_state_generation(adapter) == 0u);
    CHECK(ulcnet_accelerator_adapter_state_generation(NULL) == 0u);

    for (bin = 0; bin < grid.n_bins; ++bin) {
        error_re[bin] = (float)bin * 0.001f;
        error_im[bin] = (float)-bin * 0.002f;
    }
    CHECK(infer(&model, error_re, error_im, far_re, far_im, output_re,
                output_im) == 0);
    CHECK(runtime.calls == 1);
    CHECK(nearly_equal(error_re, output_re, (size_t)grid.n_bins));
    CHECK(nearly_equal(error_im, output_im, (size_t)grid.n_bins));
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
    host_state_free(&host);
    return 0;
}

int main(void) {
    /* The shipped grid, and a second grid as generality proof. */
    if (exercise(16000, 512) != 0 || exercise(48000, 1024) != 0) {
        return 1;
    }
    puts("ulcnet_accelerator_adapter: PASS");
    return 0;
}
