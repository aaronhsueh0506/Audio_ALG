#include "audio_pipeline_ulcnet.h"
#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Replace this body with the board runtime call. `inputs` contains spectra
 * plus CPU-owned state; the runtime must fill every tensor in `outputs`. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* TODO(board): invoke the stateless ONNX accelerator. */
}

int main(void) {
    AudioPipelineUlcnetConfig cfg =
        audio_pipeline_ulcnet_default_config(16000);
    UlcnetAcceleratorAdapter *adapter;
    AudioPipelineUlcnet *pipeline;
    UlcnetModel model;
    void *adapter_pool = NULL;
    size_t adapter_bytes;
    size_t adapter_alignment;
    float mic[256] = {0};
    float far[256] = {0};
    float output[256];
    int hop;
    int index;
    /* A product reads this from the deployed checkpoint's ONNX metadata
     * ('far_input_mode', whose strings ulcnet_far_input_mode_name() mirrors)
     * instead of hard-coding it; every current checkpoint is raw_far. */
    const int checkpoint_far_input_mode = ULCNET_FAR_RAW;

    if (ulcnet_accelerator_adapter_get_mem_size(
            8, &adapter_bytes, &adapter_alignment) != 0 ||
        posix_memalign(&adapter_pool, adapter_alignment, adapter_bytes) != 0) {
        return 1;
    }
    adapter = ulcnet_accelerator_adapter_init(
        adapter_pool, adapter_bytes, 8, checkpoint_far_input_mode,
        run_accelerator, NULL);
    if (!adapter) {
        free(adapter_pool);
        return 1;
    }

    model = ulcnet_accelerator_adapter_model(adapter);
    cfg.model = model;
    cfg.far_input_mode = ULCNET_FAR_RAW;
    pipeline = audio_pipeline_ulcnet_create(&cfg);
    if (!pipeline) {
        /* The pipeline TU has no stdio, so the far-contract disagreement it
         * rejects is named here, where both values are in hand. */
        fprintf(stderr,
                "mono_alignulcnet: pipeline init failed "
                "(pipeline far_input_mode=%s, checkpoint far_input_mode=%s)\n",
                ulcnet_far_input_mode_name((int)cfg.far_input_mode),
                ulcnet_far_input_mode_name(
                    model.io_descriptor ? model.io_descriptor->far_input_mode
                                        : -1));
        free(adapter_pool);
        return 1;
    }

    /* Host smoke path. A product calls process() once per 256-sample hop. */
    for (hop = 0; hop < 4; ++hop) {
        if (audio_pipeline_ulcnet_process(pipeline, mic, far, output) != 0) {
            audio_pipeline_ulcnet_destroy(pipeline);
            free(adapter_pool);
            return 1;
        }
        for (index = 0; index < 256; ++index) {
            if (!isfinite(output[index])) {
                audio_pipeline_ulcnet_destroy(pipeline);
                free(adapter_pool);
                return 1;
            }
        }
    }

    audio_pipeline_ulcnet_destroy(pipeline);
    free(adapter_pool);
    puts("mono_alignulcnet: fail-open board skeleton PASS");
    return 0;
}
