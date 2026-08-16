#include "audio_pipeline_4ch_ulcnet.h"
#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Replace with the product's stateless accelerator invocation. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* TODO(board): write every output tensor, then return 0. */
}

int main(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_ulcnet_default_config();
    UlcnetAcceleratorAdapter *adapter;
    AudioPipeline4ChUlcnet *pipeline;
    UlcnetModel model;
    void *adapter_pool = NULL;
    size_t adapter_bytes;
    size_t adapter_alignment;
    float microphones[256 * 4] = {0};
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
    pipeline = audio_pipeline_4ch_ulcnet_create(&cfg);
    if (!adapter || !pipeline) {
        audio_pipeline_4ch_ulcnet_destroy(pipeline);
        free(adapter_pool);
        return 1;
    }

    model = ulcnet_accelerator_adapter_model(adapter);
    if (audio_pipeline_4ch_ulcnet_set_model(pipeline, &model) != 0 ||
        audio_pipeline_4ch_ulcnet_set_far_input_mode(
            pipeline, ULCNET_FAR_RAW) != 0) {
        /* The pipeline TU has no stdio, so the far-contract disagreement it
         * rejects is named here, where both values are in hand. */
        fprintf(stderr,
                "4ch_alignulcnet: model/far-mode install failed "
                "(pipeline far_input_mode=%s, checkpoint far_input_mode=%s)\n",
                ulcnet_far_input_mode_name(
                    audio_pipeline_4ch_ulcnet_far_input_mode(pipeline)),
                ulcnet_far_input_mode_name(
                    model.io_descriptor ? model.io_descriptor->far_input_mode
                                        : -1));
        audio_pipeline_4ch_ulcnet_destroy(pipeline);
        free(adapter_pool);
        return 1;
    }

    for (hop = 0; hop < 4; ++hop) {
        if (audio_pipeline_4ch_ulcnet_process_with_activity(
                pipeline, microphones, far, 0, output) != 0) {
            audio_pipeline_4ch_ulcnet_destroy(pipeline);
            free(adapter_pool);
            return 1;
        }
        for (index = 0; index < 256; ++index) {
            if (!isfinite(output[index])) {
                audio_pipeline_4ch_ulcnet_destroy(pipeline);
                free(adapter_pool);
                return 1;
            }
        }
    }

    audio_pipeline_4ch_ulcnet_destroy(pipeline);
    free(adapter_pool);
    puts("4ch_alignulcnet: fail-open board skeleton PASS");
    return 0;
}
