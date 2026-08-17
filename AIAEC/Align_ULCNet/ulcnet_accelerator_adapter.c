#include "ulcnet_accelerator_adapter.h"

#include <stdint.h>
#include <string.h>

struct UlcnetAcceleratorAdapter {
    UlcnetModelIoState *state;
    UlcnetAcceleratorRun run;
    void *run_user;
};

int ulcnet_accelerator_adapter_get_mem_size(
    const UlcnetModelIoDescriptor *descriptor,
    size_t *bytes,
    size_t *alignment) {
    UlcnetModelIoMemReq model_req;
    size_t header_bytes;

    if (!bytes || !alignment ||
        ulcnet_model_io_get_mem_requirements(descriptor, &model_req) != 0) {
        return -1;
    }
    if (ulcnet_model_io_align_up(sizeof(UlcnetAcceleratorAdapter),
                                 model_req.alignment, &header_bytes) != 0 ||
        header_bytes > SIZE_MAX - model_req.bytes) {
        return -1;
    }
    *bytes = header_bytes + model_req.bytes;
    *alignment = model_req.alignment;
    return 0;
}

UlcnetAcceleratorAdapter *ulcnet_accelerator_adapter_init(
    void *memory,
    size_t bytes,
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetAcceleratorRun run,
    void *run_user) {
    UlcnetModelIoMemReq model_req;
    UlcnetAcceleratorAdapter *adapter;
    unsigned char *model_memory;
    size_t required;
    size_t alignment;
    size_t header_bytes;

    if (!memory ||
        ulcnet_accelerator_adapter_get_mem_size(descriptor, &required,
                                                &alignment) != 0 ||
        bytes < required || ((uintptr_t)memory % alignment) != 0u) {
        return NULL;
    }
    if (ulcnet_model_io_descriptor_validate(descriptor) != 0 ||
        ulcnet_model_io_get_mem_requirements(descriptor, &model_req) != 0) {
        return NULL;
    }

    memset(memory, 0, required);
    adapter = (UlcnetAcceleratorAdapter *)memory;
    if (ulcnet_model_io_align_up(sizeof(*adapter), model_req.alignment,
                                 &header_bytes) != 0) {
        return NULL;
    }
    model_memory = (unsigned char *)memory + header_bytes;
    adapter->state = ulcnet_model_io_init(model_memory, model_req.bytes,
                                          descriptor);
    if (!adapter->state) {
        return NULL;
    }
    adapter->run = run;
    adapter->run_user = run_user;
    return adapter;
}

static int infer(void *user,
                 const float error_re[257], const float error_im[257],
                 const float far_re[257], const float far_im[257],
                 float enhanced_re[257], float enhanced_im[257]) {
    UlcnetAcceleratorAdapter *adapter = (UlcnetAcceleratorAdapter *)user;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;

    if (!adapter || !adapter->state ||
        ulcnet_model_io_prepare(adapter->state, error_re, error_im,
                                far_re, far_im, &inputs, &outputs) != 0) {
        return -1;
    }

    if (!adapter->run ||
        adapter->run(adapter->run_user, &inputs, &outputs) != 0) {
        /* A failed run must not advance the K/V, logit or GRU rings, so it
         * must not commit: a runtime that filled every output and THEN
         * reported failure would otherwise pass commit()'s finite check and
         * step the persistent state off a frame the pipeline discards. The
         * next prepare() re-arms the transaction and re-fills the outputs
         * with NaN, so the NaN-prefill plus commit()'s finite gate remain as
         * the second line of defence against a partial write. */
        return -1;
    }
    return ulcnet_model_io_commit(adapter->state, enhanced_re, enhanced_im);
}

static void reset(void *user) {
    UlcnetAcceleratorAdapter *adapter = (UlcnetAcceleratorAdapter *)user;
    if (adapter) {
        ulcnet_model_io_reset(adapter->state);
    }
}

const UlcnetModelIoDescriptor *ulcnet_accelerator_adapter_descriptor(
    const UlcnetAcceleratorAdapter *adapter) {
    return adapter ? ulcnet_model_io_descriptor(adapter->state) : NULL;
}

UlcnetModel ulcnet_accelerator_adapter_model(UlcnetAcceleratorAdapter *adapter) {
    UlcnetModel model;
    model.user = adapter;
    model.infer = adapter ? infer : NULL;
    model.reset = adapter ? reset : NULL;
    model.io_descriptor = ulcnet_accelerator_adapter_descriptor(adapter);
    return model;
}
