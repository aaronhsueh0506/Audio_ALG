/* CPU-owned state adapter for a stateless Align-ULCNet accelerator graph.
 * Drives the ulcnet_prepost frame transaction (prepare, run, commit or discard)
 * in ULCNET_IO_FREQ: spectrum in, spectrum out, caller owns the framing clock. */
#include "ulcnet_accelerator_adapter.h"

#include <stdint.h>
#include <string.h>

#include "ulcnet_prepost.h"

struct UlcnetAcceleratorAdapter {
    UlcnetPrepost *prepost;
    UlcnetAcceleratorRun run;
    void *run_user;
};

/* Class config for this descriptor. Geometry and layout version must match this
 * build; aligned-far and out-of-range descriptors are rejected before sizing. */
static int adapter_config(const UlcnetModelIoDescriptor *descriptor,
                          UlcnetPrepostConfig *cfg) {
    if (ulcnet_model_io_descriptor_validate(descriptor) != 0) {
        return -1;
    }
    if (ulcnet_prepost_config_defaults(cfg, ULCNET_IO_FREQ,
                                       descriptor->delay_depth) != 0) return -1;
    cfg->model_layout_version = descriptor->layout_version;
    return 0;
}

/* One pool: header, then the class (model I/O state and frequency-domain staging). */
static int adapter_layout(const UlcnetModelIoDescriptor *descriptor,
                          UlcnetPrepostConfig *cfg,
                          UlcnetPrepostMemReq *req,
                          size_t *header_bytes,
                          size_t *total_bytes) {
    if (adapter_config(descriptor, cfg) != 0 ||
        ulcnet_prepost_get_mem_size(cfg, req) != 0) {
        return -1;
    }
    /* The class reports uint64_t bytes; the round-trip keeps the check
     * meaningful where size_t is 32-bit. */
    if ((uint64_t)(size_t)req->bytes != req->bytes) {
        return -1;
    }
    if (ulcnet_model_io_align_up(sizeof(UlcnetAcceleratorAdapter),
                                 (size_t)req->alignment, header_bytes) != 0 ||
        *header_bytes > SIZE_MAX - (size_t)req->bytes) {
        return -1;
    }
    *total_bytes = *header_bytes + (size_t)req->bytes;
    return 0;
}

int ulcnet_accelerator_adapter_get_mem_size(
    const UlcnetModelIoDescriptor *descriptor,
    size_t *bytes,
    size_t *alignment) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    size_t header_bytes;
    size_t total_bytes;

    if (!bytes || !alignment ||
        adapter_layout(descriptor, &cfg, &req, &header_bytes,
                       &total_bytes) != 0) {
        return -1;
    }
    *bytes = total_bytes;
    *alignment = (size_t)req.alignment;
    return 0;
}

UlcnetAcceleratorAdapter *ulcnet_accelerator_adapter_init(
    void *memory,
    size_t bytes,
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetAcceleratorRun run,
    void *run_user) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    UlcnetAcceleratorAdapter *adapter;
    size_t header_bytes;
    size_t required;

    if (!memory ||
        adapter_layout(descriptor, &cfg, &req, &header_bytes,
                       &required) != 0 ||
        bytes < required ||
        ((uintptr_t)memory % (uintptr_t)req.alignment) != 0u) {
        return NULL;
    }

    memset(memory, 0, required);
    adapter = (UlcnetAcceleratorAdapter *)memory;
    /* req is this build's own sizing, so the class re-validates the carve
     * against mode, depth and build. */
    adapter->prepost = ulcnet_prepost_init_ex(
        (unsigned char *)memory + header_bytes, (size_t)req.bytes, &cfg, &req);
    if (!adapter->prepost) {
        return NULL;
    }
    adapter->run = run;
    adapter->run_user = run_user;
    return adapter;
}

static int infer(void *user,
                 const float error_re[ULCNET_MODEL_IO_BINS], const float error_im[ULCNET_MODEL_IO_BINS],
                 const float far_re[ULCNET_MODEL_IO_BINS], const float far_im[ULCNET_MODEL_IO_BINS],
                 float enhanced_re[ULCNET_MODEL_IO_BINS], float enhanced_im[ULCNET_MODEL_IO_BINS]) {
    UlcnetAcceleratorAdapter *adapter = (UlcnetAcceleratorAdapter *)user;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;

    if (!adapter || !adapter->prepost || !adapter->run) {
        return -1;
    }
    /* ULCNET_IO_FREQ: one spectrum is one invocation; returns 1 or fails. */
    if (ulcnet_prepost_pre_process_freq(adapter->prepost, error_re, error_im,
                                        far_re, far_im) != 1 ||
        ulcnet_prepost_frame_inputs(adapter->prepost, &inputs,
                                    &outputs) != 0) {
        return -1;
    }

    if (adapter->run(adapter->run_user, &inputs, &outputs) != 0) {
        /* A failed run is discarded without commit; the caller's spectra stay untouched. */
        (void)ulcnet_prepost_frame_skip(adapter->prepost);
        return -1;
    }
    if (ulcnet_prepost_frame_commit(adapter->prepost) != 0) {
        /* Commit refused the frame: state restarted from zero, nothing written to the caller. */
        (void)ulcnet_prepost_frame_skip(adapter->prepost);
        return -1;
    }
    return ulcnet_prepost_post_process_freq(adapter->prepost, enhanced_re,
                                            enhanced_im);
}

static void reset(void *user) {
    UlcnetAcceleratorAdapter *adapter = (UlcnetAcceleratorAdapter *)user;
    if (adapter) {
        ulcnet_prepost_reset(adapter->prepost);
    }
}

uint32_t ulcnet_accelerator_adapter_state_generation(
    const UlcnetAcceleratorAdapter *adapter) {
    return adapter ? ulcnet_prepost_state_generation(adapter->prepost) : 0u;
}

const UlcnetModelIoDescriptor *ulcnet_accelerator_adapter_descriptor(
    const UlcnetAcceleratorAdapter *adapter) {
    /* Lives in the model I/O state inside the caller's pool. */
    return adapter ? ulcnet_prepost_descriptor(adapter->prepost) : NULL;
}

UlcnetModel ulcnet_accelerator_adapter_model(UlcnetAcceleratorAdapter *adapter) {
    UlcnetModel model;
    model.user = adapter;
    model.infer = adapter ? infer : NULL;
    model.reset = adapter ? reset : NULL;
    model.io_descriptor = ulcnet_accelerator_adapter_descriptor(adapter);
    return model;
}
