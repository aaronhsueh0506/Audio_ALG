/* Adapter for a stateless Align-ULCNet accelerator graph: drives the ulcnet_prepost frame transaction (open, run,
 * finish or bypass), spectrum in and out; the caller owns the framing clock and the recurrent state. */
#include "ulcnet_accelerator_adapter.h"

#include <stdint.h>
#include <string.h>

#include "utility/mem_align.h"

#include "ulcnet_prepost.h"

struct UlcnetAcceleratorAdapter {
    UlcnetPrepost *prepost;
    UlcnetAcceleratorRun run;
    void *run_user;
};

/* Class config for this descriptor and registration (NULL `state` registers nothing); aligned-far, wrong-layout and
 * inconsistent descriptors are rejected before sizing. */
static int adapter_config(const UlcnetModelIoDescriptor *descriptor,
                          const UlcnetModelIoStateBuffers *state,
                          UlcnetPrepostConfig *cfg) {
    UlcnetGrid grid;
    if (ulcnet_model_io_descriptor_grid(descriptor, &grid) != 0 ||
        ulcnet_prepost_config_defaults(cfg, &grid) != 0) return -1;
    cfg->model_layout_version = descriptor->layout_version;
    if (state) {
        cfg->key_history = state->key_history;
        cfg->key_history_elements = state->key_history_elements;
        cfg->value_history = state->value_history;
        cfg->value_history_elements = state->value_history_elements;
        cfg->logit_history = state->logit_history;
        cfg->logit_history_elements = state->logit_history_elements;
        cfg->h_gru = state->h_gru;
        cfg->gru_hidden_elements = state->gru_hidden_elements;
    }
    return 0;
}

/* One pool: header, then the class (model I/O state and frame staging). */
static int adapter_layout(const UlcnetModelIoDescriptor *descriptor,
                          const UlcnetModelIoStateBuffers *state,
                          UlcnetPrepostConfig *cfg,
                          UlcnetPrepostMemReq *req,
                          size_t *header_bytes,
                          size_t *total_bytes) {
    if (adapter_config(descriptor, state, cfg) != 0 ||
        ulcnet_prepost_get_mem_size(cfg, req) != 0) {
        return -1;
    }
    /* The class reports uint64_t bytes; the round-trip keeps the check meaningful where size_t is 32-bit. */
    if ((uint64_t)(size_t)req->bytes != req->bytes) {
        return -1;
    }
    *header_bytes = ck_align16_size(sizeof(UlcnetAcceleratorAdapter));
    *total_bytes = ck_add_size(*header_bytes, (size_t)req->bytes);
    return (MEM_SIZE_INVALID(*header_bytes) || MEM_SIZE_INVALID(*total_bytes)) ? -1 : 0;
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
        adapter_layout(descriptor, NULL, &cfg, &req, &header_bytes,
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
    const UlcnetModelIoStateBuffers *state,
    UlcnetAcceleratorRun run,
    void *run_user) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    UlcnetAcceleratorAdapter *adapter;
    size_t header_bytes;
    size_t required;

    if (!memory ||
        adapter_layout(descriptor, state, &cfg, &req, &header_bytes,
                       &required) != 0 ||
        bytes < required ||
        ((uintptr_t)memory % (uintptr_t)req.alignment) != 0u) {
        return NULL;
    }

    memset(memory, 0, required);
    adapter = (UlcnetAcceleratorAdapter *)memory;
    /* req is this build's own sizing, so the class re-validates the carve against layout, depth and build. */
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
                 const float *error_re, const float *error_im,
                 const float *far_re, const float *far_im,
                 float *enhanced_re, float *enhanced_im) {
    UlcnetAcceleratorAdapter *adapter = (UlcnetAcceleratorAdapter *)user;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;

    if (!adapter || !adapter->prepost || !adapter->run) {
        return -1;
    }
    if (ulcnet_prepost_pre_process_freq(adapter->prepost, error_re, error_im,
                                        far_re, far_im) != 1 ||
        ulcnet_prepost_frame_open(adapter->prepost, &inputs,
                                    &outputs) != 0) {
        return -1;
    }

    if (adapter->run(adapter->run_user, &inputs, &outputs) != 0) {
        /* A failed run is discarded without finishing; the caller's spectra stay untouched. */
        (void)ulcnet_prepost_frame_bypass(adapter->prepost);
        return -1;
    }
    if (ulcnet_prepost_frame_finish(adapter->prepost) != 0) {
        /* The frame is refused: the state restarts from zero and nothing is written to the caller. */
        (void)ulcnet_prepost_frame_bypass(adapter->prepost);
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
