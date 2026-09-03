/* CPU-owned state adapter for a stateless Align-ULCNet accelerator graph.
 *
 * The per-frame transaction (prepare -> run -> commit-or-discard) lives in
 * exactly ONE place, ulcnet_prepost.c, and this file drives it. The adapter
 * is the spectrum-in/spectrum-out face of that class: it carves no framing
 * state, so it runs the class in ULCNET_IO_FREQ, where the caller owns the
 * framing clock and no FftHandle or window is involved at all.
 */
#include "ulcnet_accelerator_adapter.h"

#include <stdint.h>
#include <string.h>

#include "ulcnet_prepost.h"

struct UlcnetAcceleratorAdapter {
    UlcnetPrepost *prepost;
    UlcnetAcceleratorRun run;
    void *run_user;
};

/* The class config this descriptor deploys as.
 *
 * descriptor_validate() pins every field except delay_depth to this build's
 * compile-time grid constants, so a descriptor that validates IS
 * descriptor_default(delay_depth) field for field -- handing the class the
 * depth alone loses nothing, and the class rebuilds the same descriptor.
 * Validating here also keeps the reject-first contract the header states:
 * a raw-far or out-of-range descriptor never reaches a sizing computation.
 */
static int adapter_config(const UlcnetModelIoDescriptor *descriptor,
                          UlcnetPrepostConfig *cfg) {
    if (ulcnet_model_io_descriptor_validate(descriptor) != 0) {
        return -1;
    }
    return ulcnet_prepost_config_defaults(cfg, ULCNET_IO_FREQ,
                                          descriptor->delay_depth);
}

/* One pool: this header, then the class -- which itself contains the model
 * I/O state and the frequency-domain frame staging. */
static int adapter_layout(const UlcnetModelIoDescriptor *descriptor,
                          UlcnetPrepostConfig *cfg,
                          UlcnetPrepostMemReq *req,
                          size_t *header_bytes,
                          size_t *total_bytes) {
    if (adapter_config(descriptor, cfg) != 0 ||
        ulcnet_prepost_get_mem_size(cfg, req) != 0) {
        return -1;
    }
    /* The class reports bytes as uint64_t; on a 32-bit target that is wider
     * than size_t. Written as a round-trip rather than a compare against
     * SIZE_MAX so it stays a real test instead of a tautology a 64-bit
     * -Wtype-limits build would (rightly) flag. */
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
    /* _init_ex, not _init: `req` is what THIS build just sized, so the
     * class's stale-pool gate re-proves the carve it is handed matches the
     * mode, depth and build the header above was sized for. The class pool
     * starts at a multiple of req.alignment past an equally aligned base. */
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
    /* In ULCNET_IO_FREQ one spectrum is one accelerator invocation, so this
     * returns 1 or it failed; there is no centered double emission to loop
     * over (that is the ULCNET_IO_TIME caller's job). */
    if (ulcnet_prepost_pre_process_freq(adapter->prepost, error_re, error_im,
                                        far_re, far_im) != 1 ||
        ulcnet_prepost_frame_inputs(adapter->prepost, &inputs,
                                    &outputs) != 0) {
        return -1;
    }

    if (adapter->run(adapter->run_user, &inputs, &outputs) != 0) {
        /* A failed run must not advance the K/V, logit or GRU rings, so it
         * must not commit: a runtime that filled every output and THEN
         * reported failure would otherwise pass commit()'s finite check and
         * step the persistent state off a frame the pipeline discards.
         * frame_skip() closes the transaction WITHOUT committing -- no ring
         * moves -- and the next prepare() re-arms it and re-fills the
         * outputs with NaN, so the NaN-prefill plus commit()'s finite gate
         * remain as the second line of defence against a partial write. The
         * caller's enhanced spectra stay untouched, because -1 here means
         * this frame produced nothing. */
        (void)ulcnet_prepost_frame_skip(adapter->prepost);
        return -1;
    }
    if (ulcnet_prepost_frame_commit(adapter->prepost) != 0) {
        /* Same rule from the other direction: commit refused the frame (a
         * partial or non-finite write), left the persistent state exactly as
         * it was, and wrote nothing to the caller. */
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

const UlcnetModelIoDescriptor *ulcnet_accelerator_adapter_descriptor(
    const UlcnetAcceleratorAdapter *adapter) {
    /* Lives in the model I/O state inside the caller's pool, so it outlives
     * the adapter exactly as the header promises. */
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
