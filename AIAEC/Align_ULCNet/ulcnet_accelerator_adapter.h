/* CPU-owned state adapter for a stateless Align-ULCNet accelerator graph. */
#ifndef ULCNET_ACCELERATOR_ADAPTER_H
#define ULCNET_ACCELERATOR_ADAPTER_H

#include <stddef.h>

#include "ulcnet_model_io.h"
#include "ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*UlcnetAcceleratorRun)(
    void *user,
    const UlcnetModelIoInputs *inputs,
    UlcnetModelIoOutputs *outputs);

typedef struct UlcnetAcceleratorAdapter UlcnetAcceleratorAdapter;

/* Includes the adapter and every CPU-owned K/V, logit and GRU state tensor.
 * Independent of far_input_mode -- that selects which far stream the caller
 * feeds, not how much state is stored. */
int ulcnet_accelerator_adapter_get_mem_size(int delay_depth,
                                            size_t *bytes,
                                            size_t *alignment);

/* far_input_mode is the deployed GRAPH's contract, i.e. the ONNX/JSON
 * metadata's far_input_mode mapped through ulcnet_far_input_mode_name();
 * every currently exported graph records raw_far (the exporter's explicit
 * ALIGNED override is pending). It is not a behaviour switch inside the
 * adapter: it is recorded in the descriptor this adapter publishes, so the
 * pipeline the model is installed into can reject a far branch that
 * disagrees. An undefined value is rejected here. */
UlcnetAcceleratorAdapter *ulcnet_accelerator_adapter_init(
    void *memory,
    size_t bytes,
    int delay_depth,
    int far_input_mode,
    UlcnetAcceleratorRun run,
    void *run_user);

/* The compiled model-I/O contract of this adapter's state, for a board
 * comparing it against the ONNX metadata it loaded. Valid for the adapter's
 * lifetime; NULL for a NULL adapter. */
const UlcnetModelIoDescriptor *ulcnet_accelerator_adapter_descriptor(
    const UlcnetAcceleratorAdapter *adapter);

/* UlcnetModel is copied into the mono/4ch pipeline, WITH a pointer to the
 * descriptor above. The adapter and its caller-owned pool must therefore
 * outlive that pipeline instance. */
UlcnetModel ulcnet_accelerator_adapter_model(UlcnetAcceleratorAdapter *adapter);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_ACCELERATOR_ADAPTER_H */
