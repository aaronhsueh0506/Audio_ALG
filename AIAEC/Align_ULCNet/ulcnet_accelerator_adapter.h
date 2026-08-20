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
 * The descriptor is the exported graph contract, including export-time D. */
int ulcnet_accelerator_adapter_get_mem_size(
    const UlcnetModelIoDescriptor *descriptor,
    size_t *bytes,
    size_t *alignment);

/* Initialize from the same descriptor published beside the ONNX graph.
 * Production validation requires the fixed aligned-far contract. */
/* Board integrators: the descriptor's delay_depth MUST equal the delay depth
 * the ONNX graph was exported with. Nothing verifies that -- the validator
 * only bounds-checks the range, and no C code here reads the model's metadata
 * -- and a mismatch is silent: the host rings are carved from the descriptor
 * while the runtime reads and writes the graph's shapes. There is no
 * exporter-generated descriptor header yet; populate it from the exported
 * model's own recorded depth. */
UlcnetAcceleratorAdapter *ulcnet_accelerator_adapter_init(
    void *memory,
    size_t bytes,
    const UlcnetModelIoDescriptor *descriptor,
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
