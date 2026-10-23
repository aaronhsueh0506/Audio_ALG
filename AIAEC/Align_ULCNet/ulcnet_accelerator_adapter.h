/* CPU-owned state adapter for a stateless Align-ULCNet accelerator graph. */
#ifndef ULCNET_ACCELERATOR_ADAPTER_H
#define ULCNET_ACCELERATOR_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

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

/* Adapter plus every CPU-owned K/V, logit and GRU state tensor. The descriptor
 * is the exported graph contract, including the export-time delay depth D. */
int ulcnet_accelerator_adapter_get_mem_size(
    const UlcnetModelIoDescriptor *descriptor,
    size_t *bytes,
    size_t *alignment);

/* Initialize from the descriptor published beside the ONNX graph (fixed raw-far
 * contract). descriptor.delay_depth MUST equal the depth the graph was exported
 * with; it is only range-checked. */
UlcnetAcceleratorAdapter *ulcnet_accelerator_adapter_init(
    void *memory,
    size_t bytes,
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetAcceleratorRun run,
    void *run_user);

/* Restart counter of the state tensors: advances on reset or when a refused
 * frame zeroed the state. A fixed-point accelerator refills its state with
 * quantized zeros whenever it changes. 0 for a NULL adapter. */
uint32_t ulcnet_accelerator_adapter_state_generation(
    const UlcnetAcceleratorAdapter *adapter);

/* Compiled model-I/O contract, for comparison with the loaded ONNX metadata.
 * Valid for the adapter's lifetime; NULL for a NULL adapter. */
const UlcnetModelIoDescriptor *ulcnet_accelerator_adapter_descriptor(
    const UlcnetAcceleratorAdapter *adapter);

/* UlcnetModel is copied into the pipeline together with a pointer to the
 * descriptor; the adapter and its pool must outlive that pipeline. */
UlcnetModel ulcnet_accelerator_adapter_model(UlcnetAcceleratorAdapter *adapter);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_ACCELERATOR_ADAPTER_H */
