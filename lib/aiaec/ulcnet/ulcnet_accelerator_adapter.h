/* Adapter for a stateless Align-ULCNet accelerator graph: drives the ulcnet_prepost frame transaction around a run
 * callback. The recurrent state is the application's: a float host registers its buffers at init, a board none. */
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

/* Adapter plus frame staging (the registered state is the caller's, not in the pool); sized by the grid, not D. */
int ulcnet_accelerator_adapter_get_mem_size(
    const UlcnetModelIoDescriptor *descriptor,
    size_t *bytes,
    size_t *alignment);

/* Initialize from the descriptor published beside the ONNX graph (raw-far contract; interior fields ignored).
 * descriptor.delay_depth MUST equal the graph's export depth; it is only range-checked. `state` registers the
 * application's recurrent state (UlcnetModelIoStateBuffers): borrowed, so the buffers must outlive the adapter;
 * zeroed here, handed to `run` as the state views. NULL registers nothing: `run` then gets NULL state views and holds
 * the state itself. NULL on a bad descriptor or registration, or a short or misaligned block. */
UlcnetAcceleratorAdapter *ulcnet_accelerator_adapter_init(
    void *memory,
    size_t bytes,
    const UlcnetModelIoDescriptor *descriptor,
    const UlcnetModelIoStateBuffers *state,
    UlcnetAcceleratorRun run,
    void *run_user);

/* Restart counter of the state tensors: advances on reset or when a refused frame restarted the state. A fixed-point
 * accelerator loads quantized zeros into its state at start and on every change. 0 for a NULL adapter. */
uint32_t ulcnet_accelerator_adapter_state_generation(
    const UlcnetAcceleratorAdapter *adapter);

/* The external contract the adapter runs, built from D and the layout; its five interior fields are 0 whatever the
 * init descriptor carried. Check a graph's metadata with ulcnet_model_io_descriptor_validate(), not memcmp().
 * Valid for the adapter's lifetime; NULL for a NULL adapter. */
const UlcnetModelIoDescriptor *ulcnet_accelerator_adapter_descriptor(
    const UlcnetAcceleratorAdapter *adapter);

/* The model is copied into the pipeline with a pointer to the descriptor; the adapter and pool must outlive it. */
UlcnetModel ulcnet_accelerator_adapter_model(UlcnetAcceleratorAdapter *adapter);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_ACCELERATOR_ADAPTER_H */
