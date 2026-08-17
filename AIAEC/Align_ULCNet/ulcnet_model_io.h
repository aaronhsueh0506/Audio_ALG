/* Align-ULCNet stateless-accelerator model I/O state.
 *
 * The accelerator owns no persistent state.  This helper carves CPU-owned
 * K/V history, score-convolution history and temporal-GRU hidden tensors from
 * one caller-provided pool.  The ONNX graph returns only the current K/V/logit
 * entries and next GRU hidden tensors; commit() validates and incorporates
 * them into the state used by the next invocation.
 *
 * This file does not invoke an accelerator and does not contain STFT/WOLA.
 * Audio framing remains in ulcnet_process.c; a board adapter binds the views
 * below to its runtime's ordinary tensor inputs and outputs.
 */
#ifndef ULCNET_MODEL_IO_H
#define ULCNET_MODEL_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version 3 fixes the deployed far branch to AEC-aligned far.  The exported
 * metadata separately records the checkpoint's training provenance.  Kept
 * numerically equal to export_onnx.py's STATE_LAYOUT_VERSION. */
#define ULCNET_MODEL_IO_LAYOUT_VERSION 3u
#define ULCNET_MODEL_IO_ALIGNMENT      16u
#define ULCNET_MODEL_IO_MIN_D          2
#define ULCNET_MODEL_IO_MAX_D          64

#define ULCNET_MODEL_IO_TA_CHANNELS    32
#define ULCNET_MODEL_IO_TA_BINS        26
#define ULCNET_MODEL_IO_SCORE_HISTORY  4
#define ULCNET_MODEL_IO_GRU_LAYERS     2
#define ULCNET_MODEL_IO_GRU_HIDDEN     128

/* Stable values retained for metadata diagnostics. Production descriptors
 * validate only ULCNET_FAR_ALIGNED: raw/aligned selection belongs to the
 * offline sweep tool, not to the deployed pipeline API. */
typedef enum UlcnetFarInputMode {
    ULCNET_FAR_RAW     = 0,
    ULCNET_FAR_ALIGNED = 1
} UlcnetFarInputMode;

/* far_input_mode is stored as a plain int (not the enum type) so a
 * descriptor deserialized from ONNX/JSON metadata can hold an out-of-range
 * value and still be REJECTED by descriptor_validate() rather than being an
 * out-of-range enum object. */
typedef struct UlcnetModelIoDescriptor {
    uint32_t layout_version;
    int delay_depth;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int ta_channels;
    int ta_bins;
    int score_history_frames;
    int gru_layers;
    int gru_hidden;
    int far_input_mode;   /* a UlcnetFarInputMode value */
} UlcnetModelIoDescriptor;

typedef struct UlcnetModelIoMemReq {
    size_t bytes;
    size_t alignment;
} UlcnetModelIoMemReq;

/* Shapes use row-major ONNX order with the batch/time singleton dimensions
 * omitted from the pointer type:
 *   key/value history [1,32,D-1,26], newest frame first;
 *   logit history     [1,32,4,D], oldest frame first;
 *   GRU hidden        [2,1,128].
 */
typedef struct UlcnetModelIoInputs {
    const float *linear_error_ri;
    const float *far_end_ri;
    const float *key_history;
    const float *value_history;
    const float *logit_history;
    const float *gru0_hidden;
    const float *gru1_hidden;
    size_t spectrum_ri_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoInputs;

/* Accelerator-writable delta-state outputs:
 *   key/value now [1,32,1,26];
 *   logit now     [1,32,1,D];
 *   GRU next      [2,1,128].
 * prepare() fills every element with NaN so commit() detects partial writes.
 */
typedef struct UlcnetModelIoOutputs {
    float *enhanced_ri;
    float *key_now;
    float *value_now;
    float *logit_now;
    float *gru0_hidden_next;
    float *gru1_hidden_next;
    size_t spectrum_ri_elements;
    size_t key_now_elements;
    size_t value_now_elements;
    size_t logit_now_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoOutputs;

typedef struct UlcnetModelIoState UlcnetModelIoState;

/* Fill the fixed 16 kHz / 512 / 256 model ABI for the selected export-time D.
 * The deployed far branch is always ULCNET_FAR_ALIGNED.
 * Returns 0 on success, -1 for an unsupported D or NULL output. */
int ulcnet_model_io_descriptor_default(int delay_depth,
                                       UlcnetModelIoDescriptor *descriptor);

/* Validate a descriptor loaded from ONNX/JSON metadata against this C ABI.
 * far_input_mode must be ULCNET_FAR_ALIGNED. */
int ulcnet_model_io_descriptor_validate(
    const UlcnetModelIoDescriptor *descriptor);

/* Stable name of a far-input mode, identical to the exporter's metadata
 * string: "raw_far", "aligned_far", or "unknown" for any other value.
 * Deployment accepts only ULCNET_FAR_ALIGNED, so what this is for is telling
 * an integrator WHY a descriptor was rejected -- naming the mode the
 * checkpoint's metadata actually carried, including a value outside the
 * enum. The returned pointer is a string literal with static lifetime, so a
 * caller that has stdio can report it without this file (or either pipeline
 * wrapper) linking stdio itself. */
const char *ulcnet_far_input_mode_name(int mode);

/* Checked align-up shared with the accelerator adapter: rounds `value` up
 * to a multiple of `alignment` with overflow detection (returns nonzero on
 * overflow / zero alignment). The adapter must use this instead of a local
 * unchecked copy so every pool-sizing path carries the same guarantee. */
int ulcnet_model_io_align_up(size_t value, size_t alignment, size_t *out);

/* Query exact caller-pool size.  The pool address supplied to init() must be
 * aligned to req.alignment.  RAM scales with D; no D=64 maximum arrays are
 * retained for a D=4/D=8 model. */
int ulcnet_model_io_get_mem_requirements(
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetModelIoMemReq *requirements);

/* Construct inside caller memory.  Returns NULL for a bad descriptor,
 * unaligned/undersized pool or arithmetic overflow.  The state starts reset
 * (all history/hidden tensors are zero). */
UlcnetModelIoState *ulcnet_model_io_init(
    void *pool,
    size_t pool_bytes,
    const UlcnetModelIoDescriptor *descriptor);

void ulcnet_model_io_reset(UlcnetModelIoState *state);

/* Pack separate C real/imag spectra into ONNX [1,1,257,2], return current
 * input views, and NaN-prefill every accelerator output.  Call once
 * immediately before every inference. */
int ulcnet_model_io_prepare(UlcnetModelIoState *state,
                            const float error_re[257],
                            const float error_im[257],
                            const float far_re[257],
                            const float far_im[257],
                            UlcnetModelIoInputs *inputs,
                            UlcnetModelIoOutputs *outputs);

/* Validate that prepare() started a transaction and that the accelerator
 * wrote every output, then advance the K/V/logit rings, swap in next GRU
 * hidden tensors, and unpack enhanced RI to separate C arrays.  One prepare
 * permits one commit attempt.  On failure
 * persistent model state and caller outputs remain unchanged, the transaction
 * is discarded, and -1 is returned. */
int ulcnet_model_io_commit(UlcnetModelIoState *state,
                           float enhanced_re[257],
                           float enhanced_im[257]);

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_MODEL_IO_H */
