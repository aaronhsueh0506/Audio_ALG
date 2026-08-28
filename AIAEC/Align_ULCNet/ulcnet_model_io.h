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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version 3 fixes the deployed far branch to AEC-aligned far.  The exported
 * metadata separately records the checkpoint's training provenance.  Kept
 * numerically equal to export_onnx.py's STATE_LAYOUT_VERSION. */
/* Version 4 renamed the tensors and their mirrored fields (error/far
 * inputs, output head, h_gru0/h_gru1 hiddens, *_out states); runtimes
 * bind by name, so the rename is a contract change even though every
 * shape stayed identical. */
/* Version 5 moves the fixed front/back ends to the host: the graph binds
 * five feature inputs (error_mag/far_mag/error_cos/error_sin +
 * compressed error RI, all produced inside prepare()) and returns the
 * COMPRESSED estimate; commit() applies the inverse signed power. The
 * graph starts at the learned reorient/encoder compute. */
/* Version 8 presents every recurrent hidden as rank-4 NCHW -- h_gru0/h_gru1
 * become [1,2,1,128] and the combined tensor [1,4,1,128] -- so all five
 * boundary states share the one convention the attention caches already
 * used.  Nothing about this file's arithmetic moves with it: the hiddens
 * cross the boundary as flat float arrays whose element count
 * (GRU_LAYERS * GRU_HIDDEN) is identical at either rank, so compute_counts(),
 * the pool carve and prepare()/commit() are byte-for-byte what they were.
 * That is exactly why the version has to move.  descriptor_validate()
 * compares gru_layers, gru_hidden and every *_elements count, and every one
 * of them is unchanged between rank-3 and rank-4 -- so the version constant
 * is the ONLY thing that can stop a board built for the rank-3 boundary from
 * silently binding a rank-4 graph. */
/* ⚠ Versions 3-7 are RETIRED, not free.  3, 4 and 5 were shipped rank-3
 * boundaries and 6 and 7 were reserved for rank-3 pairs; a number that once
 * denoted a rank-3 boundary must never also denote a rank-4 one.
 * export_onnx.py's boundary is the pair (feature layout, GRU state layout)
 * and its LAYOUT_VERSIONS table now names all four: ('host','split') = 8,
 * the version below and the only pair this file implements;
 * ('host','combined') = 9 stacks both subband hiddens into one h_gru tensor;
 * ('graph','split') = 10 binds the two raw RI spectra and runs the
 * front/back ends inside the graph; ('graph','combined') = 11 does both.
 * Nothing here binds anything but 8, so a board built against this header
 * refuses the other three -- which is the intent.  The next real bump of
 * this constant must therefore go to 12. */
#define ULCNET_MODEL_IO_LAYOUT_VERSION 8u
#define ULCNET_MODEL_IO_ALIGNMENT      16u
#define ULCNET_MODEL_IO_MIN_D          2
#define ULCNET_MODEL_IO_MAX_D          64

#define ULCNET_MODEL_IO_TA_CHANNELS    32
#define ULCNET_MODEL_IO_TA_BINS        26
#define ULCNET_MODEL_IO_SCORE_HISTORY  4
#define ULCNET_MODEL_IO_GRU_LAYERS     2
#define ULCNET_MODEL_IO_GRU_HIDDEN     128

/* Modified power-law compression exponent (model.py compression_exponent).
 * Deployment contract, not an implementation detail: prepare()/commit() and
 * every tool touching compressed-domain tensors must use this exact value,
 * and export_onnx.py refuses checkpoints trained with any other exponent. */
#define ULCNET_MODEL_IO_COMPRESSION_EXP 0.3f

/* sign(x) * |x|^e, the single C copy of model.py's _signed_power. */
static inline float ulcnet_model_io_signed_pow(float value, float exponent) {
    return copysignf(powf(fabsf(value), exponent), value);
}

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
 *   GRU hidden        [1,2,1,128].
 * Those are the GRAPH shapes; the pointers below stay flat, and the
 * *_elements counts are what this file actually works in.
 */
typedef struct UlcnetModelIoInputs {
    /* The five feature tensors prepare() computes from the raw spectra
     * (model layout v5): magnitudes/cos/sin are [1,1,257] and error_ri is
     * the COMPRESSED [1,1,257,2]. */
    const float *error_mag;
    const float *far_mag;
    const float *error_cos;
    const float *error_sin;
    const float *error_ri;
    const float *key_history;
    const float *value_history;
    const float *logit_history;
    const float *h_gru0;
    const float *h_gru1;
    size_t spectrum_ri_elements;
    size_t spectrum_bins_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoInputs;

/* Accelerator-writable delta-state outputs:
 *   key/value now [1,32,1,26];
 *   logit now     [1,32,1,D];
 *   GRU next      [1,2,1,128].
 * prepare() fills every element with NaN so commit() detects partial writes.
 */
typedef struct UlcnetModelIoOutputs {
    float *output;
    float *key_now;
    float *value_now;
    float *logit_now;
    float *h_gru0_out;
    float *h_gru1_out;
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

/* Run the fixed front end (signed-power compression, magnitudes, phase
 * cos/sin) over the separate C real/imag spectra, return current input
 * views, and NaN-prefill every accelerator output.  Call once immediately
 * before every inference. commit() applies the matching inverse signed
 * power to the graph's compressed estimate. */
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
