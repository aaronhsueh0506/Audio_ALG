/* Align-ULCNet stateless-accelerator model I/O state.
 *
 * The accelerator owns no persistent state.  This helper carves K/V history,
 * score-convolution history and temporal-GRU hidden tensors from one
 * caller-provided pool.  The graph returns the FULL next value of all five
 * state tensors (key_history, value_history, logit_history, h_gru0, h_gru1)
 * as *_out, with the ring shift done inside it, so the CPU never shifts a
 * ring.  Two bindings share everything else (prepare, commit, adapter,
 * pipeline):
 *   - ordinary runtime: its own output tensors, handed to
 *     ulcnet_model_io_inherit(), which validates and copies them into the
 *     state;
 *   - runtime that can bind each *_out to its input's address: it writes the
 *     state in place and the inherit call is simply left out.
 * commit() validates the frame in both cases.
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

/* The deployed boundary and its version.  This constant is the ONLY thing
 * that stops a board built for one boundary from silently binding another:
 * descriptor_validate() compares counts, and the tensor names and element
 * counts of different boundaries coincide.  It is kept numerically equal to
 * export_onnx.py's STATE_LAYOUT_VERSION, whose LAYOUT_VERSIONS table names
 * the four (feature layout, GRU state layout) pairs:
 *   ('host','split')    = 16  the only pair this file implements
 *   ('host','combined') = 17  both subband hiddens stacked into one h_gru
 *   ('graph','split')   = 14  binds the raw RI spectra and runs the fixed
 *   ('graph','combined')= 15  front/back ends (and the mask) in the graph
 * A board built against this header refuses the other three.  3-13 are
 * RETIRED, not free: a number that once denoted one boundary must never
 * denote another.  The next real bump of this constant goes to 18.
 *
 * The version-16 boundary:
 *   - inputs: error_mag, far_mag, error_cos, error_sin (each [1,1,BINS],
 *     computed by prepare()), the K/V/logit histories and the two GRU hiddens;
 *   - outputs: `output`, the learned complex mask, planar [1,2,1,BINS]
 *     (plane 0 real, plane 1 imaginary), and the FULL next value of all five
 *     state tensors as *_out, so each is bound in place;
 *   - the graph ends at the complex_mask 1x1 Conv.  prepare() keeps the
 *     compressed error spectrum, and commit() multiplies it by the mask and
 *     applies the inverse signed power; the graph carries no elementwise tail.
 * Runtimes bind by name, so a rename is a contract change too. */
#define ULCNET_MODEL_IO_LAYOUT_VERSION 16u
#define ULCNET_MODEL_IO_ALIGNMENT      16u
#define ULCNET_MODEL_IO_MIN_D          2
#define ULCNET_MODEL_IO_MAX_D          64

/* ---- Deployment grid (build parameter) ----------------------------------
 * One build serves ONE grid: the analysis/synthesis structs in
 * ulcnet_process.h carry their buffers by value, so the sizes have to be
 * compile-time. Override with -DULCNET_MODEL_IO_SR / -DULCNET_MODEL_IO_N_FFT
 * to build for the other product grid (48000 / 1024).
 *
 * Only those TWO are settable. HOP, BINS and TA_BINS are DERIVED, because a
 * grid whose hop is not N_FFT/2 or whose bin count is not N_FFT/2+1 is not a
 * grid this model has -- letting them be set independently would make an
 * inconsistent combination expressible, and it would compile.
 *
 * TA_BINS is the temporal-attention K/V feature width. It is NOT a fixed
 * model constant: the C-SamFR reorientation widens to ceil(BINS/(gamma*
 * subband_bins))*subband_bins and the encoder's (1,2) ceil-mode pool halves
 * it, which for the (gamma, subband_bins) = (5, 2) pair the exporter enforces
 * is exactly ceil(BINS/10). 26 at 16 kHz, 52 at 48 kHz. The exporter derives
 * the same number from the model itself (ta_bins_for) and
 * test_c_descriptor_constants_match_export_contract pins the two together. */
#ifndef ULCNET_MODEL_IO_SR
#define ULCNET_MODEL_IO_SR             16000
#endif
#ifndef ULCNET_MODEL_IO_N_FFT
#define ULCNET_MODEL_IO_N_FFT          512
#endif
#if !((ULCNET_MODEL_IO_SR == 16000 && ULCNET_MODEL_IO_N_FFT == 512) || \
      (ULCNET_MODEL_IO_SR == 48000 && ULCNET_MODEL_IO_N_FFT == 1024))
#error "Align-ULCNet supports only 16000/512 or 48000/1024"
#endif
#define ULCNET_MODEL_IO_HOP            (ULCNET_MODEL_IO_N_FFT / 2)
#define ULCNET_MODEL_IO_BINS           (ULCNET_MODEL_IO_N_FFT / 2 + 1)
#define ULCNET_MODEL_IO_TA_BINS        ((ULCNET_MODEL_IO_BINS + 9) / 10)

#define ULCNET_MODEL_IO_TA_CHANNELS    32
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
 * validate only ULCNET_FAR_RAW: raw/aligned selection belongs to the
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
 *   key/value history [1,32,D-1,TA_BINS], newest frame first;
 *   logit history     [1,32,4,D], oldest frame first;
 *   GRU hidden        [1,2,1,128].
 * Those are the GRAPH shapes; the pointers below stay flat, and the
 * *_elements counts are what this file actually works in.  The five state
 * tensors below are the ones the graph also returns, in full, as *_out.
 */
typedef struct UlcnetModelIoInputs {
    /* The four feature tensors prepare() computes from the raw spectra
     * (model layout v5), each [1,1,BINS].  The compressed error spectrum is
     * not among them (layout v16): only commit() reads it, so the state
     * keeps it and the graph does not bind it. */
    const float *error_mag;
    const float *far_mag;
    const float *error_cos;
    const float *error_sin;
    const float *key_history;
    const float *value_history;
    const float *logit_history;
    const float *h_gru0;
    const float *h_gru1;
    size_t spectrum_bins_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoInputs;

/* The graph's mask and the full next value of every state tensor (same
 * shapes as the inputs above).  `output` is the learned complex mask, planar:
 * BINS real values then BINS imaginary values; commit() applies it.
 * prepare() returns these pointing into the
 * pool: each *_out is the SAME address as the input it continues and stays
 * fixed for the life of the state.  A runtime with its own output tensors
 * fills a struct of this type with them and calls ulcnet_model_io_inherit();
 * a runtime that binds the pointers prepare() returned writes in place (it
 * reads every state input before it writes any state output).  prepare()
 * NaN-fills `output`, so an unwritten mask is refused at commit; a
 * partial write to the state tensors is not detectable, only a non-finite
 * one.
 *
 * Failure: on the copy path a refused inherit leaves the state as it was, the
 * accelerator callback returns non-zero (it must not report success after a
 * refusal) and the frame is skipped; bound in place,
 * a refused commit restarts all five state tensors from zero, and a frame the
 * runtime ran but the caller does not commit leaves them as the runtime wrote
 * them.
 */
typedef struct UlcnetModelIoOutputs {
    float *output;
    float *key_history_out;
    float *value_history_out;
    float *logit_history_out;
    float *h_gru0_out;
    float *h_gru1_out;
    size_t spectrum_ri_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoOutputs;

typedef struct UlcnetModelIoState UlcnetModelIoState;

/* Fill the compiled deployment-grid model ABI for the selected export-time D.
 * The deployed far branch is always ULCNET_FAR_RAW, matching training and the
 * model's own time-alignment attention input.
 * Returns 0 on success, -1 for an unsupported D or NULL output. */
int ulcnet_model_io_descriptor_default(int delay_depth,
                                       UlcnetModelIoDescriptor *descriptor);

/* Validate a descriptor loaded from ONNX/JSON metadata against this C ABI.
 * far_input_mode must be ULCNET_FAR_RAW. */
int ulcnet_model_io_descriptor_validate(
    const UlcnetModelIoDescriptor *descriptor);

/* Stable name of a far-input mode, identical to the exporter's metadata
 * string: "raw_far", "aligned_far", or "unknown" for any other value.
 * Deployment accepts only ULCNET_FAR_RAW, so what this is for is telling
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
 * views, and NaN-prefill the `output` mask (the state tensors are the
 * inputs, so they are left alone).  Call once immediately before every
 * inference. commit() multiplies the compressed error kept here by the
 * graph's mask and applies the matching inverse signed power. */
int ulcnet_model_io_prepare(UlcnetModelIoState *state,
                            const float error_re[ULCNET_MODEL_IO_BINS],
                            const float error_im[ULCNET_MODEL_IO_BINS],
                            const float far_re[ULCNET_MODEL_IO_BINS],
                            const float far_im[ULCNET_MODEL_IO_BINS],
                            UlcnetModelIoInputs *inputs,
                            UlcnetModelIoOutputs *outputs);

/* Copy path for a runtime that cannot bind a state output to its input's
 * address (the ordinary case: an ONNX-style runtime with its own output
 * tensors).  `destination` is the struct prepare() returned; `runtime` has the
 * same type and carries the runtime's own tensor pointers and the same
 * element counts.  Each tensor is copied into the matching destination
 * pointer, except one whose pointer already equals it -- the runtime wrote
 * that one in place -- which is left alone, so one call serves both bindings.
 * The tensors about to be copied are checked first (mask, newest K/V
 * slot, last logit frame, both hiddens): if any is non-finite nothing is
 * written, -1 is returned and the state is exactly as it was, so the caller
 * reports the run as failed and the frame is skipped.  A tensor written in
 * place cannot be restored on a bad frame; it is checked by commit(), which
 * restarts the state from zero.  The runtime's tensors are its own: this
 * file allocates nothing for them.  Returns 0, or -1 on NULL, a missing
 * tensor or mismatched counts. */
int ulcnet_model_io_inherit(const UlcnetModelIoOutputs *destination,
                            const UlcnetModelIoOutputs *runtime);

/* Validate that prepare() started a transaction and that the accelerator
 * wrote a finite mask and finite state, multiply the compressed error by the
 * mask, apply the inverse signed power and unpack enhanced RI to separate C
 * arrays; the state tensors need no step.  One prepare permits one
 * commit attempt.  Of the history rings only the frame the graph just wrote
 * is checked (key/value slot 0, the last logit frame): the older slots were
 * checked when they were new, and a non-finite value left in any of them
 * reaches `output` on the next frame, where it is refused.  A non-finite
 * value cannot be rolled back, because the state was written in place: all
 * five state tensors are zeroed, the transaction is discarded, the caller's
 * outputs are untouched, and -1 is returned. */
int ulcnet_model_io_commit(UlcnetModelIoState *state,
                           float enhanced_re[ULCNET_MODEL_IO_BINS],
                           float enhanced_im[ULCNET_MODEL_IO_BINS]);

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_MODEL_IO_H */
