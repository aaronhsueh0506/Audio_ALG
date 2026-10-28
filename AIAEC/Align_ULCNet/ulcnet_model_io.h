/* Align-ULCNet stateless-accelerator model I/O state. The accelerator owns no
 * persistent state: this helper carves the K/V windows, the score-convolution
 * window and the temporal-GRU hidden tensor from one caller-provided pool. The
 * graph takes each state tensor as a full window and returns its next value as
 * *_out, bound to the address of its input (the runtime reads all state inputs
 * before writing any state output). No accelerator call and no STFT/WOLA here. */
#ifndef ULCNET_MODEL_IO_H
#define ULCNET_MODEL_IO_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Deployed boundary version; equals LAYOUT_VERSION in export_onnx.py and is the
 * only guard against binding a different boundary with identical tensor names
 * and counts. Version numbers are never reused.
 * Boundary layout (row-major ONNX order, shapes listed without batch/time):
 *   - inputs: error_mag, far_mag, error_cos, error_sin (each [1,1,BINS], from
 *     prepare()) and the state tensors
 *       key_history   [1,32,D,TA_BINS]  newest frame first
 *       value_history [1,32,D,TA_BINS]  newest frame first
 *       logit_history [1,32,5,D]        oldest frame first
 *       h_gru         [1,4,1,128]       subband-0 layers, then subband-1;
 *   - outputs: `output`, the complex mask, planar [1,2,1,BINS] (real, then
 *     imaginary), and the full next value of each state tensor as *_out;
 *   - the graph ends at the complex_mask 1x1 Conv; commit() applies the mask.
 * Runtimes bind by name, so renaming a tensor is a contract change. */
#define ULCNET_MODEL_IO_LAYOUT_VERSION 18u
#define ULCNET_MODEL_IO_ALIGNMENT      16u

/* Front end the weights are trained against: the checkpoint's
 * linear_aec.version, written into the graph metadata (JSON / ONNX props) as
 * linear_aec_contract_version by export_onnx.py. The layout version above
 * describes only the tensor/state boundary and stays the same across
 * retrained checkpoints; this string is what tells an old graph from a current
 * one. The integrator compares the metadata value against it before
 * ulcnet_accelerator_adapter_init and refuses a graph that differs. Its
 * error input is the AEC analysis seam (formed_hop / beam.spectrum), the real
 * selected linear residual, never the rendering base (aec_get_output_base);
 * the metadata records this as error_input_seam = analysis_formed. */
#define ULCNET_MODEL_IO_FRONTEND_CONTRACT "aiaec-linear-error-v4"
#define ULCNET_MODEL_IO_MIN_D          2
#define ULCNET_MODEL_IO_MAX_D          64

/* Deployment grid (build parameter): one build serves one grid. Override with
 * -DULCNET_MODEL_IO_SR / -DULCNET_MODEL_IO_N_FFT (48000 / 1024 for the other
 * grid). HOP, BINS and TA_BINS = ceil(BINS/10), the temporal-attention K/V
 * width (26 at 16 kHz, 52 at 48 kHz), are derived. */
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
#define ULCNET_MODEL_IO_SCORE_WINDOW   5
/* Two subband GRUs of GRU_LAYERS layers share h_gru [1, GRU_SUBBANDS * GRU_LAYERS, 1, GRU_HIDDEN]. */
#define ULCNET_MODEL_IO_GRU_SUBBANDS   2
#define ULCNET_MODEL_IO_GRU_LAYERS     2
#define ULCNET_MODEL_IO_GRU_HIDDEN     128

/* Power-law compression exponent (model.py compression_exponent). Every tool
 * touching compressed-domain tensors must use it; export_onnx.py refuses other
 * exponents. */
#define ULCNET_MODEL_IO_COMPRESSION_EXP 0.3f

/* sign(x) * |x|^e (model.py _signed_power). */
static inline float ulcnet_model_io_signed_pow(float value, float exponent) {
    return copysignf(powf(fabsf(value), exponent), value);
}

/* Far-input modes. Production descriptors validate only ULCNET_FAR_RAW. */
typedef enum UlcnetFarInputMode {
    ULCNET_FAR_RAW     = 0,
    ULCNET_FAR_ALIGNED = 1
} UlcnetFarInputMode;

/* far_input_mode is a plain int so out-of-range values from ONNX/JSON metadata
 * can be rejected by descriptor_validate(). */
typedef struct UlcnetModelIoDescriptor {
    uint32_t layout_version;
    int delay_depth;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int ta_channels;
    int ta_bins;
    int score_window_frames;
    int gru_layers;
    int gru_hidden;
    int far_input_mode;   /* a UlcnetFarInputMode value */
} UlcnetModelIoDescriptor;

typedef struct UlcnetModelIoMemReq {
    size_t bytes;
    size_t alignment;
} UlcnetModelIoMemReq;

/* Pointers are flat; the *_elements counts give the tensor sizes. */
typedef struct UlcnetModelIoInputs {
    /* Feature tensors computed by prepare(), each [1,1,BINS]. */
    const float *error_mag;
    const float *far_mag;
    const float *error_cos;
    const float *error_sin;
    const float *key_history;
    const float *value_history;
    const float *logit_history;
    const float *h_gru;
    size_t spectrum_bins_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoInputs;

/* The graph's mask and the next value of every state tensor. `output` is
 * planar: BINS real then BINS imaginary values. Each *_out has the address of
 * the input it continues and is fixed for the life of the state. prepare()
 * NaN-fills `output`, so an unwritten mask is refused at commit. A refused
 * commit zeroes all four state tensors, increments
 * ulcnet_model_io_state_generation() and discards the frame. */
typedef struct UlcnetModelIoOutputs {
    float *output;
    float *key_history_out;
    float *value_history_out;
    float *logit_history_out;
    float *h_gru_out;
    size_t spectrum_ri_elements;
    size_t key_history_elements;
    size_t value_history_elements;
    size_t logit_history_elements;
    size_t gru_hidden_elements;
} UlcnetModelIoOutputs;

typedef struct UlcnetModelIoState UlcnetModelIoState;

/* Fill the compiled model ABI for the export-time delay depth D. The far branch
 * is always ULCNET_FAR_RAW. Returns 0, or -1 for an unsupported D or NULL output. */
int ulcnet_model_io_descriptor_default(int delay_depth,
                                       UlcnetModelIoDescriptor *descriptor);

/* Validate a descriptor loaded from ONNX/JSON metadata against this ABI;
 * far_input_mode must be ULCNET_FAR_RAW. Returns 0 if valid. */
int ulcnet_model_io_descriptor_validate(
    const UlcnetModelIoDescriptor *descriptor);

/* Metadata string of a far-input mode: "raw_far", "aligned_far" or "unknown".
 * Static string literal. */
const char *ulcnet_far_input_mode_name(int mode);

/* Checked align-up: rounds `value` up to a multiple of `alignment`; returns
 * nonzero on overflow or zero alignment. */
int ulcnet_model_io_align_up(size_t value, size_t alignment, size_t *out);

/* Exact caller-pool size. The pool address passed to init() must be aligned to
 * req.alignment. RAM scales with D. */
int ulcnet_model_io_get_mem_requirements(
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetModelIoMemReq *requirements);

/* Construct inside caller memory. Returns NULL for a bad descriptor, an
 * unaligned or undersized pool, or overflow. The state starts reset (zeroed). */
UlcnetModelIoState *ulcnet_model_io_init(
    void *pool,
    size_t pool_bytes,
    const UlcnetModelIoDescriptor *descriptor);

void ulcnet_model_io_reset(UlcnetModelIoState *state);

/* Run the fixed front end (signed-power compression, magnitudes, phase cos/sin)
 * on the separate real/imag spectra, return the input views and NaN-prefill the
 * `output` mask. Call once immediately before every inference. */
int ulcnet_model_io_prepare(UlcnetModelIoState *state,
                            const float error_re[ULCNET_MODEL_IO_BINS],
                            const float error_im[ULCNET_MODEL_IO_BINS],
                            const float far_re[ULCNET_MODEL_IO_BINS],
                            const float far_im[ULCNET_MODEL_IO_BINS],
                            UlcnetModelIoInputs *inputs,
                            UlcnetModelIoOutputs *outputs);

/* Validate that prepare() started a transaction and that the accelerator wrote
 * a finite mask and state (only the newest frame of each window plus the GRU
 * hidden are checked), apply the mask and inverse signed power, and write the
 * enhanced real/imag arrays. One prepare permits one commit. On a non-finite
 * value all state tensors are zeroed, the state generation advances, the
 * caller's outputs are untouched and -1 is returned. */
int ulcnet_model_io_commit(UlcnetModelIoState *state,
                           float enhanced_re[ULCNET_MODEL_IO_BINS],
                           float enhanced_im[ULCNET_MODEL_IO_BINS]);

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state);

/* Counter of state restarts: increments on ulcnet_model_io_reset() and when
 * commit() refuses a frame; 0 for a fresh state. A fixed-point accelerator
 * refills its state with quantized zeros when it changes. 0 for a NULL state;
 * wraps at 2^32. */
uint32_t ulcnet_model_io_state_generation(const UlcnetModelIoState *state);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_MODEL_IO_H */
