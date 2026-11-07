/* Align-ULCNet stateless-accelerator model I/O: carves the feature tensors and mask staging from a caller pool, hands
 * out the frame's tensor views, checks and applies the graph's mask. The recurrent state is the application's: the
 * state tensors belong to the application (on a board, the accelerator's own buffers, loaded with quantized zeros at
 * start and whenever ulcnet_model_io_state_generation() changes) and the library knows none of their shapes. */
#ifndef ULCNET_MODEL_IO_H
#define ULCNET_MODEL_IO_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Deployed boundary version; equals LAYOUT_VERSION in export_onnx.py and is the only guard against binding a
 * different boundary with identical tensor names and counts; never reused. Row-major ONNX order, shapes without
 * batch/time, BINS the grid's n_bins; runtimes bind by name, so renaming a tensor is a contract change.
 *   inputs   error_mag, far_mag, error_cos, error_sin [1,1,BINS]; then, in this order and under these names, the
 *            state tensors key_history, value_history [1,C,D,K] (newest frame first), logit_history [1,C,W,D] (oldest
 *            first), h_gru [1,L,1,H] (subband-0 layers, then subband-1); D the delay depth, C K W L H the model's
 *   outputs  `output`, the complex mask planar [1,2,1,BINS] (real, then imaginary), and each state tensor's full next
 *            value as *_out; enhance_spectrum() applies the mask */
#define ULCNET_MODEL_IO_LAYOUT_VERSION 18u
#define ULCNET_MODEL_IO_ALIGNMENT      16u

/* The checkpoint's linear_aec.version (metadata linear_aec_contract_version): the front end the weights are trained
 * against, error input the AEC analysis seam (metadata error_input_seam = analysis_formed); the layout version does
 * not carry it. The integrator MAY compare it; C never reads it or refuses a graph by front end (Python enforces). */
#define ULCNET_MODEL_IO_FRONTEND_CONTRACT "aiaec-linear-error-v4"

/* Delay-depth bounds of the exporter, a sanity check only: D sizes nothing here. */
#define ULCNET_MODEL_IO_MIN_D          2
#define ULCNET_MODEL_IO_MAX_D          64

/* The grid and delay depth an instance runs, a run-time value: fill a UlcnetGridSpec with the model's export grid and
 * build the UlcnetGrid with ulcnet_grid_make(): hop = n_fft / 2, n_bins = n_fft / 2 + 1 (audiolib's rule). */
typedef struct UlcnetGridSpec {
    int sample_rate;
    int n_fft;
    int delay_depth;   /* D, as exported */
} UlcnetGridSpec;

typedef struct UlcnetGrid {
    int sample_rate;
    int n_fft;
    int hop;
    int n_bins;
    int delay_depth;
} UlcnetGrid;

enum {
    ULCNET_GRID_OK = 0,
    ULCNET_GRID_ERR_NULL = -1,
    ULCNET_GRID_ERR_SAMPLE_RATE = -2,
    ULCNET_GRID_ERR_N_FFT = -3,
    ULCNET_GRID_ERR_DELAY_DEPTH = -4,
    ULCNET_GRID_ERR_INCONSISTENT = -5
};

/* Derive a grid from the spec: ULCNET_GRID_OK (grid written), or the first failing check (grid untouched):
 * sample_rate > 0; n_fft a size audiolib's framing builds (ac_stft_get_mem_size() of the default configuration is
 * not 0); delay_depth within [ULCNET_MODEL_IO_MIN_D, ULCNET_MODEL_IO_MAX_D]. */
int ulcnet_grid_make(UlcnetGrid *grid, const UlcnetGridSpec *spec);

/* ULCNET_GRID_OK or the failing check: those of ulcnet_grid_make(), plus hop and n_bins equal to what it derives. */
int ulcnet_grid_validate(const UlcnetGrid *grid);

/* Power-law compression exponent (model.py compression_exponent); every tool on compressed-domain tensors uses it. */
#define ULCNET_MODEL_IO_COMPRESSION_EXP 0.3f

static inline float ulcnet_model_io_signed_pow(float value, float exponent) {
    return copysignf(powf(fabsf(value), exponent), value);
}

typedef enum UlcnetFarInputMode {
    ULCNET_FAR_RAW     = 0,
    ULCNET_FAR_ALIGNED = 1
} UlcnetFarInputMode;

/* The external contract of a graph. far_input_mode is an int so descriptor_validate() can reject out-of-range
 * metadata. The five model-interior fields (ta_channels .. gru_hidden) are the model's: every descriptor the library
 * builds has them 0, and it never reads, validates or sizes from them; ulcnet_model_io_init() keeps a direct caller's
 * descriptor as passed. Check a graph's descriptor with descriptor_validate(), never with memcmp(). */
typedef struct UlcnetModelIoDescriptor {
    uint32_t layout_version;
    int delay_depth;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int ta_channels;          /* model interior, never read here */
    int ta_bins;              /* model interior, never read here */
    int score_window_frames;  /* model interior, never read here */
    int gru_layers;           /* model interior, never read here */
    int gru_hidden;           /* model interior, never read here */
    int far_input_mode;       /* a UlcnetFarInputMode value */
} UlcnetModelIoDescriptor;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(UlcnetModelIoDescriptor) == 48,
               "UlcnetModelIoDescriptor must stay 48 bytes");
#endif
#endif

typedef struct UlcnetModelIoMemReq {
    size_t bytes;
    size_t alignment;
} UlcnetModelIoMemReq;

/* Optional registration of the application's recurrent state: borrowed float buffers (never allocated, freed or
 * resized here; they must outlive the instance) with element counts, not bytes. Either every pair is set or all are
 * NULL/0 (nothing registered); a pointer with a count of 0, or a count with a NULL pointer, is refused. */
typedef struct UlcnetModelIoStateBuffers {
    float *key_history;
    size_t key_history_elements;
    float *value_history;
    size_t value_history_elements;
    float *logit_history;
    size_t logit_history_elements;
    float *h_gru;
    size_t gru_hidden_elements;
} UlcnetModelIoStateBuffers;

/* Flat pointers, sizes in *_elements. The state views are the registered buffers: NULL and 0 elements when none. */
typedef struct UlcnetModelIoInputs {
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

/* `output` is planar: BINS real, then BINS imaginary. Each state *_out has the address of the input it continues
 * (NULL and 0 elements when none), fixed for the life of the state: the runtime reads all state inputs before writing
 * any state output. prepare() NaN-fills `output`, so an unwritten mask is refused by enhance_spectrum(); a refused
 * frame zeroes the registered state, advances the generation and is discarded. */
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

/* Fill the external fields for `grid` (interior fields 0, far input raw). 0, or -1 on NULL or an invalid grid. */
int ulcnet_model_io_descriptor_default(const UlcnetGrid *grid,
                                       UlcnetModelIoDescriptor *descriptor);

/* Validate a descriptor against `grid`: layout version, sample_rate, fft_size, hop_size and spectrum_bins equal to
 * the grid's, far_input_mode == ULCNET_FAR_RAW, delay_depth within the exporter's bounds (not compared with
 * grid->delay_depth: the export depth is the application's); interior fields not read. 0 if valid, -1 otherwise. */
int ulcnet_model_io_descriptor_validate(
    const UlcnetGrid *grid, const UlcnetModelIoDescriptor *descriptor);

/* Derive the grid from a descriptor: 0 and *grid written if it validates against that grid, else -1 (untouched). */
int ulcnet_model_io_descriptor_grid(const UlcnetModelIoDescriptor *descriptor,
                                    UlcnetGrid *grid);

/* Metadata string of a far-input mode ("raw_far", "aligned_far" or "unknown"); a static literal. */
const char *ulcnet_far_input_mode_name(int mode);

/* 0 for a valid registration (NULL counts as nothing registered), -1 otherwise. */
int ulcnet_model_io_state_buffers_validate(
    const UlcnetModelIoStateBuffers *buffers);

/* Exact caller-pool size: the state header, the feature tensors and the mask staging, each 16-byte aligned; a
 * function of the grid, not of D. The pool address given to init() must be aligned to req.alignment. */
int ulcnet_model_io_get_mem_requirements(
    const UlcnetModelIoDescriptor *descriptor,
    UlcnetModelIoMemReq *requirements);

/* Construct inside caller memory; `state` registers the application's recurrent state (NULL registers nothing) and
 * the registered buffers are zeroed. NULL for a bad descriptor or registration, an unaligned or undersized pool,
 * or overflow; a refused call touches neither the pool nor the buffers. The generation starts at 0. */
UlcnetModelIoState *ulcnet_model_io_init(
    void *pool,
    size_t pool_bytes,
    const UlcnetModelIoDescriptor *descriptor,
    const UlcnetModelIoStateBuffers *state);

/* Zero the registered state tensors (none registered: nothing to zero), drop the open transaction and advance the
 * generation. The pool, the registration and every pointer stay. NULL is ignored. */
void ulcnet_model_io_reset(UlcnetModelIoState *state);

/* Run the fixed front end (signed-power compression, magnitudes, phase cos/sin) on the real/imag spectra
 * (spectrum_bins each), return the input views, NaN-prefill `output`. Call once right before every inference. */
int ulcnet_model_io_prepare(UlcnetModelIoState *state,
                            const float *error_re, const float *error_im,
                            const float *far_re, const float *far_im,
                            UlcnetModelIoInputs *inputs,
                            UlcnetModelIoOutputs *outputs);

/* Call after a successful inference; one prepare() permits one call. Checks that the accelerator wrote a finite mask
 * and, when state is registered, finite state (every registered buffer scanned whole), then applies the mask and the
 * inverse signed power into the enhanced arrays. On a non-finite value the registered state is zeroed, the generation
 * advances, the caller's outputs stay untouched and -1 is returned. */
int ulcnet_model_io_enhance_spectrum(UlcnetModelIoState *state,
                                     float *enhanced_re, float *enhanced_im);

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state);

/* Counter of state restarts: advances on reset and when enhance_spectrum() refuses a frame, registered state or not;
 * 0 for a fresh or NULL state; wraps at 2^32. A fixed-point accelerator (or any runtime holding the state itself)
 * loads quantized zeros into its state at start and on every change. */
uint32_t ulcnet_model_io_state_generation(const UlcnetModelIoState *state);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_MODEL_IO_H */
