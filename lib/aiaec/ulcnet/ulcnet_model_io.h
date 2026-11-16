/* Align-ULCNet stateless-accelerator model I/O: carves the feature tensors and mask staging from a caller pool and
 * provides the per-frame functions (features, views, run validation, mask application). The recurrent state is the
 * application's: the state tensors belong to the application (on a board, the accelerator's own buffers, loaded with quantized zeros at
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
 *            value as *_out; post_frame() applies the mask */
#define ULCNET_MODEL_IO_LAYOUT_VERSION 18u

/* The packet boundary (frames_per_call N in [2, ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL]); equals PACKET_LAYOUT_VERSION in
 * export_onnx.py and is a different number from the one-frame boundary, never interchangeable with it:
 *   inputs   error_mag, far_mag, error_cos, error_sin [1,N,BINS] (frame-major, oldest first), frame_reset [1,N]
 *            (0.0 or 1.0 per frame), then the same four state tensors as layout 18
 *   outputs  `output` [1,2,N,BINS]: the real plane of all N frames, then the imaginary plane; then the *_out states
 * A frame_reset frame resets the model state to zero BEFORE that frame and bypasses it inside the graph (identity mask,
 * model not stepped, state stays zero); the frames after it continue from the reset state. The host therefore never
 * restarts the state for such a frame (no generation change) and fills the frame itself with the error spectrum. */
#define ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION 19u
#define ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL   64
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
    int frames_per_call;      /* 0 or 1: the one-frame boundary (layout 18); 2..64: the packet boundary (layout 19) */
} UlcnetModelIoDescriptor;

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

/* Flat pointers, sizes in *_elements. The state views are the registered buffers: NULL and 0 elements when none. In the
 * packet boundary the four signal tensors hold N frames (spectrum_bins_elements = N * BINS each) and frame_reset points
 * at the N control values; both are NULL / 0 in the one-frame boundary. */
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
    const float *frame_reset;
    size_t frame_reset_elements;
} UlcnetModelIoInputs;

/* `output` is planar: BINS real, then BINS imaginary (packet: N * BINS real, then N * BINS imaginary). Each state *_out sits at
 * the address of the input it continues (NULL and 0 when none), fixed for the state's life; the runtime reads all state inputs
 * before writing any output. views() NaN-fills it, so validate_run() refuses an unwritten mask and zeroes the state (generation +1). */
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

/* As _default for the packet boundary: layout ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION, frames_per_call set. -1 (also) for
 * a frames_per_call outside [2, ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL]. */
int ulcnet_model_io_descriptor_default_packet(const UlcnetGrid *grid, int frames_per_call,
                                              UlcnetModelIoDescriptor *descriptor);

/* Frames one graph call runs: 1 for frames_per_call 0 or 1, else the value; 0 for NULL. */
int ulcnet_model_io_descriptor_frames(const UlcnetModelIoDescriptor *descriptor);

/* Validate a descriptor against `grid`: layout version, sample_rate, fft_size, hop_size and spectrum_bins equal to
 * the grid's, far_input_mode == ULCNET_FAR_RAW, delay_depth within the exporter's bounds (not compared with
 * grid->delay_depth: the export depth is the application's); interior fields not read. The layout and the frame count
 * must agree: LAYOUT_VERSION with 1 frame, PACKET_LAYOUT_VERSION with 2..MAX_FRAMES_PER_CALL. 0 if valid, -1
 * otherwise. */
int ulcnet_model_io_descriptor_validate(
    const UlcnetGrid *grid, const UlcnetModelIoDescriptor *descriptor);

/* Derive the grid from a descriptor: 0 and *grid written if it validates against that grid, else -1 (untouched). */
int ulcnet_model_io_descriptor_grid(const UlcnetModelIoDescriptor *descriptor,
                                    UlcnetGrid *grid);

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

/* Zero the registered state tensors (none registered: nothing to zero) and advance the generation. The pool, the
 * registration and every pointer stay. NULL is ignored. */
void ulcnet_model_io_reset(UlcnetModelIoState *state);

/* Features of frame `slot` (0 <= slot < frames): the four spectra (spectrum_bins floats, read here, not kept) into that
 * slot of the feature tensors, the compressed error kept for post_frame(), and frame_reset[slot] = event != 0 (event
 * must be 0 at one frame). Call it for every slot before views(). 0; -1 on NULL, a bad slot or event (nothing written). */
int ulcnet_model_io_pre_frame(UlcnetModelIoState *state, int slot,
                              const float *error_re, const float *error_im,
                              const float *far_re, const float *far_im, int event);

/* Once per run, after the last pre_frame: NaN-fill `output` (an unwritten mask fails validate_run()) and publish the
 * tensor views. Repeating it gives a clean NaN mask again. 0; -1 on NULL. */
int ulcnet_model_io_views(UlcnetModelIoState *state, UlcnetModelIoInputs *inputs,
                          UlcnetModelIoOutputs *outputs);

/* After a successful run, `output` (every frame) and every registered state buffer must be finite. 0 accepted; -1
 * refused: the registered state is zeroed and the generation advanced once (NULL: -1, nothing touched). */
int ulcnet_model_io_validate_run(UlcnetModelIoState *state);

/* Frame `slot` of the run: compressed error times its complex mask, inverse signed power, into out_re / out_im
 * (spectrum_bins floats each). 0; -1 on NULL or a bad slot (out untouched); -2 on a non-finite enhanced bin. State and
 * generation are never touched: the caller restarts once per run. */
int ulcnet_model_io_post_frame(const UlcnetModelIoState *state, int slot, float *out_re, float *out_im);

const UlcnetModelIoDescriptor *ulcnet_model_io_descriptor(
    const UlcnetModelIoState *state);

/* Counter of state restarts: advances on reset and when validate_run() refuses a run, registered state or not;
 * 0 for a fresh or NULL state; wraps at 2^32. A fixed-point accelerator (or any runtime holding the state itself)
 * loads quantized zeros into its state at start and on every change. */
uint32_t ulcnet_model_io_state_generation(const UlcnetModelIoState *state);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_MODEL_IO_H */
