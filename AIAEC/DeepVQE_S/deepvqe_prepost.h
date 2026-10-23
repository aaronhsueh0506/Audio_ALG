/* DeepVQE-S pre/post-processing class: one object owning everything between the
 * caller's audio and the accelerator's tensors.
 *
 *     deepvqe_prepost_pre_process(p, mic_hop, far_hop);
 *     deepvqe_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *     my_npu_run(&in, &out);                        // NPU fills `out`
 *     deepvqe_prepost_frame_commit(p);              // or _frame_skip()
 *     deepvqe_prepost_post_process(p, out_hop, &written);
 *
 * One hop in, one inference, one hop out, in both I/O modes. Analysis is a
 * rolling window (AcStft, center 0), one frame per hop. The primary input is the
 * raw microphone with no canceller in front, so frame_skip() fails closed
 * (silence, not pass-through; deepvqe_prepost_skip_policy_name()). Grid: 16 kHz /
 * 512 / 256 only. The FftHandle and window are caller-owned (DEEPVQE_IO_TIME). */

#ifndef DEEPVQE_PREPOST_H
#define DEEPVQE_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_process.h"   /* DeepVqeCcmState, DEEPVQE_TIME_ORDER, DEEPVQE_FREQ_TAPS, aiaec_process.h */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_PREPOST_DESCRIPTOR_VERSION 1u

/* Accelerator boundary version (two planar signal inputs, the CCM-tap head, the
 * sixteen rank-4 state tensors of _streaming_export.py, each returned in full).
 * Any rename, reshape, reorder or added/removed tensor bumps it. */
#define DEEPVQE_PREPOST_LAYOUT_VERSION 4u

/* Input compression exponent (model.compression_exponent); the exporter refuses others. */
#define DEEPVQE_COMPRESSION_EXP 0.3f
#define DEEPVQE_PREPOST_ALIGNMENT      16u
/* Folded into build_flags_hash; bumped on any change of pool size, carve order or descriptor. */
#define DEEPVQE_PREPOST_CARVE_VERSION  5u

/* Alignment search depth D (exporter max_delay_frames); sizes the attention rings. 32 = 0.5 s. */
#define DEEPVQE_PREPOST_MIN_D      1
#define DEEPVQE_PREPOST_MAX_D      256
#define DEEPVQE_PREPOST_DEFAULT_D  32

/* ---- DeepVQE-S topology constants (mirror DeepVQE_S/model.py) ----
 * Fixed by the checkpoint; the state shapes are built from them. */
#define DEEPVQE_HALF_(n)        (((n) + 1) / 2)
#define DEEPVQE_F0              AIAEC_N_BINS              /* 257 */
#define DEEPVQE_F1              DEEPVQE_HALF_(DEEPVQE_F0) /* 129 */
#define DEEPVQE_F2              DEEPVQE_HALF_(DEEPVQE_F1) /*  65 */
#define DEEPVQE_F3              DEEPVQE_HALF_(DEEPVQE_F2) /*  33 */
#define DEEPVQE_F4              DEEPVQE_HALF_(DEEPVQE_F3) /*  17 */

/* Causal conv window 4; the score conv (5,3) kernel gives window 5. Oldest frame first. */
#define DEEPVQE_CONV_WINDOW     4
#define DEEPVQE_SCORE_WINDOW    5
#define DEEPVQE_SIM_CHANNELS    4     /* FrameDelayAttention similarity */
#define DEEPVQE_VALUE_CHANNELS  24    /* FrameDelayAttention value */
#define DEEPVQE_GRU_LAYERS      1
#define DEEPVQE_GRU_HIDDEN      192

/* Head output: CCM taps [1,1,BINS,TIME_ORDER*FREQ_TAPS*2], last axis [time][frequency][RI]. */
#define DEEPVQE_TAPS_ELEMENTS \
    ((size_t)AIAEC_N_BINS * DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2u)

/* ---- Explicit state boundary ----
 * Graph input order after the two signal inputs (outputs: after the head). */
typedef enum DeepVqeStateId {
    DEEPVQE_STATE_ALIGN_KEY_RING = 0,
    DEEPVQE_STATE_ALIGN_VALUE_RING,
    DEEPVQE_STATE_ALIGN_SCORE_HISTORY,
    DEEPVQE_STATE_CCM_UP_HISTORY,
    DEEPVQE_STATE_FAR1_HISTORY,
    DEEPVQE_STATE_FAR2_HISTORY,
    DEEPVQE_STATE_H_GRU,
    DEEPVQE_STATE_MIC1_HISTORY,
    DEEPVQE_STATE_MIC2_HISTORY,
    DEEPVQE_STATE_MIC3_HISTORY,
    DEEPVQE_STATE_MIC4_HISTORY,
    DEEPVQE_STATE_RES2_HISTORY,
    DEEPVQE_STATE_RES3_HISTORY,
    DEEPVQE_STATE_UP1_HISTORY,
    DEEPVQE_STATE_UP2_HISTORY,
    DEEPVQE_STATE_UP3_HISTORY,
    DEEPVQE_STATE_COUNT
} DeepVqeStateId;

/* Rank of every state tensor (NCHW). */
#define DEEPVQE_STATE_MAX_RANK 4

/* Which side of the transform the caller works on; fixed at init because it
 * decides the pool size (DEEPVQE_IO_TIME also carves framing states). */
typedef enum DeepVqeIoMode {
    /* Spectrum in, spectrum out; fft/window may be NULL. */
    DEEPVQE_IO_FREQ = 0,
    /* Hop in, hop out; the class frames, windows and overlap-adds via the caller's FftHandle. */
    DEEPVQE_IO_TIME = 1
} DeepVqeIoMode;

typedef struct DeepVqePrepostConfig {
    int          io_mode;      /* a DeepVqeIoMode                           */
    int          delay_depth;  /* D; DEEPVQE_PREPOST_MIN/MAX_D              */
    FftHandle   *fft;          /* DEEPVQE_IO_TIME: required, borrowed       */
    const float *window;       /* DEEPVQE_IO_TIME: required, borrowed; AIAEC_N_FFT entries from ac_window_build() */
} DeepVqePrepostConfig;

/* The compiled model ABI, for validating a graph's ONNX/JSON metadata. */
typedef struct DeepVqePrepostDescriptor {
    uint32_t layout_version;
    int delay_depth;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int time_order;
    int freq_taps;
    int conv_window_frames;
    int score_window_frames;
    int gru_layers;
    int gru_hidden;
    int state_tensor_count;
} DeepVqePrepostDescriptor;

/* Fixed 32-byte descriptor; a pool sized for another io_mode, D or build is refused by _init_ex. */
typedef struct DeepVqePrepostMemReq {
    uint32_t descriptor_version;  /* DEEPVQE_PREPOST_DESCRIPTOR_VERSION     */
    uint32_t layout_version;      /* DEEPVQE_PREPOST_LAYOUT_VERSION         */
    uint32_t io_mode;             /* the resolved DeepVqeIoMode             */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid + io_mode + D      */
    uint32_t alignment;
    uint32_t reserved;            /* 0 */
    uint64_t bytes;
} DeepVqePrepostMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DeepVqePrepostMemReq) == 32,
               "DeepVqePrepostMemReq must stay 32 bytes");
#endif
#endif

typedef struct DeepVqePrepost DeepVqePrepost;

/* Read-only accelerator views. `mic`/`far` are the compressed spectra, planar
 * [1,2,1,BINS] (real plane, then imaginary). `state[]` is indexed by
 * DeepVqeStateId. Pointers are into the pool and valid for the instance's life. */
typedef struct DeepVqePrepostInputs {
    const float *mic;                            /* [1,2,1,BINS] */
    const float *far;                            /* [1,2,1,BINS] */
    size_t       spectrum_ri_elements;           /* 2 * BINS     */
    const float *state[DEEPVQE_STATE_COUNT];
    size_t       state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostInputs;

/* Accelerator-writable outputs; the graph returns the full next value of every
 * state tensor. `state_out[id]` has the same address as `inputs.state[id]`, so
 * the output of frame t is the input of frame t+1; read all state inputs before
 * writing any state output. Only `taps` is NaN-prefilled. */
typedef struct DeepVqePrepostOutputs {
    float *taps;   /* [1,1,BINS,TIME_ORDER*FREQ_TAPS*2], packed CCM taps */
    size_t taps_elements;
    float *state_out[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostOutputs;

/* ---- boundary naming ------------------------------------------------- */

/* Exact graph tensor names; static literals, NULL for an unknown id. */
const char *deepvqe_prepost_state_name(int state_id);      /* graph input  */
const char *deepvqe_prepost_state_name_out(int state_id);  /* graph output */

/* Graph name of the head output. */
#define DEEPVQE_PREPOST_OUTPUT_NAME "output"

/* Row-major shape of one state tensor into `dims`; returns the rank (4), or -1 on
 * NULL, unknown id or D out of range. */
int deepvqe_prepost_state_shape(int state_id, int delay_depth,
                                int dims[DEEPVQE_STATE_MAX_RANK]);

/* Element count of one state tensor, or 0 on a bad id / D. */
size_t deepvqe_prepost_state_elements(int state_id, int delay_depth);

/* "mute_fail_closed": the policy frame_skip() implements. */
const char *deepvqe_prepost_skip_policy_name(void);

/* ---- lifecycle ------------------------------------------------------- */

/* Fill a config with this build's defaults. Returns 0, or -1 on NULL or
 * out-of-range io_mode/delay_depth. */
int deepvqe_prepost_config_defaults(DeepVqePrepostConfig *cfg,
                                    int io_mode, int delay_depth);

/* Fill the compiled deployment ABI for `delay_depth`. Returns 0, or -1. */
int deepvqe_prepost_descriptor_default(int delay_depth,
                                       DeepVqePrepostDescriptor *descriptor);

/* Validate a descriptor loaded from ONNX/JSON metadata: every field except
 * delay_depth (range-checked only) must equal this build's. Returns 0 on match. */
int deepvqe_prepost_descriptor_validate(
    const DeepVqePrepostDescriptor *descriptor);

/* Exact pool size. Returns -1 (req untouched) on NULL, unknown io_mode, D out of
 * range, or DEEPVQE_IO_TIME without fft/window. */
int deepvqe_prepost_get_mem_size(const DeepVqePrepostConfig *cfg,
                                 DeepVqePrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config, misaligned or undersized
 * pool, or wrong-size FftHandle. Starts reset. */
DeepVqePrepost *deepvqe_prepost_init(void *pool, size_t bytes,
                                     const DeepVqePrepostConfig *cfg);

/* As _init, plus the stale-pool gate; `expected` NULL behaves as _init. */
DeepVqePrepost *deepvqe_prepost_init_ex(void *pool, size_t bytes,
                                        const DeepVqePrepostConfig *cfg,
                                        const DeepVqePrepostMemReq *expected);

/* get_mem_size + aligned allocation + init (host path). */
DeepVqePrepost *deepvqe_prepost_create(const DeepVqePrepostConfig *cfg);

/* Frees only what _create allocated; NULL-safe, a no-op on an _init instance. */
void deepvqe_prepost_destroy(DeepVqePrepost *p);

/* Zero every state tensor and the CCM ring, drop any open frame, reset framing. */
void deepvqe_prepost_reset(DeepVqePrepost *p);

/* Counter of state restarts (reset, or frame_commit refusing a frame); 0 for NULL.
 * A fixed-point accelerator refills its state with quantized zeros when it changes. */
uint32_t deepvqe_prepost_state_generation(const DeepVqePrepost *p);

int deepvqe_prepost_hop_size(const DeepVqePrepost *p);   /* or -1 */
int deepvqe_prepost_num_bins(const DeepVqePrepost *p);   /* or -1 */
int deepvqe_prepost_io_mode(const DeepVqePrepost *p);    /* or -1 */
const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(
    const DeepVqePrepost *p);

/* ---- per-hop stages -------------------------------------------------- */

/* DEEPVQE_IO_TIME. One hop of the raw microphone and one of the far-end
 * reference, both for the same input hop (no internal skew). Returns 1, or -1 on
 * NULL args, in DEEPVQE_IO_FREQ, or while the previous frame is still open. */
int deepvqe_prepost_pre_process(DeepVqePrepost *p,
                                const float mic_hop[AIAEC_HOP],
                                const float far_hop[AIAEC_HOP]);

/* DEEPVQE_IO_FREQ. One analysed frame per stream: unnormalised rfft of a
 * sqrt-Hann-windowed AIAEC_N_FFT block (center=False). Returns 1; -1 on NULL
 * args, in DEEPVQE_IO_TIME, or while the previous frame is open. */
int deepvqe_prepost_pre_process_freq(DeepVqePrepost *p,
                                     const float mic_re[AIAEC_N_BINS],
                                     const float mic_im[AIAEC_N_BINS],
                                     const float far_re[AIAEC_N_BINS],
                                     const float far_im[AIAEC_N_BINS]);

/* Publish the current frame's accelerator boundary; `taps` is NaN-prefilled so a
 * partial write is caught by frame_commit. Returns 0, or -1 if no frame is open. */
int deepvqe_prepost_frame_inputs(DeepVqePrepost *p,
                                 DeepVqePrepostInputs *inputs,
                                 DeepVqePrepostOutputs *outputs);

/* Validate the outputs and advance the CCM ring. Every tap and h_gru must be
 * finite (of each window only the newest frame is checked); the taps are then
 * applied to the raw microphone ring and fed to the synthesis (DEEPVQE_IO_TIME)
 * or staged for post_process_freq. Requires a frame published by frame_inputs().
 * On a non-finite value all state tensors are zeroed, the state generation
 * advances, the frame stays open and -1 is returned; call frame_skip() or re-run. */
int deepvqe_prepost_frame_commit(DeepVqePrepost *p);

/* Fail closed: emit silence for the current frame, leave the CCM ring and state
 * as they are, advance only the framing schedule. Returns 0, or -1. */
int deepvqe_prepost_frame_skip(DeepVqePrepost *p);

/* DEEPVQE_IO_TIME. Emit this hop's output: AIAEC_HOP samples, zero-filled during
 * warm-up; `*written` (may be NULL) reports how many are meaningful. Returns 0, or -1. */
int deepvqe_prepost_post_process(DeepVqePrepost *p,
                                 float out_hop[AIAEC_HOP], int *written);

/* DEEPVQE_IO_FREQ. The enhanced spectrum of the frame just committed or skipped. Returns 0, or -1. */
int deepvqe_prepost_post_process_freq(DeepVqePrepost *p,
                                      float re[AIAEC_N_BINS],
                                      float im[AIAEC_N_BINS]);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_PREPOST_H */
