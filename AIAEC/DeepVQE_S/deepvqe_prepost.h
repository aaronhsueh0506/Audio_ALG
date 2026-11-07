/* DeepVQE-S pre/post class: everything between the caller's spectra and the accelerator's
 * tensors, frequency domain only; one frame in, one inference, one frame out.
 *
 *     deepvqe_prepost_pre_process_freq(p, mic_re, mic_im, far_re, far_im);
 *     deepvqe_prepost_frame_open(p, &in, &out);   // views into the pool
 *     my_npu_run(&in, &out);                        // NPU fills `out`
 *     deepvqe_prepost_frame_finish(p);              // after every run; or _frame_bypass()
 *     deepvqe_prepost_post_process_freq(p, re, im);
 *
 * Framing: one AcStft per stream and one AcIstft for the output, all from
 * ac_stft_config_default(&sc, n_fft) (sqrt-Hann, center 0, hop n_fft / 2, spec_scale =
 * ispec_scale = 1) with sc.ola = AC_OLA_ENV_NORM (analysis ignores ola); one frame per hop from
 * the first hop, no latency beyond it. The class holds no window and no FFT handle. */

#ifndef DEEPVQE_PREPOST_H
#define DEEPVQE_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_process.h"   /* DeepVqeCcmState, DEEPVQE_TIME_ORDER, DEEPVQE_FREQ_TAPS */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_PREPOST_DESCRIPTOR_VERSION 1u

/* Accelerator boundary version (two signal inputs, the CCM-tap head, sixteen rank-4 state tensors,
 * each returned in full); any rename, reshape, reorder or change of the tensor count bumps it. */
#define DEEPVQE_PREPOST_LAYOUT_VERSION 4u

/* Input compression exponent (model.compression_exponent); the exporter refuses others. */
#define DEEPVQE_COMPRESSION_EXP 0.3f
#define DEEPVQE_PREPOST_ALIGNMENT      16u
/* Folded into build_flags_hash; bumped on any change of pool size, carve order or descriptor. */
#define DEEPVQE_PREPOST_CARVE_VERSION  8u

/* Sanity bounds for the delay depth D (exporter max_delay_frames). D has no default: the
 * application supplies the value its model is exported with; it sizes nothing here. */
#define DEEPVQE_PREPOST_MIN_D  1
#define DEEPVQE_PREPOST_MAX_D  256

typedef struct DeepVqeGridSpec {
    int sample_rate;
    int n_fft;
    int delay_depth;   /* D, as exported */
} DeepVqeGridSpec;

typedef struct DeepVqeGrid {
    int sample_rate;
    int n_fft;
    int hop;
    int n_bins;
    int delay_depth;
} DeepVqeGrid;

enum {
    DEEPVQE_GRID_OK = 0,
    DEEPVQE_GRID_ERR_NULL = -1,
    DEEPVQE_GRID_ERR_SAMPLE_RATE = -2,
    DEEPVQE_GRID_ERR_N_FFT = -3,
    DEEPVQE_GRID_ERR_DELAY_DEPTH = -4,
    DEEPVQE_GRID_ERR_INCONSISTENT = -5
};

/* Build the run-time grid from a spec filled with what the model is exported for (hop = n_fft / 2,
 * n_bins = n_fft / 2 + 1). Returns DEEPVQE_GRID_OK (grid written), or the first failing check
 * (grid untouched): sample_rate > 0; n_fft a size audiolib's framing builds (ac_stft_get_mem_size()
 * of the default config is not 0); delay_depth within [DEEPVQE_PREPOST_MIN_D, MAX_D]. */
int deepvqe_grid_make(DeepVqeGrid *grid, const DeepVqeGridSpec *spec);

/* _make's checks plus consistent hop and n_bins. Returns DEEPVQE_GRID_OK or the failing check. */
int deepvqe_grid_validate(const DeepVqeGrid *grid);

/* CCM taps [1,1,BINS,TIME_ORDER*FREQ_TAPS*2], last axis [time][frequency][RI]. */
static inline size_t deepvqe_prepost_taps_elements(const DeepVqeGrid *grid)
{
    return (size_t)grid->n_bins * DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2u;
}

/* Graph input order after the two signal inputs (outputs: after the head). */
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

typedef struct DeepVqePrepostConfig {
    DeepVqeGrid grid;    /* deepvqe_grid_make(); copied at init */
    /* The application's recurrent state by DeepVqeStateId: borrowed, must outlive the instance;
     * shapes are the application's, only element counts are given. A float host registers its
     * sixteen buffers (zeroed at init and reset, finite-checked whole at every frame_finish); a
     * board whose accelerator owns the state registers nothing and binds each state input and
     * its `_out` to one accelerator buffer. All or none: every pointer non-NULL with a nonzero
     * count, or all NULL/0, else refused. Not in the pool size or hash. */
    float *state[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostConfig;

/* The model ABI for validating a graph's ONNX/JSON metadata. The library owns layout_version,
 * delay_depth (range-checked only), the grid, time_order, freq_taps and state_tensor_count; the
 * conv/score window and GRU fields are the model's: zero from _descriptor_default(), never used. */
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

/* Fixed 32-byte descriptor; a pool sized for another grid, D or build is refused by _init_ex. */
typedef struct DeepVqePrepostMemReq {
    uint32_t descriptor_version;  /* DEEPVQE_PREPOST_DESCRIPTOR_VERSION     */
    uint32_t layout_version;      /* DEEPVQE_PREPOST_LAYOUT_VERSION         */
    uint32_t reserved0;           /* 0 */
    uint32_t build_flags_hash;    /* FNV-1a-32 over sample rate, n_fft, D and versions */
    uint32_t alignment;
    uint32_t reserved1;           /* 0 */
    uint64_t bytes;
} DeepVqePrepostMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(DeepVqePrepostMemReq) == 32,
               "DeepVqePrepostMemReq must stay 32 bytes");
#endif
#endif

typedef struct DeepVqePrepost DeepVqePrepost;

/* Read-only accelerator views: `mic`/`far` are the compressed planar spectra (real plane, then
 * imaginary; BINS = grid.n_bins); `state[]` is the registered buffers by DeepVqeStateId, NULL
 * with 0 elements when none. The pointers are constant for the instance's life. */
typedef struct DeepVqePrepostInputs {
    const float *mic;                            /* [1,2,1,BINS] */
    const float *far;                            /* [1,2,1,BINS] */
    size_t       spectrum_ri_elements;           /* 2 * BINS     */
    const float *state[DEEPVQE_STATE_COUNT];
    size_t       state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostInputs;

/* Accelerator-writable outputs; the graph returns the full next value of every state tensor.
 * `state_out[id]` has the address of `inputs.state[id]` (both NULL when nothing is registered;
 * a board binds each pair to one accelerator buffer): read every state input before writing any
 * state output. Only `taps` is NaN-prefilled. */
typedef struct DeepVqePrepostOutputs {
    float *taps;   /* [1,1,BINS,TIME_ORDER*FREQ_TAPS*2], packed CCM taps */
    size_t taps_elements;
    float *state_out[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostOutputs;

/* Exact graph tensor names; static literals, NULL for an unknown id. */
const char *deepvqe_prepost_state_name(int state_id);      /* graph input  */
const char *deepvqe_prepost_state_name_out(int state_id);  /* graph output */

#define DEEPVQE_PREPOST_OUTPUT_NAME "output"   /* graph name of the head output */

const char *deepvqe_prepost_skip_policy_name(void);   /* "mute_fail_closed" (frame_bypass()) */

/* Fill a config for `grid` (copied), nothing registered. Returns 0, or -1 on NULL/invalid grid. */
int deepvqe_prepost_config_defaults(DeepVqePrepostConfig *cfg, const DeepVqeGrid *grid);

/* Fill the external fields of the model ABI for `grid`; the interior ones are 0. Returns 0, or
 * -1 on NULL or an invalid grid. */
int deepvqe_prepost_descriptor_default(const DeepVqeGrid *grid,
                                       DeepVqePrepostDescriptor *descriptor);

/* Every external field except delay_depth (range-checked only) must equal the contract for `grid`.
 * Returns 0 on match, -1 otherwise (NULL and an invalid grid included). */
int deepvqe_prepost_descriptor_validate(
    const DeepVqeGrid *grid, const DeepVqePrepostDescriptor *descriptor);

/* Exact pool size: a function of n_bins, not of D. The caller supplies the pool and _init
 * carves it. Returns -1 (req untouched) on NULL, an invalid grid, or an inconsistent state
 * registration. */
int deepvqe_prepost_get_mem_size(const DeepVqePrepostConfig *cfg,
                                 DeepVqePrepostMemReq *req);

/* Construct inside caller memory; starts reset (the registered state, if any, is zeroed).
 * NULL on a bad config, misaligned or undersized pool. */
DeepVqePrepost *deepvqe_prepost_init(void *pool, size_t bytes,
                                     const DeepVqePrepostConfig *cfg);

/* As _init, plus the stale-pool gate; `expected` NULL behaves as _init. */
DeepVqePrepost *deepvqe_prepost_init_ex(void *pool, size_t bytes,
                                        const DeepVqePrepostConfig *cfg,
                                        const DeepVqePrepostMemReq *expected);

/* get_mem_size + aligned allocation + init (host path). */
DeepVqePrepost *deepvqe_prepost_create(const DeepVqePrepostConfig *cfg);

/* Frees only what _create allocated, never the registered state. NULL-safe; no-op for _init. */
void deepvqe_prepost_destroy(DeepVqePrepost *p);

/* Zero the registered state and the CCM ring, drop any open frame; config, every pointer and the
 * pool stay. Advances the state generation. NULL is ignored. */
void deepvqe_prepost_reset(DeepVqePrepost *p);

/* Counter of state restarts (reset, or frame_finish refusing a frame); 0 after init and for NULL;
 * kept whether or not a state is registered. A fixed-point accelerator loads quantized zeros
 * into its own copy of the state at start and on every change. */
uint32_t deepvqe_prepost_state_generation(const DeepVqePrepost *p);

int deepvqe_prepost_hop_size(const DeepVqePrepost *p);   /* or -1 */
int deepvqe_prepost_num_bins(const DeepVqePrepost *p);   /* or -1 */
const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(
    const DeepVqePrepost *p);

/* One analysed frame per stream (n_bins bins each). Returns 1; -1 on NULL args or an open frame. */
int deepvqe_prepost_pre_process_freq(DeepVqePrepost *p,
                                     const float *mic_re, const float *mic_im,
                                     const float *far_re, const float *far_im);

/* Publish the current frame's accelerator boundary; `taps` is NaN-prefilled so a partial write
 * is caught by frame_finish. Returns 0, or -1 if no frame is open. */
int deepvqe_prepost_frame_open(DeepVqePrepost *p,
                                 DeepVqePrepostInputs *inputs,
                                 DeepVqePrepostOutputs *outputs);

/* Call after every accelerator run; needs a frame published by frame_open(), else -1. Every tap
 * and every element of every registered state buffer (whole buffers, older frames of a history
 * window included) must be finite. Then the taps are applied to the CCM ring (the last
 * TIME_ORDER raw microphone spectra, advanced only here) and the enhanced spectrum is staged
 * for post_process_freq. On a non-finite value the registered buffers are zeroed (nothing
 * registered: only the taps are checked), the state generation advances, the ring is not pushed,
 * the frame stays open and -1 is returned: call frame_bypass() or re-run. */
int deepvqe_prepost_frame_finish(DeepVqePrepost *p);

/* Fail closed: the primary input is the raw microphone with no canceller in front, so emit
 * silence (not pass-through) for the current frame; the CCM ring and state stay as they are.
 * Returns 0, or -1. */
int deepvqe_prepost_frame_bypass(DeepVqePrepost *p);

/* The enhanced spectrum (n_bins bins) of the frame just finished or bypassed. Returns 0, or -1. */
int deepvqe_prepost_post_process_freq(DeepVqePrepost *p, float *re, float *im);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_PREPOST_H */
