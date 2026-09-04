/* ============================================================
 * DeepVQE-S pre/post-processing class.
 *
 * ONE object owning everything between the caller's audio and the
 * accelerator's tensors, so an integrator never touches the framing
 * states, the RI staging or the CCM spectrum ring directly:
 *
 *     deepvqe_prepost_pre_process(p, mic_hop, far_hop);
 *     deepvqe_prepost_frame_inputs(p, &in, &out);   // views into the pool
 *     my_npu_run(&in, &out);                        // NPU fills `out`
 *     deepvqe_prepost_frame_commit(p);              // or _frame_skip()
 *     deepvqe_prepost_post_process(p, out_hop, &written);
 *
 * ONE HOP IN, ONE INFERENCE, ONE HOP OUT -- always, from the very first hop,
 * in both I/O modes. There is no variable emission count to loop over.
 *
 * FRAMING: this class analyses at center=False -- a plain rolling window over
 * the last AIAEC_N_FFT samples, one frame per hop. It is NOT
 * aiaec_analysis_push's centered (center=True) schedule, which emits 0 frames
 * on the first hop and TWO on the second to carry a reflect prefix.
 *
 * That is a re-labelling, not a change of signal. At this geometry
 * (win == n_fft, hop == n_fft/2, one shared sqrt-Hann table) centered frame k
 * and non-centered frame k-1 cover exactly the same samples with exactly the
 * same window, so they are bit-identical -- measured, not assumed. The only
 * difference is the head: the centered schedule additionally emits one
 * reflect-prefix frame, which this class does not. That also makes the class
 * agree with the AEC/GSC seam, whose spectra are center=False and arrive one
 * per hop from hop 0.
 *
 * The one-hop algorithmic latency is unchanged: it comes from the synthesis
 * half-window trim, not from the analysis schedule.
 *
 * ⚠⚠ THIS MODEL'S PRIMARY INPUT IS THE RAW MICROPHONE, NOT A RESIDUAL.
 * DeepVQE-S is end-to-end AEC + RES + NS + dereverberation
 * (dataset_gen/model_views.py binds it to stems.mic_postclip /
 * stems.far_render; model.py's task is "end_to_end_aec_res_nr_dereverb").
 * There is no linear canceller in front of it. So the identity Align-ULCNet
 * takes when an inference fails -- pass stream 0 through unenhanced -- is
 * NOT available here: passing the microphone through would emit the FULL
 * UNCANCELLED ECHO, which is the one failure an AEC must never produce.
 * deepvqe_prepost_frame_skip() therefore FAILS CLOSED: it emits SILENCE for
 * that frame (deepvqe_prepost_skip_policy_name() == "mute_fail_closed").
 * The cost is a notch in the near-end, bounded by one frame; the cost of the
 * alternative is unbounded echo into the far end. A caller that wants
 * graceful degradation instead of a notch must run a canceller of its own
 * and cross-fade at the pipeline layer -- that decision does not belong to
 * this class, which cannot see whether one exists.
 *
 * WHY A THIRD TRANSLATION UNIT, not an addition to one of the two it uses:
 * aiaec_process.c and DeepVQE_S/deepvqe_process.c are deliberately
 * independently linkable -- AIAEC/tests/test_ulcnet_process_c.py compiles
 * deepvqe_process.c alone (test_deepvqe_ccm_helper_matches_python) and the
 * shared-mask tests compile aiaec_process.c without it. A class in either
 * would put an undefined symbol in the other's parity build. This file
 * composes them and neither of them changes.
 *
 * Memory: both house lifecycles (audio_common/include/fft_wrapper.h,
 * "Static memory support").
 *   size_t/req  deepvqe_prepost_get_mem_size(&cfg, &req);
 *   DeepVqePrepost *p = deepvqe_prepost_init(pool, req.bytes, &cfg); // board
 *   DeepVqePrepost *p = deepvqe_prepost_create(&cfg);                // host
 *   deepvqe_prepost_destroy(p);   // frees only what create() allocated
 *
 * The FftHandle and the sqrt-Hann window table are CALLER-OWNED and may be
 * shared with anything else on the same grid -- this class never creates or
 * destroys either. They are required in DEEPVQE_IO_TIME and unused in
 * DEEPVQE_IO_FREQ.
 *
 * Grid: 16 kHz / 512 / 256 only. That is not this file's choice --
 * aiaec_process.h is a 16 kHz boundary and carries an #error guard for it,
 * so a build whose Align-ULCNet grid moved to 48 kHz/1024 fails to compile
 * here rather than silently reinterpreting the tensors. DeepVQE-S's channel
 * schedule and the frequency ladder below are pinned to AIAEC_N_BINS anyway.
 * ============================================================ */

#ifndef DEEPVQE_PREPOST_H
#define DEEPVQE_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_process.h"   /* DeepVqeCcmState, DEEPVQE_TIME_ORDER,
                                * DEEPVQE_FREQ_TAPS and (transitively)
                                * aiaec_process.h: the grid, AiaecAnalysis/
                                * AiaecSynthesis and FftHandle             */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_PREPOST_DESCRIPTOR_VERSION 1u

/* The accelerator boundary this file binds: the two raw RI signal inputs,
 * the CCM-tap head output and the sixteen explicit state tensors emitted by
 * _streaming_export.py for DeepVQE_S. Unrelated to (and deliberately not
 * aliased from) ULCNET_MODEL_IO_LAYOUT_VERSION: the two models' boundaries
 * move independently, and a shared number would make one model's bump look
 * like the other's. Any rename, reshape, reorder or added/removed tensor at
 * the boundary bumps this. */
#define DEEPVQE_PREPOST_LAYOUT_VERSION 2u
#define DEEPVQE_PREPOST_ALIGNMENT      16u
/* Folded into build_flags_hash: bump whenever pp_layout's carve walk changes,
 * so a pool recorded by the previous carve is refused on the hash, not only
 * on `bytes`. 2: the TIME analyses became two AiaecAnalysis states. */
#define DEEPVQE_PREPOST_CARVE_VERSION  2u

/* Alignment search depth D, the exporter's max_delay_frames. It sizes the
 * attention key/value rings and the score history, so it is a pool-size
 * parameter. 63 is grid.delay_frames(1.0) on this grid -- ceil(1.0 s *
 * 16000/256), i.e. the shipped one-second search range. */
#define DEEPVQE_PREPOST_MIN_D      1
#define DEEPVQE_PREPOST_MAX_D      256
#define DEEPVQE_PREPOST_DEFAULT_D  63

/* ---- DeepVQE-S topology constants (mirrors of DeepVQE_S/model.py) -------
 * These are NOT tunables. They are the checkpoint's fixed channel schedule
 * and the encoder's stride-2 frequency ladder, restated in C because the
 * state tensors' shapes are made of them. A checkpoint built with any other
 * schedule has a different boundary and must bump the layout version. */
#define DEEPVQE_HALF_(n)        (((n) + 1) / 2)
#define DEEPVQE_F0              AIAEC_N_BINS              /* 257 */
#define DEEPVQE_F1              DEEPVQE_HALF_(DEEPVQE_F0) /* 129 */
#define DEEPVQE_F2              DEEPVQE_HALF_(DEEPVQE_F1) /*  65 */
#define DEEPVQE_F3              DEEPVQE_HALF_(DEEPVQE_F2) /*  33 */
#define DEEPVQE_F4              DEEPVQE_HALF_(DEEPVQE_F3) /*  17 */

/* Causal conv history depth = time kernel 4 - 1; the alignment score conv
 * uses a (5,3) kernel, so its history is 4. */
#define DEEPVQE_CONV_HISTORY    3
#define DEEPVQE_SCORE_HISTORY   4
#define DEEPVQE_SIM_CHANNELS    4     /* FrameDelayAttention similarity     */
#define DEEPVQE_VALUE_CHANNELS  24    /* FrameDelayAttention value          */
#define DEEPVQE_GRU_LAYERS      1
#define DEEPVQE_GRU_HIDDEN      192

/* Head output: CCM taps, graph shape
 * [1,1,BINS,TIME_ORDER*FREQ_TAPS*2]. The packed last axis is still ordered
 * [time][frequency][RI], flat-identical to deepvqe_process.h's
 * taps[AIAEC_N_BINS][DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2]. */
#define DEEPVQE_TAPS_ELEMENTS \
    ((size_t)AIAEC_N_BINS * DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2u)

/* ---- The explicit state boundary ---------------------------------------
 * Order IS the graph's input order after the two signal inputs -- it is
 * _streaming_export.py's `input_names[2:]` verbatim, and the same order the
 * outputs take after the head. An adapter that binds by index must use this
 * enum; one that binds by name uses deepvqe_prepost_state_name(). */
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

/* Maximum rank any boundary tensor has, so a caller can size a dims array
 * without allocating. Every state is rank 4 (NCHW) except h_gru, which the
 * exporter emits rank 3 as [1,1,GRU_HIDDEN]. */
#define DEEPVQE_STATE_MAX_RANK 4

/* Which side of the transform the caller works on. FIXED AT INIT because it
 * decides the pool size: DEEPVQE_IO_TIME additionally carves two analysis
 * states (each with its own transform scratch), one synthesis state and the
 * output hop staging. */
typedef enum DeepVqeIoMode {
    /* Spectrum in, spectrum out. NO framing, windowing, FFT, iFFT or
     * overlap-add -- the caller owns the transform. This is the mode for
     * chaining, and the mode in which `fft`/`window` may be NULL. */
    DEEPVQE_IO_FREQ = 0,
    /* Hop in, hop out. The class owns framing, windowing, overlap-add and
     * calls the CALLER'S FftHandle for the transforms. */
    DEEPVQE_IO_TIME = 1
} DeepVqeIoMode;

typedef struct DeepVqePrepostConfig {
    int          io_mode;      /* a DeepVqeIoMode                           */
    int          delay_depth;  /* D; DEEPVQE_PREPOST_MIN/MAX_D              */
    FftHandle   *fft;          /* DEEPVQE_IO_TIME: required, borrowed       */
    const float *window;       /* DEEPVQE_IO_TIME: required, borrowed,
                                * AIAEC_N_FFT entries from
                                * aiaec_make_window()                       */
} DeepVqePrepostConfig;

/* The compiled model ABI, for validating against a graph's ONNX/JSON
 * metadata before binding it. */
typedef struct DeepVqePrepostDescriptor {
    uint32_t layout_version;
    int delay_depth;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int time_order;
    int freq_taps;
    int conv_history_frames;
    int score_history_frames;
    int gru_layers;
    int gru_hidden;
    int state_tensor_count;
} DeepVqePrepostDescriptor;

/* Fixed 32-byte shape, same staleness-gate discipline as the pipelines'
 * memory descriptors: a pool sized for another io_mode, delay depth or build
 * is refused by _init_ex rather than reinterpreted. */
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

/* Read-only views the accelerator binds. `mic`/`far` are the RAW RI spectra
 * interleaved [bin][re,im] -- DeepVQE-S applies its own power-law
 * compression INSIDE the graph, unlike Align-ULCNet whose front end was
 * moved to the host. Nothing is compressed, scaled or masked here.
 * `state[]`/`state_elements[]` are indexed by DeepVqeStateId. Pointers are
 * into the instance's pool and stay valid until the next pre_process. */
typedef struct DeepVqePrepostInputs {
    const float *mic;                            /* [1,1,BINS,2] */
    const float *far;                            /* [1,1,BINS,2] */
    size_t       spectrum_ri_elements;           /* 2 * BINS     */
    const float *state[DEEPVQE_STATE_COUNT];
    size_t       state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostInputs;

/* Accelerator-writable outputs. Unlike Align-ULCNet, whose graph returns
 * only the delta state, DeepVQE-S returns the FULL next value of every state
 * tensor -- so the pool holds two banks and commit() swaps them. That is
 * also why the NaN prefill and the finite check below each walk the whole
 * ~700 kB state at 16 kHz/D=63: it is what makes "on failure nothing moves"
 * an enforced property rather than a claim.
 * prepare() fills every element with NaN so commit() detects partial writes. */
typedef struct DeepVqePrepostOutputs {
    float *taps;   /* [1,1,BINS,TIME_ORDER*FREQ_TAPS*2], packed CCM taps */
    size_t taps_elements;
    float *state_out[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];
} DeepVqePrepostOutputs;

/* ---- boundary naming ------------------------------------------------- */

/* The exporter's exact graph tensor names. Return string literals with
 * static lifetime, or NULL for an id outside the enum. */
const char *deepvqe_prepost_state_name(int state_id);      /* graph input  */
const char *deepvqe_prepost_state_name_out(int state_id);  /* graph output */

/* Graph name of the head output. */
#define DEEPVQE_PREPOST_OUTPUT_NAME "output"

/* Row-major graph shape of one state tensor. Writes the dimensions into
 * `dims`, zeroes the entries beyond the shape, and returns the rank (3 for
 * h_gru, 4 for the rest), or -1 on NULL / an unknown id / a D out of range.
 * `delay_depth` is taken explicitly so a tool can check a graph's schema
 * before it has an instance. */
int deepvqe_prepost_state_shape(int state_id, int delay_depth,
                                int dims[DEEPVQE_STATE_MAX_RANK]);

/* Element count of one state tensor, or 0 on a bad id / D. */
size_t deepvqe_prepost_state_elements(int state_id, int delay_depth);

/* "mute_fail_closed" -- the one policy frame_skip() implements, named so an
 * integrator can log WHICH identity this model takes without re-deriving it
 * from the header comment. See the warning at the top of this file for why
 * it is not pass-through. */
const char *deepvqe_prepost_skip_policy_name(void);

/* ---- lifecycle ------------------------------------------------------- */

/* Fill a config with this build's grid defaults. Returns 0, or -1 on NULL /
 * an io_mode or delay_depth outside its range. `fft`/`window` are left NULL
 * for the caller to set in DEEPVQE_IO_TIME. */
int deepvqe_prepost_config_defaults(DeepVqePrepostConfig *cfg,
                                    int io_mode, int delay_depth);

/* Fill the compiled deployment ABI for `delay_depth`. Returns 0, or -1. */
int deepvqe_prepost_descriptor_default(int delay_depth,
                                       DeepVqePrepostDescriptor *descriptor);

/* Validate a descriptor loaded from ONNX/JSON metadata against this C ABI.
 * Every field except delay_depth must equal this build's constant;
 * delay_depth is an export-time deployment parameter and is only range
 * checked. Returns 0 on a match, -1 otherwise. */
int deepvqe_prepost_descriptor_validate(
    const DeepVqePrepostDescriptor *descriptor);

/* Exact pool size. Reject-first: -1 with *req untouched on a NULL argument,
 * an unknown io_mode, a D out of range, or DEEPVQE_IO_TIME without a usable
 * fft/window. */
int deepvqe_prepost_get_mem_size(const DeepVqePrepostConfig *cfg,
                                 DeepVqePrepostMemReq *req);

/* Construct inside caller memory. NULL on a bad config, a misaligned or
 * undersized pool, or a wrong-size FftHandle. Starts reset. */
DeepVqePrepost *deepvqe_prepost_init(void *pool, size_t bytes,
                                     const DeepVqePrepostConfig *cfg);

/* As _init, plus refusing a pool whose recorded requirements differ from
 * this build's -- the stale-pool gate. `expected` NULL behaves as _init. */
DeepVqePrepost *deepvqe_prepost_init_ex(void *pool, size_t bytes,
                                        const DeepVqePrepostConfig *cfg,
                                        const DeepVqePrepostMemReq *expected);

/* get_mem_size + aligned allocation + init. Host tools and tests; the board
 * path is _init on a caller pool. */
DeepVqePrepost *deepvqe_prepost_create(const DeepVqePrepostConfig *cfg);

/* Frees only what _create allocated. NULL-safe. For an _init instance it is
 * a genuine no-op that may be called repeatedly -- the caller's pool stays
 * the caller's; a _create instance is gone after the first call, so calling
 * it twice on one is a use-after-free like any other double free. Never
 * touches the caller's fft or window. */
void deepvqe_prepost_destroy(DeepVqePrepost *p);

/* Zero every state tensor and the CCM spectrum ring, drop any open frame,
 * and reset the framing state -- the two analysis states and the synthesis.
 * Config and pool are untouched. */
void deepvqe_prepost_reset(DeepVqePrepost *p);

int deepvqe_prepost_hop_size(const DeepVqePrepost *p);   /* or -1 */
int deepvqe_prepost_num_bins(const DeepVqePrepost *p);   /* or -1 */
int deepvqe_prepost_io_mode(const DeepVqePrepost *p);    /* or -1 */
const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(
    const DeepVqePrepost *p);

/* ---- per-hop stages -------------------------------------------------- */

/* DEEPVQE_IO_TIME. One hop of the RAW MICROPHONE and one of the far-end
 * reference, BOTH FOR THE SAME INPUT HOP -- this class applies no internal
 * skew. A caller whose two sources are not already sample-aligned aligns
 * them before calling; the model's own alignment attention searches only
 * the D-frame range on top of that.
 *
 * Transforms both streams (center=False rolling window). ALWAYS returns 1 --
 * one hop is one accelerator invocation, from the first hop onward. Returns
 * -1 on NULL args, in DEEPVQE_IO_FREQ, or while the previous frame is still
 * open (neither committed nor skipped): a hop is refused rather than
 * silently stacked on top of an unfinished one. */
int deepvqe_prepost_pre_process(DeepVqePrepost *p,
                                const float mic_hop[AIAEC_HOP],
                                const float far_hop[AIAEC_HOP]);

/* DEEPVQE_IO_FREQ. One already-analysed frame per stream -- the RAW
 * microphone and the far-end reference -- in the model's own framing
 * (unnormalised rfft of a sqrt-Hann-windowed AIAEC_N_FFT block on the hop
 * grid, center=False -- see FRAMING above). Does NO transform. Always
 * returns 1; -1 on NULL args, in DEEPVQE_IO_TIME, or while the previous
 * frame is still open. */
int deepvqe_prepost_pre_process_freq(DeepVqePrepost *p,
                                     const float mic_re[AIAEC_N_BINS],
                                     const float mic_im[AIAEC_N_BINS],
                                     const float far_re[AIAEC_N_BINS],
                                     const float far_im[AIAEC_N_BINS]);

/* Publish the current frame's accelerator boundary. Every writable output
 * is NaN-prefilled, so a partial write is caught by frame_commit rather than
 * leaking the previous frame's values. Pointers are into this instance's
 * pool and stay valid until the next pre_process. Returns 0, or -1 if no
 * frame is open. */
int deepvqe_prepost_frame_inputs(DeepVqePrepost *p,
                                 DeepVqePrepostInputs *inputs,
                                 DeepVqePrepostOutputs *outputs);

/* Transactional: validates that the accelerator wrote every tap and every
 * state element, swaps the state banks, then applies the CCM taps to the raw
 * microphone spectrum ring (deepvqe_ccm_process) and feeds the result to the
 * synthesis (DEEPVQE_IO_TIME) or stages it for post_process_freq.
 *
 * Requires a frame opened by pre_process AND published by frame_inputs():
 * a commit with no frame_inputs() behind it is refused, so an accelerator
 * that never ran cannot pass untouched buffers off as a result.
 *
 * On failure NOTHING moves -- the state banks do not swap, the CCM ring does
 * not advance, persistent state is byte-identical, the frame stays open and
 * -1 is returned. The caller then either calls deepvqe_prepost_frame_skip()
 * to keep the framing schedule intact, or re-runs the accelerator through a
 * fresh frame_inputs(). */
int deepvqe_prepost_frame_commit(DeepVqePrepost *p);

/* Fail CLOSED for the current frame: emit SILENCE, do not step the model's
 * state banks, and do not advance the CCM spectrum ring -- so model time and
 * host time stay consistent at "this frame never happened". Only the framing
 * schedule advances. This is what a failed accelerator run and an
 * alignment-boundary reprime both need.
 *
 * It is deliberately NOT the pass-through identity Align-ULCNet takes:
 * DeepVQE-S's stream 0 is the raw microphone, so pass-through would emit the
 * uncancelled echo. See the warning at the top of this file. Returns 0,
 * or -1. */
int deepvqe_prepost_frame_skip(DeepVqePrepost *p);

/* DEEPVQE_IO_TIME. Emit this hop's output. `out_hop` is always fully
 * written: AIAEC_HOP samples, zero-filled during warm-up. `*written` (may be
 * NULL) reports how many are meaningful. Returns 0, or -1. */
int deepvqe_prepost_post_process(DeepVqePrepost *p,
                                 float out_hop[AIAEC_HOP], int *written);

/* DEEPVQE_IO_FREQ. The enhanced spectrum of the frame just committed or
 * skipped. Returns 0, or -1. */
int deepvqe_prepost_post_process_freq(DeepVqePrepost *p,
                                      float re[AIAEC_N_BINS],
                                      float im[AIAEC_N_BINS]);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_PREPOST_H */
