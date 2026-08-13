/**
 * audio_pipeline_4ch_ulcnet.h — four-channel spatial pipeline with the
 * Align-ULCNet neural post-filter as its post stage.
 *
 * Architecture per hop (hop = 256 samples @ 16 kHz, fixed):
 *
 *   4x linear AEC lanes + one shared delay matcher (FourAecNrRes process_pre)
 *     -> SRP-PHAT DOA -> GSC effective weights (existing spatial libraries)
 *     -> beamformed linear-error SPECTRUM (gsc_process_with_weights output)
 *     -> WOLA reconstruction to a time-domain beamformed error hop
 *     -> Align-ULCNet centered STFT chain (ulcnet_process.h) driving an
 *        external NPU runtime through the UlcnetModel per-frame callback
 *     -> enhanced mono hop
 *
 * The core's own post-beam RES/NR path (four_aec_nr_res_process_post) is
 * NOT used: each pre frame is released via four_aec_nr_res_abandon_pre()
 * and the Align-ULCNet chain replaces that post stage entirely.
 *
 * TIMING CONTRACT (total added algorithmic latency = 2 hops = 512 samples
 * = 32 ms @ 16 kHz):
 *
 *   1 hop  — the gsc-spectrum WOLA: the beamformed-error spectrum of hop p
 *            is only fully reconstructed in the time domain at hop p+1
 *            (sqrt-Hann analysis inside the AEC + sqrt-Hann synthesis here,
 *            50% overlap-add).
 *   1 hop  — the centered Align-ULCNet chain itself (see ulcnet_process.h:
 *            hop#0 emits nothing, hop#p output corresponds to chain input
 *            hop p-1).
 *
 *   The far branch is delayed by ONE HOP inside this wrapper (a saved-hop
 *   buffer) before entering its analysis, so the model always sees
 *   error/far frame pairs of the SAME input hop: the WOLA-reconstructed
 *   beam hop pushed at hop p is the beamformed error of input hop p-1, and
 *   the far hop pushed beside it is the far source of that same hop p-1.
 *
 *   Consequently process() emits ZEROS for hop#0 AND hop#1, and the output
 *   of hop#p (p >= 2) corresponds to the beamformed linear error of input
 *   hop p-2. The last reconstructed beamformed-error hop is exposed
 *   read-only via audio_pipeline_4ch_ulcnet_last_beamformed_error() so
 *   integrators/tests can verify exactly this relation.
 *
 * FAR-INPUT DEPLOYMENT CONTRACT (set_far_input_mode below): the selected
 * mode MUST match the checkpoint's training far input -- a mismatch is an
 * input-distribution change, not a tuning knob.
 *   - ULCNET_FAR_RAW (default): the caller's raw far_reference feeds the
 *     far branch (with the same one-hop compensation as aligned mode).
 *     Checkpoint-compatible: current checkpoints are trained with RAW far.
 *     Model application is NEVER gated on the shared delay lock (the paper
 *     contract does not depend on lock).
 *   - ULCNET_FAR_ALIGNED: pre.aligned_ref feeds the far branch and the
 *     model's output is APPLIED only while the shared delay is locked
 *     (delay.solid && delay.delay_samples >= 0) -- the Phase-2 embedded
 *     candidate. Only use with a checkpoint trained on aligned far.
 *
 * MODEL CALLBACK POLICY (first version):
 *
 *   - The model is STEPPED (infer() invoked) for EVERY emitted frame
 *     whenever a callback is installed -- constant per-hop compute and
 *     continuous runtime recurrent state, matching the mono variant.
 *   - No model set, model->infer == NULL, or infer() returning nonzero
 *     => FAIL-OPEN: the error spectrum passes through the synthesis chain
 *     unchanged (identity), keeping the timing path constant. A successful
 *     infer() whose output contains ANY non-finite value is likewise
 *     discarded (identity frame) -- NaN/Inf never reaches the WOLA. In
 *     ULCNET_FAR_ALIGNED the same identity bypass applies while the shared
 *     delay is not locked (!delay.solid || delay.delay_samples < 0) — the
 *     runtime's output is only applied for frame pairs whose far branch is
 *     genuinely delay-aligned.
 *   - On a delay change event (delay.changed): model->reset (if set) is
 *     called so the runtime flushes its far attention ring + logit history,
 *     and the beamform WOLA accumulator AND the one-hop far buffer are
 *     cleared (the core reset its lanes' analysis history at the same
 *     boundary — mixing spectra from opposite sides of the realignment
 *     would corrupt the seam). The C-side ULCNet STFT states keep running:
 *     a 1-2 frame transient is accepted in this version; crossfading is a
 *     later phase. This reset policy applies in BOTH far modes (the error
 *     branch realigns discontinuously even when the far branch is raw).
 *   - audio_pipeline_4ch_ulcnet_reset() also calls model->reset (if set).
 *
 * VAD: this wrapper provides process_with_activity() ONLY (external VAD).
 * The conservative auto-VAD fallback stays with the standard wrapper
 * (audio_pipeline_4ch.h's audio_pipeline_4ch_process()); the
 * auto_vad_threshold_dbfs/auto_vad_snr_ratio/auto_vad_hangover_frames
 * config fields are validated but never used here.
 *
 * GRID: 16 kHz / fft 512 / hop 256 ONLY. The ULCNet pre/post constants are
 * compile-time 16 kHz (ULCNET_SR/N_FFT/HOP/BINS); 48 kHz configs and any
 * core fft_size other than 0 (forced to 512) or 512 are rejected at
 * validation time.
 *
 * Lifecycle mirrors audio_pipeline_4ch.h's descriptor-tier pool-first
 * pattern (caller-owned pool via get_mem_requirements()/init_ex(), heap
 * convenience via create(); zero heap anywhere on the init_ex path).
 */
#ifndef AUDIO_PIPELINE_4CH_ULCNET_H
#define AUDIO_PIPELINE_4CH_ULCNET_H

#include "audio_pipeline_4ch.h"
#include "ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

#define AUDIO_PIPELINE_4CH_ULCNET_DESCRIPTOR_VERSION 1u
/* Carve order: self, core, srp, gsc, gsc_spectrum, gsc_weights, fft, ifft,
 * ola, synth_win, beam_hop (the ULCNet analysis/synthesis states, frame
 * scratch, one-hop far buffer, far_input_mode and the chain's shared
 * sqrt-Hann window table live inside `self`). The one carved `fft` handle
 * serves BOTH the beamform WOLA and the ULCNet chain (same 512 size;
 * strictly sequential use within a hop). Bump together with the
 * build-flags-hash token string forever after.
 * v2: self grew the one-hop far-compensation buffer + far_input_mode.
 * v3: the ULCNet chain moved onto the shared carved FftHandle (its structs
 * embed their own FFT scratch) and self grew the shared window table
 * (ulcnet_window) replacing the per-struct window copies. */
#define AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION 3u

/**
 * Fixed-width descriptor for a caller-owned static-memory pool. Same 32-byte
 * fixed-width-integer shape and same field meanings as 4aec_nr_res.h's
 * FourAecNrResMemReq (descriptor_version/layout_version/backend_id/
 * build_flags_hash/alignment/reserved/bytes) -- see that header's own doc
 * comment for the full rationale rather than repeating it here. Like
 * AudioPipeline4ChMemReq, this layer's build_flags_hash folds in the core's
 * own build_flags_hash so a core-layer layout bump also invalidates every
 * persisted composite descriptor here.
 */
typedef struct AudioPipeline4ChUlcnetMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} AudioPipeline4ChUlcnetMemReq;

_Static_assert(
    sizeof(AudioPipeline4ChUlcnetMemReq) == 32,
    "AudioPipeline4ChUlcnetMemReq must be exactly 32 bytes");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, descriptor_version) == 0,
    "descriptor_version offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, layout_version) == 4,
    "layout_version offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, backend_id) == 8,
    "backend_id offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, build_flags_hash) == 12,
    "build_flags_hash offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, alignment) == 16,
    "alignment offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, reserved) == 20,
    "reserved offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, bytes) == 24,
    "bytes offset");

typedef struct AudioPipeline4ChUlcnet AudioPipeline4ChUlcnet;

/**
 * Far-input deployment contract, shared by the mono and 4ch ULCNet pipeline
 * variants (guarded so both headers can be included in one TU). The mode
 * MUST match the checkpoint's training far input; a mismatch is an
 * input-distribution change. See this header's preamble for the per-mode
 * gating rules.
 */
#ifndef ULCNET_FAR_INPUT_MODE_DEFINED
#define ULCNET_FAR_INPUT_MODE_DEFINED
typedef enum UlcnetFarInputMode {
    ULCNET_FAR_RAW     = 0,  /* raw far; checkpoint-compatible default; no
                              * delay-lock gating of model application     */
    ULCNET_FAR_ALIGNED = 1   /* aligned far + lock gating; the Phase-2
                              * embedded candidate                         */
} UlcnetFarInputMode;
#endif

/* ============================================================================
 * Config and lifecycle
 * ========================================================================== */

/**
 * Reuses AudioPipeline4ChConfig (the spatial stage is identical). Returns
 * the standard 16 kHz defaults with core.fft_size pre-set to 512 (the only
 * grid this wrapper accepts). Validation additionally rejects
 * core.sample_rate != 16000 and core.fft_size not in {0, 512}; a value of 0
 * is forced to 512, never to the core's own 256 default.
 */
AudioPipeline4ChConfig audio_pipeline_4ch_ulcnet_default_config(void);

/**
 * Query the complete 16-byte-aligned caller pool required by cfg. Performs
 * no allocation and returns 0 on success.
 */
int audio_pipeline_4ch_ulcnet_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChUlcnetMemReq* out);

/**
 * Place the complete pipeline (core + SRP-PHAT + GSC + beamform WOLA + the
 * ULCNet chain state) in caller-owned memory. mem must satisfy the freshly
 * queried alignment and byte requirement. No heap allocation is performed by
 * this path. A dirty/poisoned pool is accepted. The model callback starts
 * unset (fail-open identity) -- see set_model() below.
 */
AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_init(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg);

/**
 * Static init plus a stale-descriptor gate. Same 8-point rejection contract
 * as 4aec_nr_res.h's four_aec_nr_res_init_ex() (descriptor_version/
 * layout_version/backend_id/build_flags_hash/alignment must match; reserved
 * must be 0; expected->bytes and the pool actually handed to this call must
 * both be >= the current build's requirement). NULL expected is equivalent
 * to audio_pipeline_4ch_ulcnet_init().
 */
AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChUlcnetMemReq* expected);

/**
 * Heap convenience wrapper over get_mem_requirements() + init(). Returns
 * NULL for an invalid config or allocation failure.
 */
AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_create(
    const AudioPipeline4ChConfig* cfg);

/**
 * Install (or clear, with NULL) the NPU runtime callback set. The struct is
 * copied by value; model->user must stay valid until replaced or the
 * pipeline is destroyed. No reset is issued here: install a freshly-reset
 * runtime, or call audio_pipeline_4ch_ulcnet_reset() after swapping models
 * mid-stream. Returns 0 on success, nonzero on a NULL/destroyed pipeline.
 */
int audio_pipeline_4ch_ulcnet_set_model(
    AudioPipeline4ChUlcnet* p,
    const UlcnetModel* model);

/**
 * Select which far stream feeds the model's far branch (see the FAR-INPUT
 * DEPLOYMENT CONTRACT in this header's preamble). Instances start in
 * ULCNET_FAR_RAW (the checkpoint-compatible default); the mode survives
 * audio_pipeline_4ch_ulcnet_reset() like the model installation does. Set
 * it before streaming: switching mid-stream changes the model's input
 * distribution for the frames already in flight (one saved far hop). The
 * mode MUST match the checkpoint's training far input. Returns 0 on
 * success, nonzero on a NULL/destroyed pipeline or an undefined mode value
 * (the current mode is then left unchanged).
 */
int audio_pipeline_4ch_ulcnet_set_far_input_mode(
    AudioPipeline4ChUlcnet* p,
    UlcnetFarInputMode mode);

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

/**
 * Complete one hop with activity supplied by the product VAD (external VAD
 * only -- see the header comment). microphones_interleaved is [hop][4],
 * far_reference and out are [hop]. vad_external gates SRP observation and
 * freezes GSC adaptation exactly like the standard wrapper's vad_raw/vad_out
 * pair (a single flag feeds both here).
 *
 * out receives zeros for hop#0 and hop#1; from hop#2 on it carries the
 * enhanced (or fail-open identity) beamformed error, 2 hops behind the
 * input (see the timing contract above).
 *
 * @return FOUR_AEC_NR_RES_OK on success; the core's own status codes
 *         otherwise (the wrapper resets itself on internal DSP errors, same
 *         convention as audio_pipeline_4ch_process_with_activity()).
 */
int audio_pipeline_4ch_ulcnet_process_with_activity(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_external,
    float* out);

/**
 * Reset all core/SRP/GSC/WOLA/ULCNet-chain state, restart the 2-hop timing
 * preamble, and call model->reset (if set). Never touches the model
 * callback installation itself.
 */
void audio_pipeline_4ch_ulcnet_reset(AudioPipeline4ChUlcnet* p);

/**
 * Destroy an instance. For caller-owned memory this is idempotent and never
 * frees the pool; for a heap-created handle it frees the one owned pool, so
 * call once as with free().
 */
void audio_pipeline_4ch_ulcnet_destroy(AudioPipeline4ChUlcnet* p);

/* ============================================================================
 * Read-only accessors
 * ========================================================================== */

int audio_pipeline_4ch_ulcnet_hop_size(const AudioPipeline4ChUlcnet* p);
int audio_pipeline_4ch_ulcnet_fft_size(const AudioPipeline4ChUlcnet* p);
int audio_pipeline_4ch_ulcnet_n_freqs(const AudioPipeline4ChUlcnet* p);
int audio_pipeline_4ch_ulcnet_sample_rate(const AudioPipeline4ChUlcnet* p);

/**
 * Read-only view of the beamformed-error hop reconstructed during the most
 * recent successful process call (hop_size samples; itself one hop behind
 * that call's input -- see the timing contract). All zeros before the first
 * process call and after reset(). Valid until the next process/reset/
 * destroy. NULL for a NULL/destroyed handle. Exposed for tests, debugging,
 * and latency verification.
 */
const float* audio_pipeline_4ch_ulcnet_last_beamformed_error(
    const AudioPipeline4ChUlcnet* p);

/**
 * Copy the shared delay state observed by the most recent successful
 * process call (drives the fail-open/bypass and model-reset policy above).
 * Returns 0 on success, nonzero on a NULL out or NULL/destroyed handle.
 */
int audio_pipeline_4ch_ulcnet_last_delay(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResDelayState* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_4CH_ULCNET_H */
