/**
 * audio_pipeline_4ch_ulcnet.h — four-channel spatial pipeline with the
 * Align-ULCNet neural post-filter as its post stage.
 *
 * Architecture per hop (hop = 256 samples @ 16 kHz, fixed):
 *
 *   4x linear AEC lanes + one shared far aligner (FourAecNrRes process_pre)
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
 * FAR-INPUT DEPLOYMENT CONTRACT: the model always receives the shared AEC
 * seam's pre.aligned_ref. On every hop the shared delay ring cannot yet serve
 * the applied offset -- before acquisition under MATCHED, and throughout the
 * ring-fill window under FIXED -- that seam carries the RAW far hop, so the
 * model still runs on real reference audio and D handles the remaining
 * offset; from the first hop the ring can serve it, the seam carries the
 * aligned far consumed by every PBFDKF lane. The switch is whole-hop and
 * coincides with pre.delay.solid, so a hop is never part raw and part
 * shifted. Runtime mode
 * selection is intentionally absent from this production API and remains an
 * offline sweep option only. A published model descriptor must carry
 * ULCNET_FAR_ALIGNED.
 *
 * MODEL CALLBACK POLICY (first version):
 *
 *   - The model is STEPPED (infer() invoked) for every emitted frame whenever
 *     a callback is installed, including before matched-delay acquisition --
 *     except during the identity reprime described below.
 *   - No model set, model->infer == NULL, or infer() returning nonzero
 *     => FAIL-OPEN: the error spectrum passes through the synthesis chain
 *     unchanged (identity), keeping the timing path constant. A successful
 *     infer() whose output contains ANY non-finite value is likewise
 *     discarded (identity frame) -- NaN/Inf never reaches the WOLA.
 *   - On a delay change event (delay.changed), or FIXED's first transition
 *     from ring-fill raw far to usable aligned far: model->reset (if set) is
 *     called so the runtime flushes its far attention ring + logit history,
 *     and the beamform WOLA accumulator AND the one-hop far buffer are
 *     cleared (the core reset its lanes' analysis history at the same
 *     boundary — mixing spectra from opposite sides of the realignment
 *     would corrupt the seam).
 *   - audio_pipeline_4ch_ulcnet_reset() also calls model->reset (if set).
 *
 * IDENTITY REPRIME ACROSS AN ALIGNMENT BOUNDARY (option A):
 *
 *   The reset above flushes the runtime's recurrent state, but the C-side
 *   ULCNet STFT states keep running, so the analysis windows already in
 *   flight still STRADDLE the boundary: their 512-sample spans cover one hop
 *   pushed before the switch and one pushed after (on the error branch, the
 *   far branch, or both). Stepping the model on such a frame would rebuild,
 *   from a half-stale error/far pair, exactly the state the reset cleared.
 *
 *   Policy: starting with the boundary hop, the next
 *   AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES emitted frames take the
 *   identity (error passthrough) path and the model is NOT stepped -- no
 *   infer() call, so no K/V ring entry, logit-history entry or GRU hidden
 *   update happens on straddling input. Stepping AND applying resume
 *   together on the first frame whose error and far analysis windows contain
 *   exclusively post-switch hops. A second boundary inside a reprime re-arms
 *   the counter (it never accumulates).
 *
 *   Derivation of the constant (MEASURED by the straddle-derivation test in
 *   tests/test_audio_pipeline_4ch_ulcnet.c, never assumed here): BOTH
 *   branches of this wrapper are pushed one hop behind the input -- the beam
 *   WOLA closes one hop late, and the far branch is delayed one hop to match
 *   it -- so the hop pushed at the boundary hop T still belongs to the
 *   pre-switch input hop T-1 (and is in fact the cleared OLA / cleared far
 *   buffer). With a centered 512/256 analysis spanning two pushed hops, the
 *   frames emitted at hops T and T+1 both cover that slot and straddle; the
 *   frame emitted at hop T+2 covers the two post-switch hops and is clean.
 *   That is TWICE the mono wrapper's reprime (audio_pipeline_ulcnet.h: 1),
 *   which pushes both branches from the current hop. The test derives the
 *   count from a marker run that contains NO boundary at all -- so the
 *   reprime logic never participates in its own measurement -- and asserts
 *   it equals the constant below, per branch.
 *
 *   Compute: a reprime frame SKIPS inference, so per-hop compute DROPS for
 *   those frames; it never doubles. The WOLA/STFT path is untouched, so the
 *   2-hop latency contract holds unchanged across a boundary.
 *
 *   Option B (keep stepping the model through the straddling frames and keep
 *   applying its output) is DEFERRED pending an audio A/B: it trades this
 *   version's short identity stretch for recurrent state built on half-stale
 *   frames. Do not switch policies without that A/B.
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
 * Alignment-boundary identity reprime
 * ========================================================================== */

/**
 * Emitted frames that still straddle an alignment boundary in this wrapper,
 * i.e. the length of the identity reprime armed at every generation change
 * (see "IDENTITY REPRIME" in this header's preamble).
 *
 * = 2: both branches are pushed one hop behind the input (beam WOLA lag +
 * the matching one-hop far compensation), so the slot pushed at the boundary
 * hop still belongs to the pre-switch input hop, and the centered 512/256
 * analysis spans two pushed slots. Derived and asserted branch by branch by
 * the straddle-derivation test; do not edit this value without re-running
 * it.
 */
enum { AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES = 2 };

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

#define AUDIO_PIPELINE_4CH_ULCNET_DESCRIPTOR_VERSION 1u
/* Carve order: self, core, srp, gsc, gsc_spectrum, gsc_weights, fft, ifft,
 * ola, synth_win, beam_hop (the ULCNet analysis/synthesis states, frame
 * scratch, one-hop far buffer and the chain's shared
 * sqrt-Hann window table live inside `self`). The one carved `fft` handle
 * serves BOTH the beamform WOLA and the ULCNet chain (same 512 size;
 * strictly sequential use within a hop). Bump together with the
 * build-flags-hash token string forever after.
 * v2: self grew the one-hop far-compensation buffer + far_input_mode.
 * v3: the ULCNet chain moved onto the shared carved FftHandle (its structs
 * embed their own FFT scratch) and self grew the shared window table
 * (ulcnet_window) replacing the per-struct window copies.
 * v4 grows the self-resident core config with explicit delay controls and
 * the pre-only selector.
 * v5 grows the self-resident UlcnetModel copy by the published model-I/O
 * descriptor pointer. v6 removes the obsolete runtime far-mode field.
 * v7 adds the identity-reprime counter to self.
 * v8: the self-resident core config grew the backward-quarantine delay
 * guard fields (left at their OFF defaults here), moving the self block.
 * v9 carries the core's own layout 10 -> 11 (its control block gained the
 * shared-delay change admission and the realign lane counters). Nothing in
 * THIS layer's carve changed, but the core sub-pool it composes grew, so
 * every offset after `core` and this layer's total moved with it -- and
 * build_flags_hash cannot say so, since it folds in the core's carve-token
 * hash, which a control-block-only change leaves alone.
 * v10 carries GSC's covariance layout from P[F][M][M] to P[M][M][F]. The
 * state count is unchanged, but its pointer tables and pool size moved.
 * Version 11: sizeof(Aec) grew (the suppressor gained its runtime
 * far-active floor retarget state), so every AEC carved out of this pool
 * moves the total and the offsets after it. Carve order and buffer set are
 * unchanged, so build_flags_hash does not move -- this counter is the only
 * signal.
 * Version 12 carries the core's own layout 12 -> 13 (its control block gained
 * the per-hop stage-timing record). Nothing in THIS layer's carve changed;
 * the composed pre-only core sub-pool simply grew, and build_flags_hash
 * cannot say so for the same carve-token reason as v9.
 * Version 13 carries the core's own layout 13 -> 14 (sizeof(Aec) grew: each
 * lane gained its per-hop stage-timing record). Same carve-token reason. */
#define AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION 13u

/**
 * Fixed-width descriptor for a caller-owned static-memory pool. Same 32-byte
 * fixed-width-integer shape and same field meanings as 4aec_nr_res.h's
 * FourAecNrResMemReq (descriptor_version/layout_version/backend_id/
 * build_flags_hash/alignment/reserved/bytes) -- see that header's own doc
 * comment for the full rationale rather than repeating it here. Like
 * AudioPipeline4ChMemReq, this layer's build_flags_hash folds in the core's
 * own build_flags_hash, so any core change that moves a carve TOKEN also
 * invalidates every persisted composite descriptor here. A core change that
 * only grows its control block moves no token and no hash on either layer:
 * that one is caught by this layer's own layout_version, which is why it
 * bumps with the core's (see the carve-order list above).
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

/* ============================================================================
 * Config and lifecycle
 * ========================================================================== */

/* Returns the trained 16 kHz / frame-FFT 512 / hop 256 defaults for the
 * PRE-ONLY profile this wrapper is the only consumer of: core.fft_size = 512,
 * core.enable_post = 0 and core.enable_cng = 0.
 *
 * Align-ULCNet replaces the post-beam RES/NR/CNG stage entirely, so with
 * enable_post = 0 the core builds no denoiser, no suppressor, no comfort
 * noise and no post iFFT. Every post-only field must therefore keep the value
 * this function returns, and is REJECTED otherwise rather than ignored:
 *
 *   core.enable_post  = 0
 *   core.enable_cng   = 0
 *   core.legacy_amin  = 0
 *   core.nr_mode      = MMSE_LSA_NR_BALANCED   (the enum has no "disabled"
 *                       value and no denoiser exists, so this is a required
 *                       sentinel, not a strength choice)
 *   auto_vad_*        = the audio_pipeline_4ch defaults (the built-in energy
 *                       VAD is unreachable here: only
 *                       _process_with_activity() exists, and it takes the
 *                       caller's VAD)
 *
 * Rejecting rather than silently accepting is deliberate: a caller who
 * believes it configured NR finds out at init, not on a board.
 *
 * AudioPipeline4ChConfig is SHARED with the standard 4-channel wrapper, so
 * this application's accepted set is narrower than that struct's. That is the
 * intended divergence and is why the list above is spelled out here: the
 * alternative -- accepting the fields and ignoring them -- is what would make
 * one struct silently mean two different things.
 *
 * The pre-stage fields (delay_*, aec_preset, filter_length,
 * capture_proxy_channel, max_delay_ms) are live and set normally. */
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
 * mid-stream.
 *
 * A model that publishes a model-I/O contract (model->io_descriptor != NULL,
 * which must then outlive this pipeline) is rejected unless that descriptor
 * matches the fixed aligned-far production ABI.
 *
 * Returns 0 on success, nonzero on a NULL/destroyed pipeline or a
 * invalid model-I/O descriptor.
 */
int audio_pipeline_4ch_ulcnet_set_model(
    AudioPipeline4ChUlcnet* p,
    const UlcnetModel* model);

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
 * preamble, and call model->reset (if set). Any pending identity reprime is
 * dropped: the analysis history is zeroed here, so the frames emitted after
 * a reset straddle nothing. Never touches the model callback installation
 * itself.
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
 * process call (drives the model-reset policy and diagnostics above).
 * Returns 0 on success, nonzero on a NULL out or NULL/destroyed handle.
 */
int audio_pipeline_4ch_ulcnet_last_delay(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResDelayState* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_4CH_ULCNET_H */
