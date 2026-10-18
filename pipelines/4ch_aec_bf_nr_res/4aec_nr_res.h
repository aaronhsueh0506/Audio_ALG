/**
 * 4ch_aec_bf_nr_res/4aec_nr_res.h
 *
 * Four-microphone linear-AEC front end with one shared far aligner.
 *
 * The external owner supplies SRP-PHAT/GSC. This module owns:
 *
 *   1 shared DelayAec3 matcher
 *   4 independent linear AEC adaptive filters
 *   the coherent projection of the four lane contexts onto one mono frame
 *
 * It does not select one microphone's RES gain and does not replicate the
 * residual-echo, noise-reduction or synthesis stages per channel. Those are
 * not part of this module: the application carves them and chains them, once,
 * over the SeSpecFrame that four_aec_fuse() returns
 * (pipelines/common/se_stage.h). Per hop:
 *
 *   four_aec_front_process()   shared delay + four linear AEC lanes; the lane
 *                              spectra go to the external SRP-PHAT/GSC
 *   (SRP-PHAT, GSC)            effective weights and the beamformed error
 *   four_aec_fuse()            project near / R^2 / comfort noise with those
 *                              weights -> SeSpecFrame
 *   stages                     se_res_stage_gain(), se_anr_stage_gain(), ...
 *   sk_min_f32 -> sk_capply_gain_f32 -> aec_cng_fill -> ac_istft_push
 *
 * The file layout, lifecycle, and naming intentionally mirror
 * pipelines/mono_aec_nr_res/audio_pipeline.h:
 *
 *   Caller-owned pool (board/static path):
 *
 *     FourAecNrResConfig cfg;
 *     FourAecNrResMemReq req;
 *     cfg = four_aec_nr_res_default_config(16000);
 *     four_aec_nr_res_get_mem_requirements(&cfg, &req);
 *     void* pool = platform_alloc(req.bytes, req.alignment);
 *     FourAecNrRes* p =
 *         four_aec_nr_res_init_ex(pool, req.bytes, &cfg, &req);
 *     ...
 *     four_aec_nr_res_destroy(p);  // never releases caller-owned pool
 *     platform_free(pool);
 *
 *   Heap convenience (desktop/test path):
 *
 *     FourAecNrRes* p = four_aec_nr_res_create(&cfg);
 *     ...
 *     four_aec_nr_res_destroy(p);  // releases create()'s one pool
 *
 * See pipelines/README.md for the call flow and
 * pipelines/4ch_aec_bf_nr_res/README.md for the architecture writeup and the
 * parity-limitation note on bounded vs unbounded R2 in the post-beam RES.
 */
#ifndef FOUR_AEC_NR_RES_H
#define FOUR_AEC_NR_RES_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"
#include "anr.h"
#include "common/se_stage.h"    /* SeSpecFrame */

#ifdef __cplusplus
extern "C" {
#endif

#define FOUR_AEC_NR_RES_CHANNELS 4

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

#define FOUR_AEC_NR_RES_DESCRIPTOR_VERSION 1u
/* Carve layout version of the pool: the control block, the four lane pools,
 * the working buffers (mic lane, aligned reference and, when cfg.enable_post,
 * the fuse buffers) and the shared delay estimator and ring. Per-lane spectra
 * are borrowed from each AEC instance and are valid only until that lane is
 * processed or reset again. The residual-echo, noise-reduction and synthesis
 * stages are carved by the application, not here. Any change of pool size,
 * carve order or descriptor content bumps it. Query a fresh descriptor for
 * every build; do not persist or synthesize one from this version number. */
#define FOUR_AEC_NR_RES_LAYOUT_VERSION 20u
#define FOUR_AEC_NR_RES_BACKEND_KISS 1u
#define FOUR_AEC_NR_RES_BACKEND_NE10 2u

/**
 * Fixed-width descriptor for a caller-owned static-memory pool.
 *
 * Query it at init time and pass the same value to init_ex(). Do not cache it
 * across library, backend, compiler-flag, or config changes.
 *
 * Same 32-byte fixed-width-integer shape and same field meanings as
 * pipelines/mono_aec_nr_res/audio_pipeline.h's `AudioPipelineMemReq` (descriptor_version =
 * this struct's own ABI version, checked first; layout_version = this
 * file's carve order/buffer-set version; backend_id = compile-time FFT
 * backend as a small integer, compared with plain `==`, never `strcmp`;
 * build_flags_hash = FNV-1a-32 over this file's own carve structure, NOT
 * over AecConfig/AnrConfig tunable values; reserved = always 0, VALIDATED
 * not assumed, because `expected` may originate from persisted/transmitted
 * bytes) -- see that header's own doc comment for the full rationale
 * rather than repeating it here.
 */
typedef struct FourAecNrResMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} FourAecNrResMemReq;

/* Bare _Static_assert, same as audio_pipeline.h's AudioPipelineMemReq pin --
 * this header is already only ever included inside an `extern "C" { ... }`
 * block for C++ callers (see above), so there is nothing a
 * static_assert/_Static_assert compatibility macro would buy here that the
 * sibling mono header doesn't already get for free. */
_Static_assert(
    sizeof(FourAecNrResMemReq) == 32,
    "FourAecNrResMemReq must be exactly 32 bytes");
_Static_assert(
    offsetof(FourAecNrResMemReq, descriptor_version) == 0,
    "descriptor_version offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, layout_version) == 4,
    "layout_version offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, backend_id) == 8,
    "backend_id offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, build_flags_hash) == 12,
    "build_flags_hash offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, alignment) == 16,
    "alignment offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, reserved) == 20,
    "reserved offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, bytes) == 24,
    "bytes offset");

/* ============================================================================
 * Config, frame handoff and status
 * ========================================================================== */

typedef enum FourAecNrResStatus {
    FOUR_AEC_NR_RES_OK = 0,
    FOUR_AEC_NR_RES_INVALID_ARGUMENT = -1,
    FOUR_AEC_NR_RES_SEQUENCE_ERROR = -2,
    FOUR_AEC_NR_RES_DSP_ERROR = -3
} FourAecNrResStatus;

typedef struct FourAecNrResConfig {
    int sample_rate;              /* 16000 or 48000                         */
    int fft_size;                 /* 0=default; 256/512 @16k, 1024 @48k   */
    int filter_length;            /* 0=rate default                         */
    AecDelayMode delay_mode;      /* shared MATCHED/FIXED/EXTERNAL policy   */
    int delay_num_filters;        /* MATCHED bank size [1,5]                */
    int fixed_delay_samples;      /* FIXED native-rate samples; -1 otherwise*/
    int capture_proxy_channel;    /* shared matcher input, [0,3]            */
    /* Backward-jump quarantine on a shared-delay CHANGE. DEFAULT OFF, so the
     * shipped path is unchanged. MATCHED only -- FIXED and EXTERNAL_ALIGNED
     * never re-decide an alignment, so there is nothing to quarantine.
     *
     * Same mechanism as lib/aec's own delay_backward_quarantine_* (see
     * aec.h for the full derivation and the measured pre-echo scene),
     * applied to the SHARED estimate:
     * a shared estimate strictly EARLIER than the accepted delay is held for
     * delay_backward_quarantine_s worth of hops while the estimator's own
     * capture lane still cancels at the applied alignment, and is ADOPTED at
     * expiry. A collapse in cancellation adopts it immediately. A FORWARD
     * estimate -- a larger delay, which pre-echo mis-attribution cannot
     * produce -- is never held.
     *
     * ONE lane, not any lane, and specifically cfg.capture_proxy_channel:
     * that is the microphone the shared estimator is actually fed from, so
     * it is the only lane whose cancellation is evidence about the estimate
     * being judged. Judging ANY of the four instead would let a single
     * microphone's surviving old reflection hold the shared update back for
     * the whole array.
     *
     * First acquisition is NOT guarded: with nothing accepted yet there is
     * no alignment to protect, and lib/aec makes the same split (Path A
     * keeps its own delay_acquire_protect_converged).
     *
     * The reading the lane answers with is 0 until its ERLE machinery has
     * run, so an unavailable metric leaves this inert rather than blocking.
     * That machinery is alive here: the lanes run enable_res=0 with
     * return_res_context=1, the seam configuration in which lib/aec caches
     * windowed ERLE. */
    int delay_backward_quarantine_enabled;  /* bool                         */
    float delay_backward_quarantine_s;      /* window, seconds; default 1.0 */
    /* Shared matched-filter duty-cycle throttle: once the published estimate
     * has held still for delay_est_init_s (fixed at lib/aec's own 0.3 s; not
     * exposed here), analyse the matched filter 1-in-round(delay_est_period_s
     * / hop_s)/5 hops instead of every hop. Same windows, same re-arm
     * triggers and same ERLE watchdog as lib/aec's own per-lane machine
     * (update_shared_delay() in 4aec_nr_res.c), but this field is the
     * WRAPPER's own contract, not a silent read-through of AecConfig's
     * field of the same name: derive_dims_and_configs() writes it into the
     * per-lane config it builds, so a future change to lib/aec's own
     * default cannot silently retime the shared estimator out from under a
     * caller who never asked for that.
     *
     * 0.0 (the default) = full rate, no decimation -- matching lib/aec's own
     * default since AMEND_35 (2026-10-18; delay_est_period_s defaulting to
     * 0.5 could make the matched filter settle on a different lag than
     * Python's always-full-rate reference, not merely settle on it slower).
     * A positive value opts back into the throttle for an embedded
     * deployment that wants the compute saving (measured 17.0->4.3 us/hop
     * for the shared delay stage, see commit 07a51b1) and accepts that
     * parity cost. Range-checked like delay_backward_quarantine_s (finite,
     * [0, 3600]), mirroring lib/aec's own AEC_CK_RANGE_F bound. */
    float delay_est_period_s;
    float max_delay_ms;           /* MATCHED reference delay-line capacity   */
    AecPreset aec_preset;
    AnrMode nr_mode;
    int enable_post;              /* 1 = the frequency-domain post context is kept:
                                    * the lanes compute R^2 and comfort noise and
                                    * the core carves the fuse buffers
                                    * four_aec_fuse() needs; 0 = front only (the
                                    * lanes keep the linear half of the AEC
                                    * context: AEC_CONTEXT_LINEAR_ONLY). The
                                    * complete 4-channel pipelines require 1,
                                    * the neural 4-channel pipeline 0.        */
    /* The next four fields configure the application's own post-beam stages
     * (stage sizes and configurations: four_aec_nr_res_stage_configs()). The
     * core validates them and reads none. */
    int enable_res;               /* bool: run the post-beam RES; when 0, G_res
                                    * is unity and CNG has nothing to fill    */
    int enable_nr;                /* bool: run the MMSE-LSA denoiser         */
    int enable_cng;               /* bool                                   */
    int legacy_amin;              /* bool: do not fold R2 into NR prior      */
} FourAecNrResConfig;

/**
 * Per-hop view of the single shared reference alignment.
 *
 * `changed` marks the hop on which a NEW USABLE alignment generation begins:
 * MATCHED sets it on first acquisition, on a relock after an unlocked stretch
 * EVEN IF the delay value is unchanged, and on an ADMITTED delay change while
 * locked -- i.e. on any not-usable -> usable transition of `solid`
 * (delay_samples is never negative here, so usability is solid alone), plus
 * admitted value changes on top of that. This mirrors lib/aec's per-hop
 * AEC_LINEAR_DELAY_CHANGED, which the mono pipelines consume: there the
 * "nothing accepted yet" state is spelled current_delay == -1, so a relock
 * always crosses a sentinel; here the accepted delay is a plain non-negative
 * sample count, so the transition has to be tracked explicitly.
 *
 * A consumer that keeps history derived from the aligned reference -- STFT/OLA
 * tails, and any recurrent model state stepped over the far branch -- must
 * flush it on `changed`, BEFORE the frame it produces for this hop. Gating
 * only on `solid` is not enough: a consumer that steps its model while
 * unlocked (for constant per-hop compute) and applies only while solid still
 * needs `changed` to tell it that everything it accumulated over the unlocked
 * stretch was built on a reference the estimator had not vouched for.
 *
 * FIXED and EXTERNAL_ALIGNED never set `changed`; both follow lib/aec, where
 * nothing bumps the generation during processing without an estimator.
 *
 * ADMITTED, for a change while locked, is lib/aec's own Path-B trio: the
 * movement must exceed 32 native samples AND be offered again -- within 16 --
 * before the held candidate's 3 hops of life run out, and only then is it
 * applied. Each generation costs an IR shift on all four lanes plus, when the
 * alignment retards, their far history, so a movement seen once and gone is
 * not worth one. What survives the trio is a SUSTAINED movement, correct or
 * not: this is a repeated-evidence rule, not a correctness test (that is
 * delay_backward_quarantine_enabled's job, on its own evidence and bound).
 * First acquisition is not subject to it -- nothing is applied yet to
 * protect. `delay_samples` therefore never moves on a hop without `changed`:
 * the alignment and the filter shift that follows it are one event.
 *
 * `solid` = "a usable accepted alignment generation exists". Under MATCHED it
 * is raised by the same acceptance test that writes the applied delay, so it
 * never leads `delay_samples`, and it is sticky: a short confidence dip does
 * not retract an alignment the audio path is still applying, and only
 * reset() clears it. Under FIXED it is the ring-fill state (raw far below,
 * shifted far from that hop on); under EXTERNAL_ALIGNED it is always 1.
 */
typedef struct FourAecNrResDelayState {
    int delay_samples;
    float confidence;
    int solid;
    int changed;        /* reset external STFT/OLA history when non-zero */
    uint64_t estimator_calls;
    int estimator_updates;
} FourAecNrResDelayState;

/**
 * What the front end returns for one hop (see four_aec_front_process()). Every
 * pointer is a read-only view into a lane's or the core's own per-hop buffer;
 * nothing is copied. The view is valid until the next four_aec_front_process(),
 * four_aec_nr_res_reset() or four_aec_nr_res_destroy() on the same instance --
 * the lanes overwrite these buffers in place on their next hop, and a reset does
 * not clear them.
 */
typedef struct FourAecFront {
    FourAecNrResDelayState delay;
    int hop_size;
    int n_channels;               /* always FOUR_AEC_NR_RES_CHANNELS        */
    int n_freqs;

    /**
     * The exact linear-AEC error spectra of this hop, one per channel
     * (n_freqs bins, audio scale). An external SRP-PHAT/GSC consumes these
     * instead of performing a duplicate STFT.
     */
    const Complex* linear_spectra[FOUR_AEC_NR_RES_CHANNELS];

    /**
     * The time-domain hop underlying each linear_spectra[channel] (hop_size
     * samples, before any synthesis). An external time-domain beamformer that
     * performs its own sqrt-Hann analysis consumes this rather than the
     * standalone AEC's separately limited return value.
     */
    const float* formed_hop[FOUR_AEC_NR_RES_CHANNELS];

    /**
     * The delay-aligned far reference actually fed to every AEC lane this hop
     * (hop_size samples): the caller's ref stream shifted by
     * delay.delay_samples on every hop the shared ring can serve that offset,
     * and the caller's RAW ref hop on the hops before it can. The switch is
     * whole-hop and coincides with `delay.solid` under FIXED. Intended
     * consumer: an external neural post-filter that needs the SAME aligned far
     * the linear filters consumed.
     */
    const float* aligned_ref;
} FourAecFront;

typedef struct FourAecNrRes FourAecNrRes;

/* ============================================================================
 * Config and lifecycle
 * ========================================================================== */

/**
 * Return the default config for sample_rate. This is the direct counterpart
 * of audio_pipeline_default_config().
 */
FourAecNrResConfig four_aec_nr_res_default_config(int sample_rate);

/**
 * Query the complete 16-byte-aligned caller pool required by cfg.
 * Performs no allocation and returns 0 on success.
 */
int four_aec_nr_res_get_mem_requirements(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemReq* out);

/**
 * Place the complete four-AEC/NR/RES pipeline in caller-owned memory.
 *
 * mem must satisfy the freshly queried alignment and byte requirement.
 * No heap allocation is performed by this path, including inside the four
 * AECs, NR, and FFT backend. A dirty/poisoned pool is accepted.
 */
FourAecNrRes* four_aec_nr_res_init(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg);

/**
 * Static init plus a stale-descriptor gate. expected should be the descriptor
 * returned by a fresh get_mem_requirements() call for this exact build/config.
 * NULL expected is equivalent to four_aec_nr_res_init().
 *
 * When `expected` is non-NULL, every one of the following must hold against
 * the CURRENT build's own get_mem_requirements(cfg, ...) before anything is
 * carved (mirrors pipelines/mono_aec_nr_res/audio_pipeline.h's audio_pipeline_init_ex()):
 *
 *   1. expected->descriptor_version == current.descriptor_version
 *   2. expected->layout_version == current.layout_version
 *   3. expected->backend_id == current.backend_id
 *   4. expected->build_flags_hash == current.build_flags_hash
 *   5. expected->alignment == current.alignment
 *   6. expected->reserved == 0
 *   7. expected->bytes >= current.bytes
 *   8. bytes (the pool ACTUALLY handed to this call) >= current.bytes
 *
 * Only once all eight hold does this proceed exactly as
 * four_aec_nr_res_init() would. Returns NULL on the first failing check
 * (no diagnostic is printed -- unlike the mono pipeline's
 * AP_LOG_ERR()-per-field messages, this module returns NULL silently on
 * every rejection path, mono or four-channel; it links no stdio symbols).
 */
FourAecNrRes* four_aec_nr_res_init_ex(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg,
    const FourAecNrResMemReq* expected);

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

/**
 * Stage 1: the shared aligner and the four linear AEC filters.
 *
 * microphones_interleaved is [hop][4], ref is [hop]. Both are read only. On
 * OK, *out is the borrowed front view of this hop (see FourAecFront). The
 * previous hop's view and every SeSpecFrame fused from it are invalidated.
 *
 * @return FOUR_AEC_NR_RES_OK on success; FOUR_AEC_NR_RES_INVALID_ARGUMENT on a
 *         NULL p/microphones_interleaved/ref/out or a non-finite sample;
 *         FOUR_AEC_NR_RES_DSP_ERROR when a lane refused the hop (the instance
 *         has been reset).
 */
int four_aec_front_process(
    FourAecNrRes* p,
    const float* microphones_interleaved,
    const float* ref,
    FourAecFront* out);

/**
 * Stage 2: project the front frame with the beamformer's effective weights and
 * return the frame the frequency-domain stages consume.
 *
 * weights is channel-major Complex[4][n_freqs] and uses
 *
 *   output[bin] = sum(weights[channel,bin] * input[channel,bin])
 *
 * without conjugation. The module projects the four lanes' near spectra,
 * residual-echo powers and comfort noise with those weights, and fills *frame
 * (pipelines/common/se_stage.h): e = the beamformed error, y = the projected
 * near spectrum, x = lane 0's far spectrum, r2 and cn = the projected
 * residual-echo and comfort-noise PSDs, far_hop = the aligned far, max_dt /
 * max_sat / all_converged taken over the lanes the weights give a nonzero
 * response, clock_drift and the delay flags of the shared aligner;
 * g_res_ctx is NULL (the lanes compute no gain of their own).
 *
 * beamformed_error is the mono error spectrum produced by the SAME external
 * beamformer invocation as `weights` (the bundled gsc_process_with_weights()
 * returns both atomically). Both arrays are checked for finite values, but
 * their semantic equality cannot be inferred from samples alone and is a
 * caller contract. Pass NULL to have the error reconstructed from `weights`;
 * the reconstruction cannot disagree with them. The frame's e then aliases an
 * instance-owned buffer; a supplied spectrum stays caller-owned and must
 * outlive the frame.
 *
 * All pointers in *frame alias instance-owned per-hop storage (a supplied
 * beamformed_error excepted) and are valid until the next
 * four_aec_front_process(), reset or destroy. Requires cfg.enable_post.
 *
 * @return FOUR_AEC_NR_RES_OK on success; FOUR_AEC_NR_RES_INVALID_ARGUMENT on a
 *         NULL p/weights/frame, an all-zero or non-finite weight set, or a
 *         core with enable_post = 0 (nothing is changed, the caller may
 *         correct and retry); FOUR_AEC_NR_RES_SEQUENCE_ERROR when no front
 *         frame has been processed since construction or the last reset;
 *         FOUR_AEC_NR_RES_DSP_ERROR when the lanes do not share one far-end
 *         spectrum or a spectrum is non-finite (the instance has been reset).
 */
int four_aec_fuse(
    FourAecNrRes* p,
    const Complex* weights,
    const Complex* beamformed_error,
    SeSpecFrame* frame);

/**
 * The configurations the application's own post-beam stages need, derived on
 * the core's grid exactly as the core derived its lane configuration: res_cfg
 * sizes and initialises the AecRes (aec_res_get_mem_size / aec_res_init) and
 * nr_cfg the denoiser (se_anr_stage_mem_size / se_anr_stage_init). Returns 0,
 * or -1 for a NULL argument or a config the core would reject.
 */
int four_aec_nr_res_stage_configs(
    const FourAecNrResConfig* cfg,
    AecConfig* res_cfg,
    AnrConfig* nr_cfg);

/**
 * Reset all delay/AEC state. Invalidates the front view and every frame fused
 * from it.
 *
 * A caller holding an outstanding FourAecFront must stop reading its pointers
 * the instant this is called. This is an API-contract rule, not a
 * memory-safety guarantee enforced by this call: formed_hop and
 * linear_spectra[channel] alias lane buffers that aec_reset() does not clear or
 * overwrite, so a violation shows as stale content, not as a fault.
 * four_aec_fuse() refuses to run until a new front frame has been processed.
 */
void four_aec_nr_res_reset(FourAecNrRes* p);

/**
 * Destroy an instance. For caller-owned memory this is idempotent and never
 * frees the pool; the caller releases/reuses it afterward. For a heap-created
 * handle it frees the one owned pool, so call once as with free().
 */
void four_aec_nr_res_destroy(FourAecNrRes* p);

/* ============================================================================
 * Heap convenience
 * ========================================================================== */

/**
 * Heap convenience wrapper over get_mem_requirements() + init(). All memory
 * is allocated here; front_process/fuse perform no allocation. Returns NULL for
 * an invalid config or allocation failure.
 */
FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg);

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int four_aec_nr_res_hop_size(const FourAecNrRes* p);
int four_aec_nr_res_fft_size(const FourAecNrRes* p);
int four_aec_nr_res_n_freqs(const FourAecNrRes* p);
int four_aec_nr_res_sample_rate(const FourAecNrRes* p);

/* Structural audit hooks. The matcher count is 1 only in MATCHED mode and 0
 * in FIXED/EXTERNAL; the linear AEC count is 4. */
int four_aec_nr_res_matched_filter_count(const FourAecNrRes* p);
int four_aec_nr_res_linear_aec_count(const FourAecNrRes* p);

/* Group 6 instrumentation, read-only: sum of aec_far_fft_real_compute_count()
 * across all four lanes (cumulative since construction or the last
 * four_aec_nr_res_reset()). Lane 0 runs its own far-end FFT every hop
 * (aec_process_context()); lanes 1-3 borrow lane 0's spectrum instead of
 * recomputing it (aec_process_context_shared_far()) -- so this total
 * increments by exactly 1 per four_aec_front_process() call, not 4.
 * Intended for tests/instrumentation proving the cross-lane far-FFT
 * sharing is genuinely active; has no effect on any processing. */
long four_aec_nr_res_far_fft_real_compute_count(const FourAecNrRes* p);

/* Realign instrumentation, read-only and cumulative since construction or the
 * last four_aec_nr_res_reset(): how the four lanes'
 * aec_apply_external_realign() calls landed.
 * Every hop that publishes `changed` sweeps all four lanes, so
 * the pair grows by exactly 4 on such a hop and not at all otherwise -- warm
 * (returned 1) means the learned IR was shifted and cancellation survived the
 * move, soft (returned 0) means the lane took a filter-only reset (taps and
 * far history cleared, WOLA/framing untouched) and re-converges from zero. A
 * sweep that adds fewer than 4 means a lane rejected the call outright, which
 * is a wiring fault, not a soft realign. Nothing branches on either count. */
long four_aec_nr_res_realign_warm_lane_count(const FourAecNrRes* p);
long four_aec_nr_res_realign_soft_lane_count(const FourAecNrRes* p);

/* The shared-delay movement currently held for confirmation, in native
 * samples, or -1 when none is held (including FIXED/EXTERNAL_ALIGNED, which
 * never re-decide an alignment). See FourAecNrResDelayState's `changed` for
 * the rule; diagnostic only. */
int four_aec_nr_res_pending_delay_candidate(const FourAecNrRes* p);

/* Matched-filter duty-cycle engagement census, read-only and cumulative since
 * construction or the last four_aec_nr_res_reset(): _total counts every hop
 * through the shared delay stage, _run counts those that actually ran the
 * matched-filter analysis. _run / _total is therefore the MEASURED engagement
 * -- the saving stops being a design intention and becomes a number a caller
 * can print -- which matters because the machine re-arms full rate whenever
 * the estimate is SEEN to move, so on an unstable echo path the realised
 * engagement is nothing like the steady-state 1-in-K. Seen, not moved: a
 * decimated hop publishes nothing, so a movement mid-period is picked up at
 * the next scheduled analysis and full rate resumes from there -- up to one
 * duty period of extra wait (~96 ms on all three shipped grids). The ERLE
 * watchdog can cut that short, but only where a lane's windowed ERLE ever
 * cleared the 6 dB collapse threshold to arm it.
 *
 * Both are 0 outside AEC_DELAY_MATCHED, where no matched filter is built and
 * there is no cost to decimate. Same counters, same contract, and the same
 * placement at the decision site as lib/aec's own pair (AecDebugStatus).
 * Diagnostic: nothing in the signal path reads them.
 *
 * The estimator is FED on every analysed hop -- decimation applies to the
 * matched-filter ANALYSIS only, never to the audio the estimator sees -- so
 * duty_hops_total is BY CONSTRUCTION equal to FourAecNrResDelayState's
 * estimator_calls (the accessor is an alias of the same counter; a second
 * field would just be a copy that could drift). */
unsigned long long four_aec_nr_res_duty_hops_total(const FourAecNrRes* p);
unsigned long long four_aec_nr_res_duty_hops_run(const FourAecNrRes* p);

/* ============================================================================
 * Diagnostic memory breakdown
 * ========================================================================== */

typedef struct FourAecNrResMemBreakdown {
    size_t aec_bytes;       /* four AEC pools combined */
    size_t wrapper_bytes;   /* control block + shared aligner and fuse buffers */
    size_t total_bytes;
    int hop_size;
    int fft_size;
    int n_freqs;
} FourAecNrResMemBreakdown;

/** Optional module/wrapper split for board memory-budget reporting. */
int four_aec_nr_res_get_mem_breakdown(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemBreakdown* out);

/**
 * Per-stage wall-clock cost of the most recent hop, in microseconds.
 *
 * Diagnostic only: nothing in the pipeline reads these back, and they do not
 * affect processing.
 *
 *   delay_us     update_shared_delay() -- the shared delay estimator only.
 *                align_render()'s ring-buffer copy and the realign sweep are
 *                deliberately outside it; they are not delay estimation.
 *   frontend_us  summed over the four lanes: each lane's work from entry up
 *                to its main filter (mic HPF, saturation, mu/RSA prep,
 *                shadow filter). Lane-internal delay work is added to
 *                delay_us instead, where it is structurally zero because
 *                every lane is AEC_DELAY_EXTERNAL_ALIGNED.
 *   linear_us    summed over the four lanes: each lane's main adaptive
 *                filter, including the far-end FFT on the lane that computes
 *                it (lanes 1-3 borrow it and pay less here).
 *   lane_res_us  summed over the four lanes: each lane's AEC3 post/RES block,
 *                which runs because the lanes set return_res_context even
 *                though no lane applies a gain.
 *   fuse_us      four_aec_fuse(): the beamformer-weighted projection of the
 *                four lane contexts into one.
 *
 * The four AEC-side figures come from aec_get_last_timing() (aec.h), read once
 * per lane and summed here. The lanes run sequentially inside one loop, so the
 * sums are wall-clock costs of each stage across the whole loop. The stages do
 * not add up to the enclosing call: align_render(), the realign sweep, the
 * de-interleave, input validation and the AEC stages between the main filter
 * and the post block fall in the caller's remainder. The stages the
 * application chains after the fuse are timed by the application.
 *
 * RESOLUTION. Microseconds, so a stage faster than 1 us reads 0.
 *
 * OFF BY DEFAULT, IN TWO HALVES. A release build takes no clock reads and
 * every field reads 0. Build with -DFOUR_AEC_NR_RES_STAGE_TIMING=1 for
 * delay/fuse and -DAEC_STAGE_TIMING=1 for the four summed off the lanes
 * (delay/frontend/linear/lane_res); `make PROFILE=1` sets both. Setting one
 * alone is legible rather than broken: the other half simply reads 0.
 *
 * A target whose libc has no POSIX CLOCK_MONOTONIC combines the flags with
 * -DFOUR_AEC_NR_RES_NOW_US=<fn> (and lib/aec's own -DAEC_NOW_US=<fn>) to name
 * its microsecond timer. A substitute that returns a constant keeps the flags
 * on and reads 0 here, which this record cannot distinguish from a build
 * without them.
 *
 * LIFETIME. The lane-side half is cleared once four_aec_front_process() has
 * accepted its arguments, so a hop that then bails out reports zero for the
 * stages it never reached; fuse_us is cleared by four_aec_fuse() the same
 * way. A call rejected outright by its argument checks leaves the record
 * untouched.
 */
typedef struct FourAecNrResLastTiming {
     uint32_t delay_us;
     uint32_t frontend_us;
     uint32_t linear_us;
     uint32_t lane_res_us;
     uint32_t fuse_us;
} FourAecNrResLastTiming;

/* Copies the last hop's stage timings into `out`. A NULL/destroyed pipeline
 * zeroes `out` rather than failing, so a diagnostic caller needs no special
 * case. `out` must not be NULL. */
void four_aec_nr_res_get_last_timing(
    const FourAecNrRes* p,
    FourAecNrResLastTiming* out);

#ifdef __cplusplus
}
#endif

#endif /* FOUR_AEC_NR_RES_H */
