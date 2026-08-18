/**
 * 4ch_aec_bf_nr_res/4aec_nr_res.h
 *
 * Four-microphone linear-AEC front end with one shared far aligner and one
 * post-beam NR/RES path.
 *
 * The external owner supplies SRP-PHAT/GSC. This module deliberately owns:
 *
 *   1 shared DelayAec3 matcher
 *   4 independent linear AEC adaptive filters
 *   1 coherent post-beam residual suppressor
 *   1 mono MMSE-LSA denoiser
 *   1 final inverse FFT / overlap-add path
 *
 * It does not select one microphone's RES gain and does not replicate NR/RES
 * per channel.
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
 * The only API-shape difference from the mono pipeline is that its one
 * process() call becomes process_pre() -> external beamformer ->
 * process_post(). Both construction paths use that same core and must be
 * byte-identical for the same inputs, weights, config, and backend.
 *
 * See pipelines/README.md ("4-ch integration") for the module table and
 * pipelines/4ch_aec_bf_nr_res/README.md for the full architecture writeup, the
 * pre/post hand-off contract with an external SRP-PHAT/GSC, and the
 * parity-limitation note on bounded vs unbounded R2 in the post-beam RES.
 */
#ifndef FOUR_AEC_NR_RES_H
#define FOUR_AEC_NR_RES_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"
#include "mmse_lsa_denoiser.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FOUR_AEC_NR_RES_CHANNELS 4

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

#define FOUR_AEC_NR_RES_DESCRIPTOR_VERSION 1u
/* Current layout: three hop-sized working buffers. Per-lane RES spectra are
 * borrowed from each AEC instance and are valid only until that lane is
 * processed or reset again. Query a fresh descriptor for every build; do not
 * persist or synthesize one from this version number.
 *
 * 2026-08-13 PreFrame ABI note: FourAecNrResPreFrame gained a trailing
 * `aligned_ref` field (see its doc comment). That struct is a caller-stack
 * hand-off, not part of the pool carve, so this layout_version is
 * deliberately NOT bumped -- the pool byte layout is unchanged. It is still
 * a recompile-the-world struct change for out-of-tree callers embedding
 * FourAecNrResPreFrame by value.
 *
 * 7 -> 9 is +2 because this revision carries TWO independent carve changes,
 * each of which alone would have required a bump (see the token string in
 * four_aec_nr_res_build_flags_hash()):
 *   8: the post-beam stage became conditional on the new cfg.enable_post --
 *      the NR/iFFT/hop1/complex4/float6/fftfloat3/postsg regions are no
 *      longer carved at all for a pre-only instance, which the token now
 *      spells as post?(...).
 *   9: the delay estimator and the shared reference delay ring became
 *      conditional on the new cfg.delay_mode -- carved for MATCHED only,
 *      spelled delayest?,delayring? (both were unconditional before).
 *  10: cfg gained delay_backward_quarantine_enabled + _s, and the control
 *      block gained the quarantine countdown pair. No new REGION and no
 *      change to the carve ORDER -- but FourAecNrResConfig is embedded in
 *      the control block, so the pool total moves (query
 *      four_aec_nr_res_get_mem_requirements() for the current figure rather
 *      than trusting a number restated here, which rots on the next bump). A
 *      descriptor captured from an older build therefore no longer sizes
 *      this one, which is exactly what this counter exists to say out loud;
 *      contrast the 2026-08-13 PreFrame note above, which did NOT bump
 *      because that struct lives on the caller's stack.
 *  11: the control block gained the shared-delay change admission (held
 *      candidate + its remaining life) and the two realign lane counters.
 *      Again no new REGION and no carve-order change, and again the control
 *      block is the first thing carved out of the pool, so the pool total
 *      moves and every offset after it with it. build_flags_hash is computed
 *      over the carve TOKEN and so does not move on a control-block-only
 *      change: this counter is the whole signal that a descriptor from a
 *      version-10 build must be refused, including when its cached byte
 *      count still covers the current pool. */
#define FOUR_AEC_NR_RES_LAYOUT_VERSION 11u
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
 * over AecConfig/MmseLsaConfig tunable values; reserved = always 0, VALIDATED
 * not assumed, because `expected` may originate from persisted/transmitted
 * bytes) -- see that header's own doc comment for the full rationale
 * (four problems the old size_t/pointer/strcmp shape had that this fixed
 * layout fixes) rather than repeating it here.
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
     * aec.h for the full derivation, the measured pre-echo scene, and the
     * unbounded-veto defect this replaced), applied to the SHARED estimate:
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
    float max_delay_ms;           /* MATCHED reference delay-line capacity   */
    AecPreset aec_preset;
    MmseLsaNrMode nr_mode;
    int enable_post;              /* direct core: 1=RES/NR/iFFT, 0=pre-only;
                                    * complete wrappers require caller value 1 */
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
 * Opaque ownership/ordering token returned by process_pre().
 * Callers must copy it without modification and return it to process_post().
 *
 * instance_epoch guards a caller-owned-pool ABA case that frame_index/
 * generation/owner_cookie alone cannot: destroy() never releases caller
 * memory, so a caller may init_ex() a brand-new instance into the exact same
 * pool bytes. init_ex() memsets that pool to zero, so the new instance's
 * frame_index/generation start at 0 again and owner_cookie (the instance
 * pointer) is identical to the destroyed instance's -- a token captured
 * before destroy() would otherwise be bit-identical to a token the new
 * instance mints for its own first frame. instance_epoch is stamped from a
 * process-wide monotonic counter at construction time (never read back from
 * the -- possibly reused -- pool bytes themselves), so it differs across any
 * two constructions even when every other field coincides.
 */
typedef struct FourAecNrResFrameToken {
    uint64_t frame_index;
    uint64_t generation;
    uintptr_t owner_cookie;
    uint64_t instance_epoch;
} FourAecNrResFrameToken;

typedef struct FourAecNrResPreFrame {
    FourAecNrResFrameToken token;
    FourAecNrResDelayState delay;
    int hop_size;
    int n_channels;               /* always FOUR_AEC_NR_RES_CHANNELS        */
    int n_freqs;

    /**
     * Read-only interleaved formed linear-AEC output [hop_size][4]:
     * linear_interleaved[sample * 4 + channel].
     * Each channel is the exact selected/crossfaded time-domain hop
     * underlying linear_spectra[channel], before the downstream WOLA
     * synthesis. An external time-domain beamformer that performs its own
     * sqrt-Hann analysis must consume this signal rather than the standalone
     * AEC's separately limited return value.
     *
     * This is a pipeline-owned scratch buffer (each lane's formed_hop is
     * copied into it, not aliased) -- unlike linear_spectra below, it is not
     * a view into any lane's own memory. Still only safe to read for one
     * frame: valid until process_post(), reset(), or destroy(), same as the
     * rest of this struct. Only one pre frame may be in flight.
     */
    const float* linear_interleaved;

    /**
     * Read-only channel pointers to the exact linear-AEC error spectra used
     * later by process_post(). Shape: linear_spectra[channel][n_freqs].
     *
     * This lets an external SRP-PHAT/GSC consume the existing analysis
     * transform instead of performing a duplicate STFT. Unlike
     * linear_interleaved above, these pointers alias each lane's own live
     * per-hop AEC buffer rather than a pipeline-owned copy -- reset() does
     * NOT guarantee the underlying memory is cleared or overwritten (a
     * lane's old spectrum may simply be left in place), so there is no
     * memory-corruption tripwire to rely on. What actually makes these safe
     * to use is the pending/token gate (this file rejects a stale token, so
     * process_post() itself can never read them after an intervening
     * reset()) plus the API lifetime contract every caller must honor:
     * valid until process_post(), reset(), or destroy(), identical to
     * linear_interleaved above -- a caller that keeps reading these past
     * reset()/destroy() is relying on undefined content, not "the old
     * hop's data, still correct by coincidence".
     */
    const Complex* linear_spectra[FOUR_AEC_NR_RES_CHANNELS];

    /**
     * Read-only view of the delay-aligned time-domain far reference actually
     * fed to every AEC lane this hop: aligned_ref[hop_size]. Content is the
     * caller's ref stream shifted by delay.delay_samples on every hop the
     * shared ring can serve that offset, and the caller's RAW ref hop on the
     * hops before it can (lib/aec's rule: while the seam is not aligned, the
     * content is the raw far). The switch is whole-hop and coincides with
     * `delay.solid` under FIXED, so this buffer never splices raw and shifted
     * audio inside one hop.
     *
     * Like linear_interleaved above, this is a pipeline-owned buffer (the
     * shared delay line's per-hop output, not a lane alias) and follows the
     * exact same lifetime contract: only safe to read for one frame -- valid
     * until process_post(), abandon_pre(), reset(), or destroy(); only one
     * pre frame may be in flight. Intended consumer: an external neural
     * post-filter that needs the SAME aligned far the linear filters
     * consumed (e.g. the Align-ULCNet post stage's far branch).
     */
    const float* aligned_ref;
} FourAecNrResPreFrame;

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
 * (no diagnostic is printed today -- unlike the mono pipeline's
 * AP_LOG_ERR()-per-field messages, this module currently returns NULL
 * silently on every rejection path, mono or four-channel; it links no stdio
 * symbols as a result. Naming the failed field on stderr, gated the same
 * NO_STDIO-safe way audio_pipeline.c does, is a reasonable follow-up but is
 * not implemented here yet).
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
 * Run the shared aligner and four linear
 * AEC filters.
 *
 * microphones_interleaved is [hop][4], ref is [hop]. Both are read only.
 * A second pre() before the matching post() returns SEQUENCE_ERROR.
 *
 * @return FOUR_AEC_NR_RES_OK on success (*out filled, token valid until the
 *         matching process_post()/reset()/destroy()); FOUR_AEC_NR_RES_INVALID_ARGUMENT
 *         on a NULL p/microphones_interleaved/ref/out; FOUR_AEC_NR_RES_SEQUENCE_ERROR
 *         if a pre frame is already pending.
 */
int four_aec_nr_res_process_pre(
    FourAecNrRes* p,
    const float* microphones_interleaved,
    const float* ref,
    FourAecNrResPreFrame* out);

/**
 * Resume one pre frame after the external SRP-PHAT/GSC has updated its
 * effective frequency-domain weights.
 *
 * weights is channel-major Complex[4][n_freqs] and uses:
 *
 *   output[bin] = sum(weights[channel,bin] * input[channel,bin])
 *
 * without conjugation. The module coherently projects the four AEC spectra,
 * calculates one post-beam RES gain, fuses it with one NR gain, and writes
 * one final mono hop to out.
 *
 * The external beamformer does not need to return a duplicate mono hop:
 * synthesis is performed from the coherently weighted spectra here. For a
 * multi-frame/time-domain GSC, weights must be its exact effective response
 * for this frame.
 *
 * @return FOUR_AEC_NR_RES_OK on success (*out fully written, token
 *         consumed); FOUR_AEC_NR_RES_INVALID_ARGUMENT on a NULL
 *         p/token/weights/out or all-zero weights (the pending frame is
 *         left intact so the caller may correct and retry -- see
 *         "process_pre"/"process_post" ordering above); FOUR_AEC_NR_RES_SEQUENCE_ERROR
 *         if token does not match the pending frame (replay, cross-instance
 *         use, or a token invalidated by an intervening reset()).
 */
int four_aec_nr_res_process_post(
    FourAecNrRes* p,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    float* out);

/**
 * Release one pending pre frame WITHOUT running this module's post-beam
 * RES/NR/synthesis path. Companion to process_pre() for pipeline variants
 * whose post stage is external (e.g. the Align-ULCNet neural post-filter
 * wrapper, which consumes linear_spectra/aligned_ref and replaces
 * process_post() entirely).
 *
 * Consumes the token exactly like a successful process_post() would (the
 * next process_pre() is legal again; all PreFrame pointers become invalid),
 * but advances NO downstream state: the mono NR, the post-beam RES, and this
 * module's own synthesis OLA are untouched. A stream that interleaves
 * abandoned and posted frames therefore resumes process_post() with a
 * synthesis OLA that is missing the abandoned hops -- this call is intended
 * for pipelines that never call any process_post() variant at all, not for
 * per-hop mixing.
 *
 * @return FOUR_AEC_NR_RES_OK on success; FOUR_AEC_NR_RES_INVALID_ARGUMENT on
 *         a NULL p/token; FOUR_AEC_NR_RES_SEQUENCE_ERROR if token does not
 *         match the pending frame (no frame pending, replay, cross-instance
 *         use, or a token invalidated by an intervening reset()).
 */
int four_aec_nr_res_abandon_pre(
    FourAecNrRes* p,
    const FourAecNrResFrameToken* token);

/**
 * Reset all delay/AEC/NR/RES/OLA state. Invalidates a pending pre-frame token.
 *
 * A caller holding an outstanding FourAecNrResPreFrame from a still-pending
 * process_pre() must stop reading its linear_interleaved/linear_spectra
 * pointers the instant this is called, not just once a later process_post()
 * call would reject the now-stale token. This is an API-contract rule, not a
 * memory-safety guarantee enforced by this call: linear_interleaved is a
 * pipeline-owned copy that IS overwritten by the explicit memset below (not
 * by aec_reset()'s own per-lane resets, which touch each Aec's own internal
 * state, not this wrapper's buffer), but linear_spectra[channel]
 * aliases a lane's own AecResContext buffer, and aec_reset() does NOT
 * guarantee that buffer's old content is cleared or overwritten -- a lane's
 * stale spectrum may simply be left in place at the same address. Do not
 * rely on reading garbage/zeroed data as a way to detect a missed
 * invalidation; the pending/token gate is what keeps this file's own
 * process_post() from ever reading post-reset lane memory, and external
 * callers must independently honor the same rule.
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
 * is allocated here; process_pre/post perform no allocation. Returns NULL for
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
 * in FIXED/EXTERNAL; the other values are 4 / enable_post / enable_post. */
int four_aec_nr_res_matched_filter_count(const FourAecNrRes* p);
int four_aec_nr_res_linear_aec_count(const FourAecNrRes* p);
int four_aec_nr_res_nr_count(const FourAecNrRes* p);
int four_aec_nr_res_post_res_count(const FourAecNrRes* p);

/* Group 6 instrumentation, read-only: sum of aec_far_fft_real_compute_count()
 * across all four lanes (cumulative since construction or the last
 * four_aec_nr_res_reset()). Lane 0 runs its own far-end FFT every hop
 * (aec_process_context()); lanes 1-3 borrow lane 0's spectrum instead of
 * recomputing it (aec_process_context_shared_far()) -- so this total
 * increments by exactly 1 per four_aec_nr_res_process_pre() call, not 4.
 * Intended for tests/instrumentation proving the cross-lane far-FFT
 * sharing is genuinely active; has no effect on any processing. */
long four_aec_nr_res_far_fft_real_compute_count(const FourAecNrRes* p);

/* Realign instrumentation, read-only and cumulative since construction or the
 * last four_aec_nr_res_reset(): how the four lanes'
 * aec_apply_external_realign() calls landed.
 * Every hop that publishes `changed` sweeps all four lanes, so
 * the pair grows by exactly 4 on such a hop and not at all otherwise -- warm
 * (returned 1) means the learned IR was shifted and cancellation survived the
 * move, soft (returned 0) means that lane re-adapts from its current taps. A
 * sweep that adds fewer than 4 means a lane rejected the call outright, which
 * is a wiring fault, not a soft realign. Nothing branches on either count. */
long four_aec_nr_res_realign_warm_lane_count(const FourAecNrRes* p);
long four_aec_nr_res_realign_soft_lane_count(const FourAecNrRes* p);

/* The shared-delay movement currently held for confirmation, in native
 * samples, or -1 when none is held (including FIXED/EXTERNAL_ALIGNED, which
 * never re-decide an alignment). See FourAecNrResDelayState's `changed` for
 * the rule; diagnostic only. */
int four_aec_nr_res_pending_delay_candidate(const FourAecNrRes* p);

/* ============================================================================
 * Diagnostic memory breakdown
 * ========================================================================== */

typedef struct FourAecNrResMemBreakdown {
    size_t aec_bytes;       /* four AEC pools combined */
    size_t nr_bytes;
    size_t fft_bytes;
    size_t wrapper_bytes;   /* control block + shared/post-beam state */
    size_t total_bytes;
    int hop_size;
    int fft_size;
    int n_freqs;
} FourAecNrResMemBreakdown;

/** Optional module/wrapper split for board memory-budget reporting. */
int four_aec_nr_res_get_mem_breakdown(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemBreakdown* out);

#ifdef __cplusplus
}
#endif

#endif /* FOUR_AEC_NR_RES_H */
