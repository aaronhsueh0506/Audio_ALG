/**
 * 4ch_pipelines/4aec_nr_res.h
 *
 * Four-microphone linear-AEC front end with one shared delay matcher and one
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
 * pipelines/audio_pipeline.h:
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
 * pipelines/4ch_pipelines/README.md for the full architecture writeup, the
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
 * persist or synthesize one from this version number. */
#define FOUR_AEC_NR_RES_LAYOUT_VERSION 7u
#define FOUR_AEC_NR_RES_BACKEND_KISS 1u
#define FOUR_AEC_NR_RES_BACKEND_NE10 2u

/**
 * Fixed-width descriptor for a caller-owned static-memory pool.
 *
 * Query it at init time and pass the same value to init_ex(). Do not cache it
 * across library, backend, compiler-flag, or config changes.
 *
 * Same 32-byte fixed-width-integer shape and same field meanings as
 * pipelines/audio_pipeline.h's `AudioPipelineMemReq` (descriptor_version =
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
    int capture_proxy_channel;    /* shared matcher input, [0,3]            */
    float max_delay_ms;           /* shared reference delay-line capacity    */
    AecPreset aec_preset;
    MmseLsaNrMode nr_mode;
    int enable_cng;               /* bool                                   */
    int legacy_amin;              /* bool: do not fold R2 into NR prior      */
} FourAecNrResConfig;

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
 * carved (mirrors pipelines/audio_pipeline.h's audio_pipeline_init_ex()):
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
 * Run the shared matcher, one shared reference delay line, and four linear
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

/* Structural audit hooks. Values are 1 / 4 / 1 / 1 for a valid handle. */
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
