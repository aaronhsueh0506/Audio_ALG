/**
 * aec_4ch/4aec_nr_res.h
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
 * The lifecycle intentionally mirrors pipelines/audio_pipeline.h:
 *
 *   Caller-owned pool (board/static path):
 *
 *     FourAecNrResConfig cfg;
 *     FourAecNrResMemReq req;
 *     four_aec_nr_res_config_defaults(&cfg, 16000);
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
 * Both construction paths use the same process_pre()/process_post() core and
 * must be byte-identical for the same inputs, weights, config, and backend.
 *
 * See pipelines/README.md ("4-ch integration") for the module table and
 * pipelines/aec_4ch/README.md for the full architecture writeup, the
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
#define FOUR_AEC_NR_RES_LAYOUT_VERSION 1u
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
    int fft_size;                 /* 0=rate default; 512 @16k, 1024 @48k   */
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
    int changed;
    uint64_t estimator_calls;
    int estimator_updates;
} FourAecNrResDelayState;

/**
 * Opaque ownership/ordering token returned by process_pre().
 * Callers must copy it without modification and return it to process_post().
 */
typedef struct FourAecNrResFrameToken {
    uint64_t frame_index;
    uint64_t generation;
    uintptr_t owner_cookie;
} FourAecNrResFrameToken;

typedef struct FourAecNrResPreFrame {
    FourAecNrResFrameToken token;
    FourAecNrResDelayState delay;
    int hop_size;
    int n_channels;               /* always FOUR_AEC_NR_RES_CHANNELS        */

    /**
     * Read-only interleaved linear-AEC output [hop_size][4]:
     * linear_interleaved[sample * 4 + channel].
     *
     * Valid until process_post(), reset(), or destroy(). Only one pre frame
     * may be in flight.
     */
    const float* linear_interleaved;
} FourAecNrResPreFrame;

typedef struct FourAecNrRes FourAecNrRes;

/* ============================================================================
 * Config and lifecycle
 * ========================================================================== */

/** Fill a validated default: 16 kHz, 512/256, balanced AEC/NR, CNG on. */
void four_aec_nr_res_config_defaults(FourAecNrResConfig* cfg,
                                     int sample_rate);

/**
 * Query the complete 16-byte-aligned caller pool required by cfg.
 * Performs no allocation and returns 0 on success.
 */
int four_aec_nr_res_get_mem_requirements(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemReq* out);

/** Optional module/wrapper split for board memory-budget reporting. */
int four_aec_nr_res_get_mem_breakdown(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemBreakdown* out);

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

/**
 * Heap convenience wrapper over get_mem_requirements() + init(). All memory
 * is allocated here; process_pre/post perform no allocation. Returns NULL for
 * an invalid config or allocation failure.
 */
FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg);

/**
 * Reset all delay/AEC/NR/RES/OLA state. Invalidates a pending pre-frame token.
 */
void four_aec_nr_res_reset(FourAecNrRes* self);

/**
 * Destroy an instance. For caller-owned memory this is idempotent and never
 * frees the pool; the caller releases/reuses it afterward. For a heap-created
 * handle it frees the one owned pool, so call once as with free().
 */
void four_aec_nr_res_destroy(FourAecNrRes* self);

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
 *         on a NULL self/microphones_interleaved/ref/out; FOUR_AEC_NR_RES_SEQUENCE_ERROR
 *         if a pre frame is already pending.
 */
int four_aec_nr_res_process_pre(
    FourAecNrRes* self,
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
 *         self/token/weights/out or all-zero weights (the pending frame is
 *         left intact so the caller may correct and retry -- see
 *         "process_pre"/"process_post" ordering above); FOUR_AEC_NR_RES_SEQUENCE_ERROR
 *         if token does not match the pending frame (replay, cross-instance
 *         use, or a token invalidated by an intervening reset()).
 */
int four_aec_nr_res_process_post(
    FourAecNrRes* self,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    float* out);

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int four_aec_nr_res_hop_size(const FourAecNrRes* self);
int four_aec_nr_res_fft_size(const FourAecNrRes* self);
int four_aec_nr_res_n_freqs(const FourAecNrRes* self);
int four_aec_nr_res_sample_rate(const FourAecNrRes* self);

/* Structural audit hooks. Values are 1 / 4 / 1 / 1 for a valid handle. */
int four_aec_nr_res_matched_filter_count(const FourAecNrRes* self);
int four_aec_nr_res_linear_aec_count(const FourAecNrRes* self);
int four_aec_nr_res_nr_count(const FourAecNrRes* self);
int four_aec_nr_res_post_res_count(const FourAecNrRes* self);

#ifdef __cplusplus
}
#endif

#endif /* FOUR_AEC_NR_RES_H */
