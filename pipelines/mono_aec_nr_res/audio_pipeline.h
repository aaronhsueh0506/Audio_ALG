/**
 * Linkable AEC(linear) -> echo-aware NR -> RES pipeline.
 *
 * Pool-first construction is the firmware path and performs no allocation:
 *
 *     AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
 *     AudioPipelineMemReq req;
 *     audio_pipeline_get_mem_requirements(&cfg, &req);
 *     void* pool = platform_alloc(req.bytes, req.alignment);
 *     AudioPipeline* p = audio_pipeline_init_ex(pool, req.bytes, &cfg, &req);
 *     ...
 *     audio_pipeline_destroy(p);  // does not release caller-owned memory
 *     platform_free(pool);
 *
 * Query the descriptor again after any library, backend, build-option or
 * config change. audio_pipeline_create()/destroy() provide a heap convenience
 * path for desktop tools. See pipelines/README.md for lifecycle and streaming
 * examples.
 */
#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"                /* AecConfig/AecPreset, Aec (non-opaque)      */
#include "mmse_lsa_denoiser.h"  /* MmseLsaConfig/MmseLsaNrMode, MmseLsaDenoiser (opaque) */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

/** AudioPipelineMemReq ABI version; independent of the pool layout version. */
#define AUDIO_PIPELINE_DESCRIPTOR_VERSION 2u

/** Stable FFT backend identifiers. Zero is reserved and never returned. */
#define AUDIO_PIPELINE_BACKEND_KISS 1u
#define AUDIO_PIPELINE_BACKEND_NE10 2u

/**
 * Fixed 32-byte same-endian descriptor. Callers may persist it for
 * diagnostics, but must re-query it before initialization after any build or
 * configuration change. layout_version and build_flags_hash describe this
 * wrapper's carve structure; dependency-internal layouts are reflected in
 * bytes. reserved must be zero. No cross-endian serialization is provided.
 */
typedef struct {
    uint32_t descriptor_version;  /* = AUDIO_PIPELINE_DESCRIPTOR_VERSION (2) */
    uint32_t layout_version;      /* carve-layout version (unchanged meaning) */
    uint32_t backend_id;          /* AUDIO_PIPELINE_BACKEND_KISS=1 / _NE10=2 */
    uint32_t build_flags_hash;    /* FNV-1a-32, unchanged meaning */
    uint32_t alignment;           /* 16 */
    uint32_t reserved;            /* 0; keeps 8-byte alignment for bytes */
    uint64_t bytes;               /* total pool size */
} AudioPipelineMemReq;

/* Pin the same-endian serialized ABI field by field. */
_Static_assert(sizeof(AudioPipelineMemReq) == 32,
               "AudioPipelineMemReq must be exactly 32 bytes (fixed-width serializable ABI)");
_Static_assert(offsetof(AudioPipelineMemReq, descriptor_version) == 0,
               "AudioPipelineMemReq.descriptor_version must be at offset 0");
_Static_assert(offsetof(AudioPipelineMemReq, layout_version) == 4,
               "AudioPipelineMemReq.layout_version must be at offset 4");
_Static_assert(offsetof(AudioPipelineMemReq, backend_id) == 8,
               "AudioPipelineMemReq.backend_id must be at offset 8");
_Static_assert(offsetof(AudioPipelineMemReq, build_flags_hash) == 12,
               "AudioPipelineMemReq.build_flags_hash must be at offset 12");
_Static_assert(offsetof(AudioPipelineMemReq, alignment) == 16,
               "AudioPipelineMemReq.alignment must be at offset 16");
_Static_assert(offsetof(AudioPipelineMemReq, reserved) == 20,
               "AudioPipelineMemReq.reserved must be at offset 20");
_Static_assert(offsetof(AudioPipelineMemReq, bytes) == 24,
               "AudioPipelineMemReq.bytes must be at offset 24");

/* ============================================================================
 * Config
 * ========================================================================== */

/**
 * Initialization contract shared by host tools and board integrations.
 * WAV paths, diagnostics and other tool-only concerns stay outside this API.
 */
typedef struct {
    int           sample_rate;   /* 8000 | 16000 | 48000                              */
    int           fft_size;      /* 0=rate default; 256/512 @16k, 1024 @48k           */
    int           filter_length; /* 0=rate default; PBFDKF taps, init-time immutable   */
    AecDelayMode  delay_mode;    /* MATCHED | FIXED | EXTERNAL_ALIGNED                 */
    int           delay_num_filters;   /* MATCHED bank size [1,5]                       */
    int           fixed_delay_samples; /* FIXED native-rate samples; -1 otherwise       */
    AecPreset     aec_preset;    /* MILD | BALANCED | AGGRESSIVE                       */
    MmseLsaNrMode nr_mode;       /* MILD | MODERATE | BALANCED | AGGRESSIVE            */
    int           aec_only;      /* 1 = skip NR/RES entirely (linear AEC output only)  */
    int           enable_cng;    /* 1 = fill AEC-suppressed bins with comfort noise    */
    int           legacy_amin;   /* 1 = prior min-only A_min_pl (--legacy-amin): NR    *
                                   * gain computed WITHOUT folding R² into the noise    *
                                   * floor, and the far/near-gated near-end floor       *
                                   * strength collapses to the fixed scalar 0.4         */
} AudioPipelineConfig;

/** Defaults: rate-default no-padding grid, balanced modes, full pipeline,
 * MATCHED delay with the five-filter reference bank and default AEC length. */
AudioPipelineConfig audio_pipeline_default_config(int sample_rate);

/* ============================================================================
 * Opaque handle
 * ========================================================================== */

typedef struct AudioPipeline AudioPipeline;

/* ============================================================================
 * Pool-first lifecycle
 * ========================================================================== */

/**
 * Query the memory descriptor for `cfg` WITHOUT allocating or touching any
 * audio state. Validates `cfg` via the same module validators
 * aec_get_mem_size()/mmse_lsa_get_mem_size() already gate on internally
 * (aec_validate_config / mmse_lsa_validate_config, both invalid-config ->
 * return 0) PLUS an explicit reject-first check up front, in
 * derive_dims_and_configs() (the one place every entry point in this file
 * funnels through): sample_rate against the {8000,16000,48000} whitelist
 * (aec_is_valid_sample_rate — e.g. sample_rate=44100 is rejected before any
 * size arithmetic runs, not just left to a downstream 0), aec_preset/nr_mode
 * against their defined enum values (rather than silently falling through
 * aec_config_from_preset's/mmse_lsa_config_for_mode's own balanced-default
 * fallback), and aec_only/enable_cng/legacy_amin against {0,1} (rather than
 * being treated as truthy by a stray nonzero value downstream).
 *
 * @return 0 on success (*out filled), -1 on NULL args or invalid cfg.
 */
int audio_pipeline_get_mem_requirements(const AudioPipelineConfig* cfg,
                                         AudioPipelineMemReq* out);

/**
 * Initialize AEC + OLA + NR and all scratch storage in a caller-owned pool.
 * `mem` must be 16-byte aligned and at least the size returned by
 * audio_pipeline_get_mem_requirements(). It need not be zero-filled, but it
 * must remain stable and exclusive until the pipeline is no longer used.
 * Equivalent to audio_pipeline_init_ex(mem, bytes, cfg, NULL).
 *
 * @return a valid handle, or NULL on invalid config, pool, or submodule init.
 */
AudioPipeline* audio_pipeline_init(void* mem, size_t bytes,
                                    const AudioPipelineConfig* cfg);

/**
 * Initialize from a caller-owned pool and optionally reject a stale memory
 * descriptor. When `expected` is non-NULL, its descriptor/layout version,
 * backend, build-flags hash, alignment, reserved field, and byte capacity
 * must match the requirements recomputed for this build and config. The
 * supplied pool must independently satisfy the current byte requirement.
 * Pass NULL to obtain audio_pipeline_init() behavior.
 *
 * @return a valid handle, or NULL on descriptor or initialization failure.
 */
AudioPipeline* audio_pipeline_init_ex(void* mem, size_t bytes,
                                       const AudioPipelineConfig* cfg,
                                       const AudioPipelineMemReq* expected);

/**
 * Process exactly one hop (audio_pipeline_hop_size(p) samples) of mic/ref
 * into `out`. Verbatim port of the static CLI's per-hop while-loop body:
 * AEC(linear) -> echo-aware NR gain -> g_total=min(g_nr,g_res) -> far/near
 * gated near-end floor lift -> S(f)=E(f)*g_total (+ CNG on the cut bins) ->
 * irfft -> sqrt-Hann OLA. `aec_only` short-circuits to the raw linear AEC
 * residual (mirrors the CLI's `--aec-only`).
 *
 * `mic`/`ref` are read-only and only for the duration of this call (they are
 * copied into pool-owned scratch before use — see audio_pipeline.c); `out`
 * is fully overwritten (never read). All three must be exactly
 * audio_pipeline_hop_size(p) floats.
 *
 * @return 0 on success, -1 if p/mic/ref/out is NULL.
 */
int audio_pipeline_process(AudioPipeline* p, const float* mic,
                            const float* ref, float* out);

/**
 * Re-zero all pipeline/AEC/NR state (OLA accumulator, comfort-noise RNG,
 * near-end-floor hangover counter, and each sub-module's own reset) without
 * touching the pool itself or re-validating cfg — equivalent to a fresh
 * audio_pipeline_init() on the SAME pool/cfg, but without the alignment/size
 * re-checks. Use after an echo-path change (speaker swap, AEC re-seat) or
 * between unrelated streams sharing one instance.
 */
void audio_pipeline_reset(AudioPipeline* p);

/**
 * Tear down in reverse carve order: NR -> pipeline FFT (the OLA irfft
 * instance) -> AEC. This is the mirror image of audio_pipeline_init's carve
 * order (AEC -> FFT -> NR -> scratch) and matches the teardown order
 * pipelines/README.md's "Two Versions" section already documents for the
 * static CLI (`mmse_lsa_destroy` / `fft_destroy` / `aec_destroy`, in that
 * order) — kept even though every one of those three calls is a genuine
 * no-op on a pool-resident (audio_pipeline_init'd) instance today: it is
 * forward-compat insurance (a future backend/module MAY hold something
 * outside the pool that a destroy call needs to release — see the NE10
 * twiddle-config caveat in aec.h/fft_wrapper.h) and is exactly what the
 * heap convenience path (audio_pipeline_create) needs for real.
 *
 * NULL-safe (destroy(NULL) is a no-op) and idempotent FOR A POOL-RESIDENT
 * INSTANCE — repeated calls are safe because each of the three underlying
 * destroy calls already promises that. For a HEAP instance (obtained via
 * audio_pipeline_create()), the SAME single free() this call performs on
 * the pool follows ordinary free() semantics: call exactly once. A second
 * call on a heap instance is a double-free, exactly as a second
 * free()/fft_destroy() on an already-freed heap handle would be — this
 * function cannot detect that case (the instance it would check is the
 * memory being freed).
 */
void audio_pipeline_destroy(AudioPipeline* p);

/* ============================================================================
 * Heap convenience (desktop CLIs / quick prototyping — NOT the board path)
 * ========================================================================== */

/**
 * audio_pipeline_get_mem_requirements() + posix_memalign(16, ...) +
 * audio_pipeline_init(), all in one call. The returned handle owns its pool;
 * audio_pipeline_destroy() frees it.
 *
 * @return a valid handle, or NULL (invalid cfg or allocation failure).
 */
AudioPipeline* audio_pipeline_create(const AudioPipelineConfig* cfg);

/* ============================================================================
 * Accessors
 * ========================================================================== */

int audio_pipeline_hop_size(const AudioPipeline* p);     /* -1 if p is NULL */
int audio_pipeline_n_freqs(const AudioPipeline* p);      /* -1 if p is NULL */
int audio_pipeline_sample_rate(const AudioPipeline* p);  /* -1 if p is NULL */

/**
 * Read-only access to the underlying module handles, for a caller's OWN
 * diagnostics (aec_debug_status/aec_get_res_context/mmse_lsa_debug_status/
 * mmse_lsa_get_gain/...) — this is how both CLIs keep their existing
 * `--debug` status line and `DUMP_CTX` per-hop dump working as thin
 * wrappers, without this header re-exposing every intermediate signal
 * itself. Do not call any _reset/_destroy/mutating entry point on these
 * directly — go through audio_pipeline_reset()/audio_pipeline_destroy() so
 * pipeline-owned state (OLA, RNG, hangover counter) stays in sync.
 */
Aec*             audio_pipeline_get_aec(const AudioPipeline* p);  /* never NULL for a valid p */
MmseLsaDenoiser* audio_pipeline_get_nr(const AudioPipeline* p);   /* NULL iff cfg.aec_only     */

/* ============================================================================
 * Diagnostic breakdown (backs --print-mem-size in both CLIs)
 * ========================================================================== */

/** Per-module byte breakdown, mirroring the static CLI's original
 * `print_mem_budget()` table (AEC / FFT / NR / pipeline-buffer columns) so
 * that diagnostic stays available without either CLI re-deriving
 * AecConfig/MmseLsaConfig/frame-dims itself. NOT part of
 * AudioPipelineMemReq: those bytes are already folded into `bytes` there;
 * this is purely for the human-readable table. */
typedef struct {
    size_t aec_bytes;
    size_t fft_bytes;         /* 0 when cfg.aec_only */
    size_t nr_bytes;          /* 0 when cfg.aec_only */
    size_t pipeline_bytes;    /* the 7 scratch buffers (0 when aec_only) */
    int    hop, frame_sz, fft_sz, n_freqs;
} AudioPipelineMemBreakdown;

/**
 * @return 0 on success (*out filled), -1 on NULL args or invalid cfg (same
 *         validation as audio_pipeline_get_mem_requirements()).
 */
int audio_pipeline_get_mem_breakdown(const AudioPipelineConfig* cfg,
                                      AudioPipelineMemBreakdown* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_H */
