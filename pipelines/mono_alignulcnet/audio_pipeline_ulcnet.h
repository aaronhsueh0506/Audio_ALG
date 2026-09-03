/**
 * Linkable AEC(linear) -> Align-ULCNet neural post-filter pipeline (mono).
 *
 * Same pool-first construction discipline as audio_pipeline.h (the
 * traditional AEC->NR->RES wrapper this variant sits beside), but the post
 * stage is the Align-ULCNet NN driven through the per-frame callback
 * boundary defined in AIAEC/Align_ULCNet/ulcnet_process.h (UlcnetModel).
 * The NN itself runs on an external runtime; this pipeline owns only the
 * verified C pre/post (rolling center=False sqrt-Hann STFT, exactly one
 * frame and at most one inference per hop, WOLA) plus the linear AEC and the delay-state policy around the model.
 *
 *     AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(16000);
 *     cfg.model = my_runtime_model;             // may be left all-zero
 *     AudioPipelineUlcnetMemReq req;
 *     audio_pipeline_ulcnet_get_mem_requirements(&cfg, &req);
 *     void* pool = platform_alloc(req.bytes, req.alignment);
 *     AudioPipelineUlcnet* p = audio_pipeline_ulcnet_init_ex(pool, req.bytes, &cfg, &req);
 *     ...per hop: audio_pipeline_ulcnet_process(p, mic, ref, out);
 *     audio_pipeline_ulcnet_destroy(p);          // never frees the caller's pool
 *
 * Signal grid is compile-time fixed by the ULCNET_* constants. Supported
 * builds are 16 kHz / FFT 512 / hop 256 / 257 bins and 48 kHz / FFT 1024 /
 * hop 512 / 513 bins. Validation accepts only the grid compiled into this
 * binary; one binary never switches grids at run time.
 *
 * Latency: end-to-end algorithmic latency is exactly ONE compiled-grid hop.
 * hop #0 emits nothing (this pipeline writes zeros); the output of
 * hop #p (p >= 1) corresponds to input hop p-1.
 *
 * ── Far-input deployment contract ─────────────────────────────────────────
 *  The model always receives AecLinearContext.aligned_far_hop, the same far
 *  hop consumed by PBFDKF. Before acquisition the seam contains raw far;
 *  the model still runs and its D window handles the remaining offset.
 *  Raw/aligned selection is intentionally absent from this production API;
 *  it remains an offline sweep option only. A published model descriptor
 *  must carry ULCNET_FAR_ALIGNED.
 *
 * ── Model callback policy (first version) ────────────────────────────────
 *  - Fail-open identity: if the config's model has infer == NULL (including
 *    an all-zero/"NULL" model), or infer() returns nonzero for a frame, the
 *    error spectrum passes through unchanged for that frame. The STFT/WOLA
 *    timing path is identical either way, so latency never depends on the
 *    model's presence or health.
 *  - NaN/Inf guard: after a successful infer(), every output value is
 *    validated; a frame with any non-finite enh value falls back to the
 *    identity (error) frame -- non-finite data never reaches the WOLA.
 *  - Delay events (AecLinearContext.delay_state, read once per hop):
 *      UNLOCKED -> infer() still runs on the seam's raw far; D provides the
 *                  model-side residual delay search.
 *      CHANGED  -> model->reset (if set) is called BEFORE this hop's
 *                  frames, so the runtime flushes its far attention ring +
 *                  logit history; the identity reprime below then starts.
 *      LOCKED   -> infer() runs and its output is applied.
 *    FIXED mode has no CHANGED estimator event, so the wrapper detects its
 *    first UNLOCKED->LOCKED ring transition and performs the same reset.
 *  - audio_pipeline_ulcnet_reset() also calls model->reset (if set).
 *
 * ── Identity reprime across an alignment boundary (option A) ─────────────
 *  A generation change flushes the runtime's recurrent state, but the C
 *  STFT states keep running: the analysis windows already in flight still
 *  STRADDLE the boundary -- their two-hop spans cover one hop pushed
 *  before the switch and one pushed after, on the error branch, the far
 *  branch, or both. Stepping the model on such a frame would rebuild, from
 *  a half-stale error/far pair, exactly the recurrent state the reset just
 *  cleared.
 *
 *  Policy: starting with the boundary hop, the next
 *  AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES emitted frames take the identity
 *  (error passthrough) path and the model is NOT stepped -- no infer() call,
 *  so no K/V ring entry, logit-history entry or GRU hidden update happens on
 *  straddling input. Stepping AND applying resume together on the first
 *  frame whose error and far analysis windows contain exclusively
 *  post-switch hops. A second boundary inside a reprime re-arms the counter
 *  (it never accumulates).
 *
 *  Derivation of the constant (MEASURED by the straddle-derivation test in
 *  tests/test_audio_pipeline_ulcnet.c, never assumed here): this wrapper
 *  pushes both branches from the CURRENT hop -- the error tap
 *  (AecResContext.formed_hop) and the aligned-far tap are same-hop, no
 *  wrapper-side far compensation exists -- and the rolling 50%-overlap analysis
 *  frame at hop T spans the two pushed hops T-1 and T. A boundary at hop T therefore leaves
 *  exactly ONE emitted frame straddling (the frame emitted at hop T, whose
 *  window covers the pre-switch hop T-1 and the post-switch hop T); the
 *  frame emitted at hop T+1 covers hops T and T+1 and is already clean. The
 *  test derives that count from a marker run that contains NO boundary at
 *  all -- so the reprime logic never participates in its own measurement --
 *  and asserts it equals the constant below, per branch.
 *
 *  Compute: a reprime frame SKIPS inference, so per-hop compute DROPS for
 *  those frames; it never doubles. The STFT/WOLA path is untouched, so the
 *  one-hop latency contract holds unchanged across a boundary.
 *
 *  Option B (keep stepping the model through the straddling frames and keep
 *  applying its output) is DEFERRED pending an audio A/B: it trades this
 *  version's short identity stretch for recurrent state built on half-stale
 *  frames. Do not switch policies without that A/B.
 */
#ifndef AUDIO_PIPELINE_ULCNET_H
#define AUDIO_PIPELINE_ULCNET_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"             /* AecConfig/AecPreset, Aec (non-opaque) */
#include "ulcnet_process.h"  /* UlcnetModel + ULCNET_* grid constants */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Alignment-boundary identity reprime
 * ========================================================================== */

/**
 * Emitted frames that still straddle an alignment boundary in this wrapper,
 * i.e. the length of the identity reprime armed at every generation change
 * (see the "Identity reprime" section of this header's preamble).
 *
 * = 1: both branches are pushed from the CURRENT hop and the rolling
 * 50%-overlap analysis frame at hop T spans hops T-1 and T, so a boundary
 * at hop T leaves exactly the frame emitted at hop T straddling. Derived and asserted branch by
 * branch by the straddle-derivation test; do not edit this value without
 * re-running it.
 */
enum { AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES = 1 };

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

/** AudioPipelineUlcnetMemReq ABI version; independent of the pool layout
 * version below. This variant's descriptor history starts at 1. */
#define AUDIO_PIPELINE_ULCNET_DESCRIPTOR_VERSION 1u

/** Stable FFT backend identifiers (same values as audio_pipeline.h's, so a
 * board-side descriptor store can share the id space; redefined here so this
 * header stays self-contained). Zero is reserved and never returned. */
#define AUDIO_PIPELINE_ULCNET_BACKEND_KISS 1u
#define AUDIO_PIPELINE_ULCNET_BACKEND_NE10 2u

/**
 * Fixed 32-byte same-endian descriptor (same shape/discipline as
 * audio_pipeline.h's AudioPipelineMemReq). Callers may persist it for
 * diagnostics, but must re-query before initialization after any build or
 * configuration change. layout_version and build_flags_hash describe THIS
 * wrapper's carve structure (self control block first, then the AEC pool,
 * then the ULCNet chain's shared compiled-grid FFT handle; the Ulcnet
 * analysis/synthesis/frame-scratch state and the shared sqrt-Hann window
 * table live INSIDE the self block -- see the token string in
 * audio_pipeline_ulcnet.c); dependency-internal layouts are reflected in
 * bytes. reserved must be zero. No cross-endian serialization is provided.
 */
typedef struct {
    uint32_t descriptor_version;  /* = AUDIO_PIPELINE_ULCNET_DESCRIPTOR_VERSION (1) */
    uint32_t layout_version;      /* carve-layout version, starts at 1        */
    uint32_t backend_id;          /* _BACKEND_KISS=1 / _BACKEND_NE10=2        */
    uint32_t build_flags_hash;    /* FNV-1a-32 over backend + carve tokens    */
    uint32_t alignment;           /* 16                                       */
    uint32_t reserved;            /* 0; keeps 8-byte alignment for bytes      */
    uint64_t bytes;               /* total pool size                          */
} AudioPipelineUlcnetMemReq;

/* Pin the same-endian serialized ABI field by field. */
_Static_assert(sizeof(AudioPipelineUlcnetMemReq) == 32,
               "AudioPipelineUlcnetMemReq must be exactly 32 bytes (fixed-width serializable ABI)");
_Static_assert(offsetof(AudioPipelineUlcnetMemReq, descriptor_version) == 0,
               "AudioPipelineUlcnetMemReq.descriptor_version must be at offset 0");
_Static_assert(offsetof(AudioPipelineUlcnetMemReq, layout_version) == 4,
               "AudioPipelineUlcnetMemReq.layout_version must be at offset 4");
_Static_assert(offsetof(AudioPipelineUlcnetMemReq, backend_id) == 8,
               "AudioPipelineUlcnetMemReq.backend_id must be at offset 8");
_Static_assert(offsetof(AudioPipelineUlcnetMemReq, build_flags_hash) == 12,
               "AudioPipelineUlcnetMemReq.build_flags_hash must be at offset 12");
_Static_assert(offsetof(AudioPipelineUlcnetMemReq, alignment) == 16,
               "AudioPipelineUlcnetMemReq.alignment must be at offset 16");
_Static_assert(offsetof(AudioPipelineUlcnetMemReq, reserved) == 20,
               "AudioPipelineUlcnetMemReq.reserved must be at offset 20");
_Static_assert(offsetof(AudioPipelineUlcnetMemReq, bytes) == 24,
               "AudioPipelineUlcnetMemReq.bytes must be at offset 24");

/* ============================================================================
 * Config
 * ========================================================================== */

/**
 * model is held BY VALUE (four plain pointers). A memset-zero model (or one
 * with infer == NULL) is the supported "no runtime attached" case: the
 * pipeline output is then the identity STFT->WOLA reconstruction of the
 * linear error, still with the one-hop latency contract. The pointers inside
 * model must outlive the pipeline instance; the pipeline never copies what
 * they point at.
 */
typedef struct {
    int         sample_rate;  /* must equal ULCNET_SR                         */
    int         fft_size;     /* 0 (resolve) or ULCNET_N_FFT                  */
    int         filter_length; /* 0=rate default; PBFDKF taps                  */
    AecDelayMode delay_mode;   /* AEC far alignment policy                     */
    int         delay_num_filters;   /* MATCHED bank size [1,5]                 */
    int         fixed_delay_samples; /* FIXED samples; -1 for other modes       */
    AecPreset   aec_preset;   /* MILD | BALANCED | AGGRESSIVE                 */
    UlcnetModel model;        /* NPU callback boundary; may be all-zero       */
} AudioPipelineUlcnetConfig;

/** Defaults: the compiled ULCNet checkpoint grid,
 * balanced preset and all-zero model (identity). sample_rate is stored as
 * passed and validated at query/init time (only ULCNET_SR passes). */
AudioPipelineUlcnetConfig audio_pipeline_ulcnet_default_config(int sample_rate);

/* ============================================================================
 * Opaque handle
 * ========================================================================== */

typedef struct AudioPipelineUlcnet AudioPipelineUlcnet;

/* ============================================================================
 * Pool-first lifecycle
 * ========================================================================== */

/**
 * Query the memory descriptor for `cfg` WITHOUT touching any audio state.
 * Reject-first validation up front: sample_rate must be ULCNET_SR, fft_size
 * must be 0 or ULCNET_N_FFT, aec_preset must be a defined enum value; then the derived
 * AecConfig must pass lib/aec's own aec_get_mem_size() validator. Model
 * callbacks may be NULL (an all-zero model is legal); when io_descriptor is
 * non-NULL it must match the fixed aligned-far model ABI.
 *
 * @return 0 on success (*out filled), -1 on NULL args or invalid cfg.
 */
int audio_pipeline_ulcnet_get_mem_requirements(const AudioPipelineUlcnetConfig* cfg,
                                               AudioPipelineUlcnetMemReq* out);

/**
 * Initialize in a caller-owned pool. `mem` must be 16-byte aligned and at
 * least the size returned by audio_pipeline_ulcnet_get_mem_requirements().
 * It need not be zero-filled (every state field is explicitly zeroed), but
 * must remain stable and exclusive until the pipeline is no longer used.
 * Equivalent to audio_pipeline_ulcnet_init_ex(mem, bytes, cfg, NULL).
 *
 * @return a valid handle, or NULL on invalid config, pool, or submodule init.
 */
AudioPipelineUlcnet* audio_pipeline_ulcnet_init(void* mem, size_t bytes,
                                                const AudioPipelineUlcnetConfig* cfg);

/**
 * Initialize from a caller-owned pool and optionally reject a stale memory
 * descriptor. When `expected` is non-NULL, its descriptor/layout version,
 * backend id, build-flags hash, alignment, reserved field, and byte capacity
 * must match the requirements recomputed for this build and config (the same
 * 8-point gate audio_pipeline_init_ex applies). The supplied pool must
 * independently satisfy the current byte requirement. Pass NULL to obtain
 * audio_pipeline_ulcnet_init() behavior.
 *
 * @return a valid handle, or NULL on descriptor or initialization failure.
 */
AudioPipelineUlcnet* audio_pipeline_ulcnet_init_ex(void* mem, size_t bytes,
                                                   const AudioPipelineUlcnetConfig* cfg,
                                                   const AudioPipelineUlcnetMemReq* expected);

/**
 * Process exactly one hop (audio_pipeline_ulcnet_hop_size(p) == ULCNET_HOP)
 * of mic/ref into `out`: AEC(linear, context-only) -> error tap
 * (AecResContext.formed_hop) + AecLinearContext.aligned_far_hop -> two
 * centered-STFT analyses -> per emitted frame,
 * the model callback (or the fail-open identity, or the identity reprime
 * after an alignment boundary, per the policy in this header's preamble)
 * -> WOLA.
 * hop #0 writes all zeros; every later call writes exactly one hop whose
 * content corresponds to the PREVIOUS call's input (one-hop latency).
 *
 * `mic`/`ref` are read-only and only for the duration of this call; `out`
 * is fully overwritten (never read). All three must be exactly one hop of
 * floats.
 *
 * @return 0 on success, -1 if p/mic/ref/out is NULL.
 */
int audio_pipeline_ulcnet_process(AudioPipelineUlcnet* p, const float* mic,
                                  const float* ref, float* out);

/**
 * Re-zero all pipeline state: AEC reset, both analysis states, the synthesis
 * state, and model->reset (if set) so the external runtime flushes its own
 * explicit states too. Any pending identity reprime is dropped: the analysis
 * history is zeroed here, so the frames emitted after a reset straddle
 * nothing. The pool itself is untouched and cfg is not re-validated --
 * equivalent to a fresh init on the SAME pool/cfg without the alignment/size
 * re-checks.
 */
void audio_pipeline_ulcnet_reset(AudioPipelineUlcnet* p);

/**
 * Tear down. For a pool-resident (init/init_ex) instance every underlying
 * destroy is a no-op today (kept for forward-compat, mirroring
 * audio_pipeline_destroy's rationale) and the call is NULL-safe and
 * idempotent; the caller keeps its pool. For a heap instance obtained via
 * audio_pipeline_ulcnet_create(), this frees the owned pool -- call exactly
 * once (ordinary free() semantics).
 */
void audio_pipeline_ulcnet_destroy(AudioPipelineUlcnet* p);

/* ============================================================================
 * Heap convenience (desktop tools / tests -- NOT the board path)
 * ========================================================================== */

/**
 * get_mem_requirements + posix_memalign(16, ...) + init, in one call. The
 * returned handle owns its pool; audio_pipeline_ulcnet_destroy() frees it.
 *
 * @return a valid handle, or NULL (invalid cfg or allocation failure).
 */
AudioPipelineUlcnet* audio_pipeline_ulcnet_create(const AudioPipelineUlcnetConfig* cfg);

/* ============================================================================
 * Accessors
 * ========================================================================== */

int audio_pipeline_ulcnet_hop_size(const AudioPipelineUlcnet* p);  /* ULCNET_HOP, or -1 */

/**
 * Read-only access to the underlying AEC handle for a caller's OWN
 * diagnostics (aec_debug_status / aec_get_res_context /
 * aec_get_linear_context / ...). Do not call any _reset/_destroy/mutating
 * entry point on it directly -- go through audio_pipeline_ulcnet_reset()/
 * _destroy() so the pipeline-owned STFT/WOLA and model states stay in sync.
 */
Aec* audio_pipeline_ulcnet_get_aec(const AudioPipelineUlcnet* p);   /* never NULL for a valid p */

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_ULCNET_H */
