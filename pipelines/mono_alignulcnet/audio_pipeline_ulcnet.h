/**
 * Linkable AEC(linear) -> Align-ULCNet neural post-filter pipeline (mono).
 *
 * Same pool-first construction discipline as audio_pipeline.h (the
 * traditional AEC->NR->RES wrapper this variant sits beside), but the post
 * stage is the Align-ULCNet NN driven through the per-frame callback
 * boundary defined in AIAEC/Align_ULCNet/ulcnet_process.h (UlcnetModel).
 * The NN itself runs on an external runtime; this pipeline owns only the
 * verified C pre/post (centered sqrt-Hann STFT, 0/2/1 frame emission,
 * WOLA) plus the linear AEC and the delay-state policy around the model.
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
 * Signal grid (compile-time fixed by the ULCNET_* constants): 16 kHz only,
 * FFT 512 / hop 256 / 257 bins. Validation rejects any other sample rate
 * (8 k/48 k included) and any fft_size other than 0 (resolve to 512) or 512.
 *
 * Latency: end-to-end algorithmic latency is exactly ONE hop (256 samples,
 * 16 ms). hop #0 emits nothing (this pipeline writes zeros); the output of
 * hop #p (p >= 1) corresponds to input hop p-1.
 *
 * ── Far-input deployment contract (far_input_mode) ───────────────────────
 *  The config's far_input_mode selects which far stream feeds the model's
 *  far branch. The chosen mode MUST match the checkpoint's training far
 *  input -- a mismatch is an input-distribution change, not a tuning knob:
 *    ULCNET_FAR_RAW     (default) -> the caller's raw ref hop, same-hop with
 *                  the error tap. Checkpoint-compatible: current checkpoints
 *                  are trained with RAW far. The model's output is applied
 *                  WITHOUT any delay-lock gating (the paper contract does
 *                  not depend on lock); only infer() failure or a non-finite
 *                  output frame falls back to the identity path.
 *    ULCNET_FAR_ALIGNED -> the AEC's aligned far (AecLinearContext.
 *                  aligned_far_hop) plus delay-lock gating of the model
 *                  APPLICATION (the Phase-2 embedded candidate; see the
 *                  delay-gating rules below). Only use with a checkpoint
 *                  trained/fine-tuned on aligned far.
 *  That match is ENFORCED, not merely documented: when cfg.model publishes a
 *  model-I/O contract (model.io_descriptor != NULL, which the accelerator
 *  adapter always does), get_mem_requirements/init/init_ex all FAIL when its
 *  far_input_mode differs from cfg.far_input_mode, or when the descriptor
 *  carries an undefined mode -- so a RAW checkpoint cannot be wired into an
 *  ALIGNED pipeline, or the reverse, and no pool is even sized for the
 *  inconsistent pair. A model with io_descriptor == NULL (the all-zero
 *  identity case) publishes no contract and is not gated. These TUs carry no
 *  stdio, so the failure is the ordinary -1/NULL return; callers that want
 *  to name both sides read cfg.far_input_mode and
 *  cfg.model.io_descriptor->far_input_mode through
 *  ulcnet_far_input_mode_name() (ulcnet_model_io.h), as main.c does.
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
 *  - Delay events (AecLinearContext.delay_state, read once per hop; the
 *    UNLOCKED application bypass applies to ULCNET_FAR_ALIGNED only --
 *    ULCNET_FAR_RAW never gates application on the lock):
 *      UNLOCKED -> ALIGNED mode: the model's output is BYPASSED (fail-open
 *                  identity). The model is still STEPPED (infer() is
 *                  invoked for every emitted frame) so the per-hop
 *                  compute/timing budget stays constant and the runtime's
 *                  recurrent states keep tracking; its result is simply not
 *                  applied, because the far tap is raw/unaligned in this
 *                  state. RAW mode: applied as normal.
 *      CHANGED  -> model->reset (if set) is called BEFORE this hop's
 *                  infer() in BOTH far modes, so the runtime flushes its
 *                  far attention ring + logit history (the error branch
 *                  realigns discontinuously at this boundary even in RAW
 *                  mode); then infer() runs and its output is applied. The
 *                  first acquisition is itself a CHANGED event, so in
 *                  ALIGNED mode anything the model accumulated from raw far
 *                  during the UNLOCKED phase is flushed at that boundary.
 *      LOCKED   -> infer() runs and its output is applied.
 *    The C STFT/WOLA states keep running across a delay change; a 1-2 frame
 *    transient in the model's output around the reset is accepted and
 *    documented (crossfade is a later phase).
 *  - audio_pipeline_ulcnet_reset() also calls model->reset (if set).
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
 * then the ULCNet chain's shared 512-point FFT handle; the Ulcnet
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

/* UlcnetFarInputMode (ULCNET_FAR_RAW / ULCNET_FAR_ALIGNED) is defined once,
 * in AIAEC/Align_ULCNet/ulcnet_model_io.h, and reaches this header through
 * ulcnet_process.h. See this header's preamble for the per-mode gating
 * rules. */

/**
 * model is held BY VALUE (three plain pointers). A memset-zero model (or one
 * with infer == NULL) is the supported "no runtime attached" case: the
 * pipeline output is then the identity STFT->WOLA reconstruction of the
 * linear error, still with the one-hop latency contract. The pointers inside
 * model must outlive the pipeline instance; the pipeline never copies what
 * they point at.
 */
typedef struct {
    int         sample_rate;  /* must be 16000 (the compiled ULCNet grid)     */
    int         fft_size;     /* 0 (resolve to 512) or 512; others rejected   */
    int         filter_length; /* 0=rate default; PBFDKF taps                  */
    AecDelayMode delay_mode;   /* AEC far alignment policy                     */
    int         delay_num_filters;   /* MATCHED bank size [1,5]                 */
    int         fixed_delay_samples; /* FIXED samples; -1 for other modes       */
    AecPreset   aec_preset;   /* MILD | BALANCED | AGGRESSIVE                 */
    UlcnetModel model;        /* NPU callback boundary; may be all-zero       */
    UlcnetFarInputMode far_input_mode;  /* RAW (default) | ALIGNED; must
                              * match the checkpoint's training far input  */
} AudioPipelineUlcnetConfig;

/** Defaults: the trained ULCNet grid (16 kHz, frame/FFT 512, hop 256),
 * balanced preset,
 * all-zero model (identity), far_input_mode = ULCNET_FAR_RAW (the
 * checkpoint-compatible deployment contract). sample_rate is stored as
 * passed and validated at query/init time (only 16000 passes). */
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
 * Reject-first validation up front: sample_rate must be 16000, fft_size must
 * be 0 or 512, aec_preset must be a defined enum value; then the derived
 * AecConfig must pass lib/aec's own aec_get_mem_size() validator. The model
 * field is NOT validated (an all-zero model is legal).
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
 * Process exactly one hop (audio_pipeline_ulcnet_hop_size(p) == 256 samples)
 * of mic/ref into `out`: AEC(linear, context-only) -> error tap
 * (AecResContext.formed_hop) + far tap (raw `ref` in ULCNET_FAR_RAW,
 * AecLinearContext.aligned_far_hop in ULCNET_FAR_ALIGNED; both same-hop
 * with the error tap) -> two centered-STFT analyses -> per emitted frame,
 * the model callback (or the fail-open identity, per the policy in this
 * header's preamble) -> WOLA.
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
 * explicit states too. The pool itself is untouched and cfg is not
 * re-validated -- equivalent to a fresh init on the SAME pool/cfg without
 * the alignment/size re-checks.
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

int audio_pipeline_ulcnet_hop_size(const AudioPipelineUlcnet* p);  /* 256, or -1 if p is NULL */

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
