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

/* ── Runtime strength control ─────────────────────────────────────────────
 *
 * Retarget the residual-echo strength on a RUNNING pipeline. The three
 * presets differ in one scalar floor, so this is a retarget rather than a
 * rebuild: the filter, the delay lock and every smoothing history carry on.
 *
 * ramp_ms is forwarded to the suppressor -- 0 applies on the next hop and
 * lands on exactly the floor a fresh instance would hold, a positive value
 * walks there linearly in dB. See aec.h for why that matters.
 *
 * Note when measuring: the far-active floor only binds on far-active,
 * non-double-talk hops, and the same gain also scales the injected comfort
 * noise, so a whole-recording average moves less than the dB step implies.
 *
 * Call between hops, serialised with audio_pipeline_process(); not
 * thread-safe. Returns 0, or -1 on NULL, an out-of-enum preset or an
 * out-of-range ramp_ms, with nothing written. */
int audio_pipeline_set_aec_preset(AudioPipeline* p, AecPreset preset,
                                  float ramp_ms);

/* Retarget the noise-reduction strength on a RUNNING pipeline.
 *
 * This recomposes THIS pipeline's own NR configuration -- the canonical
 * strength preset plus the overrides it has always applied on top -- and hands
 * the result to mmse_lsa_reconfigure(). It deliberately does not call
 * mmse_lsa_set_mode(), which composes the bare canonical preset and would
 * either be refused (its L differs) or revert those overrides.
 *
 * The tracked noise floor and the gain smoothing history survive the change;
 * use audio_pipeline_reset() for a restart.
 *
 * Returns 0, or -1 on NULL, an aec_only build (no denoiser exists), an
 * out-of-enum mode, or a rejected target. */
int audio_pipeline_set_nr_mode(AudioPipeline* p, MmseLsaNrMode mode);

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

/* ============================================================================
 * Diagnostic per-stage timing
 * ========================================================================== */

/**
 * Per-hop wall-clock cost of the stages inside audio_pipeline_process(), in
 * microseconds. Diagnostic only: nothing in the chain reads these back and
 * they do not affect processing. The stamp is described at the end of this
 * comment, along with what a target substitutes it with.
 *
 * STAGE BOUNDARIES
 *
 *   aec          the AEC's own four stages, copied verbatim from
 *                aec_get_last_timing(): aec.delay_us, aec.frontend_us,
 *                aec.linear_us, aec.res_us. AecStageTiming's doc comment
 *                (aec.h) is the single definition of what each covers --
 *                this header deliberately does not restate it.
 *
 *                Worth knowing before reading them: on the default MATCHED
 *                path aec.delay_us is routinely the LARGEST single stage of
 *                the hop, measured at roughly two thirds of it on a
 *                16 kHz/256 grid. Where the four-lane wrapper differs: there
 *                the estimator is hoisted OUT of the lanes and run once for
 *                all four, so its delay figure is the wrapper's own stage.
 *                Here it stays inside the single AEC -- same quantity,
 *                different owner.
 *   nr_us        mmse_lsa_process_gain() -- the noise-reduction gain.
 *   post_us      the gain arithmetic between the two: the r2/PSD_SCALE fold,
 *                min(G_nr, G_res), the |E|^2 hoist, the far-activity and
 *                near-VAD gate, the echo-gated near-end lift, the spectral
 *                apply, and the comfort-noise loop.
 *   synth_us     inverse transform, windowed overlap-add, and the hop
 *                emit/shift.
 *
 * WHERE THE AEC BLOCK COMES FROM
 *
 * aec_get_last_timing() (aec.h), read once after the AEC call into the
 * embedded record. They are governed by that library's OWN build flag, not this
 * one -- see the two-halves note below.
 *
 * The stages do NOT sum to the call. The remainder holds the argument
 * checks, aec_get_res_context(), and -- inside the AEC and therefore
 * invisible from here -- the stationarity refresh, e2_coarse/ERL publish,
 * the DT analyzer, EPV, shadow_rise, the misadjustment estimator, the power
 * EMAs and convergence detection. A caller presenting a full breakdown must
 * carry that remainder explicitly rather than implying the parts are whole.
 *
 * AEC-ONLY INSTANCES. cfg.aec_only returns as soon as aec_process() does,
 * so nr_us/post_us/synth_us stay 0 on every hop: those stages do not exist
 * in that mode. That zero means "no such stage", while the same zero in a
 * !aec_only build means "not measured" -- the two are distinguished by
 * audio_pipeline_get_nr() being NULL, not by the record.
 *
 * RESOLUTION. Microseconds, so a stage faster than 1 us reads 0. On the
 * development host clock_getres(CLOCK_MONOTONIC) is itself 1 us, which puts
 * post/synth near that floor -- readable as an order of magnitude, not as a
 * precise figure. linear_us, the stage this exists to find, is far above it.
 *
 * OFF BY DEFAULT, IN TWO HALVES. This record costs four clock reads per hop
 * here, on top of whatever the AEC makes inside its own call (its own
 * header carries that count). A release
 * build takes none of them and every field reads 0. Build with
 * -DAUDIO_PIPELINE_STAGE_TIMING=1 for this pipeline's own stages
 * (nr/post/synth) and -DAEC_STAGE_TIMING=1 for the embedded
 * AEC block; `make PROFILE=1` sets both, which is what a profile
 * build should use. Setting one alone is legible rather than broken: the
 * other half simply reads 0.
 *
 * A display-side flag in the CONSUMER does not enable any of this. It
 * decides whether a breakdown is printed; these decide whether there is
 * anything to print. Built one way and read the other, the report renders
 * zeros.
 *
 * A target whose libc has no POSIX CLOCK_MONOTONIC combines the flags with
 * -DAUDIO_PIPELINE_NOW_US=<fn> (and lib/aec's own -DAEC_NOW_US=<fn>) to name
 * its microsecond timer -- see the stamp's comment in audio_pipeline.c. A
 * substitute that returns a constant keeps the flags on and reads 0 here,
 * which this record cannot distinguish from a build without them.
 *
 * THE RECORD IS NOT CONDITIONAL, ONLY THE STAMPING IS. The field lives in
 * the control block whether or not either flag is set, so a profile build
 * and a release build carve byte-identical pools and one memory budget
 * covers both. Turning the flag on never moves an offset.
 *
 * LIFETIME. The values describe the last ACCEPTED hop. This pipeline has no
 * path that is accepted and then abandoned -- audio_pipeline_process()
 * validates its arguments and then runs to completion -- so every field a
 * mode has is rewritten every hop, and the three an aec_only instance does
 * not have stay zero from init. A call REJECTED by the argument checks
 * leaves the record untouched, and so does audio_pipeline_reset(): the
 * record describes hops, not instance state, and the next accepted hop is
 * what replaces it. Read it after audio_pipeline_process() returns 0.
 */
typedef struct AudioPipelineLastTiming {
    /* Verbatim from aec_get_last_timing(). Embedded rather than flattened so
     * a field the AEC adds arrives here without this header re-enumerating
     * its stage set -- and so there is exactly one place that names those
     * stage boundaries: AecStageTiming's own doc in aec.h. */
    AecStageTiming aec;
    uint32_t nr_us;
    uint32_t post_us;
    uint32_t synth_us;
} AudioPipelineLastTiming;

/* Copies the last hop's stage timings into `out`. A NULL pipeline zeroes
 * `out` rather than failing, so a diagnostic caller needs no special case.
 * `out` must not be NULL. */
void audio_pipeline_get_last_timing(const AudioPipeline* p,
                                    AudioPipelineLastTiming* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_H */
