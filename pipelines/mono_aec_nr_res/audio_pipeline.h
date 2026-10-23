/** Linkable AEC(linear) -> echo-aware NR -> RES pipeline.
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
 * Re-query the descriptor after any library, backend, build-option or config
 * change. audio_pipeline_create()/destroy() is a heap path for desktop tools. */
#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"                /* AecConfig/AecPreset, Aec (non-opaque)      */
#include "anr.h"  /* AnrConfig/AnrMode, AnrDenoiser (opaque) */
#include "common/se_stage.h"    /* SeSpecFrame */

#ifdef __cplusplus
extern "C" {
#endif

/** AudioPipelineMemReq ABI version; independent of the pool layout version. */
#define AUDIO_PIPELINE_DESCRIPTOR_VERSION 2u

/** FFT backend identifier. Zero and 1 are never returned. */
#define AUDIO_PIPELINE_BACKEND_NE10 2u

typedef struct {
    uint32_t descriptor_version;  /* = AUDIO_PIPELINE_DESCRIPTOR_VERSION (2) */
    uint32_t layout_version;      /* carve-layout version                    */
    uint32_t backend_id;          /* AUDIO_PIPELINE_BACKEND_NE10 = 2         */
    uint32_t build_flags_hash;    /* FNV-1a-32 */
    uint32_t alignment;           /* 16 */
    uint32_t reserved;            /* 0; keeps 8-byte alignment for bytes */
    uint64_t bytes;               /* total pool size */
} AudioPipelineMemReq;

/* Pins the serialized ABI field by field. */
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

/** Initialization contract shared by host tools and board integrations. */
typedef struct {
    int           sample_rate;   /* 8000 | 16000 | 48000                              */
    int           fft_size;      /* 0=rate default; 256/512 @16k, 1024 @48k           */
    int           filter_length; /* 0=rate default; PBFDKF taps, init-time immutable   */
    AecDelayMode  delay_mode;    /* MATCHED | FIXED | EXTERNAL_ALIGNED                 */
    int           delay_num_filters;   /* MATCHED bank size [1,5]                       */
    int           fixed_delay_samples; /* FIXED native-rate samples; -1 otherwise       */
    AecPreset     aec_preset;    /* MILD | BALANCED | AGGRESSIVE                       */
    AnrMode nr_mode;       /* MILD | MODERATE | BALANCED | AGGRESSIVE            */
    int           aec_only;      /* 1 = skip NR/RES (linear AEC output only); outranks enable_nr/enable_res */
    int           enable_nr;     /* 1 = run MMSE-LSA (default); 0 = no denoiser, g_total is G_res alone */
    int           enable_res;    /* 1 = apply the residual gain G_res (default); 0 = unity (CNG is skipped too) */
    int           enable_cng;    /* 1 = fill AEC-suppressed bins with comfort noise    */
    int           legacy_amin;   /* 1 = NR gain computed without folding R^2 into its noise floor; keep 0 */
} AudioPipelineConfig;

/** Defaults: rate-default no-padding grid, balanced modes, full pipeline,
 * MATCHED delay with the five-filter reference bank and default AEC length. */
AudioPipelineConfig audio_pipeline_default_config(int sample_rate);

typedef struct AudioPipeline AudioPipeline;

/** Queries the memory descriptor for `cfg` without allocating or touching audio
 * state. Rejects, before any size arithmetic: sample_rate outside {8000, 16000,
 * 48000}; aec_preset/nr_mode outside their enums; aec_only/enable_nr/enable_res/
 * enable_cng/legacy_amin outside {0,1}; then the AEC and NR validators.
 * Returns 0 on success (*out filled), -1 on NULL args or invalid cfg. */
int audio_pipeline_get_mem_requirements(const AudioPipelineConfig* cfg,
                                         AudioPipelineMemReq* out);

/** Initialises AEC + OLA + NR and all scratch in a caller-owned pool: `mem` 16-byte
 * aligned, at least the size from get_mem_requirements(), stable and exclusive for
 * the pipeline's lifetime (no zero-fill needed). Same as init_ex(mem, bytes, cfg,
 * NULL). Returns a handle, or NULL on invalid config, pool or submodule init. */
AudioPipeline* audio_pipeline_init(void* mem, size_t bytes,
                                    const AudioPipelineConfig* cfg);

/** Like init, and when `expected` is non-NULL rejects a stale descriptor: its
 * versions, backend, build-flags hash, alignment, reserved field and byte capacity
 * must match the requirements recomputed for this build and config. Returns a
 * handle, or NULL on descriptor or initialization failure. */
AudioPipeline* audio_pipeline_init_ex(void* mem, size_t bytes,
                                       const AudioPipelineConfig* cfg,
                                       const AudioPipelineMemReq* expected);

/** Processes exactly one hop (audio_pipeline_hop_size(p) samples) of mic/ref into
 * `out`: AEC(linear) -> echo-aware NR gain -> g_total = min(g_nr, g_res) ->
 * S(f) = E(f) * g_total (+ CNG on the cut bins) -> irfft -> sqrt-Hann OLA.
 * `aec_only` returns the raw linear AEC residual. `mic`/`ref` are read-only for the
 * call; `out` is fully overwritten; all are one hop of floats. Returns 0, or -1 if
 * p/mic/ref/out is NULL. */
int audio_pipeline_process(AudioPipeline* p, const float* mic,
                            const float* ref, float* out);

/** Stage seam: runs the context-only linear AEC for one hop and hands back its
 * spectra in `*frame` (se_stage.h: e, y, x, r2, cn, g_res_ctx, formed_hop, far_hop,
 * delay flags), borrowed from the AEC and valid until the next processing call or
 * reset on `p`. The caller chains its own stages (se_anr_stage_gain(), sk_min_f32,
 * sk_capply_gain_f32, aec_cng_fill) and finishes with audio_pipeline_synthesize().
 * Requires !aec_only; -1 on NULL arguments or an aec_only instance. */
int audio_pipeline_process_spec(AudioPipeline* p, const float* mic,
                                const float* ref, SeSpecFrame* frame);

/** Synthesises `spectrum` (audio_pipeline_n_freqs(p) bins) through the instance's own
 * iFFT/OLA, the same synthesis audio_pipeline_process() uses; `out` receives one
 * hop. -1 on NULL arguments or an aec_only instance. */
int audio_pipeline_synthesize(AudioPipeline* p, const Complex* spectrum,
                              float* out);

/** Re-zeros all pipeline/AEC/NR state (OLA accumulator, comfort-noise RNG, sub-module
 * resets) without touching the pool or re-validating cfg. Use after an echo-path
 * change or between unrelated streams. */
void audio_pipeline_reset(AudioPipeline* p);

/* Retargets the residual-echo strength on a running pipeline; filter, delay lock and smoothing carry on. ramp_ms 0 applies on the next hop at exactly the floor a fresh instance would hold; a positive value walks there linearly in dB. Call between hops, serialised with audio_pipeline_process(); not thread-safe. Returns 0, or -1 on NULL, an out-of-enum preset or out-of-range ramp_ms (nothing written). */
int audio_pipeline_set_aec_preset(AudioPipeline* p, AecPreset preset,
                                  float ramp_ms);

/* Retargets the noise-reduction strength on a running pipeline: recomposes this
 * pipeline's NR configuration (canonical preset plus its overrides) and calls
 * anr_reconfigure(). The tracked noise floor and gain smoothing survive; use reset()
 * for a restart. Returns 0, or -1 on NULL, an aec_only build, an out-of-enum mode or
 * a rejected target. */
int audio_pipeline_set_nr_mode(AudioPipeline* p, AnrMode mode);

/** Tears down in reverse carve order: NR -> OLA FFT -> AEC. NULL-safe; idempotent for
 * a pool-resident instance (the caller keeps its pool). For an instance from
 * audio_pipeline_create() it frees the owned pool: call exactly once. */
void audio_pipeline_destroy(AudioPipeline* p);

/** get_mem_requirements() + posix_memalign(16, ...) + init() in one call (desktop
 * tools, not the board path). The handle owns its pool; destroy() frees it. Returns
 * a handle, or NULL (invalid cfg or allocation failure). */
AudioPipeline* audio_pipeline_create(const AudioPipelineConfig* cfg);

int audio_pipeline_hop_size(const AudioPipeline* p);     /* -1 if p is NULL */
int audio_pipeline_n_freqs(const AudioPipeline* p);      /* -1 if p is NULL */
int audio_pipeline_sample_rate(const AudioPipeline* p);  /* -1 if p is NULL */

/** Read-only access to the module handles for diagnostics (aec_debug_status,
 * aec_get_res_context, anr_debug_status, anr_get_gain, ...). Do not reset, destroy
 * or mutate them directly; use audio_pipeline_reset()/audio_pipeline_destroy(). */
Aec*             audio_pipeline_get_aec(const AudioPipeline* p);  /* never NULL for a valid p */
AnrDenoiser* audio_pipeline_get_nr(const AudioPipeline* p);   /* NULL iff cfg.aec_only     */

/** Per-module byte breakdown (AEC / FFT / NR / pipeline buffers) for the
 * --print-mem-size table; already folded into AudioPipelineMemReq.bytes. */
typedef struct {
    size_t aec_bytes;
    size_t fft_bytes;         /* FFT handle inside the synthesis object (counted in the pool once); 0 when aec_only */
    size_t nr_bytes;          /* 0 when cfg.aec_only */
    size_t pipeline_bytes;    /* synthesis object without its FFT plus gain and spectrum scratch; 0 when aec_only */
    int    hop, frame_sz, fft_sz, n_freqs;
} AudioPipelineMemBreakdown;

/** Returns 0 on success (*out filled), -1 on NULL args or invalid cfg. */
int audio_pipeline_get_mem_breakdown(const AudioPipelineConfig* cfg,
                                      AudioPipelineMemBreakdown* out);

/** Per-hop wall-clock cost in microseconds of the stages in audio_pipeline_process(); diagnostic only. aec = the AEC stages (aec_get_last_timing(), aec.h); nr_us = r2 fold + anr_process_gain(); post_us = min(G_nr, G_res), apply and comfort noise; synth_us = inverse transform, OLA and hop emit. The stages do not sum to the call; nr/post/synth stay 0 for aec_only. Off by default (fields read 0): -DAUDIO_PIPELINE_STAGE_TIMING=1 for nr/post/synth and -DAEC_STAGE_TIMING=1 for the AEC block (`make PROFILE=1` sets both); a target without POSIX CLOCK_MONOTONIC names its timer with -DAUDIO_PIPELINE_NOW_US=<fn> (and -DAEC_NOW_US=<fn>). The record is always present, so builds carve identical pools. */
typedef struct AudioPipelineLastTiming {
    /* Copied from aec_get_last_timing(). */
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
