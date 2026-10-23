/** Linkable AEC(linear) -> Align-ULCNet neural post-filter pipeline (mono). Pool-first like audio_pipeline.h; the network runs on an external runtime through the per-frame UlcnetModel callback (ulcnet_process.h). This pipeline owns the C pre/post (rolling center=False sqrt-Hann STFT, one frame and at most one inference per hop, WOLA), the linear AEC and the delay-state policy.
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
 * Grid: compile-time (ULCNET_* constants): 16 kHz / FFT 512 / hop 256, or 48 kHz / FFT 1024 / hop 512; other grids are rejected. Latency: exactly one hop; hop #0 writes zeros and hop #p corresponds to input hop p-1. The model receives the caller's raw `ref` hop (a published descriptor must carry ULCNET_FAR_RAW); PBFDKF uses its own aligned copy to form the error.
 * Model policy: infer == NULL (or an all-zero model), a nonzero infer() result or any non-finite output bin passes the error spectrum through unchanged, with the same timing. On a CHANGED delay event (FIXED mode: the first UNLOCKED->LOCKED transition) model->reset (if set) runs before the hop's frames and the next AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES frames take the identity path without stepping the model, since windows in flight straddle the boundary; a second boundary re-arms the counter. reset() also calls model->reset. */
#ifndef AUDIO_PIPELINE_ULCNET_H
#define AUDIO_PIPELINE_ULCNET_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"
#include "ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Emitted frames that still straddle an alignment boundary, i.e. the length of the
 * identity reprime armed at every generation change. Both branches are pushed from
 * the current hop and the analysis frame at hop T spans hops T-1 and T. */
enum { AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES = 1 };

/** AudioPipelineUlcnetMemReq ABI version (independent of the pool layout version). */
#define AUDIO_PIPELINE_ULCNET_DESCRIPTOR_VERSION 1u

/** FFT backend identifiers (as audio_pipeline.h); 0 and 1 are never returned. */
#define AUDIO_PIPELINE_ULCNET_BACKEND_NE10 2u

/** Fixed 32-byte same-endian descriptor; re-query after any build or configuration change. layout_version and build_flags_hash describe this wrapper's carve; dependency layouts are reflected in `bytes`. reserved must be 0. */
typedef struct {
    uint32_t descriptor_version;  /* = AUDIO_PIPELINE_ULCNET_DESCRIPTOR_VERSION (1) */
    uint32_t layout_version;      /* carve-layout version, starts at 1        */
    uint32_t backend_id;          /* _BACKEND_NE10 = 2                         */
    uint32_t build_flags_hash;    /* FNV-1a-32 over backend + carve tokens    */
    uint32_t alignment;           /* 16                                       */
    uint32_t reserved;            /* 0; keeps 8-byte alignment for bytes      */
    uint64_t bytes;               /* total pool size                          */
} AudioPipelineUlcnetMemReq;

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

/** model is held by value. An all-zero model (or infer == NULL) means no runtime: the output is the identity STFT->WOLA reconstruction of the linear error, with the same latency. Pointers inside model must outlive the pipeline. */
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

/** Defaults: the compiled checkpoint grid, balanced preset, all-zero model (identity). sample_rate is validated at query/init time. */
AudioPipelineUlcnetConfig audio_pipeline_ulcnet_default_config(int sample_rate);

typedef struct AudioPipelineUlcnet AudioPipelineUlcnet;

/** Queries the memory descriptor for `cfg` without touching audio state. Requires sample_rate == ULCNET_SR, fft_size 0 or ULCNET_N_FFT, a defined aec_preset and an AecConfig that passes aec_get_mem_size(); a non-NULL model io_descriptor must match the raw-far ABI. Returns 0 (*out filled), or -1 on NULL args or invalid cfg. */
int audio_pipeline_ulcnet_get_mem_requirements(const AudioPipelineUlcnetConfig* cfg,
                                               AudioPipelineUlcnetMemReq* out);

/** Initialises in a caller-owned pool: `mem` 16-byte aligned, >= the queried size, exclusive for the pipeline's lifetime. Same as init_ex(mem, bytes, cfg, NULL). Returns a handle, or NULL on invalid config, pool or submodule init. */
AudioPipelineUlcnet* audio_pipeline_ulcnet_init(void* mem, size_t bytes,
                                                const AudioPipelineUlcnetConfig* cfg);

/** Like init; with a non-NULL `expected` a stale descriptor (versions, backend id, build-flags hash, alignment, reserved, byte capacity) is rejected. Returns a handle, or NULL. */
AudioPipelineUlcnet* audio_pipeline_ulcnet_init_ex(void* mem, size_t bytes,
                                                   const AudioPipelineUlcnetConfig* cfg,
                                                   const AudioPipelineUlcnetMemReq* expected);

/** Processes one hop (audio_pipeline_ulcnet_hop_size(p) == ULCNET_HOP) of mic/ref into `out`: linear AEC -> error tap + raw `ref` hop -> two STFT analyses -> model callback (or identity) -> WOLA. hop #0 writes zeros; later calls write the previous call's hop. `out` is fully overwritten. Returns 0, or -1 on a NULL pointer. */
int audio_pipeline_ulcnet_process(AudioPipelineUlcnet* p, const float* mic,
                                  const float* ref, float* out);

/** The same hop in phases, for a caller that runs the network itself (a reset callback, if set, is still called):
 *     audio_pipeline_ulcnet_process_begin(p, mic, ref);   // AEC, both analyses
 *     if (audio_pipeline_ulcnet_nn_begin(p, &job) > 0) {  // a frame to run
 *         rc = run_network(&job);                         // 0 = all bins written
 *         audio_pipeline_ulcnet_nn_end(p, rc);            // accept, or the identity
 *     }
 *     audio_pipeline_ulcnet_process_end(p, out);          // WOLA, one hop
 * nn_begin returns 1 with the frame in *job, 0 when the hop has no model frame (identity reprime), -1 on a sequence error. nn_end accepts the output only when rc == 0 and every bin is finite. A frame left open or never fetched is the identity. The others return 0 or -1 (NULL arguments, out-of-sequence call). */
int audio_pipeline_ulcnet_process_begin(AudioPipelineUlcnet* p, const float* mic,
                                        const float* ref);
int audio_pipeline_ulcnet_nn_begin(AudioPipelineUlcnet* p, UlcnetNnJob* job);
int audio_pipeline_ulcnet_nn_end(AudioPipelineUlcnet* p, int model_rc);
int audio_pipeline_ulcnet_process_end(AudioPipelineUlcnet* p, float* out);

/** Re-zeros all pipeline state and calls model->reset (if set); a pending reprime is dropped. The pool is untouched and cfg is not re-validated. */
void audio_pipeline_ulcnet_reset(AudioPipelineUlcnet* p);

/** Tears down. A pool-resident instance: NULL-safe idempotent no-op, the caller keeps the pool. An instance from _create(): frees its pool (call once). */
void audio_pipeline_ulcnet_destroy(AudioPipelineUlcnet* p);

/** get_mem_requirements + posix_memalign(16, ...) + init (desktop tools). The handle owns its pool; destroy() frees it. NULL on invalid cfg or allocation failure. */
AudioPipelineUlcnet* audio_pipeline_ulcnet_create(const AudioPipelineUlcnetConfig* cfg);

int audio_pipeline_ulcnet_hop_size(const AudioPipelineUlcnet* p);  /* ULCNET_HOP, or -1 */

/** Read-only AEC handle for diagnostics; do not reset, destroy or mutate it directly. */
Aec* audio_pipeline_ulcnet_get_aec(const AudioPipelineUlcnet* p);   /* never NULL for a valid p */

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_ULCNET_H */
