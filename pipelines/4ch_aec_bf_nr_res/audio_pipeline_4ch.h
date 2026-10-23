/**
 * audio_pipeline_4ch.h — complete four-channel spatial pipeline.
 *
 * Complete four-channel traditional speech-enhancement pipeline:
 *
 *   one shared far aligner -> four linear AEC lanes
 *   -> SRP-PHAT DOA -> BF -> one NR + one post-beam RES
 *
 * SRP-PHAT/BF come from the sibling audiolib/doa and audiolib/bf
 * libraries. This wrapper owns their lifetime and preserves the
 * resource boundary of 4aec_nr_res.h. Naming and lifecycle mirror
 * audio_pipeline.h/4aec_nr_res.h's own descriptor-tier pool-first pattern
 * (this layer composes core + SRP + BF + its own scratch, the same tier as
 * 4aec_nr_res.c, not the simple size_t tier SRP/BF each expose on their
 * own):
 *
 *   Caller-owned pool (board/static path):
 *
 *     AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
 *     AudioPipeline4ChMemReq req;
 *     audio_pipeline_4ch_get_mem_requirements(&cfg, &req);
 *     void* pool = platform_alloc(req.bytes, req.alignment);
 *     AudioPipeline4Ch* p =
 *         audio_pipeline_4ch_init_ex(pool, req.bytes, &cfg, &req);
 *     ...
 *     audio_pipeline_4ch_destroy(p);  // never releases caller-owned pool
 *     platform_free(pool);
 *
 *   Heap convenience (desktop/test path):
 *
 *     AudioPipeline4Ch* p = audio_pipeline_4ch_create(&cfg);
 *     ...
 *     audio_pipeline_4ch_destroy(p);  // releases create()'s one pool
 *
 * This adds the spatial stage between the core's four_aec_front_process() and
 * four_aec_fuse() calls and chains the post-beam stages itself (RES gain, NR
 * gain, min, apply, comfort noise, synthesis); both construction paths use the
 * same composition and must be byte-identical for the same inputs/config/
 * backend.
 */
#ifndef AUDIO_PIPELINE_4CH_H
#define AUDIO_PIPELINE_4CH_H

#include "4aec_nr_res.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum AudioPipeline4ChGeometry {
    AUDIO_PIPELINE_4CH_GEOMETRY_UCA = 0,
    AUDIO_PIPELINE_4CH_GEOMETRY_ULA = 1,
    AUDIO_PIPELINE_4CH_GEOMETRY_CUSTOM = 2
} AudioPipeline4ChGeometry;

typedef struct AudioPipeline4ChConfig {
    FourAecNrResConfig core;

    AudioPipeline4ChGeometry geometry;
    float uca_radius_m;
    float ula_spacing_m;
    float microphone_x_m[FOUR_AEC_NR_RES_CHANNELS];
    float microphone_y_m[FOUR_AEC_NR_RES_CHANNELS];

    int num_angles;
    float speed_of_sound_m_s;
    float doa_low_freq_hz;
    float doa_high_freq_hz;       /* 0 = min(7000, Nyquist) */
    int doa_enable_smoothing;
    int doa_switch_consecutive;
    float doa_angle_tolerance_rad;
    int doa_update_interval;
    int bf_enable;
    float bf_lambda;
    float bf_mu;
    int bf_fixed_mode;
    float bf_fixed_doa_rad;
    int bf_fixed_align_notebook;
    int bf_adapt_interval;

    /*
     * Fallback activity detector used only by audio_pipeline_4ch_process().
     * Production integrations that own a VAD should call
     * audio_pipeline_4ch_process_with_activity() instead.
     */
    float auto_vad_threshold_dbfs;
    float auto_vad_snr_ratio;
    int auto_vad_hangover_frames;
} AudioPipeline4ChConfig;

typedef struct AudioPipeline4ChFrameInfo {
    uint64_t frame_index;
    FourAecNrResDelayState delay;
    float doa_raw_rad;
    float doa_smooth_rad;
    float doa_used_rad;
    int vad_raw;
    int vad_out;
    int bf_adaptive;
    int doa_analysis_frames;
} AudioPipeline4ChFrameInfo;

typedef struct AudioPipeline4Ch AudioPipeline4Ch;

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

#define AUDIO_PIPELINE_4CH_DESCRIPTOR_VERSION 1u
/* Carve layout version of this wrapper. Any change of pool size, carve order
 * or descriptor content bumps it, including a change of the composed core's
 * pool or layout version: build_flags_hash folds in the core's carve-token
 * hash only, so a control-block-only change of the core leaves it unmoved and
 * this counter is the signal. Query a fresh descriptor for every build; do not
 * persist or synthesize one from this number. The exact totals come from
 * `make print-mem-size`. */
#define AUDIO_PIPELINE_4CH_LAYOUT_VERSION 17u

/**
 * Fixed-width descriptor for a caller-owned static-memory pool. Same 32-byte
 * fixed-width-integer shape and same field meanings as 4aec_nr_res.h's
 * FourAecNrResMemReq (descriptor_version/layout_version/backend_id/
 * build_flags_hash/alignment/reserved/bytes) -- see that header's own doc
 * comment for the full rationale rather than repeating it here.
 *
 * This layer composes the core (4aec_nr_res), SRP-PHAT, and BF sub-modules
 * one level up, so its own build_flags_hash folds in the core's
 * build_flags_hash (see audio_pipeline_4ch_get_mem_requirements()'s
 * implementation): a core-layer layout bump must also invalidate every
 * persisted composite descriptor here, never silently keep fitting a pool
 * sized for a stale core layout.
 */
typedef struct AudioPipeline4ChMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} AudioPipeline4ChMemReq;

_Static_assert(
    sizeof(AudioPipeline4ChMemReq) == 32,
    "AudioPipeline4ChMemReq must be exactly 32 bytes");
_Static_assert(
    offsetof(AudioPipeline4ChMemReq, descriptor_version) == 0,
    "descriptor_version offset");
_Static_assert(
    offsetof(AudioPipeline4ChMemReq, layout_version) == 4,
    "layout_version offset");
_Static_assert(
    offsetof(AudioPipeline4ChMemReq, backend_id) == 8,
    "backend_id offset");
_Static_assert(
    offsetof(AudioPipeline4ChMemReq, build_flags_hash) == 12,
    "build_flags_hash offset");
_Static_assert(
    offsetof(AudioPipeline4ChMemReq, alignment) == 16,
    "alignment offset");
_Static_assert(
    offsetof(AudioPipeline4ChMemReq, reserved) == 20,
    "reserved offset");
_Static_assert(
    offsetof(AudioPipeline4ChMemReq, bytes) == 24,
    "bytes offset");

/* ============================================================================
 * Config and lifecycle
 * ========================================================================== */

AudioPipeline4ChConfig audio_pipeline_4ch_default_config(int sample_rate);

/**
 * Query the complete 16-byte-aligned caller pool required by cfg. Performs
 * no allocation and returns 0 on success. Composes the core's own
 * four_aec_nr_res_get_mem_requirements()/get_mem_breakdown() with
 * srp_get_mem_size()/bf_get_mem_size() and this wrapper's own two scratch
 * buffers (bf_spectrum/bf_weights).
 */
int audio_pipeline_4ch_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChMemReq* out);

/**
 * Place the complete four-channel spatial pipeline (core + SRP-PHAT + BF +
 * this wrapper's own scratch) in caller-owned memory.
 *
 * mem must satisfy the freshly queried alignment and byte requirement. No
 * heap allocation is performed by this path, including inside the core, SRP,
 * or BF. A dirty/poisoned pool is accepted.
 */
AudioPipeline4Ch* audio_pipeline_4ch_init(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg);

/**
 * Static init plus a stale-descriptor gate. expected should be the
 * descriptor returned by a fresh get_mem_requirements() call for this exact
 * build/config. NULL expected is equivalent to audio_pipeline_4ch_init().
 * Same 8-point rejection contract as 4aec_nr_res.h's four_aec_nr_res_init_ex()
 * (descriptor_version/layout_version/backend_id/build_flags_hash/alignment
 * must match; reserved must be 0; expected->bytes and the pool actually
 * handed to this call must both be >= the current build's requirement).
 */
AudioPipeline4Ch* audio_pipeline_4ch_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChMemReq* expected);

/**
 * Heap convenience wrapper over get_mem_requirements() + init(). All memory
 * is allocated here; per-hop processing performs no allocation. Returns NULL
 * for an invalid config or allocation failure.
 */
AudioPipeline4Ch* audio_pipeline_4ch_create(
    const AudioPipeline4ChConfig* cfg);

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

/**
 * Complete one hop using the wrapper's conservative energy-VAD fallback.
 */
int audio_pipeline_4ch_process(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    float* output,
    AudioPipeline4ChFrameInfo* info);

/**
 * Complete one hop with activity supplied by the product VAD.
 *
 * vad_raw controls whether SRP-PHAT gets a fresh observation. vad_out is
 * the held target-speech state used by the DOA smoother and freezes the BF
 * RLS update when nonzero.  frequency_mask may be NULL or n_freqs integers;
 * 1 selects a bin for SRP and freezes BF adaptation in that bin.
 */
int audio_pipeline_4ch_process_with_activity(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    float* output,
    AudioPipeline4ChFrameInfo* info);

void audio_pipeline_4ch_reset(AudioPipeline4Ch* p);

/* Runtime strength control.
 *
 * The four AEC lanes run with spatial_linear_context and never reach their own
 * suppression-gain path, so the strength that shapes this product's output
 * lives in this pipeline's single post-beam suppressor (AecRes); likewise there
 * is one denoiser, not one per lane. Both setters target those, which is why
 * they exist rather than leaving the caller to loop over lanes.
 *
 * set_aec_preset retargets all three strength fields on the post-beam
 * suppressor: the two gain floors, and the split-floor-enabled gate itself.
 * The far-active floor ramps (ramp_ms 0 applies on the next hop and lands on
 * exactly the floor a fresh instance would hold, a positive value walks there
 * linearly in dB), while the DT floor and the gate both jump immediately to
 * the new preset's value (DT floor: -20 dB for mild/balanced, -30 dB for
 * aggressive; see AEC_PRESET_FLOORS in aec.c) -- ramp_ms only governs the
 * far-active step. The far-active floor only binds on far-active,
 * non-double-talk hops (during double-talk the DT floor applies instead),
 * and the same gain also scales the comfort noise, so an average over a
 * whole recording moves less than the dB step implies. Switching to/from
 * AEC_PRESET_NO_FLOOR (gate OFF) is allowed at runtime, not just at
 * construction: the gate steps on the same call that retargets the two dB
 * fields, so the instance never has a hop where the preset label and the
 * gate disagree. Requires core.enable_res.
 *
 * set_nr_mode recomposes this pipeline's own NR configuration -- the canonical
 * strength preset plus the overrides it always applies -- and hands it to
 * anr_reconfigure(); the noise floor, the min-tracking ring and the gain
 * smoothing history carry on. anr_set_mode() would be refused (its L differs)
 * or revert the overrides. Requires core.enable_nr.
 *
 * Call between hops. Return 0, or -1 with nothing written. */
int audio_pipeline_4ch_set_aec_preset(AudioPipeline4Ch* p, AecPreset preset,
                                      float ramp_ms);

/* The post-beam suppressor's far-active split floor as a linear power ratio:
 * `live` is what it applies now (it differs from `target` only while a ramp
 * walks), `target` the reset-surviving configured value; either pointer may
 * be NULL. The one strength quantity worth logging from a board. Returns 0, or
 * -1 on NULL or an instance without a suppressor. */
int audio_pipeline_4ch_post_split_floor(const AudioPipeline4Ch* p,
                                        float* live, float* target);
int audio_pipeline_4ch_set_nr_mode(AudioPipeline4Ch* p, AnrMode mode);
void audio_pipeline_4ch_destroy(AudioPipeline4Ch* p);

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int audio_pipeline_4ch_hop_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_frame_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_fft_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_n_freqs(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_sample_rate(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_doa_sample_rate(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_doa_frame_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_doa_hop_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_doa_fft_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_bf_sample_rate(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_bf_frame_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_bf_hop_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_bf_fft_size(const AudioPipeline4Ch* p);

/*
 * The adapt_interval BF actually configured (post fixed-notebook-mode
 * forcing) and the RLS forgetting factor (lambda) it was retimed for, given
 * that same interval. Exposed read-only so tests can confirm the two were
 * derived from one shared effective-interval value instead of silently
 * diverging (see bf.h's bf_effective_adapt_interval()).
 */
int audio_pipeline_4ch_bf_effective_adapt_interval(
    const AudioPipeline4Ch* p);
float audio_pipeline_4ch_bf_lambda(const AudioPipeline4Ch* p);

int audio_pipeline_4ch_matched_filter_count(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_linear_aec_count(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_nr_count(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_post_res_count(const AudioPipeline4Ch* p);
const char* audio_pipeline_4ch_spatial_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_4CH_H */
