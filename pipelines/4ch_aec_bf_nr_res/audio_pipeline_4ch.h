/** Complete four-channel spatial speech-enhancement pipeline: one shared far aligner ->
 * four linear AEC lanes -> SRP-PHAT DOA -> BF -> one NR + one post-beam RES. SRP-PHAT and
 * BF come from audiolib/doa and audiolib/bf; this wrapper owns their lifetime. Between
 * the core's four_aec_front_process() and four_aec_fuse() it runs the spatial stage, then
 * chains the post-beam stages itself (RES gain, NR gain, min, apply, comfort noise,
 * synthesis). Lifecycle follows 4aec_nr_res.h; both construction paths give identical
 * output for the same inputs, config and backend.
 * Caller-owned pool (board/static path):
 *     AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
 *     AudioPipeline4ChMemReq req;
 *     audio_pipeline_4ch_get_mem_requirements(&cfg, &req);
 *     void* pool = platform_alloc(req.bytes, req.alignment);
 *     AudioPipeline4Ch* p = audio_pipeline_4ch_init_ex(pool, req.bytes, &cfg, &req);
 *     ...
 *     audio_pipeline_4ch_destroy(p);  // never releases caller-owned pool
 *     platform_free(pool);
 * Heap convenience (desktop/test path):
 *     AudioPipeline4Ch* p = audio_pipeline_4ch_create(&cfg);
 *     ...
 *     audio_pipeline_4ch_destroy(p);  // releases create()'s one pool */
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

    /* Fallback activity detector used only by audio_pipeline_4ch_process(); integrations that
     * own a VAD call audio_pipeline_4ch_process_with_activity(). */
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

#define AUDIO_PIPELINE_4CH_DESCRIPTOR_VERSION 1u
/* Carve layout version of this wrapper; bumped on any change of pool size, carve order or
 * descriptor content, including a control-block-only change of the composed core
 * (build_flags_hash folds in only the core's carve-token hash). Query a fresh descriptor
 * for every build; never persist or synthesize one. */
#define AUDIO_PIPELINE_4CH_LAYOUT_VERSION 17u

/** Fixed 32-byte descriptor with the fields of FourAecNrResMemReq (4aec_nr_res.h). Its
 * build_flags_hash folds in the core's, so a core-layer layout bump invalidates persisted
 * descriptors here. */
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

AudioPipeline4ChConfig audio_pipeline_4ch_default_config(int sample_rate);

/** Queries the 16-byte-aligned caller pool required by cfg. No allocation; returns 0 on
 * success. Composes the core's requirements with srp_get_mem_size()/bf_get_mem_size() and
 * this wrapper's bf_spectrum/bf_weights scratch. */
int audio_pipeline_4ch_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChMemReq* out);

/** Places the pipeline (core + SRP-PHAT + BF + scratch) in caller-owned memory that
 * satisfies the queried alignment and size. No heap allocation; a dirty pool is accepted. */
AudioPipeline4Ch* audio_pipeline_4ch_init(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg);

/** init plus a stale-descriptor gate with the 8-point contract of four_aec_nr_res_init_ex()
 * (versions, backend, hash and alignment must match; reserved 0; expected->bytes and the
 * pool >= the current requirement). NULL expected is the same as init(). */
AudioPipeline4Ch* audio_pipeline_4ch_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChMemReq* expected);

/** Heap convenience over get_mem_requirements() + init(). NULL for an invalid config or
 * allocation failure. */
AudioPipeline4Ch* audio_pipeline_4ch_create(
    const AudioPipeline4ChConfig* cfg);

/** Completes one hop using the wrapper's energy-VAD fallback. */
int audio_pipeline_4ch_process(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    float* output,
    AudioPipeline4ChFrameInfo* info);

/** Completes one hop with activity from the product VAD. vad_raw controls whether SRP-PHAT
 * gets a fresh observation; vad_out is the held target-speech state used by the DOA
 * smoother, and freezes the BF RLS update when nonzero. frequency_mask may be NULL or
 * n_freqs integers; 1 selects a bin for SRP and freezes BF adaptation there. */
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

/* Runtime strength control. The lanes never reach their own suppression-gain path, so strength lives in the single post-beam suppressor (AecRes) and the single denoiser. set_aec_preset retargets the suppressor's two gain floors and split-floor gate: the far-active floor ramps (ramp_ms 0 applies on the next hop at the floor a fresh instance would hold, a positive value walks there linearly in dB); the DT floor (-20 dB mild/balanced, -30 dB aggressive) and the gate jump immediately. Switching to/from AEC_PRESET_NO_FLOOR is allowed. Requires core.enable_res. set_nr_mode recomposes this pipeline's NR configuration (canonical preset plus its overrides) and calls anr_reconfigure(); noise floor and smoothing carry on. Requires core.enable_nr. Call between hops. Returns 0, or -1 with nothing written. */
int audio_pipeline_4ch_set_aec_preset(AudioPipeline4Ch* p, AecPreset preset,
                                      float ramp_ms);

/* The suppressor's far-active split floor as a linear power ratio: `live` is what it
 * applies now (differs from `target` only during a ramp), `target` the reset-surviving
 * configured value; either may be NULL. Returns 0, or -1 on NULL or an instance without a
 * suppressor. */
int audio_pipeline_4ch_post_split_floor(const AudioPipeline4Ch* p,
                                        float* live, float* target);
int audio_pipeline_4ch_set_nr_mode(AudioPipeline4Ch* p, AnrMode mode);
void audio_pipeline_4ch_destroy(AudioPipeline4Ch* p);

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

/* The adapt_interval BF actually configured (after fixed-notebook-mode forcing) and the
 * RLS forgetting factor (lambda) retimed for it (bf.h bf_effective_adapt_interval()). */
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
