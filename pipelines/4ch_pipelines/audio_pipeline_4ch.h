/**
 * audio_pipeline_4ch.h — complete four-channel spatial pipeline.
 *
 * Complete four-channel traditional speech-enhancement pipeline:
 *
 *   one shared delay matcher -> four linear AEC lanes
 *   -> SRP-PHAT DOA -> GSC -> one NR + one post-beam RES
 *
 * SRP-PHAT/GSC come from the reusable libraries under this project's
 * third_party directory.  This
 * wrapper owns their lifetime and preserves the
 * resource boundary of 4aec_nr_res.h. Naming and lifecycle mirror
 * audio_pipeline.h; this heap wrapper adds the spatial stage between the
 * core's process_pre() and process_post() calls.
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
    int gsc_enable;
    float gsc_lambda;
    float gsc_mu;
    int gsc_fixed_mode;
    float gsc_fixed_doa_rad;
    int gsc_fixed_align_notebook;
    int gsc_adapt_interval;

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
    int gsc_adaptive;
    int doa_analysis_frames;
} AudioPipeline4ChFrameInfo;

typedef struct AudioPipeline4Ch AudioPipeline4Ch;

/* ============================================================================
 * Config and lifecycle
 * ========================================================================== */

AudioPipeline4ChConfig audio_pipeline_4ch_default_config(int sample_rate);

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
 * the held target-speech state used by the DOA smoother and freezes the GSC
 * RLS update when nonzero.  frequency_mask may be NULL or n_freqs integers;
 * 1 selects a bin for SRP and freezes GSC adaptation in that bin.
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
int audio_pipeline_4ch_gsc_sample_rate(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_gsc_frame_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_gsc_hop_size(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_gsc_fft_size(const AudioPipeline4Ch* p);

/*
 * The adapt_interval GSC actually configured (post fixed-notebook-mode
 * forcing) and the RLS forgetting factor (lambda) it was retimed for, given
 * that same interval. Exposed read-only so tests can confirm the two were
 * derived from one shared effective-interval value instead of silently
 * diverging (see gsc.h's gsc_effective_adapt_interval()).
 */
int audio_pipeline_4ch_gsc_effective_adapt_interval(
    const AudioPipeline4Ch* p);
float audio_pipeline_4ch_gsc_lambda(const AudioPipeline4Ch* p);

int audio_pipeline_4ch_matched_filter_count(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_linear_aec_count(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_nr_count(const AudioPipeline4Ch* p);
int audio_pipeline_4ch_post_res_count(const AudioPipeline4Ch* p);
const char* audio_pipeline_4ch_spatial_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_4CH_H */
