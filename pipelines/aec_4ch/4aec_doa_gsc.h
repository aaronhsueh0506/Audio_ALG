/**
 * 4aec_doa_gsc.h — complete four-channel spatial pipeline.
 *
 * Complete four-channel traditional speech-enhancement pipeline:
 *
 *   one shared delay matcher -> four linear AEC lanes
 *   -> SRP-PHAT DOA -> GSC -> one NR + one post-beam RES
 *
 * SRP-PHAT/GSC are the externally supplied implementations under
 * SE/third_party.  This wrapper owns their lifetime and preserves the
 * resource boundary of 4aec_nr_res.h. Naming and lifecycle mirror
 * audio_pipeline.h; this heap wrapper adds the spatial stage between the
 * core's process_pre() and process_post() calls.
 */
#ifndef FOUR_AEC_DOA_GSC_H
#define FOUR_AEC_DOA_GSC_H

#include "4aec_nr_res.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FourAecArrayGeometry {
    FOUR_AEC_ARRAY_UCA = 0,
    FOUR_AEC_ARRAY_ULA = 1,
    FOUR_AEC_ARRAY_CUSTOM = 2
} FourAecArrayGeometry;

typedef struct FourAecDoaGscConfig {
    FourAecNrResConfig core;

    FourAecArrayGeometry geometry;
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
    /*
     * 48 kHz main-grid only. When enabled, SRP-PHAT receives a streaming
     * 48->16 kHz resample and runs on 512/256. GSC remains on the selected
     * main grid so its weights still align with AEC/NR/RES frequency bins.
     */
    int doa_downsample_enable;

    int gsc_enable;
    float gsc_lambda;
    float gsc_mu;
    int gsc_fixed_mode;
    float gsc_fixed_doa_rad;
    int gsc_fixed_align_notebook;
    int gsc_adapt_interval;

    /*
     * Fallback activity detector used only by four_aec_doa_gsc_process().
     * Production integrations that own a VAD should call
     * four_aec_doa_gsc_process_with_activity() instead.
     */
    float auto_vad_threshold_dbfs;
    float auto_vad_snr_ratio;
    int auto_vad_hangover_frames;
} FourAecDoaGscConfig;

typedef struct FourAecDoaGscFrameInfo {
    uint64_t frame_index;
    FourAecNrResDelayState delay;
    float doa_raw_rad;
    float doa_smooth_rad;
    float doa_used_rad;
    int vad_raw;
    int vad_out;
    int gsc_adaptive;
    int doa_analysis_frames;
} FourAecDoaGscFrameInfo;

typedef struct FourAecDoaGsc FourAecDoaGsc;

/* ============================================================================
 * Config and lifecycle
 * ========================================================================== */

FourAecDoaGscConfig four_aec_doa_gsc_default_config(int sample_rate);

/* Compatibility wrapper; new code should use default_config(). */
void four_aec_doa_gsc_config_defaults(FourAecDoaGscConfig* cfg,
                                      int sample_rate);

FourAecDoaGsc* four_aec_doa_gsc_create(
    const FourAecDoaGscConfig* cfg);

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

/**
 * Complete one hop using the wrapper's conservative energy-VAD fallback.
 */
int four_aec_doa_gsc_process(
    FourAecDoaGsc* p,
    const float* microphones_interleaved,
    const float* far_reference,
    float* output,
    FourAecDoaGscFrameInfo* info);

/**
 * Complete one hop with activity supplied by the product VAD.
 *
 * vad_raw controls whether SRP-PHAT gets a fresh observation. vad_out is
 * the held target-speech state used by the DOA smoother and freezes the GSC
 * RLS update when nonzero.  frequency_mask may be NULL or n_freqs integers;
 * 1 selects a bin for SRP and freezes GSC adaptation in that bin.
 */
int four_aec_doa_gsc_process_with_activity(
    FourAecDoaGsc* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    float* output,
    FourAecDoaGscFrameInfo* info);

void four_aec_doa_gsc_reset(FourAecDoaGsc* p);
void four_aec_doa_gsc_destroy(FourAecDoaGsc* p);

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int four_aec_doa_gsc_hop_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_frame_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_fft_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_n_freqs(const FourAecDoaGsc* p);
int four_aec_doa_gsc_sample_rate(const FourAecDoaGsc* p);
int four_aec_doa_gsc_doa_sample_rate(const FourAecDoaGsc* p);
int four_aec_doa_gsc_doa_frame_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_doa_hop_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_doa_fft_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_gsc_sample_rate(const FourAecDoaGsc* p);
int four_aec_doa_gsc_gsc_frame_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_gsc_hop_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_gsc_fft_size(const FourAecDoaGsc* p);
int four_aec_doa_gsc_matched_filter_count(const FourAecDoaGsc* p);
int four_aec_doa_gsc_linear_aec_count(const FourAecDoaGsc* p);
int four_aec_doa_gsc_nr_count(const FourAecDoaGsc* p);
int four_aec_doa_gsc_post_res_count(const FourAecDoaGsc* p);
const char* four_aec_doa_gsc_spatial_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* FOUR_AEC_DOA_GSC_H */
