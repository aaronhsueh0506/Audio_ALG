/** Sample-level beamformer helper over a four-microphone array instance (aec.h): owns audiolib's SRP-PHAT
 * and BF and a fallback energy VAD, and runs the per-hop glue the four-channel examples use,
 * doa_step() + bf_process_with_weights() between aec_process_linear() / aec_get_array() and
 * aec_process_residual(). An application with its own beamformer calls those functions directly instead.
 *
 * Per hop, one of two equivalent forms:
 *     array_sample_beam_process_front()   -> [caller VAD/mask] -> array_sample_beam_process_beam()
 *     array_sample_beam_process_with_activity() / array_sample_beam_process() (fallback VAD)
 * then, optionally, array_sample_beam_fuse() for the frequency-domain post stages.
 *
 * Init: build the array first (aec_get_mem_requirements() + aec_init_ex() on an AecConfig from
 * array_sample_default_config()), then size and place this helper; the array must outlive it and is
 * driven only through this helper while it is attached:
 *     ArraySampleBeamParams bp = array_sample_beam_default_params();
 *     size_t bytes;  array_sample_beam_get_mem_size(&cfg, &bp, &bytes);
 *     ArraySampleBeam* b = array_sample_beam_init(mem, bytes, &cfg, &bp, aec);
 *     ...  array_sample_beam_destroy(b);  then aec_destroy(aec). */
#ifndef ARRAY_SAMPLE_BEAM_H
#define ARRAY_SAMPLE_BEAM_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"
#include "srp.h"
#include "bf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bounds the RLS cadence so hop * interval stays inside int on every grid. */
#define ARRAY_SAMPLE_BEAM_MAX_ADAPT_INTERVAL (1 << 20)

typedef enum ArraySampleGeometry {
    ARRAY_SAMPLE_GEOMETRY_UCA = 0,
    ARRAY_SAMPLE_GEOMETRY_ULA = 1,
    ARRAY_SAMPLE_GEOMETRY_CUSTOM = 2
} ArraySampleGeometry;

/** Array geometry, SRP-PHAT/BF tunables and the fallback VAD. */
typedef struct ArraySampleBeamParams {
    ArraySampleGeometry geometry;
    float uca_radius_m;
    float ula_spacing_m;
    float microphone_x_m[AEC_ARRAY_MICS];   /* CUSTOM only */
    float microphone_y_m[AEC_ARRAY_MICS];
    int num_angles;               /* [4, 3600]                                               */
    float speed_of_sound_m_s;
    float doa_low_freq_hz;
    float doa_high_freq_hz;       /* 0 = min(7000, Nyquist)                                  */
    int doa_enable_smoothing;
    int doa_switch_consecutive;
    float doa_angle_tolerance_rad;
    int doa_update_interval;
    int bf_enable;
    float bf_lambda;              /* RLS retention per 10 ms; retimed to the grid             */
    float bf_mu;
    int bf_fixed_mode;
    float bf_fixed_doa_rad;
    int bf_fixed_align_notebook;
    int bf_adapt_interval;        /* RLS update every N hops, [1, ARRAY_SAMPLE_BEAM_MAX_ADAPT_INTERVAL] */
    /* read only by array_sample_beam_process() */
    float auto_vad_threshold_dbfs;
    float auto_vad_snr_ratio;
    int auto_vad_hangover_frames;
} ArraySampleBeamParams;

/** SRP/BF inputs on the array's grid (bf.lambda retimed to the hop and the RLS cadence BF will
 * run, DOA band clamped to Nyquist), for an application that builds SRP/BF itself. Build the
 * geometry as {n_channels, mic_x, mic_y} (the struct holds no pointer to itself). */
typedef struct ArraySampleSpatialConfig {
    SRP_Config srp;
    BF_Config bf;
    int n_channels;
    int n_freqs;
    int num_angles;
    float mic_x[AEC_ARRAY_MICS];
    float mic_y[AEC_ARRAY_MICS];
} ArraySampleSpatialConfig;

/** The beamformed result of one hop; aliases helper memory until the next front/process call,
 * reset or destroy. */
typedef struct ArraySampleBeamResult {
    const Complex* spectrum;      /* beamformed error E, n_freqs bins                        */
    const Complex* weights;       /* effective weights, channel-major [4][n_freqs], from the
                                     same bf_process_with_weights() call as `spectrum`       */
} ArraySampleBeamResult;

typedef struct ArraySampleBeamInfo {
    uint64_t frame_index;         /* beams since init or reset                               */
    AecDelayState delay;
    float doa_raw_rad;
    float doa_smooth_rad;
    float doa_used_rad;
    int vad_raw;
    int vad_out;
    int bf_adaptive;
} ArraySampleBeamInfo;

typedef struct ArraySampleBeam ArraySampleBeam;

/** The samples' array config: balanced level, MATCHED with a full bank, AEC_RESIDUAL_FULL, AEC_ARRAY_MICS
 * microphones, on the rate's default grid. Set `residual` (STATE: lanes only), the delay fields, `fft_size`
 * or the level fields afterwards. */
static inline void array_sample_default_config(AecConfig* cfg, int sample_rate) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, sample_rate);
    cfg->n_mics = AEC_ARRAY_MICS;
    cfg->residual = AEC_RESIDUAL_FULL;
}

ArraySampleBeamParams array_sample_beam_default_params(void);

/** 0, or -1 for an invalid array config or parameter set. */
int array_sample_spatial_config(const AecConfig* cfg, const ArraySampleBeamParams* bp,
                               ArraySampleSpatialConfig* out);

/** Bytes of the helper's 16-byte-aligned pool: control block, SRP, BF, beam spectrum and
 * weights. 0 on success, -1 for an invalid config. */
int array_sample_beam_get_mem_size(const AecConfig* cfg, const ArraySampleBeamParams* bp,
                                  size_t* bytes);

/** aec: the array built from the same cfg; borrowed. NULL on any refusal. */
ArraySampleBeam* array_sample_beam_init(void* mem, size_t bytes, const AecConfig* cfg,
                                      const ArraySampleBeamParams* bp, Aec* aec);

/** Tears SRP/BF down (BF first: it borrows SRP's steering table); the array stays. */
void array_sample_beam_destroy(ArraySampleBeam* b);

/** Resets the array (aec_reset()), SRP, BF, the fallback VAD and frame_index. */
void array_sample_beam_reset(ArraySampleBeam* b);

/** aec_process_linear() + aec_get_array(); any refusal drops the pending beam. AEC_ERR_DSP leaves SRP, BF, the
 * VAD and frame_index as they were (only the array was reset). */
int array_sample_beam_process_front(ArraySampleBeam* b, const float* microphones_interleaved,
                                   const float* far_reference, AecArrayView* out);

/** SRP-PHAT and BF over this hop's front. vad_raw (0/1): SRP takes a fresh observation;
 * vad_out (0/1): held target-speech state for the DOA smoother, nonzero freezes the BF RLS
 * update; frequency_mask: NULL or n_freqs ints, 1 selects a bin for SRP and freezes BF
 * adaptation there; info optional. Returns AEC_OK; AEC_ERR_ARG for a NULL argument,
 * a non-boolean flag or a front of another grid; AEC_ERR_SEQUENCE without a pending front or
 * when this hop's beam already ran (both: nothing changed). */
int array_sample_beam_process_beam(ArraySampleBeam* b, const AecArrayView* front, int vad_raw,
                                  int vad_out, const int* frequency_mask,
                                  ArraySampleBeamResult* beam, ArraySampleBeamInfo* info);

int array_sample_beam_process_with_activity(ArraySampleBeam* b, const float* microphones_interleaved,
                                           const float* far_reference, int vad_raw, int vad_out,
                                           const int* frequency_mask, ArraySampleBeamResult* beam,
                                           ArraySampleBeamInfo* info);

/** With the fallback energy VAD (vad_raw = vad_out = its decision, no mask); a non-finite
 * far_reference is refused before the VAD moves. */
int array_sample_beam_process(ArraySampleBeam* b, const float* microphones_interleaved,
                             const float* far_reference, ArraySampleBeamResult* beam,
                             ArraySampleBeamInfo* info);

/** aec_process_residual() with this hop's beam, then aec_get_residual() into *frame. AEC_ERR_ARG for NULL,
 * AEC_ERR_SEQUENCE without a beam of this hop and AEC_ERR_MODE for a lanes-only array (AEC_RESIDUAL_STATE)
 * change nothing; any other refusal resets everything (array_sample_beam_reset()) and is returned. */
int array_sample_beam_fuse(ArraySampleBeam* b, AecResidualView* frame);

/* The adapt_interval BF runs at (after fixed-notebook forcing) and the lambda retimed for it;
 * -1 / NAN for a NULL or destroyed helper. */
int array_sample_beam_bf_effective_adapt_interval(const ArraySampleBeam* b);
float array_sample_beam_bf_lambda(const ArraySampleBeam* b);
const char* array_sample_beam_spatial_backend(void);

#ifdef __cplusplus
}
#endif
#endif /* ARRAY_SAMPLE_BEAM_H */
