/** Sample-level beamformer helper over an mcab flow (not part of the mcab module): owns
 * audiolib's SRP-PHAT and BF and a fallback energy VAD, and runs the same per-hop glue the
 * four-channel examples use, doa_step() + bf_process_with_weights() between mcab_process_front()
 * and mcab_fuse(). An application with its own beamformer calls those two mcab functions
 * directly instead.
 *
 * Per hop, one of two equivalent forms:
 *     mcab_sample_beam_process_front()   -> [caller VAD/mask] -> mcab_sample_beam_process_beam()
 *     mcab_sample_beam_process_with_activity() / mcab_sample_beam_process() (fallback VAD)
 * then, optionally, mcab_sample_beam_fuse() for the frequency-domain post stages.
 *
 * Init: build the flow first (mcab.h), then size and place this helper; the flow must outlive
 * it and is driven only through this helper while it is attached:
 *     McabSampleBeamParams bp = mcab_sample_beam_default_params();
 *     size_t bytes;  mcab_sample_beam_get_mem_size(&cfg, &bp, &bytes);
 *     McabSampleBeam* b = mcab_sample_beam_init(mem, bytes, &cfg, &bp, flow);
 *     ...  mcab_sample_beam_destroy(b);  then mcab_destroy(flow) and the lanes. */
#ifndef MCAB_SAMPLE_BEAM_H
#define MCAB_SAMPLE_BEAM_H

#include <stddef.h>
#include <stdint.h>

#include "mcab.h"
#include "srp.h"
#include "bf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bounds the RLS cadence so hop * interval stays inside int on every grid. */
#define MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL (1 << 20)

typedef enum McabSampleGeometry {
    MCAB_SAMPLE_GEOMETRY_UCA = 0,
    MCAB_SAMPLE_GEOMETRY_ULA = 1,
    MCAB_SAMPLE_GEOMETRY_CUSTOM = 2
} McabSampleGeometry;

/** Array geometry, SRP-PHAT/BF tunables and the fallback VAD. */
typedef struct McabSampleBeamParams {
    McabSampleGeometry geometry;
    float uca_radius_m;
    float ula_spacing_m;
    float microphone_x_m[MCAB_CHANNELS];   /* CUSTOM only */
    float microphone_y_m[MCAB_CHANNELS];
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
    int bf_adapt_interval;        /* RLS update every N hops, [1, MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL] */
    /* read only by mcab_sample_beam_process() */
    float auto_vad_threshold_dbfs;
    float auto_vad_snr_ratio;
    int auto_vad_hangover_frames;
} McabSampleBeamParams;

/** SRP/BF inputs on the flow's grid (bf.lambda retimed to the hop and the RLS cadence BF will
 * run, DOA band clamped to Nyquist), for an application that builds SRP/BF itself. Build the
 * geometry as {n_channels, mic_x, mic_y} (the struct holds no pointer to itself). */
typedef struct McabSampleSpatialConfig {
    SRP_Config srp;
    BF_Config bf;
    int n_channels;
    int n_freqs;
    int num_angles;
    float mic_x[MCAB_CHANNELS];
    float mic_y[MCAB_CHANNELS];
} McabSampleSpatialConfig;

/** The beamformed result of one hop; aliases helper memory until the next front/process call,
 * reset or destroy. */
typedef struct McabSampleBeamResult {
    const Complex* spectrum;      /* beamformed error E, n_freqs bins                        */
    const Complex* weights;       /* effective weights, channel-major [4][n_freqs], from the
                                     same bf_process_with_weights() call as `spectrum`       */
} McabSampleBeamResult;

typedef struct McabSampleBeamInfo {
    uint64_t frame_index;         /* beams since init or reset                               */
    McabDelayState delay;
    float doa_raw_rad;
    float doa_smooth_rad;
    float doa_used_rad;
    int vad_raw;
    int vad_out;
    int bf_adaptive;
    int doa_analysis_frames;
} McabSampleBeamInfo;

typedef struct McabSampleBeam McabSampleBeam;

McabSampleBeamParams mcab_sample_beam_default_params(void);

/** 0, or -1 for an invalid flow config or parameter set. */
int mcab_sample_spatial_config(const McabConfig* cfg, const McabSampleBeamParams* bp,
                               McabSampleSpatialConfig* out);

/** Bytes of the helper's 16-byte-aligned pool: control block, SRP, BF, beam spectrum and
 * weights. 0 on success, -1 for an invalid config. */
int mcab_sample_beam_get_mem_size(const McabConfig* cfg, const McabSampleBeamParams* bp,
                                  size_t* bytes);

/** flow: built from the same cfg; borrowed. NULL on any refusal. */
McabSampleBeam* mcab_sample_beam_init(void* mem, size_t bytes, const McabConfig* cfg,
                                      const McabSampleBeamParams* bp, Mcab* flow);

/** Tears SRP/BF down (BF first: it borrows SRP's steering table); the flow stays. */
void mcab_sample_beam_destroy(McabSampleBeam* b);

/** Resets the flow (mcab_reset()), SRP, BF, the fallback VAD and frame_index. */
void mcab_sample_beam_reset(McabSampleBeam* b);

/** mcab_process_front(); any refusal drops the pending beam. A DSP_ERROR leaves SRP, BF, the
 * VAD and frame_index as they were (only the flow was reset). */
int mcab_sample_beam_process_front(McabSampleBeam* b, const float* microphones_interleaved,
                                   const float* far_reference, McabFront* out);

/** SRP-PHAT and BF over this hop's front. vad_raw (0/1): SRP takes a fresh observation;
 * vad_out (0/1): held target-speech state for the DOA smoother, nonzero freezes the BF RLS
 * update; frequency_mask: NULL or n_freqs ints, 1 selects a bin for SRP and freezes BF
 * adaptation there; info optional. Returns MCAB_OK; MCAB_INVALID_ARGUMENT for a NULL argument,
 * a non-boolean flag or a front of another grid; MCAB_SEQUENCE_ERROR without a pending front or
 * when this hop's beam already ran (both: nothing changed). */
int mcab_sample_beam_process_beam(McabSampleBeam* b, const McabFront* front, int vad_raw,
                                  int vad_out, const int* frequency_mask,
                                  McabSampleBeamResult* beam, McabSampleBeamInfo* info);

int mcab_sample_beam_process_with_activity(McabSampleBeam* b, const float* microphones_interleaved,
                                           const float* far_reference, int vad_raw, int vad_out,
                                           const int* frequency_mask, McabSampleBeamResult* beam,
                                           McabSampleBeamInfo* info);

/** With the fallback energy VAD (vad_raw = vad_out = its decision, no mask); a non-finite
 * far_reference is refused before the VAD moves. */
int mcab_sample_beam_process(McabSampleBeam* b, const float* microphones_interleaved,
                             const float* far_reference, McabSampleBeamResult* beam,
                             McabSampleBeamInfo* info);

/** mcab_fuse() with this hop's beam. MCAB_INVALID_ARGUMENT for NULL or enable_post = 0 and
 * MCAB_SEQUENCE_ERROR without a beam of this hop (nothing changed); any other refusal resets
 * everything (mcab_sample_beam_reset()) and is returned. */
int mcab_sample_beam_fuse(McabSampleBeam* b, McabFrame* frame);

/* The adapt_interval BF runs at (after fixed-notebook forcing) and the lambda retimed for it;
 * -1 / NAN for a NULL or destroyed helper. */
int mcab_sample_beam_bf_effective_adapt_interval(const McabSampleBeam* b);
float mcab_sample_beam_bf_lambda(const McabSampleBeam* b);
const char* mcab_sample_beam_spatial_backend(void);

#ifdef __cplusplus
}
#endif
#endif /* MCAB_SAMPLE_BEAM_H */
