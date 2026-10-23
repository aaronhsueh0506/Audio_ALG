/** Four-channel linear AEC plus beamformer unit: one shared far aligner -> four linear AEC
 * lanes -> SRP-PHAT DOA -> BF. SRP-PHAT and BF come from audiolib/doa and audiolib/bf; this
 * unit owns their lifetime. The residual-echo, noise-reduction, comfort-noise and synthesis
 * stages and any neural post-filter are separate libraries that the application initialises
 * and chains over the unit's output (samples/se_sample_chain.h shows the chain).
 * Per hop, one of two equivalent forms:
 *     audio_pipeline_4ch_process_front()   shared delay + four linear AEC lanes
 *     [mask / activity from the lane spectra]
 *     audio_pipeline_4ch_process_beam()    SRP-PHAT + BF -> beamformed spectrum and weights
 *     audio_pipeline_4ch_fuse()            optional: project near / R^2 / comfort noise into a
 *                                          SeSpecFrame for the frequency-domain stages
 * or audio_pipeline_4ch_process_with_activity() / audio_pipeline_4ch_process() for front plus
 * beam in one call. With core.enable_post = 0 the lanes keep only the linear context and
 * fuse() is unavailable (the profile of a neural post-filter that reads the beam spectrum).
 * Lifecycle follows 4aec_nr_res.h; both construction paths give identical output for the same
 * inputs, config and backend.
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
     * own a VAD call audio_pipeline_4ch_process_with_activity() or the split calls. */
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
/* Carve layout version of this unit; bumped on any change of pool size, carve order or
 * descriptor content, including a control-block-only change of the composed core
 * (build_flags_hash folds in only the core's carve-token hash). Query a fresh descriptor
 * for every build; never persist or synthesize one. */
#define AUDIO_PIPELINE_4CH_LAYOUT_VERSION 18u

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
 * this unit's bf_spectrum/bf_weights scratch. */
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

/** The beamformed result of one hop. Both pointers alias unit memory and stay valid until the
 * next process_front(), process_with_activity(), process(), reset() or destroy(). */
typedef struct AudioPipeline4ChBeam {
    const Complex* spectrum;   /* beamformed error E, n_freqs bins                              */
    const Complex* weights;    /* effective weights, channel-major [4][n_freqs], from the same
                                  bf_process_with_weights() call as `spectrum`                  */
} AudioPipeline4ChBeam;

/** Starts a hop: the shared aligner and the four linear AEC lanes. microphones_interleaved is
 * [hop][4], far_reference is [hop]. On OK, *out is the core's borrowed front view of this hop
 * (delay state, lane spectra and hops, aligned reference). Always legal as the first call of a
 * hop, after any earlier state; it supersedes a pending front. Returns FOUR_AEC_NR_RES_OK;
 * INVALID_ARGUMENT for a NULL argument or a non-finite sample (no adaptive state changed);
 * DSP_ERROR when a lane refused the hop or the front disagrees with the unit's grid (the unit
 * has been reset in the second case, the core in the first). After any refusal past the NULL
 * checks no front or beam is pending: process_beam() and fuse() return SEQUENCE_ERROR. */
int audio_pipeline_4ch_process_front(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    FourAecFront* out);

/** Completes a hop begun by process_front(): SRP-PHAT and BF over `front`'s lane spectra.
 * vad_raw controls whether SRP-PHAT gets a fresh observation; vad_out is the held target-speech
 * state used by the DOA smoother, and freezes the BF RLS update when nonzero. frequency_mask may
 * be NULL or n_freqs integers; 1 selects a bin for SRP and freezes BF adaptation there. Fills
 * *beam (required) and, when non-NULL, *info. Returns OK; INVALID_ARGUMENT for a NULL argument,
 * a non-boolean flag or a front of another grid (nothing changed); SEQUENCE_ERROR without a
 * pending front, or when this hop's beam has already run. */
int audio_pipeline_4ch_process_beam(
    AudioPipeline4Ch* p,
    const FourAecFront* front,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    AudioPipeline4ChBeam* beam,
    AudioPipeline4ChFrameInfo* info);

/** process_front() plus process_beam() with activity from the product VAD. */
int audio_pipeline_4ch_process_with_activity(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    AudioPipeline4ChBeam* beam,
    AudioPipeline4ChFrameInfo* info);

/** process_with_activity() using the unit's energy-VAD fallback. */
int audio_pipeline_4ch_process(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    AudioPipeline4ChBeam* beam,
    AudioPipeline4ChFrameInfo* info);

/** Independent step after process_beam() (or process_with_activity()/process()): projects the
 * hop with the unit's own weights and beamformed spectrum into *frame (four_aec_fuse(); frame
 * pointers are valid until the next process_front() or reset). Requires core.enable_post.
 * Returns OK; INVALID_ARGUMENT for a NULL argument or enable_post = 0 (nothing changed);
 * SEQUENCE_ERROR without a beam of this hop (nothing changed). Any other refusal resets the
 * whole unit (core, SRP, BF) and is returned: the caller resets its own post stages. */
int audio_pipeline_4ch_fuse(AudioPipeline4Ch* p, SeSpecFrame* frame);

/** Resets the core, SRP-PHAT, BF and the fallback VAD. The caller resets its own post stages. */
void audio_pipeline_4ch_reset(AudioPipeline4Ch* p);

void audio_pipeline_4ch_destroy(AudioPipeline4Ch* p);

/** The composed core, read-only, for diagnostics. NULL for a NULL or destroyed handle. */
const FourAecNrRes* audio_pipeline_4ch_get_core(const AudioPipeline4Ch* p);

/** Stage timings (microseconds) of the core's most recent hop. A NULL/destroyed handle zeroes
 * *out; `out` must not be NULL. */
void audio_pipeline_4ch_last_core_timing(const AudioPipeline4Ch* p,
                                         FourAecNrResLastTiming* out);

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
const char* audio_pipeline_4ch_spatial_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_4CH_H */
