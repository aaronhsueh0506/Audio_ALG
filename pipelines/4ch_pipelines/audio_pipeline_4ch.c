/**
 * audio_pipeline_4ch.c — complete four-channel spatial pipeline.
 *
 * This is the deployable orchestrator around 4aec_nr_res.c:
 *
 *   4AEC process_pre -> SRP-PHAT DOA -> GSC effective weights
 *                    -> 4AEC process_post -> mono NR/RES
 *
 * The file follows audio_pipeline.c's order: instance, config validation,
 * default/create, processing, reset/destroy, and accessors.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "audio_pipeline_4ch.h"
#include "4aec_nr_res_internal.h"
#include "gsc.h"
#include "srp.h"
#include "steering.h"
#include "spatial_simd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PI_F ((float)M_PI)

#define DEFAULT_RADIUS_M  0.035f
#define DEFAULT_SPACING_M 0.035f
#define DEFAULT_ANGLES    72

/* ============================================================================
 * Instance
 * ========================================================================== */

struct AudioPipeline4Ch {
    AudioPipeline4ChConfig cfg;
    FourAecNrRes* core;
    SRP* srp;
    Complex*** gsc_steering;
    GSC* gsc;

    int hop_size;
    int fft_size;
    int n_freqs;
    Complex* spatial_input;
    Complex* spatial_channels[FOUR_AEC_NR_RES_CHANNELS];
    Complex* gsc_spectrum;
    Complex* gsc_weights;

    float noise_power;
    int vad_hangover;
    int vad_hangover_frames;      /* live-computed, was raw cfg.auto_vad_hangover_frames */
    float vad_speech_noise_keep;  /* live-computed, was raw literal 0.999f */
    float vad_speech_new_weight;  /* live-computed, was raw literal 0.001f */
    float vad_silence_noise_keep; /* live-computed, was raw literal 0.95f */
    float vad_silence_new_weight; /* live-computed, was raw literal 0.05f */
    uint64_t frame_index;
};

/* ============================================================================
 * Config validation
 * ========================================================================== */

static int is_bool(int value) {
    return value == 0 || value == 1;
}

static int validate_config(const AudioPipeline4ChConfig* cfg) {
    float nyquist;
    if (!cfg) return 0;
    if (cfg->core.sample_rate != 16000 &&
        cfg->core.sample_rate != 48000) return 0;
    nyquist = 0.5f * (float)cfg->core.sample_rate;
    if (cfg->geometry != AUDIO_PIPELINE_4CH_GEOMETRY_UCA &&
        cfg->geometry != AUDIO_PIPELINE_4CH_GEOMETRY_ULA &&
        cfg->geometry != AUDIO_PIPELINE_4CH_GEOMETRY_CUSTOM) return 0;
    if (cfg->num_angles < 4 || cfg->num_angles > 3600) return 0;
    if (!isfinite(cfg->speed_of_sound_m_s) ||
        cfg->speed_of_sound_m_s <= 0.0f) return 0;
    if (!isfinite(cfg->doa_low_freq_hz) ||
        cfg->doa_low_freq_hz < 0.0f ||
        cfg->doa_low_freq_hz >= nyquist) return 0;
    if (!isfinite(cfg->doa_high_freq_hz) ||
        cfg->doa_high_freq_hz < 0.0f ||
        cfg->doa_high_freq_hz > nyquist) return 0;
    if (cfg->doa_high_freq_hz > 0.0f &&
        cfg->doa_high_freq_hz < cfg->doa_low_freq_hz) return 0;
    if (!is_bool(cfg->doa_enable_smoothing) ||
        cfg->doa_switch_consecutive <= 0 ||
        !isfinite(cfg->doa_angle_tolerance_rad) ||
        cfg->doa_angle_tolerance_rad < 0.0f ||
        cfg->doa_update_interval <= 0) return 0;
    if (!is_bool(cfg->gsc_enable) ||
        !isfinite(cfg->gsc_lambda) || cfg->gsc_lambda <= 0.0f ||
        cfg->gsc_lambda > 1.0f ||
        !isfinite(cfg->gsc_mu) || cfg->gsc_mu < 0.0f ||
        !is_bool(cfg->gsc_fixed_mode) ||
        !isfinite(cfg->gsc_fixed_doa_rad) ||
        !is_bool(cfg->gsc_fixed_align_notebook) ||
        cfg->gsc_adapt_interval <= 0) return 0;
    if (!isfinite(cfg->auto_vad_threshold_dbfs) ||
        !isfinite(cfg->auto_vad_snr_ratio) ||
        cfg->auto_vad_snr_ratio < 1.0f ||
        cfg->auto_vad_hangover_frames < 0) return 0;
    if (cfg->geometry == AUDIO_PIPELINE_4CH_GEOMETRY_UCA &&
        (!isfinite(cfg->uca_radius_m) || cfg->uca_radius_m <= 0.0f))
        return 0;
    if (cfg->geometry == AUDIO_PIPELINE_4CH_GEOMETRY_ULA &&
        (!isfinite(cfg->ula_spacing_m) || cfg->ula_spacing_m <= 0.0f))
        return 0;
    if (cfg->geometry == AUDIO_PIPELINE_4CH_GEOMETRY_CUSTOM) {
        for (int m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m) {
            if (!isfinite(cfg->microphone_x_m[m]) ||
                !isfinite(cfg->microphone_y_m[m])) return 0;
        }
    }
    return 1;
}

/* ============================================================================
 * Public config and heap construction
 * ========================================================================== */

AudioPipeline4ChConfig audio_pipeline_4ch_default_config(int sample_rate) {
    AudioPipeline4ChConfig cfg;
    float radius = DEFAULT_RADIUS_M;
    memset(&cfg, 0, sizeof(cfg));
    cfg.core = four_aec_nr_res_default_config(sample_rate);
    cfg.geometry = AUDIO_PIPELINE_4CH_GEOMETRY_UCA;
    cfg.uca_radius_m = radius;
    cfg.ula_spacing_m = DEFAULT_SPACING_M;
    for (int m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m) {
        float phi = 2.0f * M_PI_F * (float)m /
                    (float)FOUR_AEC_NR_RES_CHANNELS;
        cfg.microphone_x_m[m] = radius * cosf(phi);
        cfg.microphone_y_m[m] = radius * sinf(phi);
    }
    cfg.num_angles = DEFAULT_ANGLES;
    cfg.speed_of_sound_m_s = 343.0f;
    cfg.doa_low_freq_hz = 300.0f;
    cfg.doa_high_freq_hz = 7000.0f;
    cfg.doa_enable_smoothing = 1;
    cfg.doa_switch_consecutive = 3;
    cfg.doa_angle_tolerance_rad = 10.0f * M_PI_F / 180.0f;
    cfg.doa_update_interval = 2;
    cfg.gsc_enable = 1;
    cfg.gsc_lambda = 0.995f;
    cfg.gsc_mu = 0.1f;
    cfg.gsc_fixed_mode = 0;
    cfg.gsc_fixed_doa_rad = 0.0f;
    cfg.gsc_fixed_align_notebook = 0;
    cfg.gsc_adapt_interval = 1;
    cfg.auto_vad_threshold_dbfs = -55.0f;
    cfg.auto_vad_snr_ratio = 3.0f;
    cfg.auto_vad_hangover_frames = 8;
    return cfg;
}

static ArrayGeometry* create_geometry(
    const AudioPipeline4ChConfig* cfg) {
    switch (cfg->geometry) {
        case AUDIO_PIPELINE_4CH_GEOMETRY_UCA:
            return array_geometry_create_uca(
                FOUR_AEC_NR_RES_CHANNELS, cfg->uca_radius_m);
        case AUDIO_PIPELINE_4CH_GEOMETRY_ULA:
            return array_geometry_create_ula(
                FOUR_AEC_NR_RES_CHANNELS, cfg->ula_spacing_m);
        case AUDIO_PIPELINE_4CH_GEOMETRY_CUSTOM:
            return array_geometry_create_custom(
                FOUR_AEC_NR_RES_CHANNELS,
                cfg->microphone_x_m, cfg->microphone_y_m);
        default:
            return NULL;
    }
}

AudioPipeline4Ch* audio_pipeline_4ch_create(
    const AudioPipeline4ChConfig* cfg) {
    AudioPipeline4Ch* p;
    ArrayGeometry* geometry;
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    int fft_size;
    size_t spectral_count;

    if (!validate_config(cfg)) return NULL;
    p = (AudioPipeline4Ch*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->cfg = *cfg;
    p->core = four_aec_nr_res_create(&cfg->core);
    if (!p->core) goto fail;
    p->hop_size = four_aec_nr_res_hop_size(p->core);
    p->n_freqs = four_aec_nr_res_n_freqs(p->core);
    fft_size = four_aec_nr_res_fft_size(p->core);
    p->fft_size = fft_size;

    geometry = create_geometry(cfg);
    if (!geometry) goto fail;
    memset(&srp_cfg, 0, sizeof(srp_cfg));
    srp_cfg.M = FOUR_AEC_NR_RES_CHANNELS;
    srp_cfg.F = p->n_freqs;
    srp_cfg.num_angles = cfg->num_angles;
    srp_cfg.sr = (float)cfg->core.sample_rate;
    srp_cfg.NFFT = (float)fft_size;
    srp_cfg.c = cfg->speed_of_sound_m_s;
    srp_cfg.low_freq = cfg->doa_low_freq_hz;
    srp_cfg.high_freq = cfg->doa_high_freq_hz > 0.0f
        ? fminf(cfg->doa_high_freq_hz,
                0.5f * (float)cfg->core.sample_rate)
        : fminf(7000.0f, 0.5f * (float)cfg->core.sample_rate);
    srp_cfg.enable_smoothing = cfg->doa_enable_smoothing;
    srp_cfg.switch_consec = cfg->doa_switch_consecutive;
    srp_cfg.angle_tol = cfg->doa_angle_tolerance_rad;
    srp_cfg.update_interval = cfg->doa_update_interval;
    p->srp = srp_create_from_geometry(&srp_cfg, geometry);
    p->gsc_steering = p->srp ? p->srp->a_array : NULL;
    array_geometry_destroy(geometry);
    if (!p->srp || !p->gsc_steering) goto fail;

    /* gsc_create() forces the RLS update cadence to 1 (every hop) whenever
     * fixed-notebook mode is requested (gsc_fixed_mode && gsc_fixed_align_
     * notebook), regardless of the caller's configured gsc_adapt_interval --
     * see gsc.c's gsc_effective_adapt_interval(). Derive that SAME effective
     * value here, once, and feed it into both the lambda retime scaling
     * below and gsc_cfg.adapt_interval, so the cadence lambda is calibrated
     * for and the cadence GSC actually runs at can never silently diverge
     * (previously lambda was scaled by the raw pre-forced gsc_adapt_interval
     * while gsc_create() silently forced the real cadence to 1). */
    int gsc_effective_interval = gsc_effective_adapt_interval(
        cfg->gsc_fixed_mode, cfg->gsc_fixed_align_notebook,
        cfg->gsc_adapt_interval);

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = cfg->gsc_enable;
    /* gsc_lambda is an RLS forgetting/retention factor tuned at a 10-ms
     * reference update (like NR's alpha_d/alpha_attack just above in the
     * mono/4ch NR config derivation) -- was forwarded as a raw literal
     * regardless of grid, so its real wall-clock forgetting time varied
     * with hop_size/sample_rate. Retimed via the same helper NR already
     * uses for this exact class of constant. mu is a step-size gain, not
     * a decay time-constant, so it is left as-is.
     *
     * The RLS update itself only actually applies once every
     * gsc_effective_interval hops (gsc.c gates the whole P/gain/weight-update
     * block on frame_idx % adapt_interval == 0; default adapt_interval=1,
     * i.e. every hop). Unlike AEC's ERLE 6-point cadence, the ORIGINAL
     * "10ms reference" tuning assumed one update per hop (interval=1) --
     * there is no matching batching on the authored side to cancel against
     * -- so when the effective interval > 1 the real wall-clock update
     * period is that many hops, and the retime call must scale hop_size by
     * it (no-op at the shipped default of 1). */
    gsc_cfg.lambda = mmse_lsa_retime_alpha(
        cfg->gsc_lambda, cfg->core.sample_rate,
        p->hop_size * gsc_effective_interval);
    gsc_cfg.mu = cfg->gsc_mu;
    gsc_cfg.enable_fix_mode = cfg->gsc_fixed_mode;
    gsc_cfg.fixed_doa_rad = cfg->gsc_fixed_doa_rad;
    gsc_cfg.fixed_align_notebook = cfg->gsc_fixed_align_notebook;
    gsc_cfg.adapt_interval = gsc_effective_interval;
    p->gsc = gsc_create(
        FOUR_AEC_NR_RES_CHANNELS, p->n_freqs, cfg->num_angles,
        p->gsc_steering, &gsc_cfg);
    if (!p->gsc) goto fail;

    /* auto_vad_hangover_frames + the speech/silence noise-EMA pairs were
     * raw literals (8 frames; 0.999/0.001; 0.95/0.05) applied regardless of
     * grid -- same class of bug as gsc_lambda above, fixed the same way.
     * The EMA pairs are retention/new-weight complements (old_weight +
     * new_weight == 1 by construction); retime the retention factor and
     * derive new_weight from it so that invariant still holds post-retime. */
    p->vad_hangover_frames = mmse_lsa_retime_frames(
        cfg->auto_vad_hangover_frames, cfg->core.sample_rate, p->hop_size);
    p->vad_speech_noise_keep = mmse_lsa_retime_alpha(
        0.999f, cfg->core.sample_rate, p->hop_size);
    p->vad_speech_new_weight = 1.0f - p->vad_speech_noise_keep;
    p->vad_silence_noise_keep = mmse_lsa_retime_alpha(
        0.95f, cfg->core.sample_rate, p->hop_size);
    p->vad_silence_new_weight = 1.0f - p->vad_silence_noise_keep;

    spectral_count =
        (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)p->n_freqs;
    p->spatial_input =
        (Complex*)malloc(spectral_count * sizeof(Complex));
    p->gsc_spectrum =
        (Complex*)malloc((size_t)p->n_freqs *
                             sizeof(Complex));
    p->gsc_weights =
        (Complex*)malloc(spectral_count * sizeof(Complex));
    if (!p->spatial_input || !p->gsc_spectrum ||
        !p->gsc_weights) goto fail;
    for (int m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m) {
        p->spatial_channels[m] =
            p->spatial_input + (size_t)m * p->n_freqs;
    }
    p->noise_power =
        powf(10.0f, cfg->auto_vad_threshold_dbfs / 10.0f);
    return p;

fail:
    audio_pipeline_4ch_destroy(p);
    return NULL;
}

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

static int complex_values_finite(
    const Complex* values, size_t count) {
    if (!values) return 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i].r) || !isfinite(values[i].i)) return 0;
    }
    return 1;
}

static void fill_frame_info(
    AudioPipeline4Ch* p,
    const FourAecNrResPreFrame* pre,
    int vad_raw,
    int vad_out,
    int doa_analysis_frames,
    AudioPipeline4ChFrameInfo* info) {
    if (!info) return;
    info->frame_index = p->frame_index;
    info->delay = pre->delay;
    info->doa_raw_rad = doa_get_raw(p->srp);
    info->doa_smooth_rad = doa_get_smooth(p->srp);
    info->doa_used_rad = gsc_get_doa_used(p->gsc);
    info->vad_raw = vad_raw;
    info->vad_out = vad_out;
    info->gsc_adaptive = gsc_get_adaptive(p->gsc);
    info->doa_analysis_frames = doa_analysis_frames;
}

int audio_pipeline_4ch_process_with_activity(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    float* output,
    AudioPipeline4ChFrameInfo* info) {
    FourAecNrResPreFrame pre;
    size_t spectral_count;
    int status;
    int doa_analysis_frames;
    if (!p || !microphones_interleaved || !far_reference || !output ||
        !is_bool(vad_raw) || !is_bool(vad_out)) {
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    }
    status = four_aec_nr_res_process_pre(
        p->core, microphones_interleaved, far_reference, &pre);
    if (status != FOUR_AEC_NR_RES_OK) return status;
    if (pre.n_channels != FOUR_AEC_NR_RES_CHANNELS ||
        pre.hop_size != p->hop_size ||
        pre.n_freqs != p->n_freqs) {
        audio_pipeline_4ch_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    for (int m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m) {
        memcpy(p->spatial_channels[m], pre.linear_spectra[m],
               (size_t)p->n_freqs * sizeof(Complex));
    }
    doa_step(
        p->srp, p->spatial_channels, frequency_mask,
        vad_raw, vad_out);
    doa_analysis_frames = 1;
    gsc_process_with_weights(
        p->gsc, p->spatial_channels, doa_get_smooth(p->srp),
        vad_out ? 0 : 1, frequency_mask, p->gsc_spectrum,
        p->gsc_weights);

    spectral_count =
        (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)p->n_freqs;
    if (!complex_values_finite(
            p->gsc_spectrum, (size_t)p->n_freqs) ||
        !complex_values_finite(
            p->gsc_weights, spectral_count)) {
        audio_pipeline_4ch_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    /* gsc_spectrum and gsc_weights were produced atomically by the same
     * gsc_process_with_weights() call above.  Reuse that trusted mono error
     * instead of reconstructing one weighted sum a second time; the core
     * still projects near/R2/comfort with those exact weights. */
    status = four_aec_nr_res_process_post_trusted_spectrum(
        p->core, &pre.token, p->gsc_weights,
        p->gsc_spectrum, output);
    if (status != FOUR_AEC_NR_RES_OK) {
        audio_pipeline_4ch_reset(p);
        return status;
    }
    fill_frame_info(
        p, &pre, vad_raw, vad_out, doa_analysis_frames, info);
    p->frame_index += 1;
    return FOUR_AEC_NR_RES_OK;
}

static int auto_vad(
    AudioPipeline4Ch* p, const float* microphones_interleaved) {
    float sum = 0.0f;
    float power;
    float threshold =
        powf(10.0f, p->cfg.auto_vad_threshold_dbfs / 10.0f);
    int speech;
    size_t count =
        (size_t)p->hop_size * FOUR_AEC_NR_RES_CHANNELS;

    for (size_t i = 0; i < count; ++i) {
        float value = microphones_interleaved[i];
        sum += value * value;
    }
    power = sum / (float)count;
    speech = power >= threshold &&
             power >= p->noise_power * p->cfg.auto_vad_snr_ratio;
    if (speech) {
        p->vad_hangover = p->vad_hangover_frames;
        p->noise_power =
            p->vad_speech_noise_keep * p->noise_power + p->vad_speech_new_weight * power;
    } else {
        p->noise_power =
            p->vad_silence_noise_keep * p->noise_power + p->vad_silence_new_weight * power;
        if (p->vad_hangover > 0) {
            p->vad_hangover -= 1;
            speech = 1;
        }
    }
    if (p->noise_power < 1e-12f) p->noise_power = 1e-12f;
    return speech;
}

int audio_pipeline_4ch_process(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    float* output,
    AudioPipeline4ChFrameInfo* info) {
    int speech;
    if (!p || !microphones_interleaved)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    speech = auto_vad(p, microphones_interleaved);
    return audio_pipeline_4ch_process_with_activity(
        p, microphones_interleaved, far_reference,
        speech, speech, NULL, output, info);
}

/* ============================================================================
 * Reset and teardown
 * ========================================================================== */

void audio_pipeline_4ch_reset(AudioPipeline4Ch* p) {
    if (!p) return;
    four_aec_nr_res_reset(p->core);
    srp_reset(p->srp);
    gsc_reset(p->gsc);
    p->noise_power =
        powf(10.0f, p->cfg.auto_vad_threshold_dbfs / 10.0f);
    p->vad_hangover = 0;
    p->frame_index = 0;
}

void audio_pipeline_4ch_destroy(AudioPipeline4Ch* p) {
    if (!p) return;
    /* GSC borrows the steering table; destroy it before its owner. */
    gsc_destroy(p->gsc);
    srp_destroy(p->srp);
    four_aec_nr_res_destroy(p->core);
    free(p->spatial_input);
    free(p->gsc_spectrum);
    free(p->gsc_weights);
    free(p);
}

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int audio_pipeline_4ch_hop_size(const AudioPipeline4Ch* p) {
    return p ? p->hop_size : -1;
}

int audio_pipeline_4ch_frame_size(const AudioPipeline4Ch* p) {
    return p ? p->fft_size : -1;
}

int audio_pipeline_4ch_fft_size(const AudioPipeline4Ch* p) {
    return p ? p->fft_size : -1;
}

int audio_pipeline_4ch_n_freqs(const AudioPipeline4Ch* p) {
    return p ? p->n_freqs : -1;
}

int audio_pipeline_4ch_sample_rate(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_sample_rate(p->core) : -1;
}

int audio_pipeline_4ch_doa_sample_rate(const AudioPipeline4Ch* p) {
    return audio_pipeline_4ch_sample_rate(p);
}

int audio_pipeline_4ch_doa_frame_size(const AudioPipeline4Ch* p) {
    return audio_pipeline_4ch_frame_size(p);
}

int audio_pipeline_4ch_doa_hop_size(const AudioPipeline4Ch* p) {
    return audio_pipeline_4ch_hop_size(p);
}

int audio_pipeline_4ch_doa_fft_size(const AudioPipeline4Ch* p) {
    return audio_pipeline_4ch_fft_size(p);
}

int audio_pipeline_4ch_gsc_sample_rate(const AudioPipeline4Ch* p) {
    return p ? p->cfg.core.sample_rate : -1;
}

int audio_pipeline_4ch_gsc_frame_size(const AudioPipeline4Ch* p) {
    return p ? p->fft_size : -1;
}

int audio_pipeline_4ch_gsc_hop_size(const AudioPipeline4Ch* p) {
    return p ? p->hop_size : -1;
}

int audio_pipeline_4ch_gsc_fft_size(const AudioPipeline4Ch* p) {
    return p ? p->fft_size : -1;
}

int audio_pipeline_4ch_gsc_effective_adapt_interval(
    const AudioPipeline4Ch* p) {
    return (p && p->gsc) ? p->gsc->adapt_interval : -1;
}

float audio_pipeline_4ch_gsc_lambda(const AudioPipeline4Ch* p) {
    return (p && p->gsc) ? p->gsc->lambda : NAN;
}

int audio_pipeline_4ch_matched_filter_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_matched_filter_count(p->core) : 0;
}

int audio_pipeline_4ch_linear_aec_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_linear_aec_count(p->core) : 0;
}

int audio_pipeline_4ch_nr_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_nr_count(p->core) : 0;
}

int audio_pipeline_4ch_post_res_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_post_res_count(p->core) : 0;
}

const char* audio_pipeline_4ch_spatial_backend(void) {
    return spatial_simd_backend();
}
