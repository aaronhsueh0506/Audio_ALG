/**
 * 4aec_doa_gsc.c — complete four-channel spatial wrapper.
 *
 * This is the external-beamformer counterpart around 4aec_nr_res.c:
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
#include <stddef.h>

#include "4aec_doa_gsc.h"
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

struct FourAecDoaGsc {
    FourAecDoaGscConfig cfg;
    FourAecNrRes* core;
    SRP* srp;
    kiss_fft_cpx*** gsc_steering;
    GSC* gsc;

    int hop_size;
    int fft_size;
    int n_freqs;
    kiss_fft_cpx* spatial_input;
    kiss_fft_cpx* spatial_channels[FOUR_AEC_NR_RES_CHANNELS];
    kiss_fft_cpx* gsc_spectrum;
    kiss_fft_cpx* gsc_weights;
    Complex* core_weights;
    Complex* core_spectrum;

    float noise_power;
    int vad_hangover;
    uint64_t frame_index;
};

_Static_assert(sizeof(Complex) == sizeof(kiss_fft_cpx),
               "Complex and kiss_fft_cpx must have matching storage");
_Static_assert(offsetof(Complex, r) == offsetof(kiss_fft_cpx, r),
               "real component layout mismatch");
_Static_assert(offsetof(Complex, i) == offsetof(kiss_fft_cpx, i),
               "imaginary component layout mismatch");

/* ============================================================================
 * Config validation
 * ========================================================================== */

static int is_bool(int value) {
    return value == 0 || value == 1;
}

static int validate_config(const FourAecDoaGscConfig* cfg) {
    float nyquist;
    if (!cfg) return 0;
    if (cfg->core.sample_rate != 16000 &&
        cfg->core.sample_rate != 48000) return 0;
    nyquist = 0.5f * (float)cfg->core.sample_rate;
    if (cfg->geometry != FOUR_AEC_ARRAY_UCA &&
        cfg->geometry != FOUR_AEC_ARRAY_ULA &&
        cfg->geometry != FOUR_AEC_ARRAY_CUSTOM) return 0;
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
    if (cfg->geometry == FOUR_AEC_ARRAY_UCA &&
        (!isfinite(cfg->uca_radius_m) || cfg->uca_radius_m <= 0.0f))
        return 0;
    if (cfg->geometry == FOUR_AEC_ARRAY_ULA &&
        (!isfinite(cfg->ula_spacing_m) || cfg->ula_spacing_m <= 0.0f))
        return 0;
    if (cfg->geometry == FOUR_AEC_ARRAY_CUSTOM) {
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

FourAecDoaGscConfig four_aec_doa_gsc_default_config(int sample_rate) {
    FourAecDoaGscConfig cfg;
    float radius = DEFAULT_RADIUS_M;
    memset(&cfg, 0, sizeof(cfg));
    cfg.core = four_aec_nr_res_default_config(sample_rate);
    cfg.geometry = FOUR_AEC_ARRAY_UCA;
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
    const FourAecDoaGscConfig* cfg) {
    switch (cfg->geometry) {
        case FOUR_AEC_ARRAY_UCA:
            return array_geometry_create_uca(
                FOUR_AEC_NR_RES_CHANNELS, cfg->uca_radius_m);
        case FOUR_AEC_ARRAY_ULA:
            return array_geometry_create_ula(
                FOUR_AEC_NR_RES_CHANNELS, cfg->ula_spacing_m);
        case FOUR_AEC_ARRAY_CUSTOM:
            return array_geometry_create_custom(
                FOUR_AEC_NR_RES_CHANNELS,
                cfg->microphone_x_m, cfg->microphone_y_m);
        default:
            return NULL;
    }
}

FourAecDoaGsc* four_aec_doa_gsc_create(
    const FourAecDoaGscConfig* cfg) {
    FourAecDoaGsc* p;
    ArrayGeometry* geometry;
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    int fft_size;
    size_t spectral_count;

    if (!validate_config(cfg)) return NULL;
    p = (FourAecDoaGsc*)calloc(1, sizeof(*p));
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

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = cfg->gsc_enable;
    gsc_cfg.lambda = cfg->gsc_lambda;
    gsc_cfg.mu = cfg->gsc_mu;
    gsc_cfg.enable_fix_mode = cfg->gsc_fixed_mode;
    gsc_cfg.fixed_doa_rad = cfg->gsc_fixed_doa_rad;
    gsc_cfg.fixed_align_notebook = cfg->gsc_fixed_align_notebook;
    gsc_cfg.adapt_interval = cfg->gsc_adapt_interval;
    p->gsc = gsc_create(
        FOUR_AEC_NR_RES_CHANNELS, p->n_freqs, cfg->num_angles,
        p->gsc_steering, &gsc_cfg);
    if (!p->gsc) goto fail;

    spectral_count =
        (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)p->n_freqs;
    p->spatial_input =
        (kiss_fft_cpx*)malloc(spectral_count * sizeof(kiss_fft_cpx));
    p->gsc_spectrum =
        (kiss_fft_cpx*)malloc((size_t)p->n_freqs *
                             sizeof(kiss_fft_cpx));
    p->gsc_weights =
        (kiss_fft_cpx*)malloc(spectral_count * sizeof(kiss_fft_cpx));
    p->core_weights =
        (Complex*)malloc(spectral_count * sizeof(Complex));
    p->core_spectrum =
        (Complex*)malloc((size_t)p->n_freqs * sizeof(Complex));
    if (!p->spatial_input || !p->gsc_spectrum ||
        !p->gsc_weights || !p->core_weights ||
        !p->core_spectrum) goto fail;
    for (int m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m) {
        p->spatial_channels[m] =
            p->spatial_input + (size_t)m * p->n_freqs;
    }
    p->noise_power =
        powf(10.0f, cfg->auto_vad_threshold_dbfs / 10.0f);
    return p;

fail:
    four_aec_doa_gsc_destroy(p);
    return NULL;
}

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

static int complex_values_finite(
    const kiss_fft_cpx* values, size_t count) {
    if (!values) return 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i].r) || !isfinite(values[i].i)) return 0;
    }
    return 1;
}

static void fill_frame_info(
    FourAecDoaGsc* p,
    const FourAecNrResPreFrame* pre,
    int vad_raw,
    int vad_out,
    int doa_analysis_frames,
    FourAecDoaGscFrameInfo* info) {
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

int four_aec_doa_gsc_process_with_activity(
    FourAecDoaGsc* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    float* output,
    FourAecDoaGscFrameInfo* info) {
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
        four_aec_doa_gsc_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    for (int m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m) {
        memcpy(p->spatial_channels[m], pre.linear_spectra[m],
               (size_t)p->n_freqs * sizeof(kiss_fft_cpx));
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
        four_aec_doa_gsc_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    /* Complex and kiss_fft_cpx share a pinned two-float layout but are
     * distinct C struct types.  Assign fields rather than type-punning. */
    for (size_t i = 0; i < spectral_count; ++i) {
        p->core_weights[i].r = p->gsc_weights[i].r;
        p->core_weights[i].i = p->gsc_weights[i].i;
    }
    for (int f = 0; f < p->n_freqs; ++f) {
        p->core_spectrum[f].r = p->gsc_spectrum[f].r;
        p->core_spectrum[f].i = p->gsc_spectrum[f].i;
    }
    /* gsc_spectrum and gsc_weights were produced atomically by the same
     * gsc_process_with_weights() call above.  Reuse that trusted mono error
     * instead of reconstructing one weighted sum a second time; the core
     * still projects near/R2/comfort with those exact weights. */
    status = four_aec_nr_res_process_post_trusted_spectrum(
        p->core, &pre.token, p->core_weights,
        p->core_spectrum, output);
    if (status != FOUR_AEC_NR_RES_OK) {
        four_aec_doa_gsc_reset(p);
        return status;
    }
    fill_frame_info(
        p, &pre, vad_raw, vad_out, doa_analysis_frames, info);
    p->frame_index += 1;
    return FOUR_AEC_NR_RES_OK;
}

static int auto_vad(
    FourAecDoaGsc* p, const float* microphones_interleaved) {
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
        p->vad_hangover = p->cfg.auto_vad_hangover_frames;
        p->noise_power =
            0.999f * p->noise_power + 0.001f * power;
    } else {
        p->noise_power =
            0.95f * p->noise_power + 0.05f * power;
        if (p->vad_hangover > 0) {
            p->vad_hangover -= 1;
            speech = 1;
        }
    }
    if (p->noise_power < 1e-12f) p->noise_power = 1e-12f;
    return speech;
}

int four_aec_doa_gsc_process(
    FourAecDoaGsc* p,
    const float* microphones_interleaved,
    const float* far_reference,
    float* output,
    FourAecDoaGscFrameInfo* info) {
    int speech;
    if (!p || !microphones_interleaved)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    speech = auto_vad(p, microphones_interleaved);
    return four_aec_doa_gsc_process_with_activity(
        p, microphones_interleaved, far_reference,
        speech, speech, NULL, output, info);
}

/* ============================================================================
 * Reset and teardown
 * ========================================================================== */

void four_aec_doa_gsc_reset(FourAecDoaGsc* p) {
    if (!p) return;
    four_aec_nr_res_reset(p->core);
    srp_reset(p->srp);
    gsc_reset(p->gsc);
    p->noise_power =
        powf(10.0f, p->cfg.auto_vad_threshold_dbfs / 10.0f);
    p->vad_hangover = 0;
    p->frame_index = 0;
}

void four_aec_doa_gsc_destroy(FourAecDoaGsc* p) {
    if (!p) return;
    /* GSC borrows the steering table; destroy it before its owner. */
    gsc_destroy(p->gsc);
    srp_destroy(p->srp);
    four_aec_nr_res_destroy(p->core);
    free(p->spatial_input);
    free(p->gsc_spectrum);
    free(p->gsc_weights);
    free(p->core_weights);
    free(p->core_spectrum);
    free(p);
}

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int four_aec_doa_gsc_hop_size(const FourAecDoaGsc* p) {
    return p ? p->hop_size : -1;
}

int four_aec_doa_gsc_frame_size(const FourAecDoaGsc* p) {
    return p ? p->fft_size : -1;
}

int four_aec_doa_gsc_fft_size(const FourAecDoaGsc* p) {
    return p ? p->fft_size : -1;
}

int four_aec_doa_gsc_n_freqs(const FourAecDoaGsc* p) {
    return p ? p->n_freqs : -1;
}

int four_aec_doa_gsc_sample_rate(const FourAecDoaGsc* p) {
    return p ? four_aec_nr_res_sample_rate(p->core) : -1;
}

int four_aec_doa_gsc_doa_sample_rate(const FourAecDoaGsc* p) {
    return four_aec_doa_gsc_sample_rate(p);
}

int four_aec_doa_gsc_doa_frame_size(const FourAecDoaGsc* p) {
    return four_aec_doa_gsc_frame_size(p);
}

int four_aec_doa_gsc_doa_hop_size(const FourAecDoaGsc* p) {
    return four_aec_doa_gsc_hop_size(p);
}

int four_aec_doa_gsc_doa_fft_size(const FourAecDoaGsc* p) {
    return four_aec_doa_gsc_fft_size(p);
}

int four_aec_doa_gsc_gsc_sample_rate(const FourAecDoaGsc* p) {
    return p ? p->cfg.core.sample_rate : -1;
}

int four_aec_doa_gsc_gsc_frame_size(const FourAecDoaGsc* p) {
    return p ? p->fft_size : -1;
}

int four_aec_doa_gsc_gsc_hop_size(const FourAecDoaGsc* p) {
    return p ? p->hop_size : -1;
}

int four_aec_doa_gsc_gsc_fft_size(const FourAecDoaGsc* p) {
    return p ? p->fft_size : -1;
}

int four_aec_doa_gsc_matched_filter_count(const FourAecDoaGsc* p) {
    return p ? four_aec_nr_res_matched_filter_count(p->core) : 0;
}

int four_aec_doa_gsc_linear_aec_count(const FourAecDoaGsc* p) {
    return p ? four_aec_nr_res_linear_aec_count(p->core) : 0;
}

int four_aec_doa_gsc_nr_count(const FourAecDoaGsc* p) {
    return p ? four_aec_nr_res_nr_count(p->core) : 0;
}

int four_aec_doa_gsc_post_res_count(const FourAecDoaGsc* p) {
    return p ? four_aec_nr_res_post_res_count(p->core) : 0;
}

const char* four_aec_doa_gsc_spatial_backend(void) {
    return spatial_simd_backend();
}
