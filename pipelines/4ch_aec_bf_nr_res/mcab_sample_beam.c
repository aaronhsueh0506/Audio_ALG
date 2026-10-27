/** Implementation of mcab_sample_beam.h: SRP-PHAT + BF + fallback VAD between
 * mcab_process_front() and mcab_fuse(), with the arithmetic and the refusal/reset rules that
 * header documents. Sample code, not part of the mcab module. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mcab_sample_beam.h"
#include "anr_types.h"            /* anr_retime_alpha, anr_retime_frames */
#include "spatial_simd.h"
#include "utility/mem_align.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PI_F ((float)M_PI)

#define DEFAULT_RADIUS_M  0.035f
#define DEFAULT_SPACING_M 0.035f
#define DEFAULT_ANGLES    72

struct McabSampleBeam {
    McabSampleBeamParams params;
    Mcab* flow;                   /* borrowed */
    SRP* srp;
    BF* bf;
    int hop_size;
    int n_freqs;
    Complex* bf_spectrum;
    Complex* bf_weights;
    int phase;                    /* 0 idle, 1 front done, 2 beam done */

    /* Fallback VAD, retimed to the grid once at init. */
    float noise_power;
    float vad_power_threshold;
    int vad_hangover;
    int vad_hangover_frames;
    float vad_speech_noise_keep;
    float vad_speech_new_weight;
    float vad_silence_noise_keep;
    float vad_silence_new_weight;
    uint64_t frame_index;
    int destroyed;
};

McabSampleBeamParams mcab_sample_beam_default_params(void) {
    McabSampleBeamParams bp;
    float radius = DEFAULT_RADIUS_M;
    int m;
    memset(&bp, 0, sizeof(bp));
    bp.geometry = MCAB_SAMPLE_GEOMETRY_UCA;
    bp.uca_radius_m = radius;
    bp.ula_spacing_m = DEFAULT_SPACING_M;
    /* float arithmetic for the CUSTOM defaults, double in fill_uca_geometry(): do not unify,
     * the outputs are pinned bit-exact. */
    for (m = 0; m < MCAB_CHANNELS; ++m) {
        float phi = 2.0f * M_PI_F * (float)m / (float)MCAB_CHANNELS;
        bp.microphone_x_m[m] = radius * cosf(phi);
        bp.microphone_y_m[m] = radius * sinf(phi);
    }
    bp.num_angles = DEFAULT_ANGLES;
    bp.speed_of_sound_m_s = 343.0f;
    bp.doa_low_freq_hz = 300.0f;
    bp.doa_high_freq_hz = 7000.0f;
    bp.doa_enable_smoothing = 1;
    bp.doa_switch_consecutive = 3;
    bp.doa_angle_tolerance_rad = 10.0f * M_PI_F / 180.0f;
    bp.doa_update_interval = 2;
    bp.bf_enable = 1;
    bp.bf_lambda = 0.995f;
    bp.bf_mu = 0.1f;
    bp.bf_fixed_mode = 0;
    bp.bf_fixed_doa_rad = 0.0f;
    bp.bf_fixed_align_notebook = 0;
    bp.bf_adapt_interval = 1;
    bp.auto_vad_threshold_dbfs = -55.0f;
    bp.auto_vad_snr_ratio = 3.0f;
    bp.auto_vad_hangover_frames = 8;
    return bp;
}

static int flag_valid(int v) { return v == 0 || v == 1; }

static int params_valid(const McabSampleBeamParams* bp, int sample_rate) {
    float nyquist = 0.5f * (float)sample_rate;
    int m;
    if (bp->geometry != MCAB_SAMPLE_GEOMETRY_UCA && bp->geometry != MCAB_SAMPLE_GEOMETRY_ULA &&
        bp->geometry != MCAB_SAMPLE_GEOMETRY_CUSTOM) return 0;
    if (bp->num_angles < 4 || bp->num_angles > 3600) return 0;
    if (!isfinite(bp->speed_of_sound_m_s) || bp->speed_of_sound_m_s <= 0.0f) return 0;
    if (!isfinite(bp->doa_low_freq_hz) || bp->doa_low_freq_hz < 0.0f ||
        bp->doa_low_freq_hz >= nyquist) return 0;
    if (!isfinite(bp->doa_high_freq_hz) || bp->doa_high_freq_hz < 0.0f ||
        bp->doa_high_freq_hz > nyquist) return 0;
    /* Against the band SRP is given: 0 resolves to min(7000, Nyquist) there. */
    if (bp->doa_low_freq_hz >=
        (bp->doa_high_freq_hz > 0.0f ? bp->doa_high_freq_hz : fminf(7000.0f, nyquist))) return 0;
    if (!flag_valid(bp->doa_enable_smoothing) || bp->doa_switch_consecutive <= 0 ||
        !isfinite(bp->doa_angle_tolerance_rad) || bp->doa_angle_tolerance_rad < 0.0f ||
        bp->doa_update_interval <= 0) return 0;
    if (!flag_valid(bp->bf_enable) ||
        !isfinite(bp->bf_lambda) || bp->bf_lambda <= 0.0f || bp->bf_lambda > 1.0f ||
        !isfinite(bp->bf_mu) || bp->bf_mu < 0.0f ||
        !flag_valid(bp->bf_fixed_mode) || !isfinite(bp->bf_fixed_doa_rad) ||
        !flag_valid(bp->bf_fixed_align_notebook) || bp->bf_adapt_interval <= 0 ||
        bp->bf_adapt_interval > MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL) return 0;
    if (!isfinite(bp->auto_vad_threshold_dbfs) ||
        !isfinite(bp->auto_vad_snr_ratio) || bp->auto_vad_snr_ratio < 1.0f ||
        bp->auto_vad_hangover_frames < 0) return 0;
    if (bp->geometry == MCAB_SAMPLE_GEOMETRY_UCA &&
        (!isfinite(bp->uca_radius_m) || bp->uca_radius_m <= 0.0f)) return 0;
    if (bp->geometry == MCAB_SAMPLE_GEOMETRY_ULA &&
        (!isfinite(bp->ula_spacing_m) || bp->ula_spacing_m <= 0.0f)) return 0;
    if (bp->geometry == MCAB_SAMPLE_GEOMETRY_CUSTOM) {
        for (m = 0; m < MCAB_CHANNELS; ++m) {
            if (!isfinite(bp->microphone_x_m[m]) || !isfinite(bp->microphone_y_m[m])) return 0;
        }
    }
    return 1;
}

static void fill_uca_geometry(float radius, float* x, float* y) {
    int M = MCAB_CHANNELS;
    int m;
    for (m = 0; m < M; m++) {
        float phi = 2.0f * M_PI * m / M;   /* double on purpose, see the default params */
        x[m] = radius * cosf(phi);
        y[m] = radius * sinf(phi);
    }
}

static void fill_ula_geometry(float spacing, float* x, float* y) {
    int M = MCAB_CHANNELS;
    float center = 0.5f * (M - 1);
    int m;
    for (m = 0; m < M; m++) {
        x[m] = (m - center) * spacing;
        y[m] = 0.0f;
    }
}

int mcab_sample_spatial_config(const McabConfig* cfg, const McabSampleBeamParams* bp,
                               McabSampleSpatialConfig* out) {
    McabMemBreakdown grid;
    int hop, fft, n, m;
    int bf_effective_interval;
    if (!out || !bp || mcab_get_mem_breakdown(cfg, &grid) != 0 ||
        !params_valid(bp, cfg->sample_rate)) return -1;
    hop = grid.hop_size;
    fft = grid.fft_size;
    n = grid.n_freqs;

    memset(out, 0, sizeof(*out));
    out->n_channels = MCAB_CHANNELS;
    out->n_freqs = n;
    out->num_angles = bp->num_angles;
    switch (bp->geometry) {
        case MCAB_SAMPLE_GEOMETRY_UCA: fill_uca_geometry(bp->uca_radius_m, out->mic_x, out->mic_y); break;
        case MCAB_SAMPLE_GEOMETRY_ULA: fill_ula_geometry(bp->ula_spacing_m, out->mic_x, out->mic_y); break;
        default:
            for (m = 0; m < MCAB_CHANNELS; ++m) {
                out->mic_x[m] = bp->microphone_x_m[m];
                out->mic_y[m] = bp->microphone_y_m[m];
            }
            break;
    }

    out->srp.M = MCAB_CHANNELS;
    out->srp.F = n;
    out->srp.num_angles = bp->num_angles;
    out->srp.sr = (float)cfg->sample_rate;
    out->srp.NFFT = (float)fft;
    out->srp.c = bp->speed_of_sound_m_s;
    out->srp.low_freq = bp->doa_low_freq_hz;
    out->srp.high_freq = bp->doa_high_freq_hz > 0.0f
        ? fminf(bp->doa_high_freq_hz, 0.5f * (float)cfg->sample_rate)
        : fminf(7000.0f, 0.5f * (float)cfg->sample_rate);
    out->srp.enable_smoothing = bp->doa_enable_smoothing;
    out->srp.switch_consec = bp->doa_switch_consecutive;
    out->srp.angle_tol = bp->doa_angle_tolerance_rad;
    out->srp.update_interval = bp->doa_update_interval;

    /* BF forces an every-hop cadence in fixed-notebook mode; the lambda retime and adapt_interval
     * must both use the cadence BF really runs. */
    bf_effective_interval = bf_effective_adapt_interval(
        bp->bf_fixed_mode, bp->bf_fixed_align_notebook, bp->bf_adapt_interval);
    out->bf.enable = bp->bf_enable;
    /* An RLS retention tuned at a 10 ms update is retimed like NR's constants; mu is a step size
     * and stays as is. */
    out->bf.lambda = anr_retime_alpha(bp->bf_lambda, cfg->sample_rate, hop * bf_effective_interval);
    /* A long cadence can underflow the retimed retention to 0, which BF refuses: refuse it here
     * too, so the size query and init agree. */
    if (!(out->bf.lambda > 0.0f)) return -1;
    out->bf.mu = bp->bf_mu;
    out->bf.enable_fix_mode = bp->bf_fixed_mode;
    out->bf.fixed_doa_rad = bp->bf_fixed_doa_rad;
    out->bf.fixed_align_notebook = bp->bf_fixed_align_notebook;
    out->bf.adapt_interval = bf_effective_interval;
    return 0;
}

typedef struct HelperLayout {
    McabSampleSpatialConfig sc;
    size_t srp_bytes;
    size_t bf_bytes;
    size_t total;
} HelperLayout;

static int helper_layout(const McabConfig* cfg, const McabSampleBeamParams* bp, HelperLayout* l) {
    size_t total;
    if (mcab_sample_spatial_config(cfg, bp, &l->sc) != 0) return 0;
    l->srp_bytes = srp_get_mem_size(&l->sc.srp);
    l->bf_bytes = bf_get_mem_size(l->sc.n_channels, l->sc.n_freqs);
    if (l->srp_bytes == 0 || l->bf_bytes == 0) return 0;
    total = ck_align16_size(sizeof(McabSampleBeam));
    total = ck_add_size(total, ck_align16_size(l->srp_bytes));
    total = ck_add_size(total, ck_align16_size(l->bf_bytes));
    total = ck_field_size(total, (size_t)l->sc.n_freqs, sizeof(Complex));
    total = ck_field_size(total, (size_t)MCAB_CHANNELS * (size_t)l->sc.n_freqs, sizeof(Complex));
    if (MEM_SIZE_INVALID(total)) return 0;
    l->total = total;
    return 1;
}

int mcab_sample_beam_get_mem_size(const McabConfig* cfg, const McabSampleBeamParams* bp,
                                  size_t* bytes) {
    HelperLayout l;
    if (!bytes || !cfg || !bp || !helper_layout(cfg, bp, &l)) return -1;
    *bytes = l.total;
    return 0;
}

static void* take(uint8_t** cursor, size_t* remaining, size_t bytes) {
    size_t aligned = ck_align16_size(bytes);
    void* out;
    if (bytes == 0 || MEM_SIZE_INVALID(aligned) || aligned > *remaining) return NULL;
    out = *cursor;
    *cursor += aligned;
    *remaining -= aligned;
    return out;
}

McabSampleBeam* mcab_sample_beam_init(void* mem, size_t bytes, const McabConfig* cfg,
                                      const McabSampleBeamParams* bp, Mcab* flow) {
    HelperLayout l;
    McabSampleBeam* b;
    ArrayGeometry geom;
    uint8_t* cursor;
    size_t remaining;
    void* region;
    int hop;
    if (!mem || !cfg || !bp || !flow || !MEM_IS_ALIGNED16(mem) || !helper_layout(cfg, bp, &l) ||
        bytes < l.total) return NULL;
    hop = l.sc.n_freqs - 1;   /* 50 % overlap: n_freqs = hop + 1 */
    if (mcab_n_freqs(flow) != l.sc.n_freqs || mcab_hop_size(flow) != hop ||
        mcab_sample_rate(flow) != cfg->sample_rate ||
        mcab_enable_post(flow) != cfg->enable_post) return NULL;
    memset(mem, 0, l.total);
    b = (McabSampleBeam*)mem;
    b->params = *bp;
    b->flow = flow;
    b->hop_size = hop;
    b->n_freqs = l.sc.n_freqs;
    cursor = (uint8_t*)mem + ck_align16_size(sizeof(*b));
    remaining = l.total - ck_align16_size(sizeof(*b));
    /* ArrayGeometry is read only inside srp_init(), so stack storage suffices. */
    geom.M = l.sc.n_channels;
    geom.x = l.sc.mic_x;
    geom.y = l.sc.mic_y;
    region = take(&cursor, &remaining, l.srp_bytes);
    if (!region || !(b->srp = srp_init(region, l.srp_bytes, &l.sc.srp, &geom)) ||
        !b->srp->a_array) return NULL;
    /* BF borrows SRP's steering table, which lives in this pool. */
    region = take(&cursor, &remaining, l.bf_bytes);
    if (!region || !(b->bf = bf_init(region, l.bf_bytes, l.sc.n_channels, l.sc.n_freqs,
                                     l.sc.num_angles, b->srp->a_array, &l.sc.bf)))
        return NULL;
    b->bf_spectrum = (Complex*)take(&cursor, &remaining, (size_t)b->n_freqs * sizeof(Complex));
    b->bf_weights = (Complex*)take(
        &cursor, &remaining, (size_t)MCAB_CHANNELS * (size_t)b->n_freqs * sizeof(Complex));
    if (!b->bf_spectrum || !b->bf_weights || remaining != 0) return NULL;

    /* Hangover and the noise-EMA pairs are retimed to the grid like bf_lambda; keep and new
     * weight sum to 1. */
    b->vad_hangover_frames = anr_retime_frames(bp->auto_vad_hangover_frames, cfg->sample_rate, hop);
    b->vad_speech_noise_keep = anr_retime_alpha(0.999f, cfg->sample_rate, hop);
    b->vad_speech_new_weight = 1.0f - b->vad_speech_noise_keep;
    b->vad_silence_noise_keep = anr_retime_alpha(0.95f, cfg->sample_rate, hop);
    b->vad_silence_new_weight = 1.0f - b->vad_silence_noise_keep;
    b->vad_power_threshold = powf(10.0f, bp->auto_vad_threshold_dbfs / 10.0f);
    b->noise_power = b->vad_power_threshold;
    return b;
}

void mcab_sample_beam_destroy(McabSampleBeam* b) {
    if (!b || b->destroyed) return;
    bf_destroy(b->bf);
    srp_destroy(b->srp);
    b->destroyed = 1;
}

void mcab_sample_beam_reset(McabSampleBeam* b) {
    if (!b || b->destroyed) return;
    mcab_reset(b->flow);
    srp_reset(b->srp);
    bf_reset(b->bf);
    b->phase = 0;
    b->vad_power_threshold = powf(10.0f, b->params.auto_vad_threshold_dbfs / 10.0f);
    b->noise_power = b->vad_power_threshold;
    b->vad_hangover = 0;
    b->frame_index = 0;
}

int mcab_sample_beam_process_front(McabSampleBeam* b, const float* microphones_interleaved,
                                   const float* far_reference, McabFront* out) {
    int status;
    if (!b || b->destroyed || !microphones_interleaved || !far_reference || !out)
        return MCAB_INVALID_ARGUMENT;
    status = mcab_process_front(b->flow, microphones_interleaved, far_reference, out);
    /* The flow dropped its front on any refusal, so no beam or fuse may follow. */
    b->phase = status == MCAB_OK ? 1 : 0;
    return status;
}

static void fill_info(McabSampleBeam* b, const McabFront* front, int vad_raw, int vad_out,
                      McabSampleBeamInfo* info) {
    if (!info) return;
    info->frame_index = b->frame_index;
    info->delay = front->delay;
    info->doa_raw_rad = doa_get_raw(b->srp);
    info->doa_smooth_rad = doa_get_smooth(b->srp);
    info->doa_used_rad = bf_get_doa_used(b->bf);
    info->vad_raw = vad_raw;
    info->vad_out = vad_out;
    info->bf_adaptive = bf_get_adaptive(b->bf);
    info->doa_analysis_frames = 1;
}

int mcab_sample_beam_process_beam(McabSampleBeam* b, const McabFront* front, int vad_raw,
                                  int vad_out, const int* frequency_mask,
                                  McabSampleBeamResult* beam, McabSampleBeamInfo* info) {
    int m;
    if (!b || b->destroyed || !front || !beam || !flag_valid(vad_raw) ||
        !flag_valid(vad_out) || front->n_freqs != b->n_freqs)
        return MCAB_INVALID_ARGUMENT;
    for (m = 0; m < MCAB_CHANNELS; ++m)
        if (!front->linear_spectra[m]) return MCAB_INVALID_ARGUMENT;
    if (b->phase != 1) return MCAB_SEQUENCE_ERROR;

    /* SRP/BF only read the lanes' live spectra, so no copy. */
    doa_step(b->srp, front->linear_spectra, frequency_mask, vad_raw, vad_out);
    bf_process_with_weights(b->bf, front->linear_spectra, doa_get_smooth(b->srp),
                            vad_out ? 0 : 1, frequency_mask, b->bf_spectrum, b->bf_weights);

    beam->spectrum = b->bf_spectrum;
    beam->weights = b->bf_weights;
    fill_info(b, front, vad_raw, vad_out, info);
    b->frame_index += 1;
    b->phase = 2;
    return MCAB_OK;
}

int mcab_sample_beam_process_with_activity(McabSampleBeam* b, const float* microphones_interleaved,
                                           const float* far_reference, int vad_raw, int vad_out,
                                           const int* frequency_mask, McabSampleBeamResult* beam,
                                           McabSampleBeamInfo* info) {
    McabFront front;
    int status;
    if (!b || b->destroyed || !microphones_interleaved || !far_reference || !beam ||
        !flag_valid(vad_raw) || !flag_valid(vad_out))
        return MCAB_INVALID_ARGUMENT;
    status = mcab_sample_beam_process_front(b, microphones_interleaved, far_reference, &front);
    if (status != MCAB_OK) return status;
    return mcab_sample_beam_process_beam(b, &front, vad_raw, vad_out, frequency_mask, beam, info);
}

static int floats_finite(const float* values, size_t count) {
    size_t i;
    if (!values) return 0;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static int auto_vad(McabSampleBeam* b, const float* microphones_interleaved) {
    float sum = 0.0f;
    float power;
    float threshold = b->vad_power_threshold;
    int speech;
    size_t count = (size_t)b->hop_size * MCAB_CHANNELS;

    for (size_t i = 0; i < count; ++i) {
        float value = microphones_interleaved[i];
        sum += value * value;
    }
    power = sum / (float)count;
    if (!isfinite(power)) {
        /* A non-finite hop touches no VAD state: the next valid hop decides as if it never
         * happened. */
        return b->vad_hangover > 0;
    }
    speech = power >= threshold && power >= b->noise_power * b->params.auto_vad_snr_ratio;
    if (speech) {
        b->vad_hangover = b->vad_hangover_frames;
        b->noise_power = b->vad_speech_noise_keep * b->noise_power + b->vad_speech_new_weight * power;
    } else {
        b->noise_power =
            b->vad_silence_noise_keep * b->noise_power + b->vad_silence_new_weight * power;
        if (b->vad_hangover > 0) {
            b->vad_hangover -= 1;
            speech = 1;
        }
    }
    if (b->noise_power < 1e-12f) b->noise_power = 1e-12f;
    return speech;
}

int mcab_sample_beam_process(McabSampleBeam* b, const float* microphones_interleaved,
                             const float* far_reference, McabSampleBeamResult* beam,
                             McabSampleBeamInfo* info) {
    int speech;
    if (!b || b->destroyed || !microphones_interleaved || !far_reference || !beam)
        return MCAB_INVALID_ARGUMENT;
    /* auto_vad() reads only the microphones: refuse a non-finite far before it moves. */
    if (!floats_finite(far_reference, (size_t)b->hop_size)) return MCAB_INVALID_ARGUMENT;
    speech = auto_vad(b, microphones_interleaved);
    return mcab_sample_beam_process_with_activity(b, microphones_interleaved, far_reference,
                                                  speech, speech, NULL, beam, info);
}

int mcab_sample_beam_fuse(McabSampleBeam* b, McabFrame* frame) {
    int status;
    /* The flow's own setting decides: a helper config that disagrees cannot turn the flow's
     * plain refusal into a reset. */
    if (!b || b->destroyed || !frame || mcab_enable_post(b->flow) != 1) return MCAB_INVALID_ARGUMENT;
    if (b->phase != 2) return MCAB_SEQUENCE_ERROR;
    /* Spectrum and weights come from one bf_process_with_weights() call. A refusal of what this
     * helper computed takes everything down, not only the flow; a sequence refusal changed
     * nothing. */
    status = mcab_fuse(b->flow, b->bf_weights, b->bf_spectrum, frame);
    if (status != MCAB_OK && status != MCAB_SEQUENCE_ERROR) mcab_sample_beam_reset(b);
    return status;
}

int mcab_sample_beam_bf_effective_adapt_interval(const McabSampleBeam* b) {
    return b && !b->destroyed ? b->bf->adapt_interval : -1;
}

float mcab_sample_beam_bf_lambda(const McabSampleBeam* b) {
    return b && !b->destroyed ? b->bf->lambda : NAN;
}

const char* mcab_sample_beam_spatial_backend(void) { return spatial_simd_backend(); }
