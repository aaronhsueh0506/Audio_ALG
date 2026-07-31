/**
 * 4aec_nr_res.c
 *
 * Synchronous C implementation of:
 *
 *   one shared DelayAec3 -> one aligned render
 *   -> four independent linear AEC filters
 *   -> externally supplied effective beamformer weights
 *   -> one coherent post-beam RES + one NR + one iFFT/OLA
 *
 * Construction may allocate. The two process calls do not.
 */

#include "4aec_nr_res.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "fft_wrapper.h"
#include "suppression_gain.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

#define FOUR_AEC_NR_RES_PSD_SCALE \
    (32768.0f * 32768.0f)
#define FOUR_AEC_NR_RES_RNG_SEED 0x9e3779b9u
#define FOUR_AEC_NR_RES_NE_FLOOR 0.4f
#define FOUR_AEC_NR_RES_NE_FLOOR_FAR_ACTIVE 0.2f
#define FOUR_AEC_NR_RES_FAR_GATE_THRESHOLD 1e-4f
#define FOUR_AEC_NR_RES_NEAR_GATE_THRESHOLD 1e-3f
#define FOUR_AEC_NR_RES_NEAR_HANGOVER 8

typedef struct FourAecLaneSnapshot {
    Complex* error_spec;
    Complex* echo_spec;
    Complex* far_spec;
    Complex* near_spec;
    float* r2;
    float* comfort_noise;

    float far_power;
    float erle_factor;
    float dt_indicator;
    float divergence;
    float saturation_level;
    float erl_estimate;
    int filter_converged;
} FourAecLaneSnapshot;

struct FourAecNrRes {
    FourAecNrResConfig cfg;
    int sample_rate;
    int fft_size;
    int hop_size;
    int n_freqs;
    int rate_factor;
    int initialized_lanes;

    Aec lanes[FOUR_AEC_NR_RES_CHANNELS];
    DelayAec3 shared_delay;
    MmseLsaDenoiser* nr;
    FftHandle* fft;

    SuppressionGain post_sg;
    SuppressionGainConfig post_sg_cfg;
    SuppressionGainTuning post_sg_tun;
    float* post_sg_storage;

    FourAecLaneSnapshot snapshots[FOUR_AEC_NR_RES_CHANNELS];

    float* linear_interleaved;
    float* mic_lane;
    float* lane_out;
    float* aligned_ref;
    float* render_i16;

    float* delay_ring;
    int delay_ring_size;
    uint64_t delay_samples_seen;
    int accepted_delay;
    int delay_phase;
    float* delay_capture;
    float* delay_render;
    uint64_t delay_calls;
    FourAecNrResDelayState last_delay;

    Complex* fused_error;
    Complex* fused_echo;
    Complex* fused_near;
    Complex* fused_far;
    Complex* output_spec;
    float* fused_r2;
    float* fused_comfort;
    float* post_near_power;
    float* nr_gain;
    float* total_gain;
    float* extra_noise;
    float* error_power;

    float* synth_window;
    float* ifft_buffer;
    float* ola;

    uint32_t rng_state;
    int near_hang;

    uint64_t next_frame;
    uint64_t generation;
    int pending;
    FourAecNrResFrameToken pending_token;
};

static int four_is_bool(int value) {
    return value == 0 || value == 1;
}

static int four_valid_aec_preset(AecPreset preset) {
    return preset == AEC_PRESET_MILD ||
           preset == AEC_PRESET_BALANCED ||
           preset == AEC_PRESET_AGGRESSIVE;
}

static int four_valid_nr_mode(MmseLsaNrMode mode) {
    return mode == MMSE_LSA_NR_MILD ||
           mode == MMSE_LSA_NR_MODERATE ||
           mode == MMSE_LSA_NR_BALANCED ||
           mode == MMSE_LSA_NR_AGGRESSIVE;
}

static int four_validate_config(const FourAecNrResConfig* cfg,
                                int* fft_size) {
    int expected_fft;
    if (!cfg || !fft_size) return 0;
    if (cfg->sample_rate != 16000 && cfg->sample_rate != 48000) return 0;
    expected_fft = cfg->sample_rate == 16000 ? 512 : 1024;
    if (cfg->fft_size != 0 && cfg->fft_size != expected_fft) return 0;
    if (cfg->filter_length < 0 || cfg->filter_length > 4096) return 0;
    if (cfg->capture_proxy_channel < 0 ||
        cfg->capture_proxy_channel >= FOUR_AEC_NR_RES_CHANNELS) return 0;
    if (!isfinite(cfg->max_delay_ms) ||
        cfg->max_delay_ms < 0.0f || cfg->max_delay_ms > 4096.0f) return 0;
    if (!four_valid_aec_preset(cfg->aec_preset)) return 0;
    if (!four_valid_nr_mode(cfg->nr_mode)) return 0;
    if (!four_is_bool(cfg->enable_cng) ||
        !four_is_bool(cfg->legacy_amin)) return 0;
    *fft_size = expected_fft;
    return 1;
}

static void* four_calloc(size_t count, size_t size) {
    if (count == 0 || size == 0 || count > SIZE_MAX / size) return NULL;
    return calloc(count, size);
}

static int four_alloc_lane_snapshot(FourAecLaneSnapshot* s, int n_freqs) {
    s->error_spec = (Complex*)four_calloc((size_t)n_freqs, sizeof(Complex));
    s->echo_spec = (Complex*)four_calloc((size_t)n_freqs, sizeof(Complex));
    s->far_spec = (Complex*)four_calloc((size_t)n_freqs, sizeof(Complex));
    s->near_spec = (Complex*)four_calloc((size_t)n_freqs, sizeof(Complex));
    s->r2 = (float*)four_calloc((size_t)n_freqs, sizeof(float));
    s->comfort_noise = (float*)four_calloc((size_t)n_freqs, sizeof(float));
    return s->error_spec && s->echo_spec && s->far_spec && s->near_spec &&
           s->r2 && s->comfort_noise;
}

static void four_free_lane_snapshot(FourAecLaneSnapshot* s) {
    if (!s) return;
    free(s->error_spec);
    free(s->echo_spec);
    free(s->far_spec);
    free(s->near_spec);
    free(s->r2);
    free(s->comfort_noise);
    memset(s, 0, sizeof(*s));
}

static int four_inputs_finite(const float* data, size_t count) {
    size_t i;
    if (!data) return 0;
    for (i = 0; i < count; ++i) {
        if (!isfinite(data[i])) return 0;
    }
    return 1;
}

static Complex four_complex_mul(Complex a, Complex b) {
    Complex out;
    out.r = a.r * b.r - a.i * b.i;
    out.i = a.r * b.i + a.i * b.r;
    return out;
}

static void four_complex_accumulate(Complex* dst, Complex value) {
    dst->r += value.r;
    dst->i += value.i;
}

static int four_complex_close(Complex a, Complex b) {
    float dr = fabsf(a.r - b.r);
    float di = fabsf(a.i - b.i);
    float scale = 1.0f + fabsf(a.r) + fabsf(a.i);
    return dr <= 1e-5f * scale && di <= 1e-5f * scale;
}

static float four_rng_uniform(FourAecNrRes* self) {
    uint32_t x = self->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    self->rng_state = x;
    return ((x >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

static float four_rng_gauss(FourAecNrRes* self) {
    float u1 = four_rng_uniform(self);
    float u2 = four_rng_uniform(self);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI_F * u2);
}

void four_aec_nr_res_config_defaults(FourAecNrResConfig* cfg,
                                     int sample_rate) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->sample_rate = sample_rate;
    cfg->fft_size = 0;
    cfg->filter_length = 0;
    cfg->capture_proxy_channel = 0;
    cfg->max_delay_ms = 1024.0f;
    cfg->aec_preset = AEC_PRESET_BALANCED;
    cfg->nr_mode = MMSE_LSA_NR_BALANCED;
    cfg->enable_cng = 1;
    cfg->legacy_amin = 0;
}

static int four_allocate_working_buffers(FourAecNrRes* self) {
    int ch;
    int n = self->n_freqs;
    int hop = self->hop_size;
    int fft = self->fft_size;
    size_t linear_count = (size_t)hop * FOUR_AEC_NR_RES_CHANNELS;

    self->linear_interleaved =
        (float*)four_calloc(linear_count, sizeof(float));
    self->mic_lane = (float*)four_calloc((size_t)hop, sizeof(float));
    self->lane_out = (float*)four_calloc((size_t)hop, sizeof(float));
    self->aligned_ref = (float*)four_calloc((size_t)hop, sizeof(float));
    self->render_i16 = (float*)four_calloc((size_t)hop, sizeof(float));
    self->delay_capture = (float*)four_calloc((size_t)hop, sizeof(float));
    self->delay_render = (float*)four_calloc((size_t)hop, sizeof(float));

    self->fused_error = (Complex*)four_calloc((size_t)n, sizeof(Complex));
    self->fused_echo = (Complex*)four_calloc((size_t)n, sizeof(Complex));
    self->fused_near = (Complex*)four_calloc((size_t)n, sizeof(Complex));
    self->fused_far = (Complex*)four_calloc((size_t)n, sizeof(Complex));
    self->output_spec = (Complex*)four_calloc((size_t)n, sizeof(Complex));

    self->fused_r2 = (float*)four_calloc((size_t)n, sizeof(float));
    self->fused_comfort = (float*)four_calloc((size_t)n, sizeof(float));
    self->post_near_power = (float*)four_calloc((size_t)n, sizeof(float));
    self->nr_gain = (float*)four_calloc((size_t)n, sizeof(float));
    self->total_gain = (float*)four_calloc((size_t)n, sizeof(float));
    self->extra_noise = (float*)four_calloc((size_t)n, sizeof(float));
    self->error_power = (float*)four_calloc((size_t)n, sizeof(float));

    self->synth_window = (float*)four_calloc((size_t)fft, sizeof(float));
    self->ifft_buffer = (float*)four_calloc((size_t)fft, sizeof(float));
    self->ola = (float*)four_calloc((size_t)fft, sizeof(float));

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        if (!four_alloc_lane_snapshot(&self->snapshots[ch], n)) return 0;
    }

    return self->linear_interleaved && self->mic_lane && self->lane_out &&
           self->aligned_ref && self->render_i16 &&
           self->delay_capture && self->delay_render &&
           self->fused_error && self->fused_echo && self->fused_near &&
           self->fused_far && self->output_spec &&
           self->fused_r2 && self->fused_comfort && self->post_near_power &&
           self->nr_gain && self->total_gain && self->extra_noise &&
           self->error_power && self->synth_window && self->ifft_buffer &&
           self->ola;
}

static int four_init_post_sg(FourAecNrRes* self) {
    int n = self->n_freqs;
    int ma_n;
    size_t float_count;
    float* cursor;

    self->post_sg_cfg = self->lanes[0].a3_sg.cfg;
    self->post_sg_tun = self->lanes[0].a3_sg.tun;
    ma_n = self->post_sg_cfg.nearend_smoother_n;
    if (ma_n < 1 || self->post_sg_cfg.n_bins != n ||
        self->post_sg_tun.table_len != n) return 0;

    float_count = (size_t)(10 + ma_n) * (size_t)n;
    self->post_sg_storage =
        (float*)four_calloc(float_count, sizeof(float));
    if (!self->post_sg_storage) return 0;

    cursor = self->post_sg_storage;
    {
        float* last_gain = cursor; cursor += n;
        float* last_near = cursor; cursor += n;
        float* last_echo = cursor; cursor += n;
        float* ma_buf = cursor; cursor += (size_t)ma_n * n;
        float* near_s = cursor; cursor += n;
        float* weighted = cursor; cursor += n;
        float* min_gain = cursor; cursor += n;
        float* max_gain = cursor; cursor += n;
        float* raw_gain = cursor; cursor += n;
        float* gain = cursor; cursor += n;
        float* sum = cursor;

        suppression_gain_init(
            &self->post_sg, &self->post_sg_cfg, &self->post_sg_tun,
            last_gain, last_near, last_echo, ma_buf, near_s, weighted,
            min_gain, max_gain, raw_gain, gain, sum);
    }
    return 1;
}

FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg) {
    FourAecNrRes* self;
    AecConfig aec_cfg;
    MmseLsaConfig nr_cfg;
    int fft_size;
    int ch;
    int max_delay_samples;
    int k;

    if (!four_validate_config(cfg, &fft_size)) return NULL;

    self = (FourAecNrRes*)calloc(1, sizeof(*self));
    if (!self) return NULL;
    self->cfg = *cfg;
    self->sample_rate = cfg->sample_rate;
    self->fft_size = fft_size;
    self->hop_size = fft_size / 2;
    self->n_freqs = fft_size / 2 + 1;
    self->rate_factor = cfg->sample_rate / 16000;
    self->rng_state = FOUR_AEC_NR_RES_RNG_SEED;

    aec_config_from_preset(&aec_cfg, cfg->aec_preset, cfg->sample_rate);
    aec_cfg.fft_size = fft_size;
    if (cfg->filter_length > 0) aec_cfg.filter_length = cfg->filter_length;
    aec_cfg.enable_delay_est = 0;
    aec_cfg.enable_res = 0;
    aec_cfg.return_res_context = 1;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        if (aec_create(&self->lanes[ch], &aec_cfg) != 0) {
            four_aec_nr_res_destroy(self);
            return NULL;
        }
        self->initialized_lanes += 1;
    }

    delay_aec3_init(&self->shared_delay);

    nr_cfg = mmse_lsa_config_for_mode_grid(
        cfg->sample_rate, fft_size, cfg->nr_mode);
    nr_cfg.L = mmse_lsa_retime_frames(
        150, cfg->sample_rate, self->hop_size);
    nr_cfg.alpha_d = mmse_lsa_retime_alpha(
        0.95f, cfg->sample_rate, self->hop_size);
    nr_cfg.alpha_attack = mmse_lsa_retime_alpha(
        0.3f, cfg->sample_rate, self->hop_size);
    nr_cfg.alpha_decay = nr_cfg.alpha_g;
    self->nr = mmse_lsa_create(&nr_cfg);
    self->fft = fft_create(fft_size);
    if (!self->nr || !self->fft || !four_allocate_working_buffers(self) ||
        !four_init_post_sg(self)) {
        four_aec_nr_res_destroy(self);
        return NULL;
    }

    max_delay_samples =
        (int)ceilf(cfg->max_delay_ms * (float)cfg->sample_rate / 1000.0f);
    self->delay_ring_size = max_delay_samples + 2 * self->hop_size + 1;
    self->delay_ring = (float*)four_calloc(
        (size_t)self->delay_ring_size, sizeof(float));
    if (!self->delay_ring) {
        four_aec_nr_res_destroy(self);
        return NULL;
    }

    for (k = 0; k < fft_size; ++k) {
        self->synth_window[k] = sqrtf(
            0.5f * (1.0f - cosf(2.0f * M_PI_F * (float)k /
                                (float)fft_size)));
    }
    return self;
}

static void four_reset_post_sg(FourAecNrRes* self) {
    int n;
    int ma_n;
    float* cursor;
    float* last_gain;
    float* last_near;
    float* last_echo;
    float* ma_buf;
    float* near_s;
    float* weighted;
    float* min_gain;
    float* max_gain;
    float* raw_gain;
    float* gain;
    float* sum;

    if (!self || !self->post_sg_storage) return;
    n = self->n_freqs;
    ma_n = self->post_sg_cfg.nearend_smoother_n;
    cursor = self->post_sg_storage;
    last_gain = cursor; cursor += n;
    last_near = cursor; cursor += n;
    last_echo = cursor; cursor += n;
    ma_buf = cursor; cursor += (size_t)ma_n * n;
    near_s = cursor; cursor += n;
    weighted = cursor; cursor += n;
    min_gain = cursor; cursor += n;
    max_gain = cursor; cursor += n;
    raw_gain = cursor; cursor += n;
    gain = cursor; cursor += n;
    sum = cursor;

    memset(self->post_sg_storage, 0,
           (size_t)(10 + ma_n) * (size_t)n * sizeof(float));
    suppression_gain_init(
        &self->post_sg, &self->post_sg_cfg, &self->post_sg_tun,
        last_gain, last_near, last_echo, ma_buf, near_s, weighted,
        min_gain, max_gain, raw_gain, gain, sum);
}

void four_aec_nr_res_reset(FourAecNrRes* self) {
    int ch;
    if (!self) return;

    delay_aec3_reset(&self->shared_delay);
    for (ch = 0; ch < self->initialized_lanes; ++ch) {
        aec_reset(&self->lanes[ch]);
    }
    if (self->nr) mmse_lsa_reset(self->nr);
    four_reset_post_sg(self);

    if (self->delay_ring) {
        memset(self->delay_ring, 0,
               (size_t)self->delay_ring_size * sizeof(float));
    }
    if (self->linear_interleaved) {
        memset(self->linear_interleaved, 0,
               (size_t)self->hop_size * FOUR_AEC_NR_RES_CHANNELS *
               sizeof(float));
    }
    if (self->aligned_ref) {
        memset(self->aligned_ref, 0,
               (size_t)self->hop_size * sizeof(float));
    }
    if (self->ola) {
        memset(self->ola, 0, (size_t)self->fft_size * sizeof(float));
    }
    if (self->ifft_buffer) {
        memset(self->ifft_buffer, 0,
               (size_t)self->fft_size * sizeof(float));
    }

    self->delay_samples_seen = 0;
    self->accepted_delay = 0;
    self->delay_phase = 0;
    self->delay_calls = 0;
    memset(&self->last_delay, 0, sizeof(self->last_delay));
    self->rng_state = FOUR_AEC_NR_RES_RNG_SEED;
    self->near_hang = 0;
    self->next_frame = 0;
    self->generation += 1;
    self->pending = 0;
    memset(&self->pending_token, 0, sizeof(self->pending_token));
}

void four_aec_nr_res_destroy(FourAecNrRes* self) {
    int ch;
    if (!self) return;

    if (self->nr) mmse_lsa_destroy(self->nr);
    if (self->fft) fft_destroy(self->fft);
    for (ch = 0; ch < self->initialized_lanes; ++ch) {
        aec_destroy(&self->lanes[ch]);
    }
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        four_free_lane_snapshot(&self->snapshots[ch]);
    }

    free(self->post_sg_storage);
    free(self->linear_interleaved);
    free(self->mic_lane);
    free(self->lane_out);
    free(self->aligned_ref);
    free(self->render_i16);
    free(self->delay_ring);
    free(self->delay_capture);
    free(self->delay_render);
    free(self->fused_error);
    free(self->fused_echo);
    free(self->fused_near);
    free(self->fused_far);
    free(self->output_spec);
    free(self->fused_r2);
    free(self->fused_comfort);
    free(self->post_near_power);
    free(self->nr_gain);
    free(self->total_gain);
    free(self->extra_noise);
    free(self->error_power);
    free(self->synth_window);
    free(self->ifft_buffer);
    free(self->ola);
    free(self);
}

static FourAecNrResDelayState four_update_shared_delay(
    FourAecNrRes* self,
    const float* capture,
    const float* render) {
    FourAecNrResDelayState state;
    const float* capture_16k = capture;
    const float* render_16k = render;
    int delay_hop = self->hop_size;
    int emitted;
    int estimated_16k;
    int estimated;
    int eligible;
    int i;

    if (self->rate_factor > 1) {
        int count = 0;
        for (i = 0; i < self->hop_size; ++i) {
            if (((i + self->delay_phase) % self->rate_factor) == 0) {
                self->delay_capture[count] = capture[i];
                self->delay_render[count] = render[i];
                count += 1;
            }
        }
        self->delay_phase =
            (self->delay_phase + self->hop_size) % self->rate_factor;
        capture_16k = self->delay_capture;
        render_16k = self->delay_render;
        delay_hop = count;
    }

    emitted = delay_hop > 0
        ? delay_aec3_accumulate(
              &self->shared_delay, capture_16k, render_16k, delay_hop)
        : 0;
    (void)emitted;
    self->delay_calls += 1;

    estimated_16k = delay_aec3_estimated_delay(&self->shared_delay);
    estimated = estimated_16k >= 0
        ? estimated_16k * self->rate_factor
        : -1;
    eligible = estimated >= 0 &&
               delay_aec3_is_solid(&self->shared_delay) &&
               delay_aec3_n_updates(&self->shared_delay) >= 3;

    memset(&state, 0, sizeof(state));
    state.changed = eligible && estimated != self->accepted_delay;
    if (eligible) self->accepted_delay = estimated;
    state.delay_samples = self->accepted_delay;
    state.confidence = delay_aec3_confidence(&self->shared_delay);
    state.solid = delay_aec3_is_solid(&self->shared_delay);
    state.estimator_calls = self->delay_calls;
    state.estimator_updates = delay_aec3_n_updates(&self->shared_delay);
    self->last_delay = state;
    return state;
}

static int four_align_render(FourAecNrRes* self, const float* render,
                             int delay_samples) {
    int i;
    uint64_t start;
    if (delay_samples < 0 ||
        delay_samples >= self->delay_ring_size - self->hop_size) return 0;

    start = self->delay_samples_seen;
    for (i = 0; i < self->hop_size; ++i) {
        uint64_t absolute = start + (uint64_t)i;
        self->delay_ring[absolute % (uint64_t)self->delay_ring_size] =
            render[i];
    }
    for (i = 0; i < self->hop_size; ++i) {
        uint64_t absolute = start + (uint64_t)i;
        if (absolute >= (uint64_t)delay_samples) {
            uint64_t source = absolute - (uint64_t)delay_samples;
            self->aligned_ref[i] =
                self->delay_ring[source %
                                 (uint64_t)self->delay_ring_size];
        } else {
            self->aligned_ref[i] = 0.0f;
        }
    }
    self->delay_samples_seen += (uint64_t)self->hop_size;
    return 1;
}

static int four_snapshot_context(FourAecLaneSnapshot* dst,
                                 const AecResContext* src,
                                 int n_freqs) {
    if (!dst || !src || src->n_freqs != n_freqs ||
        !src->error_spec || !src->echo_spec || !src->far_spec ||
        !src->near_spec || !src->r2 || !src->comfort_noise) return 0;

    memcpy(dst->error_spec, src->error_spec,
           (size_t)n_freqs * sizeof(Complex));
    memcpy(dst->echo_spec, src->echo_spec,
           (size_t)n_freqs * sizeof(Complex));
    memcpy(dst->far_spec, src->far_spec,
           (size_t)n_freqs * sizeof(Complex));
    memcpy(dst->near_spec, src->near_spec,
           (size_t)n_freqs * sizeof(Complex));
    memcpy(dst->r2, src->r2, (size_t)n_freqs * sizeof(float));
    memcpy(dst->comfort_noise, src->comfort_noise,
           (size_t)n_freqs * sizeof(float));

    dst->far_power = src->far_power;
    dst->erle_factor = src->erle_factor;
    dst->dt_indicator = src->dt_indicator;
    dst->divergence = src->divergence;
    dst->saturation_level = src->saturation_level;
    dst->erl_estimate = src->erl_estimate;
    dst->filter_converged = src->filter_converged;
    return 1;
}

int four_aec_nr_res_process_pre(
    FourAecNrRes* self,
    const float* microphones_interleaved,
    const float* ref,
    FourAecNrResPreFrame* out) {
    FourAecNrResDelayState delay;
    int hop;
    int ch;
    int i;

    if (!self || !microphones_interleaved || !ref || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (self->pending) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;

    hop = self->hop_size;
    if (!four_inputs_finite(
            microphones_interleaved,
            (size_t)hop * FOUR_AEC_NR_RES_CHANNELS) ||
        !four_inputs_finite(ref, (size_t)hop))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    for (i = 0; i < hop; ++i) {
        self->mic_lane[i] =
            microphones_interleaved[
                i * FOUR_AEC_NR_RES_CHANNELS +
                self->cfg.capture_proxy_channel];
    }
    delay = four_update_shared_delay(self, self->mic_lane, ref);
    if (!four_align_render(self, ref, delay.delay_samples)) {
        four_aec_nr_res_reset(self);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (delay.changed) {
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            aec_reset(&self->lanes[ch]);
        }
    }

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        AecResContext context;
        for (i = 0; i < hop; ++i) {
            self->mic_lane[i] =
                microphones_interleaved[
                    i * FOUR_AEC_NR_RES_CHANNELS + ch];
        }
        aec_process(
            &self->lanes[ch], self->mic_lane, self->aligned_ref,
            self->lane_out);
        aec_get_res_context(&self->lanes[ch], &context);
        if (!four_snapshot_context(
                &self->snapshots[ch], &context, self->n_freqs)) {
            four_aec_nr_res_reset(self);
            return FOUR_AEC_NR_RES_DSP_ERROR;
        }
        for (i = 0; i < hop; ++i) {
            self->linear_interleaved[
                i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                self->lane_out[i];
        }
    }

    self->pending_token.frame_index = self->next_frame;
    self->pending_token.generation = self->generation;
    self->pending_token.owner_cookie = (uintptr_t)self;
    self->next_frame += 1;
    self->pending = 1;

    out->token = self->pending_token;
    out->delay = delay;
    out->hop_size = self->hop_size;
    out->n_channels = FOUR_AEC_NR_RES_CHANNELS;
    out->linear_interleaved = self->linear_interleaved;
    return FOUR_AEC_NR_RES_OK;
}

static int four_token_matches(const FourAecNrRes* self,
                              const FourAecNrResFrameToken* token) {
    return self && token && self->pending &&
           token->frame_index == self->pending_token.frame_index &&
           token->generation == self->pending_token.generation &&
           token->owner_cookie == self->pending_token.owner_cookie &&
           token->owner_cookie == (uintptr_t)self;
}

static int four_validate_weights(const FourAecNrRes* self,
                                 const Complex* weights) {
    size_t count;
    size_t i;
    float sum = 0.0f;
    if (!self || !weights) return 0;
    count = (size_t)FOUR_AEC_NR_RES_CHANNELS *
            (size_t)self->n_freqs;
    for (i = 0; i < count; ++i) {
        if (!isfinite(weights[i].r) || !isfinite(weights[i].i))
            return 0;
        sum += fabsf(weights[i].r) + fabsf(weights[i].i);
    }
    return isfinite(sum) && sum > 1e-12f;
}

static int four_fuse_contexts(FourAecNrRes* self,
                              const Complex* weights,
                              int* all_converged,
                              float* max_dt,
                              float* max_saturation,
                              float* far_power) {
    int n = self->n_freqs;
    int k;
    int ch;
    float base_far_power = self->snapshots[0].far_power;

    *all_converged = 1;
    *max_dt = self->snapshots[0].dt_indicator;
    *max_saturation = self->snapshots[0].saturation_level;
    *far_power = base_far_power;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        float delta = fabsf(
            self->snapshots[ch].far_power - base_far_power);
        float scale = 1.0f + fabsf(base_far_power);
        if (delta > 1e-5f * scale) return 0;
        if (!self->snapshots[ch].filter_converged)
            *all_converged = 0;
        if (self->snapshots[ch].dt_indicator > *max_dt)
            *max_dt = self->snapshots[ch].dt_indicator;
        if (self->snapshots[ch].saturation_level > *max_saturation)
            *max_saturation = self->snapshots[ch].saturation_level;
    }

    for (k = 0; k < n; ++k) {
        Complex error = {0.0f, 0.0f};
        Complex echo = {0.0f, 0.0f};
        Complex near = {0.0f, 0.0f};
        Complex residual = {0.0f, 0.0f};
        float comfort = 0.0f;
        Complex far0 = self->snapshots[0].far_spec[k];

        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            Complex w = weights[(size_t)ch * n + k];
            Complex echo_ch = self->snapshots[ch].echo_spec[k];
            float echo_mag = sqrtf(
                echo_ch.r * echo_ch.r + echo_ch.i * echo_ch.i);
            float residual_amp = sqrtf(
                fmaxf(self->snapshots[ch].r2[k], 0.0f));
            Complex residual_ch;
            float w2;

            if (!four_complex_close(
                    self->snapshots[ch].far_spec[k], far0)) return 0;

            four_complex_accumulate(
                &error, four_complex_mul(
                    w, self->snapshots[ch].error_spec[k]));
            four_complex_accumulate(
                &echo, four_complex_mul(w, echo_ch));
            four_complex_accumulate(
                &near, four_complex_mul(
                    w, self->snapshots[ch].near_spec[k]));

            if (echo_mag > 1e-20f) {
                residual_ch.r = residual_amp * echo_ch.r / echo_mag;
                residual_ch.i = residual_amp * echo_ch.i / echo_mag;
            } else {
                residual_ch.r = residual_amp;
                residual_ch.i = 0.0f;
            }
            four_complex_accumulate(
                &residual, four_complex_mul(w, residual_ch));

            w2 = w.r * w.r + w.i * w.i;
            comfort += w2 *
                fmaxf(self->snapshots[ch].comfort_noise[k], 0.0f);
        }

        self->fused_error[k] = error;
        self->fused_echo[k] = echo;
        self->fused_near[k] = near;
        self->fused_far[k] = far0;
        self->fused_r2[k] = fmaxf(
            residual.r * residual.r + residual.i * residual.i, 0.0f);
        self->fused_comfort[k] = fmaxf(comfort, 0.0f);
    }
    return 1;
}

static int four_run_post_res_and_nr(
    FourAecNrRes* self,
    int all_converged,
    float max_dt,
    float max_saturation,
    float far_power,
    float* out) {
    const float* res_gain;
    const float* nr_extra;
    int n = self->n_freqs;
    int hop = self->hop_size;
    int fft = self->fft_size;
    int k;
    float nf_eff;

    for (k = 0; k < n; ++k) {
        float er = self->fused_error[k].r;
        float ei = self->fused_error[k].i;
        float nr = self->fused_near[k].r;
        float ni = self->fused_near[k].i;
        float e2 = er * er + ei * ei;
        float n2 = nr * nr + ni * ni;
        self->error_power[k] = e2;
        self->post_near_power[k] =
            (all_converged ? fminf(e2, n2) : n2) *
            FOUR_AEC_NR_RES_PSD_SCALE;
        self->extra_noise[k] =
            self->fused_r2[k] / FOUR_AEC_NR_RES_PSD_SCALE;
    }
    for (k = 0; k < hop; ++k) {
        self->render_i16[k] = self->aligned_ref[k] * 32768.0f;
    }

    if (self->post_sg.initial_state && all_converged)
        suppression_gain_set_initial_state(&self->post_sg, 0);
    self->post_sg.dt_protect_active = max_dt > 0.2f;
    res_gain = suppression_gain_get_gain(
        &self->post_sg,
        self->post_near_power,
        self->fused_r2,
        self->fused_r2,
        self->fused_comfort,
        self->render_i16,
        delay_aec3_has_clockdrift(&self->shared_delay),
        max_saturation > 0.5f);
    if (!res_gain) return 0;

    nr_extra = self->cfg.legacy_amin ? NULL : self->extra_noise;
    if (mmse_lsa_process_gain(
            self->nr, self->fused_error, nr_extra, self->nr_gain) != 0)
        return 0;

    for (k = 0; k < n; ++k) {
        self->total_gain[k] =
            fminf(self->nr_gain[k], res_gain[k]);
    }

    nf_eff = FOUR_AEC_NR_RES_NE_FLOOR;
    if (!self->cfg.legacy_amin) {
        int far_active =
            far_power > FOUR_AEC_NR_RES_FAR_GATE_THRESHOLD;
        float near_mean = 0.0f;
        int near_active;
        for (k = 0; k < n; ++k) near_mean += self->error_power[k];
        near_mean /= (float)n;
        if (near_mean > FOUR_AEC_NR_RES_NEAR_GATE_THRESHOLD)
            self->near_hang = FOUR_AEC_NR_RES_NEAR_HANGOVER;
        near_active = self->near_hang > 0;
        if (self->near_hang > 0) self->near_hang -= 1;
        nf_eff = (!far_active && near_active)
            ? FOUR_AEC_NR_RES_NE_FLOOR
            : FOUR_AEC_NR_RES_NE_FLOOR_FAR_ACTIVE;
    }

    for (k = 0; k < n; ++k) {
        float echo_fraction =
            self->extra_noise[k] / (self->error_power[k] + 1e-12f);
        float no_echo;
        float lift;
        if (echo_fraction < 0.0f) echo_fraction = 0.0f;
        if (echo_fraction > 1.0f) echo_fraction = 1.0f;
        no_echo = res_gain[k] * (1.0f - echo_fraction);
        lift = nf_eff * no_echo;
        self->total_gain[k] =
            (1.0f - lift) * self->total_gain[k] + lift;
        self->output_spec[k].r =
            self->fused_error[k].r * self->total_gain[k];
        self->output_spec[k].i =
            self->fused_error[k].i * self->total_gain[k];
    }

    if (self->cfg.enable_cng) {
        for (k = 1; k < n - 1; ++k) {
            float n_amp =
                self->fused_comfort[k] / FOUR_AEC_NR_RES_PSD_SCALE;
            float gain2 = 1.0f - res_gain[k] * res_gain[k];
            float amplitude;
            n_amp = n_amp > 0.0f ? sqrtf(n_amp) : 0.0f;
            gain2 = gain2 > 0.0f ? sqrtf(gain2) : 0.0f;
            amplitude = n_amp * gain2;
            self->output_spec[k].r +=
                amplitude * four_rng_gauss(self);
            self->output_spec[k].i +=
                amplitude * four_rng_gauss(self);
        }
    }

    fft_inverse(self->fft, self->output_spec, self->ifft_buffer);
    for (k = 0; k < fft; ++k) {
        self->ola[k] +=
            self->ifft_buffer[k] * self->synth_window[k];
    }
    memcpy(out, self->ola, (size_t)hop * sizeof(float));
    memmove(
        self->ola, self->ola + hop,
        (size_t)(fft - hop) * sizeof(float));
    memset(
        self->ola + (fft - hop), 0,
        (size_t)hop * sizeof(float));
    return four_inputs_finite(out, (size_t)hop);
}

int four_aec_nr_res_process_post(
    FourAecNrRes* self,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    float* out) {
    int all_converged;
    float max_dt;
    float max_saturation;
    float far_power;

    if (!self || !token || !weights || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (!four_token_matches(self, token))
        return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    if (!four_validate_weights(self, weights))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    if (!four_fuse_contexts(
            self, weights, &all_converged, &max_dt,
            &max_saturation, &far_power)) {
        four_aec_nr_res_reset(self);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (!four_run_post_res_and_nr(
            self, all_converged, max_dt, max_saturation,
            far_power, out)) {
        four_aec_nr_res_reset(self);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    self->pending = 0;
    memset(&self->pending_token, 0, sizeof(self->pending_token));
    return FOUR_AEC_NR_RES_OK;
}

int four_aec_nr_res_hop_size(const FourAecNrRes* self) {
    return self ? self->hop_size : -1;
}

int four_aec_nr_res_fft_size(const FourAecNrRes* self) {
    return self ? self->fft_size : -1;
}

int four_aec_nr_res_n_freqs(const FourAecNrRes* self) {
    return self ? self->n_freqs : -1;
}

int four_aec_nr_res_sample_rate(const FourAecNrRes* self) {
    return self ? self->sample_rate : -1;
}

int four_aec_nr_res_matched_filter_count(const FourAecNrRes* self) {
    return self ? 1 : 0;
}

int four_aec_nr_res_linear_aec_count(const FourAecNrRes* self) {
    return self ? FOUR_AEC_NR_RES_CHANNELS : 0;
}

int four_aec_nr_res_nr_count(const FourAecNrRes* self) {
    return self ? 1 : 0;
}

int four_aec_nr_res_post_res_count(const FourAecNrRes* self) {
    return self ? 1 : 0;
}
