/**
 * aec_4ch/4aec_nr_res.c
 *
 * Synchronous C implementation of:
 *
 *   one shared DelayAec3 -> one aligned render
 *   -> four independent linear AEC filters
 *   -> externally supplied effective beamformer weights
 *   -> one coherent post-beam RES + one NR + one iFFT/OLA
 *
 * Like pipelines/audio_pipeline.c, this is a pool-first core with two
 * construction paths:
 *
 *   four_aec_nr_res_get_mem_requirements() + four_aec_nr_res_init_ex()
 *       caller-owned pool; zero heap from init through destroy
 *
 *   four_aec_nr_res_create()
 *       heap convenience wrapper over that same pool-first implementation
 *
 * Both paths call the same process_pre()/process_post() implementation and
 * therefore have identical DSP arithmetic. Neither process call allocates.
 */

#include "4aec_nr_res.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "aec3_balanced_config.h"
#include "fft_wrapper.h"
#include "mem_align.h"
#include "suppression_gain.h"

#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

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

/* ============================================================================
 * Instance
 * ========================================================================== */

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

    Aec* lanes[FOUR_AEC_NR_RES_CHANNELS];
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

    void* owned_heap;
    size_t pool_size;
    int destroyed;
};

typedef struct FourPoolCursor {
    uint8_t* ptr;
    size_t remaining;
} FourPoolCursor;

/* ============================================================================
 * Config validation and small DSP helpers
 * ========================================================================== */

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

/* Pool cursor primitive: same ALIGN16 discipline as audio_pipeline.c. */
static void* four_carve(FourPoolCursor* cursor,
                        size_t count,
                        size_t element_size) {
    size_t raw;
    size_t aligned;
    void* out;
    if (!cursor || count == 0 || element_size == 0) return NULL;
    raw = ck_mul_size(count, element_size);
    aligned = ck_align16_size(raw);
    if (MEM_SIZE_INVALID(raw) || MEM_SIZE_INVALID(aligned) ||
        aligned > cursor->remaining) return NULL;
    out = cursor->ptr;
    cursor->ptr += aligned;
    cursor->remaining -= aligned;
    return out;
}

static int four_carve_lane_snapshot(FourAecLaneSnapshot* s,
                                    FourPoolCursor* cursor,
                                    int n_freqs) {
    s->error_spec =
        (Complex*)four_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->echo_spec =
        (Complex*)four_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->far_spec =
        (Complex*)four_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->near_spec =
        (Complex*)four_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->r2 = (float*)four_carve(
        cursor, (size_t)n_freqs, sizeof(float));
    s->comfort_noise = (float*)four_carve(
        cursor, (size_t)n_freqs, sizeof(float));
    return s->error_spec && s->echo_spec && s->far_spec && s->near_spec &&
           s->r2 && s->comfort_noise;
}

/* Per-hop validation/math helpers. */
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

/* ============================================================================
 * Pool sizing and carving
 *
 * This is the 4-channel counterpart of audio_pipeline.c's
 * pipeline_pool_size()/pipeline_build(): all module objects and every
 * scratch/state buffer are 16-byte-aligned segments in one pool.
 * ========================================================================== */

static int four_carve_working_buffers(FourAecNrRes* self,
                                      FourPoolCursor* cursor) {
    int ch;
    int n = self->n_freqs;
    int hop = self->hop_size;
    int fft = self->fft_size;
    size_t linear_count = (size_t)hop * FOUR_AEC_NR_RES_CHANNELS;

    self->linear_interleaved =
        (float*)four_carve(cursor, linear_count, sizeof(float));
    self->mic_lane =
        (float*)four_carve(cursor, (size_t)hop, sizeof(float));
    self->lane_out =
        (float*)four_carve(cursor, (size_t)hop, sizeof(float));
    self->aligned_ref =
        (float*)four_carve(cursor, (size_t)hop, sizeof(float));
    self->render_i16 =
        (float*)four_carve(cursor, (size_t)hop, sizeof(float));
    self->delay_capture =
        (float*)four_carve(cursor, (size_t)hop, sizeof(float));
    self->delay_render =
        (float*)four_carve(cursor, (size_t)hop, sizeof(float));

    self->fused_error =
        (Complex*)four_carve(cursor, (size_t)n, sizeof(Complex));
    self->fused_echo =
        (Complex*)four_carve(cursor, (size_t)n, sizeof(Complex));
    self->fused_near =
        (Complex*)four_carve(cursor, (size_t)n, sizeof(Complex));
    self->fused_far =
        (Complex*)four_carve(cursor, (size_t)n, sizeof(Complex));
    self->output_spec =
        (Complex*)four_carve(cursor, (size_t)n, sizeof(Complex));

    self->fused_r2 =
        (float*)four_carve(cursor, (size_t)n, sizeof(float));
    self->fused_comfort =
        (float*)four_carve(cursor, (size_t)n, sizeof(float));
    self->post_near_power =
        (float*)four_carve(cursor, (size_t)n, sizeof(float));
    self->nr_gain =
        (float*)four_carve(cursor, (size_t)n, sizeof(float));
    self->total_gain =
        (float*)four_carve(cursor, (size_t)n, sizeof(float));
    self->extra_noise =
        (float*)four_carve(cursor, (size_t)n, sizeof(float));
    self->error_power =
        (float*)four_carve(cursor, (size_t)n, sizeof(float));

    self->synth_window =
        (float*)four_carve(cursor, (size_t)fft, sizeof(float));
    self->ifft_buffer =
        (float*)four_carve(cursor, (size_t)fft, sizeof(float));
    self->ola =
        (float*)four_carve(cursor, (size_t)fft, sizeof(float));

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        if (!four_carve_lane_snapshot(
                &self->snapshots[ch], cursor, n)) return 0;
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

static int four_init_post_sg(FourAecNrRes* self,
                             FourPoolCursor* pool_cursor) {
    int n = self->n_freqs;
    int ma_n;
    size_t float_count;
    float* cursor;

    self->post_sg_cfg = self->lanes[0]->a3_sg.cfg;
    self->post_sg_tun = self->lanes[0]->a3_sg.tun;
    ma_n = self->post_sg_cfg.nearend_smoother_n;
    if (ma_n < 1 || self->post_sg_cfg.n_bins != n ||
        self->post_sg_tun.table_len != n) return 0;

    float_count = (size_t)(10 + ma_n) * (size_t)n;
    self->post_sg_storage =
        (float*)four_carve(pool_cursor, float_count, sizeof(float));
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

static int four_derive_configs(
    const FourAecNrResConfig* cfg,
    AecConfig* aec_cfg,
    MmseLsaConfig* nr_cfg,
    int* fft_size,
    int* hop_size,
    int* n_freqs,
    int* post_ma_n,
    int* delay_ring_size) {
    const Aec3BalancedRateDims* rate_dims;
    int max_delay_samples;
    if (!cfg || !aec_cfg || !nr_cfg || !fft_size || !hop_size ||
        !n_freqs || !post_ma_n || !delay_ring_size ||
        !four_validate_config(cfg, fft_size)) return 0;

    *hop_size = *fft_size / 2;
    *n_freqs = *fft_size / 2 + 1;
    rate_dims = aec3b_rate_cfg(cfg->sample_rate, *fft_size);
    if (!rate_dims || rate_dims->sg_nearend_smoother_n < 1) return 0;
    *post_ma_n = rate_dims->sg_nearend_smoother_n;

    max_delay_samples =
        (int)ceilf(cfg->max_delay_ms * (float)cfg->sample_rate / 1000.0f);
    *delay_ring_size = max_delay_samples + 2 * *hop_size + 1;
    if (*delay_ring_size <= 0) return 0;

    aec_config_from_preset(aec_cfg, cfg->aec_preset, cfg->sample_rate);
    aec_cfg->fft_size = *fft_size;
    if (cfg->filter_length > 0)
        aec_cfg->filter_length = cfg->filter_length;
    aec_cfg->enable_delay_est = 0;
    aec_cfg->enable_res = 0;
    aec_cfg->return_res_context = 1;

    *nr_cfg = mmse_lsa_config_for_mode_grid(
        cfg->sample_rate, *fft_size, cfg->nr_mode);
    nr_cfg->L = mmse_lsa_retime_frames(
        150, cfg->sample_rate, *hop_size);
    nr_cfg->alpha_d = mmse_lsa_retime_alpha(
        0.95f, cfg->sample_rate, *hop_size);
    nr_cfg->alpha_attack = mmse_lsa_retime_alpha(
        0.3f, cfg->sample_rate, *hop_size);
    nr_cfg->alpha_decay = nr_cfg->alpha_g;
    return 1;
}

static size_t four_wrapper_storage_size(
    int hop, int fft, int n, int post_ma_n, int delay_ring_size) {
    size_t total = 0;
    int ch;
    int i;

    total = ck_field_size(
        total,
        ck_mul_size((size_t)hop, FOUR_AEC_NR_RES_CHANNELS),
        sizeof(float));                                      /* linear */
    for (i = 0; i < 6; ++i)
        total = ck_field_size(total, (size_t)hop, sizeof(float));

    for (i = 0; i < 5; ++i)
        total = ck_field_size(total, (size_t)n, sizeof(Complex));
    for (i = 0; i < 7; ++i)
        total = ck_field_size(total, (size_t)n, sizeof(float));
    for (i = 0; i < 3; ++i)
        total = ck_field_size(total, (size_t)fft, sizeof(float));

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (i = 0; i < 4; ++i)
            total = ck_field_size(
                total, (size_t)n, sizeof(Complex));
        for (i = 0; i < 2; ++i)
            total = ck_field_size(
                total, (size_t)n, sizeof(float));
    }

    total = ck_field_size(
        total,
        ck_mul_size((size_t)(10 + post_ma_n), (size_t)n),
        sizeof(float));                                      /* post SG */
    total = ck_field_size(
        total, (size_t)delay_ring_size, sizeof(float));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

static uint32_t four_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "kiss") == 0)
        return FOUR_AEC_NR_RES_BACKEND_KISS;
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0)
        return FOUR_AEC_NR_RES_BACKEND_NE10;
    return 0u;
}

static uint32_t four_fnv1a(const char* text, uint32_t hash) {
    while (*text) {
        hash ^= (uint32_t)(unsigned char)*text++;
        hash *= 16777619u;
    }
    return hash;
}

static uint32_t four_build_flags_hash(void) {
    uint32_t hash = 2166136261u;
    hash = four_fnv1a(AUDIO_PIPELINE_BACKEND_STR, hash);
    hash = four_fnv1a(
        "|carve:self,aec0,aec1,aec2,aec3,nr,fft,linear,hop6,"
        "complex5,float7,fftfloat3,snapshot4x6,postsg,delayring",
        hash);
    hash = four_fnv1a("|align16", hash);
    return hash;
}

/* ============================================================================
 * Public memory query and construction
 *
 * Same order as the mono board flow:
 *   defaults -> get_mem_requirements -> init_ex
 * and the desktop convenience flow:
 *   defaults -> create
 * ========================================================================== */

int four_aec_nr_res_get_mem_breakdown(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemBreakdown* out) {
    AecConfig aec_cfg;
    MmseLsaConfig nr_cfg;
    size_t aec_one;
    size_t aec_total = 0;
    size_t nr_bytes;
    size_t fft_bytes;
    size_t wrapper_storage;
    size_t wrapper_bytes;
    size_t total;
    int fft;
    int hop;
    int n;
    int post_ma_n;
    int delay_ring_size;
    int ch;

    if (!out || !four_derive_configs(
            cfg, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size)) return -1;

    aec_one = aec_get_mem_size(&aec_cfg);
    nr_bytes = mmse_lsa_get_mem_size(&nr_cfg);
    fft_bytes = fft_get_mem_size(fft);
    wrapper_storage = four_wrapper_storage_size(
        hop, fft, n, post_ma_n, delay_ring_size);
    if (aec_one == 0 || nr_bytes == 0 || fft_bytes == 0 ||
        wrapper_storage == 0) return -1;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        aec_total = ck_add_size(aec_total, ck_align16_size(aec_one));
    wrapper_bytes = ck_add_size(
        ck_align16_size(sizeof(FourAecNrRes)), wrapper_storage);
    total = wrapper_bytes;
    total = ck_add_size(total, aec_total);
    total = ck_add_size(total, ck_align16_size(nr_bytes));
    total = ck_add_size(total, ck_align16_size(fft_bytes));
    if (MEM_SIZE_INVALID(total)) return -1;

    memset(out, 0, sizeof(*out));
    out->aec_bytes = aec_total;
    out->nr_bytes = nr_bytes;
    out->fft_bytes = fft_bytes;
    out->wrapper_bytes = wrapper_bytes;
    out->total_bytes = total;
    out->hop_size = hop;
    out->fft_size = fft;
    out->n_freqs = n;
    return 0;
}

int four_aec_nr_res_get_mem_requirements(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemReq* out) {
    FourAecNrResMemBreakdown breakdown;
    uint32_t backend;
    if (!out ||
        four_aec_nr_res_get_mem_breakdown(cfg, &breakdown) != 0)
        return -1;
    backend = four_backend_id();
    if (backend == 0u) return -1;

    memset(out, 0, sizeof(*out));
    out->descriptor_version = FOUR_AEC_NR_RES_DESCRIPTOR_VERSION;
    out->layout_version = FOUR_AEC_NR_RES_LAYOUT_VERSION;
    out->backend_id = backend;
    out->build_flags_hash = four_build_flags_hash();
    out->alignment = 16u;
    out->bytes = (uint64_t)breakdown.total_bytes;
    return 0;
}

static int four_build_from_pool(
    FourAecNrRes* self,
    FourPoolCursor* cursor,
    const AecConfig* aec_cfg,
    const MmseLsaConfig* nr_cfg) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    size_t nr_bytes = mmse_lsa_get_mem_size(nr_cfg);
    size_t fft_bytes = fft_get_mem_size(self->fft_size);
    int ch;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        void* lane_pool = four_carve(cursor, 1, aec_bytes);
        if (!lane_pool) return 0;
        self->lanes[ch] = aec_init(lane_pool, aec_bytes, aec_cfg);
        if (!self->lanes[ch]) return 0;
        self->initialized_lanes += 1;
    }

    {
        void* nr_pool = four_carve(cursor, 1, nr_bytes);
        void* fft_pool = four_carve(cursor, 1, fft_bytes);
        if (!nr_pool || !fft_pool) return 0;
        self->nr = mmse_lsa_init(nr_pool, nr_bytes, nr_cfg);
        self->fft = fft_init(fft_pool, fft_bytes, self->fft_size);
        if (!self->nr || !self->fft) return 0;
    }

    if (!four_carve_working_buffers(self, cursor) ||
        !four_init_post_sg(self, cursor)) return 0;
    self->delay_ring = (float*)four_carve(
        cursor, (size_t)self->delay_ring_size, sizeof(float));
    if (!self->delay_ring) return 0;
    return 1;
}

FourAecNrRes* four_aec_nr_res_init_ex(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg,
    const FourAecNrResMemReq* expected) {
    FourAecNrResConfig cfg_copy;
    FourAecNrResMemReq current;
    FourAecNrRes* self;
    FourPoolCursor cursor;
    AecConfig aec_cfg;
    MmseLsaConfig nr_cfg;
    int fft;
    int hop;
    int n;
    int post_ma_n;
    int delay_ring_size;
    int k;

    if (!mem || !cfg) return NULL;
    cfg_copy = *cfg;
    if (four_aec_nr_res_get_mem_requirements(
            &cfg_copy, &current) != 0) return NULL;

    if (expected) {
        if (expected->descriptor_version != current.descriptor_version ||
            expected->layout_version != current.layout_version ||
            expected->backend_id != current.backend_id ||
            expected->build_flags_hash != current.build_flags_hash ||
            expected->alignment != current.alignment ||
            expected->reserved != 0u ||
            expected->bytes < current.bytes)
            return NULL;
    }
    if (!MEM_IS_ALIGNED16(mem) ||
        current.bytes > (uint64_t)SIZE_MAX ||
        (uint64_t)bytes < current.bytes) return NULL;
    if (!four_derive_configs(
            &cfg_copy, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size)) return NULL;

    memset(mem, 0, (size_t)current.bytes);
    self = (FourAecNrRes*)mem;
    self->cfg = cfg_copy;
    self->sample_rate = cfg_copy.sample_rate;
    self->fft_size = fft;
    self->hop_size = hop;
    self->n_freqs = n;
    self->rate_factor = cfg_copy.sample_rate / 16000;
    self->delay_ring_size = delay_ring_size;
    self->rng_state = FOUR_AEC_NR_RES_RNG_SEED;
    self->pool_size = (size_t)current.bytes;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(*self));
    cursor.remaining =
        (size_t)current.bytes - ALIGN16(sizeof(*self));
    if (!four_build_from_pool(
            self, &cursor, &aec_cfg, &nr_cfg) ||
        cursor.remaining != 0) return NULL;

    delay_aec3_init(&self->shared_delay);
    for (k = 0; k < fft; ++k) {
        self->synth_window[k] = sqrtf(
            0.5f * (1.0f - cosf(
                2.0f * M_PI_F * (float)k / (float)fft)));
    }

    for (k = 0; k < FOUR_AEC_NR_RES_CHANNELS; ++k) {
        AecResContext context;
        aec_get_res_context(self->lanes[k], &context);
        if (context.hop_size != hop || context.n_freqs != n)
            return NULL;
    }
    if (mmse_lsa_get_hop_size(self->nr) != hop ||
        mmse_lsa_get_n_freqs(self->nr) != n ||
        fft_get_n_freqs(self->fft) != n) return NULL;
    return self;
}

FourAecNrRes* four_aec_nr_res_init(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg) {
    return four_aec_nr_res_init_ex(mem, bytes, cfg, NULL);
}

FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg) {
    FourAecNrResMemReq requirement;
    FourAecNrRes* self;
    void* pool = NULL;
    if (!cfg ||
        four_aec_nr_res_get_mem_requirements(
            cfg, &requirement) != 0 ||
        requirement.bytes > (uint64_t)SIZE_MAX)
        return NULL;
    if (posix_memalign(
            &pool, (size_t)requirement.alignment,
            (size_t)requirement.bytes) != 0 || !pool)
        return NULL;
    self = four_aec_nr_res_init(
        pool, (size_t)requirement.bytes, cfg);
    if (!self) {
        free(pool);
        return NULL;
    }
    self->owned_heap = pool;
    return self;
}

/* ============================================================================
 * Reset and teardown
 * ========================================================================== */

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
    if (!self || self->destroyed) return;

    delay_aec3_reset(&self->shared_delay);
    for (ch = 0; ch < self->initialized_lanes; ++ch) {
        aec_reset(self->lanes[ch]);
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
    void* owned_heap;
    if (!self || self->destroyed) return;

    if (self->nr) mmse_lsa_destroy(self->nr);
    if (self->fft) fft_destroy(self->fft);
    for (ch = 0; ch < self->initialized_lanes; ++ch) {
        aec_destroy(self->lanes[ch]);
    }
    owned_heap = self->owned_heap;
    self->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

/* ============================================================================
 * Per-hop processing: pre-beam AEC side
 * ========================================================================== */

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

    if (!self || self->destroyed ||
        !microphones_interleaved || !ref || !out)
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
            aec_reset(self->lanes[ch]);
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
            self->lanes[ch], self->mic_lane, self->aligned_ref,
            self->lane_out);
        aec_get_res_context(self->lanes[ch], &context);
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

/* ============================================================================
 * Per-hop processing: external-beam resume, one NR/RES and one iFFT/OLA
 * ========================================================================== */

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

    if (!self || self->destroyed || !token || !weights || !out)
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

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int four_aec_nr_res_hop_size(const FourAecNrRes* self) {
    return self && !self->destroyed ? self->hop_size : -1;
}

int four_aec_nr_res_fft_size(const FourAecNrRes* self) {
    return self && !self->destroyed ? self->fft_size : -1;
}

int four_aec_nr_res_n_freqs(const FourAecNrRes* self) {
    return self && !self->destroyed ? self->n_freqs : -1;
}

int four_aec_nr_res_sample_rate(const FourAecNrRes* self) {
    return self && !self->destroyed ? self->sample_rate : -1;
}

int four_aec_nr_res_matched_filter_count(const FourAecNrRes* self) {
    return self && !self->destroyed ? 1 : 0;
}

int four_aec_nr_res_linear_aec_count(const FourAecNrRes* self) {
    return self && !self->destroyed ? FOUR_AEC_NR_RES_CHANNELS : 0;
}

int four_aec_nr_res_nr_count(const FourAecNrRes* self) {
    return self && !self->destroyed ? 1 : 0;
}

int four_aec_nr_res_post_res_count(const FourAecNrRes* self) {
    return self && !self->destroyed ? 1 : 0;
}
