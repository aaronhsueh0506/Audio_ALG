/**
 * 4aec_nr_res.c — implementation of 4aec_nr_res.h.
 *
 * Four-channel counterpart of pipelines/audio_pipeline.c:
 *
 *   one shared DelayAec3 -> one aligned render
 *   -> four independent linear AEC filters
 *   -> externally supplied effective beamformer weights
 *   -> one coherent post-beam RES + one NR + one iFFT/OLA
 *
 * It follows the same pool-first layout and lifecycle as AudioPipeline:
 *
 *   four_aec_nr_res_get_mem_requirements() + four_aec_nr_res_init_ex()
 *       caller-owned pool; zero heap from init through destroy
 *
 *   four_aec_nr_res_create()
 *       heap convenience wrapper over that same pool-first implementation
 *
 * The only structural difference is the external beamformer seam: the mono
 * audio_pipeline_process() call is split into process_pre() and process_post().
 * Both construction paths use those same functions and neither allocates
 * while processing.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>

#include "4aec_nr_res.h"
#include "4aec_nr_res_internal.h"
#include "aec3_balanced_config.h"
#include "fft_wrapper.h"
#include "mem_align.h"
#include "suppression_gain.h"
#include "4aec_projection_kernels.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/* Same production NR/RES recipe names and values as audio_pipeline.c. */
#define PROD_NE_FLOOR             0.4f
#define PROD_NE_FLOOR_FAR_ACTIVE 0.2f
#define PROD_FAR_GATE_THRESH      1e-4f
#define PROD_NEAR_GATE_THRESH     1e-3f
#define PROD_NEAR_HANGOVER        8
#define PSD_SCALE                 (32768.0f * 32768.0f)
#define PIPELINE_RNG_SEED         0x9e3779b9u

/* Same compile-time backend identity used by audio_pipeline.c. */
#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

static uint32_t four_aec_nr_res_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "kiss") == 0)
        return FOUR_AEC_NR_RES_BACKEND_KISS;
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0)
        return FOUR_AEC_NR_RES_BACKEND_NE10;
    return 0u;
}

/* Process-wide monotonic construction counter -- see FourAecNrResFrameToken's
 * instance_epoch doc comment in the header for why this cannot be derived
 * from anything stored inside a (possibly reused, always-zeroed-at-init)
 * caller-owned pool. Not atomic: construction is assumed single-threaded,
 * consistent with the rest of this file (no locks/atomics anywhere else);
 * concurrent construction from multiple threads is out of scope here, same
 * as it already was before this counter existed. Starts at 1 so a
 * zero-initialized (never-constructed) token's instance_epoch of 0 can never
 * collide with a real instance's epoch. */
static uint64_t g_four_aec_nr_res_next_epoch = 1;

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
    uint64_t delay_calls;
    FourAecNrResDelayState last_delay;

    Complex* fused_error;
    Complex* fused_near;
    Complex* residual_work;
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
    int near_hangover_frames;  /* PROD_NEAR_HANGOVER retimed to this grid's hop */

    uint64_t next_frame;
    uint64_t generation;
    uint64_t construction_epoch;
    int pending;
    FourAecNrResFrameToken pending_token;

    void* owned_heap;
    size_t pool_size;
    int destroyed;
};

typedef struct PoolCursor {
    uint8_t* ptr;
    size_t remaining;
} PoolCursor;

/* ============================================================================
 * Config -> module configs + frame dimensions
 * ========================================================================== */

static int derive_dims_and_configs(
    const FourAecNrResConfig* cfg,
    AecConfig* aec_cfg,
    MmseLsaConfig* nr_cfg,
    int* fft_size,
    int* hop_size,
    int* n_freqs,
    int* post_ma_n,
    int* delay_ring_size) {
    const Aec3BalancedRateDims* rate_dims;
    int selected_fft;
    int max_delay_samples;
    if (!cfg || !aec_cfg || !nr_cfg || !fft_size || !hop_size ||
        !n_freqs || !post_ma_n || !delay_ring_size) return 0;
    if (cfg->sample_rate != 16000 && cfg->sample_rate != 48000) return 0;

#define FOUR_CK_BOOL(field) \
    do { if (cfg->field != 0 && cfg->field != 1) return 0; } while (0)

    switch (cfg->aec_preset) {
        case AEC_PRESET_MILD:
        case AEC_PRESET_BALANCED:
        case AEC_PRESET_AGGRESSIVE:
            break;
        default:
            return 0;
    }
    switch (cfg->nr_mode) {
        case MMSE_LSA_NR_MILD:
        case MMSE_LSA_NR_MODERATE:
        case MMSE_LSA_NR_BALANCED:
        case MMSE_LSA_NR_AGGRESSIVE:
            break;
        default:
            return 0;
    }
    FOUR_CK_BOOL(enable_cng);
    FOUR_CK_BOOL(legacy_amin);

#undef FOUR_CK_BOOL

    selected_fft = cfg->fft_size;
    if (selected_fft == 0)
        /* 16 kHz rate default is 256/128 (8ms hop) as of 2026-08-02/03,
         * matching both AEC's (python/modules/config.py, c_impl aec.c) and
         * NR's (core/signal_grid.py, mmse_lsa_types.h) own per-library
         * defaults -- this pipeline-level default was previously
         * independent of both and had drifted to the old 512/256 (16ms)
         * value. 512 remains a supported, explicit alternate (line below). */
        selected_fft = cfg->sample_rate == 16000 ? 256 : 1024;
    if (cfg->sample_rate == 16000) {
        if (selected_fft != 256 && selected_fft != 512) return 0;
    } else if (selected_fft != 1024) {
        return 0;
    }
    if (cfg->filter_length < 0 || cfg->filter_length > 4096) return 0;
    if (cfg->capture_proxy_channel < 0 ||
        cfg->capture_proxy_channel >= FOUR_AEC_NR_RES_CHANNELS) return 0;
    if (!isfinite(cfg->max_delay_ms) ||
        cfg->max_delay_ms < 0.0f || cfg->max_delay_ms > 4096.0f) return 0;
    *fft_size = selected_fft;

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
    /* 2026-08-03: was an implicit side effect of the C standalone default
     * (mmse_lsa_default_config_for_grid) also happening to be 0.8f -- that
     * default is now fixed to match Python's own config/v3_2_config.yaml
     * (1.0f, disabled), so this pipeline must set 0.8f explicitly to keep its
     * actual runtime behaviour unchanged. Mirrors audio_pipeline.c (mono) and
     * the deliberate overlay aec_nr_pipeline.py:_build_denoiser documents. */
    nr_cfg->broadband_threshold = 0.8f;
    /* 2026-08-03 A/B decision (824-case VCTK+DEMAND + 90-case AEC blind
     * manifest, see NR/CHANGELOG.md): take mmse_lsa_config_for_mode_grid()'s
     * canonical alpha_d/alpha_attack as-is instead of overriding them back
     * to the old L=150/alpha_d=0.95/alpha_attack=0.3-old-retime tuning --
     * that legacy tuning measured worse on the AEC-residual/double-talk
     * angle that matters for this pipeline. L and alpha_decay are untouched:
     * they already coincide with Python's canonical composition (see
     * audio_pipeline.c's mono twin for the full rationale). */
    nr_cfg->L = mmse_lsa_retime_frames(
        150, cfg->sample_rate, *hop_size);
    nr_cfg->alpha_decay = nr_cfg->alpha_g;
    return 1;
}

/* ============================================================================
 * Pool sizing and carving
 * ========================================================================== */

static void* pool_carve(PoolCursor* cursor, size_t count,
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

static int carve_lane_snapshot(FourAecLaneSnapshot* s,
                               PoolCursor* cursor,
                               int n_freqs) {
    s->error_spec =
        (Complex*)pool_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->echo_spec =
        (Complex*)pool_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->far_spec =
        (Complex*)pool_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->near_spec =
        (Complex*)pool_carve(cursor, (size_t)n_freqs, sizeof(Complex));
    s->r2 = (float*)pool_carve(
        cursor, (size_t)n_freqs, sizeof(float));
    s->comfort_noise = (float*)pool_carve(
        cursor, (size_t)n_freqs, sizeof(float));
    return s->error_spec && s->echo_spec && s->far_spec && s->near_spec &&
           s->r2 && s->comfort_noise;
}

static int carve_working_buffers(FourAecNrRes* p,
                                 PoolCursor* cursor) {
    int ch;
    int n = p->n_freqs;
    int hop = p->hop_size;
    int fft = p->fft_size;
    size_t linear_count = (size_t)hop * FOUR_AEC_NR_RES_CHANNELS;

    p->linear_interleaved =
        (float*)pool_carve(cursor, linear_count, sizeof(float));
    p->mic_lane =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    p->lane_out =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    p->aligned_ref =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    p->render_i16 =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));

    p->fused_error =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
    p->fused_near =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
    p->residual_work =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
    p->output_spec =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));

    p->fused_r2 =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));
    p->fused_comfort =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));
    p->post_near_power =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));
    p->nr_gain =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));
    p->total_gain =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));
    p->extra_noise =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));
    p->error_power =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));

    p->synth_window =
        (float*)pool_carve(cursor, (size_t)fft, sizeof(float));
    p->ifft_buffer =
        (float*)pool_carve(cursor, (size_t)fft, sizeof(float));
    p->ola =
        (float*)pool_carve(cursor, (size_t)fft, sizeof(float));

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        if (!carve_lane_snapshot(
                &p->snapshots[ch], cursor, n)) return 0;
    }

    return p->linear_interleaved && p->mic_lane && p->lane_out &&
           p->aligned_ref && p->render_i16 &&
           p->fused_error && p->fused_near &&
           p->residual_work && p->output_spec &&
           p->fused_r2 && p->fused_comfort && p->post_near_power &&
           p->nr_gain && p->total_gain && p->extra_noise &&
           p->error_power && p->synth_window && p->ifft_buffer &&
           p->ola;
}

static int init_post_sg(FourAecNrRes* p,
                             PoolCursor* pool_cursor) {
    int n = p->n_freqs;
    int ma_n;
    size_t float_count;
    float* cursor;

    p->post_sg_cfg = p->lanes[0]->a3_sg.cfg;
    p->post_sg_tun = p->lanes[0]->a3_sg.tun;
    ma_n = p->post_sg_cfg.nearend_smoother_n;
    if (ma_n < 1 || p->post_sg_cfg.n_bins != n ||
        p->post_sg_tun.table_len != n) return 0;

    float_count = (size_t)(10 + ma_n) * (size_t)n;
    p->post_sg_storage =
        (float*)pool_carve(pool_cursor, float_count, sizeof(float));
    if (!p->post_sg_storage) return 0;

    cursor = p->post_sg_storage;
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
            &p->post_sg, &p->post_sg_cfg, &p->post_sg_tun,
            last_gain, last_near, last_echo, ma_buf, near_s, weighted,
            min_gain, max_gain, raw_gain, gain, sum);
    }
    return 1;
}

static size_t pipeline_buffer_size(
    int hop, int fft, int n, int post_ma_n, int delay_ring_size) {
    size_t total = 0;
    int ch;
    int i;

    total = ck_field_size(
        total,
        ck_mul_size((size_t)hop, FOUR_AEC_NR_RES_CHANNELS),
        sizeof(float));                                      /* linear */
    /* mic_lane, lane_out, aligned_ref, render_i16 -- 4 hop-sized buffers.
     * (was 6: delay_capture/delay_render's naive external stride-pick
     * decimation was removed -- delay_aec3_init() now takes cfg_copy.
     * sample_rate directly and DelayAec3 anti-alias-filters + decimates
     * internally, the same shared sidechain the mono AEC repo uses at
     * 48kHz. Must stay in lockstep with carve_working_buffers() above.) */
    for (i = 0; i < 4; ++i)
        total = ck_field_size(total, (size_t)hop, sizeof(float));

    for (i = 0; i < 4; ++i)
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

static size_t pipeline_pool_size(
    const AecConfig* aec_cfg,
    const MmseLsaConfig* nr_cfg,
    int hop,
    int fft,
    int n,
    int post_ma_n,
    int delay_ring_size) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    size_t nr_bytes = mmse_lsa_get_mem_size(nr_cfg);
    size_t fft_bytes = fft_get_mem_size(fft);
    size_t buffer_bytes = pipeline_buffer_size(
        hop, fft, n, post_ma_n, delay_ring_size);
    size_t total = 0;
    int ch;

    if (aec_bytes == 0 || nr_bytes == 0 || fft_bytes == 0 ||
        buffer_bytes == 0) return 0;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        total = ck_add_size(total, ck_align16_size(aec_bytes));
    total = ck_add_size(total, ck_align16_size(nr_bytes));
    total = ck_add_size(total, ck_align16_size(fft_bytes));
    total = ck_add_size(total, buffer_bytes);
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/* ============================================================================
 * Carve/build — counterpart of audio_pipeline.c's pipeline_build()
 * ========================================================================== */

static int pipeline_build(
    FourAecNrRes* p,
    PoolCursor* cursor,
    const AecConfig* aec_cfg,
    const MmseLsaConfig* nr_cfg) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    size_t nr_bytes = mmse_lsa_get_mem_size(nr_cfg);
    size_t fft_bytes = fft_get_mem_size(p->fft_size);
    int ch;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        void* lane_pool = pool_carve(cursor, 1, aec_bytes);
        if (!lane_pool) return 0;
        p->lanes[ch] = aec_init(lane_pool, aec_bytes, aec_cfg);
        if (!p->lanes[ch]) return 0;
        p->initialized_lanes += 1;
    }

    {
        void* nr_pool = pool_carve(cursor, 1, nr_bytes);
        void* fft_pool = pool_carve(cursor, 1, fft_bytes);
        if (!nr_pool || !fft_pool) return 0;
        p->nr = mmse_lsa_init(nr_pool, nr_bytes, nr_cfg);
        p->fft = fft_init(fft_pool, fft_bytes, p->fft_size);
        if (!p->nr || !p->fft) return 0;
    }

    if (!carve_working_buffers(p, cursor) ||
        !init_post_sg(p, cursor)) return 0;
    p->delay_ring = (float*)pool_carve(
        cursor, (size_t)p->delay_ring_size, sizeof(float));
    if (!p->delay_ring) return 0;
    return 1;
}

/* ============================================================================
 * Build-flags hash
 * ========================================================================== */

static uint32_t fnv1a_str(const char* text, uint32_t hash) {
    while (*text) {
        hash ^= (uint32_t)(unsigned char)*text++;
        hash *= 16777619u;
    }
    return hash;
}

static uint32_t four_aec_nr_res_build_flags_hash(void) {
    uint32_t hash = 2166136261u;
    hash = fnv1a_str(AUDIO_PIPELINE_BACKEND_STR, hash);
    hash = fnv1a_str(
        "|carve:self,aec0,aec1,aec2,aec3,nr,fft,linear,hop4,"
        "complex4,float7,fftfloat3,snapshot4x6,postsg,delayring",
        hash);
    hash = fnv1a_str("|align16", hash);
    return hash;
}

/* ============================================================================
 * Public config and memory query
 * ========================================================================== */

FourAecNrResConfig four_aec_nr_res_default_config(int sample_rate) {
    FourAecNrResConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = 0;
    cfg.filter_length = 0;
    cfg.capture_proxy_channel = 0;
    cfg.max_delay_ms = 1024.0f;
    cfg.aec_preset = AEC_PRESET_BALANCED;
    cfg.nr_mode = MMSE_LSA_NR_BALANCED;
    cfg.enable_cng = 1;
    cfg.legacy_amin = 0;
    return cfg;
}

int four_aec_nr_res_get_mem_requirements(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemReq* out) {
    AecConfig aec_cfg;
    MmseLsaConfig nr_cfg;
    size_t pool_bytes;
    size_t total_bytes;
    uint32_t backend;
    int fft;
    int hop;
    int n;
    int post_ma_n;
    int delay_ring_size;

    if (!out || !derive_dims_and_configs(
            cfg, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size)) return -1;

    pool_bytes = pipeline_pool_size(
        &aec_cfg, &nr_cfg, hop, fft, n, post_ma_n, delay_ring_size);
    total_bytes = ck_add_size(
        ck_align16_size(sizeof(FourAecNrRes)), pool_bytes);
    backend = four_aec_nr_res_backend_id();
    if (pool_bytes == 0 || MEM_SIZE_INVALID(total_bytes) ||
        backend == 0u) return -1;

    memset(out, 0, sizeof(*out));
    out->descriptor_version = FOUR_AEC_NR_RES_DESCRIPTOR_VERSION;
    out->layout_version = FOUR_AEC_NR_RES_LAYOUT_VERSION;
    out->backend_id = backend;
    out->build_flags_hash = four_aec_nr_res_build_flags_hash();
    out->alignment = 16u;
    out->bytes = (uint64_t)total_bytes;
    return 0;
}

/* ============================================================================
 * Caller-pool init and heap convenience construction
 * ========================================================================== */

FourAecNrRes* four_aec_nr_res_init_ex(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg,
    const FourAecNrResMemReq* expected) {
    FourAecNrResConfig cfg_copy;
    FourAecNrResMemReq current;
    FourAecNrRes* p;
    PoolCursor cursor;
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
    if (!derive_dims_and_configs(
            &cfg_copy, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size)) return NULL;

    memset(mem, 0, (size_t)current.bytes);
    p = (FourAecNrRes*)mem;
    p->cfg = cfg_copy;
    p->sample_rate = cfg_copy.sample_rate;
    p->fft_size = fft;
    p->hop_size = hop;
    p->n_freqs = n;
    p->delay_ring_size = delay_ring_size;
    p->rng_state = PIPELINE_RNG_SEED;
    /* PROD_NEAR_HANGOVER (8) is a 10-ms-hop frame count (80 ms); was applied
     * as a raw literal regardless of grid (20-60% off at every one of this
     * pipeline's 3 real grids). Retimed the same way derive_dims_and_configs
     * already retimes the NR config's L/alpha_d/alpha_attack, and the same
     * way the mono pipeline's audio_pipeline.c now retimes this identical
     * constant. */
    p->near_hangover_frames = mmse_lsa_retime_frames(
        PROD_NEAR_HANGOVER, cfg_copy.sample_rate, hop);
    p->pool_size = (size_t)current.bytes;
    p->construction_epoch = g_four_aec_nr_res_next_epoch++;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(*p));
    cursor.remaining =
        (size_t)current.bytes - ALIGN16(sizeof(*p));
    if (!pipeline_build(
            p, &cursor, &aec_cfg, &nr_cfg) ||
        cursor.remaining != 0) return NULL;

    /* delay_aec3_init() takes the pipeline's REAL native sample_rate:
     * DelayAec3 now owns the 48kHz->16kHz anti-alias sidechain internally
     * (DaResample48 in delay_aec3.c/.h) -- the exact same shared
     * implementation the standalone mono AEC repo uses for its own 48kHz
     * DelayAec3 instance. update_shared_delay() below feeds it raw,
     * un-decimated capture/render hops directly; the estimator anti-alias
     * filters + decimates internally when constructed at 48000, and
     * delay_aec3_estimated_delay() returns the result already rescaled
     * back to this pipeline's native sample domain. This replaces the
     * naive (no anti-alias) external stride-pick decimation this file used
     * to do itself -- unifying mono and 4ch onto one resampler
     * implementation instead of two, one of which aliased. */
    delay_aec3_init(&p->shared_delay, cfg_copy.sample_rate);
    for (k = 0; k < fft; ++k) {
        p->synth_window[k] = sqrtf(
            0.5f * (1.0f - cosf(
                2.0f * M_PI_F * (float)k / (float)fft)));
    }

    for (k = 0; k < FOUR_AEC_NR_RES_CHANNELS; ++k) {
        AecResContext context;
        aec_get_res_context(p->lanes[k], &context);
        if (context.hop_size != hop || context.n_freqs != n)
            return NULL;
    }
    if (mmse_lsa_get_hop_size(p->nr) != hop ||
        mmse_lsa_get_n_freqs(p->nr) != n ||
        fft_get_n_freqs(p->fft) != n) return NULL;
    return p;
}

FourAecNrRes* four_aec_nr_res_init(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg) {
    return four_aec_nr_res_init_ex(mem, bytes, cfg, NULL);
}

FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg) {
    FourAecNrResMemReq requirement;
    FourAecNrRes* p;
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
    p = four_aec_nr_res_init(
        pool, (size_t)requirement.bytes, cfg);
    if (!p) {
        free(pool);
        return NULL;
    }
    p->owned_heap = pool;
    return p;
}

static int inputs_finite(const float* data, size_t count) {
    size_t i;
    if (!data) return 0;
    for (i = 0; i < count; ++i) {
        if (!isfinite(data[i])) return 0;
    }
    return 1;
}

static int complex_close(Complex a, Complex b) {
    float dr = fabsf(a.r - b.r);
    float di = fabsf(a.i - b.i);
    float scale = 1.0f + fabsf(a.r) + fabsf(a.i);
    return dr <= 1e-5f * scale && di <= 1e-5f * scale;
}

static float rng_uniform(FourAecNrRes* p) {
    uint32_t x = p->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    p->rng_state = x;
    return ((x >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

static float rng_gauss(FourAecNrRes* p) {
    float u1 = rng_uniform(p);
    float u2 = rng_uniform(p);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI_F * u2);
}

/* ============================================================================
 * Per-hop processing — Stage 1: shared delay + four linear AEC lanes
 * ========================================================================== */

static FourAecNrResDelayState update_shared_delay(
    FourAecNrRes* p,
    const float* capture,
    const float* render) {
    FourAecNrResDelayState state;
    int emitted;
    int estimated;
    int eligible;

    /* Raw, un-decimated hops: delay_aec3_init() constructed p->shared_delay
     * at this pipeline's real native sample_rate, so DelayAec3 itself now
     * anti-alias-filters + decimates internally at 48kHz (see delay_aec3.c's
     * DaResample48) and delay_aec3_estimated_delay() below returns the
     * result already rescaled back to this pipeline's native domain. */
    emitted = delay_aec3_accumulate(&p->shared_delay, capture, render, p->hop_size);
    (void)emitted;
    p->delay_calls += 1;

    estimated = delay_aec3_estimated_delay(&p->shared_delay);
    eligible = estimated >= 0 &&
               delay_aec3_is_solid(&p->shared_delay) &&
               delay_aec3_n_updates(&p->shared_delay) >= 3;

    memset(&state, 0, sizeof(state));
    state.changed = eligible && estimated != p->accepted_delay;
    if (eligible) p->accepted_delay = estimated;
    state.delay_samples = p->accepted_delay;
    state.confidence = delay_aec3_confidence(&p->shared_delay);
    state.solid = delay_aec3_is_solid(&p->shared_delay);
    state.estimator_calls = p->delay_calls;
    state.estimator_updates = delay_aec3_n_updates(&p->shared_delay);
    p->last_delay = state;
    return state;
}

static int align_render(FourAecNrRes* p, const float* render,
                             int delay_samples) {
    int i;
    uint64_t start;
    if (delay_samples < 0 ||
        delay_samples >= p->delay_ring_size - p->hop_size) return 0;

    start = p->delay_samples_seen;
    for (i = 0; i < p->hop_size; ++i) {
        uint64_t absolute = start + (uint64_t)i;
        p->delay_ring[absolute % (uint64_t)p->delay_ring_size] =
            render[i];
    }
    for (i = 0; i < p->hop_size; ++i) {
        uint64_t absolute = start + (uint64_t)i;
        if (absolute >= (uint64_t)delay_samples) {
            uint64_t source = absolute - (uint64_t)delay_samples;
            p->aligned_ref[i] =
                p->delay_ring[source %
                                 (uint64_t)p->delay_ring_size];
        } else {
            p->aligned_ref[i] = 0.0f;
        }
    }
    p->delay_samples_seen += (uint64_t)p->hop_size;
    return 1;
}

static int snapshot_context(FourAecLaneSnapshot* dst,
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
    FourAecNrRes* p,
    const float* microphones_interleaved,
    const float* ref,
    FourAecNrResPreFrame* out) {
    FourAecNrResDelayState delay;
    int hop;
    int ch;
    int i;

    if (!p || p->destroyed ||
        !microphones_interleaved || !ref || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->pending) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;

    hop = p->hop_size;
    if (!inputs_finite(
            microphones_interleaved,
            (size_t)hop * FOUR_AEC_NR_RES_CHANNELS) ||
        !inputs_finite(ref, (size_t)hop))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    for (i = 0; i < hop; ++i) {
        p->mic_lane[i] =
            microphones_interleaved[
                i * FOUR_AEC_NR_RES_CHANNELS +
                p->cfg.capture_proxy_channel];
    }
    delay = update_shared_delay(p, p->mic_lane, ref);
    if (!align_render(p, ref, delay.delay_samples)) {
        four_aec_nr_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (delay.changed) {
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            aec_reset(p->lanes[ch]);
        }
        /* The AEC reset starts a new WOLA sequence whose first analysis
         * frame has a zero previous half. Discard the old mono synthesis
         * tail as well; mixing it into the new sequence would join spectra
         * from opposite sides of the delay realignment. */
        memset(p->ola, 0, (size_t)p->fft_size * sizeof(float));
    }

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        AecResContext context;
        for (i = 0; i < hop; ++i) {
            p->mic_lane[i] =
                microphones_interleaved[
                    i * FOUR_AEC_NR_RES_CHANNELS + ch];
        }
        aec_process(
            p->lanes[ch], p->mic_lane, p->aligned_ref,
            p->lane_out);
        aec_get_res_context(p->lanes[ch], &context);
        if (!context.formed_hop || !snapshot_context(
                &p->snapshots[ch], &context, p->n_freqs)) {
            four_aec_nr_res_reset(p);
            return FOUR_AEC_NR_RES_DSP_ERROR;
        }
        for (i = 0; i < hop; ++i) {
            p->linear_interleaved[
                i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                context.formed_hop[i];
        }
    }

    p->pending_token.frame_index = p->next_frame;
    p->pending_token.generation = p->generation;
    p->pending_token.owner_cookie = (uintptr_t)p;
    p->pending_token.instance_epoch = p->construction_epoch;
    p->next_frame += 1;
    p->pending = 1;

    out->token = p->pending_token;
    out->delay = delay;
    out->hop_size = p->hop_size;
    out->n_channels = FOUR_AEC_NR_RES_CHANNELS;
    out->n_freqs = p->n_freqs;
    out->linear_interleaved = p->linear_interleaved;
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        out->linear_spectra[ch] = p->snapshots[ch].error_spec;
    }
    return FOUR_AEC_NR_RES_OK;
}

/* ============================================================================
 * Per-hop processing — Stages 2/3: external beamformer -> NR/RES -> OLA
 * ========================================================================== */

static int token_matches(const FourAecNrRes* p,
                              const FourAecNrResFrameToken* token) {
    return p && token && p->pending &&
           token->frame_index == p->pending_token.frame_index &&
           token->generation == p->pending_token.generation &&
           token->owner_cookie == p->pending_token.owner_cookie &&
           token->owner_cookie == (uintptr_t)p &&
           token->instance_epoch == p->construction_epoch;
}

static int validate_weights(const FourAecNrRes* p,
                                 const Complex* weights) {
    size_t count;
    size_t i;
    float sum = 0.0f;
    if (!p || !weights) return 0;
    count = (size_t)FOUR_AEC_NR_RES_CHANNELS *
            (size_t)p->n_freqs;
    for (i = 0; i < count; ++i) {
        if (!isfinite(weights[i].r) || !isfinite(weights[i].i))
            return 0;
        sum += fabsf(weights[i].r) + fabsf(weights[i].i);
    }
    return isfinite(sum) && sum > 1e-12f;
}

static int complex_vector_finite(const Complex* values, int count) {
    int i;
    if (!values || count < 0) return 0;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i].r) || !isfinite(values[i].i)) return 0;
    }
    return 1;
}

static int fuse_contexts(FourAecNrRes* p,
                         const Complex* weights,
                         const Complex* trusted_beamformed_error,
                         int* all_converged,
                         float* max_dt,
                         float* max_saturation,
                         float* far_power) {
    int n = p->n_freqs;
    int k;
    int ch;
    float base_far_power = p->snapshots[0].far_power;

    *all_converged = 1;
    *max_dt = p->snapshots[0].dt_indicator;
    *max_saturation = p->snapshots[0].saturation_level;
    *far_power = base_far_power;

    if (trusted_beamformed_error) {
        if (!complex_vector_finite(trusted_beamformed_error, n)) return 0;
        memcpy(
            p->fused_error, trusted_beamformed_error,
            (size_t)n * sizeof(Complex));
    } else {
        memset(p->fused_error, 0, (size_t)n * sizeof(Complex));
    }
    memset(p->fused_near, 0, (size_t)n * sizeof(Complex));
    /* output_spec is scratch for the coherent residual vector until
     * run_post_res_and_nr() overwrites every bin with the final spectrum. */
    memset(p->output_spec, 0, (size_t)n * sizeof(Complex));
    memset(p->fused_comfort, 0, (size_t)n * sizeof(float));

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        const Complex* lane_weights =
            weights + (size_t)ch * (size_t)n;
        float delta = fabsf(
            p->snapshots[ch].far_power - base_far_power);
        float scale = 1.0f + fabsf(base_far_power);
        if (delta > 1e-5f * scale) return 0;
        if (!p->snapshots[ch].filter_converged)
            *all_converged = 0;
        if (p->snapshots[ch].dt_indicator > *max_dt)
            *max_dt = p->snapshots[ch].dt_indicator;
        if (p->snapshots[ch].saturation_level > *max_saturation)
            *max_saturation = p->snapshots[ch].saturation_level;

        for (k = 0; k < n; ++k) {
            if (!complex_close(
                    p->snapshots[ch].far_spec[k],
                    p->snapshots[0].far_spec[k])) return 0;
        }

        if (!trusted_beamformed_error) {
            four_aec_projection_cmac(
                p->fused_error, lane_weights,
                p->snapshots[ch].error_spec, n);
        }
        four_aec_projection_cmac(
            p->fused_near, lane_weights,
            p->snapshots[ch].near_spec, n);
        four_aec_residual_vector(
            p->residual_work, p->snapshots[ch].echo_spec,
            p->snapshots[ch].r2, n);
        four_aec_projection_cmac(
            p->output_spec, lane_weights, p->residual_work, n);
        four_aec_comfort_accumulate(
            p->fused_comfort, lane_weights,
            p->snapshots[ch].comfort_noise, n);
    }

    four_aec_complex_mag2(p->fused_r2, p->output_spec, n);
    return 1;
}

static int run_post_res_and_nr(
    FourAecNrRes* p,
    int all_converged,
    float max_dt,
    float max_saturation,
    float far_power,
    float* out) {
    const float* res_gain;
    const float* nr_extra;
    int n = p->n_freqs;
    int hop = p->hop_size;
    int fft = p->fft_size;
    int k;
    float nf_eff;

    four_aec_complex_mag2(p->error_power, p->fused_error, n);
    four_aec_complex_mag2(p->post_near_power, p->fused_near, n);
    for (k = 0; k < n; ++k) {
        float e2 = p->error_power[k];
        float n2 = p->post_near_power[k];
        p->post_near_power[k] =
            (all_converged ? fminf(e2, n2) : n2) *
            PSD_SCALE;
        p->extra_noise[k] =
            p->fused_r2[k] / PSD_SCALE;
    }
    for (k = 0; k < hop; ++k) {
        p->render_i16[k] = p->aligned_ref[k] * 32768.0f;
    }

    if (p->post_sg.initial_state && all_converged)
        suppression_gain_set_initial_state(&p->post_sg, 0);
    p->post_sg.dt_protect_active = max_dt > 0.2f;
    res_gain = suppression_gain_get_gain(
        &p->post_sg,
        p->post_near_power,
        p->fused_r2,
        p->fused_r2,
        p->fused_comfort,
        p->render_i16,
        delay_aec3_has_clockdrift(&p->shared_delay),
        max_saturation > 0.5f);
    if (!res_gain) return 0;

    nr_extra = p->cfg.legacy_amin ? NULL : p->extra_noise;
    if (mmse_lsa_process_gain(
            p->nr, p->fused_error, nr_extra, p->nr_gain) != 0)
        return 0;

    for (k = 0; k < n; ++k) {
        p->total_gain[k] =
            fminf(p->nr_gain[k], res_gain[k]);
    }

    nf_eff = PROD_NE_FLOOR;
    if (!p->cfg.legacy_amin) {
        int far_active =
            far_power > PROD_FAR_GATE_THRESH;
        float near_mean = 0.0f;
        int near_active;
        for (k = 0; k < n; ++k) near_mean += p->error_power[k];
        near_mean /= (float)n;
        if (near_mean > PROD_NEAR_GATE_THRESH)
            p->near_hang = p->near_hangover_frames;
        near_active = p->near_hang > 0;
        if (p->near_hang > 0) p->near_hang -= 1;
        nf_eff = (!far_active && near_active)
            ? PROD_NE_FLOOR
            : PROD_NE_FLOOR_FAR_ACTIVE;
    }

    for (k = 0; k < n; ++k) {
        float echo_fraction =
            p->extra_noise[k] / (p->error_power[k] + 1e-12f);
        float no_echo;
        float lift;
        if (echo_fraction < 0.0f) echo_fraction = 0.0f;
        if (echo_fraction > 1.0f) echo_fraction = 1.0f;
        no_echo = res_gain[k] * (1.0f - echo_fraction);
        lift = nf_eff * no_echo;
        p->total_gain[k] =
            (1.0f - lift) * p->total_gain[k] + lift;
        p->output_spec[k].r =
            p->fused_error[k].r * p->total_gain[k];
        p->output_spec[k].i =
            p->fused_error[k].i * p->total_gain[k];
    }

    if (p->cfg.enable_cng) {
        for (k = 1; k < n - 1; ++k) {
            float n_amp =
                p->fused_comfort[k] / PSD_SCALE;
            float gain2 = 1.0f - res_gain[k] * res_gain[k];
            float amplitude;
            n_amp = n_amp > 0.0f ? sqrtf(n_amp) : 0.0f;
            gain2 = gain2 > 0.0f ? sqrtf(gain2) : 0.0f;
            amplitude = n_amp * gain2;
            p->output_spec[k].r +=
                amplitude * rng_gauss(p);
            p->output_spec[k].i +=
                amplitude * rng_gauss(p);
        }
    }

    fft_inverse(p->fft, p->output_spec, p->ifft_buffer);
    for (k = 0; k < fft; ++k) {
        p->ola[k] +=
            p->ifft_buffer[k] * p->synth_window[k];
    }
    memcpy(out, p->ola, (size_t)hop * sizeof(float));
    memmove(
        p->ola, p->ola + hop,
        (size_t)(fft - hop) * sizeof(float));
    memset(
        p->ola + (fft - hop), 0,
        (size_t)hop * sizeof(float));
    return inputs_finite(out, (size_t)hop);
}

static int process_post_impl(
    FourAecNrRes* p,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    const Complex* trusted_beamformed_error,
    float* out) {
    int all_converged;
    float max_dt;
    float max_saturation;
    float far_power;

    if (!p || p->destroyed || !token || !weights || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (!token_matches(p, token))
        return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    if (!validate_weights(p, weights))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    if (!fuse_contexts(
            p, weights, trusted_beamformed_error,
            &all_converged, &max_dt,
            &max_saturation, &far_power)) {
        four_aec_nr_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (!run_post_res_and_nr(
            p, all_converged, max_dt, max_saturation,
            far_power, out)) {
        four_aec_nr_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    p->pending = 0;
    memset(&p->pending_token, 0, sizeof(p->pending_token));
    return FOUR_AEC_NR_RES_OK;
}

int four_aec_nr_res_process_post(
    FourAecNrRes* p,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    float* out) {
    return process_post_impl(p, token, weights, NULL, out);
}

int four_aec_nr_res_process_post_trusted_spectrum(
    FourAecNrRes* p,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    const Complex* beamformed_error,
    float* out) {
    if (!beamformed_error) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    return process_post_impl(
        p, token, weights, beamformed_error, out);
}

/* ============================================================================
 * Reset and teardown
 * ========================================================================== */

static void reset_post_sg(FourAecNrRes* p) {
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

    if (!p || !p->post_sg_storage) return;
    n = p->n_freqs;
    ma_n = p->post_sg_cfg.nearend_smoother_n;
    cursor = p->post_sg_storage;
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

    memset(p->post_sg_storage, 0,
           (size_t)(10 + ma_n) * (size_t)n * sizeof(float));
    suppression_gain_init(
        &p->post_sg, &p->post_sg_cfg, &p->post_sg_tun,
        last_gain, last_near, last_echo, ma_buf, near_s, weighted,
        min_gain, max_gain, raw_gain, gain, sum);
}

void four_aec_nr_res_reset(FourAecNrRes* p) {
    int ch;
    if (!p || p->destroyed) return;

    delay_aec3_reset(&p->shared_delay);
    for (ch = 0; ch < p->initialized_lanes; ++ch) {
        aec_reset(p->lanes[ch]);
    }
    if (p->nr) mmse_lsa_reset(p->nr);
    reset_post_sg(p);

    if (p->delay_ring) {
        memset(p->delay_ring, 0,
               (size_t)p->delay_ring_size * sizeof(float));
    }
    if (p->linear_interleaved) {
        memset(p->linear_interleaved, 0,
               (size_t)p->hop_size * FOUR_AEC_NR_RES_CHANNELS *
               sizeof(float));
    }
    if (p->aligned_ref) {
        memset(p->aligned_ref, 0,
               (size_t)p->hop_size * sizeof(float));
    }
    if (p->ola) {
        memset(p->ola, 0, (size_t)p->fft_size * sizeof(float));
    }
    if (p->ifft_buffer) {
        memset(p->ifft_buffer, 0,
               (size_t)p->fft_size * sizeof(float));
    }

    p->delay_samples_seen = 0;
    p->accepted_delay = 0;
    p->delay_calls = 0;
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->rng_state = PIPELINE_RNG_SEED;
    p->near_hang = 0;
    p->next_frame = 0;
    p->generation += 1;
    p->pending = 0;
    memset(&p->pending_token, 0, sizeof(p->pending_token));
}

void four_aec_nr_res_destroy(FourAecNrRes* p) {
    int ch;
    void* owned_heap;
    if (!p || p->destroyed) return;

    if (p->nr) mmse_lsa_destroy(p->nr);
    if (p->fft) fft_destroy(p->fft);
    for (ch = 0; ch < p->initialized_lanes; ++ch) {
        aec_destroy(p->lanes[ch]);
    }
    owned_heap = p->owned_heap;
    p->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

int four_aec_nr_res_hop_size(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->hop_size : -1;
}

int four_aec_nr_res_fft_size(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->fft_size : -1;
}

int four_aec_nr_res_n_freqs(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->n_freqs : -1;
}

int four_aec_nr_res_sample_rate(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->sample_rate : -1;
}

int four_aec_nr_res_matched_filter_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? 1 : 0;
}

int four_aec_nr_res_linear_aec_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? FOUR_AEC_NR_RES_CHANNELS : 0;
}

int four_aec_nr_res_nr_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? 1 : 0;
}

int four_aec_nr_res_post_res_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? 1 : 0;
}

/* ============================================================================
 * Diagnostic memory breakdown
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

    if (!out || !derive_dims_and_configs(
            cfg, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size)) return -1;

    aec_one = aec_get_mem_size(&aec_cfg);
    nr_bytes = mmse_lsa_get_mem_size(&nr_cfg);
    fft_bytes = fft_get_mem_size(fft);
    wrapper_storage = pipeline_buffer_size(
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
