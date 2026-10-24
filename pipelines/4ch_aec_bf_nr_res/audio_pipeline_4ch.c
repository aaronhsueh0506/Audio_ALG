/** Four-channel linear AEC plus beamformer unit: four_aec_front_process -> SRP-PHAT DOA -> BF
 * effective weights, and four_aec_fuse as a separate step. Same pool-first layout and
 * lifecycle as 4aec_nr_res.c (get_mem_requirements()/init_ex() compose the core's
 * descriptor-tier API with SRP/BF's size_t tier and this unit's scratch); create() is the heap
 * convenience over the same implementation (see audio_pipeline_4ch.h). */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_pipeline_4ch.h"
#include "anr_types.h"   /* anr_retime_alpha, anr_retime_frames */
#include "bf.h"
#include "srp.h"
#include "spatial_simd.h"
#include "utility/mem_align.h"
#include "common/se_config.h"   /* se_flag_valid */
#include "common/se_stage.h"    /* SeSpecFrame */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PI_F ((float)M_PI)

#define DEFAULT_RADIUS_M  0.035f
#define DEFAULT_SPACING_M 0.035f
#define DEFAULT_ANGLES    72

struct AudioPipeline4Ch {
    AudioPipeline4ChConfig cfg;
    FourAecNrRes* core;
    SRP* srp;
    Complex*** bf_steering;
    BF* bf;

    int hop_size;
    int fft_size;
    int n_freqs;
    Complex* bf_spectrum;
    Complex* bf_weights;

    int phase;                  /* 0 idle, 1 front done, 2 beam done */

    float noise_power;
    float vad_power_threshold;    /* derived from cfg.auto_vad_threshold_dbfs and cached for auto_vad() */
    int vad_hangover;
    int vad_hangover_frames;      /* derived from cfg.auto_vad_hangover_frames */
    float vad_speech_noise_keep;  /* derived (speech noise-EMA retention) */
    float vad_speech_new_weight;  /* derived (speech noise-EMA new weight) */
    float vad_silence_noise_keep; /* derived (silence noise-EMA retention) */
    float vad_silence_new_weight; /* derived (silence noise-EMA new weight) */
    uint64_t frame_index;

    /* Non-NULL only on the audio_pipeline_4ch_create() heap path: the single block backing
     * this struct and every carved sub-region, freed by destroy(). NULL on the caller-pool
     * path, where destroy() must not free. */
    void* owned_heap;
    int destroyed;
};

/* Config validation */

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
    if (!se_flag_valid(cfg->doa_enable_smoothing) ||
        cfg->doa_switch_consecutive <= 0 ||
        !isfinite(cfg->doa_angle_tolerance_rad) ||
        cfg->doa_angle_tolerance_rad < 0.0f ||
        cfg->doa_update_interval <= 0) return 0;
    if (!se_flag_valid(cfg->bf_enable) ||
        !isfinite(cfg->bf_lambda) || cfg->bf_lambda <= 0.0f ||
        cfg->bf_lambda > 1.0f ||
        !isfinite(cfg->bf_mu) || cfg->bf_mu < 0.0f ||
        !se_flag_valid(cfg->bf_fixed_mode) ||
        !isfinite(cfg->bf_fixed_doa_rad) ||
        !se_flag_valid(cfg->bf_fixed_align_notebook) ||
        cfg->bf_adapt_interval <= 0) return 0;
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
    cfg.bf_enable = 1;
    cfg.bf_lambda = 0.995f;
    cfg.bf_mu = 0.1f;
    cfg.bf_fixed_mode = 0;
    cfg.bf_fixed_doa_rad = 0.0f;
    cfg.bf_fixed_align_notebook = 0;
    cfg.bf_adapt_interval = 1;
    cfg.auto_vad_threshold_dbfs = -55.0f;
    cfg.auto_vad_snr_ratio = 3.0f;
    cfg.auto_vad_hangover_frames = 8;
    return cfg;
}

/* Pool-first construction: file-private bump allocator. */

typedef struct PoolCursor {
    uint8_t* ptr;
    size_t remaining;
} PoolCursor;

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

static uint32_t fnv1a_str(const char* text, uint32_t hash) {
    while (*text) {
        hash ^= (uint32_t)(unsigned char)*text++;
        hash *= 16777619u;
    }
    return hash;
}

/* Plain integer mix, no snprintf/stdio (must stay NO_STDIO-safe; see Makefile
 * audit-no-stdio). Folds in the core's build_flags_hash, which sees its carve token but
 * not a control-block-only growth; bump AUDIO_PIPELINE_4CH_LAYOUT_VERSION and the token by
 * hand to invalidate a persisted composite descriptor. */
static uint32_t audio_pipeline_4ch_build_flags_hash(
    uint32_t core_build_flags_hash) {
    uint32_t hash = 2166136261u;
    hash = fnv1a_str(
        "|carve:self(corecfg-delay-v3),core,srp,bf,bf_spectrum,bf_weights",
        hash);
    hash = fnv1a_str("|align16", hash);
    hash ^= core_build_flags_hash;
    hash *= 16777619u;
    return ac_memreq_abi_hash(hash);
}

/* Stack-only array geometry: x/y are filled in caller storage because the zero-heap
 * init_ex() path cannot allocate. */

static void fill_uca_geometry(float radius, float* x, float* y) {
    int M = FOUR_AEC_NR_RES_CHANNELS;
    int m;
    for (m = 0; m < M; m++) {
        float phi = 2.0f * M_PI * m / M;
        x[m] = radius * cosf(phi);
        y[m] = radius * sinf(phi);
    }
}

static void fill_ula_geometry(float spacing, float* x, float* y) {
    int M = FOUR_AEC_NR_RES_CHANNELS;
    float center = 0.5f * (M - 1);
    int m;
    for (m = 0; m < M; m++) {
        x[m] = (m - center) * spacing;
        y[m] = 0.0f;
    }
}

static void fill_custom_geometry(
    const AudioPipeline4ChConfig* cfg, float* x, float* y) {
    int m;
    for (m = 0; m < FOUR_AEC_NR_RES_CHANNELS; m++) {
        x[m] = cfg->microphone_x_m[m];
        y[m] = cfg->microphone_y_m[m];
    }
}

/* Builds geom in place from caller-supplied x/y storage (FOUR_AEC_NR_RES_CHANNELS
 * entries). ArrayGeometry is only read transiently inside srp_init(), so a stack
 * temporary is safe. */
static int fill_stack_geometry(
    const AudioPipeline4ChConfig* cfg,
    float* x, float* y, ArrayGeometry* geom) {
    switch (cfg->geometry) {
        case AUDIO_PIPELINE_4CH_GEOMETRY_UCA:
            fill_uca_geometry(cfg->uca_radius_m, x, y);
            break;
        case AUDIO_PIPELINE_4CH_GEOMETRY_ULA:
            fill_ula_geometry(cfg->ula_spacing_m, x, y);
            break;
        case AUDIO_PIPELINE_4CH_GEOMETRY_CUSTOM:
            fill_custom_geometry(cfg, x, y);
            break;
        default:
            return 0;
    }
    geom->M = FOUR_AEC_NR_RES_CHANNELS;
    geom->x = x;
    geom->y = y;
    return 1;
}

/* Builds srp_cfg/bf_cfg from cfg plus the core's derived hop/fft/n_freqs, computed once
 * for both sizing and init. */
static int derive_spatial_configs(
    const AudioPipeline4ChConfig* cfg,
    int hop, int fft, int n_freqs,
    SRP_Config* srp_cfg, BF_Config* bf_cfg) {
    int bf_effective_interval;
    if (!cfg || !srp_cfg || !bf_cfg) return 0;

    memset(srp_cfg, 0, sizeof(*srp_cfg));
    srp_cfg->M = FOUR_AEC_NR_RES_CHANNELS;
    srp_cfg->F = n_freqs;
    srp_cfg->num_angles = cfg->num_angles;
    srp_cfg->sr = (float)cfg->core.sample_rate;
    srp_cfg->NFFT = (float)fft;
    srp_cfg->c = cfg->speed_of_sound_m_s;
    srp_cfg->low_freq = cfg->doa_low_freq_hz;
    srp_cfg->high_freq = cfg->doa_high_freq_hz > 0.0f
        ? fminf(cfg->doa_high_freq_hz,
                0.5f * (float)cfg->core.sample_rate)
        : fminf(7000.0f, 0.5f * (float)cfg->core.sample_rate);
    srp_cfg->enable_smoothing = cfg->doa_enable_smoothing;
    srp_cfg->switch_consec = cfg->doa_switch_consecutive;
    srp_cfg->angle_tol = cfg->doa_angle_tolerance_rad;
    srp_cfg->update_interval = cfg->doa_update_interval;

    /* bf_create()/bf_init() force the RLS cadence to 1 in fixed-notebook mode
     * (bf_effective_adapt_interval()); derive that effective value once and use it for both
     * the lambda retime and bf_cfg->adapt_interval. */
    bf_effective_interval = bf_effective_adapt_interval(
        cfg->bf_fixed_mode, cfg->bf_fixed_align_notebook,
        cfg->bf_adapt_interval);

    memset(bf_cfg, 0, sizeof(*bf_cfg));
    bf_cfg->enable = cfg->bf_enable;
    /* bf_lambda is an RLS retention factor tuned at a 10 ms update, so it is retimed with
     * anr_retime_alpha() like NR's constants; mu is a step size and is left as is. The RLS
     * update runs once per bf_effective_interval hops, so the retime scales hop by it (no-op
     * at the default 1). */
    bf_cfg->lambda = anr_retime_alpha(
        cfg->bf_lambda, cfg->core.sample_rate,
        hop * bf_effective_interval);
    bf_cfg->mu = cfg->bf_mu;
    bf_cfg->enable_fix_mode = cfg->bf_fixed_mode;
    bf_cfg->fixed_doa_rad = cfg->bf_fixed_doa_rad;
    bf_cfg->fixed_align_notebook = cfg->bf_fixed_align_notebook;
    bf_cfg->adapt_interval = bf_effective_interval;
    return 1;
}

static int derive_pipeline_layout(
    const AudioPipeline4ChConfig* cfg,
    FourAecNrResMemReq* core_req,
    SRP_Config* srp_cfg, BF_Config* bf_cfg,
    int* hop, int* fft, int* n_freqs,
    size_t* srp_bytes, size_t* bf_bytes) {
    FourAecNrResMemBreakdown breakdown;
    if (!cfg || !core_req || !srp_cfg || !bf_cfg || !hop || !fft ||
        !n_freqs || !srp_bytes || !bf_bytes) return 0;
    if (!validate_config(cfg)) return 0;
    if (four_aec_nr_res_get_mem_requirements(&cfg->core, core_req) != 0)
        return 0;
    /* four_aec_nr_res_get_mem_breakdown() derives hop/fft/n_freqs from cfg alone. */
    if (four_aec_nr_res_get_mem_breakdown(&cfg->core, &breakdown) != 0)
        return 0;
    *hop = breakdown.hop_size;
    *fft = breakdown.fft_size;
    *n_freqs = breakdown.n_freqs;
    if (!derive_spatial_configs(cfg, *hop, *fft, *n_freqs, srp_cfg, bf_cfg))
        return 0;
    *srp_bytes = srp_get_mem_size(srp_cfg);
    *bf_bytes = bf_get_mem_size(FOUR_AEC_NR_RES_CHANNELS, *n_freqs);
    if (*srp_bytes == 0 || *bf_bytes == 0) return 0;
    return 1;
}

int audio_pipeline_4ch_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChMemReq* out) {
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    BF_Config bf_cfg;
    int hop, fft, n_freqs;
    size_t srp_bytes, bf_bytes;
    size_t spectral_count;
    size_t total;

    if (!out || !derive_pipeline_layout(
            cfg, &core_req, &srp_cfg, &bf_cfg,
            &hop, &fft, &n_freqs, &srp_bytes, &bf_bytes)) return -1;
    if (core_req.bytes > (uint64_t)SIZE_MAX) return -1;

    total = ck_align16_size(sizeof(AudioPipeline4Ch));
    total = ck_add_size(total, ck_align16_size((size_t)core_req.bytes));
    total = ck_add_size(total, ck_align16_size(srp_bytes));
    total = ck_add_size(total, ck_align16_size(bf_bytes));

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    total = ck_field_size(
        total, (size_t)n_freqs, sizeof(Complex));       /* bf_spectrum */
    total = ck_field_size(
        total, spectral_count, sizeof(Complex));        /* bf_weights */

    if (MEM_SIZE_INVALID(total)) return -1;

    memset(out, 0, sizeof(*out));
    out->descriptor_version = AUDIO_PIPELINE_4CH_DESCRIPTOR_VERSION;
    out->layout_version = AUDIO_PIPELINE_4CH_LAYOUT_VERSION;
    out->backend_id = core_req.backend_id;
    out->build_flags_hash =
        audio_pipeline_4ch_build_flags_hash(core_req.build_flags_hash);
    out->alignment = 16u;
    out->bytes = (uint64_t)total;
    return 0;
}

AudioPipeline4Ch* audio_pipeline_4ch_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChMemReq* expected) {
    AudioPipeline4ChConfig cfg_copy;
    AudioPipeline4ChMemReq current;
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    BF_Config bf_cfg;
    int hop, fft, n_freqs;
    size_t srp_bytes, bf_bytes;
    float geom_x[FOUR_AEC_NR_RES_CHANNELS];
    float geom_y[FOUR_AEC_NR_RES_CHANNELS];
    ArrayGeometry geom;
    AudioPipeline4Ch* p;
    PoolCursor cursor;
    void* core_region;
    void* srp_region;
    void* bf_region;
    size_t spectral_count;

    if (!mem || !cfg) return NULL;
    cfg_copy = *cfg;
    if (audio_pipeline_4ch_get_mem_requirements(&cfg_copy, &current) != 0)
        return NULL;

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

    if (!derive_pipeline_layout(
            &cfg_copy, &core_req, &srp_cfg, &bf_cfg,
            &hop, &fft, &n_freqs, &srp_bytes, &bf_bytes)) return NULL;
    if (core_req.bytes > (uint64_t)SIZE_MAX) return NULL;
    if (!fill_stack_geometry(&cfg_copy, geom_x, geom_y, &geom)) return NULL;

    memset(mem, 0, (size_t)current.bytes);
    p = (AudioPipeline4Ch*)mem;
    p->cfg = cfg_copy;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(*p));
    cursor.remaining = (size_t)current.bytes - ALIGN16(sizeof(*p));

    /* Composition: query each sub-module's size, carve an equal-sized sub-region, call its
     * init/init_ex. Carving the core first selects four_aec_nr_res_init_ex() (pool) over
     * create() (heap). */
    core_region = pool_carve(&cursor, 1, (size_t)core_req.bytes);
    if (!core_region) return NULL;
    p->core = four_aec_nr_res_init_ex(
        core_region, (size_t)core_req.bytes, &cfg_copy.core, &core_req);
    if (!p->core) return NULL;

    srp_region = pool_carve(&cursor, 1, srp_bytes);
    if (!srp_region) return NULL;
    p->srp = srp_init(srp_region, srp_bytes, &srp_cfg, &geom);
    if (!p->srp) return NULL;
    /* BF never owns a_array; it points into pool memory tied to p. */
    p->bf_steering = p->srp->a_array;
    if (!p->bf_steering) return NULL;

    bf_region = pool_carve(&cursor, 1, bf_bytes);
    if (!bf_region) return NULL;
    p->bf = bf_init(
        bf_region, bf_bytes, FOUR_AEC_NR_RES_CHANNELS, n_freqs,
        cfg_copy.num_angles, p->bf_steering, &bf_cfg);
    if (!p->bf) return NULL;

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    p->bf_spectrum =
        (Complex*)pool_carve(&cursor, (size_t)n_freqs, sizeof(Complex));
    p->bf_weights =
        (Complex*)pool_carve(&cursor, spectral_count, sizeof(Complex));
    if (!p->bf_spectrum || !p->bf_weights) return NULL;

    p->hop_size = hop;
    p->fft_size = fft;
    p->n_freqs = n_freqs;

    /* auto_vad_hangover_frames and the speech/silence noise-EMA pairs are retimed to the
     * actual grid like bf_lambda; the retention factor is retimed and new_weight derived so
     * old + new == 1 holds. */
    p->vad_hangover_frames = anr_retime_frames(
        cfg_copy.auto_vad_hangover_frames, cfg_copy.core.sample_rate, hop);
    p->vad_speech_noise_keep = anr_retime_alpha(
        0.999f, cfg_copy.core.sample_rate, hop);
    p->vad_speech_new_weight = 1.0f - p->vad_speech_noise_keep;
    p->vad_silence_noise_keep = anr_retime_alpha(
        0.95f, cfg_copy.core.sample_rate, hop);
    p->vad_silence_new_weight = 1.0f - p->vad_silence_noise_keep;
    p->vad_power_threshold =
        powf(10.0f, cfg_copy.auto_vad_threshold_dbfs / 10.0f);
    p->noise_power = p->vad_power_threshold;
    p->vad_hangover = 0;
    p->frame_index = 0;
    p->owned_heap = NULL;
    p->destroyed = 0;

    /* The sizing walk and the carves must consume exactly current.bytes. */
    if (cursor.remaining != 0) return NULL;
    return p;
}

AudioPipeline4Ch* audio_pipeline_4ch_init(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg) {
    return audio_pipeline_4ch_init_ex(mem, bytes, cfg, NULL);
}

AudioPipeline4Ch* audio_pipeline_4ch_create(
    const AudioPipeline4ChConfig* cfg) {
    AudioPipeline4ChMemReq req;
    AudioPipeline4Ch* p;
    void* pool = NULL;
    if (!cfg ||
        audio_pipeline_4ch_get_mem_requirements(cfg, &req) != 0 ||
        req.bytes > (uint64_t)SIZE_MAX)
        return NULL;
    if (posix_memalign(
            &pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool)
        return NULL;
    p = audio_pipeline_4ch_init(pool, (size_t)req.bytes, cfg);
    if (!p) {
        free(pool);
        return NULL;
    }
    p->owned_heap = pool;
    return p;
}

static void fill_frame_info(
    AudioPipeline4Ch* p,
    const FourAecFront* front,
    int vad_raw,
    int vad_out,
    int doa_analysis_frames,
    AudioPipeline4ChFrameInfo* info) {
    if (!info) return;
    info->frame_index = p->frame_index;
    info->delay = front->delay;
    info->doa_raw_rad = doa_get_raw(p->srp);
    info->doa_smooth_rad = doa_get_smooth(p->srp);
    info->doa_used_rad = bf_get_doa_used(p->bf);
    info->vad_raw = vad_raw;
    info->vad_out = vad_out;
    info->bf_adaptive = bf_get_adaptive(p->bf);
    info->doa_analysis_frames = doa_analysis_frames;
}

int audio_pipeline_4ch_process_front(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    FourAecFront* out) {
    int status;
    if (!p || p->destroyed || !microphones_interleaved || !far_reference || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    status = four_aec_front_process(
        p->core, microphones_interleaved, far_reference, out);
    if (status != FOUR_AEC_NR_RES_OK) {
        /* The core dropped its front view on any refusal, so no beam or fuse may follow. */
        p->phase = 0;
        return status;
    }
    if (out->n_channels != FOUR_AEC_NR_RES_CHANNELS ||
        out->hop_size != p->hop_size ||
        out->n_freqs != p->n_freqs) {
        audio_pipeline_4ch_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    p->phase = 1;
    return FOUR_AEC_NR_RES_OK;
}

int audio_pipeline_4ch_process_beam(
    AudioPipeline4Ch* p,
    const FourAecFront* front,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    AudioPipeline4ChBeam* beam,
    AudioPipeline4ChFrameInfo* info) {
    int m;
    if (!p || p->destroyed || !front || !beam || !se_flag_valid(vad_raw) ||
        !se_flag_valid(vad_out) || front->n_freqs != p->n_freqs)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    for (m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m)
        if (!front->linear_spectra[m]) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->phase != 1) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;

    /* front->linear_spectra[ch] borrows each lane's live spectrum; SRP/BF only read it, so no
     * copy is needed. */
    doa_step(
        p->srp, front->linear_spectra, frequency_mask,
        vad_raw, vad_out);
    bf_process_with_weights(
        p->bf, front->linear_spectra, doa_get_smooth(p->srp),
        vad_out ? 0 : 1, frequency_mask, p->bf_spectrum,
        p->bf_weights);

    beam->spectrum = p->bf_spectrum;
    beam->weights = p->bf_weights;
    fill_frame_info(p, front, vad_raw, vad_out, 1, info);
    p->frame_index += 1;
    p->phase = 2;
    return FOUR_AEC_NR_RES_OK;
}

int audio_pipeline_4ch_process_with_activity(
    AudioPipeline4Ch* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_raw,
    int vad_out,
    const int* frequency_mask,
    AudioPipeline4ChBeam* beam,
    AudioPipeline4ChFrameInfo* info) {
    FourAecFront front;
    int status;
    if (!p || p->destroyed || !microphones_interleaved || !far_reference ||
        !beam || !se_flag_valid(vad_raw) || !se_flag_valid(vad_out)) {
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    }
    status = audio_pipeline_4ch_process_front(
        p, microphones_interleaved, far_reference, &front);
    if (status != FOUR_AEC_NR_RES_OK) return status;
    return audio_pipeline_4ch_process_beam(
        p, &front, vad_raw, vad_out, frequency_mask, beam, info);
}

int audio_pipeline_4ch_fuse(AudioPipeline4Ch* p, SeSpecFrame* frame) {
    int status;
    if (!p || p->destroyed || !frame || !p->cfg.core.enable_post)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->phase != 2) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    /* bf_spectrum and bf_weights come atomically from one bf_process_with_weights() call, so
     * the beamformed error is handed over as is. No finite check here: four_aec_fuse()
     * re-validates both arrays on every call. A refused fuse takes the whole unit down
     * (the core resets itself), not only the core; a sequence refusal changed nothing. */
    status = four_aec_fuse(p->core, p->bf_weights, p->bf_spectrum, frame);
    if (status != FOUR_AEC_NR_RES_OK && status != FOUR_AEC_NR_RES_SEQUENCE_ERROR)
        audio_pipeline_4ch_reset(p);
    return status;
}

static int floats_finite(const float* values, size_t count) {
    size_t i;
    if (!values) return 0;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static int auto_vad(
    AudioPipeline4Ch* p, const float* microphones_interleaved) {
    float sum = 0.0f;
    float power;
    float threshold = p->vad_power_threshold;
    int speech;
    size_t count =
        (size_t)p->hop_size * FOUR_AEC_NR_RES_CHANNELS;

    for (size_t i = 0; i < count; ++i) {
        float value = microphones_interleaved[i];
        sum += value * value;
    }
    power = sum / (float)count;
    if (!isfinite(power)) {
        /* A non-finite input hop must not touch any VAD state; only p->vad_hangover is read, so
         * the next valid frame decides as if this one never happened. */
        return p->vad_hangover > 0;
    }
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
    AudioPipeline4ChBeam* beam,
    AudioPipeline4ChFrameInfo* info) {
    int speech;
    if (!p || p->destroyed || !microphones_interleaved || !far_reference ||
        !beam)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    /* auto_vad() reads only the microphones, so a non-finite far_reference would otherwise
     * move noise_power/vad_hangover before the deeper finite check notices it. Reject before
     * that call so an invalid call touches no VAD state. */
    if (!floats_finite(far_reference, (size_t)p->hop_size))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    speech = auto_vad(p, microphones_interleaved);
    return audio_pipeline_4ch_process_with_activity(
        p, microphones_interleaved, far_reference,
        speech, speech, NULL, beam, info);
}

void audio_pipeline_4ch_reset(AudioPipeline4Ch* p) {
    if (!p || p->destroyed) return;
    four_aec_nr_res_reset(p->core);
    srp_reset(p->srp);
    bf_reset(p->bf);
    p->phase = 0;
    p->vad_power_threshold =
        powf(10.0f, p->cfg.auto_vad_threshold_dbfs / 10.0f);
    p->noise_power = p->vad_power_threshold;
    p->vad_hangover = 0;
    p->frame_index = 0;
}

void audio_pipeline_4ch_destroy(AudioPipeline4Ch* p) {
    void* owned_heap;
    if (!p || p->destroyed) return;
    /* BF borrows the steering table, so destroy it first. The sub-destroys are no-ops for
     * pool-carved objects; only this wrapper's own owned_heap (create()) is freed. */
    bf_destroy(p->bf);
    srp_destroy(p->srp);
    four_aec_nr_res_destroy(p->core);
    owned_heap = p->owned_heap;
    p->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

int audio_pipeline_4ch_hop_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->hop_size : -1;
}

int audio_pipeline_4ch_frame_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->fft_size : -1;
}

int audio_pipeline_4ch_fft_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->fft_size : -1;
}

int audio_pipeline_4ch_n_freqs(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->n_freqs : -1;
}

int audio_pipeline_4ch_sample_rate(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? four_aec_nr_res_sample_rate(p->core) : -1;
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

int audio_pipeline_4ch_bf_sample_rate(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->cfg.core.sample_rate : -1;
}

int audio_pipeline_4ch_bf_frame_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->fft_size : -1;
}

int audio_pipeline_4ch_bf_hop_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->hop_size : -1;
}

int audio_pipeline_4ch_bf_fft_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->fft_size : -1;
}

int audio_pipeline_4ch_bf_effective_adapt_interval(
    const AudioPipeline4Ch* p) {
    return (p && !p->destroyed && p->bf) ? p->bf->adapt_interval : -1;
}

float audio_pipeline_4ch_bf_lambda(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed && p->bf) ? p->bf->lambda : NAN;
}

int audio_pipeline_4ch_matched_filter_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_matched_filter_count(p->core) : 0;
}

int audio_pipeline_4ch_linear_aec_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_linear_aec_count(p->core) : 0;
}

const FourAecNrRes* audio_pipeline_4ch_get_core(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->core : NULL;
}

void audio_pipeline_4ch_last_core_timing(const AudioPipeline4Ch* p,
                                         FourAecNrResLastTiming* out) {
    if (!out) return;
    if (!p || p->destroyed) {
        memset(out, 0, sizeof(*out));
        return;
    }
    four_aec_nr_res_get_last_timing(p->core, out);
}

const char* audio_pipeline_4ch_spatial_backend(void) {
    return spatial_simd_backend();
}
