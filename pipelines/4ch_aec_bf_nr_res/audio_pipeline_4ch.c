/**
 * audio_pipeline_4ch.c — complete four-channel spatial pipeline.
 *
 * This is the deployable orchestrator around 4aec_nr_res.c:
 *
 *   four_aec_front_process -> SRP-PHAT DOA -> GSC effective weights
 *   -> four_aec_fuse -> RES/NR gains -> CNG -> ISTFT
 *
 * It follows the same pool-first layout and lifecycle as 4aec_nr_res.c
 * (descriptor-tier: get_mem_requirements()/init_ex() compose the core's own
 * descriptor-tier API with SRP/GSC's simple size_t tier plus this wrapper's
 * own scratch), with create() as a heap convenience wrapper over that same
 * pool-first implementation. See audio_pipeline_4ch.h's own doc comment for
 * the caller-pool vs. heap usage patterns.
 *
 * The file follows audio_pipeline.c's order: instance, config validation,
 * default config, pool-first construction, heap convenience, processing,
 * reset/destroy, and accessors.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_pipeline_4ch.h"
#include "gsc.h"
#include "srp.h"
#include "spatial_simd.h"
#include "mem_align.h"
#include "audio_stft.h"         /* AcIstft: inverse FFT + sqrt-Hann OLA */
#include "simd_kernels.h"       /* sk_min_f32 / sk_capply_gain_f32 */
#include "common/se_stage.h"    /* SeSpecFrame, se_res_stage_gain, comfort-noise constants */
#include "common/se_anr_stage.h"/* NR stage */

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
    Complex* gsc_spectrum;
    Complex* gsc_weights;

    /* Post-beam stages, chained in audio_pipeline_4ch_process_with_activity(). */
    AecRes* res;                /* iff cfg.core.enable_res */
    AnrDenoiser* nr;            /* iff cfg.core.enable_nr  */
    AcIstft* istft;             /* inverse FFT, sqrt-Hann window, overlap-add */
    float* total_gain;          /* min(G_nr, G_res)                         */
    float* extra_noise;         /* R^2 prior scratch of the denoiser         */
    Complex* spec;              /* E * total_gain (+ comfort noise)          */
    float* unity_gain;          /* iff !enable_res or !enable_nr: all-ones   */
    uint32_t rng_state;         /* comfort-noise generator                   */

    float noise_power;
    float vad_power_threshold;    /* live-computed from cfg.auto_vad_threshold_dbfs,
                                    * cached so auto_vad()'s per-hop path doesn't
                                    * re-derive a config-only constant every call */
    int vad_hangover;
    int vad_hangover_frames;      /* live-computed, was raw cfg.auto_vad_hangover_frames */
    float vad_speech_noise_keep;  /* live-computed, was raw literal 0.999f */
    float vad_speech_new_weight;  /* live-computed, was raw literal 0.001f */
    float vad_silence_noise_keep; /* live-computed, was raw literal 0.95f */
    float vad_silence_new_weight; /* live-computed, was raw literal 0.05f */
    uint64_t frame_index;

    /* Non-NULL only on the audio_pipeline_4ch_create() heap path: the single
     * posix_memalign()'d block backing this whole struct plus every carved
     * sub-region below it (core/srp/gsc/scratch), freed by
     * audio_pipeline_4ch_destroy(). NULL on the audio_pipeline_4ch_init()
     * caller-pool path, where the caller owns the memory and destroy() must
     * not free it. Same convention as FourAecNrRes::owned_heap/SRP::
     * owned_heap/GSC::owned_heap one layer down. */
    void* owned_heap;
    int destroyed;
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
    /* The standard wrapper always runs four_aec_fuse(). Front-only is a
     * direct-core/internal ULCNet profile, not a valid standard-wrapper cfg. */
    if (cfg->core.enable_post != 1) return 0;
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

/* ============================================================================
 * Pool-first construction: file-private PoolCursor/pool_carve(), same
 * bump-allocator shape kept file-private by 4aec_nr_res.c/srp.c/gsc.c (each
 * keeps its own copy per those files' own precedent, not shared).
 * ========================================================================== */

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

/* Plain FNV-style integer mix, no snprintf/stdio -- this path must stay
 * NO_STDIO-safe (see Makefile's audit-no-stdio). Folds in the core layer's
 * own build_flags_hash, so a change to the core's CARVE TOKEN propagates
 * here. A core LAYOUT VERSION bump does not: the core's hash is taken over
 * its backend string, carve token and alignment only, so a control-block-only
 * growth leaves it -- and therefore this hash -- unchanged. Only bumping
 * AUDIO_PIPELINE_4CH_LAYOUT_VERSION by hand invalidates a persisted composite
 * descriptor -- see AudioPipeline4ChMemReq's doc comment, which states this
 * correctly. */
static uint32_t audio_pipeline_4ch_build_flags_hash(
    uint32_t core_build_flags_hash) {
    uint32_t hash = 2166136261u;
    hash = fnv1a_str(
        "|carve:self(corecfg-delay-v2),core,srp,gsc,gsc_spectrum,gsc_weights",
        hash);
    hash = fnv1a_str("|align16", hash);
    hash ^= core_build_flags_hash;
    hash *= 16777619u;
    return hash;
}

/* ============================================================================
 * Stack-only array geometry (no malloc)
 *
 * Fill formulas copied verbatim from third_party/doa/steering.c's
 * array_geometry_create_uca/_ula/_custom() so a stack-local ArrayGeometry
 * here produces bit-identical x/y coordinates to what the malloc-based
 * geometry helper produces -- calling that helper (even though its result
 * would be freed immediately after use) would otherwise be a hidden
 * allocation inside what is supposed to be a zero-allocator init_ex() path.
 * M is kept as a local int matching the originals' own locals, so the
 * float/double promotion sequence in the UCA phi expression stays identical.
 * ========================================================================== */

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

/* Builds geom in place from caller-supplied x/y storage (sized
 * FOUR_AEC_NR_RES_CHANNELS, a compile-time constant -- no VLA needed).
 * ArrayGeometry is never stored long-term (SRP does not keep a pointer to
 * it, only reads geom->x/geom->y transiently inside srp_init()), so a
 * stack-local temporary that goes out of scope right after the srp_init()
 * call below is safe. */
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

/* ============================================================================
 * Config -> SRP/GSC module configs (shared by get_mem_requirements/init_ex)
 * ========================================================================== */

/*
 * Builds srp_cfg/gsc_cfg from cfg plus the core's own already-derived
 * hop/fft/n_freqs -- mirrors 4aec_nr_res.c's derive_dims_and_configs()'s role
 * of "compute once, reuse in both sizing and init", so
 * audio_pipeline_4ch_get_mem_requirements() and audio_pipeline_4ch_init_ex()
 * can never silently diverge on this logic.
 */
static int derive_spatial_configs(
    const AudioPipeline4ChConfig* cfg,
    int hop, int fft, int n_freqs,
    SRP_Config* srp_cfg, GSC_Config* gsc_cfg) {
    int gsc_effective_interval;
    if (!cfg || !srp_cfg || !gsc_cfg) return 0;

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

    /* gsc_create()/gsc_init() force the RLS update cadence to 1 (every hop)
     * whenever fixed-notebook mode is requested (gsc_fixed_mode &&
     * gsc_fixed_align_notebook), regardless of the caller's configured
     * gsc_adapt_interval -- see gsc.c's gsc_effective_adapt_interval().
     * Derive that SAME effective value here, once, and feed it into both the
     * lambda retime scaling below and gsc_cfg->adapt_interval, so the
     * cadence lambda is calibrated for and the cadence GSC actually runs at
     * can never silently diverge (scaling lambda by the raw pre-forced
     * gsc_adapt_interval while gsc_create() forces the real cadence to 1
     * would mis-calibrate it). */
    gsc_effective_interval = gsc_effective_adapt_interval(
        cfg->gsc_fixed_mode, cfg->gsc_fixed_align_notebook,
        cfg->gsc_adapt_interval);

    memset(gsc_cfg, 0, sizeof(*gsc_cfg));
    gsc_cfg->enable = cfg->gsc_enable;
    /* gsc_lambda is an RLS forgetting/retention factor tuned at a 10-ms
     * reference update (like NR's alpha_d/alpha_attack in the mono/4ch NR
     * config derivation) -- its real wall-clock forgetting time varies with
     * hop_size/sample_rate, so it is retimed via the same helper NR uses for
     * this exact class of constant. mu is a step-size gain, not a decay
     * time-constant, so it is left as-is.
     *
     * The RLS update itself only actually applies once every
     * gsc_effective_interval hops (gsc.c gates the whole P/gain/weight-update
     * block on frame_idx % adapt_interval == 0; default adapt_interval=1,
     * i.e. every hop). Unlike AEC's ERLE 6-point cadence, the ORIGINAL
     * "10ms reference" tuning assumed one update per hop (interval=1) --
     * there is no matching batching on the authored side to cancel against
     * -- so when the effective interval > 1 the real wall-clock update
     * period is that many hops, and the retime call must scale hop by it
     * (no-op at the shipped default of 1). */
    gsc_cfg->lambda = anr_retime_alpha(
        cfg->gsc_lambda, cfg->core.sample_rate,
        hop * gsc_effective_interval);
    gsc_cfg->mu = cfg->gsc_mu;
    gsc_cfg->enable_fix_mode = cfg->gsc_fixed_mode;
    gsc_cfg->fixed_doa_rad = cfg->gsc_fixed_doa_rad;
    gsc_cfg->fixed_align_notebook = cfg->gsc_fixed_align_notebook;
    gsc_cfg->adapt_interval = gsc_effective_interval;
    return 1;
}

/* ============================================================================
 * Memory sizing -- compute once, reuse in get_mem_requirements()/init_ex()
 * ========================================================================== */

/* The post-beam stages this wrapper carves after the core: sizes and the
 * configurations their init needs. */
typedef struct PostStages {
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    size_t res_bytes;     /* 0 when cfg.core.enable_res = 0 */
    size_t nr_bytes;      /* 0 when cfg.core.enable_nr  = 0 */
    size_t synth_bytes;
    int need_unity;
} PostStages;

static int derive_post_stages(const AudioPipeline4ChConfig* cfg, PostStages* ps) {
    AcStftConfig scfg;
    int fft;
    FourAecNrResMemBreakdown bd;
    if (four_aec_nr_res_stage_configs(&cfg->core, &ps->res_cfg, &ps->nr_cfg) != 0 ||
        four_aec_nr_res_get_mem_breakdown(&cfg->core, &bd) != 0) return 0;
    fft = bd.fft_size;
    ps->res_bytes = cfg->core.enable_res ? aec_res_get_mem_size(&ps->res_cfg) : 0;
    ps->nr_bytes = cfg->core.enable_nr ? se_anr_stage_mem_size(&ps->nr_cfg) : 0;
    ac_stft_config_default(&scfg, fft);
    ps->synth_bytes = ac_istft_get_mem_size(&scfg);
    ps->need_unity = !cfg->core.enable_res || !cfg->core.enable_nr;
    if (ps->synth_bytes == 0 ||
        (cfg->core.enable_res && ps->res_bytes == 0) ||
        (cfg->core.enable_nr && ps->nr_bytes == 0)) return 0;
    return 1;
}

static int derive_pipeline_layout(
    const AudioPipeline4ChConfig* cfg,
    FourAecNrResMemReq* core_req,
    SRP_Config* srp_cfg, GSC_Config* gsc_cfg,
    PostStages* ps,
    int* hop, int* fft, int* n_freqs,
    size_t* srp_bytes, size_t* gsc_bytes) {
    FourAecNrResMemBreakdown breakdown;
    if (!cfg || !core_req || !srp_cfg || !gsc_cfg || !ps || !hop || !fft ||
        !n_freqs || !srp_bytes || !gsc_bytes) return 0;
    if (!validate_config(cfg)) return 0;
    if (four_aec_nr_res_get_mem_requirements(&cfg->core, core_req) != 0)
        return 0;
    /* four_aec_nr_res_get_mem_breakdown() already derives hop/fft/n_freqs
     * from cfg alone -- no live core instance required. */
    if (four_aec_nr_res_get_mem_breakdown(&cfg->core, &breakdown) != 0)
        return 0;
    *hop = breakdown.hop_size;
    *fft = breakdown.fft_size;
    *n_freqs = breakdown.n_freqs;
    if (!derive_post_stages(cfg, ps)) return 0;
    if (!derive_spatial_configs(cfg, *hop, *fft, *n_freqs, srp_cfg, gsc_cfg))
        return 0;
    *srp_bytes = srp_get_mem_size(srp_cfg);
    *gsc_bytes = gsc_get_mem_size(FOUR_AEC_NR_RES_CHANNELS, *n_freqs);
    if (*srp_bytes == 0 || *gsc_bytes == 0) return 0;
    return 1;
}

int audio_pipeline_4ch_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChMemReq* out) {
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    PostStages ps;
    int hop, fft, n_freqs;
    size_t srp_bytes, gsc_bytes;
    size_t spectral_count;
    size_t total;

    if (!out || !derive_pipeline_layout(
            cfg, &core_req, &srp_cfg, &gsc_cfg, &ps,
            &hop, &fft, &n_freqs, &srp_bytes, &gsc_bytes)) return -1;
    if (core_req.bytes > (uint64_t)SIZE_MAX) return -1;

    total = ck_align16_size(sizeof(AudioPipeline4Ch));
    total = ck_add_size(total, ck_align16_size((size_t)core_req.bytes));
    total = ck_add_size(total, ck_align16_size(srp_bytes));
    total = ck_add_size(total, ck_align16_size(gsc_bytes));

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    total = ck_field_size(
        total, (size_t)n_freqs, sizeof(Complex));       /* gsc_spectrum */
    total = ck_field_size(
        total, spectral_count, sizeof(Complex));        /* gsc_weights */

    /* Post-beam stages, in carve order. */
    if (ps.res_bytes) total = ck_add_size(total, ck_align16_size(ps.res_bytes));
    if (ps.nr_bytes) total = ck_add_size(total, ck_align16_size(ps.nr_bytes));
    total = ck_add_size(total, ck_align16_size(ps.synth_bytes));
    total = ck_field_size(total, (size_t)n_freqs, sizeof(float));    /* total_gain  */
    total = ck_field_size(total, (size_t)n_freqs, sizeof(float));    /* extra_noise */
    total = ck_field_size(total, (size_t)n_freqs, sizeof(Complex));  /* spec        */
    if (ps.need_unity)
        total = ck_field_size(total, (size_t)n_freqs, sizeof(float)); /* unity_gain */

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

/* ============================================================================
 * Caller-pool init and heap convenience construction
 * ========================================================================== */

AudioPipeline4Ch* audio_pipeline_4ch_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChMemReq* expected) {
    AudioPipeline4ChConfig cfg_copy;
    AudioPipeline4ChMemReq current;
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    PostStages ps;
    int hop, fft, n_freqs;
    size_t srp_bytes, gsc_bytes;
    float geom_x[FOUR_AEC_NR_RES_CHANNELS];
    float geom_y[FOUR_AEC_NR_RES_CHANNELS];
    ArrayGeometry geom;
    AudioPipeline4Ch* p;
    PoolCursor cursor;
    void* core_region;
    void* srp_region;
    void* gsc_region;
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
            &cfg_copy, &core_req, &srp_cfg, &gsc_cfg, &ps,
            &hop, &fft, &n_freqs, &srp_bytes, &gsc_bytes)) return NULL;
    if (core_req.bytes > (uint64_t)SIZE_MAX) return NULL;
    if (!fill_stack_geometry(&cfg_copy, geom_x, geom_y, &geom)) return NULL;

    memset(mem, 0, (size_t)current.bytes);
    p = (AudioPipeline4Ch*)mem;
    p->cfg = cfg_copy;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(*p));
    cursor.remaining = (size_t)current.bytes - ALIGN16(sizeof(*p));

    /* Composition order: query each sub-module's own size -> carve an
     * equal-sized sub-region from this cursor -> call that sub-module's own
     * init/init_ex. This is the same technique 4aec_nr_res.c's
     * pipeline_build() uses to carve its four Aec lanes. Carving core first
     * is what actually switches this wrapper from four_aec_nr_res_create()
     * (heap) to four_aec_nr_res_init_ex() (pool). */
    core_region = pool_carve(&cursor, 1, (size_t)core_req.bytes);
    if (!core_region) return NULL;
    p->core = four_aec_nr_res_init_ex(
        core_region, (size_t)core_req.bytes, &cfg_copy.core, &core_req);
    if (!p->core) return NULL;

    srp_region = pool_carve(&cursor, 1, srp_bytes);
    if (!srp_region) return NULL;
    p->srp = srp_init(srp_region, srp_bytes, &srp_cfg, &geom);
    if (!p->srp) return NULL;
    /* Same borrow relationship as the heap path: GSC never owns/frees
     * a_array. It points into pool memory with a lifetime tied to p
     * itself, rather than to a separately heap-managed SRP. */
    p->gsc_steering = p->srp->a_array;
    if (!p->gsc_steering) return NULL;

    gsc_region = pool_carve(&cursor, 1, gsc_bytes);
    if (!gsc_region) return NULL;
    p->gsc = gsc_init(
        gsc_region, gsc_bytes, FOUR_AEC_NR_RES_CHANNELS, n_freqs,
        cfg_copy.num_angles, p->gsc_steering, &gsc_cfg);
    if (!p->gsc) return NULL;

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    p->gsc_spectrum =
        (Complex*)pool_carve(&cursor, (size_t)n_freqs, sizeof(Complex));
    p->gsc_weights =
        (Complex*)pool_carve(&cursor, spectral_count, sizeof(Complex));
    if (!p->gsc_spectrum || !p->gsc_weights) return NULL;

    /* Post-beam stages. */
    if (cfg_copy.core.enable_res) {
        void* res_region = pool_carve(&cursor, 1, ps.res_bytes);
        if (!res_region) return NULL;
        p->res = aec_res_init(res_region, ps.res_bytes, &ps.res_cfg);
        if (!p->res || aec_res_n_bins(p->res) != n_freqs ||
            aec_res_hop_size(p->res) != hop) return NULL;
    }
    if (cfg_copy.core.enable_nr) {
        void* nr_region = pool_carve(&cursor, 1, ps.nr_bytes);
        if (!nr_region) return NULL;
        p->nr = se_anr_stage_init(nr_region, ps.nr_bytes, &ps.nr_cfg);
        if (!p->nr || anr_get_hop_size(p->nr) != hop ||
            anr_get_n_freqs(p->nr) != n_freqs) return NULL;
    }
    {
        AcStftConfig scfg;
        void* synth_region = pool_carve(&cursor, 1, ps.synth_bytes);
        if (!synth_region) return NULL;
        ac_stft_config_default(&scfg, fft);
        p->istft = ac_istft_init(synth_region, ps.synth_bytes, &scfg);
        if (!p->istft) return NULL;
    }
    p->total_gain = (float*)pool_carve(&cursor, (size_t)n_freqs, sizeof(float));
    p->extra_noise = (float*)pool_carve(&cursor, (size_t)n_freqs, sizeof(float));
    p->spec = (Complex*)pool_carve(&cursor, (size_t)n_freqs, sizeof(Complex));
    if (!p->total_gain || !p->extra_noise || !p->spec) return NULL;
    if (ps.need_unity) {
        int k;
        p->unity_gain = (float*)pool_carve(&cursor, (size_t)n_freqs, sizeof(float));
        if (!p->unity_gain) return NULL;
        for (k = 0; k < n_freqs; ++k) p->unity_gain[k] = 1.0f;
    }
    p->rng_state = SE_CNG_RNG_SEED;

    p->hop_size = hop;
    p->fft_size = fft;
    p->n_freqs = n_freqs;

    /* auto_vad_hangover_frames + the speech/silence noise-EMA pairs are
     * authored literals (8 frames; 0.999/0.001; 0.95/0.05) and are retimed to
     * the actual grid, as gsc_lambda is in derive_spatial_configs(). The EMA pairs
     * are retention/new-weight complements (old_weight + new_weight == 1 by
     * construction); retime the retention factor and derive new_weight from
     * it so that invariant still holds post-retime. */
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

    /* Lockstep proof: audio_pipeline_4ch_get_mem_requirements()'s walk and
     * the carves above must consume exactly current.bytes, no more, no
     * less -- every sub-pointer above is already non-NULL-checked. */
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

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

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
    info->doa_used_rad = gsc_get_doa_used(p->gsc);
    info->vad_raw = vad_raw;
    info->vad_out = vad_out;
    info->gsc_adaptive = gsc_get_adaptive(p->gsc);
    info->doa_analysis_frames = doa_analysis_frames;
}

/* The frequency-domain stages after the fuse, written out: RES gain, NR gain,
 * min of the two, apply, comfort noise on the RES-cut bins, synthesis.
 * Returns FOUR_AEC_NR_RES_OK, or DSP_ERROR when a stage refused the hop or
 * the synthesised hop is not finite. */
static int run_post_stages(AudioPipeline4Ch* p, const SeSpecFrame* frame,
                           float* output) {
    const float* res_gain;
    const float* nr_gain;
    int n = p->n_freqs;
    int k;

    /* A disabled gain source reads as the unity vector, so one min serves
     * every combination: min(1, G_res) is G_res and min(G_nr, 1) is G_nr.
     * sk_min_f32 is the mono twin's kernel: a non-finite gain propagates to
     * the output finite gate instead of being masked the way fminf would
     * mask it. */
    res_gain = p->res ? se_res_stage_gain(p->res, frame) : p->unity_gain;
    if (!res_gain) return FOUR_AEC_NR_RES_DSP_ERROR;
    nr_gain = p->unity_gain;
    if (p->nr) {
        /* The denoiser's residual-echo prior is the fused R^2 (the stage
         * rescales it into extra_noise); legacy_amin hands it no prior. */
        nr_gain = se_anr_stage_gain(p->nr, frame, p->extra_noise,
                                    !p->cfg.core.legacy_amin);
        if (!nr_gain) return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    sk_min_f32(p->total_gain, nr_gain, res_gain, n);
    sk_capply_gain_f32(p->spec, frame->e, p->total_gain, n);

    /* Comfort noise on the RES-cut bins, scaled by G_nr (bounded below by
     * SE_CNG_NR_GAIN_FLOOR) so the fill follows the denoised floor rather
     * than the ambient level: the denoiser never sees the comfort noise, so
     * its gain is applied to it here. */
    if (p->cfg.core.enable_cng && p->res)
        aec_cng_fill(&p->rng_state, p->spec, frame->cn, res_gain, nr_gain,
                     SE_CNG_NR_GAIN_FLOOR, n);

    ac_istft_push(p->istft, p->spec, output);
    for (k = 0; k < p->hop_size; ++k)
        if (!isfinite(output[k])) return FOUR_AEC_NR_RES_DSP_ERROR;
    return FOUR_AEC_NR_RES_OK;
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
    FourAecFront front;
    SeSpecFrame frame;
    int status;
    int doa_analysis_frames;
    if (!p || p->destroyed || !microphones_interleaved || !far_reference ||
        !output || !is_bool(vad_raw) || !is_bool(vad_out)) {
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    }
    status = four_aec_front_process(
        p->core, microphones_interleaved, far_reference, &front);
    if (status != FOUR_AEC_NR_RES_OK) return status;
    if (front.n_channels != FOUR_AEC_NR_RES_CHANNELS ||
        front.hop_size != p->hop_size ||
        front.n_freqs != p->n_freqs) {
        audio_pipeline_4ch_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    /* front.linear_spectra[ch] borrows each lane's own live spectrum
     * (const Complex*) -- SRP/GSC only ever read through X, never write, so
     * feeding them this array directly needs no per-hop copy into a
     * wrapper-owned scratch buffer. */
    doa_step(
        p->srp, front.linear_spectra, frequency_mask,
        vad_raw, vad_out);
    doa_analysis_frames = 1;
    gsc_process_with_weights(
        p->gsc, front.linear_spectra, doa_get_smooth(p->srp),
        vad_out ? 0 : 1, frequency_mask, p->gsc_spectrum,
        p->gsc_weights);

    /* gsc_spectrum and gsc_weights were produced atomically by the same
     * gsc_process_with_weights() call above. Hand the mono error over as it
     * is instead of reconstructing one weighted sum a second time; the fuse
     * still projects near/R2/comfort with those exact weights.
     *
     * No finite check here: four_aec_fuse() unconditionally re-validates the
     * identical arrays on every call (weights, and the spectrum in the
     * projection) before either is read, and the status check below resets
     * on any failure -- a duplicate scan here can only ever agree with that
     * authoritative check, never catch something it would miss. */
    status = four_aec_fuse(p->core, p->gsc_weights, p->gsc_spectrum, &frame);
    if (status == FOUR_AEC_NR_RES_OK)
        status = run_post_stages(p, &frame, output);
    if (status != FOUR_AEC_NR_RES_OK) {
        audio_pipeline_4ch_reset(p);
        return status;
    }
    fill_frame_info(
        p, &front, vad_raw, vad_out, doa_analysis_frames, info);
    p->frame_index += 1;
    return FOUR_AEC_NR_RES_OK;
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
        /* A non-finite input hop must never touch any VAD state -- not
         * noise_power's EMA, not vad_hangover, nothing. The deeper pipeline
         * rejects this frame on its own finite check regardless of what
         * this function returns, so the return value itself is never
         * actually consumed; only reading (never writing) p->vad_hangover
         * here matters, so the NEXT valid frame's decision is computed as
         * if this rejected one never happened. */
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
    float* output,
    AudioPipeline4ChFrameInfo* info) {
    int speech;
    if (!p || p->destroyed || !microphones_interleaved || !far_reference ||
        !output)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    /* auto_vad() only reads microphones_interleaved (and guards itself
     * against non-finite mic data), so a non-finite far_reference sails
     * through unnoticed here and still reaches auto_vad() below -- on
     * perfectly finite mic data, auto_vad() runs its normal (state-
     * mutating) path and only the deeper finite check inside
     * four_aec_front_process() ever notices far_reference was bad,
     * by which point noise_power/vad_hangover have already moved. Reject
     * before that call, not just via a NULL check, so this invalid call
     * touches no VAD state at all -- same contract as the mic-NaN case. */
    if (!floats_finite(far_reference, (size_t)p->hop_size))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    speech = auto_vad(p, microphones_interleaved);
    return audio_pipeline_4ch_process_with_activity(
        p, microphones_interleaved, far_reference,
        speech, speech, NULL, output, info);
}

/* ============================================================================
 * Reset and teardown
 * ========================================================================== */

int audio_pipeline_4ch_set_aec_preset(AudioPipeline4Ch* p, AecPreset preset,
                                      float ramp_ms) {
    if (!p || p->destroyed || !p->res) return -1;
    /* The four lanes run with spatial_linear_context, so they never reach
     * their own suppressor and their floors shape nothing. The gain that
     * actually multiplies the output -- and scales the comfort noise -- comes
     * from the post-beam suppressor, so that is what a preset change has to
     * move. aec_res_set_preset() validates the preset and the ramp before it
     * writes, so a rejected call leaves the instance unchanged. The lanes are
     * left alone: retargeting them would be inert but would park all four
     * permanently mid-ramp. */
    if (aec_res_set_preset(p->res, preset, ramp_ms) != 0) return -1;
    p->cfg.core.aec_preset = preset;
    return 0;
}

int audio_pipeline_4ch_post_split_floor(const AudioPipeline4Ch* p,
                                        float* live, float* target) {
    if (!p || p->destroyed || !p->res) return -1;
    return aec_res_split_floor(p->res, live, target);
}

int audio_pipeline_4ch_set_nr_mode(AudioPipeline4Ch* p, AnrMode mode) {
    if (!p || p->destroyed || !p->nr) return -1;
    if (se_anr_stage_set_mode(p->nr, p->cfg.core.sample_rate, p->fft_size,
                              p->hop_size, mode) != 0) return -1;
    p->cfg.core.nr_mode = mode;
    return 0;
}

void audio_pipeline_4ch_reset(AudioPipeline4Ch* p) {
    if (!p || p->destroyed) return;
    four_aec_nr_res_reset(p->core);
    srp_reset(p->srp);
    gsc_reset(p->gsc);
    if (p->nr) se_anr_stage_reset(p->nr);
    if (p->res) aec_res_reset(p->res);
    ac_istft_reset(p->istft);
    p->rng_state = SE_CNG_RNG_SEED;
    p->vad_power_threshold =
        powf(10.0f, p->cfg.auto_vad_threshold_dbfs / 10.0f);
    p->noise_power = p->vad_power_threshold;
    p->vad_hangover = 0;
    p->frame_index = 0;
}

void audio_pipeline_4ch_destroy(AudioPipeline4Ch* p) {
    void* owned_heap;
    if (!p || p->destroyed) return;
    /* GSC borrows the steering table; destroy it before its owner. Each of
     * these three is a no-op free on the pool-carved sub-objects this
     * wrapper builds inside its own pool/heap block (their own owned_heap
     * fields stay NULL -- see audio_pipeline_4ch_init_ex()); only this
     * wrapper's OWN owned_heap (set only by audio_pipeline_4ch_create()) is
     * ever actually freed below, exactly once. */
    gsc_destroy(p->gsc);
    srp_destroy(p->srp);
    if (p->nr) se_anr_stage_destroy(p->nr);
    four_aec_nr_res_destroy(p->core);
    owned_heap = p->owned_heap;
    p->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

/* ============================================================================
 * Read-only shape/topology accessors
 * ========================================================================== */

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

int audio_pipeline_4ch_gsc_sample_rate(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->cfg.core.sample_rate : -1;
}

int audio_pipeline_4ch_gsc_frame_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->fft_size : -1;
}

int audio_pipeline_4ch_gsc_hop_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->hop_size : -1;
}

int audio_pipeline_4ch_gsc_fft_size(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed) ? p->fft_size : -1;
}

int audio_pipeline_4ch_gsc_effective_adapt_interval(
    const AudioPipeline4Ch* p) {
    return (p && !p->destroyed && p->gsc) ? p->gsc->adapt_interval : -1;
}

float audio_pipeline_4ch_gsc_lambda(const AudioPipeline4Ch* p) {
    return (p && !p->destroyed && p->gsc) ? p->gsc->lambda : NAN;
}

int audio_pipeline_4ch_matched_filter_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_matched_filter_count(p->core) : 0;
}

int audio_pipeline_4ch_linear_aec_count(const AudioPipeline4Ch* p) {
    return p ? four_aec_nr_res_linear_aec_count(p->core) : 0;
}

int audio_pipeline_4ch_nr_count(const AudioPipeline4Ch* p) {
    return p && !p->destroyed && p->nr ? 1 : 0;
}

int audio_pipeline_4ch_post_res_count(const AudioPipeline4Ch* p) {
    return p && !p->destroyed && p->res ? 1 : 0;
}

const char* audio_pipeline_4ch_spatial_backend(void) {
    return spatial_simd_backend();
}
