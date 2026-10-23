/** Four-channel spatial pipeline with the Align-ULCNet post-filter (see the header
 * for the timing and model-callback contract). Per hop: four_aec_front_process ->
 * SRP-PHAT DOA -> BF weights -> bf_spectrum used directly as the model's error
 * frame; the caller's raw far through a rolling AcStft; one UlcnetModel call per
 * hop (identity during the post-boundary reprime or on failure); AcIstft
 * synthesis. The core runs front-only (cfg.core.enable_post = 0). The
 * spatial-config helpers are file-private copies of audio_pipeline_4ch.c's. */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_pipeline_4ch_ulcnet.h"
#include "bf.h"
#include "srp.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"    /* skn_fill_f32 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* The supported grid is the one the ULCNet layer was built for (ulcnet_model_io.h);
 * only the FFT size is checked since hop and bins derive from it. */
_Static_assert(ULCNET_N_FFT == 512 || ULCNET_N_FFT == 1024,
               "ULCNet wrapper supports FFT 512 (16 kHz) or 1024 (48 kHz)");

struct AudioPipeline4ChUlcnet {
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

    /* The carved FFT handle serves the far analysis and the synthesis (sequential
     * within a hop). */
    FftHandle* fft;

    /* Align-ULCNet chain: two framing objects carved from the pool (borrowing the
     * shared handle above and the window table below) plus the per-hop frame scratch.
     * There is no error-branch analysis: the BF spectrum is the error frame,
     * de-interleaved into err_re/err_im. */
    AcStft  *far_analysis;
    AcIstft *synthesis;
    float ulcnet_window[ULCNET_N_FFT];  /* shared sqrt-Hann table for both framing objects */
    float err_re[ULCNET_BINS];
    float err_im[ULCNET_BINS];
    float far_re[ULCNET_BINS];
    float far_im[ULCNET_BINS];
    float enh_re[ULCNET_BINS];
    float enh_im[ULCNET_BINS];

    UlcnetModel model;    /* model.infer == NULL => fail-open identity */
    FourAecNrResDelayState last_delay;
    uint64_t frame_index;

    /* Frames still straddling the last alignment boundary; while nonzero the frame is
     * the identity and the model is not stepped. Re-armed, never accumulated. */
    int reprime_frames;

    /* The hop in flight (process_begin .. process_end). */
    int phase;                    /* 0 idle, 3 front done, 1 beam done, 2 model frame open */
    int nn_wanted;                /* the frame has not been fetched and is not reprime */
    int nn_accepted;              /* the model output replaces the error frame */

    /* Non-NULL only on the create() heap path. */
    void* owned_heap;
    int destroyed;
};

/* Config validation (narrowed to this wrapper's one supported grid). */

static int is_bool(int value) {
    return value == 0 || value == 1;
}

static int validate_config(const AudioPipeline4ChConfig* cfg) {
    float nyquist;
    if (!cfg) return 0;
    /* Only the compiled ULCNet grid is accepted. */
    if (cfg->core.sample_rate != ULCNET_SR) return 0;
    if (cfg->core.fft_size != 0 &&
        cfg->core.fft_size != ULCNET_N_FFT) return 0;
    /* Post-only fields are rejected, not ignored: the config struct is shared with the
     * standard 4-channel wrapper, and with enable_post = 0 they cannot take effect. */
    if (cfg->core.enable_post != 0) return 0;
    if (cfg->core.enable_nr != 0) return 0;
    if (cfg->core.enable_res != 0) return 0;
    if (cfg->core.enable_cng != 0) return 0;
    if (cfg->core.legacy_amin != 0) return 0;
    /* AnrMode has no "disabled" value, so the default is a required sentinel. */
    if (cfg->core.nr_mode != ANR_MODE_BALANCED) return 0;
    /* The built-in VAD is unreachable (only process_with_activity() exists); compared
     * against a freshly built default so changes to the defaults stay consistent. */
    {
        AudioPipeline4ChConfig base =
            audio_pipeline_4ch_default_config(ULCNET_SR);
        if (cfg->auto_vad_threshold_dbfs != base.auto_vad_threshold_dbfs) return 0;
        if (cfg->auto_vad_snr_ratio != base.auto_vad_snr_ratio) return 0;
        if (cfg->auto_vad_hangover_frames != base.auto_vad_hangover_frames) return 0;
    }
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
    if (!is_bool(cfg->bf_enable) ||
        !isfinite(cfg->bf_lambda) || cfg->bf_lambda <= 0.0f ||
        cfg->bf_lambda > 1.0f ||
        !isfinite(cfg->bf_mu) || cfg->bf_mu < 0.0f ||
        !is_bool(cfg->bf_fixed_mode) ||
        !isfinite(cfg->bf_fixed_doa_rad) ||
        !is_bool(cfg->bf_fixed_align_notebook) ||
        cfg->bf_adapt_interval <= 0) return 0;
    /* auto_vad_* fields are unused here but validated like the standard wrapper's. */
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

AudioPipeline4ChConfig audio_pipeline_4ch_ulcnet_default_config(void) {
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(ULCNET_SR);
    cfg.core.fft_size = ULCNET_N_FFT;
    /* The front-only profile is returned here; validate_config() rejects any other
     * post-only value so a wrong expectation fails at init. */
    cfg.core.enable_post = 0;
    cfg.core.enable_nr = 0;
    cfg.core.enable_res = 0;
    cfg.core.enable_cng = 0;
    /* core.delay_backward_quarantine_enabled stays at the core default (off). */
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

/* Folds in the core's build_flags_hash; a core layout-version bump is not seen by
 * it, so bump AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION by hand. No snprintf. */
static uint32_t audio_pipeline_4ch_ulcnet_build_flags_hash(
    uint32_t core_build_flags_hash) {
    uint32_t hash = 2166136261u;
    /* Tokens name the `self` state (core-config delay fields, ULCNet chain, model copy
     * with io_descriptor, reprime counter, window table, err/far staging) and the
     * carved objects in carve order. Bump AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION
     * together with this string. */
    hash = fnv1a_str(
        "|carve:self(corecfg-delay-v2,ulcnet-direct,model(io_descriptor),"
        "reprime,ulcnet_window,frame),"
        "core,srp,bf,bf_spectrum,bf_weights,fft,far_analysis,synthesis",
        hash);
    /* The grid is stringified from the macros so builds on different grids get
     * different hashes. */
    hash = fnv1a_str("|align|ulcnet" ULCNET_GRID_TOKEN "|sharedfft", hash);
    hash ^= core_build_flags_hash;
    hash *= 16777619u;
    return ac_memreq_abi_hash(hash);
}

/* Stack-only array geometry (copies of audio_pipeline_4ch.c's helpers). */

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

/* Config -> SRP/BF module configs (copy of audio_pipeline_4ch.c's
 * derive_spatial_configs). */

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

    bf_effective_interval = bf_effective_adapt_interval(
        cfg->bf_fixed_mode, cfg->bf_fixed_align_notebook,
        cfg->bf_adapt_interval);

    memset(bf_cfg, 0, sizeof(*bf_cfg));
    bf_cfg->enable = cfg->bf_enable;
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

/* Memory sizing, computed once for get_mem_requirements()/init_ex(). */

/* Bytes of the far analysis (is_synthesis 0) or synthesis (1) object that borrows
 * the shared handle and window table. */
static size_t framing_bytes(int is_synthesis) {
    AcStftConfig sc;
    size_t own, shared;
    ulcnet_framing_config(&sc, NULL, NULL, 0);
    own = is_synthesis ? ac_istft_get_mem_size(&sc) : ac_stft_get_mem_size(&sc);
    shared = ck_align16_size(fft_get_mem_size(ULCNET_N_FFT)) +
             ck_align16_size((size_t)ULCNET_N_FFT * sizeof(float));
    return own > shared ? own - shared : 0;
}

static int derive_pipeline_layout(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChConfig* cfg_forced,
    FourAecNrResMemReq* core_req,
    SRP_Config* srp_cfg, BF_Config* bf_cfg,
    int* hop, int* fft, int* n_freqs,
    size_t* srp_bytes, size_t* bf_bytes, size_t* fft_bytes) {
    FourAecNrResMemBreakdown breakdown;
    if (!cfg || !cfg_forced || !core_req || !srp_cfg || !bf_cfg || !hop ||
        !fft || !n_freqs || !srp_bytes || !bf_bytes || !fft_bytes) return 0;
    if (!validate_config(cfg)) return 0;
    /* Force the compiled grid before core sizing so descriptor and carve agree. */
    *cfg_forced = *cfg;
    cfg_forced->core.fft_size = ULCNET_N_FFT;
    /* enable_post is already 0 (validate_config() refuses anything else). */
    if (four_aec_nr_res_get_mem_requirements(
            &cfg_forced->core, core_req) != 0)
        return 0;
    if (four_aec_nr_res_get_mem_breakdown(
            &cfg_forced->core, &breakdown) != 0)
        return 0;
    *hop = breakdown.hop_size;
    *fft = breakdown.fft_size;
    *n_freqs = breakdown.n_freqs;
    if (*hop != ULCNET_HOP || *fft != ULCNET_N_FFT ||
        *n_freqs != ULCNET_BINS) return 0;
    if (!derive_spatial_configs(
            cfg_forced, *hop, *fft, *n_freqs, srp_cfg, bf_cfg))
        return 0;
    *srp_bytes = srp_get_mem_size(srp_cfg);
    *bf_bytes = bf_get_mem_size(FOUR_AEC_NR_RES_CHANNELS, *n_freqs);
    *fft_bytes = fft_get_mem_size(*fft);
    if (*srp_bytes == 0 || *bf_bytes == 0 || *fft_bytes == 0) return 0;
    return 1;
}

int audio_pipeline_4ch_ulcnet_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChUlcnetMemReq* out) {
    AudioPipeline4ChConfig cfg_forced;
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    BF_Config bf_cfg;
    int hop, fft, n_freqs;
    size_t srp_bytes, bf_bytes, fft_bytes;
    size_t spectral_count;
    size_t total;

    if (!out || !derive_pipeline_layout(
            cfg, &cfg_forced, &core_req, &srp_cfg, &bf_cfg,
            &hop, &fft, &n_freqs, &srp_bytes, &bf_bytes, &fft_bytes))
        return -1;
    if (core_req.bytes > (uint64_t)SIZE_MAX) return -1;

    total = ck_align16_size(sizeof(AudioPipeline4ChUlcnet));      /* self */
    total = ck_add_size(total, ck_align16_size((size_t)core_req.bytes));
    total = ck_add_size(total, ck_align16_size(srp_bytes));
    total = ck_add_size(total, ck_align16_size(bf_bytes));

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    total = ck_field_size(
        total, (size_t)n_freqs, sizeof(Complex));       /* bf_spectrum */
    total = ck_field_size(
        total, spectral_count, sizeof(Complex));        /* bf_weights */
    total = ck_add_size(total, ck_align16_size(fft_bytes));  /* fft */
    if (framing_bytes(0) == 0 || framing_bytes(1) == 0) return -1;
    total = ck_add_size(total, ck_align16_size(framing_bytes(0)));  /* far_analysis */
    total = ck_add_size(total, ck_align16_size(framing_bytes(1)));  /* synthesis */

    if (MEM_SIZE_INVALID(total)) return -1;

    memset(out, 0, sizeof(*out));
    out->descriptor_version = AUDIO_PIPELINE_4CH_ULCNET_DESCRIPTOR_VERSION;
    out->layout_version = AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION;
    out->backend_id = core_req.backend_id;
    out->build_flags_hash =
        audio_pipeline_4ch_ulcnet_build_flags_hash(core_req.build_flags_hash);
    out->alignment = 16u;
    out->bytes = (uint64_t)total;
    return 0;
}

/* Caller-pool init and heap convenience construction. */

AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChUlcnetMemReq* expected) {
    AudioPipeline4ChConfig cfg_forced;
    AudioPipeline4ChUlcnetMemReq current;
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    BF_Config bf_cfg;
    int hop, fft, n_freqs;
    size_t srp_bytes, bf_bytes, fft_bytes;
    float geom_x[FOUR_AEC_NR_RES_CHANNELS];
    float geom_y[FOUR_AEC_NR_RES_CHANNELS];
    ArrayGeometry geom;
    AudioPipeline4ChUlcnet* p;
    PoolCursor cursor;
    void* core_region;
    void* srp_region;
    void* bf_region;
    void* fft_region;
    void* region;
    AcStftConfig sc;
    size_t spectral_count;

    if (!mem || !cfg) return NULL;
    if (audio_pipeline_4ch_ulcnet_get_mem_requirements(cfg, &current) != 0)
        return NULL;

    /* 8-point descriptor gate (see the header). */
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
            cfg, &cfg_forced, &core_req, &srp_cfg, &bf_cfg,
            &hop, &fft, &n_freqs, &srp_bytes, &bf_bytes, &fft_bytes))
        return NULL;
    if (core_req.bytes > (uint64_t)SIZE_MAX) return NULL;
    if (!fill_stack_geometry(&cfg_forced, geom_x, geom_y, &geom)) return NULL;

    memset(mem, 0, (size_t)current.bytes);
    p = (AudioPipeline4ChUlcnet*)mem;
    p->cfg = cfg_forced;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(*p));
    cursor.remaining = (size_t)current.bytes - ALIGN16(sizeof(*p));

    core_region = pool_carve(&cursor, 1, (size_t)core_req.bytes);
    if (!core_region) return NULL;
    p->core = four_aec_nr_res_init_ex(
        core_region, (size_t)core_req.bytes, &cfg_forced.core, &core_req);
    if (!p->core) return NULL;

    srp_region = pool_carve(&cursor, 1, srp_bytes);
    if (!srp_region) return NULL;
    p->srp = srp_init(srp_region, srp_bytes, &srp_cfg, &geom);
    if (!p->srp) return NULL;
    p->bf_steering = p->srp->a_array;
    if (!p->bf_steering) return NULL;

    bf_region = pool_carve(&cursor, 1, bf_bytes);
    if (!bf_region) return NULL;
    p->bf = bf_init(
        bf_region, bf_bytes, FOUR_AEC_NR_RES_CHANNELS, n_freqs,
        cfg_forced.num_angles, p->bf_steering, &bf_cfg);
    if (!p->bf) return NULL;

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    p->bf_spectrum =
        (Complex*)pool_carve(&cursor, (size_t)n_freqs, sizeof(Complex));
    p->bf_weights =
        (Complex*)pool_carve(&cursor, spectral_count, sizeof(Complex));
    if (!p->bf_spectrum || !p->bf_weights) return NULL;

    fft_region = pool_carve(&cursor, 1, fft_bytes);
    if (!fft_region) return NULL;
    p->fft = fft_init(fft_region, fft_bytes, fft);
    if (!p->fft || fft_get_n_freqs(p->fft) != n_freqs) return NULL;

    p->hop_size = hop;
    p->fft_size = fft;
    p->n_freqs = n_freqs;

    /* The chain uses the carved handle; the inits re-check its bin count against the
     * compiled grid. */
    ac_window_build(p->ulcnet_window, ULCNET_N_FFT);
    ulcnet_framing_config(&sc, p->fft, p->ulcnet_window, 0);
    region = pool_carve(&cursor, 1, framing_bytes(0));
    if (!region) return NULL;
    p->far_analysis = ac_stft_init(region, framing_bytes(0), &sc);
    region = pool_carve(&cursor, 1, framing_bytes(1));
    if (!region) return NULL;
    p->synthesis = ac_istft_init(region, framing_bytes(1), &sc);
    if (!p->far_analysis || !p->synthesis) return NULL;

    memset(&p->model, 0, sizeof(p->model));   /* fail-open until set_model */
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->frame_index = 0;
    p->reprime_frames = 0;
    p->owned_heap = NULL;
    p->destroyed = 0;

    /* The sizing walk and the carves must consume exactly current.bytes. */
    if (cursor.remaining != 0) return NULL;
    return p;
}

AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_init(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg) {
    return audio_pipeline_4ch_ulcnet_init_ex(mem, bytes, cfg, NULL);
}

AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_create(
    const AudioPipeline4ChConfig* cfg) {
    AudioPipeline4ChUlcnetMemReq req;
    AudioPipeline4ChUlcnet* p;
    void* pool = NULL;
    if (!cfg ||
        audio_pipeline_4ch_ulcnet_get_mem_requirements(cfg, &req) != 0 ||
        req.bytes > (uint64_t)SIZE_MAX)
        return NULL;
    if (posix_memalign(
            &pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool)
        return NULL;
    p = audio_pipeline_4ch_ulcnet_init(pool, (size_t)req.bytes, cfg);
    if (!p) {
        free(pool);
        return NULL;
    }
    p->owned_heap = pool;
    return p;
}

int audio_pipeline_4ch_ulcnet_set_model(
    AudioPipeline4ChUlcnet* p,
    const UlcnetModel* model) {
    if (!p || p->destroyed) return -1;
    if (model) {
        /* A refused model leaves the installed one in place. A model that infers must
         * publish a descriptor (host-side rings are carved from it); an identity model may
         * leave it NULL. */
        if (model->infer && !model->io_descriptor) return -1;
        if (model->io_descriptor &&
            ulcnet_model_io_descriptor_validate(model->io_descriptor) != 0)
            return -1;
        p->model = *model;
    } else {
        memset(&p->model, 0, sizeof(p->model));
    }
    return 0;
}

/* Per-hop processing. */

int audio_pipeline_4ch_ulcnet_process_front(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    AudioPipeline4ChUlcnetFront* front) {
    FourAecFront pre;
    int status;
    int m;

    if (!p || p->destroyed || !microphones_interleaved || !far_reference ||
        !front)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->phase != 0) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;

    status = four_aec_front_process(
        p->core, microphones_interleaved, far_reference, &pre);
    if (status != FOUR_AEC_NR_RES_OK) return status;
    if (pre.n_channels != FOUR_AEC_NR_RES_CHANNELS ||
        pre.hop_size != p->hop_size ||
        pre.n_freqs != p->n_freqs ||
        !pre.aligned_ref) {
        audio_pipeline_4ch_ulcnet_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    /* MATCHED exposes generation changes directly; FIXED has none, so the first
     * raw-to-aligned ring transition is detected from the solid/frame history. States
     * whose time basis crosses the boundary are flushed before this hop's inference. */
    if (pre.delay.changed ||
        (p->frame_index != 0 && !p->last_delay.solid && pre.delay.solid)) {
        /* Flush the runtime's far attention ring and logit history. The C framing states
         * keep running; the straddling frame is covered by the identity reprime armed here.
         * The counter is armed whether or not a runtime is attached. */
        if (p->model.reset) p->model.reset(p->model.user);
        p->reprime_frames = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
    }
    p->last_delay = pre.delay;

    /* Far branch: the same hop's raw far through the one-frame-per-hop framing. The
     * error branch (BF spectrum) is already the model's analysis frame, so both
     * branches carry input hop t at pipeline hop t. */
    (void)ac_stft_push_planar(
        p->far_analysis, far_reference, p->far_re, p->far_im);

    /* One inference per frame, except during the identity reprime. */
    p->nn_wanted = 0;
    p->nn_accepted = 0;
    if (p->reprime_frames > 0) p->reprime_frames -= 1;   /* identity; model idle */
    else p->nn_wanted = 1;

    front->n_freqs = pre.n_freqs;
    for (m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m)
        front->lane_spectra[m] = pre.linear_spectra[m];
    p->phase = 3;
    return FOUR_AEC_NR_RES_OK;
}

int audio_pipeline_4ch_ulcnet_process_beam(
    AudioPipeline4ChUlcnet* p,
    const AudioPipeline4ChUlcnetFront* front,
    int vad_raw,
    int vad_out,
    const int* mask) {
    int k, m;

    if (!p || p->destroyed || !front || !is_bool(vad_raw) ||
        !is_bool(vad_out) || front->n_freqs != p->n_freqs)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    for (m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m)
        if (!front->lane_spectra[m]) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->phase != 3) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;

    doa_step(p->srp, front->lane_spectra, mask, vad_raw, vad_out);
    bf_process_with_weights(
        p->bf, front->lane_spectra, doa_get_smooth(p->srp),
        vad_out ? 0 : 1, mask, p->bf_spectrum, p->bf_weights);

    for (k = 0; k < ULCNET_BINS; ++k) {
        p->err_re[k] = p->bf_spectrum[k].r;
        p->err_im[k] = p->bf_spectrum[k].i;
    }
    p->phase = 1;
    return FOUR_AEC_NR_RES_OK;
}

int audio_pipeline_4ch_ulcnet_process_begin(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_external) {
    AudioPipeline4ChUlcnetFront front;
    int status;
    if (!is_bool(vad_external)) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    status = audio_pipeline_4ch_ulcnet_process_front(
        p, microphones_interleaved, far_reference, &front);
    if (status != FOUR_AEC_NR_RES_OK) return status;
    return audio_pipeline_4ch_ulcnet_process_beam(
        p, &front, vad_external, vad_external, NULL);
}

int audio_pipeline_4ch_ulcnet_nn_begin(AudioPipeline4ChUlcnet* p,
                                       UlcnetNnJob* job) {
    if (!p || p->destroyed || !job) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->phase != 1) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    if (!p->nn_wanted) return 0;
    p->nn_wanted = 0;
    /* Full-write contract: NaN staging lets nn_end's finite guard catch an unwritten bin. */
    skn_fill_f32(p->enh_re, ULCNET_BINS, NAN);
    skn_fill_f32(p->enh_im, ULCNET_BINS, NAN);
    job->err_re = p->err_re; job->err_im = p->err_im;
    job->far_re = p->far_re; job->far_im = p->far_im;
    job->out_re = p->enh_re; job->out_im = p->enh_im;
    p->phase = 2;
    return 1;
}

int audio_pipeline_4ch_ulcnet_nn_end(AudioPipeline4ChUlcnet* p, int model_rc) {
    if (!p || p->destroyed) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->phase != 2) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    /* Fail-open: a failed run or any non-finite bin leaves the identity. */
    p->nn_accepted = model_rc == 0 && ulcnet_frame_is_finite(p->enh_re, p->enh_im);
    p->phase = 1;
    return FOUR_AEC_NR_RES_OK;
}

int audio_pipeline_4ch_ulcnet_process_end(AudioPipeline4ChUlcnet* p, float* out) {
    const float* spec_re;
    const float* spec_im;
    int hop, written;

    if (!p || p->destroyed || !out) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->phase != 1 && p->phase != 2) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    hop = p->hop_size;
    /* A frame left open or never fetched is the identity. */
    if (p->phase == 2) p->nn_accepted = 0;
    spec_re = p->nn_accepted ? p->enh_re : p->err_re;
    spec_im = p->nn_accepted ? p->enh_im : p->err_im;
    p->phase = 0;
    written = ac_istft_push_planar(p->synthesis, spec_re, spec_im, out);
    if (written == 0) {
        /* hop#0: the synthesis emits nothing; the caller-facing output is zeros. */
        memset(out, 0, (size_t)hop * sizeof(float));
    } else if (written != hop) {
        /* Cannot happen (the synthesis emits 0 or one full hop); treated as a DSP error. */
        audio_pipeline_4ch_ulcnet_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    p->frame_index += 1;
    return FOUR_AEC_NR_RES_OK;
}

int audio_pipeline_4ch_ulcnet_process_with_activity(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_external,
    float* out) {
    UlcnetNnJob job;
    int status;
    if (!p || p->destroyed || !out) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    status = audio_pipeline_4ch_ulcnet_process_begin(
        p, microphones_interleaved, far_reference, vad_external);
    if (status != FOUR_AEC_NR_RES_OK) return status;
    /* No runtime attached: the frame stays unfetched and is the identity. */
    if (p->model.infer && audio_pipeline_4ch_ulcnet_nn_begin(p, &job) == 1)
        (void)audio_pipeline_4ch_ulcnet_nn_end(
            p, p->model.infer(p->model.user, job.err_re, job.err_im,
                              job.far_re, job.far_im, job.out_re, job.out_im));
    return audio_pipeline_4ch_ulcnet_process_end(p, out);
}

/* Reset and teardown. */

void audio_pipeline_4ch_ulcnet_reset(AudioPipeline4ChUlcnet* p) {
    if (!p || p->destroyed) return;
    four_aec_nr_res_reset(p->core);
    srp_reset(p->srp);
    bf_reset(p->bf);
    /* The objects keep the shared handle and window; reset clears their history. */
    ac_stft_reset(p->far_analysis);
    ac_istft_reset(p->synthesis);
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->frame_index = 0;
    p->phase = 0;
    p->nn_wanted = 0;
    p->nn_accepted = 0;
    /* The analysis history is zeroed above: drop any pending reprime. */
    p->reprime_frames = 0;
    /* A pipeline reset also resets the runtime. */
    if (p->model.reset) p->model.reset(p->model.user);
}

void audio_pipeline_4ch_ulcnet_destroy(AudioPipeline4ChUlcnet* p) {
    void* owned_heap;
    if (!p || p->destroyed) return;
    /* BF borrows the steering table, so destroy it first. The sub-destroys are no-ops
     * for pool-carved objects; only this wrapper's own owned_heap (create()) is freed. */
    bf_destroy(p->bf);
    srp_destroy(p->srp);
    four_aec_nr_res_destroy(p->core);
    fft_destroy(p->fft);
    owned_heap = p->owned_heap;
    p->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

int audio_pipeline_4ch_ulcnet_hop_size(const AudioPipeline4ChUlcnet* p) {
    return (p && !p->destroyed) ? p->hop_size : -1;
}

int audio_pipeline_4ch_ulcnet_fft_size(const AudioPipeline4ChUlcnet* p) {
    return (p && !p->destroyed) ? p->fft_size : -1;
}

int audio_pipeline_4ch_ulcnet_n_freqs(const AudioPipeline4ChUlcnet* p) {
    return (p && !p->destroyed) ? p->n_freqs : -1;
}

int audio_pipeline_4ch_ulcnet_sample_rate(const AudioPipeline4ChUlcnet* p) {
    return (p && !p->destroyed)
        ? four_aec_nr_res_sample_rate(p->core) : -1;
}

int audio_pipeline_4ch_ulcnet_beam_state(
    const AudioPipeline4ChUlcnet* p,
    AudioPipeline4ChUlcnetBeamState* out) {
    if (!p || p->destroyed || !out) return -1;
    out->doa_raw_rad = doa_get_raw(p->srp);
    out->doa_smooth_rad = doa_get_smooth(p->srp);
    out->doa_used_rad = bf_get_doa_used(p->bf);
    out->bf_adaptive = bf_get_adaptive(p->bf);
    return 0;
}

void audio_pipeline_4ch_ulcnet_last_core_timing(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResLastTiming* out) {
    if (!out) return;
    if (!p || p->destroyed) {
        memset(out, 0, sizeof(*out));
        return;
    }
    four_aec_nr_res_get_last_timing(p->core, out);
}

int audio_pipeline_4ch_ulcnet_last_delay(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResDelayState* out) {
    if (!p || p->destroyed || !out) return -1;
    *out = p->last_delay;
    return 0;
}
