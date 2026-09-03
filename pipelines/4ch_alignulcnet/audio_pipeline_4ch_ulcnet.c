/**
 * audio_pipeline_4ch_ulcnet.c — four-channel spatial pipeline with the
 * Align-ULCNet neural post-filter as its post stage.
 *
 * Per-hop flow (see the header's timing contract):
 *
 *   four_aec_nr_res_process_pre -> SRP-PHAT DOA -> GSC effective weights
 *     -> gsc_spectrum, taken DIRECTLY as the model's error frame: it is
 *        already the sqrt-Hann, 50%-overlap, one-frame-per-hop analysis
 *        frame of the current hop that the Align-ULCNet chain would
 *        compute, so there is no reconstruction and no re-analysis between
 *     -> far branch: the same hop's far source (the shared AEC seam's
 *        pre.aligned_ref) through ulcnet_analysis_push_frame -- the same
 *        one-frame-per-hop framing -- so both branches carry the SAME
 *        input hop with no delay buffer
 *     -> UlcnetModel callback, exactly once per hop from hop #0 (skipped
 *        only during the post-boundary identity reprime; fail-open identity
 *        when unset/error/non-finite)
 *     -> UlcnetSynthesis -> out
 *
 * The core's pending pre frame is released with four_aec_nr_res_abandon_pre;
 * no process_post() variant ever runs (the ULCNet chain replaces it).
 *
 * The file follows audio_pipeline_4ch.c's order: instance, config
 * validation, default config, pool-first construction, heap convenience,
 * processing, reset/destroy, and accessors. The spatial-config derivation
 * helpers are file-private copies of audio_pipeline_4ch.c's own (that file
 * keeps them static, per the local one-copy-per-file precedent).
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_pipeline_4ch_ulcnet.h"
#include "gsc.h"
#include "srp.h"
#include "mem_align.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* This wrapper's one supported grid IS whichever grid the ULCNet layer was
 * built for -- a build parameter of ulcnet_model_io.h, taken from there
 * rather than restated. The core is then forced onto the same grid.
 * Asserting hop == fft/2 or bins == fft/2+1 here would assert nothing:
 * ULCNET_HOP and ULCNET_BINS ARE those expressions. The FFT size is what
 * can actually differ, so that is what is pinned. */
_Static_assert(ULCNET_N_FFT == 512 || ULCNET_N_FFT == 1024,
               "ULCNet wrapper supports FFT 512 (16 kHz) or 1024 (48 kHz)");

/* ============================================================================
 * Instance
 * ========================================================================== */

struct AudioPipeline4ChUlcnet {
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

    /* The one carved FFT handle serves the ULCNet chain (far analysis and
     * synthesis; same compiled size, strictly sequential use within a hop,
     * per ulcnet_process.h's sharing contract). */
    FftHandle* fft;

    /* Align-ULCNet chain state + per-hop frame scratch (fixed-size structs,
     * part of `self` in the carve). The chain structs embed their own
     * per-call FFT scratch and point at the shared window table below.
     * There is no error-branch analysis: the GSC spectrum IS the error
     * frame, de-interleaved into err_re/err_im for the model callback. */
    UlcnetAnalysis far_analysis;
    UlcnetSynthesis synthesis;
    float ulcnet_window[ULCNET_N_FFT];  /* shared sqrt-Hann table for both
                                         * chain structs (self-owned) */
    float err_re[ULCNET_BINS];
    float err_im[ULCNET_BINS];
    float far_re[ULCNET_BINS];
    float far_im[ULCNET_BINS];
    float enh_re[ULCNET_BINS];
    float enh_im[ULCNET_BINS];

    UlcnetModel model;    /* model.infer == NULL => fail-open identity */
    FourAecNrResDelayState last_delay;
    uint64_t frame_index;

    /* Frames still straddling the last alignment boundary: armed to
     * AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES at the boundary hop,
     * decremented once per hop (one frame per hop). While nonzero the frame
     * takes the identity path and the model is not stepped. Re-armed, never
     * accumulated, by a boundary that lands inside a reprime. */
    int reprime_frames;

    /* Non-NULL only on the create() heap path (same convention as
     * audio_pipeline_4ch.c's owned_heap). */
    void* owned_heap;
    int destroyed;
};

/* ============================================================================
 * Config validation (file-private copy of audio_pipeline_4ch.c's checks,
 * narrowed to this wrapper's one supported grid)
 * ========================================================================== */

static int is_bool(int value) {
    return value == 0 || value == 1;
}

static int validate_config(const AudioPipeline4ChConfig* cfg) {
    float nyquist;
    if (!cfg) return 0;
    /* Accept only the grid compiled into this ULCNet build. */
    if (cfg->core.sample_rate != ULCNET_SR) return 0;
    if (cfg->core.fft_size != 0 &&
        cfg->core.fft_size != ULCNET_N_FFT) return 0;
    /* Post-only fields are REJECTED, not ignored. AudioPipeline4ChConfig is
     * shared with the standard 4-channel wrapper, so this application's
     * accepted set is deliberately narrower than that struct's: with
     * enable_post = 0 there is no NR instance, no suppressor, no comfort
     * noise and no post iFFT, and a value set in any of these fields could
     * not have done anything. Silently accepting them is what would make one
     * struct mean two things; refusing says so at init. */
    if (cfg->core.enable_post != 0) return 0;
    if (cfg->core.enable_nr != 0) return 0;
    if (cfg->core.enable_cng != 0) return 0;
    if (cfg->core.legacy_amin != 0) return 0;
    /* MmseLsaNrMode has no "disabled" value and no denoiser is created here,
     * so the default is a required sentinel rather than a strength choice. */
    if (cfg->core.nr_mode != MMSE_LSA_NR_BALANCED) return 0;
    /* The built-in energy VAD is unreachable too: this wrapper exposes only
     * _process_with_activity(), which takes the caller's VAD. Compared against
     * a freshly built default rather than transcribed literals, so a change to
     * the defaults cannot leave this rejecting the default config. Exact float
     * comparison is right here: these values are copied, never computed. */
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
    if (!is_bool(cfg->gsc_enable) ||
        !isfinite(cfg->gsc_lambda) || cfg->gsc_lambda <= 0.0f ||
        cfg->gsc_lambda > 1.0f ||
        !isfinite(cfg->gsc_mu) || cfg->gsc_mu < 0.0f ||
        !is_bool(cfg->gsc_fixed_mode) ||
        !isfinite(cfg->gsc_fixed_doa_rad) ||
        !is_bool(cfg->gsc_fixed_align_notebook) ||
        cfg->gsc_adapt_interval <= 0) return 0;
    /* auto_vad_* fields are unused here (external VAD only) but validated
     * for uniformity with the standard wrapper's config surface. */
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
 * Public config
 * ========================================================================== */

AudioPipeline4ChConfig audio_pipeline_4ch_ulcnet_default_config(void) {
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(ULCNET_SR);
    cfg.core.fft_size = ULCNET_N_FFT;
    /* Align-ULCNet REPLACES the post-beam RES/NR/CNG stage, so the pre-only
     * profile is what this function returns rather than something the wrapper
     * quietly rewrites afterwards. Every post-only core field must keep the
     * value set here; validate_config() rejects any other, so a caller who
     * believes it configured NR or comfort noise finds out at init instead of
     * on a board. */
    cfg.core.enable_post = 0;
    cfg.core.enable_nr = 0;
    cfg.core.enable_cng = 0;
    /* core.delay_backward_quarantine_enabled stays at the core default
     * (OFF). The guard holds backward candidates only, for a bounded window
     * after which it accepts, and judges cancellation on the estimator's
     * proxy lane -- so a mis-lock is DELAYED by the window, not cured.
     * Enabling it here is therefore a policy decision, and it waits on a
     * real-audio spot check with the deployed checkpoint. */
    return cfg;
}

/* ============================================================================
 * Pool-first construction: file-private PoolCursor/pool_carve(), same
 * bump-allocator shape kept file-private by audio_pipeline_4ch.c per that
 * file's own precedent.
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

/* Folds in the core's own build_flags_hash exactly like
 * audio_pipeline_4ch.c's hash does, so a change to the core's CARVE TOKEN
 * propagates here. A core LAYOUT VERSION bump does not -- the core's hash
 * never sees it -- so invalidating a persisted composite descriptor still
 * requires bumping AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION by hand.
 * NO_STDIO-safe (no snprintf). */
static uint32_t audio_pipeline_4ch_ulcnet_build_flags_hash(
    uint32_t core_build_flags_hash) {
    uint32_t hash = 2166136261u;
    /* v2: `self` grew the one-hop far-compensation buffer + far_input_mode
     * (self(...) names the state embedded in the control block -- part of
     * the carve even though it is not separately carved).
     * v3: `self` grew the shared window table (ulcnet_window; the chain
     * structs dropped their window copies and embed FFT scratch instead)
     * and the carved `fft` handle is now shared by the beam WOLA AND the
     * ULCNet chain.
     * v5: the self-resident UlcnetModel copy grew io_descriptor (the
     * published model-I/O contract), so the control block is bigger even
     * though the carve ORDER is unchanged. v6 removes the obsolete runtime
     * far-mode field and fixes production to aligned far. The existing
     * last_delay/frame_index fields also identify FIXED's first transition
     * from ring-fill raw far to aligned far. v7 adds the identity-reprime
     * counter. v17: the GSC spectrum feeds the model directly -- the beam
     * WOLA (ifft, ola, synth_win, beam_hop), the error-branch analysis, the
     * two-frame staging and the one-hop far buffer are gone, and `self`
     * carries one-frame err/far staging instead. Bump
     * AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION together with this string,
     * always both, forever. */
    hash = fnv1a_str(
        "|carve:self(corecfg-delay-v2,ulcnet-direct,model(io_descriptor),"
        "reprime,ulcnet_window,frame),"
        "core,srp,gsc,gsc_spectrum,gsc_weights,fft",
        hash);
    /* The grid is a build parameter now, so it cannot be spelled into the
     * token as a literal: stringify the macros instead, or two builds on
     * different grids would share one descriptor hash. */
    hash = fnv1a_str("|align|ulcnet" ULCNET_GRID_TOKEN "|sharedfft", hash);
    hash ^= core_build_flags_hash;
    hash *= 16777619u;
    return hash;
}

/* ============================================================================
 * Stack-only array geometry (no malloc) — verbatim copies of
 * audio_pipeline_4ch.c's own fill helpers (see that file's rationale).
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
 * Config -> SRP/GSC module configs (file-private copy of
 * audio_pipeline_4ch.c's derive_spatial_configs — see that file for the
 * lambda-retime/effective-cadence rationale)
 * ========================================================================== */

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

    gsc_effective_interval = gsc_effective_adapt_interval(
        cfg->gsc_fixed_mode, cfg->gsc_fixed_align_notebook,
        cfg->gsc_adapt_interval);

    memset(gsc_cfg, 0, sizeof(*gsc_cfg));
    gsc_cfg->enable = cfg->gsc_enable;
    gsc_cfg->lambda = mmse_lsa_retime_alpha(
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

static int derive_pipeline_layout(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChConfig* cfg_forced,
    FourAecNrResMemReq* core_req,
    SRP_Config* srp_cfg, GSC_Config* gsc_cfg,
    int* hop, int* fft, int* n_freqs,
    size_t* srp_bytes, size_t* gsc_bytes, size_t* fft_bytes) {
    FourAecNrResMemBreakdown breakdown;
    if (!cfg || !cfg_forced || !core_req || !srp_cfg || !gsc_cfg || !hop ||
        !fft || !n_freqs || !srp_bytes || !gsc_bytes || !fft_bytes) return 0;
    if (!validate_config(cfg)) return 0;
    /* Force the compiled grid before ANY core sizing so the descriptor and
     * the carve always describe the same layout. */
    *cfg_forced = *cfg;
    cfg_forced->core.fft_size = ULCNET_N_FFT;
    /* enable_post is already 0: validate_config() refuses anything else, so
     * the caller has declared the pre-only profile rather than had it
     * rewritten underneath. Nothing to force here. */
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
            cfg_forced, *hop, *fft, *n_freqs, srp_cfg, gsc_cfg))
        return 0;
    *srp_bytes = srp_get_mem_size(srp_cfg);
    *gsc_bytes = gsc_get_mem_size(FOUR_AEC_NR_RES_CHANNELS, *n_freqs);
    *fft_bytes = fft_get_mem_size(*fft);
    if (*srp_bytes == 0 || *gsc_bytes == 0 || *fft_bytes == 0) return 0;
    return 1;
}

int audio_pipeline_4ch_ulcnet_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChUlcnetMemReq* out) {
    AudioPipeline4ChConfig cfg_forced;
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    int hop, fft, n_freqs;
    size_t srp_bytes, gsc_bytes, fft_bytes;
    size_t spectral_count;
    size_t total;

    if (!out || !derive_pipeline_layout(
            cfg, &cfg_forced, &core_req, &srp_cfg, &gsc_cfg,
            &hop, &fft, &n_freqs, &srp_bytes, &gsc_bytes, &fft_bytes))
        return -1;
    if (core_req.bytes > (uint64_t)SIZE_MAX) return -1;

    total = ck_align16_size(sizeof(AudioPipeline4ChUlcnet));      /* self */
    total = ck_add_size(total, ck_align16_size((size_t)core_req.bytes));
    total = ck_add_size(total, ck_align16_size(srp_bytes));
    total = ck_add_size(total, ck_align16_size(gsc_bytes));

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    total = ck_field_size(
        total, (size_t)n_freqs, sizeof(Complex));       /* gsc_spectrum */
    total = ck_field_size(
        total, spectral_count, sizeof(Complex));        /* gsc_weights */
    total = ck_add_size(total, ck_align16_size(fft_bytes));  /* fft */

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

/* ============================================================================
 * Caller-pool init and heap convenience construction
 * ========================================================================== */

AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChUlcnetMemReq* expected) {
    AudioPipeline4ChConfig cfg_forced;
    AudioPipeline4ChUlcnetMemReq current;
    FourAecNrResMemReq core_req;
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    int hop, fft, n_freqs;
    size_t srp_bytes, gsc_bytes, fft_bytes;
    float geom_x[FOUR_AEC_NR_RES_CHANNELS];
    float geom_y[FOUR_AEC_NR_RES_CHANNELS];
    ArrayGeometry geom;
    AudioPipeline4ChUlcnet* p;
    PoolCursor cursor;
    void* core_region;
    void* srp_region;
    void* gsc_region;
    void* fft_region;
    size_t spectral_count;

    if (!mem || !cfg) return NULL;
    if (audio_pipeline_4ch_ulcnet_get_mem_requirements(cfg, &current) != 0)
        return NULL;

    /* 8-point descriptor gate (contract documented in the header). */
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
            cfg, &cfg_forced, &core_req, &srp_cfg, &gsc_cfg,
            &hop, &fft, &n_freqs, &srp_bytes, &gsc_bytes, &fft_bytes))
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
    p->gsc_steering = p->srp->a_array;
    if (!p->gsc_steering) return NULL;

    gsc_region = pool_carve(&cursor, 1, gsc_bytes);
    if (!gsc_region) return NULL;
    p->gsc = gsc_init(
        gsc_region, gsc_bytes, FOUR_AEC_NR_RES_CHANNELS, n_freqs,
        cfg_forced.num_angles, p->gsc_steering, &gsc_cfg);
    if (!p->gsc) return NULL;

    spectral_count = (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs;
    p->gsc_spectrum =
        (Complex*)pool_carve(&cursor, (size_t)n_freqs, sizeof(Complex));
    p->gsc_weights =
        (Complex*)pool_carve(&cursor, spectral_count, sizeof(Complex));
    if (!p->gsc_spectrum || !p->gsc_weights) return NULL;

    fft_region = pool_carve(&cursor, 1, fft_bytes);
    if (!fft_region) return NULL;
    p->fft = fft_init(fft_region, fft_bytes, fft);
    if (!p->fft || fft_get_n_freqs(p->fft) != n_freqs) return NULL;

    p->hop_size = hop;
    p->fft_size = fft;
    p->n_freqs = n_freqs;

    /* The chain uses the handle carved above (same compiled grid; strictly
     * sequential use per hop). Reject-first: the inits re-check the
     * handle's bin count against the compiled ULCNet grid. */
    ulcnet_make_window(p->ulcnet_window);
    if (ulcnet_analysis_init(&p->far_analysis, p->fft, p->ulcnet_window) != 0 ||
        ulcnet_synthesis_init(&p->synthesis, p->fft, p->ulcnet_window) != 0)
        return NULL;

    memset(&p->model, 0, sizeof(p->model));   /* fail-open until set_model */
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->frame_index = 0;
    p->reprime_frames = 0;
    p->owned_heap = NULL;
    p->destroyed = 0;

    /* Lockstep proof: the sizing walk and the carves above must consume
     * exactly current.bytes. */
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
        /* Reject-first: the previously installed model stays in place. */
        /* A model that actually infers MUST publish a descriptor. Its delay depth,
         * attention geometry and history shapes are what the host-side rings are
         * carved from, and nothing downstream can detect a mismatch: the finite
         * guard catches an UNWRITTEN output, never a WRONG-SHAPED one, so a graph
         * whose D differs from the descriptor reads and writes past the pool
         * silently. An identity model (no infer callback) has no shapes to agree
         * about and may leave it NULL. */
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

/* ============================================================================
 * Per-hop processing
 * ========================================================================== */

/* NaN/Inf guard: a model frame with ANY non-finite value must never reach
 * the WOLA (the synthesis accumulator would poison every later hop). */
static int ulcnet_frame_is_finite(const float* re, const float* im) {
    int k;
    for (k = 0; k < ULCNET_BINS; ++k) {
        if (!isfinite(re[k]) || !isfinite(im[k])) return 0;
    }
    return 1;
}

int audio_pipeline_4ch_ulcnet_process_with_activity(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_external,
    float* out) {
    FourAecNrResPreFrame pre;
    const float* spec_re;
    const float* spec_im;
    int status;
    int hop;
    int written;
    int k;

    if (!p || p->destroyed || !microphones_interleaved || !far_reference ||
        !out || !is_bool(vad_external))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    hop = p->hop_size;

    status = four_aec_nr_res_process_pre(
        p->core, microphones_interleaved, far_reference, &pre);
    if (status != FOUR_AEC_NR_RES_OK) return status;
    if (pre.n_channels != FOUR_AEC_NR_RES_CHANNELS ||
        pre.hop_size != hop ||
        pre.n_freqs != p->n_freqs ||
        !pre.aligned_ref) {
        audio_pipeline_4ch_ulcnet_reset(p);   /* also releases the frame */
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    doa_step(
        p->srp, pre.linear_spectra, NULL,
        vad_external, vad_external);
    gsc_process_with_weights(
        p->gsc, pre.linear_spectra, doa_get_smooth(p->srp),
        vad_external ? 0 : 1, NULL, p->gsc_spectrum,
        p->gsc_weights);

    /* MATCHED exposes generation changes directly. FIXED has no estimator
     * generation, so detect the first raw-to-aligned ring transition from
     * the existing solid/frame history. In both cases, flush every state
     * whose time basis crosses the boundary before this hop's inference. */
    if (pre.delay.changed ||
        (p->frame_index != 0 && !p->last_delay.solid && pre.delay.solid)) {
        /* Tell the runtime to flush its far attention ring + logit history.
         * The C-side framing states keep running across the boundary -- the
         * core's lane analysis behind the GSC spectrum, and this wrapper's
         * far analysis -- so the frame whose window still covers a
         * pre-switch hop is covered by the identity reprime armed here
         * rather than by a crossfade. The counter is armed whether or not a
         * runtime is attached, so the frame policy does not depend on the
         * model's presence. */
        if (p->model.reset) p->model.reset(p->model.user);
        p->reprime_frames = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
    }

    /* Error branch: the GSC spectrum is already the model's analysis frame
     * -- the core's lanes analyse with the same sqrt-Hann window at the same
     * 50% overlap, one frame per hop from hop #0, and the beamformer only
     * weights bins -- so it is handed over as-is. Far branch: the same hop's
     * far source through the same one-frame-per-hop framing. Both branches
     * therefore carry input hop t at pipeline hop t, with no reconstruction,
     * no re-analysis and no delay buffer in between. */
    for (k = 0; k < ULCNET_BINS; ++k) {
        p->err_re[k] = p->gsc_spectrum[k].r;
        p->err_im[k] = p->gsc_spectrum[k].i;
    }
    (void)ulcnet_analysis_push_frame(
        &p->far_analysis, pre.aligned_ref, p->far_re, p->far_im);

    /* All PreFrame pointers consumed -- release the core's pending frame
     * (no process_post() variant ever runs in this pipeline). */
    status = four_aec_nr_res_abandon_pre(p->core, &pre.token);
    p->last_delay = pre.delay;
    if (status != FOUR_AEC_NR_RES_OK) {
        audio_pipeline_4ch_ulcnet_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    /* One frame, one inference: step the model and apply its output, EXCEPT
     * while the identity reprime armed at the last boundary is still
     * running -- that frame's window still covers the pre-switch hop, and
     * stepping it would rebuild the just-flushed recurrent state from
     * half-stale input. Skipping inference lowers the cost of that frame;
     * it never doubles it. */
    spec_re = p->err_re;
    spec_im = p->err_im;
    if (p->reprime_frames > 0) {
        p->reprime_frames -= 1;   /* identity; model deliberately idle */
    } else if (p->model.infer) {
        /* Enforce ulcnet_process.h's FULL-WRITE CONTRACT: pre-fill the
         * model-output staging with NaN before every infer call, so a
         * partial write (rc == 0 without writing all ULCNET_BINS) leaves
         * non-finite bins behind and is rejected by the finite guard below
         * (fail-open identity) instead of silently applying stale finite
         * values left over from a previous frame. */
        for (k = 0; k < ULCNET_BINS; ++k) {
            p->enh_re[k] = NAN;
            p->enh_im[k] = NAN;
        }
        if (p->model.infer(
                p->model.user,
                p->err_re, p->err_im, p->far_re, p->far_im,
                p->enh_re, p->enh_im) == 0 &&
            ulcnet_frame_is_finite(p->enh_re, p->enh_im)) {
            spec_re = p->enh_re;
            spec_im = p->enh_im;
        }
    }
    written = ulcnet_synthesis_push(&p->synthesis, spec_re, spec_im, out);
    if (written == 0) {
        /* hop#0: frame #0's block lies inside the synthesis's trimmed half
         * window; the caller-facing contract is zeros. */
        memset(out, 0, (size_t)hop * sizeof(float));
    } else if (written != hop) {
        /* Structurally impossible (the synthesis emits 0 or one full hop);
         * treat as an internal DSP error rather than emitting a short hop. */
        audio_pipeline_4ch_ulcnet_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    p->frame_index += 1;
    return FOUR_AEC_NR_RES_OK;
}

/* ============================================================================
 * Reset and teardown
 * ========================================================================== */

void audio_pipeline_4ch_ulcnet_reset(AudioPipeline4ChUlcnet* p) {
    if (!p || p->destroyed) return;
    four_aec_nr_res_reset(p->core);   /* invalidates any pending pre frame */
    srp_reset(p->srp);
    gsc_reset(p->gsc);
    /* Re-init keeps the same shared handle/window (pool/instance resident,
     * untouched by reset); cannot fail for a handle already validated at
     * init time. */
    (void)ulcnet_analysis_init(&p->far_analysis, p->fft, p->ulcnet_window);
    (void)ulcnet_synthesis_init(&p->synthesis, p->fft, p->ulcnet_window);
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->frame_index = 0;
    /* The analysis history is zeroed above, so nothing emitted after this
     * point straddles anything: drop any pending reprime. */
    p->reprime_frames = 0;
    /* Documented contract: a pipeline reset also resets the runtime. */
    if (p->model.reset) p->model.reset(p->model.user);
}

void audio_pipeline_4ch_ulcnet_destroy(AudioPipeline4ChUlcnet* p) {
    void* owned_heap;
    if (!p || p->destroyed) return;
    /* GSC borrows the steering table; destroy it before its owner. All four
     * sub-destroys are no-op frees for pool-carved sub-objects (their own
     * owned_heap fields stay NULL); only this wrapper's OWN owned_heap (set
     * only by create()) is ever actually freed below, exactly once. */
    gsc_destroy(p->gsc);
    srp_destroy(p->srp);
    four_aec_nr_res_destroy(p->core);
    fft_destroy(p->fft);
    owned_heap = p->owned_heap;
    p->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

/* ============================================================================
 * Read-only accessors
 * ========================================================================== */

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

int audio_pipeline_4ch_ulcnet_last_delay(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResDelayState* out) {
    if (!p || p->destroyed || !out) return -1;
    *out = p->last_delay;
    return 0;
}
