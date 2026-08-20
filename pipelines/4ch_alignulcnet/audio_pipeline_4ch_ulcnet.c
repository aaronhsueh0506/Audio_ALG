/**
 * audio_pipeline_4ch_ulcnet.c — four-channel spatial pipeline with the
 * Align-ULCNet neural post-filter as its post stage.
 *
 * Per-hop flow (see the header's timing contract):
 *
 *   four_aec_nr_res_process_pre -> SRP-PHAT DOA -> GSC effective weights
 *     -> gsc_spectrum WOLA reconstruction (one hop behind)
 *     -> UlcnetAnalysis (error branch = beamformed hop,
 *                        far branch  = the PREVIOUS hop's far source, so
 *                        both branches carry the SAME input hop: the beam
 *                        hop is itself one hop behind, and the far source
 *                        -- the shared AEC seam's pre.aligned_ref -- goes through
 *                        a one-hop saved buffer to match it)
 *     -> UlcnetModel callback (stepped every frame except during the
 *        post-boundary identity reprime; fail-open identity when
 *        unset/error/non-finite)
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
#include "simd_kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PI_F ((float)M_PI)

/* This wrapper's one supported grid. ULCNet's constants are compile-time
 * 16 kHz / 512 / 256 / 257; the core must be forced onto the same grid. */
#define ULCNET_PIPELINE_SAMPLE_RATE 16000
#define ULCNET_PIPELINE_FFT 512

_Static_assert(ULCNET_SR == ULCNET_PIPELINE_SAMPLE_RATE,
               "ULCNet compile-time sample rate must match this wrapper");
_Static_assert(ULCNET_N_FFT == ULCNET_PIPELINE_FFT,
               "ULCNet frame must match the forced core FFT size");
_Static_assert(ULCNET_HOP == ULCNET_PIPELINE_FFT / 2,
               "ULCNet hop must match the core hop (fft/2)");
_Static_assert(ULCNET_BINS == ULCNET_PIPELINE_FFT / 2 + 1,
               "ULCNet bin count must match the core bin count");

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

    /* Beamformed-error WOLA reconstruction (one hop behind). The one
     * carved handle is ALSO the ULCNet chain's FFT (same 512 size; the
     * beam IFFT and the chain's three transforms are strictly sequential
     * within a hop, per ulcnet_process.h's sharing contract). */
    FftHandle* fft;
    float* ifft_buffer;   /* fft_size */
    float* ola;           /* fft_size */
    float* synth_window;  /* fft_size */
    float* beam_hop;      /* hop_size; accessor + analysis input */

    /* Align-ULCNet chain state + per-hop frame scratch (fixed-size structs,
     * part of `self` in the carve). The chain structs embed their own
     * per-call FFT scratch and point at the shared window table below. */
    UlcnetAnalysis err_analysis;
    UlcnetAnalysis far_analysis;
    UlcnetSynthesis synthesis;
    float ulcnet_window[ULCNET_N_FFT];  /* shared sqrt-Hann table for all
                                         * three chain structs (self-owned) */
    float frame_err_re[2][ULCNET_BINS];
    float frame_err_im[2][ULCNET_BINS];
    float frame_far_re[2][ULCNET_BINS];
    float frame_far_im[2][ULCNET_BINS];
    float enh_re[ULCNET_BINS];
    float enh_im[ULCNET_BINS];

    /* One-hop far compensation (layout v2): the beam WOLA above closes one
     * hop late, so far_analysis is fed the far source SAVED on the previous
     * call -- without this the model would see error[t-1] paired with
     * far[t] (a fixed 256-sample skew). Cleared together with the beam OLA
     * on a delay change and on reset. */
    float far_delay[ULCNET_HOP];

    UlcnetModel model;    /* model.infer == NULL => fail-open identity */
    FourAecNrResDelayState last_delay;
    uint64_t frame_index;

    /* Emitted frames still straddling the last alignment boundary: armed to
     * AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES at the boundary hop,
     * decremented once per EMITTED frame (not per hop -- the chain's second
     * hop emits two). While nonzero the frame takes the identity path and
     * the model is not stepped. Re-armed, never accumulated, by a boundary
     * that lands inside a reprime. */
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
    /* ULCNet grid gate: 16 kHz only, core fft 0 (forced to 512) or 512. */
    if (cfg->core.sample_rate != ULCNET_PIPELINE_SAMPLE_RATE) return 0;
    if (cfg->core.fft_size != 0 &&
        cfg->core.fft_size != ULCNET_PIPELINE_FFT) return 0;
    /* Post-only fields are REJECTED, not ignored. AudioPipeline4ChConfig is
     * shared with the standard 4-channel wrapper, so this application's
     * accepted set is deliberately narrower than that struct's: with
     * enable_post = 0 there is no NR instance, no suppressor, no comfort
     * noise and no post iFFT, and a value set in any of these fields could
     * not have done anything. Silently accepting them is what would make one
     * struct mean two things; refusing says so at init. */
    if (cfg->core.enable_post != 0) return 0;
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
            audio_pipeline_4ch_default_config(ULCNET_PIPELINE_SAMPLE_RATE);
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
        audio_pipeline_4ch_default_config(ULCNET_PIPELINE_SAMPLE_RATE);
    cfg.core.fft_size = ULCNET_PIPELINE_FFT;
    /* Align-ULCNet REPLACES the post-beam RES/NR/CNG stage, so the pre-only
     * profile is what this function returns rather than something the wrapper
     * quietly rewrites afterwards. Every post-only core field must keep the
     * value set here; validate_config() rejects any other, so a caller who
     * believes it configured NR or comfort noise finds out at init instead of
     * on a board. */
    cfg.core.enable_post = 0;
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
     * counter. Bump
     * AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION together with this string,
     * always both, forever. */
    hash = fnv1a_str(
        "|carve:self(corecfg-delay-v2,ulcnet,model(io_descriptor),far_delay,"
        "reprime,ulcnet_window),"
        "core,srp,gsc,gsc_spectrum,gsc_weights,fft,ifft,ola,synth_win,"
        "beam_hop",
        hash);
    hash = fnv1a_str("|align16|ulcnet16k512|sharedfft", hash);
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
    /* Force the one supported grid before ANY core sizing so the descriptor
     * and the carve always describe the same (512/256) layout. */
    *cfg_forced = *cfg;
    cfg_forced->core.fft_size = ULCNET_PIPELINE_FFT;
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
    total = ck_field_size(
        total, (size_t)fft, sizeof(float));             /* ifft_buffer */
    total = ck_field_size(
        total, (size_t)fft, sizeof(float));             /* ola */
    total = ck_field_size(
        total, (size_t)fft, sizeof(float));             /* synth_win */
    total = ck_field_size(
        total, (size_t)hop, sizeof(float));             /* beam_hop */

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
    int k;
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

    p->ifft_buffer =
        (float*)pool_carve(&cursor, (size_t)fft, sizeof(float));
    p->ola =
        (float*)pool_carve(&cursor, (size_t)fft, sizeof(float));
    p->synth_window =
        (float*)pool_carve(&cursor, (size_t)fft, sizeof(float));
    p->beam_hop =
        (float*)pool_carve(&cursor, (size_t)hop, sizeof(float));
    if (!p->ifft_buffer || !p->ola || !p->synth_window || !p->beam_hop)
        return NULL;

    p->hop_size = hop;
    p->fft_size = fft;
    p->n_freqs = n_freqs;

    /* Same sqrt-Hann synthesis window formula as 4aec_nr_res.c's own
     * synthesis path -- the reconstruction seam proven by
     * tests/test_4aec_nr_res.c's WOLA-identity test. */
    for (k = 0; k < fft; ++k) {
        p->synth_window[k] = sqrtf(
            0.5f * (1.0f - cosf(
                2.0f * M_PI_F * (float)k / (float)fft)));
    }

    /* The chain reuses the beamforming handle carved above (same 512 grid;
     * strictly sequential use per hop). Reject-first: the inits re-check
     * the handle's bin count against the compiled ULCNet grid. */
    ulcnet_make_window(p->ulcnet_window);
    if (ulcnet_analysis_init(&p->err_analysis, p->fft, p->ulcnet_window) != 0 ||
        ulcnet_analysis_init(&p->far_analysis, p->fft, p->ulcnet_window) != 0 ||
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
    int status;
    int hop;
    int fft;
    int n_err_frames;
    int n_far_frames;
    int written;
    int f;
    int k;

    if (!p || p->destroyed || !microphones_interleaved || !far_reference ||
        !out || !is_bool(vad_external))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    hop = p->hop_size;
    fft = p->fft_size;

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
        /* The core reset its lanes' analysis history before producing this
         * hop's spectra; discard the matching synthesis tail here for the
         * same reason its own mono OLA is cleared (mixing spectra from
         * opposite sides of the realignment would corrupt the seam), and
         * tell the runtime to flush its far attention ring + logit history.
         * The saved far hop straddles the same boundary, so it is cleared
         * with the OLA (a one-hop zero-far transient is
         * accepted alongside the model reset). The C-side ULCNet STFT states
         * keep running, so the frames whose windows still cover this cleared
         * slot are covered by the identity reprime armed here rather than by
         * a crossfade. The counter is armed whether or not a runtime is
         * attached, so the frame policy does not depend on the model's
         * presence. */
        memset(p->ola, 0, (size_t)fft * sizeof(float));
        memset(p->far_delay, 0, sizeof(p->far_delay));
        if (p->model.reset) p->model.reset(p->model.user);
        p->reprime_frames = AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES;
    }

    /* Reconstruct the time-domain beamformed error (one hop behind): same
     * fft_inverse + sqrt-Hann synthesis + 50% OLA recipe proven by
     * tests/test_4aec_nr_res.c's WOLA-identity test. */
    fft_inverse(p->fft, p->gsc_spectrum, p->ifft_buffer);
    sk_wola_accumulate_f32(p->ola, p->ifft_buffer, p->synth_window, fft);
    memcpy(p->beam_hop, p->ola, (size_t)hop * sizeof(float));
    memmove(p->ola, p->ola + hop, (size_t)(fft - hop) * sizeof(float));
    memset(p->ola + (fft - hop), 0, (size_t)hop * sizeof(float));

    /* Push both ULCNet analysis branches (0/2/1 emission, always in
     * lockstep since both states advance once per hop). The beam_hop just
     * reconstructed above is the PREVIOUS hop's beamformed error (the WOLA
     * closes one hop late), so the far branch must be delayed by one hop
     * too: push the far hop SAVED on the previous call, then save this
     * hop's aligned far source. Without this the model would see
     * error[t-1] paired
     * with far[t] -- a fixed 256-sample skew. */
    n_err_frames = ulcnet_analysis_push(
        &p->err_analysis, p->beam_hop, p->frame_err_re, p->frame_err_im);
    n_far_frames = ulcnet_analysis_push(
        &p->far_analysis, p->far_delay, p->frame_far_re, p->frame_far_im);
    memcpy(p->far_delay, pre.aligned_ref, (size_t)hop * sizeof(float));

    /* All PreFrame pointers consumed -- release the core's pending frame
     * (no process_post() variant ever runs in this pipeline). */
    status = four_aec_nr_res_abandon_pre(p->core, &pre.token);
    p->last_delay = pre.delay;
    if (n_err_frames != n_far_frames || status != FOUR_AEC_NR_RES_OK) {
        audio_pipeline_4ch_ulcnet_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }

    /* Per emitted frame: step the model and apply its output, EXCEPT while
     * the identity reprime armed at the last boundary is still running --
     * those frames' windows still cover the pre-switch (here: cleared) slot,
     * and stepping them would rebuild the just-flushed recurrent state from
     * half-stale input. Skipping inference lowers the cost of those frames;
     * it never doubles it. */
    written = 0;
    for (f = 0; f < n_err_frames; ++f) {
        const float* spec_re = p->frame_err_re[f];
        const float* spec_im = p->frame_err_im[f];
        if (p->reprime_frames > 0) {
            p->reprime_frames -= 1;   /* identity; model deliberately idle */
        } else if (p->model.infer) {
            /* Enforce ulcnet_process.h's FULL-WRITE CONTRACT: pre-fill the
             * model-output staging with NaN before every infer call, so a
             * partial write (rc == 0 without writing all ULCNET_BINS)
             * leaves non-finite bins behind and is rejected by the finite
             * guard below (fail-open identity) instead of silently applying
             * stale finite values left over from a previous frame. */
            for (k = 0; k < ULCNET_BINS; ++k) {
                p->enh_re[k] = NAN;
                p->enh_im[k] = NAN;
            }
            if (p->model.infer(
                    p->model.user,
                    p->frame_err_re[f], p->frame_err_im[f],
                    p->frame_far_re[f], p->frame_far_im[f],
                    p->enh_re, p->enh_im) == 0 &&
                ulcnet_frame_is_finite(p->enh_re, p->enh_im)) {
                spec_re = p->enh_re;
                spec_im = p->enh_im;
            }
        }
        written += ulcnet_synthesis_push(
            &p->synthesis, spec_re, spec_im, out + written);
    }
    if (written == 0) {
        /* hop#0 (and any other preamble hop): the chain emitted nothing;
         * the caller-facing contract is zeros. */
        memset(out, 0, (size_t)hop * sizeof(float));
    } else if (written != hop) {
        /* Structurally impossible under the 0/2/1 emission contract; treat
         * as an internal DSP error rather than emitting a short hop. */
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
    memset(p->ola, 0, (size_t)p->fft_size * sizeof(float));
    memset(p->ifft_buffer, 0, (size_t)p->fft_size * sizeof(float));
    memset(p->beam_hop, 0, (size_t)p->hop_size * sizeof(float));
    memset(p->far_delay, 0, sizeof(p->far_delay));
    /* Re-init keeps the same shared handle/window (pool/instance resident,
     * untouched by reset); cannot
     * fail for a handle already validated at init time. */
    (void)ulcnet_analysis_init(&p->err_analysis, p->fft, p->ulcnet_window);
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

const float* audio_pipeline_4ch_ulcnet_last_beamformed_error(
    const AudioPipeline4ChUlcnet* p) {
    return (p && !p->destroyed) ? p->beam_hop : NULL;
}

int audio_pipeline_4ch_ulcnet_last_delay(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResDelayState* out) {
    if (!p || p->destroyed || !out) return -1;
    *out = p->last_delay;
    return 0;
}
