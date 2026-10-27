/** Implementation of mcab.h: one shared DelayAec -> one aligned far -> four linear AEC lanes ->
 * the caller's beamformer weights -> coherent projection of the four lane contexts (mcab_fuse).
 * Every library object is injected; this file owns only the flow state carved from its pool. */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>

#include "mcab.h"
#include "mcab_internal.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"    /* skn_all_finite_f32, skn_all_finite_cf32 */

/* Matched-filter duty cycle: full-rate analysis until the estimate has been solid and unchanged
 * for DUTY_HOLD_S (lib/aec's delay_est_init_s default), then one hop in
 * K = round(delay_est_period_s / hop_s) / DUTY_PERIOD_DIVISOR. The period floor is 1, so 0.0
 * means true full rate. The watchdog leak is lib/aec's 0.1 dB per second, converted to this
 * grid's hop by mcab_duty_leak_db_for_grid(). */
#define DUTY_HOLD_S               0.3f
#define DUTY_PERIOD_DIVISOR       5
#define DUTY_ERLE_LEAK_DB         0.001f   /* per hop at DUTY_LEAK_REF_HOP_S */
#define DUTY_LEAK_REF_HOP_S       0.010f
#define DUTY_ERLE_COLLAPSE_DB     6.0f

#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

static uint32_t mcab_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0) return MCAB_BACKEND_NE10;
    return 0u;
}

/* Off by default so a release build takes no clock reads; AEC_STAGE_TIMING governs the lanes'
 * own figures. */
#ifndef MCAB_STAGE_TIMING
#define MCAB_STAGE_TIMING 0
#endif

#if MCAB_STAGE_TIMING
/* Truncated to 32 bits: consumers subtract in unsigned arithmetic. */
#ifndef MCAB_NOW_US
#include <time.h>
static uint32_t mcab_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull);
}
#else
static uint32_t mcab_now_us(void) { return MCAB_NOW_US(); }
#endif
#else
#define mcab_now_us() 0u
#endif

typedef struct McabLaneSnapshot {
    /* Borrowed views into a lane's AecResContext buffers (lifetime: bind_lane_view()). */
    const Complex* error_spec;
    const Complex* echo_spec;
    const Complex* far_spec;
    const Complex* near_spec;
    const float* r2;
    const float* comfort_noise;
    const float* formed_hop;

    float far_power;
    float dt_indicator;
    float saturation_level;
    int filter_converged;
    int usable_linear;
} McabLaneSnapshot;

struct Mcab {
    McabConfig cfg;
    int sample_rate;
    int fft_size;
    int hop_size;
    int n_freqs;

    /* Borrowed, caller-owned (McabLibs). */
    Aec* lanes[MCAB_CHANNELS];
    DelayAec* delay;

    McabLaneSnapshot snapshots[MCAB_CHANNELS];

    float* mic_lane;
    float* aligned_ref;

    float* delay_ring;
    int delay_ring_size;
    uint64_t delay_samples_seen;
    int accepted_delay;
    /* Backward-jump quarantine: _left = hops before a held earlier estimate is adopted anyway
     * (-1 = disarmed, the only state that may re-arm; 0 = armed and expired), _hops = the window
     * converted once at init. */
    int delay_quarantine_left;
    int delay_quarantine_hops;
    /* Two-step admission of a shared-delay change; cleared by acceptance, expiry and reset. */
    McabDelayAdmission delay_admission;
    long realign_warm_lanes;
    long realign_soft_lanes;
    uint64_t delay_calls;
    McabDelayState last_delay;

    Complex* fused_error;
    Complex* fused_near;
    Complex* residual_work;
    Complex* residual_proj;    /* coherent residual vector, then its power in fused_r2 */
    float* fused_r2;
    float* fused_comfort;

    int front_valid;           /* a front was processed since the last reset */

    void* owned_heap;
    size_t pool_size;
    int destroyed;

    McabLastTiming last_timing;

    /* This hop's evidence that all four lanes consumed one far spectrum (far-FFT counters); 0
     * (also after any bail-out) costs the full per-bin comparison in fuse_contexts(). */
    int far_spec_shared_hop;
    /* This hop routed a mixed warm/soft realign: the lanes' far_spec legitimately diverge, so
     * fuse_contexts() skips the agreement predicate. */
    int far_spec_mixed_hop;

    /* Matched-filter duty cycle (see update_shared_delay()). */
    int duty_active;
    int duty_stable_hops;
    int duty_pos;
    int duty_last_delay;
    float duty_erle_peak;
    int duty_hold_hops;
    int duty_period_hops;
    float duty_erle_leak_db;
    unsigned long long duty_hops_run;
};

typedef struct PoolCursor {
    uint8_t* ptr;
    size_t remaining;
} PoolCursor;

/* Exactly 1.0f at the calibration grid (hop 160 at 16 kHz), which is not built here. */
float mcab_duty_leak_db_for_grid(int hop_size, int sample_rate) {
    float hop_s;
    if (hop_size <= 0 || sample_rate <= 0) return DUTY_ERLE_LEAK_DB;
    hop_s = (float)hop_size / (float)sample_rate;
    return DUTY_ERLE_LEAK_DB * (hop_s / DUTY_LEAK_REF_HOP_S);
}

/* The flow's grid (no padding, 50 % overlap): 16 kHz 256|512, 48 kHz 1024; 0 = the rate's
 * default. */
static int flow_grid(int sample_rate, int requested_fft, int* fft, int* hop, int* n) {
    int f = requested_fft;
    if (f == 0) f = sample_rate == 48000 ? 1024 : sample_rate == 16000 ? 256 : 0;
    if (!((sample_rate == 16000 && (f == 256 || f == 512)) || (sample_rate == 48000 && f == 1024)))
        return 0;
    *fft = f;
    *hop = f / 2;
    *n = f / 2 + 1;
    return 1;
}

/* MATCHED sizes its bank, FIXED pins a bounded sample count (two minutes at the flow's rate),
 * EXTERNAL_ALIGNED takes neither. */
static int delay_request_valid(const McabConfig* cfg) {
    switch (cfg->delay_mode) {
        case AEC_DELAY_MATCHED:
            return cfg->delay_num_filters >= 1 && cfg->delay_num_filters <= DA_NUM_FILTERS &&
                   cfg->fixed_delay_samples == -1;
        case AEC_DELAY_FIXED:
            return cfg->delay_num_filters == DA_NUM_FILTERS && cfg->fixed_delay_samples >= 0 &&
                   (long)cfg->fixed_delay_samples <= 120L * (long)cfg->sample_rate;
        case AEC_DELAY_EXTERNAL_ALIGNED:
            return cfg->delay_num_filters == DA_NUM_FILTERS && cfg->fixed_delay_samples == -1;
        default:
            return 0;
    }
}

/* Returns 1 and fills the out-params, or 0 for a config the flow refuses. The flow aligns the
 * far once for all four lanes, so every lane consumes an already aligned reference. */
static int derive_flow(const McabConfig* cfg, int* fft_size, int* hop_size, int* n_freqs,
                       int* delay_ring_size) {
    int max_delay_samples;
    if (!cfg || !fft_size || !hop_size || !n_freqs || !delay_ring_size) return 0;
    if (cfg->enable_post != 0 && cfg->enable_post != 1) return 0;
    if (!flow_grid(cfg->sample_rate, cfg->fft_size, fft_size, hop_size, n_freqs)) return 0;
    if (!delay_request_valid(cfg)) return 0;
    if (cfg->capture_proxy_channel < 0 || cfg->capture_proxy_channel >= MCAB_CHANNELS) return 0;
    /* Range only, inert outside MATCHED; the window is checked even while the enable is 0. */
    if (cfg->delay_backward_quarantine_enabled != 0 &&
        cfg->delay_backward_quarantine_enabled != 1) return 0;
    if (!isfinite(cfg->delay_backward_quarantine_s) ||
        cfg->delay_backward_quarantine_s < 0.0f ||
        cfg->delay_backward_quarantine_s > 3600.0f) return 0;
    if (!isfinite(cfg->delay_est_period_s) ||
        cfg->delay_est_period_s < 0.0f || cfg->delay_est_period_s > 3600.0f) return 0;
    if (!isfinite(cfg->max_delay_ms) ||
        cfg->max_delay_ms < 0.0f || cfg->max_delay_ms > 4096.0f) return 0;

    if (cfg->delay_mode == AEC_DELAY_MATCHED) {
        max_delay_samples = (int)ceilf(cfg->max_delay_ms * (float)cfg->sample_rate / 1000.0f);
        *delay_ring_size = max_delay_samples + 2 * *hop_size + 1;
        /* An estimator that cannot be sized makes the config invalid. */
        if (*delay_ring_size <= 0 ||
            delay_aec_get_mem_size(cfg->sample_rate, *hop_size, cfg->delay_num_filters) == 0)
            return 0;
    } else if (cfg->delay_mode == AEC_DELAY_FIXED) {
        *delay_ring_size = cfg->fixed_delay_samples + *hop_size;
        if (*delay_ring_size <= 0) return 0;
    } else {
        *delay_ring_size = 0;
    }
    return 1;
}

static void* pool_carve(PoolCursor* cursor, size_t count, size_t element_size) {
    size_t raw;
    size_t aligned;
    void* out;
    if (!cursor || count == 0 || element_size == 0) return NULL;
    raw = ck_mul_size(count, element_size);
    aligned = ck_align16_size(raw);
    if (MEM_SIZE_INVALID(raw) || MEM_SIZE_INVALID(aligned) || aligned > cursor->remaining)
        return NULL;
    out = cursor->ptr;
    cursor->ptr += aligned;
    cursor->remaining -= aligned;
    return out;
}

static size_t flow_buffer_size(int hop, int n, int delay_ring_size, int enable_post) {
    size_t total = 0;
    int i;
    for (i = 0; i < 2; ++i) total = ck_field_size(total, (size_t)hop, sizeof(float));
    if (enable_post) {
        for (i = 0; i < 4; ++i) total = ck_field_size(total, (size_t)n, sizeof(Complex));
        for (i = 0; i < 2; ++i) total = ck_field_size(total, (size_t)n, sizeof(float));
    }
    if (delay_ring_size > 0)
        total = ck_field_size(total, (size_t)delay_ring_size, sizeof(float));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

static int carve_buffers(Mcab* p, PoolCursor* cursor) {
    int n = p->n_freqs;
    int hop = p->hop_size;

    p->mic_lane = (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    p->aligned_ref = (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    if (!p->mic_lane || !p->aligned_ref) return 0;
    if (p->cfg.enable_post) {
        p->fused_error = (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
        p->fused_near = (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
        p->residual_work = (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
        p->residual_proj = (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
        p->fused_r2 = (float*)pool_carve(cursor, (size_t)n, sizeof(float));
        p->fused_comfort = (float*)pool_carve(cursor, (size_t)n, sizeof(float));
        if (!p->fused_error || !p->fused_near || !p->residual_work || !p->residual_proj ||
            !p->fused_r2 || !p->fused_comfort) return 0;
    }
    if (p->delay_ring_size > 0) {
        p->delay_ring = (float*)pool_carve(cursor, (size_t)p->delay_ring_size, sizeof(float));
        if (!p->delay_ring) return 0;
    }
    return 1;
}

static uint32_t fnv1a_str(const char* text, uint32_t hash) {
    while (*text) {
        hash ^= (uint32_t)(unsigned char)*text++;
        hash *= 16777619u;
    }
    return hash;
}

/* The carve text; do not alter without bumping MCAB_LAYOUT_VERSION. Tunables are not folded in
 * (enable_post and delay_mode change `bytes`, which init compares). */
static uint32_t mcab_build_flags_hash(void) {
    uint32_t hash = 2166136261u;
    hash = fnv1a_str(AUDIO_PIPELINE_BACKEND_STR, hash);
    hash = fnv1a_str("|carve:mcab(cfg-v1),hop2,fuse?(complex4,float2),lanebind,delayring?", hash);
    hash = fnv1a_str("|align16", hash);
    return ac_memreq_abi_hash(hash);
}

McabConfig mcab_default_config(int sample_rate) {
    McabConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = 0;
    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = DA_NUM_FILTERS;
    cfg.fixed_delay_samples = -1;
    cfg.capture_proxy_channel = 0;
    cfg.delay_backward_quarantine_enabled = 0;
    cfg.delay_backward_quarantine_s = 1.0f;
    cfg.delay_est_period_s = 0.0f;
    cfg.max_delay_ms = 1024.0f;
    cfg.enable_post = 1;
    return cfg;
}

int mcab_apply_lane_contract(const McabConfig* cfg, AecConfig* lane) {
    int fft, hop, n, ring;
    if (!lane || !derive_flow(cfg, &fft, &hop, &n, &ring) ||
        lane->sample_rate != cfg->sample_rate) return -1;
    lane->fft_size = fft;
    lane->delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    lane->delay_num_filters = DA_NUM_FILTERS;
    lane->fixed_delay_samples = -1;
    lane->enable_delay_est = 0;
    lane->enable_res = 0;
    lane->return_res_context = 1;
    /* Front only: the lanes keep the linear half of the context; with a post stage their own
     * G_res is still never read. */
    lane->spatial_linear_context = cfg->enable_post ? AEC_CONTEXT_NO_GAIN : AEC_CONTEXT_LINEAR_ONLY;
    /* The lanes never estimate delay; the period is the flow's estimator duty cycle. */
    lane->delay_est_period_s = cfg->delay_est_period_s;
    return 0;
}

int mcab_lane_config(const McabConfig* cfg, AecConfig* out) {
    AecConfig lane;
    if (!cfg || !out) return -1;
    memset(&lane, 0, sizeof(lane));
    aec_config_from_preset(&lane, AEC_PRESET_BALANCED, cfg->sample_rate);
    if (mcab_apply_lane_contract(cfg, &lane) != 0) return -1;
    *out = lane;
    return 0;
}

int mcab_delay_estimator_config(const McabConfig* cfg, McabDelayEstimatorConfig* out) {
    int fft, hop, n, ring;
    if (!out || !derive_flow(cfg, &fft, &hop, &n, &ring)) return -1;
    memset(out, 0, sizeof(*out));
    out->required = cfg->delay_mode == AEC_DELAY_MATCHED;
    out->sample_rate = cfg->sample_rate;
    out->hop_size = hop;
    out->num_filters = cfg->delay_num_filters;
    return 0;
}

int mcab_get_mem_breakdown(const McabConfig* cfg, McabMemBreakdown* out) {
    int fft, hop, n, ring;
    size_t buffer_bytes, total;
    if (!out || !derive_flow(cfg, &fft, &hop, &n, &ring)) return -1;
    buffer_bytes = flow_buffer_size(hop, n, ring, cfg->enable_post);
    total = ck_add_size(ck_align16_size(sizeof(Mcab)), buffer_bytes);
    if (buffer_bytes == 0 || MEM_SIZE_INVALID(total)) return -1;
    memset(out, 0, sizeof(*out));
    out->control_bytes = ck_align16_size(sizeof(Mcab));
    out->buffer_bytes = buffer_bytes;
    out->total_bytes = total;
    out->hop_size = hop;
    out->fft_size = fft;
    out->n_freqs = n;
    return 0;
}

int mcab_get_mem_requirements(const McabConfig* cfg, McabMemReq* out) {
    McabMemBreakdown b;
    uint32_t backend = mcab_backend_id();
    if (!out || backend == 0u || mcab_get_mem_breakdown(cfg, &b) != 0) return -1;
    memset(out, 0, sizeof(*out));
    out->descriptor_version = MCAB_DESCRIPTOR_VERSION;
    out->layout_version = MCAB_LAYOUT_VERSION;
    out->backend_id = backend;
    out->build_flags_hash = mcab_build_flags_hash();
    out->alignment = 16u;
    out->bytes = (uint64_t)b.total_bytes;
    return 0;
}

/* Read-only checks of what the public lane and estimator views expose (see mcab.h). */
static int lane_matches(const Aec* lane, int hop, int n, int enable_post) {
    AecResContext rc;
    AecLinearContext lc;
    if (aec_hop_size(lane) != hop) return 0;
    aec_get_res_context(lane, &rc);
    if (rc.hop_size != hop || rc.n_freqs != n) return 0;
    /* The seam pointers exist only with the context returned (enable_res or return_res_context);
     * no own gain means a spatial context level, which in turn needs enable_res = 0. */
    if (!rc.error_spec || !rc.echo_spec || !rc.near_spec || !rc.far_spec || !rc.formed_hop ||
        rc.res_gain) return 0;
    if (enable_post ? (!rc.r2 || !rc.comfort_noise) : (rc.r2 || rc.comfort_noise)) return 0;
    /* Only EXTERNAL_ALIGNED reports a locked zero offset on a lane that has not run. */
    aec_get_linear_context(lane, &lc);
    return lc.delay_state == AEC_LINEAR_DELAY_LOCKED && lc.delay_samples == 0;
}

static int libs_match(const McabConfig* cfg, const McabLibs* libs, int hop, int n) {
    int ch, other;
    if (!libs) return 0;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        if (!libs->lanes[ch] || !lane_matches(libs->lanes[ch], hop, n, cfg->enable_post))
            return 0;
        for (other = 0; other < ch; ++other)
            if (libs->lanes[other] == libs->lanes[ch]) return 0;
    }
    if (cfg->delay_mode == AEC_DELAY_MATCHED) {
        const DelayAec* d = libs->delay;
        int rate_factor = cfg->sample_rate == 48000 ? 3 : 1;
        if (!d || d->est.matched_filter.num_filters != cfg->delay_num_filters ||
            d->rate_factor != rate_factor) return 0;
        /* The 48 kHz sidechain scratch is sized from the hop it was built with. */
        if (rate_factor == 3 && (!d->near16_scratch || !d->far16_scratch ||
                                 d->resample_cap < DA_RESAMPLE48_CAP_FOR(hop))) return 0;
    } else if (libs->delay) {
        return 0;
    }
    return 1;
}

Mcab* mcab_init_ex(void* mem, size_t bytes, const McabConfig* cfg, const McabLibs* libs,
                   const McabMemReq* expected) {
    McabConfig cfg_copy;
    McabMemReq current;
    Mcab* p;
    PoolCursor cursor;
    int fft, hop, n, ring, ch;

    if (!mem || !cfg || !libs) return NULL;
    cfg_copy = *cfg;
    if (mcab_get_mem_requirements(&cfg_copy, &current) != 0) return NULL;
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
    if (!MEM_IS_ALIGNED16(mem) || current.bytes > (uint64_t)SIZE_MAX ||
        (uint64_t)bytes < current.bytes) return NULL;
    if (!derive_flow(&cfg_copy, &fft, &hop, &n, &ring) ||
        !libs_match(&cfg_copy, libs, hop, n)) return NULL;

    memset(mem, 0, (size_t)current.bytes);
    p = (Mcab*)mem;
    p->cfg = cfg_copy;
    p->sample_rate = cfg_copy.sample_rate;
    p->fft_size = fft;
    p->hop_size = hop;
    p->n_freqs = n;
    p->delay_ring_size = ring;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) p->lanes[ch] = libs->lanes[ch];
    p->delay = cfg_copy.delay_mode == AEC_DELAY_MATCHED ? libs->delay : NULL;
    {
        /* Seconds -> hops once against the resolved grid, with lib/aec's floor of 1. The duty
         * floor is 1, not 2: 2 would cap delay_est_period_s = 0.0 at 50 % engagement. */
        float q_hop_s = (float)hop / (float)cfg_copy.sample_rate;
        int q_hops = (int)lrintf(cfg_copy.delay_backward_quarantine_s / q_hop_s);
        int hold = (int)lrintf(DUTY_HOLD_S / q_hop_s);
        int period = (int)lrintf(cfg_copy.delay_est_period_s / q_hop_s) / DUTY_PERIOD_DIVISOR;
        p->delay_quarantine_hops = q_hops < 1 ? 1 : q_hops;
        p->delay_quarantine_left = -1;
        p->duty_hold_hops = hold < 1 ? 1 : hold;
        p->duty_period_hops = period < 1 ? 1 : period;
        p->duty_erle_leak_db = mcab_duty_leak_db_for_grid(hop, cfg_copy.sample_rate);
        p->duty_last_delay = -1;
    }
    p->pool_size = (size_t)current.bytes;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(*p));
    cursor.remaining = (size_t)current.bytes - ALIGN16(sizeof(*p));
    if (!carve_buffers(p, &cursor) || cursor.remaining != 0) return NULL;
    return p;
}

Mcab* mcab_init(void* mem, size_t bytes, const McabConfig* cfg, const McabLibs* libs) {
    return mcab_init_ex(mem, bytes, cfg, libs, NULL);
}

Mcab* mcab_create(const McabConfig* cfg, const McabLibs* libs) {
    McabMemReq req;
    Mcab* p;
    void* pool = NULL;
    if (!cfg || !libs || mcab_get_mem_requirements(cfg, &req) != 0 ||
        req.bytes > (uint64_t)SIZE_MAX) return NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool)
        return NULL;
    p = mcab_init(pool, (size_t)req.bytes, cfg, libs);
    if (!p) {
        free(pool);
        return NULL;
    }
    p->owned_heap = pool;
    return p;
}

static int complex_close(Complex a, Complex b) {
    float dr = fabsf(a.r - b.r);
    float di = fabsf(a.i - b.i);
    float scale = 1.0f + fabsf(a.r) + fabsf(a.i);
    return dr <= 1e-5f * scale && di <= 1e-5f * scale;
}

/* The fuse precondition: a lane with a different far spectrum must not be folded into the beam
 * (Python: 4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts). With the borrow chain shown to have run
 * this hop, four counter reads replace 4*n_freqs compares. */
int mcab_far_spec_agrees(int shared_far_provenance, const Complex* lane_far_spec,
                         const Complex* reference_far_spec, int n_freqs) {
    int k;
    if (!lane_far_spec || !reference_far_spec || n_freqs < 0) return 0;
    if (shared_far_provenance) return 1;
    for (k = 0; k < n_freqs; ++k) {
        if (!complex_close(lane_far_spec[k], reference_far_spec[k])) return 0;
    }
    return 1;
}

void mcab_admission_age(McabDelayAdmission* admission) {
    if (!admission || admission->ttl <= 0) return;
    admission->ttl -= 1;
    if (admission->ttl <= 0) {
        admission->ttl = 0;
        admission->candidate = 0;
    }
}

int mcab_admission_offer(McabDelayAdmission* admission, int accepted_delay, int estimated) {
    if (!admission) return 0;
    /* Too small to be worth four IR shifts: absorbed and left to age (lib/aec's Path B). */
    if (abs(estimated - accepted_delay) <= MCAB_DELAY_CHANGE_MIN_SAMPLES) return 0;
    if (admission->ttl > 0 &&
        abs(estimated - admission->candidate) < MCAB_DELAY_CHANGE_CONFIRM_SAMPLES) {
        admission->candidate = 0;
        admission->ttl = 0;
        return 1;
    }
    admission->candidate = estimated;
    admission->ttl = MCAB_DELAY_CHANGE_CANDIDATE_TTL;
    return 0;
}

/* Duty watchdog signal: the best windowed ERLE of the four lanes (one quiet lane is no evidence
 * that the shared alignment collapsed); last hop's readings, 0 until the ERLE machinery ran. */
static float max_lane_erle_windowed(const Mcab* p) {
    float best = 0.0f;
    int ch;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        AecDebugStatus status;
        if (!p->lanes[ch]) continue;
        aec_debug_status(p->lanes[ch], &status);
        if (status.erle_windowed_db > best) best = status.erle_windowed_db;
    }
    return best;
}

static McabDelayState update_shared_delay(Mcab* p, const float* capture, const float* render) {
    McabDelayState state;
    int estimated;
    int eligible;
    int was_usable;
    int now_usable;
    int accepted;
    int run_filter;

    memset(&state, 0, sizeof(state));
    if (p->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) {
        state.delay_samples = 0;
        state.confidence = 1.0f;
        state.solid = 1;
        p->accepted_delay = 0;
        p->last_delay = state;
        return state;
    }
    if (p->cfg.delay_mode == AEC_DELAY_FIXED) {
        state.delay_samples = p->cfg.fixed_delay_samples;
        state.confidence = 1.0f;
        /* Runs right before align_render(): readable once the stored samples cover the delay
         * (lib/aec's `filled >= delay + hop`), so LOCKED lands on the hop the audio switches. */
        state.solid = p->delay_samples_seen >= (uint64_t)p->cfg.fixed_delay_samples;
        p->accepted_delay = p->cfg.fixed_delay_samples;
        p->last_delay = state;
        return state;
    }

    /* The estimator is fed every hop (decimators and ring stay gapless); only the analysis is
     * decimated once the estimate held still, so a movement is seen within duty_period_hops. */
    run_filter = 1;
    if (p->duty_active) {
        p->duty_pos += 1;
        if (p->duty_pos >= p->duty_period_hops) p->duty_pos = 0;
        run_filter = (p->duty_pos == 0);
    }
    if (run_filter) p->duty_hops_run += 1;
    (void)delay_aec_accumulate_ex(p->delay, capture, render, p->hop_size, run_filter);
    p->delay_calls += 1;
    {
        int cur = delay_aec_estimated_delay(p->delay);
        int solid = delay_aec_is_solid(p->delay);
        if (!solid || cur < 0 || cur != p->duty_last_delay) {
            p->duty_active = 0;
            p->duty_stable_hops = 0;
        } else if (!p->duty_active) {
            p->duty_stable_hops += 1;
            if (p->duty_stable_hops >= p->duty_hold_hops) {
                p->duty_active = 1;
                p->duty_pos = 0;
                p->duty_erle_peak = max_lane_erle_windowed(p);
            }
        }
        p->duty_last_delay = cur;
    }
    if (p->duty_active) {
        /* A drop of DUTY_ERLE_COLLAPSE_DB from the leaky peak resumes full rate; armed only once
         * the peak clears that figure. */
        float erle = max_lane_erle_windowed(p);
        if (erle > p->duty_erle_peak) p->duty_erle_peak = erle;
        else p->duty_erle_peak -= p->duty_erle_leak_db;
        if (p->duty_erle_peak > DUTY_ERLE_COLLAPSE_DB &&
            erle < p->duty_erle_peak - DUTY_ERLE_COLLAPSE_DB) {
            p->duty_active = 0;
            p->duty_stable_hops = 0;
            p->duty_erle_peak = 0.0f;
        }
    }

    estimated = delay_aec_estimated_delay(p->delay);
    eligible = estimated >= 0 && delay_aec_is_solid(p->delay) && delay_aec_n_updates(p->delay) >= 3;

    /* Aged ahead of the quarantine and admission, as lib/aec ages its pending delay. */
    mcab_admission_age(&p->delay_admission);

    /* Backward-jump quarantine: judged on the capture-proxy lane's cancellation read before this
     * hop's lanes run; armed once per backward episode, adopted at 0. */
    if (eligible && p->cfg.delay_backward_quarantine_enabled &&
        p->last_delay.solid && estimated < p->accepted_delay &&
        p->lanes[p->cfg.capture_proxy_channel] &&
        aec_linear_is_cancelling(p->lanes[p->cfg.capture_proxy_channel])) {
        if (p->delay_quarantine_left < 0) p->delay_quarantine_left = p->delay_quarantine_hops;
        if (p->delay_quarantine_left > 0) {
            p->delay_quarantine_left--;
            eligible = 0;
        }
    } else {
        p->delay_quarantine_left = -1;
    }

    /* `solid` is sticky once an accepted generation exists; `changed` marks a new usable
     * generation (a value compare would miss an acquisition or relock at delay 0). First
     * acquisition is immediate (lib/aec's Path A); later moves go through the admission, since
     * each costs an IR shift on all four lanes. accepted_delay moves only on acceptance, so the
     * served alignment and the lanes' realign land on the same hop. */
    was_usable = p->last_delay.solid;
    now_usable = was_usable || eligible;
    accepted = 0;
    if (eligible) {
        if (!was_usable) {
            accepted = 1;
            p->delay_admission.candidate = 0;
            p->delay_admission.ttl = 0;
        } else {
            accepted = mcab_admission_offer(&p->delay_admission, p->accepted_delay, estimated);
        }
    }
    state.changed = accepted;
    if (accepted) p->accepted_delay = estimated;
    state.delay_samples = p->accepted_delay;
    state.confidence = delay_aec_confidence(p->delay);
    state.solid = now_usable;
    state.estimator_calls = p->delay_calls;
    state.estimator_updates = delay_aec_n_updates(p->delay);
    p->last_delay = state;
    return state;
}

static int align_render(Mcab* p, const float* render, int delay_samples) {
    int hop;
    int ring;
    int write_at;
    int read_at;
    int first;
    uint64_t start;
    if (p->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) {
        memcpy(p->aligned_ref, render, (size_t)p->hop_size * sizeof(float));
        p->delay_samples_seen += (uint64_t)p->hop_size;
        return 1;
    }
    if (!p->delay_ring || delay_samples < 0 ||
        delay_samples > p->delay_ring_size - p->hop_size) return 0;

    hop = p->hop_size;
    ring = p->delay_ring_size;
    start = p->delay_samples_seen;
    /* One hop wraps at most once: one or two bounded runs, `start` reduced once (no per-sample
     * 64-bit divide). */
    write_at = (int)(start % (uint64_t)ring);
    first = ring - write_at;
    if (first > hop) first = hop;
    memcpy(p->delay_ring + write_at, render, (size_t)first * sizeof(float));
    if (first < hop) memcpy(p->delay_ring, render + first, (size_t)(hop - first) * sizeof(float));
    /* Whole-hop decision on the predicate of FIXED's `solid`: until the samples seen cover the
     * offset the far is the raw hop (lib/aec's rule), never silence or a splice. */
    if (start >= (uint64_t)delay_samples) {
        /* write_at - delay_samples + ring lies in (0, 2*ring): one conditional subtract. */
        read_at = write_at - delay_samples;
        if (read_at < 0) read_at += ring;
        first = ring - read_at;
        if (first > hop) first = hop;
        memcpy(p->aligned_ref, p->delay_ring + read_at, (size_t)first * sizeof(float));
        if (first < hop)
            memcpy(p->aligned_ref + first, p->delay_ring, (size_t)(hop - first) * sizeof(float));
    } else {
        memcpy(p->aligned_ref, render, (size_t)hop * sizeof(float));
    }
    p->delay_samples_seen += (uint64_t)hop;
    return 1;
}

/* Points dst at src's buffers instead of copying: AecResContext aliases the lane's per-hop
 * buffers, a lane runs again only inside mcab_process_front() or after a reset, and the front
 * view is valid until then. Reset does not clear that memory; fuse refuses a stale view. */
static int bind_lane_view(McabLaneSnapshot* dst, const AecResContext* src, int n_freqs,
                          int residual_half) {
    if (!dst || !src || src->n_freqs != n_freqs ||
        !src->error_spec || !src->echo_spec || !src->far_spec || !src->near_spec) return 0;
    if (residual_half && (!src->r2 || !src->comfort_noise)) return 0;

    dst->error_spec = src->error_spec;
    dst->echo_spec = src->echo_spec;
    dst->far_spec = src->far_spec;
    dst->near_spec = src->near_spec;
    dst->r2 = src->r2;
    dst->comfort_noise = src->comfort_noise;
    dst->formed_hop = src->formed_hop;

    dst->far_power = src->far_power;
    dst->dt_indicator = src->dt_indicator;
    dst->saturation_level = src->saturation_level;
    dst->filter_converged = src->filter_converged;
    dst->usable_linear = src->usable_linear;
    return 1;
}

/* Legal at any time; the lanes' buffers are not cleared (stale views, not a fault). */
void mcab_reset(Mcab* p) {
    int ch;
    if (!p || p->destroyed) return;
    if (p->delay) delay_aec_reset(p->delay);
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) aec_reset(p->lanes[ch]);
    if (p->delay_ring) memset(p->delay_ring, 0, (size_t)p->delay_ring_size * sizeof(float));
    if (p->aligned_ref) memset(p->aligned_ref, 0, (size_t)p->hop_size * sizeof(float));

    p->delay_samples_seen = 0;
    p->accepted_delay = p->cfg.delay_mode == AEC_DELAY_FIXED ? p->cfg.fixed_delay_samples : 0;
    /* The alignment the quarantine protected is abandoned; the config-derived window stays. */
    p->delay_quarantine_left = -1;
    p->delay_admission.candidate = 0;
    p->delay_admission.ttl = 0;
    p->realign_warm_lanes = 0;
    p->realign_soft_lanes = 0;
    p->delay_calls = 0;
    p->far_spec_shared_hop = 0;
    p->far_spec_mixed_hop = 0;
    /* The duty machine measured the abandoned alignment; the grid-derived windows stay. */
    p->duty_active = 0;
    p->duty_stable_hops = 0;
    p->duty_pos = 0;
    p->duty_last_delay = -1;
    p->duty_erle_peak = 0.0f;
    p->duty_hops_run = 0;
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->front_valid = 0;
}

int mcab_process_front(Mcab* p, const float* microphones_interleaved,
                       const float* far_reference, McabFront* out) {
    McabDelayState delay;
    int hop;
    int ch;
    int i;
    int old_align;
    uint32_t t0;
    const Complex* shared_far_spec = NULL;
    long far_fft_before[MCAB_CHANNELS];
    int lanes_borrowed;

    if (!p || p->destroyed || !microphones_interleaved || !far_reference || !out)
        return MCAB_INVALID_ARGUMENT;
    /* Cleared once the call is accepted, so a hop that bails out reports zero for stages it never
     * reached; lane fields accumulate over the four lanes; fuse_us is cleared in mcab_fuse(). */
    p->last_timing.delay_us = 0;
    p->last_timing.frontend_us = 0;
    p->last_timing.linear_us = 0;
    p->last_timing.lane_res_us = 0;
    p->far_spec_shared_hop = 0;
    p->far_spec_mixed_hop = 0;
    p->front_valid = 0;

    hop = p->hop_size;
    if (!skn_all_finite_f32(microphones_interleaved, (size_t)hop * MCAB_CHANNELS) ||
        !skn_all_finite_f32(far_reference, (size_t)hop))
        return MCAB_INVALID_ARGUMENT;

    /* Only lane 0 with proxy 0 can reuse this extraction: earlier lanes overwrite mic_lane before
     * a later proxy lane runs. */
    for (i = 0; i < hop; ++i)
        p->mic_lane[i] = microphones_interleaved[i * MCAB_CHANNELS + p->cfg.capture_proxy_channel];
    /* The alignment the lanes were served before this hop's decision (raw 0 until the estimate
     * first turns solid). */
    old_align = p->last_delay.solid ? p->accepted_delay : 0;
    t0 = mcab_now_us();
    delay = update_shared_delay(p, p->mic_lane, far_reference);
    p->last_timing.delay_us = mcab_now_us() - t0;
    if (!align_render(p, far_reference, delay.delay_samples)) {
        mcab_reset(p);
        return MCAB_DSP_ERROR;
    }
    if (delay.changed) {
        /* Realign instead of reset: each converged filter is shifted by the alignment delta (warm
         * when the evidence gate holds, filter-only reset otherwise), so cancellation survives
         * and the WOLA sequences continue. */
        int hop_warm = 0;
        int hop_soft = 0;
        for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
            int outcome = aec_apply_external_realign(p->lanes[ch], delay.delay_samples - old_align);
            if (outcome == 1) {
                p->realign_warm_lanes = mcab_count_add_sat(p->realign_warm_lanes, 1);
                hop_warm += 1;
            } else if (outcome == 0) {
                p->realign_soft_lanes = mcab_count_add_sat(p->realign_soft_lanes, 1);
                hop_soft += 1;
            }
        }
        /* A soft outcome zeroes that lane's far OLA history, so a mixed hop leaves the lanes out
         * of lockstep: no far-FFT borrowing and no far agreement requirement on this hop. */
        p->far_spec_mixed_hop = (hop_warm > 0 && hop_soft > 0);
    }

    for (ch = 0; ch < MCAB_CHANNELS; ++ch)
        far_fft_before[ch] = aec_far_fft_real_compute_count(p->lanes[ch]);
    lanes_borrowed = 1;

    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        AecResContext context;
        if (ch != 0 || p->cfg.capture_proxy_channel != 0) {
            for (i = 0; i < hop; ++i)
                p->mic_lane[i] = microphones_interleaved[i * MCAB_CHANNELS + ch];
        }
        /* Lane 0 produces this hop's far spectrum; lanes 1-3 copy it after their prior-hop RSA
         * read. On a mixed warm/soft hop all lanes compute locally. */
        if (ch == 0 || p->far_spec_mixed_hop) {
            aec_process_context(p->lanes[ch], p->mic_lane, p->aligned_ref);
        } else {
            aec_process_context_shared_far_from(p->lanes[ch], p->mic_lane, p->aligned_ref,
                                                p->lanes[0]);
        }
        aec_get_res_context(p->lanes[ch], &context);
        if (ch == 0) shared_far_spec = context.far_spec;
        if (!context.formed_hop ||
            !bind_lane_view(&p->snapshots[ch], &context, p->n_freqs, p->cfg.enable_post)) {
            mcab_reset(p);
            return MCAB_DSP_ERROR;
        }
        {
            AecStageTiming lane_time;
            aec_get_last_timing(p->lanes[ch], &lane_time);
            /* Lane-internal delay work is structurally zero (EXTERNAL_ALIGNED) and summed anyway. */
            p->last_timing.delay_us += lane_time.delay_us;
            p->last_timing.frontend_us += lane_time.frontend_us;
            p->last_timing.linear_us += lane_time.linear_us;
            p->last_timing.lane_res_us += lane_time.res_us;
        }
        /* Borrow-chain signature: the computing lane ran one far transform, a borrowing lane
         * none; anything else sends fuse_contexts() to the per-bin comparison. */
        {
            long after = aec_far_fft_real_compute_count(p->lanes[ch]);
            if (!mcab_far_fft_delta_matches(far_fft_before[ch], after, ch == 0 ? 1 : 0))
                lanes_borrowed = 0;
        }
    }
    p->far_spec_shared_hop = lanes_borrowed && shared_far_spec != NULL;

    p->front_valid = 1;

    out->delay = delay;
    out->hop_size = p->hop_size;
    out->n_channels = MCAB_CHANNELS;
    out->n_freqs = p->n_freqs;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        out->linear_spectra[ch] = p->snapshots[ch].error_spec;
        out->formed_hop[ch] = p->snapshots[ch].formed_hop;
    }
    out->aligned_ref = p->aligned_ref;
    return MCAB_OK;
}

static int complex_vector_finite(const Complex* values, int count) {
    if (!values || count < 0) return 0;
    return skn_all_finite_cf32(values, (size_t)count);
}

static int fuse_contexts(Mcab* p, const Complex* weights, const Complex* trusted_beamformed_error,
                         int* all_converged, int* usable_linear, float* max_dt,
                         float* max_saturation) {
    int n = p->n_freqs;
    int k;
    int ch;
    float base_far_power = p->snapshots[0].far_power;

    *all_converged = 1;
    *usable_linear = 1;
    *max_dt = 0.0f;
    *max_saturation = 0.0f;

    if (trusted_beamformed_error) {
        if (!complex_vector_finite(trusted_beamformed_error, n)) return 0;
        /* No copy into fused_error: the caller owns it for the whole call. */
    } else {
        memset(p->fused_error, 0, (size_t)n * sizeof(Complex));
    }
    memset(p->fused_near, 0, (size_t)n * sizeof(Complex));
    memset(p->residual_proj, 0, (size_t)n * sizeof(Complex));
    memset(p->fused_comfort, 0, (size_t)n * sizeof(float));

    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        const Complex* lane_weights = weights + (size_t)ch * (size_t)n;
        /* A lane zeroed in every bin contributes nothing, so it must not vote on the scalars that
         * steer the post stage (bypassed beamformer: [1,0,0,0]). */
        int lane_contributes = 0;
        for (k = 0; k < n; ++k) {
            if (lane_weights[k].r != 0.0f || lane_weights[k].i != 0.0f) {
                lane_contributes = 1;
                break;
            }
        }
        float delta = fabsf(p->snapshots[ch].far_power - base_far_power);
        float scale = 1.0f + fabsf(base_far_power);
        if (delta > 1e-5f * scale) return 0;
        if (lane_contributes) {
            if (!p->snapshots[ch].filter_converged) *all_converged = 0;
            if (!p->snapshots[ch].usable_linear) *usable_linear = 0;
            if (p->snapshots[ch].dt_indicator > *max_dt) *max_dt = p->snapshots[ch].dt_indicator;
            if (p->snapshots[ch].saturation_level > *max_saturation)
                *max_saturation = p->snapshots[ch].saturation_level;
        }

        /* Skipped on a mixed warm/soft-realign hop; the fused X is lane 0's far_spec either way
         * (Python: 4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts). */
        if (!p->far_spec_mixed_hop &&
            !mcab_far_spec_agrees(p->far_spec_shared_hop, p->snapshots[ch].far_spec,
                                  p->snapshots[0].far_spec, n)) return 0;

        if (!trusted_beamformed_error)
            mcab_projection_cmac(p->fused_error, lane_weights, p->snapshots[ch].error_spec, n);
        mcab_projection_cmac(p->fused_near, lane_weights, p->snapshots[ch].near_spec, n);
        mcab_residual_vector(p->residual_work, p->snapshots[ch].echo_spec,
                             p->snapshots[ch].r2, n);
        mcab_projection_cmac(p->residual_proj, lane_weights, p->residual_work, n);
        mcab_comfort_accumulate(p->fused_comfort, lane_weights,
                                p->snapshots[ch].comfort_noise, n);
    }

    mcab_complex_mag2(p->fused_r2, p->residual_proj, n);
    /* Finite weights can still overflow the projections (|w| near 1e19 squares past FLT_MAX). */
    return complex_vector_finite(p->fused_near, n) &&
           (trusted_beamformed_error || complex_vector_finite(p->fused_error, n)) &&
           skn_all_finite_f32(p->fused_r2, (size_t)n) &&
           skn_all_finite_f32(p->fused_comfort, (size_t)n);
}

int mcab_fuse(Mcab* p, const Complex* weights, const Complex* beamformed_error,
              McabFrame* frame) {
    int all_converged;
    int usable_linear;
    float max_dt;
    float max_saturation;
    uint32_t t0;

    if (!p || p->destroyed || !weights || !frame || !p->cfg.enable_post)
        return MCAB_INVALID_ARGUMENT;
    if (!p->front_valid) return MCAB_SEQUENCE_ERROR;
    if (!mcab_weights_valid(weights, (size_t)MCAB_CHANNELS * (size_t)p->n_freqs))
        return MCAB_INVALID_ARGUMENT;

    p->last_timing.fuse_us = 0;
    t0 = mcab_now_us();
    if (!fuse_contexts(p, weights, beamformed_error, &all_converged, &usable_linear, &max_dt,
                       &max_saturation)) {
        mcab_reset(p);
        return MCAB_DSP_ERROR;
    }
    p->last_timing.fuse_us = mcab_now_us() - t0;

    memset(frame, 0, sizeof(*frame));
    frame->n_freqs = p->n_freqs;
    frame->hop = p->hop_size;
    frame->e = beamformed_error ? beamformed_error : p->fused_error;
    frame->y = p->fused_near;
    frame->x = p->snapshots[0].far_spec;
    frame->r2 = p->fused_r2;
    frame->cn = p->fused_comfort;
    frame->g_res_ctx = NULL;     /* the lanes compute no gain of their own */
    frame->far_hop = p->aligned_ref;
    frame->formed_hop = NULL;    /* one hop per lane: McabFront.formed_hop */
    frame->far_power = p->snapshots[0].far_power;
    frame->max_dt = max_dt;
    frame->max_sat = max_saturation;
    frame->all_converged = all_converged;
    frame->usable_linear = usable_linear;
    frame->clock_drift = p->cfg.delay_mode == AEC_DELAY_MATCHED
        ? delay_aec_has_clockdrift(p->delay) : 0;
    frame->delay_changed = p->last_delay.changed;
    frame->delay_locked = p->last_delay.solid;
    return MCAB_OK;
}

void mcab_destroy(Mcab* p) {
    void* owned_heap;
    if (!p || p->destroyed) return;
    owned_heap = p->owned_heap;
    p->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

int mcab_hop_size(const Mcab* p) { return p && !p->destroyed ? p->hop_size : -1; }
int mcab_fft_size(const Mcab* p) { return p && !p->destroyed ? p->fft_size : -1; }
int mcab_n_freqs(const Mcab* p) { return p && !p->destroyed ? p->n_freqs : -1; }
int mcab_sample_rate(const Mcab* p) { return p && !p->destroyed ? p->sample_rate : -1; }
int mcab_enable_post(const Mcab* p) { return p && !p->destroyed ? p->cfg.enable_post : -1; }

int mcab_matched_filter_count(const Mcab* p) {
    return p && !p->destroyed && p->cfg.delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
}

int mcab_linear_aec_count(const Mcab* p) { return p && !p->destroyed ? MCAB_CHANNELS : 0; }

long mcab_far_fft_real_compute_count(const Mcab* p) {
    long total = 0;
    int ch;
    if (!p || p->destroyed) return 0;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch)
        total = mcab_count_add_sat(total, aec_far_fft_real_compute_count(p->lanes[ch]));
    return total;
}

long mcab_realign_warm_lane_count(const Mcab* p) {
    return p && !p->destroyed ? p->realign_warm_lanes : 0;
}

long mcab_realign_soft_lane_count(const Mcab* p) {
    return p && !p->destroyed ? p->realign_soft_lanes : 0;
}

unsigned long long mcab_duty_hops_total(const Mcab* p) {
    return p && !p->destroyed ? p->delay_calls : 0ull;
}

unsigned long long mcab_duty_hops_run(const Mcab* p) {
    return p && !p->destroyed ? p->duty_hops_run : 0ull;
}

int mcab_far_spec_provenance(const Mcab* p) {
    return p && !p->destroyed ? p->far_spec_shared_hop : 0;
}

/* A period of one hop is "analyse every hop" at the decision site, while arming, hold and
 * watchdog keep running and stay measurable through the census. */
void mcab_pin_duty_full_rate(Mcab* p) {
    if (!p || p->destroyed) return;
    p->duty_period_hops = 1;
}

float mcab_duty_erle_leak_db(const Mcab* p) {
    return p && !p->destroyed ? p->duty_erle_leak_db : 0.0f;
}

float mcab_duty_erle_peak(const Mcab* p) {
    return p && !p->destroyed ? p->duty_erle_peak : 0.0f;
}

int mcab_pending_delay_candidate(const Mcab* p) {
    if (!p || p->destroyed || p->delay_admission.ttl <= 0) return -1;
    return p->delay_admission.candidate;
}

void mcab_get_last_timing(const Mcab* p, McabLastTiming* out) {
    if (!out) return;
    if (!p || p->destroyed) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = p->last_timing;
}
