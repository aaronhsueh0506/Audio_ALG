/** Implementation of 4aec_nr_res.h: one shared DelayAec -> one aligned render -> four independent linear AEC filters -> externally supplied beamformer weights -> coherent projection of the four lane contexts (four_aec_fuse). The application chains the RES, NR, comfort-noise and synthesis stages over the fused frame. Pool-first layout as AudioPipeline; create() is the heap convenience. */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>

#include "4aec_nr_res.h"
#include "4aec_nr_res_internal.h"
#include "fft_wrapper.h"
#include "utility/mem_align.h"
#include "4aec_projection_kernels.h"
#include "utility/simd_kernel_nn.h"        /* skn_all_finite_cf32 */
#include "common/se_config.h"     /* se_derive(): grid + AecConfig, shared by every application */
#include "common/se_anr_stage.h"  /* se_anr_compose_config()                                     */

/* Matched-filter duty cycle: full-rate analysis until the estimate has been solid and unchanged for delay_est_init_s (lane AecConfig), then one hop in K = round(delay_est_period_s / hop_s) / DUTY_PERIOD_DIVISOR (cfg.delay_est_period_s). Divisor and hold floor are the schedule's shape; the period floor is 1, so 0.0 means true full rate. The watchdog leak is lib/aec's 0.1 dB per second, converted to this grid's hop by duty_leak_db_for_grid(). */
#define DUTY_PERIOD_DIVISOR       5
#define DUTY_ERLE_LEAK_DB         0.001f   /* per hop at DUTY_LEAK_REF_HOP_S */
#define DUTY_LEAK_REF_HOP_S       0.010f
#define DUTY_ERLE_COLLAPSE_DB     6.0f

#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

static uint32_t four_aec_nr_res_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0)
        return FOUR_AEC_NR_RES_BACKEND_NE10;
    return 0u;
}

/* Per-stage timing is diagnostic and off by default. This flag governs the wrapper's
 * own stages; AEC_STAGE_TIMING governs what the lanes report. `make PROFILE=1` sets
 * both. */
#ifndef FOUR_AEC_NR_RES_STAGE_TIMING
#define FOUR_AEC_NR_RES_STAGE_TIMING 0
#endif

#if FOUR_AEC_NR_RES_STAGE_TIMING
/* Microsecond monotonic stamp, truncated to 32 bits (consumers subtract in unsigned
 * arithmetic). A target without POSIX CLOCK_MONOTONIC names a no-argument function
 * returning uint32_t microseconds with -DFOUR_AEC_NR_RES_NOW_US=<fn> (a plain
 * identifier); lib/aec has its own AEC_NOW_US. */
#ifndef FOUR_AEC_NR_RES_NOW_US
#include <time.h>
static uint32_t four_aec_nr_res_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull
                      + (uint64_t)ts.tv_nsec / 1000ull);
}
#else
static uint32_t four_aec_nr_res_now_us(void) { return FOUR_AEC_NR_RES_NOW_US(); }
#endif
#else
#define four_aec_nr_res_now_us() 0u
#endif

typedef struct FourAecLaneSnapshot {
    /* Borrowed views into a lane's AecResContext buffers (lifetime: see
     * bind_lane_view()); const, as nothing here writes through them. */
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
} FourAecLaneSnapshot;

struct FourAecNrRes {
    FourAecNrResConfig cfg;
    int sample_rate;
    int fft_size;
    int hop_size;
    int n_freqs;
    int initialized_lanes;

    Aec* lanes[FOUR_AEC_NR_RES_CHANNELS];
    DelayAec shared_delay;

    FourAecLaneSnapshot snapshots[FOUR_AEC_NR_RES_CHANNELS];

    float* mic_lane;
    float* aligned_ref;

    float* delay_ring;
    int delay_ring_size;
    uint64_t delay_samples_seen;
    int accepted_delay;
    /* Backward-jump quarantine (cfg.delay_backward_quarantine_*): _left is the hops
     * remaining before a held earlier estimate is adopted anyway (-1 = disarmed, the only
     * state that may re-arm; 0 = armed and expired), _hops is the window converted once at
     * init. */
    int delay_quarantine_left;
    int delay_quarantine_hops;
    /* Two-step admission for a shared-delay change: the movement held for confirmation and
     * its remaining life in hops (4aec_nr_res_internal.h). Cleared by acceptance, expiry
     * and reset(). */
    FourAecDelayAdmission delay_admission;
    /* Cumulative realign outcomes across the lanes: warm (returned 1, IR shifted) vs soft (returned 0, filter-only reset). */
    long realign_warm_lanes;
    long realign_soft_lanes;
    uint64_t delay_calls;
    FourAecNrResDelayState last_delay;

    Complex* fused_error;
    Complex* fused_near;
    Complex* residual_work;
    Complex* residual_proj;    /* coherent residual vector, then its power in fused_r2 */
    float* fused_r2;
    float* fused_comfort;

    int front_valid;            /* a front frame was processed since the last reset */

    void* owned_heap;
    size_t pool_size;
    int destroyed;

    FourAecNrResLastTiming last_timing;

    /* This hop's answer to "did all four lanes consume one far spectrum", established from
     * the lanes' far-FFT counters in four_aec_front_process() and used by fuse_contexts()
     * via four_aec_nr_res_far_spec_agrees(). 0 (also after any bail-out) means not
     * established, which costs the full per-bin comparison. */
    int far_spec_shared_hop;

    /* This hop's answer to "did the lane loop route a mixed warm/soft realign outcome",
     * set in four_aec_front_process() from the per-hop outcome counts. fuse_contexts()
     * bypasses four_aec_nr_res_far_spec_agrees() on that hop, since the lanes' far_spec
     * legitimately diverge. 0 means the normal gate applies. */
    int far_spec_mixed_hop;

    /* Matched-filter duty cycle: the estimator is fed every hop and analysed at full rate until the estimate has been solid and unchanged for duty_hold_hops, then once per duty_period_hops until it is seen to move or lose confidence, or the watchdog sees cancellation collapse. duty_erle_peak is a leaky peak of the lanes' windowed ERLE decaying by duty_erle_leak_db per hop. duty_hops_run is counted where the decision is consumed; the census denominator is delay_calls. */
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

/* Watchdog leak in this grid's per-hop units; exactly 1.0f at the calibration grid (hop 160 at 16 kHz), which is not built here, hence the exposed four_aec_nr_res_duty_leak_db_for_grid(). */
static float duty_leak_db_for_grid(int hop_size, int sample_rate) {
    float hop_s;
    if (hop_size <= 0 || sample_rate <= 0) return DUTY_ERLE_LEAK_DB;
    hop_s = (float)hop_size / (float)sample_rate;
    return DUTY_ERLE_LEAK_DB * (hop_s / DUTY_LEAK_REF_HOP_S);
}

/* Returns 1 and fills the out-params, or 0 (NULL argument; rate outside {16000,
 * 48000}; grid, preset, mode, flag, filter length or delay setting outside its range;
 * or a MATCHED/FIXED request whose delay ring or estimator cannot be sized). Grid,
 * AecConfig and the delay-request rules come from se_derive(); what only this core has
 * (capture proxy, quarantine and delay-line ranges, shared ring and estimator sizes) is
 * checked here. The wrapper owns alignment once for all four lanes, so each lane's AEC
 * consumes an already-aligned reference. */
static int derive_dims_and_configs(
    const FourAecNrResConfig* cfg,
    AecConfig* aec_cfg,
    int* fft_size,
    int* hop_size,
    int* n_freqs,
    int* delay_ring_size,
    size_t* delay_estimator_bytes) {
    SeRules rules;
    SeRequest req;
    SeGrid grid;
    int max_delay_samples;
    if (!cfg || !aec_cfg || !fft_size || !hop_size ||
        !n_freqs || !delay_ring_size ||
        !delay_estimator_bytes) return 0;

    if (!se_flag_valid(cfg->enable_cng) || !se_flag_valid(cfg->legacy_amin) ||
        !se_flag_valid(cfg->enable_post) || !se_flag_valid(cfg->enable_nr) ||
        !se_flag_valid(cfg->enable_res)) return 0;

    rules.allow_8k = 0;
    rules.fixed_rate = 0;
    rules.fixed_fft = 0;
    rules.check_delay = 1;
    rules.lane_aligned = 1;
    /* A front-only core has no post stage, so lanes keep only the linear half of the
     * context; with a post stage the lanes' own G_res is still never read. */
    rules.spatial_context =
        cfg->enable_post ? AEC_CONTEXT_NO_GAIN : AEC_CONTEXT_LINEAR_ONLY;
    rules.with_nr = 1;

    req.sample_rate = cfg->sample_rate;
    req.fft_size = cfg->fft_size;
    req.filter_length = cfg->filter_length;
    req.delay_mode = cfg->delay_mode;
    req.delay_num_filters = cfg->delay_num_filters;
    req.fixed_delay_samples = cfg->fixed_delay_samples;
    req.aec_preset = cfg->aec_preset;
    req.nr_mode = cfg->nr_mode;
    if (se_derive(&req, &rules, &grid, aec_cfg) != 0) return 0;

    if (cfg->capture_proxy_channel < 0 ||
        cfg->capture_proxy_channel >= FOUR_AEC_NR_RES_CHANNELS) return 0;
    /* Range only: these knobs do not affect the carve and are inert outside MATCHED; the window is range-checked and NaN-rejected even while the enable is 0. */
    if (cfg->delay_backward_quarantine_enabled != 0 &&
        cfg->delay_backward_quarantine_enabled != 1) return 0;
    if (!isfinite(cfg->delay_backward_quarantine_s) ||
        cfg->delay_backward_quarantine_s < 0.0f ||
        cfg->delay_backward_quarantine_s > 3600.0f) return 0;
    if (!isfinite(cfg->delay_est_period_s) ||
        cfg->delay_est_period_s < 0.0f ||
        cfg->delay_est_period_s > 3600.0f) return 0;
    /* This core's duty-cycle period overwrites whatever lib/aec's preset defaulted
     * (the lanes run with enable_delay_est=0 and never read it). */
    aec_cfg->delay_est_period_s = cfg->delay_est_period_s;
    if (!isfinite(cfg->max_delay_ms) ||
        cfg->max_delay_ms < 0.0f || cfg->max_delay_ms > 4096.0f) return 0;
    *fft_size = grid.fft_size;
    *hop_size = grid.hop_size;
    *n_freqs = grid.n_freqs;

    *delay_estimator_bytes = 0;
    if (cfg->delay_mode == AEC_DELAY_MATCHED) {
        max_delay_samples =
            (int)ceilf(cfg->max_delay_ms * (float)cfg->sample_rate / 1000.0f);
        *delay_ring_size = max_delay_samples + 2 * *hop_size + 1;
        *delay_estimator_bytes = delay_aec_get_mem_size(
            cfg->sample_rate, *hop_size, cfg->delay_num_filters);
        if (*delay_ring_size <= 0 || *delay_estimator_bytes == 0) return 0;
    } else if (cfg->delay_mode == AEC_DELAY_FIXED) {
        *delay_ring_size = cfg->fixed_delay_samples + *hop_size;
        if (*delay_ring_size <= 0) return 0;
    } else {
        *delay_ring_size = 0;
    }

    return 1;
}

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

static int carve_working_buffers(FourAecNrRes* p,
                                 PoolCursor* cursor) {
    int n = p->n_freqs;
    int hop = p->hop_size;

    p->mic_lane =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    p->aligned_ref =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));

    if (!p->cfg.enable_post)
        return p->mic_lane && p->aligned_ref;

    p->fused_error =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
    p->fused_near =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
    p->residual_work =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));
    p->residual_proj =
        (Complex*)pool_carve(cursor, (size_t)n, sizeof(Complex));

    p->fused_r2 =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));
    p->fused_comfort =
        (float*)pool_carve(cursor, (size_t)n, sizeof(float));

    return p->mic_lane && p->aligned_ref &&
           p->fused_error && p->fused_near &&
           p->residual_work && p->residual_proj &&
           p->fused_r2 && p->fused_comfort;
}

static size_t pipeline_buffer_size(
    int hop, int n, int delay_ring_size,
    size_t delay_estimator_bytes, int enable_post) {
    size_t total = 0;
    int i;

    for (i = 0; i < 2; ++i)
        total = ck_field_size(total, (size_t)hop, sizeof(float));

    if (enable_post) {
        for (i = 0; i < 4; ++i)
            total = ck_field_size(total, (size_t)n, sizeof(Complex));
        for (i = 0; i < 2; ++i)
            total = ck_field_size(total, (size_t)n, sizeof(float));
    }
    if (delay_estimator_bytes > 0)
        total = ck_add_size(total, ck_align16_size(delay_estimator_bytes));
    if (delay_ring_size > 0)
        total = ck_field_size(
            total, (size_t)delay_ring_size, sizeof(float));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

static size_t pipeline_pool_size(
    const AecConfig* aec_cfg,
    int hop,
    int n,
    int delay_ring_size,
    size_t delay_estimator_bytes,
    int enable_post) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    size_t buffer_bytes = pipeline_buffer_size(
        hop, n, delay_ring_size, delay_estimator_bytes, enable_post);
    size_t total = 0;
    int ch;

    if (aec_bytes == 0 || buffer_bytes == 0) return 0;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        total = ck_add_size(total, ck_align16_size(aec_bytes));
    total = ck_add_size(total, buffer_bytes);
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

static int pipeline_build(
    FourAecNrRes* p,
    PoolCursor* cursor,
    const AecConfig* aec_cfg) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    int ch;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        void* lane_pool = pool_carve(cursor, 1, aec_bytes);
        if (!lane_pool) return 0;
        p->lanes[ch] = aec_init(lane_pool, aec_bytes, aec_cfg);
        if (!p->lanes[ch]) return 0;
        p->initialized_lanes += 1;
    }

    if (!carve_working_buffers(p, cursor)) return 0;
    if (p->cfg.delay_mode == AEC_DELAY_MATCHED) {
        size_t delay_bytes = delay_aec_get_mem_size(
            p->sample_rate, p->hop_size, p->cfg.delay_num_filters);
        void* delay_pool = pool_carve(cursor, 1, delay_bytes);
        if (!delay_pool || delay_aec_init(
                &p->shared_delay, delay_pool, delay_bytes,
                p->sample_rate, p->hop_size,
                p->cfg.delay_num_filters) != 0) return 0;
    }
    if (p->delay_ring_size > 0) {
        p->delay_ring = (float*)pool_carve(
            cursor, (size_t)p->delay_ring_size, sizeof(float));
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

/* lane i's tag in the hashed carve text; do not alter (feeds build_flags_hash). */
#define LANE_TAG(i) ",aec" #i

static uint32_t four_aec_nr_res_build_flags_hash(void) {
    uint32_t hash = 2166136261u;
    hash = fnv1a_str(AUDIO_PIPELINE_BACKEND_STR, hash);
    hash = fnv1a_str(
        "|carve:self" LANE_TAG(0) LANE_TAG(1) LANE_TAG(2) LANE_TAG(3) ",hop2,"
        "fuse?(complex4,float2),"
        "lanebind,delayest?,delayring?",
        hash);
    hash = fnv1a_str("|align16", hash);
    return ac_memreq_abi_hash(hash);
}

FourAecNrResConfig four_aec_nr_res_default_config(int sample_rate) {
    FourAecNrResConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = 0;
    cfg.filter_length = 0;
    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = DA_NUM_FILTERS;
    cfg.fixed_delay_samples = -1;
    cfg.capture_proxy_channel = 0;
    cfg.delay_backward_quarantine_enabled = 0;
    cfg.delay_backward_quarantine_s = 1.0f;
    cfg.delay_est_period_s = 0.0f;   /* full rate; see 4aec_nr_res.h */
    cfg.max_delay_ms = 1024.0f;
    cfg.aec_preset = AEC_PRESET_BALANCED;
    cfg.nr_mode = ANR_MODE_BALANCED;
    cfg.enable_post = 1;
    cfg.enable_res = 1;
    cfg.enable_nr = 1;
    cfg.enable_cng = 1;
    cfg.legacy_amin = 0;
    return cfg;
}

int four_aec_nr_res_get_mem_requirements(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemReq* out) {
    AecConfig aec_cfg;
    size_t pool_bytes;
    size_t total_bytes;
    uint32_t backend;
    int fft;
    int hop;
    int n;
    int delay_ring_size;
    size_t delay_estimator_bytes;

    if (!out || !derive_dims_and_configs(
            cfg, &aec_cfg, &fft, &hop, &n,
            &delay_ring_size,
            &delay_estimator_bytes)) return -1;

    pool_bytes = pipeline_pool_size(
        &aec_cfg, hop, n, delay_ring_size,
        delay_estimator_bytes, cfg->enable_post);
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
    int fft;
    int hop;
    int n;
    int delay_ring_size;
    size_t delay_estimator_bytes;
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
            &cfg_copy, &aec_cfg, &fft, &hop, &n,
            &delay_ring_size,
            &delay_estimator_bytes)) return NULL;
    (void)delay_estimator_bytes;  /* pipeline_build re-queries the same tuple */

    memset(mem, 0, (size_t)current.bytes);
    p = (FourAecNrRes*)mem;
    p->cfg = cfg_copy;
    p->sample_rate = cfg_copy.sample_rate;
    p->fft_size = fft;
    p->hop_size = hop;
    p->n_freqs = n;
    p->delay_ring_size = delay_ring_size;
    /* Quarantine window: seconds -> hops once, against the resolved grid, with the same
     * floor of 1 as lib/aec's aec_carve(). */
    {
        float q_hop_s = (float)hop / (float)cfg_copy.sample_rate;
        int q_hops = (int)lrintf(cfg_copy.delay_backward_quarantine_s / q_hop_s);
        p->delay_quarantine_hops = q_hops < 1 ? 1 : q_hops;
        p->delay_quarantine_left = -1;   /* disarmed */
        /* Duty-cycle windows from the lane config built above (the same two fields lib/aec
         * converts), converted once because the grid is fixed after init. Floor of 1, not 2: a
         * floor of 2 would cap delay_est_period_s = 0.0 at 50 % engagement instead of full
         * rate. */
        {
            int hold = (int)lrintf(aec_cfg.delay_est_init_s / q_hop_s);
            int period = (int)lrintf(aec_cfg.delay_est_period_s / q_hop_s) /
                         DUTY_PERIOD_DIVISOR;
            p->duty_hold_hops = hold < 1 ? 1 : hold;
            p->duty_period_hops = period < 1 ? 1 : period;
            p->duty_erle_leak_db =
                duty_leak_db_for_grid(hop, cfg_copy.sample_rate);
        }
        p->duty_last_delay = -1;   /* the rest is zeroed by the pool memset */
    }
    p->pool_size = (size_t)current.bytes;

    cursor.ptr = (uint8_t*)mem + ALIGN16(sizeof(*p));
    cursor.remaining =
        (size_t)current.bytes - ALIGN16(sizeof(*p));
    if (!pipeline_build(p, &cursor, &aec_cfg) || cursor.remaining != 0)
        return NULL;

    for (k = 0; k < FOUR_AEC_NR_RES_CHANNELS; ++k) {
        AecResContext context;
        aec_get_res_context(p->lanes[k], &context);
        if (context.hop_size != hop || context.n_freqs != n)
            return NULL;
    }
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

/* Cross-lane far-end agreement, the fuse stage's precondition: a lane with a different far spectrum must not be folded into the beam. Python reference: 4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts. When the borrow chain is shown to have run this hop (lane 0 computes the one far FFT, lanes 1-3 borrow it), four counter reads replace 4*n_freqs compares; without that evidence the full comparison runs. fuse_contexts() skips the predicate on a mixed warm/soft-realign hop. */
int four_aec_nr_res_far_spec_agrees(int shared_far_provenance,
                                    const Complex* lane_far_spec,
                                    const Complex* reference_far_spec,
                                    int n_freqs) {
    int k;
    if (!lane_far_spec || !reference_far_spec || n_freqs < 0) return 0;
    if (shared_far_provenance) return 1;
    for (k = 0; k < n_freqs; ++k) {
        if (!complex_close(lane_far_spec[k], reference_far_spec[k])) return 0;
    }
    return 1;
}

void four_aec_nr_res_admission_age(FourAecDelayAdmission* admission) {
    if (!admission || admission->ttl <= 0) return;
    admission->ttl -= 1;
    if (admission->ttl <= 0) {
        admission->ttl = 0;
        admission->candidate = 0;
    }
}

int four_aec_nr_res_admission_offer(
    FourAecDelayAdmission* admission, int accepted_delay, int estimated) {
    if (!admission) return 0;
    /* Too small to be worth four IR shifts: absorbed and left to age (as lib/aec's Path B). */
    if (abs(estimated - accepted_delay) <= FOUR_DELAY_CHANGE_MIN_SAMPLES)
        return 0;
    if (admission->ttl > 0 &&
        abs(estimated - admission->candidate) <
            FOUR_DELAY_CHANGE_CONFIRM_SAMPLES) {
        admission->candidate = 0;
        admission->ttl = 0;
        return 1;
    }
    admission->candidate = estimated;
    admission->ttl = FOUR_DELAY_CHANGE_CANDIDATE_TTL;
    return 0;
}

/* Duty-machine watchdog signal: the best windowed ERLE of the four lanes (one quiet
 * lane is not evidence that the shared alignment collapsed). Both readings are last
 * hop's; lanes report 0 until their ERLE machinery has run, which leaves the watchdog
 * inert. */
static float max_lane_erle_windowed(const FourAecNrRes* p) {
    float best = 0.0f;
    int ch;
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        AecDebugStatus status;
        if (!p->lanes[ch]) continue;
        aec_debug_status(p->lanes[ch], &status);
        if (status.erle_windowed_db > best) best = status.erle_windowed_db;
    }
    return best;
}

static FourAecNrResDelayState update_shared_delay(
    FourAecNrRes* p,
    const float* capture,
    const float* render) {
    FourAecNrResDelayState state;
    int emitted;
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
        /* update_shared_delay() runs immediately before align_render(): the current hop is
         * readable when the samples already stored cover the requested delay (as lib/aec's
         * `filled >= delay + hop`), so LOCKED is reported on the hop the audio switches. */
        state.solid = p->delay_samples_seen >=
            (uint64_t)p->cfg.fixed_delay_samples;
        p->accepted_delay = p->cfg.fixed_delay_samples;
        p->last_delay = state;
        return state;
    }

    /* Raw hops: DelayAec runs at the native sample_rate and decimates internally, returning the delay in the native domain. Duty-cycled analysis as in lib/aec: the estimator is fed every hop (decimators and render ring stay gapless) and only the matched-filter analysis is decimated once the estimate has held still. A mid-stream movement is seen at the next scheduled analysis, so exposure is bounded by duty_period_hops (1 at the default 0.0 period; ~96 ms on all shipped grids at delay_est_period_s = 0.5). The ERLE watchdog can shorten it once a lane's windowed ERLE has cleared DUTY_ERLE_COLLAPSE_DB. */
    run_filter = 1;
    if (p->duty_active) {
        p->duty_pos += 1;
        if (p->duty_pos >= p->duty_period_hops) p->duty_pos = 0;
        run_filter = (p->duty_pos == 0);
    }
    if (run_filter) p->duty_hops_run += 1;
    emitted = delay_aec_accumulate_ex(&p->shared_delay, capture, render,
                                       p->hop_size, run_filter);
    (void)emitted;
    p->delay_calls += 1;
    {
        int cur = delay_aec_estimated_delay(&p->shared_delay);
        int solid = delay_aec_is_solid(&p->shared_delay);
        if (!solid || cur < 0 || cur != p->duty_last_delay) {
            /* Estimate moved or lost confidence: full rate, hold restarts. */
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
        /* Cancellation watchdog: a leaky peak of the lanes' windowed ERLE; a drop of
         * DUTY_ERLE_COLLAPSE_DB from it resumes full-rate analysis. Armed only once the peak
         * clears that figure. */
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

    estimated = delay_aec_estimated_delay(&p->shared_delay);
    eligible = estimated >= 0 &&
               delay_aec_is_solid(&p->shared_delay) &&
               delay_aec_n_updates(&p->shared_delay) >= 3;

    /* Age the held candidate here, ahead of the quarantine and admission (as lib/aec ages its
     * pending delay). */
    four_aec_nr_res_admission_age(&p->delay_admission);

    /* Backward-jump quarantine (see delay_backward_quarantine_enabled): only for an estimate strictly earlier than the one in force and only once a usable generation exists. It judges cfg.capture_proxy_channel, read before this hop's aec_process_context() (last hop's cancellation). Armed once per qualifying backward episode, one tick per hop, adopted at 0; a forward/ineligible estimate or lost cancellation disarms it. */
    if (eligible && p->cfg.delay_backward_quarantine_enabled &&
        p->last_delay.solid && estimated < p->accepted_delay &&
        p->lanes[p->cfg.capture_proxy_channel] &&
        aec_linear_is_cancelling(p->lanes[p->cfg.capture_proxy_channel])) {
        if (p->delay_quarantine_left < 0)
            p->delay_quarantine_left = p->delay_quarantine_hops;
        if (p->delay_quarantine_left > 0) {
            p->delay_quarantine_left--;
            eligible = 0;
        }
    } else {
        p->delay_quarantine_left = -1;
    }

    /* Published `solid` = a usable accepted alignment generation exists, derived from the acceptance test that writes accepted_delay; it is sticky (a short confidence dip keeps the delay, as in lib/aec) and p->last_delay is zeroed at init and reset().
     * `changed` = this hop starts a new usable generation (a value-only comparison would miss an acquisition or relock at delay 0). A change to a usable alignment must also clear the admission in 4aec_nr_res_internal.h, since each costs an IR shift on all four lanes; a sustained wrong estimate confirms itself and is left to the backward quarantine. First acquisition is immediate (lib/aec's Path A). accepted_delay moves only on acceptance, so the served alignment and the lanes' realign land on the same hop. */
    was_usable = p->last_delay.solid;
    now_usable = was_usable || eligible;
    accepted = 0;
    if (eligible) {
        if (!was_usable) {
            accepted = 1;
            p->delay_admission.candidate = 0;
            p->delay_admission.ttl = 0;
        } else {
            accepted = four_aec_nr_res_admission_offer(
                &p->delay_admission, p->accepted_delay, estimated);
        }
    }
    state.changed = accepted;
    if (accepted) p->accepted_delay = estimated;
    state.delay_samples = p->accepted_delay;
    state.confidence = delay_aec_confidence(&p->shared_delay);
    state.solid = now_usable;
    state.estimator_calls = p->delay_calls;
    state.estimator_updates = delay_aec_n_updates(&p->shared_delay);
    p->last_delay = state;
    return state;
}

static int align_render(FourAecNrRes* p, const float* render,
                             int delay_samples) {
    int hop;
    int ring;
    int write_at;
    int read_at;
    int first;
    uint64_t start;
    if (p->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) {
        memcpy(p->aligned_ref, render,
               (size_t)p->hop_size * sizeof(float));
        p->delay_samples_seen += (uint64_t)p->hop_size;
        return 1;
    }
    if (!p->delay_ring || delay_samples < 0 ||
        delay_samples > p->delay_ring_size - p->hop_size) return 0;

    hop = p->hop_size;
    ring = p->delay_ring_size;
    start = p->delay_samples_seen;
    /* Ring write of one hop: slots start%ring .. (start+hop-1)%ring wrap at most once, so it is one or two bounded runs with `start` reduced once (no per-sample 64-bit divide). */
    write_at = (int)(start % (uint64_t)ring);
    first = ring - write_at;
    if (first > hop) first = hop;
    memcpy(p->delay_ring + write_at, render, (size_t)first * sizeof(float));
    if (first < hop) {
        memcpy(p->delay_ring, render + first,
               (size_t)(hop - first) * sizeof(float));
    }
    /* Whole-hop decision on the same predicate as `solid` under FIXED: the ring serves this offset only once the samples seen cover delay_samples; until then the far content is the raw render hop (lib/aec's rule), not silence. Per hop, so raw and aligned audio are never spliced under UNLOCKED. */
    if (start >= (uint64_t)delay_samples) {
        /* (start - delay_samples) mod ring without a second 64-bit divide: write_at -
         * delay_samples + ring lies in (0, 2*ring), so one conditional subtract reduces it. */
        read_at = write_at - delay_samples;
        if (read_at < 0) read_at += ring;
        first = ring - read_at;
        if (first > hop) first = hop;
        memcpy(p->aligned_ref, p->delay_ring + read_at,
               (size_t)first * sizeof(float));
        if (first < hop) {
            memcpy(p->aligned_ref + first, p->delay_ring,
                   (size_t)(hop - first) * sizeof(float));
        }
    } else {
        memcpy(p->aligned_ref, render,
               (size_t)hop * sizeof(float));
    }
    p->delay_samples_seen += (uint64_t)hop;
    return 1;
}

/* Points dst's spectrum/PSD fields directly at src's buffers instead of copying. AecResContext's pointers alias the AEC's per-hop buffers (aec.h), a lane runs again only inside four_aec_front_process() or after a reset, and the front view is valid until then. reset does not clear this memory; fuse refuses to run on a stale view (front_valid). */
static int bind_lane_view(FourAecLaneSnapshot* dst,
                          const AecResContext* src,
                          int n_freqs, int residual_half) {
    if (!dst || !src || src->n_freqs != n_freqs ||
        !src->error_spec || !src->echo_spec || !src->far_spec ||
        !src->near_spec) return 0;
    /* R^2 and comfort noise exist only for a core that has a post stage. */
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

int four_aec_front_process(
    FourAecNrRes* p,
    const float* microphones_interleaved,
    const float* ref,
    FourAecFront* out) {
    FourAecNrResDelayState delay;
    int hop;
    int ch;
    int i;
    int old_align;
    uint32_t t0;
    const Complex* shared_far_spec = NULL;
    long far_fft_before[FOUR_AEC_NR_RES_CHANNELS];
    int lanes_borrowed;

    if (!p || p->destroyed ||
        !microphones_interleaved || !ref || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    /* Cleared once the call is accepted, so a hop that bails out reports zero for stages it never reached; lane-side fields accumulate over the four lanes. The post half is cleared in four_aec_fuse(). */
    p->last_timing.delay_us = 0;
    p->last_timing.frontend_us = 0;
    p->last_timing.linear_us = 0;
    p->last_timing.lane_res_us = 0;
    p->far_spec_shared_hop = 0;
    p->far_spec_mixed_hop = 0;
    p->front_valid = 0;

    hop = p->hop_size;
    if (!inputs_finite(
            microphones_interleaved,
            (size_t)hop * FOUR_AEC_NR_RES_CHANNELS) ||
        !inputs_finite(ref, (size_t)hop))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    /* Only lane 0 with proxy 0 can reuse this extraction: earlier lanes overwrite
     * mic_lane before a later proxy lane runs. */
    for (i = 0; i < hop; ++i) {
        p->mic_lane[i] =
            microphones_interleaved[
                i * FOUR_AEC_NR_RES_CHANNELS +
                p->cfg.capture_proxy_channel];
    }
    /* The alignment the lanes were served before this hop's decision (raw 0 until the
     * estimate first turns solid), captured before update_shared_delay() overwrites it. */
    old_align = p->last_delay.solid ? p->accepted_delay : 0;
    t0 = four_aec_nr_res_now_us();
    delay = update_shared_delay(p, p->mic_lane, ref);
    p->last_timing.delay_us = four_aec_nr_res_now_us() - t0;
    if (!align_render(p, ref, delay.delay_samples)) {
        four_aec_nr_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (delay.changed) {
        /* Realign instead of reset: aec_apply_external_realign() shifts each converged filter by
         * the alignment delta (warm tap transfer when the evidence gate holds, a filter-only
         * reset otherwise), so cancellation survives and the WOLA sequences continue. */
        int hop_warm = 0;
        int hop_soft = 0;
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            int outcome = aec_apply_external_realign(
                p->lanes[ch], delay.delay_samples - old_align);
            if (outcome == 1) {
                p->realign_warm_lanes = four_aec_count_add_sat(
                    p->realign_warm_lanes, 1);
                hop_warm += 1;
            } else if (outcome == 0) {
                p->realign_soft_lanes = four_aec_count_add_sat(
                    p->realign_soft_lanes, 1);
                hop_soft += 1;
            }
        }
        /* Some lanes warm and some soft on the same hop: a soft outcome zeroes that lane's far_buffer OLA history, so the lanes are out of lockstep. The loop stops borrowing lane 0's far FFT for this hop and fuse_contexts() stops requiring the far spectra to agree. */
        p->far_spec_mixed_hop = (hop_warm > 0 && hop_soft > 0);
    }

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        far_fft_before[ch] = aec_far_fft_real_compute_count(p->lanes[ch]);
    lanes_borrowed = 1;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        AecResContext context;
        if (ch != 0 || p->cfg.capture_proxy_channel != 0) {
            for (i = 0; i < hop; ++i) {
                p->mic_lane[i] =
                    microphones_interleaved[
                        i * FOUR_AEC_NR_RES_CHANNELS + ch];
            }
        }
        /* Lane 0 produces this hop's far spectrum and magnitude; lanes 1-3 copy them after their
         * prior-hop RSA read. On a mixed warm/soft hop all lanes compute locally. */
        if (ch == 0 || p->far_spec_mixed_hop) {
            aec_process_context(p->lanes[ch], p->mic_lane, p->aligned_ref);
        } else {
            aec_process_context_shared_far_from(
                p->lanes[ch], p->mic_lane, p->aligned_ref, p->lanes[0]);
        }
        aec_get_res_context(p->lanes[ch], &context);
        if (ch == 0) shared_far_spec = context.far_spec;
        if (!context.formed_hop || !bind_lane_view(
                &p->snapshots[ch], &context, p->n_freqs,
                p->cfg.enable_post)) {
            four_aec_nr_res_reset(p);
            return FOUR_AEC_NR_RES_DSP_ERROR;
        }
        {
            AecStageTiming lane_time;
            aec_get_last_timing(p->lanes[ch], &lane_time);
            /* Lane-internal delay work is structurally zero (every lane is EXTERNAL_ALIGNED) and is summed into delay_us anyway. */
            p->last_timing.delay_us    += lane_time.delay_us;
            p->last_timing.frontend_us += lane_time.frontend_us;
            p->last_timing.linear_us   += lane_time.linear_us;
            p->last_timing.lane_res_us += lane_time.res_us;
        }
        /* Borrow-chain signature: the lane that computes the shared spectrum runs one far transform this hop, a borrowing lane none; anything else leaves 0 and sends fuse_contexts() to the per-bin comparison. */
        {
            long after = aec_far_fft_real_compute_count(p->lanes[ch]);
            if (!four_aec_far_fft_delta_matches(
                    far_fft_before[ch], after, ch == 0 ? 1 : 0))
                lanes_borrowed = 0;
        }
    }
    p->far_spec_shared_hop = lanes_borrowed && shared_far_spec != NULL;

    p->front_valid = 1;

    out->delay = delay;
    out->hop_size = p->hop_size;
    out->n_channels = FOUR_AEC_NR_RES_CHANNELS;
    out->n_freqs = p->n_freqs;
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        out->linear_spectra[ch] = p->snapshots[ch].error_spec;
        out->formed_hop[ch] = p->snapshots[ch].formed_hop;
    }
    out->aligned_ref = p->aligned_ref;
    return FOUR_AEC_NR_RES_OK;
}

static int validate_weights(const FourAecNrRes* p,
                                 const Complex* weights) {
    if (!p || !weights) return 0;
    return four_aec_weights_valid(
        weights,
        (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)p->n_freqs);
}

static int complex_vector_finite(const Complex* values, int count) {
    if (!values || count < 0) return 0;
    return skn_all_finite_cf32(values, (size_t)count);
}

static int fuse_contexts(FourAecNrRes* p,
                         const Complex* weights,
                         const Complex* trusted_beamformed_error,
                         int* all_converged,
                         int* usable_linear,
                         float* max_dt,
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
        /* No copy into p->fused_error: the caller owns it for the whole call. */
    } else {
        memset(p->fused_error, 0, (size_t)n * sizeof(Complex));
    }
    memset(p->fused_near, 0, (size_t)n * sizeof(Complex));
    memset(p->residual_proj, 0, (size_t)n * sizeof(Complex));
    memset(p->fused_comfort, 0, (size_t)n * sizeof(float));

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        const Complex* lane_weights =
            weights + (size_t)ch * (size_t)n;
        /* A lane the beamformer zeroed in every bin contributes nothing, so it must not vote on the scalars that steer the post stage (with the beamformer bypassed the weights are [1,0,0,0] and lanes 1-3 would still drive max_dt). */
        int lane_contributes = 0;
        for (k = 0; k < n; ++k) {
            if (lane_weights[k].r != 0.0f || lane_weights[k].i != 0.0f) {
                lane_contributes = 1;
                break;
            }
        }
        float delta = fabsf(
            p->snapshots[ch].far_power - base_far_power);
        float scale = 1.0f + fabsf(base_far_power);
        if (delta > 1e-5f * scale) return 0;
        if (lane_contributes) {
            if (!p->snapshots[ch].filter_converged)
                *all_converged = 0;
            if (!p->snapshots[ch].usable_linear)
                *usable_linear = 0;
            if (p->snapshots[ch].dt_indicator > *max_dt)
                *max_dt = p->snapshots[ch].dt_indicator;
            if (p->snapshots[ch].saturation_level > *max_saturation)
                *max_saturation = p->snapshots[ch].saturation_level;
        }

        /* A mixed warm/soft-realign hop skips this predicate. The fused X takes lane 0's far_spec either way. Python reference: 4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts. */
        if (!p->far_spec_mixed_hop && !four_aec_nr_res_far_spec_agrees(
                p->far_spec_shared_hop,
                p->snapshots[ch].far_spec,
                p->snapshots[0].far_spec, n)) return 0;

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
            p->residual_proj, lane_weights, p->residual_work, n);
        four_aec_comfort_accumulate(
            p->fused_comfort, lane_weights,
            p->snapshots[ch].comfort_noise, n);
    }

    four_aec_complex_mag2(p->fused_r2, p->residual_proj, n);
    return 1;
}

int four_aec_fuse(
    FourAecNrRes* p,
    const Complex* weights,
    const Complex* beamformed_error,
    SeSpecFrame* frame) {
    int all_converged;
    int usable_linear;
    float max_dt;
    float max_saturation;
    uint32_t t0;

    if (!p || p->destroyed || !weights || !frame || !p->cfg.enable_post)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (!p->front_valid) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    if (!validate_weights(p, weights))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    p->last_timing.fuse_us = 0;
    t0 = four_aec_nr_res_now_us();
    if (!fuse_contexts(
            p, weights, beamformed_error,
            &all_converged, &usable_linear, &max_dt, &max_saturation)) {
        four_aec_nr_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    p->last_timing.fuse_us = four_aec_nr_res_now_us() - t0;

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
    frame->formed_hop = NULL;    /* one hop per lane: FourAecFront.formed_hop */
    frame->far_power = p->snapshots[0].far_power;
    frame->max_dt = max_dt;
    frame->max_sat = max_saturation;
    frame->all_converged = all_converged;
    frame->usable_linear = usable_linear;
    frame->clock_drift = p->cfg.delay_mode == AEC_DELAY_MATCHED
        ? delay_aec_has_clockdrift(&p->shared_delay) : 0;
    frame->delay_changed = p->last_delay.changed;
    frame->delay_locked = p->last_delay.solid;
    return FOUR_AEC_NR_RES_OK;
}

int four_aec_nr_res_stage_configs(
    const FourAecNrResConfig* cfg,
    AecConfig* res_cfg,
    AnrConfig* nr_cfg) {
    int fft, hop, n, delay_ring_size;
    size_t delay_estimator_bytes;
    if (!res_cfg || !nr_cfg || !derive_dims_and_configs(
            cfg, res_cfg, &fft, &hop, &n, &delay_ring_size,
            &delay_estimator_bytes)) return -1;
    if (cfg->enable_nr)
        *nr_cfg = se_anr_compose_config(cfg->sample_rate, fft, hop, cfg->nr_mode);
    else
        memset(nr_cfg, 0, sizeof(*nr_cfg));
    return 0;
}

/* Legal at any time. A caller holding a FourAecFront must stop reading its pointers: they
 * alias lane buffers that aec_reset() does not clear (stale content, not a fault).
 * four_aec_fuse() refuses to run until a new front frame is processed. */
void four_aec_nr_res_reset(FourAecNrRes* p) {
    int ch;
    if (!p || p->destroyed) return;

    if (p->cfg.delay_mode == AEC_DELAY_MATCHED)
        delay_aec_reset(&p->shared_delay);
    for (ch = 0; ch < p->initialized_lanes; ++ch) {
        aec_reset(p->lanes[ch]);
    }
    if (p->delay_ring) {
        memset(p->delay_ring, 0,
               (size_t)p->delay_ring_size * sizeof(float));
    }
    if (p->aligned_ref) {
        memset(p->aligned_ref, 0,
               (size_t)p->hop_size * sizeof(float));
    }

    p->delay_samples_seen = 0;
    p->accepted_delay =
        p->cfg.delay_mode == AEC_DELAY_FIXED
            ? p->cfg.fixed_delay_samples : 0;
    /* A reset abandons the alignment the quarantine was protecting, so the countdown and any
     * held change candidate (with its life) are cleared; the config-derived window stays. */
    p->delay_quarantine_left = -1;
    p->delay_admission.candidate = 0;
    p->delay_admission.ttl = 0;
    p->realign_warm_lanes = 0;
    p->realign_soft_lanes = 0;
    p->delay_calls = 0;
    p->far_spec_shared_hop = 0;
    p->far_spec_mixed_hop = 0;
    /* The duty machine measured the alignment this reset abandoned: full-rate analysis, no hold, no peak, fresh census; the grid-derived windows stay. */
    p->duty_active = 0;
    p->duty_stable_hops = 0;
    p->duty_pos = 0;
    p->duty_last_delay = -1;
    p->duty_erle_peak = 0.0f;
    p->duty_hops_run = 0;
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->front_valid = 0;
}

void four_aec_nr_res_destroy(FourAecNrRes* p) {
    int ch;
    void* owned_heap;
    if (!p || p->destroyed) return;

    for (ch = 0; ch < p->initialized_lanes; ++ch) {
        aec_destroy(p->lanes[ch]);
    }
    owned_heap = p->owned_heap;
    p->destroyed = 1;
    if (owned_heap) free(owned_heap);
}

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
    return p && !p->destroyed &&
           p->cfg.delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
}

int four_aec_nr_res_linear_aec_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? FOUR_AEC_NR_RES_CHANNELS : 0;
}

long four_aec_nr_res_far_fft_real_compute_count(const FourAecNrRes* p) {
    long total = 0;
    int ch;
    if (!p || p->destroyed) return 0;
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        total = four_aec_count_add_sat(
            total, aec_far_fft_real_compute_count(p->lanes[ch]));
    return total;
}

long four_aec_nr_res_realign_warm_lane_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->realign_warm_lanes : 0;
}

long four_aec_nr_res_realign_soft_lane_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->realign_soft_lanes : 0;
}

unsigned long long four_aec_nr_res_duty_hops_total(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->delay_calls : 0ull;
}

unsigned long long four_aec_nr_res_duty_hops_run(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->duty_hops_run : 0ull;
}

int four_aec_nr_res_far_spec_provenance(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->far_spec_shared_hop : 0;
}

/* Declared in 4aec_nr_res_internal.h. A period of one hop is "analyse every hop" at the decision site, while arming, hold and watchdog keep running and stay measurable through the census. A no-op at the default 0.0 period. */
void four_aec_nr_res_pin_duty_full_rate(FourAecNrRes* p) {
    if (!p || p->destroyed) return;
    p->duty_period_hops = 1;
}

float four_aec_nr_res_duty_leak_db_for_grid(int hop_size, int sample_rate) {
    return duty_leak_db_for_grid(hop_size, sample_rate);
}

float four_aec_nr_res_duty_erle_leak_db(const FourAecNrRes* p) {
    if (!p || p->destroyed) return 0.0f;
    return p->duty_erle_leak_db;
}

float four_aec_nr_res_duty_erle_peak(const FourAecNrRes* p) {
    if (!p || p->destroyed) return 0.0f;
    return p->duty_erle_peak;
}

int four_aec_nr_res_pending_delay_candidate(const FourAecNrRes* p) {
    if (!p || p->destroyed || p->delay_admission.ttl <= 0) return -1;
    return p->delay_admission.candidate;
}

int four_aec_nr_res_get_mem_breakdown(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemBreakdown* out) {
    AecConfig aec_cfg;
    size_t aec_one;
    size_t aec_total = 0;
    size_t wrapper_storage;
    size_t wrapper_bytes;
    size_t total;
    int fft;
    int hop;
    int n;
    int delay_ring_size;
    size_t delay_estimator_bytes;
    int ch;

    if (!out || !derive_dims_and_configs(
            cfg, &aec_cfg, &fft, &hop, &n,
            &delay_ring_size,
            &delay_estimator_bytes)) return -1;

    aec_one = aec_get_mem_size(&aec_cfg);
    wrapper_storage = pipeline_buffer_size(
        hop, n, delay_ring_size, delay_estimator_bytes, cfg->enable_post);
    if (aec_one == 0 || wrapper_storage == 0) return -1;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        aec_total = ck_add_size(aec_total, ck_align16_size(aec_one));
    wrapper_bytes = ck_add_size(
        ck_align16_size(sizeof(FourAecNrRes)), wrapper_storage);
    total = ck_add_size(wrapper_bytes, aec_total);
    if (MEM_SIZE_INVALID(total)) return -1;

    memset(out, 0, sizeof(*out));
    out->aec_bytes = aec_total;
    out->wrapper_bytes = wrapper_bytes;
    out->total_bytes = total;
    out->hop_size = hop;
    out->fft_size = fft;
    out->n_freqs = n;
    return 0;
}

/* See FourAecNrResLastTiming (4aec_nr_res.h). */
void four_aec_nr_res_get_last_timing(
    const FourAecNrRes* p,
    FourAecNrResLastTiming* out) {
    if (!out) return;
    if (!p || p->destroyed) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = p->last_timing;
}
