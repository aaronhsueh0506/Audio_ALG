/**
 * 4aec_nr_res.c — implementation of 4aec_nr_res.h.
 *
 * Four-channel front end of pipelines/mono_aec_nr_res/audio_pipeline.c:
 *
 *   one shared DelayAec3 -> one aligned render
 *   -> four independent linear AEC filters (four_aec_front_process)
 *   -> externally supplied effective beamformer weights
 *   -> the coherent projection of the four lane contexts (four_aec_fuse)
 *
 * The residual-echo, noise-reduction, comfort-noise and synthesis stages are
 * not here: the application carves them and chains them over the frame the
 * fuse returns (pipelines/common/se_stage.h).
 *
 * Same pool-first layout and lifecycle as AudioPipeline:
 *
 *   four_aec_nr_res_get_mem_requirements() + four_aec_nr_res_init_ex()
 *       caller-owned pool; zero heap from init through destroy
 *
 *   four_aec_nr_res_create()
 *       heap convenience wrapper over that same pool-first implementation
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>

#include "4aec_nr_res.h"
#include "4aec_nr_res_internal.h"
#include "fft_wrapper.h"
#include "mem_align.h"
#include "4aec_projection_kernels.h"
#include "simd_kernel_nn.h"        /* skn_all_finite_cf32 */
#include "common/se_config.h"     /* se_derive(): grid + AecConfig, shared by every application */
#include "common/se_anr_stage.h"  /* se_anr_compose_config()                                     */

/* Matched-filter duty cycle. The hold WINDOW is not a literal here: it is
 * read from the lane AecConfig this file already builds at init --
 * delay_est_init_s, the same field lib/aec reads for its own machine -- so
 * the shared estimator holds for exactly as long as a lane's own estimator
 * would. The period window, delay_est_period_s, is NOT read through from
 * that same AecConfig (see derive_dims_and_configs()): it is this core's
 * cfg.delay_est_period_s instead, this wrapper's own opt-in contract rather
 * than a silent mirror of whatever lib/aec currently defaults to -- the two
 * used to be the same field by construction (commit 07a51b1), and that is
 * exactly what broke when lib/aec's own default moved (AMEND_35,
 * 2026-10-18) out from under this file's unrelated floor. Hold the analysis
 * at full rate until the estimate has been solid and unchanged for
 * delay_est_init_s, then analyse one hop in
 * K = round(delay_est_period_s/hop_s)/DUTY_PERIOD_DIVISOR.
 *
 * The divisor and the hold floor of one hop stay literals for the same
 * reason they are literals in lib/aec: they are the SHAPE of the schedule,
 * not a tuning surface, and neither library exposes them. The period floor
 * is 1, not 2 -- "a period of at least 2 so decimated always means
 * something" was this file's floor before AMEND_35 fixed the identical bug
 * in lib/aec; keeping it here after would silently re-cap this core's own
 * delay_est_period_s=0.0 default at 50% engagement. So does the watchdog's
 * collapse figure: it is a depth in dB and carries no clock.
 *
 * The watchdog's LEAK does carry one, and is the only number here that is not
 * used as written. It is lib/aec's literal, and lib/aec calibrated it on the
 * 10 ms hop its config defaults to: 0.001 dB
 * per hop there means 0.1 dB per SECOND, i.e. 60 s to leak the whole
 * DUTY_ERLE_COLLAPSE_DB threshold. Applied as a per-hop literal it would be
 * 0.125 dB/s on this pipeline's 8 ms grid and 0.0625 dB/s on its 16 ms one --
 * the watchdog would disengage the schedule on a merely quiet stretch at one
 * grid and sit on a stale peak at the other. duty_leak_db_for_grid() converts
 * it once at init, beside the two windows and for the same reason, by the
 * rule an ADDITIVE per-hop amount takes: it scales LINEARLY with the hop
 * period, where a retention would take a power law -- N hops of a leak sum to
 * N*leak, they do not compound. */
#define DUTY_PERIOD_DIVISOR       5
#define DUTY_ERLE_LEAK_DB         0.001f   /* per hop AT DUTY_LEAK_REF_HOP_S */
#define DUTY_LEAK_REF_HOP_S       0.010f
#define DUTY_ERLE_COLLAPSE_DB     6.0f

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

/* Per-stage timing is diagnostic and costs nine clock_gettime() calls per hop
 * here, on top of the five each of the four lanes makes inside lib/aec -- 29
 * in all. DEFAULT OFF for the same reason lib/aec's is: a diagnostic that has
 * to be switched off is one that ships on by accident.
 *
 * The two flags are separate because the two builds are: this one governs the
 * wrapper's own stages, AEC_STAGE_TIMING governs what the lanes report back.
 * Setting only this one leaves frontend/linear/lane_res reading 0 while the
 * wrapper's own stages measure, which is a legible state, not a broken one.
 * `make PROFILE=1` sets both, and is what a profile build should use. */
#ifndef FOUR_AEC_NR_RES_STAGE_TIMING
#define FOUR_AEC_NR_RES_STAGE_TIMING 0
#endif

#if FOUR_AEC_NR_RES_STAGE_TIMING
/* Microsecond monotonic stamp for the per-stage diagnostic timing. Truncated
 * to 32 bits: every consumer subtracts two stamps in UNSIGNED arithmetic, so
 * the difference is exact for any interval shorter than the ~71.6 minute wrap
 * -- which no hop approaches. *
 * CLOCK_MONOTONIC is POSIX, not C99. A target with a reduced libc enables the
 * timing with -DFOUR_AEC_NR_RES_STAGE_TIMING=1 -DFOUR_AEC_NR_RES_NOW_US=board_timer_us, naming a function that takes
 * no argument and returns uint32_t microseconds -- a plain identifier, because
 * these Makefiles reject parentheses in EXTRA_CFLAGS, and its declaration is
 * the integrator's to supply. The default below is then neither compiled nor
 * linked and <time.h> is not included.
 *
 * Each component carries its own override rather than sharing one: lib/aec has
 * AEC_NOW_US and the other pipeline has its own, so a chain built for such a
 * target names a timer per component it actually builds.
 */
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
/* Every stamp folds to a constant, so the subtractions fold to zero and no
 * clock is read. */
#define four_aec_nr_res_now_us() 0u
#endif

/* ============================================================================
 * Instance
 * ========================================================================== */

typedef struct FourAecLaneSnapshot {
    /* Borrowed (not owned) views into a lane's own AecResContext buffers --
     * see bind_lane_view()'s doc comment for the lifetime contract. Kept
     * const to match AecResContext's own field types and because nothing in
     * this file ever writes through these pointers, only through the
     * pointer fields themselves (i.e. re-binding to a new hop's buffers). */
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

    FourAecLaneSnapshot snapshots[FOUR_AEC_NR_RES_CHANNELS];

    float* mic_lane;
    float* aligned_ref;

    float* delay_ring;
    int delay_ring_size;
    uint64_t delay_samples_seen;
    int accepted_delay;
    /* Backward-jump quarantine (cfg.delay_backward_quarantine_*), mirroring
     * lib/aec's own pair: _left is the hops still to spend before a held
     * EARLIER estimate is adopted anyway (-1 = DISARMED, the only state that
     * may re-arm; 0 = armed and EXPIRED, i.e. this hop adopts), _hops is the
     * window converted once at init. */
    int delay_quarantine_left;
    int delay_quarantine_hops;
    /* Two-step admission for a shared-delay CHANGE: the movement seen once
     * and held for confirmation, plus its remaining life in hops (see
     * 4aec_nr_res_internal.h). Cleared by acceptance, by expiry, by reset(). */
    FourAecDelayAdmission delay_admission;
    /* Cumulative aec_apply_external_realign() outcomes across the four lanes,
     * split warm (returned 1, learned IR shifted) vs soft (returned 0). */
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

    /* Per-stage wall-clock timing -- see FourAecNrResLastTiming's doc comment
     * (4aec_nr_res.h) for what each field measures. Stored as the published
     * record itself rather than eight private twins, so the accessor is one
     * assignment and a new stage cannot be added to one side only. Appended
     * at the end, the same precedent as aec.h's Aec additions: every field
     * above keeps its offset. */
    FourAecNrResLastTiming last_timing;

    /* This hop's answer to "did all four lanes consume ONE far spectrum",
     * established from the lanes' own far-FFT counters in
     * four_aec_front_process() and consumed by fuse_contexts() through
     * four_aec_nr_res_far_spec_agrees(). 0 (the value every bail-out path
     * leaves behind) means "not established", which costs the full per-bin
     * comparison rather than skipping it. */
    int far_spec_shared_hop;

    /* Matched-filter duty cycle, the shared estimator's counterpart of
     * lib/aec's own always-on machine (the duty_* fields in aec.h): the
     * estimator is FED every hop and ANALYSED at full rate until the
     * published estimate has been solid and unchanged for duty_hold_hops,
     * then 1-in-duty_period_hops until the estimate is seen to move, seen to
     * lose confidence, or the watchdog below sees cancellation collapse. Seen
     * is the operative word for the first two: only an ANALYSED hop can
     * republish, so those two triggers land at the next scheduled analysis
     * and not before (update_shared_delay() carries the bound).
     *
     * duty_erle_peak is a leaky running peak of the LANES' windowed ERLE
     * (the wrapper's stand-in for the single instance's own reading lib/aec
     * watches -- see update_shared_delay()); the two window figures are
     * converted from seconds once at init, where lib/aec converts them per
     * hop from its config, and hold the same values on the same grid.
     * duty_erle_leak_db is that peak's per-hop decay, converted at init from
     * the same 0.1 dB/s the lib calibrated (see DUTY_ERLE_LEAK_DB); it sits
     * here rather than beside the windows above because the tail padding
     * ahead of the counter absorbs it and the control block does not grow.
     *
     * duty_hops_run is the engagement half of the census, counted at the one
     * site that consumes the decision so it cannot drift from what ran;
     * cumulative since init or the last reset(), like lib/aec's. The census
     * DENOMINATOR is delay_calls -- every analysed hop reaches this block, so
     * a separate total would count the same thing twice. */
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

/* The watchdog leak in this grid's own per-hop units -- see the
 * DUTY_ERLE_LEAK_DB banner for why it is the one duty constant that is
 * converted rather than used as written.
 *
 * Exact at the calibration grid, by construction rather than by rounding:
 * hop 160 at 16 kHz makes hop_s the same float DUTY_LEAK_REF_HOP_S is, so
 * the ratio is exactly 1.0f and the authored 0.001f comes back bit-for-bit.
 * That grid is not one this pipeline builds (its three are 8 / 16 / 10.667 ms),
 * which is why four_aec_nr_res_duty_leak_db_for_grid() exposes this for a
 * test to call at the reference directly. Both stacks pin -ffp-contract=off
 * and reject -ffast-math, so x/x == 1.0f holds. */
static float duty_leak_db_for_grid(int hop_size, int sample_rate) {
    float hop_s;
    if (hop_size <= 0 || sample_rate <= 0) return DUTY_ERLE_LEAK_DB;
    hop_s = (float)hop_size / (float)sample_rate;
    return DUTY_ERLE_LEAK_DB * (hop_s / DUTY_LEAK_REF_HOP_S);
}

/* ============================================================================
 * Config -> module configs + frame dimensions
 * ========================================================================== */

/* Returns 1 and fills every out-param, or 0 (NULL argument; a rate outside
 * {16000, 48000}; a grid, preset, mode, flag, filter length or delay setting
 * outside its range; or a MATCHED/FIXED request whose delay ring or estimator
 * cannot be sized). Grid, AecConfig and the delay-request rules come from
 * se_derive(); what only this core has -- the capture proxy, the quarantine
 * and delay-line ranges, the shared ring and estimator sizes -- is checked
 * here. The wrapper owns alignment once for all four lanes, so each lane's
 * AEC consumes an already-aligned reference and carries neither a private
 * estimator nor a ring. */
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
    /* A front-only core has no post stage, so nothing reads a lane's R^2 or
     * comfort noise: its lanes keep only the linear half of the context. With
     * a post stage the lanes' own G_res is still never read (the gain is
     * computed once from fused multi-lane data), so it is skipped too. */
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
    /* Range only. Unlike delay_num_filters, these are pure policy knobs with
     * no effect on the pool carve, and lib/aec accepts its own
     * delay_backward_quarantine_* in every mode for the same reason -- so a
     * caller that sets them once for a whole product and switches modes at
     * bring-up does not have to unset them. Outside MATCHED nothing
     * re-decides an alignment, so they are simply inert. The window is
     * range-checked (and NaN-rejected) even while the enable is 0: a config
     * that would misbehave the moment someone flips one field must not pass
     * validation. */
    if (cfg->delay_backward_quarantine_enabled != 0 &&
        cfg->delay_backward_quarantine_enabled != 1) return 0;
    if (!isfinite(cfg->delay_backward_quarantine_s) ||
        cfg->delay_backward_quarantine_s < 0.0f ||
        cfg->delay_backward_quarantine_s > 3600.0f) return 0;
    if (!isfinite(cfg->delay_est_period_s) ||
        cfg->delay_est_period_s < 0.0f ||
        cfg->delay_est_period_s > 3600.0f) return 0;
    /* The wrapper's own duty-cycle contract, not lib/aec's: se_derive() ->
     * aec_config_from_preset() -> aec_config_defaults() already set
     * aec_cfg->delay_est_period_s to whatever lib/aec defaults to, but that
     * value is an implementation detail of a library the lanes here run
     * with enable_delay_est=0 (they never read it) -- the shared estimator
     * is the only consumer, and it is this struct's field, authoritatively,
     * so it overwrites rather than defers to lib/aec's default. */
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
        *delay_estimator_bytes = delay_aec3_get_mem_size(
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

    /* p->snapshots[ch]'s Complex- and float-pointer fields are not
     * pool-carved -- bind_lane_view() points them directly at each lane's own
     * AecResContext buffers every hop instead of owning a private copy. See
     * bind_lane_view()'s doc comment for the lifetime contract this depends
     * on. */

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

    /* mic_lane + aligned_ref are always required; the frequency-domain
     * buffers exist only for a core that fuses. */
    for (i = 0; i < 2; ++i)
        total = ck_field_size(total, (size_t)hop, sizeof(float));

    if (enable_post) {
        for (i = 0; i < 4; ++i)
            total = ck_field_size(total, (size_t)n, sizeof(Complex));
        for (i = 0; i < 2; ++i)
            total = ck_field_size(total, (size_t)n, sizeof(float));
        /* No per-lane snapshot buffers here: FourAecLaneSnapshot's
         * spectrum/PSD fields are borrowed pointers into each lane's own
         * AecResContext, not pool-carved copies. See bind_lane_view(). */
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

/* ============================================================================
 * Carve/build — counterpart of audio_pipeline.c's pipeline_build()
 * ========================================================================== */

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
        size_t delay_bytes = delay_aec3_get_mem_size(
            p->sample_rate, p->hop_size, p->cfg.delay_num_filters);
        void* delay_pool = pool_carve(cursor, 1, delay_bytes);
        if (!delay_pool || delay_aec3_init(
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
        "|carve:self,aec0,aec1,aec2,aec3,hop2,"
        "fuse?(complex4,float2),"
        "lanebind,delayest?,delayring?",
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
    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = DA_NUM_FILTERS;
    cfg.fixed_delay_samples = -1;
    cfg.capture_proxy_channel = 0;
    cfg.delay_backward_quarantine_enabled = 0;
    cfg.delay_backward_quarantine_s = 1.0f;
    cfg.delay_est_period_s = 0.0f;   /* full rate; see 4aec_nr_res.h doc */
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
    /* Quarantine window: seconds -> hops ONCE, here, against the resolved
     * grid -- the same conversion (and the same floor of 1) lib/aec does in
     * aec_carve(), so the core and its lanes cannot disagree about how long
     * a window is. */
    {
        float q_hop_s = (float)hop / (float)cfg_copy.sample_rate;
        int q_hops = (int)lrintf(cfg_copy.delay_backward_quarantine_s / q_hop_s);
        p->delay_quarantine_hops = q_hops < 1 ? 1 : q_hops;
        p->delay_quarantine_left = -1;   /* disarmed */
        /* Duty-cycle windows, taken from the lane config built just above --
         * the same two fields lib/aec converts for its own machine -- on the
         * same resolved grid and with the same floors. Converted once here
         * rather than per hop because this grid cannot change after init; the
         * values are the ones lib/aec's per-hop conversion would produce from
         * the same seconds.
         *
         * Floor of 1, not 2: lib/aec's own floor was 2 until AMEND_35
         * (2026-10-18) found it silently capped delay_est_period_s=0.0 at
         * 50% engagement instead of true full-rate, changing WHICH delay the
         * matched filter could ever settle on, not just when -- see aec.c's
         * duty_period_hops init. This core's cfg.delay_est_period_s is its
         * own field (derive_dims_and_configs() overwrites whatever lib/aec's
         * preset defaulted aec_cfg.delay_est_period_s to), defaulting to 0.0
         * for the same reason, so it must floor the same way or it
         * reintroduces exactly the bug AMEND_35 fixed, one layer up. */
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

/* Cross-lane far-end agreement, the fuse stage's precondition: X is ONE
 * shared digital render reference, not four spatial observations, so a lane
 * carrying a DIFFERENT far spectrum must not be folded into the beam (the
 * "linear AEC lanes do not share one far-end spectrum" rejection, mirrored
 * from pipeline.py).
 *
 * Two ways to establish it. When the borrow chain this file drives has been
 * shown to have run this hop -- lane 0 computes the one real far FFT, lanes
 * 1-3 take it through aec_process_context_shared_far() and provably ran no
 * transform of their own -- the lanes hold a memcpy of one buffer and a
 * per-bin value comparison can only restate that. Provenance answers it in
 * four counter reads for the whole hop instead of 4*n_freqs float compares.
 *
 * Without that evidence -- a caller wiring per-lane contexts from somewhere
 * else, or any hop whose counters do not move the way the borrow requires --
 * nothing is known about where a lane's spectrum came from, and the full
 * comparison is the only answer available. Not knowing therefore costs the
 * comparison; it never skips it. */
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

/* ============================================================================
 * Per-hop processing — Stage 1: shared delay + four linear AEC lanes
 * ========================================================================== */

/* The admission band, its confirmation window and the candidate's life are
 * lib/aec's own Path-B numbers; they and the state they act on are declared
 * in 4aec_nr_res_internal.h, where the tests can reach them. */

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
    /* Too small to be worth four IR shifts: absorbed, and deliberately left
     * to age rather than clearing the candidate, so a movement is not
     * cancelled by one hop of the estimator agreeing with the alignment it is
     * moving away from. lib/aec's Path B has no such clear either. */
    if (abs(estimated - accepted_delay) <= FOUR_DELAY_CHANGE_MIN_SAMPLES)
        return 0;
    if (admission->ttl > 0 &&
        abs(estimated - admission->candidate) <
            FOUR_DELAY_CHANGE_CONFIRM_SAMPLES) {
        admission->candidate = 0;
        admission->ttl = 0;
        return 1;
    }
    /* First sighting, or a movement somewhere else entirely: this becomes the
     * candidate with a full life, replacing whatever was held. */
    admission->candidate = estimated;
    admission->ttl = FOUR_DELAY_CHANGE_CANDIDATE_TTL;
    return 0;
}

/* The duty machine's watchdog signal. lib/aec watches the one linear filter
 * it owns; the shared estimator here drives FOUR, so the reading that
 * corresponds to "the alignment this estimate produced is still working" is
 * the BEST of them -- one lane going quiet (a microphone with no echo path
 * of its own to cancel) is not evidence that the shared alignment collapsed,
 * whereas no lane cancelling anywhere is.
 *
 * aec_debug_status() is documented read-only with no hot-path cost: it copies
 * figures the engine already maintains. Both readings are last hop's, which
 * is also what lib/aec's cached last_erle_windowed is at the point its own
 * duty block runs. Lanes report 0 until their ERLE machinery has run, so an
 * unavailable metric leaves the watchdog inert (peak stays below the arming
 * threshold) rather than firing. */
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
        /* update_shared_delay() runs immediately before align_render().
         * The current hop will therefore be completely readable when the
         * samples already stored cover the requested delay. This matches
         * lib/aec, which checks `filled >= delay + hop` after writing the
         * current hop; checking delay+hop here would report LOCKED one hop
         * later than the audio actually switches to aligned far. */
        state.solid = p->delay_samples_seen >=
            (uint64_t)p->cfg.fixed_delay_samples;
        p->accepted_delay = p->cfg.fixed_delay_samples;
        p->last_delay = state;
        return state;
    }

    /* Raw, un-decimated hops: delay_aec3_init() constructed p->shared_delay
     * at this pipeline's real native sample_rate, so DelayAec3 itself
     * anti-alias-filters + decimates internally at 48kHz (see delay_aec3.c's
     * DaResample48) and delay_aec3_estimated_delay() below returns the
     * result already rescaled back to this pipeline's native domain. */
    /* Duty-cycled analysis, the same machine lib/aec runs on its own
     * estimator (aec.c's delay block): the estimator is FED on every hop
     * regardless -- delay_aec3_accumulate_ex() keeps the decimators and the
     * render ring gapless, so the matched filter never sees spliced audio --
     * and only the matched-filter ANALYSIS is decimated, and only once the
     * published estimate has held still long enough to make analysing it
     * again redundant.
     *
     * What re-arms full rate, stated as the contract it actually is: the
     * estimate is compared against the previous hop's below, and a decimated
     * hop cannot publish a new one -- so a movement mid-stream is SEEN at the
     * next SCHEDULED analysis, not on the hop it happens. Full rate then
     * resumes from the hop that sees it, and the whole exposure is bounded by
     * one duty period: at most duty_period_hops of extra wait. At the
     * default cfg.delay_est_period_s=0.0 that period is 1 (full rate, no
     * wait at all). The K=12/6/9 figures below (~96 ms on every grid this
     * pipeline ships: 16 kHz/128, 16 kHz/256, 48 kHz/512 -- the /5 divisor
     * and the rounding land all three on the same wall clock) are what
     * opting into the throttle at cfg.delay_est_period_s=0.5 reproduces --
     * the regime this file ran by default before AMEND_35 (2026-10-18)
     * moved lib/aec's own matching default to full rate. The ERLE watchdog
     * below can shorten that by
     * disengaging on a cancellation collapse the skipped hops still measure,
     * but it is armed only once a lane's windowed ERLE clears
     * DUTY_ERLE_COLLAPSE_DB: on a path where no lane ever cancels that well,
     * the 1-in-K sampling is the whole re-lock mechanism and re-acquisition
     * costs a multiple of what full rate would -- measured 2.5x on the
     * 16 kHz/256 grid, bounded at K, with both arms pinned in
     * test_4aec_nr_res.c. So the machine costs latency where the answer was
     * already static, plus one duty period wherever it stops being static.
     *
     * The census is counted HERE, at the one site that consumes the
     * decision, so it can never claim a schedule the estimator did not run. */
    run_filter = 1;
    if (p->duty_active) {
        p->duty_pos += 1;
        if (p->duty_pos >= p->duty_period_hops) p->duty_pos = 0;
        run_filter = (p->duty_pos == 0);
    }
    if (run_filter) p->duty_hops_run += 1;
    emitted = delay_aec3_accumulate_ex(&p->shared_delay, capture, render,
                                       p->hop_size, run_filter);
    (void)emitted;
    p->delay_calls += 1;
    {
        int cur = delay_aec3_estimated_delay(&p->shared_delay);
        int solid = delay_aec3_is_solid(&p->shared_delay);
        if (!solid || cur < 0 || cur != p->duty_last_delay) {
            /* Estimate moved or lost confidence: full rate, and the hold
             * starts over. */
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
        /* Cancellation watchdog: a leaky peak of the lanes' windowed ERLE,
         * with full-rate analysis resumed on a DUTY_ERLE_COLLAPSE_DB drop
         * from it. Armed only once the peak clears that same figure, so it
         * cannot fire before any lane ever converged. Same shape and same
         * collapse depth as lib/aec's; the leak is the same 0.1 dB/s, carried
         * in this grid's per-hop units by duty_erle_leak_db. */
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

    estimated = delay_aec3_estimated_delay(&p->shared_delay);
    eligible = estimated >= 0 &&
               delay_aec3_is_solid(&p->shared_delay) &&
               delay_aec3_n_updates(&p->shared_delay) >= 3;

    /* One hop of the held candidate's life, spent here -- ahead of both the
     * quarantine and the admission below, exactly where lib/aec ages its own
     * pending delay. Held candidates therefore expire on hops the quarantine
     * takes away as readily as on hops the estimate moves elsewhere. */
    four_aec_nr_res_admission_age(&p->delay_admission);

    /* Backward-jump quarantine (see delay_backward_quarantine_enabled in the
     * header). Engages only once a usable generation exists and only for an
     * estimate strictly EARLIER than the one in force: a first acquisition
     * has no alignment to protect, re-confirming the accepted value is not a
     * change, and a LARGER estimate is not the pre-echo direction. The lane
     * is read here, BEFORE this hop's aec_process_context() calls, so what it
     * answers with is last hop's cancellation -- the same one-hop-behind
     * reading lib/aec's own Path-B guard judges on.
     *
     * ONE lane: cfg.capture_proxy_channel, the microphone the shared
     * estimator is actually fed from, so the only lane whose cancellation is
     * evidence about the estimate being judged.
     *
     * Armed once per continuously qualifying backward episode, then one tick
     * per hop; at 0 the estimate is adopted. Candidate values may jitter
     * within that qualifying class without re-arming. A forward/ineligible
     * estimate or lost cancellation evidence disarms the episode. */
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

    /* Published `solid` = "a usable accepted alignment generation exists",
     * which is why it is derived from the SAME acceptance test that writes
     * accepted_delay rather than mirroring the estimator's raw confidence.
     * Nothing in DelayAec3's contract ties its confidence latch to the
     * acceptance conditions above, so a raw-confidence `solid` would be free
     * to LEAD accepted_delay on any hop the two disagree -- and a consumer
     * that flushes recurrent state on the not-usable -> usable edge would
     * then flush against the previous generation's applied delay.
     *
     * Usability is also sticky: once a generation exists, a short
     * is_solid/confidence dip keeps the accepted delay in force instead of
     * briefly retracting an alignment the audio path is still applying.
     * That is exactly lib/aec's semantics, where "nothing accepted yet" is
     * spelled current_delay == -1 and never un-spells itself without a
     * reset. p->last_delay is zeroed by init's pool memset and by reset(),
     * so was_usable is 0 at every genuine stream start.
     *
     * `changed` = "this hop starts a NEW USABLE alignment generation" (see
     * FourAecNrResDelayState's doc for why a value-only comparison misses
     * every acquisition or relock that lands on applied delay 0). The
     * previous hop's usability is exactly its published `solid`: this
     * wrapper's delay_samples is never negative (accepted_delay starts at 0
     * and only ever takes an `estimated >= 0`), so there is no -1 sentinel
     * half to test.
     *
     * A CHANGE to a usable alignment additionally has to clear the admission
     * in 4aec_nr_res_internal.h, because every one of them costs an IR shift
     * plus (on a retard) a far-history clear on all FOUR lanes:
     *
     *   - FOUR_DELAY_CHANGE_MIN_SAMPLES bounds how small a movement may
     *     disturb four converged filters at all. DelayAec3 publishes on a
     *     16-downsampled-sample grid -- 64 native samples at 16 kHz, 192 at
     *     48 kHz, since its answer is a block index shifted left -- so
     *     with this estimator this term is the floor under a finer
     *     source rather than an active filter.
     *   - the held candidate is the operative half: a value that is not
     *     offered again before it ages out never reaches the lanes.
     *
     * What this deliberately does NOT reject is a SUSTAINED wrong estimate:
     * it is re-offered every hop and confirms itself. Holding that one is the
     * backward quarantine's job above, on its own evidence and its own bound.
     *
     * First acquisition keeps its immediate path: with nothing accepted yet
     * there is no alignment to protect, the same split lib/aec makes between
     * its Path A and Path B.
     *
     * accepted_delay moves only on acceptance, so the alignment served to the
     * lanes and the realign that shifts their filters always land on the same
     * hop; a value written on the pending hop would feed them a reference
     * shifted out from under filters nothing had realigned. */
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
    state.confidence = delay_aec3_confidence(&p->shared_delay);
    state.solid = now_usable;
    state.estimator_calls = p->delay_calls;
    state.estimator_updates = delay_aec3_n_updates(&p->shared_delay);
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
    /* The hop occupies ring slots start%ring .. (start+hop-1)%ring, which are
     * contiguous and wrap at most once (the guard above leaves hop <= ring),
     * so the whole write is one or two bounded runs: the same samples land in
     * the same slots as a per-sample `absolute % ring` walk would put them,
     * without dividing 64-bit values `hop` times per hop. The divisor is a
     * runtime value, so a compiler cannot strength-reduce it away, and a
     * 目標平台 core has no fast 64-bit divide to fall back on. `start` is
     * reduced ONCE and every other index below is derived from that with
     * 32-bit adds. */
    write_at = (int)(start % (uint64_t)ring);
    first = ring - write_at;
    if (first > hop) first = hop;
    memcpy(p->delay_ring + write_at, render, (size_t)first * sizeof(float));
    if (first < hop) {
        memcpy(p->delay_ring, render + first,
               (size_t)(hop - first) * sizeof(float));
    }
    /* Whole-hop decision, taken on the SAME predicate the published `solid`
     * uses under FIXED: the ring can serve this hop's requested offset only
     * once the samples already seen cover delay_samples. Until then the far
     * content IS the raw render hop -- lib/aec's rule ("UNLOCKED means the
     * content is the raw far") rather than silence, so a consumer stepping a
     * recurrent model over the far branch sees real reference audio from the
     * first hop instead of a stretch of zeros. Serving zeros would also make
     * the leading delay_samples of every FIXED stream unmodellable for the
     * linear filters. Per-hop and not per-sample: `start` is the smallest
     * absolute index in this hop, so a partly-servable hop would otherwise
     * splice raw and aligned audio while the seam still reports UNLOCKED. */
    if (start >= (uint64_t)delay_samples) {
        /* (start - delay_samples) mod ring, without a second 64-bit divide:
         * start is congruent to write_at, and delay_samples is bounded by
         * ring above, so write_at - delay_samples + ring lands in (0, 2*ring)
         * and one conditional subtract reduces it. Same first-sample slot,
         * and the read run wraps at most once for the same reason the write
         * run does. */
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

/* Points dst's spectrum/PSD fields (error_spec, echo_spec, far_spec,
 * near_spec, r2, comfort_noise, formed_hop -- these back the front view and
 * the fuse inputs) directly at src's own buffers instead of copying them.
 *
 * Safe because AecResContext's pointers are documented to "alias the AEC's
 * internal per-hop buffers -- read before the next aec_process() call"
 * (aec.h): a lane runs again only inside four_aec_front_process() or after a
 * reset, and the front view is documented valid until exactly then.
 * four_aec_nr_res_reset() does not clear or overwrite this memory -- aec_reset()
 * makes no such promise for a lane's spectrum buffers -- so a caller that
 * keeps reading a view past a reset reads undefined content, and fuse refuses
 * to run on one (front_valid). pipeline.py's FourChannelAecPipeline keeps
 * several frames in flight and copies; this borrow design is deliberately
 * not mirrored there. */
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
    const Complex* shared_far_spec = NULL;   /* Group 6: set by lane 0, borrowed by lanes 1-3 */
    long far_fft_before[FOUR_AEC_NR_RES_CHANNELS];
    int lanes_borrowed;

    if (!p || p->destroyed ||
        !microphones_interleaved || !ref || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    /* Cleared once the call is accepted: a hop that then bails out part-way
     * reports zero for the stages it never reached rather than the previous
     * hop's numbers. The three AEC-side fields are ACCUMULATED over the four
     * lanes below, so this is also what makes each hop start from zero. The
     * post half is cleared the same way in four_aec_fuse(). */
    p->last_timing.delay_us = 0;
    p->last_timing.frontend_us = 0;
    p->last_timing.linear_us = 0;
    p->last_timing.lane_res_us = 0;
    /* Same rule for the fuse stage's far-end provenance: a hop that bails out
     * before the lane loop has established nothing, and "nothing established"
     * is what makes fuse_contexts() fall back to the full comparison. */
    p->far_spec_shared_hop = 0;
    p->front_valid = 0;

    hop = p->hop_size;
    if (!inputs_finite(
            microphones_interleaved,
            (size_t)hop * FOUR_AEC_NR_RES_CHANNELS) ||
        !inputs_finite(ref, (size_t)hop))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    /* Deinterleaves the proxy channel into p->mic_lane. update_shared_delay()/
     * align_render() below only ever read this buffer (capture is `const
     * float*`), so it still holds exactly these samples, unchanged, by the
     * time the 4-lane loop below reaches ch==capture_proxy_channel -- that
     * lane's own extraction is skipped there instead of redoing it. */
    for (i = 0; i < hop; ++i) {
        p->mic_lane[i] =
            microphones_interleaved[
                i * FOUR_AEC_NR_RES_CHANNELS +
                p->cfg.capture_proxy_channel];
    }
    /* The alignment the lanes were served BEFORE this hop's decision: raw
     * (0) until the estimate first turns solid. Captured ahead of
     * update_shared_delay(), which overwrites both fields. */
    old_align = p->last_delay.solid ? p->accepted_delay : 0;
    t0 = four_aec_nr_res_now_us();
    delay = update_shared_delay(p, p->mic_lane, ref);
    p->last_timing.delay_us = four_aec_nr_res_now_us() - t0;
    /* align_render() below is deliberately OUTSIDE the window: it is a
     * ring-buffer copy, not delay estimation, and the caller's reconciliation
     * identity accounts for it in the pre-stage remainder. */
    if (!align_render(p, ref, delay.delay_samples)) {
        four_aec_nr_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (delay.changed) {
        /* Realign instead of reset: aec_apply_external_realign() shifts each
         * converged filter by the alignment delta (warm tap-transfer when the
         * evidence gate holds, a filter-only reset otherwise), so the
         * cancellation survives and the WOLA sequences continue. A full
         * aec_reset() + OLA wipe would produce one near-zero output hop plus
         * dozens of hops of re-exposed echo (covered by lib/aec
         * test_external_realign.c and tests/test_4aec_nr_res.c's realign
         * continuity rows). */
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            int outcome = aec_apply_external_realign(
                p->lanes[ch], delay.delay_samples - old_align);
            /* Which path each lane took is otherwise invisible from here --
             * the call reports it and nothing kept the answer. Counted, not
             * branched on: a lane that goes soft is still correct, just cold
             * for a while, so this is what makes "the sweep ran, and how it
             * landed" measurable. A sweep that does not add exactly 4 means a
             * lane rejected the call (-1: no instance, or a lane not in
             * EXTERNAL_ALIGNED), which is a wiring fault rather than a soft
             * realign and must not be counted as one. */
            if (outcome == 1) p->realign_warm_lanes += 1;
            else if (outcome == 0) p->realign_soft_lanes += 1;
        }
    }

    /* Read BEFORE the loop so the verdict below rests on what the lanes
     * themselves counted this hop, not on the fact that this function
     * intended to share a spectrum -- a flag set by the code that does the
     * borrowing would be an identity, true whether or not the borrow landed.
     * Two loads and an add per lane, off the transform path entirely. */
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        far_fft_before[ch] = aec_far_fft_real_compute_count(p->lanes[ch]);
    lanes_borrowed = 1;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        AecResContext context;
        if (ch != p->cfg.capture_proxy_channel) {
            for (i = 0; i < hop; ++i) {
                p->mic_lane[i] =
                    microphones_interleaved[
                        i * FOUR_AEC_NR_RES_CHANNELS + ch];
            }
        }
        /* Every lane is context-only: downstream processing consumes
         * context.formed_hop, so the context entry points avoid an unused
         * output copy while advancing the same AEC state.
         *
         * Every lane sees the identical p->aligned_ref this hop
         * (the shared, delay-aligned reference -- same pointer passed to
         * every lane below), so lane 0's far-end FFT is byte-identical to
         * what any other lane would compute from the same signal. Lane 0
         * runs the real transform (aec_process_context(), unchanged) and
         * exposes it via context.far_spec (unconditionally populated by
         * aec_get_res_context(), independent of enable_res/
         * return_res_context); lanes 1-3 borrow it through
         * aec_process_context_shared_far() instead of each redundantly
         * recomputing an identical FFT -- 4 far-FFTs/hop become 1. This
         * borrowed pointer aliases lane 0's own persistent far_spec buffer
         * (not a copy) and stays valid for the rest of this hop's loop
         * body, since lane 0 is not touched again until the next call to
         * this function. */
        if (ch == 0) {
            aec_process_context(p->lanes[ch], p->mic_lane, p->aligned_ref);
        } else {
            aec_process_context_shared_far(
                p->lanes[ch], p->mic_lane, p->aligned_ref, shared_far_spec);
        }
        aec_get_res_context(p->lanes[ch], &context);
        if (ch == 0) shared_far_spec = context.far_spec;
        if (!context.formed_hop || !bind_lane_view(
                &p->snapshots[ch], &context, p->n_freqs,
                p->cfg.enable_post)) {
            four_aec_nr_res_reset(p);
            return FOUR_AEC_NR_RES_DSP_ERROR;
        }
        /* Sum this lane's own stage split into the hop's record. The four
         * lanes run sequentially, so summing gives the wall-clock cost of
         * each stage across the whole loop -- see FourAecNrResLastTiming. */
        {
            AecStageTiming lane_time;
            aec_get_last_timing(p->lanes[ch], &lane_time);
            /* Lane-internal delay work folds into the same delay_us the
             * shared estimator reports, because it is the same quantity:
             * time spent aligning the far end. It is structurally ZERO
             * -- every lane is built AEC_DELAY_EXTERNAL_ALIGNED, so no lane
             * owns a ring or an estimator -- and is summed anyway so that a
             * future lane-mode change cannot silently drop its cost out of
             * the breakdown. */
            p->last_timing.delay_us    += lane_time.delay_us;
            p->last_timing.frontend_us += lane_time.frontend_us;
            p->last_timing.linear_us   += lane_time.linear_us;
            p->last_timing.lane_res_us += lane_time.res_us;
        }
        /* The borrow chain's own signature, one lane at a time: the lane that
         * computes the shared spectrum runs exactly ONE real far transform
         * this hop, and a lane that borrowed it runs NONE. Anything else --
         * a lane that recomputed its own, or a lane that never reached its
         * front end at all -- leaves this 0 and sends fuse_contexts() back to
         * the per-bin comparison. */
        {
            long ran = aec_far_fft_real_compute_count(p->lanes[ch]) -
                       far_fft_before[ch];
            if (ran != (ch == 0 ? 1 : 0)) lanes_borrowed = 0;
        }
    }
    /* shared_far_spec is lane 0's live spectrum, so a NULL here means lanes
     * 1-3 were handed nothing to borrow and the counter evidence above is
     * about some other arrangement. */
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
    /* The exact delay-aligned far every lane consumed this hop. */
    out->aligned_ref = p->aligned_ref;
    return FOUR_AEC_NR_RES_OK;
}

/* ============================================================================
 * Per-hop processing — Stages 2/3: external beamformer -> NR/RES -> OLA
 * ========================================================================== */


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
                         float* max_dt,
                         float* max_saturation) {
    int n = p->n_freqs;
    int k;
    int ch;
    float base_far_power = p->snapshots[0].far_power;

    *all_converged = 1;
    /* Seeded neutral rather than from lane 0, so lane 0 is subject to the
     * same contributes-to-the-output test as every other lane below. Both
     * quantities are non-negative, so 0 is the identity for the max. At
     * least one lane always survives that test: four_aec_fuse() rejects
     * all-zero weights before reaching here. */
    *max_dt = 0.0f;
    *max_saturation = 0.0f;

    if (trusted_beamformed_error) {
        if (!complex_vector_finite(trusted_beamformed_error, n)) return 0;
        /* No copy into p->fused_error: its lifetime (owned by the caller,
         * e.g. audio_pipeline_4ch.c's p->gsc_spectrum) already covers every
         * read the post-beam stages make this call -- they read
         * trusted_beamformed_error directly instead. */
    } else {
        memset(p->fused_error, 0, (size_t)n * sizeof(Complex));
    }
    memset(p->fused_near, 0, (size_t)n * sizeof(Complex));
    /* residual_proj accumulates the coherent residual vector below;
     * fused_r2 takes its power. */
    memset(p->residual_proj, 0, (size_t)n * sizeof(Complex));
    memset(p->fused_comfort, 0, (size_t)n * sizeof(float));

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        const Complex* lane_weights =
            weights + (size_t)ch * (size_t)n;
        /* A lane the beamformer has zeroed in every bin contributes exactly
         * nothing to fused_error/fused_near/residual_proj/fused_comfort below,
         * so it must not vote on the three scalars that steer the post stage
         * either -- those govern a spectrum this lane is absent from. The
         * degenerate case is real: with the external beamformer disabled the
         * bypass emits identity weights [1,0,0,0], and lanes 1-3 would still
         * drive max_dt, which selects the residual suppressor's floor.
         *
         * With a beamformer that actually beamforms, every lane carries
         * weight somewhere and this skips nothing: the reductions see the
         * same four lanes they always did, bit for bit. */
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
            if (p->snapshots[ch].dt_indicator > *max_dt)
                *max_dt = p->snapshots[ch].dt_indicator;
            if (p->snapshots[ch].saturation_level > *max_saturation)
                *max_saturation = p->snapshots[ch].saturation_level;
        }

        if (!four_aec_nr_res_far_spec_agrees(
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

/* ============================================================================
 * Per-hop processing — Stage 2: the beamformer-weighted projection
 * ========================================================================== */

int four_aec_fuse(
    FourAecNrRes* p,
    const Complex* weights,
    const Complex* beamformed_error,
    SeSpecFrame* frame) {
    int all_converged;
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
            &all_converged, &max_dt, &max_saturation)) {
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
    frame->clock_drift = p->cfg.delay_mode == AEC_DELAY_MATCHED
        ? delay_aec3_has_clockdrift(&p->shared_delay) : 0;
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
    *nr_cfg = se_anr_compose_config(cfg->sample_rate, fft, hop, cfg->nr_mode);
    return 0;
}

/* ============================================================================
 * Reset and teardown
 * ========================================================================== */

/* Legal at any time. A caller still holding a FourAecFront must stop reading
 * its pointers the instant this is called: they alias per-lane AecResContext
 * buffers that aec_reset() does not clear, so a violation shows as stale
 * content, not as a fault. four_aec_fuse() refuses to run until a new front
 * frame has been processed. */
void four_aec_nr_res_reset(FourAecNrRes* p) {
    int ch;
    if (!p || p->destroyed) return;

    if (p->cfg.delay_mode == AEC_DELAY_MATCHED)
        delay_aec3_reset(&p->shared_delay);
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
    /* A reset abandons the alignment the quarantine was protecting, so a
     * countdown armed against it must not survive. The WINDOW
     * (delay_quarantine_hops) is config-derived and stays. Same for a held
     * change candidate and the life left on it: it describes a movement away
     * from an alignment the reset abandoned, and the next acquisition is
     * immediate anyway. */
    p->delay_quarantine_left = -1;
    p->delay_admission.candidate = 0;
    p->delay_admission.ttl = 0;
    /* Zeroed with the lanes' own counters (aec_reset() clears the per-lane
     * far-FFT count), so every instrumentation total shares one epoch. */
    p->realign_warm_lanes = 0;
    p->realign_soft_lanes = 0;
    p->delay_calls = 0;
    p->far_spec_shared_hop = 0;
    /* The duty machine measured the alignment this reset just abandoned, so
     * it re-arms with it: full-rate analysis, no hold, no peak, and a census
     * that starts over rather than blending two regimes into one ratio.
     * The two window figures are grid-derived and stay. */
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
        total += aec_far_fft_real_compute_count(p->lanes[ch]);
    return total;
}

long four_aec_nr_res_realign_warm_lane_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->realign_warm_lanes : 0;
}

long four_aec_nr_res_realign_soft_lane_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->realign_soft_lanes : 0;
}

unsigned long long four_aec_nr_res_duty_hops_total(const FourAecNrRes* p) {
    /* Alias of delay_calls: the estimator is fed on every analysed hop, so
     * the census denominator IS the call count -- no second field to drift. */
    return p && !p->destroyed ? p->delay_calls : 0ull;
}

unsigned long long four_aec_nr_res_duty_hops_run(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->duty_hops_run : 0ull;
}

int four_aec_nr_res_far_spec_provenance(const FourAecNrRes* p) {
    return p && !p->destroyed ? p->far_spec_shared_hop : 0;
}

/* Declared in 4aec_nr_res_internal.h; see there for why a harness needs it.
 * A period of ONE hop is what "analyse every hop" is spelled as at the
 * decision site: duty_pos advances into the period and immediately wraps, so
 * run_filter is true whether or not the machine ever arms, and the arming,
 * the hold and the watchdog keep running and keep being measurable through
 * the census. init's own floor is 1 too now (AMEND_35-mirrored, see the
 * carve-site comment), so at the default cfg.delay_est_period_s=0.0 this
 * call is already a no-op -- this setter earns its keep only for a core
 * built with a non-zero delay_est_period_s (the opt-in throttle), where it
 * forces the schedule back to full rate for a parity harness without
 * disturbing anything else the duty machine tracks. */
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

/* ============================================================================
 * Diagnostic memory breakdown
 * ========================================================================== */

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

/* ============================================================================
 * Diagnostic per-stage timing
 * ========================================================================== */

/* See FourAecNrResLastTiming's doc comment (4aec_nr_res.h) for the full
 * stage-boundary reference. */
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
