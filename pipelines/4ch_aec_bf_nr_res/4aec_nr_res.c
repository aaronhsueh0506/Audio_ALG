/**
 * 4aec_nr_res.c — implementation of 4aec_nr_res.h.
 *
 * Four-channel counterpart of pipelines/mono_aec_nr_res/audio_pipeline.c:
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
#include "simd_kernels.h"
#include "nr_overlay.h"

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
    MmseLsaDenoiser* nr;
    FftHandle* fft;

    SuppressionGain post_sg;
    SuppressionGainConfig post_sg_cfg;
    SuppressionGainTuning post_sg_tun;
    float* post_sg_storage;

    FourAecLaneSnapshot snapshots[FOUR_AEC_NR_RES_CHANNELS];

    float* linear_interleaved;
    float* mic_lane;
    float* aligned_ref;
    float* render_i16;

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
    Complex* output_spec;
    float* fused_r2;
    float* fused_comfort;
    float* post_near_power;
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

    /* Per-stage wall-clock timing -- see FourAecNrResLastTiming's doc comment
     * (4aec_nr_res.h) for what each field measures. Stored as the published
     * record itself rather than eight private twins, so the accessor is one
     * assignment and a new stage cannot be added to one side only. Appended
     * at the end, the same precedent as aec.h's Aec additions: every field
     * above keeps its pre-existing offset. */
    FourAecNrResLastTiming last_timing;
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
    int* delay_ring_size,
    size_t* delay_estimator_bytes) {
    const Aec3BalancedRateDims* rate_dims;
    int selected_fft;
    int max_delay_samples;
    if (!cfg || !aec_cfg || !nr_cfg || !fft_size || !hop_size ||
        !n_freqs || !post_ma_n || !delay_ring_size ||
        !delay_estimator_bytes) return 0;
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
    FOUR_CK_BOOL(enable_post);

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
    /* Bound kept in sync with the two mono apps' audio_pipeline*.c
     * (each app is self-contained). */
    if (cfg->filter_length < 0 || cfg->filter_length > 4096) return 0;
    switch (cfg->delay_mode) {
        case AEC_DELAY_MATCHED:
            if (cfg->delay_num_filters < 1 ||
                cfg->delay_num_filters > DA_NUM_FILTERS ||
                cfg->fixed_delay_samples != -1) return 0;
            break;
        case AEC_DELAY_FIXED:
            if (cfg->delay_num_filters != DA_NUM_FILTERS ||
                cfg->fixed_delay_samples < 0 ||
                (long)cfg->fixed_delay_samples >
                    120L * (long)cfg->sample_rate) return 0;
            break;
        case AEC_DELAY_EXTERNAL_ALIGNED:
            if (cfg->delay_num_filters != DA_NUM_FILTERS ||
                cfg->fixed_delay_samples != -1) return 0;
            break;
        default:
            return 0;
    }
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
     * validation today. */
    if (cfg->delay_backward_quarantine_enabled != 0 &&
        cfg->delay_backward_quarantine_enabled != 1) return 0;
    if (!isfinite(cfg->delay_backward_quarantine_s) ||
        cfg->delay_backward_quarantine_s < 0.0f ||
        cfg->delay_backward_quarantine_s > 3600.0f) return 0;
    if (!isfinite(cfg->max_delay_ms) ||
        cfg->max_delay_ms < 0.0f || cfg->max_delay_ms > 4096.0f) return 0;
    *fft_size = selected_fft;

    *hop_size = *fft_size / 2;
    *n_freqs = *fft_size / 2 + 1;
    rate_dims = aec3b_rate_cfg(cfg->sample_rate, *fft_size);
    if (!rate_dims || rate_dims->sg_nearend_smoother_n < 1) return 0;
    *post_ma_n = rate_dims->sg_nearend_smoother_n;

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

    aec_config_from_preset(aec_cfg, cfg->aec_preset, cfg->sample_rate);
    aec_cfg->fft_size = *fft_size;
    if (cfg->filter_length > 0)
        aec_cfg->filter_length = cfg->filter_length;
    /* The wrapper owns alignment once for all four lanes. Each lane therefore
     * consumes an already-aligned reference and must not allocate or run a
     * private estimator/ring. */
    aec_cfg->delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    aec_cfg->delay_num_filters = DA_NUM_FILTERS;
    aec_cfg->fixed_delay_samples = -1;
    aec_cfg->enable_delay_est = 0;
    aec_cfg->enable_res = 0;
    aec_cfg->return_res_context = 1;
    /* Every lane's own G_res is never read (see FourAecLaneSnapshot / this
     * file's fuse_contexts()+run_post_res_and_nr(), which recompute an
     * equivalent gain once from fused multi-lane data) -- skip computing it
     * per lane. */
    aec_cfg->spatial_linear_context = 1;

    *nr_cfg = pipelines_compose_nr_config(cfg->sample_rate, *fft_size, *hop_size,
                                cfg->nr_mode);
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
    int fft = p->fft_size;
    size_t linear_count = (size_t)hop * FOUR_AEC_NR_RES_CHANNELS;

    p->linear_interleaved =
        (float*)pool_carve(cursor, linear_count, sizeof(float));
    p->mic_lane =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    p->aligned_ref =
        (float*)pool_carve(cursor, (size_t)hop, sizeof(float));
    if (p->cfg.enable_post)
        p->render_i16 =
            (float*)pool_carve(cursor, (size_t)hop, sizeof(float));

    if (!p->cfg.enable_post)
        return p->linear_interleaved && p->mic_lane && p->aligned_ref;

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

    /* p->snapshots[ch]'s Complex- and float-pointer fields are no longer
     * pool-carved -- bind_lane_view() (formerly snapshot_context()) points
     * them directly at each lane's own AecResContext buffers every hop
     * instead of owning a private copy. See bind_lane_view()'s doc comment
     * for the lifetime contract this depends on. */

    return p->linear_interleaved && p->mic_lane &&
           p->aligned_ref && p->render_i16 &&
           p->fused_error && p->fused_near &&
           p->residual_work && p->output_spec &&
           p->fused_r2 && p->fused_comfort && p->post_near_power &&
           p->total_gain && p->extra_noise &&
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
    int hop, int fft, int n, int post_ma_n, int delay_ring_size,
    size_t delay_estimator_bytes, int enable_post) {
    size_t total = 0;
    int i;

    total = ck_field_size(
        total,
        ck_mul_size((size_t)hop, FOUR_AEC_NR_RES_CHANNELS),
        sizeof(float));                                      /* linear */
    /* mic_lane + aligned_ref are always required; render_i16 and all
     * frequency-domain post buffers exist only for the RES/NR post path. */
    for (i = 0; i < (enable_post ? 3 : 2); ++i)
        total = ck_field_size(total, (size_t)hop, sizeof(float));

    if (enable_post) {
    for (i = 0; i < 4; ++i)
        total = ck_field_size(total, (size_t)n, sizeof(Complex));
    for (i = 0; i < 6; ++i)
        total = ck_field_size(total, (size_t)n, sizeof(float));
    for (i = 0; i < 3; ++i)
        total = ck_field_size(total, (size_t)fft, sizeof(float));

    /* No per-lane snapshot buffers here: FourAecLaneSnapshot's spectrum/PSD
     * fields are borrowed pointers into each lane's own AecResContext, not
     * pool-carved copies. See bind_lane_view()/carve_working_buffers(). */

    total = ck_field_size(
        total,
        ck_mul_size((size_t)(10 + post_ma_n), (size_t)n),
        sizeof(float));                                      /* post SG */
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
    const MmseLsaConfig* nr_cfg,
    int hop,
    int fft,
    int n,
    int post_ma_n,
    int delay_ring_size,
    size_t delay_estimator_bytes,
    int enable_post) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    size_t nr_bytes = enable_post ? mmse_lsa_get_mem_size(nr_cfg) : 0;
    size_t fft_bytes = enable_post ? fft_get_mem_size(fft) : 0;
    size_t buffer_bytes = pipeline_buffer_size(
        hop, fft, n, post_ma_n, delay_ring_size, delay_estimator_bytes,
        enable_post);
    size_t total = 0;
    int ch;

    if (aec_bytes == 0 || buffer_bytes == 0 ||
        (enable_post && (nr_bytes == 0 || fft_bytes == 0))) return 0;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        total = ck_add_size(total, ck_align16_size(aec_bytes));
    if (enable_post) {
        total = ck_add_size(total, ck_align16_size(nr_bytes));
        total = ck_add_size(total, ck_align16_size(fft_bytes));
    }
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
    size_t nr_bytes = p->cfg.enable_post ? mmse_lsa_get_mem_size(nr_cfg) : 0;
    size_t fft_bytes = p->cfg.enable_post ? fft_get_mem_size(p->fft_size) : 0;
    int ch;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        void* lane_pool = pool_carve(cursor, 1, aec_bytes);
        if (!lane_pool) return 0;
        p->lanes[ch] = aec_init(lane_pool, aec_bytes, aec_cfg);
        if (!p->lanes[ch]) return 0;
        p->initialized_lanes += 1;
    }

    if (p->cfg.enable_post) {
        void* nr_pool = pool_carve(cursor, 1, nr_bytes);
        void* fft_pool = pool_carve(cursor, 1, fft_bytes);
        if (!nr_pool || !fft_pool) return 0;
        p->nr = mmse_lsa_init(nr_pool, nr_bytes, nr_cfg);
        p->fft = fft_init(fft_pool, fft_bytes, p->fft_size);
        if (!p->nr || !p->fft) return 0;
    }

    if (!carve_working_buffers(p, cursor) ||
        (p->cfg.enable_post && !init_post_sg(p, cursor))) return 0;
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
        "|carve:self,aec0,aec1,aec2,aec3,nr,fft,linear,hop3,"
        "post?(nr,fft,hop1,complex4,float6,fftfloat3,postsg),"
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
    cfg.max_delay_ms = 1024.0f;
    cfg.aec_preset = AEC_PRESET_BALANCED;
    cfg.nr_mode = MMSE_LSA_NR_BALANCED;
    cfg.enable_post = 1;
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
    size_t delay_estimator_bytes;

    if (!out || !derive_dims_and_configs(
            cfg, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size,
            &delay_estimator_bytes)) return -1;

    pool_bytes = pipeline_pool_size(
        &aec_cfg, &nr_cfg, hop, fft, n, post_ma_n, delay_ring_size,
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
    MmseLsaConfig nr_cfg;
    int fft;
    int hop;
    int n;
    int post_ma_n;
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
            &cfg_copy, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size,
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
    }
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

    if (p->cfg.enable_post) {
        for (k = 0; k < fft; ++k) {
            p->synth_window[k] = sqrtf(
                0.5f * (1.0f - cosf(
                    2.0f * M_PI_F * (float)k / (float)fft)));
        }
    }

    for (k = 0; k < FOUR_AEC_NR_RES_CHANNELS; ++k) {
        AecResContext context;
        aec_get_res_context(p->lanes[k], &context);
        if (context.hop_size != hop || context.n_freqs != n)
            return NULL;
    }
    if (p->cfg.enable_post &&
        (mmse_lsa_get_hop_size(p->nr) != hop ||
         mmse_lsa_get_n_freqs(p->nr) != n ||
         fft_get_n_freqs(p->fft) != n)) return NULL;
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
     *     against today's estimator this term is the floor under a finer
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
    int i;
    uint64_t start;
    if (p->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) {
        memcpy(p->aligned_ref, render,
               (size_t)p->hop_size * sizeof(float));
        p->delay_samples_seen += (uint64_t)p->hop_size;
        return 1;
    }
    if (!p->delay_ring || delay_samples < 0 ||
        delay_samples > p->delay_ring_size - p->hop_size) return 0;

    start = p->delay_samples_seen;
    for (i = 0; i < p->hop_size; ++i) {
        uint64_t absolute = start + (uint64_t)i;
        p->delay_ring[absolute % (uint64_t)p->delay_ring_size] =
            render[i];
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
        for (i = 0; i < p->hop_size; ++i) {
            uint64_t source = start + (uint64_t)i - (uint64_t)delay_samples;
            p->aligned_ref[i] =
                p->delay_ring[source % (uint64_t)p->delay_ring_size];
        }
    } else {
        memcpy(p->aligned_ref, render,
               (size_t)p->hop_size * sizeof(float));
    }
    p->delay_samples_seen += (uint64_t)p->hop_size;
    return 1;
}

/* Points dst's spectrum/PSD fields (error_spec, echo_spec, far_spec,
 * near_spec, r2, comfort_noise -- these back out->linear_spectra and the
 * internal fuse_contexts() inputs) directly at src's own buffers instead of
 * copying them. Does NOT touch linear_interleaved, which is a separate,
 * genuinely pipeline-owned buffer filled by a per-sample copy loop in the
 * caller (four_aec_nr_res_process_pre()) -- only the fields bound here
 * alias a lane's own memory.
 *
 * Safe only because AecResContext's pointers are documented to "alias the
 * AEC's internal per-hop buffers -- read before the next aec_process()
 * call" (aec.h), and this file's own pending-token gate
 * (four_aec_nr_res_process_pre()'s `if (p->pending) return ...`, plus
 * token_matches() at the top of process_post_impl()) structurally prevents
 * any lane's aec_process() from running again before the borrowed pointers
 * are consumed. four_aec_nr_res_reset() does NOT guarantee this memory is
 * cleared or overwritten -- aec_reset() carries no such promise for a
 * lane's spectrum buffers, so a stale value may simply be left in place at
 * the same address. What actually keeps this safe is the API contract
 * (linear_spectra is valid only until process_post(), reset(), or
 * destroy() -- see FourAecNrResPreFrame's doc comment) combined with the
 * pending/token gate, which is what stops THIS file's own process_post()
 * from ever reading post-reset lane memory; an external caller must
 * independently honor the same contract rather than assume reset() will
 * make a violation visible. Do not port this borrow-pointer design to
 * pipeline.py's FourChannelAecPipeline: that reference implementation's
 * contract (tests/test_pipeline.py's queued-pre-frames test) allows multiple
 * pre-frames in flight simultaneously and requires true copy semantics. */
static int bind_lane_view(FourAecLaneSnapshot* dst,
                          const AecResContext* src,
                          int n_freqs) {
    if (!dst || !src || src->n_freqs != n_freqs ||
        !src->error_spec || !src->echo_spec || !src->far_spec ||
        !src->near_spec || !src->r2 || !src->comfort_noise) return 0;

    dst->error_spec = src->error_spec;
    dst->echo_spec = src->echo_spec;
    dst->far_spec = src->far_spec;
    dst->near_spec = src->near_spec;
    dst->r2 = src->r2;
    dst->comfort_noise = src->comfort_noise;

    dst->far_power = src->far_power;
    dst->dt_indicator = src->dt_indicator;
    dst->saturation_level = src->saturation_level;
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
    int old_align;
    uint32_t t0;
    const Complex* shared_far_spec = NULL;   /* Group 6: set by lane 0, borrowed by lanes 1-3 */

    if (!p || p->destroyed ||
        !microphones_interleaved || !ref || !out)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (p->pending) return FOUR_AEC_NR_RES_SEQUENCE_ERROR;

    /* Cleared once the call is accepted: a hop that then bails out part-way
     * reports zero for the stages it never reached rather than the previous
     * hop's numbers. The three AEC-side fields are ACCUMULATED over the four
     * lanes below, so this is also what makes each hop start from zero. The
     * post half is cleared the same way in process_post_impl(). */
    p->last_timing.delay_us = 0;
    p->last_timing.frontend_us = 0;
    p->last_timing.linear_us = 0;
    p->last_timing.lane_res_us = 0;

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
         * cancellation survives and the WOLA sequences continue. The old full
         * aec_reset() + OLA wipe produced one near-zero output hop plus
         * dozens of hops of re-exposed echo -- the spectrogram vertical line
         * (regression: lib/aec test_external_realign.c and
         * tests/test_4aec_nr_res.c's realign continuity rows). */
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
                &p->snapshots[ch], &context, p->n_freqs)) {
            four_aec_nr_res_reset(p);
            return FOUR_AEC_NR_RES_DSP_ERROR;
        }
        for (i = 0; i < hop; ++i) {
            p->linear_interleaved[
                i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                context.formed_hop[i];
        }

        /* Sum this lane's own stage split into the hop's record. The four
         * lanes run sequentially, so summing gives the wall-clock cost of
         * each stage across the whole loop -- see FourAecNrResLastTiming. */
        {
            AecStageTiming lane_time;
            aec_get_last_timing(p->lanes[ch], &lane_time);
            /* Lane-internal delay work folds into the same delay_us the
             * shared estimator reports, because it is the same quantity:
             * time spent aligning the far end. It is structurally ZERO today
             * -- every lane is built AEC_DELAY_EXTERNAL_ALIGNED, so no lane
             * owns a ring or an estimator -- and is summed anyway so that a
             * future lane-mode change cannot silently drop its cost out of
             * the breakdown. */
            p->last_timing.delay_us    += lane_time.delay_us;
            p->last_timing.frontend_us += lane_time.frontend_us;
            p->last_timing.linear_us   += lane_time.linear_us;
            p->last_timing.lane_res_us += lane_time.res_us;
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
    /* Pipeline-owned per-hop output of align_render() above -- the exact
     * delay-aligned far every lane consumed this hop. Same one-frame
     * lifetime contract as linear_interleaved (see the header). */
    out->aligned_ref = p->aligned_ref;
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
    /* Seeded neutral rather than from lane 0, so lane 0 is subject to the
     * same contributes-to-the-output test as every other lane below. Both
     * quantities are non-negative, so 0 is the identity for the max. At
     * least one lane always survives that test: process_post*() rejects
     * all-zero weights before reaching here. */
    *max_dt = 0.0f;
    *max_saturation = 0.0f;
    *far_power = base_far_power;

    if (trusted_beamformed_error) {
        if (!complex_vector_finite(trusted_beamformed_error, n)) return 0;
        /* No copy into p->fused_error: its lifetime (owned by the caller,
         * e.g. audio_pipeline_4ch.c's p->gsc_spectrum) already covers every
         * read run_post_res_and_nr() does this call -- it reads
         * trusted_beamformed_error directly instead. */
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
        /* A lane the beamformer has zeroed in every bin contributes exactly
         * nothing to fused_error/fused_near/output_spec/fused_comfort below,
         * so it must not vote on the three scalars that steer the post stage
         * either -- those govern a spectrum this lane is absent from. The
         * degenerate case is real: with the external beamformer disabled the
         * bypass emits identity weights [1,0,0,0], and lanes 1-3 were still
         * driving max_dt, which selects the residual suppressor's floor.
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
    const Complex* trusted_beamformed_error,
    float* out) {
    const float* res_gain;
    const float* nr_extra;
    const Complex* error =
        trusted_beamformed_error ? trusted_beamformed_error : p->fused_error;
    int n = p->n_freqs;
    int hop = p->hop_size;
    int fft = p->fft_size;
    int k;
    float nf_eff;
    uint32_t t0, t1;

    /* RES stage: the power/reference preparation this gain depends on, plus
     * the suppression-gain computation itself. */
    t0 = four_aec_nr_res_now_us();
    four_aec_complex_mag2(p->error_power, error, n);
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
        p->cfg.delay_mode == AEC_DELAY_MATCHED
            ? delay_aec3_has_clockdrift(&p->shared_delay) : 0,
        max_saturation > 0.5f);
    /* One stamp closes RES and opens NR: the two statements between them are
     * a branch and a ternary, so a second clock read would cost more than the
     * gap it measures and leave that gap unattributed. */
    t1 = four_aec_nr_res_now_us();
    p->last_timing.res_us = t1 - t0;
    if (!res_gain) return 0;

    nr_extra = p->cfg.legacy_amin ? NULL : p->extra_noise;
    if (mmse_lsa_process_gain(
            p->nr, error, nr_extra, NULL) != 0)
        return 0;
    p->last_timing.nr_us = four_aec_nr_res_now_us() - t1;

    /* total_gain[k]=min(...) and the near_mean reduction below are mutually
     * independent (neither reads the other's output), so they share one
     * pass over n instead of two; the final echo_fraction/lift/output_spec
     * loop still has to stay separate -- nf_eff is a scalar derived from
     * near_mean plus the stateful near_hang hangover counter, so it must be
     * fully resolved before any per-bin lift can be computed. */
    {
        const float* nr_gain = mmse_lsa_get_gain(p->nr, NULL);
        float near_mean = 0.0f;
        for (k = 0; k < n; ++k) {
            p->total_gain[k] =
                fminf(nr_gain[k], res_gain[k]);
            near_mean += p->error_power[k];
        }
        near_mean /= (float)n;

        nf_eff = PROD_NE_FLOOR;
        if (!p->cfg.legacy_amin) {
            int far_active =
                far_power > PROD_FAR_GATE_THRESH;
            int near_active;
            if (near_mean > PROD_NEAR_GATE_THRESH)
                p->near_hang = p->near_hangover_frames;
            near_active = p->near_hang > 0;
            if (p->near_hang > 0) p->near_hang -= 1;
            nf_eff = (!far_active && near_active)
                ? PROD_NE_FLOOR
                : PROD_NE_FLOOR_FAR_ACTIVE;
        }
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
            error[k].r * p->total_gain[k];
        p->output_spec[k].i =
            error[k].i * p->total_gain[k];
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

    /* Synthesis: inverse transform, windowed overlap-add, and the hop
     * emit/shift. The finite check after it is validation, not synthesis, and
     * falls in the post-stage remainder along with the gain fusion, near-floor
     * gate and comfort-noise loop above. */
    t0 = four_aec_nr_res_now_us();
    fft_inverse(p->fft, p->output_spec, p->ifft_buffer);
    sk_wola_accumulate_f32(p->ola, p->ifft_buffer, p->synth_window, fft);
    memcpy(out, p->ola, (size_t)hop * sizeof(float));
    memmove(
        p->ola, p->ola + hop,
        (size_t)(fft - hop) * sizeof(float));
    memset(
        p->ola + (fft - hop), 0,
        (size_t)hop * sizeof(float));
    p->last_timing.synth_us = four_aec_nr_res_now_us() - t0;
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
    uint32_t t0;

    if (!p || p->destroyed || !token || !weights || !out ||
        !p->cfg.enable_post)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (!token_matches(p, token))
        return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    if (!validate_weights(p, weights))
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;

    /* Post half cleared here for the same reason process_pre() clears the
     * pre half: a hop that bails out reports zeros, not stale numbers. */
    p->last_timing.fuse_us = 0;
    p->last_timing.res_us = 0;
    p->last_timing.nr_us = 0;
    p->last_timing.synth_us = 0;

    /* Fuse stage: the beamformer-weighted projection of the four lanes. */
    t0 = four_aec_nr_res_now_us();
    if (!fuse_contexts(
            p, weights, trusted_beamformed_error,
            &all_converged, &max_dt,
            &max_saturation, &far_power)) {
        four_aec_nr_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    p->last_timing.fuse_us = four_aec_nr_res_now_us() - t0;
    if (!run_post_res_and_nr(
            p, all_converged, max_dt, max_saturation,
            far_power, trusted_beamformed_error, out)) {
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

int four_aec_nr_res_abandon_pre(
    FourAecNrRes* p,
    const FourAecNrResFrameToken* token) {
    if (!p || p->destroyed || !token)
        return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    if (!token_matches(p, token))
        return FOUR_AEC_NR_RES_SEQUENCE_ERROR;
    /* Same token-consumption epilogue as process_post_impl(), with none of
     * its RES/NR/synthesis work: releasing the frame is the whole job. */
    p->pending = 0;
    memset(&p->pending_token, 0, sizeof(p->pending_token));
    return FOUR_AEC_NR_RES_OK;
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

int four_aec_nr_res_post_split_floor(const FourAecNrRes* p, float* live,
                                     float* target) {
    if (!p || p->destroyed || !p->cfg.enable_post) return -1;
    if (live) *live = p->post_sg.split_floor_far_active_live;
    /* Report the RESET-SURVIVING copy, not post_sg.cfg: reset_post_sg()
     * rebuilds the suppressor from post_sg_cfg, so that is the value a caller
     * asking "what floor is this instance configured for" actually wants. */
    if (target) *target = p->post_sg_cfg.split_floor_far_active;
    return 0;
}

int four_aec_nr_res_set_aec_preset(FourAecNrRes* p, AecPreset preset,
                                   float ramp_ms) {
    float db;

    if (!p || p->destroyed) return -1;
    /* One lookup against the library's own strength table -- which also
     * refuses an out-of-enum value, where aec_config_from_preset() would fall
     * back to balanced. */
    if (aec_preset_floor_db(preset, &db) != 0) return -1;
    if (!p->cfg.enable_post) return -1;  /* no post suppressor to retarget */

    /* The four lanes run with spatial_linear_context, so they never reach
     * suppression_gain_get_gain() and their own floors shape nothing. The
     * gain that actually multiplies the output -- and scales the comfort
     * noise -- comes from the shared post-stage suppressor, so that is what
     * a preset change has to move. Validate and apply there FIRST: it refuses
     * without writing, so a rejected ramp_ms cannot leave the lanes and the
     * post stage disagreeing. */
    if (suppression_gain_set_split_floor_far_active_db(
            &p->post_sg, db, ramp_ms) != 0) {
        return -1;
    }
    /* post_sg_cfg is a separate by-value copy taken at init and re-applied by
     * reset_post_sg(); without this the next reset would silently revert the
     * change. */
    p->post_sg_cfg.split_floor_far_active =
        p->post_sg.cfg.split_floor_far_active;

    /* The lanes are deliberately left alone. Retargeting them would be inert
     * -- they never reach get_gain -- but not free: a ramped call would park
     * all four permanently mid-ramp, with live != target forever, which is
     * exactly the misleading state a diagnostic reader would trip over. The
     * mono pipeline and the Python twin take the same position. */
    p->cfg.aec_preset = preset;
    return 0;
}

int four_aec_nr_res_set_nr_mode(FourAecNrRes* p, MmseLsaNrMode mode) {
    MmseLsaConfig target;

    if (!p || p->destroyed) return -1;
    if (!p->cfg.enable_post || !p->nr) return -1;
    if (!mmse_lsa_nr_mode_is_valid(mode)) return -1;
    /* Recompose exactly what build_nr_config() composes -- the preset plus
     * this pipeline's own overrides. Handing the canonical preset to
     * mmse_lsa_set_mode() instead would either be refused (its L differs) or
     * silently revert those overrides. */
    target = pipelines_compose_nr_config(p->cfg.sample_rate, p->fft_size,
                               p->hop_size, mode);
    if (mmse_lsa_reconfigure(p->nr, &target) != 0) return -1;
    p->cfg.nr_mode = mode;
    return 0;
}

/* Legal to call while a pre-frame is pending (e.g. align_render()'s error
 * path above). If a caller is still holding a FourAecNrResPreFrame from that
 * pending frame, its linear_spectra[channel] pointers alias the per-lane
 * AecResContext buffers reset below (linear_interleaved is a separate,
 * pipeline-owned copy, memset below, not an alias). This file's own
 * token-generation bump (see the end of this function) is what keeps
 * process_post_impl() from reading linear_spectra afterward through this
 * instance's own API; it does not protect an external caller who kept a raw
 * pointer past this call -- aec_reset() carries no promise that a lane's old
 * spectrum content gets cleared or overwritten, so there is nothing for such
 * a caller to visibly notice going wrong. This is why FourAecNrResPreFrame's
 * doc comment requires callers to stop reading on reset() as an API-contract
 * rule, not on token rejection or on any memory-safety symptom.
 * Covered by tests/test_4aec_nr_res.c's reset-while-pending test. */
void four_aec_nr_res_reset(FourAecNrRes* p) {
    int ch;
    if (!p || p->destroyed) return;

    if (p->cfg.delay_mode == AEC_DELAY_MATCHED)
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
    p->accepted_delay =
        p->cfg.delay_mode == AEC_DELAY_FIXED
            ? p->cfg.fixed_delay_samples : 0;
    /* A reset abandons the alignment the quarantine was protecting, so a
     * countdown armed against it must not survive. The WINDOW
     * (delay_quarantine_hops) is config-derived and stays. Same for a held
     * change candidate and the life left on it: it was a movement away from
     * an alignment that no longer exists, and the next acquisition is
     * immediate anyway. */
    p->delay_quarantine_left = -1;
    p->delay_admission.candidate = 0;
    p->delay_admission.ttl = 0;
    /* Zeroed with the lanes' own counters (aec_reset() clears the per-lane
     * far-FFT count), so every instrumentation total shares one epoch. */
    p->realign_warm_lanes = 0;
    p->realign_soft_lanes = 0;
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
    return p && !p->destroyed &&
           p->cfg.delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
}

int four_aec_nr_res_linear_aec_count(const FourAecNrRes* p) {
    return p && !p->destroyed ? FOUR_AEC_NR_RES_CHANNELS : 0;
}

int four_aec_nr_res_nr_count(const FourAecNrRes* p) {
    return p && !p->destroyed && p->cfg.enable_post ? 1 : 0;
}

int four_aec_nr_res_post_res_count(const FourAecNrRes* p) {
    return p && !p->destroyed && p->cfg.enable_post ? 1 : 0;
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
    size_t delay_estimator_bytes;
    int ch;

    if (!out || !derive_dims_and_configs(
            cfg, &aec_cfg, &nr_cfg, &fft, &hop, &n,
            &post_ma_n, &delay_ring_size,
            &delay_estimator_bytes)) return -1;

    aec_one = aec_get_mem_size(&aec_cfg);
    nr_bytes = cfg->enable_post ? mmse_lsa_get_mem_size(&nr_cfg) : 0;
    fft_bytes = cfg->enable_post ? fft_get_mem_size(fft) : 0;
    wrapper_storage = pipeline_buffer_size(
        hop, fft, n, post_ma_n, delay_ring_size, delay_estimator_bytes,
        cfg->enable_post);
    if (aec_one == 0 || wrapper_storage == 0 ||
        (cfg->enable_post && (nr_bytes == 0 || fft_bytes == 0))) return -1;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        aec_total = ck_add_size(aec_total, ck_align16_size(aec_one));
    wrapper_bytes = ck_add_size(
        ck_align16_size(sizeof(FourAecNrRes)), wrapper_storage);
    total = wrapper_bytes;
    total = ck_add_size(total, aec_total);
    if (cfg->enable_post) {
        total = ck_add_size(total, ck_align16_size(nr_bytes));
        total = ck_add_size(total, ck_align16_size(fft_bytes));
    }
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
