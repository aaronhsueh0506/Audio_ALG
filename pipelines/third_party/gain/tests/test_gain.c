/**
 * gain/tests/test_gain.c — equivalence tests for the gain modules
 * (fix_gain, nr_gain, post_gain).
 *
 * The PHAT/BF/SRP/VAD tests live in audiolib/tests/, which does not link
 * libgain.a, so the gain tests live here with gain/ and have their own small
 * driver. The shared
 * `check_pool_contract` helper (and its CHECK_L macro and PoolInitFn /
 * PoolDestroyFn typedefs) is duplicated verbatim from that file rather than
 * shared, to keep this split a plain file move plus copy -- no new shared
 * header was introduced.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fix_gain.h"
#include "nr_gain.h"
#include "post_gain.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            return 0;                                                  \
        }                                                              \
    } while (0)

static unsigned int test_rng = 0x12345678u;

static float random_signed(void) {
    test_rng = test_rng * 1664525u + 1013904223u;
    return ((float)((test_rng >> 8) & 0x00ffffffu) /
            8388608.0f) - 1.0f;
}

/* ---------- the caller-pool contract, checked once ----------
 *
 * Every module here implements the same one: an undersized or misaligned
 * block is refused without a single byte written, a legal block puts the
 * instance at mem[0] and is never written past `need`, and destroy() leaves
 * the block reusable. */
typedef void* (*PoolInitFn)(void* mem, size_t bytes, void* ctx);
typedef void (*PoolDestroyFn)(void* inst);

/* CHECK_L rather than CHECK: one shared body serves four modules, so a bare
 * "init on a legal pool" would not say which one failed. */
#define CHECK_L(condition, message)                                        \
    do {                                                                   \
        if (!(condition)) {                                                \
            fprintf(stderr, "FAIL: %s: %s (line %d)\n",                    \
                    label, message, __LINE__);                             \
            return 0;                                                      \
        }                                                                  \
    } while (0)

static int check_pool_contract(const char* label, size_t need,
                               PoolInitFn init, PoolDestroyFn destroy,
                               void* ctx) {
    size_t slack = 256;
    size_t pool_bytes = need + slack;
    uint8_t* pool = NULL;
    void* inst;
    size_t i;

    CHECK_L(need > 0, "get_mem_size must size a legal config");
    CHECK_L(posix_memalign((void**)&pool, 16, pool_bytes) == 0 && pool,
            "pool allocation");
    memset(pool, 0xa5, pool_bytes);

    CHECK_L(init(pool, need - 1, ctx) == NULL,
            "undersized pool must be refused");
    CHECK_L(init(pool + 1, pool_bytes - 1, ctx) == NULL,
            "misaligned pool must be refused");
    /* A refusal must write NOTHING -- the whole pool is still poison. */
    for (i = 0; i < pool_bytes; ++i) {
        CHECK_L(pool[i] == 0xa5u, "a refused init wrote into the pool");
    }

    inst = init(pool, pool_bytes, ctx);
    CHECK_L(inst != NULL, "init on a legal pool");
    CHECK_L(inst == (void*)pool, "the instance must sit at mem[0]");

    /* Only the budgeted region may be touched: everything past `need` must
     * still hold the poison. */
    for (i = need; i < pool_bytes; ++i) {
        CHECK_L(pool[i] == 0xA5u, "init wrote past its own budget");
    }

    destroy(inst);   /* must NOT free caller memory */
    memset(pool, 0x5a, 1);   /* still ours to write */
    /* The same block must take a second init -- destroy left nothing behind. */
    CHECK_L(init(pool, pool_bytes, ctx) != NULL,
            "the pool must be reusable after destroy");
    free(pool);
    printf("PASS %s pool-first poison/bounds\n", label);
    return 1;
}

static void* pool_init_fix_gain(void* mem, size_t bytes, void* ctx) {
    return fix_gain_init((const FixGainConfig*)ctx, mem, bytes);
}
static void pool_destroy_fix_gain(void* inst) {
    fix_gain_destroy((FixGain*)inst);
}

static void* pool_init_nr_gain(void* mem, size_t bytes, void* ctx) {
    (void)ctx;   /* the instance is one float; no config affects its size */
    return nr_gain_init(mem, bytes);
}
static void pool_destroy_nr_gain(void* inst) { nr_gain_destroy((NrGain*)inst); }

static void* pool_init_post_gain(void* mem, size_t bytes, void* ctx) {
    return post_gain_init((const PostGainConfig*)ctx, mem, bytes);
}
static void pool_destroy_post_gain(void* inst) {
    post_gain_destroy((PostGainState*)inst);
}

/* ===================== gain library ===================== */

/* Verbatim transcription of a fused scale-and-clip loop, the reference for
 * fix_gain_process(). The module must produce these exact bytes. */
static void fix_gain_fused_reference(float* x, int n, float gain,
                                     int do_clip, float lim) {
    int i;
    for (i = 0; i < n; i++) {
        float y = x[i] * gain;
        if (do_clip) {
            if (y > lim) y = lim;
            else if (y < -lim) y = -lim;
        }
        x[i] = y;
    }
}

static FixGainConfig fix_gain_test_config(const float* per_channel) {
    FixGainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 4;
    cfg.enable = 1;
    cfg.global_gain = 2.5f;
    cfg.channel_gain = per_channel;
    cfg.enable_clip = 1;
    cfg.clip_value = 0.75f;
    return cfg;
}

static int test_fix_gain_pool_poison_and_bounds(void) {
    const float per_channel[4] = { 1.0f, 0.5f, 2.0f, 0.125f };
    FixGainConfig cfg = fix_gain_test_config(per_channel);
    float bad_channel[4] = { 1.0f, 0.5f, 2.0f, 0.125f };

    CHECK(check_pool_contract("fix_gain", fix_gain_get_mem_size(&cfg),
                              pool_init_fix_gain, pool_destroy_fix_gain, &cfg),
          "fix_gain pool contract");

    /* get_mem_size doubles as the config gate: a config it will not construct
     * must size to 0 rather than produce an instance that turns the signal
     * into NaN. */
    {
        FixGainConfig bad = cfg;
        bad.channels = 0;
        CHECK(fix_gain_get_mem_size(&bad) == 0, "channels == 0 must not size");
        CHECK(fix_gain_create(&bad) == NULL, "channels == 0 must not construct");

        bad = cfg;
        bad.global_gain = (float)NAN;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "a non-finite global gain must not size");
        bad.global_gain = (float)INFINITY;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "an infinite global gain must not size");

        bad = cfg;
        bad.clip_value = (float)NAN;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "a non-finite clip value must not size");

        bad = cfg;
        bad_channel[2] = (float)INFINITY;
        bad.channel_gain = bad_channel;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "a non-finite per-channel gain must not size");
        CHECK(fix_gain_create(&bad) == NULL,
              "a non-finite per-channel gain must not construct");
    }

    /* fix_gain_db_to_linear does not validate; the config gate above is what
     * catches what it can produce. */
    CHECK(fix_gain_db_to_linear(0.0f) == 1.0f, "0 dB is unity");
    CHECK(!isfinite(fix_gain_db_to_linear((float)INFINITY)),
          "an infinite dB value converts to a non-finite gain");

    printf("PASS fix_gain config gate rejects non-finite tuning\n");
    return 1;
}

static int test_fix_gain_block_and_per_sample_match_reference(void) {
    const int n = 257;                  /* not a multiple of 4: kernel tail */
    const float per_channel[4] = { 1.0f, 0.5f, 2.0f, 0.125f };
    FixGainConfig cfg = fix_gain_test_config(per_channel);
    size_t need = fix_gain_get_mem_size(&cfg);
    uint8_t* raw = (uint8_t*)malloc(need + 16);
    uint8_t* pool;
    FixGain* heap;
    FixGain* fromPool;
    float* src = (float*)malloc((size_t)n * sizeof(float));
    float* got_block = (float*)malloc((size_t)n * sizeof(float));
    float* got_single = (float*)malloc((size_t)n * sizeof(float));
    float* got_pool = (float*)malloc((size_t)n * sizeof(float));
    float* want = (float*)malloc((size_t)n * sizeof(float));
    int ch, i, clipped = 0;

    CHECK(raw && src && got_block && got_single && got_pool && want,
          "allocation");
    pool = (uint8_t*)(((uintptr_t)raw + 15u) & ~(uintptr_t)15u);

    heap = fix_gain_create(&cfg);
    fromPool = fix_gain_init(&cfg, pool, need);
    CHECK(heap && fromPool, "both constructors must succeed");

    test_rng = 0x0f1e2d3cu;
    for (i = 0; i < n; ++i) src[i] = random_signed() * 1.5f;

    for (ch = 0; ch < cfg.channels; ++ch) {
        const float gain = cfg.global_gain * per_channel[ch];

        memcpy(want, src, (size_t)n * sizeof(float));
        fix_gain_fused_reference(want, n, gain, cfg.enable_clip,
                                 cfg.clip_value);

        memcpy(got_block, src, (size_t)n * sizeof(float));
        fix_gain_process(heap, got_block, n, ch);

        memcpy(got_pool, src, (size_t)n * sizeof(float));
        fix_gain_process(fromPool, got_pool, n, ch);

        /* The integrator drives this module one sample at a time. */
        memcpy(got_single, src, (size_t)n * sizeof(float));
        for (i = 0; i < n; ++i) fix_gain_process(heap, got_single + i, 1, ch);

        CHECK(memcmp(got_block, want, (size_t)n * sizeof(float)) == 0,
              "block call must be byte-identical to the fused reference");
        CHECK(memcmp(got_single, want, (size_t)n * sizeof(float)) == 0,
              "per-sample calls must be byte-identical to the block call");
        CHECK(memcmp(got_pool, want, (size_t)n * sizeof(float)) == 0,
              "pool instance must be byte-identical to the heap instance");

        for (i = 0; i < n; ++i) {
            if (fabsf(want[i]) >= cfg.clip_value) clipped++;
        }
    }

    /* ⚠ Written so it can FAIL: with no sample driven past clip_value the
     * comparisons above would never exercise the clip path at all. */
    CHECK(clipped > 0, "the driver never reached the clip limit -- "
                       "the equivalence above would prove nothing");

    /* Out-of-range channel, n <= 0 and a disabled instance are no-ops. */
    memcpy(got_block, src, (size_t)n * sizeof(float));
    fix_gain_process(heap, got_block, n, cfg.channels);
    fix_gain_process(heap, got_block, 0, 0);
    CHECK(memcmp(got_block, src, (size_t)n * sizeof(float)) == 0,
          "an out-of-range channel or empty block must not touch the signal");

    fix_gain_destroy(heap);
    fix_gain_destroy(fromPool);
    free(want); free(got_pool); free(got_single); free(got_block);
    free(src); free(raw);
    printf("PASS fix_gain block/per-sample/pool match the fused reference\n");
    return 1;
}

/* Verbatim transcription of nr_gain_process()'s smoother. */
static float nr_gain_reference(float prev, const NrGainConfig* cfg,
                               float doa_gain) {
    float min_gain = cfg->min_gain;
    float max_gain = cfg->max_gain;
    float target_gain, alpha, out;

    if (!cfg->enable) return 1.0f;
    if (cfg->only_when_nr_enable && !cfg->nr_enable) return 1.0f;

    if (min_gain < 0.0f) min_gain = 0.0f;
    if (max_gain < min_gain) max_gain = min_gain;

    target_gain = isnan(doa_gain) ? cfg->noise_gain : cfg->target_gain;
    if (target_gain < min_gain) target_gain = min_gain;
    else if (target_gain > max_gain) target_gain = max_gain;

    alpha = (target_gain > prev) ? cfg->attack_alpha : cfg->release_alpha;
    if (alpha < 0.0f) alpha = 0.0f;
    else if (alpha > 0.9999f) alpha = 0.9999f;

    out = alpha * prev + (1.0f - alpha) * target_gain;
    if (out < min_gain) out = min_gain;
    else if (out > max_gain) out = max_gain;
    return out;
}

static int test_nr_gain_pool_and_smoother_reference(void) {
    const int n = 130;
    const int hops = 64;
    size_t need = nr_gain_get_mem_size();
    size_t pool_bytes = need + 128;
    uint8_t* pool = NULL;
    NrGain* ng;
    NrGain* heap;
    NrGainConfig cfg;
    float* x = (float*)malloc((size_t)n * sizeof(float));
    float* y = (float*)malloc((size_t)n * sizeof(float));
    float ref = 1.0f;
    int t, i, rose = 0, fell = 0;

    CHECK(need > 0 && x && y, "sizing/allocation");
    CHECK(check_pool_contract("nr_gain", need,
                              pool_init_nr_gain, pool_destroy_nr_gain, NULL),
          "nr_gain pool contract");

    CHECK(posix_memalign((void**)&pool, 16, pool_bytes) == 0 && pool,
          "pool allocation");
    ng = nr_gain_init(pool, pool_bytes);
    heap = nr_gain_create();
    CHECK(ng && heap, "both constructors must succeed");
    CHECK(nr_gain_get_gain(ng) == 1.0f, "a fresh instance starts at unity");

    memset(&cfg, 0, sizeof(cfg));
    cfg.enable = 1;
    cfg.only_when_nr_enable = 1;
    cfg.nr_enable = 1;
    cfg.target_gain = 2.0f;
    cfg.noise_gain = 0.25f;
    cfg.min_gain = 0.1f;
    cfg.max_gain = 4.0f;
    cfg.attack_alpha = 0.4f;
    cfg.release_alpha = 0.85f;

    test_rng = 0x5150c0deu;
    for (t = 0; t < hops; ++t) {
        /* Alternate between a located source and no estimate so both the
         * attack and the release branch run. */
        float doa_gain = (t % 8 < 4) ? 1.0f : (float)NAN;
        float got_pool, got_heap, prev = ref;

        for (i = 0; i < n; ++i) x[i] = random_signed();
        memcpy(y, x, (size_t)n * sizeof(float));

        got_pool = nr_gain_process(ng, &cfg, x, n, doa_gain);
        got_heap = nr_gain_process(heap, &cfg, y, n, doa_gain);
        ref = nr_gain_reference(prev, &cfg, doa_gain);

        CHECK(got_pool == ref, "smoothed gain must match the reference");
        CHECK(got_heap == ref, "heap instance must track the pool instance");
        CHECK(nr_gain_get_gain(ng) == ref, "the getter must report it too");
        CHECK(memcmp(x, y, (size_t)n * sizeof(float)) == 0,
              "heap and pool must scale the hop identically");

        if (ref > prev) rose++;
        if (ref < prev) fell++;
    }

    /* ⚠ Written so it can FAIL: a driver that only ever rose (or only fell)
     * would leave one of the two coefficients untested. */
    CHECK(rose > 0 && fell > 0,
          "the driver never exercised both attack and release");

    /* A disabled config pins the gain back to unity. */
    cfg.enable = 0;
    CHECK(nr_gain_process(ng, &cfg, NULL, 0, 1.0f) == 1.0f,
          "a disabled config must pin the gain to unity");
    cfg.enable = 1;
    cfg.nr_enable = 0;
    CHECK(nr_gain_process(ng, &cfg, NULL, 0, 1.0f) == 1.0f,
          "only_when_nr_enable must pin the gain while NR is off");

    nr_gain_destroy(ng);
    nr_gain_destroy(heap);
    free(pool); free(y); free(x);
    printf("PASS nr_gain smoother matches the reference\n");
    return 1;
}

/* ---------- post_gain: the straightforward formulation, transcribed ----------
 *
 * An independent re-implementation of post_gain_apply() without the
 * hoisted per-bin clamps or the sk_clip_f32/sk_capply_gain_f32 stage-5
 * clip/apply: per-bin clamp of a frame-constant gain, a copy of the
 * caller's mask even when relaxation is off, and one fused
 * smooth-clamp-apply loop. Byte-equality against this shows the module's
 * restructured form computes the same values. */
typedef struct {
    int F;
    float* prev_gain;
    float* target_gain;
    float* freq_gain;
    int* mask_work;
    int* raw_mask_frame;
    int* class_frame;
    int cnt_match;
    int cnt_suppress;
    /* How many times the final clamp actually moved a value. The bounds are
     * already applied to the frame-constant gains upstream, so in steady
     * state this stage is a no-op -- it only bites after the caller tightens
     * min_gain/max_gain under a smoother still holding older, looser values.
     * Counted so the equivalence below cannot quietly stop covering it. */
    int clip_hits;
} PgRef;

/* pg_ref_angle_near and pg_ref_angle_to_index are transcribed UNCHANGED from
 * the module, so these two are a mirror, not a gate -- a bug in the
 * module's angle maths would be copied here and compare equal. What this
 * reference does gate is every other stage: the mask-relaxation branch,
 * the hoisted clamps, the split box filter and the fused
 * smooth-clamp-apply. */
static int pg_ref_angle_near(int a, int b, int num_angles, int tol) {
    int d;
    if (a < 0 || b < 0) return 0;
    d = abs(a - b);
    if (d > num_angles / 2) d = num_angles - d;
    return (d <= tol);
}

static int pg_ref_angle_to_index(float doa_rad, int num_angles) {
    float two_pi, idx_f;
    if (num_angles <= 0 || !isfinite(doa_rad)) return -1;
    two_pi = 2.0f * (float)M_PI;
    while (doa_rad < 0.0f) doa_rad += two_pi;
    while (doa_rad >= two_pi) doa_rad -= two_pi;
    idx_f = doa_rad / two_pi * (float)num_angles;
    return (int)roundf(idx_f) % num_angles;
}

static float pg_ref_clamp(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static int pg_ref_init(PgRef* r, const PostGainConfig* cfg) {
    int f;
    r->F = cfg->F;
    r->prev_gain = (float*)malloc((size_t)r->F * sizeof(float));
    r->target_gain = (float*)malloc((size_t)r->F * sizeof(float));
    r->freq_gain = (float*)malloc((size_t)r->F * sizeof(float));
    r->mask_work = (int*)malloc((size_t)r->F * sizeof(int));
    r->raw_mask_frame = (int*)malloc((size_t)r->F * sizeof(int));
    r->class_frame = (int*)malloc((size_t)r->F * sizeof(int));
    if (!r->prev_gain || !r->target_gain || !r->freq_gain || !r->mask_work ||
        !r->raw_mask_frame || !r->class_frame) return 0;
    for (f = 0; f < r->F; f++) {
        r->prev_gain[f] = cfg->gain_match;
        r->target_gain[f] = cfg->gain_match;
        r->freq_gain[f] = cfg->gain_match;
        r->mask_work[f] = 0;
        r->raw_mask_frame[f] = 0;
        r->class_frame[f] = 0;
    }
    r->cnt_match = 0;
    r->cnt_suppress = 0;
    r->clip_hits = 0;
    return 1;
}

static void pg_ref_free(PgRef* r) {
    free(r->prev_gain); free(r->target_gain); free(r->freq_gain);
    free(r->mask_work); free(r->raw_mask_frame); free(r->class_frame);
}

static void pg_ref_apply(PgRef* st, const PostGainConfig* cfg, Complex* Y,
                         const int* mask, const int* bin_best_idx,
                         float doa_used) {
    const int* use_mask;
    int F = st->F;
    int num_angles, doa_used_idx, cnt_match = 0, cnt_suppress = 0;
    int angle_match_cnt = 0, angle_vad = 0, f;
    float angle_ratio;

    if (!cfg->enable) {
        for (f = 0; f < F; f++) {
            st->raw_mask_frame[f] = 0;
            st->class_frame[f] = 0;
            st->target_gain[f] = 1.0f;
            st->freq_gain[f] = 1.0f;
            st->prev_gain[f] = 1.0f;
        }
        st->cnt_match = 0;
        st->cnt_suppress = 0;
        return;
    }

    num_angles = cfg->num_angles;
    doa_used_idx = pg_ref_angle_to_index(doa_used, num_angles);
    if (doa_used_idx < 0) {
        for (f = 0; f < F; f++) {
            st->raw_mask_frame[f] = 0;
            st->class_frame[f] = 0;
        }
        st->cnt_match = 0;
        st->cnt_suppress = F;
        return;
    }

    if (cfg->enable_mask_relax && cfg->mask_relax_bins > 0) {
        int r = cfg->mask_relax_bins;
        for (f = 0; f < F; f++) {
            int keep = 0, k;
            for (k = -r; k <= r; k++) {
                int ff = f + k;
                if (ff >= 0 && ff < F && mask[ff]) { keep = 1; break; }
            }
            st->mask_work[f] = keep;
        }
        use_mask = st->mask_work;
    } else {
        for (f = 0; f < F; f++) st->mask_work[f] = mask[f];
        use_mask = st->mask_work;
    }

    for (f = 0; f < F; f++) {
        int angle_match = 0;
        if (use_mask[f] && pg_ref_angle_near(bin_best_idx[f], doa_used_idx,
                                             num_angles, cfg->angle_tol)) {
            angle_match = 1;
            angle_match_cnt++;
        }
        st->raw_mask_frame[f] = angle_match;
    }

    angle_ratio = (float)angle_match_cnt / (float)F;
    if (angle_ratio > cfg->angle_vad_thr) angle_vad = 1;

    for (f = 0; f < F; f++) {
        int angle_match = st->raw_mask_frame[f];
        int cls = 0;
        float gain;
        if (angle_vad && angle_match) {
            gain = cfg->gain_match; cls = 2; cnt_match++;
        } else {
            gain = cfg->gain_suppress; cls = 0; cnt_suppress++;
        }
        gain = pg_ref_clamp(gain, cfg->min_gain, cfg->max_gain);
        st->target_gain[f] = gain;
        st->class_frame[f] = cls;
    }

    if (cfg->enable_freq_smooth && cfg->freq_smooth_radius > 0) {
        int r = cfg->freq_smooth_radius;
        for (f = 0; f < F; f++) {
            float sum = 0.0f;
            int count = 0, k;
            for (k = -r; k <= r; k++) {
                int ff = f + k;
                if (ff >= 0 && ff < F) { sum += st->target_gain[ff]; count++; }
            }
            st->freq_gain[f] = sum / (float)count;
        }
    } else {
        for (f = 0; f < F; f++) st->freq_gain[f] = st->target_gain[f];
    }

    for (f = 0; f < F; f++) {
        float target = st->freq_gain[f];
        float prev = st->prev_gain[f];
        float smooth_gain;
        if (cfg->enable_time_smooth) {
            float alpha = (target > prev) ? cfg->attack_alpha
                                          : cfg->release_alpha;
            smooth_gain = alpha * prev + (1.0f - alpha) * target;
        } else {
            smooth_gain = target;
        }
        {
            float clamped = pg_ref_clamp(smooth_gain, cfg->min_gain,
                                         cfg->max_gain);
            if (clamped != smooth_gain) st->clip_hits++;
            smooth_gain = clamped;
        }
        Y[f].r *= smooth_gain;
        Y[f].i *= smooth_gain;
        st->prev_gain[f] = smooth_gain;
    }

    st->cnt_match = cnt_match;
    st->cnt_suppress = cnt_suppress;
}

static PostGainConfig post_gain_test_config(int F, int relax, int freq_smooth) {
    PostGainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.F = F;
    cfg.num_angles = 72;
    cfg.enable = 1;
    cfg.gain_match = 1.0f;
    cfg.gain_suppress = 0.25f;
    cfg.angle_tol = 3;
    cfg.angle_vad_thr = 0.1f;
    cfg.enable_freq_smooth = freq_smooth;
    cfg.freq_smooth_radius = 2;
    cfg.enable_time_smooth = 1;
    cfg.attack_alpha = 0.3f;
    cfg.release_alpha = 0.8f;
    cfg.min_gain = 0.3f;      /* below gain_suppress: the clamp must bite */
    cfg.max_gain = 0.9f;      /* below gain_match: ... at both ends */
    cfg.enable_mask_relax = relax;
    cfg.mask_relax_bins = 1;
    return cfg;
}

static int test_post_gain_pool_poison_and_bounds(void) {
    PostGainConfig cfg = post_gain_test_config(65, 1, 1);
    PostGainConfig bad = cfg;

    CHECK(check_pool_contract("post_gain", post_gain_get_mem_size(&cfg),
                              pool_init_post_gain, pool_destroy_post_gain,
                              &cfg),
          "post_gain pool contract");

    bad.F = 0;
    CHECK(post_gain_get_mem_size(&bad) == 0, "F == 0 must not size");
    CHECK(post_gain_create(&bad) == NULL, "F == 0 must not construct");
    printf("PASS post_gain config gate\n");
    return 1;
}

static int post_gain_drive_case(int relax, int freq_smooth,
                                int* out_vad_hits, int* out_clip_hits) {
    const int F = 65;                   /* not a multiple of 4: kernel tail */
    const int frames = 48;
    PostGainConfig cfg = post_gain_test_config(F, relax, freq_smooth);
    size_t need = post_gain_get_mem_size(&cfg);
    uint8_t* raw = (uint8_t*)malloc(need + 16);
    uint8_t* pool;
    PostGainState* heap;
    PostGainState* fromPool;
    const float* heap_gain;
    PgRef ref;
    Complex* Y_heap = (Complex*)malloc((size_t)F * sizeof(Complex));
    Complex* Y_pool = (Complex*)malloc((size_t)F * sizeof(Complex));
    Complex* Y_ref = (Complex*)malloc((size_t)F * sizeof(Complex));
    int* mask = (int*)malloc((size_t)F * sizeof(int));
    int* bin_best = (int*)malloc((size_t)F * sizeof(int));
    int t, f;

    CHECK(raw && Y_heap && Y_pool && Y_ref && mask && bin_best, "allocation");
    pool = (uint8_t*)(((uintptr_t)raw + 15u) & ~(uintptr_t)15u);

    heap = post_gain_create(&cfg);
    fromPool = post_gain_init(&cfg, pool, need);
    CHECK(heap && fromPool, "both constructors must succeed");
    CHECK(pg_ref_init(&ref, &cfg), "reference allocation");

    test_rng = 0x13579bdfu;
    for (t = 0; t < frames; ++t) {
        /* Sweep the steered angle so some frames clear the angle-VAD ratio
         * and some do not. */
        float doa = (float)t * 0.13f;
        const PostGainStats* stats;

        /* Half way through, tighten the gain window under a smoother still
         * carrying values from the looser one, so the final clamp has
         * something to do. */
        if (t == frames / 2) {
            cfg.min_gain = 0.7f;
            cfg.max_gain = 0.75f;
        }

        for (f = 0; f < F; ++f) {
            Y_heap[f].r = random_signed() * 2.0f;
            Y_heap[f].i = random_signed() * 2.0f;
            mask[f] = (random_signed() > -0.2f) ? 1 : 0;
            bin_best[f] = (int)((random_signed() * 0.5f + 0.5f) * 72.0f) % 72;
        }
        memcpy(Y_pool, Y_heap, (size_t)F * sizeof(Complex));
        memcpy(Y_ref, Y_heap, (size_t)F * sizeof(Complex));

        post_gain_apply(heap, &cfg, Y_heap, mask, bin_best, doa);
        post_gain_apply(fromPool, &cfg, Y_pool, mask, bin_best, doa);
        pg_ref_apply(&ref, &cfg, Y_ref, mask, bin_best, doa);

        CHECK(memcmp(Y_heap, Y_ref, (size_t)F * sizeof(Complex)) == 0,
              "shaped spectrum must be byte-identical to the reference");
        CHECK(memcmp(Y_pool, Y_ref, (size_t)F * sizeof(Complex)) == 0,
              "pool instance must be byte-identical to the reference");
        /* target_gain and freq_gain are internal to the module; they
         * both feed prev_gain, which is public and compared here, and the
         * shaped spectrum above. */
        heap_gain = post_gain_get_gain(heap);
        CHECK(heap_gain && memcmp(heap_gain, ref.prev_gain,
                                  (size_t)F * sizeof(float)) == 0,
              "applied per-bin gain must be byte-identical");
        CHECK(memcmp(post_gain_get_raw_mask(heap), ref.raw_mask_frame,
                     (size_t)F * sizeof(int)) == 0,
              "raw directional mask must be byte-identical");
        CHECK(memcmp(post_gain_get_class(heap), ref.class_frame,
                     (size_t)F * sizeof(int)) == 0,
              "per-bin class must be byte-identical");
        stats = post_gain_get_stats(heap);
        CHECK(stats->cnt_match == ref.cnt_match &&
              stats->cnt_suppress == ref.cnt_suppress,
              "frame stats must match the reference");

        if (ref.cnt_match > 0) (*out_vad_hits)++;
    }

    /* The bypass path resets state rather than shaping. */
    cfg.enable = 0;
    post_gain_apply(heap, &cfg, Y_heap, mask, bin_best, 0.25f);
    pg_ref_apply(&ref, &cfg, Y_ref, mask, bin_best, 0.25f);
    CHECK(memcmp(post_gain_get_gain(heap), ref.prev_gain,
                 (size_t)F * sizeof(float)) == 0,
          "bypass must reset the gain state exactly like the reference");
    cfg.enable = 1;

    /* A steering angle that is not finite short-circuits before any gain is
     * applied. +/-inf matters as much as NaN: the module reduces the angle by
     * repeated subtraction, so an unguarded infinity spins forever -- this
     * assertion only returns at all because the guard tests isfinite(). */
    {
        const float bad_angles[3] = { (float)NAN, (float)INFINITY,
                                      -(float)INFINITY };
        int b;
        for (b = 0; b < 3; ++b) {
            memcpy(Y_pool, Y_heap, (size_t)F * sizeof(Complex));
            post_gain_apply(heap, &cfg, Y_heap, mask, bin_best, bad_angles[b]);
            CHECK(memcmp(Y_heap, Y_pool, (size_t)F * sizeof(Complex)) == 0,
                  "an unusable steering angle must leave the spectrum "
                  "untouched");
            CHECK(post_gain_get_stats(heap)->cnt_suppress == F,
                  "an unusable steering angle must classify the frame "
                  "all-suppress");
        }
    }

    *out_clip_hits += ref.clip_hits;

    pg_ref_free(&ref);
    post_gain_destroy(heap);
    post_gain_destroy(fromPool);
    free(bin_best); free(mask); free(Y_ref); free(Y_pool); free(Y_heap);
    free(raw);
    return 1;
}

static int test_post_gain_matches_prekernel_reference(void) {
    int vad_hits = 0;
    int clip_hits = 0;

    CHECK(post_gain_drive_case(1, 1, &vad_hits, &clip_hits),
          "mask relaxation + frequency smoothing");
    CHECK(post_gain_drive_case(0, 0, &vad_hits, &clip_hits),
          "no relaxation, no frequency smoothing");

    /* ⚠ Written so it can FAIL: if no frame ever produced a matched bin the
     * comparisons above would only ever have compared the suppress branch. */
    CHECK(vad_hits > 0, "the driver never produced a directional match -- "
                        "the equivalence above would prove nothing");
    /* ⚠ Likewise for the output clamp: without a frame whose smoothed gain
     * lands outside the window, dropping that stage entirely would still
     * compare equal. */
    CHECK(clip_hits > 0, "the driver never drove a gain outside the window -- "
                         "the output clamp would be untested");
    printf("PASS post_gain matches the pre-kernel reference "
           "over both smoothing configurations\n");
    return 1;
}

static int run_all_tests(void) {
    CHECK(test_fix_gain_pool_poison_and_bounds(),
          "fix_gain pool-first poison/bounds test");
    CHECK(test_fix_gain_block_and_per_sample_match_reference(),
          "fix_gain fused-reference equivalence test");
    CHECK(test_nr_gain_pool_and_smoother_reference(),
          "nr_gain pool-first bounds and smoother-reference test");
    CHECK(test_post_gain_pool_poison_and_bounds(),
          "post_gain pool-first poison/bounds test");
    CHECK(test_post_gain_matches_prekernel_reference(),
          "post_gain pre-kernel-reference equivalence test");
    printf("All third-party gain tests passed\n");
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
