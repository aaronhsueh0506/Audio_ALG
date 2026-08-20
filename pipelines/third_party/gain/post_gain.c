#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "post_gain.h"
#include "fast_math.h"
#include "mem_align.h"
#include "simd_kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct PostGainState {
    int F;

    float* prev_gain;
    float* target_gain;
    float* freq_gain;

    /* Relaxed copy of the caller's mask. Written only when mask relaxation is
     * enabled; otherwise post_gain_apply() reads the caller's mask directly
     * and this buffer holds whatever the last relaxed frame left in it. */
    int* mask_work;

    /* per-frame log buffers */
    int* raw_mask_frame;
    int* class_frame;

    PostGainStats stats;

    /* Non-NULL only on the post_gain_create() heap path: the single block
     * backing this struct and every array above, freed by
     * post_gain_destroy(). NULL on the post_gain_init() caller-pool path. */
    void* owned_heap;
};

static int angle_idx_is_near(int a, int b, int num_angles, int tol)
{
    int d;

    if (a < 0 || b < 0) return 0;

    d = abs(a - b);
    if (d > num_angles / 2)
        d = num_angles - d;

    return (d <= tol);
}

/* Not srp_angle_to_index(): that one needs an SRP instance this module never
 * holds, and it rounds through degrees (deg/step) where this rounds in
 * radians, so the two disagree on bins near a step boundary (measured: 208 of
 * 2.4M sampled angles). Sharing it would be a behaviour change dressed up as
 * reuse.
 *
 * The finite test covers +/-inf as well as NaN, and must: the reduction below
 * is repeated subtraction, so an infinite input would spin forever, and a
 * caller passing an un-wrapped accumulated angle pays O(|doa_rad|/2pi) laps.
 * fmodf() would bound the latter but rounds differently, which the module's
 * byte-equality test would reject -- so the loops stay and the guard is what
 * makes them safe. */
static int post_gain_angle_to_index(float doa_rad, int num_angles)
{
    float two_pi, idx_f;
    int idx;

    if (num_angles <= 0 || !isfinite(doa_rad)) {
        return -1;
    }

    two_pi = 2.0f * (float)M_PI;

    while (doa_rad < 0.0f) {
        doa_rad += two_pi;
    }

    while (doa_rad >= two_pi) {
        doa_rad -= two_pi;
    }

    idx_f = doa_rad / two_pi * (float)num_angles;
    idx = (int)roundf(idx_f) % num_angles;

    return idx;
}

/* ===================== pool-first memory layout =====================
 *
 * Bump allocator carving this module's flat pool into typed sub-blocks.
 * File-private because that is this directory's convention -- masker.c,
 * vad_api.c, gsc.c and srp.c each carry their own copy. Consolidating all of
 * them next to ck_field_size() in audio_common's mem_align.h (which already
 * shares the sizing half of the same lockstep pair) is the right fix, but it
 * is a sweep across every consumer, not something to start from one new
 * module.
 */
typedef struct PoolCursor {
    uint8_t* ptr;
    size_t remaining;
} PoolCursor;

static void* pool_carve(PoolCursor* cursor, size_t count, size_t element_size)
{
    size_t raw, aligned;
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

size_t post_gain_get_mem_size(const PostGainConfig* cfg)
{
    size_t total;
    size_t F;

    if (!cfg || cfg->F <= 0) return 0;
    F = (size_t)cfg->F;

    total = ck_align16_size(sizeof(struct PostGainState));
    /* float[F] x3: prev_gain, target_gain, freq_gain */
    total = ck_field_size(total, F, sizeof(float));
    total = ck_field_size(total, F, sizeof(float));
    total = ck_field_size(total, F, sizeof(float));
    /* int[F] x3: mask_work, raw_mask_frame, class_frame */
    total = ck_field_size(total, F, sizeof(int));
    total = ck_field_size(total, F, sizeof(int));
    total = ck_field_size(total, F, sizeof(int));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/* Carve order MUST stay in lockstep with post_gain_get_mem_size() above. The
 * exhausted-exactly check at the end is what enforces it in both directions:
 * carving more than was sized runs the cursor out and returns NULL, and
 * sizing more than is carved leaves a remainder. (masker.c and vad_api.c
 * check only the first direction; gsc.c and srp.c check both, as here.) */
static struct PostGainState* post_gain_carve(const PostGainConfig* cfg,
                                             void* mem, size_t need)
{
    struct PostGainState* st;
    PoolCursor cursor;
    float init_gain;
    size_t F;
    int f;

    memset(mem, 0, need);
    st = (struct PostGainState*)mem;

    cursor.ptr = (uint8_t*)mem + ck_align16_size(sizeof(struct PostGainState));
    cursor.remaining = need - ck_align16_size(sizeof(struct PostGainState));

    st->F = cfg->F;
    F = (size_t)cfg->F;

    st->prev_gain      = (float*)pool_carve(&cursor, F, sizeof(float));
    st->target_gain    = (float*)pool_carve(&cursor, F, sizeof(float));
    st->freq_gain      = (float*)pool_carve(&cursor, F, sizeof(float));

    st->mask_work      = (int*)pool_carve(&cursor, F, sizeof(int));
    st->raw_mask_frame = (int*)pool_carve(&cursor, F, sizeof(int));
    st->class_frame    = (int*)pool_carve(&cursor, F, sizeof(int));

    if (!st->prev_gain || !st->target_gain || !st->freq_gain ||
        !st->mask_work || !st->raw_mask_frame || !st->class_frame) {
        return NULL;
    }
    if (cursor.remaining != 0) return NULL;

    init_gain = cfg->gain_match;

    for (f = 0; f < st->F; f++) {
        st->prev_gain[f]   = init_gain;
        st->target_gain[f] = init_gain;
        st->freq_gain[f]   = init_gain;
    }
    /* mask_work / raw_mask_frame / class_frame and stats are already zero
     * from the memset above. */

    return st;
}

PostGainState* post_gain_create(const PostGainConfig* cfg)
{
    struct PostGainState* st;
    void* block;
    size_t need = post_gain_get_mem_size(cfg);

    if (need == 0) return NULL;
    /* One 16-byte-aligned block backs the struct and every array, so a
     * failure unwinds with a single free() instead of six. */
    if (posix_memalign(&block, 16, need) != 0 || !block) return NULL;

    st = post_gain_carve(cfg, block, need);
    if (!st) { free(block); return NULL; }
    st->owned_heap = block;
    return st;
}

PostGainState* post_gain_init(const PostGainConfig* cfg,
                              void* mem, size_t mem_size)
{
    struct PostGainState* st;
    size_t need = post_gain_get_mem_size(cfg);

    if (!mem || need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need)
        return NULL;

    st = post_gain_carve(cfg, mem, need);
    if (!st) return NULL;
    st->owned_heap = NULL;   /* caller owns it */
    return st;
}

void post_gain_destroy(PostGainState* st)
{
    if (!st) return;
    /* Pool path (owned_heap == NULL, post_gain_init()): the caller owns the
     * memory -- struct and every array carved from it -- nothing to free. */
    if (st->owned_heap) free(st->owned_heap);
}

/* ---------- stage 4: box-average over frequency ----------
 *
 * Split into edge/interior/edge rather than testing the window bounds at every
 * bin: only the first and last `r` bins can leave the array, and the interior
 * loop then carries no branch and vectorises. The addition sequence is
 * unchanged -- still left to right over k = -r..r -- so the result is
 * bit-identical, which the module's equivalence test checks.
 *
 * Measured at F=257, r=2: 247.2 -> 34.6 ns/frame. A running add/subtract
 * accumulator would be cheaper still but reassociates the sum and is NOT
 * bit-exact, so it stays rejected. A hand-written NEON version measured 38.0
 * ns -- no better than letting the compiler have the split loop. */
static void post_gain_freq_smooth(const float* in, float* out, int F, int r)
{
    int istart = (r < F) ? r : F;
    int iend = (F - r > istart) ? (F - r) : istart;
    int f, k;

    for (f = 0; f < istart; f++) {
        float sum = 0.0f;
        int count = 0;
        for (k = -r; k <= r; k++) {
            int ff = f + k;
            if (ff >= 0 && ff < F) { sum += in[ff]; count++; }
        }
        out[f] = sum / (float)count;
    }
    for (f = istart; f < iend; f++) {
        float sum = 0.0f;
        for (k = -r; k <= r; k++) sum += in[f + k];
        /* Divide, never multiply by a hoisted 1/(2r+1): the reciprocal form
         * rounds differently and the edge loops divide by their own count. */
        out[f] = sum / (float)(2 * r + 1);
    }
    for (f = iend; f < F; f++) {
        float sum = 0.0f;
        int count = 0;
        for (k = -r; k <= r; k++) {
            int ff = f + k;
            if (ff >= 0 && ff < F) { sum += in[ff]; count++; }
        }
        out[f] = sum / (float)count;
    }
}

void post_gain_apply(PostGainState* st,
                     const PostGainConfig* cfg,
                     Complex* Y,
                     const int* mask,
                     const int* bin_best_idx,
                     float doa_used)
{
    const int* use_mask;
    float g_match, g_suppress, angle_ratio;
    int F, num_angles, doa_used_idx;
    int angle_match_cnt, angle_vad;
    int cnt_match, f;

    if (!cfg || !Y || !st || !mask || !bin_best_idx) return;
    if (cfg->F != st->F) return;

    F = st->F;
    /* ---------- bypass post-gain ---------- */
    if (!cfg->enable) {
        for (f = 0; f < F; f++) {
            st->raw_mask_frame[f] = 0;
            st->class_frame[f]    = 0;
            st->target_gain[f]    = 1.0f;
            st->freq_gain[f]      = 1.0f;
            st->prev_gain[f]      = 1.0f;
        }

        st->stats.cnt_match = 0;
        st->stats.cnt_suppress = 0;

        return;
    }
    /*------------------------------------------*/
    num_angles = cfg->num_angles;
    doa_used_idx = post_gain_angle_to_index(doa_used, num_angles);

    if (doa_used_idx < 0) {
        for (f = 0; f < F; f++) {
            st->raw_mask_frame[f] = 0;
            st->class_frame[f] = 0;
        }

        st->stats.cnt_match = 0;
        st->stats.cnt_suppress = F;

        return;
    }

    if (cfg->enable_mask_relax && cfg->mask_relax_bins > 0) {
        int r = cfg->mask_relax_bins;

        /* The scan below is the module's most expensive stage, and it is
         * already the right shape: an O(F) two-pass dilation measured SLOWER
         * (132.5 -> 201.1 ns/frame at F=257), because the early break usually
         * fires on the first k for a dense mask. Leave it. */
        for (f = 0; f < F; f++) {
            int keep = 0;
            int k;

            for (k = -r; k <= r; k++) {
                int ff = f + k;

                if (ff >= 0 && ff < F && mask[ff]) {
                    keep = 1;
                    break;
                }
            }

            st->mask_work[f] = keep;
        }

        use_mask = st->mask_work;
    } else {
        /* No relaxation: read the caller's mask directly instead of copying
         * it into mask_work first. */
        use_mask = mask;
    }

    /* ---------- stage 1: build raw directional mask ---------- */
    angle_match_cnt = 0;

    for (f = 0; f < F; f++) {
        int angle_match = 0;

        if (use_mask[f] &&
            angle_idx_is_near(bin_best_idx[f],
                              doa_used_idx,
                              num_angles,
                              cfg->angle_tol)) {
            angle_match = 1;
            angle_match_cnt++;
        }

        st->raw_mask_frame[f] = angle_match;
    }

    /* ---------- stage 2: raw directional mask -> angle_vad ---------- */
    angle_ratio = (float)angle_match_cnt / (float)F;

    angle_vad = (angle_ratio > cfg->angle_vad_thr) ? 1 : 0;

    /* ---------- stage 3: build target gain ----------
     * Both candidate gains are frame constants, so they are clipped once here
     * rather than once per bin. Likewise the two counters: every bin lands in
     * exactly one class, so cnt_match is the matched count and cnt_suppress
     * is the remainder -- no per-bin increment needed.
     *
     * `&` and not `&&`: both operands are already 0/1, so the result is
     * identical, but the short-circuit form makes angle_vad a per-bin branch
     * and stops the loop vectorising entirely (measured 103.0 -> 18.7
     * ns/frame at F=257). Do not "fix" this back. */
    g_match    = clip_f(cfg->gain_match,    cfg->min_gain, cfg->max_gain);
    g_suppress = clip_f(cfg->gain_suppress, cfg->min_gain, cfg->max_gain);

    for (f = 0; f < F; f++) {
        int matched = angle_vad & st->raw_mask_frame[f];

        st->target_gain[f] = matched ? g_match : g_suppress;
        st->class_frame[f] = matched ? 2 : 0;
    }

    cnt_match = angle_vad ? angle_match_cnt : 0;

    /* ---------- stage 4: frequency smoothing ---------- */
    if (cfg->enable_freq_smooth && cfg->freq_smooth_radius > 0) {
        post_gain_freq_smooth(st->target_gain, st->freq_gain, F,
                              cfg->freq_smooth_radius);
    } else {
        memcpy(st->freq_gain, st->target_gain, (size_t)F * sizeof(float));
    }

    /* ---------- stage 5: time smoothing + apply gain ----------
     * The smoother writes into prev_gain, which is then clipped and applied
     * whole-array: same per-bin values as the fused loop this replaced, but
     * the clip and the complex multiply each become one shared kernel pass.
     * (Fusing those two passes into one measured 50.0 -> 49.8 ns/frame at
     * F=257 -- noise -- so the shared kernels stay.)
     *
     * The four coefficients are hoisted because prev_gain is a float* that
     * may alias the config struct, so the compiler must otherwise reload
     * attack_alpha/release_alpha and recompute (1-alpha) at every bin, which
     * keeps the loop scalar (measured 96.1 -> 26.7 ns/frame at F=257).
     * (1-alpha) is exact in binary floating point, so hoisting it is
     * bit-identical to recomputing it.
     *
     * Not sk_asym_ema_f32: that kernel selects its falling coefficient on
     * `x < s` while this module selects its attack coefficient on `target >
     * prev`, so the two disagree at target == prev -- which is exactly the
     * steady state a constant-target gain sits in, and the two branches
     * genuinely differ there (17.0% of values, measured). */
    if (cfg->enable_time_smooth) {
        const float a_att = cfg->attack_alpha;
        const float a_rel = cfg->release_alpha;
        const float m_att = 1.0f - a_att;
        const float m_rel = 1.0f - a_rel;

        for (f = 0; f < F; f++) {
            float target = st->freq_gain[f];
            float prev = st->prev_gain[f];

            st->prev_gain[f] = (target > prev) ? (a_att * prev + m_att * target)
                                               : (a_rel * prev + m_rel * target);
        }
    } else {
        memcpy(st->prev_gain, st->freq_gain, (size_t)F * sizeof(float));
    }

    sk_clip_f32(st->prev_gain, cfg->min_gain, cfg->max_gain, F);
    sk_capply_gain_f32(Y, Y, st->prev_gain, F);

    st->stats.cnt_match = cnt_match;
    st->stats.cnt_suppress = F - cnt_match;
}


const int* post_gain_get_raw_mask(const PostGainState* st)
{
    return st ? st->raw_mask_frame : NULL;
}

const int* post_gain_get_class(const PostGainState* st)
{
    return st ? st->class_frame : NULL;
}

const float* post_gain_get_gain(const PostGainState* st)
{
    return st ? st->prev_gain : NULL;
}

const PostGainStats* post_gain_get_stats(const PostGainState* st)
{
    return st ? &st->stats : NULL;
}
