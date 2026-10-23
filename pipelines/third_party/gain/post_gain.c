#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "post_gain.h"
#include "utility/fast_math.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct PostGainState {
    int F;

    float* prev_gain;
    float* target_gain;
    float* freq_gain;

    /* Relaxed copy of the caller's mask; written only when mask relaxation is enabled. */
    int* mask_work;

    /* Per-frame log buffers */
    int* raw_mask_frame;
    int* class_frame;

    PostGainStats stats;

    /* Non-NULL only on the post_gain_create() heap path (one block holding the struct
     * and every array, freed by post_gain_destroy()). */
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

/* Own radian-based angle-to-bin mapping (SRP's rounds through degrees). The finite
 * test also rejects +/-inf: the reduction below is repeated subtraction. */
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

/* Bump allocator carving the flat pool into typed sub-blocks (file-private). */
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

/* Carve order must match post_gain_get_mem_size(); the exact-exhaustion check at
 * the end catches a drift in either direction. */
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
    /* mask_work / raw_mask_frame / class_frame and stats are zeroed by the memset above. */

    return st;
}

PostGainState* post_gain_create(const PostGainConfig* cfg)
{
    struct PostGainState* st;
    void* block;
    size_t need = post_gain_get_mem_size(cfg);

    if (need == 0) return NULL;
    /* One 16-byte-aligned block backs the struct and every array. */
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
    st->owned_heap = NULL;
    return st;
}

void post_gain_destroy(PostGainState* st)
{
    if (!st) return;
    /* Pool path (owned_heap == NULL): the caller owns the memory. */
    if (st->owned_heap) free(st->owned_heap);
}

/* Stage 4: box-average over frequency, split into edge/interior/edge so the
 * interior loop has no bound tests. Summation order is left to right over k = -r..r. */
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
        /* Divide by the count (not a hoisted reciprocal): edge windows have their own count. */
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

        /* Per-bin mask scan; the early break usually fires on the first k. */
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
        /* No relaxation: read the caller's mask directly. */
        use_mask = mask;
    }

    /* Stage 1: raw directional mask */
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

    /* Stage 2: raw directional mask -> angle_vad */
    angle_ratio = (float)angle_match_cnt / (float)F;

    angle_vad = (angle_ratio > cfg->angle_vad_thr) ? 1 : 0;

    /* Stage 3: target gain. Candidate gains are clipped once per frame; cnt_suppress is
     * the remainder of cnt_match. `&` (not `&&`) keeps the loop branch-free. */
    g_match    = clip_f(cfg->gain_match,    cfg->min_gain, cfg->max_gain);
    g_suppress = clip_f(cfg->gain_suppress, cfg->min_gain, cfg->max_gain);

    for (f = 0; f < F; f++) {
        int matched = angle_vad & st->raw_mask_frame[f];

        st->target_gain[f] = matched ? g_match : g_suppress;
        st->class_frame[f] = matched ? 2 : 0;
    }

    cnt_match = angle_vad ? angle_match_cnt : 0;

    /* Stage 4: frequency smoothing */
    if (cfg->enable_freq_smooth && cfg->freq_smooth_radius > 0) {
        post_gain_freq_smooth(st->target_gain, st->freq_gain, F,
                              cfg->freq_smooth_radius);
    } else {
        memcpy(st->freq_gain, st->target_gain, (size_t)F * sizeof(float));
    }

    /* Stage 5: time smoothing into prev_gain, then clip and apply. Coefficients are
     * hoisted (prev_gain may alias the config). The attack coefficient is selected on
     * `target > prev`. */
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
