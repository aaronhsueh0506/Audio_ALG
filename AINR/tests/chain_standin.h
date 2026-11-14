/* Stand-in model schedule, input generator and digest shared by the whole-chain test programs
 * (test_nn_chains.c, dfn2_runtime_grid_chain.c). Test-only. */

#ifndef AINR_TESTS_CHAIN_STANDIN_H
#define AINR_TESTS_CHAIN_STANDIN_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ---- generator and digest -------------------------------------------- */

#define RNG_SEED 0x6b1d5eedu
static uint32_t rng_state = RNG_SEED;
static uint32_t rng(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static float uniform(void) { return (float)(rng() & 0xffff) / 32768.0f - 1.0f; }

static void fill(float* x, size_t n, float scale)
{
    for (size_t i = 0; i < n; ++i) x[i] = uniform() * scale;
}

typedef struct {
    uint64_t in, state, post, status;
} Digest;

static void digest_init(Digest* d)
{
    d->in = d->state = d->post = d->status = UINT64_C(14695981039346656037);
}

static void hash_u32(uint64_t* h, uint32_t v)
{
    *h = (*h ^ v) * UINT64_C(1099511628211);
}

static void hash_f32(uint64_t* h, const float* x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        if (isnan(x[i])) bits = 0x7fc00000u;
        else memcpy(&bits, &x[i], sizeof(bits));
        hash_u32(h, bits);
    }
}

static uint64_t digest_fold(const Digest* d)
{
    uint64_t h = UINT64_C(14695981039346656037);
    const uint64_t parts[4] = { d->in, d->state, d->post, d->status };
    for (int i = 0; i < 4; ++i) {
        hash_u32(&h, (uint32_t)parts[i]);
        hash_u32(&h, (uint32_t)(parts[i] >> 32));
    }
    return h;
}

/* Speech-like analysed frame: magnitude falls with the bin, random phase.
 * Every few frames it is quiet, loud, exactly zero or subnormal. */
static void make_spectrum(int hop, int bins, float* re, float* im)
{
    float level = 0.3f;
    if (hop % 9 == 0) level = 1e-4f;
    if (hop % 31 == 7) level = 4.0f;
    if (hop % 53 == 11) level = 0.0f;
    if (hop % 71 == 13) level = 1e-20f;
    for (int k = 0; k < bins; ++k) {
        float m = level / (1.0f + 0.05f * (float)k) *
                  (0.5f + 0.5f * fabsf(uniform()));
        re[k] = m * uniform();
        im[k] = m * uniform();
    }
}

/* What the stand-in model does on a hop. */
enum { K_NORMAL, K_TINY, K_HUGE, K_BAD_HEAD, K_BAD_HEAD_INF, K_BAD_STATE_FIRST,
       K_BAD_STATE_LAST, K_BAD_STATE_MID, K_SKIP, K_UNWRITTEN };

static int hop_kind(int hop)
{
    switch (hop % 40) {
    case 5:  return K_TINY;
    case 9:  return K_HUGE;
    case 13: return K_BAD_HEAD;
    case 17: return K_BAD_HEAD_INF;
    case 21: return K_BAD_STATE_FIRST;
    case 25: return K_BAD_STATE_LAST;
    case 29: return K_BAD_STATE_MID;
    case 33: return K_SKIP;
    case 37: return K_UNWRITTEN;
    default: return K_NORMAL;
    }
}

static float kind_scale(int kind)
{
    return kind == K_TINY ? 1e-30f : kind == K_HUGE ? 1e38f : 1.0f;
}

static void poison(float* x, size_t n, int where, int hop)
{
    static const uint32_t bad[3] = { 0x7fc00000u, 0x7f800000u, 0xff800000u };
    size_t at = where == 0 ? 0 : where == 1 ? n - 1 : n / 2;
    memcpy(&x[at], &bad[(hop / 40) % 3], sizeof(float));
}

static int poison_where(int kind)
{
    return kind == K_BAD_STATE_FIRST ? 0 : kind == K_BAD_STATE_LAST ? 1 : 2;
}

static int is_bad_state(int kind)
{
    return kind == K_BAD_STATE_FIRST || kind == K_BAD_STATE_LAST ||
           kind == K_BAD_STATE_MID;
}

#endif /* AINR_TESTS_CHAIN_STANDIN_H */
