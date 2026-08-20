#include "nr_gain.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fast_math.h"
#include "mem_align.h"
#include "simd_kernels.h"

struct NrGain {
    float gain;

    /* Non-NULL only on the nr_gain_create() heap path. NULL on the
     * nr_gain_init() caller-pool path. */
    void* owned_heap;
};

size_t nr_gain_get_mem_size(void)
{
    size_t total = ck_align16_size(sizeof(struct NrGain));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

NrGain* nr_gain_create(void)
{
    void* block;
    NrGain* ng;
    size_t need = nr_gain_get_mem_size();

    if (need == 0) return NULL;
    if (posix_memalign(&block, 16, need) != 0 || !block) return NULL;

    ng = nr_gain_init(block, need);
    if (!ng) { free(block); return NULL; }
    ng->owned_heap = block;
    return ng;
}

NrGain* nr_gain_init(void* mem, size_t mem_size)
{
    NrGain* ng;
    size_t need = nr_gain_get_mem_size();

    if (!mem || need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need)
        return NULL;

    memset(mem, 0, need);
    ng = (NrGain*)mem;
    ng->gain = 1.0f;
    ng->owned_heap = NULL;   /* caller owns it */
    return ng;
}

void nr_gain_reset(NrGain* ng)
{
    if (!ng) return;
    ng->gain = 1.0f;
}

float nr_gain_process(NrGain* ng,
                      const NrGainConfig* cfg,
                      float* x,
                      int n,
                      float doa_gain)
{
    float min_gain, max_gain, target_gain, prev_gain, alpha;

    if (!ng) {
        return 1.0f;
    }

    /* No config, switched off, or NR itself is off while the compensation is
     * tied to it: pin at unity. */
    if (!cfg || !cfg->enable ||
        (cfg->only_when_nr_enable && !cfg->nr_enable)) {
        ng->gain = 1.0f;
        return ng->gain;
    }

    min_gain = cfg->min_gain;
    max_gain = cfg->max_gain;

    if (min_gain < 0.0f) min_gain = 0.0f;
    if (max_gain < min_gain) max_gain = min_gain;

    target_gain = isnan(doa_gain) ? cfg->noise_gain : cfg->target_gain;
    target_gain = clip_f(target_gain, min_gain, max_gain);

    prev_gain = ng->gain;
    alpha = (target_gain > prev_gain) ? cfg->attack_alpha
                                      : cfg->release_alpha;
    alpha = clip_f(alpha, 0.0f, 0.9999f);

    ng->gain = alpha * prev_gain + (1.0f - alpha) * target_gain;
    ng->gain = clip_f(ng->gain, min_gain, max_gain);

    if (x && n > 0) {
        sk_scale_f32(x, x, ng->gain, n);
    }

    return ng->gain;
}

float nr_gain_get_gain(const NrGain* ng)
{
    if (!ng) return 1.0f;
    return ng->gain;
}

void nr_gain_destroy(NrGain* ng)
{
    if (!ng) return;
    /* Pool path (owned_heap == NULL, nr_gain_init()): caller-owned memory,
     * nothing to free. */
    if (ng->owned_heap) free(ng->owned_heap);
}
