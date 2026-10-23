#include "fix_gain.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

struct FixGain {
    int channels;
    int enable;
    int enable_clip;
    float clip_value;

    /* Per-channel gain with the global factor folded in. */
    float* gain;

    /* Non-NULL only on the fix_gain_create() heap path (one block holding struct and
     * gain table, freed by fix_gain_destroy()). */
    void* owned_heap;
};

float fix_gain_db_to_linear(float db)
{
    return powf(10.0f, db / 20.0f);
}

size_t fix_gain_get_mem_size(const FixGainConfig* cfg)
{
    size_t total;
    int c;

    if (!cfg || cfg->channels <= 0) return 0;
    /* Gains multiply the signal directly; non-finite ones would produce NaN audio. */
    if (!isfinite(cfg->global_gain) || !isfinite(cfg->clip_value)) return 0;
    if (cfg->channel_gain) {
        for (c = 0; c < cfg->channels; c++) {
            if (!isfinite(cfg->channel_gain[c])) return 0;
        }
    }

    total = ck_align16_size(sizeof(struct FixGain));
    total = ck_field_size(total, (size_t)cfg->channels, sizeof(float));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/* Carve order must match fix_gain_get_mem_size(); a size drift returns NULL. Both
 * constructors use it, so heap and pool layouts agree. */
static FixGain* fix_gain_carve(const FixGainConfig* cfg, void* mem, size_t need)
{
    FixGain* fg;
    size_t head, body;
    float global;
    int c;

    head = ck_align16_size(sizeof(struct FixGain));
    body = ck_align16_size(ck_mul_size((size_t)cfg->channels, sizeof(float)));
    if (MEM_SIZE_INVALID(body) || ck_add_size(head, body) != need) return NULL;

    memset(mem, 0, need);
    fg = (FixGain*)mem;

    fg->channels = cfg->channels;
    fg->enable = cfg->enable;
    fg->enable_clip = cfg->enable_clip;
    /* A non-positive clip window would mute rather than limit. */
    fg->clip_value = (cfg->clip_value > 0.0f) ? cfg->clip_value : 1.0f;

    fg->gain = (float*)((uint8_t*)mem + head);

    /* A global_gain of 0 means unset (1.0). */
    global = (cfg->global_gain == 0.0f) ? 1.0f : cfg->global_gain;

    for (c = 0; c < cfg->channels; c++) {
        fg->gain[c] = global * (cfg->channel_gain ? cfg->channel_gain[c] : 1.0f);
    }

    return fg;
}

FixGain* fix_gain_create(const FixGainConfig* cfg)
{
    void* block;
    FixGain* fg;
    size_t need = fix_gain_get_mem_size(cfg);

    if (need == 0) return NULL;
    if (posix_memalign(&block, 16, need) != 0 || !block) return NULL;

    fg = fix_gain_carve(cfg, block, need);
    if (!fg) { free(block); return NULL; }
    fg->owned_heap = block;
    return fg;
}

FixGain* fix_gain_init(const FixGainConfig* cfg, void* mem, size_t mem_size)
{
    FixGain* fg;
    size_t need = fix_gain_get_mem_size(cfg);

    if (!mem || need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need)
        return NULL;

    fg = fix_gain_carve(cfg, mem, need);
    if (!fg) return NULL;
    fg->owned_heap = NULL;
    return fg;
}

void fix_gain_reset(FixGain* fg)
{
    (void)fg;
}

void fix_gain_process(FixGain* fg, float* x, int n, int ch)
{
    if (!fg || !x || n <= 0) return;
    if (!fg->enable) return;
    if (ch < 0 || ch >= fg->channels) return;

    sk_scale_f32(x, x, fg->gain[ch], n);

    /* Separate scale and clip passes through the shared kernels; clip_value > 0, so
     * the low-bound-first test is well defined. */
    if (fg->enable_clip)
        sk_clip_f32(x, -fg->clip_value, fg->clip_value, n);
}

void fix_gain_destroy(FixGain* fg)
{
    if (!fg) return;
    /* Pool path (owned_heap == NULL): the caller owns the memory. */
    if (fg->owned_heap) free(fg->owned_heap);
}
