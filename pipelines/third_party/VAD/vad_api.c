#include "vad_api.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mem_align.h"

struct VADApi {
    VADApiConfig cfg;

    MaskEstimator* masker;
    MaskVAD* mask_vad;

    const int* mask;
    int vad_raw;
    int vad_out;

    /* Non-NULL only on the vad_api_create() heap path. NULL on the
     * vad_api_init() caller-pool path. */
    void* owned_heap;
};

size_t vad_api_get_mem_size(const VADApiConfig* cfg)
{
    size_t total;
    size_t masker_need;
    size_t vad_need;

    if (!cfg || cfg->backend != VAD_BACKEND_MASKER) return 0;

    /* The two sub-configs are one contract, and nothing downstream compares
     * them. masker_step() emits NFFT/2+1 mask entries; vad_step() then reads
     * vad_cfg.F of them, so a larger F reads past the end of the mask -- a
     * silent out-of-bounds read, not a wrong answer. Both fields are
     * caller-supplied and independent, so this is the only place that can
     * refuse the combination. */
    if (cfg->vad_cfg.F != cfg->masker_cfg.NFFT / 2 + 1) return 0;
    if (!vad_config_is_valid(&cfg->vad_cfg)) return 0;
    /* Frequency smoothing has two ways to not do what was asked, and both
     * are silent. An even smooth_size asks for a k-wide window and gets k+1,
     * because the smoother centres on +-(k/2) -- only odd k spans exactly k
     * bins. A smooth_size of zero or less makes median_filter_1d() copy its
     * input straight through, so the pass runs and changes nothing. Refuse
     * both rather than widen or skip silently. */
    if (cfg->masker_cfg.enable_freq_smooth &&
        (cfg->masker_cfg.smooth_size <= 0 ||
         (cfg->masker_cfg.smooth_size & 1) == 0)) return 0;

    masker_need = masker_get_mem_size(&cfg->masker_cfg);
    vad_need    = vad_get_mem_size();
    if (masker_need == 0 || vad_need == 0) return 0;

    /* One pool: [VADApi][masker sub-pool][VAD sub-pool]. Each sub-pool is
     * handed to that module's own _init(), which carves it in turn -- the
     * sizes below must stay in lockstep with vad_api_init()'s split. */
    total = ck_align16_size(sizeof(VADApi));
    total = ck_add_size(total, ck_align16_size(masker_need));
    total = ck_add_size(total, ck_align16_size(vad_need));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/* Bump allocator over this instance's pool, bounds-checked. Hand-advancing a
 * uint8_t* would carve the sub-pools without ever comparing against what is
 * left, so a get_mem_size/init drift would run off the end silently instead of
 * returning NULL. Same shape as masker.c's copy and 4aec_nr_res.c's. */
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

VADApi* vad_api_init(const VADApiConfig* cfg, void* mem, size_t mem_size)
{
    VADApi* v;
    PoolCursor cursor;
    void* masker_mem;
    void* vad_mem;
    size_t need;
    size_t masker_need;
    size_t vad_need;

    need = vad_api_get_mem_size(cfg);
    if (!mem || need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need)
        return NULL;

    memset(mem, 0, need);
    v = (VADApi*)mem;
    v->cfg = *cfg;

    masker_need = masker_get_mem_size(&cfg->masker_cfg);
    vad_need    = vad_get_mem_size();

    cursor.ptr = (uint8_t*)mem + ck_align16_size(sizeof(VADApi));
    cursor.remaining = need - ck_align16_size(sizeof(VADApi));

    masker_mem = pool_carve(&cursor, 1, masker_need);
    vad_mem    = pool_carve(&cursor, 1, vad_need);
    if (!masker_mem || !vad_mem) return NULL;

    v->masker   = masker_init(masker_mem, masker_need, &cfg->masker_cfg);
    v->mask_vad = vad_init(vad_mem, vad_need, &cfg->vad_cfg);

    if (!v->masker || !v->mask_vad) return NULL;

    v->mask = NULL;
    v->vad_raw = 0;
    v->vad_out = 0;
    v->owned_heap = NULL;   /* caller owns it */
    return v;
}

VADApi* vad_api_create(const VADApiConfig* cfg)
{
    VADApi* v;
    void* block;
    size_t need = vad_api_get_mem_size(cfg);

    if (need == 0) return NULL;
    /* Same single-block shape as the pool path, so the heap and pool
     * constructors cannot disagree about the layout. */
    if (posix_memalign(&block, 16, need) != 0 || !block) return NULL;

    v = vad_api_init(cfg, block, need);
    if (!v) { free(block); return NULL; }
    v->owned_heap = block;
    return v;
}

void vad_api_process(VADApi* v,
                     const Complex* frame_in)
{
    if (!v || !frame_in) return;

    if (v->cfg.backend == VAD_BACKEND_MASKER) {
        masker_step(v->masker, frame_in);

        v->mask = masker_get_mask(v->masker);

        vad_step(v->mask_vad, v->mask);

        v->vad_raw = vad_get_raw(v->mask_vad);
        v->vad_out = vad_get_out(v->mask_vad);
    }
}

int vad_api_get(const VADApi* v)
{
    return v ? v->vad_out : 0;
}

int vad_api_get_raw(const VADApi* v)
{
    return v ? v->vad_raw : 0;
}

const int* vad_api_get_mask(const VADApi* v)
{
    return v ? v->mask : NULL;
}

MaskEstimator* vad_api_get_masker(VADApi* v)
{
    return v ? v->masker : NULL;
}

MaskVAD* vad_api_get_mask_vad(VADApi* v)
{
    return v ? v->mask_vad : NULL;
}

void vad_api_destroy(VADApi* v)
{
    if (!v) return;

    /* Both sub-objects live inside this instance's pool, so their destroys
     * are no-ops on that path; they are still called so the heap path and any
     * future sub-object with real teardown stay correct. */
    masker_destroy(v->masker);
    vad_destroy(v->mask_vad);

    if (v->owned_heap) free(v->owned_heap);
}