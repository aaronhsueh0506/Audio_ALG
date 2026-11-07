/* Align-ULCNet pre/post class (contract: ulcnet_prepost.h): object lifecycle, pool carve and the per-hop frame state
 * machine over ulcnet_model_io.c. Requires -ffp-contract=off. No heap in _init, no globals, no stdio. */

#include "ulcnet_prepost.h"

#include "utility/mem_align.h"

#include <stdlib.h>
#include <string.h>

/* Pool carve alignment, shared with ulcnet_model_io so one cursor serves both regions. */
#define PP_ALIGN ((size_t)ULCNET_MODEL_IO_ALIGNMENT)

struct UlcnetPrepost {
    UlcnetModelIoState *io;

    float *err_re;      /* [n_bins] */
    float *err_im;
    float *far_re;
    float *far_im;
    float *enh_re;      /* [n_bins] frame_finish staging */
    float *enh_im;

    int n_bins;
    int hop;
    int frame_open;     /* a frame is pending the accelerator */
    int prepared;       /* frame_open() armed the transaction for the open frame; implies frame_open */

    /* _create bookkeeping; 0 for an _init instance. */
    void *heap_base;
};

/* Pool arithmetic uses audiolib's saturating helpers: overflow pins the cursor to SIZE_MAX (MEM_SIZE_INVALID). */

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(ULCNET_MODEL_IO_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* Reserve count * elem bytes at *cursor, aligned; base == NULL sizes only. */
static int pp_carve(unsigned char *base, size_t *cursor, size_t count, size_t elem,
                    void **out) {
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, ck_mul_size(count, elem));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

static uint32_t pp_fnv1a(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* Build identity a pool is valid for: grid, layout and carve versions, delay depth (a pool handed across a grid or D
 * change is refused). The state registration is not in it: the buffers are borrowed and change no pool byte. */
static uint32_t pp_build_hash(const UlcnetGrid *grid, uint32_t layout) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a(h, (uint32_t)grid->sample_rate);
    h = pp_fnv1a(h, (uint32_t)grid->n_fft);
    h = pp_fnv1a(h, layout);
    h = pp_fnv1a(h, (uint32_t)ULCNET_PREPOST_CARVE_VERSION);
    h = pp_fnv1a(h, (uint32_t)grid->delay_depth);
    return ac_memreq_abi_hash(h);
}

/* Walk the layout once: base == NULL sizes it, otherwise it also carves; get_mem_size and init share this walk so
 * their order agrees. The model-I/O block is a sub-pool at its own alignment; its init (with the registration)
 * runs here. */
static int pp_layout(UlcnetPrepost *p, unsigned char *base,
                     const UlcnetModelIoDescriptor *descriptor,
                     const UlcnetModelIoStateBuffers *buffers,
                     const UlcnetModelIoMemReq *io_req,
                     size_t *total) {
    float **slots[6] = { NULL };
    size_t cursor = 0;
    size_t i;
    void *ptr = NULL;

    if (p) {
        slots[0] = &p->err_re;  slots[1] = &p->err_im;
        slots[2] = &p->far_re;  slots[3] = &p->far_im;
        slots[4] = &p->enh_re;  slots[5] = &p->enh_im;
    }
    if (pp_carve(base, &cursor, 1u, sizeof(UlcnetPrepost), &ptr) != 0) return -1;

    if (pp_carve(base, &cursor, io_req->bytes, 1u, &ptr) != 0) return -1;
    if (base && p) {
        p->io = ulcnet_model_io_init(ptr, io_req->bytes, descriptor, buffers);
        if (!p->io) return -1;
    }

    for (i = 0; i < sizeof slots / sizeof slots[0]; ++i) {
        if (pp_carve(base, &cursor, (size_t)descriptor->spectrum_bins, sizeof(float),
                     &ptr) != 0)
            return -1;
        if (base && slots[i]) *slots[i] = (float *)ptr;
    }

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

int ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg, const UlcnetGrid *grid) {
    if (!cfg || ulcnet_grid_validate(grid) != ULCNET_GRID_OK) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->grid = *grid;
    return 0;
}

/* The registration of the config, in the model-I/O form (one validator, there). */
static void pp_buffers(const UlcnetPrepostConfig *cfg,
                       UlcnetModelIoStateBuffers *buffers) {
    buffers->key_history = cfg->key_history;
    buffers->key_history_elements = cfg->key_history_elements;
    buffers->value_history = cfg->value_history;
    buffers->value_history_elements = cfg->value_history_elements;
    buffers->logit_history = cfg->logit_history;
    buffers->logit_history_elements = cfg->logit_history_elements;
    buffers->h_gru = cfg->h_gru;
    buffers->gru_hidden_elements = cfg->gru_hidden_elements;
}

static int pp_check_config(const UlcnetPrepostConfig *cfg) {
    UlcnetModelIoStateBuffers buffers;
    if (!cfg || ulcnet_grid_validate(&cfg->grid) != ULCNET_GRID_OK) return -1;
    pp_buffers(cfg, &buffers);
    return ulcnet_model_io_state_buffers_validate(&buffers);
}

static int pp_descriptor_for(const UlcnetPrepostConfig *cfg,
                             UlcnetModelIoDescriptor *out) {
    if (ulcnet_model_io_descriptor_default(&cfg->grid, out) != 0)
        return -1;
    if (cfg->model_layout_version)
        out->layout_version = cfg->model_layout_version;
    return ulcnet_model_io_descriptor_validate(&cfg->grid, out);
}

int ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg,
                                UlcnetPrepostMemReq *req) {
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoMemReq io_req;
    size_t total = 0;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_descriptor_for(cfg, &descriptor) != 0) return -1;
    if (ulcnet_model_io_get_mem_requirements(&descriptor, &io_req) != 0)
        return -1;
    if (pp_layout(NULL, NULL, &descriptor, NULL, &io_req, &total) != 0)
        return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = ULCNET_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = descriptor.layout_version;
    req->build_flags_hash = pp_build_hash(&cfg->grid, descriptor.layout_version);
    req->alignment = (uint32_t)(io_req.alignment > PP_ALIGN
                                ? io_req.alignment : PP_ALIGN);
    req->bytes = (uint64_t)total;
    return 0;
}

UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes,
                                      const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected) {
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoStateBuffers buffers;
    UlcnetModelIoMemReq io_req;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    size_t total = 0;

    if (!pool || ulcnet_prepost_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }
    if (pp_descriptor_for(cfg, &descriptor) != 0) return NULL;
    if (ulcnet_model_io_get_mem_requirements(&descriptor, &io_req) != 0)
        return NULL;
    pp_buffers(cfg, &buffers);

    /* Zero the whole control region first so a poisoned pool initialises like a zeroed one. */
    memset(pool, 0, (size_t)req.bytes);
    p = (UlcnetPrepost *)pool;

    if (pp_layout(p, (unsigned char *)pool, &descriptor, &buffers, &io_req,
                  &total) != 0)
        return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    p->n_bins = cfg->grid.n_bins;
    p->hop = cfg->grid.hop;
    return p;
}

UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes,
                                   const UlcnetPrepostConfig *cfg) {
    return ulcnet_prepost_init_ex(pool, bytes, cfg, NULL);
}

UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg) {
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    void *mem = NULL;

    if (ulcnet_prepost_get_mem_size(cfg, &req) != 0) return NULL;
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    mem = aligned_alloc((size_t)req.alignment, (size_t)req.bytes);
#else
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        mem = NULL;
#endif
    if (!mem) return NULL;
    p = ulcnet_prepost_init(mem, (size_t)req.bytes, cfg);
    if (!p) {
        free(mem);
        return NULL;
    }
    p->heap_base = mem;
    return p;
}

void ulcnet_prepost_destroy(UlcnetPrepost *p) {
    void *base;
    if (!p) return;
    base = p->heap_base;
    p->heap_base = NULL;   /* nothing to free for a pool instance; repeat calls are no-ops */
    if (base) free(base);
}

void ulcnet_prepost_reset(UlcnetPrepost *p) {
    if (!p) return;
    ulcnet_model_io_reset(p->io);
    memset(p->enh_re, 0, (size_t)p->n_bins * sizeof(float));
    memset(p->enh_im, 0, (size_t)p->n_bins * sizeof(float));
    p->frame_open = 0;
    p->prepared = 0;
}

uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p) {
    return p ? ulcnet_model_io_state_generation(p->io) : 0u;
}

int ulcnet_prepost_hop_size(const UlcnetPrepost *p) {
    return p ? p->hop : -1;
}

int ulcnet_prepost_num_bins(const UlcnetPrepost *p) {
    return p ? p->n_bins : -1;
}

const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p) {
    return p ? ulcnet_model_io_descriptor(p->io) : NULL;
}

int ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                    const float *err_re, const float *err_im,
                                    const float *far_re, const float *far_im) {
    size_t bins;

    if (!p || !err_re || !err_im || !far_re || !far_im) return -1;
    if (p->frame_open) return -1;   /* previous frame neither finished nor bypassed */

    bins = (size_t)p->n_bins * sizeof(float);
    memcpy(p->err_re, err_re, bins);
    memcpy(p->err_im, err_im, bins);
    memcpy(p->far_re, far_re, bins);
    memcpy(p->far_im, far_im, bins);

    p->frame_open = 1;
    return 1;
}

int ulcnet_prepost_frame_open(UlcnetPrepost *p,
                                UlcnetModelIoInputs *inputs,
                                UlcnetModelIoOutputs *outputs) {
    if (!p || !inputs || !outputs || !p->frame_open) return -1;
    /* Arms the transaction and NaN-fills the `output` mask, so a repeated request still gets a clean one. */
    if (ulcnet_model_io_prepare(p->io, p->err_re, p->err_im,
                                p->far_re, p->far_im, inputs, outputs) != 0)
        return -1;
    p->prepared = 1;
    return 0;
}

static void pp_close_frame(UlcnetPrepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
}

int ulcnet_prepost_frame_finish(UlcnetPrepost *p) {
    /* `prepared` is this class's own latch: after frame_bypass() model_io's transaction stays armed with stale
     * outputs, so its flag alone would let the next frame_finish accept them without frame_open(). */
    if (!p || !p->prepared) return -1;
    if (ulcnet_model_io_enhance_spectrum(p->io, p->enh_re, p->enh_im) != 0) {
        /* model_io restarted the recurrent state; the frame stays open for frame_bypass() or a fresh frame_open(). */
        p->prepared = 0;
        return -1;
    }
    pp_close_frame(p);
    return 0;
}

int ulcnet_prepost_frame_bypass(UlcnetPrepost *p) {
    size_t bins;
    if (!p || !p->frame_open) return -1;
    bins = (size_t)p->n_bins * sizeof(float);
    /* Identity: the error spectrum passes through; the armed transaction is not finished. */
    memcpy(p->enh_re, p->err_re, bins);
    memcpy(p->enh_im, p->err_im, bins);
    pp_close_frame(p);
    return 0;
}

int ulcnet_prepost_post_process_freq(UlcnetPrepost *p, float *re, float *im) {
    size_t bins;
    if (!p || !re || !im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->n_bins * sizeof(float);
    memcpy(re, p->enh_re, bins);
    memcpy(im, p->enh_im, bins);
    return 0;
}
