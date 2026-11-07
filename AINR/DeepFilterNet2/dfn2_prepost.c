/* DeepFilterNet2 pre/post class (contract: dfn2_prepost.h): combines dfn2_process.c and
 * dfn2_model_io.c and adds the lifecycle, the pool carve and the per-frame state machine.
 * Requires -ffp-contract=off. No heap in _init, no globals, no stdio. */

#include "dfn2_prepost.h"
#include "dfn2_model_io_priv.h"

#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PP_ALIGN ((size_t)DFN2_PREPOST_ALIGNMENT)

#ifndef DFN2_MASK_ONLY
#define PP_FRAME_COEFS(grid)  ((size_t)(grid)->df_bins * (size_t)(grid)->df_order * 2u)
#endif

/* The open frame is one of the first mask_lookahead frames after init or reset: its heads belong to a
 * frame before the stream start and are never used (N == 1 only; stable from pre_process to finish). */
#define PP_IN_WARMUP(p) \
    ((p)->grid.frames_per_call == 1 && dfn2_model_io_frame_slot((p)->io) < 0)

struct DFN2Prepost {
    DFN2State        *dsp;
    DFN2ModelIOState *io;
    DFN2Grid          grid;

    float atten_lim_db;

    const float *erb_fwd;   /* borrowed, never created or destroyed here */
    const float *erb_inv;

    /* Per-frame staging, one slot per frame of the call; a frame writes the slot
     * dfn2_model_io_frame_slot() gives after its push. */
    float *spec_re;     /* [N][n_bins]  frames, original scale */
    float *spec_im;
    /* apply_*: the dual entry point's application spectrum; enhanced_* point at what the enhance stage reads */
    float *apply_re;    /* [N][n_bins]                                      */
    float *apply_im;
    const float **enhanced_re;   /* [N]                                      */
    const float **enhanced_im;
    float *feat_erb;    /* [n_erb]                                          */
    float *feat_spec;   /* [2][df_bins], two contiguous segments            */
    float *enh_re;      /* [N][n_bins]  enhanced spectrum                   */
    float *enh_im;

    /* accelerator-writable heads (the ERB mask alone with DFN2_MASK_ONLY) */
    float *head_erb_mask;   /* [N][n_erb]                                   */
#ifndef DFN2_MASK_ONLY
    float *head_coefs;      /* [N][df_bins][df_order][2]                    */
    float *head_alpha;      /* [N]                                          */
#endif

    /* Identity heads for frame_bypass (unit band mask, zero coefficients; one frame, used for
     * every frame of the call), written once at init: the enhance stage refuses NULL heads. */
    float *skip_mask;       /* [n_erb], all 1.0f                            */
#ifndef DFN2_MASK_ONLY
    float *skip_coefs;      /* [df_bins * df_order * 2], all 0.0f           */
#endif

    /* The last call's N enhanced frames are emitted one per frame: slot `deliver`, 0 on the
     * frame of the commit. */
    int       deliver;
    int      *slot_emitted;   /* [N] */
    long long *slot_frame;    /* [N] */

    long long output_frame;  /* source frame of the last emission, -1 none  */
    int       frame_open;    /* a frame awaits commit or skip               */
    int       prepared;      /* frame_open() armed the accelerator transaction for the open frame */
    int       have_output;   /* this frame emits a spectrum                 */

    /* _create bookkeeping; 0 for an _init instance. */
    void *heap_base;
};

/* ---- pool arithmetic: audiolib's saturating helpers (overflow pins the cursor to SIZE_MAX) ---- */

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN2_PREPOST_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* Reserve `count` elements of `elem` bytes at *cursor, aligned; base==NULL sizes only. */
static int pp_carve(unsigned char *base, size_t *cursor, size_t count, size_t elem,
                    void **out) {
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, ck_mul_size(count, elem));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

static uint32_t pp_fnv1a_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

static uint32_t pp_fnv1a_str(uint32_t h, const char *s) {
    while (*s) {
        h ^= (uint32_t)(unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

/* Build identity a pool is valid for: the whole grid, the variant, the layout, carve and
 * feature versions, so a pool handed across a grid or checkpoint-contract change is refused.
 * The state registration is not in it: borrowed buffers change no pool byte. */
static uint32_t pp_build_hash(const DFN2Grid *grid) {
    uint32_t h = 2166136261u;
    const float tuning[] = {
        grid->erb_norm_tau_sec, grid->spec_norm_tau_sec,
        grid->erb_norm_alpha, grid->spec_norm_alpha, grid->stft_scale, grid->stft_unscale,
        grid->erb_norm_init_lo_db, grid->erb_norm_init_hi_db, grid->erb_norm_scale_db,
        grid->spec_norm_init_lo, grid->spec_norm_init_hi,
    };
    size_t i;
    DFN2_REQUIRE_VARIANT_LINK();
    h = pp_fnv1a_u32(h, (uint32_t)grid->sample_rate);
    h = pp_fnv1a_u32(h, (uint32_t)grid->n_fft);
    h = pp_fnv1a_u32(h, (uint32_t)grid->hop);
    h = pp_fnv1a_u32(h, (uint32_t)grid->n_bins);
    h = pp_fnv1a_u32(h, (uint32_t)grid->n_erb);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_bins);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_order);
    h = pp_fnv1a_u32(h, (uint32_t)grid->mask_lookahead);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_lookahead);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_history);
    h = pp_fnv1a_u32(h, (uint32_t)grid->ring);
    h = pp_fnv1a_u32(h, (uint32_t)grid->frames_per_call);
    h = pp_fnv1a_u32(h, (uint32_t)grid->input_frames);
#ifdef DFN2_MASK_ONLY
    h = pp_fnv1a_u32(h, 1u);
#else
    h = pp_fnv1a_u32(h, 0u);
#endif
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MODEL_IO_LAYOUT_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_PREPOST_CARVE_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MODEL_INPUT_FRAMES);
    h = pp_fnv1a_str(h, DFN2_FEATURE_VERSION);
    for (i = 0; i < sizeof tuning / sizeof tuning[0]; ++i) {
        uint32_t bits;
        memcpy(&bits, &tuning[i], sizeof bits);
        h = pp_fnv1a_u32(h, bits);
    }
    return ac_memreq_abi_hash(h);
}

typedef struct PpBlocks {
    void  *state;        /* DFN2State arrays */
    size_t state_bytes;
    void  *io;           /* DFN2ModelIOState windows */
    size_t io_bytes;
} PpBlocks;

/* The layout walk shared by get_mem_size and init so their order agrees: base==NULL sizes
 * it, otherwise it also carves (sub-blocks returned via blocks). */
static int pp_layout(DFN2Prepost *p, unsigned char *base,
                     const DFN2PrepostConfig *cfg, size_t *total,
                     PpBlocks *blocks) {
    const DFN2Grid *g = &cfg->grid;
    const size_t frames = (size_t)g->frames_per_call;
    const size_t frame_bins = frames * (size_t)g->n_bins;
    const size_t float_counts[] = {
        frame_bins,
        frame_bins,
        (size_t)g->n_erb,
        2u * (size_t)g->df_bins,
        frame_bins,
        frame_bins,
        dfn2_prepost_erb_mask_elements(g),
#ifndef DFN2_MASK_ONLY
        dfn2_prepost_coefs_elements(g),
        dfn2_prepost_alpha_elements(g),
#endif
        (size_t)g->n_erb
#ifndef DFN2_MASK_ONLY
        , PP_FRAME_COEFS(g)
#endif
    };
    size_t cursor = 0;
    size_t i;
    void *ptr = NULL;
    const size_t n_float_regions = sizeof float_counts / sizeof float_counts[0];
    float **slots[sizeof float_counts / sizeof float_counts[0]];

    blocks->state_bytes = dfn2_state_mem_size(g);
    blocks->io_bytes = dfn2_model_io_mem_size(g);
    if (blocks->state_bytes == 0u || blocks->io_bytes == 0u) return -1;

    if (pp_carve(base, &cursor, 1u, sizeof(DFN2Prepost), &ptr) != 0) return -1;
    if (pp_carve(base, &cursor, 1u, sizeof(DFN2State), &ptr) != 0) return -1;
    if (base && p) p->dsp = (DFN2State *)ptr;
    if (pp_carve(base, &cursor, blocks->state_bytes, 1u, &ptr) != 0) return -1;
    blocks->state = ptr;
    if (pp_carve(base, &cursor, 1u, sizeof(DFN2ModelIOState), &ptr) != 0) return -1;
    if (base && p) p->io = (DFN2ModelIOState *)ptr;
    if (pp_carve(base, &cursor, blocks->io_bytes, 1u, &ptr) != 0) return -1;
    blocks->io = ptr;

    if (p) {
        slots[0]  = &p->spec_re;       slots[1]  = &p->spec_im;
        slots[2]  = &p->feat_erb;      slots[3]  = &p->feat_spec;
        slots[4]  = &p->enh_re;        slots[5]  = &p->enh_im;
        slots[6]  = &p->head_erb_mask;
#ifdef DFN2_MASK_ONLY
        slots[7]  = &p->skip_mask;
#else
        slots[7]  = &p->head_coefs;
        slots[8]  = &p->head_alpha;    slots[9]  = &p->skip_mask;
        slots[10] = &p->skip_coefs;
#endif
    }
    for (i = 0; i < n_float_regions; ++i) {
        if (pp_carve(base, &cursor, float_counts[i], sizeof(float), &ptr) != 0)
            return -1;
        if (base && p) *slots[i] = (float *)ptr;
    }

    if (pp_carve(base, &cursor, frames, sizeof(const float *), &ptr) != 0) return -1;
    if (base && p) p->enhanced_re = (const float **)ptr;
    if (pp_carve(base, &cursor, frames, sizeof(const float *), &ptr) != 0) return -1;
    if (base && p) p->enhanced_im = (const float **)ptr;
    if (pp_carve(base, &cursor, frames, sizeof(int), &ptr) != 0) return -1;
    if (base && p) p->slot_emitted = (int *)ptr;
    if (pp_carve(base, &cursor, frames, sizeof(long long), &ptr) != 0) return -1;
    if (base && p) p->slot_frame = (long long *)ptr;

    if (pp_carve(base, &cursor, frame_bins, sizeof(float), &ptr) != 0) return -1;
    if (base && p) p->apply_re = (float *)ptr;
    if (pp_carve(base, &cursor, frame_bins, sizeof(float), &ptr) != 0) return -1;
    if (base && p) p->apply_im = (float *)ptr;

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

int dfn2_prepost_config_defaults(DFN2PrepostConfig *cfg, const DFN2Grid *grid) {
    if (!cfg) return -1;
    if (dfn2_grid_validate(grid) != DFN2_GRID_OK) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->grid = *grid;
    cfg->atten_lim_db = 0.0f;
    return 0;
}

static int pp_check_config(const DFN2PrepostConfig *cfg) {
    if (!cfg) return -1;
    if (dfn2_grid_validate(&cfg->grid) != DFN2_GRID_OK) return -1;
    /* Mask expansion and the ERB feature branch run on the spectrum side. */
    if (!cfg->erb_fwd || !cfg->erb_inv) return -1;
    if (!isfinite(cfg->atten_lim_db)) return -1;
    if (dfn2_registration_check(cfg->gru_hidden, cfg->gru_hidden_elements,
                                cfg->df_convp_history,
                                cfg->df_convp_history_elements) != 0)
        return -1;
    return 0;
}

int dfn2_prepost_get_mem_size(const DFN2PrepostConfig *cfg,
                              DFN2PrepostMemReq *req) {
    size_t total = 0;
    PpBlocks blocks;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_layout(NULL, NULL, cfg, &total, &blocks) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DFN2_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = (uint32_t)DFN2_MODEL_IO_LAYOUT_VERSION;
    req->build_flags_hash = pp_build_hash(&cfg->grid);
    req->alignment = (uint32_t)PP_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* DFN2State and DFN2ModelIOState (with the registered recurrent state) back to their init
 * values; the DSP state keeps the borrowed ERB matrices. */
static void pp_reset_states(DFN2Prepost *p) {
    dfn2_state_reset(p->dsp);
    dfn2_model_io_reset(p->io);
}

/* Bookkeeping only: the staging buffers are zero from init/reset, the warm-up frames return
 * before writing them and every later frame writes them wholesale. */
static void pp_clear_frame(DFN2Prepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    p->have_output = 0;
}

static void pp_zero_staging(DFN2Prepost *p) {
    const size_t frames = (size_t)p->grid.frames_per_call;
    const size_t frame_bins = frames * (size_t)p->grid.n_bins;
    memset(p->enh_re, 0, frame_bins * sizeof(float));
    memset(p->enh_im, 0, frame_bins * sizeof(float));
    p->deliver = 0;
    memset(p->slot_emitted, 0, frames * sizeof(int));
    memset(p->slot_frame, 0, frames * sizeof(long long));
}

DFN2Prepost *dfn2_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN2PrepostConfig *cfg,
                                  const DFN2PrepostMemReq *expected) {
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    size_t total = 0;
    PpBlocks blocks;
    int i;

    if (!pool || dfn2_prepost_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        /* Stale-pool gate: a pool sized by another build or grid is refused. */
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->reserved0 != req.reserved0 ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }

    /* Zero the whole pool first, the control region included. */
    memset(pool, 0, (size_t)req.bytes);
    p = (DFN2Prepost *)pool;
    p->grid = cfg->grid;
    p->atten_lim_db = cfg->atten_lim_db;
    p->erb_fwd = cfg->erb_fwd;
    p->erb_inv = cfg->erb_inv;

    if (pp_layout(p, (unsigned char *)pool, cfg, &total, &blocks) != 0)
        return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    for (i = 0; i < cfg->grid.n_erb; ++i) p->skip_mask[i] = 1.0f;
#ifndef DFN2_MASK_ONLY
    memset(p->skip_coefs, 0, PP_FRAME_COEFS(&cfg->grid) * sizeof(float));
#endif

    if (dfn2_state_init(p->dsp, blocks.state, blocks.state_bytes, &p->grid) != 0 ||
        dfn2_model_io_init(p->io, blocks.io, blocks.io_bytes, &p->grid, cfg->gru_hidden,
                           cfg->gru_hidden_elements, cfg->df_convp_history,
                           cfg->df_convp_history_elements) != 0)
        return NULL;       /* staging is already zero from the pool memset */
    dfn2_set_erb_matrices(p->dsp, p->erb_fwd, p->erb_inv);
    pp_clear_frame(p);
    p->output_frame = -1;
    return p;
}

DFN2Prepost *dfn2_prepost_init(void *pool, size_t bytes,
                               const DFN2PrepostConfig *cfg) {
    return dfn2_prepost_init_ex(pool, bytes, cfg, NULL);
}

DFN2Prepost *dfn2_prepost_create(const DFN2PrepostConfig *cfg) {
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *mem = NULL;

    if (dfn2_prepost_get_mem_size(cfg, &req) != 0) return NULL;
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    mem = aligned_alloc((size_t)req.alignment, (size_t)req.bytes);
#else
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        mem = NULL;
#endif
    if (!mem) return NULL;
    p = dfn2_prepost_init(mem, (size_t)req.bytes, cfg);
    if (!p) {
        free(mem);
        return NULL;
    }
    p->heap_base = mem;
    return p;
}

void dfn2_prepost_destroy(DFN2Prepost *p) {
    void *base;
    if (!p) return;
    base = p->heap_base;
    p->heap_base = NULL;   /* nothing to free for a pool instance; repeat calls are no-ops */
    if (base) free(base);
}

void dfn2_prepost_reset(DFN2Prepost *p) {
    if (!p) return;
    pp_reset_states(p);
    pp_zero_staging(p);
    pp_clear_frame(p);
    p->output_frame = -1;
}

unsigned int dfn2_prepost_state_generation(const DFN2Prepost *p) {
    return p ? dfn2_model_io_state_generation(p->io) : 0u;
}

int dfn2_prepost_hop_size(const DFN2Prepost *p) {
    return p ? p->grid.hop : -1;
}

int dfn2_prepost_num_bins(const DFN2Prepost *p) {
    return p ? p->grid.n_bins : -1;
}

const DFN2Grid *dfn2_prepost_grid(const DFN2Prepost *p) {
    return p ? &p->grid : NULL;
}

int dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p) {
    return p ? dfn2_prepost_latency_frames(&p->grid) : -1;
}

int dfn2_prepost_layout_version(const DFN2Prepost *p) {
    return p ? DFN2_MODEL_IO_LAYOUT_VERSION : -1;
}

int dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd,
                                  const float *erb_inv) {
    if (!p || !erb_fwd || !erb_inv) return -1;
    /* between hops only: a swap must not land inside an open transaction */
    if (p->frame_open) return -1;
    p->erb_fwd = erb_fwd;
    p->erb_inv = erb_inv;
    dfn2_set_erb_matrices(p->dsp, erb_fwd, erb_inv);
    return 0;
}

int dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db) {
    if (!p || !isfinite(atten_lim_db)) return -1;
    if (p->frame_open) return -1;   /* between hops only */
    p->atten_lim_db = atten_lim_db;
    return 0;
}

int dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame) {
    if (!p || !frame || p->output_frame < 0) return -1;
    *frame = p->output_frame;
    return 0;
}

/* ---- per-frame stages ------------------------------------------------ */

static int pp_all_finite(const float *values, size_t count) {
    return skn_all_finite_f32(values, count);
}

static void pp_fill_nan(float *values, size_t count) {
    skn_fill_f32(values, count, (float)NAN);
}

/* Slot the next push fills. */
static int pp_input_slot(const DFN2Prepost *p) {
    return (dfn2_model_io_frame_slot(p->io) + 1) % p->grid.frames_per_call;
}

static int pp_deliver_slot(const DFN2Prepost *p) {
    return p->deliver;
}

static void pp_deliver(DFN2Prepost *p, int slot) {
    p->deliver = slot;
    p->have_output = p->slot_emitted[slot];
    if (p->have_output) p->output_frame = p->slot_frame[slot];
}

/* Features and graph window, shared by every pre_process entry point; with N > 1 the first
 * L frames also advance the enhance clock here. `est_*` is read only inside this call; the hop's
 * source is already in enhanced_*[slot]. Returns the accelerator invocations this hop needs
 * (0 or 1; N == 1: always 1, the first L frames are warm-up frames). */
static int pp_begin_frame(DFN2Prepost *p, const float *est_re,
                          const float *est_im, int slot) {
    int heads_needed;

    pp_clear_frame(p);
    dfn2_compute_features(p->dsp, est_re, est_im, p->feat_erb,
                          p->feat_spec);
    heads_needed = dfn2_model_io_push_features(p->io, p->feat_erb, p->feat_spec);
    if (heads_needed < 0) return -1;

    if (heads_needed == 0 && p->grid.frames_per_call > 1) {
        if (dfn2_model_io_frame_slot(p->io) < 0) {
            /* N > 1 warm-up: the clock advances with heads_valid=0, which
             * dfn2_enhance_spectrum_stream() accepts exactly while current < mask_lookahead. */
            if (dfn2_enhance_spectrum_stream(p->dsp, p->enhanced_re[slot],
                                    p->enhanced_im[slot], 0, NULL, NULL, 0.0f,
                                    p->atten_lim_db, p->enh_re, p->enh_im,
                                    NULL) != 0) {
                return -1;
            }
            return 0;
        }
        /* the frame waits in its slot; this call emits the next frame of the previous call */
        pp_deliver(p, slot + 1);
        return 0;
    }
    p->frame_open = 1;
    return 1;
}

int dfn2_prepost_pre_process_freq(DFN2Prepost *p, const float *spec_re,
                                  const float *spec_im) {
    int slot;
    float *re;
    float *im;
    size_t bins;
    if (!p || !spec_re || !spec_im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->grid.n_bins * sizeof(float);
    slot = pp_input_slot(p);
    re = p->spec_re + (size_t)slot * (size_t)p->grid.n_bins;
    im = p->spec_im + (size_t)slot * (size_t)p->grid.n_bins;
    memcpy(re, spec_re, bins);
    memcpy(im, spec_im, bins);
    p->enhanced_re[slot] = re;
    p->enhanced_im[slot] = im;
    return pp_begin_frame(p, re, im, slot);
}

int dfn2_prepost_pre_process_freq_dual(DFN2Prepost *p, const float *est_re,
                                       const float *est_im,
                                       const float *app_re,
                                       const float *app_im) {
    int slot;
    float *re;
    float *im;
    size_t bins;
    if (!p || !est_re || !est_im || !app_re || !app_im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->grid.n_bins * sizeof(float);
    /* the application spectrum is read again at commit, so it is copied */
    slot = pp_input_slot(p);
    re = p->apply_re + (size_t)slot * (size_t)p->grid.n_bins;
    im = p->apply_im + (size_t)slot * (size_t)p->grid.n_bins;
    memcpy(re, app_re, bins);
    memcpy(im, app_im, bins);
    p->enhanced_re[slot] = re;
    p->enhanced_im[slot] = im;
    return pp_begin_frame(p, est_re, est_im, slot);
}

int dfn2_prepost_frame_open(DFN2Prepost *p, DFN2PrepostInputs *inputs,
                              DFN2PrepostOutputs *outputs) {
    if (!p || !inputs || !outputs || !p->frame_open) return -1;

    memset(inputs, 0, sizeof(*inputs));
    inputs->erb_window = p->io->erb_window;
    inputs->spec_window = p->io->spec_window;
    inputs->gru_hidden = p->io->gru_hidden;
    inputs->gru_hidden_elements = p->io->gru_hidden_elements;
#ifndef DFN2_MASK_ONLY
    inputs->df_convp_window = p->io->df_convp_window;
    inputs->df_convp_window_elements = p->io->df_convp_window_elements;
#endif
    inputs->erb_window_elements = dfn2_prepost_erb_window_elements(&p->grid);
    inputs->spec_window_elements = dfn2_prepost_spec_window_elements(&p->grid);

    /* NaN-fill the heads so a partial accelerator write fails commit; the recurrent state is
     * the accelerator's input too and is left as is. */
    pp_fill_nan(p->head_erb_mask, dfn2_prepost_erb_mask_elements(&p->grid));
#ifndef DFN2_MASK_ONLY
    pp_fill_nan(p->head_coefs, dfn2_prepost_coefs_elements(&p->grid));
    pp_fill_nan(p->head_alpha, dfn2_prepost_alpha_elements(&p->grid));
#endif

    memset(outputs, 0, sizeof(*outputs));
    outputs->erb_mask = p->head_erb_mask;
    outputs->gru_hidden_next = p->io->gru_hidden;
    outputs->erb_mask_elements = dfn2_prepost_erb_mask_elements(&p->grid);
    outputs->gru_hidden_elements = p->io->gru_hidden_elements;
#ifndef DFN2_MASK_ONLY
    outputs->coefs = p->head_coefs;
    outputs->alpha = p->head_alpha;
    outputs->df_convp_window_next = p->io->df_convp_window;
    outputs->coefs_elements = dfn2_prepost_coefs_elements(&p->grid);
    outputs->alpha_elements = dfn2_prepost_alpha_elements(&p->grid);
    outputs->df_convp_window_elements = p->io->df_convp_window_elements;
#endif
    p->prepared = 1;
    return 0;
}

#ifndef DFN2_MASK_ONLY
static const float pp_skip_alpha[1];
#endif

/* Run the enhance stage per frame of the call in order: frame i reads its heads at i * stride
 * frames into the arrays (stride 0 repeats one frame's heads). Closes the frame on success only;
 * the clock advances once per frame, here or (N > 1 warm-up) in pp_begin_frame. A warm-up frame
 * (N == 1) advances the clock without heads, zeroes the registered state and emits nothing. */
static int pp_close_frame(DFN2Prepost *p, const float *erb_mask,
                          const float *coefs, const float *alpha,
                          int stride) {
    int i;

#ifdef DFN2_MASK_ONLY
    (void)coefs;   /* no deep filter: only the ERB mask head exists */
    (void)alpha;
#endif
    if (PP_IN_WARMUP(p)) {
        if (dfn2_enhance_spectrum_stream(p->dsp, p->enhanced_re[0], p->enhanced_im[0], 0,
                                         NULL, NULL, 0.0f, p->atten_lim_db, p->enh_re,
                                         p->enh_im, NULL) != 0)
            return -1;
        dfn2_model_io_zero_recurrent(p->io);
        p->frame_open = 0;   /* have_output stays 0 from pp_begin_frame, output_frame untouched */
        p->prepared = 0;
        return 0;
    }
    for (i = 0; i < p->grid.frames_per_call; ++i) {
        const size_t head = (size_t)i * (size_t)stride;
        float *enh_re = p->enh_re + (size_t)i * (size_t)p->grid.n_bins;
        float *enh_im = p->enh_im + (size_t)i * (size_t)p->grid.n_bins;
        long long frame = 0;
        int emitted;

        /* The enhance stage validates pointers, head finiteness and clock alignment before its
         * first write, so -1 leaves its state untouched and the frame open for frame_bypass();
         * only the first frame can return -1. */
#ifdef DFN2_MASK_ONLY
        emitted = dfn2_enhance_spectrum_stream(p->dsp, p->enhanced_re[i],
                                      p->enhanced_im[i], 1,
                                      erb_mask + head * (size_t)p->grid.n_erb, NULL,
                                      0.0f, p->atten_lim_db, enh_re,
                                      enh_im, &frame);
#else
        emitted = dfn2_enhance_spectrum_stream(p->dsp, p->enhanced_re[i],
                                      p->enhanced_im[i], 1,
                                      erb_mask + head * (size_t)p->grid.n_erb,
                                      coefs + head * PP_FRAME_COEFS(&p->grid),
                                      alpha[head], p->atten_lim_db, enh_re,
                                      enh_im, &frame);
#endif
        if (emitted < 0) return -1;
        p->slot_emitted[i] = emitted;   /* 0: still inside the lookahead warm-up */
        p->slot_frame[i] = frame;
    }
    p->frame_open = 0;
    p->prepared = 0;
    pp_deliver(p, 0);
    return 0;
}

int dfn2_prepost_frame_finish(DFN2Prepost *p) {
    int heads_finite;
    int state_finite;
    if (!p || !p->prepared) return -1;   /* prepared implies frame_open */
    if (PP_IN_WARMUP(p)) return dfn2_prepost_frame_bypass(p);   /* heads never read */

    /* The accelerator overwrote the recurrent state in place: a non-finite head or state element
     * zeroes both registered tensors, advances the state generation and disarms the transaction;
     * the frame stays open. */
    heads_finite =
#ifdef DFN2_MASK_ONLY
        pp_all_finite(p->head_erb_mask,
                      dfn2_prepost_erb_mask_elements(&p->grid));
#else
        pp_all_finite(p->head_erb_mask,
                      dfn2_prepost_erb_mask_elements(&p->grid)) &&
        pp_all_finite(p->head_coefs, dfn2_prepost_coefs_elements(&p->grid)) &&
        pp_all_finite(p->head_alpha, dfn2_prepost_alpha_elements(&p->grid));
#endif
    state_finite = dfn2_model_io_validate_state(p->io) == 0;
    if (!state_finite || !heads_finite) {
        /* validate_state already zeroed a bad state; a bad head still restarts it */
        if (state_finite) dfn2_model_io_zero_recurrent(p->io);
        p->prepared = 0;
        return -1;
    }

    /* the one step with no undo: nothing can fail after it */
#ifdef DFN2_MASK_ONLY
    return pp_close_frame(p, p->head_erb_mask, NULL, NULL, 1);
#else
    return pp_close_frame(p, p->head_erb_mask, p->head_coefs, p->head_alpha,
                          1);
#endif
}

int dfn2_prepost_frame_bypass(DFN2Prepost *p) {
    if (!p || !p->frame_open) return -1;
    /* Identity: a unit band mask expands to unit bin gain through the partition-of-unity
     * erb_inv and alpha 0 selects the masked residual; the clocks still advance. */
#ifdef DFN2_MASK_ONLY
    return pp_close_frame(p, p->skip_mask, NULL, NULL, 0);
#else
    return pp_close_frame(p, p->skip_mask, p->skip_coefs, pp_skip_alpha, 0);
#endif
}

int dfn2_prepost_post_process_freq(DFN2Prepost *p, float *re, float *im,
                                   int *valid) {
    size_t bins;
    size_t at;
    if (!p || !re || !im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->grid.n_bins * sizeof(float);
    at = (size_t)pp_deliver_slot(p) * (size_t)p->grid.n_bins;
    memcpy(re, p->enh_re + at, bins);
    memcpy(im, p->enh_im + at, bins);
    if (valid) *valid = p->have_output;
    return 0;
}

int dfn2_model_run_frame(const DFN2Model *model, DFN2Prepost *p) {
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    if (!model || !model->infer || !p || !p->frame_open ||
        dfn2_model_io_descriptor_validate(&p->grid, model->io_descriptor) != 0)
        return -1;
    if (dfn2_prepost_frame_open(p, &inputs, &outputs) != 0) return -1;
    if (model->infer(model->user, &inputs, &outputs) != 0) {
        return dfn2_prepost_frame_bypass(p) == 0 ? 0 : -1;
    }
    if (dfn2_prepost_frame_finish(p) != 0) {
        /* refused commit: take the frame as the identity so the clocks advance once */
        return dfn2_prepost_frame_bypass(p) == 0 ? 0 : -1;
    }
    return 1;
}
