/* DeepFilterNet3 pre/post class (contract: dfn3_prepost.h): dfn3_process.c and dfn3_model_io.c combined with
 * the lifecycle, the pool carve and the per-frame state machine. Requires -ffp-contract=off. No heap in
 * _init, no globals, no stdio. */

#include "dfn3_prepost.h"

#include "dfn3_model_io_priv.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PP_ALIGN ((size_t)DFN3_PREPOST_ALIGNMENT)

/* The open frame is one of the first mask_lookahead frames after init or reset: its heads belong to a
 * frame before the stream start and are never used (stable from pre_process to finish). */
#define PP_IN_WARMUP(p) \
    ((p)->io->feature_frames_seen < (unsigned long long)dfn3_model_io_ready_frames((p)->io))

struct DFN3Prepost {
    DFN3State        *dsp;
    DFN3ModelIOState *io;
    DFN3Grid          grid;

    float atten_lim_db;

    const float *erb_fwd;   /* borrowed */
    const float *erb_inv;

    float *spec_re;     /* [n_bins] current frame */
    float *spec_im;
    float *apply_re;    /* [n_bins] application spectrum of the dual entry */
    float *apply_im;
    const float *enhanced_re;   /* what the enhance stage reads this frame: spec_* or apply_* */
    const float *enhanced_im;
    float *feat_erb;    /* [n_erb] */
    float *feat_spec;   /* [2][df_bins] */
    float *enh_re;      /* [n_bins] enhanced spectrum */
    float *enh_im;

    float *head_erb_mask;   /* [n_erb] */
    float *head_coefs;      /* [df_bins][df_order][2] */

    /* Identity heads of frame_bypass (the stream entry refuses NULL heads). */
    float *skip_mask;       /* [n_erb], all 1 */
    float *skip_coefs;      /* [df_bins][df_order][2], 1 on the tap df_history real part */

    int       frame_open;    /* a frame awaits commit or bypass */
    int       prepared;      /* frame_open() armed the transaction */
    int       have_output;   /* the enhance stage emitted this frame */
    long long output_frame;  /* source frame of the last emission, -1 none */
    void *heap_base;         /* set by _create; NULL for an _init instance */
};

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN3_PREPOST_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* Reserve `count` elements of `elem` bytes at *cursor, 16-byte aligned; base==NULL sizes only. Overflow
 * saturates (audiolib mem_align.h). */
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

/* Build identity a pool is valid for: the whole grid, the layout, carve and feature versions. The state
 * registration is not in it (borrowed buffers change no pool byte). */
static uint32_t pp_build_hash(const DFN3Grid *grid) {
    uint32_t h = 2166136261u;
    const float tuning[] = {
        grid->erb_norm_tau_sec, grid->spec_norm_tau_sec,
        grid->erb_norm_alpha, grid->spec_norm_alpha, grid->stft_scale, grid->stft_unscale,
        grid->erb_norm_init_lo_db, grid->erb_norm_init_hi_db, grid->erb_norm_scale_db,
        grid->spec_norm_init_lo, grid->spec_norm_init_hi,
    };
    size_t i;
    h = pp_fnv1a_u32(h, (uint32_t)grid->sample_rate);
    h = pp_fnv1a_u32(h, (uint32_t)grid->n_fft);
    h = pp_fnv1a_u32(h, (uint32_t)grid->hop);
    h = pp_fnv1a_u32(h, (uint32_t)grid->n_bins);
    h = pp_fnv1a_u32(h, (uint32_t)grid->n_erb);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_bins);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_order);
    h = pp_fnv1a_u32(h, (uint32_t)grid->mask_lookahead);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_lookahead);
    h = pp_fnv1a_u32(h, (uint32_t)grid->output_delay);
    h = pp_fnv1a_u32(h, (uint32_t)grid->df_history);
    h = pp_fnv1a_u32(h, (uint32_t)grid->ring);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_IO_LAYOUT_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_PREPOST_CARVE_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_INPUT_FRAMES);
    h = pp_fnv1a_str(h, DFN3_FEATURE_VERSION);
    for (i = 0; i < sizeof tuning / sizeof tuning[0]; ++i) {
        uint32_t bits;
        memcpy(&bits, &tuning[i], sizeof bits);
        h = pp_fnv1a_u32(h, bits);
    }
    return ac_memreq_abi_hash(h);
}

/* The sub-blocks pp_layout() carves for the other objects. */
typedef struct PpBlocks {
    void  *state;        /* DFN3State arrays */
    size_t state_bytes;
    void  *io;           /* DFN3ModelIOState windows */
    size_t io_bytes;
} PpBlocks;

/* The one layout walk: base==NULL sizes, otherwise it also carves. get_mem_size and init share it. */
static int pp_layout(DFN3Prepost *p, unsigned char *base,
                     const DFN3PrepostConfig *cfg, size_t *total,
                     PpBlocks *blocks) {
    const DFN3Grid *g = &cfg->grid;
    const size_t bins = (size_t)g->n_bins;
    const size_t float_counts[] = {
        bins,
        bins,
        (size_t)g->n_erb,
        2u * (size_t)g->df_bins,
        bins,
        bins,
        dfn3_prepost_erb_mask_elements(g),
        dfn3_prepost_coefs_elements(g),
        dfn3_prepost_erb_mask_elements(g),
        dfn3_prepost_coefs_elements(g),
        bins,
        bins,
    };
    const size_t n_float_regions = sizeof float_counts / sizeof float_counts[0];
    float **slots[sizeof float_counts / sizeof float_counts[0]];
    size_t cursor = 0;
    size_t i;
    void *ptr = NULL;

    blocks->state_bytes = dfn3_state_mem_size(g);
    blocks->io_bytes = dfn3_model_io_mem_size(g);
    if (blocks->state_bytes == 0u || blocks->io_bytes == 0u) return -1;

    if (pp_carve(base, &cursor, 1u, sizeof(DFN3Prepost), &ptr) != 0) return -1;
    if (pp_carve(base, &cursor, 1u, sizeof(DFN3State), &ptr) != 0) return -1;
    if (base && p) p->dsp = (DFN3State *)ptr;
    if (pp_carve(base, &cursor, blocks->state_bytes, 1u, &ptr) != 0) return -1;
    blocks->state = ptr;
    if (pp_carve(base, &cursor, 1u, sizeof(DFN3ModelIOState), &ptr) != 0) return -1;
    if (base && p) p->io = (DFN3ModelIOState *)ptr;
    if (pp_carve(base, &cursor, blocks->io_bytes, 1u, &ptr) != 0) return -1;
    blocks->io = ptr;

    if (p) {
        slots[0]  = &p->spec_re;       slots[1]  = &p->spec_im;
        slots[2]  = &p->feat_erb;      slots[3]  = &p->feat_spec;
        slots[4]  = &p->enh_re;        slots[5]  = &p->enh_im;
        slots[6]  = &p->head_erb_mask; slots[7]  = &p->head_coefs;
        slots[8]  = &p->skip_mask;     slots[9]  = &p->skip_coefs;
        slots[10] = &p->apply_re;      slots[11] = &p->apply_im;
    }
    for (i = 0; i < n_float_regions; ++i) {
        if (pp_carve(base, &cursor, float_counts[i], sizeof(float), &ptr) != 0)
            return -1;
        if (base && p) *slots[i] = (float *)ptr;
    }

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

/* ---- config / sizing ---- */

int dfn3_prepost_config_defaults(DFN3PrepostConfig *cfg, const DFN3Grid *grid) {
    if (!cfg) return -1;
    if (dfn3_grid_validate(grid) != DFN3_GRID_OK) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->grid = *grid;
    cfg->atten_lim_db = 0.0f;
    return 0;
}

static int pp_check_config(const DFN3PrepostConfig *cfg) {
    if (!cfg) return -1;
    if (dfn3_grid_validate(&cfg->grid) != DFN3_GRID_OK) return -1;
    if (!cfg->erb_fwd || !cfg->erb_inv) return -1;
    if (!isfinite(cfg->atten_lim_db)) return -1;
    if (dfn3_registration_check(cfg->gru_hidden, cfg->gru_hidden_elements,
                                cfg->df_convp_history,
                                cfg->df_convp_history_elements) != 0)
        return -1;
    return 0;
}

int dfn3_prepost_get_mem_size(const DFN3PrepostConfig *cfg,
                              DFN3PrepostMemReq *req) {
    size_t total = 0;
    PpBlocks blocks;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_layout(NULL, NULL, cfg, &total, &blocks) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DFN3_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = (uint32_t)DFN3_MODEL_IO_LAYOUT_VERSION;
    req->build_flags_hash = pp_build_hash(&cfg->grid);
    req->alignment = (uint32_t)PP_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* ---- lifecycle ---- */

/* DFN3State and DFN3ModelIOState (with the registered state) to their init values. */
static void pp_reset_states(DFN3Prepost *p) {
    dfn3_state_reset(p->dsp);
    dfn3_model_io_reset(p->io);
}

/* The staging buffers are not cleared here: zero from init/reset, the enhance stage writes them whole. */
static void pp_clear_frame(DFN3Prepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    p->have_output = 0;
}

static void pp_zero_staging(DFN3Prepost *p) {
    const size_t bins = (size_t)p->grid.n_bins * sizeof(float);
    memset(p->enh_re, 0, bins);
    memset(p->enh_im, 0, bins);
}

DFN3Prepost *dfn3_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN3PrepostConfig *cfg,
                                  const DFN3PrepostMemReq *expected) {
    DFN3PrepostMemReq req;
    DFN3Prepost *p;
    size_t total = 0;
    PpBlocks blocks;
    int i;

    if (!pool || dfn3_prepost_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->reserved0 != req.reserved0 ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }

    memset(pool, 0, (size_t)req.bytes);
    p = (DFN3Prepost *)pool;
    p->grid = cfg->grid;
    p->atten_lim_db = cfg->atten_lim_db;
    p->erb_fwd = cfg->erb_fwd;
    p->erb_inv = cfg->erb_inv;

    if (pp_layout(p, (unsigned char *)pool, cfg, &total, &blocks) != 0)
        return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    for (i = 0; i < cfg->grid.n_erb; ++i) p->skip_mask[i] = 1.0f;
    for (i = 0; i < cfg->grid.df_bins; ++i)
        p->skip_coefs[((size_t)i * (size_t)cfg->grid.df_order +
                       (size_t)cfg->grid.df_history) * 2u] = 1.0f;

    if (dfn3_state_init(p->dsp, blocks.state, blocks.state_bytes, &p->grid) != 0 ||
        dfn3_model_io_init(p->io, blocks.io, blocks.io_bytes, &p->grid, cfg->gru_hidden,
                           cfg->gru_hidden_elements, cfg->df_convp_history,
                           cfg->df_convp_history_elements) != 0)
        return NULL;       /* staging is already zero from the pool memset */
    dfn3_set_erb_matrices(p->dsp, p->erb_fwd, p->erb_inv);
    pp_clear_frame(p);
    p->output_frame = -1;
    return p;
}

DFN3Prepost *dfn3_prepost_init(void *pool, size_t bytes,
                               const DFN3PrepostConfig *cfg) {
    return dfn3_prepost_init_ex(pool, bytes, cfg, NULL);
}

DFN3Prepost *dfn3_prepost_create(const DFN3PrepostConfig *cfg) {
    DFN3PrepostMemReq req;
    DFN3Prepost *p;
    void *mem = NULL;

    if (dfn3_prepost_get_mem_size(cfg, &req) != 0) return NULL;
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    mem = aligned_alloc((size_t)req.alignment, (size_t)req.bytes);
#else
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        mem = NULL;
#endif
    if (!mem) return NULL;
    p = dfn3_prepost_init(mem, (size_t)req.bytes, cfg);
    if (!p) {
        free(mem);
        return NULL;
    }
    p->heap_base = mem;
    return p;
}

void dfn3_prepost_destroy(DFN3Prepost *p) {
    void *base;
    if (!p) return;
    base = p->heap_base;
    p->heap_base = NULL;   /* repeat calls are no-ops */
    if (base) free(base);
}

void dfn3_prepost_reset(DFN3Prepost *p) {
    if (!p) return;
    pp_reset_states(p);
    pp_zero_staging(p);
    pp_clear_frame(p);
    p->output_frame = -1;
}

unsigned int dfn3_prepost_state_generation(const DFN3Prepost *p) {
    return p ? dfn3_model_io_state_generation(p->io) : 0u;
}

/* ---- accessors ---- */

int dfn3_prepost_hop_size(const DFN3Prepost *p) {
    return p ? p->grid.hop : -1;
}

int dfn3_prepost_num_bins(const DFN3Prepost *p) {
    return p ? p->grid.n_bins : -1;
}

const DFN3Grid *dfn3_prepost_grid(const DFN3Prepost *p) {
    return p ? &p->grid : NULL;
}

int dfn3_prepost_model_lookahead_frames(const DFN3Prepost *p) {
    return p ? p->grid.output_delay : -1;
}

int dfn3_prepost_layout_version(const DFN3Prepost *p) {
    return p ? DFN3_MODEL_IO_LAYOUT_VERSION : -1;
}

int dfn3_prepost_set_erb_matrices(DFN3Prepost *p, const float *erb_fwd,
                                  const float *erb_inv) {
    if (!p || !erb_fwd || !erb_inv) return -1;
    if (p->frame_open) return -1;
    p->erb_fwd = erb_fwd;
    p->erb_inv = erb_inv;
    dfn3_set_erb_matrices(p->dsp, erb_fwd, erb_inv);
    return 0;
}

int dfn3_prepost_set_atten_lim(DFN3Prepost *p, float atten_lim_db) {
    if (!p || !isfinite(atten_lim_db)) return -1;
    if (p->frame_open) return -1;
    p->atten_lim_db = atten_lim_db;
    return 0;
}

int dfn3_prepost_output_frame_index(const DFN3Prepost *p, long long *frame) {
    if (!p || !frame || p->output_frame < 0) return -1;
    *frame = p->output_frame;
    return 0;
}

/* ---- per-frame stages ---- */

static int pp_all_finite(const float *values, size_t count) {
    return skn_all_finite_f32(values, count);
}

static void pp_fill_nan(float *values, size_t count) {
    skn_fill_f32(values, count, (float)NAN);
}

/* Features, graph window and frame open, shared by both entries. Returns 1: the accelerator runs on every
 * frame, the first mask_lookahead being warm-up frames. */
static int pp_begin_frame(DFN3Prepost *p, const float *est_re,
                          const float *est_im) {
    int heads_needed;

    pp_clear_frame(p);
    dfn3_compute_features(p->dsp, est_re, est_im, p->feat_erb, p->feat_spec);
    heads_needed = dfn3_model_io_push_features(p->io, p->feat_erb, p->feat_spec);
    if (heads_needed < 0) return -1;

    p->frame_open = 1;
    return 1;
}

int dfn3_prepost_pre_process_freq(DFN3Prepost *p, const float *spec_re,
                                  const float *spec_im) {
    size_t bins;
    if (!p || !spec_re || !spec_im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->grid.n_bins;
    if (!skn_all_finite_f32(spec_re, bins) || !skn_all_finite_f32(spec_im, bins))
        return -1;
    memcpy(p->spec_re, spec_re, bins * sizeof(float));
    memcpy(p->spec_im, spec_im, bins * sizeof(float));
    p->enhanced_re = p->spec_re;
    p->enhanced_im = p->spec_im;
    return pp_begin_frame(p, p->spec_re, p->spec_im);
}

int dfn3_prepost_pre_process_freq_dual(DFN3Prepost *p, const float *est_re,
                                       const float *est_im, const float *app_re,
                                       const float *app_im) {
    size_t bins;
    if (!p || !est_re || !est_im || !app_re || !app_im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->grid.n_bins;
    if (!skn_all_finite_f32(est_re, bins) || !skn_all_finite_f32(est_im, bins) ||
        !skn_all_finite_f32(app_re, bins) || !skn_all_finite_f32(app_im, bins))
        return -1;
    /* The application spectrum is read again at commit, so it is copied. */
    memcpy(p->apply_re, app_re, bins * sizeof(float));
    memcpy(p->apply_im, app_im, bins * sizeof(float));
    p->enhanced_re = p->apply_re;
    p->enhanced_im = p->apply_im;
    return pp_begin_frame(p, est_re, est_im);
}

int dfn3_prepost_frame_open(DFN3Prepost *p, DFN3PrepostInputs *inputs,
                            DFN3PrepostOutputs *outputs) {
    if (!p || !inputs || !outputs || !p->frame_open) return -1;

    memset(inputs, 0, sizeof(*inputs));
    inputs->erb_window = p->io->erb_window;
    inputs->spec_window = p->io->spec_window;
    inputs->gru_hidden = p->io->gru_hidden;
    inputs->gru_hidden_elements = p->io->gru_hidden_elements;
    inputs->df_convp_window = p->io->df_convp_window;
    inputs->df_convp_window_elements = p->io->df_convp_window_elements;
    inputs->erb_window_elements = dfn3_prepost_erb_window_elements(&p->grid);
    inputs->spec_window_elements = dfn3_prepost_spec_window_elements(&p->grid);

    /* NaN heads: a partial accelerator write fails the commit. The recurrent state is left as is. */
    pp_fill_nan(p->head_erb_mask, dfn3_prepost_erb_mask_elements(&p->grid));
    pp_fill_nan(p->head_coefs, dfn3_prepost_coefs_elements(&p->grid));

    memset(outputs, 0, sizeof(*outputs));
    outputs->erb_mask = p->head_erb_mask;
    outputs->coefs = p->head_coefs;
    outputs->gru_hidden_next = p->io->gru_hidden;
    outputs->df_convp_window_next = p->io->df_convp_window;
    outputs->erb_mask_elements = dfn3_prepost_erb_mask_elements(&p->grid);
    outputs->coefs_elements = dfn3_prepost_coefs_elements(&p->grid);
    outputs->gru_hidden_elements = p->io->gru_hidden_elements;
    outputs->df_convp_window_elements = p->io->df_convp_window_elements;
    p->prepared = 1;
    return 0;
}

/* Build the enhanced spectrum with the given heads. Closes the frame on success only; the enhance clock
 * advances once per frame. A warm-up frame reads no head: the clock advances with heads_valid=0 (the stream
 * entry accepts this exactly while current < mask_lookahead, so a disagreement surfaces as -1) and the
 * recurrent state restarts from zero for the first real frame. */
static int pp_close_frame(DFN3Prepost *p, const float *erb_mask,
                          const float *coefs) {
    long long frame = 0;
    int emitted;

    if (PP_IN_WARMUP(p)) {
        if (dfn3_enhance_spectrum_stream(p->dsp, p->enhanced_re, p->enhanced_im, 0,
                                         NULL, NULL, p->atten_lim_db, p->enh_re,
                                         p->enh_im, NULL) != 0)
            return -1;
        dfn3_model_io_zero_recurrent(p->io);
        p->frame_open = 0;   /* have_output stays 0 from pp_begin_frame, output_frame untouched */
        p->prepared = 0;
        return 0;
    }

    /* The stream entry validates before its first write: -1 leaves the enhance state untouched and the
     * frame open for frame_bypass(). */
    emitted = dfn3_enhance_spectrum_stream(p->dsp, p->enhanced_re, p->enhanced_im, 1,
                                           erb_mask, coefs, p->atten_lim_db,
                                           p->enh_re, p->enh_im, &frame);
    if (emitted < 0) return -1;
    p->frame_open = 0;
    p->prepared = 0;
    p->have_output = emitted;
    if (emitted) p->output_frame = frame;
    return 0;
}

int dfn3_prepost_frame_finish(DFN3Prepost *p) {
    int heads_finite;
    int state_finite;
    if (!p || !p->prepared) return -1;   /* prepared implies frame_open */
    if (PP_IN_WARMUP(p)) return dfn3_prepost_frame_bypass(p);   /* heads never read */

    /* A non-finite head or state element zeroes the registered state, advances the generation and disarms the
     * transaction; the frame stays open. Windows, enhance state and clocks are untouched. */
    heads_finite =
        pp_all_finite(p->head_erb_mask, dfn3_prepost_erb_mask_elements(&p->grid)) &&
        pp_all_finite(p->head_coefs, dfn3_prepost_coefs_elements(&p->grid));
    state_finite = dfn3_model_io_validate_state(p->io) == 0;
    if (!state_finite || !heads_finite) {
        /* validate_state zeroed and counted a bad state; a bad head still restarts the state. */
        if (state_finite) dfn3_model_io_zero_recurrent(p->io);
        p->prepared = 0;
        return -1;
    }

    /* The clock advance has no undo, so it comes last. */
    return pp_close_frame(p, p->head_erb_mask, p->head_coefs);
}

int dfn3_prepost_frame_bypass(DFN3Prepost *p) {
    if (!p || !p->frame_open) return -1;
    return pp_close_frame(p, p->skip_mask, p->skip_coefs);
}

int dfn3_prepost_post_process_freq(DFN3Prepost *p, float *re, float *im, int *valid) {
    size_t bins;
    if (!p || !re || !im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->grid.n_bins * sizeof(float);
    memcpy(re, p->enh_re, bins);
    memcpy(im, p->enh_im, bins);
    if (valid) *valid = p->have_output;
    return 0;
}

/* ---- accelerator callback boundary ---- */

int dfn3_model_run_frame(const DFN3Model *model, DFN3Prepost *p) {
    DFN3PrepostInputs inputs;
    DFN3PrepostOutputs outputs;
    if (!model || !model->infer || !p || !p->frame_open ||
        dfn3_model_io_descriptor_validate(&p->grid, model->io_descriptor) != 0)
        return -1;
    if (dfn3_prepost_frame_open(p, &inputs, &outputs) != 0) return -1;
    if (model->infer(model->user, &inputs, &outputs) != 0) {
        return dfn3_prepost_frame_bypass(p) == 0 ? 0 : -1;
    }
    if (dfn3_prepost_frame_finish(p) != 0) {
        /* A refused commit leaves the frame open: take the identity so the clocks advance once. */
        return dfn3_prepost_frame_bypass(p) == 0 ? 0 : -1;
    }
    return 1;
}
