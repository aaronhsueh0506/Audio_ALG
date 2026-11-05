/* Post-RES DeepFilterNet2 stage; see dfn_res_stage.h for the signal/timing
 * contract. Both inputs are spectra on the DFN2 grid of the config; the
 * hosting pipeline owns synthesis. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dfn_res_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

#define STAGE_ALIGN 16u

struct DfnResStage {
    DFN2Model model;          /* the only config member read after init */
    DFN2Prepost *prepost;

    float *est_re;
    float *est_im;
    float *app_re;
    float *app_im;
    float *out_re;
    float *out_im;

    long long frames_in;
    long long commits;
    long long skips;
    int phase;                /* 0 idle, 1 hop begun (no model frame), 2 model frame open */
    void *owned_heap;
};

typedef struct StageCursor {
    unsigned char *ptr;
    size_t remaining;
} StageCursor;

static size_t align_up(size_t value, size_t alignment) {
    size_t mask = alignment - 1u;
    if (value > SIZE_MAX - mask) return 0u;
    return (value + mask) & ~mask;
}

static int carve(StageCursor *c, size_t bytes, void **out) {
    uintptr_t p;
    uintptr_t aligned;
    size_t pad;
    if (!c || !out) return -1;
    p = (uintptr_t)c->ptr;
    aligned = (p + STAGE_ALIGN - 1u) & ~(uintptr_t)(STAGE_ALIGN - 1u);
    pad = (size_t)(aligned - p);
    if (pad > c->remaining || bytes > c->remaining - pad) return -1;
    *out = (void *)aligned;
    c->ptr = (unsigned char *)(aligned + bytes);
    c->remaining -= pad + bytes;
    return 0;
}

static uint32_t fnv1a_u32(uint32_t h, uint32_t value) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (value >> (8 * i)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

static uint32_t backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0)
        return DFN_RES_STAGE_BACKEND_NE10;
    return 0u;
}

static int model_valid(const DFN2Model *model, const DFN2Grid *grid) {
    if (!model) return 0;
    if (!model->infer)
        return model->reset == NULL && model->io_descriptor == NULL;
    return model->io_descriptor != NULL &&
           dfn2_model_io_descriptor_validate(grid, model->io_descriptor) == 0;
}

/* The fail-open identity is exact only when the unit band mask expands to exactly
 * 1.0f in every bin; walk the band-major matrix in the DFN2 expansion order
 * (bands outer, bins inner) so the sums match the model path. The grid is valid, so its
 * n_bins is within the library's capacity. */
static int erb_inv_is_partition_of_unity(const float *erb_inv, const DFN2Grid *grid) {
    float bin_gain[DFN2_MAX_N_BINS];
    int b, k;
    memset(bin_gain, 0, sizeof(bin_gain));
    for (b = 0; b < grid->n_erb; ++b) {
        const float *row = erb_inv + (size_t)b * (size_t)grid->n_bins;
        for (k = 0; k < grid->n_bins; ++k)
            bin_gain[k] += row[k] * 1.0f;
    }
    for (k = 0; k < grid->n_bins; ++k)
        if (bin_gain[k] != 1.0f) return 0;
    return 1;
}

/* Everything init needs, derived from the config once: the DFN2 class config and
 * pool size, and this pool's total. */
typedef struct Resolved {
    DFN2PrepostConfig pcfg;
    DFN2PrepostMemReq pp_req;
    size_t bytes;
} Resolved;

static int resolve(const DfnResStageConfig *cfg, Resolved *r) {
    size_t one;
    size_t total;
    /* This object's DFN2 capacities must be the library objects': link-time check. */
    DFN2_REQUIRE_VARIANT_LINK();
    if (!cfg || !r) return -1;
    if (dfn2_grid_validate(&cfg->grid) != DFN2_GRID_OK) return -1;
    one = (size_t)cfg->grid.n_bins * sizeof(float);
    if (!isfinite(cfg->atten_lim_db) || !cfg->erb_fwd || !cfg->erb_inv)
        return -1;
    if (!erb_inv_is_partition_of_unity(cfg->erb_inv, &cfg->grid)) return -1;
    if (!model_valid(&cfg->model, &cfg->grid) || backend_id() == 0u) return -1;
    if (dfn2_prepost_config_defaults(&r->pcfg, DFN2_IO_FREQ, &cfg->grid) != 0) return -1;
    r->pcfg.erb_fwd = cfg->erb_fwd;
    r->pcfg.erb_inv = cfg->erb_inv;
    r->pcfg.atten_lim_db = cfg->atten_lim_db;
    if (dfn2_prepost_get_mem_size(&r->pcfg, &r->pp_req) != 0) return -1;
    total = align_up(sizeof(DfnResStage), STAGE_ALIGN);
    if (!total || (size_t)r->pp_req.bytes > SIZE_MAX - total) return -1;
    total += (size_t)r->pp_req.bytes;
    total = align_up(total, STAGE_ALIGN);
    if (!total || 6u > (SIZE_MAX - total) / one) return -1;   /* est/app/out re+im */
    total += 6u * one;
    total = align_up(total, STAGE_ALIGN);
    if (!total) return -1;
    r->bytes = total;
    return 0;
}

static void fill_descriptor(const Resolved *r, DfnResStageMemReq *out) {
    uint32_t h = 2166136261u;
    h = fnv1a_u32(h, DFN_RES_STAGE_LAYOUT_VERSION);
    h = fnv1a_u32(h, r->pp_req.layout_version);
    h = fnv1a_u32(h, r->pp_req.build_flags_hash);
    h = fnv1a_u32(h, (uint32_t)DFN2_PREPOST_CARVE_VERSION);
    memset(out, 0, sizeof(*out));
    out->descriptor_version = DFN_RES_STAGE_DESCRIPTOR_VERSION;
    out->layout_version = DFN_RES_STAGE_LAYOUT_VERSION;
    out->backend_id = backend_id();
    out->build_flags_hash = ac_memreq_abi_hash(h);
    out->alignment = STAGE_ALIGN;
    out->bytes = (uint64_t)r->bytes;
}

DfnResStageConfig dfn_res_stage_default_config(const DFN2Grid *grid) {
    DfnResStageConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (grid) cfg.grid = *grid;
    return cfg;
}

int dfn_res_stage_get_mem_requirements(const DfnResStageConfig *cfg,
                                       DfnResStageMemReq *out) {
    Resolved r;
    if (!out || resolve(cfg, &r) != 0) return -1;
    fill_descriptor(&r, out);
    return 0;
}

static int descriptor_matches(const DfnResStageMemReq *a,
                              const DfnResStageMemReq *b) {
    return a && b &&
        a->descriptor_version == b->descriptor_version &&
        a->layout_version == b->layout_version &&
        a->backend_id == b->backend_id &&
        a->build_flags_hash == b->build_flags_hash &&
        a->alignment == b->alignment && a->reserved == 0u &&
        a->bytes >= b->bytes;
}

DfnResStage *dfn_res_stage_init_ex(void *mem, size_t bytes,
                                   const DfnResStageConfig *cfg,
                                   const DfnResStageMemReq *expected) {
    Resolved r;
    DfnResStageMemReq req;
    StageCursor cur;
    DfnResStage *s;
    void *region;
    size_t one;
    int n_bins;
    if (!mem || resolve(cfg, &r) != 0) return NULL;
    n_bins = cfg->grid.n_bins;
    one = (size_t)n_bins * sizeof(float);
    fill_descriptor(&r, &req);
    if (bytes < r.bytes || (uintptr_t)mem % (uintptr_t)STAGE_ALIGN != 0u)
        return NULL;
    if (expected && !descriptor_matches(expected, &req)) return NULL;

    memset(mem, 0, r.bytes);
    cur.ptr = (unsigned char *)mem;
    cur.remaining = r.bytes;
    if (carve(&cur, sizeof(*s), &region) != 0) return NULL;
    s = (DfnResStage *)region;
    s->model = cfg->model;

    if (carve(&cur, (size_t)r.pp_req.bytes, &region) != 0) return NULL;
    s->prepost = dfn2_prepost_init_ex(region, (size_t)r.pp_req.bytes,
                                      &r.pcfg, &r.pp_req);
    if (!s->prepost) return NULL;

    /* One aligned slab for the six spectra; only the slab base is part of the pool ABI. */
    if (carve(&cur, one * 6u, &region) != 0) return NULL;
    s->est_re = (float *)region;
    s->est_im = s->est_re + n_bins;
    s->app_re = s->est_im + n_bins;
    s->app_im = s->app_re + n_bins;
    s->out_re = s->app_im + n_bins;
    s->out_im = s->out_re + n_bins;
    return s;
}

DfnResStage *dfn_res_stage_init(void *mem, size_t bytes,
                                const DfnResStageConfig *cfg) {
    return dfn_res_stage_init_ex(mem, bytes, cfg, NULL);
}

DfnResStage *dfn_res_stage_create(const DfnResStageConfig *cfg) {
    DfnResStageMemReq req;
    DfnResStage *s;
    void *mem = NULL;
    if (dfn_res_stage_get_mem_requirements(cfg, &req) != 0) return NULL;
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        return NULL;
    s = dfn_res_stage_init_ex(mem, (size_t)req.bytes, cfg, &req);
    if (!s) { free(mem); return NULL; }
    s->owned_heap = mem;
    return s;
}

void dfn_res_stage_destroy(DfnResStage *s) {
    void *owned;
    if (!s) return;
    owned = s->owned_heap;
    s->owned_heap = NULL;
    dfn2_prepost_destroy(s->prepost);
    if (owned) free(owned);
}

void dfn_res_stage_reset(DfnResStage *s) {
    if (!s) return;
    dfn2_prepost_reset(s->prepost);
    s->frames_in = 0;
    s->commits = 0;
    s->skips = 0;
    s->phase = 0;
    if (s->model.reset) s->model.reset(s->model.user);
}

int dfn_res_stage_set_atten_lim(DfnResStage *s, float atten_lim_db) {
    if (!s || !isfinite(atten_lim_db)) return -1;
    return dfn2_prepost_set_atten_lim(s->prepost, atten_lim_db);
}

const DFN2Grid *dfn_res_stage_grid(const DfnResStage *s) {
    return s ? dfn2_prepost_grid(s->prepost) : NULL;
}

int dfn_res_stage_output_frame_index(const DfnResStage *s, long long *frame) {
    return s ? dfn2_prepost_output_frame_index(s->prepost, frame) : -1;
}

void dfn_res_stage_get_counters(const DfnResStage *s, long long *frames_in,
                                long long *commits, long long *skips) {
    if (frames_in) *frames_in = s ? s->frames_in : 0;
    if (commits) *commits = s ? s->commits : 0;
    if (skips) *skips = s ? s->skips : 0;
}

/* Scans both spectra for finite values before they reach the recurrent state. */
static int inputs_valid(const Complex *estimate, const Complex *apply, int n_bins) {
    if (!estimate || !apply) return 0;
    return skn_all_finite_cf32(estimate, n_bins) &&
           skn_all_finite_cf32(apply, n_bins);
}

int dfn_res_stage_begin(DfnResStage *s,
                        const Complex *estimate_spec,
                        const Complex *apply_spec,
                        DfnResStageJob *job) {
    int need_heads;
    const DFN2Grid *grid;
    if (!s || !job || s->phase != 0) return -1;
    grid = dfn2_prepost_grid(s->prepost);
    if (!grid || !inputs_valid(estimate_spec, apply_spec, grid->n_bins)) return -1;

    skn_deinterleave_scale_cf32(estimate_spec, s->est_re, s->est_im,
                                grid->n_bins, grid->stft_scale);
    skn_deinterleave_scale_cf32(apply_spec, s->app_re, s->app_im,
                                grid->n_bins, grid->stft_scale);

    need_heads = dfn2_prepost_pre_process_freq_dual(
        s->prepost, s->est_re, s->est_im, s->app_re, s->app_im);
    if (need_heads < 0) return -1;
    if (need_heads &&
        dfn2_prepost_frame_inputs(s->prepost, &job->inputs, &job->outputs) != 0)
        return -1;
    s->phase = need_heads ? 2 : 1;
    return need_heads ? 1 : 0;
}

int dfn_res_stage_end(DfnResStage *s, int model_rc, Complex *out_spec) {
    int valid = 0;
    const DFN2Grid *grid;
    if (!s || !out_spec || s->phase == 0) return -1;
    grid = dfn2_prepost_grid(s->prepost);
    if (s->phase == 2) {
        /* A failed run or an unwritten/non-finite output takes the frame as the identity;
         * the clocks advance once. */
        int committed = model_rc == 0 &&
                        dfn2_prepost_frame_commit(s->prepost) == 0;
        if (!committed && dfn2_prepost_frame_skip(s->prepost) != 0) {
            s->phase = 0;
            return -1;
        }
        if (committed) ++s->commits;
        else ++s->skips;
    }
    s->phase = 0;

    if (dfn2_prepost_post_process_freq(s->prepost, s->out_re, s->out_im,
                                       &valid) != 0)
        return -1;
    ++s->frames_in;
    if (!valid) {
        memset(out_spec, 0, (size_t)grid->n_bins * sizeof(Complex));
        return 0;
    }
    skn_interleave_scale_cf32(s->out_re, s->out_im, out_spec, grid->n_bins,
                              grid->stft_unscale);
    return 1;
}

int dfn_res_stage_infer(const DfnResStage *s, DfnResStageJob *job) {
    if (!s || !job || !s->model.infer) return -1;
    return s->model.infer(s->model.user, &job->inputs, &job->outputs);
}

int dfn_res_stage_process(DfnResStage *s,
                          const Complex *estimate_spec,
                          const Complex *apply_spec,
                          Complex *out_spec) {
    DfnResStageJob job;
    int rc;
    if (!out_spec) return -1;
    rc = dfn_res_stage_begin(s, estimate_spec, apply_spec, &job);
    if (rc < 0) return -1;
    return dfn_res_stage_end(s, rc == 1 ? dfn_res_stage_infer(s, &job) : 0,
                             out_spec);
}
