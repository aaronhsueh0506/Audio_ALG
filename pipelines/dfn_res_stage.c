/* Post-RES DeepFilterNet2 stage; see dfn_res_stage.h for the signal/timing
 * contract. This TU owns no FFT: both inputs are spectra on DFN2's native
 * 48 kHz / 1024 / 512 grid and the hosting pipeline owns synthesis. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dfn_res_stage.h"
#include "simd_kernel_nn.h"

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
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "kiss") == 0)
        return DFN_RES_STAGE_BACKEND_KISS;
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0)
        return DFN_RES_STAGE_BACKEND_NE10;
    return 0u;
}

static int model_valid(const DFN2Model *model) {
    if (!model) return 0;
    if (!model->infer)
        return model->reset == NULL && model->io_descriptor == NULL;
    return model->io_descriptor != NULL &&
           dfn2_model_io_descriptor_validate(model->io_descriptor) == 0;
}

/* The identity contract of the fail-open path (dfn2_prepost_frame_skip: unit
 * band mask, zero taps, alpha 0) is exact only when the unit mask expands to
 * exactly 1.0f in every bin. Walk the band-major matrix in the same order as
 * the DFN2 expansion (df_common_expand_mask: bands outer, bins inner, one
 * multiply-add per band), so the check sees the same float sums the model
 * path will form. A unit gain makes each product exact, so this is the
 * expansion itself, not an approximation of it. */
static int erb_inv_is_partition_of_unity(const float *erb_inv) {
    float bin_gain[DFN2_N_BINS];
    int b, k;
    memset(bin_gain, 0, sizeof(bin_gain));
    for (b = 0; b < DFN2_N_ERB; ++b) {
        const float *row = erb_inv + (size_t)b * DFN2_N_BINS;
        for (k = 0; k < DFN2_N_BINS; ++k)
            bin_gain[k] += row[k] * 1.0f;
    }
    for (k = 0; k < DFN2_N_BINS; ++k)
        if (bin_gain[k] != 1.0f) return 0;
    return 1;
}

/* Everything init needs, derived from the config exactly once: the DFN2
 * class config and pool size, and this pool's total. Reject-first, in the
 * order the header documents. */
typedef struct Resolved {
    DFN2PrepostConfig pcfg;
    DFN2PrepostMemReq pp_req;
    size_t bytes;
} Resolved;

static int resolve(const DfnResStageConfig *cfg, Resolved *r) {
    size_t one = (size_t)DFN2_N_BINS * sizeof(float);
    size_t total;
    if (!cfg || !r) return -1;
    if (cfg->sample_rate != DFN2_SR) return -1;
    if (cfg->fft_size != 0 && cfg->fft_size != DFN2_N_FFT) return -1;
    if (!isfinite(cfg->atten_lim_db) || !cfg->erb_fwd || !cfg->erb_inv)
        return -1;
    if (!erb_inv_is_partition_of_unity(cfg->erb_inv)) return -1;
    if (!model_valid(&cfg->model) || backend_id() == 0u) return -1;
    if (dfn2_prepost_config_defaults(&r->pcfg, DFN2_IO_FREQ) != 0) return -1;
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
    out->build_flags_hash = h;
    out->alignment = STAGE_ALIGN;
    out->bytes = (uint64_t)r->bytes;
}

DfnResStageConfig dfn_res_stage_default_config(int sample_rate) {
    DfnResStageConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = DFN2_N_FFT;
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
    size_t one = (size_t)DFN2_N_BINS * sizeof(float);
    if (!mem || resolve(cfg, &r) != 0) return NULL;
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

    /* One aligned slab: an individual 513-float row is not a multiple of
     * 16 bytes, so aligning every row would add hidden padding and make the
     * sizing walk disagree with the carve. SIMD kernels accept unaligned
     * rows; only the slab base is part of the pool ABI. */
    if (carve(&cur, one * 6u, &region) != 0) return NULL;
    s->est_re = (float *)region;
    s->est_im = s->est_re + DFN2_N_BINS;
    s->app_re = s->est_im + DFN2_N_BINS;
    s->app_im = s->app_re + DFN2_N_BINS;
    s->out_re = s->app_im + DFN2_N_BINS;
    s->out_im = s->out_re + DFN2_N_BINS;
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
    if (s->model.reset) s->model.reset(s->model.user);
}

int dfn_res_stage_set_atten_lim(DfnResStage *s, float atten_lim_db) {
    if (!s || !isfinite(atten_lim_db)) return -1;
    return dfn2_prepost_set_atten_lim(s->prepost, atten_lim_db);
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

/* Both spectra are scanned for finite values before any of them reaches
 * the recurrent state: a NaN that enters the feature EMA or the GRU hidden
 * state would poison every later frame, and the class's own commit-time
 * check only covers the model outputs. */
static int inputs_valid(const Complex *estimate, const Complex *apply) {
    if (!estimate || !apply) return 0;
    return skn_all_finite_cf32(estimate, DFN2_N_BINS) &&
           skn_all_finite_cf32(apply, DFN2_N_BINS);
}

int dfn_res_stage_process(DfnResStage *s,
                          const Complex *estimate_spec,
                          const Complex *apply_spec,
                          Complex *out_spec) {
    int need_heads;
    int valid = 0;
    int run_result = 0;
    if (!s || !out_spec || !inputs_valid(estimate_spec, apply_spec))
        return -1;

    skn_deinterleave_scale_cf32(estimate_spec, s->est_re, s->est_im,
                                DFN2_N_BINS, DFN_RES_STAGE_IN_SCALE);
    skn_deinterleave_scale_cf32(apply_spec, s->app_re, s->app_im,
                                DFN2_N_BINS, DFN_RES_STAGE_IN_SCALE);

    need_heads = dfn2_prepost_pre_process_freq_dual(
        s->prepost, s->est_re, s->est_im, s->app_re, s->app_im);
    if (need_heads < 0) return -1;
    if (need_heads) {
        if (s->model.infer) {
            run_result = dfn2_model_run_frame(&s->model, s->prepost);
            if (run_result < 0) return -1;
        } else {
            if (dfn2_prepost_frame_skip(s->prepost) != 0) return -1;
            run_result = 0;
        }
        if (run_result) ++s->commits;
        else ++s->skips;
    }

    if (dfn2_prepost_post_process_freq(s->prepost, s->out_re, s->out_im,
                                       &valid) != 0)
        return -1;
    ++s->frames_in;
    if (!valid) {
        memset(out_spec, 0, (size_t)DFN2_N_BINS * sizeof(Complex));
        return 0;
    }
    skn_interleave_scale_cf32(s->out_re, s->out_im, out_spec, DFN2_N_BINS,
                              DFN_RES_STAGE_OUT_SCALE);
    return 1;
}
