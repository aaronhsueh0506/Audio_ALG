#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "audio_pipeline_dfn.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"   /* sk_min_f32 / sk_capply_gain_f32 */

#define MONO_ALIGN 16u

struct MonoAecDfnRes {
    AudioPipeline *host;
    float *g_total;             /* min(unity, G_res)                     */
    float *unity;               /* all-ones gain read for the absent NR  */
    Complex *post;              /* P = E * g_total (+ comfort noise)     */
    uint32_t rng;               /* comfort-noise generator state         */
    int cng;                    /* comfort-noise fill enabled            */
    int n_freqs;
    int phase;                  /* 0 idle, 1 hop begun, 2 model frame open */
    const Complex *err_spec;    /* E of the hop in flight                  */
    DfnResStageJob job;         /* 48 kHz: the frame begun, until fetched  */
    int job_pending;            /* 48 kHz: job holds an unfetched frame    */
    int stage_ended;            /* 48 kHz: the stage frame is closed       */
    DfnResStage *stage;         /* 48 kHz: the stage, driven directly */
    Complex *stage_output;      /* 48 kHz: its output spectrum          */
    DfnRateBridge *bridge;      /* 8/16 kHz: the stage inside the bridge */
    void *owned_heap;
};

static size_t align_up(size_t v) {
    if (v > SIZE_MAX - (MONO_ALIGN - 1u)) return 0u;
    return (v + MONO_ALIGN - 1u) & ~(size_t)(MONO_ALIGN - 1u);
}

static uint32_t fnv_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (8 * i)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* Everything init needs, derived from the config once: the stage config, both
 * sub-descriptors and this pool's total. The host validates its own config. */
typedef struct Resolved {
    int use_bridge;                 /* host below 48 kHz */
    DfnResStageConfig scfg;
    DfnRateBridgeConfig bcfg;
    AudioPipelineMemReq host_req;
    DfnResStageMemReq stage_req;    /* 48 kHz */
    DfnRateBridgeMemReq bridge_req; /* 8/16 kHz */
    uint32_t sub_layout;            /* the stage's or the bridge's */
    uint32_t sub_hash;
    uint32_t backend_id;
    int n_freqs;
    size_t bytes;
} Resolved;

static int resolve(const MonoAecDfnResConfig *cfg, Resolved *r) {
    AudioPipelineMemBreakdown bd;
    size_t value;
    if (!cfg || !r ||
        (cfg->host.sample_rate != 8000 && cfg->host.sample_rate != 16000 &&
         cfg->host.sample_rate != DFN2_SR) ||
        cfg->host.aec_only != 0 || cfg->host.enable_nr != 0 ||
        cfg->host.enable_res != 1 || !isfinite(cfg->atten_lim_db))
        return -1;
    r->use_bridge = cfg->host.sample_rate != DFN2_SR;
    r->scfg = dfn_res_stage_default_config(DFN2_SR);
    r->scfg.atten_lim_db = cfg->atten_lim_db;
    r->scfg.erb_fwd = cfg->erb_fwd;
    r->scfg.erb_inv = cfg->erb_inv;
    r->scfg.model = cfg->model;
    if (audio_pipeline_get_mem_requirements(&cfg->host, &r->host_req) != 0)
        return -1;
    if (audio_pipeline_get_mem_breakdown(&cfg->host, &bd) != 0) return -1;
    r->n_freqs = bd.n_freqs;
    if (r->use_bridge) {
        r->bcfg = dfn_rate_bridge_default_config(cfg->host.sample_rate);
        r->bcfg.fft_size = cfg->host.fft_size;
        r->bcfg.stage = r->scfg;
        if (dfn_rate_bridge_get_mem_requirements(&r->bcfg, &r->bridge_req) != 0)
            return -1;
        r->sub_layout = r->bridge_req.layout_version;
        r->sub_hash = r->bridge_req.build_flags_hash;
        r->backend_id = r->bridge_req.backend_id;
    } else {
        if (cfg->host.fft_size != 0 && cfg->host.fft_size != DFN2_N_FFT)
            return -1;
        if (dfn_res_stage_get_mem_requirements(&r->scfg, &r->stage_req) != 0)
            return -1;
        r->sub_layout = r->stage_req.layout_version;
        r->sub_hash = r->stage_req.build_flags_hash;
        r->backend_id = r->stage_req.backend_id;
    }
    if (r->host_req.backend_id != r->backend_id) return -1;
    value = align_up(sizeof(MonoAecDfnRes));
#define ADD_REGION(bytes_) do { \
        size_t add_ = align_up((size_t)(bytes_)); \
        if (!add_ || value > SIZE_MAX - add_) return -1; \
        value += add_; \
    } while (0)
    ADD_REGION(r->host_req.bytes);
    ADD_REGION((size_t)r->n_freqs * sizeof(float));     /* g_total */
    ADD_REGION((size_t)r->n_freqs * sizeof(float));     /* unity   */
    ADD_REGION((size_t)r->n_freqs * sizeof(Complex));   /* post    */
    if (r->use_bridge) {
        ADD_REGION(r->bridge_req.bytes);
    } else {
        ADD_REGION(r->stage_req.bytes);
        ADD_REGION((size_t)DFN2_N_BINS * sizeof(Complex));
    }
#undef ADD_REGION
    r->bytes = value;
    return 0;
}

static void fill_descriptor(const Resolved *r, MonoAecDfnResMemReq *out) {
    uint32_t h = 2166136261u;
    h = fnv_u32(h, MONO_AEC_DFN_RES_LAYOUT_VERSION);
    h = fnv_u32(h, r->host_req.layout_version);
    h = fnv_u32(h, r->host_req.build_flags_hash);
    h = fnv_u32(h, (uint32_t)r->use_bridge);
    h = fnv_u32(h, r->sub_layout);
    h = fnv_u32(h, r->sub_hash);
    memset(out, 0, sizeof(*out));
    out->descriptor_version = MONO_AEC_DFN_RES_DESCRIPTOR_VERSION;
    out->layout_version = MONO_AEC_DFN_RES_LAYOUT_VERSION;
    out->backend_id = r->backend_id;
    out->build_flags_hash = ac_memreq_abi_hash(h);
    out->alignment = MONO_ALIGN;
    out->bytes = (uint64_t)r->bytes;
}

MonoAecDfnResConfig mono_aec_dfn_res_default_config(void) {
    MonoAecDfnResConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.host = audio_pipeline_default_config(16000);
    cfg.host.enable_nr = 0;
    cfg.host.enable_res = 1;
    cfg.host.enable_cng = 0;
    return cfg;
}

int mono_aec_dfn_res_get_mem_requirements(const MonoAecDfnResConfig *cfg,
                                          MonoAecDfnResMemReq *out) {
    Resolved r;
    if (!out || resolve(cfg, &r) != 0) return -1;
    fill_descriptor(&r, out);
    return 0;
}

static int descriptor_matches(const MonoAecDfnResMemReq *a,
                              const MonoAecDfnResMemReq *b) {
    return a && b && a->descriptor_version == b->descriptor_version &&
        a->layout_version == b->layout_version &&
        a->backend_id == b->backend_id &&
        a->build_flags_hash == b->build_flags_hash &&
        a->alignment == b->alignment && a->reserved == 0u &&
        a->bytes >= b->bytes;
}

MonoAecDfnRes *mono_aec_dfn_res_init_ex(void *mem, size_t bytes,
                                        const MonoAecDfnResConfig *cfg,
                                        const MonoAecDfnResMemReq *expected) {
    Resolved r;
    MonoAecDfnResMemReq req;
    MonoAecDfnRes *p;
    unsigned char *cursor;
    if (!mem || (uintptr_t)mem % MONO_ALIGN != 0u || resolve(cfg, &r) != 0)
        return NULL;
    fill_descriptor(&r, &req);
    if (bytes < r.bytes || (expected && !descriptor_matches(expected, &req)))
        return NULL;
    memset(mem, 0, r.bytes);
    p = (MonoAecDfnRes *)mem;
    cursor = (unsigned char *)mem + align_up(sizeof(*p));
    p->host = audio_pipeline_init_ex(cursor, (size_t)r.host_req.bytes,
                                     &cfg->host, &r.host_req);
    if (!p->host) return NULL;
    cursor += align_up((size_t)r.host_req.bytes);
    p->n_freqs = r.n_freqs;
    p->g_total = (float *)cursor;
    cursor += align_up((size_t)r.n_freqs * sizeof(float));
    p->unity = (float *)cursor;
    cursor += align_up((size_t)r.n_freqs * sizeof(float));
    p->post = (Complex *)cursor;
    cursor += align_up((size_t)r.n_freqs * sizeof(Complex));
    {
        int k;
        for (k = 0; k < r.n_freqs; ++k) p->unity[k] = 1.0f;
    }
    p->rng = SE_CNG_RNG_SEED;
    /* The comfort-noise fill is the host config's switch. */
    p->cng = cfg->host.enable_cng;
    /* The host resolves fft_size 0 to its rate default; the stage's grid (48 kHz) or
     * the bridge's native grid must be the host's. */
    if (r.use_bridge) {
        p->bridge = dfn_rate_bridge_init_ex(cursor, (size_t)r.bridge_req.bytes,
                                            &r.bcfg, &r.bridge_req);
        if (!p->bridge ||
            dfn_rate_bridge_n_freqs(p->bridge) != audio_pipeline_n_freqs(p->host) ||
            dfn_rate_bridge_hop_size(p->bridge) != audio_pipeline_hop_size(p->host))
            return NULL;
        p->stage = dfn_rate_bridge_stage(p->bridge);
        return p;
    }
    if (audio_pipeline_n_freqs(p->host) != DFN2_N_BINS ||
        audio_pipeline_hop_size(p->host) != DFN2_HOP_LEN)
        return NULL;
    p->stage = dfn_res_stage_init_ex(cursor, (size_t)r.stage_req.bytes,
                                     &r.scfg, &r.stage_req);
    if (!p->stage) return NULL;
    cursor += align_up((size_t)r.stage_req.bytes);
    p->stage_output = (Complex *)cursor;
    return p;
}

MonoAecDfnRes *mono_aec_dfn_res_init(void *mem, size_t bytes,
                                     const MonoAecDfnResConfig *cfg) {
    return mono_aec_dfn_res_init_ex(mem, bytes, cfg, NULL);
}

MonoAecDfnRes *mono_aec_dfn_res_create(const MonoAecDfnResConfig *cfg) {
    MonoAecDfnResMemReq req;
    MonoAecDfnRes *p;
    void *mem = NULL;
    if (mono_aec_dfn_res_get_mem_requirements(cfg, &req) != 0) return NULL;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return NULL;
    p = mono_aec_dfn_res_init_ex(mem, (size_t)req.bytes, cfg, &req);
    if (!p) { free(mem); return NULL; }
    p->owned_heap = mem;
    return p;
}

void mono_aec_dfn_res_reset(MonoAecDfnRes *p) {
    if (!p) return;
    p->phase = 0;
    p->job_pending = 0;
    audio_pipeline_reset(p->host);
    p->rng = SE_CNG_RNG_SEED;
    if (p->bridge) {
        dfn_rate_bridge_reset(p->bridge);   /* resets its stage too */
        return;
    }
    dfn_res_stage_reset(p->stage);
    memset(p->stage_output, 0, (size_t)DFN2_N_BINS * sizeof(Complex));
}

void mono_aec_dfn_res_destroy(MonoAecDfnRes *p) {
    void *owned;
    if (!p) return;
    owned = p->owned_heap;
    p->owned_heap = NULL;
    if (p->bridge) dfn_rate_bridge_destroy(p->bridge);
    else dfn_res_stage_destroy(p->stage);
    audio_pipeline_destroy(p->host);
    if (owned) free(owned);
}

/* A contract error anywhere in a hop resets the pipeline. */
static int fail(MonoAecDfnRes *p) {
    mono_aec_dfn_res_reset(p);
    return -1;
}

int mono_aec_dfn_res_process_begin(MonoAecDfnRes *p, const float *mic,
                                   const float *ref) {
    SeSpecFrame f;
    int r;
    if (!p || !mic || !ref || p->phase != 0) return -1;
    if (audio_pipeline_process_spec(p->host, mic, ref, &f) != 0) return fail(p);
    /* P = E * min(1, G_res), plus comfort noise on the cut bins; the absent denoiser
     * reads as the unity vector. */
    sk_min_f32(p->g_total, p->unity, f.g_res_ctx, p->n_freqs);
    sk_capply_gain_f32(p->post, f.e, p->g_total, p->n_freqs);
    if (p->cng && f.cn)
        aec_cng_fill(&p->rng, p->post, f.cn, f.g_res_ctx, p->unity,
                     SE_CNG_NR_GAIN_FLOOR, p->n_freqs);
    p->err_spec = f.e;
    p->job_pending = 0;
    p->stage_ended = 0;
    if (p->bridge) {
        if (dfn_rate_bridge_begin(p->bridge, f.e, p->post) != 0) return fail(p);
    } else {
        r = dfn_res_stage_begin(p->stage, f.e, p->post, &p->job);
        if (r < 0) return fail(p);
        p->job_pending = r == 1;
    }
    p->phase = 1;
    return 0;
}

int mono_aec_dfn_res_nn_begin(MonoAecDfnRes *p, DfnResStageJob *job) {
    int r;
    if (!p || !job || p->phase != 1) return -1;
    if (p->bridge) {
        r = dfn_rate_bridge_nn_begin(p->bridge, job);
        if (r < 0) return fail(p);
    } else if (p->job_pending) {
        *job = p->job;
        p->job_pending = 0;
        r = 1;
    } else {
        r = 0;
    }
    if (r == 1) p->phase = 2;
    return r;
}

int mono_aec_dfn_res_nn_end(MonoAecDfnRes *p, int model_rc) {
    if (!p || p->phase != 2) return -1;
    p->phase = 1;
    if (p->bridge) {
        if (dfn_rate_bridge_nn_end(p->bridge, model_rc) != 0) return fail(p);
    } else {
        if (dfn_res_stage_end(p->stage, model_rc, p->stage_output) < 0)
            return fail(p);
        p->stage_ended = 1;
    }
    return 0;
}

int mono_aec_dfn_res_process_end(MonoAecDfnRes *p, float *out) {
    if (!p || !out || p->phase == 0) return -1;
    if (p->bridge) {
        p->phase = 0;
        if (dfn_rate_bridge_finish(p->bridge, out) != 0) return fail(p);
        return 0;
    }
    /* A frame left open or unfetched takes the identity. */
    if (!p->stage_ended) {
        int open = p->job_pending || p->phase == 2;
        if (dfn_res_stage_end(p->stage, open ? -1 : 0, p->stage_output) < 0) {
            p->phase = 0;
            return fail(p);
        }
    }
    p->phase = 0;
    p->job_pending = 0;
    if (audio_pipeline_synthesize(p->host, p->stage_output, out) != 0)
        return fail(p);
    return 0;
}

int mono_aec_dfn_res_process(MonoAecDfnRes *p, const float *mic,
                             const float *ref, float *out) {
    DfnResStageJob job;
    int r;
    if (!p || !mic || !ref || !out) return -1;
    if (mono_aec_dfn_res_process_begin(p, mic, ref) != 0) return -1;
    while ((r = mono_aec_dfn_res_nn_begin(p, &job)) == 1)
        if (mono_aec_dfn_res_nn_end(p, dfn_res_stage_infer(p->stage, &job)) != 0)
            return -1;
    if (r < 0) return -1;
    return mono_aec_dfn_res_process_end(p, out);
}

int mono_aec_dfn_res_set_aec_preset(MonoAecDfnRes *p, AecPreset preset,
                                    float ramp_ms) {
    return p ? audio_pipeline_set_aec_preset(p->host, preset, ramp_ms) : -1;
}

int mono_aec_dfn_res_set_atten_lim(MonoAecDfnRes *p, float atten_lim_db) {
    return p ? dfn_res_stage_set_atten_lim(p->stage, atten_lim_db) : -1;
}

int mono_aec_dfn_res_hop_size(const MonoAecDfnRes *p) {
    return p ? audio_pipeline_hop_size(p->host) : -1;
}

int mono_aec_dfn_res_lookahead_samples(const MonoAecDfnRes *p) {
    if (!p) return -1;
    /* Host WOLA hop plus the model's two lookahead hops at 48 kHz, or the bridge's
     * added delay on a native grid. */
    if (p->bridge)
        return audio_pipeline_hop_size(p->host) +
               dfn_rate_bridge_added_delay_samples(p->bridge);
    return (DFN_RES_STAGE_LOOKAHEAD_FRAMES + 1) * DFN2_HOP_LEN;
}

AudioPipeline *mono_aec_dfn_res_get_host(const MonoAecDfnRes *p) {
    return p ? p->host : NULL;
}

Aec *mono_aec_dfn_res_get_aec(const MonoAecDfnRes *p) {
    return p ? audio_pipeline_get_aec(p->host) : NULL;
}

DfnResStage *mono_aec_dfn_res_get_stage(const MonoAecDfnRes *p) {
    return p ? p->stage : NULL;
}

DfnRateBridge *mono_aec_dfn_res_get_bridge(const MonoAecDfnRes *p) {
    return p ? p->bridge : NULL;
}
