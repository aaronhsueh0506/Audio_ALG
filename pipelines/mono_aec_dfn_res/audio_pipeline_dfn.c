#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "audio_pipeline_dfn.h"

#define MONO_ALIGN 16u

struct MonoAecDfnRes {
    AudioPipeline *host;
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

/* Everything init needs, derived from the config exactly once: the stage
 * config, both sub-descriptors and this pool's total. The host validates
 * its own config; only the seam invariants are checked here. */
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
    size_t bytes;
} Resolved;

static int resolve(const MonoAecDfnResConfig *cfg, Resolved *r) {
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
    out->build_flags_hash = h;
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
    /* The host resolves fft_size 0 to its rate default; the stage's grid
     * (48 kHz) or the bridge's native grid must be the host's, checked once
     * here. */
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
    audio_pipeline_reset(p->host);
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

int mono_aec_dfn_res_process(MonoAecDfnRes *p, const float *mic,
                             const float *ref, float *out) {
    AudioPipelinePostView view;
    int ok;
    if (!p || !mic || !ref || !out) return -1;
    ok = audio_pipeline_process_post_view(p->host, mic, ref, &view) == 0;
    if (ok && p->bridge) {
        ok = dfn_rate_bridge_process(p->bridge, view.error_spec,
                                     view.post_spectrum, out) == 0;
    } else if (ok) {
        ok = dfn_res_stage_process(p->stage, view.error_spec,
                                   view.post_spectrum, p->stage_output) >= 0 &&
             audio_pipeline_synthesize_external(p->host, p->stage_output,
                                                out) == 0;
    }
    if (!ok) {
        mono_aec_dfn_res_reset(p);
        return -1;
    }
    return 0;
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
    /* The host's synthesis WOLA hop, plus the model's two lookahead hops at
     * 48 kHz or the bridge's added delay on a native grid. */
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
