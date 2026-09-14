#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_dfn_res.h"

#define WRAP_ALIGN 16u

struct FourAecDfnRes {
    FourAecNrRes *front;
    DfnResStage *stage;         /* 48 kHz: the stage, driven directly */
    Complex *stage_output;      /* 48 kHz: its output spectrum          */
    DfnRateBridge *bridge;      /* 8/16 kHz: the stage inside the bridge */
    void *owned_heap;
};

static size_t align_up(size_t v) {
    if (v > SIZE_MAX - (WRAP_ALIGN - 1u)) return 0u;
    return (v + WRAP_ALIGN - 1u) & ~(size_t)(WRAP_ALIGN - 1u);
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
 * config, both sub-descriptors and this pool's total. The core validates
 * its own config; only the seam invariants are checked here. */
typedef struct Resolved {
    int use_bridge;                 /* core below 48 kHz */
    DfnResStageConfig scfg;
    DfnRateBridgeConfig bcfg;
    FourAecNrResMemReq front_req;
    DfnResStageMemReq stage_req;    /* 48 kHz */
    DfnRateBridgeMemReq bridge_req; /* 8/16 kHz */
    uint32_t sub_layout;            /* the stage's or the bridge's */
    uint32_t sub_hash;
    uint32_t backend_id;
    size_t bytes;
} Resolved;

static int resolve(const FourAecDfnResConfig *cfg, Resolved *r) {
    size_t total;
    if (!cfg || !r ||
        (cfg->front_end.sample_rate != 16000 &&
         cfg->front_end.sample_rate != DFN2_SR) ||
        cfg->front_end.enable_post != 1 || cfg->front_end.enable_res != 1 ||
        cfg->front_end.enable_nr != 0 || !isfinite(cfg->atten_lim_db))
        return -1;
    r->use_bridge = cfg->front_end.sample_rate != DFN2_SR;
    r->scfg = dfn_res_stage_default_config(DFN2_SR);
    r->scfg.atten_lim_db = cfg->atten_lim_db;
    r->scfg.erb_fwd = cfg->erb_fwd;
    r->scfg.erb_inv = cfg->erb_inv;
    r->scfg.model = cfg->model;
    if (four_aec_nr_res_get_mem_requirements(&cfg->front_end,
                                             &r->front_req) != 0)
        return -1;
    if (r->use_bridge) {
        r->bcfg = dfn_rate_bridge_default_config(cfg->front_end.sample_rate);
        r->bcfg.fft_size = cfg->front_end.fft_size;
        r->bcfg.stage = r->scfg;
        if (dfn_rate_bridge_get_mem_requirements(&r->bcfg, &r->bridge_req) != 0)
            return -1;
        r->sub_layout = r->bridge_req.layout_version;
        r->sub_hash = r->bridge_req.build_flags_hash;
        r->backend_id = r->bridge_req.backend_id;
    } else {
        if (cfg->front_end.fft_size != 0 &&
            cfg->front_end.fft_size != DFN2_N_FFT)
            return -1;
        if (dfn_res_stage_get_mem_requirements(&r->scfg, &r->stage_req) != 0)
            return -1;
        r->sub_layout = r->stage_req.layout_version;
        r->sub_hash = r->stage_req.build_flags_hash;
        r->backend_id = r->stage_req.backend_id;
    }
    if (r->front_req.backend_id != r->backend_id) return -1;
    total = align_up(sizeof(FourAecDfnRes));
    if (!total || r->front_req.bytes > SIZE_MAX - total) return -1;
    total += align_up((size_t)r->front_req.bytes);
    if (r->use_bridge) {
        if (r->bridge_req.bytes > SIZE_MAX - total) return -1;
        total += align_up((size_t)r->bridge_req.bytes);
    } else {
        if (r->stage_req.bytes > SIZE_MAX - total) return -1;
        total += align_up((size_t)r->stage_req.bytes);
        if ((size_t)DFN2_N_BINS * sizeof(Complex) > SIZE_MAX - total) return -1;
        total += align_up((size_t)DFN2_N_BINS * sizeof(Complex));
    }
    r->bytes = total;
    return 0;
}

static void fill_descriptor(const Resolved *r, FourAecDfnResMemReq *out) {
    uint32_t h = 2166136261u;
    h = fnv_u32(h, FOUR_AEC_DFN_RES_LAYOUT_VERSION);
    h = fnv_u32(h, r->front_req.layout_version);
    h = fnv_u32(h, r->front_req.build_flags_hash);
    h = fnv_u32(h, (uint32_t)r->use_bridge);
    h = fnv_u32(h, r->sub_layout);
    h = fnv_u32(h, r->sub_hash);
    memset(out, 0, sizeof(*out));
    out->descriptor_version = FOUR_AEC_DFN_RES_DESCRIPTOR_VERSION;
    out->layout_version = FOUR_AEC_DFN_RES_LAYOUT_VERSION;
    out->backend_id = r->front_req.backend_id;
    out->build_flags_hash = h;
    out->alignment = WRAP_ALIGN;
    out->bytes = (uint64_t)r->bytes;
}

FourAecDfnResConfig four_aec_dfn_res_default_config(void) {
    FourAecDfnResConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.front_end = four_aec_nr_res_default_config(16000);
    cfg.front_end.enable_post = 1;
    cfg.front_end.enable_res = 1;
    cfg.front_end.enable_nr = 0;
    cfg.front_end.enable_cng = 0;
    return cfg;
}

int four_aec_dfn_res_get_mem_requirements(const FourAecDfnResConfig *cfg,
                                          FourAecDfnResMemReq *out) {
    Resolved r;
    if (!out || resolve(cfg, &r) != 0) return -1;
    fill_descriptor(&r, out);
    return 0;
}

static int descriptor_matches(const FourAecDfnResMemReq *expected,
                              const FourAecDfnResMemReq *current) {
    return expected && current &&
        expected->descriptor_version == current->descriptor_version &&
        expected->layout_version == current->layout_version &&
        expected->backend_id == current->backend_id &&
        expected->build_flags_hash == current->build_flags_hash &&
        expected->alignment == current->alignment &&
        expected->reserved == 0u && expected->bytes >= current->bytes;
}

FourAecDfnRes *four_aec_dfn_res_init_ex(void *mem, size_t bytes,
                                        const FourAecDfnResConfig *cfg,
                                        const FourAecDfnResMemReq *expected) {
    Resolved r;
    FourAecDfnResMemReq req;
    unsigned char *cursor;
    FourAecDfnRes *p;
    if (!mem || (uintptr_t)mem % WRAP_ALIGN != 0u || resolve(cfg, &r) != 0)
        return NULL;
    fill_descriptor(&r, &req);
    if (bytes < r.bytes || (expected && !descriptor_matches(expected, &req)))
        return NULL;
    memset(mem, 0, r.bytes);
    p = (FourAecDfnRes *)mem;
    cursor = (unsigned char *)mem + align_up(sizeof(*p));
    p->front = four_aec_nr_res_init_ex(cursor, (size_t)r.front_req.bytes,
                                        &cfg->front_end, &r.front_req);
    if (!p->front) return NULL;
    cursor += align_up((size_t)r.front_req.bytes);
    /* The core resolves fft_size 0 to its rate default; the stage's grid
     * (48 kHz) or the bridge's native grid must be the core's, checked once
     * here. */
    if (r.use_bridge) {
        p->bridge = dfn_rate_bridge_init_ex(cursor, (size_t)r.bridge_req.bytes,
                                            &r.bcfg, &r.bridge_req);
        if (!p->bridge ||
            dfn_rate_bridge_n_freqs(p->bridge) != four_aec_nr_res_n_freqs(p->front) ||
            dfn_rate_bridge_hop_size(p->bridge) != four_aec_nr_res_hop_size(p->front))
            return NULL;
        p->stage = dfn_rate_bridge_stage(p->bridge);
        return p;
    }
    if (four_aec_nr_res_n_freqs(p->front) != DFN2_N_BINS ||
        four_aec_nr_res_hop_size(p->front) != DFN2_HOP_LEN)
        return NULL;
    p->stage = dfn_res_stage_init_ex(cursor, (size_t)r.stage_req.bytes,
                                     &r.scfg, &r.stage_req);
    if (!p->stage) return NULL;
    cursor += align_up((size_t)r.stage_req.bytes);
    p->stage_output = (Complex *)cursor;
    return p;
}

FourAecDfnRes *four_aec_dfn_res_init(void *mem, size_t bytes,
                                     const FourAecDfnResConfig *cfg) {
    return four_aec_dfn_res_init_ex(mem, bytes, cfg, NULL);
}

FourAecDfnRes *four_aec_dfn_res_create(const FourAecDfnResConfig *cfg) {
    FourAecDfnResMemReq req;
    FourAecDfnRes *p;
    void *mem = NULL;
    if (four_aec_dfn_res_get_mem_requirements(cfg, &req) != 0) return NULL;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return NULL;
    p = four_aec_dfn_res_init_ex(mem, (size_t)req.bytes, cfg, &req);
    if (!p) { free(mem); return NULL; }
    p->owned_heap = mem;
    return p;
}

void four_aec_dfn_res_reset(FourAecDfnRes *p) {
    if (!p) return;
    four_aec_nr_res_reset(p->front);
    if (p->bridge) {
        dfn_rate_bridge_reset(p->bridge);   /* resets its stage too */
        return;
    }
    dfn_res_stage_reset(p->stage);
    memset(p->stage_output, 0, (size_t)DFN2_N_BINS * sizeof(Complex));
}

void four_aec_dfn_res_destroy(FourAecDfnRes *p) {
    void *owned;
    if (!p) return;
    owned = p->owned_heap;
    p->owned_heap = NULL;
    if (p->bridge) dfn_rate_bridge_destroy(p->bridge);
    else dfn_res_stage_destroy(p->stage);
    four_aec_nr_res_destroy(p->front);
    if (owned) free(owned);
}

/* The core owns the frame sequencing: its token gate refuses a second pre
 * while a frame is pending and a post without one, so this wrapper passes
 * both calls straight through. */
int four_aec_dfn_res_process_pre(FourAecDfnRes *p,
                                 const float *microphones_interleaved,
                                 const float *ref,
                                 FourAecNrResPreFrame *out) {
    if (!p) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    /* out->delay.changed is reported for external STFT/OLA consumers of the
     * far-end; the stage consumes no far-end and is left running. */
    return four_aec_nr_res_process_pre(p->front, microphones_interleaved,
                                       ref, out);
}

static int process_post(FourAecDfnRes *p,
                        const FourAecNrResFrameToken *token,
                        const Complex *weights,
                        const Complex *trusted,
                        float *out) {
    FourAecNrResPostView view;
    int rc, ok;
    if (!p || !out) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    rc = four_aec_nr_res_process_post_view(p->front, token, weights,
                                            trusted, &view);
    if (rc != FOUR_AEC_NR_RES_OK) {
        if (rc == FOUR_AEC_NR_RES_DSP_ERROR) four_aec_dfn_res_reset(p);
        return rc;
    }
    if (p->bridge) {
        ok = dfn_rate_bridge_process(p->bridge, view.beamformed_error,
                                     view.post_spectrum, out) == 0;
    } else {
        ok = dfn_res_stage_process(p->stage, view.beamformed_error,
                                   view.post_spectrum, p->stage_output) >= 0 &&
             four_aec_nr_res_synthesize_external(
                 p->front, p->stage_output, out) == FOUR_AEC_NR_RES_OK;
    }
    if (!ok) {
        four_aec_dfn_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    return FOUR_AEC_NR_RES_OK;
}

int four_aec_dfn_res_process_post(FourAecDfnRes *p,
                                  const FourAecNrResFrameToken *token,
                                  const Complex *weights,
                                  float *out) {
    return process_post(p, token, weights, NULL, out);
}

int four_aec_dfn_res_process_post_trusted_spectrum(
    FourAecDfnRes *p,
    const FourAecNrResFrameToken *token,
    const Complex *weights,
    const Complex *beamformed_error,
    float *out) {
    if (!beamformed_error) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    return process_post(p, token, weights, beamformed_error, out);
}

int four_aec_dfn_res_set_aec_preset(FourAecDfnRes *p, AecPreset preset,
                                    float ramp_ms) {
    return p ? four_aec_nr_res_set_aec_preset(p->front, preset, ramp_ms) : -1;
}

int four_aec_dfn_res_set_atten_lim(FourAecDfnRes *p, float atten_lim_db) {
    return p ? dfn_res_stage_set_atten_lim(p->stage, atten_lim_db) : -1;
}

int four_aec_dfn_res_hop_size(const FourAecDfnRes *p) {
    return p ? four_aec_nr_res_hop_size(p->front) : -1;
}

int four_aec_dfn_res_lookahead_samples(const FourAecDfnRes *p) {
    int hop = four_aec_dfn_res_hop_size(p);
    if (hop <= 0) return -1;
    /* The core's synthesis WOLA hop, plus the model's two lookahead hops at
     * 48 kHz or the bridge's added delay on a native grid. */
    if (p->bridge) return hop + dfn_rate_bridge_added_delay_samples(p->bridge);
    return hop * (DFN_RES_STAGE_LOOKAHEAD_FRAMES + 1);
}

FourAecNrRes *four_aec_dfn_res_get_front_end(const FourAecDfnRes *p) {
    return p ? p->front : NULL;
}

DfnResStage *four_aec_dfn_res_get_stage(const FourAecDfnRes *p) {
    return p ? p->stage : NULL;
}

DfnRateBridge *four_aec_dfn_res_get_bridge(const FourAecDfnRes *p) {
    return p ? p->bridge : NULL;
}
