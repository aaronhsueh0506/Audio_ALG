#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_dfn_res.h"
#include "audio_stft.h"
#include "simd_kernels.h"

#define WRAP_ALIGN 16u

struct FourAecDfnRes {
    FourAecNrRes *front;
    AecRes *res;                /* post-beam residual-echo suppressor      */
    float *g_total;             /* min(unity, G_res)                       */
    float *unity;               /* all-ones gain read for the absent NR    */
    Complex *post;              /* P = E * g_total (+ comfort noise)       */
    AcIstft *istft;             /* 48 kHz: this application's own synthesis */
    uint32_t rng;               /* comfort-noise generator state           */
    int cng;                    /* comfort-noise fill enabled              */
    int n_freqs;
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
 * config, the sub-descriptors and this pool's total. The core validates its
 * own config; only the seam invariants are checked here. */
typedef struct Resolved {
    int use_bridge;                 /* core below 48 kHz */
    DfnResStageConfig scfg;
    DfnRateBridgeConfig bcfg;
    FourAecNrResMemReq front_req;
    DfnResStageMemReq stage_req;    /* 48 kHz */
    DfnRateBridgeMemReq bridge_req; /* 8/16 kHz */
    AecConfig res_cfg;
    size_t res_bytes;
    size_t synth_bytes;             /* 48 kHz */
    int n_freqs;
    int fft_size;
    uint32_t sub_layout;            /* the stage's or the bridge's */
    uint32_t sub_hash;
    uint32_t backend_id;
    size_t bytes;
} Resolved;

static int add_region(size_t *total, size_t bytes) {
    size_t add = align_up(bytes);
    if (!add || *total > SIZE_MAX - add) return -1;
    *total += add;
    return 0;
}

static int resolve(const FourAecDfnResConfig *cfg, Resolved *r) {
    FourAecNrResMemBreakdown bd;
    AnrConfig nr_cfg;
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
                                             &r->front_req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(&cfg->front_end, &bd) != 0 ||
        four_aec_nr_res_stage_configs(&cfg->front_end, &r->res_cfg,
                                      &nr_cfg) != 0)
        return -1;
    r->n_freqs = bd.n_freqs;
    r->fft_size = bd.fft_size;
    r->res_bytes = aec_res_get_mem_size(&r->res_cfg);
    if (r->res_bytes == 0) return -1;
    if (r->use_bridge) {
        r->bcfg = dfn_rate_bridge_default_config(cfg->front_end.sample_rate);
        r->bcfg.fft_size = cfg->front_end.fft_size;
        r->bcfg.stage = r->scfg;
        if (dfn_rate_bridge_get_mem_requirements(&r->bcfg, &r->bridge_req) != 0)
            return -1;
        r->sub_layout = r->bridge_req.layout_version;
        r->sub_hash = r->bridge_req.build_flags_hash;
        r->backend_id = r->bridge_req.backend_id;
        r->synth_bytes = 0;
    } else {
        AcStftConfig scfg;
        if (cfg->front_end.fft_size != 0 &&
            cfg->front_end.fft_size != DFN2_N_FFT)
            return -1;
        if (dfn_res_stage_get_mem_requirements(&r->scfg, &r->stage_req) != 0)
            return -1;
        r->sub_layout = r->stage_req.layout_version;
        r->sub_hash = r->stage_req.build_flags_hash;
        r->backend_id = r->stage_req.backend_id;
        ac_stft_config_default(&scfg, r->fft_size);
        r->synth_bytes = ac_istft_get_mem_size(&scfg);
        if (r->synth_bytes == 0) return -1;
    }
    if (r->front_req.backend_id != r->backend_id) return -1;
    total = align_up(sizeof(FourAecDfnRes));
    if (!total ||
        add_region(&total, (size_t)r->front_req.bytes) ||
        add_region(&total, r->res_bytes) ||
        add_region(&total, (size_t)r->n_freqs * sizeof(float)) ||    /* g_total */
        add_region(&total, (size_t)r->n_freqs * sizeof(float)) ||    /* unity   */
        add_region(&total, (size_t)r->n_freqs * sizeof(Complex)))    /* post    */
        return -1;
    if (r->use_bridge) {
        if (add_region(&total, (size_t)r->bridge_req.bytes)) return -1;
    } else {
        if (add_region(&total, r->synth_bytes) ||
            add_region(&total, (size_t)r->stage_req.bytes) ||
            add_region(&total, (size_t)DFN2_N_BINS * sizeof(Complex)))
            return -1;
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
    int k;
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
    p->res = aec_res_init(cursor, r.res_bytes, &r.res_cfg);
    if (!p->res || aec_res_n_bins(p->res) != r.n_freqs ||
        aec_res_hop_size(p->res) != four_aec_nr_res_hop_size(p->front))
        return NULL;
    cursor += align_up(r.res_bytes);
    p->n_freqs = r.n_freqs;
    p->g_total = (float *)cursor;
    cursor += align_up((size_t)r.n_freqs * sizeof(float));
    p->unity = (float *)cursor;
    cursor += align_up((size_t)r.n_freqs * sizeof(float));
    p->post = (Complex *)cursor;
    cursor += align_up((size_t)r.n_freqs * sizeof(Complex));
    for (k = 0; k < r.n_freqs; ++k) p->unity[k] = 1.0f;
    p->rng = SE_CNG_RNG_SEED;
    /* The comfort-noise fill is the front end's own switch. */
    p->cng = cfg->front_end.enable_cng;
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
    {
        AcStftConfig scfg;
        ac_stft_config_default(&scfg, r.fft_size);
        p->istft = ac_istft_init(cursor, r.synth_bytes, &scfg);
        if (!p->istft) return NULL;
        cursor += align_up(r.synth_bytes);
    }
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
    aec_res_reset(p->res);
    p->rng = SE_CNG_RNG_SEED;
    if (p->bridge) {
        dfn_rate_bridge_reset(p->bridge);   /* resets its stage too */
        return;
    }
    dfn_res_stage_reset(p->stage);
    ac_istft_reset(p->istft);
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

/* out->delay.changed is reported for external STFT/OLA consumers of the
 * far-end; the stage consumes no far-end and is left running. */
int four_aec_dfn_res_front_process(FourAecDfnRes *p,
                                   const float *microphones_interleaved,
                                   const float *ref,
                                   FourAecFront *out) {
    if (!p) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    return four_aec_front_process(p->front, microphones_interleaved, ref, out);
}

static int spectrum_finite(const Complex *s, int n) {
    int k;
    for (k = 0; k < n; ++k)
        if (!isfinite(s[k].r) || !isfinite(s[k].i)) return 0;
    return 1;
}

int four_aec_dfn_res_process(FourAecDfnRes *p, const Complex *weights,
                             const Complex *beamformed_error, float *out) {
    SeSpecFrame f;
    const float *res_gain;
    int rc, ok, k;
    if (!p || !out) return FOUR_AEC_NR_RES_INVALID_ARGUMENT;
    rc = four_aec_fuse(p->front, weights, beamformed_error, &f);
    if (rc != FOUR_AEC_NR_RES_OK) {
        if (rc == FOUR_AEC_NR_RES_DSP_ERROR) four_aec_dfn_res_reset(p);
        return rc;
    }
    res_gain = se_res_stage_gain(p->res, &f);
    if (!res_gain) {
        four_aec_dfn_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    /* P = E * min(1, G_res), plus comfort noise on the cut bins. The absent
     * denoiser reads as the unity vector, so the same four calls as the NR
     * pipeline serve. */
    sk_min_f32(p->g_total, p->unity, res_gain, p->n_freqs);
    sk_capply_gain_f32(p->post, f.e, p->g_total, p->n_freqs);
    if (p->cng)
        aec_cng_fill(&p->rng, p->post, f.cn, res_gain, p->unity,
                     SE_CNG_NR_GAIN_FLOOR, p->n_freqs);
    if (p->bridge) {
        ok = dfn_rate_bridge_process(p->bridge, f.e, p->post, out) == 0;
    } else {
        ok = dfn_res_stage_process(p->stage, f.e, p->post,
                                   p->stage_output) >= 0 &&
             spectrum_finite(p->stage_output, p->n_freqs);
        if (ok) {
            ac_istft_push(p->istft, p->stage_output, out);
            for (k = 0; k < DFN2_HOP_LEN; ++k)
                if (!isfinite(out[k])) { ok = 0; break; }
        }
    }
    if (!ok) {
        four_aec_dfn_res_reset(p);
        return FOUR_AEC_NR_RES_DSP_ERROR;
    }
    return FOUR_AEC_NR_RES_OK;
}

int four_aec_dfn_res_set_aec_preset(FourAecDfnRes *p, AecPreset preset,
                                    float ramp_ms) {
    return p ? aec_res_set_preset(p->res, preset, ramp_ms) : -1;
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
