/* Rate bridge between a pipeline's 8/16 kHz grid and the DFN2 stage's 48 kHz grid: the stage
 * (dfn_res_stage) in front of stream/ainr/bridge's AinrBridge, which owns the resampling, the 48 kHz
 * framing, the FIFOs and the delay; see dfn_rate_bridge.h for the signal, clock and delay contract. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dfn_rate_bridge.h"
#include "ainr_bridge.h"
#include "utility/mem_align.h"

#define ADAPTER_ALIGN 16u

struct DfnRateBridge {
    AinrBridge *bridge;
    DfnResStage *stage;
    Complex *out_spec;      /* the open frame's output spectrum (the bridge's buffer) */
    int frame_open;         /* a stage frame is open (dfn_res_stage_begin() ran) */
    void *owned_heap;
};

static size_t align_up(size_t v) {
    if (v > SIZE_MAX - (ADAPTER_ALIGN - 1u)) return 0u;
    return (v + ADAPTER_ALIGN - 1u) & ~(size_t)(ADAPTER_ALIGN - 1u);
}

static uint32_t fnv_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (8 * i)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* Everything init needs, derived from the config once. */
typedef struct Resolved {
    AinrBridgeConfig bridge_cfg;
    AinrBridgeMemReq bridge_req;
    DfnResStageMemReq stage_req;
    size_t bytes;
} Resolved;

static int resolve(const DfnRateBridgeConfig *cfg, Resolved *r) {
    size_t value, add;
    /* Link-time check: this object's DFN2 variant (full or mask-only) is the library's. */
    DFN2_REQUIRE_VARIANT_LINK();
    if (!cfg || !r) return -1;
    /* The resampling ratios, the per-hop inference schedule, the prefill calibration and the
     * delay arithmetic are written for a stage on DFN2's shipped 48 kHz grid; a model on another
     * grid is refused instead of running them against it. The lookahead pair the added delay
     * depends on is the stage grid's; a model of another pair is refused by the stage's
     * descriptor check below, and the stage itself refuses more than one frame per call. */
    if (cfg->stage.grid.sample_rate != AINR_BRIDGE_MODEL_RATE ||
        cfg->stage.grid.n_fft != AINR_BRIDGE_MODEL_FFT)
        return -1;
    if (dfn_res_stage_get_mem_requirements(&cfg->stage, &r->stage_req) != 0) return -1;
    r->bridge_cfg.sample_rate = cfg->sample_rate;
    r->bridge_cfg.fft_size = cfg->fft_size;
    r->bridge_cfg.lookahead_frames = dfn_res_stage_lookahead_frames(&cfg->stage.grid);
    if (ainr_bridge_get_mem_requirements(&r->bridge_cfg, &r->bridge_req) != 0) return -1;
    value = align_up(sizeof(DfnRateBridge));
    add = align_up((size_t)r->bridge_req.bytes);
    if (!value || !add || value > SIZE_MAX - add) return -1;
    value += add;
    add = align_up((size_t)r->stage_req.bytes);
    if (!add || value > SIZE_MAX - add) return -1;
    r->bytes = value + add;
    return 0;
}

static void fill_descriptor(const Resolved *r, DfnRateBridgeMemReq *out) {
    uint32_t h = 2166136261u;
    h = fnv_u32(h, DFN_RATE_BRIDGE_LAYOUT_VERSION);
    h = fnv_u32(h, r->stage_req.layout_version);
    h = fnv_u32(h, r->stage_req.build_flags_hash);
    h = fnv_u32(h, r->bridge_req.layout_version);
    h = fnv_u32(h, r->bridge_req.build_flags_hash);
    memset(out, 0, sizeof(*out));
    out->descriptor_version = DFN_RATE_BRIDGE_DESCRIPTOR_VERSION;
    out->layout_version = DFN_RATE_BRIDGE_LAYOUT_VERSION;
    out->backend_id = r->stage_req.backend_id;
    out->build_flags_hash = ac_memreq_abi_hash(h);
    out->alignment = ADAPTER_ALIGN;
    out->bytes = (uint64_t)r->bytes;
}

DfnRateBridgeConfig dfn_rate_bridge_default_config(int sample_rate,
                                                   const DFN2Grid *model_grid) {
    DfnRateBridgeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = 0;
    cfg.stage = dfn_res_stage_default_config(model_grid);
    return cfg;
}

int dfn_rate_bridge_get_mem_requirements(const DfnRateBridgeConfig *cfg,
                                         DfnRateBridgeMemReq *out) {
    Resolved r;
    if (!out || resolve(cfg, &r) != 0) return -1;
    fill_descriptor(&r, out);
    return 0;
}

static int descriptor_matches(const DfnRateBridgeMemReq *a,
                              const DfnRateBridgeMemReq *b) {
    return a && b && a->descriptor_version == b->descriptor_version &&
        a->layout_version == b->layout_version &&
        a->backend_id == b->backend_id &&
        a->build_flags_hash == b->build_flags_hash &&
        a->alignment == b->alignment && a->reserved == 0u &&
        a->bytes >= b->bytes;
}

DfnRateBridge *dfn_rate_bridge_init_ex(void *mem, size_t bytes,
                                       const DfnRateBridgeConfig *cfg,
                                       const DfnRateBridgeMemReq *expected) {
    Resolved r;
    DfnRateBridgeMemReq req;
    DfnRateBridge *b;
    unsigned char *cursor;
    if (!mem || (uintptr_t)mem % ADAPTER_ALIGN != 0u || resolve(cfg, &r) != 0)
        return NULL;
    fill_descriptor(&r, &req);
    if (bytes < r.bytes || (expected && !descriptor_matches(expected, &req)))
        return NULL;
    memset(mem, 0, r.bytes);
    b = (DfnRateBridge *)mem;
    cursor = (unsigned char *)mem + align_up(sizeof(*b));
    b->bridge = ainr_bridge_init_ex(cursor, (size_t)r.bridge_req.bytes, &r.bridge_cfg,
                                    &r.bridge_req);
    cursor += align_up((size_t)r.bridge_req.bytes);
    b->stage = dfn_res_stage_init_ex(cursor, (size_t)r.stage_req.bytes, &cfg->stage,
                                     &r.stage_req);
    if (!b->bridge || !b->stage) return NULL;
    return b;
}

DfnRateBridge *dfn_rate_bridge_init(void *mem, size_t bytes,
                                    const DfnRateBridgeConfig *cfg) {
    return dfn_rate_bridge_init_ex(mem, bytes, cfg, NULL);
}

DfnRateBridge *dfn_rate_bridge_create(const DfnRateBridgeConfig *cfg) {
    DfnRateBridgeMemReq req;
    DfnRateBridge *b;
    void *mem = NULL;
    if (dfn_rate_bridge_get_mem_requirements(cfg, &req) != 0) return NULL;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return NULL;
    b = dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, cfg, &req);
    if (!b) { free(mem); return NULL; }
    b->owned_heap = mem;
    return b;
}

void dfn_rate_bridge_destroy(DfnRateBridge *b) {
    void *owned;
    if (!b) return;
    owned = b->owned_heap;
    b->owned_heap = NULL;
    dfn_res_stage_destroy(b->stage);
    ainr_bridge_destroy(b->bridge);
    if (owned) free(owned);
}

void dfn_rate_bridge_reset(DfnRateBridge *b) {
    if (!b) return;
    dfn_res_stage_reset(b->stage);
    ainr_bridge_reset(b->bridge);
    b->frame_open = 0;
}

int dfn_rate_bridge_reset_synthesis(DfnRateBridge *b) {
    return b ? ainr_bridge_reset_synthesis(b->bridge) : -1;
}

int dfn_rate_bridge_begin(DfnRateBridge *b,
                          const Complex *estimate_spec,
                          const Complex *apply_spec) {
    if (!b) return -1;
    return ainr_bridge_begin(b->bridge, estimate_spec, apply_spec);
}

int dfn_rate_bridge_nn_begin(DfnRateBridge *b, DfnResStageJob *job) {
    AinrBridgeFrame frame;
    int r;
    if (!b || !job) return -1;
    r = ainr_bridge_model_begin(b->bridge, &frame);
    if (r != 1) return r;
    if (dfn_res_stage_begin(b->stage, frame.estimate_spec, frame.apply_spec, job) != 1)
        return -1;
    b->out_spec = frame.out_spec;
    b->frame_open = 1;
    return 1;
}

int dfn_rate_bridge_nn_end(DfnRateBridge *b, int model_rc) {
    if (!b || !b->frame_open) return -1;
    b->frame_open = 0;
    if (dfn_res_stage_end(b->stage, model_rc, b->out_spec) < 0) return -1;
    return ainr_bridge_model_end(b->bridge);
}

int dfn_rate_bridge_finish(DfnRateBridge *b, float *out_hop) {
    DfnResStageJob job;
    int r;
    if (!b || !out_hop) return -1;
    /* Frames left open or never fetched take the identity. */
    if (b->frame_open && dfn_rate_bridge_nn_end(b, -1) != 0) return -1;
    while ((r = dfn_rate_bridge_nn_begin(b, &job)) == 1)
        if (dfn_rate_bridge_nn_end(b, -1) != 0) return -1;
    if (r < 0) return -1;
    return ainr_bridge_finish(b->bridge, out_hop);
}

int dfn_rate_bridge_process(DfnRateBridge *b,
                            const Complex *estimate_spec,
                            const Complex *apply_spec,
                            float *out_hop) {
    DfnResStageJob job;
    int r;
    if (!b || !out_hop) return -1;
    if (dfn_rate_bridge_begin(b, estimate_spec, apply_spec) != 0) return -1;
    while ((r = dfn_rate_bridge_nn_begin(b, &job)) == 1)
        if (dfn_rate_bridge_nn_end(b, dfn_res_stage_infer(b->stage, &job)) != 0)
            return -1;
    if (r < 0) return -1;
    return dfn_rate_bridge_finish(b, out_hop);
}

int dfn_rate_bridge_hop_size(const DfnRateBridge *b) {
    return b ? ainr_bridge_hop_size(b->bridge) : -1;
}

int dfn_rate_bridge_n_freqs(const DfnRateBridge *b) {
    return b ? ainr_bridge_n_freqs(b->bridge) : -1;
}

int dfn_rate_bridge_added_delay_samples(const DfnRateBridge *b) {
    return b ? ainr_bridge_added_delay_samples(b->bridge) : -1;
}

int dfn_rate_bridge_added_delay_exact(const DfnRateBridge *b, int *numerator,
                                      int *denominator) {
    if (!b) return -1;
    return ainr_bridge_added_delay_exact(b->bridge, numerator, denominator);
}

DfnResStage *dfn_rate_bridge_stage(const DfnRateBridge *b) {
    return b ? b->stage : NULL;
}

unsigned int dfn_rate_bridge_state_generation(const DfnRateBridge *b) {
    return b ? dfn_res_stage_state_generation(b->stage) : 0u;
}

void dfn_rate_bridge_get_counters(const DfnRateBridge *b,
                                  long long *frames_48k, int *last_hop_frames) {
    ainr_bridge_get_counters(b ? b->bridge : NULL, frames_48k, last_hop_frames);
}

int dfn_rate_bridge_get_host_hops(const DfnRateBridge *b, const float **estimate_hop,
                                  const float **apply_hop) {
    return b ? ainr_bridge_get_host_hops(b->bridge, estimate_hop, apply_hop) : -1;
}
