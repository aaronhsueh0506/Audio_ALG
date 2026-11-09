/** Test scaffolding: builds a flow (mcab_sample_flow.h) and, for the beamformer tests, the sample
 * helper on top of it, and finds the objects again from the handle the tests pass around, so a
 * test reads like "create, run, destroy". The objects are still created by their own libraries'
 * init functions; nothing here is a new code path of the module. */
#ifndef MCAB_TEST_RIG_H
#define MCAB_TEST_RIG_H

#include <stdlib.h>
#include <string.h>

#include "mcab.h"
#include "mcab_sample_flow.h"
#ifdef MCAB_TEST_RIG_WITH_BEAM
#include "mcab_sample_beam.h"
#endif

#define MCAB_TEST_RIG_SLOTS 64

typedef struct McabTestRigSlot {
    McabSampleFlow flow;
    void* beam_mem;
    void* beam;                   /* McabSampleBeam* with MCAB_TEST_RIG_WITH_BEAM */
    int used;
} McabTestRigSlot;

static McabTestRigSlot mcab_test_rig_slots[MCAB_TEST_RIG_SLOTS];

static inline McabTestRigSlot* mcab_test_rig_take(void) {
    int i;
    for (i = 0; i < MCAB_TEST_RIG_SLOTS; ++i)
        if (!mcab_test_rig_slots[i].used) {
            memset(&mcab_test_rig_slots[i], 0, sizeof(mcab_test_rig_slots[i]));
            mcab_test_rig_slots[i].used = 1;
            return &mcab_test_rig_slots[i];
        }
    return NULL;
}

static inline McabTestRigSlot* mcab_test_rig_find_flow(const Mcab* p) {
    int i;
    for (i = 0; p && i < MCAB_TEST_RIG_SLOTS; ++i)
        if (mcab_test_rig_slots[i].used && mcab_test_rig_slots[i].flow.flow == p)
            return &mcab_test_rig_slots[i];
    return NULL;
}

/** Heap flow for cfg; NULL for an invalid config. */
static inline Mcab* mcab_test_create(const McabConfig* cfg) {
    McabTestRigSlot* s = mcab_test_rig_take();
    if (!s) return NULL;
    if (mcab_sample_flow_create(&s->flow, cfg) != 0) {
        s->used = 0;
        return NULL;
    }
    return s->flow.flow;
}

/** The lanes and estimator of a rig flow (for tests that look at them directly). */
static inline const McabSampleFlow* mcab_test_objects(const Mcab* p) {
    McabTestRigSlot* s = mcab_test_rig_find_flow(p);
    return s ? &s->flow : NULL;
}

static inline void mcab_test_destroy(Mcab* p) {
    McabTestRigSlot* s = mcab_test_rig_find_flow(p);
    if (!s) return;
    mcab_sample_flow_destroy(&s->flow);
    s->used = 0;
}

/** Lanes and estimator alone, for tests that place mcab's own pool themselves. */
typedef struct McabTestLibs {
    McabLibs libs;
    void* lane_mem[MCAB_CHANNELS];
    void* delay_mem;
} McabTestLibs;

static inline void mcab_test_libs_free(McabTestLibs* t) {
    int ch;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        if (t->libs.lanes[ch]) aec_destroy(t->libs.lanes[ch]);
        free(t->lane_mem[ch]);
    }
    free(t->libs.delay);
    free(t->delay_mem);
    memset(t, 0, sizeof(*t));
}

/** The objects of cfg's flow with the given lane config (NULL: mcab_lane_config()), which is
 * used as is. 0, or -1 (everything freed) for an invalid config. */
static inline int mcab_test_libs_create_with_lane(const McabConfig* cfg, const AecConfig* app_lane,
                                                  McabTestLibs* t) {
    AecConfig lane;
    McabDelayEstimatorConfig dc;
    size_t bytes;
    int ch;
    memset(t, 0, sizeof(*t));
    if (app_lane) lane = *app_lane;
    else if (mcab_lane_config(cfg, &lane) != 0) return -1;
    if (mcab_delay_estimator_config(cfg, &dc) != 0 || (bytes = aec_get_mem_size(&lane)) == 0)
        return -1;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        if (posix_memalign(&t->lane_mem[ch], 16, bytes) != 0 ||
            !(t->libs.lanes[ch] = aec_init(t->lane_mem[ch], bytes, &lane))) goto fail;
    }
    if (dc.required) {
        bytes = delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters);
        t->libs.delay = (DelayAec*)calloc(1, sizeof(DelayAec));
        if (!t->libs.delay || bytes == 0 || posix_memalign(&t->delay_mem, 16, bytes) != 0 ||
            delay_aec_init(t->libs.delay, t->delay_mem, bytes, dc.sample_rate, dc.hop_size,
                           dc.num_filters) != 0) goto fail;
    }
    return 0;
fail:
    mcab_test_libs_free(t);
    return -1;
}

static inline int mcab_test_libs_create(const McabConfig* cfg, McabTestLibs* t) {
    return mcab_test_libs_create_with_lane(cfg, NULL, t);
}

#ifdef MCAB_TEST_RIG_WITH_BEAM
/** A flow config and a beamformer helper config, side by side. */
typedef struct McabTestUnitConfig {
    McabConfig core;
    McabSampleBeamParams beam;
} McabTestUnitConfig;

static inline McabTestUnitConfig mcab_test_unit_default_config(int sample_rate) {
    McabTestUnitConfig c;
    c.core = mcab_default_config(sample_rate);
    c.beam = mcab_sample_beam_default_params();
    return c;
}

static inline McabTestRigSlot* mcab_test_rig_find_beam(const McabSampleBeam* b) {
    int i;
    for (i = 0; b && i < MCAB_TEST_RIG_SLOTS; ++i)
        if (mcab_test_rig_slots[i].used && mcab_test_rig_slots[i].beam == b)
            return &mcab_test_rig_slots[i];
    return NULL;
}

/** Heap flow + helper; NULL for an invalid flow or helper config. */
static inline McabSampleBeam* mcab_test_beam_create(const McabTestUnitConfig* cfg) {
    McabTestRigSlot* s;
    size_t bytes;
    if (!cfg || mcab_sample_beam_get_mem_size(&cfg->core, &cfg->beam, &bytes) != 0) return NULL;
    s = mcab_test_rig_take();
    if (!s) return NULL;
    if (mcab_sample_flow_create(&s->flow, &cfg->core) != 0 ||
        posix_memalign(&s->beam_mem, 16, bytes) != 0 ||
        !(s->beam = mcab_sample_beam_init(s->beam_mem, bytes, &cfg->core, &cfg->beam,
                                          s->flow.flow))) {
        mcab_sample_flow_destroy(&s->flow);
        free(s->beam_mem);
        s->used = 0;
        return NULL;
    }
    return (McabSampleBeam*)s->beam;
}

/** One caller pool for the flow's objects and the helper (flow first, 16-byte aligned). The
 * descriptor is mcab's own with `bytes` replaced by the whole pool; -1 for an invalid flow or
 * helper config. */
static inline int mcab_test_unit_get_mem_requirements(const McabTestUnitConfig* cfg,
                                                      McabMemReq* req) {
    size_t flow_bytes, beam_bytes;
    if (!cfg || !req || mcab_get_mem_requirements(&cfg->core, req) != 0 ||
        mcab_sample_flow_get_mem_size(&cfg->core, &flow_bytes) != 0 ||
        mcab_sample_beam_get_mem_size(&cfg->core, &cfg->beam, &beam_bytes) != 0) return -1;
    req->bytes = (uint64_t)(flow_bytes + beam_bytes);
    return 0;
}

/** Flow + helper placed in `pool` (the bytes of mcab_test_unit_get_mem_requirements()). */
static inline McabSampleBeam* mcab_test_beam_init(void* pool, size_t bytes,
                                                  const McabTestUnitConfig* cfg) {
    McabTestRigSlot* s;
    McabMemReq req;
    size_t flow_bytes;
    if (mcab_test_unit_get_mem_requirements(cfg, &req) != 0 || bytes < (size_t)req.bytes ||
        mcab_sample_flow_get_mem_size(&cfg->core, &flow_bytes) != 0) return NULL;
    s = mcab_test_rig_take();
    if (!s) return NULL;
    if (mcab_sample_flow_init(&s->flow, pool, flow_bytes, &cfg->core) != 0 ||
        !(s->beam = mcab_sample_beam_init((unsigned char*)pool + flow_bytes,
                                          (size_t)req.bytes - flow_bytes, &cfg->core,
                                          &cfg->beam, s->flow.flow))) {
        mcab_sample_flow_destroy(&s->flow);
        s->used = 0;
        return NULL;
    }
    return (McabSampleBeam*)s->beam;
}

/** The flow under a rig helper; NULL for NULL or an unknown/destroyed helper. */
static inline Mcab* mcab_test_beam_flow(const McabSampleBeam* b) {
    McabTestRigSlot* s = mcab_test_rig_find_beam(b);
    return s ? s->flow.flow : NULL;
}

/** Helper first (it borrows the flow), then the flow and its lanes. */
static inline void mcab_test_beam_destroy(McabSampleBeam* b) {
    McabTestRigSlot* s = mcab_test_rig_find_beam(b);
    if (!s) return;
    mcab_sample_beam_destroy(b);
    mcab_sample_flow_destroy(&s->flow);
    free(s->beam_mem);
    s->used = 0;
}
#endif

#endif /* MCAB_TEST_RIG_H */
