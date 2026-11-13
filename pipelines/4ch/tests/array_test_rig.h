/** Test scaffolding: builds an array instance (aec_get_mem_requirements() + aec_init_ex() in one heap
 * block) and, for the beamformer tests, the sample helper on top of it, and finds the objects again from the handle the tests pass around, so a
 * test reads like "create, run, destroy". The objects are still created by their own libraries'
 * init functions; nothing here is a new code path of the library. */
#ifndef ARRAY_TEST_RIG_H
#define ARRAY_TEST_RIG_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "array_sample_beam.h"     /* array_sample_default_config() */
#include "utility/mem_align.h"

#define ARRAY_TEST_RIG_SLOTS 64

typedef struct ArrayTestRigSlot {
    Aec* aec;
    void* pool;                   /* the heap block of the array; NULL when the caller owns the pool */
    AecConfig cfg;
    void* beam_mem;
    void* beam;                   /* ArraySampleBeam* with ARRAY_TEST_RIG_WITH_BEAM */
    int used;
} ArrayTestRigSlot;

static ArrayTestRigSlot array_test_rig_slots[ARRAY_TEST_RIG_SLOTS];

static inline ArrayTestRigSlot* array_test_rig_take(void) {
    int i;
    for (i = 0; i < ARRAY_TEST_RIG_SLOTS; ++i)
        if (!array_test_rig_slots[i].used) {
            memset(&array_test_rig_slots[i], 0, sizeof(array_test_rig_slots[i]));
            array_test_rig_slots[i].used = 1;
            return &array_test_rig_slots[i];
        }
    return NULL;
}

static inline ArrayTestRigSlot* array_test_rig_find_array(const Aec* p) {
    int i;
    for (i = 0; p && i < ARRAY_TEST_RIG_SLOTS; ++i)
        if (array_test_rig_slots[i].used && array_test_rig_slots[i].aec == p)
            return &array_test_rig_slots[i];
    return NULL;
}

/** The samples' array config at `sample_rate` (balanced, MATCHED, residual FULL, four microphones). */
static inline AecConfig array_test_default_config(int sample_rate) {
    AecConfig cfg;
    array_sample_default_config(&cfg, sample_rate);
    return cfg;
}

/** The same at an AEC level; the level reaches the gain stage of the array, not its lanes. */
static inline AecConfig array_test_level_config(int sample_rate, int fft_size, AecLevel level) {
    AecConfig cfg;
    aec_config_from_level(&cfg, level, sample_rate);
    cfg.n_mics = AEC_ARRAY_MICS;
    cfg.residual = AEC_RESIDUAL_FULL;
    cfg.fft_size = fft_size;
    return cfg;
}

/* Grid of an array handle: hop and bin count from the library, the FFT length is twice the hop. */
static inline int array_test_n_freqs(const Aec* a) { return aec_hop_size(a) + 1; }
static inline int array_test_fft_size(const Aec* a) { return 2 * aec_hop_size(a); }
/* The sample rate an array of the rig was built at; aborts for a handle the rig did not create. */
static inline int array_test_sample_rate(const Aec* a) {
    ArrayTestRigSlot* s = array_test_rig_find_array(a);
    if (!s) {
        fprintf(stderr, "array_test_sample_rate: the array was not created by the rig\n");
        abort();
    }
    return s->cfg.sample_rate;
}

/** aec_process_linear() + aec_get_array(). */
static inline int array_test_front(Aec* a, const float* mics, const float* ref, AecArrayView* v) {
    int rc = aec_process_linear(a, mics, ref);
    if (rc != AEC_OK) return rc;
    memset(v, 0, sizeof(*v));
    v->struct_size = sizeof(*v);
    return aec_get_array(a, v);
}

/** aec_process_residual() with the caller's weights and (optional) beamformed spectrum + aec_get_residual(). */
static inline int array_test_fuse(Aec* a, const Complex* weights, const Complex* trusted,
                                 AecResidualView* v) {
    AecBeam beam;
    int rc;
    beam.weights = weights;
    beam.beamformed_error = trusted;
    rc = aec_process_residual(a, &beam);
    if (rc != AEC_OK) return rc;
    memset(v, 0, sizeof(*v));
    v->struct_size = sizeof(*v);
    return aec_get_residual(a, v);
}

/** Bytes of the array pool for cfg (a multiple of 16) and its descriptor; -1 for an invalid config or one
 * that is not a four-microphone array. */
static inline int array_test_array_bytes(const AecConfig* cfg, AecMemReq* req, size_t* bytes) {
    if (!cfg || cfg->n_mics != AEC_ARRAY_MICS || aec_get_mem_requirements(cfg, req) != 0 ||
        req->bytes > (uint64_t)SIZE_MAX) return -1;
    *bytes = ALIGN16((size_t)req->bytes);
    return 0;
}

/** Places the array of cfg in `pool` (>= its bytes) and records it in the slot; -1 on any refusal. */
static inline int array_test_slot_place(ArrayTestRigSlot* s, void* pool, size_t pool_bytes,
                                       const AecConfig* cfg) {
    AecMemReq req;
    size_t bytes;
    if (array_test_array_bytes(cfg, &req, &bytes) != 0 || pool_bytes < bytes) return -1;
    s->aec = aec_init_ex(pool, bytes, cfg, &req);
    if (!s->aec) return -1;
    s->cfg = *cfg;
    return 0;
}

/** The array in one heap block owned by the slot; -1 on any refusal. */
static inline int array_test_slot_create(ArrayTestRigSlot* s, const AecConfig* cfg) {
    AecMemReq req;
    size_t bytes;
    if (array_test_array_bytes(cfg, &req, &bytes) != 0 || posix_memalign(&s->pool, 16, bytes) != 0 ||
        array_test_slot_place(s, s->pool, bytes, cfg) != 0) {
        free(s->pool);
        s->pool = NULL;
        return -1;
    }
    return 0;
}

static inline void array_test_slot_destroy(ArrayTestRigSlot* s) {
    aec_destroy(s->aec);
    free(s->pool);
    s->used = 0;
}

/** Heap array for cfg; NULL for an invalid config. */
static inline Aec* array_test_create(const AecConfig* cfg) {
    ArrayTestRigSlot* s = array_test_rig_take();
    if (!s) return NULL;
    if (array_test_slot_create(s, cfg) != 0) {
        s->used = 0;
        return NULL;
    }
    return s->aec;
}

static inline void array_test_destroy(Aec* p) {
    ArrayTestRigSlot* s = array_test_rig_find_array(p);
    if (s) array_test_slot_destroy(s);
}

#ifdef ARRAY_TEST_RIG_WITH_BEAM
/** An array config and a beamformer helper config, side by side. */
typedef struct ArrayTestUnitConfig {
    AecConfig aec_cfg;
    ArraySampleBeamParams beam;
} ArrayTestUnitConfig;

static inline ArrayTestUnitConfig array_test_unit_default_config(int sample_rate) {
    ArrayTestUnitConfig c;
    array_sample_default_config(&c.aec_cfg, sample_rate);
    c.beam = array_sample_beam_default_params();
    return c;
}

static inline ArrayTestRigSlot* array_test_rig_find_beam(const ArraySampleBeam* b) {
    int i;
    for (i = 0; b && i < ARRAY_TEST_RIG_SLOTS; ++i)
        if (array_test_rig_slots[i].used && array_test_rig_slots[i].beam == b)
            return &array_test_rig_slots[i];
    return NULL;
}

/** Heap array + helper; NULL for an invalid array or helper config. */
static inline ArraySampleBeam* array_test_beam_create(const ArrayTestUnitConfig* cfg) {
    ArrayTestRigSlot* s;
    size_t bytes;
    if (!cfg || array_sample_beam_get_mem_size(&cfg->aec_cfg, &cfg->beam, &bytes) != 0) return NULL;
    s = array_test_rig_take();
    if (!s) return NULL;
    if (array_test_slot_create(s, &cfg->aec_cfg) != 0 ||
        posix_memalign(&s->beam_mem, 16, bytes) != 0 ||
        !(s->beam = array_sample_beam_init(s->beam_mem, bytes, &cfg->aec_cfg, &cfg->beam, s->aec))) {
        array_test_slot_destroy(s);
        free(s->beam_mem);
        return NULL;
    }
    return (ArraySampleBeam*)s->beam;
}

/** One caller pool for the array and the helper (array first, 16-byte aligned). The descriptor is the
 * array's own with `bytes` replaced by the whole pool; -1 for an invalid array or helper config. */
static inline int array_test_unit_get_mem_requirements(const ArrayTestUnitConfig* cfg,
                                                      AecMemReq* req) {
    size_t aec_bytes, beam_bytes;
    if (!cfg || !req || array_test_array_bytes(&cfg->aec_cfg, req, &aec_bytes) != 0 ||
        array_sample_beam_get_mem_size(&cfg->aec_cfg, &cfg->beam, &beam_bytes) != 0) return -1;
    req->bytes = (uint64_t)(aec_bytes + beam_bytes);
    return 0;
}

/** Array + helper placed in `pool` (the bytes of array_test_unit_get_mem_requirements()). */
static inline ArraySampleBeam* array_test_beam_init(void* pool, size_t bytes,
                                                  const ArrayTestUnitConfig* cfg) {
    ArrayTestRigSlot* s;
    AecMemReq req, aec_req;
    size_t aec_bytes;
    if (array_test_unit_get_mem_requirements(cfg, &req) != 0 || bytes < (size_t)req.bytes ||
        array_test_array_bytes(&cfg->aec_cfg, &aec_req, &aec_bytes) != 0) return NULL;
    s = array_test_rig_take();
    if (!s) return NULL;
    if (array_test_slot_place(s, pool, aec_bytes, &cfg->aec_cfg) != 0 ||
        !(s->beam = array_sample_beam_init((unsigned char*)pool + aec_bytes,
                                          (size_t)req.bytes - aec_bytes, &cfg->aec_cfg,
                                          &cfg->beam, s->aec))) {
        array_test_slot_destroy(s);
        return NULL;
    }
    return (ArraySampleBeam*)s->beam;
}

/** The array under a rig helper; NULL for NULL or an unknown/destroyed helper. */
static inline Aec* array_test_beam_aec(const ArraySampleBeam* b) {
    ArrayTestRigSlot* s = array_test_rig_find_beam(b);
    return s ? s->aec : NULL;
}

/** Helper first (it borrows the array), then the array. */
static inline void array_test_beam_destroy(ArraySampleBeam* b) {
    ArrayTestRigSlot* s = array_test_rig_find_beam(b);
    if (!s) return;
    array_sample_beam_destroy(b);
    array_test_slot_destroy(s);
    free(s->beam_mem);
}
#endif

#endif /* ARRAY_TEST_RIG_H */
