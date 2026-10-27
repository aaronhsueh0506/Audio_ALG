/** Implementation of mcab_sample_flow.h. Sample code, not part of the mcab module. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mcab_sample_flow.h"
#include "utility/mem_align.h"

typedef struct FlowLayout {
    AecConfig lane;
    McabDelayEstimatorConfig dc;
    size_t lane_bytes;
    size_t delay_bytes;
    McabMemReq flow_req;
    size_t total;
} FlowLayout;

static int flow_layout(const McabConfig* cfg, FlowLayout* l) {
    size_t total = 0;
    int ch;
    memset(l, 0, sizeof(*l));
    if (!cfg || mcab_lane_config(cfg, &l->lane) != 0 ||
        mcab_delay_estimator_config(cfg, &l->dc) != 0 ||
        mcab_get_mem_requirements(cfg, &l->flow_req) != 0 ||
        l->flow_req.bytes > (uint64_t)SIZE_MAX) return 0;
    l->lane_bytes = aec_get_mem_size(&l->lane);
    if (l->lane_bytes == 0) return 0;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) total = ck_add_size(total, ck_align16_size(l->lane_bytes));
    if (l->dc.required) {
        l->delay_bytes = delay_aec_get_mem_size(l->dc.sample_rate, l->dc.hop_size,
                                                l->dc.num_filters);
        if (l->delay_bytes == 0) return 0;
        total = ck_add_size(total, ck_align16_size(sizeof(DelayAec)));
        total = ck_add_size(total, ck_align16_size(l->delay_bytes));
    }
    total = ck_add_size(total, ck_align16_size((size_t)l->flow_req.bytes));
    if (MEM_SIZE_INVALID(total)) return 0;
    l->total = total;
    return 1;
}

int mcab_sample_flow_get_mem_size(const McabConfig* cfg, size_t* bytes) {
    FlowLayout l;
    if (!bytes || !flow_layout(cfg, &l)) return -1;
    *bytes = l.total;
    return 0;
}

static void* take(uint8_t** cursor, size_t bytes) {
    void* out = *cursor;
    *cursor += ck_align16_size(bytes);
    return out;
}

int mcab_sample_flow_init(McabSampleFlow* f, void* mem, size_t bytes, const McabConfig* cfg) {
    FlowLayout l;
    McabLibs libs;
    uint8_t* cursor;
    void* region;
    int ch;
    if (!f) return -1;
    memset(f, 0, sizeof(*f));
    if (!mem || !MEM_IS_ALIGNED16(mem) || !flow_layout(cfg, &l) || bytes < l.total) return -1;
    memset(&libs, 0, sizeof(libs));
    cursor = (uint8_t*)mem;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        region = take(&cursor, l.lane_bytes);
        if (!(f->lanes[ch] = aec_init(region, l.lane_bytes, &l.lane))) goto fail;
        libs.lanes[ch] = f->lanes[ch];
    }
    if (l.dc.required) {
        f->delay = (DelayAec*)take(&cursor, sizeof(DelayAec));
        region = take(&cursor, l.delay_bytes);
        if (delay_aec_init(f->delay, region, l.delay_bytes, l.dc.sample_rate, l.dc.hop_size,
                           l.dc.num_filters) != 0) goto fail;
        libs.delay = f->delay;
    }
    region = take(&cursor, (size_t)l.flow_req.bytes);
    f->flow = mcab_init_ex(region, (size_t)l.flow_req.bytes, cfg, &libs, &l.flow_req);
    if (!f->flow) goto fail;
    return 0;
fail:
    mcab_sample_flow_destroy(f);
    return -1;
}

int mcab_sample_flow_create(McabSampleFlow* f, const McabConfig* cfg) {
    size_t bytes;
    void* pool = NULL;
    if (!f) return -1;
    memset(f, 0, sizeof(*f));
    if (mcab_sample_flow_get_mem_size(cfg, &bytes) != 0 ||
        posix_memalign(&pool, 16, bytes) != 0 || !pool) return -1;
    if (mcab_sample_flow_init(f, pool, bytes, cfg) != 0) {
        free(pool);
        return -1;
    }
    f->owned_heap = pool;
    return 0;
}

void mcab_sample_flow_destroy(McabSampleFlow* f) {
    int ch;
    if (!f) return;
    mcab_destroy(f->flow);   /* first: it borrows the objects below */
    f->flow = NULL;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        if (f->lanes[ch]) aec_destroy(f->lanes[ch]);
        f->lanes[ch] = NULL;
    }
    f->delay = NULL;
    free(f->owned_heap);
    f->owned_heap = NULL;
}
