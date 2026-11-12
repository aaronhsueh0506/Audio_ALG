#include "aiaec_front.h"
#include "aiaec_front_internal.h"
#include "aiaec_front_test.h"

#include <string.h>

#include "aec.h"
#include "delay_aec.h"
#include "stream_mem.h"

_Static_assert((int)AIAEC_FRONT_DELAY_MATCHED == (int)AEC_DELAY_MATCHED, "delay mode values");
_Static_assert((int)AIAEC_FRONT_DELAY_FIXED == (int)AEC_DELAY_FIXED, "delay mode values");
_Static_assert((int)AIAEC_FRONT_DELAY_EXTERNAL == (int)AEC_DELAY_EXTERNAL_ALIGNED, "delay mode values");
_Static_assert(DA_NUM_FILTERS == 5, "aiaec_front_config_defaults documents 5 matched filters");

#define FRONT_MAX_HOP_IN 65536

typedef struct Layout {
    size_t aec_bytes;
    size_t aec_off, mic_off, far_off, formed_off, events_off, total;
} Layout;

static size_t take(size_t *cur, size_t bytes) {
    size_t off = ck_align16_size(*cur);
    *cur = ck_add_size(off, bytes);
    return off;
}

void aiaec_front_config_defaults(AiaecFrontConfig *fc, const AiaecConfig *grid) {
    if (!fc) return;
    memset(fc, 0, sizeof(*fc));
    if (grid) fc->grid = *grid;
    fc->delay_mode = AIAEC_FRONT_DELAY_MATCHED;
    fc->delay_num_filters = 5;
    fc->fixed_delay_samples = -1;
}

static void build_aec_config(const AiaecFrontConfig *fc, AecConfig *c) {
    aec_config_from_level(c, AEC_LEVEL_BALANCED, fc->grid.sr);
    c->fft_size = fc->grid.fft;
    c->delay_mode = (AecDelayMode)fc->delay_mode;
    c->delay_num_filters = fc->delay_num_filters;
    c->fixed_delay_samples = fc->fixed_delay_samples;
    c->enable_delay_est = (fc->delay_mode == AIAEC_FRONT_DELAY_MATCHED) ? 1 : 0;
    c->enable_res = 0;
    c->return_res_context = 1;
}

static int validate(const AiaecFrontConfig *fc, int *hop) {
    AecSignalGrid g;
    if (!fc) return AIAEC_ERR_NULL;
    if (!aec_resolve_signal_grid(fc->grid.sr, fc->grid.fft, &g) || g.is_legacy) return AIAEC_ERR_GRID;
    if (fc->grid.hop_in <= 0 || fc->grid.hop_in > FRONT_MAX_HOP_IN || fc->grid.hop_in % g.hop_size != 0)
        return AIAEC_ERR_IN_HOP;
    switch (fc->delay_mode) {
    case AIAEC_FRONT_DELAY_MATCHED:
        if (fc->delay_num_filters < 1 || fc->delay_num_filters > 5 || fc->fixed_delay_samples >= 0)
            return AIAEC_FRONT_ERR_DELAY;
        break;
    case AIAEC_FRONT_DELAY_FIXED:
        if (fc->fixed_delay_samples < 0) return AIAEC_FRONT_ERR_DELAY;
        break;
    case AIAEC_FRONT_DELAY_EXTERNAL:
        if (fc->fixed_delay_samples >= 0) return AIAEC_FRONT_ERR_DELAY;
        break;
    default:
        return AIAEC_FRONT_ERR_DELAY;
    }
    *hop = g.hop_size;
    return AIAEC_OK;
}

static int plan(const AiaecFrontConfig *fc, int *hop, Layout *l) {
    AecConfig ac;
    size_t cur = 0, n;
    int rc = validate(fc, hop);
    if (rc != AIAEC_OK) return rc;
    build_aec_config(fc, &ac);
    l->aec_bytes = aec_get_mem_size(&ac);
    if (l->aec_bytes == 0) return AIAEC_FRONT_ERR_AEC;
    n = (size_t)fc->grid.hop_in;
    take(&cur, sizeof(AiaecFront));
    l->aec_off = take(&cur, l->aec_bytes);
    l->mic_off = take(&cur, n * sizeof(float));
    l->far_off = take(&cur, n * sizeof(float));
    l->formed_off = take(&cur, n * sizeof(float));
    l->events_off = take(&cur, (n / (size_t)*hop) * sizeof(AiaecAlignEvent));
    l->total = ck_align16_size(cur);
    return MEM_SIZE_INVALID(l->total) ? AIAEC_ERR_POOL : AIAEC_OK;
}

size_t aiaec_front_get_mem_size_cfg(const AiaecFrontConfig *fc) {
    Layout l;
    int hop;
    return plan(fc, &hop, &l) == AIAEC_OK ? l.total : 0;
}

AiaecFront *aiaec_front_init_cfg(void *pool, size_t pool_bytes, const AiaecFrontConfig *fc, int *err) {
    Layout l;
    AecConfig ac;
    AiaecFront *f;
    unsigned char *base = (unsigned char *)pool;
    int hop = 0, rc = plan(fc, &hop, &l);

    if (rc == AIAEC_OK && !pool) rc = AIAEC_ERR_NULL;
    if (rc == AIAEC_OK && (((uintptr_t)pool % AIAEC_FRONT_ALIGNMENT) != 0 || pool_bytes < l.total))
        rc = AIAEC_ERR_POOL;
    if (rc != AIAEC_OK) {
        if (err) *err = rc;
        return NULL;
    }
    memset(base, 0, l.total);
    f = (AiaecFront *)base;
    build_aec_config(fc, &ac);
    f->aec = aec_init(base + l.aec_off, l.aec_bytes, &ac);
    if (!f->aec) {
        if (err) *err = AIAEC_FRONT_ERR_AEC;
        return NULL;
    }
    f->mic_f = (float *)(base + l.mic_off);
    f->far_f = (float *)(base + l.far_off);
    f->formed = (float *)(base + l.formed_off);
    f->events = (AiaecAlignEvent *)(base + l.events_off);
    f->hop = hop;
    f->hop_in = fc->grid.hop_in;
    f->frames = fc->grid.hop_in / hop;
    if (err) *err = AIAEC_OK;
    return f;
}

size_t aiaec_front_get_mem_size(const AiaecConfig *grid) {
    AiaecFrontConfig fc;
    if (!grid) return 0;
    aiaec_front_config_defaults(&fc, grid);
    return aiaec_front_get_mem_size_cfg(&fc);
}

AiaecFront *aiaec_front_init(void *pool, size_t pool_bytes, const AiaecConfig *grid, int *err) {
    AiaecFrontConfig fc;
    if (!grid) {
        if (err) *err = AIAEC_ERR_NULL;
        return NULL;
    }
    aiaec_front_config_defaults(&fc, grid);
    return aiaec_front_init_cfg(pool, pool_bytes, &fc, err);
}

AiaecAlignEvent aiaec_front_event_from_delay_state(int delay_state) {
    AiaecAlignEvent ev;
    ev.changed = (delay_state == (int)AEC_LINEAR_DELAY_CHANGED);
    ev.solid = (delay_state == (int)AEC_LINEAR_DELAY_LOCKED || delay_state == (int)AEC_LINEAR_DELAY_CHANGED);
    return ev;
}

static int run(AiaecFront *f, const float *mic, const float *far, AiaecFrontOut *out) {
    int h;
    for (h = 0; h < f->frames; ++h) {
        AecResContext rc;
        AecLinearContext lc;
        aec_process_context(f->aec, mic + (size_t)h * f->hop, far + (size_t)h * f->hop);
        aec_get_res_context(f->aec, &rc);
        aec_get_linear_context(f->aec, &lc);
        if (!rc.formed_hop) return AIAEC_FRONT_ERR_AEC;
        memcpy(f->formed + (size_t)h * f->hop, rc.formed_hop, (size_t)f->hop * sizeof(float));
        f->events[h] = aiaec_front_event_from_delay_state((int)lc.delay_state);
    }
    out->formed_hop = f->formed;
    out->raw_far = far;
    out->events = f->events;
    out->event_count = f->frames;
    out->samples = f->hop_in;
    return AIAEC_OK;
}

static int gate(const AiaecFront *f, const void *mic, const void *far, int samples, AiaecFrontOut *out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!f || !mic || !far || !out) return AIAEC_ERR_NULL;
    if (samples != f->hop_in) return AIAEC_ERR_INPUT_SIZE;
    return AIAEC_OK;
}

int aiaec_front_process_f32(AiaecFront *f, const float *mic, const float *far, int samples,
                            AiaecFrontOut *out) {
    int rc = gate(f, mic, far, samples, out);
    if (rc != AIAEC_OK) return rc;
    rc = run(f, mic, far, out);
    if (rc != AIAEC_OK) memset(out, 0, sizeof(*out));
    return rc;
}

int aiaec_front_process_i16(AiaecFront *f, const int16_t *mic, const int16_t *far, int samples,
                            AiaecFrontOut *out) {
    int i, rc = gate(f, mic, far, samples, out);
    if (rc != AIAEC_OK) return rc;
    for (i = 0; i < samples; ++i) {
        f->mic_f[i] = (float)mic[i] / 32768.0f;
        f->far_f[i] = (float)far[i] / 32768.0f;
    }
    rc = run(f, f->mic_f, f->far_f, out);
    if (rc != AIAEC_OK) memset(out, 0, sizeof(*out));
    return rc;
}

void aiaec_front_reset(AiaecFront *f) {
    if (!f) return;
    aec_reset(f->aec);
    memset(f->formed, 0, (size_t)f->hop_in * sizeof(float));
    memset(f->events, 0, (size_t)f->frames * sizeof(AiaecAlignEvent));
}

void aiaec_front_fill_align_input(const AiaecFrontOut *out, AiaecInput *in) {
    if (!in) return;
    memset(in, 0, sizeof(*in));
    if (!out) return;
    in->kind = AIAEC_INPUT_ALIGN_ULCNET;
    in->samples = out->samples;
    in->u.align_ulcnet.residual_in = out->formed_hop;
    in->u.align_ulcnet.far_in = out->raw_far;
    in->u.align_ulcnet.events = out->events;
    in->u.align_ulcnet.event_count = out->event_count;
}
