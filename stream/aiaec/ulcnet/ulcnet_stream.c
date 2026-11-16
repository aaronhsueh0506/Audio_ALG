/* Align-ULCNet PCM stream: everything (no file-loader TU for this model). No <stdio.h>, no
 * malloc/calloc/posix_memalign anywhere in this TU -- it links into a NO_STDIO target. See
 * ulcnet_stream.h for the public contract. */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ulcnet_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "stream_mem.h"
#include "stream_range.h"
#include "stream_run.h"

/* IDLE: no pending hop (post_process refuses with -1). FRAME_OPEN: views were handed out; post_process validates the run
 * or restarts on ai_rc. FRAME_CLOSED: a reprime hop, identity by construction; ai_rc is ignored. */
enum {
    ULCNET_STREAM_IDLE = 0,
    ULCNET_STREAM_FRAME_OPEN = 1,
    ULCNET_STREAM_FRAME_CLOSED = 2
};

struct UlcnetStream {
    UlcnetGrid grid;
    int frames;         /* hops per accelerator job: 1, or the packet size */

    /* fft, stft_err, stft_far, istft and the scratch: the PCM half, NULL in a freq_only stream. */
    FftHandle *fft;     /* shared by stft_err / stft_far / istft */
    AcStft    *stft_err;
    AcStft    *stft_far;
    AcIstft   *istft;
    UlcnetPrepost *prepost;

    /* Per-job scratch, one slab: err_re / err_im [frames * n_bins] (this call's error spectrum, the source of every
     * identity frame), events [frames] (packet only: 1 = alignment-event hop), then the PCM half: far_re / far_im
     * [n_bins] (one far frame), enh_re / enh_im [frames * n_bins], hop_out [hop]. */
    float *err_re;
    float *err_im;
    int   *events;
    float *far_re;
    float *far_im;
    float *enh_re;
    float *enh_im;
    float *hop_out;

    int hop_state;              /* ULCNET_STREAM_IDLE / _FRAME_OPEN / _FRAME_CLOSED */
    int prev_solid;               /* delay.solid of the previous hop; 1 after init and reset */
    unsigned reprime_remaining;  /* identity hops still owed after an alignment-edge restart */
};

typedef struct UlcnetStreamLayout {
    UlcnetStream *self;
    void *fft_mem;
    void *stft_err_mem;
    void *stft_far_mem;
    void *istft_mem;
    void *prepost_mem;
    size_t prepost_bytes;
    void *scratch_mem;   /* the per-job scratch slab, carved for both families */
    uint32_t prepost_build_hash;   /* UlcnetPrepostMemReq.build_flags_hash, folded into this stream's hash */
} UlcnetStreamLayout;

#define ULCNET_STREAM_ALIGN ULCNET_STREAM_ALIGNMENT

static int cfg_frames(const UlcnetStreamConfig *cfg) {
    return cfg->frames_per_call > 1 ? cfg->frames_per_call : 1;
}

static int cfg_valid(const UlcnetStreamConfig *cfg) {
    if (!cfg) return 0;
    if (cfg->frames_per_call < 0 || cfg->frames_per_call > ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL) return 0;
    if (ulcnet_grid_validate(&cfg->grid) != ULCNET_GRID_OK) return 0;
    return 1;
}

/* One carve walk: sizing (base == NULL, *out pointers stay NULL) or carving; *total must match across both with one cfg.
 * -1 on a bad config or an overflowing carve. Order (bump LAYOUT_VERSION and build_flags_hash on a change): self, fft,
 * stft_err, stft_far, istft, prepost, scratch; cfg.freq_only: self, prepost, scratch. */
static int ulcnet_stream_layout(const UlcnetStreamConfig *cfg, unsigned char *base,
                                UlcnetStreamLayout *out, size_t *total) {
    size_t cursor = 0;
    void *ptr;
    AcStftConfig scfg;
    UlcnetPrepostConfig pcfg;
    UlcnetPrepostMemReq pp_req;
    size_t fft_bytes, stft_bytes, istft_bytes, scratch_bytes, scratch_units;
    size_t hop, plane_bins;

    if (!cfg || !out || !total || !cfg_valid(cfg)) return -1;
    memset(out, 0, sizeof(*out));
    hop = (size_t)cfg->grid.hop;
    plane_bins = (size_t)cfg_frames(cfg) * (size_t)cfg->grid.n_bins;

    if (stream_carve(base, &cursor, sizeof(UlcnetStream), &ptr) != 0) return -1;
    out->self = (UlcnetStream *)ptr;

    if (!cfg->freq_only) {
        fft_bytes = fft_get_mem_size(cfg->grid.n_fft);
        if (fft_bytes == 0 || stream_carve(base, &cursor, fft_bytes, &ptr) != 0) return -1;
        out->fft_mem = ptr;

        /* One AcStftConfig, reused for sizing stft_err/stft_far/istft; scfg.fft stays NULL here --
         * audio_stft.c's config_valid() dereferences a non-NULL cfg->fft, which doesn't exist yet
         * during sizing. */
        ac_stft_config_default(&scfg, cfg->grid.n_fft);
        scfg.ola = AC_OLA_ACCUMULATE;

        stft_bytes = ac_stft_get_mem_size(&scfg);
        if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        out->stft_err_mem = ptr;

        if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        out->stft_far_mem = ptr;

        istft_bytes = ac_istft_get_mem_size(&scfg);
        if (istft_bytes == 0 || stream_carve(base, &cursor, istft_bytes, &ptr) != 0) return -1;
        out->istft_mem = ptr;
    }

    if (ulcnet_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return -1;
    pcfg.frames_per_call = cfg->frames_per_call;
    if (ulcnet_prepost_get_mem_size(&pcfg, &pp_req) != 0) return -1;
    if (stream_carve(base, &cursor, (size_t)pp_req.bytes, &ptr) != 0) return -1;
    out->prepost_mem = ptr;
    out->prepost_bytes = (size_t)pp_req.bytes;
    out->prepost_build_hash = pp_req.build_flags_hash;

    /* err_re, err_im [frames * n_bins] + events [frames] (packet only), then for the PCM half far_re, far_im [n_bins],
     * enh_re, enh_im [frames * n_bins], hop_out [hop]: one slab, only its base is part of the pool ABI. */
    scratch_units = ck_add_size(ck_mul_size(2u, plane_bins), cfg_frames(cfg) > 1 ? (size_t)cfg_frames(cfg) : 0u);
    if (!cfg->freq_only) {
        scratch_units = ck_add_size(scratch_units,
                                    ck_add_size(ck_add_size(ck_mul_size(2u, (size_t)cfg->grid.n_bins),
                                                            ck_mul_size(2u, plane_bins)), hop));
    }
    scratch_bytes = ck_mul_size(scratch_units, sizeof(float));
    if (MEM_SIZE_INVALID(scratch_bytes) || stream_carve(base, &cursor, scratch_bytes, &ptr) != 0) return -1;
    out->scratch_mem = ptr;

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

static uint32_t build_hash(const UlcnetStreamConfig *cfg, uint32_t prepost_hash) {
    uint32_t h = AC_FNV1A_SEED;
    const UlcnetGrid *g = &cfg->grid;
    h = stream_fnv1a_u32(h, (uint32_t)g->sample_rate);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_fft);
    h = stream_fnv1a_u32(h, (uint32_t)g->hop);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_bins);
    h = stream_fnv1a_u32(h, (uint32_t)g->delay_depth);
    h = stream_fnv1a_u32(h, (uint32_t)ULCNET_STREAM_LAYOUT_VERSION);
    h = stream_fnv1a_u32(h, prepost_hash);
    h = stream_fnv1a_u32(h, cfg->freq_only ? 1u : 0u);
    if (cfg_frames(cfg) > 1) h = stream_fnv1a_u32(h, (uint32_t)cfg_frames(cfg));
    return ac_memreq_abi_hash(h);
}

int ulcnet_stream_get_mem_size(const UlcnetStreamConfig *cfg, UlcnetStreamMemReq *req) {
    UlcnetStreamLayout layout;
    size_t total = 0;

    if (!req || !cfg_valid(cfg)) return -1;
    if (ulcnet_stream_layout(cfg, NULL, &layout, &total) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = ULCNET_STREAM_DESCRIPTOR_VERSION;
    req->layout_version = ULCNET_STREAM_LAYOUT_VERSION;
    req->build_flags_hash = build_hash(cfg, layout.prepost_build_hash);
    req->alignment = ULCNET_STREAM_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* Field-by-field memory-requirement comparison for the stale-pool gate. */
static int ulcnet_stream_descriptor_matches(const UlcnetStreamMemReq *expected,
                                            const UlcnetStreamMemReq *req) {
    return expected->descriptor_version == req->descriptor_version &&
           expected->layout_version == req->layout_version &&
           expected->reserved0 == req->reserved0 &&
           expected->build_flags_hash == req->build_flags_hash &&
           expected->alignment == req->alignment &&
           expected->reserved == req->reserved &&
           expected->bytes == req->bytes;
}

/* Constructs every carved object; NULL leaves no valid instance. Called after ulcnet_stream_layout() and the caller's
 * size / alignment / stale-pool checks. */
static UlcnetStream *ulcnet_stream_finish_init(const UlcnetStreamConfig *cfg, unsigned char *base,
                                               const UlcnetStreamLayout *layout) {
    UlcnetStream *s;
    AcStftConfig scfg;
    UlcnetPrepostConfig pcfg;
    size_t n_bins, plane_bins;
    float *tail;

    if (!cfg || !base || !layout || !layout->self) return NULL;
    s = layout->self;

    s->grid = cfg->grid;
    s->frames = cfg_frames(cfg);
    s->hop_state = ULCNET_STREAM_IDLE;
    s->prev_solid = 1;
    s->reprime_remaining = 0;

    s->fft = NULL;
    s->stft_err = s->stft_far = NULL;
    s->istft = NULL;

    if (!cfg->freq_only) {
        s->fft = fft_init(layout->fft_mem, fft_get_mem_size(cfg->grid.n_fft), cfg->grid.n_fft);
        if (!s->fft) return NULL;

        /* Real, now-constructed handle, borrowed by all three framing objects -- a smaller scfg
         * than the sizing one above, which the larger carved region always fits. */
        ac_stft_config_default(&scfg, cfg->grid.n_fft);
        scfg.ola = AC_OLA_ACCUMULATE;
        scfg.fft = s->fft;

        s->stft_err = ac_stft_init(layout->stft_err_mem, ac_stft_get_mem_size(&scfg), &scfg);
        if (!s->stft_err) return NULL;

        s->stft_far = ac_stft_init(layout->stft_far_mem, ac_stft_get_mem_size(&scfg), &scfg);
        if (!s->stft_far) return NULL;

        s->istft = ac_istft_init(layout->istft_mem, ac_istft_get_mem_size(&scfg), &scfg);
        if (!s->istft) return NULL;
    }

    if (ulcnet_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return NULL;
    pcfg.frames_per_call = cfg->frames_per_call;
    s->prepost = ulcnet_prepost_init(layout->prepost_mem, layout->prepost_bytes, &pcfg);
    if (!s->prepost) return NULL;

    n_bins = (size_t)cfg->grid.n_bins;
    plane_bins = (size_t)s->frames * n_bins;
    s->err_re = (float *)layout->scratch_mem;
    s->err_im = s->err_re + plane_bins;
    s->events = NULL;
    tail = s->err_im + plane_bins;
    if (s->frames > 1) {
        s->events = (int *)tail;
        tail = (float *)(s->events + s->frames);
    }
    if (!cfg->freq_only) {
        s->far_re = tail;
        s->far_im = s->far_re + n_bins;
        s->enh_re = s->far_im + n_bins;
        s->enh_im = s->enh_re + plane_bins;
        s->hop_out = s->enh_im + plane_bins;
    }

    return s;
}

UlcnetStream *ulcnet_stream_init_ex(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes,
                                    const UlcnetStreamMemReq *expected) {
    UlcnetStreamMemReq req;
    UlcnetStreamLayout layout;
    size_t total = 0;

    if (!pool || !cfg_valid(cfg)) return NULL;
    if (ulcnet_stream_get_mem_size(cfg, &req) != 0) return NULL;
    if (pool_bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected && !ulcnet_stream_descriptor_matches(expected, &req)) return NULL;

    memset(pool, 0, (size_t)req.bytes);
    if (ulcnet_stream_layout(cfg, (unsigned char *)pool, &layout, &total) != 0) return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    return ulcnet_stream_finish_init(cfg, (unsigned char *)pool, &layout);
}

UlcnetStream *ulcnet_stream_init(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes) {
    return ulcnet_stream_init_ex(cfg, pool, pool_bytes, NULL);
}

void ulcnet_stream_reset(UlcnetStream *s) {
    if (!s) return;
    if (s->stft_err) {   /* a freq_only stream has no framing */
        ac_stft_reset(s->stft_err);
        ac_stft_reset(s->stft_far);
        ac_istft_reset(s->istft);
    }
    ulcnet_prepost_reset(s->prepost);
    s->hop_state = ULCNET_STREAM_IDLE;
    s->reprime_remaining = 0;
    s->prev_solid = 1;   /* the next hop reads like the first hop after init: no rising edge */
}

uint32_t ulcnet_stream_state_generation(const UlcnetStream *s) {
    return s ? ulcnet_prepost_state_generation(s->prepost) : 0u;
}

int ulcnet_stream_hop_size(const UlcnetStream *s) {
    return s ? s->grid.hop : -1;
}

/* Restarts the model only (state, generation, reprime): the analysis and synthesis keep running. */
static void restart_model(UlcnetStream *s) {
    ulcnet_prepost_reset(s->prepost);
    s->hop_state = ULCNET_STREAM_IDLE;
    s->reprime_remaining = ULCNET_STREAM_REPRIME_FRAMES;
}

/* Alignment-edge detection, before the hop's own analysis: a new alignment generation, or `solid` turning true
 * after a hop where it was false. prev_solid is 1 after init and reset, so a first hop only restarts on `changed`. */
static void apply_alignment_edge(UlcnetStream *s, UlcnetStreamDelay delay) {
    if (delay.changed || (!s->prev_solid && delay.solid)) restart_model(s);
    s->prev_solid = delay.solid;
}

/* Hands out the views of the open run after the last pre_frame of the call. */
static int open_block(UlcnetStream *s, UlcnetHopIO *io) {
    if (ulcnet_prepost_views(s->prepost, &io->inputs, &io->outputs) != 0) return -1;
    io->state_generation = ulcnet_prepost_state_generation(s->prepost);
    s->hop_state = ULCNET_STREAM_FRAME_OPEN;
    return 1;
}

/* Shared tail of the one-hop entries: s->err_* hold the hop's error spectrum. A reprime hop is identity: no features,
 * no views, state and generation untouched. */
static int open_hop(UlcnetStream *s, const float *far_re, const float *far_im, UlcnetHopIO *io) {
    if (s->reprime_remaining > 0) {
        s->reprime_remaining--;
        memset(io, 0, sizeof(*io));
        s->hop_state = ULCNET_STREAM_FRAME_CLOSED;
        return 0;
    }
    if (ulcnet_prepost_pre_frame(s->prepost, 0, s->err_re, s->err_im, far_re, far_im, 0) != 0) return -1;
    return open_block(s, io);
}

static int pcm_in_range(const float *err, const float *far, size_t n) {
    return stream_plane_within(err, n, ULCNET_STREAM_PCM_LIMIT_BITS) &&
           stream_plane_within(far, n, ULCNET_STREAM_PCM_LIMIT_BITS);
}

static int bins_in_range(const float *err_re, const float *err_im, const float *far_re, const float *far_im, size_t n) {
    return stream_plane_within(err_re, n, ULCNET_STREAM_BIN_LIMIT_BITS) &&
           stream_plane_within(err_im, n, ULCNET_STREAM_BIN_LIMIT_BITS) &&
           stream_plane_within(far_re, n, ULCNET_STREAM_BIN_LIMIT_BITS) &&
           stream_plane_within(far_im, n, ULCNET_STREAM_BIN_LIMIT_BITS);
}

int ulcnet_stream_pre_process_f32(UlcnetStream *s, const float *err_in, const float *far_in,
                                  UlcnetStreamDelay delay, UlcnetHopIO *io) {
    if (!s || !err_in || !far_in || !io || !s->stft_err || s->hop_state != ULCNET_STREAM_IDLE || s->frames != 1) return -1;
    if (!pcm_in_range(err_in, far_in, (size_t)s->grid.hop)) return ULCNET_STREAM_ERR_RANGE;

    apply_alignment_edge(s, delay);

    /* err_in and far_in are read once here and not stored. */
    ac_stft_push_planar(s->stft_err, err_in, s->err_re, s->err_im);
    ac_stft_push_planar(s->stft_far, far_in, s->far_re, s->far_im);

    return open_hop(s, s->far_re, s->far_im, io);
}

int ulcnet_stream_pre_process_freq(UlcnetStream *s, const float *err_re, const float *err_im,
                                   const float *far_re, const float *far_im,
                                   UlcnetStreamDelay delay, UlcnetHopIO *io) {
    const size_t bins = s ? (size_t)s->grid.n_bins : 0u;

    if (!s || !err_re || !err_im || !far_re || !far_im || !io || s->hop_state != ULCNET_STREAM_IDLE ||
        s->frames != 1)
        return -1;
    if (!bins_in_range(err_re, err_im, far_re, far_im, bins)) return ULCNET_STREAM_ERR_RANGE;

    apply_alignment_edge(s, delay);
    memcpy(s->err_re, err_re, bins * sizeof(float));
    memcpy(s->err_im, err_im, bins * sizeof(float));
    return open_hop(s, far_re, far_im, io);
}

/* The one-hop edge rule without its model restart: the graph resets and bypasses an event hop of a packet. */
static void packet_events(UlcnetStream *s, const UlcnetStreamDelay *delay) {
    int k;
    for (k = 0; k < s->frames; ++k) {
        s->events[k] = delay[k].changed || (!s->prev_solid && delay[k].solid);
        s->prev_solid = delay[k].solid;
    }
}

int ulcnet_stream_pre_process_packet_f32(UlcnetStream *s, const float *err_in, const float *far_in,
                                         const UlcnetStreamDelay *delay, UlcnetHopIO *io) {
    int k;
    const size_t hop = s ? (size_t)s->grid.hop : 0u, bins = s ? (size_t)s->grid.n_bins : 0u;

    if (!s || !err_in || !far_in || !delay || !io || !s->stft_err || s->hop_state != ULCNET_STREAM_IDLE || s->frames < 2)
        return -1;
    if (!pcm_in_range(err_in, far_in, (size_t)s->frames * hop)) return ULCNET_STREAM_ERR_RANGE;

    packet_events(s, delay);
    for (k = 0; k < s->frames; ++k) {
        float *er = s->err_re + (size_t)k * bins, *ei = s->err_im + (size_t)k * bins;
        ac_stft_push_planar(s->stft_err, err_in + (size_t)k * hop, er, ei);
        ac_stft_push_planar(s->stft_far, far_in + (size_t)k * hop, s->far_re, s->far_im);
        if (ulcnet_prepost_pre_frame(s->prepost, k, er, ei, s->far_re, s->far_im, s->events[k]) != 0) return -1;
    }
    return open_block(s, io);
}

int ulcnet_stream_pre_process_packet_freq(UlcnetStream *s, const float *err_re, const float *err_im,
                                          const float *far_re, const float *far_im,
                                          const UlcnetStreamDelay *delay, UlcnetHopIO *io) {
    int k;
    size_t plane, bins;

    if (!s || !err_re || !err_im || !far_re || !far_im || !delay || !io || s->hop_state != ULCNET_STREAM_IDLE ||
        s->frames < 2)
        return -1;
    bins = (size_t)s->grid.n_bins;
    plane = (size_t)s->frames * bins;
    if (!bins_in_range(err_re, err_im, far_re, far_im, plane)) return ULCNET_STREAM_ERR_RANGE;

    packet_events(s, delay);
    memcpy(s->err_re, err_re, plane * sizeof(float));
    memcpy(s->err_im, err_im, plane * sizeof(float));
    for (k = 0; k < s->frames; ++k) {
        const size_t o = (size_t)k * bins;
        if (ulcnet_prepost_pre_frame(s->prepost, k, err_re + o, err_im + o, far_re + o, far_im + o, s->events[k]) != 0)
            return -1;
    }
    return open_block(s, io);
}

/* Closes the call: out_re / out_im (frames * n_bins each) get the enhanced frames of a good run, else the error
 * spectrum of every frame. Every post_frame completes before the caller emits anything, so a bad frame k turns all
 * frames identity with one restart. */
static void close_block(UlcnetStream *s, int ai_rc, float *out_re, float *out_im) {
    const size_t bins = (size_t)s->grid.n_bins;
    int ok = 0;
    int k;

    if (s->hop_state == ULCNET_STREAM_FRAME_OPEN) {
        if (ai_rc == STREAM_RUN_OK) {
            if (ulcnet_prepost_validate_run(s->prepost) == 0) {
                ok = 1;
                for (k = 0; ok && k < s->frames; ++k) {
                    const size_t o = (size_t)k * bins;
                    if (s->events && s->events[k]) {
                        memcpy(out_re + o, s->err_re + o, bins * sizeof(float));
                        memcpy(out_im + o, s->err_im + o, bins * sizeof(float));
                    } else {
                        ok = ulcnet_prepost_post_frame(s->prepost, k, out_re + o, out_im + o) == ULCNET_PREPOST_OK;
                    }
                }
                if (!ok) ulcnet_prepost_zero_state(s->prepost);
            }
        } else {
            /* A packet's event hop restarts the model in the graph; without the run, restart it here. */
            int restart = ai_rc != STREAM_RUN_NOT_STARTED;
            for (k = 0; !restart && s->events && k < s->frames; ++k) restart = s->events[k] != 0;
            if (restart) ulcnet_prepost_zero_state(s->prepost);
        }
    }
    /* ULCNET_STREAM_FRAME_CLOSED: a reprime hop; ai_rc is not read. */
    if (!ok) {
        memcpy(out_re, s->err_re, (size_t)s->frames * bins * sizeof(float));
        memcpy(out_im, s->err_im, (size_t)s->frames * bins * sizeof(float));
    }
    s->hop_state = ULCNET_STREAM_IDLE;
}

int ulcnet_stream_post_process_freq(UlcnetStream *s, int ai_rc, float *re, float *im) {
    if (!s || !re || !im || s->hop_state == ULCNET_STREAM_IDLE) return -1;
    close_block(s, ai_rc, re, im);
    return 1;
}

int ulcnet_stream_post_process(UlcnetStream *s, int ai_rc, int16_t *pcm_out) {
    int hop, i, k;
    const size_t bins = s ? (size_t)s->grid.n_bins : 0u;

    if (!s || !pcm_out || !s->istft || s->hop_state == ULCNET_STREAM_IDLE) return -1;
    hop = s->grid.hop;

    close_block(s, ai_rc, s->enh_re, s->enh_im);

    /* AC_OLA_ACCUMULATE always emits a full hop with no first-push suppression (same convention
     * as dfn2_stream.c's own post_process): the first real hop out of a fresh init/reset/reprime
     * is the model's actual half-window synthesis result (may be non-zero), not forced silence. */
    for (k = 0; k < s->frames; ++k) {
        ac_istft_push_planar(s->istft, s->enh_re + (size_t)k * bins, s->enh_im + (size_t)k * bins, s->hop_out);
        for (i = 0; i < hop; ++i) pcm_out[(size_t)k * (size_t)hop + (size_t)i] = stream_saturate_s16(s->hop_out[i]);
    }
    return 1;
}
