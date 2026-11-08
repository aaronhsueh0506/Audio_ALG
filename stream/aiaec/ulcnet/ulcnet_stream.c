/* Align-ULCNet PCM stream: everything (no file-loader TU for this model). No <stdio.h>, no
 * malloc/calloc/posix_memalign anywhere in this TU -- it links into a NO_STDIO target. See
 * ulcnet_stream.h for the public contract and ulcnet_stream_internal.h for the pool layout this
 * file carves and constructs. */

#include <string.h>

#include "ulcnet_stream_internal.h"
#include "utility/mem_align.h"
#include "stream_mem.h"

#define ULCNET_STREAM_ALIGN ULCNET_STREAM_ALIGNMENT

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(ULCNET_STREAM_ALIGN == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* -1: inconsistent (some but not all of the four pointer/count pairs set, or a single pair with a
 * non-NULL pointer and a zero count or vice versa). 0: none registered (the accelerator owns its
 * own state copy). 1: all four registered. */
static int state_mode(const UlcnetStreamConfig *cfg) {
    int key_set = cfg->key_history != NULL && cfg->key_history_elements != 0;
    int key_unset = cfg->key_history == NULL && cfg->key_history_elements == 0;
    int value_set = cfg->value_history != NULL && cfg->value_history_elements != 0;
    int value_unset = cfg->value_history == NULL && cfg->value_history_elements == 0;
    int logit_set = cfg->logit_history != NULL && cfg->logit_history_elements != 0;
    int logit_unset = cfg->logit_history == NULL && cfg->logit_history_elements == 0;
    int gru_set = cfg->h_gru != NULL && cfg->gru_hidden_elements != 0;
    int gru_unset = cfg->h_gru == NULL && cfg->gru_hidden_elements == 0;
    int any_set, any_unset;

    if (!key_set && !key_unset) return -1;
    if (!value_set && !value_unset) return -1;
    if (!logit_set && !logit_unset) return -1;
    if (!gru_set && !gru_unset) return -1;

    any_set = key_set || value_set || logit_set || gru_set;
    any_unset = key_unset || value_unset || logit_unset || gru_unset;
    if (any_set && any_unset) return -1;   /* mixed across the four pairs */
    return any_set ? 1 : 0;
}

static int cfg_valid(const UlcnetStreamConfig *cfg) {
    if (!cfg) return 0;
    if (ulcnet_grid_validate(&cfg->grid) != ULCNET_GRID_OK) return 0;
    if (state_mode(cfg) < 0) return 0;
    return 1;
}

int ulcnet_stream_layout(const UlcnetStreamConfig *cfg, unsigned char *base,
                         UlcnetStreamLayout *out, size_t *total) {
    size_t cursor = 0;
    void *ptr;
    AcStftConfig scfg;
    UlcnetPrepostConfig pcfg;
    UlcnetPrepostMemReq pp_req;
    size_t fft_bytes, stft_bytes, istft_bytes, scratch_bytes;
    size_t hop, n_bins;

    if (!cfg || !out || !total || !cfg_valid(cfg)) return -1;
    memset(out, 0, sizeof(*out));
    hop = (size_t)cfg->grid.hop;
    n_bins = (size_t)cfg->grid.n_bins;

    if (stream_carve(base, &cursor, sizeof(UlcnetStream), &ptr) != 0) return -1;
    out->self = (UlcnetStream *)ptr;

    fft_bytes = fft_get_mem_size(cfg->grid.n_fft);
    if (fft_bytes == 0 || stream_carve(base, &cursor, fft_bytes, &ptr) != 0) return -1;
    out->fft_mem = ptr;

    /* One AcStftConfig, reused for sizing stft_err/stft_far/istft; scfg.fft stays NULL here
     * (own-carve sizing) for the same reason the other stream modules' layout functions do --
     * audio_stft.c's config_valid() dereferences a non-NULL cfg->fft, so sizing cannot borrow the
     * not-yet-constructed shared handle. */
    ac_stft_config_default(&scfg, cfg->grid.n_fft);
    scfg.ola = AC_OLA_ENV_NORM;

    stft_bytes = ac_stft_get_mem_size(&scfg);
    if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
    out->stft_err_mem = ptr;

    if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
    out->stft_far_mem = ptr;

    istft_bytes = ac_istft_get_mem_size(&scfg);
    if (istft_bytes == 0 || stream_carve(base, &cursor, istft_bytes, &ptr) != 0) return -1;
    out->istft_mem = ptr;

    if (ulcnet_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return -1;
    pcfg.key_history = cfg->key_history;
    pcfg.key_history_elements = cfg->key_history_elements;
    pcfg.value_history = cfg->value_history;
    pcfg.value_history_elements = cfg->value_history_elements;
    pcfg.logit_history = cfg->logit_history;
    pcfg.logit_history_elements = cfg->logit_history_elements;
    pcfg.h_gru = cfg->h_gru;
    pcfg.gru_hidden_elements = cfg->gru_hidden_elements;
    if (ulcnet_prepost_get_mem_size(&pcfg, &pp_req) != 0) return -1;
    if (stream_carve(base, &cursor, (size_t)pp_req.bytes, &ptr) != 0) return -1;
    out->prepost_mem = ptr;
    out->prepost_bytes = (size_t)pp_req.bytes;
    out->prepost_build_hash = pp_req.build_flags_hash;

    /* err_f, far_f, hop_out [hop] + err_re, err_im, far_re, far_im, enh_re, enh_im [n_bins], one
     * slab (the other stream modules' "one aligned slab" pattern); only the slab base is part of
     * the pool ABI. */
    scratch_bytes = ck_add_size(
        ck_mul_size(3u, ck_mul_size(hop, sizeof(float))),
        ck_mul_size(6u, ck_mul_size(n_bins, sizeof(float))));
    if (MEM_SIZE_INVALID(scratch_bytes) || stream_carve(base, &cursor, scratch_bytes, &ptr) != 0)
        return -1;
    out->scratch_mem = ptr;

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

static uint32_t build_hash(const UlcnetStreamConfig *cfg, uint32_t prepost_hash) {
    uint32_t h = 2166136261u;
    const UlcnetGrid *g = &cfg->grid;
    h = stream_fnv1a_u32(h, (uint32_t)g->sample_rate);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_fft);
    h = stream_fnv1a_u32(h, (uint32_t)g->hop);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_bins);
    h = stream_fnv1a_u32(h, (uint32_t)g->delay_depth);
    h = stream_fnv1a_u32(h, (uint32_t)ULCNET_STREAM_LAYOUT_VERSION);
    h = stream_fnv1a_u32(h, (uint32_t)(state_mode(cfg) > 0 ? 1u : 0u));
    h = stream_fnv1a_u32(h, prepost_hash);
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

int ulcnet_stream_descriptor_matches(const UlcnetStreamMemReq *expected,
                                     const UlcnetStreamMemReq *req) {
    return expected->descriptor_version == req->descriptor_version &&
           expected->layout_version == req->layout_version &&
           expected->reserved0 == req->reserved0 &&
           expected->build_flags_hash == req->build_flags_hash &&
           expected->alignment == req->alignment &&
           expected->reserved == req->reserved &&
           expected->bytes == req->bytes;
}

UlcnetStream *ulcnet_stream_finish_init(const UlcnetStreamConfig *cfg, unsigned char *base,
                                        const UlcnetStreamLayout *layout) {
    UlcnetStream *s;
    AcStftConfig scfg;
    UlcnetPrepostConfig pcfg;
    size_t hop, n_bins;

    if (!cfg || !base || !layout || !layout->self) return NULL;
    s = layout->self;

    s->grid = cfg->grid;
    s->hop_state = ULCNET_STREAM_IDLE;
    s->had_hop = 0;
    s->prev_solid = 0;
    s->reprime_remaining = 0;

    s->fft = fft_init(layout->fft_mem, fft_get_mem_size(cfg->grid.n_fft), cfg->grid.n_fft);
    if (!s->fft) return NULL;

    /* Real, now-constructed handle, borrowed by all three framing objects -- a *different*,
     * smaller `scfg` than ulcnet_stream_layout()'s own sizing config (which used fft = NULL so it
     * could run before this handle existed); see the comment there. */
    ac_stft_config_default(&scfg, cfg->grid.n_fft);
    scfg.ola = AC_OLA_ENV_NORM;
    scfg.fft = s->fft;

    s->stft_err = ac_stft_init(layout->stft_err_mem, ac_stft_get_mem_size(&scfg), &scfg);
    if (!s->stft_err) return NULL;

    s->stft_far = ac_stft_init(layout->stft_far_mem, ac_stft_get_mem_size(&scfg), &scfg);
    if (!s->stft_far) return NULL;

    s->istft = ac_istft_init(layout->istft_mem, ac_istft_get_mem_size(&scfg), &scfg);
    if (!s->istft) return NULL;

    if (ulcnet_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return NULL;
    pcfg.key_history = cfg->key_history;
    pcfg.key_history_elements = cfg->key_history_elements;
    pcfg.value_history = cfg->value_history;
    pcfg.value_history_elements = cfg->value_history_elements;
    pcfg.logit_history = cfg->logit_history;
    pcfg.logit_history_elements = cfg->logit_history_elements;
    pcfg.h_gru = cfg->h_gru;
    pcfg.gru_hidden_elements = cfg->gru_hidden_elements;
    s->prepost = ulcnet_prepost_init(layout->prepost_mem, layout->prepost_bytes, &pcfg);
    if (!s->prepost) return NULL;

    hop = (size_t)cfg->grid.hop;
    n_bins = (size_t)cfg->grid.n_bins;
    s->err_f = (float *)layout->scratch_mem;
    s->far_f = s->err_f + hop;
    s->hop_out = s->far_f + hop;
    s->err_re = s->hop_out + hop;
    s->err_im = s->err_re + n_bins;
    s->far_re = s->err_im + n_bins;
    s->far_im = s->far_re + n_bins;
    s->enh_re = s->far_im + n_bins;
    s->enh_im = s->enh_re + n_bins;

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
    ac_stft_reset(s->stft_err);
    ac_stft_reset(s->stft_far);
    ac_istft_reset(s->istft);
    ulcnet_prepost_reset(s->prepost);
    s->hop_state = ULCNET_STREAM_IDLE;
    s->reprime_remaining = 0;
    /* Also clear the alignment-edge tracker, so the hop right after any reset (an explicit
     * caller reset, or this same function called internally from a pre_process-detected edge) is
     * exactly like the first hop after init: had_hop == 0 suppresses the next edge check
     * regardless of whatever delay.solid value happened to be recorded before this reset. Without
     * this, an explicit mid-stream ulcnet_stream_reset() call left prev_solid stale, which could
     * spuriously re-trigger a second reset+reprime on the very next hop. */
    s->had_hop = 0;
    s->prev_solid = 0;
}

uint32_t ulcnet_stream_state_generation(const UlcnetStream *s) {
    return s ? ulcnet_prepost_state_generation(s->prepost) : 0u;
}

const UlcnetGrid *ulcnet_stream_grid(const UlcnetStream *s) {
    return s ? &s->grid : NULL;
}

int ulcnet_stream_hop_size(const UlcnetStream *s) {
    return s ? s->grid.hop : -1;
}

int ulcnet_stream_layout_version(const UlcnetStream *s) {
    const UlcnetModelIoDescriptor *d;
    if (!s) return -1;
    d = ulcnet_prepost_descriptor(s->prepost);
    return d ? (int)d->layout_version : -1;
}

int ulcnet_stream_pre_process_f32(UlcnetStream *s, const float *err_in, const float *far_in,
                                  UlcnetStreamDelay delay, UlcnetHopIO *io) {
    int edge;

    if (!s || !err_in || !far_in || !io || s->hop_state != ULCNET_STREAM_IDLE) return -1;

    /* Alignment-edge detection, before this hop's own analysis (mirrors the board sample's Stage
     * 3, run ahead of Stage 4): a new alignment generation, or the first hop where `solid` turns
     * true after a prior hop where it was false. Only the solid-transition term is gated by
     * s->had_hop (no prior solid value exists on the very first hop to compare against, matching
     * the board sample's own g_processed_hops guard on that specific sub-check); delay.changed is
     * NOT gated by had_hop, matching the board sample's own unguarded `changed` check -- a caller
     * reporting a fresh alignment generation on hop 0 correctly resets there too. */
    edge = delay.changed || (s->had_hop && !s->prev_solid && delay.solid);
    if (edge) {
        /* ulcnet_stream_reset() itself clears reprime_remaining; re-arm strictly after it
         * returns, or the arm below would be immediately undone. */
        ulcnet_stream_reset(s);
        s->reprime_remaining = ULCNET_STREAM_REPRIME_FRAMES;
    }
    s->had_hop = 1;
    s->prev_solid = delay.solid;

    /* err_in/far_in are read once each, here, and never stored -- no reason to copy into the
     * per-hop scratch slab first (that scratch exists only for ulcnet_stream_pre_process()'s own
     * int16 conversion below). */
    ac_stft_push_planar(s->stft_err, err_in, s->err_re, s->err_im);
    ac_stft_push_planar(s->stft_far, far_in, s->far_re, s->far_im);

    /* pre_process_freq() always returns 1 here (ulcnet_prepost.h: "Returns 1 (one invocation)");
     * any other return is a contract error this stream does not expect in correct use. */
    if (ulcnet_prepost_pre_process_freq(s->prepost, s->err_re, s->err_im,
                                       s->far_re, s->far_im) != 1)
        return -1;

    if (s->reprime_remaining > 0) {
        s->reprime_remaining--;
        /* Identity for this frame: frame_bypass() needs only pre_process_freq()'s own
         * p->frame_open flag (set above), not the separate frame_open()/"prepared" state --
         * confirmed against lib/aiaec/ulcnet/ulcnet_prepost.c: frame_bypass() checks
         * `!p->frame_open` only, and pre_process_freq() already set that flag. It returns 0 here
         * (the board sample ignores its return value on this exact path; this stream does not
         * need to check it either, since pre_process_freq() having just succeeded guarantees
         * p->frame_open is set and frame_bypass() cannot be refused). */
        (void)ulcnet_prepost_frame_bypass(s->prepost);
        memset(io, 0, sizeof(*io));
        s->hop_state = ULCNET_STREAM_FRAME_CLOSED;
        return 0;
    }

    if (ulcnet_prepost_frame_open(s->prepost, &io->inputs, &io->outputs) != 0) return -1;

    io->state_generation = ulcnet_prepost_state_generation(s->prepost);
    s->hop_state = ULCNET_STREAM_FRAME_OPEN;
    return 1;
}

int ulcnet_stream_pre_process(UlcnetStream *s, const int16_t *err_in, const int16_t *far_in,
                              UlcnetStreamDelay delay, UlcnetHopIO *io) {
    int hop, i;

    /* Thin int16 convenience wrapper: converts into the per-hop float scratch (never read by
     * anyone else), then runs the exact same path ulcnet_stream_pre_process_f32() runs for a
     * caller whose own lanes are already float (e.g. a linear AEC's formed_hop residual, which
     * the board sample feeds this model WITHOUT ever quantizing to int16 -- see ulcnet_stream.h's
     * own comment). The full contract check is repeated here (not just deferred to the f32 call)
     * so this wrapper never touches s->err_f/far_f on a call that was going to be refused anyway
     * -- same "no state touched before any write" shape the rest of this file keeps elsewhere. */
    if (!s || !err_in || !far_in || !io || s->hop_state != ULCNET_STREAM_IDLE) return -1;
    hop = s->grid.hop;
    for (i = 0; i < hop; ++i) {
        s->err_f[i] = (float)err_in[i] / 32768.0f;
        s->far_f[i] = (float)far_in[i] / 32768.0f;
    }
    return ulcnet_stream_pre_process_f32(s, s->err_f, s->far_f, delay, io);
}

int ulcnet_stream_post_process(UlcnetStream *s, int ai_rc, int16_t *pcm_out) {
    int hop, i, prc, written;

    if (!s || !pcm_out || s->hop_state == ULCNET_STREAM_IDLE) return -1;
    hop = s->grid.hop;

    if (s->hop_state == ULCNET_STREAM_FRAME_OPEN) {
        /* Mirror the board sample's dispatch exactly, and DFN2/DeepVQE-S's own fix for the same
         * hazard: a finish that itself fails (non-finite mask or state, despite ai_rc == 0) still
         * needs a bypass to actually close the frame -- ulcnet_prepost_frame_finish()'s doc says
         * "the frame stays open ... then frame_bypass() or a fresh frame_open()", and a frame left
         * open here would desync this stream's own bookkeeping from UlcnetPrepost's. */
        int rc = (ai_rc == 0) ? ulcnet_prepost_frame_finish(s->prepost) : -1;
        if (rc != 0) ulcnet_prepost_frame_bypass(s->prepost);
    }
    /* ULCNET_STREAM_FRAME_CLOSED: already bypassed inside pre_process; ai_rc is not read. */
    s->hop_state = ULCNET_STREAM_IDLE;

    prc = ulcnet_prepost_post_process_freq(s->prepost, s->enh_re, s->enh_im);
    if (prc != 0) {
        /* Defensive only, unreached in correct use (the hop was just closed above): zero the hop,
         * leave the synthesis untouched, same contract-error treatment the other stream modules
         * use. */
        memset(pcm_out, 0, (size_t)hop * sizeof(int16_t));
        return -1;
    }

    written = ac_istft_push_planar(s->istft, s->enh_re, s->enh_im, s->hop_out);
    if (written != hop) {
        /* AC_OLA_ENV_NORM's documented startup behavior: the very first push after
         * init/reset/reprime-triggered-reset emits nothing. Mirrors the board sample's own
         * `if (emitted != HOP) memset(...)`. */
        memset(pcm_out, 0, (size_t)hop * sizeof(int16_t));
        return 0;
    }

    for (i = 0; i < hop; ++i) pcm_out[i] = stream_saturate_s16(s->hop_out[i]);
    return 1;
}
