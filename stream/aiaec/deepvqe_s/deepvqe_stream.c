/* DeepVQE-S PCM stream: everything (no file-loader TU for this model -- there are no ERB
 * matrices to load). No <stdio.h>, no malloc/calloc/posix_memalign anywhere in this TU -- it
 * links into a NO_STDIO target. See deepvqe_stream.h for the public contract and
 * deepvqe_stream_internal.h for the pool layout this file carves and constructs. */

#include <string.h>

#include "deepvqe_stream_internal.h"
#include "utility/mem_align.h"
#include "stream_mem.h"

#define DEEPVQE_STREAM_ALIGN DEEPVQE_STREAM_ALIGNMENT

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DEEPVQE_STREAM_ALIGN == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* -1: inconsistent (some but not all of cfg->state[]/state_elements[] set, or a single entry with
 * a non-NULL pointer and a zero count or vice versa). 0: none registered (the accelerator owns
 * its own state copy). 1: all sixteen registered. */
static int state_mode(const DeepVqeStreamConfig *cfg) {
    int i, any_set = 0, any_unset = 0;
    for (i = 0; i < DEEPVQE_STATE_COUNT; ++i) {
        int set = cfg->state[i] != NULL && cfg->state_elements[i] != 0;
        int unset = cfg->state[i] == NULL && cfg->state_elements[i] == 0;
        if (!set && !unset) return -1;   /* pointer/count mismatch on this one entry */
        if (set) any_set = 1;
        if (unset) any_unset = 1;
    }
    if (any_set && any_unset) return -1;   /* mixed across entries */
    return any_set ? 1 : 0;
}

static int cfg_valid(const DeepVqeStreamConfig *cfg) {
    if (!cfg) return 0;
    if (deepvqe_grid_validate(&cfg->grid) != DEEPVQE_GRID_OK) return 0;
    if (state_mode(cfg) < 0) return 0;
    return 1;
}

int deepvqe_stream_layout(const DeepVqeStreamConfig *cfg, unsigned char *base,
                          DeepVqeStreamLayout *out, size_t *total) {
    size_t cursor = 0;
    void *ptr;
    AcStftConfig scfg;
    DeepVqePrepostConfig pcfg;
    DeepVqePrepostMemReq pp_req;
    size_t fft_bytes, stft_bytes, istft_bytes, scratch_bytes;
    size_t hop, n_bins;
    int i;

    if (!cfg || !out || !total || !cfg_valid(cfg)) return -1;
    memset(out, 0, sizeof(*out));
    hop = (size_t)cfg->grid.hop;
    n_bins = (size_t)cfg->grid.n_bins;

    if (stream_carve(base, &cursor, sizeof(DeepVqeStream), &ptr) != 0) return -1;
    out->self = (DeepVqeStream *)ptr;

    fft_bytes = fft_get_mem_size(cfg->grid.n_fft);
    if (fft_bytes == 0 || stream_carve(base, &cursor, fft_bytes, &ptr) != 0) return -1;
    out->fft_mem = ptr;

    /* One AcStftConfig, reused for sizing stft_mic/stft_far/istft exactly as deepvqe_prepost.h's
     * framing doc describes ("all from ac_stft_config_default(&sc, n_fft) ... with sc.ola =
     * AC_OLA_ENV_NORM (analysis ignores ola)"); scfg.fft stays NULL here (own-carve sizing) for
     * the same reason dfn2_stream_layout()'s does -- audio_stft.c's config_valid() dereferences a
     * non-NULL cfg->fft, so sizing cannot borrow the not-yet-constructed shared handle. */
    ac_stft_config_default(&scfg, cfg->grid.n_fft);
    scfg.ola = AC_OLA_ENV_NORM;

    stft_bytes = ac_stft_get_mem_size(&scfg);
    if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
    out->stft_mic_mem = ptr;

    if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
    out->stft_far_mem = ptr;

    istft_bytes = ac_istft_get_mem_size(&scfg);
    if (istft_bytes == 0 || stream_carve(base, &cursor, istft_bytes, &ptr) != 0) return -1;
    out->istft_mem = ptr;

    if (deepvqe_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return -1;
    for (i = 0; i < DEEPVQE_STATE_COUNT; ++i) {
        pcfg.state[i] = cfg->state[i];
        pcfg.state_elements[i] = cfg->state_elements[i];
    }
    if (deepvqe_prepost_get_mem_size(&pcfg, &pp_req) != 0) return -1;
    if (stream_carve(base, &cursor, (size_t)pp_req.bytes, &ptr) != 0) return -1;
    out->prepost_mem = ptr;
    out->prepost_bytes = (size_t)pp_req.bytes;
    out->prepost_build_hash = pp_req.build_flags_hash;

    /* mic_f, far_f, hop_out [hop] + mic_re, mic_im, far_re, far_im, enh_re, enh_im [n_bins], one
     * slab (dfn2_stream_layout()'s "one aligned slab" pattern); only the slab base is part of the
     * pool ABI. */
    scratch_bytes = ck_add_size(
        ck_mul_size(3u, ck_mul_size(hop, sizeof(float))),
        ck_mul_size(6u, ck_mul_size(n_bins, sizeof(float))));
    if (MEM_SIZE_INVALID(scratch_bytes) || stream_carve(base, &cursor, scratch_bytes, &ptr) != 0)
        return -1;
    out->scratch_mem = ptr;

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

static uint32_t build_hash(const DeepVqeStreamConfig *cfg, uint32_t prepost_hash) {
    uint32_t h = 2166136261u;
    const DeepVqeGrid *g = &cfg->grid;
    h = stream_fnv1a_u32(h, (uint32_t)g->sample_rate);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_fft);
    h = stream_fnv1a_u32(h, (uint32_t)g->hop);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_bins);
    h = stream_fnv1a_u32(h, (uint32_t)g->delay_depth);
    h = stream_fnv1a_u32(h, (uint32_t)DEEPVQE_STREAM_LAYOUT_VERSION);
    h = stream_fnv1a_u32(h, (uint32_t)(state_mode(cfg) > 0 ? 1u : 0u));
    h = stream_fnv1a_u32(h, prepost_hash);
    return ac_memreq_abi_hash(h);
}

int deepvqe_stream_get_mem_size(const DeepVqeStreamConfig *cfg, DeepVqeStreamMemReq *req) {
    DeepVqeStreamLayout layout;
    size_t total = 0;

    if (!req || !cfg_valid(cfg)) return -1;
    if (deepvqe_stream_layout(cfg, NULL, &layout, &total) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DEEPVQE_STREAM_DESCRIPTOR_VERSION;
    req->layout_version = DEEPVQE_STREAM_LAYOUT_VERSION;
    req->build_flags_hash = build_hash(cfg, layout.prepost_build_hash);
    req->alignment = DEEPVQE_STREAM_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

int deepvqe_stream_descriptor_matches(const DeepVqeStreamMemReq *expected,
                                      const DeepVqeStreamMemReq *req) {
    return expected->descriptor_version == req->descriptor_version &&
           expected->layout_version == req->layout_version &&
           expected->reserved0 == req->reserved0 &&
           expected->build_flags_hash == req->build_flags_hash &&
           expected->alignment == req->alignment &&
           expected->reserved == req->reserved &&
           expected->bytes == req->bytes;
}

DeepVqeStream *deepvqe_stream_finish_init(const DeepVqeStreamConfig *cfg, unsigned char *base,
                                          const DeepVqeStreamLayout *layout) {
    DeepVqeStream *s;
    AcStftConfig scfg;
    DeepVqePrepostConfig pcfg;
    size_t hop, n_bins;
    int i;

    if (!cfg || !base || !layout || !layout->self) return NULL;
    s = layout->self;

    s->grid = cfg->grid;
    s->open = 0;

    s->fft = fft_init(layout->fft_mem, fft_get_mem_size(cfg->grid.n_fft), cfg->grid.n_fft);
    if (!s->fft) return NULL;

    /* Real, now-constructed handle, borrowed by all three framing objects -- a *different*,
     * smaller `scfg` than deepvqe_stream_layout()'s own sizing config (which used fft = NULL so
     * it could run before this handle existed); see the comment there. */
    ac_stft_config_default(&scfg, cfg->grid.n_fft);
    scfg.ola = AC_OLA_ENV_NORM;
    scfg.fft = s->fft;

    s->stft_mic = ac_stft_init(layout->stft_mic_mem, ac_stft_get_mem_size(&scfg), &scfg);
    if (!s->stft_mic) return NULL;

    s->stft_far = ac_stft_init(layout->stft_far_mem, ac_stft_get_mem_size(&scfg), &scfg);
    if (!s->stft_far) return NULL;

    s->istft = ac_istft_init(layout->istft_mem, ac_istft_get_mem_size(&scfg), &scfg);
    if (!s->istft) return NULL;

    if (deepvqe_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return NULL;
    for (i = 0; i < DEEPVQE_STATE_COUNT; ++i) {
        pcfg.state[i] = cfg->state[i];
        pcfg.state_elements[i] = cfg->state_elements[i];
    }
    s->prepost = deepvqe_prepost_init(layout->prepost_mem, layout->prepost_bytes, &pcfg);
    if (!s->prepost) return NULL;

    hop = (size_t)cfg->grid.hop;
    n_bins = (size_t)cfg->grid.n_bins;
    s->mic_f = (float *)layout->scratch_mem;
    s->far_f = s->mic_f + hop;
    s->hop_out = s->far_f + hop;
    s->mic_re = s->hop_out + hop;
    s->mic_im = s->mic_re + n_bins;
    s->far_re = s->mic_im + n_bins;
    s->far_im = s->far_re + n_bins;
    s->enh_re = s->far_im + n_bins;
    s->enh_im = s->enh_re + n_bins;

    return s;
}

DeepVqeStream *deepvqe_stream_init_ex(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes,
                                      const DeepVqeStreamMemReq *expected) {
    DeepVqeStreamMemReq req;
    DeepVqeStreamLayout layout;
    size_t total = 0;

    if (!pool || !cfg_valid(cfg)) return NULL;
    if (deepvqe_stream_get_mem_size(cfg, &req) != 0) return NULL;
    if (pool_bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected && !deepvqe_stream_descriptor_matches(expected, &req)) return NULL;

    memset(pool, 0, (size_t)req.bytes);
    if (deepvqe_stream_layout(cfg, (unsigned char *)pool, &layout, &total) != 0) return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    return deepvqe_stream_finish_init(cfg, (unsigned char *)pool, &layout);
}

DeepVqeStream *deepvqe_stream_init(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes) {
    return deepvqe_stream_init_ex(cfg, pool, pool_bytes, NULL);
}

void deepvqe_stream_reset(DeepVqeStream *s) {
    if (!s) return;
    ac_stft_reset(s->stft_mic);
    ac_stft_reset(s->stft_far);
    ac_istft_reset(s->istft);
    deepvqe_prepost_reset(s->prepost);
    s->open = 0;
}

uint32_t deepvqe_stream_state_generation(const DeepVqeStream *s) {
    return s ? deepvqe_prepost_state_generation(s->prepost) : 0u;
}

const DeepVqeGrid *deepvqe_stream_grid(const DeepVqeStream *s) {
    return s ? &s->grid : NULL;
}

int deepvqe_stream_hop_size(const DeepVqeStream *s) {
    return s ? s->grid.hop : -1;
}

int deepvqe_stream_layout_version(const DeepVqeStream *s) {
    const DeepVqePrepostDescriptor *d;
    if (!s) return -1;
    d = deepvqe_prepost_descriptor(s->prepost);
    return d ? (int)d->layout_version : -1;
}

int deepvqe_stream_pre_process(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in,
                               DeepVqeHopIO *io) {
    int hop, i;

    if (!s || !mic_in || !far_in || !io || s->open) return -1;
    hop = s->grid.hop;

    for (i = 0; i < hop; ++i) {
        s->mic_f[i] = (float)mic_in[i] / 32768.0f;
        s->far_f[i] = (float)far_in[i] / 32768.0f;
    }

    ac_stft_push_planar(s->stft_mic, s->mic_f, s->mic_re, s->mic_im);
    ac_stft_push_planar(s->stft_far, s->far_f, s->far_re, s->far_im);

    /* pre_process_freq() always returns 1 here (deepvqe_prepost.h: "one frame per hop from the
     * first hop, no latency beyond it"); any other return is a contract error this stream does
     * not expect in correct use. */
    if (deepvqe_prepost_pre_process_freq(s->prepost, s->mic_re, s->mic_im,
                                        s->far_re, s->far_im) != 1)
        return -1;
    if (deepvqe_prepost_frame_open(s->prepost, &io->inputs, &io->outputs) != 0) return -1;

    io->state_generation = deepvqe_prepost_state_generation(s->prepost);
    s->open = 1;
    return 1;
}

int deepvqe_stream_post_process(DeepVqeStream *s, int ai_rc, int16_t *pcm_out) {
    int hop, i, prc, rc, written;

    if (!s || !pcm_out || !s->open) return -1;
    hop = s->grid.hop;

    /* Mirror the board sample's dispatch exactly, and DFN2's own fix for the same hazard: a
     * finish that itself fails (non-finite tap or state, despite ai_rc == 0) still needs a bypass
     * to actually close the frame -- deepvqe_prepost_frame_finish() leaves it open on failure
     * (deepvqe_prepost.h: "the frame stays open ... call frame_bypass() or re-run"), and a frame
     * left open here would desync this stream's own `open` bookkeeping from DeepVqePrepost's. */
    rc = (ai_rc == 0) ? deepvqe_prepost_frame_finish(s->prepost) : -1;
    if (rc != 0) deepvqe_prepost_frame_bypass(s->prepost);
    s->open = 0;

    prc = deepvqe_prepost_post_process_freq(s->prepost, s->enh_re, s->enh_im);
    if (prc != 0) {
        /* Defensive only, unreached in correct use (the hop was just closed above): zero the hop,
         * leave the synthesis untouched, same contract-error treatment DFN2 uses. */
        memset(pcm_out, 0, (size_t)hop * sizeof(int16_t));
        return -1;
    }

    written = ac_istft_push_planar(s->istft, s->enh_re, s->enh_im, s->hop_out);
    if (written != hop) {
        /* AC_OLA_ENV_NORM's documented startup behavior: the very first push after init/reset
         * emits nothing. Mirrors the board sample's own `if (written != HOP_SIZE) memset(...)`. */
        memset(pcm_out, 0, (size_t)hop * sizeof(int16_t));
        return 0;
    }

    for (i = 0; i < hop; ++i) pcm_out[i] = stream_saturate_s16(s->hop_out[i]);
    return 1;
}
