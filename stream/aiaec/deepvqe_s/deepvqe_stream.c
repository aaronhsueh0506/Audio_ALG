/* DeepVQE-S PCM stream: everything (no file-loader TU for this model -- there are no ERB
 * matrices to load). No <stdio.h>, no malloc/calloc/posix_memalign anywhere in this TU -- it
 * links into a NO_STDIO target. See deepvqe_stream.h for the public contract. */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "deepvqe_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "stream_mem.h"
#include "stream_range.h"
#include "stream_run.h"

struct DeepVqeStream {
    DeepVqeGrid grid;

    /* fft, stft_mic, stft_far, istft and the scratch: the PCM half, NULL in a freq_only stream. */
    FftHandle *fft;     /* shared by stft_mic / stft_far / istft */
    AcStft    *stft_mic;
    AcStft    *stft_far;
    AcIstft   *istft;
    DeepVqePrepost *prepost;

    /* Raw mic spectra of the open call [N * n_bins], kept from pre to post for the CCM (both families). */
    float *mic_re;
    float *mic_im;

    /* PCM scratch, one slab: mic_f / far_f / hop_out [hop], far_re / far_im [n_bins], enh_re / enh_im [N * n_bins]. */
    float *mic_f;
    float *far_f;
    float *hop_out;
    float *far_re;
    float *far_im;
    float *enh_re;
    float *enh_im;

    int pending; /* 1: a pre_process succeeded, its call is open and the post_process is due */
};

typedef struct DeepVqeStreamLayout {
    DeepVqeStream *self;
    void *fft_mem;
    void *stft_mic_mem;
    void *stft_far_mem;
    void *istft_mem;
    void *prepost_mem;
    size_t prepost_bytes;
    void *stage_mem;     /* the raw mic spectra of one call */
    void *scratch_mem;   /* the PCM scratch slab */
    uint32_t prepost_build_hash;   /* DeepVqePrepostMemReq.build_flags_hash, folded into this stream's hash */
} DeepVqeStreamLayout;

#define DEEPVQE_STREAM_ALIGN DEEPVQE_STREAM_ALIGNMENT

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

/* One carve walk: sizing (base == NULL, *out pointers stay NULL) or carving; *total must match across both with one cfg.
 * -1 on a bad config or an overflowing carve. Order (bump LAYOUT_VERSION and build_flags_hash on a change): self, fft,
 * stft_mic, stft_far, istft, prepost, stage, scratch; cfg.freq_only: self, prepost, stage. */
static int deepvqe_stream_layout(const DeepVqeStreamConfig *cfg, unsigned char *base,
                                 DeepVqeStreamLayout *out, size_t *total) {
    size_t cursor = 0;
    void *ptr;
    AcStftConfig scfg;
    DeepVqePrepostConfig pcfg;
    DeepVqePrepostMemReq pp_req;
    size_t fft_bytes, stft_bytes, istft_bytes, stage_bytes, scratch_bytes;
    size_t hop, n_bins, plane;
    int i;

    if (!cfg || !out || !total || !cfg_valid(cfg)) return -1;
    memset(out, 0, sizeof(*out));
    hop = (size_t)cfg->grid.hop;
    n_bins = (size_t)cfg->grid.n_bins;
    plane = ck_mul_size((size_t)cfg->grid.frames_per_call, n_bins);

    if (stream_carve(base, &cursor, sizeof(DeepVqeStream), &ptr) != 0) return -1;
    out->self = (DeepVqeStream *)ptr;

    if (!cfg->freq_only) {
        fft_bytes = fft_get_mem_size(cfg->grid.n_fft);
        if (fft_bytes == 0 || stream_carve(base, &cursor, fft_bytes, &ptr) != 0) return -1;
        out->fft_mem = ptr;

        /* One AcStftConfig, reused for sizing stft_mic/stft_far/istft (analysis ignores ola).
         * scfg.fft stays NULL here (own-carve sizing; the shared handle doesn't exist yet). ola =
         * AC_OLA_ACCUMULATE, this stream's real-time contract for every grid it serves. */
        ac_stft_config_default(&scfg, cfg->grid.n_fft);
        scfg.ola = AC_OLA_ACCUMULATE;

        stft_bytes = ac_stft_get_mem_size(&scfg);
        if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        out->stft_mic_mem = ptr;

        if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        out->stft_far_mem = ptr;

        istft_bytes = ac_istft_get_mem_size(&scfg);
        if (istft_bytes == 0 || stream_carve(base, &cursor, istft_bytes, &ptr) != 0) return -1;
        out->istft_mem = ptr;
    }

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

    /* mic_re, mic_im [N * n_bins] of the open call. */
    stage_bytes = ck_mul_size(2u, ck_mul_size(plane, sizeof(float)));
    if (MEM_SIZE_INVALID(stage_bytes) || stream_carve(base, &cursor, stage_bytes, &ptr) != 0) return -1;
    out->stage_mem = ptr;

    /* mic_f, far_f, hop_out [hop] + far_re, far_im [n_bins] + enh_re, enh_im [N * n_bins], one slab;
     * only the slab base is part of the pool ABI. */
    if (!cfg->freq_only) {
        scratch_bytes = ck_add_size(
            ck_mul_size(3u, ck_mul_size(hop, sizeof(float))),
            ck_add_size(ck_mul_size(2u, ck_mul_size(n_bins, sizeof(float))),
                        ck_mul_size(2u, ck_mul_size(plane, sizeof(float)))));
        if (MEM_SIZE_INVALID(scratch_bytes) || stream_carve(base, &cursor, scratch_bytes, &ptr) != 0)
            return -1;
        out->scratch_mem = ptr;
    }

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

static uint32_t build_hash(const DeepVqeStreamConfig *cfg, uint32_t prepost_hash) {
    uint32_t h = AC_FNV1A_SEED;
    const DeepVqeGrid *g = &cfg->grid;
    h = stream_fnv1a_u32(h, (uint32_t)g->sample_rate);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_fft);
    h = stream_fnv1a_u32(h, (uint32_t)g->hop);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_bins);
    h = stream_fnv1a_u32(h, (uint32_t)g->delay_depth);
    h = stream_fnv1a_u32(h, (uint32_t)DEEPVQE_STREAM_LAYOUT_VERSION);
    h = stream_fnv1a_u32(h, (uint32_t)(state_mode(cfg) > 0 ? 1u : 0u));
    h = stream_fnv1a_u32(h, cfg->freq_only ? 1u : 0u);
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

/* Field-by-field memory-requirement comparison for the stale-pool gate. */
static int deepvqe_stream_descriptor_matches(const DeepVqeStreamMemReq *expected,
                                             const DeepVqeStreamMemReq *req) {
    return expected->descriptor_version == req->descriptor_version &&
           expected->layout_version == req->layout_version &&
           expected->reserved0 == req->reserved0 &&
           expected->build_flags_hash == req->build_flags_hash &&
           expected->alignment == req->alignment &&
           expected->reserved == req->reserved &&
           expected->bytes == req->bytes;
}

/* Constructs every carved object; NULL leaves no valid instance. Called after deepvqe_stream_layout() and the caller's
 * size / alignment / stale-pool checks. */
static DeepVqeStream *deepvqe_stream_finish_init(const DeepVqeStreamConfig *cfg, unsigned char *base,
                                                 const DeepVqeStreamLayout *layout) {
    DeepVqeStream *s;
    AcStftConfig scfg;
    DeepVqePrepostConfig pcfg;
    size_t hop, n_bins, plane;
    int i;

    if (!cfg || !base || !layout || !layout->self) return NULL;
    s = layout->self;

    s->grid = cfg->grid;
    s->pending = 0;

    s->fft = NULL;
    s->stft_mic = s->stft_far = NULL;
    s->istft = NULL;

    if (!cfg->freq_only) {
        s->fft = fft_init(layout->fft_mem, fft_get_mem_size(cfg->grid.n_fft), cfg->grid.n_fft);
        if (!s->fft) return NULL;

        /* Real, now-constructed handle, borrowed by all three framing objects -- a smaller scfg
         * than the sizing one above, which the larger carved region always fits. */
        ac_stft_config_default(&scfg, cfg->grid.n_fft);
        scfg.ola = AC_OLA_ACCUMULATE;
        scfg.fft = s->fft;

        s->stft_mic = ac_stft_init(layout->stft_mic_mem, ac_stft_get_mem_size(&scfg), &scfg);
        if (!s->stft_mic) return NULL;

        s->stft_far = ac_stft_init(layout->stft_far_mem, ac_stft_get_mem_size(&scfg), &scfg);
        if (!s->stft_far) return NULL;

        s->istft = ac_istft_init(layout->istft_mem, ac_istft_get_mem_size(&scfg), &scfg);
        if (!s->istft) return NULL;
    }

    if (deepvqe_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return NULL;
    for (i = 0; i < DEEPVQE_STATE_COUNT; ++i) {
        pcfg.state[i] = cfg->state[i];
        pcfg.state_elements[i] = cfg->state_elements[i];
    }
    s->prepost = deepvqe_prepost_init(layout->prepost_mem, layout->prepost_bytes, &pcfg);
    if (!s->prepost) return NULL;

    n_bins = (size_t)cfg->grid.n_bins;
    plane = (size_t)cfg->grid.frames_per_call * n_bins;
    s->mic_re = (float *)layout->stage_mem;
    s->mic_im = s->mic_re + plane;
    if (!cfg->freq_only) {
        hop = (size_t)cfg->grid.hop;
        s->mic_f = (float *)layout->scratch_mem;
        s->far_f = s->mic_f + hop;
        s->hop_out = s->far_f + hop;
        s->far_re = s->hop_out + hop;
        s->far_im = s->far_re + n_bins;
        s->enh_re = s->far_im + n_bins;
        s->enh_im = s->enh_re + plane;
    }

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
    if (s->stft_mic) {   /* a freq_only stream has no framing */
        ac_stft_reset(s->stft_mic);
        ac_stft_reset(s->stft_far);
        ac_istft_reset(s->istft);
    }
    deepvqe_prepost_reset(s->prepost);
    s->pending = 0;
}

uint32_t deepvqe_stream_state_generation(const DeepVqeStream *s) {
    return s ? deepvqe_prepost_state_generation(s->prepost) : 0u;
}

int deepvqe_stream_hop_size(const DeepVqeStream *s) {
    return s ? s->grid.hop : -1;
}

/* N frames in, one accelerator job. The PCM entries (mic_in non-NULL) analyse hop by hop, the mic spectrum straight into
 * its slot of the call's staging; the spectrum entries copy the N * n_bins mic planes into it. Each frame goes through
 * pre_frame, then the views are published once. */
static int open_packet(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in,
                       const float *mic_re, const float *mic_im, const float *far_re, const float *far_im,
                       DeepVqeHopIO *io) {
    const int n = s->grid.frames_per_call, hop = s->grid.hop;
    const size_t nb = (size_t)s->grid.n_bins;
    int i, k;

    for (k = 0; k < n; ++k) {
        float *sr = s->mic_re + (size_t)k * nb, *si = s->mic_im + (size_t)k * nb;
        const float *fr, *fi;

        if (mic_in) {
            const size_t off = (size_t)k * (size_t)hop;
            for (i = 0; i < hop; ++i) {
                s->mic_f[i] = (float)mic_in[off + (size_t)i] / 32768.0f;
                s->far_f[i] = (float)far_in[off + (size_t)i] / 32768.0f;
            }
            ac_stft_push_planar(s->stft_mic, s->mic_f, sr, si);
            ac_stft_push_planar(s->stft_far, s->far_f, s->far_re, s->far_im);
            fr = s->far_re; fi = s->far_im;
        } else {
            memcpy(sr, mic_re + (size_t)k * nb, nb * sizeof(float));
            memcpy(si, mic_im + (size_t)k * nb, nb * sizeof(float));
            fr = far_re + (size_t)k * nb; fi = far_im + (size_t)k * nb;
        }
        if (deepvqe_prepost_pre_frame(s->prepost, k, sr, si, fr, fi) != DEEPVQE_PREPOST_OK) return -1;
    }
    if (deepvqe_prepost_views(s->prepost, &io->inputs, &io->outputs) != 0) return -1;
    io->state_generation = deepvqe_prepost_state_generation(s->prepost);
    s->pending = 1;
    return 1;
}

int deepvqe_stream_pre_process_packet(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in,
                                      DeepVqeHopIO *io) {
    if (!s || !mic_in || !far_in || !io || !s->stft_mic || s->pending) return -1;
    return open_packet(s, mic_in, far_in, NULL, NULL, NULL, NULL, io);
}

int deepvqe_stream_pre_process_packet_freq(DeepVqeStream *s, const float *mic_re, const float *mic_im,
                                           const float *far_re, const float *far_im, DeepVqeHopIO *io) {
    size_t plane;
    if (!s || !mic_re || !mic_im || !far_re || !far_im || !io || s->pending) return -1;
    plane = (size_t)s->grid.frames_per_call * (size_t)s->grid.n_bins;
    if (!stream_plane_within(mic_re, plane, DEEPVQE_STREAM_BIN_LIMIT_BITS) ||
        !stream_plane_within(mic_im, plane, DEEPVQE_STREAM_BIN_LIMIT_BITS) ||
        !stream_plane_within(far_re, plane, DEEPVQE_STREAM_BIN_LIMIT_BITS) ||
        !stream_plane_within(far_im, plane, DEEPVQE_STREAM_BIN_LIMIT_BITS))
        return DEEPVQE_STREAM_ERR_RANGE;
    return open_packet(s, NULL, NULL, mic_re, mic_im, far_re, far_im, io);
}

/* Closes the open call into re / im (N * n_bins each): commit on a clean run, else N zero frames. A run that left the
 * device state dirty restarts the state (once); a refused validate_run restarted it already; so does a non-finite
 * enhanced bin, which discards the frames computed before it. */
static void close_packet(DeepVqeStream *s, int ai_rc, float *re, float *im) {
    DeepVqePrepost *p = s->prepost;
    const int n = s->grid.frames_per_call;
    const size_t nb = (size_t)s->grid.n_bins;
    int ok = 0, k;

    s->pending = 0;
    if (ai_rc == STREAM_RUN_OK) {
        if (deepvqe_prepost_validate_run(p) == 0) {
            for (k = 0; k < n; ++k)
                if (deepvqe_prepost_post_frame(p, k, s->mic_re + (size_t)k * nb, s->mic_im + (size_t)k * nb,
                                               re + (size_t)k * nb, im + (size_t)k * nb) != DEEPVQE_PREPOST_OK) break;
            ok = k == n;
            if (!ok) deepvqe_prepost_zero_state(p);
        }
    } else if (ai_rc != STREAM_RUN_NOT_STARTED) {
        deepvqe_prepost_zero_state(p);
    }
    if (!ok) {
        memset(re, 0, (size_t)n * nb * sizeof(float));
        memset(im, 0, (size_t)n * nb * sizeof(float));
    }
}

int deepvqe_stream_post_process_packet_freq(DeepVqeStream *s, int ai_rc, float *re, float *im) {
    if (!s || !re || !im || !s->pending) return -1;
    close_packet(s, ai_rc, re, im);
    return 1;
}

int deepvqe_stream_post_process_packet(DeepVqeStream *s, int ai_rc, int16_t *pcm_out) {
    const int n = s ? s->grid.frames_per_call : 0, hop = s ? s->grid.hop : 0;
    const size_t nb = s ? (size_t)s->grid.n_bins : 0u;
    int i, k;

    if (!s || !pcm_out || !s->istft || !s->pending) return -1;
    close_packet(s, ai_rc, s->enh_re, s->enh_im);   /* all N frames before the first synthesis */
    for (k = 0; k < n; ++k) {
        const int written = ac_istft_push_planar(s->istft, s->enh_re + (size_t)k * nb, s->enh_im + (size_t)k * nb, s->hop_out);

        if (written != hop) {
            /* Defensive only: AC_OLA_ACCUMULATE always returns hop. */
            memset(pcm_out, 0, (size_t)n * (size_t)hop * sizeof(int16_t));
            return -1;
        }
        for (i = 0; i < hop; ++i) pcm_out[(size_t)k * (size_t)hop + (size_t)i] = stream_saturate_s16(s->hop_out[i]);
    }
    return 1;
}

int deepvqe_stream_pre_process(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in, DeepVqeHopIO *io) {
    return s && s->grid.frames_per_call == 1 ? deepvqe_stream_pre_process_packet(s, mic_in, far_in, io) : -1;
}

int deepvqe_stream_post_process(DeepVqeStream *s, int ai_rc, int16_t *pcm_out) {
    return s && s->grid.frames_per_call == 1 ? deepvqe_stream_post_process_packet(s, ai_rc, pcm_out) : -1;
}

int deepvqe_stream_pre_process_freq(DeepVqeStream *s, const float *mic_re, const float *mic_im,
                                    const float *far_re, const float *far_im, DeepVqeHopIO *io) {
    return s && s->grid.frames_per_call == 1
        ? deepvqe_stream_pre_process_packet_freq(s, mic_re, mic_im, far_re, far_im, io) : -1;
}

int deepvqe_stream_post_process_freq(DeepVqeStream *s, int ai_rc, float *re, float *im) {
    return s && s->grid.frames_per_call == 1 ? deepvqe_stream_post_process_packet_freq(s, ai_rc, re, im) : -1;
}
