/* DeepFilterNet2 model stream (dfn2_stream.h). Reads no file: the ERB tables are the caller's. */

#include <math.h>
#include <string.h>

#include "dfn2_stream.h"
#include "hpf.h"
#include "fft_wrapper.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "stream_mem.h"
#include "stream_range.h"
#include "stream_run.h"

#define DFN2_STREAM_ALIGN DFN2_STREAM_ALIGNMENT

struct DFN2Stream {
    DFN2Grid grid;

    /* hpf, fft, stft, istft: the PCM half, NULL in a freq_only context (the entries test stft / istft). */
    Hpf        *hpf;     /* NULL unless the config enables the input high-pass */
    FftHandle  *fft;     /* carved once, shared by stft and istft */
    AcStft     *stft;
    AcIstft    *istft;
    DFN2Prepost *prepost;

    /* The application spectra of the call, [N][n_bins] each: the STFT output (PCM) or a copy of the caller's app
     * (freq), read again at post. */
    float *spec_re;
    float *spec_im;

    /* PCM scratch, one slab: hop_in/hop_out [hop] each, enh_re/enh_im [n_bins] each (one enhanced frame
     * on its way to the ISTFT). NULL when freq_only. */
    float *hop_in;
    float *hop_out;
    float *enh_re;
    float *enh_im;

    int open;    /* 0 closed, 1: a job is out, post_process validates its heads */
    int pending; /* 0 closed, 1: pre_process succeeded and post_process is due (every hop or packet,
                  * independent of `open`: an N == 1 hop before the first window opens no frame) */
};

typedef struct DFN2StreamLayout {
    DFN2Stream *self;
    /* hpf_mem .. istft_mem and scratch_mem: NULL in a freq_only layout. */
    void       *hpf_mem;      /* NULL region when !cfg->hpf_enable */
    void       *fft_mem;
    void       *stft_mem;
    size_t      stft_bytes;
    void       *istft_mem;
    size_t      istft_bytes;
    void       *prepost_mem;
    size_t      prepost_bytes;
    void       *spec_re_mem;  /* [N][n_bins] each */
    void       *spec_im_mem;
    void       *scratch_mem;  /* the hop_in/hop_out/enh_re/enh_im slab */
    uint32_t    prepost_build_hash; /* dfn2_prepost's own DFN2PrepostMemReq.build_flags_hash, folded
                                      * into this stream's own hash by dfn2_stream_get_mem_size() */
} DFN2StreamLayout;

/* Never dereferenced: dfn2_prepost_get_mem_size()'s byte count and build_flags_hash depend only
 * on cfg->grid and on whether erb_fwd/erb_inv are NULL, never on the pointer values, so sizing
 * does not need the caller's tables. */
static const float SIZING_ERB_DUMMY = 0.0f;

/* STFT and ISTFT borrow the stream's one FftHandle (carved once, handed to both as cfg.fft by
 * dfn2_stream_finish_init()), so each needs the borrowed-handle size. audio_stft.c's config_valid()
 * dereferences a non-NULL cfg->fft, and no handle exists while sizing, so the size is taken with
 * cfg.fft = NULL and the FFT table that size then includes (one ALIGN16 field of
 * fft_get_mem_size(n_fft)) is removed. dfn2_stream_finish_init() asks the borrowed configuration for
 * its size again and refuses a region that is not exactly that. */
static size_t borrowed_fft_size(size_t own_fft_size, int n_fft) {
    size_t fft_field = ck_align16_size(fft_get_mem_size(n_fft));
    return own_fft_size > fft_field ? own_fft_size - fft_field : 0u;
}

static int cfg_valid(const DFN2StreamConfig *cfg) {
    if (!cfg) return 0;
    if (dfn2_grid_validate(&cfg->grid) != DFN2_GRID_OK) return 0;
    if (!isfinite(cfg->atten_lim_db)) return 0;
    /* hpf_init()'s own accepted range (hpf.h): finite, > 0, < 0.45 * sample_rate. Checked here too
     * so get_mem_size() -- which never touches the HPF itself -- rejects the same bad cutoffs init()
     * would, instead of reporting a mem size that then fails at construction. */
    if (cfg->hpf_enable &&
        (!isfinite(cfg->hpf_cutoff_hz) || cfg->hpf_cutoff_hz <= 0.0f ||
         cfg->hpf_cutoff_hz >= 0.45f * (float)cfg->grid.sample_rate))
        return 0;
    /* The HPF runs before the analysis, which a freq_only caller owns. */
    if (cfg->freq_only && cfg->hpf_enable) return 0;
    return 1;
}

/* One carve walk: sizing (base == NULL, every pointer in *out stays NULL) or carving (base !=
 * NULL, *out holds real addresses into `base`); get_mem_size()/init() agreement is `*total`
 * matching across both calls with the same cfg. -1 on a bad config or an overflowing carve.
 * Carve order (part of DFN2_STREAM_LAYOUT_VERSION/build_flags_hash -- bump both if this changes):
 * self, hpf (if enabled), fft, stft, istft, prepost, spec_re, spec_im, scratch slab. freq_only: self, prepost,
 * spec_re, spec_im. */
static int dfn2_stream_layout(const DFN2StreamConfig *cfg, unsigned char *base,
                              DFN2StreamLayout *out, size_t *total) {
    size_t cursor = 0;
    void *ptr;
    AcStftConfig scfg;
    DFN2PrepostConfig pcfg;
    DFN2PrepostMemReq pp_req;
    size_t fft_bytes, stft_bytes, istft_bytes, scratch_bytes, spec_bytes;
    size_t n_bins;

    if (!cfg || !out || !total || !cfg_valid(cfg)) return -1;
    memset(out, 0, sizeof(*out));
    n_bins = (size_t)cfg->grid.n_bins;

    if (stream_carve(base, &cursor, sizeof(DFN2Stream), &ptr) != 0) return -1;
    out->self = (DFN2Stream *)ptr;

    if (!cfg->freq_only) {
        if (cfg->hpf_enable) {
            size_t hb = hpf_get_mem_size();
            if (hb == 0 || stream_carve(base, &cursor, hb, &ptr) != 0) return -1;
            out->hpf_mem = ptr;
        }

        fft_bytes = fft_get_mem_size(cfg->grid.n_fft);
        if (fft_bytes == 0 || stream_carve(base, &cursor, fft_bytes, &ptr) != 0) return -1;
        out->fft_mem = ptr;

        /* One AcStftConfig, reused for both the analysis and synthesis sizing calls -- window left
         * NULL, each of AcStft/AcIstft carves its own window table independently. scfg.fft stays NULL
         * (the shared handle doesn't exist yet), see borrowed_fft_size(). */
        ac_stft_config_default(&scfg, cfg->grid.n_fft);
        scfg.spec_scale = dfn2_stft_scale(cfg->grid.n_fft);
        scfg.ispec_scale = dfn2_stft_unscale(cfg->grid.n_fft);

        stft_bytes = borrowed_fft_size(ac_stft_get_mem_size(&scfg), cfg->grid.n_fft);
        if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        out->stft_mem = ptr;
        out->stft_bytes = stft_bytes;

        istft_bytes = borrowed_fft_size(ac_istft_get_mem_size(&scfg), cfg->grid.n_fft);
        if (istft_bytes == 0 || stream_carve(base, &cursor, istft_bytes, &ptr) != 0) return -1;
        out->istft_mem = ptr;
        out->istft_bytes = istft_bytes;
    }

    if (dfn2_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return -1;
    pcfg.atten_lim_db = cfg->atten_lim_db;
    pcfg.erb_fwd = &SIZING_ERB_DUMMY;
    pcfg.erb_inv = &SIZING_ERB_DUMMY;
    if (dfn2_prepost_get_mem_size(&pcfg, &pp_req) != 0) return -1;
    if (stream_carve(base, &cursor, (size_t)pp_req.bytes, &ptr) != 0) return -1;
    out->prepost_mem = ptr;
    out->prepost_bytes = (size_t)pp_req.bytes;
    out->prepost_build_hash = pp_req.build_flags_hash;

    spec_bytes = ck_mul_size(ck_mul_size((size_t)cfg->grid.frames_per_call, n_bins), sizeof(float));
    if (MEM_SIZE_INVALID(spec_bytes) || stream_carve(base, &cursor, spec_bytes, &ptr) != 0) return -1;
    out->spec_re_mem = ptr;
    if (stream_carve(base, &cursor, spec_bytes, &ptr) != 0) return -1;
    out->spec_im_mem = ptr;

    if (!cfg->freq_only) {
        /* hop_in, hop_out [hop] + enh_re, enh_im [n_bins], one slab; only the slab base is part of the pool ABI. */
        scratch_bytes = ck_add_size(
            ck_mul_size(2u, ck_mul_size((size_t)cfg->grid.hop, sizeof(float))),
            ck_mul_size(2u, ck_mul_size(n_bins, sizeof(float))));
        if (MEM_SIZE_INVALID(scratch_bytes) || stream_carve(base, &cursor, scratch_bytes, &ptr) != 0)
            return -1;
        out->scratch_mem = ptr;
    }

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

static uint32_t build_hash(const DFN2StreamConfig *cfg, uint32_t prepost_hash) {
    uint32_t h = AC_FNV1A_SEED;
    const DFN2Grid *g = &cfg->grid;
    h = stream_fnv1a_u32(h, (uint32_t)g->sample_rate);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_fft);
    h = stream_fnv1a_u32(h, (uint32_t)g->hop);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_bins);
    h = stream_fnv1a_u32(h, (uint32_t)g->n_erb);
    h = stream_fnv1a_u32(h, (uint32_t)g->df_bins);
    h = stream_fnv1a_u32(h, (uint32_t)g->df_order);
    h = stream_fnv1a_u32(h, (uint32_t)g->mask_lookahead);
    h = stream_fnv1a_u32(h, (uint32_t)g->df_lookahead);
    h = stream_fnv1a_u32(h, (uint32_t)g->frames_per_call);
    h = stream_fnv1a_u32(h, (uint32_t)DFN2_STREAM_LAYOUT_VERSION);
#ifdef DFN2_MASK_ONLY
    h = stream_fnv1a_u32(h, 1u);
#else
    h = stream_fnv1a_u32(h, 0u);
#endif
    h = stream_fnv1a_u32(h, (uint32_t)(cfg->hpf_enable ? 1 : 0));
    h = stream_fnv1a_u32(h, (uint32_t)(cfg->freq_only ? 1 : 0));
    h = stream_fnv1a_u32(h, prepost_hash);
    return ac_memreq_abi_hash(h);
}

int dfn2_stream_get_mem_size(const DFN2StreamConfig *cfg, DFN2StreamMemReq *req) {
    DFN2StreamLayout layout;
    size_t total = 0;

    DFN2_REQUIRE_VARIANT_LINK();   /* link-time full/mask-only variant check (dfn2_process.h) */

    if (!req || !cfg_valid(cfg)) return -1;
    if (dfn2_stream_layout(cfg, NULL, &layout, &total) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DFN2_STREAM_DESCRIPTOR_VERSION;
    req->layout_version = DFN2_STREAM_LAYOUT_VERSION;
    req->build_flags_hash = build_hash(cfg, layout.prepost_build_hash);
    req->alignment = DFN2_STREAM_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* Field-by-field DFN2StreamMemReq comparison for dfn2_stream_init_ex(). */
static int dfn2_stream_descriptor_matches(const DFN2StreamMemReq *expected, const DFN2StreamMemReq *req) {
    return expected->descriptor_version == req->descriptor_version &&
           expected->layout_version == req->layout_version &&
           expected->reserved0 == req->reserved0 &&
           expected->build_flags_hash == req->build_flags_hash &&
           expected->alignment == req->alignment &&
           expected->reserved == req->reserved &&
           expected->bytes == req->bytes;
}

/* Constructs every carved object over the caller's borrowed ERB tables. Returns the live DFN2Stream*, or NULL
 * leaving nothing that looks like a valid instance; only called after dfn2_stream_prepare_pool() passed. */
static DFN2Stream *dfn2_stream_finish_init(const DFN2StreamConfig *cfg, unsigned char *base,
                                           const DFN2StreamLayout *layout,
                                           const float *erb_fwd, const float *erb_inv) {
    DFN2Stream *s;
    AcStftConfig scfg;
    DFN2PrepostConfig pcfg;
    size_t n_bins;

    if (!cfg || !base || !layout || !layout->self || !erb_fwd || !erb_inv) return NULL;
    if (!dfn2_erb_tables_valid(&cfg->grid, erb_fwd, erb_inv)) return NULL;
    s = layout->self;

    s->grid = cfg->grid;
    s->open = 0;
    s->pending = 0;
    s->hpf = NULL;
    s->fft = NULL;
    s->stft = NULL;
    s->istft = NULL;

    if (!cfg->freq_only) {
        if (cfg->hpf_enable) {
            s->hpf = hpf_init(layout->hpf_mem, hpf_get_mem_size(), cfg->hpf_cutoff_hz, cfg->grid.sample_rate);
            if (!s->hpf) return NULL;
        }

        s->fft = fft_init(layout->fft_mem, fft_get_mem_size(cfg->grid.n_fft), cfg->grid.n_fft);
        if (!s->fft) return NULL;

        /* The now-constructed handle, borrowed by both: each carved region must be exactly the
         * borrowed-handle size. */
        ac_stft_config_default(&scfg, cfg->grid.n_fft);
        scfg.spec_scale = dfn2_stft_scale(cfg->grid.n_fft);
        scfg.ispec_scale = dfn2_stft_unscale(cfg->grid.n_fft);
        scfg.fft = s->fft;

        if (ac_stft_get_mem_size(&scfg) != layout->stft_bytes ||
            ac_istft_get_mem_size(&scfg) != layout->istft_bytes)
            return NULL;
        s->stft = ac_stft_init(layout->stft_mem, layout->stft_bytes, &scfg);
        if (!s->stft) return NULL;

        s->istft = ac_istft_init(layout->istft_mem, layout->istft_bytes, &scfg);
        if (!s->istft) return NULL;
    }

    if (dfn2_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return NULL;
    pcfg.atten_lim_db = cfg->atten_lim_db;
    pcfg.erb_fwd = erb_fwd;
    pcfg.erb_inv = erb_inv;
    s->prepost = dfn2_prepost_init(layout->prepost_mem, layout->prepost_bytes, &pcfg);
    if (!s->prepost) return NULL;

    s->spec_re = (float *)layout->spec_re_mem;
    s->spec_im = (float *)layout->spec_im_mem;
    if (!cfg->freq_only) {
        n_bins = (size_t)cfg->grid.n_bins;
        s->hop_in = (float *)layout->scratch_mem;
        s->hop_out = s->hop_in + cfg->grid.hop;
        s->enh_re = s->hop_out + cfg->grid.hop;
        s->enh_im = s->enh_re + n_bins;
    }

    return s;
}

/* The size/alignment/stale-pool-descriptor/carve-agreement validation of dfn2_stream_init_ex(). On success (0),
 * `pool` is zeroed and `*layout` is a carved layout over it; -1 on any check failure, `*layout` then undefined. */
static int dfn2_stream_prepare_pool(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes,
                                    const DFN2StreamMemReq *expected, DFN2StreamLayout *layout) {
    DFN2StreamMemReq req;
    size_t total = 0;

    if (!pool || !cfg_valid(cfg)) return -1;
    if (dfn2_stream_get_mem_size(cfg, &req) != 0) return -1;
    if (pool_bytes < (size_t)req.bytes) return -1;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return -1;
    if (expected && !dfn2_stream_descriptor_matches(expected, &req)) return -1;

    memset(pool, 0, (size_t)req.bytes);
    if (dfn2_stream_layout(cfg, (unsigned char *)pool, layout, &total) != 0) return -1;
    if (total != (size_t)req.bytes) return -1;   /* sizing/carve agreement */
    return 0;
}

DFN2Stream *dfn2_stream_init_ex(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes,
                                const DFN2StreamMemReq *expected) {
    DFN2StreamLayout layout;

    if (!cfg || !cfg->erb_fwd || !cfg->erb_inv) return NULL;
    if (dfn2_stream_prepare_pool(cfg, pool, pool_bytes, expected, &layout) != 0) return NULL;

    return dfn2_stream_finish_init(cfg, (unsigned char *)pool, &layout, cfg->erb_fwd, cfg->erb_inv);
}

DFN2Stream *dfn2_stream_init(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes) {
    return dfn2_stream_init_ex(cfg, pool, pool_bytes, NULL);
}

void dfn2_stream_reset(DFN2Stream *s) {
    if (!s) return;
    if (s->hpf) hpf_reset(s->hpf);
    if (s->stft) {   /* a freq_only context has no analysis or synthesis */
        ac_stft_reset(s->stft);
        ac_istft_reset(s->istft);
    }
    dfn2_prepost_reset(s->prepost);
    s->open = 0;
    s->pending = 0;
}

unsigned int dfn2_stream_state_generation(const DFN2Stream *s) {
    return s ? dfn2_prepost_state_generation(s->prepost) : 0u;
}

int dfn2_stream_set_atten_lim(DFN2Stream *s, float atten_lim_db) {
    if (!s || s->pending) return -1;   /* between hops: a hop without a job has no open frame, pending covers it */
    return dfn2_prepost_set_atten_lim(s->prepost, atten_lim_db) != 0 ? -1 : 0;
}

int dfn2_stream_hop_size(const DFN2Stream *s) {
    return s ? s->grid.hop : -1;
}

/* Pre for the N frames staged in s->spec (est = the call's estimation spectra, frame-major): N lib pre_frame calls, then the
 * job's views. The caller has range-checked every bin of the freq entries. */
static int pre_frames(DFN2Stream *s, const float *est_re, const float *est_im, DFN2HopIO *io) {
    const int n = s->grid.frames_per_call;
    const size_t nb = (size_t)s->grid.n_bins;
    int i, ready = 0;

    for (i = 0; i < n; ++i) {
        ready = dfn2_prepost_pre_frame(s->prepost, est_re + (size_t)i * nb, est_im + (size_t)i * nb);
        if (ready < 0) return -1;
    }
    if (n > 1 && ready != 1) return -1;
    io->state_generation = dfn2_prepost_state_generation(s->prepost);
    io->action = ready == 1 ? STREAM_WORK_SUBMIT : STREAM_WORK_BYPASS;
    if (ready == 1) {
        if (dfn2_prepost_views(s->prepost, &io->inputs, &io->outputs) != 0) return -1;
        s->open = 1;
    }
    s->pending = 1;
    return 1;
}

/* The accelerator verdict of a call: 1 when the heads are accepted and every frame commits, 0 for the identity (a refused run
 * has already zeroed the state and advanced the generation once). A BYPASS hop has no job and ignores ai_rc. */
static int close_job(DFN2Stream *s, int ai_rc) {
    int commit = 0;
    if (s->open) {
        if (ai_rc == STREAM_RUN_OK) commit = dfn2_prepost_validate_run(s->prepost) == 0;
        else if (ai_rc != STREAM_RUN_NOT_STARTED) dfn2_prepost_zero_state(s->prepost);
        s->open = 0;
    }
    return commit;
}

/* Frame i of the call into out: its head when the call committed, else the identity. */
static int post_one(DFN2Stream *s, int commit, int i, float *out_re, float *out_im) {
    const size_t off = (size_t)i * (size_t)s->grid.n_bins;
    return commit ? dfn2_prepost_post_frame(s->prepost, i, s->spec_re + off, s->spec_im + off, out_re, out_im)
                  : dfn2_prepost_post_frame_identity(s->prepost, s->spec_re + off, s->spec_im + off, out_re, out_im);
}

/* The N frames of a call to PCM: post, ISTFT and saturate one frame at a time. Returns the frames with a spectrum, -1 on error. */
static int post_pcm_frames(DFN2Stream *s, int commit, int16_t *pcm_out) {
    const int hop = s->grid.hop;
    int i, k, valid = 0;
    for (i = 0; i < s->grid.frames_per_call; ++i) {
        const int r = post_one(s, commit, i, s->enh_re, s->enh_im);
        if (r < 0) return -1;
        valid += r;
        ac_istft_push_planar(s->istft, s->enh_re, s->enh_im, s->hop_out);
        for (k = 0; k < hop; ++k) pcm_out[(size_t)i * (size_t)hop + (size_t)k] = stream_saturate_s16(s->hop_out[k]);
    }
    return valid;
}

/* The N frames of a call to the caller's planes (frame-major). Returns the frames with a spectrum, -1 on error. */
static int post_freq_frames(DFN2Stream *s, int commit, float *re, float *im) {
    const size_t nb = (size_t)s->grid.n_bins;
    int i, valid = 0;
    for (i = 0; i < s->grid.frames_per_call; ++i) {
        const int r = post_one(s, commit, i, re + (size_t)i * nb, im + (size_t)i * nb);
        if (r < 0) return -1;
        valid += r;
    }
    return valid;
}

/* 0, or DFN2_STREAM_ERR_RANGE when a bin of a plane is non-finite or beyond the limit: it would poison the normaliser until reset. */
static int freq_in_range(const DFN2Stream *s, const float *a, const float *b, const float *c, const float *d) {
    const size_t count = (size_t)s->grid.n_bins * (size_t)s->grid.frames_per_call;
    return stream_plane_within(a, count, DFN2_STREAM_BIN_LIMIT_BITS) && stream_plane_within(b, count, DFN2_STREAM_BIN_LIMIT_BITS) &&
           stream_plane_within(c, count, DFN2_STREAM_BIN_LIMIT_BITS) && stream_plane_within(d, count, DFN2_STREAM_BIN_LIMIT_BITS)
               ? 0 : DFN2_STREAM_ERR_RANGE;
}

int dfn2_stream_pre_process(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io) {
    int hop, i;

    if (!s || !pcm_in || !io || s->pending || !s->stft || s->grid.frames_per_call != 1) return -1;
    hop = s->grid.hop;

    for (i = 0; i < hop; ++i) s->hop_in[i] = (float)pcm_in[i] / 32768.0f;
    if (s->hpf) hpf_process(s->hpf, s->hop_in, hop);

    ac_stft_push_planar(s->stft, s->hop_in, s->spec_re, s->spec_im);
    return pre_frames(s, s->spec_re, s->spec_im, io);
}

int dfn2_stream_post_process(DFN2Stream *s, int ai_rc, int16_t *pcm_out) {
    int valid;

    if (!s || !pcm_out || !s->pending || !s->istft || s->grid.frames_per_call != 1) return -1;
    s->pending = 0;
    valid = post_pcm_frames(s, close_job(s, ai_rc), pcm_out);
    return valid < 0 ? -1 : valid ? 1 : 0;
}

int dfn2_stream_pre_process_freq_dual(DFN2Stream *s, const float *est_re, const float *est_im,
                                      const float *app_re, const float *app_im, DFN2HopIO *io) {
    int range;

    if (!s || !est_re || !est_im || !app_re || !app_im || !io || s->pending || s->grid.frames_per_call != 1)
        return -1;
    range = freq_in_range(s, est_re, est_im, app_re, app_im);
    if (range != 0) return range;

    /* the application spectrum is read again at post, so it is copied */
    memcpy(s->spec_re, app_re, (size_t)s->grid.n_bins * sizeof(float));
    memcpy(s->spec_im, app_im, (size_t)s->grid.n_bins * sizeof(float));
    return pre_frames(s, est_re, est_im, io);
}

int dfn2_stream_post_process_freq(DFN2Stream *s, int ai_rc, float *re, float *im, int *valid) {
    int v;
    if (!s || !re || !im || !s->pending || s->grid.frames_per_call != 1) return -1;
    s->pending = 0;
    v = post_freq_frames(s, close_job(s, ai_rc), re, im);
    if (v < 0) return -1;
    if (valid) *valid = v;
    return 0;
}

int dfn2_stream_pre_process_packet(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io) {
    int hop, nb, n, i, k;

    if (!s || !pcm_in || !io || s->pending || !s->stft || s->grid.frames_per_call < 2) return -1;
    hop = s->grid.hop;
    nb = s->grid.n_bins;
    n = s->grid.frames_per_call;
    for (i = 0; i < n; ++i) {
        for (k = 0; k < hop; ++k) s->hop_in[k] = (float)pcm_in[(size_t)i * (size_t)hop + (size_t)k] / 32768.0f;
        if (s->hpf) hpf_process(s->hpf, s->hop_in, hop);
        ac_stft_push_planar(s->stft, s->hop_in, s->spec_re + (size_t)i * (size_t)nb, s->spec_im + (size_t)i * (size_t)nb);
    }
    return pre_frames(s, s->spec_re, s->spec_im, io);
}

int dfn2_stream_post_process_packet(DFN2Stream *s, int ai_rc, int16_t *pcm_out) {
    if (!s || !pcm_out || !s->pending || !s->istft || s->grid.frames_per_call < 2) return -1;
    s->pending = 0;
    return post_pcm_frames(s, close_job(s, ai_rc), pcm_out);
}

int dfn2_stream_pre_process_packet_freq_dual(DFN2Stream *s, const float *est_re, const float *est_im,
                                             const float *app_re, const float *app_im, DFN2HopIO *io) {
    int range;

    if (!s || !est_re || !est_im || !app_re || !app_im || !io || s->pending || s->grid.frames_per_call < 2)
        return -1;
    range = freq_in_range(s, est_re, est_im, app_re, app_im);
    if (range != 0) return range;
    memcpy(s->spec_re, app_re, (size_t)s->grid.n_bins * (size_t)s->grid.frames_per_call * sizeof(float));
    memcpy(s->spec_im, app_im, (size_t)s->grid.n_bins * (size_t)s->grid.frames_per_call * sizeof(float));
    return pre_frames(s, est_re, est_im, io);
}

int dfn2_stream_post_process_packet_freq(DFN2Stream *s, int ai_rc, float *re, float *im, int *valid_frames) {
    int v;
    if (!s || !re || !im || !s->pending || s->grid.frames_per_call < 2) return -1;
    s->pending = 0;
    v = post_freq_frames(s, close_job(s, ai_rc), re, im);
    if (v < 0) return -1;
    if (valid_frames) *valid_frames = v;
    return 0;
}
