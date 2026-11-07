/* DeepFilterNet2 PCM stream: memory-init entry point and everything except file loading. No
 * <stdio.h>, no malloc/calloc/posix_memalign anywhere in this TU -- it must link into a NO_STDIO
 * target. See dfn2_stream.h for the public contract and dfn2_stream_internal.h for the pool
 * layout this file carves and constructs. */

#include <math.h>
#include <string.h>

#include "dfn2_stream_internal.h"
#include "utility/mem_align.h"
#include "stream_mem.h"

#define DFN2_STREAM_ALIGN DFN2_STREAM_ALIGNMENT

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN2_STREAM_ALIGN == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif


/* Never dereferenced: dfn2_prepost_get_mem_size()'s byte count and build_flags_hash depend only on
 * cfg->grid and on whether erb_fwd/erb_inv are NULL, never on the pointer values -- confirmed
 * against lib/ainr/dfn2/dfn2_prepost.c's pp_check_config()/pp_build_hash(). Using this in file
 * mode (where the real tables aren't loaded yet at sizing time) gives byte-identical results to
 * using the real, later-loaded pointers. */
static const float SIZING_ERB_DUMMY = 0.0f;

/* NOT safe to fake with a dummy pointer: audio_stft.c's config_valid() calls
 * fft_get_n_freqs(cfg->fft) whenever cfg->fft is non-NULL, so a bogus address there segfaults.
 * Sizing therefore always uses cfg.fft = NULL (as if AcStft/AcIstft each carved their own handle,
 * which costs fft_get_mem_size(n_fft) extra bytes apiece); dfn2_stream_finish_init() then
 * actually constructs one shared FftHandle and, at real ac_stft_init()/ac_istft_init() time
 * only, points cfg.fft at it (borrowed mode needs strictly fewer bytes than owned mode -- see
 * audio_stft.c's layout_walk(): `fft_bytes = 0` when cfg->fft is set -- so the smaller real
 * requirement always fits the larger carved region). This spends a little pool space reserving
 * room neither AcStft nor AcIstft actually uses, in exchange for a sizing pass that cannot
 * dereference a handle that doesn't exist yet. */

int dfn2_stream_is_file_mode(const DFN2StreamConfig *cfg) {
    return cfg && cfg->erb_fwd == NULL && cfg->erb_inv == NULL;
}

static int cfg_valid(const DFN2StreamConfig *cfg) {
    if (!cfg) return 0;
    if (dfn2_grid_validate(&cfg->grid) != DFN2_GRID_OK) return 0;
    if (cfg->grid.frames_per_call != 1) return 0;
    if (!isfinite(cfg->atten_lim_db)) return 0;
    /* hpf_init()'s own accepted range (hpf.h): finite, > 0, < 0.45 * sample_rate. Checked here too
     * so get_mem_size() -- which never touches the HPF itself -- rejects the same bad cutoffs init()
     * would, instead of reporting a mem size that then fails at construction. */
    if (cfg->hpf_enable &&
        (!isfinite(cfg->hpf_cutoff_hz) || cfg->hpf_cutoff_hz <= 0.0f ||
         cfg->hpf_cutoff_hz >= 0.45f * (float)cfg->grid.sample_rate))
        return 0;
    /* Exactly one of erb_fwd/erb_inv set is never valid: memory mode needs both, file mode needs
     * neither (dfn2_stream_is_file_mode() reads both NULL). */
    if ((cfg->erb_fwd == NULL) != (cfg->erb_inv == NULL)) return 0;
    return 1;
}

int dfn2_stream_layout(const DFN2StreamConfig *cfg, unsigned char *base,
                       DFN2StreamLayout *out, size_t *total) {
    size_t cursor = 0;
    void *ptr;
    int file_mode;
    AcStftConfig scfg;
    DFN2PrepostConfig pcfg;
    DFN2PrepostMemReq pp_req;
    size_t fft_bytes, stft_bytes, istft_bytes, scratch_bytes, erb_bytes;
    size_t n_bins, n_erb, erb_elems;

    if (!cfg || !out || !total || !cfg_valid(cfg)) return -1;
    memset(out, 0, sizeof(*out));
    file_mode = dfn2_stream_is_file_mode(cfg);
    n_bins = (size_t)cfg->grid.n_bins;
    n_erb = (size_t)cfg->grid.n_erb;

    if (stream_carve(base, &cursor, sizeof(DFN2Stream), &ptr) != 0) return -1;
    out->self = (DFN2Stream *)ptr;

    if (cfg->hpf_enable) {
        size_t hb = hpf_get_mem_size();
        if (hb == 0 || stream_carve(base, &cursor, hb, &ptr) != 0) return -1;
        out->hpf_mem = ptr;
    }

    fft_bytes = fft_get_mem_size(cfg->grid.n_fft);
    if (fft_bytes == 0 || stream_carve(base, &cursor, fft_bytes, &ptr) != 0) return -1;
    out->fft_mem = ptr;

    /* One AcStftConfig, reused for both the analysis and synthesis sizing/init calls exactly as
     * the board sample reuses its own single `synth_cfg` -- window left NULL (each of AcStft/
     * AcIstft carves its own window table independently; the sample never shares it either). */
    ac_stft_config_default(&scfg, cfg->grid.n_fft);
    scfg.spec_scale = cfg->grid.stft_scale;
    scfg.ispec_scale = cfg->grid.stft_unscale;
    /* scfg.fft stays NULL here (own-carve sizing); see the comment on dfn2_stream_finish_init()'s
     * own scfg for why this cannot borrow the not-yet-constructed shared handle. */

    stft_bytes = ac_stft_get_mem_size(&scfg);
    if (stft_bytes == 0 || stream_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
    out->stft_mem = ptr;

    istft_bytes = ac_istft_get_mem_size(&scfg);
    if (istft_bytes == 0 || stream_carve(base, &cursor, istft_bytes, &ptr) != 0) return -1;
    out->istft_mem = ptr;

    if (dfn2_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return -1;
    pcfg.atten_lim_db = cfg->atten_lim_db;
    if (file_mode) {
        pcfg.erb_fwd = &SIZING_ERB_DUMMY;
        pcfg.erb_inv = &SIZING_ERB_DUMMY;
    } else {
        pcfg.erb_fwd = cfg->erb_fwd;
        pcfg.erb_inv = cfg->erb_inv;
    }
    if (dfn2_prepost_get_mem_size(&pcfg, &pp_req) != 0) return -1;
    if (stream_carve(base, &cursor, (size_t)pp_req.bytes, &ptr) != 0) return -1;
    out->prepost_mem = ptr;
    out->prepost_bytes = (size_t)pp_req.bytes;
    out->prepost_build_hash = pp_req.build_flags_hash;

    /* hop_in, hop_out [hop] + spec_re, spec_im, enh_re, enh_im [n_bins], one slab (dfn_res_stage.c's
     * "one aligned slab for the six spectra" pattern); only the slab base is part of the pool ABI. */
    scratch_bytes = ck_add_size(
        ck_mul_size(2u, ck_mul_size((size_t)cfg->grid.hop, sizeof(float))),
        ck_mul_size(4u, ck_mul_size(n_bins, sizeof(float))));
    if (MEM_SIZE_INVALID(scratch_bytes) || stream_carve(base, &cursor, scratch_bytes, &ptr) != 0)
        return -1;
    out->scratch_mem = ptr;

    if (file_mode) {
        erb_elems = ck_mul_size(n_bins, n_erb);
        erb_bytes = ck_mul_size(erb_elems, sizeof(float));
        if (MEM_SIZE_INVALID(erb_bytes)) return -1;

        if (stream_carve(base, &cursor, erb_bytes, &ptr) != 0) return -1;
        out->erb_fwd_storage = (float *)ptr;
        if (stream_carve(base, &cursor, erb_bytes, &ptr) != 0) return -1;
        out->erb_inv_storage = (float *)ptr;
        if (stream_carve(base, &cursor, erb_bytes, &ptr) != 0) return -1;
        out->erb_scratch = (float *)ptr;
    }

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

static uint32_t build_hash(const DFN2StreamConfig *cfg, uint32_t prepost_hash) {
    uint32_t h = 2166136261u;
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
    h = stream_fnv1a_u32(h, dfn2_stream_is_file_mode(cfg) ? 1u : 0u);
    h = stream_fnv1a_u32(h, (uint32_t)(cfg->hpf_enable ? 1 : 0));
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

int dfn2_stream_descriptor_matches(const DFN2StreamMemReq *expected, const DFN2StreamMemReq *req) {
    return expected->descriptor_version == req->descriptor_version &&
           expected->layout_version == req->layout_version &&
           expected->reserved0 == req->reserved0 &&
           expected->build_flags_hash == req->build_flags_hash &&
           expected->alignment == req->alignment &&
           expected->reserved == req->reserved &&
           expected->bytes == req->bytes;
}

DFN2Stream *dfn2_stream_finish_init(const DFN2StreamConfig *cfg, unsigned char *base,
                                    const DFN2StreamLayout *layout,
                                    const float *erb_fwd, const float *erb_inv) {
    DFN2Stream *s;
    AcStftConfig scfg;
    DFN2PrepostConfig pcfg;
    size_t n_bins;

    if (!cfg || !base || !layout || !layout->self || !erb_fwd || !erb_inv) return NULL;
    s = layout->self;

    s->grid = cfg->grid;
    s->atten_lim_db = cfg->atten_lim_db;
    s->hpf_enable = cfg->hpf_enable;
    s->open = 0;
    s->heap_base = NULL;
    s->hpf = NULL;

    if (cfg->hpf_enable) {
        s->hpf = hpf_init(layout->hpf_mem, hpf_get_mem_size(), cfg->hpf_cutoff_hz, cfg->grid.sample_rate);
        if (!s->hpf) return NULL;
    }

    s->fft = fft_init(layout->fft_mem, fft_get_mem_size(cfg->grid.n_fft), cfg->grid.n_fft);
    if (!s->fft) return NULL;

    /* Real, now-constructed handle, borrowed by both -- a *different*, smaller `scfg` than
     * dfn2_stream_layout()'s own sizing config (which used fft = NULL so it could run before this
     * handle existed). ac_stft_init()/ac_istft_init() each recheck mem_size against their own
     * freshly-computed (smaller, borrowed) requirement, which the larger carved region always
     * satisfies -- see the comment above dfn2_stream_layout()'s stft/istft sizing. */
    ac_stft_config_default(&scfg, cfg->grid.n_fft);
    scfg.spec_scale = cfg->grid.stft_scale;
    scfg.ispec_scale = cfg->grid.stft_unscale;
    scfg.fft = s->fft;

    s->stft = ac_stft_init(layout->stft_mem, ac_stft_get_mem_size(&scfg), &scfg);
    if (!s->stft) return NULL;

    s->istft = ac_istft_init(layout->istft_mem, ac_istft_get_mem_size(&scfg), &scfg);
    if (!s->istft) return NULL;

    if (dfn2_prepost_config_defaults(&pcfg, &cfg->grid) != 0) return NULL;
    pcfg.atten_lim_db = cfg->atten_lim_db;
    pcfg.erb_fwd = erb_fwd;
    pcfg.erb_inv = erb_inv;
    s->prepost = dfn2_prepost_init(layout->prepost_mem, layout->prepost_bytes, &pcfg);
    if (!s->prepost) return NULL;

    n_bins = (size_t)cfg->grid.n_bins;
    s->hop_in = (float *)layout->scratch_mem;
    s->hop_out = s->hop_in + cfg->grid.hop;
    s->spec_re = s->hop_out + cfg->grid.hop;
    s->spec_im = s->spec_re + n_bins;
    s->enh_re = s->spec_im + n_bins;
    s->enh_im = s->enh_re + n_bins;

    return s;
}

int dfn2_stream_prepare_pool(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes,
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

    if (dfn2_stream_is_file_mode(cfg)) return NULL;
    if (dfn2_stream_prepare_pool(cfg, pool, pool_bytes, expected, &layout) != 0) return NULL;

    return dfn2_stream_finish_init(cfg, (unsigned char *)pool, &layout, cfg->erb_fwd, cfg->erb_inv);
}

DFN2Stream *dfn2_stream_init(const DFN2StreamConfig *cfg, void *pool, size_t pool_bytes) {
    return dfn2_stream_init_ex(cfg, pool, pool_bytes, NULL);
}

void dfn2_stream_reset(DFN2Stream *s) {
    if (!s) return;
    if (s->hpf) hpf_reset(s->hpf);
    ac_stft_reset(s->stft);
    ac_istft_reset(s->istft);
    dfn2_prepost_reset(s->prepost);
    s->open = 0;
}

unsigned int dfn2_stream_state_generation(const DFN2Stream *s) {
    return s ? dfn2_prepost_state_generation(s->prepost) : 0u;
}

const DFN2Grid *dfn2_stream_grid(const DFN2Stream *s) {
    return s ? &s->grid : NULL;
}

int dfn2_stream_hop_size(const DFN2Stream *s) {
    return s ? s->grid.hop : -1;
}

int dfn2_stream_layout_version(const DFN2Stream *s) {
    return s ? dfn2_prepost_layout_version(s->prepost) : -1;
}

int dfn2_stream_pre_process(DFN2Stream *s, const int16_t *pcm_in, DFN2HopIO *io) {
    int hop, i;

    if (!s || !pcm_in || !io || s->open) return -1;
    hop = s->grid.hop;

    for (i = 0; i < hop; ++i) s->hop_in[i] = (float)pcm_in[i] / 32768.0f;
    if (s->hpf_enable) hpf_process(s->hpf, s->hop_in, hop);

    ac_stft_push_planar(s->stft, s->hop_in, s->spec_re, s->spec_im);

    /* N == 1 (enforced at init): pre_process_freq() returns 1 on every hop, warm-up included; any
     * other return here is a contract error this stream does not expect in correct use. */
    if (dfn2_prepost_pre_process_freq(s->prepost, s->spec_re, s->spec_im) != 1) return -1;
    if (dfn2_prepost_frame_open(s->prepost, &io->inputs, &io->outputs) != 0) return -1;

    io->state_generation = dfn2_prepost_state_generation(s->prepost);
    s->open = 1;
    return 1;
}

int dfn2_stream_post_process(DFN2Stream *s, int ai_rc, int16_t *pcm_out) {
    int hop, i, valid, prc, rc;

    if (!s || !pcm_out || !s->open) return -1;
    hop = s->grid.hop;

    /* Mirror the board sample's dispatch exactly: a finish that itself fails (non-finite head or
     * state, despite ai_rc == 0) still needs a bypass to actually close the frame -- frame_finish()
     * leaves it open on failure (dfn2_prepost.h), and a frame left open here would desync this
     * stream's own `open` bookkeeping from the DFN2Prepost object's. */
    rc = (ai_rc == 0) ? dfn2_prepost_frame_finish(s->prepost) : -1;
    if (rc != 0) dfn2_prepost_frame_bypass(s->prepost);
    s->open = 0;

    valid = 0;
    prc = dfn2_prepost_post_process_freq(s->prepost, s->enh_re, s->enh_im, &valid);
    if (prc == 0) {
        ac_istft_push_planar(s->istft, s->enh_re, s->enh_im, s->hop_out);
    } else {
        /* Defensive only, unreached in correct use (the hop was just closed above): mirror the
         * board sample's own fallback exactly -- zero the hop, leave the synthesis untouched. */
        memset(s->hop_out, 0, (size_t)hop * sizeof(float));
        valid = 0;
    }

    for (i = 0; i < hop; ++i) pcm_out[i] = stream_saturate_s16(s->hop_out[i]);

    return valid ? 1 : 0;
}
