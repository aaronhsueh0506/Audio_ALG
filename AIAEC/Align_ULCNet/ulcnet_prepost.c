/* Align-ULCNet pre/post class (contract: ulcnet_prepost.h). Composes
 * ulcnet_process.c (sqrt-Hann STFT/WOLA, one rolling frame per hop) and
 * ulcnet_model_io.c (feature front end, tensor views, recurrent windows), and
 * adds the object lifecycle, pool carve and per-hop frame state machine.
 * Requires -ffp-contract=off. No heap in _init, no globals, no stdio. */

#include "ulcnet_prepost.h"

#include "utility/mem_align.h"

#include <stdlib.h>
#include <string.h>

/* Pool carve alignment. Shared with ulcnet_model_io so one cursor serves
 * both regions. */
#define PP_ALIGN ULCNET_MODEL_IO_ALIGNMENT

struct UlcnetPrepost {
    UlcnetModelIoState *io;

    int io_mode;

    /* ULCNET_IO_TIME only; NULL in ULCNET_IO_FREQ. Rolling (center 0) analyses
     * borrow the caller's FftHandle and window and carry their own transform
     * scratch; the synthesis state is the caller's AcIstft (cfg->istft). */
    AcStft  *ana_err;
    AcStft  *ana_far;
    AcIstft *synth;

    /* Per-hop frame staging: one frame, in both I/O modes. */
    float *err_re;      /* [ULCNET_BINS] */
    float *err_im;
    float *far_re;
    float *far_im;
    float *enh_re;      /* [ULCNET_BINS] commit staging */
    float *enh_im;
    float *out_hop;     /* [ULCNET_HOP] synthesis output, TIME only; zero from init/reset, written wholesale */

    int frame_open;     /* a frame is pending the accelerator */
    int prepared;       /* frame_inputs() armed the accelerator transaction
                         * for the open frame; implies frame_open */
    int written;        /* samples the last closed frame put in out_hop:
                         * 0 (first frame after reset) or ULCNET_HOP */

    /* _create bookkeeping; 0 for an _init instance. */
    void *heap_base;
};

/* ---- pool arithmetic ----
 * Alignment via ulcnet_model_io's helper, addition via audiolib's saturating
 * ck_add_size: overflow pins the cursor to SIZE_MAX, which MEM_SIZE_INVALID
 * catches. */

/* Reserve `bytes` at *cursor, aligned. Pass base==NULL to size only. */
static int pp_carve(unsigned char *base, size_t *cursor, size_t bytes,
                    void **out) {
    size_t aligned;
    size_t next;
    if (ulcnet_model_io_align_up(*cursor, PP_ALIGN, &aligned) != 0) return -1;
    next = ck_add_size(aligned, bytes);
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

static uint32_t pp_fnv1a(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* Build identity a pool is valid for; io_mode is included because the two modes
 * carve different regions. */
static uint32_t pp_build_hash(int io_mode, int delay_depth, uint32_t layout) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a(h, (uint32_t)ULCNET_SR);
    h = pp_fnv1a(h, (uint32_t)ULCNET_N_FFT);
    h = pp_fnv1a(h, layout);
    h = pp_fnv1a(h, (uint32_t)ULCNET_PREPOST_CARVE_VERSION);
    h = pp_fnv1a(h, (uint32_t)io_mode);
    h = pp_fnv1a(h, (uint32_t)delay_depth);
    return ac_memreq_abi_hash(h);
}

/* TIME needs the caller's handle and window in the config so the analysis
 * objects are sized exactly. */
static int pp_framing_size(const UlcnetPrepostConfig *cfg, size_t *stft_bytes) {
    AcStftConfig sc;
    ulcnet_framing_config(&sc, cfg->fft, cfg->window, 0);
    *stft_bytes = ac_stft_get_mem_size(&sc);
    return *stft_bytes != 0 ? 0 : -1;
}

/* The caller's AcIstft must be the one ulcnet_framing_config() describes (the
 * model's fft size, envelope-normalised overlap-add, synthesis scale) over the
 * same window values as the analysis. */
static int pp_istft_matches(const UlcnetPrepostConfig *cfg) {
    AcStftConfig want, got;
    ulcnet_framing_config(&want, NULL, NULL, 0);
    return cfg->istft != NULL && ac_istft_get_config(cfg->istft, &got) == 0 &&
           got.fft_size == want.fft_size && got.ola == want.ola &&
           got.ispec_scale == want.ispec_scale &&
           memcmp(got.window, cfg->window,
                  (size_t)ULCNET_N_FFT * sizeof(float)) == 0;
}

/* Walk the layout once: base==NULL sizes it, otherwise it also carves.
 * get_mem_size and init share this walk so their order agrees. */
static int pp_layout(UlcnetPrepost *p, unsigned char *base,
                     const UlcnetPrepostConfig *cfg,
                     const UlcnetModelIoDescriptor *descriptor,
                     const UlcnetModelIoMemReq *io_req,
                     size_t *total) {
    size_t cursor = 0;
    const size_t bins = (size_t)ULCNET_BINS * sizeof(float);
    void *ptr;

    if (pp_carve(base, &cursor, sizeof(UlcnetPrepost), &ptr) != 0) return -1;

    /* model_io's own pool, at its own alignment. */
    {
        size_t aligned;
        if (ulcnet_model_io_align_up(cursor, io_req->alignment, &aligned) != 0)
            return -1;
        cursor = ck_add_size(aligned, io_req->bytes);
        if (MEM_SIZE_INVALID(cursor)) return -1;
        if (base && p) {
            p->io = ulcnet_model_io_init(base + aligned, io_req->bytes,
                                         descriptor);
            if (!p->io) return -1;
        }
    }

    if (cfg->io_mode == ULCNET_IO_TIME) {
        size_t stft_bytes;
        AcStftConfig sc;
        if (pp_framing_size(cfg, &stft_bytes) != 0) return -1;
        ulcnet_framing_config(&sc, cfg->fft, cfg->window, 0);
        if (pp_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        if (base && p) {
            p->ana_err = ac_stft_init(ptr, stft_bytes, &sc);
            if (!p->ana_err) return -1;
        }
        if (pp_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        if (base && p) {
            p->ana_far = ac_stft_init(ptr, stft_bytes, &sc);
            if (!p->ana_far) return -1;
        }
        if (base && p) p->synth = cfg->istft;
        if (pp_carve(base, &cursor, (size_t)ULCNET_HOP * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) p->out_hop = (float *)ptr;
    }

    /* Frame staging: one frame, in both modes. */
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->err_re = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->err_im = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->far_re = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->far_im = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->enh_re = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->enh_im = (float *)ptr;

    return ulcnet_model_io_align_up(cursor, PP_ALIGN, total);
}

/* ---- config / sizing ------------------------------------------------- */

int ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg,
                                   int io_mode, int delay_depth) {
    if (!cfg) return -1;
    if (io_mode != ULCNET_IO_TIME && io_mode != ULCNET_IO_FREQ) return -1;
    if (delay_depth < ULCNET_MODEL_IO_MIN_D ||
        delay_depth > ULCNET_MODEL_IO_MAX_D) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->io_mode = io_mode;
    cfg->delay_depth = delay_depth;
    return 0;
}

static int pp_check_config(const UlcnetPrepostConfig *cfg) {
    if (!cfg) return -1;
    if (cfg->io_mode != ULCNET_IO_TIME && cfg->io_mode != ULCNET_IO_FREQ)
        return -1;
    if (cfg->delay_depth < ULCNET_MODEL_IO_MIN_D ||
        cfg->delay_depth > ULCNET_MODEL_IO_MAX_D) return -1;
    if (cfg->io_mode == ULCNET_IO_TIME) {
        /* Reject-first: a wrong-size handle would break the feature-time contract. */
        if (!cfg->fft || !cfg->window) return -1;
        if (fft_get_n_freqs(cfg->fft) != ULCNET_BINS) return -1;
        if (!pp_istft_matches(cfg)) return -1;
    }
    return 0;
}

static int pp_descriptor_for(const UlcnetPrepostConfig *cfg,
                             UlcnetModelIoDescriptor *out) {
    if (ulcnet_model_io_descriptor_default(cfg->delay_depth, out) != 0)
        return -1;
    if (cfg->model_layout_version)
        out->layout_version = cfg->model_layout_version;
    return ulcnet_model_io_descriptor_validate(out);
}

int ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg,
                                UlcnetPrepostMemReq *req) {
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoMemReq io_req;
    size_t total = 0;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_descriptor_for(cfg, &descriptor) != 0) return -1;
    if (ulcnet_model_io_get_mem_requirements(&descriptor, &io_req) != 0)
        return -1;
    if (pp_layout(NULL, NULL, cfg, &descriptor, &io_req, &total) != 0)
        return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = ULCNET_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = descriptor.layout_version;
    req->io_mode = (uint32_t)cfg->io_mode;
    req->build_flags_hash = pp_build_hash(cfg->io_mode, cfg->delay_depth,
                                         descriptor.layout_version);
    req->alignment = (uint32_t)(io_req.alignment > PP_ALIGN
                                ? io_req.alignment : PP_ALIGN);
    req->bytes = (uint64_t)total;
    return 0;
}

/* ---- lifecycle ------------------------------------------------------- */

UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes,
                                      const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected) {
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoMemReq io_req;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    size_t total = 0;

    if (!pool || ulcnet_prepost_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        /* Stale-pool gate: a pool sized by another build, io_mode or D is
         * refused, never reinterpreted. */
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->io_mode != req.io_mode ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }
    if (pp_descriptor_for(cfg, &descriptor) != 0) return NULL;
    if (ulcnet_model_io_get_mem_requirements(&descriptor, &io_req) != 0)
        return NULL;

    /* Zero the whole control region first so a poisoned pool initialises like a zeroed one. */
    memset(pool, 0, (size_t)req.bytes);
    p = (UlcnetPrepost *)pool;
    p->io_mode = cfg->io_mode;

    if (pp_layout(p, (unsigned char *)pool, cfg, &descriptor, &io_req,
                  &total) != 0)
        return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */
    if (p->synth) ac_istft_reset(p->synth);   /* a borrowed AcIstft starts like a fresh one */

    return p;
}

UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes,
                                   const UlcnetPrepostConfig *cfg) {
    return ulcnet_prepost_init_ex(pool, bytes, cfg, NULL);
}

UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg) {
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    void *mem = NULL;

    if (ulcnet_prepost_get_mem_size(cfg, &req) != 0) return NULL;
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    mem = aligned_alloc((size_t)req.alignment, (size_t)req.bytes);
#else
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        mem = NULL;
#endif
    if (!mem) return NULL;
    p = ulcnet_prepost_init(mem, (size_t)req.bytes, cfg);
    if (!p) {
        free(mem);
        return NULL;
    }
    p->heap_base = mem;
    return p;
}

void ulcnet_prepost_destroy(UlcnetPrepost *p) {
    void *base;
    if (!p) return;
    base = p->heap_base;
    p->heap_base = NULL;   /* nothing to free for a pool instance; repeat calls are no-ops */
    if (base) free(base);
}

void ulcnet_prepost_reset(UlcnetPrepost *p) {
    if (!p) return;
    ulcnet_model_io_reset(p->io);
    if (p->io_mode == ULCNET_IO_TIME) {
        /* The objects keep their handle and window; reset clears the history
         * and overlap state, the caller's AcIstft included. */
        ac_stft_reset(p->ana_err);
        ac_stft_reset(p->ana_far);
        ac_istft_reset(p->synth);
        memset(p->out_hop, 0, (size_t)ULCNET_HOP * sizeof(float));
    }
    memset(p->enh_re, 0, (size_t)ULCNET_BINS * sizeof(float));
    memset(p->enh_im, 0, (size_t)ULCNET_BINS * sizeof(float));
    p->frame_open = 0;
    p->prepared = 0;
    p->written = 0;
}

/* ---- accessors ------------------------------------------------------- */

uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p) {
    return p ? ulcnet_model_io_state_generation(p->io) : 0u;
}

int ulcnet_prepost_hop_size(const UlcnetPrepost *p) {
    return p ? ULCNET_HOP : -1;
}

int ulcnet_prepost_num_bins(const UlcnetPrepost *p) {
    return p ? ULCNET_BINS : -1;
}

int ulcnet_prepost_io_mode(const UlcnetPrepost *p) {
    return p ? p->io_mode : -1;
}

const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p) {
    return p ? ulcnet_model_io_descriptor(p->io) : NULL;
}

/* ---- per-hop stages -------------------------------------------------- */

int ulcnet_prepost_pre_process(UlcnetPrepost *p,
                               const float err_hop[ULCNET_HOP],
                               const float far_hop[ULCNET_HOP]) {
    if (!p || !err_hop || !far_hop) return -1;
    if (p->io_mode != ULCNET_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* previous frame neither committed nor skipped */

    /* center=False rolling analysis, one frame per hop from the first hop; both
     * streams use the same hop index (no internal skew). */
    (void)ac_stft_push_planar(p->ana_err, err_hop, p->err_re, p->err_im);
    (void)ac_stft_push_planar(p->ana_far, far_hop, p->far_re, p->far_im);

    p->frame_open = 1;
    return 1;   /* invariant: exactly one accelerator invocation per hop */
}

int ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                    const float err_re[ULCNET_BINS],
                                    const float err_im[ULCNET_BINS],
                                    const float far_re[ULCNET_BINS],
                                    const float far_im[ULCNET_BINS]) {
    const size_t bins = (size_t)ULCNET_BINS * sizeof(float);

    if (!p || !err_re || !err_im || !far_re || !far_im) return -1;
    if (p->io_mode != ULCNET_IO_FREQ) return -1;
    if (p->frame_open) return -1;   /* previous frame neither committed nor skipped */

    memcpy(p->err_re, err_re, bins);
    memcpy(p->err_im, err_im, bins);
    memcpy(p->far_re, far_re, bins);
    memcpy(p->far_im, far_im, bins);

    p->frame_open = 1;
    return 1;
}

int ulcnet_prepost_frame_inputs(UlcnetPrepost *p,
                                UlcnetModelIoInputs *inputs,
                                UlcnetModelIoOutputs *outputs) {
    if (!p || !inputs || !outputs || !p->frame_open) return -1;
    /* Arms the transaction and NaN-fills the `output` mask, so a repeated request still gets a clean one. */
    if (ulcnet_model_io_prepare(p->io, p->err_re, p->err_im,
                                p->far_re, p->far_im, inputs, outputs) != 0)
        return -1;
    p->prepared = 1;
    return 0;
}

/* Push the finished frame through the synthesis (TIME) or stage it (FREQ) and
 * close it. Cannot fail: the synthesis reports 0 samples (first frame) or a full
 * ULCNET_HOP, so a failed commit leaves framing and synthesis state untouched. */
static void pp_close_frame(UlcnetPrepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    if (p->io_mode == ULCNET_IO_TIME) {
        p->written = ac_istft_push_planar(p->synth, p->enh_re, p->enh_im,
                                          p->out_hop);
    }
}

int ulcnet_prepost_frame_commit(UlcnetPrepost *p) {
    /* `prepared` is this class's own latch: after frame_skip() model_io's
     * transaction stays armed with stale outputs, so its flag alone would let the
     * next commit accept them without frame_inputs(). */
    if (!p || !p->prepared) return -1;
    if (ulcnet_model_io_commit(p->io, p->enh_re, p->enh_im) != 0) {
        /* model_io discarded the transaction and restarted the recurrent state;
         * the frame stays open for frame_skip() or a fresh frame_inputs(). */
        p->prepared = 0;
        return -1;
    }
    pp_close_frame(p);
    return 0;
}

int ulcnet_prepost_frame_skip(UlcnetPrepost *p) {
    const size_t bins = (size_t)ULCNET_BINS * sizeof(float);
    if (!p || !p->frame_open) return -1;
    /* Identity: the error spectrum passes through; the armed transaction is not committed. */
    memcpy(p->enh_re, p->err_re, bins);
    memcpy(p->enh_im, p->err_im, bins);
    pp_close_frame(p);
    return 0;
}

int ulcnet_prepost_post_process(UlcnetPrepost *p,
                                float out_hop[ULCNET_HOP], int *written) {
    if (!p || !out_hop) return -1;
    if (p->io_mode != ULCNET_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* a frame is still awaiting the model */
    /* Always a full defined hop: out_hop is zero from init/reset and written
     * wholesale, so the warm-up frame yields silence. */
    memcpy(out_hop, p->out_hop, (size_t)ULCNET_HOP * sizeof(float));
    if (written) *written = p->written;
    return 0;
}

int ulcnet_prepost_post_process_freq(UlcnetPrepost *p,
                                     float re[ULCNET_BINS],
                                     float im[ULCNET_BINS]) {
    const size_t bins = (size_t)ULCNET_BINS * sizeof(float);
    if (!p || !re || !im) return -1;
    if (p->io_mode != ULCNET_IO_FREQ) return -1;
    if (p->frame_open) return -1;
    memcpy(re, p->enh_re, bins);
    memcpy(im, p->enh_im, bins);
    return 0;
}
