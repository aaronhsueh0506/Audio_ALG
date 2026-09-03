/* Align-ULCNet pre/post class. Contract: see ulcnet_prepost.h.
 *
 * This file OWNS no signal processing of its own. It composes the two
 * parity-tested translation units -- ulcnet_process.c (centered sqrt-Hann
 * STFT/WOLA) and ulcnet_model_io.c (feature front end, accelerator tensor
 * views, recurrent rings) -- and adds only the object lifecycle, the pool
 * carve and the per-hop frame state machine. Neither of those files
 * changes, so both of their standalone parity builds keep linking.
 *
 * Constraint inherited from both: -ffp-contract=off. No heap in _init, no
 * globals, no stdio. */

#include "ulcnet_prepost.h"

#include "mem_align.h"

#include <stdlib.h>
#include <string.h>

/* Pool carve alignment. Shared with ulcnet_model_io so one cursor serves
 * both regions. */
#define PP_ALIGN ULCNET_MODEL_IO_ALIGNMENT

struct UlcnetPrepost {
    UlcnetModelIoState *io;

    int io_mode;

    /* Borrowed, never created or destroyed here. NULL in ULCNET_IO_FREQ. */
    FftHandle   *fft;
    const float *window;

    /* ULCNET_IO_TIME only; NULL in ULCNET_IO_FREQ.
     *
     * The analysis is a plain 1-frame-per-hop rolling window, NOT
     * ulcnet_analysis_push's centered schedule -- see the header's FRAMING
     * note. `seg`/`spec` are the shared per-call transform scratch: the two
     * analyses and the synthesis are strictly sequential within a hop, so
     * one set serves all three (the same rule the FftHandle is shared
     * under). Struct-owned, never stack, so an embedded task stack never
     * sees a multi-KB frame. */
    float           *hist_err;   /* [ULCNET_N_FFT] */
    float           *hist_far;   /* [ULCNET_N_FFT] */
    float           *seg;        /* [ULCNET_N_FFT] windowed staging  */
    Complex         *spec;       /* [ULCNET_BINS]  rfft staging      */
    UlcnetSynthesis *synth;

    /* Per-hop frame staging: one frame, in both I/O modes. */
    float *err_re;      /* [ULCNET_BINS] */
    float *err_im;
    float *far_re;
    float *far_im;
    float *enh_re;      /* [ULCNET_BINS] commit staging */
    float *enh_im;
    float *out_hop;     /* [ULCNET_HOP] the synthesis output; TIME only.
                         * Zero from init/reset, then only ever written
                         * wholesale by the synthesis. */

    int frame_open;     /* a frame is pending the accelerator */
    int prepared;       /* frame_inputs() armed the accelerator transaction
                         * for the open frame; implies frame_open */
    int written;        /* samples the last closed frame put in out_hop:
                         * 0 (first frame after reset) or ULCNET_HOP */

    /* _create bookkeeping; 0 for an _init instance. */
    void *heap_base;
};

/* ---- pool arithmetic ---------------------------------------------------
 * Alignment through ulcnet_model_io's exported helper (its region carries
 * its own alignment), addition through audio_common's saturating
 * ck_add_size: an overflow pins the cursor to SIZE_MAX, which
 * MEM_SIZE_INVALID catches. */

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

/* The build identity a pool is only valid for. io_mode is IN here because
 * the two modes carve different regions at the same total-size class. */
static uint32_t pp_build_hash(int io_mode, int delay_depth) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a(h, (uint32_t)ULCNET_SR);
    h = pp_fnv1a(h, (uint32_t)ULCNET_N_FFT);
    h = pp_fnv1a(h, (uint32_t)ULCNET_MODEL_IO_LAYOUT_VERSION);
    h = pp_fnv1a(h, (uint32_t)io_mode);
    h = pp_fnv1a(h, (uint32_t)delay_depth);
    return h;
}

/* Walk the layout once. base==NULL sizes it; base!=NULL also carves it.
 * get_mem_size and init MUST walk in this same order -- that is the whole
 * reason it is one function. */
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
        const size_t nfft = (size_t)ULCNET_N_FFT * sizeof(float);
        if (pp_carve(base, &cursor, nfft, &ptr) != 0) return -1;
        if (base && p) p->hist_err = (float *)ptr;
        if (pp_carve(base, &cursor, nfft, &ptr) != 0) return -1;
        if (base && p) p->hist_far = (float *)ptr;
        if (pp_carve(base, &cursor, nfft, &ptr) != 0) return -1;
        if (base && p) p->seg = (float *)ptr;
        if (pp_carve(base, &cursor, (size_t)ULCNET_BINS * sizeof(Complex),
                     &ptr) != 0) return -1;
        if (base && p) p->spec = (Complex *)ptr;
        if (pp_carve(base, &cursor, sizeof(UlcnetSynthesis), &ptr) != 0) return -1;
        if (base && p) p->synth = (UlcnetSynthesis *)ptr;
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
        /* Reject-first, same rule as ulcnet_analysis_init: a handle of the
         * wrong size would silently break the feature-time contract. */
        if (!cfg->fft || !cfg->window) return -1;
        if (fft_get_n_freqs(cfg->fft) != ULCNET_BINS) return -1;
    }
    return 0;
}

static int pp_descriptor_for(const UlcnetPrepostConfig *cfg,
                             UlcnetModelIoDescriptor *out) {
    if (ulcnet_model_io_descriptor_default(cfg->delay_depth, out) != 0)
        return -1;
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
    req->layout_version = ULCNET_MODEL_IO_LAYOUT_VERSION;
    req->io_mode = (uint32_t)cfg->io_mode;
    req->build_flags_hash = pp_build_hash(cfg->io_mode, cfg->delay_depth);
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

    /* Whole control region zeroed first, so a poisoned pool initialises
     * identically to a zeroed one. */
    memset(pool, 0, (size_t)req.bytes);
    p = (UlcnetPrepost *)pool;
    p->io_mode = cfg->io_mode;
    p->fft = (cfg->io_mode == ULCNET_IO_TIME) ? cfg->fft : NULL;
    p->window = (cfg->io_mode == ULCNET_IO_TIME) ? cfg->window : NULL;

    if (pp_layout(p, (unsigned char *)pool, cfg, &descriptor, &io_req,
                  &total) != 0)
        return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    if (cfg->io_mode == ULCNET_IO_TIME &&
        ulcnet_synthesis_init(p->synth, p->fft, p->window) != 0) {
        return NULL;   /* the rolling analysis has no state but its history,
                        * which the whole-pool memset above already cleared */
    }
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
    p->heap_base = NULL;   /* a pool instance owns nothing to free and
                            * stays usable: repeat calls on it are no-ops */
    if (base) free(base);
}

void ulcnet_prepost_reset(UlcnetPrepost *p) {
    if (!p) return;
    ulcnet_model_io_reset(p->io);
    if (p->io_mode == ULCNET_IO_TIME) {
        /* Re-init IS the reset for the synthesis (see ulcnet_process.h);
         * the rolling analysis resets by zeroing its history. */
        memset(p->hist_err, 0, (size_t)ULCNET_N_FFT * sizeof(float));
        memset(p->hist_far, 0, (size_t)ULCNET_N_FFT * sizeof(float));
        ulcnet_synthesis_init(p->synth, p->fft, p->window);
        memset(p->out_hop, 0, (size_t)ULCNET_HOP * sizeof(float));
    }
    memset(p->enh_re, 0, (size_t)ULCNET_BINS * sizeof(float));
    memset(p->enh_im, 0, (size_t)ULCNET_BINS * sizeof(float));
    p->frame_open = 0;
    p->prepared = 0;
    p->written = 0;
}

/* ---- accessors ------------------------------------------------------- */

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

/* center=False rolling analysis: append one hop, then transform the last
 * ULCNET_N_FFT samples. Exactly one frame per hop, from the very first hop.
 *
 * Arithmetic is deliberately identical -- same window multiply, same
 * fft_forward_scratch, same expression order -- to ulcnet_process.c's
 * ulcnet_rfft, so a frame produced here is bit-for-bit the frame the
 * centered analysis emits LAST on the same hop. That equality is what makes
 * this a pure re-labelling of the schedule rather than a change of signal. */
static void pp_roll(UlcnetPrepost *p, float *history,
                    const float hop_in[ULCNET_HOP],
                    float *out_re, float *out_im) {
    int k;
    memmove(history, history + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memcpy(history + ULCNET_N_FFT - ULCNET_HOP, hop_in,
           (size_t)ULCNET_HOP * sizeof(float));
    for (k = 0; k < ULCNET_N_FFT; ++k) p->seg[k] = history[k] * p->window[k];
    fft_forward_scratch(p->fft, p->seg, p->spec);
    for (k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = p->spec[k].r;
        out_im[k] = p->spec[k].i;
    }
}

int ulcnet_prepost_pre_process(UlcnetPrepost *p,
                               const float err_hop[ULCNET_HOP],
                               const float far_hop[ULCNET_HOP]) {
    if (!p || !err_hop || !far_hop) return -1;
    if (p->io_mode != ULCNET_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* previous frame neither committed nor skipped */

    /* Both streams are transformed from the SAME hop index -- this class
     * applies no internal skew -- so they cannot desynchronise. */
    pp_roll(p, p->hist_err, err_hop, p->err_re, p->err_im);
    pp_roll(p, p->hist_far, far_hop, p->far_re, p->far_im);

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
    /* Arms the transaction and NaN-fills every accelerator output, so a
     * caller that asks twice still gets a clean one rather than a
     * half-written one. */
    if (ulcnet_model_io_prepare(p->io, p->err_re, p->err_im,
                                p->far_re, p->far_im, inputs, outputs) != 0)
        return -1;
    p->prepared = 1;
    return 0;
}

/* Push the finished frame through the synthesis (TIME) or stage it (FREQ)
 * and close it. Cannot fail: the synthesis only ever reports 0 samples (the
 * very first frame, whose block lies inside the trimmed half window) or a
 * full ULCNET_HOP, and never reads `out`. So the only fallible step of a
 * commit is the validation that precedes this, and "on failure nothing
 * moves" is structural rather than argued. */
static void pp_close_frame(UlcnetPrepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    if (p->io_mode == ULCNET_IO_TIME) {
        p->written = ulcnet_synthesis_push(p->synth, p->enh_re, p->enh_im,
                                           p->out_hop);
    }
}

int ulcnet_prepost_frame_commit(UlcnetPrepost *p) {
    /* `prepared` implies frame_open. It is the class's OWN latch, not
     * model_io's: a frame_skip() leaves model_io's transaction armed with
     * stale outputs, and on model_io's flag alone the next frame's commit
     * would accept them with no frame_inputs() behind it. */
    if (!p || !p->prepared) return -1;
    if (ulcnet_model_io_commit(p->io, p->enh_re, p->enh_im) != 0) {
        /* model_io already discarded the transaction and left persistent
         * state untouched. The frame stays open so the caller can take the
         * identity with frame_skip(), or re-arm with frame_inputs(). */
        p->prepared = 0;
        return -1;
    }
    pp_close_frame(p);
    return 0;
}

int ulcnet_prepost_frame_skip(UlcnetPrepost *p) {
    const size_t bins = (size_t)ULCNET_BINS * sizeof(float);
    if (!p || !p->frame_open) return -1;
    /* Identity: the error spectrum passes through. The armed transaction is
     * simply not committed, so no ring advances; the next prepare() re-arms
     * it and re-fills the accelerator outputs with NaN. */
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
    /* Always a full, defined hop: out_hop is zero from init/reset and the
     * synthesis only ever writes it wholesale, so the one warm-up frame that
     * emits nothing hands over silence, never stale samples. */
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
