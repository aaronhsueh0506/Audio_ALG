/* DeepFilterNet3 pre/post class (contract: dfn3_prepost.h). Composes
 * dfn3_process.c (ERB and complex features, mask expansion, deep filter,
 * attenuation limit; the center=False sqrt-Hann framing is audiolib's
 * AcStft/AcIstft) and dfn3_model_io.c (graph feature windows, recurrent caches),
 * and adds the lifecycle, pool carve and per-hop frame state machine.
 * Requires -ffp-contract=off. No heap in _init, no globals, no stdio. */

#include "dfn3_prepost.h"

#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PP_ALIGN ((size_t)DFN3_PREPOST_ALIGNMENT)

/* DFN3 reads raw future low bins, so the two lookaheads overlap. */
#define PP_MODEL_LOOKAHEAD (DFN3_MASK_LOOKAHEAD)

struct DFN3Prepost {
    DFN3State        *dsp;
    DFN3ModelIOState *io;

    int   io_mode;
    float atten_lim_db;

    /* Framing objects carved from the pool in DFN3_IO_TIME (NULL in DFN3_IO_FREQ);
     * they borrow the caller's FftHandle and, when given, window. */
    AcStft      *stft;
    AcIstft     *istft;

    /* Borrowed, never created or destroyed here. */
    const float *erb_fwd;
    const float *erb_inv;

    /* Per-hop staging. */
    float *spec_re;     /* [DFN3_N_BINS]  current frame, original scale     */
    float *spec_im;
    /* DFN3_IO_FREQ only: application spectrum of the dual entry point.
     * compose_re/compose_im select what the compose stage reads this hop (spec_*
     * on single-input entries, apply_* on the dual one). */
    float *apply_re;    /* [DFN3_N_BINS]                                    */
    float *apply_im;
    const float *compose_re;
    const float *compose_im;
    float *feat_erb;    /* [DFN3_N_ERB]                                     */
    float *feat_spec;   /* [2][DFN3_DF_BINS], two contiguous segments       */
    float *enh_re;      /* [DFN3_N_BINS]  compose output                    */
    float *enh_im;
    float *out_hop;     /* [DFN3_HOP_LEN], DFN3_IO_TIME only                */

    /* Accelerator-writable boundary: the two heads. The recurrent state is
     * DFN3ModelIOState's own two arrays at fixed addresses; a runtime writes its
     * state outputs to the same addresses as the inputs. */
    float *head_erb_mask;   /* [DFN3_N_ERB]                                 */
    float *head_coefs;      /* [DFN3_PREPOST_COEFS_ELEMENTS]                */

    /* Identity heads for frame_skip: unit band mask and unit center tap. Written
     * once at init; dfn3_compose_stream refuses NULL heads, so pass-through is
     * expressed as heads. */
    float *skip_mask;       /* [DFN3_N_ERB], all 1.0f                       */
    float *skip_coefs;      /* [DFN3_PREPOST_COEFS_ELEMENTS], center tap 1   */

    int       frame_open;    /* a frame awaits commit or skip               */
    int       prepared;      /* frame_inputs() armed the accelerator transaction for the open frame */
    int       have_output;   /* the compose stage emitted this hop          */
    long long output_frame;  /* source frame of the last emission, -1 none  */

    /* _create bookkeeping; 0 for an _init instance. */
    void *heap_base;
};

/* ---- pool arithmetic ----
 * audiolib's saturating helpers (mem_align.h): overflow pins the cursor to
 * SIZE_MAX, which MEM_SIZE_INVALID catches. */

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN3_PREPOST_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* Reserve `bytes` at *cursor, aligned. Pass base==NULL to size only. */
static int pp_carve(unsigned char *base, size_t *cursor, size_t bytes,
                    void **out) {
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, bytes);
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

static uint32_t pp_fnv1a_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

static uint32_t pp_fnv1a_str(uint32_t h, const char *s) {
    while (*s) {
        h ^= (uint32_t)(unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

/* Build identity a pool is valid for. io_mode and the feature version are
 * included so a pool handed across a mode or checkpoint-contract change is refused. */
static uint32_t pp_build_hash(int io_mode) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_SR);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_N_FFT);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_HOP_LEN);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_N_ERB);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_DF_BINS);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_DF_ORDER);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MASK_LOOKAHEAD);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_DF_LOOKAHEAD);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_IO_LAYOUT_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_PREPOST_CARVE_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_INPUT_FRAMES);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_GRU_HIDDEN);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_ENCODER_CHANNELS);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_GRU_ROWS);
    h = pp_fnv1a_u32(h, (uint32_t)DFN3_MODEL_DF_PATHWAY_WINDOW);
    h = pp_fnv1a_str(h, DFN3_FEATURE_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)io_mode);
    return ac_memreq_abi_hash(h);
}

/* Sizing a borrowed table needs only whether the pointer is NULL;
 * ac_stft/ac_istft get_mem_size never read it. */
static const float pp_sizing_window[1];

/* DFN3_IO_TIME: bytes of the two framing objects, which share the caller's FFT
 * handle and one window table (the caller's, else the one the analysis object
 * carves). */
static int pp_framing_sizes(const DFN3PrepostConfig *cfg, size_t *stft_bytes,
                            size_t *istft_bytes) {
    AcStftConfig sc;
    dfn3_framing_config(&sc, cfg->fft, cfg->window);
    *stft_bytes = ac_stft_get_mem_size(&sc);
    dfn3_framing_config(&sc, cfg->fft,
                        cfg->window ? cfg->window : pp_sizing_window);
    *istft_bytes = ac_istft_get_mem_size(&sc);
    return (*stft_bytes != 0 && *istft_bytes != 0) ? 0 : -1;
}

/* Walk the layout once: base==NULL sizes it, otherwise it also carves (framing
 * blocks returned via stft_block / istft_block, NULL in DFN3_IO_FREQ).
 * get_mem_size and init share this walk so their order agrees. */
static int pp_layout(DFN3Prepost *p, unsigned char *base,
                     const DFN3PrepostConfig *cfg, size_t *total,
                     void **stft_block, void **istft_block) {
    const int io_mode = cfg->io_mode;
    static const size_t float_counts[] = {
        DFN3_N_BINS,
        DFN3_N_BINS,
        DFN3_N_ERB,
        2u * DFN3_DF_BINS,
        DFN3_N_BINS,
        DFN3_N_BINS,
        DFN3_PREPOST_ERB_MASK_ELEMENTS,
        DFN3_PREPOST_COEFS_ELEMENTS,
        DFN3_PREPOST_ERB_MASK_ELEMENTS,
        DFN3_PREPOST_COEFS_ELEMENTS
    };
    const size_t n_float_regions =
        sizeof(float_counts) / sizeof(float_counts[0]);
    size_t cursor = 0;
    size_t i;
    void *ptr = NULL;
    float **slots[sizeof(float_counts) / sizeof(float_counts[0])];

    if (pp_carve(base, &cursor, sizeof(DFN3Prepost), &ptr) != 0) return -1;
    if (pp_carve(base, &cursor, sizeof(DFN3State), &ptr) != 0) return -1;
    if (base && p) p->dsp = (DFN3State *)ptr;
    if (pp_carve(base, &cursor, sizeof(DFN3ModelIOState), &ptr) != 0)
        return -1;
    if (base && p) p->io = (DFN3ModelIOState *)ptr;

    /* DFN3_IO_TIME only: the output hop staging and the two framing objects. */
    if (io_mode == DFN3_IO_TIME) {
        size_t stft_bytes, istft_bytes;
        if (pp_framing_sizes(cfg, &stft_bytes, &istft_bytes) != 0) return -1;
        if (pp_carve(base, &cursor, (size_t)DFN3_HOP_LEN * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) p->out_hop = (float *)ptr;
        if (pp_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        if (stft_block) *stft_block = ptr;
        if (pp_carve(base, &cursor, istft_bytes, &ptr) != 0) return -1;
        if (istft_block) *istft_block = ptr;
    }

    if (p) {
        slots[0]  = &p->spec_re;       slots[1]  = &p->spec_im;
        slots[2]  = &p->feat_erb;      slots[3]  = &p->feat_spec;
        slots[4]  = &p->enh_re;        slots[5]  = &p->enh_im;
        slots[6]  = &p->head_erb_mask; slots[7]  = &p->head_coefs;
        slots[8]  = &p->skip_mask;     slots[9]  = &p->skip_coefs;
    }
    for (i = 0; i < n_float_regions; ++i) {
        if (pp_carve(base, &cursor, float_counts[i] * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) *slots[i] = (float *)ptr;
    }

    /* DFN3_IO_FREQ only: the dual entry point's application spectrum, appended
     * last so other regions' offsets do not depend on io_mode. */
    if (io_mode == DFN3_IO_FREQ) {
        if (pp_carve(base, &cursor, (size_t)DFN3_N_BINS * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) p->apply_re = (float *)ptr;
        if (pp_carve(base, &cursor, (size_t)DFN3_N_BINS * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) p->apply_im = (float *)ptr;
    }

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

/* ---- config / sizing ------------------------------------------------- */

int dfn3_prepost_config_defaults(DFN3PrepostConfig *cfg, int io_mode) {
    if (!cfg) return -1;
    if (io_mode != DFN3_IO_TIME && io_mode != DFN3_IO_FREQ) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->io_mode = io_mode;
    cfg->atten_lim_db = 0.0f;
    return 0;
}

static int pp_check_config(const DFN3PrepostConfig *cfg) {
    if (!cfg) return -1;
    if (cfg->io_mode != DFN3_IO_TIME && cfg->io_mode != DFN3_IO_FREQ)
        return -1;
    /* Required in both modes: mask expansion and the ERB feature branch run on
     * the spectrum side. */
    if (!cfg->erb_fwd || !cfg->erb_inv) return -1;
    if (!isfinite(cfg->atten_lim_db)) return -1;
    if (cfg->io_mode == DFN3_IO_TIME) {
        /* Reject-first: a wrong-size handle would break the grid silently. */
        if (!cfg->fft) return -1;
        if (fft_get_n_freqs(cfg->fft) != DFN3_N_BINS) return -1;
    }
    return 0;
}

int dfn3_prepost_get_mem_size(const DFN3PrepostConfig *cfg,
                              DFN3PrepostMemReq *req) {
    size_t total = 0;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_layout(NULL, NULL, cfg, &total, NULL, NULL) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DFN3_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = (uint32_t)DFN3_MODEL_IO_LAYOUT_VERSION;
    req->io_mode = (uint32_t)cfg->io_mode;
    req->build_flags_hash = pp_build_hash(cfg->io_mode);
    req->alignment = (uint32_t)PP_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* ---- lifecycle ------------------------------------------------------- */

/* Bring DFN3State (with the framing objects) and DFN3ModelIOState to their
 * init values and reapply the borrowed ERB matrices. */
static void pp_reset_states(DFN3Prepost *p) {
    dfn3_state_init(p->dsp, p->stft, p->istft);
    dfn3_set_erb_matrices(p->dsp, p->erb_fwd, p->erb_inv);
    dfn3_model_io_reset(p->io);
}

/* Per-hop bookkeeping only. The staging buffers are zero from init/reset, the
 * compose stage returns before writing them on the one warm-up hop, and every
 * later hop writes them wholesale, so they are not cleared here. */
static void pp_clear_frame(DFN3Prepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    p->have_output = 0;
}

static void pp_zero_staging(DFN3Prepost *p) {
    memset(p->enh_re, 0, (size_t)DFN3_N_BINS * sizeof(float));
    memset(p->enh_im, 0, (size_t)DFN3_N_BINS * sizeof(float));
    if (p->io_mode == DFN3_IO_TIME)
        memset(p->out_hop, 0, (size_t)DFN3_HOP_LEN * sizeof(float));
}

DFN3Prepost *dfn3_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN3PrepostConfig *cfg,
                                  const DFN3PrepostMemReq *expected) {
    DFN3PrepostMemReq req;
    DFN3Prepost *p;
    size_t total = 0;
    void *stft_block = NULL;
    void *istft_block = NULL;
    int i;

    if (!pool || dfn3_prepost_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        /* Stale-pool gate: a pool sized by another build or io_mode is refused. */
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->io_mode != req.io_mode ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }

    /* Zero the whole control region first so a poisoned pool initialises like a
     * zeroed one. */
    memset(pool, 0, (size_t)req.bytes);
    p = (DFN3Prepost *)pool;
    p->io_mode = cfg->io_mode;
    p->atten_lim_db = cfg->atten_lim_db;
    p->erb_fwd = cfg->erb_fwd;
    p->erb_inv = cfg->erb_inv;

    if (pp_layout(p, (unsigned char *)pool, cfg, &total, &stft_block,
                  &istft_block) != 0)
        return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    if (cfg->io_mode == DFN3_IO_TIME) {
        size_t stft_bytes, istft_bytes;
        AcStftConfig sc;
        if (pp_framing_sizes(cfg, &stft_bytes, &istft_bytes) != 0)
            return NULL;
        dfn3_framing_config(&sc, cfg->fft, cfg->window);
        p->stft = ac_stft_init(stft_block, stft_bytes, &sc);
        if (!p->stft) return NULL;
        dfn3_framing_config(&sc, cfg->fft, ac_stft_window(p->stft));
        p->istft = ac_istft_init(istft_block, istft_bytes, &sc);
        if (!p->istft) return NULL;
    }

    for (i = 0; i < DFN3_N_ERB; ++i) p->skip_mask[i] = 1.0f;
    memset(p->skip_coefs, 0,
           (size_t)DFN3_PREPOST_COEFS_ELEMENTS * sizeof(float));

    for (i = 0; i < DFN3_DF_BINS; ++i)
        p->skip_coefs[(i * DFN3_DF_ORDER + DFN3_DF_HISTORY) * 2] = 1.0f;

    pp_reset_states(p);   /* staging is already zero from the pool memset */
    pp_clear_frame(p);
    p->output_frame = -1;
    return p;
}

DFN3Prepost *dfn3_prepost_init(void *pool, size_t bytes,
                               const DFN3PrepostConfig *cfg) {
    return dfn3_prepost_init_ex(pool, bytes, cfg, NULL);
}

DFN3Prepost *dfn3_prepost_create(const DFN3PrepostConfig *cfg) {
    DFN3PrepostMemReq req;
    DFN3Prepost *p;
    void *mem = NULL;

    if (dfn3_prepost_get_mem_size(cfg, &req) != 0) return NULL;
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    mem = aligned_alloc((size_t)req.alignment, (size_t)req.bytes);
#else
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        mem = NULL;
#endif
    if (!mem) return NULL;
    p = dfn3_prepost_init(mem, (size_t)req.bytes, cfg);
    if (!p) {
        free(mem);
        return NULL;
    }
    p->heap_base = mem;
    return p;
}

void dfn3_prepost_destroy(DFN3Prepost *p) {
    void *base;
    if (!p) return;
    base = p->heap_base;
    p->heap_base = NULL;   /* nothing to free for a pool instance; repeat calls are no-ops */
    if (base) free(base);
}

void dfn3_prepost_reset(DFN3Prepost *p) {
    if (!p) return;
    pp_reset_states(p);
    pp_zero_staging(p);
    pp_clear_frame(p);
    p->output_frame = -1;
}

unsigned int dfn3_prepost_state_generation(const DFN3Prepost *p) {
    return p ? dfn3_model_io_state_generation(p->io) : 0u;
}

/* ---- accessors ------------------------------------------------------- */

int dfn3_prepost_hop_size(const DFN3Prepost *p) {
    return p ? DFN3_HOP_LEN : -1;
}

int dfn3_prepost_num_bins(const DFN3Prepost *p) {
    return p ? DFN3_N_BINS : -1;
}

int dfn3_prepost_io_mode(const DFN3Prepost *p) {
    return p ? p->io_mode : -1;
}

int dfn3_prepost_model_lookahead_frames(const DFN3Prepost *p) {
    return p ? PP_MODEL_LOOKAHEAD : -1;
}

int dfn3_prepost_layout_version(const DFN3Prepost *p) {
    return p ? DFN3_MODEL_IO_LAYOUT_VERSION : -1;
}

int dfn3_prepost_set_erb_matrices(DFN3Prepost *p, const float *erb_fwd,
                                  const float *erb_inv) {
    if (!p || !erb_fwd || !erb_inv) return -1;
    /* Between hops only: a swap must not land inside an open transaction
     * (atomicity per hop; see dfn3_prepost.h). */
    if (p->frame_open) return -1;
    p->erb_fwd = erb_fwd;
    p->erb_inv = erb_inv;
    dfn3_set_erb_matrices(p->dsp, erb_fwd, erb_inv);
    return 0;
}

int dfn3_prepost_set_atten_lim(DFN3Prepost *p, float atten_lim_db) {
    if (!p || !isfinite(atten_lim_db)) return -1;
    if (p->frame_open) return -1;   /* between hops only, like the ERB pair */
    p->atten_lim_db = atten_lim_db;
    return 0;
}

int dfn3_prepost_output_frame_index(const DFN3Prepost *p, long long *frame) {
    if (!p || !frame || p->output_frame < 0) return -1;
    *frame = p->output_frame;
    return 0;
}

/* ---- per-hop stages -------------------------------------------------- */

static int pp_all_finite(const float *values, size_t count) {
    return skn_all_finite_f32(values, count);
}

static void pp_fill_nan(float *values, size_t count) {
    skn_fill_f32(values, count, (float)NAN);
}

/* Features, graph window and compose clock, shared by every pre_process entry
 * point. `est_*` is read only inside this call. Returns the number of
 * accelerator invocations this hop needs (0 or 1). */
static int pp_begin_frame(DFN3Prepost *p, const float *est_re,
                          const float *est_im) {
    int heads_needed;

    pp_clear_frame(p);
    dfn3_compute_features(p->dsp, est_re, est_im, p->feat_erb,
                          p->feat_spec);
    heads_needed = dfn3_model_io_push_features(
        p->io, p->feat_erb,
        (const float (*)[DFN3_DF_BINS])p->feat_spec);
    if (heads_needed < 0) return -1;

    if (heads_needed == 0) {
        /* Left warm-up: no right-hand neighbour yet, so the compose clock advances
         * with heads_valid=0. dfn3_compose_stream() accepts this exactly while
         * current < DFN3_MASK_LOOKAHEAD; a disagreement returns -1 and is surfaced. */
        if (dfn3_compose_stream(p->dsp, p->compose_re, p->compose_im, 0,
                                NULL, NULL, p->atten_lim_db, p->enh_re,
                                p->enh_im, NULL) != 0) {
            return -1;
        }
        return 0;
    }
    p->frame_open = 1;
    return 1;
}

int dfn3_prepost_pre_process(DFN3Prepost *p,
                             const float in_hop[DFN3_HOP_LEN]) {
    if (!p || !in_hop) return -1;
    if (p->io_mode != DFN3_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* previous frame never closed */
    if (!skn_all_finite_f32(in_hop, DFN3_HOP_LEN)) return -1;
    dfn3_analysis(p->dsp, in_hop, p->spec_re, p->spec_im);
    p->compose_re = p->spec_re;
    p->compose_im = p->spec_im;
    return pp_begin_frame(p, p->spec_re, p->spec_im);
}

int dfn3_prepost_pre_process_freq(DFN3Prepost *p,
                                  const float spec_re[DFN3_N_BINS],
                                  const float spec_im[DFN3_N_BINS]) {
    const size_t bins = (size_t)DFN3_N_BINS * sizeof(float);
    if (!p || !spec_re || !spec_im) return -1;
    if (p->io_mode != DFN3_IO_FREQ) return -1;
    if (p->frame_open) return -1;
    if (!skn_all_finite_f32(spec_re, DFN3_N_BINS) ||
        !skn_all_finite_f32(spec_im, DFN3_N_BINS)) return -1;
    memcpy(p->spec_re, spec_re, bins);
    memcpy(p->spec_im, spec_im, bins);
    p->compose_re = p->spec_re;
    p->compose_im = p->spec_im;
    return pp_begin_frame(p, p->spec_re, p->spec_im);
}

int dfn3_prepost_pre_process_freq_dual(DFN3Prepost *p,
                                       const float est_re[DFN3_N_BINS],
                                       const float est_im[DFN3_N_BINS],
                                       const float app_re[DFN3_N_BINS],
                                       const float app_im[DFN3_N_BINS]) {
    const size_t bins = (size_t)DFN3_N_BINS * sizeof(float);
    if (!p || !est_re || !est_im || !app_re || !app_im) return -1;
    if (p->io_mode != DFN3_IO_FREQ) return -1;
    if (p->frame_open) return -1;
    if (!skn_all_finite_f32(est_re, DFN3_N_BINS) ||
        !skn_all_finite_f32(est_im, DFN3_N_BINS) ||
        !skn_all_finite_f32(app_re, DFN3_N_BINS) ||
        !skn_all_finite_f32(app_im, DFN3_N_BINS)) return -1;
    /* The application spectrum is read again at commit, so it is copied; the
     * estimate feeds only this hop's features. */
    memcpy(p->apply_re, app_re, bins);
    memcpy(p->apply_im, app_im, bins);
    p->compose_re = p->apply_re;
    p->compose_im = p->apply_im;
    return pp_begin_frame(p, est_re, est_im);
}

int dfn3_prepost_frame_inputs(DFN3Prepost *p, DFN3PrepostInputs *inputs,
                              DFN3PrepostOutputs *outputs) {
    if (!p || !inputs || !outputs || !p->frame_open) return -1;

    memset(inputs, 0, sizeof(*inputs));
    inputs->erb_window =
        (const float (*)[DFN3_N_ERB])p->io->erb_window;
    inputs->spec_window =
        (const float (*)[DFN3_MODEL_INPUT_FRAMES][DFN3_DF_BINS])
            p->io->spec_window;
    inputs->gru_hidden =
        (const float (*)[DFN3_MODEL_GRU_HIDDEN])p->io->gru_hidden;
    inputs->df_convp_window =
        (const float (*)[DFN3_MODEL_DF_PATHWAY_WINDOW][DFN3_DF_BINS])
            p->io->df_convp_window;
    inputs->erb_window_elements = DFN3_PREPOST_ERB_WINDOW_ELEMENTS;
    inputs->spec_window_elements = DFN3_PREPOST_SPEC_WINDOW_ELEMENTS;
    inputs->gru_hidden_elements = DFN3_PREPOST_GRU_HIDDEN_ELEMENTS;
    inputs->df_convp_window_elements = DFN3_PREPOST_CONVP_WINDOW_ELEMENTS;

    /* NaN-fill the two heads so a repeated request gets a clean boundary and a
     * partial accelerator write fails commit. The recurrent state is the
     * accelerator's input too and is left as is. */
    pp_fill_nan(p->head_erb_mask, DFN3_PREPOST_ERB_MASK_ELEMENTS);
    pp_fill_nan(p->head_coefs, DFN3_PREPOST_COEFS_ELEMENTS);

    memset(outputs, 0, sizeof(*outputs));
    outputs->erb_mask = p->head_erb_mask;
    outputs->coefs = p->head_coefs;
    outputs->gru_hidden_next = p->io->gru_hidden;
    outputs->df_convp_window_next = p->io->df_convp_window;
    outputs->erb_mask_elements = DFN3_PREPOST_ERB_MASK_ELEMENTS;
    outputs->coefs_elements = DFN3_PREPOST_COEFS_ELEMENTS;
    outputs->gru_hidden_elements = DFN3_PREPOST_GRU_HIDDEN_ELEMENTS;
    outputs->df_convp_window_elements = DFN3_PREPOST_CONVP_WINDOW_ELEMENTS;
    p->prepared = 1;
    return 0;
}

/* Run the compose stage with the given heads and, in DFN3_IO_TIME, the
 * synthesis. Closes the frame on success only; the compose clock advances once
 * per hop, here or in the warm-up branch of pp_begin_frame. */
static int pp_close_frame(DFN3Prepost *p, const float *erb_mask,
                          const float *coefs) {
    long long frame = 0;
    int emitted;

    /* The compose stage validates pointers, head finiteness and clock alignment
     * before its first write, so -1 leaves the compose state untouched and the
     * frame open for frame_skip(). */
    emitted = dfn3_compose_stream(p->dsp, p->compose_re, p->compose_im, 1,
                                  erb_mask, coefs, p->atten_lim_db,
                                  p->enh_re, p->enh_im, &frame);
    if (emitted < 0) return -1;
    p->frame_open = 0;
    p->prepared = 0;
    p->have_output = emitted;
    if (!emitted) return 0;   /* still inside the lookahead warm-up */

    p->output_frame = frame;
    if (p->io_mode == DFN3_IO_TIME) {
        /* One synthesis push per emitted frame and none on warm-up hops. */
        dfn3_synthesis(p->dsp, p->enh_re, p->enh_im, p->out_hop);
    }
    return 0;
}

int dfn3_prepost_frame_commit(DFN3Prepost *p) {
    int heads_finite;
    int state_finite;
    if (!p || !p->prepared) return -1;   /* prepared implies frame_open */

    /* The accelerator overwrote the recurrent state in place, so a non-finite head
     * or state element zeroes both recurrent tensors, advances the state
     * generation, disarms the transaction and leaves the frame open for
     * frame_skip() or a fresh frame_inputs(). Windows, compose state and clocks are
     * untouched. */
    heads_finite =
        pp_all_finite(p->head_erb_mask, DFN3_PREPOST_ERB_MASK_ELEMENTS) &&
        pp_all_finite(p->head_coefs, DFN3_PREPOST_COEFS_ELEMENTS);
    state_finite = dfn3_model_io_validate_state(p->io) == 0;
    if (!state_finite || !heads_finite) {
        /* validate_state already zeroed and counted a bad state; a bad head with a
         * finite state still restarts the recurrent state. */
        if (state_finite) dfn3_model_io_zero_recurrent(p->io);
        p->prepared = 0;
        return -1;
    }

    /* The compose stage advances its clock and closes the frame: the one step
     * with no undo, placed where nothing can fail after it. */
    return pp_close_frame(p, p->head_erb_mask, p->head_coefs);
}

int dfn3_prepost_frame_skip(DFN3Prepost *p) {
    if (!p || !p->frame_open) return -1;
    /* Unit ERB gain on the high bins and a unit center tap on the raw low bins:
     * every band passes through at the same source frame with the same lookahead
     * timing. Recurrent state is untouched. */
    return pp_close_frame(p, p->skip_mask, p->skip_coefs);
}

int dfn3_prepost_post_process(DFN3Prepost *p, float out_hop[DFN3_HOP_LEN],
                              int *written) {
    if (!p || !out_hop) return -1;
    if (p->io_mode != DFN3_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* a frame still awaits the model */
    /* Always a full defined hop: out_hop is zero from init/reset and written
     * wholesale by the synthesis, so warm-up hops yield silence. */
    memcpy(out_hop, p->out_hop, (size_t)DFN3_HOP_LEN * sizeof(float));
    if (written) *written = p->have_output ? DFN3_HOP_LEN : 0;
    return 0;
}

int dfn3_prepost_post_process_freq(DFN3Prepost *p, float re[DFN3_N_BINS],
                                   float im[DFN3_N_BINS], int *valid) {
    const size_t bins = (size_t)DFN3_N_BINS * sizeof(float);
    if (!p || !re || !im) return -1;
    if (p->io_mode != DFN3_IO_FREQ) return -1;
    if (p->frame_open) return -1;
    memcpy(re, p->enh_re, bins);
    memcpy(im, p->enh_im, bins);
    if (valid) *valid = p->have_output;
    return 0;
}

/* ---- the accelerator callback boundary ------------------------------ */

int dfn3_model_run_frame(const DFN3Model *model, DFN3Prepost *p) {
    DFN3PrepostInputs inputs;
    DFN3PrepostOutputs outputs;
    if (!model || !model->infer || !p || !p->frame_open ||
        dfn3_model_io_descriptor_validate(model->io_descriptor) != 0) return -1;
    if (dfn3_prepost_frame_inputs(p, &inputs, &outputs) != 0) return -1;
    if (model->infer(model->user, &inputs, &outputs) != 0) {
        return dfn3_prepost_frame_skip(p) == 0 ? 0 : -1;
    }
    if (dfn3_prepost_frame_commit(p) != 0) {
        /* A refused commit leaves the frame open with the state zeroed; take it as
         * the identity so the clocks advance once. */
        return dfn3_prepost_frame_skip(p) == 0 ? 0 : -1;
    }
    return 1;
}
