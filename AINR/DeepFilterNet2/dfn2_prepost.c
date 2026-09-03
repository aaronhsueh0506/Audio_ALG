/* DeepFilterNet2 pre/post class. Contract: see dfn2_prepost.h.
 *
 * This file OWNS no signal processing of its own. It composes the two
 * parity-tested translation units -- dfn2_process.c (streaming center=False
 * sqrt-Hann STFT/WOLA, ERB and complex features, mask expansion, deep filter,
 * attenuation limit) and dfn2_model_io.c (graph feature windows and recurrent
 * caches) -- and adds only the object lifecycle, the pool carve and the
 * per-hop frame state machine. Neither of those files changes, so both of
 * their standalone parity builds keep linking.
 *
 * Constraint inherited from both: -ffp-contract=off. No heap in _init, no
 * globals, no stdio. */

#include "dfn2_prepost.h"

#include "mem_align.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PP_ALIGN ((size_t)DFN2_PREPOST_ALIGNMENT)

/* Frames of end-to-end model lookahead. A cascade, so the two lookaheads add
 * rather than overlap -- see dfn2_process.h's dfn2_compose_stream() note. */
#define PP_MODEL_LOOKAHEAD (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD)

struct DFN2Prepost {
    DFN2State        *dsp;
    DFN2ModelIOState *io;

    int   io_mode;
    float atten_lim_db;

    /* Borrowed, never created or destroyed here. */
    FftHandle   *fft;        /* NULL in DFN2_IO_FREQ                        */
    const float *erb_fwd;
    const float *erb_inv;
    /* Copied into DFN2State's embedded table at init and at every reset;
     * kept here only so reset can reapply it. See the header's window note. */
    const float *window;

    /* Per-hop staging. */
    float *spec_re;     /* [DFN2_N_BINS]  current frame, original scale     */
    float *spec_im;
    float *feat_erb;    /* [DFN2_N_ERB]                                     */
    float *feat_spec;   /* [2][DFN2_DF_BINS], two contiguous segments       */
    float *enh_re;      /* [DFN2_N_BINS]  compose output                    */
    float *enh_im;
    float *out_hop;     /* [DFN2_HOP_LEN], DFN2_IO_TIME only                */

    /* Accelerator-writable boundary. The next-state tensors are a full
     * shadow of DFN2ModelIOState: the graph writes them, and they must be
     * validated in full before a single byte of the live state moves. */
    float *head_erb_mask;   /* [DFN2_N_ERB]                                 */
    float *head_coefs;      /* [DFN2_PREPOST_COEFS_ELEMENTS]                */
    float *head_alpha;      /* [1]                                          */
    float *next_encoder;    /* [DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS]       */
    float *next_erb;        /* [DFN2_PREPOST_ERB_HIDDEN_ELEMENTS]           */
    float *next_df;         /* [DFN2_PREPOST_DF_HIDDEN_ELEMENTS]            */
    float *next_convp;      /* [DFN2_PREPOST_CONVP_HISTORY_ELEMENTS]        */

    /* Identity heads for frame_skip: a unit band mask and zero coefficients.
     * Written once at init; dfn2_compose_stream refuses NULL heads, so the
     * pass-through has to be expressed as heads rather than as a bypass. */
    float *skip_mask;       /* [DFN2_N_ERB], all 1.0f                       */
    float *skip_coefs;      /* [DFN2_PREPOST_COEFS_ELEMENTS], all 0.0f      */

    int       frame_open;    /* a frame awaits commit or skip               */
    int       prepared;      /* frame_inputs() armed the accelerator
                              * transaction for the open frame              */
    int       have_output;   /* the compose stage emitted this hop          */
    long long output_frame;  /* source frame of the last emission, -1 none  */

    /* _create bookkeeping; 0 for an _init instance. */
    void *heap_base;
};

/* ---- pool arithmetic ---------------------------------------------------
 * audio_common's saturating helpers (mem_align.h): an overflow anywhere in
 * the walk pins the cursor to SIZE_MAX, which MEM_SIZE_INVALID catches. */

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DFN2_PREPOST_ALIGNMENT == 16u,
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

/* The build identity a pool is only valid for. io_mode is IN here because the
 * two modes carve different regions; the feature version is IN because a pool
 * handed across a checkpoint-contract change should be refused, not reused. */
static uint32_t pp_build_hash(int io_mode) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_SR);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_N_FFT);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_HOP_LEN);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_N_ERB);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_DF_BINS);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_DF_ORDER);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MASK_LOOKAHEAD);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_DF_LOOKAHEAD);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MODEL_IO_LAYOUT_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MODEL_INPUT_FRAMES);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MODEL_GRU_HIDDEN);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MODEL_ENCODER_CHANNELS);
    h = pp_fnv1a_u32(h, (uint32_t)DFN2_MODEL_DF_PATHWAY_HISTORY);
    h = pp_fnv1a_str(h, DFN2_FEATURE_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)io_mode);
    return h;
}

/* Walk the layout once. base==NULL sizes it; base!=NULL also carves it.
 * get_mem_size and init MUST walk in this same order -- that is the whole
 * reason it is one function. */
static int pp_layout(DFN2Prepost *p, unsigned char *base, int io_mode,
                     size_t *total) {
    static const size_t float_counts[] = {
        DFN2_N_BINS,                             /* spec_re       */
        DFN2_N_BINS,                             /* spec_im       */
        DFN2_N_ERB,                              /* feat_erb      */
        2u * DFN2_DF_BINS,                       /* feat_spec     */
        DFN2_N_BINS,                             /* enh_re        */
        DFN2_N_BINS,                             /* enh_im        */
        DFN2_PREPOST_ERB_MASK_ELEMENTS,          /* head_erb_mask */
        DFN2_PREPOST_COEFS_ELEMENTS,             /* head_coefs    */
        DFN2_PREPOST_ALPHA_ELEMENTS,             /* head_alpha    */
        DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS,    /* next_encoder  */
        DFN2_PREPOST_ERB_HIDDEN_ELEMENTS,        /* next_erb      */
        DFN2_PREPOST_DF_HIDDEN_ELEMENTS,         /* next_df       */
        DFN2_PREPOST_CONVP_HISTORY_ELEMENTS,     /* next_convp    */
        DFN2_PREPOST_ERB_MASK_ELEMENTS,          /* skip_mask     */
        DFN2_PREPOST_COEFS_ELEMENTS              /* skip_coefs    */
    };
    const size_t n_float_regions =
        sizeof(float_counts) / sizeof(float_counts[0]);
    size_t cursor = 0;
    size_t i;
    void *ptr;
    float **slots[sizeof(float_counts) / sizeof(float_counts[0])];

    if (pp_carve(base, &cursor, sizeof(DFN2Prepost), &ptr) != 0) return -1;
    if (pp_carve(base, &cursor, sizeof(DFN2State), &ptr) != 0) return -1;
    if (base && p) p->dsp = (DFN2State *)ptr;
    if (pp_carve(base, &cursor, sizeof(DFN2ModelIOState), &ptr) != 0)
        return -1;
    if (base && p) p->io = (DFN2ModelIOState *)ptr;

    /* DFN2_IO_TIME only: the output hop staging. Everything else the framing
     * needs is embedded in DFN2State by value (see the header's honest note
     * on how little the two modes differ). */
    if (io_mode == DFN2_IO_TIME) {
        if (pp_carve(base, &cursor, (size_t)DFN2_HOP_LEN * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) p->out_hop = (float *)ptr;
    }

    if (p) {
        slots[0]  = &p->spec_re;       slots[1]  = &p->spec_im;
        slots[2]  = &p->feat_erb;      slots[3]  = &p->feat_spec;
        slots[4]  = &p->enh_re;        slots[5]  = &p->enh_im;
        slots[6]  = &p->head_erb_mask; slots[7]  = &p->head_coefs;
        slots[8]  = &p->head_alpha;    slots[9]  = &p->next_encoder;
        slots[10] = &p->next_erb;      slots[11] = &p->next_df;
        slots[12] = &p->next_convp;    slots[13] = &p->skip_mask;
        slots[14] = &p->skip_coefs;
    }
    for (i = 0; i < n_float_regions; ++i) {
        if (pp_carve(base, &cursor, float_counts[i] * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) *slots[i] = (float *)ptr;
    }

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

/* ---- config / sizing ------------------------------------------------- */

int dfn2_prepost_config_defaults(DFN2PrepostConfig *cfg, int io_mode) {
    if (!cfg) return -1;
    if (io_mode != DFN2_IO_TIME && io_mode != DFN2_IO_FREQ) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->io_mode = io_mode;
    cfg->atten_lim_db = 0.0f;
    return 0;
}

static int pp_check_config(const DFN2PrepostConfig *cfg) {
    if (!cfg) return -1;
    if (cfg->io_mode != DFN2_IO_TIME && cfg->io_mode != DFN2_IO_FREQ)
        return -1;
    /* Required in BOTH modes: mask expansion and the ERB feature branch run
     * on the spectrum side, not the framing side. */
    if (!cfg->erb_fwd || !cfg->erb_inv) return -1;
    if (!isfinite(cfg->atten_lim_db)) return -1;
    if (cfg->io_mode == DFN2_IO_TIME) {
        /* Reject-first: a handle of the wrong size would silently break the
         * grid rather than fail. */
        if (!cfg->fft) return -1;
        if (fft_get_n_freqs(cfg->fft) != DFN2_N_BINS) return -1;
    }
    return 0;
}

int dfn2_prepost_get_mem_size(const DFN2PrepostConfig *cfg,
                              DFN2PrepostMemReq *req) {
    size_t total = 0;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_layout(NULL, NULL, cfg->io_mode, &total) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DFN2_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = (uint32_t)DFN2_MODEL_IO_LAYOUT_VERSION;
    req->io_mode = (uint32_t)cfg->io_mode;
    req->build_flags_hash = pp_build_hash(cfg->io_mode);
    req->alignment = (uint32_t)PP_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* ---- lifecycle ------------------------------------------------------- */

/* Bring DFN2State and DFN2ModelIOState to their init values and reapply the
 * borrowed tables. dfn2_state_init() memsets and rebuilds the window, so the
 * caller's window override has to be replayed after it, every time. */
static void pp_reset_states(DFN2Prepost *p) {
    dfn2_state_init(p->dsp, p->fft);
    if (p->window) {
        memcpy(p->dsp->window, p->window,
               (size_t)DFN2_WIN_LEN * sizeof(float));
    }
    dfn2_set_erb_matrices(p->dsp, p->erb_fwd, p->erb_inv);
    dfn2_model_io_init(p->io);
}

/* Per-hop bookkeeping only. The staging buffers are deliberately NOT
 * cleared here: they are zero from init/reset, the compose stage returns
 * before writing them on the two warm-up hops, and every later hop writes
 * them wholesale -- so a per-hop clear would be 6 KB of zeroing that the
 * same hop immediately overwrites. */
static void pp_clear_frame(DFN2Prepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    p->have_output = 0;
}

static void pp_zero_staging(DFN2Prepost *p) {
    memset(p->enh_re, 0, (size_t)DFN2_N_BINS * sizeof(float));
    memset(p->enh_im, 0, (size_t)DFN2_N_BINS * sizeof(float));
    if (p->io_mode == DFN2_IO_TIME)
        memset(p->out_hop, 0, (size_t)DFN2_HOP_LEN * sizeof(float));
}

DFN2Prepost *dfn2_prepost_init_ex(void *pool, size_t bytes,
                                  const DFN2PrepostConfig *cfg,
                                  const DFN2PrepostMemReq *expected) {
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    size_t total = 0;
    int i;

    if (!pool || dfn2_prepost_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        /* Stale-pool gate: a pool sized by another build or io_mode is
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

    /* Whole control region zeroed first, so a poisoned pool initialises
     * identically to a zeroed one. */
    memset(pool, 0, (size_t)req.bytes);
    p = (DFN2Prepost *)pool;
    p->io_mode = cfg->io_mode;
    p->atten_lim_db = cfg->atten_lim_db;
    p->fft = (cfg->io_mode == DFN2_IO_TIME) ? cfg->fft : NULL;
    p->window = cfg->window;
    p->erb_fwd = cfg->erb_fwd;
    p->erb_inv = cfg->erb_inv;

    if (pp_layout(p, (unsigned char *)pool, cfg->io_mode, &total) != 0)
        return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    for (i = 0; i < DFN2_N_ERB; ++i) p->skip_mask[i] = 1.0f;
    memset(p->skip_coefs, 0,
           (size_t)DFN2_PREPOST_COEFS_ELEMENTS * sizeof(float));

    pp_reset_states(p);   /* the pool memset above already zeroed staging */
    pp_clear_frame(p);
    p->output_frame = -1;
    return p;
}

DFN2Prepost *dfn2_prepost_init(void *pool, size_t bytes,
                               const DFN2PrepostConfig *cfg) {
    return dfn2_prepost_init_ex(pool, bytes, cfg, NULL);
}

DFN2Prepost *dfn2_prepost_create(const DFN2PrepostConfig *cfg) {
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *mem = NULL;

    if (dfn2_prepost_get_mem_size(cfg, &req) != 0) return NULL;
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    mem = aligned_alloc((size_t)req.alignment, (size_t)req.bytes);
#else
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        mem = NULL;
#endif
    if (!mem) return NULL;
    p = dfn2_prepost_init(mem, (size_t)req.bytes, cfg);
    if (!p) {
        free(mem);
        return NULL;
    }
    p->heap_base = mem;
    return p;
}

void dfn2_prepost_destroy(DFN2Prepost *p) {
    void *base;
    if (!p) return;
    base = p->heap_base;
    p->heap_base = NULL;   /* a pool instance owns nothing to free and
                            * stays usable: repeat calls on it are no-ops */
    if (base) free(base);
}

void dfn2_prepost_reset(DFN2Prepost *p) {
    if (!p) return;
    pp_reset_states(p);
    pp_zero_staging(p);
    pp_clear_frame(p);
    p->output_frame = -1;
}

/* ---- accessors ------------------------------------------------------- */

int dfn2_prepost_hop_size(const DFN2Prepost *p) {
    return p ? DFN2_HOP_LEN : -1;
}

int dfn2_prepost_num_bins(const DFN2Prepost *p) {
    return p ? DFN2_N_BINS : -1;
}

int dfn2_prepost_io_mode(const DFN2Prepost *p) {
    return p ? p->io_mode : -1;
}

int dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p) {
    return p ? PP_MODEL_LOOKAHEAD : -1;
}

int dfn2_prepost_layout_version(const DFN2Prepost *p) {
    return p ? DFN2_MODEL_IO_LAYOUT_VERSION : -1;
}

int dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd,
                                  const float *erb_inv) {
    if (!p || !erb_fwd || !erb_inv) return -1;
    /* Between hops only: a swap must not land inside an open transaction,
     * between the features just taken through erb_fwd and the mask
     * expansion still pending through erb_inv. Atomicity per hop; per
     * source-frame consistency across the model lookahead is a stream
     * boundary matter (see the header). */
    if (p->frame_open) return -1;
    p->erb_fwd = erb_fwd;
    p->erb_inv = erb_inv;
    dfn2_set_erb_matrices(p->dsp, erb_fwd, erb_inv);
    return 0;
}

int dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db) {
    if (!p || !isfinite(atten_lim_db)) return -1;
    if (p->frame_open) return -1;   /* between hops only, like the ERB pair */
    p->atten_lim_db = atten_lim_db;
    return 0;
}

int dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame) {
    if (!p || !frame || p->output_frame < 0) return -1;
    *frame = p->output_frame;
    return 0;
}

/* ---- per-hop stages -------------------------------------------------- */

static int pp_all_finite(const float *values, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static void pp_fill_nan(float *values, size_t count) {
    const float nan_value = (float)NAN;
    size_t i;
    for (i = 0; i < count; ++i) values[i] = nan_value;
}

/* Features, graph window, compose clock. Shared by both pre_process entry
 * points so the two modes cannot drift in what they advance. Returns the
 * number of accelerator invocations this hop needs: 0 or 1. */
static int pp_begin_frame(DFN2Prepost *p) {
    int heads_needed;

    pp_clear_frame(p);
    dfn2_compute_features(p->dsp, p->spec_re, p->spec_im, p->feat_erb,
                          p->feat_spec);
    heads_needed = dfn2_model_io_push_features(
        p->io, p->feat_erb,
        (const float (*)[DFN2_DF_BINS])p->feat_spec);
    if (heads_needed < 0) return -1;

    if (heads_needed == 0) {
        /* Left warm-up. The graph cannot produce heads for a frame with no
         * right-hand neighbour yet, so the compose clock is advanced here
         * with heads_valid=0. dfn2_compose_stream() enforces that this is
         * legal exactly while current < DFN2_MASK_LOOKAHEAD -- if the two
         * counters ever disagree it returns -1 and we surface it rather than
         * pairing a mask with the wrong frame. */
        if (dfn2_compose_stream(p->dsp, p->spec_re, p->spec_im, 0, NULL, NULL,
                                0.0f, p->atten_lim_db, p->enh_re, p->enh_im,
                                NULL) != 0) {
            return -1;
        }
        return 0;
    }
    p->frame_open = 1;
    return 1;
}

int dfn2_prepost_pre_process(DFN2Prepost *p,
                             const float in_hop[DFN2_HOP_LEN]) {
    if (!p || !in_hop) return -1;
    if (p->io_mode != DFN2_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* previous frame never closed */
    dfn2_analysis(p->dsp, in_hop, p->spec_re, p->spec_im);
    return pp_begin_frame(p);
}

int dfn2_prepost_pre_process_freq(DFN2Prepost *p,
                                  const float spec_re[DFN2_N_BINS],
                                  const float spec_im[DFN2_N_BINS]) {
    const size_t bins = (size_t)DFN2_N_BINS * sizeof(float);
    if (!p || !spec_re || !spec_im) return -1;
    if (p->io_mode != DFN2_IO_FREQ) return -1;
    if (p->frame_open) return -1;
    memcpy(p->spec_re, spec_re, bins);
    memcpy(p->spec_im, spec_im, bins);
    return pp_begin_frame(p);
}

int dfn2_prepost_frame_inputs(DFN2Prepost *p, DFN2PrepostInputs *inputs,
                              DFN2PrepostOutputs *outputs) {
    if (!p || !inputs || !outputs || !p->frame_open) return -1;

    memset(inputs, 0, sizeof(*inputs));
    inputs->erb_window =
        (const float (*)[DFN2_N_ERB])p->io->erb_window;
    inputs->spec_window =
        (const float (*)[DFN2_MODEL_INPUT_FRAMES][DFN2_DF_BINS])
            p->io->spec_window;
    inputs->encoder_gru_hidden =
        (const float (*)[DFN2_MODEL_GRU_HIDDEN])p->io->encoder_gru_hidden;
    inputs->erb_gru_hidden =
        (const float (*)[DFN2_MODEL_GRU_HIDDEN])p->io->erb_gru_hidden;
    inputs->df_gru_hidden =
        (const float (*)[DFN2_MODEL_GRU_HIDDEN])p->io->df_gru_hidden;
    inputs->df_convp_history =
        (const float (*)[DFN2_MODEL_DF_PATHWAY_HISTORY][DFN2_DF_BINS])
            p->io->df_convp_history;
    inputs->erb_window_elements = DFN2_PREPOST_ERB_WINDOW_ELEMENTS;
    inputs->spec_window_elements = DFN2_PREPOST_SPEC_WINDOW_ELEMENTS;
    inputs->encoder_gru_hidden_elements =
        DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS;
    inputs->erb_gru_hidden_elements = DFN2_PREPOST_ERB_HIDDEN_ELEMENTS;
    inputs->df_gru_hidden_elements = DFN2_PREPOST_DF_HIDDEN_ELEMENTS;
    inputs->df_convp_history_elements = DFN2_PREPOST_CONVP_HISTORY_ELEMENTS;

    /* NaN-fill every writable output, so a caller that asks twice still gets
     * a clean boundary rather than a half-written one, and so a partial
     * accelerator write fails commit instead of replaying last frame. */
    pp_fill_nan(p->head_erb_mask, DFN2_PREPOST_ERB_MASK_ELEMENTS);
    pp_fill_nan(p->head_coefs, DFN2_PREPOST_COEFS_ELEMENTS);
    pp_fill_nan(p->head_alpha, DFN2_PREPOST_ALPHA_ELEMENTS);
    pp_fill_nan(p->next_encoder, DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS);
    pp_fill_nan(p->next_erb, DFN2_PREPOST_ERB_HIDDEN_ELEMENTS);
    pp_fill_nan(p->next_df, DFN2_PREPOST_DF_HIDDEN_ELEMENTS);
    pp_fill_nan(p->next_convp, DFN2_PREPOST_CONVP_HISTORY_ELEMENTS);

    memset(outputs, 0, sizeof(*outputs));
    outputs->erb_mask = p->head_erb_mask;
    outputs->coefs = p->head_coefs;
    outputs->alpha = p->head_alpha;
    outputs->encoder_gru_hidden_next =
        (float (*)[DFN2_MODEL_GRU_HIDDEN])p->next_encoder;
    outputs->erb_gru_hidden_next =
        (float (*)[DFN2_MODEL_GRU_HIDDEN])p->next_erb;
    outputs->df_gru_hidden_next =
        (float (*)[DFN2_MODEL_GRU_HIDDEN])p->next_df;
    outputs->df_convp_history_next =
        (float (*)[DFN2_MODEL_DF_PATHWAY_HISTORY][DFN2_DF_BINS])
            p->next_convp;
    outputs->erb_mask_elements = DFN2_PREPOST_ERB_MASK_ELEMENTS;
    outputs->coefs_elements = DFN2_PREPOST_COEFS_ELEMENTS;
    outputs->alpha_elements = DFN2_PREPOST_ALPHA_ELEMENTS;
    outputs->encoder_gru_hidden_elements =
        DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS;
    outputs->erb_gru_hidden_elements = DFN2_PREPOST_ERB_HIDDEN_ELEMENTS;
    outputs->df_gru_hidden_elements = DFN2_PREPOST_DF_HIDDEN_ELEMENTS;
    outputs->df_convp_history_elements = DFN2_PREPOST_CONVP_HISTORY_ELEMENTS;
    p->prepared = 1;
    return 0;
}

/* Run the compose stage with the given heads and, in DFN2_IO_TIME, the
 * synthesis. Closes the frame ON SUCCESS ONLY. The compose clock advances
 * exactly once per hop -- here or in pp_begin_frame's warm-up branch, never
 * both. */
static int pp_close_frame(DFN2Prepost *p, const float *erb_mask,
                          const float *coefs, float alpha) {
    long long frame = 0;
    int emitted;

    /* The compose stage validates every precondition -- pointers, the
     * heads' finiteness and its own frame-clock alignment -- before its
     * first write, so a -1 here leaves the compose state untouched and the
     * frame OPEN for the caller to take with frame_skip(). */
    emitted = dfn2_compose_stream(p->dsp, p->spec_re, p->spec_im, 1, erb_mask,
                                  coefs, alpha, p->atten_lim_db, p->enh_re,
                                  p->enh_im, &frame);
    if (emitted < 0) return -1;
    p->frame_open = 0;
    p->prepared = 0;
    p->have_output = emitted;
    if (!emitted) return 0;   /* still inside the lookahead warm-up */

    p->output_frame = frame;
    if (p->io_mode == DFN2_IO_TIME) {
        /* One synthesis push per emitted frame, and none on the warm-up hops.
         * Pushing the warm-up's zero spectrum would happen to be harmless
         * today -- a zero block leaves the OLA buffer zero -- but the
         * invariant worth keeping is the one-to-one pairing, not that
         * accident. */
        dfn2_synthesis(p->dsp, p->enh_re, p->enh_im, p->out_hop);
    }
    return 0;
}

int dfn2_prepost_frame_commit(DFN2Prepost *p) {
    if (!p || !p->prepared) return -1;   /* prepared implies frame_open */

    /* Reject-first, in full, before anything moves. Every accelerator
     * output is validated up front -- the three heads AND the four
     * next-state tensors -- and a refusal disarms the transaction: the
     * caller either takes frame_skip() or re-runs the accelerator through
     * a fresh frame_inputs(). Nothing persistent has been touched. */
    if (!pp_all_finite(p->head_erb_mask, DFN2_PREPOST_ERB_MASK_ELEMENTS) ||
        !pp_all_finite(p->head_coefs, DFN2_PREPOST_COEFS_ELEMENTS) ||
        !pp_all_finite(p->head_alpha, DFN2_PREPOST_ALPHA_ELEMENTS) ||
        !pp_all_finite(p->next_encoder, DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS) ||
        !pp_all_finite(p->next_erb, DFN2_PREPOST_ERB_HIDDEN_ELEMENTS) ||
        !pp_all_finite(p->next_df, DFN2_PREPOST_DF_HIDDEN_ELEMENTS) ||
        !pp_all_finite(p->next_convp, DFN2_PREPOST_CONVP_HISTORY_ELEMENTS)) {
        p->prepared = 0;
        return -1;
    }

    /* Recoverable step first, irreversible step last. The recurrent commit
     * touches only p->io and, after the preflight, has nothing left to
     * refuse (dfn2_model_io_commit_state() re-validates the same four
     * tensors and then only copies); were it ever to refuse, the frame is
     * still open and the compose clock has not moved. The compose stage then
     * advances that clock and closes the frame -- the one step with no undo,
     * placed where nothing can fail after it. The two touch disjoint state,
     * so this order is byte-identical to the reverse. */
    if (dfn2_model_io_commit_state(
            p->io,
            (const float (*)[DFN2_MODEL_GRU_HIDDEN])p->next_encoder,
            (const float (*)[DFN2_MODEL_GRU_HIDDEN])p->next_erb,
            (const float (*)[DFN2_MODEL_GRU_HIDDEN])p->next_df,
            (const float (*)[DFN2_MODEL_DF_PATHWAY_HISTORY][DFN2_DF_BINS])
                p->next_convp) != 0) {
        return -1;
    }
    return pp_close_frame(p, p->head_erb_mask, p->head_coefs,
                          p->head_alpha[0]);
}

int dfn2_prepost_frame_skip(DFN2Prepost *p) {
    if (!p || !p->frame_open) return -1;
    /* Identity: a unit band mask expands to unit bin gain through the shipped
     * partition-of-unity erb_inv, and alpha 0 selects the masked residual
     * rather than the deep filter, so the noisy spectrum passes through and
     * the attenuation limit becomes a no-op on it. The recurrent state is not
     * stepped -- commit_state is simply not called -- while the deep-filter
     * and compose clocks still advance. */
    return pp_close_frame(p, p->skip_mask, p->skip_coefs, 0.0f);
}

int dfn2_prepost_post_process(DFN2Prepost *p, float out_hop[DFN2_HOP_LEN],
                              int *written) {
    if (!p || !out_hop) return -1;
    if (p->io_mode != DFN2_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* a frame still awaits the model */
    /* Always a full, defined hop: out_hop is zero from init/reset and only
     * ever overwritten wholesale by the synthesis, so the warm-up hops hand
     * over silence, never stale samples. */
    memcpy(out_hop, p->out_hop, (size_t)DFN2_HOP_LEN * sizeof(float));
    if (written) *written = p->have_output ? DFN2_HOP_LEN : 0;
    return 0;
}

int dfn2_prepost_post_process_freq(DFN2Prepost *p, float re[DFN2_N_BINS],
                                   float im[DFN2_N_BINS], int *valid) {
    const size_t bins = (size_t)DFN2_N_BINS * sizeof(float);
    if (!p || !re || !im) return -1;
    if (p->io_mode != DFN2_IO_FREQ) return -1;
    if (p->frame_open) return -1;
    memcpy(re, p->enh_re, bins);
    memcpy(im, p->enh_im, bins);
    if (valid) *valid = p->have_output;
    return 0;
}
