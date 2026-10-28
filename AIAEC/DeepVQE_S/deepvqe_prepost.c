/* DeepVQE-S pre/post class (contract: deepvqe_prepost.h). Composes
 * deepvqe_process.c (CCM kernel) and audiolib's AcStft/AcIstft on the AIAEC
 * grid, and adds the object lifecycle, pool carve, accelerator tensor views and
 * per-hop frame state machine. It does not use ulcnet_model_io.c (different
 * boundary), so the small pool arithmetic is repeated here. Requires
 * -ffp-contract=off. No heap in _init, no globals, no stdio. */

#include "deepvqe_prepost.h"

#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PP_ALIGN ((size_t)DEEPVQE_PREPOST_ALIGNMENT)

/* Graph tensor names in DeepVqeStateId order (the exporter's input_names[2:]);
 * two tables of static literals. */
static const char *const dv_state_names[] = {
    "state_align_key_ring",
    "state_align_value_ring",
    "state_align_score_history",
    "state_ccm_up_history",
    "state_far1_history",
    "state_far2_history",
    "h_gru",
    "state_mic1_history",
    "state_mic2_history",
    "state_mic3_history",
    "state_mic4_history",
    "state_res2_history",
    "state_res3_history",
    "state_up1_history",
    "state_up2_history",
    "state_up3_history"
};

static const char *const dv_state_names_out[] = {
    "state_align_key_ring_out",
    "state_align_value_ring_out",
    "state_align_score_history_out",
    "state_ccm_up_history_out",
    "state_far1_history_out",
    "state_far2_history_out",
    "h_gru_out",
    "state_mic1_history_out",
    "state_mic2_history_out",
    "state_mic3_history_out",
    "state_mic4_history_out",
    "state_res2_history_out",
    "state_res3_history_out",
    "state_up1_history_out",
    "state_up2_history_out",
    "state_up3_history_out"
};

/* Sized by the initialiser, so a missing entry is a compile error. */
#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(dv_state_names) / sizeof(dv_state_names[0]) ==
               DEEPVQE_STATE_COUNT,
               "dv_state_names must name every DeepVqeStateId");
_Static_assert(sizeof(dv_state_names_out) / sizeof(dv_state_names_out[0]) ==
               DEEPVQE_STATE_COUNT,
               "dv_state_names_out must name every DeepVqeStateId");
#endif
#endif

struct DeepVqePrepost {
    DeepVqePrepostDescriptor descriptor;

    int io_mode;

    /* One buffer per state tensor, bound as both the accelerator input and its _out output. */
    float *state[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];

    /* Head output and the compressed planar spectra the graph binds. */
    float *taps;        /* [DEEPVQE_TAPS_ELEMENTS]                          */
    float *mic_ri;      /* [2 * AIAEC_N_BINS] real plane, imaginary plane   */
    float *far_ri;      /* [2 * AIAEC_N_BINS] real plane, imaginary plane   */

    /* Host-owned raw-microphone spectrum ring the CCM taps convolve over (popped
     * out of the graph state by the exporter). */
    DeepVqeCcmState *ccm;

    /* DEEPVQE_IO_TIME only; NULL in DEEPVQE_IO_FREQ. Rolling (center 0) analyses
     * borrow the caller's FftHandle and window and carry their own transform
     * scratch; the synthesis state is the caller's AcIstft (cfg->istft). */
    AcStft         *ana_mic;
    AcStft         *ana_far;
    AcIstft        *synth;
    float          *out_hop;    /* [AIAEC_HOP] synthesis output; zero from init/reset, written wholesale */

    /* Per-hop frame staging: one frame, in both I/O modes. */
    float *mic_re;      /* [AIAEC_N_BINS] */
    float *mic_im;
    float *far_re;
    float *far_im;
    float *enh_re;      /* [AIAEC_N_BINS] commit/skip staging */
    float *enh_im;

    uint32_t state_generation;  /* advances whenever the state is zeroed */

    int frame_open;     /* a frame is armed and awaiting commit/skip */
    int prepared;       /* frame_inputs() armed the accelerator transaction */
    int written;        /* samples the last closed frame put in out_hop:
                         * 0 (first frame after reset) or AIAEC_HOP */

    /* _create bookkeeping; 0 for an _init instance. */
    void *heap_base;
};

/* ---- boundary shapes ------------------------------------------------- */

static int dv_valid_depth(int delay_depth) {
    return delay_depth >= DEEPVQE_PREPOST_MIN_D &&
           delay_depth <= DEEPVQE_PREPOST_MAX_D;
}

int deepvqe_prepost_state_shape(int state_id, int delay_depth,
                                int dims[DEEPVQE_STATE_MAX_RANK]) {
    int c = 0;
    int t = 0;
    int f = 0;

    if (!dims || !dv_valid_depth(delay_depth)) return -1;
    /* Single source of truth for the boundary shapes (pool carve, element counts,
     * schema checks). Channel counts follow DeepVQE_S/model.py; the frequency
     * ladder is the encoder's stride-2 halving. */
    switch (state_id) {
    case DEEPVQE_STATE_ALIGN_KEY_RING:
        c = DEEPVQE_SIM_CHANNELS;   t = delay_depth; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_ALIGN_VALUE_RING:
        c = DEEPVQE_VALUE_CHANNELS; t = delay_depth; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_ALIGN_SCORE_HISTORY:
        c = DEEPVQE_SIM_CHANNELS;   t = DEEPVQE_SCORE_WINDOW;
        f = delay_depth; break;
    case DEEPVQE_STATE_CCM_UP_HISTORY:
        c = 32; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F1; break;
    case DEEPVQE_STATE_FAR1_HISTORY:
        c = 2;  t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F0; break;
    case DEEPVQE_STATE_FAR2_HISTORY:
        c = 8;  t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F1; break;
    case DEEPVQE_STATE_H_GRU:
        /* [1,GRU_LAYERS,1,GRU_HIDDEN] */
        dims[0] = 1;
        dims[1] = DEEPVQE_GRU_LAYERS;
        dims[2] = 1;
        dims[3] = DEEPVQE_GRU_HIDDEN;
        return DEEPVQE_STATE_MAX_RANK;
    case DEEPVQE_STATE_MIC1_HISTORY:
        c = 2;  t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F0; break;
    case DEEPVQE_STATE_MIC2_HISTORY:
        c = 16; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F1; break;
    case DEEPVQE_STATE_MIC3_HISTORY:
        c = 64; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_MIC4_HISTORY:
        c = 56; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F3; break;
    case DEEPVQE_STATE_RES2_HISTORY:
        c = 32; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_RES3_HISTORY:
        c = 40; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F3; break;
    case DEEPVQE_STATE_UP1_HISTORY:
        c = 32; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_UP2_HISTORY:
        c = 40; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F3; break;
    case DEEPVQE_STATE_UP3_HISTORY:
        c = 24; t = DEEPVQE_CONV_WINDOW; f = DEEPVQE_F4; break;
    default:
        return -1;
    }
    dims[0] = 1;
    dims[1] = c;
    dims[2] = t;
    dims[3] = f;
    return DEEPVQE_STATE_MAX_RANK;
}

size_t deepvqe_prepost_state_elements(int state_id, int delay_depth) {
    int dims[DEEPVQE_STATE_MAX_RANK];
    int rank = deepvqe_prepost_state_shape(state_id, delay_depth, dims);
    size_t elements = 1;
    int index;

    if (rank < 0) return 0;
    for (index = 0; index < rank; ++index) {
        elements *= (size_t)dims[index];
    }
    return elements;
}

const char *deepvqe_prepost_state_name(int state_id) {
    if (state_id < 0 || state_id >= DEEPVQE_STATE_COUNT) return NULL;
    return dv_state_names[state_id];
}

const char *deepvqe_prepost_state_name_out(int state_id) {
    if (state_id < 0 || state_id >= DEEPVQE_STATE_COUNT) return NULL;
    return dv_state_names_out[state_id];
}

const char *deepvqe_prepost_skip_policy_name(void) {
    return "mute_fail_closed";
}

/* ---- pool arithmetic ----
 * audiolib's saturating helpers (mem_align.h): overflow pins the cursor to
 * SIZE_MAX, which MEM_SIZE_INVALID catches. */

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DEEPVQE_PREPOST_ALIGNMENT == 16u,
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
static uint32_t pp_build_hash(int io_mode, int delay_depth) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a(h, (uint32_t)AIAEC_SR);
    h = pp_fnv1a(h, (uint32_t)AIAEC_N_FFT);
    h = pp_fnv1a(h, (uint32_t)DEEPVQE_PREPOST_LAYOUT_VERSION);
    h = pp_fnv1a(h, (uint32_t)DEEPVQE_PREPOST_CARVE_VERSION);
    h = pp_fnv1a(h, (uint32_t)io_mode);
    h = pp_fnv1a(h, (uint32_t)delay_depth);
    return ac_memreq_abi_hash(h);
}

void deepvqe_framing_config(AcStftConfig *cfg, FftHandle *fft,
                            const float *window) {
    if (!cfg) return;
    ac_stft_config_default(cfg, AIAEC_N_FFT);
    cfg->ola = AC_OLA_ENV_NORM;
    cfg->fft = fft;
    cfg->window = window;
}

/* The caller's AcIstft must be the one deepvqe_framing_config() describes (the
 * model's fft size, envelope-normalised overlap-add, synthesis scale) over the
 * same window values as the analysis. */
static int pp_istft_matches(const DeepVqePrepostConfig *cfg) {
    AcStftConfig want, got;
    deepvqe_framing_config(&want, NULL, NULL);
    return cfg->istft != NULL && ac_istft_get_config(cfg->istft, &got) == 0 &&
           got.fft_size == want.fft_size && got.ola == want.ola &&
           got.ispec_scale == want.ispec_scale &&
           memcmp(got.window, cfg->window,
                  (size_t)AIAEC_N_FFT * sizeof(float)) == 0;
}

/* Walk the layout once: base==NULL sizes it, otherwise it also carves.
 * get_mem_size and init share this walk so their order agrees. */
static int pp_layout(DeepVqePrepost *p, unsigned char *base,
                     const DeepVqePrepostConfig *cfg, size_t *total) {
    size_t cursor = 0;
    const size_t bins = (size_t)AIAEC_N_BINS * sizeof(float);
    void *ptr;
    int id;

    if (pp_carve(base, &cursor, sizeof(DeepVqePrepost), &ptr) != 0) return -1;

    /* One buffer per state tensor: read by the accelerator, next value written back to the same address. */
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        size_t elements =
            deepvqe_prepost_state_elements(id, cfg->delay_depth);
        size_t bytes;
        if (elements == 0) return -1;
        bytes = ck_mul_size(elements, sizeof(float));
        if (MEM_SIZE_INVALID(bytes)) return -1;
        if (pp_carve(base, &cursor, bytes, &ptr) != 0) return -1;
        if (base && p) {
            p->state[id] = (float *)ptr;
            p->state_elements[id] = elements;
        }
    }

    if (pp_carve(base, &cursor, DEEPVQE_TAPS_ELEMENTS * sizeof(float),
                 &ptr) != 0) return -1;
    if (base && p) p->taps = (float *)ptr;
    if (pp_carve(base, &cursor, 2u * bins, &ptr) != 0) return -1;
    if (base && p) p->mic_ri = (float *)ptr;
    if (pp_carve(base, &cursor, 2u * bins, &ptr) != 0) return -1;
    if (base && p) p->far_ri = (float *)ptr;

    if (pp_carve(base, &cursor, sizeof(DeepVqeCcmState), &ptr) != 0) return -1;
    if (base && p) p->ccm = (DeepVqeCcmState *)ptr;

    if (cfg->io_mode == DEEPVQE_IO_TIME) {
        AcStftConfig sc;
        size_t stft_bytes;
        deepvqe_framing_config(&sc, cfg->fft, cfg->window);
        stft_bytes = ac_stft_get_mem_size(&sc);
        if (stft_bytes == 0) return -1;
        if (pp_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        if (base && p) {
            p->ana_mic = ac_stft_init(ptr, stft_bytes, &sc);
            if (!p->ana_mic) return -1;
        }
        if (pp_carve(base, &cursor, stft_bytes, &ptr) != 0) return -1;
        if (base && p) {
            p->ana_far = ac_stft_init(ptr, stft_bytes, &sc);
            if (!p->ana_far) return -1;
        }
        if (base && p) p->synth = cfg->istft;
        if (pp_carve(base, &cursor, (size_t)AIAEC_HOP * sizeof(float),
                     &ptr) != 0) return -1;
        if (base && p) p->out_hop = (float *)ptr;
    }

    /* Frame staging: one frame, in both modes. */
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->mic_re = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->mic_im = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->far_re = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->far_im = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->enh_re = (float *)ptr;
    if (pp_carve(base, &cursor, bins, &ptr) != 0) return -1;
    if (base && p) p->enh_im = (float *)ptr;

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

/* ---- config / descriptor / sizing ------------------------------------ */

int deepvqe_prepost_config_defaults(DeepVqePrepostConfig *cfg,
                                    int io_mode, int delay_depth) {
    if (!cfg) return -1;
    if (io_mode != DEEPVQE_IO_TIME && io_mode != DEEPVQE_IO_FREQ) return -1;
    if (!dv_valid_depth(delay_depth)) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->io_mode = io_mode;
    cfg->delay_depth = delay_depth;
    return 0;
}

int deepvqe_prepost_descriptor_default(int delay_depth,
                                       DeepVqePrepostDescriptor *descriptor) {
    if (!descriptor || !dv_valid_depth(delay_depth)) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->layout_version = DEEPVQE_PREPOST_LAYOUT_VERSION;
    descriptor->delay_depth = delay_depth;
    descriptor->sample_rate = AIAEC_SR;
    descriptor->fft_size = AIAEC_N_FFT;
    descriptor->hop_size = AIAEC_HOP;
    descriptor->spectrum_bins = AIAEC_N_BINS;
    descriptor->time_order = DEEPVQE_TIME_ORDER;
    descriptor->freq_taps = DEEPVQE_FREQ_TAPS;
    descriptor->conv_window_frames = DEEPVQE_CONV_WINDOW;
    descriptor->score_window_frames = DEEPVQE_SCORE_WINDOW;
    descriptor->gru_layers = DEEPVQE_GRU_LAYERS;
    descriptor->gru_hidden = DEEPVQE_GRU_HIDDEN;
    descriptor->state_tensor_count = DEEPVQE_STATE_COUNT;
    return 0;
}

int deepvqe_prepost_descriptor_validate(
    const DeepVqePrepostDescriptor *descriptor) {
    if (!descriptor) return -1;
    /* Field by field, not memcmp, so struct padding cannot decide the verdict.
     * delay_depth is only range-checked: it is an export-time parameter and the
     * pool is sized from it. */
    if (!dv_valid_depth(descriptor->delay_depth)) return -1;
    if (descriptor->layout_version != DEEPVQE_PREPOST_LAYOUT_VERSION ||
        descriptor->sample_rate != AIAEC_SR ||
        descriptor->fft_size != AIAEC_N_FFT ||
        descriptor->hop_size != AIAEC_HOP ||
        descriptor->spectrum_bins != AIAEC_N_BINS ||
        descriptor->time_order != DEEPVQE_TIME_ORDER ||
        descriptor->freq_taps != DEEPVQE_FREQ_TAPS ||
        descriptor->conv_window_frames != DEEPVQE_CONV_WINDOW ||
        descriptor->score_window_frames != DEEPVQE_SCORE_WINDOW ||
        descriptor->gru_layers != DEEPVQE_GRU_LAYERS ||
        descriptor->gru_hidden != DEEPVQE_GRU_HIDDEN ||
        descriptor->state_tensor_count != DEEPVQE_STATE_COUNT) {
        return -1;
    }
    return 0;
}

static int pp_check_config(const DeepVqePrepostConfig *cfg) {
    if (!cfg) return -1;
    if (cfg->io_mode != DEEPVQE_IO_TIME && cfg->io_mode != DEEPVQE_IO_FREQ)
        return -1;
    if (!dv_valid_depth(cfg->delay_depth)) return -1;
    if (cfg->io_mode == DEEPVQE_IO_TIME) {
        /* Reject-first: a wrong-size handle would break the feature-time contract. */
        if (!cfg->fft || !cfg->window) return -1;
        if (fft_get_n_freqs(cfg->fft) != AIAEC_N_BINS) return -1;
        if (!pp_istft_matches(cfg)) return -1;
    }
    return 0;
}

int deepvqe_prepost_get_mem_size(const DeepVqePrepostConfig *cfg,
                                 DeepVqePrepostMemReq *req) {
    size_t total = 0;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_layout(NULL, NULL, cfg, &total) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DEEPVQE_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = DEEPVQE_PREPOST_LAYOUT_VERSION;
    req->io_mode = (uint32_t)cfg->io_mode;
    req->build_flags_hash = pp_build_hash(cfg->io_mode, cfg->delay_depth);
    req->alignment = (uint32_t)PP_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* ---- lifecycle ------------------------------------------------------- */

DeepVqePrepost *deepvqe_prepost_init_ex(void *pool, size_t bytes,
                                        const DeepVqePrepostConfig *cfg,
                                        const DeepVqePrepostMemReq *expected) {
    DeepVqePrepostMemReq req;
    DeepVqePrepost *p;
    size_t total = 0;

    if (!pool || deepvqe_prepost_get_mem_size(cfg, &req) != 0) return NULL;
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

    /* Zero the whole control region first so a poisoned pool initialises like a
     * zeroed one and every state tensor starts at the reset value. */
    memset(pool, 0, (size_t)req.bytes);
    p = (DeepVqePrepost *)pool;
    if (deepvqe_prepost_descriptor_default(cfg->delay_depth,
                                           &p->descriptor) != 0) {
        return NULL;
    }
    p->io_mode = cfg->io_mode;

    if (pp_layout(p, (unsigned char *)pool, cfg, &total) != 0) return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    deepvqe_ccm_init(p->ccm);
    if (p->synth) ac_istft_reset(p->synth);   /* a borrowed AcIstft starts like a fresh one */
    return p;
}

DeepVqePrepost *deepvqe_prepost_init(void *pool, size_t bytes,
                                     const DeepVqePrepostConfig *cfg) {
    return deepvqe_prepost_init_ex(pool, bytes, cfg, NULL);
}

DeepVqePrepost *deepvqe_prepost_create(const DeepVqePrepostConfig *cfg) {
    DeepVqePrepostMemReq req;
    DeepVqePrepost *p;
    void *mem = NULL;

    if (deepvqe_prepost_get_mem_size(cfg, &req) != 0) return NULL;
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    mem = aligned_alloc((size_t)req.alignment, (size_t)req.bytes);
#else
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        mem = NULL;
#endif
    if (!mem) return NULL;
    p = deepvqe_prepost_init(mem, (size_t)req.bytes, cfg);
    if (!p) {
        free(mem);
        return NULL;
    }
    p->heap_base = mem;
    return p;
}

void deepvqe_prepost_destroy(DeepVqePrepost *p) {
    void *base;
    if (!p) return;
    base = p->heap_base;
    p->heap_base = NULL;   /* nothing to free for a pool instance; repeat calls are no-ops */
    if (base) free(base);
}

static void zero_state(DeepVqePrepost *p) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        memset(p->state[id], 0, p->state_elements[id] * sizeof(float));
    ++p->state_generation;
}

void deepvqe_prepost_reset(DeepVqePrepost *p) {
    if (!p) return;
    zero_state(p);
    memset(p->taps, 0, DEEPVQE_TAPS_ELEMENTS * sizeof(float));
    deepvqe_ccm_init(p->ccm);
    if (p->io_mode == DEEPVQE_IO_TIME) {
        /* The objects keep their handle and window; reset clears the history
         * and overlap state, the caller's AcIstft included. */
        ac_stft_reset(p->ana_mic);
        ac_stft_reset(p->ana_far);
        ac_istft_reset(p->synth);
        memset(p->out_hop, 0, (size_t)AIAEC_HOP * sizeof(float));
    }
    memset(p->enh_re, 0, (size_t)AIAEC_N_BINS * sizeof(float));
    memset(p->enh_im, 0, (size_t)AIAEC_N_BINS * sizeof(float));
    p->frame_open = 0;
    p->prepared = 0;
    p->written = 0;
}

/* ---- accessors ------------------------------------------------------- */

uint32_t deepvqe_prepost_state_generation(const DeepVqePrepost *p) {
    return p ? p->state_generation : 0u;
}

int deepvqe_prepost_hop_size(const DeepVqePrepost *p) {
    return p ? AIAEC_HOP : -1;
}

int deepvqe_prepost_num_bins(const DeepVqePrepost *p) {
    return p ? AIAEC_N_BINS : -1;
}

int deepvqe_prepost_io_mode(const DeepVqePrepost *p) {
    return p ? p->io_mode : -1;
}

const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(
    const DeepVqePrepost *p) {
    return p ? &p->descriptor : NULL;
}

/* ---- per-hop stages -------------------------------------------------- */

static void fill_nan(float *values, size_t elements) {
    skn_fill_f32(values, elements, NAN);
}

static int all_finite(const float *values, size_t elements) {
    return skn_all_finite_f32(values, elements);
}

int deepvqe_prepost_pre_process(DeepVqePrepost *p,
                                const float mic_hop[AIAEC_HOP],
                                const float far_hop[AIAEC_HOP]) {
    if (!p || !mic_hop || !far_hop) return -1;
    if (p->io_mode != DEEPVQE_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* previous frame neither committed nor skipped */

    /* center=False rolling analysis, one frame per hop from the first hop; both
     * streams use the same hop index (no internal skew). */
    (void)ac_stft_push_planar(p->ana_mic, mic_hop, p->mic_re, p->mic_im);
    (void)ac_stft_push_planar(p->ana_far, far_hop, p->far_re, p->far_im);

    p->frame_open = 1;
    return 1;   /* invariant: exactly one accelerator invocation per hop */
}

int deepvqe_prepost_pre_process_freq(DeepVqePrepost *p,
                                     const float mic_re[AIAEC_N_BINS],
                                     const float mic_im[AIAEC_N_BINS],
                                     const float far_re[AIAEC_N_BINS],
                                     const float far_im[AIAEC_N_BINS]) {
    const size_t bins = (size_t)AIAEC_N_BINS * sizeof(float);

    if (!p || !mic_re || !mic_im || !far_re || !far_im) return -1;
    if (p->io_mode != DEEPVQE_IO_FREQ) return -1;
    if (p->frame_open) return -1;   /* previous frame neither committed nor skipped */

    memcpy(p->mic_re, mic_re, bins);
    memcpy(p->mic_im, mic_im, bins);
    memcpy(p->far_re, far_re, bins);
    memcpy(p->far_im, far_im, bins);

    p->frame_open = 1;
    return 1;
}

/* x * |x|^(e-1) with |x| = sqrt(re^2 + im^2 + 1e-12), as passes over the planes
 * (vector sqrt and scaling, powf per element); the imaginary output plane is
 * scratch for |x| and then the scale. Unfused products (-ffp-contract=off).
 * Python reference: aiaec_common.compressed_ri_feature. */
static void compress_planar(const float *re, const float *im, float *out) {
    float *out_re = out;
    float *out_im = out + AIAEC_N_BINS;
    const float exponent = DEEPVQE_COMPRESSION_EXP - 1.0f;
    int k;

    skn_mag_eps_f32(out_im, re, im, AIAEC_N_BINS, 1e-12f);
    for (k = 0; k < AIAEC_N_BINS; ++k)
        out_im[k] = powf(out_im[k], exponent);
    skn_mul_f32(out_re, re, out_im, AIAEC_N_BINS);
    skn_mul_f32(out_im, im, out_im, AIAEC_N_BINS);
}

int deepvqe_prepost_frame_inputs(DeepVqePrepost *p,
                                 DeepVqePrepostInputs *inputs,
                                 DeepVqePrepostOutputs *outputs) {
    int id;

    if (!p || !inputs || !outputs || !p->frame_open) return -1;

    /* Host-side front end: power-law compression of both spectra into
     * [1,2,1,BINS] planar tensors. The raw mic spectrum stays in mic_re/mic_im
     * for the CCM ring. */
    compress_planar(p->mic_re, p->mic_im, p->mic_ri);
    compress_planar(p->far_re, p->far_im, p->far_ri);

    /* Arms the transaction and NaN-fills the head output; state tensors are not
     * filled (they are the accelerator's input too). */
    fill_nan(p->taps, DEEPVQE_TAPS_ELEMENTS);

    memset(inputs, 0, sizeof(*inputs));
    memset(outputs, 0, sizeof(*outputs));
    inputs->mic = p->mic_ri;
    inputs->far = p->far_ri;
    inputs->spectrum_ri_elements = 2u * (size_t)AIAEC_N_BINS;
    outputs->taps = p->taps;
    outputs->taps_elements = DEEPVQE_TAPS_ELEMENTS;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        inputs->state[id] = p->state[id];
        inputs->state_elements[id] = p->state_elements[id];
        outputs->state_out[id] = p->state[id];
        outputs->state_elements[id] = p->state_elements[id];
    }
    p->prepared = 1;
    return 0;
}

/* Push the finished frame through the synthesis (TIME) or stage it (FREQ) and
 * close it. Cannot fail: the synthesis reports 0 samples (first frame) or a full
 * AIAEC_HOP, so a failed commit leaves the CCM ring untouched. */
static void pp_close_frame(DeepVqePrepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    if (p->io_mode == DEEPVQE_IO_TIME) {
        p->written = ac_istft_push_planar(p->synth, p->enh_re, p->enh_im,
                                         p->out_hop);
    }
}

/* Finite check over what the graph wrote into one state tensor: only the newest
 * frame of a window (slot 0 of an attention ring, last frame of a conv or score
 * window) and h_gru whole. Older frames were checked when new, and a non-finite
 * value left in them reaches `taps` on a later frame. */
static int newest_frame_finite(const DeepVqePrepost *p, int id) {
    int dims[DEEPVQE_STATE_MAX_RANK];
    size_t channels;
    size_t frames;
    size_t width;
    size_t newest;

    if (id == DEEPVQE_STATE_H_GRU)
        return all_finite(p->state[id], p->state_elements[id]);
    if (deepvqe_prepost_state_shape(id, p->descriptor.delay_depth, dims) < 0)
        return 0;
    channels = (size_t)dims[1];
    frames = (size_t)dims[2];
    width = (size_t)dims[3];
    newest = (id == DEEPVQE_STATE_ALIGN_KEY_RING ||
              id == DEEPVQE_STATE_ALIGN_VALUE_RING) ? 0u : frames - 1u;
    return skn_all_finite_rows_f32(p->state[id] + newest * width, channels,
                           frames * width, width);
}

int deepvqe_prepost_frame_commit(DeepVqePrepost *p) {
    int ok;
    int id;

    if (!p || !p->prepared) return -1;   /* prepared implies frame_open */

    ok = all_finite(p->taps, DEEPVQE_TAPS_ELEMENTS);
    for (id = 0; ok && id < DEEPVQE_STATE_COUNT; ++id)
        ok = newest_frame_finite(p, id);
    if (!ok) {
        /* State was overwritten in place: restart the recurrence cold. The CCM
         * ring is not pushed and the frame stays open for frame_skip(). */
        zero_state(p);
        p->prepared = 0;
        return -1;
    }

    /* The CCM ring is host state: deepvqe_ccm_process pushes this frame's raw
     * microphone spectrum and convolves the taps over (t, t-1, t-2). It advances
     * only on a validated frame. */
    deepvqe_ccm_process(
        p->ccm, p->mic_re, p->mic_im,
        (const float (*)[DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2])p->taps,
        p->enh_re, p->enh_im);
    pp_close_frame(p);
    return 0;
}

int deepvqe_prepost_frame_skip(DeepVqePrepost *p) {
    const size_t bins = (size_t)AIAEC_N_BINS * sizeof(float);
    if (!p || !p->frame_open) return -1;
    /* Fail closed: stream 0 is the raw microphone, so pass-through would emit the
     * uncancelled echo (see deepvqe_prepost.h). Emit silence; the CCM ring is not
     * pushed and the state stays as the accelerator left it. */
    memset(p->enh_re, 0, bins);
    memset(p->enh_im, 0, bins);
    pp_close_frame(p);
    return 0;
}

int deepvqe_prepost_post_process(DeepVqePrepost *p,
                                 float out_hop[AIAEC_HOP], int *written) {
    if (!p || !out_hop) return -1;
    if (p->io_mode != DEEPVQE_IO_TIME) return -1;
    if (p->frame_open) return -1;   /* a frame is still awaiting the model */
    /* Always a full defined hop: out_hop is zero from init/reset and written
     * wholesale, so the warm-up frame yields silence. */
    memcpy(out_hop, p->out_hop, (size_t)AIAEC_HOP * sizeof(float));
    if (written) *written = p->written;
    return 0;
}

int deepvqe_prepost_post_process_freq(DeepVqePrepost *p,
                                      float re[AIAEC_N_BINS],
                                      float im[AIAEC_N_BINS]) {
    const size_t bins = (size_t)AIAEC_N_BINS * sizeof(float);
    if (!p || !re || !im) return -1;
    if (p->io_mode != DEEPVQE_IO_FREQ) return -1;
    if (p->frame_open) return -1;
    memcpy(re, p->enh_re, bins);
    memcpy(im, p->enh_im, bins);
    return 0;
}
