/* DeepVQE-S pre/post class. Contract: see deepvqe_prepost.h.
 *
 * This file OWNS no signal processing of its own. It composes the two
 * parity-tested translation units -- aiaec_process.c (the centered sqrt-Hann
 * STFT/WOLA it shares with Align-ULCNet) and deepvqe_process.c (the CCM
 * kernel) -- and adds only the object lifecycle, the pool carve, the
 * accelerator tensor views and the per-hop frame state machine. Neither
 * file's existing entry points change, so both of their standalone parity
 * builds keep linking.
 *
 * It deliberately does NOT use ulcnet_model_io.c: DeepVQE-S's boundary is a
 * different one (compressed planar spectra in, full next state out) and
 * pulling that TU in would bind this model to Align-ULCNet's layout version. The small amount of pool
 * arithmetic it would have shared is duplicated below instead.
 *
 * Constraint inherited from both: -ffp-contract=off. No heap in _init, no
 * globals, no stdio. */

#include "deepvqe_prepost.h"

#include "mem_align.h"
#include "simd_kernel_nn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PP_ALIGN ((size_t)DEEPVQE_PREPOST_ALIGNMENT)

/* Graph tensor names, in DeepVqeStateId order == the exporter's
 * input_names[2:]. Two tables rather than one plus a runtime "_out" suffix
 * concatenation: this file has no scratch string storage and must return
 * literals with static lifetime. */
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

/* Sized by the initialiser, not by the enum, so a missing entry is a
 * compile error here rather than a NULL name at runtime. */
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

    /* Borrowed, never created or destroyed here. NULL in DEEPVQE_IO_FREQ. */
    FftHandle   *fft;
    const float *window;

    /* One buffer per state tensor, bound as BOTH the accelerator's input and
     * its `_out` output for the life of the instance: the frame-t output is
     * already the frame-t+1 input, so nothing is copied or swapped. */
    float *state[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];

    /* Head output and the compressed planar spectra the graph binds. */
    float *taps;        /* [DEEPVQE_TAPS_ELEMENTS]                          */
    float *mic_ri;      /* [2 * AIAEC_N_BINS] real plane, imaginary plane   */
    float *far_ri;      /* [2 * AIAEC_N_BINS] real plane, imaginary plane   */

    /* Host-owned raw-microphone spectrum ring the CCM taps convolve over.
     * The exporter pops it out of the graph's state on purpose
     * (_streaming_export.py: stream_state.pop('spec_ring')). */
    DeepVqeCcmState *ccm;

    /* DEEPVQE_IO_TIME only; NULL in DEEPVQE_IO_FREQ.
     *
     * Both analyses are driven with aiaec_analysis_push_frame -- the plain
     * one-frame-per-hop rolling window, NOT aiaec_analysis_push's centered
     * schedule -- see the header's FRAMING note. The structs embed their
     * own transform scratch (struct-owned, never stack). */
    AiaecAnalysis  *ana_mic;
    AiaecAnalysis  *ana_far;
    AiaecSynthesis *synth;
    float          *out_hop;    /* [AIAEC_HOP] the synthesis output. Zero
                                 * from init/reset, then only ever written
                                 * wholesale by the synthesis. */

    /* Per-hop frame staging: one frame, in both I/O modes. */
    float *mic_re;      /* [AIAEC_N_BINS] */
    float *mic_im;
    float *far_re;
    float *far_im;
    float *enh_re;      /* [AIAEC_N_BINS] commit/skip staging */
    float *enh_im;

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
    int rank = 4;
    int c = 0;
    int t = 0;
    int f = 0;

    if (!dims || !dv_valid_depth(delay_depth)) return -1;
    /* One switch is the single source of truth for the boundary: the pool
     * carve, the published element counts and any schema check a tool runs
     * all read it, so a shape can never be right in one and wrong in
     * another. Channel counts mirror DeepVQE_S/model.py's schedule; the
     * frequency ladder is the encoder's stride-2 halving. */
    switch (state_id) {
    case DEEPVQE_STATE_ALIGN_KEY_RING:
        c = DEEPVQE_SIM_CHANNELS;   t = delay_depth; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_ALIGN_VALUE_RING:
        c = DEEPVQE_VALUE_CHANNELS; t = delay_depth; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_ALIGN_SCORE_HISTORY:
        c = DEEPVQE_SIM_CHANNELS;   t = DEEPVQE_SCORE_HISTORY;
        f = delay_depth; break;
    case DEEPVQE_STATE_CCM_UP_HISTORY:
        c = 32; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F1; break;
    case DEEPVQE_STATE_FAR1_HISTORY:
        c = 2;  t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F0; break;
    case DEEPVQE_STATE_FAR2_HISTORY:
        c = 8;  t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F1; break;
    case DEEPVQE_STATE_H_GRU:
        /* The one rank-3 tensor at this boundary: [1,1,GRU_HIDDEN]. */
        rank = 3;
        dims[0] = 1;
        dims[1] = DEEPVQE_GRU_LAYERS;
        dims[2] = DEEPVQE_GRU_HIDDEN;
        dims[3] = 0;
        return rank;
    case DEEPVQE_STATE_MIC1_HISTORY:
        c = 2;  t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F0; break;
    case DEEPVQE_STATE_MIC2_HISTORY:
        c = 16; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F1; break;
    case DEEPVQE_STATE_MIC3_HISTORY:
        c = 64; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_MIC4_HISTORY:
        c = 56; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F3; break;
    case DEEPVQE_STATE_RES2_HISTORY:
        c = 32; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_RES3_HISTORY:
        c = 40; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F3; break;
    case DEEPVQE_STATE_UP1_HISTORY:
        c = 32; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F2; break;
    case DEEPVQE_STATE_UP2_HISTORY:
        c = 40; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F3; break;
    case DEEPVQE_STATE_UP3_HISTORY:
        c = 24; t = DEEPVQE_CONV_HISTORY; f = DEEPVQE_F4; break;
    default:
        return -1;
    }
    dims[0] = 1;
    dims[1] = c;
    dims[2] = t;
    dims[3] = f;
    return rank;
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

/* ---- pool arithmetic ---------------------------------------------------
 * audio_common's saturating helpers (mem_align.h): an overflow anywhere in
 * the walk pins the cursor to SIZE_MAX, which MEM_SIZE_INVALID catches. */

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

/* The build identity a pool is only valid for. io_mode is IN here because
 * the two modes carve different regions at the same total-size class. */
static uint32_t pp_build_hash(int io_mode, int delay_depth) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a(h, (uint32_t)AIAEC_SR);
    h = pp_fnv1a(h, (uint32_t)AIAEC_N_FFT);
    h = pp_fnv1a(h, (uint32_t)DEEPVQE_PREPOST_LAYOUT_VERSION);
    h = pp_fnv1a(h, (uint32_t)DEEPVQE_PREPOST_CARVE_VERSION);
    h = pp_fnv1a(h, (uint32_t)io_mode);
    h = pp_fnv1a(h, (uint32_t)delay_depth);
    return h;
}

/* Walk the layout once. base==NULL sizes it; base!=NULL also carves it.
 * get_mem_size and init MUST walk in this same order -- that is the whole
 * reason it is one function. */
static int pp_layout(DeepVqePrepost *p, unsigned char *base,
                     const DeepVqePrepostConfig *cfg, size_t *total) {
    size_t cursor = 0;
    const size_t bins = (size_t)AIAEC_N_BINS * sizeof(float);
    void *ptr;
    int id;

    if (pp_carve(base, &cursor, sizeof(DeepVqePrepost), &ptr) != 0) return -1;

    /* One buffer per state tensor: the accelerator reads it and writes its
     * next value back to the same address. */
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
        if (pp_carve(base, &cursor, sizeof(AiaecAnalysis), &ptr) != 0) return -1;
        if (base && p) p->ana_mic = (AiaecAnalysis *)ptr;
        if (pp_carve(base, &cursor, sizeof(AiaecAnalysis), &ptr) != 0) return -1;
        if (base && p) p->ana_far = (AiaecAnalysis *)ptr;
        if (pp_carve(base, &cursor, sizeof(AiaecSynthesis), &ptr) != 0)
            return -1;
        if (base && p) p->synth = (AiaecSynthesis *)ptr;
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
    descriptor->conv_history_frames = DEEPVQE_CONV_HISTORY;
    descriptor->score_history_frames = DEEPVQE_SCORE_HISTORY;
    descriptor->gru_layers = DEEPVQE_GRU_LAYERS;
    descriptor->gru_hidden = DEEPVQE_GRU_HIDDEN;
    descriptor->state_tensor_count = DEEPVQE_STATE_COUNT;
    return 0;
}

int deepvqe_prepost_descriptor_validate(
    const DeepVqePrepostDescriptor *descriptor) {
    if (!descriptor) return -1;
    /* Field by field, not memcmp: a descriptor deserialized from ONNX/JSON
     * metadata may carry whatever the writer left in the struct padding, and
     * that must not decide the verdict.
     *
     * delay_depth is checked for RANGE only, deliberately: it is the one
     * genuine deployment parameter here, chosen at export time, so a
     * descriptor is not wrong for carrying a D this build was not compiled
     * with -- the pool is sized from it. Everything else below is a compiled
     * constant, and a graph disagreeing on any of them is a different
     * boundary. */
    if (!dv_valid_depth(descriptor->delay_depth)) return -1;
    if (descriptor->layout_version != DEEPVQE_PREPOST_LAYOUT_VERSION ||
        descriptor->sample_rate != AIAEC_SR ||
        descriptor->fft_size != AIAEC_N_FFT ||
        descriptor->hop_size != AIAEC_HOP ||
        descriptor->spectrum_bins != AIAEC_N_BINS ||
        descriptor->time_order != DEEPVQE_TIME_ORDER ||
        descriptor->freq_taps != DEEPVQE_FREQ_TAPS ||
        descriptor->conv_history_frames != DEEPVQE_CONV_HISTORY ||
        descriptor->score_history_frames != DEEPVQE_SCORE_HISTORY ||
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
        /* Reject-first, same rule as aiaec_analysis_init: a handle of the
         * wrong size would silently break the feature-time contract. */
        if (!cfg->fft || !cfg->window) return -1;
        if (fft_get_n_freqs(cfg->fft) != AIAEC_N_BINS) return -1;
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

    /* Whole control region zeroed first, so a poisoned pool initialises
     * identically to a zeroed one -- and so every state tensor starts at the
     * model's reset value without a second pass. */
    memset(pool, 0, (size_t)req.bytes);
    p = (DeepVqePrepost *)pool;
    if (deepvqe_prepost_descriptor_default(cfg->delay_depth,
                                           &p->descriptor) != 0) {
        return NULL;
    }
    p->io_mode = cfg->io_mode;
    p->fft = (cfg->io_mode == DEEPVQE_IO_TIME) ? cfg->fft : NULL;
    p->window = (cfg->io_mode == DEEPVQE_IO_TIME) ? cfg->window : NULL;

    if (pp_layout(p, (unsigned char *)pool, cfg, &total) != 0) return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    deepvqe_ccm_init(p->ccm);
    if (cfg->io_mode == DEEPVQE_IO_TIME &&
        (aiaec_analysis_init(p->ana_mic, p->fft, p->window) != 0 ||
         aiaec_analysis_init(p->ana_far, p->fft, p->window) != 0 ||
         aiaec_synthesis_init(p->synth, p->fft, p->window) != 0)) {
        return NULL;
    }
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
    p->heap_base = NULL;   /* a pool instance owns nothing to free and
                            * stays usable: repeat calls on it are no-ops */
    if (base) free(base);
}

static void zero_state(DeepVqePrepost *p) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        memset(p->state[id], 0, p->state_elements[id] * sizeof(float));
}

void deepvqe_prepost_reset(DeepVqePrepost *p) {
    if (!p) return;
    zero_state(p);
    memset(p->taps, 0, DEEPVQE_TAPS_ELEMENTS * sizeof(float));
    deepvqe_ccm_init(p->ccm);
    if (p->io_mode == DEEPVQE_IO_TIME) {
        /* Re-init IS the reset for all three framing states (see
         * aiaec_process.h); the handle and window are unchanged. */
        aiaec_analysis_init(p->ana_mic, p->fft, p->window);
        aiaec_analysis_init(p->ana_far, p->fft, p->window);
        aiaec_synthesis_init(p->synth, p->fft, p->window);
        memset(p->out_hop, 0, (size_t)AIAEC_HOP * sizeof(float));
    }
    memset(p->enh_re, 0, (size_t)AIAEC_N_BINS * sizeof(float));
    memset(p->enh_im, 0, (size_t)AIAEC_N_BINS * sizeof(float));
    p->frame_open = 0;
    p->prepared = 0;
    p->written = 0;
}

/* ---- accessors ------------------------------------------------------- */

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

    /* center=False rolling analysis, one frame per hop from the very first
     * hop (see aiaec_process.h). Both streams are transformed from the SAME
     * hop index -- this class applies no internal skew -- so they cannot
     * desynchronise. */
    (void)aiaec_analysis_push_frame(p->ana_mic, mic_hop, p->mic_re, p->mic_im);
    (void)aiaec_analysis_push_frame(p->ana_far, far_hop, p->far_re, p->far_im);

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

/* x * |x|^(e-1) with |x| = sqrt(re^2 + im^2 + 1e-12), the model's
 * compressed_ri_feature (aiaec_common.py), as passes over the planes so the
 * sqrt and the scaling are vector loops; only the powf is per element. The
 * imaginary output plane is the scratch for |x| and then the scale.
 * Products stay unfused (-ffp-contract=off). */
static void compress_planar(const float *re, const float *im, float *out) {
    float *out_re = out;
    float *out_im = out + AIAEC_N_BINS;
    const float exponent = DEEPVQE_COMPRESSION_EXP - 1.0f;
    int k;

    for (k = 0; k < AIAEC_N_BINS; ++k)
        out_im[k] = sqrtf(re[k] * re[k] + im[k] * im[k] + 1e-12f);
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

    /* The graph's fixed front end, run on the host: the power-law
     * compression of both spectra into [1,2,1,BINS] planar tensors. The raw
     * mic spectrum stays in mic_re/mic_im for the CCM ring. */
    compress_planar(p->mic_re, p->mic_im, p->mic_ri);
    compress_planar(p->far_re, p->far_im, p->far_ri);

    /* Arms the transaction and NaN-fills the head output, so a caller that
     * asks twice still gets a clean one rather than a half-written one. The
     * state tensors are NOT filled: each is the accelerator's input as well
     * as its output. */
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

int deepvqe_prepost_outputs_inherit(const DeepVqePrepostOutputs *destination,
                                    const DeepVqePrepostOutputs *runtime) {
    int id;

    if (!destination || !runtime) return -1;
    if (!destination->taps || !runtime->taps) return -1;
    if (runtime->taps_elements != destination->taps_elements) return -1;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        if (!destination->state_out[id] || !runtime->state_out[id]) return -1;
        if (runtime->state_elements[id] != destination->state_elements[id])
            return -1;
    }

    /* Validate every tensor that would be copied before copying any, so a
     * refused inherit leaves the pool exactly as it was. A tensor the runtime
     * wrote at the destination's own address is not copied and not checked
     * here: frame_commit checks it. */
    if (runtime->taps != destination->taps &&
        !all_finite(runtime->taps, runtime->taps_elements))
        return -1;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        if (runtime->state_out[id] != destination->state_out[id] &&
            !all_finite(runtime->state_out[id], runtime->state_elements[id]))
            return -1;
    }

    if (runtime->taps != destination->taps)
        memcpy(destination->taps, runtime->taps,
               destination->taps_elements * sizeof(float));
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        if (runtime->state_out[id] != destination->state_out[id])
            memcpy(destination->state_out[id], runtime->state_out[id],
                   destination->state_elements[id] * sizeof(float));
    }
    return 0;
}

/* Push the finished frame through the synthesis (TIME) or stage it (FREQ)
 * and close it. Cannot fail: the synthesis only ever reports 0 samples (the
 * very first frame, whose block lies inside the trimmed half window) or a
 * full AIAEC_HOP, and never reads `out`. So the only fallible step of a
 * commit is the validation that precedes the CCM ring advance, so a failed
 * commit leaves the ring untouched structurally rather than by argument. */
static void pp_close_frame(DeepVqePrepost *p) {
    p->frame_open = 0;
    p->prepared = 0;
    if (p->io_mode == DEEPVQE_IO_TIME) {
        p->written = aiaec_synthesis_push(p->synth, p->enh_re, p->enh_im,
                                          p->out_hop);
    }
}

int deepvqe_prepost_frame_commit(DeepVqePrepost *p) {
    int ok;
    int id;

    if (!p || !p->prepared) return -1;   /* prepared implies frame_open */

    ok = all_finite(p->taps, DEEPVQE_TAPS_ELEMENTS);
    for (id = 0; ok && id < DEEPVQE_STATE_COUNT; ++id)
        ok = all_finite(p->state[id], p->state_elements[id]);
    if (!ok) {
        /* The accelerator already overwrote the state in place, so there is
         * nothing to roll back to: restart the recurrence from the cold
         * state. The CCM ring has not been pushed, and the frame stays open
         * so the caller can take the fail-closed identity with frame_skip(). */
        zero_state(p);
        p->prepared = 0;
        return -1;
    }

    /* The CCM ring is host state (the exporter pops spec_ring out of the
     * graph): deepvqe_ccm_process pushes THIS frame's raw microphone
     * spectrum and convolves the taps over the resulting (t, t-1, t-2)
     * history. It advances only here, on a frame whose outputs validated. */
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
    /* FAIL CLOSED. DeepVQE-S's stream 0 is the RAW MICROPHONE, so the
     * pass-through identity a post-filter can take would emit the full
     * uncancelled echo -- see deepvqe_prepost.h. Silence instead: bounded,
     * audible as a one-frame notch, and never echo.
     *
     * The CCM ring is not pushed, and the state tensors are left exactly as
     * the accelerator wrote them: a runtime that reports failure did not
     * write, so there is nothing to undo. The armed transaction is simply not
     * committed; the next frame_inputs() re-arms it and re-fills the head
     * output with NaN. */
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
    /* Always a full, defined hop: out_hop is zero from init/reset and the
     * synthesis only ever writes it wholesale, so the one warm-up frame that
     * emits nothing hands over silence, never stale samples. */
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
