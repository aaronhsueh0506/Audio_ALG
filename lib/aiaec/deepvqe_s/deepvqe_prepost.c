/* DeepVQE-S pre/post class (contract: deepvqe_prepost.h): lifecycle, pool carve, accelerator
 * views and the per-frame state machine over the CCM kernel of deepvqe_process.c.
 * Requires -ffp-contract=off. No heap in _init, no globals, no stdio. */

#include "deepvqe_prepost.h"

#include "audio_stft.h"            /* ac_stft_get_mem_size(): the n_fft rule is audiolib's */
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PP_ALIGN ((size_t)DEEPVQE_PREPOST_ALIGNMENT)

/* Graph tensor names in DeepVqeStateId order (the exporter's input_names[2:]). */
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

    /* Registered state (borrowed; all NULL/0 when none): one buffer per tensor, input and _out. */
    float *state[DEEPVQE_STATE_COUNT];
    size_t state_elements[DEEPVQE_STATE_COUNT];

    float *taps;        /* [n_bins * TIME_ORDER * FREQ_TAPS * 2]            */
    float *mic_ri;      /* [2 * n_bins] real plane, imaginary plane         */
    float *far_ri;      /* [2 * n_bins] real plane, imaginary plane         */

    /* Raw-microphone spectrum ring the CCM taps convolve over (arrays carved from the pool). */
    DeepVqeCcmState ccm;

    float *mic_re;      /* [frames_per_call * n_bins], oldest pending frame first */
    float *mic_im;
    float *far_re;
    float *far_im;
    float *enh_re;      /* [frames_per_call * n_bins] finish/bypass staging, oldest first */
    float *enh_im;

    size_t taps_elements;
    int n_bins;
    int hop;
    int frames_per_call;

    int pending;         /* frames accumulated into the open batch via pre_process_freq */
    int deliver_slot;    /* next frame post_process_freq drains from enh_re/enh_im */

    uint32_t state_generation;  /* advances whenever the state is zeroed */

    int frame_open;     /* a batch is armed and awaiting finish/bypass */
    int prepared;       /* frame_open() armed the accelerator transaction */

    void *heap_base;    /* _create's allocation; NULL for an _init instance */
};

static int dv_valid_depth(int delay_depth) {
    return delay_depth >= DEEPVQE_PREPOST_MIN_D &&
           delay_depth <= DEEPVQE_PREPOST_MAX_D;
}

static int dv_valid_frames_per_call(int frames_per_call) {
    return frames_per_call >= 1 &&
           frames_per_call <= DEEPVQE_MODEL_MAX_FRAMES_PER_CALL;
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

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(DEEPVQE_PREPOST_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* Reserve `bytes` at *cursor, aligned (base == NULL: size only). Saturating arithmetic: overflow
 * pins the cursor to SIZE_MAX, which MEM_SIZE_INVALID catches. */
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

/* Identity a pool is valid for: grid, layout and carve versions, D. The state registration is
 * not in it: the buffers are borrowed and change no pool byte. */
static uint32_t pp_build_hash(const DeepVqeGrid *grid) {
    uint32_t h = 2166136261u;
    h = pp_fnv1a(h, (uint32_t)grid->sample_rate);
    h = pp_fnv1a(h, (uint32_t)grid->n_fft);
    h = pp_fnv1a(h, (uint32_t)DEEPVQE_PREPOST_LAYOUT_VERSION);
    h = pp_fnv1a(h, (uint32_t)DEEPVQE_PREPOST_CARVE_VERSION);
    h = pp_fnv1a(h, (uint32_t)grid->delay_depth);
    h = pp_fnv1a(h, (uint32_t)grid->frames_per_call);
    return ac_memreq_abi_hash(h);
}

/* One walk for sizing (base == NULL) and carving, so get_mem_size and init agree. */
static int pp_layout(DeepVqePrepost *p, unsigned char *base, const DeepVqeGrid *grid,
                     size_t *total) {
    size_t cursor = 0;
    const size_t bins = (size_t)grid->n_bins * sizeof(float);
    const size_t n_bins = (size_t)grid->frames_per_call * bins;   /* N frames, oldest first */
    void *ptr;

    if (pp_carve(base, &cursor, sizeof(DeepVqePrepost), &ptr) != 0) return -1;

    if (pp_carve(base, &cursor, deepvqe_prepost_taps_elements(grid) * sizeof(float),
                 &ptr) != 0) return -1;
    if (base && p) p->taps = (float *)ptr;
    if (pp_carve(base, &cursor, 2u * n_bins, &ptr) != 0) return -1;
    if (base && p) p->mic_ri = (float *)ptr;
    if (pp_carve(base, &cursor, 2u * n_bins, &ptr) != 0) return -1;
    if (base && p) p->far_ri = (float *)ptr;

    if (pp_carve(base, &cursor, (size_t)DEEPVQE_TIME_ORDER * bins, &ptr) != 0) return -1;
    if (base && p) p->ccm.spectrum_re = (float *)ptr;
    if (pp_carve(base, &cursor, (size_t)DEEPVQE_TIME_ORDER * bins, &ptr) != 0) return -1;
    if (base && p) p->ccm.spectrum_im = (float *)ptr;

    if (pp_carve(base, &cursor, n_bins, &ptr) != 0) return -1;
    if (base && p) p->mic_re = (float *)ptr;
    if (pp_carve(base, &cursor, n_bins, &ptr) != 0) return -1;
    if (base && p) p->mic_im = (float *)ptr;
    if (pp_carve(base, &cursor, n_bins, &ptr) != 0) return -1;
    if (base && p) p->far_re = (float *)ptr;
    if (pp_carve(base, &cursor, n_bins, &ptr) != 0) return -1;
    if (base && p) p->far_im = (float *)ptr;
    if (pp_carve(base, &cursor, n_bins, &ptr) != 0) return -1;
    if (base && p) p->enh_re = (float *)ptr;
    if (pp_carve(base, &cursor, n_bins, &ptr) != 0) return -1;
    if (base && p) p->enh_im = (float *)ptr;

    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

int deepvqe_grid_make(DeepVqeGrid *grid, const DeepVqeGridSpec *spec) {
    AcStftConfig framing;
    if (!grid || !spec) return DEEPVQE_GRID_ERR_NULL;
    if (spec->sample_rate <= 0) return DEEPVQE_GRID_ERR_SAMPLE_RATE;
    ac_stft_config_default(&framing, spec->n_fft);
    if (ac_stft_get_mem_size(&framing) == 0u) return DEEPVQE_GRID_ERR_N_FFT;
    if (!dv_valid_depth(spec->delay_depth)) return DEEPVQE_GRID_ERR_DELAY_DEPTH;
    if (!dv_valid_frames_per_call(spec->frames_per_call))
        return DEEPVQE_GRID_ERR_FRAMES_PER_CALL;
    grid->sample_rate = spec->sample_rate;
    grid->n_fft = spec->n_fft;
    grid->hop = spec->n_fft / 2;
    grid->n_bins = spec->n_fft / 2 + 1;
    grid->delay_depth = spec->delay_depth;
    grid->frames_per_call = spec->frames_per_call;
    return DEEPVQE_GRID_OK;
}

int deepvqe_grid_validate(const DeepVqeGrid *grid) {
    DeepVqeGridSpec spec;
    DeepVqeGrid want;
    int status;
    if (!grid) return DEEPVQE_GRID_ERR_NULL;
    spec.sample_rate = grid->sample_rate;
    spec.n_fft = grid->n_fft;
    spec.delay_depth = grid->delay_depth;
    spec.frames_per_call = grid->frames_per_call;
    status = deepvqe_grid_make(&want, &spec);
    if (status != DEEPVQE_GRID_OK) return status;
    return (grid->hop == want.hop && grid->n_bins == want.n_bins)
               ? DEEPVQE_GRID_OK : DEEPVQE_GRID_ERR_INCONSISTENT;
}

int deepvqe_prepost_config_defaults(DeepVqePrepostConfig *cfg, const DeepVqeGrid *grid) {
    if (!cfg || deepvqe_grid_validate(grid) != DEEPVQE_GRID_OK) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->grid = *grid;
    return 0;
}

int deepvqe_prepost_descriptor_default(const DeepVqeGrid *grid,
                                       DeepVqePrepostDescriptor *descriptor) {
    if (!descriptor || deepvqe_grid_validate(grid) != DEEPVQE_GRID_OK) return -1;
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->layout_version = DEEPVQE_PREPOST_LAYOUT_VERSION;
    descriptor->delay_depth = grid->delay_depth;
    descriptor->sample_rate = grid->sample_rate;
    descriptor->fft_size = grid->n_fft;
    descriptor->hop_size = grid->hop;
    descriptor->spectrum_bins = grid->n_bins;
    descriptor->time_order = DEEPVQE_TIME_ORDER;
    descriptor->freq_taps = DEEPVQE_FREQ_TAPS;
    descriptor->state_tensor_count = DEEPVQE_STATE_COUNT;
    return 0;
}

int deepvqe_prepost_descriptor_validate(
    const DeepVqeGrid *grid, const DeepVqePrepostDescriptor *descriptor) {
    if (!descriptor || deepvqe_grid_validate(grid) != DEEPVQE_GRID_OK) return -1;
    /* Field by field, not memcmp: padding must not decide. delay_depth is only range-checked. */
    if (!dv_valid_depth(descriptor->delay_depth)) return -1;
    if (descriptor->layout_version != DEEPVQE_PREPOST_LAYOUT_VERSION ||
        descriptor->sample_rate != grid->sample_rate ||
        descriptor->fft_size != grid->n_fft ||
        descriptor->hop_size != grid->hop ||
        descriptor->spectrum_bins != grid->n_bins ||
        descriptor->time_order != DEEPVQE_TIME_ORDER ||
        descriptor->freq_taps != DEEPVQE_FREQ_TAPS ||
        descriptor->state_tensor_count != DEEPVQE_STATE_COUNT) {
        return -1;
    }
    return 0;
}

/* All or nothing (deepvqe_prepost.h); a count too large to express in bytes is refused. */
static int pp_registration_ok(const DeepVqePrepostConfig *cfg) {
    int registered = 0;
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        const int has_pointer = cfg->state[id] != NULL;
        const int has_count = cfg->state_elements[id] != 0u;
        if (has_pointer != has_count) return 0;
        if (MEM_SIZE_INVALID(ck_mul_size(cfg->state_elements[id], sizeof(float))))
            return 0;
        registered += has_pointer;
    }
    return registered == 0 || registered == DEEPVQE_STATE_COUNT;
}

static int pp_check_config(const DeepVqePrepostConfig *cfg) {
    if (!cfg) return -1;
    if (deepvqe_grid_validate(&cfg->grid) != DEEPVQE_GRID_OK) return -1;
    if (!pp_registration_ok(cfg)) return -1;
    return 0;
}

int deepvqe_prepost_get_mem_size(const DeepVqePrepostConfig *cfg,
                                 DeepVqePrepostMemReq *req) {
    size_t total = 0;

    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_layout(NULL, NULL, &cfg->grid, &total) != 0) return -1;

    memset(req, 0, sizeof(*req));
    req->descriptor_version = DEEPVQE_PREPOST_DESCRIPTOR_VERSION;
    req->layout_version = DEEPVQE_PREPOST_LAYOUT_VERSION;
    req->build_flags_hash = pp_build_hash(&cfg->grid);
    req->alignment = (uint32_t)PP_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

/* Unregistered (NULL) entries are skipped. */
static void zero_registered(DeepVqePrepost *p) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        if (p->state[id])
            memset(p->state[id], 0, p->state_elements[id] * sizeof(float));
}

DeepVqePrepost *deepvqe_prepost_init_ex(void *pool, size_t bytes,
                                        const DeepVqePrepostConfig *cfg,
                                        const DeepVqePrepostMemReq *expected) {
    DeepVqePrepostMemReq req;
    DeepVqePrepost *p;
    size_t total = 0;
    int id;

    if (!pool || deepvqe_prepost_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        /* Stale-pool gate: a pool sized by another build or D is refused, never reinterpreted. */
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }

    /* Zero the control region first so a poisoned pool initialises like a zeroed one. */
    memset(pool, 0, (size_t)req.bytes);
    p = (DeepVqePrepost *)pool;
    if (deepvqe_prepost_descriptor_default(&cfg->grid, &p->descriptor) != 0) {
        return NULL;
    }

    if (pp_layout(p, (unsigned char *)pool, &cfg->grid, &total) != 0) return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    p->n_bins = cfg->grid.n_bins;
    p->hop = cfg->grid.hop;
    p->frames_per_call = cfg->grid.frames_per_call;
    p->taps_elements = deepvqe_prepost_taps_elements(&cfg->grid);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        p->state[id] = cfg->state[id];
        p->state_elements[id] = cfg->state_elements[id];
    }
    if (deepvqe_ccm_init(&p->ccm, p->n_bins, p->ccm.spectrum_re, p->ccm.spectrum_im) != 0)
        return NULL;
    zero_registered(p);
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
    p->heap_base = NULL;   /* a pool instance has nothing to free; repeat calls are no-ops */
    if (base) free(base);
}

static void zero_state(DeepVqePrepost *p) {
    zero_registered(p);
    ++p->state_generation;
}

void deepvqe_prepost_reset(DeepVqePrepost *p) {
    size_t n_bins;
    if (!p) return;
    n_bins = (size_t)p->frames_per_call * (size_t)p->n_bins * sizeof(float);
    zero_state(p);
    memset(p->taps, 0, p->taps_elements * sizeof(float));
    deepvqe_ccm_reset(&p->ccm);
    memset(p->mic_re, 0, n_bins);
    memset(p->mic_im, 0, n_bins);
    memset(p->far_re, 0, n_bins);
    memset(p->far_im, 0, n_bins);
    memset(p->enh_re, 0, n_bins);
    memset(p->enh_im, 0, n_bins);
    p->frame_open = 0;
    p->prepared = 0;
    p->pending = 0;
    p->deliver_slot = 0;
}

uint32_t deepvqe_prepost_state_generation(const DeepVqePrepost *p) {
    return p ? p->state_generation : 0u;
}

int deepvqe_prepost_hop_size(const DeepVqePrepost *p) {
    return p ? p->hop : -1;
}

int deepvqe_prepost_num_bins(const DeepVqePrepost *p) {
    return p ? p->n_bins : -1;
}

const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(
    const DeepVqePrepost *p) {
    return p ? &p->descriptor : NULL;
}

static void fill_nan(float *values, size_t elements) {
    skn_fill_f32(values, elements, NAN);
}

static int all_finite(const float *values, size_t elements) {
    return skn_all_finite_f32(values, elements);
}

int deepvqe_prepost_pre_process_freq(DeepVqePrepost *p,
                                     const float *mic_re, const float *mic_im,
                                     const float *far_re, const float *far_im) {
    size_t bins;
    size_t at;

    if (!p || !mic_re || !mic_im || !far_re || !far_im) return -1;
    if (p->frame_open) return -1;   /* previous batch neither finished nor bypassed */

    bins = (size_t)p->n_bins * sizeof(float);
    at = (size_t)p->pending * (size_t)p->n_bins;
    memcpy(p->mic_re + at, mic_re, bins);
    memcpy(p->mic_im + at, mic_im, bins);
    memcpy(p->far_re + at, far_re, bins);
    memcpy(p->far_im + at, far_im, bins);

    p->pending += 1;
    if (p->pending < p->frames_per_call) return 0;   /* batch still gathering */
    p->pending = 0;
    p->frame_open = 1;
    return 1;
}

/* x * |x|^(e-1), |x| = sqrt(re^2 + im^2 + 1e-12): vector sqrt, powf per element, vector scale;
 * out_im is scratch for |x| and then the scale, same as out_re == out, out_im == out + n_bins
 * would be for one frame -- the caller places each frame's real/imaginary destinations itself
 * (planar packing: all N real planes, then all N imaginary planes), so the two need not be
 * contiguous for N > 1. Unfused products (-ffp-contract=off). Python reference:
 * aiaec_common.compressed_ri_feature. */
static void compress_planar(const float *re, const float *im,
                            float *out_re, float *out_im, int n_bins) {
    const float exponent = DEEPVQE_COMPRESSION_EXP - 1.0f;
    int k;

    skn_mag_eps_f32(out_im, re, im, (size_t)n_bins, 1e-12f);
    for (k = 0; k < n_bins; ++k)
        out_im[k] = powf(out_im[k], exponent);
    skn_mul_f32(out_re, re, out_im, (size_t)n_bins);
    skn_mul_f32(out_im, im, out_im, (size_t)n_bins);
}

int deepvqe_prepost_frame_open(DeepVqePrepost *p,
                                 DeepVqePrepostInputs *inputs,
                                 DeepVqePrepostOutputs *outputs) {
    int id, i;
    int n_bins;
    int n;

    if (!p || !inputs || !outputs || !p->frame_open) return -1;
    n_bins = p->n_bins;
    n = p->frames_per_call;

    /* The raw mic spectrum stays in mic_re/mic_im for the CCM ring. Planar packing: mic_ri's
     * first n*n_bins floats are the real plane (one frame's n_bins at a time, oldest first),
     * the next n*n_bins are the imaginary plane -- same layout the exporter's [1,2,N,BINS]
     * mic/far inputs trace to. */
    for (i = 0; i < n; ++i) {
        const size_t at = (size_t)i * (size_t)n_bins;
        compress_planar(p->mic_re + at, p->mic_im + at,
                        p->mic_ri + at, p->mic_ri + (size_t)n * (size_t)n_bins + at, n_bins);
        compress_planar(p->far_re + at, p->far_im + at,
                        p->far_ri + at, p->far_ri + (size_t)n * (size_t)n_bins + at, n_bins);
    }

    /* NaN-fill the head only: the state tensors are the accelerator's input too. */
    fill_nan(p->taps, p->taps_elements);

    memset(inputs, 0, sizeof(*inputs));
    memset(outputs, 0, sizeof(*outputs));
    inputs->mic = p->mic_ri;
    inputs->far = p->far_ri;
    inputs->spectrum_ri_elements = 2u * (size_t)n * (size_t)n_bins;
    outputs->taps = p->taps;
    outputs->taps_elements = p->taps_elements;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        inputs->state[id] = p->state[id];
        inputs->state_elements[id] = p->state_elements[id];
        outputs->state_out[id] = p->state[id];
        outputs->state_elements[id] = p->state_elements[id];
    }
    p->prepared = 1;
    return 0;
}

int deepvqe_prepost_frame_finish(DeepVqePrepost *p) {
    int ok;
    int id, i;
    int n_bins;
    size_t taps_per_frame;

    if (!p || !p->prepared) return -1;   /* prepared implies frame_open */

    /* Whole state buffers, older frames included: the accelerator overwrote them in place, so a
     * non-finite value anywhere would reach a later frame. Taps cover all N frames of the batch. */
    ok = all_finite(p->taps, p->taps_elements);
    for (id = 0; ok && id < DEEPVQE_STATE_COUNT; ++id)
        ok = p->state[id] == NULL ||
             all_finite(p->state[id], p->state_elements[id]);
    if (!ok) {
        /* Restart cold; the CCM ring is not pushed and the batch stays open for frame_bypass(). */
        zero_state(p);
        p->prepared = 0;
        return -1;
    }

    /* Pushes each frame's raw mic spectrum in order and convolves its taps over (t, t-1, t-2);
     * the ring advances once per frame, oldest batch frame first, only on a validated batch. */
    n_bins = p->n_bins;
    taps_per_frame = p->taps_elements / (size_t)p->frames_per_call;
    for (i = 0; i < p->frames_per_call; ++i) {
        const size_t at = (size_t)i * (size_t)n_bins;
        deepvqe_ccm_process(&p->ccm, p->mic_re + at, p->mic_im + at,
                            p->taps + (size_t)i * taps_per_frame,
                            p->enh_re + at, p->enh_im + at);
    }
    p->deliver_slot = 0;
    p->frame_open = 0;
    p->prepared = 0;
    return 0;
}

int deepvqe_prepost_frame_bypass(DeepVqePrepost *p) {
    size_t bins;
    if (!p || !p->frame_open) return -1;
    bins = (size_t)p->frames_per_call * (size_t)p->n_bins * sizeof(float);
    /* Fail closed: pass-through of the raw microphone would emit the uncancelled echo; emit
     * silence for every frame of the batch and leave the CCM ring and state as they are. */
    memset(p->enh_re, 0, bins);
    memset(p->enh_im, 0, bins);
    p->deliver_slot = 0;
    p->frame_open = 0;
    p->prepared = 0;
    return 0;
}

int deepvqe_prepost_post_process_freq(DeepVqePrepost *p, float *re, float *im) {
    size_t bins;
    size_t at;
    if (!p || !re || !im) return -1;
    if (p->frame_open) return -1;
    bins = (size_t)p->n_bins * sizeof(float);
    at = (size_t)p->deliver_slot * (size_t)p->n_bins;
    memcpy(re, p->enh_re + at, bins);
    memcpy(im, p->enh_im + at, bins);
    p->deliver_slot = (p->deliver_slot + 1) % p->frames_per_call;
    return 0;
}
