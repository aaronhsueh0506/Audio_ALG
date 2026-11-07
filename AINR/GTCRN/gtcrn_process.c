/* GTCRN pre/post-processing (contract: gtcrn_process.h). audio_stft.h is included only to ask
 * audiolib whether an n_fft is valid. Requires -ffp-contract=off. No heap, no globals, no stdio. */

#include "gtcrn_process.h"

#include <string.h>

#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"

#if SK_HAVE_NEON
#include <arm_neon.h>
#endif
#define DF_COMMON_HAVE_NEON SK_HAVE_NEON

#define PP_ALIGN ((size_t)GTCRN_PROCESS_ALIGNMENT)

struct GTCRNProcess {
    GTCRNGrid grid;

    /* Borrowed, never created or destroyed here. */
    const float* erb_fwd;
    const float* erb_inv;

    /* Scratch, [n_bins] each: model_input uses all three, model_output the last two (the mask).
     * Always written before it is read. */
    float* scratch_mag;
    float* scratch_re;
    float* scratch_im;

    /* Registered recurrent state: NULL and 0 when nothing is registered (all or none). */
    float* state[GTCRN_STATE_COUNT];
    size_t state_elements[GTCRN_STATE_COUNT];

    unsigned int state_generation;
};

static int n_fft_accepted(int n_fft)
{
    AcStftConfig sc;
    ac_stft_config_default(&sc, n_fft);
    return ac_stft_get_mem_size(&sc) != 0u;
}

int gtcrn_grid_make(GTCRNGrid* grid, const GTCRNGridSpec* spec)
{
    GTCRNGrid g;
    if (!grid || !spec) return GTCRN_GRID_ERR_NULL;
    if (!n_fft_accepted(spec->n_fft)) return GTCRN_GRID_ERR_N_FFT;
    g.n_fft = spec->n_fft;
    g.hop = spec->n_fft / 2;
    g.n_bins = spec->n_fft / 2 + 1;
    if (spec->erb_low_bins < 0 || spec->erb_low_bins >= g.n_bins)
        return GTCRN_GRID_ERR_ERB_LOW_BINS;
    g.erb_low_bins = spec->erb_low_bins;
    g.erb_high_bins = g.n_bins - spec->erb_low_bins;
    if (spec->erb_high_bands < 1 || spec->erb_high_bands > g.erb_high_bins)
        return GTCRN_GRID_ERR_ERB_HIGH_BANDS;
    g.erb_high_bands = spec->erb_high_bands;
    g.erb_bands = g.erb_low_bins + g.erb_high_bands;
    *grid = g;
    return GTCRN_GRID_OK;
}

int gtcrn_grid_validate(const GTCRNGrid* grid)
{
    GTCRNGridSpec spec;
    GTCRNGrid want;
    int status;
    if (!grid) return GTCRN_GRID_ERR_NULL;
    spec.n_fft = grid->n_fft;
    spec.erb_low_bins = grid->erb_low_bins;
    spec.erb_high_bands = grid->erb_high_bands;
    status = gtcrn_grid_make(&want, &spec);
    if (status != GTCRN_GRID_OK) return status;
    return grid->hop == want.hop && grid->n_bins == want.n_bins &&
                   grid->erb_bands == want.erb_bands &&
                   grid->erb_high_bins == want.erb_high_bins
               ? GTCRN_GRID_OK
               : GTCRN_GRID_ERR_INCONSISTENT;
}

const char* gtcrn_grid_status_text(int status)
{
    switch (status) {
    case GTCRN_GRID_OK:                  return "ok";
    case GTCRN_GRID_ERR_NULL:            return "null argument";
    case GTCRN_GRID_ERR_N_FFT:           return "n_fft refused by audiolib";
    case GTCRN_GRID_ERR_ERB_LOW_BINS:    return "erb_low_bins out of range";
    case GTCRN_GRID_ERR_ERB_HIGH_BANDS:  return "erb_high_bands out of range";
    case GTCRN_GRID_ERR_INCONSISTENT:    return "derived fields do not match the spec";
    default:                             return "unknown status";
    }
}

/* Scalar twin of a NEON kernel; it is the whole kernel where SK_HAVE_NEON is 0. */
#define GTCRN_TWIN static inline __attribute__((unused))

/* Low bins pass through; high bins use the shared scalar kernel (ascending-bin, unfused sums). */
GTCRN_TWIN void erb_band_channel_scalar(const GTCRNGrid* g, const float* full,
                                        const float* erb_fwd, float* banded)
{
    for (int k = 0; k < g->erb_low_bins; ++k) banded[k] = full[k];
    skn_erb_band_accumulate_f32_scalar(full + g->erb_low_bins, erb_fwd, g->erb_high_bins,
                                       g->erb_high_bands, banded + g->erb_low_bins);
}

#if DF_COMMON_HAVE_NEON
/* Quads of bins the mask-expansion kernel keeps in registers. */
#define MASK_BIN_QUADS 8

/* One table pass for the three channels; each channel/band sums in ascending-bin, unfused order. */
static inline void erb_band3_neon(const GTCRNGrid* g, const float* const full[3],
                                  const float* erb_fwd, float* const banded[3])
{
    const float* x3[3];
    float* b3[3];
    for (int c = 0; c < 3; ++c) {
        memcpy(banded[c], full[c], (size_t)g->erb_low_bins * sizeof(float));
        x3[c] = full[c] + g->erb_low_bins;
        b3[c] = banded[c] + g->erb_low_bins;
    }
    skn_erb_band_accumulate_nch_f32(x3, 3, erb_fwd, g->erb_high_bins, g->erb_high_bands, b3);
}
#endif

static inline void erb_band3(const GTCRNGrid* g, const float* const full[3],
                             const float* erb_fwd, float* const banded[3])
{
#if DF_COMMON_HAVE_NEON
    erb_band3_neon(g, full, erb_fwd, banded);
#else
    for (int c = 0; c < 3; ++c) erb_band_channel_scalar(g, full[c], erb_fwd, banded[c]);
#endif
}

void gtcrn_model_input(GTCRNProcess* p, const float* spectrum, float* mag, float* real_part,
                       float* imag_part)
{
    const float* full[3];
    float* banded[3];
    if (!p || !spectrum || !mag || !real_part || !imag_part) return;
    full[0] = p->scratch_mag;
    full[1] = p->scratch_re;
    full[2] = p->scratch_im;
    banded[0] = mag;
    banded[1] = real_part;
    banded[2] = imag_part;
    skn_deinterleave_cf32((const Complex*)(const void*)spectrum, p->scratch_re, p->scratch_im,
                          (size_t)p->grid.n_bins);
    skn_mag_eps_f32(p->scratch_mag, p->scratch_re, p->scratch_im, (size_t)p->grid.n_bins,
                    1e-12f);
    erb_band3(&p->grid, full, p->erb_fwd, banded);
}

/* mask[low + k] = sum over the high bands b ascending of inv[b][k] * gain[b] (band-major table). */
GTCRN_TWIN void mask_expand_scalar(const GTCRNGrid* g, const float* mask_erb,
                                   const float* erb_inv, float* mask_re, float* mask_im)
{
    const int low = g->erb_low_bins;
    for (int k = 0; k < g->erb_high_bins; ++k) {
        mask_re[low + k] = 0.0f;
        mask_im[low + k] = 0.0f;
    }
    for (int b = 0; b < g->erb_high_bands; ++b) {
        float band_re = mask_erb[2 * (size_t)(low + b)];
        float band_im = mask_erb[2 * (size_t)(low + b) + 1];
        const float* row = erb_inv + (size_t)b * (size_t)g->erb_high_bins;
        for (int k = 0; k < g->erb_high_bins; ++k) {
            mask_re[low + k] += row[k] * band_re;
            mask_im[low + k] += row[k] * band_im;
        }
    }
}

#if DF_COMMON_HAVE_NEON
/* The scalar twin's sums with each bin block in registers: MASK_BIN_QUADS quads, then single
 * quads, then single bins. A bin's sum is independent of its neighbours, so tiling changes no
 * value. */
static inline void mask_expand_neon(const GTCRNGrid* g, const float* mask_erb,
                                    const float* erb_inv, float* mask_re, float* mask_im)
{
    const int low = g->erb_low_bins;
    const int hi_bins = g->erb_high_bins;
    const int hi_bands = g->erb_high_bands;
    int k0 = 0;
    for (; k0 + 4 * MASK_BIN_QUADS <= hi_bins; k0 += 4 * MASK_BIN_QUADS) {
        float32x4_t acc_re[MASK_BIN_QUADS], acc_im[MASK_BIN_QUADS];
        for (int q = 0; q < MASK_BIN_QUADS; ++q) {
            acc_re[q] = vdupq_n_f32(0.0f);
            acc_im[q] = vdupq_n_f32(0.0f);
        }
        for (int b = 0; b < hi_bands; ++b) {
            const float* row = erb_inv + (size_t)b * (size_t)hi_bins + (size_t)k0;
            const float32x4_t gr = vdupq_n_f32(mask_erb[2 * (size_t)(low + b)]);
            const float32x4_t gi = vdupq_n_f32(mask_erb[2 * (size_t)(low + b) + 1]);
            for (int q = 0; q < MASK_BIN_QUADS; ++q) {
                const float32x4_t w = vld1q_f32(row + 4 * q);
                acc_re[q] = vaddq_f32(acc_re[q], vmulq_f32(w, gr));
                acc_im[q] = vaddq_f32(acc_im[q], vmulq_f32(w, gi));
            }
        }
        for (int q = 0; q < MASK_BIN_QUADS; ++q) {
            vst1q_f32(mask_re + low + k0 + 4 * q, acc_re[q]);
            vst1q_f32(mask_im + low + k0 + 4 * q, acc_im[q]);
        }
    }
    for (; k0 + 4 <= hi_bins; k0 += 4) {
        float32x4_t acc_re = vdupq_n_f32(0.0f);
        float32x4_t acc_im = vdupq_n_f32(0.0f);
        for (int b = 0; b < hi_bands; ++b) {
            const float32x4_t w =
                vld1q_f32(erb_inv + (size_t)b * (size_t)hi_bins + (size_t)k0);
            acc_re = vaddq_f32(acc_re, vmulq_f32(w, vdupq_n_f32(mask_erb[2 * (size_t)(low + b)])));
            acc_im = vaddq_f32(
                acc_im, vmulq_f32(w, vdupq_n_f32(mask_erb[2 * (size_t)(low + b) + 1])));
        }
        vst1q_f32(mask_re + low + k0, acc_re);
        vst1q_f32(mask_im + low + k0, acc_im);
    }
    for (; k0 < hi_bins; ++k0) {
        float sum_re = 0.0f, sum_im = 0.0f;
        for (int b = 0; b < hi_bands; ++b) {
            const float w = erb_inv[(size_t)b * (size_t)hi_bins + (size_t)k0];
            sum_re += w * mask_erb[2 * (size_t)(low + b)];
            sum_im += w * mask_erb[2 * (size_t)(low + b) + 1];
        }
        mask_re[low + k0] = sum_re;
        mask_im[low + k0] = sum_im;
    }
}
#endif

static inline void mask_expand(const GTCRNGrid* g, const float* mask_erb, const float* erb_inv,
                               float* mask_re, float* mask_im)
{
#if DF_COMMON_HAVE_NEON
    mask_expand_neon(g, mask_erb, erb_inv, mask_re, mask_im);
#else
    mask_expand_scalar(g, mask_erb, erb_inv, mask_re, mask_im);
#endif
}

void gtcrn_model_output(GTCRNProcess* p, const float* mask_erb, const float* spectrum,
                        float* enhanced)
{
    float* mask_re;
    float* mask_im;
    if (!p || !mask_erb || !spectrum || !enhanced) return;
    mask_re = p->scratch_re;
    mask_im = p->scratch_im;
    for (int k = 0; k < p->grid.erb_low_bins; ++k) {
        mask_re[k] = mask_erb[2 * (size_t)k];
        mask_im[k] = mask_erb[2 * (size_t)k + 1];
    }
    mask_expand(&p->grid, mask_erb, p->erb_inv, mask_re, mask_im);
    /* Complex ratio mask (model.py Mask). */
    for (int k = 0; k < p->grid.n_bins; ++k) {
        float sr = spectrum[2 * (size_t)k], si = spectrum[2 * (size_t)k + 1];
        float mr = mask_re[k], mi = mask_im[k];
        enhanced[2 * (size_t)k] = sr * mr - si * mi;
        enhanced[2 * (size_t)k + 1] = si * mr + sr * mi;
    }
}

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(GTCRN_PROCESS_ALIGNMENT == 16u,
               "ck_align16_size is the shared 16-byte carve alignment");
#endif
#endif

/* Reserve `count` elements of `elem` bytes at *cursor, aligned; base==NULL sizes only. The audiolib
 * helpers saturate: overflow pins the cursor to SIZE_MAX, which MEM_SIZE_INVALID catches. */
static int pp_carve(unsigned char* base, size_t* cursor, size_t count, size_t elem, void** out)
{
    size_t aligned = ck_align16_size(*cursor);
    size_t next = ck_add_size(aligned, ck_mul_size(count, elem));
    if (MEM_SIZE_INVALID(next)) return -1;
    if (base && out) *out = base + aligned;
    *cursor = next;
    return 0;
}

/* One walk for sizing (base==NULL) and carving (p receives the scratch pointers), so get_mem_size
 * and init agree on the order. */
static int pp_layout(GTCRNProcess* p, unsigned char* base, const GTCRNGrid* g, size_t* total)
{
    const size_t n_bins = (size_t)g->n_bins;
    size_t cursor = 0;
    void* ptr = NULL;
    if (pp_carve(base, &cursor, 1u, sizeof(GTCRNProcess), &ptr) != 0) return -1;
    if (pp_carve(base, &cursor, n_bins, sizeof(float), &ptr) != 0) return -1;
    if (base && p) p->scratch_mag = (float*)ptr;
    if (pp_carve(base, &cursor, n_bins, sizeof(float), &ptr) != 0) return -1;
    if (base && p) p->scratch_re = (float*)ptr;
    if (pp_carve(base, &cursor, n_bins, sizeof(float), &ptr) != 0) return -1;
    if (base && p) p->scratch_im = (float*)ptr;
    *total = ck_align16_size(cursor);
    return MEM_SIZE_INVALID(*total) ? -1 : 0;
}

static uint32_t pp_fnv1a_u32(uint32_t h, uint32_t v)
{
    for (int i = 0; i < 4; ++i) {
        h ^= (v >> (i * 8)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* Build identity a pool is valid for: the whole grid, the layout and carve versions. The state
 * registration is not in it: the buffers are borrowed and change no pool byte. */
static uint32_t pp_build_hash(const GTCRNGrid* g)
{
    uint32_t h = 2166136261u;
    h = pp_fnv1a_u32(h, (uint32_t)g->n_fft);
    h = pp_fnv1a_u32(h, (uint32_t)g->hop);
    h = pp_fnv1a_u32(h, (uint32_t)g->n_bins);
    h = pp_fnv1a_u32(h, (uint32_t)g->erb_low_bins);
    h = pp_fnv1a_u32(h, (uint32_t)g->erb_high_bands);
    h = pp_fnv1a_u32(h, (uint32_t)g->erb_bands);
    h = pp_fnv1a_u32(h, (uint32_t)g->erb_high_bins);
    h = pp_fnv1a_u32(h, (uint32_t)GTCRN_MODEL_LAYOUT_VERSION);
    h = pp_fnv1a_u32(h, (uint32_t)GTCRN_PROCESS_CARVE_VERSION);
    return ac_memreq_abi_hash(h);
}

int gtcrn_process_config_defaults(GTCRNProcessConfig* cfg, const GTCRNGrid* grid)
{
    if (!cfg) return -1;
    if (gtcrn_grid_validate(grid) != GTCRN_GRID_OK) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->grid = *grid;
    return 0;
}

/* Eight pairs, each a pointer with a nonzero count whose byte size fits, or none at all. */
static int pp_registration_ok(const GTCRNProcessConfig* cfg)
{
    int any = 0;
    int valid = 0;
    for (int i = 0; i < GTCRN_STATE_COUNT; ++i) {
        any |= cfg->state[i] != NULL || cfg->state_elements[i] != 0u;
        valid += cfg->state[i] != NULL && cfg->state_elements[i] != 0u &&
                 cfg->state_elements[i] <= SIZE_MAX / sizeof(float);
    }
    return !any || valid == GTCRN_STATE_COUNT;
}

static int pp_check_config(const GTCRNProcessConfig* cfg)
{
    if (!cfg) return -1;
    if (gtcrn_grid_validate(&cfg->grid) != GTCRN_GRID_OK) return -1;
    if (!cfg->erb_fwd || !cfg->erb_inv) return -1;
    return pp_registration_ok(cfg) ? 0 : -1;
}

int gtcrn_process_get_mem_size(const GTCRNProcessConfig* cfg, GTCRNProcessMemReq* req)
{
    size_t total = 0;
    if (!req || pp_check_config(cfg) != 0) return -1;
    if (pp_layout(NULL, NULL, &cfg->grid, &total) != 0) return -1;
    memset(req, 0, sizeof(*req));
    req->descriptor_version = GTCRN_PROCESS_DESCRIPTOR_VERSION;
    req->layout_version = (uint32_t)GTCRN_MODEL_LAYOUT_VERSION;
    req->build_flags_hash = pp_build_hash(&cfg->grid);
    req->alignment = (uint32_t)PP_ALIGN;
    req->bytes = (uint64_t)total;
    return 0;
}

static void pp_zero_registered(GTCRNProcess* p)
{
    for (int i = 0; i < GTCRN_STATE_COUNT; ++i)
        if (p->state[i]) memset(p->state[i], 0, p->state_elements[i] * sizeof(float));
}

GTCRNProcess* gtcrn_process_init_ex(void* pool, size_t bytes, const GTCRNProcessConfig* cfg,
                                    const GTCRNProcessMemReq* expected)
{
    GTCRNProcessMemReq req;
    GTCRNProcess* p;
    size_t total = 0;

    if (!pool || gtcrn_process_get_mem_size(cfg, &req) != 0) return NULL;
    if (bytes < (size_t)req.bytes) return NULL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return NULL;
    if (expected) {
        if (expected->descriptor_version != req.descriptor_version ||
            expected->layout_version != req.layout_version ||
            expected->reserved0 != req.reserved0 ||
            expected->build_flags_hash != req.build_flags_hash ||
            expected->alignment != req.alignment || expected->reserved1 != req.reserved1 ||
            expected->bytes != req.bytes) {
            return NULL;
        }
    }

    memset(pool, 0, (size_t)req.bytes);
    p = (GTCRNProcess*)pool;
    p->grid = cfg->grid;
    p->erb_fwd = cfg->erb_fwd;
    p->erb_inv = cfg->erb_inv;
    if (pp_layout(p, (unsigned char*)pool, &cfg->grid, &total) != 0) return NULL;
    if (total != (size_t)req.bytes) return NULL;   /* sizing/carve agreement */

    for (int i = 0; i < GTCRN_STATE_COUNT; ++i) {
        p->state[i] = cfg->state[i];
        p->state_elements[i] = cfg->state_elements[i];
    }
    pp_zero_registered(p);
    p->state_generation = 1u;
    return p;
}

GTCRNProcess* gtcrn_process_init(void* pool, size_t bytes, const GTCRNProcessConfig* cfg)
{
    return gtcrn_process_init_ex(pool, bytes, cfg, NULL);
}

void gtcrn_process_reset(GTCRNProcess* p)
{
    if (p == NULL) return;
    pp_zero_registered(p);
    ++p->state_generation;
}

unsigned int gtcrn_process_state_generation(const GTCRNProcess* p)
{
    return p == NULL ? 0u : p->state_generation;
}

const GTCRNGrid* gtcrn_process_grid(const GTCRNProcess* p)
{
    return p == NULL ? NULL : &p->grid;
}

int gtcrn_process_set_erb_matrices(GTCRNProcess* p, const float* erb_fwd, const float* erb_inv)
{
    if (!p || !erb_fwd || !erb_inv) return -1;
    p->erb_fwd = erb_fwd;
    p->erb_inv = erb_inv;
    return 0;
}

int gtcrn_process_state_validate(GTCRNProcess* p)
{
    if (p == NULL) return -1;
    if (p->state[0] == NULL) return 0;   /* nothing registered: nothing to scan */
    for (int i = 0; i < GTCRN_STATE_COUNT; ++i) {
        if (!skn_all_finite_f32(p->state[i], p->state_elements[i])) {
            /* Zero is the only state a stream can restart from after an in-place overwrite. */
            pp_zero_registered(p);
            ++p->state_generation;
            return -1;
        }
    }
    return 0;
}

int gtcrn_process_mask_validate(GTCRNProcess* p, const float* mask_erb)
{
    if (p == NULL || mask_erb == NULL) return -1;
    if (skn_all_finite_f32(mask_erb, 2u * (size_t)p->grid.erb_bands)) return 0;
    pp_zero_registered(p);
    ++p->state_generation;
    return -1;
}

const char* gtcrn_simd_backend(void)
{
    return SK_NEON_BACKEND_NAME;
}
