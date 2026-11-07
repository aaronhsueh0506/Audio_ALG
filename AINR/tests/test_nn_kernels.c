/* Bit-exactness of the model-local NEON kernels against their scalar twins
 * and against a plain statement of the contract, on random values and on
 * every special value (NaN, signed zero, infinity, subnormal, extremes). The
 * kernels are static, so this file includes their translation units. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility/simd_test_mode.h"
#include "DeepFilterNet2/dfn2_process.c"
#include "GTCRN/gtcrn_process.c"
/* The kernels take their extents as arguments or from a grid; the grid macros only
 * size this program's buffers (DFN2_N_BINS, DFN2_N_ERB: dfn2_test_grid.h; GTCRN:
 * gtcrn_test_grid.h). */
#include "dfn2_test_grid.h"
#include "gtcrn_test_grid.h"

#define G_BINS   GTCRN_TEST_N_BINS
#define G_BANDS  GTCRN_TEST_ERB_BANDS
#define G_KEPT   GTCRN_TEST_ERB_LOW_BINS
#define G_HIGH_BANDS GTCRN_TEST_ERB_HIGH_BANDS
#define G_HIGH_BINS  GTCRN_TEST_ERB_HIGH_BINS

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
        return 0;                                                           \
    }                                                                       \
} while (0)

static uint32_t rng_state = 0x1234abcdu;
static uint32_t rng(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static float uniform(void) { return (float)(rng() & 0xffff) / 32768.0f - 1.0f; }

static float special(void)
{
    static const uint32_t bits[] = {
        0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x00000001u,
        0x80000001u, 0x007fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu,
        0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u,
        0x7fc12345u, 0x33800000u, 0x7e800000u };
    uint32_t b = bits[rng() % (sizeof(bits) / sizeof(bits[0]))];
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* Fill with random values; `rate` of 1/n replaces an element by a special
 * value, 0 means none. */
static void fill(float* x, size_t n, float scale, unsigned rate)
{
    for (size_t i = 0; i < n; ++i)
        x[i] = (rate != 0 && rng() % rate == 0) ? special() : uniform() * scale;
}

/* Equal bit patterns, except that two NaNs match whatever their payloads:
 * which payload an operation with two NaN operands returns is the compiler's
 * operand order, not part of the contract. */
static int same_bits(const float* a, const float* b, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (isnan(a[i]) && isnan(b[i])) continue;
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) return 0;
    }
    return 1;
}

/* The contract, stated plainly: per band, the products of the bins in
 * ascending order added to a running sum that starts at +0. Spectra and masks
 * are interleaved [n][2]. */
static void ref_input(const GTCRNGrid* g, const float* spectrum, const float* fwd,
                      float* const out[3])
{
    for (int c = 0; c < 3; ++c) {
        float full[G_BINS];   /* n_bins <= the shipped grid's */
        for (int k = 0; k < g->n_bins; ++k) {
            float re = spectrum[2 * k], im = spectrum[2 * k + 1];
            full[k] = c == 0 ? sqrtf(re * re + im * im + 1e-12f)
                             : (c == 1 ? re : im);
        }
        for (int k = 0; k < g->erb_low_bins; ++k) out[c][k] = full[k];
        for (int b = 0; b < g->erb_high_bands; ++b) {
            float sum = 0.0f;
            for (int k = 0; k < g->erb_high_bins; ++k) {
                float product = fwd[(size_t)k * g->erb_high_bands + b] *
                                full[g->erb_low_bins + k];
                sum = sum + product;
            }
            out[c][g->erb_low_bins + b] = sum;
        }
    }
}

static void ref_output(const GTCRNGrid* g, const float* mask_erb, const float* inv,
                       const float* spectrum, float* enhanced)
{
    for (int k = 0; k < g->n_bins; ++k) {
        float mr, mi;
        if (k < g->erb_low_bins) {
            mr = mask_erb[2 * k];
            mi = mask_erb[2 * k + 1];
        } else {
            mr = 0.0f;
            mi = 0.0f;
            for (int b = 0; b < g->erb_high_bands; ++b) {
                float w = inv[(size_t)b * g->erb_high_bins + (k - g->erb_low_bins)];
                float pr = w * mask_erb[2 * (g->erb_low_bins + b)];
                float pi = w * mask_erb[2 * (g->erb_low_bins + b) + 1];
                mr = mr + pr;
                mi = mi + pi;
            }
        }
        {
            float sr = spectrum[2 * k], si = spectrum[2 * k + 1];
            float a = sr * mr, b = si * mi, c = si * mr, d = sr * mi;
            enhanced[2 * k] = a - b;
            enhanced[2 * k + 1] = c + d;
        }
    }
}

static int test_gtcrn_kernels(void)
{
    static float spectrum[G_BINS * 2], mask[G_BANDS * 2];
    static float fwd[G_HIGH_BINS * G_HIGH_BANDS];
    static float inv[G_HIGH_BANDS * G_HIGH_BINS];
    static const unsigned rates[] = { 0, 0, 0, 64, 7, 2 };
    static const float scales[] = { 1.0f, 1e-3f, 1e3f, 1.0f, 1.0f, 1.0f };
    const int cases = (int)(sizeof(rates) / sizeof(rates[0]));
    const GTCRNGrid* grid = gtcrn_test_grid();
    GTCRNProcessConfig cfg;
    GTCRNProcessMemReq req;
    GTCRNProcess* p;
    void* pool;

    /* The tables are borrowed: the instance reads whatever the arrays hold when called. */
    if (gtcrn_process_config_defaults(&cfg, grid) != 0) return 0;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    if (gtcrn_process_get_mem_size(&cfg, &req) != 0) return 0;
    pool = aligned_alloc(16, ((size_t)req.bytes + 15u) & ~(size_t)15u);
    p = pool ? gtcrn_process_init(pool, (size_t)req.bytes, &cfg) : NULL;
    CHECK(p != NULL, "GTCRN process starts for the kernel tests");

    for (int round = 0; round < 40; ++round) {
        int c = round % cases;
        float m_a[G_BANDS], r_a[G_BANDS], i_a[G_BANDS];
        float m_b[G_BANDS], r_b[G_BANDS], i_b[G_BANDS];
        float* const want[3] = { m_b, r_b, i_b };
        static float en_a[G_BINS * 2], en_b[G_BINS * 2], en_c[G_BINS * 2];
        fill(spectrum, 2 * G_BINS, scales[c], rates[c]);
        fill(fwd, sizeof(fwd) / sizeof(float), 1.0f, rates[c]);
        fill(inv, sizeof(inv) / sizeof(float), 1.0f, rates[c]);
        fill(mask, 2 * G_BANDS, 1.0f, rates[c]);
        if (round % 3 == 0) {  /* sparse tables, like the shipped ones */
            for (size_t i = 0; i < sizeof(fwd) / sizeof(float); ++i)
                if (rng() % 8 != 0) fwd[i] = 0.0f;
            for (size_t i = 0; i < sizeof(inv) / sizeof(float); ++i)
                if (rng() % 8 != 0) inv[i] = 0.0f;
        }

        gtcrn_model_input(p, spectrum, m_a, r_a, i_a);
        ref_input(grid, spectrum, fwd, want);
        CHECK(same_bits(m_a, m_b, G_BANDS) &&
              same_bits(r_a, r_b, G_BANDS) &&
              same_bits(i_a, i_b, G_BANDS),
              "GTCRN model_input equals the plain contract");

        gtcrn_model_output(p, mask, spectrum, en_a);
        ref_output(grid, mask, inv, spectrum, en_b);
        CHECK(same_bits(en_a, en_b, 2 * G_BINS),
              "GTCRN model_output equals the plain contract");

        /* the scalar twins against the plain contract */
        {
            float full_mag[G_BINS], full_re[G_BINS], full_im[G_BINS];
            float t0[G_BANDS], t1[G_BANDS], t2[G_BANDS];
            float mr[G_BINS], mi[G_BINS];
            for (int k = 0; k < G_BINS; ++k) {
                float re = spectrum[2 * k], im = spectrum[2 * k + 1];
                full_mag[k] = sqrtf(re * re + im * im + 1e-12f);
                full_re[k] = re;
                full_im[k] = im;
            }
            erb_band_channel_scalar(grid, full_mag, fwd, t0);
            erb_band_channel_scalar(grid, full_re, fwd, t1);
            erb_band_channel_scalar(grid, full_im, fwd, t2);
            CHECK(same_bits(t0, m_b, G_BANDS) &&
                  same_bits(t1, r_b, G_BANDS) &&
                  same_bits(t2, i_b, G_BANDS),
                  "GTCRN scalar band twin equals the plain contract");
            for (int k = 0; k < G_KEPT; ++k) {
                mr[k] = mask[2 * k];
                mi[k] = mask[2 * k + 1];
            }
            mask_expand_scalar(grid, mask, inv, mr, mi);
            for (int k = 0; k < G_BINS; ++k) {
                float sr = spectrum[2 * k], si = spectrum[2 * k + 1];
                float a = sr * mr[k], b = si * mi[k], cc = si * mr[k],
                      d = sr * mi[k];
                en_c[2 * k] = a - b;
                en_c[2 * k + 1] = cc + d;
            }
            CHECK(same_bits(en_c, en_b, 2 * G_BINS),
                  "GTCRN scalar mask twin equals the plain contract");
        }
#if DF_COMMON_HAVE_NEON
        {
            float full_mag[G_BINS], full_re[G_BINS], full_im[G_BINS];
            const float* const full[3] = { full_mag, full_re, full_im };
            float t0[G_BANDS], t1[G_BANDS], t2[G_BANDS];
            float s0[G_BANDS], s1[G_BANDS], s2[G_BANDS];
            float* const nb[3] = { t0, t1, t2 };
            float mr0[G_BINS], mi0[G_BINS], mr1[G_BINS], mi1[G_BINS];
            for (int k = 0; k < G_BINS; ++k) {
                float re = spectrum[2 * k], im = spectrum[2 * k + 1];
                full_mag[k] = sqrtf(re * re + im * im + 1e-12f);
                full_re[k] = re;
                full_im[k] = im;
            }
            erb_band3_neon(grid, full, fwd, nb);
            erb_band_channel_scalar(grid, full_mag, fwd, s0);
            erb_band_channel_scalar(grid, full_re, fwd, s1);
            erb_band_channel_scalar(grid, full_im, fwd, s2);
            CHECK(same_bits(t0, s0, G_BANDS) &&
                  same_bits(t1, s1, G_BANDS) &&
                  same_bits(t2, s2, G_BANDS),
                  "GTCRN NEON band kernel equals its scalar twin");
            mask_expand_neon(grid, mask, inv, mr0, mi0);
            mask_expand_scalar(grid, mask, inv, mr1, mi1);
            CHECK(same_bits(mr0 + G_KEPT, mr1 + G_KEPT, G_HIGH_BINS) &&
                  same_bits(mi0 + G_KEPT, mi1 + G_KEPT, G_HIGH_BINS),
                  "GTCRN NEON mask kernel equals its scalar twin");
        }
#endif
    }
    free(pool);
    return 1;
}


/* A grid the shipped model never has, so every tail of the NEON kernels runs: mask_expand tiles
 * erb_high_bins (45 = 32 + three quads + 1), the band kernel tiles erb_high_bands
 * (21 = 16 + 4 + 1). The public path, the scalar twins and the NEON kernels all equal the plain
 * contract, so the NEON and the scalar build of this program agree with each other. */
static int test_gtcrn_unshipped_geometry(void)
{
    enum { N_FFT = 128, LOW = 20, HIGH_BANDS = 21, BINS = N_FFT / 2 + 1,
           HIGH_BINS = BINS - LOW, BANDS = LOW + HIGH_BANDS };
    static float spectrum[BINS * 2], mask[BANDS * 2];
    static float fwd[HIGH_BINS * HIGH_BANDS], inv[HIGH_BANDS * HIGH_BINS];
    static const unsigned rates[] = { 0, 0, 0, 64, 7, 2 };
    static const float scales[] = { 1.0f, 1e-3f, 1e3f, 1.0f, 1.0f, 1.0f };
    const GTCRNGridSpec spec = {
        .n_fft = N_FFT, .erb_low_bins = LOW, .erb_high_bands = HIGH_BANDS };
    GTCRNGrid grid;
    GTCRNProcessConfig cfg;
    GTCRNProcessMemReq req;
    GTCRNProcess* p;
    void* pool;

    CHECK(HIGH_BINS % 32 >= 8 && HIGH_BINS % 4 != 0 && HIGH_BANDS % 16 >= 4 &&
          HIGH_BANDS % 4 != 0 && BINS <= G_BINS && BANDS <= G_BANDS,
          "GTCRN the unshipped grid reaches every kernel tail");
    CHECK(gtcrn_grid_make(&grid, &spec) == GTCRN_GRID_OK && grid.n_bins == BINS &&
          grid.erb_high_bins == HIGH_BINS && grid.erb_bands == BANDS,
          "GTCRN the unshipped grid is accepted");
    CHECK(gtcrn_process_config_defaults(&cfg, &grid) == 0, "GTCRN unshipped config");
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    CHECK(gtcrn_process_get_mem_size(&cfg, &req) == 0, "GTCRN unshipped size");
    pool = aligned_alloc(16, ((size_t)req.bytes + 15u) & ~(size_t)15u);
    p = pool ? gtcrn_process_init(pool, (size_t)req.bytes, &cfg) : NULL;
    CHECK(p != NULL, "GTCRN process starts on the unshipped grid");

    for (int round = 0; round < 30; ++round) {
        const int c = round % (int)(sizeof(rates) / sizeof(rates[0]));
        float m_a[BANDS], r_a[BANDS], i_a[BANDS], m_b[BANDS], r_b[BANDS], i_b[BANDS];
        float* const want[3] = { m_b, r_b, i_b };
        float en_a[BINS * 2], en_b[BINS * 2];
        float full_mag[BINS], full_re[BINS], full_im[BINS];
        float t0[BANDS], t1[BANDS], t2[BANDS], mr1[BINS], mi1[BINS];
        fill(spectrum, 2 * BINS, scales[c], rates[c]);
        fill(fwd, sizeof(fwd) / sizeof(float), 1.0f, rates[c]);
        fill(inv, sizeof(inv) / sizeof(float), 1.0f, rates[c]);
        fill(mask, 2 * BANDS, 1.0f, rates[c]);

        gtcrn_model_input(p, spectrum, m_a, r_a, i_a);
        ref_input(&grid, spectrum, fwd, want);
        CHECK(same_bits(m_a, m_b, BANDS) && same_bits(r_a, r_b, BANDS) &&
              same_bits(i_a, i_b, BANDS),
              "GTCRN unshipped grid: model_input equals the plain contract");
        gtcrn_model_output(p, mask, spectrum, en_a);
        ref_output(&grid, mask, inv, spectrum, en_b);
        CHECK(same_bits(en_a, en_b, 2 * BINS),
              "GTCRN unshipped grid: model_output equals the plain contract");

        for (int k = 0; k < BINS; ++k) {
            const float re = spectrum[2 * k], im = spectrum[2 * k + 1];
            full_mag[k] = sqrtf(re * re + im * im + 1e-12f);
            full_re[k] = re;
            full_im[k] = im;
        }
        erb_band_channel_scalar(&grid, full_mag, fwd, t0);
        erb_band_channel_scalar(&grid, full_re, fwd, t1);
        erb_band_channel_scalar(&grid, full_im, fwd, t2);
        CHECK(same_bits(t0, m_b, BANDS) && same_bits(t1, r_b, BANDS) &&
              same_bits(t2, i_b, BANDS),
              "GTCRN unshipped grid: scalar band twin equals the plain contract");
        for (int k = 0; k < LOW; ++k) {
            mr1[k] = mask[2 * k];
            mi1[k] = mask[2 * k + 1];
        }
        mask_expand_scalar(&grid, mask, inv, mr1, mi1);
        for (int k = 0; k < BINS; ++k) {
            const float sr = spectrum[2 * k], si = spectrum[2 * k + 1];
            en_a[2 * k] = sr * mr1[k] - si * mi1[k];
            en_a[2 * k + 1] = si * mr1[k] + sr * mi1[k];
        }
        CHECK(same_bits(en_a, en_b, 2 * BINS),
              "GTCRN unshipped grid: scalar mask twin equals the plain contract");
#if DF_COMMON_HAVE_NEON
        {
            const float* const full[3] = { full_mag, full_re, full_im };
            float* const banded[3] = { t0, t1, t2 };
            float mr0[BINS], mi0[BINS], s0[BANDS], s1[BANDS], s2[BANDS];
            erb_band3_neon(&grid, full, fwd, banded);
            erb_band_channel_scalar(&grid, full_mag, fwd, s0);
            erb_band_channel_scalar(&grid, full_re, fwd, s1);
            erb_band_channel_scalar(&grid, full_im, fwd, s2);
            CHECK(same_bits(t0, s0, BANDS) && same_bits(t1, s1, BANDS) &&
                  same_bits(t2, s2, BANDS),
                  "GTCRN unshipped grid: NEON band kernel equals its scalar twin");
            mask_expand_neon(&grid, mask, inv, mr0, mi0);
            CHECK(same_bits(mr0 + LOW, mr1 + LOW, HIGH_BINS) &&
                  same_bits(mi0 + LOW, mi1 + LOW, HIGH_BINS),
                  "GTCRN unshipped grid: NEON mask kernel equals its scalar twin");
        }
#endif
    }
    free(pool);
    return 1;
}


/* Mask expansion: the contract is the dense sum over every band in
 * ascending order. Skipping a band's zero weights is exact for a finite
 * gain, and a non-finite gain takes the whole row. */
static void ref_expand(const float* gain, const float* inv, float* out)
{
    for (int k = 0; k < DFN2_N_BINS; ++k) {
        float sum = 0.0f;
        for (int b = 0; b < DFN2_N_ERB; ++b) {
            float product = inv[(size_t)b * DFN2_N_BINS + k] * gain[b];
            sum = sum + product;
        }
        out[k] = sum;
    }
}

static int test_dfn2_expand_mask(void)
{
    static float inv[DFN2_N_ERB * DFN2_N_BINS];
    uint16_t lo[DFN2_N_ERB], hi[DFN2_N_ERB];
    float gain[DFN2_N_ERB];
    static const unsigned rates[] = { 0, 0, 0, 0, 40, 5 };

    for (int round = 0; round < 120; ++round) {
        float want[DFN2_N_BINS], twin[DFN2_N_BINS], got[DFN2_N_BINS];
        unsigned rate = rates[round % (int)(sizeof(rates) / sizeof(rates[0]))];
        memset(inv, 0, sizeof(inv));
        for (int b = 0; b < DFN2_N_ERB; ++b) {
            int first, width;
            switch (round % 4) {
            case 0:  /* triangular, ERB-like: widths grow with the band */
                first = b * b * 13 / 31 / 2 + b;
                width = 2 + b * 3 + (int)(rng() % 3);
                break;
            case 1:  /* arbitrary spans, including empty and edge-touching */
                first = (int)(rng() % DFN2_N_BINS);
                width = (int)(rng() % 9);
                break;
            case 2:  /* spans ending exactly at the last bin */
                width = 1 + (int)(rng() % 12);
                first = DFN2_N_BINS - width;
                break;
            default: /* wide, overlapping */
                first = (int)(rng() % 300);
                width = 1 + (int)(rng() % 200);
                break;
            }
            for (int k = first; k < first + width && k < DFN2_N_BINS; ++k)
                inv[(size_t)b * DFN2_N_BINS + k] =
                    (rng() % 11 == 0) ? 0.0f : uniform();
            if (rng() % 7 == 0 && first < DFN2_N_BINS)
                inv[(size_t)b * DFN2_N_BINS + first] = -0.0f;
        }
        if (rate != 0) {
            for (int i = 0; i < 24; ++i)
                inv[rng() % (DFN2_N_ERB * DFN2_N_BINS)] = special();
        }
        erb_spans(inv, DFN2_N_ERB, DFN2_N_BINS, DFN2_N_BINS, 1, lo, hi);
        fill(gain, DFN2_N_ERB, 1.0f, rate);

        ref_expand(gain, inv, want);
        df_common_expand_mask_scalar(gain, inv, lo, hi, DFN2_N_BINS, DFN2_N_ERB,
                                     twin);
        CHECK(same_bits(twin, want, DFN2_N_BINS),
              "DFN2 scalar mask expansion equals the dense contract");
        df_common_expand_mask(gain, inv, lo, hi, DFN2_N_BINS, DFN2_N_ERB, got);
        CHECK(same_bits(got, want, DFN2_N_BINS),
              "DFN2 mask expansion equals the dense contract");
#if DF_COMMON_HAVE_NEON
        df_common_expand_mask_neon(gain, inv, lo, hi, DFN2_N_BINS, DFN2_N_ERB,
                                   got);
        CHECK(same_bits(got, twin, DFN2_N_BINS),
              "DFN2 NEON mask expansion equals its scalar twin");
#endif
    }
    return 1;
}

int main(void)
{
    SK_MATCH_SCALAR_TO_VECTOR_MODE();
    if (!test_dfn2_expand_mask()) return EXIT_FAILURE;
    if (!test_gtcrn_kernels()) return EXIT_FAILURE;
    if (!test_gtcrn_unshipped_geometry()) return EXIT_FAILURE;
    printf("nn kernels: backend=%s/%s ok\n", dfn2_simd_backend(),
           gtcrn_simd_backend());
    return EXIT_SUCCESS;
}
