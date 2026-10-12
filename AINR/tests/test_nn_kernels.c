/* Bit-exactness of the model-local NEON kernels against their scalar twins
 * and against a plain statement of the contract, on random values and on
 * every special value (NaN, signed zero, infinity, subnormal, extremes). The
 * kernels are static, so this file includes their translation units. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_process.c"
#include "GTCRN/gtcrn_process.c"

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
 * ascending order added to a running sum that starts at +0. */
static void ref_input(const float spectrum[GTCRN_N_BINS][2], const float* fwd,
                      float* const out[3])
{
    for (int c = 0; c < 3; ++c) {
        float full[GTCRN_N_BINS];
        for (int k = 0; k < GTCRN_N_BINS; ++k) {
            float re = spectrum[k][0], im = spectrum[k][1];
            full[k] = c == 0 ? sqrtf(re * re + im * im + 1e-12f)
                             : (c == 1 ? re : im);
        }
        for (int k = 0; k < GTCRN_MODEL_ERB_KEPT; ++k) out[c][k] = full[k];
        for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b) {
            float sum = 0.0f;
            for (int k = 0; k < GTCRN_MODEL_ERB_HIGH_BINS; ++k) {
                float product = fwd[(size_t)k * GTCRN_MODEL_ERB_HIGH_BANDS + b] *
                                full[GTCRN_MODEL_ERB_KEPT + k];
                sum = sum + product;
            }
            out[c][GTCRN_MODEL_ERB_KEPT + b] = sum;
        }
    }
}

static void ref_output(const float mask_erb[GTCRN_MODEL_ERB_BANDS][2],
                       const float* inv, const float spectrum[GTCRN_N_BINS][2],
                       float enhanced[GTCRN_N_BINS][2])
{
    for (int k = 0; k < GTCRN_N_BINS; ++k) {
        float mr, mi;
        if (k < GTCRN_MODEL_ERB_KEPT) {
            mr = mask_erb[k][0];
            mi = mask_erb[k][1];
        } else {
            mr = 0.0f;
            mi = 0.0f;
            for (int b = 0; b < GTCRN_MODEL_ERB_HIGH_BANDS; ++b) {
                float w = inv[(size_t)b * GTCRN_MODEL_ERB_HIGH_BINS +
                              (k - GTCRN_MODEL_ERB_KEPT)];
                float pr = w * mask_erb[GTCRN_MODEL_ERB_KEPT + b][0];
                float pi = w * mask_erb[GTCRN_MODEL_ERB_KEPT + b][1];
                mr = mr + pr;
                mi = mi + pi;
            }
        }
        {
            float sr = spectrum[k][0], si = spectrum[k][1];
            float a = sr * mr, b = si * mi, c = si * mr, d = sr * mi;
            enhanced[k][0] = a - b;
            enhanced[k][1] = c + d;
        }
    }
}

static int test_gtcrn_kernels(void)
{
    static float spectrum[GTCRN_N_BINS][2], mask[GTCRN_MODEL_ERB_BANDS][2];
    static float fwd[GTCRN_MODEL_ERB_HIGH_BINS * GTCRN_MODEL_ERB_HIGH_BANDS];
    static float inv[GTCRN_MODEL_ERB_HIGH_BANDS * GTCRN_MODEL_ERB_HIGH_BINS];
    static const unsigned rates[] = { 0, 0, 0, 64, 7, 2 };
    static const float scales[] = { 1.0f, 1e-3f, 1e3f, 1.0f, 1.0f, 1.0f };
    const int cases = (int)(sizeof(rates) / sizeof(rates[0]));

    for (int round = 0; round < 40; ++round) {
        int c = round % cases;
        float m_a[GTCRN_MODEL_ERB_BANDS], r_a[GTCRN_MODEL_ERB_BANDS],
            i_a[GTCRN_MODEL_ERB_BANDS];
        float m_b[GTCRN_MODEL_ERB_BANDS], r_b[GTCRN_MODEL_ERB_BANDS],
            i_b[GTCRN_MODEL_ERB_BANDS];
        float* const want[3] = { m_b, r_b, i_b };
        static float en_a[GTCRN_N_BINS][2], en_b[GTCRN_N_BINS][2],
            en_c[GTCRN_N_BINS][2];
        fill(&spectrum[0][0], 2 * GTCRN_N_BINS, scales[c], rates[c]);
        fill(fwd, sizeof(fwd) / sizeof(float), 1.0f, rates[c]);
        fill(inv, sizeof(inv) / sizeof(float), 1.0f, rates[c]);
        fill(&mask[0][0], 2 * GTCRN_MODEL_ERB_BANDS, 1.0f, rates[c]);
        if (round % 3 == 0) {  /* sparse tables, like the shipped ones */
            for (size_t i = 0; i < sizeof(fwd) / sizeof(float); ++i)
                if (rng() % 8 != 0) fwd[i] = 0.0f;
            for (size_t i = 0; i < sizeof(inv) / sizeof(float); ++i)
                if (rng() % 8 != 0) inv[i] = 0.0f;
        }

        gtcrn_model_input(spectrum, fwd, m_a, r_a, i_a);
        ref_input(spectrum, fwd, want);
        CHECK(same_bits(m_a, m_b, GTCRN_MODEL_ERB_BANDS) &&
              same_bits(r_a, r_b, GTCRN_MODEL_ERB_BANDS) &&
              same_bits(i_a, i_b, GTCRN_MODEL_ERB_BANDS),
              "GTCRN model_input equals the plain contract");

        gtcrn_model_output(mask, inv, spectrum, en_a);
        ref_output(mask, inv, spectrum, en_b);
        CHECK(same_bits(&en_a[0][0], &en_b[0][0], 2 * GTCRN_N_BINS),
              "GTCRN model_output equals the plain contract");

        /* the scalar twins against the plain contract */
        {
            float full_mag[GTCRN_N_BINS], full_re[GTCRN_N_BINS],
                full_im[GTCRN_N_BINS];
            float t0[GTCRN_MODEL_ERB_BANDS], t1[GTCRN_MODEL_ERB_BANDS],
                t2[GTCRN_MODEL_ERB_BANDS];
            float mr[GTCRN_N_BINS], mi[GTCRN_N_BINS];
            for (int k = 0; k < GTCRN_N_BINS; ++k) {
                float re = spectrum[k][0], im = spectrum[k][1];
                full_mag[k] = sqrtf(re * re + im * im + 1e-12f);
                full_re[k] = re;
                full_im[k] = im;
            }
            erb_band_channel_scalar(full_mag, fwd, t0);
            erb_band_channel_scalar(full_re, fwd, t1);
            erb_band_channel_scalar(full_im, fwd, t2);
            CHECK(same_bits(t0, m_b, GTCRN_MODEL_ERB_BANDS) &&
                  same_bits(t1, r_b, GTCRN_MODEL_ERB_BANDS) &&
                  same_bits(t2, i_b, GTCRN_MODEL_ERB_BANDS),
                  "GTCRN scalar band twin equals the plain contract");
            for (int k = 0; k < GTCRN_MODEL_ERB_KEPT; ++k) {
                mr[k] = mask[k][0];
                mi[k] = mask[k][1];
            }
            mask_expand_scalar(mask, inv, mr, mi);
            for (int k = 0; k < GTCRN_N_BINS; ++k) {
                float sr = spectrum[k][0], si = spectrum[k][1];
                float a = sr * mr[k], b = si * mi[k], cc = si * mr[k],
                      d = sr * mi[k];
                en_c[k][0] = a - b;
                en_c[k][1] = cc + d;
            }
            CHECK(same_bits(&en_c[0][0], &en_b[0][0], 2 * GTCRN_N_BINS),
                  "GTCRN scalar mask twin equals the plain contract");
        }
#if DF_COMMON_HAVE_NEON
        {
            float full_mag[GTCRN_N_BINS], full_re[GTCRN_N_BINS],
                full_im[GTCRN_N_BINS];
            const float* const full[3] = { full_mag, full_re, full_im };
            float t0[GTCRN_MODEL_ERB_BANDS], t1[GTCRN_MODEL_ERB_BANDS],
                t2[GTCRN_MODEL_ERB_BANDS], s0[GTCRN_MODEL_ERB_BANDS],
                s1[GTCRN_MODEL_ERB_BANDS], s2[GTCRN_MODEL_ERB_BANDS];
            float* const nb[3] = { t0, t1, t2 };
            float mr0[GTCRN_N_BINS], mi0[GTCRN_N_BINS], mr1[GTCRN_N_BINS],
                mi1[GTCRN_N_BINS];
            for (int k = 0; k < GTCRN_N_BINS; ++k) {
                float re = spectrum[k][0], im = spectrum[k][1];
                full_mag[k] = sqrtf(re * re + im * im + 1e-12f);
                full_re[k] = re;
                full_im[k] = im;
            }
            erb_band3_neon(full, fwd, nb);
            erb_band_channel_scalar(full_mag, fwd, s0);
            erb_band_channel_scalar(full_re, fwd, s1);
            erb_band_channel_scalar(full_im, fwd, s2);
            CHECK(same_bits(t0, s0, GTCRN_MODEL_ERB_BANDS) &&
                  same_bits(t1, s1, GTCRN_MODEL_ERB_BANDS) &&
                  same_bits(t2, s2, GTCRN_MODEL_ERB_BANDS),
                  "GTCRN NEON band kernel equals its scalar twin");
            mask_expand_neon(mask, inv, mr0, mi0);
            mask_expand_scalar(mask, inv, mr1, mi1);
            CHECK(same_bits(mr0 + GTCRN_MODEL_ERB_KEPT, mr1 + GTCRN_MODEL_ERB_KEPT,
                            GTCRN_MODEL_ERB_HIGH_BINS) &&
                  same_bits(mi0 + GTCRN_MODEL_ERB_KEPT, mi1 + GTCRN_MODEL_ERB_KEPT,
                            GTCRN_MODEL_ERB_HIGH_BINS),
                  "GTCRN NEON mask kernel equals its scalar twin");
        }
#endif
    }
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
    CHECK(test_dfn2_expand_mask(), "DFN2 kernels");
    CHECK(test_gtcrn_kernels(), "GTCRN kernels");
    printf("nn kernels: backend=%s/%s ok\n", dfn2_simd_backend(),
           gtcrn_simd_backend());
    return EXIT_SUCCESS;
}
