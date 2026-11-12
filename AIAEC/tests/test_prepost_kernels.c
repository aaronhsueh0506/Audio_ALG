/* Bit-exactness of the model-local NEON kernels against their scalar twins
 * and against a plain statement of the contract, on random values and on
 * every special value (NaN, signed zero, infinity, subnormal, extremes). The
 * kernels are static, so this file includes their translation units.
 * KERNEL_TEST_ULCNET selects ulcnet_model_io.c, KERNEL_TEST_DEEPVQE selects
 * deepvqe_prepost.c. The grid is the argument grid=<sample rate>,<n_fft>. */
/* posix_memalign is declared only with this under -std=c11 on glibc. */
#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility/simd_test_mode.h"

#if defined(KERNEL_TEST_ULCNET)
#include "ulcnet_model_io.c"
#elif defined(KERNEL_TEST_DEEPVQE)
#include "deepvqe_prepost.c"
#else
#error "define KERNEL_TEST_ULCNET or KERNEL_TEST_DEEPVQE"
#endif

#define BINS_MAX 513

/* The grid under test (run-time). */
static int g_sample_rate = 16000;
static int g_n_fft = 512;
static int g_bins = 257;
#define BINS g_bins

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
        0x7fc12345u, 0x33800000u, 0x7e800000u, 0x1e3ce508u, 0x9e3ce508u };
    uint32_t b = bits[rng() % (sizeof(bits) / sizeof(bits[0]))];
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* Random values; 1/rate of them replaced by a special value (0: none). */
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

static const unsigned kRates[] = { 0, 0, 0, 97, 9, 2 };
static const float kScales[] = { 1.0f, 1e-3f, 1e3f, 1.0f, 1.0f, 1.0f };
#define KCASES ((int)(sizeof(kRates) / sizeof(kRates[0])))

/* The finite predicate over strided rows, against a plain double loop; the
 * gaps between rows hold non-finite values that must not matter. */
static int test_rows_all_finite(void)
{
    static float buf[8 * 64];
    for (int round = 0; round < 4000; ++round) {
        size_t width = 1 + rng() % 40;
        size_t stride = width + rng() % 24;
        size_t rows = 1 + rng() % 8;
        int want = 1;
        for (size_t i = 0; i < sizeof(buf) / sizeof(float); ++i) buf[i] = uniform();
        for (size_t r = 0; r < rows; ++r)
            for (size_t i = width; i < stride; ++i) buf[r * stride + i] = special();
        if (round % 3 != 0) {
            size_t r = rng() % rows, i = rng() % width;
            buf[r * stride + i] = special();
        }
        for (size_t r = 0; r < rows; ++r)
            for (size_t i = 0; i < width; ++i)
                if (!isfinite(buf[r * stride + i])) want = 0;
        CHECK(skn_all_finite_rows_f32(buf, rows, stride, width) == want,
              "skn_all_finite_rows_f32 equals the plain predicate");
        CHECK(skn_all_finite_rows_f32_scalar(buf, rows, stride, width) == want,
              "skn_all_finite_rows_f32 scalar twin equals the plain predicate");
    }
    return 1;
}

#if defined(KERNEL_TEST_ULCNET)
/* State table of the shipped graph (pinned to the exporter by
 * test_export_streaming_ulcnet.py). */
#define STATE_CHANNELS 32
#define STATE_SCORE_FRAMES 5
#define STATE_GRU_ROWS 4
#define STATE_GRU_HIDDEN 128
#define STATE_TA_BINS(n_bins) (((n_bins) + 9) / 10)
#define STATE_DEPTH 12

/* A state over the grid under test with the application's four recurrent buffers
 * registered: the window checks below need real windows. */
static UlcnetModelIoState* make_state(void)
{
    const UlcnetGridSpec spec = {
        .sample_rate = g_sample_rate, .n_fft = g_n_fft, .delay_depth = STATE_DEPTH };
    const size_t kv = (size_t)STATE_CHANNELS * STATE_DEPTH * (size_t)STATE_TA_BINS(g_bins);
    const size_t logit = (size_t)STATE_CHANNELS * STATE_SCORE_FRAMES * STATE_DEPTH;
    const size_t gru = (size_t)STATE_GRU_ROWS * STATE_GRU_HIDDEN;
    UlcnetModelIoStateBuffers buffers;
    UlcnetModelIoDescriptor d;
    UlcnetModelIoMemReq req;
    UlcnetGrid grid;
    void* pool;
    if (ulcnet_grid_make(&grid, &spec) != ULCNET_GRID_OK) return NULL;
    if (ulcnet_model_io_descriptor_default(&grid, &d) != 0) return NULL;
    if (ulcnet_model_io_get_mem_requirements(&d, &req) != 0) return NULL;
    if (posix_memalign(&pool, req.alignment, req.bytes) != 0) return NULL;
    buffers = (UlcnetModelIoStateBuffers){
        .key_history = (float*)malloc(kv * sizeof(float)), .key_history_elements = kv,
        .value_history = (float*)malloc(kv * sizeof(float)), .value_history_elements = kv,
        .logit_history = (float*)malloc(logit * sizeof(float)), .logit_history_elements = logit,
        .h_gru = (float*)malloc(gru * sizeof(float)), .gru_hidden_elements = gru,
    };
    if (!buffers.key_history || !buffers.value_history || !buffers.logit_history ||
        !buffers.h_gru) return NULL;
    return ulcnet_model_io_init(pool, req.bytes, &d, &buffers);
}

#if SK_HAVE_NEON
static int same_features(const UlcnetModelIoState* a, const UlcnetModelIoState* b)
{
    return same_bits(a->error_mag, b->error_mag, BINS) &&
           same_bits(a->far_mag, b->far_mag, BINS) &&
           same_bits(a->error_cos, b->error_cos, BINS) &&
           same_bits(a->error_sin, b->error_sin, BINS) &&
           same_bits(a->error_ri, b->error_ri, 2 * BINS);
}
#endif

/* The contract, per bin, with the same libm calls. */
static void ref_front_end(const float* er, const float* ei, const float* fr,
                          const float* fi, float* mag, float* far_mag,
                          float* cos_out, float* sin_out, float* ri)
{
    for (int k = 0; k < BINS; ++k) {
        float c_re = copysignf(powf(fabsf(er[k]), 0.3f), er[k]);
        float c_im = copysignf(powf(fabsf(ei[k]), 0.3f), ei[k]);
        float f_re = copysignf(powf(fabsf(fr[k]), 0.3f), fr[k]);
        float f_im = copysignf(powf(fabsf(fi[k]), 0.3f), fi[k]);
        float norm = hypotf(c_re, c_im);
        float a = c_re * c_re, b = c_im * c_im, c = f_re * f_re, d = f_im * f_im;
        mag[k] = sqrtf(a + b + 1e-12f);
        far_mag[k] = sqrtf(c + d + 1e-12f);
        cos_out[k] = norm > 0.0f ? c_re / norm : 1.0f;
        sin_out[k] = norm > 0.0f ? c_im / norm : 0.0f;
        ri[2 * k] = c_re;
        ri[2 * k + 1] = c_im;
    }
}

static int test_ulcnet_kernels(void)
{
    UlcnetModelIoState* a = make_state();
    UlcnetModelIoState* b = make_state();
    UlcnetModelIoState* c = make_state();
    static float er[BINS_MAX], ei[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX];
    static float w_mag[BINS_MAX], w_far[BINS_MAX], w_cos[BINS_MAX], w_sin[BINS_MAX],
        w_ri[2 * BINS_MAX];
    static float enh_a[2][BINS_MAX], enh_b[2][BINS_MAX], enh_c[2][BINS_MAX];
    UlcnetModelIoInputs in;
    UlcnetModelIoOutputs out;
    CHECK(a && b && c, "state allocation");

    for (int round = 0; round < 60; ++round) {
        int k = round % KCASES;
        fill(er, BINS, kScales[k], kRates[k]);
        fill(ei, BINS, kScales[k], kRates[k]);
        fill(fr, BINS, kScales[k], kRates[k]);
        fill(fi, BINS, kScales[k], kRates[k]);
        if (round % 5 == 0) {  /* exact zero vectors and one-sided zeros */
            for (int i = 0; i < BINS; i += 3) { er[i] = 0.0f; ei[i] = 0.0f; }
            for (int i = 1; i < BINS; i += 7) ei[i] = 0.0f;
        }

        ref_front_end(er, ei, fr, fi, w_mag, w_far, w_cos, w_sin, w_ri);
        CHECK(ulcnet_model_io_prepare(a, er, ei, fr, fi, &in, &out) == 0,
              "prepare");
        CHECK(same_bits(a->error_mag, w_mag, BINS) &&
              same_bits(a->far_mag, w_far, BINS) &&
              same_bits(a->error_cos, w_cos, BINS) &&
              same_bits(a->error_sin, w_sin, BINS) &&
              same_bits(a->error_ri, w_ri, 2 * BINS),
              "prepare equals the plain contract");
        {
            const FramePlanes planes_b = frame_planes(b, 0u);
            front_end_scalar(b, &planes_b, er, ei, fr, fi);
        }
        CHECK(same_bits(b->error_mag, w_mag, BINS) &&
              same_bits(b->far_mag, w_far, BINS) &&
              same_bits(b->error_cos, w_cos, BINS) &&
              same_bits(b->error_sin, w_sin, BINS) &&
              same_bits(b->error_ri, w_ri, 2 * BINS),
              "front end scalar twin equals the plain contract");
#if SK_HAVE_NEON
        {
            const FramePlanes planes_c = frame_planes(c, 0u);
            front_end_neon(c, &planes_c, er, ei, fr, fi);
        }
        CHECK(same_features(c, b), "front end NEON kernel equals its scalar twin");
#endif

        /* back end: mask from the graph (any values), compressed error kept */
        fill(a->output, 2 * BINS, 1.0f, kRates[k]);
        fill(a->error_ri, 2 * BINS, 1.0f, kRates[k]);
        memcpy(b->output, a->output, 2 * BINS * sizeof(float));
        memcpy(b->error_ri, a->error_ri, 2 * BINS * sizeof(float));
        back_end_scalar(b, 0u, enh_b[0], enh_b[1]);
        back_end(a, 0u, enh_a[0], enh_a[1]);
        CHECK(same_bits(enh_a[0], enh_b[0], BINS) && same_bits(enh_a[1], enh_b[1], BINS),
              "back end equals its scalar twin");
        for (int i = 0; i < BINS; ++i) {
            float er_ = a->error_ri[2 * i], ei_ = a->error_ri[2 * i + 1];
            float mr = a->output[i], mi = a->output[BINS + i];
            float p0 = er_ * mr, p1 = ei_ * mi, p2 = er_ * mi, p3 = ei_ * mr;
            float re = p0 - p1, im = p2 + p3;
            enh_c[0][i] = copysignf(powf(fabsf(re), 1.0f / 0.3f), re);
            enh_c[1][i] = copysignf(powf(fabsf(im), 1.0f / 0.3f), im);
        }
        CHECK(same_bits(enh_c[0], enh_b[0], BINS) && same_bits(enh_c[1], enh_b[1], BINS),
              "back end scalar twin equals the plain contract");
#if SK_HAVE_NEON
        back_end_neon(a, 0u, enh_a[0], enh_a[1]);
        CHECK(same_bits(enh_a[0], enh_b[0], BINS) && same_bits(enh_a[1], enh_b[1], BINS),
              "back end NEON kernel equals its scalar twin");
#endif
    }
    return 1;
}

/* frame_is_finite reads the mask and the whole of every registered window: a
 * non-finite value in any slot, the oldest and the dropped one included, is seen. */
static int test_ulcnet_frame_finite(void)
{
    UlcnetModelIoState* s = make_state();
    const size_t channels = STATE_CHANNELS;
    const size_t ta_bins = (size_t)STATE_TA_BINS(g_bins);
    float bad = INFINITY;
    size_t key_stride, logit_stride, depth, newest_logit;
    CHECK(s != NULL, "state allocation");
    /* The window views are the registered buffers: a missing registration must
     * not turn the scans below into empty loops. */
    CHECK(s->key_history != NULL && s->value_history != NULL &&
          s->logit_history != NULL && s->h_gru != NULL, "state registration");
    key_stride = s->key_history_elements / channels;
    logit_stride = s->logit_history_elements / channels;
    depth = logit_stride / STATE_SCORE_FRAMES;
    newest_logit = logit_stride - depth;
    CHECK(key_stride == (size_t)STATE_DEPTH * ta_bins && depth == (size_t)STATE_DEPTH,
          "registered element counts follow the table");
    fill(s->output, s->spectrum_ri_elements, 1.0f, 0);
    fill(s->h_gru, s->gru_hidden_elements, 1.0f, 0);
    fill(s->key_history, s->key_history_elements, 1.0f, 0);
    fill(s->value_history, s->value_history_elements, 1.0f, 0);
    fill(s->logit_history, s->logit_history_elements, 1.0f, 0);
    CHECK(frame_is_finite(s) == 1, "a finite frame is accepted");
    for (size_t ch = 0; ch < channels; ch += 5) {
        for (size_t t = 0; t < ta_bins; t += 7) {
            float* slot = &s->key_history[ch * key_stride + t];
            float keep = *slot;
            *slot = bad;
            CHECK(frame_is_finite(s) == 0, "a non-finite newest key is refused");
            *slot = keep;
            slot = &s->value_history[ch * key_stride + t];
            keep = *slot;
            *slot = NAN;
            CHECK(frame_is_finite(s) == 0, "a non-finite newest value is refused");
            *slot = keep;
        }
        for (size_t d = 0; d < depth; ++d) {
            float* slot = &s->logit_history[ch * logit_stride + newest_logit + d];
            float keep = *slot;
            *slot = -bad;
            CHECK(frame_is_finite(s) == 0, "a non-finite newest logit is refused");
            *slot = keep;
        }
        /* an older frame is read as well */
        {
            float* old_key = &s->key_history[ch * key_stride + ta_bins];
            float* oldest_key = &s->key_history[ch * key_stride + key_stride - 1];
            float* old_value = &s->value_history[ch * key_stride + key_stride - 1];
            float* old_logit = &s->logit_history[ch * logit_stride];
            float keep_key = *old_key, keep_oldest = *oldest_key;
            float keep_value = *old_value, keep_logit = *old_logit;
            *old_key = NAN;
            CHECK(frame_is_finite(s) == 0, "a non-finite older key slot is refused");
            *old_key = keep_key;
            *oldest_key = NAN;
            CHECK(frame_is_finite(s) == 0, "a non-finite oldest key slot is refused");
            *oldest_key = keep_oldest;
            *old_value = INFINITY;
            CHECK(frame_is_finite(s) == 0, "a non-finite oldest value slot is refused");
            *old_value = keep_value;
            *old_logit = NAN;
            CHECK(frame_is_finite(s) == 0, "a non-finite older logit frame is refused");
            *old_logit = keep_logit;
        }
        CHECK(frame_is_finite(s) == 1, "a restored window is accepted again");
    }
    s->h_gru[s->gru_hidden_elements - 1] = NAN;
    CHECK(frame_is_finite(s) == 0, "a non-finite hidden state is refused");
    return 1;
}
#endif

#if defined(KERNEL_TEST_DEEPVQE)
static int test_deepvqe_kernels(void)
{
    static float re[BINS_MAX], im[BINS_MAX], got[2 * BINS_MAX], want[2 * BINS_MAX];
    const float exponent = DEEPVQE_COMPRESSION_EXP - 1.0f;
    for (int round = 0; round < 60; ++round) {
        int k = round % KCASES;
        fill(re, BINS, kScales[k], kRates[k]);
        fill(im, BINS, kScales[k], kRates[k]);
        compress_planar(re, im, got, got + BINS, BINS);
        for (int i = 0; i < BINS; ++i) {
            float a = re[i] * re[i], b = im[i] * im[i];
            float scale = powf(sqrtf(a + b + 1e-12f), exponent);
            want[i] = re[i] * scale;
            want[BINS + i] = im[i] * scale;
        }
        CHECK(same_bits(got, want, 2 * BINS),
              "compress_planar equals the plain contract");
    }
    return 1;
}
#endif

int main(int argc, char** argv)
{
    SK_MATCH_SCALAR_TO_VECTOR_MODE();
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "grid=", 5) != 0 ||
            sscanf(argv[i] + 5, "%d,%d", &g_sample_rate, &g_n_fft) != 2) {
            fprintf(stderr, "usage: %s [grid=<sample rate>,<n_fft>]\n", argv[0]);
            return EXIT_FAILURE;
        }
        g_bins = g_n_fft / 2 + 1;
        if (g_bins > BINS_MAX) return EXIT_FAILURE;
    }
    if (!test_rows_all_finite()) return EXIT_FAILURE;
#if defined(KERNEL_TEST_ULCNET)
    if (!test_ulcnet_kernels()) return EXIT_FAILURE;
    if (!test_ulcnet_frame_finite()) return EXIT_FAILURE;
#else
    if (!test_deepvqe_kernels()) return EXIT_FAILURE;
#endif
    printf("prepost kernels: backend=%s ok\n", SK_NEON_BACKEND_NAME);
    return EXIT_SUCCESS;
}
