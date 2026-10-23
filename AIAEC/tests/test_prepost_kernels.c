/* Bit-exactness of the model-local NEON kernels against their scalar twins
 * and against a plain statement of the contract, on random values and on
 * every special value (NaN, signed zero, infinity, subnormal, extremes). The
 * kernels are static, so this file includes their translation units.
 * KERNEL_TEST_ULCNET selects ulcnet_model_io.c, KERNEL_TEST_DEEPVQE selects
 * deepvqe_prepost.c. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(KERNEL_TEST_ULCNET)
#include "ulcnet_model_io.c"
#elif defined(KERNEL_TEST_DEEPVQE)
#include "deepvqe_prepost.c"
#else
#error "define KERNEL_TEST_ULCNET or KERNEL_TEST_DEEPVQE"
#endif

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
#define BINS ULCNET_MODEL_IO_BINS

static UlcnetModelIoState* make_state(void)
{
    UlcnetModelIoDescriptor d;
    UlcnetModelIoMemReq req;
    void* pool;
    if (ulcnet_model_io_descriptor_default(12, &d) != 0) return NULL;
    if (ulcnet_model_io_get_mem_requirements(&d, &req) != 0) return NULL;
    if (posix_memalign(&pool, req.alignment, req.bytes) != 0) return NULL;
    return ulcnet_model_io_init(pool, req.bytes, &d);
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
    static float er[BINS], ei[BINS], fr[BINS], fi[BINS];
    static float w_mag[BINS], w_far[BINS], w_cos[BINS], w_sin[BINS], w_ri[2 * BINS];
    static float enh_a[2][BINS], enh_b[2][BINS], enh_c[2][BINS];
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
        front_end_scalar(b, er, ei, fr, fi);
        CHECK(same_bits(b->error_mag, w_mag, BINS) &&
              same_bits(b->far_mag, w_far, BINS) &&
              same_bits(b->error_cos, w_cos, BINS) &&
              same_bits(b->error_sin, w_sin, BINS) &&
              same_bits(b->error_ri, w_ri, 2 * BINS),
              "front end scalar twin equals the plain contract");
#if SK_HAVE_NEON
        front_end_neon(c, er, ei, fr, fi);
        CHECK(same_features(c, b), "front end NEON kernel equals its scalar twin");
#endif

        /* back end: mask from the graph (any values), compressed error kept */
        fill(a->output, 2 * BINS, 1.0f, kRates[k]);
        fill(a->error_ri, 2 * BINS, 1.0f, kRates[k]);
        memcpy(b->output, a->output, 2 * BINS * sizeof(float));
        memcpy(b->error_ri, a->error_ri, 2 * BINS * sizeof(float));
        back_end_scalar(b, enh_b[0], enh_b[1]);
        back_end(a, enh_a[0], enh_a[1]);
        CHECK(same_bits(&enh_a[0][0], &enh_b[0][0], 2 * BINS),
              "back end equals its scalar twin");
        for (int i = 0; i < BINS; ++i) {
            float er_ = a->error_ri[2 * i], ei_ = a->error_ri[2 * i + 1];
            float mr = a->output[i], mi = a->output[BINS + i];
            float p0 = er_ * mr, p1 = ei_ * mi, p2 = er_ * mi, p3 = ei_ * mr;
            float re = p0 - p1, im = p2 + p3;
            enh_c[0][i] = copysignf(powf(fabsf(re), 1.0f / 0.3f), re);
            enh_c[1][i] = copysignf(powf(fabsf(im), 1.0f / 0.3f), im);
        }
        CHECK(same_bits(&enh_c[0][0], &enh_b[0][0], 2 * BINS),
              "back end scalar twin equals the plain contract");
#if SK_HAVE_NEON
        back_end_neon(a, enh_a[0], enh_a[1]);
        CHECK(same_bits(&enh_a[0][0], &enh_b[0][0], 2 * BINS),
              "back end NEON kernel equals its scalar twin");
#endif
    }
    return 1;
}

/* frame_is_finite reads exactly the tensors and slots one frame writes. */
static int test_ulcnet_frame_finite(void)
{
    UlcnetModelIoState* s = make_state();
    const size_t channels = ULCNET_MODEL_IO_TA_CHANNELS;
    const size_t key_stride = s->key_history_elements / channels;
    const size_t logit_stride = s->logit_history_elements / channels;
    const size_t depth = logit_stride / ULCNET_MODEL_IO_SCORE_WINDOW;
    const size_t newest_logit = logit_stride - depth;
    float bad = INFINITY;
    CHECK(s != NULL, "state allocation");
    fill(s->output, s->spectrum_ri_elements, 1.0f, 0);
    fill(s->h_gru, s->gru_hidden_elements, 1.0f, 0);
    fill(s->key_history, s->key_history_elements, 1.0f, 0);
    fill(s->value_history, s->value_history_elements, 1.0f, 0);
    fill(s->logit_history, s->logit_history_elements, 1.0f, 0);
    CHECK(frame_is_finite(s) == 1, "a finite frame is accepted");
    for (size_t ch = 0; ch < channels; ch += 5) {
        for (size_t t = 0; t < ULCNET_MODEL_IO_TA_BINS; t += 7) {
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
        /* an older frame is not read */
        s->key_history[ch * key_stride + ULCNET_MODEL_IO_TA_BINS] = NAN;
        s->logit_history[ch * logit_stride] = NAN;
        CHECK(frame_is_finite(s) == 1, "older frames are not read");
        s->key_history[ch * key_stride + ULCNET_MODEL_IO_TA_BINS] = 0.5f;
        s->logit_history[ch * logit_stride] = 0.5f;
    }
    s->h_gru[s->gru_hidden_elements - 1] = NAN;
    CHECK(frame_is_finite(s) == 0, "a non-finite hidden state is refused");
    return 1;
}
#endif

#if defined(KERNEL_TEST_DEEPVQE)
static int test_deepvqe_kernels(void)
{
    static float re[AIAEC_N_BINS], im[AIAEC_N_BINS], got[2 * AIAEC_N_BINS],
        want[2 * AIAEC_N_BINS];
    const float exponent = DEEPVQE_COMPRESSION_EXP - 1.0f;
    for (int round = 0; round < 60; ++round) {
        int k = round % KCASES;
        fill(re, AIAEC_N_BINS, kScales[k], kRates[k]);
        fill(im, AIAEC_N_BINS, kScales[k], kRates[k]);
        compress_planar(re, im, got);
        for (int i = 0; i < AIAEC_N_BINS; ++i) {
            float a = re[i] * re[i], b = im[i] * im[i];
            float scale = powf(sqrtf(a + b + 1e-12f), exponent);
            want[i] = re[i] * scale;
            want[AIAEC_N_BINS + i] = im[i] * scale;
        }
        CHECK(same_bits(got, want, 2 * AIAEC_N_BINS),
              "compress_planar equals the plain contract");
    }
    return 1;
}

/* newest_frame_finite against a plain per-channel loop. */
static int test_deepvqe_newest_frame(void)
{
    DeepVqePrepostConfig cfg;
    DeepVqePrepost* p;
    deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ,
                                    DEEPVQE_PREPOST_DEFAULT_D);
    p = deepvqe_prepost_create(&cfg);
    CHECK(p != NULL, "prepost allocation");
    for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        float* state = (float*)p->state[id];
        int dims[DEEPVQE_STATE_MAX_RANK];
        size_t channels, frames, width, newest;
        fill(state, p->state_elements[id], 1.0f, 0);
        CHECK(newest_frame_finite(p, id) == 1, "a finite state is accepted");
        if (id == DEEPVQE_STATE_H_GRU) {
            state[p->state_elements[id] - 1] = NAN;
            CHECK(newest_frame_finite(p, id) == 0, "a bad hidden state is refused");
            continue;
        }
        CHECK(deepvqe_prepost_state_shape(id, p->descriptor.delay_depth, dims) > 0,
              "state shape");
        channels = (size_t)dims[1];
        frames = (size_t)dims[2];
        width = (size_t)dims[3];
        newest = (id == DEEPVQE_STATE_ALIGN_KEY_RING ||
                  id == DEEPVQE_STATE_ALIGN_VALUE_RING) ? 0u : frames - 1u;
        for (size_t ch = 0; ch < channels; ++ch) {
            for (size_t i = 0; i < width; i += (width > 33 ? 5 : 1)) {
                float* slot = &state[(ch * frames + newest) * width + i];
                float keep = *slot;
                *slot = special();
                {
                    int want = isfinite(*slot) ? 1 : 0;
                    CHECK(newest_frame_finite(p, id) == want,
                          "the newest frame decides");
                }
                *slot = keep;
            }
            if (frames > 1) {  /* an older frame is not read */
                size_t older = (newest == 0u) ? 1u : 0u;
                float* slot = &state[(ch * frames + older) * width];
                *slot = NAN;
                CHECK(newest_frame_finite(p, id) == 1, "older frames are not read");
                *slot = 0.25f;
            }
        }
    }
    deepvqe_prepost_destroy(p);
    return 1;
}
#endif

int main(void)
{
    CHECK(test_rows_all_finite(), "finite predicate over rows");
#if defined(KERNEL_TEST_ULCNET)
    CHECK(test_ulcnet_kernels(), "Align-ULCNet front and back end");
    CHECK(test_ulcnet_frame_finite(), "Align-ULCNet frame finite");
#else
    CHECK(test_deepvqe_kernels(), "DeepVQE-S compression");
    CHECK(test_deepvqe_newest_frame(), "DeepVQE-S newest frame");
#endif
    printf("prepost kernels: backend=%s ok\n", SK_HAVE_NEON ? "aarch64-neon" : "scalar");
    return EXIT_SUCCESS;
}
