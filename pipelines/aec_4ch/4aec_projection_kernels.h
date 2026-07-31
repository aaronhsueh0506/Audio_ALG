/**
 * 4aec_projection_kernels.h — scalar/NEON kernels used by post-beam context
 * projection.
 *
 * Keep these kernels local to the four-channel integration layer: unlike the
 * AEC filter's fmaf-based complex MAC, projection historically uses separate
 * multiply/subtract/add operations under -ffp-contract=off.  The NEON path
 * deliberately preserves that operation sequence so SIMD=1 and SIMD=0 stay
 * byte-identical for finite inputs.
 */
#ifndef FOUR_AEC_PROJECTION_KERNELS_H
#define FOUR_AEC_PROJECTION_KERNELS_H

#include "simd_kernels.h"

static inline void four_aec_projection_cmac_scalar(
    Complex* acc, const Complex* weights, const Complex* input, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float wr = weights[i].r;
        float wi = weights[i].i;
        float xr = input[i].r;
        float xi = input[i].i;
        float real = wr * xr - wi * xi;
        float imag = wr * xi + wi * xr;
        acc[i].r += real;
        acc[i].i += imag;
    }
}

#if SK_HAVE_NEON
static inline void four_aec_projection_cmac(
    Complex* acc, const Complex* weights, const Complex* input, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t w = sk__cquad_load(weights + i);
        float32x4x2_t x = sk__cquad_load(input + i);
        float32x4x2_t a = sk__cquad_load(acc + i);
        float32x4x2_t out;
        float32x4_t real = vsubq_f32(
            vmulq_f32(w.val[0], x.val[0]),
            vmulq_f32(w.val[1], x.val[1]));
        float32x4_t imag = vaddq_f32(
            vmulq_f32(w.val[0], x.val[1]),
            vmulq_f32(w.val[1], x.val[0]));
        out.val[0] = vaddq_f32(a.val[0], real);
        out.val[1] = vaddq_f32(a.val[1], imag);
        sk__cquad_store(acc + i, out);
    }
    four_aec_projection_cmac_scalar(
        acc + i, weights + i, input + i, n - i);
}
#else
static inline void four_aec_projection_cmac(
    Complex* acc, const Complex* weights, const Complex* input, int n) {
    four_aec_projection_cmac_scalar(acc, weights, input, n);
}
#endif

static inline void four_aec_complex_mag2_scalar(
    float* out, const Complex* input, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float real = input[i].r;
        float imag = input[i].i;
        out[i] = real * real + imag * imag;
    }
}

#if SK_HAVE_NEON
static inline void four_aec_complex_mag2(
    float* out, const Complex* input, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t value = sk__cquad_load(input + i);
        float32x4_t mag2 = vaddq_f32(
            vmulq_f32(value.val[0], value.val[0]),
            vmulq_f32(value.val[1], value.val[1]));
        vst1q_f32(out + i, mag2);
    }
    four_aec_complex_mag2_scalar(out + i, input + i, n - i);
}
#else
static inline void four_aec_complex_mag2(
    float* out, const Complex* input, int n) {
    four_aec_complex_mag2_scalar(out, input, n);
}
#endif

/* Convert each lane's scalar residual power into a phase-bearing complex
 * residual aligned to that lane's estimated echo:
 *
 *   residual = echo * sqrt(max(r2, 0) / |echo|^2)
 *
 * A near-zero echo has no usable phase, so it falls back to +real sqrt(r2),
 * matching the previous implementation.  The ratio form needs one sqrt and
 * one divide instead of two sqrt and two divides per channel/bin. */
static inline void four_aec_residual_vector_scalar(
    Complex* out, const Complex* echo, const float* r2, int n) {
    const float phase_floor2 = 1.0e-40f; /* (1e-20)^2 */
    int i;
    for (i = 0; i < n; ++i) {
        float er = echo[i].r;
        float ei = echo[i].i;
        float mag2 = er * er + ei * ei;
        float power = fmaxf(r2[i], 0.0f);
        int has_phase = mag2 > phase_floor2;
        float denominator = has_phase ? mag2 : 1.0f;
        float scale = sqrtf(power / denominator);
        out[i].r = has_phase ? er * scale : scale;
        out[i].i = has_phase ? ei * scale : 0.0f;
    }
}

#if SK_HAVE_NEON
static inline void four_aec_residual_vector(
    Complex* out, const Complex* echo, const float* r2, int n) {
    const float phase_floor2 = 1.0e-40f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4x2_t e = sk__cquad_load(echo + i);
        float32x4_t mag2 = vaddq_f32(
            vmulq_f32(e.val[0], e.val[0]),
            vmulq_f32(e.val[1], e.val[1]));
        float32x4_t raw_power = vld1q_f32(r2 + i);
        uint32x4_t power_positive = vcgtq_f32(raw_power, zero);
        float32x4_t power = vbslq_f32(power_positive, raw_power, zero);
        uint32x4_t has_phase =
            vcgtq_f32(mag2, vdupq_n_f32(phase_floor2));
        float32x4_t denominator = vbslq_f32(has_phase, mag2, one);
        float32x4_t scale = vsqrtq_f32(vdivq_f32(power, denominator));
        float32x4x2_t residual;
        residual.val[0] = vbslq_f32(
            has_phase, vmulq_f32(e.val[0], scale), scale);
        residual.val[1] = vbslq_f32(
            has_phase, vmulq_f32(e.val[1], scale), zero);
        sk__cquad_store(out + i, residual);
    }
    four_aec_residual_vector_scalar(
        out + i, echo + i, r2 + i, n - i);
}
#else
static inline void four_aec_residual_vector(
    Complex* out, const Complex* echo, const float* r2, int n) {
    four_aec_residual_vector_scalar(out, echo, r2, n);
}
#endif

static inline void four_aec_comfort_accumulate_scalar(
    float* acc, const Complex* weights, const float* comfort, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float wr = weights[i].r;
        float wi = weights[i].i;
        float weight2 = wr * wr + wi * wi;
        acc[i] += weight2 * fmaxf(comfort[i], 0.0f);
    }
}

#if SK_HAVE_NEON
static inline void four_aec_comfort_accumulate(
    float* acc, const Complex* weights, const float* comfort, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        float32x4x2_t w = sk__cquad_load(weights + i);
        float32x4_t weight2 = vaddq_f32(
            vmulq_f32(w.val[0], w.val[0]),
            vmulq_f32(w.val[1], w.val[1]));
        float32x4_t raw_comfort = vld1q_f32(comfort + i);
        uint32x4_t comfort_positive = vcgtq_f32(raw_comfort, zero);
        float32x4_t nonnegative =
            vbslq_f32(comfort_positive, raw_comfort, zero);
        float32x4_t value = vaddq_f32(
            vld1q_f32(acc + i), vmulq_f32(weight2, nonnegative));
        vst1q_f32(acc + i, value);
    }
    four_aec_comfort_accumulate_scalar(
        acc + i, weights + i, comfort + i, n - i);
}
#else
static inline void four_aec_comfort_accumulate(
    float* acc, const Complex* weights, const float* comfort, int n) {
    four_aec_comfort_accumulate_scalar(acc, weights, comfort, n);
}
#endif

#endif /* FOUR_AEC_PROJECTION_KERNELS_H */
