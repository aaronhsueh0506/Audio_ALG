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

#include <float.h>
#include <math.h>

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

/* Rebuild the residual-echo PSD as a complex vector carrying the echo
 * estimate's PHASE and the residual's MAGNITUDE:
 *
 *     |out[i]|^2 == max(r2[i], 0)
 *
 * That identity is the kernel's whole contract -- echo supplies direction
 * only, and its magnitude divides out. The Python reference spells it as
 * sqrt(r2) * exp(1j * angle(echo)); the ratio form here is the same thing
 * without atan2/sincos EXCEPT on the phase floor, which only this side has:
 * for 0 < |echo|^2 <= phase_floor2 (the mic-path HPF routinely puts the DC
 * bin there) C pins the result to the +real axis while Python follows
 * np.angle. That divergence predates the range handling below.
 *
 * There is deliberately no fixed upper bound on `scale`. r2 arrives
 * int16^2-scaled (aec.h) while echo_spec is audio-scaled, so
 * sqrt(r2/|echo|^2) carries a factor of 32768 unrelated to residual gain. A
 * bound would make |out| smaller than sqrt(r2) and understate residual echo.
 * The phase floor handles estimates without useful phase; the overflow
 * fallback below preserves the same magnitude contract for every finite
 * float32 input.
 *
 * Neither overflow mode announces itself. power/|echo|^2 saturating yields
 * inf and poisons every downstream stage; |echo|^2 saturating yields
 * sqrt(power/inf) == 0, reporting NO residual echo on that bin and quietly
 * stopping the suppressor -- the same class of failure as bounding `scale`,
 * just further out. The exceptional path below normalizes the phase vector
 * before applying the magnitude, which is the reference formulation and
 * cannot form the overflowing intermediate at all. */
static inline Complex four_aec_residual_sample(
    float er, float ei, float raw_r2) {
    const float phase_floor2 = 1.0e-12f; /* (1e-6)^2 */
    Complex out;
    float mag2 = er * er + ei * ei;
    float power = fmaxf(raw_r2, 0.0f);
    float ratio, axis, unit_r, unit_i, scale;

    if (!(mag2 > phase_floor2)) {
        out.r = sqrtf(power);
        out.i = 0.0f;
        return out;
    }

    /* mag2 > phase_floor2 > 0, so the ratio is never 0/0 and can be formed
     * unconditionally; only its range is in question. */
    ratio = power / mag2;
    if (isfinite(mag2) && isfinite(ratio)) {
        scale = sqrtf(ratio);
        out.r = er * scale;
        out.i = ei * scale;
        return out;
    }

    /* Exceptional path: scale the phase vector down before combining, so no
     * intermediate ever leaves float32 range. Dividing by the larger
     * component puts the sum of squares in [1, 2], which is why a plain
     * sqrt suffices here where the direct form overflowed. */
    axis = fmaxf(fabsf(er), fabsf(ei));
    unit_r = er / axis;
    unit_i = ei / axis;
    scale = sqrtf(power) / sqrtf(unit_r * unit_r + unit_i * unit_i);
    out.r = unit_r * scale;
    out.i = unit_i * scale;
    return out;
}

static inline void four_aec_residual_vector_scalar(
    Complex* out, const Complex* echo, const float* r2, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        out[i] = four_aec_residual_sample(
            echo[i].r, echo[i].i, r2[i]);
    }
}

#if SK_HAVE_NEON
static inline void four_aec_residual_vector(
    Complex* out, const Complex* echo, const float* r2, int n) {
    /* Must match four_aec_residual_sample's phase_floor2 exactly -- see that
     * function's comment for why -- so SIMD=1 and SIMD=0 stay byte-identical
     * for finite inputs, per this header's own documented invariant. Neither
     * form bounds `scale`; that comment says why. */
    const float phase_floor2 = 1.0e-12f;
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

        /* Both overflow modes are visible in values this block already holds,
         * so detecting them costs no arithmetic: |echo|^2 saturating shows up
         * in mag2, and power/|echo|^2 saturating shows up as an infinite
         * scale. Both terms are needed -- the first mode degrades scale to a
         * perfectly finite zero. Recomputing the test in scalar instead would
         * make the repair decision depend on the vector and scalar forms
         * agreeing bit-for-bit, and would have to re-read echo after the
         * store just above, which is what would forbid an in-place call.
         *
         * The reduction is false on every hop of real audio, leaving the
         * vectorized path exactly as it was. When it does trip, all four
         * lanes go back through the scalar form: for a lane that did not
         * overflow it reproduces the stored bytes, which is the same
         * SIMD/scalar identity the tests already assert. */
        {
            const float32x4_t huge = vdupq_n_f32(FLT_MAX);
            uint32x4_t overflowed = vandq_u32(
                has_phase,
                vorrq_u32(vcgtq_f32(mag2, huge), vcgtq_f32(scale, huge)));
            if (vmaxvq_u32(overflowed)) {
                float lane_r[4], lane_i[4], lane_power[4];
                int lane;
                vst1q_f32(lane_r, e.val[0]);
                vst1q_f32(lane_i, e.val[1]);
                vst1q_f32(lane_power, raw_power);
                for (lane = 0; lane < 4; ++lane) {
                    out[i + lane] = four_aec_residual_sample(
                        lane_r[lane], lane_i[lane], lane_power[lane]);
                }
            }
        }
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
