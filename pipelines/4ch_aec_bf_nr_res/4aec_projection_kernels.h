/** Scalar/NEON kernels used by post-beam context projection.
 * four_aec_projection_cmac/_scalar are aliases of utility/simd_kernels.h's
 * sk_cmac_f32/sk_cmac_f32_scalar: the same unfused multiply/subtract/add sequence under
 * -ffp-contract=off (unlike the AEC filter's fmaf-based sk_cmac_fma_f32_scalar). */
#ifndef FOUR_AEC_PROJECTION_KERNELS_H
#define FOUR_AEC_PROJECTION_KERNELS_H

#include "utility/simd_kernels.h"

#include <float.h>
#include <math.h>

static inline void four_aec_projection_cmac_scalar(
    Complex* acc, const Complex* weights, const Complex* input, int n) {
    sk_cmac_f32_scalar(acc, weights, input, n);
}

static inline void four_aec_projection_cmac(
    Complex* acc, const Complex* weights, const Complex* input, int n) {
    sk_cmac_f32(acc, weights, input, n);
}

/* out[i] = input[i].r^2 + input[i].i^2 with no epsilon (the sum of two squares is
 * never -0.0, so adding eps = 0 leaves sk_cmag_f32's result unchanged). */
static inline void four_aec_complex_mag2_scalar(
    float* out, const Complex* input, int n) {
    sk_cmag_f32_scalar(out, input, n, 0.0f);
}

static inline void four_aec_complex_mag2(
    float* out, const Complex* input, int n) {
    sk_cmag_f32(out, input, n, 0.0f);
}

/* Rebuilds the residual-echo PSD as a complex vector with the echo estimate's phase and the residual's magnitude: |out[i]|^2 == max(r2[i], 0). Python reference: 4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts (sqrt(r2) * exp(1j * echo_phase)); the ratio form avoids atan2/sincos, and for 0 < |echo|^2 <= phase_floor2 (e.g. the DC bin after the mic HPF) the result is pinned to the +real axis.
 * `scale` has no upper bound: r2 is int16^2-scaled while echo_spec is audio-scaled, so sqrt(r2/|echo|^2) legitimately carries a factor of 32768. Overflow modes: power/|echo|^2 saturating gives inf, and |echo|^2 saturating gives 0 (no residual echo reported). The exceptional path normalises the phase vector first, so no intermediate leaves float32 range. */
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

    /* mag2 > phase_floor2 > 0, so the ratio is never 0/0; only its range is in question. */
    ratio = power / mag2;
    if (isfinite(mag2) && isfinite(ratio)) {
        scale = sqrtf(ratio);
        out.r = er * scale;
        out.i = ei * scale;
        return out;
    }

    /* Exceptional path: divide by the larger component first (sum of squares in [1, 2]),
     * so a plain sqrt suffices. */
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
    /* Must match four_aec_residual_sample's phase_floor2 so the SIMD and scalar forms
     * agree for finite inputs. */
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

        /* Both overflow modes are visible in values already held (mag2 saturating, scale
         * infinite or degraded to zero), so detection costs no arithmetic and keeps an
         * in-place call legal. The test is false on real audio; when it trips, all four lanes
         * go back through the scalar form, which reproduces the stored bytes for lanes that did
         * not overflow. */
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

/* Weight-matrix gate: 1 when every component is finite and the sequential sum of
 * |re| + |im| is finite and above 1e-12, else 0. The vector screen decides the common
 * case without the sum (terms <= 2^100 and fewer than 2^24 of them cannot overflow,
 * and one term above 1e-12 decides); any other input runs the sequential loop. */
static inline int four_aec_weights_valid_scalar(
    const Complex* weights, size_t count) {
    size_t i;
    float sum = 0.0f;
    for (i = 0; i < count; ++i) {
        if (!isfinite(weights[i].r) || !isfinite(weights[i].i)) return 0;
        sum += fabsf(weights[i].r) + fabsf(weights[i].i);
    }
    return isfinite(sum) && sum > 1e-12f;
}

#if SK_HAVE_NEON
static inline int four_aec_weights_valid(
    const Complex* weights, size_t count) {
    const float32x4_t bound = vdupq_n_f32(0x1p100f);
    const float32x4_t floor_value = vdupq_n_f32(1e-12f);
    uint32x4_t bounded = vdupq_n_u32(~0u);
    uint32x4_t above = vdupq_n_u32(0u);
    size_t i = 0;
    if (count >= ((size_t)1 << 24))
        return four_aec_weights_valid_scalar(weights, count);
    for (; i + 4 <= count; i += 4) {
        float32x4x2_t w = sk__cquad_load(weights + i);
        float32x4_t term =
            vaddq_f32(vabsq_f32(w.val[0]), vabsq_f32(w.val[1]));
        bounded = vandq_u32(bounded, vcleq_f32(term, bound));
        above = vorrq_u32(above, vcgtq_f32(term, floor_value));
    }
    if (vminvq_u32(bounded) == ~0u) {
        int tail_above = 0;
        int tail_bounded = 1;
        for (; i < count; ++i) {
            float term = fabsf(weights[i].r) + fabsf(weights[i].i);
            tail_bounded &= term <= 0x1p100f;
            tail_above |= term > 1e-12f;
        }
        if (tail_bounded && (tail_above || vmaxvq_u32(above) != 0u))
            return 1;
    }
    return four_aec_weights_valid_scalar(weights, count);
}
#else
static inline int four_aec_weights_valid(
    const Complex* weights, size_t count) {
    return four_aec_weights_valid_scalar(weights, count);
}
#endif

#endif /* FOUR_AEC_PROJECTION_KERNELS_H */
