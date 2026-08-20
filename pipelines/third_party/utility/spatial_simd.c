#include "spatial_simd.h"

#include <math.h>
/* simd_kernels.h supplies sk__cquad_load/sk__cquad_store -- the only
 * sanctioned way to move 4 lanes of a {float r; float i;} struct through
 * vld2q_f32/vst2q_f32 without a strict-aliasing-violating Complex-to-float
 * pointer cast. Included before the NEON-gate macro below so this file's
 * dispatch is derived from SK_HAVE_NEON directly, instead of
 * independently re-deriving `defined(__aarch64__) && defined(__ARM_NEON)` --
 * the previous independent derivation could disagree with simd_kernels.h's
 * own gate. It also transitively includes
 * <arm_neon.h> under the identical condition SK_HAVE_NEON already gates on,
 * so no separate NEON header include is needed here. */
#include "simd_kernels.h"

#if SK_HAVE_NEON
#define THIRD_PARTY_SPATIAL_NEON 1
#else
#define THIRD_PARTY_SPATIAL_NEON 0
#endif

#define SPATIAL_PHAT_EPSILON 1e-8f

void spatial_phat_cross_scalar(const Complex* x_i,
                               const Complex* x_j,
                               Complex* out,
                               int count)
{
    int f;
    if (!x_i || !x_j || !out || count <= 0) return;
    for (f = 0; f < count; ++f) {
        float rr = x_i[f].r * x_j[f].r +
                   x_i[f].i * x_j[f].i;
        float ri = x_i[f].i * x_j[f].r -
                   x_i[f].r * x_j[f].i;
        float mag = sqrtf(rr * rr + ri * ri) + SPATIAL_PHAT_EPSILON;
        out[f].r = rr / mag;
        out[f].i = ri / mag;
    }
}

void spatial_phat_cross(const Complex* x_i,
                        const Complex* x_j,
                        Complex* out,
                        int count)
{
#if THIRD_PARTY_SPATIAL_NEON
    int f = 0;
    const float32x4_t epsilon = vdupq_n_f32(SPATIAL_PHAT_EPSILON);
    if (!x_i || !x_j || !out || count <= 0) return;
    for (; f + 4 <= count; f += 4) {
        float32x4x2_t xi = sk__cquad_load(x_i + f);
        float32x4x2_t xj = sk__cquad_load(x_j + f);
        float32x4_t rr =
            vaddq_f32(vmulq_f32(xi.val[0], xj.val[0]),
                      vmulq_f32(xi.val[1], xj.val[1]));
        float32x4_t ri =
            vsubq_f32(vmulq_f32(xi.val[1], xj.val[0]),
                      vmulq_f32(xi.val[0], xj.val[1]));
        float32x4_t mag =
            vaddq_f32(vsqrtq_f32(
                          vaddq_f32(vmulq_f32(rr, rr),
                                    vmulq_f32(ri, ri))),
                      epsilon);
        float32x4x2_t value;
        value.val[0] = vdivq_f32(rr, mag);
        value.val[1] = vdivq_f32(ri, mag);
        sk__cquad_store(out + f, value);
    }
    spatial_phat_cross_scalar(x_i + f, x_j + f, out + f, count - f);
#else
    spatial_phat_cross_scalar(x_i, x_j, out, count);
#endif
}

void spatial_conj_beamform_scalar(
    const Complex* const* weights,
    const Complex* const* inputs,
    int channels,
    int bins,
    float scale,
    Complex* out)
{
    if (!weights || !inputs || !out || channels <= 0 || bins <= 0) return;
    for (int f = 0; f < bins; ++f) {
        float real = 0.0f;
        float imag = 0.0f;
        for (int m = 0; m < channels; ++m) {
            real += weights[m][f].r * inputs[m][f].r +
                    weights[m][f].i * inputs[m][f].i;
            imag += weights[m][f].r * inputs[m][f].i -
                    weights[m][f].i * inputs[m][f].r;
        }
        out[f].r = real * scale;
        out[f].i = imag * scale;
    }
}

void spatial_conj_beamform(
    const Complex* const* weights,
    const Complex* const* inputs,
    int channels,
    int bins,
    float scale,
    Complex* out)
{
#if THIRD_PARTY_SPATIAL_NEON
    int f = 0;
    float32x4_t vscale = vdupq_n_f32(scale);
    if (!weights || !inputs || !out || channels <= 0 || bins <= 0) return;
    for (; f + 4 <= bins; f += 4) {
        float32x4_t real = vdupq_n_f32(0.0f);
        float32x4_t imag = vdupq_n_f32(0.0f);
        for (int m = 0; m < channels; ++m) {
            float32x4x2_t w =
                sk__cquad_load(weights[m] + f);
            float32x4x2_t x =
                sk__cquad_load(inputs[m] + f);
            real = vaddq_f32(
                real, vaddq_f32(vmulq_f32(w.val[0], x.val[0]),
                                vmulq_f32(w.val[1], x.val[1])));
            imag = vaddq_f32(
                imag, vsubq_f32(vmulq_f32(w.val[0], x.val[1]),
                                vmulq_f32(w.val[1], x.val[0])));
        }
        {
            float32x4x2_t value;
            value.val[0] = vmulq_f32(real, vscale);
            value.val[1] = vmulq_f32(imag, vscale);
            sk__cquad_store(out + f, value);
        }
    }
    if (f < bins) {
        /* The scalar helper starts each channel at bin zero, so handle this
         * short offset tail explicitly rather than manufacturing pointer
         * arrays on the stack. */
        for (int tail = f; tail < bins; ++tail) {
            float real = 0.0f, imag = 0.0f;
            for (int m = 0; m < channels; ++m) {
                real += weights[m][tail].r * inputs[m][tail].r +
                        weights[m][tail].i * inputs[m][tail].i;
                imag += weights[m][tail].r * inputs[m][tail].i -
                        weights[m][tail].i * inputs[m][tail].r;
            }
            out[tail].r = real * scale;
            out[tail].i = imag * scale;
        }
    }
#else
    spatial_conj_beamform_scalar(
        weights, inputs, channels, bins, scale, out);
#endif
}

static void spatial_gsc_projection_scalar_range(
    const Complex* const* steering,
    const Complex* const* inputs,
    const Complex* das,
    int channels,
    int bins,
    int first,
    Complex* u)
{
    for (int f = first; f < bins; ++f) {
        float denom = 0.0f;
        for (int m = 0; m < channels; ++m) {
            float ar = steering[m][f].r;
            float ai = steering[m][f].i;
            denom += ar * ar + ai * ai;
        }
        if (denom < 1e-12f) denom = 1e-12f;
        {
            float scale = (float)channels / denom;
            float pr = das[f].r * scale;
            float pi = das[f].i * scale;
            for (int m = 0; m < channels; ++m) {
                float ar = steering[m][f].r;
                float ai = steering[m][f].i;
                float rr = ar * pr - ai * pi;
                float ri = ar * pi + ai * pr;
                u[(size_t)m * (size_t)bins + (size_t)f].r =
                    inputs[m][f].r - rr;
                u[(size_t)m * (size_t)bins + (size_t)f].i =
                    inputs[m][f].i - ri;
            }
        }
    }
}

void spatial_gsc_projection_scalar(
    const Complex* const* steering,
    const Complex* const* inputs,
    const Complex* das,
    int channels,
    int bins,
    Complex* u)
{
    if (!steering || !inputs || !das || !u || channels <= 0 || bins <= 0)
        return;
    spatial_gsc_projection_scalar_range(
        steering, inputs, das, channels, bins, 0, u);
}

void spatial_gsc_projection(
    const Complex* const* steering,
    const Complex* const* inputs,
    const Complex* das,
    int channels,
    int bins,
    Complex* u)
{
#if THIRD_PARTY_SPATIAL_NEON
    int f = 0;
    const float32x4_t floor_value = vdupq_n_f32(1e-12f);
    if (!steering || !inputs || !das || !u || channels <= 0 || bins <= 0)
        return;
    for (; f + 4 <= bins; f += 4) {
        float32x4_t denom = vdupq_n_f32(0.0f);
        for (int m = 0; m < channels; ++m) {
            float32x4x2_t a = sk__cquad_load(steering[m] + f);
            denom = vaddq_f32(
                denom,
                vaddq_f32(vmulq_f32(a.val[0], a.val[0]),
                           vmulq_f32(a.val[1], a.val[1])));
        }
        denom = vbslq_f32(vcltq_f32(denom, floor_value),
                           floor_value, denom);
        {
            float32x4_t scale =
                vdivq_f32(vdupq_n_f32((float)channels), denom);
            float32x4x2_t d = sk__cquad_load(das + f);
            float32x4_t pr = vmulq_f32(d.val[0], scale);
            float32x4_t pi = vmulq_f32(d.val[1], scale);
            for (int m = 0; m < channels; ++m) {
                float32x4x2_t a = sk__cquad_load(steering[m] + f);
                float32x4x2_t x = sk__cquad_load(inputs[m] + f);
                float32x4x2_t value;
                float32x4_t rr = vsubq_f32(
                    vmulq_f32(a.val[0], pr), vmulq_f32(a.val[1], pi));
                float32x4_t ri = vaddq_f32(
                    vmulq_f32(a.val[0], pi), vmulq_f32(a.val[1], pr));
                value.val[0] = vsubq_f32(x.val[0], rr);
                value.val[1] = vsubq_f32(x.val[1], ri);
                sk__cquad_store(u + (size_t)m * (size_t)bins + (size_t)f,
                                value);
            }
        }
    }
    spatial_gsc_projection_scalar_range(
        steering, inputs, das, channels, bins, f, u);
#else
    spatial_gsc_projection_scalar(steering, inputs, das, channels, bins, u);
#endif
}

void spatial_complex_sub_array_scalar(
    const Complex* lhs,
    const Complex* rhs,
    Complex* out,
    int count)
{
    if (!lhs || !rhs || !out || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        out[i].r = lhs[i].r - rhs[i].r;
        out[i].i = lhs[i].i - rhs[i].i;
    }
}

void spatial_complex_sub_array(
    const Complex* lhs,
    const Complex* rhs,
    Complex* out,
    int count)
{
#if THIRD_PARTY_SPATIAL_NEON
    int i = 0;
    if (!lhs || !rhs || !out || count <= 0) return;
    for (; i + 4 <= count; i += 4) {
        float32x4x2_t a = sk__cquad_load(lhs + i);
        float32x4x2_t b = sk__cquad_load(rhs + i);
        float32x4x2_t value;
        value.val[0] = vsubq_f32(a.val[0], b.val[0]);
        value.val[1] = vsubq_f32(a.val[1], b.val[1]);
        sk__cquad_store(out + i, value);
    }
    spatial_complex_sub_array_scalar(lhs + i, rhs + i, out + i, count - i);
#else
    spatial_complex_sub_array_scalar(lhs, rhs, out, count);
#endif
}

static void spatial_gsc_effective_weights_scalar_range(
    const Complex* const* steering,
    const Complex* const* adaptive_weights,
    int channels,
    int bins,
    int first,
    Complex* out)
{
    for (int f = first; f < bins; ++f) {
        float denom = 0.0f;
        float beta_r = 0.0f;
        float beta_i = 0.0f;
        for (int m = 0; m < channels; ++m) {
            float wr = adaptive_weights[m][f].r;
            float wi = adaptive_weights[m][f].i;
            float ar = steering[m][f].r;
            float ai = steering[m][f].i;
            beta_r += wr * ar + wi * ai;
            beta_i += wr * ai - wi * ar;
            denom += ar * ar + ai * ai;
        }
        if (denom < 1e-12f) denom = 1e-12f;
        beta_r /= denom;
        beta_i /= denom;
        for (int m = 0; m < channels; ++m) {
            float wr = adaptive_weights[m][f].r;
            float wi = adaptive_weights[m][f].i;
            float ar = steering[m][f].r;
            float ai = steering[m][f].i;
            Complex* weight = out + (size_t)m * (size_t)bins + (size_t)f;
            weight->r = ar / (float)channels +
                        (beta_r * ar + beta_i * ai) - wr;
            weight->i = -ai / (float)channels +
                        (-beta_r * ai + beta_i * ar) + wi;
        }
    }
}

void spatial_gsc_effective_weights_scalar(
    const Complex* const* steering,
    const Complex* const* adaptive_weights,
    int channels,
    int bins,
    Complex* out)
{
    if (!steering || !adaptive_weights || !out || channels <= 0 || bins <= 0)
        return;
    spatial_gsc_effective_weights_scalar_range(
        steering, adaptive_weights, channels, bins, 0, out);
}

void spatial_gsc_effective_weights(
    const Complex* const* steering,
    const Complex* const* adaptive_weights,
    int channels,
    int bins,
    Complex* out)
{
#if THIRD_PARTY_SPATIAL_NEON
    int f = 0;
    const float32x4_t floor_value = vdupq_n_f32(1e-12f);
    const float32x4_t channels_v = vdupq_n_f32((float)channels);
    if (!steering || !adaptive_weights || !out || channels <= 0 || bins <= 0)
        return;
    for (; f + 4 <= bins; f += 4) {
        float32x4_t denom = vdupq_n_f32(0.0f);
        float32x4_t beta_r = vdupq_n_f32(0.0f);
        float32x4_t beta_i = vdupq_n_f32(0.0f);
        for (int m = 0; m < channels; ++m) {
            float32x4x2_t w = sk__cquad_load(adaptive_weights[m] + f);
            float32x4x2_t a = sk__cquad_load(steering[m] + f);
            beta_r = vaddq_f32(
                beta_r,
                vaddq_f32(vmulq_f32(w.val[0], a.val[0]),
                           vmulq_f32(w.val[1], a.val[1])));
            beta_i = vaddq_f32(
                beta_i,
                vsubq_f32(vmulq_f32(w.val[0], a.val[1]),
                           vmulq_f32(w.val[1], a.val[0])));
            denom = vaddq_f32(
                denom,
                vaddq_f32(vmulq_f32(a.val[0], a.val[0]),
                           vmulq_f32(a.val[1], a.val[1])));
        }
        denom = vbslq_f32(vcltq_f32(denom, floor_value),
                           floor_value, denom);
        beta_r = vdivq_f32(beta_r, denom);
        beta_i = vdivq_f32(beta_i, denom);
        for (int m = 0; m < channels; ++m) {
            float32x4x2_t w = sk__cquad_load(adaptive_weights[m] + f);
            float32x4x2_t a = sk__cquad_load(steering[m] + f);
            float32x4x2_t value;
            value.val[0] = vsubq_f32(
                vaddq_f32(
                    vdivq_f32(a.val[0], channels_v),
                    vaddq_f32(vmulq_f32(beta_r, a.val[0]),
                               vmulq_f32(beta_i, a.val[1]))),
                w.val[0]);
            value.val[1] = vaddq_f32(
                vaddq_f32(
                    vdivq_f32(vnegq_f32(a.val[1]), channels_v),
                    vaddq_f32(vmulq_f32(vnegq_f32(beta_r), a.val[1]),
                               vmulq_f32(beta_i, a.val[0]))),
                w.val[1]);
            sk__cquad_store(
                out + (size_t)m * (size_t)bins + (size_t)f, value);
        }
    }
    spatial_gsc_effective_weights_scalar_range(
        steering, adaptive_weights, channels, bins, f, out);
#else
    spatial_gsc_effective_weights_scalar(
        steering, adaptive_weights, channels, bins, out);
#endif
}

void spatial_pair_score_accumulate_scalar(
    const Complex* phat,
    const Complex* steering,
    float* score,
    int count)
{
    if (!phat || !steering || !score || count <= 0) return;
    for (int f = 0; f < count; ++f) {
        score[f] += 2.0f *
            (phat[f].r * steering[f].r - phat[f].i * steering[f].i);
    }
}

void spatial_pair_score_accumulate(
    const Complex* phat,
    const Complex* steering,
    float* score,
    int count)
{
#if THIRD_PARTY_SPATIAL_NEON
    int f = 0;
    float32x4_t two = vdupq_n_f32(2.0f);
    if (!phat || !steering || !score || count <= 0) return;
    for (; f + 4 <= count; f += 4) {
        float32x4x2_t p =
            sk__cquad_load(phat + f);
        float32x4x2_t s =
            sk__cquad_load(steering + f);
        float32x4_t real =
            vsubq_f32(vmulq_f32(p.val[0], s.val[0]),
                      vmulq_f32(p.val[1], s.val[1]));
        vst1q_f32(score + f,
                  vaddq_f32(vld1q_f32(score + f),
                            vmulq_f32(two, real)));
    }
    spatial_pair_score_accumulate_scalar(
        phat + f, steering + f, score + f, count - f);
#else
    spatial_pair_score_accumulate_scalar(phat, steering, score, count);
#endif
}

const char* spatial_simd_backend(void)
{
#if THIRD_PARTY_SPATIAL_NEON
    return "aarch64-neon";
#else
    return "scalar";
#endif
}
