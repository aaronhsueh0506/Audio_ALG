#include "spatial_simd.h"

#include <math.h>
#include <stddef.h>

/* simd_kernels.h supplies sk__cquad_load/sk__cquad_store -- the only
 * sanctioned way to move 4 lanes of a {float r; float i;} struct through
 * vld2q_f32/vst2q_f32 (a raw kiss_fft_cpx*->float* cast is a strict-aliasing
 * violation, C11 6.5p7, and this exact cast previously miscompiled on the
 * target embedded toolchain). Its `Complex` type is storage-compatible with
 * `kiss_fft_cpx`, asserted below. Included before the NEON-gate macro below
 * so this file's dispatch is derived from SK_HAVE_NEON directly, instead of
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

SK_STATIC_ASSERT(sizeof(Complex) == sizeof(kiss_fft_cpx),
                  "Complex and kiss_fft_cpx must have matching storage");
SK_STATIC_ASSERT(offsetof(Complex, r) == offsetof(kiss_fft_cpx, r),
                  "real component layout mismatch");
SK_STATIC_ASSERT(offsetof(Complex, i) == offsetof(kiss_fft_cpx, i),
                  "imaginary component layout mismatch");

#define SPATIAL_PHAT_EPSILON 1e-8f

void spatial_phat_cross_scalar(const kiss_fft_cpx* x_i,
                               const kiss_fft_cpx* x_j,
                               kiss_fft_cpx* out,
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

void spatial_phat_cross(const kiss_fft_cpx* x_i,
                        const kiss_fft_cpx* x_j,
                        kiss_fft_cpx* out,
                        int count)
{
#if THIRD_PARTY_SPATIAL_NEON
    int f = 0;
    const float32x4_t epsilon = vdupq_n_f32(SPATIAL_PHAT_EPSILON);
    if (!x_i || !x_j || !out || count <= 0) return;
    for (; f + 4 <= count; f += 4) {
        float32x4x2_t xi = sk__cquad_load((const Complex*)(x_i + f));
        float32x4x2_t xj = sk__cquad_load((const Complex*)(x_j + f));
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
        sk__cquad_store((Complex*)(out + f), value);
    }
    spatial_phat_cross_scalar(x_i + f, x_j + f, out + f, count - f);
#else
    spatial_phat_cross_scalar(x_i, x_j, out, count);
#endif
}

void spatial_conj_beamform_scalar(
    const kiss_fft_cpx* const* weights,
    const kiss_fft_cpx* const* inputs,
    int channels,
    int bins,
    float scale,
    kiss_fft_cpx* out)
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
    const kiss_fft_cpx* const* weights,
    const kiss_fft_cpx* const* inputs,
    int channels,
    int bins,
    float scale,
    kiss_fft_cpx* out)
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
                sk__cquad_load((const Complex*)(weights[m] + f));
            float32x4x2_t x =
                sk__cquad_load((const Complex*)(inputs[m] + f));
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
            sk__cquad_store((Complex*)(out + f), value);
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

void spatial_pair_score_accumulate_scalar(
    const kiss_fft_cpx* phat,
    const kiss_fft_cpx* steering,
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
    const kiss_fft_cpx* phat,
    const kiss_fft_cpx* steering,
    float* score,
    int count)
{
#if THIRD_PARTY_SPATIAL_NEON
    int f = 0;
    float32x4_t two = vdupq_n_f32(2.0f);
    if (!phat || !steering || !score || count <= 0) return;
    for (; f + 4 <= count; f += 4) {
        float32x4x2_t p =
            sk__cquad_load((const Complex*)(phat + f));
        float32x4x2_t s =
            sk__cquad_load((const Complex*)(steering + f));
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
