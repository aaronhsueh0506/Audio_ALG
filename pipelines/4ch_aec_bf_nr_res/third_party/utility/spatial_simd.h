#ifndef THIRD_PARTY_UTILITY_SPATIAL_SIMD_H
#define THIRD_PARTY_UTILITY_SPATIAL_SIMD_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Scalar golden and compile-time-dispatched PHAT cross-spectrum kernels. */
void spatial_phat_cross_scalar(const Complex* x_i,
                               const Complex* x_j,
                               Complex* out,
                               int count);

void spatial_phat_cross(const Complex* x_i,
                        const Complex* x_j,
                        Complex* out,
                        int count);

/* out[f] = scale * sum_m(conj(weights[m][f]) * inputs[m][f]). */
void spatial_conj_beamform_scalar(
    const Complex* const* weights,
    const Complex* const* inputs,
    int channels,
    int bins,
    float scale,
    Complex* out);

void spatial_conj_beamform(
    const Complex* const* weights,
    const Complex* const* inputs,
    int channels,
    int bins,
    float scale,
    Complex* out);

/* Projection-form GSC blocking output. `u` is channel-major [channels][bins]. */
void spatial_gsc_projection_scalar(
    const Complex* const* steering,
    const Complex* const* inputs,
    const Complex* das,
    int channels,
    int bins,
    Complex* u);

void spatial_gsc_projection(
    const Complex* const* steering,
    const Complex* const* inputs,
    const Complex* das,
    int channels,
    int bins,
    Complex* u);

void spatial_complex_sub_array_scalar(
    const Complex* lhs,
    const Complex* rhs,
    Complex* out,
    int count);

void spatial_complex_sub_array(
    const Complex* lhs,
    const Complex* rhs,
    Complex* out,
    int count);

/* Effective channel-major [channels][bins] response represented by the
 * current GSC steering vectors and adaptive weights. */
void spatial_gsc_effective_weights_scalar(
    const Complex* const* steering,
    const Complex* const* adaptive_weights,
    int channels,
    int bins,
    Complex* out);

void spatial_gsc_effective_weights(
    const Complex* const* steering,
    const Complex* const* adaptive_weights,
    int channels,
    int bins,
    Complex* out);

/* score[f] += 2*real(phat[f] * steering[f]). */
void spatial_pair_score_accumulate_scalar(
    const Complex* phat,
    const Complex* steering,
    float* score,
    int count);

void spatial_pair_score_accumulate(
    const Complex* phat,
    const Complex* steering,
    float* score,
    int count);

const char* spatial_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* THIRD_PARTY_UTILITY_SPATIAL_SIMD_H */
