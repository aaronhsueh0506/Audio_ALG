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
