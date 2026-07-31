#ifndef THIRD_PARTY_UTILITY_SPATIAL_SIMD_H
#define THIRD_PARTY_UTILITY_SPATIAL_SIMD_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Scalar golden and compile-time-dispatched PHAT cross-spectrum kernels. */
void spatial_phat_cross_scalar(const kiss_fft_cpx* x_i,
                               const kiss_fft_cpx* x_j,
                               kiss_fft_cpx* out,
                               int count);

void spatial_phat_cross(const kiss_fft_cpx* x_i,
                        const kiss_fft_cpx* x_j,
                        kiss_fft_cpx* out,
                        int count);

/* out[f] = scale * sum_m(conj(weights[m][f]) * inputs[m][f]). */
void spatial_conj_beamform_scalar(
    const kiss_fft_cpx* const* weights,
    const kiss_fft_cpx* const* inputs,
    int channels,
    int bins,
    float scale,
    kiss_fft_cpx* out);

void spatial_conj_beamform(
    const kiss_fft_cpx* const* weights,
    const kiss_fft_cpx* const* inputs,
    int channels,
    int bins,
    float scale,
    kiss_fft_cpx* out);

/* score[f] += 2*real(phat[f] * steering[f]). */
void spatial_pair_score_accumulate_scalar(
    const kiss_fft_cpx* phat,
    const kiss_fft_cpx* steering,
    float* score,
    int count);

void spatial_pair_score_accumulate(
    const kiss_fft_cpx* phat,
    const kiss_fft_cpx* steering,
    float* score,
    int count);

const char* spatial_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* THIRD_PARTY_UTILITY_SPATIAL_SIMD_H */
