#ifndef COMPLEX_UTILS_H
#define COMPLEX_UTILS_H

#include <math.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

Complex spatial_complex_zero(void);
Complex spatial_complex_from_real(float x);

Complex spatial_complex_add(Complex a, Complex b);
Complex spatial_complex_sub(Complex a, Complex b);
Complex spatial_complex_mul(Complex a, Complex b);
Complex spatial_complex_div(Complex a, Complex b);
Complex spatial_complex_conj(Complex a);

Complex spatial_complex_scale(Complex a, float x);
Complex spatial_complex_div_real(Complex a, float x);

float spatial_complex_abs2(Complex a);
float spatial_complex_abs(Complex a);

#ifdef __cplusplus
}
#endif

#endif
