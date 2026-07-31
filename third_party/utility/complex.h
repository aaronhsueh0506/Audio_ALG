#ifndef COMPLEX_UTILS_H
#define COMPLEX_UTILS_H

#include <math.h>
#include "kiss_fft.h"

kiss_fft_cpx c_zero(void);
kiss_fft_cpx c_from_real(float x);

kiss_fft_cpx c_add(kiss_fft_cpx a, kiss_fft_cpx b);
kiss_fft_cpx c_sub(kiss_fft_cpx a, kiss_fft_cpx b);
kiss_fft_cpx c_mul(kiss_fft_cpx a, kiss_fft_cpx b);
kiss_fft_cpx c_div(kiss_fft_cpx a, kiss_fft_cpx b);
kiss_fft_cpx c_conj(kiss_fft_cpx a);

kiss_fft_cpx c_scale(kiss_fft_cpx a, float x);
kiss_fft_cpx c_div_real(kiss_fft_cpx a, float x);

float c_abs2(kiss_fft_cpx a);
float c_abs(kiss_fft_cpx a);

#endif
