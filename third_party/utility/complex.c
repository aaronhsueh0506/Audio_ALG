#include "complex.h"
#include <math.h>

kiss_fft_cpx c_zero(void)
{
    kiss_fft_cpx y;
    y.r = 0.0f;
    y.i = 0.0f;
    return y;
}

kiss_fft_cpx c_from_real(float x)
{
    kiss_fft_cpx y;
    y.r = x;
    y.i = 0.0f;
    return y;
}

kiss_fft_cpx c_add(kiss_fft_cpx a, kiss_fft_cpx b)
{
    kiss_fft_cpx y;
    y.r = a.r + b.r;
    y.i = a.i + b.i;
    return y;
}

kiss_fft_cpx c_sub(kiss_fft_cpx a, kiss_fft_cpx b)
{
    kiss_fft_cpx y;
    y.r = a.r - b.r;
    y.i = a.i - b.i;
    return y;
}

kiss_fft_cpx c_mul(kiss_fft_cpx a, kiss_fft_cpx b)
{
    kiss_fft_cpx y;
    y.r = a.r * b.r - a.i * b.i;
    y.i = a.r * b.i + a.i * b.r;
    return y;
}

kiss_fft_cpx c_div(kiss_fft_cpx a, kiss_fft_cpx b)
{
    kiss_fft_cpx y;
    float den = b.r * b.r + b.i * b.i;

    if (den < 1e-12f) {
        y.r = 0.0f;
        y.i = 0.0f;
        return y;
    }

    y.r = (a.r * b.r + a.i * b.i) / den;
    y.i = (a.i * b.r - a.r * b.i) / den;
    return y;
}

kiss_fft_cpx c_conj(kiss_fft_cpx a)
{
    kiss_fft_cpx y;
    y.r = a.r;
    y.i = -a.i;
    return y;
}

kiss_fft_cpx c_scale(kiss_fft_cpx a, float x)
{
    kiss_fft_cpx y;
    y.r = a.r * x;
    y.i = a.i * x;
    return y;
}

kiss_fft_cpx c_div_real(kiss_fft_cpx a, float x)
{
    kiss_fft_cpx y;
    if (fabsf(x) < 1e-12f) {
        y.r = 0.0f;
        y.i = 0.0f;
        return y;
    }
    y.r = a.r / x;
    y.i = a.i / x;
    return y;
}

float c_abs2(kiss_fft_cpx a)
{
    return a.r * a.r + a.i * a.i;
}

float c_abs(kiss_fft_cpx a)
{
    return sqrtf(c_abs2(a));
}
