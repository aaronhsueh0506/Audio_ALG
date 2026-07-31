#include "complex.h"
#include <math.h>

Complex spatial_complex_zero(void)
{
    Complex y;
    y.r = 0.0f;
    y.i = 0.0f;
    return y;
}

Complex spatial_complex_from_real(float x)
{
    Complex y;
    y.r = x;
    y.i = 0.0f;
    return y;
}

Complex spatial_complex_add(Complex a, Complex b)
{
    Complex y;
    y.r = a.r + b.r;
    y.i = a.i + b.i;
    return y;
}

Complex spatial_complex_sub(Complex a, Complex b)
{
    Complex y;
    y.r = a.r - b.r;
    y.i = a.i - b.i;
    return y;
}

Complex spatial_complex_mul(Complex a, Complex b)
{
    Complex y;
    y.r = a.r * b.r - a.i * b.i;
    y.i = a.r * b.i + a.i * b.r;
    return y;
}

Complex spatial_complex_div(Complex a, Complex b)
{
    Complex y;
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

Complex spatial_complex_conj(Complex a)
{
    Complex y;
    y.r = a.r;
    y.i = -a.i;
    return y;
}

Complex spatial_complex_scale(Complex a, float x)
{
    Complex y;
    y.r = a.r * x;
    y.i = a.i * x;
    return y;
}

Complex spatial_complex_div_real(Complex a, float x)
{
    Complex y;
    if (fabsf(x) < 1e-12f) {
        y.r = 0.0f;
        y.i = 0.0f;
        return y;
    }
    y.r = a.r / x;
    y.i = a.i / x;
    return y;
}

float spatial_complex_abs2(Complex a)
{
    return a.r * a.r + a.i * a.i;
}

float spatial_complex_abs(Complex a)
{
    return sqrtf(spatial_complex_abs2(a));
}
