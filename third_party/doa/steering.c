#include <stdlib.h>
#include <math.h>
#include "steering.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ArrayGeometry* array_geometry_create(int M)
{
    ArrayGeometry* g;
    if (M <= 0) return NULL;
    g = (ArrayGeometry*)calloc(1, sizeof(ArrayGeometry));
    if (!g) return NULL;
    g->M = M;
    g->x = (float*)calloc(M, sizeof(float));
    g->y = (float*)calloc(M, sizeof(float));
    if (!g->x || !g->y) {
        array_geometry_destroy(g);
        return NULL;
    }
    return g;
}

ArrayGeometry* array_geometry_create_uca(int M, float radius)
{
    ArrayGeometry* g = array_geometry_create(M);
    if (!g || !isfinite(radius) || radius <= 0.0f) {
        array_geometry_destroy(g);
        return NULL;
    }

    for (int m = 0; m < M; m++) {
        float phi = 2.0f * M_PI * m / M;
        g->x[m] = radius * cosf(phi);
        g->y[m] = radius * sinf(phi);
    }

    return g;
}

ArrayGeometry* array_geometry_create_ula(int M, float spacing)
{
    ArrayGeometry* g = array_geometry_create(M);
    if (!g || !isfinite(spacing) || spacing <= 0.0f) {
        array_geometry_destroy(g);
        return NULL;
    }

    float center = 0.5f * (M - 1);
    for (int m = 0; m < M; m++) {
        g->x[m] = (m - center) * spacing;
        g->y[m] = 0.0f;
    }

    return g;
}

ArrayGeometry* array_geometry_create_custom(int M, const float* x, const float* y)
{
    ArrayGeometry* g = array_geometry_create(M);
    if (!g || !x || !y) {
        array_geometry_destroy(g);
        return NULL;
    }

    for (int m = 0; m < M; m++) {
        g->x[m] = x[m];
        g->y[m] = y[m];
    }

    return g;
}

void array_geometry_destroy(ArrayGeometry* g)
{
    if (!g) return;
    free(g->x);
    free(g->y);
    free(g);
}

float* srp_create_uniform_angles(int num_angles)
{
    float* angles;
    if (num_angles <= 0) return NULL;
    angles = (float*)malloc(num_angles * sizeof(float));
    if (!angles) return NULL;
    for (int i = 0; i < num_angles; i++) {
        angles[i] = i * 2.0f * M_PI / num_angles;
    }
    return angles;
}

kiss_fft_cpx*** srp_build_steering(
    const SRP_Config* cfg,
    const ArrayGeometry* geom,
    const float* angles
)
{
    int M;
    int F;
    int num_angles;
    float c;
    float sr;
    float NFFT;

    kiss_fft_cpx*** a_array;
    if (!cfg || !geom || !angles) return NULL;
    M = cfg->M;
    F = cfg->F;
    num_angles = cfg->num_angles;
    c = cfg->c;
    sr = cfg->sr;
    NFFT = cfg->NFFT;
    if (M <= 0 || F <= 0 ||
        num_angles <= 0 || geom->M != M || !geom->x || !geom->y ||
        !isfinite(c) || c <= 0.0f || !isfinite(sr) || sr <= 0.0f ||
        !isfinite(NFFT) || NFFT <= 0.0f) {
        return NULL;
    }
    a_array = (kiss_fft_cpx***)calloc(
        num_angles, sizeof(kiss_fft_cpx**));
    if (!a_array) return NULL;

    for (int a = 0; a < num_angles; a++) {
        a_array[a] =
            (kiss_fft_cpx**)calloc(M, sizeof(kiss_fft_cpx*));
        if (!a_array[a]) goto fail;
        for (int m = 0; m < M; m++) {
            a_array[a][m] = (kiss_fft_cpx*)malloc(F * sizeof(kiss_fft_cpx));
            if (!a_array[a][m]) goto fail;
        }
    }

    for (int a = 0; a < num_angles; a++) {
        float theta = angles[a];
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        for (int m = 0; m < M; m++) {
            float tau = -(geom->x[m] * cos_t + geom->y[m] * sin_t) / c;

            for (int f = 0; f < F; f++) {
                float freq = (float)f * sr / NFFT;
                float phase = -2.0f * M_PI * freq * tau;

                a_array[a][m][f].r = cosf(phase);
                a_array[a][m][f].i = sinf(phase);
            }
        }
    }

    return a_array;

fail:
    srp_destroy_steering(a_array, num_angles, M);
    return NULL;
}

void srp_destroy_steering(
    kiss_fft_cpx*** steering,
    int num_angles,
    int microphones)
{
    if (!steering) return;
    for (int angle = 0; angle < num_angles; ++angle) {
        if (!steering[angle]) continue;
        for (int microphone = 0; microphone < microphones; ++microphone) {
            free(steering[angle][microphone]);
        }
        free(steering[angle]);
    }
    free(steering);
}
