/**
 * hpf.c - 2nd-order Butterworth IIR High-Pass Filter
 *
 * Bilinear transform from analog prototype.
 * Direct Form II transposed implementation.
 * Matches Python HighPassFilter exactly.
 */

#include "hpf.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Hpf {
    float b0, b1, b2;
    float a1, a2;
    float z1, z2;
};

Hpf* hpf_create(float cutoff_hz, int sample_rate) {
    if (cutoff_hz <= 0 || sample_rate <= 0) return NULL;

    Hpf* h = (Hpf*)calloc(1, sizeof(Hpf));
    if (!h) return NULL;

    /* Bilinear transform: pre-warp analog frequency */
    float wc = 2.0f * (float)M_PI * cutoff_hz / (float)sample_rate;
    float wc_w = tanf(wc / 2.0f);
    float k = wc_w * wc_w;
    float sqrt2 = 1.41421356237f;
    float norm = 1.0f / (1.0f + sqrt2 * wc_w + k);

    /* Transfer function coefficients */
    h->b0 =  norm;
    h->b1 = -2.0f * norm;
    h->b2 =  norm;
    h->a1 =  2.0f * (k - 1.0f) * norm;
    h->a2 =  (1.0f - sqrt2 * wc_w + k) * norm;

    h->z1 = 0.0f;
    h->z2 = 0.0f;

    return h;
}

void hpf_destroy(Hpf* hpf) {
    free(hpf);
}

void hpf_process(Hpf* h, float* data, int n) {
    if (!h || !data) return;

    float b0 = h->b0, b1 = h->b1, b2 = h->b2;
    float a1 = h->a1, a2 = h->a2;
    float z1 = h->z1, z2 = h->z2;

    for (int i = 0; i < n; i++) {
        float xi = data[i];
        float yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        data[i] = yi;
    }

    h->z1 = z1;
    h->z2 = z2;
}

void hpf_reset(Hpf* hpf) {
    if (hpf) {
        hpf->z1 = 0.0f;
        hpf->z2 = 0.0f;
    }
}
