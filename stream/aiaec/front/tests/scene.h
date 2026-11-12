/* Synthetic echo scene: white far end, mic = delayed two-tap echo of it + weak noise. */
#ifndef FRONT_TEST_SCENE_H
#define FRONT_TEST_SCENE_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Scene {
    int n;
    int16_t *mic, *far;
} Scene;

static uint32_t scene_rng(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

static float scene_unit(uint32_t *s) {
    return ((float)scene_rng(s) / 8388608.0f) - 1.0f;
}

static Scene scene_make(int n, int delay) {
    Scene sc;
    uint32_t r = 12345u;
    int i;
    sc.n = n;
    sc.mic = (int16_t *)calloc((size_t)n, sizeof(int16_t));
    sc.far = (int16_t *)calloc((size_t)n, sizeof(int16_t));
    for (i = 0; i < n; ++i) sc.far[i] = (int16_t)lrintf(scene_unit(&r) * 9000.0f);
    for (i = 0; i < n; ++i) {
        float e = 0.0f;
        if (i >= delay) e += 0.5f * (float)sc.far[i - delay];
        if (i >= delay + 7) e += 0.25f * (float)sc.far[i - delay - 7];
        e += scene_unit(&r) * 200.0f;
        sc.mic[i] = (int16_t)lrintf(e);
    }
    return sc;
}

static void scene_free(Scene *sc) {
    free(sc->mic);
    free(sc->far);
}

#endif
