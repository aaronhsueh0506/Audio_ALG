/* Test helper: deterministic mic / far-end streams that exercise the AEC's output guard and its hard
 * post-chain reset event. Header-only; test code only (heap, no error recovery beyond the status code).
 *
 * guard_scene_make() builds speech-like far-end bursts (or a steady one), a piecewise echo path whose
 * delay changes at 1/3 and 2/3 of the stream (each change that restarts the filter moves
 * aec_post_reset_serial()), an optional near-end burst and an optional silent head with no echo path
 * (the estimate never becomes usable and the guard fires). The application-issued aec_reset() hops of a
 * scene are listed in reset_at (-1 terminated); the test applies them to every instance. Shared by
 * mono_aec_nr_res/tests/test_mono_chain.c and ulcnet/mono_alignulcnet/tests/test_mono_ulcnet.c. */
#ifndef GUARD_SCENE_H
#define GUARD_SCENE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_gs = 0x2545F491u;
static inline float gs_rnd(void) {                   /* xorshift32, uniform in [-1, 1) */
    g_gs ^= g_gs << 13; g_gs ^= g_gs >> 17; g_gs ^= g_gs << 5;
    return (float)(g_gs & 0xFFFFFFu) / (float)0x800000u - 1.0f;
}

typedef struct GuardScene {
    const char* name;
    int sr, fft, hop, nh;
    float *mic, *far;
    int enable_cng;
    int reset_at[4];        /* hops on which the application calls aec_reset(); -1 terminates */
} GuardScene;

/* Speech-like far-end bursts (or a steady one), a piecewise echo path whose delay changes at 1/3 and
 * 2/3 of the stream, an optional near-end burst, an optional silent head. */
static inline int guard_scene_make(GuardScene* s, const char* name, int sr, int fft, int seconds, uint32_t seed,
                            int speech_far, int echo, int near_burst, int silent_head_hops) {
    int n, i, seg = sr / 4;
    int d1 = 480 * (sr / 16000), d2 = 1100 * (sr / 16000), d3 = 240 * (sr / 16000);
    float env = 0.2f, lp = 0.0f, nlp = 0.0f;
    float* nearv;
    memset(s, 0, sizeof(*s));
    s->name = name; s->sr = sr; s->fft = fft; s->hop = fft / 2;
    s->nh = seconds * sr / s->hop;
    s->enable_cng = 1;
    for (i = 0; i < 4; ++i) s->reset_at[i] = -1;
    n = s->nh * s->hop;
    s->mic = (float*)calloc((size_t)n, sizeof(float));
    s->far = (float*)calloc((size_t)n, sizeof(float));
    nearv = (float*)calloc((size_t)n, sizeof(float));
    if (!s->mic || !s->far || !nearv) { free(nearv); return -1; }
    g_gs = seed;
    for (i = 0; i < n; ++i) {
        float v;
        if (i % seg == 0) {
            float r = gs_rnd();
            env = speech_far ? (r > 0.3f ? 0.35f : (r > -0.4f ? 0.12f : 0.0f)) : 0.45f;
        }
        lp = 0.6f * lp + 0.4f * gs_rnd();
        v = env * lp * 2.0f;
        s->far[i] = (i < silent_head_hops * s->hop) ? 0.0f : v;
        nlp = 0.7f * nlp + 0.3f * gs_rnd();
        nearv[i] = (near_burst && i > n / 3 && i < n / 3 + n / 6) ? 0.25f * nlp * 2.0f : 0.004f * nlp;
    }
    for (i = 0; i < n; ++i) {
        int d = i < n / 3 ? d1 : (i < 2 * n / 3 ? d2 : d3);
        float e = 0.0f;
        if (echo) {
            if (i - d >= 0) e += 0.45f * s->far[i - d];
            if (i - d - 37 >= 0) e += 0.2f * s->far[i - d - 37];
        }
        s->mic[i] = nearv[i] + e;
    }
    free(nearv);
    return 0;
}

static inline void guard_scene_free(GuardScene* s) { free(s->mic); free(s->far); s->mic = s->far = NULL; }

#endif /* GUARD_SCENE_H */
