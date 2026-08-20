/* Per-stage cost of masker_step(), plus the cost of the clock used to
 * measure it. Build and run ON THE TARGET -- the whole point is that the
 * development host and a Cortex-A53 disagree about which of these is
 * expensive, so a host number answers nothing.
 *
 *   make BACKEND=ne10 SIMD=1 bench-masker
 *   make BACKEND=ne10 SIMD=0 bench-masker      <- the decisive A/B
 *
 * If SIMD=0 is FASTER, the shared NEON kernels are the regression: on an
 * A53 the NEON datapath is 64-bit internally, so a 128-bit op issues as two
 * micro-ops, and vld2q_f32 (which sk_cmag_f32 uses to de-interleave Complex)
 * is a multi-cycle structure load. Both erase the 4x a 128-bit ISA suggests.
 *
 * Stage totals are reported separately from masker_step() so a regression
 * can be attributed instead of guessed at.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "masker.h"
#include "simd_kernels.h"

#ifndef BENCH_FRAMES
#define BENCH_FRAMES 2000
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static unsigned int rng = 0x1234567u;
static float noise(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((float)(rng >> 8) * (1.0f / 16777216.0f)) - 0.5f;
}

int main(int argc, char** argv) {
    int nfft = argc > 1 ? atoi(argv[1]) : 512;
    int F = nfft / 2 + 1;
    MaskerConfig cfg;
    MaskEstimator* m;
    Complex* frame;
    float* a; float* b;
    double t0, clock_cost, total, s_cmag, s_db, s_ema;
    int i, f;

    memset(&cfg, 0, sizeof(cfg));
    cfg.NFFT = nfft; cfg.sr = 16000;
    cfg.E_alpha_up = 0.995f; cfg.E_alpha_down = 0.9f; cfg.margin_dB = 6.0f;
    cfg.low_freq = 300.0f; cfg.high_freq = 7000.0f;
    cfg.M_alpha = 0.95f; cfg.spp_thr = 0.5f; cfg.spp_upd_thr = 0.3f;
    cfg.enable_freq_smooth = 1; cfg.smooth_size = 5;
    cfg.enable_time_smooth = 1; cfg.T_alpha = 0.7f;
    cfg.enable_energy = 1; cfg.enable_spp = 1; cfg.enable_band = 1;

    m = masker_create(&cfg);
    frame = (Complex*)malloc((size_t)F * sizeof(Complex));
    a = (float*)malloc((size_t)F * sizeof(float));
    b = (float*)malloc((size_t)F * sizeof(float));
    if (!m || !frame || !a || !b) { printf("alloc failed\n"); return 1; }
    for (f = 0; f < F; ++f) {
        frame[f].r = noise() * 4.0f; frame[f].i = noise() * 4.0f;
        a[f] = 1e-3f + (float)f * 1e-4f; b[f] = a[f];
    }

    /* What one clock read costs here -- the number that decides whether the
     * AEC/pipeline stage timing is worth compiling out on this target. */
    t0 = now_s();
    for (i = 0; i < 200000; ++i) (void)now_s();
    clock_cost = (now_s() - t0) / 200000.0;

    t0 = now_s();
    for (i = 0; i < BENCH_FRAMES; ++i) masker_step(m, frame);
    total = (now_s() - t0) / BENCH_FRAMES;

    t0 = now_s();
    for (i = 0; i < BENCH_FRAMES; ++i) sk_cmag_f32(a, frame, F, 1e-8f);
    s_cmag = (now_s() - t0) / BENCH_FRAMES;

    t0 = now_s();
    for (i = 0; i < BENCH_FRAMES; ++i) sk_linear_to_db_f32(b, a, F);
    s_db = (now_s() - t0) / BENCH_FRAMES;

    t0 = now_s();
    for (i = 0; i < BENCH_FRAMES; ++i)
        sk_asym_ema_f32(a, b, F, cfg.E_alpha_up, cfg.E_alpha_down);
    s_ema = (now_s() - t0) / BENCH_FRAMES;

    /* Printed first because a wrong answer here dwarfs anything measured
     * below: an -O0 build or a scalar-only build explains a large regression
     * on its own, and neither is visible from a timing number alone. */
    printf("backend=%s optimize=%s nfft=%d F=%d frames=%d\n",
           SK_HAVE_NEON ? "neon" : "scalar",
#ifdef __OPTIMIZE__
           "yes",
#else
           "NO (-O0! this alone explains a large slowdown)",
#endif
           nfft, F, BENCH_FRAMES);
    printf("clock_gettime      %8.1f ns/call\n", clock_cost * 1e9);
    printf("masker_step        %8.2f us/frame   (100%%)\n", total * 1e6);
    printf("  sk_cmag_f32      %8.2f us/frame   (%5.1f%%)\n",
           s_cmag * 1e6, 100.0 * s_cmag / total);
    printf("  sk_linear_to_db  %8.2f us/frame   (%5.1f%%)\n",
           s_db * 1e6, 100.0 * s_db / total);
    printf("  sk_asym_ema_f32  %8.2f us/frame   (%5.1f%%)\n",
           s_ema * 1e6, 100.0 * s_ema / total);
    printf("  remainder        %8.2f us/frame   (%5.1f%%)  "
           "<- per-bin SPP loop (log1pf+expf), masks, smoothing\n",
           (total - s_cmag - s_db - s_ema) * 1e6,
           100.0 * (total - s_cmag - s_db - s_ema) / total);

    masker_destroy(m); free(frame); free(a); free(b);
    return 0;
}
