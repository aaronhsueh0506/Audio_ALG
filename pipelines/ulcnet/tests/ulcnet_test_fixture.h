/* Shared fixture of the Align-ULCNet pipeline tests (mono, 4ch and the stage alone): the grids they run, the grid under test, the
 * descriptor of that grid, and the stand-in models that need no state of their own. Header-only; define nothing it
 * names before including it. A test includes it after "ulcnet/ulcnet_nn_stage.h". */

#ifndef ULCNET_TEST_FIXTURE_H
#define ULCNET_TEST_FIXTURE_H

#include <math.h>
#include <string.h>

enum { HOP_MAX = 512, BINS_MAX = 513 };

static const UlcnetGridSpec k_grids[] = {
    { .sample_rate = 16000, .n_fft = 512,  .delay_depth = 8 },
    { .sample_rate = 48000, .n_fft = 1024, .delay_depth = 8 },
};

static UlcnetGrid g_grid;   /* the grid under test */
#define HOP (g_grid.hop)
#define BINS (g_grid.n_bins)

/* Every test model that actually infers must publish a descriptor: the stage refuses infer-without-descriptor,
 * because a shape disagreement between the graph and the stage's grid is undetectable downstream (the finite guard
 * catches an unwritten output, never a wrong-shaped one). The descriptor of the grid under test. */
static inline const UlcnetModelIoDescriptor* test_io_descriptor(void) {
    static UlcnetModelIoDescriptor d;
    return ulcnet_model_io_descriptor_default(&g_grid, &d) == 0 ? &d : NULL;
}

/* The table row that is not the grid under test. */
static inline UlcnetGrid other_grid(void) {
    UlcnetGrid other;
    memset(&other, 0, sizeof(other));
    for (size_t i = 0; i < sizeof(k_grids) / sizeof(k_grids[0]); ++i)
        if (k_grids[i].n_fft != g_grid.n_fft) (void)ulcnet_grid_make(&other, &k_grids[i]);
    return other;
}

/* Reprime hops a block of `frames` hops skips out of the `*armed` still owed. */
static inline int reprime_take(int* armed, int frames) {
    int skipped = *armed < frames ? *armed : frames;
    *armed -= skipped;
    return skipped;
}

/* The model returns the error spectrum. */
static inline int gate_infer_identity(void* user,
                                      const float* err_re, const float* err_im,
                                      const float* far_re, const float* far_im,
                                      float* out_re, float* out_im) {
    (void)user; (void)far_re; (void)far_im;
    memcpy(out_re, err_re, sizeof(float) * BINS);
    memcpy(out_im, err_im, sizeof(float) * BINS);
    return 0;
}

/* The model returns the far spectrum. */
static inline int passthrough_far_infer(void* user,
                                        const float* err_re, const float* err_im,
                                        const float* far_re, const float* far_im,
                                        float* out_re, float* out_im) {
    (void)user; (void)err_re; (void)err_im;
    memcpy(out_re, far_re, BINS * sizeof(float));
    memcpy(out_im, far_im, BINS * sizeof(float));
    return 0;
}

static inline float spec_peak(const float* re, const float* im) {
    float m = 0.0f;
    for (int k = 0; k < BINS; ++k) {
        float a = fabsf(re[k]);
        if (a > m) m = a;
        a = fabsf(im[k]);
        if (a > m) m = a;
    }
    return m;
}

#endif /* ULCNET_TEST_FIXTURE_H */
