/* Fixtures shared by the DFN2 pipeline tests and the board skeletons: the
 * unit ERB partition that makes the fail-open path an exact identity, a
 * deterministic far-end, and an identity model callback. stdio-free. */
#ifndef DFN_TEST_FIXTURE_H
#define DFN_TEST_FIXTURE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "dfn2_prepost.h"
#include "dfn/dfn_unit_partition.h"
#include "audio_window.h"

/* The one pi of the fixtures (a double literal; the float-precision tones use its float rounding). */
#define DFN_FIXTURE_PI 3.14159265358979

/* The periodic sqrt-Hann window the pipelines analyse and synthesise with. */
static inline void dfn_fixture_root_hann(float *window, int n) {
    ac_window_build(window, n);
}

/* Uniform noise in [-0.25, 0.25) from the LCG the sibling tests use. */
static inline void dfn_fixture_lcg_far(float *dst, size_t count, uint32_t seed) {
    size_t i;
    for (i = 0; i < count; ++i) {
        seed = seed * 1664525u + 1013904223u;
        dst[i] = ((float)(int)(seed >> 9) / 4194304.0f - 1.0f) * 0.25f;
    }
}

/* A band-limited far-end (16 tones between 211 and 2999 Hz, fixed phases)
 * for the native-grid gates: rich enough for the AEC to adapt on, and
 * inside the resamplers' pass band, so the round trip through the rate
 * bridge is compared where it is specified to be transparent. */
static inline void dfn_fixture_multitone(float *dst, size_t count,
                                         int sample_rate) {
    static const float tones[16] = {
        211.0f, 347.0f, 503.0f, 659.0f, 811.0f, 977.0f, 1123.0f, 1301.0f,
        1487.0f, 1693.0f, 1901.0f, 2129.0f, 2357.0f, 2609.0f, 2803.0f, 2999.0f };
    size_t i;
    int k;
    for (i = 0; i < count; ++i) {
        float t = (float)i / (float)sample_rate, v = 0.0f;
        for (k = 0; k < 16; ++k)
            v += 0.06f * sinf(2.0f * (float)DFN_FIXTURE_PI * tones[k] * t + (float)k);
        dst[i] = v;
    }
}

/* Hann-windowed sinc, 2 * half + 1 taps: kernel[m + half] is the low-pass of normalised cutoff
 * `cutoff_norm` sampled at m - shift, windowed over +-(half + 1). Returns the kernel sum, which
 * the callers divide by so the DC gain is exactly one. */
static inline double dfn_fixture_hann_sinc(double *kernel, int half, double cutoff_norm,
                                           double shift) {
    double wsum = 0.0;
    int m;
    for (m = -half; m <= half; ++m) {
        double u = (double)m - shift;
        double x = 2.0 * cutoff_norm * u;
        double sinc = fabs(x) < 1e-12 ? 1.0 : sin(DFN_FIXTURE_PI * x) / (DFN_FIXTURE_PI * x);
        double w = 0.5 + 0.5 * cos(DFN_FIXTURE_PI * u / (double)(half + 1));
        kernel[m + half] = 2.0 * cutoff_norm * sinc * w;
        wsum += kernel[m + half];
    }
    return wsum;
}

/* Test-side Hann-windowed-sinc low-pass (odd `taps`, zero-phase by
 * compensating the group delay), so a native-grid comparison measures the
 * rate bridge inside the resamplers' pass band: the conventional pipeline's
 * own output carries low-level content up to Nyquist (adaptation noise,
 * gain switching) that the bridge's FIR round trip is specified to remove. */
static inline void dfn_fixture_lowpass(const float *in, float *out, int n,
                                       int taps, float cutoff_norm) {
    double kernel[257];
    double wsum;
    int half = taps / 2, i, k;
    if (taps > 257) taps = 257, half = 128;
    wsum = dfn_fixture_hann_sinc(kernel, half, (double)cutoff_norm, 0.0);
    for (i = 0; i < n; ++i) {
        double acc = 0.0;
        for (k = -half; k <= half; ++k) {
            int j = i + k;
            if (j >= 0 && j < n) acc += kernel[k + half] * (double)in[j];
        }
        out[i] = (float)(acc / wsum);
    }
}

/* A model whose heads are the exact identity: unit ERB mask, zero deep
 * filter, alpha 0. The recurrent state passes through unchanged by writing
 * nothing: each state output is the same buffer as its input. */
static inline int dfn_fixture_identity_infer(void *user, const DFN2PrepostInputs *in,
                                      DFN2PrepostOutputs *out) {
    size_t i;
    (void)user;
    (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 1.0f;
    memset(out->coefs, 0, out->coefs_elements * sizeof(float));
    out->alpha[0] = 0.0f;
    return 0;
}

/* A model whose heads are a uniform 0.5 band mask, zero deep filter and alpha 0: an output that
 * differs from the input, for comparing the callback-driven hop with the caller-run one. */
static inline int dfn_fixture_half_mask_infer(void *user, const DFN2PrepostInputs *in,
                                              DFN2PrepostOutputs *out) {
    size_t i;
    (void)user;
    (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.5f;
    memset(out->coefs, 0, out->coefs_elements * sizeof(float));
    out->alpha[0] = 0.0f;
    return 0;
}

/* What a phased-hop test does with an open model frame: run the model, report a failure, leave
 * the frame open, or never fetch it. */
enum { BP_RUN, BP_FAIL, BP_OPEN, BP_UNFETCHED };

/* The stage's output delay at 48 kHz, (L + 1) * 512 samples with L = mask + df lookahead. In
 * native samples at `sr` (8 or 16 kHz) it is a whole number only when L + 1 is a multiple of 3
 * (the shipped 1/1 pair); otherwise the rate bridge's exact delay carries a third of a sample. */
#define DFN_FIXTURE_STAGE_DELAY_48K ((DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD + 1) * DFN2_HOP_LEN)

/* Native samples, in sixths, the rate bridge adds at `sr` / `hop`: the prefill of the grid's
 * schedule, the 16-sample group delay of each resampler and the stage's delay (shipped pair at
 * 16 kHz hop 128 / 16 kHz hop 256 / 8 kHz hop 128: 672 / 629 / 330 samples). */
static inline int dfn_fixture_bridge_added_delay_sixths(int sr, int hop) {
    int prefill = (sr == 16000 && hop == 128) ? 128 : (sr == 16000 && hop == 256) ? 85 : 42;
    return (prefill + 16 + 16) * 6 + (int)((long long)DFN_FIXTURE_STAGE_DELAY_48K * sr / 8000);
}

/* The same delay floored to whole native samples (dfn_rate_bridge_added_delay_samples()). */
static inline int dfn_fixture_bridge_added_delay(int sr, int hop) {
    return dfn_fixture_bridge_added_delay_sixths(sr, hop) / 6;
}

/* out[i] = in[i - delay] for a delay of `delay_sixths` / 6 native samples, by a Hann-windowed
 * sinc of 129 taps (exact for a whole delay). Exact only for content well below Nyquist; the
 * first and last 64 samples see a truncated kernel. */
static inline void dfn_fixture_delay_sixths(const float *in, float *out, int n, int delay_sixths) {
    const int whole = delay_sixths / 6;
    const double frac = (double)(delay_sixths % 6) / 6.0;
    double kernel[129], wsum;
    int i, m;
    wsum = dfn_fixture_hann_sinc(kernel, 64, 0.5, frac);
    for (i = 0; i < n; ++i) {
        double acc = 0.0;
        for (m = -64; m <= 64; ++m) {
            int j = i - whole - m;
            if (j >= 0 && j < n) acc += kernel[m + 64] * (double)in[j];
        }
        out[i] = (float)(acc / wsum);
    }
}

#endif /* DFN_TEST_FIXTURE_H */
