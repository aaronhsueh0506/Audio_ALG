/* Fixtures shared by the DFN2 pipeline tests and the board skeletons: the
 * unit ERB partition that makes the fail-open path an exact identity, a
 * deterministic far-end, and an identity model callback. stdio-free. */
#ifndef DFN_TEST_FIXTURE_H
#define DFN_TEST_FIXTURE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "dfn2_prepost.h"
#include "dfn_unit_partition.h"

/* The periodic sqrt-Hann window the pipelines analyse and synthesise with. */
static inline void dfn_fixture_root_hann(float *window, int n) {
    int i;
    for (i = 0; i < n; ++i)
        window[i] = sqrtf(0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)n));
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
            v += 0.06f * sinf(2.0f * 3.14159265f * tones[k] * t + (float)k);
        dst[i] = v;
    }
}

/* Test-side Hann-windowed-sinc low-pass (odd `taps`, zero-phase by
 * compensating the group delay), so a native-grid comparison measures the
 * rate bridge inside the resamplers' pass band: the conventional pipeline's
 * own output carries low-level content up to Nyquist (adaptation noise,
 * gain switching) that the bridge's FIR round trip is specified to remove. */
static inline void dfn_fixture_lowpass(const float *in, float *out, int n,
                                       int taps, float cutoff_norm) {
    double kernel[257];
    double wsum = 0.0;
    int half = taps / 2, i, k;
    if (taps > 257) taps = 257, half = 128;
    for (k = -half; k <= half; ++k) {
        double x = 2.0 * (double)cutoff_norm * (double)k;
        double sinc = k == 0 ? 1.0 : sin(3.14159265358979 * x) / (3.14159265358979 * x);
        double w = 0.5 + 0.5 * cos(3.14159265358979 * (double)k / (double)(half + 1));
        kernel[k + half] = 2.0 * (double)cutoff_norm * sinc * w;
        wsum += kernel[k + half];
    }
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
 * filter, alpha 0, recurrent state passed through unchanged. */
static inline int dfn_fixture_identity_infer(void *user, const DFN2PrepostInputs *in,
                                      DFN2PrepostOutputs *out) {
    size_t i;
    (void)user;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 1.0f;
    memset(out->coefs, 0, out->coefs_elements * sizeof(float));
    out->alpha[0] = 0.0f;
    memcpy(out->encoder_gru_hidden_next, in->encoder_gru_hidden,
           out->encoder_gru_hidden_elements * sizeof(float));
    memcpy(out->erb_gru_hidden_next, in->erb_gru_hidden,
           out->erb_gru_hidden_elements * sizeof(float));
    memcpy(out->df_gru_hidden_next, in->df_gru_hidden,
           out->df_gru_hidden_elements * sizeof(float));
    memcpy(out->df_convp_history_next, in->df_convp_history,
           out->df_convp_history_elements * sizeof(float));
    return 0;
}

#endif /* DFN_TEST_FIXTURE_H */
