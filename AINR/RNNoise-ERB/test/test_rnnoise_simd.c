/* Scalar/SIMD equivalence test. */
#include "process.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t rng_state = 0x31415926u;

static float next_sample(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)((rng_state >> 8) & 0x00ffffffu) / 8388608.0f - 1.0f)
        * 0.2f;
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    while (bytes-- > 0) {
        hash ^= (uint64_t)*p++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(void) {
    RNNoiseState state;
    float frame[RNNOISE_WIN_LEN];
    float re[RNNOISE_N_BINS];
    float im[RNNOISE_N_BINS];
    float erb[3][RNNOISE_N_BANDS];
    float spec[3][2][RNNOISE_SPEC_BINS];
    float band_gain[RNNOISE_N_BANDS];
    float bin_gain[RNNOISE_N_BINS];
    float output[RNNOISE_HOP_LEN];
    uint64_t hash = UINT64_C(1469598103934665603);

    rnnoise_state_init(&state);
    for (int frame_index = 0; frame_index < 24; ++frame_index) {
        for (int i = 0; i < RNNOISE_WIN_LEN; ++i) frame[i] = next_sample();
        for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
            band_gain[b] = 0.05f + 0.9f *
                (float)((b * 7 + frame_index * 3) % 23) / 22.0f;
        }
        rnnoise_analysis(&state, frame, re, im);
        hash = hash_bytes(hash, re, sizeof(re));
        hash = hash_bytes(hash, im, sizeof(im));
        if (rnnoise_compute_features(&state, re, im, erb, spec)) {
            hash = hash_bytes(hash, erb, sizeof(erb));
            hash = hash_bytes(hash, spec, sizeof(spec));
        }
        rnnoise_apply_atten_lim(band_gain, 18.0f);
        rnnoise_expand_gains(band_gain, bin_gain);
        hash = hash_bytes(hash, bin_gain, sizeof(bin_gain));
        rnnoise_synthesis(&state, re, im, bin_gain, output);
        hash = hash_bytes(hash, output, sizeof(output));
    }
    hash = hash_bytes(hash, state.erb_norm_state,
                      sizeof(state.erb_norm_state));
    hash = hash_bytes(hash, state.spec_norm_state,
                      sizeof(state.spec_norm_state));
    fprintf(stderr, "rnnoise preprocessing backend=%s\n",
            rnnoise_simd_backend());
    printf("%016llx\n", (unsigned long long)hash);
    return 0;
}
