#define _POSIX_C_SOURCE 200809L
/* Scalar/SIMD equivalence: one digest over the whole pre/post chain (analysis, features,
 * attenuation limit, expansion, gains, synthesis, normaliser states); the Makefile compares the
 * digests of the two builds. Usage: test_rnnoise_simd ERB_FWD_FILE ERB_INV_FILE */
#include "rnnoise_test_support.h"

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

int main(int argc, char **argv) {
    static float frame[RNN_TEST_N_FFT];
    static float re[RNN_TEST_N_BINS], im[RNN_TEST_N_BINS];
    static float erb[3][RNN_TEST_N_BANDS];
    static float spec[3][2][RNN_TEST_SPEC_BINS];
    static float band_gain[RNN_TEST_N_BANDS], bin_gain[RNN_TEST_N_BINS];
    static float output[RNN_TEST_HOP];
    uint64_t hash = UINT64_C(1469598103934665603);
    RNNoiseGrid grid;
    RnnErb mats;
    RnnPool pool;
    RnnFraming framing;
    AcStftConfig sc;

    if (argc != 3) {
        fprintf(stderr, "usage: %s ERB_FWD_FILE ERB_INV_FILE\n", argv[0]);
        return 2;
    }
    if (rnnoise_grid_make(&grid, &RNN_TEST_SHIPPED) != RNNOISE_GRID_OK) return 1;
    /* The hashed lengths below are the shipped grid's: pin them. */
    if (grid.n_fft != RNN_TEST_N_FFT || grid.hop != RNN_TEST_HOP ||
        grid.n_bins != RNN_TEST_N_BINS || grid.n_bands != RNN_TEST_N_BANDS ||
        grid.spec_bins != RNN_TEST_SPEC_BINS) {
        fprintf(stderr, "FAIL: shipped grid differs from the test's array sizes\n");
        return 1;
    }
    if (rnn_erb_read_grid(argv[1], argv[2], &grid, &mats) != RNN_ERB_OK) {
        fprintf(stderr, "FAIL: cannot read matching ERB matrix files: %s %s\n", argv[1], argv[2]);
        return 1;
    }
    memset(&pool, 0, sizeof(pool));
    if (rnnoise_config_defaults(&pool.cfg, &grid) != 0) return 1;
    pool.cfg.erb_fwd = mats.fwd;
    pool.cfg.erb_inv = mats.inv;
    if (rnn_pool_open(&pool) != 0) return 1;
    rnn_framing_config(&sc, &grid);
    if (rnn_framing_open(&framing, &sc) != 0) return 1;

    for (int frame_index = 0; frame_index < 24; ++frame_index) {
        for (int i = 0; i < RNN_TEST_N_FFT; ++i) frame[i] = next_sample();
        for (int b = 0; b < RNN_TEST_N_BANDS; ++b) {
            band_gain[b] = 0.05f + 0.9f *
                (float)((b * 7 + frame_index * 3) % 23) / 22.0f;
        }
        ac_stft_frame_planar(framing.stft, frame, re, im);
        hash = hash_bytes(hash, re, sizeof(re));
        hash = hash_bytes(hash, im, sizeof(im));
        if (rnnoise_compute_features(pool.st, re, im, &erb[0][0], &spec[0][0][0]) == 1) {
            hash = hash_bytes(hash, erb, sizeof(erb));
            hash = hash_bytes(hash, spec, sizeof(spec));
        }
        rnnoise_apply_atten_lim(pool.st, band_gain, 18.0f);
        rnnoise_expand_gains(pool.st, band_gain, bin_gain);
        hash = hash_bytes(hash, bin_gain, sizeof(bin_gain));
        rnnoise_apply_gains(pool.st, re, im, bin_gain);
        if (ac_istft_push_planar(framing.istft, re, im, output) != RNN_TEST_HOP) return 1;
        hash = hash_bytes(hash, output, sizeof(output));
    }
    hash = hash_bytes(hash, pool.st->erb_norm_state,
                      (size_t)grid.n_bands * sizeof(float));
    hash = hash_bytes(hash, pool.st->spec_norm_state,
                      (size_t)grid.spec_bins * sizeof(float));
    fprintf(stderr, "rnnoise preprocessing backend=%s\n",
            rnnoise_simd_backend());
    printf("%016llx\n", (unsigned long long)hash);
    rnn_framing_close(&framing);
    free(pool.pool);
    rnn_erb_free(&mats);
    return 0;
}
