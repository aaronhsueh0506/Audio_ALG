#define _POSIX_C_SOURCE 200809L
#include "process.h"
#include "rnnoise_test_framing.h"
#include <math.h>
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL: line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static int rejected(const AcStftConfig *analysis, const AcStftConfig *synthesis) {
    RNNoiseState state;
    size_t a = ac_stft_get_mem_size(analysis), b = ac_istft_get_mem_size(synthesis);
    void *amem = rnnoise_test_aligned_block(a), *bmem = rnnoise_test_aligned_block(b);
    AcStft *stft = ac_stft_init(amem, a, analysis);
    AcIstft *istft = ac_istft_init(bmem, b, synthesis);
    if (!stft || !istft) {
        fprintf(stderr, "FAIL: rejection fixture did not construct both framing objects\n");
        free(amem); free(bmem);
        return 0;
    }
    int result = rnnoise_state_init(&state, stft, istft);
    int ok = result == -1 && state.stft == NULL && state.istft == NULL;
    free(amem); free(bmem);
    return ok;
}

int main(void) {
    RNNoiseState state;
    RNNoiseTestFraming framing;
    AcStftConfig cfg, other, readback;
    float window[RNNOISE_N_FFT];
    float geometry_window[1024] = {0};
    float re[RNNOISE_N_BINS], im[RNNOISE_N_BINS], gain[RNNOISE_N_BINS];
    float out[RNNOISE_HOP_LEN + 2], frame[RNNOISE_N_FFT];
    CHECK(RNNOISE_MODEL_IO_LAYOUT_VERSION == 1);
    CHECK(RNNOISE_PREPOST_LAYOUT_VERSION == 2);
    CHECK(rnnoise_test_framing_init(&framing, &state) == 0);
    rnnoise_framing_config(&cfg, framing.fft);
    CHECK(ac_stft_get_config(NULL, &readback) == -1);
    CHECK(ac_stft_get_config(framing.analysis, NULL) == -1);
    CHECK(ac_istft_get_config(NULL, &readback) == -1);
    CHECK(ac_istft_get_config(framing.synthesis, NULL) == -1);
    CHECK(ac_stft_get_config(framing.analysis, &readback) == 0);
    CHECK(readback.fft_size == 512 && readback.center == 0);
    CHECK(readback.spec_scale == cfg.spec_scale && readback.window == cfg.window);
    readback.fft_size = 1024;
    CHECK(ac_stft_get_config(framing.analysis, &readback) == 0 && readback.fft_size == 512);
    CHECK(ac_istft_get_config(framing.synthesis, &readback) == 0);
    CHECK(readback.ola == AC_OLA_ACCUMULATE && readback.ispec_scale == 1.0f);

    /* Full-sized borrowed table for the largest fixture. Its first 512 values
     * match the model table, so geometry alone must reject these objects. */
    memcpy(geometry_window, cfg.window, sizeof(window));
    other = cfg; other.fft = NULL; other.window = geometry_window;
    other.fft_size = 1024;
    CHECK(rejected(&other, &cfg)); CHECK(rejected(&cfg, &other));
    other.fft_size = 256;
    CHECK(rejected(&other, &cfg)); CHECK(rejected(&cfg, &other));
    other = cfg; other.center = 1; CHECK(rejected(&other, &cfg));
    other = cfg; other.spec_scale = 1.0f; CHECK(rejected(&other, &cfg));
    other = cfg; other.ispec_scale = 22.62741699796952f; CHECK(rejected(&cfg, &other));
    other = cfg; other.ola = AC_OLA_ENV_NORM; CHECK(rejected(&cfg, &other));
    other = cfg; other.spec_scale = NAN; CHECK(rejected(&other, &cfg));
    other = cfg; other.ispec_scale = NAN; CHECK(rejected(&cfg, &other));
    memcpy(window, cfg.window, sizeof(window)); window[17] = 0.0f;
    other = cfg; other.window = window;
    CHECK(rejected(&other, &cfg)); CHECK(rejected(&cfg, &other));

    for (int i = 0; i < RNNOISE_N_FFT; ++i) frame[i] = sinf((float)i * 0.123f) * 0.1f;
    for (int i = 0; i < RNNOISE_N_BINS; ++i) re[i] = im[i] = gain[i] = 1.0f;
    for (int i = 0; i < RNNOISE_HOP_LEN + 2; ++i) out[i] = 123.0f;
    CHECK(rnnoise_state_init(&state, NULL, framing.synthesis) == -1);
    CHECK(rnnoise_analysis(&state, frame, re, im) == -1);
    CHECK(rnnoise_synthesis(&state, re, im, gain, out + 1) == -1);
    for (int i = 0; i < RNNOISE_N_BINS; ++i) CHECK(re[i] == 1.0f && im[i] == 1.0f);
    for (int i = 0; i < RNNOISE_HOP_LEN + 2; ++i) CHECK(out[i] == 123.0f);
    CHECK(rnnoise_state_init(&state, framing.analysis, framing.synthesis) == 0);
    CHECK(rnnoise_analysis(&state, frame, re, im) == 0);
    CHECK(rnnoise_synthesis(&state, re, im, gain, out + 1) == 0);
    CHECK(out[0] == 123.0f && out[RNNOISE_HOP_LEN + 1] == 123.0f);
    CHECK(rnnoise_state_init(NULL, framing.analysis, framing.synthesis) == -1);

    size_t a = ac_stft_get_mem_size(&cfg), b = ac_istft_get_mem_size(&cfg);
    printf("INFO: state=%zu FFT=%zu STFT=%zu ISTFT=%zu total=%zu bytes\n",
           sizeof(state), fft_get_mem_size(512), a, b,
           sizeof(state) + fft_get_mem_size(512) + a + b);
    rnnoise_test_framing_free(&framing);
    puts("PASS: external framing geometry, scale, window, mode and output bounds");
    return 0;
}
