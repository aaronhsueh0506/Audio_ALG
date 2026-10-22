#ifndef RNNOISE_TEST_FRAMING_H
#define RNNOISE_TEST_FRAMING_H
#include "process.h"
#include <stdlib.h>
#include <string.h>

static void *rnnoise_test_aligned_block(size_t bytes) {
    void *mem = NULL;
    if (bytes == 0 || posix_memalign(&mem, 16, bytes) != 0) return NULL;
    memset(mem, 0, bytes);
    return mem;
}

typedef struct {
    FftHandle *fft;
    void *analysis_mem, *synthesis_mem;
    AcStft *analysis;
    AcIstft *synthesis;
} RNNoiseTestFraming;

static void rnnoise_test_framing_free(RNNoiseTestFraming *f) {
    free(f->analysis_mem);
    free(f->synthesis_mem);
    fft_destroy(f->fft);
}

static int rnnoise_test_framing_init(RNNoiseTestFraming *f, RNNoiseState *st) {
    AcStftConfig cfg;
    memset(f, 0, sizeof(*f));
    f->fft = fft_create(RNNOISE_N_FFT);
    if (!f->fft) return -1;
    rnnoise_framing_config(&cfg, f->fft);
    size_t a = ac_stft_get_mem_size(&cfg), b = ac_istft_get_mem_size(&cfg);
    f->analysis_mem = rnnoise_test_aligned_block(a);
    f->synthesis_mem = rnnoise_test_aligned_block(b);
    f->analysis = ac_stft_init(f->analysis_mem, a, &cfg);
    f->synthesis = ac_istft_init(f->synthesis_mem, b, &cfg);
    if (rnnoise_state_init(st, f->analysis, f->synthesis) == 0) return 0;
    rnnoise_test_framing_free(f);
    return -1;
}
#endif
