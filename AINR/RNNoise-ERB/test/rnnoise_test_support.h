#ifndef RNNOISE_TEST_SUPPORT_H
#define RNNOISE_TEST_SUPPORT_H
/* Shared by the RNNoise-ERB C tests: the shipped grid, pool and framing setup on top of the
 * ERB matrix reader. Include after defining _POSIX_C_SOURCE 200809L. */

#include "process.h"
#include "audio_stft.h"
#include "rnnoise_test_erb.h"

/* The config.ini values; tests/test_feature_contract.py pins them. */
static const RNNoiseGridSpec RNN_TEST_SHIPPED = {
    .sample_rate = 16000,
    .n_fft = 512,
    .n_bands = 22,
    .min_bins_per_band = 2,
    .lookahead_frames = 1,
    .spec_max_hz = 4000,
    .erb_norm_tau_sec = 1.0,
    .erb_norm_init_lo_db = -20.0f,
    .erb_norm_init_hi_db = -45.0f,
    .erb_norm_scale_db = 40.0f,
    .spec_norm_tau_sec = 1.0,
    .spec_norm_init_lo = 0.04f,
    .spec_norm_init_hi = 0.008f,
    .spec_norm_eps = 1e-12f,
};
enum { RNN_TEST_N_FFT = 512, RNN_TEST_HOP = 256, RNN_TEST_N_BINS = 257,
       RNN_TEST_N_BANDS = 22, RNN_TEST_SPEC_BINS = 129 };

static inline int rnn_erb_read_grid(const char *fwd_path, const char *inv_path,
                                    const RNNoiseGrid *grid, RnnErb *out)
{
    return rnn_erb_read_paths(fwd_path, inv_path, grid->n_bins, grid->n_bands, out);
}

/* A library state in a freshly allocated pool; fill p->cfg (rnnoise_config_defaults(), the
 * matrices, any registration) first. */
typedef struct RnnPool {
    RNNoiseConfig  cfg;
    RNNoiseMemReq  req;
    void          *pool;
    RNNoiseState  *st;
} RnnPool;

static inline int rnn_pool_open(RnnPool *p)
{
    if (rnnoise_get_mem_size(&p->cfg, &p->req) != 0) return -1;
    p->pool = rnn_test_aligned((size_t)p->req.bytes);
    if (!p->pool) return -1;
    p->st = rnnoise_init(p->pool, (size_t)p->req.bytes, &p->cfg);
    return p->st ? 0 : -1;
}

/* The framing process.h prescribes, owned by the application. */
typedef struct RnnFraming {
    void    *analysis_mem;
    void    *synthesis_mem;
    AcStft  *stft;
    AcIstft *istft;
} RnnFraming;

static inline void rnn_framing_config(AcStftConfig *sc, const RNNoiseGrid *grid)
{
    ac_stft_config_default(sc, grid->n_fft);
    sc->spec_scale = grid->stft_scale;
}

static inline int rnn_framing_open(RnnFraming *f, const AcStftConfig *sc)
{
    const size_t a = ac_stft_get_mem_size(sc);
    const size_t s = ac_istft_get_mem_size(sc);
    memset(f, 0, sizeof(*f));
    f->analysis_mem = rnn_test_aligned(a);
    f->synthesis_mem = rnn_test_aligned(s);
    if (f->analysis_mem && f->synthesis_mem) {
        f->stft = ac_stft_init(f->analysis_mem, a, sc);
        f->istft = ac_istft_init(f->synthesis_mem, s, sc);
    }
    return f->stft && f->istft ? 0 : -1;
}

static inline void rnn_framing_close(RnnFraming *f)
{
    free(f->analysis_mem);
    free(f->synthesis_mem);
    memset(f, 0, sizeof(*f));
}

#endif
