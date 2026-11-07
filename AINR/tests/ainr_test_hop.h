/* Hop-in / hop-out framing for the AINR test programs (header-only, test use): AcStft ->
 * a frequency-domain library -> AcIstft, built the way the model headers state the framing
 * contract: audiolib's sqrt-Hann window, centre 0, hop = n_fft / 2, overlap-add accumulate.
 * The scales are the family's:
 *
 *   DeepFilterNet2 / 3   spec_scale = grid.stft_scale (1/sqrt(n_fft)), ispec_scale = grid.stft_unscale
 *   GTCRN                1.0f, 1.0f (unnormalised spectra, interleaved [n_bins][2])
 *
 * One object per stream; the objects live in aligned blocks from aligned_alloc(). */

#ifndef AINR_TEST_HOP_H
#define AINR_TEST_HOP_H

#include <stdlib.h>

#include "audio_stft.h"

typedef struct AinrTestHop {
    AcStft  *stft;
    AcIstft *istft;
    void    *stft_mem;
    void    *istft_mem;
} AinrTestHop;

static inline void *ainr_test_hop_block(size_t bytes)
{
    return aligned_alloc(16, (bytes + 15u) & ~(size_t)15u);
}

/* 0, or -1 for an n_fft audiolib refuses or a failed allocation (the object is then closed). */
static inline int ainr_test_hop_open(AinrTestHop *h, int n_fft, float spec_scale,
                                     float ispec_scale)
{
    AcStftConfig cfg;
    size_t analysis_bytes, synthesis_bytes;
    h->stft = NULL;
    h->istft = NULL;
    h->stft_mem = NULL;
    h->istft_mem = NULL;
    ac_stft_config_default(&cfg, n_fft);
    cfg.spec_scale = spec_scale;
    cfg.ispec_scale = ispec_scale;
    analysis_bytes = ac_stft_get_mem_size(&cfg);
    synthesis_bytes = ac_istft_get_mem_size(&cfg);
    if (analysis_bytes == 0u || synthesis_bytes == 0u) return -1;
    h->stft_mem = ainr_test_hop_block(analysis_bytes);
    h->istft_mem = ainr_test_hop_block(synthesis_bytes);
    if (h->stft_mem == NULL || h->istft_mem == NULL) {
        free(h->stft_mem);
        free(h->istft_mem);
        h->stft_mem = NULL;
        h->istft_mem = NULL;
        return -1;
    }
    h->stft = ac_stft_init(h->stft_mem, analysis_bytes, &cfg);
    h->istft = ac_istft_init(h->istft_mem, synthesis_bytes, &cfg);
    return h->stft != NULL && h->istft != NULL ? 0 : -1;
}

static inline void ainr_test_hop_close(AinrTestHop *h)
{
    free(h->stft_mem);
    free(h->istft_mem);
    h->stft_mem = NULL;
    h->istft_mem = NULL;
    h->stft = NULL;
    h->istft = NULL;
}

/* One hop of samples in, one spectrum out (planar re/im, n_bins each). */
static inline void ainr_test_hop_analysis(AinrTestHop *h, const float *in_hop, float *re,
                                          float *im)
{
    ac_stft_push_planar(h->stft, in_hop, re, im);
}

/* One spectrum in, one hop of samples out. */
static inline void ainr_test_hop_synthesis(AinrTestHop *h, const float *re, const float *im,
                                           float *out_hop)
{
    ac_istft_push_planar(h->istft, re, im, out_hop);
}

/* The interleaved [n_bins][2] twins, the layout GTCRN takes. */
static inline void ainr_test_hop_analysis_il(AinrTestHop *h, const float *in_hop,
                                             float *spec)
{
    ac_stft_push(h->stft, in_hop, (Complex *)(void *)spec);
}

static inline void ainr_test_hop_synthesis_il(AinrTestHop *h, const float *spec,
                                              float *out_hop)
{
    ac_istft_push(h->istft, (const Complex *)(const void *)spec, out_hop);
}

#endif /* AINR_TEST_HOP_H */
