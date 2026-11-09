/* External-consumer stand-in for the variant-guard test (Makefile's test-variant-guard): compiled
 * once per full/-DDFN2_MASK_ONLY, then linked against EITHER variant's prebuilt libdfn2_stream.a
 * by the Makefile, never against this SDK's own .c files directly -- a real out-of-tree caller
 * only ever sees dfn2_stream.h and the archive. A mismatched pairing (this TU's -DDFN2_MASK_ONLY
 * disagreeing with the archive it's linked against) must fail at link time, on the public API's
 * variant-suffixed symbol names (see dfn2_stream.h); this file has no cross-variant logic of its
 * own -- a correctly-paired link must also run a few hops of real audio, not just succeed. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"

static float *alloc_filled(size_t n, float fill) {
    float *p = (float *)malloc(n * sizeof(float));
    size_t i;
    if (!p) { fprintf(stderr, "malloc fail\n"); exit(1); }
    for (i = 0; i < n; ++i) p[i] = fill;
    return p;
}

static int fake_accelerator(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    size_t i;
    (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.3f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.0f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.0f;
#endif
    for (i = 0; i < out->gru_hidden_elements; ++i) out->gru_hidden_next[i] = 0.0f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->df_convp_window_elements; ++i) out->df_convp_window_next[i] = 0.0f;
#endif
    return 0;
}

int main(void) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    DFN2Stream *s;
    void *pool;
    float *erb_fwd, *erb_inv;
    int16_t *pcm_in, *pcm_out;
    int hop, i, k;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 48000;
    spec.n_fft = 1024;
    spec.n_erb = 32;
    spec.df_bins = 96;
    spec.mask_lookahead = 1;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
    spec.df_lookahead = 0;
#else
    spec.df_order = 5;
    spec.df_lookahead = 1;
#endif
    spec.frames_per_call = 1;

    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make fail\n"); return 1; }

    erb_fwd = alloc_filled((size_t)grid.n_bins * (size_t)grid.n_erb, 1.0f / (float)grid.n_erb);
    erb_inv = alloc_filled((size_t)grid.n_erb * (size_t)grid.n_bins, 1.0f);

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;

    if (dfn2_stream_get_mem_size(&cfg, &req) != 0) { fprintf(stderr, "get_mem_size fail\n"); return 1; }
    pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "pool alloc fail\n");
        return 1;
    }
    s = dfn2_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) { fprintf(stderr, "init_ex fail\n"); return 1; }

    hop = dfn2_stream_hop_size(s);
    pcm_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    for (i = 0; i < grid.mask_lookahead + grid.df_lookahead + 4; ++i) {
        DFN2HopIO io;
        int ai_rc;
        for (k = 0; k < hop; ++k) pcm_in[k] = (int16_t)((k * 11 + i * 7) % 151 - 75);
        if (dfn2_stream_pre_process(s, pcm_in, &io) != 1) { fprintf(stderr, "pre fail\n"); return 1; }
        ai_rc = fake_accelerator(&io.inputs, &io.outputs);
        if (dfn2_stream_post_process(s, ai_rc, pcm_out) < 0) { fprintf(stderr, "post fail\n"); return 1; }
    }

    printf("variant_consumer (%s): ran %d hops, generation=%u -- PASS\n",
#ifdef DFN2_MASK_ONLY
          "mask_only",
#else
          "full",
#endif
          grid.mask_lookahead + grid.df_lookahead + 4, dfn2_stream_state_generation(s));

    free(pcm_in); free(pcm_out); free(erb_fwd); free(erb_inv); free(pool);
    return 0;
}
