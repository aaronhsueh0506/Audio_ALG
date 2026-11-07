/* Smoke build only: does dfn2_stream link and run one hop without crashing. Not a correctness
 * test -- it does not check PCM against a reference; that's the separate verification harness. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"

static float *alloc_filled(size_t n, float fill) {
    float *p = (float *)malloc(n * sizeof(float));
    size_t i;
    if (!p) {
        fprintf(stderr, "malloc fail\n");
        exit(1);
    }
    for (i = 0; i < n; ++i) p[i] = fill;
    return p;
}

/* Writes zeros/success, every output element -- shape correctness only, no numerical realism. */
static int fake_accelerator(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    size_t i;
    (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.0f;
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
    int hop, hops, i, k;

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
    spec.erb_norm_tau_sec = 1.0f;
    spec.erb_norm_init_lo_db = -60.0f;
    spec.erb_norm_init_hi_db = -90.0f;
    spec.erb_norm_scale_db = 40.0f;
    spec.spec_norm_tau_sec = 1.0f;
    spec.spec_norm_init_lo = 0.001f;
    spec.spec_norm_init_hi = 0.0001f;

    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) {
        fprintf(stderr, "dfn2_grid_make failed\n");
        return 1;
    }

    erb_fwd = alloc_filled((size_t)grid.n_bins * (size_t)grid.n_erb, 1.0f / (float)grid.n_erb);
    erb_inv = alloc_filled((size_t)grid.n_erb * (size_t)grid.n_bins, 1.0f);

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    cfg.atten_lim_db = 0.0f;
    cfg.hpf_enable = 0;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;

    if (dfn2_stream_get_mem_size(&cfg, &req) != 0) {
        fprintf(stderr, "dfn2_stream_get_mem_size failed\n");
        return 1;
    }

    pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "pool allocation failed (%llu bytes)\n", (unsigned long long)req.bytes);
        return 1;
    }

    s = dfn2_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) {
        fprintf(stderr, "dfn2_stream_init_ex failed\n");
        return 1;
    }

    hop = dfn2_stream_hop_size(s);
    if (hop <= 0) {
        fprintf(stderr, "dfn2_stream_hop_size returned %d\n", hop);
        return 1;
    }
    pcm_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    if (!pcm_in || !pcm_out) {
        fprintf(stderr, "pcm buffer allocation failed\n");
        return 1;
    }

    hops = grid.mask_lookahead + grid.df_lookahead + 8;
    for (i = 0; i < hops; ++i) {
        DFN2HopIO io;
        int pre_rc, post_rc, ai_rc;

        for (k = 0; k < hop; ++k) pcm_in[k] = (int16_t)((k * 7 + i * 3) % 101 - 50);

        pre_rc = dfn2_stream_pre_process(s, pcm_in, &io);
        if (pre_rc != 1) {
            fprintf(stderr, "pre_process rc=%d at hop %d (expected 1)\n", pre_rc, i);
            return 1;
        }

        ai_rc = fake_accelerator(&io.inputs, &io.outputs);

        post_rc = dfn2_stream_post_process(s, ai_rc, pcm_out);
        if (post_rc != 0 && post_rc != 1) {
            fprintf(stderr, "post_process rc=%d at hop %d (expected 0 or 1)\n", post_rc, i);
            return 1;
        }
    }

    dfn2_stream_reset(s);
    printf("smoke_dfn2_stream: %d hops, hop=%d, generation after reset=%u -- PASS\n",
          hops, hop, dfn2_stream_state_generation(s));

    free(pcm_in);
    free(pcm_out);
    free(erb_fwd);
    free(erb_inv);
    free(pool);
    return 0;
}
