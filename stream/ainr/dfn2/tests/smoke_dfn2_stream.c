/* Smoke build only: does dfn2_stream link and run one hop without crashing. Not a correctness
 * test -- it does not check PCM against a reference; that's the separate verification harness. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "stream_mem.h"   /* stream_saturate_s16(): shared by all three stream modules, checked once here */

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

/* stream_saturate_s16(NaN) must return 0, not reach the (int16_t) cast of a non-finite float --
 * that cast is undefined behavior (confirmed under -fsanitize=undefined: "nan is outside the
 * range of representable values of type 'short'"), reachable whenever finite-but-extreme
 * accelerator output (e.g. a large mask/CCM tap, accepted since it's finite) overflows to NaN in
 * the DSP between the accelerator and this saturate. +-Inf already clip correctly without this
 * check (they compare true against a clip bound), so they're asserted here too, as a guard
 * against a future "fix" routing them through the same NaN-only early return. */
static void check_saturate_s16_nonfinite(void) {
    if (stream_saturate_s16(NAN) != 0) { fprintf(stderr, "stream_saturate_s16(NAN) != 0\n"); exit(1); }
    if (stream_saturate_s16((float)INFINITY) != 32767) {
        fprintf(stderr, "stream_saturate_s16(+Inf) != 32767\n"); exit(1);
    }
    if (stream_saturate_s16((float)-INFINITY) != -32768) {
        fprintf(stderr, "stream_saturate_s16(-Inf) != -32768\n"); exit(1);
    }
}

/* Runs the full init/hop-loop/reset cycle for one frames_per_call N. Every hop calls
 * pre_process, calls the fake accelerator only when io.job_ready (as the header's usage example
 * does), then always calls post_process -- the one loop shape serves N == 1 and N > 1 alike.
 * Cross-checks io.job_ready against the documented cadence (ready once mask_lookahead + N hops
 * have been pushed, then every N hops): a job on the wrong hop, or the wrong number of jobs
 * total, fails the test instead of just not crashing. */
static int run_smoke(int frames_per_call) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    DFN2Stream *s;
    void *pool;
    float *erb_fwd, *erb_inv;
    int16_t *pcm_in, *pcm_out;
    int hop, hops, i, k, jobs_run, jobs_expected;

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
    spec.frames_per_call = frames_per_call;

    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) {
        fprintf(stderr, "dfn2_grid_make failed (N=%d)\n", frames_per_call);
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
        fprintf(stderr, "dfn2_stream_get_mem_size failed (N=%d)\n", frames_per_call);
        return 1;
    }

    pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "pool allocation failed (%llu bytes, N=%d)\n",
               (unsigned long long)req.bytes, frames_per_call);
        return 1;
    }

    s = dfn2_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) {
        fprintf(stderr, "dfn2_stream_init_ex failed (N=%d)\n", frames_per_call);
        return 1;
    }

    hop = dfn2_stream_hop_size(s);
    if (hop <= 0) {
        fprintf(stderr, "dfn2_stream_hop_size returned %d (N=%d)\n", hop, frames_per_call);
        return 1;
    }
    pcm_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    if (!pcm_in || !pcm_out) {
        fprintf(stderr, "pcm buffer allocation failed (N=%d)\n", frames_per_call);
        return 1;
    }

    hops = grid.mask_lookahead + grid.df_lookahead + 3 * frames_per_call + 8;
    jobs_run = 0;
    for (i = 0; i < hops; ++i) {
        DFN2HopIO io;
        int pre_rc, post_rc, ai_rc, ready_expected;

        for (k = 0; k < hop; ++k) pcm_in[k] = (int16_t)((k * 7 + i * 3) % 101 - 50);

        pre_rc = dfn2_stream_pre_process(s, pcm_in, &io);
        if (pre_rc != 1) {
            fprintf(stderr, "pre_process rc=%d at hop %d (expected 1, N=%d)\n",
                   pre_rc, i, frames_per_call);
            return 1;
        }

        /* Same cadence as dfn2_prepost_pre_process_freq()/AINR/DeepFilterNet2/reference/
         * multiframe.md: N == 1 keeps the old per-hop warm-up flow (ready every hop); N > 1 has
         * no warm-up at all -- ready once mask_lookahead + N hops have been pushed, then every N. */
        ready_expected = frames_per_call == 1 ? 1 :
                        (i + 1 >= grid.mask_lookahead + frames_per_call) &&
                        (i + 1 - grid.mask_lookahead) % frames_per_call == 0;
        if (io.job_ready != ready_expected) {
            fprintf(stderr, "job_ready=%d at hop %d, expected %d (N=%d)\n",
                   io.job_ready, i, ready_expected, frames_per_call);
            return 1;
        }

        ai_rc = 0;
        if (io.job_ready) {
            ai_rc = fake_accelerator(&io.inputs, &io.outputs);
            ++jobs_run;
        }

        post_rc = dfn2_stream_post_process(s, ai_rc, pcm_out);
        if (post_rc != 0 && post_rc != 1) {
            fprintf(stderr, "post_process rc=%d at hop %d (expected 0 or 1, N=%d)\n",
                   post_rc, i, frames_per_call);
            return 1;
        }
    }

    jobs_expected = frames_per_call == 1 ? hops : (hops - grid.mask_lookahead) / frames_per_call;
    if (jobs_run != jobs_expected) {
        fprintf(stderr, "ran %d accelerator jobs over %d hops, expected %d (N=%d)\n",
               jobs_run, hops, jobs_expected, frames_per_call);
        return 1;
    }

    dfn2_stream_reset(s);
    printf("smoke_dfn2_stream: N=%d, %d hops, %d accelerator jobs, hop=%d, generation after reset=%u -- PASS\n",
          frames_per_call, hops, jobs_run, hop, dfn2_stream_state_generation(s));

    free(pcm_in);
    free(pcm_out);
    free(erb_fwd);
    free(erb_inv);
    free(pool);
    return 0;
}

int main(void) {
    check_saturate_s16_nonfinite();
    if (run_smoke(1) != 0) return 1;   /* every hop is a job, as before frames_per_call was accepted */
    if (run_smoke(4) != 0) return 1;   /* one job every 4 hops */
    return 0;
}
