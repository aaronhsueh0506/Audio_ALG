/* Smoke build only: does dfn2_stream link and run one hop without crashing. Not a correctness
 * test -- it does not check PCM against a reference; that's the separate verification harness. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "stream_mem.h"   /* stream_saturate_s16(): shared by all three stream modules, checked once here */
#include "erb_fixture.h"

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
static int fake_accelerator(DFN2PrepostOutputs *out) {
    size_t i;
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

/* Runs the full init/hop-loop/reset cycle of the per-hop entries (frames_per_call N == 1). Every hop calls
 * pre_process, calls the fake accelerator only on STREAM_WORK_SUBMIT (as the header's usage example does), then
 * always calls post_process. Cross-checks io.action against the documented cadence (a job on every hop from
 * hop mask_lookahead): a job on the wrong hop, or the wrong number of jobs total, fails the test instead of just
 * not crashing. */
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
    erb_inv = alloc_filled((size_t)grid.n_erb * (size_t)grid.n_bins, 0.0f);
    erb_fixture_unit_inv(erb_inv, grid.n_bins, grid.n_erb);

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

        /* Same cadence as the first window of heads: every hop from hop mask_lookahead. */
        ready_expected = i >= grid.mask_lookahead;
        if ((io.action == STREAM_WORK_SUBMIT) != ready_expected) {
            fprintf(stderr, "action=%d at hop %d, expected %d (N=%d)\n",
                   (int)io.action, i, ready_expected, frames_per_call);
            return 1;
        }

        /* between hops only: refused while this hop awaits its post_process, whether it has a job or not */
        if (dfn2_stream_set_atten_lim(s, 6.0f) != -1) {
            fprintf(stderr, "set_atten_lim accepted inside hop %d (action %d, N=%d)\n",
                   i, (int)io.action, frames_per_call);
            return 1;
        }

        ai_rc = 0;
        if (io.action == STREAM_WORK_SUBMIT) {
            ai_rc = fake_accelerator(&io.outputs);
            ++jobs_run;
        }

        post_rc = dfn2_stream_post_process(s, ai_rc, pcm_out);
        if (post_rc != 0 && post_rc != 1) {
            fprintf(stderr, "post_process rc=%d at hop %d (expected 0 or 1, N=%d)\n",
                   post_rc, i, frames_per_call);
            return 1;
        }
        if (dfn2_stream_set_atten_lim(s, 0.0f) != 0) {
            fprintf(stderr, "set_atten_lim refused between hops (after hop %d, N=%d)\n", i, frames_per_call);
            return 1;
        }
    }

    jobs_expected = hops - grid.mask_lookahead;
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

/* The packet entries (frames_per_call N > 1, mask_lookahead 0): every packet of N hops is one job, the
 * per-hop entries refuse the stream, and a stream with mask_lookahead > 0 is refused at sizing. */
static int run_smoke_packet(int sample_rate, int n_fft, int n_erb, int frames_per_call) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    DFN2Stream *s;
    void *pool = NULL;
    float *erb_fwd, *erb_inv;
    int16_t *pcm_in, *pcm_out;
    int hop, packets, i, k, jobs_run = 0;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = sample_rate;
    spec.n_fft = n_fft;
    spec.n_erb = n_erb;
    spec.df_bins = 96;
    spec.mask_lookahead = 0;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
    spec.df_lookahead = 0;
#else
    spec.df_order = 5;
    spec.df_lookahead = 0;
#endif
    spec.frames_per_call = frames_per_call;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) {
        fprintf(stderr, "dfn2_grid_make failed (%d Hz, N=%d)\n", sample_rate, frames_per_call);
        return 1;
    }
    erb_fwd = alloc_filled((size_t)grid.n_bins * (size_t)grid.n_erb, 1.0f / (float)grid.n_erb);
    erb_inv = alloc_filled((size_t)grid.n_erb * (size_t)grid.n_bins, 0.0f);
    erb_fixture_unit_inv(erb_inv, grid.n_bins, grid.n_erb);
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    if (dfn2_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0) {
        fprintf(stderr, "packet stream sizing or allocation failed (%d Hz, N=%d)\n", sample_rate, frames_per_call);
        return 1;
    }
    s = dfn2_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) {
        fprintf(stderr, "dfn2_stream_init_ex failed (%d Hz, N=%d)\n", sample_rate, frames_per_call);
        return 1;
    }
    hop = dfn2_stream_hop_size(s);
    pcm_in = (int16_t *)calloc((size_t)hop * (size_t)frames_per_call, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)hop * (size_t)frames_per_call, sizeof(int16_t));
    if (!pcm_in || !pcm_out) return 1;

    {   /* the per-hop entries refuse an N > 1 stream and leave it usable */
        DFN2HopIO io;
        if (dfn2_stream_pre_process(s, pcm_in, &io) != -1) {
            fprintf(stderr, "per-hop pre_process accepted an N=%d stream\n", frames_per_call);
            return 1;
        }
    }
    packets = 6;
    for (i = 0; i < packets; ++i) {
        DFN2HopIO io;
        int valid;
        for (k = 0; k < hop * frames_per_call; ++k) pcm_in[k] = (int16_t)((k * 7 + i * 3) % 101 - 50);
        if (dfn2_stream_pre_process_packet(s, pcm_in, &io) != 1 || io.action != STREAM_WORK_SUBMIT) {
            fprintf(stderr, "pre_process_packet: no job at packet %d (N=%d)\n", i, frames_per_call);
            return 1;
        }
        if (dfn2_stream_set_atten_lim(s, 6.0f) != -1) {
            fprintf(stderr, "set_atten_lim accepted inside packet %d (N=%d)\n", i, frames_per_call);
            return 1;
        }
        if (io.inputs.erb_window_elements != (size_t)(frames_per_call + 2) * (size_t)grid.n_erb ||
            io.outputs.erb_mask_elements != (size_t)frames_per_call * (size_t)grid.n_erb) {
            fprintf(stderr, "packet window or head sizes wrong (N=%d)\n", frames_per_call);
            return 1;
        }
        fake_accelerator(&io.outputs);
        ++jobs_run;
        valid = dfn2_stream_post_process_packet(s, 0, pcm_out);
        if (valid != frames_per_call) {   /* L = D = 0: every hop of every packet carries a spectrum */
            fprintf(stderr, "post_process_packet rc=%d at packet %d, expected %d\n", valid, i, frames_per_call);
            return 1;
        }
        if (dfn2_stream_set_atten_lim(s, 0.0f) != 0) {
            fprintf(stderr, "set_atten_lim refused between packets (after packet %d)\n", i);
            return 1;
        }
    }
    if (jobs_run != packets) return 1;
    {   /* a mask_lookahead > 0 N > 1 grid is sized like any other (tests/lookahead_blocks.c runs it) */
        DFN2GridSpec lk = spec;
        DFN2Grid lkg;
        DFN2StreamConfig lkc = cfg;
        DFN2StreamMemReq lkr;
        lk.mask_lookahead = 1;
        if (dfn2_grid_make(&lkg, &lk) != DFN2_GRID_OK) { fprintf(stderr, "grid with lookahead refused\n"); return 1; }
        lkc.grid = lkg;
        if (dfn2_stream_get_mem_size(&lkc, &lkr) != 0 || lkr.bytes == 0u) {
            fprintf(stderr, "mask_lookahead 1 with N=%d refused by the stream\n", frames_per_call);
            return 1;
        }
    }
    printf("smoke_dfn2_stream: packet N=%d (%d Hz), %d packets, %d accelerator jobs, hop=%d -- PASS\n",
           frames_per_call, sample_rate, packets, jobs_run, hop);
    free(pcm_in);
    free(pcm_out);
    free(erb_fwd);
    free(erb_inv);
    free(pool);
    return 0;
}

int main(void) {
    check_saturate_s16_nonfinite();
    if (run_smoke(1) != 0) return 1;   /* a job on every hop from hop mask_lookahead */
    if (run_smoke_packet(16000, 256, 24, 8) != 0) return 1;     /* one job per packet of 8 hops */
    if (run_smoke_packet(48000, 1024, 32, 2) != 0) return 1;
    return 0;
}
