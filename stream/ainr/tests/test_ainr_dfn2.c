/* Host-only regression suite for ainr.h/ainr.c against a stand-in accelerator: ainr_preprocess()/postprocess() versus the
 * same scenario driven through dfn2_stream_pre_process()/post_process(), plus the error paths (init validation order,
 * post-process failure, sequence misuse, pool size). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr.h"
#include "ainr_product.h"
#include "ainr_test_fixture.h"
#include "dfn2_stream.h"
#include "graph_io_dfn2.h"

/* ---- deterministic stand-in accelerator, same fill on both the ainr path and the reference
 * path driven directly through dfn2_stream -- the fill values themselves are arbitrary, what
 * matters is that both paths get exactly the same ones. ---- */

static int raw_fill_accel(DFN2PrepostOutputs *out) {
    size_t i;
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

static void ainr_fill_accel(StreamWork *io) {
    size_t i;
    for (i = 0; i < io->outputs[0].elements; ++i) io->outputs[0].data[i] = 0.3f;    /* erb_mask */
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < io->outputs[1].elements; ++i) io->outputs[1].data[i] = 0.0f;    /* coefs */
    for (i = 0; i < io->outputs[2].elements; ++i) io->outputs[2].data[i] = 0.0f;    /* alpha */
#endif
}

/* ---- the two catalog grids, duplicated here (not #include-able from ainr.c) so the reference
 * path can be driven independently through dfn2_stream_* with the same numbers. ---- */

static DFN2Grid make_grid_16k_256(int frames_per_call) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000;
    spec.n_fft = 256;
    spec.n_erb = 24;
    spec.df_bins = 96;
    spec.df_order = 5;
    spec.mask_lookahead = 0;
    spec.df_lookahead = 0;
    spec.frames_per_call = frames_per_call;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make fail\n"); exit(1); }
    return grid;
}

static void run_reference_hops(DFN2Stream *s, const int16_t *pcm_in, int hops, int hop, int16_t *pcm_out) {
    int i;
    for (i = 0; i < hops; ++i) {
        DFN2HopIO io;
        int ai_rc = 0;
        if (dfn2_stream_pre_process(s, pcm_in + (size_t)i * (size_t)hop, &io) != 1) {
            fprintf(stderr, "reference pre_process fail at hop %d\n", i);
            exit(1);
        }
        if (io.action == STREAM_WORK_SUBMIT) ai_rc = raw_fill_accel(&io.outputs);
        if (dfn2_stream_post_process(s, ai_rc, pcm_out + (size_t)i * (size_t)hop) < 0) {
            fprintf(stderr, "reference post_process fail at hop %d\n", i);
            exit(1);
        }
    }
}

static int run_ainr_blocks(AinrCtx *ctx, const int16_t *pcm_in, int blocks, int hop_in, int16_t *pcm_out) {
    int i;
    for (i = 0; i < blocks; ++i) {
        StreamWork io;
        int rc = ainr_preprocess(ctx, pcm_in + (size_t)i * (size_t)hop_in, &io);
        if (rc != AINR_OK) { fprintf(stderr, "ainr_preprocess rc=%d at block %d\n", rc, i); return -1; }
        ainr_fill_accel(&io);
        rc = ainr_postprocess(ctx, 0, pcm_out + (size_t)i * (size_t)hop_in);
        if (rc != AINR_OK) { fprintf(stderr, "ainr_postprocess rc=%d at block %d\n", rc, i); return -1; }
    }
    return 0;
}

static AinrInfo g_info;

/* 1. N=1 catalog entry: ainr_* must be byte-identical to driving dfn2_stream_* directly. */
static int test_n1_matches_reference(const ErbTables *erb) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    DFN2StreamConfig scfg;
    DFN2StreamMemReq sreq;
    void *ainr_pool, *ref_pool;
    AinrCtx *ctx;
    DFN2Stream *ref;
    int hop = grid.hop, hops = 20;
    int16_t *pcm_in, *ainr_out, *ref_out;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;   /* N = 1 */

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    ainr_pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_t(&cfg, erb, ainr_pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init fail\n"); return 1; }

    memset(&scfg, 0, sizeof(scfg));
    scfg.grid = grid;
    scfg.erb_fwd = erb->fwd;
    scfg.erb_inv = erb->inv;
    if (dfn2_stream_get_mem_size(&scfg, &sreq) != 0) { fprintf(stderr, "dfn2_stream_get_mem_size fail\n"); return 1; }
    ref_pool = alloc_aligned((size_t)sreq.bytes, sreq.alignment);
    ref = dfn2_stream_init(&scfg, ref_pool, (size_t)sreq.bytes);
    if (!ref) { fprintf(stderr, "dfn2_stream_init fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    ainr_out = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    ref_out = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, hops * hop, 1);

    if (run_ainr_blocks(ctx, pcm_in, hops, hop, ainr_out) != 0) return 1;
    run_reference_hops(ref, pcm_in, hops, hop, ref_out);

    if (memcmp(ainr_out, ref_out, (size_t)hops * (size_t)hop * sizeof(int16_t)) != 0) {
        fprintf(stderr, "test_n1_matches_reference: PCM mismatch\n");
        return 1;
    }
    printf("test_n1_matches_reference: PASS\n");
    free(pcm_in); free(ainr_out); free(ref_out); free(ainr_pool); free(ref_pool);
    return 0;
}

/* 2 & 3. N=8, mask_lookahead=0, df_lookahead=0: one packet job per block, and the PCM is the N=1 stream's, hop
 * for hop with no extra delay. The reference is the same grid at N=1 driven per hop (this stand-in accelerator's
 * heads do not depend on the window, so both reach the same PCM). Two blocks, the first carrying a non-silent
 * signal and the second all zero: block 1 already carries the signal (one hop of framing delay, none for the
 * packet), and the tail of its last hop lands in block 2's first hop. */
static int test_n8_two_blocks_matches_reference(const ErbTables *erb) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    DFN2StreamConfig scfg;
    DFN2StreamMemReq sreq;
    void *ainr_pool, *ref_pool;
    AinrCtx *ctx;
    DFN2Stream *ref;
    int hop = grid.hop, blocks = 2, hops = blocks * 8;
    int hop_in = 8 * hop;
    int16_t *pcm_in, *ainr_out, *ref_out;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop_in;   /* N = 8 */

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    ainr_pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_t(&cfg, erb, ainr_pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init fail\n"); return 1; }

    memset(&scfg, 0, sizeof(scfg));
    scfg.grid = grid;
    scfg.erb_fwd = erb->fwd;
    scfg.erb_inv = erb->inv;
    if (dfn2_stream_get_mem_size(&scfg, &sreq) != 0) { fprintf(stderr, "dfn2_stream_get_mem_size fail\n"); return 1; }
    ref_pool = alloc_aligned((size_t)sreq.bytes, sreq.alignment);
    ref = dfn2_stream_init(&scfg, ref_pool, (size_t)sreq.bytes);
    if (!ref) { fprintf(stderr, "dfn2_stream_init fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    ainr_out = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    ref_out = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, 8 * hop, 2);                                  /* block 1: non-silent */
    memset(pcm_in + (size_t)8 * hop, 0, (size_t)8 * hop * sizeof(int16_t));  /* block 2: silence */

    if (run_ainr_blocks(ctx, pcm_in, blocks, hop_in, ainr_out) != 0) return 1;
    run_reference_hops(ref, pcm_in, hops, hop, ref_out);

    if (memcmp(ainr_out, ref_out, (size_t)hops * (size_t)hop * sizeof(int16_t)) != 0) {
        fprintf(stderr, "test_n8_two_blocks_matches_reference: PCM differs from the N=1 stream\n");
        return 1;
    }
    {
        int16_t *block1 = ainr_out, *block2 = ainr_out + (size_t)hop_in;
        int silent1 = 1, silent2_head = 1, i;
        for (i = hop; i < hop_in; ++i) if (block1[i] != 0) { silent1 = 0; break; }
        for (i = 0; i < hop; ++i) if (block2[i] != 0) { silent2_head = 0; break; }
        if (silent1 || silent2_head) {
            fprintf(stderr, "test_n8_two_blocks_matches_reference: block 1 %s, the first hop of block 2 %s\n",
                    silent1 ? "is silent" : "carries signal", silent2_head ? "is silent" : "carries signal");
            return 1;
        }
    }
    printf("test_n8_two_blocks_matches_reference: PASS\n");
    free(pcm_in); free(ainr_out); free(ref_out); free(ainr_pool); free(ref_pool);
    return 0;
}

/* 4. Validation ordering: grid, then hop_in, then N. Every N in 1..DFN2_MODEL_MAX_FRAMES_PER_CALL is accepted at every
 * grid of the bound model (which N has a graph is the sample's graph_io table, not the library's); one more is
 * AINR_ERR_FRAMES_PER_CALL. */
static int test_validation_ordering(void) {
    static const struct { int sr, fft; } GRIDS[] = { { 16000, 256 }, { 48000, 1024 } };
    AinrConfig cfg = {0};
    StreamMemReq req;
    size_t g;
    int n, status;

    for (g = 0; g < sizeof(GRIDS) / sizeof(GRIDS[0]); ++g) {
        const int hop = GRIDS[g].fft / 2;
        cfg.sr = GRIDS[g].sr;
        cfg.fft = GRIDS[g].fft;
        for (n = 1; n <= 64; ++n) {
            cfg.hop_in = n * hop;
            status = ainr_get_mem_size(&cfg, &req);
            if (status != AINR_OK) {
                fprintf(stderr, "test_validation_ordering: %d Hz N=%d -> %d (%s), expected AINR_OK\n", cfg.sr, n, status,
                        ainr_status_text(status));
                return 1;
            }
        }
        cfg.hop_in = 65 * hop;
        status = ainr_get_mem_size(&cfg, &req);
        if (status != AINR_ERR_FRAMES_PER_CALL) {
            fprintf(stderr, "test_validation_ordering: %d Hz N=65 -> %d (%s), expected AINR_ERR_FRAMES_PER_CALL\n", cfg.sr,
                    status, ainr_status_text(status));
            return 1;
        }
        cfg.hop_in = 65 * hop + 1;   /* hop_in is checked before N */
        status = ainr_get_mem_size(&cfg, &req);
        if (status != AINR_ERR_IN_HOP) {
            fprintf(stderr, "test_validation_ordering: %d Hz hop_in not a multiple -> %d, expected AINR_ERR_IN_HOP\n",
                    cfg.sr, status);
            return 1;
        }
        cfg.hop_in = 0;
        if (ainr_get_mem_size(&cfg, &req) != AINR_ERR_IN_HOP) {
            fprintf(stderr, "test_validation_ordering: %d Hz hop_in 0 not AINR_ERR_IN_HOP\n", cfg.sr);
            return 1;
        }
    }

    cfg.sr = 22000; cfg.fft = 333; cfg.hop_in = 65 * 166;   /* the grid is checked before hop_in and N */
    status = ainr_get_mem_size(&cfg, &req);
    if (status != AINR_ERR_GRID) {
        fprintf(stderr, "test_validation_ordering: unknown grid -> %d, expected AINR_ERR_GRID\n", status);
        return 1;
    }

    printf("test_validation_ordering: PASS\n");
    return 0;
}

/* 5. A run that did not succeed (run_status != 0) still closes the block and leaves the next one
 * able to proceed normally -- ainr_postprocess's return value is a contract-success flag, not a
 * pass-through of the model's internal spectral-validity bit. A failed run moves the state
 * generation by one; STREAM_RUN_NOT_STARTED leaves it. */
static int test_postprocess_failure_recovery(const ErbTables *erb) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    void *pool;
    AinrCtx *ctx;
    int hop = grid.hop;
    int16_t *pcm_in, *pcm_out;
    StreamWork io;
    int rc;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_t(&cfg, erb, pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    pcm_out = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, hop, 3);

    {
        unsigned int gen_before = ainr_state_generation(ctx);
        int all_zero = 1, i, pass;

        /* The ERB table is a partition of unity, so a bypassed hop is an exact pass-through and the
         * first output hop is the (zero) history half of the overlap-add; the second hop carries the
         * first one's samples. */
        for (pass = 0; pass < 2; ++pass) {
            rc = ainr_preprocess(ctx, pcm_in, &io);
            if (rc != AINR_OK) { fprintf(stderr, "test_postprocess_failure_recovery: preprocess rc=%d\n", rc); return 1; }
            ainr_fill_accel(&io);   /* the accelerator "ran" successfully, but report failure anyway */
            rc = ainr_postprocess(ctx, -1 /* simulated bind/infer/readback failure */, pcm_out);
            if (rc != AINR_OK) {
                fprintf(stderr, "test_postprocess_failure_recovery: postprocess after failure rc=%d, "
                                "expected AINR_OK (PCM still written)\n", rc);
                return 1;
            }
        }
        /* Bypass is a pass-through, not silence -- a nonzero input must come back nonzero, not a
         * zeroed block (this is what ainr.h's doc comment claims). */
        for (i = 0; i < hop; ++i) { if (pcm_out[i] != 0) { all_zero = 0; break; } }
        if (all_zero) {
            fprintf(stderr, "test_postprocess_failure_recovery: bypassed block came back all-zero "
                            "(expected pass-through, not silence)\n");
            return 1;
        }
        /* Each failed run (status -1) advanced the generation by one; the library zeroes its side and tells the
         * sample to reload the device state. */
        if (ainr_state_generation(ctx) != gen_before + 2u) {
            fprintf(stderr, "test_postprocess_failure_recovery: two failed runs moved state_generation "
                            "%u -> %u, want +2\n", gen_before, ainr_state_generation(ctx));
            return 1;
        }
        /* A run that never started touches nothing. */
        gen_before = ainr_state_generation(ctx);
        rc = ainr_preprocess(ctx, pcm_in, &io);
        if (rc != AINR_OK) { fprintf(stderr, "test_postprocess_failure_recovery: preprocess rc=%d\n", rc); return 1; }
        rc = ainr_postprocess(ctx, STREAM_RUN_NOT_STARTED, pcm_out);
        if (rc != AINR_OK || ainr_state_generation(ctx) != gen_before) {
            fprintf(stderr, "test_postprocess_failure_recovery: STREAM_RUN_NOT_STARTED rc=%d, generation %u -> %u\n",
                    rc, gen_before, ainr_state_generation(ctx));
            return 1;
        }
    }

    /* Next block must proceed normally -- nothing left stuck open. */
    rc = ainr_preprocess(ctx, pcm_in, &io);
    if (rc != AINR_OK) { fprintf(stderr, "test_postprocess_failure_recovery: next block preprocess rc=%d\n", rc); return 1; }
    ainr_fill_accel(&io);
    rc = ainr_postprocess(ctx, 0, pcm_out);
    if (rc != AINR_OK) { fprintf(stderr, "test_postprocess_failure_recovery: next block postprocess rc=%d\n", rc); return 1; }

    printf("test_postprocess_failure_recovery: PASS\n");
    free(pcm_in); free(pcm_out); free(pool);
    return 0;
}

/* 6. Sequence misuse: a second preprocess without a matching postprocess must be refused; reset
 * must clear the stuck state so a fresh block works afterwards. */
static int test_sequence_misuse_and_reset(const ErbTables *erb) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    void *pool;
    AinrCtx *ctx;
    int hop = grid.hop;
    int16_t *pcm_in, *pcm_out;
    StreamWork io;
    int rc;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_t(&cfg, erb, pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    pcm_out = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, hop, 4);

    rc = ainr_preprocess(ctx, pcm_in, &io);
    if (rc != AINR_OK) { fprintf(stderr, "test_sequence_misuse_and_reset: first preprocess rc=%d\n", rc); return 1; }
    rc = ainr_preprocess(ctx, pcm_in, &io);   /* no matching postprocess yet -- must be refused */
    if (rc != AINR_ERR_SEQUENCE) {
        fprintf(stderr, "test_sequence_misuse_and_reset: second preprocess rc=%d, expected AINR_ERR_SEQUENCE\n", rc);
        return 1;
    }

    ainr_reset(ctx);   /* must clear the stuck "block_open" state */
    rc = ainr_preprocess(ctx, pcm_in, &io);
    if (rc != AINR_OK) { fprintf(stderr, "test_sequence_misuse_and_reset: preprocess after reset rc=%d\n", rc); return 1; }
    ainr_fill_accel(&io);
    rc = ainr_postprocess(ctx, 0, pcm_out);
    if (rc != AINR_OK) { fprintf(stderr, "test_sequence_misuse_and_reset: postprocess after reset rc=%d\n", rc); return 1; }

    printf("test_sequence_misuse_and_reset: PASS\n");
    free(pcm_in); free(pcm_out); free(pool);
    return 0;
}

/* 8. A pool larger than the config needs is accepted: the init writes only the bytes
 * ainr_get_mem_size() reported (the rest keeps its content) and the stream behaves byte-identically
 * to one on an exact pool. */
static int test_larger_pool_accepted(const ErbTables *erb) {
    AinrConfig cfg_big = {0}, cfg_small = {0};
    StreamMemReq req_big, req_small;
    unsigned char *pool_big, *pool_exact;
    AinrCtx *ctx_big = NULL, *ctx_exact = NULL;
    int16_t pcm_in[128 * 4], out_big[128 * 4], out_exact[128 * 4];
    size_t i;
    int b;

    cfg_big.sr = 16000;
    cfg_big.fft = 256;
    cfg_big.hop_in = 8 * 128;   /* N = 8: the block staging, bigger pool */

    cfg_small.sr = 16000;
    cfg_small.fft = 256;
    cfg_small.hop_in = 128;     /* N = 1: smaller pool */

    if (ainr_get_mem_size(&cfg_big, &req_big) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size(big) fail\n"); return 1; }
    if (ainr_get_mem_size(&cfg_small, &req_small) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size(small) fail\n"); return 1; }
    if (req_big.bytes <= req_small.bytes) {
        fprintf(stderr, "test_larger_pool_accepted: N=8 pool (%llu) is not bigger than N=1 pool (%llu) -- "
                        "test assumption broken\n",
               (unsigned long long)req_big.bytes, (unsigned long long)req_small.bytes);
        return 1;
    }

    pool_big = (unsigned char *)alloc_aligned((size_t)req_big.bytes, req_big.alignment);
    pool_exact = (unsigned char *)alloc_aligned((size_t)req_small.bytes, req_small.alignment);
    memset(pool_big, 0xA5, (size_t)req_big.bytes);
    if (ainr_init_t(&cfg_small, erb, pool_big, (size_t)req_big.bytes, &ctx_big, &g_info) != AINR_OK || !ctx_big ||
        ainr_init_t(&cfg_small, erb, pool_exact, (size_t)req_small.bytes, &ctx_exact, &g_info) != AINR_OK || !ctx_exact) {
        fprintf(stderr, "test_larger_pool_accepted: init refused (big pool %d)\n", ctx_big != NULL);
        return 1;
    }
    for (i = (size_t)req_small.bytes; i < (size_t)req_big.bytes; ++i) {
        if (pool_big[i] != 0xA5) {
            fprintf(stderr, "test_larger_pool_accepted: byte %zu beyond the reported size was written\n", i);
            return 1;
        }
    }
    fill_pcm(pcm_in, 128 * 4, 3);
    for (b = 0; b < 4; ++b) {
        StreamWork io;
        if (ainr_preprocess(ctx_big, pcm_in + 128 * b, &io) != AINR_OK) return 1;
        ainr_fill_accel(&io);
        if (ainr_postprocess(ctx_big, 0, out_big + 128 * b) != AINR_OK) return 1;
        if (ainr_preprocess(ctx_exact, pcm_in + 128 * b, &io) != AINR_OK) return 1;
        ainr_fill_accel(&io);
        if (ainr_postprocess(ctx_exact, 0, out_exact + 128 * b) != AINR_OK) return 1;
    }
    if (memcmp(out_big, out_exact, sizeof(out_big)) != 0) {
        fprintf(stderr, "test_larger_pool_accepted: PCM differs between the larger and the exact pool\n");
        return 1;
    }

    printf("test_larger_pool_accepted: PASS\n");
    free(pool_big); free(pool_exact);
    return 0;
}

/* StreamWork's shapes come from the grid math while its data pointers and element counts come from the
 * authoritative DFN2Prepost views -- the one place those two could silently disagree. Every shape field
 * of a real block is checked against the exporter's tensor shapes spelled out for the 16k/256 N=1 grid
 * (graph inputs erb [1,1,T,24], spec [1,2,T,96] with T = N + 2; heads erb_mask [1,1,1,24], full also
 * df_coefs [1,1,96,10] and df_alpha [1,1,1,1]), and element counts against the product of the axes. */
typedef struct WorkShape { unsigned int b, c, h, w; } WorkShape;

static int test_work_matches_exporter_shapes(const ErbTables *erb) {
    static const WorkShape want_in[2] = { { 1, 1, 3, 24 }, { 1, 2, 3, 96 } };
#ifdef DFN2_MASK_ONLY
    static const WorkShape want_out[1] = { { 1, 1, 1, 24 } };
#else
    static const WorkShape want_out[3] = { { 1, 1, 1, 24 }, { 1, 1, 96, 10 }, { 1, 1, 1, 1 } };
#endif
    const int want_out_count = (int)(sizeof(want_out) / sizeof(want_out[0]));
    AinrConfig cfg = {0};
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    void *pool;
    AinrCtx *ctx;
    int hop = grid.hop;
    int16_t *pcm_in, *pcm_out;
    AinrInfo info;
    StreamWork real_io;
    int rc, i;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_t(&cfg, erb, pool, (size_t)req.bytes, &ctx, &info);
    if (!ctx) { fprintf(stderr, "ainr_init fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    pcm_out = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, hop, 5);

    rc = ainr_preprocess(ctx, pcm_in, &real_io);
    if (rc != AINR_OK) { fprintf(stderr, "test_work_matches_exporter_shapes: preprocess rc=%d\n", rc); return 1; }

    if (real_io.input_count != 2 || real_io.output_count != want_out_count) {
        fprintf(stderr, "test_work_matches_exporter_shapes: counts %d/%d, want 2/%d\n",
                real_io.input_count, real_io.output_count, want_out_count);
        return 1;
    }
    for (i = 0; i < real_io.input_count; ++i) {
        const StreamTensorIn *t = &real_io.inputs[i];
        if (t->b != want_in[i].b || t->c != want_in[i].c || t->h != want_in[i].h || t->w != want_in[i].w ||
            t->elements != (size_t)t->b * t->c * t->h * t->w || !t->data) {
            fprintf(stderr, "test_work_matches_exporter_shapes: inputs[%d] shape mismatch\n", i);
            return 1;
        }
    }
    for (i = 0; i < real_io.output_count; ++i) {
        const StreamTensorOut *t = &real_io.outputs[i];
        if (t->b != want_out[i].b || t->c != want_out[i].c || t->h != want_out[i].h || t->w != want_out[i].w ||
            t->elements != (size_t)t->b * t->c * t->h * t->w || !t->data) {
            fprintf(stderr, "test_work_matches_exporter_shapes: outputs[%d] shape mismatch\n", i);
            return 1;
        }
    }

    ainr_fill_accel(&real_io);
    rc = ainr_postprocess(ctx, 0, pcm_out);
    if (rc != AINR_OK) { fprintf(stderr, "test_work_matches_exporter_shapes: postprocess rc=%d\n", rc); return 1; }

    printf("test_work_matches_exporter_shapes: PASS\n");
    free(pcm_in); free(pcm_out); free(pool);
    return 0;
}

/* 9. The init info and the graph_io table: mask_lookahead is the grid's (16 kHz 0, 48 kHz 1) and, with (sr, fft, N), the
 * key of the sample's graph row. The library accepts any N; a config without a row (16 kHz N=3, 48 kHz N=4) is the
 * sample's refusal, GRAPH_IO_ERR_NO_ROW. N=65 and an unknown grid leave ctx NULL and info untouched. */
static int init_for(const AinrConfig *cfg, const ErbTables *erb, AinrCtx **ctx, AinrInfo *info) {
    _Alignas(16) static unsigned char pool[1 << 21];   /* larger than any config below needs */
    return ainr_init_t(cfg, erb, pool, sizeof(pool), ctx, info);
}

static int has_row(const AinrConfig *cfg, const AinrInfo *info) {
    const int n = graph_io_frames(cfg->fft, cfg->hop_in);
    return graph_io_dfn2_lookup(GRAPH_IO_DFN2_FULL, cfg->sr, cfg->fft, n, info->mask_lookahead) != NULL;
}

static int test_info_and_graph_rows(const ErbTables *erb) {
    AinrConfig cfg = {0};
    AinrInfo bi, bi_n8, sentinel;
    const GraphIoDfn2Row *r1, *r8;
    AinrCtx *ctx = NULL;

    cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = 128;   /* N = 1 */
    if (init_for(&cfg, erb, &ctx, &bi) != AINR_OK || !ctx || bi.mask_lookahead != 0 || bi.queue_frames != 0 ||
        !(r1 = graph_io_dfn2_lookup(GRAPH_IO_DFN2_FULL, 16000, 256, 1, bi.mask_lookahead))) {
        fprintf(stderr, "test_info_and_graph_rows: 16 kHz N=1: mask_lookahead %d, queue %d or no row\n", bi.mask_lookahead,
                bi.queue_frames);
        return 1;
    }

    cfg.hop_in = 8 * 128;   /* N = 8 */
    if (init_for(&cfg, erb, &ctx, &bi_n8) != AINR_OK || !ctx || bi_n8.mask_lookahead != 0 ||
        !(r8 = graph_io_dfn2_lookup(GRAPH_IO_DFN2_FULL, 16000, 256, 8, bi_n8.mask_lookahead)) ||
        strcmp(r1->row.file, r8->row.file) == 0) {
        fprintf(stderr, "test_info_and_graph_rows: N=1 and N=8 did not resolve to distinct graph rows\n");
        return 1;
    }

    cfg.hop_in = 3 * 128;   /* N = 3: accepted, no graph row */
    if (init_for(&cfg, erb, &ctx, &bi_n8) != AINR_OK || !ctx || has_row(&cfg, &bi_n8)) {
        fprintf(stderr, "test_info_and_graph_rows: N=3 expected an accepted init and no graph row\n");
        return 1;
    }
    {
        GraphIoConfig gc;
        if (graph_io_dfn2_find(GRAPH_IO_DFN2_FULL, 16000, 256, 3, bi_n8.mask_lookahead, &gc) != GRAPH_IO_ERR_NO_ROW) {
            fprintf(stderr, "test_info_and_graph_rows: N=3 expected GRAPH_IO_ERR_NO_ROW\n");
            return 1;
        }
    }

    memset(&sentinel, 0x5a, sizeof(sentinel));
    bi_n8 = sentinel;
    ctx = NULL;
    cfg.hop_in = 65 * 128;   /* N = 65 */
    if (init_for(&cfg, erb, &ctx, &bi_n8) != AINR_ERR_FRAMES_PER_CALL || ctx != NULL ||
        memcmp(&bi_n8, &sentinel, sizeof(bi_n8)) != 0) {
        fprintf(stderr, "test_info_and_graph_rows: N=65 expected AINR_ERR_FRAMES_PER_CALL, ctx NULL and info untouched\n");
        return 1;
    }

    bi_n8 = sentinel;
    cfg.sr = 22000; cfg.fft = 333; cfg.hop_in = 166;   /* unknown grid */
    if (init_for(&cfg, erb, &ctx, &bi_n8) != AINR_ERR_GRID || ctx != NULL ||
        memcmp(&bi_n8, &sentinel, sizeof(bi_n8)) != 0) {
        fprintf(stderr, "test_info_and_graph_rows: unknown grid expected AINR_ERR_GRID, ctx NULL and info untouched\n");
        return 1;
    }

#if !defined(DFN2_MASK_ONLY) && AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_FULL
    {   /* the 48 kHz grid (mask_lookahead 1) has graph rows for N = 1 and N = 2, each its own; N = 4 is accepted without one */
        ErbTables erb48 = erb_tables_synthetic(513, 32);
        AinrInfo n1, n2, n4;
        int rc1, rc2, rc4;
        cfg.sr = 48000; cfg.fft = 1024; cfg.hop_in = 512;
        rc1 = init_for(&cfg, &erb48, &ctx, &n1);
        cfg.hop_in = 1024;
        rc2 = init_for(&cfg, &erb48, &ctx, &n2);
        cfg.hop_in = 2048;
        rc4 = init_for(&cfg, &erb48, &ctx, &n4);
        erb_tables_free(&erb48);
        if (rc1 != AINR_OK || rc2 != AINR_OK || n1.mask_lookahead != 1 || n2.mask_lookahead != 1 || n1.queue_frames != 2 ||
            n2.queue_frames != 2 ||
            !(r1 = graph_io_dfn2_lookup(GRAPH_IO_DFN2_FULL, 48000, 1024, 1, n1.mask_lookahead)) ||
            !(r8 = graph_io_dfn2_lookup(GRAPH_IO_DFN2_FULL, 48000, 1024, 2, n2.mask_lookahead)) ||
            strcmp(r1->row.file, r8->row.file) == 0) {
            fprintf(stderr, "test_info_and_graph_rows: 48 kHz N=1 / N=2 (%d, %d): mask_lookahead %d / %d\n", rc1, rc2,
                    n1.mask_lookahead, n2.mask_lookahead);
            return 1;
        }
        if (rc4 != AINR_OK || has_row(&cfg, &n4)) {
            fprintf(stderr, "test_info_and_graph_rows: 48 kHz N=4 expected an accepted init and no graph row (%d)\n", rc4);
            return 1;
        }
    }
#endif

    printf("test_info_and_graph_rows: PASS\n");
    return 0;
}

int main(void) {
    DFN2Grid grid16 = make_grid_16k_256(1);
    ErbTables erb = erb_tables_synthetic(grid16.n_bins, grid16.n_erb);

    if (test_n1_matches_reference(&erb) != 0) return 1;
    if (test_n8_two_blocks_matches_reference(&erb) != 0) return 1;
    if (test_validation_ordering() != 0) return 1;
    if (test_postprocess_failure_recovery(&erb) != 0) return 1;
    if (test_sequence_misuse_and_reset(&erb) != 0) return 1;
    if (test_larger_pool_accepted(&erb) != 0) return 1;
    if (test_work_matches_exporter_shapes(&erb) != 0) return 1;
    if (test_info_and_graph_rows(&erb) != 0) return 1;

    erb_tables_free(&erb);
    printf("test_ainr_dfn2: ALL PASS\n");
    return 0;
}
