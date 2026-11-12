/* Host-only regression suite for ainr.h/ainr.c, against a stand-in accelerator (no NPU). Each test compares
 * ainr_preprocess()/ainr_postprocess() against the same scenario driven directly through
 * dfn2_stream_pre_process()/post_process(), or exercises an error path (init validation order, post-process
 * failure, sequence misuse, pool size). Not built under NO_STDIO (it initialises from an ERB file). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "dfn2_stream.h"

/* ---- synthetic ERB file, same convention as tests/check_erb_file.c ---- */

static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *inv_bin_major = (float *)malloc(elems * sizeof(float));
    size_t i;
    int32_t hdr[2];
    FILE *fp;

    if (!fwd || !inv_bin_major) { fprintf(stderr, "malloc fail\n"); exit(1); }
    for (i = 0; i < elems; ++i) {
        fwd[i] = (float)(i % 17) * 0.01f;
        inv_bin_major[i] = (float)((i + 3) % 11) * 0.02f;
    }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins;
    hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv_bin_major, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd);
    free(inv_bin_major);
}

/* ---- deterministic stand-in accelerator, same fill on both the ainr path and the reference
 * path driven directly through dfn2_stream -- the fill values themselves are arbitrary, what
 * matters is that both paths get exactly the same ones. ---- */

static int raw_fill_accel(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
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

static void ainr_fill_accel(AinrWork *io) {
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

static void *alloc_aligned(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
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
        if (io.job_ready) ai_rc = raw_fill_accel(&io.inputs, &io.outputs);
        if (dfn2_stream_post_process(s, ai_rc, pcm_out + (size_t)i * (size_t)hop) < 0) {
            fprintf(stderr, "reference post_process fail at hop %d\n", i);
            exit(1);
        }
    }
}

static int run_ainr_blocks(AinrCtx *ctx, const int16_t *pcm_in, int blocks, int hop_in, int16_t *pcm_out) {
    int i;
    for (i = 0; i < blocks; ++i) {
        AinrWork io;
        int rc = ainr_preprocess(ctx, pcm_in + (size_t)i * (size_t)hop_in, &io);
        if (rc != AINR_OK) { fprintf(stderr, "ainr_preprocess rc=%d at block %d\n", rc, i); return -1; }
        ainr_fill_accel(&io);
        rc = ainr_postprocess(ctx, 0, pcm_out + (size_t)i * (size_t)hop_in);
        if (rc != AINR_OK) { fprintf(stderr, "ainr_postprocess rc=%d at block %d\n", rc, i); return -1; }
    }
    return 0;
}

static AinrInfo g_info;

static void fill_pcm(int16_t *pcm, int n, int seed) {
    int i;
    for (i = 0; i < n; ++i) pcm[i] = (int16_t)((i * 13 + seed * 7) % 211 - 100);
}

/* 1. N=1 catalog entry: ainr_* must be byte-identical to driving dfn2_stream_* directly. */
static int test_n1_matches_reference(const char *erb_path) {
    AinrConfig cfg;
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
    ainr_init_from_erb_file(&cfg, erb_path, ainr_pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init_from_erb_file fail\n"); return 1; }

    memset(&scfg, 0, sizeof(scfg));
    scfg.grid = grid;
    if (dfn2_stream_get_mem_size(&scfg, &sreq) != 0) { fprintf(stderr, "dfn2_stream_get_mem_size fail\n"); return 1; }
    ref_pool = alloc_aligned((size_t)sreq.bytes, sreq.alignment);
    ref = dfn2_stream_init_from_erb_file(&scfg, erb_path, ref_pool, (size_t)sreq.bytes);
    if (!ref) { fprintf(stderr, "dfn2_stream_init_from_erb_file fail\n"); return 1; }

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

/* 2 & 3. N=8, mask_lookahead=0, df_lookahead=0: PCM delay is exactly one hop_in block (8 hops).
 * Two blocks, the first carrying a non-silent signal and the second all zero, so the tail that
 * lands in the second block's output actually exercises the cross-block stash/boundary logic,
 * not just "both blocks happen to be silence". */
static int test_n8_two_blocks_matches_reference(const char *erb_path) {
    AinrConfig cfg;
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(8);
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
    ainr_init_from_erb_file(&cfg, erb_path, ainr_pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init_from_erb_file fail\n"); return 1; }

    memset(&scfg, 0, sizeof(scfg));
    scfg.grid = grid;
    if (dfn2_stream_get_mem_size(&scfg, &sreq) != 0) { fprintf(stderr, "dfn2_stream_get_mem_size fail\n"); return 1; }
    ref_pool = alloc_aligned((size_t)sreq.bytes, sreq.alignment);
    ref = dfn2_stream_init_from_erb_file(&scfg, erb_path, ref_pool, (size_t)sreq.bytes);
    if (!ref) { fprintf(stderr, "dfn2_stream_init_from_erb_file fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    ainr_out = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    ref_out = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, 8 * hop, 2);                                  /* block 1: non-silent */
    memset(pcm_in + (size_t)8 * hop, 0, (size_t)8 * hop * sizeof(int16_t));  /* block 2: silence */

    if (run_ainr_blocks(ctx, pcm_in, blocks, hop_in, ainr_out) != 0) return 1;
    run_reference_hops(ref, pcm_in, hops, hop, ref_out);

    if (memcmp(ainr_out, ref_out, (size_t)hops * (size_t)hop * sizeof(int16_t)) != 0) {
        fprintf(stderr, "test_n8_two_blocks_matches_reference: PCM mismatch\n");
        return 1;
    }
    /* The PCM delay (M+D+N hops = 8 hops = one whole hop_in block) means block 1's real signal
     * can only finish draining inside block 2's output -- confirm block 2 isn't trivially all
     * zero (that would mean the stash/boundary logic silently dropped the tail). */
    {
        int16_t *block2 = ainr_out + (size_t)hop_in;
        int all_zero = 1, i;
        for (i = 0; i < hop_in; ++i) if (block2[i] != 0) { all_zero = 0; break; }
        if (all_zero) {
            fprintf(stderr, "test_n8_two_blocks_matches_reference: block 2 is all-zero, "
                            "tail of block 1's signal appears to have been dropped\n");
            return 1;
        }
    }
    printf("test_n8_two_blocks_matches_reference: PASS\n");
    free(pcm_in); free(ainr_out); free(ref_out); free(ainr_pool); free(ref_pool);
    return 0;
}

/* 4. Validation ordering: an unshipped N must not be confused with an unsupported lookahead/N>1
 * combination -- the two are different failure reasons and must get different error codes. */
static int test_validation_ordering(void) {
    AinrConfig cfg_unshipped, cfg_lookahead_blocked;
    StreamMemReq req;
    int status;

    cfg_unshipped.sr = 16000;
    cfg_unshipped.fft = 256;
    cfg_unshipped.hop_in = 3 * 128;   /* N = 3: legal timing (mask_lookahead == 0), not shipped */
    status = ainr_get_mem_size(&cfg_unshipped, &req);
    if (status != AINR_ERR_FRAMES_PER_CALL) {
        fprintf(stderr, "test_validation_ordering: 16kHz N=3 -> %d (%s), expected AINR_ERR_FRAMES_PER_CALL\n",
               status, ainr_status_text(status));
        return 1;
    }

    cfg_lookahead_blocked.sr = 48000;
    cfg_lookahead_blocked.fft = 1024;
    cfg_lookahead_blocked.hop_in = 8 * 512;   /* N = 8 against the shipped mask_lookahead=1 grid */
    status = ainr_get_mem_size(&cfg_lookahead_blocked, &req);
    if (status != AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED) {
        fprintf(stderr, "test_validation_ordering: 48kHz N=8 -> %d (%s), expected AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED\n",
               status, ainr_status_text(status));
        return 1;
    }

    printf("test_validation_ordering: PASS\n");
    return 0;
}

/* 5. A failed accelerator run (infer_status != 0) still closes the block and leaves the next one
 * able to proceed normally -- ainr_postprocess's return value is a contract-success flag, not a
 * pass-through of the model's internal spectral-validity bit. */
static int test_postprocess_failure_recovery(const char *erb_path) {
    AinrConfig cfg;
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    void *pool;
    AinrCtx *ctx;
    int hop = grid.hop;
    int16_t *pcm_in, *pcm_out;
    AinrWork io;
    int rc;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init_from_erb_file fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    pcm_out = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, hop, 3);

    {
        unsigned int gen_before = ainr_state_generation(ctx);
        int all_zero = 1, i;

        rc = ainr_preprocess(ctx, pcm_in, &io);
        if (rc != AINR_OK) { fprintf(stderr, "test_postprocess_failure_recovery: preprocess rc=%d\n", rc); return 1; }
        ainr_fill_accel(&io);   /* the accelerator "ran" successfully, but report failure anyway */
        rc = ainr_postprocess(ctx, -1 /* simulated bind/infer/readback failure */, pcm_out);
        if (rc != AINR_OK) {
            fprintf(stderr, "test_postprocess_failure_recovery: postprocess after failure rc=%d, "
                            "expected AINR_OK (PCM still written)\n", rc);
            return 1;
        }
        /* Bypass is a near-identity pass-through, not silence -- a nonzero input must come back
         * nonzero, not a zeroed block (this is what ainr.h's doc comment claims; nothing in this
         * test checked the actual samples before). */
        for (i = 0; i < hop; ++i) { if (pcm_out[i] != 0) { all_zero = 0; break; } }
        if (all_zero) {
            fprintf(stderr, "test_postprocess_failure_recovery: bypassed block came back all-zero "
                            "(expected pass-through, not silence)\n");
            return 1;
        }
        /* A bypass alone (no reset, nothing non-finite to detect since no state is registered
         * host-side in this deployment) must not advance state_generation. */
        if (ainr_state_generation(ctx) != gen_before) {
            fprintf(stderr, "test_postprocess_failure_recovery: state_generation changed from a "
                            "bypass alone (%u -> %u)\n", gen_before, ainr_state_generation(ctx));
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
static int test_sequence_misuse_and_reset(const char *erb_path) {
    AinrConfig cfg;
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    void *pool;
    AinrCtx *ctx;
    int hop = grid.hop;
    int16_t *pcm_in, *pcm_out;
    AinrWork io;
    int rc;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "ainr_init_from_erb_file fail\n"); return 1; }

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
static int test_larger_pool_accepted(const char *erb_path) {
    AinrConfig cfg_big, cfg_small;
    StreamMemReq req_big, req_small;
    unsigned char *pool_big, *pool_exact;
    AinrCtx *ctx_big = NULL, *ctx_exact = NULL;
    int16_t pcm_in[128 * 4], out_big[128 * 4], out_exact[128 * 4];
    size_t i;
    int b;

    cfg_big.sr = 16000;
    cfg_big.fft = 256;
    cfg_big.hop_in = 8 * 128;   /* N = 8: needs stash space, bigger pool */

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
    if (ainr_init_from_erb_file(&cfg_small, erb_path, pool_big, (size_t)req_big.bytes, &ctx_big, &g_info) != AINR_OK || !ctx_big ||
        ainr_init_from_erb_file(&cfg_small, erb_path, pool_exact, (size_t)req_small.bytes, &ctx_exact, &g_info) != AINR_OK || !ctx_exact) {
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
        AinrWork io;
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

/* AinrWork's shapes come from the grid math while its data pointers and element counts come from the
 * authoritative DFN2Prepost views -- the one place those two could silently disagree. Every shape field
 * of a real block is checked against the exporter's tensor shapes spelled out for the 16k/256 N=1 grid
 * (graph inputs erb [1,1,T,24], spec [1,2,T,96] with T = N + 2; heads erb_mask [1,1,1,24], full also
 * df_coefs [1,1,96,10] and df_alpha [1,1,1,1]), and element counts against the product of the axes. */
typedef struct WorkShape { unsigned int b, c, h, w; } WorkShape;

static int test_work_matches_exporter_shapes(const char *erb_path) {
    static const WorkShape want_in[2] = { { 1, 1, 3, 24 }, { 1, 2, 3, 96 } };
#ifdef DFN2_MASK_ONLY
    static const WorkShape want_out[1] = { { 1, 1, 1, 24 } };
#else
    static const WorkShape want_out[3] = { { 1, 1, 1, 24 }, { 1, 1, 96, 10 }, { 1, 1, 1, 1 } };
#endif
    const int want_out_count = (int)(sizeof(want_out) / sizeof(want_out[0]));
    AinrConfig cfg;
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    void *pool;
    AinrCtx *ctx;
    int hop = grid.hop;
    int16_t *pcm_in, *pcm_out;
    AinrInfo info;
    AinrWork real_io;
    int rc, i;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &info);
    if (!ctx) { fprintf(stderr, "ainr_init_from_erb_file fail\n"); return 1; }

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
        const AinrTensorIn *t = &real_io.inputs[i];
        if (t->b != want_in[i].b || t->c != want_in[i].c || t->h != want_in[i].h || t->w != want_in[i].w ||
            t->elements != (size_t)t->b * t->c * t->h * t->w || !t->data) {
            fprintf(stderr, "test_work_matches_exporter_shapes: inputs[%d] shape mismatch\n", i);
            return 1;
        }
    }
    for (i = 0; i < real_io.output_count; ++i) {
        const AinrTensorOut *t = &real_io.outputs[i];
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

/* 9. The init info's graph_path: resolved per (sr, fft, N) for the product's bound model -- the same
 * grid at N=1 and N=8 must resolve to two different graphs (N fixes the input window length); an
 * unregistered grid or an N without a delivered graph is refused with its own code and leaves ctx
 * NULL and info untouched. The quantization alternates (ainr_product.h) are reached only through the
 * product binding, see tests/test_ainr_product.c. */
static int init_for(const AinrConfig *cfg, const char *erb_path, AinrCtx **ctx, AinrInfo *info) {
    _Alignas(16) static unsigned char pool[1 << 21];   /* larger than any registered config needs */
    return ainr_init_from_erb_file(cfg, erb_path, pool, sizeof(pool), ctx, info);
}

static int test_graph_path_resolution(const char *erb_path) {
    AinrConfig cfg;
    AinrInfo bi, bi_n8, sentinel;
    AinrCtx *ctx = NULL;

    cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = 128;   /* N = 1 */
    if (init_for(&cfg, erb_path, &ctx, &bi) != AINR_OK || !ctx || !bi.graph_path ||
        strstr(bi.graph_path, "16x16") == NULL || strstr(bi.graph_path, "16k") == NULL) {
        fprintf(stderr, "test_graph_path_resolution: FULL N=1 path unexpected: %s\n", bi.graph_path ? bi.graph_path : "(null)");
        return 1;
    }

    cfg.hop_in = 8 * 128;   /* N = 8 */
    if (init_for(&cfg, erb_path, &ctx, &bi_n8) != AINR_OK || !ctx || !bi_n8.graph_path ||
        strcmp(bi.graph_path, bi_n8.graph_path) == 0) {
        fprintf(stderr, "test_graph_path_resolution: N=1 and N=8 did not resolve to distinct graphs\n");
        return 1;
    }

    memset(&sentinel, 0x5a, sizeof(sentinel));
    bi_n8 = sentinel;
    cfg.hop_in = 3 * 128;   /* N = 3: no graph */
    if (init_for(&cfg, erb_path, &ctx, &bi_n8) != AINR_ERR_FRAMES_PER_CALL || ctx != NULL ||
        memcmp(&bi_n8, &sentinel, sizeof(bi_n8)) != 0) {
        fprintf(stderr, "test_graph_path_resolution: N=3 expected AINR_ERR_FRAMES_PER_CALL, ctx NULL and info untouched\n");
        return 1;
    }

    bi_n8 = sentinel;
    cfg.sr = 22000; cfg.fft = 333; cfg.hop_in = 166;   /* unknown grid */
    if (init_for(&cfg, erb_path, &ctx, &bi_n8) != AINR_ERR_GRID || ctx != NULL ||
        memcmp(&bi_n8, &sentinel, sizeof(bi_n8)) != 0) {
        fprintf(stderr, "test_graph_path_resolution: unknown grid expected AINR_ERR_GRID, ctx NULL and info untouched\n");
        return 1;
    }

    printf("test_graph_path_resolution: PASS\n");
    return 0;
}

/* 10. ainr_init() (resolves and loads its own ERB path from the private release-asset record)
 * against ainr_init_from_erb_file() (explicit path) pointed at the identical file content -- same
 * N=1 block must produce byte-identical PCM either way, confirming the new entry point's formula
 * ("%s/fb_%d_%d.bin", AINR_ERB_MATRIX_DIR, sr, n_erb; this test binary points AINR_ERB_MATRIX_DIR
 * at this config's own OBJ_DIR, see the Makefile) is not a second, divergent code path. */
static int test_init_matches_erb_file(void) {
    AinrConfig cfg;
    StreamMemReq req;
    DFN2Grid grid = make_grid_16k_256(1);
    char private_erb_path[128];
    void *pool_a, *pool_b;
    AinrCtx *ctx_a, *ctx_b;
    int hop = grid.hop, hops = 5;
    int16_t *pcm_in, *out_a, *out_b;
    int i;

    cfg.sr = 16000;
    cfg.fft = 256;
    cfg.hop_in = hop;

    /* This is the exact path ainr_init()'s internal formula computes for this grid (AINR_ERB_MATRIX_DIR
     * is this build's own OBJ_DIR, see the Makefile -- never a shared location like /tmp, so this
     * can never collide with a real deployment's or another tool's file of the same name) --
     * write the synthetic ERB content there so ainr_init() can actually load it on this host. */
    snprintf(private_erb_path, sizeof(private_erb_path), "%s/fb_%d_%d.bin",
             AINR_ERB_MATRIX_DIR, grid.sample_rate, grid.n_erb);
    write_synthetic_erb(private_erb_path, grid.n_bins, grid.n_erb);

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }

    pool_a = alloc_aligned((size_t)req.bytes, req.alignment);
    ctx_a = NULL;
    ainr_init(&cfg, pool_a, (size_t)req.bytes, &ctx_a, &g_info);
    if (!ctx_a) { fprintf(stderr, "test_init_matches_erb_file: ainr_init fail\n"); return 1; }

    pool_b = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_from_erb_file(&cfg, private_erb_path, pool_b, (size_t)req.bytes, &ctx_b, &g_info);
    if (!ctx_b) { fprintf(stderr, "test_init_matches_erb_file: ainr_init_from_erb_file fail\n"); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    out_a = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    out_b = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    fill_pcm(pcm_in, hops * hop, 9);

    for (i = 0; i < hops; ++i) {
        AinrWork io_a, io_b;
        int rc;
        rc = ainr_preprocess(ctx_a, pcm_in + (size_t)i * (size_t)hop, &io_a);
        if (rc != AINR_OK) { fprintf(stderr, "test_init_matches_erb_file: ctx_a preprocess rc=%d\n", rc); return 1; }
        ainr_fill_accel(&io_a);
        rc = ainr_postprocess(ctx_a, 0, out_a + (size_t)i * (size_t)hop);
        if (rc != AINR_OK) { fprintf(stderr, "test_init_matches_erb_file: ctx_a postprocess rc=%d\n", rc); return 1; }

        rc = ainr_preprocess(ctx_b, pcm_in + (size_t)i * (size_t)hop, &io_b);
        if (rc != AINR_OK) { fprintf(stderr, "test_init_matches_erb_file: ctx_b preprocess rc=%d\n", rc); return 1; }
        ainr_fill_accel(&io_b);
        rc = ainr_postprocess(ctx_b, 0, out_b + (size_t)i * (size_t)hop);
        if (rc != AINR_OK) { fprintf(stderr, "test_init_matches_erb_file: ctx_b postprocess rc=%d\n", rc); return 1; }
    }

    if (memcmp(out_a, out_b, (size_t)hops * (size_t)hop * sizeof(int16_t)) != 0) {
        fprintf(stderr, "test_init_matches_erb_file: PCM mismatch between ainr_init() and "
                        "ainr_init_from_erb_file()\n");
        return 1;
    }

    printf("test_init_matches_erb_file: PASS\n");
    free(pcm_in); free(out_a); free(out_b); free(pool_a); free(pool_b);
    unlink(private_erb_path);
    return 0;
}

int main(void) {
    char erb_path[64];
    int n_bins, n_erb;
    DFN2Grid grid16 = make_grid_16k_256(1);

    n_bins = grid16.n_bins;
    n_erb = grid16.n_erb;
    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_dfn2_scratch_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, n_bins, n_erb);

    if (test_n1_matches_reference(erb_path) != 0) return 1;
    if (test_n8_two_blocks_matches_reference(erb_path) != 0) return 1;
    if (test_validation_ordering() != 0) return 1;
    if (test_postprocess_failure_recovery(erb_path) != 0) return 1;
    if (test_sequence_misuse_and_reset(erb_path) != 0) return 1;
    if (test_larger_pool_accepted(erb_path) != 0) return 1;
    if (test_work_matches_exporter_shapes(erb_path) != 0) return 1;
    if (test_graph_path_resolution(erb_path) != 0) return 1;
    if (test_init_matches_erb_file() != 0) return 1;

    unlink(erb_path);
    printf("test_ainr_dfn2: ALL PASS\n");
    return 0;
}
