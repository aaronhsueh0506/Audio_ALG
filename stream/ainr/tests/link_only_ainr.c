/* Link-only check: this file includes only ainr.h and links against the built libainr.a and
 * the DFN2 archive (never compiled with ainr.c's sources, unlike the other tests): the shipped archive
 * exports ainr_*, runs a real round trip per family on its own catalog row, and StreamWork's counts and
 * shapes reflect the variant it was built as. Pass -DAINR_TEST_EXPECT_LITE for the mask-only archive. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr.h"

#ifdef STREAM_COMMON_BIND_H
#error "ainr.h must not pull in the sample-side graph/device wiring header"
#endif

#ifdef AINR_TEST_EXPECT_LITE
#define EXPECT_SR           48000
#define EXPECT_FFT          1024
#define EXPECT_N_BINS       513   /* fft/2 + 1 */
#define EXPECT_N_ERB        32
#define EXPECT_INPUT_COUNT  2
#define EXPECT_OUTPUT_COUNT 1
#define EXPECT_BYPASS       1     /* blocks without a job after init (mask_lookahead) */
#else
#define EXPECT_SR           16000
#define EXPECT_FFT          256
#define EXPECT_N_BINS       129   /* fft/2 + 1 */
#define EXPECT_N_ERB        24
#define EXPECT_INPUT_COUNT  2
#define EXPECT_OUTPUT_COUNT 3
#define EXPECT_BYPASS       0
#endif

static void *alloc_aligned(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
}

/* A partition of unity: bin i splits 0.75 / 0.25 over bands i % n_erb and (i + 1) % n_erb. fwd bin-major [n_bins][n_erb],
 * inv band-major [n_erb][n_bins]. */
static void make_tables(int n_bins, int n_erb, float **fwd, float **inv) {
    size_t elems = (size_t)n_bins * (size_t)n_erb, i;
    *fwd = (float *)malloc(elems * sizeof(float));
    *inv = (float *)calloc(elems, sizeof(float));
    if (!*fwd || !*inv) { fprintf(stderr, "malloc fail\n"); exit(1); }
    for (i = 0; i < elems; ++i) (*fwd)[i] = (float)(i % 17) * 0.01f;
    for (i = 0; i < (size_t)n_bins; ++i) {
        (*inv)[(i % (size_t)n_erb) * (size_t)n_bins + i] = 0.75f;
        (*inv)[((i + 1) % (size_t)n_erb) * (size_t)n_bins + i] = 0.25f;
    }
}


/* A block without a job: action BYPASS and an all-zero work. */
static int check_bypass_work(const StreamWork *io) {
    if (io->action != STREAM_WORK_BYPASS || io->input_count != 0 || io->output_count != 0 || io->state_generation != 0u) {
        fprintf(stderr, "link_only_ainr: block without a job: action %d, %d inputs, %d outputs, generation %u\n",
                (int)io->action, io->input_count, io->output_count, io->state_generation);
        return 1;
    }
    return 0;
}

/* Every tensor of the work has data, a non-zero shape and elements equal to the product of its axes. */
static int check_work_tensors(const StreamWork *io) {
    int i;
    for (i = 0; i < io->input_count; ++i) {
        const StreamTensorIn *t = &io->inputs[i];
        if (!t->data || t->elements == 0 || t->elements != (size_t)t->b * t->c * t->h * t->w) {
            fprintf(stderr, "link_only_ainr: input %d shape/elements\n", i);
            return 1;
        }
    }
    for (i = 0; i < io->output_count; ++i) {
        const StreamTensorOut *t = &io->outputs[i];
        if (!t->data || t->elements == 0 || t->elements != (size_t)t->b * t->c * t->h * t->w) {
            fprintf(stderr, "link_only_ainr: output %d shape/elements\n", i);
            return 1;
        }
    }
    return 0;
}

static int run_round_trip(void) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    void *pool;
    AinrCtx *ctx = NULL;
    AinrInfo info;
    float *erb_fwd, *erb_inv;
    int hop = EXPECT_FFT / 2;
    int16_t pcm_in[EXPECT_FFT / 2], pcm_out[EXPECT_FFT / 2];
    StreamWork io;
    int rc, i, b;

    make_tables(EXPECT_N_BINS, EXPECT_N_ERB, &erb_fwd, &erb_inv);

    cfg.sr = EXPECT_SR;
    cfg.fft = EXPECT_FFT;
    cfg.hop_in = hop;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.n_erb = EXPECT_N_ERB;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init(&cfg, pool, (size_t)req.bytes, &ctx, &info) != AINR_OK || !ctx) {
        fprintf(stderr, "ainr_init fail\n"); return 1;
    }

    if (info.mask_lookahead != EXPECT_BYPASS) {
        fprintf(stderr, "ainr_init: mask_lookahead %d, expected %d\n", info.mask_lookahead, EXPECT_BYPASS);
        return 1;
    }
    for (i = 0; i < hop; ++i) pcm_in[i] = (int16_t)((i * 13) % 211 - 100);
    for (b = 0; b < EXPECT_BYPASS; ++b) {
        rc = ainr_preprocess(ctx, pcm_in, &io);
        if (rc != AINR_OK || check_bypass_work(&io) != 0) { fprintf(stderr, "ainr_preprocess rc=%d\n", rc); return 1; }
        rc = ainr_postprocess(ctx, STREAM_RUN_NOT_STARTED, pcm_out);
        if (rc != AINR_OK) { fprintf(stderr, "ainr_postprocess rc=%d\n", rc); return 1; }
    }
    rc = ainr_preprocess(ctx, pcm_in, &io);
    if (rc != AINR_OK) { fprintf(stderr, "ainr_preprocess rc=%d\n", rc); return 1; }

    if (io.input_count != EXPECT_INPUT_COUNT || io.output_count != EXPECT_OUTPUT_COUNT) {
        fprintf(stderr, "link_only_ainr: input_count=%d output_count=%d, expected %d/%d\n",
               io.input_count, io.output_count, EXPECT_INPUT_COUNT, EXPECT_OUTPUT_COUNT);
        return 1;
    }
    if (check_work_tensors(&io) != 0) return 1;
    for (i = 0; i < io.output_count; ++i) {
        size_t k;
        for (k = 0; k < io.outputs[i].elements; ++k) io.outputs[i].data[k] = 0.0f;
    }
    rc = ainr_postprocess(ctx, 0, pcm_out);
    if (rc != AINR_OK) { fprintf(stderr, "ainr_postprocess rc=%d\n", rc); return 1; }

    free(pool);
    free(erb_fwd);
    free(erb_inv);
    printf("link_only_ainr: round trip OK (%d Hz/fft %d), input_count=%d output_count=%d\n",
          EXPECT_SR, EXPECT_FFT, io.input_count, io.output_count);
    return 0;
}

/* As run_round_trip(), for the "_freq" family (spectrum in, enhanced spectrum out) --
 * E == P (same synthetic spectrum passed as both est and app). */
static int run_round_trip_freq(void) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    void *pool;
    AinrFreqCtx *ctx = NULL;
    AinrInfo info;
    float *erb_fwd, *erb_inv;
    int hop = EXPECT_FFT / 2;
    float in_re[EXPECT_N_BINS], in_im[EXPECT_N_BINS];
    float out_re[EXPECT_N_BINS], out_im[EXPECT_N_BINS];
    StreamWork io;
    int rc, i, b;

    make_tables(EXPECT_N_BINS, EXPECT_N_ERB, &erb_fwd, &erb_inv);

    cfg.sr = EXPECT_SR;
    cfg.fft = EXPECT_FFT;
    cfg.hop_in = hop;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.n_erb = EXPECT_N_ERB;

    if (ainr_get_mem_size_freq(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size_freq fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init_freq(&cfg, pool, (size_t)req.bytes, &ctx, &info) != AINR_OK || !ctx) { fprintf(stderr, "ainr_init_freq fail\n"); return 1; }

    for (i = 0; i < EXPECT_N_BINS; ++i) { in_re[i] = 0.01f * (float)i; in_im[i] = -0.005f * (float)i; }
    for (b = 0; b < EXPECT_BYPASS; ++b) {
        rc = ainr_preprocess_freq(ctx, in_re, in_im, in_re, in_im, &io);
        if (rc != AINR_OK || check_bypass_work(&io) != 0) { fprintf(stderr, "ainr_preprocess_freq rc=%d\n", rc); return 1; }
        rc = ainr_postprocess_freq(ctx, STREAM_RUN_NOT_STARTED, out_re, out_im);
        if (rc != AINR_OK) { fprintf(stderr, "ainr_postprocess_freq rc=%d\n", rc); return 1; }
    }
    rc = ainr_preprocess_freq(ctx, in_re, in_im, in_re, in_im, &io);
    if (rc != AINR_OK) { fprintf(stderr, "ainr_preprocess_freq rc=%d\n", rc); return 1; }

    if (io.input_count != EXPECT_INPUT_COUNT || io.output_count != EXPECT_OUTPUT_COUNT) {
        fprintf(stderr, "link_only_ainr (_freq): input_count=%d output_count=%d, expected %d/%d\n",
               io.input_count, io.output_count, EXPECT_INPUT_COUNT, EXPECT_OUTPUT_COUNT);
        return 1;
    }
    if (check_work_tensors(&io) != 0) return 1;
    for (i = 0; i < io.output_count; ++i) {
        size_t k;
        for (k = 0; k < io.outputs[i].elements; ++k) io.outputs[i].data[k] = 0.0f;
    }
    rc = ainr_postprocess_freq(ctx, 0, out_re, out_im);
    if (rc != AINR_OK) { fprintf(stderr, "ainr_postprocess_freq rc=%d\n", rc); return 1; }

    free(pool);
    free(erb_fwd);
    free(erb_inv);
    printf("link_only_ainr (_freq): round trip OK (%d Hz/fft %d)\n",
          EXPECT_SR, EXPECT_FFT);
    return 0;
}

int main(void) {
    if (run_round_trip() != 0) return 1;
    if (run_round_trip_freq() != 0) return 1;
    return 0;
}
