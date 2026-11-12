/* Link-only check: this file includes only ainr.h and ainr_test.h and links against the built libainr.a and
 * the DFN2 archive (never compiled with ainr.c's sources, unlike the other tests): the shipped archive
 * exports ainr_*, runs a real round trip per family on its own catalog row, and AinrWork's counts and
 * shapes reflect the variant it was built as. Pass -DAINR_TEST_EXPECT_LITE for the mask-only archive. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"

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
#else
#define EXPECT_SR           16000
#define EXPECT_FFT          256
#define EXPECT_N_BINS       129   /* fft/2 + 1 */
#define EXPECT_N_ERB        24
#define EXPECT_INPUT_COUNT  2
#define EXPECT_OUTPUT_COUNT 3
#endif

static void *alloc_aligned(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
}

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

/* Every tensor of the work has data, a non-zero shape and elements equal to the product of its axes. */
static int check_work_tensors(const AinrWork *io) {
    int i;
    for (i = 0; i < io->input_count; ++i) {
        const AinrTensorIn *t = &io->inputs[i];
        if (!t->data || t->elements == 0 || t->elements != (size_t)t->b * t->c * t->h * t->w) {
            fprintf(stderr, "link_only_ainr: input %d shape/elements\n", i);
            return 1;
        }
    }
    for (i = 0; i < io->output_count; ++i) {
        const AinrTensorOut *t = &io->outputs[i];
        if (!t->data || t->elements == 0 || t->elements != (size_t)t->b * t->c * t->h * t->w) {
            fprintf(stderr, "link_only_ainr: output %d shape/elements\n", i);
            return 1;
        }
    }
    return 0;
}

static int run_round_trip(void) {
    AinrConfig cfg;
    StreamMemReq req;
    void *pool;
    AinrCtx *ctx = NULL;
    AinrInfo info;
    char erb_path[64];
    int hop = EXPECT_FFT / 2;
    int16_t pcm_in[EXPECT_FFT / 2], pcm_out[EXPECT_FFT / 2];
    AinrWork io;
    int rc, i;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_link_only_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, EXPECT_N_BINS, EXPECT_N_ERB);

    cfg.sr = EXPECT_SR;
    cfg.fft = EXPECT_FFT;
    cfg.hop_in = hop;

    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &info) != AINR_OK || !ctx) {
        fprintf(stderr, "ainr_init_from_erb_file fail\n"); return 1;
    }

    if (info.graph_path == NULL || info.tail_zero_blocks < 1) {
        fprintf(stderr, "ainr_init: graph_path %p, tail_zero_blocks %d\n", (const void *)info.graph_path, info.tail_zero_blocks);
        return 1;
    }
    for (i = 0; i < hop; ++i) pcm_in[i] = (int16_t)((i * 13) % 211 - 100);
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

    unlink(erb_path);
    free(pool);
    printf("link_only_ainr: round trip OK (%d Hz/fft %d), input_count=%d output_count=%d\n",
          EXPECT_SR, EXPECT_FFT, io.input_count, io.output_count);
    return 0;
}

/* As run_round_trip(), for the "_spec" family (PCM in, enhanced spectrum out) -- confirms the
 * shipped archive exports ainr_*_spec too and that it round-trips on both variants' own
 * shipped-grid catalog row, same as the PCM family above. N = 1, so exactly one frame out. */
static int run_round_trip_spec(void) {
    AinrConfig cfg;
    StreamMemReq req;
    void *pool;
    AinrSpecCtx *ctx = NULL;
    AinrInfo info;
    char erb_path[64];
    int hop = EXPECT_FFT / 2;
    int16_t pcm_in[EXPECT_FFT / 2];
    float out_re[EXPECT_N_BINS], out_im[EXPECT_N_BINS];
    AinrWork io;
    int rc, i;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_link_only_spec_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, EXPECT_N_BINS, EXPECT_N_ERB);

    cfg.sr = EXPECT_SR;
    cfg.fft = EXPECT_FFT;
    cfg.hop_in = hop;

    if (ainr_get_mem_size_spec(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size_spec fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init_spec_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &info) != AINR_OK || !ctx) { fprintf(stderr, "ainr_init_spec_from_erb_file fail\n"); return 1; }

    for (i = 0; i < hop; ++i) pcm_in[i] = (int16_t)((i * 13) % 211 - 100);
    rc = ainr_preprocess_spec(ctx, pcm_in, &io);
    if (rc != AINR_OK) { fprintf(stderr, "ainr_preprocess_spec rc=%d\n", rc); return 1; }

    if (io.input_count != EXPECT_INPUT_COUNT || io.output_count != EXPECT_OUTPUT_COUNT) {
        fprintf(stderr, "link_only_ainr (_spec): input_count=%d output_count=%d, expected %d/%d\n",
               io.input_count, io.output_count, EXPECT_INPUT_COUNT, EXPECT_OUTPUT_COUNT);
        return 1;
    }
    if (check_work_tensors(&io) != 0) return 1;
    for (i = 0; i < io.output_count; ++i) {
        size_t k;
        for (k = 0; k < io.outputs[i].elements; ++k) io.outputs[i].data[k] = 0.0f;
    }
    rc = ainr_postprocess_spec(ctx, 0, out_re, out_im);
    if (rc != AINR_OK) { fprintf(stderr, "ainr_postprocess_spec rc=%d\n", rc); return 1; }

    unlink(erb_path);
    free(pool);
    printf("link_only_ainr (_spec): round trip OK (%d Hz/fft %d)\n",
          EXPECT_SR, EXPECT_FFT);
    return 0;
}

/* As run_round_trip_spec(), for the "_freq" family (spectrum in, enhanced spectrum out) --
 * E == P (same synthetic spectrum passed as both est and app). */
static int run_round_trip_freq(void) {
    AinrConfig cfg;
    StreamMemReq req;
    void *pool;
    AinrFreqCtx *ctx = NULL;
    AinrInfo info;
    char erb_path[64];
    int hop = EXPECT_FFT / 2;
    float in_re[EXPECT_N_BINS], in_im[EXPECT_N_BINS];
    float out_re[EXPECT_N_BINS], out_im[EXPECT_N_BINS];
    AinrWork io;
    int rc, i;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_link_only_freq_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, EXPECT_N_BINS, EXPECT_N_ERB);

    cfg.sr = EXPECT_SR;
    cfg.fft = EXPECT_FFT;
    cfg.hop_in = hop;

    if (ainr_get_mem_size_freq(&cfg, &req) != AINR_OK) { fprintf(stderr, "ainr_get_mem_size_freq fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init_freq_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &info) != AINR_OK || !ctx) { fprintf(stderr, "ainr_init_freq_from_erb_file fail\n"); return 1; }

    for (i = 0; i < EXPECT_N_BINS; ++i) { in_re[i] = 0.01f * (float)i; in_im[i] = -0.005f * (float)i; }
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

    unlink(erb_path);
    free(pool);
    printf("link_only_ainr (_freq): round trip OK (%d Hz/fft %d)\n",
          EXPECT_SR, EXPECT_FFT);
    return 0;
}

int main(void) {
    if (run_round_trip() != 0) return 1;
    if (run_round_trip_spec() != 0) return 1;
    if (run_round_trip_freq() != 0) return 1;
    return 0;
}
