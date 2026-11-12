/* The tensor views (AinrWork) a real preprocess call hands the sample, for every registered config and each
 * of the three families (PCM, _spec, _freq): counts, per-tensor [b,c,h,w], element counts and non-NULL data
 * pointers, compared with a table spelled out from the exporter's graph (AINR/DeepFilterNet2/export_onnx.py),
 * not derived from the library. Signal inputs are erb [1,1,T,E] and spec [1,2,T,96] with T = N + 2 time
 * steps; heads are erb_mask [1,1,N,E] and, in the full variant, df_coefs [1,N,96,10] and df_alpha [1,1,N,1].
 * The recurrent state is the accelerator's and is not in the work. Also: the refusals of ainr_get_mem_size()
 * come back from every init family with *ctx NULL and *info untouched, and the shapes repeat on the next block.
 * Not built under NO_STDIO (ainr.c is file-mode). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Shape { unsigned int b, c, h, w; } Shape;

static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb, i;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *inv = (float *)malloc(elems * sizeof(float));
    int32_t hdr[2];
    FILE *fp;
    if (!fwd || !inv) exit(1);
    for (i = 0; i < elems; ++i) { fwd[i] = (float)(i % 17) * 0.01f; inv[i] = (float)((i + 3) % 11) * 0.02f; }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd); free(inv);
}

static void check_work(const AinrWork *w, int n, int n_erb, const char *label) {
    const unsigned int N = (unsigned int)n, E = (unsigned int)n_erb, T = N + 2u;
    Shape in[2], out[3];
    int n_out, i;
    in[0] = (Shape){ 1, 1, T, E };
    in[1] = (Shape){ 1, 2, T, 96 };
    out[0] = (Shape){ 1, 1, N, E };
#ifdef DFN2_MASK_ONLY
    n_out = 1;
#else
    out[1] = (Shape){ 1, N, 96, 10 };
    out[2] = (Shape){ 1, 1, N, 1 };
    n_out = 3;
#endif
    CHECK(w->input_count == 2, "%s: input_count %d", label, w->input_count);
    CHECK(w->output_count == n_out, "%s: output_count %d, want %d", label, w->output_count, n_out);
    for (i = 0; i < 2 && i < w->input_count; ++i) {
        const AinrTensorIn *t = &w->inputs[i];
        CHECK(t->b == in[i].b && t->c == in[i].c && t->h == in[i].h && t->w == in[i].w,
              "%s: input %d shape [%u,%u,%u,%u]", label, i, t->b, t->c, t->h, t->w);
        CHECK(t->elements == (size_t)in[i].b * in[i].c * in[i].h * in[i].w, "%s: input %d elements %zu", label, i, t->elements);
        CHECK(t->data != NULL, "%s: input %d data NULL", label, i);
    }
    for (i = 0; i < n_out && i < w->output_count; ++i) {
        const AinrTensorOut *t = &w->outputs[i];
        CHECK(t->b == out[i].b && t->c == out[i].c && t->h == out[i].h && t->w == out[i].w,
              "%s: output %d shape [%u,%u,%u,%u]", label, i, t->b, t->c, t->h, t->w);
        CHECK(t->elements == (size_t)out[i].b * out[i].c * out[i].h * out[i].w, "%s: output %d elements %zu", label, i, t->elements);
        CHECK(t->data != NULL, "%s: output %d data NULL", label, i);
    }
}

static void fill_heads(AinrWork *w) {
    int i;
    size_t k;
    for (i = 0; i < w->output_count; ++i)
        for (k = 0; k < w->outputs[i].elements; ++k) w->outputs[i].data[k] = i == 0 ? 0.5f : 0.0f;
}

/* Two blocks per family: the second block carries the same shapes as the first. */
static void run_config(int sr, int fft, int n_erb, int n, const char *label) {
    AinrConfig cfg;
    StreamMemReq req;
    AinrInfo info;
    char erb[96], what[96];
    void *pool = NULL;
    int16_t *pcm;
    float *re, *im, *ore, *oim;
    int hop = fft / 2, bins = fft / 2 + 1, fam, blk, i;

    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = n * hop;
    snprintf(erb, sizeof(erb), "/tmp/ainr_work_shapes_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, bins, n_erb);
    pcm = (int16_t *)calloc((size_t)cfg.hop_in, sizeof(int16_t));
    re = (float *)calloc((size_t)n * bins, sizeof(float));
    im = (float *)calloc((size_t)n * bins, sizeof(float));
    ore = (float *)calloc((size_t)n * bins, sizeof(float));
    oim = (float *)calloc((size_t)n * bins, sizeof(float));
    if (!pcm || !re || !im || !ore || !oim) exit(1);
    for (i = 0; i < cfg.hop_in; ++i) pcm[i] = (int16_t)((i * 13) % 211 - 100);
    for (i = 0; i < n * bins; ++i) { re[i] = 0.01f * (float)(i % 7); im[i] = -0.01f * (float)(i % 5); }

    for (fam = 0; fam < 3; ++fam) {
        AinrCtx *c0 = NULL;
        AinrSpecCtx *c1 = NULL;
        AinrFreqCtx *c2 = NULL;
        int rc;

        rc = fam == 0 ? ainr_get_mem_size(&cfg, &req) : fam == 1 ? ainr_get_mem_size_spec(&cfg, &req)
                                                                 : ainr_get_mem_size_freq(&cfg, &req);
        CHECK(rc == AINR_OK, "%s fam %d: get_mem_size rc %d", label, fam, rc);
        if (rc != AINR_OK || posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(1);
        memset(&info, 0, sizeof(info));
        rc = fam == 0 ? ainr_init_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &c0, &info)
           : fam == 1 ? ainr_init_spec_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &c1, &info)
                      : ainr_init_freq_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &c2, &info);
        CHECK(rc == AINR_OK, "%s fam %d: init rc %d", label, fam, rc);
        if (rc != AINR_OK) exit(1);
        CHECK(info.graph_path != NULL, "%s fam %d: graph", label, fam);

        for (blk = 0; blk < 2; ++blk) {
            AinrWork w;
            memset(&w, 0, sizeof(w));
            snprintf(what, sizeof(what), "%s fam %d block %d", label, fam, blk);
            rc = fam == 0 ? ainr_preprocess(c0, pcm, &w) : fam == 1 ? ainr_preprocess_spec(c1, pcm, &w)
                                                                    : ainr_preprocess_freq(c2, re, im, re, im, &w);
            CHECK(rc == AINR_OK, "%s: preprocess rc %d", what, rc);
            if (rc != AINR_OK) break;
            check_work(&w, n, n_erb, what);
            fill_heads(&w);
            rc = fam == 0 ? ainr_postprocess(c0, 0, pcm) : fam == 1 ? ainr_postprocess_spec(c1, 0, ore, oim)
                                                                    : ainr_postprocess_freq(c2, 0, ore, oim);
            CHECK(rc == AINR_OK, "%s: postprocess rc %d", what, rc);
        }
        free(pool);
        pool = NULL;
    }
    unlink(erb);
    free(pcm); free(re); free(im); free(ore); free(oim);
}

/* Every refusal of the sizing call comes back from the three init families, with *ctx NULL and *info untouched. */
static void check_refusals(const char *erb) {
    static const struct { int sr, fft, hop_in, want; const char *what; } cases[] = {
        { 44100, 256, 128, AINR_ERR_GRID, "unregistered grid" },
        { 48000, 1024, 100, AINR_ERR_IN_HOP, "hop_in not a multiple of model_hop" },
        { 48000, 1024, 1024, AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED, "lookahead grid with N = 2" },
#ifndef DFN2_MASK_ONLY
        { 16000, 256, 512, AINR_ERR_FRAMES_PER_CALL, "N = 4 has no graph" },
#endif
    };
    _Alignas(16) static unsigned char pool[1 << 21];
    size_t i;
    AinrInfo info, sentinel;
    AinrConfig cfg;
    StreamMemReq req;
    memset(&sentinel, 0x5a, sizeof(sentinel));
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        AinrCtx *c0 = (AinrCtx *)pool;
        AinrSpecCtx *c1 = (AinrSpecCtx *)pool;
        AinrFreqCtx *c2 = (AinrFreqCtx *)pool;
        cfg.sr = cases[i].sr; cfg.fft = cases[i].fft; cfg.hop_in = cases[i].hop_in;
        CHECK(ainr_get_mem_size(&cfg, &req) == cases[i].want, "refusal (%s): get_mem_size", cases[i].what);
        info = sentinel;
        CHECK(ainr_init_from_erb_file(&cfg, erb, pool, sizeof(pool), &c0, &info) == cases[i].want, "refusal (%s): init", cases[i].what);
        CHECK(c0 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init", cases[i].what);
        info = sentinel;
        CHECK(ainr_init_spec_from_erb_file(&cfg, erb, pool, sizeof(pool), &c1, &info) == cases[i].want, "refusal (%s): init_spec", cases[i].what);
        CHECK(c1 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init_spec", cases[i].what);
        info = sentinel;
        CHECK(ainr_init_freq_from_erb_file(&cfg, erb, pool, sizeof(pool), &c2, &info) == cases[i].want, "refusal (%s): init_freq", cases[i].what);
        CHECK(c2 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init_freq", cases[i].what);
    }
}

int main(void) {
    char erb[96];
    snprintf(erb, sizeof(erb), "/tmp/ainr_work_shapes_refusal_%d.bin", (int)getpid());
    write_synthetic_erb(erb, 513, 32);
    check_refusals(erb);
    unlink(erb);
#ifdef DFN2_MASK_ONLY
    run_config(48000, 1024, 32, 1, "work[lite 48k/1024 N=1]");
#else
    run_config(16000, 256, 24, 1, "work[full 16k/256 N=1]");
    run_config(16000, 256, 24, 8, "work[full 16k/256 N=8]");
    run_config(48000, 1024, 32, 1, "work[full 48k/1024 N=1]");
#endif
    if (g_fail) { printf("test_ainr_work_shapes: %d check(s) FAILED\n", g_fail); return 1; }
    printf("test_ainr_work_shapes: PASS\n");
    return 0;
}
