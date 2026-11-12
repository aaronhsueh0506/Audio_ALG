/* ainr_set_atten_lim*(): the limit reaches the enhance stage (a fully suppressing mask leaves more
 * energy under a 20 dB limit than with the limit disabled), the value survives ainr_reset(), and the
 * three families refuse NULL, NaN/Inf and a call with a block open. Not built under NO_STDIO. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"

#define FAIL(...) do { fprintf(stderr, "%s: ", label); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } while (0)

static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb, i;
    float *fwd = (float *)malloc(elems * sizeof(float)), *inv = (float *)malloc(elems * sizeof(float));
    int32_t hdr[2] = { n_bins, n_erb };
    FILE *fp = fopen(path, "wb");
    if (!fwd || !inv || !fp) exit(1);
    for (i = 0; i < elems; ++i) { fwd[i] = (float)(i % 17) * 0.01f; inv[i] = (float)((i + 3) % 11) * 0.02f; }
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd); free(inv);
}

static void *pool_for(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, bytes) != 0) exit(1);
    return p;
}

/* Energy of `blocks` PCM blocks through a fully suppressing mask (every head zero). */
static double run_energy(AinrCtx *ctx, int hop_in, int blocks) {
    int16_t *in = (int16_t *)calloc((size_t)hop_in, sizeof(int16_t));
    int16_t *out = (int16_t *)calloc((size_t)hop_in, sizeof(int16_t));
    double e = 0.0;
    int b, i, o;
    size_t k;
    for (b = 0; b < blocks; ++b) {
        AinrWork io;
        for (i = 0; i < hop_in; ++i) in[i] = (int16_t)((((i * 37 + b * 11) % 201) - 100) * 40);
        if (ainr_preprocess(ctx, in, &io) != AINR_OK) exit(1);
        for (o = 0; o < io.output_count; ++o)
            for (k = 0; k < io.outputs[o].elements; ++k) io.outputs[o].data[k] = 0.0f;
        if (ainr_postprocess(ctx, 0, out) != AINR_OK) exit(1);
        for (i = 0; i < hop_in; ++i) e += (double)out[i] * (double)out[i];
    }
    free(in); free(out);
    return e;
}

static AinrCtx *make_pcm(const AinrConfig *cfg, const char *erb) {
    StreamMemReq req;
    if (ainr_get_mem_size(cfg, &req) != AINR_OK) return NULL;
    AinrCtx *ctx = NULL;
    AinrInfo info;
    if (ainr_init_from_erb_file(cfg, erb, pool_for((size_t)req.bytes, req.alignment), (size_t)req.bytes, &ctx, &info) != AINR_OK)
        return NULL;
    return ctx;
}

static int run_case(int sr, int fft, int n_erb, int n, const char *label) {
    AinrConfig cfg;
    StreamMemReq req;
    AinrCtx *ctx;
    AinrSpecCtx *sp;
    AinrFreqCtx *fq;
    AinrWork io;
    AinrInfo info;
    int16_t *buf;
    char erb[160];
    double e_off, e_lim, e_lim2;
    int n_bins = fft / 2 + 1;
    float *z;
    const int blocks = 24;

    snprintf(erb, sizeof(erb), "/tmp/ainr_attenlim_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, n_bins, n_erb);
    memset(&cfg, 0, sizeof(cfg));
    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = n * (fft / 2);

    /* Behaviour: default (catalog, disabled) vs 20 dB, same input; the value survives a reset. */
    ctx = make_pcm(&cfg, erb);
    if (!ctx) FAIL("init failed");
    e_off = run_energy(ctx, cfg.hop_in, blocks);
    ainr_reset(ctx);
    if (ainr_set_atten_lim(ctx, 20.0f) != AINR_OK) FAIL("set 20 dB refused");
    e_lim = run_energy(ctx, cfg.hop_in, blocks);
    ainr_reset(ctx);
    e_lim2 = run_energy(ctx, cfg.hop_in, blocks);
    if (!(e_lim > 4.0 * e_off + 1.0)) FAIL("20 dB limit left %.3g vs %.3g with it disabled", e_lim, e_off);
    if (e_lim2 != e_lim) FAIL("value lost across reset: %.6g vs %.6g", e_lim2, e_lim);
    ainr_reset(ctx);
    if (ainr_set_atten_lim(ctx, 0.0f) != AINR_OK) FAIL("set 0 dB refused");
    if (run_energy(ctx, cfg.hop_in, blocks) != e_off) FAIL("0 dB does not restore the disabled output");

    /* Refusals: NaN/Inf, a block open, NULL. A refused call changes nothing. */
    if (ainr_set_atten_lim(ctx, NAN) != AINR_ERR_NONFINITE_INPUT) FAIL("NaN not refused");
    if (ainr_set_atten_lim(ctx, INFINITY) != AINR_ERR_NONFINITE_INPUT) FAIL("Inf not refused");
    if (ainr_set_atten_lim(NULL, 6.0f) != AINR_ERR_NULL) FAIL("NULL ctx not refused");
    buf = (int16_t *)calloc((size_t)cfg.hop_in, sizeof(int16_t));
    if (ainr_preprocess(ctx, buf, &io) != AINR_OK) FAIL("pre failed");
    if (ainr_set_atten_lim(ctx, 6.0f) != AINR_ERR_SEQUENCE) FAIL("open block not refused");
    if (ainr_postprocess(ctx, -1, buf) != AINR_OK) FAIL("post failed");
    if (ainr_set_atten_lim(ctx, 6.0f) != AINR_OK) FAIL("set after post refused");
    free(buf);

    /* Same refusals on the other two families. */
    if (ainr_get_mem_size_spec(&cfg, &req) != AINR_OK) FAIL("spec size");
    sp = NULL;
    if (ainr_init_spec_from_erb_file(&cfg, erb, pool_for((size_t)req.bytes, req.alignment), (size_t)req.bytes, &sp, &info) != AINR_OK) sp = NULL;
    if (ainr_get_mem_size_freq(&cfg, &req) != AINR_OK) FAIL("freq size");
    fq = NULL;
    if (ainr_init_freq_from_erb_file(&cfg, erb, pool_for((size_t)req.bytes, req.alignment), (size_t)req.bytes, &fq, &info) != AINR_OK) fq = NULL;
    if (!sp || !fq) FAIL("spec/freq init failed");
    z = (float *)calloc((size_t)n * (size_t)n_bins * 4u, sizeof(float));
    buf = (int16_t *)calloc((size_t)cfg.hop_in, sizeof(int16_t));
    if (ainr_set_atten_lim_spec(sp, 12.0f) != AINR_OK || ainr_set_atten_lim_freq(fq, 12.0f) != AINR_OK) FAIL("spec/freq set refused");
    if (ainr_set_atten_lim_spec(sp, NAN) != AINR_ERR_NONFINITE_INPUT || ainr_set_atten_lim_freq(fq, NAN) != AINR_ERR_NONFINITE_INPUT)
        FAIL("spec/freq NaN not refused");
    if (ainr_set_atten_lim_spec(NULL, 1.0f) != AINR_ERR_NULL || ainr_set_atten_lim_freq(NULL, 1.0f) != AINR_ERR_NULL)
        FAIL("spec/freq NULL not refused");
    if (ainr_preprocess_spec(sp, buf, &io) != AINR_OK) FAIL("spec pre failed");
    if (ainr_set_atten_lim_spec(sp, 6.0f) != AINR_ERR_SEQUENCE) FAIL("spec open block not refused");
    if (ainr_postprocess_spec(sp, -1, z, z + (size_t)n * (size_t)n_bins) != AINR_OK) FAIL("spec post failed");
    if (ainr_preprocess_freq(fq, z, z + (size_t)n * (size_t)n_bins, z, z + (size_t)n * (size_t)n_bins, &io) != AINR_OK) FAIL("freq pre failed");
    if (ainr_set_atten_lim_freq(fq, 6.0f) != AINR_ERR_SEQUENCE) FAIL("freq open block not refused");
    if (ainr_postprocess_freq(fq, -1, z + (size_t)n * (size_t)n_bins * 2u, z + (size_t)n * (size_t)n_bins * 3u) != AINR_OK) FAIL("freq post failed");
    free(z); free(buf);
    unlink(erb);
    printf("%s: atten-lim setter OK (limit energy %.3g vs %.3g disabled)\n", label, e_lim, e_off);
    return 0;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, "attenlim[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, "attenlim[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, "attenlim[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, "attenlim[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_atten_lim: FAIL\n"); return 1; }
    printf("test_ainr_atten_lim: PASS\n");
    return 0;
}
