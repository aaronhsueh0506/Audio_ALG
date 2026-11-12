/* The state-generation contract the samples reload their accelerator state by, for all three
 * families (PCM, _spec, _freq): AinrWork.state_generation must equal the live
 * ainr_state_generation*() at the moment it is read (the seed read right after init included), and
 * the counter must move exactly where ainr.h says:
 *
 *   +1 on ainr_reset(); +1 after each of the first mask_lookahead warm-up blocks (48 kHz);
 *   +1 on a refused commit (postprocess status 0 with the heads left NaN);
 *   unchanged on a caller bypass (postprocess status != 0).
 *
 * A fault that hands the sample a stale or zero generation passes every functional test, so each
 * value is compared against the live counter after it has moved off zero. Not built under
 * NO_STDIO. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"

typedef enum { FAM_PCM = 0, FAM_SPEC, FAM_FREQ } Family;

typedef struct Fam {
    Family family;
    AinrCtx *pcm;
    AinrSpecCtx *spec;
    AinrFreqCtx *freq;
    AinrConfig cfg;
    int n_bins, n;
    int16_t *pcm_in, *pcm_out;
    float *re, *im, *out_re, *out_im;
} Fam;

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

static void *alloc_aligned(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) exit(1);
    return p;
}

static const char *fam_name(Family f) { return f == FAM_PCM ? "pcm" : f == FAM_SPEC ? "spec" : "freq"; }

static int fam_init(Fam *f, const char *erb) {
    StreamMemReq req;
    AinrInfo info;
    void *pool;
    switch (f->family) {
    case FAM_PCM:
        if (ainr_get_mem_size(&f->cfg, &req) != AINR_OK) return 1;
        pool = alloc_aligned((size_t)req.bytes, req.alignment);
        return ainr_init_from_erb_file(&f->cfg, erb, pool, (size_t)req.bytes, &f->pcm, &info) != AINR_OK || !f->pcm;
    case FAM_SPEC:
        if (ainr_get_mem_size_spec(&f->cfg, &req) != AINR_OK) return 1;
        pool = alloc_aligned((size_t)req.bytes, req.alignment);
        return ainr_init_spec_from_erb_file(&f->cfg, erb, pool, (size_t)req.bytes, &f->spec, &info) != AINR_OK || !f->spec;
    default:
        if (ainr_get_mem_size_freq(&f->cfg, &req) != AINR_OK) return 1;
        pool = alloc_aligned((size_t)req.bytes, req.alignment);
        return ainr_init_freq_from_erb_file(&f->cfg, erb, pool, (size_t)req.bytes, &f->freq, &info) != AINR_OK || !f->freq;
    }
}

static unsigned int fam_gen(const Fam *f) {
    return f->family == FAM_PCM ? ainr_state_generation(f->pcm)
         : f->family == FAM_SPEC ? ainr_state_generation_spec(f->spec)
         : ainr_state_generation_freq(f->freq);
}

static void fam_reset(Fam *f) {
    if (f->family == FAM_PCM) ainr_reset(f->pcm);
    else if (f->family == FAM_SPEC) ainr_reset_spec(f->spec);
    else ainr_reset_freq(f->freq);
}

static int fam_pre(Fam *f, int blk, AinrWork *io) {
    size_t i, cnt = (size_t)f->n * (size_t)f->n_bins;
    if (f->family == FAM_FREQ) {
        for (i = 0; i < cnt; ++i) {
            f->re[i] = 0.01f * (float)((i * 7 + (size_t)blk * 13) % 31) - 0.15f;
            f->im[i] = 0.01f * (float)((i * 5 + (size_t)blk * 3) % 29) - 0.14f;
        }
        return ainr_preprocess_freq(f->freq, f->re, f->im, f->re, f->im, io);
    }
    for (i = 0; i < (size_t)f->cfg.hop_in; ++i) f->pcm_in[i] = (int16_t)(((i * 13 + (size_t)blk * 7) % 211) - 100);
    return f->family == FAM_PCM ? ainr_preprocess(f->pcm, f->pcm_in, io)
                                        : ainr_preprocess_spec(f->spec, f->pcm_in, io);
}

static int fam_post(Fam *f, int status) {
    return f->family == FAM_PCM ? ainr_postprocess(f->pcm, status, f->pcm_out)
         : f->family == FAM_SPEC ? ainr_postprocess_spec(f->spec, status, f->out_re, f->out_im)
         : ainr_postprocess_freq(f->freq, status, f->out_re, f->out_im);
}

static void fill_heads(AinrWork *io) {
    int o;
    size_t i;
    for (o = 0; o < io->output_count; ++o)
        for (i = 0; i < io->outputs[o].elements; ++i) io->outputs[o].data[i] = (o == 0 ? 0.5f : 0.01f);
}

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "%s/%s: ", label, fam_name(f->family)); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

/* `warm` = mask_lookahead of the grid (hops after init/reset that advance the generation). */
static int run_family(Fam *f, int warm, const char *label) {
    AinrWork io;
    unsigned int g, seed;
    int blk = 0, w;

    /* The seed a sample reads right after init is what the first block reports. */
    seed = fam_gen(f);
    CHECK(fam_pre(f, blk, &io) == AINR_OK, "first pre after init failed");
    CHECK(io.state_generation == seed, "first block: io %u != seed %u read after init", io.state_generation, seed);

    /* Reset moves the counter off its initial value; everything below is compared against a live
     * value that is not zero. */
    g = fam_gen(f);
    fam_reset(f);
    CHECK(fam_gen(f) == g + 1u, "reset: %u -> %u, want +1", g, fam_gen(f));
    CHECK(fam_gen(f) != 0u, "reset: generation still 0");

    /* Warm-up blocks: the sample sees the live generation, +1 per warm-up hop once it commits. */
    for (w = 0; w < warm; ++w) {
        g = fam_gen(f);
        CHECK(fam_pre(f, blk++, &io) == AINR_OK, "warm-up pre failed");
        CHECK(io.state_generation == fam_gen(f), "warm-up pre: io %u != live %u", io.state_generation, fam_gen(f));
        fill_heads(&io);
        CHECK(fam_post(f, 0) == AINR_OK, "warm-up post failed");
        CHECK(fam_gen(f) == g + 1u, "warm-up block %d: %u -> %u, want +1", w, g, fam_gen(f));
    }

    /* Steady state: commit leaves it alone; io carries the live value. */
    g = fam_gen(f);
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(io.state_generation == fam_gen(f) && io.state_generation == g && g != 0u,
          "pre: io %u, live %u, expected %u (nonzero)", io.state_generation, fam_gen(f), g);
    fill_heads(&io);
    CHECK(fam_post(f, 0) == AINR_OK, "commit post failed");
    CHECK(fam_gen(f) == g, "good commit moved the generation %u -> %u", g, fam_gen(f));

    /* A caller bypass (failed run) does not move it. */
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(io.state_generation == fam_gen(f), "pre before bypass: io %u != live %u", io.state_generation, fam_gen(f));
    CHECK(fam_post(f, -1) == AINR_OK, "bypass post failed");
    CHECK(fam_gen(f) == g, "caller bypass moved the generation %u -> %u", g, fam_gen(f));

    /* A refused commit (status 0, heads still NaN) moves it by one, and the next block shows it. */
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(fam_post(f, 0) == AINR_OK, "refused-commit post failed");
    CHECK(fam_gen(f) == g + 1u, "refused commit: %u -> %u, want +1", g, fam_gen(f));
    g = fam_gen(f);
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(io.state_generation == g, "pre after refusal: io %u != live %u", io.state_generation, g);
    fill_heads(&io);
    CHECK(fam_post(f, 0) == AINR_OK, "post failed");
    return 0;
}

static int run_case(int sr, int fft, int n_erb, int n, int warm, const char *label) {
    int fi, rc = 0;
    char erb[160];
    int n_bins = fft / 2 + 1;
    snprintf(erb, sizeof(erb), "/tmp/ainr_stategen_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, n_bins, n_erb);
    for (fi = 0; fi < 3; ++fi) {
        Fam f;
        size_t cnt = (size_t)n * (size_t)n_bins;
        memset(&f, 0, sizeof(f));
        f.family = (Family)fi;
        f.cfg.sr = sr; f.cfg.fft = fft; f.cfg.hop_in = n * (fft / 2);
        f.n_bins = n_bins; f.n = n;
        f.pcm_in = (int16_t *)calloc((size_t)f.cfg.hop_in, sizeof(int16_t));
        f.pcm_out = (int16_t *)calloc((size_t)f.cfg.hop_in, sizeof(int16_t));
        f.re = (float *)calloc(cnt, sizeof(float)); f.im = (float *)calloc(cnt, sizeof(float));
        f.out_re = (float *)calloc(cnt, sizeof(float)); f.out_im = (float *)calloc(cnt, sizeof(float));
        if (fam_init(&f, erb)) { fprintf(stderr, "%s/%s: init failed\n", label, fam_name(f.family)); rc = 1; continue; }
        if (run_family(&f, warm, label)) rc = 1;
        else printf("%s/%s: state generation contract OK\n", label, fam_name(f.family));
    }
    unlink(erb);
    return rc;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, 1, "stategen[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, 0, "stategen[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, 0, "stategen[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, 1, "stategen[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_state_generation: FAIL\n"); return 1; }
    printf("test_ainr_state_generation: PASS\n");
    return 0;
}
