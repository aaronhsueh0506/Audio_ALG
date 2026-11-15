/* The state-generation contract the samples reload their accelerator state by, for both
 * families (PCM, _freq): StreamWork.state_generation must equal the live
 * ainr_state_generation*() at the moment it is read (the seed read right after init included), and
 * the counter must move exactly where ainr.h says:
 *
 *   +1 on ainr_reset(); +1 after each of the first mask_lookahead warm-up blocks (48 kHz);
 *   +1 on a refused commit (postprocess status 0 with the heads left NaN, or one NaN/Inf element in any head);
 *   +1 on a run status that is neither STREAM_RUN_OK nor STREAM_RUN_NOT_STARTED (the device state may be dirty);
 *   unchanged on STREAM_RUN_NOT_STARTED (no run started).
 *
 * run_status_table() closes the same block with each status next to a twin that closes it with NOT_STARTED: the output
 * is the same bypass whatever the status, the heads of a failing run are never committed, and the generation steps by
 * the rule above.
 *
 * A fault that hands the sample a stale or zero generation passes every functional test, so each
 * value is compared against the live counter after it has moved off zero. Not built under
 * NO_STDIO. */

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "ainr_test_fixture.h"

typedef enum { FAM_PCM = 0, FAM_FREQ } Family;

typedef struct Fam {
    Family family;
    AinrCtx *pcm;
    AinrFreqCtx *freq;
    AinrConfig cfg;
    int n_bins, n;
    int16_t *pcm_in, *pcm_out;
    float *re, *im, *out_re, *out_im;
} Fam;

static const char *fam_name(Family f) { return f == FAM_PCM ? "pcm" : "freq"; }

static int fam_init(Fam *f, const char *erb) {
    StreamMemReq req;
    AinrInfo info;
    void *pool;
    switch (f->family) {
    case FAM_PCM:
        if (ainr_get_mem_size(&f->cfg, &req) != AINR_OK) return 1;
        pool = alloc_aligned((size_t)req.bytes, req.alignment);
        return ainr_init_from_erb_file(&f->cfg, erb, pool, (size_t)req.bytes, &f->pcm, &info) != AINR_OK || !f->pcm;
    default:
        if (ainr_get_mem_size_freq(&f->cfg, &req) != AINR_OK) return 1;
        pool = alloc_aligned((size_t)req.bytes, req.alignment);
        return ainr_init_freq_from_erb_file(&f->cfg, erb, pool, (size_t)req.bytes, &f->freq, &info) != AINR_OK || !f->freq;
    }
}

static unsigned int fam_gen(const Fam *f) {
    return f->family == FAM_PCM ? ainr_state_generation(f->pcm) : ainr_state_generation_freq(f->freq);
}

static void fam_reset(Fam *f) {
    if (f->family == FAM_PCM) ainr_reset(f->pcm);
    else ainr_reset_freq(f->freq);
}

static int fam_pre(Fam *f, int blk, StreamWork *io) {
    size_t i, cnt = (size_t)f->n * (size_t)f->n_bins;
    if (f->family == FAM_FREQ) {
        for (i = 0; i < cnt; ++i) {
            f->re[i] = 0.01f * (float)((i * 7 + (size_t)blk * 13) % 31) - 0.15f;
            f->im[i] = 0.01f * (float)((i * 5 + (size_t)blk * 3) % 29) - 0.14f;
        }
        return ainr_preprocess_freq(f->freq, f->re, f->im, f->re, f->im, io);
    }
    for (i = 0; i < (size_t)f->cfg.hop_in; ++i) f->pcm_in[i] = (int16_t)(((i * 13 + (size_t)blk * 7) % 211) - 100);
    return ainr_preprocess(f->pcm, f->pcm_in, io);
}

static int fam_post(Fam *f, int status) {
    return f->family == FAM_PCM ? ainr_postprocess(f->pcm, status, f->pcm_out)
         : ainr_postprocess_freq(f->freq, status, f->out_re, f->out_im);
}

static void fill_heads(StreamWork *io) {
    int o;
    size_t i;
    for (o = 0; o < io->output_count; ++o)
        for (i = 0; i < io->outputs[o].elements; ++i) io->outputs[o].data[i] = (o == 0 ? 0.5f : 0.01f);
}

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "%s/%s: ", label, fam_name(f->family)); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

/* `warm` = mask_lookahead of the grid (hops after init/reset that advance the generation). */
static int run_family(Fam *f, int warm, const char *label) {
    StreamWork io;
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

    /* A run that did not start leaves it alone; a failed run moves it by one, and the next block shows it. */
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(io.state_generation == fam_gen(f), "pre before bypass: io %u != live %u", io.state_generation, fam_gen(f));
    CHECK(fam_post(f, STREAM_RUN_NOT_STARTED) == AINR_OK, "not-started post failed");
    CHECK(fam_gen(f) == g, "STREAM_RUN_NOT_STARTED moved the generation %u -> %u", g, fam_gen(f));
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(fam_post(f, STREAM_RUN_FAILED) == AINR_OK, "failed-run post failed");
    CHECK(fam_gen(f) == g + 1u, "STREAM_RUN_FAILED: %u -> %u, want +1", g, fam_gen(f));
    g = fam_gen(f);
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(io.state_generation == g, "pre after a failed run: io %u != live %u", io.state_generation, g);
    fill_heads(&io);
    CHECK(fam_post(f, 0) == AINR_OK, "post after a failed run failed");
    CHECK(fam_gen(f) == g, "a commit after a failed run moved the generation");

    /* A refused commit (status 0, heads still NaN) moves it by one, and the next block shows it. */
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(fam_post(f, 0) == AINR_OK, "refused-commit post failed");
    CHECK(fam_gen(f) == g + 1u, "refused commit: %u -> %u, want +1", g, fam_gen(f));
    g = fam_gen(f);
    CHECK(fam_pre(f, blk++, &io) == AINR_OK, "pre failed");
    CHECK(io.state_generation == g, "pre after refusal: io %u != live %u", io.state_generation, g);
    fill_heads(&io);
    CHECK(fam_post(f, 0) == AINR_OK, "post failed");

    /* A float head from the accelerator can carry one non-finite element among finite ones: any head, either end,
     * NaN or Inf, is a refused commit. */
    {
        static const float bad_value[3] = { NAN, INFINITY, -INFINITY };
        int head, v, at;
        for (head = 0; head < io.output_count; ++head)
            for (v = 0; v < 3; ++v)
                for (at = 0; at < 2; ++at) {
                    StreamWork pw;
                    size_t idx;
                    g = fam_gen(f);
                    CHECK(fam_pre(f, blk++, &pw) == AINR_OK, "pre failed");
                    fill_heads(&pw);
                    idx = at ? pw.outputs[head].elements - 1u : 0u;
                    pw.outputs[head].data[idx] = bad_value[v];
                    CHECK(fam_post(f, 0) == AINR_OK, "poisoned post failed");
                    CHECK(fam_gen(f) == g + 1u, "head %d element %zu = %g: generation %u -> %u, want +1", head, idx,
                          (double)bad_value[v], g, fam_gen(f));
                    g = fam_gen(f);
                    CHECK(fam_pre(f, blk++, &pw) == AINR_OK, "pre failed");
                    fill_heads(&pw);
                    CHECK(fam_post(f, 0) == AINR_OK, "clean post failed");
                    CHECK(fam_gen(f) == g, "head %d: a finite block after the refusal moved the generation", head);
                }
    }
    return 0;
}

static void fill_heads_with(StreamWork *io, float mask, float other) {
    int o;
    size_t i;
    for (o = 0; o < io->output_count; ++o)
        for (i = 0; i < io->outputs[o].elements; ++i) io->outputs[o].data[i] = (o == 0 ? mask : other);
}

static int out_equal(const Fam *a, const Fam *b) {
    size_t cnt = (size_t)a->n * (size_t)a->n_bins;
    if (a->family == FAM_PCM)
        return memcmp(a->pcm_out, b->pcm_out, (size_t)a->cfg.hop_in * sizeof(int16_t)) == 0;
    return memcmp(a->out_re, b->out_re, cnt * sizeof(float)) == 0 && memcmp(a->out_im, b->out_im, cnt * sizeof(float)) == 0;
}

static unsigned int status_step(int status) { return (status == STREAM_RUN_OK || status == STREAM_RUN_NOT_STARTED) ? 0u : 1u; }

/* One status against its NOT_STARTED twin: same input, same history. A failing status is the twin's bypass output and
 * moves the generation by status_step(); status 0 commits the heads (a different output) and leaves it. */
static int run_status_one(Fam *f, Fam *twin, int warm, int status, const char *label) {
    StreamWork io, tio;
    unsigned int g, tg;
    int blk = 0, w, k;

    for (w = 0; w < warm + 2; ++w) {   /* the warm-up blocks, then two committed ones */
        CHECK(fam_pre(f, blk, &io) == AINR_OK && fam_pre(twin, blk, &tio) == AINR_OK, "lead-in pre failed");
        fill_heads(&io);
        fill_heads(&tio);
        CHECK(fam_post(f, 0) == AINR_OK && fam_post(twin, 0) == AINR_OK, "lead-in post failed");
        ++blk;
    }
    CHECK(fam_gen(f) == fam_gen(twin), "lead-in generations differ");
    g = fam_gen(f);
    tg = fam_gen(twin);
    for (k = 0; k < 3; ++k) {   /* the status block, then two more committed blocks on both */
        CHECK(fam_pre(f, blk, &io) == AINR_OK && fam_pre(twin, blk, &tio) == AINR_OK, "pre failed");
        CHECK(io.state_generation == fam_gen(f) && tio.state_generation == fam_gen(twin), "io generation is not live");
        if (k == 0) {
            fill_heads_with(&io, 0.9f, 0.7f);   /* a run that failed may have written anything */
            fill_heads_with(&tio, 0.2f, 0.1f);
            CHECK(fam_post(f, status) == AINR_OK && fam_post(twin, STREAM_RUN_NOT_STARTED) == AINR_OK, "post failed");
            CHECK(fam_gen(f) == g + status_step(status), "status %d: generation %u -> %u, want +%u", status, g,
                  fam_gen(f), status_step(status));
            CHECK(fam_gen(twin) == tg, "twin (NOT_STARTED) moved the generation");
            CHECK(status == 0 ? !out_equal(f, twin) : out_equal(f, twin),
                  "status %d: output %s the bypass", status, status == 0 ? "equals" : "differs from");
        } else {
            fill_heads(&io);
            fill_heads(&tio);
            CHECK(fam_post(f, 0) == AINR_OK && fam_post(twin, 0) == AINR_OK, "follow-up post failed");
            CHECK(fam_gen(f) == g + status_step(status), "status %d: a later commit moved the generation", status);
            if (status != 0)
                CHECK(out_equal(f, twin), "status %d: block %d after the failed one differs from the twin's", status, k);
        }
        ++blk;
    }
    return 0;
}

static int run_status_table(int sr, int fft, int n_erb, int n, int warm, const char *label) {
    static const int statuses[] = { STREAM_RUN_NOT_STARTED, STREAM_RUN_FAILED, 1, 5, 7, -2,
                                    STREAM_RUN_NOT_STARTED + 1, STREAM_RUN_NOT_STARTED - 1, INT_MAX, INT_MIN, 0 };
    int fi, si, rc = 0;
    char erb[160];
    int n_bins = fft / 2 + 1;
    snprintf(erb, sizeof(erb), "/tmp/ainr_runstatus_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, n_bins, n_erb);
    for (fi = 0; fi < 2; ++fi)
        for (si = 0; si < (int)(sizeof(statuses) / sizeof(statuses[0])); ++si) {
            Fam fs[2];
            size_t cnt = (size_t)n * (size_t)n_bins;
            int j;
            for (j = 0; j < 2; ++j) {
                memset(&fs[j], 0, sizeof(fs[j]));
                fs[j].family = (Family)fi;
                fs[j].cfg.sr = sr; fs[j].cfg.fft = fft; fs[j].cfg.hop_in = n * (fft / 2);
                fs[j].n_bins = n_bins; fs[j].n = n;
                fs[j].pcm_in = (int16_t *)calloc((size_t)fs[j].cfg.hop_in, sizeof(int16_t));
                fs[j].pcm_out = (int16_t *)calloc((size_t)fs[j].cfg.hop_in, sizeof(int16_t));
                fs[j].re = (float *)calloc(cnt, sizeof(float)); fs[j].im = (float *)calloc(cnt, sizeof(float));
                fs[j].out_re = (float *)calloc(cnt, sizeof(float)); fs[j].out_im = (float *)calloc(cnt, sizeof(float));
                if (fam_init(&fs[j], erb)) { fprintf(stderr, "%s: init failed\n", label); return 1; }
            }
            if (run_status_one(&fs[0], &fs[1], warm, statuses[si], label)) rc = 1;
        }
    unlink(erb);
    if (!rc) printf("%s: run status table OK (%d statuses x 2 families)\n", label,
                    (int)(sizeof(statuses) / sizeof(statuses[0])));
    return rc;
}

static int run_case(int sr, int fft, int n_erb, int n, int warm, const char *label) {
    int fi, rc = 0;
    char erb[160];
    int n_bins = fft / 2 + 1;
    snprintf(erb, sizeof(erb), "/tmp/ainr_stategen_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, n_bins, n_erb);
    for (fi = 0; fi < 2; ++fi) {
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
    rc |= run_status_table(48000, 1024, 32, 1, 1, "runstatus[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, 0, "stategen[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, 0, "stategen[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, 1, "stategen[full 48k/1024 N=1]");
    rc |= run_status_table(16000, 256, 24, 1, 0, "runstatus[full 16k/256 N=1]");
    rc |= run_status_table(16000, 256, 24, 8, 0, "runstatus[full 16k/256 N=8]");
    rc |= run_status_table(48000, 1024, 32, 1, 1, "runstatus[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_state_generation: FAIL\n"); return 1; }
    printf("test_ainr_state_generation: PASS\n");
    return 0;
}
