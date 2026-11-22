/* The product API at N > 1 against N == 1, mask_lookahead 0 model (id 3), one stateful
 * stand-in accelerator: PCM and _freq equal hop for hop, bit for bit, through a failed run, a non-finite head and a reset; N + 2
 * frame windows, N heads, queue_frames 0, N times fewer jobs; a wrong window slice is detected. Rows 16 kHz N 8, 32 kHz N 4, 48 kHz N 2. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr.h"
#include "ainr_test_fixture.h"
#include "graph_io_dfn2.h"

#define STATE_N 8
#define BLOCKS 20

typedef struct Row { int sr, fft, n_erb, n; } Row;

static const Row ROWS[] = {
    { 16000, 256, 24, 8 },
    { 32000, 512, 28, 4 },
    { 48000, 1024, 32, 2 },
};

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Dev {
    float state[STATE_N];
    unsigned seen;
    int synced, dirty, jobs, slice_shift;
} Dev;

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

/* mode: 0 ok, 1 failed run (no heads, state moved), 2 a non-finite head. Returns the run status. */
static int device(Dev *dev, const StreamWork *w, int mode) {
    const int frames = (int)w->outputs[0].h, n_erb = (int)w->outputs[0].w;
    int i, e;
    size_t k;
    if (!dev->synced || w->state_generation != dev->seen || dev->dirty) {
        memset(dev->state, 0, sizeof dev->state);
        dev->seen = w->state_generation;
        dev->synced = 1;
        dev->dirty = 0;
    }
    ++dev->jobs;
    for (i = 0; i < frames; ++i) {
        const int at = (i + dev->slice_shift) % frames;
        const float *win = w->inputs[0].data + (size_t)at * (size_t)n_erb;
        float m = 0.0f;
        for (e = 0; e < 3 * n_erb; ++e) m += win[e];
        m /= (float)(3 * n_erb);
        for (e = 0; e < STATE_N; ++e) dev->state[e] = 0.9f * dev->state[e] + 0.1f * m + 0.01f * (float)e;
        if (mode == 1) continue;
        for (k = 0; k < (size_t)n_erb; ++k)
            w->outputs[0].data[(size_t)i * (size_t)n_erb + k] = sigm(0.3f * m + dev->state[k % STATE_N] - 0.2f);
        {
            const size_t per = w->outputs[1].elements / (size_t)frames;
            for (k = 0; k < per; ++k)
                w->outputs[1].data[(size_t)i * per + k] = 0.05f * sinf(0.37f * (float)k + dev->state[k % STATE_N]);
            w->outputs[2].data[i] = sigm(dev->state[0] - 0.5f);
        }
    }
    if (mode == 1) { dev->dirty = 1; return STREAM_RUN_FAILED; }
    if (mode == 2) w->outputs[2].data[frames - 1] = (float)NAN;
    return STREAM_RUN_OK;
}

static int event_of(int b) { return b == 5 ? 1 : b == 9 ? 2 : b == 13 ? 3 : 0; }   /* 1 failed, 2 non-finite, 3 reset */

typedef struct Out { int16_t *pcm; float *spec; int queue, jobs; } Out;

static void make_pcm(int16_t *x, int hops, int first, int hop) {
    int i, k;
    for (i = 0; i < hops; ++i)
        for (k = 0; k < hop; ++k) {
            const long t = (long)(first + i) * hop + k;
            x[(size_t)i * (size_t)hop + (size_t)k] = (int16_t)(8000.0 * sin(0.013 * (double)t) + 2000.0 * sin(0.31 * (double)t + 1.0));
        }
}

static void make_spec(float *re, float *im, float *ar, float *ai, int hops, int first, int nb) {
    int i, k;
    for (i = 0; i < hops; ++i)
        for (k = 0; k < nb; ++k) {
            const int h = first + i;
            const size_t at = (size_t)i * (size_t)nb + (size_t)k;
            re[at] = 0.4f * sinf(0.013f * (float)(h * nb + k)) + 0.1f * cosf(0.31f * (float)(h + k));
            im[at] = 0.3f * cosf(0.017f * (float)(h * nb + k)) - 0.05f * sinf(0.7f * (float)(h - k));
            ar[at] = h % 3 == 0 ? 0.5f * re[at] + 0.01f : re[at];
            ai[at] = h % 3 == 0 ? 0.5f * im[at] : im[at];
        }
}

/* `n` frames per call (the block stream, or 1: the one-frame stream running N hops per block of the other). */
static void run(const Row *r, int n, int freq, int slice_shift, int total_n, Out *out) {
    AinrConfig cfg = { r->sr, r->fft, n * (r->fft / 2), NULL, NULL, 0 };
    const int hop = r->fft / 2, nb = hop + 1;
    StreamMemReq req;
    AinrInfo info;
    AinrCtx *pcm = NULL;
    AinrFreqCtx *fq = NULL;
    void *pool;
    Dev dev;
    int16_t *pin = (int16_t *)malloc((size_t)hop * (size_t)n * sizeof(int16_t));
    float *x = (float *)malloc((size_t)nb * (size_t)n * 6u * sizeof(float));
    float *re = x, *im = x + (size_t)nb * n, *ar = x + 2u * (size_t)nb * n, *ai = x + 3u * (size_t)nb * n;
    float *ore = x + 4u * (size_t)nb * n, *oim = x + 5u * (size_t)nb * n;
    const int calls = BLOCKS * (total_n / n);
    int c;

    memset(&dev, 0, sizeof dev);
    dev.slice_shift = n > 1 ? slice_shift : 0;
    ErbTables erb = erb_tables_synthetic(nb, r->n_erb);
    CHECK((freq ? ainr_get_mem_size_freq(&cfg, &req) : ainr_get_mem_size(&cfg, &req)) == AINR_OK, "%d Hz N %d: sizing", r->sr, n);
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    CHECK((freq ? ainr_init_freq_t(&cfg, &erb, pool, (size_t)req.bytes, &fq, &info)
                : ainr_init_t(&cfg, &erb, pool, (size_t)req.bytes, &pcm, &info)) == AINR_OK,
          "%d Hz N %d: init", r->sr, n);
    CHECK(info.queue_frames == 0, "%d Hz N %d: queue_frames %d, want 0 (L + D)", r->sr, n, info.queue_frames);
    CHECK(info.mask_lookahead == 0 && graph_io_dfn2_lookup(GRAPH_IO_DFN2_FULL, r->sr, r->fft, n, info.mask_lookahead) != NULL,
          "%d Hz N %d: mask_lookahead %d or no graph row", r->sr, n, info.mask_lookahead);
    out->queue = info.queue_frames;
    out->pcm = freq ? NULL : (int16_t *)calloc((size_t)hop * (size_t)total_n * BLOCKS, sizeof(int16_t));
    out->spec = freq ? (float *)calloc((size_t)nb * 2u * (size_t)total_n * BLOCKS, sizeof(float)) : NULL;

    for (c = 0; c < calls; ++c) {
        StreamWork w;
        const int first_hop = c * n;
        const int ev = event_of(first_hop / total_n);
        int rc, k, i;
        if (ev == 3 && first_hop % total_n == 0) { if (freq) ainr_reset_freq(fq); else ainr_reset(pcm); }
        if (freq) {
            make_spec(re, im, ar, ai, n, first_hop, nb);
            /* the product takes audio-scale spectra: the spectra above are used as they are */
            rc = ainr_preprocess_freq(fq, re, im, ar, ai, &w);
        } else {
            make_pcm(pin, n, first_hop, hop);
            rc = ainr_preprocess(pcm, pin, &w);
        }
        CHECK(rc == AINR_OK, "%d Hz N %d: pre call %d rc %d", r->sr, n, c, rc);
        if (rc != AINR_OK) break;
        CHECK(w.action == STREAM_WORK_SUBMIT && w.input_count == 2 && w.output_count == 3 &&
              w.inputs[0].h == (unsigned)(n + 2) && w.outputs[0].h == (unsigned)n && w.outputs[2].h == (unsigned)n,
              "%d Hz N %d: work shape (action %d, window %u, heads %u)", r->sr, n, (int)w.action, w.inputs[0].h, w.outputs[0].h);
        k = device(&dev, &w, ev == 1 ? 1 : ev == 2 ? 2 : 0);
        if (freq) {
            rc = ainr_postprocess_freq(fq, k, ore, oim);
            for (i = 0; i < n; ++i) {
                memcpy(out->spec + (size_t)(first_hop + i) * 2u * (size_t)nb, ore + (size_t)i * (size_t)nb, (size_t)nb * sizeof(float));
                memcpy(out->spec + ((size_t)(first_hop + i) * 2u + 1u) * (size_t)nb, oim + (size_t)i * (size_t)nb, (size_t)nb * sizeof(float));
            }
        } else {
            rc = ainr_postprocess(pcm, k, out->pcm + (size_t)first_hop * (size_t)hop);
        }
        CHECK(rc == AINR_OK, "%d Hz N %d: post call %d rc %d", r->sr, n, c, rc);
    }
    out->jobs = dev.jobs;
    free(pin); free(x); free(pool);
    erb_tables_free(&erb);
}

static void run_row(const Row *r) {
    const int hop = r->fft / 2, nb = hop + 1, N = r->n;
    int freq;
    for (freq = 0; freq < 2; ++freq) {
        Out one, blk, bad;
        const size_t items = (size_t)BLOCKS * (size_t)N;
        const size_t total = freq ? items * 2u * (size_t)nb * sizeof(float) : items * (size_t)hop * sizeof(int16_t);
        size_t i, nz = 0, count = total / (freq ? sizeof(float) : sizeof(int16_t));
        run(r, 1, freq, 0, N, &one);
        run(r, N, freq, 0, N, &blk);
        run(r, N, freq, 1, N, &bad);
        CHECK(memcmp(freq ? (void *)one.spec : (void *)one.pcm, freq ? (void *)blk.spec : (void *)blk.pcm, total) == 0,
              "%d Hz N %d %s: the block stream differs from the N == 1 stream", r->sr, N, freq ? "_freq" : "PCM");
        CHECK(blk.jobs == BLOCKS && one.jobs == BLOCKS * N, "%d Hz N %d: jobs %d vs %d, want %d vs %d", r->sr, N, blk.jobs,
              one.jobs, BLOCKS, BLOCKS * N);
        for (i = 0; i < count; ++i) nz += freq ? one.spec[i] != 0.0f : one.pcm[i] != 0;
        CHECK(nz > count / 2, "%d Hz N %d: the N == 1 output is mostly silent", r->sr, N);
        CHECK(memcmp(freq ? (void *)one.spec : (void *)one.pcm, freq ? (void *)bad.spec : (void *)bad.pcm, total) != 0,
              "%d Hz N %d %s: a stand-in on the wrong window slice was not detected", r->sr, N, freq ? "_freq" : "PCM");
        free(one.pcm); free(one.spec); free(blk.pcm); free(blk.spec); free(bad.pcm); free(bad.spec);
    }
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof ROWS / sizeof ROWS[0]; ++i) run_row(&ROWS[i]);
    if (g_fail) { printf("test_ainr_blocks: FAIL\n"); return 1; }
    printf("test_ainr_blocks: PASS (16 kHz N 8, 32 kHz N 4, 48 kHz N 2: the block stream equals the N == 1 stream, PCM and _freq)\n");
    return 0;
}
