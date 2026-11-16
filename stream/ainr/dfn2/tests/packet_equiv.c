/* The packet entries (N > 1, mask_lookahead 0) against N one-frame hops of one stateful stand-in accelerator (head i from window
 * slice i, state carried): PCM and spectrum output equal hop for hop, bit for bit, through a failed run, a non-finite head and a reset
 * on packet boundaries, with N times fewer jobs; a stand-in on the wrong slice must differ. Four grids, 16 to 48 kHz. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "erb_fixture.h"

#define STATE_N 8
#define PACKETS 24

typedef struct Row {
    int sr, fft, n_erb, df_lookahead, n;
} Row;

static const Row ROWS[] = {
    { 16000, 256, 24, 0, 8 },
    { 16000, 256, 24, 1, 4 },    /* a deep-filter lookahead: the first packet carries fewer valid hops */
    { 48000, 1024, 32, 0, 2 },
    { 32000, 512, 28, 0, 4 },
};

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Device {
    float state[STATE_N];
    unsigned seen;
    int dirty, synced;
    int slice_shift;
    int jobs;
} Device;

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

/* mode: 0 ok, 1 failed run (state moved, no head written), 2 a non-finite head (status 0). */
static int device_run(Device *dev, const DFN2HopIO *io, const DFN2Grid *g, int mode) {
    const int frames = g->frames_per_call, n_erb = g->n_erb;
    int i, e;
    size_t k;
    if (!dev->synced || io->state_generation != dev->seen || dev->dirty) {
        memset(dev->state, 0, sizeof dev->state);
        dev->seen = io->state_generation;
        dev->synced = 1;
        dev->dirty = 0;
    }
    ++dev->jobs;
    for (i = 0; i < frames; ++i) {
        const int at = (i + dev->slice_shift) % frames;
        const float *w = io->inputs.erb_window + (size_t)at * (size_t)n_erb;   /* slice at: three frames */
        float m = 0.0f;
        for (e = 0; e < 3 * n_erb; ++e) m += w[e];
        m /= (float)(3 * n_erb);
        for (e = 0; e < STATE_N; ++e) dev->state[e] = 0.9f * dev->state[e] + 0.1f * m + 0.01f * (float)e;
        if (mode == 1) continue;
        for (k = 0; k < (size_t)n_erb; ++k)
            io->outputs.erb_mask[(size_t)i * (size_t)n_erb + k] = sigm(0.3f * m + dev->state[k % STATE_N] - 0.2f);
#ifndef DFN2_MASK_ONLY
        {
            const size_t per = io->outputs.coefs_elements / (size_t)frames;
            for (k = 0; k < per; ++k)
                io->outputs.coefs[(size_t)i * per + k] = 0.05f * sinf(0.37f * (float)k + dev->state[k % STATE_N]);
            io->outputs.alpha[i] = sigm(dev->state[0] - 0.5f);
        }
#endif
    }
    if (mode == 1) { dev->dirty = 1; return STREAM_RUN_FAILED; }
    if (mode == 2) {
#ifdef DFN2_MASK_ONLY
        io->outputs.erb_mask[0] = (float)NAN;
#else
        io->outputs.alpha[frames - 1] = (float)NAN;
#endif
    }
    return STREAM_RUN_OK;
}

/* The event of packet pk: 0 none, 1 failed run, 2 a non-finite head, 3 a reset before the packet. */
static int event_of(int pk) { return pk == 6 ? 1 : pk == 11 ? 2 : pk == 15 ? 3 : 0; }

typedef struct Out {
    int16_t *pcm;       /* PCM entries: hop * N * PACKETS samples */
    float *spec;        /* spectrum entries: n_bins * 2 * N * PACKETS floats, re then im per hop */
    int valid_hops;
    int jobs;
} Out;

static void input_pcm(int16_t *x, int n, int first_hop, int hop) {
    int i, k;
    for (i = 0; i < n; ++i)
        for (k = 0; k < hop; ++k) {
            const long t = (long)(first_hop + i) * hop + k;
            x[(size_t)i * (size_t)hop + (size_t)k] =
                (int16_t)(8000.0 * sin(0.013 * (double)t) + 2000.0 * sin(0.31 * (double)t + 1.0));
        }
}

static void input_spec(float *re, float *im, float *ar, float *ai, int n, int first_hop, int nb) {
    int i, k;
    for (i = 0; i < n; ++i)
        for (k = 0; k < nb; ++k) {
            const int h = first_hop + i;
            const size_t at = (size_t)i * (size_t)nb + (size_t)k;
            re[at] = 0.4f * sinf(0.013f * (float)(h * nb + k)) + 0.1f * cosf(0.31f * (float)(h + k));
            im[at] = 0.3f * cosf(0.017f * (float)(h * nb + k)) - 0.05f * sinf(0.7f * (float)(h - k));
            if (h % 3 == 0) { ar[at] = 0.5f * re[at] + 0.01f; ai[at] = 0.5f * im[at]; }
            else { ar[at] = re[at]; ai[at] = im[at]; }
        }
}

/* One stream of PACKETS packets of N hops: `block` 1 runs packets of N hops, 0 runs N one-frame hops per packet.
 * `freq` selects the spectrum entries. The device's slice_shift applies to packets only. */
static int run(const DFN2Grid *g, const float *fwd, const float *inv, int block, int freq, int slice_shift, Out *out) {
    const int N = g->frames_per_call, hop = g->hop, nb = g->n_bins;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    DFN2Stream *s;
    void *pool = NULL;
    Device dev;
    int16_t *pcm_in = (int16_t *)malloc((size_t)hop * (size_t)N * sizeof(int16_t));
    float *x = (float *)malloc((size_t)nb * (size_t)N * 6u * sizeof(float));
    float *re = x, *im = x + (size_t)nb * N, *ar = x + 2u * (size_t)nb * N, *ai = x + 3u * (size_t)nb * N;
    float *ore = x + 4u * (size_t)nb * N, *oim = x + 5u * (size_t)nb * N;
    int pk, i;

    memset(&dev, 0, sizeof dev);
    dev.slice_shift = block ? slice_shift : 0;
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = *g;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    cfg.freq_only = freq;
    if (!block) cfg.grid.frames_per_call = 1;
    if (cfg.grid.frames_per_call == 1) {   /* the one-frame grid of the same signal grid */
        DFN2GridSpec spec;
        memset(&spec, 0, sizeof spec);
        spec.sample_rate = g->sample_rate; spec.n_fft = g->n_fft; spec.n_erb = g->n_erb; spec.df_bins = g->df_bins;
        spec.df_order = g->df_order; spec.mask_lookahead = g->mask_lookahead; spec.df_lookahead = g->df_lookahead;
        spec.frames_per_call = 1;
        if (dfn2_grid_make(&cfg.grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid\n"); exit(2); }
    }
    if (dfn2_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(2);
    s = dfn2_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) { fprintf(stderr, "init\n"); exit(2); }

    out->pcm = freq ? NULL : (int16_t *)calloc((size_t)hop * (size_t)N * PACKETS, sizeof(int16_t));
    out->spec = freq ? (float *)calloc((size_t)nb * 2u * (size_t)N * PACKETS, sizeof(float)) : NULL;
    out->valid_hops = 0;

    for (pk = 0; pk < PACKETS; ++pk) {
        const int ev = event_of(pk);
        if (ev == 3) dfn2_stream_reset(s);
        if (block) {
            DFN2HopIO io;
            int rc = 0, valid = 0;
            if (freq) {
                input_spec(re, im, ar, ai, N, pk * N, nb);
                CHECK(dfn2_stream_pre_process_packet_freq_dual(s, re, im, ar, ai, &io) == 1, "pre packet %d", pk);
            } else {
                input_pcm(pcm_in, N, pk * N, hop);
                CHECK(dfn2_stream_pre_process_packet(s, pcm_in, &io) == 1, "pre packet %d", pk);
            }
            CHECK(io.job_ready == 1, "packet %d has no job", pk);
            rc = device_run(&dev, &io, &cfg.grid, ev == 1 ? 1 : ev == 2 ? 2 : 0);
            if (freq) {
                CHECK(dfn2_stream_post_process_packet_freq(s, rc, ore, oim, &valid) == 0, "post packet %d", pk);
                for (i = 0; i < N; ++i) {
                    memcpy(out->spec + (size_t)(pk * N + i) * 2u * (size_t)nb, ore + (size_t)i * (size_t)nb, (size_t)nb * sizeof(float));
                    memcpy(out->spec + ((size_t)(pk * N + i) * 2u + 1u) * (size_t)nb, oim + (size_t)i * (size_t)nb, (size_t)nb * sizeof(float));
                }
            } else {
                valid = dfn2_stream_post_process_packet(s, rc, out->pcm + (size_t)pk * (size_t)N * (size_t)hop);
                CHECK(valid >= 0, "post packet %d", pk);
            }
            out->valid_hops += valid;
        } else {
            for (i = 0; i < N; ++i) {
                DFN2HopIO io;
                int rc = 0, valid = 0;
                const int h = pk * N + i;
                if (freq) {
                    input_spec(re, im, ar, ai, 1, h, nb);
                    CHECK(dfn2_stream_pre_process_freq_dual(s, re, im, ar, ai, &io) == 1, "pre hop %d", h);
                } else {
                    input_pcm(pcm_in, 1, h, hop);
                    CHECK(dfn2_stream_pre_process(s, pcm_in, &io) == 1, "pre hop %d", h);
                }
                if (io.job_ready) rc = device_run(&dev, &io, &cfg.grid, ev == 1 ? 1 : ev == 2 ? 2 : 0);
                if (freq) {
                    CHECK(dfn2_stream_post_process_freq(s, rc, ore, oim, &valid) == 0, "post hop %d", h);
                    memcpy(out->spec + (size_t)h * 2u * (size_t)nb, ore, (size_t)nb * sizeof(float));
                    memcpy(out->spec + ((size_t)h * 2u + 1u) * (size_t)nb, oim, (size_t)nb * sizeof(float));
                } else {
                    valid = dfn2_stream_post_process(s, rc, out->pcm + (size_t)h * (size_t)hop);
                    CHECK(valid >= 0, "post hop %d", h);
                }
                out->valid_hops += valid ? 1 : 0;
            }
        }
    }
    out->jobs = dev.jobs;
    free(pcm_in);
    free(x);
    free(pool);
    return 0;
}

static void run_row(const Row *r) {
    DFN2GridSpec spec;
    DFN2Grid g;
    float *fwd, *inv;
    Out one, blk, bad;
    int freq, nb, N = r->n;
    size_t total;
    char label[64];

    memset(&spec, 0, sizeof spec);
    spec.sample_rate = r->sr; spec.n_fft = r->fft; spec.n_erb = r->n_erb; spec.df_bins = 96;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
    if (r->df_lookahead != 0) return;
#else
    spec.df_order = 5; spec.df_lookahead = r->df_lookahead;
#endif
    spec.frames_per_call = N;
    if (dfn2_grid_make(&g, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make\n"); exit(2); }
    snprintf(label, sizeof label, "%d/%d D=%d N=%d", r->sr, r->fft, r->df_lookahead, N);
    nb = g.n_bins;
    fwd = (float *)malloc((size_t)nb * (size_t)g.n_erb * sizeof(float));
    inv = (float *)malloc((size_t)nb * (size_t)g.n_erb * sizeof(float));
    {
        size_t i;
        for (i = 0; i < (size_t)nb * (size_t)g.n_erb; ++i) fwd[i] = (float)(i % 17) * 0.01f;
    }
    erb_fixture_split_inv_band_major(inv, nb, g.n_erb);
    CHECK(dfn2_prepost_latency_frames(&g) == r->df_lookahead, "%s: latency %d", label, dfn2_prepost_latency_frames(&g));

    for (freq = 0; freq < 2; ++freq) {
        total = freq ? (size_t)nb * 2u * (size_t)N * PACKETS * sizeof(float) : (size_t)g.hop * (size_t)N * PACKETS * sizeof(int16_t);
        run(&g, fwd, inv, 0, freq, 0, &one);
        run(&g, fwd, inv, 1, freq, 0, &blk);
        CHECK(memcmp(freq ? (void *)one.spec : (void *)one.pcm, freq ? (void *)blk.spec : (void *)blk.pcm, total) == 0,
              "%s %s: the packet stream differs from the N == 1 stream", label, freq ? "spectrum" : "PCM");
        CHECK(blk.valid_hops == one.valid_hops, "%s: valid hops %d vs %d", label, blk.valid_hops, one.valid_hops);
        CHECK(blk.jobs == PACKETS && one.jobs == PACKETS * N, "%s: jobs %d (want %d) vs %d (want %d)", label, blk.jobs, PACKETS,
              one.jobs, PACKETS * N);
        {   /* a non-vacuous comparison: the output carries signal and the faults reached it */
            size_t i, nz = 0, n = total / (freq ? sizeof(float) : sizeof(int16_t));
            for (i = 0; i < n; ++i) nz += freq ? one.spec[i] != 0.0f : one.pcm[i] != 0;
            CHECK(nz > n / 2, "%s: the N == 1 output is mostly silent", label);
        }
        /* the stand-in reading slice (i + 1) % N for head i must be caught */
        run(&g, fwd, inv, 1, freq, 1, &bad);
        CHECK(memcmp(freq ? (void *)one.spec : (void *)one.pcm, freq ? (void *)bad.spec : (void *)bad.pcm, total) != 0,
              "%s %s: a stand-in on the wrong window slice was not detected", label, freq ? "spectrum" : "PCM");
        free(one.pcm); free(one.spec); free(blk.pcm); free(blk.spec); free(bad.pcm); free(bad.spec);
    }
    free(fwd);
    free(inv);
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof ROWS / sizeof ROWS[0]; ++i) run_row(&ROWS[i]);
    if (g_fail) { printf("packet_equiv: %d check(s) FAILED\n", g_fail); return 1; }
    printf("packet_equiv: %d grids, PCM and spectrum, packet == N x one hop bit for bit -- PASS\n",
           (int)(sizeof ROWS / sizeof ROWS[0]));
    return 0;
}
