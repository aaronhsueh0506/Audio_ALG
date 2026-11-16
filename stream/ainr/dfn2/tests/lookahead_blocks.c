/* Blocks of N > 1 frames on a grid with mask_lookahead M > 0, PCM entries and spectrum entries, against a reference that feeds the
 * same stand-in accelerator one frame at a time from the first input frame (the heads of frames -M .. -1 included, no warm-up
 * skip) through the lib's per-frame functions. The stand-in is stateful and input dependent (head i from window slots i .. i + 2
 * and its running state, which is zeroed when the generation changes), so a block of N heads equals N single-frame runs. A call
 * is the newest N + 2 input frames and its heads are the frames T - N + 1 - M .. T - M; the first block's first M heads are before
 * the stream start and come out as silence. The script has a failed run, a run that never started, a non-finite head and a reset;
 * the generation moves once per refused or failed block. A stand-in reading its window one slot off must be caught. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_stft.h"
#include "dfn2_stream.h"
#include "erb_fixture.h"
#include "fft_wrapper.h"
#include "stream_mem.h"

#define BLOCKS 14

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Row { int L, D, N; } Row;

static const Row ROWS[] = {
    { 1, 0, 2 }, { 1, 0, 3 }, { 1, 0, 8 }, { 2, 0, 2 }, { 2, 0, 3 }, { 2, 0, 8 },
    { 1, 1, 2 }, { 1, 1, 3 }, { 1, 1, 8 }, { 2, 1, 2 }, { 2, 1, 3 }, { 2, 1, 8 },
};

/* The script of block b: 0 ok, 1 failed run (state dirty), 2 not started, 3 a non-finite head, 4 an odd status. */
static int kind_of(int b) { return b == 4 ? 1 : b == 6 ? 2 : b == 8 ? 3 : b == 10 ? 4 : 0; }
static int reset_before(int b) { return b == 1 || b == 12; }

typedef struct Dev {
    float st[8];
    unsigned gen;
    int synced;
} Dev;

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

/* One head from a three-frame window: erb3 [3][n_erb]; sre / sim [3][df_bins] contiguous each. */
static void dev_head(Dev *d, const float *erb3, const float *sre, const float *sim, int n_erb, int df_bins, float *mask,
                     float *coefs, size_t per, float *alpha) {
    float m = 0.0f, q = 0.0f;
    int e;
    size_t k;
    for (e = 0; e < 3 * n_erb; ++e) m += erb3[e];
    for (e = 0; e < 3 * df_bins; ++e) q += sre[e] + 0.5f * sim[e];
    m /= (float)(3 * n_erb);
    q /= (float)(3 * df_bins);
    for (e = 0; e < 8; ++e) d->st[e] = 0.9f * d->st[e] + 0.1f * m + 0.03f * q + 0.01f * (float)e;
    for (k = 0; k < (size_t)n_erb; ++k) mask[k] = sigm(0.3f * m + d->st[k % 8u] - 0.2f);
#ifdef DFN2_MASK_ONLY
    (void)coefs; (void)per; (void)alpha;
#else
    for (k = 0; k < per; ++k) coefs[k] = 0.05f * sinf(0.37f * (float)k + d->st[k % 8u] + 0.5f * q);
    *alpha = sigm(d->st[0] - 0.5f);
#endif
}

static void dev_sync(Dev *d, unsigned gen) {
    if (!d->synced || d->gen != gen) { memset(d->st, 0, sizeof d->st); d->gen = gen; d->synced = 1; }
}

/* The stream's job: N heads from the N + 2 window, head i from slots i + shift .. . Returns the run status. */
static int dev_run(Dev *d, const DFN2HopIO *io, const DFN2Grid *g, int kind, int shift) {
    const int n = g->frames_per_call, ne = g->n_erb, db = g->df_bins, nf = g->input_frames;
    int i;
    if (kind == 2) return STREAM_RUN_NOT_STARTED;
    dev_sync(d, io->state_generation);
    for (i = 0; i < n; ++i) {
        const int at = (i + shift) % n;
        const float *sre = io->inputs.spec_window + (size_t)at * (size_t)db;
        const float *sim = io->inputs.spec_window + ((size_t)nf + (size_t)at) * (size_t)db;
#ifdef DFN2_MASK_ONLY
        dev_head(d, io->inputs.erb_window + (size_t)at * (size_t)ne, sre, sim, ne, db, io->outputs.erb_mask + (size_t)i * (size_t)ne, NULL, 0, NULL);
#else
        const size_t per = io->outputs.coefs_elements / (size_t)n;
        dev_head(d, io->inputs.erb_window + (size_t)at * (size_t)ne, sre, sim, ne, db, io->outputs.erb_mask + (size_t)i * (size_t)ne,
                 io->outputs.coefs + (size_t)i * per, per, io->outputs.alpha + i);
#endif
    }
    if (kind == 3) io->outputs.erb_mask[(size_t)(n - 1) * (size_t)ne + 2] = (float)NAN;
    if (kind == 1) return STREAM_RUN_FAILED;
    if (kind == 4) return 4242;
    return STREAM_RUN_OK;
}

static void input_spec(float *re, float *im, float *ar, float *ai, int frames, int first, int nb) {
    int i, k;
    for (i = 0; i < frames; ++i)
        for (k = 0; k < nb; ++k) {
            const int h = first + i;
            const size_t at = (size_t)i * (size_t)nb + (size_t)k;
            re[at] = 0.4f * sinf(0.013f * (float)(h * nb + k)) + 0.1f * cosf(0.31f * (float)(h + k));
            im[at] = 0.3f * cosf(0.017f * (float)(h * nb + k)) - 0.05f * sinf(0.7f * (float)(h - k));
            if (h % 3 == 0) { ar[at] = 0.5f * re[at] + 0.01f; ai[at] = 0.5f * im[at]; }
            else { ar[at] = re[at]; ai[at] = im[at]; }
        }
}

static void input_pcm(int16_t *x, int frames, int first, int hop) {
    int i, k;
    for (i = 0; i < frames; ++i)
        for (k = 0; k < hop; ++k) {
            const long t = (long)(first + i) * hop + k;
            x[(size_t)i * (size_t)hop + (size_t)k] = (int16_t)(8000.0 * sin(0.013 * (double)t) + 2000.0 * sin(0.31 * (double)t + 1.0));
        }
}

typedef struct Out {
    float *spec;      /* freq: BLOCKS * N * 2 * n_bins floats; PCM: unused */
    int16_t *pcm;     /* PCM: BLOCKS * N * hop samples */
    int valid[BLOCKS];
    unsigned gen[BLOCKS];
} Out;

/* The stream: BLOCKS blocks of N frames. */
static void run_stream(const DFN2Grid *g, const float *fwd, const float *inv, int freq, int shift, Out *out) {
    const int N = g->frames_per_call, nb = g->n_bins, hop = g->hop;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    DFN2Stream *s;
    void *pool = NULL;
    Dev dev;
    float *x = (float *)malloc((size_t)nb * (size_t)N * 6u * sizeof(float));
    float *re = x, *im = x + (size_t)nb * N, *ar = x + 2u * (size_t)nb * N, *ai = x + 3u * (size_t)nb * N;
    float *ore = x + 4u * (size_t)nb * N, *oim = x + 5u * (size_t)nb * N;
    int16_t *pcm_in = (int16_t *)malloc((size_t)hop * (size_t)N * sizeof(int16_t));
    int b, i;

    memset(&dev, 0, sizeof dev);
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = *g; cfg.erb_fwd = fwd; cfg.erb_inv = inv; cfg.freq_only = freq;
    if (dfn2_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(2);
    s = dfn2_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) { fprintf(stderr, "init\n"); exit(2); }
    out->spec = freq ? (float *)calloc((size_t)BLOCKS * N * 2u * nb, sizeof(float)) : NULL;
    out->pcm = freq ? NULL : (int16_t *)calloc((size_t)BLOCKS * N * hop, sizeof(int16_t));
    for (b = 0; b < BLOCKS; ++b) {
        DFN2HopIO io;
        int rc = STREAM_RUN_OK, valid = -1, kind = kind_of(b);
        if (reset_before(b)) dfn2_stream_reset(s);
        if (freq) {
            input_spec(re, im, ar, ai, N, b * N, nb);
            CHECK(dfn2_stream_pre_process_packet_freq_dual(s, re, im, ar, ai, &io) == 1, "pre block %d", b);
        } else {
            input_pcm(pcm_in, N, b * N, hop);
            CHECK(dfn2_stream_pre_process_packet(s, pcm_in, &io) == 1, "pre block %d", b);
        }
        CHECK(io.action == STREAM_WORK_SUBMIT, "block %d has no job (every block of a lookahead grid submits)", b);
        rc = dev_run(&dev, &io, g, kind, shift);
        if (freq) {
            CHECK(dfn2_stream_post_process_packet_freq(s, rc, ore, oim, &valid) == 0, "post block %d", b);
            for (i = 0; i < N; ++i) {
                memcpy(out->spec + ((size_t)b * N + i) * 2u * nb, ore + (size_t)i * nb, (size_t)nb * sizeof(float));
                memcpy(out->spec + (((size_t)b * N + i) * 2u + 1u) * nb, oim + (size_t)i * nb, (size_t)nb * sizeof(float));
            }
        } else {
            valid = dfn2_stream_post_process_packet(s, rc, out->pcm + (size_t)b * N * hop);
            CHECK(valid >= 0, "post block %d", b);
        }
        out->valid[b] = valid;
        out->gen[b] = dfn2_stream_state_generation(s);
    }
    free(pcm_in); free(x); free(pool);
}

/* The reference: one lib instance of the N == 1 grid of the same signal grid, one frame at a time. */
static void run_reference(const DFN2Grid *g, const float *fwd, const float *inv, int freq, int N, Out *out) {
    const int nb = g->n_bins, hop = g->hop, ne = g->n_erb, db = g->df_bins;
    DFN2GridSpec spec;
    DFN2Grid g1;
    DFN2PrepostConfig pcfg;
    DFN2PrepostMemReq preq;
    DFN2Prepost *p;
    void *pool = NULL;
    FftHandle *fft = NULL;
    AcStft *stft = NULL;
    AcIstft *istft = NULL;
    void *stft_mem = NULL, *istft_mem = NULL;
    Dev dev;
    float *hm, *hc, *ha;   /* the block's heads, frame-major */
    float *x = (float *)malloc((size_t)nb * (size_t)N * 8u * sizeof(float));
    float *re = x, *im = x + (size_t)nb * N, *ar = x + 2u * (size_t)nb * N, *ai = x + 3u * (size_t)nb * N;
    float *er = x + 4u * (size_t)nb * N, *ei = x + 5u * (size_t)nb * N;   /* enhanced frames of the block */
    float *sr = x + 6u * (size_t)nb * N, *si = x + 7u * (size_t)nb * N;   /* spectra of the PCM frames */
    int16_t *pcm_in = (int16_t *)malloc((size_t)hop * (size_t)N * sizeof(int16_t));
    float *hop_f = (float *)malloc((size_t)hop * sizeof(float));
    unsigned gen = 1u;
    size_t per = 0;
    int b, i, k;

    memset(&spec, 0, sizeof spec);
    spec.sample_rate = g->sample_rate; spec.n_fft = g->n_fft; spec.n_erb = g->n_erb; spec.df_bins = g->df_bins;
    spec.df_order = g->df_order; spec.mask_lookahead = g->mask_lookahead; spec.df_lookahead = g->df_lookahead; spec.frames_per_call = 1;
    if (dfn2_grid_make(&g1, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid\n"); exit(2); }
#ifndef DFN2_MASK_ONLY
    per = (size_t)g->df_bins * (size_t)g->df_order * 2u;
#endif
    hm = (float *)calloc((size_t)N * (size_t)ne, sizeof(float));
    hc = (float *)calloc((size_t)N * (per ? per : 1u), sizeof(float));
    ha = (float *)calloc((size_t)N, sizeof(float));
    if (dfn2_prepost_config_defaults(&pcfg, &g1) != 0) exit(2);
    pcfg.erb_fwd = fwd; pcfg.erb_inv = inv;
    if (dfn2_prepost_get_mem_size(&pcfg, &preq) != 0 || posix_memalign(&pool, preq.alignment, (size_t)preq.bytes) != 0) exit(2);
    p = dfn2_prepost_init(pool, (size_t)preq.bytes, &pcfg);
    if (!p) { fprintf(stderr, "prepost init\n"); exit(2); }
    if (!freq) {
        AcStftConfig scfg;
        size_t sb, ib;
        fft = fft_create(g->n_fft);
        ac_stft_config_default(&scfg, g->n_fft);
        scfg.spec_scale = dfn2_stft_scale(g->n_fft); scfg.ispec_scale = dfn2_stft_unscale(g->n_fft); scfg.fft = fft;
        sb = ac_stft_get_mem_size(&scfg); ib = ac_istft_get_mem_size(&scfg);
        if (!fft || posix_memalign(&stft_mem, 16, sb) != 0 || posix_memalign(&istft_mem, 16, ib) != 0) exit(2);
        stft = ac_stft_init(stft_mem, sb, &scfg);
        istft = ac_istft_init(istft_mem, ib, &scfg);
        if (!stft || !istft) exit(2);
    }
    memset(&dev, 0, sizeof dev);
    out->spec = freq ? (float *)calloc((size_t)BLOCKS * N * 2u * nb, sizeof(float)) : NULL;
    out->pcm = freq ? NULL : (int16_t *)calloc((size_t)BLOCKS * N * hop, sizeof(int16_t));

    for (b = 0; b < BLOCKS; ++b) {
        const int kind = kind_of(b);
        DFN2PrepostInputs in;
        DFN2PrepostOutputs outv;
        int valid = 0;
        if (reset_before(b)) {
            dfn2_prepost_reset(p);
            if (stft) { ac_stft_reset(stft); ac_istft_reset(istft); }
            ++gen;
        }
        if (freq) input_spec(re, im, ar, ai, N, b * N, nb);
        else {
            input_pcm(pcm_in, N, b * N, hop);
            for (i = 0; i < N; ++i) {
                for (k = 0; k < hop; ++k) hop_f[k] = (float)pcm_in[(size_t)i * hop + k] / 32768.0f;
                ac_stft_push_planar(stft, hop_f, sr + (size_t)i * nb, si + (size_t)i * nb);
            }
            re = ar = sr; im = ai = si;
        }
        /* a run per frame from the first one: ready or not, the window of the frame is the three newest */
        for (i = 0; i < N; ++i) {
            CHECK(dfn2_prepost_pre_frame(p, re + (size_t)i * nb, im + (size_t)i * nb) >= 0, "ref pre_frame");
            CHECK(dfn2_prepost_views(p, &in, &outv) == 0, "ref views");
            if (kind != 2) {
                dev_sync(&dev, gen);
                dev_head(&dev, in.erb_window, in.spec_window, in.spec_window + (size_t)g1.input_frames * (size_t)db, ne, db,
                         hm + (size_t)i * ne, hc + (size_t)i * per,
                         per,
#ifdef DFN2_MASK_ONLY
                         NULL
#else
                         ha + i
#endif
                         );
            }
        }
        for (i = 0; i < N; ++i) {
            const size_t at = (size_t)i * nb;
            int r;
            if (kind == 0) {
                memcpy(outv.erb_mask, hm + (size_t)i * ne, (size_t)ne * sizeof(float));
#ifndef DFN2_MASK_ONLY
                memcpy(outv.coefs, hc + (size_t)i * per, per * sizeof(float));
                outv.alpha[0] = ha[i];
#endif
                r = dfn2_prepost_post_frame(p, 0, ar + at, ai + at, er + at, ei + at);
            } else {
                r = dfn2_prepost_post_frame_identity(p, ar + at, ai + at, er + at, ei + at);
            }
            CHECK(r >= 0, "ref post");
            valid += r;
        }
        if (kind == 1 || kind == 3 || kind == 4) ++gen;
        for (i = 0; i < N; ++i) {
            if (freq) {
                memcpy(out->spec + ((size_t)b * N + i) * 2u * nb, er + (size_t)i * nb, (size_t)nb * sizeof(float));
                memcpy(out->spec + (((size_t)b * N + i) * 2u + 1u) * nb, ei + (size_t)i * nb, (size_t)nb * sizeof(float));
            } else {
                ac_istft_push_planar(istft, er + (size_t)i * nb, ei + (size_t)i * nb, hop_f);
                for (k = 0; k < hop; ++k) out->pcm[((size_t)b * N + i) * hop + k] = stream_saturate_s16(hop_f[k]);
            }
        }
        out->valid[b] = valid;
        out->gen[b] = gen;
        if (!freq) { re = x; im = x + (size_t)nb * N; ar = x + 2u * (size_t)nb * N; ai = x + 3u * (size_t)nb * N; }
    }
    free(x); free(pcm_in); free(hop_f); free(hm); free(hc); free(ha); free(pool);
    free(stft_mem); free(istft_mem);
}

static int same(const Out *a, const Out *b, const DFN2Grid *g, int freq) {
    const size_t n = (size_t)BLOCKS * (size_t)g->frames_per_call;
    return memcmp(a->valid, b->valid, sizeof a->valid) == 0 && memcmp(a->gen, b->gen, sizeof a->gen) == 0 &&
           (freq ? memcmp(a->spec, b->spec, n * 2u * (size_t)g->n_bins * sizeof(float)) == 0
                 : memcmp(a->pcm, b->pcm, n * (size_t)g->hop * sizeof(int16_t)) == 0);
}

static void run_row(const Row *r) {
    DFN2GridSpec spec;
    DFN2Grid g;
    float *fwd, *inv;
    int freq, nb;
    char label[64];

    memset(&spec, 0, sizeof spec);
    spec.sample_rate = 16000; spec.n_fft = 256; spec.n_erb = 24; spec.df_bins = 96;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
    if (r->D != 0) return;
#else
    spec.df_order = 5; spec.df_lookahead = r->D;
#endif
    spec.mask_lookahead = r->L; spec.frames_per_call = r->N;
    if (dfn2_grid_make(&g, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make\n"); exit(2); }
    snprintf(label, sizeof label, "L=%d D=%d N=%d", r->L, r->D, r->N);
    nb = g.n_bins;
    fwd = (float *)malloc((size_t)nb * (size_t)g.n_erb * sizeof(float));
    inv = (float *)malloc((size_t)nb * (size_t)g.n_erb * sizeof(float));
    {
        size_t i;
        for (i = 0; i < (size_t)nb * (size_t)g.n_erb; ++i) fwd[i] = (float)(i % 17) * 0.01f;
    }
    erb_fixture_split_inv_band_major(inv, nb, g.n_erb);
    for (freq = 0; freq < 2; ++freq) {
        Out st, ref, bad;
        size_t i, n = (size_t)BLOCKS * (size_t)r->N * (freq ? 2u * (size_t)nb : (size_t)g.hop), nz = 0;
        run_stream(&g, fwd, inv, freq, 0, &st);
        run_reference(&g, fwd, inv, freq, r->N, &ref);
        CHECK(same(&st, &ref, &g, freq), "%s %s: the block stream differs from the one-frame reference", label, freq ? "spectrum" : "PCM");
        for (i = 0; i < n; ++i) nz += freq ? st.spec[i] != 0.0f : st.pcm[i] != 0;
        CHECK(nz > n / 2, "%s: the output is mostly silent", label);
        CHECK(st.valid[0] == (r->N > r->L + r->D ? r->N - r->L - r->D : 0), "%s: first block has %d spectra", label, st.valid[0]);
        CHECK(st.gen[BLOCKS - 1] == 1u + 2u + 3u, "%s: generation %u after 2 resets and 3 refused or failed blocks", label, st.gen[BLOCKS - 1]);
        run_stream(&g, fwd, inv, freq, 1, &bad);   /* head i reads window slot i + 1 */
        CHECK(!same(&bad, &ref, &g, freq), "%s %s: a stand-in one window slot off was not detected", label, freq ? "spectrum" : "PCM");
        free(st.spec); free(st.pcm); free(ref.spec); free(ref.pcm); free(bad.spec); free(bad.pcm);
    }
    free(fwd); free(inv);
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof ROWS / sizeof ROWS[0]; ++i) run_row(&ROWS[i]);
    if (g_fail) { printf("lookahead_blocks: %d check(s) FAILED\n", g_fail); return 1; }
    printf("lookahead_blocks: %d grids, mask_lookahead 1 and 2, PCM and spectrum, N > 1 == the one-frame reference bit for bit -- PASS\n",
           (int)(sizeof ROWS / sizeof ROWS[0]));
    return 0;
}
