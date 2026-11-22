/* The spectrum-only context of dfn2_stream (cfg.freq_only). Per grid row:
 *   - a pool of exactly get_mem_size() bytes between two canary zones initialises, 16 bytes less is refused;
 *   - a run through the _freq entries (model failure, non-finite heads, refused bins, attenuation limit, resets
 *     between hops and inside one) returns, hop for hop, the codes, job tensors, generation, validity flag and
 *     spectra of a full context fed the same;
 *   - the canary zones are intact afterwards: nothing was written outside the pool;
 *   - the PCM entries (per-hop and packet) return -1 and write nothing, hpf_enable is refused, and a descriptor of the
 *     other kind of context is refused in both directions.
 * A row with N > 1 runs the packet entries (N frames per call, mask_lookahead 0). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dfn2_stream.h"
#include "erb_fixture.h"

typedef struct Row { int sr, fft, n_erb, mask_lookahead, df_lookahead, n; } Row;

#ifdef DFN2_MASK_ONLY
static const Row ROWS[] = {
    { 48000, 1024, 32, 1, 0, 1 },
    { 16000, 256, 24, 0, 0, 8 },
    { 48000, 1024, 96, 1, 0, 1 },
};
#else
static const Row ROWS[] = {
    { 16000, 256, 24, 0, 0, 1 },
    { 16000, 256, 24, 0, 0, 8 },
    { 48000, 1024, 32, 1, 1, 1 },
    { 32000, 512, 28, 2, 4, 1 },
    { 48000, 1024, 96, 1, 1, 1 },
};
#endif

#define GUARD 4096u
#define CANARY 0xC3
#define HOPS 420

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

static DFN2Grid make_grid(const Row *r) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = r->sr;
    spec.n_fft = r->fft;
    spec.n_erb = r->n_erb;
    spec.df_bins = 96;
    spec.mask_lookahead = r->mask_lookahead;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
#else
    spec.df_order = 5;
    spec.df_lookahead = r->df_lookahead;
#endif
    spec.frames_per_call = r->n;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make fail\n"); exit(2); }
    return grid;
}

/* A pool of `bytes` with GUARD canary bytes on each side; *raw is what to free. */
static unsigned char *guarded_pool(size_t bytes, unsigned char **raw) {
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes + 2u * GUARD) != 0) exit(2);
    memset(p, CANARY, bytes + 2u * GUARD);
    *raw = (unsigned char *)p;
    return *raw + GUARD;
}

static int guards_intact(const unsigned char *raw, size_t bytes) {
    size_t k;
    for (k = 0; k < GUARD; ++k)
        if (raw[k] != CANARY || raw[GUARD + bytes + k] != CANARY) return 0;
    return 1;
}

/* Heads of hop t, the same for both contexts; returns the model status. `kind` 1: fails after half the mask,
 * 2: a NaN in the mask with status 0. */
static int fill_heads(DFN2PrepostOutputs *out, int t, int kind) {
    size_t i;
    for (i = 0; i < out->erb_mask_elements; ++i) {
        out->erb_mask[i] = 0.3f + 0.05f * (float)((i + (size_t)t) % 11);
        if (kind == 1 && i == out->erb_mask_elements / 2) return -1;
    }
    if (kind == 2) out->erb_mask[out->erb_mask_elements / 3] = (float)NAN;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.03f * (float)((i + (size_t)t) % 7) - 0.08f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.25f * (float)((t + (int)i) % 4);
    for (i = 0; i < out->df_convp_window_elements; ++i) out->df_convp_window_next[i] = 0.0f;
#endif
    for (i = 0; i < out->gru_hidden_elements; ++i) out->gru_hidden_next[i] = 0.0f;
    return 0;
}

/* The _freq entries of one hop (N == 1) or one packet of N frames. */
static int pre_freq(DFN2Stream *s, const DFN2Grid *g, const float *er, const float *ei, const float *pr, const float *pi_,
                    DFN2HopIO *io) {
    return g->frames_per_call == 1 ? dfn2_stream_pre_process_freq_dual(s, er, ei, pr, pi_, io)
                                   : dfn2_stream_pre_process_packet_freq_dual(s, er, ei, pr, pi_, io);
}
static int post_freq(DFN2Stream *s, const DFN2Grid *g, int rc, float *re, float *im, int *valid) {
    return g->frames_per_call == 1 ? dfn2_stream_post_process_freq(s, rc, re, im, valid)
                                   : dfn2_stream_post_process_packet_freq(s, rc, re, im, valid);
}

static int same_job(const DFN2HopIO *a, const DFN2HopIO *b) {
    if (a->action != b->action || a->state_generation != b->state_generation) return 0;
    if (a->action != STREAM_WORK_SUBMIT) return 1;
    return a->inputs.erb_window_elements == b->inputs.erb_window_elements &&
           a->inputs.spec_window_elements == b->inputs.spec_window_elements &&
           memcmp(a->inputs.erb_window, b->inputs.erb_window, a->inputs.erb_window_elements * sizeof(float)) == 0 &&
           memcmp(a->inputs.spec_window, b->inputs.spec_window, a->inputs.spec_window_elements * sizeof(float)) == 0;
}

/* `slim` (freq_only) against `full`, hop for hop. Returns the number of differing hops. */
static int run_pair(DFN2Stream *slim, DFN2Stream *full, const DFN2Grid *g, const char *label) {
    const int nb = g->n_bins * g->frames_per_call;   /* bins of one call */
    float *x = (float *)malloc((size_t)nb * 8u * sizeof(float));
    float *er = x, *ei = x + nb, *pr = x + 2 * nb, *pi_ = x + 3 * nb;
    float *sre = x + 4 * nb, *sim = x + 5 * nb, *fre = x + 6 * nb, *fim = x + 7 * nb;
    uint32_t lcg = 2463534242u + (uint32_t)g->n_fft;
    int t, k, bad = 0;

    if (!x) exit(2);
    for (t = 0; t < HOPS; ++t) {
        DFN2HopIO sio, fio;
        int kind = (t % 7) == 3 ? 1 : (t % 11) == 5 ? 2 : 0;
        int srm = 0, frm = 0, sv = -9, fv = -9, src, frc;

        for (k = 0; k < nb; ++k) {
            float *dst[4];
            int j;
            dst[0] = er; dst[1] = ei; dst[2] = pr; dst[3] = pi_;
            for (j = 0; j < 4; ++j) {
                lcg = lcg * 1664525u + 1013904223u;
                dst[j][k] = ((float)((lcg >> 8) & 0xFFFFu) / 32768.0f - 1.0f) * ((t % 29) == 20 ? 0.0f : 2.5f);
            }
        }
        if (t == 50) CHECK(dfn2_stream_set_atten_lim(slim, 12.0f) == 0 && dfn2_stream_set_atten_lim(full, 12.0f) == 0, "%s: atten", label);
        if (t == 200) CHECK(dfn2_stream_set_atten_lim(slim, 0.0f) == 0 && dfn2_stream_set_atten_lim(full, 0.0f) == 0, "%s: atten off", label);
        if (t == 100 || t == 101) { dfn2_stream_reset(slim); dfn2_stream_reset(full); }

        if ((t % 13) == 6) {   /* a bin beyond the limit, then a NaN: refused, nothing consumed */
            float keep = er[5];
            er[5] = 1.0e21f;
            src = pre_freq(slim, g, er, ei, pr, pi_, &sio);
            frc = pre_freq(full, g, er, ei, pr, pi_, &fio);
            er[5] = (float)NAN;
            if (src != DFN2_STREAM_ERR_RANGE || frc != DFN2_STREAM_ERR_RANGE || pre_freq(slim, g, er, ei, pr, pi_, &sio) != DFN2_STREAM_ERR_RANGE) ++bad;
            er[5] = keep;
        }
        memset(&sio, 0, sizeof(sio));
        memset(&fio, 0, sizeof(fio));
        src = pre_freq(slim, g, er, ei, (t % 5) == 2 ? er : pr, (t % 5) == 2 ? ei : pi_, &sio);
        frc = pre_freq(full, g, er, ei, (t % 5) == 2 ? er : pr, (t % 5) == 2 ? ei : pi_, &fio);
        if (src != 1 || frc != 1 || !same_job(&sio, &fio)) { ++bad; break; }
        if (sio.action == STREAM_WORK_SUBMIT) { srm = fill_heads(&sio.outputs, t, kind); frm = fill_heads(&fio.outputs, t, kind); }
        if (t == 150 || t == 300) {   /* reset with the hop open: the hop is dropped */
            dfn2_stream_reset(slim); dfn2_stream_reset(full);
            if (post_freq(slim, g, 0, sre, sim, &sv) != -1 || post_freq(full, g, 0, fre, fim, &fv) != -1) ++bad;
            continue;
        }
        src = post_freq(slim, g, srm, sre, sim, &sv);
        frc = post_freq(full, g, frm, fre, fim, &fv);
        if (src != 0 || frc != 0 || sv != fv || srm != frm ||
            dfn2_stream_state_generation(slim) != dfn2_stream_state_generation(full) ||
            memcmp(sre, fre, (size_t)nb * sizeof(float)) != 0 || memcmp(sim, fim, (size_t)nb * sizeof(float)) != 0)
            ++bad;
    }
    free(x);
    return bad;
}

/* The PCM entries on a freq_only context: -1, nothing written, and an open _freq hop is left as it was. */
static void check_pcm_refused(DFN2Stream *slim, const DFN2Grid *g, const char *label) {
    const int nb = g->n_bins * g->frames_per_call, hop = g->hop * g->frames_per_call;   /* one call */
    int16_t *pcm = (int16_t *)malloc((size_t)hop * 2u * sizeof(int16_t));
    float *z = (float *)calloc((size_t)nb * 2u, sizeof(float));
    DFN2HopIO io, before;
    int k, untouched = 1;

    if (!pcm || !z) exit(2);
    for (k = 0; k < 2 * hop; ++k) pcm[k] = (int16_t)(1234 + k);
    memset(&io, 0x5A, sizeof(io));
    before = io;
    CHECK(dfn2_stream_pre_process(slim, pcm, &io) == -1, "%s: PCM pre_process accepted", label);
    CHECK(dfn2_stream_pre_process_packet(slim, pcm, &io) == -1, "%s: PCM pre_process_packet accepted", label);
    CHECK(memcmp(&io, &before, sizeof(io)) == 0, "%s: a refused PCM pre_process wrote io", label);
    CHECK(pre_freq(slim, g, z, z + nb, z, z + nb, &io) == 1, "%s: _freq pre after the refusal", label);
    CHECK(dfn2_stream_post_process(slim, 0, pcm + hop) == -1, "%s: PCM post_process accepted", label);
    CHECK(dfn2_stream_post_process_packet(slim, 0, pcm + hop) == -1, "%s: PCM post_process_packet accepted", label);
    for (k = 0; k < 2 * hop; ++k) untouched &= pcm[k] == (int16_t)(1234 + k);
    CHECK(untouched, "%s: a refused PCM entry wrote PCM", label);
    CHECK(post_freq(slim, g, 1, z, z + nb, NULL) == 0, "%s: the open _freq hop was lost", label);
    CHECK(dfn2_stream_hop_size(slim) == g->hop, "%s: hop size", label);
    free(pcm);
    free(z);
}

static void run_row(const Row *r) {
    DFN2Grid g = make_grid(r);
    const size_t elems = (size_t)g.n_bins * (size_t)g.n_erb;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *inv = (float *)malloc(elems * sizeof(float));
    DFN2StreamConfig cfull, cslim;
    DFN2StreamMemReq rfull, rslim;
    unsigned char *raw, *pool;
    void *fpool = NULL;
    DFN2Stream *slim, *full;
    char label[64];
    size_t i;

    if (!fwd || !inv) exit(2);
    snprintf(label, sizeof(label), "%d/%d n_erb %d N=%d", r->sr, r->fft, r->n_erb, r->n);
    for (i = 0; i < elems; ++i) fwd[i] = (float)(i % 17) * 0.01f;
    erb_fixture_split_inv_band_major(inv, g.n_bins, g.n_erb);

    memset(&cfull, 0, sizeof(cfull));
    cfull.grid = g;
    cfull.erb_fwd = fwd;
    cfull.erb_inv = inv;
    cslim = cfull;
    cslim.freq_only = 1;
    CHECK(dfn2_stream_get_mem_size(&cfull, &rfull) == 0 && dfn2_stream_get_mem_size(&cslim, &rslim) == 0, "%s: sizing", label);
    CHECK(rslim.bytes < rfull.bytes && rslim.build_flags_hash != rfull.build_flags_hash &&
          rslim.layout_version == DFN2_STREAM_LAYOUT_VERSION, "%s: the two descriptors do not differ", label);
    {
        DFN2StreamConfig chpf = cslim;
        DFN2StreamMemReq rhpf;
        chpf.hpf_enable = 1;
        chpf.hpf_cutoff_hz = 100.0f;
        CHECK(dfn2_stream_get_mem_size(&chpf, &rhpf) == -1, "%s: freq_only with hpf_enable sized", label);
    }

    if (posix_memalign(&fpool, 16, (size_t)rfull.bytes) != 0) exit(2);
    pool = guarded_pool((size_t)rslim.bytes, &raw);
    /* a descriptor of the other kind is refused, whatever the pool size */
    CHECK(dfn2_stream_init_ex(&cslim, fpool, (size_t)rfull.bytes, &rfull) == NULL, "%s: freq_only init took a full descriptor", label);
    CHECK(dfn2_stream_init_ex(&cfull, fpool, (size_t)rfull.bytes, &rslim) == NULL, "%s: full init took a freq_only descriptor", label);
    CHECK(dfn2_stream_init_ex(&cslim, pool, (size_t)rslim.bytes - 16u, &rslim) == NULL, "%s: undersized pool accepted", label);
    CHECK(guards_intact(raw, (size_t)rslim.bytes), "%s: a refused init wrote outside the pool", label);
    slim = dfn2_stream_init_ex(&cslim, pool, (size_t)rslim.bytes, &rslim);
    full = dfn2_stream_init_ex(&cfull, fpool, (size_t)rfull.bytes, &rfull);
    CHECK(slim != NULL && full != NULL, "%s: init on the exact pool size", label);
    if (slim && full) {
        CHECK(run_pair(slim, full, &g, label) == 0, "%s: the freq_only context differs from the full context", label);
        check_pcm_refused(slim, &g, label);
        CHECK(guards_intact(raw, (size_t)rslim.bytes), "%s: the freq_only context wrote outside the pool", label);
    }
    free(raw);

    free(fpool);
    free(fwd);
    free(inv);
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof(ROWS) / sizeof(ROWS[0]); ++i) run_row(&ROWS[i]);
    if (g_fail) { printf("freq_only: %d check(s) FAILED\n", g_fail); return 1; }
    printf("freq_only: %d grid(s) -- PASS\n", (int)(sizeof(ROWS) / sizeof(ROWS[0])));
    return 0;
}
