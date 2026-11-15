/* The spectrum-only stream of ulcnet_stream (cfg.freq_only), per grid row, one-hop and packet:
 *   - a pool of exactly get_mem_size() bytes between two canary zones initialises, 16 bytes less is refused; its size
 *     is the pinned one and the full stream's less the FFT, two STFT, one ISTFT and the scratch slab, as audiolib's own
 *     size queries give them;
 *   - a run through the _freq entries (alignment events and their reprime hops, model failure, NaN / Inf / 3e30 /
 *     unwritten masks, a NaN input bin, resets between hops and inside one, calls out of order) returns, hop for hop,
 *     the codes, job tensors, generation, restart mask and spectra of a full stream fed the same;
 *   - the canary zones are intact afterwards;
 *   - every PCM entry returns -1 and changes nothing (pool bytes, generation, io, PCM); an open _freq hop survives it;
 *   - a descriptor of the other kind of stream is refused in both directions; any non-zero freq_only is the same build. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"
#include "utility/mem_align.h"

typedef struct Row { int sr, fft, depth, n; size_t slim_bytes; } Row;

/* The pinned bytes are host (LP64) numbers. */
static const Row ROWS[] = {
    { 16000, 256, 63, 1, 7824 },
    { 16000, 256, 63, 8, 58320 },
    { 16000, 512, 32, 1, 14992 },
    { 16000, 512, 32, 8, 115664 },
    { 48000, 1024, 47, 1, 29328 },
    { 48000, 1024, 47, 8, 230352 },
    { 16000, 512, 32, 64, 922064 },
};

#define GUARD 4096u
#define CANARY 0xC3
#define HOPS 420

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

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

static uint32_t lcg_next(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

/* Writes the mask of job `t` (`elements` = 2 * N * n_bins) and returns the job status. kind 1: fails after writing,
 * 2: one NaN, 3: one -Inf, 4: one 3e30, 5: nothing written. */
static int fill_mask(float *out, size_t elements, int t, int kind) {
    uint32_t h = 2654435761u * (uint32_t)(t + 1);
    size_t i;
    if (kind == 5) return 0;
    for (i = 0; i < elements; ++i) {
        lcg_next(&h);
        out[i] = i < elements / 2u ? 0.2f + 0.9f * (float)((h >> 8) & 0xffff) / 65535.0f
                                   : 0.3f * ((float)((h >> 8) & 0xffff) / 32768.0f - 1.0f);
    }
    if (kind == 2) out[(2u * elements) / 3u] = (float)NAN;
    if (kind == 3) out[elements / 3u] = -INFINITY;
    if (kind == 4) out[5] = 3.0e30f;
    return kind == 1 ? -1 : 0;
}

static int job_kind(int t) {
    return t % 7 == 3 ? 1 : t % 11 == 5 ? 2 : t % 19 == 9 ? 3 : t % 13 == 6 ? 4 : t % 17 == 8 ? 5 : 0;
}

static int same_planes(const float *a, const float *b, size_t n) {
    return (a == NULL) == (b == NULL) && (a == NULL || memcmp(a, b, n * sizeof(float)) == 0);
}

static int same_job(const UlcnetHopIO *a, const UlcnetHopIO *b) {
    const UlcnetModelIoInputs *x = &a->inputs, *y = &b->inputs;
    const size_t n = x->spectrum_bins_elements;
    if (a->state_generation != b->state_generation || n != y->spectrum_bins_elements ||
        x->frame_reset_elements != y->frame_reset_elements ||
        a->outputs.spectrum_ri_elements != b->outputs.spectrum_ri_elements ||
        (a->outputs.output == NULL) != (b->outputs.output == NULL))
        return 0;
    return same_planes(x->error_mag, y->error_mag, n) && same_planes(x->far_mag, y->far_mag, n) &&
           same_planes(x->error_cos, y->error_cos, n) && same_planes(x->error_sin, y->error_sin, n) &&
           same_planes(x->frame_reset, y->frame_reset, x->frame_reset_elements) &&
           (a->outputs.output == NULL ||
            memcmp(a->outputs.output, b->outputs.output, a->outputs.spectrum_ri_elements * sizeof(float)) == 0);
}

/* Alignment event of model hop h: a change every 53 hops (two in a row at 63..64), solid from hop 30, unlocked over
 * 120..139 (a relock at 140). */
static UlcnetStreamDelay event_of(int h) {
    UlcnetStreamDelay d;
    d.changed = h % 53 == 10 || h == 64;
    d.solid = h >= 30 && !(h >= 120 && h < 140);
    return d;
}

static void fill_spectra(uint32_t *seed, int h, float *er, float *ei, float *fr, float *fi, size_t bins) {
    const float level = h % 31 == 7 ? 0.0f : h % 29 == 3 ? 4.0f : 0.2f;
    size_t k;
    for (k = 0; k < bins; ++k) {
        er[k] = level * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
        ei[k] = level * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
        fr[k] = 0.3f * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
        fi[k] = 0.3f * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
    }
    if (h % 97 == 40) er[3] = (float)NAN;
}

/* `slim` (freq_only) against `full`, job for job; n hops per job. Returns the number of differing calls. */
static int run_pair(UlcnetStream *slim, UlcnetStream *full, const UlcnetGrid *g, int n, const char *label) {
    const size_t nb = (size_t)g->n_bins, plane = (size_t)n * nb;
    const int jobs = n == 1 ? HOPS : (HOPS / n < 16 ? 16 : HOPS / n);
    float *x = (float *)malloc(plane * 8u * sizeof(float));
    float *er, *ei, *fr, *fi, *sre, *sim, *fre, *fim;
    UlcnetStreamDelay d[ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL];
    uint32_t seed = 2463534242u + (uint32_t)g->n_fft + (uint32_t)n;
    int t, k, bad = 0, enhanced = 0, reprimes = 0;

    if (!x) exit(2);
    er = x; ei = x + plane; fr = x + 2 * plane; fi = x + 3 * plane;
    sre = x + 4 * plane; sim = x + 5 * plane; fre = x + 6 * plane; fim = x + 7 * plane;
    for (t = 0; t < jobs; ++t) {
        UlcnetHopIO sio, fio;
        int src, frc, srm = -1, frm = -1, opened;

        for (k = 0; k < n; ++k) {
            fill_spectra(&seed, t * n + k, er + k * nb, ei + k * nb, fr + k * nb, fi + k * nb, nb);
            d[k] = event_of(t * n + k);
        }
        if (t == jobs / 4) { ulcnet_stream_reset(slim); ulcnet_stream_reset(full); }
        if (t == jobs / 2) {   /* out of order: a post with nothing open, then a second pre in one hop */
            if (ulcnet_stream_post_process_freq(slim, 0, sre, sim) != -1 ||
                ulcnet_stream_post_process_freq(full, 0, fre, fim) != -1) ++bad;
        }
        memset(&sio, 0x5A, sizeof sio);
        memset(&fio, 0x5A, sizeof fio);
        if (n == 1) {
            src = ulcnet_stream_pre_process_freq(slim, er, ei, fr, fi, d[0], &sio);
            frc = ulcnet_stream_pre_process_freq(full, er, ei, fr, fi, d[0], &fio);
        } else {
            src = ulcnet_stream_pre_process_packet_freq(slim, er, ei, fr, fi, d, &sio);
            frc = ulcnet_stream_pre_process_packet_freq(full, er, ei, fr, fi, d, &fio);
        }
        if (src < 0 || src != frc || ulcnet_stream_restart_mask(slim) != ulcnet_stream_restart_mask(full) ||
            (src == 1 && !same_job(&sio, &fio)) || (src == 0 && memcmp(&sio, &fio, sizeof sio) != 0)) {
            ++bad;
            break;
        }
        if (t == jobs / 2) {
            UlcnetHopIO again;
            if ((n == 1 ? ulcnet_stream_pre_process_freq(slim, er, ei, fr, fi, d[0], &again)
                        : ulcnet_stream_pre_process_packet_freq(slim, er, ei, fr, fi, d, &again)) != -1)
                ++bad;
        }
        opened = src;
        if (src == 1) {
            srm = fill_mask(sio.outputs.output, sio.outputs.spectrum_ri_elements, t, job_kind(t));
            frm = fill_mask(fio.outputs.output, fio.outputs.spectrum_ri_elements, t, job_kind(t));
        }
        if (t == jobs / 3 || t == (2 * jobs) / 3) {   /* reset with the hop open: the hop is dropped */
            ulcnet_stream_reset(slim);
            ulcnet_stream_reset(full);
            if (ulcnet_stream_post_process_freq(slim, srm, sre, sim) != -1 ||
                ulcnet_stream_post_process_freq(full, frm, fre, fim) != -1) ++bad;
            continue;
        }
        src = ulcnet_stream_post_process_freq(slim, srm, sre, sim);
        frc = ulcnet_stream_post_process_freq(full, frm, fre, fim);
        if (src != 1 || frc != 1 ||
            ulcnet_stream_state_generation(slim) != ulcnet_stream_state_generation(full) ||
            memcmp(sre, fre, plane * sizeof(float)) != 0 || memcmp(sim, fim, plane * sizeof(float)) != 0)
            ++bad;
        enhanced += memcmp(sre, er, plane * sizeof(float)) != 0;
        reprimes += opened == 0;
    }
    /* not vacuous: jobs are enhanced (a third of them up to N = 8; 64-hop packets mostly hold a refused frame) and the
     * one-hop run has reprime hops */
    if (enhanced < 4 || (n <= 8 && enhanced < jobs / 3) || (n == 1 && reprimes < 5)) ++bad;
    if (ulcnet_stream_state_generation(slim) < 4u) ++bad;   /* the run did restart the model */
    if (bad) fprintf(stderr, "%s: %d differing call(s), the last at job %d\n", label, bad, t);
    free(x);
    return bad;
}

/* The PCM entries on a freq_only stream, from a reset: -1, the pool bytes, generation, io and PCM unchanged, and an
 * open _freq hop is left as it was. */
static void check_pcm_refused(UlcnetStream *slim, const unsigned char *pool, size_t bytes, const UlcnetGrid *g, int n,
                              const char *label) {
    const size_t hop = (size_t)g->hop, nb = (size_t)g->n_bins, plane = (size_t)n * nb;
    float *pcm_f = (float *)calloc(hop * (size_t)n, sizeof(float));
    int16_t *pcm = (int16_t *)malloc(hop * sizeof(int16_t));
    float *z = (float *)calloc(plane * 2u, sizeof(float));
    unsigned char *snap = (unsigned char *)malloc(bytes);
    UlcnetStreamDelay d[ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL];
    UlcnetHopIO io, before;
    uint32_t gen;
    size_t k;
    int untouched = 1;

    if (!pcm_f || !pcm || !z || !snap) exit(2);
    for (k = 0; k < hop; ++k) pcm[k] = (int16_t)(1234 + (int)k);
    for (k = 0; k < (size_t)n; ++k) { d[k].changed = 1; d[k].solid = 1; }
    ulcnet_stream_reset(slim);
    gen = ulcnet_stream_state_generation(slim);
    memset(&io, 0x5A, sizeof io);
    before = io;
    memcpy(snap, pool, bytes);
    if (n == 1) {
        CHECK(ulcnet_stream_pre_process_f32(slim, pcm_f, pcm_f, d[0], &io) == -1, "%s: f32 pre_process accepted", label);
        CHECK(memcmp(snap, pool, bytes) == 0 && memcmp(&io, &before, sizeof io) == 0 &&
              ulcnet_stream_state_generation(slim) == gen, "%s: a refused f32 pre_process changed the stream", label);
        CHECK(ulcnet_stream_pre_process(slim, pcm, pcm, d[0], &io) == -1, "%s: int16 pre_process accepted", label);
        CHECK(memcmp(snap, pool, bytes) == 0 && memcmp(&io, &before, sizeof io) == 0 &&
              ulcnet_stream_state_generation(slim) == gen, "%s: a refused int16 pre_process changed the stream", label);
    } else {
        CHECK(ulcnet_stream_pre_process_packet_f32(slim, pcm_f, pcm_f, d, &io) == -1,
              "%s: f32 packet pre_process accepted", label);
        CHECK(memcmp(snap, pool, bytes) == 0 && memcmp(&io, &before, sizeof io) == 0 &&
              ulcnet_stream_state_generation(slim) == gen, "%s: a refused packet pre_process changed the stream", label);
    }
    for (k = 0; k < (size_t)n; ++k) d[k].changed = 0;
    CHECK((n == 1 ? ulcnet_stream_pre_process_freq(slim, z, z + plane, z, z + plane, d[0], &io)
                  : ulcnet_stream_pre_process_packet_freq(slim, z, z + plane, z, z + plane, d, &io)) == 1,
          "%s: _freq pre after the refusals", label);
    memcpy(snap, pool, bytes);
    CHECK(ulcnet_stream_post_process(slim, 0, pcm) == -1, "%s: PCM post_process accepted", label);
    for (k = 0; k < hop; ++k) untouched &= pcm[k] == (int16_t)(1234 + (int)k);
    CHECK(untouched && memcmp(snap, pool, bytes) == 0, "%s: a refused PCM post_process wrote PCM or the pool", label);
    CHECK(fill_mask(io.outputs.output, io.outputs.spectrum_ri_elements, 1, 0) == 0 &&
          ulcnet_stream_post_process_freq(slim, 0, z, z + plane) == 1, "%s: the open _freq hop was lost", label);
    CHECK(ulcnet_stream_hop_size(slim) == g->hop, "%s: hop size", label);
    free(pcm_f);
    free(pcm);
    free(z);
    free(snap);
}

/* What a freq_only pool leaves out, from audiolib's own size queries (each carve 16-aligned). */
static size_t framing_bytes(const UlcnetGrid *g, int n) {
    AcStftConfig scfg;
    const size_t hop = (size_t)g->hop, plane = (size_t)n * (size_t)g->n_bins;
    ac_stft_config_default(&scfg, g->n_fft);
    scfg.ola = AC_OLA_ACCUMULATE;
    return ck_align16_size(fft_get_mem_size(g->n_fft)) + 2u * ck_align16_size(ac_stft_get_mem_size(&scfg)) +
           ck_align16_size(ac_istft_get_mem_size(&scfg)) + ck_align16_size((3u * hop + 6u * plane) * sizeof(float));
}

static void run_row(const Row *r) {
    UlcnetGridSpec spec;
    UlcnetGrid g;
    UlcnetStreamConfig cfull, cslim, cseven;
    UlcnetStreamMemReq rfull, rslim, rseven;
    unsigned char *raw, *pool;
    void *fpool = NULL;
    UlcnetStream *slim, *full;
    char label[64];

    snprintf(label, sizeof label, "%d/%d D %d N=%d", r->sr, r->fft, r->depth, r->n);
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = r->sr;
    spec.n_fft = r->fft;
    spec.delay_depth = r->depth;
    if (ulcnet_grid_make(&g, &spec) != ULCNET_GRID_OK) { fprintf(stderr, "%s: grid_make\n", label); exit(2); }

    memset(&cfull, 0, sizeof cfull);
    cfull.grid = g;
    cfull.frames_per_call = r->n;
    cslim = cfull;
    cslim.freq_only = 1;
    cseven = cfull;
    cseven.freq_only = 7;
    CHECK(ulcnet_stream_get_mem_size(&cfull, &rfull) == 0 && ulcnet_stream_get_mem_size(&cslim, &rslim) == 0 &&
          ulcnet_stream_get_mem_size(&cseven, &rseven) == 0, "%s: sizing", label);
    CHECK(rslim.build_flags_hash != rfull.build_flags_hash && rslim.layout_version == ULCNET_STREAM_LAYOUT_VERSION &&
          rslim.alignment == ULCNET_STREAM_ALIGNMENT, "%s: the two descriptors do not differ", label);
    CHECK(memcmp(&rseven, &rslim, sizeof rslim) == 0, "%s: freq_only 7 is not the build of freq_only 1", label);
    CHECK((size_t)rslim.bytes == r->slim_bytes, "%s: freq_only pool %llu bytes, pinned %zu", label,
          (unsigned long long)rslim.bytes, r->slim_bytes);
    CHECK((size_t)rfull.bytes - (size_t)rslim.bytes == framing_bytes(&g, r->n),
          "%s: full - freq_only = %llu bytes, the framing is %zu", label,
          (unsigned long long)(rfull.bytes - rslim.bytes), framing_bytes(&g, r->n));

    if (posix_memalign(&fpool, 16, (size_t)rfull.bytes) != 0) exit(2);
    pool = guarded_pool((size_t)rslim.bytes, &raw);
    CHECK(ulcnet_stream_init_ex(&cslim, fpool, (size_t)rfull.bytes, &rfull) == NULL,
          "%s: freq_only init took a full descriptor", label);
    CHECK(ulcnet_stream_init_ex(&cfull, fpool, (size_t)rfull.bytes, &rslim) == NULL,
          "%s: full init took a freq_only descriptor", label);
    CHECK(ulcnet_stream_init_ex(&cslim, pool, (size_t)rslim.bytes - 16u, &rslim) == NULL &&
          ulcnet_stream_init(&cslim, pool, (size_t)rslim.bytes - 16u) == NULL, "%s: undersized pool accepted", label);
    CHECK(guards_intact(raw, (size_t)rslim.bytes), "%s: a refused init wrote outside the pool", label);
    slim = ulcnet_stream_init_ex(&cslim, pool, (size_t)rslim.bytes, &rslim);
    full = ulcnet_stream_init_ex(&cfull, fpool, (size_t)rfull.bytes, &rfull);
    CHECK(slim != NULL && full != NULL, "%s: init on the exact pool size", label);
    if (slim && full) {
        CHECK(run_pair(slim, full, &g, r->n, label) == 0, "%s: the freq_only stream differs from the full one", label);
        check_pcm_refused(slim, pool, (size_t)rslim.bytes, &g, r->n, label);
        CHECK(guards_intact(raw, (size_t)rslim.bytes), "%s: the freq_only stream wrote outside its pool", label);
    }
    free(raw);
    free(fpool);
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof ROWS / sizeof ROWS[0]; ++i) run_row(&ROWS[i]);
    if (g_fail) { printf("freq_only: %d check(s) FAILED\n", g_fail); return 1; }
    printf("freq_only: %d grid(s) -- PASS\n", (int)(sizeof ROWS / sizeof ROWS[0]));
    return 0;
}
