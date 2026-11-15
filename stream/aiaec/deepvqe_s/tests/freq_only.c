/* The spectrum-only stream of deepvqe_stream (cfg.freq_only), per grid row (N = 1 and a batched N):
 *   - a pool of exactly get_mem_size() bytes between two canary zones initialises, 16 bytes less is refused; its size
 *     is the pinned one and the full stream's less the FFT, two STFT, one ISTFT and the scratch slab, as audiolib's own
 *     size queries give them;
 *   - a run through the _freq entries (model failure, NaN / Inf / 3e30 / unwritten taps, a NaN input bin, resets between
 *     hops and inside one, calls out of order, the N - 1 pre-roll hops) returns, hop for hop, the codes, job tensors,
 *     generation and spectra of a full stream fed the same;
 *   - the canary zones are intact afterwards;
 *   - both PCM entries return -1 and change nothing (pool bytes, generation, io, PCM); a due _freq hop survives them;
 *   - a descriptor of the other kind of stream is refused in both directions; any non-zero freq_only is the same build. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepvqe_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"
#include "utility/mem_align.h"

typedef struct Row { int sr, fft, depth, n; size_t slim_bytes; } Row;

/* The pinned bytes are host (LP64) numbers. */
static const Row ROWS[] = {
    { 16000, 256, 63, 1, 18272 },
    { 16000, 256, 63, 8, 119312 },
    { 16000, 512, 32, 1, 35680 },
    { 16000, 512, 32, 8, 237072 },
};

#define GUARD 4096u
#define CANARY 0xC3
#define HOPS 300

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

/* Writes the taps of job `t` and returns the job status. kind 1: fails after writing, 2: one NaN, 3: one -Inf,
 * 4: one 3e30, 5: nothing written. */
static int fill_taps(float *taps, size_t elements, int t, int kind) {
    uint32_t h = 2654435761u * (uint32_t)(t + 1);
    size_t i;
    if (kind == 5) return 0;
    for (i = 0; i < elements; ++i) {
        lcg_next(&h);
        taps[i] = 0.08f * ((float)((h >> 8) & 0xffff) / 32768.0f - 1.0f) + (i % 18u == 8u ? 0.6f : 0.0f);
    }
    if (kind == 2) taps[(2u * elements) / 3u] = (float)NAN;
    if (kind == 3) taps[elements / 3u] = -INFINITY;
    if (kind == 4) taps[5] = 3.0e30f;
    return kind == 1 ? -1 : 0;
}

static int job_kind(int t) {
    return t % 7 == 3 ? 1 : t % 11 == 5 ? 2 : t % 19 == 9 ? 3 : t % 13 == 6 ? 4 : t % 17 == 8 ? 5 : 0;
}

static int same_job(const DeepVqeHopIO *a, const DeepVqeHopIO *b) {
    const size_t n = a->inputs.spectrum_ri_elements;
    if (a->state_generation != b->state_generation || n != b->inputs.spectrum_ri_elements ||
        a->outputs.taps_elements != b->outputs.taps_elements || !a->inputs.mic || !b->inputs.mic ||
        !a->inputs.far || !b->inputs.far || !a->outputs.taps || !b->outputs.taps)
        return 0;
    return memcmp(a->inputs.mic, b->inputs.mic, n * sizeof(float)) == 0 &&
           memcmp(a->inputs.far, b->inputs.far, n * sizeof(float)) == 0 &&
           memcmp(a->outputs.taps, b->outputs.taps, a->outputs.taps_elements * sizeof(float)) == 0;
}

static void fill_spectra(uint32_t *seed, int h, float *mr, float *mi, float *fr, float *fi, size_t bins) {
    const float level = h % 31 == 7 ? 0.0f : h % 29 == 3 ? 4.0f : 0.2f;
    size_t k;
    for (k = 0; k < bins; ++k) {
        mr[k] = level * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
        mi[k] = level * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
        fr[k] = 0.3f * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
        fi[k] = 0.3f * ((float)(lcg_next(seed) >> 9) / 4194304.0f - 1.0f);
    }
    if (h % 97 == 40) mr[3] = (float)NAN;
}

/* `slim` (freq_only) against `full`, hop for hop. Returns the number of differing calls. */
static int run_pair(DeepVqeStream *slim, DeepVqeStream *full, const DeepVqeGrid *g, const char *label) {
    const size_t nb = (size_t)g->n_bins;
    float *x = (float *)malloc(nb * 8u * sizeof(float));
    float *mr, *mi, *fr, *fi, *sre, *sim, *fre, *fim;
    uint32_t seed = 2463534242u + (uint32_t)g->n_fft + (uint32_t)g->frames_per_call;
    int t, bad = 0, jobs = 0, real = 0;

    if (!x) exit(2);
    mr = x; mi = x + nb; fr = x + 2 * nb; fi = x + 3 * nb;
    sre = x + 4 * nb; sim = x + 5 * nb; fre = x + 6 * nb; fim = x + 7 * nb;
    for (t = 0; t < HOPS; ++t) {
        DeepVqeHopIO sio, fio;
        int src, frc, srm = -1, frm = -1;

        fill_spectra(&seed, t, mr, mi, fr, fi, nb);
        if (t == HOPS / 4) { deepvqe_stream_reset(slim); deepvqe_stream_reset(full); }
        if (t == HOPS / 2 &&
            (deepvqe_stream_post_process_freq(slim, 0, sre, sim) != -1 ||
             deepvqe_stream_post_process_freq(full, 0, fre, fim) != -1))
            ++bad;   /* a post with nothing due */
        memset(&sio, 0x5A, sizeof sio);
        memset(&fio, 0x5A, sizeof fio);
        src = deepvqe_stream_pre_process_freq(slim, mr, mi, fr, fi, &sio);
        frc = deepvqe_stream_pre_process_freq(full, mr, mi, fr, fi, &fio);
        if (src < 0 || src != frc || (src == 1 && !same_job(&sio, &fio)) ||
            (src == 0 && memcmp(&sio, &fio, sizeof sio) != 0)) {
            ++bad;
            break;
        }
        if (t == HOPS / 2) {
            DeepVqeHopIO again;
            if (deepvqe_stream_pre_process_freq(slim, mr, mi, fr, fi, &again) != -1) ++bad;
        }
        if (src == 1) {
            ++jobs;
            srm = fill_taps(sio.outputs.taps, sio.outputs.taps_elements, jobs, job_kind(jobs));
            frm = fill_taps(fio.outputs.taps, fio.outputs.taps_elements, jobs, job_kind(jobs));
        }
        if (t == HOPS / 3 || t == (2 * HOPS) / 3) {   /* reset with the hop due: it is dropped */
            deepvqe_stream_reset(slim);
            deepvqe_stream_reset(full);
            if (deepvqe_stream_post_process_freq(slim, srm, sre, sim) != -1 ||
                deepvqe_stream_post_process_freq(full, frm, fre, fim) != -1) ++bad;
            continue;
        }
        src = deepvqe_stream_post_process_freq(slim, srm, sre, sim);
        frc = deepvqe_stream_post_process_freq(full, frm, fre, fim);
        if (src < 0 || src != frc ||
            deepvqe_stream_state_generation(slim) != deepvqe_stream_state_generation(full) ||
            memcmp(sre, fre, nb * sizeof(float)) != 0 || memcmp(sim, fim, nb * sizeof(float)) != 0)
            ++bad;
        real += src == 1 && memcmp(sre, mr, nb * sizeof(float)) != 0;
    }
    /* not vacuous: jobs ran, most hops carry a real spectrum, the run restarted the state */
    if (jobs < HOPS / (2 * g->frames_per_call) || real < HOPS / 2 || deepvqe_stream_state_generation(slim) < 4u) ++bad;
    if (bad) fprintf(stderr, "%s: %d differing call(s), the last at hop %d\n", label, bad, t);
    free(x);
    return bad;
}

/* The PCM entries on a freq_only stream, from a reset: -1, the pool bytes, generation, io and PCM unchanged, and a due
 * _freq hop is left as it was. */
static void check_pcm_refused(DeepVqeStream *slim, const unsigned char *pool, size_t bytes, const DeepVqeGrid *g,
                              const char *label) {
    const size_t hop = (size_t)g->hop, nb = (size_t)g->n_bins;
    int16_t *pcm = (int16_t *)malloc(hop * sizeof(int16_t));
    float *z = (float *)calloc(nb * 2u, sizeof(float));
    unsigned char *snap = (unsigned char *)malloc(bytes);
    DeepVqeHopIO io, before;
    uint32_t gen;
    size_t k;
    int untouched = 1, pre;

    if (!pcm || !z || !snap) exit(2);
    for (k = 0; k < hop; ++k) pcm[k] = (int16_t)(1234 + (int)k);
    deepvqe_stream_reset(slim);
    gen = deepvqe_stream_state_generation(slim);
    memset(&io, 0x5A, sizeof io);
    before = io;
    memcpy(snap, pool, bytes);
    CHECK(deepvqe_stream_pre_process(slim, pcm, pcm, &io) == -1, "%s: PCM pre_process accepted", label);
    CHECK(memcmp(snap, pool, bytes) == 0 && memcmp(&io, &before, sizeof io) == 0 &&
          deepvqe_stream_state_generation(slim) == gen, "%s: a refused PCM pre_process changed the stream", label);
    pre = deepvqe_stream_pre_process_freq(slim, z, z + nb, z, z + nb, &io);
    CHECK(pre == (g->frames_per_call == 1 ? 1 : 0), "%s: _freq pre after the refusal", label);
    memcpy(snap, pool, bytes);
    CHECK(deepvqe_stream_post_process(slim, 0, pcm) == -1, "%s: PCM post_process accepted", label);
    for (k = 0; k < hop; ++k) untouched &= pcm[k] == (int16_t)(1234 + (int)k);
    CHECK(untouched && memcmp(snap, pool, bytes) == 0, "%s: a refused PCM post_process wrote PCM or the pool", label);
    if (pre == 1) fill_taps(io.outputs.taps, io.outputs.taps_elements, 1, 0);
    CHECK(deepvqe_stream_post_process_freq(slim, 0, z, z + nb) == (g->frames_per_call == 1 ? 1 : 0),
          "%s: the due _freq hop was lost", label);
    CHECK(deepvqe_stream_hop_size(slim) == g->hop, "%s: hop size", label);
    free(pcm);
    free(z);
    free(snap);
}

/* What a freq_only pool leaves out, from audiolib's own size queries (each carve 16-aligned). */
static size_t framing_bytes(const DeepVqeGrid *g) {
    AcStftConfig scfg;
    ac_stft_config_default(&scfg, g->n_fft);
    scfg.ola = AC_OLA_ACCUMULATE;
    return ck_align16_size(fft_get_mem_size(g->n_fft)) + 2u * ck_align16_size(ac_stft_get_mem_size(&scfg)) +
           ck_align16_size(ac_istft_get_mem_size(&scfg)) +
           ck_align16_size((3u * (size_t)g->hop + 6u * (size_t)g->n_bins) * sizeof(float));
}

static void run_row(const Row *r) {
    DeepVqeGridSpec spec;
    DeepVqeGrid g;
    DeepVqeStreamConfig cfull, cslim, cseven;
    DeepVqeStreamMemReq rfull, rslim, rseven;
    unsigned char *raw, *pool;
    void *fpool = NULL;
    DeepVqeStream *slim, *full;
    char label[64];

    snprintf(label, sizeof label, "%d/%d D %d N=%d", r->sr, r->fft, r->depth, r->n);
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = r->sr;
    spec.n_fft = r->fft;
    spec.delay_depth = r->depth;
    spec.frames_per_call = r->n;
    if (deepvqe_grid_make(&g, &spec) != DEEPVQE_GRID_OK) { fprintf(stderr, "%s: grid_make\n", label); exit(2); }

    memset(&cfull, 0, sizeof cfull);
    cfull.grid = g;
    cslim = cfull;
    cslim.freq_only = 1;
    cseven = cfull;
    cseven.freq_only = 7;
    CHECK(deepvqe_stream_get_mem_size(&cfull, &rfull) == 0 && deepvqe_stream_get_mem_size(&cslim, &rslim) == 0 &&
          deepvqe_stream_get_mem_size(&cseven, &rseven) == 0, "%s: sizing", label);
    CHECK(rslim.build_flags_hash != rfull.build_flags_hash && rslim.layout_version == DEEPVQE_STREAM_LAYOUT_VERSION &&
          rslim.alignment == DEEPVQE_STREAM_ALIGNMENT, "%s: the two descriptors do not differ", label);
    CHECK(memcmp(&rseven, &rslim, sizeof rslim) == 0, "%s: freq_only 7 is not the build of freq_only 1", label);
    CHECK((size_t)rslim.bytes == r->slim_bytes, "%s: freq_only pool %llu bytes, pinned %zu", label,
          (unsigned long long)rslim.bytes, r->slim_bytes);
    CHECK((size_t)rfull.bytes - (size_t)rslim.bytes == framing_bytes(&g),
          "%s: full - freq_only = %llu bytes, the framing is %zu", label,
          (unsigned long long)(rfull.bytes - rslim.bytes), framing_bytes(&g));

    if (posix_memalign(&fpool, 16, (size_t)rfull.bytes) != 0) exit(2);
    pool = guarded_pool((size_t)rslim.bytes, &raw);
    CHECK(deepvqe_stream_init_ex(&cslim, fpool, (size_t)rfull.bytes, &rfull) == NULL,
          "%s: freq_only init took a full descriptor", label);
    CHECK(deepvqe_stream_init_ex(&cfull, fpool, (size_t)rfull.bytes, &rslim) == NULL,
          "%s: full init took a freq_only descriptor", label);
    CHECK(deepvqe_stream_init_ex(&cslim, pool, (size_t)rslim.bytes - 16u, &rslim) == NULL &&
          deepvqe_stream_init(&cslim, pool, (size_t)rslim.bytes - 16u) == NULL, "%s: undersized pool accepted", label);
    CHECK(guards_intact(raw, (size_t)rslim.bytes), "%s: a refused init wrote outside the pool", label);
    slim = deepvqe_stream_init_ex(&cslim, pool, (size_t)rslim.bytes, &rslim);
    full = deepvqe_stream_init_ex(&cfull, fpool, (size_t)rfull.bytes, &rfull);
    CHECK(slim != NULL && full != NULL, "%s: init on the exact pool size", label);
    if (slim && full) {
        CHECK(run_pair(slim, full, &g, label) == 0, "%s: the freq_only stream differs from the full one", label);
        check_pcm_refused(slim, pool, (size_t)rslim.bytes, &g, label);
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
