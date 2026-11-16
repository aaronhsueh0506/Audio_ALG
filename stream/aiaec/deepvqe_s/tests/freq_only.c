/* The spectrum-only stream of deepvqe_stream (cfg.freq_only), per grid row (N = 1 and a batched N), through the packet
 * entries (N frames per call):
 *   - a pool of exactly get_mem_size() bytes between two canary zones initialises, 16 bytes less is refused; its size
 *     is the pinned one and the full stream's less the FFT, two STFT, one ISTFT and the scratch slab, as audiolib's own
 *     size queries give them;
 *   - a run through the _freq entries (model failure, NaN / Inf / 3e30 / unwritten taps, an overflowing enhanced bin, resets
 *     between packets and inside one, calls out of order) returns, packet for packet, the codes, job tensors, generation and
 *     spectra of a full stream fed the same; a NaN input bin is DEEPVQE_STREAM_ERR_RANGE on both, with no job;
 *   - the canary zones are intact afterwards;
 *   - the PCM entries (packet and one-hop) return -1 and change nothing (pool bytes, generation, io, PCM); a due _freq
 *     packet survives them;
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
    { 16000, 256, 63, 1, 16064 },
    { 16000, 256, 63, 8, 102720 },
    { 16000, 512, 32, 1, 31424 },
    { 16000, 512, 32, 8, 204096 },
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
 * 4: one 3e30, 5: nothing written, 6: taps of 2 on bin 3 of frame `frame` (an enhanced bin overflows over a 3e38 mic bin). */
static int fill_taps(float *taps, size_t elements, int t, int kind, int frame, size_t bins) {
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
    if (kind == 6) for (i = 0; i < 9u; ++i) taps[((size_t)frame * bins + 3u) * 18u + 2u * i] = 2.0f;
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
    if (h % 89 == 17) mr[3] = 3.0e38f;   /* finite, in range: the enhanced bin overflows under the taps of job kind 6 */
}

/* `slim` (freq_only) against `full`, packet for packet. Returns the number of differing calls. */
static int run_pair(DeepVqeStream *slim, DeepVqeStream *full, const DeepVqeGrid *g, const char *label) {
    const size_t nb = (size_t)g->n_bins, np = nb * (size_t)g->frames_per_call;
    float *x = (float *)malloc(np * 8u * sizeof(float));
    float *mr, *mi, *fr, *fi, *sre, *sim, *fre, *fim;
    uint32_t seed = 2463534242u + (uint32_t)g->n_fft + (uint32_t)g->frames_per_call;
    uint32_t gen_before = 0;
    int t, k, bad = 0, jobs = 0, real = 0, refused = 0, overflow = 0;

    if (!x) exit(2);
    mr = x; mi = x + np; fr = x + 2 * np; fi = x + 3 * np;
    sre = x + 4 * np; sim = x + 5 * np; fre = x + 6 * np; fim = x + 7 * np;
    for (t = 0; t < HOPS; ++t) {
        DeepVqeHopIO sio, fio;
        int src, frc, srm, frm, nan_hop = 0, huge_frame = -1, kind;

        for (k = 0; k < g->frames_per_call; ++k) {
            const size_t o = (size_t)k * nb;
            const int h = t * g->frames_per_call + k;
            fill_spectra(&seed, h, mr + o, mi + o, fr + o, fi + o, nb);
            if (h % 97 == 40) nan_hop = 1;
            if (h % 89 == 17) huge_frame = k;
        }
        if (t == HOPS / 4) { deepvqe_stream_reset(slim); deepvqe_stream_reset(full); }
        if (t == HOPS / 2 &&
            (deepvqe_stream_post_process_packet_freq(slim, 0, sre, sim) != -1 ||
             deepvqe_stream_post_process_packet_freq(full, 0, fre, fim) != -1))
            ++bad;   /* a post with nothing due */
        memset(&sio, 0x5A, sizeof sio);
        memset(&fio, 0x5A, sizeof fio);
        src = deepvqe_stream_pre_process_packet_freq(slim, mr, mi, fr, fi, &sio);
        frc = deepvqe_stream_pre_process_packet_freq(full, mr, mi, fr, fi, &fio);
        if (nan_hop) {   /* refused before anything moved, and no job is open on either */
            const uint32_t gs = deepvqe_stream_state_generation(slim), gf = deepvqe_stream_state_generation(full);
            if (src != DEEPVQE_STREAM_ERR_RANGE || frc != DEEPVQE_STREAM_ERR_RANGE || gs != gf ||
                deepvqe_stream_post_process_packet_freq(slim, 0, sre, sim) != -1 ||
                deepvqe_stream_post_process_packet_freq(full, 0, fre, fim) != -1) ++bad;
            ++refused;
            continue;
        }
        if (src != 1 || frc != 1 || !same_job(&sio, &fio)) {
            ++bad;
            break;
        }
        if (t == HOPS / 2) {
            DeepVqeHopIO again;
            if (deepvqe_stream_pre_process_packet_freq(slim, mr, mi, fr, fi, &again) != -1) ++bad;
        }
        ++jobs;
        kind = huge_frame >= 0 ? 6 : job_kind(jobs);
        srm = fill_taps(sio.outputs.taps, sio.outputs.taps_elements, jobs, kind, huge_frame, nb);
        frm = fill_taps(fio.outputs.taps, fio.outputs.taps_elements, jobs, kind, huge_frame, nb);
        if (t == HOPS / 3 || t == (2 * HOPS) / 3) {   /* reset with the packet due: it is dropped */
            deepvqe_stream_reset(slim);
            deepvqe_stream_reset(full);
            if (deepvqe_stream_post_process_packet_freq(slim, srm, sre, sim) != -1 ||
                deepvqe_stream_post_process_packet_freq(full, frm, fre, fim) != -1) ++bad;
            continue;
        }
        gen_before = deepvqe_stream_state_generation(slim);
        src = deepvqe_stream_post_process_packet_freq(slim, srm, sre, sim);
        frc = deepvqe_stream_post_process_packet_freq(full, frm, fre, fim);
        if (src != 1 || frc != 1 ||
            deepvqe_stream_state_generation(slim) != deepvqe_stream_state_generation(full) ||
            memcmp(sre, fre, np * sizeof(float)) != 0 || memcmp(sim, fim, np * sizeof(float)) != 0)
            ++bad;
        real += memcmp(sre, mr, np * sizeof(float)) != 0;
        overflow += kind == 6 && memcmp(sre, mr, np * sizeof(float)) != 0 && sre[(size_t)huge_frame * nb + 3u] == 0.0f &&
                    deepvqe_stream_state_generation(slim) > gen_before;
    }
    /* not vacuous: jobs ran, most packets carry a real spectrum, the run restarted the state, a NaN input was refused, an
     * enhanced overflow was caught */
    if (jobs < HOPS / 2 || real < HOPS / 2 || deepvqe_stream_state_generation(slim) < 4u || refused < 1 || overflow < 1) ++bad;
    if (bad) fprintf(stderr, "%s: %d differing call(s), the last at packet %d\n", label, bad, t);
    free(x);
    return bad;
}

/* The PCM entries on a freq_only stream, from a reset: -1, the pool bytes, generation, io and PCM unchanged, and a due
 * _freq packet is left as it was. */
static void check_pcm_refused(DeepVqeStream *slim, const unsigned char *pool, size_t bytes, const DeepVqeGrid *g,
                              const char *label) {
    const size_t n = (size_t)g->frames_per_call, hop = (size_t)g->hop, nb = (size_t)g->n_bins;
    int16_t *pcm = (int16_t *)malloc(n * hop * sizeof(int16_t));
    float *z = (float *)calloc(n * nb * 2u, sizeof(float));
    unsigned char *snap = (unsigned char *)malloc(bytes);
    DeepVqeHopIO io, before;
    uint32_t gen;
    size_t k;
    int untouched = 1, pre;

    if (!pcm || !z || !snap) exit(2);
    for (k = 0; k < n * hop; ++k) pcm[k] = (int16_t)(1234 + (int)k);
    deepvqe_stream_reset(slim);
    gen = deepvqe_stream_state_generation(slim);
    memset(&io, 0x5A, sizeof io);
    before = io;
    memcpy(snap, pool, bytes);
    CHECK(deepvqe_stream_pre_process_packet(slim, pcm, pcm, &io) == -1 &&
          deepvqe_stream_pre_process(slim, pcm, pcm, &io) == -1, "%s: PCM pre_process accepted", label);
    CHECK(memcmp(snap, pool, bytes) == 0 && memcmp(&io, &before, sizeof io) == 0 &&
          deepvqe_stream_state_generation(slim) == gen, "%s: a refused PCM pre_process changed the stream", label);
    pre = deepvqe_stream_pre_process_packet_freq(slim, z, z + n * nb, z, z + n * nb, &io);
    CHECK(pre == 1, "%s: _freq pre after the refusal", label);
    memcpy(snap, pool, bytes);
    CHECK(deepvqe_stream_post_process_packet(slim, 0, pcm) == -1 && deepvqe_stream_post_process(slim, 0, pcm) == -1,
          "%s: PCM post_process accepted", label);
    for (k = 0; k < n * hop; ++k) untouched &= pcm[k] == (int16_t)(1234 + (int)k);
    CHECK(untouched && memcmp(snap, pool, bytes) == 0, "%s: a refused PCM post_process wrote PCM or the pool", label);
    if (pre == 1) fill_taps(io.outputs.taps, io.outputs.taps_elements, 1, 0, 0, nb);
    CHECK(deepvqe_stream_post_process_packet_freq(slim, 0, z, z + n * nb) == 1, "%s: the due _freq packet was lost",
          label);
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
           ck_align16_size((3u * (size_t)g->hop + 2u * (size_t)g->n_bins * (1u + (size_t)g->frames_per_call)) * sizeof(float));
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
