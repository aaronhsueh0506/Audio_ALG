/* The PCM entries of ulcnet_stream through alignment events. An event restarts only the model (state, generation, one
 * identity hop): the stream's analysis and synthesis keep running, so the PCM of every hop, the event hops included,
 * equals the spectrum entries driven by an AcStft / AcIstft pair of the caller's. Checked here sample for sample on two
 * grids, for the one-hop f32 and int16 entries and for the packet entry (N = 8, against the one-hop stream), through
 * changed events, a solid false -> true edge, an event on hop #0 and two events in a row. The stand-in accelerator's mask
 * depends on the hops since the last model restart, so a model that was not restarted (or restarted twice) shows. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet_stream.h"
#include "stream_mem.h"
#include "stream_run.h"
#include "audio_stft.h"
#include "fft_wrapper.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL (line %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

enum { NF = 8, HOPS = 64, HOP_MAX = 512, BINS_MAX = 513 };

typedef struct Fake {
    unsigned seen;
    long since;   /* graph calls since the last restart */
    long jobs;
} Fake;

static void fake_fill(float *out, int bins, int frames, int k, long since) {
    int b;
    for (b = 0; b < bins; ++b) {
        out[(size_t)k * bins + b] = 0.55f + 0.04f * (float)((since + b) % 7);
        out[((size_t)frames + k) * bins + b] = 0.05f * (float)((since * 3 + b) % 5) - 0.1f;
    }
}

/* One job: frames 0..n-1 of the tensor. frame_reset (packets) marks the frames the graph restarts and bypasses. */
static void fake_run(Fake *f, const UlcnetHopIO *io, int frames) {
    const int bins = (int)(io->outputs.spectrum_ri_elements / (2u * (size_t)frames));
    int k;
    if (io->state_generation != f->seen) { f->seen = io->state_generation; f->since = 0; }
    for (k = 0; k < frames; ++k) {
        if (frames > 1 && io->inputs.frame_reset[k] != 0.0f) { f->since = -1; fake_fill(io->outputs.output, bins, frames, k, 0); }
        else fake_fill(io->outputs.output, bins, frames, k, f->since);
        ++f->since;
    }
    ++f->jobs;
}

static void signal_hop(int h, int hop, float *err, float *far) {
    int i;
    for (i = 0; i < hop; ++i) {
        const float t = (float)(h * hop + i);
        err[i] = 0.2f * sinf(0.037f * t) + 0.05f * sinf(0.31f * t);
        far[i] = 0.1f * sinf(0.051f * t + 1.0f);
    }
}

/* Events: changed at 0, 5, 21, 22; unlocked 28..32 so hop 33 is a solid false -> true edge; a last change at 50. */
static UlcnetStreamDelay event_of(int h) {
    UlcnetStreamDelay d;
    d.changed = (h == 0 || h == 5 || h == 21 || h == 22 || h == 50);
    d.solid = !(h >= 28 && h < 33);
    return d;
}

/* The edge rule, written here: hop #0 has no previous solid. */
static int is_edge_hop(int h) {
    const UlcnetStreamDelay d = event_of(h);
    return d.changed || (h > 0 && !event_of(h - 1).solid && d.solid);
}

static void *pool_of(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes ? bytes : 16) != 0) abort();
    return p;
}

static UlcnetStream *stream_make(const UlcnetGrid *g, int frames, int freq_only) {
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = *g;
    cfg.frames_per_call = frames;
    cfg.freq_only = freq_only;
    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0) abort();
    return ulcnet_stream_init(&cfg, pool_of((size_t)req.bytes), (size_t)req.bytes);
}

/* The caller's own framing around the spectrum entries. */
typedef struct Framing {
    FftHandle *fft;
    AcStft *err, *far;
    AcIstft *istft;
} Framing;

static void framing_make(Framing *fr, int n_fft) {
    AcStftConfig sc;
    size_t fb = fft_get_mem_size(n_fft), sb, ib;
    fr->fft = fft_init(pool_of(fb), fb, n_fft);
    ac_stft_config_default(&sc, n_fft);
    sc.ola = AC_OLA_ACCUMULATE;
    sc.fft = fr->fft;
    sb = ac_stft_get_mem_size(&sc);
    ib = ac_istft_get_mem_size(&sc);
    fr->err = ac_stft_init(pool_of(sb), sb, &sc);
    fr->far = ac_stft_init(pool_of(sb), sb, &sc);
    fr->istft = ac_istft_init(pool_of(ib), ib, &sc);
}

static long long energy(const int16_t *x, int n) {
    long long e = 0;
    int i;
    for (i = 0; i < n; ++i) e += (long long)x[i] * x[i];
    return e;
}

static void run_grid(int sample_rate, int n_fft) {
    const UlcnetGridSpec spec = { sample_rate, n_fft, 8 };
    UlcnetGrid g;
    UlcnetStream *s_f32, *s_i16, *s_ref, *s_pkt;
    Framing fr;
    Fake f_f32 = {0, 0, 0}, f_i16 = {0, 0, 0}, f_ref = {0, 0, 0}, f_pkt = {0, 0, 0};
    static int16_t pcm_f32[HOPS * HOP_MAX], pcm_i16[HOPS * HOP_MAX], pcm_ref[HOPS * HOP_MAX], pcm_pkt[HOPS * HOP_MAX];
    static float err[NF * HOP_MAX], far[NF * HOP_MAX], er[BINS_MAX], ei[BINS_MAX], fr_[BINS_MAX], fi[BINS_MAX];
    static float out_re[BINS_MAX], out_im[BINS_MAX], hop_out[HOP_MAX];
    static int16_t err_q[HOP_MAX], far_q[HOP_MAX];
    long long edge_energy = 0;
    int h, i, edges = 0, edge_hops_nonzero = 0;

    if (ulcnet_grid_make(&g, &spec) != ULCNET_GRID_OK) abort();
    s_f32 = stream_make(&g, 1, 0);
    s_i16 = stream_make(&g, 1, 0);
    s_ref = stream_make(&g, 1, 1);
    s_pkt = stream_make(&g, NF, 0);
    framing_make(&fr, g.n_fft);

    for (h = 0; h < HOPS; ++h) {
        UlcnetHopIO io;
        const UlcnetStreamDelay d = event_of(h);
        int pre;

        edges += is_edge_hop(h);
        signal_hop(h, g.hop, err, far);

        pre = ulcnet_stream_pre_process_f32(s_f32, err, far, d, &io);
        if (pre == 1) fake_run(&f_f32, &io, 1);
        CHECK(ulcnet_stream_post_process(s_f32, pre == 1 ? STREAM_RUN_OK : STREAM_RUN_NOT_STARTED, pcm_f32 + (size_t)h * g.hop) == 1,
              "%d: f32 post", h);

        /* the int16 entry: the same signal quantised the way the entry quantises */
        for (i = 0; i < g.hop; ++i) {
            err_q[i] = stream_saturate_s16(err[i] * 32768.0f);
            far_q[i] = stream_saturate_s16(far[i] * 32768.0f);
        }
        pre = ulcnet_stream_pre_process(s_i16, err_q, far_q, d, &io);
        if (pre == 1) fake_run(&f_i16, &io, 1);
        CHECK(ulcnet_stream_post_process(s_i16, pre == 1 ? STREAM_RUN_OK : STREAM_RUN_NOT_STARTED, pcm_i16 + (size_t)h * g.hop) == 1,
              "%d: i16 post", h);

        /* the reference: the caller's analysis and synthesis around the spectrum entries, never restarted */
        (void)ac_stft_push_planar(fr.err, err, er, ei);
        (void)ac_stft_push_planar(fr.far, far, fr_, fi);
        pre = ulcnet_stream_pre_process_freq(s_ref, er, ei, fr_, fi, d, &io);
        if (pre == 1) fake_run(&f_ref, &io, 1);
        CHECK(ulcnet_stream_post_process_freq(s_ref, pre == 1 ? STREAM_RUN_OK : STREAM_RUN_NOT_STARTED, out_re, out_im) == 1,
              "%d: ref post", h);
        CHECK(ac_istft_push_planar(fr.istft, out_re, out_im, hop_out) == g.hop, "%d: ref istft", h);
        for (i = 0; i < g.hop; ++i) pcm_ref[(size_t)h * g.hop + i] = stream_saturate_s16(hop_out[i]);
    }
    CHECK(memcmp(pcm_f32, pcm_ref, sizeof(int16_t) * (size_t)HOPS * g.hop) == 0,
          "%d Hz: the f32 entry differs from the spectrum entries around the caller's framing", sample_rate);
    CHECK(f_f32.jobs == f_ref.jobs && f_f32.jobs == HOPS - edges, "%d Hz: jobs f32 %ld ref %ld, want %d", sample_rate, f_f32.jobs,
          f_ref.jobs, HOPS - edges);

    /* the int16 entry equals the f32 entry fed the same quantised signal: its own check against the reference above is
     * not exact (quantisation), so the second reference is a stream driven through the f32 entry with err_q / 32768 */
    {
        UlcnetStream *s_q = stream_make(&g, 1, 0);
        Fake f_q = {0, 0, 0};
        static int16_t pcm_q[HOPS * HOP_MAX];
        for (h = 0; h < HOPS; ++h) {
            UlcnetHopIO io;
            int pre;
            signal_hop(h, g.hop, err, far);
            for (i = 0; i < g.hop; ++i) {
                err[i] = (float)stream_saturate_s16(err[i] * 32768.0f) / 32768.0f;
                far[i] = (float)stream_saturate_s16(far[i] * 32768.0f) / 32768.0f;
            }
            pre = ulcnet_stream_pre_process_f32(s_q, err, far, event_of(h), &io);
            if (pre == 1) fake_run(&f_q, &io, 1);
            CHECK(ulcnet_stream_post_process(s_q, pre == 1 ? STREAM_RUN_OK : STREAM_RUN_NOT_STARTED, pcm_q + (size_t)h * g.hop) == 1,
                  "%d: q post", h);
        }
        CHECK(memcmp(pcm_i16, pcm_q, sizeof(int16_t) * (size_t)HOPS * g.hop) == 0,
              "%d Hz: the int16 entry differs from the f32 entry on the same quantised signal", sample_rate);
    }

    /* the packet entry: N = 8 hops per job equal the one-hop stream */
    for (h = 0; h < HOPS; h += NF) {
        UlcnetHopIO io;
        UlcnetStreamDelay d[NF];
        for (i = 0; i < NF; ++i) {
            signal_hop(h + i, g.hop, err + (size_t)i * g.hop, far + (size_t)i * g.hop);
            d[i] = event_of(h + i);
        }
        CHECK(ulcnet_stream_pre_process_packet_f32(s_pkt, err, far, d, &io) == 1, "%d: packet pre", h);
        fake_run(&f_pkt, &io, NF);
        CHECK(ulcnet_stream_post_process(s_pkt, STREAM_RUN_OK, pcm_pkt + (size_t)h * g.hop) == 1, "%d: packet post", h);
    }
    CHECK(memcmp(pcm_f32, pcm_pkt, sizeof(int16_t) * (size_t)HOPS * g.hop) == 0,
          "%d Hz: the N = 8 packet differs from the one-hop stream through the events", sample_rate);
    CHECK(f_pkt.jobs == HOPS / NF, "%d Hz: %ld packet jobs", sample_rate, f_pkt.jobs);

    /* An event hop carries signal (it used to be exactly zero). Hop #0 is the start of the stream, whose analysis has no
     * history either way, so it is left out of the count. */
    for (h = 1, i = 0; h < HOPS; ++h) {
        if (is_edge_hop(h)) {
            const long long e = energy(pcm_f32 + (size_t)h * g.hop, g.hop);
            ++i;
            if (e > 0) ++edge_hops_nonzero;
            edge_energy += e;
        }
    }
    CHECK(i == edges - 1 && edges >= 6, "%d Hz: %d edge hops after hop #0 (want >= 5)", sample_rate, i);
    CHECK(edge_hops_nonzero == i && edge_energy > 1000000LL, "%d Hz: %d of %d event hops carry signal (energy %lld)", sample_rate,
          edge_hops_nonzero, i, edge_energy);
    CHECK(ulcnet_stream_state_generation(s_f32) == (uint32_t)edges && ulcnet_stream_state_generation(s_pkt) == 0u,
          "%d Hz: generation f32 %u (want %d, one per event), packet %u (want 0: the graph restarts the model)", sample_rate,
          ulcnet_stream_state_generation(s_f32), edges, ulcnet_stream_state_generation(s_pkt));
    printf("  %d Hz / fft %d: %d hops, %d events (every one after hop #0 carries signal, energy %lld), f32 == spectrum entries + framing, int16 == f32 on the quantised signal, N = 8 == N = 1\n",
           sample_rate, n_fft, HOPS, edges, edge_energy);
}

int main(void) {
    run_grid(16000, 256);
    run_grid(48000, 1024);
    if (g_fail) { printf("test_ulcnet_pcm_events: %d failure(s)\n", g_fail); return 1; }
    printf("test_ulcnet_pcm_events: ALL PASS\n");
    return 0;
}
