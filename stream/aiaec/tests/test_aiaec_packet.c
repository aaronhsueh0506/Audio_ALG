/* Align-ULCNet packet graph (frames_per_call N = 8 at 16k/256, layout 19) through the real public API, with a
 * deterministic host stand-in for the accelerator that implements the graph's semantics: a frame whose frame_reset
 * is 1.0 resets the model state to zero before the frame and bypasses it (identity mask, state stays zero), any
 * other frame steps the state. Align-ULCNet product only; built with -DAIAEC_TEST_CATALOG, which gives the N=8 row a
 * placeholder graph path (the production catalog refuses it).
 *
 *   spectra   the _freq family at N=8 equals the _freq family at N=1 applying the same events, bit for bit, for
 *             event streams: none, frame 1, frame 3, frame 8, two in one packet, adjacent, one in every packet, every
 *             frame, a solid false -> true edge on frame 2 of a packet, and one on frame 1 of a packet (the previous hop's
 *             solid carried across the packet boundary). One accelerator call per packet at N=8, none for an event hop at N=1.
 *   events    frame_reset carries exactly the edge pattern; the state generation never changes for an event at N=8;
 *             an all-event packet is still one SUBMIT job.
 *   families  _spec at N=8 equals _freq at N=8 fed by an independent analyser restarted at the event hops, and the
 *             PCM family equals _spec plus an external AcIstft reset at the hops AiaecWork.synthesis_restart_mask
 *             names, sample for sample, through the events.
 *   realign   the PCM family and _spec (+ external ISTFT) at N=8 equal the N=1 stream sample for sample with the same
 *             events (none, one hop, adjacent, first, last, across a packet boundary, solid false -> true after
 *             earlier hops, every hop, a full reset in the middle), with the accelerator's end-of-packet state and
 *             the vendor call count compared: one call per packet at N=8, one per non-event hop at N=1.
 *   pin       the PCM family returns exactly silence for an event hop and drops the audio of the hop before it; N = 8
 *             equals N = 1 through the event (the documented behaviour, pinned so a change must be deliberate).
 *   tail      a 16000-count sample at the last real input sample comes back model_hop later, through the zero block
 *             tail_zero_blocks asks for, and is lost without it.
 *   failure   an inference failure and a non-finite mask pass all N error spectra through; only the non-finite one
 *             advances the generation.
 *   gates     event count, NULL events, sequence, the work layout. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"
#include "audio_stft.h"
#include "stream_mem.h"
#include "ulcnet_prepost.h"

#if AIAEC_PRODUCT_MODEL_ID != AIAEC_MODEL_ID_ALIGN_ULCNET
#error "test_aiaec_packet.c is the Align-ULCNet product's test"
#endif

#define FFT 256
#define HOP 128
#define BINS 129
#define NF 8
#define HOP_IN (NF * HOP)
#define MAX_PACKETS 6
#define MAX_HOPS (MAX_PACKETS * NF)

static int g_fail = 0;
#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } while (0)
#define CHECK(cond, ...) do { if (!(cond)) FAIL(__VA_ARGS__); } while (0)

static void *alloc_aligned(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) { fprintf(stderr, "alloc fail\n"); exit(1); }
    memset(p, 0xA5, bytes);
    return p;
}

static AiaecConfig cfg_n(int n) {
    AiaecConfig cfg;
    cfg.sr = 16000; cfg.fft = FFT; cfg.hop_in = HOP * n;
    return cfg;
}

typedef struct Ctxs {
    AiaecCtx *blk;
    AiaecSpecCtx *spec;
    AiaecFreqCtx *freq;
    void *pb, *ps, *pf;
    AiaecInfo ib, is, if_;
} Ctxs;

static int ctxs_make(Ctxs *c, int n, int want_blk, int want_spec, int want_freq) {
    AiaecConfig cfg = cfg_n(n);
    StreamMemReq rq;
    int e = 0;
    memset(c, 0, sizeof(*c));
    if (want_blk) {
        if (aiaec_get_mem_size(&cfg, &rq) != AIAEC_OK) { FAIL("block mem size n=%d", n); return -1; }
        c->pb = alloc_aligned((size_t)rq.bytes);
        e = aiaec_init(&cfg, c->pb, (size_t)rq.bytes, &c->blk, &c->ib);
        CHECK(e == AIAEC_OK && c->blk != NULL, "init n=%d err=%d", n, e);
    }
    if (want_spec) {
        if (aiaec_get_mem_size_spec(&cfg, &rq) != AIAEC_OK) { FAIL("spec mem size n=%d", n); return -1; }
        c->ps = alloc_aligned((size_t)rq.bytes);
        e = aiaec_init_spec(&cfg, c->ps, (size_t)rq.bytes, &c->spec, &c->is);
        CHECK(e == AIAEC_OK && c->spec != NULL, "init_spec n=%d err=%d", n, e);
    }
    if (want_freq) {
        if (aiaec_get_mem_size_freq(&cfg, &rq) != AIAEC_OK) { FAIL("freq mem size n=%d", n); return -1; }
        c->pf = alloc_aligned((size_t)rq.bytes);
        e = aiaec_init_freq(&cfg, c->pf, (size_t)rq.bytes, &c->freq, &c->if_);
        CHECK(e == AIAEC_OK && c->freq != NULL, "init_freq n=%d err=%d", n, e);
    }
    return 0;
}

static void ctxs_free(Ctxs *c) { free(c->pb); free(c->ps); free(c->pf); }

/* ---- the accelerator stand-in ------------------------------------------------------------------------------- */

typedef enum { ACC_MODEL = 0, ACC_IDENTITY = 1, ACC_NAN = 2 } AccMode;

typedef struct Accel {
    float st[BINS];
    unsigned gen;
    int have_gen;
    int calls;
    unsigned gen_changes;   /* state_generation values the accelerator saw change */
    AccMode mode;
} Accel;

static void accel_init(Accel *a, unsigned gen, AccMode mode) {
    memset(a, 0, sizeof(*a));
    a->gen = gen; a->have_gen = 1; a->mode = mode;
}

/* The graph of layout 18 (N == 1, no frame_reset input) and 19 (frame_reset = inputs[4]). */
static void accel_run(Accel *a, const AiaecWork *w, int n) {
    const float *reset = n > 1 ? w->inputs[4].data : NULL;
    float *out = w->outputs[0].data;
    int k, b;

    if (w->state_generation != a->gen) { memset(a->st, 0, sizeof(a->st)); a->gen = w->state_generation; ++a->gen_changes; }
    ++a->calls;
    for (k = 0; k < n; ++k) {
        const float *emag = w->inputs[0].data + (size_t)k * BINS, *fmag = w->inputs[1].data + (size_t)k * BINS;
        const float *esin = w->inputs[3].data + (size_t)k * BINS;
        float *mre = out + (size_t)k * BINS, *mim = out + ((size_t)n + (size_t)k) * BINS;
        const int event = reset && reset[k] != 0.0f;
        if (event) memset(a->st, 0, sizeof(a->st));
        for (b = 0; b < BINS; ++b) {
            if (a->mode == ACC_IDENTITY || event) { mre[b] = 1.0f; mim[b] = 0.0f; continue; }
            a->st[b] = 0.5f * a->st[b] + 0.1f * emag[b] + 0.05f * fmag[b];
            {
                const float q = a->st[b] / (1.0f + fabsf(a->st[b]));
                mre[b] = 0.5f + 0.25f * q;
                mim[b] = 0.1f * esin[b] * q;
            }
        }
    }
    if (a->mode == ACC_NAN) out[BINS / 2] = NAN;
}

/* ---- the signal and its independent analysis ----------------------------------------------------------------- */

static uint32_t g_rng = 12345u;
static float rnd(void) { g_rng = g_rng * 1664525u + 1013904223u; return (float)((g_rng >> 8) & 0xffff) / 32768.0f - 1.0f; }

static float g_resid[MAX_HOPS * HOP], g_far[MAX_HOPS * HOP];
static float g_ere[MAX_HOPS * BINS], g_eim[MAX_HOPS * BINS], g_fre[MAX_HOPS * BINS], g_fim[MAX_HOPS * BINS];

typedef struct Ext { AcStft *stft; AcIstft *istft; void *ms, *mi; } Ext;

static void ext_make(Ext *x, int want_stft, int want_istft) {
    AcStftConfig sc;
    memset(x, 0, sizeof(*x));
    ac_stft_config_default(&sc, FFT);
    sc.ola = AC_OLA_ACCUMULATE;
    if (want_stft) {
        x->ms = alloc_aligned(ac_stft_get_mem_size(&sc));
        x->stft = ac_stft_init(x->ms, ac_stft_get_mem_size(&sc), &sc);
    }
    if (want_istft) {
        x->mi = alloc_aligned(ac_istft_get_mem_size(&sc));
        x->istft = ac_istft_init(x->mi, ac_istft_get_mem_size(&sc), &sc);
    }
    if ((want_stft && !x->stft) || (want_istft && !x->istft)) { fprintf(stderr, "framing init fail\n"); exit(1); }
}

static void ext_free(Ext *x) { free(x->ms); free(x->mi); }

static void make_signal(int hops) {
    Ext ae, af;
    int i, h;
    for (i = 0; i < hops * HOP; ++i) {
        g_far[i] = 0.2f * rnd();
        g_resid[i] = 0.05f * rnd() + 0.3f * ((i > 40) ? g_far[i - 40] : 0.0f);
    }
    ext_make(&ae, 1, 0); ext_make(&af, 1, 0);
    for (h = 0; h < hops; ++h) {
        ac_stft_push_planar(ae.stft, g_resid + (size_t)h * HOP, g_ere + (size_t)h * BINS, g_eim + (size_t)h * BINS);
        ac_stft_push_planar(af.stft, g_far + (size_t)h * HOP, g_fre + (size_t)h * BINS, g_fim + (size_t)h * BINS);
    }
    ext_free(&ae); ext_free(&af);
}

/* ---- event schedules ------------------------------------------------------------------------------------------ */

typedef struct Sched {
    const char *name;
    int packets;
    int reset_hop;   /* aiaec_reset_*() before this hop (a packet boundary), -1: none */
    AiaecAlignEvent ev[MAX_HOPS];
} Sched;

static void sched_clear(Sched *s, const char *name, int packets) {
    memset(s, 0, sizeof(*s));
    s->name = name; s->packets = packets; s->reset_hop = -1;
}

/* what the edge rule says, with the previous hop's solid carried across packets */
static void expected_edges(const Sched *s, int *edge) {
    int had = 0, prev = 0, h;
    for (h = 0; h < s->packets * NF; ++h) {
        edge[h] = s->ev[h].changed || (had && !prev && s->ev[h].solid);
        had = 1; prev = s->ev[h].solid;
    }
}

/* ---- drivers ------------------------------------------------------------------------------------------------- */

static void freq_input(AiaecFreqInput *fin, int hop0, int frames, const AiaecAlignEvent *ev) {
    memset(fin, 0, sizeof(*fin));
    fin->kind = AIAEC_INPUT_ALIGN_ULCNET;
    fin->frames = frames;
    fin->u.align_ulcnet.err_re = g_ere + (size_t)hop0 * BINS; fin->u.align_ulcnet.err_im = g_eim + (size_t)hop0 * BINS;
    fin->u.align_ulcnet.far_re = g_fre + (size_t)hop0 * BINS; fin->u.align_ulcnet.far_im = g_fim + (size_t)hop0 * BINS;
    fin->u.align_ulcnet.events = ev + hop0;
    fin->u.align_ulcnet.event_count = frames;
}

typedef struct Run {
    float re[MAX_HOPS * BINS], im[MAX_HOPS * BINS];
    int calls;
    unsigned gen_changes;
    unsigned first_gen, last_gen;
    int resets_seen;                      /* frame_reset values equal to 1 over the run */
    int reset_pattern_ok;                 /* frame_reset == the edge rule, packet by packet */
    int bypass_hops;                      /* N=1: BYPASS blocks */
} Run;

static void run_freq_n1(const Sched *s, Run *r) {
    Ctxs c;
    Accel acc;
    int h;
    memset(r, 0, sizeof(*r));
    if (ctxs_make(&c, 1, 0, 0, 1) != 0) return;
    accel_init(&acc, aiaec_state_generation_freq(c.freq), ACC_MODEL);
    for (h = 0; h < s->packets * NF; ++h) {
        AiaecFreqInput fin;
        AiaecWork w;
        int rc;
        freq_input(&fin, h, 1, s->ev);
        rc = aiaec_preprocess_freq(c.freq, &fin, &w);
        CHECK(rc == AIAEC_OK, "n1 %s: preprocess hop %d rc=%d", s->name, h, rc);
        if (w.action == AIAEC_SUBMIT) accel_run(&acc, &w, 1); else ++r->bypass_hops;
        rc = aiaec_postprocess_freq(c.freq, 0, r->re + (size_t)h * BINS, r->im + (size_t)h * BINS);
        CHECK(rc == AIAEC_OK, "n1 %s: postprocess hop %d rc=%d", s->name, h, rc);
    }
    r->calls = acc.calls; r->gen_changes = acc.gen_changes;
    ctxs_free(&c);
}

static void run_freq_n8(const Sched *s, Run *r, AccMode mode) {
    Ctxs c;
    Accel acc;
    int edge[MAX_HOPS];
    int p, k;
    memset(r, 0, sizeof(*r));
    r->reset_pattern_ok = 1;
    expected_edges(s, edge);
    if (ctxs_make(&c, NF, 0, 0, 1) != 0) return;
    accel_init(&acc, aiaec_state_generation_freq(c.freq), mode);
    r->first_gen = acc.gen;
    for (p = 0; p < s->packets; ++p) {
        AiaecFreqInput fin;
        AiaecWork w;
        int rc;
        freq_input(&fin, p * NF, NF, s->ev);
        rc = aiaec_preprocess_freq(c.freq, &fin, &w);
        CHECK(rc == AIAEC_OK, "n8 %s: preprocess packet %d rc=%d", s->name, p, rc);
        CHECK(w.action == AIAEC_SUBMIT && w.input_count == 5 && w.output_count == 1 && w.synthesis_restarted == 0,
              "n8 %s: packet %d work action=%d inputs=%d restarted=%d", s->name, p, (int)w.action, w.input_count, w.synthesis_restarted);
        for (k = 0; k < NF && w.input_count == 5; ++k) {
            const float want = edge[p * NF + k] ? 1.0f : 0.0f;
            if (w.inputs[4].data[k] != want) r->reset_pattern_ok = 0;
            r->resets_seen += w.inputs[4].data[k] == 1.0f;
        }
        accel_run(&acc, &w, NF);
        r->last_gen = w.state_generation;
        rc = aiaec_postprocess_freq(c.freq, 0, r->re + (size_t)p * NF * BINS, r->im + (size_t)p * NF * BINS);
        CHECK(rc == AIAEC_OK, "n8 %s: postprocess packet %d rc=%d", s->name, p, rc);
    }
    r->calls = acc.calls; r->gen_changes = acc.gen_changes;
    ctxs_free(&c);
}

/* the frames that are events must carry the error spectrum unchanged, the others must not */
static void check_event_frames(const Sched *s, const Run *r, const char *what) {
    int edge[MAX_HOPS], h;
    expected_edges(s, edge);
    for (h = 0; h < s->packets * NF; ++h) {
        const int same = memcmp(r->re + (size_t)h * BINS, g_ere + (size_t)h * BINS, BINS * sizeof(float)) == 0 &&
                         memcmp(r->im + (size_t)h * BINS, g_eim + (size_t)h * BINS, BINS * sizeof(float)) == 0;
        CHECK(same == (edge[h] != 0), "%s %s: hop %d %s the error spectrum (event %d)", what, s->name, h,
              same ? "equals" : "differs from", edge[h]);
    }
}

static int section_spectra(void) {
    Sched s[11];
    int i, h, n = 0;

    sched_clear(&s[n], "none", 3); ++n;
    sched_clear(&s[n], "frame1", 3); s[n].ev[0].changed = 1; ++n;
    sched_clear(&s[n], "frame3", 3); s[n].ev[2].changed = 1; ++n;
    sched_clear(&s[n], "frame8", 3); s[n].ev[7].changed = 1; ++n;
    sched_clear(&s[n], "two_in_one_packet", 3); s[n].ev[2].changed = 1; s[n].ev[5].changed = 1; ++n;
    sched_clear(&s[n], "adjacent", 3); s[n].ev[1].changed = 1; s[n].ev[2].changed = 1; ++n;
    sched_clear(&s[n], "every_packet", 3);
    s[n].ev[2].changed = 1; s[n].ev[8].changed = 1; s[n].ev[15].changed = 1; s[n].ev[19].changed = 1; ++n;
    sched_clear(&s[n], "all_frames", 3); for (h = 0; h < NF; ++h) s[n].ev[h].changed = 1; ++n;
    /* solid false -> true on frame 2 of packet 1 */
    sched_clear(&s[n], "solid_edge_frame2", 3); for (h = 9; h < 24; ++h) s[n].ev[h].solid = 1; ++n;
    /* the previous hop of the edge is the last hop of the previous packet */
    sched_clear(&s[n], "solid_edge_across_packets", 3); for (h = 8; h < 24; ++h) s[n].ev[h].solid = 1; ++n;
    /* solid true from the first hop is no edge (no previous hop), then it drops and returns inside one packet */
    sched_clear(&s[n], "solid_drop_return", 3);
    for (h = 0; h < 24; ++h) s[n].ev[h].solid = !(h >= 10 && h < 12); ++n;

    for (i = 0; i < n; ++i) {
        static Run r1, r8;
        int edge[MAX_HOPS], edges = 0;
        run_freq_n1(&s[i], &r1);
        run_freq_n8(&s[i], &r8, ACC_MODEL);
        expected_edges(&s[i], edge);
        for (h = 0; h < s[i].packets * NF; ++h) edges += edge[h];
        CHECK(memcmp(r1.re, r8.re, sizeof(float) * (size_t)s[i].packets * NF * BINS) == 0 &&
              memcmp(r1.im, r8.im, sizeof(float) * (size_t)s[i].packets * NF * BINS) == 0,
              "%s: N=8 spectra differ from the N=1 stream applying the same events", s[i].name);
        CHECK(r8.calls == s[i].packets, "%s: %d vendor calls for %d packets", s[i].name, r8.calls, s[i].packets);
        CHECK(r1.calls == s[i].packets * NF - edges && r1.bypass_hops == edges, "%s: N=1 calls %d bypass %d for %d events",
              s[i].name, r1.calls, r1.bypass_hops, edges);
        CHECK(r8.gen_changes == 0 && r8.last_gen == r8.first_gen, "%s: N=8 state generation changed (%u changes) on events", s[i].name, r8.gen_changes);
        CHECK(r8.reset_pattern_ok && r8.resets_seen == edges, "%s: frame_reset pattern (%d ones, %d edges)", s[i].name, r8.resets_seen, edges);
        check_event_frames(&s[i], &r8, "n8");
        check_event_frames(&s[i], &r1, "n1");
        printf("section_spectra[%s]: %d packets, %d events, %d vendor calls (N=1: %d), bit-equal to N=1\n", s[i].name,
               s[i].packets, edges, r8.calls, r1.calls);
    }
    return 0;
}

/* ---- the three families --------------------------------------------------------------------------------------- */

static void block_input(AiaecInput *in, int block, const AiaecAlignEvent *ev) {
    memset(in, 0, sizeof(*in));
    in->kind = AIAEC_INPUT_ALIGN_ULCNET;
    in->samples = HOP_IN;
    in->u.align_ulcnet.residual_in = g_resid + (size_t)block * HOP_IN;
    in->u.align_ulcnet.far_in = g_far + (size_t)block * HOP_IN;
    in->u.align_ulcnet.events = ev + (size_t)block * NF;
    in->u.align_ulcnet.event_count = NF;
}

/* _freq at N=8 fed by an analyser that restarts (zero history) at every event hop of the schedule: what _spec must equal */
static void run_freq_restarted_analyser(const Sched *s, Run *r) {
    static float ere[MAX_HOPS * BINS], eim[MAX_HOPS * BINS], fre[MAX_HOPS * BINS], fim[MAX_HOPS * BINS];
    int edge[MAX_HOPS], h;
    Ext ae, af;
    Ctxs c;
    Accel acc;
    int p;

    expected_edges(s, edge);
    ext_make(&ae, 1, 0); ext_make(&af, 1, 0);
    for (h = 0; h < s->packets * NF; ++h) {
        if (edge[h]) { ac_stft_reset(ae.stft); ac_stft_reset(af.stft); }
        ac_stft_push_planar(ae.stft, g_resid + (size_t)h * HOP, ere + (size_t)h * BINS, eim + (size_t)h * BINS);
        ac_stft_push_planar(af.stft, g_far + (size_t)h * HOP, fre + (size_t)h * BINS, fim + (size_t)h * BINS);
    }
    ext_free(&ae); ext_free(&af);
    memset(r, 0, sizeof(*r));
    if (ctxs_make(&c, NF, 0, 0, 1) != 0) return;
    accel_init(&acc, aiaec_state_generation_freq(c.freq), ACC_MODEL);
    for (p = 0; p < s->packets; ++p) {
        AiaecFreqInput fin;
        AiaecWork w;
        memset(&fin, 0, sizeof(fin));
        fin.kind = AIAEC_INPUT_ALIGN_ULCNET; fin.frames = NF;
        fin.u.align_ulcnet.err_re = ere + (size_t)p * NF * BINS; fin.u.align_ulcnet.err_im = eim + (size_t)p * NF * BINS;
        fin.u.align_ulcnet.far_re = fre + (size_t)p * NF * BINS; fin.u.align_ulcnet.far_im = fim + (size_t)p * NF * BINS;
        fin.u.align_ulcnet.events = s->ev + (size_t)p * NF; fin.u.align_ulcnet.event_count = NF;
        CHECK(aiaec_preprocess_freq(c.freq, &fin, &w) == AIAEC_OK && w.synthesis_restart_mask == 0u && w.synthesis_restarted == 0,
              "freq with a restarted analyser: packet %d (the _freq family never reports a restart)", p);
        accel_run(&acc, &w, NF);
        CHECK(aiaec_postprocess_freq(c.freq, 0, r->re + (size_t)p * NF * BINS, r->im + (size_t)p * NF * BINS) == AIAEC_OK, "freq post %d", p);
    }
    ctxs_free(&c);
}

static uint64_t edge_mask(const int *edge, int packet) {
    uint64_t m = 0;
    int k;
    for (k = 0; k < NF; ++k) if (edge[packet * NF + k]) m |= (uint64_t)1u << k;
    return m;
}

static int section_families(void) {
    static Sched s;
    static Run rf;
    static float sre[MAX_HOPS * BINS], sim[MAX_HOPS * BINS];
    static int16_t pcm[MAX_HOPS * HOP], want_pcm[MAX_HOPS * HOP];
    Ctxs c;
    Accel a_blk, a_spec;
    Ext syn;
    float hopbuf[HOP];
    int edge[MAX_HOPS];
    int p, k, i, h;

    sched_clear(&s, "families", 4);
    s.ev[2].changed = 1; s.ev[3].changed = 1; s.ev[9].changed = 1; s.ev[16].changed = 1;
    for (h = 12; h < 32; ++h) s.ev[h].solid = 1;   /* solid edge at hop 12 */
    expected_edges(&s, edge);

    run_freq_restarted_analyser(&s, &rf);
    if (ctxs_make(&c, NF, 1, 1, 0) != 0) return 1;
    accel_init(&a_blk, aiaec_state_generation(c.blk), ACC_MODEL);
    accel_init(&a_spec, aiaec_state_generation_spec(c.spec), ACC_MODEL);
    ext_make(&syn, 0, 1);

    for (p = 0; p < s.packets; ++p) {
        AiaecInput in;
        AiaecWork w;
        const uint64_t want_mask = edge_mask(edge, p);
        int rc;

        block_input(&in, p, s.ev);
        rc = aiaec_preprocess_spec(c.spec, &in, &w);
        CHECK(rc == AIAEC_OK && w.action == AIAEC_SUBMIT && w.synthesis_restart_mask == want_mask &&
              w.synthesis_restarted == (want_mask != 0u), "spec packet %d rc=%d mask %llx want %llx", p, rc,
              (unsigned long long)w.synthesis_restart_mask, (unsigned long long)want_mask);
        accel_run(&a_spec, &w, NF);
        rc = aiaec_postprocess_spec(c.spec, 0, sre + (size_t)p * NF * BINS, sim + (size_t)p * NF * BINS);
        CHECK(rc == AIAEC_OK, "postprocess_spec packet %d rc=%d", p, rc);

        rc = aiaec_preprocess(c.blk, &in, &w);
        CHECK(rc == AIAEC_OK && w.action == AIAEC_SUBMIT && w.synthesis_restart_mask == want_mask &&
              w.synthesis_restarted == (want_mask != 0u), "block packet %d rc=%d mask %llx", p, rc,
              (unsigned long long)w.synthesis_restart_mask);
        accel_run(&a_blk, &w, NF);
        rc = aiaec_postprocess(c.blk, 0, pcm + (size_t)p * HOP_IN);
        CHECK(rc == AIAEC_OK, "postprocess packet %d rc=%d", p, rc);

        for (k = 0; k < NF; ++k) {
            const size_t hop = (size_t)p * NF + (size_t)k;
            if ((want_mask >> k) & 1u) ac_istft_reset(syn.istft);
            ac_istft_push_planar(syn.istft, sre + hop * BINS, sim + hop * BINS, hopbuf);
            for (i = 0; i < HOP; ++i) want_pcm[hop * HOP + (size_t)i] = stream_saturate_s16(hopbuf[i]);
        }
    }
    CHECK(memcmp(sre, rf.re, sizeof(float) * (size_t)s.packets * NF * BINS) == 0 &&
          memcmp(sim, rf.im, sizeof(float) * (size_t)s.packets * NF * BINS) == 0,
          "_spec at N=8 differs from _freq at N=8 fed by an analyser restarted at the event hops");
    CHECK(memcmp(pcm, want_pcm, sizeof(int16_t) * (size_t)s.packets * HOP_IN) == 0,
          "PCM family differs from _spec + an external AcIstft reset at synthesis_restart_mask");
    CHECK(a_blk.calls == s.packets && a_spec.calls == s.packets && a_blk.gen_changes == 0 && a_spec.gen_changes == 0,
          "calls %d/%d generation changes %u/%u", a_blk.calls, a_spec.calls, a_blk.gen_changes, a_spec.gen_changes);
    {   /* the PCM output is not silence: it must carry the signal */
        long long energy = 0;
        for (i = 0; i < s.packets * HOP_IN; ++i) energy += (long long)pcm[i] * pcm[i];
        CHECK(energy > 100000, "PCM output is (nearly) silent: energy %lld", energy);
    }
    ext_free(&syn); ctxs_free(&c);
    printf("section_families: _spec == _freq(restarted analyser), PCM == _spec + ISTFT reset at the mask, through %d events at N=8\n", 5);
    return 0;
}

/* ---- the PCM-domain entries against the N=1 stream ----------------------------------------------------------- */

typedef struct PcmRun {
    int16_t pcm[MAX_HOPS * HOP];
    float spec_re[MAX_HOPS * BINS], spec_im[MAX_HOPS * BINS];   /* the _spec family, N=1 or N=8 */
    float st_end[MAX_PACKETS][BINS];                            /* the accelerator state at each packet end */
    int restart[MAX_HOPS];                                      /* hop restarted (block family mask) */
    int calls, pre_ok, post_ok;
    unsigned gen_changes;
} PcmRun;

/* the accelerator state a following hop would start from: a pending generation change zeroes it first */
static void effective_state(const Accel *a, unsigned gen_now, float *out) {
    if (gen_now != a->gen) memset(out, 0, sizeof(a->st)); else memcpy(out, a->st, sizeof(a->st));
}

/* n = 1 or NF; PCM block family when !spec, else the _spec family with an external ISTFT reset at the mask */
static void run_pcm(const Sched *s, int n, int spec, PcmRun *r) {
    Ctxs c;
    Accel acc;
    Ext syn;
    float hopbuf[HOP];
    const int hops = s->packets * NF;
    int b, k, i;

    memset(r, 0, sizeof(*r));
    if (ctxs_make(&c, n, !spec, spec, 0) != 0) return;
    accel_init(&acc, spec ? aiaec_state_generation_spec(c.spec) : aiaec_state_generation(c.blk), ACC_MODEL);
    ext_make(&syn, 0, spec);
    for (b = 0; b < hops / n; ++b) {
        AiaecInput in;
        AiaecWork w;
        const int h0 = b * n;
        int rc;

        if (h0 == s->reset_hop) {
            if (spec) { aiaec_reset_spec(c.spec); ac_istft_reset(syn.istft); } else aiaec_reset(c.blk);
        }
        memset(&in, 0, sizeof(in));
        in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.samples = HOP * n;
        in.u.align_ulcnet.residual_in = g_resid + (size_t)h0 * HOP; in.u.align_ulcnet.far_in = g_far + (size_t)h0 * HOP;
        in.u.align_ulcnet.events = s->ev + h0; in.u.align_ulcnet.event_count = n;
        rc = spec ? aiaec_preprocess_spec(c.spec, &in, &w) : aiaec_preprocess(c.blk, &in, &w);
        CHECK(rc == AIAEC_OK, "%s n=%d: preprocess block %d rc=%d", s->name, n, b, rc);
        if (rc != AIAEC_OK) break;
        ++r->pre_ok;
        CHECK(w.synthesis_restarted == (w.synthesis_restart_mask != 0u) && (n == NF || (w.synthesis_restart_mask >> 1) == 0u),
              "%s n=%d: block %d restart fields %d / %llx", s->name, n, b, w.synthesis_restarted, (unsigned long long)w.synthesis_restart_mask);
        for (k = 0; k < n; ++k) r->restart[h0 + k] = (int)((w.synthesis_restart_mask >> k) & 1u);
        if (w.action == AIAEC_SUBMIT) accel_run(&acc, &w, n);
        if (spec) {
            float *re = r->spec_re + (size_t)h0 * BINS, *im = r->spec_im + (size_t)h0 * BINS;
            rc = aiaec_postprocess_spec(c.spec, 0, re, im);
            for (k = 0; k < n; ++k) {
                if ((w.synthesis_restart_mask >> k) & 1u) ac_istft_reset(syn.istft);
                ac_istft_push_planar(syn.istft, re + (size_t)k * BINS, im + (size_t)k * BINS, hopbuf);
                for (i = 0; i < HOP; ++i) r->pcm[(size_t)(h0 + k) * HOP + (size_t)i] = stream_saturate_s16(hopbuf[i]);
            }
        } else {
            rc = aiaec_postprocess(c.blk, 0, r->pcm + (size_t)h0 * HOP);
        }
        CHECK(rc == AIAEC_OK, "%s n=%d: postprocess block %d rc=%d", s->name, n, b, rc);
        if (rc == AIAEC_OK) ++r->post_ok;
        if ((h0 + n) % NF == 0)
            effective_state(&acc, spec ? aiaec_state_generation_spec(c.spec) : aiaec_state_generation(c.blk),
                            r->st_end[(h0 + n) / NF - 1]);
    }
    r->calls = acc.calls; r->gen_changes = acc.gen_changes;
    ext_free(&syn); ctxs_free(&c);
}

static int first_diff(const int16_t *a, const int16_t *b, int n, int *maxd, int *count) {
    int i, first = -1;
    *maxd = 0; *count = 0;
    for (i = 0; i < n; ++i) {
        const int d = abs((int)a[i] - (int)b[i]);
        if (d) { if (first < 0) first = i; ++*count; }
        if (d > *maxd) *maxd = d;
    }
    return first;
}

static void realign_case(const Sched *s, const PcmRun *none) {
    static PcmRun b1, b8, p8, n8;
    int edge[MAX_HOPS], edges = 0, h, p, maxd, cnt, first;
    const int hops = s->packets * NF;
    Ctxs i1, i8;

    expected_edges(s, edge);
    for (h = 0; h < hops; ++h) edges += edge[h];
    run_pcm(s, 1, 0, &b1);
    run_pcm(s, NF, 0, &b8);
    run_pcm(s, NF, 1, &p8);
    run_pcm(s, 1, 1, &n8);

    CHECK(ctxs_make(&i1, 1, 1, 0, 0) == 0 && ctxs_make(&i8, NF, 1, 0, 0) == 0 &&
          i1.ib.trim_samples == HOP && i8.ib.trim_samples == HOP, "%s: PCM trim N=1 %d N=8 %d", s->name, i1.ib.trim_samples, i8.ib.trim_samples);
    ctxs_free(&i1); ctxs_free(&i8);
    first = first_diff(b1.pcm, b8.pcm, hops * HOP, &maxd, &cnt);
    CHECK(first < 0, "%s: PCM N=8 differs from N=1 in %d samples (first %d, max %d)", s->name, cnt, first, maxd);
    first = first_diff(b1.pcm, p8.pcm, hops * HOP, &maxd, &cnt);
    CHECK(first < 0, "%s: _spec N=8 + external ISTFT differs from N=1 PCM in %d samples (first %d, max %d)", s->name, cnt, first, maxd);
    first = first_diff(b1.pcm, n8.pcm, hops * HOP, &maxd, &cnt);
    CHECK(first < 0, "%s: _spec N=1 + external ISTFT differs from N=1 PCM in %d samples (first %d)", s->name, cnt, first);
    CHECK(memcmp(n8.spec_re, p8.spec_re, sizeof(float) * (size_t)hops * BINS) == 0 &&
          memcmp(n8.spec_im, p8.spec_im, sizeof(float) * (size_t)hops * BINS) == 0, "%s: _spec spectra N=8 differ from N=1", s->name);
    for (p = 0; p < s->packets; ++p)
        CHECK(memcmp(b1.st_end[p], b8.st_end[p], sizeof(b1.st_end[p])) == 0, "%s: accelerator state at the end of packet %d differs (N=1 vs N=8)", s->name, p);
    for (h = 0; h < hops; ++h)
        CHECK(b8.restart[h] == b1.restart[h] && b1.restart[h] == (edge[h] != 0) && p8.restart[h] == b1.restart[h],
              "%s: hop %d restart flags N=1 %d N=8 %d _spec %d, edge %d", s->name, h, b1.restart[h], b8.restart[h], p8.restart[h], edge[h]);
    CHECK(b8.calls == s->packets && p8.calls == s->packets && b1.calls == hops - edges && n8.calls == hops - edges,
          "%s: vendor calls N=8 %d/%d (want %d), N=1 %d/%d (want %d)", s->name, b8.calls, p8.calls, s->packets, b1.calls, n8.calls, hops - edges);
    CHECK(b8.pre_ok == s->packets && b8.post_ok == s->packets && b1.pre_ok == hops && b1.post_ok == hops && p8.pre_ok == p8.post_ok &&
          n8.pre_ok == n8.post_ok, "%s: pre/post pairing", s->name);
    CHECK(b8.gen_changes == (unsigned)(s->reset_hop >= 0), "%s: N=8 accelerator saw %u state-generation changes (reset only)", s->name, b8.gen_changes);
    if (none) {   /* hops before the first event use the incoming state and history: identical to the event-free run */
        int fe = hops;
        for (h = hops - 1; h >= 0; --h) if (edge[h]) fe = h;
        CHECK(memcmp(b8.pcm, none->pcm, sizeof(int16_t) * (size_t)fe * HOP) == 0, "%s: output before the first event hop (%d) differs from the event-free run", s->name, fe);
        if (fe < hops)
            CHECK(memcmp(b8.pcm + (size_t)fe * HOP, none->pcm + (size_t)fe * HOP, sizeof(int16_t) * HOP) != 0 || fe >= hops,
                  "%s: the event hop %d left the output unchanged (the event had no effect)", s->name, fe);
    }
    printf("section_realign[%s]: %d hops, %d events, PCM N=8 == N=1 (%d samples), _spec+ISTFT == PCM; vendor calls N=8 %d (packets %d), N=1 %d\n",
           s->name, hops, edges, hops * HOP, b8.calls, s->packets, b1.calls);
}

static int section_realign(void) {
    static Sched sc[14];
    static PcmRun none;
    int i, h, n = 0;

    sched_clear(&sc[n], "none", 3); ++n;
    sched_clear(&sc[n], "hop3", 3); sc[n].ev[2].changed = 1; ++n;
    sched_clear(&sc[n], "adjacent", 3); sc[n].ev[2].changed = 1; sc[n].ev[3].changed = 1; ++n;
    sched_clear(&sc[n], "first_of_stream", 3); sc[n].ev[0].changed = 1; ++n;
    sched_clear(&sc[n], "last_of_packet", 3); sc[n].ev[7].changed = 1; ++n;
    sched_clear(&sc[n], "last_of_stream", 3); sc[n].ev[23].changed = 1; ++n;
    sched_clear(&sc[n], "first_of_next_packet", 3); sc[n].ev[8].changed = 1; ++n;
    sched_clear(&sc[n], "across_boundary", 3); sc[n].ev[7].changed = 1; sc[n].ev[8].changed = 1; ++n;
    sched_clear(&sc[n], "event_in_next_packet", 3); sc[n].ev[12].changed = 1; ++n;
    sched_clear(&sc[n], "solid_false_true", 3); for (h = 10; h < 24; ++h) sc[n].ev[h].solid = 1; ++n;
    sched_clear(&sc[n], "solid_false_true_and_changed", 3);
    for (h = 5; h < 24; ++h) sc[n].ev[h].solid = 1;
    sc[n].ev[13].changed = 1; sc[n].ev[18].changed = 1; ++n;
    sched_clear(&sc[n], "every_hop", 3); for (h = 0; h < 24; ++h) sc[n].ev[h].changed = 1; ++n;
    sched_clear(&sc[n], "full_reset_after_events", 4);
    sc[n].ev[2].changed = 1; sc[n].ev[13].changed = 1; sc[n].ev[26].changed = 1; sc[n].reset_hop = 16; ++n;

    run_pcm(&sc[0], NF, 0, &none);
    for (i = 0; i < n; ++i) realign_case(&sc[i], &none);

    {   /* a full reset: what follows equals a fresh stream fed the same remaining input, on both entries */
        static PcmRun full, fresh;
        static Sched tail;
        const Sched *s = &sc[n - 1];
        int fe = s->reset_hop;
        sched_clear(&tail, "fresh_tail", s->packets - fe / NF);
        memcpy(tail.ev, s->ev + fe, sizeof(AiaecAlignEvent) * (size_t)(s->packets * NF - fe));
        run_pcm(s, NF, 0, &full);
        /* the fresh stream sees the remaining input as its own start: shift the global signal */
        {
            static float rs[MAX_HOPS * HOP], fs[MAX_HOPS * HOP];
            memcpy(rs, g_resid, sizeof(rs)); memcpy(fs, g_far, sizeof(fs));
            memmove(g_resid, g_resid + (size_t)fe * HOP, sizeof(float) * (size_t)(s->packets * NF - fe) * HOP);
            memmove(g_far, g_far + (size_t)fe * HOP, sizeof(float) * (size_t)(s->packets * NF - fe) * HOP);
            run_pcm(&tail, NF, 0, &fresh);
            memcpy(g_resid, rs, sizeof(rs)); memcpy(g_far, fs, sizeof(fs));
        }
        CHECK(memcmp(full.pcm + (size_t)fe * HOP, fresh.pcm, sizeof(int16_t) * (size_t)(s->packets * NF - fe) * HOP) == 0,
              "after a full reset the PCM family differs from a fresh stream fed the same remaining input");
    }
    printf("section_realign: %d scenarios, PCM and _spec at N=8 equal the N=1 stream\n", n);
    return 0;
}

/* ---- the PCM family at an event hop ---------------------------------------------------------------------------- */

static long long hop_energy(const int16_t *pcm, int hop) {
    long long e = 0;
    int i;
    for (i = 0; i < HOP; ++i) e += (long long)pcm[hop * HOP + i] * pcm[hop * HOP + i];
    return e;
}

/* The PCM entries restart their own STFT/ISTFT at an event hop, so the hop that carries the event returns exactly
 * silence and the audio of the input hop before it is not output (one hop of audio per event); hops before the event are
 * those of the event-free stream, the hop after it is not. N = 8 returns the same PCM as N = 1. This is the documented
 * behaviour (aiaec.h "Alignment events"); the _freq family, which restarts nothing but the model, has no such gap. */
static int section_event_hop_pin(void) {
    static Sched none, ev;
    static PcmRun r_none, r_ev1, r_ev8;
    const int event = 11, hops = 3 * NF;
    int h, i;

    sched_clear(&none, "pin_none", 3);
    sched_clear(&ev, "pin_event", 3);
    ev.ev[event].changed = 1;
    run_pcm(&none, 1, 0, &r_none);
    run_pcm(&ev, 1, 0, &r_ev1);
    run_pcm(&ev, NF, 0, &r_ev8);

    for (i = 0; i < HOP; ++i) {
        CHECK(r_ev1.pcm[event * HOP + i] == 0 && r_ev8.pcm[event * HOP + i] == 0, "event hop sample %d is not exactly zero (N=1 %d, N=8 %d)", i,
              r_ev1.pcm[event * HOP + i], r_ev8.pcm[event * HOP + i]);
    }
    CHECK(hop_energy(r_none.pcm, event) > 1000000, "control: the event-free stream is silent at hop %d (%lld)", event, hop_energy(r_none.pcm, event));
    CHECK(hop_energy(r_ev1.pcm, event - 1) > 1000000 && hop_energy(r_ev1.pcm, event + 1) > 1000000,
          "the hops around the event are silent: %lld / %lld", hop_energy(r_ev1.pcm, event - 1), hop_energy(r_ev1.pcm, event + 1));
    CHECK(memcmp(r_ev1.pcm, r_none.pcm, sizeof(int16_t) * (size_t)event * HOP) == 0, "hops before the event differ from the event-free stream");
    CHECK(memcmp(r_ev1.pcm + (event + 1) * HOP, r_none.pcm + (event + 1) * HOP, sizeof(int16_t) * HOP) != 0,
          "the hop after the event equals the event-free stream (the history was not discarded)");
    CHECK(memcmp(r_ev1.pcm, r_ev8.pcm, sizeof(int16_t) * (size_t)hops * HOP) == 0, "PCM at N=8 differs from N=1 through an event");
    for (h = 0; h < hops; ++h) CHECK(r_ev1.restart[h] == (h == event), "restart flag at hop %d", h);
    printf("section_event_hop_pin: the event hop is exactly silent, N=8 == N=1, neighbours carry signal\n");
    return 0;
}

/* ---- the generic binding ------------------------------------------------------------------------------------ */

/* AiaecWork's views against the exporter's literal shapes (16k/256: 129 bins): four feature signals [1,1,n,129], at
 * N = 8 frame_reset [1,1,1,8] as the fifth and last, one head [1,2,n,129]; at N = 1 nothing past the four. */
static void check_work_views(const AiaecWork *w, int n, const char *what) {
    const int want_in = n > 1 ? 5 : 4;
    int i;
    CHECK(w->action == AIAEC_SUBMIT && w->input_count == want_in && w->output_count == 1,
          "%s: work counts %d/%d", what, w->input_count, w->output_count);
    for (i = 0; i < 4; ++i)
        CHECK(w->inputs[i].data != NULL && w->inputs[i].b == 1u && w->inputs[i].c == 1u && w->inputs[i].h == (unsigned)n &&
              w->inputs[i].w == 129u && w->inputs[i].elements == (size_t)n * 129u, "%s: work.inputs[%d] shape", what, i);
    if (n > 1)
        CHECK(w->inputs[4].data != NULL && w->inputs[4].b == 1u && w->inputs[4].c == 1u && w->inputs[4].h == 1u &&
              w->inputs[4].w == (unsigned)n && w->inputs[4].elements == (size_t)n, "%s: work.inputs[4] is frame_reset [1,1,1,%d]", what, n);
    else
        CHECK(w->inputs[4].data == NULL && w->inputs[4].elements == 0u && w->inputs[4].w == 0u, "%s: nothing past the 4 signals", what);
    CHECK(w->outputs[0].data != NULL && w->outputs[0].b == 1u && w->outputs[0].c == 2u && w->outputs[0].h == (unsigned)n &&
          w->outputs[0].w == 129u && w->outputs[0].elements == (size_t)2u * n * 129u, "%s: work.outputs[0] shape", what);
}

static int section_frame_reset_inputs(void) {
    static Sched s;
    static int16_t pcm[HOP_IN];
    static float ore[NF * BINS], oim[NF * BINS];
    Ctxs c, c1;
    Accel acc[3];
    int edge[MAX_HOPS], p, k, fam, h;

    sched_clear(&s, "reset_inputs", 3);
    s.ev[2].changed = 1; s.ev[8].changed = 1; s.ev[15].changed = 1; s.ev[16].changed = 1;
    for (h = 20; h < 3 * NF; ++h) s.ev[h].solid = 1;   /* a solid false -> true edge at hop 20 */
    expected_edges(&s, edge);
    if (ctxs_make(&c, NF, 1, 1, 1) != 0) return 1;
    CHECK(c.ib.graph_path == c.is.graph_path && c.ib.graph_path == c.if_.graph_path && c.ib.trim_samples == c.is.trim_samples &&
          c.ib.trim_samples == c.if_.trim_samples && c.ib.tail_zero_blocks == c.is.tail_zero_blocks &&
          c.ib.tail_zero_blocks == c.if_.tail_zero_blocks, "N=8: the three families report one info");
    accel_init(&acc[0], aiaec_state_generation(c.blk), ACC_MODEL);
    accel_init(&acc[1], aiaec_state_generation_spec(c.spec), ACC_MODEL);
    accel_init(&acc[2], aiaec_state_generation_freq(c.freq), ACC_MODEL);
    for (p = 0; p < s.packets; ++p) {
        AiaecInput in;
        AiaecFreqInput fin;
        AiaecWork w[3];
        block_input(&in, p, s.ev);
        freq_input(&fin, p * NF, NF, s.ev);
        CHECK(aiaec_preprocess(c.blk, &in, &w[0]) == AIAEC_OK && aiaec_preprocess_spec(c.spec, &in, &w[1]) == AIAEC_OK &&
              aiaec_preprocess_freq(c.freq, &fin, &w[2]) == AIAEC_OK, "packet %d: preprocess", p);
        for (fam = 0; fam < 3; ++fam) {
            const float *reset = w[fam].inputs[4].data;
            check_work_views(&w[fam], NF, "N=8");
            if (p == 0) {   /* the event at hop 2 of packet 0 alone: frame_reset 00100000 */
                static const float want0[NF] = { 0, 0, 1, 0, 0, 0, 0, 0 };
                CHECK(memcmp(reset, want0, sizeof(want0)) == 0, "family %d: packet 0 frame_reset is not 00100000", fam);
            }
            for (k = 0; k < NF; ++k)
                CHECK(reset[k] == (edge[p * NF + k] ? 1.0f : 0.0f), "family %d packet %d: frame_reset[%d] is %g, event %d", fam, p, k,
                      (double)reset[k], edge[p * NF + k]);
            accel_run(&acc[fam], &w[fam], NF);
        }
        CHECK(aiaec_postprocess(c.blk, 0, pcm) == AIAEC_OK && aiaec_postprocess_spec(c.spec, 0, ore, oim) == AIAEC_OK &&
              aiaec_postprocess_freq(c.freq, 0, ore, oim) == AIAEC_OK, "packet %d: postprocess", p);
    }
    ctxs_free(&c);

    /* N = 1: four signals, no frame_reset, nothing at inputs[4] */
    if (ctxs_make(&c1, 1, 1, 1, 1) != 0) return 1;
    {
        AiaecAlignEvent none[1] = { { 0, 0 } };
        AiaecInput in;
        AiaecFreqInput fin;
        AiaecWork w[3];
        block_input(&in, 0, none);
        in.samples = HOP; in.u.align_ulcnet.event_count = 1;
        freq_input(&fin, 0, 1, none);
        CHECK(aiaec_preprocess(c1.blk, &in, &w[0]) == AIAEC_OK && aiaec_preprocess_spec(c1.spec, &in, &w[1]) == AIAEC_OK &&
              aiaec_preprocess_freq(c1.freq, &fin, &w[2]) == AIAEC_OK, "N=1: preprocess");
        for (fam = 0; fam < 3; ++fam) {
            check_work_views(&w[fam], 1, "N=1");
            memset(w[fam].outputs[0].data, 0, w[fam].outputs[0].elements * sizeof(float));
        }
        CHECK(aiaec_postprocess(c1.blk, 0, pcm) == AIAEC_OK && aiaec_postprocess_spec(c1.spec, 0, ore, oim) == AIAEC_OK &&
              aiaec_postprocess_freq(c1.freq, 0, ore, oim) == AIAEC_OK, "N=1: postprocess");
    }
    ctxs_free(&c1);
    printf("section_frame_reset_inputs: work views equal the exporter's shapes in all three families, frame_reset is the last signal at N=8 and carries the %d edges\n",
           edge[2] + edge[8] + edge[15] + edge[16] + edge[20]);
    return 0;
}

/* ---- delay and tail ------------------------------------------------------------------------------------------- */

static int section_tail(void) {
    static float resid[4 * HOP_IN], farz[4 * HOP_IN];
    static int16_t out[4 * HOP_IN];
    static AiaecAlignEvent ev[NF];
    AiaecInfo bi;
    Ctxs c;
    Accel acc;
    int blocks, b, i, peak = 0, tail;
    const int last_real = 3 * HOP_IN - 1;

    if (ctxs_make(&c, NF, 1, 0, 0) != 0) return 1;
    bi = c.ib;
    tail = bi.tail_zero_blocks;
    CHECK(bi.trim_samples == HOP && bi.queue_frames == 0 && tail == 1, "trim %d queue %d tail %d", bi.trim_samples,
          bi.queue_frames, tail);
    resid[last_real] = 0.5f;   /* 16384 counts after the int16 conversion */
    blocks = 3 + tail;
    accel_init(&acc, aiaec_state_generation(c.blk), ACC_IDENTITY);
    for (b = 0; b < blocks; ++b) {
        AiaecInput in;
        AiaecWork w;
        memset(&in, 0, sizeof(in));
        in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.samples = HOP_IN;
        in.u.align_ulcnet.residual_in = resid + (size_t)b * HOP_IN; in.u.align_ulcnet.far_in = farz + (size_t)b * HOP_IN;
        in.u.align_ulcnet.events = ev; in.u.align_ulcnet.event_count = NF;
        CHECK(aiaec_preprocess(c.blk, &in, &w) == AIAEC_OK, "tail preprocess %d", b);
        accel_run(&acc, &w, NF);
        CHECK(aiaec_postprocess(c.blk, 0, out + (size_t)b * HOP_IN) == AIAEC_OK, "tail postprocess %d", b);
    }
    for (i = 0; i < blocks * HOP_IN; ++i) if (abs(out[i]) > abs(out[peak])) peak = i;
    CHECK(peak == last_real + bi.trim_samples && abs(out[peak]) > 16000 && abs(out[peak]) < 16800,
          "pulse at %d came back at %d (%d), want %d (~16384)", last_real, peak, out[peak], last_real + bi.trim_samples);
    {   /* without the tail block the pulse is not in the output */
        int lost = 0;
        for (i = 0; i < 3 * HOP_IN; ++i) if (abs(out[i]) > lost) lost = abs(out[i]);
        CHECK(lost < 1000, "the pulse shows up before the tail block (%d)", lost);
    }
    ctxs_free(&c);
    printf("section_tail: pulse at %d -> %d (delay %d), tail_zero_blocks %d\n", last_real, peak, bi.trim_samples, tail);
    return 0;
}

/* ---- failure -------------------------------------------------------------------------------------------------- */

static int section_failure(void) {
    static Sched s;
    static Run r;
    Ctxs c;
    Accel acc;
    int p, h;
    unsigned gen_before, gen_after;

    sched_clear(&s, "failure", 4);
    s.ev[10].changed = 1;
    /* non-finite mask in every packet: all frames pass through, generation advances once per packet */
    run_freq_n8(&s, &r, ACC_NAN);   /* the driver does not look at ai_rc; the library must catch the NaN */
    for (h = 0; h < s.packets * NF; ++h)
        CHECK(memcmp(r.re + (size_t)h * BINS, g_ere + (size_t)h * BINS, BINS * sizeof(float)) == 0 &&
              memcmp(r.im + (size_t)h * BINS, g_eim + (size_t)h * BINS, BINS * sizeof(float)) == 0,
              "non-finite mask: hop %d is not the error spectrum", h);
    CHECK(r.gen_changes == (unsigned)s.packets - 1u, "non-finite mask: generation changes seen by the accelerator %u", r.gen_changes);

    /* inference failure: infer_status != 0 passes the packet through and leaves the generation alone */
    if (ctxs_make(&c, NF, 0, 0, 1) != 0) return 1;
    accel_init(&acc, aiaec_state_generation_freq(c.freq), ACC_MODEL);
    gen_before = aiaec_state_generation_freq(c.freq);
    for (p = 0; p < 3; ++p) {
        AiaecFreqInput fin;
        AiaecWork w;
        static float re[NF * BINS], im[NF * BINS];
        freq_input(&fin, p * NF, NF, s.ev);
        CHECK(aiaec_preprocess_freq(c.freq, &fin, &w) == AIAEC_OK, "failure preprocess %d", p);
        accel_run(&acc, &w, NF);
        CHECK(aiaec_postprocess_freq(c.freq, p == 1 ? 7 : 0, re, im) == AIAEC_OK, "failure postprocess %d", p);
        if (p == 1)
            CHECK(memcmp(re, g_ere + (size_t)p * NF * BINS, sizeof(re)) == 0 && memcmp(im, g_eim + (size_t)p * NF * BINS, sizeof(im)) == 0,
                  "infer_status != 0: packet %d is not the error spectrum", p);
        else
            CHECK(memcmp(re, g_ere + (size_t)p * NF * BINS, sizeof(re)) != 0, "packet %d should be enhanced", p);
    }
    gen_after = aiaec_state_generation_freq(c.freq);
    CHECK(gen_after == gen_before, "infer_status != 0 changed the generation %u -> %u", gen_before, gen_after);
    ctxs_free(&c);
    printf("section_failure: failed and non-finite packets pass all %d error spectra through\n", NF);
    return 0;
}

/* ---- gates ---------------------------------------------------------------------------------------------------- */

static int section_gates(void) {
    static Sched s;
    static float re[NF * BINS], im[NF * BINS];
    static int16_t pcm[HOP_IN];
    Ctxs c;
    AiaecInput in;
    AiaecWork w;
    AiaecAlignEvent one[1] = { { 0, 0 } };
    AiaecConfig n1 = cfg_n(1), n2 = cfg_n(2), n16 = cfg_n(16), bad = cfg_n(NF);
    StreamMemReq rq;
    unsigned int gen0;

    sched_clear(&s, "gates", 1);
    if (ctxs_make(&c, NF, 1, 0, 0) != 0) return 1;
    gen0 = aiaec_state_generation(c.blk);

    block_input(&in, 0, s.ev);
    in.u.align_ulcnet.event_count = 1; in.u.align_ulcnet.events = one;
    CHECK(aiaec_preprocess(c.blk, &in, &w) == AIAEC_ERR_INPUT_SIZE, "one event for an N=8 block must be refused");
    block_input(&in, 0, s.ev);
    in.u.align_ulcnet.events = NULL;
    CHECK(aiaec_preprocess(c.blk, &in, &w) == AIAEC_ERR_NULL, "NULL events");
    CHECK(aiaec_state_generation(c.blk) == gen0, "state advanced after NULL events");
    block_input(&in, 0, s.ev);
    in.samples = HOP;
    CHECK(aiaec_preprocess(c.blk, &in, &w) == AIAEC_ERR_INPUT_SIZE, "one hop of samples for an N=8 block");
    CHECK(aiaec_postprocess(c.blk, 0, pcm) == AIAEC_ERR_SEQUENCE, "postprocess without preprocess");

    block_input(&in, 0, s.ev);
    CHECK(aiaec_preprocess(c.blk, &in, &w) == AIAEC_OK, "preprocess");
    CHECK(w.input_count == 5 && w.output_count == 1 && w.inputs[0].h == NF && w.inputs[0].w == BINS &&
          w.inputs[0].elements == (size_t)NF * BINS && w.inputs[4].elements == NF && w.inputs[4].w == NF && w.inputs[4].h == 1u &&
          w.outputs[0].c == 2u && w.outputs[0].h == NF && w.outputs[0].w == BINS && w.outputs[0].elements == (size_t)2 * NF * BINS,
          "work layout");
    CHECK(aiaec_preprocess(c.blk, &in, &w) == AIAEC_ERR_SEQUENCE, "second preprocess without a postprocess");
    memset(w.outputs[0].data, 0, w.outputs[0].elements * sizeof(float));
    CHECK(aiaec_postprocess(c.blk, 0, pcm) == AIAEC_OK, "postprocess");
    CHECK(aiaec_postprocess(c.blk, 0, pcm) == AIAEC_ERR_SEQUENCE, "second postprocess");
    (void)re; (void)im;

    CHECK(aiaec_get_mem_size(&n2, &rq) == AIAEC_ERR_FRAMES_PER_CALL, "N=2 is not in the catalog");
    CHECK(aiaec_get_mem_size(&n16, &rq) == AIAEC_ERR_FRAMES_PER_CALL, "N=16 is not in the catalog");
    bad.hop_in = HOP_IN + 1;
    CHECK(aiaec_get_mem_size(&bad, &rq) == AIAEC_ERR_IN_HOP, "hop_in 1025 is refused");
    {   /* a pool carved for N=1 is too small for N=8 */
        StreamMemReq r1, r8;
        void *pool;
        AiaecCtx *ctx = NULL;
        AiaecInfo info;
        int rc;
        CHECK(aiaec_get_mem_size(&n1, &r1) == AIAEC_OK && aiaec_get_mem_size(&bad, &r8) != AIAEC_OK, "n1/n8 sizes");
        bad.hop_in = HOP_IN;
        CHECK(aiaec_get_mem_size(&bad, &r8) == AIAEC_OK && r8.bytes > r1.bytes, "N=8 pool must be larger than N=1's");
        pool = alloc_aligned((size_t)r8.bytes);
        rc = aiaec_init(&bad, pool, (size_t)r1.bytes, &ctx, &info);
        CHECK(rc == AIAEC_ERR_POOL && ctx == NULL, "an N=1 sized pool for N=8, rc=%d", rc);
        free(pool);
    }
    ctxs_free(&c);
    printf("section_gates: done\n");
    return 0;
}

/* ---- the model-I/O pin ------------------------------------------------------------------------------------- */

static int section_model_io_pin(void) {
    UlcnetGridSpec spec;
    UlcnetGrid grid;
    UlcnetModelIoDescriptor d1, d8, bad;
    UlcnetModelIoMemReq m1, m8;
    UlcnetPrepostConfig pc;
    UlcnetPrepostMemReq q1, q8;
    int frames;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000; spec.n_fft = FFT; spec.delay_depth = 63;
    CHECK(ulcnet_grid_make(&grid, &spec) == ULCNET_GRID_OK, "grid");
    CHECK(ULCNET_MODEL_IO_LAYOUT_VERSION == 18u && ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION == 19u, "layout numbers");
    CHECK(ulcnet_model_io_descriptor_default(&grid, &d1) == 0 && ulcnet_model_io_descriptor_validate(&grid, &d1) == 0 &&
          ulcnet_model_io_descriptor_frames(&d1) == 1, "layout 18, one frame");
    CHECK(ulcnet_model_io_descriptor_default_packet(&grid, NF, &d8) == 0 && d8.layout_version == 19u && d8.frames_per_call == NF &&
          ulcnet_model_io_descriptor_validate(&grid, &d8) == 0 && ulcnet_model_io_descriptor_frames(&d8) == NF, "layout 19, N=8");

    bad = d1; bad.frames_per_call = 1;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) == 0, "layout 18 with frames_per_call 1");
    bad = d1; bad.layout_version = 19u;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) != 0, "layout 19 with one frame");
    bad = d1; bad.layout_version = 19u; bad.frames_per_call = 1;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) != 0, "layout 19 with frames_per_call 1");
    bad = d8; bad.layout_version = 18u;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) != 0, "layout 18 with 8 frames");
    bad = d8; bad.layout_version = 17u;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) != 0, "retired layout 17");
    bad = d8; bad.layout_version = 20u;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) != 0, "layout 20 was never assigned");
    bad = d8; bad.frames_per_call = ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL + 1;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) != 0, "more frames than the exporter's maximum");
    bad = d8; bad.frames_per_call = -8;
    CHECK(ulcnet_model_io_descriptor_validate(&grid, &bad) != 0, "negative frames");
    for (frames = -1; frames <= 1; ++frames)
        CHECK(ulcnet_model_io_descriptor_default_packet(&grid, frames, &bad) != 0, "default_packet(%d) must be refused", frames);
    CHECK(ulcnet_model_io_descriptor_default_packet(&grid, ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL + 1, &bad) != 0, "default_packet(65)");
    CHECK(ulcnet_model_io_descriptor_default_packet(&grid, ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL, &bad) == 0, "default_packet(64)");

    CHECK(ulcnet_model_io_get_mem_requirements(&d1, &m1) == 0 && ulcnet_model_io_get_mem_requirements(&d8, &m8) == 0 &&
          m8.bytes > m1.bytes, "the packet pool carries N frames of feature and mask planes");

    /* the prepost class: the layout follows frames_per_call, and a mismatch is refused before a pool is sized */
    CHECK(ulcnet_prepost_config_defaults(&pc, &grid) == 0 && ulcnet_prepost_get_mem_size(&pc, &q1) == 0 && q1.layout_version == 18u,
          "prepost one frame");
    pc.frames_per_call = NF;
    CHECK(ulcnet_prepost_get_mem_size(&pc, &q8) == 0 && q8.layout_version == 19u && q8.build_flags_hash != q1.build_flags_hash &&
          q8.bytes > q1.bytes, "prepost packet");
    pc.model_layout_version = 18u;
    CHECK(ulcnet_prepost_get_mem_size(&pc, &q8) != 0, "prepost N=8 with layout 18");
    pc.model_layout_version = 19u;
    CHECK(ulcnet_prepost_get_mem_size(&pc, &q8) == 0, "prepost N=8 with layout 19");
    pc.frames_per_call = 1;
    CHECK(ulcnet_prepost_get_mem_size(&pc, &q1) != 0, "prepost N=1 with layout 19");
    pc.frames_per_call = -1; pc.model_layout_version = 0u;
    CHECK(ulcnet_prepost_get_mem_size(&pc, &q1) != 0, "prepost negative frames");

    /* one-frame and packet entries refuse each other's state */
    {
        UlcnetPrepostConfig c1, c8;
        UlcnetPrepost *p1, *p8;
        void *m1p, *m8p;
        float z[NF * BINS];
        int ev[NF] = { 0 };
        UlcnetModelIoInputs in;
        UlcnetModelIoOutputs out;
        memset(z, 0, sizeof(z));
        ulcnet_prepost_config_defaults(&c1, &grid); ulcnet_prepost_config_defaults(&c8, &grid); c8.frames_per_call = NF;
        ulcnet_prepost_get_mem_size(&c1, &q1); ulcnet_prepost_get_mem_size(&c8, &q8);
        m1p = alloc_aligned((size_t)q1.bytes); m8p = alloc_aligned((size_t)q8.bytes);
        p1 = ulcnet_prepost_init(m1p, (size_t)q1.bytes, &c1); p8 = ulcnet_prepost_init(m8p, (size_t)q8.bytes, &c8);
        CHECK(p1 && p8 && ulcnet_prepost_frames_per_call(p1) == 1 && ulcnet_prepost_frames_per_call(p8) == NF, "prepost init");
        if (p1 && p8) {
            CHECK(ulcnet_prepost_pre_process_freq(p8, z, z, z, z) == -1, "one-frame entry on a packet instance");
            CHECK(ulcnet_prepost_pre_process_packet(p1, z, z, z, z, ev) == -1, "packet entry on a one-frame instance");
            CHECK(ulcnet_prepost_pre_process_packet(p8, z, z, z, z, NULL) == -1, "packet entry without events");
            CHECK(ulcnet_prepost_pre_process_packet(p8, z, z, z, z, ev) == 1 && ulcnet_prepost_frame_open(p8, &in, &out) == 0 &&
                  in.frame_reset != NULL && in.frame_reset_elements == NF && in.spectrum_bins_elements == (size_t)NF * BINS &&
                  out.spectrum_ri_elements == (size_t)2 * NF * BINS, "packet views");
            CHECK(ulcnet_prepost_pre_process_freq(p1, z, z, z, z) == 1 && ulcnet_prepost_frame_open(p1, &in, &out) == 0 &&
                  in.frame_reset == NULL && in.frame_reset_elements == 0u && out.spectrum_ri_elements == (size_t)2 * BINS,
                  "one-frame views carry no frame_reset");
        }
        free(m1p); free(m8p);
    }
    printf("section_model_io_pin: layout 18 <-> one frame, layout 19 <-> 2..%d frames\n", ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL);
    return 0;
}

int main(void) {
    make_signal(MAX_HOPS);
    section_spectra();
    section_families();
    section_realign();
    section_event_hop_pin();
    section_frame_reset_inputs();
    section_tail();
    section_failure();
    section_gates();
    section_model_io_pin();
    printf("test_aiaec_packet: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
