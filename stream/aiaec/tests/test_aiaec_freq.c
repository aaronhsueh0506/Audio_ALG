/* Host coverage for the spectrum -> spectrum ("_freq") family against the PCM block family through
 * the real public API, for the model the build binds (aiaec_product.h). Compiled once per product with
 * -DAIAEC_TEST_CATALOG; the accelerator is a host stand-in (deterministic heads, an identity head, or a
 * bypass).
 *
 *   equivalence  the PCM block family and _freq fed by an independent AcStft run in lockstep, neither of them
 *                restarted at an alignment event: the PCM output equals the externally synthesized _freq output
 *                sample for sample, an Align-ULCNet event hop included (where the output is the input error
 *                spectrum unchanged). DeepVQE-S at 16k/256 N=1 and 16k/512 N=1 and N=4 with a forced inference
 *                failure; Align-ULCNet with changed / solid events and a forced failure.
 *   events       Align-ULCNet: per hop SUBMIT or BYPASS (changed and solid false -> true hops bypass,
 *                the next hop submits), in both families.
 *   continuity   Align-ULCNet _freq with never-restarted analysers and ISTFT, bypassing every hop
 *                across alignment events: the output is the input delayed by one hop throughout.
 *   scale        the spectra at the boundary are audio scale: an identity head returns the supplied
 *                spectrum, a fail-open frame returns its input error spectrum bit for bit.
 *   tail         a single 16000-count sample at the last real input sample comes back one model hop later
 *                after the test's own all-zero drain block, through both families.
 *   bind/pool    the info is shared by both families; pools: short / misaligned refused, larger constructs,
 *                also when the pool is undersized; the _freq pool is the pinned size and the PCM pool is larger
 *                by the framing (FFT, two STFT, one ISTFT, PCM scratch). Every section runs each family in a pool of
 *                exactly its size with a canary zone behind it.
 *   gates        kind / size / null / sequence rejections for the _freq entries. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_test_fixture.h"
#include "aiaec_product.h"
#include "audio_stft.h"
#include "fft_wrapper.h"
#include "stream_mem.h"

#define PRODUCT_DV (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S)
#define PRODUCT_AL (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_ALIGN_ULCNET)
#define NINPUTS (PRODUCT_DV ? 2 : 4)

static int g_fail = 0;
#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } while (0)
#define CHECK(cond, ...) do { if (!(cond)) FAIL(__VA_ARGS__); } while (0)

static AiaecConfig make_cfg(int fft, int n) {
    AiaecConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sr = fft == 1024 ? 48000 : 16000; cfg.fft = fft; cfg.hop_in = (fft / 2) * n;
    return cfg;
}

/* One context per family on its own pool of exactly its size, followed by PAIR_CANARY bytes of 0xC3. */
enum { PAIR_CANARY = 64 };
typedef struct Pair {
    AiaecCtx *blk;
    AiaecFreqCtx *freq;
    void *p_blk, *p_freq;
    size_t b_blk, b_freq;
    AiaecInfo info_blk, info_freq;
} Pair;

static void *pair_pool(size_t bytes) {
    unsigned char *p = (unsigned char *)alloc_aligned(bytes + PAIR_CANARY);
    memset(p + bytes, 0xC3, PAIR_CANARY);
    return p;
}

static int canary_intact(const void *pool, size_t bytes) {
    const unsigned char *p = (const unsigned char *)pool + bytes;
    size_t k;
    for (k = 0; k < PAIR_CANARY; ++k) if (p[k] != 0xC3) return 0;
    return 1;
}

static int pair_make(Pair *t, const AiaecConfig *cfg) {
    StreamMemReq rb, rf;
    int e;
    memset(t, 0, sizeof(*t));
    if (aiaec_get_mem_size(cfg, &rb) != AIAEC_OK || aiaec_get_mem_size_freq(cfg, &rf) != AIAEC_OK) {
        FAIL("pair: get_mem_size failed for fft=%d hop_in=%d", cfg->fft, cfg->hop_in);
        return -1;
    }
    t->b_blk = (size_t)rb.bytes; t->b_freq = (size_t)rf.bytes;
    t->p_blk = pair_pool(t->b_blk); t->p_freq = pair_pool(t->b_freq);
    e = aiaec_init(cfg, t->p_blk, (size_t)rb.bytes, &t->blk, &t->info_blk);
    CHECK(e == AIAEC_OK && t->blk != NULL, "pair: init err=%d", e);
    e = aiaec_init_freq(cfg, t->p_freq, (size_t)rf.bytes, &t->freq, &t->info_freq);
    CHECK(e == AIAEC_OK && t->freq != NULL, "pair: init_freq err=%d", e);
    return t->blk && t->freq ? 0 : -1;
}

static void pair_free(Pair *t) {
    CHECK(canary_intact(t->p_blk, t->b_blk) && canary_intact(t->p_freq, t->b_freq),
          "pair: a context wrote past its pool (block/freq canary %d/%d)", canary_intact(t->p_blk, t->b_blk),
          canary_intact(t->p_freq, t->b_freq));
    free(t->p_blk); free(t->p_freq);
}

/* An external framing object of the customer's: default config, accumulate OLA, scale 1. */
typedef struct Frame { AcStft *stft; AcIstft *istft; void *mem_s, *mem_i; } Frame;

static void frame_make(Frame *f, int fft, int want_stft, int want_istft) {
    AcStftConfig sc;
    memset(f, 0, sizeof(*f));
    ac_stft_config_default(&sc, fft);
    sc.ola = AC_OLA_ACCUMULATE;
    if (want_stft) {
        f->mem_s = alloc_aligned(ac_stft_get_mem_size(&sc));
        f->stft = ac_stft_init(f->mem_s, ac_stft_get_mem_size(&sc), &sc);
        if (!f->stft) { fprintf(stderr, "stft init fail\n"); exit(1); }
    }
    if (want_istft) {
        f->mem_i = alloc_aligned(ac_istft_get_mem_size(&sc));
        f->istft = ac_istft_init(f->mem_i, ac_istft_get_mem_size(&sc), &sc);
        if (!f->istft) { fprintf(stderr, "istft init fail\n"); exit(1); }
    }
}

static void frame_free(Frame *f) { free(f->mem_s); free(f->mem_i); }

static void synth_frames(Frame *f, int n, int hop, int n_bins, const float *re, const float *im, int16_t *pcm_out,
                         float *scratch_hop) {
    int k, i;
    for (k = 0; k < n; ++k) {
        ac_istft_push_planar(f->istft, re + (size_t)k * (size_t)n_bins, im + (size_t)k * (size_t)n_bins, scratch_hop);
        for (i = 0; i < hop; ++i) pcm_out[(size_t)k * (size_t)hop + (size_t)i] = stream_saturate_s16(scratch_hop[i]);
    }
}

/* Deterministic accelerator heads keyed by a block counter. */
static void fake_head(StreamWork *w, long long ghop) {
    size_t i;
    float *d;
    if (w->action != STREAM_WORK_SUBMIT) return;
    d = w->outputs[0].data;
    for (i = 0; i < w->outputs[0].elements; ++i)
        d[i] = PRODUCT_DV ? 0.05f * (float)(((ghop + (long long)i) % 37) - 18)
                          : 0.1f * (float)(((ghop + (long long)i) % 23) - 11);
}

/* CCM taps forced to (time 0, centre frequency) = 1 + 0j: the head is then the identity on the mic. */
static void identity_head(StreamWork *w, int n) {
    const size_t tapn = 9, bins = w->outputs[0].elements / (tapn * 2u * (size_t)n);
    float *taps = w->outputs[0].data;
    size_t f, b;
    memset(taps, 0, w->outputs[0].elements * sizeof(float));
    for (f = 0; f < (size_t)n; ++f)
        for (b = 0; b < bins; ++b) taps[(f * bins + b) * tapn * 2u + 1u * 2u] = 1.0f;
}

typedef struct Ev { int changed, solid, fail; } Ev;

/* ---------------------------------------------------------------------------------------------
 * equivalence: block == freq + external ISTFT
 * ------------------------------------------------------------------------------------------- */
static int section_equivalence(int fft, int n, const Ev *sched, int blocks) {
    const int hop = fft / 2, nb = fft / 2 + 1, hop_in = hop * n;
    AiaecConfig cfg = make_cfg(fft, n);
    Pair t;
    Frame ist_f, an_a, an_b;             /* external ISTFT and analysers for freq */
    int16_t *mic = calloc((size_t)hop_in, sizeof(int16_t)), *far16 = calloc((size_t)hop_in, sizeof(int16_t));
    float *resid = calloc((size_t)hop_in, sizeof(float)), *farf = calloc((size_t)hop_in, sizeof(float));
    float *a_re = calloc((size_t)n * nb, sizeof(float)), *a_im = calloc((size_t)n * nb, sizeof(float));
    float *b_re = calloc((size_t)n * nb, sizeof(float)), *b_im = calloc((size_t)n * nb, sizeof(float));
    float *f_re = calloc((size_t)n * nb, sizeof(float)), *f_im = calloc((size_t)n * nb, sizeof(float));
    float *hopf = calloc((size_t)hop, sizeof(float)), *hopg = calloc((size_t)hop, sizeof(float));
    int16_t *p_blk = calloc((size_t)hop_in, sizeof(int16_t));
    int16_t *p_freq = calloc((size_t)hop_in, sizeof(int16_t));
    AiaecAlignEvent ev[8];
    int b, k, bad = 0, had_hop = 0, prev_solid = 0, edges = 0, fails = 0;
    long long ghop = 0;

    (void)ev; (void)had_hop; (void)prev_solid;
    if (pair_make(&t, &cfg) != 0) return 1;
    frame_make(&ist_f, fft, 0, 1);
    frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0);

    for (b = 0; b < blocks; ++b) {
        AiaecInput in;
        AiaecFreqInput fin;
        StreamWork wb, wf;
        const Ev *e = &sched[b % (PRODUCT_AL ? blocks : 1)];
        int fail = PRODUCT_DV ? (b == 3) : e->fail;
        int edge = 0;

        for (k = 0; k < hop_in; ++k) {
            mic[k] = (int16_t)((((long long)k * 5 + ghop * 3) % 97 - 48) * 8);
            far16[k] = (int16_t)((((long long)k * 11 + ghop * 7) % 89 - 44) * 8);
            resid[k] = 0.001f * (float)(((long long)k * 5 + ghop * 3) % 97 - 48) * 8.0f;
            farf[k] = 0.001f * (float)(((long long)k * 11 + ghop * 7) % 89 - 44) * 8.0f;
        }
        memset(&in, 0, sizeof(in));
        in.samples = hop_in;
        memset(&fin, 0, sizeof(fin));
        fin.frames = n;
#if PRODUCT_AL
        ev[0].changed = e->changed; ev[0].solid = e->solid;
        in.kind = AIAEC_INPUT_ALIGN_ULCNET;
        in.u.align_ulcnet.residual_in = resid; in.u.align_ulcnet.far_in = farf;
        in.u.align_ulcnet.events = ev; in.u.align_ulcnet.event_count = 1;
        /* no analyser or synthesis is restarted at an event: the model alone is (aiaec.h) */
        edge = e->changed || (had_hop && !prev_solid && e->solid);
        had_hop = 1; prev_solid = e->solid;
        for (k = 0; k < n; ++k) {
            ac_stft_push_planar(an_a.stft, resid + (size_t)k * hop, a_re + (size_t)k * nb, a_im + (size_t)k * nb);
            ac_stft_push_planar(an_b.stft, farf + (size_t)k * hop, b_re + (size_t)k * nb, b_im + (size_t)k * nb);
        }
        fin.kind = AIAEC_INPUT_ALIGN_ULCNET;
        fin.u.align_ulcnet.err_re = a_re; fin.u.align_ulcnet.err_im = a_im;
        fin.u.align_ulcnet.far_re = b_re; fin.u.align_ulcnet.far_im = b_im;
        fin.u.align_ulcnet.events = ev; fin.u.align_ulcnet.event_count = 1;
#else
        in.kind = AIAEC_INPUT_DEEPVQE_S;
        in.u.deepvqe_s.mic_in = mic; in.u.deepvqe_s.far_in = far16;
        for (k = 0; k < n; ++k) {
            int i;
            for (i = 0; i < hop; ++i) {
                hopf[i] = (float)mic[(size_t)k * hop + (size_t)i] / 32768.0f;
                hopg[i] = (float)far16[(size_t)k * hop + (size_t)i] / 32768.0f;
            }
            ac_stft_push_planar(an_a.stft, hopf, a_re + (size_t)k * nb, a_im + (size_t)k * nb);
            ac_stft_push_planar(an_b.stft, hopg, b_re + (size_t)k * nb, b_im + (size_t)k * nb);
        }
        fin.kind = AIAEC_INPUT_DEEPVQE_S;
        fin.u.deepvqe_s.mic_re = a_re; fin.u.deepvqe_s.mic_im = a_im;
        fin.u.deepvqe_s.far_re = b_re; fin.u.deepvqe_s.far_im = b_im;
#endif
        CHECK(aiaec_preprocess(t.blk, &in, &wb) == AIAEC_OK, "fft=%d N=%d block %d: block preprocess", fft, n, b);
        CHECK(aiaec_preprocess_freq(t.freq, &fin, &wf) == AIAEC_OK, "fft=%d N=%d block %d: freq preprocess", fft, n, b);

        CHECK(wb.action == wf.action, "fft=%d N=%d block %d: action differs across families (%d/%d, edge %d)",
              fft, n, b, (int)wb.action, (int)wf.action, edge);
        edges += edge;
        if (wb.action == STREAM_WORK_SUBMIT) {
            for (k = 0; k < NINPUTS; ++k)
                CHECK(wb.inputs[k].elements == wf.inputs[k].elements &&
                      memcmp(wb.inputs[k].data, wf.inputs[k].data, wb.inputs[k].elements * sizeof(float)) == 0,
                      "fft=%d N=%d block %d: accelerator input %d differs across families", fft, n, b, k);
            CHECK(wb.state_generation == wf.state_generation, "block %d: state_generation differs across families", b);
            if (!fail) { fake_head(&wb, ghop); fake_head(&wf, ghop); }
        }
        fails += fail && wb.action == STREAM_WORK_SUBMIT;
        CHECK(aiaec_postprocess(t.blk, fail, p_blk) == AIAEC_OK, "block %d: block postprocess", b);
        CHECK(aiaec_postprocess_freq(t.freq, fail, f_re, f_im) == AIAEC_OK, "block %d: freq postprocess", b);

        synth_frames(&ist_f, n, hop, nb, f_re, f_im, p_freq, hopf);
        if (edge && (memcmp(a_re, f_re, (size_t)nb * sizeof(float)) != 0 || memcmp(a_im, f_im, (size_t)nb * sizeof(float)) != 0)) {
            /* the event hop is an identity hop: _freq returns the front end's error spectrum unchanged */
            FAIL("fft=%d N=%d block %d: _freq event hop is not its input error spectrum", fft, n, b); bad = 1;
        }
        if (memcmp(p_blk, p_freq, (size_t)hop_in * sizeof(int16_t)) != 0) {
            FAIL("fft=%d N=%d block %d: block PCM != _freq + external ISTFT", fft, n, b); bad = 1;
        }
        ghop += n;
    }
    printf("section_equivalence(fft=%d N=%d): %d blocks (%d forced failures, %d events) block == freq+ISTFT%s\n",
           fft, n, blocks, fails, edges, bad ? " -- DIVERGED" : "");
    frame_free(&ist_f); frame_free(&an_a); frame_free(&an_b);
    pair_free(&t);
    free(mic); free(far16); free(resid); free(farf); free(a_re); free(a_im); free(b_re); free(b_im);
    free(f_re); free(f_im); free(hopf); free(hopg); free(p_blk); free(p_freq);
    return bad;
}

#if PRODUCT_AL
/* ---------------------------------------------------------------------------------------------
 * events: SUBMIT / BYPASS per hop, both families
 * ------------------------------------------------------------------------------------------- */
static int section_events(void) {
    /* changed hop, solid false -> true hop, and the hop after each. */
    static const struct { int changed, solid, submit; } h[] = {
        { 0, 0, 1 },
        { 1, 0, 0 },   /* changed */
        { 0, 0, 1 },   /* next hop: runs */
        { 0, 1, 0 },   /* solid false -> true */
        { 0, 1, 1 },   /* next hop: runs */
        { 0, 1, 1 },
    };
    const int hops = (int)(sizeof(h) / sizeof(h[0]));
    AiaecConfig cfg = make_cfg(256, 1);
    Pair t;
    float resid[128], farf[128], re[129], im[129], a_re[129], a_im[129], b_re[129], b_im[129];
    int16_t pcm[128];
    Frame an_a, an_b;
    int i, k, sub_blk = 0, sub_freq = 0, bad = 0;

    if (pair_make(&t, &cfg) != 0) return 1;
    frame_make(&an_a, 256, 1, 0); frame_make(&an_b, 256, 1, 0);
    for (i = 0; i < hops; ++i) {
        AiaecInput in;
        AiaecFreqInput fin;
        StreamWork wb, wf;
        AiaecAlignEvent ev;
        for (k = 0; k < 128; ++k) { resid[k] = 0.002f * (float)((k * 5 + i * 3) % 97 - 48); farf[k] = 0.002f * (float)((k * 11 + i) % 89 - 44); }
        ev.changed = h[i].changed; ev.solid = h[i].solid;
        memset(&in, 0, sizeof(in));
        in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.samples = 128;
        in.u.align_ulcnet.residual_in = resid; in.u.align_ulcnet.far_in = farf;
        in.u.align_ulcnet.events = &ev; in.u.align_ulcnet.event_count = 1;
        ac_stft_push_planar(an_a.stft, resid, a_re, a_im);
        ac_stft_push_planar(an_b.stft, farf, b_re, b_im);
        memset(&fin, 0, sizeof(fin));
        fin.kind = AIAEC_INPUT_ALIGN_ULCNET; fin.frames = 1;
        fin.u.align_ulcnet.err_re = a_re; fin.u.align_ulcnet.err_im = a_im;
        fin.u.align_ulcnet.far_re = b_re; fin.u.align_ulcnet.far_im = b_im;
        fin.u.align_ulcnet.events = &ev; fin.u.align_ulcnet.event_count = 1;
        CHECK(aiaec_preprocess(t.blk, &in, &wb) == AIAEC_OK && aiaec_preprocess_freq(t.freq, &fin, &wf) == AIAEC_OK,
              "events hop %d: preprocess", i);
        sub_blk += wb.action == STREAM_WORK_SUBMIT; sub_freq += wf.action == STREAM_WORK_SUBMIT;
        if ((wb.action == STREAM_WORK_SUBMIT) != (h[i].submit != 0) || (wf.action == STREAM_WORK_SUBMIT) != (h[i].submit != 0)) {
            FAIL("events hop %d: submit block/freq = %d/%d, want %d", i, wb.action == STREAM_WORK_SUBMIT,
                 wf.action == STREAM_WORK_SUBMIT, h[i].submit);
            bad = 1;
        }
        if (wb.action == STREAM_WORK_BYPASS)
            CHECK(wb.input_count == 0 && wb.output_count == 0 && wb.state_generation == 0u && wf.input_count == 0,
                  "events hop %d: bypass work not zeroed", i);
        /* the run status is not read on a bypass hop */
        CHECK(aiaec_postprocess(t.blk, wb.action == STREAM_WORK_SUBMIT ? 1 : 0xBAD1, pcm) == AIAEC_OK &&
              aiaec_postprocess_freq(t.freq, wf.action == STREAM_WORK_SUBMIT ? 1 : 0xBAD1, re, im) == AIAEC_OK,
              "events hop %d: postprocess", i);
    }
    CHECK(sub_blk == 4 && sub_freq == 4, "events: submissions block/freq = %d/%d, want 4 of %d hops", sub_blk, sub_freq, hops);
    printf("section_events: %d hops, submissions %d/%d (block/freq)\n", hops, sub_blk, sub_freq);
    frame_free(&an_a); frame_free(&an_b); pair_free(&t);
    return bad;
}

/* _freq with analysers and an ISTFT that are never restarted, every hop bypassed (fail open), across
 * changed / solid events: the output is the input delayed by one hop, event hops included. */
static int section_freq_continuity(int fft) {
    const int hop = fft / 2, blocks = 14;
    static const Ev sched[14] = {
        { 0, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 },
        { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, 0 },
    };
    AiaecConfig cfg = make_cfg(fft, 1);
    Pair t;
    Frame an_a, an_b, ist;
    float *resid = calloc((size_t)hop * blocks, sizeof(float)), *farf = calloc((size_t)hop, sizeof(float));
    float a_re[257], a_im[257], b_re[257], b_im[257], o_re[257], o_im[257], out[256];
    int b, k, bad = 0, events = 0;

    if (pair_make(&t, &cfg) != 0) return 1;
    frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0); frame_make(&ist, fft, 0, 1);
    for (k = 0; k < hop * blocks; ++k) resid[k] = 0.01f * (float)(((long long)k * 7 + 3) % 61 - 30);
    for (k = 0; k < hop; ++k) farf[k] = 0.003f * (float)((k * 11) % 53 - 26);
    for (b = 0; b < blocks; ++b) {
        AiaecFreqInput fin;
        StreamWork w;
        AiaecAlignEvent ev;
        ev.changed = sched[b].changed; ev.solid = sched[b].solid;
        events += sched[b].changed;
        ac_stft_push_planar(an_a.stft, resid + (size_t)b * hop, a_re, a_im);
        ac_stft_push_planar(an_b.stft, farf, b_re, b_im);
        memset(&fin, 0, sizeof(fin));
        fin.kind = AIAEC_INPUT_ALIGN_ULCNET; fin.frames = 1;
        fin.u.align_ulcnet.err_re = a_re; fin.u.align_ulcnet.err_im = a_im;
        fin.u.align_ulcnet.far_re = b_re; fin.u.align_ulcnet.far_im = b_im;
        fin.u.align_ulcnet.events = &ev; fin.u.align_ulcnet.event_count = 1;
        CHECK(aiaec_preprocess_freq(t.freq, &fin, &w) == AIAEC_OK, "continuity block %d: preprocess", b);
        CHECK(aiaec_postprocess_freq(t.freq, w.action == STREAM_WORK_SUBMIT ? 1 : 0, o_re, o_im) == AIAEC_OK, "continuity block %d: postprocess", b);
        ac_istft_push_planar(ist.istft, o_re, o_im, out);
        if (b == 0) continue;
        for (k = 0; k < hop; ++k) {
            int d = (int)stream_saturate_s16(out[k]) - (int)stream_saturate_s16(resid[(size_t)(b - 1) * hop + (size_t)k]);
            if (d < -2 || d > 2) { FAIL("continuity block %d sample %d: off by %d from the input delayed by one hop", b, k, d); bad = 1; break; }
        }
    }
    printf("section_freq_continuity(fft=%d): %d blocks across %d changed events, output == input delayed by one hop\n", fft, blocks, events);
    frame_free(&an_a); frame_free(&an_b); frame_free(&ist); pair_free(&t); free(resid); free(farf);
    return bad;
}
#endif

/* ---------------------------------------------------------------------------------------------
 * scale: audio scale at the boundary
 * ------------------------------------------------------------------------------------------- */
static int section_scale(void) {
    const int fft = 256, hop = 128, nb = 129;
    AiaecConfig cfg = make_cfg(fft, 1);
    Pair t;
    Frame an_a, an_b;
    float hopf[128], hopg[128], a_re[129], a_im[129], b_re[129], b_im[129], o_re[129], o_im[129];
    int b, k, bad = 0;
    double max_err = 0.0, max_abs = 0.0;

    if (pair_make(&t, &cfg) != 0) return 1;
    frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0);
    for (b = 0; b < 6; ++b) {
        AiaecFreqInput fin;
        StreamWork w;
        AiaecAlignEvent ev;
        for (k = 0; k < hop; ++k) {
            hopf[k] = (float)(((k * 7 + b * 13) % 201) - 100) / 400.0f;
            hopg[k] = (float)(((k * 3 + b * 5) % 151) - 75) / 400.0f;
        }
        ac_stft_push_planar(an_a.stft, hopf, a_re, a_im);
        ac_stft_push_planar(an_b.stft, hopg, b_re, b_im);
        memset(&fin, 0, sizeof(fin));
        fin.frames = 1;
        ev.changed = 0; ev.solid = 0;
#if PRODUCT_DV
        fin.kind = AIAEC_INPUT_DEEPVQE_S;
        fin.u.deepvqe_s.mic_re = a_re; fin.u.deepvqe_s.mic_im = a_im;
        fin.u.deepvqe_s.far_re = b_re; fin.u.deepvqe_s.far_im = b_im;
#else
        fin.kind = AIAEC_INPUT_ALIGN_ULCNET;
        fin.u.align_ulcnet.err_re = a_re; fin.u.align_ulcnet.err_im = a_im;
        fin.u.align_ulcnet.far_re = b_re; fin.u.align_ulcnet.far_im = b_im;
        fin.u.align_ulcnet.events = &ev; fin.u.align_ulcnet.event_count = 1;
#endif
        CHECK(aiaec_preprocess_freq(t.freq, &fin, &w) == AIAEC_OK && w.action == STREAM_WORK_SUBMIT, "scale block %d: preprocess", b);
#if PRODUCT_DV
        identity_head(&w, 1);
        CHECK(aiaec_postprocess_freq(t.freq, 0, o_re, o_im) == AIAEC_OK, "scale block %d: postprocess", b);
        for (k = 0; k < nb; ++k) {
            double e = fabs((double)o_re[k] - (double)a_re[k]) + fabs((double)o_im[k] - (double)a_im[k]);
            if (e > max_err) max_err = e;
            if (fabs((double)a_re[k]) > max_abs) max_abs = fabs((double)a_re[k]);
        }
#else
        CHECK(aiaec_postprocess_freq(t.freq, 1, o_re, o_im) == AIAEC_OK, "scale block %d: postprocess", b);
        CHECK(memcmp(o_re, a_re, sizeof(o_re)) == 0 && memcmp(o_im, a_im, sizeof(o_im)) == 0,
              "scale block %d: a fail-open frame is not its input error spectrum bit for bit", b);
        for (k = 0; k < nb; ++k) if (fabs((double)a_re[k]) > max_abs) max_abs = fabs((double)a_re[k]);
#endif
    }
    CHECK(max_abs > 0.5, "scale: the probe spectrum is too small to prove a scale (max |re| %.3g)", max_abs);
    CHECK(max_err <= 1e-4 * max_abs, "scale: identity head returns a spectrum off by %.3g (max |re| %.3g)", max_err, max_abs);
    if (max_err > 1e-4 * max_abs) bad = 1;
    printf("section_scale: boundary spectra are audio scale (max |re| %.3g, identity/fail-open error %.3g)\n", max_abs, max_err);
    frame_free(&an_a); frame_free(&an_b); pair_free(&t);
    return bad;
}

/* ---------------------------------------------------------------------------------------------
 * tail
 * ------------------------------------------------------------------------------------------- */
#define PULSE 16000
#define BLOCKS_REAL 4

static int check_pulse(const float *out, size_t n, size_t idx, const char *label, double *value_out) {
    size_t i, peak = 0;
    double v;
    if (idx >= n) { fprintf(stderr, "%s: pulse index %zu beyond the %zu produced samples\n", label, idx, n); return 1; }
    for (i = 0; i < n; ++i) if (fabsf(out[i]) > fabsf(out[peak])) peak = i;
    v = (double)out[idx] * 32768.0;
    if (value_out) *value_out = v;
    if (fabs(v - (double)PULSE) > 3.0 || peak != idx) {
        fprintf(stderr, "%s: tail lost -- out[%zu] = %.1f (want %d +-3), loudest sample at %zu\n", label, idx, v, PULSE, peak);
        return 1;
    }
    return 0;
}

/* The accelerator stand-in of the tail test: identity head for DeepVQE-S, a bypass for Align-ULCNet
 * (fail open: the error spectrum passes through). Returns the run status to close the block with. */
static int tail_accel(StreamWork *w, int n) {
    if (PRODUCT_DV) { identity_head(w, n); return 0; }
    return 1;
}

static void tail_input(AiaecInput *in, AiaecFreqInput *fin, const int16_t *pcm, float *resid, int hop_in,
                       AiaecAlignEvent *ev, const float *a_re, const float *a_im, const float *b_re, const float *b_im, int n) {
    int k;
    memset(in, 0, sizeof(*in)); memset(fin, 0, sizeof(*fin));
    in->samples = hop_in; fin->frames = n;
    ev->changed = 0; ev->solid = 0;
#if PRODUCT_DV
    (void)resid; (void)ev; (void)k;
    in->kind = AIAEC_INPUT_DEEPVQE_S;
    in->u.deepvqe_s.mic_in = pcm; in->u.deepvqe_s.far_in = pcm;   /* far == mic is fine: the identity head ignores it */
    fin->kind = AIAEC_INPUT_DEEPVQE_S;
    fin->u.deepvqe_s.mic_re = a_re; fin->u.deepvqe_s.mic_im = a_im;
    fin->u.deepvqe_s.far_re = b_re; fin->u.deepvqe_s.far_im = b_im;
#else
    for (k = 0; k < hop_in; ++k) resid[k] = (float)pcm[k] / 32768.0f;
    in->kind = AIAEC_INPUT_ALIGN_ULCNET;
    in->u.align_ulcnet.residual_in = resid; in->u.align_ulcnet.far_in = resid;
    for (k = 0; k < n && k < 8; ++k) ev[k] = ev[0];
    in->u.align_ulcnet.events = ev; in->u.align_ulcnet.event_count = n;
    fin->kind = AIAEC_INPUT_ALIGN_ULCNET;
    fin->u.align_ulcnet.err_re = a_re; fin->u.align_ulcnet.err_im = a_im;
    fin->u.align_ulcnet.far_re = b_re; fin->u.align_ulcnet.far_im = b_im;
    fin->u.align_ulcnet.events = ev; fin->u.align_ulcnet.event_count = n;
#endif
}

/* The info of one family (0 PCM, 1 _freq) of cfg, from a throw-away init. */
static int family_info(const AiaecConfig *cfg, int family, AiaecInfo *info) {
    StreamMemReq rq;
    void *pool;
    int rc = family == 0 ? aiaec_get_mem_size(cfg, &rq) : aiaec_get_mem_size_freq(cfg, &rq);
    if (rc != AIAEC_OK) return rc;
    pool = alloc_aligned((size_t)rq.bytes);
    if (family == 0) { AiaecCtx *c = NULL; rc = aiaec_init(cfg, pool, (size_t)rq.bytes, &c, info); }
    else { AiaecFreqCtx *c = NULL; rc = aiaec_init_freq(cfg, pool, (size_t)rq.bytes, &c, info); }
    free(pool);
    return rc;
}

/* The round trip on the final PCM, from the grid alone: one hop, both models. */
static int expected_delay(int hop) { return hop; }

static int section_tail(int fft, int n) {
    const int hop = fft / 2, nb = fft / 2 + 1, hop_in = hop * n;
    const int want_delay = expected_delay(hop), tail = (want_delay + hop_in - 1) / hop_in;
    AiaecConfig cfg = make_cfg(fft, n);
    Pair t;
    Frame ist_f, an_a, an_b;
    int total_blocks, b, f, k, bad = 0;
    size_t s_real = (size_t)BLOCKS_REAL * (size_t)hop_in, total, idx;
    int16_t *pcm, *out16;
    float *outq, *pf, *resid, *fre, *fim, *a_re, *a_im, *b_re, *b_im, *hopf;
    AiaecAlignEvent ev[8];
    double v;
    char label[64];

    snprintf(label, sizeof(label), "tail(fft=%d N=%d)", fft, n);
    total_blocks = BLOCKS_REAL + tail;
    total = (size_t)total_blocks * (size_t)hop_in;
    idx = s_real - 1 + (size_t)want_delay;

    pcm = calloc(total, sizeof(int16_t)); out16 = calloc(total, sizeof(int16_t));
    outq = calloc(total, sizeof(float)); pf = calloc(total, sizeof(float));
    resid = calloc((size_t)hop_in, sizeof(float));
    fre = calloc((size_t)n * nb, sizeof(float)); fim = calloc((size_t)n * nb, sizeof(float));
    a_re = calloc((size_t)n * nb, sizeof(float)); a_im = calloc((size_t)n * nb, sizeof(float));
    b_re = calloc((size_t)n * nb, sizeof(float)); b_im = calloc((size_t)n * nb, sizeof(float));
    hopf = calloc((size_t)hop, sizeof(float));
    pcm[s_real - 1] = PULSE;

    if (pair_make(&t, &cfg) != 0) return 1;
    frame_make(&ist_f, fft, 0, 1); frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0);

    for (b = 0; b < total_blocks; ++b) {
        AiaecInput in;
        AiaecFreqInput fin;
        StreamWork w;
        int st;
        const int16_t *blk = pcm + (size_t)b * (size_t)hop_in;

        for (f = 0; f < n; ++f) {   /* the upstream analyser sees the same blocks, zero blocks included */
            for (k = 0; k < hop; ++k) hopf[k] = (float)blk[(size_t)f * hop + (size_t)k] / 32768.0f;
            ac_stft_push_planar(an_a.stft, hopf, a_re + (size_t)f * nb, a_im + (size_t)f * nb);
            ac_stft_push_planar(an_b.stft, hopf, b_re + (size_t)f * nb, b_im + (size_t)f * nb);
        }
        tail_input(&in, &fin, blk, resid, hop_in, ev, a_re, a_im, b_re, b_im, n);

        if (aiaec_preprocess(t.blk, &in, &w) != AIAEC_OK) { FAIL("%s: block pre %d", label, b); return 1; }
        st = w.action == STREAM_WORK_SUBMIT ? tail_accel(&w, n) : 0;
        if (aiaec_postprocess(t.blk, st, out16 + (size_t)b * hop_in) != AIAEC_OK) { FAIL("%s: block post %d", label, b); return 1; }

        if (aiaec_preprocess_freq(t.freq, &fin, &w) != AIAEC_OK) { FAIL("%s: freq pre %d", label, b); return 1; }
        st = w.action == STREAM_WORK_SUBMIT ? tail_accel(&w, n) : 0;
        if (aiaec_postprocess_freq(t.freq, st, fre, fim) != AIAEC_OK) { FAIL("%s: freq post %d", label, b); return 1; }
        for (f = 0; f < n; ++f)
            ac_istft_push_planar(ist_f.istft, fre + (size_t)f * nb, fim + (size_t)f * nb, outq + (size_t)b * hop_in + (size_t)f * hop);
    }

    for (k = 0; (size_t)k < total; ++k) pf[k] = (float)out16[k] / 32768.0f;
    bad |= check_pulse(pf, total, idx, label, &v);
    if (!bad) printf("%s: PCM family OK, out[%zu] = %.1f (drain blocks %d)\n", label, idx, v, tail);
    bad |= check_pulse(outq, total, idx, label, &v);
    if (!bad) printf("%s: _freq (upstream analyser + external ISTFT) OK, out[%zu] = %.1f\n", label, idx, v);
    frame_free(&ist_f); frame_free(&an_a); frame_free(&an_b); pair_free(&t);

    free(pcm); free(out16); free(outq); free(pf); free(resid); free(fre); free(fim); free(a_re); free(a_im);
    free(b_re); free(b_im); free(hopf);
    return bad;
}

/* ---------------------------------------------------------------------------------------------
 * info, delays, pools, gates
 * ------------------------------------------------------------------------------------------- */
static int section_bind_and_pool(int fft, int n) {
    const int hop = fft / 2;
    AiaecConfig cfg = make_cfg(fft, n);
    AiaecInfo pb, pf;
    StreamMemReq rb, rf;
    Pair t;
    unsigned char *pool;
    int rc, bad = 0;
    unsigned int gen_f;

    if (family_info(&cfg, 0, &pb) != AIAEC_OK || family_info(&cfg, 1, &pf) != AIAEC_OK) { FAIL("bind: info"); return 1; }
    CHECK(memcmp(&pb, &pf, sizeof(pb)) == 0, "bind(fft=%d N=%d): the info differs across families", fft, n);
    CHECK(pb.graph_path && pf.graph_path && strcmp(pb.graph_path, pf.graph_path) == 0, "bind: graph path differs across families");
    {
        AiaecConfig bad_cfg = make_cfg(fft, n);
        AiaecFreqCtx *cf = NULL;
        AiaecInfo scratch;
        unsigned char small[64];
        bad_cfg.hop_in += 1;
        CHECK(aiaec_init_freq(&bad_cfg, small, sizeof(small), &cf, &scratch) == AIAEC_ERR_IN_HOP && cf == NULL &&
              aiaec_get_mem_size_freq(&bad_cfg, &rf) == AIAEC_ERR_IN_HOP,
              "bind: a bad hop_in must propagate AIAEC_ERR_IN_HOP through every family");
    }

    if (pair_make(&t, &cfg) != 0) return 1;
    CHECK(memcmp(&t.info_blk, &pb, sizeof(pb)) == 0 && memcmp(&t.info_freq, &pb, sizeof(pb)) == 0,
          "bind: the init info equals the throw-away init's");
    gen_f = aiaec_state_generation_freq(t.freq);
    CHECK(aiaec_state_generation_freq(NULL) == 0u, "bind: NULL ctx");
    aiaec_reset_freq(t.freq);
    CHECK(aiaec_state_generation_freq(t.freq) != gen_f, "bind: reset must advance the generation");
    pair_free(&t);

    /* pools: short and misaligned are refused per family, the exact size and a larger pool construct */
    aiaec_get_mem_size(&cfg, &rb); aiaec_get_mem_size_freq(&cfg, &rf);
    pool = alloc_aligned((size_t)(rb.bytes > rf.bytes ? rb.bytes : rf.bytes) + 64);
    {
        AiaecCtx *cb = NULL;
        AiaecFreqCtx *cf = NULL;
        AiaecInfo info;
        rc = aiaec_init(&cfg, pool, (size_t)rb.bytes - 1u, &cb, &info);
        CHECK(rc == AIAEC_ERR_POOL && cb == NULL, "pool: undersized block rc=%d", rc);
        rc = aiaec_init_freq(&cfg, pool, (size_t)rf.bytes - 1u, &cf, &info);
        CHECK(rc == AIAEC_ERR_POOL && cf == NULL, "pool: undersized freq rc=%d", rc);
        rc = aiaec_init_freq(&cfg, pool + 1, (size_t)rf.bytes, &cf, &info);
        CHECK(rc == AIAEC_ERR_POOL && cf == NULL, "pool: misaligned freq rc=%d", rc);
        CHECK(aiaec_init(&cfg, pool, (size_t)rb.bytes, &cb, &info) == AIAEC_OK && cb != NULL, "pool: the exact block size must construct");
        CHECK(aiaec_init_freq(&cfg, pool, (size_t)rf.bytes + 64u, &cf, &info) == AIAEC_OK && cf != NULL, "pool: a larger freq pool must construct");
    }
    {   /* the _freq pool carries no framing: pinned (host, both models compiled); the PCM pool adds it */
        AcStftConfig sc;
        const size_t plane = (size_t)(PRODUCT_AL ? n : 1) * (size_t)(fft / 2 + 1);
        size_t framing, pinned;
        ac_stft_config_default(&sc, fft);
        sc.ola = AC_OLA_ACCUMULATE;
#if PRODUCT_DV
        framing = ck_align16_size(fft_get_mem_size(fft)) + 2u * ck_align16_size(ac_stft_get_mem_size(&sc)) +
                  ck_align16_size(ac_istft_get_mem_size(&sc)) +
                  ck_align16_size((3u * (size_t)hop + 2u * plane * (1u + (size_t)n)) * sizeof(float));
        pinned = fft == 1024 ? (n == 2 ? 111440u : 62224u) : fft == 512 ? (n == 4 ? 105488u : 31504u) : 16144u;
#else
        {   /* the stream's scratch slab: the error planes and events are in both pools, far / enhanced / hop are PCM only */
            const size_t bins = (size_t)(fft / 2 + 1);
            const size_t shared = 2u * plane + (n > 1 ? (size_t)n : 0u);
            framing = ck_align16_size(fft_get_mem_size(fft)) + 2u * ck_align16_size(ac_stft_get_mem_size(&sc)) +
                      ck_align16_size(ac_istft_get_mem_size(&sc)) +
                      ck_align16_size((shared + 2u * bins + 2u * plane + (size_t)hop) * sizeof(float)) -
                      ck_align16_size(shared * sizeof(float));
        }
        pinned = fft == 1024 ? (n == 2 ? 41552u : 21040u) : fft == 512 ? (n == 4 ? 41600u : 10800u) : 5680u;
#endif
        CHECK(rb.bytes - rf.bytes == framing, "pool(fft=%d N=%d): block %llu - _freq %llu is not the framing %zu", fft, n,
                  (unsigned long long)rb.bytes, (unsigned long long)rf.bytes, framing);
        CHECK((size_t)rf.bytes == pinned, "pool(fft=%d N=%d): _freq pool %llu bytes, pinned %zu", fft, n,
              (unsigned long long)rf.bytes, pinned);
    }
    printf("section_bind_and_pool(fft=%d N=%d): info shared by both families, pool sizes enforced (block/freq %llu/%llu)\n",
           fft, n, (unsigned long long)rb.bytes, (unsigned long long)rf.bytes);
    free(pool);
    return bad;
}

static int section_gates(void) {
    AiaecConfig cfg = make_cfg(256, 1);
    Pair t;
    float re[129], im[129], zeros[129];
    AiaecFreqInput fin;
    StreamWork w;
    AiaecAlignEvent ev[2];
    unsigned int gen;

    if (pair_make(&t, &cfg) != 0) return 1;
    memset(zeros, 0, sizeof(zeros)); memset(ev, 0, sizeof(ev));
    gen = aiaec_state_generation_freq(t.freq);

    memset(&fin, 0, sizeof(fin));
    fin.frames = 1;
#if PRODUCT_DV
    fin.kind = AIAEC_INPUT_DEEPVQE_S;
    fin.u.deepvqe_s.mic_re = zeros; fin.u.deepvqe_s.mic_im = zeros; fin.u.deepvqe_s.far_re = zeros; fin.u.deepvqe_s.far_im = zeros;
#else
    fin.kind = AIAEC_INPUT_ALIGN_ULCNET;
    fin.u.align_ulcnet.err_re = zeros; fin.u.align_ulcnet.err_im = zeros; fin.u.align_ulcnet.far_re = zeros; fin.u.align_ulcnet.far_im = zeros;
    fin.u.align_ulcnet.events = ev; fin.u.align_ulcnet.event_count = 1;
#endif
    {
        AiaecFreqInput g = fin;
        g.kind = PRODUCT_DV ? AIAEC_INPUT_ALIGN_ULCNET : AIAEC_INPUT_DEEPVQE_S;
        CHECK(aiaec_preprocess_freq(t.freq, &g, &w) == AIAEC_ERR_INPUT_KIND, "gates: wrong kind");
        g = fin; g.frames = 2;
        CHECK(aiaec_preprocess_freq(t.freq, &g, &w) == AIAEC_ERR_INPUT_SIZE, "gates: frames != N");
        g = fin; g.frames = 0;
        CHECK(aiaec_preprocess_freq(t.freq, &g, &w) == AIAEC_ERR_INPUT_SIZE, "gates: frames 0");
#if PRODUCT_AL
        g = fin; g.u.align_ulcnet.event_count = 2;
        CHECK(aiaec_preprocess_freq(t.freq, &g, &w) == AIAEC_ERR_INPUT_SIZE, "gates: event_count != N");
        g = fin; g.u.align_ulcnet.events = NULL;
        CHECK(aiaec_preprocess_freq(t.freq, &g, &w) == AIAEC_ERR_NULL, "gates: NULL events");
        g = fin; g.u.align_ulcnet.far_im = NULL;
        CHECK(aiaec_preprocess_freq(t.freq, &g, &w) == AIAEC_ERR_NULL, "gates: NULL far_im");
#else
        g = fin; g.u.deepvqe_s.mic_im = NULL;
        CHECK(aiaec_preprocess_freq(t.freq, &g, &w) == AIAEC_ERR_NULL, "gates: NULL mic_im");
#endif
    }
    CHECK(aiaec_state_generation_freq(t.freq) == gen, "gates: a rejected freq input advanced the state");
    CHECK(aiaec_preprocess_freq(NULL, &fin, &w) == AIAEC_ERR_NULL && aiaec_preprocess_freq(t.freq, NULL, &w) == AIAEC_ERR_NULL &&
          aiaec_preprocess_freq(t.freq, &fin, NULL) == AIAEC_ERR_NULL, "gates: NULL args freq");
    CHECK(aiaec_postprocess_freq(t.freq, 0, re, im) == AIAEC_ERR_SEQUENCE, "gates: postprocess without preprocess");
    CHECK(aiaec_postprocess_freq(NULL, 0, re, im) == AIAEC_ERR_NULL && aiaec_postprocess_freq(t.freq, 0, NULL, im) == AIAEC_ERR_NULL &&
          aiaec_postprocess_freq(t.freq, 0, re, NULL) == AIAEC_ERR_NULL, "gates: NULL outputs");

    CHECK(aiaec_preprocess_freq(t.freq, &fin, &w) == AIAEC_OK, "gates: valid freq preprocess");
    CHECK(aiaec_preprocess_freq(t.freq, &fin, &w) == AIAEC_ERR_SEQUENCE, "gates: second freq preprocess before postprocess");
    if (w.action == STREAM_WORK_SUBMIT) fake_head(&w, 0);
    CHECK(aiaec_postprocess_freq(t.freq, 0, NULL, NULL) == AIAEC_ERR_NULL, "gates: NULL outputs keep the block open");
    CHECK(aiaec_postprocess_freq(t.freq, 0, re, im) == AIAEC_OK, "gates: freq postprocess after recovery");

    printf("section_gates: kind/size/null/sequence rejections precede any state write\n");
    pair_free(&t);
    return 0;
}

#if PRODUCT_DV
typedef enum FailKind { FAIL_STATUS = 0, FAIL_NONFINITE = 1, FAIL_BOTH = 2, FAIL_NOT_STARTED = 3 } FailKind;

static int frame_is_zero(const float *re, const float *im, int nb) {
    int i;
    for (i = 0; i < nb; ++i) if (re[i] != 0.0f || im[i] != 0.0f) return 0;
    return 1;
}

static float frame_max_diff(const float *a_re, const float *a_im, const float *b_re, const float *b_im, int nb, float *peak) {
    float d = 0.0f;
    int i;
    *peak = 0.0f;
    for (i = 0; i < nb; ++i) {
        d = fmaxf(d, fmaxf(fabsf(a_re[i] - b_re[i]), fabsf(a_im[i] - b_im[i])));
        *peak = fmaxf(*peak, fmaxf(fabsf(b_re[i]), fabsf(b_im[i])));
    }
    return d;
}

/* The _freq entry fails closed too. An identity accelerator returns the mic spectrum; the failed job's N frames are
 * zero spectra: the failed block's own N frames, which must be exactly zero, with a normal frame on either side. The failure is a dirty run status with valid taps,
 * non-finite taps with status OK, both, and STREAM_RUN_NOT_STARTED with valid taps. The state generation each SUBMIT
 * block carries must be the live one, across the failure and a reset: it moves by one for a dirty status or a refused
 * commit (both together count once) and stays for a run that never started. */
static int section_fail_closed_freq(int fft, int n, FailKind kind) {
    enum { BLOCKS = 6, FAILED = 2 };
    const int hop = fft / 2, nb = fft / 2 + 1, hop_in = hop * n, frames = BLOCKS * n;
    AiaecConfig cfg = make_cfg(fft, n);
    Pair t;
    Frame an;
    int16_t *mic = calloc((size_t)hop_in * BLOCKS, sizeof(int16_t));
    float *e_re = calloc((size_t)frames * nb, sizeof(float)), *e_im = calloc((size_t)frames * nb, sizeof(float));
    float *o_re = calloc((size_t)frames * nb, sizeof(float)), *o_im = calloc((size_t)frames * nb, sizeof(float));
    float *z = calloc((size_t)n * (size_t)nb, sizeof(float)), *hopf = calloc((size_t)hop, sizeof(float));
    unsigned int gen_seen[BLOCKS];
    const int fail_lo = FAILED * n, fail_hi = FAILED * n + n - 1;   /* output frame positions of the failed job */
    int b, i, p, bad = g_fail;

    if (pair_make(&t, &cfg) != 0) return 1;
    frame_make(&an, fft, 1, 0);
    for (i = 0; i < hop_in * BLOCKS; ++i) mic[i] = (int16_t)((((i * 977) % 20001) - 10000) / 2);
    for (p = 0; p < frames; ++p) {
        for (i = 0; i < hop; ++i) hopf[i] = (float)mic[(size_t)p * hop + (size_t)i] / 32768.0f;
        ac_stft_push_planar(an.stft, hopf, e_re + (size_t)p * nb, e_im + (size_t)p * nb);
    }
    for (b = 0; b < BLOCKS; ++b) {
        AiaecFreqInput fin;
        StreamWork w;
        const int fail = b == FAILED;
        int rc;

        memset(&fin, 0, sizeof(fin));
        fin.kind = AIAEC_INPUT_DEEPVQE_S; fin.frames = n;
        fin.u.deepvqe_s.mic_re = e_re + (size_t)b * n * nb; fin.u.deepvqe_s.mic_im = e_im + (size_t)b * n * nb;
        fin.u.deepvqe_s.far_re = z; fin.u.deepvqe_s.far_im = z;   /* n * nb floats are read: the far stays silent */
        rc = aiaec_preprocess_freq(t.freq, &fin, &w);
        CHECK(rc == AIAEC_OK && w.action == STREAM_WORK_SUBMIT, "fail_closed block %d: preprocess rc=%d", b, rc);
        gen_seen[b] = w.state_generation;
        CHECK(w.state_generation == aiaec_state_generation_freq(t.freq), "fail_closed block %d: work generation %u, live %u", b,
              w.state_generation, aiaec_state_generation_freq(t.freq));
        identity_head(&w, n);
        if (fail && (kind == FAIL_NONFINITE || kind == FAIL_BOTH)) w.outputs[0].data[w.outputs[0].elements / 3] = NAN;
        {
            const int status = !fail ? STREAM_RUN_OK : kind == FAIL_NOT_STARTED ? STREAM_RUN_NOT_STARTED : kind == FAIL_NONFINITE ? STREAM_RUN_OK : 1;
            float *re = o_re + (size_t)b * n * nb, *im = o_im + (size_t)b * n * nb;
            rc = aiaec_postprocess_freq(t.freq, status, re, im);
        }
        CHECK(rc == AIAEC_OK, "fail_closed block %d: postprocess rc=%d", b, rc);
    }
    /* output frame p is the identity on input frame p, except what the failed job owns */
    for (p = 0; p < frames; ++p) {
        const float *ore = o_re + (size_t)p * nb, *oim = o_im + (size_t)p * nb;
        if (p >= fail_lo && p <= fail_hi) {
            CHECK(frame_is_zero(ore, oim, nb), "fail_closed(fft=%d N=%d kind=%d): frame %d not exactly zero (a failed batch must be silent, not the mic)",
                  fft, n, (int)kind, p);
        } else {
            float peak, d = frame_max_diff(ore, oim, e_re + (size_t)p * nb, e_im + (size_t)p * nb, nb, &peak);
            CHECK(peak > 0.0f && d <= 2e-3f * peak, "fail_closed(fft=%d N=%d kind=%d): frame %d is not the mic spectrum (diff %g, peak %g)",
                  fft, n, (int)kind, p, (double)d, (double)peak);
        }
    }
    CHECK(gen_seen[FAILED + 1] == gen_seen[FAILED] + (kind == FAIL_NOT_STARTED ? 0u : 1u) && gen_seen[FAILED + 2] == gen_seen[FAILED + 1] &&
          gen_seen[FAILED] == gen_seen[FAILED - 1],
          "fail_closed(N=%d kind=%d): generation %u -> %u -> %u around the failed job", n, (int)kind,
          gen_seen[FAILED], gen_seen[FAILED + 1], gen_seen[FAILED + 2]);
    {   /* a reset advances the generation and the next SUBMIT block carries it */
        AiaecFreqInput fin;
        StreamWork w;
        const unsigned int before = aiaec_state_generation_freq(t.freq);
        unsigned int live;
        int rc;
        aiaec_reset_freq(t.freq);
        memset(&fin, 0, sizeof(fin));
        fin.kind = AIAEC_INPUT_DEEPVQE_S; fin.frames = n;
        fin.u.deepvqe_s.mic_re = e_re; fin.u.deepvqe_s.mic_im = e_im; fin.u.deepvqe_s.far_re = z; fin.u.deepvqe_s.far_im = z;
        rc = aiaec_preprocess_freq(t.freq, &fin, &w);
        live = aiaec_state_generation_freq(t.freq);
        CHECK(rc == AIAEC_OK && w.state_generation == live && w.state_generation != before && w.state_generation != 0u,
              "fail_closed: after reset work generation %u, live %u, before %u", w.state_generation, live, before);
        identity_head(&w, n);
        rc = aiaec_postprocess_freq(t.freq, 0, o_re, o_im);
        CHECK(rc == AIAEC_OK, "fail_closed: postprocess after reset");
    }
    printf("section_fail_closed_freq(fft=%d N=%d kind=%d): failed job's frames exactly zero, others the mic spectrum, generation tracked\n",
           fft, n, (int)kind);
    frame_free(&an);
    pair_free(&t);
    free(mic); free(e_re); free(e_im); free(o_re); free(o_im); free(z); free(hopf);
    return g_fail != bad;
}

static int section_fail_closed_all(void) {
    static const struct { int fft, n; } grids[3] = { { 256, 1 }, { 512, 1 }, { 512, 4 } };
    int g, k, bad = 0;
    for (g = 0; g < 3; ++g)
        for (k = FAIL_STATUS; k <= FAIL_NOT_STARTED; ++k) bad |= section_fail_closed_freq(grids[g].fft, grids[g].n, (FailKind)k);
    return bad;
}
#endif /* PRODUCT_DV */

int main(void) {
#if PRODUCT_DV
    static const Ev none[1] = { { 0, 0, 0 } };
    g_fail |= section_equivalence(256, 1, none, 6);
    g_fail |= section_equivalence(512, 1, none, 6);
    g_fail |= section_equivalence(512, 4, none, 6);
    g_fail |= section_tail(256, 1);
    g_fail |= section_tail(512, 1);
    g_fail |= section_tail(512, 4);
    g_fail |= section_bind_and_pool(256, 1);
    g_fail |= section_bind_and_pool(512, 1);
    g_fail |= section_bind_and_pool(512, 4);
    g_fail |= section_bind_and_pool(1024, 1);
    g_fail |= section_bind_and_pool(1024, 2);
    g_fail |= section_fail_closed_all();
#else
    static const Ev sched[12] = {
        { 0, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0, 1, 0 },
        { 0, 1, 1 }, { 0, 1, 0 }, { 1, 1, 1 }, { 0, 1, 0 }, { 0, 0, 0 }, { 0, 1, 0 },
    };
    g_fail |= section_equivalence(256, 1, sched, 12);
    g_fail |= section_equivalence(512, 1, sched, 12);
    g_fail |= section_events();
    g_fail |= section_freq_continuity(256);
    g_fail |= section_freq_continuity(512);
    g_fail |= section_tail(256, 1);
    g_fail |= section_tail(512, 1);
    g_fail |= section_bind_and_pool(256, 1);
    g_fail |= section_bind_and_pool(512, 1);
    g_fail |= section_bind_and_pool(512, 4);
    g_fail |= section_bind_and_pool(1024, 1);
    g_fail |= section_bind_and_pool(1024, 2);
#endif
    g_fail |= section_scale();
    g_fail |= section_gates();

    printf("test_aiaec_freq[%s]: %s\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
