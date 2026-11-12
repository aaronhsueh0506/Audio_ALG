/* Host coverage for the PCM -> spectrum ("_spec") and spectrum -> spectrum ("_freq") families through
 * the real public API, for the model the build binds (aiaec_product.h). Compiled once per product with
 * -DAIAEC_TEST_CATALOG; the accelerator is a host stand-in (deterministic heads, an identity head, or a
 * bypass).
 *
 *   equivalence  the PCM block family, _spec + an external AcIstft (reset on synthesis_restarted), and
 *                _freq fed by an independent, never-restarted AcStft run in lockstep: the PCM output
 *                equals the externally synthesized _spec output sample for sample, and the _freq
 *                spectra equal the _spec spectra bit for bit on every hop except an Align-ULCNet
 *                restart hop (where _freq must return its input error spectrum unchanged). DeepVQE-S
 *                at 16k/256 N=1 and N=8 and 16k/512 N=1 with a forced inference failure; Align-ULCNet
 *                with changed / solid events and a forced failure.
 *   events       Align-ULCNet: per hop SUBMIT or BYPASS (changed and solid false -> true hops bypass,
 *                the next hop submits) and synthesis_restarted (PCM and _spec; 0 for _freq).
 *   continuity   Align-ULCNet _freq with never-restarted analysers and ISTFT, bypassing every hop
 *                across alignment events: the output is the input delayed by one hop throughout.
 *   scale        the spectra at the boundary are audio scale: an identity head returns the supplied
 *                spectrum, a fail-open frame returns its input error spectrum bit for bit.
 *   tail         a single 16000-count sample at the last real input sample comes back at
 *                (real length - 1 + trim_samples) after the documented tail_zero_blocks, through every
 *                family; the queue-only drain with one zero spectrum loses it.
 *   bind/pool    per-family trim / queue / tail, ports; pools: short / misaligned refused, larger constructs,
 *                also when the pool is undersized.
 *   gates        kind / size / null / sequence rejections for the new entries. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"
#include "audio_stft.h"
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
    cfg.sr = 16000; cfg.fft = fft; cfg.hop_in = (fft / 2) * n;
    return cfg;
}

static void *alloc_aligned(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) { fprintf(stderr, "alloc fail\n"); exit(1); }
    return p;
}

/* One context per family on its own pool. */
typedef struct Trio {
    AiaecCtx *blk;
    AiaecSpecCtx *spec;
    AiaecFreqCtx *freq;
    void *p_blk, *p_spec, *p_freq;
    AiaecInfo info_blk, info_spec, info_freq;
} Trio;

static int trio_make(Trio *t, const AiaecConfig *cfg) {
    StreamMemReq rb, rs, rf;
    int e;
    memset(t, 0, sizeof(*t));
    if (aiaec_get_mem_size(cfg, &rb) != AIAEC_OK || aiaec_get_mem_size_spec(cfg, &rs) != AIAEC_OK ||
        aiaec_get_mem_size_freq(cfg, &rf) != AIAEC_OK) {
        FAIL("trio: get_mem_size failed for fft=%d hop_in=%d", cfg->fft, cfg->hop_in);
        return -1;
    }
    t->p_blk = alloc_aligned((size_t)rb.bytes); t->p_spec = alloc_aligned((size_t)rs.bytes);
    t->p_freq = alloc_aligned((size_t)rf.bytes);
    e = aiaec_init(cfg, t->p_blk, (size_t)rb.bytes, &t->blk, &t->info_blk);
    CHECK(e == AIAEC_OK && t->blk != NULL, "trio: init err=%d", e);
    e = aiaec_init_spec(cfg, t->p_spec, (size_t)rs.bytes, &t->spec, &t->info_spec);
    CHECK(e == AIAEC_OK && t->spec != NULL, "trio: init_spec err=%d", e);
    e = aiaec_init_freq(cfg, t->p_freq, (size_t)rf.bytes, &t->freq, &t->info_freq);
    CHECK(e == AIAEC_OK && t->freq != NULL, "trio: init_freq err=%d", e);
    return t->blk && t->spec && t->freq ? 0 : -1;
}

static void trio_free(Trio *t) { free(t->p_blk); free(t->p_spec); free(t->p_freq); }

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
static void fake_head(AiaecWork *w, long long ghop) {
    size_t i;
    float *d;
    if (w->action != AIAEC_SUBMIT) return;
    d = w->outputs[0].data;
    for (i = 0; i < w->outputs[0].elements; ++i)
        d[i] = PRODUCT_DV ? 0.05f * (float)(((ghop + (long long)i) % 37) - 18)
                          : 0.1f * (float)(((ghop + (long long)i) % 23) - 11);
}

/* CCM taps forced to (time 0, centre frequency) = 1 + 0j: the head is then the identity on the mic. */
static void identity_head(AiaecWork *w, int n) {
    const size_t tapn = 9, bins = w->outputs[0].elements / (tapn * 2u * (size_t)n);
    float *taps = w->outputs[0].data;
    size_t f, b;
    memset(taps, 0, w->outputs[0].elements * sizeof(float));
    for (f = 0; f < (size_t)n; ++f)
        for (b = 0; b < bins; ++b) taps[(f * bins + b) * tapn * 2u + 1u * 2u] = 1.0f;
}

typedef struct Ev { int changed, solid, fail; } Ev;

/* ---------------------------------------------------------------------------------------------
 * equivalence: block == spec + external ISTFT; freq == spec
 * ------------------------------------------------------------------------------------------- */
static int section_equivalence(int fft, int n, const Ev *sched, int blocks) {
    const int hop = fft / 2, nb = fft / 2 + 1, hop_in = hop * n;
    AiaecConfig cfg = make_cfg(fft, n);
    Trio t;
    Frame ist_s, ist_f, an_a, an_b;      /* external ISTFT for spec / freq; analysers for freq */
    int16_t *mic = calloc((size_t)hop_in, sizeof(int16_t)), *far16 = calloc((size_t)hop_in, sizeof(int16_t));
    float *resid = calloc((size_t)hop_in, sizeof(float)), *farf = calloc((size_t)hop_in, sizeof(float));
    float *a_re = calloc((size_t)n * nb, sizeof(float)), *a_im = calloc((size_t)n * nb, sizeof(float));
    float *b_re = calloc((size_t)n * nb, sizeof(float)), *b_im = calloc((size_t)n * nb, sizeof(float));
    float *s_re = calloc((size_t)n * nb, sizeof(float)), *s_im = calloc((size_t)n * nb, sizeof(float));
    float *f_re = calloc((size_t)n * nb, sizeof(float)), *f_im = calloc((size_t)n * nb, sizeof(float));
    float *hopf = calloc((size_t)hop, sizeof(float)), *hopg = calloc((size_t)hop, sizeof(float));
    int16_t *p_blk = calloc((size_t)hop_in, sizeof(int16_t)), *p_spec = calloc((size_t)hop_in, sizeof(int16_t));
    int16_t *p_freq = calloc((size_t)hop_in, sizeof(int16_t));
    AiaecAlignEvent ev[8];
    int b, k, i, bad = 0, had_hop = 0, prev_solid = 0, restarts = 0, fails = 0;
    long long ghop = 0;

    (void)ev; (void)had_hop; (void)prev_solid;
    if (trio_make(&t, &cfg) != 0) return 1;
    frame_make(&ist_s, fft, 0, 1); frame_make(&ist_f, fft, 0, 1);
    frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0);

    for (b = 0; b < blocks; ++b) {
        AiaecInput in;
        AiaecFreqInput fin;
        AiaecWork wb, ws, wf;
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
        /* the front end's analysers are never restarted (aiaec.h, "Alignment restart") */
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
        CHECK(aiaec_preprocess_spec(t.spec, &in, &ws) == AIAEC_OK, "fft=%d N=%d block %d: spec preprocess", fft, n, b);
        CHECK(aiaec_preprocess_freq(t.freq, &fin, &wf) == AIAEC_OK, "fft=%d N=%d block %d: freq preprocess", fft, n, b);

        CHECK(wb.action == ws.action && wb.action == wf.action &&
              wb.synthesis_restarted == ws.synthesis_restarted && wf.synthesis_restarted == 0 &&
              wb.synthesis_restarted == edge, "fft=%d N=%d block %d: action/restart differ across families (%d/%d/%d restart %d/%d/%d, edge %d)",
              fft, n, b, (int)wb.action, (int)ws.action, (int)wf.action, wb.synthesis_restarted, ws.synthesis_restarted,
              wf.synthesis_restarted, edge);
        restarts += wb.synthesis_restarted;
        if (wb.action == AIAEC_SUBMIT) {
            for (k = 0; k < NINPUTS; ++k)
                CHECK(wb.inputs[k].elements == ws.inputs[k].elements && wb.inputs[k].elements == wf.inputs[k].elements &&
                      memcmp(wb.inputs[k].data, ws.inputs[k].data, wb.inputs[k].elements * sizeof(float)) == 0 &&
                      memcmp(wb.inputs[k].data, wf.inputs[k].data, wb.inputs[k].elements * sizeof(float)) == 0,
                      "fft=%d N=%d block %d: accelerator input %d differs across families", fft, n, b, k);
            CHECK(wb.state_generation == ws.state_generation && wb.state_generation == wf.state_generation,
                  "block %d: state_generation differs across families", b);
            if (!fail) { fake_head(&wb, ghop); fake_head(&ws, ghop); fake_head(&wf, ghop); }
        }
        fails += fail && wb.action == AIAEC_SUBMIT;
        CHECK(aiaec_postprocess(t.blk, fail, p_blk) == AIAEC_OK, "block %d: block postprocess", b);
        CHECK(aiaec_postprocess_spec(t.spec, fail, s_re, s_im) == AIAEC_OK, "block %d: spec postprocess", b);
        CHECK(aiaec_postprocess_freq(t.freq, fail, f_re, f_im) == AIAEC_OK, "block %d: freq postprocess", b);

        if (ws.synthesis_restarted) ac_istft_reset(ist_s.istft);   /* _freq's ISTFT is never restarted */
        synth_frames(&ist_s, n, hop, nb, s_re, s_im, p_spec, hopf);
        synth_frames(&ist_f, n, hop, nb, f_re, f_im, p_freq, hopf);
        if (memcmp(p_blk, p_spec, (size_t)hop_in * sizeof(int16_t)) != 0) {
            FAIL("fft=%d N=%d block %d: block PCM != _spec + external ISTFT", fft, n, b); bad = 1;
        }
        if (edge) {
            /* the restart hop: _freq passes the front end's error spectrum through unchanged */
            if (memcmp(a_re, f_re, (size_t)nb * sizeof(float)) != 0 || memcmp(a_im, f_im, (size_t)nb * sizeof(float)) != 0) {
                FAIL("fft=%d N=%d block %d: _freq restart hop is not its input error spectrum", fft, n, b); bad = 1;
            }
        } else if (memcmp(s_re, f_re, (size_t)n * nb * sizeof(float)) != 0 || memcmp(s_im, f_im, (size_t)n * nb * sizeof(float)) != 0) {
            FAIL("fft=%d N=%d block %d: _freq spectra != _spec spectra", fft, n, b); bad = 1;
        }
        if (restarts == 0 && memcmp(p_blk, p_freq, (size_t)hop_in * sizeof(int16_t)) != 0) {
            FAIL("fft=%d N=%d block %d: block PCM != _freq + external ISTFT", fft, n, b); bad = 1;
        }
        if (PRODUCT_DV && n > 1 && b == 0) {
            int zero = 1;
            for (i = 0; i < (n - 1) * nb; ++i) if (s_re[i] != 0.0f || s_im[i] != 0.0f) zero = 0;
            CHECK(zero, "N=%d: the first block's leading %d spectrum frames are not zero", n, n - 1);
        }
        ghop += n;
    }
    printf("section_equivalence(fft=%d N=%d): %d blocks (%d forced failures, %d restarts) block == spec+ISTFT == freq+ISTFT, spec == freq%s\n",
           fft, n, blocks, fails, restarts, bad ? " -- DIVERGED" : "");
    frame_free(&ist_s); frame_free(&ist_f); frame_free(&an_a); frame_free(&an_b);
    trio_free(&t);
    free(mic); free(far16); free(resid); free(farf); free(a_re); free(a_im); free(b_re); free(b_im);
    free(s_re); free(s_im); free(f_re); free(f_im); free(hopf); free(hopg); free(p_blk); free(p_spec); free(p_freq);
    return bad;
}

#if PRODUCT_AL
/* ---------------------------------------------------------------------------------------------
 * events: SUBMIT / BYPASS per hop and synthesis_restarted, all three families
 * ------------------------------------------------------------------------------------------- */
static int section_events(void) {
    /* changed hop, solid false -> true hop, and the hop after each. */
    static const struct { int changed, solid, submit, restarted; } h[] = {
        { 0, 0, 1, 0 },
        { 1, 0, 0, 1 },   /* changed */
        { 0, 0, 1, 0 },   /* next hop: runs */
        { 0, 1, 0, 1 },   /* solid false -> true */
        { 0, 1, 1, 0 },   /* next hop: runs */
        { 0, 1, 1, 0 },
    };
    const int hops = (int)(sizeof(h) / sizeof(h[0]));
    AiaecConfig cfg = make_cfg(256, 1);
    Trio t;
    float resid[128], farf[128], re[129], im[129], a_re[129], a_im[129], b_re[129], b_im[129];
    int16_t pcm[128];
    Frame an_a, an_b;
    int i, k, sub_blk = 0, sub_spec = 0, sub_freq = 0, bad = 0;

    if (trio_make(&t, &cfg) != 0) return 1;
    frame_make(&an_a, 256, 1, 0); frame_make(&an_b, 256, 1, 0);
    for (i = 0; i < hops; ++i) {
        AiaecInput in;
        AiaecFreqInput fin;
        AiaecWork wb, ws, wf;
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
        CHECK(aiaec_preprocess(t.blk, &in, &wb) == AIAEC_OK && aiaec_preprocess_spec(t.spec, &in, &ws) == AIAEC_OK &&
              aiaec_preprocess_freq(t.freq, &fin, &wf) == AIAEC_OK, "events hop %d: preprocess", i);
        sub_blk += wb.action == AIAEC_SUBMIT; sub_spec += ws.action == AIAEC_SUBMIT; sub_freq += wf.action == AIAEC_SUBMIT;
        if ((wb.action == AIAEC_SUBMIT) != (h[i].submit != 0) || (ws.action == AIAEC_SUBMIT) != (h[i].submit != 0) ||
            (wf.action == AIAEC_SUBMIT) != (h[i].submit != 0)) {
            FAIL("events hop %d: submit block/spec/freq = %d/%d/%d, want %d", i, wb.action == AIAEC_SUBMIT,
                 ws.action == AIAEC_SUBMIT, wf.action == AIAEC_SUBMIT, h[i].submit);
            bad = 1;
        }
        CHECK(wb.synthesis_restarted == h[i].restarted && ws.synthesis_restarted == h[i].restarted &&
              wf.synthesis_restarted == 0, "events hop %d: synthesis_restarted %d/%d/%d, want %d/%d/0", i,
              wb.synthesis_restarted, ws.synthesis_restarted, wf.synthesis_restarted, h[i].restarted, h[i].restarted);
        if (wb.action == AIAEC_BYPASS)
            CHECK(wb.input_count == 0 && wb.output_count == 0 && wb.state_generation == 0u &&
                  ws.input_count == 0 && wf.input_count == 0, "events hop %d: bypass work not zeroed", i);
        /* infer_status is not read on a bypass hop */
        CHECK(aiaec_postprocess(t.blk, wb.action == AIAEC_SUBMIT ? 1 : 0xBAD1, pcm) == AIAEC_OK &&
              aiaec_postprocess_spec(t.spec, ws.action == AIAEC_SUBMIT ? 1 : 0xBAD1, re, im) == AIAEC_OK &&
              aiaec_postprocess_freq(t.freq, wf.action == AIAEC_SUBMIT ? 1 : 0xBAD1, re, im) == AIAEC_OK,
              "events hop %d: postprocess", i);
    }
    CHECK(sub_blk == 4 && sub_spec == 4 && sub_freq == 4, "events: submissions block/spec/freq = %d/%d/%d, want 4 of %d hops",
          sub_blk, sub_spec, sub_freq, hops);
    printf("section_events: %d hops, submissions %d/%d/%d (block/spec/freq), restart flags as scheduled\n", hops, sub_blk, sub_spec, sub_freq);
    frame_free(&an_a); frame_free(&an_b); trio_free(&t);
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
    Trio t;
    Frame an_a, an_b, ist;
    float *resid = calloc((size_t)hop * blocks, sizeof(float)), *farf = calloc((size_t)hop, sizeof(float));
    float a_re[257], a_im[257], b_re[257], b_im[257], o_re[257], o_im[257], out[256];
    int b, k, bad = 0, events = 0;

    if (trio_make(&t, &cfg) != 0) return 1;
    frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0); frame_make(&ist, fft, 0, 1);
    for (k = 0; k < hop * blocks; ++k) resid[k] = 0.01f * (float)(((long long)k * 7 + 3) % 61 - 30);
    for (k = 0; k < hop; ++k) farf[k] = 0.003f * (float)((k * 11) % 53 - 26);
    for (b = 0; b < blocks; ++b) {
        AiaecFreqInput fin;
        AiaecWork w;
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
        CHECK(aiaec_postprocess_freq(t.freq, w.action == AIAEC_SUBMIT ? 1 : 0, o_re, o_im) == AIAEC_OK, "continuity block %d: postprocess", b);
        ac_istft_push_planar(ist.istft, o_re, o_im, out);
        if (b == 0) continue;
        for (k = 0; k < hop; ++k) {
            int d = (int)stream_saturate_s16(out[k]) - (int)stream_saturate_s16(resid[(size_t)(b - 1) * hop + (size_t)k]);
            if (d < -2 || d > 2) { FAIL("continuity block %d sample %d: off by %d from the input delayed by one hop", b, k, d); bad = 1; break; }
        }
    }
    printf("section_freq_continuity(fft=%d): %d blocks across %d changed events, output == input delayed by one hop\n", fft, blocks, events);
    frame_free(&an_a); frame_free(&an_b); frame_free(&ist); trio_free(&t); free(resid); free(farf);
    return bad;
}
#endif

/* ---------------------------------------------------------------------------------------------
 * scale: audio scale at the boundary
 * ------------------------------------------------------------------------------------------- */
static int section_scale(void) {
    const int fft = 256, hop = 128, nb = 129;
    AiaecConfig cfg = make_cfg(fft, 1);
    Trio t;
    Frame an_a, an_b;
    float hopf[128], hopg[128], a_re[129], a_im[129], b_re[129], b_im[129], o_re[129], o_im[129];
    int b, k, bad = 0;
    double max_err = 0.0, max_abs = 0.0;

    if (trio_make(&t, &cfg) != 0) return 1;
    frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0);
    for (b = 0; b < 6; ++b) {
        AiaecFreqInput fin;
        AiaecWork w;
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
        CHECK(aiaec_preprocess_freq(t.freq, &fin, &w) == AIAEC_OK && w.action == AIAEC_SUBMIT, "scale block %d: preprocess", b);
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
    frame_free(&an_a); frame_free(&an_b); trio_free(&t);
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
 * (fail open: the error spectrum passes through). Returns the infer_status to close the block with. */
static int tail_accel(AiaecWork *w, int n) {
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

/* The info of one family (0 PCM, 1 _spec, 2 _freq) of cfg, from a throw-away init. */
static int family_info(const AiaecConfig *cfg, int family, AiaecInfo *info) {
    StreamMemReq rq;
    void *pool;
    int rc = family == 0 ? aiaec_get_mem_size(cfg, &rq) : family == 1 ? aiaec_get_mem_size_spec(cfg, &rq)
                                                                  : aiaec_get_mem_size_freq(cfg, &rq);
    if (rc != AIAEC_OK) return rc;
    pool = alloc_aligned((size_t)rq.bytes);
    if (family == 0) { AiaecCtx *c = NULL; rc = aiaec_init(cfg, pool, (size_t)rq.bytes, &c, info); }
    else if (family == 1) { AiaecSpecCtx *c = NULL; rc = aiaec_init_spec(cfg, pool, (size_t)rq.bytes, &c, info); }
    else { AiaecFreqCtx *c = NULL; rc = aiaec_init_freq(cfg, pool, (size_t)rq.bytes, &c, info); }
    free(pool);
    return rc;
}

/* The round trip on the final PCM and the model queue, from the grid alone: DeepVQE-S N * hop and N - 1
 * frames, Align-ULCNet one hop and no queue. */
static int expected_trim(int hop, int n) { return PRODUCT_DV ? n * hop : hop; }
static int expected_queue(int n) { return PRODUCT_DV ? n - 1 : 0; }

static int section_tail(int fft, int n) {
    const int hop = fft / 2, nb = fft / 2 + 1, hop_in = hop * n;
    const int want_trim = expected_trim(hop, n), want_queue = expected_queue(n);
    AiaecConfig cfg = make_cfg(fft, n);
    AiaecInfo bi_pcm, bi_spec, bi_freq;
    Trio t;
    Frame ist_s, ist_f, an_a, an_b, ist_x;
    int tail, total_blocks, b, f, k, bad = 0;
    size_t s_real = (size_t)BLOCKS_REAL * (size_t)hop_in, total, idx;
    int16_t *pcm, *out16, *zero16;
    float *outf, *outq, *pf, *resid, *fre, *fim, *a_re, *a_im, *b_re, *b_im, *hopf, *zre, *zim;
    AiaecAlignEvent ev[8];
    double v;
    char label[64];

    snprintf(label, sizeof(label), "tail(fft=%d N=%d)", fft, n);
    if (family_info(&cfg, 0, &bi_pcm) != AIAEC_OK || family_info(&cfg, 1, &bi_spec) != AIAEC_OK ||
        family_info(&cfg, 2, &bi_freq) != AIAEC_OK) { FAIL("%s: info", label); return 1; }
    tail = bi_pcm.tail_zero_blocks;
    CHECK(tail == bi_spec.tail_zero_blocks && tail == bi_freq.tail_zero_blocks &&
          tail == (want_trim + hop_in - 1) / hop_in, "%s: tail_zero_blocks %d/%d/%d", label, bi_pcm.tail_zero_blocks, bi_spec.tail_zero_blocks, bi_freq.tail_zero_blocks);
    CHECK(bi_pcm.trim_samples == want_trim && bi_spec.trim_samples == want_trim && bi_freq.trim_samples == want_trim &&
          bi_pcm.queue_frames == want_queue && bi_spec.queue_frames == want_queue && bi_freq.queue_frames == want_queue,
          "%s: trim %d/%d/%d (want %d), queue frames %d/%d/%d (want %d)", label, bi_pcm.trim_samples, bi_spec.trim_samples,
          bi_freq.trim_samples, want_trim, bi_pcm.queue_frames, bi_spec.queue_frames, bi_freq.queue_frames, want_queue);
    total_blocks = BLOCKS_REAL + tail;
    total = (size_t)total_blocks * (size_t)hop_in;
    idx = s_real - 1 + (size_t)want_trim;

    pcm = calloc(total, sizeof(int16_t)); out16 = calloc(total, sizeof(int16_t)); zero16 = calloc((size_t)hop_in, sizeof(int16_t));
    outf = calloc(total, sizeof(float)); outq = calloc(total, sizeof(float)); pf = calloc(total, sizeof(float));
    resid = calloc((size_t)hop_in, sizeof(float));
    fre = calloc((size_t)n * nb, sizeof(float)); fim = calloc((size_t)n * nb, sizeof(float));
    a_re = calloc((size_t)n * nb, sizeof(float)); a_im = calloc((size_t)n * nb, sizeof(float));
    b_re = calloc((size_t)n * nb, sizeof(float)); b_im = calloc((size_t)n * nb, sizeof(float));
    hopf = calloc((size_t)hop, sizeof(float)); zre = calloc((size_t)nb, sizeof(float)); zim = calloc((size_t)nb, sizeof(float));
    pcm[s_real - 1] = PULSE;

    if (trio_make(&t, &cfg) != 0) return 1;
    frame_make(&ist_s, fft, 0, 1); frame_make(&ist_f, fft, 0, 1); frame_make(&an_a, fft, 1, 0); frame_make(&an_b, fft, 1, 0);

    for (b = 0; b < total_blocks; ++b) {
        AiaecInput in;
        AiaecFreqInput fin;
        AiaecWork w;
        int st;
        const int16_t *blk = pcm + (size_t)b * (size_t)hop_in;

        for (f = 0; f < n; ++f) {   /* the upstream analyser sees the same blocks, zero blocks included */
            for (k = 0; k < hop; ++k) hopf[k] = (float)blk[(size_t)f * hop + (size_t)k] / 32768.0f;
            ac_stft_push_planar(an_a.stft, hopf, a_re + (size_t)f * nb, a_im + (size_t)f * nb);
            ac_stft_push_planar(an_b.stft, hopf, b_re + (size_t)f * nb, b_im + (size_t)f * nb);
        }
        tail_input(&in, &fin, blk, resid, hop_in, ev, a_re, a_im, b_re, b_im, n);

        if (aiaec_preprocess(t.blk, &in, &w) != AIAEC_OK) { FAIL("%s: block pre %d", label, b); return 1; }
        st = w.action == AIAEC_SUBMIT ? tail_accel(&w, n) : 0;
        if (aiaec_postprocess(t.blk, st, out16 + (size_t)b * hop_in) != AIAEC_OK) { FAIL("%s: block post %d", label, b); return 1; }

        if (aiaec_preprocess_spec(t.spec, &in, &w) != AIAEC_OK) { FAIL("%s: spec pre %d", label, b); return 1; }
        st = w.action == AIAEC_SUBMIT ? tail_accel(&w, n) : 0;
        if (aiaec_postprocess_spec(t.spec, st, fre, fim) != AIAEC_OK) { FAIL("%s: spec post %d", label, b); return 1; }
        for (f = 0; f < n; ++f)
            ac_istft_push_planar(ist_s.istft, fre + (size_t)f * nb, fim + (size_t)f * nb, outf + (size_t)b * hop_in + (size_t)f * hop);

        if (aiaec_preprocess_freq(t.freq, &fin, &w) != AIAEC_OK) { FAIL("%s: freq pre %d", label, b); return 1; }
        st = w.action == AIAEC_SUBMIT ? tail_accel(&w, n) : 0;
        if (aiaec_postprocess_freq(t.freq, st, fre, fim) != AIAEC_OK) { FAIL("%s: freq post %d", label, b); return 1; }
        for (f = 0; f < n; ++f)
            ac_istft_push_planar(ist_f.istft, fre + (size_t)f * nb, fim + (size_t)f * nb, outq + (size_t)b * hop_in + (size_t)f * hop);
    }

    for (k = 0; (size_t)k < total; ++k) pf[k] = (float)out16[k] / 32768.0f;
    bad |= check_pulse(pf, total, idx, label, &v);
    if (!bad) printf("%s: PCM family OK, out[%zu] = %.1f (tail blocks %d)\n", label, idx, v, tail);
    bad |= check_pulse(outf, total, idx, label, &v);
    if (!bad) printf("%s: _spec + external ISTFT OK, out[%zu] = %.1f\n", label, idx, v);
    bad |= check_pulse(outq, total, idx, label, &v);
    if (!bad) printf("%s: _freq (upstream analyser + external ISTFT) OK, out[%zu] = %.1f\n", label, idx, v);
    frame_free(&ist_s); frame_free(&ist_f); frame_free(&an_a); frame_free(&an_b); trio_free(&t);

    /* The failing procedure (N = 1): only the reported queue_frames in zero PCM hops, then one zero
     * SPECTRUM to the external ISTFT -- the frame holding the last real hop is never analysed. */
    if (n == 1) {
        const int zero_hops = bi_spec.queue_frames;
        StreamMemReq rs;
        void *pool;
        AiaecSpecCtx *cs = NULL;
        AiaecInfo is;
        if (aiaec_get_mem_size_spec(&cfg, &rs) != AIAEC_OK) return 1;
        pool = alloc_aligned((size_t)rs.bytes);
        if (aiaec_init_spec(&cfg, pool, (size_t)rs.bytes, &cs, &is) != AIAEC_OK) return 1;
        frame_make(&ist_x, fft, 0, 1);
        memset(outf, 0, total * sizeof(float));
        for (b = 0; b < BLOCKS_REAL + zero_hops; ++b) {
            AiaecInput in;
            AiaecFreqInput fin;
            AiaecWork w;
            tail_input(&in, &fin, pcm + (size_t)b * hop_in, resid, hop_in, ev, a_re, a_im, b_re, b_im, n);
            if (aiaec_preprocess_spec(cs, &in, &w) != AIAEC_OK) return 1;
            if (aiaec_postprocess_spec(cs, w.action == AIAEC_SUBMIT ? tail_accel(&w, n) : 0, fre, fim) != AIAEC_OK) return 1;
            ac_istft_push_planar(ist_x.istft, fre, fim, outf + (size_t)b * hop);
        }
        ac_istft_push_planar(ist_x.istft, zre, zim, outf + (size_t)b * hop);
        v = (double)outf[idx] * 32768.0;
        if (fabs(v - (double)PULSE) < 0.5 * PULSE) { FAIL("%s: the queue-only drain unexpectedly kept the pulse (%.1f)", label, v); bad = 1; }
        else printf("%s: queue-only drain + zero spectrum loses the tail: out[%zu] = %.1f (want %d)\n", label, idx, v, PULSE);
        frame_free(&ist_x); free(pool);
    }
    free(pcm); free(out16); free(zero16); free(outf); free(outq); free(pf); free(resid); free(fre); free(fim); free(a_re); free(a_im);
    free(b_re); free(b_im); free(hopf); free(zre); free(zim);
    return bad;
}

/* ---------------------------------------------------------------------------------------------
 * info, delays, pools, gates
 * ------------------------------------------------------------------------------------------- */
static int section_bind_and_pool(int fft, int n) {
    const int hop = fft / 2, want_trim = expected_trim(hop, n), want_queue = expected_queue(n);
    AiaecConfig cfg = make_cfg(fft, n);
    AiaecInfo pb, ps, pf;
    StreamMemReq rb, rs, rf;
    Trio t;
    unsigned char *pool;
    int rc, bad = 0;
    unsigned int gen_s, gen_f;

    if (family_info(&cfg, 0, &pb) != AIAEC_OK || family_info(&cfg, 1, &ps) != AIAEC_OK ||
        family_info(&cfg, 2, &pf) != AIAEC_OK) { FAIL("bind: info"); return 1; }
    CHECK(memcmp(&pb, &ps, sizeof(pb)) == 0 && memcmp(&pb, &pf, sizeof(pb)) == 0,
          "bind(fft=%d N=%d): the info (graph, trim, queue, tail) differs across families", fft, n);
    CHECK(pb.graph_path && ps.graph_path && pf.graph_path && strcmp(pb.graph_path, ps.graph_path) == 0 && strcmp(pb.graph_path, pf.graph_path) == 0,
          "bind: graph path differs across families");
    {
        AiaecConfig bad_cfg = make_cfg(fft, n);
        AiaecSpecCtx *cs = NULL;
        AiaecFreqCtx *cf = NULL;
        AiaecInfo scratch;
        unsigned char small[64];
        bad_cfg.hop_in += 1;
        CHECK(aiaec_init_spec(&bad_cfg, small, sizeof(small), &cs, &scratch) == AIAEC_ERR_IN_HOP && cs == NULL &&
              aiaec_init_freq(&bad_cfg, small, sizeof(small), &cf, &scratch) == AIAEC_ERR_IN_HOP && cf == NULL &&
              aiaec_get_mem_size_spec(&bad_cfg, &rs) == AIAEC_ERR_IN_HOP && aiaec_get_mem_size_freq(&bad_cfg, &rf) == AIAEC_ERR_IN_HOP,
              "bind: a bad hop_in must propagate AIAEC_ERR_IN_HOP through every family");
    }

    if (trio_make(&t, &cfg) != 0) return 1;
    CHECK(memcmp(&t.info_blk, &pb, sizeof(pb)) == 0 && memcmp(&t.info_spec, &pb, sizeof(pb)) == 0 &&
          memcmp(&t.info_freq, &pb, sizeof(pb)) == 0, "bind: the init info equals the throw-away init's");
    gen_s = aiaec_state_generation_spec(t.spec); gen_f = aiaec_state_generation_freq(t.freq);
    CHECK(pb.trim_samples == want_trim && ps.trim_samples == want_trim && pf.trim_samples == want_trim &&
          pb.queue_frames == want_queue && ps.queue_frames == want_queue && pf.queue_frames == want_queue,
          "bind(fft=%d N=%d): trim %d/%d/%d (want %d), queue %d/%d/%d (want %d)", fft, n, pb.trim_samples, ps.trim_samples,
          pf.trim_samples, want_trim, pb.queue_frames, ps.queue_frames, pf.queue_frames, want_queue);
    CHECK(aiaec_state_generation_spec(NULL) == 0u && aiaec_state_generation_freq(NULL) == 0u, "bind: NULL ctx");
    aiaec_reset_spec(t.spec); aiaec_reset_freq(t.freq);
    CHECK(aiaec_state_generation_spec(t.spec) != gen_s && aiaec_state_generation_freq(t.freq) != gen_f, "bind: reset must advance the generation");
    trio_free(&t);

    /* pools: short and misaligned are refused per family, the exact size and a larger pool construct */
    aiaec_get_mem_size(&cfg, &rb); aiaec_get_mem_size_spec(&cfg, &rs); aiaec_get_mem_size_freq(&cfg, &rf);
    pool = alloc_aligned((size_t)(rb.bytes > rs.bytes ? rb.bytes : rs.bytes) + (size_t)(rf.bytes) + 64);
    {
        AiaecCtx *cb = NULL;
        AiaecSpecCtx *cs = NULL;
        AiaecFreqCtx *cf = NULL;
        AiaecInfo info;
        rc = aiaec_init(&cfg, pool, (size_t)rb.bytes - 1u, &cb, &info);
        CHECK(rc == AIAEC_ERR_POOL && cb == NULL, "pool: undersized block rc=%d", rc);
        rc = aiaec_init_spec(&cfg, pool, (size_t)rs.bytes - 1u, &cs, &info);
        CHECK(rc == AIAEC_ERR_POOL && cs == NULL, "pool: undersized spec rc=%d", rc);
        rc = aiaec_init_freq(&cfg, pool, (size_t)rf.bytes - 1u, &cf, &info);
        CHECK(rc == AIAEC_ERR_POOL && cf == NULL, "pool: undersized freq rc=%d", rc);
        rc = aiaec_init_freq(&cfg, pool + 1, (size_t)rf.bytes, &cf, &info);
        CHECK(rc == AIAEC_ERR_POOL && cf == NULL, "pool: misaligned freq rc=%d", rc);
        CHECK(aiaec_init(&cfg, pool, (size_t)rb.bytes, &cb, &info) == AIAEC_OK && cb != NULL, "pool: the exact block size must construct");
        CHECK(aiaec_init_spec(&cfg, pool, (size_t)rs.bytes, &cs, &info) == AIAEC_OK && cs != NULL, "pool: the exact spec size must construct");
        CHECK(aiaec_init_freq(&cfg, pool, (size_t)rf.bytes + 64u, &cf, &info) == AIAEC_OK && cf != NULL, "pool: a larger freq pool must construct");
    }
    printf("section_bind_and_pool(fft=%d N=%d): info shared, trim %d and queue %d frames in every family, pool sizes enforced\n",
           fft, n, want_trim, want_queue);
    free(pool);
    return bad;
}

static int section_gates(void) {
    const int nb = 129;
    AiaecConfig cfg = make_cfg(256, 1);
    Trio t;
    float re[129], im[129], zeros[129];
    int16_t pcm[128];
    AiaecInput in;
    AiaecFreqInput fin;
    AiaecWork w;
    AiaecAlignEvent ev[2];
    unsigned int gen;
    int rc;

    if (trio_make(&t, &cfg) != 0) return 1;
    memset(zeros, 0, sizeof(zeros)); memset(ev, 0, sizeof(ev)); memset(pcm, 0, sizeof(pcm));
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
    CHECK(aiaec_postprocess_freq(t.freq, 0, re, im) == AIAEC_ERR_SEQUENCE && aiaec_postprocess_spec(t.spec, 0, re, im) == AIAEC_ERR_SEQUENCE,
          "gates: postprocess without preprocess");
    CHECK(aiaec_postprocess_freq(NULL, 0, re, im) == AIAEC_ERR_NULL && aiaec_postprocess_freq(t.freq, 0, NULL, im) == AIAEC_ERR_NULL &&
          aiaec_postprocess_spec(t.spec, 0, re, NULL) == AIAEC_ERR_NULL, "gates: NULL outputs");

    CHECK(aiaec_preprocess_freq(t.freq, &fin, &w) == AIAEC_OK, "gates: valid freq preprocess");
    CHECK(aiaec_preprocess_freq(t.freq, &fin, &w) == AIAEC_ERR_SEQUENCE, "gates: second freq preprocess before postprocess");
    if (w.action == AIAEC_SUBMIT) fake_head(&w, 0);
    CHECK(aiaec_postprocess_freq(t.freq, 0, NULL, NULL) == AIAEC_ERR_NULL, "gates: NULL outputs keep the block open");
    CHECK(aiaec_postprocess_freq(t.freq, 0, re, im) == AIAEC_OK, "gates: freq postprocess after recovery");

    /* spec: kind / size gates are the block family's */
    memset(&in, 0, sizeof(in));
    in.samples = 128;
#if PRODUCT_DV
    in.kind = AIAEC_INPUT_DEEPVQE_S; in.u.deepvqe_s.mic_in = pcm; in.u.deepvqe_s.far_in = pcm;
#else
    in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.u.align_ulcnet.residual_in = zeros; in.u.align_ulcnet.far_in = zeros;
    in.u.align_ulcnet.events = ev; in.u.align_ulcnet.event_count = 1;
#endif
    gen = aiaec_state_generation_spec(t.spec);
    in.samples = 64;
    CHECK(aiaec_preprocess_spec(t.spec, &in, &w) == AIAEC_ERR_INPUT_SIZE, "gates: spec wrong samples");
    in.samples = 128; in.kind = PRODUCT_DV ? AIAEC_INPUT_ALIGN_ULCNET : AIAEC_INPUT_DEEPVQE_S;
    CHECK(aiaec_preprocess_spec(t.spec, &in, &w) == AIAEC_ERR_INPUT_KIND, "gates: spec wrong kind");
    CHECK(aiaec_state_generation_spec(t.spec) == gen, "gates: a rejected spec input advanced the state");
    in.kind = PRODUCT_DV ? AIAEC_INPUT_DEEPVQE_S : AIAEC_INPUT_ALIGN_ULCNET;
    rc = aiaec_preprocess_spec(t.spec, &in, &w);
    CHECK(rc == AIAEC_OK, "gates: spec preprocess rc=%d", rc);
    CHECK(aiaec_preprocess_spec(t.spec, &in, &w) == AIAEC_ERR_SEQUENCE, "gates: second spec preprocess");
    if (w.action == AIAEC_SUBMIT) fake_head(&w, 0);
    CHECK(aiaec_postprocess_spec(t.spec, 0, re, im) == AIAEC_OK, "gates: spec postprocess");
    (void)nb;
    printf("section_gates: kind/size/null/sequence rejections precede any state write\n");
    trio_free(&t);
    return 0;
}

#if PRODUCT_DV
typedef enum FailKind { FAIL_STATUS = 0, FAIL_NONFINITE = 1, FAIL_BOTH = 2 } FailKind;

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

/* The spectrum entries fail closed too. An identity accelerator returns the mic spectrum, delayed by N - 1 frames; the
 * failed job's N frames are zero spectra: the failed block's last frame and the next block's first N - 1 frames, which
 * must be exactly zero, with a normal frame on either side. The failure is a reported infer_status != 0 with valid
 * taps, non-finite taps with infer_status 0, and both. The state generation each SUBMIT block carries must be the
 * live one, across the refused commit and a reset. `freq` selects the _freq entry (spectra in) over _spec (PCM in). */
static int section_fail_closed_spec_freq(int fft, int n, int freq, FailKind kind) {
    enum { BLOCKS = 6, FAILED = 2 };
    const int hop = fft / 2, nb = fft / 2 + 1, hop_in = hop * n, frames = BLOCKS * n;
    AiaecConfig cfg = make_cfg(fft, n);
    Trio t;
    Frame an;
    int16_t *mic = calloc((size_t)hop_in * BLOCKS, sizeof(int16_t)), *far = calloc((size_t)hop_in, sizeof(int16_t));
    float *e_re = calloc((size_t)frames * nb, sizeof(float)), *e_im = calloc((size_t)frames * nb, sizeof(float));
    float *o_re = calloc((size_t)frames * nb, sizeof(float)), *o_im = calloc((size_t)frames * nb, sizeof(float));
    float *z = calloc((size_t)nb, sizeof(float)), *hopf = calloc((size_t)hop, sizeof(float));
    unsigned int gen_seen[BLOCKS];
    const int fail_lo = FAILED * n + n - 1, fail_hi = (FAILED + 1) * n + n - 2;   /* output frame positions of the failed job */
    int b, i, p, bad = g_fail;

    if (trio_make(&t, &cfg) != 0) return 1;
    frame_make(&an, fft, 1, 0);
    for (i = 0; i < hop_in * BLOCKS; ++i) mic[i] = (int16_t)((((i * 977) % 20001) - 10000) / 2);
    for (p = 0; p < frames; ++p) {
        for (i = 0; i < hop; ++i) hopf[i] = (float)mic[(size_t)p * hop + (size_t)i] / 32768.0f;
        ac_stft_push_planar(an.stft, hopf, e_re + (size_t)p * nb, e_im + (size_t)p * nb);
    }
    for (b = 0; b < BLOCKS; ++b) {
        AiaecInput in;
        AiaecFreqInput fin;
        AiaecWork w;
        const int fail = b == FAILED;
        int rc;

        memset(&in, 0, sizeof(in));
        in.kind = AIAEC_INPUT_DEEPVQE_S; in.samples = hop_in;
        in.u.deepvqe_s.mic_in = mic + (size_t)b * hop_in; in.u.deepvqe_s.far_in = far;
        memset(&fin, 0, sizeof(fin));
        fin.kind = AIAEC_INPUT_DEEPVQE_S; fin.frames = n;
        fin.u.deepvqe_s.mic_re = e_re + (size_t)b * n * nb; fin.u.deepvqe_s.mic_im = e_im + (size_t)b * n * nb;
        fin.u.deepvqe_s.far_re = z; fin.u.deepvqe_s.far_im = z;   /* n * nb floats are read: the far stays silent */
        if (freq) rc = aiaec_preprocess_freq(t.freq, &fin, &w); else rc = aiaec_preprocess_spec(t.spec, &in, &w);
        CHECK(rc == AIAEC_OK && w.action == AIAEC_SUBMIT, "fail_closed_sf block %d: preprocess rc=%d", b, rc);
        gen_seen[b] = w.state_generation;
        CHECK(w.state_generation == (freq ? aiaec_state_generation_freq(t.freq) : aiaec_state_generation_spec(t.spec)),
              "fail_closed_sf block %d: work generation %u, live %u", b, w.state_generation,
              freq ? aiaec_state_generation_freq(t.freq) : aiaec_state_generation_spec(t.spec));
        identity_head(&w, n);
        if (fail && kind != FAIL_STATUS) w.outputs[0].data[w.outputs[0].elements / 3] = NAN;
        {
            const int status = fail && kind != FAIL_NONFINITE ? 1 : 0;
            float *re = o_re + (size_t)b * n * nb, *im = o_im + (size_t)b * n * nb;
            rc = freq ? aiaec_postprocess_freq(t.freq, status, re, im) : aiaec_postprocess_spec(t.spec, status, re, im);
        }
        CHECK(rc == AIAEC_OK, "fail_closed_sf block %d: postprocess rc=%d", b, rc);
    }
    /* output frame p is the identity on input frame p - (N - 1), except what the failed job owns and the lead-in */
    for (p = 0; p < frames; ++p) {
        const float *ore = o_re + (size_t)p * nb, *oim = o_im + (size_t)p * nb;
        if (p < n - 1 || p == fail_lo || (p > fail_lo && p <= fail_hi)) {
            CHECK(frame_is_zero(ore, oim, nb), "fail_closed_sf(%s fft=%d N=%d kind=%d): frame %d not exactly zero (a failed batch must be silent, not the mic)",
                  freq ? "freq" : "spec", fft, n, (int)kind, p);
        } else {
            float peak, d = frame_max_diff(ore, oim, e_re + (size_t)(p - (n - 1)) * nb, e_im + (size_t)(p - (n - 1)) * nb, nb, &peak);
            CHECK(peak > 0.0f && d <= 2e-3f * peak, "fail_closed_sf(%s fft=%d N=%d kind=%d): frame %d is not the delayed mic spectrum (diff %g, peak %g)",
                  freq ? "freq" : "spec", fft, n, (int)kind, p, (double)d, (double)peak);
        }
    }
    if (kind == FAIL_NONFINITE)
        CHECK(gen_seen[FAILED + 1] != gen_seen[FAILED] && gen_seen[FAILED + 1] != 0u && gen_seen[FAILED + 2] == gen_seen[FAILED + 1],
              "fail_closed_sf(%s N=%d): generation %u -> %u -> %u around the refused job", freq ? "freq" : "spec", n,
              gen_seen[FAILED], gen_seen[FAILED + 1], gen_seen[FAILED + 2]);
    {   /* a reset advances the generation and the next SUBMIT block carries it */
        AiaecInput in;
        AiaecFreqInput fin;
        AiaecWork w;
        const unsigned int before = freq ? aiaec_state_generation_freq(t.freq) : aiaec_state_generation_spec(t.spec);
        unsigned int live;
        int rc;
        if (freq) aiaec_reset_freq(t.freq); else aiaec_reset_spec(t.spec);
        memset(&in, 0, sizeof(in));
        in.kind = AIAEC_INPUT_DEEPVQE_S; in.samples = hop_in; in.u.deepvqe_s.mic_in = mic; in.u.deepvqe_s.far_in = far;
        memset(&fin, 0, sizeof(fin));
        fin.kind = AIAEC_INPUT_DEEPVQE_S; fin.frames = n;
        fin.u.deepvqe_s.mic_re = e_re; fin.u.deepvqe_s.mic_im = e_im; fin.u.deepvqe_s.far_re = z; fin.u.deepvqe_s.far_im = z;
        rc = freq ? aiaec_preprocess_freq(t.freq, &fin, &w) : aiaec_preprocess_spec(t.spec, &in, &w);
        live = freq ? aiaec_state_generation_freq(t.freq) : aiaec_state_generation_spec(t.spec);
        CHECK(rc == AIAEC_OK && w.state_generation == live && w.state_generation != before && w.state_generation != 0u,
              "fail_closed_sf(%s): after reset work generation %u, live %u, before %u", freq ? "freq" : "spec", w.state_generation, live, before);
        identity_head(&w, n);
        rc = freq ? aiaec_postprocess_freq(t.freq, 0, o_re, o_im) : aiaec_postprocess_spec(t.spec, 0, o_re, o_im);
        CHECK(rc == AIAEC_OK, "fail_closed_sf: postprocess after reset");
    }
    printf("section_fail_closed_spec_freq(%s fft=%d N=%d kind=%d): failed job's frames exactly zero, others the delayed mic spectrum, generation tracked\n",
           freq ? "freq" : "spec", fft, n, (int)kind);
    frame_free(&an);
    trio_free(&t);
    free(mic); free(far); free(e_re); free(e_im); free(o_re); free(o_im); free(z); free(hopf);
    return g_fail != bad;
}

static int section_fail_closed_all(void) {
    static const struct { int fft, n; } grids[3] = { { 256, 1 }, { 256, 8 }, { 512, 1 } };
    int g, f, k, bad = 0;
    for (g = 0; g < 3; ++g)
        for (f = 0; f < 2; ++f)
            for (k = FAIL_STATUS; k <= FAIL_BOTH; ++k) bad |= section_fail_closed_spec_freq(grids[g].fft, grids[g].n, f, (FailKind)k);
    return bad;
}
#endif /* PRODUCT_DV */

int main(void) {
#if PRODUCT_DV
    static const Ev none[1] = { { 0, 0, 0 } };
    g_fail |= section_equivalence(256, 1, none, 6);
    g_fail |= section_equivalence(256, 8, none, 6);
    g_fail |= section_equivalence(512, 1, none, 6);
    g_fail |= section_tail(256, 1);
    g_fail |= section_tail(256, 8);
    g_fail |= section_tail(512, 1);
    g_fail |= section_bind_and_pool(256, 1);
    g_fail |= section_bind_and_pool(256, 8);
    g_fail |= section_bind_and_pool(512, 1);
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
#endif
    g_fail |= section_scale();
    g_fail |= section_gates();

    printf("test_aiaec_spec_freq[%s]: %s\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
