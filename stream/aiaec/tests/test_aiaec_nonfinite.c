/* Non-finite input to the AIAEC product entries, for the model the build binds (aiaec_product.h); compiled once per
 * product with -DAIAEC_TEST_CATALOG.
 *
 *   limit    +-FLT_MAX in every plane of the _freq family (both models), and +-2^112 (AIAEC_PCM_LIMIT_BITS) in the
 *            Align-ULCNet residual and far of the PCM family, with each sign, an alternating sign, sign patterns that
 *            peak the DC, mid and Nyquist bins, and one plane at a time: the entry accepts it, every tensor handed to
 *            the accelerator is finite, and the block closed with the stand-in head (DeepVQE-S identity taps,
 *            Align-ULCNet a fixed mask) is finite. A spectrum bin therefore needs no bound beyond finite; the float
 *            PCM input of Align-ULCNet does: its own analysis overflows from 2^120 up, so a sample beyond
 *            AIAEC_PCM_LIMIT_BITS (the next float up, FLT_MAX) is refused like a NaN.
 *   refusal  NaN, +Inf and -Inf at the first and at the last element of every input plane: the entry returns
 *            AIAEC_ERR_NONFINITE_INPUT before any state change. No block is open (postprocess is a sequence error),
 *            the generation is unchanged, and a run that was offered the bad block in between equals, bit for bit,
 *            a twin that never saw it: the tensors handed out, the state generation and every output.
 *   overflow DeepVQE-S _freq: a mic spectrum near FLT_MAX is accepted and finite taps overflow the CCM sum; the block
 *            closes with zero spectra (never an Inf), the generation moves by one, and the next block equals a fresh
 *            instance's first block.
 *   order    a wrong kind, a wrong size or a NULL plane is reported as such even when the planes also hold NaN.
 *   mixed    a bad input offered while a block is open is a sequence error and leaves that block intact. */

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_test_fixture.h"
#include "aiaec_product.h"

#define PRODUCT_DV (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S)
#define PRODUCT_AL (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_ALIGN_ULCNET)

static int g_fail = 0;
#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } while (0)
#define CHECK(cond, ...) do { if (!(cond)) FAIL(__VA_ARGS__); } while (0)

typedef struct Grid { int sr, fft, n; } Grid;
static const Grid GRIDS[] = { { 16000, 256, 1 }, { 16000, 256, 8 }, { 16000, 512, 1 }, { 48000, 1024, 1 }, { 48000, 1024, 2 } };
#define N_GRIDS ((int)(sizeof(GRIDS) / sizeof(GRIDS[0])))

/* One instance of one family on a pool of exactly its size, and the buffers one block needs. */
typedef struct Inst {
    int freq, sr, fft, n, hop, nb, hop_in;
    AiaecCtx *pcm;
    AiaecFreqCtx *fc;
    void *pool;
    /* input planes: PCM family  Align-ULCNet [0] residual_in [1] far_in (hop_in floats); DeepVQE-S int16 mic16/far16
     * _freq family            four planes of n * nb floats (mic/err re, im, far re, im) */
    float plane[4][8 * 513];
    int16_t mic16[8 * 512], far16[8 * 512];
    AiaecAlignEvent ev[8];
} Inst;

static size_t plane_len(const Inst *x) { return (size_t)x->n * (size_t)x->nb; }
static int plane_count(const Inst *x) { return x->freq ? 4 : (PRODUCT_AL ? 2 : 0); }
static size_t this_plane_len(const Inst *x, int p) { return x->freq ? plane_len(x) : (size_t)x->hop_in * (p >= 0); }

static int inst_open(Inst *x, int sr, int fft, int n, int freq) {
    AiaecConfig cfg;
    StreamMemReq req;
    AiaecInfo info;
    int rc;
    memset(x, 0, sizeof(*x));
    x->freq = freq; x->sr = sr; x->fft = fft; x->n = n; x->hop = fft / 2; x->nb = fft / 2 + 1; x->hop_in = x->hop * n;
    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = x->hop_in;
    rc = freq ? aiaec_get_mem_size_freq(&cfg, &req) : aiaec_get_mem_size(&cfg, &req);
    if (rc != AIAEC_OK) { FAIL("get_mem_size fft=%d n=%d rc=%d", fft, n, rc); return -1; }
    x->pool = alloc_aligned((size_t)req.bytes);
    rc = freq ? aiaec_init_freq(&cfg, x->pool, (size_t)req.bytes, &x->fc, &info) : aiaec_init(&cfg, x->pool, (size_t)req.bytes, &x->pcm, &info);
    if (rc != AIAEC_OK) { FAIL("init fft=%d n=%d rc=%d", fft, n, rc); return -1; }
    return 0;
}

static void inst_close(Inst *x) { free(x->pool); }

static unsigned int inst_gen(const Inst *x) { return x->freq ? aiaec_state_generation_freq(x->fc) : aiaec_state_generation(x->pcm); }

/* Deterministic benign input of block `blk`. */
static void fill_clean(Inst *x, int blk) {
    uint32_t r = 0x9e3779b9u * (uint32_t)(blk * 8 + x->freq * 4 + 1);
    size_t i, k;
    for (i = 0; i < (size_t)x->hop_in; ++i) {
        r = r * 1664525u + 1013904223u;
        x->mic16[i] = (int16_t)(((int)(r >> 20) - 2048) * 4);
        r = r * 1664525u + 1013904223u;
        x->far16[i] = (int16_t)(((int)(r >> 20) - 2048) * 4);
    }
    for (k = 0; k < 4; ++k)
        for (i = 0; i < 8u * 513u; ++i) {
            r = r * 1664525u + 1013904223u;
            x->plane[k][i] = ((float)(r >> 9) / 4194304.0f - 1.0f) * (x->freq ? 0.5f : 0.3f);
        }
    memset(x->ev, 0, sizeof(x->ev));
}

/* Preprocess from the instance's current buffers. */
static int inst_pre(Inst *x, StreamWork *w) {
    if (x->freq) {
        AiaecFreqInput fin;
        memset(&fin, 0, sizeof(fin));
        fin.frames = x->n;
        if (PRODUCT_DV) {
            fin.kind = AIAEC_INPUT_DEEPVQE_S;
            fin.u.deepvqe_s.mic_re = x->plane[0]; fin.u.deepvqe_s.mic_im = x->plane[1];
            fin.u.deepvqe_s.far_re = x->plane[2]; fin.u.deepvqe_s.far_im = x->plane[3];
        } else {
            fin.kind = AIAEC_INPUT_ALIGN_ULCNET;
            fin.u.align_ulcnet.err_re = x->plane[0]; fin.u.align_ulcnet.err_im = x->plane[1];
            fin.u.align_ulcnet.far_re = x->plane[2]; fin.u.align_ulcnet.far_im = x->plane[3];
            fin.u.align_ulcnet.events = x->ev; fin.u.align_ulcnet.event_count = x->n;
        }
        return aiaec_preprocess_freq(x->fc, &fin, w);
    } else {
        AiaecInput in;
        memset(&in, 0, sizeof(in));
        in.samples = x->hop_in;
        if (PRODUCT_DV) {
            in.kind = AIAEC_INPUT_DEEPVQE_S;
            in.u.deepvqe_s.mic_in = x->mic16; in.u.deepvqe_s.far_in = x->far16;
        } else {
            in.kind = AIAEC_INPUT_ALIGN_ULCNET;
            in.u.align_ulcnet.residual_in = x->plane[0]; in.u.align_ulcnet.far_in = x->plane[1];
            in.u.align_ulcnet.events = x->ev; in.u.align_ulcnet.event_count = x->n;
        }
        return aiaec_preprocess(x->pcm, &in, w);
    }
}

/* Output of one block: PCM (hop_in int16) or the spectra (n * nb floats each). */
typedef struct Out { int16_t pcm[8 * 512]; float re[8 * 513], im[8 * 513]; } Out;

static int inst_post(Inst *x, int status, Out *o) {
    return x->freq ? aiaec_postprocess_freq(x->fc, status, o->re, o->im) : aiaec_postprocess(x->pcm, status, o->pcm);
}

/* The accelerator stand-in: identity CCM taps for DeepVQE-S, a fixed finite mask for Align-ULCNet. */
static void run_accel(const StreamWork *w, int n) {
    float *d = w->outputs[0].data;
    size_t i, b, f;
    if (w->action != STREAM_WORK_SUBMIT) return;
    if (PRODUCT_DV) {
        const size_t tapn = 9, bins = w->outputs[0].elements / (tapn * 2u * (size_t)n);
        memset(d, 0, w->outputs[0].elements * sizeof(float));
        for (f = 0; f < (size_t)n; ++f)
            for (b = 0; b < bins; ++b) d[(f * bins + b) * tapn * 2u + 1u * 2u] = 1.0f;
    } else {
        for (i = 0; i < w->outputs[0].elements; ++i) d[i] = i < w->outputs[0].elements / 2u ? 0.7f : 0.1f;
    }
}

static int all_finite(const float *p, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (!isfinite(p[i])) return 0;
    return 1;
}

static int out_finite(const Inst *x, const Out *o) {
    return x->freq ? all_finite(o->re, plane_len(x)) && all_finite(o->im, plane_len(x)) : 1;
}

static int work_finite(const StreamWork *w) {
    int i;
    for (i = 0; i < w->input_count; ++i) if (!all_finite(w->inputs[i].data, w->inputs[i].elements)) return 0;
    return 1;
}

/* ---------------------------------------------------------------------------------------------
 * limit
 * ------------------------------------------------------------------------------------------- */
typedef enum { PAT_POS_ALL, PAT_NEG_ALL, PAT_ALT, PAT_ONE_POS, PAT_ONE_NEG, PAT_COS_DC, PAT_COS_MID, PAT_COS_NYQ } Pattern;

/* The largest accepted magnitude of the instance's input: FLT_MAX for spectra, 2^112 for Align-ULCNet float PCM. */
static float limit_mag(const Inst *x) {
    uint32_t bits = AIAEC_PCM_LIMIT_BITS;
    float v;
    if (x->freq) return FLT_MAX;
    memcpy(&v, &bits, sizeof v);
    return v;
}

static float above_limit(const Inst *x) { return nextafterf(limit_mag(x), INFINITY); }

static void fill_limit(Inst *x, Pattern pat, int only) {
    int p;
    size_t i;
    for (p = 0; p < plane_count(x); ++p) {
        const size_t len = this_plane_len(x, p);
        for (i = 0; i < len; ++i) {
            const float m = limit_mag(x);
            const double ph = 6.283185307179586 * (double)i / (double)x->fft;
            float v = m;
            switch (pat) {
            case PAT_POS_ALL: v = m; break;
            case PAT_NEG_ALL: v = -m; break;
            case PAT_ALT: v = ((i + (size_t)p) & 1u) ? -m : m; break;
            case PAT_ONE_POS: v = p == only ? m : x->plane[p][i]; break;
            case PAT_ONE_NEG: v = p == only ? -m : x->plane[p][i]; break;
            case PAT_COS_DC: v = m; break;
            case PAT_COS_MID: v = cos(ph * (double)(x->fft / 4)) >= 0.0 ? m : -m; break;
            case PAT_COS_NYQ: v = cos(ph * (double)(x->fft / 2)) >= 0.0 ? m : -m; break;
            }
            x->plane[p][i] = v;
        }
    }
}

static int section_limit(void) {
    static Out out;
    int g, fam, pat, only, cases = 0, bad = 0;
    for (g = 0; g < N_GRIDS; ++g) {
        for (fam = 0; fam < 2; ++fam) {
            if (PRODUCT_DV && !fam) continue;   /* DeepVQE-S PCM input is int16: it cannot be non-finite */
            for (pat = PAT_POS_ALL; pat <= PAT_COS_NYQ; ++pat) {
                const int n_only = (pat == PAT_ONE_POS || pat == PAT_ONE_NEG) ? 4 : 1;
                for (only = 0; only < n_only; ++only) {
                    Inst x;
                    StreamWork w;
                    int rc;
                    if (!fam && only >= 2) continue;
                    if (fam && pat >= PAT_COS_DC) continue;   /* the sign patterns exist for the PCM input only */
                    if (inst_open(&x, GRIDS[g].sr, GRIDS[g].fft, GRIDS[g].n, fam) != 0) return 1;
                    fill_clean(&x, 0);
                    fill_limit(&x, (Pattern)pat, only);
                    rc = inst_pre(&x, &w);
                    if (rc != AIAEC_OK) {
                        FAIL("limit fft=%d N=%d %s pat=%d plane=%d: input at the limit refused rc=%d (%s)", GRIDS[g].fft, GRIDS[g].n,
                             fam ? "freq" : "pcm", pat, only, rc, aiaec_status_text(rc));
                        bad = 1; inst_close(&x); continue;
                    }
                    if (!work_finite(&w)) {
                        FAIL("limit fft=%d N=%d %s pat=%d plane=%d: a feature tensor handed to the accelerator is not finite",
                             GRIDS[g].fft, GRIDS[g].n, fam ? "freq" : "pcm", pat, only);
                        bad = 1;
                    }
                    run_accel(&w, x.n);
                    rc = inst_post(&x, STREAM_RUN_OK, &out);
                    CHECK(rc == AIAEC_OK && out_finite(&x, &out), "limit fft=%d N=%d %s pat=%d plane=%d: postprocess rc=%d finite=%d",
                          GRIDS[g].fft, GRIDS[g].n, fam ? "freq" : "pcm", pat, only, rc, out_finite(&x, &out));
                    ++cases;
                    inst_close(&x);
                }
            }
        }
    }
    printf("section_limit: %d cases at the largest accepted magnitude (FLT_MAX spectra, 2^112 float PCM) accepted, every feature tensor finite, every block closed with the stand-in head finite\n", cases);
    return bad;
}

#if PRODUCT_DV
/* ---------------------------------------------------------------------------------------------
 * overflow
 * ------------------------------------------------------------------------------------------- */
static int section_overflow(void) {
    static Out out, fresh_out;
    int g, bad = 0;
    for (g = 0; g < N_GRIDS; ++g) {
        Inst x, y;
        StreamWork w;
        size_t i, len;
        unsigned int before;
        int rc, zero = 1;
        if (inst_open(&x, GRIDS[g].sr, GRIDS[g].fft, GRIDS[g].n, 1) != 0 || inst_open(&y, GRIDS[g].sr, GRIDS[g].fft, GRIDS[g].n, 1) != 0) return 1;
        len = plane_len(&x);
        fill_clean(&x, 0);
        for (i = 0; i < len; ++i) { x.plane[0][i] = 3.0e38f; x.plane[1][i] = 0.0f; }
        before = inst_gen(&x);
        rc = inst_pre(&x, &w);
        CHECK(rc == AIAEC_OK, "overflow fft=%d N=%d: a finite mic spectrum near FLT_MAX refused rc=%d", GRIDS[g].fft, GRIDS[g].n, rc);
        for (i = 0; rc == AIAEC_OK && i < w.outputs[0].elements; ++i) w.outputs[0].data[i] = 0.5f;
        rc = inst_post(&x, STREAM_RUN_OK, &out);
        for (i = 0; i < len; ++i) zero = zero && out.re[i] == 0.0f && out.im[i] == 0.0f;
        CHECK(rc == AIAEC_OK && out_finite(&x, &out) && zero, "overflow fft=%d N=%d: postprocess rc=%d, spectra finite %d, zero %d",
              GRIDS[g].fft, GRIDS[g].n, rc, out_finite(&x, &out), zero);
        CHECK(inst_gen(&x) == before + 1u, "overflow fft=%d N=%d: generation %u -> %u, want +1", GRIDS[g].fft, GRIDS[g].n, before, inst_gen(&x));

        fill_clean(&x, 1);
        fill_clean(&y, 1);
        CHECK(inst_pre(&x, &w) == AIAEC_OK, "overflow: next block refused");
        run_accel(&w, x.n);
        CHECK(inst_post(&x, STREAM_RUN_OK, &out) == AIAEC_OK, "overflow: next block postprocess");
        CHECK(inst_pre(&y, &w) == AIAEC_OK, "overflow: fresh block refused");
        run_accel(&w, y.n);
        CHECK(inst_post(&y, STREAM_RUN_OK, &fresh_out) == AIAEC_OK, "overflow: fresh block postprocess");
        CHECK(memcmp(out.re, fresh_out.re, len * sizeof(float)) == 0 && memcmp(out.im, fresh_out.im, len * sizeof(float)) == 0,
              "overflow fft=%d N=%d: the block after the refusal differs from a fresh instance's first block", GRIDS[g].fft, GRIDS[g].n);
        bad |= g_fail;
        inst_close(&x);
        inst_close(&y);
    }
    printf("section_overflow: a finite mic spectrum near FLT_MAX with finite taps closes with zero spectra, one generation step, cold restart\n");
    return bad;
}
#endif

/* ---------------------------------------------------------------------------------------------
 * refusal
 * ------------------------------------------------------------------------------------------- */
#define TRACE_BLOCKS 6
#define BAD_AT 3

typedef struct Trace {
    unsigned char *buf;
    size_t len, cap;
} Trace;

static void trace_add(Trace *t, const void *p, size_t n) {
    if (t->len + n > t->cap) { fprintf(stderr, "trace overflow\n"); exit(1); }
    memcpy(t->buf + t->len, p, n);
    t->len += n;
}

/* Runs TRACE_BLOCKS clean blocks. With bad_plane >= 0 a corrupt block (value at index idx of that plane) is offered
 * before block BAD_AT and must be refused; it is never offered again. Appends, per clean block, the action, the
 * generation the work carries, the tensors handed to the accelerator and the output. */
static int run_trace(int sr, int fft, int n, int fam, int bad_plane, int idx_last, float bad_value, Trace *t, const char *label) {
    static Out out;
    Inst x;
    int blk, rc, ok = 1;

    if (inst_open(&x, sr, fft, n, fam) != 0) return 1;
    t->len = 0;
    for (blk = 0; blk < TRACE_BLOCKS; ++blk) {
        StreamWork w;
        unsigned int gen_before;
        int i;

        fill_clean(&x, blk);
        if (blk == BAD_AT && bad_plane >= 0) {
            const size_t len = this_plane_len(&x, bad_plane);
            float keep[8 * 513];
            const size_t at = idx_last ? len - 1u : 0u;
            Out scratch;
            memcpy(keep, x.plane[bad_plane], len * sizeof(float));
            x.plane[bad_plane][at] = bad_value;
            gen_before = inst_gen(&x);
            memset(&w, 0x5a, sizeof(w));
            rc = inst_pre(&x, &w);
            if (rc != AIAEC_ERR_NONFINITE_INPUT) {
                FAIL("%s: plane %d %s %g: preprocess rc=%d (%s), want AIAEC_ERR_NONFINITE_INPUT", label, bad_plane, idx_last ? "last" : "first",
                     (double)bad_value, rc, aiaec_status_text(rc));
                ok = 0;
            }
            CHECK(inst_gen(&x) == gen_before, "%s: a refused block advanced the generation", label);
            CHECK(inst_post(&x, STREAM_RUN_OK, &scratch) == AIAEC_ERR_SEQUENCE, "%s: a block is open after the refusal", label);
            memcpy(x.plane[bad_plane], keep, len * sizeof(float));
            if (rc != AIAEC_ERR_NONFINITE_INPUT) { inst_close(&x); return 1; }
        }
        rc = inst_pre(&x, &w);
        if (rc != AIAEC_OK) { FAIL("%s: clean block %d refused rc=%d", label, blk, rc); inst_close(&x); return 1; }
        trace_add(t, &w.action, sizeof(w.action));
        trace_add(t, &w.state_generation, sizeof(w.state_generation));
        for (i = 0; i < w.input_count; ++i) trace_add(t, w.inputs[i].data, w.inputs[i].elements * sizeof(float));
        run_accel(&w, x.n);
        rc = inst_post(&x, STREAM_RUN_OK, &out);
        if (rc != AIAEC_OK) { FAIL("%s: postprocess block %d rc=%d", label, blk, rc); inst_close(&x); return 1; }
        if (x.freq) { trace_add(t, out.re, plane_len(&x) * sizeof(float)); trace_add(t, out.im, plane_len(&x) * sizeof(float)); }
        else trace_add(t, out.pcm, (size_t)x.hop_in * sizeof(int16_t));
        {
            const unsigned int g = inst_gen(&x);
            trace_add(t, &g, sizeof(g));
        }
    }
    inst_close(&x);
    return ok ? 0 : 1;
}

static int section_refusal(void) {
    Trace twin, probe;
    int g, fam, pl, last, v, cases = 0, bad = 0;

    twin.cap = probe.cap = 4u << 20;
    twin.buf = (unsigned char *)malloc(twin.cap);
    probe.buf = (unsigned char *)malloc(probe.cap);
    for (g = 0; g < N_GRIDS; ++g) {
        for (fam = 0; fam < 2; ++fam) {
            Inst x;
            int planes, n_bad;
            float BAD[6];
            char label[96];
            if (PRODUCT_DV && !fam) continue;
            if (inst_open(&x, GRIDS[g].sr, GRIDS[g].fft, GRIDS[g].n, fam) != 0) return 1;
            planes = plane_count(&x);
            BAD[0] = NAN; BAD[1] = INFINITY; BAD[2] = -INFINITY;
            n_bad = 3;
            if (!fam) {   /* float PCM: a finite sample beyond the limit is refused too */
                BAD[n_bad++] = above_limit(&x); BAD[n_bad++] = -above_limit(&x); BAD[n_bad++] = FLT_MAX;
            }
            inst_close(&x);
            snprintf(label, sizeof(label), "refusal fft=%d N=%d %s twin", GRIDS[g].fft, GRIDS[g].n, fam ? "freq" : "pcm");
            if (run_trace(GRIDS[g].sr, GRIDS[g].fft, GRIDS[g].n, fam, -1, 0, 0.0f, &twin, label) != 0) { bad = 1; continue; }
            for (pl = 0; pl < planes; ++pl)
                for (last = 0; last < 2; ++last)
                    for (v = 0; v < n_bad; ++v) {
                        snprintf(label, sizeof(label), "refusal fft=%d N=%d %s", GRIDS[g].fft, GRIDS[g].n, fam ? "freq" : "pcm");
                        if (run_trace(GRIDS[g].sr, GRIDS[g].fft, GRIDS[g].n, fam, pl, last, BAD[v], &probe, label) != 0) { bad = 1; continue; }
                        if (probe.len != twin.len || memcmp(probe.buf, twin.buf, twin.len) != 0) {
                            FAIL("%s: plane %d %s %g: the run that was offered the bad block differs from the twin (tensors, generation or output)",
                                 label, pl, last ? "last" : "first", (double)BAD[v]);
                            bad = 1;
                        }
                        ++cases;
                    }
        }
    }
    free(twin.buf); free(probe.buf);
    printf("section_refusal: %d injections (NaN, +-Inf; Align-ULCNet PCM also the next float above the limit and FLT_MAX) refused with AIAEC_ERR_NONFINITE_INPUT, state untouched (equal to a twin that never saw them)\n", cases);
    return bad;
}

/* ---------------------------------------------------------------------------------------------
 * order and an open block
 * ------------------------------------------------------------------------------------------- */
static int section_order(void) {
    static Out out;
    Inst x;
    StreamWork w;
    const int fam = PRODUCT_AL ? 0 : 1;
    int bad = 0;

    if (inst_open(&x, 16000, 256, 1, fam) != 0) return 1;
    fill_clean(&x, 0);
    x.plane[0][0] = NAN;
    if (PRODUCT_AL && !fam) {
        AiaecInput in;
        memset(&in, 0, sizeof(in));
        in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.samples = x.hop_in;
        in.u.align_ulcnet.residual_in = x.plane[0]; in.u.align_ulcnet.far_in = x.plane[1];
        in.u.align_ulcnet.events = x.ev; in.u.align_ulcnet.event_count = x.n;
        in.kind = AIAEC_INPUT_DEEPVQE_S;
        CHECK(aiaec_preprocess(x.pcm, &in, &w) == AIAEC_ERR_INPUT_KIND, "order: wrong kind with a NaN plane is not INPUT_KIND");
        in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.samples = x.hop_in + 1;
        CHECK(aiaec_preprocess(x.pcm, &in, &w) == AIAEC_ERR_INPUT_SIZE, "order: wrong size with a NaN plane is not INPUT_SIZE");
        in.samples = x.hop_in; in.u.align_ulcnet.event_count = 2;
        CHECK(aiaec_preprocess(x.pcm, &in, &w) == AIAEC_ERR_INPUT_SIZE, "order: wrong event count with a NaN plane is not INPUT_SIZE");
        in.u.align_ulcnet.event_count = x.n; in.u.align_ulcnet.far_in = NULL;
        CHECK(aiaec_preprocess(x.pcm, &in, &w) == AIAEC_ERR_NULL, "order: a NULL plane with a NaN plane is not ERR_NULL");
        in.u.align_ulcnet.far_in = x.plane[1];
        CHECK(aiaec_preprocess(x.pcm, &in, &w) == AIAEC_ERR_NONFINITE_INPUT, "order: the NaN plane alone is not refused");
        /* a block is open: the bad input is a sequence error, the open block closes normally */
        x.plane[0][0] = 0.1f;
        CHECK(aiaec_preprocess(x.pcm, &in, &w) == AIAEC_OK, "order: clean preprocess");
        x.plane[0][0] = NAN;
        CHECK(aiaec_preprocess(x.pcm, &in, &w) == AIAEC_ERR_SEQUENCE, "order: a bad input while a block is open is not ERR_SEQUENCE");
        x.plane[0][0] = 0.1f;
        run_accel(&w, x.n);
        CHECK(aiaec_postprocess(x.pcm, STREAM_RUN_OK, out.pcm) == AIAEC_OK, "order: the open block did not close");
    } else {
        AiaecFreqInput fin;
        memset(&fin, 0, sizeof(fin));
        fin.frames = x.n;
        if (PRODUCT_DV) {
            fin.kind = AIAEC_INPUT_DEEPVQE_S;
            fin.u.deepvqe_s.mic_re = x.plane[0]; fin.u.deepvqe_s.mic_im = x.plane[1];
            fin.u.deepvqe_s.far_re = x.plane[2]; fin.u.deepvqe_s.far_im = x.plane[3];
        } else {
            fin.kind = AIAEC_INPUT_ALIGN_ULCNET;
            fin.u.align_ulcnet.err_re = x.plane[0]; fin.u.align_ulcnet.err_im = x.plane[1];
            fin.u.align_ulcnet.far_re = x.plane[2]; fin.u.align_ulcnet.far_im = x.plane[3];
            fin.u.align_ulcnet.events = x.ev; fin.u.align_ulcnet.event_count = x.n;
        }
        fin.kind = PRODUCT_DV ? AIAEC_INPUT_ALIGN_ULCNET : AIAEC_INPUT_DEEPVQE_S;
        CHECK(aiaec_preprocess_freq(x.fc, &fin, &w) == AIAEC_ERR_INPUT_KIND, "order: wrong kind with a NaN plane is not INPUT_KIND");
        fin.kind = PRODUCT_DV ? AIAEC_INPUT_DEEPVQE_S : AIAEC_INPUT_ALIGN_ULCNET; fin.frames = x.n + 1;
        CHECK(aiaec_preprocess_freq(x.fc, &fin, &w) == AIAEC_ERR_INPUT_SIZE, "order: wrong frames with a NaN plane is not INPUT_SIZE");
        fin.frames = x.n;
        if (PRODUCT_DV) fin.u.deepvqe_s.far_im = NULL; else fin.u.align_ulcnet.far_im = NULL;
        CHECK(aiaec_preprocess_freq(x.fc, &fin, &w) == AIAEC_ERR_NULL, "order: a NULL plane with a NaN plane is not ERR_NULL");
        if (PRODUCT_DV) fin.u.deepvqe_s.far_im = x.plane[3]; else fin.u.align_ulcnet.far_im = x.plane[3];
        CHECK(aiaec_preprocess_freq(x.fc, &fin, &w) == AIAEC_ERR_NONFINITE_INPUT, "order: the NaN plane alone is not refused");
        x.plane[0][0] = 0.1f;
        CHECK(aiaec_preprocess_freq(x.fc, &fin, &w) == AIAEC_OK, "order: clean preprocess");
        x.plane[0][0] = NAN;
        CHECK(aiaec_preprocess_freq(x.fc, &fin, &w) == AIAEC_ERR_SEQUENCE, "order: a bad input while a block is open is not ERR_SEQUENCE");
        x.plane[0][0] = 0.1f;
        run_accel(&w, x.n);
        CHECK(aiaec_postprocess_freq(x.fc, STREAM_RUN_OK, out.re, out.im) == AIAEC_OK, "order: the open block did not close");
    }
    CHECK(strstr(aiaec_status_text(AIAEC_ERR_NONFINITE_INPUT), "non-finite") != NULL, "status text names the refusal");
    CHECK(AIAEC_ERR_NONFINITE_INPUT == -10, "AIAEC_ERR_NONFINITE_INPUT is %d, want -10", (int)AIAEC_ERR_NONFINITE_INPUT);
    inst_close(&x);
    printf("section_order: kind, size and NULL gates come first; a bad input while a block is open is a sequence error\n");
    return bad;
}

int main(void) {
    g_fail |= section_limit();
#if PRODUCT_DV
    g_fail |= section_overflow();
#endif
    g_fail |= section_refusal();
    g_fail |= section_order();
    printf("test_aiaec_nonfinite[%s]: %s\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
