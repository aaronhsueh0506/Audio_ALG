/* A NaN/Inf bin, or a finite bin too large for the feature power (|bin| * 1/sqrt(fft) > 2^63), in the
 * caller's spectra to the spectrum-in family. ainr_preprocess_freq() must refuse the whole block
 * (AINR_ERR_NONFINITE_INPUT) before any frame enters the stream: no block open, generation unchanged,
 * the feature normaliser and frame queue exactly as if the block had never been offered. Checked two
 * ways per injection, bit for bit:
 *
 *   skip   the refused block is never fed again; the windows handed to the accelerator and the
 *          delivered spectra afterwards equal a run that never saw the block.
 *   retry  the same block is re-offered clean right after the refusal; the whole run equals a
 *          clean run.
 *
 * Every window handed out and every delivered spectrum must also be finite. N = 1 and N = 8 (full
 * 16 kHz/256), 48 kHz/1024 N = 1 (full or lite by the archive variant). A second part drives the
 * DFN2Stream freq entry directly: its own guard returns -1 with the stream untouched. A third part
 * pins the limit: a bin at exactly 2^63 (model scale) is accepted, the next float up refused. Not built
 * under NO_STDIO. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "dfn2_grid.h"
#include "dfn2_stream.h"

#define BLOCKS    10
#define BAD_BLOCK 3

static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *inv = (float *)malloc(elems * sizeof(float));
    size_t i;
    int32_t hdr[2];
    FILE *fp;
    if (!fwd || !inv) { fprintf(stderr, "malloc fail\n"); exit(1); }
    for (i = 0; i < elems; ++i) {
        fwd[i] = (float)(i % 17) * 0.01f;
        inv[i] = (float)((i + 3) % 11) * 0.02f;
    }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd); free(inv);
}

static void *alloc_aligned(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
}

static int all_finite(const float *p, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (!isfinite(p[i])) return 0;
    return 1;
}

/* Deterministic audio-scale spectrum values, different for every (block, array, element). */
static void gen_block(float *buf, size_t n, int block, int array) {
    uint32_t x = 0x9e3779b9u * (uint32_t)(block * 4 + array + 1);
    size_t i;
    for (i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        buf[i] = ((float)(x >> 9) / 4194304.0f - 1.0f) * 0.5f;
    }
}

static void fill_heads(AinrWork *io, int blk) {
    int o;
    size_t i;
    for (o = 0; o < io->output_count; ++o)
        for (i = 0; i < io->outputs[o].elements; ++i)
            io->outputs[o].data[i] = (o == 0 ? 0.4f : 0.01f) + 0.01f * (float)((i + (size_t)blk) % 7);
}

typedef struct Trace {
    float *buf;
    size_t len, cap;
    unsigned int gens[BLOCKS];
    int blocks;
} Trace;

static void trace_add(Trace *t, const float *p, size_t n) {
    if (t->len + n > t->cap) { fprintf(stderr, "trace overflow\n"); exit(1); }
    memcpy(t->buf + t->len, p, n * sizeof(float));
    t->len += n;
}

typedef enum { INJ_NONE = 0, INJ_EST_NAN_FIRST, INJ_APP_INF_LAST, INJ_EST_INF_MID, INJ_SINGLE_NAN,
               INJ_EST_HUGE_FIRST, INJ_APP_HUGE_LAST, INJ_SINGLE_HUGE } Injection;
/* CLEAN: every block as is. OMIT: block BAD_BLOCK never offered. SKIP / RETRY: block BAD_BLOCK is
 * corrupt and must be refused (SKIP: not offered again; RETRY: offered clean right after). */
typedef enum { MODE_CLEAN = 0, MODE_OMIT, MODE_SKIP, MODE_RETRY } Mode;

static const char *inj_name(Injection j) {
    switch (j) {
    case INJ_EST_NAN_FIRST: return "est NaN, first frame";
    case INJ_APP_INF_LAST:  return "app Inf, last frame";
    case INJ_EST_INF_MID:   return "est Inf, middle frame";
    case INJ_SINGLE_NAN:    return "E==P NaN";
    case INJ_EST_HUGE_FIRST: return "est finite 1e30, first frame";
    case INJ_APP_HUGE_LAST: return "app finite -1e30, last frame";
    case INJ_SINGLE_HUGE:   return "E==P finite 3e38";
    default:                return "none";
    }
}

static int run_stream(const AinrConfig *cfg, const char *erb, int n, int n_bins, Mode mode,
                      Injection inj, Trace *t, const char *label) {
    StreamMemReq req;
    AinrInfo info;
    void *pool;
    AinrFreqCtx *ctx;
    size_t blk_n = (size_t)n * (size_t)n_bins;
    int single = (inj == INJ_SINGLE_NAN || inj == INJ_SINGLE_HUGE);
    float *a[4], *b[4], *out_re, *out_im;
    int blk, k;

    for (k = 0; k < 4; ++k) {
        a[k] = (float *)malloc(blk_n * sizeof(float));
        b[k] = (float *)malloc(blk_n * sizeof(float));
    }
    out_re = (float *)malloc(blk_n * sizeof(float));
    out_im = (float *)malloc(blk_n * sizeof(float));
    if (ainr_get_mem_size_freq(cfg, &req) != AINR_OK) { fprintf(stderr, "%s: mem size fail\n", label); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ctx = NULL;
    if (ainr_init_freq_from_erb_file(cfg, erb, pool, (size_t)req.bytes, &ctx, &info) != AINR_OK || !ctx) { fprintf(stderr, "%s: init fail\n", label); return 1; }
    t->len = 0;
    t->blocks = 0;

    for (blk = 0; blk < BLOCKS; ++blk) {
        int bad_blk = (blk == BAD_BLOCK);
        int pass, npass = (bad_blk && mode == MODE_RETRY) ? 2 : 1;

        if (bad_blk && mode == MODE_OMIT) continue;
        for (k = 0; k < 4; ++k) gen_block(a[k], blk_n, blk, k);   /* est_re, est_im, app_re, app_im */

        for (pass = 0; pass < npass; ++pass) {
            AinrWork io;
            unsigned int gen_before;
            int corrupt = bad_blk && pass == 0 && (mode == MODE_SKIP || mode == MODE_RETRY);
            int rc, o;
            const float *ar, *ai;

            for (k = 0; k < 4; ++k) memcpy(b[k], a[k], blk_n * sizeof(float));
            if (corrupt) {
                switch (inj) {
                case INJ_EST_NAN_FIRST: b[0][3] = NAN; break;
                case INJ_APP_INF_LAST:  b[3][blk_n - 1] = INFINITY; break;
                case INJ_EST_INF_MID:   b[1][(size_t)(n / 2) * (size_t)n_bins + (size_t)n_bins - 1] = -INFINITY; break;
                case INJ_SINGLE_NAN:    b[0][blk_n - 2] = NAN; break;
                case INJ_EST_HUGE_FIRST: b[0][3] = 1e30f; break;
                case INJ_APP_HUGE_LAST: b[3][blk_n - 1] = -1e30f; break;
                case INJ_SINGLE_HUGE:   b[0][blk_n - 2] = 3e38f; break;
                default: break;
                }
            }
            ar = single ? b[0] : b[2];
            ai = single ? b[1] : b[3];

            gen_before = ainr_state_generation_freq(ctx);
            rc = ainr_preprocess_freq(ctx, b[0], b[1], ar, ai, &io);
            if (corrupt) {
                if (rc != AINR_ERR_NONFINITE_INPUT) {
                    fprintf(stderr, "%s [%s]: block %d returned %d, want AINR_ERR_NONFINITE_INPUT (%d)\n",
                            label, inj_name(inj), blk, rc, AINR_ERR_NONFINITE_INPUT);
                    return 1;
                }
                if (ainr_state_generation_freq(ctx) != gen_before) {
                    fprintf(stderr, "%s [%s]: refusal moved the state generation\n", label, inj_name(inj));
                    return 1;
                }
                if (ainr_postprocess_freq(ctx, 0, out_re, out_im) != AINR_ERR_SEQUENCE) {
                    fprintf(stderr, "%s [%s]: a block is open after the refusal\n", label, inj_name(inj));
                    return 1;
                }
                continue;
            }
            if (rc != AINR_OK) { fprintf(stderr, "%s [%s]: block %d preprocess rc=%d\n", label, inj_name(inj), blk, rc); return 1; }
            for (o = 0; o < io.input_count; ++o) {
                if (!all_finite(io.inputs[o].data, io.inputs[o].elements)) {
                    fprintf(stderr, "%s [%s]: block %d input window %d is not finite\n",
                            label, inj_name(inj), blk, o);
                    return 1;
                }
                trace_add(t, io.inputs[o].data, io.inputs[o].elements);
            }
            fill_heads(&io, blk);
            if (ainr_postprocess_freq(ctx, 0, out_re, out_im) != AINR_OK) {
                fprintf(stderr, "%s [%s]: block %d postprocess failed\n", label, inj_name(inj), blk);
                return 1;
            }
            if (!all_finite(out_re, blk_n) || !all_finite(out_im, blk_n)) {
                fprintf(stderr, "%s [%s]: block %d delivered a non-finite spectrum\n", label, inj_name(inj), blk);
                return 1;
            }
            trace_add(t, out_re, blk_n);
            trace_add(t, out_im, blk_n);
            t->gens[t->blocks++] = io.state_generation;
        }
    }
    free(pool);
    for (k = 0; k < 4; ++k) { free(a[k]); free(b[k]); }
    free(out_re); free(out_im);
    return 0;
}

static int traces_equal(const Trace *x, const Trace *y) {
    return x->blocks == y->blocks && x->len == y->len &&
           memcmp(x->buf, y->buf, x->len * sizeof(float)) == 0 &&
           memcmp(x->gens, y->gens, (size_t)x->blocks * sizeof(unsigned int)) == 0;
}

static int run_case(int sr, int fft, int n_erb, int n, const char *label) {
    AinrConfig cfg;
    int n_bins = fft / 2 + 1, hop = fft / 2, bad = 0, j;
    char erb[160];
    Trace clean, omit, got;
    size_t cap = (size_t)BLOCKS * (size_t)n * (size_t)n_bins * 8u;

    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = n * hop;
    snprintf(erb, sizeof(erb), "/tmp/ainr_nonfinite_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, n_bins, n_erb);
    clean.buf = (float *)malloc(cap * sizeof(float)); clean.cap = cap;
    omit.buf = (float *)malloc(cap * sizeof(float));  omit.cap = cap;
    got.buf = (float *)malloc(cap * sizeof(float));   got.cap = cap;

    if (strcmp(ainr_status_text(AINR_ERR_NONFINITE_INPUT), "unknown ainr status") == 0) {
        fprintf(stderr, "%s: AINR_ERR_NONFINITE_INPUT has no status text\n", label); bad = 1;
    }
    for (j = INJ_EST_NAN_FIRST; j <= INJ_SINGLE_HUGE; ++j) {
        Injection inj = (Injection)j;
        if (run_stream(&cfg, erb, n, n_bins, MODE_CLEAN, inj, &clean, label) ||
            run_stream(&cfg, erb, n, n_bins, MODE_OMIT, inj, &omit, label) ||
            run_stream(&cfg, erb, n, n_bins, MODE_SKIP, inj, &got, label)) return 1;
        if (!traces_equal(&got, &omit)) {
            fprintf(stderr, "%s [%s]: skip run differs from a run that never saw the block\n", label, inj_name(inj));
            bad = 1;
        } else {
            printf("%s [%s]: refused, state untouched (skip == never offered)\n", label, inj_name(inj));
        }
        if (run_stream(&cfg, erb, n, n_bins, MODE_RETRY, inj, &got, label)) return 1;
        if (!traces_equal(&got, &clean)) {
            fprintf(stderr, "%s [%s]: retry run differs from the clean run\n", label, inj_name(inj));
            bad = 1;
        } else {
            printf("%s [%s]: refused, retried clean == clean run\n", label, inj_name(inj));
        }
    }
    unlink(erb);
    free(clean.buf); free(omit.buf); free(got.buf);
    return bad;
}

/* The limit itself, through ainr_preprocess_freq(): fft 256 and 1024 scale by an exact power of two
 * (1/16, 1/32), so a bin of 2^63 * sqrt(fft) lands on exactly 2^63 at model scale. */
static int run_boundary(int sr, int fft, int n_erb, const char *label) {
    AinrConfig cfg;
    StreamMemReq req;
    AinrInfo info;
    AinrFreqCtx *ctx = NULL;
    AinrWork io;
    int n_bins = fft / 2 + 1, bad = 0, a, o;
    size_t bin = 5, i;
    float scale = dfn2_stft_scale(fft), edge = ldexpf(1.0f, 63) / scale, over = nextafterf(edge, INFINITY);
    float *x[4], *out_re = (float *)malloc((size_t)n_bins * sizeof(float)),
          *out_im = (float *)malloc((size_t)n_bins * sizeof(float));
    char erb[160];
    void *pool;
    unsigned int gen;

    if (edge * scale != ldexpf(1.0f, 63)) { fprintf(stderr, "%s: scale is not a power of two\n", label); return 1; }
    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = fft / 2;
    snprintf(erb, sizeof(erb), "/tmp/ainr_nonfinite_edge_%d_%d.bin", fft, (int)getpid());
    write_synthetic_erb(erb, n_bins, n_erb);
    if (ainr_get_mem_size_freq(&cfg, &req) != AINR_OK) { fprintf(stderr, "%s: mem size fail\n", label); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init_freq_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &ctx, &info) != AINR_OK || !ctx) {
        fprintf(stderr, "%s: init fail\n", label); return 1;
    }
    for (a = 0; a < 4; ++a) { x[a] = (float *)malloc((size_t)n_bins * sizeof(float)); gen_block(x[a], (size_t)n_bins, 0, a); }
    gen = ainr_state_generation_freq(ctx);

    /* one array at a time one float above the limit (the others at the limit, signs mixed): refused */
    for (a = 0; a < 4; ++a) {
        int b;
        for (b = 0; b < 4; ++b) x[b][bin] = (b & 1) ? -edge : edge;
        x[a][bin] = (a & 1) ? -over : over;
        if (ainr_preprocess_freq(ctx, x[0], x[1], x[2], x[3], &io) != AINR_ERR_NONFINITE_INPUT ||
            ainr_state_generation_freq(ctx) != gen ||
            ainr_postprocess_freq(ctx, 0, out_re, out_im) != AINR_ERR_SEQUENCE) {
            fprintf(stderr, "%s: array %d one float above the limit was not refused cleanly\n", label, a);
            bad = 1;
        }
    }
    /* all four at the limit: accepted, every window and delivered value finite */
    for (a = 0; a < 4; ++a) x[a][bin] = (a & 1) ? -edge : edge;
    if (ainr_preprocess_freq(ctx, x[0], x[1], x[2], x[3], &io) != AINR_OK) {
        fprintf(stderr, "%s: a bin at the limit was refused\n", label); return 1;
    }
    for (o = 0; o < io.input_count; ++o)
        if (!all_finite(io.inputs[o].data, io.inputs[o].elements)) {
            fprintf(stderr, "%s: window %d not finite at the limit\n", label, o); bad = 1;
        }
    fill_heads(&io, 0);
    if (ainr_postprocess_freq(ctx, 0, out_re, out_im) != AINR_OK ||
        !all_finite(out_re, (size_t)n_bins) || !all_finite(out_im, (size_t)n_bins)) {
        fprintf(stderr, "%s: postprocess at the limit failed or delivered a non-finite spectrum\n", label); bad = 1;
    }
    for (i = 0; i < 4; ++i) free(x[i]);
    free(out_re); free(out_im); free(pool);
    unlink(erb);
    if (!bad) printf("%s: 2^63 accepted, next float refused, state untouched\n", label);
    return bad;
}

/* DFN2Stream freq entry: the stream's own guard. */
static int run_stream_guard(const char *label) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    DFN2StreamConfig scfg;
    DFN2StreamMemReq req;
    DFN2Stream *s[2];
    float *x[4], *out_re, *out_im, *tr[2];
    size_t nb, tl[2] = {0, 0};
    char erb[160];
    int blk, w, bad = 0;

    memset(&spec, 0, sizeof(spec));
#ifdef DFN2_MASK_ONLY
    spec.sample_rate = 48000; spec.n_fft = 1024; spec.n_erb = 32; spec.df_bins = 96;
    spec.df_order = 0; spec.mask_lookahead = 1; spec.df_lookahead = 0;
#else
    spec.sample_rate = 16000; spec.n_fft = 256; spec.n_erb = 24; spec.df_bins = 96;
    spec.df_order = 5; spec.mask_lookahead = 0; spec.df_lookahead = 0;
#endif
    spec.frames_per_call = 1;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "%s: grid fail\n", label); return 1; }
    nb = (size_t)grid.n_bins;
    snprintf(erb, sizeof(erb), "/tmp/ainr_nonfinite_stream_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb, grid.n_bins, grid.n_erb);
    memset(&scfg, 0, sizeof(scfg));
    scfg.grid = grid;
    if (dfn2_stream_get_mem_size(&scfg, &req) != 0) { fprintf(stderr, "%s: mem size fail\n", label); return 1; }
    for (w = 0; w < 2; ++w) {
        void *pool = alloc_aligned((size_t)req.bytes, req.alignment);
        s[w] = dfn2_stream_init_from_erb_file(&scfg, erb, pool, (size_t)req.bytes);
        if (!s[w]) { fprintf(stderr, "%s: init fail\n", label); return 1; }
        tr[w] = (float *)malloc((size_t)BLOCKS * nb * 8u * sizeof(float));
    }
    for (w = 0; w < 4; ++w) x[w] = (float *)malloc(nb * sizeof(float));
    out_re = (float *)malloc(nb * sizeof(float));
    out_im = (float *)malloc(nb * sizeof(float));

    for (blk = 0; blk < BLOCKS; ++blk) {
        for (w = 0; w < 2; ++w) {   /* w = 0: gets the corrupt offers; w = 1: never does */
            DFN2HopIO io;
            int k, v;
            for (k = 0; k < 4; ++k) gen_block(x[k], nb, blk, k);
            if (w == 0 && blk == BAD_BLOCK) {
                float *y[4];
                for (k = 0; k < 4; ++k) { y[k] = (float *)malloc(nb * sizeof(float)); memcpy(y[k], x[k], nb * sizeof(float)); }
                for (v = 0; v < 8; ++v) {   /* 0..3: NaN/Inf, 4..7: finite, above the limit */
                    size_t at = nb - 1 - (size_t)(v & 3);
                    y[v & 3][at] = v >= 4 ? ((v & 1) ? -1e25f : 1e25f) : ((v & 1) ? INFINITY : NAN);
                    if (dfn2_stream_pre_process_freq_dual(s[0], y[0], y[1], y[2], y[3], &io) != -1) {
                        fprintf(stderr, "%s: array %d with a %s bin was accepted\n", label, v & 3,
                                v >= 4 ? "too large" : "non-finite");
                        bad = 1;
                    }
                    y[v & 3][at] = x[v & 3][at];
                }
                for (k = 0; k < 4; ++k) free(y[k]);
            }
            if (dfn2_stream_pre_process_freq_dual(s[w], x[0], x[1], x[2], x[3], &io) != 1) {
                fprintf(stderr, "%s: block %d preprocess failed (stream %d)\n", label, blk, w);
                return 1;
            }
            memcpy(tr[w] + tl[w], io.inputs.erb_window, io.inputs.erb_window_elements * sizeof(float));
            tl[w] += io.inputs.erb_window_elements;
            memcpy(tr[w] + tl[w], io.inputs.spec_window, io.inputs.spec_window_elements * sizeof(float));
            tl[w] += io.inputs.spec_window_elements;
            if (!all_finite(io.inputs.erb_window, io.inputs.erb_window_elements) ||
                !all_finite(io.inputs.spec_window, io.inputs.spec_window_elements)) {
                fprintf(stderr, "%s: block %d window not finite (stream %d)\n", label, blk, w);
                bad = 1;
            }
            {
                size_t i;
                for (i = 0; i < io.outputs.erb_mask_elements; ++i) io.outputs.erb_mask[i] = 0.5f;
#ifndef DFN2_MASK_ONLY
                for (i = 0; i < io.outputs.coefs_elements; ++i) io.outputs.coefs[i] = 0.01f;
                for (i = 0; i < io.outputs.alpha_elements; ++i) io.outputs.alpha[i] = 0.3f;
#endif
            }
            if (dfn2_stream_post_process_freq(s[w], 0, out_re, out_im, NULL) != 0) return 1;
            memcpy(tr[w] + tl[w], out_re, nb * sizeof(float)); tl[w] += nb;
            memcpy(tr[w] + tl[w], out_im, nb * sizeof(float)); tl[w] += nb;
        }
    }
    if (tl[0] != tl[1] || memcmp(tr[0], tr[1], tl[0] * sizeof(float)) != 0) {
        fprintf(stderr, "%s: refused offers changed the stream's later windows/outputs\n", label);
        bad = 1;
    } else if (!bad) {
        printf("%s: DFN2Stream freq guard OK (8 refused offers, state untouched)\n", label);
    }
    {   /* the limit: one float above 2^63 refused, 2^63 accepted, on a fresh stream */
        void *pool = alloc_aligned((size_t)req.bytes, req.alignment);
        DFN2Stream *f = dfn2_stream_init_from_erb_file(&scfg, erb, pool, (size_t)req.bytes);
        float lim = ldexpf(1.0f, 63), over = nextafterf(lim, INFINITY), *z = (float *)calloc(nb, sizeof(float));
        DFN2HopIO io;
        z[2] = lim;
        if (!f) { fprintf(stderr, "%s: init fail\n", label); return 1; }
        z[3] = over;
        if (dfn2_stream_pre_process_freq_dual(f, z, z, z, z, &io) != -1) {
            fprintf(stderr, "%s: a bin one float above 2^63 was accepted\n", label); bad = 1;
        }
        z[3] = -lim;
        if (dfn2_stream_pre_process_freq_dual(f, z, z, z, z, &io) != 1 ||
            dfn2_stream_post_process_freq(f, 1, out_re, out_im, NULL) != 0) {
            fprintf(stderr, "%s: a bin at +-2^63 was refused\n", label); bad = 1;
        }
        free(z); free(pool);
    }
    unlink(erb);
    return bad;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, "nonfinite[lite 48k/1024 N=1]");
    rc |= run_boundary(48000, 1024, 32, "limit[lite 48k/1024]");
    rc |= run_stream_guard("nonfinite_stream[lite]");
#else
    rc |= run_case(16000, 256, 24, 1, "nonfinite[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, "nonfinite[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, "nonfinite[full 48k/1024 N=1]");
    rc |= run_boundary(16000, 256, 24, "limit[full 16k/256]");
    rc |= run_boundary(48000, 1024, 32, "limit[full 48k/1024]");
    rc |= run_stream_guard("nonfinite_stream[full]");
#endif
    if (rc) { printf("test_ainr_freq_nonfinite: FAIL\n"); return 1; }
    printf("test_ainr_freq_nonfinite: PASS\n");
    return 0;
}
