/* Host coverage for aiaec_preprocess() / aiaec_postprocess() through the real public API,
 * end to end, for the model the build binds (aiaec_product.h). The Makefile compiles this file once
 * per product (-DAIAEC_PRODUCT_MODEL_ID=...) with -DAIAEC_TEST_CATALOG, which substitutes placeholder
 * graph paths for the rows below so the gated API constructs; the production catalog (no delivered
 * graph for most rows) is covered by test_aiaec_catalog.c.
 *
 *   DeepVQE-S (N=1 and N=8 at 16k/256, N=1 at 16k/512): the block path against deepvqe_stream_*
 *     driven directly with the same stash/drain loop, bit-exact PCM and identical accelerator-side
 *     tensors; batching against an independent N=1 stream; fail-closed; identity-accelerator delay.
 *   Align-ULCNet (N=1): normal, realignment-bypass and forced-failure hops against ulcnet_stream_*
 *     driven directly; infer_status ignored on a bypass hop; identity delay.
 *   Both: live work-view shapes, kind/size rejection before any state write, reset, and the pool
 *     size / alignment gates of init. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"
#include "deepvqe_stream.h"
#include "ulcnet_stream.h"

#define PRODUCT_DV (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S)
#define PRODUCT_AL (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_ALIGN_ULCNET)

static int g_fail = 0;
#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } while (0)
#define CHECK(cond, ...) do { if (!(cond)) FAIL(__VA_ARGS__); } while (0)

static AiaecConfig make_cfg(int fft, int n) {
    AiaecConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sr = 16000; cfg.fft = fft; cfg.hop_in = (fft / 2) * n;
    return cfg;
}

static AiaecCtx *make_ctx_info(const AiaecConfig *cfg, void **pool_out, const char *who, AiaecInfo *info) {
    StreamMemReq req;
    AiaecCtx *ctx = NULL;
    int rc;
    *pool_out = NULL;
    if (aiaec_get_mem_size(cfg, &req) != AIAEC_OK) {
        FAIL("%s: aiaec_get_mem_size failed -- is AIAEC_TEST_CATALOG wired?", who);
        return NULL;
    }
    if (posix_memalign(pool_out, (size_t)req.alignment, (size_t)req.bytes) != 0) {
        FAIL("%s: pool alloc failed", who);
        return NULL;
    }
    rc = aiaec_init(cfg, *pool_out, (size_t)req.bytes, &ctx, info);
    if (rc != AIAEC_OK || !ctx) FAIL("%s: aiaec_init failed rc=%d (%s)", who, rc, aiaec_status_text(rc));
    return ctx;
}

static AiaecCtx *make_ctx(const AiaecConfig *cfg, void **pool_out, const char *who) {
    AiaecInfo info;
    return make_ctx_info(cfg, pool_out, who, &info);
}

/* Allows 2 LSB of int16 drift: the identity-delay sections compare the pipeline's own float->int16
 * conversion with this test's, which is not guaranteed to round identically; the shift being
 * measured is a whole-hop quantity, far coarser. */
static int pcm16_close(const int16_t *a, const int16_t *b, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        int diff = (int)a[i] - (int)b[i];
        if (diff < -2 || diff > 2) return 0;
    }
    return 1;
}

/* Deterministic fakes keyed by a global hop/call counter, for same-N, same-sequence A/B comparisons
 * (two contexts fed bit-identical inputs and outputs in the same order must agree bit for bit). */
#if PRODUCT_DV
static void fake_taps(long long ghop, float *dst, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i)
        dst[i] = 0.05f * (float)(((ghop + (long long)i) % 37) - 18);
}
#endif

#if PRODUCT_AL
static void fake_mask(long long ghop, float *dst, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i)
        dst[i] = 0.1f * (float)(((ghop + (long long)i) % 23) - 11);
}
#endif

#if PRODUCT_DV
/* A pure function of the input mic/far spectra, frame by frame: required wherever two drivers run
 * different frames_per_call (frame f of an N=8 batch and the N=1 hop with the same content must
 * produce the same taps). mic/far are [2][N][bins]: real plane, then imaginary plane. */
static void content_taps(const AiaecWork *w, int n, float *taps) {
    const size_t bins = w->inputs[0].w;
    const int tapw = 18;
    const float *mic_re = w->inputs[0].data, *mic_im = mic_re + (size_t)n * bins;
    const float *far_re = w->inputs[1].data, *far_im = far_re + (size_t)n * bins;
    size_t f, b;
    int t;
    for (f = 0; f < (size_t)n; ++f)
        for (b = 0; b < bins; ++b) {
            float v = 0.01f * (mic_re[f * bins + b] + mic_im[f * bins + b] + far_re[f * bins + b] + far_im[f * bins + b]);
            for (t = 0; t < tapw; ++t) taps[(f * bins + b) * (size_t)tapw + (size_t)t] = v + 0.001f * (float)t;
        }
}

static void dv_input(AiaecInput *in, const int16_t *mic, const int16_t *far, int samples) {
    memset(in, 0, sizeof(*in));
    in->kind = AIAEC_INPUT_DEEPVQE_S;
    in->samples = samples;
    in->u.deepvqe_s.mic_in = mic;
    in->u.deepvqe_s.far_in = far;
}

static int dv_grid(int fft, int n, DeepVqeGrid *grid) {
    DeepVqeGridSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000; spec.n_fft = fft;
    spec.delay_depth = (int)ceilf(0.5f * 16000.0f / (float)(fft / 2));
    spec.frames_per_call = n;
    return deepvqe_grid_make(grid, &spec) == DEEPVQE_GRID_OK ? 0 : -1;
}

static DeepVqeStream *dv_stream(const DeepVqeGrid *grid) {
    DeepVqeStreamConfig scfg;
    DeepVqeStreamMemReq sreq;
    void *pool = NULL;
    memset(&scfg, 0, sizeof(scfg));
    scfg.grid = *grid;
    if (deepvqe_stream_get_mem_size(&scfg, &sreq) != 0) return NULL;
    if (posix_memalign(&pool, (size_t)sreq.alignment, (size_t)sreq.bytes) != 0) return NULL;
    return deepvqe_stream_init_ex(&scfg, pool, (size_t)sreq.bytes, &sreq);
}

/* The block path against deepvqe_stream_* driven directly with the same stash/drain loop: PCM bit
 * for bit, and the tensors the accelerator sees (element counts, input spectra) identical. The 8x
 * amplitude keeps the compare above the int16 rounding threshold, where an earlier cold-start bug
 * passed with zero mismatches. */
static int section_deepvqe_block_vs_stream(int n, int fft) {
    AiaecConfig cfg = make_cfg(fft, n);
    void *pool = NULL;
    AiaecCtx *ctx = make_ctx(&cfg, &pool, "block_vs_stream");
    DeepVqeGrid grid;
    DeepVqeStream *ref;
    const int hop = fft / 2, blocks = 6;
    int16_t *mic_in, *far_in, *pcm_blk, *pcm_ref, *stash;
    long long ghop = 0;
    int b, k, i, bad = 0;

    if (!ctx) return 1;
    if (dv_grid(fft, n, &grid) != 0 || !(ref = dv_stream(&grid))) { FAIL("block_vs_stream: reference stream"); return 1; }
    mic_in = calloc((size_t)hop * (size_t)n, sizeof(int16_t));
    far_in = calloc((size_t)hop * (size_t)n, sizeof(int16_t));
    pcm_blk = calloc((size_t)hop * (size_t)n, sizeof(int16_t));
    pcm_ref = calloc((size_t)hop * (size_t)n, sizeof(int16_t));
    stash = calloc((size_t)hop * (size_t)n, sizeof(int16_t));

    for (b = 0; b < blocks; ++b) {
        AiaecInput input;
        AiaecWork work;
        DeepVqeHopIO hio;
        int rc;

        for (k = 0; k < hop * n; ++k) {
            mic_in[k] = (int16_t)((((long long)k * 5 + ghop * 3) % 97 - 48) * 8);
            far_in[k] = (int16_t)((((long long)k * 11 + ghop * 7) % 89 - 44) * 8);
        }

        dv_input(&input, mic_in, far_in, hop * n);
        rc = aiaec_preprocess(ctx, &input, &work);
        CHECK(rc == AIAEC_OK && work.action == AIAEC_SUBMIT, "N=%d fft=%d block %d: preprocess rc=%d action=%d", n, fft, b, rc, (int)work.action);
        CHECK(work.input_count == 2 && work.output_count == 1, "N=%d block %d: counts %d/%d", n, b, work.input_count, work.output_count);

        for (i = 0; i < n - 1; ++i) {
            CHECK(deepvqe_stream_pre_process(ref, mic_in + (size_t)i * hop, far_in + (size_t)i * hop, &hio) == 0, "ref pre %d", i);
            CHECK(deepvqe_stream_post_process(ref, 0, stash + (size_t)i * hop) >= 0, "ref post %d", i);
        }
        CHECK(deepvqe_stream_pre_process(ref, mic_in + (size_t)(n - 1) * hop, far_in + (size_t)(n - 1) * hop, &hio) == 1, "ref last pre");
        CHECK(work.inputs[0].elements == hio.inputs.spectrum_ri_elements && work.inputs[1].elements == hio.inputs.spectrum_ri_elements &&
              work.outputs[0].elements == hio.outputs.taps_elements, "N=%d block %d: tensor element counts differ from deepvqe_stream", n, b);
        CHECK(memcmp(work.inputs[0].data, hio.inputs.mic, work.inputs[0].elements * sizeof(float)) == 0 &&
              memcmp(work.inputs[1].data, hio.inputs.far, work.inputs[1].elements * sizeof(float)) == 0,
              "N=%d block %d: accelerator input spectra differ from deepvqe_stream", n, b);
        CHECK(work.state_generation == hio.state_generation, "N=%d block %d: state_generation", n, b);

        fake_taps(ghop, work.outputs[0].data, work.outputs[0].elements);
        fake_taps(ghop, hio.outputs.taps, hio.outputs.taps_elements);
        CHECK(aiaec_postprocess(ctx, 0, pcm_blk) == AIAEC_OK, "N=%d block %d: postprocess", n, b);
        CHECK(deepvqe_stream_post_process(ref, 0, pcm_ref + (size_t)(n - 1) * hop) >= 0, "ref last post");
        memcpy(pcm_ref, stash, (size_t)(n - 1) * (size_t)hop * sizeof(int16_t));

        if (memcmp(pcm_ref, pcm_blk, (size_t)hop * (size_t)n * sizeof(int16_t)) != 0) {
            FAIL("N=%d fft=%d block %d: block pcm_out differs from deepvqe_stream_* driven directly", n, fft, b);
            bad = 1;
        }
        ghop += n;
    }
    printf("section_deepvqe_block_vs_stream(fft=%d N=%d): %d blocks bit-exact vs deepvqe_stream_*\n", fft, n, blocks);
    free(mic_in); free(far_in); free(pcm_blk); free(pcm_ref); free(stash); free(pool);
    return bad;
}

/* N=8 batching against an independent N=1 stream fed the same per-hop PCM. The batched side is
 * (N-1)*hop samples later than the reference (both carry the same base hop of STFT/ISTFT framing,
 * which cancels), and block boundaries of the two do not line up, so one global concatenated buffer
 * per side is compared with that shift applied. */
static int section_batching_vs_n1(int n) {
    AiaecConfig cfg = make_cfg(256, n);
    void *pool = NULL;
    AiaecCtx *ctx = make_ctx(&cfg, &pool, "batching");
    DeepVqeGrid rgrid;
    DeepVqeStream *ref;
    const int hop = 128, blocks = 6;
    const size_t total = (size_t)blocks * (size_t)n * (size_t)hop, shift = (size_t)(n - 1) * hop;
    int16_t *mic_in, *far_in, *full_blk, *full_ref;
    float *taps_ref;
    long long ghop = 0;
    int b, k, bad = 0;

    if (!ctx) return 1;
    if (dv_grid(256, 1, &rgrid) != 0 || !(ref = dv_stream(&rgrid))) { FAIL("batching: reference stream"); return 1; }
    mic_in = calloc((size_t)hop * (size_t)n, sizeof(int16_t));
    far_in = calloc((size_t)hop * (size_t)n, sizeof(int16_t));
    full_blk = calloc(total, sizeof(int16_t));
    full_ref = calloc(total, sizeof(int16_t));
    taps_ref = malloc(deepvqe_prepost_taps_elements(&rgrid) * sizeof(float));

    for (b = 0; b < blocks; ++b) {
        AiaecInput input;
        AiaecWork work;
        for (k = 0; k < hop * n; ++k) {
            mic_in[k] = (int16_t)((((long long)k * 5 + ghop * 3) % 97 - 48) * 8);
            far_in[k] = (int16_t)((((long long)k * 11 + ghop * 7) % 89 - 44) * 8);
        }
        dv_input(&input, mic_in, far_in, hop * n);
        CHECK(aiaec_preprocess(ctx, &input, &work) == AIAEC_OK, "batching block %d: preprocess", b);
        content_taps(&work, n, work.outputs[0].data);
        CHECK(aiaec_postprocess(ctx, 0, full_blk + (size_t)b * n * hop) == AIAEC_OK, "batching block %d: postprocess", b);

        for (k = 0; k < n; ++k) {
            DeepVqeHopIO io;
            AiaecWork one;
            CHECK(deepvqe_stream_pre_process(ref, mic_in + (size_t)k * hop, far_in + (size_t)k * hop, &io) == 1, "ref pre");
            memset(&one, 0, sizeof(one));
            one.inputs[0].data = io.inputs.mic; one.inputs[0].w = (unsigned)rgrid.n_bins;
            one.inputs[1].data = io.inputs.far; one.inputs[1].w = (unsigned)rgrid.n_bins;
            content_taps(&one, 1, taps_ref);
            memcpy(io.outputs.taps, taps_ref, io.outputs.taps_elements * sizeof(float));
            deepvqe_stream_post_process(ref, 0, full_ref + ((size_t)b * n + (size_t)k) * hop);
        }
        ghop += n;
    }
    if (memcmp(full_blk + shift, full_ref, (total - shift) * sizeof(int16_t)) != 0) {
        size_t s;
        for (s = 0; s < total - shift; ++s)
            if (full_blk[shift + s] != full_ref[s]) { FAIL("N=%d: batched PCM diverges from the N=1 stream at sample %zu", n, s); break; }
        bad = 1;
    }
    printf("section_batching_vs_n1(N=%d): %zu samples compared at shift %zu, %s\n", n, total - shift, shift, bad ? "DIVERGED" : "bit-exact");
    free(mic_in); free(far_in); free(full_blk); free(full_ref); free(taps_ref); free(pool);
    return bad;
}

/* CCM taps forced to (delay 0, centre frequency tap) = 1 + 0j: the accelerator is the identity on the mic. */
static void fill_identity_taps(float *taps, size_t elements, int n) {
    const size_t tapn = 9, bins = elements / (tapn * 2u * (size_t)n);
    size_t f, b;
    memset(taps, 0, elements * sizeof(float));
    for (f = 0; f < (size_t)n; ++f)
        for (b = 0; b < bins; ++b) taps[(f * bins + b) * tapn * 2u + 1u * 2u] = 1.0f;
}


typedef enum FailKind {
    FAIL_STATUS = 0,       /* the run reports infer_status != 0 and its taps are valid and finite */
    FAIL_NONFINITE = 1,    /* infer_status 0, the taps are not finite (the library refuses the commit) */
    FAIL_BOTH = 2          /* both */
} FailKind;

static int all_zero16(const int16_t *x, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (x[i] != 0) return 0;
    return 1;
}

static long long energy16(const int16_t *x, size_t n) {
    long long e = 0;
    size_t i;
    for (i = 0; i < n; ++i) e += (long long)x[i] * x[i];
    return e;
}

/* A failed block contributes silence, never the mic: with an identity accelerator every block returns the previous
 * block's input (delay-aligned), except the samples the failed job owns. The failed job's N frames are zero spectra; in
 * the PCM family that is the last hop of the failing block (half of it still the previous frame's overlap) and the
 * first N - 1 hops of the next block, which are exactly zero; the next block's last hop is again half. Everything else
 * is the delayed mic. The failure is made unmistakable: valid finite taps with infer_status != 0 (the library must act on
 * the status), and non-finite taps with infer_status 0 (the library must refuse them), and both. */
static int section_fail_closed_one(FailKind kind) {
    enum { N = 8, HOP = 128, BS = N * HOP, BLOCKS = 6, FAILED = 2 };
    AiaecConfig cfg = make_cfg(256, N);
    void *pool = NULL;
    AiaecCtx *ctx = make_ctx(&cfg, &pool, "fail_closed");
    int16_t *mic_in, *far_zero, *pcm_out;
    unsigned int gen_seen[BLOCKS];
    int b, k, h, bad = g_fail;

    if (!ctx) return 1;
    mic_in = calloc((size_t)BS * BLOCKS, sizeof(int16_t));
    far_zero = calloc((size_t)BS, sizeof(int16_t));
    pcm_out = calloc((size_t)BS * BLOCKS, sizeof(int16_t));
    for (k = 0; k < BS * BLOCKS; ++k) mic_in[k] = (int16_t)(((k * 977) % 20001) - 10000);
    for (b = 0; b < BLOCKS; ++b) {
        AiaecInput input;
        AiaecWork work;
        const int fail = b == FAILED;
        dv_input(&input, mic_in + (size_t)b * BS, far_zero, BS);
        CHECK(aiaec_preprocess(ctx, &input, &work) == AIAEC_OK && work.action == AIAEC_SUBMIT, "fail_closed block %d: preprocess", b);
        gen_seen[b] = work.state_generation;
        CHECK(work.state_generation == aiaec_state_generation(ctx), "fail_closed block %d: work generation %u, live %u", b,
              work.state_generation, aiaec_state_generation(ctx));
        fill_identity_taps((float *)work.outputs[0].data, work.outputs[0].elements, N);
        if (fail && kind != FAIL_STATUS) ((float *)work.outputs[0].data)[work.outputs[0].elements / 3] = NAN;
        CHECK(aiaec_postprocess(ctx, fail && kind != FAIL_NONFINITE ? 1 : 0, pcm_out + (size_t)b * BS) == AIAEC_OK,
              "fail_closed block %d: postprocess", b);
    }
    for (b = 1; b < BLOCKS; ++b) {
        const int16_t *out = pcm_out + (size_t)b * BS, *want = mic_in + (size_t)(b - 1) * BS;
        for (h = 0; h < N; ++h) {
            const int16_t *o = out + (size_t)h * HOP, *w = want + (size_t)h * HOP;
            if (b == FAILED && h == N - 1) {
                CHECK(!pcm16_close(o, w, HOP) && energy16(o, HOP) > 0, "fail_closed(%d) block %d hop %d: the failing hop is %s", (int)kind, b, h,
                      energy16(o, HOP) > 0 ? "the delayed mic" : "silent (it keeps the previous frame's overlap)");
            } else if (b == FAILED + 1 && h < N - 1) {
                CHECK(all_zero16(o, HOP), "fail_closed(%d) block %d hop %d: not exactly zero (a failed batch must be silent, not the mic)", (int)kind, b, h);
            } else if (b == FAILED + 1 && h == N - 1) {
                CHECK(!pcm16_close(o, w, HOP) && energy16(o, HOP) > 0, "fail_closed(%d) block %d hop %d: the recovery hop is %s", (int)kind, b, h,
                      energy16(o, HOP) > 0 ? "the delayed mic" : "silent");
            } else {
                CHECK(pcm16_close(o, w, HOP), "fail_closed(%d) block %d hop %d: not the delayed mic", (int)kind, b, h);
            }
        }
    }
    if (kind == FAIL_NONFINITE) {
        /* a refused commit (a reported failure only bypasses) advances the generation, and the block carries the new value */
        CHECK(gen_seen[FAILED + 1] != gen_seen[FAILED] && gen_seen[FAILED + 1] != 0u && gen_seen[FAILED + 2] == gen_seen[FAILED + 1],
              "fail_closed(%d): generation %u -> %u -> %u around the refused job", (int)kind, gen_seen[FAILED], gen_seen[FAILED + 1], gen_seen[FAILED + 2]);
    }
    {   /* a reset advances the generation and the next SUBMIT block carries it */
        AiaecInput input;
        AiaecWork work;
        const unsigned int before = aiaec_state_generation(ctx);
        aiaec_reset(ctx);
        dv_input(&input, mic_in, far_zero, BS);
        CHECK(aiaec_preprocess(ctx, &input, &work) == AIAEC_OK && work.state_generation == aiaec_state_generation(ctx) &&
              work.state_generation != before && work.state_generation != 0u, "fail_closed(%d): after reset work generation %u, live %u, before %u",
              (int)kind, work.state_generation, aiaec_state_generation(ctx), before);
        fill_identity_taps((float *)work.outputs[0].data, work.outputs[0].elements, N);
        CHECK(aiaec_postprocess(ctx, 0, pcm_out) == AIAEC_OK, "fail_closed: postprocess after reset");
    }
    printf("section_fail_closed(%s): failed batch exactly silent (N-1 hops), neighbours are the delayed mic, generation tracked\n",
           kind == FAIL_STATUS ? "infer_status" : kind == FAIL_NONFINITE ? "non-finite taps" : "both");
    free(mic_in); free(far_zero); free(pcm_out); free(pool);
    return g_fail != bad;
}

static int section_fail_closed(void) {
    return section_fail_closed_one(FAIL_STATUS) | section_fail_closed_one(FAIL_NONFINITE) | section_fail_closed_one(FAIL_BOTH);
}

/* AiaecInfo.trim_samples (frames_per_call * hop) against an IDENTITY accelerator: CCM taps
 * forced to the (delay 0, centre frequency tap) coefficient 1+0j, so the only reason input and
 * output can differ is the pure sample shift being measured. Every block's output must equal the
 * previous block's input. */
static int section_trim(int n, int fft) {
    AiaecConfig cfg = make_cfg(fft, n);
    void *pool = NULL;
    AiaecInfo bi;
    AiaecCtx *ctx = make_ctx_info(&cfg, &pool, "delay", &bi);
    const int hop = fft / 2, blocks = 5, bs = hop * n;
    int16_t *mic_in, *far_zero, *pcm_out;
    int k, bidx;

    if (!ctx) return 1;
    CHECK(bi.trim_samples == n * hop && bi.queue_frames == n - 1,
          "delay N=%d: trim %d queue %d, want %d / %d", n, bi.trim_samples, bi.queue_frames, n * hop, n - 1);
    mic_in = calloc((size_t)bs * blocks, sizeof(int16_t));
    far_zero = calloc((size_t)bs, sizeof(int16_t));
    pcm_out = calloc((size_t)bs, sizeof(int16_t));
    for (k = 0; k < bs * blocks; ++k) mic_in[k] = (int16_t)(((k * 977) % 20001) - 10000);
    for (bidx = 0; bidx < blocks; ++bidx) {
        AiaecInput input;
        AiaecWork work;
        dv_input(&input, mic_in + (size_t)bidx * bs, far_zero, bs);
        CHECK(aiaec_preprocess(ctx, &input, &work) == AIAEC_OK, "delay N=%d block %d: preprocess", n, bidx);
        fill_identity_taps((float *)work.outputs[0].data, work.outputs[0].elements, n);
        CHECK(aiaec_postprocess(ctx, 0, pcm_out) == AIAEC_OK, "delay N=%d block %d: postprocess", n, bidx);
        if (bidx == 0) continue;
        CHECK(pcm16_close(pcm_out, mic_in + (size_t)(bidx - 1) * bs, (size_t)bs) &&
              !pcm16_close(pcm_out, mic_in + (size_t)bidx * bs, (size_t)bs),
              "delay N=%d fft=%d block %d: output is not exactly the previous block's input", n, fft, bidx);
    }
    printf("section_trim(N=%d fft=%d): identity accelerator confirms %d samples\n", n, fft, n * hop);
    free(mic_in); free(far_zero); free(pcm_out); free(pool);
    return 0;
}
#endif /* PRODUCT_DV */

#if PRODUCT_AL
static void al_input(AiaecInput *in, const float *residual, const float *far, const AiaecAlignEvent *ev, int samples, int n_events) {
    memset(in, 0, sizeof(*in));
    in->kind = AIAEC_INPUT_ALIGN_ULCNET;
    in->samples = samples;
    in->u.align_ulcnet.residual_in = residual;
    in->u.align_ulcnet.far_in = far;
    in->u.align_ulcnet.events = ev;
    in->u.align_ulcnet.event_count = n_events;
}

/* Normal hops, realignment/event bypass hops and a forced inference failure, against an independent
 * UlcnetStream driven directly with ulcnet_stream_pre_process_f32()/post_process(). */
static int section_align_block(void) {
    AiaecConfig cfg = make_cfg(256, 1);
    void *pool = NULL, *rpool = NULL;
    AiaecCtx *ctx = make_ctx(&cfg, &pool, "align");
    UlcnetGridSpec gspec;
    UlcnetGrid grid;
    UlcnetStreamConfig rcfg;
    UlcnetStreamMemReq rreq;
    UlcnetStream *ref;
    int hop, i, bad = 0, submit_calls = 0;
    int16_t *pcm_blk, *pcm_ref;
    float *residual, *far;
    long long ghop = 0;
    /* changed, solid, force_fail. Hop 2 is a changed=1 realignment; hop 4 the first solid=1 after a
     * false->true edge; hop 7 a normal submit hop whose inference is forced to fail. */
    const struct { int changed, solid, force_fail; } hops[8] = {
        { 0, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 },
        { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 1 },
    };
    const int n_hops = 8;

    if (!ctx) return 1;
    memset(&gspec, 0, sizeof(gspec));
    gspec.sample_rate = 16000; gspec.n_fft = 256; gspec.delay_depth = 63;
    if (ulcnet_grid_make(&grid, &gspec) != ULCNET_GRID_OK) { FAIL("align: ulcnet_grid_make"); return 1; }
    hop = grid.hop;
    memset(&rcfg, 0, sizeof(rcfg));
    rcfg.grid = grid;
    if (ulcnet_stream_get_mem_size(&rcfg, &rreq) != 0) { FAIL("align: ref get_mem_size"); return 1; }
    if (posix_memalign(&rpool, (size_t)rreq.alignment, (size_t)rreq.bytes) != 0) { FAIL("align: ref pool"); return 1; }
    ref = ulcnet_stream_init_ex(&rcfg, rpool, (size_t)rreq.bytes, &rreq);
    if (!ref) { FAIL("align: ref init"); return 1; }

    residual = calloc((size_t)hop, sizeof(float));
    far = calloc((size_t)hop, sizeof(float));
    pcm_blk = calloc((size_t)hop, sizeof(int16_t));
    pcm_ref = calloc((size_t)hop, sizeof(int16_t));

    for (i = 0; i < n_hops; ++i) {
        AiaecInput input;
        AiaecWork work;
        AiaecAlignEvent ev;
        UlcnetStreamDelay rdelay;
        UlcnetHopIO rio;
        int rc, rrc, k, ai_rc, all_zero;

        for (k = 0; k < hop; ++k) {
            residual[k] = 0.001f * (float)(((long long)k * 5 + ghop * 3) % 97 - 48);
            far[k]      = 0.001f * (float)(((long long)k * 11 + ghop * 7) % 89 - 44);
        }
        ev.changed = hops[i].changed; ev.solid = hops[i].solid;
        al_input(&input, residual, far, &ev, hop, 1);

        rc = aiaec_preprocess(ctx, &input, &work);
        CHECK(rc == AIAEC_OK, "align hop %d: preprocess rc=%d", i, rc);
        rdelay.changed = ev.changed; rdelay.solid = ev.solid;
        rrc = ulcnet_stream_pre_process_f32(ref, residual, far, rdelay, &rio);
        CHECK(rrc >= 0, "align hop %d: ref pre rc=%d", i, rrc);

        if (rrc == 1) {
            CHECK(work.action == AIAEC_SUBMIT, "align hop %d: expected SUBMIT, got %d", i, (int)work.action);
            CHECK(work.input_count == 4 && work.output_count == 1, "align hop %d: counts %d/%d", i, work.input_count, work.output_count);
            CHECK(work.inputs[0].elements == rio.inputs.spectrum_bins_elements && work.inputs[3].elements == rio.inputs.spectrum_bins_elements &&
                  work.outputs[0].elements == rio.outputs.spectrum_ri_elements, "align hop %d: element counts", i);
            fake_mask(ghop, work.outputs[0].data, work.outputs[0].elements);
            fake_mask(ghop, rio.outputs.output, rio.outputs.spectrum_ri_elements);
            submit_calls++;
            ai_rc = hops[i].force_fail ? 1 : 0;
        } else {
            CHECK(work.action == AIAEC_BYPASS, "align hop %d: expected BYPASS, got %d", i, (int)work.action);
            CHECK(work.input_count == 0 && work.output_count == 0 && work.state_generation == 0u, "align hop %d: bypass work not zeroed", i);
            ai_rc = 0;
        }
        CHECK(aiaec_postprocess(ctx, ai_rc, pcm_blk) == AIAEC_OK, "align hop %d: postprocess", i);
        CHECK(ulcnet_stream_post_process(ref, ai_rc, pcm_ref) == 1, "align hop %d: ref post", i);
        if (memcmp(pcm_blk, pcm_ref, (size_t)hop * sizeof(int16_t)) != 0) {
            FAIL("align hop %d: block path differs from ulcnet_stream_* driven directly", i);
            bad = 1;
        }
        /* Fail-OPEN: a forced failure on an ordinary submit hop still passes the residual through.
         * A realignment bypass hop is not the place to assert non-silence: the reprime hop coincides
         * with the reset, so its first synthesis half is history that does not exist yet. */
        if (hops[i].force_fail) {
            all_zero = 1;
            for (k = 0; k < hop; ++k) if (pcm_blk[k] != 0) { all_zero = 0; break; }
            CHECK(!all_zero, "align hop %d: forced-failure submit hop produced silence; must fail open", i);
        }
        ghop++;
    }
    CHECK(submit_calls == 6, "align: expected 6 submits across %d hops, got %d", n_hops, submit_calls);
    printf("section_align_block: %d hops, %d submits, bit-exact vs ulcnet_stream_* (2 bypass hops, 1 forced-failure submit)\n", n_hops, submit_calls);
    free(residual); free(far); free(pcm_blk); free(pcm_ref); free(pool); free(rpool);
    return bad;
}

/* infer_status must be truly ignored on a bypass hop, not just conventionally 0: two contexts get
 * the same hops; at the bypass hop one is closed with 0 and one with garbage. */
static int section_bypass_ignores_infer_status(void) {
    AiaecConfig cfg = make_cfg(256, 1);
    void *pool_a = NULL, *pool_b = NULL;
    AiaecCtx *ctx_a = make_ctx(&cfg, &pool_a, "bypass_a"), *ctx_b = make_ctx(&cfg, &pool_b, "bypass_b");
    int hop = 128, i, bad = 0;
    int16_t pcm_a[128], pcm_b[128];
    float residual[128], far[128];
    const struct { int changed, solid; } hops[2] = { { 0, 0 }, { 1, 0 } };

    if (!ctx_a || !ctx_b) return 1;
    for (i = 0; i < hop; ++i) {
        residual[i] = 0.001f * (float)((i * 5) % 97 - 48);
        far[i]      = 0.001f * (float)((i * 11) % 89 - 44);
    }
    for (i = 0; i < 2; ++i) {
        AiaecInput input;
        AiaecWork wa, wb;
        AiaecAlignEvent ev;
        int rc_a, rc_b, ai_a = 0, ai_b = 0;

        ev.changed = hops[i].changed; ev.solid = hops[i].solid;
        al_input(&input, residual, far, &ev, hop, 1);
        rc_a = aiaec_preprocess(ctx_a, &input, &wa);
        rc_b = aiaec_preprocess(ctx_b, &input, &wb);
        CHECK(rc_a == AIAEC_OK && rc_b == AIAEC_OK && wa.action == wb.action, "bypass hop %d: preprocess", i);
        if (i == 1) {
            CHECK(wa.action == AIAEC_BYPASS, "bypass: hop 1 was not flagged BYPASS (test setup is wrong)");
            ai_b = 0xBAD1;
        } else {
            fake_mask(i, wa.outputs[0].data, wa.outputs[0].elements);
            fake_mask(i, wb.outputs[0].data, wb.outputs[0].elements);
        }
        CHECK(aiaec_postprocess(ctx_a, ai_a, pcm_a) == AIAEC_OK, "bypass hop %d: post a", i);
        CHECK(aiaec_postprocess(ctx_b, ai_b, pcm_b) == AIAEC_OK, "bypass hop %d: post b", i);
        if (memcmp(pcm_a, pcm_b, sizeof(pcm_a)) != 0) { FAIL("bypass hop %d: infer_status leaked into a bypass hop", i); bad = 1; }
    }
    printf("section_bypass_ignores_infer_status: bypass hop identical with infer_status 0 and 0xBAD1\n");
    free(pool_a); free(pool_b);
    return bad;
}

/* Fail-OPEN identity: infer_status != 0 every hop, so pcm_out must equal the previous hop's
 * residual_in (int16, saturated). */
static int section_trim_align(void) {
    AiaecConfig cfg = make_cfg(256, 1);
    void *pool = NULL;
    AiaecInfo bi;
    AiaecCtx *ctx = make_ctx_info(&cfg, &pool, "delay_align", &bi);
    const int hop = 128, blocks = 5;
    float *residual, *far_zero;
    int16_t *pcm_out, *expect_pcm;
    int k, bidx;

    if (!ctx) return 1;
    CHECK(bi.trim_samples == hop && bi.queue_frames == 0,
          "delay(align): trim %d queue %d, want %d / 0", bi.trim_samples, bi.queue_frames, hop);
    residual = calloc((size_t)hop * blocks, sizeof(float));
    far_zero = calloc((size_t)hop, sizeof(float));
    pcm_out = calloc((size_t)hop, sizeof(int16_t));
    expect_pcm = calloc((size_t)hop * blocks, sizeof(int16_t));
    for (k = 0; k < hop * blocks; ++k) {
        float v;
        residual[k] = 0.001f * (float)(((k * 977) % 20001) - 10000);
        v = residual[k] * 32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        expect_pcm[k] = (int16_t)v;
    }
    for (bidx = 0; bidx < blocks; ++bidx) {
        AiaecInput input;
        AiaecWork work;
        AiaecAlignEvent ev;
        ev.changed = 0; ev.solid = 0;
        al_input(&input, residual + (size_t)bidx * hop, far_zero, &ev, hop, 1);
        CHECK(aiaec_preprocess(ctx, &input, &work) == AIAEC_OK, "delay(align) block %d: preprocess", bidx);
        CHECK(aiaec_postprocess(ctx, -1, pcm_out) == AIAEC_OK, "delay(align) block %d: postprocess", bidx);
        if (bidx == 0) continue;
        CHECK(pcm16_close(pcm_out, expect_pcm + (size_t)(bidx - 1) * hop, (size_t)hop),
              "delay(align) block %d: pcm_out is not the previous block's residual_in", bidx);
    }
    printf("section_trim_align: fail-open identity confirms %d samples\n", hop);
    free(residual); free(far_zero); free(pcm_out); free(expect_pcm); free(pool);
    return 0;
}
#endif /* PRODUCT_AL */

/* Fills a valid input for the product's model; samples = cfg hop_in, zeros. */
typedef struct Feed {
    int16_t mic[1024], far16[1024];
    float resid[1024], farf[1024];
    AiaecAlignEvent ev[8];
} Feed;

static void feed_input(const Feed *f, int samples, int n, AiaecInput *in) {
    memset(in, 0, sizeof(*in));
    in->samples = samples;
#if PRODUCT_DV
    (void)n;
    in->kind = AIAEC_INPUT_DEEPVQE_S;
    in->u.deepvqe_s.mic_in = f->mic;
    in->u.deepvqe_s.far_in = f->far16;
#else
    in->kind = AIAEC_INPUT_ALIGN_ULCNET;
    in->u.align_ulcnet.residual_in = f->resid;
    in->u.align_ulcnet.far_in = f->farf;
    in->u.align_ulcnet.events = f->ev;
    in->u.align_ulcnet.event_count = n;
#endif
}

static void fake_head(AiaecWork *w) {
    if (w->action != AIAEC_SUBMIT) return;
#if PRODUCT_DV
    fake_taps(0, w->outputs[0].data, w->outputs[0].elements);
#else
    fake_mask(0, w->outputs[0].data, w->outputs[0].elements);
#endif
}

/* A live first block's views at 16k/256 N=1, against the exporter's literal numbers (129 bins). */
static int section_work_views(void) {
    AiaecConfig cfg = make_cfg(256, 1);
    AiaecWork live;
    AiaecInput input;
    static Feed feed;
    void *pool = NULL;
    AiaecCtx *ctx = make_ctx(&cfg, &pool, "views");
    int16_t pcm[128];
    int k;

    if (!ctx) return 1;
    feed_input(&feed, 128, 1, &input);
    CHECK(aiaec_preprocess(ctx, &input, &live) == AIAEC_OK && live.action == AIAEC_SUBMIT, "views: live preprocess");
    CHECK(live.input_count == (PRODUCT_DV ? 2 : 4) && live.output_count == 1, "views: counts %d/%d", live.input_count, live.output_count);
    for (k = 0; k < live.input_count; ++k) {
        const unsigned c = PRODUCT_DV ? 2u : 1u;
        CHECK(live.inputs[k].data != NULL && live.inputs[k].b == 1u && live.inputs[k].c == c && live.inputs[k].h == 1u &&
              live.inputs[k].w == 129u && live.inputs[k].elements == (size_t)c * 129u, "views: inputs[%d] shape", k);
    }
    CHECK(live.outputs[0].data != NULL && live.outputs[0].b == 1u && live.outputs[0].elements ==
          (PRODUCT_DV ? 129u * 18u : 2u * 129u), "views: outputs[0] shape");
    CHECK(PRODUCT_DV ? (live.outputs[0].c == 1u && live.outputs[0].h == 129u && live.outputs[0].w == 18u)
                     : (live.outputs[0].c == 2u && live.outputs[0].h == 1u && live.outputs[0].w == 129u),
          "views: outputs[0] axes");
    fake_head(&live);
    CHECK(aiaec_postprocess(ctx, 0, pcm) == AIAEC_OK, "views: cleanup postprocess");
    printf("section_work_views: first-block views equal the exporter's shapes\n");
    free(pool);
    return 0;
}

/* Kind / size / event-count mismatches are rejected before any state write: by return code, by
 * state_generation staying put, and by a correct call on the same ctx still working. */
static int section_mismatch(void) {
    AiaecConfig cfg = make_cfg(256, 1);
    AiaecInput input;
    AiaecWork work;
    static Feed feed;
    void *pool = NULL;
    AiaecCtx *ctx = make_ctx(&cfg, &pool, "mismatch");
    unsigned int gen_before, gen_reset;
    int16_t pcm[128];
    int rc;

    if (!ctx) return 1;
    gen_before = aiaec_state_generation(ctx);

    /* wrong kind: the other model's input */
    feed_input(&feed, 128, 1, &input);
    input.kind = PRODUCT_DV ? AIAEC_INPUT_ALIGN_ULCNET : AIAEC_INPUT_DEEPVQE_S;
    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_ERR_INPUT_KIND, "mismatch: wrong kind rc=%d", rc);
    CHECK(aiaec_state_generation(ctx) == gen_before, "mismatch: state advanced after a rejected kind");

    /* right kind, wrong sample count */
    feed_input(&feed, 64, 1, &input);
    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_ERR_INPUT_SIZE, "mismatch: wrong samples rc=%d", rc);
    CHECK(aiaec_state_generation(ctx) == gen_before, "mismatch: state advanced after a bad sample count");

#if PRODUCT_AL
    feed_input(&feed, 128, 0, &input);
    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_ERR_INPUT_SIZE, "mismatch: event_count=0 rc=%d", rc);
    feed_input(&feed, 128, 2, &input);
    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_ERR_INPUT_SIZE, "mismatch: event_count=2 rc=%d", rc);
    CHECK(aiaec_state_generation(ctx) == gen_before, "mismatch: state advanced after a bad event count");
    feed_input(&feed, 128, 1, &input);
    input.u.align_ulcnet.events = NULL;
    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_ERR_NULL, "mismatch: NULL events with event_count=1 rc=%d", rc);
    CHECK(aiaec_state_generation(ctx) == gen_before, "mismatch: state advanced after NULL events");
#endif

    /* NULL pieces */
    CHECK(aiaec_preprocess(NULL, &input, &work) == AIAEC_ERR_NULL, "mismatch: NULL ctx");
    CHECK(aiaec_preprocess(ctx, NULL, &work) == AIAEC_ERR_NULL, "mismatch: NULL input");
    CHECK(aiaec_postprocess(ctx, 0, pcm) == AIAEC_ERR_SEQUENCE, "mismatch: postprocess without preprocess");

    /* a correct call after every rejection; then a double preprocess is a sequence error */
    feed_input(&feed, 128, 1, &input);
    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_OK && work.action == AIAEC_SUBMIT, "mismatch: correct call after rejections rc=%d", rc);
    CHECK(aiaec_preprocess(ctx, &input, &work) == AIAEC_ERR_SEQUENCE, "mismatch: second preprocess before postprocess");
    fake_head(&work);
    CHECK(aiaec_postprocess(ctx, 0, pcm) == AIAEC_OK, "mismatch: postprocess after recovery");

    /* reset: generation advances, ctx usable at once */
    gen_reset = aiaec_state_generation(ctx);
    aiaec_reset(ctx);
    CHECK(aiaec_state_generation(ctx) != gen_reset, "mismatch: reset did not advance state_generation");
    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_OK, "mismatch: preprocess after reset rc=%d", rc);
    fake_head(&work);
    CHECK(aiaec_postprocess(ctx, 0, pcm) == AIAEC_OK, "mismatch: postprocess after reset");

    printf("section_mismatch: rejections precede any state write; recovery and reset work\n");
    free(pool);
    return 0;
}

/* The pool: init recomputes the size itself. A pool smaller than get_mem_size()'s, a misaligned one and
 * is refused (AIAEC_ERR_POOL) with ctx NULL and info untouched; the exact
 * size and a larger pool construct, and a larger pool is zeroed only up to the size needed. */
static int section_pool(void) {
    AiaecConfig cfg = make_cfg(256, 1);
    StreamMemReq req;
    AiaecInfo info, ref;
    AiaecCtx *ctx = NULL;
    unsigned char *pool = NULL;
    int rc;

    if (aiaec_get_mem_size(&cfg, &req) != AIAEC_OK) { FAIL("pool: get_mem_size"); return 1; }
    if (posix_memalign((void **)&pool, 16, (size_t)req.bytes + 64) != 0) { FAIL("pool alloc"); return 1; }
    memset(&ref, 0xA5, sizeof(ref));

    memset(&info, 0xA5, sizeof(info));
    ctx = (AiaecCtx *)pool;
    rc = aiaec_init(&cfg, pool, 32u, &ctx, &info);
    CHECK(rc == AIAEC_ERR_POOL && ctx == NULL && memcmp(&info, &ref, sizeof(info)) == 0, "pool: tiny pool rc=%d", rc);
    ctx = (AiaecCtx *)pool;
    rc = aiaec_init(&cfg, pool, (size_t)req.bytes - 1u, &ctx, &info);
    CHECK(rc == AIAEC_ERR_POOL && ctx == NULL && memcmp(&info, &ref, sizeof(info)) == 0, "pool: one byte short rc=%d", rc);
    ctx = (AiaecCtx *)pool;
    rc = aiaec_init(&cfg, pool + 1, (size_t)req.bytes, &ctx, &info);
    CHECK(rc == AIAEC_ERR_POOL && ctx == NULL && memcmp(&info, &ref, sizeof(info)) == 0, "pool: misaligned rc=%d", rc);
    CHECK(aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx, &info) == AIAEC_OK && ctx != NULL, "pool: the exact size must construct");

    memset(pool, 0x5A, (size_t)req.bytes + 64);
    rc = aiaec_init(&cfg, pool, (size_t)req.bytes + 64, &ctx, &info);
    CHECK(rc == AIAEC_OK && ctx != NULL, "pool: a larger pool must construct, rc=%d", rc);
    {
        size_t t;
        int untouched = 1;
        for (t = (size_t)req.bytes; t < (size_t)req.bytes + 64; ++t) if (pool[t] != 0x5A) untouched = 0;
        CHECK(untouched, "pool: bytes beyond get_mem_size() were written");
    }
    printf("section_pool: short and misaligned pools refused; exact and larger pools construct\n");
    free(pool);
    return 0;
}

int main(void) {
#if PRODUCT_DV
    g_fail |= section_deepvqe_block_vs_stream(1, 256);
    g_fail |= section_deepvqe_block_vs_stream(8, 256);
    g_fail |= section_deepvqe_block_vs_stream(1, 512);
    g_fail |= section_batching_vs_n1(8);
    g_fail |= section_fail_closed();
    g_fail |= section_trim(1, 256);
    g_fail |= section_trim(8, 256);
    g_fail |= section_trim(1, 512);
#endif
#if PRODUCT_AL
    g_fail |= section_align_block();
    g_fail |= section_bypass_ignores_infer_status();
    g_fail |= section_trim_align();
#endif
    g_fail |= section_work_views();
    g_fail |= section_mismatch();
    g_fail |= section_pool();

    printf("test_aiaec_block[%s]: %s\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
