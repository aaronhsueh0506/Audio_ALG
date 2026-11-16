/* The PRODUCTION catalog through the real public API: this file is built without
 * -DAIAEC_TEST_CATALOG, so only the rows with a delivered graph resolve. Built once per product.
 *
 *   Each model: 16k/256 N=1 and N=8, 16k/512 N=1 and 48k/1024 N=1 and N=2 resolve, all to the one graph path of that
 *     model; every other N on those grids (16k/256 N=2, 16k/512 N=8, 48k/1024 N=4) is refused with
 *     AIAEC_ERR_FRAMES_PER_CALL before any SDK call, ctx / info untouched.
 *   Both: a hop_in that is not a multiple of the model hop and a grid that is not registered are refused. The
 *     refusal order is model, grid, hop_in, frames_per_call.
 * The test-catalog behavior (every row resolving, work views, N derivation) is test_aiaec_graph_io.c. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"

#define PRODUCT_DV (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S)

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static AiaecConfig cfg_of(int sr, int fft, int hop_in) {
    AiaecConfig c;
    c.sr = sr; c.fft = fft; c.hop_in = hop_in;
    return c;
}

/* One config refused with `want` by every entry that validates, with nothing allocated or written. */
static void expect_refused(const AiaecConfig *cfg, int want, const char *what) {
    StreamMemReq req;
    AiaecInfo info, info_ref;
    AiaecCtx *ctx;
    AiaecFreqCtx *cf;
    unsigned char pool[256];
    int rc;

    memset(&req, 0xA5, sizeof(req));
    rc = aiaec_get_mem_size(cfg, &req);
    CHECK(rc == want, "%s: get_mem_size rc=%d (%s), want %d", what, rc, aiaec_status_text(rc), want);
    CHECK(req.bytes == 0xA5A5A5A5A5A5A5A5ull, "%s: req touched on a refusal", what);
    memset(&info, 0xA5, sizeof(info));
    memset(&info_ref, 0xA5, sizeof(info_ref));
    ctx = (AiaecCtx *)pool;
    rc = aiaec_init(cfg, pool, sizeof(pool), &ctx, &info);
    CHECK(rc == want && ctx == NULL, "%s: init rc=%d (%s)", what, rc, aiaec_status_text(rc));
    CHECK(memcmp(&info, &info_ref, sizeof(info)) == 0, "%s: info touched on a refusal", what);
    cf = (AiaecFreqCtx *)pool;
    CHECK(aiaec_init_freq(cfg, pool, sizeof(pool), &cf, &info) == want && cf == NULL, "%s: init_freq", what);
    CHECK(memcmp(&info, &info_ref, sizeof(info)) == 0, "%s: info touched by _freq on a refusal", what);
}

static void section_refusal_order(void) {
    AiaecConfig c;

    c = cfg_of(32000, 1024, 512);   expect_refused(&c, AIAEC_ERR_GRID, "no (sr,fft) row");
    c = cfg_of(48000, 512, 256);    expect_refused(&c, AIAEC_ERR_GRID, "48k with fft 512 not registered");
    c = cfg_of(16000, 1024, 512);   expect_refused(&c, AIAEC_ERR_GRID, "fft not registered");
    c = cfg_of(16000, 256, 129);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 129 at 16k/256");
    c = cfg_of(16000, 256, 100);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 100");
    c = cfg_of(16000, 256, 0);      expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 0");
    c = cfg_of(16000, 256, -128);   expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in negative");
    c = cfg_of(16000, 512, 255);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 255 at 16k/512");
    /* grid precedes hop_in: an unregistered grid with a bad hop_in is a grid error */
    c = cfg_of(32000, 1024, 129);   expect_refused(&c, AIAEC_ERR_GRID, "grid before hop_in");
    c = cfg_of(48000, 1024, 513);   expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 513 at 48k/1024");
    /* registered grids and frames_per_call values without a delivered graph */
    c = cfg_of(16000, 256, 256);    expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/256 N=2 (not in the catalog)");
    c = cfg_of(16000, 512, 512);    expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/512 N=2 (not in the catalog)");
    c = cfg_of(16000, 512, 2048);   expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/512 N=8");
    c = cfg_of(48000, 1024, 2048);  expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "48k/1024 N=4");
    c = cfg_of(48000, 1024, 4096);  expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "48k/1024 N=8");
    printf("section_refusal_order: done\n");
}

/* One delivered row: constructs, reports `path`, and its first pre call carries the row's work. */
static void expect_row(int sr, int fft, int hop_in, const char *path) {
    AiaecConfig cfg = cfg_of(sr, fft, hop_in);
    const unsigned bins = (unsigned)(fft / 2 + 1), nf = (unsigned)(hop_in / (fft / 2));
    StreamMemReq req, req_f;
    AiaecInfo info, info_f;
    AiaecCtx *ctx = NULL;
    AiaecFreqCtx *cf = NULL;
    void *pool = NULL, *pool_f = NULL;
    int rc;

    rc = aiaec_get_mem_size(&cfg, &req);
    CHECK(rc == AIAEC_OK, "row %d/%d: get_mem_size rc=%d (%s)", fft, hop_in, rc, aiaec_status_text(rc));
    if (rc != AIAEC_OK) return;
    CHECK(posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) == 0, "row %d/%d: pool", fft, hop_in);
    rc = aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx, &info);
    CHECK(rc == AIAEC_OK && ctx != NULL, "row %d/%d: init rc=%d (%s)", fft, hop_in, rc, aiaec_status_text(rc));
    if (ctx) {
#if PRODUCT_DV
        static const int16_t zeros16[1024];
#else
        static const float zeros_f[1024];
        AiaecAlignEvent no_event[8];
#endif
        AiaecInput in;
        StreamWork work;

        CHECK(info.graph_path && strcmp(info.graph_path, path) == 0, "row %d/%d: graph path %s, want %s", fft, hop_in,
              info.graph_path ? info.graph_path : "(null)", path);
        CHECK(aiaec_state_generation(ctx) == 0u, "row %d/%d: generation after init", fft, hop_in);
        memset(&in, 0, sizeof(in));
        in.samples = hop_in;
#if PRODUCT_DV
        in.kind = AIAEC_INPUT_DEEPVQE_S;
        in.u.deepvqe_s.mic_in = zeros16; in.u.deepvqe_s.far_in = zeros16;
#else
        memset(no_event, 0, sizeof(no_event));
        in.kind = AIAEC_INPUT_ALIGN_ULCNET;
        in.u.align_ulcnet.residual_in = zeros_f; in.u.align_ulcnet.far_in = zeros_f;
        in.u.align_ulcnet.events = no_event; in.u.align_ulcnet.event_count = (int)nf;
#endif
        CHECK(aiaec_preprocess(ctx, &in, &work) == AIAEC_OK && work.action == STREAM_WORK_SUBMIT &&
              work.output_count >= 1 && work.inputs[0].w == bins,
              "row %d/%d: work (%d in, %d out)", fft, hop_in, work.input_count, work.output_count);
#if PRODUCT_DV
        /* mic, far [1,2,N,bins]; CCM taps [1,N,bins,18] */
        CHECK(work.input_count == 2 && work.output_count == 1 &&
              work.inputs[0].b == 1u && work.inputs[0].c == 2u && work.inputs[0].h == nf &&
              work.inputs[0].elements == 2u * nf * bins && work.inputs[1].w == bins && work.inputs[1].elements == 2u * nf * bins &&
              work.outputs[0].b == 1u && work.outputs[0].c == nf && work.outputs[0].h == bins && work.outputs[0].w == 18u &&
              work.outputs[0].elements == 18u * nf * bins,
              "row %d/%d: DeepVQE-S work views", fft, hop_in);
#else
        /* four signals [1,1,N,bins], frame_reset [1,1,1,N] last at N > 1; the mask [1,2,N,bins] */
        CHECK(work.input_count == (nf > 1u ? 5 : 4) && work.output_count == 1 &&
              work.inputs[0].h == nf && work.inputs[0].elements == nf * bins &&
              work.outputs[0].c == 2u && work.outputs[0].h == nf && work.outputs[0].elements == 2u * nf * bins &&
              (nf == 1u || (work.inputs[4].h == 1u && work.inputs[4].w == nf)),
              "row %d/%d: Align-ULCNet work views", fft, hop_in);
#endif
        aiaec_reset(ctx);   /* closes the open block and restarts */
        CHECK(aiaec_state_generation(ctx) != 0u, "row %d/%d: a reset advances the generation", fft, hop_in);
    }
    CHECK(aiaec_get_mem_size_freq(&cfg, &req_f) == AIAEC_OK &&
          posix_memalign(&pool_f, (size_t)req_f.alignment, (size_t)req_f.bytes) == 0, "row %d/%d: freq pool", fft, hop_in);
    CHECK(aiaec_init_freq(&cfg, pool_f, (size_t)req_f.bytes, &cf, &info_f) == AIAEC_OK && cf != NULL &&
          memcmp(&info, &info_f, sizeof(info)) == 0, "row %d/%d: the _freq family reports the same info", fft, hop_in);
    CHECK(aiaec_init(NULL, pool, (size_t)req.bytes, &ctx, &info) == AIAEC_ERR_NULL && ctx == NULL, "row %d/%d: NULL cfg", fft, hop_in);
    CHECK(aiaec_init(&cfg, NULL, (size_t)req.bytes, &ctx, &info) == AIAEC_ERR_NULL, "row %d/%d: NULL pool", fft, hop_in);
    CHECK(aiaec_init(&cfg, pool, (size_t)req.bytes, NULL, &info) == AIAEC_ERR_NULL, "row %d/%d: NULL ctx", fft, hop_in);
    CHECK(aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx, NULL) == AIAEC_ERR_NULL && ctx == NULL, "row %d/%d: NULL info", fft, hop_in);
    free(pool);
    free(pool_f);
}

static void section_delivered_rows(void) {
#if PRODUCT_DV
#define ROW_PATH "/mnt/sd/audio_model/16x16/deepvqes/nvt_model.bin"
#else
#define ROW_PATH "/mnt/sd/audio_model/16x16/alignulcnet/nvt_model.bin"
#endif
    expect_row(16000, 256, 128, ROW_PATH);
    expect_row(16000, 256, 1024, ROW_PATH);
    expect_row(16000, 512, 256, ROW_PATH);
    expect_row(48000, 1024, 512, ROW_PATH);
    expect_row(48000, 1024, 1024, ROW_PATH);
    printf("section_delivered_rows: %s 16k/256 N=1 and N=8, 16k/512 N=1, 48k/1024 N=1 and N=2 construct and report their info\n",
           PRODUCT_DV ? "DeepVQE-S" : "Align-ULCNet");
}

int main(void) {
    section_refusal_order();
    section_delivered_rows();
    printf("test_aiaec_catalog[%s]: %s\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
