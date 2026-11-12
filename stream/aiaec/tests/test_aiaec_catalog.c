/* The PRODUCTION catalog through the real public API: this file is built without
 * -DAIAEC_TEST_CATALOG, so only the rows with a delivered graph resolve. Built once per product.
 *
 *   DeepVQE-S: the 16k/256 rows (N=1, N=8) are registered but have no delivered graph, so
 *     get_mem_size / init refuse them with AIAEC_ERR_FRAMES_PER_CALL before any SDK
 *     call and ctx / info are left untouched; the 16k/512 row with N=1 is delivered and constructs; N=8
 *     there is refused. The refusal order is model, grid, hop_in, frames_per_call.
 *   Align-ULCNet: every row, N=8 included, is refused with AIAEC_ERR_FRAMES_PER_CALL (no delivered graph).
 * The test-catalog behavior (every row resolving, work views, N derivation) is test_aiaec_ports.c. */

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
    AiaecSpecCtx *cs;
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
    cs = (AiaecSpecCtx *)pool;
    cf = (AiaecFreqCtx *)pool;
    CHECK(aiaec_init_spec(cfg, pool, sizeof(pool), &cs, &info) == want && cs == NULL &&
          aiaec_init_freq(cfg, pool, sizeof(pool), &cf, &info) == want && cf == NULL, "%s: init_spec/_freq", what);
    CHECK(memcmp(&info, &info_ref, sizeof(info)) == 0, "%s: info touched by _spec/_freq on a refusal", what);
}

static void section_refusal_order(void) {
    AiaecConfig c;

    c = cfg_of(48000, 1024, 512);   expect_refused(&c, AIAEC_ERR_GRID, "no (sr,fft) row");
    c = cfg_of(16000, 1024, 512);   expect_refused(&c, AIAEC_ERR_GRID, "fft not registered");
    c = cfg_of(16000, 256, 129);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 129 at 16k/256");
    c = cfg_of(16000, 256, 100);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 100");
    c = cfg_of(16000, 256, 0);      expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 0");
    c = cfg_of(16000, 256, -128);   expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in negative");
    c = cfg_of(16000, 512, 255);    expect_refused(&c, AIAEC_ERR_IN_HOP, "hop_in 255 at 16k/512");
    /* grid precedes hop_in: an unregistered grid with a bad hop_in is a grid error */
    c = cfg_of(48000, 1024, 129);   expect_refused(&c, AIAEC_ERR_GRID, "grid before hop_in");
    /* the registered grids without a delivered graph */
    c = cfg_of(16000, 256, 128);    expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/256 N=1");
    c = cfg_of(16000, 256, 1024);   expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/256 N=8");
    c = cfg_of(16000, 256, 256);    expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/256 N=2 (not in the catalog)");
    c = cfg_of(16000, 512, 2048);   expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "16k/512 N=8");
#if !PRODUCT_DV
    c = cfg_of(16000, 512, 256);    expect_refused(&c, AIAEC_ERR_FRAMES_PER_CALL, "Align 16k/512 N=1 (no delivered graph)");
#endif
    printf("section_refusal_order: done\n");
}

#if PRODUCT_DV
/* The one delivered row: DeepVQE-S 16k/512 N=1. */
static void section_shipped_row(void) {
    AiaecConfig cfg = cfg_of(16000, 512, 256);
    StreamMemReq req;
    AiaecInfo info, info_s;
    AiaecCtx *ctx = NULL;
    AiaecSpecCtx *cs = NULL;
    void *pool = NULL, *pool_s = NULL;
    int rc;

    rc = aiaec_get_mem_size(&cfg, &req);
    CHECK(rc == AIAEC_OK, "shipped row: get_mem_size rc=%d (%s)", rc, aiaec_status_text(rc));
    if (rc != AIAEC_OK) return;
    CHECK(posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) == 0, "shipped row: pool");
    rc = aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx, &info);
    CHECK(rc == AIAEC_OK && ctx != NULL, "shipped row: init rc=%d (%s)", rc, aiaec_status_text(rc));
    if (ctx) {
        CHECK(info.graph_path && strcmp(info.graph_path, "/mnt/sd/audio_model/16x16/deepvqes/nvt_model.bin") == 0,
              "shipped row: graph path %s", info.graph_path ? info.graph_path : "(null)");
        CHECK(info.trim_samples == 256 && info.queue_frames == 0 && info.tail_zero_blocks == 1,
              "shipped row: info fields (trim %d queue %d tail %d)", info.trim_samples, info.queue_frames, info.tail_zero_blocks);
        CHECK(aiaec_state_generation(ctx) == 0u, "shipped row: generation after init");
        {   /* the delivered graph's views: mic, far [1,2,1,257]; CCM taps [1,1,257,18] */
            static const int16_t zeros[256];
            AiaecInput in;
            AiaecWork work;
            memset(&in, 0, sizeof(in));
            in.kind = AIAEC_INPUT_DEEPVQE_S; in.samples = 256;
            in.u.deepvqe_s.mic_in = zeros; in.u.deepvqe_s.far_in = zeros;
            CHECK(aiaec_preprocess(ctx, &in, &work) == AIAEC_OK && work.action == AIAEC_SUBMIT &&
                  work.input_count == 2 && work.output_count == 1 &&
                  work.inputs[0].b == 1u && work.inputs[0].c == 2u && work.inputs[0].h == 1u && work.inputs[0].w == 257u &&
                  work.inputs[0].elements == 514u && work.inputs[1].w == 257u && work.inputs[1].elements == 514u &&
                  work.outputs[0].b == 1u && work.outputs[0].c == 1u && work.outputs[0].h == 257u && work.outputs[0].w == 18u &&
                  work.outputs[0].elements == 4626u,
                  "shipped row: work views (%d in, %d out)", work.input_count, work.output_count);
            aiaec_reset(ctx);   /* closes the open block and restarts */
            CHECK(aiaec_state_generation(ctx) != 0u, "shipped row: a reset advances the generation");
        }
        CHECK(posix_memalign(&pool_s, (size_t)req.alignment, (size_t)req.bytes) == 0, "shipped row: spec pool");
        CHECK(aiaec_init_spec(&cfg, pool_s, (size_t)req.bytes, &cs, &info_s) == AIAEC_OK && cs != NULL &&
              memcmp(&info, &info_s, sizeof(info)) == 0, "shipped row: the _spec family reports the same info");
    }
    CHECK(aiaec_init(NULL, pool, (size_t)req.bytes, &ctx, &info) == AIAEC_ERR_NULL && ctx == NULL, "shipped row: NULL cfg");
    CHECK(aiaec_init(&cfg, NULL, (size_t)req.bytes, &ctx, &info) == AIAEC_ERR_NULL, "shipped row: NULL pool");
    CHECK(aiaec_init(&cfg, pool, (size_t)req.bytes, NULL, &info) == AIAEC_ERR_NULL, "shipped row: NULL ctx");
    CHECK(aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx, NULL) == AIAEC_ERR_NULL && ctx == NULL, "shipped row: NULL info");
    free(pool);
    free(pool_s);
    printf("section_shipped_row: DeepVQE-S 16k/512 N=1 constructs and reports its info\n");
}
#endif

int main(void) {
    section_refusal_order();
#if PRODUCT_DV
    section_shipped_row();
#endif
    printf("test_aiaec_catalog[%s]: %s\n", PRODUCT_DV ? "deepvqe_s" : "align_ulcnet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
