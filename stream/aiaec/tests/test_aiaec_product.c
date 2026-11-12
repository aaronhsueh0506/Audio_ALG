/* The product binding (aiaec_product.h): aiaec.c compiled with -DAIAEC_PRODUCT_MODEL_ID=<id> resolves
 * the same customer config {sr, fft, hop_in} to that model's own catalog rows. One source, three
 * builds selected by -DAIAEC_TEST_PRODUCT=1|2|3, all with -DAIAEC_TEST_CATALOG:
 *   1  DeepVQE-S: 16k/256 resolves at N=1 and N=8, hop_in 129 is refused at init.
 *   2  Align-ULCNet: 16k/256 resolves at N=1 and N=8 (packet graph, 9 inputs, frame_reset last signal).
 *   3  an id that names no model: every entry refuses with AIAEC_ERR_MODEL_ID, no graph path.
 * Builds 1 and 2 print the pool size of the same config; the Makefile asserts they differ
 * (a pool carved for one product is not the other's).
 * The single-model archives (a bound model the build does not contain) are test_single_model.c. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

#if AIAEC_TEST_PRODUCT != 3
/* init of cfg into a freshly allocated pool of exactly the get_mem_size() bytes. */
static int init_cfg(const AiaecConfig *cfg, AiaecInfo *info, void **pool_out, AiaecCtx **ctx_out) {
    StreamMemReq req;
    int rc = aiaec_get_mem_size(cfg, &req);
    *pool_out = NULL;
    *ctx_out = NULL;
    if (rc != AIAEC_OK) return rc;
    if (posix_memalign(pool_out, (size_t)req.alignment, (size_t)req.bytes) != 0) return -99;
    return aiaec_init(cfg, *pool_out, (size_t)req.bytes, ctx_out, info);
}

/* One zero N=8 (hop_in 1024) block through a real pre; the views of the first signal. */
static int pre_zero_n8(AiaecCtx *ctx, AiaecWork *work) {
    static const int16_t pcm[1024];
    static const float f[1024];
    static const AiaecAlignEvent ev[8];
    AiaecInput in;
    memset(&in, 0, sizeof(in));
    in.samples = 1024;
#if AIAEC_TEST_PRODUCT == 1
    in.kind = AIAEC_INPUT_DEEPVQE_S;
    in.u.deepvqe_s.mic_in = pcm; in.u.deepvqe_s.far_in = pcm;
#else
    in.kind = AIAEC_INPUT_ALIGN_ULCNET;
    in.u.align_ulcnet.residual_in = f; in.u.align_ulcnet.far_in = f;
    in.u.align_ulcnet.events = ev; in.u.align_ulcnet.event_count = 8;
    (void)pcm;
#endif
#if AIAEC_TEST_PRODUCT == 1
    (void)f; (void)ev;
#endif
    return aiaec_preprocess(ctx, &in, work);
}
#endif

int main(void) {
    AiaecConfig n1, n8, odd;
    StreamMemReq req;
    AiaecInfo info;
    AiaecWork work;
    AiaecCtx *ctx = NULL;
    void *pool = NULL;
    int rc;
    unsigned char small[64];

    n1.sr = 16000; n1.fft = 256; n1.hop_in = 128;
    n8 = n1; n8.hop_in = 1024;
    odd = n1; odd.hop_in = 129;

#if AIAEC_TEST_PRODUCT == 3
    CHECK(aiaec_get_mem_size(&n1, &req) == AIAEC_ERR_MODEL_ID, "unknown id: get_mem_size");
    ctx = (AiaecCtx *)small;
    CHECK(aiaec_init(&n1, small, sizeof(small), &ctx, &info) == AIAEC_ERR_MODEL_ID && ctx == NULL, "unknown id: init");
    CHECK(aiaec_get_mem_size(&odd, &req) == AIAEC_ERR_MODEL_ID, "unknown id is refused before hop_in");
#else
    rc = aiaec_get_mem_size(&n1, &req);
    CHECK(rc == AIAEC_OK, "N=1 get_mem_size rc=%d (%s)", rc, aiaec_status_text(rc));
    rc = init_cfg(&n1, &info, &pool, &ctx);
    CHECK(rc == AIAEC_OK && info.graph_path != NULL, "N=1 info rc=%d", rc);
    free(pool);
    CHECK(aiaec_get_mem_size(&odd, &req) == AIAEC_ERR_IN_HOP, "hop_in 129 must be refused at init");
    ctx = (AiaecCtx *)small;
    CHECK(aiaec_init(&odd, small, sizeof(small), &ctx, &info) == AIAEC_ERR_IN_HOP && ctx == NULL, "hop_in 129: init");
#if AIAEC_TEST_PRODUCT == 1
    rc = init_cfg(&n8, &info, &pool, &ctx);
    CHECK(rc == AIAEC_OK && info.trim_samples == 1024 && info.queue_frames == 7 && info.tail_zero_blocks == 1 &&
          pre_zero_n8(ctx, &work) == AIAEC_OK && work.input_count == 2 && work.inputs[0].h == 8u && work.inputs[1].h == 8u &&
          work.outputs[0].c == 8u, "deepvqe_s: hop_in 1024 is N=8 (2 signals over 8 frames)");
    CHECK(strstr(info.graph_path, "n8") != NULL, "deepvqe_s: N=8 picks its graph");
    free(pool);
    rc = init_cfg(&n1, &info, &pool, &ctx);
    CHECK(rc == AIAEC_OK && strstr(info.graph_path, "n1") != NULL, "deepvqe_s: N=1 picks its graph");
    free(pool);
#else
    rc = init_cfg(&n8, &info, &pool, &ctx);
    CHECK(rc == AIAEC_OK && info.trim_samples == 128 && info.queue_frames == 0 && info.tail_zero_blocks == 1 &&
          pre_zero_n8(ctx, &work) == AIAEC_OK && work.input_count == 5 && work.inputs[0].h == 8u &&
          work.inputs[4].w == 8u && work.inputs[4].h == 1u,
          "align_ulcnet: hop_in 1024 is N=8 (packet graph, 5 signals, frame_reset last)");
    CHECK(strstr(info.graph_path, "n8") != NULL, "align_ulcnet: N=8 picks its graph");
    free(pool);
    rc = init_cfg(&n1, &info, &pool, &ctx);
    CHECK(rc == AIAEC_OK && strstr(info.graph_path, "n1") != NULL, "align_ulcnet: N=1 picks its graph");
    free(pool);
#endif
    rc = aiaec_get_mem_size(&n1, &req);
    printf("pool product=%d bytes=%llu\n", (int)AIAEC_PRODUCT_MODEL_ID, (unsigned long long)req.bytes);
#endif
    (void)n8; (void)small; (void)rc; (void)req; (void)info; (void)work; (void)ctx; (void)pool;

    if (g_fail) { printf("test_aiaec_product[%d]: FAIL\n", AIAEC_TEST_PRODUCT); return 1; }
    printf("test_aiaec_product[%d]: PASS (AIAEC_PRODUCT_MODEL_ID=%d)\n", AIAEC_TEST_PRODUCT, (int)AIAEC_PRODUCT_MODEL_ID);
    return 0;
}
