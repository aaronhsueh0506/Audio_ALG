/* aiaec.c bound with -DAIAEC_PRODUCT_MODEL_ID resolves {sr, fft, hop_in} to that model's rows (-DAIAEC_TEST_PRODUCT=1
 * DeepVQE-S, 2 Align-ULCNet, 3 unknown id: AIAEC_ERR_MODEL_ID); N=8 inits but has no graph_io row, N=65 is refused;
 * builds 1 and 2 print their pool sizes, which the Makefile asserts differ. test_single_model.c: single-model archives. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"
#include "graph_io.h"
#include "graph_io_alignulcnet.h"
#include "graph_io_deepvqes.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

#if AIAEC_TEST_PRODUCT == 1
#define FIND(sr, fft, n, mc) graph_io_deepvqes_find(sr, fft, n, mc)
#elif AIAEC_TEST_PRODUCT == 2
#define FIND(sr, fft, n, mc) graph_io_alignulcnet_find(sr, fft, n, mc)
#endif

#if AIAEC_TEST_PRODUCT != 3
/* init of cfg into a freshly allocated pool of exactly the get_mem_size() bytes. */
static int init_cfg(const AiaecConfig *cfg, void **pool_out, AiaecCtx **ctx_out) {
    StreamMemReq req;
    int rc = aiaec_get_mem_size(cfg, &req);
    *pool_out = NULL;
    *ctx_out = NULL;
    if (rc != AIAEC_OK) return rc;
    if (posix_memalign(pool_out, (size_t)req.alignment, (size_t)req.bytes) != 0) return -99;
    return aiaec_init(cfg, *pool_out, (size_t)req.bytes, ctx_out);
}

/* One zero 16k/512 N=4 (hop_in 1024) block through a real pre; the views of the first signal. */
static int pre_zero_n4(AiaecCtx *ctx, StreamWork *work) {
    static const int16_t pcm[1024];
    static const float f[1024];
    static const AiaecAlignEvent ev[4];
    AiaecInput in;
    memset(&in, 0, sizeof(in));
    in.samples = 1024;
#if AIAEC_TEST_PRODUCT == 1
    in.kind = AIAEC_INPUT_DEEPVQE_S;
    in.u.deepvqe_s.mic_in = pcm; in.u.deepvqe_s.far_in = pcm;
    (void)f; (void)ev;
#else
    in.kind = AIAEC_INPUT_ALIGN_ULCNET;
    in.u.align_ulcnet.residual_in = f; in.u.align_ulcnet.far_in = f;
    in.u.align_ulcnet.events = ev; in.u.align_ulcnet.event_count = 4;
    (void)pcm;
#endif
    return aiaec_preprocess(ctx, &in, work);
}

#endif

int main(void) {
    AiaecConfig n1, n4, n8, n65, odd;
    StreamMemReq req;
    StreamWork work;
    AiaecCtx *ctx = NULL;
    void *pool = NULL;
    int rc;
    unsigned char small[64];

    n1.sr = 16000; n1.fft = 256; n1.hop_in = 128;
    n4 = n1; n4.fft = 512; n4.hop_in = 1024;
    n8 = n1; n8.hop_in = 1024;
    n65 = n1; n65.hop_in = 65 * 128;
    odd = n1; odd.hop_in = 129;

#if AIAEC_TEST_PRODUCT == 3
    CHECK(aiaec_get_mem_size(&n1, &req) == AIAEC_ERR_MODEL_ID, "unknown id: get_mem_size");
    ctx = (AiaecCtx *)small;
    CHECK(aiaec_init(&n1, small, sizeof(small), &ctx) == AIAEC_ERR_MODEL_ID && ctx == NULL, "unknown id: init");
    CHECK(aiaec_get_mem_size(&odd, &req) == AIAEC_ERR_MODEL_ID, "unknown id is refused before hop_in");
    (void)n4; (void)n8; (void)n65; (void)work; (void)pool; (void)rc;
#else
    rc = aiaec_get_mem_size(&n1, &req);
    CHECK(rc == AIAEC_OK, "N=1 get_mem_size rc=%d (%s)", rc, aiaec_status_text(rc));
    rc = init_cfg(&n1, &pool, &ctx);
    CHECK(rc == AIAEC_OK && ctx != NULL, "N=1 init rc=%d", rc);
    free(pool);
    CHECK(aiaec_get_mem_size(&odd, &req) == AIAEC_ERR_IN_HOP, "hop_in 129 must be refused at init");
    ctx = (AiaecCtx *)small;
    CHECK(aiaec_init(&odd, small, sizeof(small), &ctx) == AIAEC_ERR_IN_HOP && ctx == NULL, "hop_in 129: init");
    /* N=8 is within the model's range: the library accepts it, only the graph_io table has no row */
    rc = init_cfg(&n8, &pool, &ctx);
    CHECK(rc == AIAEC_OK && ctx != NULL, "16k/256 N=8: init rc=%d (%s)", rc, aiaec_status_text(rc));
    free(pool);
    {
        GraphIoConfig mc;
        CHECK(FIND(16000, 256, 8, &mc) == GRAPH_IO_ERR_NO_ROW, "16k/256 N=8: graph_io must have no row");
    }
    /* N=65 is beyond the model stream's range */
    CHECK(aiaec_get_mem_size(&n65, &req) == AIAEC_ERR_FRAMES_PER_CALL, "16k/256 hop_in 8320 (N=65) must be refused");
    ctx = (AiaecCtx *)small;
    CHECK(aiaec_init(&n65, small, sizeof(small), &ctx) == AIAEC_ERR_FRAMES_PER_CALL && ctx == NULL, "16k/256 N=65: init");
#if AIAEC_TEST_PRODUCT == 1
    rc = init_cfg(&n4, &pool, &ctx);
    CHECK(rc == AIAEC_OK && pre_zero_n4(ctx, &work) == AIAEC_OK && work.input_count == 2 && work.inputs[0].h == 4u &&
          work.inputs[1].h == 4u && work.inputs[0].w == 257u && work.outputs[0].c == 4u,
          "deepvqe_s: 16k/512 hop_in 1024 is N=4 (2 signals over 4 frames)");
    free(pool);
#else
    rc = init_cfg(&n4, &pool, &ctx);
    CHECK(rc == AIAEC_OK && pre_zero_n4(ctx, &work) == AIAEC_OK && work.input_count == 5 && work.inputs[0].h == 4u &&
          work.inputs[0].w == 257u && work.inputs[4].w == 4u && work.inputs[4].h == 1u,
          "align_ulcnet: 16k/512 hop_in 1024 is N=4 (packet graph, 5 signals, frame_reset last)");
    free(pool);
#endif
    rc = aiaec_get_mem_size(&n1, &req);
    printf("pool product=%d bytes=%llu\n", (int)AIAEC_PRODUCT_MODEL_ID, (unsigned long long)req.bytes);
#endif
    (void)small; (void)req; (void)ctx;

    if (g_fail) { printf("test_aiaec_product[%d]: FAIL\n", AIAEC_TEST_PRODUCT); return 1; }
    printf("test_aiaec_product[%d]: PASS (AIAEC_PRODUCT_MODEL_ID=%d)\n", AIAEC_TEST_PRODUCT, (int)AIAEC_PRODUCT_MODEL_ID);
    return 0;
}
