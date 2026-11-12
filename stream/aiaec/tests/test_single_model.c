/* Single-model build proof (aiaec.c's AIAEC_ENABLE_DEEPVQE_S / AIAEC_ENABLE_ALIGN_ULCNET, the
 * Makefile's AIAEC_MODEL knob), built and run once per variant (deepvqe-only, align-only) in two
 * forms, driven by the SAME compile-time flags aiaec.c sees:
 *
 *   default product (the enabled model): a normal SUBMIT hop and a forced-bypass hop through the real
 *     public block API, one hop through each of the _spec and _freq families, the info, and the
 *     work views of the first block (counts and shapes).
 *   -DAIAEC_TEST_OTHER_BOUND=1 with -DAIAEC_PRODUCT_MODEL_ID=<the model this build does NOT contain>:
 *     every entry refuses with AIAEC_ERR_MODEL_ID.
 *
 * Compiled with -DAIAEC_TEST_CATALOG so the enabled model's row has a graph path (the production
 * catalog has none for the rows used here). The nm -u proof that the other model's symbols are
 * absent is the Makefile's, not this program's. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"

static int g_fail = 0;
#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } while (0)
#define CHECK(cond, ...) do { if (!(cond)) FAIL(__VA_ARGS__); } while (0)

#ifdef AIAEC_TEST_OTHER_BOUND
static int check_bound_model_refused(void) {
    AiaecConfig cfg = { 16000, 256, 128 };
    StreamMemReq req;
    AiaecInfo info;
    AiaecCtx *ctx = (AiaecCtx *)&req;
    AiaecSpecCtx *cs = (AiaecSpecCtx *)&req;
    AiaecFreqCtx *cf = (AiaecFreqCtx *)&req;
    unsigned char pool[64];

    CHECK(aiaec_get_mem_size(&cfg, &req) == AIAEC_ERR_MODEL_ID, "get_mem_size must be AIAEC_ERR_MODEL_ID");
    CHECK(aiaec_init(&cfg, pool, sizeof(pool), &ctx, &info) == AIAEC_ERR_MODEL_ID && ctx == NULL, "init");
    CHECK(aiaec_get_mem_size_spec(&cfg, &req) == AIAEC_ERR_MODEL_ID && aiaec_get_mem_size_freq(&cfg, &req) == AIAEC_ERR_MODEL_ID,
          "get_mem_size_spec/_freq must be AIAEC_ERR_MODEL_ID");
    CHECK(aiaec_init_spec(&cfg, pool, sizeof(pool), &cs, &info) == AIAEC_ERR_MODEL_ID && cs == NULL, "init_spec");
    CHECK(aiaec_init_freq(&cfg, pool, sizeof(pool), &cf, &info) == AIAEC_ERR_MODEL_ID && cf == NULL, "init_freq");
    printf("test_single_model: bound model %d absent from this build -- every entry refused -- %s\n",
           (int)AIAEC_PRODUCT_MODEL_ID, g_fail ? "FAIL" : "PASS");
    return g_fail;
}
#else
static int run_roundtrip(void) {
    AiaecConfig cfg = { 16000, 256, 128 };
    StreamMemReq req;
    AiaecInfo info;
    AiaecCtx *ctx = NULL;
    AiaecInput input;
    AiaecWork work;
    void *pool = NULL;
    static int16_t mic[128], far16[128], pcm_out[128];
    static float resid[128], farf[128];
    AiaecAlignEvent ev;
    const int deepvqe = (AIAEC_PRODUCT_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S);
    int k, rc;

    if (aiaec_get_mem_size(&cfg, &req) != AIAEC_OK) { FAIL("aiaec_get_mem_size failed -- is AIAEC_TEST_CATALOG wired?"); return 1; }
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0) { FAIL("pool alloc failed"); return 1; }
    rc = aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx, &info);
    if (rc != AIAEC_OK || !ctx) { FAIL("aiaec_init failed rc=%d", rc); free(pool); return 1; }

    CHECK(info.graph_path != NULL && info.trim_samples == 128 && info.queue_frames == 0 && info.tail_zero_blocks == 1, "info");
    for (k = 0; k < 128; ++k) {
        mic[k] = (int16_t)(k * 3 - 50); far16[k] = (int16_t)(k * 5 - 80);
        resid[k] = 0.001f * (float)(k * 5 - 48); farf[k] = 0.001f * (float)(k * 7 - 44);
    }
    ev.changed = 0; ev.solid = 0;
    memset(&input, 0, sizeof(input));
    input.samples = 128;
    if (deepvqe) {
        input.kind = AIAEC_INPUT_DEEPVQE_S;
        input.u.deepvqe_s.mic_in = mic;
        input.u.deepvqe_s.far_in = far16;
    } else {
        input.kind = AIAEC_INPUT_ALIGN_ULCNET;
        input.u.align_ulcnet.residual_in = resid;
        input.u.align_ulcnet.far_in = farf;
        input.u.align_ulcnet.events = &ev;
        input.u.align_ulcnet.event_count = 1;
    }

    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_OK, "preprocess (normal) rc=%d", rc);
    CHECK(work.action == AIAEC_SUBMIT && work.input_count == (deepvqe ? 2 : 4) && work.output_count == 1 &&
          work.inputs[0].w == 129u && work.inputs[0].h == 1u && work.inputs[0].c == (deepvqe ? 2u : 1u) &&
          work.outputs[0].elements == (deepvqe ? 129u * 18u : 2u * 129u),
          "work views %d/%d, first signal [%u,%u,%u,%u]", work.input_count, work.output_count, work.inputs[0].b,
          work.inputs[0].c, work.inputs[0].h, work.inputs[0].w);
    if (work.action == AIAEC_SUBMIT)
        for (k = 0; k < (int)work.outputs[0].elements; ++k) ((float *)work.outputs[0].data)[k] = 0.01f * (float)k;
    CHECK(deepvqe ? work.action == AIAEC_SUBMIT : 1, "DeepVQE-S expects SUBMIT");
    rc = aiaec_postprocess(ctx, 0, pcm_out);
    CHECK(rc == AIAEC_OK, "postprocess (normal) rc=%d", rc);

    rc = aiaec_preprocess(ctx, &input, &work);
    CHECK(rc == AIAEC_OK, "preprocess (bypass setup) rc=%d", rc);
    rc = aiaec_postprocess(ctx, -1, pcm_out);   /* infer_status != 0: forced bypass */
    CHECK(rc == AIAEC_OK, "postprocess (bypass) rc=%d", rc);

    free(pool);

    /* _spec and _freq: one normal hop each, on their own pools */
    {
        StreamMemReq rs, rf;
        AiaecSpecCtx *cs;
        AiaecFreqCtx *cf;
        AiaecFreqInput fin;
        static float zeros[129], o_re[129], o_im[129];
        void *ps = NULL, *pf = NULL;

        if (aiaec_get_mem_size_spec(&cfg, &rs) != AIAEC_OK || aiaec_get_mem_size_freq(&cfg, &rf) != AIAEC_OK ||
            posix_memalign(&ps, 16, (size_t)rs.bytes) != 0 || posix_memalign(&pf, 16, (size_t)rf.bytes) != 0) {
            FAIL("spec/freq pool setup failed"); return 1;
        }
        if (aiaec_init_spec(&cfg, ps, (size_t)rs.bytes, &cs, &info) != AIAEC_OK ||
            aiaec_init_freq(&cfg, pf, (size_t)rf.bytes, &cf, &info) != AIAEC_OK || !cs || !cf) {
            FAIL("spec/freq init failed"); return 1;
        }
        rc = aiaec_preprocess_spec(cs, &input, &work);
        CHECK(rc == AIAEC_OK, "spec preprocess rc=%d", rc);
        if (work.action == AIAEC_SUBMIT)
            for (k = 0; k < (int)work.outputs[0].elements; ++k) ((float *)work.outputs[0].data)[k] = 0.01f * (float)k;
        CHECK(aiaec_postprocess_spec(cs, 0, o_re, o_im) == AIAEC_OK, "spec postprocess");

        memset(&fin, 0, sizeof(fin));
        fin.frames = 1;
        if (deepvqe) {
            fin.kind = AIAEC_INPUT_DEEPVQE_S;
            fin.u.deepvqe_s.mic_re = zeros; fin.u.deepvqe_s.mic_im = zeros;
            fin.u.deepvqe_s.far_re = zeros; fin.u.deepvqe_s.far_im = zeros;
        } else {
            fin.kind = AIAEC_INPUT_ALIGN_ULCNET;
            fin.u.align_ulcnet.err_re = zeros; fin.u.align_ulcnet.err_im = zeros;
            fin.u.align_ulcnet.far_re = zeros; fin.u.align_ulcnet.far_im = zeros;
            fin.u.align_ulcnet.events = &ev; fin.u.align_ulcnet.event_count = 1;
        }
        rc = aiaec_preprocess_freq(cf, &fin, &work);
        CHECK(rc == AIAEC_OK, "freq preprocess rc=%d", rc);
        if (work.action == AIAEC_SUBMIT)
            for (k = 0; k < (int)work.outputs[0].elements; ++k) ((float *)work.outputs[0].data)[k] = 0.01f * (float)k;
        CHECK(aiaec_postprocess_freq(cf, 0, o_re, o_im) == AIAEC_OK, "freq postprocess");
        free(ps); free(pf);
    }
    printf("test_single_model: %s round trip (block normal + bypass, spec, freq), info, work views -- %s\n",
           deepvqe ? "DeepVQE-S" : "Align-ULCNet", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
#endif

int main(void) {
#ifdef AIAEC_TEST_OTHER_BOUND
    if (check_bound_model_refused() != 0) return 1;
#else
    if (run_roundtrip() != 0) return 1;
#endif
    printf("test_single_model: ALL PASS\n");
    return 0;
}
