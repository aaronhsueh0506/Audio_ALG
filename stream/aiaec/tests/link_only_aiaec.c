/* Only includes aiaec.h and links against the already-built libaiaec.a plus the model stream
 * archives (never recompiling any core/stream source) -- confirms the shipped archive actually
 * exports the customer entry points, constructs one registered row (DeepVQE-S 16k/256 N=1) and
 * size-checks the 16k/512 row. The archive's product is the default binding (DeepVQE-S in a
 * build that contains both models). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"

/* The customer header carries the memory-requirement type only; the graph/device port check is the samples'. */
#ifdef STREAM_COMMON_BIND_H
#error "aiaec.h must not include stream_bind.h"
#endif

int main(void) {
    AiaecConfig cfg;
    StreamMemReq req;
    int rc, status = 0;

    cfg.sr = 16000; cfg.fft = 512; cfg.hop_in = 256;
    rc = aiaec_get_mem_size(&cfg, &req);
    printf("link_only_aiaec: aiaec_get_mem_size(16k/512) rc=%d (%s)\n", rc, aiaec_status_text(rc));
    if (rc != AIAEC_OK) status = 1;

    /* A registered grid (DeepVQE-S, 16kHz/256, N=1) against THIS archive. */
    cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = 128;
    rc = aiaec_get_mem_size(&cfg, &req);
    if (rc != AIAEC_OK) {
        fprintf(stderr, "link_only_aiaec: FAIL shipped 16kHz/256/N=1 row: rc=%d (%s)\n", rc, aiaec_status_text(rc));
        status = 1;
    } else {
        void *pool = NULL, *pool_f = NULL;
        AiaecCtx *ctx = NULL;
        AiaecFreqCtx *cf = NULL;
        StreamMemReq rf;
        if (aiaec_get_mem_size_freq(&cfg, &rf) != AIAEC_OK ||
            posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool ||
            posix_memalign(&pool_f, (size_t)rf.alignment, (size_t)rf.bytes) != 0 || !pool_f) {
            fprintf(stderr, "link_only_aiaec: FAIL shipped row pool allocation\n");
            status = 1;
        } else {
            rc = aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx);
            if (rc != AIAEC_OK || !ctx) {
                fprintf(stderr, "link_only_aiaec: FAIL shipped row init rc=%d (%s)\n", rc, aiaec_status_text(rc));
                status = 1;
            } else {
                StreamWork work;
                AiaecInput in;
                int16_t mic[128] = {0}, far[128] = {0}, out[128];
                printf("link_only_aiaec: shipped 16kHz/256/N=1 row constructs\n");
                memset(&in, 0, sizeof in);
                in.kind = AIAEC_INPUT_DEEPVQE_S; in.samples = 128;
                in.u.deepvqe_s.mic_in = mic; in.u.deepvqe_s.far_in = far;
                if (aiaec_preprocess(ctx, &in, &work) != AIAEC_OK || work.input_count != 2 ||
                    work.output_count != 1 || work.inputs[0].w != 129u || work.outputs[0].w != 18u ||
                    aiaec_postprocess(ctx, 0, out) != AIAEC_OK) {
                    fprintf(stderr, "link_only_aiaec: FAIL shipped row block round trip\n");
                    status = 1;
                }
            }
            /* the _freq family answers for the same row */
            if (aiaec_init_freq(&cfg, pool_f, (size_t)rf.bytes, &cf) != AIAEC_OK) {
                fprintf(stderr, "link_only_aiaec: FAIL shipped row _freq entries\n");
                status = 1;
            } else {
                printf("link_only_aiaec: shipped row _freq pool %llu bytes\n", (unsigned long long)rf.bytes);
            }
        }
        free(pool); free(pool_f);
    }

    printf("link_only_aiaec: %s (aiaec_* symbols link and run)\n", status ? "FAIL" : "PASS");
    return status;
}
