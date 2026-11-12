/* Only includes aiaec.h and links against the already-built libaiaec.a plus the model stream
 * archives (never recompiling any core/stream source) -- confirms the shipped archive actually
 * exports the customer entry points and constructs its one delivered row. The archive's product is
 * the default binding (DeepVQE-S in a build that contains both models). */

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

    cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = 128;
    rc = aiaec_get_mem_size(&cfg, &req);
    printf("link_only_aiaec: aiaec_get_mem_size(16k/256) rc=%d (%s)\n", rc, aiaec_status_text(rc));

    /* The one row with a delivered graph (DeepVQE-S, 16kHz/512, N=1), against THIS production
     * archive (no placeholder paths anywhere). */
    cfg.sr = 16000; cfg.fft = 512; cfg.hop_in = 256;
    rc = aiaec_get_mem_size(&cfg, &req);
    if (rc != AIAEC_OK) {
        fprintf(stderr, "link_only_aiaec: FAIL shipped 16kHz/512/N=1 row: rc=%d (%s)\n", rc, aiaec_status_text(rc));
        status = 1;
    } else {
        void *pool = NULL, *pool_s = NULL, *pool_f = NULL;
        AiaecCtx *ctx = NULL;
        AiaecSpecCtx *cs = NULL;
        AiaecFreqCtx *cf = NULL;
        AiaecInfo info, is, if_;
        StreamMemReq rs, rf;
        if (aiaec_get_mem_size_spec(&cfg, &rs) != AIAEC_OK || aiaec_get_mem_size_freq(&cfg, &rf) != AIAEC_OK ||
            posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool ||
            posix_memalign(&pool_s, (size_t)rs.alignment, (size_t)rs.bytes) != 0 || !pool_s ||
            posix_memalign(&pool_f, (size_t)rf.alignment, (size_t)rf.bytes) != 0 || !pool_f) {
            fprintf(stderr, "link_only_aiaec: FAIL shipped row pool allocation\n");
            status = 1;
        } else {
            rc = aiaec_init(&cfg, pool, (size_t)req.bytes, &ctx, &info);
            if (rc != AIAEC_OK || !ctx || info.graph_path == NULL || info.trim_samples != 256 || info.queue_frames != 0 ||
                info.tail_zero_blocks != 1) {
                fprintf(stderr, "link_only_aiaec: FAIL shipped row init rc=%d (%s)\n", rc, aiaec_status_text(rc));
                status = 1;
            } else {
                AiaecWork work;
                AiaecInput in;
                int16_t mic[256] = {0}, far[256] = {0}, out[256];
                printf("link_only_aiaec: shipped 16kHz/512/N=1 row constructs via %s\n", info.graph_path);
                memset(&in, 0, sizeof in);
                in.kind = AIAEC_INPUT_DEEPVQE_S; in.samples = 256;
                in.u.deepvqe_s.mic_in = mic; in.u.deepvqe_s.far_in = far;
                if (info.tail_zero_blocks < 0 || aiaec_preprocess(ctx, &in, &work) != AIAEC_OK || work.input_count != 2 ||
                    work.output_count != 1 || work.inputs[0].w != 257u || work.outputs[0].w != 18u ||
                    aiaec_postprocess(ctx, 0, out) != AIAEC_OK) {
                    fprintf(stderr, "link_only_aiaec: FAIL shipped row block round trip\n");
                    status = 1;
                }
            }
            /* the other two families answer for the same row */
            if (aiaec_init_spec(&cfg, pool_s, (size_t)rs.bytes, &cs, &is) != AIAEC_OK ||
                aiaec_init_freq(&cfg, pool_f, (size_t)rf.bytes, &cf, &if_) != AIAEC_OK ||
                is.trim_samples != if_.trim_samples || is.queue_frames != if_.queue_frames || is.tail_zero_blocks != 1 ||
                if_.graph_path != info.graph_path || if_.trim_samples != 256) {
                fprintf(stderr, "link_only_aiaec: FAIL shipped row _spec/_freq entries\n");
                status = 1;
            } else {
                printf("link_only_aiaec: shipped row _spec/_freq pools %llu/%llu bytes, trim %d samples, queue %d frames\n",
                       (unsigned long long)rs.bytes, (unsigned long long)rf.bytes, is.trim_samples, is.queue_frames);
            }
        }
        free(pool); free(pool_s); free(pool_f);
    }

    printf("link_only_aiaec: %s (aiaec_* symbols link and run)\n", status ? "FAIL" : "PASS");
    return status;
}
