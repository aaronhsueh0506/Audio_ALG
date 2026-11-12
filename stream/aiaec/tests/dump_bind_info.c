/* Prints, for one {sr, fft, hop_in} of the product this binary is built for, the init result and the AiaecWork
 * views of the first real pre call, one line per fact, for AIAEC/tests/test_aiaec_bind_info_pin.py to compare
 * with the ONNX export. Built by `make bind-dumpers` with -DAIAEC_TEST_CATALOG (placeholder graph paths, so the
 * rows without a delivered graph still resolve) and -DAIAEC_PRODUCT_MODEL_ID=<id>.
 *
 *   dump_bind_info <sr> <fft> <hop_in> [family [event_hop]]
 *       family 0 PCM (default), 1 _spec, 2 _freq; event_hop >= 0 (Align-ULCNet) marks that model hop of the block
 *       as an alignment change; all-zero input otherwise.
 *     rc <int>                          first refusal of aiaec_get_mem_size*() / aiaec_init*() / aiaec_preprocess*(),
 *                                       0 when all succeed (a refused config prints this line only)
 *     graph <path>                      AiaecInfo.graph_path
 *     info trim <int> queue <int> tail <int>      AiaecInfo trim_samples / queue_frames / tail_zero_blocks
 *     work action <int> inputs <int> outputs <int>   AiaecWork.action (0 BYPASS, 1 SUBMIT), input_count, output_count
 *     in <i> <b> <c> <h> <w> <elements>   AiaecWork.inputs[i], the signals in graph order (none on a BYPASS block)
 *     out <i> <b> <c> <h> <w> <elements>  AiaecWork.outputs[i], the heads (none on a BYPASS block)
 *     frame_reset <v0> ... <vN-1>         Align-ULCNet at N > 1 only: the last signal's N control values
 *
 * Only the signal and head views a sample binds are printed: the graph's recurrent-state ports are not part of
 * AiaecWork, and the library has no record of them. N and hop_in are not printed: they are the arguments. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"

int main(int argc, char **argv) {
    AiaecConfig cfg;
    AiaecInfo info;
    AiaecWork work;
    StreamMemReq req;
    AiaecCtx *pcm = NULL;
    AiaecSpecCtx *spec = NULL;
    AiaecFreqCtx *freq = NULL;
    AiaecInput in;
    AiaecFreqInput fin;
    AiaecAlignEvent *events = NULL;
    void *pool = NULL;
    int16_t *s16 = NULL;
    float *f32 = NULL;
    int rc, i, family, event_hop = -1, n;

    if (argc < 4 || argc > 6) return 2;
    cfg.sr = atoi(argv[1]); cfg.fft = atoi(argv[2]); cfg.hop_in = atoi(argv[3]);
    family = argc >= 5 ? atoi(argv[4]) : 0;
    if (argc == 6) event_hop = atoi(argv[5]);
    if (family < 0 || family > 2) return 2;

    rc = family == 0 ? aiaec_get_mem_size(&cfg, &req)
       : family == 1 ? aiaec_get_mem_size_spec(&cfg, &req) : aiaec_get_mem_size_freq(&cfg, &req);
    if (rc == AIAEC_OK) {
        if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) return 2;
        rc = family == 0 ? aiaec_init(&cfg, pool, (size_t)req.bytes, &pcm, &info)
           : family == 1 ? aiaec_init_spec(&cfg, pool, (size_t)req.bytes, &spec, &info)
                         : aiaec_init_freq(&cfg, pool, (size_t)req.bytes, &freq, &info);
    }
    if (rc != AIAEC_OK) { printf("rc %d\n", rc); free(pool); return 0; }

    n = cfg.hop_in / (cfg.fft / 2);
    events = (AiaecAlignEvent *)calloc((size_t)n, sizeof(*events));
    s16 = (int16_t *)calloc((size_t)cfg.hop_in, sizeof(*s16));
    f32 = (float *)calloc((size_t)(cfg.hop_in > n * (cfg.fft / 2 + 1) ? cfg.hop_in : n * (cfg.fft / 2 + 1)), sizeof(*f32));
    if (!events || !s16 || !f32) return 2;
    if (event_hop >= 0 && event_hop < n) events[event_hop].changed = 1;

    memset(&work, 0x5a, sizeof(work));   /* stale bytes must not pass for a zeroed BYPASS block */
    memset(&in, 0, sizeof(in));
    memset(&fin, 0, sizeof(fin));
#if AIAEC_PRODUCT_MODEL_ID == 10
    in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.samples = cfg.hop_in;
    in.u.align_ulcnet.residual_in = f32; in.u.align_ulcnet.far_in = f32;
    in.u.align_ulcnet.events = events; in.u.align_ulcnet.event_count = n;
    fin.kind = AIAEC_INPUT_ALIGN_ULCNET; fin.frames = n;
    fin.u.align_ulcnet.err_re = fin.u.align_ulcnet.err_im = fin.u.align_ulcnet.far_re = fin.u.align_ulcnet.far_im = f32;
    fin.u.align_ulcnet.events = events; fin.u.align_ulcnet.event_count = n;
#else
    in.kind = AIAEC_INPUT_DEEPVQE_S; in.samples = cfg.hop_in;
    in.u.deepvqe_s.mic_in = s16; in.u.deepvqe_s.far_in = s16;
    fin.kind = AIAEC_INPUT_DEEPVQE_S; fin.frames = n;
    fin.u.deepvqe_s.mic_re = fin.u.deepvqe_s.mic_im = fin.u.deepvqe_s.far_re = fin.u.deepvqe_s.far_im = f32;
#endif
    rc = family == 0 ? aiaec_preprocess(pcm, &in, &work)
       : family == 1 ? aiaec_preprocess_spec(spec, &in, &work) : aiaec_preprocess_freq(freq, &fin, &work);
    printf("rc %d\n", rc);
    if (rc != AIAEC_OK) goto done;
    printf("graph %s\n", info.graph_path);
    printf("info trim %d queue %d tail %d\n", info.trim_samples, info.queue_frames, info.tail_zero_blocks);
    printf("work action %d inputs %d outputs %d\n", (int)work.action, work.input_count, work.output_count);
    for (i = 0; i < work.input_count; ++i)
        printf("in %d %u %u %u %u %zu\n", i, work.inputs[i].b, work.inputs[i].c, work.inputs[i].h, work.inputs[i].w,
               work.inputs[i].elements);
    for (i = 0; i < work.output_count; ++i)
        printf("out %d %u %u %u %u %zu\n", i, work.outputs[i].b, work.outputs[i].c, work.outputs[i].h, work.outputs[i].w,
               work.outputs[i].elements);
#if AIAEC_PRODUCT_MODEL_ID == 10
    if (work.input_count == 5) {
        printf("frame_reset");
        for (i = 0; i < (int)work.inputs[4].elements; ++i) printf(" %g", (double)work.inputs[4].data[i]);
        printf("\n");
    }
#endif
done:
    free(pool); free(events); free(s16); free(f32);
    return 0;
}
