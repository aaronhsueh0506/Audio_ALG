/* Prints, for one {sr, fft, hop_in} of the variant this binary is built for (full, or lite under
 * -DDFN2_MASK_ONLY), the init result and the StreamWork views of the first block with a job (the blocks before it, which
 * carry no job on a grid with a mask lookahead, are closed unrun), one line per fact, for the
 * Python pins and the sample model-config check to compare against the exporter. Built by `make work-dumper`.
 *
 *   dump_work <sr> <fft> <hop_in> [family]        family 0 PCM (default), 2 _freq (1 was the retired _spec)
 *     rc <int>                                    status of ainr_get_mem_size*() / ainr_init*() / the
 *                                                 preprocess calls (a refused config prints this line only)
 *     graph <path>                                AinrInfo.graph_path
 *     info trim <int> queue <int> tail <int>      AinrInfo trim_samples / queue_frames / tail_zero_blocks
 *     counts inputs <int> outputs <int>           StreamWork input_count / output_count
 *     in <i> <b> <c> <h> <w> <elements>           StreamWork.inputs[i], signal inputs only
 *     out <i> <b> <c> <h> <w> <elements>          StreamWork.outputs[i], heads only
 *
 * The recurrent state is the accelerator's and is not in the work. N and hop_in are the arguments. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"

/* An ERB file of the grid's size: zero forward table, inverse with band 0 owning every bin (init checks the
 * header, the length and that the inverse is a partition of unity). */
static int write_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb * 2u, i;
    int32_t hdr[2];
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    if (fwrite(hdr, sizeof(int32_t), 2, fp) != 2) { fclose(fp); return -1; }
    for (i = 0; i < elems; ++i) {
        float v = (i >= elems / 2u && (i - elems / 2u) % (size_t)n_erb == 0u) ? 1.0f : 0.0f;
        if (fwrite(&v, sizeof(float), 1, fp) != 1) { fclose(fp); return -1; }
    }
    return fclose(fp);
}

int main(int argc, char **argv) {
    AinrConfig cfg;
    AinrInfo info;
    StreamWork work;
    StreamMemReq req;
    AinrCtx *pcm = NULL;
    AinrFreqCtx *freq = NULL;
    void *pool = NULL;
    int16_t *pcm_in = NULL, *pcm_out = NULL;
    float *zeros = NULL;
    char erb[96];
    int rc, i, family, n_erb, hop, bins;

    if (argc != 4 && argc != 5) return 2;
    cfg.sr = atoi(argv[1]); cfg.fft = atoi(argv[2]); cfg.hop_in = atoi(argv[3]);
    family = argc == 5 ? atoi(argv[4]) : 0;
    if (family != 0 && family != 2) return 2;

    rc = family == 0 ? ainr_get_mem_size(&cfg, &req) : ainr_get_mem_size_freq(&cfg, &req);
    if (rc == AINR_OK) {
        hop = cfg.fft / 2;
        bins = hop + 1;
        n_erb = cfg.sr == 16000 ? 24 : 32;
        snprintf(erb, sizeof(erb), "/tmp/ainr_dump_work_%d.bin", (int)getpid());
        if (write_erb(erb, bins, n_erb) != 0) return 2;
        if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) return 2;
        rc = family == 0 ? ainr_init_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &pcm, &info)
                         : ainr_init_freq_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &freq, &info);
        unlink(erb);
    }
    if (rc == AINR_OK) {
        pcm_in = (int16_t *)calloc((size_t)cfg.hop_in, sizeof(int16_t));
        pcm_out = (int16_t *)calloc((size_t)cfg.hop_in, sizeof(int16_t));
        zeros = (float *)calloc(3u * (size_t)(cfg.hop_in / hop) * (size_t)bins, sizeof(float));   /* input, out_re, out_im */
        if (!pcm_in || !pcm_out || !zeros) return 2;
        for (i = 0; rc == AINR_OK; ++i) {   /* a block without a job is closed unrun; its status is not read */
            rc = family == 0 ? ainr_preprocess(pcm, pcm_in, &work)
                             : ainr_preprocess_freq(freq, zeros, zeros, zeros, zeros, &work);
            if (rc != AINR_OK || work.action == STREAM_WORK_SUBMIT) break;
            rc = family == 0 ? ainr_postprocess(pcm, STREAM_RUN_NOT_STARTED, pcm_out)
                             : ainr_postprocess_freq(freq, STREAM_RUN_NOT_STARTED, zeros + (size_t)(cfg.hop_in / hop) * (size_t)bins,
                                                     zeros + 2u * (size_t)(cfg.hop_in / hop) * (size_t)bins);
            if (i >= 8) rc = AINR_ERR_SEQUENCE;
        }
    }
    printf("rc %d\n", rc);
    if (rc != AINR_OK) { free(pool); free(pcm_in); free(pcm_out); free(zeros); return 0; }
    printf("graph %s\n", info.graph_path);
    printf("info trim %d queue %d tail %d\n", info.trim_samples, info.queue_frames, info.tail_zero_blocks);
    printf("counts inputs %d outputs %d\n", work.input_count, work.output_count);
    for (i = 0; i < work.input_count; ++i)
        printf("in %d %u %u %u %u %zu\n", i, work.inputs[i].b, work.inputs[i].c, work.inputs[i].h, work.inputs[i].w,
               work.inputs[i].elements);
    for (i = 0; i < work.output_count; ++i)
        printf("out %d %u %u %u %u %zu\n", i, work.outputs[i].b, work.outputs[i].c, work.outputs[i].h, work.outputs[i].w,
               work.outputs[i].elements);
    free(pool); free(pcm_in); free(pcm_out); free(zeros);
    return 0;
}
