/* Prints, for one {sr, fft, hop_in} of the variant this binary is built for (full, or lite under
 * -DDFN2_MASK_ONLY), the init result and the AinrWork views of the first block, one line per fact, for the
 * Python pins and the sample model-config check to compare against the exporter. Built by `make work-dumper`.
 *
 *   dump_work <sr> <fft> <hop_in> [family]        family 0 PCM (default), 1 _spec, 2 _freq
 *     rc <int>                                    status of ainr_get_mem_size*() / ainr_init*() / the first
 *                                                 preprocess (a refused config prints this line only)
 *     graph <path>                                AinrInfo.graph_path
 *     info trim <int> queue <int> tail <int>      AinrInfo trim_samples / queue_frames / tail_zero_blocks
 *     counts inputs <int> outputs <int>           AinrWork input_count / output_count
 *     in <i> <b> <c> <h> <w> <elements>           AinrWork.inputs[i], signal inputs only
 *     out <i> <b> <c> <h> <w> <elements>          AinrWork.outputs[i], heads only
 *
 * The recurrent state is the accelerator's and is not in the work. N and hop_in are the arguments. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"

/* A zero-filled ERB file of the grid's size: init only checks the header and the length. */
static int write_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb * 2u, i;
    int32_t hdr[2];
    float z = 0.0f;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    if (fwrite(hdr, sizeof(int32_t), 2, fp) != 2) { fclose(fp); return -1; }
    for (i = 0; i < elems; ++i)
        if (fwrite(&z, sizeof(float), 1, fp) != 1) { fclose(fp); return -1; }
    return fclose(fp);
}

int main(int argc, char **argv) {
    AinrConfig cfg;
    AinrInfo info;
    AinrWork work;
    StreamMemReq req;
    AinrCtx *pcm = NULL;
    AinrSpecCtx *spec = NULL;
    AinrFreqCtx *freq = NULL;
    void *pool = NULL;
    int16_t *pcm_in = NULL;
    float *zeros = NULL;
    char erb[96];
    int rc, i, family, n_erb, hop, bins;

    if (argc != 4 && argc != 5) return 2;
    cfg.sr = atoi(argv[1]); cfg.fft = atoi(argv[2]); cfg.hop_in = atoi(argv[3]);
    family = argc == 5 ? atoi(argv[4]) : 0;
    if (family < 0 || family > 2) return 2;

    rc = family == 0 ? ainr_get_mem_size(&cfg, &req)
       : family == 1 ? ainr_get_mem_size_spec(&cfg, &req) : ainr_get_mem_size_freq(&cfg, &req);
    if (rc == AINR_OK) {
        hop = cfg.fft / 2;
        bins = hop + 1;
        n_erb = cfg.sr == 16000 ? 24 : 32;
        snprintf(erb, sizeof(erb), "/tmp/ainr_dump_work_%d.bin", (int)getpid());
        if (write_erb(erb, bins, n_erb) != 0) return 2;
        if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) return 2;
        rc = family == 0 ? ainr_init_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &pcm, &info)
           : family == 1 ? ainr_init_spec_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &spec, &info)
                         : ainr_init_freq_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &freq, &info);
        unlink(erb);
    }
    if (rc == AINR_OK) {
        pcm_in = (int16_t *)calloc((size_t)cfg.hop_in, sizeof(int16_t));
        zeros = (float *)calloc((size_t)(cfg.hop_in / hop) * (size_t)bins, sizeof(float));
        if (!pcm_in || !zeros) return 2;
        rc = family == 0 ? ainr_preprocess(pcm, pcm_in, &work)
           : family == 1 ? ainr_preprocess_spec(spec, pcm_in, &work)
                         : ainr_preprocess_freq(freq, zeros, zeros, zeros, zeros, &work);
    }
    printf("rc %d\n", rc);
    if (rc != AINR_OK) { free(pool); free(pcm_in); free(zeros); return 0; }
    printf("graph %s\n", info.graph_path);
    printf("info trim %d queue %d tail %d\n", info.trim_samples, info.queue_frames, info.tail_zero_blocks);
    printf("counts inputs %d outputs %d\n", work.input_count, work.output_count);
    for (i = 0; i < work.input_count; ++i)
        printf("in %d %u %u %u %u %zu\n", i, work.inputs[i].b, work.inputs[i].c, work.inputs[i].h, work.inputs[i].w,
               work.inputs[i].elements);
    for (i = 0; i < work.output_count; ++i)
        printf("out %d %u %u %u %u %zu\n", i, work.outputs[i].b, work.outputs[i].c, work.outputs[i].h, work.outputs[i].w,
               work.outputs[i].elements);
    free(pool); free(pcm_in); free(zeros);
    return 0;
}
