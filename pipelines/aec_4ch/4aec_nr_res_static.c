/**
 * aec_4ch/4aec_nr_res_static.c
 *
 * REFERENCE ONLY host-side caller-pool example for 4aec_nr_res.h.
 *
 * This intentionally follows pipelines/aec_nr_pipeline_static.c:
 *
 *   config -> get_mem_requirements -> platform-aligned allocation
 *          -> init_ex -> process -> destroy -> platform release
 *
 * posix_memalign/free stand in for the board memory manager. The pipeline
 * itself, including four AECs, NR, FFT, post-beam RES, and all scratch/state,
 * lives in that one pool and performs no heap allocation.
 *
 * SRP-PHAT/GSC remains externally owned. This executable uses fixed equal
 * weights only as a deterministic smoke adapter; it is not a production
 * beamformer and is never selected by the library.
 *
 * Build:  make -C pipelines 4aec_nr_res_static
 * Usage:
 *   ./4aec_nr_res_static [--sample-rate 16000|48000]
 *   ./4aec_nr_res_static --print-mem-size
 *                        [--sample-rate 16000|48000]
 */

#include "4aec_nr_res.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int print_mem_budget(int sample_rate) {
    FourAecNrResConfig cfg;
    FourAecNrResMemReq req;
    FourAecNrResMemBreakdown breakdown;

    four_aec_nr_res_config_defaults(&cfg, sample_rate);
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(&cfg, &breakdown) != 0) {
        fprintf(stderr, "Error: invalid 4-channel config\n");
        return -1;
    }

    printf("Memory Budget (4AEC + post-beam NR/RES static pipeline)\n");
    printf("======================================================\n");
    printf("  sample_rate=%d hop=%d fft=%d n_freqs=%d\n",
           sample_rate, breakdown.hop_size, breakdown.fft_size,
           breakdown.n_freqs);
    printf("  AEC x4:         %9zu bytes (%7.1f KB)\n",
           breakdown.aec_bytes, (float)breakdown.aec_bytes / 1024.0f);
    printf("  NR:             %9zu bytes (%7.1f KB)\n",
           breakdown.nr_bytes, (float)breakdown.nr_bytes / 1024.0f);
    printf("  FFT (OLA):      %9zu bytes (%7.1f KB)\n",
           breakdown.fft_bytes, (float)breakdown.fft_bytes / 1024.0f);
    printf("  Shared/wrapper: %9zu bytes (%7.1f KB)\n",
           breakdown.wrapper_bytes,
           (float)breakdown.wrapper_bytes / 1024.0f);
    printf("  TOTAL:          %9zu bytes (%7.1f KB)\n",
           breakdown.total_bytes,
           (float)breakdown.total_bytes / 1024.0f);
    printf("  descriptor=%u layout=%u backend=%u hash=0x%08x align=%u\n",
           req.descriptor_version, req.layout_version, req.backend_id,
           req.build_flags_hash, req.alignment);
    return 0;
}

static void fill_hop(float* microphones, float* reference,
                     int hop, int sample_rate, int frame_index) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        int absolute = frame_index * hop + i;
        float phase =
            2.0f * 3.14159265358979323846f * 440.0f *
            (float)absolute / (float)sample_rate;
        reference[i] = 0.08f * sinf(phase);
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            microphones[
                i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                reference[i] * (0.35f + 0.05f * (float)ch);
        }
    }
}

static int run_static_example(int sample_rate, int process_frames) {
    FourAecNrResConfig cfg;
    FourAecNrResMemReq req;
    FourAecNrRes* pipeline = NULL;
    FourAecNrResPreFrame pre;
    void* pool = NULL;
    float* microphones = NULL;
    float* reference = NULL;
    float* output = NULL;
    Complex* weights = NULL;
    int hop;
    int n_freqs;
    int frame;
    int ch;
    int k;
    int rc = -1;

    four_aec_nr_res_config_defaults(&cfg, sample_rate);
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0)
        goto cleanup;

    /* Host stand-in for platform_alloc(req.bytes, req.alignment). */
    if (posix_memalign(
            &pool, (size_t)req.alignment, (size_t)req.bytes) != 0)
        goto cleanup;
    memset(pool, 0xa5, (size_t)req.bytes);
    pipeline = four_aec_nr_res_init_ex(
        pool, (size_t)req.bytes, &cfg, &req);
    if (!pipeline) goto cleanup;

    hop = four_aec_nr_res_hop_size(pipeline);
    n_freqs = four_aec_nr_res_n_freqs(pipeline);
    if (!process_frames) {
        printf("n_freqs agreement OK (n_freqs=%d, hop=%d) at %d Hz\n",
               n_freqs, hop, sample_rate);
        rc = 0;
        goto cleanup;
    }

    /* These are host I/O/beamformer-adapter buffers, not pipeline state.
     * Board integration supplies equivalent buffers from its audio task. */
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    reference = (float*)calloc((size_t)hop, sizeof(float));
    output = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs,
        sizeof(Complex));
    if (!microphones || !reference || !output || !weights)
        goto cleanup;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
    }

    for (frame = 0; frame < 6; ++frame) {
        fill_hop(
            microphones, reference, hop, sample_rate, frame);
        if (four_aec_nr_res_process_pre(
                pipeline, microphones, reference, &pre) !=
            FOUR_AEC_NR_RES_OK)
            goto cleanup;
        if (four_aec_nr_res_process_post(
                pipeline, &pre.token, weights, output) !=
            FOUR_AEC_NR_RES_OK)
            goto cleanup;
        for (k = 0; k < hop; ++k) {
            if (!isfinite(output[k])) goto cleanup;
        }
    }
    rc = 0;

cleanup:
    four_aec_nr_res_destroy(pipeline);
    free(weights);
    free(output);
    free(reference);
    free(microphones);
    free(pool);
    return rc;
}

int main(int argc, char** argv) {
    int sample_rate = 16000;
    int print_only = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--print-mem-size") == 0) {
            print_only = 1;
        } else if (strcmp(argv[i], "--sample-rate") == 0 &&
                   i + 1 < argc) {
            sample_rate = atoi(argv[++i]);
        } else {
            fprintf(
                stderr,
                "Usage: %s [--print-mem-size] "
                "[--sample-rate 16000|48000]\n",
                argv[0]);
            return 1;
        }
    }

    if (print_mem_budget(sample_rate) != 0) return 1;
    if (run_static_example(sample_rate, !print_only) != 0) {
        fprintf(stderr, "4aec_nr_res_static: example FAILED\n");
        return 1;
    }
    if (print_only) return 0;
    printf(
        "4aec_nr_res_static: smoke PASS "
        "(equal weights are test-only)\n");
    return 0;
}
