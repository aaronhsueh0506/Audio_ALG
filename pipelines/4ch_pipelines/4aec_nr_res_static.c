/**
 * 4aec_nr_res_static.c — 4x AEC(linear) -> beamformer -> NR/RES
 *                        (Version B: static memory)
 *
 * Thin host reference over 4aec_nr_res.h. It intentionally follows
 * pipelines/aec_nr_pipeline_static.c:
 *
 *   default config -> get_mem_requirements -> caller allocation
 *                  -> init -> process -> destroy -> caller release
 *
 * The four-channel-only difference is the split process boundary. This
 * executable supplies fixed equal weights between process_pre() and
 * process_post() because the real SRP-PHAT/GSC is externally owned. Equal
 * weights are a deterministic smoke adapter, not a production beamformer.
 *
 * Build:
 *   make -C pipelines/4ch_pipelines 4aec_nr_res_static
 *
 * Usage:
 *   ./4aec_nr_res_static [aec-preset]
 *       [--nr-preset mild|balanced|aggressive]
 *       [--sample-rate 16000|48000] [--fft-size 256|512|1024]
 *       [--capture-proxy 0|1|2|3] [--legacy-amin] [--no-cng]
 *   ./4aec_nr_res_static --print-mem-size [same options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "4aec_nr_res.h"

/* ------------------------------------------------------------------ */

static AecPreset parse_preset(const char* s) {
    if (strcmp(s, "mild") == 0) return AEC_PRESET_MILD;
    if (strcmp(s, "aggressive") == 0) return AEC_PRESET_AGGRESSIVE;
    return AEC_PRESET_BALANCED;
}

static const char* preset_name(AecPreset p) {
    switch (p) {
        case AEC_PRESET_MILD:       return "mild";
        case AEC_PRESET_AGGRESSIVE: return "aggressive";
        default:                    return "balanced";
    }
}

static MmseLsaNrMode parse_nr_mode(const char* s) {
    if (strcmp(s, "mild") == 0) return MMSE_LSA_NR_MILD;
    if (strcmp(s, "aggressive") == 0)
        return MMSE_LSA_NR_AGGRESSIVE;
    return MMSE_LSA_NR_BALANCED;
}

static const char* nr_mode_name(MmseLsaNrMode mode) {
    switch (mode) {
        case MMSE_LSA_NR_MILD:       return "mild";
        case MMSE_LSA_NR_AGGRESSIVE: return "aggressive";
        default:                     return "balanced";
    }
}

static const char* backend_id_name(uint32_t backend_id) {
    switch (backend_id) {
        case FOUR_AEC_NR_RES_BACKEND_KISS: return "kiss";
        case FOUR_AEC_NR_RES_BACKEND_NE10: return "ne10";
        default:                           return "unknown";
    }
}

static int print_mem_budget(const FourAecNrResConfig* cfg) {
    FourAecNrResMemReq req;
    FourAecNrResMemBreakdown b;

    if (four_aec_nr_res_get_mem_requirements(cfg, &req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(cfg, &b) != 0) {
        fprintf(stderr, "Error: invalid 4-channel config\n");
        return -1;
    }

    printf("Memory Budget (4AEC + post-beam NR/RES static pipeline)\n");
    printf("======================================================\n");
    printf("  sample_rate=%d hop=%d fft=%d n_freqs=%d\n",
           cfg->sample_rate, b.hop_size, b.fft_size, b.n_freqs);
    printf("  AEC x4:         %9zu bytes (%7.1f KB)\n",
           b.aec_bytes, (float)b.aec_bytes / 1024.0f);
    printf("  FFT (OLA):      %9zu bytes (%7.1f KB)\n",
           b.fft_bytes, (float)b.fft_bytes / 1024.0f);
    printf("  NR (MMSE-LSA):  %9zu bytes (%7.1f KB)\n",
           b.nr_bytes, (float)b.nr_bytes / 1024.0f);
    printf("  Pipeline bufs:  %9zu bytes (%7.1f KB)\n",
           b.wrapper_bytes,
           (float)b.wrapper_bytes / 1024.0f);
    printf("  --------------------------------------\n");
    printf("  Total:          %9llu bytes (%7.1f KB)\n",
           (unsigned long long)req.bytes,
           (float)req.bytes / 1024.0f);
    printf("  Descriptor:     descriptor_version=%u alignment=%u "
           "layout_version=%u backend_id=%u (%s) "
           "build_flags_hash=0x%08x\n\n",
           req.descriptor_version, req.alignment, req.layout_version,
           req.backend_id, backend_id_name(req.backend_id),
           req.build_flags_hash);
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
            microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                reference[i] * (0.35f + 0.05f * (float)ch);
        }
    }
}

/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    AecPreset preset = AEC_PRESET_BALANCED;
    MmseLsaNrMode nr_mode = MMSE_LSA_NR_BALANCED;
    int sample_rate = 16000;
    int fft_size = 0;
    int capture_proxy = 0;
    int legacy = 0;
    int no_cng = 0;
    int print_only = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--print-mem-size") == 0) {
            print_only = 1;
        } else if (strcmp(argv[i], "--legacy-amin") == 0) {
            legacy = 1;
        } else if (strcmp(argv[i], "--no-cng") == 0) {
            no_cng = 1;
        } else if (strcmp(argv[i], "--sample-rate") == 0 &&
                   i + 1 < argc) {
            sample_rate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fft-size") == 0 &&
                   i + 1 < argc) {
            fft_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--capture-proxy") == 0 &&
                   i + 1 < argc) {
            capture_proxy = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nr-preset") == 0 &&
                   i + 1 < argc) {
            nr_mode = parse_nr_mode(argv[++i]);
        } else if (argv[i][0] != '-') {
            preset = parse_preset(argv[i]);
        } else {
            fprintf(
                stderr,
                "Usage: %s [aec-preset] [--nr-preset ...] "
                "[--sample-rate 16000|48000] "
                "[--fft-size 256|512|1024] [--capture-proxy 0|1|2|3] "
                "[--legacy-amin] [--no-cng] [--print-mem-size]\n",
                argv[0]);
            return 1;
        }
    }

    FourAecNrResConfig cfg =
        four_aec_nr_res_default_config(sample_rate);
    cfg.fft_size = fft_size;
    cfg.capture_proxy_channel = capture_proxy;
    cfg.aec_preset = preset;
    cfg.nr_mode = nr_mode;
    cfg.enable_cng = !no_cng;
    cfg.legacy_amin = legacy;

    printf("4x AEC(linear) -> external beamformer -> NR/RES"
           "  (static memory%s)\n",
           legacy ? ", legacy min-only" : "");
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n",
           preset_name(preset), nr_mode_name(nr_mode),
           no_cng ? "off" : "on");
    printf("  Beamformer: fixed equal weights (smoke adapter only)\n\n");

    if (print_mem_budget(&cfg) != 0) return 1;

    /* Query -> allocate -> init: same caller-pool sequence as the mono
     * aec_nr_pipeline_static.c. malloc is a host stand-in for the board
     * memory manager; the pipeline owns none of this memory. */
    FourAecNrResMemReq req;
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        return 1;
    }

    void* pool = malloc((size_t)req.bytes);
    if (!pool) {
        fprintf(stderr, "Error: malloc failed (%llu bytes)\n",
                (unsigned long long)req.bytes);
        return 1;
    }
    if (((uintptr_t)pool) % req.alignment != 0) {
        fprintf(stderr, "Error: pool not %u-byte aligned (%p)\n",
                req.alignment, pool);
        free(pool);
        return 1;
    }

    FourAecNrRes* pipe =
        four_aec_nr_res_init(pool, (size_t)req.bytes, &cfg);
    if (!pipe) {
        free(pool);
        return 1;
    }

    int hop = four_aec_nr_res_hop_size(pipe);
    int n_freqs = four_aec_nr_res_n_freqs(pipe);
    printf("n_freqs agreement OK (n_freqs=%d, hop=%d) at %d Hz\n",
           n_freqs, hop, sample_rate);
    if (print_only) {
        four_aec_nr_res_destroy(pipe);
        free(pool);
        return 0;
    }

    float* mic_buf = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    float* ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* out_buf = (float*)malloc((size_t)hop * sizeof(float));
    Complex* weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs,
        sizeof(Complex));
    if (!mic_buf || !ref_buf || !out_buf || !weights) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        four_aec_nr_res_destroy(pipe);
        free(pool);
        free(mic_buf);
        free(ref_buf);
        free(out_buf);
        free(weights);
        return 1;
    }

    int ch;
    int k;
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
    }

    /* === Processing loop === */
    int frame;
    for (frame = 0; frame < 6; ++frame) {
        FourAecNrResPreFrame pre;
        fill_hop(mic_buf, ref_buf, hop, sample_rate, frame);
        if (four_aec_nr_res_process_pre(
                pipe, mic_buf, ref_buf, &pre) != FOUR_AEC_NR_RES_OK ||
            four_aec_nr_res_process_post(
                pipe, &pre.token, weights, out_buf) !=
                FOUR_AEC_NR_RES_OK) {
            fprintf(stderr, "Error: 4AEC pipeline process failed\n");
            four_aec_nr_res_destroy(pipe);
            free(pool);
            free(mic_buf);
            free(ref_buf);
            free(out_buf);
            free(weights);
            return 1;
        }
        for (k = 0; k < hop; ++k) {
            if (!isfinite(out_buf[k])) {
                fprintf(stderr, "Error: non-finite output\n");
                four_aec_nr_res_destroy(pipe);
                free(pool);
                free(mic_buf);
                free(ref_buf);
                free(out_buf);
                free(weights);
                return 1;
            }
        }
    }

    printf("Processed: %d frames (%d samples)\n", frame, frame * hop);
    printf("4aec_nr_res_static: smoke PASS\n");

    /* === Cleanup === */
    four_aec_nr_res_destroy(pipe);
    free(pool);
    free(mic_buf);
    free(ref_buf);
    free(out_buf);
    free(weights);
    return 0;
}
