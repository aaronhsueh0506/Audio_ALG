/** 4x AEC(linear) -> beamformer -> NR/RES with static memory and the stages chained
 * by the caller. Minimal example of the frequency-domain stage chain over
 * 4aec_nr_res.h: default config -> get_mem_requirements -> caller allocation -> init ->
 * process -> destroy -> release.
 * The front end owns the shared aligner and the four lanes; this file places the
 * post-beam chain (samples/se_sample_chain.h: AecRes, denoiser, comfort noise, synthesis,
 * each a separate library) behind it in the same allocation:
 *   front_process -> (weights) -> fuse -> se_sample_chain_form -> se_sample_chain_synthesize
 * The beamformer is externally owned in a product; this executable supplies fixed
 * equal weights, a deterministic smoke adapter and not a production beamformer.
 * Build:  make -C pipelines/4ch_aec_bf_nr_res 4aec_nr_res_static
 * Usage:
 *   AEC preset: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted.
 *   ./4aec_nr_res_static [aec-preset] [--nr-preset mild|moderate|balanced|aggressive]
 *       [--sample-rate 16000|48000] [--fft-size 256|512|1024]
 *       [--capture-proxy 0|1|2|3] [--legacy-amin] [--no-cng] [--no-res]
 *   ./4aec_nr_res_static --print-mem-size [same options] */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "4aec_nr_res.h"
#include "utility/mem_align.h"
#include "se_sample_chain.h"

/* CLI aliases only; unknown names fall back to balanced. */
static AecPreset parse_preset(const char* s) {
    if (strcmp(s, "mild") == 0 || strcmp(s, "0") == 0)
        return AEC_PRESET_MILD;
    if (strcmp(s, "balanced") == 0 || strcmp(s, "1") == 0)
        return AEC_PRESET_BALANCED;
    if (strcmp(s, "aggressive") == 0 || strcmp(s, "2") == 0)
        return AEC_PRESET_AGGRESSIVE;
    if (strcmp(s, "no_floor") == 0 || strcmp(s, "3") == 0)
        return AEC_PRESET_NO_FLOOR;
    return AEC_PRESET_BALANCED;
}

static const char* preset_name(AecPreset p) {
    switch (p) {
        case AEC_PRESET_MILD:       return "mild";
        case AEC_PRESET_AGGRESSIVE: return "aggressive";
        case AEC_PRESET_NO_FLOOR:   return "no_floor";
        default:                    return "balanced";
    }
}

/* Name table lives beside the enum (anr_types.h); unknown names fall back to balanced. */
static AnrMode parse_nr_mode(const char* s) {
    int m = anr_nr_mode_from_name(s);
    return m < 0 ? ANR_MODE_BALANCED : (AnrMode)m;
}

static const char* nr_mode_name(AnrMode mode) {
    return anr_nr_mode_name(mode);
}

static const char* backend_id_name(uint32_t backend_id) {
    switch (backend_id) {
        case FOUR_AEC_NR_RES_BACKEND_NE10: return "ne10";
        default:                           return "unknown";
    }
}

static int print_mem_budget(const FourAecNrResConfig* cfg, const SeSampleChainConfig* sc) {
    FourAecNrResMemReq req;
    FourAecNrResMemBreakdown b;
    SeSampleChainSizes z;

    if (four_aec_nr_res_get_mem_requirements(cfg, &req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(cfg, &b) != 0 ||
        se_sample_chain_sizes(sc, &z) != 0) {
        fprintf(stderr, "Error: invalid 4-channel config\n");
        return -1;
    }

    printf("Memory Budget (4AEC + post-beam NR/RES static pipeline)\n");
    printf("======================================================\n");
    printf("  sample_rate=%d hop=%d fft=%d n_freqs=%d\n",
           cfg->sample_rate, b.hop_size, b.fft_size, b.n_freqs);
    printf("  AEC x4:         %9zu bytes (%7.1f KB)\n",
           b.aec_bytes, (float)b.aec_bytes / 1024.0f);
    printf("  Front end:      %9zu bytes (%7.1f KB)\n",
           (size_t)req.bytes - b.aec_bytes,
           (float)((size_t)req.bytes - b.aec_bytes) / 1024.0f);
    printf("  RES (AecRes):   %9zu bytes (%7.1f KB)\n",
           z.res_bytes, (float)z.res_bytes / 1024.0f);
    printf("  NR (MMSE-LSA):  %9zu bytes (%7.1f KB)\n",
           z.nr_bytes, (float)z.nr_bytes / 1024.0f);
    printf("  Synthesis(OLA): %9zu bytes (%7.1f KB)\n",
           z.synth_bytes, (float)z.synth_bytes / 1024.0f);
    printf("  Chain bufs:     %9zu bytes (%7.1f KB)\n",
           z.total - ALIGN16(z.res_bytes) - ALIGN16(z.nr_bytes) - ALIGN16(z.synth_bytes),
           (float)(z.total - ALIGN16(z.res_bytes) - ALIGN16(z.nr_bytes) -
                   ALIGN16(z.synth_bytes)) / 1024.0f);
    printf("  --------------------------------------\n");
    printf("  Total:          %9llu bytes (%7.1f KB)\n",
           (unsigned long long)req.bytes + z.total,
           (float)((size_t)req.bytes + z.total) / 1024.0f);
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

int main(int argc, char* argv[]) {
    AecPreset preset = AEC_PRESET_BALANCED;
    AnrMode nr_mode = ANR_MODE_BALANCED;
    int sample_rate = 16000;
    int fft_size = 0;
    int capture_proxy = 0;
    int legacy = 0;
    int no_cng = 0;
    int no_res = 0;
    int print_only = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--print-mem-size") == 0) {
            print_only = 1;
        } else if (strcmp(argv[i], "--legacy-amin") == 0) {
            legacy = 1;
        } else if (strcmp(argv[i], "--no-cng") == 0) {
            no_cng = 1;
        } else if (strcmp(argv[i], "--no-res") == 0) {
            no_res = 1;
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
                "[--legacy-amin] [--no-cng] [--no-res] [--print-mem-size]\n"
                "AEC preset: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted.\n",
                argv[0]);
            return 1;
        }
    }

    FourAecNrResConfig cfg =
        four_aec_nr_res_default_config(sample_rate);
    cfg.fft_size = fft_size;
    cfg.capture_proxy_channel = capture_proxy;

    printf("4x AEC(linear) -> external beamformer -> NR/RES"
           "  (static memory%s)\n",
           legacy ? ", legacy min-only" : "");
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n",
           preset_name(preset), nr_mode_name(nr_mode),
           no_cng ? "off" : "on");
    printf("  Beamformer: fixed equal weights (smoke adapter only)\n\n");

    /* Query -> allocate -> init. The aligned allocation is a host stand-in for the board
     * memory manager. */
    FourAecNrResMemReq req;
    FourAecNrResMemBreakdown bd;
    SeSampleChainConfig sc;
    SeSampleChainSizes st;
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(&cfg, &bd) != 0) {
        fprintf(stderr, "Error: invalid 4-channel config\n");
        return 1;
    }
    sc = se_sample_chain_default_config(sample_rate, bd.fft_size);
    sc.res = no_res ? SE_SAMPLE_RES_OFF : SE_SAMPLE_RES_AEC_RES;
    sc.res_preset = preset;
    sc.nr_mode = nr_mode;
    sc.legacy_amin = legacy;
    sc.enable_cng = !no_cng;

    if (print_mem_budget(&cfg, &sc) != 0) return 1;
    if (se_sample_chain_sizes(&sc, &st) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        return 1;
    }
    size_t pool_bytes = (size_t)req.bytes + st.total;

    void* pool = NULL;
    if (posix_memalign(&pool, req.alignment, pool_bytes) != 0) pool = NULL;
    if (!pool) {
        fprintf(stderr, "Error: aligned allocation failed (%zu bytes)\n", pool_bytes);
        return 1;
    }
    if (((uintptr_t)pool) % req.alignment != 0) {
        fprintf(stderr, "Error: pool not %u-byte aligned (%p)\n",
                req.alignment, pool);
        free(pool);
        return 1;
    }

    /* Place the front end, then the chain behind it. */
    FourAecNrRes* pipe = four_aec_nr_res_init(pool, (size_t)req.bytes, &cfg);
    if (!pipe) {
        free(pool);
        return 1;
    }
    SeSampleChain* chain =
        se_sample_chain_init((uint8_t*)pool + (size_t)req.bytes, st.total, &sc);
    if (!chain) {
        four_aec_nr_res_destroy(pipe);
        free(pool);
        return 1;
    }

    int hop = four_aec_nr_res_hop_size(pipe);
    int n_freqs = four_aec_nr_res_n_freqs(pipe);
    printf("n_freqs agreement OK (n_freqs=%d, hop=%d) at %d Hz\n",
           n_freqs, hop, sample_rate);
    if (print_only) {
        se_sample_chain_destroy(chain);
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
        se_sample_chain_destroy(chain);
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

    /* Processing loop */
    int frame;
    for (frame = 0; frame < 6; ++frame) {
        FourAecFront front;
        SeSpecFrame f;
        const Complex* spec;
        fill_hop(mic_buf, ref_buf, hop, sample_rate, frame);
        if (four_aec_front_process(
                pipe, mic_buf, ref_buf, &front) != FOUR_AEC_NR_RES_OK ||
            four_aec_fuse(pipe, weights, NULL, &f) != FOUR_AEC_NR_RES_OK ||
            !(spec = se_sample_chain_form(chain, &f)) ||
            se_sample_chain_synthesize(chain, spec, out_buf) != 0) {
            fprintf(stderr, "Error: 4AEC pipeline process failed\n");
            se_sample_chain_destroy(chain);
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
                se_sample_chain_destroy(chain);
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

    /* Cleanup */
    se_sample_chain_destroy(chain);
    four_aec_nr_res_destroy(pipe);
    free(pool);
    free(mic_buf);
    free(ref_buf);
    free(out_buf);
    free(weights);
    return 0;
}
