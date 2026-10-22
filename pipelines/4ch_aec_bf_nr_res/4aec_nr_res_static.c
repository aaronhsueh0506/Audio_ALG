/**
 * 4aec_nr_res_static.c — 4x AEC(linear) -> beamformer -> NR/RES
 *                        (static memory, stages chained by the caller)
 *
 * Minimal example of the frequency-domain stage chain over 4aec_nr_res.h.
 * It follows pipelines/mono_aec_nr_res/static_main.c:
 *
 *   default config -> get_mem_requirements -> caller allocation
 *                  -> init -> process -> destroy -> caller release
 *
 * The front end (four_aec_front_process / four_aec_fuse) owns the shared
 * aligner and the four lanes; this file carves the post-beam stages from the
 * same caller pool and chains them itself:
 *
 *   front_process -> (weights) -> fuse -> se_res_stage_gain, se_anr_stage_gain
 *     -> sk_min_f32 -> sk_capply_gain_f32 -> aec_cng_fill -> ac_istft_push
 *
 * The beamformer is externally owned in a real product (SRP-PHAT/BF); this
 * executable supplies fixed equal weights between the two front calls. Equal
 * weights are a deterministic smoke adapter, not a production beamformer.
 *
 * Build:
 *   make -C pipelines/4ch_aec_bf_nr_res 4aec_nr_res_static
 *
 * Usage:
 *   AEC preset: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted.
 *   ./4aec_nr_res_static [aec-preset]
 *       [--nr-preset mild|moderate|balanced|aggressive]
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
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "common/se_anr_stage.h"
#include "common/se_stage.h"

/* ------------------------------------------------------------------ */

/* CLI aliases only; unknown names retain the historical balanced fallback. */
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

/* Name table lives beside the enum (lib/nr anr_types.h); an unknown
 * name keeps this tool's long-standing fallback to balanced. */
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

/* The post-beam stages this example carves behind the front end. */
typedef struct Stages {
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    size_t res_bytes, nr_bytes, synth_bytes, fft_bytes;
    size_t vec_bytes;                     /* one float[n_freqs], 16-byte padded */
    size_t cvec_bytes;                    /* one Complex[n_freqs], padded       */
    size_t total;                         /* pool: front end + every stage       */
} Stages;

static int plan_stages(const FourAecNrResConfig* cfg, const FourAecNrResMemReq* req,
                       const FourAecNrResMemBreakdown* b, Stages* s) {
    AcStftConfig scfg;
    if (four_aec_nr_res_stage_configs(cfg, &s->res_cfg, &s->nr_cfg) != 0) return -1;
    s->res_bytes = aec_res_get_mem_size(&s->res_cfg);
    s->nr_bytes = cfg->enable_nr ? se_anr_stage_mem_size(&s->nr_cfg) : 0;
    ac_stft_config_default(&scfg, b->fft_size);
    s->synth_bytes = ac_istft_get_mem_size(&scfg);
    s->fft_bytes = fft_get_mem_size(b->fft_size);
    s->vec_bytes = ALIGN16((size_t)b->n_freqs * sizeof(float));
    s->cvec_bytes = ALIGN16((size_t)b->n_freqs * sizeof(Complex));
    if (!s->res_bytes || !s->synth_bytes || (cfg->enable_nr && !s->nr_bytes)) return -1;
    s->total = (size_t)req->bytes + ALIGN16(s->res_bytes) + ALIGN16(s->nr_bytes) +
               ALIGN16(s->synth_bytes) + 2 * s->vec_bytes + s->cvec_bytes;
    return 0;
}

static int print_mem_budget(const FourAecNrResConfig* cfg) {
    FourAecNrResMemReq req;
    FourAecNrResMemBreakdown b;
    Stages s;

    if (four_aec_nr_res_get_mem_requirements(cfg, &req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(cfg, &b) != 0 ||
        plan_stages(cfg, &req, &b, &s) != 0) {
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
           s.fft_bytes, (float)s.fft_bytes / 1024.0f);
    printf("  NR (MMSE-LSA):  %9zu bytes (%7.1f KB)\n",
           s.nr_bytes, (float)s.nr_bytes / 1024.0f);
    printf("  Pipeline bufs:  %9zu bytes (%7.1f KB)\n",
           s.total - b.aec_bytes - ALIGN16(s.fft_bytes) - ALIGN16(s.nr_bytes),
           (float)(s.total - b.aec_bytes - ALIGN16(s.fft_bytes) -
                   ALIGN16(s.nr_bytes)) / 1024.0f);
    printf("  --------------------------------------\n");
    printf("  Total:          %9llu bytes (%7.1f KB)\n",
           (unsigned long long)s.total,
           (float)s.total / 1024.0f);
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
    AnrMode nr_mode = ANR_MODE_BALANCED;
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
                "[--legacy-amin] [--no-cng] [--print-mem-size]\n"
                "AEC preset: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted.\n",
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
     * static_main.c. Aligned allocation is a host stand-in for the board
     * memory manager; the pipeline owns none of this memory. */
    FourAecNrResMemReq req;
    FourAecNrResMemBreakdown bd;
    Stages st;
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(&cfg, &bd) != 0 ||
        plan_stages(&cfg, &req, &bd, &st) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        return 1;
    }

    void* pool = NULL;
    if (posix_memalign(&pool, req.alignment, st.total) != 0) pool = NULL;
    if (!pool) {
        fprintf(stderr, "Error: aligned allocation failed (%zu bytes)\n", st.total);
        return 1;
    }
    if (((uintptr_t)pool) % req.alignment != 0) {
        fprintf(stderr, "Error: pool not %u-byte aligned (%p)\n",
                req.alignment, pool);
        free(pool);
        return 1;
    }

    /* Carve: front end, then the stages. */
    uint8_t* cur = (uint8_t*)pool;
    FourAecNrRes* pipe = four_aec_nr_res_init(pool, (size_t)req.bytes, &cfg);
    cur += (size_t)req.bytes;
    if (!pipe) {
        free(pool);
        return 1;
    }
    AecRes* res = aec_res_init(cur, st.res_bytes, &st.res_cfg);
    cur += ALIGN16(st.res_bytes);
    AnrDenoiser* nr = se_anr_stage_init(cur, st.nr_bytes, &st.nr_cfg);
    cur += ALIGN16(st.nr_bytes);
    AcStftConfig scfg;
    ac_stft_config_default(&scfg, bd.fft_size);
    AcIstft* istft = ac_istft_init(cur, st.synth_bytes, &scfg);
    cur += ALIGN16(st.synth_bytes);
    float* total_gain = (float*)cur; cur += st.vec_bytes;
    float* extra_noise = (float*)cur; cur += st.vec_bytes;
    Complex* spec = (Complex*)cur;
    if (!res || !nr || !istft) {
        four_aec_nr_res_destroy(pipe);
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
    uint32_t rng = SE_CNG_RNG_SEED;

    /* === Processing loop === */
    int frame;
    for (frame = 0; frame < 6; ++frame) {
        FourAecFront front;
        SeSpecFrame f;
        const float* res_gain;
        const float* nr_gain;
        fill_hop(mic_buf, ref_buf, hop, sample_rate, frame);
        if (four_aec_front_process(
                pipe, mic_buf, ref_buf, &front) != FOUR_AEC_NR_RES_OK ||
            four_aec_fuse(pipe, weights, NULL, &f) != FOUR_AEC_NR_RES_OK ||
            !(res_gain = se_res_stage_gain(res, &f)) ||
            !(nr_gain = se_anr_stage_gain(nr, &f, extra_noise, !legacy))) {
            fprintf(stderr, "Error: 4AEC pipeline process failed\n");
            four_aec_nr_res_destroy(pipe);
            free(pool);
            free(mic_buf);
            free(ref_buf);
            free(out_buf);
            free(weights);
            return 1;
        }
        sk_min_f32(total_gain, nr_gain, res_gain, n_freqs);
        sk_capply_gain_f32(spec, f.e, total_gain, n_freqs);
        if (!no_cng)
            aec_cng_fill(&rng, spec, f.cn, res_gain, nr_gain,
                         SE_CNG_NR_GAIN_FLOOR, n_freqs);
        ac_istft_push(istft, spec, out_buf);
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
