/** 4x AEC(linear) -> beamformer -> NR/RES with static memory and the stages chained
 * by the caller. Minimal example of the array instance of aec.h: config (n_mics = 4) ->
 * aec_get_mem_requirements -> aec_init_ex -> process -> destroy -> release, the array and
 * the post-beam stages in one caller allocation.
 * The array drives the shared aligner, the four lanes and the residual gain stage (AEC_RESIDUAL_FULL;
 * --no-res selects ESTIMATE: fusion without a gain); the post-beam stages go behind it in the same
 * allocation, each a separate library initialised here: the denoiser and the synthesis:
 *   aec_process_linear -> (weights) -> aec_process_residual -> NR gain -> min with G_res -> CNG -> synthesis
 * The beamformer is externally owned in a product; this executable supplies fixed
 * equal weights, a deterministic smoke adapter and not a production beamformer.
 * Build:  make -C pipelines/4ch 4ch_external_static
 * Usage:
 *   AEC level: 0=mild, 1=balanced, 2=aggressive, 3=no_floor (the number). The positional
 *   [aec-level] and --aec-level <level> are the same setting; give one of them once.
 *   NR level: 0=mild, 1=moderate, 2=balanced, 3=aggressive; --nr-preset is the same option.
 *   ./4ch_external_static [aec-level | --aec-level <level>] [--nr-level 0|1|2|3]
 *       [--sample-rate 16000|48000] [--fft-size 256|512|1024]
 *       [--capture-proxy 0|1|2|3] [--legacy-amin] [--no-cng] [--no-res]
 *   ./4ch_external_static --print-mem-size [same options]
 * Numbers are whole decimal ints. Exit: 0 done, 1 refused config or failed run, 2 command line. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "aec.h"
#include "anr.h"
#include "audio_stft.h"
#include "array_sample_post.h"
#include "common/example_cli.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"   /* skn_all_finite_f32 */

/* A level is its enum number, given as a whole decimal string 0..3; -1 for anything else. */
static int parse_level(const char* s) {
    int v;
    return example_cli_parse_int(s, &v) == 0 && v >= 0 && v <= 3 ? v : -1;
}

static const char* level_name(AecLevel level) {
    switch (level) {
        case AEC_LEVEL_MILD:       return "mild";
        case AEC_LEVEL_AGGRESSIVE: return "aggressive";
        case AEC_LEVEL_NO_FLOOR:   return "no_floor";
        default:                   return "balanced";
    }
}

static const char* nr_level_name(AnrLevel level) {
    return anr_nr_level_name(level);
}

static const char* backend_id_name(uint32_t backend_id) {
    switch (backend_id) {
        case AEC_BACKEND_NE10: return "ne10";
        default:               return "unknown";
    }
}

/* The array: one pool, one aec_init_ex(). The CLI level sets the residual gain stage (the lanes'
 * own level has no effect). */
typedef struct Flow {
    AecConfig cfg;
    AecMemReq req;
    AecMemBreakdown breakdown;
    size_t total;
    Aec* aec;
} Flow;

static int flow_plan(Flow* fl, int sample_rate, int fft_size, AecLevel level, int capture_proxy,
                     int with_res) {
    memset(fl, 0, sizeof(*fl));
    aec_config_from_level(&fl->cfg, level, sample_rate);
    if (fft_size) fl->cfg.fft_size = fft_size;
    fl->cfg.n_mics = AEC_ARRAY_MICS;
    fl->cfg.capture_proxy_channel = capture_proxy;
    fl->cfg.residual = with_res ? AEC_RESIDUAL_FULL : AEC_RESIDUAL_ESTIMATE;
    if (aec_get_mem_requirements(&fl->cfg, &fl->req) != 0 ||
        aec_get_mem_breakdown(&fl->cfg, &fl->breakdown) != AEC_OK) return -1;
    fl->total = ALIGN16((size_t)fl->req.bytes);
    return 0;
}

static void flow_destroy(Flow* fl) {
    aec_destroy(fl->aec);
    fl->aec = NULL;
}

static int flow_init(Flow* fl, void* mem) {
    fl->aec = aec_init_ex(mem, (size_t)fl->req.bytes, &fl->cfg, &fl->req);
    return fl->aec ? 0 : -1;
}

/* The stages after the fuse, carved in this order behind the array. */
typedef struct Post {
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t nr_bytes, synth_bytes, vec_bytes, spec_bytes, total;
    int use_r2, enable_cng, with_res;
    AnrDenoiser* nr;
    AcIstft* istft;
    float *unity, *extra_noise;
    Complex* spec;
    uint32_t reset_handled;   /* AecResidualView.post_reset_serial the synthesis last followed */
    int n_freqs;
} Post;

static int post_plan(Post* q, int sample_rate, int fft_size, AnrLevel nr_level) {
    int n_freqs = fft_size / 2 + 1;
    q->nr_cfg = anr_config_for_level_grid(sample_rate, fft_size, nr_level);
    ac_stft_config_default(&q->synth_cfg, fft_size);
    q->nr_bytes = anr_get_mem_size(&q->nr_cfg);
    q->synth_bytes = ac_istft_get_mem_size(&q->synth_cfg);
    q->vec_bytes = ALIGN16((size_t)n_freqs * sizeof(float));
    q->spec_bytes = ALIGN16((size_t)n_freqs * sizeof(Complex));
    if (q->nr_bytes == 0 || q->synth_bytes == 0) return -1;
    /* nr, synthesis, unity, NR prior, output spectrum */
    q->total = ALIGN16(q->nr_bytes) + ALIGN16(q->synth_bytes) + 2 * q->vec_bytes + q->spec_bytes;
    q->n_freqs = n_freqs;
    return 0;
}

static int post_init(Post* q, void* mem) {
    unsigned char* cur = (unsigned char*)mem;
    int k;
    q->nr = anr_init(cur, q->nr_bytes, &q->nr_cfg);
    cur += ALIGN16(q->nr_bytes);
    q->istft = ac_istft_init(cur, q->synth_bytes, &q->synth_cfg);
    cur += ALIGN16(q->synth_bytes);
    if (!q->nr || !q->istft) return -1;
    q->unity = (float*)cur;       cur += q->vec_bytes;
    q->extra_noise = (float*)cur; cur += q->vec_bytes;
    q->spec = (Complex*)cur;
    for (k = 0; k < q->n_freqs; ++k) q->unity[k] = 1.0f;
    q->reset_handled = 0u;
    return 0;
}

/* One hop after the fuse into out: G_nr estimated on E, P = B * min(G_nr, G_res) over the rendering
 * base B (G_res = 1 without the gain stage), comfort noise on the bins the gain cut, synthesis (cleared
 * first when a lane had a post-chain reset). 0, or -1 when a stage refuses the hop. */
static int post_hop(Post* q, Aec* aec, const AecResidualView* v, float* out) {
    const float* nr = array_sample_anr_gain(q->nr, v, q->extra_noise, q->use_r2);
    if (!nr) return -1;
    if (array_sample_post_reset_due(&q->reset_handled, v)) ac_istft_reset(q->istft);
    if (q->with_res) {
        if (aec_apply_gain(aec, nr, q->spec) != AEC_OK) return -1;
        if (q->enable_cng && aec_apply_cng(aec, nr, q->spec) != AEC_OK) return -1;
    } else {
        aec_apply_gain_spec(q->spec, v->base_spec, q->unity, nr, q->n_freqs);
    }
    (void)ac_istft_push(q->istft, q->spec, out);
    return 0;
}

static void post_destroy(Post* q) {
    if (q->nr) anr_destroy(q->nr);
    q->nr = NULL;
}

/* "AEC x4" is the four lanes; "Front end" everything else of the array pool (the shared
 * estimator and ring, the fusion buffers and the residual gain stage); the descriptor is the array's. */
static void print_mem_budget(const Flow* fl, const Post* q) {
    const AecMemReq req = fl->req;
    const AecMemBreakdown b = fl->breakdown;
    AecSignalGrid grid;
    size_t bufs = 2 * q->vec_bytes + q->spec_bytes;
    size_t aec_bytes = b.lane_bytes;

    (void)aec_resolve_signal_grid(fl->cfg.sample_rate, fl->cfg.fft_size, &grid);
    printf("Memory Budget (4AEC + post-beam NR/RES static pipeline)\n");
    printf("======================================================\n");
    printf("  sample_rate=%d hop=%d fft=%d n_freqs=%d\n",
           fl->cfg.sample_rate, grid.hop_size, grid.fft_size, grid.n_freqs);
    printf("  AEC x4:         %9zu bytes (%7.1f KB)\n",
           aec_bytes, (float)aec_bytes / 1024.0f);
    printf("  Front end:      %9zu bytes (%7.1f KB)\n",
           fl->total - aec_bytes,
           (float)(fl->total - aec_bytes) / 1024.0f);
    printf("  NR (MMSE-LSA):  %9zu bytes (%7.1f KB)\n",
           q->nr_bytes, (float)q->nr_bytes / 1024.0f);
    printf("  Synthesis(OLA): %9zu bytes (%7.1f KB)\n",
           q->synth_bytes, (float)q->synth_bytes / 1024.0f);
    printf("  Buffers:        %9zu bytes (%7.1f KB)\n", bufs, (float)bufs / 1024.0f);
    printf("  --------------------------------------\n");
    printf("  Total:          %9llu bytes (%7.1f KB)\n",
           (unsigned long long)(fl->total + q->total),
           (float)(fl->total + q->total) / 1024.0f);
    printf("  Descriptor:     descriptor_version=%u alignment=%u "
           "layout_version=%u backend_id=%u (%s) "
           "build_flags_hash=0x%08x\n\n",
           req.descriptor_version, req.alignment, req.layout_version,
           req.backend_id, backend_id_name(req.backend_id),
           req.build_flags_hash);
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
        for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
            microphones[i * AEC_ARRAY_MICS + ch] =
                reference[i] * (0.35f + 0.05f * (float)ch);
        }
    }
}

/* The whole-number value of argv[*i] (an option name): *i moves onto it. 2, with a message, when
 * the value is missing (the line ends, or the next token is an option) or is not an int. */
static int int_option(int argc, char* argv[], int* i, int* value) {
    const char* name = argv[*i];
    const char* text = example_cli_option_value(argc, argv, i, argv[0]);
    if (!text) return 2;
    if (example_cli_parse_int(text, value) != 0) {
        fprintf(stderr, "%s: %s '%s' is not an integer\n", argv[0], name, text);
        return 2;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    AecLevel level = AEC_LEVEL_BALANCED;
    AnrLevel nr_level = ANR_LEVEL_BALANCED;
    int sample_rate = 16000;
    int fft_size = 0;
    int capture_proxy = 0;
    int legacy = 0;
    int no_cng = 0;
    int no_res = 0;
    int print_only = 0;
    int have_level = 0;
    int rc = 1;
    int i, ch, k, frame, hop, n_freqs;
    size_t pool_bytes;
    void* pool = NULL;
    float *mic_buf = NULL, *ref_buf = NULL, *out_buf = NULL;
    Complex* weights = NULL;
    Flow flow;
    Post post;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--print-mem-size") == 0) {
            print_only = 1;
        } else if (strcmp(argv[i], "--legacy-amin") == 0) {
            legacy = 1;
        } else if (strcmp(argv[i], "--no-cng") == 0) {
            no_cng = 1;
        } else if (strcmp(argv[i], "--no-res") == 0) {
            no_res = 1;
        } else if (strcmp(argv[i], "--sample-rate") == 0) {
            if (int_option(argc, argv, &i, &sample_rate) != 0) return 2;
        } else if (strcmp(argv[i], "--fft-size") == 0) {
            if (int_option(argc, argv, &i, &fft_size) != 0) return 2;
        } else if (strcmp(argv[i], "--capture-proxy") == 0) {
            if (int_option(argc, argv, &i, &capture_proxy) != 0) return 2;
        } else if (strcmp(argv[i], "--nr-level") == 0 || strcmp(argv[i], "--nr-preset") == 0) {
            const char* name = argv[i];
            const char* text = example_cli_option_value(argc, argv, &i, argv[0]);
            int level_id;
            if (!text) return 2;
            level_id = parse_level(text);
            if (level_id < 0) {
                fprintf(stderr, "%s: %s '%s' is not an NR level "
                        "(accepted: 0 mild, 1 moderate, 2 balanced, 3 aggressive)\n", argv[0], name, text);
                return 2;
            }
            nr_level = (AnrLevel)level_id;
        } else if (strcmp(argv[i], "--aec-level") == 0 || argv[i][0] != '-') {
            const char* text = argv[i];
            int p;
            if (argv[i][0] == '-') {
                text = example_cli_option_value(argc, argv, &i, argv[0]);
                if (!text) return 2;
            }
            p = parse_level(text);
            if (p < 0) {
                fprintf(stderr, "%s: AEC level '%s' is not a level "
                        "(accepted: 0 mild, 1 balanced, 2 aggressive, 3 no_floor)\n", argv[0], text);
                return 2;
            }
            if (have_level) {
                fprintf(stderr, "%s: a second AEC level '%s'; give one\n", argv[0], text);
                return 2;
            }
            level = (AecLevel)p;
            have_level = 1;
        } else {
            fprintf(
                stderr,
                "Usage: %s [aec-level | --aec-level <level>] [--nr-level ...] "
                "[--sample-rate 16000|48000] "
                "[--fft-size 256|512|1024] [--capture-proxy 0|1|2|3] "
                "[--legacy-amin] [--no-cng] [--no-res] [--print-mem-size]\n"
                "AEC level: 0=mild, 1=balanced, 2=aggressive, 3=no_floor (the number).\n"
                "The positional [aec-level] and --aec-level are the same setting; give one of them once.\n"
                "NR level: 0=mild, 1=moderate, 2=balanced, 3=aggressive; --nr-preset is the same option.\n",
                argv[0]);
            return 2;
        }
    }

    printf("4x AEC(linear) -> external beamformer -> NR/RES"
           "  (static memory%s)\n",
           legacy ? ", legacy min-only" : "");
    /* tests/test_cli_strict.sh matches this line. */
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n",
           level_name(level), nr_level_name(nr_level),
           no_cng ? "off" : "on");
    printf("  Beamformer: fixed equal weights (smoke adapter only)\n\n");

    /* Query -> allocate -> init. The aligned allocation is a host stand-in for the board
     * memory manager. Every owner starts empty, so one teardown serves every exit. */
    memset(&flow, 0, sizeof(flow));
    memset(&post, 0, sizeof(post));
    if (flow_plan(&flow, sample_rate, fft_size, level, capture_proxy, !no_res) != 0) {
        fprintf(stderr, "Error: invalid 4-channel config\n");
        goto done;
    }
    if (post_plan(&post, sample_rate, flow.cfg.fft_size, nr_level) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        goto done;
    }
    post.use_r2 = !legacy;
    post.enable_cng = !no_cng;
    post.with_res = !no_res;

    print_mem_budget(&flow, &post);
    pool_bytes = flow.total + post.total;
    if (posix_memalign(&pool, flow.req.alignment, pool_bytes) != 0) pool = NULL;
    if (!pool) {
        fprintf(stderr, "Error: aligned allocation failed (%zu bytes)\n", pool_bytes);
        goto done;
    }

    /* Place the array, then the post stages behind it. */
    if (flow_init(&flow, pool) != 0 ||
        post_init(&post, (uint8_t*)pool + flow.total) != 0) goto done;

    hop = aec_hop_size(flow.aec);
    n_freqs = hop + 1;
    printf("n_freqs agreement OK (n_freqs=%d, hop=%d) at %d Hz\n",
           n_freqs, hop, sample_rate);
    if (print_only) {
        rc = 0;
        goto done;
    }

    mic_buf = (float*)malloc((size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    out_buf = (float*)malloc((size_t)hop * sizeof(float));
    weights = (Complex*)calloc((size_t)AEC_ARRAY_MICS * n_freqs, sizeof(Complex));
    if (!mic_buf || !ref_buf || !out_buf || !weights) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        goto done;
    }
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
        for (k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
    }

    /* Processing loop */
    for (frame = 0; frame < 6; ++frame) {
        AecResidualView f;
        AecBeam beam;
        beam.weights = weights;
        beam.beamformed_error = NULL;
        memset(&f, 0, sizeof(f));
        f.struct_size = sizeof(f);
        fill_hop(mic_buf, ref_buf, hop, sample_rate, frame);
        if (aec_process_linear(flow.aec, mic_buf, ref_buf) != AEC_OK ||
            aec_process_residual(flow.aec, &beam) != AEC_OK ||
            aec_get_residual(flow.aec, &f) != AEC_OK ||
            post_hop(&post, flow.aec, &f, out_buf) != 0) {
            fprintf(stderr, "Error: 4AEC pipeline process failed\n");
            goto done;
        }
        if (!skn_all_finite_f32(out_buf, (size_t)hop)) {
            fprintf(stderr, "Error: non-finite output\n");
            goto done;
        }
    }

    printf("Processed: %d frames (%d samples)\n", frame, frame * hop);
    printf("4ch_external_static: smoke PASS\n");
    rc = 0;

done:
    post_destroy(&post);
    flow_destroy(&flow);
    free(pool);
    free(mic_buf);
    free(ref_buf);
    free(out_buf);
    free(weights);
    return rc;
}
