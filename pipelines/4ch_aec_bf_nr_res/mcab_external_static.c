/** 4x AEC(linear) -> beamformer -> NR/RES with static memory and the stages chained
 * by the caller. Minimal example of the application side of mcab.h: default config ->
 * the lanes' and the estimator's own get_mem_size/init -> mcab's get_mem_requirements/init ->
 * process -> destroy -> release, every object in one caller allocation.
 * mcab drives the shared aligner and the four lanes this file creates; the post-beam stages go
 * behind them in the same allocation, each a separate library initialised here: the RES
 * (AecRes: the lanes compute no gain of their own), the denoiser and the synthesis:
 *   process_front -> (weights) -> fuse -> RES gain -> NR gain -> min -> CNG -> synthesis
 * The beamformer is externally owned in a product; this executable supplies fixed
 * equal weights, a deterministic smoke adapter and not a production beamformer.
 * Build:  make -C pipelines/4ch_aec_bf_nr_res mcab_external_static
 * Usage:
 *   AEC level: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted. The positional
 *   [aec-level] and --aec-level <level> are the same setting; give one of them once.
 *   NR level: mild, moderate, balanced, aggressive or 0..3; --nr-preset is the same option.
 *   ./mcab_external_static [aec-level | --aec-level <level>] [--nr-level mild|moderate|balanced|aggressive]
 *       [--sample-rate 16000|48000] [--fft-size 256|512|1024]
 *       [--capture-proxy 0|1|2|3] [--legacy-amin] [--no-cng] [--no-res]
 *   ./mcab_external_static --print-mem-size [same options]
 * Numbers are whole decimal ints. Exit: 0 done, 1 refused config or failed run, 2 command line. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "mcab.h"
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "mcab_sample_post.h"
#include "common/example_cli.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "utility/simd_kernel_nn.h"   /* skn_all_finite_f32 */

/* CLI aliases only; -1 for a name that is none of them. */
static int parse_level(const char* s) {
    if (strcmp(s, "mild") == 0 || strcmp(s, "0") == 0)
        return AEC_LEVEL_MILD;
    if (strcmp(s, "balanced") == 0 || strcmp(s, "1") == 0)
        return AEC_LEVEL_BALANCED;
    if (strcmp(s, "aggressive") == 0 || strcmp(s, "2") == 0)
        return AEC_LEVEL_AGGRESSIVE;
    if (strcmp(s, "no_floor") == 0 || strcmp(s, "3") == 0)
        return AEC_LEVEL_NO_FLOOR;
    return -1;
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
        case MCAB_BACKEND_NE10: return "ne10";
        default:                           return "unknown";
    }
}

/* The application side of mcab.h, in the order the objects are carved: four lanes, the shared
 * estimator object and its pool (MATCHED only), mcab's pool. The lanes' parameters are this
 * program's (balanced; the CLI level drives the post-beam RES); mcab stamps its contract. */
typedef struct Flow {
    AecConfig lane_cfg;
    McabDelayEstimatorConfig dc;
    McabMemReq req;           /* mcab's own pool */
    McabMemBreakdown grid;
    size_t lane_bytes, delay_bytes, total;
    Aec* lanes[MCAB_CHANNELS];
    DelayAec* delay;
    Mcab* p;
} Flow;

static int flow_plan(Flow* fl, const McabConfig* cfg) {
    int ch;
    memset(fl, 0, sizeof(*fl));
    aec_config_from_level(&fl->lane_cfg, AEC_LEVEL_BALANCED, cfg->sample_rate);
    if (mcab_apply_lane_contract(cfg, &fl->lane_cfg) != 0 ||
        mcab_delay_estimator_config(cfg, &fl->dc) != 0 ||
        mcab_get_mem_requirements(cfg, &fl->req) != 0 ||
        mcab_get_mem_breakdown(cfg, &fl->grid) != 0) return -1;
    fl->lane_bytes = aec_get_mem_size(&fl->lane_cfg);
    if (fl->lane_bytes == 0) return -1;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) fl->total += ALIGN16(fl->lane_bytes);
    if (fl->dc.required) {
        fl->delay_bytes = delay_aec_get_mem_size(fl->dc.sample_rate, fl->dc.hop_size,
                                                 fl->dc.num_filters);
        if (fl->delay_bytes == 0) return -1;
        fl->total += ALIGN16(sizeof(DelayAec)) + ALIGN16(fl->delay_bytes);
    }
    fl->total += ALIGN16((size_t)fl->req.bytes);
    return 0;
}

/* The flow tears down before the objects it borrows. */
static void flow_destroy(Flow* fl) {
    int ch;
    mcab_destroy(fl->p);
    fl->p = NULL;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        if (fl->lanes[ch]) aec_destroy(fl->lanes[ch]);
        fl->lanes[ch] = NULL;
    }
}

static int flow_init(Flow* fl, void* mem, const McabConfig* cfg) {
    unsigned char* cur = (unsigned char*)mem;
    McabLibs libs;
    int ch;
    memset(&libs, 0, sizeof(libs));
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        fl->lanes[ch] = aec_init(cur, fl->lane_bytes, &fl->lane_cfg);
        if (!fl->lanes[ch]) return -1;
        libs.lanes[ch] = fl->lanes[ch];
        cur += ALIGN16(fl->lane_bytes);
    }
    if (fl->dc.required) {
        fl->delay = (DelayAec*)cur;
        cur += ALIGN16(sizeof(DelayAec));
        if (delay_aec_init(fl->delay, cur, fl->delay_bytes, fl->dc.sample_rate, fl->dc.hop_size,
                           fl->dc.num_filters) != 0) return -1;
        cur += ALIGN16(fl->delay_bytes);
        libs.delay = fl->delay;
    }
    fl->p = mcab_init_ex(cur, (size_t)fl->req.bytes, cfg, &libs, &fl->req);
    return fl->p ? 0 : -1;
}

/* The stages after the fuse, carved in this order behind the flow. */
typedef struct Post {
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t res_bytes, nr_bytes, synth_bytes, vec_bytes, spec_bytes, total;
    int use_r2, enable_cng;
    AecRes* res;              /* NULL with --no-res */
    AnrDenoiser* nr;
    AcIstft* istft;
    float *unity, *total_gain, *extra_noise;
    Complex* spec;
    uint32_t rng;
    int n_freqs;
} Post;

static int post_plan(Post* q, int sample_rate, int fft_size, int with_res, AecLevel level,
                     AnrLevel nr_level) {
    int n_freqs = fft_size / 2 + 1;
    q->res_bytes = 0;
    if (with_res) {
        aec_config_from_level(&q->res_cfg, level, sample_rate);
        q->res_cfg.fft_size = fft_size;
        q->res_bytes = aec_res_get_mem_size(&q->res_cfg);
        if (q->res_bytes == 0) return -1;
    }
    q->nr_cfg = anr_config_for_level_grid(sample_rate, fft_size, nr_level);
    ac_stft_config_default(&q->synth_cfg, fft_size);
    q->nr_bytes = anr_get_mem_size(&q->nr_cfg);
    q->synth_bytes = ac_istft_get_mem_size(&q->synth_cfg);
    q->vec_bytes = ALIGN16((size_t)n_freqs * sizeof(float));
    q->spec_bytes = ALIGN16((size_t)n_freqs * sizeof(Complex));
    if (q->nr_bytes == 0 || q->synth_bytes == 0) return -1;
    /* res, nr, synthesis, unity, total gain, NR prior, output spectrum */
    q->total = ALIGN16(q->res_bytes) + ALIGN16(q->nr_bytes) + ALIGN16(q->synth_bytes) +
               3 * q->vec_bytes + q->spec_bytes;
    q->n_freqs = n_freqs;
    return 0;
}

static int post_init(Post* q, void* mem) {
    unsigned char* cur = (unsigned char*)mem;
    int k;
    q->res = NULL;
    if (q->res_bytes) {
        q->res = aec_res_init(cur, q->res_bytes, &q->res_cfg);
        if (!q->res) return -1;
        cur += ALIGN16(q->res_bytes);
    }
    q->nr = anr_init(cur, q->nr_bytes, &q->nr_cfg);
    cur += ALIGN16(q->nr_bytes);
    q->istft = ac_istft_init(cur, q->synth_bytes, &q->synth_cfg);
    cur += ALIGN16(q->synth_bytes);
    if (!q->nr || !q->istft) return -1;
    q->unity = (float*)cur;       cur += q->vec_bytes;
    q->total_gain = (float*)cur;  cur += q->vec_bytes;
    q->extra_noise = (float*)cur; cur += q->vec_bytes;
    q->spec = (Complex*)cur;
    for (k = 0; k < q->n_freqs; ++k) q->unity[k] = 1.0f;
    q->rng = MCAB_SAMPLE_CNG_RNG_SEED;
    return 0;
}

/* One hop after the fuse into out: P = E * min(G_nr, G_res), comfort noise on the bins the RES
 * cut, synthesis. 0, or -1 when a stage refuses the hop. */
static int post_hop(Post* q, const McabFrame* f, float* out) {
    const float* res = q->unity;
    const float* nr;
    if (q->res) {
        AecResInput in;
        mcab_sample_res_input(f, &in);
        res = aec_res_process(q->res, &in);
    }
    if (!res) return -1;
    nr = mcab_sample_anr_gain(q->nr, f, q->extra_noise, q->use_r2);
    if (!nr) return -1;
    sk_min_f32(q->total_gain, nr, res, q->n_freqs);   /* NaN in the second operand reaches the finite gate; in the first it is replaced */
    sk_capply_gain_f32(q->spec, f->e, q->total_gain, q->n_freqs);
    if (q->enable_cng && q->res && f->cn)
        aec_cng_fill(&q->rng, q->spec, f->cn, res, nr, MCAB_SAMPLE_CNG_NR_GAIN_FLOOR, q->n_freqs);
    (void)ac_istft_push(q->istft, q->spec, out);
    return 0;
}

static void post_destroy(Post* q) {
    if (q->nr) anr_destroy(q->nr);
    q->nr = NULL;
}

/* "AEC x4" is the four lanes; "Front end" everything else the flow needs (the shared
 * estimator and mcab's own pool); the descriptor is mcab's. */
static void print_mem_budget(const McabConfig* cfg, const Flow* fl, const Post* q) {
    const McabMemReq req = fl->req;
    const McabMemBreakdown b = fl->grid;
    size_t bufs = 3 * q->vec_bytes + q->spec_bytes;
    size_t aec_bytes = 4u * ALIGN16(fl->lane_bytes);

    printf("Memory Budget (4AEC + post-beam NR/RES static pipeline)\n");
    printf("======================================================\n");
    printf("  sample_rate=%d hop=%d fft=%d n_freqs=%d\n",
           cfg->sample_rate, b.hop_size, b.fft_size, b.n_freqs);
    printf("  AEC x4:         %9zu bytes (%7.1f KB)\n",
           aec_bytes, (float)aec_bytes / 1024.0f);
    printf("  Front end:      %9zu bytes (%7.1f KB)\n",
           fl->total - aec_bytes,
           (float)(fl->total - aec_bytes) / 1024.0f);
    printf("  RES (AecRes):   %9zu bytes (%7.1f KB)\n",
           q->res_bytes, (float)q->res_bytes / 1024.0f);
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
        for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
            microphones[i * MCAB_CHANNELS + ch] =
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
    Mcab* pipe;
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
            level_id = anr_nr_level_from_name(text);
            if (level_id < 0) {
                fprintf(stderr, "%s: %s '%s' is not an NR level "
                        "(accepted: mild|moderate|balanced|aggressive or 0..3)\n", argv[0], name, text);
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
                        "(accepted: mild|balanced|aggressive|no_floor or 0..3)\n", argv[0], text);
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
                "AEC level: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted.\n"
                "The positional [aec-level] and --aec-level are the same setting; give one of them once.\n"
                "NR level: mild, moderate, balanced, aggressive or 0..3; --nr-preset is the same option.\n",
                argv[0]);
            return 2;
        }
    }

    McabConfig cfg = mcab_default_config(sample_rate);
    cfg.fft_size = fft_size;
    cfg.capture_proxy_channel = capture_proxy;

    printf("4x AEC(linear) -> external beamformer -> NR/RES"
           "  (static memory%s)\n",
           legacy ? ", legacy min-only" : "");
    /* Wording kept as in earlier releases: tests/test_cli_strict.sh matches this line and the baseline
     * matrix compares this program's stdout byte for byte. */
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n",
           level_name(level), nr_level_name(nr_level),
           no_cng ? "off" : "on");
    printf("  Beamformer: fixed equal weights (smoke adapter only)\n\n");

    /* Query -> allocate -> init. The aligned allocation is a host stand-in for the board
     * memory manager. Every owner starts empty, so one teardown serves every exit. */
    memset(&flow, 0, sizeof(flow));
    memset(&post, 0, sizeof(post));
    if (flow_plan(&flow, &cfg) != 0) {
        fprintf(stderr, "Error: invalid 4-channel config\n");
        goto done;
    }
    if (post_plan(&post, sample_rate, flow.grid.fft_size, !no_res, level, nr_level) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        goto done;
    }
    post.use_r2 = !legacy;
    post.enable_cng = !no_cng;

    print_mem_budget(&cfg, &flow, &post);
    pool_bytes = flow.total + post.total;
    if (posix_memalign(&pool, flow.req.alignment, pool_bytes) != 0) pool = NULL;
    if (!pool) {
        fprintf(stderr, "Error: aligned allocation failed (%zu bytes)\n", pool_bytes);
        goto done;
    }

    /* Place the lanes, the estimator and the flow, then the post stages behind them. */
    if (flow_init(&flow, pool, &cfg) != 0 ||
        post_init(&post, (uint8_t*)pool + flow.total) != 0) goto done;
    pipe = flow.p;

    hop = mcab_hop_size(pipe);
    n_freqs = mcab_n_freqs(pipe);
    printf("n_freqs agreement OK (n_freqs=%d, hop=%d) at %d Hz\n",
           n_freqs, hop, sample_rate);
    if (print_only) {
        rc = 0;
        goto done;
    }

    mic_buf = (float*)malloc((size_t)hop * MCAB_CHANNELS * sizeof(float));
    ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    out_buf = (float*)malloc((size_t)hop * sizeof(float));
    weights = (Complex*)calloc((size_t)MCAB_CHANNELS * n_freqs, sizeof(Complex));
    if (!mic_buf || !ref_buf || !out_buf || !weights) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        goto done;
    }
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
    }

    /* Processing loop */
    for (frame = 0; frame < 6; ++frame) {
        McabFront front;
        McabFrame f;
        fill_hop(mic_buf, ref_buf, hop, sample_rate, frame);
        if (mcab_process_front(pipe, mic_buf, ref_buf, &front) != MCAB_OK ||
            mcab_fuse(pipe, weights, NULL, &f) != MCAB_OK ||
            post_hop(&post, &f, out_buf) != 0) {
            fprintf(stderr, "Error: 4AEC pipeline process failed\n");
            goto done;
        }
        if (!skn_all_finite_f32(out_buf, (size_t)hop)) {
            fprintf(stderr, "Error: non-finite output\n");
            goto done;
        }
    }

    printf("Processed: %d frames (%d samples)\n", frame, frame * hop);
    printf("mcab_external_static: smoke PASS\n");
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
