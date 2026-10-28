/** AEC(linear) -> echo-aware NR -> RES, composed here from the library calls. Example code: Aec
 * (lib/aec) for the linear stage and its own RES gain, AnrDenoiser (lib/nr) and AcIstft (audiolib)
 * initialised here. Per hop:
 *     aec_process_context(aec, mic, ref);
 *     aec_get_res_context(aec, &ctx);                  // E, R2, CN and G_res (ctx.res_gain)
 *     G_nr = anr_process_gain(nr, E, R2 / 32768^2) + anr_get_gain(nr);
 *     P = E * min(G_nr, G_res) + comfort noise;        // mono_form()
 *     ac_istft_push(istft, P, out);
 * A refused call resets every stage together. The stages are carved in the order AEC, denoiser,
 * synthesis, gain/spectrum buffers with no malloc after the pool allocation. One source, two
 * binaries:
 *   aec_nr_pipeline_static  one aligned caller-owned pool; also `--print-mem-size`
 *   aec_nr_pipeline         -DAEC_NR_POOL_SPLIT=1: the AEC in its own block, the post stages in
 *                           a second one
 * Build:  make -C pipelines            (builds both binaries)
 * Usage:
 *   AEC level: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted. The positional
 *   [aec-level] and --aec-level <level> are the same setting; give one of them once.
 *   NR level: mild, moderate, balanced, aggressive or 0..3; --nr-preset is the same option.
 *   ./aec_nr_pipeline_static <mic.wav> <ref.wav> <out.wav> [aec-level | --aec-level <level>]
 *                     [--nr-level mild|moderate|balanced|aggressive] [--aec-only]
 *                     [--no-nr] [--no-res] [--legacy-amin] [--debug]
 *   ./aec_nr_pipeline_static --print-mem-size [aec-level | --aec-level <level>] [--nr-level ...] [--aec-only]
 *                     [--no-nr] [--no-res] [--sample-rate <hz>]
 * --debug: once per second of processed audio, one status line to stderr from aec_debug_status() + anr_debug_status() (read-only). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "aec.h"
#include "aec_res.h"           /* aec_cng_fill */
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "wav_io.h"
#include "common/example_cli.h"

#ifndef AEC_NR_POOL_SPLIT
#define AEC_NR_POOL_SPLIT 0
#endif

/* Comfort noise follows the denoiser gain of its bin but never below this amplitude (-10 dB),
 * so the residual echo it masks does not surface. */
#define CNG_NR_GAIN_FLOOR 0.31622777f
/* Start state of the comfort-noise generator (aec_cng_fill). */
#define CNG_RNG_SEED 0x9e3779b9u
/* The AEC's residual-echo powers are on the int16^2 scale, the denoiser's on the audio scale;
 * the reciprocal is a power of two, so the multiply is exact. */
#define PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))

/* CLI aliases only; unknown names fall back to balanced. */
static AecLevel parse_level(const char* s) {
    if (strcmp(s, "mild") == 0 || strcmp(s, "0") == 0)
        return AEC_LEVEL_MILD;
    if (strcmp(s, "balanced") == 0 || strcmp(s, "1") == 0)
        return AEC_LEVEL_BALANCED;
    if (strcmp(s, "aggressive") == 0 || strcmp(s, "2") == 0)
        return AEC_LEVEL_AGGRESSIVE;
    if (strcmp(s, "no_floor") == 0 || strcmp(s, "3") == 0)
        return AEC_LEVEL_NO_FLOOR;
    return AEC_LEVEL_BALANCED;
}

static const char* level_name(AecLevel level) {
    switch (level) {
        case AEC_LEVEL_MILD:       return "mild";
        case AEC_LEVEL_AGGRESSIVE: return "aggressive";
        case AEC_LEVEL_NO_FLOOR:   return "no_floor";
        default:                   return "balanced";
    }
}

/* Name table lives beside the enum (anr_types.h); unknown names fall back to balanced. */
static AnrLevel parse_nr_level(const char* s) {
    int level = anr_nr_level_from_name(s);
    return level < 0 ? ANR_LEVEL_BALANCED : (AnrLevel)level;
}

static const char* nr_level_name(AnrLevel level) {
    return anr_nr_level_name(level);
}

/* --nr-level is the option; --nr-preset is the same option under its earlier name. */
static int is_nr_level_option(const char* arg) {
    return strcmp(arg, "--nr-level") == 0 || strcmp(arg, "--nr-preset") == 0;
}

/* The mono lane: the rate's grid (fft_size 0: the rate's default) and the linear-stage
 * AecConfig. The lane keeps its own gain in the context (enable_res = 0, return_res_context = 1);
 * mono_form() applies it. Returns 0, or -1 for a pair libaec has no grid for. */
static int derive_lane(int sample_rate, int fft_size, AecLevel level, AecSignalGrid* grid,
                       AecConfig* aec_cfg) {
    if (!aec_resolve_signal_grid(sample_rate,
                                 fft_size ? fft_size : aec_default_fft_size(sample_rate), grid))
        return -1;
    aec_config_from_level(aec_cfg, level, sample_rate);
    aec_cfg->fft_size = grid->fft_size;
    aec_cfg->delay_mode = AEC_DELAY_MATCHED;
    aec_cfg->delay_num_filters = DA_NUM_FILTERS;
    aec_cfg->fixed_delay_samples = -1;
    aec_cfg->enable_delay_est = 1;
    aec_cfg->enable_res = 0;
    aec_cfg->return_res_context = 1;
    return 0;
}

/* What runs after the AEC, and the pool it needs. */
typedef struct PostPlan {
    int with_res, with_nr, use_r2, enable_cng;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    int n_freqs;
    size_t nr_bytes, synth_bytes, buf_bytes;
} PostPlan;

/* One aligned pool: the AEC first, then the denoiser, the synthesis and the gain/spectrum
 * buffers. The split build puts everything after the AEC in post_pool. */
typedef struct Rig {
    Aec* aec;
    void* pool;
    void* post_pool;          /* split build only */
    int with_post;            /* 0 with --aec-only */
    PostPlan plan;
    AnrDenoiser* nr;          /* NULL with --no-nr */
    AcIstft* istft;
    float *unity, *total_gain, *extra_noise;
    Complex* spec;
    uint32_t rng;
    const float* last_nr;     /* the NR gain of the last formed hop (unity without NR) */
} Rig;

static void rig_free(Rig* r) {
    if (r->nr) anr_destroy(r->nr);
    if (r->aec) aec_destroy(r->aec);
    free(r->post_pool);
    free(r->pool);
    memset(r, 0, sizeof(*r));
}

/* 0, or -1 for a grid or mode a stage refuses. */
static int post_plan(PostPlan* q, int sample_rate, const AecSignalGrid* grid, const AecConfig* aec_cfg,
                     AnrLevel nr_level, int no_nr, int no_res, int no_cng, int legacy) {
    memset(q, 0, sizeof(*q));
    q->with_res = !no_res;
    q->with_nr = !no_nr;
    q->use_r2 = !legacy;
    /* Comfort noise fills only bins the RES gain cut, and only when the AEC config allows it. */
    q->enable_cng = aec_cfg->enable_cng && !no_cng;
    q->n_freqs = grid->n_freqs;
    if (q->with_nr) {
        q->nr_cfg = anr_config_for_level_grid(sample_rate, grid->fft_size, nr_level);
        q->nr_bytes = anr_get_mem_size(&q->nr_cfg);
        if (q->nr_bytes == 0) return -1;
    }
    ac_stft_config_default(&q->synth_cfg, grid->fft_size);
    q->synth_bytes = ac_istft_get_mem_size(&q->synth_cfg);
    /* unity, total gain, NR prior, output spectrum */
    q->buf_bytes = 3 * ALIGN16((size_t)q->n_freqs * sizeof(float)) +
                   ALIGN16((size_t)q->n_freqs * sizeof(Complex));
    return q->synth_bytes ? 0 : -1;
}

/* Bytes of the post stages: denoiser, synthesis, gain/spectrum buffers. */
static size_t post_bytes(const PostPlan* q) {
    return ALIGN16(q->nr_bytes) + ALIGN16(q->synth_bytes) + q->buf_bytes;
}

#if !AEC_NR_POOL_SPLIT
/* Bytes of the pool; 0 for an invalid config. `q` NULL: AEC only. */
static size_t pool_bytes(const AecConfig* aec_cfg, const PostPlan* q) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    if (aec_bytes == 0) return 0;
    return ALIGN16(aec_bytes) + (q ? post_bytes(q) : 0);
}
#endif

static int rig_build(Rig* r, const AecConfig* aec_cfg, const PostPlan* q) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    size_t vec;
    uint8_t* cur;
    int k;
    memset(r, 0, sizeof(*r));
#if AEC_NR_POOL_SPLIT
    if (aec_bytes == 0 || posix_memalign(&r->pool, 16, aec_bytes) != 0) { r->pool = NULL; return -1; }
    r->aec = aec_init(r->pool, aec_bytes, aec_cfg);
    if (!r->aec) { rig_free(r); return -1; }
    if (!q) return 0;
    if (posix_memalign(&r->post_pool, 16, post_bytes(q)) != 0) {
        r->post_pool = NULL;
        rig_free(r);
        return -1;
    }
    cur = (uint8_t*)r->post_pool;
#else
    size_t total = pool_bytes(aec_cfg, q);
    if (total == 0) return -1;
    if (posix_memalign(&r->pool, 16, total) != 0) { r->pool = NULL; return -1; }
    cur = (uint8_t*)r->pool;
    r->aec = aec_init(cur, aec_bytes, aec_cfg);
    if (!r->aec) { rig_free(r); return -1; }
    if (!q) return 0;
    cur += ALIGN16(aec_bytes);
#endif
    r->with_post = 1;
    r->plan = *q;
    if (q->with_nr) {
        r->nr = anr_init(cur, q->nr_bytes, &q->nr_cfg);
        if (!r->nr) { rig_free(r); return -1; }
        cur += ALIGN16(q->nr_bytes);
    }
    r->istft = ac_istft_init(cur, q->synth_bytes, &q->synth_cfg);
    if (!r->istft) { rig_free(r); return -1; }
    cur += ALIGN16(q->synth_bytes);
    vec = ALIGN16((size_t)q->n_freqs * sizeof(float));
    r->unity = (float*)cur;       cur += vec;
    r->total_gain = (float*)cur;  cur += vec;
    r->extra_noise = (float*)cur; cur += vec;
    r->spec = (Complex*)cur;
    for (k = 0; k < q->n_freqs; ++k) r->unity[k] = 1.0f;
    r->rng = CNG_RNG_SEED;
    return 0;
}

/* A stage refused a hop: back to the state of a fresh instance. */
static void rig_reset(Rig* r) {
    aec_reset(r->aec);
    if (r->nr) anr_reset(r->nr);
    if (r->istft) ac_istft_reset(r->istft);
    r->rng = CNG_RNG_SEED;
    r->last_nr = NULL;
}

/* The denoiser gain of this hop. With use_r2 and a usable linear estimate the residual-echo
 * powers fold into its noise floor; otherwise plain noise reduction. NULL for a refused hop. */
static const float* mono_nr_gain(Rig* r, const AecResContext* c) {
    const float* extra = NULL;
    if (r->plan.use_r2 && c->r2 && c->usable_linear) {
        sk_scale_f32(r->extra_noise, c->r2, PSD_SCALE_INV, r->plan.n_freqs);
        extra = r->extra_noise;
    }
    if (anr_process_gain(r->nr, c->error_spec, extra, NULL) != 0) return NULL;
    return anr_get_gain(r->nr, NULL);
}

/* P = E * min(G_nr, G_res) plus comfort noise on the bins the RES cut; an absent gain is the
 * unity vector. NULL when a stage refuses the hop. */
static const Complex* mono_form(Rig* r, const AecResContext* c) {
    const PostPlan* q = &r->plan;
    const float* res = q->with_res ? c->res_gain : r->unity;
    const float* nr = r->unity;
    if (!res) return NULL;
    if (r->nr && !(nr = mono_nr_gain(r, c))) return NULL;
    sk_min_f32(r->total_gain, nr, res, q->n_freqs);   /* NaN in the second operand reaches the finite gate; in the first it is replaced */
    sk_capply_gain_f32(r->spec, c->error_spec, r->total_gain, q->n_freqs);
    if (q->enable_cng && q->with_res && c->comfort_noise)
        aec_cng_fill(&r->rng, r->spec, c->comfort_noise, res, nr, CNG_NR_GAIN_FLOOR, q->n_freqs);
    r->last_nr = nr;
    return r->spec;
}

/* --debug: one status line/sec to stderr from the read-only aec_debug_status() /
 * anr_debug_status() queries; the NR half is omitted when no denoiser exists. */
static void print_debug_status(const Aec* aec, const AnrDenoiser* nr, int aec_only, float seconds) {
    AecDebugStatus a;
    aec_debug_status(aec, &a);
    if (aec_only || !nr) {
        fprintf(stderr,
            "[dbg %5.1fs] aec: delay=%d conf=%.1f upd=%d erle=%.1fdB lin=%d conv=%d "
            "near=%.2e out=%.2e | nr: n/a (--aec-only)\n",
            seconds, a.delay_samples, a.delay_confidence, a.delay_updates,
            a.erle_windowed_db, a.usable_linear, a.filter_converged,
            a.near_power, a.out_power);
        return;
    }
    AnrDebugStatus n;
    anr_debug_status(nr, &n);
    fprintf(stderr,
        "[dbg %5.1fs] aec: delay=%d conf=%.1f upd=%d erle=%.1fdB lin=%d conv=%d "
        "near=%.2e out=%.2e | nr: init=%d gain=%.1f/%.1fdB spp=%.2f noise=%.1fdB\n",
        seconds, a.delay_samples, a.delay_confidence, a.delay_updates,
        a.erle_windowed_db, a.usable_linear, a.filter_converged,
        a.near_power, a.out_power,
        n.initialized, n.mean_gain_db, n.min_gain_db, n.mean_spp, n.noise_floor_db);
}

/* --timing: per-stage cost summed over the run and reported as a mean per hop. The AEC rows
 * come from aec_get_last_timing() (built with -DAEC_STAGE_TIMING=1, `make PROFILE=1`); the
 * gain-chain and synthesis rows are this program's own clock around the calls. */
typedef struct {
    uint64_t delay, frontend, linear, res, chain, synth, wall;
    uint64_t hops;
} StageAccum;

static void stage_accum_add(StageAccum* a, const AecStageTiming* t, uint32_t chain_us,
                            uint32_t synth_us, uint32_t wall_us) {
    a->delay    += t->delay_us;
    a->frontend += t->frontend_us;
    a->linear   += t->linear_us;
    a->res      += t->res_us;
    a->chain    += chain_us;
    a->synth    += synth_us;
    a->wall     += wall_us;
    a->hops     += 1u;
}

static void stage_accum_row(const char* name, uint64_t total,
                            const StageAccum* a) {
    double per_hop = a->hops ? (double)total / (double)a->hops : 0.0;
    double share   = a->wall ? 100.0 * (double)total / (double)a->wall : 0.0;
    printf("  %-14s %8.2f  %6.1f%%\n", name, per_hop, share);
}

static void print_stage_timing(const StageAccum* a) {
    uint64_t aec = a->delay + a->frontend + a->linear + a->res;
    uint64_t measured = aec + a->chain + a->synth;
    if (!a->hops) { printf("\nStage timing: no hops processed.\n"); return; }
    printf("\nStage timing over %llu hops (mean us/hop, share of wall):\n",
           (unsigned long long)a->hops);
    if (aec == 0u) {
        printf("  the AEC rows read 0 -- this binary was built WITHOUT\n"
               "  -DAEC_STAGE_TIMING=1. Rebuild with `make PROFILE=1` to\n"
               "  measure them; a build that substituted a constant stamp\n"
               "  through AEC_NOW_US reads 0 here too, and this report cannot\n"
               "  tell the two apart.\n");
    }
    stage_accum_row("AEC delay", a->delay, a);
    stage_accum_row("AEC frontend", a->frontend, a);
    stage_accum_row("AEC linear", a->linear, a);
    stage_accum_row("AEC post/RES", a->res, a);
    stage_accum_row("gain chain", a->chain, a);
    stage_accum_row("synthesis", a->synth, a);
    printf("  ------------------------------------\n");
    stage_accum_row("measured", measured, a);
    stage_accum_row("unattributed",
                    a->wall > measured ? a->wall - measured : 0u, a);
    stage_accum_row("call wall", a->wall, a);
}

static uint32_t timing_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull
                      + (uint64_t)ts.tv_nsec / 1000ull);
}

/* One hop. --aec-only is the AEC's own time-domain output; otherwise the post stages over the
 * AEC's frame. The clock is read only when `timed`. Returns 0, or -1 after a refused hop (the
 * caller resets). */
static int process_hop(Rig* r, const float* mic, const float* ref, float* out, int timed,
                       uint32_t* chain_us, uint32_t* synth_us) {
    AecResContext ctx;
    const Complex* spec;
    uint32_t t0, t1;
    *chain_us = 0u;
    *synth_us = 0u;
    if (!r->with_post) {
        aec_process(r->aec, mic, ref, out);
        return 0;
    }
    aec_process_context(r->aec, mic, ref);
    aec_get_res_context(r->aec, &ctx);
    if (!ctx.error_spec) return -1;
    t0 = timed ? timing_now_us() : 0u;
    spec = mono_form(r, &ctx);
    t1 = timed ? timing_now_us() : 0u;
    *chain_us = t1 - t0;
    if (!spec) return -1;
    (void)ac_istft_push(r->istft, spec, out);
    *synth_us = timed ? timing_now_us() - t1 : 0u;
    return 0;
}

#if !AEC_NR_POOL_SPLIT
/* --print-mem-size table: the AEC block and the post stages. */
static void print_mem_budget(int sample_rate, const AecSignalGrid* grid, const AecConfig* aec_cfg,
                             const PostPlan* q) {
    size_t aec_bytes = aec_get_mem_size(aec_cfg);
    size_t total = pool_bytes(aec_cfg, q);
    if (total == 0) {
        fprintf(stderr, "Error: invalid config for memory budget query\n");
        return;
    }
    printf("Memory Budget (Static Pipeline)\n");
    printf("================================\n");
    printf("  sample_rate=%d hop=%d fft_sz=%d n_freqs=%d\n",
           sample_rate, grid->hop_size, grid->fft_size, grid->n_freqs);
    printf("  AEC:            %7zu bytes (%6.1f KB)\n", aec_bytes, (float)aec_bytes / 1024.0f);
    if (q) {
        printf("  Synthesis:      %7zu bytes (%6.1f KB)\n", q->synth_bytes, (float)q->synth_bytes / 1024.0f);
        printf("  NR (MMSE-LSA):  %7zu bytes (%6.1f KB)\n", q->nr_bytes, (float)q->nr_bytes / 1024.0f);
        printf("  Gain/spectrum buffers: %zu bytes\n", q->buf_bytes);
    }
    printf("  --------------------------------\n");
    printf("  Total:          %7zu bytes (%6.1f KB)\n", total, (float)total / 1024.0f);
    printf("\n");
}
#endif

int main(int argc, char* argv[]) {
#if !AEC_NR_POOL_SPLIT
    /* --print-mem-size: report the pool byte budget and run a real init of every stage on
     * it, without WAV I/O. */
    if (argc >= 2 && strcmp(argv[1], "--print-mem-size") == 0) {
        AecLevel      level       = AEC_LEVEL_BALANCED;
        AnrLevel      nr_level    = ANR_LEVEL_BALANCED;
        int           aec_only    = 0;
        int           no_nr = 0, no_res = 0, no_cng = 0;
        int           have_level  = 0;
        int           sample_rate = 16000;
        int           fft_size    = 0;

        for (int i = 2; i < argc; i++) {
            if      (strcmp(argv[i], "--aec-only") == 0) aec_only = 1;
            else if (strcmp(argv[i], "--no-nr") == 0) no_nr = 1;
            else if (strcmp(argv[i], "--no-res") == 0) no_res = 1;
            else if (strcmp(argv[i], "--no-cng") == 0) no_cng = 1;
            else if (is_nr_level_option(argv[i]) ||
                     strcmp(argv[i], "--sample-rate") == 0 ||
                     strcmp(argv[i], "--fft-size") == 0) {
                const char* name = argv[i];
                const char* value = example_cli_option_value(argc, argv, &i, argv[0]);
                if (!value) return 2;
                if (is_nr_level_option(name)) nr_level = parse_nr_level(value);
                else if (strcmp(name, "--sample-rate") == 0) sample_rate = atoi(value);
                else fft_size = atoi(value);
            }
            else if (strcmp(argv[i], "--aec-level") == 0 || argv[i][0] != '-') {
                /* The positional and --aec-level are the same setting: one per command line. */
                const char* text = argv[i];
                if (argv[i][0] == '-') {
                    text = example_cli_option_value(argc, argv, &i, argv[0]);
                    if (!text) return 2;
                }
                if (have_level) {
                    fprintf(stderr, "%s: a second AEC level '%s'; give one\n", argv[0], text);
                    return 2;
                }
                level = parse_level(text);
                have_level = 1;
            }
        }

        AecSignalGrid grid;
        AecConfig aec_cfg;
        if (derive_lane(sample_rate, fft_size, level, &grid, &aec_cfg) != 0) {
            fprintf(stderr, "Error: invalid config (sample_rate=%d?)\n", sample_rate);
            return 1;
        }
        PostPlan plan;
        if (post_plan(&plan, sample_rate, &grid, &aec_cfg, nr_level, no_nr, no_res, no_cng, 0) != 0) {
            fprintf(stderr, "Error: invalid config (sample_rate=%d?)\n", sample_rate);
            return 1;
        }
        const PostPlan* pp = aec_only ? NULL : &plan;

        print_mem_budget(sample_rate, &grid, &aec_cfg, pp);

        Rig rig;
        if (rig_build(&rig, &aec_cfg, pp) != 0) {
            fprintf(stderr, "Error: invalid config (sample_rate=%d?)\n", sample_rate);
            return 1;
        }
        printf("n_freqs agreement OK (n_freqs=%d, hop=%d) at %d Hz\n",
               grid.n_freqs, grid.hop_size, sample_rate);
        rig_free(&rig);
        return 0;
    }
#endif

    if (argc < 4) {
        printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [aec-level | --aec-level <level>] "
               "[--nr-level mild|moderate|balanced|aggressive] [--aec-only] [--no-nr] [--no-res] [--legacy-amin] "
               "[--fft-size 256|512|1024] [--debug] [--timing]\n"
               "AEC level: 0=mild, 1=balanced, 2=aggressive, 3=no_floor; names also accepted. The positional [aec-level] and --aec-level are the same setting; give one of them once.\n"
               "NR level: mild, moderate, balanced, aggressive or 0..3; --nr-preset is the same option.\n",
               argv[0]);
#if !AEC_NR_POOL_SPLIT
        printf("       %s --print-mem-size [aec-level | --aec-level <level>] [--nr-level ...] [--aec-only] [--no-nr] [--no-res] "
               "[--sample-rate <hz>] [--fft-size <n>]\n", argv[0]);
#endif
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    AecLevel      level    = AEC_LEVEL_BALANCED;
    AnrLevel      nr_level = ANR_LEVEL_BALANCED;
    int           have_level = 0;
    int           aec_only = 0;
    int           legacy   = 0;   /* --legacy-amin → min-only A_min_pl */
    int           no_nr    = 0;   /* --no-nr → no denoiser, g_total is G_res */
    int           no_res   = 0;   /* --no-res → no residual gain, g_total is G_nr */
    int           no_cng   = 0;   /* --no-cng -> disable comfort noise */
    int           debug_status = 0; /* --debug → periodic aec+nr status line   */
    int           show_timing  = 0; /* --timing → per-stage cost summary at exit */
    StageAccum    acc; memset(&acc, 0, sizeof(acc));
    int           fft_size = 0;

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "--aec-only") == 0)    aec_only = 1;
        else if (strcmp(argv[i], "--legacy-amin") == 0) legacy = 1;
        else if (strcmp(argv[i], "--no-nr") == 0)       no_nr = 1;
        else if (strcmp(argv[i], "--no-res") == 0)      no_res = 1;
        else if (strcmp(argv[i], "--no-cng") == 0)      no_cng = 1;
        else if (strcmp(argv[i], "--debug") == 0)       debug_status = 1;
        else if (strcmp(argv[i], "--timing") == 0)      show_timing = 1;
        else if (strcmp(argv[i], "--fft-size") == 0 ||
                 is_nr_level_option(argv[i])) {
            const char* name = argv[i];
            const char* value = example_cli_option_value(argc, argv, &i, argv[0]);
            if (!value) return 2;
            if (strcmp(name, "--fft-size") == 0) fft_size = atoi(value);
            else nr_level = parse_nr_level(value);
        }
        else if (strcmp(argv[i], "--aec-level") == 0 || argv[i][0] != '-') {
            const char* text = argv[i];
            if (argv[i][0] == '-') {
                text = example_cli_option_value(argc, argv, &i, argv[0]);
                if (!text) return 2;
            }
            if (have_level) {
                fprintf(stderr, "%s: a second AEC level '%s'; give one\n", argv[0], text);
                return 2;
            }
            level = parse_level(text);
            have_level = 1;
        }
    }

    /* Open inputs */
    WavReader* mic_r = wav_open_read(mic_path);
    WavReader* ref_r = wav_open_read(ref_path);
    if (!mic_r || !ref_r) { fprintf(stderr, "Error: cannot open inputs\n"); return 1; }
    if (mic_r->info.sample_rate != ref_r->info.sample_rate) {
        fprintf(stderr, "Error: sample-rate mismatch\n");
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }
    /* CLI rate whitelist, checked before any buffers are sized. */
    if (!aec_is_valid_sample_rate(mic_r->info.sample_rate)) {
        fprintf(stderr, "Error: unsupported sample rate %d Hz (supported: 8000, 16000, 48000)\n",
                mic_r->info.sample_rate);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    int sr        = mic_r->info.sample_rate;
    int n_samples = (mic_r->info.num_samples < ref_r->info.num_samples)
                  ? mic_r->info.num_samples : ref_r->info.num_samples;

    AecSignalGrid grid;
    AecConfig aec_cfg;
    int lane_ok = derive_lane(sr, fft_size, level, &grid, &aec_cfg) == 0;

    printf("AEC(linear) -> echo-aware NR -> RES  (%s%s)\n",
           AEC_NR_POOL_SPLIT ? "freq A_min_pl" : "static memory", legacy ? ", legacy min-only" : "");
    printf("  Input:  %s (%.2fs)\n", mic_path, (float)n_samples / sr);
    /* Wording kept as in earlier releases: the baseline matrix compares this program's stdout byte for byte. */
    printf("  AEC preset: %s   NR preset: %s   NR: %s   RES: %s   CNG: %s\n\n",
           level_name(level), nr_level_name(nr_level), !no_nr ? "on" : "off",
           !no_res ? "on" : "off", !no_cng ? "on" : "off");

    /* Size the pool, then allocate it (a host stand-in for a platform memory block; a board
     * replaces this malloc with its own allocator). The split build plans no post stages with
     * --aec-only. */
    PostPlan plan;
    memset(&plan, 0, sizeof(plan));
#if AEC_NR_POOL_SPLIT
    if (lane_ok && !aec_only)
        lane_ok = post_plan(&plan, sr, &grid, &aec_cfg, nr_level, no_nr, no_res, no_cng, legacy) == 0;
#else
    if (lane_ok) {
        lane_ok = post_plan(&plan, sr, &grid, &aec_cfg, nr_level, no_nr, no_res, no_cng, legacy) == 0;
        if (lane_ok) print_mem_budget(sr, &grid, &aec_cfg, aec_only ? NULL : &plan);
    }
#endif

    Rig rig;
    if (!lane_ok || rig_build(&rig, &aec_cfg, aec_only ? NULL : &plan) != 0) {
        fputs(AEC_NR_POOL_SPLIT ? "Error: pipeline create failed\n" : "Error: invalid config\n", stderr);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    int hop = grid.hop_size;
    int n_freqs = grid.n_freqs;
    float* mic_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* out_buf = (float*)malloc((size_t)hop * sizeof(float));
    if (!mic_buf || !ref_buf || !out_buf) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        rig_free(&rig);
        free(mic_buf); free(ref_buf); free(out_buf);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    WavWriter* writer = wav_open_write(out_path, sr, 1);
    if (!writer) {
        fprintf(stderr, "Error: cannot create output\n");
        rig_free(&rig);
        free(mic_buf); free(ref_buf); free(out_buf);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    /* Optional per-frame context dump (DUMP_CTX=<path>): header [n_freqs, hop]; then per
     * frame error_spec(2*nf) res_gain(nf) r2(nf) comfort_noise(nf) far_power(1) g_nr(nf)
     * out_hop(hop), all float32. res_gain is the lane's own gain. g_nr is the NR gain, the
     * unity vector without a denoiser. Written only for hops whose freq seam was available. */
    FILE* dctx = NULL;
    const char* dpath = getenv("DUMP_CTX");
    if (dpath && !aec_only) {
        dctx = fopen(dpath, "wb");
        if (dctx) { int hdr[2] = { n_freqs, hop }; fwrite(hdr, sizeof(int), 2, dctx); }
    }

    /* Processing loop */
    int processed = 0;
    while (processed + hop <= n_samples) {
        uint32_t chain_us, synth_us, t_in = 0u, wall = 0u;
        wav_read_float(mic_r, mic_buf, hop);
        wav_read_float(ref_r, ref_buf, hop);

        if (show_timing) t_in = timing_now_us();
        if (process_hop(&rig, mic_buf, ref_buf, out_buf, show_timing, &chain_us, &synth_us) != 0) {
            rig_reset(&rig);
            memset(out_buf, 0, (size_t)hop * sizeof(float));
        }
        if (show_timing) {
            AecStageTiming t;
            wall = timing_now_us() - t_in;
            aec_get_last_timing(rig.aec, &t);
            stage_accum_add(&acc, &t, chain_us, synth_us, wall);
        }
        wav_write_float(writer, out_buf, hop);

        if (dctx) {
            AecResContext ctx;
            aec_get_res_context(rig.aec, &ctx);
            const float* g_nr = rig.last_nr;
            if (ctx.error_spec && ctx.res_gain && g_nr) {
                fwrite(ctx.error_spec, sizeof(Complex), n_freqs, dctx);
                fwrite(ctx.res_gain, sizeof(float), n_freqs, dctx);
                fwrite(ctx.r2, sizeof(float), n_freqs, dctx);
                fwrite(ctx.comfort_noise, sizeof(float), n_freqs, dctx);
                float fp = ctx.far_power; fwrite(&fp, sizeof(float), 1, dctx);
                fwrite(g_nr, sizeof(float), n_freqs, dctx);
                fwrite(out_buf, sizeof(float), hop, dctx);
            }
        }

        processed += hop;
        if (processed % sr == 0) {
            printf("."); fflush(stdout);
            if (debug_status)
                print_debug_status(rig.aec, rig.nr, aec_only,
                                   (float)processed / sr);
        }
    }
    if (dctx) fclose(dctx);

    printf("\nProcessed: %d samples (%.2fs)\n", processed, (float)processed / sr);
    if (show_timing) print_stage_timing(&acc);
    printf("Output: %s\n", out_path);

    /* Cleanup: the stages live in the rig's pool (pools); rig_free() destroys them, then frees. */
    wav_close_read(mic_r);
    wav_close_read(ref_r);
    wav_close_write(writer);
    rig_free(&rig);
    free(mic_buf); free(ref_buf); free(out_buf);
    return 0;
}
