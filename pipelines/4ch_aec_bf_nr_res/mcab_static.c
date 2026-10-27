/** Four-channel linear AEC flow (mcab.h) with the sample SRP-PHAT/BF helper (mcab_sample_beam.h)
 * followed by the post-beam RES (lib/aec AecRes), NR, comfort noise and synthesis, static memory.
 * Host reference of the application side: the lanes' and the estimator's own get_mem_size/init,
 * mcab's get_mem_requirements/init_ex, the helper on top -> process -> fuse -> RES / NR / min /
 * CNG / synthesis -> destroy -> release, with the real SRP-PHAT DOA and beamformer running in this
 * process. The unit and the post stages are initialised separately, each in its own pool.
 * Build:  make -C pipelines/4ch_aec_bf_nr_res mcab_static
 * Usage:
 *   ./mcab_static [--sample-rate 16000|48000] [--fft-size 256|512|1024]
 *       [--uca-radius-m 0.035] [--fixed-doa-deg DEG]
 *   ./mcab_static --print-mem-size [same options]
 * Numbers are whole decimals. Exit: 0 done, 1 refused config or failed run, 2 command line. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec_res.h"
#include "anr.h"
#include "mcab.h"
#include "mcab_sample_beam.h"
#include "audio_stft.h"
#include "mcab_sample_driver.h"
#include "mcab_sample_post.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "utility/simd_kernel_nn.h"   /* skn_all_finite_f32 */

static void fill_hop(float* microphones, float* reference, int hop,
                     int sample_rate, int frame_index) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        int absolute = frame_index * hop + i;
        float phase =
            2.0f * 3.14159265358979323846f * 440.0f *
            (float)absolute / (float)sample_rate;
        reference[i] = 0.08f * sinf(phase);
        for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
            /* Per-channel scale and a small phase offset give SRP-PHAT/BF a non-degenerate
             * four-channel signal. */
            float ch_phase = phase - 0.05f * (float)ch;
            microphones[i * MCAB_CHANNELS + ch] =
                (0.35f + 0.05f * (float)ch) * sinf(ch_phase);
        }
    }
}

/* The application side of mcab.h plus the sample beamformer helper, carved in this order from
 * one pool: four lanes built from mcab_lane_config(), the shared estimator object and its pool
 * (MATCHED only), mcab's own pool, the helper (SRP-PHAT, BF, fallback VAD). */
typedef struct Unit {
    AecConfig lane_cfg;
    McabDelayEstimatorConfig dc;
    McabMemReq req;           /* mcab's own pool */
    size_t lane_bytes, delay_bytes, beam_bytes, total;
    Aec* lanes[MCAB_CHANNELS];
    DelayAec* delay;
    Mcab* flow;
    McabSampleBeam* beam;
} Unit;

static int unit_plan(Unit* u, const McabConfig* cfg, const McabSampleBeamParams* bp) {
    int ch;
    memset(u, 0, sizeof(*u));
    if (mcab_lane_config(cfg, &u->lane_cfg) != 0 ||
        mcab_delay_estimator_config(cfg, &u->dc) != 0 ||
        mcab_get_mem_requirements(cfg, &u->req) != 0 ||
        mcab_sample_beam_get_mem_size(cfg, bp, &u->beam_bytes) != 0) return -1;
    u->lane_bytes = aec_get_mem_size(&u->lane_cfg);
    if (u->lane_bytes == 0) return -1;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) u->total += ALIGN16(u->lane_bytes);
    if (u->dc.required) {
        u->delay_bytes = delay_aec_get_mem_size(u->dc.sample_rate, u->dc.hop_size,
                                                u->dc.num_filters);
        if (u->delay_bytes == 0) return -1;
        u->total += ALIGN16(sizeof(DelayAec)) + ALIGN16(u->delay_bytes);
    }
    u->total += ALIGN16((size_t)u->req.bytes) + u->beam_bytes;
    return 0;
}

/* Each object tears down before the ones it borrows: helper, flow, lanes. */
static void unit_destroy(Unit* u) {
    int ch;
    mcab_sample_beam_destroy(u->beam);
    u->beam = NULL;
    mcab_destroy(u->flow);
    u->flow = NULL;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        if (u->lanes[ch]) aec_destroy(u->lanes[ch]);
        u->lanes[ch] = NULL;
    }
}

static int unit_init(Unit* u, void* mem, const McabConfig* cfg, const McabSampleBeamParams* bp) {
    unsigned char* cur = (unsigned char*)mem;
    McabLibs libs;
    int ch;
    memset(&libs, 0, sizeof(libs));
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        u->lanes[ch] = aec_init(cur, u->lane_bytes, &u->lane_cfg);
        if (!u->lanes[ch]) return -1;
        libs.lanes[ch] = u->lanes[ch];
        cur += ALIGN16(u->lane_bytes);
    }
    if (u->dc.required) {
        u->delay = (DelayAec*)cur;
        cur += ALIGN16(sizeof(DelayAec));
        if (delay_aec_init(u->delay, cur, u->delay_bytes, u->dc.sample_rate, u->dc.hop_size,
                           u->dc.num_filters) != 0) return -1;
        cur += ALIGN16(u->delay_bytes);
        libs.delay = u->delay;
    }
    u->flow = mcab_init_ex(cur, (size_t)u->req.bytes, cfg, &libs, &u->req);
    if (!u->flow) return -1;
    cur += ALIGN16((size_t)u->req.bytes);
    u->beam = mcab_sample_beam_init(cur, u->beam_bytes, cfg, bp, u->flow);
    return u->beam ? 0 : -1;
}

/* The stages after the fuse, carved in this order from one aligned pool. */
typedef struct PostStages {
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t res_bytes, nr_bytes, synth_bytes, vec_bytes, spec_bytes, total;
} PostStages;

static int post_plan(int sample_rate, int fft_size, PostStages *ps) {
    int n_freqs = fft_size / 2 + 1;
    aec_config_from_preset(&ps->res_cfg, AEC_PRESET_BALANCED, sample_rate);
    ps->res_cfg.fft_size = fft_size;
    ps->res_bytes = aec_res_get_mem_size(&ps->res_cfg);
    ps->nr_cfg = anr_config_for_mode_grid(sample_rate, fft_size, ANR_MODE_BALANCED);
    ac_stft_config_default(&ps->synth_cfg, fft_size);
    ps->nr_bytes = anr_get_mem_size(&ps->nr_cfg);
    ps->synth_bytes = ac_istft_get_mem_size(&ps->synth_cfg);
    ps->vec_bytes = ALIGN16((size_t)n_freqs * sizeof(float));
    ps->spec_bytes = ALIGN16((size_t)n_freqs * sizeof(Complex));
    if (ps->res_bytes == 0 || ps->nr_bytes == 0 || ps->synth_bytes == 0) return -1;
    /* res, nr, synthesis, total gain, NR prior, output spectrum */
    ps->total = ALIGN16(ps->res_bytes) + ALIGN16(ps->nr_bytes) + ALIGN16(ps->synth_bytes) +
                2 * ps->vec_bytes + ps->spec_bytes;
    return 0;
}

typedef struct StaticSample {
    McabSampleSpatialOptions options;
    McabConfig cfg;
    McabSampleBeamParams bp;
    Unit unit;
    PostStages post;
    void *pool, *post_pool;
    McabSampleBeam *p;
    AecRes *res;
    AnrDenoiser *nr;
    AcIstft *istft;
    float *total_gain, *extra_noise;
    Complex *spec;
    uint32_t rng;
    int n_freqs;
    McabSampleBeamResult beam;
    float *mic_buf,*ref_buf,*out_buf;
    int hop,saw_finite_doa_raw,saw_bf_adaptive;
    McabSampleBeamInfo info;
} StaticSample;
static int static_parse(void *user, int argc, char **argv) {
    StaticSample *s=user;
    if (mcab_sample_parse_spatial_cli(argc,argv,0,&s->options) != 0) {
        fprintf(stderr,
            "Usage: %s [--sample-rate 16000|48000] "
            "[--fft-size 256|512|1024] [--uca-radius-m 0.035] "
            "[--fixed-doa-deg DEG] [--print-mem-size]\n",argv[0]);
        return 2;
    }
    return 0;
}
static int static_setup(void *user, McabSampleHopView *view) {
    StaticSample *s=user;
    s->cfg = mcab_default_config(s->options.sample_rate);
    s->bp = mcab_sample_beam_default_params();
    if (s->options.fft_size != 0) s->cfg.fft_size = s->options.fft_size;
    s->bp.uca_radius_m = s->options.uca_radius_m;
    if (s->options.fixed_doa) {
        s->bp.bf_fixed_mode = 1;
        s->bp.bf_fixed_doa_rad = s->options.fixed_doa_rad;
    }

    if (post_plan(s->cfg.sample_rate,
                  mcab_sample_resolved_fft(s->cfg.sample_rate, s->cfg.fft_size),
                  &s->post) != 0) {
        fprintf(stderr, "Error: invalid post-stage grid\n");
        return 1;
    }

    printf("4ch linear AEC + BF unit and post-beam chain (static memory)\n");
    printf("  Geometry: UCA radius=%.4f m   num_angles=%d\n",
           s->bp.uca_radius_m, s->bp.num_angles);

    {
        McabSampleMemLine lines[4] = {
            {"RES:", s->post.res_bytes}, {"NR:", s->post.nr_bytes},
            {"Synthesis:", s->post.synth_bytes},
            {"Buffers:", 2 * s->post.vec_bytes + s->post.spec_bytes}};
        if (mcab_sample_print_mem_budget(
                "4ch linear AEC + BF unit, RES, NR and synthesis", &s->cfg, &s->bp,
                mcab_sample_unit_memory, lines, 4) != 0)
            return 1;
    }

    /* Query -> allocate -> init; the aligned allocation stands in for the board memory
     * manager. */
    if (unit_plan(&s->unit, &s->cfg, &s->bp) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        return 1;
    }

    s->pool = NULL;
    if (posix_memalign(&s->pool, s->unit.req.alignment, s->unit.total) != 0)
        s->pool = NULL;
    if (!s->pool) {
        fprintf(stderr, "Error: aligned allocation failed (%llu bytes)\n",
                (unsigned long long)s->unit.total);
        return 1;
    }

    if (unit_init(&s->unit, s->pool, &s->cfg, &s->bp) != 0) {
        fprintf(stderr, "Error: mcab_init_ex failed\n");
        return 1;
    }
    s->p = s->unit.beam;

    {
        unsigned char *cur;
        if (posix_memalign(&s->post_pool, 16, s->post.total) != 0) s->post_pool = NULL;
        cur = s->post_pool;
        if (cur) {
            s->res = aec_res_init(cur, s->post.res_bytes, &s->post.res_cfg);
            cur += ALIGN16(s->post.res_bytes);
            s->nr = anr_init(cur, s->post.nr_bytes, &s->post.nr_cfg);
            cur += ALIGN16(s->post.nr_bytes);
            s->istft = ac_istft_init(cur, s->post.synth_bytes, &s->post.synth_cfg);
            cur += ALIGN16(s->post.synth_bytes);
            s->total_gain = (float *)cur;  cur += s->post.vec_bytes;
            s->extra_noise = (float *)cur; cur += s->post.vec_bytes;
            s->spec = (Complex *)cur;
        }
        if (!s->res || !s->nr || !s->istft) {
            fprintf(stderr, "Error: post-beam stage init failed\n");
            return 1;
        }
        s->n_freqs = mcab_n_freqs(s->unit.flow);
        s->rng = MCAB_SAMPLE_CNG_RNG_SEED;
    }

    s->hop = mcab_hop_size(s->unit.flow);
    printf("hop=%d fft=%d n_freqs=%d at %d Hz\n", s->hop,
           mcab_fft_size(s->unit.flow), mcab_n_freqs(s->unit.flow),
           s->options.sample_rate);
    if (s->options.print_mem_size) {
        return 2;
    }

    s->mic_buf =
        (float*)malloc((size_t)s->hop * MCAB_CHANNELS * sizeof(float));
    s->ref_buf = (float*)malloc((size_t)s->hop * sizeof(float));
    s->out_buf = (float*)malloc((size_t)s->hop * sizeof(float));
    if (!s->mic_buf || !s->ref_buf || !s->out_buf) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        return 1;
    }

    view->hop=s->hop; view->microphones=s->mic_buf;
    view->reference=s->ref_buf; view->output=s->out_buf;
    return 0;
}
static int static_read(void *user, McabSampleHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    if (frame >= 48) return 0;
    fill_hop(s->mic_buf,s->ref_buf,s->hop,s->options.sample_rate,frame);
    return 1;
}
/* One hop after the fuse into s->out_buf: G_res, G_nr, P = E * min(G_nr, G_res), comfort noise on
 * the bins the RES cut, synthesis. 0, or -1 when the RES or the denoiser refuses the hop. */
static int static_chain(StaticSample *s, const McabFrame *f) {
    AecResInput in;
    const float *res, *nr;
    mcab_sample_res_input(f, &in);
    res = aec_res_process(s->res, &in);
    nr = res ? mcab_sample_anr_gain(s->nr, f, s->extra_noise, 1) : NULL;
    if (!nr) return -1;
    sk_min_f32(s->total_gain, nr, res, s->n_freqs);   /* NaN in the second operand reaches the finite gate; in the first it is replaced */
    sk_capply_gain_f32(s->spec, f->e, s->total_gain, s->n_freqs);
    if (f->cn)
        aec_cng_fill(&s->rng, s->spec, f->cn, res, nr, MCAB_SAMPLE_CNG_NR_GAIN_FLOOR,
                     s->n_freqs);
    (void)ac_istft_push(s->istft, s->spec, s->out_buf);
    return 0;
}

static void static_post_reset(StaticSample *s) {
    aec_res_reset(s->res);
    anr_reset(s->nr);
    ac_istft_reset(s->istft);
    s->rng = MCAB_SAMPLE_CNG_RNG_SEED;
}

/* Unit, fuse, post stages. A refusal anywhere takes them all down: the unit resets itself on its
 * own refusals, the rest is reset here. */
static int static_hop(StaticSample *s, int vad_active) {
    McabFrame frame;
    int status=mcab_sample_beam_process_with_activity(s->p,s->mic_buf,s->ref_buf,
            1,vad_active,NULL,&s->beam,&s->info);
    if (status != MCAB_OK) { static_post_reset(s); return status; }
    status=mcab_sample_beam_fuse(s->p,&frame);
    if (status == MCAB_OK &&
        (static_chain(s,&frame) != 0 || !skn_all_finite_f32(s->out_buf, (size_t)s->hop))) {
        mcab_sample_beam_reset(s->p);
        status = MCAB_DSP_ERROR;
    }
    if (status != MCAB_OK) static_post_reset(s);
    return status;
}
static int static_process(void *user, McabSampleHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    if (static_hop(s,frame < 8 ? 1 : 0) != MCAB_OK) {
        fprintf(stderr,"Error: 4ch hop (unit, fuse, post-beam chain) failed\n");
        return 1;
    }
    return 0;
}
static int static_consume(void *user, McabSampleHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    if (isfinite(s->info.doa_raw_rad)) s->saw_finite_doa_raw = 1;
    if (s->info.bf_adaptive) s->saw_bf_adaptive = 1;
    if (frame == 48 - 1) {
        printf("final frame: doa_raw=%.4f doa_smooth=%.4f doa_used=%.4f "
               "vad_out=%d bf_adaptive=%d\n",
               s->info.doa_raw_rad, s->info.doa_smooth_rad, s->info.doa_used_rad,
               s->info.vad_out, s->info.bf_adaptive);
    }
    return 0;
}
static int static_finish(void *user, McabSampleHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    printf("Processed: %d frames (%d samples)\n", frame, frame * s->hop);
    printf("  saw a finite doa_raw at least once: %s\n",
           s->saw_finite_doa_raw ? "yes" : "no");
    printf("  saw bf_adaptive=1 at least once:   %s\n",
           s->saw_bf_adaptive ? "yes" : "no");
    if (!s->saw_finite_doa_raw || !s->saw_bf_adaptive) {
        fprintf(stderr,
                "Error: SRP/BF never reached the expected adaptive state "
                "over %d frames (finite doa_raw=%d, bf_adaptive=%d) -- "
                "the pipeline ran without crashing, but that alone does "
                "not prove SRP-PHAT/BF actually did anything\n",
                frame, s->saw_finite_doa_raw, s->saw_bf_adaptive);
        return 1;
    }
    printf("mcab_static: smoke PASS\n");
    return 0;
}
static void static_cleanup(void *user) {
    StaticSample *s=user;
    if (s->nr) anr_destroy(s->nr);
    free(s->post_pool);
    unit_destroy(&s->unit); free(s->pool);
    free(s->mic_buf); free(s->ref_buf); free(s->out_buf);
}
int main(int argc, char **argv) {
    StaticSample sample={0};
    static const McabSampleOps ops={static_parse,static_setup,static_read,
        static_process,static_consume,NULL,static_finish,static_cleanup,NULL};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
