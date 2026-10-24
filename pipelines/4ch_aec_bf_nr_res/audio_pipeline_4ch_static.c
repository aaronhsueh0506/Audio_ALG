/** Four-channel linear AEC + beamformer unit (core + SRP-PHAT + BF) followed by the post-beam RES
 * (lib/aec AecRes), NR, comfort noise and synthesis, static memory. Thin host reference over
 * audio_pipeline_4ch.h: query -> allocate -> init_ex -> process -> fuse -> RES / NR / min / CNG /
 * synthesis -> destroy -> release, with the real SRP-PHAT DOA and beamformer running in this
 * process. The unit and the post stages are initialised separately, each in its own pool. audio_pipeline_4ch_raw.c (the
 * recording-validation CLI) uses the heap create() path.
 * Build:  make -C pipelines/4ch_aec_bf_nr_res audio_pipeline_4ch_static
 * Usage:
 *   ./audio_pipeline_4ch_static [--sample-rate 16000|48000] [--fft-size 256|512|1024]
 *       [--uca-radius-m 0.035] [--fixed-doa-deg DEG]
 *   ./audio_pipeline_4ch_static --print-mem-size [same options] */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec_res.h"
#include "anr.h"
#include "audio_pipeline_4ch.h"
#include "audio_stft.h"
#include "mcab_sample_driver.h"
#include "se_anr_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            /* Per-channel scale and a small phase offset give SRP-PHAT/BF a non-degenerate
             * four-channel signal. */
            float ch_phase = phase - 0.05f * (float)ch;
            microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                (0.35f + 0.05f * (float)ch) * sinf(ch_phase);
        }
    }
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
    McabSpatialOptions options;
    AudioPipeline4ChConfig cfg;
    AudioPipeline4ChMemReq req;
    PostStages post;
    void *pool, *post_pool;
    AudioPipeline4Ch *p;
    AecRes *res;
    AnrDenoiser *nr;
    AcIstft *istft;
    float *total_gain, *extra_noise;
    Complex *spec;
    uint32_t rng;
    int n_freqs;
    AudioPipeline4ChBeam beam;
    float *mic_buf,*ref_buf,*out_buf;
    int hop,saw_finite_doa_raw,saw_bf_adaptive;
    AudioPipeline4ChFrameInfo info;
} StaticSample;
static int static_parse(void *user, int argc, char **argv) {
    StaticSample *s=user;
    if (mcab_parse_spatial_cli(argc,argv,0,&s->options) != 0) {
        fprintf(stderr,
            "Usage: %s [--sample-rate 16000|48000] "
            "[--fft-size 256|512|1024] [--uca-radius-m 0.035] "
            "[--fixed-doa-deg DEG] [--print-mem-size]\n",argv[0]);
        return 2;
    }
    return 0;
}
static int static_setup(void *user, McabHopView *view) {
    StaticSample *s=user;
    s->cfg =
        audio_pipeline_4ch_default_config(s->options.sample_rate);
    if (s->options.fft_size != 0) s->cfg.core.fft_size = s->options.fft_size;
    s->cfg.uca_radius_m = s->options.uca_radius_m;
    if (s->options.fixed_doa) {
        s->cfg.bf_fixed_mode = 1;
        s->cfg.bf_fixed_doa_rad = s->options.fixed_doa_rad;
    }

    if (post_plan(s->cfg.core.sample_rate,
                  mcab_resolved_fft(s->cfg.core.sample_rate, s->cfg.core.fft_size),
                  &s->post) != 0) {
        fprintf(stderr, "Error: invalid post-stage grid\n");
        return 1;
    }

    printf("4ch linear AEC + BF unit and post-beam chain (static memory)\n");
    printf("  Geometry: UCA radius=%.4f m   num_angles=%d\n",
           s->cfg.uca_radius_m, s->cfg.num_angles);

    {
        McabMemLine lines[4] = {
            {"RES:", s->post.res_bytes}, {"NR:", s->post.nr_bytes},
            {"Synthesis:", s->post.synth_bytes},
            {"Buffers:", 2 * s->post.vec_bytes + s->post.spec_bytes}};
        if (mcab_print_mem_budget(
                "4ch linear AEC + BF unit, RES, NR and synthesis", &s->cfg,
                audio_pipeline_4ch_get_mem_requirements, lines, 4) != 0)
            return 1;
    }

    /* Query -> allocate -> init_ex; the aligned allocation stands in for the board memory
     * manager. */
    if (audio_pipeline_4ch_get_mem_requirements(&s->cfg, &s->req) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        return 1;
    }

    s->pool = NULL;
    if (posix_memalign(&s->pool, s->req.alignment, (size_t)s->req.bytes) != 0)
        s->pool = NULL;
    if (!s->pool) {
        fprintf(stderr, "Error: aligned allocation failed (%llu bytes)\n",
                (unsigned long long)s->req.bytes);
        return 1;
    }
    if (((uintptr_t)s->pool) % s->req.alignment != 0) {
        fprintf(stderr, "Error: pool not %u-byte aligned (%p)\n",
                s->req.alignment, s->pool);
        return 1;
    }

    s->p =
        audio_pipeline_4ch_init_ex(s->pool, (size_t)s->req.bytes, &s->cfg, &s->req);
    if (!s->p) {
        fprintf(stderr, "Error: audio_pipeline_4ch_init_ex failed\n");
        return 1;
    }

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
        s->n_freqs = audio_pipeline_4ch_n_freqs(s->p);
        s->rng = SE_CNG_RNG_SEED;
    }

    s->hop = audio_pipeline_4ch_hop_size(s->p);
    printf("hop=%d fft=%d n_freqs=%d at %d Hz\n", s->hop,
           audio_pipeline_4ch_fft_size(s->p), audio_pipeline_4ch_n_freqs(s->p),
           s->options.sample_rate);
    if (s->options.print_mem_size) {
        return 2;
    }

    s->mic_buf =
        (float*)malloc((size_t)s->hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    s->ref_buf = (float*)malloc((size_t)s->hop * sizeof(float));
    s->out_buf = (float*)malloc((size_t)s->hop * sizeof(float));
    if (!s->mic_buf || !s->ref_buf || !s->out_buf) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        return 1;
    }

    view->hop=s->hop; view->microphones=s->mic_buf;
    view->reference=s->ref_buf; view->output=s->out_buf; view->check_finite=1;
    return 0;
}
static int static_read(void *user, McabHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    if (frame >= 48) return 0;
    fill_hop(s->mic_buf,s->ref_buf,s->hop,s->options.sample_rate,frame);
    return 1;
}
/* One hop after the fuse into s->out_buf: G_res, G_nr, P = E * min(G_nr, G_res), comfort noise on
 * the bins the RES cut, synthesis. 0, or -1 when the RES or the denoiser refuses the hop. */
static int static_chain(StaticSample *s, const SeSpecFrame *f) {
    const float *res = se_res_stage_gain(s->res, f);
    const float *nr = res ? se_anr_stage_gain(s->nr, f, s->extra_noise, 1) : NULL;
    if (!nr) return -1;
    sk_min_f32(s->total_gain, nr, res, s->n_freqs);   /* keeps a non-finite gain visible */
    sk_capply_gain_f32(s->spec, f->e, s->total_gain, s->n_freqs);
    if (f->cn)
        aec_cng_fill(&s->rng, s->spec, f->cn, res, nr, SE_CNG_NR_GAIN_FLOOR, s->n_freqs);
    (void)ac_istft_push(s->istft, s->spec, s->out_buf);
    return 0;
}

static void static_post_reset(StaticSample *s) {
    aec_res_reset(s->res);
    anr_reset(s->nr);
    ac_istft_reset(s->istft);
    s->rng = SE_CNG_RNG_SEED;
}

/* Unit, fuse, post stages. A refusal anywhere takes them all down: the unit resets itself on its
 * own refusals, the rest is reset here. */
static int static_hop(StaticSample *s, int vad_active) {
    SeSpecFrame frame;
    int status=audio_pipeline_4ch_process_with_activity(s->p,s->mic_buf,s->ref_buf,
            1,vad_active,NULL,&s->beam,&s->info);
    if (status != FOUR_AEC_NR_RES_OK) { static_post_reset(s); return status; }
    status=audio_pipeline_4ch_fuse(s->p,&frame);
    if (status == FOUR_AEC_NR_RES_OK &&
        (static_chain(s,&frame) != 0 || !se_hop_is_finite(s->out_buf,s->hop))) {
        audio_pipeline_4ch_reset(s->p);
        status = FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (status != FOUR_AEC_NR_RES_OK) static_post_reset(s);
    return status;
}
static int static_process(void *user, McabHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    if (static_hop(s,frame < 8 ? 1 : 0) != FOUR_AEC_NR_RES_OK) {
        fprintf(stderr,"Error: 4ch hop (unit, fuse, post-beam chain) failed\n");
        return 1;
    }
    return 0;
}
static void static_nonfinite(void *user) {
    (void)user; fprintf(stderr,"Error: non-finite output\n");
}
static int static_consume(void *user, McabHopView *view, int frame) {
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
static int static_finish(void *user, McabHopView *view, int frame) {
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
    printf("audio_pipeline_4ch_static: smoke PASS\n");
    return 0;
}
static void static_cleanup(void *user) {
    StaticSample *s=user;
    if (s->nr) anr_destroy(s->nr);
    free(s->post_pool);
    audio_pipeline_4ch_destroy(s->p); free(s->pool);
    free(s->mic_buf); free(s->ref_buf); free(s->out_buf);
}
int main(int argc, char **argv) {
    StaticSample sample={0};
    static const McabSampleOps ops={static_parse,1,static_setup,static_read,
        static_process,static_consume,static_nonfinite,static_finish,static_cleanup,NULL};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
