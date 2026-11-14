/** Four-microphone array AEC instance (aec.h, n_mics = 4) with the sample SRP-PHAT/BF helper
 * (array_sample_beam.h) followed by NR, comfort noise and synthesis, static memory.
 * Host reference of the application side: aec_get_mem_requirements/aec_init_ex of the array, the
 * helper on top -> process -> fuse -> NR / gain / CNG / synthesis -> destroy -> release, with the
 * real SRP-PHAT DOA and beamformer running in this process. The unit and the post stages are
 * initialised separately, each in its own pool.
 * Build:  make -C pipelines/4ch 4ch_static
 * Usage:
 *   ./4ch_static [--sample-rate 16000|48000] [--fft-size 256|512|1024]
 *       [--uca-radius-m 0.035] [--fixed-doa-deg DEG]
 *   ./4ch_static --print-mem-size [same options]
 * Numbers are whole decimals. Exit: 0 done, 1 refused config or failed run, 2 command line. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "anr.h"
#include "array_sample_beam.h"
#include "array_sample_driver.h"
#include "array_sample_unit.h"
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
        for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
            /* Per-channel scale and a small phase offset give SRP-PHAT/BF a non-degenerate
             * four-channel signal. */
            float ch_phase = phase - 0.05f * (float)ch;
            microphones[i * AEC_ARRAY_MICS + ch] =
                (0.35f + 0.05f * (float)ch) * sinf(ch_phase);
        }
    }
}

typedef struct StaticSample {
    ArraySampleSpatialOptions options;
    AecConfig cfg;
    AecSignalGrid grid;
    ArraySampleBeamParams bp;
    ArraySampleUnit unit;
    ArraySampleStages post;
    void *pool, *post_pool;
    ArraySampleBeamResult beam;
    float *mic_buf,*ref_buf,*out_buf;
    int hop,saw_finite_doa_raw,saw_bf_adaptive;
    ArraySampleBeamInfo info;
} StaticSample;
static int static_parse(void *user, int argc, char **argv) {
    StaticSample *s=user;
    if (array_sample_parse_spatial_cli(argc,argv,0,&s->options) != 0) {
        fprintf(stderr,
            "Usage: %s [--sample-rate 16000|48000] "
            "[--fft-size 256|512|1024] [--uca-radius-m 0.035] "
            "[--fixed-doa-deg DEG] [--print-mem-size]\n",argv[0]);
        return 2;
    }
    return 0;
}
static int static_setup(void *user, ArraySampleHopView *view) {
    StaticSample *s=user;
    array_sample_default_config(&s->cfg, s->options.sample_rate);
    s->bp = array_sample_beam_default_params();
    if (s->options.fft_size != 0) s->cfg.fft_size = s->options.fft_size;
    s->bp.uca_radius_m = s->options.uca_radius_m;
    if (s->options.fixed_doa) {
        s->bp.bf_fixed_mode = 1;
        s->bp.bf_fixed_doa_rad = s->options.fixed_doa_rad;
    }

    if (aec_resolve_signal_grid(s->cfg.sample_rate, s->cfg.fft_size, &s->grid) != AEC_OK ||
        array_sample_stages_plan(&s->post, s->cfg.sample_rate, s->grid.fft_size) != 0) {
        fprintf(stderr, "Error: invalid post-stage grid\n");
        return 1;
    }

    printf("4ch array AEC + BF unit and post-beam chain (static memory)\n");
    printf("  Geometry: UCA radius=%.4f m   num_angles=%d\n",
           s->bp.uca_radius_m, s->bp.num_angles);

    {
        ArraySampleMemLine lines[3] = {
            {"NR:", s->post.nr_bytes}, {"Synthesis:", s->post.synth_bytes},
            {"Buffers:", s->post.vec_bytes + s->post.spec_bytes}};
        if (array_sample_print_mem_budget(
                "4ch array AEC + BF unit, NR and synthesis", &s->cfg, &s->bp,
                array_sample_unit_memory, lines, 3) != 0)
            return 1;
    }

    /* Query -> allocate -> init; the aligned allocation stands in for the board memory
     * manager. */
    if (array_sample_unit_plan(&s->unit, &s->cfg, &s->bp) != 0) {
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

    if (array_sample_unit_init(&s->unit, s->pool, &s->cfg, &s->bp) != 0) {
        fprintf(stderr, "Error: array init failed\n");
        return 1;
    }
    if (posix_memalign(&s->post_pool, 16, s->post.total) != 0) s->post_pool = NULL;
    if (array_sample_stages_init(&s->post, s->post_pool) != 0) {
        fprintf(stderr, "Error: post-beam stage init failed\n");
        return 1;
    }

    s->hop = s->grid.hop_size;
    printf("hop=%d fft=%d n_freqs=%d at %d Hz\n", s->hop,
           s->grid.fft_size, s->grid.n_freqs, s->options.sample_rate);
    if (s->options.print_mem_size) {
        return 2;
    }

    s->mic_buf =
        (float*)malloc((size_t)s->hop * AEC_ARRAY_MICS * sizeof(float));
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
static int static_read(void *user, ArraySampleHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    if (frame >= 48) return 0;
    fill_hop(s->mic_buf,s->ref_buf,s->hop,s->options.sample_rate,frame);
    return 1;
}
/* Unit, fuse, post stages. A refusal anywhere takes them all down: the unit resets itself on its
 * own refusals, the rest is reset here. */
static int static_hop(StaticSample *s, int vad_active) {
    AecResidualView frame;
    int status=array_sample_beam_process_with_activity(s->unit.beam,s->mic_buf,s->ref_buf,
            1,vad_active,NULL,&s->beam,&s->info);
    if (status != AEC_OK) { array_sample_stages_reset(&s->post); return status; }
    status=array_sample_beam_fuse(s->unit.beam,&frame);
    if (status == AEC_OK &&
        (array_sample_stages_chain(&s->post,s->unit.aec,&frame,s->out_buf) != 0 ||
         !skn_all_finite_f32(s->out_buf, (size_t)s->hop))) {
        array_sample_beam_reset(s->unit.beam);
        status = AEC_ERR_DSP;
    }
    if (status != AEC_OK) array_sample_stages_reset(&s->post);
    return status;
}
static int static_process(void *user, ArraySampleHopView *view, int frame) {
    StaticSample *s=user; (void)view;
    if (static_hop(s,frame < 8 ? 1 : 0) != AEC_OK) {
        fprintf(stderr,"Error: 4ch hop (unit, fuse, post-beam chain) failed\n");
        return 1;
    }
    return 0;
}
static int static_consume(void *user, ArraySampleHopView *view, int frame) {
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
static int static_finish(void *user, ArraySampleHopView *view, int frame) {
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
    printf("4ch_static: smoke PASS\n");
    return 0;
}
static void static_cleanup(void *user) {
    StaticSample *s=user;
    if (s->post.nr) anr_destroy(s->post.nr);
    free(s->post_pool);
    array_sample_unit_destroy(&s->unit); free(s->pool);
    free(s->mic_buf); free(s->ref_buf); free(s->out_buf);
}
int main(int argc, char **argv) {
    StaticSample sample={0};
    static const ArraySampleOps ops={static_parse,static_setup,static_read,
        static_process,static_consume,NULL,static_finish,static_cleanup,NULL};
    return array_sample_main(argc,argv,&ops,&sample);
}
