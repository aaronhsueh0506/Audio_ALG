/** Host-only raw-float runner for recording validation, written as an example of the
 * composition: the linear AEC + BF unit (audio_pipeline_4ch.h) followed by the post-beam RES
 * (lib/aec AecRes), NR, comfort noise and synthesis, each library initialised here and chained
 * per hop in raw_chain().
 * Input mic: interleaved float32 [frames][4]; input ref: mono float32 [frames];
 * optional VAD: uint8 [complete_hops], nonzero = target speech; output: mono float32
 * [complete_hops * hop]. */

#include <errno.h>
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

static void usage(const char* program) {
    fprintf(stderr,
            "Usage: %s --mic-raw MIC.f32 --ref-raw REF.f32 "
            "--output-raw OUT.f32 --sample-rate 16000|48000 "
            "[--fft-size 256|512|1024] "
            "[--vad-u8 VAD.u8] [--uca-radius-m 0.035] "
            "[--fixed-doa-deg DEG]\n"
            "       %s --print-mem-size --sample-rate 16000|48000 "
            "[--fft-size 256|512|1024]\n",
            program, program);
}

static void json_float(char* output, size_t capacity, float value) {
    if (!output || capacity == 0) return;
    if (isfinite(value)) {
        (void)snprintf(output, capacity, "%.9g", value);
    } else {
        (void)snprintf(output, capacity, "null");
    }
}

/* The stages after the fuse, carved in this order from one aligned block. */
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

typedef struct RawSample {
    McabSpatialOptions options;
    AudioPipeline4ChConfig cfg;
    PostStages post;
    AudioPipeline4Ch *p;
    void *post_mem;
    AecRes *res;
    AnrDenoiser *nr;
    AcIstft *istft;
    float *total_gain, *extra_noise;
    Complex *spec;
    uint32_t rng;
    int n_freqs;
    AudioPipeline4ChBeam beam;
    AudioPipeline4ChFrameInfo info;
    FILE *mic_file, *ref_file, *output_file, *vad_file;
    float *microphones, *reference, *output;
    int hop, adaptive_frames, doa_frames, doa_analysis_frames;
    unsigned char activity;
} RawSample;
static int raw_parse(void *user, int argc, char **argv) {
    RawSample *s = user;
    if (mcab_parse_spatial_cli(argc, argv, 1, &s->options) != 0) {
        usage(argv[0]); return 2;
    }
    return 0;
}
static int raw_setup(void *user, McabHopView *view) {
    RawSample *s = user;
    s->cfg = audio_pipeline_4ch_default_config(s->options.sample_rate);
    s->cfg.core.fft_size = s->options.fft_size;
    s->cfg.uca_radius_m = s->options.uca_radius_m;
    if (s->options.fixed_doa) {
        s->cfg.bf_fixed_mode = 1;
        s->cfg.bf_fixed_doa_rad = s->options.fixed_doa_rad;
    }
    if (post_plan(s->cfg.core.sample_rate,
                  mcab_resolved_fft(s->cfg.core.sample_rate, s->cfg.core.fft_size),
                  &s->post) != 0) {
        fprintf(stderr, "invalid post-stage grid\n");
        return 1;
    }
    if (s->options.print_mem_size) {
        McabMemLine lines[4] = {
            {"RES:", s->post.res_bytes}, {"NR:", s->post.nr_bytes},
            {"Synthesis:", s->post.synth_bytes},
            {"Buffers:", 2 * s->post.vec_bytes + s->post.spec_bytes}};
        return mcab_print_mem_budget(
                   "4ch linear AEC + BF unit, RES, NR and synthesis", &s->cfg,
                   audio_pipeline_4ch_get_mem_requirements, lines, 4) == 0
                   ? 2
                   : 1;
    }
    s->p = audio_pipeline_4ch_create(&s->cfg);
    if (!s->p) {
        fprintf(stderr, "failed to create 4ch DOA/BF pipeline\n");
        return 1;
    }
    {
        unsigned char *cur;
        if (posix_memalign(&s->post_mem, 16, s->post.total) != 0) s->post_mem = NULL;
        cur = s->post_mem;
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
            fprintf(stderr, "failed to create the post-beam stages\n");
            return 1;
        }
        s->n_freqs = audio_pipeline_4ch_n_freqs(s->p);
        s->rng = SE_CNG_RNG_SEED;
    }
    s->hop = audio_pipeline_4ch_hop_size(s->p);
    s->microphones =
        (float*)malloc((size_t)s->hop * FOUR_AEC_NR_RES_CHANNELS *
                       sizeof(float));
    s->reference = (float*)malloc((size_t)s->hop * sizeof(float));
    s->output = (float*)malloc((size_t)s->hop * sizeof(float));
    if (!s->microphones || !s->reference || !s->output) {
        fprintf(stderr, "failed to allocate hop buffers\n");
        return 1;
    }
    s->mic_file = fopen(s->options.mic_path, "rb");
    s->ref_file = fopen(s->options.ref_path, "rb");
    s->output_file = fopen(s->options.output_path, "wb");
    if (s->options.vad_path) s->vad_file = fopen(s->options.vad_path, "rb");
    if (!s->mic_file || !s->ref_file || !s->output_file ||
        (s->options.vad_path && !s->vad_file)) {
        fprintf(stderr, "failed to open raw input/output\n");
        return 1;
    }

    view->hop=s->hop; view->microphones=s->microphones;
    view->reference=s->reference; view->output=s->output;
    return 0;
}
static int raw_read(void *user, McabHopView *view, int frame) {
    RawSample *s=user; (void)view; (void)frame;
    size_t mic_count=fread(s->microphones,sizeof(float),
        (size_t)s->hop*FOUR_AEC_NR_RES_CHANNELS,s->mic_file);
    size_t ref_count=fread(s->reference,sizeof(float),(size_t)s->hop,s->ref_file);
    if (mic_count==0 && ref_count==0) return 0;
    if (mic_count!=(size_t)s->hop*FOUR_AEC_NR_RES_CHANNELS ||
        ref_count!=(size_t)s->hop) return 0; /* Ignore trailing incomplete hop. */
    if (s->vad_file && fread(&s->activity,1,1,s->vad_file)!=1) {
        fprintf(stderr,"VAD file ended before audio\n"); return -1;
    }
    return 1;
}
/* One hop after the fuse into s->output: G_res, G_nr, P = E * min(G_nr, G_res), comfort noise on
 * the bins the RES cut, synthesis. 0, or -1 when the RES or the denoiser refuses the hop. */
static int raw_chain(RawSample *s, const SeSpecFrame *f) {
    const float *res = se_res_stage_gain(s->res, f);
    const float *nr = res ? se_anr_stage_gain(s->nr, f, s->extra_noise, 1) : NULL;
    if (!nr) return -1;
    sk_min_f32(s->total_gain, nr, res, s->n_freqs);   /* keeps a non-finite gain visible */
    sk_capply_gain_f32(s->spec, f->e, s->total_gain, s->n_freqs);
    if (f->cn)
        aec_cng_fill(&s->rng, s->spec, f->cn, res, nr, SE_CNG_NR_GAIN_FLOOR, s->n_freqs);
    (void)ac_istft_push(s->istft, s->spec, s->output);
    return 0;
}

static void raw_post_reset(RawSample *s) {
    aec_res_reset(s->res);
    anr_reset(s->nr);
    ac_istft_reset(s->istft);
    s->rng = SE_CNG_RNG_SEED;
}

/* Fuse, then the chain. A refusal anywhere takes the unit and the post stages down together:
 * the unit resets itself on its own refusals, the rest is reset here. */
static int raw_post(RawSample *s) {
    SeSpecFrame frame;
    int status = audio_pipeline_4ch_fuse(s->p, &frame);
    if (status == FOUR_AEC_NR_RES_OK &&
        (raw_chain(s, &frame) != 0 || !se_hop_is_finite(s->output, s->hop))) {
        audio_pipeline_4ch_reset(s->p);
        status = FOUR_AEC_NR_RES_DSP_ERROR;
    }
    if (status != FOUR_AEC_NR_RES_OK) raw_post_reset(s);
    return status;
}
static int raw_process(void *user, McabHopView *view, int frame) {
    RawSample *s=user; (void)view;
    int status;
    if (s->vad_file)
        status=audio_pipeline_4ch_process_with_activity(s->p,s->microphones,
            s->reference,s->activity!=0,s->activity!=0,NULL,&s->beam,&s->info);
    else status=audio_pipeline_4ch_process(s->p,s->microphones,s->reference,
                                          &s->beam,&s->info);
    if (status == FOUR_AEC_NR_RES_OK) status=raw_post(s);
    else raw_post_reset(s);
    if (status != FOUR_AEC_NR_RES_OK) {
        fprintf(stderr,"pipeline failed at frame %d: %d\n",frame,status); return 1;
    }
    return 0;
}
static int raw_consume(void *user, McabHopView *view, int frame) {
    RawSample *s=user; (void)view; (void)frame;
    if (fwrite(s->output, sizeof(float), (size_t)s->hop, s->output_file) !=
        (size_t)s->hop) {
        fprintf(stderr, "failed to write output frame\n");
        return 1;
    }
    s->adaptive_frames += s->info.bf_adaptive != 0;
    s->doa_frames += isfinite(s->info.doa_raw_rad);
    s->doa_analysis_frames += s->info.doa_analysis_frames;
    return 0;
}
static int raw_finish(void *user, McabHopView *view, int processed) {
    RawSample *s=user; (void)view;
    char final_raw[32], final_smooth[32], final_used[32];
    if (ferror(s->mic_file) || ferror(s->ref_file) || ferror(s->output_file)) {
        fprintf(stderr, "raw file I/O failure\n");
        return 1;
    }
    json_float(
        final_raw, sizeof(final_raw),
        processed > 0 ? s->info.doa_raw_rad : NAN);
    json_float(
        final_smooth, sizeof(final_smooth),
        processed > 0 ? s->info.doa_smooth_rad : NAN);
    json_float(
        final_used, sizeof(final_used),
        processed > 0 ? s->info.doa_used_rad : NAN);

    printf("{\"frames\":%d,\"hop\":%d,\"frame_size\":%d,"
           "\"fft_size\":%d,\"n_freqs\":%d,\"sample_rate\":%d,"
           "\"doa_sample_rate\":%d,\"doa_frame_size\":%d,"
           "\"doa_hop_size\":%d,\"doa_fft_size\":%d,"
           "\"bf_sample_rate\":%d,\"bf_frame_size\":%d,"
           "\"bf_hop_size\":%d,\"bf_fft_size\":%d,"
           "\"matched_filters\":%d,\"linear_aecs\":%d,\"nr\":%d,"
           "\"post_res\":%d,\"spatial_backend\":\"%s\","
           "\"doa_analysis_frames\":%d,\"doa_update_frames\":%d,"
           "\"bf_adaptive_frames\":%d,"
           "\"final_delay_samples\":%d,\"final_delay_solid\":%d,"
           "\"final_doa_raw_rad\":%s,\"final_doa_smooth_rad\":%s,"
           "\"final_doa_used_rad\":%s}\n",
           processed, s->hop,
           audio_pipeline_4ch_frame_size(s->p),
           audio_pipeline_4ch_fft_size(s->p),
           audio_pipeline_4ch_n_freqs(s->p),
           s->options.sample_rate,
           audio_pipeline_4ch_doa_sample_rate(s->p),
           audio_pipeline_4ch_doa_frame_size(s->p),
           audio_pipeline_4ch_doa_hop_size(s->p),
           audio_pipeline_4ch_doa_fft_size(s->p),
           audio_pipeline_4ch_bf_sample_rate(s->p),
           audio_pipeline_4ch_bf_frame_size(s->p),
           audio_pipeline_4ch_bf_hop_size(s->p),
           audio_pipeline_4ch_bf_fft_size(s->p),
           audio_pipeline_4ch_matched_filter_count(s->p),
           audio_pipeline_4ch_linear_aec_count(s->p),
           s->nr ? 1 : 0,
           s->res ? 1 : 0,
           audio_pipeline_4ch_spatial_backend(),
           s->doa_analysis_frames, s->doa_frames, s->adaptive_frames,
           processed > 0 ? s->info.delay.delay_samples : 0,
           processed > 0 ? s->info.delay.solid : 0,
           final_raw, final_smooth, final_used);
    return processed > 0 ? 0 : 1;
}
static void raw_cleanup(void *user) {
    RawSample *s=user;
    if (s->vad_file) fclose(s->vad_file);
    if (s->output_file) fclose(s->output_file);
    if (s->ref_file) fclose(s->ref_file);
    if (s->mic_file) fclose(s->mic_file);
    free(s->output);
    free(s->reference);
    free(s->microphones);
    if (s->nr) anr_destroy(s->nr);
    free(s->post_mem);
    audio_pipeline_4ch_destroy(s->p);
}
int main(int argc, char **argv) {
    RawSample sample={0};
    static const McabSampleOps ops={raw_parse,2,raw_setup,raw_read,raw_process,
                                  raw_consume,NULL,raw_finish,raw_cleanup,NULL};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
