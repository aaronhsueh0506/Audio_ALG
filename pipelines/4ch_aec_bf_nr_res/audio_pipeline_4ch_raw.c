/** Host-only raw-float runner for recording validation.
 * Input mic: interleaved float32 [frames][4]; input ref: mono float32 [frames];
 * optional VAD: uint8 [complete_hops], nonzero = target speech; output: mono float32
 * [complete_hops * hop]. */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_pipeline_4ch.h"
#include "mcab_sample_driver.h"

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

typedef struct RawSample {
    McabSpatialOptions options;
    AudioPipeline4ChConfig cfg;
    AudioPipeline4Ch *p;
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
    if (s->options.print_mem_size) {
        return mcab_print_mem_budget(
                   "4ch spatial pipeline: core + SRP-PHAT + BF", &s->cfg, audio_pipeline_4ch_get_mem_requirements) == 0
                   ? 2
                   : 1;
    }
    s->p = audio_pipeline_4ch_create(&s->cfg);
    if (!s->p) {
        fprintf(stderr, "failed to create 4ch DOA/BF pipeline\n");
        return 1;
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
static int raw_process(void *user, McabHopView *view, int frame) {
    RawSample *s=user; (void)view;
    int status;
    if (s->vad_file)
        status=audio_pipeline_4ch_process_with_activity(s->p,s->microphones,
            s->reference,s->activity!=0,s->activity!=0,NULL,s->output,&s->info);
    else status=audio_pipeline_4ch_process(s->p,s->microphones,s->reference,
                                          s->output,&s->info);
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
           audio_pipeline_4ch_nr_count(s->p),
           audio_pipeline_4ch_post_res_count(s->p),
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
    audio_pipeline_4ch_destroy(s->p);
}
int main(int argc, char **argv) {
    RawSample sample={0};
    static const McabSampleOps ops={raw_parse,2,raw_setup,raw_read,raw_process,
                                  raw_consume,NULL,raw_finish,raw_cleanup,NULL};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
