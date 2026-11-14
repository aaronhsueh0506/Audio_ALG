/** Host-only raw-float runner for recording validation, written as an example of the
 * composition: the four-microphone array instance of lib/aec (aec.h: four lanes, the shared
 * estimator and the residual gain stage in one pool) with the sample SRP-PHAT/BF helper
 * (array_sample_beam.h), followed by NR, comfort noise and synthesis, each library initialised
 * here and chained per hop in array_sample_stages_chain() (array_sample_unit.h).
 * Input mic: interleaved float32 [frames][4]; input ref: mono float32 [frames];
 * optional VAD: uint8 [complete_hops], nonzero = target speech; output: mono float32
 * [complete_hops * hop]. */

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
    ArraySampleSpatialOptions options;
    AecConfig cfg;
    AecSignalGrid grid;
    ArraySampleBeamParams bp;
    ArraySampleUnit unit;
    void *unit_mem;
    ArraySampleStages post;
    void *post_mem;
    ArraySampleBeamResult beam;
    ArraySampleBeamInfo info;
    FILE *mic_file, *ref_file, *output_file, *vad_file;
    float *microphones, *reference, *output;
    int hop, frames, adaptive_frames, doa_frames;
    unsigned char activity;
} RawSample;
static int raw_parse(void *user, int argc, char **argv) {
    RawSample *s = user;
    if (array_sample_parse_spatial_cli(argc, argv, 1, &s->options) != 0) {
        usage(argv[0]); return 2;
    }
    return 0;
}
static int raw_setup(void *user, ArraySampleHopView *view) {
    RawSample *s = user;
    array_sample_default_config(&s->cfg, s->options.sample_rate);
    s->bp = array_sample_beam_default_params();
    if (s->options.fft_size) s->cfg.fft_size = s->options.fft_size;
    s->bp.uca_radius_m = s->options.uca_radius_m;
    if (s->options.fixed_doa) {
        s->bp.bf_fixed_mode = 1;
        s->bp.bf_fixed_doa_rad = s->options.fixed_doa_rad;
    }
    if (aec_resolve_signal_grid(s->cfg.sample_rate, s->cfg.fft_size, &s->grid) != AEC_OK ||
        array_sample_stages_plan(&s->post, s->cfg.sample_rate, s->grid.fft_size) != 0) {
        fprintf(stderr, "invalid post-stage grid\n");
        return 1;
    }
    if (s->options.print_mem_size) {
        ArraySampleMemLine lines[3] = {
            {"NR:", s->post.nr_bytes}, {"Synthesis:", s->post.synth_bytes},
            {"Buffers:", s->post.vec_bytes + s->post.spec_bytes}};
        return array_sample_print_mem_budget(
                   "4ch array AEC + BF unit, NR and synthesis", &s->cfg, &s->bp,
                   array_sample_unit_memory, lines, 3) == 0
                   ? 2
                   : 1;
    }
    s->unit_mem = NULL;
    if (array_sample_unit_plan(&s->unit, &s->cfg, &s->bp) != 0 ||
        posix_memalign(&s->unit_mem, s->unit.req.alignment, s->unit.total) != 0)
        s->unit_mem = NULL;
    if (!s->unit_mem || array_sample_unit_init(&s->unit, s->unit_mem, &s->cfg, &s->bp) != 0) {
        fprintf(stderr, "failed to create 4ch DOA/BF pipeline\n");
        return 1;
    }
    if (posix_memalign(&s->post_mem, 16, s->post.total) != 0) s->post_mem = NULL;
    if (array_sample_stages_init(&s->post, s->post_mem) != 0) {
        fprintf(stderr, "failed to create the post-beam stages\n");
        return 1;
    }
    s->hop = s->grid.hop_size;
    s->microphones =
        (float*)malloc((size_t)s->hop * AEC_ARRAY_MICS *
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
static int raw_read(void *user, ArraySampleHopView *view, int frame) {
    RawSample *s=user; (void)view; (void)frame;
    size_t mic_count=fread(s->microphones,sizeof(float),
        (size_t)s->hop*AEC_ARRAY_MICS,s->mic_file);
    size_t ref_count=fread(s->reference,sizeof(float),(size_t)s->hop,s->ref_file);
    if (mic_count==0 && ref_count==0) return 0;
    if (mic_count!=(size_t)s->hop*AEC_ARRAY_MICS ||
        ref_count!=(size_t)s->hop) return 0; /* Ignore trailing incomplete hop. */
    if (s->vad_file && fread(&s->activity,1,1,s->vad_file)!=1) {
        fprintf(stderr,"VAD file ended before audio\n"); return -1;
    }
    return 1;
}
/* Fuse, then the chain. A refusal anywhere takes the unit and the post stages down together:
 * the unit resets itself on its own refusals, the rest is reset here. */
static int raw_post(RawSample *s) {
    AecResidualView frame;
    int status = array_sample_beam_fuse(s->unit.beam, &frame);
    if (status == AEC_OK &&
        (array_sample_stages_chain(&s->post, s->unit.aec, &frame, s->output) != 0 ||
         !skn_all_finite_f32(s->output, (size_t)s->hop))) {
        array_sample_beam_reset(s->unit.beam);
        status = AEC_ERR_DSP;
    }
    if (status != AEC_OK) array_sample_stages_reset(&s->post);
    return status;
}
static int raw_process(void *user, ArraySampleHopView *view, int frame) {
    RawSample *s=user; (void)view;
    int status;
    if (s->vad_file)
        status=array_sample_beam_process_with_activity(s->unit.beam,s->microphones,
            s->reference,s->activity!=0,s->activity!=0,NULL,&s->beam,&s->info);
    else status=array_sample_beam_process(s->unit.beam,s->microphones,s->reference,
                                          &s->beam,&s->info);
    if (status == AEC_OK) status=raw_post(s);
    else array_sample_stages_reset(&s->post);
    if (status != AEC_OK) {
        fprintf(stderr,"pipeline failed at frame %d: %d\n",frame,status); return 1;
    }
    return 0;
}
static int raw_consume(void *user, ArraySampleHopView *view, int frame) {
    RawSample *s=user; (void)view; (void)frame;
    if (fwrite(s->output, sizeof(float), (size_t)s->hop, s->output_file) !=
        (size_t)s->hop) {
        fprintf(stderr, "failed to write output frame\n");
        return 1;
    }
    s->adaptive_frames += s->info.bf_adaptive != 0;
    s->doa_frames += isfinite(s->info.doa_raw_rad);
    ++s->frames;
    return 0;
}
static int raw_finish(void *user, ArraySampleHopView *view, int processed) {
    RawSample *s=user; (void)view;
    char final_raw[32], final_smooth[32], final_used[32];
    int write_failed;
    /* The last buffered hops reach the file only here: a failed flush is an I/O failure too. */
    write_failed = ferror(s->output_file) != 0;
    write_failed |= fclose(s->output_file) != 0;
    s->output_file = NULL;
    if (ferror(s->mic_file) || ferror(s->ref_file) || write_failed) {
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
           /* SRP-PHAT and BF run on the array's own grid: frame_size and the doa_* / bf_* keys
            * repeat it. */
           s->grid.fft_size,
           s->grid.fft_size,
           s->grid.n_freqs,
           s->options.sample_rate,
           s->grid.sample_rate,
           s->grid.fft_size,
           s->grid.hop_size,
           s->grid.fft_size,
           s->grid.sample_rate,
           s->grid.fft_size,
           s->grid.hop_size,
           s->grid.fft_size,
           aec_array_matched_filter_count(s->unit.aec),
           aec_array_linear_aec_count(s->unit.aec),
           s->post.nr ? 1 : 0,
           s->cfg.residual == AEC_RESIDUAL_FULL ? 1 : 0,
           array_sample_beam_spatial_backend(),
           s->frames, s->doa_frames, s->adaptive_frames,
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
    if (s->post.nr) anr_destroy(s->post.nr);
    free(s->post_mem);
    array_sample_unit_destroy(&s->unit);
    free(s->unit_mem);
}
int main(int argc, char **argv) {
    RawSample sample={0};
    static const ArraySampleOps ops={raw_parse,raw_setup,raw_read,raw_process,
                                  raw_consume,NULL,raw_finish,raw_cleanup,NULL};
    return array_sample_main(argc,argv,&ops,&sample);
}
