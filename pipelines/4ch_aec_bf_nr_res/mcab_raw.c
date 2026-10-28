/** Host-only raw-float runner for recording validation, written as an example of the
 * composition: the four lanes, the shared estimator and the mcab flow (mcab.h) with the sample
 * SRP-PHAT/BF helper (mcab_sample_beam.h), followed by the post-beam RES (lib/aec AecRes), NR,
 * comfort noise and synthesis, each library initialised here and chained per hop in
 * raw_chain().
 * Input mic: interleaved float32 [frames][4]; input ref: mono float32 [frames];
 * optional VAD: uint8 [complete_hops], nonzero = target speech; output: mono float32
 * [complete_hops * hop]. */

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

/* The stages after the fuse, carved in this order from one aligned block. */
typedef struct PostStages {
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t res_bytes, nr_bytes, synth_bytes, vec_bytes, spec_bytes, total;
} PostStages;

static int post_plan(int sample_rate, int fft_size, PostStages *ps) {
    int n_freqs = fft_size / 2 + 1;
    aec_config_from_level(&ps->res_cfg, AEC_LEVEL_BALANCED, sample_rate);
    ps->res_cfg.fft_size = fft_size;
    ps->res_bytes = aec_res_get_mem_size(&ps->res_cfg);
    ps->nr_cfg = anr_config_for_level_grid(sample_rate, fft_size, ANR_LEVEL_BALANCED);
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
    McabSampleSpatialOptions options;
    McabConfig cfg;
    McabSampleBeamParams bp;
    Unit unit;
    void *unit_mem;
    PostStages post;
    McabSampleBeam *p;
    void *post_mem;
    AecRes *res;
    AnrDenoiser *nr;
    AcIstft *istft;
    float *total_gain, *extra_noise;
    Complex *spec;
    uint32_t rng;
    int n_freqs;
    McabSampleBeamResult beam;
    McabSampleBeamInfo info;
    FILE *mic_file, *ref_file, *output_file, *vad_file;
    float *microphones, *reference, *output;
    int hop, adaptive_frames, doa_frames, doa_analysis_frames;
    unsigned char activity;
} RawSample;
static int raw_parse(void *user, int argc, char **argv) {
    RawSample *s = user;
    if (mcab_sample_parse_spatial_cli(argc, argv, 1, &s->options) != 0) {
        usage(argv[0]); return 2;
    }
    return 0;
}
static int raw_setup(void *user, McabSampleHopView *view) {
    RawSample *s = user;
    s->cfg = mcab_default_config(s->options.sample_rate);
    s->bp = mcab_sample_beam_default_params();
    s->cfg.fft_size = s->options.fft_size;
    s->bp.uca_radius_m = s->options.uca_radius_m;
    if (s->options.fixed_doa) {
        s->bp.bf_fixed_mode = 1;
        s->bp.bf_fixed_doa_rad = s->options.fixed_doa_rad;
    }
    if (post_plan(s->cfg.sample_rate,
                  mcab_sample_resolved_fft(s->cfg.sample_rate, s->cfg.fft_size),
                  &s->post) != 0) {
        fprintf(stderr, "invalid post-stage grid\n");
        return 1;
    }
    if (s->options.print_mem_size) {
        McabSampleMemLine lines[4] = {
            {"RES:", s->post.res_bytes}, {"NR:", s->post.nr_bytes},
            {"Synthesis:", s->post.synth_bytes},
            {"Buffers:", 2 * s->post.vec_bytes + s->post.spec_bytes}};
        return mcab_sample_print_mem_budget(
                   "4ch linear AEC + BF unit, RES, NR and synthesis", &s->cfg, &s->bp,
                   mcab_sample_unit_memory, lines, 4) == 0
                   ? 2
                   : 1;
    }
    s->unit_mem = NULL;
    if (unit_plan(&s->unit, &s->cfg, &s->bp) != 0 ||
        posix_memalign(&s->unit_mem, s->unit.req.alignment, s->unit.total) != 0)
        s->unit_mem = NULL;
    s->p = s->unit_mem && unit_init(&s->unit, s->unit_mem, &s->cfg, &s->bp) == 0
        ? s->unit.beam : NULL;
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
        s->n_freqs = mcab_n_freqs(s->unit.flow);
        s->rng = MCAB_SAMPLE_CNG_RNG_SEED;
    }
    s->hop = mcab_hop_size(s->unit.flow);
    s->microphones =
        (float*)malloc((size_t)s->hop * MCAB_CHANNELS *
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
static int raw_read(void *user, McabSampleHopView *view, int frame) {
    RawSample *s=user; (void)view; (void)frame;
    size_t mic_count=fread(s->microphones,sizeof(float),
        (size_t)s->hop*MCAB_CHANNELS,s->mic_file);
    size_t ref_count=fread(s->reference,sizeof(float),(size_t)s->hop,s->ref_file);
    if (mic_count==0 && ref_count==0) return 0;
    if (mic_count!=(size_t)s->hop*MCAB_CHANNELS ||
        ref_count!=(size_t)s->hop) return 0; /* Ignore trailing incomplete hop. */
    if (s->vad_file && fread(&s->activity,1,1,s->vad_file)!=1) {
        fprintf(stderr,"VAD file ended before audio\n"); return -1;
    }
    return 1;
}
/* One hop after the fuse into s->output: G_res, G_nr, P = E * min(G_nr, G_res), comfort noise on
 * the bins the RES cut, synthesis. 0, or -1 when the RES or the denoiser refuses the hop. */
static int raw_chain(RawSample *s, const McabFrame *f) {
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
    (void)ac_istft_push(s->istft, s->spec, s->output);
    return 0;
}

static void raw_post_reset(RawSample *s) {
    aec_res_reset(s->res);
    anr_reset(s->nr);
    ac_istft_reset(s->istft);
    s->rng = MCAB_SAMPLE_CNG_RNG_SEED;
}

/* Fuse, then the chain. A refusal anywhere takes the unit and the post stages down together:
 * the unit resets itself on its own refusals, the rest is reset here. */
static int raw_post(RawSample *s) {
    McabFrame frame;
    int status = mcab_sample_beam_fuse(s->p, &frame);
    if (status == MCAB_OK &&
        (raw_chain(s, &frame) != 0 || !skn_all_finite_f32(s->output, (size_t)s->hop))) {
        mcab_sample_beam_reset(s->p);
        status = MCAB_DSP_ERROR;
    }
    if (status != MCAB_OK) raw_post_reset(s);
    return status;
}
static int raw_process(void *user, McabSampleHopView *view, int frame) {
    RawSample *s=user; (void)view;
    int status;
    if (s->vad_file)
        status=mcab_sample_beam_process_with_activity(s->p,s->microphones,
            s->reference,s->activity!=0,s->activity!=0,NULL,&s->beam,&s->info);
    else status=mcab_sample_beam_process(s->p,s->microphones,s->reference,
                                          &s->beam,&s->info);
    if (status == MCAB_OK) status=raw_post(s);
    else raw_post_reset(s);
    if (status != MCAB_OK) {
        fprintf(stderr,"pipeline failed at frame %d: %d\n",frame,status); return 1;
    }
    return 0;
}
static int raw_consume(void *user, McabSampleHopView *view, int frame) {
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
static int raw_finish(void *user, McabSampleHopView *view, int processed) {
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
           /* SRP-PHAT and BF run on the flow's own grid: frame_size and the doa_* / bf_* keys
            * repeat it. */
           mcab_fft_size(s->unit.flow),
           mcab_fft_size(s->unit.flow),
           mcab_n_freqs(s->unit.flow),
           s->options.sample_rate,
           mcab_sample_rate(s->unit.flow),
           mcab_fft_size(s->unit.flow),
           mcab_hop_size(s->unit.flow),
           mcab_fft_size(s->unit.flow),
           mcab_sample_rate(s->unit.flow),
           mcab_fft_size(s->unit.flow),
           mcab_hop_size(s->unit.flow),
           mcab_fft_size(s->unit.flow),
           mcab_matched_filter_count(s->unit.flow),
           mcab_linear_aec_count(s->unit.flow),
           s->nr ? 1 : 0,
           s->res ? 1 : 0,
           mcab_sample_beam_spatial_backend(),
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
    unit_destroy(&s->unit);
    free(s->unit_mem);
}
int main(int argc, char **argv) {
    RawSample sample={0};
    static const McabSampleOps ops={raw_parse,raw_setup,raw_read,raw_process,
                                  raw_consume,NULL,raw_finish,raw_cleanup,NULL};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
