/* 4ch_aec_bf_dfn_res board skeleton: four-channel linear AEC -> beamformer (bf) -> RES ->
 * DeepFilterNet2 composed from the library calls. Example code: the linear AEC + BF unit
 * (audio_pipeline_4ch.h), the post-beam RES (lib/aec AecRes) initialised here, P = E * G_res
 * (+ comfort noise) formed here, and samples/dfn_hop.h for the DFN2 hop. Per hop:
 *
 *     audio_pipeline_4ch_process(unit, mics, far, &beam, &info);   // delay, lanes, SRP-PHAT, BF
 *     audio_pipeline_4ch_fuse(unit, &frame);                        // E, R2, CN
 *     G_res = se_res_stage_gain(res, &frame);                       // the RES over that frame
 *     P = E * G_res (+ comfort noise);                              // before the DFN hop begins
 *     dfn_hop_process(&hop, frame.e, P, out);                       // or begin / nn_begin / ...
 *
 * 48 kHz hosts drive dfn_res_stage.h and synthesise through an AcIstft; 16 kHz hosts run the same
 * stage on its 48 kHz grid inside dfn_rate_bridge.h. The first output hops are zero: two model hops
 * at 48 kHz, the bridge's added delay otherwise (printed below). The stage does not reset on a delay
 * change of the shared aligner.
 *
 * The program builds the pool from the command-line delay profile and ERB matrices, installs a
 * fail-open model callback and runs a few silent hops. Replace run_accelerator() with the board
 * runtime call. */
#include "aec_res.h"
#include "audio_pipeline_4ch.h"
#include "samples/dfn_hop.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "dfn_example_args.h"
#include "mcab_sample_driver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];

/* Replace with the board runtime call (see mono_aec_dfn_res). */
static int run_accelerator(void *user, const DFN2PrepostInputs *inputs,
                           DFN2PrepostOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* BOARD: invoke the stateless DFN2 graph. */
}

/* Host grids accepted at parse time (0-terminated; no 8 kHz). */
static const int HOST_RATES[] = { 16000, 48000, 0 };

static const char *name="4ch_aec_bf_dfn_res";
static float microphones[DFN2_HOP_LEN*FOUR_AEC_NR_RES_CHANNELS];
static float far[DFN2_HOP_LEN],output[DFN2_HOP_LEN];
typedef struct DfnSample {
    DfnExampleArgs args;
    McabDelayOptions delay;
    AudioPipeline4ChConfig unit_cfg;
    AudioPipeline4ChMemReq unit_req;
    FourAecNrResMemBreakdown grid;
    DfnResStageConfig stage_cfg;
    DfnRateBridgeConfig bridge_cfg;
    DFN2ModelIoDescriptor descriptor;
    AecConfig res_cfg;
    int use_bridge;                   /* host below 48 kHz */
    size_t vec_bytes, spec_bytes, res_bytes, stage_bytes, istft_bytes, total;
    void *pool;
    AudioPipeline4Ch *unit;
    AecRes *res;
    float *unity, *total_gain;        /* the RES gain path: min(1, G_res) */
    Complex *post;                    /* P */
    uint32_t rng;
    int n_freqs;
    DfnResStage *stage;
    DfnRateBridge *bridge;
    AcIstft *istft;
    DfnHop hop;
    const char *program;
    char rates_text[64];
} DfnSample;
static int dfn_unknown(void *user, const char *arg) {
    DfnSample *s=user;
    fprintf(stderr,"%s: unknown argument '%s'\n",name,arg);
    dfn_example_usage(s->program,s->rates_text); return 2;
}
static int dfn_set(void *user, const char *arg, const char *value) {
    DfnSample *s=user;
    if (!strcmp(arg,"-h") || !strcmp(arg,"--help")) {
        dfn_example_usage(s->program,s->rates_text); return 1;
    }
    if (!strcmp(arg,"--sample-rate")) {
        if (dfn_example_parse_int(value,&s->args.sample_rate) != 0 ||
            !dfn_example_rate_known(HOST_RATES,s->args.sample_rate)) {
            fprintf(stderr,"%s: --sample-rate '%s' is not a host grid of "
                "this skeleton (accepted: %s)\n",name,value,s->rates_text); return 2;
        }
    } else if (!strcmp(arg,"--erb-fwd")) s->args.erb_fwd_path=value;
    else if (!strcmp(arg,"--erb-inv")) s->args.erb_inv_path=value;
    else if (!strcmp(arg,"--atten-lim")) {
        if (dfn_example_parse_float(value,&s->args.atten_lim_db) != 0) {
            fprintf(stderr,"%s: --atten-lim '%s' is not a finite dB value\n",name,value);
            return 2;
        }
    } else if (!strcmp(arg,"--cng")) s->args.enable_cng=1;
    else return mcab_set_delay_option(&s->delay,arg,value,1,name);
    return 0;
}
static int dfn_parse(void *user, int argc, char **argv) {
    DfnSample *s=user;
    static const McabCliOption options[]={
        {"-h",0},{"--help",0},{"--sample-rate",1},{"--delay-mode",1},
        {"--delay-num-filters",1},{"--fixed-delay",1},{"--erb-fwd",1},
        {"--erb-inv",1},{"--atten-lim",1},{"--cng",0}
    };
    s->program=argv[0];
    dfn_example_rates_text(HOST_RATES,s->rates_text,sizeof(s->rates_text));
    /* Seed the DFN defaults without consuming user arguments. */
    if (dfn_example_parse_args(1,argv,name,HOST_RATES,&s->args) != 0) return 2;
    s->delay.mode=s->args.delay_mode;
    s->delay.num_filters=s->args.delay_num_filters;
    s->delay.fixed_samples=s->args.fixed_delay_samples;
    int rc=mcab_scan_cli(argc,argv,options,sizeof(options)/sizeof(options[0]),
                          s,dfn_set,dfn_unknown);
    if (rc != 0) return rc;
    if ((s->args.erb_fwd_path==NULL)!=(s->args.erb_inv_path==NULL)) {
        fprintf(stderr,"%s: --erb-fwd and --erb-inv go together\n",name); return 2;
    }
    rc=mcab_validate_delay_options(&s->delay,name);
    s->args.delay_mode=s->delay.mode;
    s->args.delay_num_filters=s->delay.num_filters;
    s->args.fixed_delay_samples=s->delay.fixed_samples;
    return rc;
}

/* Everything the pool holds, sized once. 0, or -1 for a configuration no stage accepts. */
static int dfn_plan(DfnSample *s) {
    DfnResStageMemReq stage_req;
    DfnRateBridgeMemReq bridge_req;
    AcStftConfig scfg;
    if (audio_pipeline_4ch_get_mem_requirements(&s->unit_cfg,&s->unit_req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(&s->unit_cfg.core,&s->grid) != 0 ||
        !isfinite(s->args.atten_lim_db))
        return -1;
    /* The RES: the balanced preset's AecConfig on the unit's grid. */
    aec_config_from_preset(&s->res_cfg,AEC_PRESET_BALANCED,s->args.sample_rate);
    s->res_cfg.fft_size=s->grid.fft_size;
    s->res_bytes=aec_res_get_mem_size(&s->res_cfg);
    if (s->res_bytes == 0) return -1;
    s->use_bridge=s->args.sample_rate != DFN2_SR;
    s->n_freqs=s->grid.n_freqs;
    s->vec_bytes=ALIGN16((size_t)s->n_freqs*sizeof(float));
    s->spec_bytes=ALIGN16((size_t)s->n_freqs*sizeof(Complex));

    s->stage_cfg=dfn_res_stage_default_config(DFN2_SR);
    s->stage_cfg.atten_lim_db=s->args.atten_lim_db;
    s->stage_cfg.erb_fwd=g_erb_fwd;
    s->stage_cfg.erb_inv=g_erb_inv;
    s->stage_cfg.model.user=NULL;
    s->stage_cfg.model.infer=run_accelerator;
    s->stage_cfg.model.reset=NULL;
    s->stage_cfg.model.io_descriptor=&s->descriptor;
    if (s->use_bridge) {
        s->bridge_cfg=dfn_rate_bridge_default_config(s->args.sample_rate);
        s->bridge_cfg.fft_size=s->unit_cfg.core.fft_size;
        s->bridge_cfg.stage=s->stage_cfg;
        if (dfn_rate_bridge_get_mem_requirements(&s->bridge_cfg,&bridge_req) != 0) return -1;
        s->stage_bytes=(size_t)bridge_req.bytes;
        s->istft_bytes=0;
    } else {
        if (s->unit_cfg.core.fft_size != 0 && s->unit_cfg.core.fft_size != DFN2_N_FFT) return -1;
        if (dfn_res_stage_get_mem_requirements(&s->stage_cfg,&stage_req) != 0) return -1;
        ac_stft_config_default(&scfg,s->grid.fft_size);
        s->istft_bytes=ac_istft_get_mem_size(&scfg);
        if (s->istft_bytes == 0) return -1;
        s->stage_bytes=(size_t)stage_req.bytes;
    }
    /* unit, RES, unity, total gain, P, then the stage (or bridge) and its synthesis */
    s->total=ALIGN16((size_t)s->unit_req.bytes)+ALIGN16(s->res_bytes)+2*s->vec_bytes+s->spec_bytes+
             ALIGN16(s->stage_bytes)+ALIGN16(s->istft_bytes)+
             (s->use_bridge ? 0 : ALIGN16((size_t)DFN2_N_BINS*sizeof(Complex)));
    return 0;
}

/* A refused hop: back to the state of a fresh pipeline, every stage together. */
static void dfn_reset(DfnSample *s) {
    audio_pipeline_4ch_reset(s->unit);
    aec_res_reset(s->res);
    s->rng=SE_CNG_RNG_SEED;
    dfn_hop_reset(&s->hop);
}

static void dfn_cleanup(void *user) {
    DfnSample *s=user;
    if (s->bridge) dfn_rate_bridge_destroy(s->bridge);
    else if (s->stage) dfn_res_stage_destroy(s->stage);
    if (s->unit) audio_pipeline_4ch_destroy(s->unit);
    free(s->pool);
    s->pool=NULL;
    s->bridge=NULL; s->stage=NULL; s->unit=NULL; s->res=NULL; s->istft=NULL;
}

/* Carves the planned pool: the unit, the RES, the P buffers, then the stage (or bridge) and its
 * synthesis. */
static int dfn_build(DfnSample *s) {
    unsigned char *cur;
    int k;
    if (posix_memalign(&s->pool,s->unit_req.alignment,s->total) != 0) { s->pool=NULL; return -1; }
    cur=s->pool;
    s->unit=audio_pipeline_4ch_init_ex(cur,(size_t)s->unit_req.bytes,&s->unit_cfg,&s->unit_req);
    cur+=ALIGN16((size_t)s->unit_req.bytes);
    if (!s->unit) return -1;
    s->res=aec_res_init(cur,s->res_bytes,&s->res_cfg);
    cur+=ALIGN16(s->res_bytes);
    if (!s->res || aec_res_n_bins(s->res) != audio_pipeline_4ch_n_freqs(s->unit) ||
        aec_res_hop_size(s->res) != audio_pipeline_4ch_hop_size(s->unit))
        return -1;
    s->unity=(float *)cur;      cur+=s->vec_bytes;
    s->total_gain=(float *)cur; cur+=s->vec_bytes;
    s->post=(Complex *)cur;     cur+=s->spec_bytes;
    for (k=0; k<s->n_freqs; ++k) s->unity[k]=1.0f;
    s->rng=SE_CNG_RNG_SEED;
    if (s->use_bridge) {
        s->bridge=dfn_rate_bridge_init(cur,s->stage_bytes,&s->bridge_cfg);
        if (!s->bridge ||
            dfn_rate_bridge_n_freqs(s->bridge) != audio_pipeline_4ch_n_freqs(s->unit) ||
            dfn_rate_bridge_hop_size(s->bridge) != audio_pipeline_4ch_hop_size(s->unit))
            return -1;
        s->stage=dfn_rate_bridge_stage(s->bridge);
        dfn_hop_init_bridge(&s->hop,s->bridge);
    } else {
        AcStftConfig scfg;
        Complex *stage_out;
        if (audio_pipeline_4ch_n_freqs(s->unit) != DFN2_N_BINS ||
            audio_pipeline_4ch_hop_size(s->unit) != DFN2_HOP_LEN)
            return -1;
        s->stage=dfn_res_stage_init(cur,s->stage_bytes,&s->stage_cfg);
        cur+=ALIGN16(s->stage_bytes);
        if (!s->stage) return -1;
        ac_stft_config_default(&scfg,s->grid.fft_size);
        s->istft=ac_istft_init(cur,s->istft_bytes,&scfg);
        cur+=ALIGN16(s->istft_bytes);
        if (!s->istft) return -1;
        stage_out=(Complex *)cur;
        memset(stage_out,0,(size_t)DFN2_N_BINS*sizeof(Complex));
        dfn_hop_init_stage(&s->hop,s->stage,s->istft,stage_out);
        s->hop.check_finite=1;
    }
    return 0;
}

static int dfn_setup(void *user, McabHopView *view) {
    DfnSample *s=user;
    int hop_size, latency;
    if (dfn_example_load_matrices(&s->args, g_erb_fwd, g_erb_inv, name) != 0)
        return 1;

    /* The RES (balanced, built in dfn_plan) is always on; the comfort-noise fill is the program's
     * switch: E and P must share a phase for the DFN2 stage, so it is off unless asked for. */
    s->unit_cfg = audio_pipeline_4ch_default_config(s->args.sample_rate);
    s->unit_cfg.core.fft_size = 0;   /* the core's rate default */
    s->unit_cfg.core.delay_mode = s->args.delay_mode;
    s->unit_cfg.core.delay_num_filters = s->args.delay_num_filters;
    s->unit_cfg.core.fixed_delay_samples = s->args.fixed_delay_samples;
    /* Compiled-in geometry, not the model's metadata (see mono_aec_dfn_res). */
    if (dfn2_model_io_descriptor_default(&s->descriptor) != 0) return 1;

    if (dfn_plan(s) != 0) {
        fprintf(stderr, "%s: pool query rejected the configuration "
                "(host %d Hz, mode=%s n=%d fixed_delay=%d)\n", name,
                s->args.sample_rate, dfn_example_delay_mode_name(s->args.delay_mode),
                s->args.delay_num_filters, s->args.fixed_delay_samples);
        return 1;
    }
    printf("%s: core %d Hz, delay profile mode=%s n=%d fixed_delay=%d (one "
           "shared estimator) cng=%d atten_lim=%.1f dB -> pool %llu bytes "
           "(align %lu), model weights external; model I/O descriptor = "
           "compiled-in layout v%u\n", name, s->args.sample_rate,
           dfn_example_delay_mode_name(s->args.delay_mode),
           s->args.delay_num_filters, s->args.fixed_delay_samples, s->args.enable_cng,
           (double)s->args.atten_lim_db, (unsigned long long)s->total,
           (unsigned long)s->unit_req.alignment, (unsigned)s->descriptor.layout_version);

    if (dfn_build(s) != 0) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        dfn_cleanup(s);
        return 1;
    }
    hop_size = audio_pipeline_4ch_hop_size(s->unit);
    /* Core WOLA hop plus the bridge's added delay, or the model's two lookahead hops plus the
     * core hop at 48 kHz. */
    latency = s->use_bridge
        ? hop_size + dfn_rate_bridge_added_delay_samples(s->bridge)
        : hop_size * (DFN_RES_STAGE_LOOKAHEAD_FRAMES + 1);
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name, hop_size, latency);
    view->hop=hop_size; view->microphones=microphones;
    view->reference=far; view->output=output; view->check_finite=1;
    return 0;
}
static int dfn_read(void *user, McabHopView *view, int frame) {
    (void)user; (void)view; return frame < 8;
}
/* P = E * min(1, G_res), plus comfort noise on the bins the RES cut when enabled (no denoiser:
 * its gain is the unity vector). */
static void dfn_form(DfnSample *s, const SeSpecFrame *f, const float *g_res) {
    sk_min_f32(s->total_gain,s->unity,g_res,s->n_freqs);
    sk_capply_gain_f32(s->post,f->e,s->total_gain,s->n_freqs);
    if (s->args.enable_cng && f->cn)
        aec_cng_fill(&s->rng,s->post,f->cn,g_res,s->unity,SE_CNG_NR_GAIN_FLOOR,s->n_freqs);
}

static int dfn_process(void *user, McabHopView *view, int frame) {
    DfnSample *s=user; (void)view; (void)frame;
    AudioPipeline4ChBeam beam;
    SeSpecFrame spec;
    const float *g_res;
    if (audio_pipeline_4ch_process(s->unit,microphones,far,&beam,NULL) != FOUR_AEC_NR_RES_OK ||
        audio_pipeline_4ch_fuse(s->unit,&spec) != FOUR_AEC_NR_RES_OK ||
        !(g_res=se_res_stage_gain(s->res,&spec))) {
        dfn_reset(s);
        return 1;
    }
    dfn_form(s,&spec,g_res);
    if (dfn_hop_process(&s->hop,spec.e,s->post,output) != 0) {
        dfn_reset(s);
        return 1;
    }
    return 0;
}
static void dfn_complete(void *user) {
    (void)user;
    printf("%s: fail-open board skeleton PASS\n",name);
}
int main(int argc, char **argv) {
    DfnSample sample={0};
    static const McabSampleOps ops={dfn_parse,2,dfn_setup,dfn_read,dfn_process,
                                  NULL,NULL,NULL,dfn_cleanup,dfn_complete};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
