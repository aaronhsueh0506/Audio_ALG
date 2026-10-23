/* 4ch_aec_bf_dfn_res board skeleton: four-channel linear AEC -> external beamformer (bf) -> RES ->
 * DeepFilterNet2 composed from the library calls. Example code: the four-lane core
 * (4aec_nr_res.h) for the linear stage, samples/se_sample_chain.h (RES gain only, no denoiser, no
 * synthesis) for P = E * G_res (+ comfort noise), and samples/dfn_hop.h for the DFN2 hop. Per hop:
 *
 *     four_aec_front_process(core, mics, far, &front);      // shared delay + four lanes
 *     four_aec_fuse(core, weights, NULL, &frame);           // beamformer weights -> E, R2, CN
 *     P = se_sample_chain_form(chain, &frame);              // formed before the DFN hop begins
 *     dfn_hop_process(&hop, frame.e, P, out);               // or begin / nn_begin / nn_end / end
 *
 * 48 kHz hosts drive dfn_res_stage.h and synthesise through an AcIstft; 16 kHz hosts run the same
 * stage on its 48 kHz grid inside dfn_rate_bridge.h. The first output hops are zero: two model hops
 * at 48 kHz, the bridge's added delay otherwise (printed below). The stage does not reset on a delay
 * change of the shared aligner.
 *
 * The program builds the pool from the command-line delay profile and ERB matrices, installs a
 * fail-open model callback and runs a few silent hops with uniform beamformer weights standing in
 * for the product's beamformer. Replace run_accelerator() with the board runtime call and the
 * weights with the beamformer's effective response. */
#include "4aec_nr_res.h"
#include "samples/dfn_hop.h"
#include "samples/se_sample_chain.h"
#include "utility/mem_align.h"
#include "dfn_example_args.h"
#include "mcab_sample_driver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static Complex g_weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];

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
    FourAecNrResConfig core_cfg;
    FourAecNrResMemReq core_req;
    FourAecNrResMemBreakdown grid;
    SeSampleChainConfig chain_cfg;
    DfnResStageConfig stage_cfg;
    DfnRateBridgeConfig bridge_cfg;
    DFN2ModelIoDescriptor descriptor;
    int use_bridge;                   /* host below 48 kHz */
    size_t chain_bytes, stage_bytes, istft_bytes, total;
    void *pool;
    FourAecNrRes *core;
    SeSampleChain *chain;
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
    if (four_aec_nr_res_get_mem_requirements(&s->core_cfg,&s->core_req) != 0 ||
        four_aec_nr_res_get_mem_breakdown(&s->core_cfg,&s->grid) != 0 ||
        !isfinite(s->args.atten_lim_db))
        return -1;
    s->use_bridge=s->args.sample_rate != DFN2_SR;

    /* RES gain only, no denoiser, no synthesis here; the comfort-noise fill is the program's
     * switch. E and P must share a phase for the DFN2 stage, so it is off unless asked for. */
    s->chain_cfg=se_sample_chain_default_config(s->args.sample_rate,s->grid.fft_size);
    s->chain_cfg.res=SE_SAMPLE_RES_AEC_RES;
    s->chain_cfg.res_preset=AEC_PRESET_BALANCED;
    s->chain_cfg.enable_nr=0;
    s->chain_cfg.enable_cng=s->args.enable_cng;
    s->chain_cfg.with_synthesis=0;
    s->chain_bytes=se_sample_chain_mem_size(&s->chain_cfg);
    if (s->chain_bytes == 0) return -1;

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
        s->bridge_cfg.fft_size=s->core_cfg.fft_size;
        s->bridge_cfg.stage=s->stage_cfg;
        if (dfn_rate_bridge_get_mem_requirements(&s->bridge_cfg,&bridge_req) != 0) return -1;
        s->stage_bytes=(size_t)bridge_req.bytes;
        s->istft_bytes=0;
    } else {
        if (s->core_cfg.fft_size != 0 && s->core_cfg.fft_size != DFN2_N_FFT) return -1;
        if (dfn_res_stage_get_mem_requirements(&s->stage_cfg,&stage_req) != 0) return -1;
        ac_stft_config_default(&scfg,s->grid.fft_size);
        s->istft_bytes=ac_istft_get_mem_size(&scfg);
        if (s->istft_bytes == 0) return -1;
        s->stage_bytes=(size_t)stage_req.bytes;
    }
    s->total=ALIGN16((size_t)s->core_req.bytes)+ALIGN16(s->chain_bytes)+
             ALIGN16(s->stage_bytes)+ALIGN16(s->istft_bytes)+
             (s->use_bridge ? 0 : ALIGN16((size_t)DFN2_N_BINS*sizeof(Complex)));
    return 0;
}

/* A refused hop: back to the state of a fresh pipeline, every stage together. */
static void dfn_reset(DfnSample *s) {
    four_aec_nr_res_reset(s->core);
    se_sample_chain_reset(s->chain);
    dfn_hop_reset(&s->hop);
}

static void dfn_cleanup(void *user) {
    DfnSample *s=user;
    if (s->bridge) dfn_rate_bridge_destroy(s->bridge);
    else if (s->stage) dfn_res_stage_destroy(s->stage);
    if (s->chain) se_sample_chain_destroy(s->chain);
    if (s->core) four_aec_nr_res_destroy(s->core);
    free(s->pool);
    s->pool=NULL;
    s->bridge=NULL; s->stage=NULL; s->chain=NULL; s->core=NULL; s->istft=NULL;
}

/* Carves the planned pool: core, chain, then the stage (or bridge) and its synthesis. */
static int dfn_build(DfnSample *s) {
    unsigned char *cur;
    if (posix_memalign(&s->pool,s->core_req.alignment,s->total) != 0) { s->pool=NULL; return -1; }
    cur=s->pool;
    s->core=four_aec_nr_res_init_ex(cur,(size_t)s->core_req.bytes,&s->core_cfg,&s->core_req);
    cur+=ALIGN16((size_t)s->core_req.bytes);
    if (!s->core) return -1;
    s->chain=se_sample_chain_init(cur,s->chain_bytes,&s->chain_cfg);
    cur+=ALIGN16(s->chain_bytes);
    if (!s->chain) return -1;
    if (s->use_bridge) {
        s->bridge=dfn_rate_bridge_init(cur,s->stage_bytes,&s->bridge_cfg);
        if (!s->bridge ||
            dfn_rate_bridge_n_freqs(s->bridge) != four_aec_nr_res_n_freqs(s->core) ||
            dfn_rate_bridge_hop_size(s->bridge) != four_aec_nr_res_hop_size(s->core))
            return -1;
        s->stage=dfn_rate_bridge_stage(s->bridge);
        dfn_hop_init_bridge(&s->hop,s->bridge);
    } else {
        AcStftConfig scfg;
        Complex *stage_out;
        if (four_aec_nr_res_n_freqs(s->core) != DFN2_N_BINS ||
            four_aec_nr_res_hop_size(s->core) != DFN2_HOP_LEN)
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
    int hop_size, latency, index;
    if (dfn_example_load_matrices(&s->args, g_erb_fwd, g_erb_inv, name) != 0)
        return 1;
    for (index = 0; index < FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS; ++index) {
        g_weights[index].r = 1.0f / (float)FOUR_AEC_NR_RES_CHANNELS;
        g_weights[index].i = 0.0f;
    }

    s->core_cfg = four_aec_nr_res_default_config(s->args.sample_rate);
    s->core_cfg.fft_size = 0;   /* the core's rate default */
    s->core_cfg.delay_mode = s->args.delay_mode;
    s->core_cfg.delay_num_filters = s->args.delay_num_filters;
    s->core_cfg.fixed_delay_samples = s->args.fixed_delay_samples;
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
           (unsigned long)s->core_req.alignment, (unsigned)s->descriptor.layout_version);

    if (dfn_build(s) != 0) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        dfn_cleanup(s);
        return 1;
    }
    hop_size = four_aec_nr_res_hop_size(s->core);
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
static int dfn_process(void *user, McabHopView *view, int frame) {
    DfnSample *s=user; (void)view; (void)frame;
    FourAecFront front;
    SeSpecFrame spec;
    const Complex *post;
    if (four_aec_front_process(s->core,microphones,far,&front) != FOUR_AEC_NR_RES_OK ||
        four_aec_fuse(s->core,g_weights,NULL,&spec) != FOUR_AEC_NR_RES_OK ||
        !(post=se_sample_chain_form(s->chain,&spec)) ||
        dfn_hop_process(&s->hop,spec.e,post,output) != 0) {
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
