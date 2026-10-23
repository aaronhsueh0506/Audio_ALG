/* 4ch_aec_bf_dfn_res board skeleton: builds the wrapper from the command-line delay
 * profile and ERB matrices, installs a fail-open model callback and runs a few
 * silent hops with uniform beamformer weights standing in for the product's
 * beamformer. Replace run_accelerator() with the board runtime call and the weights
 * with the beamformer's effective response. */
#include "4aec_dfn_res.h"
#include "dfn_example_args.h"
#include "mcab_sample_driver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    FourAecDfnResConfig cfg;
    FourAecDfnResMemReq req;
    DFN2ModelIoDescriptor descriptor;
    FourAecDfnRes *pipeline;
    const char *program;
    char rates_text[64];
    int hop_size;
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
static int dfn_setup(void *user, McabHopView *view) {
    DfnSample *s=user;
    if (dfn_example_load_matrices(&s->args, g_erb_fwd, g_erb_inv, name) != 0)
        return 1;
    for (int index = 0; index < FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS; ++index) {
        g_weights[index].r = 1.0f / (float)FOUR_AEC_NR_RES_CHANNELS;
        g_weights[index].i = 0.0f;
    }

    s->cfg = four_aec_dfn_res_default_config();
    s->cfg.front_end.sample_rate = s->args.sample_rate;
    s->cfg.front_end.fft_size = 0;   /* the core's rate default */
    s->cfg.front_end.delay_mode = s->args.delay_mode;
    s->cfg.front_end.delay_num_filters = s->args.delay_num_filters;
    s->cfg.front_end.fixed_delay_samples = s->args.fixed_delay_samples;
    s->cfg.front_end.enable_cng = s->args.enable_cng;
    s->cfg.atten_lim_db = s->args.atten_lim_db;
    s->cfg.erb_fwd = g_erb_fwd;
    s->cfg.erb_inv = g_erb_inv;
    /* Compiled-in geometry, not the model's metadata (see mono_aec_dfn_res). */
    if (dfn2_model_io_descriptor_default(&s->descriptor) != 0) return 1;
    s->cfg.model.user = NULL;
    s->cfg.model.infer = run_accelerator;
    s->cfg.model.reset = NULL;
    s->cfg.model.io_descriptor = &s->descriptor;

    if (four_aec_dfn_res_get_mem_requirements(&s->cfg, &s->req) != 0) {
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
           (double)s->args.atten_lim_db, (unsigned long long)s->req.bytes,
           (unsigned long)s->req.alignment, (unsigned)s->descriptor.layout_version);

    s->pipeline = four_aec_dfn_res_create(&s->cfg);
    if (!s->pipeline) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        return 1;
    }
    s->hop_size = four_aec_dfn_res_hop_size(s->pipeline);
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name,
           s->hop_size, four_aec_dfn_res_lookahead_samples(s->pipeline));
    view->hop=s->hop_size; view->microphones=microphones;
    view->reference=far; view->output=output; view->check_finite=1;
    return 0;
}
static int dfn_read(void *user, McabHopView *view, int frame) {
    (void)user; (void)view; return frame < 8;
}
static int dfn_process(void *user, McabHopView *view, int frame) {
    DfnSample *s=user; (void)view; (void)frame;
    FourAecFront front;
    return four_aec_dfn_res_front_process(s->pipeline,microphones,far,&front) != 0 ||
           four_aec_dfn_res_process(s->pipeline,g_weights,NULL,output) != 0;
}
static void dfn_complete(void *user) {
    (void)user;
    printf("%s: fail-open board skeleton PASS\n",name);
}
static void dfn_cleanup(void *user) {
    DfnSample *s=user; four_aec_dfn_res_destroy(s->pipeline);
}
int main(int argc, char **argv) {
    DfnSample sample={0};
    static const McabSampleOps ops={dfn_parse,2,dfn_setup,dfn_read,dfn_process,
                                  NULL,NULL,NULL,dfn_cleanup,dfn_complete};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
