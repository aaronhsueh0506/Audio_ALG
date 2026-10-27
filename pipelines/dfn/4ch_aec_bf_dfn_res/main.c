/* 4ch_aec_bf_dfn_res board skeleton: four-channel linear AEC -> beamformer (bf) -> RES ->
 * DeepFilterNet2 composed from the library calls. Example code: the four lanes, the shared
 * estimator and the mcab flow (mcab_sample_flow.h builds them in the application's order) with the
 * sample SRP-PHAT/BF helper (mcab_sample_beam.h), the post-beam RES (lib/aec AecRes) initialised
 * here, P = E * G_res (+ comfort noise) formed here, and dfn/dfn_hop.h for the DFN2 hop (sized
 * and carved by dfn/dfn_hop_pool.h). Per hop:
 *
 *     mcab_sample_beam_process(unit, mics, far, &beam, &info);    // delay, lanes, SRP-PHAT, BF
 *     mcab_sample_beam_fuse(unit, &frame);                         // E, R2, CN
 *     mcab_sample_res_input(&frame, &in);                          // the RES input of that frame
 *     G_res = aec_res_process(res, &in);
 *     P = E * G_res (+ comfort noise);                              // before the DFN hop begins
 *     dfn_hop_process(&hop, frame.e, P, out);                       // or begin / nn_begin / ...
 *
 * 48 kHz hosts drive dfn_res_stage.h and synthesise through an AcIstft; 16 kHz hosts run the same
 * stage on its 48 kHz grid inside dfn_rate_bridge.h. The first output hops are zero: the model's
 * DFN_RES_STAGE_LOOKAHEAD_FRAMES hops at 48 kHz (two for the shipped 1/1 lookahead pair), the
 * bridge's added delay otherwise (printed below). The stage does not reset on a delay
 * change of the shared aligner.
 *
 * The program builds the pool from the command-line delay profile and ERB matrices, installs a
 * fail-open model callback and runs a few silent hops. Replace run_accelerator() with the board
 * runtime call. */
#include "aec_res.h"
#include "4ch_aec_bf_nr_res/mcab.h"
#include "4ch_aec_bf_nr_res/mcab_sample_beam.h"
#include "4ch_aec_bf_nr_res/mcab_sample_flow.h"
#include "4ch_aec_bf_nr_res/mcab_sample_post.h"
#include "dfn/dfn_hop.h"
#include "dfn/dfn_hop_pool.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "dfn/dfn_example_args.h"
#include "4ch_aec_bf_nr_res/mcab_sample_driver.h"

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
static float microphones[DFN2_HOP_LEN*MCAB_CHANNELS];
static float far[DFN2_HOP_LEN],output[DFN2_HOP_LEN];
typedef struct DfnSample {
    DfnExampleArgs args;
    ExampleCliDelayOptions delay;
    McabConfig unit_cfg;
    McabSampleBeamParams unit_bp;
    McabMemReq unit_req;              /* mcab's own descriptor: the pool alignment */
    McabMemBreakdown grid;
    size_t flow_bytes, beam_bytes;    /* lanes + estimator + mcab pool; SRP/BF helper */
    McabSampleFlow objs;
    DfnHopPoolPlan dfn;               /* the stage (or bridge), its synthesis and output */
    DFN2ModelIoDescriptor descriptor;
    AecConfig res_cfg;
    size_t vec_bytes, spec_bytes, res_bytes, total;
    void *pool;
    McabSampleBeam *unit;
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
        if (example_cli_parse_int(value,&s->args.sample_rate) != 0 ||
            !dfn_example_rate_known(HOST_RATES,s->args.sample_rate)) {
            fprintf(stderr,"%s: --sample-rate '%s' is not a host grid of "
                "this skeleton (accepted: %s)\n",name,value,s->rates_text); return 2;
        }
    } else if (!strcmp(arg,"--erb-fwd")) s->args.erb_fwd_path=value;
    else if (!strcmp(arg,"--erb-inv")) s->args.erb_inv_path=value;
    else if (!strcmp(arg,"--atten-lim")) {
        if (example_cli_parse_float(value,&s->args.atten_lim_db) != 0) {
            fprintf(stderr,"%s: --atten-lim '%s' is not a finite dB value\n",name,value);
            return 2;
        }
    } else if (!strcmp(arg,"--cng")) s->args.enable_cng=1;
    else return example_cli_delay_set_option(&s->delay,arg,value,name);
    return 0;
}
static int dfn_parse(void *user, int argc, char **argv) {
    DfnSample *s=user;
    static const McabSampleCliOption options[]={
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
    int rc=mcab_sample_scan_cli(argc,argv,options,sizeof(options)/sizeof(options[0]),
                          s,dfn_set,dfn_unknown);
    if (rc != 0) return rc;
    if ((s->args.erb_fwd_path==NULL)!=(s->args.erb_inv_path==NULL)) {
        fprintf(stderr,"%s: --erb-fwd and --erb-inv go together\n",name); return 2;
    }
    rc=example_cli_delay_validate(&s->delay,name);
    s->args.delay_mode=s->delay.mode;
    s->args.delay_num_filters=s->delay.num_filters;
    s->args.fixed_delay_samples=s->delay.fixed_samples;
    return rc;
}

/* Everything the pool holds, sized once. 0, or -1 for a configuration no stage accepts. */
static int dfn_plan(DfnSample *s) {
    if (mcab_get_mem_requirements(&s->unit_cfg,&s->unit_req) != 0 ||
        mcab_get_mem_breakdown(&s->unit_cfg,&s->grid) != 0 ||
        mcab_sample_flow_get_mem_size(&s->unit_cfg,&s->flow_bytes) != 0 ||
        mcab_sample_beam_get_mem_size(&s->unit_cfg,&s->unit_bp,&s->beam_bytes) != 0 ||
        !isfinite(s->args.atten_lim_db))
        return -1;
    /* The RES: the balanced preset's AecConfig on the unit's grid. */
    aec_config_from_preset(&s->res_cfg,AEC_PRESET_BALANCED,s->args.sample_rate);
    s->res_cfg.fft_size=s->grid.fft_size;
    s->res_bytes=aec_res_get_mem_size(&s->res_cfg);
    if (s->res_bytes == 0) return -1;
    s->n_freqs=s->grid.n_freqs;
    s->vec_bytes=ALIGN16((size_t)s->n_freqs*sizeof(float));
    s->spec_bytes=ALIGN16((size_t)s->n_freqs*sizeof(Complex));

    s->dfn.stage_cfg=dfn_res_stage_default_config(DFN2_SR);
    s->dfn.stage_cfg.atten_lim_db=s->args.atten_lim_db;
    s->dfn.stage_cfg.erb_fwd=g_erb_fwd;
    s->dfn.stage_cfg.erb_inv=g_erb_inv;
    s->dfn.stage_cfg.model.user=NULL;
    s->dfn.stage_cfg.model.infer=run_accelerator;
    s->dfn.stage_cfg.model.reset=NULL;
    s->dfn.stage_cfg.model.io_descriptor=&s->descriptor;
    if (dfn_hop_pool_plan(&s->dfn,s->args.sample_rate,s->unit_cfg.fft_size) != 0) return -1;
    /* flow objects, helper, RES, unity, total gain, P, then the stage (or bridge) and its synthesis */
    s->total=ALIGN16(s->flow_bytes)+ALIGN16(s->beam_bytes)+ALIGN16(s->res_bytes)+2*s->vec_bytes+s->spec_bytes+
             dfn_hop_pool_bytes(&s->dfn);
    return 0;
}

/* A refused hop: back to the state of a fresh pipeline, every stage together. */
static void dfn_reset(DfnSample *s) {
    mcab_sample_beam_reset(s->unit);
    aec_res_reset(s->res);
    s->rng=MCAB_SAMPLE_CNG_RNG_SEED;
    dfn_hop_reset(&s->hop);
}

static void dfn_cleanup(void *user) {
    DfnSample *s=user;
    if (s->bridge) dfn_rate_bridge_destroy(s->bridge);
    else if (s->stage) dfn_res_stage_destroy(s->stage);
    if (s->unit) mcab_sample_beam_destroy(s->unit);
    mcab_sample_flow_destroy(&s->objs);
    free(s->pool);
    s->pool=NULL;
    s->bridge=NULL; s->stage=NULL; s->unit=NULL; s->res=NULL; s->istft=NULL;
}

/* Carves the planned pool: the flow's objects, the helper, the RES, the P buffers, then the stage
 * (or bridge) and its synthesis. */
static int dfn_build(DfnSample *s) {
    unsigned char *cur;
    int k;
    if (posix_memalign(&s->pool,s->unit_req.alignment,s->total) != 0) { s->pool=NULL; return -1; }
    cur=s->pool;
    if (mcab_sample_flow_init(&s->objs,cur,s->flow_bytes,&s->unit_cfg) != 0) return -1;
    cur+=ALIGN16(s->flow_bytes);
    s->unit=mcab_sample_beam_init(cur,s->beam_bytes,&s->unit_cfg,&s->unit_bp,s->objs.flow);
    cur+=ALIGN16(s->beam_bytes);
    if (!s->unit) return -1;
    s->res=aec_res_init(cur,s->res_bytes,&s->res_cfg);
    cur+=ALIGN16(s->res_bytes);
    if (!s->res || aec_res_n_bins(s->res) != mcab_n_freqs(s->objs.flow) ||
        aec_res_hop_size(s->res) != mcab_hop_size(s->objs.flow))
        return -1;
    s->unity=(float *)cur;      cur+=s->vec_bytes;
    s->total_gain=(float *)cur; cur+=s->vec_bytes;
    s->post=(Complex *)cur;     cur+=s->spec_bytes;
    for (k=0; k<s->n_freqs; ++k) s->unity[k]=1.0f;
    s->rng=MCAB_SAMPLE_CNG_RNG_SEED;
    return dfn_hop_pool_build(&s->dfn,cur,mcab_n_freqs(s->objs.flow),mcab_hop_size(s->objs.flow),1,
                              &s->hop,&s->stage,&s->bridge,&s->istft);
}

static int dfn_setup(void *user, McabSampleHopView *view) {
    DfnSample *s=user;
    int hop_size, latency;
    if (dfn_example_load_matrices(&s->args, g_erb_fwd, g_erb_inv, name) != 0)
        return 1;

    /* The RES (balanced, built in dfn_plan) is always on; the comfort-noise fill is the program's
     * switch: E and P must share a phase for the DFN2 stage, so it is off unless asked for. */
    s->unit_cfg = mcab_default_config(s->args.sample_rate);
    s->unit_bp = mcab_sample_beam_default_params();
    s->unit_cfg.fft_size = 0;   /* the flow's rate default */
    s->unit_cfg.delay_mode = s->args.delay_mode;
    s->unit_cfg.delay_num_filters = s->args.delay_num_filters;
    s->unit_cfg.fixed_delay_samples = s->args.fixed_delay_samples;
    /* Compiled-in geometry, not the model's metadata (see mono_aec_dfn_res). */
    if (dfn2_model_io_descriptor_default(&s->descriptor) != 0) return 1;

    if (dfn_plan(s) != 0) {
        fprintf(stderr, "%s: pool query rejected the configuration "
                "(host %d Hz, mode=%s n=%d fixed_delay=%d)\n", name,
                s->args.sample_rate, example_cli_delay_mode_name(s->args.delay_mode),
                s->args.delay_num_filters, s->args.fixed_delay_samples);
        return 1;
    }
    printf("%s: core %d Hz, delay profile mode=%s n=%d fixed_delay=%d (one "
           "shared estimator) cng=%d atten_lim=%.1f dB -> pool %llu bytes "
           "(align %lu), model weights external; model I/O descriptor = "
           "compiled-in layout v%u\n", name, s->args.sample_rate,
           example_cli_delay_mode_name(s->args.delay_mode),
           s->args.delay_num_filters, s->args.fixed_delay_samples, s->args.enable_cng,
           (double)s->args.atten_lim_db, (unsigned long long)s->total,
           (unsigned long)s->unit_req.alignment, (unsigned)s->descriptor.layout_version);

    if (dfn_build(s) != 0) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        dfn_cleanup(s);
        return 1;
    }
    hop_size = mcab_hop_size(s->objs.flow);
    /* Core WOLA hop plus the bridge's added delay, or the model's lookahead hops plus the
     * core hop at 48 kHz. */
    DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK();   /* the pair printed is the archive's */
    latency = s->dfn.use_bridge
        ? hop_size + dfn_rate_bridge_added_delay_samples(s->bridge)
        : hop_size * (DFN_RES_STAGE_LOOKAHEAD_FRAMES + 1);
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name, hop_size, latency);
    view->hop=hop_size; view->microphones=microphones;
    view->reference=far; view->output=output; view->check_finite=1;
    return 0;
}
static int dfn_read(void *user, McabSampleHopView *view, int frame) {
    (void)user; (void)view; return frame < 8;
}
/* P = E * min(1, G_res), plus comfort noise on the bins the RES cut when enabled (no denoiser:
 * its gain is the unity vector). */
static void dfn_form(DfnSample *s, const McabFrame *f, const float *g_res) {
    sk_min_f32(s->total_gain,s->unity,g_res,s->n_freqs);
    sk_capply_gain_f32(s->post,f->e,s->total_gain,s->n_freqs);
    if (s->args.enable_cng && f->cn)
        aec_cng_fill(&s->rng,s->post,f->cn,g_res,s->unity,MCAB_SAMPLE_CNG_NR_GAIN_FLOOR,
                     s->n_freqs);
}

static int dfn_process(void *user, McabSampleHopView *view, int frame) {
    DfnSample *s=user; (void)view; (void)frame;
    McabSampleBeamResult beam;
    McabFrame spec;
    AecResInput in;
    const float *g_res = NULL;
    if (mcab_sample_beam_process(s->unit,microphones,far,&beam,NULL) == MCAB_OK &&
        mcab_sample_beam_fuse(s->unit,&spec) == MCAB_OK) {
        mcab_sample_res_input(&spec,&in);
        g_res=aec_res_process(s->res,&in);
    }
    if (!g_res) {
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
