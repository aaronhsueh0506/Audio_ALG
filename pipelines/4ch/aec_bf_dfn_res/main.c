/* 4ch_aec_bf_dfn_res board skeleton: four-microphone array AEC -> beamformer (bf) -> RES ->
 * DeepFilterNet2 composed from the library calls. Example code: the array instance of lib/aec
 * (aec.h, n_mics = 4: four lanes, the shared estimator and the residual gain stage; one pool placed
 * by aec_get_mem_requirements() + aec_init_ex()) with the sample SRP-PHAT/BF helper (array_sample_beam.h), P = B * G_res (+ comfort noise)
 * formed here over the rendering base B, and dfn/dfn_hop.h for the DFN2 hop (sized and carved by
 * dfn/dfn_hop_pool.h). Per hop:
 *
 *     array_sample_beam_process(unit, mics, far, &beam, &info);    // delay, lanes, SRP-PHAT, BF
 *     array_sample_beam_fuse(unit, &frame);                         // aec_process_residual + aec_get_residual
 *     aec_apply_gain(aec, NULL, P);                                // P = B * G_res (estimated on E)
 *     aec_apply_cng(aec, NULL, P);                                 // optional comfort noise
 *     dfn_hop_process(&hop, frame.error_spec, P, out);             // estimated on E, applied on P
 *
 * B is E except on hops where a contributing lane's output guard fired (aec.h); there P carries
 * that lane's capture while the model still estimates on E. The array reseeds its comfort noise
 * on a lane's post-chain reset (frame.post_reset_serial moved) before P is formed. On the bridge path
 * (host rate differs from the model grid) the host-grid synthesis is inside the bridge and runs at
 * the top of its begin(), so ainr_bridge_reset_synthesis() on that hop clears just its
 * overlap-add tail (DFN2 and the resamplers keep running, no delay line). On the direct path (host
 * rate equals the model grid's) the frame built on the hop of the event reaches the synthesis
 * dfn_hop_lookahead_frames() hops later: a per-hop shift register (starting at 0) delays the event
 * and ac_istft_reset() clears only the hop's synthesis just before that push; DFN2 keeps running. Here
 * that serial moves only with an array reset, which the refused-call path below pairs with a hop reset.
 *
 * A host on the model's own rate (--ai-sample-rate, default 48 kHz) drives the model stream
 * (stream/ainr/dfn2, memory mode) and synthesises through an AcIstft; any other host rate runs the
 * same stream on its 48 kHz grid inside stream/ainr/bridge's AinrBridge (which needs the 48 kHz /
 * 1024 model). The model grid is built from the --ai-sample-rate / --ai-fft-size pair
 * (dfn_example_args.h) with dfn2_grid_make() and handed to every object. The first output hops are
 * zero: the model's dfn_hop_lookahead_frames() hops on the direct path (two for the shipped 1/1
 * lookahead pair), the bridge's added delay otherwise (printed below). The hop does not reset on a
 * delay change of the shared aligner.
 *
 * The program builds the pool from the command-line delay profile and ERB matrices, installs a
 * fail-open model callback and runs a few silent hops. Replace run_accelerator() with the board
 * runtime call. */
#include "aec.h"
#include "4ch/array_sample_beam.h"
#include "4ch/array_sample_post.h"
#include "dfn/dfn_hop.h"
#include "dfn/dfn_hop_pool.h"
#include "utility/mem_align.h"
#include "dfn/dfn_example_args.h"
#include "4ch/array_sample_driver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Host hop buffers: the largest host hop HOST_RATES offers (48 kHz, FFT 1024). */
#define HOST_MAX_HOP 512

/* Replace with the board runtime call (see mono_aec_dfn_res). */
static int run_accelerator(void *user, const DFN2PrepostInputs *inputs,
                           DFN2PrepostOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return STREAM_RUN_NOT_STARTED; /* BOARD: invoke the stateless DFN2 graph. */
}

/* Host grids accepted at parse time (0-terminated; no 8 kHz). */
static const int HOST_RATES[] = { 16000, 48000, 0 };

static const char *name="4ch_aec_bf_dfn_res";
static float microphones[HOST_MAX_HOP*AEC_ARRAY_MICS];
static float far[HOST_MAX_HOP],output[HOST_MAX_HOP];
typedef struct DfnSample {
    DfnExampleArgs args;
    ExampleCliDelayOptions delay;
    AecConfig unit_cfg;
    ArraySampleBeamParams unit_bp;
    AecMemReq unit_req;               /* the array's descriptor: the pool alignment */
    AecSignalGrid grid;
    size_t aec_bytes, beam_bytes;     /* the array pool; SRP/BF helper */
    Aec *aec;
    DfnHopPoolPlan dfn;               /* the model stream, its conversion buffers, the synthesis or bridge */
    DfnExampleErb erb;                /* the model stream borrows these matrices until the pool is freed */
    DFN2ModelIoDescriptor descriptor;
    size_t spec_bytes, total;
    void *pool;
    ArraySampleBeam *unit;
    Complex *post;                    /* P */
    uint32_t reset_handled;           /* AecResidualView.post_reset_serial last followed */
    uint32_t reset_pending;           /* bit k: a hard reset was seen k hops ago (the direct path's delay line) */
    int reset_delay;                  /* hops the direct path delays a frame by (the model's lookahead) */
    DfnHop hop;                       /* hop.bridge / hop.istft: the bridge, or the synthesis */
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
    } else if (!strcmp(arg,"--ai-sample-rate") || !strcmp(arg,"--ai-fft-size")) {
        if (example_cli_parse_int(value,!strcmp(arg,"--ai-sample-rate") ? &s->args.ai_sample_rate
                                                                       : &s->args.ai_fft_size) != 0) {
            fprintf(stderr,"%s: %s '%s' is not an integer\n",name,arg,value); return 2;
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
    static const ArraySampleCliOption options[]={
        {"-h",0},{"--help",0},{"--sample-rate",1},{"--ai-sample-rate",1},{"--ai-fft-size",1},
        {"--delay-mode",1},
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
    int rc=array_sample_scan_cli(argc,argv,options,sizeof(options)/sizeof(options[0]),
                          s,dfn_set,dfn_unknown);
    if (rc != 0) return rc;
    if (dfn_example_resolve_model_grid(&s->args,name) != 0) return 2;
    if ((s->args.erb_fwd_path==NULL)!=(s->args.erb_inv_path==NULL)) {
        fprintf(stderr,"%s: --erb-fwd and --erb-inv go together\n",name); return 2;
    }
    rc=example_cli_delay_validate(&s->delay,name);
    s->args.delay_mode=s->delay.mode;
    s->args.delay_num_filters=s->delay.num_filters;
    s->args.fixed_delay_samples=s->delay.fixed_samples;
    return rc;
}

/* Everything the pool holds, sized once. 0, or -1 for a configuration nothing accepts. */
static int dfn_plan(DfnSample *s) {
    if (aec_get_mem_requirements(&s->unit_cfg,&s->unit_req) != 0 ||
        aec_resolve_signal_grid(s->unit_cfg.sample_rate,s->unit_cfg.fft_size,&s->grid) != AEC_OK ||
        array_sample_beam_get_mem_size(&s->unit_cfg,&s->unit_bp,&s->beam_bytes) != 0 ||
        s->grid.fft_size/2 > HOST_MAX_HOP || !isfinite(s->args.atten_lim_db))
        return -1;
    s->aec_bytes=ALIGN16((size_t)s->unit_req.bytes);
    s->spec_bytes=ALIGN16((size_t)s->grid.n_freqs*sizeof(Complex));

    dfn_hop_pool_plan_init(&s->dfn,&s->args.model_grid);
    s->dfn.stream_cfg.atten_lim_db=s->args.atten_lim_db;
    s->dfn.stream_cfg.erb_fwd=s->erb.fwd;
    s->dfn.stream_cfg.erb_inv=s->erb.inv;
    s->dfn.model.user=NULL;
    s->dfn.model.infer=run_accelerator;
    s->dfn.model.reset=NULL;
    s->dfn.model.io_descriptor=&s->descriptor;
    if (dfn_hop_pool_plan(&s->dfn,s->args.sample_rate,0) != 0) return -1;   /* the rate's default grid */
    /* array, helper, P, then the model stream, its buffers and the synthesis (or bridge) */
    s->total=s->aec_bytes+ALIGN16(s->beam_bytes)+s->spec_bytes+
             dfn_hop_pool_bytes(&s->dfn);
    return 0;
}

/* A refused hop: back to the state of a fresh pipeline, every stage together. */
static void dfn_reset(DfnSample *s) {
    array_sample_beam_reset(s->unit);
    s->reset_handled=aec_post_reset_serial(s->aec);
    s->reset_pending=0u;
    dfn_hop_reset(&s->hop);
}

static void dfn_cleanup(void *user) {
    DfnSample *s=user;
    if (s->unit) array_sample_beam_destroy(s->unit);
    aec_destroy(s->aec);
    s->aec=NULL;
    free(s->pool);
    s->pool=NULL;
    dfn_example_erb_free(&s->erb);
    s->unit=NULL;
    memset(&s->hop,0,sizeof(s->hop));
}

/* Carves the planned pool: the array, the helper, the P buffer, then the model stream, its
 * buffers and the synthesis (or bridge). */
static int dfn_build(DfnSample *s) {
    unsigned char *cur;
    if (posix_memalign(&s->pool,s->unit_req.alignment,s->total) != 0) { s->pool=NULL; return -1; }
    cur=s->pool;
    s->aec=aec_init_ex(cur,s->aec_bytes,&s->unit_cfg,&s->unit_req);
    if (!s->aec) return -1;
    cur+=s->aec_bytes;
    s->unit=array_sample_beam_init(cur,s->beam_bytes,&s->unit_cfg,&s->unit_bp,s->aec);
    cur+=ALIGN16(s->beam_bytes);
    if (!s->unit) return -1;
    s->post=(Complex *)cur;     cur+=s->spec_bytes;
    s->reset_handled=0u;
    s->reset_pending=0u;
    s->reset_delay=dfn_hop_lookahead_frames(&s->args.model_grid);
    return dfn_hop_pool_build(&s->dfn,cur,s->grid.n_freqs,s->grid.hop_size,1,&s->hop);
}

static int dfn_setup(void *user, ArraySampleHopView *view) {
    DfnSample *s=user;
    int hop_size, latency;
    if (dfn_example_load_matrices(&s->args, &s->erb, name) != 0)
        return 1;

    /* The residual gain stage (balanced, in the array) is always on; the comfort-noise fill is the
     * program's switch: E and P must share a phase for the DFN2 hop, so it is off unless asked for. */
    array_sample_default_config(&s->unit_cfg, s->args.sample_rate);
    s->unit_bp = array_sample_beam_default_params();
    s->unit_cfg.delay_mode = s->args.delay_mode;
    s->unit_cfg.delay_num_filters = s->args.delay_num_filters;
    s->unit_cfg.fixed_delay_samples = s->args.fixed_delay_samples;
    /* The external fields for the model grid, not the model's metadata (see mono_aec_dfn_res). */
    if (dfn2_model_io_descriptor_default(&s->args.model_grid, &s->descriptor) != 0) return 1;

    if (dfn_plan(s) != 0) {
        fprintf(stderr, "%s: pool query rejected the configuration "
                "(host %d Hz, model grid %d Hz / FFT %d, mode=%s n=%d fixed_delay=%d)\n", name,
                s->args.sample_rate, s->args.model_grid.sample_rate, s->args.model_grid.n_fft,
                example_cli_delay_mode_name(s->args.delay_mode),
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
    printf("%s: model grid %d Hz / FFT %d (n_erb %d, df_bins %d, lookahead %d/%d), %s\n", name,
           s->args.model_grid.sample_rate, s->args.model_grid.n_fft, s->args.model_grid.n_erb,
           s->args.model_grid.df_bins, s->args.model_grid.mask_lookahead,
           s->args.model_grid.df_lookahead,
           s->dfn.use_bridge ? "through the rate bridge" : "the model stream on the host grid");

    if (dfn_build(s) != 0) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        dfn_cleanup(s);
        return 1;
    }
    hop_size = s->grid.hop_size;
    /* Core WOLA hop plus the bridge's added delay, or the model's lookahead hops plus the
     * core hop on the direct path. */
    DFN2_REQUIRE_VARIANT_LINK();   /* this unit's DFN2 variant is the archive's */
    latency = s->dfn.use_bridge
        ? hop_size + ainr_bridge_added_delay_samples(s->hop.bridge)
        : hop_size * (dfn_hop_lookahead_frames(&s->args.model_grid) + 1);
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name, hop_size, latency);
    view->hop=hop_size; view->output=output; view->check_finite=1;
    return 0;
}
static int dfn_read(void *user, int frame) {
    (void)user; return frame < 8;
}
/* P = B * G_res, plus comfort noise on the bins the gain cut when enabled (no denoiser: unity extra
 * gain); a lane's post-chain reset first clears the synthesis that receives the first post-reset frame: the
 * bridge's host-grid tail now on the bridge path, the hop's own AcIstft dfn_hop_lookahead_frames() hops
 * later on the direct path. Before the hop enters the DFN hop. 0, or -1 on a refusal. */
static int dfn_form(DfnSample *s, const AecResidualView *f) {
    uint32_t moved=(uint32_t)array_sample_post_reset_due(&s->reset_handled,f);
    if (s->hop.bridge) {
        if (moved) (void)ainr_bridge_reset_synthesis(s->hop.bridge);   /* between hops: cannot refuse */
    } else if (s->hop.istft) {
        s->reset_pending=(s->reset_pending<<1)|moved;
        if ((s->reset_pending>>s->reset_delay)&1u) ac_istft_reset(s->hop.istft);
    }
    if (aec_apply_gain(s->aec,NULL,s->post) != AEC_OK) return -1;
    if (s->args.enable_cng && aec_apply_cng(s->aec,NULL,s->post) != AEC_OK) return -1;
    return 0;
}

static int dfn_process(void *user, int frame) {
    DfnSample *s=user; (void)frame;
    ArraySampleBeamResult beam;
    AecResidualView spec;
    if (array_sample_beam_process(s->unit,microphones,far,&beam,NULL) != AEC_OK ||
        array_sample_beam_fuse(s->unit,&spec) != AEC_OK || dfn_form(s,&spec) != 0) {
        dfn_reset(s);
        return 1;
    }
    if (dfn_hop_process(&s->hop,spec.error_spec,s->post,output) != 0) {
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
    static const ArraySampleOps ops={dfn_parse,dfn_setup,dfn_read,dfn_process,
                                  NULL,NULL,NULL,dfn_cleanup,dfn_complete};
    return array_sample_main(argc,argv,&ops,&sample);
}
