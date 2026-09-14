/* 4ch_aec_bf_dfn_res board skeleton: builds the wrapper from the
 * command-line delay profile and ERB matrices, installs a fail-open model
 * callback, and runs a few silent hops through the pre/post seam with
 * uniform beamformer weights standing in for the product's BF/GSC. Replace
 * run_accelerator() with the board runtime call and the weights with the
 * beamformer's effective response. */
#include "4aec_dfn_res.h"
#include "dfn_example_args.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static Complex g_weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];

/* Replace this body with the board runtime call (see mono_aec_dfn_res). */
static int run_accelerator(void *user, const DFN2PrepostInputs *inputs,
                           DFN2PrepostOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* TODO(board): invoke the stateless DFN2 graph. */
}

/* Host grids this skeleton accepts at parse time (0-terminated): the core's own grids (no 8 kHz). */
static const int HOST_RATES[] = { 16000, 48000, 0 };

int main(int argc, char **argv) {
    static const char *name = "4ch_aec_bf_dfn_res";
    DfnExampleArgs args;
    FourAecDfnResConfig cfg;
    FourAecDfnResMemReq req;
    DFN2ModelIoDescriptor descriptor;
    FourAecDfnRes *pipeline;
    static float microphones[DFN2_HOP_LEN * FOUR_AEC_NR_RES_CHANNELS];
    static float far[DFN2_HOP_LEN];
    static float output[DFN2_HOP_LEN];
    int hop, index, rc, hop_size;

    rc = dfn_example_parse_args(argc, argv, name, HOST_RATES, &args);
    if (rc != 0) return rc == 1 ? 0 : rc;
    if (dfn_example_load_matrices(&args, g_erb_fwd, g_erb_inv, name) != 0)
        return 1;
    for (index = 0; index < FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS; ++index) {
        g_weights[index].r = 1.0f / (float)FOUR_AEC_NR_RES_CHANNELS;
        g_weights[index].i = 0.0f;
    }

    cfg = four_aec_dfn_res_default_config();
    cfg.front_end.sample_rate = args.sample_rate;
    cfg.front_end.fft_size = 0;   /* the core's rate default */
    cfg.front_end.delay_mode = args.delay_mode;
    cfg.front_end.delay_num_filters = args.delay_num_filters;
    cfg.front_end.fixed_delay_samples = args.fixed_delay_samples;
    cfg.front_end.enable_cng = args.enable_cng;
    cfg.atten_lim_db = args.atten_lim_db;
    cfg.erb_fwd = g_erb_fwd;
    cfg.erb_inv = g_erb_inv;
    /* Compiled-in geometry, not the model's metadata: see mono_aec_dfn_res. */
    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    cfg.model.user = NULL;
    cfg.model.infer = run_accelerator;
    cfg.model.reset = NULL;
    cfg.model.io_descriptor = &descriptor;

    if (four_aec_dfn_res_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "%s: pool query rejected the configuration "
                "(host %d Hz, mode=%s n=%d fixed_delay=%d)\n", name,
                args.sample_rate, dfn_example_delay_mode_name(args.delay_mode),
                args.delay_num_filters, args.fixed_delay_samples);
        return 1;
    }
    printf("%s: core %d Hz, delay profile mode=%s n=%d fixed_delay=%d (one "
           "shared estimator) cng=%d atten_lim=%.1f dB -> pool %llu bytes "
           "(align %lu), model weights external; model I/O descriptor = "
           "compiled-in layout v%u\n", name, args.sample_rate,
           dfn_example_delay_mode_name(args.delay_mode),
           args.delay_num_filters, args.fixed_delay_samples, args.enable_cng,
           (double)args.atten_lim_db, (unsigned long long)req.bytes,
           (unsigned long)req.alignment, (unsigned)descriptor.layout_version);

    pipeline = four_aec_dfn_res_create(&cfg);
    if (!pipeline) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        return 1;
    }
    hop_size = four_aec_dfn_res_hop_size(pipeline);
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name,
           hop_size, four_aec_dfn_res_lookahead_samples(pipeline));
    for (hop = 0; hop < 8; ++hop) {
        FourAecNrResPreFrame pre;
        if (four_aec_dfn_res_process_pre(pipeline, microphones, far, &pre) != 0 ||
            four_aec_dfn_res_process_post(pipeline, &pre.token, g_weights,
                                          output) != 0) {
            four_aec_dfn_res_destroy(pipeline);
            return 1;
        }
        for (index = 0; index < hop_size; ++index) {
            if (!isfinite(output[index])) {
                four_aec_dfn_res_destroy(pipeline);
                return 1;
            }
        }
    }
    four_aec_dfn_res_destroy(pipeline);
    printf("%s: fail-open board skeleton PASS\n", name);
    return 0;
}
