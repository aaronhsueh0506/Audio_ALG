/* mono_aec_dfn_res board skeleton: builds the pipeline from the command-line
 * delay profile and ERB matrices, installs a fail-open model callback, and
 * runs a few silent hops. Replace run_accelerator() with the board runtime
 * call; everything else is the product's per-hop loop. */
#include "audio_pipeline_dfn.h"
#include "dfn_example_args.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];

/* Replace this body with the board runtime call. `inputs` carries the
 * feature windows plus the CPU-owned recurrent state; the runtime must fill
 * every tensor in `outputs` and return 0. A nonzero return takes the frame
 * as the exact identity (dfn_res_stage.h, FAIL-OPEN). */
static int run_accelerator(void *user, const DFN2PrepostInputs *inputs,
                           DFN2PrepostOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* TODO(board): invoke the stateless DFN2 graph. */
}

int main(int argc, char **argv) {
    static const char *name = "mono_aec_dfn_res";
    DfnExampleArgs args;
    MonoAecDfnResConfig cfg;
    MonoAecDfnResMemReq req;
    DFN2ModelIoDescriptor descriptor;
    MonoAecDfnRes *pipeline;
    static float mic[DFN2_HOP_LEN];
    static float far[DFN2_HOP_LEN];
    static float output[DFN2_HOP_LEN];
    int hop, index, rc, hop_size;

    rc = dfn_example_parse_args(argc, argv, name, &args);
    if (rc != 0) return rc == 1 ? 0 : rc;
    if (dfn_example_load_matrices(&args, g_erb_fwd, g_erb_inv, name) != 0)
        return 1;

    cfg = mono_aec_dfn_res_default_config();
    cfg.host.sample_rate = args.sample_rate;
    cfg.host.fft_size = 0;   /* the host's rate default */
    cfg.host.delay_mode = args.delay_mode;
    cfg.host.delay_num_filters = args.delay_num_filters;
    cfg.host.fixed_delay_samples = args.fixed_delay_samples;
    cfg.host.enable_cng = args.enable_cng;
    cfg.atten_lim_db = args.atten_lim_db;
    cfg.erb_fwd = g_erb_fwd;
    cfg.erb_inv = g_erb_inv;
    /* dfn2_model_io_descriptor_default() publishes the geometry THIS build
     * was compiled against; it reads nothing from the model, so a graph
     * exported for another geometry is NOT detected here. A product fills
     * the descriptor from its model's exported metadata (export_onnx.py
     * writes it into the ONNX model properties) and hands that in instead. */
    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    cfg.model.user = NULL;
    cfg.model.infer = run_accelerator;
    cfg.model.reset = NULL;
    cfg.model.io_descriptor = &descriptor;

    /* Print the RESOLVED profile with the pool the SAME config costs: query
     * and init must agree on one (grid, n, matrices). */
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "%s: pool query rejected the configuration "
                "(host %d Hz, mode=%s n=%d fixed_delay=%d)\n", name,
                args.sample_rate, dfn_example_delay_mode_name(args.delay_mode),
                args.delay_num_filters, args.fixed_delay_samples);
        return 1;
    }
    printf("%s: host %d Hz, delay profile mode=%s n=%d fixed_delay=%d cng=%d "
           "atten_lim=%.1f dB -> pool %llu bytes (align %lu), model weights "
           "external; model I/O descriptor = compiled-in layout v%u\n", name,
           args.sample_rate, dfn_example_delay_mode_name(args.delay_mode),
           args.delay_num_filters, args.fixed_delay_samples, args.enable_cng,
           (double)args.atten_lim_db, (unsigned long long)req.bytes,
           (unsigned long)req.alignment, (unsigned)descriptor.layout_version);

    pipeline = mono_aec_dfn_res_create(&cfg);
    if (!pipeline) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        return 1;
    }
    hop_size = mono_aec_dfn_res_hop_size(pipeline);
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name,
           hop_size, mono_aec_dfn_res_lookahead_samples(pipeline));
    /* Host smoke path. A product calls process() once per host hop; the
     * first hops (the latency above) come out silent. */
    for (hop = 0; hop < 8; ++hop) {
        if (mono_aec_dfn_res_process(pipeline, mic, far, output) != 0) {
            mono_aec_dfn_res_destroy(pipeline);
            return 1;
        }
        for (index = 0; index < hop_size; ++index) {
            if (!isfinite(output[index])) {
                mono_aec_dfn_res_destroy(pipeline);
                return 1;
            }
        }
    }
    mono_aec_dfn_res_destroy(pipeline);
    printf("%s: fail-open board skeleton PASS\n", name);
    return 0;
}
