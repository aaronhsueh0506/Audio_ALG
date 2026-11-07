/* mono_alignulcnet board skeleton: the mono linear AEC feeding the Align-ULCNet post-filter
 * stage, composed here from the library calls. Example code: lib/aec's Aec
 * for the linear stage and ulcnet/ulcnet_nn_stage.h for the framing, the model call policy and
 * the WOLA. Per hop:
 *
 *     aec_process_context(aec, mic, ref);
 *     aec_get_res_context / aec_get_linear_context      // formed hop, alignment state
 *     ulcnet_nn_stage_begin_time(stage, ctx.formed_hop, ref, delay);
 *     if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)   // or ulcnet_nn_stage_process()
 *         ulcnet_nn_stage_nn_end(stage, run_network(&job));
 *     ulcnet_nn_stage_end(stage, out);
 *
 * Grid: --sample-rate 16000 (FFT 512, default) or 48000 (FFT 1024); the UlcnetGrid built from it
 * sizes the lane, the stage and the adapter. The model gets the caller's raw `ref` hop. Replace
 * run_accelerator() with the board runtime call.
 * Data domain: the stage reads the AEC's analysis seam (ctx.formed_hop), the signal the model was trained
 * on, not the rendering base of aec_get_output_base() (which does not pair with formed_hop on a hop where
 * the output guard fires). There is no comfort noise and no synthesis besides the stage's own WOLA, so
 * aec_post_reset_serial() has nothing to clear or reseed here; the stage's alignment policy is the
 * delay.changed flag below (an in-call hard reset of the AEC is also a delay.changed hop). */
#include "aec.h"
#include "common/example_cli.h"
#include "ulcnet/ulcnet_nn_stage.h"
#include "ulcnet_accelerator_adapter.h"
#include "utility/mem_align.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Delay depth D of the grid table below for this smoke example (override with
 * -DULCNET_EXAMPLE_DELAY_FRAMES=N). It must equal the D the model was exported with: the
 * descriptor validator only bounds-checks the range, so a mismatch is silent. The shipped
 * checkpoints are not all D=8. */
#ifndef ULCNET_EXAMPLE_DELAY_FRAMES
#define ULCNET_EXAMPLE_DELAY_FRAMES 8
#endif
_Static_assert(ULCNET_EXAMPLE_DELAY_FRAMES >= ULCNET_MODEL_IO_MIN_D &&
               ULCNET_EXAMPLE_DELAY_FRAMES <= ULCNET_MODEL_IO_MAX_D,
               "ULCNET_EXAMPLE_DELAY_FRAMES outside the descriptor's range");

/* The shipped grids: one row per sample rate. */
static const UlcnetGridSpec k_grids[] = {
    { .sample_rate = 16000, .n_fft = 512,  .delay_depth = ULCNET_EXAMPLE_DELAY_FRAMES },
    { .sample_rate = 48000, .n_fft = 1024, .delay_depth = ULCNET_EXAMPLE_DELAY_FRAMES },
};
enum { K_GRIDS = (int)(sizeof k_grids / sizeof k_grids[0]) };

/* Delay profile is a deployment decision: `n` (matched-filter bank size) sets the bulk-delay search range and comes from the command line (--delay-mode / --delay-num-filters / --fixed-delay, as lib/aec's aec_wav); the resolved profile is printed. Reliable range per bank: n=1 ~125 ms, 2 ~221 ms, 3 ~317 ms, 4 ~413 ms, 5 ~509 ms. DA_NUM_FILTERS is the bank-size cap and the value non-MATCHED modes require. */

/* MATCHED bank size n for this example, overridable with -DULCNET_EXAMPLE_DELAY_NUM_FILTERS=N; --delay-num-filters wins. MATCHED only; changing it needs a rebuild and a fresh pool query. */
#ifndef ULCNET_EXAMPLE_DELAY_NUM_FILTERS
#define ULCNET_EXAMPLE_DELAY_NUM_FILTERS DA_NUM_FILTERS
#endif
_Static_assert(ULCNET_EXAMPLE_DELAY_NUM_FILTERS >= 1 &&
               ULCNET_EXAMPLE_DELAY_NUM_FILTERS <= DA_NUM_FILTERS,
               "ULCNET_EXAMPLE_DELAY_NUM_FILTERS outside lib/aec's bank range");

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --sample-rate {16000|48000} grid of the stage and the lane (default 16000)\n"
        "  --delay-mode {matched|fixed|external}\n"
        "                              far alignment policy (default: matched)\n"
        "  --delay-num-filters <1..5>  MATCHED matched-filter bank size\n"
        "                              (default %d). Reliable bulk-delay\n"
        "                              search: 1~125ms 2~221ms 3~317ms\n"
        "                              4~413ms 5~509ms. Not a runtime knob:\n"
        "                              changing it re-queries the pool.\n"
        "  --fixed-delay <samples>     FIXED delay in native-rate samples (>=0)\n"
        "  -h, --help                  this message\n",
        prog, ULCNET_EXAMPLE_DELAY_NUM_FILTERS);
}

/* The row of k_grids for `rate`, or -1. */
static int grid_row(int rate) {
    int row;
    for (row = 0; row < K_GRIDS; ++row)
        if (k_grids[row].sample_rate == rate) return row;
    return -1;
}

/* Returns 0 on success, 1 on --help, 2 on a rejected profile or rate. A rejection names the
 * requested value and what this build accepts. */
static int parse_options(int argc, char** argv, ExampleCliDelayOptions* out, int* grid_index) {
    int i;

    example_cli_delay_options_default(out, ULCNET_EXAMPLE_DELAY_NUM_FILTERS);
    *grid_index = 0;

    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((!strcmp(arg, "-h") || !strcmp(arg, "--help"))) {
            usage(argv[0]);
            return 1;
        } else if (!strcmp(arg, "--sample-rate") && i + 1 < argc) {
            int rate;
            ++i;
            if (example_cli_parse_int(argv[i], &rate) != 0 || grid_row(rate) < 0) {
                fprintf(stderr, "mono_alignulcnet: --sample-rate '%s' is not a shipped grid "
                        "(accepted: 16000|48000)\n", argv[i]);
                return 2;
            }
            *grid_index = grid_row(rate);
        } else if ((!strcmp(arg, "--delay-mode") || !strcmp(arg, "--delay-num-filters") ||
                    !strcmp(arg, "--fixed-delay")) && i + 1 < argc) {
            if (example_cli_delay_set_option(out, arg, argv[++i], "mono_alignulcnet") != 0)
                return 2;
        } else {
            fprintf(stderr, "mono_alignulcnet: unknown argument '%s'\n", arg);
            usage(argv[0]);
            return 2;
        }
    }

    return example_cli_delay_validate(out, "mono_alignulcnet");
}

/* Replace with the board runtime call. `inputs` holds the feature tensors; the four state
 * tensors (key_history, value_history, logit_history, h_gru) live in the accelerator's own
 * buffers, so the state views are NULL here. Bind each *_out of the graph to the buffer of
 * the state input of the same name, write `outputs->output`, and return 0, or -1 when the
 * run failed. The graph shifts every window itself; read every state input before writing any
 * state output. When the accelerator keeps fixed-point state, load quantized zeros at start
 * and whenever ulcnet_accelerator_adapter_state_generation() changes. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* BOARD: invoke the stateless ONNX accelerator. */
}

/* The mono lane on the stage's grid: the linear context only; the stage takes the formed hop. */
static void derive_lane(const ExampleCliDelayOptions* profile, const UlcnetGrid* grid,
                        AecConfig* aec_cfg) {
    aec_config_from_level(aec_cfg, AEC_LEVEL_BALANCED, grid->sample_rate);
    aec_cfg->fft_size = grid->n_fft;
    aec_cfg->delay_mode = profile->mode;
    aec_cfg->delay_num_filters = profile->num_filters;
    aec_cfg->fixed_delay_samples = profile->fixed_samples;
    aec_cfg->enable_delay_est = profile->mode == AEC_DELAY_MATCHED;
    aec_cfg->enable_res = 0;
    aec_cfg->return_res_context = 1;
    aec_cfg->spatial_linear_context = AEC_CONTEXT_LINEAR_ONLY;
}

/* One hop: linear AEC, then the stage with the lane's formed hop, the raw reference and the
 * alignment events (changed = CHANGED, solid = not UNLOCKED). */
static int process_hop(Aec* aec, UlcnetNnStage* stage, const float* mic, const float* ref,
                       float* out) {
    AecResContext rctx;
    AecLinearContext lctx;
    UlcnetNnDelay delay;
    aec_process_context(aec, mic, ref);
    aec_get_res_context(aec, &rctx);
    aec_get_linear_context(aec, &lctx);
    delay.changed = lctx.delay_state == AEC_LINEAR_DELAY_CHANGED;
    delay.solid = lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
    if (ulcnet_nn_stage_begin_time(stage, rctx.formed_hop, ref, delay) != 0) return -1;
    if (ulcnet_nn_stage_process(stage, out) != 0) return -1;
    return 0;
}

int main(int argc, char** argv) {
    AecConfig aec_cfg;
    ExampleCliDelayOptions profile;
    UlcnetGrid grid;
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModelIoDescriptor model_descriptor;
    UlcnetNnStage *stage = NULL;
    Aec *aec = NULL;
    UlcnetModel model;
    void *adapter_pool = NULL;
    void *pool = NULL;
    float *mic = NULL, *far = NULL, *output = NULL;
    size_t adapter_bytes;
    size_t adapter_alignment;
    size_t aec_bytes, stage_bytes, pool_bytes;
    int grid_index;
    int hop;
    int index;
    int rc;
    rc = parse_options(argc, argv, &profile, &grid_index);
    if (rc != 0) return rc == 1 ? 0 : rc;
    if (ulcnet_grid_make(&grid, &k_grids[grid_index]) != ULCNET_GRID_OK) return 1;

    /* Print the resolved profile with the pool the same config costs; query and init
     * must agree on one (sample_rate, hop, n). */
    stage_bytes = ulcnet_nn_stage_mem_size(&grid, 1);
    derive_lane(&profile, &grid, &aec_cfg);
    aec_bytes = aec_get_mem_size(&aec_cfg);
    if (aec_bytes == 0 || stage_bytes == 0) {
        fprintf(stderr,
                "mono_alignulcnet: pool query rejected the delay profile "
                "(mode=%s n=%d fixed_delay=%d)\n",
                example_cli_delay_mode_name(profile.mode), profile.num_filters,
                profile.fixed_samples);
        return 1;
    }
    pool_bytes = ALIGN16(aec_bytes) + ALIGN16(stage_bytes);
    printf("mono_alignulcnet: grid %d Hz / fft %d / hop %d\n", grid.sample_rate, grid.n_fft,
           grid.hop);
    printf("mono_alignulcnet: delay profile mode=%s n=%d fixed_delay=%d "
           "-> pool %llu bytes (align %lu)\n",
           example_cli_delay_mode_name(profile.mode), profile.num_filters,
           profile.fixed_samples, (unsigned long long)pool_bytes, 16ul);

    /* The board loads this descriptor from the exported model metadata. */
    if (ulcnet_model_io_descriptor_default(&grid, &model_descriptor) != 0 ||
        ulcnet_accelerator_adapter_get_mem_size(
            &model_descriptor, &adapter_bytes, &adapter_alignment) != 0 ||
        posix_memalign(&adapter_pool, adapter_alignment, adapter_bytes) != 0) {
        return 1;
    }
    printf("%s: model I/O D=%d -> accelerator adapter pool %zu bytes (align %zu)\n",
           argv[0], grid.delay_depth, adapter_bytes, adapter_alignment);
    /* NULL state: a board's accelerator holds the recurrent state. */
    adapter = ulcnet_accelerator_adapter_init(
        adapter_pool, adapter_bytes, &model_descriptor, NULL,
        run_accelerator, NULL);
    if (!adapter) {
        free(adapter_pool);
        return 1;
    }

    model = ulcnet_accelerator_adapter_model(adapter);

    mic = (float*)calloc((size_t)grid.hop, sizeof(float));
    far = (float*)calloc((size_t)grid.hop, sizeof(float));
    output = (float*)calloc((size_t)grid.hop, sizeof(float));
    if (!mic || !far || !output || posix_memalign(&pool, 16, pool_bytes) != 0) {
        free(mic); free(far); free(output);
        free(adapter_pool);
        return 1;
    }
    aec = aec_init(pool, aec_bytes, &aec_cfg);
    stage = aec ? ulcnet_nn_stage_init((unsigned char*)pool + ALIGN16(aec_bytes), stage_bytes,
                                       &grid, 1)
                : NULL;
    if (!stage || ulcnet_nn_stage_set_model(stage, &model) != 0) {
        fprintf(stderr, "mono_alignulcnet: pipeline init failed\n");
        rc = 1;
    } else {
        /* Host smoke path: a product calls the hop once per grid.hop-sample hop. */
        for (hop = 0; hop < 4 && rc == 0; ++hop) {
            if (process_hop(aec, stage, mic, far, output) != 0) rc = 1;
            for (index = 0; index < grid.hop && rc == 0; ++index)
                if (!isfinite(output[index])) rc = 1;
        }
    }

    if (aec) aec_destroy(aec);
    free(pool);
    free(mic); free(far); free(output);
    free(adapter_pool);
    if (rc != 0) return 1;
    puts("mono_alignulcnet: fail-open board skeleton PASS");
    return 0;
}
