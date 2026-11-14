/* mono_alignulcnet board skeleton: the mono linear AEC feeding the Align-ULCNet post-filter
 * stage, composed here from the library calls. Example code: lib/aec's Aec
 * for the linear stage and ulcnet/ulcnet_nn_stage.h for the model call policy; the framing (an AcStft
 * for the formed hop, one for the raw reference, an AcIstft for the output spectrum) is this file's.
 * Per hop:
 *
 *     aec_process_linear(aec, mic, ref);  aec_process_residual(aec, NULL);
 *     aec_get_residual / aec_get_linear                 // formed hop, alignment state
 *     ac_stft_push(err_stft, rctx.formed_hop, err_spec);  ac_stft_push(far_stft, ref, far_spec);
 *     ulcnet_nn_stage_begin_spec(stage, err_spec, far_spec, delay);
 *     if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)   // or ulcnet_nn_stage_process()
 *         ulcnet_nn_stage_nn_end(stage, run_network(&job));
 *     ulcnet_nn_stage_end_spec(stage, out_spec);
 *     ac_istft_push(istft, out_spec, out);               // one hop
 *
 * Grid: --sample-rate 16000 (FFT 512, default) or 48000 (FFT 1024); the UlcnetGrid built from it
 * sizes the lane, the stage and the adapter. The model gets the caller's raw `ref` hop. Replace
 * run_accelerator() with the board runtime call.
 * Data domain: the stage reads the AEC's analysis seam (rctx.formed_hop), the signal the model was trained
 * on, not the rendering base_spec (which does not pair with formed_hop on a hop where the output guard
 * fires). There is no comfort noise and no synthesis besides this file's AcIstft, so
 * aec_post_reset_serial() has nothing to clear or reseed here; the stage's alignment policy is the
 * delay.changed flag below (an in-call hard reset of the AEC is also a delay.changed hop). A refused
 * call resets the AEC, the stage and the three transform objects together. */
#include "aec.h"
#include "audio_stft.h"
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

/* Delay profile is a deployment decision: `n` (matched-filter bank size) sets the bulk-delay search range and comes from the command line (--delay-mode / --delay-num-filters / --fixed-delay, as lib/aec's aec_wav); the resolved profile is printed. Reliable range per bank: n=1 ~125 ms, 2 ~221 ms, 3 ~317 ms, 4 ~413 ms, 5 ~509 ms. AEC_DELAY_NUM_FILTERS_MAX is the bank-size cap and the value non-MATCHED modes require. */

/* MATCHED bank size n for this example, overridable with -DULCNET_EXAMPLE_DELAY_NUM_FILTERS=N; --delay-num-filters wins. MATCHED only; changing it needs a rebuild and a fresh pool query. */
#ifndef ULCNET_EXAMPLE_DELAY_NUM_FILTERS
#define ULCNET_EXAMPLE_DELAY_NUM_FILTERS AEC_DELAY_NUM_FILTERS_MAX
#endif
_Static_assert(ULCNET_EXAMPLE_DELAY_NUM_FILTERS >= 1 &&
               ULCNET_EXAMPLE_DELAY_NUM_FILTERS <= AEC_DELAY_NUM_FILTERS_MAX,
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

/* The mono lane on the stage's grid: the state of the linear residual only (AEC_RESIDUAL_STATE); the stage
 * takes the formed hop. */
static void derive_lane(const ExampleCliDelayOptions* profile, const UlcnetGrid* grid,
                        AecConfig* aec_cfg) {
    aec_config_from_level(aec_cfg, AEC_LEVEL_BALANCED, grid->sample_rate);
    aec_cfg->fft_size = grid->n_fft;
    aec_cfg->delay_mode = profile->mode;
    aec_cfg->delay_num_filters = profile->num_filters;
    aec_cfg->fixed_delay_samples = profile->fixed_samples;
    aec_cfg->residual = AEC_RESIDUAL_STATE;
}

/* The application's framing of the stage (audiolib defaults on the grid: sqrt-Hann, centre 0, one
 * frame per hop; the synthesis accumulates). */
typedef struct Framing {
    AcStft *far_stft, *err_stft;
    AcIstft* istft;
    Complex *far_spec, *err_spec, *out_spec;   /* grid.n_bins each */
} Framing;

static void framing_config(AcStftConfig* sc, const UlcnetGrid* grid) {
    ac_stft_config_default(sc, grid->n_fft);
    sc->ola = AC_OLA_ACCUMULATE;
}

/* Two analyses, one synthesis and three spectra, each in its own 16-byte slot. */
static size_t framing_bytes(const UlcnetGrid* grid) {
    AcStftConfig sc;
    framing_config(&sc, grid);
    return 2 * ALIGN16(ac_stft_get_mem_size(&sc)) + ALIGN16(ac_istft_get_mem_size(&sc)) +
           3 * ALIGN16((size_t)grid->n_bins * sizeof(Complex));
}

static int framing_init(Framing* f, void* mem, size_t bytes, const UlcnetGrid* grid) {
    AcStftConfig sc;
    unsigned char* cur = (unsigned char*)mem;
    size_t stft_b, istft_b, spec_b = ALIGN16((size_t)grid->n_bins * sizeof(Complex));
    if (!mem || bytes < framing_bytes(grid)) return -1;
    framing_config(&sc, grid);
    stft_b = ac_stft_get_mem_size(&sc);
    istft_b = ac_istft_get_mem_size(&sc);
    f->far_stft = ac_stft_init(cur, stft_b, &sc);   cur += ALIGN16(stft_b);
    f->err_stft = ac_stft_init(cur, stft_b, &sc);   cur += ALIGN16(stft_b);
    f->istft = ac_istft_init(cur, istft_b, &sc);    cur += ALIGN16(istft_b);
    f->far_spec = (Complex*)cur;                    cur += spec_b;
    f->err_spec = (Complex*)cur;                    cur += spec_b;
    f->out_spec = (Complex*)cur;
    return f->far_stft && f->err_stft && f->istft ? 0 : -1;
}

/* The stage and the transforms restart together. */
static void reset_pipeline(Aec* aec, UlcnetNnStage* stage, Framing* f) {
    aec_reset(aec);
    ulcnet_nn_stage_reset(stage);
    ac_stft_reset(f->far_stft);
    ac_stft_reset(f->err_stft);
    ac_istft_reset(f->istft);
}

/* One hop: linear AEC, then the framing of the lane's formed hop and of the raw reference, the stage
 * with the alignment events (changed = CHANGED, solid = not UNLOCKED), and the synthesis. */
static int process_hop(Aec* aec, UlcnetNnStage* stage, Framing* f, const float* mic, const float* ref,
                       float* out) {
    AecResidualView rctx;
    AecLinearView lctx;
    UlcnetNnDelay delay;
    if (aec_process_linear(aec, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(aec, NULL) != AEC_OK) return -1;
    memset(&rctx, 0, sizeof(rctx));
    rctx.struct_size = sizeof(rctx);
    memset(&lctx, 0, sizeof(lctx));
    lctx.struct_size = sizeof(lctx);
    if (aec_get_residual(aec, &rctx) != AEC_OK || aec_get_linear(aec, &lctx) != AEC_OK) return -1;
    delay.changed = lctx.delay_state == AEC_LINEAR_DELAY_CHANGED;
    delay.solid = lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
    (void)ac_stft_push(f->err_stft, rctx.formed_hop, f->err_spec);
    (void)ac_stft_push(f->far_stft, ref, f->far_spec);
    if (ulcnet_nn_stage_begin_spec(stage, f->err_spec, f->far_spec, delay) != 0) return -1;
    if (ulcnet_nn_stage_process(stage, f->out_spec) != 0) return -1;
    return ac_istft_push(f->istft, f->out_spec, out) == ulcnet_nn_stage_grid(stage)->hop ? 0 : -1;
}

int main(int argc, char** argv) {
    AecConfig aec_cfg;
    ExampleCliDelayOptions profile;
    UlcnetGrid grid;
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModelIoDescriptor model_descriptor;
    UlcnetNnStage *stage = NULL;
    Framing framing;
    Aec *aec = NULL;
    UlcnetModel model;
    void *adapter_pool = NULL;
    void *pool = NULL;
    float *mic = NULL, *far = NULL, *output = NULL;
    size_t adapter_bytes;
    size_t adapter_alignment;
    size_t aec_bytes, stage_bytes, framing_pool_bytes, pool_bytes;
    int grid_index;
    int hop;
    int index;
    int rc;
    rc = parse_options(argc, argv, &profile, &grid_index);
    if (rc != 0) return rc == 1 ? 0 : rc;
    if (ulcnet_grid_make(&grid, &k_grids[grid_index]) != ULCNET_GRID_OK) return 1;

    /* Print the resolved profile with the pool the same config costs; query and init
     * must agree on one (sample_rate, hop, n). */
    stage_bytes = ulcnet_nn_stage_mem_size(&grid);
    framing_pool_bytes = framing_bytes(&grid);
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
    pool_bytes = ALIGN16(aec_bytes) + ALIGN16(stage_bytes) + ALIGN16(framing_pool_bytes);
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
    stage = aec ? ulcnet_nn_stage_init((unsigned char*)pool + ALIGN16(aec_bytes), stage_bytes, &grid)
                : NULL;
    if (!stage || ulcnet_nn_stage_set_model(stage, &model) != 0 ||
        framing_init(&framing, (unsigned char*)pool + ALIGN16(aec_bytes) + ALIGN16(stage_bytes),
                     framing_pool_bytes, &grid) != 0) {
        fprintf(stderr, "mono_alignulcnet: pipeline init failed\n");
        rc = 1;
    } else {
        /* Host smoke path: a product calls the hop once per grid.hop-sample hop. */
        for (hop = 0; hop < 4 && rc == 0; ++hop) {
            if (process_hop(aec, stage, &framing, mic, far, output) != 0) {
                reset_pipeline(aec, stage, &framing);   /* the AEC, the stage and the transforms restart together */
                rc = 1;
            }
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
