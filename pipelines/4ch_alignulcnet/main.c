/* 4ch_alignulcnet board skeleton: the 4-channel linear AEC + beamformer unit feeding the
 * Align-ULCNet post-filter stage, composed here from the library calls. Example code: the unit
 * (audio_pipeline_4ch.h, core.enable_post = 0) owns the lanes, SRP-PHAT and BF;
 * samples/ulcnet_nn_stage.h owns the framing, the model call policy and the WOLA. Per hop:
 *
 *     audio_pipeline_4ch_process_with_activity(unit, mics, far, vad, vad, NULL, &beam, &info);
 *     ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, {info.delay.changed, .solid});
 *     if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)   // or ulcnet_nn_stage_process()
 *         ulcnet_nn_stage_nn_end(stage, run_network(&job));
 *     ulcnet_nn_stage_end(stage, out);
 *
 * The model gets the caller's raw `far` hop, never the unit's aligned copy. Any refused call
 * takes the unit and the stage down together (audio_pipeline_4ch_reset + ulcnet_nn_stage_reset).
 * Grid: compile-time (ULCNET_* constants), 16 kHz / FFT 512 or 48 kHz / FFT 1024. */
#include "audio_pipeline_4ch.h"
#include "mcab_sample_driver.h"
#include "ulcnet_nn_stage.h"
#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Delay depth D for this smoke example only. D is a runtime descriptor field; this
 * macro is the example's compile-time default (override with
 * -DULCNET_EXAMPLE_DELAY_FRAMES=N). It must equal the D the model was exported
 * with: the descriptor validator only bounds-checks the range, so a mismatch is
 * silent. The shipped checkpoints are not all D=8. */
#ifndef ULCNET_EXAMPLE_DELAY_FRAMES
#define ULCNET_EXAMPLE_DELAY_FRAMES 8
#endif
_Static_assert(ULCNET_EXAMPLE_DELAY_FRAMES >= ULCNET_MODEL_IO_MIN_D &&
               ULCNET_EXAMPLE_DELAY_FRAMES <= ULCNET_MODEL_IO_MAX_D,
               "ULCNET_EXAMPLE_DELAY_FRAMES outside the descriptor's range");

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
        "  --delay-mode {matched|fixed|external}\n"
        "                              far alignment policy (default: matched)\n"
        "  --delay-num-filters <1..5>  MATCHED matched-filter bank size for the\n"
        "                              ONE shared estimator (default %d).\n"
        "                              Reliable bulk-delay search: 1~125ms\n"
        "                              2~221ms 3~317ms 4~413ms 5~509ms. Not a\n"
        "                              runtime knob: changing it re-queries\n"
        "                              the pool.\n"
        "  --fixed-delay <samples>     FIXED delay in native-rate samples (>=0)\n"
        "  -h, --help                  this message\n",
        prog, ULCNET_EXAMPLE_DELAY_NUM_FILTERS);
}

/* Replace with the product's stateless accelerator invocation: bind each *_out of
 * the graph to the address of the state input of the same name (the pointers in
 * `outputs`), write `outputs->output`, read every state input before writing any
 * state output, and return 0 (or -1 on a failed run). See ulcnet_model_io.h. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* BOARD: write every output tensor, then return 0. */
}

/* The unit on the compiled grid, linear context only (the stage takes the beamformed
 * spectrum); the stage takes ULCNET_BINS bins per hop, so the grids must agree. */
static AudioPipeline4ChConfig ulcnet_example_unit_config(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(ULCNET_SR);
    cfg.core.fft_size = ULCNET_N_FFT;
    cfg.core.enable_post = 0;
    return cfg;
}

typedef struct UlcSample {
    AudioPipeline4ChConfig cfg;
    McabDelayOptions profile;
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModelIoDescriptor model_descriptor;
    AudioPipeline4Ch *unit;
    UlcnetNnStage *stage;
    UlcnetModel model;
    void *adapter_pool, *unit_pool, *stage_pool;
    size_t adapter_bytes,adapter_alignment,stage_bytes;
    AudioPipeline4ChMemReq req;
    float microphones[ULCNET_HOP*4],far[ULCNET_HOP],output[ULCNET_HOP];
    const char *program;
} UlcSample;
static int ulc_unknown(void *user, const char *arg) {
    UlcSample *s=user;
    fprintf(stderr,"4ch_alignulcnet: unknown argument '%s'\n",arg);
    usage(s->program); return 2;
}
static int ulc_set(void *user, const char *arg, const char *value) {
    UlcSample *s=user;
    if (!strcmp(arg,"-h") || !strcmp(arg,"--help")) {
        usage(s->program); return 1;
    }
    return mcab_set_delay_option(&s->profile,arg,value,0,"4ch_alignulcnet");
}
static int ulc_parse(void *user, int argc, char **argv) {
    UlcSample *s=user;
    static const McabCliOption options[]={
        {"-h",0},{"--help",0},{"--delay-mode",1},
        {"--delay-num-filters",1},{"--fixed-delay",1}
    };
    s->program=argv[0]; s->profile.mode=AEC_DELAY_MATCHED;
    s->profile.num_filters=ULCNET_EXAMPLE_DELAY_NUM_FILTERS;
    s->profile.fixed_samples=-1;
    int rc=mcab_scan_cli(argc,argv,options,sizeof(options)/sizeof(options[0]),
                          s,ulc_set,ulc_unknown);
    return rc != 0 ? rc : mcab_validate_delay_options(&s->profile,"4ch_alignulcnet");
}
static int ulc_setup(void *user, McabHopView *view) {
    UlcSample *s=user;
    s->cfg = ulcnet_example_unit_config();
    s->cfg.core.delay_mode = s->profile.mode;
    s->cfg.core.delay_num_filters = s->profile.num_filters;
    s->cfg.core.fixed_delay_samples = s->profile.fixed_samples;

    /* Print the resolved profile with the pools the same config costs; query and init
     * must agree on one (sample_rate, hop, n). */
    s->stage_bytes = ulcnet_nn_stage_mem_size(0);
    if (audio_pipeline_4ch_get_mem_requirements(&s->cfg, &s->req) != 0 ||
        s->stage_bytes == 0) {
        fprintf(stderr,
                "4ch_alignulcnet: pool query rejected the delay profile "
                "(mode=%s n=%d fixed_delay=%d)\n",
                mcab_delay_mode_name(s->profile.mode), s->profile.num_filters,
                s->profile.fixed_samples);
        return 1;
    }
    printf("4ch_alignulcnet: delay profile mode=%s n=%d fixed_delay=%d "
           "(one shared estimator) -> unit pool %llu bytes (align %lu) + stage pool %zu bytes\n",
           mcab_delay_mode_name(s->profile.mode), s->profile.num_filters,
           s->profile.fixed_samples, (unsigned long long)s->req.bytes,
           (unsigned long)s->req.alignment, s->stage_bytes);

    /* The board loads this descriptor from the exported model metadata. */
    if (ulcnet_model_io_descriptor_default(
            ULCNET_EXAMPLE_DELAY_FRAMES, &s->model_descriptor) != 0 ||
        ulcnet_accelerator_adapter_get_mem_size(
            &s->model_descriptor, &s->adapter_bytes, &s->adapter_alignment) != 0 ||
        posix_memalign(&s->adapter_pool, s->adapter_alignment, s->adapter_bytes) != 0) {
        return 1;
    }
    printf("%s: model I/O D=%d -> accelerator adapter pool %zu bytes (align %zu)\n",
           view->program, ULCNET_EXAMPLE_DELAY_FRAMES, s->adapter_bytes,
           s->adapter_alignment);
    s->adapter = ulcnet_accelerator_adapter_init(
        s->adapter_pool, s->adapter_bytes, &s->model_descriptor,
        run_accelerator, NULL);
    if (!s->adapter) return 1;

    if (posix_memalign(&s->unit_pool, s->req.alignment, (size_t)s->req.bytes) != 0)
        s->unit_pool = NULL;
    if (posix_memalign(&s->stage_pool, 16, s->stage_bytes) != 0)
        s->stage_pool = NULL;
    s->unit = s->unit_pool
        ? audio_pipeline_4ch_init_ex(s->unit_pool, (size_t)s->req.bytes, &s->cfg, &s->req)
        : NULL;
    s->stage = s->stage_pool
        ? ulcnet_nn_stage_init(s->stage_pool, s->stage_bytes, 0) : NULL;
    if (!s->unit || !s->stage ||
        audio_pipeline_4ch_hop_size(s->unit) != ULCNET_HOP ||
        audio_pipeline_4ch_n_freqs(s->unit) != ULCNET_BINS) {
        return 1;
    }

    s->model = ulcnet_accelerator_adapter_model(s->adapter);
    if (ulcnet_nn_stage_set_model(s->stage, &s->model) != 0) {
        fprintf(stderr, "4ch_alignulcnet: model install failed\n");
        return 1;
    }

    view->hop=ULCNET_HOP; view->microphones=s->microphones;
    view->reference=s->far; view->output=s->output; view->check_finite=1;
    return 0;
}
static int ulc_read(void *user, McabHopView *view, int frame) {
    (void)user; (void)view; return frame < 4;
}

/* One hop: the unit, then the stage with its beamformed spectrum, the raw far and the
 * alignment events. */
static int ulc_hop(UlcSample *s) {
    AudioPipeline4ChBeam beam;
    AudioPipeline4ChFrameInfo info;
    UlcnetNnDelay delay;
    if (audio_pipeline_4ch_process_with_activity(s->unit, s->microphones, s->far, 0, 0, NULL,
                                                  &beam, &info) != FOUR_AEC_NR_RES_OK)
        return -1;
    delay.changed = info.delay.changed;
    delay.solid = info.delay.solid;
    if (ulcnet_nn_stage_begin_complex(s->stage, beam.spectrum, s->far, delay) != 0)
        return -1;
    return ulcnet_nn_stage_process(s->stage, s->output) == 0 ? 0 : -1;
}
static int ulc_process(void *user, McabHopView *view, int frame) {
    UlcSample *s=user; (void)view; (void)frame;
    if (ulc_hop(s) == 0) return 0;
    audio_pipeline_4ch_reset(s->unit);
    ulcnet_nn_stage_reset(s->stage);
    return 1;
}
static void ulc_complete(void *user) {
    (void)user;
    puts("4ch_alignulcnet: fail-open board skeleton PASS");
}
static void ulc_cleanup(void *user) {
    UlcSample *s=user;
    audio_pipeline_4ch_destroy(s->unit);
    free(s->unit_pool); free(s->stage_pool); free(s->adapter_pool);
}
int main(int argc, char **argv) {
    UlcSample sample={0};
    static const McabSampleOps ops={ulc_parse,2,ulc_setup,ulc_read,ulc_process,
                                  NULL,NULL,NULL,ulc_cleanup,ulc_complete};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
