#include "audio_pipeline_4ch_ulcnet.h"
#include "ulcnet_accelerator_adapter.h"
#include "mcab_sample_driver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Delay depth D for THIS standalone smoke example only.
 *
 * D is a runtime descriptor field, not a library constant: the C chain serves
 * whatever depth the attached model publishes. This macro is only the example
 * binary's compile-time default, overridable with
 * -DULCNET_EXAMPLE_DELAY_FRAMES=N at build time. It is NOT a switch the built
 * binary can flip -- changing D means rebuilding.
 *
 * It MUST equal the D the model in use was exported with. Nothing checks that
 * for you: the descriptor validator only bounds-checks the range, so a
 * mismatch between this value and the graph is silent and corrupting (see
 * ulcnet_process.h's I/O contract). The shipped checkpoints are not all D=8;
 * set this to match the model you are actually running. */
#ifndef ULCNET_EXAMPLE_DELAY_FRAMES
#define ULCNET_EXAMPLE_DELAY_FRAMES 8
#endif
_Static_assert(ULCNET_EXAMPLE_DELAY_FRAMES >= ULCNET_MODEL_IO_MIN_D &&
               ULCNET_EXAMPLE_DELAY_FRAMES <= ULCNET_MODEL_IO_MAX_D,
               "ULCNET_EXAMPLE_DELAY_FRAMES outside the descriptor's range");

/* The delay profile is a product deployment decision, not a property of this
 * source file: `n` (matched-filter bank size) sets how far the estimator can
 * search for the bulk far-to-mic delay, and it has to be chosen from the
 * measured delay distribution of the SKU/route this binary is deployed on.
 * It is therefore read from the command line here -- the same
 * --delay-mode / --delay-num-filters / --fixed-delay spelling lib/aec's
 * aec_wav example uses -- and the resolved profile is printed, so a board
 * bring-up run states which profile it actually ran instead of leaving it
 * buried in a literal. Changing n means re-querying the pool and re-init;
 * there is no runtime setter.
 *
 * This is a 4-channel pipeline but there is still exactly ONE matched bank:
 * the shared estimator in the core. The four lane AECs run
 * EXTERNAL_ALIGNED off its single aligned reference, so n does not multiply
 * by four and the lane pools do not move with it.
 *
 * Reliable bulk-delay search range per bank (lib/aec's contract value, not a
 * geometric span): n=1 ~125 ms, 2 ~221 ms, 3 ~317 ms, 4 ~413 ms, 5 ~509 ms.
 *
 * lib/aec's DA_NUM_FILTERS is both the bank-size cap and the exact value the
 * non-MATCHED modes require -- aec_validate_config() rejects any other n once
 * the mode is not MATCHED -- so it is used for both roles below. The MATCHED
 * default is a separate decision; see ULCNET_EXAMPLE_DELAY_NUM_FILTERS. */

/* MATCHED bank size n for THIS standalone smoke example only, overridable with
 * -DULCNET_EXAMPLE_DELAY_NUM_FILTERS=N so a per-SKU build can bake in its
 * measured delay profile instead of passing the flag on every run.
 *
 * It is only the DEFAULT: --delay-num-filters still wins. It applies to
 * MATCHED alone, because the other two modes build no bank and lib/aec
 * requires DA_NUM_FILTERS there. Like D, it is NOT a switch the built binary
 * can flip -- n sizes the matched bank, so changing it means a rebuild, a
 * fresh pool query and a re-init. */
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

/* Replace with the product's stateless accelerator invocation: bind each
 * *_out of the graph to the address of the state input of the same name (the
 * pointers in `outputs`), write `outputs->output`, read every state input
 * before writing any state output, and return 0 (or -1 on a failed run). See
 * ulcnet_model_io.h. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* TODO(board): write every output tensor, then return 0. */
}

typedef struct UlcSample {
    AudioPipeline4ChConfig cfg;
    McabDelayOptions profile;
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModelIoDescriptor model_descriptor;
    AudioPipeline4ChUlcnet *pipeline;
    UlcnetModel model;
    void *adapter_pool;
    size_t adapter_bytes,adapter_alignment;
    AudioPipeline4ChUlcnetMemReq req;
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
    s->cfg = audio_pipeline_4ch_ulcnet_default_config();
    s->cfg.core.delay_mode = s->profile.mode;
    s->cfg.core.delay_num_filters = s->profile.num_filters;
    s->cfg.core.fixed_delay_samples = s->profile.fixed_samples;

    /* Print the RESOLVED profile with the pool the SAME config actually
     * costs -- query and init must agree on one (sample_rate, hop, n). */
    if (audio_pipeline_4ch_ulcnet_get_mem_requirements(&s->cfg, &s->req) != 0) {
        fprintf(stderr,
                "4ch_alignulcnet: pool query rejected the delay profile "
                "(mode=%s n=%d fixed_delay=%d)\n",
                mcab_delay_mode_name(s->profile.mode), s->profile.num_filters,
                s->profile.fixed_samples);
        return 1;
    }
    printf("4ch_alignulcnet: delay profile mode=%s n=%d fixed_delay=%d "
           "(one shared estimator) -> pool %llu bytes (align %lu)\n",
           mcab_delay_mode_name(s->profile.mode), s->profile.num_filters,
           s->profile.fixed_samples, (unsigned long long)s->req.bytes,
           (unsigned long)s->req.alignment);

    /* The board loads this descriptor from the exported model metadata;
     * see ULCNET_EXAMPLE_DELAY_FRAMES at the top of this file. */
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
    s->pipeline = audio_pipeline_4ch_ulcnet_create(&s->cfg);
    if (!s->adapter || !s->pipeline) {
        return 1;
    }

    s->model = ulcnet_accelerator_adapter_model(s->adapter);
    if (audio_pipeline_4ch_ulcnet_set_model(s->pipeline, &s->model) != 0) {
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
static int ulc_process(void *user, McabHopView *view, int frame) {
    UlcSample *s=user; (void)view; (void)frame;
    return audio_pipeline_4ch_ulcnet_process_with_activity(s->pipeline,
        s->microphones,s->far,0,s->output) != 0;
}
static int ulc_finish(void *user, McabHopView *view, int frames) {
    (void)user; (void)view; (void)frames;
    return 0;
}
static void ulc_complete(void *user) {
    (void)user;
    puts("4ch_alignulcnet: fail-open board skeleton PASS");
}
static void ulc_cleanup(void *user) {
    UlcSample *s=user;
    audio_pipeline_4ch_ulcnet_destroy(s->pipeline); free(s->adapter_pool);
}
int main(int argc, char **argv) {
    UlcSample sample={0};
    static const McabSampleOps ops={ulc_parse,2,ulc_setup,ulc_read,ulc_process,
                                  NULL,NULL,ulc_finish,ulc_cleanup,ulc_complete};
    return mcab_sample_main(argc,argv,&ops,&sample);
}
