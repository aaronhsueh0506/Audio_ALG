/* 4ch_alignulcnet board skeleton: the 4-channel linear AEC + beamformer unit feeding the
 * Align-ULCNet post-filter stage, composed here from the library calls. Example code: the unit is
 * the four-microphone array instance of lib/aec (AEC_RESIDUAL_STATE: lanes only; one pool placed by
 * aec_get_mem_requirements() + aec_init_ex())
 * with the sample SRP-PHAT/BF helper (array_sample_beam.h) on top;
 * ulcnet/ulcnet_nn_stage.h owns the model call policy; the framing (an AcStft for the raw far hop, an
 * AcIstft for the output spectrum) is this file's. Per hop:
 *
 *     array_sample_beam_process_with_activity(unit, mics, far, vad, vad, NULL, &beam, &info);
 *     ac_stft_push(far_stft, far, far_spec);
 *     ulcnet_nn_stage_begin_spec(stage, beam.spectrum, far_spec, {info.delay.changed, .solid});
 *     ulcnet_nn_stage_process(stage, out_spec);          // the model call, then the stage output
 *     ac_istft_push(istft, out_spec, out);               // one hop
 *
 * The model gets the caller's raw `far` hop, never the unit's aligned copy. Any refused call
 * takes the unit, the stage and the two transform objects down together (array_sample_beam_reset +
 * ulcnet_nn_stage_reset + ac_stft_reset + ac_istft_reset).
 * Grid: --sample-rate 16000 (FFT 512, default) or 48000 (FFT 1024); the UlcnetGrid built from it
 * sizes the unit, the stage and the adapter. */
#include "aec.h"
#include "4ch/array_sample_beam.h"
#include "utility/mem_align.h"
#include "4ch/array_sample_driver.h"
#include "common/example_cli.h"
#include "audio_stft.h"
#include "ulcnet/ulcnet_nn_stage.h"
#include "ulcnet_accelerator_adapter.h"

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
        "  --sample-rate {16000|48000} grid of the stage and the unit (default 16000)\n"
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

/* Replace with the product's stateless accelerator invocation. The four state tensors live in
 * the accelerator's own buffers, so the state views in `inputs` are NULL: bind each *_out of
 * the graph to the buffer of the state input of the same name, write `outputs->output`, read
 * every state input before writing any state output, and return 0 (or -1 on a failed run).
 * See ulcnet_model_io.h. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* BOARD: write every output tensor, then return 0. */
}

/* The unit on the stage's grid, lanes only (the stage takes the beamformed spectrum); the
 * stage takes grid.n_bins bins per hop, so the grids must agree. */
static AecConfig ulcnet_example_unit_config(const UlcnetGrid* grid) {
    AecConfig cfg;
    array_sample_default_config(&cfg, grid->sample_rate);
    cfg.fft_size = grid->n_fft;
    cfg.residual = AEC_RESIDUAL_STATE;
    return cfg;
}

typedef struct UlcSample {
    UlcnetGrid grid;
    int grid_index;
    AecConfig cfg;
    ArraySampleBeamParams bp;
    ExampleCliDelayOptions profile;
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModelIoDescriptor model_descriptor;
    Aec *aec;
    ArraySampleBeam *unit;
    UlcnetNnStage *stage;
    AcStft *far_stft;                /* the application's far analysis and output synthesis */
    AcIstft *istft;
    Complex *far_spec, *out_spec;    /* grid.n_bins each, in frame_pool */
    UlcnetModel model;
    void *adapter_pool, *unit_pool, *stage_pool, *frame_pool;
    size_t adapter_bytes,adapter_alignment,stage_bytes,frame_bytes;
    size_t aec_bytes, beam_bytes;    /* the unit pool: the array, then the helper */
    AecMemReq req;                   /* the array's descriptor: the pool alignment */
    float *microphones, *far, *output;   /* grid.hop frames; the microphones interleaved */
    const char *program;
} UlcSample;
static int ulc_unknown(void *user, const char *arg) {
    UlcSample *s=user;
    fprintf(stderr,"4ch_alignulcnet: unknown argument '%s'\n",arg);
    usage(s->program); return 2;
}
/* The row of k_grids for `rate`, or -1. */
static int grid_row(int rate) {
    int row;
    for (row = 0; row < K_GRIDS; ++row)
        if (k_grids[row].sample_rate == rate) return row;
    return -1;
}
static int ulc_set(void *user, const char *arg, const char *value) {
    UlcSample *s=user;
    if (!strcmp(arg,"-h") || !strcmp(arg,"--help")) {
        usage(s->program); return 1;
    }
    if (!strcmp(arg,"--sample-rate")) {
        int rate;
        if (example_cli_parse_int(value,&rate) != 0 || grid_row(rate) < 0) {
            fprintf(stderr,"4ch_alignulcnet: --sample-rate '%s' is not a shipped grid "
                    "(accepted: 16000|48000)\n",value);
            return 2;
        }
        s->grid_index=grid_row(rate);
        return 0;
    }
    return example_cli_delay_set_option(&s->profile,arg,value,"4ch_alignulcnet");
}
static int ulc_parse(void *user, int argc, char **argv) {
    UlcSample *s=user;
    static const ArraySampleCliOption options[]={
        {"-h",0},{"--help",0},{"--sample-rate",1},{"--delay-mode",1},
        {"--delay-num-filters",1},{"--fixed-delay",1}
    };
    s->program=argv[0];
    s->grid_index=0;
    example_cli_delay_options_default(&s->profile,ULCNET_EXAMPLE_DELAY_NUM_FILTERS);
    int rc=array_sample_scan_cli(argc,argv,options,sizeof(options)/sizeof(options[0]),
                          s,ulc_set,ulc_unknown);
    return rc != 0 ? rc : example_cli_delay_validate(&s->profile,"4ch_alignulcnet");
}
static int ulc_setup(void *user, ArraySampleHopView *view) {
    UlcSample *s=user;
    AcStftConfig sc;
    size_t stft_bytes, istft_bytes, spec_bytes;
    unsigned char *cur;
    if (ulcnet_grid_make(&s->grid, &k_grids[s->grid_index]) != ULCNET_GRID_OK) return 1;
    s->cfg = ulcnet_example_unit_config(&s->grid);
    s->bp = array_sample_beam_default_params();
    s->cfg.delay_mode = s->profile.mode;
    s->cfg.delay_num_filters = s->profile.num_filters;
    s->cfg.fixed_delay_samples = s->profile.fixed_samples;

    /* Print the resolved profile with the pools the same config costs; query and init
     * must agree on one (sample_rate, hop, n). */
    s->stage_bytes = ulcnet_nn_stage_mem_size(&s->grid);
    ac_stft_config_default(&sc, s->grid.n_fft);
    sc.ola = AC_OLA_ACCUMULATE;
    stft_bytes = ac_stft_get_mem_size(&sc);
    istft_bytes = ac_istft_get_mem_size(&sc);
    spec_bytes = ALIGN16((size_t)s->grid.n_bins * sizeof(Complex));
    s->frame_bytes = ALIGN16(stft_bytes) + ALIGN16(istft_bytes) + 2 * spec_bytes;
    if (aec_get_mem_requirements(&s->cfg, &s->req) != 0 ||
        array_sample_beam_get_mem_size(&s->cfg, &s->bp, &s->beam_bytes) != 0 ||
        s->stage_bytes == 0 || stft_bytes == 0 || istft_bytes == 0) {
        fprintf(stderr,
                "4ch_alignulcnet: pool query rejected the delay profile "
                "(mode=%s n=%d fixed_delay=%d)\n",
                example_cli_delay_mode_name(s->profile.mode), s->profile.num_filters,
                s->profile.fixed_samples);
        return 1;
    }
    s->aec_bytes = ALIGN16((size_t)s->req.bytes);
    printf("4ch_alignulcnet: grid %d Hz / fft %d / hop %d\n", s->grid.sample_rate,
           s->grid.n_fft, s->grid.hop);
    printf("4ch_alignulcnet: delay profile mode=%s n=%d fixed_delay=%d "
           "(one shared estimator) -> unit pool %llu bytes (align %lu) + stage pool %zu bytes\n",
           example_cli_delay_mode_name(s->profile.mode), s->profile.num_filters,
           s->profile.fixed_samples,
           (unsigned long long)(s->aec_bytes + s->beam_bytes),
           (unsigned long)s->req.alignment, s->stage_bytes);
    printf("4ch_alignulcnet: application framing (far AcStft, output AcIstft, two spectra) pool %zu bytes\n",
           s->frame_bytes);

    /* The board loads this descriptor from the exported model metadata. */
    if (ulcnet_model_io_descriptor_default(&s->grid, &s->model_descriptor) != 0 ||
        ulcnet_accelerator_adapter_get_mem_size(
            &s->model_descriptor, &s->adapter_bytes, &s->adapter_alignment) != 0 ||
        posix_memalign(&s->adapter_pool, s->adapter_alignment, s->adapter_bytes) != 0) {
        return 1;
    }
    printf("%s: model I/O D=%d -> accelerator adapter pool %zu bytes (align %zu)\n",
           s->program, s->grid.delay_depth, s->adapter_bytes,
           s->adapter_alignment);
    /* NULL state: a board's accelerator holds the recurrent state. */
    s->adapter = ulcnet_accelerator_adapter_init(
        s->adapter_pool, s->adapter_bytes, &s->model_descriptor, NULL,
        run_accelerator, NULL);
    if (!s->adapter) return 1;

    if (posix_memalign(&s->unit_pool, s->req.alignment,
                       s->aec_bytes + s->beam_bytes) != 0)
        s->unit_pool = NULL;
    if (posix_memalign(&s->stage_pool, 16, s->stage_bytes) != 0)
        s->stage_pool = NULL;
    if (posix_memalign(&s->frame_pool, 16, s->frame_bytes) != 0)
        s->frame_pool = NULL;
    if (s->unit_pool) s->aec = aec_init_ex(s->unit_pool, s->aec_bytes, &s->cfg, &s->req);
    s->unit = s->aec
        ? array_sample_beam_init((unsigned char *)s->unit_pool + s->aec_bytes,
                                s->beam_bytes, &s->cfg, &s->bp, s->aec)
        : NULL;
    s->stage = s->stage_pool
        ? ulcnet_nn_stage_init(s->stage_pool, s->stage_bytes, &s->grid) : NULL;
    if (s->frame_pool) {
        cur = (unsigned char *)s->frame_pool;
        s->far_stft = ac_stft_init(cur, stft_bytes, &sc);   cur += ALIGN16(stft_bytes);
        s->istft = ac_istft_init(cur, istft_bytes, &sc);    cur += ALIGN16(istft_bytes);
        s->far_spec = (Complex *)cur;                       cur += spec_bytes;
        s->out_spec = (Complex *)cur;
    }
    if (!s->unit || !s->stage || !s->far_stft || !s->istft ||
        aec_hop_size(s->aec) != s->grid.hop ||
        aec_hop_size(s->aec) + 1 != s->grid.n_bins) {
        return 1;
    }

    s->model = ulcnet_accelerator_adapter_model(s->adapter);
    if (ulcnet_nn_stage_set_model(s->stage, &s->model) != 0) {
        fprintf(stderr, "4ch_alignulcnet: model install failed\n");
        return 1;
    }

    s->microphones = calloc((size_t)s->grid.hop * AEC_ARRAY_MICS, sizeof(float));
    s->far = calloc((size_t)s->grid.hop, sizeof(float));
    s->output = calloc((size_t)s->grid.hop, sizeof(float));
    if (!s->microphones || !s->far || !s->output) return 1;
    view->hop=s->grid.hop; view->output=s->output; view->check_finite=1;
    return 0;
}
static int ulc_read(void *user, int frame) {
    (void)user; return frame < 4;
}

/* One hop: the unit, the application's analysis of the raw far, the stage with the beamformed
 * spectrum, the far spectrum and the alignment events, and the synthesis of its output. */
static int ulc_hop(UlcSample *s) {
    ArraySampleBeamResult beam;
    ArraySampleBeamInfo info;
    UlcnetNnDelay delay;
    if (array_sample_beam_process_with_activity(s->unit, s->microphones, s->far, 0, 0, NULL,
                                               &beam, &info) != AEC_OK)
        return -1;
    delay.changed = info.delay.changed;
    delay.solid = info.delay.solid;
    (void)ac_stft_push(s->far_stft, s->far, s->far_spec);
    if (ulcnet_nn_stage_begin_spec(s->stage, beam.spectrum, s->far_spec, delay) != 0)
        return -1;
    if (ulcnet_nn_stage_process(s->stage, s->out_spec) != 0) return -1;
    return ac_istft_push(s->istft, s->out_spec, s->output) == s->grid.hop ? 0 : -1;
}
static int ulc_process(void *user, int frame) {
    UlcSample *s=user; (void)frame;
    if (ulc_hop(s) == 0) return 0;
    array_sample_beam_reset(s->unit);
    ulcnet_nn_stage_reset(s->stage);
    ac_stft_reset(s->far_stft);
    ac_istft_reset(s->istft);
    return 1;
}
static void ulc_complete(void *user) {
    (void)user;
    puts("4ch_alignulcnet: fail-open board skeleton PASS");
}
static void ulc_cleanup(void *user) {
    UlcSample *s=user;
    array_sample_beam_destroy(s->unit);
    aec_destroy(s->aec);
    free(s->unit_pool); free(s->stage_pool); free(s->frame_pool); free(s->adapter_pool);
    free(s->microphones); free(s->far); free(s->output);
}
int main(int argc, char **argv) {
    UlcSample sample={0};
    static const ArraySampleOps ops={ulc_parse,ulc_setup,ulc_read,ulc_process,
                                  NULL,NULL,NULL,ulc_cleanup,ulc_complete};
    return array_sample_main(argc,argv,&ops,&sample);
}
