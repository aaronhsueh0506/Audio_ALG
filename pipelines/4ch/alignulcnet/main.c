/* 4ch_alignulcnet board skeleton: the 4-channel linear AEC + beamformer unit feeding the
 * Align-ULCNet post-filter, composed here from the library calls. Example code: the unit is
 * the four-microphone array instance of lib/aec (AEC_RESIDUAL_STATE: lanes only; one pool placed by
 * aec_get_mem_requirements() + aec_init_ex())
 * with the sample SRP-PHAT/BF helper (array_sample_beam.h) on top; the model stream's spectrum
 * entries (stream/aiaec/ulcnet/ulcnet_stream.h) own the model call policy; the framing (an AcStft for
 * the raw far hop, an AcIstft for the output spectrum) is this file's. Per hop:
 *
 *     array_sample_beam_process_with_activity(unit, mics, far, vad, vad, NULL, &beam, &info);
 *     ac_stft_push_planar(far_stft, far, far_re, far_im);
 *     err_re[k] = beam.spectrum[k].r;  err_im[k] = beam.spectrum[k].i;        // the beamformed spectrum, planar
 *     pre = ulcnet_stream_pre_process_freq(stream, err_re, err_im, far_re, far_im,
 *                                          {info.delay.changed, .solid}, &io);
 *     if (pre == 1) ai_rc = run_accelerator(NULL, &io.inputs, &io.outputs);   // 0 = every tensor written
 *     ulcnet_stream_post_process_freq(stream, ai_rc, out_re, out_im);         // accepted frame, else the error
 *     ac_istft_push_planar(istft, out_re, out_im, out);                       // one hop
 *
 * The model gets the caller's raw `far` hop, never the unit's aligned copy. Any refused call
 * takes the unit, the stream and the two transform objects down together (array_sample_beam_reset +
 * ulcnet_stream_reset + ac_stft_reset + ac_istft_reset).
 * Grid: --sample-rate 16000 (FFT 512, default) or 48000 (FFT 1024); the UlcnetGrid built from it
 * sizes the unit and the stream. */
#include "aec.h"
#include "4ch/array_sample_beam.h"
#include "utility/mem_align.h"
#include "4ch/array_sample_driver.h"
#include "example_cli.h"
#include "audio_stft.h"
#include "ulcnet_stream.h"

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

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --sample-rate {16000|48000} grid of the stream and the unit (default 16000)\n"
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
 * every state input before writing any state output, and return 0 (or -1 on a failed run). Load
 * quantized zeros into fixed-point state at start and whenever UlcnetHopIO.state_generation
 * changes. See ulcnet_model_io.h. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* BOARD: write every output tensor, then return 0. */
}

/* The unit on the stream's grid, lanes only (the stream takes the beamformed spectrum); the
 * stream takes grid.n_bins bins per hop, so the grids must agree. */
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
    UlcnetModelIoDescriptor model_descriptor;
    UlcnetStreamConfig stream_cfg;
    UlcnetStreamMemReq stream_req;
    Aec *aec;
    ArraySampleBeam *unit;
    UlcnetStream *stream;
    AcStft *far_stft;                /* the application's far analysis and output synthesis */
    AcIstft *istft;
    float *err_re, *err_im, *far_re, *far_im, *out_re, *out_im;   /* grid.n_bins each, in frame_pool */
    void *unit_pool, *stream_pool, *frame_pool;
    size_t stream_bytes,frame_bytes;
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
    size_t stft_bytes, istft_bytes, plane_bytes;
    unsigned char *cur;
    if (ulcnet_grid_make(&s->grid, &k_grids[s->grid_index]) != ULCNET_GRID_OK) return 1;
    s->cfg = ulcnet_example_unit_config(&s->grid);
    s->bp = array_sample_beam_default_params();
    s->cfg.delay_mode = s->profile.mode;
    s->cfg.delay_num_filters = s->profile.num_filters;
    s->cfg.fixed_delay_samples = s->profile.fixed_samples;

    /* Print the resolved profile with the pools the same config costs; query and init
     * must agree on one (sample_rate, hop, n). */
    memset(&s->stream_cfg, 0, sizeof(s->stream_cfg));
    s->stream_cfg.grid = s->grid;   /* one hop per accelerator job */
    s->stream_bytes = ulcnet_stream_get_mem_size(&s->stream_cfg, &s->stream_req) == 0
        ? (size_t)s->stream_req.bytes : 0;
    ac_stft_config_default(&sc, s->grid.n_fft);
    sc.ola = AC_OLA_ACCUMULATE;
    stft_bytes = ac_stft_get_mem_size(&sc);
    istft_bytes = ac_istft_get_mem_size(&sc);
    plane_bytes = ALIGN16((size_t)s->grid.n_bins * sizeof(float));
    s->frame_bytes = ALIGN16(stft_bytes) + ALIGN16(istft_bytes) + 6 * plane_bytes;
    if (aec_get_mem_requirements(&s->cfg, &s->req) != 0 ||
        array_sample_beam_get_mem_size(&s->cfg, &s->bp, &s->beam_bytes) != 0 ||
        s->stream_bytes == 0 || stft_bytes == 0 || istft_bytes == 0) {
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
           "(one shared estimator) -> unit pool %llu bytes (align %lu) + stream pool %zu bytes\n",
           example_cli_delay_mode_name(s->profile.mode), s->profile.num_filters,
           s->profile.fixed_samples,
           (unsigned long long)(s->aec_bytes + s->beam_bytes),
           (unsigned long)s->req.alignment, s->stream_bytes);
    printf("4ch_alignulcnet: application framing (far AcStft, output AcIstft, six spectrum planes) pool %zu bytes\n",
           s->frame_bytes);

    /* The board loads this descriptor from the exported model metadata and refuses a graph of another
     * layout, grid or far input, or a packet graph, before any hop runs; here it is the grid's own. */
    if (ulcnet_model_io_descriptor_default(&s->grid, &s->model_descriptor) != 0 ||
        ulcnet_model_io_descriptor_validate(&s->grid, &s->model_descriptor) != 0 ||
        ulcnet_model_io_descriptor_frames(&s->model_descriptor) != 1) {
        return 1;
    }
    printf("%s: model I/O D=%d -> model stream pool %zu bytes (align %u)\n",
           s->program, s->grid.delay_depth, s->stream_bytes,
           (unsigned)s->stream_req.alignment);

    if (posix_memalign(&s->unit_pool, s->req.alignment,
                       s->aec_bytes + s->beam_bytes) != 0)
        s->unit_pool = NULL;
    if (posix_memalign(&s->stream_pool, s->stream_req.alignment, s->stream_bytes) != 0)
        s->stream_pool = NULL;
    if (posix_memalign(&s->frame_pool, 16, s->frame_bytes) != 0)
        s->frame_pool = NULL;
    if (s->unit_pool) s->aec = aec_init_ex(s->unit_pool, s->aec_bytes, &s->cfg, &s->req);
    s->unit = s->aec
        ? array_sample_beam_init((unsigned char *)s->unit_pool + s->aec_bytes,
                                s->beam_bytes, &s->cfg, &s->bp, s->aec)
        : NULL;
    /* The stream holds no recurrent state: a board's accelerator does. */
    s->stream = s->stream_pool
        ? ulcnet_stream_init_ex(&s->stream_cfg, s->stream_pool, s->stream_bytes, &s->stream_req) : NULL;
    if (s->frame_pool) {
        cur = (unsigned char *)s->frame_pool;
        s->far_stft = ac_stft_init(cur, stft_bytes, &sc);   cur += ALIGN16(stft_bytes);
        s->istft = ac_istft_init(cur, istft_bytes, &sc);    cur += ALIGN16(istft_bytes);
        s->err_re = (float *)cur;                           cur += plane_bytes;
        s->err_im = (float *)cur;                           cur += plane_bytes;
        s->far_re = (float *)cur;                           cur += plane_bytes;
        s->far_im = (float *)cur;                           cur += plane_bytes;
        s->out_re = (float *)cur;                           cur += plane_bytes;
        s->out_im = (float *)cur;
    }
    if (!s->unit || !s->stream || !s->far_stft || !s->istft ||
        aec_hop_size(s->aec) != s->grid.hop ||
        aec_hop_size(s->aec) + 1 != s->grid.n_bins) {
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

/* One hop: the unit, the application's analysis of the raw far, the stream with the beamformed
 * spectrum, the far spectrum and the alignment events, the model run and the synthesis. A hop without
 * a frame (the reprime hop after an alignment boundary) and a failed run pass the beamformed spectrum
 * through. */
static int ulc_hop(UlcSample *s) {
    ArraySampleBeamResult beam;
    ArraySampleBeamInfo info;
    UlcnetStreamDelay delay;
    UlcnetHopIO io;
    int k, pre, ai_rc = -1;
    if (array_sample_beam_process_with_activity(s->unit, s->microphones, s->far, 0, 0, NULL,
                                               &beam, &info) != AEC_OK)
        return -1;
    delay.changed = info.delay.changed;
    delay.solid = info.delay.solid;
    (void)ac_stft_push_planar(s->far_stft, s->far, s->far_re, s->far_im);
    for (k = 0; k < s->grid.n_bins; ++k) {
        s->err_re[k] = beam.spectrum[k].r;
        s->err_im[k] = beam.spectrum[k].i;
    }
    pre = ulcnet_stream_pre_process_freq(s->stream, s->err_re, s->err_im, s->far_re, s->far_im, delay, &io);
    if (pre < 0) return -1;
    if (pre == 1) ai_rc = run_accelerator(NULL, &io.inputs, &io.outputs);
    if (ulcnet_stream_post_process_freq(s->stream, ai_rc, s->out_re, s->out_im) != 1) return -1;
    return ac_istft_push_planar(s->istft, s->out_re, s->out_im, s->output) == s->grid.hop ? 0 : -1;
}
static int ulc_process(void *user, int frame) {
    UlcSample *s=user; (void)frame;
    if (ulc_hop(s) == 0) return 0;
    array_sample_beam_reset(s->unit);
    ulcnet_stream_reset(s->stream);
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
    free(s->unit_pool); free(s->stream_pool); free(s->frame_pool);
    free(s->microphones); free(s->far); free(s->output);
}
int main(int argc, char **argv) {
    UlcSample sample={0};
    static const ArraySampleOps ops={ulc_parse,ulc_setup,ulc_read,ulc_process,
                                  NULL,NULL,NULL,ulc_cleanup,ulc_complete};
    return array_sample_main(argc,argv,&ops,&sample);
}
