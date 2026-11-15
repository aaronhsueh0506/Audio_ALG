/* mono_alignulcnet board skeleton: the mono linear AEC feeding the Align-ULCNet post-filter,
 * composed here from the library calls. Example code: lib/aec's Aec for the linear stage and the
 * model stream's spectrum entries (stream/aiaec/ulcnet/ulcnet_stream.h) for the model call policy;
 * the framing (an AcStft for the formed hop, one for the raw reference, an AcIstft for the output
 * spectrum) is this file's. Per hop:
 *
 *     aec_process_linear(aec, mic, ref);  aec_process_residual(aec, NULL);
 *     aec_get_residual / aec_get_linear                 // formed hop, alignment state
 *     ac_stft_push_planar(err_stft, rctx.formed_hop, err_re, err_im);
 *     ac_stft_push_planar(far_stft, ref, far_re, far_im);
 *     pre = ulcnet_stream_pre_process_freq(stream, err_re, err_im, far_re, far_im, delay, &io);
 *     if (pre == 1) ai_rc = run_accelerator(NULL, &io.inputs, &io.outputs);   // 0 = every tensor written
 *     ulcnet_stream_post_process_freq(stream, ai_rc, out_re, out_im);         // accepted frame, else the error
 *     ac_istft_push_planar(istft, out_re, out_im, out);                       // one hop
 *
 * Grid: --sample-rate 16000 (FFT 512, default) or 48000 (FFT 1024); the UlcnetGrid built from it
 * sizes the lane and the stream. The model gets the caller's raw `ref` hop. Replace
 * run_accelerator() with the board runtime call.
 * Data domain: the stream reads the AEC's analysis seam (rctx.formed_hop), the signal the model was trained
 * on, not the rendering base_spec (which does not pair with formed_hop on a hop where the output guard
 * fires). There is no comfort noise and no synthesis besides this file's AcIstft, so
 * aec_post_reset_serial() has nothing to clear or reseed here; the stream's alignment policy is the
 * delay.changed flag below (an in-call hard reset of the AEC is also a delay.changed hop). A refused
 * call resets the AEC, the stream and the three transform objects together. */
#include "aec.h"
#include "audio_stft.h"
#include "example_cli.h"
#include "ulcnet_stream.h"
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
        "  --sample-rate {16000|48000} grid of the stream and the lane (default 16000)\n"
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
 * and whenever UlcnetHopIO.state_generation changes. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* BOARD: invoke the stateless ONNX accelerator. */
}

/* The mono lane on the stream's grid: the state of the linear residual only (AEC_RESIDUAL_STATE); the
 * stream takes the formed hop's spectrum. */
static void derive_lane(const ExampleCliDelayOptions* profile, const UlcnetGrid* grid,
                        AecConfig* aec_cfg) {
    aec_config_from_level(aec_cfg, AEC_LEVEL_BALANCED, grid->sample_rate);
    aec_cfg->fft_size = grid->n_fft;
    aec_cfg->delay_mode = profile->mode;
    aec_cfg->delay_num_filters = profile->num_filters;
    aec_cfg->fixed_delay_samples = profile->fixed_samples;
    aec_cfg->residual = AEC_RESIDUAL_STATE;
}

/* The application's framing around the stream (audiolib defaults on the grid: sqrt-Hann, centre 0, one
 * frame per hop; the synthesis accumulates). */
typedef struct Framing {
    AcStft *far_stft, *err_stft;
    AcIstft* istft;
    float *err_re, *err_im, *far_re, *far_im, *out_re, *out_im;   /* grid.n_bins each */
    int hop;
} Framing;

static void framing_config(AcStftConfig* sc, const UlcnetGrid* grid) {
    ac_stft_config_default(sc, grid->n_fft);
    sc->ola = AC_OLA_ACCUMULATE;
}

/* Two analyses, one synthesis and six spectrum planes, each in its own 16-byte slot. */
static size_t framing_bytes(const UlcnetGrid* grid) {
    AcStftConfig sc;
    framing_config(&sc, grid);
    return 2 * ALIGN16(ac_stft_get_mem_size(&sc)) + ALIGN16(ac_istft_get_mem_size(&sc)) +
           6 * ALIGN16((size_t)grid->n_bins * sizeof(float));
}

static int framing_init(Framing* f, void* mem, size_t bytes, const UlcnetGrid* grid) {
    AcStftConfig sc;
    unsigned char* cur = (unsigned char*)mem;
    size_t stft_b, istft_b, plane_b = ALIGN16((size_t)grid->n_bins * sizeof(float));
    if (!mem || bytes < framing_bytes(grid)) return -1;
    framing_config(&sc, grid);
    stft_b = ac_stft_get_mem_size(&sc);
    istft_b = ac_istft_get_mem_size(&sc);
    f->far_stft = ac_stft_init(cur, stft_b, &sc);   cur += ALIGN16(stft_b);
    f->err_stft = ac_stft_init(cur, stft_b, &sc);   cur += ALIGN16(stft_b);
    f->istft = ac_istft_init(cur, istft_b, &sc);    cur += ALIGN16(istft_b);
    f->err_re = (float*)cur;                        cur += plane_b;
    f->err_im = (float*)cur;                        cur += plane_b;
    f->far_re = (float*)cur;                        cur += plane_b;
    f->far_im = (float*)cur;                        cur += plane_b;
    f->out_re = (float*)cur;                        cur += plane_b;
    f->out_im = (float*)cur;
    f->hop = grid->hop;
    return f->far_stft && f->err_stft && f->istft ? 0 : -1;
}

/* The stream and the transforms restart together. */
static void reset_pipeline(Aec* aec, UlcnetStream* stream, Framing* f) {
    aec_reset(aec);
    ulcnet_stream_reset(stream);
    ac_stft_reset(f->far_stft);
    ac_stft_reset(f->err_stft);
    ac_istft_reset(f->istft);
}

/* One hop: linear AEC, then the framing of the lane's formed hop and of the raw reference, the stream
 * with the alignment events (changed = CHANGED, solid = not UNLOCKED), the model run and the synthesis.
 * A hop without a frame (the reprime hop after an alignment boundary) and a failed run pass the error
 * spectrum through. */
static int process_hop(Aec* aec, UlcnetStream* stream, Framing* f, const float* mic, const float* ref,
                       float* out) {
    AecResidualView rctx;
    AecLinearView lctx;
    UlcnetStreamDelay delay;
    UlcnetHopIO io;
    int pre, ai_rc = -1;
    if (aec_process_linear(aec, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(aec, NULL) != AEC_OK) return -1;
    memset(&rctx, 0, sizeof(rctx));
    rctx.struct_size = sizeof(rctx);
    memset(&lctx, 0, sizeof(lctx));
    lctx.struct_size = sizeof(lctx);
    if (aec_get_residual(aec, &rctx) != AEC_OK || aec_get_linear(aec, &lctx) != AEC_OK) return -1;
    delay.changed = lctx.delay_state == AEC_LINEAR_DELAY_CHANGED;
    delay.solid = lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
    (void)ac_stft_push_planar(f->err_stft, rctx.formed_hop, f->err_re, f->err_im);
    (void)ac_stft_push_planar(f->far_stft, ref, f->far_re, f->far_im);
    pre = ulcnet_stream_pre_process_freq(stream, f->err_re, f->err_im, f->far_re, f->far_im, delay, &io);
    if (pre < 0) return -1;
    if (pre == 1) ai_rc = run_accelerator(NULL, &io.inputs, &io.outputs);
    if (ulcnet_stream_post_process_freq(stream, ai_rc, f->out_re, f->out_im) != 1) return -1;
    return ac_istft_push_planar(f->istft, f->out_re, f->out_im, out) == f->hop ? 0 : -1;
}

int main(int argc, char** argv) {
    AecConfig aec_cfg;
    ExampleCliDelayOptions profile;
    UlcnetGrid grid;
    UlcnetModelIoDescriptor model_descriptor;
    UlcnetStreamConfig stream_cfg;
    UlcnetStreamMemReq stream_req;
    UlcnetStream *stream = NULL;
    Framing framing;
    Aec *aec = NULL;
    void *pool = NULL;
    float *mic = NULL, *far = NULL, *output = NULL;
    size_t aec_bytes, stream_bytes, framing_pool_bytes, pool_bytes;
    int grid_index;
    int hop;
    int index;
    int rc;
    rc = parse_options(argc, argv, &profile, &grid_index);
    if (rc != 0) return rc == 1 ? 0 : rc;
    if (ulcnet_grid_make(&grid, &k_grids[grid_index]) != ULCNET_GRID_OK) return 1;

    /* Print the resolved profile with the pool the same config costs; query and init
     * must agree on one (sample_rate, hop, n). */
    memset(&stream_cfg, 0, sizeof(stream_cfg));
    stream_cfg.grid = grid;   /* one hop per accelerator job */
    stream_cfg.freq_only = 1; /* the application runs the analysis and the ISTFT */
    stream_bytes = ulcnet_stream_get_mem_size(&stream_cfg, &stream_req) == 0 ? (size_t)stream_req.bytes : 0;
    framing_pool_bytes = framing_bytes(&grid);
    derive_lane(&profile, &grid, &aec_cfg);
    aec_bytes = aec_get_mem_size(&aec_cfg);
    if (aec_bytes == 0 || stream_bytes == 0) {
        fprintf(stderr,
                "mono_alignulcnet: pool query rejected the delay profile "
                "(mode=%s n=%d fixed_delay=%d)\n",
                example_cli_delay_mode_name(profile.mode), profile.num_filters,
                profile.fixed_samples);
        return 1;
    }
    pool_bytes = ALIGN16(aec_bytes) + ALIGN16(stream_bytes) + ALIGN16(framing_pool_bytes);
    printf("mono_alignulcnet: grid %d Hz / fft %d / hop %d\n", grid.sample_rate, grid.n_fft,
           grid.hop);
    printf("mono_alignulcnet: delay profile mode=%s n=%d fixed_delay=%d "
           "-> pool %llu bytes (align %lu)\n",
           example_cli_delay_mode_name(profile.mode), profile.num_filters,
           profile.fixed_samples, (unsigned long long)pool_bytes, 16ul);

    /* The board loads this descriptor from the exported model metadata and refuses a graph of another
     * layout, grid or far input, or a packet graph, before any hop runs; here it is the grid's own. */
    if (ulcnet_model_io_descriptor_default(&grid, &model_descriptor) != 0 ||
        ulcnet_model_io_descriptor_validate(&grid, &model_descriptor) != 0 ||
        ulcnet_model_io_descriptor_frames(&model_descriptor) != 1) {
        return 1;
    }
    printf("%s: model I/O D=%d -> model stream %zu bytes of that pool (align %u)\n",
           argv[0], grid.delay_depth, stream_bytes, (unsigned)stream_req.alignment);

    mic = (float*)calloc((size_t)grid.hop, sizeof(float));
    far = (float*)calloc((size_t)grid.hop, sizeof(float));
    output = (float*)calloc((size_t)grid.hop, sizeof(float));
    if (!mic || !far || !output || posix_memalign(&pool, 16, pool_bytes) != 0) {
        free(mic); free(far); free(output);
        return 1;
    }
    aec = aec_init(pool, aec_bytes, &aec_cfg);
    /* The stream holds no recurrent state: a board's accelerator does. */
    stream = aec ? ulcnet_stream_init_ex(&stream_cfg, (unsigned char*)pool + ALIGN16(aec_bytes), stream_bytes,
                                         &stream_req)
                 : NULL;
    if (!stream ||
        framing_init(&framing, (unsigned char*)pool + ALIGN16(aec_bytes) + ALIGN16(stream_bytes),
                     framing_pool_bytes, &grid) != 0) {
        fprintf(stderr, "mono_alignulcnet: pipeline init failed\n");
        rc = 1;
    } else {
        /* Host smoke path: a product calls the hop once per grid.hop-sample hop. */
        for (hop = 0; hop < 4 && rc == 0; ++hop) {
            if (process_hop(aec, stream, &framing, mic, far, output) != 0) {
                reset_pipeline(aec, stream, &framing);   /* the AEC, the stream and the transforms restart together */
                rc = 1;
            }
            for (index = 0; index < grid.hop && rc == 0; ++index)
                if (!isfinite(output[index])) rc = 1;
        }
    }

    if (aec) aec_destroy(aec);
    free(pool);
    free(mic); free(far); free(output);
    if (rc != 0) return 1;
    puts("mono_alignulcnet: fail-open board skeleton PASS");
    return 0;
}
