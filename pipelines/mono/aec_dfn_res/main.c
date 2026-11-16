/* mono_aec_dfn_res board skeleton: AEC -> RES -> DeepFilterNet2 composed from the library calls.
 * Example code: lib/aec's Aec for the linear stage and its own RES gain, P = B * G_res (+ comfort
 * noise) formed here, and dfn/dfn_hop.h for the DFN2 hop (sized and carved by
 * dfn/dfn_hop_pool.h). Per hop:
 *
 *     aec_process_linear(aec, mic, ref);  aec_process_residual(aec, NULL);
 *     follow_post_reset();                                // new aec_post_reset_serial(): clear the OLA tail
 *     aec_get_residual(aec, &v);                          // v.base_spec = B: the rendering base (E, or E + echo on a guard hop)
 *     aec_apply_gain(aec, NULL, P);                       // P = B * G_res
 *     aec_apply_cng(aec, NULL, P);                        // (+ comfort noise), before the DFN hop begins
 *     dfn_hop_process(&hop, B, P, out);                   // or begin / nn_begin / nn_end / end
 *
 * The AEC residual view carries two signals. E (v.error_spec, v.formed_hop) is the analysis seam, never
 * altered by the output guard; B is what aec_process() itself would multiply by its gain, E bit for bit
 * on every hop the guard leaves alone. The DFN2 hop reads the pair (B, P): its estimate and the
 * spectrum it filters stay the same signal, so a guard hop does not pair a capture-based P with a
 * residual-based estimate. There is no denoiser in this chain; the hop keeps running through an AEC
 * reset event (it never resets on one). On a new aec_post_reset_serial() the library reseeds its
 * comfort-noise generator before that hop's first draw. With the host on the model grid's rate the
 * host synthesis also drops its overlap-add tail, dfn_hop_lookahead_frames() hops later (0 for
 * the 16 kHz row), before the push of the first frame built after the reset: the model delays
 * every frame by that many hops, so the event is delayed with it. On the bridge path the host-grid
 * synthesis lives inside the AinrBridge and runs at the top of its begin(): there
 * ainr_bridge_reset_synthesis() on the hop of the event clears just that overlap-add tail (DFN2,
 * the resamplers and the FIFOs keep running), with no delay line.
 *
 * A host on the model's own rate (--ai-sample-rate, default 48 kHz) drives the model stream
 * (stream/ainr/dfn2, memory mode) and synthesises through an AcIstft; any other host rate runs the
 * same stream on its 48 kHz grid inside stream/ainr/bridge's AinrBridge, which needs the 48 kHz /
 * 1024 model. The model grid is built from the --ai-sample-rate / --ai-fft-size pair
 * (dfn_example_args.h) with dfn2_grid_make() and handed to every object; nothing here is fixed at
 * build time. The first output hops are zero: the model's dfn_hop_lookahead_frames() hops on the
 * direct path (two for the shipped 1/1 lookahead pair), the bridge's added delay otherwise
 * (printed below).
 *
 * The program builds the pool from the command-line delay profile and ERB matrices, installs a
 * fail-open model callback and runs a few silent hops. Replace run_accelerator() with the board
 * runtime call. */
#include "aec.h"
#include "dfn/dfn_hop.h"
#include "dfn/dfn_hop_pool.h"
#include "utility/mem_align.h"
#include "dfn/dfn_example_args.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Host hop buffers: the largest host hop HOST_RATES offers (48 kHz, FFT 1024). */
#define HOST_MAX_HOP 512

/* Replace with the board runtime call. `inputs` carries the feature windows; the recurrent state
 * is the accelerator's own (its views are NULL/0 here), bound input to output in place and
 * refilled with quantized zeros whenever job.state_generation (dfn_hop_state_generation()) changes.
 * The first mask_lookahead hops after a start have no model frame and do not call this. Fill the heads
 * in `outputs` and return 0. Any other return takes the frame
 * as the identity (dfn_hop.h): STREAM_RUN_NOT_STARTED when no run began (the device state is untouched),
 * STREAM_RUN_FAILED, or any other nonzero value, once it did (the library then advances the state generation). */
static int run_accelerator(void *user, const DFN2PrepostInputs *inputs,
                           DFN2PrepostOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return STREAM_RUN_NOT_STARTED; /* BOARD: invoke the stateless DFN2 graph. */
}

/* Host grids accepted at parse time (0-terminated). */
static const int HOST_RATES[] = { 8000, 16000, 48000, 0 };

/* The mono lane: the rate's default grid and the AecConfig keeping its own gain in the residual
 * view (AEC_RESIDUAL_FULL). Returns 0, or -1 for a rate libaec has no grid for. */
static int derive_lane(const DfnExampleArgs *args, AecSignalGrid *grid, AecConfig *aec_cfg) {
    if (aec_resolve_signal_grid(args->sample_rate, aec_default_fft_size(args->sample_rate), grid) != AEC_OK)
        return -1;
    aec_config_from_level(aec_cfg, AEC_LEVEL_BALANCED, args->sample_rate);
    aec_cfg->fft_size = grid->fft_size;
    aec_cfg->delay_mode = args->delay_mode;
    aec_cfg->delay_num_filters = args->delay_num_filters;
    aec_cfg->fixed_delay_samples = args->fixed_delay_samples;
    aec_cfg->residual = AEC_RESIDUAL_FULL;
    return 0;
}

/* Everything the pool holds, sized once. */
typedef struct Plan {
    AecConfig aec_cfg;
    AecSignalGrid grid;
    int enable_cng;
    DfnHopPoolPlan dfn;               /* the model stream, its conversion buffers, the synthesis or bridge */
    size_t aec_bytes, spec_bytes;
    size_t total;
} Plan;

static int make_plan(const DfnExampleArgs *args, const DfnExampleErb *erb,
                     const DFN2Model *model, Plan *p) {
    memset(p, 0, sizeof(*p));
    if (derive_lane(args, &p->grid, &p->aec_cfg) != 0 || p->grid.hop_size > HOST_MAX_HOP)
        return -1;

    /* No denoiser, RES gain only; the comfort-noise fill is the program's switch. E and P must
     * share a phase for the DFN2 hop, so it is off unless asked for. */
    p->enable_cng = args->enable_cng;
    p->spec_bytes = ALIGN16((size_t)p->grid.n_freqs * sizeof(Complex));

    dfn_hop_pool_plan_init(&p->dfn, &args->model_grid);
    p->dfn.stream_cfg.atten_lim_db = args->atten_lim_db;
    p->dfn.stream_cfg.erb_fwd = erb->fwd;
    p->dfn.stream_cfg.erb_inv = erb->inv;
    p->dfn.model = *model;

    p->aec_bytes = aec_get_mem_size(&p->aec_cfg);
    if (p->aec_bytes == 0 || !isfinite(args->atten_lim_db)) return -1;
    if (dfn_hop_pool_plan(&p->dfn, args->sample_rate, 0) != 0) return -1;
    /* AEC, P, then the model stream, its buffers and the synthesis (or bridge) */
    p->total = ALIGN16(p->aec_bytes) + p->spec_bytes + dfn_hop_pool_bytes(&p->dfn);
    return 0;
}

/* The built hop pipeline, all of it inside one aligned block. */
typedef struct Rig {
    void *pool;
    Aec *aec;
    Complex *post;                    /* P */
    uint32_t reset_serial;            /* the aec_post_reset_serial() value already handled */
    int enable_cng;
    DfnResetFollow follow;            /* the synthesis clear that follows a hard reset */
    DfnHop hop;                       /* hop.bridge / hop.istft: the bridge, or the synthesis */
} Rig;

static void rig_free(Rig *r) {
    if (r->aec) aec_destroy(r->aec);
    free(r->pool);
    memset(r, 0, sizeof(*r));
}

static int rig_build(Rig *r, const Plan *p) {
    unsigned char *cur;
    memset(r, 0, sizeof(*r));
    if (posix_memalign(&r->pool, 16, p->total) != 0) { r->pool = NULL; return -1; }
    cur = (unsigned char *)r->pool;
    r->aec = aec_init(cur, p->aec_bytes, &p->aec_cfg);
    cur += ALIGN16(p->aec_bytes);
    if (!r->aec) goto fail;
    r->enable_cng = p->enable_cng;
    dfn_reset_follow_init(&r->follow, &p->dfn.stream_cfg.grid);
    r->post = (Complex *)cur;     cur += p->spec_bytes;
    r->reset_serial = aec_post_reset_serial(r->aec);
    if (dfn_hop_pool_build(&p->dfn, cur, p->grid.n_freqs, p->grid.hop_size, 0, &r->hop) != 0)
        goto fail;
    return 0;
fail:
    rig_free(r);
    return -1;
}

/* A refused hop: back to the state of a fresh pipeline, every stage together. */
static void rig_reset(Rig *r) {
    aec_reset(r->aec);
    r->reset_serial = aec_post_reset_serial(r->aec);
    dfn_reset_follow_clear(&r->follow);
    dfn_hop_reset(&r->hop);
}

/* The AEC's hard post-chain reset event (aec.h, aec_post_reset_serial): the reset happened inside this
 * hop's aec_process_linear(), before its synthesis; the library reseeds its comfort-noise generator.
 * The direct path (host rate equals the model grid's) owns its synthesis: its output for the frame built on
 * this hop appears dfn_hop_lookahead_frames() hops later, so the overlap-add tail is dropped
 * before the push of that frame. The bridge path synthesises the spectra it is given at the top of
 * its begin(), so its tail is dropped now, before this hop enters it. Call this before
 * dfn_hop_process() of every hop. Nothing else follows: the AEC is not reset in response and the
 * model stream keeps running. */
static void follow_post_reset(Rig *r) {
    uint32_t serial = aec_post_reset_serial(r->aec);
    uint32_t moved = serial != r->reset_serial;
    r->reset_serial = serial;
    (void)dfn_reset_follow(&r->follow, &r->hop, moved != 0);   /* between hops: cannot refuse */
}

/* One host hop: P = B * G_res, B the rendering base, plus comfort noise on the bins the RES cut when
 * enabled (no denoiser: no extra gain). The model's estimate is B too. Returns 0, or -1 after a refused
 * call (the caller resets). */
static int rig_process(Rig *r, const float *mic, const float *ref, float *out) {
    AecResidualView v;
    if (aec_process_linear(r->aec, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(r->aec, NULL) != AEC_OK) return -1;
    memset(&v, 0, sizeof(v));
    v.struct_size = sizeof(v);
    if (aec_get_residual(r->aec, &v) != AEC_OK) return -1;
    follow_post_reset(r);
    if (aec_apply_gain(r->aec, NULL, r->post) != AEC_OK) return -1;
    if (r->enable_cng && aec_apply_cng(r->aec, NULL, r->post) != AEC_OK) return -1;
    return dfn_hop_process(&r->hop, v.base_spec, r->post, out) == 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    static const char *name = "mono_aec_dfn_res";
    DfnExampleArgs args;
    DfnExampleErb erb;
    DFN2Model model;
    DFN2ModelIoDescriptor descriptor;
    Plan plan;
    Rig rig;
    static float mic[HOST_MAX_HOP];
    static float far[HOST_MAX_HOP];
    static float output[HOST_MAX_HOP];
    int hop, index, rc, hop_size, latency, refused = 0, status = 1;

    rc = dfn_example_parse_args(argc, argv, name, HOST_RATES, &args);
    if (rc != 0) return rc == 1 ? 0 : rc;
    if (dfn_example_load_matrices(&args, &erb, name) != 0)
        return 1;

    /* dfn2_model_io_descriptor_default() fills the external fields for the model grid
     * (args.model_grid) and does not inspect the model; the interior fields stay 0 and are never
     * read. A host that reads the exported model metadata fills it from there; the board samples
     * check ports only, which do not depend on the lookahead pair. */
    if (dfn2_model_io_descriptor_default(&args.model_grid, &descriptor) != 0) goto done;
    memset(&model, 0, sizeof(model));
    model.user = NULL;
    model.infer = run_accelerator;
    model.reset = NULL;
    model.io_descriptor = &descriptor;

    if (make_plan(&args, &erb, &model, &plan) != 0) {
        fprintf(stderr, "%s: pool query rejected the configuration "
                "(host %d Hz, model grid %d Hz / FFT %d, mode=%s n=%d fixed_delay=%d)\n", name,
                args.sample_rate, args.model_grid.sample_rate, args.model_grid.n_fft,
                example_cli_delay_mode_name(args.delay_mode),
                args.delay_num_filters, args.fixed_delay_samples);
        goto done;
    }
    printf("%s: host %d Hz, delay profile mode=%s n=%d fixed_delay=%d cng=%d "
           "atten_lim=%.1f dB -> pool %llu bytes (align %lu), model weights "
           "external; model I/O descriptor = compiled-in layout v%u\n", name,
           args.sample_rate, example_cli_delay_mode_name(args.delay_mode),
           args.delay_num_filters, args.fixed_delay_samples, args.enable_cng,
           (double)args.atten_lim_db, (unsigned long long)plan.total,
           16ul, (unsigned)descriptor.layout_version);
    printf("%s: model grid %d Hz / FFT %d (n_erb %d, df_bins %d, lookahead %d/%d), %s\n", name,
           args.model_grid.sample_rate, args.model_grid.n_fft, args.model_grid.n_erb,
           args.model_grid.df_bins, args.model_grid.mask_lookahead, args.model_grid.df_lookahead,
           plan.dfn.use_bridge ? "through the rate bridge" : "the model stream on the host grid");

    if (rig_build(&rig, &plan) != 0) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        goto done;
    }
    hop_size = plan.grid.hop_size;
    /* Host WOLA hop plus the bridge's added delay, or the model's lookahead hops plus the
     * host hop on the direct path. */
    DFN2_REQUIRE_VARIANT_LINK();   /* this unit's DFN2 variant is the archive's */
    latency = plan.dfn.use_bridge
        ? hop_size + ainr_bridge_added_delay_samples(rig.hop.bridge)
        : (dfn_hop_lookahead_frames(&args.model_grid) + 1) * args.model_grid.hop;
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name, hop_size, latency);
    /* Host smoke path: a product calls the hop once per host hop; the first hops
     * (the latency) come out silent. */
    for (hop = 0; hop < 8; ++hop) {
        if (rig_process(&rig, mic, far, output) != 0) {
            /* A refused hop (e.g. AEC_ERR_DSP, the AEC already reset): silence, every stage restarts, the next hop runs. */
            rig_reset(&rig);
            memset(output, 0, (size_t)hop_size * sizeof(float));
            ++refused;
        }
        for (index = 0; index < hop_size; ++index) {
            if (!isfinite(output[index])) {
                rig_free(&rig);
                goto done;
            }
        }
    }
    rig_free(&rig);
    if (refused) {   /* the smoke input is benign: any refusal is a failure */
        fprintf(stderr, "%s: %d of 8 hops refused\n", name, refused);
        goto done;
    }
    printf("%s: fail-open board skeleton PASS\n", name);
    status = 0;
done:
    dfn_example_erb_free(&erb);
    return status;
}
