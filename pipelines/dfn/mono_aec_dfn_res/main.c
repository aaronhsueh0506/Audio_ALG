/* mono_aec_dfn_res board skeleton: AEC -> RES -> DeepFilterNet2 composed from the library calls.
 * Example code: lib/aec's Aec for the linear stage and its own RES gain, P = E * G_res (+ comfort
 * noise) formed here, and dfn/dfn_hop.h for the DFN2 hop (sized and carved by
 * dfn/dfn_hop_pool.h). Per hop:
 *
 *     aec_process_context(aec, mic, ref);
 *     follow_post_reset();                                // new aec_post_reset_serial(): reseed the CNG, clear the OLA tail
 *     aec_get_res_context(aec, &ctx);                     // CN and G_res (ctx.res_gain)
 *     aec_get_output_base(aec, &base);                    // B: the rendering base (E, or E + echo on a guard hop)
 *     P = B * G_res (+ comfort noise);                    // before the DFN hop begins
 *     dfn_hop_process(&hop, B, P, out);                   // or begin / nn_begin / nn_end / end
 *
 * The AEC context carries two signals. E (ctx.error_spec, ctx.formed_hop) is the analysis seam, never
 * altered by the output guard; B is what aec_process() itself would multiply by its gain, E bit for bit
 * on every hop the guard leaves alone. The DFN2 stage reads the pair (B, P): its estimate and the
 * spectrum it filters stay the same signal, so a guard hop does not pair a capture-based P with a
 * residual-based estimate. There is no denoiser in this chain; the stage keeps running through an AEC
 * reset event (it never resets on one). On a new aec_post_reset_serial() the comfort noise restarts from
 * AEC_CNG_SEED before that hop's first draw. At 48 kHz the host synthesis also drops its overlap-add
 * tail, DFN_RES_STAGE_LOOKAHEAD_FRAMES hops later, before the push of the first frame built after the
 * reset: the stage delays every frame by that many hops, so the event is delayed with it. Below 48 kHz
 * the host-grid synthesis lives inside dfn_rate_bridge and runs at the top of its begin(): there
 * dfn_rate_bridge_reset_synthesis() on the hop of the event clears just that overlap-add tail (DFN2,
 * the resamplers and the FIFOs keep running), with no delay line.
 *
 * 48 kHz hosts drive dfn_res_stage.h and synthesise through an AcIstft; 8/16 kHz hosts run the
 * same stage on its 48 kHz grid inside dfn_rate_bridge.h. The first output hops are zero: the
 * model's DFN_RES_STAGE_LOOKAHEAD_FRAMES hops at 48 kHz (two for the shipped 1/1 lookahead pair),
 * the bridge's added delay otherwise (printed below).
 *
 * The program builds the pool from the command-line delay profile and ERB matrices, installs a
 * fail-open model callback and runs a few silent hops. Replace run_accelerator() with the board
 * runtime call. */
#include "aec.h"
#include "aec_res.h"           /* aec_cng_fill */
#include "dfn/dfn_hop.h"
#include "dfn/dfn_hop_pool.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "dfn/dfn_example_args.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];

/* Replace with the board runtime call. `inputs` carries the feature windows and the
 * CPU-owned recurrent state; fill every tensor in `outputs` and return 0. A nonzero
 * return takes the frame as the identity (dfn_res_stage.h). */
static int run_accelerator(void *user, const DFN2PrepostInputs *inputs,
                           DFN2PrepostOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* BOARD: invoke the stateless DFN2 graph. */
}

/* Host grids accepted at parse time (0-terminated). */
static const int HOST_RATES[] = { 8000, 16000, 48000, 0 };

/* Comfort noise follows the denoiser gain of its bin but never below this amplitude (-10 dB),
 * so the residual echo it masks does not surface. */
#define CNG_NR_GAIN_FLOOR 0.31622777f

/* The mono lane: the rate's default grid and the linear-stage AecConfig keeping its own gain in
 * the context. Returns 0, or -1 for a rate libaec has no grid for. */
static int derive_lane(const DfnExampleArgs *args, AecSignalGrid *grid, AecConfig *aec_cfg) {
    if (!aec_resolve_signal_grid(args->sample_rate, aec_default_fft_size(args->sample_rate), grid))
        return -1;
    aec_config_from_level(aec_cfg, AEC_LEVEL_BALANCED, args->sample_rate);
    aec_cfg->fft_size = grid->fft_size;
    aec_cfg->delay_mode = args->delay_mode;
    aec_cfg->delay_num_filters = args->delay_num_filters;
    aec_cfg->fixed_delay_samples = args->fixed_delay_samples;
    aec_cfg->enable_delay_est = args->delay_mode == AEC_DELAY_MATCHED;
    aec_cfg->enable_res = 0;
    aec_cfg->return_res_context = 1;
    return 0;
}

/* Everything the pool holds, sized once. */
typedef struct Plan {
    AecConfig aec_cfg;
    AecSignalGrid grid;
    int enable_cng;
    DfnHopPoolPlan dfn;               /* the stage (or bridge), its synthesis and output */
    size_t aec_bytes, vec_bytes, spec_bytes;
    size_t total;
} Plan;

static int make_plan(const DfnExampleArgs *args, const DFN2Model *model, Plan *p) {
    memset(p, 0, sizeof(*p));
    if (derive_lane(args, &p->grid, &p->aec_cfg) != 0) return -1;

    /* No denoiser, RES gain only; the comfort-noise fill is the program's switch. E and P must
     * share a phase for the DFN2 stage, so it is off unless asked for. */
    p->enable_cng = args->enable_cng;
    p->vec_bytes = ALIGN16((size_t)p->grid.n_freqs * sizeof(float));
    p->spec_bytes = ALIGN16((size_t)p->grid.n_freqs * sizeof(Complex));

    p->dfn.stage_cfg = dfn_res_stage_default_config(DFN2_SR);
    p->dfn.stage_cfg.atten_lim_db = args->atten_lim_db;
    p->dfn.stage_cfg.erb_fwd = g_erb_fwd;
    p->dfn.stage_cfg.erb_inv = g_erb_inv;
    p->dfn.stage_cfg.model = *model;

    p->aec_bytes = aec_get_mem_size(&p->aec_cfg);
    if (p->aec_bytes == 0 || !isfinite(args->atten_lim_db)) return -1;
    if (dfn_hop_pool_plan(&p->dfn, args->sample_rate, 0) != 0) return -1;
    /* AEC, unity, total gain, P, then the stage (or bridge) and its synthesis */
    p->total = ALIGN16(p->aec_bytes) + 2 * p->vec_bytes + p->spec_bytes +
               dfn_hop_pool_bytes(&p->dfn);
    return 0;
}

/* The built hop pipeline, all of it inside one aligned block. */
typedef struct Rig {
    void *pool;
    Aec *aec;
    float *unity, *total_gain;        /* the RES gain path: min(1, G_res) */
    Complex *post;                    /* P */
    uint32_t rng;                     /* comfort-noise state, AEC_CNG_SEED after every hard post-chain reset */
    uint32_t reset_serial;            /* the aec_post_reset_serial() value already handled */
    uint32_t reset_pending;           /* bit k: a hard reset was seen k hops ago (the stage path's delay line) */
    int n_freqs, enable_cng;
    DfnResStage *stage;
    DfnRateBridge *bridge;
    AcIstft *istft;
    DfnHop hop;
} Rig;

static void rig_free(Rig *r) {
    if (r->bridge) dfn_rate_bridge_destroy(r->bridge);
    else if (r->stage) dfn_res_stage_destroy(r->stage);
    if (r->aec) aec_destroy(r->aec);
    free(r->pool);
    memset(r, 0, sizeof(*r));
}

static int rig_build(Rig *r, const Plan *p) {
    unsigned char *cur;
    int k;
    memset(r, 0, sizeof(*r));
    if (posix_memalign(&r->pool, 16, p->total) != 0) { r->pool = NULL; return -1; }
    cur = (unsigned char *)r->pool;
    r->aec = aec_init(cur, p->aec_bytes, &p->aec_cfg);
    cur += ALIGN16(p->aec_bytes);
    if (!r->aec) goto fail;
    r->n_freqs = p->grid.n_freqs;
    r->enable_cng = p->enable_cng;
    r->unity = (float *)cur;      cur += p->vec_bytes;
    r->total_gain = (float *)cur; cur += p->vec_bytes;
    r->post = (Complex *)cur;     cur += p->spec_bytes;
    for (k = 0; k < r->n_freqs; ++k) r->unity[k] = 1.0f;
    r->rng = AEC_CNG_SEED;
    r->reset_serial = aec_post_reset_serial(r->aec);
    if (dfn_hop_pool_build(&p->dfn, cur, p->grid.n_freqs, p->grid.hop_size, 0, &r->hop,
                           &r->stage, &r->bridge, &r->istft) != 0)
        goto fail;
    return 0;
fail:
    rig_free(r);
    return -1;
}

/* A refused hop: back to the state of a fresh pipeline, every stage together. */
static void rig_reset(Rig *r) {
    aec_reset(r->aec);
    r->rng = AEC_CNG_SEED;
    r->reset_serial = aec_post_reset_serial(r->aec);
    r->reset_pending = 0;
    dfn_hop_reset(&r->hop);
}

/* The AEC's hard post-chain reset event (aec.h, aec_post_reset_serial): the reset happened inside this
 * hop's aec_process_context(), before its synthesis. The comfort noise restarts from AEC_CNG_SEED before
 * this hop's first draw. The stage path (48 kHz) owns its synthesis: its output for the frame built on
 * this hop appears DFN_RES_STAGE_LOOKAHEAD_FRAMES hops later, so the overlap-add tail is dropped
 * before the push of that frame. The bridge path synthesises the spectra it is given at the top of
 * its begin(), so its tail is dropped now, before this hop enters it. Call this before
 * dfn_hop_process() of every hop. Nothing else follows: the AEC is not reset in response and the
 * stage keeps running. */
static void follow_post_reset(Rig *r) {
    uint32_t serial = aec_post_reset_serial(r->aec);
    uint32_t moved = serial != r->reset_serial;
    r->reset_serial = serial;
    if (moved) r->rng = AEC_CNG_SEED;
    if (moved && r->bridge) (void)dfn_rate_bridge_reset_synthesis(r->bridge);   /* between hops: cannot refuse */
    if (!r->istft) return;
    r->reset_pending = (r->reset_pending << 1) | moved;
    if ((r->reset_pending >> DFN_RES_STAGE_LOOKAHEAD_FRAMES) & 1u) ac_istft_reset(r->istft);
}

/* One host hop: P = B * min(1, G_res), B the rendering base, plus comfort noise on the bins the RES
 * cut when enabled (no denoiser: its gain is the unity vector). The stage's estimate is B too.
 * Returns 0, or -1 after a refused call (the caller resets). */
static int rig_process(Rig *r, const float *mic, const float *ref, float *out) {
    AecResContext c;
    AecOutputBase base;
    aec_process_context(r->aec, mic, ref);
    aec_get_res_context(r->aec, &c);
    if (!c.error_spec || !c.res_gain) return -1;
    memset(&base, 0, sizeof(base));
    base.struct_size = sizeof(base);
    if (aec_get_output_base(r->aec, &base) != 0) return -1;
    follow_post_reset(r);
    sk_min_f32(r->total_gain, r->unity, c.res_gain, r->n_freqs);
    sk_capply_gain_f32(r->post, base.base_spec, r->total_gain, r->n_freqs);
    if (r->enable_cng && c.comfort_noise)
        aec_cng_fill(&r->rng, r->post, c.comfort_noise, c.res_gain, r->unity, CNG_NR_GAIN_FLOOR,
                     r->n_freqs);
    return dfn_hop_process(&r->hop, base.base_spec, r->post, out) == 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    static const char *name = "mono_aec_dfn_res";
    DfnExampleArgs args;
    DFN2Model model;
    DFN2ModelIoDescriptor descriptor;
    Plan plan;
    Rig rig;
    static float mic[DFN2_HOP_LEN];
    static float far[DFN2_HOP_LEN];
    static float output[DFN2_HOP_LEN];
    int hop, index, rc, hop_size, latency;

    rc = dfn_example_parse_args(argc, argv, name, HOST_RATES, &args);
    if (rc != 0) return rc == 1 ? 0 : rc;
    if (dfn_example_load_matrices(&args, g_erb_fwd, g_erb_inv, name) != 0)
        return 1;

    /* dfn2_model_io_descriptor_default() describes the geometry this build was compiled
     * for and does not inspect the model; a product fills the descriptor from its
     * model's exported metadata. */
    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    memset(&model, 0, sizeof(model));
    model.user = NULL;
    model.infer = run_accelerator;
    model.reset = NULL;
    model.io_descriptor = &descriptor;

    if (make_plan(&args, &model, &plan) != 0) {
        fprintf(stderr, "%s: pool query rejected the configuration "
                "(host %d Hz, mode=%s n=%d fixed_delay=%d)\n", name,
                args.sample_rate, example_cli_delay_mode_name(args.delay_mode),
                args.delay_num_filters, args.fixed_delay_samples);
        return 1;
    }
    printf("%s: host %d Hz, delay profile mode=%s n=%d fixed_delay=%d cng=%d "
           "atten_lim=%.1f dB -> pool %llu bytes (align %lu), model weights "
           "external; model I/O descriptor = compiled-in layout v%u\n", name,
           args.sample_rate, example_cli_delay_mode_name(args.delay_mode),
           args.delay_num_filters, args.fixed_delay_samples, args.enable_cng,
           (double)args.atten_lim_db, (unsigned long long)plan.total,
           16ul, (unsigned)descriptor.layout_version);

    if (rig_build(&rig, &plan) != 0) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        return 1;
    }
    hop_size = plan.grid.hop_size;
    /* Host WOLA hop plus the bridge's added delay, or the model's lookahead hops plus the
     * host hop at 48 kHz. */
    DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK();   /* the pair printed is the archive's */
    latency = plan.dfn.use_bridge
        ? hop_size + dfn_rate_bridge_added_delay_samples(rig.bridge)
        : (DFN_RES_STAGE_LOOKAHEAD_FRAMES + 1) * DFN2_HOP_LEN;
    printf("%s: hop %d samples, algorithmic latency %d samples\n", name, hop_size, latency);
    /* Host smoke path: a product calls the hop once per host hop; the first hops
     * (the latency) come out silent. */
    for (hop = 0; hop < 8; ++hop) {
        if (rig_process(&rig, mic, far, output) != 0) {
            rig_reset(&rig);
            rig_free(&rig);
            return 1;
        }
        for (index = 0; index < hop_size; ++index) {
            if (!isfinite(output[index])) {
                rig_free(&rig);
                return 1;
            }
        }
    }
    rig_free(&rig);
    printf("%s: fail-open board skeleton PASS\n", name);
    return 0;
}
