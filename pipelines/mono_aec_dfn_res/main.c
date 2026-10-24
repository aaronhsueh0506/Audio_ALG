/* mono_aec_dfn_res board skeleton: AEC -> RES -> DeepFilterNet2 composed from the library calls.
 * Example code: lib/aec's Aec for the linear stage and its own RES gain, se_frame_from_aec() for
 * the frame view, P = E * G_res (+ comfort noise) formed here, and samples/dfn_hop.h for the
 * DFN2 hop. Per hop:
 *
 *     aec_process_context(aec, mic, ref);
 *     se_frame_from_aec(aec, &frame);                     // E, CN and G_res (frame.g_res_ctx)
 *     P = E * G_res (+ comfort noise);                    // before the DFN hop begins
 *     dfn_hop_process(&hop, frame.e, P, out);             // or begin / nn_begin / nn_end / end
 *
 * 48 kHz hosts drive dfn_res_stage.h and synthesise through an AcIstft; 8/16 kHz hosts run the
 * same stage on its 48 kHz grid inside dfn_rate_bridge.h. The first output hops are zero: two
 * model hops at 48 kHz, the bridge's added delay otherwise (printed below).
 *
 * The program builds the pool from the command-line delay profile and ERB matrices, installs a
 * fail-open model callback and runs a few silent hops. Replace run_accelerator() with the board
 * runtime call. */
#include "aec.h"
#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/dfn_hop.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "dfn_example_args.h"

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

/* The mono lane: the rate's grid and the linear-stage AecConfig. */
static int derive_lane(const DfnExampleArgs *args, SeGrid *grid, AecConfig *aec_cfg) {
    static const SeRules rules = { .allow_8k = 1, .spatial_context = SE_CONTEXT_PRESET };
    SeRequest req;
    memset(&req, 0, sizeof(req));
    req.sample_rate = args->sample_rate;
    req.fft_size = 0;   /* the host's rate default */
    req.delay_mode = args->delay_mode;
    req.delay_num_filters = args->delay_num_filters;
    req.fixed_delay_samples = args->fixed_delay_samples;
    req.aec_preset = AEC_PRESET_BALANCED;
    return se_derive(&req, &rules, grid, aec_cfg);
}

/* Everything the pool holds, sized once. */
typedef struct Plan {
    AecConfig aec_cfg;
    SeGrid grid;
    int enable_cng;
    DfnResStageConfig stage_cfg;
    DfnRateBridgeConfig bridge_cfg;
    int use_bridge;                   /* host below 48 kHz */
    size_t aec_bytes, vec_bytes, spec_bytes, stage_bytes, istft_bytes;
    size_t total;
} Plan;

static int make_plan(const DfnExampleArgs *args, const DFN2Model *model, Plan *p) {
    DfnResStageMemReq stage_req;
    DfnRateBridgeMemReq bridge_req;
    AcStftConfig scfg;
    memset(p, 0, sizeof(*p));
    if (derive_lane(args, &p->grid, &p->aec_cfg) != 0) return -1;

    /* No denoiser, RES gain only; the comfort-noise fill is the program's switch. E and P must
     * share a phase for the DFN2 stage, so it is off unless asked for. */
    p->enable_cng = args->enable_cng;
    p->vec_bytes = ALIGN16((size_t)p->grid.n_freqs * sizeof(float));
    p->spec_bytes = ALIGN16((size_t)p->grid.n_freqs * sizeof(Complex));

    p->stage_cfg = dfn_res_stage_default_config(DFN2_SR);
    p->stage_cfg.atten_lim_db = args->atten_lim_db;
    p->stage_cfg.erb_fwd = g_erb_fwd;
    p->stage_cfg.erb_inv = g_erb_inv;
    p->stage_cfg.model = *model;

    p->use_bridge = args->sample_rate != DFN2_SR;
    p->aec_bytes = aec_get_mem_size(&p->aec_cfg);
    if (p->aec_bytes == 0 || !isfinite(args->atten_lim_db)) return -1;
    if (p->use_bridge) {
        p->bridge_cfg = dfn_rate_bridge_default_config(args->sample_rate);
        p->bridge_cfg.fft_size = 0;
        p->bridge_cfg.stage = p->stage_cfg;
        if (dfn_rate_bridge_get_mem_requirements(&p->bridge_cfg, &bridge_req) != 0) return -1;
        p->stage_bytes = (size_t)bridge_req.bytes;
    } else {
        if (dfn_res_stage_get_mem_requirements(&p->stage_cfg, &stage_req) != 0) return -1;
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        p->istft_bytes = ac_istft_get_mem_size(&scfg);
        if (p->istft_bytes == 0) return -1;
        p->stage_bytes = (size_t)stage_req.bytes;
    }
    /* AEC, unity, total gain, P, then the stage (or bridge) and its synthesis */
    p->total = ALIGN16(p->aec_bytes) + 2 * p->vec_bytes + p->spec_bytes +
               ALIGN16(p->stage_bytes) + ALIGN16(p->istft_bytes) +
               (p->use_bridge ? 0 : ALIGN16((size_t)DFN2_N_BINS * sizeof(Complex)));
    return 0;
}

/* The built hop pipeline, all of it inside one aligned block. */
typedef struct Rig {
    void *pool;
    Aec *aec;
    float *unity, *total_gain;        /* the RES gain path: min(1, G_res) */
    Complex *post;                    /* P */
    uint32_t rng;
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
    r->rng = SE_CNG_RNG_SEED;
    if (p->use_bridge) {
        r->bridge = dfn_rate_bridge_init(cur, p->stage_bytes, &p->bridge_cfg);
        if (!r->bridge ||
            dfn_rate_bridge_n_freqs(r->bridge) != p->grid.n_freqs ||
            dfn_rate_bridge_hop_size(r->bridge) != p->grid.hop_size)
            goto fail;
        dfn_hop_init_bridge(&r->hop, r->bridge);
        r->stage = dfn_rate_bridge_stage(r->bridge);
    } else {
        AcStftConfig scfg;
        Complex *stage_out;
        if (p->grid.n_freqs != DFN2_N_BINS || p->grid.hop_size != DFN2_HOP_LEN) goto fail;
        r->stage = dfn_res_stage_init(cur, p->stage_bytes, &p->stage_cfg);
        cur += ALIGN16(p->stage_bytes);
        if (!r->stage) goto fail;
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        r->istft = ac_istft_init(cur, p->istft_bytes, &scfg);
        cur += ALIGN16(p->istft_bytes);
        if (!r->istft) goto fail;
        stage_out = (Complex *)cur;
        memset(stage_out, 0, (size_t)DFN2_N_BINS * sizeof(Complex));
        dfn_hop_init_stage(&r->hop, r->stage, r->istft, stage_out);
    }
    return 0;
fail:
    rig_free(r);
    return -1;
}

/* A refused hop: back to the state of a fresh pipeline, every stage together. */
static void rig_reset(Rig *r) {
    aec_reset(r->aec);
    r->rng = SE_CNG_RNG_SEED;
    dfn_hop_reset(&r->hop);
}

/* One host hop: P = E * min(1, G_res), plus comfort noise on the bins the RES cut when enabled
 * (no denoiser: its gain is the unity vector). Returns 0, or -1 after a refused call (the
 * caller resets). */
static int rig_process(Rig *r, const float *mic, const float *ref, float *out) {
    SeSpecFrame f;
    aec_process_context(r->aec, mic, ref);
    if (se_frame_from_aec(r->aec, &f) != 0 || !f.g_res_ctx) return -1;
    sk_min_f32(r->total_gain, r->unity, f.g_res_ctx, r->n_freqs);
    sk_capply_gain_f32(r->post, f.e, r->total_gain, r->n_freqs);
    if (r->enable_cng && f.cn)
        aec_cng_fill(&r->rng, r->post, f.cn, f.g_res_ctx, r->unity, SE_CNG_NR_GAIN_FLOOR,
                     r->n_freqs);
    return dfn_hop_process(&r->hop, f.e, r->post, out) == 0 ? 0 : -1;
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
                args.sample_rate, dfn_example_delay_mode_name(args.delay_mode),
                args.delay_num_filters, args.fixed_delay_samples);
        return 1;
    }
    printf("%s: host %d Hz, delay profile mode=%s n=%d fixed_delay=%d cng=%d "
           "atten_lim=%.1f dB -> pool %llu bytes (align %lu), model weights "
           "external; model I/O descriptor = compiled-in layout v%u\n", name,
           args.sample_rate, dfn_example_delay_mode_name(args.delay_mode),
           args.delay_num_filters, args.fixed_delay_samples, args.enable_cng,
           (double)args.atten_lim_db, (unsigned long long)plan.total,
           16ul, (unsigned)descriptor.layout_version);

    if (rig_build(&rig, &plan) != 0) {
        fprintf(stderr, "%s: pipeline init failed\n", name);
        return 1;
    }
    hop_size = plan.grid.hop_size;
    /* Host WOLA hop plus the bridge's added delay, or the model's two lookahead hops plus the
     * host hop at 48 kHz. */
    latency = plan.use_bridge
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
