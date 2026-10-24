/* The call-flow examples of pipelines/README.md, compiled and run.
 *
 * Each function below is the code the README shows for one question: chaining
 * a new frequency-domain stage behind the AEC, finishing the hop, the memory a
 * stage needs, and running a neural stage on an accelerator. The checks make
 * the examples tests: the chained mono composition with a unity stage is
 * bit-identical to the hand-written chain of tests/conv_chain_mono.h, and a
 * phased neural hop with the caller's model is bit-identical to the
 * callback-driven one. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/dfn_hop.h"
#include "samples/se_anr_stage.h"
#include "samples/ulcnet_nn_stage.h"
#include "dfn_unit_partition.h"
#include "tests/dfn_test_fixture.h"
#include "mono_aec_nr_res/tests/conv_chain_mono.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

static int g_fail = 0;
#define CHECK(cond, msg) do { \
        if (cond) printf("PASS: %s\n", msg); \
        else { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    } while (0)

static uint32_t g_lcg = 0x2468ACEu;
static float lcg(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return ((float)(int)(g_lcg >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

/* The mono lane every example below starts from. */
static int derive_mono_lane(int sample_rate, int fft_size, SeGrid* grid, AecConfig* aec_cfg) {
    static const SeRules rules = { .allow_8k = 1, .spatial_context = SE_CONTEXT_PRESET };
    SeRequest req;
    memset(&req, 0, sizeof(req));
    req.sample_rate = sample_rate;
    req.fft_size = fft_size;
    req.delay_mode = AEC_DELAY_MATCHED;
    req.delay_num_filters = DA_NUM_FILTERS;
    req.fixed_delay_samples = -1;
    req.aec_preset = AEC_PRESET_BALANCED;
    return se_derive(&req, &rules, grid, aec_cfg);
}

/* ---- Q: how do I chain a new frequency-domain stage? --------------------- */

/* A gain-type stage: its state lives in a block the caller provides, and its
 * _gain() returns n_freqs amplitudes valid until the next call. */
typedef struct HfTilt {
    int bin_hi;       /* bins above this one are attenuated */
    float amount;
    float* g;         /* n_freqs floats, inside the caller's pool */
} HfTilt;

static size_t hf_tilt_mem_size(int n_freqs) {
    return ALIGN16(sizeof(HfTilt)) + ALIGN16((size_t)n_freqs * sizeof(float));
}

static HfTilt* hf_tilt_init(void* mem, int n_freqs, int bin_hi, float amount) {
    HfTilt* s = (HfTilt*)mem;          /* mem: 16-byte aligned, hf_tilt_mem_size() bytes */
    s->bin_hi = bin_hi;
    s->amount = amount;
    s->g = (float*)((uint8_t*)mem + ALIGN16(sizeof(HfTilt)));
    (void)n_freqs;
    return s;
}

static const float* hf_tilt_gain(HfTilt* s, const SeSpecFrame* f) {
    int k;
    for (k = 0; k < f->n_freqs; ++k) s->g[k] = k > s->bin_hi ? s->amount : 1.0f;
    return s->g;
}

/* One hop: the AEC, then the stages written out in the order the product
 * applies them -- the denoiser, the AEC's own residual gain, the new stage --
 * min of the gains, apply, comfort noise, synthesis. Each stage is its own
 * library object in the caller's pool. */
typedef struct Chain {
    Aec* aec;
    AnrDenoiser* nr;
    AcIstft* istft;
    HfTilt* tilt;
    float *g_nr_scratch, *g_min, *g_total;   /* n_freqs floats each */
    Complex* spec;                           /* n_freqs bins */
    uint32_t rng;
    int n_freqs;
} Chain;

static int chain_hop(Chain* c, const float* mic, const float* ref, float* out) {
    SeSpecFrame f;
    const float* nr_gain;
    const float* tilt;
    aec_process_context(c->aec, mic, ref);
    if (se_frame_from_aec(c->aec, &f) != 0) return -1;
    nr_gain = se_anr_stage_gain(c->nr, &f, c->g_nr_scratch, 1);
    if (!nr_gain) return -1;
    tilt = hf_tilt_gain(c->tilt, &f);
    sk_min_f32(c->g_min, nr_gain, f.g_res_ctx, f.n_freqs);
    sk_min_f32(c->g_total, c->g_min, tilt, f.n_freqs);
    sk_capply_gain_f32(c->spec, f.e, c->g_total, f.n_freqs);
    aec_cng_fill(&c->rng, c->spec, f.cn, f.g_res_ctx, nr_gain,
                 SE_CNG_NR_GAIN_FLOOR, f.n_freqs);
    (void)ac_istft_push(c->istft, c->spec, out);
    return 0;
}

static double energy_above(const float* x, int n) {
    double e = 0.0;
    int i;
    for (i = n / 2; i < n; ++i) e += (double)x[i] * x[i];
    return e;
}

/* Builds the chain's objects in one caller block at `mem` (returns the bytes used, 0 on
 * failure). `bin_hi`/`amount` configure the new stage. */
static size_t chain_build(Chain* c, void* mem, size_t bytes, const SeGrid* grid,
                          const AecConfig* aec_cfg, int sample_rate, int bin_hi, float amount) {
    uint8_t* cur = (uint8_t*)mem;
    size_t aec_b = aec_get_mem_size(aec_cfg), nr_b, synth_b, vec, used;
    AnrConfig nc = anr_config_for_mode_grid(sample_rate, grid->fft_size, ANR_MODE_BALANCED);
    AcStftConfig sc;
    int n = grid->n_freqs;
    ac_stft_config_default(&sc, grid->fft_size);
    nr_b = anr_get_mem_size(&nc);
    synth_b = ac_istft_get_mem_size(&sc);
    vec = ALIGN16((size_t)n * sizeof(float));
    used = ALIGN16(aec_b) + ALIGN16(nr_b) + ALIGN16(synth_b) + hf_tilt_mem_size(n) + 3 * vec +
           ALIGN16((size_t)n * sizeof(Complex));
    if (bytes < used) return 0;
    memset(c, 0, sizeof(*c));
    c->n_freqs = n;
    c->rng = SE_CNG_RNG_SEED;
    c->aec = aec_init(cur, aec_b, aec_cfg);            cur += ALIGN16(aec_b);
    c->nr = anr_init(cur, nr_b, &nc);                  cur += ALIGN16(nr_b);
    c->istft = ac_istft_init(cur, synth_b, &sc);       cur += ALIGN16(synth_b);
    c->tilt = hf_tilt_init(cur, n, bin_hi, amount);    cur += hf_tilt_mem_size(n);
    c->g_nr_scratch = (float*)cur;                     cur += vec;
    c->g_min = (float*)cur;                            cur += vec;
    c->g_total = (float*)cur;                          cur += vec;
    c->spec = (Complex*)cur;
    return c->aec && c->nr && c->istft ? used : 0;
}

static size_t chain_bytes(const SeGrid* grid, const AecConfig* aec_cfg, int sample_rate) {
    AnrConfig nc = anr_config_for_mode_grid(sample_rate, grid->fft_size, ANR_MODE_BALANCED);
    AcStftConfig sc;
    int n = grid->n_freqs;
    ac_stft_config_default(&sc, grid->fft_size);
    return ALIGN16(aec_get_mem_size(aec_cfg)) + ALIGN16(anr_get_mem_size(&nc)) +
           ALIGN16(ac_istft_get_mem_size(&sc)) + hf_tilt_mem_size(n) +
           3 * ALIGN16((size_t)n * sizeof(float)) + ALIGN16((size_t)n * sizeof(Complex));
}

static void example_chain_new_stage(void) {
    SeGrid grid;
    AecConfig aec_cfg;
    ConvMonoConfig ocfg = conv_mono_default_config(16000, 256);
    ConvMono* oracle;
    Chain c, c2;
    uint8_t *pool_a, *pool_b;
    size_t bytes;
    int hop, h, i, identical = 1, differs = 0;
    float mic[256], far[256], out_a[256], out_b[256], out_c[256];
    double e_ref = 0.0, e_tilt = 0.0;

    CHECK(derive_mono_lane(16000, 256, &grid, &aec_cfg) == 0, "readme/chain: lane derivation");
    bytes = chain_bytes(&grid, &aec_cfg, 16000);
    pool_a = pool_b = NULL;
    if (posix_memalign((void**)&pool_a, 16, bytes) != 0 ||
        posix_memalign((void**)&pool_b, 16, bytes) != 0) return;
    CHECK(chain_build(&c, pool_a, bytes, &grid, &aec_cfg, 16000, grid.n_freqs, 1.0f) == bytes &&
          chain_build(&c2, pool_b, bytes, &grid, &aec_cfg, 16000, 8, 0.0f) == bytes,
          "readme/chain: the stages in the caller's pools");   /* bin_hi = n: a unity stage; second: cut everything above bin 8 */
    oracle = conv_mono_create(&ocfg);
    CHECK(oracle != NULL, "readme/chain: the hand-written chain builds");
    if (!oracle || !c.aec || !c2.aec) return;
    hop = grid.hop_size;

    for (h = 0; h < 120; ++h) {
        for (i = 0; i < hop; ++i) {
            far[i] = lcg();
            mic[i] = 0.3f * far[i] + 0.05f * lcg();
        }
        if (chain_hop(&c, mic, far, out_a) != 0 ||
            conv_mono_hop(oracle, mic, far, out_b) != 0 ||
            chain_hop(&c2, mic, far, out_c) != 0) { identical = 0; break; }
        if (memcmp(out_a, out_b, (size_t)hop * sizeof(float)) != 0) identical = 0;
        if (memcmp(out_a, out_c, (size_t)hop * sizeof(float)) != 0) differs = 1;
        e_ref += energy_above(out_b, hop);
        e_tilt += energy_above(out_c, hop);
    }
    CHECK(identical, "readme/chain: the chain with a unity stage equals the hand-written chain bit for bit");
    CHECK(differs && e_tilt < 0.5 * e_ref,
          "readme/chain: a real stage changes the output (energy above half-band drops)");
    anr_destroy(c.nr);
    anr_destroy(c2.nr);
    aec_destroy(c.aec);
    aec_destroy(c2.aec);
    conv_mono_destroy(oracle);
    free(pool_a);
    free(pool_b);
}

/* ---- Q: how do I run the NN stage on the accelerator? -------------------- */

/* The accelerator, stood in for by a function writing the heads a model
 * would: a uniform 0.5 band mask. A board binds job->inputs / job->outputs
 * to its tensors instead. */
static int run_npu(DfnResStageJob* job) {
    size_t i;
    for (i = 0; i < job->outputs.erb_mask_elements; ++i) job->outputs.erb_mask[i] = 0.5f;
    memset(job->outputs.coefs, 0, job->outputs.coefs_elements * sizeof(float));
    job->outputs.alpha[0] = 0.0f;
    return 0;                                   /* 0 = every output written */
}

static int npu_callback(void* user, const DFN2PrepostInputs* in, DFN2PrepostOutputs* out) {
    DfnResStageJob job;
    (void)user;
    job.inputs = *in;
    job.outputs = *out;
    return run_npu(&job);
}

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB], g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];

/* One DFN2 host: the AEC (its own RES gain), P = E * min(1, G_res) without denoiser or comfort
 * noise (E and P must share a phase), and the rate bridge with its DfnHop. */
typedef struct DfnHost {
    Aec* aec;
    float *unity, *total;
    Complex* post;
    int n_freqs;
    DfnRateBridge* bridge;
    DfnHop hop;
    uint8_t* mem;
} DfnHost;

static int dfn_host_build(DfnHost* d, const DFN2Model* model) {
    SeGrid grid;
    AecConfig aec_cfg;
    DfnRateBridgeConfig bc = dfn_rate_bridge_default_config(16000);
    DfnRateBridgeMemReq breq;
    size_t aec_b, vec, cvec;
    uint8_t* cur;
    int k;
    memset(d, 0, sizeof(*d));
    if (derive_mono_lane(16000, 0, &grid, &aec_cfg) != 0) return -1;
    bc.stage.erb_fwd = g_erb_fwd;
    bc.stage.erb_inv = g_erb_inv;
    bc.stage.model = *model;
    aec_b = aec_get_mem_size(&aec_cfg);
    vec = ALIGN16((size_t)grid.n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)grid.n_freqs * sizeof(Complex));
    if (dfn_rate_bridge_get_mem_requirements(&bc, &breq) != 0) return -1;
    if (posix_memalign((void**)&d->mem, 16, ALIGN16(aec_b) + 2 * vec + cvec + (size_t)breq.bytes) != 0)
        return -1;
    cur = d->mem;
    d->aec = aec_init(cur, aec_b, &aec_cfg);  cur += ALIGN16(aec_b);
    d->unity = (float*)cur;                    cur += vec;
    d->total = (float*)cur;                    cur += vec;
    d->post = (Complex*)cur;                   cur += cvec;
    d->n_freqs = grid.n_freqs;
    for (k = 0; k < d->n_freqs; ++k) d->unity[k] = 1.0f;
    d->bridge = dfn_rate_bridge_init(cur, (size_t)breq.bytes, &bc);
    if (!d->aec || !d->bridge) return -1;
    dfn_hop_init_bridge(&d->hop, d->bridge);
    return 0;
}

static void dfn_host_free(DfnHost* d) {
    if (d->bridge) dfn_rate_bridge_destroy(d->bridge);
    if (d->aec) aec_destroy(d->aec);
    free(d->mem);
}

/* AEC, RES gain, P; the DFN hop begins on (E, P). */
static int dfn_host_begin(DfnHost* d, const float* mic, const float* ref) {
    SeSpecFrame f;
    aec_process_context(d->aec, mic, ref);
    if (se_frame_from_aec(d->aec, &f) != 0 || !f.g_res_ctx) return -1;
    sk_min_f32(d->total, d->unity, f.g_res_ctx, d->n_freqs);
    sk_capply_gain_f32(d->post, f.e, d->total, d->n_freqs);
    return dfn_hop_begin(&d->hop, f.e, d->post) == 0 ? 0 : -1;
}

static void example_dfn_phased(void) {
    DFN2Model none, cb_model;
    DFN2ModelIoDescriptor descriptor;
    DfnHost p, cb;
    float mic[256], ref[256], out[256], out_cb[256];
    DfnResStageJob job;
    int hop, h, i, identical = 1, frames = 0;

    dfn_unit_partition(g_erb_fwd, g_erb_inv);
    dfn2_model_io_descriptor_default(&descriptor);
    memset(&none, 0, sizeof(none));            /* no model attached: the board runs the NPU */
    cb_model = none;
    cb_model.infer = npu_callback;             /* the callback-driven twin */
    cb_model.io_descriptor = &descriptor;
    CHECK(dfn_host_build(&p, &none) == 0 && dfn_host_build(&cb, &cb_model) == 0,
          "readme/dfn: both hosts build");   /* 16 kHz: the rate bridge, up to two frames per hop */
    if (!p.bridge || !cb.bridge) return;
    hop = dfn_rate_bridge_hop_size(p.bridge);
    for (h = 0; h < 80; ++h) {
        int r;
        for (i = 0; i < hop; ++i) { ref[i] = lcg(); mic[i] = 0.4f * ref[i] + 0.02f * lcg(); }

        dfn_host_begin(&p, mic, ref);                               /* AEC, RES, stage input   */
        while ((r = dfn_hop_nn_begin(&p.hop, &job)) == 1) {         /* a model frame is open   */
            dfn_hop_nn_end(&p.hop, run_npu(&job));                  /* commit, or the identity */
            ++frames;
        }
        dfn_hop_end(&p.hop, out);                                   /* synthesis, one hop      */

        dfn_host_begin(&cb, mic, ref);
        while ((r = dfn_hop_nn_begin(&cb.hop, &job)) == 1)
            dfn_hop_nn_end(&cb.hop, dfn_res_stage_infer(dfn_hop_stage(&cb.hop), &job));
        dfn_hop_end(&cb.hop, out_cb);
        if (r < 0 || memcmp(out, out_cb, (size_t)hop * sizeof(float)) != 0) identical = 0;
    }
    CHECK(frames > 0 && identical,
          "readme/dfn: begin / nn_begin / nn_end / end equals the callback-driven hop");
    dfn_host_free(&p);
    dfn_host_free(&cb);
}

static int half_infer(void* user, const float* er, const float* ei, const float* fr,
                      const float* fi, float* or_, float* oi) {
    int k;
    (void)user; (void)fr; (void)fi;
    for (k = 0; k < ULCNET_BINS; ++k) { or_[k] = 0.5f * er[k]; oi[k] = 0.5f * ei[k]; }
    return 0;
}

static int run_ulcnet_npu(const UlcnetNnJob* job) {
    return half_infer(NULL, job->err_re, job->err_im, job->far_re, job->far_im,
                      job->out_re, job->out_im);
}

/* The mono linear AEC (linear context only) feeding the stage: the alignment events are the
 * AEC's delay state. */
typedef struct UlcHost {
    Aec* aec;
    UlcnetNnStage* stage;
    uint8_t* mem;
} UlcHost;

static int ulc_host_build(UlcHost* u, const UlcnetModel* model) {
    static const SeRules rules = { .fixed_rate = ULCNET_SR, .fixed_fft = ULCNET_N_FFT,
                                   .spatial_context = AEC_CONTEXT_LINEAR_ONLY };
    SeRequest req;
    SeGrid grid;
    AecConfig aec_cfg;
    size_t aec_b, stage_b = ulcnet_nn_stage_mem_size(1);
    memset(u, 0, sizeof(*u));
    memset(&req, 0, sizeof(req));
    req.sample_rate = ULCNET_SR;
    req.fft_size = ULCNET_N_FFT;
    req.delay_mode = AEC_DELAY_MATCHED;
    req.delay_num_filters = DA_NUM_FILTERS;
    req.fixed_delay_samples = -1;
    req.aec_preset = AEC_PRESET_BALANCED;
    if (se_derive(&req, &rules, &grid, &aec_cfg) != 0) return -1;
    aec_b = aec_get_mem_size(&aec_cfg);
    if (posix_memalign((void**)&u->mem, 16, ALIGN16(aec_b) + ALIGN16(stage_b)) != 0) return -1;
    u->aec = aec_init(u->mem, aec_b, &aec_cfg);
    u->stage = ulcnet_nn_stage_init(u->mem + ALIGN16(aec_b), stage_b, 1);
    if (!u->aec || !u->stage || ulcnet_nn_stage_set_model(u->stage, model) != 0) return -1;
    return 0;
}

static void ulc_host_free(UlcHost* u) {
    if (u->aec) aec_destroy(u->aec);
    free(u->mem);
}

/* AEC, then the stage's begin: the lane's formed hop, the raw reference, the delay events. */
static int ulc_host_begin(UlcHost* u, const float* mic, const float* ref) {
    AecResContext rctx;
    AecLinearContext lctx;
    UlcnetNnDelay delay;
    aec_process_context(u->aec, mic, ref);
    aec_get_res_context(u->aec, &rctx);
    aec_get_linear_context(u->aec, &lctx);
    delay.changed = lctx.delay_state == AEC_LINEAR_DELAY_CHANGED;
    delay.solid = lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
    return ulcnet_nn_stage_begin_time(u->stage, rctx.formed_hop, ref, delay);
}

static void example_ulcnet_phased(void) {
    UlcnetModelIoDescriptor d;
    UlcnetModel none, cb_model;
    UlcHost p, cb;
    float mic[ULCNET_HOP], ref[ULCNET_HOP], out[ULCNET_HOP], out_cb[ULCNET_HOP];
    UlcnetNnJob job;
    int h, i, identical = 1, ran = 0;

    ulcnet_model_io_descriptor_default(8, &d);
    memset(&none, 0, sizeof(none));            /* no infer callback: the board runs the network */
    cb_model = none;
    cb_model.infer = half_infer;
    cb_model.io_descriptor = &d;
    CHECK(ulc_host_build(&p, &none) == 0 && ulc_host_build(&cb, &cb_model) == 0,
          "readme/ulcnet: both hosts build");
    if (!p.stage || !cb.stage) return;
    for (h = 0; h < 60; ++h) {
        for (i = 0; i < ULCNET_HOP; ++i) { ref[i] = lcg(); mic[i] = 0.3f * ref[i] + 0.05f * lcg(); }

        ulc_host_begin(&p, mic, ref);                                /* AEC, both analyses        */
        if (ulcnet_nn_stage_nn_begin(p.stage, &job) > 0) {           /* a frame to run            */
            ulcnet_nn_stage_nn_end(p.stage, run_ulcnet_npu(&job));   /* accept, or the identity   */
            ++ran;
        }
        ulcnet_nn_stage_end(p.stage, out);                           /* WOLA, one hop             */

        ulc_host_begin(&cb, mic, ref);
        ulcnet_nn_stage_process(cb.stage, out_cb);                   /* the callback-driven hop   */
        if (memcmp(out, out_cb, sizeof(out)) != 0) identical = 0;
    }
    CHECK(ran > 0 && identical,
          "readme/ulcnet: begin / nn_begin / nn_end / end equals the callback-driven hop");
    ulc_host_free(&p);
    ulc_host_free(&cb);
}

int main(void) {
    example_chain_new_stage();
    example_dfn_phased();
    example_ulcnet_phased();
    if (g_fail) { fprintf(stderr, "%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
