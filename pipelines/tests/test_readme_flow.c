/* The call-flow examples of pipelines/reference/call-flow.md, compiled and run.
 *
 * Each function below is the code call-flow.md shows for one question: chaining
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
#include "anr.h"
#include "dfn/dfn_hop.h"
#include "ulcnet/ulcnet_nn_stage.h"
#include "dfn/dfn_unit_partition.h"
#include "dfn/tests/dfn_test_fixture.h"
#include "mono/aec_nr_res/tests/conv_chain_mono.h"
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

#define PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))   /* int16^2 -> audio scale, exact */

/* The mono lane every example below starts from: the grid (fft_size 0: the rate's default) and
 * the AEC keeping its own gain in the residual view (AEC_RESIDUAL_FULL). */
static int derive_mono_lane(int sample_rate, int fft_size, AecSignalGrid* grid,
                            AecConfig* aec_cfg) {
    if (aec_resolve_signal_grid(sample_rate,
                                 fft_size ? fft_size : aec_default_fft_size(sample_rate), grid) != AEC_OK)
        return -1;
    aec_config_from_level(aec_cfg, AEC_LEVEL_BALANCED, sample_rate);
    aec_cfg->fft_size = grid->fft_size;
    aec_cfg->delay_mode = AEC_DELAY_MATCHED;
    aec_cfg->delay_num_filters = DA_NUM_FILTERS;
    aec_cfg->fixed_delay_samples = -1;
    aec_cfg->residual = AEC_RESIDUAL_FULL;
    return 0;
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

static const float* hf_tilt_gain(HfTilt* s, int n_freqs) {
    int k;
    for (k = 0; k < n_freqs; ++k) s->g[k] = k > s->bin_hi ? s->amount : 1.0f;
    return s->g;
}

/* One hop: the AEC, then the stages written out in the order the product
 * applies them -- the denoiser, the AEC's own residual gain, the new stage --
 * the extra gain handed to aec_apply_gain / aec_apply_cng, comfort noise, synthesis. Each stage is its own
 * library object in the caller's pool. */
typedef struct Chain {
    Aec* aec;
    AnrDenoiser* nr;
    AcIstft* istft;
    HfTilt* tilt;
    float *g_nr_scratch, *g_extra;           /* n_freqs floats each */
    Complex* spec;                           /* n_freqs bins */
    uint32_t reset_serial;                   /* the aec_post_reset_serial() value already handled */
    int n_freqs;
} Chain;

static int chain_hop(Chain* c, const float* mic, const float* ref, float* out) {
    AecResidualView f;
    const float* residual_psd_audio = NULL;
    const float* nr_gain;
    const float* tilt;
    uint32_t serial;
    if (aec_process_linear(c->aec, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(c->aec, NULL) != AEC_OK) return -1;
    memset(&f, 0, sizeof(f));
    f.struct_size = sizeof(f);
    if (aec_get_residual(c->aec, &f) != AEC_OK) return -1;   /* f.base_spec: the spectrum the gains are applied to */
    serial = aec_post_reset_serial(c->aec);
    if (serial != c->reset_serial) {        /* hard post-chain reset inside this hop: before the push */
        c->reset_serial = serial;
        ac_istft_reset(c->istft);
    }
    if (f.r2 && f.usable_linear) {         /* residual-echo powers into the noise floor */
        sk_scale_f32(c->g_nr_scratch, f.r2, PSD_SCALE_INV, f.n_freqs);
        residual_psd_audio = c->g_nr_scratch;
    }
    nr_gain = anr_process_gain_with_residual_psd(c->nr, f.base_spec, residual_psd_audio);
    if (!nr_gain) return -1;
    tilt = hf_tilt_gain(c->tilt, f.n_freqs);
    sk_min_f32(c->g_extra, nr_gain, tilt, f.n_freqs);        /* every stage but the AEC's own gain */
    if (aec_apply_gain(c->aec, c->g_extra, c->spec) != AEC_OK) return -1;
    if (aec_apply_cng(c->aec, c->g_extra, c->spec) != AEC_OK) return -1;
    (void)ac_istft_push(c->istft, c->spec, out);
    return 0;
}

static double energy_above(const float* x, int n) {
    double e = 0.0;
    int i;
    for (i = n / 2; i < n; ++i) e += (double)x[i] * x[i];
    return e;
}

/* The synthesis borrows the AEC's own window table (aec_synthesis_window()), which exists once the AEC is
 * initialised, after the block is sized. A borrowed table carves nothing, so any non-NULL pointer sizes the
 * synthesis the same; this one stands in until chain_build() installs the AEC's. */
static const float k_sizing_window[1] = {0.0f};

/* Builds the chain's objects in one caller block at `mem` (returns the bytes used, 0 on
 * failure). `bin_hi`/`amount` configure the new stage. */
static size_t chain_build(Chain* c, void* mem, size_t bytes, const AecSignalGrid* grid,
                          const AecConfig* aec_cfg, int sample_rate, int bin_hi, float amount) {
    uint8_t* cur = (uint8_t*)mem;
    size_t aec_b = aec_get_mem_size(aec_cfg), nr_b, synth_b, vec, used;
    AnrConfig nc = anr_config_for_level_grid(sample_rate, grid->fft_size, ANR_LEVEL_BALANCED);
    AcStftConfig sc;
    int n = grid->n_freqs, win_n = 0;
    ac_stft_config_default(&sc, grid->fft_size);
    sc.window = k_sizing_window;
    nr_b = anr_get_mem_size(&nc);
    synth_b = ac_istft_get_mem_size(&sc);
    vec = ALIGN16((size_t)n * sizeof(float));
    used = ALIGN16(aec_b) + ALIGN16(nr_b) + ALIGN16(synth_b) + hf_tilt_mem_size(n) + 2 * vec +
           ALIGN16((size_t)n * sizeof(Complex));
    if (bytes < used) return 0;
    memset(c, 0, sizeof(*c));
    c->n_freqs = n;
    c->aec = aec_init(cur, aec_b, aec_cfg);            cur += ALIGN16(aec_b);
    c->reset_serial = c->aec ? aec_post_reset_serial(c->aec) : 0u;
    c->nr = anr_init(cur, nr_b, &nc);                  cur += ALIGN16(nr_b);
    /* the AEC's own table: the chain then equals aec_process() */
    if (c->aec && aec_synthesis_window(c->aec, &sc.window, &win_n) == 0 && win_n == sc.fft_size)
        c->istft = ac_istft_init(cur, synth_b, &sc);
    cur += ALIGN16(synth_b);
    c->tilt = hf_tilt_init(cur, n, bin_hi, amount);    cur += hf_tilt_mem_size(n);
    c->g_nr_scratch = (float*)cur;                     cur += vec;
    c->g_extra = (float*)cur;                          cur += vec;
    c->spec = (Complex*)cur;
    return c->aec && c->nr && c->istft ? used : 0;
}

static size_t chain_bytes(const AecSignalGrid* grid, const AecConfig* aec_cfg, int sample_rate) {
    AnrConfig nc = anr_config_for_level_grid(sample_rate, grid->fft_size, ANR_LEVEL_BALANCED);
    AcStftConfig sc;
    int n = grid->n_freqs;
    ac_stft_config_default(&sc, grid->fft_size);
    sc.window = k_sizing_window;   /* borrowed, as chain_build() */
    return ALIGN16(aec_get_mem_size(aec_cfg)) + ALIGN16(anr_get_mem_size(&nc)) +
           ALIGN16(ac_istft_get_mem_size(&sc)) + hf_tilt_mem_size(n) +
           2 * ALIGN16((size_t)n * sizeof(float)) + ALIGN16((size_t)n * sizeof(Complex));
}

static void example_chain_new_stage(void) {
    AecSignalGrid grid;
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

/* One DFN2 host: the AEC (its own RES gain), P = B * G_res without denoiser or comfort
 * noise (B and P must share a phase), and the rate bridge with its DfnHop. */
typedef struct DfnHost {
    Aec* aec;
    Complex* post;
    DfnRateBridge* bridge;
    DfnHop hop;
    uint8_t* mem;
} DfnHost;

static int dfn_host_build(DfnHost* d, const DFN2Model* model) {
    AecSignalGrid grid;
    AecConfig aec_cfg;
    DfnRateBridgeConfig bc = dfn_rate_bridge_default_config(16000, dfn_test_grid());
    DfnRateBridgeMemReq breq;
    size_t aec_b, cvec;
    uint8_t* cur;
    memset(d, 0, sizeof(*d));
    if (derive_mono_lane(16000, 0, &grid, &aec_cfg) != 0) return -1;
    bc.stage.erb_fwd = g_erb_fwd;
    bc.stage.erb_inv = g_erb_inv;
    bc.stage.model = *model;
    aec_b = aec_get_mem_size(&aec_cfg);
    cvec = ALIGN16((size_t)grid.n_freqs * sizeof(Complex));
    if (dfn_rate_bridge_get_mem_requirements(&bc, &breq) != 0) return -1;
    if (posix_memalign((void**)&d->mem, 16, ALIGN16(aec_b) + cvec + (size_t)breq.bytes) != 0)
        return -1;
    cur = d->mem;
    d->aec = aec_init(cur, aec_b, &aec_cfg);  cur += ALIGN16(aec_b);
    d->post = (Complex*)cur;                   cur += cvec;
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

/* AEC, RES gain, P; the DFN hop begins on (B, P). */
static int dfn_host_begin(DfnHost* d, const float* mic, const float* ref) {
    AecResidualView f;
    if (aec_process_linear(d->aec, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(d->aec, NULL) != AEC_OK) return -1;
    memset(&f, 0, sizeof(f));
    f.struct_size = sizeof(f);
    if (aec_get_residual(d->aec, &f) != AEC_OK) return -1;
    if (aec_apply_gain(d->aec, NULL, d->post) != AEC_OK) return -1;
    return dfn_hop_begin(&d->hop, f.base_spec, d->post) == 0 ? 0 : -1;
}

static void example_dfn_phased(void) {
    DFN2Model none, cb_model;
    DFN2ModelIoDescriptor descriptor;
    DfnHost p, cb;
    float mic[256], ref[256], out[256], out_cb[256];
    DfnResStageJob job;
    int hop, h, i, identical = 1, frames = 0;

    dfn_unit_partition(g_erb_fwd, g_erb_inv, dfn_test_grid());
    dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor);
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

/* The shipped Align-ULCNet grid (hop 256) and its export delay depth: the grid the stage, the lane and the
 * model descriptor below are built for. */
static const UlcnetGridSpec k_ulcnet_spec = { .sample_rate = 16000, .n_fft = 512, .delay_depth = 8 };
enum { ULC_HOP_MAX = 256, BINS_MAX_ULC = 257 };
static UlcnetGrid g_ulc_grid;

static int half_infer(void* user, const float* er, const float* ei, const float* fr,
                      const float* fi, float* or_, float* oi) {
    const UlcnetGrid* grid = (const UlcnetGrid*)user;
    int k;
    (void)fr; (void)fi;
    for (k = 0; k < grid->n_bins; ++k) { or_[k] = 0.5f * er[k]; oi[k] = 0.5f * ei[k]; }
    return 0;
}

static int run_ulcnet_npu(const UlcnetNnJob* job) {
    return half_infer(&g_ulc_grid, job->err_re, job->err_im, job->far_re, job->far_im,
                      job->out_re, job->out_im);
}

/* The mono linear AEC (linear context only) feeding the stage: the alignment events are the
 * AEC's delay state. */
typedef struct UlcHost {
    Aec* aec;
    UlcnetNnStage* stage;
    AcStft *far_stft, *err_stft;    /* the application's analyses and synthesis */
    AcIstft* istft;
    Complex far_spec[BINS_MAX_ULC], err_spec[BINS_MAX_ULC], out_spec[BINS_MAX_ULC];
    uint8_t* mem;
} UlcHost;

static int ulc_host_build(UlcHost* u, const UlcnetModel* model) {
    AecConfig aec_cfg;
    AcStftConfig sc;
    size_t aec_b, stage_b = ulcnet_nn_stage_mem_size(&g_ulc_grid), stft_b, istft_b;
    uint8_t* cur;
    memset(u, 0, sizeof(*u));
    ac_stft_config_default(&sc, g_ulc_grid.n_fft);
    sc.ola = AC_OLA_ACCUMULATE;
    stft_b = ac_stft_get_mem_size(&sc);
    istft_b = ac_istft_get_mem_size(&sc);
    /* the stage's grid; the lane keeps the state of the linear residual only */
    aec_config_from_level(&aec_cfg, AEC_LEVEL_BALANCED, g_ulc_grid.sample_rate);
    aec_cfg.fft_size = g_ulc_grid.n_fft;
    aec_cfg.delay_mode = AEC_DELAY_MATCHED;
    aec_cfg.delay_num_filters = DA_NUM_FILTERS;
    aec_cfg.fixed_delay_samples = -1;
    aec_cfg.residual = AEC_RESIDUAL_STATE;
    aec_b = aec_get_mem_size(&aec_cfg);
    if (posix_memalign((void**)&u->mem, 16,
                       ALIGN16(aec_b) + ALIGN16(stage_b) + 2 * ALIGN16(stft_b) + ALIGN16(istft_b)) != 0) return -1;
    u->aec = aec_init(u->mem, aec_b, &aec_cfg);
    cur = u->mem + ALIGN16(aec_b);
    u->stage = ulcnet_nn_stage_init(cur, stage_b, &g_ulc_grid);
    cur += ALIGN16(stage_b);
    u->far_stft = ac_stft_init(cur, stft_b, &sc);
    cur += ALIGN16(stft_b);
    u->err_stft = ac_stft_init(cur, stft_b, &sc);
    cur += ALIGN16(stft_b);
    u->istft = ac_istft_init(cur, istft_b, &sc);
    if (!u->aec || !u->stage || !u->far_stft || !u->err_stft || !u->istft ||
        ulcnet_nn_stage_set_model(u->stage, model) != 0) return -1;
    return 0;
}

static void ulc_host_free(UlcHost* u) {
    if (u->aec) aec_destroy(u->aec);
    free(u->mem);
}

/* AEC, then the application's analyses of the lane's formed hop and the raw reference, then the
 * stage's begin_spec with the delay events. */
static int ulc_host_begin(UlcHost* u, const float* mic, const float* ref) {
    AecResidualView rv;
    AecLinearView lv;
    UlcnetNnDelay delay;
    if (aec_process_linear(u->aec, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(u->aec, NULL) != AEC_OK) return -1;
    memset(&rv, 0, sizeof(rv));
    rv.struct_size = sizeof(rv);
    memset(&lv, 0, sizeof(lv));
    lv.struct_size = sizeof(lv);
    if (aec_get_residual(u->aec, &rv) != AEC_OK || aec_get_linear(u->aec, &lv) != AEC_OK) return -1;
    delay.changed = lv.delay_state == AEC_LINEAR_DELAY_CHANGED;
    delay.solid = lv.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
    (void)ac_stft_push(u->err_stft, rv.formed_hop, u->err_spec);
    (void)ac_stft_push(u->far_stft, ref, u->far_spec);
    return ulcnet_nn_stage_begin_spec(u->stage, u->err_spec, u->far_spec, delay);
}

static void example_ulcnet_phased(void) {
    UlcnetModelIoDescriptor d;
    UlcnetModel none, cb_model;
    UlcHost p, cb;
    float mic[ULC_HOP_MAX], ref[ULC_HOP_MAX], out[ULC_HOP_MAX], out_cb[ULC_HOP_MAX];
    UlcnetNnJob job;
    int hop, h, i, identical = 1, ran = 0;

    CHECK(ulcnet_grid_make(&g_ulc_grid, &k_ulcnet_spec) == ULCNET_GRID_OK &&
          g_ulc_grid.hop == ULC_HOP_MAX && ulcnet_model_io_descriptor_default(&g_ulc_grid, &d) == 0,
          "readme/ulcnet: the grid and its descriptor");
    if (g_ulc_grid.hop != ULC_HOP_MAX) return;
    hop = g_ulc_grid.hop;
    memset(&none, 0, sizeof(none));            /* no infer callback: the board runs the network */
    cb_model = none;
    cb_model.infer = half_infer;
    cb_model.user = &g_ulc_grid;
    cb_model.io_descriptor = &d;
    CHECK(ulc_host_build(&p, &none) == 0 && ulc_host_build(&cb, &cb_model) == 0,
          "readme/ulcnet: both hosts build");
    if (!p.stage || !cb.stage) return;
    for (h = 0; h < 60; ++h) {
        for (i = 0; i < hop; ++i) { ref[i] = lcg(); mic[i] = 0.3f * ref[i] + 0.05f * lcg(); }

        ulc_host_begin(&p, mic, ref);                                /* AEC, analyses, begin_spec */
        if (ulcnet_nn_stage_nn_begin(p.stage, &job) > 0) {           /* a frame to run            */
            ulcnet_nn_stage_nn_end(p.stage, run_ulcnet_npu(&job));   /* accept, or the identity   */
            ++ran;
        }
        ulcnet_nn_stage_end_spec(p.stage, p.out_spec);               /* the output spectrum       */
        ac_istft_push(p.istft, p.out_spec, out);                     /* the application's WOLA    */

        ulc_host_begin(&cb, mic, ref);
        ulcnet_nn_stage_process(cb.stage, cb.out_spec);              /* the callback-driven hop   */
        ac_istft_push(cb.istft, cb.out_spec, out_cb);
        if (memcmp(out, out_cb, (size_t)hop * sizeof(float)) != 0) identical = 0;
    }
    CHECK(ran > 0 && identical,
          "readme/ulcnet: begin_spec / nn_begin / nn_end / end_spec equals the callback-driven hop");
    ulc_host_free(&p);
    ulc_host_free(&cb);
}

int main(void) {
    dfn_test_report_pair("test_readme_flow", -1);
    example_chain_new_stage();
    example_dfn_phased();
    example_ulcnet_phased();
    if (g_fail) { fprintf(stderr, "%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
