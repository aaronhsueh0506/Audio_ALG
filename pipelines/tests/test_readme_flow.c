/* The call-flow examples of pipelines/README.md, compiled and run.
 *
 * Each function below is the code the README shows for one question: chaining
 * a new frequency-domain stage, finishing the hop, the memory a stage needs,
 * and running a neural stage on an accelerator. The checks make the examples
 * tests: the chained mono pipeline with a unity stage is bit-identical to
 * audio_pipeline_process(), and a phased neural hop with the caller's model
 * is bit-identical to the callback-driven one. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_pipeline.h"
#include "audio_pipeline_dfn.h"
#include "audio_pipeline_ulcnet.h"
#include "dfn_unit_partition.h"
#include "tests/dfn_test_fixture.h"
#include "common/se_anr_stage.h"
#include "mem_align.h"
#include "simd_kernels.h"

static int g_fail = 0;
#define CHECK(cond, msg) do { \
        if (cond) printf("PASS: %s\n", (msg)); \
        else { fprintf(stderr, "FAIL: %s\n", (msg)); g_fail++; } \
    } while (0)

static uint32_t g_lcg = 0x2468ACEu;
static float lcg(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return ((float)(int)(g_lcg >> 9) / 4194304.0f - 1.0f) * 0.25f;
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
 * applies them -- the denoiser (when the pipeline owns one), the AEC's own
 * residual gain, the new stage -- min of the gains, apply, comfort noise,
 * synthesis. */
typedef struct Chain {
    AudioPipeline* p;
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
    if (audio_pipeline_process_spec(c->p, mic, ref, &f) != 0) return -1;
    nr_gain = se_anr_stage_gain(audio_pipeline_get_nr(c->p), &f, c->g_nr_scratch, 1);
    tilt = hf_tilt_gain(c->tilt, &f);
    sk_min_f32(c->g_min, nr_gain, f.g_res_ctx, f.n_freqs);
    sk_min_f32(c->g_total, c->g_min, tilt, f.n_freqs);
    sk_capply_gain_f32(c->spec, f.e, c->g_total, f.n_freqs);
    aec_cng_fill(&c->rng, c->spec, f.cn, f.g_res_ctx, nr_gain,
                 SE_CNG_NR_GAIN_FLOOR, f.n_freqs);
    return audio_pipeline_synthesize(c->p, c->spec, out);
}

static double energy_above(const float* x, int n) {
    double e = 0.0;
    int i;
    for (i = n / 2; i < n; ++i) e += (double)x[i] * x[i];
    return e;
}

static void example_chain_new_stage(void) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
    AudioPipelineMemReq req;
    AudioPipeline *p, *ref_p, *p2;
    Chain c, c2;
    uint8_t *pool, *stage_mem;
    int hop, n, h, i, identical = 1, differs = 0;
    float mic[256], far[256], out_a[256], out_b[256], out_c[256];
    double e_ref = 0.0, e_tilt = 0.0;
    size_t stage_bytes;

    cfg.fft_size = 256;
    CHECK(audio_pipeline_get_mem_requirements(&cfg, &req) == 0, "readme/chain: pool query");
    /* One caller block: the pipeline's pool, then the stage's. */
    n = 256 / 2 + 1;
    stage_bytes = hf_tilt_mem_size(n) + 3 * ALIGN16((size_t)n * sizeof(float)) +
                  ALIGN16((size_t)n * sizeof(Complex));
    pool = NULL;
    if (posix_memalign((void**)&pool, 16, ALIGN16((size_t)req.bytes + 2 * stage_bytes)) != 0) return;
    stage_mem = pool + ALIGN16((size_t)req.bytes);
    p = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p != NULL && pool != NULL, "readme/chain: pipeline in the caller's pool");
    if (!p) return;
    hop = audio_pipeline_hop_size(p);
    c.p = p; c.n_freqs = n; c.rng = SE_CNG_RNG_SEED;
    c.tilt = hf_tilt_init(stage_mem, n, n, 1.0f);   /* bin_hi = n: a unity stage */
    c.g_nr_scratch = (float*)(stage_mem + hf_tilt_mem_size(n));
    c.g_min = (float*)((uint8_t*)c.g_nr_scratch + ALIGN16((size_t)n * sizeof(float)));
    c.g_total = (float*)((uint8_t*)c.g_min + ALIGN16((size_t)n * sizeof(float)));
    c.spec = (Complex*)((uint8_t*)c.g_total + ALIGN16((size_t)n * sizeof(float)));

    ref_p = audio_pipeline_create(&cfg);
    p2 = audio_pipeline_create(&cfg);
    c2 = c; c2.p = p2;
    c2.tilt = hf_tilt_init(stage_mem + stage_bytes, n, 8, 0.0f);   /* cut everything above bin 8 */
    c2.g_nr_scratch = (float*)((uint8_t*)c2.tilt + hf_tilt_mem_size(n));
    c2.g_min = (float*)((uint8_t*)c2.g_nr_scratch + ALIGN16((size_t)n * sizeof(float)));
    c2.g_total = (float*)((uint8_t*)c2.g_min + ALIGN16((size_t)n * sizeof(float)));
    c2.spec = (Complex*)((uint8_t*)c2.g_total + ALIGN16((size_t)n * sizeof(float)));
    c2.rng = SE_CNG_RNG_SEED;

    for (h = 0; h < 120; ++h) {
        for (i = 0; i < hop; ++i) {
            far[i] = lcg();
            mic[i] = 0.3f * far[i] + 0.05f * lcg();
        }
        if (chain_hop(&c, mic, far, out_a) != 0 ||
            audio_pipeline_process(ref_p, mic, far, out_b) != 0 ||
            chain_hop(&c2, mic, far, out_c) != 0) { identical = 0; break; }
        if (memcmp(out_a, out_b, (size_t)hop * sizeof(float)) != 0) identical = 0;
        if (memcmp(out_a, out_c, (size_t)hop * sizeof(float)) != 0) differs = 1;
        e_ref += energy_above(out_b, hop);
        e_tilt += energy_above(out_c, hop);
    }
    CHECK(identical, "readme/chain: the chain with a unity stage equals audio_pipeline_process() bit for bit");
    CHECK(differs && e_tilt < 0.5 * e_ref,
          "readme/chain: a real stage changes the output (energy above half-band drops)");
    audio_pipeline_destroy(p);
    audio_pipeline_destroy(ref_p);
    audio_pipeline_destroy(p2);
    free(pool);
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

static void example_dfn_phased(void) {
    MonoAecDfnResConfig cfg = mono_aec_dfn_res_default_config();
    MonoAecDfnResConfig cb_cfg;
    DFN2ModelIoDescriptor descriptor;
    MonoAecDfnRes *p, *cb;
    float mic[256], ref[256], out[256], out_cb[256];
    DfnResStageJob job;
    int hop, h, i, identical = 1, frames = 0;

    dfn_unit_partition(g_erb_fwd, g_erb_inv);
    dfn2_model_io_descriptor_default(&descriptor);
    cfg.host.sample_rate = 16000;          /* below 48 kHz: the rate bridge, up to two frames per hop */
    cfg.erb_fwd = g_erb_fwd;
    cfg.erb_inv = g_erb_inv;
    cb_cfg = cfg;
    cb_cfg.model.infer = npu_callback;      /* the callback-driven twin */
    cb_cfg.model.io_descriptor = &descriptor;
    p = mono_aec_dfn_res_create(&cfg);      /* no model attached: the board runs the NPU */
    cb = mono_aec_dfn_res_create(&cb_cfg);
    CHECK(p && cb, "readme/dfn: both pipelines create");
    if (!p || !cb) return;
    hop = mono_aec_dfn_res_hop_size(p);
    for (h = 0; h < 80; ++h) {
        int r;
        for (i = 0; i < hop; ++i) { ref[i] = lcg(); mic[i] = 0.4f * ref[i] + 0.02f * lcg(); }

        mono_aec_dfn_res_process_begin(p, mic, ref);              /* AEC, RES, stage input   */
        while ((r = mono_aec_dfn_res_nn_begin(p, &job)) == 1) {   /* a model frame is open   */
            mono_aec_dfn_res_nn_end(p, run_npu(&job));            /* commit, or the identity */
            ++frames;
        }
        mono_aec_dfn_res_process_end(p, out);                     /* synthesis, one hop      */

        mono_aec_dfn_res_process(cb, mic, ref, out_cb);
        if (r < 0 || memcmp(out, out_cb, (size_t)hop * sizeof(float)) != 0) identical = 0;
    }
    CHECK(frames > 0 && identical,
          "readme/dfn: begin / nn_begin / nn_end / process_end equals the callback-driven pipeline");
    mono_aec_dfn_res_destroy(p);
    mono_aec_dfn_res_destroy(cb);
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

static void example_ulcnet_phased(void) {
    UlcnetModelIoDescriptor d;
    AudioPipelineUlcnetConfig cfg = audio_pipeline_ulcnet_default_config(ULCNET_SR);
    AudioPipelineUlcnetConfig cb_cfg = cfg;
    AudioPipelineUlcnet *p, *cb;
    float mic[ULCNET_HOP], ref[ULCNET_HOP], out[ULCNET_HOP], out_cb[ULCNET_HOP];
    UlcnetNnJob job;
    int h, i, identical = 1, ran = 0;

    ulcnet_model_io_descriptor_default(8, &d);
    cb_cfg.model.infer = half_infer;
    cb_cfg.model.io_descriptor = &d;
    p = audio_pipeline_ulcnet_create(&cfg);        /* no infer callback: the board runs the network */
    cb = audio_pipeline_ulcnet_create(&cb_cfg);
    CHECK(p && cb, "readme/ulcnet: both pipelines create");
    if (!p || !cb) return;
    for (h = 0; h < 60; ++h) {
        for (i = 0; i < ULCNET_HOP; ++i) { ref[i] = lcg(); mic[i] = 0.3f * ref[i] + 0.05f * lcg(); }

        audio_pipeline_ulcnet_process_begin(p, mic, ref);          /* AEC, both analyses        */
        if (audio_pipeline_ulcnet_nn_begin(p, &job) > 0) {         /* a frame to run            */
            audio_pipeline_ulcnet_nn_end(p, run_ulcnet_npu(&job)); /* accept, or the identity   */
            ++ran;
        }
        audio_pipeline_ulcnet_process_end(p, out);                 /* WOLA, one hop             */

        audio_pipeline_ulcnet_process(cb, mic, ref, out_cb);
        if (memcmp(out, out_cb, sizeof(out)) != 0) identical = 0;
    }
    CHECK(ran > 0 && identical,
          "readme/ulcnet: process_begin / nn_begin / nn_end / process_end equals the callback-driven pipeline");
    audio_pipeline_ulcnet_destroy(p);
    audio_pipeline_ulcnet_destroy(cb);
}

int main(void) {
    example_chain_new_stage();
    example_dfn_phased();
    example_ulcnet_phased();
    if (g_fail) { fprintf(stderr, "%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
