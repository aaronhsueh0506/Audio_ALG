/* Test helper: the conventional 4-channel chain written out over the front end, an independent
 * oracle for tests that compare an application against it. It does not include the
 * example headers; every recipe is written out here.
 *
 *   mcab_process_front -> mcab_fuse -> AecRes gain (if enabled, on E)
 *     -> NR gain (if enabled, on E) -> sk_min_f32 -> [post-chain reset: ac_istft_reset, reseed]
 *     -> sk_capply_gain_f32 over the rendering base B -> aec_cng_fill -> ac_istft_push
 *
 * The chain follows the rules of the applications: a disabled gain source reads as the unity
 * vector, and the comfort-noise fill runs only with a suppressor and enable_cng. RES strength
 * and the denoiser level come from ConvChainOptions, not from the front end's config. The
 * stage objects share one heap block. Test code only: heap, no error recovery beyond the
 * status codes. */
#ifndef CONV_CHAIN_MCAB_H
#define CONV_CHAIN_MCAB_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mcab.h"
#include "mcab_test_rig.h"     /* conv_create(): a flow of its own */
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

/* The comfort-noise floor of the example programs; the seed is the library's. */
#define CONV_MCAB_CNG_NR_GAIN_FLOOR 0.31622777f
#define CONV_MCAB_CNG_RNG_SEED AEC_CNG_SEED

typedef struct ConvChainOptions {
    AecLevel aec_level;      /* RES gain floors */
    AnrLevel nr_level;
    int enable_res;            /* 0: G_res is unity and CNG has nothing to fill */
    int enable_nr;
    int enable_cng;
    int legacy_amin;           /* 1: the denoiser gets no residual-echo prior */
} ConvChainOptions;

static inline ConvChainOptions conv_default_options(void) {
    ConvChainOptions o;
    o.aec_level = AEC_LEVEL_BALANCED;
    o.nr_level = ANR_LEVEL_BALANCED;
    o.enable_res = 1;
    o.enable_nr = 1;
    o.enable_cng = 1;
    o.legacy_amin = 0;
    return o;
}

typedef struct ConvChain {
    ConvChainOptions opt;
    Mcab* core;
    int owns_core;
    AecRes* res;
    AnrDenoiser* nr;
    AcIstft* istft;
    void* mem;
    float* unity;
    float* total_gain;
    float* extra_noise;
    Complex* spec;
    uint32_t rng;
    uint32_t reset_handled;  /* McabFrame.post_reset_serial the synthesis last followed */
    int n_freqs;
    int hop;
    /* The last hop's fused frame and front view; the tests read both. */
    McabFrame frame;
    McabFront front;
} ConvChain;

/* The denoiser configuration of the conventional chain: the canonical level for `level` on a
 * no-padding 50 % overlap grid, with the 512 ms minimum window written out in whole hops
 * rather than taken from the level. */
static inline AnrConfig conv_nr_config(int sample_rate, int fft_size, int hop, AnrLevel level) {
    AnrConfig nr_cfg = anr_config_for_level_grid(sample_rate, fft_size, level);
    nr_cfg.L = (512 * sample_rate + 1000 * hop - 1) / (1000 * hop);
    return nr_cfg;
}

static inline void conv_destroy(ConvChain* c) {
    if (!c) return;
    if (c->nr) anr_destroy(c->nr);
    if (c->owns_core) mcab_test_destroy(c->core);
    free(c->mem);
    free(c);
}

/* The stages behind an existing front end `core`, built for `cfg` (the config the core was
 * created with) and `opt` (NULL: conv_default_options()). The core stays owned by the caller.
 * With core == NULL a new core is created and owned by the chain. */
static inline ConvChain* conv_attach(Mcab* core, const McabConfig* cfg,
                                     const ConvChainOptions* opt) {
    ConvChain* c = (ConvChain*)calloc(1, sizeof(*c));
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    AcStftConfig scfg;
    size_t res_b, nr_b, synth_b, vec, cvec, total;
    uint8_t* cur;
    int k, fft;
    if (!c) return NULL;
    c->opt = opt ? *opt : conv_default_options();
    c->core = core ? core : mcab_test_create(cfg);
    c->owns_core = core == NULL;
    if (!c->core) {
        conv_destroy(c);
        return NULL;
    }
    c->n_freqs = mcab_n_freqs(c->core);
    c->hop = mcab_hop_size(c->core);
    fft = mcab_fft_size(c->core);
    aec_config_from_level(&res_cfg, c->opt.aec_level, cfg->sample_rate);
    res_cfg.fft_size = fft;
    nr_cfg = conv_nr_config(cfg->sample_rate, fft, c->hop, c->opt.nr_level);
    ac_stft_config_default(&scfg, fft);
    res_b = ALIGN16(aec_res_get_mem_size(&res_cfg));
    nr_b = c->opt.enable_nr ? ALIGN16(anr_get_mem_size(&nr_cfg)) : 0;
    synth_b = ALIGN16(ac_istft_get_mem_size(&scfg));
    vec = ALIGN16((size_t)c->n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)c->n_freqs * sizeof(Complex));
    total = res_b + nr_b + synth_b + 3 * vec + cvec;
    if (posix_memalign(&c->mem, 16, total) != 0) { conv_destroy(c); return NULL; }
    cur = (uint8_t*)c->mem;
    if (c->opt.enable_res) c->res = aec_res_init(cur, aec_res_get_mem_size(&res_cfg), &res_cfg);
    cur += res_b;
    if (c->opt.enable_nr) c->nr = anr_init(cur, anr_get_mem_size(&nr_cfg), &nr_cfg);
    cur += nr_b;
    c->istft = ac_istft_init(cur, ac_istft_get_mem_size(&scfg), &scfg);
    cur += synth_b;
    c->unity = (float*)cur; cur += vec;
    c->total_gain = (float*)cur; cur += vec;
    c->extra_noise = (float*)cur; cur += vec;
    c->spec = (Complex*)cur;
    for (k = 0; k < c->n_freqs; ++k) c->unity[k] = 1.0f;
    c->rng = CONV_MCAB_CNG_RNG_SEED;
    if ((c->opt.enable_res && !c->res) || (c->opt.enable_nr && !c->nr) || !c->istft) {
        conv_destroy(c);
        return NULL;
    }
    return c;
}

/* What an application's reset does: the front end and every stage it chains. */
static inline void conv_reset(ConvChain* c) {
    mcab_reset(c->core);
    if (c->nr) anr_reset(c->nr);
    if (c->res) aec_res_reset(c->res);
    ac_istft_reset(c->istft);
    c->rng = CONV_MCAB_CNG_RNG_SEED;
}

/* The denoiser gain over the fused frame; the residual-echo prior is R^2 / 2^30, applied only
 * while the linear estimate is usable. NULL when the denoiser refuses the hop. */
static inline const float* conv_nr_gain(ConvChain* c) {
    const float* extra = NULL;
    if (!c->nr) return c->unity;
    if (!c->opt.legacy_amin && c->frame.r2 && c->frame.usable_linear) {
        sk_scale_f32(c->extra_noise, c->frame.r2, 0x1p-30f, c->n_freqs);
        extra = c->extra_noise;
    }
    if (anr_process_gain(c->nr, c->frame.e, extra, NULL) != 0) return NULL;
    return anr_get_gain(c->nr, NULL);
}

/* The RES gain over the fused frame; NULL when the suppressor refuses the hop. */
static inline const float* conv_res_gain(ConvChain* c) {
    AecResInput in;
    if (!c->res) return c->unity;
    in.error_spec = c->frame.e;
    in.near_spec = c->frame.y;
    in.r2 = c->frame.r2;
    in.comfort_noise = c->frame.cn;
    in.render_hop = c->frame.far_hop;
    in.all_converged = c->frame.all_converged;
    in.max_dt = c->frame.max_dt;
    in.max_saturation = c->frame.max_sat;
    in.clock_drift = c->frame.clock_drift;
    return aec_res_process(c->res, &in);
}

/* The fuse and the chain on an already processed front frame. `trusted` may be
 * NULL (the error is rebuilt from `weights`). Returns an MCAB_* status. */
static inline int conv_post(ConvChain* c, const Complex* weights, const Complex* trusted,
                     float* out) {
    const float* res_gain;
    const float* nr_gain;
    int rc = mcab_fuse(c->core, weights, trusted, &c->frame);
    if (rc != MCAB_OK) return rc;
    res_gain = conv_res_gain(c);
    nr_gain = res_gain ? conv_nr_gain(c) : NULL;
    if (!res_gain || !nr_gain) return MCAB_DSP_ERROR;
    sk_min_f32(c->total_gain, nr_gain, res_gain, c->n_freqs);
    if (c->frame.post_reset_serial != c->reset_handled) {
        c->reset_handled = c->frame.post_reset_serial;
        ac_istft_reset(c->istft);
        c->rng = CONV_MCAB_CNG_RNG_SEED;
    }
    sk_capply_gain_f32(c->spec, c->frame.b, c->total_gain, c->n_freqs);
    if (c->opt.enable_cng && c->res)
        aec_cng_fill(&c->rng, c->spec, c->frame.cn, res_gain, nr_gain,
                     CONV_MCAB_CNG_NR_GAIN_FLOOR, c->n_freqs);
    ac_istft_push(c->istft, c->spec, out);
    return MCAB_OK;
}

/* One hop: mcab_process_front, then conv_post. */
static inline int conv_hop(ConvChain* c, const float* mics, const float* ref,
                    const Complex* weights, const Complex* trusted, float* out) {
    int rc = mcab_process_front(c->core, mics, ref, &c->front);
    if (rc != MCAB_OK) return rc;
    return conv_post(c, weights, trusted, out);
}

static inline ConvChain* conv_create(const McabConfig* cfg, const ConvChainOptions* opt) {
    return conv_attach(NULL, cfg, opt);
}

#endif /* CONV_CHAIN_MCAB_H */
