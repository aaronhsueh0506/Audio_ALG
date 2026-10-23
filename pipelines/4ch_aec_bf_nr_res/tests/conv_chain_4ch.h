/* Test helper: the conventional 4-channel chain written out over the front end, an independent
 * oracle for tests that compare an application against it. It does not include the
 * samples/ headers; every recipe is written out here.
 *
 *   four_aec_front_process -> four_aec_fuse -> AecRes gain (if enabled)
 *     -> NR gain (if enabled) -> sk_min_f32 -> sk_capply_gain_f32
 *     -> aec_cng_fill -> ac_istft_push
 *
 * The chain follows the rules of the applications: a disabled gain source reads as the unity
 * vector, and the comfort-noise fill runs only with a suppressor and enable_cng. RES strength
 * and the denoiser preset come from ConvChainOptions, not from the front end's config. The
 * stage objects share one heap block. Test code only: heap, no error recovery beyond the
 * status codes. */
#ifndef CONV_CHAIN_4CH_H
#define CONV_CHAIN_4CH_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_nr_res.h"
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "common/se_stage.h"

typedef struct ConvChainOptions {
    AecPreset aec_preset;      /* RES gain floors */
    AnrMode nr_mode;
    int enable_res;            /* 0: G_res is unity and CNG has nothing to fill */
    int enable_nr;
    int enable_cng;
    int legacy_amin;           /* 1: the denoiser gets no residual-echo prior */
} ConvChainOptions;

static inline ConvChainOptions conv_default_options(void) {
    ConvChainOptions o;
    o.aec_preset = AEC_PRESET_BALANCED;
    o.nr_mode = ANR_MODE_BALANCED;
    o.enable_res = 1;
    o.enable_nr = 1;
    o.enable_cng = 1;
    o.legacy_amin = 0;
    return o;
}

typedef struct ConvChain {
    ConvChainOptions opt;
    FourAecNrRes* core;
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
    int n_freqs;
    int hop;
    /* The last hop's fused frame and front view; the tests read both. */
    SeSpecFrame frame;
    FourAecFront front;
} ConvChain;

/* The denoiser configuration of the conventional chain: the canonical preset for `mode` on a
 * no-padding 50 % overlap grid, with the 512 ms minimum window written out in whole hops
 * rather than taken from the preset. */
static inline AnrConfig conv_nr_config(int sample_rate, int fft_size, int hop, AnrMode mode) {
    AnrConfig nr_cfg = anr_config_for_mode_grid(sample_rate, fft_size, mode);
    nr_cfg.L = (512 * sample_rate + 1000 * hop - 1) / (1000 * hop);
    return nr_cfg;
}

static inline void conv_destroy(ConvChain* c) {
    if (!c) return;
    if (c->nr) anr_destroy(c->nr);
    if (c->owns_core) four_aec_nr_res_destroy(c->core);
    free(c->mem);
    free(c);
}

/* The stages behind an existing front end `core`, built for `cfg` (the config the core was
 * created with) and `opt` (NULL: conv_default_options()). The core stays owned by the caller.
 * With core == NULL a new core is created and owned by the chain. */
static inline ConvChain* conv_attach(FourAecNrRes* core, const FourAecNrResConfig* cfg,
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
    c->core = core ? core : four_aec_nr_res_create(cfg);
    c->owns_core = core == NULL;
    if (!c->core) {
        conv_destroy(c);
        return NULL;
    }
    c->n_freqs = four_aec_nr_res_n_freqs(c->core);
    c->hop = four_aec_nr_res_hop_size(c->core);
    fft = four_aec_nr_res_fft_size(c->core);
    aec_config_from_preset(&res_cfg, c->opt.aec_preset, cfg->sample_rate);
    res_cfg.fft_size = fft;
    nr_cfg = conv_nr_config(cfg->sample_rate, fft, c->hop, c->opt.nr_mode);
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
    c->rng = SE_CNG_RNG_SEED;
    if ((c->opt.enable_res && !c->res) || (c->opt.enable_nr && !c->nr) || !c->istft) {
        conv_destroy(c);
        return NULL;
    }
    return c;
}

/* What an application's reset does: the front end and every stage it chains. */
static inline void conv_reset(ConvChain* c) {
    four_aec_nr_res_reset(c->core);
    if (c->nr) anr_reset(c->nr);
    if (c->res) aec_res_reset(c->res);
    ac_istft_reset(c->istft);
    c->rng = SE_CNG_RNG_SEED;
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
 * NULL (the error is rebuilt from `weights`). Returns a FOUR_AEC_NR_RES_* status. */
static inline int conv_post(ConvChain* c, const Complex* weights, const Complex* trusted,
                     float* out) {
    const float* res_gain;
    const float* nr_gain;
    int rc = four_aec_fuse(c->core, weights, trusted, &c->frame);
    if (rc != FOUR_AEC_NR_RES_OK) return rc;
    res_gain = conv_res_gain(c);
    nr_gain = res_gain ? conv_nr_gain(c) : NULL;
    if (!res_gain || !nr_gain) return FOUR_AEC_NR_RES_DSP_ERROR;
    sk_min_f32(c->total_gain, nr_gain, res_gain, c->n_freqs);
    sk_capply_gain_f32(c->spec, c->frame.e, c->total_gain, c->n_freqs);
    if (c->opt.enable_cng && c->res)
        aec_cng_fill(&c->rng, c->spec, c->frame.cn, res_gain, nr_gain,
                     SE_CNG_NR_GAIN_FLOOR, c->n_freqs);
    ac_istft_push(c->istft, c->spec, out);
    return FOUR_AEC_NR_RES_OK;
}

/* One hop: front_process, then conv_post. */
static inline int conv_hop(ConvChain* c, const float* mics, const float* ref,
                    const Complex* weights, const Complex* trusted, float* out) {
    int rc = four_aec_front_process(c->core, mics, ref, &c->front);
    if (rc != FOUR_AEC_NR_RES_OK) return rc;
    return conv_post(c, weights, trusted, out);
}

static inline ConvChain* conv_create(const FourAecNrResConfig* cfg, const ConvChainOptions* opt) {
    return conv_attach(NULL, cfg, opt);
}

#endif /* CONV_CHAIN_4CH_H */
