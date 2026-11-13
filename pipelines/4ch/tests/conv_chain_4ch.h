/* Test helper: the conventional 4-channel chain written out over the array instance, an independent
 * oracle for tests that compare an application against it. It does not include the example
 * headers; every recipe is written out here, on explicit arrays (AecResidualView.res_gain,
 * sk_min_f32, sk_capply_gain_f32, aec_apply_cng_spec with its own generator) rather than the
 * library-owned aec_apply_gain / aec_apply_cng the applications call.
 *
 *   aec_process_linear -> aec_process_residual -> G_res (FULL array) -> NR gain on E (if enabled)
 *     -> sk_min_f32 -> [post-chain reset: ac_istft_reset, reseed]
 *     -> sk_capply_gain_f32 over the rendering base -> aec_apply_cng_spec -> ac_istft_push
 *
 * A disabled gain source reads as the unity vector, and the comfort-noise fill runs only with a
 * suppressor and enable_cng. enable_res = 0 builds the array with AEC_RESIDUAL_ESTIMATE (no gain
 * stage). The RES level is the level of the AecConfig the array is built from; the denoiser level
 * comes from ConvChainOptions. The stage objects share one heap block. Test code only: heap, no
 * error recovery beyond the status codes. */
#ifndef CONV_CHAIN_4CH_H
#define CONV_CHAIN_4CH_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "array_test_rig.h"     /* conv_create(): an array of its own */
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

/* The comfort-noise floor of the example programs; the oracle's fill (aec_apply_cng_spec) uses the library's
 * AEC_CNG_NR_GAIN_FLOOR, which conv_attach() checks against this literal. */
#define CONV_4CH_CNG_NR_GAIN_FLOOR 0.31622777f

typedef struct ConvChainOptions {
    AnrLevel nr_level;
    int enable_res;            /* 0: G_res is unity and CNG has nothing to fill */
    int enable_nr;
    int enable_cng;
    int legacy_amin;           /* 1: the denoiser gets no residual-echo prior */
} ConvChainOptions;

static inline ConvChainOptions conv_default_options(void) {
    ConvChainOptions o;
    o.nr_level = ANR_LEVEL_BALANCED;
    o.enable_res = 1;
    o.enable_nr = 1;
    o.enable_cng = 1;
    o.legacy_amin = 0;
    return o;
}

typedef struct ConvChain {
    ConvChainOptions opt;
    Aec* aec;
    int owns_aec;
    AnrDenoiser* nr;
    AcIstft* istft;
    void* mem;
    float* unity;
    float* total_gain;
    float* extra_noise;
    Complex* spec;
    uint32_t rng;
    uint32_t reset_handled;  /* AecResidualView.post_reset_serial the synthesis last followed */
    int n_freqs;
    int hop;
    /* The last hop's fused frame and front view; the tests read both. */
    AecResidualView frame;
    AecArrayView front;
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
    if (c->owns_aec) array_test_destroy(c->aec);
    free(c->mem);
    free(c);
}

/* The stages behind an existing array `aec` (built from `cfg`; NULL cfg: an array of the rig), for
 * `opt` (NULL: conv_default_options()). The array stays owned by the caller; it must have been built with residual FULL (enable_res) or
 * ESTIMATE. With aec == NULL a new array is created from `cfg` (residual FULL, or ESTIMATE
 * without enable_res) and owned by the chain. */
static inline ConvChain* conv_attach(Aec* aec, const AecConfig* cfg,
                                     const ConvChainOptions* opt) {
    ConvChain* c = (ConvChain*)calloc(1, sizeof(*c));
    AnrConfig nr_cfg;
    AcStftConfig scfg;
    size_t nr_b, synth_b, vec, cvec, total;
    uint8_t* cur;
    int k, fft;
    if (!c) return NULL;
    if (CONV_4CH_CNG_NR_GAIN_FLOOR != AEC_CNG_NR_GAIN_FLOOR) { free(c); return NULL; }
    c->opt = opt ? *opt : conv_default_options();
    if (!aec) {
        AecConfig own = *cfg;
        own.residual = c->opt.enable_res ? AEC_RESIDUAL_FULL : AEC_RESIDUAL_ESTIMATE;
        aec = array_test_create(&own);
        c->owns_aec = 1;
    }
    c->aec = aec;
    if (!c->aec) {
        conv_destroy(c);
        return NULL;
    }
    c->hop = aec_hop_size(c->aec);
    c->n_freqs = c->hop + 1;
    fft = 2 * c->hop;
    nr_cfg = conv_nr_config(cfg ? cfg->sample_rate : array_test_sample_rate(c->aec), fft, c->hop,
                           c->opt.nr_level);
    ac_stft_config_default(&scfg, fft);
    nr_b = c->opt.enable_nr ? ALIGN16(anr_get_mem_size(&nr_cfg)) : 0;
    synth_b = ALIGN16(ac_istft_get_mem_size(&scfg));
    vec = ALIGN16((size_t)c->n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)c->n_freqs * sizeof(Complex));
    total = nr_b + synth_b + 3 * vec + cvec;
    if (posix_memalign(&c->mem, 16, total) != 0) { c->mem = NULL; conv_destroy(c); return NULL; }
    cur = (uint8_t*)c->mem;
    if (c->opt.enable_nr) c->nr = anr_init(cur, anr_get_mem_size(&nr_cfg), &nr_cfg);
    cur += nr_b;
    c->istft = ac_istft_init(cur, ac_istft_get_mem_size(&scfg), &scfg);
    cur += synth_b;
    c->unity = (float*)cur; cur += vec;
    c->total_gain = (float*)cur; cur += vec;
    c->extra_noise = (float*)cur; cur += vec;
    c->spec = (Complex*)cur;
    for (k = 0; k < c->n_freqs; ++k) c->unity[k] = 1.0f;
    c->rng = AEC_CNG_SEED;
    if ((c->opt.enable_nr && !c->nr) || !c->istft) {
        conv_destroy(c);
        return NULL;
    }
    return c;
}

/* What an application's reset does: the array and every stage it chains. */
static inline void conv_reset(ConvChain* c) {
    aec_reset(c->aec);
    if (c->nr) anr_reset(c->nr);
    ac_istft_reset(c->istft);
    c->rng = AEC_CNG_SEED;
}

/* The denoiser gain over the fused frame; the residual-echo prior is R^2 / 2^30, applied only
 * while the linear estimate is usable. NULL when the denoiser refuses the hop. */
static inline const float* conv_nr_gain(ConvChain* c) {
    const float* residual_psd_audio = NULL;
    if (!c->nr) return c->unity;
    if (!c->opt.legacy_amin && c->frame.r2 && c->frame.usable_linear) {
        sk_scale_f32(c->extra_noise, c->frame.r2, 0x1p-30f, c->n_freqs);
        residual_psd_audio = c->extra_noise;
    }
    return anr_process_gain_with_residual_psd(c->nr, c->frame.error_spec, residual_psd_audio);
}

/* The RES gain of the fused frame (the array's gain stage); the unity vector without enable_res.
 * NULL when the array has none. Valid after array_test_fuse(). */
static inline const float* conv_res_gain(ConvChain* c) {
    return c->opt.enable_res ? c->frame.res_gain : c->unity;
}

/* The fuse and the chain on an already processed front frame. `trusted` may be
 * NULL (the error is rebuilt from `weights`). Returns an AEC_* status. */
static inline int conv_post(ConvChain* c, const Complex* weights, const Complex* trusted,
                     float* out) {
    const float* res_gain;
    const float* nr_gain;
    int rc = array_test_fuse(c->aec, weights, trusted, &c->frame);
    if (rc != AEC_OK) return rc;
    res_gain = conv_res_gain(c);
    nr_gain = res_gain ? conv_nr_gain(c) : NULL;
    if (!res_gain || !nr_gain) return AEC_ERR_DSP;
    sk_min_f32(c->total_gain, nr_gain, res_gain, c->n_freqs);
    if (c->frame.post_reset_serial != c->reset_handled) {
        c->reset_handled = c->frame.post_reset_serial;
        ac_istft_reset(c->istft);
        c->rng = AEC_CNG_SEED;
    }
    sk_capply_gain_f32(c->spec, c->frame.base_spec, c->total_gain, c->n_freqs);
    if (c->opt.enable_cng && c->opt.enable_res)
        aec_apply_cng_spec(&c->rng, c->spec, c->frame.comfort_noise, res_gain, nr_gain, c->n_freqs);
    ac_istft_push(c->istft, c->spec, out);
    return AEC_OK;
}

/* One hop: aec_process_linear, then conv_post. */
static inline int conv_hop(ConvChain* c, const float* mics, const float* ref,
                    const Complex* weights, const Complex* trusted, float* out) {
    int rc = array_test_front(c->aec, mics, ref, &c->front);
    if (rc != AEC_OK) return rc;
    return conv_post(c, weights, trusted, out);
}

static inline ConvChain* conv_create(const AecConfig* cfg, const ConvChainOptions* opt) {
    return conv_attach(NULL, cfg, opt);
}

#endif /* CONV_CHAIN_4CH_H */
