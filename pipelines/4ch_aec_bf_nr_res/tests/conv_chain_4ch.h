/* Test helper: the conventional 4-channel chain written out over the front end,
 * for tests that compare an application against it.
 *
 *   four_aec_front_process -> four_aec_fuse -> se_res_stage_gain (if enabled)
 *     -> se_anr_stage_gain (if enabled) -> sk_min_f32 -> sk_capply_gain_f32
 *     -> aec_cng_fill -> ac_istft_push
 *
 * The chain follows the same rules as the applications: a disabled gain source
 * reads as the unity vector, and the comfort-noise fill runs only with a
 * suppressor and enable_cng. The stage objects share one heap block. Test code
 * only: heap, no error recovery beyond the status codes. */
#ifndef CONV_CHAIN_4CH_H
#define CONV_CHAIN_4CH_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_nr_res.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "common/se_anr_stage.h"
#include "common/se_stage.h"

typedef struct ConvChain {
    FourAecNrResConfig cfg;
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
    /* The last hop's gains and frame, for tests that inspect them. */
    const float* res_gain;
    const float* nr_gain;
    SeSpecFrame frame;
    FourAecFront front;
} ConvChain;

static inline void conv_destroy(ConvChain* c) {
    if (!c) return;
    if (c->owns_core) four_aec_nr_res_destroy(c->core);
    free(c->mem);
    free(c);
}

/* The stages behind an existing front end `core`, built for `cfg` (the config
 * the core was created with). The core stays owned by the caller. With
 * core == NULL a new core is created and owned by the chain. */
static inline ConvChain* conv_attach(FourAecNrRes* core, const FourAecNrResConfig* cfg) {
    ConvChain* c = (ConvChain*)calloc(1, sizeof(*c));
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    AcStftConfig scfg;
    size_t res_b, nr_b, synth_b, vec, cvec, total;
    uint8_t* cur;
    int k;
    if (!c) return NULL;
    c->cfg = *cfg;
    c->core = core ? core : four_aec_nr_res_create(cfg);
    c->owns_core = core == NULL;
    if (!c->core || four_aec_nr_res_stage_configs(cfg, &res_cfg, &nr_cfg) != 0) {
        conv_destroy(c);
        return NULL;
    }
    c->n_freqs = four_aec_nr_res_n_freqs(c->core);
    c->hop = four_aec_nr_res_hop_size(c->core);
    ac_stft_config_default(&scfg, four_aec_nr_res_fft_size(c->core));
    res_b = ALIGN16(aec_res_get_mem_size(&res_cfg));
    nr_b = cfg->enable_nr ? ALIGN16(se_anr_stage_mem_size(&nr_cfg)) : 0;
    synth_b = ALIGN16(ac_istft_get_mem_size(&scfg));
    vec = ALIGN16((size_t)c->n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)c->n_freqs * sizeof(Complex));
    total = res_b + nr_b + synth_b + 3 * vec + cvec;
    if (posix_memalign(&c->mem, 16, total) != 0) { conv_destroy(c); return NULL; }
    cur = (uint8_t*)c->mem;
    if (cfg->enable_res) c->res = aec_res_init(cur, aec_res_get_mem_size(&res_cfg), &res_cfg);
    cur += res_b;
    if (cfg->enable_nr) c->nr = se_anr_stage_init(cur, se_anr_stage_mem_size(&nr_cfg), &nr_cfg);
    cur += nr_b;
    c->istft = ac_istft_init(cur, ac_istft_get_mem_size(&scfg), &scfg);
    cur += synth_b;
    c->unity = (float*)cur; cur += vec;
    c->total_gain = (float*)cur; cur += vec;
    c->extra_noise = (float*)cur; cur += vec;
    c->spec = (Complex*)cur;
    for (k = 0; k < c->n_freqs; ++k) c->unity[k] = 1.0f;
    c->rng = SE_CNG_RNG_SEED;
    if ((cfg->enable_res && !c->res) || (cfg->enable_nr && !c->nr) || !c->istft) {
        conv_destroy(c);
        return NULL;
    }
    return c;
}

/* What an application's reset does: the front end and every stage it chains. */
static inline void conv_reset(ConvChain* c) {
    four_aec_nr_res_reset(c->core);
    if (c->nr) se_anr_stage_reset(c->nr);
    if (c->res) aec_res_reset(c->res);
    ac_istft_reset(c->istft);
    c->rng = SE_CNG_RNG_SEED;
}

/* The fuse and the chain on an already processed front frame. `trusted` may be
 * NULL (the error is rebuilt from `weights`). Returns a FOUR_AEC_NR_RES_* status. */
static inline int conv_post(ConvChain* c, const Complex* weights, const Complex* trusted,
                     float* out) {
    int rc = four_aec_fuse(c->core, weights, trusted, &c->frame);
    if (rc != FOUR_AEC_NR_RES_OK) return rc;
    c->res_gain = c->res ? se_res_stage_gain(c->res, &c->frame) : c->unity;
    c->nr_gain = c->nr ? se_anr_stage_gain(c->nr, &c->frame, c->extra_noise,
                                           !c->cfg.legacy_amin) : c->unity;
    if (!c->res_gain || !c->nr_gain) return FOUR_AEC_NR_RES_DSP_ERROR;
    sk_min_f32(c->total_gain, c->nr_gain, c->res_gain, c->n_freqs);
    sk_capply_gain_f32(c->spec, c->frame.e, c->total_gain, c->n_freqs);
    if (c->cfg.enable_cng && c->res)
        aec_cng_fill(&c->rng, c->spec, c->frame.cn, c->res_gain, c->nr_gain,
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

static inline ConvChain* conv_create(const FourAecNrResConfig* cfg) {
    return conv_attach(NULL, cfg);
}

#endif /* CONV_CHAIN_4CH_H */
