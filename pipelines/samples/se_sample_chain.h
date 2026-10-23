/* se_sample_chain.h -- example: the gain sequence behind a frequency-domain AEC front end,
 * from the SeSpecFrame of one hop to the time-domain output. Header-only, no ABI, no
 * layout version: copy and adapt.
 *
 *   RES gain -> NR gain -> min -> apply to E -> comfort noise on the RES-cut bins -> synthesis
 *
 * A stage that is off reads as the unity vector. Comfort noise runs only with a RES source and
 * enable_cng, at the level of the RES gain before the min. Every stage is a separate library
 * object (AecRes, AnrDenoiser, AcIstft) in one caller-owned pool; the chain only sequences
 * them. A failed call leaves recovery to the caller: reset the front end and the chain together
 * (se_sample_chain_reset also restarts the comfort-noise generator). */
#ifndef PIPELINES_SAMPLES_SE_SAMPLE_CHAIN_H
#define PIPELINES_SAMPLES_SE_SAMPLE_CHAIN_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "se_anr_stage.h"
#include "common/se_config.h"   /* se_flag_valid */
#include "common/se_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SeSampleRes {
    SE_SAMPLE_RES_OFF = 0,
    SE_SAMPLE_RES_FRAME_GAIN,   /* the front end's own gain: SeSpecFrame.g_res_ctx */
    SE_SAMPLE_RES_AEC_RES       /* a separate AecRes over the frame, strength res_preset */
} SeSampleRes;

typedef struct SeSampleChainConfig {
    int sample_rate;
    int fft_size;
    SeSampleRes res;
    AecPreset res_preset;       /* SE_SAMPLE_RES_AEC_RES only */
    int enable_nr;
    AnrMode nr_mode;
    int legacy_amin;            /* 1: the denoiser gets no residual-echo prior */
    int enable_cng;
    int with_synthesis;         /* 0: the chain stops at the spectrum P */
} SeSampleChainConfig;

typedef struct SeSampleChain {
    SeSampleChainConfig cfg;
    int n_freqs;
    int hop;
    AecRes* res;
    AnrDenoiser* nr;
    AcIstft* istft;
    float* unity;
    float* total_gain;
    float* extra_noise;
    Complex* spec;
    uint32_t rng;
    const float* last_res;
    const float* last_nr;
} SeSampleChain;

typedef struct SeSampleChainSizes {
    int n_freqs;
    int hop;
    size_t res_bytes;
    size_t nr_bytes;
    size_t synth_bytes;
    size_t total;
} SeSampleChainSizes;

/* All stages on at the product strengths. */
static inline SeSampleChainConfig se_sample_chain_default_config(int sample_rate,
                                                                 int fft_size) {
    SeSampleChainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = fft_size;
    cfg.res = SE_SAMPLE_RES_AEC_RES;
    cfg.res_preset = AEC_PRESET_BALANCED;
    cfg.enable_nr = 1;
    cfg.nr_mode = ANR_MODE_BALANCED;
    cfg.enable_cng = 1;
    cfg.with_synthesis = 1;
    return cfg;
}

static inline void se_sample_chain_res_config(const SeSampleChainConfig* cfg, AecConfig* out) {
    aec_config_from_preset(out, cfg->res_preset, cfg->sample_rate);
    out->fft_size = cfg->fft_size;
}

/* 0 and the carve sizes for `cfg`, or -1 for an invalid configuration or grid. */
static inline int se_sample_chain_sizes(const SeSampleChainConfig* cfg,
                                        SeSampleChainSizes* out) {
    AecSignalGrid grid;
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    AcStftConfig scfg;
    size_t vec, cvec, total;
    float floor_db;
    if (!cfg || !out) return -1;
    if (!aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, &grid)) return -1;
    if (cfg->res != SE_SAMPLE_RES_OFF && cfg->res != SE_SAMPLE_RES_FRAME_GAIN &&
        cfg->res != SE_SAMPLE_RES_AEC_RES) return -1;
    if (cfg->res == SE_SAMPLE_RES_AEC_RES && aec_preset_floor_db(cfg->res_preset, &floor_db) != 0)
        return -1;
    if (!se_flag_valid(cfg->enable_nr) || !se_flag_valid(cfg->legacy_amin) ||
        !se_flag_valid(cfg->enable_cng) || !se_flag_valid(cfg->with_synthesis)) return -1;
    if (cfg->enable_nr && !anr_nr_mode_is_valid(cfg->nr_mode)) return -1;

    out->n_freqs = grid.n_freqs;
    out->hop = grid.hop_size;
    out->res_bytes = 0;
    out->nr_bytes = 0;
    out->synth_bytes = 0;
    if (cfg->res == SE_SAMPLE_RES_AEC_RES) {
        se_sample_chain_res_config(cfg, &res_cfg);
        out->res_bytes = aec_res_get_mem_size(&res_cfg);
        if (out->res_bytes == 0) return -1;
    }
    if (cfg->enable_nr) {
        nr_cfg = anr_config_for_mode_grid(cfg->sample_rate, cfg->fft_size, cfg->nr_mode);
        out->nr_bytes = anr_get_mem_size(&nr_cfg);
        if (out->nr_bytes == 0) return -1;
    }
    if (cfg->with_synthesis) {
        ac_stft_config_default(&scfg, cfg->fft_size);
        out->synth_bytes = ac_istft_get_mem_size(&scfg);
        if (out->synth_bytes == 0) return -1;
    }
    vec = ALIGN16((size_t)grid.n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)grid.n_freqs * sizeof(Complex));
    total = ALIGN16(sizeof(SeSampleChain)) + ALIGN16(out->res_bytes) + ALIGN16(out->nr_bytes) +
            ALIGN16(out->synth_bytes) + 3 * vec + cvec;
    out->total = total;
    return 0;
}

/* Bytes of the pool for `cfg`; 0 for an invalid configuration. */
static inline size_t se_sample_chain_mem_size(const SeSampleChainConfig* cfg) {
    SeSampleChainSizes s;
    return se_sample_chain_sizes(cfg, &s) == 0 ? s.total : 0;
}

/* Builds the chain in a 16-byte aligned block of at least se_sample_chain_mem_size(cfg)
 * bytes; the block may hold stale data. NULL on failure. */
static inline SeSampleChain* se_sample_chain_init(void* mem, size_t bytes,
                                                  const SeSampleChainConfig* cfg) {
    SeSampleChainSizes s;
    SeSampleChain* c;
    uint8_t* cur;
    int k;
    if (!mem || !MEM_IS_ALIGNED16(mem) || se_sample_chain_sizes(cfg, &s) != 0 ||
        bytes < s.total) return NULL;
    memset(mem, 0, s.total);
    c = (SeSampleChain*)mem;
    c->cfg = *cfg;
    c->n_freqs = s.n_freqs;
    c->hop = s.hop;
    cur = (uint8_t*)mem + ALIGN16(sizeof(*c));
    if (cfg->res == SE_SAMPLE_RES_AEC_RES) {
        AecConfig res_cfg;
        se_sample_chain_res_config(cfg, &res_cfg);
        c->res = aec_res_init(cur, s.res_bytes, &res_cfg);
        if (!c->res) return NULL;
        cur += ALIGN16(s.res_bytes);
    }
    if (cfg->enable_nr) {
        AnrConfig nr_cfg = anr_config_for_mode_grid(cfg->sample_rate, cfg->fft_size,
                                                    cfg->nr_mode);
        c->nr = anr_init(cur, s.nr_bytes, &nr_cfg);
        if (!c->nr) return NULL;
        cur += ALIGN16(s.nr_bytes);
    }
    if (cfg->with_synthesis) {
        AcStftConfig scfg;
        ac_stft_config_default(&scfg, cfg->fft_size);
        c->istft = ac_istft_init(cur, s.synth_bytes, &scfg);
        if (!c->istft) return NULL;
        cur += ALIGN16(s.synth_bytes);
    }
    c->unity = (float*)cur;
    cur += ALIGN16((size_t)s.n_freqs * sizeof(float));
    c->total_gain = (float*)cur;
    cur += ALIGN16((size_t)s.n_freqs * sizeof(float));
    c->extra_noise = (float*)cur;
    cur += ALIGN16((size_t)s.n_freqs * sizeof(float));
    c->spec = (Complex*)cur;
    for (k = 0; k < s.n_freqs; ++k) c->unity[k] = 1.0f;
    c->rng = SE_CNG_RNG_SEED;
    return c;
}

/* The RES stage (NULL unless res = SE_SAMPLE_RES_AEC_RES): aec_res_set_preset() and
 * aec_res_split_floor() act on it. */
static inline AecRes* se_sample_chain_res(const SeSampleChain* c) {
    return c ? c->res : NULL;
}

/* The denoiser (NULL without NR): the library preset for nr_mode on the chain's grid, so
 * anr_set_mode() acts on it. */
static inline AnrDenoiser* se_sample_chain_nr(const SeSampleChain* c) {
    return c ? c->nr : NULL;
}

/* The gains of the last form() call: the RES gain before the min and the NR gain (the unity
 * vector for an absent stage), owned by their stages and valid until the next call. NULL
 * before the first call and after a reset. */
static inline const float* se_sample_chain_last_res_gain(const SeSampleChain* c) {
    return c ? c->last_res : NULL;
}

static inline const float* se_sample_chain_last_nr_gain(const SeSampleChain* c) {
    return c ? c->last_nr : NULL;
}

/* One hop over `frame` (needs e; g_res_ctx for FRAME_GAIN; the inputs of se_res_stage_gain
 * for AEC_RES): the spectrum P, n_freqs bins owned by the chain and valid until its next
 * call. NULL for a NULL argument, a frame of another grid or a stage that refused the hop. */
static inline const Complex* se_sample_chain_form(SeSampleChain* c, const SeSpecFrame* frame) {
    const float* res_gain;
    const float* nr_gain;
    if (!c || !frame || !frame->e || frame->n_freqs != c->n_freqs) return NULL;
    switch (c->cfg.res) {
        case SE_SAMPLE_RES_FRAME_GAIN: res_gain = frame->g_res_ctx; break;
        case SE_SAMPLE_RES_AEC_RES: res_gain = se_res_stage_gain(c->res, frame); break;
        default: res_gain = c->unity; break;
    }
    if (!res_gain) return NULL;
    nr_gain = c->unity;
    if (c->nr) {
        nr_gain = se_anr_stage_gain(c->nr, frame, c->extra_noise, !c->cfg.legacy_amin);
        if (!nr_gain) return NULL;
    }
    c->last_res = res_gain;
    c->last_nr = nr_gain;
    /* sk_min_f32 propagates a non-finite gain to the output; fminf would mask it. */
    sk_min_f32(c->total_gain, nr_gain, res_gain, c->n_freqs);
    sk_capply_gain_f32(c->spec, frame->e, c->total_gain, c->n_freqs);
    /* Comfort noise on the RES-cut bins, scaled by G_nr (floored at SE_CNG_NR_GAIN_FLOOR)
     * because the denoiser never sees the fill. */
    if (c->cfg.enable_cng && c->cfg.res != SE_SAMPLE_RES_OFF && frame->cn)
        aec_cng_fill(&c->rng, c->spec, frame->cn, res_gain, nr_gain, SE_CNG_NR_GAIN_FLOOR,
                     c->n_freqs);
    return c->spec;
}

/* One synthesis hop of `spectrum` (normally the result of form()) into out[hop]. Returns 0,
 * or -1 for a NULL argument or a chain without synthesis. */
static inline int se_sample_chain_synthesize(SeSampleChain* c, const Complex* spectrum,
                                             float* out) {
    if (!c || !c->istft || !spectrum || !out) return -1;
    (void)ac_istft_push(c->istft, spectrum, out);
    return 0;
}

/* 1 when all n samples are finite: the check an integrator runs on an emitted hop before
 * trusting it. */
static inline int se_hop_is_finite(const float* x, int n) {
    int k;
    for (k = 0; k < n; ++k)
        if (!isfinite(x[k])) return 0;
    return 1;
}

/* Back to the state of a fresh chain: RES, NR and synthesis history and the comfort-noise
 * generator. */
static inline void se_sample_chain_reset(SeSampleChain* c) {
    if (!c) return;
    if (c->res) aec_res_reset(c->res);
    if (c->nr) anr_reset(c->nr);
    if (c->istft) ac_istft_reset(c->istft);
    c->rng = SE_CNG_RNG_SEED;
    c->last_res = NULL;
    c->last_nr = NULL;
}

/* Releases the denoiser's resources; the pool itself belongs to the caller. */
static inline void se_sample_chain_destroy(SeSampleChain* c) {
    if (c && c->nr) anr_destroy(c->nr);
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SAMPLES_SE_SAMPLE_CHAIN_H */
