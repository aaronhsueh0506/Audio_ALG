/* Test helper: the conventional mono chain written out over lib/aec's Aec, for tests that
 * compare an application against it.
 *
 *   aec_process_context -> aec_get_res_context + aec_get_output_base -> follow aec_post_reset_serial
 *   (clear the OLA tail, reseed the comfort noise) -> NR gain on the rendering base (if enabled) ->
 *   min with the lane's RES gain (if enabled) -> apply to the base -> comfort noise -> ac_istft_push
 *
 * The rendering base is the library's own choice per hop (error spectrum, or error + echo when the
 * output guard fires); the NR estimator reads it and the gain multiplies it, so the full-mode chain
 * (RES only, aec_process()) is the special case nr = unity. The synthesis borrows the AEC's window
 * table unless `default_synth_window` asks for audiolib's default (the DFN stage synthesises with
 * that one). The NR keeps running across a reset event.
 *
 * Independent of the example headers: the AecConfig, the NR configuration (anr_config_for_level_grid with
 * the 512 ms minimum window written out, r2 scaled by 2^-30 as the residual-echo prior) and the
 * min/apply loops are spelled out here, so a defect in the shared sample code cannot hide behind itself. A disabled
 * gain source reads as the unity vector; the comfort-noise fill runs only with the RES gain and
 * enable_cng. The Aec is heap-built (aec_create). Test code only: heap, no error recovery
 * beyond the status codes. */
#ifndef CONV_CHAIN_MONO_H
#define CONV_CHAIN_MONO_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"

/* The comfort-noise floor of the example programs (the seed is AEC_CNG_SEED). */
#define CONV_MONO_CNG_NR_GAIN_FLOOR 0.31622777f

typedef struct ConvMonoConfig {
    int sample_rate;
    int fft_size;                 /* 0: the rate's default grid */
    AecDelayMode delay_mode;
    int delay_num_filters;
    int fixed_delay_samples;
    AecLevel aec_level;
    AnrLevel nr_level;
    int aec_only;                 /* the AEC's own time-domain hop (ctx.linear_hop) */
    int enable_nr, enable_res, enable_cng, legacy_amin;
    int default_synth_window;     /* 1: audiolib's default window (as the DFN stage), 0: the AEC's table */
} ConvMonoConfig;

static inline ConvMonoConfig conv_mono_default_config(int sample_rate, int fft_size) {
    ConvMonoConfig c;
    memset(&c, 0, sizeof(c));
    c.sample_rate = sample_rate;
    c.fft_size = fft_size;
    c.delay_mode = AEC_DELAY_MATCHED;
    c.delay_num_filters = DA_NUM_FILTERS;
    c.fixed_delay_samples = -1;
    c.aec_level = AEC_LEVEL_BALANCED;
    c.nr_level = ANR_LEVEL_BALANCED;
    c.enable_nr = c.enable_res = c.enable_cng = 1;
    return c;
}

typedef struct ConvMono {
    ConvMonoConfig cfg;
    Aec* aec;
    AnrDenoiser* nr;
    AcIstft* istft;
    void* mem;
    float* unity;
    float* total_gain;
    float* extra_noise;
    Complex* spec;
    uint32_t rng;
    uint32_t serial;              /* the aec_post_reset_serial() value already handled */
    long guard_hops, reset_hops;  /* hops on which the guard fired / the serial moved (statistics for tests) */
    int n_freqs;
    int hop;
} ConvMono;

static inline void conv_mono_destroy(ConvMono* c) {
    if (!c) return;
    if (c->nr) anr_destroy(c->nr);
    if (c->aec) { aec_destroy(c->aec); free(c->aec); }
    free(c->mem);
    free(c);
}

/* The lane the applications derive: level config on the rate's grid, the delay request as
 * given, RES left to the chain and the context kept (enable_res 0, return_res_context 1). */
static inline void conv_mono_lane_config(const ConvMonoConfig* cfg, AecConfig* lane) {
    aec_config_from_level(lane, cfg->aec_level, cfg->sample_rate);
    lane->fft_size = cfg->fft_size ? cfg->fft_size : aec_default_fft_size(cfg->sample_rate);
    lane->delay_mode = cfg->delay_mode;
    lane->delay_num_filters = cfg->delay_num_filters;
    lane->fixed_delay_samples = cfg->fixed_delay_samples;
    lane->enable_delay_est = cfg->delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
    lane->enable_res = 0;
    lane->return_res_context = 1;
}

static inline ConvMono* conv_mono_create(const ConvMonoConfig* cfg) {
    ConvMono* c = (ConvMono*)calloc(1, sizeof(*c));
    AecConfig lane;
    AnrConfig nc;
    AcStftConfig sc;
    size_t nr_b = 0, synth_b, vec, cvec;
    uint8_t* cur;
    int k;
    if (!c) return NULL;
    c->cfg = *cfg;
    conv_mono_lane_config(cfg, &lane);
    c->aec = (Aec*)calloc(1, sizeof(Aec));
    if (!c->aec || aec_create(c->aec, &lane) != 0) {
        free(c->aec);
        c->aec = NULL;
        conv_mono_destroy(c);
        return NULL;
    }
    c->hop = lane.fft_size / 2;
    c->n_freqs = lane.fft_size / 2 + 1;
    if (cfg->aec_only) return c;
    ac_stft_config_default(&sc, lane.fft_size);
    if (!cfg->default_synth_window) {
        int win_n = 0;
        if (aec_synthesis_window(c->aec, &sc.window, &win_n) != 0 || win_n != lane.fft_size) {
            conv_mono_destroy(c);
            return NULL;
        }
    }
    synth_b = ALIGN16(ac_istft_get_mem_size(&sc));
    if (cfg->enable_nr) {
        nc = anr_config_for_level_grid(cfg->sample_rate, lane.fft_size, cfg->nr_level);
        /* minimum window 512 ms in whole hops, written out rather than taken from the level */
        nc.L = (512 * cfg->sample_rate + 1000 * c->hop - 1) / (1000 * c->hop);
        nr_b = ALIGN16(anr_get_mem_size(&nc));
    }
    vec = ALIGN16((size_t)c->n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)c->n_freqs * sizeof(Complex));
    if (posix_memalign(&c->mem, 16, nr_b + synth_b + 3 * vec + cvec) != 0) {
        c->mem = NULL;
        conv_mono_destroy(c);
        return NULL;
    }
    cur = (uint8_t*)c->mem;
    if (cfg->enable_nr) c->nr = anr_init(cur, anr_get_mem_size(&nc), &nc);
    cur += nr_b;
    c->istft = ac_istft_init(cur, ac_istft_get_mem_size(&sc), &sc);
    cur += synth_b;
    c->unity = (float*)cur; cur += vec;
    c->total_gain = (float*)cur; cur += vec;
    c->extra_noise = (float*)cur; cur += vec;
    c->spec = (Complex*)cur;
    for (k = 0; k < c->n_freqs; ++k) c->unity[k] = 1.0f;
    c->rng = AEC_CNG_SEED;
    c->serial = aec_post_reset_serial(c->aec);
    if ((cfg->enable_nr && !c->nr) || !c->istft) {
        conv_mono_destroy(c);
        return NULL;
    }
    return c;
}

/* One hop into out (hop samples). Returns 0, or -1 when a stage refuses. */
static inline int conv_mono_hop(ConvMono* c, const float* mic, const float* ref, float* out) {
    const float* extra = NULL;
    const float* res_gain;
    const float* nr_gain;
    AecResContext ctx;
    AecOutputBase ob;
    const Complex* base;
    uint32_t serial;
    int k;
    aec_process_context(c->aec, mic, ref);
    aec_get_res_context(c->aec, &ctx);
    if (c->cfg.aec_only) {
        memcpy(out, ctx.linear_hop, (size_t)c->hop * sizeof(float));
        return 0;
    }
    memset(&ob, 0, sizeof(ob));
    ob.struct_size = sizeof(ob);
    if (aec_get_output_base(c->aec, &ob) != 0) return -1;
    base = ob.base_spec;
    if (ob.guard_active) ++c->guard_hops;
    serial = aec_post_reset_serial(c->aec);
    if (serial != c->serial) {
        c->serial = serial;
        ++c->reset_hops;
        ac_istft_reset(c->istft);
        c->rng = AEC_CNG_SEED;
    }
    res_gain = c->cfg.enable_res ? ctx.res_gain : c->unity;
    nr_gain = c->unity;
    if (c->nr) {
        if (!c->cfg.legacy_amin && ctx.usable_linear) {
            for (k = 0; k < c->n_freqs; ++k) c->extra_noise[k] = ctx.r2[k] * 0x1p-30f;
            extra = c->extra_noise;
        }
        if (anr_process_gain(c->nr, base, extra, NULL) != 0) return -1;
        nr_gain = anr_get_gain(c->nr, NULL);
        if (!nr_gain) return -1;
    }
    for (k = 0; k < c->n_freqs; ++k) {
        float a = nr_gain[k], b = res_gain[k];
        c->total_gain[k] = a < b ? a : b;
        c->spec[k].r = base[k].r * c->total_gain[k];
        c->spec[k].i = base[k].i * c->total_gain[k];
    }
    if (c->cfg.enable_cng && c->cfg.enable_res && ctx.comfort_noise)
        aec_cng_fill(&c->rng, c->spec, ctx.comfort_noise, ctx.res_gain, nr_gain,
                     CONV_MONO_CNG_NR_GAIN_FLOOR, c->n_freqs);
    (void)ac_istft_push(c->istft, c->spec, out);
    return 0;
}

#endif /* CONV_CHAIN_MONO_H */
