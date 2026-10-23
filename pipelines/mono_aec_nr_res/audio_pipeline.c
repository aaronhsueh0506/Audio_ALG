/** Implementation of audio_pipeline.h: pool sizing/carving and per-hop processing in
 * one linkable TU shared by both CLIs. The synthesis state and the comfort-noise
 * generator (lib/aec's aec_cng_fill()) are per instance; every pipeline-owned
 * scratch buffer is zeroed at carve time, so a dirty pool inits like a zeroed one. */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "audio_pipeline.h"
#include "common/se_config.h"     /* se_derive(): grid + AecConfig, shared by every application */
#include "common/se_stage.h"      /* SeSpecFrame                                                  */
#include "common/se_anr_stage.h"  /* NR configuration, init, per-hop gain, strength setter       */
#include "fft_wrapper.h"       /* fft_get_mem_size, Complex */
#include "utility/mem_align.h"       /* ALIGN16 */
#include "audio_stft.h"        /* AcIstft: inverse FFT + sqrt-Hann OLA                */
#include "utility/simd_kernels.h"      /* sk_min_f32 / sk_capply_gain_f32                                     */
#include "aec_res.h"          /* aec_cng_fill                                                        */

/** Diagnostics (init/build-time reject reasons only) are advisory; every failure also returns NULL/-1. -DAUDIO_PIPELINE_NO_STDIO (Makefile NO_STDIO=1) compiles AP_LOG_ERR() to a no-op and drops <stdio.h>. */
#ifndef AUDIO_PIPELINE_NO_STDIO
#include <stdio.h>
#define AP_LOG_ERR(...) fprintf(stderr, __VA_ARGS__)
#else
#define AP_LOG_ERR(...) ((void)0)
#endif

/* Carve-layout version; bumped on any change of pool size, carve order or descriptor
 * content, including a change of the composed AEC, NR or FFT pool. The token string
 * in audio_pipeline_build_flags_hash() changes with it. */
#define AUDIO_PIPELINE_LAYOUT_VERSION 15u

/* Compile-time FFT backend identity passed by pipelines/Makefile; the fallback below
 * applies only outside that Makefile. */
#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

/* Maps the compile-time AUDIO_PIPELINE_BACKEND_STR to the integer a serializable descriptor carries; 0 (unknown) is rejected by get_mem_requirements().
 * Stage timing is diagnostic and off by default; this flag governs the stages measured here, AEC_STAGE_TIMING what aec_get_last_timing() reports (`make PROFILE=1` sets both). */
#ifndef AUDIO_PIPELINE_STAGE_TIMING
#define AUDIO_PIPELINE_STAGE_TIMING 0
#endif

#if AUDIO_PIPELINE_STAGE_TIMING
/* Microsecond monotonic stamp truncated to 32 bits (consumers subtract in unsigned arithmetic). A target without POSIX CLOCK_MONOTONIC names a no-argument uint32_t-microsecond function with -DAUDIO_PIPELINE_NOW_US=<fn> (and lib/aec's -DAEC_NOW_US=<fn>). */
#ifndef AUDIO_PIPELINE_NOW_US
#include <time.h>
static uint32_t audio_pipeline_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull
                      + (uint64_t)ts.tv_nsec / 1000ull);
}
#else
static uint32_t audio_pipeline_now_us(void) { return AUDIO_PIPELINE_NOW_US(); }
#endif
#else
#define audio_pipeline_now_us() 0u
#endif

static uint32_t audio_pipeline_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0) return AUDIO_PIPELINE_BACKEND_NE10;
    return 0u;
}

struct AudioPipeline {
    int aec_only;
    int enable_res;
    int legacy_amin;
    int enable_cng_effective;   /* aec_cfg.enable_cng && cfg.enable_cng */
    int sample_rate, hop, frame_sz, fft_sz, n_freqs;

    Aec*             aec;
    AnrDenoiser* nr;

    /* Scratch buffers (point into `pool`; all NULL when aec_only). There is no gain
     * copy: ctx.res_gain and anr_get_gain() alias the producers' buffers for the hop,
     * and mic/ref/out go straight to the AEC and the OLA. */
    AcIstft* istft;        /* iff !aec_only: inverse FFT, sqrt-Hann window, overlap-add */
    float*   g_total;      /* iff !aec_only */
    float*   extra;        /* iff !aec_only */
    Complex* spec;         /* iff !aec_only */
    float*   unity_gain;   /* iff !aec_only && unity_gain_needed(): the all-ones gain vector */

    uint32_t rng_state;

    /* Per-hop stage timing (see AudioPipelineLastTiming); always present so the carve
     * does not depend on the build flag. */
    AudioPipelineLastTiming last_timing;

    void*  pool;          /* sub-pool AFTER this struct: AEC+FFT+NR+scratch */
    size_t pool_size;
    void*  owned_heap;     /* non-NULL iff obtained via audio_pipeline_create(); freed by destroy() */
};

/* Returns 0 and fills the out-params, or -1 (NULL cfg; sample_rate outside
 * {8000,16000,48000}; fft_size outside the rate's grids; aec_preset/nr_mode outside
 * their enums; filter_length outside [0, SE_FILTER_LENGTH_MAX]; or a 0/1 flag
 * holding another value). This is the reject-first gate every entry point uses.
 * Delay settings are validated by lib/aec when it sizes the AEC. */
static int derive_dims_and_configs(const AudioPipelineConfig* cfg,
                                    AecConfig* aec_cfg, AnrConfig* nr_cfg,
                                    int* hop, int* frame_sz, int* fft_sz, int* n_freqs) {
    static const SeRules rules = {
        /* allow_8k */ 1, /* fixed_rate */ 0, /* fixed_fft */ 0, /* check_delay */ 0,
        /* lane_aligned */ 0, /* spatial_context */ SE_CONTEXT_PRESET, /* with_nr */ 1
    };
    SeRequest req;
    SeGrid grid;

    if (!cfg) return -1;
    if (!se_flag_valid(cfg->aec_only) || !se_flag_valid(cfg->enable_nr) ||
        !se_flag_valid(cfg->enable_res) || !se_flag_valid(cfg->enable_cng) ||
        !se_flag_valid(cfg->legacy_amin)) return -1;

    req.sample_rate = cfg->sample_rate;
    req.fft_size = cfg->fft_size;
    req.filter_length = cfg->filter_length;
    req.delay_mode = cfg->delay_mode;
    req.delay_num_filters = cfg->delay_num_filters;
    req.fixed_delay_samples = cfg->fixed_delay_samples;
    req.aec_preset = cfg->aec_preset;
    req.nr_mode = cfg->nr_mode;
    if (se_derive(&req, &rules, &grid, aec_cfg) != 0) return -1;

    *hop = grid.hop_size;
    *frame_sz = grid.fft_size;
    *fft_sz = grid.fft_size;
    *n_freqs = grid.n_freqs;
    *nr_cfg = se_anr_compose_config(cfg->sample_rate, grid.fft_size, grid.hop_size,
                                    cfg->nr_mode);
    return 0;
}

/* A disabled gain source is read as the unity vector; named once so sizing and carve
 * agree on when it exists. */
static int unity_gain_needed(int enable_nr, int enable_res) {
    return !enable_nr || !enable_res;
}

/* The synthesis object: inverse FFT, sqrt-Hann window and overlap-add state (it
 * carves its own FFT handle). */
static size_t synthesis_mem_size(int fft_sz) {
    AcStftConfig scfg;
    ac_stft_config_default(&scfg, fft_sz);
    return ac_istft_get_mem_size(&scfg);
}

static size_t pipeline_pool_size(const AecConfig* aec_cfg, const AnrConfig* nr_cfg,
                                  int hop, int frame_sz, int fft_sz, int n_freqs,
                                  int aec_only, int enable_nr, int enable_res) {
    (void)hop;       /* kept for signature symmetry with pipeline_build() */
    (void)frame_sz;
    size_t aec_sz     = aec_get_mem_size(aec_cfg);
    size_t synth_sz   = aec_only ? 0 : synthesis_mem_size(fft_sz);
    size_t nr_sz      = (aec_only || !enable_nr) ? 0 : se_anr_stage_mem_size(nr_cfg);

    size_t pipe = 0;
    if (!aec_only) {
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_total   */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* extra     */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(Complex)); /* spec      */
        if (unity_gain_needed(enable_nr, enable_res))
            pipe += ALIGN16((size_t)n_freqs * sizeof(float)); /* unity_gain */
    }

    return ALIGN16(aec_sz) + ALIGN16(synth_sz) + ALIGN16(nr_sz) + pipe;
}

static int pipeline_build(AudioPipeline* p, void* pool, size_t pool_size,
                           const AecConfig* aec_cfg, const AnrConfig* nr_cfg,
                           int hop, int frame_sz, int fft_sz, int n_freqs,
                           int aec_only, int enable_nr, int enable_res) {
    size_t needed = pipeline_pool_size(aec_cfg, nr_cfg, hop, frame_sz, fft_sz, n_freqs,
                                        aec_only, enable_nr, enable_res);
    if (!pool || pool_size < needed) {
        AP_LOG_ERR("audio_pipeline: sub-pool too small (%zu < %zu)\n", pool_size, needed);
        return -1;
    }
    uint8_t* ptr = (uint8_t*)pool;

    /* AEC (its internal FFT is separate from the synthesis object's). */
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    p->aec = aec_init(ptr, aec_sz, aec_cfg);
    ptr += ALIGN16(aec_sz);
    if (!p->aec) { AP_LOG_ERR("audio_pipeline: aec_init failed\n"); return -1; }

    if (!aec_only) {
        AcStftConfig scfg;
        size_t synth_sz = synthesis_mem_size(fft_sz);
        ac_stft_config_default(&scfg, fft_sz);
        p->istft = ac_istft_init(ptr, synth_sz, &scfg);
        ptr += ALIGN16(synth_sz);

        if (enable_nr) {
            size_t nr_sz = se_anr_stage_mem_size(nr_cfg);
            p->nr = se_anr_stage_init(ptr, nr_sz, nr_cfg);
            ptr += ALIGN16(nr_sz);
        }

        if (!p->istft || (enable_nr && !p->nr)) { AP_LOG_ERR("audio_pipeline: NR/FFT init failed\n"); return -1; }
    }

    if (!aec_only) {
        p->g_total   = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->extra     = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->spec      = (Complex*)ptr; ptr += ALIGN16((size_t)n_freqs  * sizeof(Complex));
        if (unity_gain_needed(enable_nr, enable_res)) {
            p->unity_gain = (float*)ptr; ptr += ALIGN16((size_t)n_freqs * sizeof(float));
            for (int k = 0; k < n_freqs; k++) p->unity_gain[k] = 1.0f;
        }

        /* Zero every pipeline-owned state buffer; the synthesis object zeroes its own. */
        memset(p->g_total,   0, (size_t)n_freqs  * sizeof(float));
        memset(p->extra,     0, (size_t)n_freqs  * sizeof(float));
        memset(p->spec,      0, (size_t)n_freqs  * sizeof(Complex));
    }

    /* n_freqs/hop agreement guard, checked at init before any sample is read. */
    AecResContext ctx0;
    aec_get_res_context(p->aec, &ctx0);
    if (ctx0.n_freqs != n_freqs || ctx0.hop_size != hop) {
        AP_LOG_ERR("audio_pipeline: FATAL grid mismatch -- pipeline n_freqs=%d hop=%d, "
                        "AEC n_freqs=%d hop=%d\n", n_freqs, hop, ctx0.n_freqs, ctx0.hop_size);
        return -1;
    }
    if (!aec_only && p->nr) {
        int nr_nf = anr_get_n_freqs(p->nr);
        if (nr_nf != n_freqs) {
            AP_LOG_ERR("audio_pipeline: FATAL grid mismatch -- pipeline n_freqs=%d, "
                            "nr n_freqs=%d\n", n_freqs, nr_nf);
            return -1;
        }
    }
    return 0;
}

static uint32_t fnv1a_str(const char* s, uint32_t h) {
    while (*s) { h ^= (uint32_t)(unsigned char)(*s++); h *= 16777619u; }
    return h;
}

static uint32_t audio_pipeline_build_flags_hash(void) {
    uint32_t h = 2166136261u;   /* FNV-1a 32-bit offset basis */
    h = fnv1a_str(AUDIO_PIPELINE_BACKEND_STR, h);
    /* Carve-order token list; bump AUDIO_PIPELINE_LAYOUT_VERSION whenever the buffer
     * set or order changes. */
    h = fnv1a_str("|carve:self(config-delay-v2),aec,istft(fft,window,acc,time,spec),nr?,"
                  "g_total,extra,spec,unity?", h);
    h = fnv1a_str("|align16", h);
    return ac_memreq_abi_hash(h);
}

AudioPipelineConfig audio_pipeline_default_config(int sample_rate) {
    AudioPipelineConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size    = 0;
    cfg.filter_length = 0;
    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = 5;
    cfg.fixed_delay_samples = -1;
    cfg.aec_preset  = AEC_PRESET_BALANCED;
    cfg.nr_mode     = ANR_MODE_BALANCED;
    cfg.aec_only    = 0;
    cfg.enable_nr   = 1;
    cfg.enable_res  = 1;
    cfg.enable_cng  = 1;
    cfg.legacy_amin = 0;
    return cfg;
}

int audio_pipeline_get_mem_requirements(const AudioPipelineConfig* cfg,
                                         AudioPipelineMemReq* out) {
    if (!cfg || !out) return -1;

    AecConfig aec_cfg; AnrConfig nr_cfg;
    int hop, frame_sz, fft_sz, n_freqs;
    if (derive_dims_and_configs(cfg, &aec_cfg, &nr_cfg, &hop, &frame_sz, &fft_sz, &n_freqs) != 0)
        return -1;

    /* The module validators return 0 for a rejected config; catch that here instead of
     * folding it into a too-small total. */
    size_t aec_sz = aec_get_mem_size(&aec_cfg);
    if (aec_sz == 0) return -1;
    if (!cfg->aec_only) {
        if (synthesis_mem_size(fft_sz) == 0 ||
            (cfg->enable_nr && se_anr_stage_mem_size(&nr_cfg) == 0)) return -1;
    }

    /* Reject up front a build whose backend maps to no known id (outside
     * pipelines/Makefile); a descriptor with backend_id 0 is never returned. */
    uint32_t backend_id = audio_pipeline_backend_id();
    if (backend_id == 0u) return -1;

    size_t sub_total = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs,
                                           cfg->aec_only, cfg->enable_nr, cfg->enable_res);
    size_t self_sz   = ALIGN16(sizeof(AudioPipeline));
    /* Operands are bounded well under SIZE_MAX (sub_total uses saturating ck_* helpers),
     * so a plain add is safe. */
    out->descriptor_version = AUDIO_PIPELINE_DESCRIPTOR_VERSION;
    out->layout_version     = AUDIO_PIPELINE_LAYOUT_VERSION;
    out->backend_id         = backend_id;
    out->build_flags_hash   = audio_pipeline_build_flags_hash();
    out->alignment          = 16u;
    out->reserved           = 0u;
    out->bytes              = (uint64_t)(self_sz + sub_total);
    return 0;
}

AudioPipeline* audio_pipeline_init_ex(void* mem, size_t bytes, const AudioPipelineConfig* cfg,
                                       const AudioPipelineMemReq* expected) {
    if (!mem || !cfg) return NULL;

    if (expected) {
        AudioPipelineMemReq cur;
        if (audio_pipeline_get_mem_requirements(cfg, &cur) != 0) {
            AP_LOG_ERR("audio_pipeline_init_ex: cfg rejected while recomputing the current "
                       "descriptor to validate `expected` against\n");
            return NULL;
        }
        if (expected->descriptor_version != cur.descriptor_version) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- descriptor_version mismatch "
                       "(expected=%u, current build=%u)\n",
                       expected->descriptor_version, cur.descriptor_version);
            return NULL;
        }
        if (expected->layout_version != cur.layout_version) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- layout_version mismatch "
                       "(expected=%u, current build=%u)\n",
                       expected->layout_version, cur.layout_version);
            return NULL;
        }
        if (expected->backend_id != cur.backend_id) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- backend_id mismatch "
                       "(expected=%u, current build=%u)\n",
                       expected->backend_id, cur.backend_id);
            return NULL;
        }
        if (expected->build_flags_hash != cur.build_flags_hash) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- build_flags_hash mismatch "
                       "(expected=0x%08x, current build=0x%08x)\n",
                       expected->build_flags_hash, cur.build_flags_hash);
            return NULL;
        }
        if (expected->alignment != cur.alignment) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- alignment mismatch "
                       "(expected=%u, current build=%u)\n",
                       expected->alignment, cur.alignment);
            return NULL;
        }
        if (expected->reserved != 0u) {
            /* `expected` may arrive from persisted bytes: reserved == 0 is validated, not
             * assumed. */
            AP_LOG_ERR("audio_pipeline_init_ex: corrupt descriptor -- reserved must be 0 "
                       "(got %u)\n", expected->reserved);
            return NULL;
        }
        if (expected->bytes < cur.bytes) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- expected->bytes too small "
                       "for the current build (expected->bytes=%llu < current requirement=%llu)\n",
                       (unsigned long long)expected->bytes, (unsigned long long)cur.bytes);
            return NULL;
        }
        if ((uint64_t)bytes < cur.bytes) {
            AP_LOG_ERR("audio_pipeline_init_ex: pool too small for the current build's "
                       "requirement (bytes=%zu < current requirement=%llu)\n",
                       bytes, (unsigned long long)cur.bytes);
            return NULL;
        }
    }

    if (!MEM_IS_ALIGNED16(mem)) {
        AP_LOG_ERR("audio_pipeline_init: pool not 16-byte aligned (%p)\n", mem);
        return NULL;
    }

    AecConfig aec_cfg; AnrConfig nr_cfg;
    int hop, frame_sz, fft_sz, n_freqs;
    if (derive_dims_and_configs(cfg, &aec_cfg, &nr_cfg, &hop, &frame_sz, &fft_sz, &n_freqs) != 0)
        return NULL;

    size_t self_sz = ALIGN16(sizeof(AudioPipeline));
    if (bytes < self_sz) {
        AP_LOG_ERR("audio_pipeline_init: pool too small for the control block (%zu < %zu)\n",
                bytes, self_sz);
        return NULL;
    }
    size_t sub_bytes = bytes - self_sz;
    size_t sub_needed = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs,
                                            cfg->aec_only, cfg->enable_nr, cfg->enable_res);
    if (sub_needed == 0 || sub_bytes < sub_needed) {
        AP_LOG_ERR("audio_pipeline_init: pool too small (%zu available < %zu needed)\n",
                sub_bytes, sub_needed);
        return NULL;
    }

    AudioPipeline* p = (AudioPipeline*)mem;
    memset(p, 0, sizeof(*p));
    void* sub_pool = (uint8_t*)mem + self_sz;

    if (pipeline_build(p, sub_pool, sub_bytes, &aec_cfg, &nr_cfg,
                        hop, frame_sz, fft_sz, n_freqs, cfg->aec_only,
                        cfg->enable_nr, cfg->enable_res) != 0) {
        return NULL;
    }

    p->aec_only             = cfg->aec_only;
    p->legacy_amin          = cfg->legacy_amin;
    p->enable_res           = cfg->enable_res;
    /* Comfort noise fills only bins the AEC residual gain cut. */
    p->enable_cng_effective = aec_cfg.enable_cng && cfg->enable_cng && cfg->enable_res;
    p->sample_rate          = cfg->sample_rate;
    p->hop = hop; p->frame_sz = frame_sz; p->fft_sz = fft_sz; p->n_freqs = n_freqs;
    p->rng_state  = SE_CNG_RNG_SEED;
    p->pool       = sub_pool;
    p->pool_size  = sub_bytes;
    p->owned_heap = NULL;
    return p;
}

AudioPipeline* audio_pipeline_init(void* mem, size_t bytes, const AudioPipelineConfig* cfg) {
    return audio_pipeline_init_ex(mem, bytes, cfg, NULL);
}

AudioPipeline* audio_pipeline_create(const AudioPipelineConfig* cfg) {
    if (!cfg) return NULL;

    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(cfg, &req) != 0) return NULL;

    void* mem = NULL;
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0 || !mem) {
        return NULL;
    }

    AudioPipeline* p = audio_pipeline_init(mem, (size_t)req.bytes, cfg);
    if (!p) { free(mem); return NULL; }

    p->owned_heap = mem;
    return p;
}

/* The AEC block is copied through from the AEC; it is governed by AEC_STAGE_TIMING,
 * not by AUDIO_PIPELINE_STAGE_TIMING. */
#if AEC_STAGE_TIMING
static void audio_pipeline_copy_aec_timing(AudioPipeline* p) {
    aec_get_last_timing(p->aec, &p->last_timing.aec);
}
#else
/* Guarded because aec_get_last_timing() is a cross-TU call; with the AEC half off
 * the record keeps its init-time zero. */
#define audio_pipeline_copy_aec_timing(p) ((void)(p))
#endif

/* Analysis stage of a hop: context-only linear AEC, then the borrowed view of its
 * spectra (valid until the next AEC processing call). */
static void analyse_hop(AudioPipeline* p, const float* mic, const float* ref,
                        SeSpecFrame* f) {
    AecResContext ctx;
    AecLinearContext lin;
    aec_process_context(p->aec, mic, ref);
    audio_pipeline_copy_aec_timing(p);
    aec_get_res_context(p->aec, &ctx);
    aec_get_linear_context(p->aec, &lin);
    /* enable_res=0/return_res_context=1 is set for every !aec_only instance, so the
     * seam pointers are non-NULL. */
    f->n_freqs = p->n_freqs;
    f->hop = p->hop;
    f->e = ctx.error_spec;
    f->y = ctx.near_spec;
    f->x = ctx.far_spec;
    f->r2 = ctx.r2;
    f->cn = ctx.comfort_noise;
    f->g_res_ctx = ctx.res_gain;
    f->far_hop = lin.aligned_far_hop;
    f->formed_hop = ctx.formed_hop;
    f->far_power = ctx.far_power;
    f->max_dt = ctx.dt_indicator;
    f->max_sat = ctx.saturation_level;
    f->all_converged = ctx.filter_converged;
    f->usable_linear = ctx.usable_linear;
    f->clock_drift = 0;
    f->delay_changed = lin.delay_state == AEC_LINEAR_DELAY_CHANGED;
    f->delay_locked = lin.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
}

/* One IFFT + sqrt-Hann synthesis + OLA, then hop emit/shift, with the synth timing
 * stamp; shared by audio_pipeline_process() and audio_pipeline_synthesize(). */
static void synthesize_hop(AudioPipeline* p, const Complex* spectrum,
                           float* out) {
    uint32_t t2 = audio_pipeline_now_us();
    ac_istft_push(p->istft, spectrum, out);
    p->last_timing.synth_us = audio_pipeline_now_us() - t2;
}

int audio_pipeline_process(AudioPipeline* p, const float* mic, const float* ref, float* out) {
    SeSpecFrame f;
    const float *nr_gain, *res_gain;
    uint32_t t0, t1;

    if (!p || !mic || !ref || !out) return -1;

    /* Every field of last_timing a mode owns is rewritten each hop; the others keep
     * their init-time zero. */

    /* aec_only: the AEC's output is the pipeline output, written directly into `out`.
     * Otherwise aec_process_context() skips that copy. */
    if (p->aec_only) {
        aec_process(p->aec, mic, ref, out);
        audio_pipeline_copy_aec_timing(p);
        return 0;
    }
    analyse_hop(p, mic, ref, &f);
    t0 = audio_pipeline_now_us();

    /* Noise-reduction gain: r2 folds into the noise floor (xi = S^2/(N^2+R^2)) through
     * p->extra; legacy_amin gives the denoiser no prior. */
    nr_gain = p->unity_gain;
    if (p->nr) {
        nr_gain = se_anr_stage_gain(p->nr, &f, p->extra, !p->legacy_amin);
        if (!nr_gain) return -1;
    }
    t1 = audio_pipeline_now_us();
    p->last_timing.nr_us = p->nr ? t1 - t0 : 0;
    res_gain = p->enable_res ? f.g_res_ctx : p->unity_gain;

    /* g_total = min(G_nr, G_res). Pre-min G_res sets the comfort-noise level so the fill
     * reflects AEC suppression only. */
    sk_min_f32(p->g_total, nr_gain, res_gain, p->n_freqs);

    sk_capply_gain_f32(p->spec, f.e, p->g_total, p->n_freqs);

    /* Comfort noise on the cut bins: level sqrt(N^2/PSD_SCALE) scaled by
     * sqrt(1 - G_res^2) (bins 1..N-2) and by G_nr (floored at SE_CNG_NR_GAIN_FLOOR), as
     * the denoiser never sees the fill. */
    if (p->enable_cng_effective && f.cn)
        aec_cng_fill(&p->rng_state, p->spec, f.cn, f.g_res_ctx,
                     nr_gain, SE_CNG_NR_GAIN_FLOOR, p->n_freqs);

    p->last_timing.post_us = audio_pipeline_now_us() - t1;

    synthesize_hop(p, p->spec, out);
    return 0;
}

int audio_pipeline_process_spec(AudioPipeline* p, const float* mic,
                                const float* ref, SeSpecFrame* frame) {
    if (!p || !mic || !ref || !frame || p->aec_only) return -1;
    analyse_hop(p, mic, ref, frame);
    return 0;
}

int audio_pipeline_synthesize(AudioPipeline* p, const Complex* spectrum,
                              float* out) {
    if (!p || !spectrum || !out || p->aec_only) return -1;
    synthesize_hop(p, spectrum, out);
    return 0;
}

int audio_pipeline_set_aec_preset(AudioPipeline* p, AecPreset preset,
                                  float ramp_ms) {
    if (!p || !p->aec) return -1;
    /* aec_set_preset updates the AEC's own AecConfig, the single authoritative copy. */
    return aec_set_preset(p->aec, preset, ramp_ms);
}

int audio_pipeline_set_nr_mode(AudioPipeline* p, AnrMode mode) {
    if (!p || !p->nr) return -1;   /* aec_only builds have no denoiser */
    /* The denoiser's AnrConfig is the authoritative copy. */
    return se_anr_stage_set_mode(p->nr, p->sample_rate, p->fft_sz, p->hop, mode);
}

void audio_pipeline_reset(AudioPipeline* p) {
    if (!p) return;

    aec_reset(p->aec);
    if (!p->aec_only) {
        if (p->nr) se_anr_stage_reset(p->nr);

        ac_istft_reset(p->istft);
        memset(p->g_total,  0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->extra,    0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->spec,     0, (size_t)p->n_freqs  * sizeof(Complex));
    }

    p->rng_state = SE_CNG_RNG_SEED;
}

void audio_pipeline_destroy(AudioPipeline* p) {
    if (!p) return;
    /* Reverse carve order: NR -> synthesis -> AEC (no-ops for a pool-resident instance). */
    if (p->nr)  se_anr_stage_destroy(p->nr);
    if (p->aec) aec_destroy(p->aec);

    if (p->owned_heap) {
        void* heap = p->owned_heap;
        free(heap);   /* frees `p` itself too (create() carves p at mem[0]) */
    }
}

int audio_pipeline_hop_size(const AudioPipeline* p)    { return p ? p->hop        : -1; }
int audio_pipeline_n_freqs(const AudioPipeline* p)     { return p ? p->n_freqs    : -1; }
int audio_pipeline_sample_rate(const AudioPipeline* p) { return p ? p->sample_rate: -1; }

Aec* audio_pipeline_get_aec(const AudioPipeline* p) { return p ? p->aec : NULL; }
AnrDenoiser* audio_pipeline_get_nr(const AudioPipeline* p) { return p ? p->nr : NULL; }

int audio_pipeline_get_mem_breakdown(const AudioPipelineConfig* cfg,
                                      AudioPipelineMemBreakdown* out) {
    if (!cfg || !out) return -1;

    AecConfig aec_cfg; AnrConfig nr_cfg;
    int hop, frame_sz, fft_sz, n_freqs;
    if (derive_dims_and_configs(cfg, &aec_cfg, &nr_cfg, &hop, &frame_sz, &fft_sz, &n_freqs) != 0)
        return -1;

    size_t aec_sz = aec_get_mem_size(&aec_cfg);
    if (aec_sz == 0) return -1;
    size_t fft_mem = 0, nr_sz = 0;
    if (!cfg->aec_only) {
        fft_mem = fft_get_mem_size(fft_sz);
        if (cfg->enable_nr) nr_sz = se_anr_stage_mem_size(&nr_cfg);
        if (fft_mem == 0 || (cfg->enable_nr && nr_sz == 0)) return -1;
    }

    size_t sub_total = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs,
                                           cfg->aec_only, cfg->enable_nr, cfg->enable_res);
    size_t pipe_bufs = sub_total - ALIGN16(aec_sz) - ALIGN16(fft_mem) - ALIGN16(nr_sz);

    out->aec_bytes      = aec_sz;
    out->fft_bytes      = fft_mem;
    out->nr_bytes       = nr_sz;
    out->pipeline_bytes = pipe_bufs;
    out->hop = hop; out->frame_sz = frame_sz; out->fft_sz = fft_sz; out->n_freqs = n_freqs;
    return 0;
}

void audio_pipeline_get_last_timing(const AudioPipeline* p,
                                    AudioPipelineLastTiming* out) {
    if (!out) return;
    if (!p) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = p->last_timing;
}
