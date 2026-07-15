/**
 * audio_pipeline.c — implementation of audio_pipeline.h (review F20).
 *
 * Ports pipelines/aec_nr_pipeline_static.c's file-local `pipeline_pool_size` /
 * `pipeline_build` / `pipeline_destroy` (the pool-sizing/carving) and the
 * per-hop while-loop BODY from both CLIs' `main()` (the processing) into one
 * linkable TU, verbatim in carve order/sizes and per-hop arithmetic — see
 * audio_pipeline.h for the API contract and pipelines/README.md ("Board
 * Integration") for the consumer sequence.
 *
 * What changed vs. the CLI-embedded originals (behaviourally):
 *   - mic_buf/ref_buf/aec_out/... are now per-INSTANCE (an AudioPipeline
 *     field), not per-process-invocation locals — multiple instances (or one
 *     instance across an audio_pipeline_reset()) never share state.
 *   - The comfort-noise RNG and the near-end-floor hangover counter
 *     (`near_hang`) move from a file-global / a `main()` stack local into
 *     per-instance fields for the same reason — same seed (0x9e3779b9u), same
 *     xorshift+Box-Muller sequence, so a single instance's FIRST render is
 *     bit-for-bit identical to the old CLI's (see the anchors this file was
 *     gated against in the F20 review — 16k/8k/48k, both backends, both old
 *     binaries).
 *   - Every pipeline-owned scratch buffer is explicitly zeroed at carve time
 *     (audio_pipeline_init) instead of relying on the CLI's blanket
 *     `memset(pool, 0, total)` before pipeline_build ran — a caller handing
 *     in a poisoned pool (e.g. `memset(pool, 0xA5, bytes)`, the pattern
 *     lib/aec's own zero-heap test uses) now inits identically either way.
 *
 * Nothing about the DSP arithmetic itself changed — same operations, same
 * order, same constants (PROD_NE_FLOOR/PROD_FAR_GATE_THRESH/PSD_SCALE/...).
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "audio_pipeline.h"
#include "fft_wrapper.h"       /* fft_get_mem_size/fft_init/fft_inverse/fft_destroy, Complex, ALIGN16 */
#include "simd_kernels.h"      /* sk_min_f32 / sk_capply_gain_f32                                     */
#include "pipeline_dims.h"     /* compute_frame_dims() -- shared with both CLIs                       */

/* Deliberately NO "wav_io.h" here: this TU takes raw float* mic/ref/out
 * buffers only (audio_pipeline_process) -- no WAV path, no stdio-based I/O.
 * WAV reading/writing stays entirely in the CLIs (aec_nr_pipeline.c /
 * aec_nr_pipeline_static.c), which are now thin shells over this API. */

/* ============================================================================
 * No-stdio build gate (re-review R09)
 * ========================================================================== */

/**
 * This TU's diagnostics (init/build-time reject reasons only -- nothing on
 * the per-hop audio_pipeline_process() path ever logs) are advisory: every
 * failure this file can hit is ALSO signalled through its return value
 * (NULL / -1), so a caller that cannot or will not link libc's stdio (a
 * board/firmware image with no console, or one that forbids the stdio
 * symbol set outright) must still get a fully-functional library -- it
 * simply loses the human-readable "why" that would otherwise go to stderr.
 *
 * -DAUDIO_PIPELINE_NO_STDIO (pipelines/Makefile's `NO_STDIO=1`) compiles
 * every AP_LOG_ERR() call below to a no-op and drops the <stdio.h> include
 * entirely, so a NO_STDIO build of audio_pipeline.o pulls in none of
 * fprintf/printf/puts/fputs/stderr (see `make audit-no-stdio`, which
 * verifies exactly this with `nm` over the archive). Default (unset)
 * behaviour is unchanged: AP_LOG_ERR() is fprintf(stderr, ...), same
 * wording as before this gate existed.
 */
#ifndef AUDIO_PIPELINE_NO_STDIO
#include <stdio.h>
#define AP_LOG_ERR(...) fprintf(stderr, __VA_ARGS__)
#else
#define AP_LOG_ERR(...) ((void)0)
#endif

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/* Production recipe constants (mirror Python PROD_* in aec_nr_pipeline.py and
 * both CLIs' own copies -- must stay byte-identical). */
#define PROD_NE_FLOOR             0.4f
#define PROD_NE_FLOOR_FAR_ACTIVE  0.2f
#define PROD_FAR_GATE_THRESH      1e-4f
#define PROD_NEAR_GATE_THRESH     1e-3f
#define PROD_NEAR_HANGOVER        8
#define PSD_SCALE                 (32768.0f * 32768.0f)  /* int16^2 (Python _PSD_SCALE) */

/* Comfort-noise RNG seed -- identical constant to both CLIs' old file-global
 * g_rng, so the first hop's noise sequence out of a fresh instance matches
 * theirs exactly. */
#define AUDIO_PIPELINE_RNG_SEED 0x9e3779b9u

/* This file's own carve-layout version (see audio_pipeline.h's
 * AudioPipelineMemReq doc for the bump rule). */
#define AUDIO_PIPELINE_LAYOUT_VERSION 1u

/* Compile-time FFT backend identity. pipelines/Makefile passes
 * -DAUDIO_PIPELINE_BACKEND_STR=\"kiss\" or \"ne10\" to match its own
 * BACKEND= selection (the same backend libaudio_common.a this TU links
 * against); the fallback below only fires if someone compiles this file
 * outside that Makefile. */
#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

/* ============================================================================
 * Instance
 * ========================================================================== */

struct AudioPipeline {
    /* effective per-instance config (already resolved from AudioPipelineConfig
     * + preset -- see derive_dims_and_configs) */
    int aec_only;
    int legacy_amin;
    int enable_cng_effective;   /* aec_cfg.enable_cng (preset, always 1 today) && cfg.enable_cng */
    int sample_rate, hop, frame_sz, fft_sz, n_freqs;

    /* sub-modules (point into `pool` below; NULL fft/nr iff aec_only) */
    Aec*             aec;
    FftHandle*       fft;
    MmseLsaDenoiser* nr;

    /* the pipeline's 13 scratch buffers (point into `pool` below; the nine
     * marked "iff !aec_only" are NULL when aec_only) */
    float*   mic_buf;
    float*   ref_buf;
    float*   aec_out;
    float*   out_buf;      /* iff !aec_only */
    float*   synth_win;    /* iff !aec_only */
    float*   ola;          /* iff !aec_only */
    float*   ifft_buf;     /* iff !aec_only */
    float*   g_nr;         /* iff !aec_only */
    float*   g_total;      /* iff !aec_only */
    float*   g_aec;        /* iff !aec_only */
    float*   extra;        /* iff !aec_only */
    float*   e2;           /* iff !aec_only */
    Complex* spec;         /* iff !aec_only */

    /* per-instance comfort-noise RNG + near-end-floor hangover counter */
    uint32_t rng_state;
    int      near_hang;

    /* pool bookkeeping */
    void*  pool;          /* sub-pool AFTER this struct: AEC+FFT+NR+scratch */
    size_t pool_size;
    void*  owned_heap;     /* non-NULL iff obtained via audio_pipeline_create(); freed by destroy() */
};

/* ============================================================================
 * Config -> module configs + frame dims (shared by every entry point that
 * needs to know the carve/process shape for a given AudioPipelineConfig)
 * ========================================================================== */

/* Returns 0 and fills every out-param, or -1 (NULL cfg; sample_rate outside
 * the {8000,16000,48000} whitelist; aec_preset/nr_mode outside their defined
 * enum values; or aec_only/enable_cng/legacy_amin holding anything but 0/1)
 * -- ALL checked HERE, up front, before a single module config is derived,
 * so an invalid AudioPipelineConfig never reaches aec_config_from_preset/
 * mmse_lsa_config_for_mode. This is the one reject-first gate every entry
 * point (get_mem_requirements, init, get_mem_breakdown, and create() via
 * get_mem_requirements+init) already funnels through.
 *
 * Without this, an out-of-enum aec_preset/nr_mode would silently fall
 * through aec_config_from_preset's/mmse_lsa_config_for_mode's own internal
 * `default:` case (a documented fallback to balanced, not an error -- see
 * aec.c/mmse_lsa_types.h), and an out-of-{0,1} bool would just be treated as
 * truthy by every `if (cfg->x)` downstream in audio_pipeline_process --
 * neither is a loud failure on its own, so a caller passing a garbage/
 * adversarial config would never find out short of this explicit check.
 * Mirrors the derivation both CLIs' main() used to do inline. */
static int derive_dims_and_configs(const AudioPipelineConfig* cfg,
                                    AecConfig* aec_cfg, MmseLsaConfig* nr_cfg,
                                    int* hop, int* frame_sz, int* fft_sz, int* n_freqs) {
    if (!cfg) return -1;
    if (!aec_is_valid_sample_rate(cfg->sample_rate)) return -1;

#define AP_CK_BOOL(field) \
    do { if (cfg->field != 0 && cfg->field != 1) return -1; } while (0)

    switch (cfg->aec_preset) {
        case AEC_PRESET_MILD:
        case AEC_PRESET_BALANCED:
        case AEC_PRESET_AGGRESSIVE:
            break;
        default:
            return -1;
    }
    switch (cfg->nr_mode) {
        case MMSE_LSA_NR_MILD:
        case MMSE_LSA_NR_MODERATE:
        case MMSE_LSA_NR_BALANCED:
        case MMSE_LSA_NR_AGGRESSIVE:
            break;
        default:
            return -1;
    }
    AP_CK_BOOL(aec_only);
    AP_CK_BOOL(enable_cng);
    AP_CK_BOOL(legacy_amin);

#undef AP_CK_BOOL

    aec_config_from_preset(aec_cfg, cfg->aec_preset, cfg->sample_rate);
    aec_cfg->enable_res         = 0;   /* linear AEC + external NR/RES seam */
    aec_cfg->return_res_context = 1;

    *nr_cfg = mmse_lsa_config_for_mode(cfg->sample_rate, cfg->nr_mode);
    /* Match the Python pipeline _build_denoiser STRUCTURAL tuning (identical
     * to both CLIs' prior inline copies): L=150, alpha_d=0.95, alpha_attack/
     * alpha_decay pinned off the C-only per-mode values. See
     * aec_nr_pipeline.py:_build_denoiser. */
    nr_cfg->L            = 150;
    nr_cfg->alpha_d      = 0.95f;
    nr_cfg->alpha_attack = 0.3f;
    nr_cfg->alpha_decay  = nr_cfg->alpha_g;

    compute_frame_dims(cfg->sample_rate, hop, frame_sz, fft_sz, n_freqs);
    /* Force the NR config onto the SAME fft grid the pipeline/AEC derive (the
     * "8 kHz FFT mismatch" fix both CLIs already carried -- see
     * pipeline_dims.h). At >=12.8 kHz this is a no-op. */
    nr_cfg->fft_size   = *fft_sz;
    nr_cfg->frame_size = *frame_sz;
    nr_cfg->hop_size   = *hop;
    return 0;
}

/* ============================================================================
 * Pool sizing (verbatim port of aec_nr_pipeline_static.c's file-local
 * pipeline_pool_size -- same field order, same ALIGN16 bumps)
 * ========================================================================== */

static size_t pipeline_pool_size(const AecConfig* aec_cfg, const MmseLsaConfig* nr_cfg,
                                  int hop, int frame_sz, int fft_sz, int n_freqs,
                                  int aec_only) {
    size_t aec_sz     = aec_get_mem_size(aec_cfg);
    size_t fft_sz_mem = aec_only ? 0 : fft_get_mem_size(fft_sz);
    size_t nr_sz      = aec_only ? 0 : mmse_lsa_get_mem_size(nr_cfg);

    size_t pipe = 0;
    pipe += ALIGN16((size_t)hop * sizeof(float));   /* mic_buf */
    pipe += ALIGN16((size_t)hop * sizeof(float));   /* ref_buf */
    pipe += ALIGN16((size_t)hop * sizeof(float));   /* aec_out */
    if (!aec_only) {
        pipe += ALIGN16((size_t)frame_sz * sizeof(float));   /* synth_win */
        pipe += ALIGN16((size_t)frame_sz * sizeof(float));   /* ola       */
        pipe += ALIGN16((size_t)fft_sz   * sizeof(float));   /* ifft_buf  */
        pipe += ALIGN16((size_t)hop      * sizeof(float));   /* out_buf   */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_nr      */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_total   */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_aec     */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* extra     */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* e2        */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(Complex)); /* spec      */
    }

    return ALIGN16(aec_sz) + ALIGN16(fft_sz_mem) + ALIGN16(nr_sz) + pipe;
}

/* ============================================================================
 * Carve (verbatim port of aec_nr_pipeline_static.c's file-local
 * pipeline_build, PLUS an explicit zero of each of the 13 pipeline buffers --
 * see audio_pipeline.h's audio_pipeline_init doc for why)
 * ========================================================================== */

static int pipeline_build(AudioPipeline* p, void* pool, size_t pool_size,
                           const AecConfig* aec_cfg, const MmseLsaConfig* nr_cfg,
                           int hop, int frame_sz, int fft_sz, int n_freqs,
                           int aec_only) {
    size_t needed = pipeline_pool_size(aec_cfg, nr_cfg, hop, frame_sz, fft_sz, n_freqs, aec_only);
    if (!pool || pool_size < needed) {
        AP_LOG_ERR("audio_pipeline: sub-pool too small (%zu < %zu)\n", pool_size, needed);
        return -1;
    }
    uint8_t* ptr = (uint8_t*)pool;

    /* AEC (its own internal post_fft is sized/placed inside aec_get_mem_size /
     * aec_init -- a separate FFT instance from the pipeline's own `fft` below). */
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    p->aec = aec_init(ptr, aec_sz, aec_cfg);
    ptr += ALIGN16(aec_sz);
    if (!p->aec) { AP_LOG_ERR("audio_pipeline: aec_init failed\n"); return -1; }

    if (!aec_only) {
        size_t fft_mem = fft_get_mem_size(fft_sz);
        p->fft = fft_init(ptr, fft_mem, fft_sz);
        ptr += ALIGN16(fft_mem);

        size_t nr_sz = mmse_lsa_get_mem_size(nr_cfg);
        p->nr = mmse_lsa_init(ptr, nr_sz, nr_cfg);
        ptr += ALIGN16(nr_sz);

        if (!p->fft || !p->nr) { AP_LOG_ERR("audio_pipeline: NR/FFT init failed\n"); return -1; }
    }

    p->mic_buf = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    p->ref_buf = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    p->aec_out = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    memset(p->mic_buf, 0, (size_t)hop * sizeof(float));
    memset(p->ref_buf, 0, (size_t)hop * sizeof(float));
    memset(p->aec_out, 0, (size_t)hop * sizeof(float));

    if (!aec_only) {
        p->synth_win = (float*)ptr;   ptr += ALIGN16((size_t)frame_sz * sizeof(float));
        p->ola       = (float*)ptr;   ptr += ALIGN16((size_t)frame_sz * sizeof(float));
        p->ifft_buf  = (float*)ptr;   ptr += ALIGN16((size_t)fft_sz   * sizeof(float));
        p->out_buf   = (float*)ptr;   ptr += ALIGN16((size_t)hop      * sizeof(float));
        p->g_nr      = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->g_total   = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->g_aec     = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->extra     = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->e2        = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->spec      = (Complex*)ptr; ptr += ALIGN16((size_t)n_freqs  * sizeof(Complex));

        /* Explicit zero of every pipeline-owned STATE buffer (F20: a dirty
         * pool must be safe without the caller's blanket memset). `synth_win`
         * is zeroed too even though the fill loop right below overwrites all
         * frame_sz elements unconditionally -- it holds a deterministic
         * constant, not accumulated state, so this memset is redundant
         * belt-and-braces, not load-bearing, but keeps the "all 13 buffers
         * explicitly zeroed at carve time" contract literal and unambiguous. */
        memset(p->synth_win, 0, (size_t)frame_sz * sizeof(float));
        memset(p->ola,       0, (size_t)frame_sz * sizeof(float));
        memset(p->ifft_buf,  0, (size_t)fft_sz   * sizeof(float));
        memset(p->out_buf,   0, (size_t)hop      * sizeof(float));
        memset(p->g_nr,      0, (size_t)n_freqs  * sizeof(float));
        memset(p->g_total,   0, (size_t)n_freqs  * sizeof(float));
        memset(p->g_aec,     0, (size_t)n_freqs  * sizeof(float));
        memset(p->extra,     0, (size_t)n_freqs  * sizeof(float));
        memset(p->e2,        0, (size_t)n_freqs  * sizeof(float));
        memset(p->spec,      0, (size_t)n_freqs  * sizeof(Complex));

        /* sqrt of periodic Hann (denom = block_size) -- matches Python run_res
         * synth_win = sqrt(0.5*(1 - cos(2*pi*k/block_size))), byte-identical
         * to both CLIs' old synthesis window. */
        for (int k = 0; k < frame_sz; k++)
            p->synth_win[k] = sqrtf(0.5f * (1.0f - cosf(2.0f * M_PI_F * k / frame_sz)));
    }

    /* --- n_freqs/hop agreement guard (the "8 kHz FFT mismatch" fix) ---
     * aec_get_res_context() is readable right after aec_init (a->n_freqs /
     * a->hop_size are set at init time, before any frame is processed), so
     * this check runs at INIT, before a single sample is read. */
    AecResContext ctx0;
    aec_get_res_context(p->aec, &ctx0);
    if (ctx0.n_freqs != n_freqs || ctx0.hop_size != hop) {
        AP_LOG_ERR("audio_pipeline: FATAL grid mismatch -- pipeline n_freqs=%d hop=%d, "
                        "AEC n_freqs=%d hop=%d\n", n_freqs, hop, ctx0.n_freqs, ctx0.hop_size);
        return -1;
    }
    if (!aec_only) {
        int fft_nf = fft_get_n_freqs(p->fft);
        int nr_nf  = mmse_lsa_get_n_freqs(p->nr);
        if (fft_nf != n_freqs || nr_nf != n_freqs) {
            AP_LOG_ERR("audio_pipeline: FATAL grid mismatch -- pipeline n_freqs=%d, "
                            "fft n_freqs=%d, nr n_freqs=%d\n", n_freqs, fft_nf, nr_nf);
            return -1;
        }
    }
    return 0;
}

/* ============================================================================
 * Build-flags hash (FNV-1a-32)
 * ========================================================================== */

static uint32_t fnv1a_str(const char* s, uint32_t h) {
    while (*s) { h ^= (uint32_t)(unsigned char)(*s++); h *= 16777619u; }
    return h;
}

static uint32_t audio_pipeline_build_flags_hash(void) {
    uint32_t h = 2166136261u;   /* FNV-1a 32-bit offset basis */
    h = fnv1a_str(AUDIO_PIPELINE_BACKEND_STR, h);
    /* Literal carve-order token list -- bump AUDIO_PIPELINE_LAYOUT_VERSION
     * (and update this string) whenever the buffer set/order changes. */
    h = fnv1a_str("|carve:aec,fft,nr,mic_buf,ref_buf,aec_out,synth_win,ola,"
                  "ifft_buf,out_buf,g_nr,g_total,g_aec,extra,e2,spec", h);
    h = fnv1a_str("|align16", h);
    return h;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

AudioPipelineConfig audio_pipeline_default_config(int sample_rate) {
    AudioPipelineConfig cfg;
    cfg.sample_rate = sample_rate;
    cfg.aec_preset  = AEC_PRESET_BALANCED;
    cfg.nr_mode     = MMSE_LSA_NR_BALANCED;
    cfg.aec_only    = 0;
    cfg.enable_cng  = 1;
    cfg.legacy_amin = 0;
    return cfg;
}

int audio_pipeline_get_mem_requirements(const AudioPipelineConfig* cfg,
                                         AudioPipelineMemReq* out) {
    if (!cfg || !out) return -1;

    AecConfig aec_cfg; MmseLsaConfig nr_cfg;
    int hop, frame_sz, fft_sz, n_freqs;
    if (derive_dims_and_configs(cfg, &aec_cfg, &nr_cfg, &hop, &frame_sz, &fft_sz, &n_freqs) != 0)
        return -1;

    /* Module validators (F05-style): aec_get_mem_size/mmse_lsa_get_mem_size
     * both return 0 for a config their own aec_validate_config/
     * mmse_lsa_validate_config rejects -- catch that explicitly here (rather
     * than let it silently fold into a too-small `pipe`-only total below). */
    size_t aec_sz = aec_get_mem_size(&aec_cfg);
    if (aec_sz == 0) return -1;
    if (!cfg->aec_only) {
        size_t fft_mem = fft_get_mem_size(fft_sz);
        size_t nr_sz   = mmse_lsa_get_mem_size(&nr_cfg);
        if (fft_mem == 0 || nr_sz == 0) return -1;
    }

    size_t sub_total = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs,
                                           cfg->aec_only);
    size_t self_sz   = ALIGN16(sizeof(AudioPipeline));
    /* self_sz + sub_total: both operands are bounded well under SIZE_MAX in
     * any realistic config (sub_total already went through mem_align.h's
     * saturating ck_* helpers inside each _get_mem_size call), so a plain
     * add is safe here. */
    out->bytes            = self_sz + sub_total;
    out->alignment        = 16;
    out->layout_version   = AUDIO_PIPELINE_LAYOUT_VERSION;
    out->backend          = AUDIO_PIPELINE_BACKEND_STR;
    out->build_flags_hash = audio_pipeline_build_flags_hash();
    return 0;
}

/* ============================================================================
 * audio_pipeline_init_ex (re-review R09) — audio_pipeline_init() PLUS an
 * optional `expected` descriptor gate. See audio_pipeline.h for the full
 * contract; the six-condition check below (run only when `expected` is
 * non-NULL) is the literal implementation of that doc's numbered list, in
 * the SAME order, each on its own named diagnostic.
 * ========================================================================== */

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
        if (expected->layout_version != cur.layout_version) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- layout_version mismatch "
                       "(expected=%u, current build=%u)\n",
                       expected->layout_version, cur.layout_version);
            return NULL;
        }
        if (!expected->backend || strcmp(expected->backend, cur.backend) != 0) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- backend mismatch "
                       "(expected=\"%s\", current build=\"%s\")\n",
                       expected->backend ? expected->backend : "(null)", cur.backend);
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
                       "(expected=%zu, current build=%zu)\n",
                       expected->alignment, cur.alignment);
            return NULL;
        }
        if (expected->bytes < cur.bytes) {
            AP_LOG_ERR("audio_pipeline_init_ex: stale descriptor -- expected->bytes too small "
                       "for the current build (expected->bytes=%zu < current requirement=%zu)\n",
                       expected->bytes, cur.bytes);
            return NULL;
        }
        if (bytes < cur.bytes) {
            AP_LOG_ERR("audio_pipeline_init_ex: pool too small for the current build's "
                       "requirement (bytes=%zu < current requirement=%zu)\n",
                       bytes, cur.bytes);
            return NULL;
        }
    }

    if (!MEM_IS_ALIGNED16(mem)) {
        AP_LOG_ERR("audio_pipeline_init: pool not 16-byte aligned (%p)\n", mem);
        return NULL;
    }

    AecConfig aec_cfg; MmseLsaConfig nr_cfg;
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
                                            cfg->aec_only);
    if (sub_needed == 0 || sub_bytes < sub_needed) {
        AP_LOG_ERR("audio_pipeline_init: pool too small (%zu available < %zu needed)\n",
                sub_bytes, sub_needed);
        return NULL;
    }

    AudioPipeline* p = (AudioPipeline*)mem;
    memset(p, 0, sizeof(*p));
    void* sub_pool = (uint8_t*)mem + self_sz;

    if (pipeline_build(p, sub_pool, sub_bytes, &aec_cfg, &nr_cfg,
                        hop, frame_sz, fft_sz, n_freqs, cfg->aec_only) != 0) {
        return NULL;
    }

    p->aec_only            = cfg->aec_only;
    p->legacy_amin          = cfg->legacy_amin;
    p->enable_cng_effective = aec_cfg.enable_cng && cfg->enable_cng;
    p->sample_rate          = cfg->sample_rate;
    p->hop = hop; p->frame_sz = frame_sz; p->fft_sz = fft_sz; p->n_freqs = n_freqs;
    p->rng_state  = AUDIO_PIPELINE_RNG_SEED;
    p->near_hang  = 0;
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
    if (posix_memalign(&mem, req.alignment, req.bytes) != 0 || !mem) {
        return NULL;
    }

    AudioPipeline* p = audio_pipeline_init(mem, req.bytes, cfg);
    if (!p) { free(mem); return NULL; }

    p->owned_heap = mem;
    return p;
}

/* ---- comfort-noise RNG (per-instance xorshift32 + Box-Muller; identical
 * arithmetic/seed to both CLIs' old file-global generator) ---- */
static float rng_uniform(AudioPipeline* p) {                    /* (0,1) */
    uint32_t x = p->rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    p->rng_state = x;
    return ((x >> 8) + 0.5f) * (1.0f / 16777216.0f);
}
static float rng_gauss(AudioPipeline* p) {                       /* Box-Muller */
    float u1 = rng_uniform(p), u2 = rng_uniform(p);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI_F * u2);
}

int audio_pipeline_process(AudioPipeline* p, const float* mic, const float* ref, float* out) {
    if (!p || !mic || !ref || !out) return -1;

    const int hop      = p->hop;
    const int n_freqs   = p->n_freqs;
    const int frame_sz  = p->frame_sz;

    /* Copy the caller's mic/ref into pool-owned scratch: decouples the
     * caller's buffer alignment/lifetime from what aec_process() reads, and
     * is the byte-identical verbatim continuation of the carve (mic_buf/
     * ref_buf were always separate pool buffers, even in the old CLI, where
     * they were filled by wav_read_float() instead). */
    memcpy(p->mic_buf, mic, (size_t)hop * sizeof(float));
    memcpy(p->ref_buf, ref, (size_t)hop * sizeof(float));

    /* Stage 1: AEC (linear residual in aec_out; freq seam in ctx). */
    aec_process(p->aec, p->mic_buf, p->ref_buf, p->aec_out);

    if (p->aec_only) {
        memcpy(out, p->aec_out, (size_t)hop * sizeof(float));
        return 0;
    }

    AecResContext ctx;
    aec_get_res_context(p->aec, &ctx);
    if (!ctx.error_spec || !ctx.res_gain) {   /* seam unavailable -> linear fallback */
        memcpy(out, p->aec_out, (size_t)hop * sizeof(float));
        return 0;
    }

    /* Stage 2: echo-aware NR gain. extra = R^2/PSD_SCALE folds the residual
     * echo into the noise floor (xi = S^2/(N^2+R^2)); off in legacy. */
    const float* nr_extra = NULL;
    if (!p->legacy_amin && ctx.r2) {
        for (int k = 0; k < n_freqs; k++)
            p->extra[k] = ctx.r2[k] / PSD_SCALE;
        nr_extra = p->extra;
    }
    mmse_lsa_process_gain(p->nr, ctx.error_spec, nr_extra, p->g_nr);

    /* Stage 3a: g_total = min(G_nr, G_res). g_aec (= G_res, pre-min) sets the
     * comfort-noise level so CNG reflects AEC suppression only. */
    memcpy(p->g_aec, ctx.res_gain, (size_t)n_freqs * sizeof(float));
    sk_min_f32(p->g_total, p->g_nr, p->g_aec, n_freqs);

    /* |E(f)|^2 scratch hoist: both the near-energy mean below and the
     * echo-gated lift loop need re*re+im*im per bin. */
    for (int k = 0; k < n_freqs; k++) {
        float re = ctx.error_spec[k].r, im = ctx.error_spec[k].i;
        p->e2[k] = re * re + im * im;
    }

    /* Stage 3b: far-activity + near-VAD gated near-end floor strength. */
    float nf_eff = PROD_NE_FLOOR;
    if (!p->legacy_amin) {
        int far_active = ctx.far_power > PROD_FAR_GATE_THRESH;
        float ne = 0.0f;
        for (int k = 0; k < n_freqs; k++) {
            ne += p->e2[k];
        }
        ne /= (float)n_freqs;
        if (ne > PROD_NEAR_GATE_THRESH) p->near_hang = PROD_NEAR_HANGOVER;
        int near_active = p->near_hang > 0;
        if (p->near_hang > 0) p->near_hang--;
        int protect = (!far_active) && near_active;
        nf_eff = protect ? PROD_NE_FLOOR : PROD_NE_FLOOR_FAR_ACTIVE;
    }

    /* Per-bin echo-gated near-end lift (ne_gate='both': G_res*(1-echo_frac)). */
    if (nf_eff > 0.0f && ctx.r2) {
        for (int k = 0; k < n_freqs; k++) {
            float r2_nr = ctx.r2[k] / PSD_SCALE;
            float echo_frac = r2_nr / (p->e2[k] + 1e-12f);
            if (echo_frac < 0.0f) echo_frac = 0.0f;
            if (echo_frac > 1.0f) echo_frac = 1.0f;
            float no_echo = ctx.res_gain[k] * (1.0f - echo_frac);
            float lift = nf_eff * no_echo;
            p->g_total[k] = (1.0f - lift) * p->g_total[k] + lift;   /* blend toward 1 */
        }
    }

    /* S(f) = E(f) . g_total */
    sk_capply_gain_f32(p->spec, ctx.error_spec, p->g_total, n_freqs);

    /* Comfort noise on the cut bins: level = sqrt(N^2/PSD_SCALE), scaled by
     * sqrt(1 - g_aec^2) so it fills only what the AEC suppressed (bins
     * 1..N-2). */
    if (p->enable_cng_effective && ctx.comfort_noise) {
        for (int k = 1; k < n_freqs - 1; k++) {
            float n_amp = ctx.comfort_noise[k] / PSD_SCALE;
            n_amp = (n_amp > 0.0f) ? sqrtf(n_amp) : 0.0f;
            float ng2 = 1.0f - p->g_aec[k] * p->g_aec[k];
            float noise_gain = (ng2 > 0.0f) ? sqrtf(ng2) : 0.0f;
            float a = noise_gain * n_amp;
            p->spec[k].r += a * rng_gauss(p);
            p->spec[k].i += a * rng_gauss(p);
        }
    }

    /* irfft -> sqrt-Hann OLA -> output one hop. */
    fft_inverse(p->fft, p->spec, p->ifft_buf);
    for (int k = 0; k < frame_sz; k++) p->ola[k] += p->ifft_buf[k] * p->synth_win[k];
    memcpy(p->out_buf, p->ola, (size_t)hop * sizeof(float));
    memmove(p->ola, p->ola + hop, (size_t)(frame_sz - hop) * sizeof(float));
    memset(p->ola + (frame_sz - hop), 0, (size_t)hop * sizeof(float));

    memcpy(out, p->out_buf, (size_t)hop * sizeof(float));
    return 0;
}

void audio_pipeline_reset(AudioPipeline* p) {
    if (!p) return;

    aec_reset(p->aec);
    if (!p->aec_only && p->nr) {
        mmse_lsa_reset(p->nr);

        /* Re-zero pipeline-owned accumulator/scratch state (mirrors the
         * explicit zero at carve time). `synth_win` is a deterministic
         * constant (never re-derived from state) -- intentionally left
         * alone, same rationale as at init. */
        memset(p->ola,      0, (size_t)p->frame_sz * sizeof(float));
        memset(p->ifft_buf, 0, (size_t)p->fft_sz   * sizeof(float));
        memset(p->out_buf,  0, (size_t)p->hop      * sizeof(float));
        memset(p->g_nr,     0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->g_total,  0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->g_aec,    0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->extra,    0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->e2,       0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->spec,     0, (size_t)p->n_freqs  * sizeof(Complex));
    }
    memset(p->mic_buf, 0, (size_t)p->hop * sizeof(float));
    memset(p->ref_buf, 0, (size_t)p->hop * sizeof(float));
    memset(p->aec_out, 0, (size_t)p->hop * sizeof(float));

    p->rng_state = AUDIO_PIPELINE_RNG_SEED;
    p->near_hang = 0;
}

void audio_pipeline_destroy(AudioPipeline* p) {
    if (!p) return;
    /* Reverse carve order: NR -> pipeline FFT -> AEC. Each call is a genuine
     * no-op for a pool-resident (audio_pipeline_init'd) instance today; kept
     * for the heap path and forward-compat -- see audio_pipeline.h's doc. */
    if (p->nr)  mmse_lsa_destroy(p->nr);
    if (p->fft) fft_destroy(p->fft);
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
MmseLsaDenoiser* audio_pipeline_get_nr(const AudioPipeline* p) { return p ? p->nr : NULL; }

int audio_pipeline_get_mem_breakdown(const AudioPipelineConfig* cfg,
                                      AudioPipelineMemBreakdown* out) {
    if (!cfg || !out) return -1;

    AecConfig aec_cfg; MmseLsaConfig nr_cfg;
    int hop, frame_sz, fft_sz, n_freqs;
    if (derive_dims_and_configs(cfg, &aec_cfg, &nr_cfg, &hop, &frame_sz, &fft_sz, &n_freqs) != 0)
        return -1;

    size_t aec_sz = aec_get_mem_size(&aec_cfg);
    if (aec_sz == 0) return -1;
    size_t fft_mem = 0, nr_sz = 0;
    if (!cfg->aec_only) {
        fft_mem = fft_get_mem_size(fft_sz);
        nr_sz   = mmse_lsa_get_mem_size(&nr_cfg);
        if (fft_mem == 0 || nr_sz == 0) return -1;
    }

    size_t sub_total = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs,
                                           cfg->aec_only);
    size_t pipe_bufs = sub_total - ALIGN16(aec_sz) - ALIGN16(fft_mem) - ALIGN16(nr_sz);

    out->aec_bytes      = aec_sz;
    out->fft_bytes      = fft_mem;
    out->nr_bytes       = nr_sz;
    out->pipeline_bytes = pipe_bufs;
    out->hop = hop; out->frame_sz = frame_sz; out->fft_sz = fft_sz; out->n_freqs = n_freqs;
    return 0;
}
