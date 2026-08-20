/**
 * audio_pipeline.c — implementation of audio_pipeline.h.
 *
 * Ports pipelines/aec_nr_pipeline_static.c's file-local `pipeline_pool_size` /
 * `pipeline_build` / `pipeline_destroy` (the pool-sizing/carving) and the
 * per-hop while-loop BODY from both CLIs' `main()` (the processing) into one
 * linkable TU, verbatim in carve order/sizes and per-hop arithmetic — see
 * audio_pipeline.h for the API contract and pipelines/README.md ("Board
 * Integration") for the consumer sequence.
 *
 * What changed vs. the CLI-embedded originals (behaviourally):
 *   - synth_win/ola/... are now per-INSTANCE (an AudioPipeline
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
#include "nr_overlay.h"
#include "fft_wrapper.h"       /* fft_get_mem_size/fft_init/fft_inverse/fft_destroy, Complex, ALIGN16 */
#include "simd_kernels.h"      /* sk_min_f32 / sk_capply_gain_f32                                     */
#include "pipeline_dims.h"     /* compute_frame_dims() -- shared with both CLIs                       */

/* Deliberately NO "wav_io.h" here: this TU takes raw float* mic/ref/out
 * buffers only (audio_pipeline_process) -- no WAV path, no stdio-based I/O.
 * WAV reading/writing stays entirely in the CLIs (aec_nr_pipeline.c /
 * aec_nr_pipeline_static.c), which are now thin shells over this API. */

/* ============================================================================
 * No-stdio build gate
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
 * AudioPipelineMemReq doc for the bump rule). Bumped 1->2: g_aec (a per-hop
 * memcpy'd duplicate of AecResContext.res_gain) was removed from the carve --
 * both call sites that used to read p->g_aec now read ctx.res_gain directly
 * (that pointer is already stable for the whole hop per aec.h's own doc).
 * Bumped 2->3 (2026-08-05): g_nr is gone the same way -- mmse_lsa_get_gain()
 * already exposes the denoiser's own gain buffer with no copy, so
 * mmse_lsa_process_gain() is now called with gain_out=NULL and both former
 * p->g_nr readers call mmse_lsa_get_gain(p->nr, NULL) instead.
 * Bumped 3->4 (2026-08-05): mic_buf/ref_buf/out_buf are gone -- aec_process()
 * already copies mic/ref into its own buffers, so this pipeline's own copy
 * was a redundant second layer; the caller's mic/ref pointers now go
 * straight into aec_process(), and the OLA write lands directly in the
 * caller's `out` instead of an intermediate staging buffer.
 * Bumped 4->5 (2026-08-05): aec_out is gone -- its only reader was the
 * aec_only branch of audio_pipeline_process(), which immediately memcpy'd
 * it straight into the caller's own `out` buffer every hop. aec_process()'s
 * `out` parameter is not optional, and the caller's `out` is already
 * validated non-NULL and hop-sized at function entry, so that branch now
 * passes `out` to aec_process() directly -- one fewer buffer, one fewer
 * copy, same bytes in `out` either way.
 * Bumped 5->6: the self-resident AudioPipelineConfig gained independent
 * filter-length and delay-mode/bank/fixed-delay initialization fields.
 * Bumped 6->7: sizeof(Aec) grew (the suppressor gained its runtime far-active
 * floor retarget state), so the AEC carved out of this pool moves the total
 * and every offset after it. Carve order and buffer set are unchanged, so
 * build_flags_hash does not move -- this counter is the only signal.
 * Bumped 7->8: sizeof(Aec) grew again (each AEC instance gained its per-hop
 * stage-timing record -- aec_get_last_timing(), aec.h), so every AEC
 * carved out of this pool moves the total and the offsets after it. Carve
 * order and buffer set are unchanged, so build_flags_hash does not move --
 * this counter is the only signal. */
#define AUDIO_PIPELINE_LAYOUT_VERSION 8u

/* Compile-time FFT backend identity. pipelines/Makefile passes
 * -DAUDIO_PIPELINE_BACKEND_STR=\"kiss\" or \"ne10\" to match its own
 * BACKEND= selection (the same backend libaudio_common.a this TU links
 * against); the fallback below only fires if someone compiles this file
 * outside that Makefile. */
#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

/* Maps AUDIO_PIPELINE_BACKEND_STR -- THIS TU's OWN compile-time literal,
 * never caller-supplied data -- to the small stable integer a serializable
 * descriptor can carry (see audio_pipeline.h's
 * AudioPipelineMemReq.backend_id doc). One strcmp per
 * audio_pipeline_get_mem_requirements() call against a compiled-in literal
 * is not the caller-facing string hazard the header doc warns about (that
 * hazard was `strcmp` against a CALLER-supplied `expected->backend`, which
 * this file no longer does at all -- audio_pipeline_init_ex() below compares
 * backend_id with plain integer `==`). Returns 0 ("unknown") for anything
 * else; get_mem_requirements() rejects that -- see its own comment. */
static uint32_t audio_pipeline_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "kiss") == 0) return AUDIO_PIPELINE_BACKEND_KISS;
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0) return AUDIO_PIPELINE_BACKEND_NE10;
    return 0u;
}

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

    /* the pipeline's 7 scratch buffers (point into `pool` below; all seven
     * are NULL when aec_only). There is no g_aec
     * buffer: Stage 3a used to memcpy ctx.res_gain into one every hop purely
     * to have a stable pointer for sk_min_f32()/the CNG loop, but
     * AecResContext's own doc guarantees ctx.res_gain aliases AEC's internal
     * buffer for the whole hop (until the next AEC processing call) -- so
     * both call sites now read ctx.res_gain directly instead (layout v2).
     * There is likewise no g_nr buffer (layout v3): mmse_lsa_get_gain()
     * exposes the denoiser's own gain buffer directly, no copy needed.
     * There is likewise no mic_buf/ref_buf/out_buf (layout v4): aec_process()
     * already copies mic/ref into its own near_hop/far_hop at the top of the
     * call, so this pipeline's own decoupling copy was a redundant second
     * layer -- the caller's mic/ref pointers are passed straight through,
     * and the OLA write lands directly in the caller's `out`. There is
     * likewise no aec_out (layout v5): its only reader (the aec_only branch
     * of audio_pipeline_process()) now passes the caller's own `out` buffer
     * straight to aec_process() instead of an intermediate staging copy. */
    float*   synth_win;    /* iff !aec_only */
    float*   ola;          /* iff !aec_only */
    float*   ifft_buf;     /* iff !aec_only */
    float*   g_total;      /* iff !aec_only */
    float*   extra;        /* iff !aec_only */
    float*   e2;           /* iff !aec_only */
    Complex* spec;         /* iff !aec_only */

    /* per-instance comfort-noise RNG + near-end-floor hangover counter */
    uint32_t rng_state;
    int      near_hang;
    int      near_hangover_frames;  /* PROD_NEAR_HANGOVER retimed to this grid's hop */

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
 * enum values; invalid filter/delay settings; or
 * aec_only/enable_cng/legacy_amin holding anything but 0/1)
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

    if (compute_frame_dims(cfg->sample_rate, cfg->fft_size,
                           hop, frame_sz, fft_sz, n_freqs) != 0) return -1;

    aec_config_from_preset(aec_cfg, cfg->aec_preset, cfg->sample_rate);
    aec_cfg->fft_size          = *fft_sz;
    /* Bound kept in sync with mono_alignulcnet/audio_pipeline_ulcnet.c and
     * 4ch_aec_bf_nr_res/4aec_nr_res.c (each app is self-contained). */
    if (cfg->filter_length < 0 || cfg->filter_length > 4096) return -1;
    if (cfg->filter_length > 0)
        aec_cfg->filter_length = cfg->filter_length;
    aec_cfg->delay_mode = cfg->delay_mode;
    aec_cfg->delay_num_filters = cfg->delay_num_filters;
    aec_cfg->fixed_delay_samples = cfg->fixed_delay_samples;
    aec_cfg->enable_delay_est =
        cfg->delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
    aec_cfg->enable_res         = 0;   /* linear AEC + external NR/RES seam */
    aec_cfg->return_res_context = 1;

    *nr_cfg = pipelines_compose_nr_config(cfg->sample_rate, *fft_sz, *hop,
                                cfg->nr_mode);
    return 0;
}

/* ============================================================================
 * Pool sizing (verbatim port of aec_nr_pipeline_static.c's file-local
 * pipeline_pool_size -- same field order, same ALIGN16 bumps)
 * ========================================================================== */

static size_t pipeline_pool_size(const AecConfig* aec_cfg, const MmseLsaConfig* nr_cfg,
                                  int hop, int frame_sz, int fft_sz, int n_freqs,
                                  int aec_only) {
    (void)hop;   /* kept for signature symmetry with pipeline_build() below
                  * (same field order/call shape); no buffer here is sized
                  * from hop since aec_out (the last one that was) is gone. */
    size_t aec_sz     = aec_get_mem_size(aec_cfg);
    size_t fft_sz_mem = aec_only ? 0 : fft_get_mem_size(fft_sz);
    size_t nr_sz      = aec_only ? 0 : mmse_lsa_get_mem_size(nr_cfg);

    size_t pipe = 0;
    if (!aec_only) {
        pipe += ALIGN16((size_t)frame_sz * sizeof(float));   /* synth_win */
        pipe += ALIGN16((size_t)frame_sz * sizeof(float));   /* ola       */
        pipe += ALIGN16((size_t)fft_sz   * sizeof(float));   /* ifft_buf  */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_total   */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* extra     */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* e2        */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(Complex)); /* spec      */
    }

    return ALIGN16(aec_sz) + ALIGN16(fft_sz_mem) + ALIGN16(nr_sz) + pipe;
}

/* ============================================================================
 * Carve (verbatim port of aec_nr_pipeline_static.c's file-local
 * pipeline_build, PLUS an explicit zero of each of the 7 pipeline buffers --
 * see audio_pipeline.h's audio_pipeline_init doc for why. See
 * AUDIO_PIPELINE_LAYOUT_VERSION's doc for the buffer-set history.)
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

    if (!aec_only) {
        p->synth_win = (float*)ptr;   ptr += ALIGN16((size_t)frame_sz * sizeof(float));
        p->ola       = (float*)ptr;   ptr += ALIGN16((size_t)frame_sz * sizeof(float));
        p->ifft_buf  = (float*)ptr;   ptr += ALIGN16((size_t)fft_sz   * sizeof(float));
        p->g_total   = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->extra     = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->e2        = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->spec      = (Complex*)ptr; ptr += ALIGN16((size_t)n_freqs  * sizeof(Complex));

        /* Explicit zero of every pipeline-owned STATE buffer (F20: a dirty
         * pool must be safe without the caller's blanket memset). `synth_win`
         * is zeroed too even though the fill loop right below overwrites all
         * frame_sz elements unconditionally -- it holds a deterministic
         * constant, not accumulated state, so this memset is redundant
         * belt-and-braces, not load-bearing, but keeps the "all 7 buffers
         * explicitly zeroed at carve time" contract literal and unambiguous. */
        memset(p->synth_win, 0, (size_t)frame_sz * sizeof(float));
        memset(p->ola,       0, (size_t)frame_sz * sizeof(float));
        memset(p->ifft_buf,  0, (size_t)fft_sz   * sizeof(float));
        memset(p->g_total,   0, (size_t)n_freqs  * sizeof(float));
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
    h = fnv1a_str("|carve:self(config-delay-v2),aec,fft,nr,synth_win,ola,"
                  "ifft_buf,g_total,extra,e2,spec", h);
    h = fnv1a_str("|align16", h);
    return h;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

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

    /* backend_id: reject up front (same reject-first shape as
     * the module validators above) if this TU's own AUDIO_PIPELINE_BACKEND_STR
     * doesn't map to a known backend -- e.g. a build outside pipelines/
     * Makefile that never set -DAUDIO_PIPELINE_BACKEND_STR at all (falls
     * back to "unknown" above). A descriptor with backend_id==0 would be
     * meaningless to a board's `expected` comparison (0 means "no backend"),
     * so this library never actually returns one. */
    uint32_t backend_id = audio_pipeline_backend_id();
    if (backend_id == 0u) return -1;

    size_t sub_total = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs,
                                           cfg->aec_only);
    size_t self_sz   = ALIGN16(sizeof(AudioPipeline));
    /* self_sz + sub_total: both operands are bounded well under SIZE_MAX in
     * any realistic config (sub_total already went through mem_align.h's
     * saturating ck_* helpers inside each _get_mem_size call), so a plain
     * add is safe here. */
    out->descriptor_version = AUDIO_PIPELINE_DESCRIPTOR_VERSION;
    out->layout_version     = AUDIO_PIPELINE_LAYOUT_VERSION;
    out->backend_id         = backend_id;
    out->build_flags_hash   = audio_pipeline_build_flags_hash();
    out->alignment          = 16u;
    out->reserved           = 0u;
    out->bytes              = (uint64_t)(self_sz + sub_total);
    return 0;
}

/* ============================================================================
 * audio_pipeline_init_ex descriptor validation —
 * audio_pipeline_init() PLUS an optional `expected` descriptor gate. See
 * audio_pipeline.h for the full contract; the eight-condition check below
 * (run only when `expected` is non-NULL) is the literal implementation of
 * that doc's numbered list, in the SAME order, each on its own named
 * diagnostic. Every comparison below is a plain integer `==`/`<` over
 * fixed-width fields — no strings, no %s of caller data (see
 * AudioPipelineMemReq.backend_id's doc for why that matters: `expected` may
 * originate from persisted/transmitted bytes this library never validated).
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
            /* The header contract requires reserved to be zero in
             * any descriptor this library produced -- validated here, not
             * assumed, because `expected` may arrive from persisted/
             * transmitted bytes this library never wrote. */
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
    /* PROD_NEAR_HANGOVER (8) is a 10-ms-hop frame count (80 ms); was applied
     * as a raw literal regardless of grid (20-60% off at every one of this
     * pipeline's 3 real grids). Retimed the same way derive_dims_and_configs
     * already retimes nr_cfg->L/alpha_d/alpha_attack just above, and the
     * same way the Python reference (aec_nr_pipeline.py) already retimes
     * this exact constant via retime_frame_count(). */
    p->near_hangover_frames = mmse_lsa_retime_frames(
        PROD_NEAR_HANGOVER, cfg->sample_rate, hop);
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

    /* Stage 1: AEC. aec_process() already copies mic/ref into its own
     * near_hop/far_hop at the top of the call, so passing the caller's
     * pointers straight through is exactly as safe as an extra pipeline-
     * owned decoupling copy would have been, without paying for one.
     *
     * Dispatch on p->aec_only, fixed at construction. The two entry points
     * differ only by whether the result is copied out, so this is purely a
     * cost choice. aec_only means the AEC's own output IS the pipeline's
     * output -- needs aec_process(), writing directly into the caller's
     * `out` (already validated non-NULL and hop-sized above). !aec_only
     * means downstream NR/RES reads only ctx.error_spec and never the
     * emitted hop, so aec_process_context() skips that copy. */
    if (p->aec_only) {
        aec_process(p->aec, mic, ref, out);
        return 0;
    }
    aec_process_context(p->aec, mic, ref);

    AecResContext ctx;
    aec_get_res_context(p->aec, &ctx);
    /* enable_res=0/return_res_context=1 is set unconditionally for every
     * !aec_only instance (derive_dims_and_configs() above), so
     * ctx.error_spec/ctx.res_gain are always non-NULL here -- there is no
     * live "seam unavailable" case to fall back from. */

    /* Stage 2: echo-aware NR gain. extra = R^2/PSD_SCALE folds the residual
     * echo into the noise floor (xi = S^2/(N^2+R^2)); off in legacy.
     * p->extra[] is populated whenever ctx.r2 is available (independent of
     * legacy_amin) because the near-end-lift loop below (Stage 3b) also
     * needs this exact value -- computing it once here and reading it there
     * avoids a redundant ctx.r2[k]/PSD_SCALE division per bin per hop.
     * Only the POINTER handed to the NR gain call is gated on legacy_amin. */
    const float* nr_extra = NULL;
    if (ctx.r2) {
        for (int k = 0; k < n_freqs; k++)
            p->extra[k] = ctx.r2[k] / PSD_SCALE;
        nr_extra = p->legacy_amin ? NULL : p->extra;
    }
    /* gain_out=NULL: mmse_lsa_get_gain() below reads the same buffer this
     * call just filled, no per-hop copy needed (layout v3). */
    mmse_lsa_process_gain(p->nr, ctx.error_spec, nr_extra, NULL);

    /* Stage 3a: g_total = min(G_nr, G_res). ctx.res_gain (= G_res, pre-min)
     * also sets the comfort-noise level below so CNG reflects AEC
     * suppression only. Read directly from the AEC's own seam buffer rather
     * than a local copy -- aec.h's AecResContext doc guarantees these seam
     * pointers alias AEC's internal per-hop buffers and stay valid until the
     * next aec_process() call, which doesn't happen again before this hop
     * finishes (verified below: ctx.res_gain is read again, unchanged, at
     * the near-end-lift loop and the CNG loop further down). Likewise
     * mmse_lsa_get_gain()'s buffer is stable until the next
     * mmse_lsa_process_gain()/mmse_lsa_process() call on this instance,
     * which also doesn't happen again before this hop finishes. */
    sk_min_f32(p->g_total, mmse_lsa_get_gain(p->nr, NULL), ctx.res_gain,
               n_freqs);

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
        if (ne > PROD_NEAR_GATE_THRESH) p->near_hang = p->near_hangover_frames;
        int near_active = p->near_hang > 0;
        if (p->near_hang > 0) p->near_hang--;
        int protect = (!far_active) && near_active;
        nf_eff = protect ? PROD_NE_FLOOR : PROD_NE_FLOOR_FAR_ACTIVE;
    }

    /* Per-bin echo-gated near-end lift (ne_gate='both': G_res*(1-echo_frac)). */
    if (nf_eff > 0.0f && ctx.r2) {
        for (int k = 0; k < n_freqs; k++) {
            float r2_nr = p->extra[k];   /* == ctx.r2[k] / PSD_SCALE, already computed above */
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
     * sqrt(1 - G_res^2) so it fills only what the AEC suppressed (bins
     * 1..N-2). */
    if (p->enable_cng_effective && ctx.comfort_noise) {
        for (int k = 1; k < n_freqs - 1; k++) {
            float n_amp = ctx.comfort_noise[k] / PSD_SCALE;
            n_amp = (n_amp > 0.0f) ? sqrtf(n_amp) : 0.0f;
            float ng2 = 1.0f - ctx.res_gain[k] * ctx.res_gain[k];
            float noise_gain = (ng2 > 0.0f) ? sqrtf(ng2) : 0.0f;
            float a = noise_gain * n_amp;
            p->spec[k].r += a * rng_gauss(p);
            p->spec[k].i += a * rng_gauss(p);
        }
    }

    /* ctx.error_spec already contains the matching sqrt-Hann analysis frame;
     * complete the 50%-overlap WOLA with one IFFT + synthesis + OLA. */
    fft_inverse(p->fft, p->spec, p->ifft_buf);
    sk_wola_accumulate_f32(p->ola, p->ifft_buf, p->synth_win, frame_sz);
    memcpy(out, p->ola, (size_t)hop * sizeof(float));
    memmove(p->ola, p->ola + hop, (size_t)(frame_sz - hop) * sizeof(float));
    memset(p->ola + (frame_sz - hop), 0, (size_t)hop * sizeof(float));

    return 0;
}

int audio_pipeline_set_aec_preset(AudioPipeline* p, AecPreset preset,
                                  float ramp_ms) {
    if (!p || !p->aec) return -1;
    /* No pipeline-level mirror: this instance stores resolved dimensions, not
     * the caller's config, and aec_set_preset already updates the AEC's own
     * AecConfig -- the single authoritative copy. */
    return aec_set_preset(p->aec, preset, ramp_ms);
}

int audio_pipeline_set_nr_mode(AudioPipeline* p, MmseLsaNrMode mode) {
    MmseLsaConfig target;
    if (!p || !p->nr) return -1;   /* aec_only builds have no denoiser */
    if (!mmse_lsa_nr_mode_is_valid(mode)) return -1;
    /* Recompose THIS pipeline's configuration, not the canonical preset:
     * mmse_lsa_set_mode() would refuse it (its L differs) or revert the
     * overrides pipelines_compose_nr_config() applies. */
    target = pipelines_compose_nr_config(p->sample_rate, p->fft_sz, p->hop, mode);
    /* mmse_lsa_reconfigure updates the denoiser's own MmseLsaConfig, which is
     * the authoritative copy; this instance keeps no config of its own. */
    return mmse_lsa_reconfigure(p->nr, &target);
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
        memset(p->g_total,  0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->extra,    0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->e2,       0, (size_t)p->n_freqs  * sizeof(float));
        memset(p->spec,     0, (size_t)p->n_freqs  * sizeof(Complex));
    }

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
