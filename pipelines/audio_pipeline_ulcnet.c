/**
 * audio_pipeline_ulcnet.c -- implementation of audio_pipeline_ulcnet.h.
 *
 * Mono AEC(linear) -> Align-ULCNet neural post-filter pipeline. Follows
 * audio_pipeline.c's construction conventions verbatim (caller-owned pool,
 * zero heap on the init/init_ex path, 32-byte descriptor, reject-first
 * validation, explicit zeroing, reverse-order destroy) but drives the
 * AIAEC/Align_ULCNet C pre/post (ulcnet_process.c) instead of NR/RES.
 *
 * Per hop:
 *   1. aec_process_context(aec, mic, ref)  -- context-only linear AEC; the
 *      AEC aligns far internally (delay estimation stays on, preset default).
 *   2. error tap  = AecResContext.formed_hop (the refined/coarse-selected +
 *      crossfaded linear error -- the hop the AEC's own spectra describe).
 *   3. far tap + delay status = aec_get_linear_context(): aligned_far_hop
 *      is byte-identical to what the linear filter consumed this hop;
 *      delay_state gates the model per the policy in the header.
 *   4. Both hops go into two UlcnetAnalysis instances (0/2/1 emission); for
 *      each emitted frame pair the model callback runs (stepped even while
 *      UNLOCKED -- constant compute -- but its output is applied only when
 *      the delay is locked and infer() returned 0; otherwise the error
 *      frame passes through unchanged, fail-open); the chosen frame goes
 *      into UlcnetSynthesis (WOLA). hop #0 emits nothing -> zeros; hop #p
 *      output corresponds to input hop p-1 (one-hop latency).
 *   5. On a CHANGED delay event, model->reset (if set) runs BEFORE that
 *      hop's infer() so the runtime flushes its far attention ring + logit
 *      history; the C STFT states keep running (1-2 frame transient
 *      accepted; crossfade is a later phase).
 *
 * Constraints: C99, -ffp-contract=off (pipelines/Makefile appends it last),
 * no stdio in this TU (all failures are signalled by NULL/-1 returns only),
 * no heap on the init/init_ex path (create() is the explicit heap
 * convenience). The Ulcnet analysis/synthesis structs and the per-frame
 * spectrum scratch are plain fixed-size arrays kept INSIDE the instance
 * struct (part of the `self` carve) rather than stack locals -- ~10 KB of
 * frame scratch would be unsafe headroom for an embedded RTOS task stack
 * (same rationale as lib/aec's Tier-1 stack-safety fix).
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "audio_pipeline_ulcnet.h"
#include "mem_align.h"       /* ALIGN16 / MEM_IS_ALIGNED16 */
#include "pipeline_dims.h"   /* compute_frame_dims() -- shared grid resolver */

/* This wrapper's own carve-layout version (see AudioPipelineUlcnetMemReq's
 * doc). History starts at 1 (2026-08: first version). Bump TOGETHER with the
 * token string in audio_pipeline_ulcnet_build_flags_hash() below, forever
 * after, whenever the carve structure changes -- including changes to the
 * self-resident Ulcnet state block, which is part of the carve even though
 * it is not a separately-carved pointer. */
#define AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION 1u

/* Compile-time FFT backend identity -- same mechanism as audio_pipeline.c:
 * pipelines/Makefile passes -DAUDIO_PIPELINE_BACKEND_STR=\"kiss\"/\"ne10\"
 * to every TU it compiles; the fallback only fires outside that Makefile
 * and is rejected by get_mem_requirements (backend_id 0 is reserved). */
#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

static uint32_t audio_pipeline_ulcnet_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "kiss") == 0)
        return AUDIO_PIPELINE_ULCNET_BACKEND_KISS;
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0)
        return AUDIO_PIPELINE_ULCNET_BACKEND_NE10;
    return 0u;
}

/* The compiled ULCNet grid must agree with pipeline_dims.h's 16k/512 row:
 * hop = fft/2, bins = fft/2+1. Checked at compile time here and re-checked
 * at runtime against the live AEC instance in ulcnet_pipeline_build(). */
_Static_assert(ULCNET_N_FFT == 2 * ULCNET_HOP,
               "ULCNet grid must be 50%-overlap (fft == 2*hop)");
_Static_assert(ULCNET_BINS == ULCNET_N_FFT / 2 + 1,
               "ULCNet bins must be fft/2+1");
_Static_assert(ULCNET_SR == 16000, "ULCNet deployment grid is 16 kHz");

/* ============================================================================
 * Instance
 * ========================================================================== */

struct AudioPipelineUlcnet {
    int sample_rate, hop, fft_sz, n_freqs;

    UlcnetModel model;            /* by-value copy of cfg.model              */

    Aec* aec;                     /* points into `pool` below                */

    /* Ulcnet C pre/post state -- plain fixed-size structs, part of the
     * `self` carve (listed as self(...) tokens in the build-flags hash). */
    UlcnetAnalysis  ana_err;      /* linear-error analysis                   */
    UlcnetAnalysis  ana_far;      /* aligned-far analysis                    */
    UlcnetSynthesis synth;        /* enhanced-spectrum WOLA                  */

    /* Per-hop frame scratch (up to 2 frames per push on hop #1). Kept in
     * the instance, not the stack -- see the header comment of this file. */
    float err_re[2][ULCNET_BINS], err_im[2][ULCNET_BINS];
    float far_re[2][ULCNET_BINS], far_im[2][ULCNET_BINS];
    float mdl_re[ULCNET_BINS],    mdl_im[ULCNET_BINS];

    /* pool bookkeeping */
    void*  pool;                  /* sub-pool AFTER this struct: AEC only    */
    size_t pool_size;
    void*  owned_heap;            /* non-NULL iff from create(); freed by destroy() */
};

/* ============================================================================
 * Config -> AEC config + dims (the one reject-first gate every entry point
 * funnels through, mirroring audio_pipeline.c's derive_dims_and_configs)
 * ========================================================================== */

/* Returns 0 and fills every out-param, or -1: NULL cfg; sample_rate != 16000
 * (the ULCNet checkpoint's feature-time contract is compiled at 16 kHz --
 * 8 k/48 k are rejected here, not left to a downstream 0); fft_size neither
 * 0 nor 512 (0 resolves to 512, the ULCNet deployment grid; pipeline_dims'
 * 16 kHz rate-default of 256 does NOT apply to this variant -- a 256 grid
 * would not match the compiled ULCNET_* constants); or aec_preset outside
 * its defined enum values (rather than silently falling through
 * aec_config_from_preset's own balanced-default fallback). The model field
 * is deliberately NOT validated: an all-zero model is the supported
 * identity/fail-open case. */
static int ulcnet_derive_dims_and_config(const AudioPipelineUlcnetConfig* cfg,
                                         AecConfig* aec_cfg,
                                         int* hop, int* fft_sz, int* n_freqs) {
    int frame_sz;

    if (!cfg) return -1;
    if (cfg->sample_rate != ULCNET_SR) return -1;
    if (cfg->fft_size != 0 && cfg->fft_size != ULCNET_N_FFT) return -1;

    switch (cfg->aec_preset) {
        case AEC_PRESET_MILD:
        case AEC_PRESET_BALANCED:
        case AEC_PRESET_AGGRESSIVE:
            break;
        default:
            return -1;
    }

    if (compute_frame_dims(ULCNET_SR, ULCNET_N_FFT,
                           hop, &frame_sz, fft_sz, n_freqs) != 0) return -1;
    /* Defensive: the resolver's 16k/512 row must agree with the compiled
     * ULCNet grid (also pinned by the _Static_asserts above). */
    if (*hop != ULCNET_HOP || *fft_sz != ULCNET_N_FFT || *n_freqs != ULCNET_BINS)
        return -1;

    aec_config_from_preset(aec_cfg, cfg->aec_preset, cfg->sample_rate);
    aec_cfg->fft_size           = *fft_sz;
    aec_cfg->enable_res         = 0;   /* linear AEC + external NN post seam */
    aec_cfg->return_res_context = 1;   /* fills formed_hop/error_spec/...    */
    /* spatial_linear_context stays 0: this is the single-lane seam, and the
     * res context (formed_hop) must be populated. Delay estimation stays on
     * (preset default) -- the AEC aligns far internally. */
    return 0;
}

/* ============================================================================
 * Pool sizing / carve. Carve order: self control block (which embeds the
 * Ulcnet analysis/synthesis/frame-scratch state), then the AEC pool.
 * ========================================================================== */

static size_t ulcnet_sub_pool_size(const AecConfig* aec_cfg) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    if (aec_sz == 0) return 0;   /* invalid AecConfig per lib/aec's validator */
    return ALIGN16(aec_sz);
}

static int ulcnet_pipeline_build(AudioPipelineUlcnet* p, void* pool, size_t pool_size,
                                 const AecConfig* aec_cfg) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    AecResContext ctx0;

    if (!pool || aec_sz == 0 || pool_size < ALIGN16(aec_sz)) return -1;

    p->aec = aec_init(pool, aec_sz, aec_cfg);
    if (!p->aec) return -1;

    /* Grid-agreement guard, readable right after aec_init (before any frame
     * is processed) -- same shape as audio_pipeline.c's pipeline_build. */
    aec_get_res_context(p->aec, &ctx0);
    if (ctx0.n_freqs != ULCNET_BINS || ctx0.hop_size != ULCNET_HOP) return -1;

    ulcnet_analysis_init(&p->ana_err);
    ulcnet_analysis_init(&p->ana_far);
    ulcnet_synthesis_init(&p->synth);
    return 0;
}

/* ============================================================================
 * Build-flags hash (FNV-1a-32)
 * ========================================================================== */

static uint32_t ulcnet_fnv1a_str(const char* s, uint32_t h) {
    while (*s) { h ^= (uint32_t)(unsigned char)(*s++); h *= 16777619u; }
    return h;
}

static uint32_t audio_pipeline_ulcnet_build_flags_hash(void) {
    uint32_t h = 2166136261u;   /* FNV-1a 32-bit offset basis */
    h = ulcnet_fnv1a_str(AUDIO_PIPELINE_BACKEND_STR, h);
    /* Literal carve-order token list for THIS wrapper. The self(...) tokens
     * name the Ulcnet state embedded in the control block -- it is part of
     * the carve even though it is not separately carved. Bump
     * AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION (and update this string)
     * whenever this structure changes -- always both, forever. */
    h = ulcnet_fnv1a_str("|carve:self(model,ana_err,ana_far,synth,"
                         "frame_scratch),aec", h);
    h = ulcnet_fnv1a_str("|align16", h);
    return h;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

AudioPipelineUlcnetConfig audio_pipeline_ulcnet_default_config(int sample_rate) {
    AudioPipelineUlcnetConfig cfg;
    memset(&cfg, 0, sizeof(cfg));       /* all-zero model == identity */
    cfg.sample_rate = sample_rate;
    cfg.fft_size    = 0;                /* resolve to the ULCNet grid (512) */
    cfg.aec_preset  = AEC_PRESET_BALANCED;
    return cfg;
}

int audio_pipeline_ulcnet_get_mem_requirements(const AudioPipelineUlcnetConfig* cfg,
                                               AudioPipelineUlcnetMemReq* out) {
    AecConfig aec_cfg;
    int hop, fft_sz, n_freqs;
    uint32_t backend_id;
    size_t sub_total, self_sz;

    if (!cfg || !out) return -1;
    if (ulcnet_derive_dims_and_config(cfg, &aec_cfg, &hop, &fft_sz, &n_freqs) != 0)
        return -1;

    sub_total = ulcnet_sub_pool_size(&aec_cfg);
    if (sub_total == 0) return -1;      /* lib/aec's own validator rejected it */

    /* Reject-first: a build outside pipelines/Makefile (backend "unknown")
     * would produce a descriptor meaningless to a board's `expected`
     * comparison, so it is never returned. */
    backend_id = audio_pipeline_ulcnet_backend_id();
    if (backend_id == 0u) return -1;

    self_sz = ALIGN16(sizeof(AudioPipelineUlcnet));
    out->descriptor_version = AUDIO_PIPELINE_ULCNET_DESCRIPTOR_VERSION;
    out->layout_version     = AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION;
    out->backend_id         = backend_id;
    out->build_flags_hash   = audio_pipeline_ulcnet_build_flags_hash();
    out->alignment          = 16u;
    out->reserved           = 0u;
    out->bytes              = (uint64_t)(self_sz + sub_total);
    return 0;
}

/* init_ex: the 8-point descriptor gate, in audio_pipeline_init_ex's exact
 * order -- (1) descriptor_version, (2) layout_version, (3) backend_id,
 * (4) build_flags_hash, (5) alignment, (6) reserved == 0, (7)
 * expected->bytes >= current requirement, (8) pool bytes >= current
 * requirement. Every comparison is a plain integer ==/< over fixed-width
 * fields; `expected` may originate from persisted/transmitted bytes this
 * library never wrote, so nothing in it is trusted or dereferenced beyond
 * the struct itself. */
AudioPipelineUlcnet* audio_pipeline_ulcnet_init_ex(void* mem, size_t bytes,
                                                   const AudioPipelineUlcnetConfig* cfg,
                                                   const AudioPipelineUlcnetMemReq* expected) {
    AecConfig aec_cfg;
    int hop, fft_sz, n_freqs;
    size_t self_sz, sub_bytes, sub_needed;
    AudioPipelineUlcnet* p;
    void* sub_pool;

    if (!mem || !cfg) return NULL;

    if (expected) {
        AudioPipelineUlcnetMemReq cur;
        if (audio_pipeline_ulcnet_get_mem_requirements(cfg, &cur) != 0) return NULL;
        if (expected->descriptor_version != cur.descriptor_version) return NULL;
        if (expected->layout_version     != cur.layout_version)     return NULL;
        if (expected->backend_id         != cur.backend_id)         return NULL;
        if (expected->build_flags_hash   != cur.build_flags_hash)   return NULL;
        if (expected->alignment          != cur.alignment)          return NULL;
        if (expected->reserved           != 0u)                     return NULL;
        if (expected->bytes < cur.bytes)                            return NULL;
        if ((uint64_t)bytes < cur.bytes)                            return NULL;
    }

    if (!MEM_IS_ALIGNED16(mem)) return NULL;

    if (ulcnet_derive_dims_and_config(cfg, &aec_cfg, &hop, &fft_sz, &n_freqs) != 0)
        return NULL;

    self_sz = ALIGN16(sizeof(AudioPipelineUlcnet));
    if (bytes < self_sz) return NULL;
    sub_bytes  = bytes - self_sz;
    sub_needed = ulcnet_sub_pool_size(&aec_cfg);
    if (sub_needed == 0 || sub_bytes < sub_needed) return NULL;

    p = (AudioPipelineUlcnet*)mem;
    /* Explicit zero of the WHOLE control block -- covers every embedded
     * Ulcnet state struct and the frame scratch, so a dirty/poisoned pool
     * inits identically to a zeroed one (the *_init calls in build then
     * rewrite the deterministic window tables). */
    memset(p, 0, sizeof(*p));
    sub_pool = (uint8_t*)mem + self_sz;

    if (ulcnet_pipeline_build(p, sub_pool, sub_bytes, &aec_cfg) != 0) return NULL;

    p->sample_rate = cfg->sample_rate;
    p->hop = hop; p->fft_sz = fft_sz; p->n_freqs = n_freqs;
    p->model      = cfg->model;
    p->pool       = sub_pool;
    p->pool_size  = sub_bytes;
    p->owned_heap = NULL;
    return p;
}

AudioPipelineUlcnet* audio_pipeline_ulcnet_init(void* mem, size_t bytes,
                                                const AudioPipelineUlcnetConfig* cfg) {
    return audio_pipeline_ulcnet_init_ex(mem, bytes, cfg, NULL);
}

AudioPipelineUlcnet* audio_pipeline_ulcnet_create(const AudioPipelineUlcnetConfig* cfg) {
    AudioPipelineUlcnetMemReq req;
    void* mem = NULL;
    AudioPipelineUlcnet* p;

    if (!cfg) return NULL;
    if (audio_pipeline_ulcnet_get_mem_requirements(cfg, &req) != 0) return NULL;
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0 || !mem)
        return NULL;

    p = audio_pipeline_ulcnet_init(mem, (size_t)req.bytes, cfg);
    if (!p) { free(mem); return NULL; }
    p->owned_heap = mem;
    return p;
}

int audio_pipeline_ulcnet_process(AudioPipelineUlcnet* p, const float* mic,
                                  const float* ref, float* out) {
    AecResContext rctx;
    AecLinearContext lctx;
    int n_frames, f, wrote;

    if (!p || !mic || !ref || !out) return -1;

    /* Stage 1: context-only linear AEC (no time-output copy; downstream
     * reads only the seam taps). */
    aec_process_context(p->aec, mic, ref);

    /* Stage 2: taps. Both alias AEC-internal per-hop buffers, valid until
     * the next AEC processing call -- they are consumed within this hop, so
     * no copy is needed. formed_hop is non-NULL for every hop of this
     * config (enable_res=0/return_res_context=1 is set unconditionally). */
    aec_get_res_context(p->aec, &rctx);
    aec_get_linear_context(p->aec, &lctx);

    /* Stage 3: delay-change event -> flush the runtime's far attention ring
     * + logit history BEFORE this hop's inference. CHANGED is reported
     * exactly on the hop whose processing bumped the alignment generation
     * (first acquisition included); the next hop reads LOCKED again. The C
     * STFT states below keep running across the change (1-2 frame transient
     * accepted; crossfade is a later phase). */
    if (lctx.delay_state == AEC_LINEAR_DELAY_CHANGED && p->model.reset)
        p->model.reset(p->model.user);

    /* Stage 4: push BOTH hops (the two analyses must stay frame-locked, so
     * each is fed every hop unconditionally). 0/2/1 emission: hop #0 emits
     * nothing, hop #1 emits two frames, then one per hop. */
    n_frames = ulcnet_analysis_push(&p->ana_err, rctx.formed_hop, p->err_re, p->err_im);
    (void)ulcnet_analysis_push(&p->ana_far, lctx.aligned_far_hop, p->far_re, p->far_im);

    /* Stage 5: per emitted frame pair, run the model and synthesize. The
     * model is STEPPED whenever an infer callback exists (constant per-hop
     * compute/timing); its output is APPLIED only when infer() returned 0
     * AND the delay is locked (LOCKED/CHANGED). UNLOCKED means
     * aligned_far_hop is raw/unaligned far -- the result computed from it
     * is not trusted, and the error frame passes through unchanged
     * (fail-open identity). Any garbage the runtime accumulated from raw
     * far during UNLOCKED is flushed by the CHANGED reset above at the
     * first acquisition. */
    wrote = 0;
    for (f = 0; f < n_frames; f++) {
        const float* sre = p->err_re[f];
        const float* sim = p->err_im[f];
        if (p->model.infer) {
            int rc = p->model.infer(p->model.user,
                                    p->err_re[f], p->err_im[f],
                                    p->far_re[f], p->far_im[f],
                                    p->mdl_re, p->mdl_im);
            if (rc == 0 && lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED) {
                sre = p->mdl_re;
                sim = p->mdl_im;
            }
        }
        wrote += ulcnet_synthesis_push(&p->synth, sre, sim, out + wrote);
    }

    /* hop #0 (n_frames == 0, nothing emitted): the one-hop-latency contract
     * says the caller's output for this hop is silence. Every later hop
     * emits exactly one hop of samples (frame #0's share of hop #1 lands in
     * the trimmed half window), so this memset never runs again. */
    if (wrote < p->hop)
        memset(out + wrote, 0, (size_t)(p->hop - wrote) * sizeof(float));
    return 0;
}

void audio_pipeline_ulcnet_reset(AudioPipelineUlcnet* p) {
    if (!p) return;
    aec_reset(p->aec);
    ulcnet_analysis_init(&p->ana_err);
    ulcnet_analysis_init(&p->ana_far);
    ulcnet_synthesis_init(&p->synth);
    /* The external runtime owns the NN's explicit states -- tell it to
     * flush them too, same contract as the CHANGED event. */
    if (p->model.reset) p->model.reset(p->model.user);
}

void audio_pipeline_ulcnet_destroy(AudioPipelineUlcnet* p) {
    if (!p) return;
    /* Reverse carve order (only the AEC sits in the sub-pool). A genuine
     * no-op for a pool-resident instance today; kept for the heap path and
     * forward-compat, mirroring audio_pipeline_destroy's rationale. */
    if (p->aec) aec_destroy(p->aec);

    if (p->owned_heap) {
        void* heap = p->owned_heap;
        free(heap);   /* frees `p` itself too (create() carves p at mem[0]) */
    }
}

int audio_pipeline_ulcnet_hop_size(const AudioPipelineUlcnet* p) {
    return p ? p->hop : -1;
}

Aec* audio_pipeline_ulcnet_get_aec(const AudioPipelineUlcnet* p) {
    return p ? p->aec : NULL;
}
