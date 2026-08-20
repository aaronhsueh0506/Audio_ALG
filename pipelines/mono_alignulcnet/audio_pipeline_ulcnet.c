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
 *   3. far tap + delay status = aec_get_linear_context(): the model always
 *      receives aligned_far_hop, byte-identical to the far hop consumed by
 *      the linear filter. Before acquisition this seam deliberately carries
 *      raw far, so the model's D window can handle the remaining offset.
 *   4. Both hops go into two UlcnetAnalysis instances (0/2/1 emission); for
 *      each emitted frame pair the model callback runs; its output is
 *      applied only when infer() returned 0 and the output frame is fully
 *      finite; otherwise the error passes unchanged (fail-open); the chosen
 *      frame goes into UlcnetSynthesis
 *      (WOLA). hop #0 emits nothing -> zeros; hop #p output corresponds to
 *      input hop p-1 (one-hop latency).
 *   5. On a CHANGED event, or FIXED's first UNLOCKED->LOCKED transition,
 *      model->reset (if set) runs BEFORE that hop's frames so the runtime
 *      flushes its far attention ring + logit history, and the identity
 *      reprime is armed: the C STFT states keep running, so the next
 *      AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES emitted frames still straddle
 *      the boundary and are emitted as identity WITHOUT stepping the model
 *      (header: "Identity reprime"; the constant is measured by the
 *      straddle-derivation test, and option B is deferred).
 *
 * Constraints: C99, -ffp-contract=off (pipelines/Makefile appends it last),
 * no stdio in this TU (all failures are signalled by NULL/-1 returns only),
 * no heap on the init/init_ex path (create() is the explicit heap
 * convenience). The Ulcnet analysis/synthesis structs (which embed their
 * own per-call FFT scratch), the per-frame spectrum scratch, and the one
 * shared sqrt-Hann window table are plain fixed-size arrays kept INSIDE
 * the instance struct (part of the `self` carve) rather than stack locals
 * -- multi-KB frame scratch would be unsafe headroom for an embedded RTOS
 * task stack (same rationale as lib/aec's Tier-1 stack-safety fix). The
 * chain's FFT is ONE pool-carved 512-point fft_wrapper handle shared by
 * err-analysis + far-analysis + synthesis (strictly sequential use within
 * a hop, per ulcnet_process.h's sharing contract), so BACKEND=kiss/ne10
 * genuinely selects the ULCNet FFT backend too.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>       /* isfinite + NAN -- model-output guard/pre-fill */

#include "audio_pipeline_ulcnet.h"
#include "mem_align.h"       /* ALIGN16 / MEM_IS_ALIGNED16 */

/* This wrapper's own carve-layout version (see AudioPipelineUlcnetMemReq's
 * doc). History starts at 1 (2026-08: first version); 2 adds the
 * far_input_mode field to the self control block; 3 moves the ULCNet chain
 * onto a pool-carved shared FftHandle (the carve grew a trailing `fft`
 * region), embeds the per-call FFT scratch in the chain structs, and
 * replaces their per-struct window copies with one self-resident shared
 * table (ulcnet_window). Bump TOGETHER with the token string in
 * audio_pipeline_ulcnet_build_flags_hash() below, forever after, whenever
 * the carve structure changes -- including changes to the self-resident
 * Ulcnet state block, which is part of the carve even though it is not a
 * separately-carved pointer. Version 4 grows the self-resident config with
 * independent filter-length and AEC delay controls. Version 5 grows the
 * self-resident UlcnetModel copy by the published model-I/O descriptor
 * pointer. Version 6 removes the obsolete runtime far-mode field, fixes
 * production to the AEC aligned-far seam, and adds delay-transition
 * bookkeeping. Version 7 adds the identity-reprime counter to the control
 * block. Version 8: sizeof(Aec) grew (the suppressor gained its runtime
 * far-active floor retarget state), so every AEC carved out of this pool
 * moves the total and the offsets after it. Carve order and buffer set are
 * unchanged, so build_flags_hash does not move -- this counter is the only
 * signal. */
#define AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION 8u

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

/* The compiled ULCNet deployment grid is fixed at 16 kHz/512/256:
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

    /* MATCHED acquisitions report CHANGED. FIXED mode instead moves from
     * raw far during ring fill directly to LOCKED, so track that first
     * usable transition and flush external recurrent state at the seam. */
    AecLinearDelayState last_delay_state;
    uint64_t processed_hops;

    /* Emitted frames still straddling the last alignment boundary: armed to
     * AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES at the boundary hop, decremented
     * once per EMITTED frame (not per hop -- hop #1 emits two). While
     * nonzero the frame takes the identity path and the model is not
     * stepped. Re-armed, never accumulated, by a boundary during a reprime. */
    int reprime_frames;

    Aec* aec;                     /* points into `pool` below                */
    FftHandle* fft;               /* points into `pool`; ONE shared 512-point
                                   * handle for the whole ULCNet chain
                                   * (strictly sequential use per hop)      */

    /* Ulcnet C pre/post state -- plain fixed-size structs, part of the
     * `self` carve (listed as self(...) tokens in the build-flags hash).
     * Each embeds its own per-call FFT scratch and points at the shared
     * window table below (no per-struct window copies). */
    UlcnetAnalysis  ana_err;      /* linear-error analysis                   */
    UlcnetAnalysis  ana_far;      /* aligned-far analysis                    */
    UlcnetSynthesis synth;        /* enhanced-spectrum WOLA                  */
    float ulcnet_window[ULCNET_N_FFT];  /* shared sqrt-Hann table; all three
                                   * chain structs point at it (self-owned) */

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
 * 0 nor 512 (0 resolves to 512, the ULCNet deployment grid; the conventional
 * 16 kHz rate-default of 256 does NOT apply to this variant -- a 256 grid
 * would not match the compiled ULCNET_* constants); or aec_preset outside
 * its defined enum values (rather than silently falling through
 * aec_config_from_preset's own balanced-default fallback). The model's
 * callbacks are deliberately NOT validated -- an all-zero model is the
 * supported identity/fail-open case. A published model-I/O descriptor must
 * match the fixed aligned-far production ABI. */
static int ulcnet_derive_dims_and_config(const AudioPipelineUlcnetConfig* cfg,
                                         AecConfig* aec_cfg,
                                         int* hop, int* fft_sz, int* n_freqs) {
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

    /* A model that actually infers MUST publish a descriptor. Its delay depth,
     * attention geometry and history shapes are what the host-side rings are
     * carved from, and nothing downstream can detect a mismatch: the finite
     * guard catches an UNWRITTEN output, never a WRONG-SHAPED one, so a graph
     * whose D differs from the descriptor reads and writes past the pool
     * silently. An identity model (no infer callback) has no shapes to agree
     * about and may leave it NULL. */
    if (cfg->model.infer && !cfg->model.io_descriptor) return -1;
    if (cfg->model.io_descriptor &&
        ulcnet_model_io_descriptor_validate(cfg->model.io_descriptor) != 0)
        return -1;

    *hop = ULCNET_HOP;
    *fft_sz = ULCNET_N_FFT;
    *n_freqs = ULCNET_BINS;

    aec_config_from_preset(aec_cfg, cfg->aec_preset, cfg->sample_rate);
    aec_cfg->fft_size           = *fft_sz;
    /* Bound kept in sync with mono_aec_nr_res/audio_pipeline.c and
     * 4ch_aec_bf_nr_res/4aec_nr_res.c (each app is self-contained). */
    if (cfg->filter_length < 0 || cfg->filter_length > 4096) return -1;
    if (cfg->filter_length > 0)
        aec_cfg->filter_length = cfg->filter_length;
    aec_cfg->delay_mode = cfg->delay_mode;
    aec_cfg->delay_num_filters = cfg->delay_num_filters;
    aec_cfg->fixed_delay_samples = cfg->fixed_delay_samples;
    aec_cfg->enable_delay_est =
        cfg->delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
    aec_cfg->enable_res         = 0;   /* linear AEC + external NN post seam */
    aec_cfg->return_res_context = 1;   /* fills formed_hop/error_spec/...    */
    /* delay_backward_quarantine_enabled stays at lib/aec's default (OFF).
     * The guard holds backward candidates only, for a bounded window after
     * which it accepts -- so a pre-echo mis-lock is DELAYED by the window,
     * not cured. Enabling it here is therefore a policy decision, and it
     * waits on a real-audio spot check with the deployed checkpoint. */
    /* spatial_linear_context stays 0: this is the single-lane seam, and the
     * res context (formed_hop) must be populated. Delay estimation stays on
     * (preset default) -- the AEC aligns far internally. */
    return 0;
}

/* ============================================================================
 * Pool sizing / carve. Carve order: self control block (which embeds the
 * Ulcnet analysis/synthesis/frame-scratch state and the shared window
 * table), then the AEC pool, then the chain's shared FFT handle.
 * ========================================================================== */

static size_t ulcnet_sub_pool_size(const AecConfig* aec_cfg) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    size_t fft_sz = fft_get_mem_size(ULCNET_N_FFT);
    if (aec_sz == 0 || fft_sz == 0) return 0;   /* validator rejected it */
    return ALIGN16(aec_sz) + ALIGN16(fft_sz);
}

static int ulcnet_pipeline_build(AudioPipelineUlcnet* p, void* pool, size_t pool_size,
                                 const AecConfig* aec_cfg) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    size_t fft_sz = fft_get_mem_size(ULCNET_N_FFT);
    uint8_t* ptr = (uint8_t*)pool;
    AecResContext ctx0;

    if (!pool || aec_sz == 0 || fft_sz == 0 ||
        pool_size < ALIGN16(aec_sz) + ALIGN16(fft_sz)) return -1;

    p->aec = aec_init(ptr, aec_sz, aec_cfg);
    if (!p->aec) return -1;
    ptr += ALIGN16(aec_sz);

    /* ONE shared 512-point handle for the whole ULCNet chain (err-analysis
     * + far-analysis + synthesis) -- their transforms are strictly
     * sequential within a hop, per ulcnet_process.h's sharing contract.
     * Separate from the AEC's own internal FFT instance. */
    p->fft = fft_init(ptr, fft_sz, ULCNET_N_FFT);
    if (!p->fft || fft_get_n_freqs(p->fft) != ULCNET_BINS) return -1;

    /* Grid-agreement guard, readable right after aec_init (before any frame
     * is processed) -- same shape as audio_pipeline.c's pipeline_build. */
    aec_get_res_context(p->aec, &ctx0);
    if (ctx0.n_freqs != ULCNET_BINS || ctx0.hop_size != ULCNET_HOP) return -1;

    ulcnet_make_window(p->ulcnet_window);
    if (ulcnet_analysis_init(&p->ana_err, p->fft, p->ulcnet_window) != 0 ||
        ulcnet_analysis_init(&p->ana_far, p->fft, p->ulcnet_window) != 0 ||
        ulcnet_synthesis_init(&p->synth, p->fft, p->ulcnet_window) != 0)
        return -1;
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
     * whenever this structure changes -- always both, forever.
     * v2: far_input_mode joined the control block.
     * v3: ulcnet_window joined the control block (shared table replacing
     * the per-struct copies; the chain structs also grew embedded FFT
     * scratch) and the carve grew a trailing shared `fft` handle region.
     * v5: the self-resident `model` copy grew io_descriptor, so the token
     * names it explicitly -- the control block is bigger even though the
     * carve ORDER is unchanged. v6 removes the runtime far-mode field and
     * adds delay-transition bookkeeping. v7 adds the identity-reprime
     * counter beside it. */
    h = ulcnet_fnv1a_str("|carve:self(model(io_descriptor),"
                         "aec_delay_cfg,delay_transition,reprime,ana_err,"
                         "ana_far,synth,frame_scratch,ulcnet_window),aec,fft", h);
    h = ulcnet_fnv1a_str("|align16", h);
    return h;
}

/* NaN/Inf guard: a model frame with ANY non-finite value must never reach
 * the WOLA (the synthesis accumulator would poison every later hop). */
static int ulcnet_frame_is_finite(const float* re, const float* im) {
    int k;
    for (k = 0; k < ULCNET_BINS; k++) {
        if (!isfinite(re[k]) || !isfinite(im[k])) return 0;
    }
    return 1;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

AudioPipelineUlcnetConfig audio_pipeline_ulcnet_default_config(int sample_rate) {
    AudioPipelineUlcnetConfig cfg;
    memset(&cfg, 0, sizeof(cfg));       /* all-zero model == identity */
    cfg.sample_rate = sample_rate;
    cfg.fft_size    = ULCNET_N_FFT;     /* trained grid: frame/FFT 512 */
    cfg.filter_length = 0;
    cfg.delay_mode = AEC_DELAY_MATCHED;
    cfg.delay_num_filters = 5;
    cfg.fixed_delay_samples = -1;
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
    int n_frames, f, wrote, k;

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

    /* Stage 3: alignment boundary -> flush the runtime's far attention ring
     * + logit history BEFORE this hop's inference, and arm the identity
     * reprime that keeps the straddling frames out of the freshly cleared
     * state (header: "Identity reprime"). MATCHED reports CHANGED on the hop
     * that bumps the generation. FIXED has no estimator event, so its first
     * UNLOCKED->LOCKED transition is detected locally. The C STFT states
     * keep running across the change; the reprime, not a crossfade, is what
     * covers the straddling frames in this version. The counter is armed
     * whether or not a runtime is attached, so the frame policy does not
     * depend on the model's presence. */
    if (lctx.delay_state == AEC_LINEAR_DELAY_CHANGED ||
        (p->processed_hops != 0 &&
         p->last_delay_state == AEC_LINEAR_DELAY_UNLOCKED &&
         lctx.delay_state == AEC_LINEAR_DELAY_LOCKED)) {
        if (p->model.reset) p->model.reset(p->model.user);
        p->reprime_frames = AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES;
    }
    p->last_delay_state = lctx.delay_state;
    p->processed_hops += 1;

    /* Stage 4: push BOTH hops (the two analyses must stay frame-locked, so
     * each is fed every hop unconditionally). 0/2/1 emission: hop #0 emits
     * nothing, hop #1 emits two frames, then one per hop. The far branch
     * source is always the AEC's aligned far. */
    n_frames = ulcnet_analysis_push(&p->ana_err, rctx.formed_hop, p->err_re, p->err_im);
    (void)ulcnet_analysis_push(&p->ana_far,
                               lctx.aligned_far_hop,
                               p->far_re, p->far_im);

    /* Stage 5: per emitted frame pair, run the model and synthesize. The
     * model always receives the AEC seam's best available far: raw before
     * acquisition, aligned afterward. Its D window handles any remaining
     * offset. A CHANGED event resets state before this hop's inference, and
     * the frames whose analysis windows still straddle that boundary emit
     * the identity WITHOUT stepping the model (they would otherwise rebuild
     * the just-cleared state from half-stale input). Skipping inference
     * lowers the per-frame cost of those frames; it never doubles it. */
    wrote = 0;
    for (f = 0; f < n_frames; f++) {
        const float* sre = p->err_re[f];
        const float* sim = p->err_im[f];
        if (p->reprime_frames > 0) {
            p->reprime_frames--;      /* identity; model deliberately idle */
        } else if (p->model.infer) {
            int rc;
            /* Enforce ulcnet_process.h's FULL-WRITE CONTRACT: pre-fill the
             * model-output staging with NaN before every infer call, so a
             * partial write (rc == 0 without writing all ULCNET_BINS)
             * leaves non-finite bins behind and is rejected by the finite
             * guard below (fail-open identity) instead of silently applying
             * stale finite values left over from a previous frame. */
            for (k = 0; k < ULCNET_BINS; k++) {
                p->mdl_re[k] = NAN;
                p->mdl_im[k] = NAN;
            }
            rc = p->model.infer(p->model.user,
                                p->err_re[f], p->err_im[f],
                                p->far_re[f], p->far_im[f],
                                p->mdl_re, p->mdl_im);
            if (rc == 0 &&
                ulcnet_frame_is_finite(p->mdl_re, p->mdl_im)) {
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
    /* Re-init with the SAME shared handle/window (both pool/instance
     * resident and untouched by reset); cannot fail for a handle already
     * validated at build time. */
    (void)ulcnet_analysis_init(&p->ana_err, p->fft, p->ulcnet_window);
    (void)ulcnet_analysis_init(&p->ana_far, p->fft, p->ulcnet_window);
    (void)ulcnet_synthesis_init(&p->synth, p->fft, p->ulcnet_window);
    p->last_delay_state = AEC_LINEAR_DELAY_UNLOCKED;
    p->processed_hops = 0;
    /* The analysis history is zeroed above, so nothing emitted after this
     * point straddles anything: drop any pending reprime. */
    p->reprime_frames = 0;
    /* The external runtime owns the NN's explicit states -- tell it to
     * flush them too, same contract as the CHANGED event. */
    if (p->model.reset) p->model.reset(p->model.user);
}

void audio_pipeline_ulcnet_destroy(AudioPipelineUlcnet* p) {
    if (!p) return;
    /* Reverse carve order (AEC then the shared FFT handle sit in the
     * sub-pool). Genuine no-ops for a pool-resident instance today; kept
     * for the heap path and forward-compat, mirroring
     * audio_pipeline_destroy's rationale. */
    if (p->fft) fft_destroy(p->fft);
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
