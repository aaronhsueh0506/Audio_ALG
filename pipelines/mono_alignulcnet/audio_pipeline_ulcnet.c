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
 *   3. far tap = the caller's raw reference.  This is the signal used during
 *      training; the model's own TA block, not the PBFDKF delay buffer, owns
 *      neural far/error alignment.  AecLinearContext is still read only for
 *      the error-branch delay-transition reset policy.
 *   4. Both hops go into two rolling (center 0) AcStft analysis objects (exactly one frame per hop from hop #0); on that
 *      frame pair the model callback runs once; its output is
 *      applied only when infer() returned 0 and the output frame is fully
 *      finite; otherwise the error passes unchanged (fail-open); the chosen
 *      frame goes into the AcIstft synthesis
 *      (WOLA). hop #0 emits nothing -> zeros; hop #p output corresponds to
 *      input hop p-1 (one-hop latency).
 *   5. On a CHANGED event, or FIXED's first UNLOCKED->LOCKED transition,
 *      model->reset (if set) runs BEFORE that hop's frames so the runtime
 *      flushes its far attention ring + logit history, and the identity
 *      reprime is armed: the C STFT states keep running, so the next
 *      AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES emitted frames still straddle
 *      the boundary and are emitted as identity WITHOUT stepping the model
 *      (header: "Identity reprime"; the constant is measured by the
 *      straddle-derivation test).
 *
 * Constraints: C99, -ffp-contract=off (pipelines/Makefile appends it last),
 * no stdio in this TU (all failures are signalled by NULL/-1 returns only),
 * no heap on the init/init_ex path (create() is the explicit heap
 * convenience). The per-frame spectrum scratch and the one shared sqrt-Hann
 * window table are plain fixed-size arrays kept INSIDE the instance struct
 * (part of the `self` carve) rather than stack locals, and the analysis /
 * synthesis objects carry their own per-call FFT scratch in the pool --
 * multi-KB frame scratch would be unsafe headroom for an embedded RTOS task
 * stack (same rationale as lib/aec's stack-safety rule). The
 * chain's FFT is ONE pool-carved compiled-grid fft_wrapper handle shared by
 * err-analysis + far-analysis + synthesis (strictly sequential use within
 * a hop, per ulcnet_process.h's sharing contract).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "audio_pipeline_ulcnet.h"
#include "utility/mem_align.h"       /* ALIGN16 / MEM_IS_ALIGNED16 */
#include "common/se_config.h"  /* se_derive(): grid + AecConfig, shared by every application */
#include "utility/simd_kernel_nn.h"    /* skn_fill_f32 */

/* This wrapper's own carve-layout version (see AudioPipelineUlcnetMemReq's
 * doc). Any change of pool size, carve order or descriptor content bumps it,
 * including a change of the self-resident Ulcnet state block or of a composed
 * pool; the token string in audio_pipeline_ulcnet_build_flags_hash() moves
 * with a changed carve. */
#define AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION 14u

/* Compile-time FFT backend identity -- same mechanism as audio_pipeline.c:
 * pipelines/Makefile passes -DAUDIO_PIPELINE_BACKEND_STR=\"ne10\"
 * to every TU it compiles; the fallback only fires outside that Makefile
 * and is rejected by get_mem_requirements (backend_id 0 is reserved). */
#ifndef AUDIO_PIPELINE_BACKEND_STR
#define AUDIO_PIPELINE_BACKEND_STR "unknown"
#endif

static uint32_t audio_pipeline_ulcnet_backend_id(void) {
    if (strcmp(AUDIO_PIPELINE_BACKEND_STR, "ne10") == 0)
        return AUDIO_PIPELINE_ULCNET_BACKEND_NE10;
    return 0u;
}

/* The deployment grid is a build parameter of ulcnet_model_io.h, so this
 * wrapper follows whichever grid the ULCNet layer was built for rather than
 * pinning a rate. Asserting hop == fft/2 or bins == fft/2+1 here would
 * assert nothing: ULCNET_HOP and ULCNET_BINS ARE those expressions, so the
 * comparison is an identity for every -D. What can actually fail is the FFT
 * size itself, and ulcnet_model_io.h's own #error rejects a bad pair first;
 * this repeats the FFT half so a wrapper compiled against a future third
 * grid stops here rather than sizing its buffers for one it never saw. */
_Static_assert(ULCNET_N_FFT == 512 || ULCNET_N_FFT == 1024,
               "ULCNet wrapper supports FFT 512 (16 kHz) or 1024 (48 kHz)");

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
     * once per hop (the rolling push emits exactly one frame per hop).
     * While nonzero the frame takes the identity path and the model is not
     * stepped. Re-armed, never accumulated, by a boundary during a reprime. */
    int reprime_frames;

    /* The hop in flight (process_begin .. process_end). */
    int phase;                    /* 0 idle, 1 begun, 2 model frame open */
    int nn_wanted;                /* the frame has not been fetched and is not reprime */
    int nn_accepted;              /* the model output replaces the error frame */

    Aec* aec;                     /* points into `pool` below                */
    FftHandle* fft;               /* points into `pool`; ONE shared compiled-grid
                                   * handle for the whole ULCNet chain
                                   * (strictly sequential use per hop)      */

    /* Ulcnet framing objects, carved from `pool`. Each carries its own
     * per-call FFT scratch and borrows the shared handle above and the
     * window table below. */
    AcStft  *ana_err;             /* linear-error analysis                   */
    AcStft  *ana_far;             /* raw-far analysis                        */
    AcIstft *synth;               /* enhanced-spectrum WOLA                  */
    float ulcnet_window[ULCNET_N_FFT];  /* shared sqrt-Hann table; all three
                                   * framing objects borrow it (self-owned) */

    /* Per-hop frame scratch (one frame per hop). Kept in the instance, not
     * the stack -- see the header comment of this file. */
    float err_re[ULCNET_BINS], err_im[ULCNET_BINS];
    float far_re[ULCNET_BINS], far_im[ULCNET_BINS];
    float mdl_re[ULCNET_BINS],    mdl_im[ULCNET_BINS];

    /* pool bookkeeping */
    void*  pool;                  /* sub-pool AFTER this struct: AEC only    */
    size_t pool_size;
    void*  owned_heap;            /* non-NULL iff from create(); freed by destroy() */
};

/* ============================================================================
 * Config -> AEC config + dims (the one reject-first gate every entry point
 * funnels through)
 * ========================================================================== */

_Static_assert(ULCNET_HOP == ULCNET_N_FFT / 2 && ULCNET_BINS == ULCNET_N_FFT / 2 + 1,
               "the compiled grid is a no-padding 50 % overlap grid");

/* Returns 0 and fills every out-param, or -1: NULL cfg; sample_rate differs
 * from ULCNET_SR; fft_size is neither 0 nor ULCNET_N_FFT (0 resolves to the
 * compiled checkpoint grid); aec_preset lies outside its defined enum values
 * (rather than silently falling through aec_config_from_preset's own
 * balanced-default fallback); or filter_length is outside
 * [0, SE_FILTER_LENGTH_MAX]. The model's callbacks are deliberately NOT
 * validated -- an all-zero model is the supported identity/fail-open case. A
 * published model-I/O descriptor must match the fixed raw-far production ABI.
 *
 * The delay settings are passed through to the AEC, which aligns the far
 * reference itself and validates them when it sizes its pool. The seam keeps
 * only the linear context (formed_hop and the linear spectra); r2, comfort
 * noise and G_res are not read here, and the linear outputs are bit-identical
 * either way. delay_backward_quarantine keeps lib/aec's default (off). */
static int ulcnet_derive_dims_and_config(const AudioPipelineUlcnetConfig* cfg,
                                         AecConfig* aec_cfg,
                                         int* hop, int* fft_sz, int* n_freqs) {
    static const SeRules rules = {
        /* allow_8k */ 0, /* fixed_rate */ ULCNET_SR, /* fixed_fft */ ULCNET_N_FFT,
        /* check_delay */ 0, /* lane_aligned */ 0,
        /* spatial_context */ AEC_CONTEXT_LINEAR_ONLY, /* with_nr */ 0
    };
    SeRequest req;
    SeGrid grid;

    if (!cfg) return -1;

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

    req.sample_rate = cfg->sample_rate;
    req.fft_size = cfg->fft_size;
    req.filter_length = cfg->filter_length;
    req.delay_mode = cfg->delay_mode;
    req.delay_num_filters = cfg->delay_num_filters;
    req.fixed_delay_samples = cfg->fixed_delay_samples;
    req.aec_preset = cfg->aec_preset;
    req.nr_mode = ANR_MODE_BALANCED;   /* unused: no NR stage */
    if (se_derive(&req, &rules, &grid, aec_cfg) != 0) return -1;

    *hop = grid.hop_size;
    *fft_sz = grid.fft_size;
    *n_freqs = grid.n_freqs;
    return 0;
}

/* ============================================================================
 * Pool sizing / carve. Carve order: self control block (which embeds the
 * frame scratch and the shared window table), then the AEC pool, the
 * chain's shared FFT handle, the two analysis objects and the synthesis
 * object.
 * ========================================================================== */

/* Bytes of an analysis (is_synthesis 0) or synthesis (1) object on the shared
 * handle and window table: the size of the object that carves its own, less
 * those two regions. The shared handle exists only after the pool does, so
 * the size cannot be asked for it directly. */
static size_t ulcnet_framing_bytes(int is_synthesis) {
    AcStftConfig sc;
    size_t own, shared;
    ulcnet_framing_config(&sc, NULL, NULL, 0);
    own = is_synthesis ? ac_istft_get_mem_size(&sc) : ac_stft_get_mem_size(&sc);
    shared = ALIGN16(fft_get_mem_size(ULCNET_N_FFT)) +
             ALIGN16((size_t)ULCNET_N_FFT * sizeof(float));
    return own > shared ? own - shared : 0;
}

static size_t ulcnet_sub_pool_size(const AecConfig* aec_cfg) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    size_t fft_sz = fft_get_mem_size(ULCNET_N_FFT);
    size_t ana_sz = ulcnet_framing_bytes(0);
    size_t syn_sz = ulcnet_framing_bytes(1);
    if (aec_sz == 0 || fft_sz == 0 || ana_sz == 0 || syn_sz == 0)
        return 0;   /* validator rejected it */
    return ALIGN16(aec_sz) + ALIGN16(fft_sz) + 2 * ALIGN16(ana_sz) +
           ALIGN16(syn_sz);
}

static int ulcnet_pipeline_build(AudioPipelineUlcnet* p, void* pool, size_t pool_size,
                                 const AecConfig* aec_cfg) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    size_t fft_sz = fft_get_mem_size(ULCNET_N_FFT);
    size_t ana_sz = ulcnet_framing_bytes(0);
    size_t syn_sz = ulcnet_framing_bytes(1);
    uint8_t* ptr = (uint8_t*)pool;
    AecResContext ctx0;
    AcStftConfig sc;

    if (!pool || aec_sz == 0 || fft_sz == 0 || ana_sz == 0 || syn_sz == 0 ||
        pool_size < ulcnet_sub_pool_size(aec_cfg)) return -1;

    p->aec = aec_init(ptr, aec_sz, aec_cfg);
    if (!p->aec) return -1;
    ptr += ALIGN16(aec_sz);

    /* ONE shared compiled-grid handle for the whole ULCNet chain (err-analysis
     * + far-analysis + synthesis) -- their transforms are strictly
     * sequential within a hop, per ulcnet_process.h's sharing contract.
     * Separate from the AEC's own internal FFT instance. */
    p->fft = fft_init(ptr, fft_sz, ULCNET_N_FFT);
    if (!p->fft || fft_get_n_freqs(p->fft) != ULCNET_BINS) return -1;
    ptr += ALIGN16(fft_sz);

    /* Grid-agreement guard, readable right after aec_init (before any frame
     * is processed) -- same shape as audio_pipeline.c's pipeline_build. */
    aec_get_res_context(p->aec, &ctx0);
    if (ctx0.n_freqs != ULCNET_BINS || ctx0.hop_size != ULCNET_HOP) return -1;

    ac_window_build(p->ulcnet_window, ULCNET_N_FFT);
    ulcnet_framing_config(&sc, p->fft, p->ulcnet_window, 0);
    p->ana_err = ac_stft_init(ptr, ana_sz, &sc);
    ptr += ALIGN16(ana_sz);
    p->ana_far = ac_stft_init(ptr, ana_sz, &sc);
    ptr += ALIGN16(ana_sz);
    p->synth = ac_istft_init(ptr, syn_sz, &sc);
    if (!p->ana_err || !p->ana_far || !p->synth) return -1;
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
     * whenever this structure changes -- always both. The self-resident
     * `model` copy carries io_descriptor, the delay-transition bookkeeping
     * and the identity-reprime counter; the frame scratch is one frame per
     * branch (the rolling push emits exactly one frame per hop). */
    h = ulcnet_fnv1a_str("|carve:self(model(io_descriptor),"
                         "aec_delay_cfg,delay_transition,reprime,"
                         "frame_scratch[1],ulcnet_window),aec,fft,"
                         "ana_err,ana_far,synth", h);
    h = ulcnet_fnv1a_str("|align|ulcnet" ULCNET_GRID_TOKEN, h);
    return ac_memreq_abi_hash(h);
}

/* NaN/Inf guard: a model frame with ANY non-finite value must never reach
 * the WOLA (the synthesis accumulator would poison every later hop). */
/* ============================================================================
 * Public API
 * ========================================================================== */

AudioPipelineUlcnetConfig audio_pipeline_ulcnet_default_config(int sample_rate) {
    AudioPipelineUlcnetConfig cfg;
    memset(&cfg, 0, sizeof(cfg));       /* all-zero model == identity */
    cfg.sample_rate = sample_rate;
    cfg.fft_size    = ULCNET_N_FFT;     /* compiled checkpoint grid */
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

int audio_pipeline_ulcnet_process_begin(AudioPipelineUlcnet* p, const float* mic,
                                        const float* ref) {
    AecResContext rctx;
    AecLinearContext lctx;

    if (!p || !mic || !ref || p->phase != 0) return -1;

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
     * covers the straddling frames. The counter is armed
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

    /* Stage 4: frame BOTH branches from the CURRENT hop with the rolling
     * (center=False) analysis: exactly one frame per hop from hop #0, over
     * the last N_FFT samples of each branch, so the two analyses stay
     * frame-locked by construction. The far branch source is the raw caller
     * reference, matching checkpoint training; PBFDKF alignment affects only
     * the linear-error branch. */
    (void)ac_stft_push_planar(p->ana_err, rctx.formed_hop, p->err_re, p->err_im);
    (void)ac_stft_push_planar(p->ana_far, ref, p->far_re, p->far_im);

    /* Stage 5 decision: the model runs once on this hop's frame pair, EXCEPT
     * while the identity reprime armed at the last boundary is still running:
     * that frame's window still covers the pre-switch hop, and stepping it
     * would rebuild the just-cleared state from half-stale input. Skipping
     * inference lowers the cost of that frame; it never doubles it: at most
     * ONE inference per hop, always. */
    p->nn_wanted = 0;
    p->nn_accepted = 0;
    if (p->reprime_frames > 0) p->reprime_frames--;   /* identity; model idle */
    else p->nn_wanted = 1;
    p->phase = 1;
    return 0;
}

int audio_pipeline_ulcnet_nn_begin(AudioPipelineUlcnet* p, UlcnetNnJob* job) {
    if (!p || !job || p->phase != 1) return -1;
    if (!p->nn_wanted) return 0;
    p->nn_wanted = 0;
    /* FULL-WRITE CONTRACT: the staging is NaN, so a model that reports
     * success without writing every bin is caught by nn_end's finite guard. */
    skn_fill_f32(p->mdl_re, ULCNET_BINS, NAN);
    skn_fill_f32(p->mdl_im, ULCNET_BINS, NAN);
    job->err_re = p->err_re; job->err_im = p->err_im;
    job->far_re = p->far_re; job->far_im = p->far_im;
    job->out_re = p->mdl_re; job->out_im = p->mdl_im;
    p->phase = 2;
    return 1;
}

int audio_pipeline_ulcnet_nn_end(AudioPipelineUlcnet* p, int model_rc) {
    if (!p || p->phase != 2) return -1;
    /* Fail-open: a failed run, or any non-finite bin, leaves the identity. */
    p->nn_accepted = model_rc == 0 && ulcnet_frame_is_finite(p->mdl_re, p->mdl_im);
    p->phase = 1;
    return 0;
}

int audio_pipeline_ulcnet_process_end(AudioPipelineUlcnet* p, float* out) {
    int wrote;
    const float* sre;
    const float* sim;
    if (!p || !out || p->phase == 0) return -1;
    /* A frame left open or never fetched is the identity. */
    if (p->phase == 2) p->nn_accepted = 0;
    sre = p->nn_accepted ? p->mdl_re : p->err_re;
    sim = p->nn_accepted ? p->mdl_im : p->err_im;
    p->phase = 0;
    wrote = ac_istft_push_planar(p->synth, sre, sim, out);

    /* hop #0: the synthesis holds its first frame back (that frame's
     * first half is the trimmed half window), so nothing is emitted and the
     * one-hop-latency contract says the caller's output for this hop is
     * silence. Every later hop emits exactly one hop of samples, so this
     * memset never runs again. */
    if (wrote < p->hop)
        memset(out + wrote, 0, (size_t)(p->hop - wrote) * sizeof(float));
    return 0;
}

int audio_pipeline_ulcnet_process(AudioPipelineUlcnet* p, const float* mic,
                                  const float* ref, float* out) {
    UlcnetNnJob job;
    if (!p || !mic || !ref || !out) return -1;
    if (audio_pipeline_ulcnet_process_begin(p, mic, ref) != 0) return -1;
    /* No runtime attached: the frame stays unfetched and is the identity. */
    if (p->model.infer && audio_pipeline_ulcnet_nn_begin(p, &job) == 1)
        (void)audio_pipeline_ulcnet_nn_end(
            p, p->model.infer(p->model.user, job.err_re, job.err_im,
                              job.far_re, job.far_im, job.out_re, job.out_im));
    return audio_pipeline_ulcnet_process_end(p, out);
}

void audio_pipeline_ulcnet_reset(AudioPipelineUlcnet* p) {
    if (!p) return;
    aec_reset(p->aec);
    /* The objects keep the shared handle and window (both pool/instance
     * resident and untouched by reset); reset clears their history. */
    ac_stft_reset(p->ana_err);
    ac_stft_reset(p->ana_far);
    ac_istft_reset(p->synth);
    p->last_delay_state = AEC_LINEAR_DELAY_UNLOCKED;
    p->processed_hops = 0;
    p->phase = 0;
    p->nn_wanted = 0;
    p->nn_accepted = 0;
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
     * sub-pool). Genuine no-ops for a pool-resident instance; kept
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
