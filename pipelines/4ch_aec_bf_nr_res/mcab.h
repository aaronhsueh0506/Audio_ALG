/** mcab: the four-microphone linear-AEC flow. It composes libaec objects the application
 * created and owns: four lanes (Aec) and, in MATCHED, the shared delay estimator (DelayAec).
 * mcab never allocates, resets-by-destroying or frees one of them; it keeps only flow state in
 * its own pool (delay admission, aligned far, per-lane views, fuse buffers). The beamformer is
 * the caller's: it runs on McabFront.linear_spectra and hands its weights to mcab_fuse(). RES,
 * NR, comfort noise, synthesis and neural post-filters are chained by the application over the
 * frame mcab_fuse() returns.
 *
 * Per hop:
 *     mcab_process_front()  shared delay + four lanes -> McabFront (lane spectra, aligned far)
 *     caller SRP/BF         weights W[4][n_freqs] (and its beamformed spectrum E)
 *     mcab_fuse(p, W, E)    project near / R^2 / comfort noise -> McabFrame
 * With cfg.enable_post = 0 the lanes keep only the linear context and mcab_fuse() is refused.
 *
 * Init sequence (every object in caller memory, no heap anywhere):
 *     McabConfig cfg = mcab_default_config(16000);
 *     AecConfig lane;  aec_config_from_level(&lane, level, 16000);   // the application's lane:
 *     lane.filter_length = ...;                                        // level, filter length, ...
 *     mcab_apply_lane_contract(&cfg, &lane);    // (mcab_lane_config(): the balanced default)
 *     McabLibs libs = {0};
 *     for each ch: libs.lanes[ch] = aec_init(mem_ch, aec_get_mem_size(&lane), &lane);
 *     McabDelayEstimatorConfig dc;  mcab_delay_estimator_config(&cfg, &dc);
 *     if (dc.required) { delay_aec_init(&delay, mem_d, delay_aec_get_mem_size(dc.sample_rate,
 *                            dc.hop_size, dc.num_filters), dc.sample_rate, dc.hop_size,
 *                            dc.num_filters);  libs.delay = &delay; }
 *     McabMemReq req;  mcab_get_mem_requirements(&cfg, &req);
 *     Mcab* p = mcab_init_ex(pool, req.bytes, &cfg, &libs, &req);
 *     ...  mcab_destroy(p);  then the caller tears its lanes down and frees the memory.
 *
 * What mcab_init() can and cannot check: every lane's hop and bin count, that it returns the
 * linear context without a gain of its own (enable_res = 0, return_res_context = 1), that its
 * post half matches cfg.enable_post, that it consumes an already aligned far (EXTERNAL_ALIGNED
 * reports a locked zero offset) and that the four are distinct; the estimator's bank size, rate
 * and 48 kHz scratch. Aec has no config getter: the lane parameters (level, filter length, ...)
 * are the application's and are not checked, and the contract fields are only as right as the
 * mcab_apply_lane_contract() call that stamped them.
 *
 * The lanes and the estimator must be fresh: newly initialised, or reset (aec_reset() /
 * delay_aec_reset(), or mcab_reset() of the flow that ran them) before mcab_init(). mcab cannot
 * see that an object has already run; a used estimator would hand the new flow a stale delay. */
#ifndef MCAB_H
#define MCAB_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"               /* Aec, AecConfig, DelayAec, AecDelayMode, Complex */

#ifdef __cplusplus
extern "C" {
#endif

#define MCAB_CHANNELS 4

#define MCAB_DESCRIPTOR_VERSION 1u
/* Carve layout of mcab's own pool: control block, mic lane, aligned far, the fuse buffers
 * (enable_post) and the alignment ring (MATCHED/FIXED). Library objects are not in it. 21..24
 * belonged to the predecessor layers and are never reused. Query a fresh descriptor for every
 * build; never persist or synthesize one. */
#define MCAB_LAYOUT_VERSION 25u
#define MCAB_BACKEND_NE10 2u          /* backend_id 1 is never returned */

/** Fixed 32-byte descriptor of the caller pool: descriptor_version (ABI, checked first),
 * layout_version, backend_id (compared with ==), build_flags_hash (FNV-1a-32 over the carve
 * structure, not tunable values), alignment, reserved (0, validated), bytes. */
typedef struct McabMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} McabMemReq;
_Static_assert(sizeof(McabMemReq) == 32, "McabMemReq must be exactly 32 bytes");
_Static_assert(offsetof(McabMemReq, descriptor_version) == 0, "descriptor_version offset");
_Static_assert(offsetof(McabMemReq, layout_version) == 4, "layout_version offset");
_Static_assert(offsetof(McabMemReq, backend_id) == 8, "backend_id offset");
_Static_assert(offsetof(McabMemReq, build_flags_hash) == 12, "build_flags_hash offset");
_Static_assert(offsetof(McabMemReq, alignment) == 16, "alignment offset");
_Static_assert(offsetof(McabMemReq, reserved) == 20, "reserved offset");
_Static_assert(offsetof(McabMemReq, bytes) == 24, "bytes offset");

typedef enum McabStatus {
    MCAB_OK = 0,
    MCAB_INVALID_ARGUMENT = -1,   /* refused before touching adaptive state (see each call) */
    MCAB_SEQUENCE_ERROR = -2,     /* call out of order; nothing changed                      */
    MCAB_DSP_ERROR = -3           /* a stage refused the hop; the flow has been reset        */
} McabStatus;

/** The flow's parameters. The lane parameters (level, filter length, ...) are not here: the
 * application composes each lane's AecConfig and stamps this flow's contract onto it with
 * mcab_apply_lane_contract(). */
typedef struct McabConfig {
    int sample_rate;              /* 16000 or 48000                                          */
    int fft_size;                 /* 0 = rate default; 256/512 @16k, 1024 @48k               */
    AecDelayMode delay_mode;      /* shared MATCHED / FIXED / EXTERNAL_ALIGNED policy         */
    int delay_num_filters;        /* MATCHED bank size [1, DA_NUM_FILTERS]                   */
    int fixed_delay_samples;      /* FIXED: native-rate samples; -1 otherwise                */
    int capture_proxy_channel;    /* shared estimator's capture input, [0, 3]                */
    /* Backward-jump quarantine (MATCHED only, default off): an estimate earlier than the delay
     * in force is held for delay_backward_quarantine_s while the capture-proxy lane still
     * cancels, then adopted; lost cancellation adopts it at once. Forward moves and first
     * acquisition are never held. */
    int delay_backward_quarantine_enabled; /* bool                                           */
    float delay_backward_quarantine_s;     /* [0, 3600]; default 1.0                          */
    /* Estimator duty cycle: once the estimate held still for 0.3 s, analyse one hop in
     * round(delay_est_period_s / hop_s) / 5. 0.0 (default) = full rate. [0, 3600]. */
    float delay_est_period_s;
    float max_delay_ms;           /* MATCHED alignment-ring capacity, [0, 4096]              */
    int enable_post;              /* 1: R^2/comfort noise/fuse buffers for mcab_fuse();
                                     0: front only (lanes use AEC_CONTEXT_LINEAR_ONLY)       */
} McabConfig;

/** The libaec objects one flow drives, created and owned by the caller; every one must outlive
 * the Mcab handle. */
typedef struct McabLibs {
    Aec* lanes[MCAB_CHANNELS];    /* aec_init() of a contract-stamped config; four distinct  */
    DelayAec* delay;              /* MATCHED: delay_aec_init() per mcab_delay_estimator_config();
                                     NULL otherwise                                          */
} McabLibs;

/** Parameters of the shared delay estimator a config needs. */
typedef struct McabDelayEstimatorConfig {
    int required;                 /* 1 in MATCHED (McabLibs.delay must be given), else 0     */
    int sample_rate;
    int hop_size;
    int num_filters;
} McabDelayEstimatorConfig;

/** Per-hop view of the shared reference alignment. `changed`: a new usable alignment generation
 * begins on this hop (MATCHED only: the first acquisition after init or reset, or an admitted
 * move while locked: more than 32 samples, offered again within 16 before its 3 hops run out;
 * FIXED and EXTERNAL_ALIGNED never set it); a consumer holding history derived from the aligned
 * far must flush it before the frame it produces for this hop. `solid`: a usable accepted
 * alignment exists (MATCHED: sticky from the first acquisition until reset; FIXED: ring fill;
 * EXTERNAL_ALIGNED: always 1). delay_samples moves only with `changed`. */
typedef struct McabDelayState {
    int delay_samples;
    float confidence;
    int solid;
    int changed;
    uint64_t estimator_calls;
    int estimator_updates;
} McabDelayState;

/** What mcab_process_front() returns: read-only views into lane and mcab memory, valid until the
 * next mcab_process_front(), mcab_reset() or mcab_destroy() on the same instance. */
typedef struct McabFront {
    McabDelayState delay;
    int hop_size;
    int n_channels;                                 /* always MCAB_CHANNELS                  */
    int n_freqs;
    const Complex* linear_spectra[MCAB_CHANNELS];   /* lane error spectra, n_freqs bins      */
    const float* formed_hop[MCAB_CHANNELS];         /* time-domain hop under each spectrum   */
    /* The aligned far fed to every lane: the caller's far shifted by delay.delay_samples once
     * the ring can serve that offset, the raw far hop before (whole-hop switch). */
    const float* aligned_ref;
} McabFront;

/** What mcab_fuse() returns: a borrowed view of one fused hop, never written through. Scales:
 * e, y, x in audio scale; r2 and cn in int16^2 power scale; gains are amplitudes. */
typedef struct McabFrame {
    int n_freqs;              /* bins of every spectrum / gain vector            */
    int hop;                  /* time-domain hop of far_hop / formed_hop         */
    const Complex* e;         /* the beamformed linear-AEC error E(f)            */
    const Complex* y;         /* capture spectrum matching e                     */
    const Complex* x;         /* far-end spectrum                                */
    const float* r2;          /* residual-echo PSD R^2(f)                        */
    const float* cn;          /* comfort-noise PSD N^2(f)                        */
    const float* g_res_ctx;   /* always NULL: the lanes compute no gain of their own */
    const float* far_hop;     /* aligned far-end samples of this hop             */
    const float* formed_hop;  /* NULL: one hop per lane in McabFront.formed_hop  */
    float far_power;
    float max_dt;             /* worst double-talk indicator                     */
    float max_sat;            /* worst saturation level                          */
    int all_converged;        /* every contributing linear filter has converged  */
    int usable_linear;        /* r2 derives from a usable linear estimate        */
    int clock_drift;
    int delay_changed;        /* this hop starts a new alignment generation      */
    int delay_locked;         /* a usable alignment is in force                  */
} McabFrame;

/** Split of mcab's own pool (library objects excluded). */
typedef struct McabMemBreakdown {
    size_t control_bytes;         /* control block                                           */
    size_t buffer_bytes;          /* mic lane, aligned far, fuse buffers, alignment ring      */
    size_t total_bytes;           /* == McabMemReq.bytes                                     */
    int hop_size;
    int fft_size;
    int n_freqs;
} McabMemBreakdown;

/** Wall clock of the most recent hop in microseconds, diagnostic only; all 0 unless built with
 * -DMCAB_STAGE_TIMING=1 (delay, fuse) and -DAEC_STAGE_TIMING=1 (lane figures); `make PROFILE=1`
 * sets both. A target without CLOCK_MONOTONIC names its timer with -DMCAB_NOW_US=<fn>. A call
 * refused for a NULL argument or a destroyed handle leaves the record untouched. Any other
 * mcab_process_front() call first zeroes the four front fields (delay_us, frontend_us,
 * linear_us, lane_res_us), so a hop that stops early reports 0 for the stages it never reached,
 * a non-finite hop included; fuse_us keeps its previous value there and is zeroed only by a
 * mcab_fuse() call that passes its argument, sequence and weight checks. */
typedef struct McabLastTiming {
    uint32_t delay_us;            /* shared delay estimator                                  */
    uint32_t frontend_us;         /* four lanes: entry to the main filter                    */
    uint32_t linear_us;           /* four lanes: main adaptive filter incl. far FFT          */
    uint32_t lane_res_us;         /* four lanes: AEC post block                              */
    uint32_t fuse_us;             /* mcab_fuse()                                             */
} McabLastTiming;

typedef struct Mcab Mcab;

/* ---- configuration helpers (no allocation; 0 on success, -1 for an invalid config) -------- */

McabConfig mcab_default_config(int sample_rate);

/** Stamps the flow's contract onto an application-composed lane config: fft_size = the flow's
 * grid, delay_mode = EXTERNAL_ALIGNED (delay_num_filters = DA_NUM_FILTERS,
 * fixed_delay_samples = -1, enable_delay_est = 0), enable_res = 0, return_res_context = 1,
 * spatial_linear_context = AEC_CONTEXT_NO_GAIN with cfg.enable_post, else
 * AEC_CONTEXT_LINEAR_ONLY, delay_est_period_s = cfg's. Every other field (level values,
 * filter_length, ...) is left as given; idempotent. -1 (lane unchanged) for an invalid cfg or a
 * lane built for another sample rate. */
int mcab_apply_lane_contract(const McabConfig* cfg, AecConfig* lane);

/** The default lane: aec_config_from_level(balanced, cfg rate) + mcab_apply_lane_contract(). */
int mcab_lane_config(const McabConfig* cfg, AecConfig* out);

/** The shared estimator's (sample_rate, hop_size, num_filters); required only in MATCHED. */
int mcab_delay_estimator_config(const McabConfig* cfg, McabDelayEstimatorConfig* out);

/* ---- lifecycle ------------------------------------------------------------------------- */

int mcab_get_mem_requirements(const McabConfig* cfg, McabMemReq* out);
int mcab_get_mem_breakdown(const McabConfig* cfg, McabMemBreakdown* out);

/** mem: 16-byte-aligned caller pool of >= the queried bytes (dirty is fine); cfg: copied; libs:
 * the objects to drive (the struct is copied, the objects are borrowed). NULL on any refusal,
 * before the pool is written. Same as mcab_init_ex(..., NULL). */
Mcab* mcab_init(void* mem, size_t bytes, const McabConfig* cfg, const McabLibs* libs);

/** init plus the stale-descriptor gate: when expected is non-NULL, descriptor_version,
 * layout_version, backend_id, build_flags_hash and alignment must equal the current build's,
 * reserved must be 0, and expected->bytes and `bytes` must be >= the current bytes. */
Mcab* mcab_init_ex(void* mem, size_t bytes, const McabConfig* cfg, const McabLibs* libs,
                   const McabMemReq* expected);

/** Heap convenience for mcab's own pool only (the objects in libs stay the caller's). */
Mcab* mcab_create(const McabConfig* cfg, const McabLibs* libs);

/** Ends the flow; touches no library object. Caller pool: idempotent. mcab_create() handle:
 * frees its block (call once). */
void mcab_destroy(Mcab* p);

/** Resets the flow and, through their own reset functions, the lanes and the estimator.
 * Outstanding McabFront pointers then show stale content (not a fault); mcab_fuse() refuses
 * until a new front. */
void mcab_reset(Mcab* p);

/* ---- per hop --------------------------------------------------------------------------- */

/** Shared delay + four linear AEC lanes. microphones_interleaved: [hop][4]; far_reference:
 * [hop]. On OK the previous view and every frame fused from it are invalidated. Returns
 * MCAB_OK; MCAB_INVALID_ARGUMENT for a NULL argument (nothing changed) or a non-finite sample
 * (no adaptive state changed, no front pending); MCAB_DSP_ERROR when a lane refused the hop
 * (the flow has been reset). */
int mcab_process_front(Mcab* p, const float* microphones_interleaved,
                       const float* far_reference, McabFront* out);

/** Projects this hop's lane contexts with the caller's beamformer into *frame:
 * e = beamformed error, y = projected near, x = lane 0's far spectrum, r2/cn = projected
 * residual-echo and comfort-noise PSDs, far_hop = aligned far, max_dt/max_sat/all_converged/
 * usable_linear over lanes with a nonzero weight, clock_drift and the delay flags; g_res_ctx is
 * NULL. weights: channel-major Complex[4][n_freqs], out = sum(w * in), no conjugation.
 * beamformed_error: the spectrum of the same beamformer call (caller-owned, must outlive the
 * frame) or NULL to rebuild it from weights. Needs cfg.enable_post and a front; a second fuse of
 * the same front is allowed. Frame pointers are valid until the next front, reset or destroy.
 * Returns MCAB_OK; MCAB_INVALID_ARGUMENT (nothing changed) for a NULL argument, enable_post = 0
 * or a zero/non-finite weight set; MCAB_SEQUENCE_ERROR (nothing changed) without a front;
 * MCAB_DSP_ERROR when the lanes do not share one far spectrum, the beamformed spectrum is
 * non-finite or the projection overflowed (finite but huge weights) (the flow has been reset). */
int mcab_fuse(Mcab* p, const Complex* weights, const Complex* beamformed_error,
              McabFrame* frame);

/* ---- queries (NULL/destroyed handle: -1, 0 or a zeroed record as noted) ------------------ */
int mcab_hop_size(const Mcab* p);
int mcab_fft_size(const Mcab* p);
int mcab_n_freqs(const Mcab* p);
int mcab_sample_rate(const Mcab* p);
int mcab_enable_post(const Mcab* p);             /* cfg.enable_post: 1/0; -1 for NULL     */
int mcab_matched_filter_count(const Mcab* p);   /* 1 in MATCHED, else 0                      */
int mcab_linear_aec_count(const Mcab* p);       /* 4                                         */
void mcab_get_last_timing(const Mcab* p, McabLastTiming* out);  /* out must not be NULL      */

/* ---- read-only instrumentation (cumulative since init or reset) ------------------------- */
/* Lane 0 runs the far FFT and lanes 1-3 borrow it: +1 per front. Saturates at LONG_MAX. */
long mcab_far_fft_real_compute_count(const Mcab* p);
/* aec_apply_external_realign() outcomes; a `changed` hop adds 4 to the pair (warm: IR shifted,
 * soft: filter-only reset); fewer means a lane rejected the call. Saturates at LONG_MAX. */
long mcab_realign_warm_lane_count(const Mcab* p);
long mcab_realign_soft_lane_count(const Mcab* p);
int mcab_pending_delay_candidate(const Mcab* p);   /* -1 when none is held                  */
/* Duty census: _total = hops through the estimator (== estimator_calls), _run = analysed. */
unsigned long long mcab_duty_hops_total(const Mcab* p);
unsigned long long mcab_duty_hops_run(const Mcab* p);

#ifdef __cplusplus
}
#endif
#endif /* MCAB_H */
