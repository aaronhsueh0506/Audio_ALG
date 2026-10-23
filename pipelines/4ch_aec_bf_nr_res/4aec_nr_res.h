/** Four-microphone linear-AEC front end with one shared far aligner. The external
 * owner supplies SRP-PHAT/BF. This module owns one shared DelayAec matcher, four
 * independent linear AEC adaptive filters and the coherent projection of the four
 * lane contexts onto one mono frame. The residual-echo, noise-reduction, comfort-noise
 * and synthesis stages are separate libraries that the application initialises and
 * chains over the SeSpecFrame that four_aec_fuse() returns (common/se_stage.h; the
 * chain is shown in samples/se_sample_chain.h). Per hop:
 *   four_aec_front_process()  shared delay + four linear AEC lanes; the lane spectra
 *                             go to the external SRP-PHAT/BF
 *   (SRP-PHAT, BF)            effective weights and the beamformed error
 *   four_aec_fuse()           project near / R^2 / comfort noise -> SeSpecFrame
 *   stages                    se_res_stage_gain(), anr_process_gain(), ...
 *   sk_min_f32 -> sk_capply_gain_f32 -> aec_cng_fill -> ac_istft_push
 * Caller-owned pool (board/static path):
 *     FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
 *     FourAecNrResMemReq req;
 *     four_aec_nr_res_get_mem_requirements(&cfg, &req);
 *     void* pool = platform_alloc(req.bytes, req.alignment);
 *     FourAecNrRes* p = four_aec_nr_res_init_ex(pool, req.bytes, &cfg, &req);
 *     ...
 *     four_aec_nr_res_destroy(p);  // never releases caller-owned pool
 *     platform_free(pool);
 * Heap convenience (desktop/test path):
 *     FourAecNrRes* p = four_aec_nr_res_create(&cfg);
 *     ...
 *     four_aec_nr_res_destroy(p);  // releases create()'s one pool
 * See pipelines/README.md and 4ch_aec_bf_nr_res/README.md. */
#ifndef FOUR_AEC_NR_RES_H
#define FOUR_AEC_NR_RES_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"
#include "common/se_stage.h"    /* SeSpecFrame */

#ifdef __cplusplus
extern "C" {
#endif

#define FOUR_AEC_NR_RES_CHANNELS 4

#define FOUR_AEC_NR_RES_DESCRIPTOR_VERSION 1u
/* Carve layout version: control block, four lane pools, working buffers (mic lane,
 * aligned reference and, with cfg.enable_post, the fuse buffers) and the shared
 * delay estimator and ring. Bumped on any change of pool size, carve order or
 * descriptor content. Query a fresh descriptor for every build; never persist or
 * synthesize one from this number. */
#define FOUR_AEC_NR_RES_LAYOUT_VERSION 23u
#define FOUR_AEC_NR_RES_BACKEND_NE10 2u   /* backend_id 1 is never returned */

/** Fixed 32-byte descriptor for a caller-owned pool: descriptor_version (ABI, checked first), layout_version, backend_id (compared with ==), build_flags_hash (FNV-1a-32 over the carve structure, not tunable values), alignment, reserved (0, validated), bytes. Query it at init time and pass it to init_ex(); do not cache it across library, backend, flag or config changes. */
typedef struct FourAecNrResMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} FourAecNrResMemReq;

_Static_assert(
    sizeof(FourAecNrResMemReq) == 32,
    "FourAecNrResMemReq must be exactly 32 bytes");
_Static_assert(
    offsetof(FourAecNrResMemReq, descriptor_version) == 0,
    "descriptor_version offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, layout_version) == 4,
    "layout_version offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, backend_id) == 8,
    "backend_id offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, build_flags_hash) == 12,
    "build_flags_hash offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, alignment) == 16,
    "alignment offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, reserved) == 20,
    "reserved offset");
_Static_assert(
    offsetof(FourAecNrResMemReq, bytes) == 24,
    "bytes offset");

typedef enum FourAecNrResStatus {
    FOUR_AEC_NR_RES_OK = 0,
    FOUR_AEC_NR_RES_INVALID_ARGUMENT = -1,
    FOUR_AEC_NR_RES_SEQUENCE_ERROR = -2,
    FOUR_AEC_NR_RES_DSP_ERROR = -3
} FourAecNrResStatus;

typedef struct FourAecNrResConfig {
    int sample_rate;              /* 16000 or 48000                         */
    int fft_size;                 /* 0=default; 256/512 @16k, 1024 @48k   */
    int filter_length;            /* 0=rate default                         */
    AecDelayMode delay_mode;      /* shared MATCHED/FIXED/EXTERNAL policy   */
    int delay_num_filters;        /* MATCHED bank size [1,5]                */
    int fixed_delay_samples;      /* FIXED native-rate samples; -1 otherwise*/
    int capture_proxy_channel;    /* shared matcher input, [0,3]            */
    /* Backward-jump quarantine on a shared-delay change; default off; MATCHED only. A shared estimate strictly earlier than the accepted delay is held for delay_backward_quarantine_s while the estimator's capture lane (cfg.capture_proxy_channel) still cancels at the applied alignment, then adopted; a collapse in cancellation adopts it immediately. A forward estimate is never held and first acquisition is not guarded. The lane's ERLE reads 0 until its ERLE machinery has run, which leaves the guard inert. See aec.h. */
    int delay_backward_quarantine_enabled;  /* bool                         */
    float delay_backward_quarantine_s;      /* window, seconds; default 1.0 */
    /* Shared matched-filter duty-cycle throttle: once the published estimate has held still for delay_est_init_s (0.3 s, fixed), the matched filter is analysed once in round(delay_est_period_s / hop_s) / 5 hops instead of every hop (update_shared_delay() in 4aec_nr_res.c). 0.0 (default) = full rate. A positive value saves compute but may settle on a different lag. Range: finite, [0, 3600]. */
    float delay_est_period_s;
    float max_delay_ms;           /* MATCHED reference delay-line capacity   */
    int enable_post;              /* 1 = keep the post context (R^2, comfort noise, fuse buffers) for four_aec_fuse(); 0 = front only (AEC_CONTEXT_LINEAR_ONLY) */
} FourAecNrResConfig;

/** Per-hop view of the single shared reference alignment. `changed` marks the hop on which a new usable alignment generation begins: under MATCHED on first acquisition, on a relock after an unlocked stretch (even with an unchanged delay value) and on an admitted delay change while locked. A consumer holding history derived from the aligned reference (STFT/OLA tails, recurrent model state) must flush it on `changed`, before the frame it produces for this hop; `solid` alone is not enough. FIXED and EXTERNAL_ALIGNED never set `changed`. An admitted change while locked follows lib/aec's Path-B rule: movement above 32 native samples, offered again within 16, before the candidate's 3 hops run out; first acquisition is exempt. `delay_samples` never moves on a hop without `changed`.
 * `solid` = a usable accepted alignment exists. MATCHED: raised by the acceptance test that writes the applied delay, sticky until reset(); FIXED: the ring-fill state (raw far below, shifted far from that hop on); EXTERNAL_ALIGNED: always 1. */
typedef struct FourAecNrResDelayState {
    int delay_samples;
    float confidence;
    int solid;
    int changed;        /* reset external STFT/OLA history when non-zero */
    uint64_t estimator_calls;
    int estimator_updates;
} FourAecNrResDelayState;

/** What the front end returns for one hop (four_aec_front_process()). Every pointer is
 * a read-only view into a lane's or the core's per-hop buffer, valid until the next
 * four_aec_front_process(), four_aec_nr_res_reset() or four_aec_nr_res_destroy() on
 * the same instance. */
typedef struct FourAecFront {
    FourAecNrResDelayState delay;
    int hop_size;
    int n_channels;               /* always FOUR_AEC_NR_RES_CHANNELS        */
    int n_freqs;

    /** The linear-AEC error spectra of this hop, one per channel (n_freqs bins, audio
     * scale); an external SRP-PHAT/BF consumes these instead of a duplicate STFT. */
    const Complex* linear_spectra[FOUR_AEC_NR_RES_CHANNELS];

    /** The time-domain hop underlying each linear_spectra[channel] (hop_size samples,
     * before synthesis), for an external time-domain beamformer that does its own
     * sqrt-Hann analysis. */
    const float* formed_hop[FOUR_AEC_NR_RES_CHANNELS];

    /** The delay-aligned far reference fed to every AEC lane this hop (hop_size samples):
     * the caller's ref shifted by delay.delay_samples on hops the shared ring can serve
     * that offset, the caller's raw ref hop before that (whole-hop switch, coinciding
     * with `delay.solid` under FIXED). For an external neural post-filter that needs the
     * same aligned far as the linear filters. */
    const float* aligned_ref;
} FourAecFront;

typedef struct FourAecNrRes FourAecNrRes;

/** Returns the default config for sample_rate. */
FourAecNrResConfig four_aec_nr_res_default_config(int sample_rate);

/** Queries the 16-byte-aligned caller pool required by cfg. No allocation; returns 0
 * on success. */
int four_aec_nr_res_get_mem_requirements(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemReq* out);

/** Places the pipeline in caller-owned memory that satisfies the queried alignment
 * and size. No heap allocation, including inside the four AECs, NR and FFT backend.
 * A dirty pool is accepted. */
FourAecNrRes* four_aec_nr_res_init(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg);

/** init plus a stale-descriptor gate. `expected` should come from a fresh get_mem_requirements() for this exact build/config; NULL is the same as init(). When non-NULL, descriptor_version, layout_version, backend_id, build_flags_hash and alignment must equal the current build's, reserved must be 0, and expected->bytes and the pool bytes must be >= the current bytes, else NULL is returned (no diagnostic; no stdio symbols are linked). */
FourAecNrRes* four_aec_nr_res_init_ex(
    void* mem,
    size_t bytes,
    const FourAecNrResConfig* cfg,
    const FourAecNrResMemReq* expected);

/** Stage 1: the shared aligner and the four linear AEC filters. microphones_interleaved
 * is [hop][4], ref is [hop]; both read only. On OK, *out is the borrowed front view of
 * this hop; the previous hop's view and every SeSpecFrame fused from it are
 * invalidated. Returns FOUR_AEC_NR_RES_OK; INVALID_ARGUMENT on a NULL argument or a
 * non-finite sample; DSP_ERROR when a lane refused the hop (the instance has been
 * reset). */
int four_aec_front_process(
    FourAecNrRes* p,
    const float* microphones_interleaved,
    const float* ref,
    FourAecFront* out);

/** Stage 2: projects the front frame with the beamformer's effective weights and returns the frame the frequency-domain stages consume. weights is channel-major Complex[4][n_freqs]; output[bin] = sum(weights[channel,bin] * input[channel,bin]) (no conjugation). Fills *frame (se_stage.h): e = beamformed error, y = projected near spectrum, x = lane 0's far spectrum, r2/cn = projected residual-echo and comfort-noise PSDs, far_hop = aligned far, max_dt/max_sat/all_converged over lanes with a nonzero response, clock_drift and the delay flags; g_res_ctx is NULL. beamformed_error comes from the same bf_process_with_weights() call as `weights` (consistency is a caller contract); NULL reconstructs it from `weights`. A supplied spectrum stays caller-owned and must outlive the frame. Frame pointers are valid until the next four_aec_front_process(), reset or destroy. Requires cfg.enable_post. Returns FOUR_AEC_NR_RES_OK; INVALID_ARGUMENT (nothing changed) for a NULL argument, an all-zero/non-finite weight set or enable_post = 0; SEQUENCE_ERROR without a processed front frame; DSP_ERROR when the lanes do not share one far-end spectrum or a spectrum is non-finite (the instance has been reset). */
int four_aec_fuse(
    FourAecNrRes* p,
    const Complex* weights,
    const Complex* beamformed_error,
    SeSpecFrame* frame);

/** Resets all delay/AEC state and invalidates the front view and every frame fused from
 * it. A caller holding an outstanding FourAecFront must stop reading its pointers: they
 * alias lane buffers that reset does not clear (stale content, not a fault).
 * four_aec_fuse() refuses to run until a new front frame has been processed. */
void four_aec_nr_res_reset(FourAecNrRes* p);

/** Destroys an instance. Caller-owned memory: idempotent, the pool is not freed.
 * Heap-created handle: frees its pool (call once). */
void four_aec_nr_res_destroy(FourAecNrRes* p);

/** Heap convenience over get_mem_requirements() + init(). Returns NULL for an invalid
 * config or allocation failure. */
FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg);

int four_aec_nr_res_hop_size(const FourAecNrRes* p);
int four_aec_nr_res_fft_size(const FourAecNrRes* p);
int four_aec_nr_res_n_freqs(const FourAecNrRes* p);
int four_aec_nr_res_sample_rate(const FourAecNrRes* p);

/* Structural audit hooks: the matcher count is 1 only in MATCHED mode and 0 in
 * FIXED/EXTERNAL; the linear AEC count is 4. */
int four_aec_nr_res_matched_filter_count(const FourAecNrRes* p);
int four_aec_nr_res_linear_aec_count(const FourAecNrRes* p);

/* Read-only instrumentation: sum of aec_far_fft_real_compute_count() over the four
 * lanes (cumulative since construction or reset). Lane 0 runs the far-end FFT; lanes
 * 1-3 borrow its spectrum, so the total grows by 1 per four_aec_front_process() call.
 * Saturates at LONG_MAX. */
long four_aec_nr_res_far_fft_real_compute_count(const FourAecNrRes* p);

/* Realign instrumentation (read-only, cumulative since construction or reset, saturating at LONG_MAX): how the four lanes' aec_apply_external_realign() calls landed. A hop publishing `changed` sweeps all four lanes, so the pair grows by 4. Warm (1): the learned IR was shifted and cancellation survived; soft (0): filter-only reset, re-converging from zero. A sweep adding fewer than 4 means a lane rejected the call (a wiring fault). */
long four_aec_nr_res_realign_warm_lane_count(const FourAecNrRes* p);
long four_aec_nr_res_realign_soft_lane_count(const FourAecNrRes* p);

/* The shared-delay movement currently held for confirmation, in native samples, or -1
 * when none is held (including FIXED/EXTERNAL_ALIGNED). Diagnostic only. */
int four_aec_nr_res_pending_delay_candidate(const FourAecNrRes* p);

/* Matched-filter duty-cycle census (read-only, cumulative since construction or reset): _total counts hops through the shared delay stage, _run those that ran the matched-filter analysis; both are 0 outside AEC_DELAY_MATCHED. A movement during a decimated hop is picked up at the next scheduled analysis (up to one duty period, ~96 ms on the shipped grids). duty_hops_total equals FourAecNrResDelayState's estimator_calls. Diagnostic only. */
unsigned long long four_aec_nr_res_duty_hops_total(const FourAecNrRes* p);
unsigned long long four_aec_nr_res_duty_hops_run(const FourAecNrRes* p);

typedef struct FourAecNrResMemBreakdown {
    size_t aec_bytes;       /* four AEC pools combined */
    size_t wrapper_bytes;   /* control block + shared aligner and fuse buffers */
    size_t total_bytes;
    int hop_size;
    int fft_size;
    int n_freqs;
} FourAecNrResMemBreakdown;

/** Optional module/wrapper split for board memory-budget reporting. */
int four_aec_nr_res_get_mem_breakdown(
    const FourAecNrResConfig* cfg,
    FourAecNrResMemBreakdown* out);

/** Per-stage wall-clock cost of the most recent hop, in microseconds. Diagnostic only.
 *   delay_us     update_shared_delay(), the shared delay estimator only
 *   frontend_us  summed over the four lanes: entry to the main filter
 *   linear_us    summed over the four lanes: main adaptive filter incl. the far-end FFT
 *   lane_res_us  summed over the four lanes: the AEC post/RES block
 *   fuse_us      four_aec_fuse()
 * Lane figures come from aec_get_last_timing() (aec.h). The stages do not add up to the enclosing call. Resolution is 1 us. Timing is off by default and every field reads 0: build with -DFOUR_AEC_NR_RES_STAGE_TIMING=1 for delay/fuse and -DAEC_STAGE_TIMING=1 for the lane figures (`make PROFILE=1` sets both); a target without POSIX CLOCK_MONOTONIC names its timer with -DFOUR_AEC_NR_RES_NOW_US=<fn> (and -DAEC_NOW_US=<fn>). A call rejected by its argument checks leaves the record untouched. */
typedef struct FourAecNrResLastTiming {
     uint32_t delay_us;
     uint32_t frontend_us;
     uint32_t linear_us;
     uint32_t lane_res_us;
     uint32_t fuse_us;
} FourAecNrResLastTiming;

/* Copies the last hop's stage timings into `out`. A NULL/destroyed pipeline zeroes
 * `out`; `out` must not be NULL. */
void four_aec_nr_res_get_last_timing(
    const FourAecNrRes* p,
    FourAecNrResLastTiming* out);

#ifdef __cplusplus
}
#endif

#endif /* FOUR_AEC_NR_RES_H */
